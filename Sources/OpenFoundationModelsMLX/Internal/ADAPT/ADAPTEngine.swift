import Foundation
import MLX
import MLXLMCommon
import MLXLLM
@preconcurrency import Tokenizers

public actor ADAPTEngine {
    
    private var schemaIndexCache: LRUCache<String, SchemaTrieIndex>  // SchemaTrieIndex用LRUキャッシュ
    private let maxCacheSize: Int
    
    public init(maxCacheSize: Int = 100) {
        self.maxCacheSize = maxCacheSize
        self.schemaIndexCache = LRUCache<String, SchemaTrieIndex>(maxSize: maxCacheSize)
    }
    
    public enum ADAPTError: LocalizedError {
        case noTokenizer
        case constraintGenerationFailed(String)
        case validationFailed(String)
        case abortedGeneration(String)
        
        public var errorDescription: String? {
            switch self {
            case .noTokenizer:
                return "No tokenizer available for constraint generation"
            case .constraintGenerationFailed(let reason):
                return "Failed to generate constraints: \(reason)"
            case .validationFailed(let reason):
                return "Validation failed: \(reason)"
            case .abortedGeneration(let reason):
                return "Generation aborted: \(reason)"
            }
        }
    }
    
    /// Generate text with schema constraints using the ADAPT system
    func generateWithSchema(
        executor: MLXExecutor,
        prompt: String,
        schema: SchemaNode,
        parameters: SamplingParameters
    ) async throws -> String {
        Logger.debug("[ADAPT] Schema received with \(schema.objectKeys.count) keys: \(schema.objectKeys)")
        
        // キャッシュキーを事前に生成（tokenizerに依存しない部分）
        let schemaKey = schema.objectKeys.sorted().joined()
        
        // Get tokenizer safely without nesting perform
        let (adapter, fingerprint) = try await executor.withTokenizer { tokenizer in
            let adapter = MLXLLMTokenizer(tokenizer: tokenizer)
            return (adapter, adapter.fingerprint())
        }
        
        // キャッシュアクセスはクロージャの外で行う
        let cacheKey = "\(fingerprint)|\(schemaKey)"
        let schemaIndex: SchemaTrieIndex
        if let cached = schemaIndexCache.get(cacheKey) {
            Logger.debug("[ADAPT] Using cached SchemaTrieIndex for key: \(cacheKey)")
            schemaIndex = cached
        } else {
            Logger.debug("[ADAPT] Building new SchemaTrieIndex for key: \(cacheKey)")
            schemaIndex = SchemaTrieIndex(root: schema, tokenizer: adapter)
            schemaIndexCache.set(cacheKey, value: schemaIndex)
        }
        
        // ProcessorはSchemaIndexを使って作成
        let constraintProcessor = try await executor.withTokenizer { tokenizer in
            let processor = TokenTrieLogitProcessor(
                schema: schema,
                tokenizer: tokenizer,
                cachedIndex: schemaIndex  // キャッシュされたIndexを渡す
            )
            processor.clearError()
            Logger.debug("[ADAPT] TokenTrieLogitProcessor created successfully")
            return processor
        }
        
        // Convert sampling parameters to generation parameters
        let genParams = GenerateParameters(
            maxTokens: parameters.maxTokens ?? 1024,
            temperature: Float(parameters.temperature ?? 0.7),
            topP: Float(parameters.topP ?? 1.0)
        )
        
        // Execute with constraints - no nested perform
        let result = try await executor.execute(
            prompt: prompt,
            parameters: genParams,
            logitProcessor: constraintProcessor
        )
        
        let isValid = JSONValidator.validate(text: result, schema: schema)
        Logger.debug("[ADAPT] Validation result: \(isValid ? "success" : "failure")")
        
        if !isValid {
            Logger.debug("[ADAPT] Generated text: \(result)")
            throw ADAPTError.validationFailed("Generated JSON does not match schema")
        }
        
        return result
    }
    
    /// Stream text generation with schema constraints
    func streamWithSchema(
        executor: MLXExecutor,
        prompt: String,
        schema: SchemaNode,
        parameters: SamplingParameters
    ) -> AsyncThrowingStream<String, Error> {
        AsyncThrowingStream { continuation in
            Task {
                do {
                    // キャッシュキーを事前に生成（tokenizerに依存しない部分）
                    let schemaKey = schema.objectKeys.sorted().joined()
                    
                    // Get tokenizer safely without nesting perform
                    let (adapter, fingerprint) = try await executor.withTokenizer { tokenizer in
                        let adapter = MLXLLMTokenizer(tokenizer: tokenizer)
                        return (adapter, adapter.fingerprint())
                    }
                    
                    // キャッシュアクセスはクロージャの外で行う
                    let cacheKey = "\(fingerprint)|\(schemaKey)"
                    let schemaIndex: SchemaTrieIndex
                    if let cached = await self.schemaIndexCache.get(cacheKey) {
                        Logger.debug("[ADAPT] [Stream] Using cached SchemaTrieIndex for key: \(cacheKey)")
                        schemaIndex = cached
                    } else {
                        Logger.debug("[ADAPT] [Stream] Building new SchemaTrieIndex for key: \(cacheKey)")
                        schemaIndex = SchemaTrieIndex(root: schema, tokenizer: adapter)
                        await self.schemaIndexCache.set(cacheKey, value: schemaIndex)
                    }
                    
                    // ProcessorはSchemaIndexを使って作成
                    let constraintProcessor = try await executor.withTokenizer { tokenizer in
                        let processor = TokenTrieLogitProcessor(
                            schema: schema,
                            tokenizer: tokenizer,
                            cachedIndex: schemaIndex  // キャッシュされたIndexを渡す
                        )
                        processor.clearError()
                        Logger.debug("[ADAPT] [Stream] TokenTrieLogitProcessor created successfully")
                        return processor
                    }
                    
                    // Convert parameters
                    let genParams = GenerateParameters(
                        maxTokens: parameters.maxTokens ?? 1024,
                        temperature: Float(parameters.temperature ?? 0.7),
                        topP: Float(parameters.topP ?? 1.0)
                    )
                    
                    // Get base stream from executor - no nested perform
                    let baseStream = await executor.executeStream(
                        prompt: prompt,
                        parameters: genParams,
                        logitProcessor: constraintProcessor
                    )
                    
                    // Process stream with ADAPT constraints
                    // Note: TokenTrieLogitProcessor already handles JSON state tracking and validation
                    for try await chunk in baseStream {
                        // Simply pass through - constraint processor ensures valid JSON
                        continuation.yield(chunk)
                    }
                    
                    continuation.finish()
                } catch {
                    continuation.finish(throwing: error)
                }
            }
        }
    }
    
    /// Create a LogitProcessor for the given schema
    public func createLogitProcessor(
        schema: SchemaNode,
        tokenizer: Tokenizer
    ) -> LogitProcessor {
        return TokenTrieLogitProcessor(
            schema: schema,
            tokenizer: tokenizer
        )
    }
    
    
    public func clearCache() {
        schemaIndexCache.clear()
        Logger.info("[ADAPTEngine] Cache cleared")
    }
    
    public func cacheSize() -> Int {
        return schemaIndexCache.count
    }
    
    public func getMetrics() -> ADAPTMetrics {
        return ADAPTMetrics(
            cacheSize: schemaIndexCache.count,
            maxCacheSize: maxCacheSize
        )
    }
}


public struct ADAPTMetrics: Sendable {
    public let cacheSize: Int
    public let maxCacheSize: Int
}