import Foundation
import MLX
import MLXLMCommon
import MLXLLM
@preconcurrency import Tokenizers

public actor ADAPTEngine {
    
    // DPDAKeyTrieLogitProcessor is now the only processor
    
    private var schemaIndexCache: LRUCache<String, SchemaTrieIndex>
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
    
    private func prepareSchemaIndex(
        executor: MLXExecutor,
        schema: SchemaNode
    ) async throws -> (SchemaTrieIndex, MLXLLMTokenizer, String) {
        let (adapter, fingerprint) = try await executor.withTokenizer { tokenizer in
            let adapter = MLXLLMTokenizer(tokenizer: tokenizer)
            return (adapter, adapter.fingerprint())
        }
        
        let cacheKey = "\(fingerprint)|\(schema.cacheKey())"
        
        if let cached = schemaIndexCache.get(cacheKey) {
            Logger.debug("[ADAPT] Using cached SchemaTrieIndex for key: \(cacheKey)")
            return (cached, adapter, cacheKey)
        } else {
            Logger.debug("[ADAPT] Building new SchemaTrieIndex for key: \(cacheKey)")
            let schemaIndex = SchemaTrieIndex(root: schema, tokenizer: adapter)
            schemaIndexCache.set(cacheKey, value: schemaIndex)
            return (schemaIndex, adapter, cacheKey)
        }
    }
    
    func generateWithSchema(
        executor: MLXExecutor,
        prompt: String,
        schema: SchemaNode,
        parameters: SamplingParameters
    ) async throws -> String {
        Logger.debug("[ADAPT] Schema received with \(schema.objectKeys.count) keys: \(schema.objectKeys)")
        
        let (schemaIndex, _, _) = try await prepareSchemaIndex(executor: executor, schema: schema)
        
        let constraintProcessor = try await executor.withTokenizer { tokenizer in
            Logger.debug("[ADAPT] Using DPDAKeyTrieLogitProcessor")
            let processor = DPDAKeyTrieLogitProcessor(
                schema: schema,
                tokenizer: tokenizer,
                cachedIndex: schemaIndex
            )
            processor.clearError()
            return processor
        }
        
        let genParams = GenerateParameters(
            maxTokens: parameters.maxTokens ?? 1024,
            temperature: Float(parameters.temperature ?? 0.7),
            topP: Float(parameters.topP ?? 1.0)
        )
        
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
    
    func streamWithSchema(
        executor: MLXExecutor,
        prompt: String,
        schema: SchemaNode,
        parameters: SamplingParameters
    ) -> AsyncThrowingStream<String, Error> {
        AsyncThrowingStream { continuation in
            let task = Task {
                do {
                    try Task.checkCancellation()
                    
                    let (schemaIndex, _, _) = try await self.prepareSchemaIndex(executor: executor, schema: schema)
                    
                    try Task.checkCancellation()
                    
                    let constraintProcessor = try await executor.withTokenizer { tokenizer in
                        Logger.debug("[ADAPT] [Stream] Using DPDAKeyTrieLogitProcessor")
                        let processor = DPDAKeyTrieLogitProcessor(
                            schema: schema,
                            tokenizer: tokenizer,
                            cachedIndex: schemaIndex
                        )
                        processor.clearError()
                        return processor
                    }
                    
                    try Task.checkCancellation()
                    
                    let genParams = GenerateParameters(
                        maxTokens: parameters.maxTokens ?? 1024,
                        temperature: Float(parameters.temperature ?? 0.7),
                        topP: Float(parameters.topP ?? 1.0)
                    )
                    
                    let baseStream = await executor.executeStream(
                        prompt: prompt,
                        parameters: genParams,
                        logitProcessor: constraintProcessor
                    )
                    
                    defer {
                        constraintProcessor.clearError()
                    }
                    
                    for try await chunk in baseStream {
                        try Task.checkCancellation()
                        continuation.yield(chunk)
                    }
                    
                    continuation.finish()
                } catch is CancellationError {
                    Stream().synchronize()
                    continuation.finish(throwing: CancellationError())
                } catch {
                    Stream().synchronize()
                    continuation.finish(throwing: error)
                }
            }
            
            continuation.onTermination = { @Sendable _ in
                task.cancel()
            }
        }
    }
    
    public func createLogitProcessor(
        schema: SchemaNode,
        tokenizer: Tokenizer
    ) -> LogitProcessor {
        return DPDAKeyTrieLogitProcessor(
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