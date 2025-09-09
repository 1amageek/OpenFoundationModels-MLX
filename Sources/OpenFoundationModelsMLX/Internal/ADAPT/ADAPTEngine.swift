import Foundation
import MLX
import MLXLMCommon
import MLXLLM
import Tokenizers

/// ADAPTEngine manages the Adaptive Dynamic Assertion Protocol for Transformers.
/// It handles schema-constrained generation, token-level constraints, and JSON validation.
/// This engine operates independently of the execution layer and orchestration logic.
public actor ADAPTEngine {
    
    // MARK: - Properties
    
    private var trieCache: [String: TokenTrie] = [:]
    private let maxCacheSize = 100
    
    // MARK: - Errors
    
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
    
    // MARK: - Schema-Constrained Generation
    
    /// Generate text with schema constraints using the ADAPT system
    /// - Parameters:
    ///   - executor: The MLXExecutor to use for model execution
    ///   - prompt: The input prompt
    ///   - schema: The hierarchical schema node for constraints
    ///   - parameters: Sampling parameters
    /// - Returns: Generated text that conforms to the schema
    func generateWithSchema(
        executor: MLXExecutor,
        prompt: String,
        schema: SchemaNode,
        parameters: SamplingParameters
    ) async throws -> String {
        // Get model container from executor
        guard let container = await executor.getModelContainer() else {
            throw ADAPTError.noTokenizer
        }
        
        // Execute within model context to get tokenizer
        return try await container.perform { (context: ModelContext) async throws -> String in
            let tokenizer = context.tokenizer
            
            // Create constraint processor with nested schema support
            let constraintProcessor = TokenTrieLogitProcessor(
                schema: schema,
                tokenizer: tokenizer
            )
            constraintProcessor.clearError()
            
            // Convert sampling parameters to generation parameters
            let genParams = GenerateParameters(
                maxTokens: parameters.maxTokens ?? 1024,
                temperature: Float(parameters.temperature ?? 0.7),
                topP: Float(parameters.topP ?? 1.0)
            )
            
            // Execute with constraints
            let result = try await executor.execute(
                prompt: prompt,
                parameters: genParams,
                logitProcessor: constraintProcessor
            )
            
            // Validate the result
            print("🔍 [ADAPTEngine] Validating generated JSON...")
            print("📝 [ADAPTEngine] Generated text: \(result)")
            print("📋 [ADAPTEngine] Schema root keys: \(schema.objectKeys)")
            print("📋 [ADAPTEngine] Required fields: \(schema.required)")
            
            // Use hierarchical validator for nested schema
            let isValid = JSONValidator.validate(text: result, schema: schema)
            print("✅ [ADAPTEngine] Validation result: \(isValid)")
            
            if !isValid {
                print("❌ [ADAPTEngine] Validation failed!")
                // For now, skip snap correction (would need SchemaSnapParser to be public)
                // This could be added back if SchemaSnapParser is made public
                throw ADAPTError.validationFailed("Generated JSON does not match schema")
            }
            
            return result
        }
    }
    
    /// Stream text generation with schema constraints
    /// - Parameters:
    ///   - executor: The MLXExecutor to use for model execution
    ///   - prompt: The input prompt
    ///   - schema: The hierarchical schema node for constraints
    ///   - parameters: Sampling parameters
    /// - Returns: Stream of generated text chunks
    func streamWithSchema(
        executor: MLXExecutor,
        prompt: String,
        schema: SchemaNode,
        parameters: SamplingParameters
    ) -> AsyncThrowingStream<String, Error> {
        AsyncThrowingStream { continuation in
            Task {
                do {
                    // Get model container from executor
                    guard let container = await executor.getModelContainer() else {
                        throw ADAPTError.noTokenizer
                    }
                    
                    // Need to create processor within container context
                    try await container.perform { (context: ModelContext) async throws in
                        let tokenizer = context.tokenizer
                        
                        // Create constraint processor with nested schema support
                        let constraintProcessor = TokenTrieLogitProcessor(
                            schema: schema,
                            tokenizer: tokenizer
                        )
                        constraintProcessor.clearError()
                        
                        // Convert parameters
                        let genParams = GenerateParameters(
                            maxTokens: parameters.maxTokens ?? 1024,
                            temperature: Float(parameters.temperature ?? 0.7),
                            topP: Float(parameters.topP ?? 1.0)
                        )
                        
                        // Get base stream from executor
                        let baseStream = await executor.executeStream(
                            prompt: prompt,
                            parameters: genParams,
                            logitProcessor: constraintProcessor
                        )
                        
                        // Buffer for validation
                        var buffer = ""
                        let jsonState = JSONStateMachine()
                        
                        // Process stream with ADAPT constraints
                        for try await chunk in baseStream {
                            buffer += chunk
                            
                            // Update JSON state
                            for char in chunk {
                                jsonState.processCharacter(char)
                            }
                            
                            // Check for JSON completion
                            if jsonState.isComplete() {
                                // Validate complete JSON with hierarchical schema
                                if JSONValidator.validate(text: buffer, schema: schema) {
                                    continuation.yield(chunk)
                                    continuation.finish()
                                    return
                                } else {
                                    throw ADAPTError.validationFailed("Complete JSON does not match schema")
                                }
                            }
                            
                            // Check for errors
                            if jsonState.isError() {
                                throw ADAPTError.abortedGeneration("JSON parsing error detected")
                            }
                            
                            // Yield chunk if valid so far
                            continuation.yield(chunk)
                        }
                        
                        // Final validation with hierarchical schema
                        if !JSONValidator.validate(text: buffer, schema: schema) {
                            throw ADAPTError.validationFailed("Final JSON does not match schema")
                        }
                        
                        continuation.finish()
                    }
                } catch {
                    continuation.finish(throwing: error)
                }
            }
        }
    }
    
    // MARK: - Constraint Management
    
    /// Create a LogitProcessor for the given schema
    /// - Parameters:
    ///   - schema: The hierarchical schema node
    ///   - tokenizer: The tokenizer to use
    /// - Returns: A configured LogitProcessor
    public func createLogitProcessor(
        schema: SchemaNode,
        tokenizer: Tokenizer
    ) -> LogitProcessor {
        return TokenTrieLogitProcessor(
            schema: schema,
            tokenizer: tokenizer
        )
    }
    
    // MARK: - Private Methods
    
    private func getOrCreateTrie(schema: SchemaMeta, tokenizer: Tokenizer) -> TokenTrie {
        // Create a cache key from schema keys
        let cacheKey = schema.keys.sorted().joined(separator: "|")
        
        if let cached = trieCache[cacheKey] {
            return cached
        }
        
        // Create new trie (wrap tokenizer in adapter)
        let adapter = MLXLLMTokenizer(tokenizer: tokenizer)
        let trie = TokenTrieBuilder.buildCached(schema: schema, tokenizer: adapter)
        
        // Cache management
        if trieCache.count >= maxCacheSize {
            // Remove oldest entry (simple FIFO for now)
            if let firstKey = trieCache.keys.first {
                trieCache.removeValue(forKey: firstKey)
            }
        }
        
        trieCache[cacheKey] = trie
        return trie
    }
    
    // MARK: - Cache Management
    
    /// Clear the TokenTrie cache
    public func clearCache() {
        trieCache.removeAll()
        Logger.info("[ADAPTEngine] Cache cleared")
    }
    
    /// Get current cache size
    public func cacheSize() -> Int {
        return trieCache.count
    }
    
    // MARK: - Metrics
    
    /// Get ADAPT system metrics
    public func getMetrics() -> ADAPTMetrics {
        return ADAPTMetrics(
            cacheSize: trieCache.count,
            maxCacheSize: maxCacheSize
        )
    }
}

// MARK: - Supporting Types

public struct ADAPTMetrics: Sendable {
    public let cacheSize: Int
    public let maxCacheSize: Int
}