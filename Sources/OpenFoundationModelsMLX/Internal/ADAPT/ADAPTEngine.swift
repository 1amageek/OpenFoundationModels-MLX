import Foundation
import MLX
import MLXLMCommon
import MLXLLM
import Tokenizers

public actor ADAPTEngine {
    
    private var trieCache: [String: TokenTrie] = [:]
    private let maxCacheSize = 100
    
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
            
            let isValid = JSONValidator.validate(text: result, schema: schema)
            
            if !isValid {
                throw ADAPTError.validationFailed("Generated JSON does not match schema")
            }
            
            return result
        }
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
        trieCache.removeAll()
        Logger.info("[ADAPTEngine] Cache cleared")
    }
    
    public func cacheSize() -> Int {
        return trieCache.count
    }
    
    public func getMetrics() -> ADAPTMetrics {
        return ADAPTMetrics(
            cacheSize: trieCache.count,
            maxCacheSize: maxCacheSize
        )
    }
}


public struct ADAPTMetrics: Sendable {
    public let cacheSize: Int
    public let maxCacheSize: Int
}