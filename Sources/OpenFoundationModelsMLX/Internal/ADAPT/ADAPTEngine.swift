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
        Logger.debug("[ADAPT] Schema received with \(schema.objectKeys.count) keys: \(schema.objectKeys)")
        
        // Get tokenizer safely without nesting perform
        let constraintProcessor = try await executor.withTokenizer { tokenizer in
            Logger.debug("[ADAPT] Building TokenTrie for schema with kind: \(schema.kind)")
            let processor = TokenTrieLogitProcessor(
                schema: schema,
                tokenizer: tokenizer
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
                    // Get tokenizer safely without nesting perform
                    let constraintProcessor = try await executor.withTokenizer { tokenizer in
                        Logger.debug("[ADAPT] [Stream] Building TokenTrie for schema with kind: \(schema.kind)")
                        let processor = TokenTrieLogitProcessor(
                            schema: schema,
                            tokenizer: tokenizer
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
                            Logger.debug("[ADAPT] [Stream] JSON completed, validating...")
                            // Validate complete JSON with hierarchical schema
                            if JSONValidator.validate(text: buffer, schema: schema) {
                                Logger.debug("[ADAPT] [Stream] Validation success")
                                continuation.yield(chunk)
                                continuation.finish()
                                return
                            } else {
                                Logger.debug("[ADAPT] [Stream] Validation failure for: \(buffer)")
                                throw ADAPTError.validationFailed("Complete JSON does not match schema")
                            }
                        }
                        
                        // Check for errors before yielding
                        if jsonState.isError() {
                            throw ADAPTError.abortedGeneration("JSON parsing error detected")
                        }
                        
                        // Yield chunk only if no error
                        continuation.yield(chunk)
                    }
                    
                    // Final validation with hierarchical schema
                    let finalValid = JSONValidator.validate(text: buffer, schema: schema)
                    Logger.debug("[ADAPT] [Stream] Final validation: \(finalValid ? "success" : "failure")")
                    if !finalValid {
                        Logger.debug("[ADAPT] [Stream] Final buffer: \(buffer)")
                        throw ADAPTError.validationFailed("Final JSON does not match schema")
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