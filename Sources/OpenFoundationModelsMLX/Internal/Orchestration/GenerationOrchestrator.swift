import Foundation
import MLXLMCommon

/// GenerationOrchestrator coordinates the generation process across multiple layers.
/// It handles request processing, parameter conversion, retry logic, and response formatting.
/// This is the high-level orchestration layer that ties together execution and ADAPT.
actor GenerationOrchestrator {
    
    // MARK: - Properties
    
    private let executor: MLXExecutor
    private let adaptEngine: ADAPTEngine
    private let maxRetries: Int
    private let maxBufferSizeKB: Int
    
    // MARK: - Errors
    
    private enum OrchestratorError: Error {
        case validationFailed
        case bufferLimitExceeded
        case jsonMalformed
        case schemaViolations
        case maxRetriesExceeded
    }
    
    // MARK: - Initialization
    
    /// Initialize with executor and ADAPT engine
    /// - Parameters:
    ///   - executor: The MLXExecutor for model execution
    ///   - adaptEngine: The ADAPTEngine for constraint management
    ///   - maxRetries: Maximum retry attempts (default: 2)
    ///   - maxBufferSizeKB: Maximum buffer size for streaming (default: 2048KB)
    init(
        executor: MLXExecutor,
        adaptEngine: ADAPTEngine,
        maxRetries: Int = 2,
        maxBufferSizeKB: Int = 2048
    ) {
        self.executor = executor
        self.adaptEngine = adaptEngine
        self.maxRetries = maxRetries
        self.maxBufferSizeKB = maxBufferSizeKB
    }
    
    // MARK: - Generation Methods
    
    /// Generate a response for the given request
    /// - Parameter request: The chat request
    /// - Returns: The chat response
    func generate(_ request: ChatRequest) async throws -> ChatResponse {
        let prompt = request.prompt
        
        // Convert parameters if needed
        let sampling = convertParameters(request)
        
        // Attempt generation with retry logic
        var lastError: Error?
        var attempts = 0
        
        while attempts < maxRetries {
            attempts += 1
            
            do {
                let text: String
                
                // Choose generation method based on schema presence
                if let schema = request.schema {
                    text = try await adaptEngine.generateWithSchema(
                        executor: executor,
                        prompt: prompt,
                        schema: schema,
                        parameters: sampling
                    )
                } else {
                    // Direct execution without constraints
                    let genParams = GenerateParameters(
                        maxTokens: sampling.maxTokens ?? 1024,
                        temperature: Float(sampling.temperature ?? 0.7),
                        topP: Float(sampling.topP ?? 1.0)
                    )
                    text = try await executor.execute(
                        prompt: prompt,
                        parameters: genParams
                    )
                }
                
                // Post-generation validation if needed
                if case .jsonSchema = request.responseFormat,
                   let schema = request.schema {
                    let validator = JSONValidator(allowExtraKeys: false, enableSnap: true)
                    if !validator.validate(text: text, schema: schema) {
                        lastError = OrchestratorError.validationFailed
                        continue // Retry
                    }
                }
                
                // Success - create response
                let choice = ChatChoice(content: text, finishReason: "stop")
                return ChatResponse(
                    choices: [choice],
                    usage: .init(promptTokens: 0, completionTokens: 0),
                    meta: .init()
                )
                
            } catch {
                lastError = error
                Logger.warning("[GenerationOrchestrator] Attempt \(attempts) failed: \(error)")
                
                // Don't retry for certain errors
                if case MLXExecutor.ExecutorError.noModelSet = error {
                    throw error
                }
            }
        }
        
        // All retries exhausted
        throw lastError ?? OrchestratorError.maxRetriesExceeded
    }
    
    /// Stream generation for the given request
    /// - Parameter request: The chat request
    /// - Returns: Stream of chat chunks
    func stream(_ request: ChatRequest) -> AsyncThrowingStream<ChatChunk, Error> {
        let hasSchema = request.schema != nil
        let schemaKeys = request.schema?.keys ?? []
        
        return AsyncThrowingStream { continuation in
            let task = Task {
                do {
                    let prompt = request.prompt
                    let sampling = convertParameters(request)
                    
                    if let schema = request.schema {
                        // Stream with ADAPT constraints
                        let stream = await adaptEngine.streamWithSchema(
                            executor: executor,
                            prompt: prompt,
                            schema: schema,
                            parameters: sampling
                        )
                        
                        // Buffer for validation if needed
                        if hasSchema && !schemaKeys.isEmpty {
                            var buffer = ""
                            let bufferLimit = maxBufferSizeKB * 1024
                            var tracker = JSONKeyTracker(schemaKeys: schemaKeys)
                            let jsonState = JSONStateMachine()
                            
                            for try await chunk in stream {
                                buffer += chunk
                                
                                // Check buffer limit
                                if buffer.utf8.count > bufferLimit {
                                    Logger.warning("[GenerationOrchestrator] Buffer exceeded limit")
                                    throw OrchestratorError.bufferLimitExceeded
                                }
                                
                                // Update JSON state
                                for char in chunk {
                                    jsonState.processCharacter(char)
                                }
                                
                                // Check for completion
                                if jsonState.isComplete() {
                                    continuation.yield(ChatChunk(deltas: [.init(deltaText: chunk, finishReason: "stop")]))
                                    continuation.finish()
                                    return
                                }
                                
                                // Check for errors
                                if jsonState.isError() {
                                    throw OrchestratorError.jsonMalformed
                                }
                                
                                // Track violations
                                tracker.consume(chunk)
                                if tracker.violationCount >= 3 {
                                    throw OrchestratorError.schemaViolations
                                }
                                
                                // Yield chunk
                                continuation.yield(ChatChunk(deltas: [.init(deltaText: chunk, finishReason: nil)]))
                            }
                            
                            // Final validation
                            if case .jsonSchema = request.responseFormat,
                               let meta = request.schema {
                                let validator = JSONValidator(allowExtraKeys: false, enableSnap: true)
                                if !validator.validate(text: buffer, schema: meta) {
                                    throw OrchestratorError.validationFailed
                                }
                            }
                        } else {
                            // Pass through without buffering
                            for try await chunk in stream {
                                continuation.yield(ChatChunk(deltas: [.init(deltaText: chunk, finishReason: nil)]))
                            }
                        }
                    } else {
                        // Direct streaming without constraints
                        let genParams = GenerateParameters(
                            maxTokens: sampling.maxTokens ?? 1024,
                            temperature: Float(sampling.temperature ?? 0.7),
                            topP: Float(sampling.topP ?? 1.0)
                        )
                        
                        let stream = await executor.executeStream(
                            prompt: prompt,
                            parameters: genParams
                        )
                        
                        for try await chunk in stream {
                            continuation.yield(ChatChunk(deltas: [.init(deltaText: chunk, finishReason: nil)]))
                        }
                    }
                    
                    // Send final chunk
                    continuation.yield(ChatChunk(deltas: [.init(deltaText: nil, finishReason: "stop")]))
                    continuation.finish()
                    
                } catch {
                    continuation.finish(throwing: error)
                }
            }
            
            continuation.onTermination = { _ in
                task.cancel()
            }
        }
    }
    
    // MARK: - Private Methods
    
    private func convertParameters(_ request: ChatRequest) -> SamplingParameters {
        if let p = request.parameters {
            // Convert GenerateParameters to SamplingParameters
            return SamplingParameters(
                temperature: Double(p.temperature),
                topP: Double(p.topP),
                topK: nil,  // GenerateParameters doesn't have topK
                maxTokens: p.maxTokens,
                stop: nil,
                seed: nil   // GenerateParameters doesn't have seed
            )
        } else {
            return request.sampling
        }
    }
}

// MARK: - Convenience Methods

extension GenerationOrchestrator {
    
    /// Check if the orchestrator has a model ready
    func hasModel() async -> Bool {
        return await executor.hasModel()
    }
    
    /// Get the current model ID
    func currentModel() async -> String? {
        return await executor.currentModel()
    }
    
    /// Clear ADAPT cache
    func clearCache() async {
        await adaptEngine.clearCache()
    }
    
    /// Get metrics from both layers
    func getMetrics() async -> OrchestratorMetrics {
        let adaptMetrics = await adaptEngine.getMetrics()
        let modelID = await executor.currentModel()
        
        return OrchestratorMetrics(
            modelID: modelID,
            adaptCacheSize: adaptMetrics.cacheSize,
            maxRetries: maxRetries,
            maxBufferSizeKB: maxBufferSizeKB
        )
    }
}

// MARK: - Supporting Types

public struct OrchestratorMetrics: Sendable {
    public let modelID: String?
    public let adaptCacheSize: Int
    public let maxRetries: Int
    public let maxBufferSizeKB: Int
}