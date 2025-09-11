import Foundation
import MLXLMCommon
import MLX

/// GenerationOrchestrator coordinates the generation process across multiple layers.
/// It handles request processing, parameter conversion, retry logic, and response formatting.
/// This is the high-level orchestration layer that ties together execution and ADAPT.
actor GenerationOrchestrator {
    
    private let executor: MLXExecutor
    private let adaptEngine: ADAPTEngine
    private let maxRetries: Int
    
    private enum OrchestratorError: Error {
        case validationFailed
        case bufferLimitExceeded
        case jsonMalformed
        case schemaViolations
        case maxRetriesExceeded
    }
    
    /// Initialize with executor and ADAPT engine
    /// - Parameters:
    ///   - executor: The MLXExecutor for model execution
    ///   - adaptEngine: The ADAPTEngine for constraint management
    ///   - maxRetries: Maximum retry attempts (default: 2)
    init(
        executor: MLXExecutor,
        adaptEngine: ADAPTEngine,
        maxRetries: Int = 2
    ) {
        self.executor = executor
        self.adaptEngine = adaptEngine
        self.maxRetries = maxRetries
    }
    
    /// Generate a response for the given request
    /// - Parameter request: The chat request
    /// - Returns: The chat response
    func generate(_ request: ChatRequest) async throws -> ChatResponse {
        let prompt = request.prompt
        Logger.info("[GenerationOrchestrator] Processing prompt with schema: \(request.schema != nil)")
        if request.schema != nil {
            Logger.info("[GenerationOrchestrator] Schema keys: \(request.schema!.objectKeys)")
        }
        
        let sampling = convertParameters(request)
        var lastError: Error?
        var attempts = 0
        
        while attempts < maxRetries {
            attempts += 1
            
            do {
                let text: String
                
                if let schemaNode = request.schema {
                    text = try await adaptEngine.generateWithSchema(
                        executor: executor,
                        prompt: prompt,
                        schema: schemaNode,
                        parameters: sampling
                    )
                } else {
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
                
                if case .jsonSchema = request.responseFormat {
                    let isValid: Bool
                    if let schemaNode = request.schema {
                        isValid = JSONValidator.validate(text: text, schema: schemaNode)
                    } else {
                        isValid = true
                    }
                    
                    if !isValid {
                        lastError = OrchestratorError.validationFailed
                        continue
                    }
                }
                
                let choice = ChatChoice(content: text, finishReason: "stop")
                return ChatResponse(
                    choices: [choice],
                    usage: .init(promptTokens: 0, completionTokens: 0),
                    meta: .init()
                )
                
            } catch {
                Stream().synchronize()
                
                lastError = error
                Logger.warning("[GenerationOrchestrator] Attempt \(attempts) failed: \(error)")
                
                if case MLXExecutor.ExecutorError.noModelSet = error {
                    throw error
                }
            }
        }
        
        throw lastError ?? OrchestratorError.maxRetriesExceeded
    }
    
    /// Stream generation for the given request
    /// - Parameter request: The chat request
    /// - Returns: Stream of chat chunks
    func stream(_ request: ChatRequest) -> AsyncThrowingStream<ChatChunk, Error> {
        return AsyncThrowingStream { continuation in
            let task = Task {
                do {
                    try Task.checkCancellation()
                    
                    let prompt = request.prompt
                    let sampling = convertParameters(request)
                    
                    if let schemaNode = request.schema {
                        try Task.checkCancellation()
                        
                        let stream = await adaptEngine.streamWithSchema(
                            executor: executor,
                            prompt: prompt,
                            schema: schemaNode,
                            parameters: sampling
                        )
                        
                        for try await chunk in stream {
                            try Task.checkCancellation()
                            let delta = ChatDelta(deltaText: chunk, finishReason: nil)
                            continuation.yield(ChatChunk(deltas: [delta]))
                        }
                    } else {
                        try Task.checkCancellation()
                        
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
                            try Task.checkCancellation()
                            continuation.yield(ChatChunk(deltas: [.init(deltaText: chunk, finishReason: nil)]))
                        }
                    }
                    
                    try Task.checkCancellation()
                    continuation.yield(ChatChunk(deltas: [.init(deltaText: nil, finishReason: "stop")]))
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
    
    private func convertParameters(_ request: ChatRequest) -> SamplingParameters {
        if let p = request.parameters {
            return SamplingParameters(
                temperature: Double(p.temperature),
                topP: Double(p.topP),
                topK: nil,
                maxTokens: p.maxTokens,
                stop: nil,
                seed: nil
            )
        } else {
            return request.sampling
        }
    }
}

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
            maxRetries: maxRetries
        )
    }
}

public struct OrchestratorMetrics: Sendable {
    public let modelID: String?
    public let adaptCacheSize: Int
    public let maxRetries: Int
}