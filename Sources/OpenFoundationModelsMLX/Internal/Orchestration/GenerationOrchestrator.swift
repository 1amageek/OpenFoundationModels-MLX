import Foundation
import MLXLMCommon
import MLX
import Tokenizers

/// GenerationOrchestrator coordinates the generation process across multiple layers.
/// It handles request processing, parameter conversion, retry logic, and response formatting.
/// This is the high-level orchestration layer that uses the new pipeline architecture.
actor GenerationOrchestrator {
    
    private let executor: MLXExecutor
    private let pipeline: GenerationPipeline
    private let maxRetries: Int
    
    private enum OrchestratorError: Error {
        case validationFailed
        case bufferLimitExceeded
        case jsonMalformed
        case schemaViolations
        case maxRetriesExceeded
    }
    
    /// Initialize with executor
    /// - Parameters:
    ///   - executor: The MLXExecutor for model execution
    ///   - maxRetries: Maximum retry attempts (default: 2)
    init(
        executor: MLXExecutor,
        maxRetries: Int = 2
    ) {
        self.executor = executor
        self.maxRetries = maxRetries
        
        // Use AdaptiveConstraintEngine for unified pipeline
        self.pipeline = GenerationPipeline(
            executor: executor,
            constraints: AdaptiveConstraintEngine(),
            retryPolicy: RetryPolicy(maxAttempts: maxRetries),
            telemetry: NoOpTelemetry()
        )
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
        let genParams = GenerateParameters(
            maxTokens: sampling.maxTokens ?? 1024,
            temperature: Float(sampling.temperature ?? 0.7),
            topP: Float(sampling.topP ?? 1.0)
        )
        
        do {
            let text = try await pipeline.run(
                prompt: prompt,
                schema: request.schema,
                parameters: genParams
            )
            
            let choice = ChatChoice(content: text, finishReason: "stop")
            return ChatResponse(
                choices: [choice],
                usage: .init(promptTokens: 0, completionTokens: 0),
                meta: ChatResponseMeta()
            )
        } catch {
            Stream().synchronize()
            Logger.warning("[GenerationOrchestrator] Generation failed: \(error)")
            throw error
        }
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
                    let genParams = GenerateParameters(
                        maxTokens: sampling.maxTokens ?? 1024,
                        temperature: Float(sampling.temperature ?? 0.7),
                        topP: Float(sampling.topP ?? 1.0)
                    )
                    
                    let stream = pipeline.stream(
                        prompt: prompt,
                        schema: request.schema,
                        parameters: genParams
                    )
                    
                    var buffer = ""
                    let bufferLimit = 30000
                    
                    for try await chunk in stream {
                        try Task.checkCancellation()
                        
                        buffer += chunk
                        if buffer.count > bufferLimit {
                            throw OrchestratorError.bufferLimitExceeded
                        }
                        
                        let delta = ChatDelta(deltaText: chunk, finishReason: nil)
                        let chatChunk = ChatChunk(deltas: [delta])
                        continuation.yield(chatChunk)
                    }
                    
                    // Final chunk with finish reason
                    let finalDelta = ChatDelta(deltaText: "", finishReason: "stop")
                    let finalChunk = ChatChunk(deltas: [finalDelta])
                    continuation.yield(finalChunk)
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
    
    /// Convert request parameters to sampling parameters
    private func convertParameters(_ request: ChatRequest) -> SamplingParameters {
        // Use the sampling parameters directly from the request
        // or convert from legacy parameters if present
        if let params = request.parameters {
            return SamplingParameters(
                temperature: Double(params.temperature),
                topP: Double(params.topP),
                topK: nil,
                maxTokens: params.maxTokens,
                stop: nil,
                seed: nil
            )
        }
        
        return request.sampling
    }
}

