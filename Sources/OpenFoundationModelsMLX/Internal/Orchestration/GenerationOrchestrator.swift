import Foundation
import MLXLMCommon
import MLX
import Tokenizers

/// GenerationOrchestrator coordinates the generation process across multiple layers.
/// Simplified to directly work with primitive parameters instead of request/response objects.
actor GenerationOrchestrator {
    
    private let executor: MLXExecutor
    private let pipeline: GenerationPipeline
    private let maxRetries: Int
    
    private enum OrchestratorError: Error {
        case bufferLimitExceeded
    }
    
    /// Initialize with executor
    /// - Parameters:
    ///   - executor: The MLXExecutor for model execution
    ///   - maxRetries: Maximum retry attempts (default: 2)
    init(
        executor: MLXExecutor,
        maxRetries: Int = 2,
        additionalProcessors: [LogitProcessor] = []
    ) {
        self.executor = executor
        self.maxRetries = maxRetries
        
        // Use AdaptiveConstraintEngine for unified pipeline
        self.pipeline = GenerationPipeline(
            executor: executor,
            constraints: AdaptiveConstraintEngine(),
            additionalProcessors: additionalProcessors
        )
    }
    
    /// Generate text with optional schema constraints
    /// - Parameters:
    ///   - prompt: The prompt text
    ///   - schema: Optional schema node for constrained generation
    ///   - parameters: Generation parameters
    ///   - modelCard: Optional model card for processor control
    /// - Returns: Generated text
    func generate(
        prompt: String,
        schema: SchemaNode? = nil,
        parameters: GenerateParameters,
        modelCard: (any ModelCard)? = nil
    ) async throws -> String {
        Logger.info("[GenerationOrchestrator] Processing prompt with schema: \(schema != nil)")
        if let schema = schema {
            Logger.info("[GenerationOrchestrator] Schema keys: \(schema.objectKeys)")
        }
        
        do {
            let text = try await pipeline.run(
                prompt: prompt,
                schema: schema,
                parameters: parameters,
                modelCard: modelCard
            )
            
            return text
        } catch {
            Stream().synchronize()
            Logger.warning("[GenerationOrchestrator] Generation failed: \(error)")
            throw error
        }
    }
    
    /// Stream text generation with optional schema constraints
    /// - Parameters:
    ///   - prompt: The prompt text
    ///   - schema: Optional schema node for constrained generation
    ///   - parameters: Generation parameters
    ///   - modelCard: Optional model card for processor control
    /// - Returns: Stream of generated text chunks
    func stream(
        prompt: String,
        schema: SchemaNode? = nil,
        parameters: GenerateParameters,
        modelCard: (any ModelCard)? = nil
    ) -> AsyncThrowingStream<String, Error> {
        return AsyncThrowingStream { continuation in
            let task = Task {
                do {
                    try Task.checkCancellation()
                    
                    let stream = pipeline.stream(
                        prompt: prompt,
                        schema: schema,
                        parameters: parameters,
                        modelCard: modelCard
                    )
                    
                    var buffer = ""
                    let bufferLimit = 30000
                    
                    for try await chunk in stream {
                        try Task.checkCancellation()
                        
                        buffer += chunk
                        if buffer.count > bufferLimit {
                            throw OrchestratorError.bufferLimitExceeded
                        }
                        
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
}

