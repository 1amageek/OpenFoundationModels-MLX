import Foundation
import MLXLMCommon
import MLXLLM

public actor MLXBackend {
    
    
    let executor: MLXExecutor
    private let orchestrator: GenerationOrchestrator
    
    
    public enum MLXBackendError: LocalizedError {
        case noModelSet
        case generationFailed(String)
        
        public var errorDescription: String? {
            switch self {
            case .noModelSet:
                return "No model has been set. Call setModel() with a loaded ModelContainer first."
            case .generationFailed(let reason):
                return "Generation failed: \(reason)"
            }
        }
    }
    
    
    public init(additionalProcessors: [LogitProcessor] = []) {
        self.executor = MLXExecutor()
        self.orchestrator = GenerationOrchestrator(
            executor: executor,
            additionalProcessors: additionalProcessors
        )
    }
    
    
    public func setModel(_ container: ModelContainer, modelID: String? = nil) async {
        await executor.setModel(container, modelID: modelID)
    }
    
    public func clearModel() async {
        await executor.clearModel()
    }
    
    public func currentModel() async -> String? {
        return await executor.currentModel()
    }
    
    public func hasModel() async -> Bool {
        return await executor.hasModel()
    }
    
    
    func orchestratedGenerate(
        prompt: String,
        sampling: SamplingParameters,
        schema: SchemaNode? = nil,
        schemaJSON: String? = nil,
        modelCard: (any ModelCard)? = nil
    ) async throws -> String {
        guard await hasModel() else {
            throw MLXBackendError.noModelSet
        }
        
        // Convert sampling parameters to GenerateParameters
        let parameters = GenerateParameters(
            maxTokens: sampling.maxTokens ?? 1024,
            temperature: Float(sampling.temperature ?? 0.7),
            topP: Float(sampling.topP ?? 1.0)
        )
        
        // All requests go through the simplified orchestrator
        return try await orchestrator.generate(
            prompt: prompt,
            schema: schema,
            parameters: parameters,
            modelCard: modelCard
        )
    }
    
    
    func orchestratedStream(
        prompt: String,
        sampling: SamplingParameters,
        schema: SchemaNode? = nil,
        schemaJSON: String? = nil,
        modelCard: (any ModelCard)? = nil
    ) -> AsyncThrowingStream<String, Error> {
        AsyncThrowingStream { continuation in
            let task = Task {
                do {
                    try Task.checkCancellation()
                    
                    guard await hasModel() else {
                        continuation.finish(throwing: MLXBackendError.noModelSet)
                        return
                    }
                    
                    try Task.checkCancellation()
                    
                    // Convert sampling parameters to GenerateParameters
                    let parameters = GenerateParameters(
                        maxTokens: sampling.maxTokens ?? 1024,
                        temperature: Float(sampling.temperature ?? 0.7),
                        topP: Float(sampling.topP ?? 1.0)
                    )
                    
                    try Task.checkCancellation()
                    
                    // Use simplified orchestrator stream
                    let stream = await orchestrator.stream(
                        prompt: prompt,
                        schema: schema,
                        parameters: parameters,
                        modelCard: modelCard
                    )
                    
                    for try await chunk in stream {
                        try Task.checkCancellation()
                        continuation.yield(chunk)
                    }
                    continuation.finish()
                } catch is CancellationError {
                    continuation.finish(throwing: CancellationError())
                } catch {
                    continuation.finish(throwing: error)
                }
            }
            
            continuation.onTermination = { @Sendable _ in
                task.cancel()
            }
        }
    }
}