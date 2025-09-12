import Foundation
import MLXLMCommon

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
    
    
    public init() {
        self.executor = MLXExecutor()
        self.orchestrator = GenerationOrchestrator(
            executor: executor
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
        schemaJSON: String? = nil
    ) async throws -> String {
        guard await hasModel() else {
            throw MLXBackendError.noModelSet
        }
        
        // Determine response format
        let responseFormat: ResponseFormatSpec
        if let schemaJSON = schemaJSON, !schemaJSON.isEmpty {
            responseFormat = .jsonSchema(schemaJSON: schemaJSON)
        } else {
            responseFormat = .text
        }
        
        // For text mode, bypass the pipeline and execute directly
        if case .text = responseFormat {
            let params = GenerateParameters(
                maxTokens: sampling.maxTokens ?? 1024,
                temperature: Float(sampling.temperature ?? 0.7),
                topP: Float(sampling.topP ?? 1.0)
            )
            return try await executor.execute(
                prompt: prompt,
                parameters: params,
                logitProcessor: nil
            )
        }
        
        // For JSON schema mode, use the pipeline
        let modelID = await currentModel() ?? "unknown"
        let req = ChatRequest(
            modelID: modelID,
            prompt: prompt,
            responseFormat: responseFormat,
            sampling: sampling,
            schema: schema,
            parameters: nil
        )
        
        let response = try await orchestrator.generate(req)
        return response.choices.first?.content ?? ""
    }
    
    
    func orchestratedStream(
        prompt: String,
        sampling: SamplingParameters,
        schema: SchemaNode? = nil,
        schemaJSON: String? = nil
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
                    
                    // Determine response format
                    let responseFormat: ResponseFormatSpec
                    if let schemaJSON = schemaJSON, !schemaJSON.isEmpty {
                        responseFormat = .jsonSchema(schemaJSON: schemaJSON)
                    } else {
                        responseFormat = .text
                    }
                    
                    // For text mode, bypass the pipeline and stream directly
                    if case .text = responseFormat {
                        let params = GenerateParameters(
                            maxTokens: sampling.maxTokens ?? 1024,
                            temperature: Float(sampling.temperature ?? 0.7),
                            topP: Float(sampling.topP ?? 1.0)
                        )
                        
                        let directStream = await executor.executeStream(
                            prompt: prompt,
                            parameters: params,
                            logitProcessor: nil
                        )
                        
                        for try await text in directStream {
                            try Task.checkCancellation()
                            continuation.yield(text)
                        }
                        continuation.finish()
                        return
                    }
                    
                    // For JSON schema mode, use the pipeline
                    let modelID = await currentModel() ?? "unknown"
                    let req = ChatRequest(
                        modelID: modelID,
                        prompt: prompt,
                        responseFormat: responseFormat,
                        sampling: sampling,
                        schema: schema,
                        parameters: nil
                    )
                    
                    try Task.checkCancellation()
                    
                    let stream = await orchestrator.stream(req)
                    
                    for try await chunk in stream {
                        try Task.checkCancellation()
                        for delta in chunk.deltas {
                            if let text = delta.deltaText {
                                continuation.yield(text)
                            }
                        }
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