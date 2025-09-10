import Foundation
import MLXLMCommon

/// MLXBackend V2 - Simplified facade that delegates to the new architecture.
/// This maintains backward compatibility while using the new separated layers.
public actor MLXBackend {
    
    // MARK: - Properties
    
    private let executor: MLXExecutor
    private let adaptEngine: ADAPTEngine
    private let orchestrator: GenerationOrchestrator
    
    // MARK: - Legacy Error Type (for compatibility)
    
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
    
    // MARK: - Initialization
    
    public init() {
        self.executor = MLXExecutor()
        self.adaptEngine = ADAPTEngine()
        self.orchestrator = GenerationOrchestrator(
            executor: executor,
            adaptEngine: adaptEngine
        )
    }
    
    // MARK: - Model Management (Legacy Interface)
    
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
    
    // MARK: - Text Generation (Legacy Interface)
    
    func generateText(
        prompt: String,
        sampling: SamplingParameters
    ) async throws -> String {
        guard await hasModel() else {
            throw MLXBackendError.noModelSet
        }
        
        // Convert to GenerateParameters
        let genParams = GenerateParameters(
            maxTokens: sampling.maxTokens ?? 1024,
            temperature: Float(sampling.temperature ?? 0.7),
            topP: Float(sampling.topP ?? 1.0)
        )
        
        return try await executor.execute(
            prompt: prompt,
            parameters: genParams
        )
    }
    
    // Generate text with schema constraints
    func generateTextWithSchema(
        prompt: String,
        sampling: SamplingParameters,
        schema: SchemaNode
    ) async throws -> String {
        guard await hasModel() else {
            throw MLXBackendError.noModelSet
        }
        
        return try await adaptEngine.generateWithSchema(
            executor: executor,
            prompt: prompt,
            schema: schema,
            parameters: sampling
        )
    }
    
    // MARK: - Streaming Generation (Legacy Interface)
    
    func streamText(
        prompt: String,
        sampling: SamplingParameters
    ) -> AsyncThrowingStream<String, Error> {
        AsyncThrowingStream { continuation in
            Task {
                guard await hasModel() else {
                    continuation.finish(throwing: MLXBackendError.noModelSet)
                    return
                }
                
                // Convert to GenerateParameters
                let genParams = GenerateParameters(
                    maxTokens: sampling.maxTokens ?? 1024,
                    temperature: Float(sampling.temperature ?? 0.7),
                    topP: Float(sampling.topP ?? 1.0)
                )
                
                let stream = await executor.executeStream(
                    prompt: prompt,
                    parameters: genParams
                )
                
                do {
                    for try await chunk in stream {
                        continuation.yield(chunk)
                    }
                    continuation.finish()
                } catch {
                    continuation.finish(throwing: error)
                }
            }
        }
    }
    
    // Stream text generation with schema constraints
    func streamTextWithSchema(
        prompt: String,
        sampling: SamplingParameters,
        schema: SchemaNode
    ) -> AsyncThrowingStream<String, Error> {
        AsyncThrowingStream { continuation in
            Task {
                guard await hasModel() else {
                    continuation.finish(throwing: MLXBackendError.noModelSet)
                    return
                }
                
                let stream = await adaptEngine.streamWithSchema(
                    executor: executor,
                    prompt: prompt,
                    schema: schema,
                    parameters: sampling
                )
                
                do {
                    for try await chunk in stream {
                        continuation.yield(chunk)
                    }
                    continuation.finish()
                } catch {
                    continuation.finish(throwing: error)
                }
            }
        }
    }
}