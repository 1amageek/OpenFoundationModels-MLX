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
    
    func generateTextConstrained(
        prompt: String,
        sampling: SamplingParameters,
        schema: SchemaMeta
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
    
    func streamTextConstrained(
        prompt: String,
        sampling: SamplingParameters,
        schema: SchemaMeta
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
    
    // MARK: - JSON Generation Convenience (Legacy Interface)
    
    func generateJSON<T: Decodable>(
        prompt: String,
        schema: T.Type,
        sampling: SamplingParameters = SamplingParameters()
    ) async throws -> T {
        // For now, we need to manually specify the schema keys
        // In a real implementation, this would be extracted from the Decodable type
        let schemaMeta = SchemaMeta(keys: [], required: [])
        
        // Generate constrained text
        let jsonText = try await generateTextConstrained(
            prompt: prompt,
            sampling: sampling,
            schema: schemaMeta
        )
        
        // Parse JSON
        guard let data = jsonText.data(using: .utf8) else {
            throw MLXBackendError.generationFailed("Invalid UTF-8 in generated JSON")
        }
        
        let decoder = JSONDecoder()
        return try decoder.decode(T.self, from: data)
    }
}