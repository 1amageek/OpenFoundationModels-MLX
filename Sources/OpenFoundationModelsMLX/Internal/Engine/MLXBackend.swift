import Foundation
import MLXLMCommon

/// MLXBackend V2 - Simplified facade that delegates to the new architecture.
/// This maintains backward compatibility while using the new separated layers.
public actor MLXBackend {
    
    
    private let executor: MLXExecutor
    private let adaptEngine: ADAPTEngine
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
        self.adaptEngine = ADAPTEngine()
        self.orchestrator = GenerationOrchestrator(
            executor: executor,
            adaptEngine: adaptEngine
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
    
    
    // Orchestrator経由のメソッド（MLXChatEngineが使用）
    func orchestratedGenerate(
        prompt: String,
        sampling: SamplingParameters,
        schema: SchemaNode? = nil,
        schemaJSON: String? = nil
    ) async throws -> String {
        guard await hasModel() else {
            throw MLXBackendError.noModelSet
        }
        
        let modelID = await currentModel() ?? "unknown"
        let responseFormat: ResponseFormatSpec
        if let schemaJSON = schemaJSON, !schemaJSON.isEmpty {
            responseFormat = .jsonSchema(schemaJSON: schemaJSON)
        } else if schema != nil {
            // schemaNodeはあるがJSONがない場合は、スキーマから生成したダミーJSONを使用
            responseFormat = .jsonSchema(schemaJSON: "{}")
        } else {
            responseFormat = .text
        }
        
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
    
    
    // Orchestrator経由のストリーミング（MLXChatEngineが使用）
    func orchestratedStream(
        prompt: String,
        sampling: SamplingParameters,
        schema: SchemaNode? = nil,
        schemaJSON: String? = nil
    ) -> AsyncThrowingStream<String, Error> {
        AsyncThrowingStream { continuation in
            Task {
                guard await hasModel() else {
                    continuation.finish(throwing: MLXBackendError.noModelSet)
                    return
                }
                
                let modelID = await currentModel() ?? "unknown"
                let responseFormat: ResponseFormatSpec
                if let schemaJSON = schemaJSON, !schemaJSON.isEmpty {
                    responseFormat = .jsonSchema(schemaJSON: schemaJSON)
                } else if schema != nil {
                    // schemaNodeはあるがJSONがない場合は、スキーマから生成したダミーJSONを使用
                    responseFormat = .jsonSchema(schemaJSON: "{}")
                } else {
                    responseFormat = .text
                }
                
                let req = ChatRequest(
                    modelID: modelID,
                    prompt: prompt,
                    responseFormat: responseFormat,
                    sampling: sampling,
                    schema: schema,
                    parameters: nil
                )
                
                let stream = await orchestrator.stream(req)
                
                do {
                    for try await chunk in stream {
                        // Extract text from ChatChunk
                        for delta in chunk.deltas {
                            if let text = delta.deltaText {
                                continuation.yield(text)
                            }
                        }
                    }
                    continuation.finish()
                } catch {
                    continuation.finish(throwing: error)
                }
            }
        }
    }
}