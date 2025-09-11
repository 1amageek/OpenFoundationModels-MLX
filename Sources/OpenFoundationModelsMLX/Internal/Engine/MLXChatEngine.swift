import Foundation

actor MLXChatEngine {
    private let backend: MLXBackend

    init(backend: MLXBackend) {
        self.backend = backend
    }

    func generate(_ req: ChatRequest) async throws -> ChatResponse {
        let prompt = req.prompt
        
        let sampling: SamplingParameters
        if let p = req.parameters {
            sampling = SamplingParameters(
                temperature: Double(p.temperature),
                topP: Double(p.topP),
                topK: nil,
                maxTokens: p.maxTokens,
                stop: nil,
                seed: nil
            )
        } else {
            sampling = req.sampling
        }
        
        let schemaJSON: String? = {
            if case .jsonSchema(let json) = req.responseFormat {
                return json
            }
            return nil
        }()
        
        let text = try await backend.orchestratedGenerate(
            prompt: prompt,
            sampling: sampling,
            schema: req.schema,
            schemaJSON: schemaJSON
        )
        
        switch req.responseFormat {
        case .text: break
        case .jsonSchema:
            break
        case .jsonSchemaRef: break
        }

        let choice = ChatChoice(content: text, finishReason: "stop")
        return ChatResponse(
            choices: [choice], 
            usage: .init(promptTokens: 0, completionTokens: 0), 
            meta: .init()
        )
    }

    func stream(_ req: ChatRequest) -> AsyncThrowingStream<ChatChunk, Error> {
        return AsyncThrowingStream { continuation in
            let mainTask = Task {
                do {
                    try Task.checkCancellation()
                    
                    let prompt = req.prompt
                    
                    let sampling: SamplingParameters
                    if let p = req.parameters {
                        sampling = SamplingParameters(
                            temperature: Double(p.temperature),
                            topP: Double(p.topP),
                            topK: nil,
                            maxTokens: p.maxTokens,
                            stop: nil,
                            seed: nil
                        )
                    } else {
                        sampling = req.sampling
                    }
                    
                            let schemaJSON: String? = {
                        if case .jsonSchema(let json) = req.responseFormat {
                            return json
                        }
                        return nil
                    }()
                    
                    try Task.checkCancellation()
                    
                    let textStream = await backend.orchestratedStream(
                        prompt: prompt,
                        sampling: sampling,
                        schema: req.schema,
                        schemaJSON: schemaJSON
                    )
                    
                    for try await piece in textStream {
                        try Task.checkCancellation()
                        continuation.yield(ChatChunk(deltas: [.init(deltaText: piece, finishReason: nil)]))
                    }
                    
                    try Task.checkCancellation()
                    continuation.yield(ChatChunk(deltas: [.init(deltaText: nil, finishReason: "stop")]))
                    continuation.finish()
                } catch is CancellationError {
                    continuation.finish(throwing: CancellationError())
                } catch {
                    continuation.finish(throwing: error)
                }
            }
            continuation.onTermination = { @Sendable _ in
                mainTask.cancel()
            }
        }
    }
}