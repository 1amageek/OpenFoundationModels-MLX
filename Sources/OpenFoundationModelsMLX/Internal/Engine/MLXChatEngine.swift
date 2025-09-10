import Foundation

// Core engine actor that orchestrates text generation through MLXBackend.
// Delegates all schema handling to ADAPT system.
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
        
        let text: String
        if let schemaNode = req.schema {
            text = try await backend.generateTextWithSchema(prompt: prompt, sampling: sampling, schema: schemaNode)
        } else {
            text = try await backend.generateText(prompt: prompt, sampling: sampling)
        }
        
        switch req.responseFormat {
        case .text: break
        case .jsonSchema:
            if let schemaNode = req.schema {
                if JSONValidator.validate(text: text, schema: schemaNode) == false { 
                    throw ValidationError.schemaUnsatisfied 
                }
            }
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
                let prompt = req.prompt
                
                // Convert parameters
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
                
                // Get stream from backend (ADAPT handles all schema constraints)
                let textStream: AsyncThrowingStream<String, Error>
                if let schema = req.schema {
                    textStream = await backend.streamTextWithSchema(prompt: prompt, sampling: sampling, schema: schema)
                } else {
                    textStream = await backend.streamText(prompt: prompt, sampling: sampling)
                }
                
                // Simply pass through the stream
                do {
                    for try await piece in textStream {
                        if Task.isCancelled { break }
                        continuation.yield(ChatChunk(deltas: [.init(deltaText: piece, finishReason: nil)]))
                    }
                    continuation.yield(ChatChunk(deltas: [.init(deltaText: nil, finishReason: "stop")]))
                    continuation.finish()
                } catch {
                    continuation.finish(throwing: error)
                }
            }
            continuation.onTermination = { _ in
                mainTask.cancel()
            }
        }
    }
}


private enum ValidationError: Error { 
    case schemaUnsatisfied
}