import Foundation
import MLXLMCommon

// Optional developer-facing facade that mirrors a minimal Chat API.
// This is not part of the public LanguageModel surface but can aid local tests
// and tooling. It delegates to MLXChatEngine.
// NOTE: With the separation of concerns, this now requires a pre-configured backend.
struct MLXChatClient: Sendable {
    struct Request: Sendable {
        var card: any ModelCard
        var prompt: String  // Pre-rendered prompt from ModelCard
        var responseFormat: ResponseFormatSpec = .text
        var sampling: SamplingParameters = .init(temperature: nil, topP: nil, topK: nil, maxTokens: nil, stop: nil, seed: nil)
        var schema: SchemaNode? = nil
        
        // Direct prompt initializer (the only way now)
        init(card: any ModelCard, prompt: String) {
            self.card = card
            self.prompt = prompt
        }
    }

    private let backend: MLXBackend
    
    /// Initialize with a pre-configured backend
    /// The backend must have a model already set via setModel()
    init(backend: MLXBackend) {
        self.backend = backend
    }

    func create(_ req: Request) async throws -> ChatResponse {
        let schemaJSON: String? = {
            if case .jsonSchema(let json) = req.responseFormat {
                return json
            }
            return nil
        }()
        
        let text = try await backend.orchestratedGenerate(
            prompt: req.prompt,
            sampling: req.sampling,
            schema: req.schema,
            schemaJSON: schemaJSON
        )
        
        let choice = ChatChoice(content: text, finishReason: "stop")
        return ChatResponse(
            choices: [choice],
            usage: .init(promptTokens: 0, completionTokens: 0),
            meta: .init()
        )
    }

    func createStream(_ req: Request) -> AsyncThrowingStream<ChatChunk, Error> {
        return AsyncThrowingStream { continuation in
            Task {
                let schemaJSON: String? = {
                    if case .jsonSchema(let json) = req.responseFormat {
                        return json
                    }
                    return nil
                }()
                
                let stream = await backend.orchestratedStream(
                    prompt: req.prompt,
                    sampling: req.sampling,
                    schema: req.schema,
                    schemaJSON: schemaJSON
                )
                
                do {
                    for try await text in stream {
                        let chunk = ChatChunk(deltas: [.init(deltaText: text, finishReason: nil)])
                        continuation.yield(chunk)
                    }
                    let finalChunk = ChatChunk(deltas: [.init(deltaText: nil, finishReason: "stop")])
                    continuation.yield(finalChunk)
                    continuation.finish()
                } catch {
                    continuation.finish(throwing: error)
                }
            }
        }
    }
}