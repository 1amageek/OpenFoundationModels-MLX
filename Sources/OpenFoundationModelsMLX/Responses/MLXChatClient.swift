import Foundation
import MLXLMCommon

// Optional developer-facing facade that mirrors a minimal Chat API.
// This is not part of the public LanguageModel surface but can aid local tests
// and tooling. It delegates to MLXChatEngine.
struct MLXChatClient: Sendable {
    struct Request: Sendable {
        var card: any ModelCard
        var prompt: String  // Pre-rendered prompt from ModelCard
        var responseFormat: ResponseFormatSpec = .text
        var sampling: SamplingParameters = .init(temperature: nil, topP: nil, topK: nil, maxTokens: nil, stop: nil, seed: nil)
        var schema: SchemaMeta? = nil
        
        // Direct prompt initializer (the only way now)
        init(card: any ModelCard, prompt: String) {
            self.card = card
            self.prompt = prompt
        }
    }

    init() {}

    func create(_ req: Request) async throws -> ChatResponse {
        let engine = try await MLXChatEngine(modelID: req.card.id)
        return try await engine.generate(.init(
            modelID: req.card.id, 
            prompt: req.prompt,
            responseFormat: req.responseFormat, 
            sampling: req.sampling, 
            schema: req.schema, 
            parameters: nil
        ))
    }

    func createStream(_ req: Request) -> AsyncThrowingStream<ChatChunk, Error> {
        return AsyncThrowingStream { continuation in
            Task {
                do {
                    let engine = try await MLXChatEngine(modelID: req.card.id)
                    let stream = await engine.stream(.init(
                        modelID: req.card.id, 
                        prompt: req.prompt,
                        responseFormat: req.responseFormat, 
                        sampling: req.sampling, 
                        schema: req.schema, 
                        parameters: nil
                    ))
                    do {
                        for try await chunk in stream {
                            continuation.yield(chunk)
                        }
                        continuation.finish()
                    } catch {
                        continuation.finish(throwing: error)
                    }
                } catch {
                    // Handle initialization failure to prevent hang
                    continuation.finish(throwing: error)
                }
            }
        }
    }
}