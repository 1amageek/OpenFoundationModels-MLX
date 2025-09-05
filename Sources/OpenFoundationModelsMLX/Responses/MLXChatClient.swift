import Foundation

// Optional developer-facing facade that mirrors a minimal Chat API.
// This is not part of the public LanguageModel surface but can aid local tests
// and tooling. It delegates to MLXChatEngine.
struct MLXChatClient: Sendable {
    struct Request: Sendable {
        var modelID: String
        var messages: [ChatMessage]
        var responseFormat: ResponseFormatSpec = .text
        var sampling: SamplingParameters = .init(temperature: nil, topP: nil, topK: nil, maxTokens: nil, stop: nil, seed: nil)
        var policy: ChatPolicy = .init()
        init(modelID: String, messages: [ChatMessage]) { self.modelID = modelID; self.messages = messages }
    }

    init() {}

    func create(_ req: Request) async throws -> ChatResponse {
        let engine = try await MLXChatEngine(modelID: req.modelID)
        return try await engine.generate(.init(modelID: req.modelID, messages: req.messages, responseFormat: req.responseFormat, sampling: req.sampling, policy: req.policy, schema: nil))
    }

    func createStream(_ req: Request) -> AsyncThrowingStream<ChatChunk, Error> {
        return AsyncThrowingStream { continuation in
            Task {
                do {
                    let engine = try await MLXChatEngine(modelID: req.modelID)
                    let stream = await engine.stream(.init(modelID: req.modelID, messages: req.messages, responseFormat: req.responseFormat, sampling: req.sampling, policy: req.policy, schema: nil))
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
