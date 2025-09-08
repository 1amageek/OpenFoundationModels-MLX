import Foundation

// Optional developer-facing facade that mirrors a minimal Chat API.
// This is not part of the public LanguageModel surface but can aid local tests
// and tooling. It delegates to MLXChatEngine.
struct MLXChatClient: Sendable {
    struct Request: Sendable {
        var card: any ModelCard
        var messages: [ChatMessage]
        var responseFormat: ResponseFormatSpec = .text
        var sampling: SamplingParameters = .init(temperature: nil, topP: nil, topK: nil, maxTokens: nil, stop: nil, seed: nil)
        var policy: ChatPolicy = .init()
        init(card: any ModelCard, messages: [ChatMessage]) { self.card = card; self.messages = messages }
    }

    init() {}

    func create(_ req: Request) async throws -> ChatResponse {
        let engine = try await MLXChatEngine(modelID: req.card.id)
        // Render via card
        let df = ISO8601DateFormatter(); df.formatOptions = [.withFullDate]
        let now = df.string(from: Date())
        let input = ModelCardInput(
            currentDate: now,
            system: req.messages.first(where: { $0.role == .system })?.content,
            messages: req.messages
                .filter { $0.role != .system }  // Exclude system to avoid duplication
                .map { m in
                    switch m.role {
                    case .user: return .init(role: .user, content: m.content)
                    case .assistant: return .init(role: .assistant, content: m.content)
                    case .system: return .init(role: .system, content: m.content)  // Should not reach
                    }
                },
            tools: []
        )
        let prompt = try req.card.render(input: input)
        // Developer utility: prefer sampling on explicit Request.sampling
        return try await engine.generate(.init(modelID: req.card.id, messages: req.messages, responseFormat: req.responseFormat, sampling: req.sampling, policy: req.policy, schema: nil, promptOverride: prompt, parameters: nil))
    }

    func createStream(_ req: Request) -> AsyncThrowingStream<ChatChunk, Error> {
        return AsyncThrowingStream { continuation in
            Task {
                do {
                    let engine = try await MLXChatEngine(modelID: req.card.id)
                    let df = ISO8601DateFormatter(); df.formatOptions = [.withFullDate]
                    let now = df.string(from: Date())
                    let input = ModelCardInput(
                        currentDate: now,
                        system: req.messages.first(where: { $0.role == .system })?.content,
                        messages: req.messages
                            .filter { $0.role != .system }  // Exclude system to avoid duplication
                            .map { m in
                                switch m.role {
                                case .user: return .init(role: .user, content: m.content)
                                case .assistant: return .init(role: .assistant, content: m.content)
                                case .system: return .init(role: .system, content: m.content)  // Should not reach
                                }
                            },
                        tools: []
                    )
                    let prompt = try req.card.render(input: input)
                    let stream = await engine.stream(.init(modelID: req.card.id, messages: req.messages, responseFormat: req.responseFormat, sampling: req.sampling, policy: req.policy, schema: nil, promptOverride: prompt, parameters: nil))
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
