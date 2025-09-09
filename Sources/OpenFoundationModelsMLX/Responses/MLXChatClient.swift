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
        let engine = MLXChatEngine(backend: backend)
        return try await engine.generate(ChatRequest(
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
                let engine = MLXChatEngine(backend: backend)
                let stream = await engine.stream(ChatRequest(
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
            }
        }
    }
}