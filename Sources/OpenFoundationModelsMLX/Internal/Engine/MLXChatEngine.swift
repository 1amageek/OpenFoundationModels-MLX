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
        
        // schemaJSONを抽出
        let schemaJSON: String? = {
            if case .jsonSchema(let json) = req.responseFormat {
                return json
            }
            return nil
        }()
        
        // GenerationOrchestrator経由で実行（リトライ、バリデーション機能あり）
        let text = try await backend.orchestratedGenerate(
            prompt: prompt,
            sampling: sampling,
            schema: req.schema,
            schemaJSON: schemaJSON
        )
        
        // Orchestratorがバリデーションも行うため、ここでの追加チェックは不要
        // ただし互換性のため responseFormat のチェックは残す
        switch req.responseFormat {
        case .text: break
        case .jsonSchema:
            // Orchestratorで既にバリデーション済み
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
                
                // schemaJSONを抽出
                let schemaJSON: String? = {
                    if case .jsonSchema(let json) = req.responseFormat {
                        return json
                    }
                    return nil
                }()
                
                // GenerationOrchestrator経由でストリーミング（リトライ、ADAPT制約を含む）
                let textStream = await backend.orchestratedStream(
                    prompt: prompt,
                    sampling: sampling,
                    schema: req.schema,
                    schemaJSON: schemaJSON
                )
                
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