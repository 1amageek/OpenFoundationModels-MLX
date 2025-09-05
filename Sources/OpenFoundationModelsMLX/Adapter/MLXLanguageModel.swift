import Foundation
import OpenFoundationModels
import OpenFoundationModelsExtra

// MLXLanguageModel is the provider adapter that conforms to the
// OpenFoundationModels LanguageModel protocol, delegating core work to the
// internal MLXChatEngine. The public API surface remains 100% compatible.
public struct MLXLanguageModel: LanguageModel, Sendable {
    private let modelID: String
    private let engine: MLXChatEngine

    public init(modelID: String) async throws {
        self.modelID = modelID
        self.engine = try await MLXChatEngine(modelID: modelID)
    }

    public var isAvailable: Bool { true }

    public func supports(locale: Locale) -> Bool { true }

    public func generate(transcript: Transcript, options: GenerationOptions?) async throws -> Transcript.Entry {
        let req = PromptRenderer.buildRequest(modelID: modelID, transcript: transcript, options: options)
        do {
            let res = try await engine.generate(req)
            // Convert the assistant text into a Transcript.Entry.response in a
            // conservative way: return plain text response segment.
            if let text = res.choices.first?.message.content {
                if let entry = ToolCallDetector.entryIfPresent(text) {
                    return entry
                }
                return .response(.init(assetIDs: [], segments: [.text(.init(content: text))]))
            }
        } catch {
            // Map internal errors onto the public GenerationError surface.
            throw GenerationError.decodingFailure(.init(debugDescription: String(describing: error)))
        }
        // If no content was produced, surface a decoding failure consistent with
        // the public API semantics.
        throw GenerationError.decodingFailure(.init(debugDescription: "empty response"))
    }

    public func stream(transcript: Transcript, options: GenerationOptions?) -> AsyncStream<Transcript.Entry> {
        let req = PromptRenderer.buildRequest(modelID: modelID, transcript: transcript, options: options)
        let expectsTool = TranscriptAccess.extract(from: transcript).toolDefs.isEmpty == false
        return AsyncStream { continuation in
            Task {
                let stream = await engine.stream(req)
                do {
                    var buffer = ""
                    var emittedToolCalls = false
                    for try await chunk in stream {
                        // Yield text deltas as response segments (not accumulated)
                        for delta in chunk.deltas {
                            if let text = delta.deltaText, !text.isEmpty {
                                if expectsTool {
                                    buffer += text
                                    if let entry = ToolCallDetector.entryIfPresent(buffer) {
                                        continuation.yield(entry)
                                        emittedToolCalls = true
                                        continuation.finish()
                                        return
                                    }
                                } else {
                                    continuation.yield(.response(.init(assetIDs: [], segments: [.text(.init(content: text))])))
                                }
                            }
                            if let reason = delta.finishReason, !reason.isEmpty {
                                // We don't emit a special entry for finish; the
                                // stream naturally ends after this iteration.
                                _ = reason
                            }
                        }
                    }
                    if expectsTool && !emittedToolCalls {
                        // Fall back to text if no tool_calls detected.
                        if !buffer.isEmpty {
                            continuation.yield(.response(.init(assetIDs: [], segments: [.text(.init(content: buffer))])))
                        }
                    }
                    continuation.finish()
                } catch {
                    // Log error since AsyncStream cannot propagate errors
                    Logger.error("[MLXLanguageModel] Stream error: \(error)")
                    
                    // If possible, send an error indicator as text
                    // This helps the client know an error occurred
                    let errorMessage = "[Error: \(error.localizedDescription)]"
                    continuation.yield(.response(.init(assetIDs: [], segments: [.text(.init(content: errorMessage))])))
                    
                    continuation.finish()
                }
            }
        }
    }
}
