import Foundation
import OpenFoundationModels
import OpenFoundationModelsExtra
import MLXLMCommon

// MLXLanguageModel is the provider adapter that conforms to the
// OpenFoundationModels LanguageModel protocol, delegating core work to the
// internal MLXChatEngine. The public API surface remains 100% compatible.
public struct MLXLanguageModel: OpenFoundationModels.LanguageModel, Sendable {
    private let card: any ModelCard
    private let engine: MLXChatEngine

    /// Initialize with a ModelCard. The card fully defines prompt rendering and default parameters.
    public init(card: any ModelCard) async throws {
        self.card = card
        self.engine = try await MLXChatEngine(modelID: card.id)
    }

    public var isAvailable: Bool { true }

    public func supports(locale: Locale) -> Bool { true }

    public func generate(transcript: Transcript, options: GenerationOptions?) async throws -> Transcript.Entry {
        let req = try PromptRenderer.buildRequest(card: card, transcript: transcript, options: options)
        do {
            let res = try await engine.generate(req)
            // Convert the assistant text into a Transcript.Entry.response in a
            // conservative way: return plain text response segment.
            if let text = res.choices.first?.content {
                if let entry = ToolCallDetector.entryIfPresent(text) {
                    return entry
                }
                return .response(.init(assetIDs: [], segments: [.text(.init(content: text))]))
            }
        } catch let error as CancellationError {
            // Rethrow cancellation errors without wrapping
            throw error
        } catch let error as MLXBackendError {
            // Map backend errors with more detail
            throw GenerationError.decodingFailure(.init(debugDescription: "Backend error: \(error.localizedDescription)"))
        } catch {
            // Map other internal errors onto the public GenerationError surface
            throw GenerationError.decodingFailure(.init(debugDescription: String(describing: error)))
        }
        // If no content was produced, surface a decoding failure consistent with
        // the public API semantics.
        throw GenerationError.decodingFailure(.init(debugDescription: "empty response"))
    }

    public func stream(transcript: Transcript, options: GenerationOptions?) -> AsyncStream<Transcript.Entry> {
        // Fail-fast: if request building fails, emit a single error entry and finish.
        guard let req = try? PromptRenderer.buildRequest(card: card, transcript: transcript, options: options) else {
            return AsyncStream { continuation in
                continuation.yield(.response(.init(assetIDs: [], segments: [.text(.init(content: "[Error] Prompt rendering failed"))])))
                continuation.finish()
            }
        }
        let expectsTool = TranscriptAccess.extract(from: transcript).toolDefs.isEmpty == false
        return AsyncStream { continuation in
            let task = Task {
                let stream = await engine.stream(req)
                do {
                    var buffer = ""
                    var emittedToolCalls = false
                    for try await chunk in stream {
                        // Check for task cancellation
                        if Task.isCancelled {
                            break
                        }
                        
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
                    
                    // Send error indicator with consistent format
                    // Prefix with [Error] for clear identification
                    let errorMessage = "[Error] Stream generation failed: \(error.localizedDescription)"
                    continuation.yield(.response(.init(assetIDs: [], segments: [.text(.init(content: errorMessage))])))
                    
                    continuation.finish()
                }
            }
            
            // Set up cancellation handler
            continuation.onTermination = { _ in
                task.cancel()
            }
        }
    }
}
