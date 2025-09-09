import Foundation
import OpenFoundationModels
import OpenFoundationModelsExtra
import MLXLMCommon

/// MLXLanguageModel is the provider adapter that conforms to the
/// OpenFoundationModels LanguageModel protocol, delegating core work to the
/// internal MLXChatEngine. This class focuses exclusively on inference with
/// pre-loaded models and does NOT handle model loading.
public struct MLXLanguageModel: OpenFoundationModels.LanguageModel, Sendable {
    private let card: any ModelCard
    private let engine: MLXChatEngine
    private let modelID: String

    /// Initialize with a pre-loaded ModelContainer and ModelCard.
    /// The model must be loaded separately using ModelLoader.
    /// - Parameters:
    ///   - modelContainer: Pre-loaded model from ModelLoader
    ///   - card: ModelCard that defines prompt rendering and parameters
    ///   - modelID: Identifier for the model (for reference)
    public init(modelContainer: ModelContainer, card: any ModelCard, modelID: String) async throws {
        self.card = card
        self.modelID = modelID
        
        // Create backend with the pre-loaded model
        let backend = MLXBackend()
        await backend.setModel(modelContainer, modelID: modelID)
        
        // Initialize engine with the backend
        self.engine = MLXChatEngine(backend: backend)
    }
    
    /// Convenience initializer when you have a pre-configured backend
    /// - Parameters:
    ///   - backend: Pre-configured MLXBackend with model already set
    ///   - card: ModelCard that defines prompt rendering and parameters
    ///   - modelID: Identifier for the model
    public init(backend: MLXBackend, card: any ModelCard, modelID: String) {
        self.card = card
        self.modelID = modelID
        self.engine = MLXChatEngine(backend: backend)
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
        } catch let error as MLXBackend.MLXBackendError {
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
                                    // Buffer limit check (2MB, same as engine)
                                    let bufferLimitBytes = 2 * 1024 * 1024
                                    if buffer.utf8.count > bufferLimitBytes {
                                        Logger.warning("[MLXLanguageModel] Tool detection buffer exceeded (\(bufferLimitBytes/1024)KB)")
                                        let msg = "[Error] Stream buffer exceeded during tool detection"
                                        continuation.yield(.response(.init(assetIDs: [], segments: [.text(.init(content: msg))])))
                                        continuation.finish()
                                        return
                                    }
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
