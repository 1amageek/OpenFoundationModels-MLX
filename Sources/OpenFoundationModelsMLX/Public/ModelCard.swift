import Foundation
import MLXLMCommon
import OpenFoundationModels

/// Public model card interface.
/// Cards fully own prompt rendering and default generation parameters.
public protocol ModelCard: Identifiable, Sendable where ID == String {
    /// Backend model identifier (e.g., Hugging Face repo id or local id).
    var id: String { get }

    /// Generate prompt from transcript and options using OpenFoundationModels' PromptBuilder
    func prompt(transcript: Transcript, options: GenerationOptions?) -> Prompt

    /// Default generation parameters for this model (used as-is; no fallback/merge in MLX layer).
    var params: GenerateParameters { get }

    /// Process generated text output and create a Transcript.Entry
    /// - Parameters:
    ///   - raw: The raw generated text from the model
    ///   - options: Generation options that might affect output processing
    /// - Returns: A Transcript.Entry with processed content
    func generate(from raw: String, options: GenerationOptions?) -> Transcript.Entry

    /// Process streaming output chunks into Transcript.Entry stream
    /// - Parameters:
    ///   - chunks: Raw text chunks from the model stream
    ///   - options: Generation options that might affect output processing
    /// - Returns: A stream of processed Transcript.Entry items
    func stream(
        from chunks: AsyncThrowingStream<String, Error>,
        options: GenerationOptions?
    ) -> AsyncThrowingStream<Transcript.Entry, Error>
}

// Default implementations
extension ModelCard {
    /// Default implementation: returns raw text as assistant entry
    public func generate(from raw: String, options: GenerationOptions?) -> Transcript.Entry {
        return .response(.init(assetIDs: [], segments: [.text(.init(content: raw))]))
    }

    /// Default implementation: streams raw chunks as assistant responses
    public func stream(
        from chunks: AsyncThrowingStream<String, Error>,
        options: GenerationOptions?
    ) -> AsyncThrowingStream<Transcript.Entry, Error> {
        return AsyncThrowingStream { continuation in
            Task {
                do {
                    for try await chunk in chunks {
                        let entry = Transcript.Entry.response(.init(
                            assetIDs: [],
                            segments: [.text(.init(content: chunk))]
                        ))
                        continuation.yield(entry)
                    }
                    continuation.finish()
                } catch {
                    continuation.finish(throwing: error)
                }
            }
        }
    }
}
