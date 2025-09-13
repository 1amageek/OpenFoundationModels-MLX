import Foundation
import OpenFoundationModels
import OpenFoundationModelsExtra
import MLXLMCommon
import MLXLLM

/// MLXLanguageModel is the provider adapter that conforms to the
/// OpenFoundationModels LanguageModel protocol, delegating core work to the
/// internal MLXChatEngine. This class focuses exclusively on inference with
/// pre-loaded models and does NOT handle model loading.
public struct MLXLanguageModel: OpenFoundationModels.LanguageModel, Sendable {
    private let card: any ModelCard
    private let backend: MLXBackend

    /// Initialize with a pre-loaded ModelContainer and ModelCard.
    /// The model must be loaded separately using ModelLoader.
    /// - Parameters:
    ///   - modelContainer: Pre-loaded model from ModelLoader
    ///   - card: ModelCard that defines prompt rendering and parameters
    ///   - additionalProcessors: Optional additional LogitProcessors (e.g., KeyDetectionLogitProcessor)
    public init(
        modelContainer: ModelContainer, 
        card: any ModelCard,
        additionalProcessors: [LogitProcessor] = []
    ) async throws {
        self.card = card
        
        let backend = MLXBackend(additionalProcessors: additionalProcessors)
        await backend.setModel(modelContainer, modelID: card.id)
        self.backend = backend
    }
    
    /// Convenience initializer when you have a pre-configured backend
    /// - Parameters:
    ///   - backend: Pre-configured MLXBackend with model already set
    ///   - card: ModelCard that defines prompt rendering and parameters
    public init(backend: MLXBackend, card: any ModelCard) {
        self.card = card
        self.backend = backend
    }

    public var isAvailable: Bool { true }

    public func supports(locale: Locale) -> Bool { true }

    public func generate(transcript: Transcript, options: GenerationOptions?) async throws -> Transcript.Entry {
        let prompt = card.prompt(transcript: transcript, options: options)
        let ext = TranscriptAccess.extract(from: transcript)
        
        // Extract schema if present
        let schemaNode: SchemaNode? = {
            if let schemaJSON = ext.schemaJSON, !schemaJSON.isEmpty {
                return JSONSchemaExtractor.buildSchemaNode(from: schemaJSON)
            }
            return nil
        }()
        
        // Prepare parameters
        let sampling = OptionsMapper.map(options)
        
        do {
            // Generate raw text through backend
            let raw = try await backend.orchestratedGenerate(
                prompt: prompt.description,
                sampling: sampling,
                schema: schemaNode,
                schemaJSON: ext.schemaJSON,
                modelCard: card
            )
            
            // Debug: Log generated content before processing
            Logger.info("[MLXLanguageModel] Generated content:")
            Logger.info("[MLXLanguageModel] ========== START ==========")
            Logger.info(raw)
            Logger.info("[MLXLanguageModel] ========== END ==========")
            
            // Use ModelCard to process the output
            let entry = card.generate(from: raw, options: options)
            
            // Check for tool calls if needed
            if case .response(let response) = entry,
               let segment = response.segments.first,
               case .text(let textSegment) = segment,
               let toolEntry = ToolCallDetector.entryIfPresent(textSegment.content) {
                return toolEntry
            }
            
            return entry
        } catch let error as CancellationError {
            throw error
        } catch let error as MLXBackend.MLXBackendError {
            throw GenerationError.decodingFailure(.init(debugDescription: "Backend error: \(error.localizedDescription)"))
        } catch {
            throw GenerationError.decodingFailure(.init(debugDescription: String(describing: error)))
        }
    }

    public func stream(transcript: Transcript, options: GenerationOptions?) -> AsyncStream<Transcript.Entry> {
        let prompt = card.prompt(transcript: transcript, options: options)
        let ext = TranscriptAccess.extract(from: transcript)
        let expectsTool = ext.toolDefs.isEmpty == false
        
        // Extract schema if present
        let schemaNode: SchemaNode? = {
            if let schemaJSON = ext.schemaJSON, !schemaJSON.isEmpty {
                return JSONSchemaExtractor.buildSchemaNode(from: schemaJSON)
            }
            return nil
        }()
        
        // Prepare parameters
        let sampling = OptionsMapper.map(options)
        
        return AsyncStream { continuation in
            let task = Task {
                do {
                    try Task.checkCancellation()
                    
                    // Get raw stream from backend
                    let rawStream = await backend.orchestratedStream(
                        prompt: prompt.description,
                        sampling: sampling,
                        schema: schemaNode,
                        schemaJSON: ext.schemaJSON,
                        modelCard: card
                    )
                    
                    // Convert to AsyncThrowingStream for ModelCard
                    let throwingStream = AsyncThrowingStream<String, Error> { streamContinuation in
                        Task {
                            do {
                                for try await chunk in rawStream {
                                    streamContinuation.yield(chunk)
                                }
                                streamContinuation.finish()
                            } catch {
                                streamContinuation.finish(throwing: error)
                            }
                        }
                    }
                    
                    // Use ModelCard to process the stream
                    let processedStream = card.stream(from: throwingStream, options: options)
                    
                    // Handle tool detection if needed
                    if expectsTool {
                        var buffer = ""
                        var emittedToolCalls = false
                        let bufferLimitBytes = 2 * 1024 * 1024
                        
                        for try await entry in processedStream {
                            try Task.checkCancellation()
                            
                            // Extract text content from entry
                            if case .response(let response) = entry,
                               let segment = response.segments.first,
                               case .text(let textSegment) = segment {
                                buffer += textSegment.content
                                
                                if buffer.utf8.count > bufferLimitBytes {
                                    Logger.warning("[MLXLanguageModel] Tool detection buffer exceeded")
                                    let msg = "[Error] Stream buffer exceeded during tool detection"
                                    continuation.yield(.response(.init(assetIDs: [], segments: [.text(.init(content: msg))])))
                                    continuation.finish()
                                    return
                                }
                                
                                if let toolEntry = ToolCallDetector.entryIfPresent(buffer) {
                                    continuation.yield(toolEntry)
                                    emittedToolCalls = true
                                    continuation.finish()
                                    return
                                }
                            }
                            
                            // Stream the entry if not buffering for tools
                            if !expectsTool {
                                continuation.yield(entry)
                            }
                        }
                        
                        // If we were buffering for tools but found none, emit the buffer
                        if !emittedToolCalls && !buffer.isEmpty {
                            continuation.yield(.response(.init(assetIDs: [], segments: [.text(.init(content: buffer))])))
                        }
                    } else {
                        // No tool detection needed, stream directly
                        for try await entry in processedStream {
                            try Task.checkCancellation()
                            continuation.yield(entry)
                        }
                    }
                    
                    continuation.finish()
                } catch is CancellationError {
                    continuation.finish()
                } catch {
                    Logger.error("[MLXLanguageModel] Stream error: \(error)")
                    let errorMessage = "[Error] Stream generation failed: \(error.localizedDescription)"
                    continuation.yield(.response(.init(assetIDs: [], segments: [.text(.init(content: errorMessage))])))
                    continuation.finish()
                }
            }
            
            continuation.onTermination = { @Sendable _ in
                task.cancel()
            }
        }
    }
}
