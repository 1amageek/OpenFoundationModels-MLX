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

    /// Initialize with a pre-loaded ModelContainer and ModelCard.
    /// The model must be loaded separately using ModelLoader.
    /// - Parameters:
    ///   - modelContainer: Pre-loaded model from ModelLoader
    ///   - card: ModelCard that defines prompt rendering and parameters
    public init(modelContainer: ModelContainer, card: any ModelCard) async throws {
        self.card = card
        
        let backend = MLXBackend()
        await backend.setModel(modelContainer, modelID: card.id)
        
        self.engine = MLXChatEngine(backend: backend)
    }
    
    /// Convenience initializer when you have a pre-configured backend
    /// - Parameters:
    ///   - backend: Pre-configured MLXBackend with model already set
    ///   - card: ModelCard that defines prompt rendering and parameters
    public init(backend: MLXBackend, card: any ModelCard) {
        self.card = card
        self.engine = MLXChatEngine(backend: backend)
    }

    public var isAvailable: Bool { true }

    public func supports(locale: Locale) -> Bool { true }

    public func generate(transcript: Transcript, options: GenerationOptions?) async throws -> Transcript.Entry {
        let prompt = card.prompt(transcript: transcript, options: options)
        
        let ext = TranscriptAccess.extract(from: transcript)
        
        let responseFormat: ResponseFormatSpec = {
            if let schemaJSON = ext.schemaJSON, !schemaJSON.isEmpty { 
                return .jsonSchema(schemaJSON: schemaJSON) 
            }
            return .text
        }()
        
        let schemaNode: SchemaNode? = {
            switch responseFormat {
            case .jsonSchema(let json):
                if let node = SchemaBuilder.fromJSONString(json) {
                    return node
                }
                return nil
            default: 
                return nil
            }
        }()
        
        
        let mergedSampling: SamplingParameters
        let mergedParams: GenerateParameters?
        
        if options != nil {
            mergedSampling = OptionsMapper.map(options)
            mergedParams = nil
        } else {
            let p = card.params
            mergedSampling = SamplingParameters(
                temperature: Double(p.temperature),
                topP: Double(p.topP),
                topK: nil,
                maxTokens: p.maxTokens,
                stop: nil,
                seed: nil
            )
            mergedParams = nil
        }
        
        let req = ChatRequest(
            modelID: card.id,
            prompt: prompt.description,
            responseFormat: responseFormat,
            sampling: mergedSampling,
            schema: schemaNode,
            parameters: mergedParams
        )
        
        do {
            let res = try await engine.generate(req)
            if let text = res.choices.first?.content {
                if let entry = ToolCallDetector.entryIfPresent(text) {
                    return entry
                }
                return .response(.init(assetIDs: [], segments: [.text(.init(content: text))]))
            }
        } catch let error as CancellationError {
            throw error
        } catch let error as MLXBackend.MLXBackendError {
            throw GenerationError.decodingFailure(.init(debugDescription: "Backend error: \(error.localizedDescription)"))
        } catch {
            throw GenerationError.decodingFailure(.init(debugDescription: String(describing: error)))
        }
        throw GenerationError.decodingFailure(.init(debugDescription: "empty response"))
    }

    public func stream(transcript: Transcript, options: GenerationOptions?) -> AsyncStream<Transcript.Entry> {
        let prompt = card.prompt(transcript: transcript, options: options)
        
        let ext = TranscriptAccess.extract(from: transcript)
        let expectsTool = ext.toolDefs.isEmpty == false
        
        let responseFormat: ResponseFormatSpec = {
            if let schemaJSON = ext.schemaJSON, !schemaJSON.isEmpty { 
                return .jsonSchema(schemaJSON: schemaJSON) 
            }
            return .text
        }()
        
        let schemaNode: SchemaNode? = {
            switch responseFormat {
            case .jsonSchema(let json):
                if let node = SchemaBuilder.fromJSONString(json) {
                    return node
                }
                return nil
            default: 
                return nil
            }
        }()
        
        
        let mergedSampling: SamplingParameters
        let mergedParams: GenerateParameters?
        
        if options != nil {
            mergedSampling = OptionsMapper.map(options)
            mergedParams = nil
        } else {
            let p = card.params
            mergedSampling = SamplingParameters(
                temperature: Double(p.temperature),
                topP: Double(p.topP),
                topK: nil,
                maxTokens: p.maxTokens,
                stop: nil,
                seed: nil
            )
            mergedParams = nil
        }
        
        let req = ChatRequest(
            modelID: card.id,
            prompt: prompt.description,
            responseFormat: responseFormat,
            sampling: mergedSampling,
            schema: schemaNode,
            parameters: mergedParams
        )
        return AsyncStream { continuation in
            let task = Task {
                do {
                    try Task.checkCancellation()
                    
                    let stream = await engine.stream(req)
                    var buffer = ""
                    var emittedToolCalls = false
                    
                    for try await chunk in stream {
                        try Task.checkCancellation()
                        
                        for delta in chunk.deltas {
                            if let text = delta.deltaText, !text.isEmpty {
                                if expectsTool {
                                    buffer += text
                                    let bufferLimitBytes = 2 * 1024 * 1024
                                    if buffer.utf8.count > bufferLimitBytes {
                                        Logger.warning("[MLXLanguageModel] Tool detection buffer exceeded (\(bufferLimitBytes/1024)KB)")
                                        let msg = "[Error] Stream buffer exceeded during tool detection"
                                        continuation.yield(.response(.init(assetIDs: [], segments: [.text(.init(content: msg))])))
                                        continuation.finish()
                                        return
                                    }
                                    
                                    try Task.checkCancellation()
                                    
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
                                _ = reason
                            }
                        }
                    }
                    
                    try Task.checkCancellation()
                    
                    if expectsTool && !emittedToolCalls {
                        if !buffer.isEmpty {
                            continuation.yield(.response(.init(assetIDs: [], segments: [.text(.init(content: buffer))])))
                        }
                    }
                    continuation.finish()
                } catch is CancellationError {
                    // Clean up buffer on cancellation
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
