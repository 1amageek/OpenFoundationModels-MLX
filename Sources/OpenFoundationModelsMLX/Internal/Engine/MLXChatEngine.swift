import Foundation

// Core engine actor that orchestrates text generation through MLXBackend.
// Now operates as a simple low-level executor - no prompt rendering, no retries.
actor MLXChatEngine {
    private let backend: MLXBackend
    private let maxBufferSizeKB: Int  // Simple buffer size configuration

    init(backend: MLXBackend, maxBufferSizeKB: Int = 2048) {
        self.backend = backend
        self.maxBufferSizeKB = maxBufferSizeKB
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
        
        let text: String
        if let schemaNode = req.schema {
            text = try await backend.generateTextWithSchema(prompt: prompt, sampling: sampling, schema: schemaNode)
        } else {
            text = try await backend.generateText(prompt: prompt, sampling: sampling)
        }
        
        switch req.responseFormat {
        case .text: break
        case .jsonSchema:
            if let schemaNode = req.schema {
                if JSONValidator.validate(text: text, schema: schemaNode) == false { 
                    throw ValidationError.schemaUnsatisfied 
                }
            }
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
        let hasSchema: Bool
        let schemaKeys: [String]
        switch req.responseFormat {
        case .text: hasSchema = false; schemaKeys = []
        case .jsonSchema:
            hasSchema = true
            schemaKeys = req.schema?.objectKeys ?? []
        case .jsonSchemaRef:
            hasSchema = false; schemaKeys = []
        }

        return AsyncThrowingStream { continuation in
            let mainTask = Task {
                let prompt = req.prompt
                
                // Create a cancellable task for stream generation
                let streamTask = Task { () -> AsyncThrowingStream<String, Error> in
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
                    
                    if let schema = req.schema {
                        return await backend.streamTextWithSchema(prompt: prompt, sampling: sampling, schema: schema)
                    } else {
                        return await backend.streamText(prompt: prompt, sampling: sampling)
                    }
                }
                
                defer {
                    streamTask.cancel()
                }
                
                let textStream = await streamTask.value

                if hasSchema && !schemaKeys.isEmpty {
                    let bufferLimit = maxBufferSizeKB * 1024  // Convert KB to bytes
                    var tracker = JSONKeyTracker(schemaKeys: schemaKeys)
                    let jsonState = JSONStateMachine()
                    var buffer = ""
                    
                    do {
                        for try await piece in textStream {
                            if Task.isCancelled {
                                break
                            }
                            
                            buffer += piece
                            
                            if buffer.utf8.count > bufferLimit {
                                Logger.warning("[MLXChatEngine] Buffer size exceeded limit (\(bufferLimit/1024)KB)")
                                throw ValidationError.bufferLimitExceeded
                            }
                            
                            for char in piece {
                                jsonState.processCharacter(char)
                            }
                            
                            if jsonState.isError() {
                                Logger.debug("[MLXChatEngine] JSON error state detected")
                                throw ValidationError.jsonMalformed
                            }
                            
                            if jsonState.isComplete() {
                                Logger.debug("[MLXChatEngine] JSON complete at depth 0, early termination")
                                streamTask.cancel()
                                break
                            }
                            
                            tracker.consume(piece)
                            if tracker.violationCount >= 3 {
                                Logger.debug("[MLXChatEngine] Violation threshold exceeded")
                                streamTask.cancel()
                                throw ValidationError.schemaViolations
                            }
                        }
                    } catch {
                        streamTask.cancel()
                        continuation.finish(throwing: error)
                        return
                    }

                    if case .jsonSchema = req.responseFormat,
                       let schemaNode = req.schema {
                        if JSONValidator.validate(text: buffer, schema: schemaNode) == false {
                            continuation.finish(throwing: ValidationError.schemaUnsatisfied)
                            return
                        }
                    }
                    
                    let chunks = Self.chunkify(buffer, size: 512)
                    for c in chunks {
                        continuation.yield(ChatChunk(deltas: [.init(deltaText: c, finishReason: nil)]))
                    }
                    continuation.yield(ChatChunk(deltas: [.init(deltaText: nil, finishReason: "stop")]))
                    continuation.finish()
                } else {
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
            }
            continuation.onTermination = { _ in
                mainTask.cancel()
            }
        }
    }
}


private enum ValidationError: Error { 
    case schemaUnsatisfied
    case bufferLimitExceeded
    case jsonMalformed
    case schemaViolations
}


private extension MLXChatEngine {
    static func chunkify(_ s: String, size: Int) -> [String] {
        guard size > 0, s.count > size else { return [s] }
        var res: [String] = []
        var idx = s.startIndex
        while idx < s.endIndex {
            let next = s.index(idx, offsetBy: size, limitedBy: s.endIndex) ?? s.endIndex
            res.append(String(s[idx..<next]))
            idx = next
        }
        return res
    }
}