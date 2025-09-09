import Foundation

// Core engine actor that orchestrates text generation through MLXBackend.
// Now operates as a simple low-level executor - no prompt rendering, no retries.
actor MLXChatEngine {
    private let backend: MLXBackend
    private let maxBufferSizeKB: Int  // Simple buffer size configuration

    /// Initialize with a pre-configured backend
    /// - Parameters:
    ///   - backend: MLXBackend with model already set
    ///   - maxBufferSizeKB: Maximum buffer size for streaming (default 2048KB)
    init(backend: MLXBackend, maxBufferSizeKB: Int = 2048) {
        self.backend = backend
        self.maxBufferSizeKB = maxBufferSizeKB
    }

    func generate(_ req: ChatRequest) async throws -> ChatResponse {
        // Single execution - no retries, no temperature adjustment
        let prompt = req.prompt  // Use the pre-rendered prompt directly
        
        // Use SamplingParameters, converting from GenerateParameters if needed
        let sampling: SamplingParameters
        if let p = req.parameters {
            // Convert GenerateParameters to SamplingParameters
            sampling = SamplingParameters(
                temperature: Double(p.temperature),
                topP: Double(p.topP),
                topK: nil,  // GenerateParameters doesn't have topK
                maxTokens: p.maxTokens,
                stop: nil,
                seed: nil   // GenerateParameters doesn't have seed
            )
        } else {
            sampling = req.sampling
        }
        
        let text: String
        if let schema = req.schema {
            text = try await backend.generateTextConstrained(prompt: prompt, sampling: sampling, schema: schema)
        } else {
            text = try await backend.generateText(prompt: prompt, sampling: sampling)
        }
        
        // Single validation - throw immediately on failure
        switch req.responseFormat {
        case .text: break
        case .jsonSchema:
            if let meta = req.schema {
                let validator = JSONValidator(allowExtraKeys: false, enableSnap: true)
                if validator.validate(text: text, schema: meta) == false { 
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
        // Single execution stream - no retries
        let hasSchema: Bool
        let schemaKeys: [String]
        switch req.responseFormat {
        case .text: hasSchema = false; schemaKeys = []
        case .jsonSchema:
            hasSchema = true
            schemaKeys = req.schema?.keys ?? []
        case .jsonSchemaRef:
            hasSchema = false; schemaKeys = []
        }

        return AsyncThrowingStream { continuation in
            let mainTask = Task {
                let prompt = req.prompt  // Use the pre-rendered prompt directly
                
                // Create a cancellable task for stream generation
                let streamTask = Task { () -> AsyncThrowingStream<String, Error> in
                    // Convert GenerateParameters to SamplingParameters if needed
                    let sampling: SamplingParameters
                    if let p = req.parameters {
                        sampling = SamplingParameters(
                            temperature: Double(p.temperature),
                            topP: Double(p.topP),
                            topK: nil,  // GenerateParameters doesn't have topK
                            maxTokens: p.maxTokens,
                            stop: nil,
                            seed: nil   // GenerateParameters doesn't have seed
                        )
                    } else {
                        sampling = req.sampling
                    }
                    
                    if let schema = req.schema {
                        return await backend.streamTextConstrained(prompt: prompt, sampling: sampling, schema: schema)
                    } else {
                        return await backend.streamText(prompt: prompt, sampling: sampling)
                    }
                }
                
                // Ensure cleanup on scope exit
                defer {
                    streamTask.cancel()
                }
                
                let textStream = await streamTask.value

                // If schema validation is needed, buffer and validate; otherwise pass-through
                if hasSchema && !schemaKeys.isEmpty {
                    let bufferLimit = maxBufferSizeKB * 1024  // Convert KB to bytes
                    var tracker = JSONKeyTracker(schemaKeys: schemaKeys)
                    let jsonState = JSONStateMachine()
                    var buffer = ""
                    
                    do {
                        for try await piece in textStream {
                            // Check for task cancellation
                            if Task.isCancelled {
                                break
                            }
                            
                            buffer += piece
                            
                            // Check buffer size limit
                            if buffer.utf8.count > bufferLimit {
                                Logger.warning("[MLXChatEngine] Buffer size exceeded limit (\(bufferLimit/1024)KB)")
                                throw ValidationError.bufferLimitExceeded
                            }
                            
                            // Update JSON state for early completion detection
                            for char in piece {
                                jsonState.processCharacter(char)
                            }
                            
                            // Check for JSON error state (early termination)
                            if jsonState.isError() {
                                Logger.debug("[MLXChatEngine] JSON error state detected")
                                throw ValidationError.jsonMalformed
                            }
                            
                            // Check for early completion
                            if jsonState.isComplete() {
                                Logger.debug("[MLXChatEngine] JSON complete at depth 0, early termination")
                                streamTask.cancel()  // Cancel upstream generation immediately
                                break
                            }
                            
                            // Check for violations - fail immediately on violations
                            tracker.consume(piece)
                            if tracker.violationCount >= 3 {
                                Logger.debug("[MLXChatEngine] Violation threshold exceeded")
                                streamTask.cancel()  // Cancel upstream generation immediately
                                throw ValidationError.schemaViolations
                            }
                        }
                    } catch {
                        streamTask.cancel()  // Ensure cancellation on any error
                        continuation.finish(throwing: error)
                        return
                    }

                    // Validate complete JSON - fail immediately if invalid
                    if case .jsonSchema = req.responseFormat,
                       let meta = req.schema {
                        let validator = JSONValidator(allowExtraKeys: false, enableSnap: true)
                        if validator.validate(text: buffer, schema: meta) == false {
                            continuation.finish(throwing: ValidationError.schemaUnsatisfied)
                            return
                        }
                    }
                    
                    // Success: yield buffered content as chunks
                    let chunks = Self.chunkify(buffer, size: 512)
                    for c in chunks {
                        continuation.yield(ChatChunk(deltas: [.init(deltaText: c, finishReason: nil)]))
                    }
                    continuation.yield(ChatChunk(deltas: [.init(deltaText: nil, finishReason: "stop")]))
                    continuation.finish()
                } else {
                    // No schema; stream through directly
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

// MARK: - Validation errors

private enum ValidationError: Error { 
    case schemaUnsatisfied
    case bufferLimitExceeded
    case jsonMalformed
    case schemaViolations
}

// MARK: - Utility helpers

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