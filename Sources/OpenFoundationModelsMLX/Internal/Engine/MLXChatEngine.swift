import Foundation

/// Dynamic buffer size adjustment based on JSON complexity and memory pressure
private struct DynamicBufferSizer {
    private let baseSize: Int = 512 * 1024  // 512KB base
    private let minSize: Int = 256 * 1024   // 256KB minimum  
    private let maxSize: Int = 8 * 1024 * 1024  // 8MB maximum
    
    private let schemaComplexity: Int
    private var currentSize: Int
    
    init(schemaKeys: [String]) {
        // Calculate schema complexity based on key count and estimated depth
        let keyCount = schemaKeys.count
        let avgKeyLength = schemaKeys.isEmpty ? 0 : schemaKeys.map(\.count).reduce(0, +) / schemaKeys.count
        
        // Complexity heuristic: more keys and longer names = more complex JSON
        self.schemaComplexity = keyCount + (avgKeyLength / 4)
        
        // Start with base size adjusted for complexity
        let complexityMultiplier = 1.0 + Double(schemaComplexity) / 100.0
        self.currentSize = min(Int(Double(baseSize) * complexityMultiplier), maxSize)
    }
    
    mutating func currentBufferLimit() -> Int {
        return currentSize
    }
    
    mutating func adjustForContent(buffer: String, jsonState: JSONStateMachine) {
        let bufferLength = buffer.utf8.count
        let stackDepth = jsonState.stack.count
        
        // Increase size if we're using most of the current buffer and nesting is deep
        if bufferLength > Int(Double(currentSize) * 0.75) && stackDepth > 3 {
            let growthFactor = 1.0 + Double(stackDepth) / 20.0
            let newSize = min(Int(Double(currentSize) * growthFactor), maxSize)
            
            if newSize > currentSize {
                currentSize = newSize
                Logger.debug("[DynamicBufferSizer] Increased buffer size to \(currentSize/1024)KB (depth: \(stackDepth))")
            }
        }
        
        // Decrease size if buffer usage is consistently low (but keep minimum)
        if bufferLength < currentSize / 4 && currentSize > baseSize {
            currentSize = max(Int(Double(currentSize) * 0.8), minSize)
            Logger.debug("[DynamicBufferSizer] Decreased buffer size to \(currentSize/1024)KB")
        }
    }
    
    mutating func adjustForMemoryPressure() {
        // Simple memory pressure response - reduce to minimum
        if currentSize > minSize {
            currentSize = minSize
            Logger.warning("[DynamicBufferSizer] Memory pressure detected, reduced buffer to \(minSize/1024)KB")
        }
    }
}

// Core engine actor that orchestrates text generation through MLXBackend.
// Handles model lifetime, schema-constrained decoding, snap, retry, and logprobs.
actor MLXChatEngine {
    let modelID: String
    private let backend: MLXBackend

    init(modelID: String) async throws {
        self.modelID = modelID
        self.backend = try await MLXBackend(modelID: modelID)
    }

    func generate(_ req: ChatRequest) async throws -> ChatResponse {
        // Minimize side effects: sampling/prompt remain unchanged on retry.
        // However, only 1 attempt when seed is specified for deterministic generation.
        let hasSeed = (req.sampling.seed != nil)
        let maxTries = max(1, (hasSeed ? 1 : req.policy.retryMaxTries))
        
        // Base temperature for variation strategy
        let baseTemperature = req.sampling.temperature ?? 0.7
        let temperatureIncrement = 0.1  // Increase by 0.1 each retry
        
        let response = try await RetryOrchestrator.run(maxTries: maxTries) { tryIndex in
            // Vary temperature on retry (but not if seed is set for deterministic generation)
            var sampling = req.sampling
            if !hasSeed && tryIndex > 0 {
                // Increase temperature slightly to encourage variation
                let newTemp = min(baseTemperature + Double(tryIndex) * temperatureIncrement, 1.5)
                sampling.temperature = newTemp
                Logger.debug("[MLXChatEngine] Retry \(tryIndex + 1) with temperature: \(newTemp)")
            }
            
            let prompt = req.promptOverride ?? Self.composePrompt(req: req)

            let text: String
            if let schema = req.schema {
                if let p = req.parameters {
                    text = try await backend.generateTextConstrained(prompt: prompt, parameters: p, schema: schema)
                } else {
                    text = try await backend.generateTextConstrained(prompt: prompt, sampling: sampling, schema: schema)
                }
            } else if let p = req.parameters {
                text = try await backend.generateText(prompt: prompt, parameters: p)
            } else {
                text = try await backend.generateText(prompt: prompt, sampling: sampling)
            }
            // Validate when schema requested; throw to trigger retry on failure.
            switch req.responseFormat {
            case .text: break
            case .jsonSchema:
                if let meta = req.schema {
                    let validator = JSONValidator(allowExtraKeys: false, enableSnap: true)
                    if validator.validate(text: text, schema: meta) == false { throw ValidationError.schemaUnsatisfied }
                }
            case .jsonSchemaRef: break
            }

            let choice = ChatChoice(message: .init(role: .assistant, content: text), finishReason: "stop")
            return ChatResponse(choices: [choice], usage: .init(promptTokens: 0, completionTokens: 0), meta: .init(retries: tryIndex))
        }
        return response
    }

    func stream(_ req: ChatRequest) -> AsyncThrowingStream<ChatChunk, Error> {
        // Minimize side effects: only 1 attempt when seed is specified.
        let hasSeed = (req.sampling.seed != nil)
        let maxTries = max(1, (hasSeed ? 1 : req.policy.retryMaxTries))
        let hasSchema: Bool
        let schemaKeys: [String]
        switch req.responseFormat {
        case .text: hasSchema = false; schemaKeys = []
        case .jsonSchema(let schemaJSON):
            hasSchema = true
            schemaKeys = req.schema?.keys ?? Self.schemaMeta(from: schemaJSON).keys
        case .jsonSchemaRef:
            hasSchema = false; schemaKeys = [] // unresolved here
        }

        return AsyncThrowingStream { continuation in
            let mainTask = Task {
                attemptLoop: for tryIndex in 0..<maxTries {
                    // Vary temperature on retry (but not if seed is set for deterministic generation)
                    var sampling = req.sampling
                    if !hasSeed && tryIndex > 0 {
                        // Increase temperature slightly to encourage variation
                        let baseTemp = req.sampling.temperature ?? 0.7
                        let newTemp = min(baseTemp + Double(tryIndex) * 0.1, 1.5)
                        sampling.temperature = newTemp
                        Logger.debug("[MLXChatEngine] Stream retry \(tryIndex + 1) with temperature: \(newTemp)")
                    }
                    
                    let prompt = req.promptOverride ?? Self.composePrompt(req: req)
                    
                    // Create a copy of sampling for the closure to capture
                    let capturedSampling = sampling
                    
                    // Create a cancellable task for stream generation
                    let streamTask = Task { () -> AsyncThrowingStream<String, Error> in
                        if let schema = req.schema {
                            if let p = req.parameters {
                                return await backend.streamTextConstrained(prompt: prompt, parameters: p, schema: schema)
                            } else {
                                return await backend.streamTextConstrained(prompt: prompt, sampling: capturedSampling, schema: schema)
                            }
                        } else if let p = req.parameters {
                            return await backend.streamText(prompt: prompt, parameters: p)
                        } else {
                            return await backend.streamText(prompt: prompt, sampling: capturedSampling)
                        }
                    }
                    
                    // Ensure cleanup on scope exit
                    defer {
                        streamTask.cancel()
                    }
                    
                    let textStream = await streamTask.value

                    // If schema + SCD is enabled, buffer attempt and validate; otherwise pass-through.
                    if hasSchema && !schemaKeys.isEmpty {
                        var bufferSizer = DynamicBufferSizer(schemaKeys: schemaKeys)
                        var tracker = JSONKeyTracker(schemaKeys: schemaKeys)
                        var jsonState = JSONStateMachine()
                        var buffer = ""
                        var aborted = false
                        
                        do {
                            for try await piece in textStream {
                                // Check for task cancellation
                                if Task.isCancelled {
                                    break
                                }
                                
                                buffer += piece
                                
                                // Adjust buffer size dynamically based on content
                                bufferSizer.adjustForContent(buffer: buffer, jsonState: jsonState)
                                let currentBufferLimit = bufferSizer.currentBufferLimit()
                                
                                // Check buffer size limit
                                if buffer.utf8.count > currentBufferLimit {
                                    Logger.warning("[MLXChatEngine] Buffer size exceeded limit (\(currentBufferLimit/1024)KB), aborting")
                                    aborted = true
                                    break
                                }
                                
                                // Update JSON state for early completion detection
                                for char in piece {
                                    jsonState.processCharacter(char)
                                }
                                
                                // Check for JSON error state (early termination)
                                if jsonState.isError() {
                                    Logger.debug("[MLXChatEngine] JSON error state detected, aborting stream")
                                    aborted = true
                                    break
                                }
                                
                                // Check for early completion
                                if jsonState.isComplete() {
                                    Logger.debug("[MLXChatEngine] JSON complete at depth 0, early termination")
                                    break
                                }
                                
                                // Check for violations (increased threshold for stability)
                                tracker.consume(piece)
                                if tracker.violationCount >= 3 {
                                    Logger.debug("[MLXChatEngine] Violation threshold exceeded, aborting stream")
                                    aborted = true
                                    break
                                }
                            }
                        } catch {
                            continuation.finish(throwing: error)
                            return
                        }

                        if aborted {
                            // Stream task will be cancelled by defer block
                            continue attemptLoop
                        } else {
                            // validate complete JSON
                            if case .jsonSchema(let schemaJSON) = req.responseFormat,
                               Self.validateJSON(text: buffer, schema: (req.schema ?? Self.schemaMeta(from: schemaJSON))) == false {
                                // retry
                                continue attemptLoop
                            }
                            // success: yield buffered content as chunks
                            let chunks = Self.chunkify(buffer, size: 512)
                            for c in chunks {
                                continuation.yield(ChatChunk(deltas: [.init(deltaText: c, finishReason: nil)], tryIndex: tryIndex))
                            }
                            continuation.yield(ChatChunk(deltas: [.init(deltaText: nil, finishReason: "stop")], tryIndex: tryIndex))
                            continuation.finish()
                            return
                        }
                    } else {
                        // No schema; stream through.
                        do {
                            for try await piece in textStream {
                                if Task.isCancelled { break }
                                continuation.yield(ChatChunk(deltas: [.init(deltaText: piece, finishReason: nil)], tryIndex: tryIndex))
                            }
                            continuation.yield(ChatChunk(deltas: [.init(deltaText: nil, finishReason: "stop")], tryIndex: tryIndex))
                            continuation.finish()
                            return
                        } catch {
                            continuation.finish(throwing: error)
                            return
                        }
                    }
                }
                // If we reach here, retries exhausted: finish with error.
                continuation.finish(throwing: RetryError.exhausted)
            }
            continuation.onTermination = { _ in
                mainTask.cancel()
            }
        }
    }
}

// MARK: - Validation / Prompt composition helpers

private enum ValidationError: Error { case schemaUnsatisfied }

private extension MLXChatEngine {
    static func composePrompt(req: ChatRequest) -> String {
        let base = req.messages.map { m in
            switch m.role { case .system: return "[SYSTEM]\n\(m.content)"; case .user: return "[USER]\n\(m.content)"; case .assistant: return "[ASSISTANT]\n\(m.content)" }
        }.joined(separator: "\n\n")
        return base
    }

    static func validateJSON(text: String, schema: SchemaMeta) -> Bool {
        let validator = JSONValidator(allowExtraKeys: false, enableSnap: true)
        return validator.validate(text: text, schema: schema)
    }

    static func schemaMeta(from json: String) -> SchemaMeta {
        guard let data = json.data(using: .utf8),
              let dict = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            return SchemaMeta(keys: [], required: [])
        }
        let keys = (dict["properties"] as? [String: Any])?.keys.map { String($0) } ?? []
        let req = (dict["required"] as? [String]) ?? []
        return SchemaMeta(keys: keys, required: req)
    }


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
