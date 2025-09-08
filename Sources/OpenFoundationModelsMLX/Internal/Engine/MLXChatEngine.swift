import Foundation
import PRECISE
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
// Now operates as a simple low-level executor - no prompt rendering, no retries.
actor MLXChatEngine {
    let modelID: String
    private let backend: MLXBackend

    init(modelID: String) async throws {
        self.modelID = modelID
        self.backend = try await MLXBackend(modelID: modelID)
    }

    func generate(_ req: ChatRequest) async throws -> ChatResponse {
        // Single execution - no retries, no temperature adjustment
        let prompt = req.prompt  // Use the pre-rendered prompt directly
        
        let text: String
        if let schema = req.schema {
            if let p = req.parameters {
                text = try await backend.generateTextConstrained(prompt: prompt, parameters: p, schema: schema)
            } else {
                text = try await backend.generateTextConstrained(prompt: prompt, sampling: req.sampling, schema: schema)
            }
        } else if let p = req.parameters {
            text = try await backend.generateText(prompt: prompt, parameters: p)
        } else {
            text = try await backend.generateText(prompt: prompt, sampling: req.sampling)
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
                    if let schema = req.schema {
                        if let p = req.parameters {
                            return await backend.streamTextConstrained(prompt: prompt, parameters: p, schema: schema)
                        } else {
                            return await backend.streamTextConstrained(prompt: prompt, sampling: req.sampling, schema: schema)
                        }
                    } else if let p = req.parameters {
                        return await backend.streamText(prompt: prompt, parameters: p)
                    } else {
                        return await backend.streamText(prompt: prompt, sampling: req.sampling)
                    }
                }
                
                // Ensure cleanup on scope exit
                defer {
                    streamTask.cancel()
                }
                
                let textStream = await streamTask.value

                // If schema validation is needed, buffer and validate; otherwise pass-through
                if hasSchema && !schemaKeys.isEmpty {
                    var bufferSizer = DynamicBufferSizer(schemaKeys: schemaKeys)
                    var tracker = JSONKeyTracker(schemaKeys: schemaKeys)
                    var jsonState = JSONStateMachine()
                    var buffer = ""
                    
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
                                Logger.warning("[MLXChatEngine] Buffer size exceeded limit (\(currentBufferLimit/1024)KB)")
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
                                break
                            }
                            
                            // Check for violations - fail immediately on violations
                            tracker.consume(piece)
                            if tracker.violationCount >= 3 {
                                Logger.debug("[MLXChatEngine] Violation threshold exceeded")
                                throw ValidationError.schemaViolations
                            }
                        }
                    } catch {
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