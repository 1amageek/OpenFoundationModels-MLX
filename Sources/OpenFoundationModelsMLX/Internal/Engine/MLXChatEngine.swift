import Foundation

// Placeholder actor for the core engine. In future iterations this will host
// model lifetime, schema-constrained decoding, snap, retry, and logprobs.
actor MLXChatEngine {
    let modelID: String
    #if canImport(MLXLLM)
    private let backend: MLXBackend
    #endif

    init(modelID: String) async throws {
        self.modelID = modelID
        #if canImport(MLXLLM)
        self.backend = try await MLXBackend(modelID: modelID)
        #endif
    }

    func generate(_ req: ChatRequest) async throws -> ChatResponse {
        // 副作用最小化: リトライでもサンプリング/プロンプトは不変。
        // ただし seed 指定時は決定的生成のため 1 回のみ。
        let maxTries = max(1, (req.sampling.seed != nil ? 1 : req.policy.retryMaxTries))
        let response = try await RetryOrchestrator.run(maxTries: maxTries) { tryIndex in
            let sampling = req.sampling
            let prompt = Self.composePrompt(req: req)

            #if canImport(MLXLLM)
            let text = try await backend.generateText(prompt: prompt, sampling: sampling)
            #else
            let text = "[MLXChatEngine stub] " + prompt
            #endif
            // Validate when schema requested; throw to trigger retry on failure.
            switch req.responseFormat {
            case .text:
                break
            case .jsonSchema(let schemaJSON):
                let meta = req.schema ?? Self.schemaMeta(from: schemaJSON)
                if Self.validateJSON(text: text, schema: meta) == false {
                    throw ValidationError.schemaUnsatisfied
                }
            case .jsonSchemaRef:
                // Without resolved schema JSON, skip strict validation here.
                break
            }

            let choice = ChatChoice(message: .init(role: .assistant, content: text), finishReason: "stop")
            return ChatResponse(choices: [choice], usage: .init(promptTokens: 0, completionTokens: 0), meta: .init(retries: tryIndex))
        }
        return response
    }

    func stream(_ req: ChatRequest) -> AsyncThrowingStream<ChatChunk, Error> {
        // 副作用最小化: seed 指定時は 1 回のみ。
        let maxTries = max(1, (req.sampling.seed != nil ? 1 : req.policy.retryMaxTries))
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
            Task {
                attemptLoop: for tryIndex in 0..<maxTries {
                    // リトライでも sampling/prompt は不変
                    let sampling = req.sampling
                    let prompt = Self.composePrompt(req: req)
                    
                    // Create cancellable task for generation
                    let generationTask = Task {
                        #if canImport(MLXLLM)
                        if let schema = req.schema, req.policy.enableSCD {
                            return await backend.streamTextConstrained(prompt: prompt, sampling: sampling, schema: schema)
                        } else {
                            return await backend.streamText(prompt: prompt, sampling: sampling)
                        }
                        #else
                        return AsyncThrowingStream<String, Error> { inner in
                            let text = "[MLXChatEngine stub] " + prompt
                            inner.yield(text)
                            inner.finish()
                        }
                        #endif
                    }
                    
                    let textStream = await generationTask.value

                    // If schema + SCD is enabled, buffer attempt and validate; otherwise pass-through.
                    if req.policy.enableSCD && hasSchema && !schemaKeys.isEmpty {
                        var tracker = JSONKeyTracker(schemaKeys: schemaKeys)
                        var buffer = ""
                        var aborted = false
                        do {
                            for try await piece in textStream {
                                // Check for task cancellation
                                if Task.isCancelled {
                                    break
                                }
                                buffer += piece
                                tracker.consume(piece)
                                if tracker.violationCount >= 2 {
                                    aborted = true
                                    break
                                }
                            }
                        } catch {
                            generationTask.cancel()
                            continuation.finish(throwing: error)
                            return
                        }

                        if aborted {
                            // Cancel the generation task before retrying
                            generationTask.cancel()
                            // try again
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
                // If we reach here, retries exhausted.
                continuation.finish()
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
        guard let obj = firstJSONObject(in: text) else { return false }
        // Snap keys and re-check required set.
        let snapped = snapObject(obj, schemaKeys: schema.keys)
        for req in schema.required {
            if snapped[req] == nil { return false }
        }
        return true
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

    static func firstJSONObject(in text: String) -> [String: Any]? { JSONUtils.firstTopLevelObject(in: text) }

    static func snapObject(_ obj: [String: Any], schemaKeys: [String]) -> [String: Any] {
        var out: [String: Any] = [:]
        for (k, v) in obj { out[SchemaSnapParser.snapKey(k, against: schemaKeys) ?? k] = v }
        return out
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
