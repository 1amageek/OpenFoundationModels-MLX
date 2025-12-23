import Foundation
import OpenFoundationModels
import OpenFoundationModelsExtra
import OpenFoundationModelsMLX
import MLXLMCommon

/// ModelCard implementation for Google's FunctionGemma models
/// Specialized for function calling with structured output format
public struct FunctionGemmaModelCard: ModelCard {
    public let id: String

    public init(id: String = "mlx-community/functiongemma-270m-it-bf16") {
        self.id = id
    }

    public var params: GenerateParameters {
        GenerateParameters(
            maxTokens: 128,
            temperature: 0.7,
            topP: 0.9
        )
    }

    /// Stop tokens for FunctionGemma
    /// Generation should stop when any of these tokens are produced
    public var stopTokens: Set<String> {
        [
            "<end_function_call>",  // End of function call
            "<end_of_turn>",        // End of model turn
            "<eos>"                 // End of sequence
        ]
    }

    public func prompt(transcript: Transcript, options: GenerationOptions?) -> Prompt {
        let ext = TranscriptAccess.extract(from: transcript)
        let messages = ext.messages.filter { $0.role != .system }

        return Prompt {
            // Developer turn - system instructions and tool definitions
            "<start_of_turn>user\n"

            // System instructions
            if let system = ext.systemText {
                system
                "\n\n"
            }

            // Tool definitions in JSON Schema format (FunctionGemma spec)
            if !ext.toolDefs.isEmpty {
                "You have access to the following tools:\n\n"

                for tool in ext.toolDefs {
                    // Format as JSON Schema (FunctionGemma expected format)
                    "{\n"
                    "  \"type\": \"function\",\n"
                    "  \"function\": {\n"
                    "    \"name\": \"\(tool.name)\""

                    if let description = tool.description {
                        ",\n    \"description\": \"\(escapeJSON(description))\""
                    }

                    if let parametersJSON = tool.parametersJSON {
                        ",\n    \"parameters\": \(parametersJSON)"
                    }

                    "\n  }\n"
                    "}\n\n"
                }

                "When you need to call a function, respond with:\n"
                "<start_function_call>call:function_name{param:<escape>value<escape>}<end_function_call>\n\n"
            }

            // Response schema (if provided)
            if let schemaJSON = ext.schemaJSON {
                "You must respond with JSON matching this schema:\n"
                schemaJSON
                "\n\n"
            }

            "<end_of_turn>\n"

            // Conversation history
            for message in messages {
                switch message.role {
                case .user:
                    "<start_of_turn>user\n"
                    message.content
                    "<end_of_turn>\n"

                case .assistant:
                    "<start_of_turn>model\n"
                    message.content
                    "<end_of_turn>\n"

                case .tool:
                    // Tool results go in user turn
                    "<start_of_turn>user\n"
                    if let toolName = message.toolName {
                        "[Result from \(toolName)]: "
                    }
                    message.content
                    "<end_of_turn>\n"

                default:
                    ""
                }
            }

            // Generation prompt
            "<start_of_turn>model\n"
        }
    }

    /// Escape special characters for JSON string
    private func escapeJSON(_ string: String) -> String {
        string
            .replacingOccurrences(of: "\\", with: "\\\\")
            .replacingOccurrences(of: "\"", with: "\\\"")
            .replacingOccurrences(of: "\n", with: "\\n")
            .replacingOccurrences(of: "\r", with: "\\r")
            .replacingOccurrences(of: "\t", with: "\\t")
    }

    // MARK: - Output Processing

    /// Process raw output to extract function calls or text
    public func generate(from raw: String, options: GenerationOptions?) -> Transcript.Entry {
        // Check for function call
        if let functionCall = FunctionGemmaParser.parseFunctionCall(raw) {
            // Create ToolCall with proper API
            if let toolCallsEntry = createToolCallsEntry(from: functionCall) {
                return toolCallsEntry
            }
        }

        // Clean up output (remove end tokens and incomplete function call tokens)
        var cleaned = raw
            .replacingOccurrences(of: "<end_of_turn>", with: "")
            .replacingOccurrences(of: "<eos>", with: "")

        // Remove incomplete <start_function_call> tokens (not followed by <end_function_call>)
        if let range = cleaned.range(of: "<start_function_call>") {
            // Check if there's a complete function call
            if !cleaned.contains("<end_function_call>") {
                // Truncate at the start of incomplete function call
                cleaned = String(cleaned[..<range.lowerBound])
            }
        }

        // Remove any remaining special tokens
        cleaned = cleaned
            .replacingOccurrences(of: "<escape>", with: "")
            .trimmingCharacters(in: .whitespacesAndNewlines)

        return .response(.init(
            assetIDs: [],
            segments: [.text(.init(content: cleaned))]
        ))
    }

    /// Stream processing for FunctionGemma output
    public func stream(
        from chunks: AsyncThrowingStream<String, Error>,
        options: GenerationOptions?
    ) -> AsyncThrowingStream<Transcript.Entry, Error> {
        return AsyncThrowingStream { continuation in
            Task {
                var buffer = ""
                var pendingFunctionCall = false

                do {
                    for try await chunk in chunks {
                        buffer += chunk

                        // Check if we have a complete function call
                        if buffer.contains("<end_function_call>") {
                            if let functionCall = FunctionGemmaParser.parseFunctionCall(buffer),
                               let toolCallsEntry = createToolCallsEntry(from: functionCall) {
                                continuation.yield(toolCallsEntry)
                                continuation.finish()
                                return
                            }
                        }

                        // Check if we're in the middle of a function call
                        if buffer.contains("<start_function_call>") {
                            pendingFunctionCall = true
                        }

                        // Don't stream if we're building a function call
                        if pendingFunctionCall {
                            continue
                        }

                        // Stream text content (filtering out special tokens)
                        let cleanChunk = chunk
                            .replacingOccurrences(of: "<end_of_turn>", with: "")
                            .replacingOccurrences(of: "<eos>", with: "")
                            .replacingOccurrences(of: "<escape>", with: "")

                        if !cleanChunk.isEmpty {
                            continuation.yield(.response(.init(
                                assetIDs: [],
                                segments: [.text(.init(content: cleanChunk))]
                            )))
                        }
                    }

                    // Final check for function call in buffer
                    if let functionCall = FunctionGemmaParser.parseFunctionCall(buffer),
                       let toolCallsEntry = createToolCallsEntry(from: functionCall) {
                        continuation.yield(toolCallsEntry)
                    } else if pendingFunctionCall {
                        // Incomplete function call - extract text before it
                        var cleaned = buffer
                            .replacingOccurrences(of: "<end_of_turn>", with: "")
                            .replacingOccurrences(of: "<eos>", with: "")

                        if let range = cleaned.range(of: "<start_function_call>") {
                            cleaned = String(cleaned[..<range.lowerBound])
                        }

                        cleaned = cleaned
                            .replacingOccurrences(of: "<escape>", with: "")
                            .trimmingCharacters(in: .whitespacesAndNewlines)

                        if !cleaned.isEmpty {
                            continuation.yield(.response(.init(
                                assetIDs: [],
                                segments: [.text(.init(content: cleaned))]
                            )))
                        }
                    }

                    continuation.finish()
                } catch {
                    continuation.finish(throwing: error)
                }
            }
        }
    }

    // MARK: - Private Helpers

    /// Create a toolCalls entry from a parsed function call
    private func createToolCallsEntry(from functionCall: FunctionGemmaParser.FunctionCall) -> Transcript.Entry? {
        do {
            let arguments = try GeneratedContent(json: functionCall.arguments)
            let toolCall = Transcript.ToolCall(
                id: UUID().uuidString,
                toolName: functionCall.name,
                arguments: arguments
            )
            let toolCalls = Transcript.ToolCalls(id: UUID().uuidString, [toolCall])
            return .toolCalls(toolCalls)
        } catch {
            return nil
        }
    }
}
