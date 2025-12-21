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

    public func prompt(transcript: Transcript, options: GenerationOptions?) -> Prompt {
        let ext = TranscriptAccess.extract(from: transcript)
        let messages = ext.messages.filter { $0.role != .system }

        return Prompt {
            // Developer role (system message) - required for function calling
            "<start_of_turn>user\n"

            // System instructions
            if let system = ext.systemText {
                system
                "\n\n"
            } else {
                "You are a helpful assistant that can call functions.\n\n"
            }

            // Tool definitions
            if !ext.toolDefs.isEmpty {
                "You have access to the following functions:\n\n"

                for tool in ext.toolDefs {
                    "Function: \(tool.name)\n"

                    if let description = tool.description {
                        "Description: \(description)\n"
                    }

                    if let parametersJSON = tool.parametersJSON {
                        "Parameters: \(parametersJSON)\n"
                    }

                    "\n"
                }

                "To call a function, use this format:\n"
                "<start_function_call>call:function_name{param:<escape>value<escape>}<end_function_call>\n\n"
            }

            // Response schema (if provided)
            if let schemaJSON = ext.schemaJSON {
                "You must respond with JSON matching this schema:\n"
                schemaJSON
                "\n\n"
            }

            // User messages
            for message in messages {
                switch message.role {
                case .user:
                    message.content
                    "\n"

                case .assistant:
                    "<end_of_turn>\n"
                    "<start_of_turn>model\n"
                    message.content
                    "<end_of_turn>\n"
                    "<start_of_turn>user\n"

                case .tool:
                    if let toolName = message.toolName {
                        "[Function result from \(toolName)]: "
                    }
                    message.content
                    "\n"

                default:
                    ""
                }
            }

            "<end_of_turn>\n"
            "<start_of_turn>model\n"
        }
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

        // Clean up output (remove end tokens)
        let cleaned = raw
            .replacingOccurrences(of: "<end_of_turn>", with: "")
            .replacingOccurrences(of: "<eos>", with: "")
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

                        // Stream text content (filtering out special tokens)
                        let cleanChunk = chunk
                            .replacingOccurrences(of: "<end_of_turn>", with: "")
                            .replacingOccurrences(of: "<eos>", with: "")

                        if !cleanChunk.isEmpty && !cleanChunk.hasPrefix("<start_function_call>") {
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
