import Foundation
import OpenFoundationModels
import OpenFoundationModelsExtra
import OpenFoundationModelsMLX
import MLXLMCommon

/// ModelCard implementation for Google's FunctionGemma models
/// Specialized for function calling with structured output format
///
/// FunctionGemma uses specific control tokens:
/// - `<start_function_declaration>` / `<end_function_declaration>` for tool definitions
/// - `<start_function_call>` / `<end_function_call>` for model's tool requests
/// - `<start_function_response>` / `<end_function_response>` for tool results
/// - `<escape>` token wraps all string values
///
/// Reference: https://ai.google.dev/gemma/docs/functiongemma/formatting-and-best-practices
public struct FunctionGemmaModelCard: ModelCard {
    public let id: String

    public init(id: String = "mlx-community/functiongemma-270m-it-bf16") {
        self.id = id
    }

    public var params: GenerateParameters {
        GenerateParameters(
            maxTokens: 256,
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

    // MARK: - Prompt Generation

    public func prompt(transcript: Transcript, options: GenerationOptions?) -> Prompt {
        let ext = TranscriptAccess.extract(from: transcript)
        let messages = ext.messages.filter { $0.role != .system }

        return Prompt {
            // Developer turn - required system message and function declarations
            "<start_of_turn>developer\n"

            // Required system message for function calling capability
            "You are a model that can do function calling with the following functions"

            // Function declarations using FunctionGemma format
            for tool in ext.toolDefs {
                formatFunctionDeclaration(tool)
            }

            // Additional system instructions
            if let system = ext.systemText {
                "\n\n"
                system
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
                    // Tool results use function response format
                    if let toolName = message.toolName {
                        formatFunctionResponse(toolName: toolName, result: message.content)
                    }

                default:
                    ""
                }
            }

            // Generation prompt
            "<start_of_turn>model\n"
        }
    }

    // MARK: - Function Declaration Formatting

    /// Format a tool definition using FunctionGemma's declaration syntax
    /// Format: <start_function_declaration>declaration:NAME{description:<escape>DESC<escape>,parameters:{...}}<end_function_declaration>
    private func formatFunctionDeclaration(_ tool: (name: String, description: String?, parametersJSON: String?)) -> String {
        var declaration = "<start_function_declaration>declaration:\(tool.name){"

        // Description
        if let description = tool.description {
            declaration += "description:<escape>\(description)<escape>"
        }

        // Parameters
        if let parametersJSON = tool.parametersJSON {
            if tool.description != nil {
                declaration += ","
            }
            // Convert JSON to FunctionGemma format (escape string values)
            let formattedParams = formatParametersForGemma(parametersJSON)
            declaration += "parameters:\(formattedParams)"
        } else {
            // Empty parameters
            if tool.description != nil {
                declaration += ","
            }
            declaration += "parameters:{properties:{},required:[],type:<escape>object<escape>}"
        }

        declaration += "}<end_function_declaration>"
        return declaration
    }

    /// Convert JSON parameters to FunctionGemma format
    /// Wraps string values with <escape> tokens
    private func formatParametersForGemma(_ json: String) -> String {
        // Parse and reformat the JSON with escape tokens
        // For simplicity, we pass through as-is but wrap the type field
        // A more complete implementation would parse and transform the entire structure
        var result = json
            .replacingOccurrences(of: "\"type\":", with: "type:")
            .replacingOccurrences(of: "\"object\"", with: "<escape>object<escape>")
            .replacingOccurrences(of: "\"string\"", with: "<escape>string<escape>")
            .replacingOccurrences(of: "\"number\"", with: "<escape>number<escape>")
            .replacingOccurrences(of: "\"integer\"", with: "<escape>integer<escape>")
            .replacingOccurrences(of: "\"boolean\"", with: "<escape>boolean<escape>")
            .replacingOccurrences(of: "\"array\"", with: "<escape>array<escape>")

        // Handle description fields
        result = result.replacingOccurrences(of: "\"description\":", with: "description:")

        // Handle properties and required
        result = result.replacingOccurrences(of: "\"properties\":", with: "properties:")
        result = result.replacingOccurrences(of: "\"required\":", with: "required:")

        return result
    }

    /// Format a function response for tool results
    /// Format: <start_function_response>response:NAME{result:<escape>VALUE<escape>}<end_function_response>
    private func formatFunctionResponse(toolName: String, result: String) -> String {
        return "<start_function_response>response:\(toolName){result:<escape>\(result)<escape>}<end_function_response>\n"
    }

    // MARK: - Output Processing

    /// Process raw output to extract function calls or text
    public func generate(from raw: String, options: GenerationOptions?) -> Transcript.Entry {
        // Check for function call
        if let functionCall = FunctionGemmaParser.parseFunctionCall(raw) {
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
            if !cleaned.contains("<end_function_call>") {
                cleaned = String(cleaned[..<range.lowerBound])
            }
        }

        // Remove any remaining special tokens
        cleaned = cleaned
            .replacingOccurrences(of: "<escape>", with: "")
            .replacingOccurrences(of: "<start_function_declaration>", with: "")
            .replacingOccurrences(of: "<end_function_declaration>", with: "")
            .replacingOccurrences(of: "<start_function_response>", with: "")
            .replacingOccurrences(of: "<end_function_response>", with: "")
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
