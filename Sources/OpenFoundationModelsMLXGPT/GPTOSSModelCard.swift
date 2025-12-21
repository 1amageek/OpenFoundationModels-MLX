import Foundation
import OpenFoundationModels
import OpenFoundationModelsExtra
import OpenFoundationModelsMLX
import MLXLMCommon
import MLXLLM

/// Model card for OpenAI's GPT-OSS models
/// Uses the same format as Ollama's implementation
public struct GPTOSSModelCard: ModelCard {
    public let id: String

    public init(id: String = "lmstudio-community/gpt-oss-20b-MLX-8bit") {
        self.id = id
    }

    public var params: GenerateParameters {
        GenerateParameters(
            maxTokens: 8192,
            temperature: 0.7,
            topP: 0.95
        )
    }

    // MARK: - Output Processing

    /// Process Harmony format output to extract the final channel
    public func generate(from raw: String, options: GenerationOptions?) -> Transcript.Entry {
        let parsed = HarmonyParser.parse(raw)

        return .response(.init(
            assetIDs: [],
            segments: [.text(.init(content: parsed.displayContent))]
        ))
    }

    /// Stream Harmony format output, filtering to only final channel content
    public func stream(
        from chunks: AsyncThrowingStream<String, Error>,
        options: GenerationOptions?
    ) -> AsyncThrowingStream<Transcript.Entry, Error> {
        return AsyncThrowingStream { continuation in
            Task {
                var streamState = HarmonyParser.StreamState()

                do {
                    for try await chunk in chunks {
                        // Process chunk through Harmony parser
                        if let finalContent = streamState.processChunk(chunk) {
                            // Stream only final channel content
                            let entry = Transcript.Entry.response(.init(
                                assetIDs: [],
                                segments: [.text(.init(content: finalContent))]
                            ))
                            continuation.yield(entry)
                        }
                    }

                    // Flush any remaining content
                    if let remaining = streamState.flush() {
                        let entry = Transcript.Entry.response(.init(
                            assetIDs: [],
                            segments: [.text(.init(content: remaining))]
                        ))
                        continuation.yield(entry)
                    }

                    continuation.finish()
                } catch {
                    continuation.finish(throwing: error)
                }
            }
        }
    }

    public func prompt(transcript: Transcript, options: GenerationOptions?) -> Prompt {
        let ext = TranscriptAccess.extract(from: transcript)
        let currentDate = ISO8601DateFormatter().string(from: Date())
        let hasTools = !ext.toolDefs.isEmpty

        return Prompt {
            // System message
            "<|start|>system<|message|>"
            "You are ChatGPT, a large language model trained by OpenAI."
            "Please perform the task exactly as instructed by the user."
            "Knowledge cutoff: 2024-06\n"
            "Current date: \(currentDate)\n"

            "<|end|>"

            "<|start|>developer<|message|>"
            "You must respond ONLY with a JSON object that matches the schema defined below."
            "Do NOT output the schema itself."
            "Do NOT include explanations, commentary, or extra keys."
            "Respond only with valid JSON data."

            // Developer message with tools and instructions
            if hasTools || ext.systemText != nil {

                // Tool definitions
                if hasTools {
                    "# Tools"
                    "## functions"
                    "namespace functions {"

                    for tool in ext.toolDefs {
                        if let desc = tool.description {
                            "// \(desc)"
                        }
                        "type \(tool.name) = (_: "
                        if let params = tool.parametersJSON {
                            params
                        } else {
                            "{}"
                        }
                        ") => any;"
                    }

                    "} // namespace functions"
                }

                // System instructions
                if let system = ext.systemText {
                    "\n# Instructions\n\n\(system)"
                }

                // Response format schema (Harmony spec)
                if let schemaJSON = ext.schemaJSON, !schemaJSON.isEmpty {
                    "\n# Response Formats"
                    "Generate a JSON data instance that conforms to this schema. Response is only included JSON."
                    "DO NOT copy the schema structure - create actual data values.\n"
                    "Example: for {\"name\": {\"type\": \"string\"}}, generate {\"name\": \"John Doe\"}\n"
                    "## response_format"
                    schemaJSON
                }
            }

            "<|end|>"

            // Message history
            for message in ext.messages.filter({ $0.role != .system }) {
                switch message.role {
                case .user:
                    "<|start|>user<|message|>\(message.content)<|end|>"

                case .assistant:
                    "<|start|>assistant<|message|>\(message.content)<|end|>"

                case .tool:
                    if let toolName = message.toolName {
                        "<|start|>functions.\(toolName) to=assistant<|message|>\(message.content)<|end|>"
                    } else {
                        "<|start|>tool to=assistant<|message|>\(message.content)<|end|>"
                    }

                default:
                    ""
                }
            }

            // Start assistant response
            // The model will choose the appropriate channel
            "<|start|>assistant"
        }
    }
}
