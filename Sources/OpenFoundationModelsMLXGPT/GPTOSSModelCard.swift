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

    // Debug flag - can be set via environment variable
    private static let isDebugMode = ProcessInfo.processInfo.environment["DEBUG_MODELCARD"] != nil
    
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
        
        // Build metadata if needed (for future extension)
//        let metadata = parsed.metadata(includeAnalysis: false)
        
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
    
    // MARK: - Processor Control
    
    /// Determine if a LogitProcessor should be active based on generated text
    /// For GPT-OSS, only activate key detection in the final channel
    public func shouldActivateProcessor(_ raw: String, processor: any LogitProcessor) -> Bool {
        // Check if this is a KeyDetectionLogitProcessor
        if processor is KeyDetectionLogitProcessor {
            // Find the last channel marker to determine current channel
            if let lastChannelRange = raw.range(of: "<|channel|>", options: .backwards) {
                let afterChannel = String(raw[lastChannelRange.upperBound...])

                // Check which channel we're in
                if afterChannel.hasPrefix("final") {
                    // We're in the final channel, activate key detection
                    if Self.isDebugMode {
                        print("[GPTOSSModelCard] ✅ In 'final' channel - activating KeyDetectionLogitProcessor")
                    }
                    return true
                } else if afterChannel.hasPrefix("analysis") {
                    // We're in the analysis channel, don't constrain
                    if Self.isDebugMode {
                        print("[GPTOSSModelCard] ⏸️ In 'analysis' channel - skipping constraints")
                    }
                    return false
                } else {
                    // Unknown channel
                    if Self.isDebugMode {
                        let preview = String(afterChannel.prefix(20))
                        print("[GPTOSSModelCard] ❓ Unknown channel content: '\(preview)...'")
                    }
                }
            } else {
                // No channel found yet, don't activate
                if Self.isDebugMode {
                    let preview = String(raw.suffix(50))
                    print("[GPTOSSModelCard] ⏳ No channel marker found yet. Last 50 chars: '\(preview)'")
                }
            }

            return false
        }
        
        // For other processors, always activate
        return true
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
