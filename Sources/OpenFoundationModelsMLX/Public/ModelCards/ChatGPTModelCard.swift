import Foundation
import OpenFoundationModels
import OpenFoundationModelsExtra
import MLXLMCommon

/// ModelCard implementation for ChatGPT/GPT-4 format
public struct ChatGPTModelCard: ModelCard {
    public let id: String
    
    public init(id: String = "gpt-4") {
        self.id = id
    }
    
    public var params: GenerateParameters {
        GenerateParameters(
            maxTokens: 4096,
            temperature: 0.7,
            topP: 1.0
        )
    }
    
    public func prompt(transcript: Transcript, options: GenerationOptions?) -> Prompt {
        // Extract necessary data
        let ext = TranscriptAccess.extract(from: transcript)
        let currentDate = ISO8601DateFormatter().string(from: Date())
        // TODO: Add thinking mode support when GenerationOptions supports metadata
        let thinking = false
        let hasTools = !ext.toolDefs.isEmpty
        
        return Prompt {
            // System message
            "<|start|>system<|message|>"
            "You are ChatGPT, a large language model trained by OpenAI.\n"
            "Knowledge cutoff: 2024-06\n"
            "Current date: \(currentDate)"
            
            // Thinking mode
            if thinking {
                "\n\nReasoning: medium"
            }
            
            // Channel definitions
            if thinking || hasTools {
                "\n\n# Valid channels: "
                if thinking { "analysis, " }
                if hasTools { "commentary, " }
                "final. Channel must be included for every message."
            }
            
            "<|end|>"
            
            // Developer message (tools and system instructions)
            if hasTools || ext.systemText != nil {
                "\n<|start|>developer<|message|>"
                
                // Tool definitions
                if hasTools {
                    "# Tools\n\nnamespace functions {\n"
                    for tool in ext.toolDefs {
                        if let desc = tool.description {
                            "// \(desc)\n"
                        }
                        "type \(tool.name) = ("
                        if let params = tool.parametersJSON {
                            "_: \(params)"
                        }
                        ") => any;\n"
                    }
                    "} // namespace functions\n"
                }
                
                // System instructions
                if let system = ext.systemText {
                    "\n# Instructions\n\n\(system)"
                }
                
                "<|end|>"
            }
            
            // Message history
            for message in ext.messages.filter({ $0.role != .system }) {
                switch message.role {
                case .user:
                    "\n<|start|>user<|message|>\(message.content)<|end|>"
                    
                case .assistant:
                    "\n<|start|>assistant"
                    if thinking {
                        "<|channel|>analysis"
                    }
                    "<|message|>\(message.content)<|end|>"
                    
                case .tool:
                    "\n<|start|>functions.\(message.toolName ?? "unknown") to=assistant"
                    "<|message|>\(message.content)<|end|>"
                    
                default:
                    ""
                }
            }
            
            // Start assistant
            "\n<|start|>assistant"
            if thinking {
                "<|channel|>analysis"
            }
        }
    }
}