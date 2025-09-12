import Foundation
import OpenFoundationModels
import OpenFoundationModelsExtra
import MLXLMCommon

/// Model card for OpenAI's GPT-OSS models
/// Uses the same format as Ollama's implementation
public struct GPTOSSModelCard: ModelCard {
    public let id: String
    
    public init(id: String = "lmstudio-community/gpt-oss-20b-MLX-8bit") {
        self.id = id
    }
    
    public var params: GenerateParameters {
        GenerateParameters(
            maxTokens: 2048,
            temperature: 0.7,
            topP: 0.95
        )
    }
    
    public func prompt(transcript: Transcript, options: GenerationOptions?) -> Prompt {
        let ext = TranscriptAccess.extract(from: transcript)
        let currentDate = ISO8601DateFormatter().string(from: Date())
        let hasTools = !ext.toolDefs.isEmpty
        
        return Prompt {
            // System message
            "<|start|>system<|message|>"
            "You are ChatGPT, a large language model trained by OpenAI.\n"
            "Knowledge cutoff: 2024-06\n"
            "Current date: \(currentDate)\n"
            
            
            
            "<|end|>"
            
            // Developer message with tools and instructions
            if hasTools || ext.systemText != nil {
                "<|start|>developer<|message|>"
                
                // Tool definitions
                if hasTools {
                    "# Tools\n\n"
                    "## functions\n\n"
                    "namespace functions {\n"
                    
                    for tool in ext.toolDefs {
                        if let desc = tool.description {
                            "// \(desc)\n"
                        }
                        "type \(tool.name) = (_: "
                        if let params = tool.parametersJSON {
                            params
                        } else {
                            "{}"
                        }
                        ") => any;\n"
                    }
                    
                    "} // namespace functions\n"
                }
                
                // System instructions
                if let system = ext.systemText {
                    "\n# Instructions\n\n\(system)\n"
                }
                
                "<|end|>"
            }
            
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
            
            // Let the model start its own response
        }
    }
}