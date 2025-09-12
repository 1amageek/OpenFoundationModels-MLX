import Foundation
import OpenFoundationModels
import OpenFoundationModelsExtra
import MLXLMCommon

/// ModelCard implementation for Llama 2 format
public struct LlamaModelCard: ModelCard {
    public let id: String
    
    public init(id: String) {
        self.id = id
    }
    
    public var params: GenerateParameters {
        GenerateParameters(
            maxTokens: 2048,
            temperature: 0.7,
            topP: 0.9,
            repetitionPenalty: 1.1
        )
    }
    
    public func prompt(transcript: Transcript, options: GenerationOptions?) -> Prompt {
        // Extract necessary data
        let ext = TranscriptAccess.extract(from: transcript)
        let currentDate = ISO8601DateFormatter().string(from: Date())
        let messages = ext.messages.filter { $0.role != .system }
        
        return Prompt {
            "<s>[INST] "
            
            // System message
            if ext.systemText != nil || ext.schemaJSON != nil {
                "<<SYS>>\n"
                if let system = ext.systemText {
                    system
                    "\n"
                }
                "Current date: \(currentDate)"
                
                // Response schema
                if let schemaJSON = ext.schemaJSON {
                    "\n\nResponse Format:\n"
                    "Generate a JSON data instance that conforms to this JSONSchema.\n"
                    "DO NOT copy the schema structure - create actual data values.\n"
                    "Example: for {\"name\": {\"type\": \"string\"}}, generate {\"name\": \"John Doe\"}\n"
                    "```json\n"
                    "\(schemaJSON)\n"
                    "```"
                }
                
                "\n<</SYS>>\n\n"
            }
            
            // Handle conversation history
            for (index, message) in messages.enumerated() {
                let isFirstUserMessage = messages.prefix(index).filter { $0.role == .user }.isEmpty
                
                switch message.role {
                case .user:
                    if !isFirstUserMessage {
                        // For subsequent user messages, close previous exchange and start new
                        " [/INST] "
                        message.content
                        " </s><s>[INST] "
                    } else {
                        // First user message
                        message.content
                    }
                    
                case .assistant:
                    " [/INST] "
                    message.content
                    " </s><s>[INST] "
                    
                case .tool:
                    // Include tool responses as part of conversation
                    if let toolName = message.toolName {
                        "\n[Tool Response from \(toolName)]:\n"
                    }
                    message.content
                    "\n"
                    
                default:
                    ""
                }
            }
            
            // Tool definitions
            if !ext.toolDefs.isEmpty {
                "\n\nYou have access to the following tools:\n"
                
                for tool in ext.toolDefs {
                    "- \(tool.name)"
                    
                    if let description = tool.description {
                        ": \(description)"
                    }
                    
                    if let parametersJSON = tool.parametersJSON {
                        "\n  Parameters: \(parametersJSON)"
                    }
                    
                    "\n"
                }
                
                "\nTo use a tool, respond with a JSON object containing 'tool_calls'."
            }
            
            " [/INST]"
        }
    }
}

/// ModelCard implementation for Llama 3.2 Instruct format
public struct Llama3ModelCard: ModelCard {
    public let id: String
    
    public init(id: String) {
        self.id = id
    }
    
    public var params: GenerateParameters {
        GenerateParameters(
            maxTokens: 2048,
            temperature: 0.7,
            topP: 0.9,
            repetitionPenalty: 1.1
        )
    }
    
    public func prompt(transcript: Transcript, options: GenerationOptions?) -> Prompt {
        // Extract necessary data
        let ext = TranscriptAccess.extract(from: transcript)
        let currentDate = ISO8601DateFormatter().string(from: Date())
        let messages = ext.messages.filter { $0.role != .system }
        
        return Prompt {
            // System message
            if ext.systemText != nil || ext.schemaJSON != nil || !ext.toolDefs.isEmpty {
                "<|start_header_id|>system<|end_header_id|>\n\n"
                
                if let system = ext.systemText {
                    system
                    "\n\n"
                }
                
                "Current date: \(currentDate)"
                
                // Response schema
                if let schemaJSON = ext.schemaJSON {
                    "\n\nResponse Format:\n"
                    "Generate a JSON data instance that conforms to this JSONSchema.\n"
                    "DO NOT copy the schema structure - create actual data values.\n"
                    "Example: for {\"name\": {\"type\": \"string\"}}, generate {\"name\": \"John Doe\"}\n"
                    "```json\n"
                    "\(schemaJSON)\n"
                    "```"
                }
                
                // Tool definitions
                if !ext.toolDefs.isEmpty {
                    "\n\nYou have access to the following tools:\n\n"
                    
                    for tool in ext.toolDefs {
                        "- \(tool.name)"
                        
                        if let description = tool.description {
                            ": \(description)"
                        }
                        
                        if let parametersJSON = tool.parametersJSON {
                            "\n  Parameters: \(parametersJSON)"
                        }
                        
                        "\n"
                    }
                    
                    "\nTo use a tool, respond with a JSON object containing 'tool_calls'."
                }
                
                "<|eot_id|>"
            }
            
            // Handle conversation history
            for message in messages {
                switch message.role {
                case .user:
                    "<|start_header_id|>user<|end_header_id|>\n\n"
                    message.content
                    "<|eot_id|>"
                    
                case .assistant:
                    "<|start_header_id|>assistant<|end_header_id|>\n\n"
                    message.content
                    "<|eot_id|>"
                    
                case .tool:
                    "<|start_header_id|>tool<|end_header_id|>\n\n"
                    if let toolName = message.toolName {
                        "[Tool Response from \(toolName)]:\n"
                    }
                    message.content
                    "<|eot_id|>"
                    
                default:
                    ""
                }
            }
            
            // Start assistant response
            "<|start_header_id|>assistant<|end_header_id|>\n\n"
        }
    }
}
