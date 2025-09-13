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
                    "\n\nResponse Format:"
                    "You must generate a JSON object with ACTUAL DATA VALUES."
                    "The following is a JSON Schema that describes the STRUCTURE your response must follow."
                    "DO NOT output the schema itself. Generate REAL DATA that matches this structure."
                    "\n\nIMPORTANT:"
                    "- For type: \"string\" → generate actual string values like \"John Doe\" or \"example@email.com\""
                    "- For type: \"integer\" → generate actual numbers like 42 or 2020"
                    "- For enum values → use one of the allowed values, not \"enum\" or \"type\""
                    "- DO NOT include \"properties\", \"type\", \"required\" or other schema keywords in your response"
                    "\n\nSchema to follow:"
                    "```json"
                    schemaJSON
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
            // Begin of text marker (required for Llama 3.2)
            "<|begin_of_text|>"
            
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
                    "\n\nResponse Format:"
                    "You must respond with a JSON object containing ACTUAL DATA.\n"
                    
                    "CRITICAL INSTRUCTIONS:"
                    "1. Generate REAL DATA VALUES, not the schema structure"
                    "2. DO NOT output \"properties\", \"type\", \"required\", \"enum\" or any schema keywords"
                    "3. Response is only included JSON"
                    "4. Fill in actual values:\n"
                    "   - Strings: real names, emails, addresses (e.g., \"John Smith\", \"john@example.com\")"
                    "   - Numbers: actual numbers (e.g., 25, 2024, 99.50)"
                    "   - Enums: pick ONE of the allowed values"
                    "   - Arrays: populate with actual items\n"
                    
                    "Example of WRONG response (DO NOT DO THIS):"
                    "{\"properties\": {\"name\": {\"type\": \"string\"}}}"
                    
                    "Example of CORRECT response:"
                    "{\"name\": \"Alice Johnson\", \"age\": 28}"
                    
                    "Schema structure to follow (generate data matching this structure):"
                    "```json"
                    schemaJSON
                    "\n```"
                }
                
                // Tool definitions
                if !ext.toolDefs.isEmpty {
                    "You have access to the following tools:"
                    
                    for tool in ext.toolDefs {
                        "- \(tool.name)"
                        
                        if let description = tool.description {
                            ": \(description)"
                        }
                        
                        if let parametersJSON = tool.parametersJSON {
                            "Parameters: \(parametersJSON)"
                        }
                        
                    }
                    
                    "To use a tool, respond with a JSON object containing 'tool_calls'."
                }
                
                "<|eot_id|>"
            }
            
            // Handle conversation history
            for message in messages {
                switch message.role {
                case .user:
                    "<|start_header_id|>user<|end_header_id|>"
                    message.content
                    "<|eot_id|>"
                    
                case .assistant:
                    "<|start_header_id|>assistant<|end_header_id|>"
                    message.content
                    "<|eot_id|>"
                    
                case .tool:
                    "<|start_header_id|>tool<|end_header_id|>"
                    if let toolName = message.toolName {
                        "[Tool Response from \(toolName)]:"
                    }
                    message.content
                    "<|eot_id|>"
                    
                default:
                    ""
                }
            }
            
            // Start assistant response
            "<|start_header_id|>assistant<|end_header_id|>"
        }
    }
}
