import Foundation
import OpenFoundationModels
import OpenFoundationModelsExtra
import MLXLMCommon

/// ModelCard implementation for ChatML format
/// Used by many open-source models
public struct ChatMLModelCard: ModelCard {
    public let id: String
    
    public init(id: String) {
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
        // Extract necessary data
        let ext = TranscriptAccess.extract(from: transcript)
        let currentDate = ISO8601DateFormatter().string(from: Date())
        
        return Prompt {
            // System message
            if let system = ext.systemText {
                "<|im_start|>system\n"
                system
                "\nCurrent date: \(currentDate)"
                "<|im_end|>\n"
            }
            
            // Conversation messages
            for message in ext.messages.filter({ $0.role != .system }) {
                "<|im_start|>"
                
                // Role
                message.role.rawValue
                
                // Add tool name for tool responses
                if message.role == .tool, let toolName = message.toolName {
                    " name=\(toolName)"
                }
                
                "\n"
                message.content
                "<|im_end|>\n"
            }
            
            // Tool definitions
            if !ext.toolDefs.isEmpty {
                "<|im_start|>system\n"
                "Available tools:\n"
                
                for tool in ext.toolDefs {
                    "- \(tool.name)"
                    
                    if let description = tool.description {
                        ": \(description)"
                    }
                    
                    "\n"
                    
                    if let parametersJSON = tool.parametersJSON {
                        "  Schema: \(parametersJSON)\n"
                    }
                }
                
                "\nUse tools by responding with JSON in the format: "
                "{\"tool_calls\": [{\"name\": \"tool_name\", \"arguments\": {...}}]}"
                "<|im_end|>\n"
            }
            
            // Start assistant message
            "<|im_start|>assistant\n"
        }
    }
}