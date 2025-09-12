import Foundation
import MLXLMCommon
import Tokenizers

public final class SoftConstraintEngine: ConstraintEngine, @unchecked Sendable {
    public let mode: ConstraintMode = .soft
    private let schemaPreamble: String
    
    public init(schemaPreamble: String? = nil) {
        self.schemaPreamble = schemaPreamble ?? """
            You must return a JSON object that matches the following JSON Schema.
            Start with { and end with }. No extra text before or after the JSON.
            """
    }
    
    public func prepare(schema: SchemaNode?, tokenizer: any Tokenizer) async throws {
    }
    
    public func softPrompt(for schema: SchemaNode?) -> String? {
        guard let schema = schema else { return nil }
        
        var prompt = schemaPreamble + "\n\n"
        prompt += "JSON Schema:\n"
        prompt += formatSchema(schema)
        prompt += "\n\nRemember: Return ONLY valid JSON matching this schema."
        
        return prompt
    }
    
    public func logitProcessors() async -> [LogitProcessor] {
        return []
    }
    
    public func validator() -> (any JSONValidatorProtocol)? {
        return SharedJSONValidator()
    }
    
    private func formatSchema(_ node: SchemaNode, indent: Int = 0) -> String {
        let spacing = String(repeating: "  ", count: indent)
        var result = ""
        
        switch node.kind {
        case .object:
            result += "{\n"
            if !node.properties.isEmpty {
                result += spacing + "  \"type\": \"object\",\n"
                result += spacing + "  \"properties\": {\n"
                
                for (index, (key, value)) in node.properties.enumerated() {
                    result += spacing + "    \"\(key)\": "
                    result += formatSchema(value, indent: indent + 2)
                    if index < node.properties.count - 1 {
                        result += ","
                    }
                    result += "\n"
                }
                
                result += spacing + "  }"
                
                if !node.required.isEmpty {
                    result += ",\n"
                    result += spacing + "  \"required\": ["
                    result += node.required.sorted().map { "\"\($0)\"" }.joined(separator: ", ")
                    result += "]"
                }
                result += "\n" + spacing + "}"
            } else {
                result += spacing + "  \"type\": \"object\"\n" + spacing + "}"
            }
            
        case .array:
            result += "{\n"
            result += spacing + "  \"type\": \"array\""
            if let items = node.items {
                result += ",\n"
                result += spacing + "  \"items\": "
                result += formatSchema(items, indent: indent + 1)
            }
            result += "\n" + spacing + "}"
            
        case .string:
            result += "{ \"type\": \"string\" }"
            
        case .number:
            result += "{ \"type\": \"number\" }"
            
        case .boolean:
            result += "{ \"type\": \"boolean\" }"
            
        case .null:
            result += "{ \"type\": \"null\" }"
            
        case .any:
            result += "{}"
        }
        
        return result
    }
}