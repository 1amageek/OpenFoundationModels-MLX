import Foundation

/// Extracts constraint information directly from JSON Schema
public enum JSONSchemaExtractor {
    
    /// Extract property keys from a JSON Schema string
    /// - Parameter schemaJSON: The JSON Schema as a string
    /// - Returns: List of allowed property keys, or nil if parsing fails
    public static func extractKeys(from schemaJSON: String) -> [String]? {
        guard let data = schemaJSON.data(using: .utf8),
              let dict = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            return nil
        }
        
        return extractKeys(from: dict)
    }
    
    /// Extract property keys from a JSON Schema dictionary
    /// - Parameter schemaDict: The JSON Schema as a dictionary
    /// - Returns: List of allowed property keys
    public static func extractKeys(from schemaDict: [String: Any]) -> [String]? {
        guard let properties = schemaDict["properties"] as? [String: Any] else {
            return nil
        }
        
        return Array(properties.keys).sorted()
    }
    
    /// Extract required keys from a JSON Schema string
    /// - Parameter schemaJSON: The JSON Schema as a string
    /// - Returns: Set of required property keys, or nil if parsing fails
    public static func extractRequiredKeys(from schemaJSON: String) -> Set<String>? {
        guard let data = schemaJSON.data(using: .utf8),
              let dict = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            return nil
        }
        
        return extractRequiredKeys(from: dict)
    }
    
    /// Extract required keys from a JSON Schema dictionary
    /// - Parameter schemaDict: The JSON Schema as a dictionary
    /// - Returns: Set of required property keys
    public static func extractRequiredKeys(from schemaDict: [String: Any]) -> Set<String>? {
        guard let required = schemaDict["required"] as? [String] else {
            return nil
        }
        
        return Set(required)
    }
    
    /// Build a simple SchemaNode from JSON Schema for compatibility
    /// This creates a SchemaNode with actual properties from the schema
    public static func buildSchemaNode(from schemaJSON: String) -> SchemaNode? {
        guard let data = schemaJSON.data(using: .utf8),
              let dict = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            return nil
        }
        
        return buildSchemaNode(from: dict)
    }
    
    /// Build a simple SchemaNode from JSON Schema dictionary
    public static func buildSchemaNode(from schemaDict: [String: Any]) -> SchemaNode? {
        // Extract properties
        let properties = schemaDict["properties"] as? [String: Any] ?? [:]
        let required = Set(schemaDict["required"] as? [String] ?? [])
        
        // Build simple SchemaNode properties (not fully recursive for now)
        var schemaProperties: [String: SchemaNode] = [:]
        for (key, value) in properties {
            if let propDict = value as? [String: Any],
               let type = propDict["type"] as? String {
                let kind = schemaKind(from: type)
                
                // Handle nested objects recursively
                if kind == .object,
                   let nestedProps = propDict["properties"] as? [String: Any] {
                    let nestedRequired = Set(propDict["required"] as? [String] ?? [])
                    var nestedSchemaProps: [String: SchemaNode] = [:]
                    
                    for (nestedKey, nestedValue) in nestedProps {
                        if let nestedPropDict = nestedValue as? [String: Any],
                           let nestedType = nestedPropDict["type"] as? String {
                            nestedSchemaProps[nestedKey] = SchemaNode(
                                kind: schemaKind(from: nestedType),
                                properties: [:],
                                required: []
                            )
                        }
                    }
                    
                    schemaProperties[key] = SchemaNode(
                        kind: .object,
                        properties: nestedSchemaProps,
                        required: nestedRequired
                    )
                } else if kind == .array,
                          let items = propDict["items"] as? [String: Any],
                          let itemType = items["type"] as? String {
                    // Handle array items
                    schemaProperties[key] = SchemaNode(
                        kind: .array,
                        properties: [:],
                        required: [],
                        items: SchemaNode(kind: schemaKind(from: itemType))
                    )
                } else {
                    // Simple type
                    schemaProperties[key] = SchemaNode(kind: kind)
                }
            }
        }
        
        return SchemaNode(
            kind: .object,
            properties: schemaProperties,
            required: required
        )
    }
    
    private static func schemaKind(from type: String) -> SchemaNode.Kind {
        switch type {
        case "object":
            return .object
        case "array":
            return .array
        case "string":
            return .string
        case "number", "integer":
            return .number
        case "boolean":
            return .boolean
        case "null":
            return .null
        default:
            return .any
        }
    }
}