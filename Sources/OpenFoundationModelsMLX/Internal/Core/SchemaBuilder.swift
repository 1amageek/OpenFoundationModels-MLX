import Foundation

/// Converts JSON Schema (dictionary format) to hierarchical SchemaNode
/// Handles nested objects and arrays, preserving the full schema structure
public enum SchemaBuilder {
    
    /// Convert a JSON Schema dictionary to SchemaNode
    /// - Parameter dict: JSON Schema as dictionary (from @Generable macro)
    /// - Returns: Hierarchical SchemaNode representing the schema
    public static func fromJSONSchema(_ dict: [String: Any]) -> SchemaNode {
        // Helper to check if a type matches
        func hasType(_ targetType: String, in typeValue: Any?) -> Bool {
            if let singleType = typeValue as? String {
                return singleType == targetType
            }
            if let multipleTypes = typeValue as? [String] {
                return multipleTypes.contains(targetType)
            }
            return false
        }
        
        // Object type
        if hasType("object", in: dict["type"]) || dict["properties"] != nil {
            let propertiesDict = (dict["properties"] as? [String: Any]) ?? [:]
            var properties: [String: SchemaNode] = [:]
            
            // Recursively process each property
            for (key, value) in propertiesDict {
                if let childSchema = value as? [String: Any] {
                    properties[key] = fromJSONSchema(childSchema)
                } else {
                    // Fallback for malformed schema
                    properties[key] = SchemaNode.any
                }
            }
            
            let requiredKeys = Set((dict["required"] as? [String]) ?? [])
            
            print("🔨 [SchemaBuilder] Created object node with keys: \(properties.keys.sorted()), required: \(requiredKeys)")
            
            return SchemaNode(
                kind: .object,
                properties: properties,
                required: requiredKeys
            )
        }
        
        // Array type
        if hasType("array", in: dict["type"]) {
            let itemsSchema: SchemaNode?
            if let itemsDict = dict["items"] as? [String: Any] {
                itemsSchema = fromJSONSchema(itemsDict)
            } else {
                itemsSchema = nil
            }
            
            print("🔨 [SchemaBuilder] Created array node with items: \(itemsSchema?.kind ?? .any)")
            
            return SchemaNode(kind: .array, items: itemsSchema)
        }
        
        // Primitive types
        if hasType("string", in: dict["type"]) {
            return SchemaNode.string
        }
        if hasType("integer", in: dict["type"]) || hasType("number", in: dict["type"]) {
            return SchemaNode.number
        }
        if hasType("boolean", in: dict["type"]) {
            return SchemaNode.boolean
        }
        if hasType("null", in: dict["type"]) {
            return SchemaNode.null
        }
        
        // Handle union types (e.g., ["string", "null"] for optional strings)
        if let types = dict["type"] as? [String] {
            // For now, just use the first non-null type
            for type in types {
                if type != "null" {
                    return fromJSONSchema(["type": type])
                }
            }
        }
        
        // Fallback to any
        print("⚠️ [SchemaBuilder] Unknown schema type, defaulting to 'any': \(dict)")
        return SchemaNode.any
    }
    
    /// Extract SchemaNode from JSON Schema string
    public static func fromJSONString(_ jsonString: String) -> SchemaNode? {
        guard let data = jsonString.data(using: .utf8),
              let dict = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            print("❌ [SchemaBuilder] Failed to parse JSON Schema string")
            return nil
        }
        return fromJSONSchema(dict)
    }
    
    /// Debug helper to print schema structure
    public static func debugPrint(_ node: SchemaNode, indent: Int = 0) {
        let prefix = String(repeating: "  ", count: indent)
        
        switch node.kind {
        case .object:
            print("\(prefix)Object {")
            print("\(prefix)  required: \(node.required)")
            for (key, child) in node.properties.sorted(by: { $0.key < $1.key }) {
                print("\(prefix)  \(key):")
                debugPrint(child, indent: indent + 2)
            }
            print("\(prefix)}")
        case .array:
            print("\(prefix)Array [")
            if let items = node.items {
                debugPrint(items, indent: indent + 1)
            } else {
                print("\(prefix)  <any>")
            }
            print("\(prefix)]")
        default:
            print("\(prefix)\(node.kind)")
        }
    }
}