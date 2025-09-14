import Foundation

/// Schema node that is fully Sendable compliant
/// Represents JSON Schema structure in a type-safe way
public final class SchemaNode: Sendable {
    public enum Kind: Sendable {
        case object
        case array
        case string
        case number
        case boolean
        case null
        case any
    }

    public let kind: Kind
    public let properties: [String: SchemaNode]
    public let required: Set<String>
    public let items: SchemaNode?
    public let enumValues: [String]?

    public init(
        kind: Kind,
        properties: [String: SchemaNode] = [:],
        required: Set<String> = [],
        items: SchemaNode? = nil,
        enumValues: [String]? = nil
    ) {
        self.kind = kind
        self.properties = properties
        self.required = required
        self.items = items
        self.enumValues = enumValues
    }

    public var objectKeys: [String] {
        guard kind == .object else { return [] }
        return properties.keys.sorted()
    }

    public var isEmpty: Bool {
        switch kind {
        case .object:
            return properties.isEmpty
        case .array:
            return items == nil
        default:
            return false
        }
    }

    // MARK: - JSON Schema Conversion

    /// Create SchemaNode from JSON Schema dictionary
    public static func from(jsonSchema: [String: Any]) -> SchemaNode {
        let type = jsonSchema["type"] as? String ?? "any"

        switch type {
        case "object":
            var properties: [String: SchemaNode] = [:]
            if let props = jsonSchema["properties"] as? [String: [String: Any]] {
                for (key, value) in props {
                    properties[key] = from(jsonSchema: value)
                }
            }

            let required = Set(jsonSchema["required"] as? [String] ?? [])
            return SchemaNode(kind: .object, properties: properties, required: required)

        case "array":
            var items: SchemaNode?
            if let itemSchema = jsonSchema["items"] as? [String: Any] {
                items = from(jsonSchema: itemSchema)
            }
            return SchemaNode(kind: .array, items: items)

        case "string":
            let enumValues = jsonSchema["enum"] as? [String]
            return SchemaNode(kind: .string, enumValues: enumValues)

        case "number", "integer":
            return SchemaNode(kind: .number)

        case "boolean":
            return SchemaNode(kind: .boolean)

        case "null":
            return SchemaNode(kind: .null)

        default:
            // Handle union types
            if let types = jsonSchema["type"] as? [String] {
                // For simplicity, if it contains null and another type, use the other type
                if types.contains("null") {
                    if types.contains("string") { return SchemaNode(kind: .string) }
                    if types.contains("number") { return SchemaNode(kind: .number) }
                    if types.contains("boolean") { return SchemaNode(kind: .boolean) }
                    if types.contains("object") { return from(jsonSchema: jsonSchema) }
                    if types.contains("array") { return from(jsonSchema: jsonSchema) }
                }
            }
            return SchemaNode(kind: .any)
        }
    }

    /// Convert SchemaNode back to JSON Schema dictionary
    public func toJSONSchema() -> [String: Any] {
        var result: [String: Any] = [:]

        switch kind {
        case .object:
            result["type"] = "object"
            if !properties.isEmpty {
                var props: [String: Any] = [:]
                for (key, node) in properties {
                    props[key] = node.toJSONSchema()
                }
                result["properties"] = props
            }
            if !required.isEmpty {
                result["required"] = Array(required)
            }

        case .array:
            result["type"] = "array"
            if let items = items {
                result["items"] = items.toJSONSchema()
            }

        case .string:
            result["type"] = "string"
            if let enumValues = enumValues {
                result["enum"] = enumValues
            }

        case .number:
            result["type"] = "number"

        case .boolean:
            result["type"] = "boolean"

        case .null:
            result["type"] = "null"

        case .any:
            // Don't specify type for any
            break
        }

        return result
    }

    // MARK: - Path Navigation

    /// Get schema node at a specific path
    public func node(at path: [String]) -> SchemaNode? {
        guard !path.isEmpty else { return self }

        let segment = path[0]
        let remainingPath = Array(path.dropFirst())

        switch kind {
        case .object:
            guard let childNode = properties[segment] else { return nil }
            return childNode.node(at: remainingPath)

        case .array:
            // Array indices are handled by going to items
            guard let items = items else { return nil }
            // Skip numeric indices in path
            if Int(segment) != nil {
                return items.node(at: remainingPath)
            } else {
                return items.node(at: path)
            }

        default:
            return nil
        }
    }

    /// Get available keys at current node
    public func availableKeys() -> [String] {
        switch kind {
        case .object:
            return objectKeys
        case .array:
            // Arrays don't have keys, but their items might
            return items?.availableKeys() ?? []
        default:
            return []
        }
    }

    /// Get schema node for a path notation like "departments[].manager"
    public func node(atSchemaPath schemaPath: String) -> SchemaNode? {
        let components = schemaPath.split(separator: ".").map(String.init)
        var currentNode: SchemaNode? = self

        for component in components {
            guard let node = currentNode else { return nil }

            if component.hasSuffix("[]") {
                // Array notation
                let key = String(component.dropLast(2))
                if node.kind == .object, let arrayNode = node.properties[key] {
                    currentNode = arrayNode.items
                } else {
                    return nil
                }
            } else {
                // Regular object property
                if node.kind == .object {
                    currentNode = node.properties[component]
                } else {
                    return nil
                }
            }
        }

        return currentNode
    }
}

// MARK: - Helper Extensions

extension SchemaNode {
    /// Check if this node represents a primitive type
    public var isPrimitive: Bool {
        switch kind {
        case .string, .number, .boolean, .null:
            return true
        case .object, .array, .any:
            return false
        }
    }

    /// Check if this node represents a container type
    public var isContainer: Bool {
        switch kind {
        case .object, .array:
            return true
        default:
            return false
        }
    }

    /// Get all nested keys (flattened)
    public func allKeys(prefix: String = "") -> [String] {
        switch kind {
        case .object:
            var keys: [String] = []
            for (key, node) in properties {
                let fullKey = prefix.isEmpty ? key : "\(prefix).\(key)"
                keys.append(fullKey)
                keys.append(contentsOf: node.allKeys(prefix: fullKey))
            }
            return keys
        case .array:
            if let items = items {
                let arrayPrefix = prefix.isEmpty ? "[]" : "\(prefix)[]"
                return items.allKeys(prefix: arrayPrefix)
            }
            return []
        default:
            return []
        }
    }
}