import Foundation

/// Hierarchical schema node supporting nested object structures
/// Each object node maintains its own key space, enabling proper constraint switching
/// when entering nested objects like `contact: { email, phone }`
public final class SchemaNode: @unchecked Sendable {
    
    /// The type of JSON value this node represents
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
    
    // For object nodes
    public let properties: [String: SchemaNode]
    public let required: Set<String>
    
    // For array nodes
    public let items: SchemaNode?
    
    public init(
        kind: Kind,
        properties: [String: SchemaNode] = [:],
        required: Set<String> = [],
        items: SchemaNode? = nil
    ) {
        self.kind = kind
        self.properties = properties
        self.required = required
        self.items = items
    }
    
    /// Get sorted keys for object nodes
    public var objectKeys: [String] {
        guard kind == .object else { return [] }
        return properties.keys.sorted()
    }
    
    /// Check if this is an empty schema (no constraints)
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
    
    /// Generate a cache key for this schema node
    /// Used by ADAPTEngine to cache TokenTrie structures
    public func cacheKey() -> String {
        switch kind {
        case .object:
            let sortedKeys = properties.keys.sorted()
            let keyString = sortedKeys.joined(separator: ",")
            let requiredString = required.sorted().joined(separator: ",")
            return "object:\(keyString)|required:\(requiredString)"
        case .array:
            if let items = items {
                return "array:[\(items.cacheKey())]"
            }
            return "array:[]"
        case .string:
            return "string"
        case .number:
            return "number"
        case .boolean:
            return "boolean"
        case .null:
            return "null"
        case .any:
            return "any"
        }
    }
}

// MARK: - Convenience Initializers

extension SchemaNode {
    /// Create an object node with properties
    public static func object(
        properties: [String: SchemaNode],
        required: Set<String> = []
    ) -> SchemaNode {
        SchemaNode(kind: .object, properties: properties, required: required)
    }
    
    /// Create an array node with item schema
    public static func array(items: SchemaNode?) -> SchemaNode {
        SchemaNode(kind: .array, items: items)
    }
    
    /// Create primitive nodes
    public static let string = SchemaNode(kind: .string)
    public static let number = SchemaNode(kind: .number)
    public static let boolean = SchemaNode(kind: .boolean)
    public static let null = SchemaNode(kind: .null)
    public static let any = SchemaNode(kind: .any)
}