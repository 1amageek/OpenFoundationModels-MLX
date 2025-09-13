import Foundation

/// Minimal schema node to support the constraint system
/// This will be replaced with a proper implementation later
public final class SchemaNode: @unchecked Sendable {
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
}