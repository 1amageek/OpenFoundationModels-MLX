import Foundation

// Pre-parsed schema summary to drive constrained decoding and validation.
public struct SchemaMeta: Sendable {
    public let keys: [String]
    public let required: [String]
    
    public init(keys: [String], required: [String]) {
        self.keys = keys
        self.required = required
    }
}