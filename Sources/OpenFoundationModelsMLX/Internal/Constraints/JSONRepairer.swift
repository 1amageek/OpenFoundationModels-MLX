import Foundation

public protocol JSONRepairer: Sendable {
    func repair(_ invalidJSON: String, error: ValidationError, schema: SchemaNode?) async -> String?
    
    func canRepair(_ error: ValidationError) -> Bool
}

public extension JSONRepairer {
    func canRepair(_ error: ValidationError) -> Bool {
        return !error.violations.isEmpty
    }
}