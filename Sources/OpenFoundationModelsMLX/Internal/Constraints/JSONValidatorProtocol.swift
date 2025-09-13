import Foundation

public struct ValidationError: Error, LocalizedError, Sendable {
    public let message: String
    public let path: String?
    public let violations: [String]
    
    public init(message: String, path: String? = nil, violations: [String] = []) {
        self.message = message
        self.path = path
        self.violations = violations
    }
    
    public var errorDescription: String? {
        var description = message
        if let path = path {
            description += " at path: \(path)"
        }
        if !violations.isEmpty {
            description += "\nViolations:\n" + violations.map { "  - \($0)" }.joined(separator: "\n")
        }
        return description
    }
}

public protocol JSONValidatorProtocol: Sendable {
    func validate(_ json: String, schema: SchemaNode?) async -> Result<Void, ValidationError>
    
    func validate(_ object: Any, schema: SchemaNode?) async -> Result<Void, ValidationError>
}