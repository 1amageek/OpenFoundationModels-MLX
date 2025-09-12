import Foundation

/// Shared JSON validator that performs basic JSON validation
public final class SharedJSONValidator: JSONValidatorProtocol, @unchecked Sendable {
    public init() {}
    
    public func validate(_ json: String, schema: SchemaNode?) async -> Result<Void, ValidationError> {
        guard let data = json.data(using: .utf8) else {
            return .failure(ValidationError(message: "Invalid UTF-8 string"))
        }
        
        do {
            let parsed = try JSONSerialization.jsonObject(with: data, options: [])
            
            // If schema provided, validate against it
            if let schema = schema {
                let violations = validateObject(parsed, against: schema, path: "$")
                if !violations.isEmpty {
                    return .failure(ValidationError(
                        message: "Schema validation failed",
                        path: "$",
                        violations: violations
                    ))
                }
            }
            
            return .success(())
        } catch {
            return .failure(ValidationError(message: "Invalid JSON: \(error.localizedDescription)"))
        }
    }
    
    public func validate(_ object: Any, schema: SchemaNode?) async -> Result<Void, ValidationError> {
        if let schema = schema {
            let violations = validateObject(object, against: schema, path: "$")
            if !violations.isEmpty {
                return .failure(ValidationError(
                    message: "Schema validation failed",
                    path: "$",
                    violations: violations
                ))
            }
        }
        
        // Also verify it's valid JSON
        do {
            _ = try JSONSerialization.data(withJSONObject: object, options: [])
            return .success(())
        } catch {
            return .failure(ValidationError(message: "Invalid JSON object: \(error.localizedDescription)"))
        }
    }
    
    private func validateObject(_ object: Any, against schema: SchemaNode, path: String) -> [String] {
        var violations: [String] = []
        
        switch schema.kind {
        case .object:
            guard let dict = object as? [String: Any] else {
                violations.append("\(path): Expected object, got \(type(of: object))")
                return violations
            }
            
            // Check required keys
            for requiredKey in schema.required {
                if dict[requiredKey] == nil {
                    violations.append("\(path): Missing required key '\(requiredKey)'")
                }
            }
            
            // Validate properties
            for (key, value) in dict {
                if let propSchema = schema.properties[key] {
                    violations.append(contentsOf: validateObject(value, against: propSchema, path: "\(path).\(key)"))
                }
            }
            
        case .array:
            guard let array = object as? [Any] else {
                violations.append("\(path): Expected array, got \(type(of: object))")
                return violations
            }
            
            if let itemSchema = schema.items {
                for (index, item) in array.enumerated() {
                    violations.append(contentsOf: validateObject(item, against: itemSchema, path: "\(path)[\(index)]"))
                }
            }
            
        case .string:
            if !(object is String) {
                violations.append("\(path): Expected string, got \(type(of: object))")
            }
            
        case .number:
            if !(object is NSNumber) && !(object is Int) && !(object is Double) && !(object is Float) {
                violations.append("\(path): Expected number, got \(type(of: object))")
            }
            
        case .boolean:
            if !(object is Bool) {
                violations.append("\(path): Expected boolean, got \(type(of: object))")
            }
            
        case .null:
            if !(object is NSNull) {
                violations.append("\(path): Expected null, got \(type(of: object))")
            }
            
        case .any:
            // Any type is always valid
            break
        }
        
        return violations
    }
}