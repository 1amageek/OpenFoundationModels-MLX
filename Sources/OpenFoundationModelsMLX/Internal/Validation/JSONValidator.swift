import Foundation

/// Hierarchical JSON validator that supports nested schema validation
public struct JSONValidator {
    
    /// Validate JSON text against a hierarchical schema
    /// - Parameters:
    ///   - text: JSON string to validate
    ///   - schema: Hierarchical schema node
    /// - Returns: true if valid, false otherwise
    public static func validate(text: String, schema: SchemaNode) -> Bool {
        print("✅ ADAPT: Starting validation")
        print("✅ ADAPT: Text to validate: \(text)")
        guard let data = text.data(using: .utf8),
              let root = try? JSONSerialization.jsonObject(with: data) else {
            print("✅ ADAPT: Validation ❌ failed - could not parse JSON")
            print("❌ [JSONValidator] Failed to parse JSON")
            return false
        }
        
        let result = validateValue(root, against: schema, path: "root")
        if result {
            print("✅ ADAPT: Validation ✅ passed")
            print("✅ [JSONValidator] JSON validation passed")
        } else {
            print("✅ ADAPT: Validation ❌ failed - schema mismatch")
            print("❌ [JSONValidator] JSON validation failed")
        }
        return result
    }
    
    /// Recursively validate a value against a schema node
    private static func validateValue(_ value: Any, against schema: SchemaNode, path: String) -> Bool {
        switch schema.kind {
        case .object:
            guard let object = value as? [String: Any] else {
                print("❌ [JSONValidator] Expected object at \(path), got \(type(of: value))")
                return false
            }
            
            // Check required fields
            for requiredKey in schema.required {
                guard let fieldValue = object[requiredKey],
                      !(fieldValue is NSNull) else {
                    print("✅ ADAPT: Validation ❌ failed - missing required field '\(requiredKey)' at \(path)")
                    print("❌ [JSONValidator] Missing required field '\(requiredKey)' at \(path)")
                    return false
                }
            }
            
            // Validate each property
            for (key, childSchema) in schema.properties {
                if let fieldValue = object[key] {
                    if !validateValue(fieldValue, against: childSchema, path: "\(path).\(key)") {
                        return false
                    }
                }
            }
            
            // Check for extra keys (optional - can be made configurable)
            let allowedKeys = Set(schema.properties.keys)
            let actualKeys = Set(object.keys)
            let extraKeys = actualKeys.subtracting(allowedKeys)
            if !extraKeys.isEmpty {
                print("⚠️ [JSONValidator] Extra keys found at \(path): \(extraKeys)")
                // For now, allow extra keys but log them
                // return false  // Uncomment to disallow extra keys
            }
            
            return true
            
        case .array:
            guard let array = value as? [Any] else {
                print("❌ [JSONValidator] Expected array at \(path), got \(type(of: value))")
                return false
            }
            
            // Validate each item if schema is provided
            if let itemSchema = schema.items {
                for (index, item) in array.enumerated() {
                    if !validateValue(item, against: itemSchema, path: "\(path)[\(index)]") {
                        return false
                    }
                }
            }
            
            return true
            
        case .string:
            guard value is String else {
                print("❌ [JSONValidator] Expected string at \(path), got \(type(of: value))")
                return false
            }
            return true
            
        case .number:
            // Bool is a subtype of NSNumber, so exclude it first
            if value is Bool {
                print("❌ [JSONValidator] Expected number at \(path), got boolean")
                return false
            }
            guard value is NSNumber || value is Int || value is Double || value is Float else {
                print("❌ [JSONValidator] Expected number at \(path), got \(type(of: value))")
                return false
            }
            return true
            
        case .boolean:
            guard value is Bool else {
                print("❌ [JSONValidator] Expected boolean at \(path), got \(type(of: value))")
                return false
            }
            return true
            
        case .null:
            guard value is NSNull else {
                print("❌ [JSONValidator] Expected null at \(path), got \(type(of: value))")
                return false
            }
            return true
            
        case .any:
            // Any value is acceptable
            return true
        }
    }
}