import Foundation

public struct JSONValidator {
    public static func validate(text: String, schema: SchemaNode) -> Bool {
        guard let data = text.data(using: .utf8),
              let root = try? JSONSerialization.jsonObject(with: data) else {
            return false
        }
        
        return validateValue(root, against: schema, path: "root")
    }
    
    private static func validateValue(_ value: Any, against schema: SchemaNode, path: String) -> Bool {
        switch schema.kind {
        case .object:
            guard let object = value as? [String: Any] else {
                return false
            }
            
            for requiredKey in schema.required {
                guard let fieldValue = object[requiredKey],
                      !(fieldValue is NSNull) else {
                    return false
                }
            }
            for (key, childSchema) in schema.properties {
                if let fieldValue = object[key] {
                    if !validateValue(fieldValue, against: childSchema, path: "\(path).\(key)") {
                        return false
                    }
                }
            }
            
            return true
            
        case .array:
            guard let array = value as? [Any] else {
                return false
            }
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
                return false
            }
            return true
            
        case .number:
            if value is Bool {
                return false
            }
            guard value is NSNumber || value is Int || value is Double || value is Float else {
                return false
            }
            return true
            
        case .boolean:
            guard value is Bool else {
                return false
            }
            return true
            
        case .null:
            guard value is NSNull else {
                return false
            }
            return true
            
        case .any:
            return true
        }
    }
}