import Foundation

/// Detects the current context in a partial JSON string and returns available schema keys
/// Uses SchemaNode for type-safe schema representation
public struct JSONSchemaContextDetector: Sendable {

    // MARK: - Types

    /// Represents the current position in a JSON structure
    public enum JSONPosition: Sendable, Equatable {
        case beforeFirstKey           // `{` - ready for first key
        case insideKey               // `{"na` - typing a key
        case afterKey                // `{"name"` - after closing quote of key
        case afterColon              // `{"name":` - after colon, expecting value
        case insideStringValue       // `{"name":"Jo` - inside string value
        case insideNumberValue       // `{"age":2` - inside number value
        case insideBooleanValue      // `{"active":tru` - inside boolean value
        case insideNullValue         // `{"data":nul` - inside null value
        case afterValue              // `{"name":"John"` - after value, expecting comma or close
        case afterComma              // `{"name":"John",` - after comma, expecting next key
        case complete                // Complete JSON
        case invalid                 // Invalid state
    }

    /// Represents a path in the JSON structure
    public struct JSONPath: Sendable, Equatable {
        let segments: [String]

        /// Convert to schema path notation
        /// e.g., ["departments", "0", "manager"] -> "departments[].manager"
        var schemaPath: String {
            var result: [String] = []
            var i = 0
            while i < segments.count {
                let segment = segments[i]
                // Check if this is an array index
                if i > 0, let _ = Int(segment) {
                    // Previous segment is an array, add [] notation
                    if !result.isEmpty && !result[result.count - 1].hasSuffix("[]") {
                        result[result.count - 1] += "[]"
                    }
                } else {
                    result.append(segment)
                }
                i += 1
            }
            return result.joined(separator: ".")
        }

        init(_ segments: [String] = []) {
            self.segments = segments
        }
    }

    // MARK: - Properties

    private let schemaNode: SchemaNode

    // MARK: - Initialization

    public init(schema: SchemaNode) {
        self.schemaNode = schema
    }

    /// Convenience initializer from JSON Schema dictionary
    public init(schema: [String: Any]) {
        self.schemaNode = SchemaNode.from(jsonSchema: schema)
    }

    // MARK: - Public Methods

    /// Get available keys for a partial JSON string
    public func getAvailableKeys(from partialJSON: String) -> [String] {
        // Parse the partial JSON to understand current context
        let context = parseContext(from: partialJSON)

        // If we're not in a position to add keys, return empty
        guard shouldShowKeys(at: context.position) else {
            return []
        }

        // Get the schema node for the current path
        let node = getSchemaNode(at: context.path)

        // Extract available keys from schema node
        let availableKeys = node?.availableKeys() ?? []

        // Filter out already used keys
        let remainingKeys = availableKeys.filter { !context.usedKeys.contains($0) }

        return remainingKeys
    }

    // MARK: - Private Methods

    private func shouldShowKeys(at position: JSONPosition) -> Bool {
        switch position {
        case .beforeFirstKey, .afterComma, .insideKey:
            return true
        default:
            return false
        }
    }

    private func parseContext(from partialJSON: String) -> (position: JSONPosition, path: JSONPath, usedKeys: Set<String>) {
        var position: JSONPosition = .invalid
        var path: [String] = []
        var usedKeys: Set<String> = []
        var currentKey = ""

        var inString = false
        var isKey = false
        var escaped = false
        var depth = 0
        var arrayDepths: Set<Int> = []

        for char in partialJSON {
            if escaped {
                escaped = false
                if inString && isKey {
                    currentKey.append(char)
                }
                continue
            }

            if char == "\\" {
                escaped = true
                if inString && isKey {
                    currentKey.append(char)
                }
                continue
            }

            switch char {
            case "\"":
                if !inString {
                    inString = true
                    if position == .beforeFirstKey || position == .afterComma {
                        isKey = true
                        currentKey = ""
                        position = .insideKey
                    } else if position == .afterColon {
                        position = .insideStringValue
                    }
                } else {
                    inString = false
                    if isKey {
                        isKey = false
                        position = .afterKey
                    } else if position == .insideStringValue {
                        position = .afterValue
                    }
                }

            case ":":
                if !inString && position == .afterKey {
                    position = .afterColon
                    // Add current key to path temporarily
                    path.append(currentKey)
                }

            case ",":
                if !inString {
                    if position == .afterValue {
                        position = .afterComma
                        // Mark the last key as used and remove from path
                        if !path.isEmpty {
                            usedKeys.insert(path.removeLast())
                        }
                    }
                }

            case "{":
                if !inString {
                    depth += 1
                    if depth == 1 {
                        position = .beforeFirstKey
                    } else if position == .afterColon {
                        // Entering nested object
                        position = .beforeFirstKey
                    }
                }

            case "}":
                if !inString {
                    depth -= 1
                    if depth == 0 {
                        position = .complete
                    } else {
                        position = .afterValue
                        // Pop the current object from path
                        if !path.isEmpty && !arrayDepths.contains(path.count) {
                            usedKeys.insert(path.removeLast())
                        }
                    }
                }

            case "[":
                if !inString && position == .afterColon {
                    // Mark this depth as array
                    arrayDepths.insert(path.count)
                    path.append("0")  // Array index
                }

            case "]":
                if !inString {
                    position = .afterValue
                    // Remove array index from path
                    if !path.isEmpty && arrayDepths.contains(path.count) {
                        arrayDepths.remove(path.count)
                        path.removeLast()
                    }
                }

            default:
                if inString && isKey {
                    currentKey.append(char)
                } else if !inString {
                    // Handle non-string values
                    if position == .afterColon {
                        if char.isNumber || char == "-" {
                            position = .insideNumberValue
                        } else if char == "t" || char == "f" {
                            position = .insideBooleanValue
                        } else if char == "n" {
                            position = .insideNullValue
                        }
                    }
                }
            }
        }

        // Handle incomplete key
        if isKey && !currentKey.isEmpty {
            path.append(currentKey)
        }

        return (position, JSONPath(path), usedKeys)
    }

    private func getSchemaNode(at path: JSONPath) -> SchemaNode? {
        // Convert path to schema path and get node
        let schemaPath = path.schemaPath
        if schemaPath.isEmpty {
            return schemaNode
        }
        return schemaNode.node(atSchemaPath: schemaPath)
    }
}