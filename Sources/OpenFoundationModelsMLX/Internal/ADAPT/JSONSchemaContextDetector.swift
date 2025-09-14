import Foundation

/// Detects the current context in a partial JSON string and returns available schema keys
/// This is a standalone component with no dependencies on other parts of the system
public struct JSONSchemaContextDetector {

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

    private let schema: [String: Any]

    // MARK: - Initialization

    public init(schema: [String: Any]) {
        self.schema = schema
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

        // Get the schema definition for the current path
        let schemaNode = getSchemaNode(at: context.path)

        // Extract available keys from schema
        let allKeys = extractKeys(from: schemaNode)

        // Filter out already used keys at this level
        let usedKeys = context.usedKeysAtCurrentLevel
        let availableKeys = allKeys.filter { !usedKeys.contains($0) }

        return availableKeys.sorted()
    }

    // MARK: - Private Methods

    /// Parse context from partial JSON using a more robust approach
    private func parseContext(from partialJSON: String) -> (position: JSONPosition, path: JSONPath, usedKeysAtCurrentLevel: Set<String>) {
        guard !partialJSON.isEmpty else {
            return (.invalid, JSONPath(), [])
        }

        // Track nested context
        struct Context {
            var path: [String] = []
            var usedKeys: [String] = []
            var pendingKey: String? = nil
        }

        var contextStack: [Context] = []
        var currentContext = Context()
        var position: JSONPosition = .invalid

        var inString = false
        var inKey = false
        var escaped = false
        var currentKey = ""
        var depth = 0

        var i = partialJSON.startIndex
        while i < partialJSON.endIndex {
            let char = partialJSON[i]

            // Handle escape sequences
            if escaped {
                escaped = false
                if inKey {
                    currentKey.append(char)
                }
                i = partialJSON.index(after: i)
                continue
            }

            if char == "\\" && inString {
                escaped = true
                i = partialJSON.index(after: i)
                continue
            }

            // Main character processing
            if !inString {
                switch char {
                case "{":
                    depth += 1

                    // If we have a pending key, this object belongs to that key
                    if let pendingKey = currentContext.pendingKey {
                        // Save current context before going deeper
                        contextStack.append(currentContext)

                        // Create new context for nested object
                        currentContext = Context()
                        currentContext.path = contextStack.last?.path ?? []
                        currentContext.path.append(pendingKey)
                        currentContext.usedKeys = []
                        currentContext.pendingKey = nil
                    } else if depth > 1 {
                        // Object inside array or anonymous object
                        contextStack.append(currentContext)
                        currentContext = Context()
                        currentContext.path = contextStack.last?.path ?? []
                    }

                    position = .beforeFirstKey

                case "}":
                    depth -= 1

                    // Restore previous context
                    if !contextStack.isEmpty {
                        currentContext = contextStack.removeLast()
                    }

                    if depth == 0 {
                        position = .complete
                    } else {
                        position = .afterValue
                    }

                    // Clear pending key as we've exited the object
                    currentContext.pendingKey = nil

                case "[":
                    depth += 1

                    // If we have a pending key, this array belongs to that key
                    if let pendingKey = currentContext.pendingKey {
                        // Save current context
                        contextStack.append(currentContext)

                        // Create new context for array
                        currentContext = Context()
                        currentContext.path = contextStack.last?.path ?? []
                        currentContext.path.append(pendingKey)
                        currentContext.path.append("0")  // Array index
                        currentContext.usedKeys = []
                        currentContext.pendingKey = nil
                    }

                    position = .afterColon  // Arrays expect values

                case "]":
                    depth -= 1

                    // Restore previous context
                    if !contextStack.isEmpty {
                        currentContext = contextStack.removeLast()
                    }

                    position = .afterValue

                case "\"":
                    inString = true
                    if position == .beforeFirstKey || position == .afterComma {
                        inKey = true
                        currentKey = ""
                        position = .insideKey
                    } else if position == .afterColon {
                        position = .insideStringValue
                    }

                case ":":
                    position = .afterColon

                case ",":
                    position = .afterComma
                    // Clear pending key if value was not object/array
                    currentContext.pendingKey = nil

                case " ", "\t", "\n", "\r":
                    break  // Skip whitespace

                default:
                    // Handle numbers, booleans, null
                    if position == .afterColon {
                        if char.isNumber || char == "-" {
                            position = .insideNumberValue
                        } else if char == "t" || char == "f" {
                            position = .insideBooleanValue
                        } else if char == "n" {
                            position = .insideNullValue
                        }
                        // Clear pending key for primitive values
                        currentContext.pendingKey = nil
                    }
                }
            } else {
                // Inside a string
                if char == "\"" {
                    inString = false
                    if inKey {
                        inKey = false
                        // Key completed
                        currentContext.usedKeys.append(currentKey)
                        currentContext.pendingKey = currentKey
                        position = .afterKey
                    } else {
                        // String value completed
                        position = .afterValue
                        currentContext.pendingKey = nil
                    }
                } else {
                    if inKey {
                        currentKey.append(char)
                    }
                }
            }

            i = partialJSON.index(after: i)
        }

        // Handle incomplete states
        if inString && inKey {
            position = .insideKey
        }

        let finalPath = JSONPath(currentContext.path)
        let finalUsedKeys = Set(currentContext.usedKeys)

        return (position, finalPath, finalUsedKeys)
    }

    /// Check if keys should be shown at this position
    private func shouldShowKeys(at position: JSONPosition) -> Bool {
        switch position {
        case .beforeFirstKey, .afterComma:
            return true
        default:
            return false
        }
    }

    /// Get schema node for a given path
    private func getSchemaNode(at path: JSONPath) -> [String: Any]? {
        var currentNode = schema

        for segment in path.segments {
            // Skip array indices
            if Int(segment) != nil {
                continue
            }

            // Navigate to nested object or array
            if let properties = currentNode["properties"] as? [String: Any],
               let nextNode = properties[segment] as? [String: Any] {

                if nextNode["type"] as? String == "array",
                   let items = nextNode["items"] as? [String: Any] {
                    currentNode = items
                } else {
                    currentNode = nextNode
                }
            } else {
                return nil
            }
        }

        return currentNode
    }

    /// Extract keys from a schema node
    private func extractKeys(from schemaNode: [String: Any]?) -> [String] {
        guard let node = schemaNode,
              let properties = node["properties"] as? [String: Any] else {
            return []
        }

        return Array(properties.keys)
    }
}

// MARK: - Character Extension

private extension Character {
    var isNumber: Bool {
        return self >= "0" && self <= "9"
    }
}