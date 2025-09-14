import Foundation

/// Tracks the current context within a JSON structure during parsing
/// Provides accurate path information for nested objects and arrays
final class JSONContextTracker: @unchecked Sendable {

    // MARK: - Types

    /// Represents a context level in the JSON structure
    enum Context: Sendable, Equatable {
        case root
        case object(key: String?)       // key is the key name for this object (nil for root)
        case array(key: String)         // key is the array's key name
    }

    // MARK: - Properties

    private var contextStack: [Context] = [.root]
    private var lastDetectedKey: String? = nil
    private var isWaitingForValue: Bool = false

    // MARK: - Public Interface

    /// Called when a JSON key is detected
    func keyDetected(_ key: String) {
        lastDetectedKey = key
        isWaitingForValue = true
    }

    /// Called when entering an object
    func enterObject() {
        // If we're waiting for a value and have a key, this object belongs to that key
        if isWaitingForValue, let key = lastDetectedKey {
            contextStack.append(.object(key: key))
            lastDetectedKey = nil
            isWaitingForValue = false
        } else if case .array = contextStack.last {
            // We're in an array, this is an array item object
            contextStack.append(.object(key: nil))
        } else {
            // Root object or nested without key
            contextStack.append(.object(key: nil))
        }
    }

    /// Called when entering an array
    func enterArray() {
        // If we have a pending key, this array belongs to that key
        if isWaitingForValue, let key = lastDetectedKey {
            contextStack.append(.array(key: key))
            lastDetectedKey = nil
            isWaitingForValue = false
        } else {
            // Fallback: array without a clear key (shouldn't happen in well-formed JSON)
            // But handle it gracefully
            contextStack.append(.array(key: "unknown"))
        }
    }

    /// Called when exiting a context (object or array)
    func exitContext() {
        if contextStack.count > 1 {
            contextStack.removeLast()
        }
    }

    /// Reset to initial state
    func reset() {
        contextStack = [.root]
        lastDetectedKey = nil
        isWaitingForValue = false
    }

    /// Get the current context path for schema lookup
    /// Returns paths like "headquarters", "departments[]", "departments[].manager"
    func getCurrentPath() -> String {
        var path: [String] = []
        var inArray = false
        var arrayKey: String? = nil

        for context in contextStack {
            switch context {
            case .root:
                continue

            case .object(let key):
                if inArray {
                    // This is an object within an array
                    if let key = key {
                        // This object has a key (nested object in array item)
                        path.append(key)
                    }
                    // If key is nil, it's the array item itself, don't add to path
                } else if let key = key {
                    // Regular object with a key
                    path.append(key)
                }

            case .array(let key):
                // Mark that we're in an array context
                inArray = true
                arrayKey = key
                // Add array notation to path
                path.append("\(key)[]")
            }
        }

        return path.joined(separator: ".")
    }

    /// Check if we're currently in an array context
    func isInArray() -> Bool {
        return contextStack.contains { context in
            if case .array = context {
                return true
            }
            return false
        }
    }

    /// Get the current array context if any
    func getCurrentArrayContext() -> String? {
        // Find the most recent array in the stack
        for context in contextStack.reversed() {
            if case .array(let key) = context {
                return "\(key)[]"
            }
        }
        return nil
    }

    /// Get the depth of nesting
    var nestingDepth: Int {
        return contextStack.count - 1  // Subtract root
    }

    /// Get keys that should be shown based on current context
    func getContextKeys(nestedSchemas: [String: [String]]?, rootKeys: [String]?) -> [String] {
        let path = getCurrentPath()

        // If we have a path, try to find nested keys
        if !path.isEmpty, let nestedSchemas = nestedSchemas {
            if let keys = nestedSchemas[path] {
                return keys
            }
        }

        // Fall back to root keys
        return rootKeys ?? []
    }

    // MARK: - Debugging

    func debugDescription() -> String {
        return "Context: \(getCurrentPath()), Stack: \(contextStack), LastKey: \(lastDetectedKey ?? "none")"
    }
}