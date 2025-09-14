import Foundation

/// Extracts JSON from text streams, handling various formats including embedded JSON
/// This component is responsible for finding JSON content within arbitrary text,
/// such as LLM outputs with special tokens or markdown code blocks
public struct JSONExtractor: Sendable {

    /// Current state of the JSON extraction process
    public enum State: Sendable, Equatable {
        case scanning      // Looking for JSON start
        case inJSON       // Processing JSON content
    }

    /// Current extraction state
    public private(set) var state: State = .scanning

    /// Track if we've found at least one JSON structure
    private var hasFoundJSON: Bool = false

    /// Buffer for potential literal detection (true, false, null)
    private var literalBuffer: String = ""

    /// Track JSON nesting depth for end detection
    private var nestingDepth: Int = 0

    /// Track if we're inside a string (to handle quotes correctly)
    private var inString: Bool = false
    private var escaped: Bool = false

    /// Internal JSON state machine for accurate JSON parsing
    private var jsonStateMachine = JSONStateMachine()

    public init() {}

    /// Process a character and determine if it should be passed to JSONStateMachine
    /// - Parameter char: The character to process
    /// - Returns: true if the character should be processed by JSONStateMachine, false to skip
    public mutating func processCharacter(_ char: Character) -> Bool {
        switch state {
        case .scanning:
            // Clear literal buffer on word boundaries
            if char.isWhitespace || "{}[],:\"()".contains(char) {
                // Check if we completed a literal before clearing
                if literalBuffer == "true" || literalBuffer == "false" || literalBuffer == "null" {
                    // We found a complete literal, start JSON processing
                    state = .inJSON
                    hasFoundJSON = true
                    jsonStateMachine.reset()

                    // Process the entire literal through the state machine
                    for literalChar in literalBuffer {
                        jsonStateMachine.processCharacter(literalChar)
                    }
                    literalBuffer = ""

                    // Check if JSON is already complete (for standalone literals)
                    if jsonStateMachine.isComplete {
                        // For standalone literals, they complete immediately
                        return true
                    }

                    // Now process the current character if it's part of JSON
                    if !char.isWhitespace {
                        jsonStateMachine.processCharacter(char)
                        return true
                    }
                    return false
                }
                literalBuffer = ""
            }

            switch char {
            case "{", "[":
                // Found JSON object or array start
                state = .inJSON
                hasFoundJSON = true
                nestingDepth = 1
                jsonStateMachine.reset()
                jsonStateMachine.processCharacter(char)
                return true

            case "\"":
                // Could be start of a standalone string value
                state = .inJSON
                hasFoundJSON = true
                inString = true
                jsonStateMachine.reset()
                jsonStateMachine.processCharacter(char)
                return true

            case "t", "f", "n":
                // Build potential literal
                literalBuffer.append(char)

                // Check if this could be part of a literal
                if "true".hasPrefix(literalBuffer) ||
                   "false".hasPrefix(literalBuffer) ||
                   "null".hasPrefix(literalBuffer) {
                    // Continue building literal
                    return false
                } else {
                    // Not a valid literal start, reset
                    literalBuffer = ""
                    return false
                }

            default:
                // Continue scanning for JSON start
                if !char.isWhitespace && char.isLetter {
                    literalBuffer.append(char)
                    // Check if we've completed a literal
                    if literalBuffer == "true" || literalBuffer == "false" || literalBuffer == "null" {
                        // Complete literal found
                        state = .inJSON
                        hasFoundJSON = true
                        jsonStateMachine.reset()

                        // Process the entire literal
                        for literalChar in literalBuffer {
                            jsonStateMachine.processCharacter(literalChar)
                        }

                        // Mark as complete since literals are standalone
                        literalBuffer = ""
                        return true
                    }
                } else {
                    literalBuffer = ""
                }
                return false
            }

        case .inJSON:
            // Process character through state machine
            jsonStateMachine.processCharacter(char)

            // Check if JSON is complete AFTER processing this character
            if jsonStateMachine.isComplete || jsonStateMachine.isError {
                // JSON ended, return to scanning mode
                state = .scanning
                nestingDepth = 0
                inString = false
                escaped = false
                // Reset state machine for next JSON
                jsonStateMachine.reset()

                // Include this final character in processing
                return true
            }

            return true
        }
    }

    /// Reset the extractor to initial state
    public mutating func reset() {
        state = .scanning
        hasFoundJSON = false
        literalBuffer = ""
        nestingDepth = 0
        inString = false
        escaped = false
        jsonStateMachine.reset()
    }

    /// Check if currently processing JSON content
    public var isInJSON: Bool {
        return state == .inJSON
    }

    /// Check if JSON has been found
    public var jsonFound: Bool {
        return hasFoundJSON
    }

    /// Notify that JSON has ended (can be called externally based on JSONStateMachine)
    public mutating func notifyJSONEnded() {
        if state == .inJSON {
            state = .scanning
            nestingDepth = 0
            inString = false
            escaped = false
        }
    }

    /// Get current JSON parsing phase
    public func getCurrentPhase() -> JSONStateMachine.Phase {
        return jsonStateMachine.phase
    }

    /// Get completed key if one was just detected
    public func getCompletedKey() -> String? {
        // Check if we're transitioning from key to colon
        let previousPhase = jsonStateMachine.phase
        if case .inString(.body(kind: .key, escaped: false)) = previousPhase {
            // We need to check if the next phase would be expectColon
            // For now, return the current key if available
            return jsonStateMachine.currentKey
        }
        return nil
    }

    /// Get current nesting level in JSON
    public func getNestingLevel() -> Int {
        return jsonStateMachine.nestingLevel
    }
}