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
    public private(set) var jsonFound: Bool = false
    
    /// Buffer for potential literal detection (true, false, null)
    private var literalBuffer: String = ""
    
    public init() {}
    
    /// Process a character and determine if it should be passed to JSONStateMachine
    /// - Parameter char: The character to process
    /// - Returns: true if the character should be processed by JSONStateMachine, false to skip
    public mutating func processCharacter(_ char: Character) -> Bool {
        switch state {
        case .scanning:
            // Clear literal buffer if we hit whitespace or other non-literal char
            if char.isWhitespace || "{}[],:\"".contains(char) {
                literalBuffer = ""
            }
            
            switch char {
            case "{", "[":
                // Found JSON object or array start
                state = .inJSON
                jsonFound = true
                return true
                
            case "\"":
                // Could be start of a standalone string value
                state = .inJSON
                jsonFound = true
                return true
                
            case "-", "0"..."9":
                // Could be start of a number
                state = .inJSON
                jsonFound = true
                return true
                
            case "t", "f", "n":
                // Could be start of true, false, or null
                literalBuffer.append(char)
                
                // Check if we have a complete literal
                if literalBuffer == "true" || literalBuffer == "false" || literalBuffer == "null" {
                    state = .inJSON
                    jsonFound = true
                    // Need to replay the entire literal to JSONStateMachine
                    // For now, just mark as found
                    return true
                } else if "true".hasPrefix(literalBuffer) || 
                         "false".hasPrefix(literalBuffer) || 
                         "null".hasPrefix(literalBuffer) {
                    // Potential literal in progress, keep scanning
                    return false
                } else {
                    // Not a valid literal start, reset buffer
                    literalBuffer = ""
                    return false
                }
                
            default:
                // Continue scanning for JSON start
                if !char.isWhitespace {
                    literalBuffer.append(char)
                }
                return false
            }
            
        case .inJSON:
            // Once we're in JSON, pass everything through
            // The JSONStateMachine will handle validation and completion
            return true
        }
    }
    
    /// Reset the extractor to initial state
    public mutating func reset() {
        state = .scanning
        jsonFound = false
        literalBuffer = ""
    }
    
    /// Check if currently processing JSON content
    public var isInJSON: Bool {
        return state == .inJSON
    }
}