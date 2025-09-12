import Foundation

/// Simple JSON key detector without MLX dependency
public class SimpleKeyDetector {
    private var stateMachine = SimpleJSONStateMachine()
    private var detectedKeys: [String] = []
    private var currentKeyBuffer = ""
    private var nestingStack: [String] = []
    
    public init() {}
    
    public func processJSON(_ text: String) -> [String] {
        // Reset state
        stateMachine = SimpleJSONStateMachine()
        detectedKeys = []
        currentKeyBuffer = ""
        nestingStack = []
        
        // Find where JSON actually starts
        let jsonStart = findJSONStart(in: text)
        let jsonString = String(text.dropFirst(jsonStart))
        
        if jsonStart > 0 {
            print("  📌 Skipped \(jsonStart) characters before JSON start")
        }
        
        // Process each character
        for char in jsonString {
            let wasInKey = stateMachine.isInKeyGeneration
            
            stateMachine.processCharacter(char)
            
            // Track key buffer
            if stateMachine.isInKeyGeneration {
                if char != "\"" {
                    currentKeyBuffer.append(char)
                }
            } else if wasInKey && !stateMachine.isInKeyGeneration && !currentKeyBuffer.isEmpty {
                // Key completed
                detectedKeys.append(currentKeyBuffer)
                print("  🔑 Detected key: \"\(currentKeyBuffer)\"")
                currentKeyBuffer = ""
            }
        }
        
        return detectedKeys
    }
    
    private func findJSONStart(in text: String) -> Int {
        // Skip common prefixes
        var index = 0
        var inCodeBlock = false
        let chars = Array(text)
        
        while index < chars.count {
            let char = chars[index]
            
            // Check for markdown code block
            if index + 7 < chars.count {
                let next7 = String(chars[index..<index+7])
                if next7 == "```json" || next7 == "```JSON" {
                    index += 7
                    inCodeBlock = true
                    // Skip to next line
                    while index < chars.count && chars[index] != "\n" {
                        index += 1
                    }
                    if index < chars.count {
                        index += 1 // Skip the newline
                    }
                    continue
                }
            }
            
            // Check for triple backticks alone
            if index + 3 < chars.count {
                let next3 = String(chars[index..<index+3])
                if next3 == "```" {
                    index += 3
                    if inCodeBlock {
                        // End of code block, but might not be our JSON
                        inCodeBlock = false
                    }
                    continue
                }
            }
            
            // Look for JSON start
            if char == "{" || char == "[" {
                return index
            }
            
            // Skip whitespace and common text
            index += 1
        }
        
        return 0 // Default to start if no JSON found
    }
}

/// Simplified JSON state machine
public struct SimpleJSONStateMachine {
    public enum Phase {
        case root
        case inObject
        case inKey
        case afterKey
        case inValue
        case inString
        case done
        case error
    }
    
    public private(set) var phase: Phase = .root
    private var escapeNext = false
    private var nestingLevel = 0
    
    public var isInKeyGeneration: Bool {
        return phase == .inKey
    }
    
    public mutating func processCharacter(_ char: Character) {
        // Handle escape sequences
        if escapeNext {
            escapeNext = false
            return
        }
        
        if char == "\\" {
            escapeNext = true
            return
        }
        
        switch phase {
        case .root:
            if char == "{" {
                phase = .inObject
                nestingLevel = 1
            } else if char == "[" {
                // Arrays not fully handled in this simple version
                phase = .inValue
            }
            
        case .inObject:
            if char == "\"" {
                phase = .inKey
            } else if char == "}" {
                nestingLevel -= 1
                if nestingLevel == 0 {
                    phase = .done
                }
            }
            
        case .inKey:
            if char == "\"" && !escapeNext {
                phase = .afterKey
            }
            
        case .afterKey:
            if char == ":" {
                phase = .inValue
            }
            
        case .inValue:
            if char == "\"" {
                phase = .inString
            } else if char == "{" {
                phase = .inObject
                nestingLevel += 1
            } else if char == "[" {
                // Simplified array handling
            } else if char == "," {
                phase = .inObject
            } else if char == "}" {
                nestingLevel -= 1
                if nestingLevel == 0 {
                    phase = .done
                } else {
                    phase = .inObject
                }
            } else if char != " " && char != "\t" && char != "\n" && char != "\r" {
                // Number, boolean, or null - stay in value
            }
            
        case .inString:
            if char == "\"" && !escapeNext {
                phase = .inValue
            }
            
        case .done, .error:
            break
        }
    }
}