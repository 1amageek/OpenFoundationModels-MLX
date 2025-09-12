import Foundation

/// State machine for tracking JSON parsing phases
/// Processes character-by-character to maintain current JSON context
public struct JSONStateMachine: Sendable {
    
    // MARK: - Phase Definitions
    
    public enum Phase: Sendable, Equatable {
        case root
        case inObject(ObjectPhase)
        case inArray(ArrayPhase)
        case inString(StringPhase)
        case inNumber(NumberPhase)
        case inLiteral(LiteralPhase)
        case done
        case error
    }
    
    public enum ObjectPhase: Sendable, Equatable {
        case expectKeyOrEnd          // After { or after a value
        case expectKeyFirstQuote     // Expecting opening quote for key
        case inKey                   // Inside key string
        case expectKeyEndQuote       // Expecting closing quote for key
        case expectColon             // After key, expecting :
        case expectValue             // After :, expecting value
        case expectCommaOrEnd        // After value, expecting , or }
    }
    
    public enum ArrayPhase: Sendable, Equatable {
        case expectValue             // After [ or ,
        case expectCommaOrEnd        // After value, expecting , or ]
    }
    
    public enum StringPhase: Sendable, Equatable {
        case body(kind: StringKind, escaped: Bool)
        
        public enum StringKind: Sendable, Equatable {
            case key
            case value
        }
    }
    
    public enum NumberPhase: Sendable, Equatable {
        case integer
        case decimal
        case exponent
    }
    
    public enum LiteralPhase: Sendable, Equatable {
        case inProgress(String)  // Building literal (true, false, null)
    }
    
    // MARK: - State Properties
    
    public private(set) var phase: Phase = .root
    public private(set) var nestingLevel: Int = 0
    public private(set) var currentKey: String = ""
    public private(set) var isGenerating: Bool = false
    private var buffer: String = ""
    private var contextStack: [Phase] = []  // Track parent contexts
    
    // MARK: - Public Interface
    
    public init() {}
    
    /// Process a single character and update state
    public mutating func processCharacter(_ char: Character) {
        guard phase != .error && phase != .done else { return }
        
        switch phase {
        case .root:
            processRootCharacter(char)
            
        case .inObject(let objPhase):
            processObjectCharacter(char, phase: objPhase)
            
        case .inArray(let arrPhase):
            processArrayCharacter(char, phase: arrPhase)
            
        case .inString(let strPhase):
            processStringCharacter(char, phase: strPhase)
            
        case .inNumber(let numPhase):
            processNumberCharacter(char, phase: numPhase)
            
        case .inLiteral(let litPhase):
            processLiteralCharacter(char, phase: litPhase)
            
        case .done, .error:
            break
        }
    }
    
    /// Process a complete string (convenience method)
    public mutating func processString(_ string: String) {
        for char in string {
            processCharacter(char)
        }
    }
    
    /// Reset to initial state
    public mutating func reset() {
        phase = .root
        nestingLevel = 0
        currentKey = ""
        buffer = ""
        isGenerating = false
        contextStack = []
    }
    
    /// Check if in a state where we're generating a JSON key
    public var isInKeyGeneration: Bool {
        if case .inObject(let phase) = phase {
            return phase == .inKey || phase == .expectKeyFirstQuote
        }
        if case .inString(.body(kind: .key, _)) = phase {
            return true
        }
        return false
    }
    
    /// Check if the state machine has completed successfully
    public var isComplete: Bool {
        return phase == .done
    }
    
    /// Check if the state machine is in an error state
    public var isError: Bool {
        return phase == .error
    }
    
    // MARK: - Private Processing Methods
    
    private mutating func processRootCharacter(_ char: Character) {
        switch char {
        case "{":
            phase = .inObject(.expectKeyOrEnd)
            nestingLevel = 1
            isGenerating = true
            // No need to push context at root level
            
        case "[":
            phase = .inArray(.expectValue)
            nestingLevel = 1
            isGenerating = true
            // No need to push context at root level
            
        case "\"":
            phase = .inString(.body(kind: .value, escaped: false))
            isGenerating = true
            
        case "-", "0"..."9":
            phase = .inNumber(.integer)
            buffer = String(char)
            isGenerating = true
            
        case "t", "f", "n":
            phase = .inLiteral(.inProgress(String(char)))
            isGenerating = true
            
        case " ", "\t", "\n", "\r":
            break  // Skip whitespace
            
        default:
            phase = .error
        }
    }
    
    private mutating func processObjectCharacter(_ char: Character, phase objPhase: ObjectPhase) {
        switch objPhase {
        case .expectKeyOrEnd:
            switch char {
            case "}":
                nestingLevel -= 1
                if nestingLevel == 0 {
                    self.phase = .done
                    isGenerating = false
                } else {
                    // Pop and return to parent context
                    if let parentContext = contextStack.popLast() {
                        // When returning to an array after completing an object value,
                        // we should be expecting comma or end, not another value
                        switch parentContext {
                        case .inArray(.expectValue):
                            self.phase = .inArray(.expectCommaOrEnd)
                        default:
                            self.phase = parentContext
                        }
                    } else {
                        self.phase = .inObject(.expectCommaOrEnd)
                    }
                }
                
            case "\"":
                self.phase = .inString(.body(kind: .key, escaped: false))
                currentKey = ""
                
            case " ", "\t", "\n", "\r":
                break  // Skip whitespace
                
            default:
                self.phase = .error
            }
            
        case .expectKeyFirstQuote:
            switch char {
            case "\"":
                self.phase = .inString(.body(kind: .key, escaped: false))
                currentKey = ""  // Reset key buffer when starting a new key
                
            case " ", "\t", "\n", "\r":
                break  // Skip whitespace
                
            default:
                self.phase = .error
            }
            
        case .inKey:
            // This state is handled by processStringCharacter
            self.phase = .error
            
        case .expectKeyEndQuote:
            switch char {
            case "\"":
                self.phase = .inObject(.expectColon)
                
            default:
                self.phase = .error
            }
            
        case .expectColon:
            switch char {
            case ":":
                self.phase = .inObject(.expectValue)
                
            case " ", "\t", "\n", "\r":
                break  // Skip whitespace
                
            default:
                self.phase = .error
            }
            
        case .expectValue:
            processValueStart(char, fromArray: false)
            
        case .expectCommaOrEnd:
            switch char {
            case ",":
                self.phase = .inObject(.expectKeyFirstQuote)
                
            case "}":
                nestingLevel -= 1
                if nestingLevel == 0 {
                    self.phase = .done
                    isGenerating = false
                } else {
                    // Pop and return to parent context
                    if let parentContext = contextStack.popLast() {
                        // When returning to an array after completing an object value,
                        // we should be expecting comma or end, not another value
                        switch parentContext {
                        case .inArray(.expectValue):
                            self.phase = .inArray(.expectCommaOrEnd)
                        default:
                            self.phase = parentContext
                        }
                    } else {
                        self.phase = .inObject(.expectCommaOrEnd)
                    }
                }
                
            case " ", "\t", "\n", "\r":
                break  // Skip whitespace
                
            default:
                self.phase = .error
            }
        }
    }
    
    private mutating func processArrayCharacter(_ char: Character, phase arrPhase: ArrayPhase) {
        switch arrPhase {
        case .expectValue:
            if char == "]" {
                nestingLevel -= 1
                if nestingLevel == 0 {
                    self.phase = .done
                    isGenerating = false
                } else {
                    // Empty array - pop and return to parent context
                    if let parentContext = contextStack.popLast() {
                        // When returning to an object after completing an array value,
                        // we should be expecting comma or end, not another value
                        switch parentContext {
                        case .inObject(.expectValue):
                            self.phase = .inObject(.expectCommaOrEnd)
                        default:
                            self.phase = parentContext
                        }
                    } else {
                        self.phase = .inArray(.expectCommaOrEnd)
                    }
                }
            } else {
                processValueStart(char, fromArray: true)
            }
            
        case .expectCommaOrEnd:
            switch char {
            case ",":
                self.phase = .inArray(.expectValue)
                
            case "]":
                nestingLevel -= 1
                if nestingLevel == 0 {
                    self.phase = .done
                    isGenerating = false
                } else {
                    // Pop and return to parent context
                    if let parentContext = contextStack.popLast() {
                        // When returning to an object after completing an array value,
                        // we should be expecting comma or end, not another value
                        switch parentContext {
                        case .inObject(.expectValue):
                            self.phase = .inObject(.expectCommaOrEnd)
                        default:
                            self.phase = parentContext
                        }
                    } else {
                        self.phase = .inArray(.expectCommaOrEnd)
                    }
                }
                
            case " ", "\t", "\n", "\r":
                break  // Skip whitespace
                
            default:
                self.phase = .error
            }
        }
    }
    
    private mutating func processStringCharacter(_ char: Character, phase strPhase: StringPhase) {
        switch strPhase {
        case .body(let kind, let escaped):
            if escaped {
                // Add character to buffer and continue
                if kind == .key {
                    currentKey.append(char)
                }
                self.phase = .inString(.body(kind: kind, escaped: false))
            } else if char == "\\" {
                self.phase = .inString(.body(kind: kind, escaped: true))
            } else if char == "\"" {
                // String ended
                if kind == .key {
                    self.phase = .inObject(.expectColon)
                } else {
                    // Value string ended, determine next phase based on context
                    // We need to check if we're directly in an object or an array
                    // The key indicator: if we had a key before this value, we're in an object
                    if nestingLevel > 0 {
                        if !currentKey.isEmpty {
                            // We're in an object (had a key before this value)
                            self.phase = .inObject(.expectCommaOrEnd)
                            currentKey = "" // Clear the key after processing the value
                        } else {
                            // No key means we're in an array
                            self.phase = .inArray(.expectCommaOrEnd)
                        }
                    } else {
                        self.phase = .done
                        isGenerating = false
                    }
                }
            } else {
                // Regular character
                if kind == .key {
                    currentKey.append(char)
                }
            }
        }
    }
    
    private mutating func processNumberCharacter(_ char: Character, phase numPhase: NumberPhase) {
        switch numPhase {
        case .integer:
            switch char {
            case "0"..."9":
                buffer.append(char)
                
            case ".":
                buffer.append(char)
                self.phase = .inNumber(.decimal)
                
            case "e", "E":
                buffer.append(char)
                self.phase = .inNumber(.exponent)
                
            case ",", "}", "]", " ", "\t", "\n", "\r":
                // Number ended
                buffer = ""
                processNumberEnd(char)
                
            default:
                self.phase = .error
            }
            
        case .decimal:
            switch char {
            case "0"..."9":
                buffer.append(char)
                
            case "e", "E":
                buffer.append(char)
                self.phase = .inNumber(.exponent)
                
            case ",", "}", "]", " ", "\t", "\n", "\r":
                // Number ended
                buffer = ""
                processNumberEnd(char)
                
            default:
                self.phase = .error
            }
            
        case .exponent:
            switch char {
            case "0"..."9", "+", "-":
                buffer.append(char)
                
            case ",", "}", "]", " ", "\t", "\n", "\r":
                // Number ended
                buffer = ""
                processNumberEnd(char)
                
            default:
                self.phase = .error
            }
        }
    }
    
    private mutating func processLiteralCharacter(_ char: Character, phase litPhase: LiteralPhase) {
        switch litPhase {
        case .inProgress(let literal):
            let expected: String
            switch literal {
            case "t": expected = "true"
            case "f": expected = "false"
            case "n": expected = "null"
            default: 
                phase = .error
                return
            }
            
            if literal.count < expected.count {
                let nextIndex = literal.index(literal.startIndex, offsetBy: literal.count)
                if expected[nextIndex] == char {
                    let newLiteral = literal + String(char)
                    if newLiteral == expected {
                        // Literal complete - determine next phase based on context
                        if nestingLevel > 0 {
                            if let parentContext = contextStack.last {
                                switch parentContext {
                                case .inArray:
                                    self.phase = .inArray(.expectCommaOrEnd)
                                case .inObject:
                                    self.phase = .inObject(.expectCommaOrEnd)
                                default:
                                    self.phase = .inObject(.expectCommaOrEnd)
                                }
                            } else {
                                self.phase = .inObject(.expectCommaOrEnd)
                            }
                        } else {
                            self.phase = .done
                            isGenerating = false
                        }
                    } else {
                        self.phase = .inLiteral(.inProgress(newLiteral))
                    }
                } else {
                    self.phase = .error
                }
            } else {
                processLiteralEnd(char)
            }
        }
    }
    
    private mutating func processValueStart(_ char: Character, fromArray: Bool) {
        switch char {
        case "{":
            // Push current context before entering new object
            if nestingLevel > 0 {
                // We're already inside something, push the current context
                contextStack.append(phase)
            }
            nestingLevel += 1
            phase = .inObject(.expectKeyOrEnd)
            
        case "[":
            // Push current context before entering new array
            if nestingLevel > 0 {
                // We're already inside something, push the current context
                contextStack.append(phase)
            }
            nestingLevel += 1
            phase = .inArray(.expectValue)
            
        case "\"":
            phase = .inString(.body(kind: .value, escaped: false))
            
        case "-", "0"..."9":
            phase = .inNumber(.integer)
            buffer = String(char)
            
        case "t", "f", "n":
            phase = .inLiteral(.inProgress(String(char)))
            
        case " ", "\t", "\n", "\r":
            break  // Skip whitespace
            
        default:
            phase = .error
        }
    }
    
    private mutating func processNumberEnd(_ char: Character) {
        if nestingLevel > 0 {
            // We need to determine if we're directly in an array or an object
            // Check the current phase context, not the stack
            // If we were processing a number after "id":1, we're in an object
            
            // Look at what comes after the number to determine context
            switch char {
            case ",":
                // Could be in object or array - need to check our immediate context
                // If we have an object on top, we're in object context
                if case .inObject = phase {
                    phase = .inObject(.expectCommaOrEnd)
                    processObjectCharacter(char, phase: .expectCommaOrEnd)
                } else {
                    // We must determine from parsing context
                    // Numbers in objects follow colons, in arrays they don't
                    // For now, assume object context in nested structures
                    phase = .inObject(.expectCommaOrEnd)
                    processObjectCharacter(char, phase: .expectCommaOrEnd)
                }
            case "}":
                // Ending an object
                phase = .inObject(.expectCommaOrEnd)
                processObjectCharacter(char, phase: .expectCommaOrEnd)
            case "]":
                // Ending an array
                phase = .inArray(.expectCommaOrEnd)
                processArrayCharacter(char, phase: .expectCommaOrEnd)
            default:
                // Default to object context
                phase = .inObject(.expectCommaOrEnd)
                processObjectCharacter(char, phase: .expectCommaOrEnd)
            }
        } else {
            phase = .done
            isGenerating = false
        }
    }
    
    private mutating func processLiteralEnd(_ char: Character) {
        if nestingLevel > 0 {
            // Determine parent context
            if let parentContext = contextStack.last {
                switch parentContext {
                case .inArray:
                    phase = .inArray(.expectCommaOrEnd)
                    processArrayCharacter(char, phase: .expectCommaOrEnd)
                case .inObject:
                    phase = .inObject(.expectCommaOrEnd)
                    processObjectCharacter(char, phase: .expectCommaOrEnd)
                default:
                    phase = .inObject(.expectCommaOrEnd)
                    processObjectCharacter(char, phase: .expectCommaOrEnd)
                }
            } else {
                phase = .inObject(.expectCommaOrEnd)
                processObjectCharacter(char, phase: .expectCommaOrEnd)
            }
        } else {
            phase = .done
            isGenerating = false
        }
    }
}