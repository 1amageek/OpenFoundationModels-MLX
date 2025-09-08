import Foundation

// Complete JSON state machine for Schema-Constrained Decoding (SCD)
// Based on comprehensive design specification for token-level JSON generation
public struct JSONStateMachine: Sendable {
    
    // MARK: - Core Phase Definitions
    
    public enum Phase: Sendable, Equatable {
        case root                        // Top-level value start/reading
        case inObject(ObjectPhase)       // Inside object
        case inArray(ArrayPhase)         // Inside array
        case inString(StringPhase)       // Key or value string
        case inNumber(NumberPhase)       // Numeric value
        case inLiteral(LiteralPhase)     // true/false/null
        case done                        // Root JSON complete (only WS allowed)
        case error                       // Invalid state (recovery/abort decision)
    }
    
    // MARK: - Container Phases
    
    public enum ObjectPhase: Sendable, Equatable {
        case expectKeyFirstQuote         // Expecting " or }
        case expectColon                 // After key, expecting :
        case expectValueStart            // After :, expecting value
        case afterValue                  // After value, expecting , or }
    }
    
    public enum ArrayPhase: Sendable, Equatable {
        case expectValue                 // Expecting value or ]
        case afterValue                  // After value, expecting , or ]
    }
    
    // MARK: - Value Phases
    
    public enum StringKind: Sendable, Equatable {
        case key                         // Object key string
        case value                       // Value string
    }
    
    public enum StringPhase: Sendable, Equatable {
        case body(kind: StringKind, escaped: Bool)         // String content
        case unicode(kind: StringKind, remaining: Int)     // \uXXXX processing
    }
    
    public enum NumberPhase: Sendable, Equatable {
        case start                       // Just saw - or digit
        case afterMinus                  // After -, need 0-9
        case intZero                     // Leading 0 (no more digits allowed)
        case intNonZero                  // 1-9 sequence
        case fracStart                   // After ., need digit
        case frac                        // Fractional digits
        case expStart                    // After e/E, need +/-/digit
        case expSign                     // After +/-, need digit
        case exp                         // Exponent digits
    }
    
    public enum LiteralPhase: Sendable, Equatable {
        case t1, t2, t3                  // 't', 'tr', 'tru'
        case f1, f2, f3, f4              // 'f', 'fa', 'fal', 'fals'
        case n1, n2, n3                  // 'n', 'nu', 'nul'
        case done                        // Literal complete
    }
    
    // MARK: - Container Frame Stack
    
    public struct ObjectFrame: Sendable {
        var expectingKey: Bool = true
        var expectingColon: Bool = false
        var afterValue: Bool = false
    }
    
    public struct ArrayFrame: Sendable {
        var expectingValue: Bool = true
        var afterValue: Bool = false
    }
    
    public enum Frame: Sendable {
        case object(ObjectFrame)
        case array(ArrayFrame)
    }
    
    // MARK: - State Context
    
    public struct Context: Sendable {
        public let phase: Phase
        public let stack: [Frame]
        public let depth: Int
        public let sawTopValue: Bool
        public let currentKey: String
        public let keyBuffer: String
        public let violationCount: Int
    }
    
    // MARK: - Instance State
    
    private(set) public var phase: Phase = .root
    private(set) public var stack: [Frame] = []
    private(set) public var depth: Int = 0                    // { nesting level
    private(set) public var sawTopValue: Bool = false         // Root value started
    private(set) public var currentKey: String = ""           // Last completed key
    private(set) public var keyBuffer: String = ""            // Key being built
    private(set) public var violationCount: Int = 0           // Consecutive violations
    private var lastSignificant: Character?                   // Last non-WS character
    
    public init() {}
    
    // MARK: - State Transitions
    
    public mutating func processCharacter(_ char: Character) {
        // Track significant characters (non-whitespace)
        if !char.isWhitespace {
            lastSignificant = char
        }
        
        // Handle based on current phase
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
            
        case .done:
            // Only whitespace allowed after completion
            if !char.isWhitespace {
                phase = .error
            }
            
        case .error:
            // Stay in error state
            break
        }
    }
    
    // MARK: - Root Processing
    
    private mutating func processRootCharacter(_ char: Character) {
        guard !char.isWhitespace else { return }
        
        switch char {
        case "{":
            stack.append(.object(ObjectFrame()))
            phase = .inObject(.expectKeyFirstQuote)
            depth += 1
            sawTopValue = true
            
        case "[":
            stack.append(.array(ArrayFrame()))
            phase = .inArray(.expectValue)
            sawTopValue = true
            
        case "\"":
            phase = .inString(.body(kind: .value, escaped: false))
            sawTopValue = true
            
        case "-", "0"..."9":
            startNumber(with: char)
            sawTopValue = true
            
        case "t":
            phase = .inLiteral(.t1)
            sawTopValue = true
            
        case "f":
            phase = .inLiteral(.f1)
            sawTopValue = true
            
        case "n":
            phase = .inLiteral(.n1)
            sawTopValue = true
            
        default:
            phase = .error
        }
    }
    
    // MARK: - Object Processing
    
    private mutating func processObjectCharacter(_ char: Character, phase objPhase: ObjectPhase) {
        switch objPhase {
        case .expectKeyFirstQuote:
            guard !char.isWhitespace else { return }
            
            switch char {
            case "\"":
                self.phase = .inString(.body(kind: .key, escaped: false))
                keyBuffer = ""
                violationCount = 0
                
            case "}":
                // Empty object or end
                if let frame = stack.last {
                    _ = stack.popLast()
                    depth -= 1
                    transitionAfterContainerEnd(from: frame)
                }
                
            default:
                self.phase = .error
            }
            
        case .expectColon:
            guard !char.isWhitespace else { return }
            
            if char == ":" {
                self.phase = .inObject(.expectValueStart)
            } else {
                self.phase = .error
            }
            
        case .expectValueStart:
            guard !char.isWhitespace else { return }
            
            startValue(char)
            
        case .afterValue:
            guard !char.isWhitespace else { return }
            
            switch char {
            case ",":
                self.phase = .inObject(.expectKeyFirstQuote)
                
            case "}":
                if let frame = stack.last {
                    _ = stack.popLast()
                    depth -= 1
                    transitionAfterContainerEnd(from: frame)
                }
                
            default:
                self.phase = .error
            }
        }
    }
    
    // MARK: - Array Processing
    
    private mutating func processArrayCharacter(_ char: Character, phase arrPhase: ArrayPhase) {
        switch arrPhase {
        case .expectValue:
            guard !char.isWhitespace else { return }
            
            if char == "]" {
                // Empty array or end
                if let frame = stack.last {
                    _ = stack.popLast()
                    transitionAfterContainerEnd(from: frame)
                }
            } else {
                startValue(char)
            }
            
        case .afterValue:
            guard !char.isWhitespace else { return }
            
            switch char {
            case ",":
                self.phase = .inArray(.expectValue)
                
            case "]":
                if let frame = stack.last {
                    _ = stack.popLast()
                    transitionAfterContainerEnd(from: frame)
                }
                
            default:
                self.phase = .error
            }
        }
    }
    
    // MARK: - String Processing
    
    private mutating func processStringCharacter(_ char: Character, phase strPhase: StringPhase) {
        switch strPhase {
        case .body(let kind, let escaped):
            if escaped {
                // Process escaped character
                switch char {
                case "u":
                    self.phase = .inString(.unicode(kind: kind, remaining: 4))
                default:
                    // Regular escape sequence - continue string
                    if kind == .key {
                        keyBuffer.append(char)
                    }
                    self.phase = .inString(.body(kind: kind, escaped: false))
                }
            } else if char == "\\" {
                // Start escape sequence
                self.phase = .inString(.body(kind: kind, escaped: true))
            } else if char == "\"" {
                // String complete
                if kind == .key {
                    currentKey = keyBuffer
                    self.phase = .inObject(.expectColon)
                } else {
                    transitionAfterValue()
                }
            } else {
                // Regular character
                if kind == .key {
                    keyBuffer.append(char)
                }
            }
            
        case .unicode(let kind, let remaining):
            // Process Unicode hex digit
            if char.isHexDigit {
                if kind == .key {
                    keyBuffer.append(char)
                }
                
                if remaining > 1 {
                    self.phase = .inString(.unicode(kind: kind, remaining: remaining - 1))
                } else {
                    // Unicode sequence complete
                    self.phase = .inString(.body(kind: kind, escaped: false))
                }
            } else {
                self.phase = .error
            }
        }
    }
    
    // MARK: - Number Processing
    
    private mutating func processNumberCharacter(_ char: Character, phase numPhase: NumberPhase) {
        switch numPhase {
        case .start:
            // Should not reach here - start is transient
            self.phase = .error
            
        case .afterMinus:
            if char >= "0" && char <= "9" {
                self.phase = char == "0" ? .inNumber(.intZero) : .inNumber(.intNonZero)
            } else {
                self.phase = .error
            }
            
        case .intZero:
            // After leading 0, only . or e/E allowed
            switch char {
            case ".":
                self.phase = .inNumber(.fracStart)
            case "e", "E":
                self.phase = .inNumber(.expStart)
            default:
                // Number complete - check if valid terminator
                if isValueTerminator(char) {
                    processCharacterAfterValueEnd(char)
                } else {
                    self.phase = .error
                }
            }
            
        case .intNonZero:
            switch char {
            case "0"..."9":
                // Continue integer part
                break
            case ".":
                self.phase = .inNumber(.fracStart)
            case "e", "E":
                self.phase = .inNumber(.expStart)
            default:
                // Number complete
                if isValueTerminator(char) {
                    processCharacterAfterValueEnd(char)
                } else {
                    self.phase = .error
                }
            }
            
        case .fracStart:
            if char >= "0" && char <= "9" {
                self.phase = .inNumber(.frac)
            } else {
                self.phase = .error
            }
            
        case .frac:
            switch char {
            case "0"..."9":
                // Continue fraction
                break
            case "e", "E":
                self.phase = .inNumber(.expStart)
            default:
                // Number complete
                if isValueTerminator(char) {
                    processCharacterAfterValueEnd(char)
                } else {
                    self.phase = .error
                }
            }
            
        case .expStart:
            switch char {
            case "+", "-":
                self.phase = .inNumber(.expSign)
            case "0"..."9":
                self.phase = .inNumber(.exp)
            default:
                self.phase = .error
            }
            
        case .expSign:
            if char >= "0" && char <= "9" {
                self.phase = .inNumber(.exp)
            } else {
                self.phase = .error
            }
            
        case .exp:
            if char >= "0" && char <= "9" {
                // Continue exponent
                break
            } else if isValueTerminator(char) {
                processCharacterAfterValueEnd(char)
            } else {
                self.phase = .error
            }
        }
    }
    
    // MARK: - Literal Processing
    
    private mutating func processLiteralCharacter(_ char: Character, phase litPhase: LiteralPhase) {
        switch litPhase {
        // true
        case .t1 where char == "r":
            self.phase = .inLiteral(.t2)
        case .t2 where char == "u":
            self.phase = .inLiteral(.t3)
        case .t3 where char == "e":
            self.phase = .inLiteral(.done)
            
        // false
        case .f1 where char == "a":
            self.phase = .inLiteral(.f2)
        case .f2 where char == "l":
            self.phase = .inLiteral(.f3)
        case .f3 where char == "s":
            self.phase = .inLiteral(.f4)
        case .f4 where char == "e":
            self.phase = .inLiteral(.done)
            
        // null
        case .n1 where char == "u":
            self.phase = .inLiteral(.n2)
        case .n2 where char == "l":
            self.phase = .inLiteral(.n3)
        case .n3 where char == "l":
            self.phase = .inLiteral(.done)
            
        case .done:
            // Literal complete - check terminator
            if isValueTerminator(char) {
                processCharacterAfterValueEnd(char)
            } else if !char.isWhitespace {
                self.phase = .error
            }
            
        default:
            self.phase = .error
        }
        
        // Check if we just completed the literal
        if case .inLiteral(.done) = self.phase {
            transitionAfterValue()
        }
    }
    
    // MARK: - Helper Methods
    
    private mutating func startValue(_ char: Character) {
        switch char {
        case "{":
            stack.append(.object(ObjectFrame()))
            phase = .inObject(.expectKeyFirstQuote)
            depth += 1
            
        case "[":
            stack.append(.array(ArrayFrame()))
            phase = .inArray(.expectValue)
            
        case "\"":
            phase = .inString(.body(kind: .value, escaped: false))
            
        case "-", "0"..."9":
            startNumber(with: char)
            
        case "t":
            phase = .inLiteral(.t1)
            
        case "f":
            phase = .inLiteral(.f1)
            
        case "n":
            phase = .inLiteral(.n1)
            
        default:
            phase = .error
        }
    }
    
    private mutating func startNumber(with char: Character) {
        switch char {
        case "-":
            phase = .inNumber(.afterMinus)
        case "0":
            phase = .inNumber(.intZero)
        case "1"..."9":
            phase = .inNumber(.intNonZero)
        default:
            phase = .error
        }
    }
    
    private func isValueTerminator(_ char: Character) -> Bool {
        return char == "," || char == "}" || char == "]" || char.isWhitespace
    }
    
    private mutating func processCharacterAfterValueEnd(_ char: Character) {
        // Transition to appropriate state and reprocess character
        transitionAfterValue()
        if !char.isWhitespace {
            processCharacter(char)
        }
    }
    
    private mutating func transitionAfterValue() {
        // Check what container we're in
        if let last = stack.last {
            switch last {
            case .object:
                phase = .inObject(.afterValue)
            case .array:
                phase = .inArray(.afterValue)
            }
        } else {
            // Top-level value complete
            phase = .done
        }
    }
    
    private mutating func transitionAfterContainerEnd(from frame: Frame) {
        // Container ended - check parent
        if let parent = stack.last {
            switch parent {
            case .object:
                phase = .inObject(.afterValue)
            case .array:
                phase = .inArray(.afterValue)
            }
        } else {
            // Top-level complete
            phase = .done
        }
    }
    
    // MARK: - Query Methods
    
    public func isInKeyEmission() -> Bool {
        if case .inString(.body(kind: .key, _)) = phase {
            return true
        }
        if case .inString(.unicode(kind: .key, _)) = phase {
            return true
        }
        return false
    }
    
    public func isExpectingColon() -> Bool {
        if case .inObject(.expectColon) = phase {
            return true
        }
        return false
    }
    
    public func isComplete() -> Bool {
        return phase == .done
    }
    
    public func isError() -> Bool {
        return phase == .error
    }
    
    public func getCurrentContext() -> Context {
        return Context(
            phase: phase,
            stack: stack,
            depth: depth,
            sawTopValue: sawTopValue,
            currentKey: currentKey,
            keyBuffer: keyBuffer,
            violationCount: violationCount
        )
    }
    
    // MARK: - Violation Tracking
    
    public mutating func recordViolation() {
        violationCount += 1
    }
    
    public mutating func clearViolations() {
        violationCount = 0
    }
    
    // MARK: - Reset
    
    public mutating func reset() {
        phase = .root
        stack.removeAll()
        depth = 0
        sawTopValue = false
        currentKey = ""
        keyBuffer = ""
        violationCount = 0
        lastSignificant = nil
    }
}

// MARK: - Character Extensions

private extension Character {
    var isHexDigit: Bool {
        return ("0"..."9").contains(self) || ("a"..."f").contains(self) || ("A"..."F").contains(self)
    }
}