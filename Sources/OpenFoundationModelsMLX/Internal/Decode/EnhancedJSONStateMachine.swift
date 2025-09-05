import Foundation

// Enhanced JSON state machine for token-based Schema-Constrained Decoding
// Tracks JSON parsing state at token level and manages key path during generation
struct EnhancedJSONStateMachine: Sendable {
    
    // Detailed phases for JSON parsing
    enum Phase: Sendable, Equatable {
        case outside            // Not in JSON structure
        case expectOpenBrace    // Waiting for '{'
        case inObject          // Inside object, may expect key or '}'
        case expectKey         // After '{' or ',', waiting for key
        case inKey            // Inside key string (between quotes)
        case expectColon      // After key closing quote, waiting for ':'
        case inValue         // After ':', parsing value
        case expectCommaOrClose // After value, waiting for ',' or '}'
    }
    
    // Context information for current parsing state
    struct Context: Sendable {
        let phase: Phase
        let depth: Int
        let currentKey: String
        let isInArray: Bool
        let bracketStack: [Character]
    }
    
    private(set) var phase: Phase = .outside
    private(set) var depth: Int = 0
    private(set) var currentKey: String = ""
    private(set) var keyBuffer: String = ""
    private var bracketStack: [Character] = []
    private var isInString: Bool = false
    private var escapeNext: Bool = false
    
    init() {}
    
    // MARK: - State Transitions
    
    mutating func processToken(_ tokenID: Int32, tokenizer: TokenizerAdapter) {
        let text = tokenizer.decode([tokenID])
        processText(text, tokenID: tokenID)
    }
    
    mutating func processText(_ text: String, tokenID: Int32? = nil) {
        // Process each character for state transitions
        // Note: tokenID is ignored here as token path tracking is handled by TokenTrieLogitProcessor
        for char in text {
            processCharacter(char)
        }
    }
    
    private mutating func processCharacter(_ char: Character) {
        // Handle escape sequences
        if escapeNext {
            escapeNext = false
            if phase == .inKey {
                keyBuffer.append(char)
            }
            return
        }
        
        if char == "\\" && (phase == .inKey || phase == .inValue) {
            escapeNext = true
            return
        }
        
        // Main state transition logic
        switch phase {
        case .outside:
            if char == "{" {
                phase = .expectKey
                depth += 1
                bracketStack.append("{")
            } else if char == "\"" && depth > 0 {
                // Starting a key after comma
                phase = .inKey
                keyBuffer = ""
            }
            
        case .expectOpenBrace:
            if char == "{" {
                phase = .expectKey
                depth += 1
                bracketStack.append("{")
            }
            
        case .inObject, .expectKey:
            if char == "\"" {
                phase = .inKey
                keyBuffer = ""
            } else if char == "}" && depth > 0 {
                depth -= 1
                _ = bracketStack.popLast()
                phase = depth > 0 ? .expectCommaOrClose : .outside
            }
            
        case .inKey:
            if char == "\"" && !escapeNext {
                // Key completed
                currentKey = keyBuffer
                phase = .expectColon
            } else {
                keyBuffer.append(char)
            }
            
        case .expectColon:
            if char == ":" {
                phase = .inValue
            }
            
        case .inValue:
            // Track value parsing (simplified - would need full JSON parsing)
            if char == "\"" {
                isInString.toggle()
            } else if !isInString {
                if char == "{" {
                    depth += 1
                    bracketStack.append("{")
                } else if char == "[" {
                    bracketStack.append("[")
                } else if char == "}" {
                    if bracketStack.last == "{" {
                        _ = bracketStack.popLast()
                        depth -= 1
                        phase = depth > 0 ? .expectCommaOrClose : .outside
                    }
                } else if char == "]" {
                    if bracketStack.last == "[" {
                        _ = bracketStack.popLast()
                    }
                } else if char == "," && bracketStack.last == "{" {
                    phase = .expectKey
                }
            }
            
        case .expectCommaOrClose:
            if char == "," {
                phase = .expectKey
            } else if char == "}" {
                depth -= 1
                _ = bracketStack.popLast()
                phase = depth > 0 ? .expectCommaOrClose : .outside
            }
        }
    }
    
    // MARK: - Query Methods
    
    func isInKeyEmission() -> Bool {
        return phase == .inKey
    }
    
    func isExpectingColon() -> Bool {
        return phase == .expectColon
    }
    
    func isOutside() -> Bool {
        return phase == .outside
    }
    
    func getCurrentContext() -> Context {
        return Context(
            phase: phase,
            depth: depth,
            currentKey: currentKey,
            isInArray: bracketStack.contains("["),
            bracketStack: bracketStack
        )
    }
    
    // MARK: - Reset
    
    mutating func reset() {
        phase = .outside
        depth = 0
        currentKey = ""
        keyBuffer = ""
        bracketStack.removeAll()
        isInString = false
        escapeNext = false
    }
    
    // MARK: - Simplified Interface (backward compatibility)
    
    mutating func onPieceEmitted(_ piece: String) {
        processText(piece)
    }
}

// MARK: - Token-aware extensions

extension EnhancedJSONStateMachine {
    
    // Process token with special handling for JSON symbols
    mutating func processTokenWithSpecialHandling(
        _ tokenID: Int32,
        tokenizer: TokenizerAdapter,
        specialTokens: MLXLLMTokenizer.SpecialTokens?
    ) {
        // Check if this is a special JSON token
        if let special = specialTokens {
            if special.quoteTokens.contains(tokenID) {
                // Handle quote token specially
                if phase == .expectKey || phase == .inObject {
                    phase = .inKey
                    keyBuffer = ""
                } else if phase == .inKey {
                    currentKey = keyBuffer
                    phase = .expectColon
                }
                return
            }
            
            if special.colonTokens.contains(tokenID) && phase == .expectColon {
                phase = .inValue
                return
            }
            
            if special.braceOpenTokens.contains(tokenID) {
                if phase == .outside || phase == .inValue {
                    depth += 1
                    bracketStack.append("{")
                    phase = .expectKey
                }
                return
            }
            
            if special.commaTokens.contains(tokenID) && phase == .expectCommaOrClose {
                phase = .expectKey
                return
            }
        }
        
        // Default processing
        processToken(tokenID, tokenizer: tokenizer)
    }
}