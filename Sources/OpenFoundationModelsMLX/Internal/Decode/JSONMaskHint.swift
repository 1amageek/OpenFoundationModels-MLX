import Foundation

// Mask hint generation for JSON state machine
// Provides allowed token sets based on current parsing state
public struct JSONMaskHint: Sendable {
    
    public enum Mode: Sendable {
        case hard    // Complete mask (disallow everything else)
        case soft    // Weight adjustment (prefer but don't require)
    }
    
    public let allow: Set<Int32>       // Allowed token IDs
    public let prefer: Set<Int32>      // Preferred token IDs (optional)
    public let mode: Mode              // Masking mode
    
    public init(allow: Set<Int32>, prefer: Set<Int32> = [], mode: Mode = .hard) {
        self.allow = allow
        self.prefer = prefer
        self.mode = mode
    }
}

// MARK: - Mask Hint Generator

public struct JSONMaskHintGenerator {
    
    private let specialTokens: MLXLLMTokenizer.SpecialTokens
    private let includeWhitespace: Bool
    
    public init(specialTokens: MLXLLMTokenizer.SpecialTokens, includeWhitespace: Bool = true) {
        self.specialTokens = specialTokens
        self.includeWhitespace = includeWhitespace
    }
    
    /// Generate mask hint for current JSON state
    func maskHint(
        for state: JSONStateMachine,
        tokenTrie: TokenTrie? = nil,
        tokenPath: TokenTrie.Path? = nil
    ) -> JSONMaskHint? {
        
        let phase = state.phase
        
        // Add whitespace tokens to all allowed sets if enabled
        let ws = includeWhitespace ? specialTokens.whitespaceTokens : Set<Int32>()
        
        switch phase {
        case .root:
            // Allow any value start
            let allowed = specialTokens.quoteTokens
                .union(specialTokens.braceOpenTokens)
                .union(specialTokens.bracketOpenTokens)
                .union(ws)
            // Note: Numbers and literals need unrestricted tokens, so we return soft mode
            return JSONMaskHint(allow: allowed, mode: .soft)
            
        case .inObject(let objPhase):
            return maskHintForObject(objPhase: objPhase, ws: ws, tokenTrie: tokenTrie, tokenPath: tokenPath)
            
        case .inArray(let arrPhase):
            return maskHintForArray(arrPhase: arrPhase, ws: ws)
            
        case .inString(let strPhase):
            return maskHintForString(strPhase: strPhase, tokenTrie: tokenTrie, tokenPath: tokenPath)
            
        case .inNumber, .inLiteral:
            // Numbers and literals are too complex to constrain at token level
            // Allow unrestricted generation
            return nil
            
        case .done:
            // After JSON completion, only allow EOS token to prevent tail garbage
            // The EOS token will be added by the applyMask function in TokenTrieLogitProcessor
            // so we return an empty set with hard mode to block everything except EOS
            return JSONMaskHint(allow: [], mode: .hard)
            
        case .error:
            // No generation in error state
            return JSONMaskHint(allow: [])
        }
    }
    
    // MARK: - Object State Hints
    
    private func maskHintForObject(
        objPhase: JSONStateMachine.ObjectPhase,
        ws: Set<Int32>,
        tokenTrie: TokenTrie?,
        tokenPath: TokenTrie.Path?
    ) -> JSONMaskHint? {
        
        switch objPhase {
        case .expectKeyFirstQuote:
            // Allow quote to start key or } to close object
            let allowed = specialTokens.quoteTokens
                .union(specialTokens.braceCloseTokens)
                .union(ws)
            return JSONMaskHint(allow: allowed)
            
        case .expectColon:
            // Only colon allowed (plus whitespace)
            let allowed = specialTokens.colonTokens.union(ws)
            return JSONMaskHint(allow: allowed)
            
        case .expectValueStart:
            // Allow any value start
            let allowed = specialTokens.quoteTokens
                .union(specialTokens.braceOpenTokens)
                .union(specialTokens.bracketOpenTokens)
                .union(ws)
            // Soft mode because numbers/literals need unrestricted tokens
            return JSONMaskHint(allow: allowed, mode: .soft)
            
        case .afterValue:
            // Allow comma or closing brace
            let allowed = specialTokens.commaTokens
                .union(specialTokens.braceCloseTokens)
                .union(ws)
            return JSONMaskHint(allow: allowed)
        }
    }
    
    // MARK: - Array State Hints
    
    private func maskHintForArray(
        arrPhase: JSONStateMachine.ArrayPhase,
        ws: Set<Int32>
    ) -> JSONMaskHint? {
        
        switch arrPhase {
        case .expectValue:
            // Allow any value start or ] to close array
            let allowed = specialTokens.quoteTokens
                .union(specialTokens.braceOpenTokens)
                .union(specialTokens.bracketOpenTokens)
                .union(specialTokens.bracketCloseTokens)
                .union(ws)
            // Soft mode for numbers/literals
            return JSONMaskHint(allow: allowed, mode: .soft)
            
        case .afterValue:
            // Allow comma or closing bracket
            let allowed = specialTokens.commaTokens
                .union(specialTokens.bracketCloseTokens)
                .union(ws)
            return JSONMaskHint(allow: allowed)
        }
    }
    
    // MARK: - String State Hints
    
    private func maskHintForString(
        strPhase: JSONStateMachine.StringPhase,
        tokenTrie: TokenTrie?,
        tokenPath: TokenTrie.Path?
    ) -> JSONMaskHint? {
        
        switch strPhase {
        case .body(let kind, _):
            if kind == .key {
                // Apply TokenTrie constraints for key strings
                if let trie = tokenTrie, let path = tokenPath {
                    var allowed = trie.getAllowedTokens(for: path)
                    
                    // Always allow backslash for escape sequences in keys
                    allowed.formUnion(specialTokens.backslashTokens)
                    
                    // If at terminal, also allow quote to close the key
                    if path.isAtTerminal() {
                        allowed.formUnion(specialTokens.quoteTokens)
                    }
                    
                    // If no valid continuations, fall back to quotes and backslash
                    if allowed.isEmpty {
                        allowed = specialTokens.quoteTokens.union(specialTokens.backslashTokens)
                    }
                    
                    return JSONMaskHint(allow: allowed)
                }
            }
            // For value strings or when no trie available, allow any tokens
            // (string content is unrestricted except for control characters)
            return nil
            
        case .unicode:
            // Unicode escape sequences need hex digits - too complex for token mask
            return nil
        }
    }
}

// MARK: - Integration with TokenTrieLogitProcessor

extension JSONMaskHintGenerator {
    
    /// Create a specialized generator for schema-constrained decoding
    public static func forSchemaConstrainedDecoding(
        specialTokens: MLXLLMTokenizer.SpecialTokens,
        includeWhitespace: Bool = false  // Stricter by default for SCD
    ) -> JSONMaskHintGenerator {
        return JSONMaskHintGenerator(
            specialTokens: specialTokens,
            includeWhitespace: includeWhitespace
        )
    }
    
    /// Check if a token violates current constraints
    func isViolation(
        tokenID: Int32,
        state: JSONStateMachine,
        tokenTrie: TokenTrie? = nil,
        tokenPath: TokenTrie.Path? = nil
    ) -> Bool {
        guard let hint = maskHint(for: state, tokenTrie: tokenTrie, tokenPath: tokenPath) else {
            // No constraints - not a violation
            return false
        }
        
        if hint.mode == .hard {
            // Hard mode: must be in allowed set
            return !hint.allow.contains(tokenID)
        } else {
            // Soft mode: violations only for definitely wrong tokens
            // (This could be refined based on specific rules)
            return false
        }
    }
}