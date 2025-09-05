import Foundation

// Constraint sampler that applies token-level constraints using TokenTrie
// for Schema-Constrained Decoding (SCD)
struct ConstrainedSampler: Sendable {
    let tokenTrie: TokenTrie
    let specialTokens: MLXLLMTokenizer.SpecialTokens?
    
    // Fallback to character-based trie if needed
    let keyTrie: KeyTrie?

    init(tokenTrie: TokenTrie, specialTokens: MLXLLMTokenizer.SpecialTokens? = nil, keyTrie: KeyTrie? = nil) {
        self.tokenTrie = tokenTrie
        self.specialTokens = specialTokens
        self.keyTrie = keyTrie
    }

    enum Decision: Equatable { 
        case ok(allowed: Set<Int32>)
        case noConstraint  // No constraints applied
        case noCandidate   // No valid candidates - trigger retry
    }

    // Get allowed tokens based on current state and path
    func allowedTokens(
        state: EnhancedJSONStateMachine.Phase,
        tokenPath: TokenTrie.Path,
        keyPrefix: String = ""
    ) -> Decision {
        switch state {
        case .outside, .expectOpenBrace, .inObject, .expectKey, .inValue, .expectCommaOrClose:
            // No constraints outside of key emission
            return .noConstraint
            
        case .inKey:
            // Get allowed tokens from TokenTrie
            let allowedIds = tokenTrie.getAllowedTokens(for: tokenPath)
            
            if allowedIds.isEmpty {
                // No valid continuations - this triggers retry
                return .noCandidate
            }
            
            var finalAllowed = allowedIds
            
            // If at terminal node, also allow quote tokens to close the key
            if tokenPath.isAtTerminal(), let quotes = specialTokens?.quoteTokens {
                finalAllowed.formUnion(quotes)
            }
            
            return .ok(allowed: finalAllowed)
            
        case .expectColon:
            // Only colon tokens allowed
            if let colons = specialTokens?.colonTokens, !colons.isEmpty {
                return .ok(allowed: colons)
            } else {
                // If no specific colon tokens, allow generation to continue
                return .noConstraint
            }
        }
    }
    
    // Fallback to character-based validation when TokenTrie not available
    func allowedTokensFallback(
        state: JSONStateMachine.Phase,
        normalizedKeyPrefix: String
    ) -> Decision {
        guard let trie = keyTrie else {
            return .noConstraint
        }
        
        switch state {
        case .outside:
            return .noConstraint
        case .inKey:
            let hasPrefix = trie.hasPrefix(SchemaSnapParser.normalize(normalizedKeyPrefix))
            if !hasPrefix { 
                return .noCandidate 
            }
            // In fallback mode, we can't provide specific token IDs
            return .noConstraint
        case .expectColon:
            if let colons = specialTokens?.colonTokens, !colons.isEmpty {
                return .ok(allowed: colons)
            }
            return .noConstraint
        }
    }

    // Apply logits mask to enforce constraints
    static func applyMask(logits: inout [Float], allowed: Set<Int32>) {
        guard !allowed.isEmpty else { return }
        
        let negInf: Float = -.infinity
        
        // Efficient masking: set all to -inf first, then restore allowed
        if allowed.count < logits.count / 2 {
            // If allowed set is small, iterate through all and mask
            for i in 0..<logits.count {
                if !allowed.contains(Int32(i)) {
                    logits[i] = negInf
                }
            }
        } else {
            // If allowed set is large, save and restore
            let savedLogits = logits
            logits = Array(repeating: negInf, count: logits.count)
            for tokenID in allowed {
                let idx = Int(tokenID)
                if idx >= 0 && idx < logits.count {
                    logits[idx] = savedLogits[idx]
                }
            }
        }
    }
    
    // Alternative masking that preserves relative probabilities
    static func applySoftMask(logits: inout [Float], allowed: Set<Int32>, penalty: Float = -100.0) {
        guard !allowed.isEmpty else { return }
        
        for i in 0..<logits.count {
            if !allowed.contains(Int32(i)) {
                logits[i] += penalty // Soft penalty instead of -inf
            }
        }
    }
}

