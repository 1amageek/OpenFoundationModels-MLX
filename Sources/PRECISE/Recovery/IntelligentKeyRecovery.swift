import Foundation

/// Recovers from invalid paths by finding the best schema key match.
/// Uses edit distance and partial matching to suggest corrections.
public final class IntelligentKeyRecovery: PRECISE, @unchecked Sendable {
    private let tokenizer: MLXLLMTokenizer
    private let specialTokens: MLXLLMTokenizer.SpecialTokens
    private var statistics = PRECISEStatistics()
    private let maxEditDistance = 2
    
    public init(tokenizer: MLXLLMTokenizer) {
        self.tokenizer = tokenizer
        self.specialTokens = tokenizer.findSpecialTokens()
    }
    
    // MARK: - PRECISE Protocol
    
    public func validateFuturePaths(
        from path: TokenTrie.Path,
        tokenTrie: TokenTrie,
        depth: Int
    ) -> PathValidation {
        // Delegate to PredictivePathValidator
        return PathValidation(
            hasValidPaths: false,
            tokenScores: [:],
            diagnostics: "Validation delegated to PredictivePathValidator"
        )
    }
    
    public func recoverFromInvalidPath(
        partialKey: String,
        schemaKeys: [String],
        currentTokens: [Int32]
    ) -> RecoveryStrategy {
        let startTime = Date()
        defer {
            let elapsed = Date().timeIntervalSince(startTime)
            statistics.recordValidation(success: true, time: elapsed)
        }
        
        // Find best matching schema key
        let matches = findBestMatches(
            partial: partialKey,
            candidates: schemaKeys
        )
        
        guard let bestMatch = matches.first else {
            statistics.recordRecovery(success: false)
            return .abort(reason: "No viable key matches found")
        }
        
        // Determine recovery strategy based on match quality
        if bestMatch.distance == 0 {
            // Exact prefix match - complete the key
            let remaining = String(bestMatch.key.dropFirst(partialKey.count))
            let completionTokens = tokenizer.encode(remaining)
            statistics.recordRecovery(success: true)
            return .completeToKey(
                target: bestMatch.key,
                completionTokens: completionTokens
            )
        } else if bestMatch.distance <= 1 {
            // Close match - try to correct
            let correctionTokens = computeCorrectionTokens(
                from: partialKey,
                to: bestMatch.key,
                currentTokens: currentTokens
            )
            if !correctionTokens.isEmpty {
                statistics.recordRecovery(success: true)
                return .completeToKey(
                    target: bestMatch.key,
                    completionTokens: correctionTokens
                )
            }
        }
        
        // Try to close current structure and skip
        if let closeTokens = getClosingTokens(partialKey: partialKey) {
            statistics.recordRecovery(success: true)
            return .closeCurrentKey(tokens: closeTokens)
        }
        
        // Last resort: skip to next valid position
        if let skipTokens = computeSkipTokens() {
            statistics.recordRecovery(success: true)
            return .skipToNext(tokens: skipTokens)
        }
        
        statistics.recordRecovery(success: false)
        return .abort(reason: "Unable to recover from invalid path")
    }
    
    public func selectConstraintMode(
        context: GenerationContext
    ) -> ConstraintMode {
        // Delegate to AdaptiveConstraintEngine
        return .hard(allowedTokens: [])
    }
    
    // MARK: - Private Methods
    
    private struct KeyMatch {
        let key: String
        let distance: Int
        let score: Float
    }
    
    private func findBestMatches(
        partial: String,
        candidates: [String]
    ) -> [KeyMatch] {
        var matches: [KeyMatch] = []
        
        for candidate in candidates {
            // Check prefix match
            if candidate.hasPrefix(partial) {
                matches.append(KeyMatch(
                    key: candidate,
                    distance: 0,
                    score: 1.0
                ))
                continue
            }
            
            // Check edit distance
            let distance = editDistance(partial, candidate)
            if distance <= maxEditDistance {
                let score = 1.0 - (Float(distance) / Float(max(partial.count, candidate.count)))
                matches.append(KeyMatch(
                    key: candidate,
                    distance: distance,
                    score: score
                ))
            }
            
            // Check if partial is substring
            if candidate.contains(partial) && partial.count >= 3 {
                matches.append(KeyMatch(
                    key: candidate,
                    distance: candidate.count - partial.count,
                    score: 0.5
                ))
            }
        }
        
        // Sort by score descending
        return matches.sorted { $0.score > $1.score }
    }
    
    private func editDistance(_ s1: String, _ s2: String) -> Int {
        let m = s1.count
        let n = s2.count
        
        guard m > 0 && n > 0 else {
            return max(m, n)
        }
        
        var matrix = Array(repeating: Array(repeating: 0, count: n + 1), count: m + 1)
        
        for i in 0...m {
            matrix[i][0] = i
        }
        for j in 0...n {
            matrix[0][j] = j
        }
        
        let s1Array = Array(s1)
        let s2Array = Array(s2)
        
        for i in 1...m {
            for j in 1...n {
                let cost = s1Array[i-1] == s2Array[j-1] ? 0 : 1
                matrix[i][j] = min(
                    matrix[i-1][j] + 1,      // deletion
                    matrix[i][j-1] + 1,      // insertion
                    matrix[i-1][j-1] + cost  // substitution
                )
            }
        }
        
        return matrix[m][n]
    }
    
    private func computeCorrectionTokens(
        from partial: String,
        to target: String,
        currentTokens: [Int32]
    ) -> [Int32] {
        // Find common prefix
        let commonPrefix = partial.commonPrefix(with: target)
        
        // If we need to backtrack, we can't do it with current architecture
        if commonPrefix.count < partial.count {
            return [] // Cannot backtrack
        }
        
        // Compute tokens for remaining part
        let remaining = String(target.dropFirst(partial.count))
        return tokenizer.encode(remaining)
    }
    
    private func getClosingTokens(partialKey: String) -> [Int32]? {
        // Try to close with quote and move on
        guard let quoteToken = specialTokens.quote else {
            return nil
        }
        
        // Return quote, colon, and a default value
        var tokens: [Int32] = [quoteToken]
        
        if let colonToken = specialTokens.colon {
            tokens.append(colonToken)
        }
        
        // Add a null value as default
        let nullTokens = tokenizer.encode("null")
        tokens.append(contentsOf: nullTokens)
        
        return tokens
    }
    
    private func computeSkipTokens() -> [Int32]? {
        // Generate tokens to skip to next valid position
        // This might mean closing current structure and starting fresh
        var tokens: [Int32] = []
        
        // Close string if needed
        if let quoteToken = specialTokens.quote {
            tokens.append(quoteToken)
        }
        
        // Add comma to move to next field
        if let commaToken = specialTokens.comma {
            tokens.append(commaToken)
        }
        
        return tokens.isEmpty ? nil : tokens
    }
    
    // MARK: - Statistics
    
    public func getStatistics() -> PRECISEStatistics {
        return statistics
    }
    
    public func resetStatistics() {
        statistics = PRECISEStatistics()
    }
}

// MARK: - String Extension

private extension String {
    func commonPrefix(with other: String) -> String {
        let minLength = min(self.count, other.count)
        var prefix = ""
        
        for i in 0..<minLength {
            let index1 = self.index(self.startIndex, offsetBy: i)
            let index2 = other.index(other.startIndex, offsetBy: i)
            
            if self[index1] == other[index2] {
                prefix.append(self[index1])
            } else {
                break
            }
        }
        
        return prefix
    }
}