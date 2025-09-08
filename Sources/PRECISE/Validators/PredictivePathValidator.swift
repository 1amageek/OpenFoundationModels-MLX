import Foundation
@preconcurrency import MLX
import MLXLMCommon

/// Validates future paths from current position to prevent dead-ends in JSON generation.
/// Uses look-ahead analysis to score token candidates based on their viability.
public final class PredictivePathValidator: PRECISE, @unchecked Sendable {
    private let tokenizer: MLXLLMTokenizer
    private let specialTokens: MLXLLMTokenizer.SpecialTokens
    private var statistics = PRECISEStatistics()
    
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
        let startTime = Date()
        defer {
            let elapsed = Date().timeIntervalSince(startTime)
            statistics.recordValidation(success: true, time: elapsed)
        }
        
        // Get current allowed tokens
        let allowedTokens = tokenTrie.getAllowedTokens(for: path)
        guard !allowedTokens.isEmpty else {
            let result = PathValidation(
                hasValidPaths: false,
                tokenScores: [:],
                recommendedToken: nil,
                diagnostics: "No allowed tokens at current path"
            )
            return result
        }
        
        // Score each allowed token based on future viability
        var tokenScores: [Int32: Float] = [:]
        var bestToken: Int32?
        var bestScore: Float = -1.0
        
        for tokenID in allowedTokens {
            let score = evaluateToken(
                tokenID: tokenID,
                currentPath: path,
                tokenTrie: tokenTrie,
                depth: depth
            )
            tokenScores[tokenID] = score
            
            if score > bestScore {
                bestScore = score
                bestToken = tokenID
            }
        }
        
        // Check if terminal is reachable
        let canComplete = tokenTrie.canComplete(from: path)
        if canComplete {
            // Boost score for quote token if we can complete
            if let quoteToken = specialTokens.quote {
                let baseScore = tokenScores[quoteToken] ?? 0.5
                tokenScores[quoteToken] = min(1.0, baseScore + 0.3)
                if tokenScores[quoteToken]! > bestScore {
                    bestToken = quoteToken
                }
            }
        }
        
        let result = PathValidation(
            hasValidPaths: !tokenScores.isEmpty,
            tokenScores: tokenScores,
            recommendedToken: bestToken,
            diagnostics: "Evaluated \(tokenScores.count) tokens at depth \(depth)"
        )
        
        return result
    }
    
    public func recoverFromInvalidPath(
        partialKey: String,
        schemaKeys: [String],
        currentTokens: [Int32]
    ) -> RecoveryStrategy {
        // Delegate to IntelligentKeyRecovery
        return .abort(reason: "Recovery delegated to IntelligentKeyRecovery")
    }
    
    public func selectConstraintMode(
        context: GenerationContext
    ) -> ConstraintMode {
        // Delegate to AdaptiveConstraintEngine
        return .hard(allowedTokens: [])
    }
    
    // MARK: - Private Methods
    
    private func evaluateToken(
        tokenID: Int32,
        currentPath: TokenTrie.Path,
        tokenTrie: TokenTrie,
        depth: Int
    ) -> Float {
        // Base case: no look-ahead
        guard depth > 0 else {
            return 0.5 // Neutral score
        }
        
        // Simulate appending this token
        var testPath = currentPath
        let appendSuccess = testPath.append(tokenID, in: tokenTrie)
        
        guard appendSuccess else {
            return 0.0 // Invalid token
        }
        
        // Check if this leads to a terminal
        if testPath.isAtTerminal() {
            return 1.0 // Perfect score for reaching terminal
        }
        
        // Recursive look-ahead (simplified for performance)
        if depth > 1 {
            let futureTokens = tokenTrie.getAllowedTokens(for: testPath)
            if futureTokens.isEmpty {
                return 0.1 // Dead-end path
            }
            
            // Score based on branching factor
            let branchingScore = min(1.0, Float(futureTokens.count) / 10.0)
            
            // Check if any future path leads to terminal
            var hasTerminalPath = false
            for futureToken in futureTokens.prefix(5) { // Limit for performance
                var futurePath = testPath
                if futurePath.append(futureToken, in: tokenTrie) {
                    if futurePath.isAtTerminal() {
                        hasTerminalPath = true
                        break
                    }
                }
            }
            
            return hasTerminalPath ? 0.9 : branchingScore * 0.7
        }
        
        // Simple heuristic for depth 1
        let futureOptions = tokenTrie.getAllowedTokens(for: testPath).count
        return min(1.0, Float(futureOptions) / 5.0) * 0.6
    }
    
    // MARK: - Statistics
    
    public func getStatistics() -> PRECISEStatistics {
        return statistics
    }
    
    public func resetStatistics() {
        statistics = PRECISEStatistics()
    }
}