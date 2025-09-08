import Foundation
@preconcurrency import MLX
import MLXLMCommon
import MLXLLM

/// LogitProcessor that integrates all PRECISE components for enhanced JSON generation.
/// Combines predictive validation, intelligent recovery, and adaptive constraints.
public final class PRECISELogitProcessor: LogitProcessor, @unchecked Sendable {
    private let tokenTrie: TokenTrie
    private let tokenizer: MLXLLMTokenizer
    private let configuration: PRECISEConfiguration
    
    // PRECISE components
    private let pathValidator: PredictivePathValidator
    private let keyRecovery: IntelligentKeyRecovery
    private let constraintEngine: AdaptiveConstraintEngine
    
    // State tracking
    private var jsonState: JSONStateMachine
    private var tokenPath: TokenTrie.Path
    private var generatedTokens: [Int32] = []
    private var recentSuccessRate: Float = 0.95
    private var consecutiveFailures: Int = 0
    
    // Performance monitoring
    private var statistics = PRECISEStatistics()
    private let startTime = Date()
    
    public init(
        schema: SchemaMeta,
        tokenizer: MLXLLMTokenizer,
        configuration: PRECISEConfiguration = .balanced
    ) {
        self.tokenizer = tokenizer
        self.configuration = configuration
        
        // Build TokenTrie
        self.tokenTrie = TokenTrieBuilder.buildCached(
            schema: schema,
            tokenizer: tokenizer
        )
        
        // Initialize PRECISE components
        self.pathValidator = PredictivePathValidator(tokenizer: tokenizer)
        self.keyRecovery = IntelligentKeyRecovery(tokenizer: tokenizer)
        self.constraintEngine = AdaptiveConstraintEngine(tokenizer: tokenizer)
        
        // Initialize state
        self.jsonState = JSONStateMachine()
        self.tokenPath = TokenTrie.Path(root: tokenTrie.root)
    }
    
    // MARK: - LogitProcessor Protocol
    
    public func prompt(_ prompt: MLXArray) {
        // Reset state for new generation
        jsonState.reset()
        tokenPath.reset(to: tokenTrie.root)
        generatedTokens.removeAll()
        consecutiveFailures = 0
    }
    
    public func process(logits: MLXArray) -> MLXArray {
        let processingStart = Date()
        
        // Build generation context
        let context = GenerationContext(
            jsonState: jsonState,
            tokenHistory: generatedTokens,
            isToolCall: true, // Assume tool call for now
            performanceMode: configuration.performanceMode,
            recentSuccessRate: recentSuccessRate,
            memoryPressure: estimateMemoryPressure()
        )
        
        // Select constraint mode using AdaptiveConstraintEngine
        let constraintMode = constraintEngine.selectConstraintMode(context: context)
        
        // Apply constraints based on mode
        let processedLogits: MLXArray
        
        switch constraintMode {
        case .hard(let allowedTokens):
            processedLogits = applyHardConstraints(
                logits: logits,
                allowedTokens: enhanceAllowedTokens(allowedTokens)
            )
            
        case .soft(let preferredTokens, let bias):
            processedLogits = applySoftConstraints(
                logits: logits,
                preferredTokens: enhanceAllowedTokens(preferredTokens),
                bias: bias
            )
            
        case .adaptive(let baseMode, _):
            // Apply base mode with adaptive adjustments
            processedLogits = processAdaptiveMode(
                logits: logits,
                baseMode: baseMode,
                context: context
            )
            
        case .none:
            processedLogits = logits
        }
        
        // Record processing time
        let processingTime = Date().timeIntervalSince(processingStart)
        statistics.recordValidation(success: true, time: processingTime)
        
        return processedLogits
    }
    
    public func didSample(token: MLXArray) {
        let tokenID = Int32(token.item(Int.self))
        generatedTokens.append(tokenID)
        
        // Decode token to update state
        let tokenText = tokenizer.decodeToken(tokenID)
        for char in tokenText {
            jsonState.processCharacter(char)
        }
        
        // Update token path if in key state
        if isInKeyState() {
            let success = tokenPath.append(tokenID, in: tokenTrie)
            
            if !success {
                consecutiveFailures += 1
                
                // Try recovery if enabled
                if configuration.enableRecovery && consecutiveFailures >= 2 {
                    attemptRecovery()
                }
            } else {
                consecutiveFailures = 0
                updateSuccessRate(success: true)
            }
        } else {
            // Reset path when not in key state
            if tokenPath.tokens.count > 0 {
                tokenPath.reset(to: tokenTrie.root)
                consecutiveFailures = 0
            }
        }
    }
    
    // MARK: - Private Methods
    
    private func enhanceAllowedTokens(_ baseTokens: Set<Int32>) -> Set<Int32> {
        guard isInKeyState() else {
            return baseTokens
        }
        
        // Use PredictivePathValidator to enhance token selection
        let validation = pathValidator.validateFuturePaths(
            from: tokenPath,
            tokenTrie: tokenTrie,
            depth: configuration.maxLookAheadDepth
        )
        
        if validation.hasValidPaths {
            // Filter tokens by score threshold
            let enhancedTokens = validation.tokenScores
                .filter { $0.value >= configuration.minPathScore }
                .map { $0.key }
            
            return baseTokens.union(Set(enhancedTokens))
        }
        
        return baseTokens
    }
    
    private func applyHardConstraints(
        logits: MLXArray,
        allowedTokens: Set<Int32>
    ) -> MLXArray {
        let vocabSize = logits.dim(logits.ndim - 1)
        
        // Always allow EOS token
        var allow = allowedTokens
        if let eos = tokenizer.eosTokenId() {
            allow.insert(eos)
        }
        
        // Create mask using MLX-compatible operations
        let allowedIndices = Array(allow.filter { $0 >= 0 && $0 < vocabSize })
        
        // Create mask array directly - MLX will handle GPU optimization
        let maskArray: MLXArray
        if !allowedIndices.isEmpty {
            // Create a boolean mask array
            var mask = [Float](repeating: 0, count: vocabSize)
            for idx in allowedIndices {
                mask[Int(idx)] = 1
            }
            maskArray = MLXArray(mask)
        } else {
            maskArray = MLX.zeros([vocabSize])
        }
        
        // Reshape for broadcasting
        var shape = Array(repeating: 1, count: logits.ndim)
        shape[logits.ndim - 1] = vocabSize
        let reshapedMask = maskArray.reshaped(shape)
        
        // Apply mask using where operation
        let negInf = MLX.full(logits.shape, values: -Float.infinity)
        return MLX.where(reshapedMask .> 0, logits, negInf)
    }
    
    private func applySoftConstraints(
        logits: MLXArray,
        preferredTokens: Set<Int32>,
        bias: Float
    ) -> MLXArray {
        guard !preferredTokens.isEmpty else {
            return logits
        }
        
        let vocabSize = logits.dim(logits.ndim - 1)
        
        // Create bias using MLX-compatible operations
        let preferredIndices = Array(preferredTokens.filter { $0 >= 0 && $0 < vocabSize })
        
        // Create bias array directly - MLX will handle GPU optimization
        let biasArray: MLXArray
        if !preferredIndices.isEmpty {
            var biasValues = [Float](repeating: 0, count: vocabSize)
            for idx in preferredIndices {
                biasValues[Int(idx)] = bias
            }
            biasArray = MLXArray(biasValues)
        } else {
            biasArray = MLX.zeros([vocabSize])
        }
        
        // Reshape for broadcasting
        var shape = Array(repeating: 1, count: logits.ndim)
        shape[logits.ndim - 1] = vocabSize
        let reshapedBias = biasArray.reshaped(shape)
        
        // Apply bias
        return logits + reshapedBias
    }
    
    private func processAdaptiveMode(
        logits: MLXArray,
        baseMode: ConstraintMode,
        context: GenerationContext
    ) -> MLXArray {
        // Validate future paths
        let validation = pathValidator.validateFuturePaths(
            from: tokenPath,
            tokenTrie: tokenTrie,
            depth: configuration.maxLookAheadDepth
        )
        
        // Apply base mode with adjustments based on validation
        if !validation.hasValidPaths {
            // No valid paths - try recovery or relax constraints
            return logits // Fallback to unconstrained
        }
        
        // Apply base mode
        switch baseMode {
        case .hard(let tokens):
            return applyHardConstraints(
                logits: logits,
                allowedTokens: tokens.union(Set(validation.tokenScores.keys))
            )
        case .soft(let tokens, let bias):
            // Adjust bias based on path scores
            let avgScore = validation.tokenScores.values.reduce(0, +) / Float(validation.tokenScores.count)
            let adjustedBias = bias * avgScore
            return applySoftConstraints(
                logits: logits,
                preferredTokens: tokens.union(Set(validation.tokenScores.keys)),
                bias: adjustedBias
            )
        default:
            return logits
        }
    }
    
    private func attemptRecovery() {
        // Get partial key from token history
        let partialKey = reconstructPartialKey()
        
        // Use IntelligentKeyRecovery to find recovery strategy
        let strategy = keyRecovery.recoverFromInvalidPath(
            partialKey: partialKey,
            schemaKeys: Array(tokenTrie.allKeys),
            currentTokens: generatedTokens
        )
        
        switch strategy {
        case .completeToKey(let target, _):
            Logger.debug("[PRECISE] Attempting recovery to key: \(target)")
            // Reset path for new key attempt
            tokenPath.reset(to: tokenTrie.root)
            consecutiveFailures = 0
            
        case .closeCurrentKey(_):
            Logger.debug("[PRECISE] Closing current key")
            tokenPath.reset(to: tokenTrie.root)
            consecutiveFailures = 0
            
        case .skipToNext(_):
            Logger.debug("[PRECISE] Skipping to next position")
            tokenPath.reset(to: tokenTrie.root)
            consecutiveFailures = 0
            
        case .insertDefault(let value, _):
            Logger.debug("[PRECISE] Inserting default value: \(value)")
            
        case .abort(let reason):
            Logger.debug("[PRECISE] Recovery failed: \(reason)")
            updateSuccessRate(success: false)
        }
        
        // Check if recovery was successful
        let success: Bool
        if case .abort = strategy {
            success = false
        } else {
            success = true
        }
        statistics.recordRecovery(success: success)
    }
    
    private func isInKeyState() -> Bool {
        if case .inString(let strPhase) = jsonState.phase,
           case .body(let kind, _) = strPhase,
           kind == .key {
            return true
        }
        return false
    }
    
    private func reconstructPartialKey() -> String {
        // Reconstruct key from token path
        return tokenizer.decode(tokenPath.tokens)
    }
    
    private func estimateMemoryPressure() -> Float {
        // Simple heuristic based on token count
        let pressure = Float(generatedTokens.count) / 1000.0
        return min(1.0, pressure)
    }
    
    private func updateSuccessRate(success: Bool) {
        // Exponential moving average
        let alpha: Float = 0.1
        recentSuccessRate = (1 - alpha) * recentSuccessRate + alpha * (success ? 1.0 : 0.0)
    }
    
    // MARK: - Statistics
    
    public func getStatistics() -> PRECISEStatistics {
        var combinedStats = statistics
        
        // Aggregate stats from components
        let validatorStats = pathValidator.getStatistics()
        let recoveryStats = keyRecovery.getStatistics()
        _ = constraintEngine.getStatistics()
        
        combinedStats.totalValidations += validatorStats.totalValidations
        combinedStats.successfulValidations += validatorStats.successfulValidations
        combinedStats.recoveryAttempts += recoveryStats.recoveryAttempts
        combinedStats.successfulRecoveries += recoveryStats.successfulRecoveries
        
        return combinedStats
    }
    
    public func debugInfo() -> String {
        let stats = getStatistics()
        return """
        PRECISE LogitProcessor Debug:
        - Performance Mode: \(configuration.performanceMode)
        - JSON State: \(jsonState.phase)
        - Token Path Length: \(tokenPath.tokens.count)
        - Generated Tokens: \(generatedTokens.count)
        - Success Rate: \(String(format: "%.1f%%", recentSuccessRate * 100))
        - Consecutive Failures: \(consecutiveFailures)
        - Validation Success: \(String(format: "%.1f%%", stats.successRate * 100))
        - Recovery Success: \(String(format: "%.1f%%", stats.recoverySuccessRate * 100))
        - Uptime: \(String(format: "%.2fs", Date().timeIntervalSince(startTime)))
        """
    }
}