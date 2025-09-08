import Foundation

/// Selects appropriate constraint modes based on generation context.
/// Dynamically adjusts between hard and soft constraints for optimal results.
public final class AdaptiveConstraintEngine: PRECISE, @unchecked Sendable {
    private let tokenizer: MLXLLMTokenizer
    private var statistics = PRECISEStatistics()
    
    // Adaptive thresholds
    private let hardConstraintThreshold: Float = 0.95
    private let softConstraintThreshold: Float = 0.80
    private let adaptiveSwitchThreshold: Float = 0.90
    
    // Bias values for soft constraints
    private let defaultSoftBias: Float = 0.2
    private let adaptiveBiasRange: ClosedRange<Float> = 0.1...0.5
    
    public init(tokenizer: MLXLLMTokenizer) {
        self.tokenizer = tokenizer
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
        // Delegate to IntelligentKeyRecovery
        return .abort(reason: "Recovery delegated to IntelligentKeyRecovery")
    }
    
    public func selectConstraintMode(
        context: GenerationContext
    ) -> ConstraintMode {
        let startTime = Date()
        defer {
            let elapsed = Date().timeIntervalSince(startTime)
            statistics.recordValidation(success: true, time: elapsed)
        }
        
        // Determine base constraint mode based on context
        let baseMode = determineBaseMode(context: context)
        
        // Apply adaptive adjustments if needed
        if context.performanceMode == .auto {
            return applyAdaptiveAdjustments(
                baseMode: baseMode,
                context: context
            )
        }
        
        return baseMode
    }
    
    // MARK: - Private Methods
    
    private func determineBaseMode(context: GenerationContext) -> ConstraintMode {
        // Analyze JSON state to determine constraint needs
        let phase = context.jsonState.phase
        
        switch phase {
        case .inString(let strPhase):
            // Check if we're in a key
            if case .body(let kind, _) = strPhase, kind == .key {
                // Hard constraints for keys when success rate is high
                if context.recentSuccessRate >= hardConstraintThreshold {
                    // Get allowed tokens from context (would need to be passed)
                    return .hard(allowedTokens: [])
                } else if context.recentSuccessRate >= softConstraintThreshold {
                    // Soft constraints with moderate bias
                    return .soft(preferredTokens: [], bias: defaultSoftBias)
                } else {
                    // Adaptive mode for poor success rates
                    let adaptiveBias = calculateAdaptiveBias(
                        successRate: context.recentSuccessRate
                    )
                    return .adaptive(
                        baseMode: .soft(preferredTokens: [], bias: adaptiveBias),
                        successRate: context.recentSuccessRate
                    )
                }
            }
            
        case .inNumber, .inLiteral:
            // No constraints for values
            return .none
            
        case .inObject(let objPhase):
            // Check if we're expecting a key
            if objPhase == .expectKeyFirstQuote {
                // Moderate constraints when expecting keys
                return .soft(
                    preferredTokens: [],
                    bias: defaultSoftBias * 0.5
                )
            }
            
        default:
            break
        }
        
        // Default: no constraints
        return .none
    }
    
    private func applyAdaptiveAdjustments(
        baseMode: ConstraintMode,
        context: GenerationContext
    ) -> ConstraintMode {
        // Consider memory pressure
        if context.memoryPressure > 0.8 {
            // Reduce constraint complexity under memory pressure
            switch baseMode {
            case .hard(let tokens):
                // Switch to soft constraints to reduce memory usage
                return .soft(
                    preferredTokens: tokens,
                    bias: defaultSoftBias * 0.7
                )
            case .adaptive:
                // Simplify adaptive mode
                return .soft(
                    preferredTokens: [],
                    bias: defaultSoftBias
                )
            default:
                return baseMode
            }
        }
        
        // Consider performance requirements
        switch context.performanceMode {
        case .fast:
            // Minimize constraint overhead
            switch baseMode {
            case .hard, .adaptive:
                return .soft(preferredTokens: [], bias: 0.1)
            default:
                return baseMode
            }
            
        case .accurate:
            // Maximize constraint enforcement
            switch baseMode {
            case .soft(let tokens, _):
                return .hard(allowedTokens: tokens)
            case .none:
                return .soft(preferredTokens: [], bias: 0.05)
            default:
                return baseMode
            }
            
        default:
            break
        }
        
        // Consider tool call context
        if context.isToolCall {
            // Strengthen constraints for tool calls
            switch baseMode {
            case .soft(let tokens, let bias):
                return .soft(
                    preferredTokens: tokens,
                    bias: min(0.5, bias * 1.5)
                )
            case .none:
                return .soft(preferredTokens: [], bias: 0.15)
            default:
                return baseMode
            }
        }
        
        // Apply success rate adjustments
        if context.recentSuccessRate < adaptiveSwitchThreshold {
            let adjustment = calculateSuccessRateAdjustment(
                mode: baseMode,
                successRate: context.recentSuccessRate
            )
            return adjustment
        }
        
        return baseMode
    }
    
    private func calculateAdaptiveBias(successRate: Float) -> Float {
        // Linear interpolation based on success rate
        let normalized = max(0, min(1, successRate))
        let range = adaptiveBiasRange.upperBound - adaptiveBiasRange.lowerBound
        return adaptiveBiasRange.lowerBound + (1.0 - normalized) * range
    }
    
    private func calculateSuccessRateAdjustment(
        mode: ConstraintMode,
        successRate: Float
    ) -> ConstraintMode {
        // Adjust constraint strength based on success rate
        let strengthMultiplier = 1.0 + (1.0 - successRate) * 0.5
        
        switch mode {
        case .soft(let tokens, let bias):
            let adjustedBias = min(
                adaptiveBiasRange.upperBound,
                bias * Float(strengthMultiplier)
            )
            return .adaptive(
                baseMode: .soft(preferredTokens: tokens, bias: adjustedBias),
                successRate: successRate
            )
            
        case .none:
            // Add mild constraints if success rate is low
            if successRate < 0.7 {
                return .soft(
                    preferredTokens: [],
                    bias: defaultSoftBias * 0.3
                )
            }
            
        default:
            break
        }
        
        return mode
    }
    
    // MARK: - Configuration
    
    public struct Configuration {
        public var hardConstraintThreshold: Float = 0.95
        public var softConstraintThreshold: Float = 0.80
        public var adaptiveSwitchThreshold: Float = 0.90
        public var defaultSoftBias: Float = 0.2
        public var adaptiveBiasRange: ClosedRange<Float> = 0.1...0.5
        
        public init() {}
    }
    
    private var configuration = Configuration()
    
    public func configure(_ config: Configuration) {
        self.configuration = config
    }
    
    // MARK: - Statistics
    
    public func getStatistics() -> PRECISEStatistics {
        return statistics
    }
    
    public func resetStatistics() {
        statistics = PRECISEStatistics()
    }
}