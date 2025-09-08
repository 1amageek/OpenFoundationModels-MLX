import Foundation

// PRECISE: Predictive Recovery Enhanced Constraint Intelligence for Schema Enforcement
// A modular system for improving JSON generation success rates through intelligent
// path validation, key recovery, and adaptive constraints.

/// Core protocol defining the PRECISE system capabilities
public protocol PRECISE: Sendable {
    /// Validates future paths from the current position to prevent dead-ends
    func validateFuturePaths(
        from path: TokenTrie.Path,
        tokenTrie: TokenTrie,
        depth: Int
    ) -> PathValidation
    
    /// Recovers from invalid paths by finding the best schema key match
    func recoverFromInvalidPath(
        partialKey: String,
        schemaKeys: [String],
        currentTokens: [Int32]
    ) -> RecoveryStrategy
    
    /// Selects the appropriate constraint mode based on generation context
    func selectConstraintMode(
        context: GenerationContext
    ) -> ConstraintMode
}

/// Result of path validation indicating viability and quality
public struct PathValidation: Sendable {
    /// Whether any valid paths exist
    public let hasValidPaths: Bool
    
    /// Scored token candidates (token ID -> quality score)
    public let tokenScores: [Int32: Float]
    
    /// Best next token based on future path analysis
    public let recommendedToken: Int32?
    
    /// Diagnostic information for debugging
    public let diagnostics: String?
    
    public init(
        hasValidPaths: Bool,
        tokenScores: [Int32: Float],
        recommendedToken: Int32? = nil,
        diagnostics: String? = nil
    ) {
        self.hasValidPaths = hasValidPaths
        self.tokenScores = tokenScores
        self.recommendedToken = recommendedToken
        self.diagnostics = diagnostics
    }
}

/// Strategy for recovering from invalid generation paths
public enum RecoveryStrategy: Sendable {
    /// Complete to the nearest valid schema key
    case completeToKey(target: String, completionTokens: [Int32])
    
    /// Close the current key and move to next structure
    case closeCurrentKey(tokens: [Int32])
    
    /// Insert a default value to maintain schema validity
    case insertDefault(value: String, tokens: [Int32])
    
    /// Skip to the next valid structure element
    case skipToNext(tokens: [Int32])
    
    /// No recovery possible, terminate generation
    case abort(reason: String)
}

/// Constraint application mode determining how strictly to enforce schema
public indirect enum ConstraintMode: Sendable {
    /// Hard constraints - only schema-valid tokens allowed
    case hard(allowedTokens: Set<Int32>)
    
    /// Soft constraints - prefer schema tokens but allow flexibility
    case soft(preferredTokens: Set<Int32>, bias: Float)
    
    /// Adaptive - dynamically adjust based on success rate
    case adaptive(baseMode: ConstraintMode, successRate: Float)
    
    /// No constraints - free generation
    case none
}

/// Context information for generation decisions
public struct GenerationContext: Sendable {
    /// Current JSON parsing state
    public let jsonState: JSONStateMachine
    
    /// Token generation history
    public let tokenHistory: [Int32]
    
    /// Whether this is for tool call generation
    public let isToolCall: Bool
    
    /// Required performance mode
    public let performanceMode: PerformanceMode
    
    /// Recent generation success rate
    public let recentSuccessRate: Float
    
    /// Current buffer size and memory pressure
    public let memoryPressure: Float
    
    public init(
        jsonState: JSONStateMachine,
        tokenHistory: [Int32],
        isToolCall: Bool,
        performanceMode: PerformanceMode,
        recentSuccessRate: Float = 0.95,
        memoryPressure: Float = 0.0
    ) {
        self.jsonState = jsonState
        self.tokenHistory = tokenHistory
        self.isToolCall = isToolCall
        self.performanceMode = performanceMode
        self.recentSuccessRate = recentSuccessRate
        self.memoryPressure = memoryPressure
    }
}

/// Performance mode balancing speed vs accuracy
public enum PerformanceMode: String, Sendable, CaseIterable {
    /// Fast mode - minimal validation (~1ms/token)
    case fast
    
    /// Balanced mode - moderate validation (~3ms/token)
    case balanced
    
    /// Accurate mode - full validation (~5-10ms/token)
    case accurate
    
    /// Auto mode - adapt based on context
    case auto
    
    /// Get the look-ahead depth for this mode
    public var lookAheadDepth: Int {
        switch self {
        case .fast: return 0
        case .balanced: return 1
        case .accurate: return 3
        case .auto: return 2
        }
    }
    
    /// Get the number of candidates to evaluate
    public var topKCandidates: Int {
        switch self {
        case .fast: return 1
        case .balanced: return 3
        case .accurate: return 10
        case .auto: return 5
        }
    }
    
    /// Whether to enable recovery strategies
    public var enableRecovery: Bool {
        switch self {
        case .fast: return false
        case .balanced: return true
        case .accurate: return true
        case .auto: return true
        }
    }
}

/// Statistics for monitoring PRECISE performance
public struct PRECISEStatistics: Sendable {
    public var totalValidations: Int = 0
    public var successfulValidations: Int = 0
    public var recoveryAttempts: Int = 0
    public var successfulRecoveries: Int = 0
    public var averageValidationTime: TimeInterval = 0
    
    public var successRate: Float {
        guard totalValidations > 0 else { return 0 }
        return Float(successfulValidations) / Float(totalValidations)
    }
    
    public var recoverySuccessRate: Float {
        guard recoveryAttempts > 0 else { return 0 }
        return Float(successfulRecoveries) / Float(recoveryAttempts)
    }
    
    public init() {}
    
    public mutating func recordValidation(success: Bool, time: TimeInterval) {
        totalValidations += 1
        if success { successfulValidations += 1 }
        
        // Update running average
        let alpha: Float = 0.1  // Exponential moving average factor
        averageValidationTime = (1 - Double(alpha)) * averageValidationTime + Double(alpha) * time
    }
    
    public mutating func recordRecovery(success: Bool) {
        recoveryAttempts += 1
        if success { successfulRecoveries += 1 }
    }
}