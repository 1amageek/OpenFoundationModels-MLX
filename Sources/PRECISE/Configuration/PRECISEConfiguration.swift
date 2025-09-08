import Foundation

/// Centralized configuration for the PRECISE system.
/// Manages settings for all PRECISE components.
public struct PRECISEConfiguration: Sendable {
    // MARK: - Performance Settings
    
    /// Performance mode for the PRECISE system
    public var performanceMode: PerformanceMode = .balanced
    
    /// Maximum depth for path validation look-ahead
    public var maxLookAheadDepth: Int = 3
    
    /// Number of top candidates to evaluate
    public var topKCandidates: Int = 5
    
    /// Enable or disable recovery strategies
    public var enableRecovery: Bool = true
    
    // MARK: - Validation Settings
    
    /// Cache size for path validation results
    public var validationCacheSize: Int = 100
    
    /// Timeout for validation operations (seconds)
    public var validationTimeout: TimeInterval = 0.01
    
    /// Minimum score threshold for valid paths
    public var minPathScore: Float = 0.3
    
    // MARK: - Recovery Settings
    
    /// Maximum edit distance for key recovery
    public var maxEditDistance: Int = 2
    
    /// Enable partial key matching
    public var enablePartialMatching: Bool = true
    
    /// Minimum key length for substring matching
    public var minSubstringLength: Int = 3
    
    // MARK: - Constraint Settings
    
    /// Threshold for switching to hard constraints
    public var hardConstraintThreshold: Float = 0.95
    
    /// Threshold for enabling soft constraints
    public var softConstraintThreshold: Float = 0.80
    
    /// Threshold for adaptive mode switching
    public var adaptiveSwitchThreshold: Float = 0.90
    
    /// Default bias for soft constraints
    public var defaultSoftBias: Float = 0.2
    
    /// Range of bias values for adaptive adjustment
    public var adaptiveBiasRange: ClosedRange<Float> = 0.1...0.5
    
    // MARK: - Memory Management
    
    /// Maximum memory usage for caches (MB)
    public var maxCacheMemoryMB: Int = 10
    
    /// Memory pressure threshold for constraint relaxation
    public var memoryPressureThreshold: Float = 0.8
    
    // MARK: - Initialization
    
    public init() {}
    
    /// Create configuration optimized for speed
    public static var fast: PRECISEConfiguration {
        var config = PRECISEConfiguration()
        config.performanceMode = .fast
        config.maxLookAheadDepth = 0
        config.topKCandidates = 1
        config.enableRecovery = false
        config.validationCacheSize = 50
        config.validationTimeout = 0.001
        return config
    }
    
    /// Create configuration optimized for accuracy
    public static var accurate: PRECISEConfiguration {
        var config = PRECISEConfiguration()
        config.performanceMode = .accurate
        config.maxLookAheadDepth = 5
        config.topKCandidates = 10
        config.enableRecovery = true
        config.validationCacheSize = 200
        config.validationTimeout = 0.05
        config.minPathScore = 0.5
        config.hardConstraintThreshold = 0.90
        return config
    }
    
    /// Create configuration with balanced trade-offs
    public static var balanced: PRECISEConfiguration {
        return PRECISEConfiguration()
    }
    
    /// Create configuration optimized for tool calls
    public static var toolCall: PRECISEConfiguration {
        var config = PRECISEConfiguration()
        config.performanceMode = .accurate
        config.maxLookAheadDepth = 4
        config.enableRecovery = true
        config.hardConstraintThreshold = 0.92
        config.softConstraintThreshold = 0.75
        config.defaultSoftBias = 0.3
        return config
    }
    
    // MARK: - Environment Variables
    
    /// Load configuration from environment variables
    public static func fromEnvironment() -> PRECISEConfiguration {
        var config = PRECISEConfiguration()
        
        // Performance mode
        if let modeStr = ProcessInfo.processInfo.environment["OFM_PRECISE_MODE"] {
            config.performanceMode = PerformanceMode(rawValue: modeStr) ?? .balanced
        }
        
        // Look-ahead depth
        if let depthStr = ProcessInfo.processInfo.environment["OFM_PRECISE_LOOKAHEAD"],
           let depth = Int(depthStr) {
            config.maxLookAheadDepth = max(0, min(10, depth))
        }
        
        // Recovery
        if let recoveryStr = ProcessInfo.processInfo.environment["OFM_PRECISE_RECOVERY"] {
            config.enableRecovery = recoveryStr.lowercased() != "false"
        }
        
        // Soft bias
        if let biasStr = ProcessInfo.processInfo.environment["OFM_PRECISE_BIAS"],
           let bias = Float(biasStr) {
            config.defaultSoftBias = max(0.0, min(1.0, bias))
        }
        
        // Cache size
        if let cacheStr = ProcessInfo.processInfo.environment["OFM_PRECISE_CACHE_MB"],
           let cacheMB = Int(cacheStr) {
            config.maxCacheMemoryMB = max(1, min(100, cacheMB))
        }
        
        return config
    }
    
    // MARK: - Validation
    
    /// Validate configuration values
    public mutating func validate() {
        // Ensure values are within reasonable ranges
        maxLookAheadDepth = max(0, min(10, maxLookAheadDepth))
        topKCandidates = max(1, min(20, topKCandidates))
        validationCacheSize = max(10, min(1000, validationCacheSize))
        maxEditDistance = max(1, min(5, maxEditDistance))
        minSubstringLength = max(2, min(10, minSubstringLength))
        
        // Ensure thresholds are in [0, 1]
        hardConstraintThreshold = max(0, min(1, hardConstraintThreshold))
        softConstraintThreshold = max(0, min(1, softConstraintThreshold))
        adaptiveSwitchThreshold = max(0, min(1, adaptiveSwitchThreshold))
        defaultSoftBias = max(0, min(1, defaultSoftBias))
        memoryPressureThreshold = max(0, min(1, memoryPressureThreshold))
        
        // Ensure threshold ordering
        if softConstraintThreshold > hardConstraintThreshold {
            softConstraintThreshold = hardConstraintThreshold * 0.85
        }
        
        // Validate adaptive bias range
        let lower = max(0.0, min(1.0, adaptiveBiasRange.lowerBound))
        let upper = max(lower, min(1.0, adaptiveBiasRange.upperBound))
        adaptiveBiasRange = lower...upper
    }
}

// MARK: - Configuration Builder

/// Builder pattern for creating PRECISE configurations
public class PRECISEConfigurationBuilder {
    private var config = PRECISEConfiguration()
    
    public init() {}
    
    @discardableResult
    public func performanceMode(_ mode: PerformanceMode) -> Self {
        config.performanceMode = mode
        return self
    }
    
    @discardableResult
    public func lookAheadDepth(_ depth: Int) -> Self {
        config.maxLookAheadDepth = depth
        return self
    }
    
    @discardableResult
    public func enableRecovery(_ enable: Bool) -> Self {
        config.enableRecovery = enable
        return self
    }
    
    @discardableResult
    public func softBias(_ bias: Float) -> Self {
        config.defaultSoftBias = bias
        return self
    }
    
    @discardableResult
    public func cacheMemory(_ mb: Int) -> Self {
        config.maxCacheMemoryMB = mb
        return self
    }
    
    public func build() -> PRECISEConfiguration {
        config.validate()
        return config
    }
}