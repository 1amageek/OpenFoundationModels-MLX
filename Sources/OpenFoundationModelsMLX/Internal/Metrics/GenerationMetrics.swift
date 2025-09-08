import Foundation

/// Metrics collection for generation performance monitoring
public actor GenerationMetrics {
    public struct Metrics: Sendable {
        public var totalTokensGenerated: Int = 0
        public var totalGenerationTime: TimeInterval = 0
        public var totalRetries: Int = 0
        public var successfulGenerations: Int = 0
        public var failedGenerations: Int = 0
        public var cacheHits: Int = 0
        public var cacheMisses: Int = 0
        public var errorsByType: [String: Int] = [:]
        public var averageTokensPerGeneration: Double = 0
        public var peakMemoryUsage: UInt64 = 0
        
        public var averageTokensPerSecond: Double {
            guard totalGenerationTime > 0 else { return 0 }
            return Double(totalTokensGenerated) / totalGenerationTime
        }
        
        public var successRate: Double {
            let total = successfulGenerations + failedGenerations
            guard total > 0 else { return 0 }
            return Double(successfulGenerations) / Double(total)
        }
        
        public var cacheHitRate: Double {
            let total = cacheHits + cacheMisses
            guard total > 0 else { return 0 }
            return Double(cacheHits) / Double(total)
        }
    }
    
    private var metrics = Metrics()
    private let startTime = Date()
    
    public init() {}
    
    // MARK: - Recording Methods
    
    public func recordGeneration(tokens: Int, duration: TimeInterval, success: Bool) {
        metrics.totalTokensGenerated += tokens
        metrics.totalGenerationTime += duration
        
        if success {
            metrics.successfulGenerations += 1
        } else {
            metrics.failedGenerations += 1
        }
        
        // Update average
        let totalGenerations = metrics.successfulGenerations + metrics.failedGenerations
        if totalGenerations > 0 {
            metrics.averageTokensPerGeneration = Double(metrics.totalTokensGenerated) / Double(totalGenerations)
        }
    }
    
    public func recordRetry() {
        metrics.totalRetries += 1
    }
    
    public func recordCacheHit() {
        metrics.cacheHits += 1
    }
    
    public func recordCacheMiss() {
        metrics.cacheMisses += 1
    }
    
    public func recordError(_ error: Error) {
        let errorType = String(describing: type(of: error))
        metrics.errorsByType[errorType, default: 0] += 1
    }
    
    public func recordMemoryUsage(_ bytes: UInt64) {
        metrics.peakMemoryUsage = max(metrics.peakMemoryUsage, bytes)
    }
    
    // MARK: - Reporting
    
    public func getMetrics() -> Metrics {
        return metrics
    }
    
    public func reset() {
        metrics = Metrics()
    }
    
    public func generateReport() -> String {
        let uptime = Date().timeIntervalSince(startTime)
        let uptimeFormatted = formatDuration(uptime)
        
        return """
        === Generation Metrics Report ===
        Uptime: \(uptimeFormatted)
        
        Generation Stats:
        - Total Tokens: \(metrics.totalTokensGenerated)
        - Total Time: \(formatDuration(metrics.totalGenerationTime))
        - Average Speed: \(String(format: "%.1f", metrics.averageTokensPerSecond)) tokens/sec
        - Average per Generation: \(String(format: "%.1f", metrics.averageTokensPerGeneration)) tokens
        
        Success Rate:
        - Successful: \(metrics.successfulGenerations)
        - Failed: \(metrics.failedGenerations)
        - Success Rate: \(String(format: "%.1f%%", metrics.successRate * 100))
        - Total Retries: \(metrics.totalRetries)
        
        Cache Performance:
        - Cache Hits: \(metrics.cacheHits)
        - Cache Misses: \(metrics.cacheMisses)
        - Hit Rate: \(String(format: "%.1f%%", metrics.cacheHitRate * 100))
        
        Memory:
        - Peak Usage: \(formatBytes(metrics.peakMemoryUsage))
        
        Errors by Type:
        \(formatErrors())
        """
    }
    
    // MARK: - Private Helpers
    
    private func formatDuration(_ seconds: TimeInterval) -> String {
        let hours = Int(seconds) / 3600
        let minutes = (Int(seconds) % 3600) / 60
        let secs = Int(seconds) % 60
        
        if hours > 0 {
            return String(format: "%dh %dm %ds", hours, minutes, secs)
        } else if minutes > 0 {
            return String(format: "%dm %ds", minutes, secs)
        } else {
            return String(format: "%.1fs", seconds)
        }
    }
    
    private func formatBytes(_ bytes: UInt64) -> String {
        let units = ["B", "KB", "MB", "GB"]
        var size = Double(bytes)
        var unitIndex = 0
        
        while size >= 1024 && unitIndex < units.count - 1 {
            size /= 1024
            unitIndex += 1
        }
        
        return String(format: "%.1f %@", size, units[unitIndex])
    }
    
    private func formatErrors() -> String {
        guard !metrics.errorsByType.isEmpty else {
            return "  None"
        }
        
        return metrics.errorsByType
            .sorted { $0.value > $1.value }
            .map { "  - \($0.key): \($0.value)" }
            .joined(separator: "\n")
    }
}

// MARK: - Global Metrics Instance

/// Shared metrics collector for the entire framework
public let sharedMetrics = GenerationMetrics()