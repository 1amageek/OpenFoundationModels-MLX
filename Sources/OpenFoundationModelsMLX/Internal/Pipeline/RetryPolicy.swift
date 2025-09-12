import Foundation

public struct RetryPolicy: Sendable {
    public let maxAttempts: Int
    public let jitter: Bool
    public let backoffMultiplier: Double
    public let maxBackoffSeconds: TimeInterval
    
    public init(
        maxAttempts: Int = 3,
        jitter: Bool = false,
        backoffMultiplier: Double = 2.0,
        maxBackoffSeconds: TimeInterval = 30.0
    ) {
        self.maxAttempts = max(1, maxAttempts)
        self.jitter = jitter
        self.backoffMultiplier = max(1.0, backoffMultiplier)
        self.maxBackoffSeconds = max(0, maxBackoffSeconds)
    }
    
    public func shouldRetry(attempt: Int) -> Bool {
        return attempt < maxAttempts
    }
    
    public func backoffInterval(for attempt: Int) -> TimeInterval {
        guard attempt > 0 else { return 0 }
        
        let baseDelay = pow(backoffMultiplier, Double(attempt - 1))
        var delay = min(baseDelay, maxBackoffSeconds)
        
        if jitter {
            let jitterAmount = Double.random(in: 0...0.3) * delay
            delay += jitterAmount
        }
        
        return delay
    }
    
    public static let noRetry = RetryPolicy(maxAttempts: 1)
    
    public static let standard = RetryPolicy(maxAttempts: 3, jitter: true)
    
    public static let aggressive = RetryPolicy(
        maxAttempts: 5,
        jitter: true,
        backoffMultiplier: 1.5,
        maxBackoffSeconds: 10.0
    )
}