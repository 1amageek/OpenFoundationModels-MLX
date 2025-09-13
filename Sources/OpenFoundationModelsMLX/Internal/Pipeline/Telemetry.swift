import Foundation

public enum TelemetryEvent: String, CaseIterable, Sendable {
    case pipelineStarted = "pipeline.started"
    case pipelineCompleted = "pipeline.completed"
    case pipelineFailed = "pipeline.failed"
    
    case attemptStarted = "attempt.started"
    case attemptCompleted = "attempt.completed"
    case attemptFailed = "attempt.failed"
    
    case promptBuilt = "prompt.built"
    case constraintsPrepared = "constraints.prepared"
    case generationStarted = "generation.started"
    case generationCompleted = "generation.completed"
    
    case validationStarted = "validation.started"
    case validationPassed = "validation.passed"
    case validationFailed = "validation.failed"
    
    case repairAttempted = "repair.attempted"
    case repairSucceeded = "repair.succeeded"
    case repairFailed = "repair.failed"
    
    case retryScheduled = "retry.scheduled"
}

public protocol Telemetry: Sendable {
    func event(_ name: TelemetryEvent, metadata: [String: Any]) async
    
    func measure<T>(_ name: String, block: () async throws -> T) async rethrows -> T
}

public extension Telemetry {
    func measure<T>(_ name: String, block: () async throws -> T) async rethrows -> T {
        let start = Date()
        defer {
            let duration = Date().timeIntervalSince(start)
            Task {
                await event(.pipelineCompleted, metadata: [
                    "metric": name,
                    "duration_ms": Int(duration * 1000)
                ])
            }
        }
        return try await block()
    }
}

public final class NoOpTelemetry: Telemetry, Sendable {
    public init() {}
    
    public func event(_ name: TelemetryEvent, metadata: [String: Any]) async {
    }
}