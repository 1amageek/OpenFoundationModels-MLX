import Foundation

/// JSON generation errors
public enum JSONGenerationError: Error, LocalizedError, Sendable {
    case invalidTokenSelected(token: Int32, partialKey: String, expectedTokens: Set<Int32>)
    case emptyConstraints
    case schemaViolation(reason: String)
    case unexpectedJSONStructure(phase: String)
    case abortedDueToError(position: Int)
    
    public var errorDescription: String? {
        switch self {
        case .invalidTokenSelected(let token, let partial, let expected):
            return "Invalid token \(token) selected. Partial key: '\(partial)'. Expected one of: \(expected)"
        case .emptyConstraints:
            return "No valid tokens available for current JSON phase"
        case .schemaViolation(let reason):
            return "Schema violation: \(reason)"
        case .unexpectedJSONStructure(let phase):
            return "Unexpected JSON structure at phase: \(phase)"
        case .abortedDueToError(let position):
            return "Generation aborted at position \(position) due to constraint violation"
        }
    }
}

/// Protocol for processors that can track and report errors
protocol ErrorCheckable: Sendable {
    func hasError() -> Bool
    func hasFatalError() -> Bool
    func getLastError() -> JSONGenerationError?
    func clearError()
}

// DPDAKeyTrieLogitProcessor already conforms to ErrorCheckable in its main file