import Foundation

/// Protocol for processors that can track and report errors
protocol ErrorCheckable: Sendable {
    func hasError() -> Bool
    func hasFatalError() -> Bool
    func getLastError() -> JSONGenerationError?
    func clearError()
}

/// Extension to make TokenTrieLogitProcessor conform to ErrorCheckable
extension TokenTrieLogitProcessor: ErrorCheckable {
    // Already implements all required methods
}