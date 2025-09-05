import Foundation

enum RetryError: Error, Sendable { case exhausted }

enum RetryOrchestrator {
    static func run(
        maxTries: Int,
        attempt: @Sendable (_ tryIndex: Int) async throws -> ChatResponse
    ) async throws -> ChatResponse {
        var lastError: Error?
        for i in 0..<maxTries {
            do { return try await attempt(i) } catch { lastError = error }
        }
        throw lastError ?? RetryError.exhausted
    }
}

