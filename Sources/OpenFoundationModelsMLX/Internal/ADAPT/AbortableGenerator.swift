import Foundation
import MLXLMCommon
import MLXLLM
import Synchronization

/// Error-aware generation wrapper that can abort immediately on constraint violations
final class AbortableGenerator: Sendable {
    private let processor: ErrorCheckable
    private let abortPosition = Mutex<Int?>(nil)
    
    init(processor: ErrorCheckable) {
        self.processor = processor
    }
    
    /// Get the token position where generation was aborted (if any) - thread-safe
    func getAbortPosition() -> Int? {
        abortPosition.withLock { $0 }
    }
    
    /// Set abort position - thread-safe
    private func setAbortPosition(_ position: Int?) {
        abortPosition.withLock { $0 = position }
    }
    
    /// Generate with error polling - aborts immediately on fatal errors
    /// Handles AsyncStream (non-throwing)
    func generate<T: Sendable>(
        baseStream: AsyncStream<T>
    ) -> AsyncThrowingStream<T, Error> {
        AsyncThrowingStream { continuation in
            Task { @Sendable in
                var tokenCount = 0
                var localAbortedAt: Int? = nil
                
                do {
                    for await output in baseStream {
                        // Check for task cancellation
                        try Task.checkCancellation()
                        
                        // Track token count BEFORE checking for errors (so we record the correct position)
                        tokenCount += 1
                        
                        // Check for fatal errors after each chunk
                        if self.processor.hasFatalError() {
                            let error = self.processor.getLastError()!
                            localAbortedAt = tokenCount
                            Logger.warning("[AbortableGenerator] Fatal error detected at token \(tokenCount), aborting: \(error)")
                            throw error
                        }
                        
                        continuation.yield(output)
                    }
                    // Only set abort position if we actually aborted
                    if let abortPos = localAbortedAt {
                        self.setAbortPosition(abortPos)
                    }
                    continuation.finish()
                } catch {
                    // Set abort position if we tracked it
                    if let abortPos = localAbortedAt {
                        self.setAbortPosition(abortPos)
                    }
                    continuation.finish(throwing: error)
                }
            }
        }
    }
    
    /// Generate with error polling for throwing streams
    func generateThrowing<T: Sendable>(
        baseStream: AsyncThrowingStream<T, Error>
    ) -> AsyncThrowingStream<T, Error> {
        AsyncThrowingStream { continuation in
            Task { @Sendable in
                var tokenCount = 0
                var localAbortedAt: Int? = nil
                
                do {
                    for try await output in baseStream {
                        // Check for task cancellation
                        try Task.checkCancellation()
                        
                        // Track token count BEFORE checking for errors (so we record the correct position)
                        tokenCount += 1
                        
                        // Check for fatal errors after each chunk
                        if self.processor.hasFatalError() {
                            let error = self.processor.getLastError()!
                            localAbortedAt = tokenCount
                            Logger.warning("[AbortableGenerator] Fatal error detected at token \(tokenCount), aborting: \(error)")
                            throw error
                        }
                        
                        continuation.yield(output)
                    }
                    // Only set abort position if we actually aborted
                    if let abortPos = localAbortedAt {
                        self.setAbortPosition(abortPos)
                    }
                    continuation.finish()
                } catch {
                    // Set abort position if we tracked it
                    if let abortPos = localAbortedAt {
                        self.setAbortPosition(abortPos)
                    }
                    continuation.finish(throwing: error)
                }
            }
        }
    }
    
    /// Generate with detailed tracking for debugging
    /// Uses generic type T to work with any MLXLMCommon output type
    func generateWithTracking<T: Sendable>(
        baseStream: AsyncThrowingStream<T, Error>
    ) -> AsyncThrowingStream<(output: T, tokenCount: Int), Error> where T: Sendable {
        AsyncThrowingStream { continuation in
            Task { @Sendable in
                var tokenCount = 0
                var localAbortedAt: Int? = nil
                
                do {
                    for try await output in baseStream {
                        // Check for any errors (not just fatal)
                        if self.processor.hasError() {
                            let error = self.processor.getLastError()!
                            let isFatal = self.processor.hasFatalError()
                            
                            if isFatal {
                                localAbortedAt = tokenCount
                                Logger.warning("[AbortableGenerator] Fatal error at token \(tokenCount): \(error)")
                                throw error
                            } else {
                                Logger.debug("[AbortableGenerator] Non-fatal error at token \(tokenCount): \(error)")
                            }
                        }
                        
                        // Track token count
                        tokenCount += 1
                        
                        continuation.yield((output, tokenCount))
                    }
                    self.setAbortPosition(localAbortedAt)
                    continuation.finish()
                } catch {
                    // Set abort position if we tracked it
                    if let abortPos = localAbortedAt {
                        self.setAbortPosition(abortPos)
                    }
                    continuation.finish(throwing: error)
                }
            }
        }
    }
}