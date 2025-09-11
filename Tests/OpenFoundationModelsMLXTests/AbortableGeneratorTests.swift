import Testing
@testable import OpenFoundationModelsMLX
import MLX

struct AbortableGeneratorTests {
    
    // MARK: - Test Normal Stream Processing
    
    @Test("Normal stream processing without errors")
    func normalStreamProcessing() async throws {
        // Setup
        let processor = MockLogitProcessor()
        processor.shouldTriggerFatalError = false
        
        let generator = AbortableGenerator(processor: processor)
        
        // Create a mock stream with test data
        let baseStream = AsyncStream<String> { continuation in
            Task {
                for i in 1...5 {
                    // Simulate processing (no error expected)
                    _ = processor.process(logits: MLX.zeros([1, 100]))
                    continuation.yield("chunk\(i)")
                    try? await Task.sleep(nanoseconds: 10_000) // Small delay
                }
                continuation.finish()
            }
        }
        
        // Process stream
        let resultStream = generator.generate(baseStream: baseStream)
        
        // Collect results
        var results: [String] = []
        do {
            for try await chunk in resultStream {
                results.append(chunk)
            }
        } catch {
            Issue.record("Should not throw error: \(error)")
        }
        
        // Verify
        #expect(results.count == 5)
        #expect(results == ["chunk1", "chunk2", "chunk3", "chunk4", "chunk5"])
        #expect(generator.getAbortPosition() == nil)
    }
    
    // MARK: - Test Immediate Abort on Fatal Error
    
    @Test("Immediate abort on fatal error")
    func immediateAbortOnFatalError() async throws {
        // Setup
        let processor = MockLogitProcessor()
        processor.fatalErrorAtToken = 3 // Trigger error at token 3
        processor.mockError = .abortedDueToError(position: 3)
        
        let generator = AbortableGenerator(processor: processor)
        
        // Create a mock stream that simulates processing
        let baseStream = AsyncStream<String> { continuation in
            Task {
                for i in 1...10 {
                    // Simulate processing to trigger error state
                    _ = processor.process(logits: MLX.zeros([1, 100]))
                    continuation.yield("chunk\(i)")
                    try? await Task.sleep(nanoseconds: 10_000)
                }
                continuation.finish()
            }
        }
        
        // Process stream
        let resultStream = generator.generate(baseStream: baseStream)
        
        // Collect results and expect error
        var results: [String] = []
        var caughtError: Error?
        
        do {
            for try await chunk in resultStream {
                results.append(chunk)
            }
        } catch {
            caughtError = error
        }
        
        // Verify
        #expect(caughtError != nil)
        #expect(results.count < 10, "Should abort before processing all chunks")
        
        if let jsonError = caughtError as? JSONGenerationError {
            switch jsonError {
            case .abortedDueToError(let position):
                #expect(position == 3)
            default:
                Issue.record("Wrong error type")
            }
        } else {
            Issue.record("Expected JSONGenerationError")
        }
    }
    
    // MARK: - Test Token Count Tracking
    
    @Test("Token count tracking")
    func tokenCountTracking() async throws {
        // Setup
        let processor = MockLogitProcessor()
        processor.shouldTriggerFatalError = false
        
        let generator = AbortableGenerator(processor: processor)
        
        // Create a mock throwing stream
        let baseStream = AsyncThrowingStream<String, Error> { continuation in
            Task {
                for i in 1...3 {
                    continuation.yield("token\(i)")
                }
                continuation.finish()
            }
        }
        
        // Use generateWithTracking to get token counts
        let trackingStream = generator.generateWithTracking(baseStream: baseStream)
        
        // Collect results with token counts
        var results: [(String, Int)] = []
        do {
            for try await (output, tokenCount) in trackingStream {
                results.append((output, tokenCount))
            }
        } catch {
            Issue.record("Should not throw error: \(error)")
        }
        
        // Verify token counts
        #expect(results.count == 3)
        #expect(results[0].1 == 1) // First token count
        #expect(results[1].1 == 2) // Second token count
        #expect(results[2].1 == 3) // Third token count
    }
    
    // MARK: - Test Abort Position Recording
    
    @Test("Abort position recording")
    func abortPositionRecording() async throws {
        // Setup
        let processor = MockLogitProcessor()
        let generator = AbortableGenerator(processor: processor)
        
        // Create a stream that will trigger error at specific position
        let baseStream = AsyncStream<String> { continuation in
            Task { @Sendable in
                for i in 1...10 {
                    // Trigger error after 4th chunk
                    if i == 4 {
                        processor.shouldTriggerFatalError = true
                        processor.mockError = .invalidTokenSelected(
                            token: 123,
                            partialKey: "test",
                            expectedTokens: Set([456, 789])
                        )
                    }
                    // Simulate processing to trigger error state
                    _ = processor.process(logits: MLX.zeros([1, 100]))
                    continuation.yield("chunk\(i)")
                    try? await Task.sleep(nanoseconds: 10_000)
                }
                continuation.finish()
            }
        }
        
        // Process stream
        let resultStream = generator.generate(baseStream: baseStream)
        
        // Consume stream and expect error
        do {
            for try await _ in resultStream {
                // Process chunks
            }
        } catch {
            // Expected error
        }
        
        // Verify abort position was recorded
        let abortPosition = generator.getAbortPosition()
        #expect(abortPosition != nil, "Abort position should be recorded when fatal error occurs")
        if let pos = abortPosition {
            // Note: Position might be 4 or 5 depending on timing
            #expect(pos >= 4 && pos <= 5, "Abort position \(pos) should be between 4 and 5")
        }
    }
    
    // MARK: - Test Non-Fatal Error Continuation
    
    @Test("Non-fatal error continuation")
    func nonFatalErrorContinuation() async throws {
        // Setup
        let processor = MockLogitProcessor()
        processor.shouldTriggerNonFatalError = true // Non-fatal error
        processor.mockError = .emptyConstraints
        
        let generator = AbortableGenerator(processor: processor)
        
        // Create a mock stream
        let baseStream = AsyncStream<String> { continuation in
            Task {
                for i in 1...5 {
                    continuation.yield("chunk\(i)")
                }
                continuation.finish()
            }
        }
        
        // Process stream - should continue despite non-fatal error
        let resultStream = generator.generate(baseStream: baseStream)
        
        // Collect results
        var results: [String] = []
        var errorThrown = false
        
        do {
            for try await chunk in resultStream {
                results.append(chunk)
            }
        } catch {
            errorThrown = true
        }
        
        // Verify - should process all chunks despite non-fatal error
        #expect(!errorThrown, "Non-fatal error should not abort stream")
        #expect(results.count == 5)
    }
    
    // MARK: - Test Multiple Errors
   
}
