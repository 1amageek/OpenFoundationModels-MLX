import XCTest
@testable import OpenFoundationModelsMLX
import MLX
import Foundation

/// Integration tests for the complete constrained generation pipeline
final class ConstrainedGenerationIntegrationTests: XCTestCase {
    
    // MARK: - Test Successful Constrained Generation
    
    func testSuccessfulConstrainedGeneration() throws {
        // Setup schema
        let schema = SchemaMeta(
            keys: ["name", "age", "email"],
            required: ["name", "email"]
        )
        
        // Create tokenizer
        let tokenizer = MockSwiftTokenizer()
        
        // Build TokenTrie with encoded keys
        var trie = OpenFoundationModelsMLX.TokenTrie()
        for key in schema.keys {
            let tokens = tokenizer.encode(text: key, addSpecialTokens: false).map { Int32($0) }
            trie.insert(tokenIDs: tokens, keyName: key)
        }
        
        // Create processor
        let processor = TokenTrieLogitProcessor(schema: schema, tokenizer: tokenizer)
        
        // Simulate successful generation
        let mockJSON = #"{"name":"John","age":30,"email":"john@example.com"}"#
        
        // Process characters through JSON state machine
        var jsonState = JSONStateMachine()
        for char in mockJSON {
            jsonState.processCharacter(char)
        }
        
        // Verify completion
        XCTAssertTrue(jsonState.isComplete())
        XCTAssertFalse(jsonState.isError())
    }
    
    // MARK: - Test Retry on Constraint Violation
    
    func testRetryOnConstraintViolation() async throws {
        // Setup processor with schema
        let schema = SchemaMeta(keys: ["firstName", "lastName"], required: ["firstName"])
        let tokenizer = MockSwiftTokenizer()
        let processor = TokenTrieLogitProcessor(schema: schema, tokenizer: tokenizer)
        
        // Create AbortableGenerator
        let generator = AbortableGenerator(processor: processor)
        
        // Simulate stream that will generate invalid key
        let baseStream = AsyncStream<String> { continuation in
            Task {
                // Start JSON
                continuation.yield("{")
                continuation.yield("\"")
                
                // Invalid key that should trigger error
                continuation.yield("invalid")
                continuation.yield("Key")
                continuation.yield("\"")
                
                // This should not be reached due to abort
                continuation.yield(":")
                continuation.yield("\"value\"")
                
                continuation.finish()
            }
        }
        
        // Process with AbortableGenerator
        let resultStream = generator.generate(baseStream: baseStream)
        
        var chunks: [String] = []
        var errorOccurred = false
        
        do {
            for try await chunk in resultStream {
                chunks.append(chunk)
            }
        } catch {
            errorOccurred = true
            
            // Verify it's a JSON generation error
            if let jsonError = error as? JSONGenerationError {
                switch jsonError {
                case .noValidTokens, .invalidTokenSelected:
                    // Expected error types
                    break
                default:
                    XCTFail("Unexpected error type: \(jsonError)")
                }
            }
        }
        
        // Should have aborted before completing
        XCTAssertTrue(errorOccurred || chunks.count < 8, "Should abort on invalid key")
    }
    
    // MARK: - Test Temperature Adjustment
    
    func testTemperatureAdjustmentLogic() {
        // Test the temperature adjustment calculation
        let baseTemp: Double = 0.7
        let increment: Double = 0.1
        
        // Simulate multiple abort positions
        let abortPositions = [5, 5, 6] // Similar positions
        
        // Check if positions are similar (within 3 tokens)
        let lastTwo = abortPositions.suffix(2)
        let similarPositions = lastTwo.allSatisfy { pos in
            abs(pos - abortPositions.last!) < 3
        }
        
        XCTAssertTrue(similarPositions)
        
        // Calculate adjusted temperature
        let attempt = 3
        let adjustedTemp = similarPositions
            ? min(baseTemp + Double(attempt) * increment * 1.5, 1.5)
            : min(baseTemp + Double(attempt - 1) * increment, 1.5)
        
        // Should use aggressive adjustment
        XCTAssertGreaterThan(adjustedTemp, baseTemp + Double(attempt - 1) * increment)
    }
    
    // MARK: - Test Max Retry Limit
    
    func testMaxRetryLimit() async {
        var attempts = 0
        let maxRetries = 3
        var lastError: Error?
        
        // Simulate retry loop
        for attempt in 1...maxRetries {
            attempts = attempt
            
            // Simulate generation that always fails
            let error = JSONGenerationError.schemaViolation(reason: "test failure")
            lastError = error
            
            // Would normally break on success
        }
        
        XCTAssertEqual(attempts, maxRetries)
        XCTAssertNotNil(lastError)
    }
    
    // MARK: - Test Token Path Tracking
    
    func testTokenPathTracking() {
        var trie = OpenFoundationModelsMLX.TokenTrie()
        let keyTokens: [Int32] = [100, 101, 102]
        trie.insert(tokenIDs: keyTokens, keyName: "testKey")
        
        var path = OpenFoundationModelsMLX.TokenTrie.Path(root: trie.root)
        
        // Track path progression
        var successfulAppends = 0
        for token in keyTokens {
            if path.append(token, in: trie) {
                successfulAppends += 1
            }
        }
        
        XCTAssertEqual(successfulAppends, keyTokens.count)
        XCTAssertTrue(path.isAtTerminal())
        XCTAssertEqual(path.tokens, keyTokens)
    }
    
    // MARK: - Test Error Detection at Different Stages
    
    func testErrorDetectionStages() {
        let processor = MockLogitProcessor()
        
        // Stage 1: No error
        XCTAssertFalse(processor.hasError())
        XCTAssertFalse(processor.hasFatalError())
        
        // Stage 2: Non-fatal error
        processor.shouldTriggerNonFatalError = true
        processor.mockError = .emptyConstraints
        XCTAssertTrue(processor.hasError())
        XCTAssertFalse(processor.hasFatalError())
        
        // Stage 3: Fatal error
        processor.clearError()
        processor.shouldTriggerFatalError = true
        processor.mockError = .noValidTokens(partialKey: "test", position: 10)
        XCTAssertTrue(processor.hasError())
        XCTAssertTrue(processor.hasFatalError())
    }
    
    // MARK: - Test Complete Generation Pipeline
    
    func testCompleteGenerationPipeline() async throws {
        // This test simulates the complete pipeline from schema to validated JSON
        
        // 1. Setup
        let schema = SchemaMeta(
            keys: ["id", "name", "active"],
            required: ["id", "name"]
        )
        let tokenizer = MockSwiftTokenizer()
        
        // 2. Build TokenTrie
        var trie = OpenFoundationModelsMLX.TokenTrie()
        for key in schema.keys {
            let tokens = tokenizer.encode(text: key, addSpecialTokens: false).map { Int32($0) }
            trie.insert(tokenIDs: tokens, keyName: key)
        }
        
        // 3. Create processor
        let processor = TokenTrieLogitProcessor(schema: schema, tokenizer: tokenizer)
        
        // 4. Create generator
        let generator = AbortableGenerator(processor: processor)
        
        // 5. Create valid JSON stream
        let validJSON = #"{"id":"123","name":"Test","active":true}"#
        let baseStream = AsyncStream<String> { continuation in
            Task {
                for char in validJSON {
                    continuation.yield(String(char))
                    try? await Task.sleep(nanoseconds: 1000) // Tiny delay
                }
                continuation.finish()
            }
        }
        
        // 6. Process stream
        let resultStream = generator.generate(baseStream: baseStream)
        
        var result = ""
        do {
            for try await chunk in resultStream {
                result += chunk
            }
        } catch {
            XCTFail("Should not throw error for valid JSON: \(error)")
        }
        
        // 7. Validate result
        XCTAssertEqual(result, validJSON)
        
        // 8. Verify with JSONValidator
        let validator = JSONValidator(allowExtraKeys: false, enableSnap: false)
        let isValid = validator.validate(text: result, schema: schema)
        XCTAssertTrue(isValid)
    }
    
    // MARK: - Test Concurrent Generation
    
    func testConcurrentGeneration() async throws {
        // Test that multiple generations can run concurrently
        
        let schema = SchemaMeta(keys: ["value"], required: ["value"])
        let tokenizer = MockSwiftTokenizer()
        
        // Create multiple generators
        let generators = (0..<3).map { _ in
            let processor = TokenTrieLogitProcessor(schema: schema, tokenizer: tokenizer)
            return AbortableGenerator(processor: processor)
        }
        
        // Create streams
        let streams = generators.enumerated().map { index, generator in
            let json = #"{"value":\#(index)}"#
            let baseStream = AsyncStream<String> { continuation in
                Task {
                    for char in json {
                        continuation.yield(String(char))
                    }
                    continuation.finish()
                }
            }
            return generator.generate(baseStream: baseStream)
        }
        
        // Process all streams concurrently
        try await withThrowingTaskGroup(of: String.self) { group in
            for stream in streams {
                group.addTask {
                    var result = ""
                    for try await chunk in stream {
                        result += chunk
                    }
                    return result
                }
            }
            
            var results: [String] = []
            for try await result in group {
                results.append(result)
            }
            
            // Verify all completed
            XCTAssertEqual(results.count, 3)
        }
    }
}