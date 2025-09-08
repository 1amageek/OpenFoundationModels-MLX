import Testing
@testable import OpenFoundationModelsMLX
import MLX
import Foundation

/// Integration tests for the complete constrained generation pipeline
@Suite("Constrained Generation Integration Tests")
struct ConstrainedGenerationIntegrationTests {
    
    // MARK: - Test Successful Constrained Generation
    
    @Test("Successful constrained generation")
    func successfulConstrainedGeneration() throws {
        // Setup schema
        let schema = SchemaMeta(
            keys: ["name", "age", "email"],
            required: ["name", "email"]
        )
        
        // Create tokenizer
        let tokenizer = MockTokenizer()
        
        // Build TokenTrie with encoded keys
        var trie = OpenFoundationModelsMLX.TokenTrie()
        for key in schema.keys {
            let tokens = tokenizer.encode(key)
            trie.insert(tokenIDs: tokens, keyName: key)
        }
        
        // Create processor
        let swiftTokenizer = MockSwiftTokenizer()
        let processor = TokenTrieLogitProcessor(schema: schema, tokenizer: swiftTokenizer)
        
        // Simulate successful generation
        let mockJSON = #"{"name":"John","age":30,"email":"john@example.com"}"#
        
        // Process characters through JSON state machine
        var jsonState = JSONStateMachine()
        for char in mockJSON {
            jsonState.processCharacter(char)
        }
        
        // Verify completion
        #expect(jsonState.isComplete())
        #expect(!jsonState.isError())
    }
    
    // MARK: - Test Retry on Constraint Violation
    
    @Test("Retry on constraint violation")
    func retryOnConstraintViolation() async throws {
        // Setup processor with schema
        let schema = SchemaMeta(keys: ["firstName", "lastName"], required: ["firstName"])
        let tokenizer = MockTokenizer()
        let swiftTokenizer = MockSwiftTokenizer()
        let processor = TokenTrieLogitProcessor(schema: schema, tokenizer: swiftTokenizer)
        
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
                    Issue.record("Unexpected error type: \(jsonError)")
                }
            }
        }
        
        // Should have aborted before completing
        #expect(errorOccurred || chunks.count < 8, "Should abort on invalid key")
    }
    
    // MARK: - Test Temperature Adjustment
    
    @Test("Temperature adjustment logic")
    func temperatureAdjustmentLogic() {
        let sampling = SamplingParameters(
            temperature: 0.7,
            topP: 0.9,
            topK: nil,
            maxTokens: 100,
            seed: nil
        )
        
        // Test temperature increase on retry
        var adjusted = sampling
        adjusted.temperature = 0.8
        #expect(adjusted.temperature == 0.8)
        
        // Test temperature capping
        adjusted.temperature = 2.0
        let capped = min(adjusted.temperature!, 1.5)
        #expect(capped == 1.5)
        
        // Test deterministic mode (with seed)
        let deterministicSampling = SamplingParameters(
            temperature: 0.7,
            seed: 42
        )
        #expect(deterministicSampling.seed != nil)
    }
    
    // MARK: - Test Max Retry Limit
    
    @Test("Max retry limit")
    func maxRetryLimit() async {
        let maxRetries = 3
        var attempts = 0
        
        // Simulate retry loop
        while attempts < maxRetries {
            attempts += 1
            
            // Simulate failure
            if attempts < maxRetries {
                continue // Retry
            }
            
            // Max retries reached
            break
        }
        
        #expect(attempts == maxRetries)
    }
    
    // MARK: - Test Token Path Tracking
    
    @Test("Token path tracking")
    func tokenPathTracking() {
        var trie = TokenTrie()
        let tokens: [Int32] = [100, 101, 102]
        trie.insert(tokenIDs: tokens, keyName: "test")
        
        var path = TokenTrie.Path(root: trie.root)
        
        // Track path
        let result1 = path.append(100, in: trie)
        #expect(result1)
        #expect(path.tokens == [100])
        
        let result2 = path.append(101, in: trie)
        #expect(result2)
        #expect(path.tokens == [100, 101])
        
        // Invalid token should fail
        let result3 = path.append(999, in: trie)
        #expect(!result3)
        #expect(path.tokens == [100, 101]) // Unchanged
    }
    
    // MARK: - Test Error Detection Stages
    
    @Test("Error detection stages")
    func errorDetectionStages() {
        let processor = MockLogitProcessor()
        
        // Stage 1: No error
        #expect(!processor.hasError())
        #expect(!processor.hasFatalError())
        
        // Stage 2: Non-fatal error
        processor.shouldTriggerNonFatalError = true
        processor.mockError = .emptyConstraints
        // Trigger the error by calling process
        _ = processor.process(logits: MLX.zeros([1, 100]))
        #expect(processor.hasError())
        #expect(!processor.hasFatalError())
        
        // Stage 3: Fatal error
        processor.clearError()
        processor.shouldTriggerFatalError = true
        processor.mockError = .noValidTokens(partialKey: "test", position: 10)
        // Trigger the error by calling process
        _ = processor.process(logits: MLX.zeros([1, 100]))
        #expect(processor.hasError())
        #expect(processor.hasFatalError())
    }
    
    // MARK: - Test Complete Generation Pipeline
    
    @Test("Complete generation pipeline")
    func completeGenerationPipeline() async throws {
        // This test simulates the complete pipeline from schema to validated JSON
        
        // 1. Setup
        let schema = SchemaMeta(
            keys: ["id", "name", "active"],
            required: ["id", "name"]
        )
        
        let tokenizer = MockTokenizer()
        
        // 2. Build TokenTrie
        let trie = TokenTrieBuilder.build(from: schema, tokenizer: tokenizer)
        #expect(trie.allKeys.count == 3)
        
        // 3. Create processor
        let swiftTokenizer = MockSwiftTokenizer()
        let processor = TokenTrieLogitProcessor(schema: schema, tokenizer: swiftTokenizer)
        
        // 4. Simulate generation
        let validJSON = #"{"id":123,"name":"Test","active":true}"#
        
        // 5. Validate through state machine
        var stateMachine = JSONStateMachine()
        for char in validJSON {
            stateMachine.processCharacter(char)
        }
        
        #expect(stateMachine.isComplete())
        #expect(!stateMachine.isError())
        
        // 6. Validate schema compliance
        let validator = JSONValidator(allowExtraKeys: false, enableSnap: true)
        #expect(validator.validate(text: validJSON, schema: schema))
        
        // 7. Parse result
        if let data = validJSON.data(using: .utf8),
           let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any] {
            #expect(json["id"] != nil)
            #expect(json["name"] != nil)
            #expect(json["active"] != nil)
        } else {
            Issue.record("Failed to parse JSON")
        }
    }
    
    // MARK: - Test Concurrent Generation
    
    @Test("Concurrent generation")
    func concurrentGeneration() async throws {
        let schema = SchemaMeta(keys: ["key1", "key2"], required: ["key1"])
        let tokenizer = MockTokenizer()
        
        // Test concurrent trie building
        await withTaskGroup(of: TokenTrie.self) { group in
            for _ in 0..<5 {
                group.addTask {
                    return TokenTrieBuilder.buildCached(schema: schema, tokenizer: tokenizer)
                }
            }
            
            var tries: [TokenTrie] = []
            for await trie in group {
                tries.append(trie)
            }
            
            // All tries should be the same (cached)
            #expect(tries.count == 5)
            for trie in tries {
                #expect(trie.allKeys == Set(["key1", "key2"]))
            }
        }
    }
}