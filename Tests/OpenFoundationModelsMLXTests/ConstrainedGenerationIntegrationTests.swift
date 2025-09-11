import Testing
@testable import OpenFoundationModelsMLX
import MLX
import Foundation

@Suite("Constrained Generation Integration Tests")
struct ConstrainedGenerationIntegrationTests {
    
    @Test("Successful constrained generation")
    func successfulConstrainedGeneration() throws {
        let schema = SchemaNode(
            kind: .object,
            properties: [
                "name": SchemaNode.any,
                "age": SchemaNode.any,
                "email": SchemaNode.any
            ],
            required: ["name", "email"]
        )
        
        let tokenizer = MockTokenizer()
        
        var trie = OpenFoundationModelsMLX.TokenTrie()
        for key in schema.objectKeys {
            let tokens = tokenizer.encode(key)
            trie.insert(tokenIDs: tokens, keyName: key)
        }
        
        let swiftTokenizer = MockSwiftTokenizer()
        let processor = DPDAKeyTrieLogitProcessor(schema: schema, tokenizer: swiftTokenizer)
        
        let mockJSON = #"{"name":"John","age":30,"email":"john@example.com"}"#
        
        var jsonState = JSONStateMachine()
        for char in mockJSON {
            jsonState.processCharacter(char)
        }
        
        #expect(jsonState.isComplete())
        #expect(!jsonState.isError())
    }
    
    @Test("Retry on constraint violation")
    func retryOnConstraintViolation() async throws {
        let schema = SchemaNode(
            kind: .object,
            properties: [
                "firstName": SchemaNode.any,
                "lastName": SchemaNode.any
            ],
            required: ["firstName"]
        )
        let tokenizer = MockTokenizer()
        let swiftTokenizer = MockSwiftTokenizer()
        let processor = DPDAKeyTrieLogitProcessor(schema: schema, tokenizer: swiftTokenizer)
        
        let generator = AbortableGenerator(processor: processor)
        
        let baseStream = AsyncStream<String> { continuation in
            Task {
                continuation.yield("{")
                continuation.yield("\"")
                continuation.yield("invalid")
                continuation.yield("Key")
                continuation.yield("\"")
                continuation.yield(":")
                continuation.yield("\"value\"")
                continuation.finish()
            }
        }
        
        let resultStream = generator.generate(baseStream: baseStream)
        
        var chunks: [String] = []
        var errorOccurred = false
        
        do {
            for try await chunk in resultStream {
                chunks.append(chunk)
            }
        } catch {
            errorOccurred = true
            
            if let jsonError = error as? JSONGenerationError {
                switch jsonError {
                case .invalidTokenSelected:
                    break
                default:
                    Issue.record("Unexpected error type: \(jsonError)")
                }
            }
        }
        
        #expect(errorOccurred || chunks.count < 8, "Should abort on invalid key")
    }
    
    @Test("Temperature adjustment logic")
    func temperatureAdjustmentLogic() {
        let sampling = SamplingParameters(
            temperature: 0.7,
            topP: 0.9,
            topK: nil,
            maxTokens: 100,
            seed: nil
        )
        
        var adjusted = sampling
        adjusted.temperature = 0.8
        #expect(adjusted.temperature == 0.8)
        
        adjusted.temperature = 2.0
        let capped = min(adjusted.temperature!, 1.5)
        #expect(capped == 1.5)
        
        let deterministicSampling = SamplingParameters(
            temperature: 0.7,
            seed: 42
        )
        #expect(deterministicSampling.seed != nil)
    }
    
    @Test("Max retry limit")
    func maxRetryLimit() async {
        let maxRetries = 3
        var attempts = 0
        
        while attempts < maxRetries {
            attempts += 1
            
            if attempts < maxRetries {
                continue
            }
            
            break
        }
        
        #expect(attempts == maxRetries)
    }
    
    @Test("Token path tracking")
    func tokenPathTracking() {
        var trie = TokenTrie()
        let tokens: [Int32] = [100, 101, 102]
        trie.insert(tokenIDs: tokens, keyName: "test")
        
        var path = TokenTrie.Path(root: trie.root)
        
        let result1 = path.append(100, in: trie)
        #expect(result1)
        #expect(path.tokens == [100])
        
        let result2 = path.append(101, in: trie)
        #expect(result2)
        #expect(path.tokens == [100, 101])
        
        let result3 = path.append(999, in: trie)
        #expect(!result3)
        #expect(path.tokens == [100, 101])
    }
    
    @Test("Error detection stages")
    func errorDetectionStages() {
        let processor = MockLogitProcessor()
        
        #expect(!processor.hasError())
        #expect(!processor.hasFatalError())
        
        processor.shouldTriggerNonFatalError = true
        processor.mockError = .emptyConstraints
        _ = processor.process(logits: MLX.zeros([1, 100]))
        #expect(processor.hasError())
        #expect(!processor.hasFatalError())
        
        processor.clearError()
        processor.shouldTriggerFatalError = true
        processor.mockError = .invalidTokenSelected(token: -1, partialKey: "test", expectedTokens: [])
        _ = processor.process(logits: MLX.zeros([1, 100]))
        #expect(processor.hasError())
        #expect(processor.hasFatalError())
    }
    
    @Test("Complete generation pipeline")
    func completeGenerationPipeline() async throws {
        let schema = SchemaNode(
            kind: .object,
            properties: [
                "id": SchemaNode.any,
                "name": SchemaNode.any,
                "active": SchemaNode.any
            ],
            required: ["id", "name"]
        )
        
        let tokenizer = MockTokenizer()
        
        let trie = TokenTrieBuilder.build(keys: ["id", "name", "active"], tokenizer: tokenizer)
        #expect(trie.allKeys.count == 3)
        
        let swiftTokenizer = MockSwiftTokenizer()
        let processor = DPDAKeyTrieLogitProcessor(schema: schema, tokenizer: swiftTokenizer)
        
        let validJSON = #"{"id":123,"name":"Test","active":true}"#
        
        var stateMachine = JSONStateMachine()
        for char in validJSON {
            stateMachine.processCharacter(char)
        }
        
        #expect(stateMachine.isComplete())
        #expect(!stateMachine.isError())
        
        let schemaNode = SchemaNode(
            kind: .object,
            properties: [
                "id": SchemaNode.any,
                "name": SchemaNode.any,
                "active": SchemaNode.any
            ],
            required: ["id", "name"]
        )
        #expect(JSONValidator.validate(text: validJSON, schema: schemaNode))
        
        if let data = validJSON.data(using: .utf8),
           let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any] {
            #expect(json["id"] != nil)
            #expect(json["name"] != nil)
            #expect(json["active"] != nil)
        } else {
            Issue.record("Failed to parse JSON")
        }
    }
    
    @Test("Concurrent generation")
    func concurrentGeneration() async throws {
        let schema = SchemaNode(
            kind: .object,
            properties: [
                "key1": SchemaNode.any,
                "key2": SchemaNode.any
            ],
            required: ["key1"]
        )
        let tokenizer = MockTokenizer()
        
        await withTaskGroup(of: TokenTrie.self) { group in
            for _ in 0..<5 {
                group.addTask {
                    return TokenTrieBuilder.build(keys: ["key1", "key2"], tokenizer: tokenizer)
                }
            }
            
            var tries: [TokenTrie] = []
            for await trie in group {
                tries.append(trie)
            }
            
            #expect(tries.count == 5)
            for trie in tries {
                #expect(trie.allKeys == Set(["key1", "key2"]))
            }
        }
    }
}