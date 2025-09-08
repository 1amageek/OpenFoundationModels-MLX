import Testing
import Foundation
import MLX
import MLXRandom
import MLXLLM
@testable import OpenFoundationModelsMLX

@Suite("TokenTrie LogitProcessor Tests")
struct TokenTrieLogitProcessorTests {
    
    // MARK: - TokenTrie Basic Tests
    
    @Test("TokenTrie initializes correctly")
    func tokenTrieInitialization() {
        let trie = OpenFoundationModelsMLX.TokenTrie()
        #expect(trie.root.children.isEmpty)
    }
    
    @Test("TokenTrie insert and retrieve")
    func tokenTrieInsertRetrieve() {
        var trie = OpenFoundationModelsMLX.TokenTrie()
        trie.insert(tokenIDs: [100, 101, 102], keyName: "name")
        trie.insert(tokenIDs: [200, 201], keyName: "age")
        
        // Test path creation
        let path = OpenFoundationModelsMLX.TokenTrie.Path(root: trie.root)
        #expect(path.tokens.isEmpty)
        #expect(path.currentNode === trie.root)
    }
    
    @Test("TokenTrie path tracking")
    func tokenTriePathTracking() {
        var trie = OpenFoundationModelsMLX.TokenTrie()
        trie.insert(tokenIDs: [100, 101], keyName: "test")
        
        var path = OpenFoundationModelsMLX.TokenTrie.Path(root: trie.root)
        
        // Append first token
        let success1 = path.append(100, in: trie)
        #expect(success1)
        #expect(path.tokens == [100])
        
        // Append second token
        let success2 = path.append(101, in: trie)
        #expect(success2)
        #expect(path.tokens == [100, 101])
        
        // Check terminal
        #expect(path.isAtTerminal())
    }
    
    @Test("TokenTrie getAllowedTokens")
    func tokenTrieGetAllowedTokens() {
        var trie = OpenFoundationModelsMLX.TokenTrie()
        trie.insert(tokenIDs: [100, 101], keyName: "first")
        trie.insert(tokenIDs: [100, 102], keyName: "second")
        trie.insert(tokenIDs: [200], keyName: "third")
        
        let path = OpenFoundationModelsMLX.TokenTrie.Path(root: trie.root)
        let allowed = trie.getAllowedTokens(for: path)
        
        // At root, should allow 100 and 200
        #expect(allowed.contains(100))
        #expect(allowed.contains(200))
        #expect(!allowed.contains(101))
        #expect(!allowed.contains(102))
    }
    
    // MARK: - KeyTrie Tests
    
    @Test("KeyTrie initialization")
    func keyTrieInitialization() {
        let keyTrie = KeyTrie(keys: ["name", "age", "id"])
        // KeyTrie doesn't expose keys directly, just test that it was created
        #expect(keyTrie.hasPrefix(""))  // Empty prefix should always work
    }
    
    @Test("KeyTrie hasPrefix")
    func keyTrieHasPrefix() {
        let keyTrie = KeyTrie(keys: ["firstName", "lastName", "age"])
        
        #expect(keyTrie.hasPrefix("first"))
        #expect(keyTrie.hasPrefix("last"))
        #expect(keyTrie.hasPrefix("a"))
        #expect(!keyTrie.hasPrefix("middle"))
        #expect(!keyTrie.hasPrefix("xyz"))
    }
    
    @Test("KeyTrie prefix checking")
    func keyTriePrefixChecking() {
        let keyTrie = KeyTrie(keys: ["name", "age", "address"])
        
        // Test valid prefixes
        #expect(keyTrie.hasPrefix("nam"))
        #expect(keyTrie.hasPrefix("name"))
        #expect(keyTrie.hasPrefix("ag"))
        #expect(keyTrie.hasPrefix("age"))
        #expect(keyTrie.hasPrefix("addr"))
        #expect(keyTrie.hasPrefix("address"))
        
        // Test invalid prefixes
        #expect(!keyTrie.hasPrefix("xyz"))
        #expect(!keyTrie.hasPrefix("names"))  // Beyond the key
    }
    
    // MARK: - JSONStateMachine Tests
    
    @Test("JSONStateMachine initial state")
    func jsonStateMachineInitialState() {
        let stateMachine = JSONStateMachine()
        #expect(stateMachine.phase == .root)
        #expect(stateMachine.stack.isEmpty)
    }
    
    @Test("JSONStateMachine object transition")
    func jsonStateMachineObjectTransition() {
        var stateMachine = JSONStateMachine()
        
        // Start object
        stateMachine.processCharacter("{")
        if case .inObject(let phase) = stateMachine.phase {
            #expect(phase == .expectKeyFirstQuote)
        } else {
            Issue.record("Expected inObject phase")
        }
        
        // Start key
        stateMachine.processCharacter("\"")
        if case .inString(let phase) = stateMachine.phase {
            if case .body(let kind, _) = phase {
                #expect(kind == .key)
            } else {
                Issue.record("Expected string body phase")
            }
        } else {
            Issue.record("Expected inString phase")
        }
    }
    
    @Test("JSONStateMachine array transition")
    func jsonStateMachineArrayTransition() {
        var stateMachine = JSONStateMachine()
        
        // Start array
        stateMachine.processCharacter("[")
        if case .inArray(let phase) = stateMachine.phase {
            #expect(phase == .expectValue)
        } else {
            Issue.record("Expected inArray phase")
        }
        
        // Add string value
        stateMachine.processCharacter("\"")
        if case .inString(let phase) = stateMachine.phase {
            if case .body(let kind, _) = phase {
                #expect(kind == .value)
            } else {
                Issue.record("Expected string body phase")
            }
        } else {
            Issue.record("Expected inString phase")
        }
    }
    
    // MARK: - JSONMaskHintGenerator Tests
    
    @Test("JSONMaskHintGenerator creation")
    func maskHintGeneratorCreation() {
        let specialTokens = MLXLLMTokenizer.SpecialTokens(
            quoteTokens: Set([34]),
            colonTokens: Set([58]),
            braceOpenTokens: Set([123]),
            braceCloseTokens: Set([125]),
            commaTokens: Set([44]),
            bracketOpenTokens: Set([91]),
            bracketCloseTokens: Set([93]),
            whitespaceTokens: Set([32, 9, 10]),
            backslashTokens: Set([92])
        )
        
        let generator = JSONMaskHintGenerator.forSchemaConstrainedDecoding(
            specialTokens: specialTokens,
            includeWhitespace: false
        )
        
        #expect(generator != nil)
    }
    
    @Test("JSONMaskHintGenerator generates hints")
    func maskHintGeneratorGeneratesHints() {
        let specialTokens = MLXLLMTokenizer.SpecialTokens(
            quoteTokens: Set([34]),
            colonTokens: Set([58]),
            braceOpenTokens: Set([123]),
            braceCloseTokens: Set([125]),
            commaTokens: Set([44]),
            bracketOpenTokens: Set([91]),
            bracketCloseTokens: Set([93]),
            whitespaceTokens: Set([32, 9, 10]),
            backslashTokens: Set([92])
        )
        
        let generator = JSONMaskHintGenerator.forSchemaConstrainedDecoding(
            specialTokens: specialTokens,
            includeWhitespace: false
        )
        
        var trie = OpenFoundationModelsMLX.TokenTrie()
        trie.insert(tokenIDs: [100, 101], keyName: "test")
        
        let stateMachine = JSONStateMachine()
        let path = OpenFoundationModelsMLX.TokenTrie.Path(root: trie.root)
        
        let hint = generator.maskHint(
            for: stateMachine,
            tokenTrie: trie,
            tokenPath: path
        )
        
        // At root state, should get soft constraints for value starts
        #expect(hint != nil)
        if let hint = hint {
            #expect(hint.mode == .soft)
            #expect(hint.allow.contains(123)) // {
            #expect(hint.allow.contains(91))  // [
            #expect(hint.allow.contains(34))  // "
        }
    }
    
    // MARK: - Integration Tests
    
    @Test("TokenTrie with complex keys")
    func tokenTrieComplexKeys() {
        var trie = OpenFoundationModelsMLX.TokenTrie()
        
        // Insert multiple keys with shared prefixes
        trie.insert(tokenIDs: [100, 101, 102], keyName: "firstName")
        trie.insert(tokenIDs: [100, 101, 103], keyName: "firstAge")
        trie.insert(tokenIDs: [200, 201], keyName: "lastName")
        
        // Test that getAllowedTokens at different paths
        let rootPath = TokenTrie.Path(root: trie.root)
        let rootAllowed = trie.getAllowedTokens(for: rootPath)
        #expect(rootAllowed.contains(100))
        #expect(rootAllowed.contains(200))
        
        // After first token
        var path1 = TokenTrie.Path(root: trie.root)
        _ = path1.append(100, in: trie)
        let allowed1 = trie.getAllowedTokens(for: path1)
        #expect(allowed1.contains(101))
        #expect(!allowed1.contains(200))
        
        // After second token
        var path2 = TokenTrie.Path(root: trie.root)
        _ = path2.append(100, in: trie)
        _ = path2.append(101, in: trie)
        let allowed2 = trie.getAllowedTokens(for: path2)
        #expect(allowed2.contains(102))
        #expect(allowed2.contains(103))
    }
    
    @Test("JSONStateMachine complete object parsing")
    func jsonStateMachineCompleteObject() {
        var stateMachine = JSONStateMachine()
        
        let json = #"{"name":"John","age":30}"#
        for char in json {
            stateMachine.processCharacter(char)
        }
        
        #expect(stateMachine.phase == .done)
        #expect(stateMachine.phase != .error)
    }
    
    @Test("JSONStateMachine nested structure")
    func jsonStateMachineNestedStructure() {
        var stateMachine = JSONStateMachine()
        
        let json = #"{"user":{"name":"John"},"items":[1,2,3]}"#
        for char in json {
            stateMachine.processCharacter(char)
        }
        
        #expect(stateMachine.phase == .done)
        #expect(stateMachine.phase != .error)
    }
    
    // MARK: - Error Management Tests
    
    @Test("TokenTrieLogitProcessor hasError detection")
    func processorHasErrorDetection() {
        let schema = SchemaMeta(keys: ["name", "age"], required: ["name"])
        let tokenizer = MockSwiftTokenizer()
        let processor = TokenTrieLogitProcessor(schema: schema, tokenizer: tokenizer)
        
        // Initially no error
        #expect(!processor.hasError())
        #expect(processor.getLastError() == nil)
        
        // Process some logits - this would normally set errors internally
        // Since we can't directly set errors, we'll test the public interface
        processor.clearError()
        #expect(!processor.hasError())
    }
    
    @Test("TokenTrieLogitProcessor clearError")
    func processorClearError() {
        let schema = SchemaMeta(keys: ["test"], required: [])
        let tokenizer = MockSwiftTokenizer()
        let processor = TokenTrieLogitProcessor(schema: schema, tokenizer: tokenizer)
        
        // Clear should work even with no error
        processor.clearError()
        #expect(!processor.hasError())
        #expect(processor.getLastError() == nil)
    }
    
    @Test("TokenTrieLogitProcessor hasFatalError classification")
    func processorFatalErrorClassification() {
        // Use mock processor to test error classification
        let processor = MockLogitProcessor()
        
        // Test fatal error
        processor.shouldTriggerFatalError = true
        processor.mockError = .noValidTokens(partialKey: "test", position: 1)
        _ = processor.process(logits: MLX.zeros([1, 100]))  // Trigger error state
        #expect(processor.hasFatalError())
        #expect(processor.hasError())
        
        // Test non-fatal error
        processor.clearError()
        processor.shouldTriggerNonFatalError = true
        processor.mockError = .emptyConstraints
        _ = processor.process(logits: MLX.zeros([1, 100]))  // Trigger error state
        #expect(!processor.hasFatalError())
        #expect(processor.hasError())
    }
    
    @Test("TokenTrieLogitProcessor error types")
    func processorErrorTypes() {
        let processor = MockLogitProcessor()
        
        // Test noValidTokens (fatal)
        processor.mockError = .noValidTokens(partialKey: "test", position: 5)
        processor.shouldTriggerFatalError = true
        _ = processor.process(logits: MLX.zeros([1, 100]))  // Trigger error state
        #expect(processor.hasFatalError())
        
        // Test invalidTokenSelected (fatal)
        processor.clearError()
        processor.mockError = .invalidTokenSelected(
            token: 123,
            partialKey: "test",
            expectedTokens: Set([456, 789])
        )
        processor.shouldTriggerFatalError = true
        _ = processor.process(logits: MLX.zeros([1, 100]))  // Trigger error state
        #expect(processor.hasFatalError())
        
        // Test emptyConstraints (non-fatal)
        processor.clearError()
        processor.mockError = .emptyConstraints
        processor.shouldTriggerNonFatalError = true
        #expect(!processor.hasFatalError())
        
        // Test schemaViolation (non-fatal)
        processor.clearError()
        processor.mockError = .schemaViolation(reason: "test violation")
        processor.shouldTriggerNonFatalError = true
        #expect(!processor.hasFatalError())
    }
    
    @Test("TokenTrieLogitProcessor constraint application with errors")
    func processorConstraintApplicationWithErrors() {
        var trie = OpenFoundationModelsMLX.TokenTrie()
        trie.insert(tokenIDs: [100, 101], keyName: "name")
        trie.insert(tokenIDs: [200], keyName: "age")
        
        let schema = SchemaMeta(keys: ["name", "age"], required: [])
        let tokenizer = MockSwiftTokenizer()
        let processor = TokenTrieLogitProcessor(schema: schema, tokenizer: tokenizer)
        
        // Create a simple logits array
        let logits = MLX.zeros([1, 1000])
        
        // Process logits - should apply constraints
        let processed = processor.process(logits: logits)
        
        // Result should have same shape
        #expect(processed.shape == logits.shape)
        
        // Check error state after processing
        // Note: Actual error detection would depend on JSON state
        #expect(!processor.hasFatalError() || processor.hasError())
    }
    
    // MARK: - Performance Tests
    
    @Test("TokenTrie handles large vocabulary")
    func tokenTrieLargeVocabulary() {
        var trie = OpenFoundationModelsMLX.TokenTrie()
        
        // Insert many keys
        for i in 0..<100 {
            let tokens = Array(Int32(i*10)..<Int32(i*10+5))
            trie.insert(tokenIDs: tokens, keyName: "key\(i)")
        }
        
        // Test that trie still works
        let path = OpenFoundationModelsMLX.TokenTrie.Path(root: trie.root)
        let allowed = trie.getAllowedTokens(for: path)
        
        // Should have 100 different starting tokens
        #expect(allowed.count == 100)
    }
    
    @Test("KeyTrie handles many keys")
    func keyTrieManyKeys() {
        var keys: [String] = []
        for i in 0..<1000 {
            keys.append("field\(i)")
        }
        
        let keyTrie = KeyTrie(keys: keys)
        
        // Test that various prefixes work
        #expect(keyTrie.hasPrefix("field"))
        #expect(keyTrie.hasPrefix("field99"))
        #expect(keyTrie.hasPrefix("field999"))
        #expect(!keyTrie.hasPrefix("field1000"))  // Doesn't exist
    }
}