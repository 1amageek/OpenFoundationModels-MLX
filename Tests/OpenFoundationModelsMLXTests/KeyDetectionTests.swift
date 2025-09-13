import Testing
import Foundation
import MLX
import MLXLMCommon
@testable import OpenFoundationModelsMLX

@Suite("Key Detection Tests")
struct KeyDetectionTests {
    
    // Mock tokenizer for testing
    final class MockTokenizer: TokenizerAdapter {
        private let tokenMap: [Int32: String]
        
        init(tokenMap: [Int32: String] = [:]) {
            self.tokenMap = tokenMap
        }
        
        func encode(_ text: String) -> [Int32] {
            // Simple character-based encoding for testing
            return text.map { Int32($0.asciiValue ?? 0) }
        }
        
        func decode(_ tokens: [Int32]) -> String {
            if let customText = tokenMap[tokens.first ?? -1] {
                return customText
            }
            // Default to character decoding
            return String(tokens.compactMap { 
                if let scalar = UnicodeScalar(Int($0)) {
                    return Character(scalar)
                }
                return nil
            })
        }
        
        var eosTokenId: Int32? { return 0 }
        var bosTokenId: Int32? { return 1 }
        
        func getVocabSize() -> Int? { return 50000 }
        func fingerprint() -> String { return "mock-tokenizer" }
    }
    
    @Test("Detect simple object keys")
    func detectSimpleKeys() {
        let tokenizer = MockTokenizer()
        var processor = KeyDetectionLogitProcessor(tokenizer: tokenizer, verbose: false)
        
        // Initialize processor
        let prompt = MLX.zeros([1, 10])
        processor.prompt(prompt)
        
        // Simulate generating: {"name":"John","age":30}
        let tokens: [(Int32, String)] = [
            (123, "{"),     // {
            (34, "\""),     // "
            (110, "n"),     // n
            (97, "a"),      // a
            (109, "m"),     // m
            (101, "e"),     // e
            (34, "\""),     // "
            (58, ":"),      // :
            (34, "\""),     // "
            (74, "J"),      // J
            (111, "o"),     // o
            (104, "h"),     // h
            (110, "n"),     // n
            (34, "\""),     // "
            (44, ","),      // ,
            (34, "\""),     // "
            (97, "a"),      // a
            (103, "g"),     // g
            (101, "e"),     // e
            (34, "\""),     // "
            (58, ":"),      // :
            (51, "3"),      // 3
            (48, "0"),      // 0
            (125, "}")      // }
        ]
        
        // Process each token
        for (tokenId, _) in tokens {
            let token = MLXArray([tokenId])
            processor.didSample(token: token)
        }
        
        // Check detected keys
        let detectedKeys = processor.allDetectedKeys
        #expect(detectedKeys.contains("name"))
        #expect(detectedKeys.contains("age"))
        #expect(detectedKeys.count == 2)
    }
    
    @Test("Detect nested object keys")
    func detectNestedKeys() {
        let tokenizer = MockTokenizer()
        var processor = KeyDetectionLogitProcessor(tokenizer: tokenizer, verbose: false)
        
        // Initialize processor
        let prompt = MLX.zeros([1, 10])
        processor.prompt(prompt)
        
        // Simulate generating: {"user":{"name":"Alice","email":"alice@example.com"}}
        let jsonString = "{\"user\":{\"name\":\"Alice\",\"email\":\"alice@example.com\"}}"
        
        // Process each character as a token
        for char in jsonString {
            let tokenId = Int32(char.asciiValue ?? 0)
            let token = MLXArray([tokenId])
            processor.didSample(token: token)
        }
        
        // Check detected keys
        let detectedKeys = processor.allDetectedKeys
        #expect(detectedKeys.contains("user"))
        #expect(detectedKeys.contains("name"))
        #expect(detectedKeys.contains("email"))
        #expect(detectedKeys.count == 3)
    }
    
    @Test("Track JSON parsing phases")
    func trackParsingPhases() {
        let tokenizer = MockTokenizer()
        var processor = KeyDetectionLogitProcessor(tokenizer: tokenizer, verbose: false)
        
        // Initialize processor
        let prompt = MLX.zeros([1, 10])
        processor.prompt(prompt)
        
        // Start with root phase
        #expect(processor.currentPhase == .root)
        
        // Process opening brace
        processor.didSample(token: MLXArray([123])) // {
        if case .inObject = processor.currentPhase {
            // Expected
        } else {
            Issue.record("Expected to be in object phase")
        }
        
        // Process opening quote for key
        processor.didSample(token: MLXArray([34])) // "
        if case .inString(.body(kind: .key, escaped: false)) = processor.currentPhase {
            // Expected
        } else {
            Issue.record("Expected to be in string key phase")
        }
        
        // Process key characters
        processor.didSample(token: MLXArray([107])) // k
        processor.didSample(token: MLXArray([101])) // e
        processor.didSample(token: MLXArray([121])) // y
        
        // Still in key generation
        #expect(processor.isGeneratingKey == true)
        
        // Close key quote
        processor.didSample(token: MLXArray([34])) // "
        
        // Should no longer be generating key
        #expect(processor.isGeneratingKey == false)
        
        // Check that key was detected
        let detectedKeys = processor.allDetectedKeys
        #expect(detectedKeys.contains("key"))
    }
    
    @Test("Handle array with objects")
    func handleArrayWithObjects() {
        let tokenizer = MockTokenizer()
        var processor = KeyDetectionLogitProcessor(tokenizer: tokenizer, verbose: false)
        
        // Initialize processor
        let prompt = MLX.zeros([1, 10])
        processor.prompt(prompt)
        
        // Simulate generating: [{"id":1},{"id":2}]
        let jsonString = "[{\"id\":1},{\"id\":2}]"
        
        // Process each character as a token
        for char in jsonString {
            let tokenId = Int32(char.asciiValue ?? 0)
            let token = MLXArray([tokenId])
            processor.didSample(token: token)
        }
        
        // Check detected keys (should detect "id" twice)
        let detectedKeys = processor.allDetectedKeys
        #expect(detectedKeys.filter { $0 == "id" }.count == 2)
    }
    
    @Test("Handle escape sequences in keys")
    func handleEscapeSequences() {
        let tokenizer = MockTokenizer()
        var processor = KeyDetectionLogitProcessor(tokenizer: tokenizer, verbose: false)
        
        // Initialize processor
        let prompt = MLX.zeros([1, 10])
        processor.prompt(prompt)
        
        // Simulate generating: {"key\"with\"quotes":"value"}
        let tokens: [Int32] = [
            123, // {
            34,  // "
            107, 101, 121, // key
            92, 34, // \"
            119, 105, 116, 104, // with
            92, 34, // \"
            113, 117, 111, 116, 101, 115, // quotes
            34,  // "
            58,  // :
            34,  // "
            118, 97, 108, 117, 101, // value
            34,  // "
            125  // }
        ]
        
        // Process each token
        for tokenId in tokens {
            let token = MLXArray([tokenId])
            processor.didSample(token: token)
        }
        
        // Check that the escaped key was properly detected
        let detectedKeys = processor.allDetectedKeys
        #expect(detectedKeys.contains("key\"with\"quotes"))
    }
    
    @Test("JSONStateMachine basic transitions")
    func stateMachineTransitions() {
        var machine = JSONStateMachine()
        
        // Initial state
        #expect(machine.phase == .root)
        #expect(machine.isComplete == false)
        #expect(machine.isError == false)
        
        // Start object
        machine.processCharacter("{")
        if case .inObject(.expectKeyOrEnd) = machine.phase {
            // Expected
        } else {
            Issue.record("Expected to be in object.expectKeyOrEnd phase")
        }
        
        // Start key
        machine.processCharacter("\"")
        if case .inString(.body(kind: .key, escaped: false)) = machine.phase {
            // Expected
        } else {
            Issue.record("Expected to be in string key phase")
        }
        
        // Key content
        machine.processCharacter("k")
        machine.processCharacter("e")
        machine.processCharacter("y")
        
        // End key
        machine.processCharacter("\"")
        if case .inObject(.expectColon) = machine.phase {
            // Expected
        } else {
            Issue.record("Expected to be expecting colon")
        }
        
        // Colon
        machine.processCharacter(":")
        if case .inObject(.expectValue) = machine.phase {
            // Expected
        } else {
            Issue.record("Expected to be expecting value")
        }
        
        // String value
        machine.processCharacter("\"")
        if case .inString(.body(kind: .value, escaped: false)) = machine.phase {
            // Expected
        } else {
            Issue.record("Expected to be in string value phase")
        }
        
        // Value content
        machine.processCharacter("v")
        machine.processCharacter("a")
        machine.processCharacter("l")
        
        // End value
        machine.processCharacter("\"")
        if case .inObject(.expectCommaOrEnd) = machine.phase {
            // Expected
        } else {
            Issue.record("Expected to be expecting comma or end")
        }
        
        // End object
        machine.processCharacter("}")
        #expect(machine.phase == .done)
        #expect(machine.isComplete == true)
    }
    
    @Test("JSONStateMachine error handling")
    func stateMachineErrorHandling() {
        var machine = JSONStateMachine()
        
        // Invalid character at root
        machine.processCharacter("x")
        #expect(machine.phase == .error)
        #expect(machine.isError == true)
        
        // Reset and test invalid sequence
        machine.reset()
        machine.processCharacter("{")
        machine.processCharacter("x") // Invalid: expecting quote or }
        #expect(machine.phase == .error)
    }
}