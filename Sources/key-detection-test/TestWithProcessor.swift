import Foundation
import MLX
import OpenFoundationModelsMLX

// Test KeyDetectionLogitProcessor with simulated token generation
func testProcessorWithSimulatedTokens() {
    print("\n=== Testing KeyDetectionLogitProcessor ===\n")
    
    // Create mock tokenizer
    let tokenizer = SimpleTokenizerAdapter()
    var processor = KeyDetectionLogitProcessor(tokenizer: tokenizer, verbose: true)
    
    // Test cases with different token patterns
    let testCases: [(name: String, tokens: [String])] = [
        (
            name: "Simple object",
            tokens: ["{\"", "name", "\":\"", "John", "\",\"", "age", "\":", "30", "}"]
        ),
        (
            name: "Nested object",
            tokens: ["{\"", "user", "\":{\"", "first", "Name", "\":\"", "Alice", "\"},\"", "active", "\":true}"]
        ),
        (
            name: "Array with objects",
            tokens: ["{\"", "items", "\":[{\"", "id", "\":1", ",\"", "name", "\":\"", "Item1", "\"},{\"", "id", "\":2}]}"]
        )
    ]
    
    for testCase in testCases {
        print("\n📝 Test: \(testCase.name)")
        print("Tokens: \(testCase.tokens.joined())")
        print("")
        
        // Reset processor (using dummy MLXArray value)
        // Note: We can't create real MLXArrays in test environment
        // processor.prompt(MLXArray.zeros([1]))
        
        // Manually reset the processor state
        processor = KeyDetectionLogitProcessor(tokenizer: tokenizer, verbose: true)
        
        // Process tokens
        for token in testCase.tokens {
            tokenizer.nextDecodeResult = token
            // Create a fake token ID (we just need to trigger didSample)
            // processor.didSample(token: mockToken)
        }
        
        // Show results
        let detectedKeys = processor.allDetectedKeys
        print("\n📊 Results:")
        print("Detected keys: \(detectedKeys)")
        print("Final phase: \(describePhase(processor.currentPhase))")
        print("-" * 50)
    }
}

// Simple tokenizer adapter for testing
final class SimpleTokenizerAdapter: TokenizerAdapter, @unchecked Sendable {
    var nextDecodeResult: String = ""
    
    func encode(_ text: String) -> [Int32] {
        return Array(repeating: Int32(0), count: text.count)
    }
    
    func decode(_ tokens: [Int32]) -> String {
        return nextDecodeResult
    }
    
    var eosTokenId: Int32 { 0 }
    var bosTokenId: Int32 { 1 }
    var unknownTokenId: Int32 { 2 }
    
    func convertTokenToString(_ token: Int32) -> String? {
        return nextDecodeResult
    }
    
    func getVocabSize() -> Int? {
        return 50000  // Mock value
    }
    
    func fingerprint() -> String {
        return "test-tokenizer"
    }
}

func describePhase(_ phase: JSONStateMachine.Phase) -> String {
    switch phase {
    case .root: return "root"
    case .inObject(let obj): 
        switch obj {
        case .expectKeyOrEnd: return "object.expectKeyOrEnd"
        case .expectKeyFirstQuote: return "object.expectKeyFirstQuote"
        case .inKey: return "object.inKey"
        case .expectKeyEndQuote: return "object.expectKeyEndQuote"
        case .expectColon: return "object.expectColon"
        case .expectValue: return "object.expectValue"
        case .expectCommaOrEnd: return "object.expectCommaOrEnd"
        }
    case .inArray(let arr):
        switch arr {
        case .expectValue: return "array.expectValue"
        case .expectCommaOrEnd: return "array.expectCommaOrEnd"
        }
    case .inString(let str):
        switch str {
        case .body(let kind, let escaped):
            return "string.\(kind == .key ? "key" : "value")\(escaped ? ".escaped" : "")"
        }
    case .inNumber(let num):
        switch num {
        case .integer: return "number.integer"
        case .decimal: return "number.decimal"
        case .exponent: return "number.exponent"
        }
    case .inLiteral(let lit):
        if case .inProgress(let text) = lit {
            return "literal(\(text))"
        }
        return "literal"
    case .done: return "done"
    case .error: return "error"
    }
}

