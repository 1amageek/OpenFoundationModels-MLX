import Foundation
import Testing
import MLX
import MLXLMCommon
@testable import OpenFoundationModelsMLX

@Suite("KeyDetectionLogitProcessor Tests")
struct KeyDetectionLogitProcessorTests {
    
    @Test("Detects keys in simple object")
    func testSimpleObjectKeyDetection() {
        // Create a mock tokenizer
        let tokenizer = MockTokenizerAdapter()
        var processor = KeyDetectionLogitProcessor(tokenizer: tokenizer, verbose: false)
        
        // Simulate JSON generation token by token
        let jsonString = #"{"name":"John","age":30}"#
        
        // Initialize processor
        processor.prompt(MLXArray.zeros([1]))
        
        // Process each character as a token
        for char in jsonString {
            let mockToken = MLXArray(Int32(0))  // Token ID doesn't matter for this test
            tokenizer.nextDecodeResult = String(char)
            processor.didSample(token: mockToken)
        }
        
        // Check detected keys
        let detectedKeys = processor.allDetectedKeys
        #expect(detectedKeys == ["name", "age"])
    }
    
    @Test("Detects keys in nested objects")
    func testNestedObjectKeyDetection() {
        let tokenizer = MockTokenizerAdapter()
        var processor = KeyDetectionLogitProcessor(tokenizer: tokenizer, verbose: false)
        
        let jsonString = #"{"user":{"firstName":"Alice","lastName":"Smith"},"active":true}"#
        
        processor.prompt(MLXArray.zeros([1]))
        
        for char in jsonString {
            let mockToken = MLXArray(Int32(0))
            tokenizer.nextDecodeResult = String(char)
            processor.didSample(token: mockToken)
        }
        
        let detectedKeys = processor.allDetectedKeys
        #expect(detectedKeys == ["user", "firstName", "lastName", "active"])
    }
    
    @Test("Detects keys in arrays with objects")
    func testArrayObjectKeyDetection() {
        let tokenizer = MockTokenizerAdapter()
        var processor = KeyDetectionLogitProcessor(tokenizer: tokenizer, verbose: false)
        
        let jsonString = #"{"items":[{"id":1,"name":"Item1"},{"id":2,"name":"Item2"}],"total":2}"#
        
        processor.prompt(MLXArray.zeros([1]))
        
        for char in jsonString {
            let mockToken = MLXArray(Int32(0))
            tokenizer.nextDecodeResult = String(char)
            processor.didSample(token: mockToken)
        }
        
        let detectedKeys = processor.allDetectedKeys
        #expect(detectedKeys == ["items", "id", "name", "id", "name", "total"])
    }
    
    @Test("Handles JSON with prefix text")
    func testJSONWithPrefix() {
        let tokenizer = MockTokenizerAdapter()
        var processor = KeyDetectionLogitProcessor(tokenizer: tokenizer, verbose: false)
        
        let jsonString = "Here is the JSON response: {\"status\":\"success\",\"code\":200}"
        
        processor.prompt(MLXArray.zeros([1]))
        
        for char in jsonString {
            let mockToken = MLXArray(Int32(0))
            tokenizer.nextDecodeResult = String(char)
            processor.didSample(token: mockToken)
        }
        
        let detectedKeys = processor.allDetectedKeys
        #expect(detectedKeys == ["status", "code"])
    }
    
    @Test("Handles multi-character tokens")
    func testMultiCharacterTokens() {
        let tokenizer = MockTokenizerAdapter()
        var processor = KeyDetectionLogitProcessor(tokenizer: tokenizer, verbose: false)
        
        processor.prompt(MLXArray.zeros([1]))
        
        // Simulate tokens that decode to multiple characters
        let tokens = [
            "{\"",
            "name",
            "\":\"",
            "John",
            "\",\"",
            "age",
            "\":",
            "30",
            "}"
        ]
        
        for tokenText in tokens {
            let mockToken = MLXArray(Int32(0))
            tokenizer.nextDecodeResult = tokenText
            processor.didSample(token: mockToken)
        }
        
        let detectedKeys = processor.allDetectedKeys
        #expect(detectedKeys == ["name", "age"])
    }
}

// Mock tokenizer for testing
class MockTokenizerAdapter: TokenizerAdapter {
    var nextDecodeResult: String = ""
    
    func encode(_ text: String) -> [Int32] {
        // Simple mock: one token per character
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
}