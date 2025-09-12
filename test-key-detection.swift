#!/usr/bin/env swift

import Foundation
import MLX
import OpenFoundationModelsMLX

// Mock tokenizer for testing
class TestTokenizer: TokenizerAdapter {
    var tokens: [String] = []
    var currentIndex = 0
    
    func encode(_ text: String) -> [Int32] {
        return [0]
    }
    
    func decode(_ tokens: [Int32]) -> String {
        guard currentIndex < self.tokens.count else { return "" }
        let result = self.tokens[currentIndex]
        currentIndex += 1
        return result
    }
    
    func reset() {
        currentIndex = 0
    }
}

// Test delayed evaluation with multi-character tokens
func testDelayedEvaluation() {
    print("🧪 Testing KeyDetectionLogitProcessor with delayed evaluation\n")
    print("=" * 60)
    
    let tokenizer = TestTokenizer()
    let processor = KeyDetectionLogitProcessor(
        tokenizer: tokenizer,
        verbose: true,
        topK: 3,
        showProbabilities: true
    )
    
    // Simulate tokens that would cause the old implementation to fail
    // Token sequence: { " name " : " John ", \n "  age
    let tokenSequence = [
        "{",
        "\"",
        "name",
        "\"",
        " : ",        // Multi-character token with colon and space
        "\"John\",",  // Multi-character token with value and comma
        "\n  \"",     // Multi-character token with newline and quote
        "age",
        "\""
    ]
    
    tokenizer.tokens = tokenSequence
    
    // Initialize
    processor.prompt(MLX.zeros([1]))
    
    print("\n📝 Processing tokens:\n")
    
    for (index, token) in tokenSequence.enumerated() {
        tokenizer.currentIndex = index
        
        print("Token \(index): \"\(token)\"")
        
        // Create dummy logits
        let logits = MLX.zeros([1, 50000])
        
        // Process logits (this now shows entropy for PREVIOUS token)
        _ = processor.process(logits: logits)
        
        // Simulate sampling
        let sampledToken = MLX.array([Int32(index)])
        processor.didSample(token: sampledToken)
        
        print("")
    }
    
    // Process final pending info
    processor.finish()
    
    print("\n" + "=" * 60)
    print("✅ Test completed\n")
}

// Helper to repeat string
extension String {
    static func * (left: String, right: Int) -> String {
        return String(repeating: left, count: right)
    }
}

// Run test
testDelayedEvaluation()