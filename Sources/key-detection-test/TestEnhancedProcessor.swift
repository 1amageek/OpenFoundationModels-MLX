import Foundation
import MLX
import OpenFoundationModelsMLX

// Test the enhanced KeyDetectionLogitProcessor
func testEnhancedProcessor() {
    print("\n=== Testing Enhanced KeyDetectionLogitProcessor ===\n")
    print("This processor now includes:")
    print("  • Probability distribution analysis")
    print("  • Entropy calculation (confidence measurement)")
    print("  • Top-K candidate display during key generation")
    print("  • Visual probability bars")
    print("")
    
    // Test with different configurations
    let configurations: [(name: String, verbose: Bool, showProb: Bool, topK: Int)] = [
        ("Basic (no probabilities)", true, false, 5),
        ("With probabilities (top-3)", true, true, 3),
        ("With probabilities (top-5)", true, true, 5),
    ]
    
    let testJSON = #"{"user":{"name":"Alice","age":30},"status":"active"}"#
    
    for config in configurations {
        print("\n" + String(repeating: "=", count: 60))
        print("Configuration: \(config.name)")
        print("  verbose: \(config.verbose), showProbabilities: \(config.showProb), topK: \(config.topK)")
        print(String(repeating: "=", count: 60))
        
        // Create tokenizer adapter (mock)
        let tokenizer = MockEnhancedTokenizer()
        var processor = KeyDetectionLogitProcessor(
            tokenizer: tokenizer,
            verbose: config.verbose,
            topK: config.topK,
            showProbabilities: config.showProb
        )
        
        // Initialize processor
        processor.prompt(MLXArray.zeros([1]))
        
        // Simulate token-by-token generation
        let tokens = simulateTokenization(testJSON)
        
        for (i, token) in tokens.enumerated() {
            tokenizer.nextDecodeResult = token
            
            // In real usage, process() would be called with actual logits
            // Here we're just testing the key detection part
            let mockToken = MLXArray(Int32(i))
            processor.didSample(token: mockToken)
        }
        
        // Show results
        print("\n📊 Summary:")
        print("  Detected keys: \(processor.allDetectedKeys)")
        print("  Expected: [\"user\", \"name\", \"age\", \"status\"]")
        
        let expected = ["user", "name", "age", "status"]
        if processor.allDetectedKeys == expected {
            print("  ✅ All keys detected correctly!")
        } else {
            print("  ❌ Key detection mismatch")
        }
    }
    
    print("\n" + "=" * 60)
    print("Enhanced processor test complete!")
    print("=" * 60)
}

// Simulate tokenization of JSON
func simulateTokenization(_ json: String) -> [String] {
    // Simulate realistic token boundaries
    var tokens: [String] = []
    var current = ""
    
    for char in json {
        if char == "{" || char == "}" || char == "[" || char == "]" ||
           char == ":" || char == "," {
            if !current.isEmpty {
                tokens.append(current)
                current = ""
            }
            tokens.append(String(char))
        } else if char == "\"" {
            if !current.isEmpty && !current.hasPrefix("\"") {
                tokens.append(current)
                current = String(char)
            } else {
                current.append(char)
                if current.count > 1 && current.hasPrefix("\"") && current.hasSuffix("\"") {
                    tokens.append(current)
                    current = ""
                }
            }
        } else {
            current.append(char)
        }
    }
    
    if !current.isEmpty {
        tokens.append(current)
    }
    
    return tokens
}

// Enhanced mock tokenizer
class MockEnhancedTokenizer: TokenizerAdapter, @unchecked Sendable {
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
        return 50000
    }
    
    func fingerprint() -> String {
        return "mock-enhanced-tokenizer"
    }
}

