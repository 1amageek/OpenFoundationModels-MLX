import Foundation
import MLX
import OpenFoundationModelsMLX

// Test incomplete JSON (LLM-style token-by-token generation)
func testIncompleteJSON() {
    print("\n=== Testing Incomplete JSON (LLM-style token-by-token) ===\n")
    
    // LLMが生成するような不完全なJSONのステップ
    let incompleteJSONSteps = [
        "{",
        "{\n",
        "{\n  \"",
        "{\n  \"name",
        "{\n  \"name\"",
        "{\n  \"name\":",
        "{\n  \"name\": \"",
        "{\n  \"name\": \"John",
        "{\n  \"name\": \"John\"",
        "{\n  \"name\": \"John\",",
        "{\n  \"name\": \"John\",\n  \"",
        "{\n  \"name\": \"John\",\n  \"age",
        "{\n  \"name\": \"John\",\n  \"age\"",
        "{\n  \"name\": \"John\",\n  \"age\":",
        "{\n  \"name\": \"John\",\n  \"age\": 30",
        "{\n  \"name\": \"John\",\n  \"age\": 30,",
        "{\n  \"name\": \"John\",\n  \"age\": 30,\n  \"",
        "{\n  \"name\": \"John\",\n  \"age\": 30,\n  \"nested",
        "{\n  \"name\": \"John\",\n  \"age\": 30,\n  \"nested\"",
        "{\n  \"name\": \"John\",\n  \"age\": 30,\n  \"nested\": {",
        "{\n  \"name\": \"John\",\n  \"age\": 30,\n  \"nested\": {\n    \"",
        "{\n  \"name\": \"John\",\n  \"age\": 30,\n  \"nested\": {\n    \"key",
        "{\n  \"name\": \"John\",\n  \"age\": 30,\n  \"nested\": {\n    \"key\"",
    ]
    
    let tokenizer = MockEnhancedTokenizer()
    var processor = KeyDetectionLogitProcessor(
        tokenizer: tokenizer,
        verbose: true,
        topK: 5,
        showProbabilities: false  // Disable probability display for clarity
    )
    
    processor.prompt(MLXArray.zeros([1]))
    
    for (i, json) in incompleteJSONSteps.enumerated() {
        print("\nStep \(i + 1): Processing partial JSON:")
        print("  Input: \(json.replacingOccurrences(of: "\n", with: "\\n"))")
        
        // Process the last character(s) added
        if i > 0 {
            let prevJSON = incompleteJSONSteps[i - 1]
            let newChars = String(json.dropFirst(prevJSON.count))
            
            for char in newChars {
                tokenizer.nextDecodeResult = String(char)
                let mockToken = MLXArray(Int32(i * 10 + Int(char.unicodeScalars.first?.value ?? 0)))
                processor.didSample(token: mockToken)
            }
        } else {
            // Process the first character
            tokenizer.nextDecodeResult = json
            let mockToken = MLXArray(Int32(0))
            processor.didSample(token: mockToken)
        }
        
        // Check if key is being generated
        print("  Is in key generation: \(processor.isGeneratingKey)")
        print("  Current phase: \(processor.currentPhase)")
        print("  Detected keys so far: \(processor.allDetectedKeys)")
    }
}

// Test token-split keys (real LLM behavior)
func testTokenSplitKeys() {
    print("\n=== Testing Token-Split Keys (Real LLM Behavior) ===\n")
    
    // LLMがキーを複数トークンに分割する場合
    let tokenSequences: [(String, [String])] = [
        // Case 1: キーが複数トークンに分割
        ("Key split across tokens", ["{\n  \"", "desc", "ription", "\"", ": \""]),
        
        // Case 2: 引用符とキーが一体化
        ("Quote and key combined", ["{\n  ", "\"name\"", ": \""]),
        
        // Case 3: キーの途中で分割
        ("Key split mid-word", ["{\n  \"", "head", "Count", "\"", ": "]),
        
        // Case 4: 配列内のオブジェクトのキー
        ("Key in array object", ["[{\"", "id", "\"", ": 1, \"", "status", "\"", ": \"active\"}]"]),
    ]
    
    for (caseNum, (description, tokens)) in tokenSequences.enumerated() {
        print("\n--- Case \(caseNum + 1): \(description) ---")
        print("Tokens: \(tokens)")
        
        let tokenizer = MockEnhancedTokenizer()
        var processor = KeyDetectionLogitProcessor(
            tokenizer: tokenizer,
            verbose: false,  // Disable verbose for cleaner output
            topK: 5,
            showProbabilities: false
        )
        
        processor.prompt(MLXArray.zeros([1]))
        
        var generatedText = ""
        for (i, token) in tokens.enumerated() {
            tokenizer.nextDecodeResult = token
            let mockToken = MLXArray(Int32(i))
            
            print("  Token \(i): \"\(token)\"")
            processor.didSample(token: mockToken)
            generatedText += token
            
            if processor.isGeneratingKey {
                print("    -> Currently generating key")
            }
        }
        
        print("  Generated: \(generatedText)")
        print("  Final detected keys: \(processor.allDetectedKeys)")
    }
}

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

