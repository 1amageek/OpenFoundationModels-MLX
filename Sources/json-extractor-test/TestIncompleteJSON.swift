import Foundation
import OpenFoundationModelsMLX

// Test JSONExtractor with incomplete/streaming JSON as LLMs generate
func testIncompleteJSON() {
    print("\n📝 Testing JSONExtractor with Incomplete/Streaming JSON")
    print("=" * 60)
    
    // Test 1: GPT OSS format with incomplete JSON
    testGPTOSSIncomplete()
    
    // Test 2: Realistic token streaming
    testRealisticTokenStreaming()
    
    // Test 3: JSON cut off mid-key
    testJSONMidKey()
    
    // Test 4: JSON cut off mid-value
    testJSONMidValue()
}

func testGPTOSSIncomplete() {
    print("\n[Test 1: GPT OSS Format - Incomplete JSON]")
    print("-" * 50)
    
    // Simulate GPT OSS output that gets cut off
    let tokens = [
        "<|channel|>",
        "analysis",
        "<|message|>",
        "Let me analyze this. ",
        "We need to create a proper JSON response.\n\n",
        "<|end|>",
        "<|start|>",
        "assistant",
        "<|channel|>",
        "final",
        "<|message|>",
        "```json\n",
        "{\n",
        "  \"",
        "name",
        "\": \"",
        "Alice",
        "\",\n",
        "  \"",
        "age",
        "\": ",
        "30",
        ",\n",
        "  \"",
        "city"  // Cuts off here - no closing quote, no value
    ]
    
    processTokenStream(
        tokens: tokens,
        testName: "GPT OSS Incomplete",
        expectedKeys: ["name", "age"],
        expectComplete: false
    )
}

func testRealisticTokenStreaming() {
    print("\n[Test 2: Realistic Token Streaming]")
    print("-" * 50)
    
    // Tokens as they might come from an LLM
    // Keys and values can be split across tokens
    let tokens = [
        "{\"",
        "user",          // Key in one token
        "Name",          // Key continues
        "\": \"",
        "John",
        " Doe",          // Value split
        "\", \"",
        "em",            // Key "email" split
        "ail",
        "\": \"",
        "john",
        "@",
        "example",
        ".com",
        "\", \"",
        "is",            // Key "isActive" starts
        "Active",        // Key continues
        "\": tr",        // Value "true" starts
        "ue, \"",
        "score",         // New key starts
        "\": 9",         // Value starts
        "5"              // Incomplete - no closing brace
    ]
    
    processTokenStream(
        tokens: tokens,
        testName: "Realistic Streaming",
        expectedKeys: ["userName", "email", "isActive", "score"],
        expectComplete: false
    )
}

func testJSONMidKey() {
    print("\n[Test 3: JSON Cut Off Mid-Key]")
    print("-" * 50)
    
    let tokens = [
        "Starting JSON: ",
        "{\"",
        "first",
        "Key",
        "\": \"value1\", \"",
        "second",
        "K"  // Cuts off in the middle of "secondKey"
    ]
    
    processTokenStream(
        tokens: tokens,
        testName: "Mid-Key Cutoff",
        expectedKeys: ["firstKey"],
        expectComplete: false
    )
}

func testJSONMidValue() {
    print("\n[Test 4: JSON Cut Off Mid-Value]")
    print("-" * 50)
    
    let tokens = [
        "{\"status\": \"",
        "in_",
        "prog",  // Value "in_progress" incomplete
        // Stream ends here
    ]
    
    processTokenStream(
        tokens: tokens,
        testName: "Mid-Value Cutoff",
        expectedKeys: ["status"],
        expectComplete: false
    )
}

// Helper function to process token stream
func processTokenStream(
    tokens: [String],
    testName: String,
    expectedKeys: [String],
    expectComplete: Bool
) {
    var extractor = JSONExtractor()
    var stateMachine = JSONStateMachine()
    var jsonBuffer = ""
    var detectedKeys: [String] = []
    var jsonStarted = false
    var jsonStartToken = -1
    
    print("\nProcessing \(tokens.count) tokens for: \(testName)")
    
    // Process each token
    for (index, token) in tokens.enumerated() {
        var tokenProcessedJSON = false
        
        // Process each character in the token
        for char in token {
            let shouldProcess = extractor.processCharacter(char)
            
            if shouldProcess {
                // Mark JSON as started
                if !jsonStarted {
                    jsonStarted = true
                    jsonStartToken = index
                    print("  ✅ JSON detected at token \(index + 1): \"\(token)\"")
                }
                
                tokenProcessedJSON = true
                jsonBuffer.append(char)
                
                // Process with state machine
                let previousPhase = stateMachine.phase
                stateMachine.processCharacter(char)
                
                // Check for completed keys
                if case .inString(.body(kind: .key, escaped: false)) = previousPhase {
                    if case .inObject(.expectColon) = stateMachine.phase {
                        let key = stateMachine.currentKey
                        if !key.isEmpty && !detectedKeys.contains(key) {
                            detectedKeys.append(key)
                            print("  🔑 Key detected at token \(index + 1): \"\(key)\"")
                        }
                    }
                }
            }
        }
        
        // Show progress for tokens that contribute to JSON
        if tokenProcessedJSON && index < 10 {  // Show first 10 for brevity
            print("    Token \(index + 1): \"\(token)\" → Buffer: \(String(jsonBuffer.suffix(30)))")
        }
    }
    
    // Results
    print("\n📊 Results for \(testName):")
    print("  Total tokens: \(tokens.count)")
    if jsonStartToken >= 0 {
        print("  JSON started at token: \(jsonStartToken + 1)")
        print("  JSON buffer length: \(jsonBuffer.count) chars")
    } else {
        print("  ⚠️ No JSON detected")
    }
    
    print("  Detected keys: \(detectedKeys)")
    print("  Expected keys: \(expectedKeys)")
    
    // Validate key detection
    let missingKeys = expectedKeys.filter { !detectedKeys.contains($0) }
    if missingKeys.isEmpty {
        print("  ✅ All expected keys detected")
    } else {
        print("  ⚠️ Missing keys: \(missingKeys)")
    }
    
    // Check completion status
    let phase = stateMachine.phase
    print("  Final state: \(describePhase(phase))")
    
    switch phase {
    case .done:
        if expectComplete {
            print("  ✅ JSON completed as expected")
        } else {
            print("  ⚠️ JSON completed but expected incomplete")
        }
    case .error:
        print("  ❌ JSON parsing error")
    default:
        if !expectComplete {
            print("  ✅ JSON incomplete as expected (in state: \(describePhase(phase)))")
        } else {
            print("  ⚠️ JSON incomplete but expected complete")
        }
    }
    
    // Show the JSON buffer (truncated)
    if jsonBuffer.count > 0 {
        let display = jsonBuffer.count > 100 ? String(jsonBuffer.prefix(100)) + "..." : jsonBuffer
        print("  JSON Buffer: \(display)")
    }
    
    print("-" * 50)
}

// Phase description helper
func describePhase(_ phase: JSONStateMachine.Phase) -> String {
    switch phase {
    case .root: return "root"
    case .inObject(let obj):
        switch obj {
        case .expectKeyOrEnd: return "object.expectKeyOrEnd"
        case .expectKeyFirstQuote: return "object.expectKeyQuote"
        case .inKey: return "object.inKey"
        case .expectKeyEndQuote: return "object.expectEndQuote"
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

// String extension helper
extension String {
    static func * (left: String, right: Int) -> String {
        return String(repeating: left, count: right)
    }
}