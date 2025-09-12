import Foundation
import OpenFoundationModelsMLX

// Test JSONExtractor with various text formats
func testJSONExtractor() {
    print("\n📝 Testing JSONExtractor with embedded JSON")
    print("-" * 50)
    
    // Test cases with different formats
    let testCases: [(name: String, input: String, expectedJSON: String)] = [
        (
            name: "Plain JSON",
            input: #"{"name": "John", "age": 30}"#,
            expectedJSON: #"{"name": "John", "age": 30}"#
        ),
        (
            name: "GPT OSS Format",
            input: #"<|channel|>analysis<|message|>We need to analyze...{"name": "Alice", "city": "NYC"}"#,
            expectedJSON: #"{"name": "Alice", "city": "NYC"}"#
        ),
        (
            name: "Markdown Code Block",
            input: """
            Here's the JSON:
            ```json
            {"user": {"id": 123, "email": "test@example.com"}}
            ```
            """,
            expectedJSON: #"{"user": {"id": 123, "email": "test@example.com"}}"#
        ),
        (
            name: "Multiple JSONs",
            input: #"First: {"a": 1} and second: {"b": 2}"#,
            expectedJSON: #"{"a": 1}"#  // Should find first one
        ),
        (
            name: "Array JSON",
            input: #"Some text before [{"id": 1}, {"id": 2}] and after"#,
            expectedJSON: #"[{"id": 1}, {"id": 2}]"#
        ),
        (
            name: "Complex GPT OSS",
            input: """
            <|channel|>analysis<|message|>Let me analyze this request. We need to create a JSON response.
            The structure should be clear and well-formatted.
            <|end|><|start|>assistant<|channel|>final<|message|>```json
            {
              "company": "TechCorp",
              "departments": [
                {"name": "Engineering", "size": 50},
                {"name": "Sales", "size": 20}
              ]
            }
            ```
            """,
            expectedJSON: #"{"company": "TechCorp", "departments": [{"name": "Engineering", "size": 50}, {"name": "Sales", "size": 20}]}"#
        )
    ]
    
    for testCase in testCases {
        print("\n[Test: \(testCase.name)]")
        print("Input: \(String(testCase.input.prefix(100)))\(testCase.input.count > 100 ? "..." : "")")
        
        testWithJSONExtractor(input: testCase.input, expectedJSON: testCase.expectedJSON)
    }
}

func testWithJSONExtractor(input: String, expectedJSON: String) {
    var extractor = JSONExtractor()
    var stateMachine = JSONStateMachine()
    var jsonBuffer = ""
    var detectedKeys: [String] = []
    
    print("\n--- Phase 1: JSONExtractor ---")
    var jsonStarted = false
    var skippedCount = 0
    
    for char in input {
        let shouldProcess = extractor.processCharacter(char)
        
        if shouldProcess {
            if !jsonStarted {
                jsonStarted = true
                print("  ✅ JSON detected after skipping \(skippedCount) characters")
                print("  📍 Starting with: '\(char)'")
            }
            jsonBuffer.append(char)
            
            // Process with JSONStateMachine
            let previousPhase = stateMachine.phase
            stateMachine.processCharacter(char)
            
            // Check for key detection
            if case .inString(.body(kind: .key, escaped: false)) = previousPhase {
                if case .inObject(.expectColon) = stateMachine.phase {
                    let key = stateMachine.currentKey
                    if !key.isEmpty && !detectedKeys.contains(key) {
                        detectedKeys.append(key)
                    }
                }
            }
        } else {
            if !jsonStarted {
                skippedCount += 1
            }
        }
    }
    
    print("\n--- Phase 2: JSONStateMachine Results ---")
    print("  Extracted JSON: \(jsonBuffer)")
    print("  Final state: \(describePhaseDetailed(stateMachine.phase))")
    print("  Detected keys: \(detectedKeys)")
    
    // Verify extraction
    let success = jsonBuffer.trimmingCharacters(in: .whitespacesAndNewlines)
        .replacingOccurrences(of: " ", with: "")
        .replacingOccurrences(of: "\n", with: "")
        .contains(expectedJSON.replacingOccurrences(of: " ", with: ""))
    
    print("\n  Result: \(success ? "✅ SUCCESS" : "❌ FAILED")")
}

// Test integration with enhanced scenarios
func testJSONExtractorIntegration() {
    print("\n\n📝 Testing JSONExtractor + JSONStateMachine Integration")
    print("=" * 60)
    
    // Simulate GPT OSS token-by-token generation
    print("\n[Scenario: GPT OSS Token-by-Token Generation]")
    
    let tokens = [
        "<|channel|>",
        "analysis",
        "<|message|>",
        "Let me ",
        "think about ",
        "this...",
        "\n\n",
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
    
    var extractor = JSONExtractor()
    var stateMachine = JSONStateMachine()
    var jsonBuffer = ""
    var detectedKeys: [String] = []
    
    print("Processing tokens:")
    for (index, token) in tokens.enumerated() {
        print("\n  Token \(index + 1): \"\(token)\"")
        
        for char in token {
            let shouldProcess = extractor.processCharacter(char)
            
            if shouldProcess {
                jsonBuffer.append(char)
                
                let previousPhase = stateMachine.phase
                stateMachine.processCharacter(char)
                
                // Detect keys
                if case .inString(.body(kind: .key, escaped: false)) = previousPhase {
                    if case .inObject(.expectColon) = stateMachine.phase {
                        let key = stateMachine.currentKey
                        if !key.isEmpty && !detectedKeys.contains(key) {
                            detectedKeys.append(key)
                            print("    🔑 Key detected: \"\(key)\"")
                        }
                    }
                }
                
                // Show state changes
                if previousPhase != stateMachine.phase {
                    print("    State: \(describePhaseDetailed(previousPhase)) → \(describePhaseDetailed(stateMachine.phase))")
                }
            }
        }
        
        if extractor.isInJSON {
            print("    Buffer: \(jsonBuffer)")
        }
    }
    
    print("\n📊 Final Results:")
    print("  JSON Buffer: \(jsonBuffer)")
    print("  All Keys: \(detectedKeys)")
    print("  Final State: \(describePhaseDetailed(stateMachine.phase))")
}

private func describePhaseDetailed(_ phase: JSONStateMachine.Phase) -> String {
    switch phase {
    case .root: return "root"
    case .inObject(let obj):
        switch obj {
        case .expectKeyOrEnd: return "obj.expectKeyOrEnd"
        case .expectKeyFirstQuote: return "obj.expectKeyQuote"
        case .inKey: return "obj.inKey"
        case .expectKeyEndQuote: return "obj.expectEndQuote"
        case .expectColon: return "obj.expectColon"
        case .expectValue: return "obj.expectValue"
        case .expectCommaOrEnd: return "obj.expectCommaOrEnd"
        }
    case .inArray(let arr):
        switch arr {
        case .expectValue: return "arr.expectValue"
        case .expectCommaOrEnd: return "arr.expectCommaOrEnd"
        }
    case .inString(let str):
        switch str {
        case .body(let kind, let escaped):
            return "str.\(kind == .key ? "key" : "value")\(escaped ? ".esc" : "")"
        }
    case .inNumber(let num):
        switch num {
        case .integer: return "num.int"
        case .decimal: return "num.dec"
        case .exponent: return "num.exp"
        }
    case .inLiteral(let lit):
        if case .inProgress(let text) = lit {
            return "lit(\(text))"
        }
        return "literal"
    case .done: return "done"
    case .error: return "error"
    }
}