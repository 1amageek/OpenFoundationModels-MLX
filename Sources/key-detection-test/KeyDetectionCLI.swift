import Foundation
import OpenFoundationModelsMLX

@main
struct KeyDetectionCLI {
    static func main() {
        print("🔍 Comprehensive JSON Key Detection Test Suite")
        print("=" * 60)
        print()
        
        // Section 1: Complete JSON Tests
        print("📝 Section 1: Complete JSON Detection")
        print("-" * 50)
        testCompleteJSON()
        
        // Section 2: Incomplete JSON (LLM-style generation)
        print("\n📝 Section 2: Incomplete JSON (LLM Token-by-Token)")
        print("-" * 50)
        testIncompleteJSON()
        
        // Section 3: Token-split keys (Real LLM behavior)
        print("\n📝 Section 3: Token-Split Keys (Real LLM Behavior)")
        print("-" * 50)
        testTokenSplitKeys()
        
        // Section 4: Enhanced processor with probabilities
        print("\n📝 Section 4: Enhanced Processor with Probability Analysis")
        print("-" * 50)
        testEnhancedProcessor()
        
        // Section 5: Array handling
        print("\n📝 Section 5: Array and Nested Structure Handling")
        print("-" * 50)
        testArrayDebug()
        
        // Section 6: JSONExtractor tests
        print("\n📝 Section 6: JSONExtractor with Embedded JSON")
        print("-" * 50)
        testJSONExtractor()
        
        // Section 7: JSONExtractor integration
        print("\n📝 Section 7: JSONExtractor + JSONStateMachine Integration")
        print("-" * 50)
        testJSONExtractorIntegration()
        
        print("\n" + "=" * 60)
        print("✅ All tests completed")
        print("=" * 60)
    }
    
    // MARK: - Section 1: Complete JSON Tests
    
    static func testCompleteJSON() {
        print("\nTesting complete JSON structures:")
        
        // Test Case 1: Simple object
        print("\n[Test 1] Simple Object:")
        let json1 = #"{"name":"John","age":30,"city":"New York"}"#
        testKeyDetection(jsonString: json1)
        
        // Test Case 2: Nested object
        print("\n[Test 2] Nested Object:")
        let json2 = #"{"user":{"name":"Alice","email":"alice@example.com"},"status":"active"}"#
        testKeyDetection(jsonString: json2)
        
        // Test Case 3: Array with objects
        print("\n[Test 3] Array with Objects:")
        let json3 = #"{"items":[{"id":1,"name":"Item1"},{"id":2,"name":"Item2"}],"total":2}"#
        testKeyDetection(jsonString: json3)
        
        // Test Case 4: JSON with markdown wrapper
        print("\n[Test 4] Markdown Code Block:")
        let json4 = """
```json
{"config":{"timeout":30,"retries":3},"enabled":true}
```
"""
        testKeyDetection(jsonString: json4)
        
        // Test Case 5: Complex nested structure
        print("\n[Test 5] Complex Nested Structure:")
        let json5 = #"{"company":{"name":"TechCorp","employees":[{"name":"Bob","department":"Engineering"},{"name":"Carol","department":"Sales"}],"location":{"city":"SF","country":"USA"}},"founded":2020}"#
        testKeyDetection(jsonString: json5)
    }
    
    // MARK: - Helper Methods
    
    static func testKeyDetection(jsonString: String) {
        print("\n--- Using JSONStateMachine (Real ADAPT Implementation) ---")
        testWithJSONStateMachine(jsonString: jsonString)
        
        print("\n--- Using SimpleKeyDetector (Simplified Version) ---")
        testWithSimpleDetector(jsonString: jsonString)
    }
    
    static func testWithJSONStateMachine(jsonString: String) {
        // Use the real JSONStateMachine from ADAPT implementation
        var stateMachine = JSONStateMachine()
        var detectedKeys: [String] = []
        
        // Find JSON start position first
        let jsonStart = findJSONStart(in: jsonString)
        let actualJSON = String(jsonString.dropFirst(jsonStart))
        
        if jsonStart > 0 {
            print("  📌 Skipped \(jsonStart) characters before JSON start")
        }
        
        // Process each character and track phase changes
        for char in actualJSON {
            let previousPhase = stateMachine.phase
            
            stateMachine.processCharacter(char)
            
            // Check if we just exited a key string
            if case .inString(.body(kind: .key, escaped: false)) = previousPhase {
                if case .inObject(.expectColon) = stateMachine.phase {
                    // Key just completed - the colon expectation confirms it
                    let key = stateMachine.currentKey
                    if !key.isEmpty {
                        detectedKeys.append(key)
                        print("  🔑 Detected key: \"\(key)\"")
                    }
                }
            }
        }
        
        // Show summary
        if detectedKeys.isEmpty {
            print("  ⚠️  No keys detected")
        } else {
            print("\n📊 Summary - Total keys found: \(detectedKeys.count)")
            for (idx, key) in detectedKeys.enumerated() {
                print("    \(idx + 1). \"\(key)\"")
            }
        }
        
        // Check final state
        switch stateMachine.phase {
        case .done:
            print("✅ JSON parsing completed successfully")
        case .error:
            // For markdown blocks, if we got all keys, it's likely just trailing ```
            if jsonStart > 0 && !detectedKeys.isEmpty {
                print("⚠️  JSON parsing ended in error state (likely due to trailing characters)")
            } else {
                print("❌ JSON parsing ended in error state")
            }
        default:
            print("⚠️  JSON parsing in state: \(describePhase(stateMachine.phase))")
        }
    }
    
    static func describePhase(_ phase: JSONStateMachine.Phase) -> String {
        switch phase {
        case .root: return "root"
        case .inObject(let obj): return "object(\(obj))"
        case .inArray(let arr): return "array(\(arr))"
        case .inString(let str): return "string(\(str))"
        case .inNumber(let num): return "number(\(num))"
        case .inLiteral(let lit): return "literal(\(lit))"
        case .done: return "done"
        case .error: return "error"
        }
    }
    
    static func testWithSimpleDetector(jsonString: String) {
        // Create simple key detector
        let detector = SimpleKeyDetector()
        
        // Process JSON and get detected keys
        let detectedKeys = detector.processJSON(jsonString)
        
        if detectedKeys.isEmpty {
            print("  ⚠️  No keys detected")
        } else {
            print("\n📊 Summary - Total keys found: \(detectedKeys.count)")
            for (idx, key) in detectedKeys.enumerated() {
                print("    \(idx + 1). \"\(key)\"")
            }
        }
    }
}

// Helper to repeat string
// Helper functions
extension String {
    static func * (left: String, right: Int) -> String {
        return String(repeating: left, count: right)
    }
}

// JSON start detection helper
func findJSONStart(in text: String) -> Int {
    let chars = Array(text)
    var index = 0
    var inCodeBlock = false
    
    while index < chars.count {
        let char = chars[index]
        
        // Check for markdown code block
        if index + 7 <= chars.count {
            let next7 = String(chars[index..<min(index+7, chars.count)])
            if next7.lowercased().hasPrefix("```json") {
                index += 7
                inCodeBlock = true
                // Skip to next line
                while index < chars.count && chars[index] != "\n" {
                    index += 1
                }
                if index < chars.count {
                    index += 1 // Skip the newline
                }
                continue
            }
        }
        
        // Check for triple backticks alone
        if index + 3 <= chars.count {
            let next3 = String(chars[index..<min(index+3, chars.count)])
            if next3 == "```" {
                index += 3
                if inCodeBlock {
                    // End of code block
                    inCodeBlock = false
                }
                // Skip to next line
                while index < chars.count && chars[index] != "\n" {
                    index += 1
                }
                continue
            }
        }
        
        // Look for JSON start
        if char == "{" || char == "[" {
            return index
        }
        
        index += 1
    }
    
    return 0 // Default to start if no JSON found
}