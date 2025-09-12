import Foundation
import OpenFoundationModelsMLX

@main
struct KeyDetectionCLI {
    static func main() {
        // Run array debug test first if needed
        if CommandLine.arguments.contains("--debug-array") {
            testArrayDebug()
            return
        }
        
        // Test with processor
        if CommandLine.arguments.contains("--test-processor") {
            testProcessorWithSimulatedTokens()
            return
        }
        
        // Simple test without MLX
        if CommandLine.arguments.contains("--test-simple") {
            testProcessorSimple()
            return
        }
        
        // Test enhanced processor
        if CommandLine.arguments.contains("--test-enhanced") {
            testEnhancedProcessor()
            return
        }
        
        print("🔍 JSON Key Detection Test")
        print("=" * 50)
        
        // Test JSON samples
        let testCases = [
            // Simple flat object
            #"{"name":"John","age":30,"city":"NYC"}"#,
            
            // Nested object
            #"{"user":{"firstName":"Alice","lastName":"Smith","email":"alice@example.com"},"active":true}"#,
            
            // Array with objects
            #"{"items":[{"id":1,"name":"Item1"},{"id":2,"name":"Item2"}],"total":2}"#,
            
            // Complex nested structure
            #"{"company":"TechCorp","contact":{"phone":"555-1234","address":{"street":"123 Main","city":"SF"}}}"#,
            
            // With markdown code block
            """
            ```json
            {"format":"markdown","content":"test","valid":true}
            ```
            """,
            
            // With text prefix
            """
            Here is the generated JSON response:
            {"status":"success","code":200,"message":"OK"}
            """,
            
            // With channel tags (like GPT models)
            """
            <|channel|>analysis<|message|>Let me create the JSON<|end|>
            <|channel|>final<|message|>{"result":"computed","value":42}
            """
        ]
        
        for (index, jsonString) in testCases.enumerated() {
            print("\n📝 Test Case \(index + 1):")
            print("Input: \(jsonString)")
            print("\nDetected Keys:")
            
            testKeyDetection(jsonString: jsonString)
            
            print("-" * 50)
        }
    }
    
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