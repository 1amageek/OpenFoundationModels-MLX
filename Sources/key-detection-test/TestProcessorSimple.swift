import Foundation
import OpenFoundationModelsMLX

// Test KeyDetectionLogitProcessor without MLX dependency
func testProcessorSimple() {
    print("\n=== Testing Key Detection (Simplified) ===\n")
    
    // Test JSON samples with expected keys
    let testCases: [(json: String, expectedKeys: [String])] = [
        (
            json: #"{"name":"John","age":30}"#,
            expectedKeys: ["name", "age"]
        ),
        (
            json: #"{"user":{"firstName":"Alice","lastName":"Smith"},"active":true}"#,
            expectedKeys: ["user", "firstName", "lastName", "active"]
        ),
        (
            json: #"{"items":[{"id":1,"name":"Item1"},{"id":2,"name":"Item2"}],"total":2}"#,
            expectedKeys: ["items", "id", "name", "id", "name", "total"]
        )
    ]
    
    for (index, testCase) in testCases.enumerated() {
        print("📝 Test Case \(index + 1):")
        print("JSON: \(testCase.json)")
        
        // Use JSONStateMachine directly for testing
        var stateMachine = JSONStateMachine()
        var detectedKeys: [String] = []
        
        // Process each character
        for char in testCase.json {
            let previousPhase = stateMachine.phase
            stateMachine.processCharacter(char)
            
            // Check if we just completed a key
            if case .inString(.body(kind: .key, escaped: false)) = previousPhase {
                if case .inObject(.expectColon) = stateMachine.phase {
                    let key = stateMachine.currentKey
                    if !key.isEmpty {
                        detectedKeys.append(key)
                    }
                }
            }
        }
        
        // Verify results
        print("Expected: \(testCase.expectedKeys)")
        print("Detected: \(detectedKeys)")
        
        if detectedKeys == testCase.expectedKeys {
            print("✅ PASSED\n")
        } else {
            print("❌ FAILED\n")
        }
    }
    
    print("\n=== Token-based Detection Simulation ===\n")
    
    // Simulate token-based processing
    let tokenPatterns = [
        ["{\"", "name", "\":\"", "John", "\",\"", "age", "\":", "30", "}"],
        ["{\"user", "\":{\"", "first", "Name", "\":\"", "Alice", "\"}}"]
    ]
    
    for (index, tokens) in tokenPatterns.enumerated() {
        print("📝 Token Pattern \(index + 1):")
        print("Tokens: \(tokens)")
        
        var stateMachine = JSONStateMachine()
        var detectedKeys: [String] = []
        
        // Process each token
        for token in tokens {
            // Process each character in the token
            for char in token {
                let previousPhase = stateMachine.phase
                stateMachine.processCharacter(char)
                
                // Check if we just completed a key
                if case .inString(.body(kind: .key, escaped: false)) = previousPhase {
                    if case .inObject(.expectColon) = stateMachine.phase {
                        let key = stateMachine.currentKey
                        if !key.isEmpty {
                            detectedKeys.append(key)
                            print("  🔑 Detected key: \"\(key)\"")
                        }
                    }
                }
            }
        }
        
        print("All keys: \(detectedKeys)")
        print("")
    }
}