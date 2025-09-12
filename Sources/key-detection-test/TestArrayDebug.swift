import Foundation
import OpenFoundationModelsMLX

// Simple test to debug array handling
func testArrayDebug() {
    let jsonString = #"{"items":[{"id":1,"name":"Item1"},{"id":2,"name":"Item2"}],"total":2}"#
    
    print("\n=== Debugging Array Handling ===")
    print("JSON: \(jsonString)")
    
    var stateMachine = JSONStateMachine()
    var detectedKeys: [String] = []
    var charCount = 0
    
    for char in jsonString {
        let previousPhase = stateMachine.phase
        stateMachine.processCharacter(char)
        
        // Print phase transitions for debugging
        if charCount >= 15 && charCount <= 60 {  // Extended range to see more
            print("  [\(charCount)] '\(char)': \(describePhase(previousPhase)) → \(describePhase(stateMachine.phase))")
        }
        
        // Detect completed keys
        if case .inString(.body(kind: .key, escaped: false)) = previousPhase {
            if case .inObject(.expectColon) = stateMachine.phase {
                let key = stateMachine.currentKey
                if !key.isEmpty {
                    detectedKeys.append(key)
                    print("  Key #\(detectedKeys.count): \"\(key)\" at position \(charCount)")
                }
            }
        }
        
        // Check for errors
        if case .error = stateMachine.phase {
            print("  ❌ ERROR at position \(charCount), char: '\(char)'")
            print("     Previous phase was: \(describePhase(previousPhase))")
            break
        }
        
        charCount += 1
    }
    
    func describePhase(_ phase: JSONStateMachine.Phase) -> String {
        switch phase {
        case .root: return "root"
        case .inObject(let obj): 
            switch obj {
            case .expectKeyOrEnd: return "obj.expectKeyOrEnd"
            case .expectKeyFirstQuote: return "obj.expectKeyFirstQuote"
            case .inKey: return "obj.inKey"
            case .expectKeyEndQuote: return "obj.expectKeyEndQuote"
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
            case .body(let kind, _):
                return "string.\(kind == .key ? "key" : "value")"
            }
        case .inNumber(_): return "number"
        case .inLiteral(_): return "literal"
        case .done: return "done"
        case .error: return "error"
        }
    }
    
    print("\nDetected keys: \(detectedKeys)")
    print("Expected: [\"items\", \"id\", \"name\", \"id\", \"name\", \"total\"]")
    
    if detectedKeys.count == 6 {
        print("✅ All keys detected!")
    } else {
        print("❌ Missing keys: expected 6, got \(detectedKeys.count)")
    }
}