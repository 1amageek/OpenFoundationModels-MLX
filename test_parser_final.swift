#!/usr/bin/env swift

import Foundation

// Test output with extra content after JSON
let testOutput = """
<|channel|>analysis<|message|>Analyzing the request...<|end|><|start|>assistant<|channel|>final<|message|>{
  "name": "TestCompany",
  "founded": 2020,
  "type": "startup",
  "employeeCount": 25
}
========== END OUTPUT ==========
Extra text that should not be included
"""

// Test output with just JSON
let cleanOutput = """
<|channel|>final<|message|>{"name": "CleanCompany", "founded": 2021}
"""

// Test output with nested JSON
let complexOutput = """
<|channel|>final<|message|>{
  "name": "ComplexCorp",
  "departments": [
    {"name": "Engineering", "count": 10},
    {"name": "Sales", "count": 5}
  ]
}Some extra text after
"""

func testExtraction(_ name: String, _ output: String) {
    print("\nTesting: \(name)")
    print(String(repeating: "-", count: 40))
    
    // Find final channel
    if let channelStart = output.range(of: "<|channel|>final<|message|>") {
        let content = String(output[channelStart.upperBound...])
        print("Raw content after marker: [\(content)]")
        
        // Extract JSON using brace counting
        if let jsonStart = content.firstIndex(of: "{") {
            var braceCount = 0
            var inString = false
            var escapeNext = false
            var jsonEnd: String.Index?
            
            for (index, char) in content[jsonStart...].enumerated() {
                let actualIndex = content.index(jsonStart, offsetBy: index)
                
                if escapeNext {
                    escapeNext = false
                    continue
                }
                
                if char == "\\" {
                    escapeNext = true
                    continue
                }
                
                if char == "\"" && !escapeNext {
                    inString = !inString
                    continue
                }
                
                if !inString {
                    if char == "{" {
                        braceCount += 1
                    } else if char == "}" {
                        braceCount -= 1
                        if braceCount == 0 {
                            jsonEnd = content.index(actualIndex, offsetBy: 1)
                            break
                        }
                    }
                }
            }
            
            if let endIdx = jsonEnd {
                let json = String(content[jsonStart..<endIdx])
                print("Extracted JSON: [\(json)]")
                
                // Validate JSON
                if let data = json.data(using: .utf8) {
                    do {
                        let _ = try JSONSerialization.jsonObject(with: data)
                        print("✅ Valid JSON extracted!")
                    } catch {
                        print("❌ Invalid JSON: \(error)")
                    }
                }
            } else {
                print("❌ Could not find JSON end")
            }
        }
    }
}

print("Testing HarmonyParser JSON extraction fix")
print(String(repeating: "=", count: 60))

testExtraction("Output with extra text", testOutput)
testExtraction("Clean output", cleanOutput)
testExtraction("Complex nested output", complexOutput)