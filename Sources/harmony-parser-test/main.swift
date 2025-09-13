import Foundation
import OpenFoundationModelsMLX

// Integration test for HarmonyParser
print("🧪 HarmonyParser Integration Test")
print(String(repeating: "=", count: 60))

// Test Case 1: Actual GPT-OSS output pattern
let actualOutput = """
<|channel|>analysis<|message|>We need to output JSON that matches the schema. Must provide company name, founded to be 2020, type "startup". 

Let's craft JSON:
{
  "name": "NovaTech Solutions"
}

All good.<|end|><|start|>assistant<|channel|>final<|message|>{"name":"NovaTech Solutions","founded":2020,"type":"startup","employeeCount":35,"headquarters":{"street":"123 Innovation Drive","city":"San Francisco","country":"USA","postalCode":"94107"},"departments":[{"name":"Engineering","type":"engineering","headCount":20,"manager":{"firstName":"Elena","lastName":"Martinez","email":"elena.martinez@novatech.com","level":"lead","yearsExperience":12},"projects":[{"name":"Project Aurora","status":"active","startDate":"2023-01-15","teamSize":8,"budget":1200000},{"name":"Project Nebula","status":"planning","startDate":"2024-06-01","teamSize":4,"budget":null}]},{"name":"Sales","type":"sales","headCount":15,"manager":{"firstName":"Michael","lastName":"Johnson","email":"michael.johnson@novatech.com","level":"manager","yearsExperience":9},"projects":[{"name":"Market Expansion Q3","status":"active","startDate":"2023-04-01","teamSize":5,"budget":500000}]}]}
"""

print("\n📝 Test Case 1: Actual GPT-OSS Output Pattern")
print("Input length: \(actualOutput.count) characters")

let parsed1 = HarmonyParser.parse(actualOutput)

// Check analysis channel
if let analysis = parsed1.analysis {
    if !analysis.isEmpty {
        print("✅ Analysis channel extracted: \(analysis.prefix(50))...")
    } else {
        print("❌ Analysis channel is empty!")
    }
} else {
    print("❌ Analysis channel is nil!")
}

// Check final channel
if let final = parsed1.final {
    if !final.isEmpty {
        print("✅ Final channel extracted: \(final.prefix(50))...")
        
        // Verify JSON is valid
        if let data = final.data(using: .utf8),
           let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any] {
            print("✅ JSON is valid!")
            print("   - name: \(json["name"] as? String ?? "nil")")
            print("   - founded: \(json["founded"] as? Int ?? 0)")
            print("   - type: \(json["type"] as? String ?? "nil")")
            print("   - employeeCount: \(json["employeeCount"] as? Int ?? 0)")
        } else {
            print("❌ JSON is invalid!")
        }
    } else {
        print("❌ Final channel is empty!")
    }
} else {
    print("❌ Final channel is nil!")
}

// Test Case 2: Simple mixed channels
print("\n📝 Test Case 2: Simple Mixed Channels")

let simpleInput = """
<|channel|>analysis<|message|>Analyzing the request<|end|>
<|start|>assistant<|channel|>final<|message|>{"success": true, "value": 42}<|end|>
"""

let parsed2 = HarmonyParser.parse(simpleInput)

print("Analysis: \(parsed2.analysis ?? "nil")")
print("Final: \(parsed2.final ?? "nil")")

if let final = parsed2.final,
   let data = final.data(using: .utf8),
   let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any] {
    print("✅ Simple JSON is valid: success=\(json["success"] as? Bool ?? false), value=\(json["value"] as? Int ?? 0)")
} else {
    print("❌ Simple JSON parsing failed")
}

// Test Case 3: displayContent
print("\n📝 Test Case 3: Display Content")
print("DisplayContent returns: \(parsed1.displayContent.prefix(50))...")
print("Should equal final: \(parsed1.displayContent == parsed1.final ? "✅ Yes" : "❌ No")")

// Summary
print("\n" + String(repeating: "=", count: 60))
if parsed1.final != nil && !parsed1.final!.isEmpty && 
   parsed1.analysis != nil && !parsed1.analysis!.isEmpty {
    print("✅ HarmonyParser is working correctly!")
    print("Both analysis and final channels are properly extracted.")
} else {
    print("❌ HarmonyParser has issues!")
    print("One or more channels were not properly extracted.")
    exit(1)
}