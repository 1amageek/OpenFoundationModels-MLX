import Foundation
@testable import OpenFoundationModelsMLX

// Simple test of HarmonyParser with actual GPT-OSS output
let actualOutput = """
<|channel|>analysis<|message|>We need to produce JSON data instance conforming to schema: a company profile. Must include name, founded 2020, type startup, employeeCount, headquarters with street city country postalCode, departments array with at least two departments: engineering and sales. Each dept has 1-2 projects. Provide manager details with level enum junior senior lead manager. Provide headCount integer. Provide projects with required fields.

Ensure dates ISO format, e.g., startDate "2023-03-15". Provide budgets numeric or null.

Return only JSON object, no explanation. Need to include all required fields.

Let's craft.

Name: "NovaTech Labs"

Founded: 2020

Type: "startup"

EmployeeCount: maybe 45

Headquarters: street "12 Innovation Drive", city "San Francisco", country "USA", postalCode "94107"

Departments: engineering and sales.

Engineering department:

headCount: 25

manager: firstName "Alice", lastName "Chen", email "alice.chen@novatech.com", level "lead", yearsExperience 8

projects: maybe two projects.

Project1: name "Quantum AI Platform", status "active", startDate "2023-01-10", teamSize 12, budget 500000.0

Project2: name "Edge Computing SDK", status "planning", startDate "2024-02-01", teamSize 6, budget null

Sales department:

headCount: 20

manager: firstName "Mark", lastName "Rogers", email "mark.rogers@novatech.com", level "manager", yearsExperience 10

projects: one project

Project1: name "Global Market Expansion", status "active", startDate "2023-06-15", teamSize 8, budget 300000.0

We must ensure departments array length 2. Each department includes required fields: name, type, headCount, manager. Optionally projects array.

All values valid.

Return JSON.

Let's produce.

<|end|><|start|>assistant<|channel|>final<|message|>{"name":"NovaTech Labs","founded":2020,"type":"startup","employeeCount":45,"headquarters":{"street":"12 Innovation Drive","city":"San Francisco","country":"USA","postalCode":"94107"},"departments":[{"name":"Engineering","type":"engineering","headCount":25,"manager":{"firstName":"Alice","lastName":"Chen","email":"alice.chen@novatech.com","level":"lead","yearsExperience":8},"projects":[{"name":"Quantum AI Platform","status":"active","startDate":"2023-01-10","teamSize":12,"budget":500000.0},{"name":"Edge Computing SDK","status":"planning","startDate":"2024-02-01","teamSize":6,"budget":null}]},{"name":"Sales","type":"sales","headCount":20,"manager":{"firstName":"Mark","lastName":"Rogers","email":"mark.rogers@novatech.com","level":"manager","yearsExperience":10},"projects":[{"name":"Global Market Expansion","status":"active","startDate":"2023-06-15","teamSize":8,"budget":300000.0}]}]}
"""

print("Testing HarmonyParser with actual GPT-OSS output...")
print("=" * 60)

let parsed = HarmonyParser.parse(actualOutput)

print("Has analysis channel: \(parsed.analysis != nil)")
print("Has final channel: \(parsed.final != nil)")

if let analysis = parsed.analysis {
    print("\nAnalysis channel (first 100 chars):")
    print(String(analysis.prefix(100)))
}

if let final = parsed.final {
    print("\nFinal channel content:")
    print(final)
    
    print("\nTrying to decode as JSON...")
    if let data = final.data(using: .utf8) {
        do {
            let json = try JSONSerialization.jsonObject(with: data) as? [String: Any]
            print("✅ JSON parsed successfully!")
            print("Company name: \(json?["name"] as? String ?? "nil")")
            print("Founded: \(json?["founded"] as? Int ?? 0)")
            print("Type: \(json?["type"] as? String ?? "nil")")
            print("Employee count: \(json?["employeeCount"] as? Int ?? 0)")
        } catch {
            print("❌ JSON parsing failed: \(error)")
            print("\nActual final content bytes:")
            print(Array(final.utf8.prefix(50)))
        }
    }
} else {
    print("❌ No final channel found!")
}

print("\nDisplay content (what would be returned):")
print(parsed.displayContent)