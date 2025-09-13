#!/usr/bin/env swift

import Foundation

// Simplified HarmonyParser for testing
struct HarmonyParser {
    static func parse(_ raw: String) -> (analysis: String?, final: String?) {
        var channels: [String: String] = [:]
        
        // Look for channel blocks that may or may not have <|start|>assistant prefix
        // Pattern: [<|start|>assistant]<|channel|>{name}<|message|>{content}[<|end|>]
        
        // Find analysis channel
        if let channelStart = raw.range(of: "<|channel|>analysis<|message|>") {
            let afterChannel = String(raw[channelStart.upperBound...])
            if let endMarker = afterChannel.range(of: "<|end|>") {
                channels["analysis"] = String(afterChannel[..<endMarker.lowerBound])
            } else if let nextChannel = afterChannel.range(of: "<|channel|>") {
                channels["analysis"] = String(afterChannel[..<nextChannel.lowerBound])
            } else if let nextStart = afterChannel.range(of: "<|start|>") {
                channels["analysis"] = String(afterChannel[..<nextStart.lowerBound])
            } else {
                channels["analysis"] = afterChannel
            }
        }
        
        // Find final channel
        if let channelStart = raw.range(of: "<|channel|>final<|message|>") {
            let afterChannel = String(raw[channelStart.upperBound...])
            if let endMarker = afterChannel.range(of: "<|end|>") {
                channels["final"] = String(afterChannel[..<endMarker.lowerBound])
            } else if let nextChannel = afterChannel.range(of: "<|channel|>") {
                channels["final"] = String(afterChannel[..<nextChannel.lowerBound])
            } else if let nextStart = afterChannel.range(of: "<|start|>") {
                channels["final"] = String(afterChannel[..<nextStart.lowerBound])
            } else {
                channels["final"] = afterChannel
            }
        }
        
        // If no channels found but there's JSON, extract it
        if channels["final"] == nil {
            if let jsonStart = raw.firstIndex(of: "{"),
               let jsonEnd = raw.lastIndex(of: "}") {
                let json = String(raw[jsonStart...jsonEnd])
                if let data = json.data(using: .utf8),
                   let _ = try? JSONSerialization.jsonObject(with: data) {
                    channels["final"] = json
                }
            }
        }
        
        return (channels["analysis"], channels["final"])
    }
}

// Test with actual GPT-OSS output
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

print("Testing simplified HarmonyParser...")
print(String(repeating: "=", count: 60))

let (analysis, final) = HarmonyParser.parse(actualOutput)

print("Has analysis channel: \(analysis != nil)")
print("Has final channel: \(final != nil)")

if let analysis = analysis {
    print("\nAnalysis channel (first 100 chars):")
    print(String(analysis.prefix(100)))
}

if let final = final {
    print("\nFinal channel JSON extracted!")
    print("Length: \(final.count) characters")
    
    // Try to parse as JSON
    if let data = final.data(using: .utf8) {
        do {
            let json = try JSONSerialization.jsonObject(with: data) as? [String: Any]
            print("\n✅ JSON parsed successfully!")
            print("Company name: \(json?["name"] as? String ?? "nil")")
            print("Founded: \(json?["founded"] as? Int ?? 0)")
            print("Type: \(json?["type"] as? String ?? "nil")")
            print("Employee count: \(json?["employeeCount"] as? Int ?? 0)")
            
            // Check departments
            if let departments = json?["departments"] as? [[String: Any]] {
                print("\nDepartments: \(departments.count)")
                for dept in departments {
                    print("  - \(dept["name"] as? String ?? "unknown"): \(dept["headCount"] as? Int ?? 0) employees")
                }
            }
        } catch {
            print("\n❌ JSON parsing failed: \(error)")
        }
    }
} else {
    print("\n❌ No final channel found!")
}