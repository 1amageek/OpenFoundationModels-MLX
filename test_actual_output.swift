#!/usr/bin/env swift

import Foundation

// Test with the exact output from the log
let actualOutput = """
<|channel|>analysis<|message|>We need to output JSON with fields: name, founded, type, employeeCount, headquarters, departments. The user said "technology startup company founded in 2020 with 2 departments (engineering and sales). Each department should have 1-2 projects."

We need to fill realistic data. We can invent a name, e.g., "AstraTech Solutions". Founded 2020. Type "startup". employeeCount maybe 25. Headquarters maybe a city in US. Each department: engineering and sales. Each with manager details. headCount maybe 15 for engineering, 10 for sales. Projects: For engineering: 2 projects: "AI Platform", "Cloud Migration". For sales: 1 project: "Enterprise Outreach". Provide startDate and status, budget etc. Budget can be number or null. Provide teamSize. Budget maybe 500000 for AI Platform, 200000 for Cloud Migration, 150000 for Enterprise Outreach.

Also need manager details: email, firstName, lastName, level (junior, senior, lead, manager). For engineering manager: level "lead" maybe senior. For sales manager: level "manager" maybe senior.

EmployeeCount: sum of headCounts? 15+10=25. Good.

Headquarters: street, city, country, postalCode. Use e.g., "123 Innovation Drive", "San Francisco", "USA", "94107".

We must include required fields: name, founded, type, employeeCount, headquarters, departments.

Also for each department: name, type, headCount, manager, projects (array or null). For each project: name, status, startDate, teamSize, budget (optional null). Provide startDate in ISO format. Use 2022-01-15, etc.

Let's craft JSON accordingly.

We need to ensure all required fields present. For projects array: each object with required fields: name, status, startDate, teamSize. budget is optional. Provide budget as number.

We must not include any extra keys.

Let's produce final JSON.<|end|><|start|>assistant<|channel|>final<|message|>{
  "name": "AstraTech Solutions",
  "founded": 2020,
  "type": "startup",
  "employeeCount": 25,
  "headquarters": {
    "street": "123 Innovation Drive",
    "city": "San Francisco",
    "country": "USA",
    "postalCode": "94107"
  },
  "departments": [
    {
      "name": "Engineering",
      "type": "engineering",
      "headCount": 15,
      "manager": {
        "firstName": "Liam",
        "lastName": "Hernandez",
        "email": "liam.hernandez@astratech.com",
        "level": "lead",
        "yearsExperience": 8
      },
      "projects": [
        {
          "name": "AI Platform",
          "status": "active",
          "startDate": "2023-03-01",
          "teamSize": 6,
          "budget": 500000
        },
        {
          "name": "Cloud Migration",
          "status": "planning",
          "startDate": "2024-01-15",
          "teamSize": 4,
          "budget": 200000
        }
      ]
    },
    {
      "name": "Sales",
      "type": "sales",
      "headCount": 10,
      "manager": {
        "firstName": "Sophia",
        "lastName": "Nguyen",
        "email": "sophia.nguyen@astratech.com",
        "level": "manager",
        "yearsExperience": 6
      },
      "projects": [
        {
          "name": "Enterprise Outreach",
          "status": "active",
          "startDate": "2022-11-01",
          "teamSize": 5,
          "budget": 150000
        }
      ]
    }
  ]
}
========== END OUTPUT ==========
"""

// Extract just the final channel content
func extractFinalChannel(_ raw: String) -> String? {
    if let channelStart = raw.range(of: "<|channel|>final<|message|>") {
        let afterChannel = String(raw[channelStart.upperBound...])
        
        // Find where JSON ends (at the closing brace, not at end markers)
        var braceCount = 0
        var inString = false
        var escapeNext = false
        var jsonEndIndex: String.Index?
        
        for (index, char) in afterChannel.enumerated() {
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
                        // Found the end of JSON
                        jsonEndIndex = afterChannel.index(afterChannel.startIndex, offsetBy: index + 1)
                        break
                    }
                }
            }
        }
        
        if let endIndex = jsonEndIndex {
            return String(afterChannel[..<endIndex])
        }
    }
    return nil
}

print("Testing JSON extraction from actual output...")
print(String(repeating: "=", count: 60))

if let finalContent = extractFinalChannel(actualOutput) {
    print("Extracted final channel content:")
    print("Length: \(finalContent.count) characters")
    print("\nLast 50 characters of extracted content:")
    print("[\(String(finalContent.suffix(50)))]")
    
    // Try to parse as JSON
    if let data = finalContent.data(using: .utf8) {
        do {
            let json = try JSONSerialization.jsonObject(with: data) as? [String: Any]
            print("\n✅ JSON parsed successfully!")
            print("Company name: \(json?["name"] as? String ?? "nil")")
            print("Founded: \(json?["founded"] as? Int ?? 0)")
            print("Type: \(json?["type"] as? String ?? "nil")")
        } catch {
            print("\n❌ JSON parsing failed: \(error)")
            print("\nFirst 100 chars of content:")
            print(String(finalContent.prefix(100)))
            print("\nLast 100 chars of content:")
            print(String(finalContent.suffix(100)))
        }
    }
} else {
    print("❌ Could not extract final channel!")
}