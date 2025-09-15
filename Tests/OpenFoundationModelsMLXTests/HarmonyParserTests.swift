import Testing
import Foundation
@testable import OpenFoundationModelsMLX
import OpenFoundationModelsMLXGPT

@Suite("HarmonyParser Tests")
struct HarmonyParserTests {
    
    // MARK: - Basic Channel Parsing Tests
    
    @Test("Parse analysis channel without start marker")
    func parseAnalysisChannelWithoutStartMarker() throws {
        // Test analysis channel that appears WITHOUT <|start|>assistant prefix
        let input = """
        <|channel|>analysis<|message|>This is the analysis content.
        It can span multiple lines.
        <|end|>
        """
        
        let parsed = HarmonyParser.parse(input)
        
        #expect(parsed.analysis != nil, "Should extract analysis channel")
        #expect(parsed.analysis == "This is the analysis content.\nIt can span multiple lines.")
        #expect(parsed.final == nil, "Should not have final channel")
        #expect(parsed.commentary == nil, "Should not have commentary channel")
    }
    
    @Test("Parse final channel with start marker")
    func parseFinalChannelWithStartMarker() throws {
        // Test final channel that appears WITH <|start|>assistant prefix
        let input = """
        <|start|>assistant<|channel|>final<|message|>{"result": "success", "value": 42}<|end|>
        """
        
        let parsed = HarmonyParser.parse(input)
        
        #expect(parsed.final != nil, "Should extract final channel")
        #expect(parsed.final == "{\"result\": \"success\", \"value\": 42}")
        #expect(parsed.analysis == nil, "Should not have analysis channel")
        
        // Verify JSON is valid
        let data = parsed.final!.data(using: .utf8)!
        let json = try JSONSerialization.jsonObject(with: data) as? [String: Any]
        #expect(json != nil)
        #expect(json?["result"] as? String == "success")
        #expect(json?["value"] as? Int == 42)
    }
    
    @Test("Parse mixed channel formats")
    func parseMixedChannelFormats() throws {
        // Test the actual GPT-OSS output pattern:
        // - analysis channel WITHOUT start marker
        // - final channel WITH start marker
        let input = """
        <|channel|>analysis<|message|>Analyzing the request.
        Need to generate JSON.
        <|end|>
        <|start|>assistant<|channel|>final<|message|>{"name": "test", "id": 123}<|end|>
        """
        
        let parsed = HarmonyParser.parse(input)
        
        #expect(parsed.analysis != nil, "Should extract analysis channel")
        #expect(parsed.analysis?.contains("Analyzing the request") ?? false)
        
        #expect(parsed.final != nil, "Should extract final channel")
        #expect(parsed.final == "{\"name\": \"test\", \"id\": 123}")
        
        // displayContent should return final when available
        #expect(parsed.displayContent == parsed.final)
    }
    
    @Test("Multi-channel parsing with actual GPT-OSS format")
    func multiChannelParsing() throws {
        // Test input with both analysis and final channels (actual GPT-OSS format)
        let input = """
        <|start|>assistant<|channel|>analysis<|message|>We must output JSON conforming to schema. Provide company name, founded 2020, type startup, employeeCount maybe sum of headCounts. Provide headquarters with address. Departments: engineering and sales. Each with headCount, manager object, type, projects array 1-2 items. Projects include name, status, startDate, teamSize, maybe budget null or number. Use realistic data.

        We need manager details: email, firstName, lastName, level enum: junior, senior, lead, manager. yearsExperience integer.

        Let's craft.

        Company name: "NovaTech Solutions". Headcount: For engineering maybe 15, sales 10. Total employeeCount = 25.

        Headquarters: street "123 Innovation Drive", city "San Francisco", country "USA", postalCode "94107".

        Departments:

        Engineering: headCount 15, manager: firstName "Ethan", lastName "Wang", email "ethan.wang@novatech.io", level "lead", yearsExperience 10. type "engineering". Projects: 2 projects.

        Project 1: "Alpha Platform" name, status "active", startDate "2023-01-15", teamSize 8, budget 500000. Project 2: "Beta API" status "planning", startDate "2024-03-01", teamSize 4, budget null.

        Sales: headCount 10, manager: firstName "Maya", lastName "Chung", email "maya.chung@novatech.io", level "manager", yearsExperience 12. type "sales". Projects: 1 project.

        Project: "Enterprise Outreach" status "active", startDate "2023-06-01", teamSize 6, budget 200000.

        Yes.

        Check JSON validity.

        Also each department must have required fields: name, type, headCount, manager. manager required fields: firstName, lastName, email, level, yearsExperience. Projects optional null or array. Provide array.

        Ensure all date strings are ISO format YYYY-MM-DD.

        Return as JSON object.<|end|><|start|>assistant<|channel|>final<|message|>{
          "name": "NovaTech Solutions",
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
                "firstName": "Ethan",
                "lastName": "Wang",
                "email": "ethan.wang@novatech.io",
                "level": "lead",
                "yearsExperience": 10
              },
              "projects": [
                {
                  "name": "Alpha Platform",
                  "status": "active",
                  "startDate": "2023-01-15",
                  "teamSize": 8,
                  "budget": 500000
                },
                {
                  "name": "Beta API",
                  "status": "planning",
                  "startDate": "2024-03-01",
                  "teamSize": 4,
                  "budget": null
                }
              ]
            },
            {
              "name": "Sales",
              "type": "sales",
              "headCount": 10,
              "manager": {
                "firstName": "Maya",
                "lastName": "Chung",
                "email": "maya.chung@novatech.io",
                "level": "manager",
                "yearsExperience": 12
              },
              "projects": [
                {
                  "name": "Enterprise Outreach",
                  "status": "active",
                  "startDate": "2023-06-01",
                  "teamSize": 6,
                  "budget": 200000
                }
              ]
            }
          ]
        }
        """
        
        let parsed = HarmonyParser.parse(input)
        
        // Check that both channels were extracted
        #expect(parsed.analysis != nil)
        #expect(parsed.final != nil)
        
        // Check that analysis channel contains expected content
        #expect(parsed.analysis?.contains("We must output JSON") ?? false)
        
        // Check that final channel contains valid JSON
        #expect(parsed.final != nil)
        if let finalContent = parsed.final {
            print("Final content extracted:")
            print("=========================")
            print(finalContent)
            print("=========================")
            
            // Try to parse as JSON
            let data = finalContent.data(using: .utf8)!
            let json = try JSONSerialization.jsonObject(with: data) as? [String: Any]
            #expect(json != nil)
            #expect(json?["name"] as? String == "NovaTech Solutions")
            #expect(json?["founded"] as? Int == 2020)
            #expect(json?["type"] as? String == "startup")
            #expect(json?["employeeCount"] as? Int == 25)
        }
        
        // Check displayContent returns final channel
        #expect(parsed.displayContent == parsed.final)
    }
    
    @Test("Formatted JSON extraction")
    func formattedJSONExtraction() throws {
        // Test with formatted (pretty-printed) JSON
        let input = """
        <|start|>assistant<|channel|>final<|message|>{
          "name": "Test Company",
          "founded": 2020,
          "nested": {
            "key1": "value1",
            "key2": "value2"
          },
          "array": [
            {
              "item": 1
            },
            {
              "item": 2
            }
          ]
        }<|end|>
        """
        
        let parsed = HarmonyParser.parse(input)
        
        #expect(parsed.final != nil)
        if let finalContent = parsed.final {
            print("Formatted JSON extracted:")
            print("=========================")
            print(finalContent)
            print("=========================")
            
            // Verify it's valid JSON
            let data = finalContent.data(using: .utf8)!
            let json = try JSONSerialization.jsonObject(with: data) as? [String: Any]
            #expect(json != nil)
            #expect(json?["name"] as? String == "Test Company")
            #expect(json?["founded"] as? Int == 2020)
        }
    }
    
    @Test("JSON without end marker")
    func jsonWithoutEndMarker() throws {
        // Test JSON that might not have explicit end marker
        let input = """
        <|start|>assistant<|channel|>final<|message|>{
          "simple": "json",
          "number": 42
        }
        """
        
        let parsed = HarmonyParser.parse(input)
        
        #expect(parsed.final != nil)
        if let finalContent = parsed.final {
            print("JSON without end marker:")
            print("=========================")
            print(finalContent)
            print("=========================")
            
            // Should still extract the JSON
            let data = finalContent.data(using: .utf8)!
            let json = try JSONSerialization.jsonObject(with: data) as? [String: Any]
            #expect(json != nil)
            #expect(json?["simple"] as? String == "json")
            #expect(json?["number"] as? Int == 42)
        }
    }
    
    // MARK: - JSON Extraction Tests
    
    @Test("JSON extraction from final channel")
    func jsonExtractionFromFinalChannel() throws {
        // Test that JSON is properly extracted from final channel content
        let input = """
        <|start|>assistant<|channel|>final<|message|>Some text before JSON
        {
          "extracted": true,
          "nested": {
            "value": 100
          }
        }
        Some text after JSON<|end|>
        """
        
        let parsed = HarmonyParser.parse(input)
        
        #expect(parsed.final != nil, "Should extract final channel")
        // Should extract just the JSON, not surrounding text
        #expect(parsed.final!.hasPrefix("{"))
        #expect(parsed.final!.hasSuffix("}"))
        
        // Verify extracted JSON is valid
        let data = parsed.final!.data(using: .utf8)!
        let json = try JSONSerialization.jsonObject(with: data) as? [String: Any]
        #expect(json != nil)
        #expect(json?["extracted"] as? Bool == true)
        if let nested = json?["nested"] as? [String: Any] {
            #expect(nested["value"] as? Int == 100)
        } else {
            Issue.record("Nested object not found")
        }
    }
    
    @Test("JSON array extraction")
    func jsonArrayExtraction() throws {
        // Test extraction of JSON array
        let input = """
        <|start|>assistant<|channel|>final<|message|>[
          {"id": 1, "name": "Item 1"},
          {"id": 2, "name": "Item 2"}
        ]<|end|>
        """
        
        let parsed = HarmonyParser.parse(input)
        
        #expect(parsed.final != nil)
        #expect(parsed.final!.hasPrefix("["))
        #expect(parsed.final!.hasSuffix("]"))
        
        let data = parsed.final!.data(using: .utf8)!
        let json = try JSONSerialization.jsonObject(with: data) as? [[String: Any]]
        #expect(json != nil)
        #expect(json?.count == 2)
        #expect(json?[0]["id"] as? Int == 1)
        #expect(json?[1]["name"] as? String == "Item 2")
    }
    
    @Test("Fallback JSON extraction")
    func fallbackJSONExtraction() throws {
        // Test fallback when no channels are found but JSON is present
        let input = """
        {
          "direct": "json",
          "no_channels": true
        }
        """
        
        let parsed = HarmonyParser.parse(input)
        
        #expect(parsed.final != nil, "Should extract JSON even without channels")
        let data = parsed.final!.data(using: .utf8)!
        let json = try JSONSerialization.jsonObject(with: data) as? [String: Any]
        #expect(json?["direct"] as? String == "json")
        #expect(json?["no_channels"] as? Bool == true)
    }
    
    // MARK: - Edge Cases
    
    @Test("Empty channel content")
    func emptyChannelContent() throws {
        // Test channel with empty content
        let input = """
        <|channel|>analysis<|message|><|end|>
        <|start|>assistant<|channel|>final<|message|><|end|>
        """
        
        let parsed = HarmonyParser.parse(input)
        
        // Empty content should be captured as empty string
        #expect(parsed.analysis != nil)
        #expect(parsed.analysis == "")
        #expect(parsed.final != nil)
        #expect(parsed.final == "")
    }
    
    @Test("Channel without end marker")
    func channelWithoutEndMarker() throws {
        // Test channel that doesn't have explicit end marker
        let input = """
        <|channel|>analysis<|message|>Content without end marker
        """
        
        let parsed = HarmonyParser.parse(input)
        
        #expect(parsed.analysis != nil)
        #expect(parsed.analysis == "Content without end marker")
    }
    
    @Test("Multiple channels of same type")
    func multipleChannelsOfSameType() throws {
        // Test when the same channel appears multiple times (should use last one)
        let input = """
        <|channel|>final<|message|>First attempt<|end|>
        <|start|>assistant<|channel|>final<|message|>{"correct": "json"}<|end|>
        """
        
        let parsed = HarmonyParser.parse(input)
        
        // Should prefer the one with start marker
        #expect(parsed.final != nil)
        #expect(parsed.final == "{\"correct\": \"json\"}")
    }
    
    @Test("Complex nested JSON")
    func complexNestedJSON() throws {
        // Test with complex nested JSON structure
        let input = """
        <|start|>assistant<|channel|>final<|message|>{
          "company": {
            "name": "TechCorp",
            "departments": [
              {
                "name": "Engineering",
                "employees": 50,
                "projects": [
                  {"name": "Project A", "status": "active"}
                ]
              }
            ]
          }
        }<|end|>
        """
        
        let parsed = HarmonyParser.parse(input)
        
        #expect(parsed.final != nil)
        let data = parsed.final!.data(using: .utf8)!
        let json = try JSONSerialization.jsonObject(with: data) as? [String: Any]
        #expect(json != nil)
        
        if let company = json?["company"] as? [String: Any],
           let departments = company["departments"] as? [[String: Any]] {
            #expect(departments.count == 1)
            #expect(departments[0]["name"] as? String == "Engineering")
            #expect(departments[0]["employees"] as? Int == 50)
        } else {
            Issue.record("Complex JSON structure not parsed correctly")
        }
    }
    
    @Test("Actual GPT-OSS output parsing")
    func actualGPTOSSOutput() throws {
        // Test with the exact output from the log
        let input = """
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
        
        let parsed = HarmonyParser.parse(input)
        
        print("Actual GPT-OSS output parsing:")
        print("===============================")
        print("Has analysis: \(parsed.analysis != nil)")
        print("Has final: \(parsed.final != nil)")
        
        if let finalContent = parsed.final {
            print("\nFinal content (first 200 chars):")
            print(String(finalContent.prefix(200)))
            
            // Try to parse as JSON
            if let data = finalContent.data(using: .utf8) {
                let json = try JSONSerialization.jsonObject(with: data) as? [String: Any]
                #expect(json != nil, "JSON should be parseable")
                print("\nJSON parsed successfully!")
                print("Company name: \(json?["name"] as? String ?? "nil")")
                print("Founded: \(json?["founded"] as? Int ?? 0)")
                print("Type: \(json?["type"] as? String ?? "nil")")
            }
        } else {
            Issue.record("No final channel extracted from actual GPT-OSS output")
        }
    }
}