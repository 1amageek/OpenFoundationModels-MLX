import Testing
import Foundation
@testable import OpenFoundationModelsMLX

@Suite("HarmonyParser Standalone Tests")
struct HarmonyParserStandaloneTest {
    
    // Test the exact problematic pattern from the logs
    @Test("Actual GPT-OSS output pattern")
    func actualGPTOSSOutputPattern() throws {
        // This is the exact pattern that was failing:
        // - analysis channel WITHOUT <|start|>assistant
        // - final channel WITH <|start|>assistant
        let input = """
        <|channel|>analysis<|message|>We need to output JSON that matches the schema. Must provide company name, founded to be 2020, type "startup". 

        Let's craft JSON:
        {
          "name": "NovaTech Solutions"
        }

        All good.<|end|><|start|>assistant<|channel|>final<|message|>{"name":"NovaTech Solutions","founded":2020,"type":"startup","employeeCount":35,"headquarters":{"street":"123 Innovation Drive","city":"San Francisco","country":"USA","postalCode":"94107"},"departments":[{"name":"Engineering","type":"engineering","headCount":20,"manager":{"firstName":"Elena","lastName":"Martinez","email":"elena.martinez@novatech.com","level":"lead","yearsExperience":12},"projects":[{"name":"Project Aurora","status":"active","startDate":"2023-01-15","teamSize":8,"budget":1200000},{"name":"Project Nebula","status":"planning","startDate":"2024-06-01","teamSize":4,"budget":null}]},{"name":"Sales","type":"sales","headCount":15,"manager":{"firstName":"Michael","lastName":"Johnson","email":"michael.johnson@novatech.com","level":"manager","yearsExperience":9},"projects":[{"name":"Market Expansion Q3","status":"active","startDate":"2023-04-01","teamSize":5,"budget":500000}]}]}
        """
        
        print("\n🔍 Testing actual GPT-OSS output pattern...")
        let parsed = HarmonyParser.parse(input)
        
        // CRITICAL: Both channels must be extracted
        #expect(parsed.analysis != nil, "Analysis channel must be extracted")
        #expect(parsed.final != nil, "Final channel must be extracted")
        
        // Verify analysis content is not empty
        #expect(!(parsed.analysis?.isEmpty ?? true), "Analysis channel must not be empty")
        print("✅ Analysis channel extracted: \(parsed.analysis?.prefix(50) ?? "nil")...")
        
        // Verify final content is not empty
        #expect(!(parsed.final?.isEmpty ?? true), "Final channel must not be empty")
        print("✅ Final channel extracted: \(parsed.final?.prefix(50) ?? "nil")...")
        
        // Most important: Final channel must contain valid JSON
        guard let finalContent = parsed.final else {
            Issue.record("Final channel is nil")
            return
        }
        
        guard let data = finalContent.data(using: .utf8) else {
            Issue.record("Cannot convert final content to data")
            return
        }
        
        do {
            let json = try JSONSerialization.jsonObject(with: data) as? [String: Any]
            #expect(json != nil, "Final channel must contain valid JSON")
            
            // Verify key fields from the JSON
            #expect(json?["name"] as? String == "NovaTech Solutions")
            #expect(json?["founded"] as? Int == 2020)
            #expect(json?["type"] as? String == "startup")
            #expect(json?["employeeCount"] as? Int == 35)
            
            if let headquarters = json?["headquarters"] as? [String: Any] {
                #expect(headquarters["city"] as? String == "San Francisco")
            }
            
            if let departments = json?["departments"] as? [[String: Any]] {
                #expect(departments.count == 2, "Should have 2 departments")
            }
            
            print("✅ JSON is valid and contains expected structure")
            
        } catch {
            Issue.record("Failed to parse JSON from final channel: \(error)")
        }
        
        // Verify displayContent returns final channel
        #expect(parsed.displayContent == parsed.final, "displayContent should return final channel when available")
        
        print("✅ All assertions passed!")
    }
    
    // Test simplified case
    @Test("Simplified mixed channels")
    func simplifiedMixedChannels() throws {
        let input = """
        <|channel|>analysis<|message|>Analyzing<|end|>
        <|start|>assistant<|channel|>final<|message|>{"test": true}<|end|>
        """
        
        let parsed = HarmonyParser.parse(input)
        
        #expect(parsed.analysis == "Analyzing")
        #expect(parsed.final == "{\"test\": true}")
        
        // Verify JSON is valid
        let data = parsed.final!.data(using: .utf8)!
        let json = try JSONSerialization.jsonObject(with: data) as? [String: Any]
        #expect(json?["test"] as? Bool == true)
    }
}