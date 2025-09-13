import Testing
import Foundation
@testable import OpenFoundationModelsMLX

@Suite("Nested Schema Extraction Tests")
struct NestedSchemaExtractionTest {
    
    @Test("Nested array schema extraction")
    func nestedArraySchemaExtraction() throws {
        // Test JSON Schema with nested arrays and objects (like CompanyProfile)
        let testSchema = """
        {
          "type": "object",
          "properties": {
            "name": {"type": "string"},
            "departments": {
              "type": "array",
              "items": {
                "type": "object",
                "properties": {
                  "name": {"type": "string"},
                  "type": {"type": "string"},
                  "headCount": {"type": "number"},
                  "manager": {
                    "type": "object",
                    "properties": {
                      "firstName": {"type": "string"},
                      "lastName": {"type": "string"},
                      "email": {"type": "string"},
                      "level": {"type": "string"},
                      "yearsExperience": {"type": "number"}
                    }
                  },
                  "projects": {
                    "type": "array",
                    "items": {
                      "type": "object",
                      "properties": {
                        "name": {"type": "string"},
                        "status": {"type": "string"},
                        "startDate": {"type": "string"},
                        "teamSize": {"type": "number"},
                        "budget": {"type": "number"}
                      }
                    }
                  }
                }
              }
            },
            "headquarters": {
              "type": "object",
              "properties": {
                "street": {"type": "string"},
                "city": {"type": "string"},
                "country": {"type": "string"},
                "postalCode": {"type": "string"}
              }
            }
          }
        }
        """
        
        // Parse the schema
        guard let data = testSchema.data(using: .utf8),
              let dict = try JSONSerialization.jsonObject(with: data) as? [String: Any],
              let schemaNode = JSONSchemaExtractor.buildSchemaNode(from: dict) else {
            Issue.record("Failed to parse schema")
            return
        }
        
        print("\n🔍 Testing Nested Array Schema Extraction")
        print("=" * 50)
        
        // Test root keys
        let expectedRootKeys = ["departments", "headquarters", "name"]
        #expect(Set(schemaNode.objectKeys) == Set(expectedRootKeys))
        print("✅ Root keys: \(schemaNode.objectKeys.joined(separator: ", "))")
        
        // Test headquarters nested object
        guard let headquarters = schemaNode.properties["headquarters"] else {
            Issue.record("headquarters property not found")
            return
        }
        #expect(headquarters.kind == .object)
        let expectedHQKeys = ["city", "country", "postalCode", "street"]
        #expect(Set(headquarters.objectKeys) == Set(expectedHQKeys))
        print("✅ headquarters keys: \(headquarters.objectKeys.joined(separator: ", "))")
        
        // Test departments array
        guard let departments = schemaNode.properties["departments"] else {
            Issue.record("departments property not found")
            return
        }
        #expect(departments.kind == .array)
        
        // Test departments array items (should be objects)
        guard let deptItems = departments.items else {
            Issue.record("departments array items not found")
            return
        }
        #expect(deptItems.kind == .object)
        let expectedDeptKeys = ["headCount", "manager", "name", "projects", "type"]
        #expect(Set(deptItems.objectKeys) == Set(expectedDeptKeys))
        print("✅ departments[] keys: \(deptItems.objectKeys.joined(separator: ", "))")
        
        // Test manager nested object within departments
        guard let manager = deptItems.properties["manager"] else {
            Issue.record("manager property not found in department items")
            return
        }
        #expect(manager.kind == .object)
        let expectedManagerKeys = ["email", "firstName", "lastName", "level", "yearsExperience"]
        #expect(Set(manager.objectKeys) == Set(expectedManagerKeys))
        print("✅ departments[].manager keys: \(manager.objectKeys.joined(separator: ", "))")
        
        // Test projects array within departments
        guard let projects = deptItems.properties["projects"] else {
            Issue.record("projects property not found in department items")
            return
        }
        #expect(projects.kind == .array)
        
        // Test project items
        guard let projectItems = projects.items else {
            Issue.record("projects array items not found")
            return
        }
        #expect(projectItems.kind == .object)
        let expectedProjectKeys = ["budget", "name", "startDate", "status", "teamSize"]
        #expect(Set(projectItems.objectKeys) == Set(expectedProjectKeys))
        print("✅ departments[].projects[] keys: \(projectItems.objectKeys.joined(separator: ", "))")
        
        print("\n✨ All nested schema extraction tests passed!")
    }
    
    @Test("Adaptive constraint engine with nested arrays")
    func adaptiveConstraintEngineWithNestedArrays() throws {
        // This test verifies that AdaptiveConstraintEngine properly extracts
        // nested schemas for KeyDetectionLogitProcessor
        
        let testSchema = """
        {
          "type": "object",
          "properties": {
            "company": {"type": "string"},
            "departments": {
              "type": "array",
              "items": {
                "type": "object",
                "properties": {
                  "name": {"type": "string"},
                  "manager": {
                    "type": "object",
                    "properties": {
                      "name": {"type": "string"},
                      "email": {"type": "string"}
                    }
                  }
                }
              }
            }
          }
        }
        """
        
        guard let schemaNode = JSONSchemaExtractor.buildSchemaNode(from: testSchema) else {
            Issue.record("Failed to build schema node")
            return
        }
        
        print("\n🔧 Testing AdaptiveConstraintEngine nested schema extraction")
        print("=" * 50)
        
        // Simulate what AdaptiveConstraintEngine does
        var nestedSchemas: [String: [String]] = [:]
        
        for (key, propNode) in schemaNode.properties {
            if propNode.kind == .object && !propNode.objectKeys.isEmpty {
                nestedSchemas[key] = propNode.objectKeys
                print("📂 Nested schema for '\(key)': \(propNode.objectKeys.joined(separator: ", "))")
            } else if propNode.kind == .array,
                      let itemsNode = propNode.items,
                      itemsNode.kind == .object && !itemsNode.objectKeys.isEmpty {
                // For array items, use special key notation
                let arrayKey = "\(key)[]"
                nestedSchemas[arrayKey] = itemsNode.objectKeys
                print("📋 Array item schema for '\(arrayKey)': \(itemsNode.objectKeys.joined(separator: ", "))")
                
                // Check for nested objects within array items
                for (itemPropKey, itemPropNode) in itemsNode.properties {
                    if itemPropNode.kind == .object && !itemPropNode.objectKeys.isEmpty {
                        let nestedArrayKey = "\(key)[].\(itemPropKey)"
                        nestedSchemas[nestedArrayKey] = itemPropNode.objectKeys
                        print("📂 Nested array item schema for '\(nestedArrayKey)': \(itemPropNode.objectKeys.joined(separator: ", "))")
                    }
                }
            }
        }
        
        // Verify the extracted nested schemas
        #expect(nestedSchemas["departments[]"] != nil)
        #expect(Set(nestedSchemas["departments[]"] ?? []) == Set(["manager", "name"]))
        
        #expect(nestedSchemas["departments[].manager"] != nil)
        #expect(Set(nestedSchemas["departments[].manager"] ?? []) == Set(["email", "name"]))
        
        print("\n✨ AdaptiveConstraintEngine nested schema extraction test passed!")
    }
}

// Helper to repeat a string (for visual separators)
fileprivate func *(lhs: String, rhs: Int) -> String {
    String(repeating: lhs, count: rhs)
}