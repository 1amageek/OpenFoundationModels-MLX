import Foundation
import Testing
@testable import OpenFoundationModelsMLX

@Suite("JSON Detection + Schema Constraints Integration")
struct JSONDetectionIntegrationTest {

    @Test("Complete integration: LLM output → JSON detection → Schema constraints")
    func testCompleteIntegration() {
        // Company profile schema
        let schema: [String: Any] = [
            "type": "object",
            "properties": [
                "name": ["type": "string"],
                "founded": ["type": "integer"],
                "type": ["type": "string"],
                "employeeCount": ["type": "integer"],
                "headquarters": [
                    "type": "object",
                    "properties": [
                        "street": ["type": "string"],
                        "city": ["type": "string"],
                        "country": ["type": "string"],
                        "postalCode": ["type": "string"]
                    ]
                ],
                "departments": [
                    "type": "array",
                    "items": [
                        "type": "object",
                        "properties": [
                            "name": ["type": "string"],
                            "type": ["type": "string"],
                            "headCount": ["type": "integer"],
                            "manager": [
                                "type": "object",
                                "properties": [
                                    "firstName": ["type": "string"],
                                    "lastName": ["type": "string"],
                                    "email": ["type": "string"],
                                    "level": ["type": "string"],
                                    "yearsExperience": ["type": "integer"]
                                ]
                            ]
                        ]
                    ]
                ]
            ]
        ]

        // Simulate LLM output with analysis and final JSON
        let llmOutput = """
        <|channel|>analysis<|message|>
        I need to create a company profile. Let me think about the structure.
        The company needs: name, founded year, type, employee count, headquarters, and departments.
        Each department should have a manager with their details.
        <|end|>
        <|channel|>final<|message|>
        {
          "name": "TechCorp",
          "founded": 2020,
          "type": "startup",
          "employeeCount": 150,
          "headquarters": {
            "street": "123 Innovation Way",
            "city": "San Francisco",
            "country": "USA",
            "postalCode": "94107"
          },
          "departments": [
            {
              "name": "Engineering",
              "type": "engineering",
              "headCount": 80,
              "manager": {
                "firstName": "Alice",
                "lastName": "Chen",
                "email": "alice.chen@techcorp.com",
                "level": "senior",
                "yearsExperience": 10
              }
            },
            {
              "name": "Sales",
              "type": "sales",
              "headCount": 40,
              "manager": {
                "firstName": "Bob",
                "lastName": "Smith",
                "email": "bob.smith@techcorp.com",
                "level": "lead",
                "yearsExperience": 8
              }
            }
          ]
        }
        <|end|>
        """

        var extractor = JSONExtractor()
        let detector = JSONSchemaContextDetector(schema: schema)
        var jsonStateMachine = JSONStateMachine()

        var partialJSON = ""
        var detectedKeys: [String] = []
        var constraintSnapshots: [(context: String, availableKeys: [String], usedKeys: [String])] = []

        // Process the LLM output character by character
        for char in llmOutput {
            let shouldProcess = extractor.processCharacter(char)

            if shouldProcess {
                partialJSON.append(char)

                // Update JSON state machine
                let previousPhase = jsonStateMachine.phase
                jsonStateMachine.processCharacter(char)

                // Detect completed keys
                if case .inString(.body(kind: .key, escaped: false)) = previousPhase {
                    if case .inObject(.expectColon) = jsonStateMachine.phase {
                        let key = jsonStateMachine.currentKey
                        if !key.isEmpty {
                            detectedKeys.append(key)
                        }
                    }
                }

                // Capture constraints at various points
                if char == "{" {
                    let availableKeys = detector.getAvailableKeys(from: partialJSON)
                    let context = partialJSON.count < 50 ? "root" :
                                 partialJSON.contains("headquarters") ? "headquarters" :
                                 partialJSON.contains("manager") ? "manager" :
                                 partialJSON.contains("departments") ? "department" : "unknown"

                    constraintSnapshots.append((
                        context: context,
                        availableKeys: availableKeys,
                        usedKeys: detectedKeys
                    ))
                }
            }

            // Reset when JSON ends
            if jsonStateMachine.isComplete {
                jsonStateMachine.reset()
            }
        }

        // Verify results
        print("Detected keys: \(detectedKeys)")
        print("\nConstraint snapshots:")
        for snapshot in constraintSnapshots {
            print("  Context: \(snapshot.context)")
            print("    Available: \(snapshot.availableKeys)")
            print("    Used so far: \(snapshot.usedKeys.suffix(3))")
        }

        // Test assertions
        #expect(detectedKeys.count > 0, "Should detect JSON keys")
        #expect(detectedKeys.contains("name"), "Should detect 'name' key")
        #expect(detectedKeys.contains("founded"), "Should detect 'founded' key")
        #expect(detectedKeys.contains("headquarters"), "Should detect 'headquarters' key")
        #expect(detectedKeys.contains("departments"), "Should detect 'departments' key")

        // Check that constraints change with context
        let rootSnapshot = constraintSnapshots.first { $0.context == "root" }
        let hqSnapshot = constraintSnapshots.first { $0.context == "headquarters" }

        if let root = rootSnapshot {
            #expect(root.availableKeys.contains("name"), "Root should have 'name' available")
            #expect(root.availableKeys.contains("founded"), "Root should have 'founded' available")
        }

        if let hq = hqSnapshot {
            #expect(hq.availableKeys.contains("city"), "Headquarters should have 'city' available")
            #expect(hq.availableKeys.contains("country"), "Headquarters should have 'country' available")
            #expect(!hq.availableKeys.contains("name"), "Headquarters should not have 'name' available")
        }
    }

    @Test("Progressive key constraint reduction")
    func testProgressiveConstraintReduction() {
        let schema: [String: Any] = [
            "type": "object",
            "properties": [
                "first": ["type": "string"],
                "second": ["type": "string"],
                "third": ["type": "string"],
                "fourth": ["type": "string"]
            ]
        ]

        // JSON being generated progressively
        let partialJSONs = [
            "{",
            #"{"first":"#,
            #"{"first":"a","#,
            #"{"first":"a","second":"#,
            #"{"first":"a","second":"b","#,
            #"{"first":"a","second":"b","third":"#,
            #"{"first":"a","second":"b","third":"c","#
        ]

        let detector = JSONSchemaContextDetector(schema: schema)
        var constraintHistory: [[String]] = []

        for partial in partialJSONs {
            let availableKeys = detector.getAvailableKeys(from: partial)
            constraintHistory.append(availableKeys)
            print("Partial: \(partial.suffix(20).replacingOccurrences(of: "\n", with: "\\n"))")
            print("  Available keys: \(availableKeys)")
        }

        // Verify progressive reduction
        #expect(constraintHistory[0].count == 4, "Should start with all 4 keys")
        #expect(constraintHistory[2].count == 3, "After 'first', should have 3 keys")
        #expect(constraintHistory[4].count == 2, "After 'first' and 'second', should have 2 keys")
        #expect(constraintHistory[6].count == 1, "After three keys, should have 1 key left")
    }

    @Test("Multiple JSON detection with different schemas")
    func testMultipleJSONsWithDifferentConstraints() {
        // First schema: User
        let userSchema: [String: Any] = [
            "type": "object",
            "properties": [
                "username": ["type": "string"],
                "email": ["type": "string"],
                "age": ["type": "integer"]
            ]
        ]

        // Second schema: Product
        let productSchema: [String: Any] = [
            "type": "object",
            "properties": [
                "productName": ["type": "string"],
                "price": ["type": "number"],
                "inStock": ["type": "boolean"]
            ]
        ]

        let llmOutput = """
        Creating a user: {"username": "alice", "email": "alice@example.com", "age": 25}

        Now creating a product: {"productName": "Widget", "price": 19.99, "inStock": true}
        """

        var extractor = JSONExtractor()
        var jsonCount = 0
        var currentJSON = ""
        var completedJSONs: [String] = []

        for char in llmOutput {
            let shouldProcess = extractor.processCharacter(char)

            if shouldProcess {
                if currentJSON.isEmpty && char == "{" {
                    jsonCount += 1
                    currentJSON = "{"
                } else if !currentJSON.isEmpty {
                    currentJSON.append(char)
                }

                // Check if we transitioned back to scanning (JSON ended)
                if !extractor.isInJSON && !currentJSON.isEmpty {
                    completedJSONs.append(currentJSON)
                    currentJSON = ""
                }
            }
        }

        // Add last JSON if still in buffer
        if !currentJSON.isEmpty && currentJSON.contains("}") {
            completedJSONs.append(currentJSON)
        }

        print("Detected \(completedJSONs.count) JSONs")

        #expect(completedJSONs.count == 2, "Should detect both JSONs")

        // Check constraints for each JSON
        if completedJSONs.count >= 1 {
            let userDetector = JSONSchemaContextDetector(schema: userSchema)
            let userKeys = userDetector.getAvailableKeys(from: "{")
            #expect(userKeys.sorted() == ["age", "email", "username"],
                    "User schema should have correct keys")
        }

        if completedJSONs.count >= 2 {
            let productDetector = JSONSchemaContextDetector(schema: productSchema)
            let productKeys = productDetector.getAvailableKeys(from: "{")
            #expect(productKeys.sorted() == ["inStock", "price", "productName"],
                    "Product schema should have correct keys")
        }
    }
}