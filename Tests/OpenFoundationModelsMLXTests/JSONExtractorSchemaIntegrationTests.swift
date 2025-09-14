import Foundation
import Testing
@testable import OpenFoundationModelsMLX

@Suite("JSONExtractor + Schema Integration Tests")
struct JSONExtractorSchemaIntegrationTests {

    // MARK: - Test LLM output with multiple JSONs and schema constraints

    @Test("Detects constraints in LLM output with single JSON")
    func testSingleJSONWithConstraints() {
        // Schema definition
        let schema: [String: Any] = [
            "type": "object",
            "properties": [
                "name": ["type": "string"],
                "age": ["type": "integer"],
                "email": ["type": "string"]
            ]
        ]

        let llmOutput = """
        Let me create a user profile for you.

        Here's the JSON data:
        {"name": "Alice", "age": 30
        """

        var extractor = JSONExtractor()
        let detector = JSONSchemaContextDetector(schema: schema)
        var partialJSON = ""
        var constraintsHistory: [(position: String, constraints: [String])] = []

        for char in llmOutput {
            let shouldProcess = extractor.processCharacter(char)

            if shouldProcess {
                partialJSON.append(char)

                // Get available keys at this point
                let availableKeys = detector.getAvailableKeys(from: partialJSON)

                // Record constraints at key positions
                if char == "{" || char == "," || char == "\"" {
                    constraintsHistory.append((
                        position: String(partialJSON.suffix(10)),
                        constraints: availableKeys
                    ))
                }
            } else if extractor.isInJSON && !shouldProcess {
                // JSON ended, reset for next one
                partialJSON = ""
            }
        }

        // Verify constraints were detected
        #expect(constraintsHistory.count > 0, "Should have detected constraint positions")

        // Check that after opening brace, all keys are available
        if let firstConstraint = constraintsHistory.first(where: { $0.position.contains("{") }) {
            #expect(firstConstraint.constraints.sorted() == ["age", "email", "name"],
                    "Should have all keys available at start")
        }

        // Check that after "name" and "age", only "email" remains
        let lastConstraints = constraintsHistory.last?.constraints ?? []
        #expect(lastConstraints == ["email"], "Should only have 'email' left after name and age")
    }

    @Test("Detects constraints in multiple JSONs")
    func testMultipleJSONsWithConstraints() {
        // Schema for first JSON type
        let userSchema: [String: Any] = [
            "type": "object",
            "properties": [
                "userId": ["type": "string"],
                "username": ["type": "string"],
                "active": ["type": "boolean"]
            ]
        ]

        // Schema for second JSON type
        let productSchema: [String: Any] = [
            "type": "object",
            "properties": [
                "productId": ["type": "integer"],
                "name": ["type": "string"],
                "price": ["type": "number"],
                "inStock": ["type": "boolean"]
            ]
        ]

        let llmOutput = """
        First, let's create a user:
        {"userId": "u123", "username": "alice", "active": true}

        Now for the product:
        {"productId": 456, "name": "Widget", "price": 19.99, "inStock": false}
        """

        var extractor = JSONExtractor()
        var currentJSON = ""
        var allJSONs: [(json: String, isComplete: Bool)] = []

        for char in llmOutput {
            let shouldProcess = extractor.processCharacter(char)

            if shouldProcess {
                if currentJSON.isEmpty && (char == "{" || char == "[") {
                    // Starting new JSON
                    currentJSON = String(char)
                } else {
                    currentJSON.append(char)
                }

                // Check if JSON might be complete
                if (char == "}" || char == "]") && !currentJSON.isEmpty {
                    allJSONs.append((json: currentJSON, isComplete: true))
                    currentJSON = ""
                }
            }
        }

        #expect(allJSONs.count == 2, "Should detect both JSON objects")

        // Test constraints for first JSON (user)
        let userDetector = JSONSchemaContextDetector(schema: userSchema)
        let userJSON = allJSONs[0].json
        let userConstraints = userDetector.getAvailableKeys(from: "{")
        #expect(userConstraints.sorted() == ["active", "userId", "username"],
                "Should get user schema constraints")

        // Test constraints for second JSON (product)
        let productDetector = JSONSchemaContextDetector(schema: productSchema)
        let productJSON = allJSONs[1].json
        let productConstraints = productDetector.getAvailableKeys(from: "{")
        #expect(productConstraints.sorted() == ["inStock", "name", "price", "productId"],
                "Should get product schema constraints")
    }

    @Test("Complex nested JSON with hierarchical constraints")
    func testNestedJSONWithHierarchicalConstraints() {
        // Complex schema with nesting
        let companySchema: [String: Any] = [
            "type": "object",
            "properties": [
                "company": ["type": "string"],
                "founded": ["type": "integer"],
                "headquarters": [
                    "type": "object",
                    "properties": [
                        "city": ["type": "string"],
                        "country": ["type": "string"],
                        "address": ["type": "string"]
                    ]
                ],
                "departments": [
                    "type": "array",
                    "items": [
                        "type": "object",
                        "properties": [
                            "name": ["type": "string"],
                            "manager": ["type": "string"],
                            "employees": ["type": "integer"]
                        ]
                    ]
                ]
            ]
        ]

        let llmOutput = """
        <|channel|>analysis<|message|>
        Let me create a company profile with all the required fields.
        The structure needs headquarters and departments.
        <|end|>
        <|channel|>final<|message|>
        {
          "company": "TechCorp",
          "founded": 2020,
          "headquarters": {
            "city": "San Francisco",
            "country": "USA"
        """

        var extractor = JSONExtractor()
        let detector = JSONSchemaContextDetector(schema: companySchema)
        var partialJSON = ""
        var constraintSnapshots: [(context: String, constraints: [String])] = []

        for char in llmOutput {
            let shouldProcess = extractor.processCharacter(char)

            if shouldProcess {
                partialJSON.append(char)

                // Capture constraints at specific points
                if partialJSON.contains("headquarters") && partialJSON.last == "{" {
                    let constraints = detector.getAvailableKeys(from: partialJSON)
                    constraintSnapshots.append((
                        context: "headquarters_start",
                        constraints: constraints
                    ))
                }

                if partialJSON.contains("\"city\"") && partialJSON.contains("\"country\"") {
                    let constraints = detector.getAvailableKeys(from: partialJSON)
                    constraintSnapshots.append((
                        context: "after_city_country",
                        constraints: constraints
                    ))
                }
            }
        }

        // Verify nested constraints
        if let hqStart = constraintSnapshots.first(where: { $0.context == "headquarters_start" }) {
            #expect(hqStart.constraints.sorted() == ["address", "city", "country"],
                    "Should have headquarters properties available")
        }

        if let afterTwo = constraintSnapshots.first(where: { $0.context == "after_city_country" }) {
            #expect(afterTwo.constraints == ["address"],
                    "Should only have 'address' left after city and country")
        }
    }

    @Test("GPT-OSS format with thinking and final JSON")
    func testGPTOSSFormatWithConstraints() {
        let schema: [String: Any] = [
            "type": "object",
            "properties": [
                "status": ["type": "string"],
                "result": ["type": "object"],
                "timestamp": ["type": "string"]
            ]
        ]

        let llmOutput = """
        <|channel|>analysis<|message|>
        I need to analyze this request and provide a structured response.
        First, let me check the status: {"status": "analyzing"}

        Now processing the main result.
        <|end|>
        <|channel|>final<|message|>
        {"status": "success", "result": {
        """

        var extractor = JSONExtractor()
        let detector = JSONSchemaContextDetector(schema: schema)
        var jsonCount = 0
        var currentJSON = ""
        var allConstraints: [[String]] = []

        for char in llmOutput {
            let shouldProcess = extractor.processCharacter(char)

            if shouldProcess {
                if currentJSON.isEmpty && char == "{" {
                    jsonCount += 1
                    currentJSON = String(char)

                    // Get constraints for new JSON
                    let constraints = detector.getAvailableKeys(from: currentJSON)
                    allConstraints.append(constraints)
                } else {
                    currentJSON.append(char)
                }

                // Reset on JSON end (simplified)
                if char == "}" && currentJSON.count > 1 {
                    currentJSON = ""
                }
            }
        }

        #expect(jsonCount >= 2, "Should detect at least 2 JSON starts")
        #expect(allConstraints.allSatisfy { $0.contains("status") },
                "All JSONs should have 'status' in constraints")
    }

    @Test("Streaming tokens with progressive constraint detection")
    func testStreamingWithProgressiveConstraints() {
        let schema: [String: Any] = [
            "type": "object",
            "properties": [
                "action": ["type": "string"],
                "target": ["type": "string"],
                "value": ["type": "number"]
            ]
        ]

        // Simulate token-by-token generation
        let tokens = [
            "Here's ",
            "the ",
            "command: ",
            "{\"",
            "action",
            "\":\"",
            "update",
            "\",\"",
            "target",
            "\":\"",
            "database",
            "\",\"",
            "value",
            "\":",
            "42",
            "}"
        ]

        var extractor = JSONExtractor()
        let detector = JSONSchemaContextDetector(schema: schema)
        var partialJSON = ""
        var constraintEvolution: [(stage: String, constraints: [String])] = []

        for token in tokens {
            for char in token {
                let shouldProcess = extractor.processCharacter(char)

                if shouldProcess {
                    partialJSON.append(char)

                    // Track constraint evolution at key points
                    if partialJSON == "{" {
                        constraintEvolution.append((
                            stage: "start",
                            constraints: detector.getAvailableKeys(from: partialJSON)
                        ))
                    } else if partialJSON.contains("\"action\"") &&
                             !partialJSON.contains("\"target\"") &&
                             char == "," {
                        constraintEvolution.append((
                            stage: "after_action",
                            constraints: detector.getAvailableKeys(from: partialJSON)
                        ))
                    } else if partialJSON.contains("\"target\"") &&
                             !partialJSON.contains("\"value\"") &&
                             char == "," {
                        constraintEvolution.append((
                            stage: "after_target",
                            constraints: detector.getAvailableKeys(from: partialJSON)
                        ))
                    }
                }
            }
        }

        // Verify constraint evolution
        #expect(constraintEvolution.count >= 2, "Should track multiple stages")

        if let start = constraintEvolution.first(where: { $0.stage == "start" }) {
            #expect(start.constraints.sorted() == ["action", "target", "value"],
                    "Should have all keys at start")
        }

        if let afterAction = constraintEvolution.first(where: { $0.stage == "after_action" }) {
            #expect(afterAction.constraints.sorted() == ["target", "value"],
                    "Should have remaining keys after action")
        }

        if let afterTarget = constraintEvolution.first(where: { $0.stage == "after_target" }) {
            #expect(afterTarget.constraints == ["value"],
                    "Should only have value left")
        }
    }

    @Test("Multiple JSON types in markdown blocks")
    func testMarkdownJSONsWithDifferentSchemas() {
        let configSchema: [String: Any] = [
            "type": "object",
            "properties": [
                "debug": ["type": "boolean"],
                "timeout": ["type": "integer"],
                "endpoint": ["type": "string"]
            ]
        ]

        let llmOutput = """
        ## Configuration

        Here's the default config:
        ```json
        {
          "debug": false,
          "timeout": 30
        }
        ```

        And here's the production config:
        ```json
        {
          "debug": false,
          "timeout": 60,
          "endpoint": "https://api.example.com"
        }
        ```
        """

        var extractor = JSONExtractor()
        let detector = JSONSchemaContextDetector(schema: configSchema)
        var jsonBlocks: [String] = []
        var currentJSON = ""

        for char in llmOutput {
            let shouldProcess = extractor.processCharacter(char)

            if shouldProcess {
                if currentJSON.isEmpty && char == "{" {
                    currentJSON = String(char)
                } else {
                    currentJSON.append(char)
                }

                if char == "}" && !currentJSON.isEmpty {
                    jsonBlocks.append(currentJSON)

                    // Check constraints for incomplete JSON
                    let partial = String(currentJSON.dropLast()) // Remove closing }
                    let constraints = detector.getAvailableKeys(from: partial)

                    // First JSON missing "endpoint"
                    if jsonBlocks.count == 1 {
                        #expect(constraints.contains("endpoint"),
                                "First JSON should be missing endpoint")
                    }

                    // Second JSON has all fields
                    if jsonBlocks.count == 2 {
                        #expect(constraints.isEmpty,
                                "Second JSON should have all fields")
                    }

                    currentJSON = ""
                }
            }
        }

        #expect(jsonBlocks.count == 2, "Should detect both JSON blocks")
    }
}