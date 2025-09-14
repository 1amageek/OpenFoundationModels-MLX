import Foundation
import Testing
import MLX
import MLXLMCommon
@testable import OpenFoundationModelsMLX

@Suite("KeyDetection Context Integration Tests")
struct KeyDetectionContextIntegrationTests {

    @Test("Displays correct context keys for CompanyProfile")
    func testCompanyProfileContextKeys() {
        // Create tokenizer
        let tokenizer = MockTokenizerAdapter()

        // Create CompanyProfile schema
        let schema: [String: Any] = [
            "type": "object",
            "properties": [
                "name": ["type": "string"],
                "founded": ["type": "integer"],
                "type": [
                    "type": "string",
                    "enum": ["startup", "corporation", "nonprofit"]
                ],
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
                            "type": [
                                "type": "string",
                                "enum": ["engineering", "sales", "marketing", "operations"]
                            ],
                            "headCount": ["type": "integer"],
                            "manager": [
                                "type": "object",
                                "properties": [
                                    "firstName": ["type": "string"],
                                    "lastName": ["type": "string"],
                                    "email": ["type": "string"],
                                    "level": [
                                        "type": "string",
                                        "enum": ["junior", "senior", "lead", "manager"]
                                    ],
                                    "yearsExperience": ["type": "integer"]
                                ]
                            ],
                            "projects": [
                                "type": "array",
                                "items": [
                                    "type": "object",
                                    "properties": [
                                        "name": ["type": "string"],
                                        "status": [
                                            "type": "string",
                                            "enum": ["planning", "active", "completed", "onHold"]
                                        ],
                                        "startDate": ["type": "string"],
                                        "teamSize": ["type": "integer"],
                                        "budget": ["type": ["number", "null"]]
                                    ]
                                ]
                            ]
                        ]
                    ]
                ]
            ]
        ]

        // Create processor with schema
        let processor = KeyDetectionLogitProcessor(
            tokenizer: tokenizer,
            jsonSchema: schema,
            verbose: false
        )

        // Simulate generation and collect available keys at each step
        var collectedContexts: [(json: String, keys: [String])] = []

        // Create a test version that exposes internal state
        let testProcessor = TestKeyDetectionProcessor(
            tokenizer: tokenizer,
            jsonSchema: schema
        )

        // Test at root level
        testProcessor.simulateGeneration("{")
        collectedContexts.append((
            json: "{",
            keys: testProcessor.getCurrentAvailableKeys()
        ))
        #expect(testProcessor.getCurrentAvailableKeys().sorted() ==
                ["departments", "employeeCount", "founded", "headquarters", "name", "type"])

        // Test inside headquarters
        testProcessor.simulateGeneration(#"{"name":"InnoTech","founded":2020,"headquarters":{"#)
        collectedContexts.append((
            json: #"{"name":"InnoTech","founded":2020,"headquarters":{"#,
            keys: testProcessor.getCurrentAvailableKeys()
        ))
        #expect(testProcessor.getCurrentAvailableKeys().sorted() ==
                ["city", "country", "postalCode", "street"])

        // Test inside departments array
        testProcessor.simulateGeneration(#"{"name":"InnoTech","departments":[{"#)
        collectedContexts.append((
            json: #"{"name":"InnoTech","departments":[{"#,
            keys: testProcessor.getCurrentAvailableKeys()
        ))
        #expect(testProcessor.getCurrentAvailableKeys().sorted() ==
                ["headCount", "manager", "name", "projects", "type"])

        // Test inside manager object
        testProcessor.simulateGeneration(#"{"departments":[{"name":"Engineering","manager":{"#)
        collectedContexts.append((
            json: #"{"departments":[{"name":"Engineering","manager":{"#,
            keys: testProcessor.getCurrentAvailableKeys()
        ))
        #expect(testProcessor.getCurrentAvailableKeys().sorted() ==
                ["email", "firstName", "lastName", "level", "yearsExperience"])

        // Test inside projects array
        testProcessor.simulateGeneration(#"{"departments":[{"name":"Eng","projects":[{"#)
        collectedContexts.append((
            json: #"{"departments":[{"name":"Eng","projects":[{"#,
            keys: testProcessor.getCurrentAvailableKeys()
        ))
        #expect(testProcessor.getCurrentAvailableKeys().sorted() ==
                ["budget", "name", "startDate", "status", "teamSize"])

        // Verify context changes appropriately
        #expect(collectedContexts[0].keys != collectedContexts[1].keys)
        #expect(collectedContexts[1].keys != collectedContexts[2].keys)
        #expect(collectedContexts[2].keys != collectedContexts[3].keys)
        #expect(collectedContexts[3].keys != collectedContexts[4].keys)
    }
}

// Test helper class that exposes internal state
private final class TestKeyDetectionProcessor {
    private let tokenizer: TokenizerAdapter
    private let schemaDetector: JSONSchemaContextDetector
    private var generatedText = ""

    init(tokenizer: TokenizerAdapter, jsonSchema: [String: Any]) {
        self.tokenizer = tokenizer
        self.schemaDetector = JSONSchemaContextDetector(schema: jsonSchema)
    }

    func simulateGeneration(_ text: String) {
        generatedText = text
    }

    func getCurrentAvailableKeys() -> [String] {
        let partialJSON = extractPartialJSON()
        return schemaDetector.getAvailableKeys(from: partialJSON)
    }

    private func extractPartialJSON() -> String {
        if let jsonStart = generatedText.firstIndex(of: "{") {
            return String(generatedText[jsonStart...])
        }
        return ""
    }
}