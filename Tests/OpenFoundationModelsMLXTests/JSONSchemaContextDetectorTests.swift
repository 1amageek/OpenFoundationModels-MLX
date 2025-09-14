import Foundation
import Testing
@testable import OpenFoundationModelsMLX

@Suite("JSONSchemaContextDetector Tests")
struct JSONSchemaContextDetectorTests {

    // MARK: - Test Case 1: Simple Flat Schema

    @Test("Simple flat object - empty start")
    func testSimpleFlatEmpty() {
        // Input: Schema
        let schema: [String: Any] = [
            "type": "object",
            "properties": [
                "name": ["type": "string"],
                "age": ["type": "integer"],
                "email": ["type": "string"]
            ],
            "required": ["name", "age", "email"]
        ]

        // Input: Partial JSON
        let partialJSON = "{"

        // Expected Output: Available keys
        let expectedKeys = ["age", "email", "name"]  // Sorted

        // Test
        let detector = JSONSchemaContextDetector(schema: schema)
        let actualKeys = detector.getAvailableKeys(from: partialJSON)
        #expect(actualKeys == expectedKeys)
    }

    @Test("Simple flat object - after first key-value")
    func testSimpleFlatAfterFirst() {
        // Input: Schema
        let schema: [String: Any] = [
            "type": "object",
            "properties": [
                "name": ["type": "string"],
                "age": ["type": "integer"],
                "email": ["type": "string"]
            ]
        ]

        // Input: Partial JSON
        let partialJSON = #"{"name":"John","#

        // Expected Output: Available keys (remaining keys)
        let expectedKeys = ["age", "email"]

        // Test
        let detector = JSONSchemaContextDetector(schema: schema)
        let actualKeys = detector.getAvailableKeys(from: partialJSON)
        #expect(actualKeys == expectedKeys)
    }

    @Test("Simple flat object - inside string value")
    func testSimpleFlatInsideValue() {
        // Input: Schema
        let schema: [String: Any] = [
            "type": "object",
            "properties": [
                "name": ["type": "string"],
                "age": ["type": "integer"]
            ]
        ]

        // Input: Partial JSON (cursor is inside string value)
        let partialJSON = #"{"name":"Joh"#

        // Expected Output: No keys available when inside a value
        let expectedKeys: [String] = []

        // Test
        let detector = JSONSchemaContextDetector(schema: schema)
        let actualKeys = detector.getAvailableKeys(from: partialJSON)
        #expect(actualKeys == expectedKeys)
    }

    // MARK: - Test Case 2: Single Nested Object

    @Test("Nested object - at root level")
    func testNestedObjectRoot() {
        // Input: Schema
        let schema: [String: Any] = [
            "type": "object",
            "properties": [
                "user": [
                    "type": "object",
                    "properties": [
                        "firstName": ["type": "string"],
                        "lastName": ["type": "string"]
                    ]
                ],
                "timestamp": ["type": "string"]
            ]
        ]

        // Input: Partial JSON
        let partialJSON = "{"

        // Expected Output: Root level keys
        let expectedKeys = ["timestamp", "user"]  // Sorted

        // Test
        let detector = JSONSchemaContextDetector(schema: schema)
        let actualKeys = detector.getAvailableKeys(from: partialJSON)
        #expect(actualKeys == expectedKeys)
    }

    @Test("Nested object - inside nested object")
    func testNestedObjectInside() {
        // Input: Schema
        let schema: [String: Any] = [
            "type": "object",
            "properties": [
                "user": [
                    "type": "object",
                    "properties": [
                        "firstName": ["type": "string"],
                        "lastName": ["type": "string"],
                        "email": ["type": "string"]
                    ]
                ],
                "timestamp": ["type": "string"]
            ]
        ]

        // Input: Partial JSON
        let partialJSON = #"{"user":{"#

        // Expected Output: Keys from user object
        let expectedKeys = ["email", "firstName", "lastName"]  // Sorted

        // Test
        let detector = JSONSchemaContextDetector(schema: schema)
        let actualKeys = detector.getAvailableKeys(from: partialJSON)
        #expect(actualKeys == expectedKeys)
    }

    @Test("Nested object - back to root after nested")
    func testNestedObjectBackToRoot() {
        // Input: Schema
        let schema: [String: Any] = [
            "type": "object",
            "properties": [
                "user": [
                    "type": "object",
                    "properties": [
                        "firstName": ["type": "string"],
                        "lastName": ["type": "string"]
                    ]
                ],
                "timestamp": ["type": "string"]
            ]
        ]

        // Input: Partial JSON
        let partialJSON = #"{"user":{"firstName":"Alice","lastName":"Smith"},"#

        // Expected Output: Remaining root keys
        let expectedKeys = ["timestamp"]

        // Test
        let detector = JSONSchemaContextDetector(schema: schema)
        let actualKeys = detector.getAvailableKeys(from: partialJSON)
        #expect(actualKeys == expectedKeys)
    }

    // MARK: - Test Case 3: Array with Objects

    @Test("Array - inside array item object")
    func testArrayItemObject() {
        // Input: Schema
        let schema: [String: Any] = [
            "type": "object",
            "properties": [
                "items": [
                    "type": "array",
                    "items": [
                        "type": "object",
                        "properties": [
                            "id": ["type": "integer"],
                            "name": ["type": "string"],
                            "price": ["type": "number"]
                        ]
                    ]
                ]
            ]
        ]

        // Input: Partial JSON
        let partialJSON = #"{"items":[{"#

        // Expected Output: Keys for array item object
        let expectedKeys = ["id", "name", "price"]

        // Test
        let detector = JSONSchemaContextDetector(schema: schema)
        let actualKeys = detector.getAvailableKeys(from: partialJSON)
        #expect(actualKeys == expectedKeys)
    }

    @Test("Array - second array item")
    func testArraySecondItem() {
        // Input: Schema
        let schema: [String: Any] = [
            "type": "object",
            "properties": [
                "items": [
                    "type": "array",
                    "items": [
                        "type": "object",
                        "properties": [
                            "id": ["type": "integer"],
                            "name": ["type": "string"]
                        ]
                    ]
                ]
            ]
        ]

        // Input: Partial JSON
        let partialJSON = #"{"items":[{"id":1,"name":"First"},{"#

        // Expected Output: Same keys for second array item
        let expectedKeys = ["id", "name"]

        // Test
        let detector = JSONSchemaContextDetector(schema: schema)
        let actualKeys = detector.getAvailableKeys(from: partialJSON)
        #expect(actualKeys == expectedKeys)
    }

    // MARK: - Test Case 4: CompanyProfile Complex Cases

    @Test("CompanyProfile - root level")
    func testCompanyProfileRoot() {
        // Input: Schema
        let schema = companyProfileSchema()

        // Input: Partial JSON
        let partialJSON = "{"

        // Expected Output: Root keys
        let expectedKeys = ["departments", "employeeCount", "founded", "headquarters", "name", "type"]  // Sorted

        // Test
        let detector = JSONSchemaContextDetector(schema: schema)
        let actualKeys = detector.getAvailableKeys(from: partialJSON)
        #expect(actualKeys == expectedKeys)
    }

    @Test("CompanyProfile - inside headquarters")
    func testCompanyProfileHeadquarters() {
        // Input: Schema
        let schema = companyProfileSchema()

        // Input: Partial JSON
        let partialJSON = #"{"name":"InnoTech","founded":2020,"headquarters":{"#

        // Expected Output: Headquarters keys
        let expectedKeys = ["city", "country", "postalCode", "street"]  // Sorted

        // Test
        let detector = JSONSchemaContextDetector(schema: schema)
        let actualKeys = detector.getAvailableKeys(from: partialJSON)
        #expect(actualKeys == expectedKeys)
    }

    @Test("CompanyProfile - inside departments array item")
    func testCompanyProfileDepartment() {
        // Input: Schema
        let schema = companyProfileSchema()

        // Input: Partial JSON
        let partialJSON = #"{"name":"InnoTech","departments":[{"#

        // Expected Output: Department keys
        let expectedKeys = ["headCount", "manager", "name", "projects", "type"]  // Sorted

        // Test
        let detector = JSONSchemaContextDetector(schema: schema)
        let actualKeys = detector.getAvailableKeys(from: partialJSON)
        #expect(actualKeys == expectedKeys)
    }

    @Test("CompanyProfile - inside department manager")
    func testCompanyProfileManager() {
        // Input: Schema
        let schema = companyProfileSchema()

        // Input: Partial JSON
        let partialJSON = #"{"departments":[{"name":"Engineering","manager":{"#

        // Expected Output: Manager keys
        let expectedKeys = ["email", "firstName", "lastName", "level", "yearsExperience"]  // Sorted

        // Test
        let detector = JSONSchemaContextDetector(schema: schema)
        let actualKeys = detector.getAvailableKeys(from: partialJSON)
        #expect(actualKeys == expectedKeys)
    }

    @Test("CompanyProfile - inside projects array item")
    func testCompanyProfileProject() {
        // Input: Schema
        let schema = companyProfileSchema()

        // Input: Partial JSON
        let partialJSON = #"{"departments":[{"name":"Eng","projects":[{"#

        // Expected Output: Project keys
        let expectedKeys = ["budget", "name", "startDate", "status", "teamSize"]  // Sorted

        // Test
        let detector = JSONSchemaContextDetector(schema: schema)
        let actualKeys = detector.getAvailableKeys(from: partialJSON)
        #expect(actualKeys == expectedKeys)
    }

    @Test("CompanyProfile - complex nested with partial keys")
    func testCompanyProfileComplexPartial() {
        // Input: Schema
        let schema = companyProfileSchema()

        // Input: Partial JSON (inside manager, some keys already used)
        let partialJSON = #"{"departments":[{"name":"Engineering","type":"engineering","headCount":30,"manager":{"firstName":"Alice","lastName":"Smith","#

        // Expected Output: Remaining manager keys
        let expectedKeys = ["email", "level", "yearsExperience"]

        // Test
        let detector = JSONSchemaContextDetector(schema: schema)
        let actualKeys = detector.getAvailableKeys(from: partialJSON)
        #expect(actualKeys == expectedKeys)
    }

    // MARK: - Test Case 5: Edge Cases

    @Test("Edge case - empty JSON")
    func testEdgeCaseEmpty() {
        // Input: Schema
        let schema: [String: Any] = [
            "type": "object",
            "properties": [
                "key": ["type": "string"]
            ]
        ]

        // Input: Partial JSON
        let partialJSON = ""

        // Expected Output: Empty (invalid JSON)
        let expectedKeys: [String] = []

        // Test
        let detector = JSONSchemaContextDetector(schema: schema)
        let actualKeys = detector.getAvailableKeys(from: partialJSON)
        #expect(actualKeys == expectedKeys)
    }

    @Test("Edge case - inside key name")
    func testEdgeCaseInsideKey() {
        // Input: Schema
        let schema: [String: Any] = [
            "type": "object",
            "properties": [
                "firstName": ["type": "string"],
                "lastName": ["type": "string"]
            ]
        ]

        // Input: Partial JSON (cursor is inside key name)
        let partialJSON = #"{"first"#

        // Expected Output: No keys (currently typing a key)
        let expectedKeys: [String] = []

        // Test
        let detector = JSONSchemaContextDetector(schema: schema)
        let actualKeys = detector.getAvailableKeys(from: partialJSON)
        #expect(actualKeys == expectedKeys)
    }

    @Test("Edge case - after colon before value")
    func testEdgeCaseAfterColon() {
        // Input: Schema
        let schema: [String: Any] = [
            "type": "object",
            "properties": [
                "name": ["type": "string"],
                "age": ["type": "integer"]
            ]
        ]

        // Input: Partial JSON (after colon, before value)
        let partialJSON = #"{"name":"#

        // Expected Output: No keys (expecting a value)
        let expectedKeys: [String] = []

        // Test
        let detector = JSONSchemaContextDetector(schema: schema)
        let actualKeys = detector.getAvailableKeys(from: partialJSON)
        #expect(actualKeys == expectedKeys)
    }

    @Test("Edge case - incomplete number value")
    func testEdgeCaseIncompleteNumber() {
        // Input: Schema
        let schema: [String: Any] = [
            "type": "object",
            "properties": [
                "age": ["type": "integer"],
                "score": ["type": "number"]
            ]
        ]

        // Input: Partial JSON
        let partialJSON = #"{"age":25,"#

        // Expected Output: Remaining keys
        let expectedKeys = ["score"]

        // Test
        let detector = JSONSchemaContextDetector(schema: schema)
        let actualKeys = detector.getAvailableKeys(from: partialJSON)
        #expect(actualKeys == expectedKeys)
    }

    // MARK: - Helper Functions

    private func companyProfileSchema() -> [String: Any] {
        return [
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
    }
}