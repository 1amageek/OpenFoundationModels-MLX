import Foundation
import MLX
import MLXLMCommon
@testable import OpenFoundationModelsMLX

// MARK: - Mock Tokenizer Implementations

/// Basic mock tokenizer that encodes text character by character
public final class MockTokenizer: TokenizerAdapter, @unchecked Sendable {
    private let tokenMap: [Int32: String]

    public init(tokenMap: [Int32: String] = [:]) {
        self.tokenMap = tokenMap
    }

    public func encode(_ text: String) -> [Int32] {
        return text.map { Int32($0.asciiValue ?? 0) }
    }

    public func decode(_ tokens: [Int32]) -> String {
        if let customText = tokenMap[tokens.first ?? -1] {
            return customText
        }
        return String(tokens.compactMap {
            if let scalar = UnicodeScalar(Int($0)) {
                return Character(scalar)
            }
            return nil
        })
    }

    public var eosTokenId: Int32? { 0 }
    public var bosTokenId: Int32? { 1 }
    public var unknownTokenId: Int32? { 2 }

    public func convertTokenToString(_ token: Int32) -> String? {
        if let scalar = UnicodeScalar(Int(token)) {
            return String(Character(scalar))
        }
        return nil
    }

    public func getVocabSize() -> Int? { 50000 }
    public func fingerprint() -> String { "mock-tokenizer" }
}

/// Character-based tokenizer that returns one character at a time
public final class CharacterTokenizer: TokenizerAdapter, @unchecked Sendable {
    public init() {}

    public func encode(_ text: String) -> [Int32] {
        return text.map { Int32($0.asciiValue ?? 0) }
    }

    public func decode(_ tokens: [Int32]) -> String {
        if let first = tokens.first,
           let scalar = UnicodeScalar(Int(first)) {
            return String(Character(scalar))
        }
        return ""
    }

    public var eosTokenId: Int32? { 0 }
    public var bosTokenId: Int32? { 1 }
    public var unknownTokenId: Int32? { 2 }

    public func convertTokenToString(_ token: Int32) -> String? {
        if let scalar = UnicodeScalar(Int(token)) {
            return String(Character(scalar))
        }
        return nil
    }

    public func getVocabSize() -> Int? { 50000 }
    public func fingerprint() -> String { "character-tokenizer" }
}

/// Configurable mock tokenizer for testing specific behaviors
public final class ConfigurableTokenizer: TokenizerAdapter, @unchecked Sendable {
    public var nextDecodeResult: String = ""
    public var nextChar: String = ""

    public init() {}

    public func encode(_ text: String) -> [Int32] {
        return Array(repeating: Int32(0), count: text.count)
    }

    public func decode(_ tokens: [Int32]) -> String {
        return nextDecodeResult.isEmpty ? nextChar : nextDecodeResult
    }

    public var eosTokenId: Int32? { 0 }
    public var bosTokenId: Int32? { 1 }
    public var unknownTokenId: Int32? { 2 }

    public func convertTokenToString(_ token: Int32) -> String? {
        return nextDecodeResult.isEmpty ? nextChar : nextDecodeResult
    }

    public func getVocabSize() -> Int? { 50000 }
    public func fingerprint() -> String { "configurable-tokenizer" }
}


// MARK: - Common Test Schemas

public enum TestSchemas {
    /// CompanyProfile schema used in multiple tests
    @MainActor
    public static let companyProfileSchema: [String: Any] = [
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

    /// Nested schemas for CompanyProfile
    @MainActor
    public static let companyProfileNestedSchemas = [
        "headquarters": ["city", "country", "postalCode", "street"],
        "departments[]": ["headCount", "manager", "name", "projects", "type"],
        "departments[].manager": ["email", "firstName", "lastName", "level", "yearsExperience"],
        "departments[].projects[]": ["budget", "name", "startDate", "status", "teamSize"]
    ]

    /// Root keys for CompanyProfile
    @MainActor
    public static let companyProfileRootKeys = ["departments", "employeeCount", "founded", "headquarters", "name", "type"]

    /// Simple user schema
    @MainActor
    public static let userSchema: [String: Any] = [
        "type": "object",
        "properties": [
            "name": ["type": "string"],
            "email": ["type": "string"],
            "age": ["type": "integer"]
        ]
    ]

    /// Simple product schema
    @MainActor
    public static let productSchema: [String: Any] = [
        "type": "object",
        "properties": [
            "id": ["type": "integer"],
            "name": ["type": "string"],
            "price": ["type": "number"],
            "inStock": ["type": "boolean"]
        ]
    ]
}

// MARK: - Test Helper Functions

/// Helper to create a test KeyDetectionProcessor with exposed internal state
public final class TestKeyDetectionProcessor {
    private let tokenizer: TokenizerAdapter
    private let schemaDetector: JSONSchemaContextDetector?
    private var generatedText = ""

    public init(tokenizer: TokenizerAdapter, jsonSchema: [String: Any]? = nil) {
        self.tokenizer = tokenizer
        self.schemaDetector = jsonSchema != nil ? JSONSchemaContextDetector(schema: jsonSchema!) : nil
    }

    public func simulateGeneration(_ text: String) {
        generatedText = text
    }

    public func getCurrentAvailableKeys() -> [String] {
        guard let detector = schemaDetector else { return [] }
        let partialJSON = extractPartialJSON()
        return detector.getAvailableKeys(from: partialJSON)
    }

    private func extractPartialJSON() -> String {
        if let jsonStart = generatedText.firstIndex(of: "{") {
            return String(generatedText[jsonStart...])
        }
        return ""
    }
}

// MARK: - Mock LogitProcessor

public final class MockLogitProcessor: LogitProcessor, @unchecked Sendable {
    public var promptCalled = false
    public var processCalled = false
    public var didSampleCalled = false

    public init() {}

    public func prompt(_ prompt: MLXArray) {
        promptCalled = true
    }

    public func process(logits: MLXArray) -> MLXArray {
        processCalled = true
        return logits
    }

    public func didSample(token: MLXArray) {
        didSampleCalled = true
    }
}