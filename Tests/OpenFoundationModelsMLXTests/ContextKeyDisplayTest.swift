import Foundation
import Testing
import MLX
@testable import OpenFoundationModelsMLX

@Suite("Context Key Display Tests")
struct ContextKeyDisplayTest {

    final class MockTokenizer: TokenizerAdapter, @unchecked Sendable {
        func encode(_ text: String) -> [Int32] {
            return text.map { Int32($0.asciiValue ?? 0) }
        }

        func decode(_ tokens: [Int32]) -> String {
            if let first = tokens.first,
               let scalar = UnicodeScalar(Int(first)) {
                return String(Character(scalar))
            }
            return ""
        }

        var eosTokenId: Int32 { 0 }
        var bosTokenId: Int32 { 1 }
        var unknownTokenId: Int32 { 2 }

        func convertTokenToString(_ token: Int32) -> String? {
            if let scalar = UnicodeScalar(Int(token)) {
                return String(Character(scalar))
            }
            return nil
        }

        func getVocabSize() -> Int? { 50000 }
        func fingerprint() -> String { "mock-tokenizer" }
    }

    @Test("Shows correct context keys for nested structures")
    func testNestedContextKeys() {
        // Setup CompanyProfile-like schema
        let nestedSchemas = [
            "headquarters": ["city", "country", "postalCode", "street"],
            "departments[]": ["headCount", "manager", "name", "projects", "type"],
            "departments[].manager": ["email", "firstName", "lastName", "level", "yearsExperience"],
            "departments[].projects[]": ["budget", "name", "startDate", "status", "teamSize"]
        ]

        let rootKeys = ["departments", "employeeCount", "founded", "headquarters", "name", "type"]

        let tokenizer = MockTokenizer()
        let processor = KeyDetectionLogitProcessor(
            tokenizer: tokenizer,
            schemaKeys: rootKeys,
            nestedSchemas: nestedSchemas,
            verbose: true,  // Enable verbose to see context keys
            showProbabilities: true
        )

        // Test JSON that navigates through different contexts
        let testCases: [(json: String, expectedContext: String, expectedKeys: [String])] = [
            // Root level
            (#"{"#, "", rootKeys),

            // After "headquarters" key
            (#"{"headquarters":{"#, "headquarters", ["city", "country", "postalCode", "street"]),

            // After "departments" array
            (#"{"departments":[{"#, "departments[]", ["headCount", "manager", "name", "projects", "type"]),

            // Inside manager object within departments
            (#"{"departments":[{"manager":{"#, "departments[].manager", ["email", "firstName", "lastName", "level", "yearsExperience"]),

            // Inside projects array within departments
            (#"{"departments":[{"projects":[{"#, "departments[].projects[]", ["budget", "name", "startDate", "status", "teamSize"])
        ]

        for testCase in testCases {
            print("\n=== Testing JSON: \(testCase.json) ===")
            print("Expected context: '\(testCase.expectedContext)'")
            print("Expected keys: \(testCase.expectedKeys.joined(separator: ", "))")

            // Reset processor
            processor.prompt(MLXArray.zeros([1]))

            // Process JSON character by character
            for char in testCase.json {
                let tokenId = Int32(char.asciiValue ?? 0)
                let token = MLXArray([tokenId])

                // Process logits (this would display context keys)
                let dummyLogits = MLXArray.zeros([1, 50000])
                _ = processor.process(logits: dummyLogits)

                // Sample the token
                processor.didSample(token: token)
            }

            print("=== End test case ===\n")
        }
    }

    @Test("Updates context correctly when entering and exiting")
    func testContextTransitions() {
        let nestedSchemas = [
            "user": ["name", "email"],
            "settings": ["theme", "notifications"]
        ]

        let rootKeys = ["user", "settings", "version"]

        let tokenizer = MockTokenizer()
        let processor = KeyDetectionLogitProcessor(
            tokenizer: tokenizer,
            schemaKeys: rootKeys,
            nestedSchemas: nestedSchemas,
            verbose: false,
            showProbabilities: false
        )

        // JSON that enters and exits contexts
        let json = #"{"user":{"name":"Alice","email":"alice@example.com"},"settings":{"theme":"dark"}}"#

        processor.prompt(MLXArray.zeros([1]))

        var detectedKeys: [String] = []
        for char in json {
            let tokenId = Int32(char.asciiValue ?? 0)
            let token = MLXArray([tokenId])
            processor.didSample(token: token)
        }

        // Check that all keys were detected
        detectedKeys = processor.allDetectedKeys
        #expect(detectedKeys.contains("user"))
        #expect(detectedKeys.contains("name"))
        #expect(detectedKeys.contains("email"))
        #expect(detectedKeys.contains("settings"))
        #expect(detectedKeys.contains("theme"))
    }
}