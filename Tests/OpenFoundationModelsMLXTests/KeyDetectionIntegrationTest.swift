import Foundation
import Testing
import MLX
@testable import OpenFoundationModelsMLX

@Suite("KeyDetection Integration Tests")
struct KeyDetectionIntegrationTest {

    // Mock tokenizer that returns text character by character
    final class CharacterTokenizer: TokenizerAdapter, @unchecked Sendable {
        private var buffer: String = ""

        func encode(_ text: String) -> [Int32] {
            return text.map { Int32($0.asciiValue ?? 0) }
        }

        func decode(_ tokens: [Int32]) -> String {
            // For this test, return one character at a time
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
        func fingerprint() -> String { "character-tokenizer" }
    }

    @Test("Detects keys with correct context in CompanyProfile")
    func testCompanyProfileContextDetection() {
        // Setup nested schemas like CompanyProfile
        let nestedSchemas = [
            "headquarters": ["city", "country", "postalCode", "street"],
            "departments[]": ["headCount", "manager", "name", "projects", "type"],
            "departments[].manager": ["email", "firstName", "lastName", "level", "yearsExperience"],
            "departments[].projects[]": ["budget", "name", "startDate", "status", "teamSize"]
        ]

        let rootKeys = ["departments", "employeeCount", "founded", "headquarters", "name", "type"]

        let tokenizer = CharacterTokenizer()
        let processor = KeyDetectionLogitProcessor(
            tokenizer: tokenizer,
            schemaKeys: rootKeys,
            nestedSchemas: nestedSchemas,
            verbose: false,
            showProbabilities: false
        )

        // Simplified CompanyProfile JSON focusing on nested structure
        let json = """
        {
          "name": "TechCorp",
          "headquarters": {
            "city": "SF",
            "country": "USA"
          },
          "departments": [
            {
              "name": "Eng",
              "manager": {
                "firstName": "Alice",
                "email": "alice@co.com"
              },
              "projects": [
                {
                  "name": "Alpha",
                  "budget": 1000
                }
              ]
            }
          ]
        }
        """

        // Initialize processor
        processor.prompt(MLXArray.zeros([1]))

        // Process JSON character by character
        for char in json {
            let tokenId = Int32(char.asciiValue ?? 0)
            let token = MLXArray([tokenId])
            processor.didSample(token: token)
        }

        // Check detected keys
        let detectedKeys = processor.allDetectedKeys

        // Root level keys
        #expect(detectedKeys.contains("name"), "Should detect root 'name' key")
        #expect(detectedKeys.contains("headquarters"), "Should detect 'headquarters' key")
        #expect(detectedKeys.contains("departments"), "Should detect 'departments' key")

        // Headquarters object keys
        #expect(detectedKeys.contains("city"), "Should detect 'city' in headquarters")
        #expect(detectedKeys.contains("country"), "Should detect 'country' in headquarters")

        // Department array item keys
        let deptNameCount = detectedKeys.filter { $0 == "name" }.count
        #expect(deptNameCount >= 2, "Should detect 'name' at multiple levels")

        // Manager object keys
        #expect(detectedKeys.contains("firstName"), "Should detect 'firstName' in manager")
        #expect(detectedKeys.contains("email"), "Should detect 'email' in manager")

        // Project array item keys
        #expect(detectedKeys.contains("budget"), "Should detect 'budget' in projects")
    }

    @Test("Tracks context path correctly")
    func testContextPathTracking() {
        let tokenizer = CharacterTokenizer()
        let processor = KeyDetectionLogitProcessor(
            tokenizer: tokenizer,
            verbose: false
        )

        // Test nested object: {"user": {"profile": {"name": "test"}}}
        let json = #"{"user":{"profile":{"name":"test"}}}"#

        processor.prompt(MLXArray.zeros([1]))

        for char in json {
            let tokenId = Int32(char.asciiValue ?? 0)
            let token = MLXArray([tokenId])
            processor.didSample(token: token)
        }

        let detectedKeys = processor.allDetectedKeys
        #expect(detectedKeys == ["user", "profile", "name"])
    }

    @Test("Handles arrays with multiple objects")
    func testArrayWithMultipleObjects() {
        let tokenizer = CharacterTokenizer()
        let processor = KeyDetectionLogitProcessor(
            tokenizer: tokenizer,
            verbose: false
        )

        // Test array: {"items": [{"id": 1, "name": "A"}, {"id": 2, "name": "B"}]}
        let json = #"{"items":[{"id":1,"name":"A"},{"id":2,"name":"B"}]}"#

        processor.prompt(MLXArray.zeros([1]))

        for char in json {
            let tokenId = Int32(char.asciiValue ?? 0)
            let token = MLXArray([tokenId])
            processor.didSample(token: token)
        }

        let detectedKeys = processor.allDetectedKeys

        // Should detect "items" once and "id", "name" twice each
        #expect(detectedKeys.filter { $0 == "items" }.count == 1)
        #expect(detectedKeys.filter { $0 == "id" }.count == 2)
        #expect(detectedKeys.filter { $0 == "name" }.count == 2)
    }
}