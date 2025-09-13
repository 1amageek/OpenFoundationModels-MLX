import Testing
import Foundation
@testable import OpenFoundationModelsMLX

@Suite("JSONExtractor Tests")
struct JSONExtractorTests {
    
    @Test("Plain JSON extraction")
    func plainJSONExtraction() {
        let input = #"{"name": "John", "age": 30}"#
        let expectedJSON = #"{"name": "John", "age": 30}"#
        
        testWithJSONExtractor(input: input, expectedJSON: expectedJSON)
    }
    
    @Test("GPT OSS format extraction")
    func gptOSSFormatExtraction() {
        let input = #"<|channel|>analysis<|message|>We need to analyze...{"name": "Alice", "city": "NYC"}"#
        let expectedJSON = #"{"name": "Alice", "city": "NYC"}"#
        
        testWithJSONExtractor(input: input, expectedJSON: expectedJSON)
    }
    
    @Test("Markdown code block extraction")
    func markdownCodeBlockExtraction() {
        let input = """
        Here's the JSON:
        ```json
        {"user": {"id": 123, "email": "test@example.com"}}
        ```
        """
        let expectedJSON = #"{"user": {"id": 123, "email": "test@example.com"}}"#
        
        testWithJSONExtractor(input: input, expectedJSON: expectedJSON)
    }
    
    @Test("Multiple JSONs extraction")
    func multipleJSONsExtraction() {
        let input = #"First: {"a": 1} and second: {"b": 2}"#
        let expectedJSON = #"{"a": 1}"#  // Should find first one
        
        testWithJSONExtractor(input: input, expectedJSON: expectedJSON)
    }
    
    @Test("Array JSON extraction")
    func arrayJSONExtraction() {
        let input = #"Some text before [{"id": 1}, {"id": 2}] and after"#
        let expectedJSON = #"[{"id": 1}, {"id": 2}]"#
        
        testWithJSONExtractor(input: input, expectedJSON: expectedJSON)
    }
    
    @Test("Complex GPT OSS format")
    func complexGPTOSSFormat() {
        let input = """
        <|channel|>analysis<|message|>Let me analyze this request. We need to create a JSON response.
        The structure should be clear and well-formatted.
        <|end|><|start|>assistant<|channel|>final<|message|>```json
        {
          "company": "TechCorp",
          "departments": [
            {"name": "Engineering", "size": 50},
            {"name": "Sales", "size": 20}
          ]
        }
        ```
        """
        let expectedJSON = #"{"company": "TechCorp", "departments": [{"name": "Engineering", "size": 50}, {"name": "Sales", "size": 20}]}"#
        
        testWithJSONExtractor(input: input, expectedJSON: expectedJSON)
    }
    
    @Test("Token-by-token integration")
    func tokenByTokenIntegration() {
        // Simulate GPT OSS token-by-token generation
        let tokens = [
            "<|channel|>",
            "analysis",
            "<|message|>",
            "Let me ",
            "think about ",
            "this...",
            "\n\n",
            "{\"",
            "name",
            "\":\"",
            "John",
            "\",\"",
            "age",
            "\":",
            "30",
            "}"
        ]
        
        var extractor = JSONExtractor()
        var stateMachine = JSONStateMachine()
        var jsonBuffer = ""
        var detectedKeys: [String] = []
        
        for token in tokens {
            for char in token {
                let shouldProcess = extractor.processCharacter(char)
                
                if shouldProcess {
                    jsonBuffer.append(char)
                    
                    let previousPhase = stateMachine.phase
                    stateMachine.processCharacter(char)
                    
                    // Detect keys
                    if case .inString(.body(kind: .key, escaped: false)) = previousPhase {
                        if case .inObject(.expectColon) = stateMachine.phase {
                            let key = stateMachine.currentKey
                            if !key.isEmpty && !detectedKeys.contains(key) {
                                detectedKeys.append(key)
                            }
                        }
                    }
                }
            }
        }
        
        #expect(jsonBuffer == #"{"name":"John","age":30}"#)
        #expect(detectedKeys.contains("name"))
        #expect(detectedKeys.contains("age"))
        #expect(stateMachine.phase == .done)
    }
    
    @Test("Incomplete JSON streaming")
    func incompleteJSONStreaming() {
        // Test various incomplete JSON scenarios
        let incompleteJSONs = [
            "{\"name\":\"John\",\"age\":", // Incomplete value
            "[{\"id\":1},{\"id\":", // Incomplete array
            "{\"nested\":{\"key\"", // Incomplete nested object
            "{\"str\":\"hello", // Incomplete string
            "{\"num\":12.34e-", // Incomplete scientific notation
        ]
        
        for incomplete in incompleteJSONs {
            var extractor = JSONExtractor()
            var stateMachine = JSONStateMachine()
            var jsonStarted = false
            
            for char in incomplete {
                let shouldProcess = extractor.processCharacter(char)
                if shouldProcess && !jsonStarted {
                    jsonStarted = true
                }
                
                if shouldProcess {
                    stateMachine.processCharacter(char)
                }
            }
            
            // Should have started JSON processing
            #expect(jsonStarted, "JSON processing should have started for: \(incomplete)")
            
            // Should not be in error state for incomplete JSON
            #expect(stateMachine.phase != .error, "Should not be in error state for incomplete JSON: \(incomplete)")
            
            // Should not be complete
            #expect(!stateMachine.isComplete, "Should not be complete for incomplete JSON: \(incomplete)")
        }
    }
    
    // Helper function
    private func testWithJSONExtractor(input: String, expectedJSON: String) {
        var extractor = JSONExtractor()
        var stateMachine = JSONStateMachine()
        var jsonBuffer = ""
        var detectedKeys: [String] = []
        
        for char in input {
            let shouldProcess = extractor.processCharacter(char)
            
            if shouldProcess {
                jsonBuffer.append(char)
                
                // Process with JSONStateMachine
                let previousPhase = stateMachine.phase
                stateMachine.processCharacter(char)
                
                // Check for key detection
                if case .inString(.body(kind: .key, escaped: false)) = previousPhase {
                    if case .inObject(.expectColon) = stateMachine.phase {
                        let key = stateMachine.currentKey
                        if !key.isEmpty && !detectedKeys.contains(key) {
                            detectedKeys.append(key)
                        }
                    }
                }
            }
        }
        
        // Verify extraction
        let normalizedBuffer = jsonBuffer.trimmingCharacters(in: .whitespacesAndNewlines)
            .replacingOccurrences(of: " ", with: "")
            .replacingOccurrences(of: "\n", with: "")
        let normalizedExpected = expectedJSON.replacingOccurrences(of: " ", with: "")
        
        #expect(normalizedBuffer.contains(normalizedExpected), "Expected JSON not found in buffer. Buffer: \(jsonBuffer), Expected: \(expectedJSON)")
    }
}