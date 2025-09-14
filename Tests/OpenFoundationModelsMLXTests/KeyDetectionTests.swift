import Testing
import Foundation
import MLX
import MLXLMCommon
@testable import OpenFoundationModelsMLX

/// Consolidated tests for key detection functionality
/// Combines unit tests, integration tests, and context-aware tests
@Suite("Key Detection Tests")
struct KeyDetectionTests {
    
    @Test("Detect simple object keys")
    func detectSimpleKeys() {
        let tokenizer = MockTokenizer()
        let modelCard = MockModelCard()
        let processor = KeyDetectionLogitProcessor(tokenizer: tokenizer, modelCard: modelCard, verbose: false)
        
        // Initialize processor
        let prompt = MLX.zeros([1, 10])
        processor.prompt(prompt)
        
        // Simulate generating: {"name":"John","age":30}
        let tokens: [(Int32, String)] = [
            (123, "{"),     // {
            (34, "\""),     // "
            (110, "n"),     // n
            (97, "a"),      // a
            (109, "m"),     // m
            (101, "e"),     // e
            (34, "\""),     // "
            (58, ":"),      // :
            (34, "\""),     // "
            (74, "J"),      // J
            (111, "o"),     // o
            (104, "h"),     // h
            (110, "n"),     // n
            (34, "\""),     // "
            (44, ","),      // ,
            (34, "\""),     // "
            (97, "a"),      // a
            (103, "g"),     // g
            (101, "e"),     // e
            (34, "\""),     // "
            (58, ":"),      // :
            (51, "3"),      // 3
            (48, "0"),      // 0
            (125, "}")      // }
        ]
        
        // Process each token
        for (tokenId, _) in tokens {
            let token = MLXArray([tokenId])
            processor.didSample(token: token)
        }
        
        // Check detected keys
        let detectedKeys = processor.allDetectedKeys
        #expect(detectedKeys.contains("name"))
        #expect(detectedKeys.contains("age"))
        #expect(detectedKeys.count == 2)
    }
    
    @Test("Detect nested object keys")
    func detectNestedKeys() {
        let tokenizer = MockTokenizer()
        let modelCard = MockModelCard()
        let processor = KeyDetectionLogitProcessor(tokenizer: tokenizer, modelCard: modelCard, verbose: false)
        
        // Initialize processor
        let prompt = MLX.zeros([1, 10])
        processor.prompt(prompt)
        
        // Simulate generating: {"user":{"name":"Alice","email":"alice@example.com"}}
        let jsonString = "{\"user\":{\"name\":\"Alice\",\"email\":\"alice@example.com\"}}"
        
        // Process each character as a token
        for char in jsonString {
            let tokenId = Int32(char.asciiValue ?? 0)
            let token = MLXArray([tokenId])
            processor.didSample(token: token)
        }
        
        // Check detected keys
        let detectedKeys = processor.allDetectedKeys
        #expect(detectedKeys.contains("user"))
        #expect(detectedKeys.contains("name"))
        #expect(detectedKeys.contains("email"))
        #expect(detectedKeys.count == 3)
    }
    
    @Test("Track JSON parsing phases")
    func trackParsingPhases() {
        let tokenizer = MockTokenizer()
        let modelCard = MockModelCard()
        let processor = KeyDetectionLogitProcessor(tokenizer: tokenizer, modelCard: modelCard, verbose: false)

        // Initialize processor
        let prompt = MLX.zeros([1, 10])
        processor.prompt(prompt)

        // Process opening brace
        processor.didSample(token: MLXArray([123])) // {

        // Process opening quote for key
        processor.didSample(token: MLXArray([34])) // "

        // Process key characters
        processor.didSample(token: MLXArray([107])) // k
        processor.didSample(token: MLXArray([101])) // e
        processor.didSample(token: MLXArray([121])) // y

        // Close key quote
        processor.didSample(token: MLXArray([34])) // "

        // Process colon
        processor.didSample(token: MLXArray([58])) // :

        // Check that key was detected after colon
        let detectedKeys = processor.allDetectedKeys
        #expect(detectedKeys.contains("key"))
    }
    
    @Test("Handle array with objects")
    func handleArrayWithObjects() {
        let tokenizer = MockTokenizer()
        let modelCard = MockModelCard()
        let processor = KeyDetectionLogitProcessor(tokenizer: tokenizer, modelCard: modelCard, verbose: false)
        
        // Initialize processor
        let prompt = MLX.zeros([1, 10])
        processor.prompt(prompt)
        
        // Simulate generating: [{"id":1},{"id":2}]
        let jsonString = "[{\"id\":1},{\"id\":2}]"
        
        // Process each character as a token
        for char in jsonString {
            let tokenId = Int32(char.asciiValue ?? 0)
            let token = MLXArray([tokenId])
            processor.didSample(token: token)
        }
        
        // Check detected keys (should detect "id" twice)
        let detectedKeys = processor.allDetectedKeys
        #expect(detectedKeys.filter { $0 == "id" }.count == 2)
    }
    
    @Test("Handle escape sequences in keys")
    func handleEscapeSequences() {
        let tokenizer = MockTokenizer()
        let modelCard = MockModelCard()
        let processor = KeyDetectionLogitProcessor(tokenizer: tokenizer, modelCard: modelCard, verbose: false)
        
        // Initialize processor
        let prompt = MLX.zeros([1, 10])
        processor.prompt(prompt)
        
        // Simulate generating: {"key\"with\"quotes":"value"}
        let tokens: [Int32] = [
            123, // {
            34,  // "
            107, 101, 121, // key
            92, 34, // \"
            119, 105, 116, 104, // with
            92, 34, // \"
            113, 117, 111, 116, 101, 115, // quotes
            34,  // "
            58,  // :
            34,  // "
            118, 97, 108, 117, 101, // value
            34,  // "
            125  // }
        ]
        
        // Process each token
        for tokenId in tokens {
            let token = MLXArray([tokenId])
            processor.didSample(token: token)
        }
        
        // Check that the escaped key was properly detected
        let detectedKeys = processor.allDetectedKeys
        #expect(detectedKeys.contains("key\"with\"quotes"))
    }
    
    @Test("JSONStateMachine basic transitions")
    func stateMachineTransitions() {
        var machine = JSONStateMachine()
        
        // Initial state
        #expect(machine.phase == .root)
        #expect(machine.isComplete == false)
        #expect(machine.isError == false)
        
        // Start object
        machine.processCharacter("{")
        if case .inObject(.expectKeyOrEnd) = machine.phase {
            // Expected
        } else {
            Issue.record("Expected to be in object.expectKeyOrEnd phase")
        }
        
        // Start key
        machine.processCharacter("\"")
        if case .inString(.body(kind: .key, escaped: false)) = machine.phase {
            // Expected
        } else {
            Issue.record("Expected to be in string key phase")
        }
        
        // Key content
        machine.processCharacter("k")
        machine.processCharacter("e")
        machine.processCharacter("y")
        
        // End key
        machine.processCharacter("\"")
        if case .inObject(.expectColon) = machine.phase {
            // Expected
        } else {
            Issue.record("Expected to be expecting colon")
        }
        
        // Colon
        machine.processCharacter(":")
        if case .inObject(.expectValue) = machine.phase {
            // Expected
        } else {
            Issue.record("Expected to be expecting value")
        }
        
        // String value
        machine.processCharacter("\"")
        if case .inString(.body(kind: .value, escaped: false)) = machine.phase {
            // Expected
        } else {
            Issue.record("Expected to be in string value phase")
        }
        
        // Value content
        machine.processCharacter("v")
        machine.processCharacter("a")
        machine.processCharacter("l")
        
        // End value
        machine.processCharacter("\"")
        if case .inObject(.expectCommaOrEnd) = machine.phase {
            // Expected
        } else {
            Issue.record("Expected to be expecting comma or end")
        }
        
        // End object
        machine.processCharacter("}")
        #expect(machine.phase == .done)
        #expect(machine.isComplete == true)
    }
    
    @Test("JSONStateMachine error handling")
    func stateMachineErrorHandling() {
        var machine = JSONStateMachine()

        // Invalid character at root
        machine.processCharacter("x")
        #expect(machine.phase == .error)
        #expect(machine.isError == true)

        // Reset and test invalid sequence
        machine.reset()
        machine.processCharacter("{")
        machine.processCharacter("x") // Invalid: expecting quote or }
        #expect(machine.phase == .error)
    }
}

// MARK: - Integration Tests

@Suite("Key Detection Integration Tests")
struct KeyDetectionIntegrationTests {

    @Test("Detects keys with correct context in CompanyProfile")
    func testCompanyProfileContextDetection() {
        let nestedSchemas = TestSchemas.companyProfileNestedSchemas
        let rootKeys = TestSchemas.companyProfileRootKeys

        let tokenizer = CharacterTokenizer()
        let modelCard = MockModelCard()
        let processor = KeyDetectionLogitProcessor(
            tokenizer: tokenizer,
            modelCard: modelCard,
            schemaKeys: rootKeys,
            nestedSchemas: nestedSchemas,
            verbose: false,
            showProbabilities: false
        )

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

        processor.prompt(MLXArray.zeros([1]))

        for char in json {
            let tokenId = Int32(char.asciiValue ?? 0)
            let token = MLXArray([tokenId])
            processor.didSample(token: token)
        }

        let detectedKeys = processor.allDetectedKeys

        #expect(detectedKeys.contains("name"))
        #expect(detectedKeys.contains("headquarters"))
        #expect(detectedKeys.contains("departments"))
        #expect(detectedKeys.contains("city"))
        #expect(detectedKeys.contains("country"))
        #expect(detectedKeys.contains("firstName"))
        #expect(detectedKeys.contains("email"))
        #expect(detectedKeys.contains("budget"))
    }

    @Test("Complete JSON detection")
    func completeJSONDetection() {
        let testCases: [(json: String, expectedKeys: [String])] = [
            (
                json: #"{"name":"John","age":30}"#,
                expectedKeys: ["name", "age"]
            ),
            (
                json: #"{"user":{"firstName":"Alice","lastName":"Smith"},"active":true}"#,
                expectedKeys: ["user", "firstName", "lastName"]
            ),
            (
                json: #"{"items":[{"id":1,"name":"Item1"},{"id":2,"name":"Item2"}],"total":2}"#,
                expectedKeys: ["items", "id", "name", "id", "name", "total"]
            )
        ]

        for testCase in testCases {
            var stateMachine = JSONStateMachine()
            var detectedKeys: [String] = []

            for char in testCase.json {
                let previousPhase = stateMachine.phase
                stateMachine.processCharacter(char)

                if case .inString(.body(kind: .key, escaped: false)) = previousPhase {
                    if case .inObject(.expectColon) = stateMachine.phase {
                        let key = stateMachine.currentKey
                        if !key.isEmpty {
                            detectedKeys.append(key)
                        }
                    }
                }
            }

            for expectedKey in testCase.expectedKeys {
                #expect(detectedKeys.contains(expectedKey))
            }
        }
    }

    @Test("Token-split keys")
    func tokenSplitKeys() {
        let splitKeyScenarios = [
            (tokens: ["{\"" , "na", "me", "\":\"" , "John", "\"}"], expectedKey: "name"),
            (tokens: ["{\"" , "first", "Name", "\":\"" , "Alice", "\"}"], expectedKey: "firstName"),
            (tokens: ["{\"" , "email", "Address", "\":\"" , "test@", "example.com", "\"}"], expectedKey: "emailAddress")
        ]

        for scenario in splitKeyScenarios {
            var stateMachine = JSONStateMachine()
            var detectedKeys: [String] = []

            for token in scenario.tokens {
                for char in token {
                    let previousPhase = stateMachine.phase
                    stateMachine.processCharacter(char)

                    if case .inString(.body(kind: .key, escaped: false)) = previousPhase {
                        if case .inObject(.expectColon) = stateMachine.phase {
                            let key = stateMachine.currentKey
                            if !key.isEmpty {
                                detectedKeys.append(key)
                            }
                        }
                    }
                }
            }

            #expect(detectedKeys.contains(scenario.expectedKey))
        }
    }

    @Test("JSONExtractor embedded JSON")
    func jsonExtractorEmbeddedJSON() {
        let testCases = [
            (
                input: "Here's some JSON: {\"key\": \"value\"} and more text",
                expectedJSON: "{\"key\": \"value\"}"
            ),
            (
                input: "<|channel|>final<|message|>{\"name\": \"Alice\", \"age\": 30}",
                expectedJSON: "{\"name\": \"Alice\", \"age\": 30}"
            )
        ]

        for testCase in testCases {
            var extractor = JSONExtractor()
            var jsonBuffer = ""

            for char in testCase.input {
                let shouldProcess = extractor.processCharacter(char)
                if shouldProcess {
                    jsonBuffer.append(char)
                }
            }

            let normalizedBuffer = jsonBuffer.replacingOccurrences(of: " ", with: "")
            let normalizedExpected = testCase.expectedJSON.replacingOccurrences(of: " ", with: "")

            #expect(normalizedBuffer.contains(normalizedExpected) || normalizedBuffer == normalizedExpected)
        }
    }
}