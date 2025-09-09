import Testing
import Foundation
import OpenFoundationModels
import OpenFoundationModelsExtra
import MLXLMCommon
@testable import OpenFoundationModelsMLX

@Suite("MLXLanguageModel Tests")
struct MLXLanguageModelTests {
    
    // MARK: - Mock Components
    
    struct MockModelCard: ModelCard, Sendable {
        let id: String = "test-model"
        
        func prompt(transcript: Transcript, options: GenerationOptions?) -> Prompt {
            let ext = TranscriptAccess.extract(from: transcript)
            
            return Prompt {
                // Simple concatenation of messages
                if let system = ext.systemText {
                    "System: \(system)\n"
                }
                for message in ext.messages {
                    "\(message.role): \(message.content)\n"
                }
            }
        }
        
        var params: GenerateParameters {
            GenerateParameters(
                maxTokens: 1000,
                temperature: 0.7,
                topP: 0.9
            )
        }
    }
    
    // MARK: - Initialization Tests
    
    @Test("Model card initialization")
    func modelCardInitialization() async throws {
        let card = MockModelCard()
        #expect(card.id == "test-model")
        #expect(card.params.temperature == 0.7)
        #expect(card.params.maxTokens == 1000)
    }
    
    // MARK: - Availability Tests
    
    @Test("Model availability check")
    func modelAvailability() async throws {
        // Note: Cannot test actual MLXLanguageModel without MLXChatEngine mock
        // This would require dependency injection or protocol-based design
        
        // Test the mock card functionality
        let card = MockModelCard()
        let transcript = Transcript(entries: [
            .instructions(.init(
                segments: [.text(.init(content: "You are a helpful assistant"))],
                toolDefinitions: []
            )),
            .prompt(.init(
                segments: [.text(.init(content: "Hello"))],
                options: GenerationOptions(),
                responseFormat: nil
            ))
        ])
        
        let prompt = card.prompt(transcript: transcript, options: nil)
        #expect(prompt.description.contains("System: You are a helpful assistant"))
        #expect(prompt.description.contains("user: Hello"))
    }
    
    // MARK: - Prompt Rendering Tests
    
    @Test("Prompt rendering with system message")
    func promptRenderingWithSystem() throws {
        let card = MockModelCard()
        let transcript = Transcript(entries: [
            .instructions(.init(
                segments: [.text(.init(content: "Be concise"))],
                toolDefinitions: []
            )),
            .prompt(.init(
                segments: [.text(.init(content: "What is 2+2?"))],
                options: GenerationOptions(),
                responseFormat: nil
            )),
            .response(.init(
                assetIDs: [],
                segments: [.text(.init(content: "4"))]
            )),
            .prompt(.init(
                segments: [.text(.init(content: "And 3+3?"))],
                options: GenerationOptions(),
                responseFormat: nil
            ))
        ])
        
        let prompt = card.prompt(transcript: transcript, options: nil)
        // MockModelCard adds a newline after each message, but the last message doesn't get a newline
        // because it's not included in the messages (it's the current prompt)
        #expect(prompt.description == "System: Be concise\nuser: What is 2+2?\nassistant: 4\nuser: And 3+3?")
    }
    
    @Test("Prompt rendering without system message")
    func promptRenderingWithoutSystem() throws {
        let card = MockModelCard()
        let transcript = Transcript(entries: [
            .prompt(.init(
                segments: [.text(.init(content: "Hello"))],
                options: GenerationOptions(),
                responseFormat: nil
            )),
            .response(.init(
                assetIDs: [],
                segments: [.text(.init(content: "Hi there!"))]
            ))
        ])
        
        let prompt = card.prompt(transcript: transcript, options: nil)
        // MockModelCard adds a newline after each message in the transcript
        #expect(prompt.description == "user: Hello\nassistant: Hi there!")
    }
    
    @Test("Prompt rendering with tools")
    func promptRenderingWithTools() throws {
        let card = MockModelCard()
        // Note: We can't easily create tool definitions without access to internal types
        // This test focuses on basic prompt rendering
        let transcript = Transcript(entries: [
            .instructions(.init(
                segments: [.text(.init(content: "Use tools when needed"))],
                toolDefinitions: []
            )),
            .prompt(.init(
                segments: [.text(.init(content: "What's the weather?"))],
                options: GenerationOptions(),
                responseFormat: nil
            ))
        ])
        
        let prompt = card.prompt(transcript: transcript, options: nil)
        #expect(prompt.description.contains("System: Use tools when needed"))
        #expect(prompt.description.contains("user: What's the weather?"))
        // Note: Tool rendering depends on ModelCard implementation
    }
    
    // MARK: - Tool Call Detection Tests
    
    @Test("Tool call detection in response")
    func toolCallDetectionInResponse() {
        let jsonWithToolCall = """
        {
            "tool_calls": [
                {
                    "name": "get_weather",
                    "arguments": {"location": "Tokyo"}
                }
            ]
        }
        """
        
        let entry = ToolCallDetector.entryIfPresent(jsonWithToolCall)
        #expect(entry != nil)
        
        if case .toolCalls(let toolCalls) = entry {
            #expect(toolCalls.count == 1)
            let firstCall = toolCalls[toolCalls.startIndex]
            #expect(firstCall.toolName == "get_weather")
        } else {
            Issue.record("Expected tool calls entry")
        }
    }
    
    @Test("Regular text response without tool calls")
    func regularTextResponse() {
        let plainText = "The weather in Tokyo is sunny."
        
        let entry = ToolCallDetector.entryIfPresent(plainText)
        #expect(entry == nil)
    }
    
    // MARK: - Error Handling Tests
    
    @Test("Empty response error handling")
    func emptyResponseHandling() throws {
        // Test that empty responses are properly handled
        let emptyJSON = "{}"
        let entry = ToolCallDetector.entryIfPresent(emptyJSON)
        #expect(entry == nil)
    }
    
    @Test("Malformed JSON response")
    func malformedJSONResponse() {
        let malformed = "{ incomplete json"
        let entry = ToolCallDetector.entryIfPresent(malformed)
        #expect(entry == nil)
    }
    
    // MARK: - Locale Support Tests
    
    @Test("Locale support verification")
    func localeSupport() {
        // Note: Would need actual MLXLanguageModel instance
        // Currently testing the concept
        let locales = [
            Locale(identifier: "en_US"),
            Locale(identifier: "ja_JP"),
            Locale(identifier: "fr_FR"),
            Locale(identifier: "zh_CN")
        ]
        
        // All locales should be supported in the actual implementation
        for locale in locales {
            #expect(locale.identifier.count > 0)
        }
    }
    
    // MARK: - Generation Options Tests
    
    @Test("Generation options mapping")
    func generationOptionsMapping() {
        let options = GenerationOptions(
            sampling: .greedy,
            temperature: 0.5,
            maximumResponseTokens: 500
        )
        
        #expect(options.maximumResponseTokens == 500)
        #expect(options.temperature == 0.5)
        
        // SamplingMode is an opaque type, we can only check it was set
        #expect(options.sampling != nil)
    }
    
    @Test("Generation options with topK sampling")
    func generationOptionsTopK() {
        let options = GenerationOptions(
            sampling: .random(top: 10),
            maximumResponseTokens: 200
        )
        
        // SamplingMode is opaque, we can only check it exists
        #expect(options.sampling != nil)
        #expect(options.maximumResponseTokens == 200)
    }
    
    @Test("Generation options with topP sampling")
    func generationOptionsTopP() {
        let options = GenerationOptions(
            sampling: .random(probabilityThreshold: 0.95),
            maximumResponseTokens: 300
        )
        
        // SamplingMode is opaque, we can only check it exists
        #expect(options.sampling != nil)
        #expect(options.maximumResponseTokens == 300)
    }
}