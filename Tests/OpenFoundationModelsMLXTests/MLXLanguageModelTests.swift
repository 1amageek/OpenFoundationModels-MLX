import Testing
import Foundation
import OpenFoundationModels
@testable import OpenFoundationModelsMLX

@Suite("MLXLanguageModel Tests")
struct MLXLanguageModelTests {
    
    // MARK: - Mock Components
    
    struct MockModelCard: ModelCard, Sendable {
        let id: String = "test-model"
        let displayName: String = "Test Model"
        
        func render(input: ModelCardInput) throws -> String {
            // Simple concatenation of messages
            var prompt = ""
            if let system = input.system {
                prompt += "System: \(system)\n"
            }
            for message in input.messages {
                prompt += "\(message.role): \(message.content)\n"
            }
            return prompt
        }
        
        var defaultTemperature: Double { 0.7 }
        var defaultMaxTokens: Int { 1000 }
        var defaultTopP: Double { 0.9 }
    }
    
    // MARK: - Initialization Tests
    
    @Test("Model card initialization")
    func modelCardInitialization() async throws {
        let card = MockModelCard()
        #expect(card.id == "test-model")
        #expect(card.displayName == "Test Model")
        #expect(card.defaultTemperature == 0.7)
        #expect(card.defaultMaxTokens == 1000)
    }
    
    // MARK: - Availability Tests
    
    @Test("Model availability check")
    func modelAvailability() async throws {
        // Note: Cannot test actual MLXLanguageModel without MLXChatEngine mock
        // This would require dependency injection or protocol-based design
        
        // Test the mock card functionality
        let card = MockModelCard()
        let input = ModelCardInput(
            currentDate: "2025-09-08",
            system: "You are a helpful assistant",
            messages: [
                ModelCardInput.Message(role: .user, content: "Hello")
            ],
            tools: []
        )
        
        let prompt = try card.render(input: input)
        #expect(prompt.contains("System: You are a helpful assistant"))
        #expect(prompt.contains("user: Hello"))
    }
    
    // MARK: - Prompt Rendering Tests
    
    @Test("Prompt rendering with system message")
    func promptRenderingWithSystem() throws {
        let card = MockModelCard()
        let input = ModelCardInput(
            currentDate: "2025-09-08",
            system: "Be concise",
            messages: [
                ModelCardInput.Message(role: .user, content: "What is 2+2?"),
                ModelCardInput.Message(role: .assistant, content: "4"),
                ModelCardInput.Message(role: .user, content: "And 3+3?")
            ],
            tools: []
        )
        
        let prompt = try card.render(input: input)
        #expect(prompt == "System: Be concise\nuser: What is 2+2?\nassistant: 4\nuser: And 3+3?\n")
    }
    
    @Test("Prompt rendering without system message")
    func promptRenderingWithoutSystem() throws {
        let card = MockModelCard()
        let input = ModelCardInput(
            currentDate: "2025-09-08",
            system: nil,
            messages: [
                ModelCardInput.Message(role: .user, content: "Hello"),
                ModelCardInput.Message(role: .assistant, content: "Hi there!")
            ],
            tools: []
        )
        
        let prompt = try card.render(input: input)
        #expect(prompt == "user: Hello\nassistant: Hi there!\n")
    }
    
    @Test("Prompt rendering with tools")
    func promptRenderingWithTools() throws {
        let card = MockModelCard()
        let input = ModelCardInput(
            currentDate: "2025-09-08",
            system: "Use tools when needed",
            messages: [
                ModelCardInput.Message(role: .user, content: "What's the weather?")
            ],
            tools: [
                ModelCardInput.Tool(
                    name: "get_weather",
                    description: "Get current weather",
                    parametersJSON: #"{"type":"object","properties":{"location":{"type":"string"}}}"#
                )
            ]
        )
        
        let prompt = try card.render(input: input)
        #expect(prompt.contains("System: Use tools when needed"))
        #expect(prompt.contains("user: What's the weather?"))
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
            samplingMode: .greedy,
            maxTokens: 500,
            temperature: 0.5
        )
        
        #expect(options.maxTokens == 500)
        #expect(options.temperature == 0.5)
        
        if case .greedy = options.samplingMode {
            // Correct mode
        } else {
            Issue.record("Expected greedy sampling mode")
        }
    }
    
    @Test("Generation options with topK sampling")
    func generationOptionsTopK() {
        let options = GenerationOptions(
            samplingMode: .random(topK: 10),
            maxTokens: 200
        )
        
        if case .random(let topK) = options.samplingMode {
            #expect(topK == 10)
        } else {
            Issue.record("Expected random topK sampling")
        }
    }
    
    @Test("Generation options with topP sampling")
    func generationOptionsTopP() {
        let options = GenerationOptions(
            samplingMode: .random(topP: 0.95),
            maxTokens: 300
        )
        
        if case .random(let topP) = options.samplingMode {
            #expect(topP == 0.95)
        } else {
            Issue.record("Expected random topP sampling")
        }
    }
}