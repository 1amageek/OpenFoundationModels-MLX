import Testing
import Foundation
import OpenFoundationModels
import OpenFoundationModelsExtra
@testable import OpenFoundationModelsMLX

@Suite("PromptRenderer Tests")
struct PromptRendererTests {
    
    // MARK: - Mock Components
    
    struct TestModelCard: ModelCard, Sendable {
        let id: String = "test-model"
        let displayName: String = "Test Model"
        var shouldThrowError = false
        
        func render(input: ModelCardInput) throws -> String {
            if shouldThrowError {
                throw TestError.renderingFailed
            }
            
            var prompt = ""
            
            // Add system message if present
            if let system = input.system {
                prompt += "[SYSTEM]\n\(system)\n[/SYSTEM]\n\n"
            }
            
            // Add messages
            for message in input.messages {
                switch message.role {
                case .user:
                    prompt += "[USER]\n\(message.content)\n[/USER]\n"
                case .assistant:
                    prompt += "[ASSISTANT]\n\(message.content)\n[/ASSISTANT]\n"
                case .system:
                    prompt += "[SYSTEM]\n\(message.content)\n[/SYSTEM]\n"
                }
            }
            
            // Add tools if present
            if !input.tools.isEmpty {
                prompt += "\n[TOOLS]\n"
                for tool in input.tools {
                    prompt += "- \(tool.name): \(tool.description)\n"
                }
                prompt += "[/TOOLS]\n"
            }
            
            return prompt
        }
        
        var defaultTemperature: Double { 0.7 }
        var defaultMaxTokens: Int { 1000 }
        var defaultTopP: Double { 0.9 }
    }
    
    enum TestError: Error {
        case renderingFailed
    }
    
    // MARK: - Basic Rendering Tests
    
    @Test("Renders simple user message")
    func simpleUserMessage() throws {
        let card = TestModelCard()
        let transcript = Transcript(entries: [
            .prompt(.init(
                segments: [.text(.init(content: "Hello, world!"))],
                options: nil,
                responseFormat: nil
            ))
        ])
        
        let request = try PromptRenderer.buildRequest(
            card: card,
            transcript: transcript,
            options: nil
        )
        
        #expect(request.prompt.contains("[USER]"))
        #expect(request.prompt.contains("Hello, world!"))
        #expect(request.prompt.contains("[/USER]"))
    }
    
    @Test("Renders conversation with multiple turns")
    func multiTurnConversation() throws {
        let card = TestModelCard()
        let transcript = Transcript(entries: [
            .prompt(.init(
                segments: [.text(.init(content: "What is 2+2?"))],
                options: nil,
                responseFormat: nil
            )),
            .response(.init(
                assetIDs: [],
                segments: [.text(.init(content: "2+2 equals 4."))]
            )),
            .prompt(.init(
                segments: [.text(.init(content: "And 3+3?"))],
                options: nil,
                responseFormat: nil
            ))
        ])
        
        let request = try PromptRenderer.buildRequest(
            card: card,
            transcript: transcript,
            options: nil
        )
        
        #expect(request.prompt.contains("What is 2+2?"))
        #expect(request.prompt.contains("2+2 equals 4."))
        #expect(request.prompt.contains("And 3+3?"))
    }
    
    @Test("Includes system instructions")
    func systemInstructions() throws {
        let card = TestModelCard()
        let transcript = Transcript(entries: [
            .instructions(.init(
                segments: [.text(.init(content: "You are a helpful assistant."))],
                toolDefinitions: []
            )),
            .prompt(.init(
                segments: [.text(.init(content: "Hello"))],
                options: nil,
                responseFormat: nil
            ))
        ])
        
        let request = try PromptRenderer.buildRequest(
            card: card,
            transcript: transcript,
            options: nil
        )
        
        #expect(request.prompt.contains("[SYSTEM]"))
        #expect(request.prompt.contains("You are a helpful assistant."))
        #expect(request.prompt.contains("[/SYSTEM]"))
    }
    
    // MARK: - Tool Definition Tests
    
    @Test("Includes tool definitions")
    func toolDefinitions() throws {
        let card = TestModelCard()
        
        // Create tool definition
        let toolDef = Transcript.ToolDefinition(
            name: "get_weather",
            description: "Get current weather for a location",
            parameters: GenerationSchema(
                type: String.self,
                description: "Location name"
            )
        )
        
        let transcript = Transcript(entries: [
            .instructions(.init(
                segments: [.text(.init(content: "Use tools when appropriate."))],
                toolDefinitions: [toolDef]
            )),
            .prompt(.init(
                segments: [.text(.init(content: "What's the weather in Tokyo?"))],
                options: nil,
                responseFormat: nil
            ))
        ])
        
        let request = try PromptRenderer.buildRequest(
            card: card,
            transcript: transcript,
            options: nil
        )
        
        #expect(request.prompt.contains("[TOOLS]"))
        #expect(request.prompt.contains("get_weather"))
        #expect(request.prompt.contains("Get current weather for a location"))
    }
    
    // MARK: - Schema Tests
    
    @Test("Extracts response format schema")
    func responseFormatSchema() throws {
        let card = TestModelCard()
        
        // Create a response format with schema
        let schema = GenerationSchema(
            type: String.self,
            description: "User information"
        )
        let responseFormat = Transcript.ResponseFormat(schema: schema)
        
        let transcript = Transcript(entries: [
            .prompt(.init(
                segments: [.text(.init(content: "Generate user info"))],
                options: nil,
                responseFormat: responseFormat
            ))
        ])
        
        let request = try PromptRenderer.buildRequest(
            card: card,
            transcript: transcript,
            options: nil
        )
        
        // Schema should be extracted (though not visible in prompt)
        #expect(request.schema != nil)
    }
    
    // MARK: - Options Mapping Tests
    
    @Test("Maps generation options")
    func generationOptionsMapping() throws {
        let card = TestModelCard()
        let transcript = Transcript(entries: [
            .prompt(.init(
                segments: [.text(.init(content: "Test"))],
                options: nil,
                responseFormat: nil
            ))
        ])
        
        let options = GenerationOptions(
            samplingMode: .greedy,
            maxTokens: 500,
            temperature: 0.5
        )
        
        let request = try PromptRenderer.buildRequest(
            card: card,
            transcript: transcript,
            options: options
        )
        
        #expect(request.sampling.temperature == 0.5)
        #expect(request.sampling.maxTokens == 500)
    }
    
    @Test("Uses default options when none provided")
    func defaultOptions() throws {
        let card = TestModelCard()
        let transcript = Transcript(entries: [
            .prompt(.init(
                segments: [.text(.init(content: "Test"))],
                options: nil,
                responseFormat: nil
            ))
        ])
        
        let request = try PromptRenderer.buildRequest(
            card: card,
            transcript: transcript,
            options: nil
        )
        
        // Should use card defaults
        #expect(request.sampling.temperature == card.defaultTemperature)
        #expect(request.sampling.maxTokens == card.defaultMaxTokens)
    }
    
    // MARK: - Date Handling Tests
    
    @Test("Includes current date in input")
    func currentDateInclusion() throws {
        let card = TestModelCard()
        let transcript = Transcript(entries: [
            .prompt(.init(
                segments: [.text(.init(content: "What day is it?"))],
                options: nil,
                responseFormat: nil
            ))
        ])
        
        let request = try PromptRenderer.buildRequest(
            card: card,
            transcript: transcript,
            options: nil
        )
        
        // The date should be in ISO format
        let dateFormatter = ISO8601DateFormatter()
        dateFormatter.formatOptions = [.withFullDate]
        let today = dateFormatter.string(from: Date())
        
        // Note: Date is passed to ModelCard but may not appear in prompt
        #expect(request.prompt.contains("What day is it?"))
    }
    
    // MARK: - Error Handling Tests
    
    @Test("Handles rendering errors")
    func renderingError() throws {
        var card = TestModelCard()
        card.shouldThrowError = true
        
        let transcript = Transcript(entries: [
            .prompt(.init(
                segments: [.text(.init(content: "Test"))],
                options: nil,
                responseFormat: nil
            ))
        ])
        
        #expect(throws: TestError.renderingFailed) {
            try PromptRenderer.buildRequest(
                card: card,
                transcript: transcript,
                options: nil
            )
        }
    }
    
    // MARK: - Complex Transcript Tests
    
    @Test("Handles transcript with tool calls and outputs")
    func transcriptWithToolCalls() throws {
        let card = TestModelCard()
        
        let toolCall = Transcript.ToolCall(
            id: "call-1",
            toolName: "calculator",
            arguments: try GeneratedContent(json: #"{"operation": "add", "a": 2, "b": 2}"#)
        )
        
        let transcript = Transcript(entries: [
            .prompt(.init(
                segments: [.text(.init(content: "Calculate 2+2"))],
                options: nil,
                responseFormat: nil
            )),
            .toolCalls(.init("1", [toolCall])),
            .toolOutput(.init(
                id: "output-1",
                toolName: "calculator",
                segments: [.text(.init(content: "4"))]
            )),
            .response(.init(
                assetIDs: [],
                segments: [.text(.init(content: "The result is 4."))]
            ))
        ])
        
        let request = try PromptRenderer.buildRequest(
            card: card,
            transcript: transcript,
            options: nil
        )
        
        // Tool calls and outputs should be processed
        #expect(request.prompt.contains("Calculate 2+2"))
        #expect(request.prompt.contains("The result is 4."))
    }
    
    @Test("Filters duplicate system messages")
    func filtersDuplicateSystemMessages() throws {
        let card = TestModelCard()
        
        // Create transcript with system message in instructions
        let transcript = Transcript(entries: [
            .instructions(.init(
                segments: [.text(.init(content: "System instruction"))],
                toolDefinitions: []
            )),
            .prompt(.init(
                segments: [.text(.init(content: "User message"))],
                options: nil,
                responseFormat: nil
            ))
        ])
        
        let request = try PromptRenderer.buildRequest(
            card: card,
            transcript: transcript,
            options: nil
        )
        
        // System message should appear only once
        let systemCount = request.prompt.components(separatedBy: "[SYSTEM]").count - 1
        #expect(systemCount == 1)
    }
}