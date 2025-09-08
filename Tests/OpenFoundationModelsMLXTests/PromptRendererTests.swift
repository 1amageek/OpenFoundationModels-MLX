import Testing
import Foundation
import OpenFoundationModels
import OpenFoundationModelsExtra
import MLXLMCommon
@testable import OpenFoundationModelsMLX

@Suite("PromptRenderer Tests")
struct PromptRendererTests {
    
    // MARK: - Mock Components
    
    struct TestModelCard: ModelCard, Sendable {
        let id: String = "test-model"
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
                case .tool:
                    prompt += "[TOOL]\n\(message.content)\n[/TOOL]\n"
                }
            }
            
            // Add tools if present
            if !input.tools.isEmpty {
                prompt += "\n[TOOLS]\n"
                for tool in input.tools {
                    prompt += "- \(tool.name): \(tool.description ?? "")\n"
                }
                prompt += "[/TOOLS]\n"
            }
            
            return prompt
        }
        
        var params: GenerateParameters {
            GenerateParameters(
                maxTokens: 1000,
                temperature: 0.7,
                topP: 0.9
            )
        }
    }
    
    enum TestError: Error {
        case renderingFailed
    }
    
    // MARK: - Mock Generable Types
    
    struct LocationParam: Generable {
        var location: String
        
        init(_ content: GeneratedContent) throws {
            self.location = try content.value(String.self, forProperty: "location")
        }
        
        init(location: String) {
            self.location = location
        }
        
        var generatedContent: GeneratedContent {
            GeneratedContent(kind: .structure(
                properties: ["location": GeneratedContent(kind: .string(location))],
                orderedKeys: ["location"]
            ))
        }
        
        static var generationSchema: GenerationSchema {
            GenerationSchema(
                type: LocationParam.self,
                description: "Location parameter",
                properties: [
                    GenerationSchema.Property(
                        name: "location",
                        description: "Location name",
                        type: String.self,
                        guides: []
                    )
                ]
            )
        }
    }
    
    struct UserInfo: Generable {
        var name: String
        
        init(_ content: GeneratedContent) throws {
            self.name = try content.value(String.self, forProperty: "name")
        }
        
        init(name: String) {
            self.name = name
        }
        
        var generatedContent: GeneratedContent {
            GeneratedContent(kind: .structure(
                properties: ["name": GeneratedContent(kind: .string(name))],
                orderedKeys: ["name"]
            ))
        }
        
        static var generationSchema: GenerationSchema {
            GenerationSchema(
                type: UserInfo.self,
                description: "User information",
                properties: [
                    GenerationSchema.Property(
                        name: "name",
                        description: "User name",
                        type: String.self,
                        guides: []
                    )
                ]
            )
        }
    }
    
    // MARK: - Basic Rendering Tests
    
    @Test("Creates ChatRequest with rendered prompt")
    func createsRequestWithPrompt() throws {
        let card = TestModelCard()
        let transcript = Transcript(entries: [
            .prompt(.init(
                segments: [.text(.init(content: "Hello, world!"))],
                options: GenerationOptions(),
                responseFormat: nil
            ))
        ])
        
        let request = try PromptRenderer.buildRequest(
            card: card,
            transcript: transcript,
            options: nil
        )
        
        // Check the request has the rendered prompt
        #expect(request.prompt.contains("[USER]\nHello, world!\n[/USER]"))
        #expect(request.modelID == "test-model")
    }
    
    @Test("Handles conversation with multiple turns")
    func multiTurnConversation() throws {
        let card = TestModelCard()
        let transcript = Transcript(entries: [
            .prompt(.init(
                segments: [.text(.init(content: "What is 2+2?"))],
                options: GenerationOptions(),
                responseFormat: nil
            )),
            .response(.init(
                assetIDs: [],
                segments: [.text(.init(content: "2+2 equals 4."))]
            )),
            .prompt(.init(
                segments: [.text(.init(content: "And 3+3?"))],
                options: GenerationOptions(),
                responseFormat: nil
            ))
        ])
        
        let request = try PromptRenderer.buildRequest(
            card: card,
            transcript: transcript,
            options: nil
        )
        
        // Check the prompt contains all conversation turns
        #expect(request.prompt.contains("[USER]\nWhat is 2+2?\n[/USER]"))
        #expect(request.prompt.contains("[ASSISTANT]\n2+2 equals 4.\n[/ASSISTANT]"))
        #expect(request.prompt.contains("[USER]\nAnd 3+3?\n[/USER]"))
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
                options: GenerationOptions(),
                responseFormat: nil
            ))
        ])
        
        let request = try PromptRenderer.buildRequest(
            card: card,
            transcript: transcript,
            options: nil
        )
        
        // System message should be included in the prompt
        #expect(request.prompt.contains("[SYSTEM]\nYou are a helpful assistant.\n[/SYSTEM]"))
        #expect(request.prompt.contains("[USER]\nHello\n[/USER]"))
    }
    
    // MARK: - Tool Definition Tests
    
    @Test("Processes tool definitions")
    func toolDefinitions() throws {
        let card = TestModelCard()
        
        // Create tool definition using Generable type
        let toolDef = Transcript.ToolDefinition(
            name: "get_weather",
            description: "Get current weather for a location",
            parameters: LocationParam.generationSchema
        )
        
        let transcript = Transcript(entries: [
            .instructions(.init(
                segments: [.text(.init(content: "Use tools when appropriate."))],
                toolDefinitions: [toolDef]
            )),
            .prompt(.init(
                segments: [.text(.init(content: "What's the weather in Tokyo?"))],
                options: GenerationOptions(),
                responseFormat: nil
            ))
        ])
        
        let request = try PromptRenderer.buildRequest(
            card: card,
            transcript: transcript,
            options: nil
        )
        
        // Check that tools are included in the prompt
        #expect(request.prompt.contains("[SYSTEM]\nUse tools when appropriate.\n[/SYSTEM]"))
        #expect(request.prompt.contains("[USER]\nWhat's the weather in Tokyo?\n[/USER]"))
        #expect(request.prompt.contains("[TOOLS]"))
        #expect(request.prompt.contains("get_weather"))
    }
    
    // MARK: - Schema Tests
    
    @Test("Extracts response format schema")
    func responseFormatSchema() throws {
        let card = TestModelCard()
        
        // Create a response format with schema
        let responseFormat = Transcript.ResponseFormat(schema: UserInfo.generationSchema)
        
        let transcript = Transcript(entries: [
            .prompt(.init(
                segments: [.text(.init(content: "Generate user info"))],
                options: GenerationOptions(),
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
                options: GenerationOptions(),
                responseFormat: nil
            ))
        ])
        
        let options = GenerationOptions(
            sampling: .greedy,
            temperature: 0.5,
            maximumResponseTokens: 500
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
                options: GenerationOptions(),
                responseFormat: nil
            ))
        ])
        
        let request = try PromptRenderer.buildRequest(
            card: card,
            transcript: transcript,
            options: nil
        )
        
        // When no options provided, parameters should be set directly
        #expect(request.parameters != nil)
        #expect(request.parameters?.temperature == card.params.temperature)
        #expect(request.parameters?.maxTokens == card.params.maxTokens)
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
                options: GenerationOptions(),
                responseFormat: nil
            )),
            .toolCalls(.init(id: "1", [toolCall])),
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
        
        // Tool calls and outputs should be included in the prompt
        #expect(request.prompt.contains("[USER]\nCalculate 2+2\n[/USER]"))
        #expect(request.prompt.contains("[ASSISTANT]\nThe result is 4.\n[/ASSISTANT]"))
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
                options: GenerationOptions(),
                responseFormat: nil
            ))
        ])
        
        let request = try PromptRenderer.buildRequest(
            card: card,
            transcript: transcript,
            options: nil
        )
        
        // System message should appear only once in the prompt
        let systemCount = request.prompt.components(separatedBy: "[SYSTEM]").count - 1
        #expect(systemCount == 1)
    }
    
    // MARK: - Error Handling Tests
    
    @Test("Handles rendering errors")
    func renderingError() throws {
        var card = TestModelCard()
        card.shouldThrowError = true
        
        let transcript = Transcript(entries: [
            .prompt(.init(
                segments: [.text(.init(content: "Test"))],
                options: GenerationOptions(),
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
}