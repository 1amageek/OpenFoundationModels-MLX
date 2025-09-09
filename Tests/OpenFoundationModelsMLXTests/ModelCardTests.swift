import Testing
import Foundation
import OpenFoundationModels
import OpenFoundationModelsExtra
import MLXLMCommon
@testable import OpenFoundationModelsMLX

@Suite("ModelCard Tests")
struct ModelCardTests {
    
    // MARK: - Mock Components
    
    struct TestModelCard: ModelCard, Sendable {
        let id: String = "test-model"
        
        var params: GenerateParameters {
            GenerateParameters(
                maxTokens: 1000,
                temperature: 0.7,
                topP: 0.9
            )
        }
        
        func prompt(transcript: Transcript, options: GenerationOptions?) -> Prompt {
            let ext = TranscriptAccess.extract(from: transcript)
            
            return Prompt {
                // Add system message if present
                if let system = ext.systemText {
                    "[SYSTEM]\n\(system)\n[/SYSTEM]\n\n"
                }
                
                // Add messages
                for message in ext.messages {
                    switch message.role {
                    case .user:
                        "[USER]\n\(message.content)\n[/USER]\n"
                    case .assistant:
                        "[ASSISTANT]\n\(message.content)\n[/ASSISTANT]\n"
                    case .system:
                        "[SYSTEM]\n\(message.content)\n[/SYSTEM]\n"
                    case .tool:
                        "[TOOL]\n\(message.content)\n[/TOOL]\n"
                    }
                }
                
                // Add tools if present
                if !ext.toolDefs.isEmpty {
                    "\n[TOOLS]\n"
                    for tool in ext.toolDefs {
                        "- \(tool.name): \(tool.description ?? "")\n"
                    }
                    "[/TOOLS]\n"
                }
            }
        }
    }
    
    // MARK: - Basic Rendering Tests
    
    @Test("Creates prompt from transcript")
    func createsPromptFromTranscript() throws {
        let card = TestModelCard()
        let transcript = Transcript(entries: [
            .prompt(.init(
                segments: [.text(.init(content: "Hello, world!"))],
                options: GenerationOptions(),
                responseFormat: nil
            ))
        ])
        
        let prompt = card.prompt(transcript: transcript, options: nil)
        
        // Check the prompt contains expected content
        #expect(prompt.description.contains("[USER]\nHello, world!\n[/USER]"))
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
        
        let prompt = card.prompt(transcript: transcript, options: nil)
        
        // Check the prompt contains all conversation turns
        #expect(prompt.description.contains("[USER]\nWhat is 2+2?\n[/USER]"))
        #expect(prompt.description.contains("[ASSISTANT]\n2+2 equals 4.\n[/ASSISTANT]"))
        #expect(prompt.description.contains("[USER]\nAnd 3+3?\n[/USER]"))
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
        
        let prompt = card.prompt(transcript: transcript, options: nil)
        
        // Check system instructions are included
        #expect(prompt.description.contains("[SYSTEM]\nYou are a helpful assistant.\n[/SYSTEM]"))
        #expect(prompt.description.contains("[USER]\nHello\n[/USER]"))
    }
    
    @Test("ChatGPT ModelCard generates correct format")
    func chatGPTFormat() throws {
        let card = ChatGPTModelCard(id: "gpt-4")
        let transcript = Transcript(entries: [
            .prompt(.init(
                segments: [.text(.init(content: "Hello"))],
                options: GenerationOptions(),
                responseFormat: nil
            ))
        ])
        
        let prompt = card.prompt(transcript: transcript, options: nil)
        
        // Check ChatGPT-specific format
        #expect(prompt.description.contains("<|start|>"))
        #expect(prompt.description.contains("<|end|>"))
        #expect(prompt.description.contains("You are ChatGPT"))
    }
    
    @Test("Llama ModelCard generates correct format")
    func llamaFormat() throws {
        let card = LlamaModelCard(id: "llama-3")
        let transcript = Transcript(entries: [
            .prompt(.init(
                segments: [.text(.init(content: "Hello"))],
                options: GenerationOptions(),
                responseFormat: nil
            ))
        ])
        
        let prompt = card.prompt(transcript: transcript, options: nil)
        
        // Check Llama-specific format
        #expect(prompt.description.contains("[INST]"))
        #expect(prompt.description.contains("[/INST]"))
    }
    
    @Test("ChatML ModelCard generates correct format")
    func chatMLFormat() throws {
        let card = ChatMLModelCard(id: "model")
        let transcript = Transcript(entries: [
            .prompt(.init(
                segments: [.text(.init(content: "Hello"))],
                options: GenerationOptions(),
                responseFormat: nil
            ))
        ])
        
        let prompt = card.prompt(transcript: transcript, options: nil)
        
        // Check ChatML-specific format
        #expect(prompt.description.contains("<|im_start|>"))
        #expect(prompt.description.contains("<|im_end|>"))
    }
}