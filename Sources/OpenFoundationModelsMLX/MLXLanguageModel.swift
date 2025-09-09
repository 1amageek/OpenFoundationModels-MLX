import Foundation
import OpenFoundationModels
import MLXLMCommon
import MLXLLM

/// MLXLanguageModel is a simple adapter that bridges MLX's ModelContainer
/// to OpenFoundationModels' LanguageModel protocol.
public struct MLXLanguageModel: OpenFoundationModels.LanguageModel, Sendable {
    private let modelContainer: ModelContainer
    
    /// Initialize with a pre-loaded ModelContainer.
    /// The model should already be loaded using MLX-swift-examples' ModelFactory.
    /// 
    /// Example:
    /// ```swift
    /// let modelFactory = LLMModelFactory.shared
    /// let config = ModelConfiguration(id: "mlx-community/Llama-3.2-1B-Instruct-4bit")
    /// let container = try await modelFactory.loadContainer(configuration: config)
    /// let model = MLXLanguageModel(modelContainer: container)
    /// ```
    public init(modelContainer: ModelContainer) {
        self.modelContainer = modelContainer
    }
    
    public var isAvailable: Bool { true }
    
    public func supports(locale: Locale) -> Bool { true }
    
    public func generate(transcript: Transcript, options: GenerationOptions?) async throws -> Transcript.Entry {
        // Convert transcript to Chat.Message array
        let messages = convertTranscriptToMessages(transcript)
        
        // Create UserInput with chat messages
        let userInput = UserInput(chat: messages)
        
        // Get generation parameters from options
        let parameters = GenerateParameters(
            maxTokens: options?.maximumResponseTokens ?? 2048,
            temperature: options?.temperature.map { Float($0) } ?? 0.7
        )
        
        // Generate using ModelContainer
        let generatedText = try await modelContainer.perform { (context: ModelContext) in
            let input = try await context.processor.prepare(input: userInput)
            
            // Use generate with callback to get GenerateResult
            let result = try MLXLMCommon.generate(
                input: input,
                parameters: parameters,
                context: context
            ) { (_: [Int]) in
                // Continue generation
                return .more
            }
            
            // Return the output text
            return result.output
        }
        
        // Return the generated text as a response
        return .response(.init(
            assetIDs: [],
            segments: [.text(.init(content: generatedText))]
        ))
    }
    
    public func stream(transcript: Transcript, options: GenerationOptions?) -> AsyncStream<Transcript.Entry> {
        AsyncStream { continuation in
            Task {
                do {
                    // Convert transcript to Chat.Message array
                    let messages = convertTranscriptToMessages(transcript)
                    
                    // Create UserInput with chat messages
                    let userInput = UserInput(chat: messages)
                    
                    // Get generation parameters from options
                    let parameters = GenerateParameters(
                        maxTokens: options?.maximumResponseTokens ?? 2048,
                        temperature: options?.temperature.map { Float($0) } ?? 0.7
                    )
                    
                    // Stream using ModelContainer
                    try await modelContainer.perform { (context: ModelContext) in
                        let input = try await context.processor.prepare(input: userInput)
                        
                        // Use the streaming generate function
                        let stream = try MLXLMCommon.generate(
                            input: input,
                            parameters: parameters,
                            context: context
                        )
                        
                        var accumulatedText = ""
                        
                        for await generation in stream {
                            switch generation {
                            case .chunk(let text):
                                accumulatedText += text
                                // Yield partial response
                                continuation.yield(.response(.init(
                                    assetIDs: [],
                                    segments: [.text(.init(content: accumulatedText))]
                                )))
                            case .info, .toolCall:
                                // Skip non-text generation
                                break
                            }
                        }
                    }
                    
                    continuation.finish()
                } catch {
                    continuation.finish()
                }
            }
        }
    }
    
    // MARK: - Private Helpers
    
    private func convertTranscriptToMessages(_ transcript: Transcript) -> [Chat.Message] {
        var messages: [Chat.Message] = []
        
        // Build conversation from transcript entries
        for entry in transcript {
            switch entry {
            case .instructions(let instructions):
                // Collect system instructions text
                var content = ""
                for segment in instructions.segments {
                    switch segment {
                    case .text(let text):
                        content += text.content
                    default:
                        break
                    }
                }
                if !content.isEmpty {
                    messages.append(.system(content))
                }
                
            case .prompt(let prompt):
                // Collect user prompt text
                var content = ""
                for segment in prompt.segments {
                    switch segment {
                    case .text(let text):
                        content += text.content
                    default:
                        break
                    }
                }
                if !content.isEmpty {
                    messages.append(.user(content))
                }
                
            case .response(let response):
                // Collect assistant response text
                var content = ""
                for segment in response.segments {
                    switch segment {
                    case .text(let text):
                        content += text.content
                    default:
                        break
                    }
                }
                if !content.isEmpty {
                    messages.append(.assistant(content))
                }
                
            case .toolCalls, .toolOutput:
                // Skip tool-related entries for now
                break
            }
        }
        
        // If empty, provide a default user message
        if messages.isEmpty {
            messages.append(.user("Hello"))
        }
        
        return messages
    }
}