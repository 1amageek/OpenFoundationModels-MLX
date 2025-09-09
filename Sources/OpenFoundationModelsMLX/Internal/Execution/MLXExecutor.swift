import Foundation
import MLX
import MLXLMCommon
import MLXLLM
import Tokenizers

/// MLXExecutor is the lowest-level component that directly interfaces with MLXLLM.
/// It handles pure model execution without any business logic, validation, or constraints.
/// This actor is responsible only for running the model and returning raw results.
public actor MLXExecutor {
    
    // MARK: - Properties
    
    private var modelContainer: ModelContainer?
    private var modelID: String?
    
    // MARK: - Errors
    
    public enum ExecutorError: LocalizedError {
        case noModelSet
        case executionFailed(String)
        
        public var errorDescription: String? {
            switch self {
            case .noModelSet:
                return "No model has been set. Call setModel() with a loaded ModelContainer first."
            case .executionFailed(let reason):
                return "Execution failed: \(reason)"
            }
        }
    }
    
    // MARK: - Model Management
    
    /// Set the model container for execution
    /// - Parameters:
    ///   - container: Pre-loaded ModelContainer from ModelLoader
    ///   - modelID: Identifier for the model (for reference)
    public func setModel(_ container: ModelContainer, modelID: String? = nil) {
        self.modelContainer = container
        self.modelID = modelID
        Logger.info("[MLXExecutor] Model set: \(modelID ?? "unknown")")
    }
    
    /// Clear the current model
    public func clearModel() {
        self.modelContainer = nil
        self.modelID = nil
        Logger.info("[MLXExecutor] Model cleared")
    }
    
    /// Get the current model ID
    public func currentModel() -> String? {
        return modelID
    }
    
    /// Check if a model is loaded
    public func hasModel() -> Bool {
        return modelContainer != nil
    }
    
    // MARK: - Text Generation (Pure Execution)
    
    /// Execute text generation without any constraints or validation
    /// - Parameters:
    ///   - prompt: The input prompt
    ///   - parameters: Generation parameters
    ///   - logitProcessor: Optional logit processor for constraints
    /// - Returns: Generated text
    public func execute(
        prompt: String,
        parameters: GenerateParameters,
        logitProcessor: LogitProcessor? = nil
    ) async throws -> String {
        guard let container = modelContainer else {
            throw ExecutorError.noModelSet
        }
        
        return try await container.perform { (context: ModelContext) async throws -> String in
            // Prepare input
            let userInput = UserInput(prompt: .text(prompt))
            let input = try await context.processor.prepare(input: userInput)
            
            // Generate based on whether we have a logit processor
            if let processor = logitProcessor {
                // Use custom iterator with logit processor
                let sampler = parameters.sampler()
                let iterator = try TokenIterator(
                    input: input,
                    model: context.model,
                    cache: nil,
                    processor: processor,
                    sampler: sampler,
                    prefillStepSize: parameters.prefillStepSize,
                    maxTokens: parameters.maxTokens
                )
                
                // Generate with custom iterator
                let stream = MLXLMCommon.generate(
                    input: input,
                    context: context,
                    iterator: iterator
                )
                
                // Collect output
                var result = ""
                for try await generation in stream {
                    switch generation {
                    case .chunk(let text):
                        result += text
                    case .info, .toolCall:
                        break
                    }
                }
                return result
            } else {
                // Standard generation without constraints
                let output = try MLXLMCommon.generate(
                    input: input,
                    parameters: parameters,
                    context: context,
                    didGenerate: { (tokens: [Int]) in
                        return .more
                    }
                )
                return output.output
            }
        }
    }
    
    /// Execute streaming text generation
    /// - Parameters:
    ///   - prompt: The input prompt
    ///   - parameters: Generation parameters
    ///   - logitProcessor: Optional logit processor for constraints
    /// - Returns: Stream of generated text chunks
    public func executeStream(
        prompt: String,
        parameters: GenerateParameters,
        logitProcessor: LogitProcessor? = nil
    ) -> AsyncThrowingStream<String, Error> {
        AsyncThrowingStream { continuation in
            Task {
                do {
                    guard let container = modelContainer else {
                        throw ExecutorError.noModelSet
                    }
                    
                    try await container.perform { (context: ModelContext) async throws in
                        let userInput = UserInput(prompt: .text(prompt))
                        let input = try await context.processor.prepare(input: userInput)
                        
                        let stream: AsyncStream<Generation>
                        
                        if let processor = logitProcessor {
                            // Use custom iterator with logit processor
                            let sampler = parameters.sampler()
                            let iterator = try TokenIterator(
                                input: input,
                                model: context.model,
                                cache: nil,
                                processor: processor,
                                sampler: sampler,
                                prefillStepSize: parameters.prefillStepSize,
                                maxTokens: parameters.maxTokens
                            )
                            
                            stream = MLXLMCommon.generate(
                                input: input,
                                context: context,
                                iterator: iterator
                            )
                        } else {
                            // Standard streaming without constraints
                            stream = try MLXLMCommon.generate(
                                input: input,
                                parameters: parameters,
                                context: context
                            )
                        }
                        
                        for try await generation in stream {
                            switch generation {
                            case .chunk(let text):
                                continuation.yield(text)
                            case .info, .toolCall:
                                break
                            }
                        }
                        
                        continuation.finish()
                    }
                } catch {
                    continuation.finish(throwing: error)
                }
            }
        }
    }
    
    // MARK: - Model Context Access
    
    /// Get access to the model container for ADAPT processing
    /// - Returns: The model container or nil if no model is set
    public func getModelContainer() -> ModelContainer? {
        return modelContainer
    }
}