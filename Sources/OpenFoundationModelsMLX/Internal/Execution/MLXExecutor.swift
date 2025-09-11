import Foundation
import MLX
import MLXLMCommon
import MLXLLM
import Tokenizers

/// MLXExecutor is the lowest-level component that directly interfaces with MLXLLM.
/// It handles pure model execution without any business logic, validation, or constraints.
/// This actor is responsible only for running the model and returning raw results.
public actor MLXExecutor {
    private var modelContainer: ModelContainer?
    private var modelID: String?
    
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
        
        Logger.info("[MLXExecutor] Prompt being sent to LLM:")
        Logger.info("================== PROMPT START ==================")
        Logger.info(prompt)
        Logger.info("================== PROMPT END ====================")
        
        return try await container.perform { (context: ModelContext) async throws -> String in
            let userInput = UserInput(prompt: .text(prompt))
            let input = try await context.processor.prepare(input: userInput)
            
            if let processor = logitProcessor {
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
                
                let baseStream = MLXLMCommon.generate(
                    input: input,
                    context: context,
                    iterator: iterator
                )
                
                if let errorCheckable = processor as? ErrorCheckable {
                    let abortor = AbortableGenerator(processor: errorCheckable)
                    let stream = abortor.generate(baseStream: baseStream)
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
                    var result = ""
                    for await generation in baseStream {
                        switch generation {
                        case .chunk(let text):
                            result += text
                        case .info, .toolCall:
                            break
                        }
                    }
                    return result
                }
            } else {
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
            let task = Task {
                do {
                    print("🔥 [DEBUG] MLXExecutor.executeStream: Task started")
                    try Task.checkCancellation()
                    guard let container = modelContainer else {
                        throw ExecutorError.noModelSet
                    }
                    
                    Logger.info("[MLXExecutor] Prompt being sent to LLM (streaming):")
                    Logger.info("================== PROMPT START ==================")
                    Logger.info(prompt)
                    Logger.info("================== PROMPT END ====================")
                    
                    try await container.perform { (context: ModelContext) async throws in
                        try Task.checkCancellation()
                        print("🔥 [DEBUG] MLXExecutor: Starting stream processing")
                        let userInput = UserInput(prompt: .text(prompt))
                        let input = try await context.processor.prepare(input: userInput)
                        
                        let stream: AsyncThrowingStream<Generation, Error>
                        
                        if let processor = logitProcessor {
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
                            
                            let baseStream = MLXLMCommon.generate(
                                input: input,
                                context: context,
                                iterator: iterator
                            )
                            
                            if let errorCheckable = processor as? ErrorCheckable {
                                let abortor = AbortableGenerator(processor: errorCheckable)
                                stream = abortor.generate(baseStream: baseStream)
                            } else {
                                stream = AsyncThrowingStream { continuation in
                                    let innerTask = Task {
                                        do {
                                            for await generation in baseStream {
                                                try Task.checkCancellation()
                                                continuation.yield(generation)
                                            }
                                            continuation.finish()
                                        } catch {
                                            continuation.finish(throwing: error)
                                        }
                                    }
                                    continuation.onTermination = { _ in
                                        innerTask.cancel()
                                    }
                                }
                            }
                        } else {
                            let baseStream = try MLXLMCommon.generate(
                                input: input,
                                parameters: parameters,
                                context: context
                            )
                            stream = AsyncThrowingStream { continuation in
                                let innerTask = Task {
                                    do {
                                        for await generation in baseStream {
                                            try Task.checkCancellation()
                                            continuation.yield(generation)
                                        }
                                        continuation.finish()
                                    } catch {
                                        continuation.finish(throwing: error)
                                    }
                                }
                                continuation.onTermination = { _ in
                                    innerTask.cancel()
                                }
                            }
                        }
                        
                        for try await generation in stream {
                            try Task.checkCancellation()
                            switch generation {
                            case .chunk(let text):
                                continuation.yield(text)
                            case .info, .toolCall:
                                break
                            }
                        }
                        
                        print("🔥 [DEBUG] MLXExecutor: Stream processing completed")
                        continuation.finish()
                    }
                } catch is CancellationError {
                    print("🔥 [DEBUG] MLXExecutor.executeStream: Task cancelled")
                    Stream().synchronize()
                    continuation.finish(throwing: CancellationError())
                } catch {
                    print("🔥 [DEBUG] MLXExecutor.executeStream: Error at \(Date()): \(error)")
                    print("🔥 [DEBUG] MLXExecutor.executeStream: Force synchronizing MLX operations")
                    Stream().synchronize()
                    print("🔥 [DEBUG] MLXExecutor.executeStream: Finishing continuation with error")
                    continuation.finish(throwing: error)
                    print("🔥 [DEBUG] MLXExecutor.executeStream: Continuation finished")
                }
            }
            
            continuation.onTermination = { @Sendable _ in
                print("🔥 [DEBUG] MLXExecutor.executeStream: onTermination called, cancelling task")
                task.cancel()
            }
        }
    }
    /// Get access to the model container for ADAPT processing
    /// - Returns: The model container or nil if no model is set
    public func getModelContainer() -> ModelContainer? {
        return modelContainer
    }
    
    /// Safely access tokenizer without nested perform calls
    /// - Parameter body: Closure that receives the tokenizer
    /// - Returns: The result of the body closure
    public func withTokenizer<T: Sendable>(_ body: @Sendable (any Tokenizer) throws -> T) async throws -> T {
        guard let container = modelContainer else {
            throw ExecutorError.noModelSet
        }
        
        return try await container.perform { (context: ModelContext) in
            try body(context.tokenizer)
        }
    }
}