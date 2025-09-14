import Foundation
import MLXLMCommon
import MLXLLM
import Tokenizers
import OpenFoundationModels

/// Errors specific to the generation pipeline
enum GenerationPipelineError: Error {
    case missingModelCard
}

struct GenerationPipeline: Sendable {
    
    let executor: MLXExecutor
    let constraints: any ConstraintEngine
    let additionalProcessors: [LogitProcessor]
    
    init(
        executor: MLXExecutor,
        constraints: any ConstraintEngine,
        additionalProcessors: [LogitProcessor] = []
    ) {
        self.executor = executor
        self.constraints = constraints
        self.additionalProcessors = additionalProcessors
    }
    
    func run(
        prompt: String,
        schema: SchemaNode? = nil,
        parameters: GenerateParameters,
        modelCard: (any ModelCard)? = nil
    ) async throws -> String {
        var finalPrompt = prompt
        
        if constraints.mode == .soft, let softPrompt = constraints.softPrompt(for: schema) {
            finalPrompt = prompt + "\n\n" + softPrompt
        }
        
        // Always prepare constraints to enable observation for all modes
        try await executor.withTokenizer { tokenizer in
            try await constraints.prepare(schema: schema, tokenizer: tokenizer, modelCard: modelCard)
        }
        let constraintProcessors = await constraints.logitProcessors()
        
        // Combine constraint processors with additional processors if any
        let finalProcessor: LogitProcessor? = {
            var allProcessors: [LogitProcessor] = []
            
            // Add constraint processors
            allProcessors.append(contentsOf: constraintProcessors)
            
            // Add additional processors (like KeyDetectionLogitProcessor)
            allProcessors.append(contentsOf: additionalProcessors)
            
            // Return appropriate processor
            if allProcessors.isEmpty {
                return nil
            } else if allProcessors.count == 1 {
                return allProcessors.first
            } else {
                // For now, just use the first processor when multiple exist
                // TODO: Implement proper chaining if needed
                return allProcessors.first
            }
        }()
        
        // Execute generation without retry logic
        do {
                
                var raw = ""
                let stream = await executor.executeStream(
                    prompt: finalPrompt,
                    parameters: parameters,
                    logitProcessor: finalProcessor
                )
                
                for try await chunk in stream {
                    raw += chunk
                }
                
                // Log the generated output
                Logger.info("[GenerationPipeline] Generated output (\(raw.count) characters):")
                Logger.info("========== START OUTPUT ==========")
                Logger.info(raw)
                Logger.info("========== END OUTPUT ==========")
                
                // ModelCard is required for proper output processing
                guard let card = modelCard else {
                    throw GenerationPipelineError.missingModelCard
                }
                
                // Process output through ModelCard to extract final content
                let entry = card.generate(from: raw, options: nil)
                let output: String = {
                    switch entry {
                    case .response(let response):
                        return response.segments.compactMap { segment in
                            if case .text(let text) = segment {
                                return text.content
                            }
                            return nil
                        }.joined()
                    default:
                        // For non-response entries, return raw
                        return raw
                    }
                }()
                
                Logger.info("[GenerationPipeline] Extracted output from ModelCard: \(output)")
                
                // No validation needed - constraints are enforced during generation
                return output
                
        } catch {
            throw error
        }
    }
    
    func stream(
        prompt: String,
        schema: SchemaNode? = nil,
        parameters: GenerateParameters,
        modelCard: (any ModelCard)? = nil
    ) -> AsyncThrowingStream<String, Error> {
        AsyncThrowingStream { continuation in
            Task {
                do {
                    // ModelCard is required for proper output processing
                    guard let card = modelCard else {
                        throw GenerationPipelineError.missingModelCard
                    }
                    
                    var finalPrompt = prompt
                    
                    if constraints.mode == .soft, let softPrompt = constraints.softPrompt(for: schema) {
                        finalPrompt = prompt + "\n\n" + softPrompt
                    }
                    
                    // Always prepare constraints to enable observation for all modes
                    try await executor.withTokenizer { tokenizer in
                        try await constraints.prepare(schema: schema, tokenizer: tokenizer, modelCard: card)
                    }
                    
                    let processors = await constraints.logitProcessors()
                    
                    let baseStream = await executor.executeStream(
                        prompt: finalPrompt,
                        parameters: parameters,
                        logitProcessor: processors.first
                    )
                    
                    var buffer = ""
                    
                    for try await chunk in baseStream {
                        buffer += chunk
                        continuation.yield(chunk)
                    }
                    
                    // No validation needed - constraints are enforced during generation
                    
                    continuation.finish()
                    
                } catch {
                    continuation.finish(throwing: error)
                }
            }
        }
    }
    
}

extension MLXExecutor {
    func withTokenizer<T: Sendable>(_ block: @Sendable (any Tokenizer) async throws -> T) async throws -> T {
        guard let container = await getModelContainer() else {
            throw ExecutorError.noModelSet
        }
        
        return try await container.perform { (context: ModelContext) async throws -> T in
            try await block(context.tokenizer)
        }
    }
    
    func getModelContainer() async -> ModelContainer? {
        return modelContainer
    }
}