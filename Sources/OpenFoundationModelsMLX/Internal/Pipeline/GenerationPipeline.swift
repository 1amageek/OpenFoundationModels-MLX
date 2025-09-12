import Foundation
import MLXLMCommon
import MLXLLM
import Tokenizers

struct GenerationPipeline: Sendable {
    
    let executor: MLXExecutor
    let constraints: any ConstraintEngine
    let retryPolicy: RetryPolicy
    let telemetry: any Telemetry
    let additionalProcessors: [LogitProcessor]
    
    init(
        executor: MLXExecutor,
        constraints: any ConstraintEngine,
        retryPolicy: RetryPolicy = .standard,
        telemetry: any Telemetry = NoOpTelemetry(),
        additionalProcessors: [LogitProcessor] = []
    ) {
        self.executor = executor
        self.constraints = constraints
        self.retryPolicy = retryPolicy
        self.telemetry = telemetry
        self.additionalProcessors = additionalProcessors
    }
    
    func run(
        prompt: String,
        schema: SchemaNode? = nil,
        parameters: GenerateParameters
    ) async throws -> String {
        await telemetry.event(.pipelineStarted, metadata: [
            "constraint_mode": constraints.mode.rawValue,
            "has_schema": schema != nil
        ])
        
        var finalPrompt = prompt
        
        if constraints.mode == .soft, let softPrompt = constraints.softPrompt(for: schema) {
            finalPrompt = prompt + "\n\n" + softPrompt
            await telemetry.event(.promptBuilt, metadata: ["soft_constraints_added": true])
        }
        
        // Always prepare constraints to enable observation for all modes
        try await executor.withTokenizer { tokenizer in
            try await constraints.prepare(schema: schema, tokenizer: tokenizer)
        }
        await telemetry.event(.constraintsPrepared, metadata: ["mode": constraints.mode.rawValue])
        
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
                return ChainedLogitProcessor(processors: allProcessors)
            }
        }()
        
        var attempt = 0
        var lastError: Error?
        
        while retryPolicy.shouldRetry(attempt: attempt) {
            if attempt > 0 {
                let backoff = retryPolicy.backoffInterval(for: attempt)
                if backoff > 0 {
                    await telemetry.event(.retryScheduled, metadata: [
                        "attempt": attempt,
                        "backoff_seconds": backoff
                    ])
                    try await Task.sleep(nanoseconds: UInt64(backoff * 1_000_000_000))
                }
            }
            
            attempt += 1
            await telemetry.event(.attemptStarted, metadata: ["attempt": attempt])
            
            do {
                await telemetry.event(.generationStarted, metadata: [:])
                
                // Use streaming to show real-time generation
                Logger.info("[GenerationPipeline] Starting LLM generation...")
                Logger.info("[GenerationPipeline] ========== RAW LLM OUTPUT START ==========")
                
                var raw = ""
                let stream = await executor.executeStream(
                    prompt: finalPrompt,
                    parameters: parameters,
                    logitProcessor: finalProcessor
                )
                
                for try await chunk in stream {
                    raw += chunk
                    // Print for debugging raw output
                    print(chunk, terminator: "")
                    fflush(stdout)
                }
                
                Logger.info("\n[GenerationPipeline] ========== RAW LLM OUTPUT END ==========")
                Logger.info("[GenerationPipeline] Total output length: \(raw.count) characters")
                
                await telemetry.event(.generationCompleted, metadata: [
                    "output_length": raw.count
                ])
                
                // Always extract JSON for structured generation
                let output = extractJSONFromText(raw)
                Logger.info("[GenerationPipeline] Extracted JSON length: \(output.count) characters")
                
                if let validator = constraints.validator() {
                    await telemetry.event(.validationStarted, metadata: [:])
                    
                    switch await validator.validate(output, schema: schema) {
                    case .success:
                        await telemetry.event(.validationPassed, metadata: [:])
                        await telemetry.event(.attemptCompleted, metadata: ["attempt": attempt])
                        await telemetry.event(.pipelineCompleted, metadata: ["attempts": attempt])
                        return output
                        
                    case .failure(let error):
                        await telemetry.event(.validationFailed, metadata: [
                            "error": error.message,
                            "violations": error.violations.count
                        ])
                        
                        lastError = error
                    }
                } else {
                    await telemetry.event(.attemptCompleted, metadata: ["attempt": attempt])
                    await telemetry.event(.pipelineCompleted, metadata: ["attempts": attempt])
                    return output
                }
                
            } catch {
                await telemetry.event(.attemptFailed, metadata: [
                    "attempt": attempt,
                    "error": String(describing: error)
                ])
                lastError = error
            }
        }
        
        await telemetry.event(.pipelineFailed, metadata: [
            "attempts": attempt,
            "last_error": String(describing: lastError ?? ValidationError(message: "Unknown error"))
        ])
        
        throw lastError ?? ValidationError(message: "Failed to generate valid output after \(attempt) attempts")
    }
    
    func stream(
        prompt: String,
        schema: SchemaNode? = nil,
        parameters: GenerateParameters
    ) -> AsyncThrowingStream<String, Error> {
        AsyncThrowingStream { continuation in
            Task {
                do {
                    await telemetry.event(.pipelineStarted, metadata: [
                        "constraint_mode": constraints.mode.rawValue,
                        "has_schema": schema != nil,
                        "streaming": true
                    ])
                    
                    var finalPrompt = prompt
                    
                    if constraints.mode == .soft, let softPrompt = constraints.softPrompt(for: schema) {
                        finalPrompt = prompt + "\n\n" + softPrompt
                    }
                    
                    // Always prepare constraints to enable observation for all modes
                    try await executor.withTokenizer { tokenizer in
                        try await constraints.prepare(schema: schema, tokenizer: tokenizer)
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
                    
                    if let validator = constraints.validator() {
                        // Always extract JSON for structured generation
                        let output = extractJSONFromText(buffer)
                        
                        if case .failure(let error) = await validator.validate(output, schema: schema) {
                            throw error
                        }
                    }
                    
                    continuation.finish()
                    
                } catch {
                    continuation.finish(throwing: error)
                }
            }
        }
    }
    
    private func extractJSONFromText(_ text: String) -> String {
        let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
        
        // Log what we're trying to extract from
        Logger.info("[extractJSONFromText] Input text length: \(trimmed.count)")
        if trimmed.count < 200 {
            Logger.info("[extractJSONFromText] Full text: \(trimmed)")
        } else {
            Logger.info("[extractJSONFromText] First 200 chars: \(trimmed.prefix(200))")
        }
        
        // Return empty string if input is empty
        if trimmed.isEmpty {
            Logger.warning("[extractJSONFromText] Empty input, returning empty string")
            return ""
        }
        
        // Find JSON object
        if let jsonStart = trimmed.range(of: "{") {
            // Search for closing brace after the opening brace
            let searchStartIndex = trimmed.index(after: jsonStart.lowerBound)
            if searchStartIndex < trimmed.endIndex,
               let jsonEnd = trimmed.range(of: "}", options: .backwards, range: searchStartIndex..<trimmed.endIndex) {
                // Use jsonEnd.lowerBound instead of upperBound to avoid index out of bounds
                let endIndex = jsonEnd.lowerBound
                if endIndex >= jsonStart.lowerBound && endIndex < trimmed.endIndex {
                    let extracted = String(trimmed[jsonStart.lowerBound...endIndex])
                    Logger.info("[extractJSONFromText] Found JSON object, length: \(extracted.count)")
                    return extracted
                }
            }
        }
        
        // Find JSON array
        if let jsonStart = trimmed.range(of: "[") {
            // Search for closing bracket after the opening bracket
            let searchStartIndex = trimmed.index(after: jsonStart.lowerBound)
            if searchStartIndex < trimmed.endIndex,
               let jsonEnd = trimmed.range(of: "]", options: .backwards, range: searchStartIndex..<trimmed.endIndex) {
                // Use jsonEnd.lowerBound instead of upperBound to avoid index out of bounds
                let endIndex = jsonEnd.lowerBound
                if endIndex >= jsonStart.lowerBound && endIndex < trimmed.endIndex {
                    let extracted = String(trimmed[jsonStart.lowerBound...endIndex])
                    Logger.info("[extractJSONFromText] Found JSON array, length: \(extracted.count)")
                    return extracted
                }
            }
        }
        
        Logger.warning("[extractJSONFromText] No JSON found, returning trimmed text")
        return trimmed
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