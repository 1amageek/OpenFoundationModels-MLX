import Foundation
import MLX
import MLXLMCommon
import MLXLLM
import Tokenizers
import Hub
import Darwin

// Cache wrapper to handle Sendable requirements
private final class ModelCacheWrapper: @unchecked Sendable {
    let cache = NSCache<NSString, AnyObject>()
}

// MLXBackend integrates with MLXLLM to provide constrained text generation
// using TokenTrie-based Schema-Constrained Decoding (SCD)
actor MLXBackend {
    // Memory size helpers
    public enum MemorySize {
        public static func MB(_ value: Int) -> Int {
            return value * 1024 * 1024
        }
        
        public static func GB(_ value: Int) -> Int {
            return value * 1024 * 1024 * 1024
        }
    }
    
    // Model management - shared across all instances for efficiency
    private static let modelCacheWrapper = ModelCacheWrapper()
    private var currentModelContainer: ModelContainer?
    private var currentModelID: String?
    private let hubApi: HubApi
    
    // Initialize with required model ID
    // maxCacheMemory: Maximum cache memory in bytes (default: 2GB)
    init(modelID: String, maxCacheMemory: Int = MemorySize.GB(2)) async throws {
        // Use the default Hub API which manages downloads to ~/Library/Caches/huggingface
        self.hubApi = HubApi()
        self.currentModelID = modelID
        
        // Configure cache limits (one-time setup)
        if MLXBackend.modelCacheWrapper.cache.countLimit == 0 {
            MLXBackend.modelCacheWrapper.cache.countLimit = 3  // Keep max 3 models in memory
            MLXBackend.modelCacheWrapper.cache.totalCostLimit = maxCacheMemory  // Set memory limit in bytes
        }
        
        // Load the model
        try await loadModel(modelID)
    }
    
    // MARK: - Model Management
    
    /// Load a model by ID (e.g., "openai/gpt-oss-20b" or "mlx-community/llama-3-8b-4bit")
    public func loadModel(_ modelID: String) async throws {
        let cacheKey = modelID as NSString
        
        // Check memory cache first
        if let cached = MLXBackend.modelCacheWrapper.cache.object(forKey: cacheKey) as? ModelContainer {
            currentModelContainer = cached
            currentModelID = modelID
            Logger.debug("[MLXBackend] Using cached model: \(modelID)")
            return
        }
        
        Logger.info("[MLXBackend] Loading model: \(modelID)")
        
        // Create model configuration
        let config = ModelConfiguration(id: modelID)
        
        // Load using MLXLMCommon's loadModelContainer
        // This handles download, caching, and initialization automatically
        let container = try await loadModelContainer(
            hub: hubApi,
            configuration: config
        ) { progress in
            let percentage = Int(progress.fractionCompleted * 100)
            if percentage % 10 == 0 {  // Log every 10%
                Logger.verbose("[MLXBackend] Loading \(modelID): \(percentage)%")
            }
        }
        
        // Store in cache with a fixed cost - NSCache will respect totalCostLimit
        // Using 1GB as default cost per model for simplicity
        let modelCost = 1024 * 1024 * 1024  // 1GB per model
        MLXBackend.modelCacheWrapper.cache.setObject(container as AnyObject, forKey: cacheKey, cost: modelCost)
        currentModelContainer = container
        currentModelID = modelID
        Logger.info("[MLXBackend] Model loaded successfully: \(modelID)")
    }
    
    /// Get the currently active model
    public func currentModel() -> String? {
        return currentModelID
    }
    
    /// List available models in cache
    public func cachedModels() -> [String] {
        // Note: NSCache doesn't provide enumeration, so we track separately if needed
        return currentModelID != nil ? [currentModelID!] : []
    }
    
    /// Unload a specific model or the current model from memory
    public func unloadModel(_ modelID: String? = nil) async {
        let targetID = modelID ?? currentModelID
        guard let id = targetID else {
            Logger.debug("[MLXBackend] No model to unload")
            return
        }
        
        // Remove from cache
        MLXBackend.modelCacheWrapper.cache.removeObject(forKey: id as NSString)
        Logger.info("[MLXBackend] Unloaded model from cache: \(id)")
        
        // Clear current references if it's the active model
        if id == currentModelID {
            currentModelContainer = nil
            currentModelID = nil
            Logger.info("[MLXBackend] Cleared current model references")
        }
    }
    
    /// Clear all models from memory
    public func clearAllModels() async {
        MLXBackend.modelCacheWrapper.cache.removeAllObjects()
        currentModelContainer = nil
        currentModelID = nil
        Logger.info("[MLXBackend] Cleared all models from memory")
    }
    
    /// Check memory pressure and automatically unload models if needed
    public func handleMemoryPressure(maxResidentBytes: UInt64 = 1_500_000_000) async {
        let rss = currentResidentMemory()
        
        if rss > maxResidentBytes {
            Logger.warning("[MLXBackend] High process RSS \(rss) > \(maxResidentBytes). Trimming cache.")
            
            // Keep only the current model, clear others
            if currentModelID != nil {
                // NSCache will handle eviction, but we force clear for immediate effect
                MLXBackend.modelCacheWrapper.cache.countLimit = 1
            } else {
                await clearAllModels()
            }
        }
    }
    
    /// Get current process resident memory in bytes
    private func currentResidentMemory() -> UInt64 {
        var info = mach_task_basic_info()
        var count = mach_msg_type_number_t(MemoryLayout<mach_task_basic_info>.size / MemoryLayout<integer_t>.size)
        
        let kerr = withUnsafeMutablePointer(to: &info) {
            $0.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
                task_info(mach_task_self_, task_flavor_t(MACH_TASK_BASIC_INFO), $0, &count)
            }
        }
        
        return kerr == KERN_SUCCESS ? UInt64(info.resident_size) : 0
    }
    
    // MARK: - Text Generation
    
    func generateText(prompt: String, sampling: SamplingParameters) async throws -> String {
        guard let container = currentModelContainer else {
            await sharedMetrics.recordError(MLXBackendError.noModelLoaded)
            throw MLXBackendError.noModelLoaded
        }
        
        return try await container.perform { (context: ModelContext) async throws -> String in
            // Prepare input
            let userInput = UserInput(prompt: .text(prompt))
            let input = try await context.processor.prepare(input: userInput)
            
            // Configure generation parameters
            let parameters = GenerateParameters(
                maxTokens: sampling.maxTokens ?? 1024,
                temperature: Float(sampling.temperature ?? 0.7),
                topP: Float(sampling.topP ?? 1.0)
            )
            
            // Generate text
            var result = ""
            let stream = try MLXLMCommon.generate(
                input: input,
                parameters: parameters,
                context: context
            )
            
            for await generation in stream {
                switch generation {
                case .chunk(let text):
                    result += text
                case .info(let info):
                    Logger.debug("[MLXBackend] Generation complete: \(info.tokensPerSecond) tokens/s")
                case .toolCall:
                    break
                }
            }
            
            return result
        }
    }
    
    func generateTextConstrained(
        prompt: String,
        sampling: SamplingParameters,
        schema: SchemaMeta
    ) async throws -> String {
        guard let container = currentModelContainer else {
            throw MLXBackendError.noModelLoaded
        }
        
        let result = try await attemptConstrainedGeneration(
            prompt: prompt,
            sampling: sampling,
            schema: schema,
            container: container
        )
        
        // Validate the result
        let validator = JSONValidator(allowExtraKeys: false, enableSnap: true)
        guard validator.validate(text: result, schema: schema) else {
            throw JSONGenerationError.schemaViolation(
                reason: "Generated JSON does not match schema"
            )
        }
        
        return result
    }
    
    private func attemptConstrainedGeneration(
        prompt: String,
        sampling: SamplingParameters,
        schema: SchemaMeta,
        container: ModelContainer
    ) async throws -> String {
        return try await container.perform { (context: ModelContext) async throws -> String in
            // Prepare input
            let userInput = UserInput(prompt: .text(prompt))
            let input = try await context.processor.prepare(input: userInput)
            
            // Create constrained logit processor and clear any previous errors
            let constraintProcessor = TokenTrieLogitProcessor(
                schema: schema,
                tokenizer: context.tokenizer
            )
            constraintProcessor.clearError()
            
            // Configure generation parameters
            // See note above regarding card vs. sampling flows.
            let parameters = GenerateParameters(
                maxTokens: sampling.maxTokens ?? 1024,
                temperature: Float(sampling.temperature ?? 0.7),
                topP: Float(sampling.topP ?? 1.0)
            )
            
            // Create custom token iterator with constraint processor
            let sampler = parameters.sampler()
            let iterator = try TokenIterator(
                input: input,
                model: context.model,
                cache: nil,
                processor: constraintProcessor,
                sampler: sampler,
                prefillStepSize: parameters.prefillStepSize,
                maxTokens: parameters.maxTokens
            )
            
            // Create base generation stream
            let baseStream = MLXLMCommon.generate(
                input: input,
                context: context,
                iterator: iterator
            )
            
            // Wrap with AbortableGenerator for error polling
            let abortableGen = AbortableGenerator(processor: constraintProcessor)
            let stream = abortableGen.generate(baseStream: baseStream)
            
            // Generate with immediate abort on errors
            var result = ""
            var tokenCount = 0
            
            for try await generation in stream {
                switch generation {
                case .chunk(let text):
                    result += text
                    tokenCount += 1
                    
                    // Additional safety check every few tokens
                    if tokenCount % 5 == 0 && constraintProcessor.hasError() {
                        let error = constraintProcessor.getLastError()!
                        Logger.warning("[MLXBackend] Error detected after \(tokenCount) tokens: \(error)")
                        throw error
                    }
                    
                case .info(let info):
                    Logger.debug("[MLXBackend] Constrained generation: \(info.tokensPerSecond) tokens/s")
                    
                case .toolCall:
                    break
                }
            }
            
            // Final error check
            if let error = constraintProcessor.getLastError() {
                Logger.warning("[MLXBackend] Final error check failed: \(error)")
                throw error
            }
            
            return result
        }
    }

    // Constrained generation using explicit GenerateParameters from ModelCard
    func generateTextConstrained(
        prompt: String,
        parameters: GenerateParameters,
        schema: SchemaMeta
    ) async throws -> String {
        guard let container = currentModelContainer else {
            throw MLXBackendError.noModelLoaded
        }
        
        let result = try await attemptConstrainedGenerationWithParams(
            prompt: prompt,
            parameters: parameters,
            schema: schema,
            container: container
        )
        
        // Validate the result
        let validator = JSONValidator(allowExtraKeys: false, enableSnap: true)
        guard validator.validate(text: result, schema: schema) else {
            throw JSONGenerationError.schemaViolation(
                reason: "Generated JSON does not match schema"
            )
        }
        
        return result
    }
    
    private func attemptConstrainedGenerationWithParams(
        prompt: String,
        parameters: GenerateParameters,
        schema: SchemaMeta,
        container: ModelContainer
    ) async throws -> String {
        return try await container.perform { (context: ModelContext) async throws -> String in
            let userInput = UserInput(prompt: .text(prompt))
            let input = try await context.processor.prepare(input: userInput)

            // Setup constraints processor with error clearing
            let constraintProcessor = TokenTrieLogitProcessor(
                schema: schema,
                tokenizer: context.tokenizer
            )
            constraintProcessor.clearError()

            // Create custom token iterator with constraint processor and provided parameters
            let sampler = parameters.sampler()
            let iterator = try TokenIterator(
                input: input,
                model: context.model,
                cache: nil,
                processor: constraintProcessor,
                sampler: sampler,
                prefillStepSize: parameters.prefillStepSize,
                maxTokens: parameters.maxTokens
            )

            // Create base stream and wrap with AbortableGenerator
            let baseStream = MLXLMCommon.generate(
                input: input,
                context: context,
                iterator: iterator
            )
            
            let abortableGen = AbortableGenerator(processor: constraintProcessor)
            let stream = abortableGen.generate(baseStream: baseStream)
            
            var result = ""
            for try await generation in stream {
                if case .chunk(let text) = generation { 
                    result += text 
                }
            }
            
            // Final error check
            if let error = constraintProcessor.getLastError() {
                throw error
            }
            
            return result
        }
    }
    
    // MARK: - Streaming Generation
    
    func streamText(prompt: String, sampling: SamplingParameters) -> AsyncThrowingStream<String, Error> {
        AsyncThrowingStream { continuation in
            let task = Task {
                guard let container = currentModelContainer else {
                    continuation.finish(throwing: MLXBackendError.noModelLoaded)
                    return
                }
                
                do {
                    try await container.perform { (context: ModelContext) async throws in
                        let userInput = UserInput(prompt: .text(prompt))
                        let input = try await context.processor.prepare(input: userInput)
                        // See note above regarding card vs. sampling flows.
                        let parameters = GenerateParameters(
                            maxTokens: sampling.maxTokens ?? 1024,
                            temperature: Float(sampling.temperature ?? 0.7),
                            topP: Float(sampling.topP ?? 1.0)
                        )
                        
                        let stream = try MLXLMCommon.generate(
                            input: input,
                            parameters: parameters,
                            context: context
                        )
                        
                        for await generation in stream {
                            if Task.isCancelled {
                                break
                            }
                            if case .chunk(let text) = generation {
                                continuation.yield(text)
                            }
                        }
                        
                        continuation.finish()
                    }
                } catch {
                    if !Task.isCancelled {
                        continuation.finish(throwing: error)
                    }
                }
            }
            continuation.onTermination = { _ in
                task.cancel()
            }
        }
    }
    
    func streamTextConstrained(
        prompt: String,
        sampling: SamplingParameters,
        schema: SchemaMeta
    ) -> AsyncThrowingStream<String, Error> {
        AsyncThrowingStream { continuation in
            let task = Task {
                guard let container = currentModelContainer else {
                    continuation.finish(throwing: MLXBackendError.noModelLoaded)
                    return
                }
                
                do {
                    try await container.perform { (context: ModelContext) async throws in
                        let userInput = UserInput(prompt: .text(prompt))
                        let input = try await context.processor.prepare(input: userInput)
                        
                        // Setup constraints with error clearing
                        let constraintProcessor = TokenTrieLogitProcessor(
                            schema: schema,
                            tokenizer: context.tokenizer
                        )
                        constraintProcessor.clearError()
                        
                        // Configure generation parameters
                        // See note above regarding card vs. sampling flows.
                        let parameters = GenerateParameters(
                            maxTokens: sampling.maxTokens ?? 1024,
                            temperature: Float(sampling.temperature ?? 0.7),
                            topP: Float(sampling.topP ?? 1.0)
                        )
                        
                        // Create custom token iterator with constraint processor
                        let sampler = parameters.sampler()
                        let iterator = try TokenIterator(
                            input: input,
                            model: context.model,
                            cache: nil,
                            processor: constraintProcessor,
                            sampler: sampler,
                            prefillStepSize: parameters.prefillStepSize,
                            maxTokens: parameters.maxTokens
                        )
                        
                        // Create base stream and wrap with AbortableGenerator
                        let baseStream = MLXLMCommon.generate(
                            input: input,
                            context: context,
                            iterator: iterator
                        )
                        
                        let abortableGen = AbortableGenerator(processor: constraintProcessor)
                        let stream = abortableGen.generate(baseStream: baseStream)
                        
                        for try await generation in stream {
                            if Task.isCancelled {
                                break
                            }
                            
                            if case .chunk(let text) = generation {
                                // Check for fatal errors before yielding
                                if constraintProcessor.hasFatalError() {
                                    throw constraintProcessor.getLastError()!
                                }
                                continuation.yield(text)
                            }
                        }
                        
                        // Final error check
                        if let error = constraintProcessor.getLastError() {
                            throw error
                        }
                        
                        continuation.finish()
                    }
                } catch {
                    if !Task.isCancelled {
                        continuation.finish(throwing: error)
                    }
                }
            }
            continuation.onTermination = { _ in
                task.cancel()
            }
        }
    }

    func streamTextConstrained(
        prompt: String,
        parameters: GenerateParameters,
        schema: SchemaMeta
    ) -> AsyncThrowingStream<String, Error> {
        AsyncThrowingStream { continuation in
            let task = Task {
                guard let container = currentModelContainer else {
                    continuation.finish(throwing: MLXBackendError.noModelLoaded)
                    return
                }
                do {
                    try await container.perform { (context: ModelContext) async throws in
                        let userInput = UserInput(prompt: .text(prompt))
                        let input = try await context.processor.prepare(input: userInput)

                        let constraintProcessor = TokenTrieLogitProcessor(
                            schema: schema,
                            tokenizer: context.tokenizer
                        )
                        constraintProcessor.clearError()
                        
                        let sampler = parameters.sampler()
                        let iterator = try TokenIterator(
                            input: input,
                            model: context.model,
                            cache: nil,
                            processor: constraintProcessor,
                            sampler: sampler,
                            prefillStepSize: parameters.prefillStepSize,
                            maxTokens: parameters.maxTokens
                        )
                        
                        // Create base stream and wrap with AbortableGenerator
                        let baseStream = MLXLMCommon.generate(
                            input: input,
                            context: context,
                            iterator: iterator
                        )
                        
                        let abortableGen = AbortableGenerator(processor: constraintProcessor)
                        let stream = abortableGen.generate(baseStream: baseStream)
                        
                        for try await gen in stream {
                            if Task.isCancelled {
                                break
                            }
                            if case .chunk(let text) = gen { 
                                // Check for fatal errors before yielding
                                if constraintProcessor.hasFatalError() {
                                    throw constraintProcessor.getLastError()!
                                }
                                continuation.yield(text) 
                            }
                        }
                        
                        // Final error check
                        if let error = constraintProcessor.getLastError() {
                            throw error
                        }
                        
                        continuation.finish()
                    }
                } catch {
                    if !Task.isCancelled {
                        continuation.finish(throwing: error)
                    }
                }
            }
            continuation.onTermination = { _ in
                task.cancel()
            }
        }
    }
}

// MARK: - Extensions

extension MLXBackend {
    // Direct parameterized generation (ModelCard provided parameters). MLX does not modify or fallback.
    func generateText(prompt: String, parameters: GenerateParameters) async throws -> String {
        guard let container = currentModelContainer else {
            throw MLXBackendError.noModelLoaded
        }
        return try await container.perform { (context: ModelContext) async throws -> String in
            let userInput = UserInput(prompt: .text(prompt))
            let input = try await context.processor.prepare(input: userInput)
            var result = ""
            let stream = try MLXLMCommon.generate(
                input: input,
                parameters: parameters,
                context: context
            )
            for await generation in stream {
                if case .chunk(let text) = generation { result += text }
            }
            return result
        }
    }

    func streamText(prompt: String, parameters: GenerateParameters) -> AsyncThrowingStream<String, Error> {
        AsyncThrowingStream { continuation in
            let task = Task {
                guard let container = currentModelContainer else {
                    continuation.finish(throwing: MLXBackendError.noModelLoaded)
                    return
                }
                do {
                    try await container.perform { (context: ModelContext) async throws in
                        let userInput = UserInput(prompt: .text(prompt))
                        let input = try await context.processor.prepare(input: userInput)
                        let stream = try MLXLMCommon.generate(
                            input: input,
                            parameters: parameters,
                            context: context
                        )
                        for await gen in stream {
                            if Task.isCancelled {
                                break
                            }
                            if case .chunk(let text) = gen { continuation.yield(text) }
                        }
                        continuation.finish()
                    }
                } catch {
                    if !Task.isCancelled {
                        continuation.finish(throwing: error)
                    }
                }
            }
            continuation.onTermination = { _ in
                task.cancel()
            }
        }
    }
    /// Generate with automatic retry on schema violations
    /// Tracks abort positions and adjusts strategy based on failure patterns
    func generateTextConstrainedWithRetry(
        prompt: String,
        sampling: SamplingParameters,
        schema: SchemaMeta,
        maxRetries: Int = 3
    ) async throws -> String {
        var lastError: Error?
        var abortPositions: [Int] = []  // Track where each attempt was aborted
        
        // Base temperature for variation strategy
        let baseTemperature = sampling.temperature ?? 0.7
        let temperatureIncrement = 0.1  // Increase by 0.1 each retry
        
        for attempt in 1...maxRetries {
            // Vary temperature on retry (but not if seed is set for deterministic generation)
            var adjustedSampling = sampling
            if sampling.seed == nil && attempt > 1 {
                // Check if we're consistently failing at the same position
                if abortPositions.count >= 2 {
                    let lastTwo = abortPositions.suffix(2)
                    let similarPositions = lastTwo.allSatisfy { pos in 
                        abs(pos - abortPositions.last!) < 3
                    }
                    
                    if similarPositions {
                        // Increase temperature more aggressively if stuck at same position
                        let newTemp = min(baseTemperature + Double(attempt) * temperatureIncrement * 1.5, 1.5)
                        adjustedSampling.temperature = newTemp
                        Logger.info("[MLXBackend] Stuck at position ~\(abortPositions.last!), increasing temperature to \(newTemp)")
                    } else {
                        // Normal temperature increase
                        let newTemp = min(baseTemperature + Double(attempt - 1) * temperatureIncrement, 1.5)
                        adjustedSampling.temperature = newTemp
                        Logger.debug("[MLXBackend] Retry \(attempt) with temperature: \(newTemp)")
                    }
                } else {
                    // Normal temperature increase for early retries
                    let newTemp = min(baseTemperature + Double(attempt - 1) * temperatureIncrement, 1.5)
                    adjustedSampling.temperature = newTemp
                    Logger.debug("[MLXBackend] Retry \(attempt) with temperature: \(newTemp)")
                }
            }
            
            do {
                let result = try await generateTextConstrained(
                    prompt: prompt,
                    sampling: adjustedSampling,
                    schema: schema
                )
                
                // Validate the generated JSON
                if validateJSON(result, schema: schema) {
                    if !abortPositions.isEmpty {
                        Logger.info("[MLXBackend] Success after \(attempt) attempts (previous aborts at: \(abortPositions))")
                    }
                    return result
                } else {
                    Logger.warning("[MLXBackend] Attempt \(attempt): Schema validation failed, retrying...")
                    lastError = MLXBackendError.schemaViolation
                }
            } catch let jsonError as JSONGenerationError {
                // Track abort position if available
                switch jsonError {
                case .noValidTokens(_, let position):
                    abortPositions.append(position)
                    Logger.warning("[MLXBackend] Attempt \(attempt) aborted at token position \(position)")
                case .invalidTokenSelected(_, _, _):
                    // Token-level error, track as early abort
                    abortPositions.append(0)
                    Logger.warning("[MLXBackend] Attempt \(attempt) aborted early due to invalid token")
                default:
                    Logger.warning("[MLXBackend] Attempt \(attempt) failed: \(jsonError)")
                }
                lastError = jsonError
            } catch {
                Logger.warning("[MLXBackend] Attempt \(attempt) failed: \(error.localizedDescription)")
                lastError = error
                
                // Don't retry on certain terminal errors
                if error is CancellationError {
                    throw error
                }
            }
        }
        
        throw lastError ?? MLXBackendError.maxRetriesExceeded
    }
    
    private func validateJSON(_ text: String, schema: SchemaMeta) -> Bool {
        // Use unified validator (Snap enabled, extra keys disallowed)
        JSONValidator(allowExtraKeys: false, enableSnap: true).validate(text: text, schema: schema)
    }
}

// MARK: - Convenience Methods

extension MLXBackend {
    /// Simple JSON generation with schema
    func generateJSON(
        prompt: String,
        keys: [String],
        required: [String] = [],
        temperature: Float = 0.7
    ) async throws -> [String: Any] {
        let schema = SchemaMeta(keys: keys, required: required)
        let sampling = SamplingParameters(
            temperature: Double(temperature),
            maxTokens: 500
        )
        
        let jsonString = try await generateTextConstrainedWithRetry(
            prompt: prompt,
            sampling: sampling,
            schema: schema
        )
        
        guard let data = jsonString.data(using: String.Encoding.utf8),
              let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            throw MLXBackendError.invalidJSON
        }
        
        return json
    }
}

// MARK: - Errors

enum MLXBackendError: LocalizedError {
    case noModelLoaded
    case modelLoadFailed(String)
    case schemaViolation
    case invalidJSON
    case maxRetriesExceeded
    case insufficientMemory
    
    var errorDescription: String? {
        switch self {
        case .noModelLoaded:
            return "No model is currently loaded. Call loadModel() first."
        case .modelLoadFailed(let model):
            return "Failed to load model: \(model)"
        case .schemaViolation:
            return "Generated JSON does not conform to schema"
        case .invalidJSON:
            return "Generated text is not valid JSON"
        case .maxRetriesExceeded:
            return "Maximum retry attempts exceeded"
        case .insufficientMemory:
            return "Insufficient memory to load model"
        }
    }
}
