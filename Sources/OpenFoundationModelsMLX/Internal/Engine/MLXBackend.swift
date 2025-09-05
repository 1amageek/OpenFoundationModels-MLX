import Foundation
import MLX
import MLXLMCommon
import MLXLLM
import Tokenizers
import Hub

// MLXBackend integrates with MLXLLM to provide constrained text generation
// using TokenTrie-based Schema-Constrained Decoding (SCD)
actor MLXBackend {
    // Model management
    private var modelCache = NSCache<NSString, AnyObject>()
    private var currentModelContainer: ModelContainer?
    private var currentModelID: String?
    private let hubApi: HubApi
    
    // Initialize with optional model ID for auto-loading
    init(modelID: String? = nil) async throws {
        // Use the default Hub API which manages downloads to ~/Library/Caches/huggingface
        self.hubApi = HubApi()
        
        // Configure cache limits
        modelCache.countLimit = 3  // Keep max 3 models in memory
        
        // Load initial model if provided
        if let modelID = modelID {
            try await loadModel(modelID)
        }
    }
    
    // MARK: - Model Management
    
    /// Load a model by ID (e.g., "openai/gpt-oss-20b" or "mlx-community/llama-3-8b-4bit")
    public func loadModel(_ modelID: String) async throws {
        let cacheKey = modelID as NSString
        
        // Check memory cache first
        if let cached = modelCache.object(forKey: cacheKey) as? ModelContainer {
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
        
        // Store in cache and set as current
        modelCache.setObject(container as AnyObject, forKey: cacheKey)
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
    
    // MARK: - Text Generation
    
    func generateText(prompt: String, sampling: SamplingParameters) async throws -> String {
        guard let container = currentModelContainer else {
            throw MLXBackendError.noModelLoaded
        }
        
        return try await container.perform { (context: ModelContext) async throws -> String in
            // Prepare input
            let userInput = UserInput(prompt: .text(prompt))
            let input = try await context.processor.prepare(input: userInput)
            
            // Configure generation parameters
            let parameters = GenerateParameters(
                maxTokens: sampling.maxTokens,
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
        
        return try await container.perform { (context: ModelContext) async throws -> String in
            // Prepare input
            let userInput = UserInput(prompt: .text(prompt))
            let input = try await context.processor.prepare(input: userInput)
            
            // Create constrained logit processor
            let constraintProcessor = TokenTrieLogitProcessor(
                schema: schema,
                tokenizer: context.tokenizer
            )
            
            // Configure generation parameters
            let parameters = GenerateParameters(
                maxTokens: sampling.maxTokens,
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
            
            // Generate with constraints using the custom iterator
            var result = ""
            let stream = MLXLMCommon.generate(
                input: input,
                context: context,
                iterator: iterator
            )
            
            for await generation in stream {
                switch generation {
                case .chunk(let text):
                    result += text
                case .info(let info):
                    Logger.debug("[MLXBackend] Constrained generation complete: \(info.tokensPerSecond) tokens/s")
                case .toolCall:
                    break
                }
            }
            
            return result
        }
    }
    
    // MARK: - Streaming Generation
    
    func streamText(prompt: String, sampling: SamplingParameters) -> AsyncThrowingStream<String, Error> {
        AsyncThrowingStream { continuation in
            Task {
                guard let container = currentModelContainer else {
                    continuation.finish(throwing: MLXBackendError.noModelLoaded)
                    return
                }
                
                do {
                    try await container.perform { (context: ModelContext) async throws in
                        let userInput = UserInput(prompt: .text(prompt))
                        let input = try await context.processor.prepare(input: userInput)
                        let parameters = GenerateParameters(
                            maxTokens: sampling.maxTokens,
                            temperature: Float(sampling.temperature ?? 0.7),
                            topP: Float(sampling.topP ?? 1.0)
                        )
                        
                        let stream = try MLXLMCommon.generate(
                            input: input,
                            parameters: parameters,
                            context: context
                        )
                        
                        for await generation in stream {
                            if case .chunk(let text) = generation {
                                continuation.yield(text)
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
    
    func streamTextConstrained(
        prompt: String,
        sampling: SamplingParameters,
        schema: SchemaMeta
    ) -> AsyncThrowingStream<String, Error> {
        AsyncThrowingStream { continuation in
            Task {
                guard let container = currentModelContainer else {
                    continuation.finish(throwing: MLXBackendError.noModelLoaded)
                    return
                }
                
                do {
                    try await container.perform { (context: ModelContext) async throws in
                        let userInput = UserInput(prompt: .text(prompt))
                        let input = try await context.processor.prepare(input: userInput)
                        
                        // Setup constraints
                        let constraintProcessor = TokenTrieLogitProcessor(
                            schema: schema,
                            tokenizer: context.tokenizer
                        )
                        
                        // Configure generation parameters
                        let parameters = GenerateParameters(
                            maxTokens: sampling.maxTokens,
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
                        
                        // Generate with constraints using the custom iterator
                        let stream = MLXLMCommon.generate(
                            input: input,
                            context: context,
                            iterator: iterator
                        )
                        
                        for await generation in stream {
                            if case .chunk(let text) = generation {
                                continuation.yield(text)
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
}

// MARK: - Extensions

extension MLXBackend {
    /// Generate with automatic retry on schema violations
    func generateTextConstrainedWithRetry(
        prompt: String,
        sampling: SamplingParameters,
        schema: SchemaMeta,
        maxRetries: Int = 3
    ) async throws -> String {
        var lastError: Error?
        
        for attempt in 1...maxRetries {
            do {
                let result = try await generateTextConstrained(
                    prompt: prompt,
                    sampling: sampling,
                    schema: schema
                )
                
                // Validate the generated JSON
                if validateJSON(result, schema: schema) {
                    return result
                } else {
                    Logger.warning("[MLXBackend] Attempt \(attempt): Schema validation failed, retrying...")
                    lastError = MLXBackendError.schemaViolation
                }
            } catch {
                Logger.warning("[MLXBackend] Attempt \(attempt) failed: \(error.localizedDescription)")
                lastError = error
            }
        }
        
        throw lastError ?? MLXBackendError.maxRetriesExceeded
    }
    
    private func validateJSON(_ text: String, schema: SchemaMeta) -> Bool {
        // Try to parse as JSON
        guard let data = text.data(using: .utf8),
              let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            return false
        }
        
        // Check required keys
        for requiredKey in schema.required {
            if json[requiredKey] == nil {
                return false
            }
        }
        
        // Check all keys are in schema
        for key in json.keys {
            if !schema.keys.contains(key) {
                return false
            }
        }
        
        return true
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