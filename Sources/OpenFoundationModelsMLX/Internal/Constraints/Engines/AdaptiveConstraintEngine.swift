import Foundation
import MLX
import MLXLMCommon
import Tokenizers
import Synchronization

/// Adaptive constraint engine that selects appropriate constraints based on the request
/// This ensures all generation (Text/JSON) goes through the same pipeline with observable output
final class AdaptiveConstraintEngine: ConstraintEngine, Sendable {
    private let mutex = Mutex<State>(.init())
    
    private struct State: Sendable {
        var mode: ConstraintMode = .off
        var preparedSchema: SchemaNode?
        var observableProcessor: ObservableLogitProcessor?
        var schemaProcessors: [any LogitProcessor] = []
    }
    
    init() {}
    
    var mode: ConstraintMode {
        mutex.withLock { $0.mode }
    }
    
    func prepare(schema: SchemaNode?, tokenizer: any Tokenizer) async throws {
        // Create tokenizer adapter
        let tokenizerAdapter = MLXLLMTokenizer(tokenizer: tokenizer)
        
        // Always create ObservableLogitProcessor for all modes
        let observableProcessor = ObservableLogitProcessor(
            tokenizer: tokenizerAdapter,
            topK: 10,
            verbose: true  // Always verbose for debugging
        )
        
        // Determine mode and additional processors based on schema
        if let schema = schema, !schema.isEmpty {
            // JSON mode with schema constraints
            // Future: Add TokenTrieLogitProcessor or other ADAPT implementations here
            
            mutex.withLock {
                $0.mode = .hard
                $0.preparedSchema = schema
                $0.observableProcessor = observableProcessor
                $0.schemaProcessors = [] // Will be populated with ADAPT processors
            }
        } else {
            // Text mode - only observation, no constraints
            mutex.withLock {
                $0.mode = .off
                $0.preparedSchema = nil
                $0.observableProcessor = observableProcessor
                $0.schemaProcessors = []
            }
        }
    }
    
    func softPrompt(for schema: SchemaNode?) -> String? {
        // No soft prompts needed for adaptive engine
        return nil
    }
    
    func logitProcessors() async -> [LogitProcessor] {
        return mutex.withLock { state in
            var processors: [any LogitProcessor] = []
            
            // Always include ObservableLogitProcessor first
            if let observable = state.observableProcessor {
                processors.append(observable)
            }
            
            // Add schema-specific processors if in JSON mode
            processors.append(contentsOf: state.schemaProcessors)
            
            return processors
        }
    }
    
    func validator() -> (any JSONValidatorProtocol)? {
        let currentMode = mutex.withLock { $0.mode }
        
        // Only validate in hard constraint mode with schema
        if currentMode == .hard {
            return SharedJSONValidator()
        }
        return nil
    }
}
