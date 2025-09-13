import Foundation
import MLX
import MLXLMCommon

/// ChainedLogitProcessor combines multiple LogitProcessors into a single processor
/// Each processor is applied in sequence, allowing for composition of different constraints and observers
public struct ChainedLogitProcessor: LogitProcessor {
    private var processors: [LogitProcessor]
    
    /// Initialize with an array of processors to chain
    /// - Parameter processors: The processors to apply in sequence
    public init(processors: [LogitProcessor]) {
        self.processors = processors
    }
    
    /// Initialize with variadic processors
    /// - Parameter processors: The processors to apply in sequence
    public init(_ processors: LogitProcessor...) {
        self.processors = processors
    }
    
    // MARK: - LogitProcessor Protocol
    
    public mutating func prompt(_ prompt: MLXArray) {
        // Call prompt on all processors
        for i in processors.indices {
            processors[i].prompt(prompt)
        }
    }
    
    public func process(logits: MLXArray) -> MLXArray {
        // Apply each processor in sequence
        var result = logits
        for processor in processors {
            result = processor.process(logits: result)
        }
        return result
    }
    
    public mutating func didSample(token: MLXArray) {
        // Notify all processors of the sampled token
        for i in processors.indices {
            processors[i].didSample(token: token)
        }
    }
    
    // MARK: - Utility Methods
    
    /// Add a processor to the chain
    /// - Parameter processor: The processor to add
    public mutating func add(_ processor: LogitProcessor) {
        processors.append(processor)
    }
    
    /// Remove all processors
    public mutating func removeAll() {
        processors.removeAll()
    }
    
    /// Get the count of processors in the chain
    public var count: Int {
        processors.count
    }
    
    /// Check if the chain is empty
    public var isEmpty: Bool {
        processors.isEmpty
    }
}