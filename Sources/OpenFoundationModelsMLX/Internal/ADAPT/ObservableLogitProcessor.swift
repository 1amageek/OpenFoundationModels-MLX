import Foundation
import MLX
import MLXLMCommon

/// LogitProcessor that observes and reports the generation process
/// This is a production feature for monitoring token generation, not a debug tool
public struct ObservableLogitProcessor: LogitProcessor {
    private let tokenizer: TokenizerAdapter?
    private let topK: Int
    private let verbose: Bool
    
    // Mutable state managed like RepetitionContext
    private var stepCount: Int = 0
    private var lastInfo: LogitInfo?
    
    /// Initialize the observable processor
    /// - Parameters:
    ///   - tokenizer: Optional tokenizer for decoding token IDs to text
    ///   - topK: Number of top candidates to track (default: 10)
    ///   - verbose: Whether to print observation output (default: true)
    public init(
        tokenizer: TokenizerAdapter? = nil,
        topK: Int = 10,
        verbose: Bool = true
    ) {
        self.tokenizer = tokenizer
        self.topK = topK
        self.verbose = verbose
    }
    
    // MARK: - LogitProcessor Protocol
    
    public mutating func prompt(_ prompt: MLXArray) {
        stepCount = 0
        lastInfo = nil
        
        if verbose {
            print("\n" + String(repeating: "=", count: 50))
            print("Generation Started")
            print(String(repeating: "=", count: 50))
            
            let tokens = prompt.asArray(Int32.self)
            print("Prompt length: \(tokens.count) tokens")
            
            // Optionally show last few tokens of prompt
            if let tokenizer = tokenizer, tokens.count > 0 {
                let lastTokens = Array(tokens.suffix(10))
                let text = tokenizer.decode(lastTokens)
                print("Prompt tail: ...\(text)")
            }
            
            print(String(repeating: "-", count: 50) + "\n")
        }
    }
    
    public func process(logits: MLXArray) -> MLXArray {
        // Build structured logit information without mutation
        let currentStep = stepCount + 1  // Will be incremented in didSample
        let info = buildLogitInfo(from: logits, step: currentStep)
        
        // Display if verbose mode is enabled
        if verbose {
            displayStep(info)
        }
        
        // Store for later use (will be updated in didSample)
        // Note: We can't mutate here, so we just display and return
        
        // Return unmodified logits (observation only, no modification)
        return logits
    }
    
    public mutating func didSample(token: MLXArray) {
        stepCount += 1
        let tokenId = token.item(Int32.self)
        
        // Now we can update state since this is mutating
        if verbose {
            let text = tokenizer?.decode([tokenId]) ?? "<unknown>"
            let displayText = text
                .replacingOccurrences(of: "\n", with: "\\n")
                .replacingOccurrences(of: "\t", with: "\\t")
                .replacingOccurrences(of: "\r", with: "\\r")
            
            print("→ Selected: [\(tokenId)] '\(displayText)'")
            print(String(repeating: "-", count: 50) + "\n")
        }
    }
    
    // MARK: - Private Methods
    
    private func buildLogitInfo(from logits: MLXArray, step: Int) -> LogitInfo {
        let vocabSize = logits.dim(logits.ndim - 1)
        
        // Ensure float32 for numerical stability
        var logits = logits
        if logits.dtype == .bfloat16 {
            logits = logits.asType(.float32)
        }
        
        // Calculate probabilities using softmax
        let probs = softmax(logits, axis: -1)
        
        // Extract top-k candidates efficiently
        let candidates = extractTopCandidates(logits: logits, probs: probs, k: topK)
        
        // Calculate entropy of the distribution
        let entropy = calculateEntropy(probs: probs)
        
        return LogitInfo(
            step: step,
            vocabSize: vocabSize,
            topCandidates: candidates,
            entropy: entropy,
            selectedToken: nil
        )
    }
    
    private func extractTopCandidates(logits: MLXArray, probs: MLXArray, k: Int) -> [LogitInfo.Candidate] {
        // Use argSort to get indices sorted by probability
        let sortedIndices = argSort(probs, axis: -1)
        
        // Get vocabulary size and effective k
        let vocabSize = logits.dim(logits.ndim - 1)
        let effectiveK = min(k, vocabSize)
        
        // Evaluate arrays for extraction
        sortedIndices.eval()
        probs.eval()
        logits.eval()
        
        // Get indices and convert to arrays
        let allIndices = sortedIndices.asArray(Int32.self)
        let topIndices = Array(allIndices.suffix(effectiveK))
        let probsArray = probs.asArray(Float.self)
        let logitsArray = logits.asArray(Float.self)
        
        // Build candidates, reversing to get descending order
        return topIndices.reversed().map { tokenId in
            LogitInfo.Candidate(
                tokenId: tokenId,
                probability: probsArray[Int(tokenId)],
                logit: logitsArray[Int(tokenId)],
                text: tokenizer?.decode([tokenId])
            )
        }
    }
    
    private func calculateEntropy(probs: MLXArray) -> Float {
        // H = -Σ p * log(p)
        // Add small epsilon to avoid log(0)
        let logProbs = log(probs + 1e-10)
        let entropy = -sum(probs * logProbs, axis: -1)
        
        // Evaluate and extract scalar
        entropy.eval()
        return entropy.item(Float.self)
    }
    
    private func displayStep(_ info: LogitInfo) {
        print("Step \(info.step) (vocab size: \(info.vocabSize), entropy: \(String(format: "%.2f", info.entropy)))")
        print(info.formatCandidates(limit: min(5, topK)))
    }
}