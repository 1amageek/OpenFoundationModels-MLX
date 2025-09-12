import Foundation
import MLX
import MLXLMCommon

/// LogitProcessor that detects and logs JSON key generation with probability analysis
/// Uses JSONStateMachine to track parsing state and identify key tokens
/// Provides detailed statistics about token selection during key generation
public struct KeyDetectionLogitProcessor: LogitProcessor, @unchecked Sendable {
    
    // MARK: - Properties
    
    private let tokenizer: TokenizerAdapter
    private let verbose: Bool
    private let topK: Int
    private let showProbabilities: Bool
    
    // Mutable state
    private var stateMachine = JSONStateMachine()
    private var generatedText = ""
    private var detectedKeys: [String] = []
    private var nestingStack: [String] = []  // Track nested object context
    private var stepCount: Int = 0
    
    // MARK: - Initialization
    
    /// Initialize the key detection processor
    /// - Parameters:
    ///   - tokenizer: Tokenizer for decoding tokens to text
    ///   - verbose: Whether to print detailed output (default: true)
    ///   - topK: Number of top candidates to track (default: 5)
    ///   - showProbabilities: Whether to show probability distributions (default: true)
    public init(
        tokenizer: TokenizerAdapter,
        verbose: Bool = true,
        topK: Int = 5,
        showProbabilities: Bool = true
    ) {
        self.tokenizer = tokenizer
        self.verbose = verbose
        self.topK = topK
        self.showProbabilities = showProbabilities
    }
    
    // MARK: - LogitProcessor Protocol
    
    public mutating func prompt(_ prompt: MLXArray) {
        // Reset state for new generation
        stateMachine.reset()
        generatedText = ""
        detectedKeys = []
        nestingStack = []
        stepCount = 0
        
        if verbose {
            print("\n[KeyDetection] 🔍 Starting JSON key detection with probability analysis...")
            if showProbabilities {
                print("[KeyDetection] Tracking top-\(topK) candidates with entropy")
            }
        }
    }
    
    public func process(logits: MLXArray) -> MLXArray {
        // Build logit information if probability display is enabled
        if showProbabilities {
            let info = buildLogitInfo(from: logits, step: stepCount + 1)
            
            // Display probability info if we're in key generation phase
            if verbose && stateMachine.isInKeyGeneration {
                displayKeyGenerationStep(info)
            }
            
            // Note: We'll store the info in didSample where we can mutate
        }
        
        // This processor only observes, doesn't modify logits
        return logits
    }
    
    public mutating func didSample(token: MLXArray) {
        stepCount += 1
        let tokenId = token.item(Int32.self)
        let text = tokenizer.decode([tokenId])
        
        // Track generated text
        generatedText.append(text)
        
        // Show selected token info if probability display is enabled
        // Note: We can't access the logits here, so we can't show rank/probability for the selected token
        if showProbabilities && verbose && stateMachine.isInKeyGeneration {
            print("[KeyDetection] ✅ Selected token [\(tokenId)]: \"\(formatTokenForDisplay(text))\"")
        }
        
        // Process each character through the state machine
        for char in text {
            // Store previous phase for comparison
            let previousPhase = stateMachine.phase
            
            // Process the character
            stateMachine.processCharacter(char)
            
            // Check if we just exited key generation
            if case .inString(.body(kind: .key, escaped: false)) = previousPhase {
                if case .inObject(.expectColon) = stateMachine.phase {
                    // Key just completed - capture it immediately
                    let keyName = stateMachine.currentKey
                    if !keyName.isEmpty {
                        detectedKeys.append(keyName)
                        
                        if verbose {
                            let level = stateMachine.nestingLevel - 1
                            let indent = String(repeating: "  ", count: level)
                            
                            if level > 0 {
                                let context = nestingStack.last ?? "object"
                                print("[KeyDetection] \(indent)📍 Level \(level) key: \"\(keyName)\" (in \(context))")
                            } else {
                                print("[KeyDetection] 🔑 Root key: \"\(keyName)\"")
                            }
                        }
                    }
                }
            }
            
            // Check for context changes (entering/exiting objects)
            handleContextChanges(previousPhase: previousPhase)
            
            // Debug output for phase transitions (only if not showing probabilities to reduce noise)
            if verbose && !showProbabilities && hasPhaseChanged(from: previousPhase, to: stateMachine.phase) {
                printPhaseTransition(from: previousPhase, to: stateMachine.phase)
            }
        }
    }
    
    // MARK: - Private Methods
    
    private mutating func handleContextChanges(previousPhase: JSONStateMachine.Phase) {
        // Check if we entered a new object
        if case .inObject(.expectValue) = previousPhase,
           case .inObject(.expectKeyOrEnd) = stateMachine.phase {
            // Entered a nested object after a key
            if let lastKey = detectedKeys.last {
                nestingStack.append(lastKey)
                if verbose {
                    print("[KeyDetection] 📂 Entering object: \(lastKey)")
                }
            }
        }
        
        // Check if we exited an object
        if stateMachine.nestingLevel < nestingStack.count {
            if let exitedContext = nestingStack.popLast() {
                if verbose {
                    print("[KeyDetection] 📁 Exiting object: \(exitedContext)")
                }
            }
        }
    }
    
    private func hasPhaseChanged(from oldPhase: JSONStateMachine.Phase, to newPhase: JSONStateMachine.Phase) -> Bool {
        // Simple equality check for phase changes
        switch (oldPhase, newPhase) {
        case (.root, .root),
             (.done, .done),
             (.error, .error):
            return false
        case (.inObject(let old), .inObject(let new)):
            return old != new
        case (.inArray(let old), .inArray(let new)):
            return old != new
        case (.inString(let old), .inString(let new)):
            return old != new
        case (.inNumber(let old), .inNumber(let new)):
            return old != new
        case (.inLiteral(let old), .inLiteral(let new)):
            return old != new
        default:
            return true
        }
    }
    
    private func printPhaseTransition(from oldPhase: JSONStateMachine.Phase, to newPhase: JSONStateMachine.Phase) {
        let oldDesc = phaseDescription(oldPhase)
        let newDesc = phaseDescription(newPhase)
        
        // Only print significant transitions
        if shouldPrintTransition(from: oldPhase, to: newPhase) {
            print("[KeyDetection] Phase: \(oldDesc) → \(newDesc)")
        }
    }
    
    private func shouldPrintTransition(from oldPhase: JSONStateMachine.Phase, to newPhase: JSONStateMachine.Phase) -> Bool {
        // Filter out noisy transitions
        switch (oldPhase, newPhase) {
        case (.inString, .inString):
            // Don't print character-by-character string updates
            return false
        case (.inNumber, .inNumber):
            // Don't print digit-by-digit number updates
            return false
        case (_, .inObject(.expectKeyFirstQuote)),
             (_, .inObject(.inKey)),
             (_, .inObject(.expectColon)):
            // These are key-related transitions worth showing
            return true
        default:
            // Show other significant transitions
            return true
        }
    }
    
    private func phaseDescription(_ phase: JSONStateMachine.Phase) -> String {
        switch phase {
        case .root:
            return "root"
        case .inObject(let objPhase):
            return "object.\(objectPhaseDescription(objPhase))"
        case .inArray(let arrPhase):
            return "array.\(arrayPhaseDescription(arrPhase))"
        case .inString(let strPhase):
            return "string.\(stringPhaseDescription(strPhase))"
        case .inNumber(let numPhase):
            return "number.\(numberPhaseDescription(numPhase))"
        case .inLiteral(let litPhase):
            if case .inProgress(let literal) = litPhase {
                return "literal(\(literal))"
            }
            return "literal"
        case .done:
            return "done"
        case .error:
            return "error"
        }
    }
    
    private func objectPhaseDescription(_ phase: JSONStateMachine.ObjectPhase) -> String {
        switch phase {
        case .expectKeyOrEnd: return "expectKeyOrEnd"
        case .expectKeyFirstQuote: return "expectKeyQuote"
        case .inKey: return "inKey"
        case .expectKeyEndQuote: return "expectEndQuote"
        case .expectColon: return "expectColon"
        case .expectValue: return "expectValue"
        case .expectCommaOrEnd: return "expectCommaOrEnd"
        }
    }
    
    private func arrayPhaseDescription(_ phase: JSONStateMachine.ArrayPhase) -> String {
        switch phase {
        case .expectValue: return "expectValue"
        case .expectCommaOrEnd: return "expectCommaOrEnd"
        }
    }
    
    private func stringPhaseDescription(_ phase: JSONStateMachine.StringPhase) -> String {
        switch phase {
        case .body(let kind, let escaped):
            let kindStr = kind == .key ? "key" : "value"
            let escStr = escaped ? ",escaped" : ""
            return "\(kindStr)\(escStr)"
        }
    }
    
    private func numberPhaseDescription(_ phase: JSONStateMachine.NumberPhase) -> String {
        switch phase {
        case .integer: return "integer"
        case .decimal: return "decimal"
        case .exponent: return "exponent"
        }
    }
    
    // MARK: - Public Accessors
    
    /// Get all detected keys in order
    public var allDetectedKeys: [String] {
        return detectedKeys
    }
    
    /// Get the current JSON parsing phase
    public var currentPhase: JSONStateMachine.Phase {
        return stateMachine.phase
    }
    
    /// Check if currently generating a key
    public var isGeneratingKey: Bool {
        return stateMachine.isInKeyGeneration
    }
    
    // MARK: - Probability Analysis Methods
    
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
                text: tokenizer.decode([tokenId])
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
    
    private func displayKeyGenerationStep(_ info: LogitInfo) {
        print("\n[KeyDetection] Step \(info.step) | Entropy: \(String(format: "%.2f", info.entropy)) \(entropyDescription(info.entropy))")
        print("[KeyDetection] Top-\(min(topK, info.topCandidates.count)) key candidates:")
        
        for (index, candidate) in info.topCandidates.prefix(topK).enumerated() {
            let probValue = String(format: "%.1f%%", candidate.probability * 100)
            
            // Format text for display
            let text = candidate.text ?? "<unknown>"
            let displayText = formatTokenForDisplay(text)
            
            // Simple bar visualization
            let barLength = Int(candidate.probability * 10)
            let bar = String(repeating: "▓", count: barLength) + String(repeating: "░", count: 10 - barLength)
            
            print("  \(index + 1). [\(String(format: "%6d", candidate.tokenId))] \(bar) \"\(displayText)\" \(probValue)")
        }
    }
    
    private func formatTokenForDisplay(_ text: String) -> String {
        if text == "\n" {
            return "↵"
        } else if text == "\t" {
            return "→"
        } else if text == " " {
            return "␣"
        } else if text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
            return "⎵"
        } else {
            // Truncate long text and escape special characters
            return String(text.prefix(20))
                .replacingOccurrences(of: "\n", with: "\\n")
                .replacingOccurrences(of: "\t", with: "\\t")
        }
    }
    
    private func entropyDescription(_ entropy: Float) -> String {
        switch entropy {
        case ..<0.5:
            return "🟢 Very Confident"
        case 0.5..<1.5:
            return "🟡 Confident"
        case 1.5..<3.0:
            return "🟠 Moderate"
        case 3.0..<5.0:
            return "🔴 Uncertain"
        default:
            return "⚫ Very Uncertain"
        }
    }
}