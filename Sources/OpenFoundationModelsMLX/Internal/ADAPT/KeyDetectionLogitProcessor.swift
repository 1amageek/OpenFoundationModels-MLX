import Foundation
import MLX
import MLXLMCommon

/// LogitProcessor that detects and logs JSON key generation with probability analysis
/// Uses JSONStateMachine to track parsing state and identify key tokens
/// Provides detailed statistics about token selection during key generation
public final class KeyDetectionLogitProcessor: LogitProcessor, @unchecked Sendable {
    
    // MARK: - Properties
    
    private let tokenizer: TokenizerAdapter
    private let verbose: Bool
    private let topK: Int
    private let showProbabilities: Bool
    private let modelCard: (any ModelCard)?
    private let schemaKeys: [String]?
    private let nestedSchemas: [String: [String]]?
    
    // Mutable state
    private var jsonExtractor = JSONExtractor()  // Extract JSON from arbitrary text
    private var stateMachine = JSONStateMachine()
    private var generatedText = ""
    private var detectedKeys: [String] = []
    private var nestingStack: [String] = []  // Track nested object context
    private var stepCount: Int = 0
    private var lastProcessedText = ""  // Track last token to determine context
    private var isActive: Bool = false  // Whether this processor is currently active
    
    // Delayed evaluation state for accurate key detection
    private var pendingLogitInfo: LogitInfo?
    private var wasInKeyGeneration: Bool = false
    
    // MARK: - Initialization
    
    /// Initialize the key detection processor
    /// - Parameters:
    ///   - tokenizer: Tokenizer for decoding tokens to text
    ///   - modelCard: Optional ModelCard for activation control
    ///   - schemaKeys: Optional list of allowed keys from JSON schema
    ///   - nestedSchemas: Optional nested object schemas (key -> list of allowed keys)
    ///   - verbose: Whether to print detailed output (default: true)
    ///   - topK: Number of top candidates to track (default: 5)
    ///   - showProbabilities: Whether to show probability distributions (default: true)
    public init(
        tokenizer: TokenizerAdapter,
        modelCard: (any ModelCard)? = nil,
        schemaKeys: [String]? = nil,
        nestedSchemas: [String: [String]]? = nil,
        verbose: Bool = true,
        topK: Int = 5,
        showProbabilities: Bool = true
    ) {
        self.tokenizer = tokenizer
        self.modelCard = modelCard
        self.schemaKeys = schemaKeys
        self.nestedSchemas = nestedSchemas
        self.verbose = verbose
        self.topK = topK
        self.showProbabilities = showProbabilities
        // If no modelCard provided, always active
        self.isActive = (modelCard == nil)
    }
    
    // MARK: - LogitProcessor Protocol
    
    public func prompt(_ prompt: MLXArray) {
        // Reset state for new generation
        jsonExtractor.reset()
        stateMachine.reset()
        generatedText = ""
        detectedKeys = []
        nestingStack = []
        stepCount = 0
        pendingLogitInfo = nil
        wasInKeyGeneration = false
        
        if verbose {
            print("\n===== 🚀 Generation Started =====")
            let tokens = prompt.asArray(Int32.self)
            print("Prompt: \(tokens.count) tokens\n")
            
            if showProbabilities {
                print("[KeyDetection] JSON key detection enabled")
                print("[KeyDetection] Will show enhanced analysis for key generation")
            }
        }
    }
    
    public func process(logits: MLXArray) -> MLXArray {
        // Only process if active
        guard isActive else {
            return logits
        }
        
        // Delayed evaluation: Show entropy for the PREVIOUS token if it was in key generation
        if let pending = pendingLogitInfo,
           wasInKeyGeneration,
           showProbabilities,
           verbose {
            // Check if the pending token was a quote (which ends the key)
            if let topCandidate = pending.topCandidates.first,
               let text = topCandidate.text,
               text == "\"" || text.contains("\"") {
                // Skip displaying for quote tokens
            } else {
                displayStep(pending, isKey: true)
            }
        }
        
        // Build and store current logit info for next iteration
        let currentInfo = buildLogitInfo(from: logits, step: stepCount + 1)
        pendingLogitInfo = currentInfo
        
        // Note: We'll update wasInKeyGeneration in didSample AFTER processing the token
        // This ensures we check the state after JSONExtractor has had a chance to find JSON
        
        // This processor only observes, doesn't modify logits
        return logits
    }
    
    public func didSample(token: MLXArray) {
        stepCount += 1
        let tokenId = token.item(Int32.self)
        let text = tokenizer.decode([tokenId])
        
        // Track generated text
        generatedText.append(text)
        lastProcessedText = text
        
        // Check with ModelCard if we should be active
        if let card = modelCard {
            isActive = card.shouldActivateProcessor(generatedText, processor: self)
        }
        
        // Only process if active
        guard isActive else {
            return
        }
        
        // Process each character through the JSON extractor first
        for char in text {
            // Check if JSONStateMachine needs reset (done or error state)
            if stateMachine.phase == .done || stateMachine.phase == .error {
                // Reset both to prepare for new JSON
                jsonExtractor.reset()
                stateMachine.reset()
            }
            
            // Use JSONExtractor to find JSON content
            let shouldProcess = jsonExtractor.processCharacter(char)
            
            // Only process with JSONStateMachine if we're in JSON content
            if shouldProcess {
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
            } else if verbose && !jsonExtractor.jsonFound {
                // Optionally log that we're still scanning for JSON
                // This is useful for debugging GPT OSS format issues
                if char == "<" || char == "|" || char == ">" {
                    // Likely GPT OSS special tokens, don't spam logs
                } else if !char.isWhitespace {
                    // Log non-whitespace skipped characters sparingly
                }
            }
        }
        
        // Now show the output for the actually selected token (with correct state)
        if verbose && showProbabilities && wasInKeyGeneration {
            // Skip if this is a quote token
            if text == "\"" || text.contains("\"") {
                // Don't mark quotes as key tokens
            } else {
                let displayText = formatTokenForDisplay(text)
                print("✅ [\(tokenId)] → \"\(displayText)\" 🔑 KEY TOKEN")
                print("   Output: \(String(generatedText.suffix(50)))")
            }
        }
        
        // Update key generation state after processing the token
        // This ensures we check the state after JSONExtractor has had a chance to find JSON
        wasInKeyGeneration = jsonExtractor.isInJSON && stateMachine.isInKeyGeneration
    }
    
    // MARK: - Private Methods
    
    private func getCurrentContextKeys() -> [String] {
        // If we have nested context, use nested schema keys
        if !nestingStack.isEmpty,
           let lastContext = nestingStack.last,
           let nestedKeys = nestedSchemas?[lastContext] {
            return nestedKeys
        }
        // Otherwise use root schema keys
        return schemaKeys ?? []
    }
    
    private func handleContextChanges(previousPhase: JSONStateMachine.Phase) {
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
        let epsilon: Float = 1e-10
        let clippedProbs = probs + epsilon
        let logProbs = log(clippedProbs)
        let entropy = -sum(probs * logProbs, axis: -1)
        
        // Evaluate and extract scalar
        entropy.eval()
        let result = entropy.item(Float.self)
        
        // Check for NaN and return 0 (perfect certainty)
        if result.isNaN || result.isInfinite {
            return 0.0
        }
        
        // Clamp very small values to 0
        if result < epsilon {
            return 0.0
        }
        
        return result
    }
    
    private func displayStep(_ info: LogitInfo, isKey: Bool) {
        // Use consistent format for all tokens
        var keyMarker = isKey ? " 🔑 KEY" : ""
        
        // Add schema constraints if available and in key generation
        if isKey {
            // Determine which keys to show based on current context
            let keysToShow = getCurrentContextKeys()
            if !keysToShow.isEmpty {
                let constraintList = keysToShow.joined(separator: ", ")
                keyMarker += " [\(constraintList)]"
            }
        }
        
        let entropyStr = info.entropy.isNaN ? "0.00" : String(format: "%.2f", info.entropy)
        print("\n[Step \(info.step)] Entropy: \(entropyStr) (\(entropyDescription(info.entropy)))\(keyMarker)")
        
        // Show top candidates with consistent formatting
        for (index, candidate) in info.topCandidates.prefix(min(5, topK)).enumerated() {
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
        // Handle NaN and special cases
        if entropy.isNaN {
            return "🟢 Perfect Confidence (100%)"
        }
        
        switch entropy {
        case ..<0.01:
            return "🟢 Perfect Confidence"
        case 0.01..<0.5:
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
    
    // MARK: - Finish Method
    
    /// Process any remaining pending logit info when generation completes
    public func finish() {
        // Display the last pending entropy if it was in key generation
        if let pending = pendingLogitInfo,
           wasInKeyGeneration,
           showProbabilities,
           verbose {
            // Check if the pending token was a quote
            if let topCandidate = pending.topCandidates.first,
               let text = topCandidate.text,
               text == "\"" || text.contains("\"") {
                // Skip displaying for quote tokens
            } else {
                print("\n[Final Token]")
                displayStep(pending, isKey: true)
            }
        }
        
        // Clear pending state
        pendingLogitInfo = nil
        wasInKeyGeneration = false
        
        if verbose {
            print("\n[KeyDetection] Generation completed")
            print("[KeyDetection] Total keys detected: \(detectedKeys.count)")
            if !detectedKeys.isEmpty {
                print("[KeyDetection] Keys: \(detectedKeys.joined(separator: ", "))")
            }
        }
    }
}