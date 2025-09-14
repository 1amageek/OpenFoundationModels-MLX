import Foundation
import MLX
import MLXLMCommon

/// LogitProcessor that detects and logs JSON key generation with context-aware schema validation
/// Uses JSONContextTracker for accurate structure tracking
public final class KeyDetectionLogitProcessor: LogitProcessor, @unchecked Sendable {

    // MARK: - Nested Types

    /// Information about logit probabilities at a generation step
    struct LogitInfo {
        let step: Int
        let vocabSize: Int
        let topCandidates: [Candidate]
        let entropy: Float
        var selectedToken: Int32?
        let contextPath: String        // Current context path when this was generated
        let contextKeys: [String]      // Available keys at this context

        struct Candidate {
            let tokenId: Int32
            let probability: Float
            let logit: Float
            let text: String?
        }
    }

    // MARK: - Properties

    private let tokenizer: TokenizerAdapter
    private let verbose: Bool
    private let topK: Int
    private let showProbabilities: Bool
    private let modelCard: (any ModelCard)?
    private let schemaKeys: [String]?
    private let nestedSchemas: [String: [String]]?

    // Core state management
    private var jsonExtractor = JSONExtractor()
    private var stateMachine = JSONStateMachine()
    private var contextTracker = JSONContextTracker()
    private var generatedText = ""
    private var detectedKeys: [String] = []
    private var stepCount: Int = 0
    private var isActive: Bool = false
    private var pendingContextKey: String? = nil  // Store key name for context switch

    // Public access for testing
    public var allDetectedKeys: [String] {
        return detectedKeys
    }

    // Logit analysis state
    private var pendingLogitInfo: LogitInfo?
    private var lastWasInKeyGeneration: Bool = false

    // MARK: - Initialization

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
        self.isActive = (modelCard == nil)
    }

    // MARK: - LogitProcessor Protocol

    public func prompt(_ prompt: MLXArray) {
        // Reset all state for new generation
        jsonExtractor.reset()
        stateMachine.reset()
        contextTracker.reset()
        generatedText = ""
        detectedKeys = []
        stepCount = 0
        pendingLogitInfo = nil
        lastWasInKeyGeneration = false
        pendingContextKey = nil

        if verbose {
            print("\n===== 🚀 Generation Started =====")
            let tokens = prompt.asArray(Int32.self)
            print("Prompt: \(tokens.count) tokens\n")

            if showProbabilities {
                print("[KeyDetection] JSON key detection enabled")
                print("[KeyDetection] Will show context-aware key analysis")
            }
        }
    }

    public func process(logits: MLXArray) -> MLXArray {
        stepCount += 1

        // Check if we should be active
        if !shouldBeActive() {
            return logits
        }

        // Display pending logit info from previous step
        if let pending = pendingLogitInfo {
            if verbose && showProbabilities {
                // Context keys are already saved in LogitInfo, no need to get them again
                displayStep(pending, isKey: lastWasInKeyGeneration, contextKeys: [])
            }
            pendingLogitInfo = nil
        }

        // Always save current logit info for display in next step
        // This ensures we show entropy for all steps, not just key generation
        if jsonExtractor.isInJSON {
            pendingLogitInfo = buildLogitInfo(from: logits, step: stepCount)
        }

        return logits
    }

    public func didSample(token: MLXArray) {
        guard shouldBeActive() else { return }

        let tokenId = token.item(Int32.self)
        let text = tokenizer.decode([tokenId])
        generatedText += text

        // Process each character in the decoded text
        for char in text {
            // Try to extract JSON
            jsonExtractor.processCharacter(char)

            if jsonExtractor.isInJSON {
                // Track previous state for comparison
                let previousPhase = stateMachine.phase

                // Process the character in state machine
                stateMachine.processCharacter(char)

                // Handle key detection
                if case .inString(.body(kind: .key, escaped: false)) = previousPhase {
                    if case .inObject(.expectColon) = stateMachine.phase {
                        // Key just completed
                        let keyName = stateMachine.currentKey
                        if !keyName.isEmpty {
                            detectedKeys.append(keyName)
                            pendingContextKey = keyName  // Store for potential context switch

                            if verbose {
                                printKeyDetection(keyName)
                            }
                        }
                    }
                }

                // Handle context changes with improved logic
                handleContextChangesImproved(char: char, previousPhase: previousPhase, currentPhase: stateMachine.phase)
            }
        }

        // Update key generation state for next process() call
        lastWasInKeyGeneration = jsonExtractor.isInJSON && stateMachine.isInKeyGeneration

        // Show selected token info
        if verbose && showProbabilities && jsonExtractor.isInJSON {
            let displayText = formatTokenForDisplay(text)
            if lastWasInKeyGeneration {
                print("✅ [\(tokenId)] → \"\(displayText)\" 🔑 KEY TOKEN")
            } else {
                print("✅ [\(tokenId)] → \"\(displayText)\"")
            }
            // Show a snippet of the current generation
            if stepCount % 10 == 0 {  // Show every 10th step to avoid clutter
                print("   Output: \(String(generatedText.suffix(50)))")
            }
        }
    }

    public func finish() {
        // Display any pending logit info
        if let pending = pendingLogitInfo {
            if verbose && showProbabilities {
                // Context keys are already saved in LogitInfo
                displayStep(pending, isKey: lastWasInKeyGeneration, contextKeys: [])
            }
            pendingLogitInfo = nil
        }

        if verbose {
            print("\n[KeyDetection] Generation completed")
            print("[KeyDetection] Total keys detected: \(detectedKeys.count)")
            if !detectedKeys.isEmpty {
                print("[KeyDetection] Keys: \(detectedKeys.joined(separator: ", "))")
            }
        }
    }

    // MARK: - Private Methods

    private func shouldBeActive() -> Bool {
        if let card = modelCard {
            return card.isKeyDetectionEnabled ?? false
        }
        return isActive
    }

    private func handleContextChangesImproved(char: Character, previousPhase: JSONStateMachine.Phase, currentPhase: JSONStateMachine.Phase) {
        // Handle opening brace - entering an object
        if char == "{" {
            // Check if we have a pending key (means we're entering that key's object)
            if let key = pendingContextKey {
                contextTracker.keyDetected(key)
                contextTracker.enterObject()
                pendingContextKey = nil

                if verbose {
                    let path = contextTracker.getCurrentPath()
                    print("[Context] 📂 Entered object for key '\(key)'. Path: \(path)")
                }
            } else if case .inArray(.expectValue) = previousPhase {
                // Object in array without a key
                contextTracker.enterObject()
                if verbose {
                    print("[Context] 📂 Entered array item object. Path: \(contextTracker.getCurrentPath())")
                }
            } else if stateMachine.nestingLevel == 1 {
                // Root object
                if verbose {
                    print("[Context] 📂 Entered root object")
                }
            }
        }

        // Handle opening bracket - entering an array
        else if char == "[" {
            // Check if we have a pending key (means we're entering that key's array)
            if let key = pendingContextKey {
                contextTracker.keyDetected(key)
                contextTracker.enterArray()
                pendingContextKey = nil

                if verbose {
                    let path = contextTracker.getCurrentPath()
                    print("[Context] 📋 Entered array for key '\(key)'. Path: \(path)")
                }
            }
        }

        // Handle closing brace or bracket - exiting a context
        else if char == "}" || char == "]" {
            // Check if we need to exit context based on nesting level
            if stateMachine.nestingLevel < contextTracker.nestingDepth {
                contextTracker.exitContext()
                if verbose {
                    print("[Context] 🔙 Exited context. Path: \(contextTracker.getCurrentPath())")
                }
            }
        }

        // Clear pending key if we're moving to a different state without using it
        else if case .inObject(.expectCommaOrEnd) = currentPhase {
            // Value was not an object or array, clear pending key
            if pendingContextKey != nil {
                pendingContextKey = nil
            }
        }
    }

    private func getCurrentContextKeys() -> [String] {
        return contextTracker.getContextKeys(
            nestedSchemas: nestedSchemas,
            rootKeys: schemaKeys
        )
    }

    private func printKeyDetection(_ keyName: String) {
        let path = contextTracker.getCurrentPath()
        let level = contextTracker.nestingDepth

        if level == 0 {
            print("[KeyDetection] 🔑 Root key: \"\(keyName)\"")
        } else if contextTracker.isInArray() {
            let arrayContext = contextTracker.getCurrentArrayContext() ?? "array"
            print("[KeyDetection]   📍 Array item key: \"\(keyName)\" in \(arrayContext)")
        } else {
            let indent = String(repeating: "  ", count: level)
            let context = path.isEmpty ? "object" : path
            print("[KeyDetection] \(indent)📍 Level \(level) key: \"\(keyName)\" in \(context)")
        }
    }

    // MARK: - Probability Analysis

    private func buildLogitInfo(from logits: MLXArray, step: Int) -> LogitInfo {
        let vocabSize = logits.dim(logits.ndim - 1)

        var logits = logits
        if logits.dtype == .bfloat16 {
            logits = logits.asType(.float32)
        }

        let probs = softmax(logits, axis: -1)
        let candidates = extractTopCandidates(logits: logits, probs: probs, k: topK)
        let entropy = calculateEntropy(probs: probs)

        // Capture current context at the time of generation
        let currentPath = contextTracker.getCurrentPath()
        let currentKeys = contextTracker.getContextKeys(
            nestedSchemas: nestedSchemas,
            rootKeys: schemaKeys
        )

        return LogitInfo(
            step: step,
            vocabSize: vocabSize,
            topCandidates: candidates,
            entropy: entropy,
            selectedToken: nil,
            contextPath: currentPath,
            contextKeys: currentKeys
        )
    }

    private func extractTopCandidates(logits: MLXArray, probs: MLXArray, k: Int) -> [LogitInfo.Candidate] {
        let sortedIndices = argSort(probs, axis: -1)
        let vocabSize = logits.dim(logits.ndim - 1)
        let effectiveK = min(k, vocabSize)

        sortedIndices.eval()
        probs.eval()
        logits.eval()

        let allIndices = sortedIndices.asArray(Int32.self)
        let topIndices = Array(allIndices.suffix(effectiveK))
        let probsArray = probs.asArray(Float.self)
        let logitsArray = logits.asArray(Float.self)

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
        let epsilon: Float = 1e-10
        let clippedProbs = probs + epsilon
        let logProbs = log(clippedProbs)
        let entropy = -sum(probs * logProbs, axis: -1)

        entropy.eval()
        let result = entropy.item(Float.self)

        if result.isNaN || result.isInfinite {
            return 0.0
        }

        return max(0.0, result)
    }

    private func displayStep(_ info: LogitInfo, isKey: Bool, contextKeys: [String]) {
        // Build status markers and context information
        var statusMarker = ""

        // Use the context keys that were saved with the LogitInfo
        let actualContextKeys = info.contextKeys  // Use saved context, not passed parameter

        // Always show context keys when we're about to generate a key
        if isKey {
            statusMarker = " 🔑 KEY"
            if !actualContextKeys.isEmpty {
                let constraintList = actualContextKeys.joined(separator: ", ")
                statusMarker += " [Available: \(constraintList)]"
            } else {
                statusMarker += " [No schema constraints]"
            }
        } else if stateMachine.isInKeyGeneration {
            // We're in the middle of generating a key
            statusMarker = " 🔑 IN KEY"
        } else if case .inObject(.expectKeyOrEnd) = stateMachine.phase {
            // We're about to start a key
            statusMarker = " 📝 EXPECTING KEY"
            if !actualContextKeys.isEmpty {
                let constraintList = actualContextKeys.joined(separator: ", ")
                statusMarker += " [Available: \(constraintList)]"
            }
        } else if case .inObject(.expectValue) = stateMachine.phase {
            statusMarker = " 📦 EXPECTING VALUE"
        }

        // Show context path if not at root
        if !info.contextPath.isEmpty {
            statusMarker += " @\(info.contextPath)"
        }

        let entropyStr = info.entropy.isNaN ? "0.00" : String(format: "%.2f", info.entropy)
        print("\n[Step \(info.step)] Entropy: \(entropyStr) (\(entropyDescription(info.entropy)))\(statusMarker)")

        for (index, candidate) in info.topCandidates.prefix(min(5, topK)).enumerated() {
            let probValue = String(format: "%.1f%%", candidate.probability * 100)
            let text = candidate.text ?? "<unknown>"
            let displayText = formatTokenForDisplay(text)
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
            return String(text.prefix(20))
                .replacingOccurrences(of: "\n", with: "\\n")
                .replacingOccurrences(of: "\t", with: "\\t")
        }
    }

    private func entropyDescription(_ entropy: Float) -> String {
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
}

// MARK: - ModelCard Extension

extension ModelCard {
    var isKeyDetectionEnabled: Bool? {
        // This can be customized per model card
        return true
    }
}