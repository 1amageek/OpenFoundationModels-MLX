import Foundation
import MLX
import MLXLMCommon

/// LogitProcessor that detects and logs JSON key generation with context-aware schema validation
/// Uses JSONSchemaContextDetector for accurate context detection
public final class KeyDetectionLogitProcessor: LogitProcessor, @unchecked Sendable {

    // MARK: - Nested Types


    // MARK: - Properties

    private let tokenizer: TokenizerAdapter
    private let verbose: Bool
    private let topK: Int
    private let showProbabilities: Bool
    private let modelCard: (any ModelCard)?
    private let jsonSchema: [String: Any]?
    private var schemaDetector: JSONSchemaContextDetector?

    // Core state management
    private var jsonExtractor = JSONExtractor()
    private var generatedText = ""
    private var detectedKeys: [String] = []
    private var stepCount: Int = 0
    private var isActive: Bool = false

    // Public access for testing
    public var allDetectedKeys: [String] {
        return detectedKeys
    }

    // Logit analysis state
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
        self.verbose = verbose
        self.topK = topK
        self.showProbabilities = showProbabilities
        self.isActive = (modelCard == nil)

        // Convert old schema format to new JSON Schema format if provided
        if let rootKeys = schemaKeys {
            var schema: [String: Any] = [
                "type": "object",
                "properties": [:]
            ]

            var properties: [String: Any] = [:]

            // Add root keys
            for key in rootKeys {
                properties[key] = ["type": "object"]
            }

            // Add nested schemas
            if let nested = nestedSchemas {
                for (path, keys) in nested {
                    // Parse path to update schema
                    if path == "headquarters" {
                        var headquartersProps: [String: Any] = [:]
                        for key in keys {
                            headquartersProps[key] = ["type": "string"]
                        }
                        properties["headquarters"] = [
                            "type": "object",
                            "properties": headquartersProps
                        ]
                    } else if path.contains("[]") {
                        // Handle array paths like "departments[]"
                        let basePath = path.replacingOccurrences(of: "[]", with: "")
                        let pathComponents = basePath.split(separator: ".")

                        if pathComponents.count == 1 {
                            // e.g., "departments[]"
                            var itemProps: [String: Any] = [:]
                            for key in keys {
                                itemProps[key] = ["type": "object"]
                            }
                            properties[String(pathComponents[0])] = [
                                "type": "array",
                                "items": [
                                    "type": "object",
                                    "properties": itemProps
                                ]
                            ]
                        }
                    }
                }
            }

            schema["properties"] = properties
            self.jsonSchema = schema
            self.schemaDetector = JSONSchemaContextDetector(schema: schema)
        } else {
            // Even if no schema keys provided, create empty schema
            let emptySchema: [String: Any] = [
                "type": "object",
                "properties": [:]
            ]
            self.jsonSchema = emptySchema
            self.schemaDetector = JSONSchemaContextDetector(schema: emptySchema)
        }
    }

    // Alternative init with direct JSON Schema
    public init(
        tokenizer: TokenizerAdapter,
        jsonSchema: [String: Any],
        modelCard: (any ModelCard)? = nil,
        verbose: Bool = true,
        topK: Int = 5,
        showProbabilities: Bool = true
    ) {
        self.tokenizer = tokenizer
        self.jsonSchema = jsonSchema
        self.modelCard = modelCard
        self.verbose = verbose
        self.topK = topK
        self.showProbabilities = showProbabilities
        self.isActive = (modelCard == nil)

        // Always initialize schemaDetector
        self.schemaDetector = JSONSchemaContextDetector(schema: jsonSchema)

        if verbose {
            print("[KeyDetection] Initialized with JSON Schema")
            if let properties = jsonSchema["properties"] as? [String: Any] {
                print("[KeyDetection] Root keys: \(properties.keys.sorted())")
            } else if !jsonSchema.isEmpty {
                print("[KeyDetection] WARNING: No properties in JSON Schema!")
            }
        }
    }

    // MARK: - LogitProcessor Protocol

    public func prompt(_ prompt: MLXArray) {
        // Reset all state for new generation
        jsonExtractor.reset()
        generatedText = ""
        detectedKeys = []
        stepCount = 0
        lastWasInKeyGeneration = false

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

        // Display logit info if verbose and in JSON
        if verbose && showProbabilities && jsonExtractor.isInJSON {
            displayStepInfo(from: logits, step: stepCount, isKey: lastWasInKeyGeneration)
        }

        return logits
    }

    public func didSample(token: MLXArray) {
        guard shouldBeActive() else { return }

        let tokenId = token.item(Int32.self)
        let text = tokenizer.decode([tokenId])
        generatedText += text

        // Track previous phase for key detection
        var previousPhase: JSONStateMachine.Phase? = nil

        // Process each character in the decoded text
        for char in text {
            // Get phase before processing
            if jsonExtractor.isInJSON {
                previousPhase = jsonExtractor.getCurrentPhase()
            }

            // Try to extract JSON - now handles multiple JSONs automatically
            let shouldProcess = jsonExtractor.processCharacter(char)

            if shouldProcess && jsonExtractor.isInJSON {
                // Get current phase after processing
                let currentPhase = jsonExtractor.getCurrentPhase()

                // Handle key detection - check if we transitioned from key to colon
                if let prevPhase = previousPhase {
                    if case .inString(.body(kind: .key, escaped: false)) = prevPhase {
                        if case .inObject(.expectColon) = currentPhase {
                            // Key just completed - use the accumulated key from JSONExtractor
                            let keyName = extractCurrentKey()
                            if !keyName.isEmpty {
                                detectedKeys.append(keyName)

                                if verbose {
                                    printKeyDetection(keyName)
                                }
                            }
                        }
                    }
                }

                // Update key generation state based on current phase
                switch currentPhase {
                case .inObject(.expectKeyFirstQuote),
                     .inObject(.inKey),
                     .inString(.body(kind: .key, _)):
                    lastWasInKeyGeneration = true
                default:
                    lastWasInKeyGeneration = false
                }
            } else {
                lastWasInKeyGeneration = false
            }
        }

        // Show selected token info only for key generation
        if verbose && showProbabilities && jsonExtractor.isInJSON && lastWasInKeyGeneration {
            let displayText = formatTokenForDisplay(text)
            print("✅ [\(tokenId)] → \"\(displayText)\" 🔑 KEY TOKEN")
        }

        // Always display available keys when in JSON and about to generate or generating a key
        if jsonExtractor.isInJSON {
            let currentPhase = jsonExtractor.getCurrentPhase()

            // Display constraints when we're generating a key (including in the middle of a key)
            switch currentPhase {
            case .inObject(.expectKeyFirstQuote), .inObject(.inKey), .inString(.body(kind: .key, _)):
                let partialJSON = extractPartialJSON()
                let availableKeys = getCurrentAvailableKeys()

                print("\n📋 Available keys at current context: [\(availableKeys.joined(separator: ", "))]")

                // Show the current partial JSON position
                let preview = partialJSON.suffix(80)
                if !preview.isEmpty {
                    print("   Current position: ...\(preview)")
                }
            default:
                break
            }
        }
    }

    public func finish() {

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

    private func getCurrentAvailableKeys() -> [String] {
        // Use JSONSchemaContextDetector to get available keys
        guard let detector = schemaDetector else {
            // This should never happen since we always initialize schemaDetector
            print("[KeyDetection] FATAL: schemaDetector is nil!")
            return []
        }

        // Get the partial JSON generated so far
        let partialJSON = extractPartialJSON()

        // Get available keys from the detector
        let keys = detector.getAvailableKeys(from: partialJSON)
        return keys
    }

    private func extractPartialJSON() -> String {
        // Find the start of JSON in generated text
        if let jsonStart = generatedText.firstIndex(of: "{") {
            return String(generatedText[jsonStart...])
        }
        return ""
    }

    private func extractCurrentKey() -> String {
        // Extract the key from the current position in the JSON
        // Look for the most recent quote-delimited string before the current position
        let partialJSON = extractPartialJSON()

        // Find the last opening quote for a key
        var keyStart: String.Index? = nil
        var keyEnd: String.Index? = nil
        var inString = false
        var escaped = false
        var currentKey = ""

        for (index, char) in partialJSON.enumerated() {
            let strIndex = partialJSON.index(partialJSON.startIndex, offsetBy: index)

            if escaped {
                escaped = false
                if inString {
                    currentKey.append(char)
                }
                continue
            }

            if char == "\\" {
                escaped = true
                if inString {
                    currentKey.append(char)
                }
                continue
            }

            if char == "\"" {
                if !inString {
                    // Start of a string
                    inString = true
                    keyStart = strIndex
                    currentKey = ""
                } else {
                    // End of a string
                    inString = false
                    keyEnd = strIndex

                    // Check if this is followed by a colon (making it a key)
                    var foundColon = false
                    var checkIndex = partialJSON.index(after: strIndex)
                    while checkIndex < partialJSON.endIndex {
                        let nextChar = partialJSON[checkIndex]
                        if nextChar == ":" {
                            foundColon = true
                            break
                        } else if !nextChar.isWhitespace {
                            break
                        }
                        checkIndex = partialJSON.index(after: checkIndex)
                    }

                    if !foundColon {
                        // This is the current key being generated
                        return currentKey
                    }
                }
            } else if inString {
                currentKey.append(char)
            }
        }

        // If we're still in a string, return the current accumulated key
        if inString {
            return currentKey
        }

        return ""
    }

    private func printKeyDetection(_ keyName: String) {
        let level = jsonExtractor.getNestingLevel()

        if level == 0 {
            print("[KeyDetection] 🔑 Root key: \"\(keyName)\"")
        } else {
            let indent = String(repeating: "  ", count: level - 1)
            print("[KeyDetection] \(indent)📍 Level \(level) key: \"\(keyName)\"")
        }
    }

    // MARK: - Display Helpers

    private func displayStepInfo(from logits: MLXArray, step: Int, isKey: Bool) {
        let vocabSize = logits.dim(logits.ndim - 1)

        var logits = logits
        if logits.dtype == .bfloat16 {
            logits = logits.asType(.float32)
        }

        let probs = softmax(logits, axis: -1)
        let candidates = extractTopCandidates(logits: logits, probs: probs, k: topK)
        let entropy = calculateEntropy(probs: probs)

        // Get current available keys
        let availableKeys = getCurrentAvailableKeys()

        // Display the info
        displayStep(step: step, entropy: entropy, availableKeys: availableKeys, topCandidates: candidates, isKey: isKey)
    }

    private func extractTopCandidates(logits: MLXArray, probs: MLXArray, k: Int) -> [(tokenId: Int32, probability: Float, logit: Float, text: String?)] {
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
            (tokenId: tokenId,
             probability: probsArray[Int(tokenId)],
             logit: logitsArray[Int(tokenId)],
             text: tokenizer.decode([tokenId]))
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

    private func displayStep(step: Int, entropy: Float, availableKeys: [String], topCandidates: [(tokenId: Int32, probability: Float, logit: Float, text: String?)], isKey: Bool) {
        // Only display entropy for key generation
        guard isKey else { return }

        // Build constraint display for key generation context
        let constraintDisplay: String
        if !availableKeys.isEmpty {
            constraintDisplay = " [\(availableKeys.joined(separator: ", "))]"
        } else {
            constraintDisplay = " []"
        }

        let entropyStr = entropy.isNaN ? "0.00" : String(format: "%.2f", entropy)
        print("\n[Step \(step)] Entropy: \(entropyStr) (\(entropyDescription(entropy)))\(constraintDisplay)")

        for (index, candidate) in topCandidates.prefix(min(5, topK)).enumerated() {
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