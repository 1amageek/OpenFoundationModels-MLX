import Foundation
import MLX
import MLXNN
import MLXLMCommon
import Synchronization

/// LogitProcessor that detects and logs JSON key generation with context-aware schema validation
/// Uses JSONSchemaContextDetector with SchemaNode for type-safe schema handling
public final class KeyDetectionLogitProcessor: LogitProcessor, Sendable {

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
    private let modelCard: any ModelCard
    private let schemaNode: SchemaNode
    private let schemaDetector: JSONSchemaContextDetector

    // Core state management - protected by Mutex
    private struct MutableState {
        var jsonExtractor = JSONExtractor()
        var generatedText = ""
        var detectedKeys: [String] = []
        var stepCount: Int = 0
        var lastWasInKeyGeneration: Bool = false
        var pendingLogitInfo: LogitInfo? = nil
    }

    private let state = Mutex(MutableState())

    // Public access for testing
    public var allDetectedKeys: [String] {
        return state.withLock { $0.detectedKeys }
    }

    // MARK: - Initialization

    public init(
        tokenizer: TokenizerAdapter,
        modelCard: any ModelCard,
        schemaKeys: [String]? = nil,
        nestedSchemas: [String: [String]]? = nil,
        verbose: Bool = true,
        topK: Int = 5,
        showProbabilities: Bool = true
    ) {
        // Build schema first
        let schemaNode: SchemaNode
        if let rootKeys = schemaKeys {
            var tempSchema: [String: Any] = [
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

            tempSchema["properties"] = properties
            schemaNode = SchemaNode.from(jsonSchema: tempSchema)
        } else {
            // Even if no schema keys provided, create empty schema
            schemaNode = SchemaNode(kind: .object)
        }

        // Initialize all properties
        self.tokenizer = tokenizer
        self.modelCard = modelCard
        self.verbose = verbose
        self.topK = topK
        self.showProbabilities = showProbabilities
        self.schemaNode = schemaNode
        self.schemaDetector = JSONSchemaContextDetector(schema: schemaNode)

        // ModelCard is now required, no need for isActive flag
    }

    // Alternative init with direct JSON Schema
    public init(
        tokenizer: TokenizerAdapter,
        jsonSchema: [String: Any],
        modelCard: any ModelCard,
        verbose: Bool = true,
        topK: Int = 5,
        showProbabilities: Bool = true
    ) {
        self.tokenizer = tokenizer
        self.schemaNode = SchemaNode.from(jsonSchema: jsonSchema)
        self.modelCard = modelCard
        self.verbose = verbose
        self.topK = topK
        self.showProbabilities = showProbabilities
        self.schemaDetector = JSONSchemaContextDetector(schema: self.schemaNode)

        // ModelCard is now required, no need for isActive flag

        if verbose {
            print("[KeyDetection] Initialized with JSON Schema")
            let rootKeys = self.schemaNode.objectKeys
            if !rootKeys.isEmpty {
                print("[KeyDetection] Root keys: \(rootKeys)")
            } else {
                print("[KeyDetection] WARNING: No properties in JSON Schema!")
            }
        }
    }

    // Init with SchemaNode directly (preferred for type safety)
    public init(
        tokenizer: TokenizerAdapter,
        schemaNode: SchemaNode,
        modelCard: any ModelCard,
        verbose: Bool = true,
        topK: Int = 5,
        showProbabilities: Bool = true
    ) {
        self.tokenizer = tokenizer
        self.schemaNode = schemaNode
        self.modelCard = modelCard
        self.verbose = verbose
        self.topK = topK
        self.showProbabilities = showProbabilities
        self.schemaDetector = JSONSchemaContextDetector(schema: schemaNode)

        // ModelCard is now required, no need for isActive flag

        if verbose {
            print("[KeyDetection] Initialized with SchemaNode")
            let rootKeys = schemaNode.objectKeys
            if !rootKeys.isEmpty {
                print("[KeyDetection] Root keys: \(rootKeys)")
            } else {
                print("[KeyDetection] WARNING: No properties in schema!")
            }
        }
    }

    // MARK: - LogitProcessor Protocol

    public func prompt(_ prompt: MLXArray) {
        // Reset all state for new generation
        state.withLock { state in
            state.jsonExtractor.reset()
            state.generatedText = ""
            state.detectedKeys = []
            state.stepCount = 0
            state.lastWasInKeyGeneration = false
            state.pendingLogitInfo = nil
        }

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
        let (stepCount, jsonExtractor, lastWasInKeyGeneration, pendingInfo) = state.withLock { state in
            state.stepCount += 1
            let pending = state.pendingLogitInfo
            return (state.stepCount, state.jsonExtractor, state.lastWasInKeyGeneration, pending)
        }

        // Display any pending logit info from previous step
        if let pending = pendingInfo {
            if verbose && showProbabilities {
                // Context keys are already saved in LogitInfo
                displayStep(pending, isKey: lastWasInKeyGeneration, contextKeys: [])
            }
            state.withLock { $0.pendingLogitInfo = nil }
        }

        // Check if we're in JSON key generation phase specifically
        let isKeyGenerationPhase: Bool = {
            guard jsonExtractor.isInJSON else { return false }

            // Check the current phase to see if we're generating a key
            switch jsonExtractor.getCurrentPhase() {
            case .inObject(.expectKeyFirstQuote),   // About to start a key
                 .inObject(.expectKeyOrEnd),         // Could start a new key
                 .inString(.body(kind: .key, _)):    // Currently in a key string
                return true
            default:
                return false
            }
        }()

        // Only save logit info when we're in key generation phase
        if isKeyGenerationPhase {
            let logitInfo = buildLogitInfo(from: logits, step: stepCount)
            state.withLock { $0.pendingLogitInfo = logitInfo }
        }

        return logits
    }

    public func didSample(token: MLXArray) {
        // Always process for observation, regardless of constraint mode

        let tokenId = token.item(Int32.self)
        let text = tokenizer.decode([tokenId])

        state.withLock { state in
            state.generatedText += text

            // Track previous phase for key detection
            var previousPhase: JSONStateMachine.Phase? = nil

            // Process each character in the decoded text
            for char in text {
                // Get phase before processing
                if state.jsonExtractor.isInJSON {
                    previousPhase = state.jsonExtractor.getCurrentPhase()
                }

                // Try to extract JSON - now handles multiple JSONs automatically
                let shouldProcess = state.jsonExtractor.processCharacter(char)

                if shouldProcess && state.jsonExtractor.isInJSON {
                    // Get current phase after processing
                    let currentPhase = state.jsonExtractor.getCurrentPhase()

                    // Handle key detection - check if we transitioned from key to colon
                    if let prevPhase = previousPhase {
                        if case .inString(.body(kind: .key, escaped: false)) = prevPhase {
                            if case .inObject(.expectColon) = currentPhase {
                                // Key just completed - use the accumulated key from JSONExtractor
                                let keyName = extractCurrentKey(from: state.generatedText)
                                if !keyName.isEmpty {
                                    state.detectedKeys.append(keyName)

                                    if verbose {
                                        let level = state.jsonExtractor.getNestingLevel()
                                        printKeyDetection(keyName, level: level)
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
                        state.lastWasInKeyGeneration = true
                    default:
                        state.lastWasInKeyGeneration = false
                    }
                } else {
                    state.lastWasInKeyGeneration = false
                }
            }

            // Show selected token info only for key generation
            if verbose && showProbabilities && state.jsonExtractor.isInJSON && state.lastWasInKeyGeneration {
                let displayText = formatTokenForDisplay(text)
                print("✅ [\(tokenId)] → \"\(displayText)\" 🔑 KEY TOKEN")
  
                let currentPhase = state.jsonExtractor.getCurrentPhase()

                // Display constraints when we're generating a key (including in the middle of a key)
                switch currentPhase {
                case .inObject(.expectKeyFirstQuote), .inObject(.inKey), .inString(.body(kind: .key, _)):
                    let partialJSON = extractPartialJSON(from: state.generatedText)
                    let availableKeys = getCurrentAvailableKeys(from: state.generatedText)

                    // Include entropy if available from pending logit info
                    if let pendingInfo = state.pendingLogitInfo {
                        let entropyDesc = entropyDescription(pendingInfo.entropy)
                        print("\n📋 [Entropy: \(String(format: "%.2f", pendingInfo.entropy)) (\(entropyDesc))] Available keys: [\(availableKeys.joined(separator: ", "))]")
                    } else {
                        print("\n📋 Available keys at current context: [\(availableKeys.joined(separator: ", "))]")
                    }

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
    }

    public func finish() {
        // Display any pending logit info
        let (pendingInfo, lastWasInKeyGeneration, detectedKeys) = state.withLock { state in
            (state.pendingLogitInfo, state.lastWasInKeyGeneration, state.detectedKeys)
        }

        if let pending = pendingInfo {
            if verbose && showProbabilities {
                // Context keys are already saved in LogitInfo
                displayStep(pending, isKey: lastWasInKeyGeneration, contextKeys: [])
            }
            state.withLock { $0.pendingLogitInfo = nil }
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

    private func shouldApplyConstraints() -> Bool {
        let generatedText = state.withLock { $0.generatedText }
        return modelCard.shouldActivateProcessor(generatedText, processor: self)
    }

    private func getCurrentAvailableKeys(from generatedText: String) -> [String] {
        // Get the partial JSON generated so far
        let partialJSON = extractPartialJSON(from: generatedText)

        // Get available keys from the detector
        let keys = schemaDetector.getAvailableKeys(from: partialJSON)
        return keys
    }

    private func getCurrentAvailableKeys() -> [String] {
        let generatedText = state.withLock { $0.generatedText }
        return getCurrentAvailableKeys(from: generatedText)
    }

    private func extractPartialJSON(from generatedText: String) -> String {
        // Find the start of JSON in generated text
        if let jsonStart = generatedText.firstIndex(of: "{") {
            return String(generatedText[jsonStart...])
        }
        return ""
    }

    private func extractPartialJSON() -> String {
        let generatedText = state.withLock { $0.generatedText }
        return extractPartialJSON(from: generatedText)
    }

    private func extractCurrentKey(from generatedText: String) -> String {
        // Extract the key from the current position in the JSON
        // Look for the most recent quote-delimited string before the current position
        let partialJSON = extractPartialJSON(from: generatedText)

        // Find the last opening quote for a key
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
                    currentKey = ""
                } else {
                    // End of a string
                    inString = false

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

    private func printKeyDetection(_ keyName: String, level: Int) {
        if level == 0 {
            print("[KeyDetection] 🔑 Root key: \"\(keyName)\"")
        } else {
            let indent = String(repeating: "  ", count: level - 1)
            print("[KeyDetection] \(indent)📍 Level \(level) key: \"\(keyName)\"")
        }
    }

    // MARK: - Logit Analysis

    private func buildLogitInfo(from logits: MLXArray, step: Int) -> LogitInfo {
        let vocabSize = logits.dim(logits.ndim - 1)

        var logits = logits
        if logits.dtype == .bfloat16 {
            logits = logits.asType(.float32)
        }

        let probs = softmax(logits, axis: -1)
        let candidates = extractTopCandidates(logits: logits, probs: probs, k: topK)
        let entropy = calculateEntropy(probs: probs)

        // Get current context
        let currentPath = "" // Will be filled from context tracking
        let currentKeys = getCurrentAvailableKeys()

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
        let allProbs = probs.asArray(Float.self)
        let allLogits = logits.asArray(Float.self)

        return topIndices.reversed().map { index in
            LogitInfo.Candidate(
                tokenId: index,
                probability: allProbs[Int(index)],
                logit: allLogits[Int(index)],
                text: tokenizer.decode([index])
            )
        }
    }

    private func calculateEntropy(probs: MLXArray) -> Float {
        let epsilon: Float = 1e-10
        let clippedProbs = probs + epsilon
        let logProbs = MLX.log(clippedProbs)
        let entropy = -sum(probs * logProbs, axis: -1)

        entropy.eval()
        let result = entropy.item(Float.self)

        if result.isNaN || result.isInfinite {
            return 0.0
        }

        return max(0.0, result)
    }

    // MARK: - Display Helpers

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
        text.replacingOccurrences(of: "\n", with: "\\n")
            .replacingOccurrences(of: "\t", with: "\\t")
            .replacingOccurrences(of: "\r", with: "\\r")
    }


    private func entropyDescription(_ entropy: Float) -> String {
        switch entropy {
        case ..<0.5:
            return "🟢 Very Confident"
        case 0.5..<1.5:
            return "🟡 Confident"
        case 1.5..<3.0:
            return "🟠 Somewhat Uncertain"
        case 3.0..<5.0:
            return "🔴 Uncertain"
        default:
            return "⚫ Very Uncertain"
        }
    }
}
