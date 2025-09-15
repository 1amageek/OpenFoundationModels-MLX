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
        self.schemaNode = schemaNode
        self.schemaDetector = JSONSchemaContextDetector(schema: schemaNode)

        // ModelCard is now required, no need for isActive flag
    }

    // Alternative init with direct JSON Schema
    public init(
        tokenizer: TokenizerAdapter,
        jsonSchema: [String: Any],
        modelCard: any ModelCard,
        verbose: Bool = false,
        topK: Int = 5,
        showProbabilities: Bool = true
    ) {
        self.tokenizer = tokenizer
        self.schemaNode = SchemaNode.from(jsonSchema: jsonSchema)
        self.modelCard = modelCard
        self.verbose = verbose
        self.topK = topK
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
    ) {
        self.tokenizer = tokenizer
        self.schemaNode = schemaNode
        self.modelCard = modelCard
        self.verbose = verbose
        self.topK = topK
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
            // Safely handle prompt tokens - avoid MLX operations in test environment
            let tokens = prompt.asArray(Int32.self)
            print("Prompt: \(tokens.count) tokens\n")
        }
    }

    public func process(logits: MLXArray) -> MLXArray {
        let (stepCount, jsonExtractor, lastWasInKeyGeneration, pendingInfo, generatedText) = state.withLock { state in
            state.stepCount += 1
            let pending = state.pendingLogitInfo
            return (state.stepCount, state.jsonExtractor, state.lastWasInKeyGeneration, pending, state.generatedText)
        }

        // Display any pending logit info from previous step
        if let pending = pendingInfo {
            if verbose {
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

        // Apply constraints if we should and we're in the right phase
        var modifiedLogits = logits
        if shouldApplyConstraints() && isKeyGenerationPhase {
            modifiedLogits = applyKeyConstraints(logits: logits, generatedText: generatedText)
        }

        return modifiedLogits
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
            if verbose && state.jsonExtractor.isInJSON && state.lastWasInKeyGeneration {
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
            if verbose {
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

    // MARK: - ADAPT Key Constraint Application

    private func applyKeyConstraints(logits: MLXArray, generatedText: String) -> MLXArray {
        // Get the current JSON phase
        let phase = state.withLock { $0.jsonExtractor.getCurrentPhase() }

        // Check if we're currently generating a key (in the middle of it)
        if case .inString(.body(kind: .key, _)) = phase {
            // We're in the middle of generating a key
            return applyKeyTokenConstraints(logits: logits, generatedText: generatedText)
        }

        // Check if we're about to start a new key
        guard case .inObject(let objPhase) = phase,
              (objPhase == .expectKeyFirstQuote || objPhase == .expectKeyOrEnd) else {
            // Not at a position where we're starting a new key
            return logits
        }

        // Get available keys for current context
        let availableKeys = getCurrentAvailableKeys(from: generatedText)

        // If no schema keys available, return unmodified
        guard !availableKeys.isEmpty else {
            return logits
        }

        // Get the opening quote token for keys
        guard let quoteTokenId = findQuoteTokenId() else {
            // Can't find quote token, return unmodified
            return logits
        }

        // Simple constraint: boost the quote token to start a key properly
        // This guides the model to start generating a valid key
        var modifiedLogits = logits

        // Convert to float32 if needed for manipulation
        if modifiedLogits.dtype == .bfloat16 {
            modifiedLogits = modifiedLogits.asType(.float32)
        }

        // Boost the quote token probability slightly
        // This encourages starting a key when we have valid keys to generate
        let boostAmount: Float = 2.0
        var logitsArray = modifiedLogits.asArray(Float.self)
        logitsArray[Int(quoteTokenId)] += boostAmount

        // Create new MLXArray with modified logits
        modifiedLogits = MLXArray(logitsArray.map { Float32($0) }).reshaped(logits.shape)

        if verbose {
            print("[ADAPT] Applied constraint boost for starting key generation. Available keys: \(availableKeys.joined(separator: ", "))")
        }

        return modifiedLogits
    }

    private func applyKeyTokenConstraints(logits: MLXArray, generatedText: String) -> MLXArray {
        // Get the partial key being generated
        let partialKey = extractPartialKeyBeingGenerated(from: generatedText)

        // Get available keys for current context
        let availableKeys = getCurrentAvailableKeys(from: generatedText)

        // If no schema keys available, return unmodified
        guard !availableKeys.isEmpty else {
            return logits
        }

        // Find keys that match the partial key so far
        let matchingKeys = availableKeys.filter { $0.hasPrefix(partialKey) }

        // If no keys match the partial, return unmodified (no error, as requested)
        guard !matchingKeys.isEmpty else {
            if verbose && !partialKey.isEmpty {
                print("[ADAPT] No matching keys for partial: '\(partialKey)'. Available: \(availableKeys.joined(separator: ", "))")
            }
            return logits
        }

        // Get all possible next characters from matching keys
        var validNextChars = Set<String>()
        for key in matchingKeys {
            if key.count > partialKey.count {
                let nextCharIndex = key.index(key.startIndex, offsetBy: partialKey.count)
                let nextChar = String(key[nextCharIndex])
                validNextChars.insert(nextChar)
            }
        }

        // Special case: if all matching keys are complete (same length as partial), add quote
        if matchingKeys.allSatisfy({ $0.count == partialKey.count }) {
            validNextChars.insert("\"")
        }

        // If no valid next characters, return unmodified
        guard !validNextChars.isEmpty else {
            return logits
        }

        // Convert to float32 if needed
        var modifiedLogits = logits
        if modifiedLogits.dtype == .bfloat16 {
            modifiedLogits = modifiedLogits.asType(.float32)
        }

        // Apply boost to tokens that represent valid next characters
        var logitsArray = modifiedLogits.asArray(Float.self)
        let boostAmount: Float = 3.0

        var boostedTokens = 0
        for nextChar in validNextChars {
            // Encode the character to get token IDs
            let tokenIds = tokenizer.encode(nextChar)
            for tokenId in tokenIds {
                if tokenId >= 0 && tokenId < Int32(logitsArray.count) {
                    logitsArray[Int(tokenId)] += boostAmount
                    boostedTokens += 1
                }
            }
        }

        // Create new MLXArray with modified logits
        modifiedLogits = MLXArray(logitsArray.map { Float32($0) }).reshaped(logits.shape)

        if verbose && boostedTokens > 0 {
            print("[ADAPT] Boosted \(boostedTokens) tokens for key continuation. Partial: '\(partialKey)' → Matching: \(matchingKeys.joined(separator: ", "))")
        }

        return modifiedLogits
    }

    private func extractPartialKeyBeingGenerated(from generatedText: String) -> String {
        // Find the last opening quote for a key that hasn't been closed
        let partialJSON = extractPartialJSON(from: generatedText)

        var inString = false
        var isKey = false
        var escaped = false
        var currentKey = ""
        var depth = 0

        for char in partialJSON {
            if escaped {
                escaped = false
                if inString && isKey {
                    currentKey.append(char)
                }
                continue
            }

            if char == "\\" {
                escaped = true
                continue
            }

            switch char {
            case "\"":
                if !inString {
                    // Starting a string - could be a key
                    inString = true
                    isKey = true  // Assume it's a key until we see otherwise
                    currentKey = ""
                } else {
                    // Ending a string
                    inString = false
                    if !isKey {
                        currentKey = ""
                    }
                    // Don't clear currentKey here, we might still be building it
                }

            case ":":
                if !inString {
                    // This confirms the previous string was a key
                    currentKey = ""  // Key is complete, reset
                    isKey = false
                }

            case ",", "{", "[":
                if !inString {
                    isKey = false
                    if char == "{" { depth += 1 }
                    if char == "[" { isKey = false }
                }

            case "}", "]":
                if !inString {
                    depth -= 1
                    isKey = false
                }

            default:
                if inString && isKey {
                    currentKey.append(char)
                }
            }
        }

        // If we're still in a string and it's a key, return the partial
        return inString && isKey ? currentKey : ""
    }

    private func findQuoteTokenId() -> Int32? {
        // Try to find the quote token ID
        // First check if tokenizer has special token detection
        if let mlxTokenizer = tokenizer as? MLXLLMTokenizer {
            let specialTokens = mlxTokenizer.findSpecialTokens()
            return specialTokens.quote
        }

        // Fallback: encode a quote and get the first token
        let quoteTokens = tokenizer.encode("\"")
        return quoteTokens.first
    }
}
