import Foundation
@preconcurrency import MLX
import MLXLMCommon
import MLXLLM
@preconcurrency import Tokenizers
import Synchronization

/// Error types for JSON generation with helpful recovery suggestions
public enum JSONGenerationError: Error, LocalizedError {
    case noValidTokens(partialKey: String, position: Int)
    case invalidTokenSelected(token: Int32, partialKey: String, expectedTokens: Set<Int32>)
    case emptyConstraints
    case schemaViolation(reason: String)
    
    public var errorDescription: String? {
        switch self {
        case .noValidTokens(let partialKey, let position):
            return """
                JSON generation failed: No valid tokens found for key '\(partialKey)' at position \(position).
                
                Possible solutions:
                1. Check that the schema keys are correctly spelled
                2. Verify the tokenizer is compatible with the model
                3. Try increasing temperature to allow more variation
                4. Ensure the model has sufficient context for JSON generation
                """
            
        case .invalidTokenSelected(let token, let partialKey, let expectedTokens):
            let expectedStr = expectedTokens.prefix(5).map(String.init).joined(separator: ", ")
            let suffix = expectedTokens.count > 5 ? " (and \(expectedTokens.count - 5) more)" : ""
            return """
                JSON generation failed: Invalid token \(token) selected for key '\(partialKey)'.
                Expected one of: \(expectedStr)\(suffix)
                
                Possible solutions:
                1. This is an internal consistency error - please retry
                2. If the error persists, try a different model
                3. Report this issue with the model ID and prompt
                """
            
        case .emptyConstraints:
            return """
                JSON generation failed: No valid tokens available for constraint.
                
                Possible solutions:
                1. Verify the schema has at least one key defined
                2. Check that the tokenizer vocabulary includes JSON symbols
                3. Ensure the model supports JSON generation
                """
            
        case .schemaViolation(let reason):
            return """
                JSON generation failed: \(reason)
                
                Possible solutions:
                1. Try regenerating with a higher temperature
                2. Ensure the prompt clearly specifies the expected JSON format
                3. Check if the model tends to generate the required keys
                4. Consider simplifying the schema if it's very complex
                """
        }
    }
    
    public var recoverySuggestion: String? {
        switch self {
        case .noValidTokens:
            return "The generation will be retried automatically with adjusted parameters."
        case .invalidTokenSelected:
            return "The generation will be retried from the last valid state."
        case .emptyConstraints:
            return "Please verify your schema configuration and try again."
        case .schemaViolation:
            return "The generation will be retried with exponential backoff."
        }
    }
}


/// TokenTrie-based constraint implementation as LogitProcessor
/// Hard masks schema keys during JSON generation, uses soft hints for values
public final class TokenTrieLogitProcessor: LogitProcessor, Sendable {
    
    private let schemaRoot: SchemaNode?
    private let schemaIndex: SchemaTrieIndex?
    private let tokenizer: any Tokenizer
    private let tokenizerAdapter: MLXLLMTokenizer
    private let specialTokens: MLXLLMTokenizer.SpecialTokens
    private let maskHintGenerator: JSONMaskHintGenerator
    
    
    private enum Context: Sendable {
        case object(SchemaNode?)  // Made optional to handle unknown objects
        case array(SchemaNode?)
    }
    
    private struct ProcessorSnapshot: Sendable {
        let jsonPhase: JSONStateMachine.Phase
        let tokenPath: TokenTrie.Path
        let isGenerating: Bool
        let tokenCount: Int
        let contextStack: [Context]
        let pendingKey: String?
        let currentObjectNode: SchemaNode?
        
        init(jsonPhase: JSONStateMachine.Phase = .root, 
             tokenPath: TokenTrie.Path, 
             isGenerating: Bool = false, 
             tokenCount: Int = 0,
             contextStack: [Context] = [],
             pendingKey: String? = nil,
             currentObjectNode: SchemaNode? = nil) {
            self.jsonPhase = jsonPhase
            self.tokenPath = tokenPath
            self.isGenerating = isGenerating
            self.tokenCount = tokenCount
            self.contextStack = contextStack
            self.pendingKey = pendingKey
            self.currentObjectNode = currentObjectNode
        }
    }
    
    private let lightweightState: Mutex<ProcessorSnapshot>
    private let errorState = Mutex<JSONGenerationError?>(nil)
    
    private struct HeavyState: Sendable {
        var jsonStateMachine: JSONStateMachine = JSONStateMachine()
        var generatedTokens: [Int32] = []
        var promptTokens: [Int32] = []
    }
    
    private let heavyState = Mutex<HeavyState>(HeavyState())
    
    // New init with SchemaNode for nested object support
    public init(schema: SchemaNode?, tokenizer: any Tokenizer) {
        let adapter = MLXLLMTokenizer(tokenizer: tokenizer)
        self.schemaRoot = schema
        self.tokenizer = tokenizer
        self.tokenizerAdapter = adapter
        self.specialTokens = adapter.findSpecialTokens()
        self.maskHintGenerator = JSONMaskHintGenerator.forSchemaConstrainedDecoding(
            specialTokens: specialTokens,
            includeWhitespace: false
        )
        
        // Build schema index if schema provided
        if let schema = schema {
            self.schemaIndex = SchemaTrieIndex(root: schema, tokenizer: adapter)
        } else {
            self.schemaIndex = nil
        }
        
        var initialContextStack: [Context] = []
        var initialObjectNode: SchemaNode? = nil
        if let root = schema {
            initialContextStack = [.object(root)]
            initialObjectNode = root
        }
        
        let initialPath = TokenTrie.Path()
        let initialSnapshot = ProcessorSnapshot(
            tokenPath: initialPath,
            contextStack: initialContextStack,
            currentObjectNode: initialObjectNode
        )
        self.lightweightState = Mutex(initialSnapshot)
        
    }
    
    
    
    public func prompt(_ prompt: MLXArray) {
            heavyState.withLock { state in
            let flat = prompt.reshaped([-1])
            let count = flat.dim(0)
            state.promptTokens = (0..<count).map { i in
                Int32(flat[i].item(Int.self))
            }
            state.jsonStateMachine.reset()
            state.generatedTokens.removeAll()
        }
        
        var newContextStack: [Context] = []
        var newObjectNode: SchemaNode? = nil
        if let root = schemaRoot {
            newContextStack = [.object(root)]
            newObjectNode = root
        }
        
        let newPath = TokenTrie.Path()
        let newSnapshot = ProcessorSnapshot(
            jsonPhase: .root,
            tokenPath: newPath,
            isGenerating: false,
            tokenCount: 0,
            contextStack: newContextStack,
            pendingKey: nil,
            currentObjectNode: newObjectNode
        )
        lightweightState.withLock { $0 = newSnapshot }
        errorState.withLock { $0 = nil }
    }
    
    
    private func getCurrentTrie() -> TokenTrie? {
        let snap = lightweightState.withLock { $0 }
        
        // Get trie for current object node
        if let node = snap.currentObjectNode,
           let index = schemaIndex {
            return index.trie(for: node)
        }
        
        return nil
    }
    
    private func updateContextForValue(key: String?, valueStart: Character, currentSnap: ProcessorSnapshot) -> ProcessorSnapshot {
        var newStack = currentSnap.contextStack
        var newObjectNode = currentSnap.currentObjectNode
        var newPendingKey: String? = currentSnap.pendingKey
        
        switch valueStart {
        case "{":
            // Entering nested object
            if let key = key,
               let parentNode = currentSnap.currentObjectNode,
               let childNode = parentNode.properties[key],
               childNode.kind == .object {
                newStack.append(.object(childNode))
                newObjectNode = childNode
            } else {
                // Unknown object structure
                newStack.append(.object(nil))
                // Keep parent node as current
            }
            newPendingKey = nil
            
        case "[":
            // Entering array
            if let key = key,
               let parentNode = currentSnap.currentObjectNode,
               let childNode = parentNode.properties[key],
               childNode.kind == .array {
                newStack.append(.array(childNode.items))
            } else {
                newStack.append(.array(nil))
            }
            newPendingKey = nil
            
        default:
            // Primitive value
            newPendingKey = nil
        }
        
        return ProcessorSnapshot(
            jsonPhase: currentSnap.jsonPhase,
            tokenPath: currentSnap.tokenPath,
            isGenerating: currentSnap.isGenerating,
            tokenCount: currentSnap.tokenCount,
            contextStack: newStack,
            pendingKey: newPendingKey,
            currentObjectNode: newObjectNode
        )
    }
    
    private func popContext(type: Context, from snapshot: ProcessorSnapshot) -> ProcessorSnapshot {
        var newStack = snapshot.contextStack
        var newObjectNode = snapshot.currentObjectNode
        
        // Pop until we find the matching context type
        while let last = newStack.last {
            newStack.removeLast()
            
            switch (type, last) {
            case (.object, .object):
                // Found matching object, update current node
                newObjectNode = nil
                for ctx in newStack.reversed() {
                    if case .object(let node) = ctx, let actualNode = node {
                        newObjectNode = actualNode
                        break
                    }
                }
                break
                
            case (.array, .array):
                break
                
            default:
                continue
            }
            break
        }
        
        return ProcessorSnapshot(
            jsonPhase: snapshot.jsonPhase,
            tokenPath: snapshot.tokenPath,
            isGenerating: snapshot.isGenerating,
            tokenCount: snapshot.tokenCount,
            contextStack: newStack,
            pendingKey: snapshot.pendingKey,
            currentObjectNode: newObjectNode
        )
    }
    
    
    private func processOptimized(logits: MLXArray) throws -> MLXArray {
        let snap = lightweightState.withLock { $0 }
        let jsonState = heavyState.withLock { $0.jsonStateMachine }
        let isInKey = isInKeyState(phase: snap.jsonPhase)

        lightweightState.withLock {
            $0 = ProcessorSnapshot(
                jsonPhase: snap.jsonPhase,
                tokenPath: snap.tokenPath,
                isGenerating: true,
                tokenCount: snap.tokenCount,
                contextStack: snap.contextStack,
                pendingKey: snap.pendingKey,
                currentObjectNode: snap.currentObjectNode
            )
        }
        
        let currentTrie = getCurrentTrie()

        let hint: JSONMaskHint? = currentTrie.flatMap { trie in
            maskHintGenerator.maskHint(
                for: jsonState,
                tokenTrie: trie,
                tokenPath: snap.tokenPath
            )
        }
        

        var allowed = Set<Int32>()
        var useHardMask = false

        if isInKey {
            if let trie = currentTrie {
                allowed = trie.getAllowedTokens(for: snap.tokenPath)
                
                if snap.tokenPath.isAtTerminal() {
                    let quoteCandidates = dynamicQuoteCandidates(from: logits, fallback: specialTokens.quoteTokens)
                    allowed.formUnion(quoteCandidates)
                    allowed.formUnion(specialTokens.backslashTokens)
                }
                useHardMask = true
                
                if allowed.isEmpty && !snap.tokenPath.isAtTerminal() {
                    throw JSONGenerationError.noValidTokens(
                        partialKey: tokenizerAdapter.decode(snap.tokenPath.tokens),
                        position: snap.tokenPath.tokens.count
                    )
                }
            } else {
                // No trie available - don't apply mask (use soft hints or raw logits)
                useHardMask = false
            }
        } else {
            if let h = hint {
                switch h.mode {
                case .hard:
                    allowed = h.allow
                    useHardMask = true
                case .soft:
                    // Empty allowed for soft bias
                    break
                }
            }
        }

        if useHardMask {
            return try applyHardMaskOptimized(to: logits, allowedTokens: allowed)
        }
        if let h = hint {
            if h.mode == .soft, !h.prefer.isEmpty {
                return MLXUtils.applySoftBias(logits: logits, preferredTokens: h.prefer, bias: 2.5)
            }
        }
        return logits
    }
    
    public func process(logits: MLXArray) -> MLXArray {
        do {
            errorState.withLock { $0 = nil }
            
            return try processOptimized(logits: logits)
        } catch let error as JSONGenerationError {
            errorState.withLock { $0 = error }
            Logger.error("[TokenTrieLogitProcessor] Error: \(error)")
            return applySafetyConstraints(logits)
        } catch {
            Logger.error("[TokenTrieLogitProcessor] Unexpected error: \(error)")
            return applySafetyConstraints(logits)
        }
    }
    
    private func applySafetyConstraints(_ logits: MLXArray) -> MLXArray {
        guard let eosToken = tokenizerAdapter.eosTokenId() else {
            return logits
        }
        
        let vocabSize = logits.dim(logits.ndim - 1)
        guard eosToken >= 0 && eosToken < vocabSize else {
            return logits
        }
        
        let eosMask = MLXUtils.createVocabMask(vocabSize: vocabSize, tokens: [eosToken])
        var shape = Array(repeating: 1, count: logits.ndim)
        shape[logits.ndim - 1] = vocabSize
        
        let boost = eosMask.reshaped(shape) * 5.0
        return logits + boost
    }
    
    public func didSample(token: MLXArray) {
        let tokenID = Int32(token.item(Int.self))
        
        do {
                let (newPhase, generatedText) = heavyState.withLock { state in
                state.generatedTokens.append(tokenID)
                
                let text = tokenizerAdapter.decodeToken(tokenID)
                for char in text {
                    state.jsonStateMachine.processCharacter(char)
                }
                
                return (state.jsonStateMachine.phase, text)
            }
            
            let prevSnapshot = lightweightState.withLock { $0 }
            var newTokenPath = prevSnapshot.tokenPath
            var newContextStack = prevSnapshot.contextStack
            var newPendingKey = prevSnapshot.pendingKey
            var newObjectNode = prevSnapshot.currentObjectNode
            
            let wasInKey = isInKeyState(phase: prevSnapshot.jsonPhase)
            let nowInKey = isInKeyState(phase: newPhase)
            
            
            let currentTrie = getCurrentTrie()
            
            if wasInKey && nowInKey {
                if let trie = currentTrie {
                    let success = newTokenPath.append(tokenID, in: trie)
                    if !success {
                        throw JSONGenerationError.invalidTokenSelected(
                            token: tokenID,
                            partialKey: tokenizerAdapter.decode(prevSnapshot.tokenPath.tokens),
                            expectedTokens: trie.getAllowedTokens(for: prevSnapshot.tokenPath)
                        )
                    }
                }
            } else if wasInKey && !nowInKey {
                if prevSnapshot.tokenPath.isAtTerminal() {
                    newPendingKey = prevSnapshot.tokenPath.getKeyName()
                }
                if let trie = currentTrie {
                    newTokenPath.reset(to: trie.root)
                } else {
                    newTokenPath.reset()
                }
            }
            
            if generatedText.contains("{") {
                // Check if this is the root object start
                let prevPhase = lightweightState.withLock { $0.jsonPhase }
                
                if prevPhase == .root {
                    // Root object: keep existing context [.object(root)] and currentObjectNode
                    // Just reset the token path for the root trie
                    if let node = newObjectNode, let trie = schemaIndex?.trie(for: node) {
                        newTokenPath.reset(to: trie.root)
                    } else {
                        newTokenPath.reset()
                    }
                } else if let last = newContextStack.last,
                   case .array(let itemNode) = last,
                   let objNode = itemNode, objNode.kind == .object {
                    // Array-of-object: descend into items' object node
                    newContextStack.append(.object(objNode))
                    newObjectNode = objNode
                    if let trie = schemaIndex?.trie(for: objNode) {
                        newTokenPath.reset(to: trie.root)
                    } else {
                        newTokenPath.reset()
                    }
                } else if let key = newPendingKey,
                          let parentNode = newObjectNode,
                          let childNode = parentNode.properties[key],
                          childNode.kind == .object {
                    // Object property that is another object
                    newContextStack.append(.object(childNode))
                    newObjectNode = childNode
                    if let trie = schemaIndex?.trie(for: childNode) {
                        newTokenPath.reset(to: trie.root)
                    } else {
                        newTokenPath.reset()
                    }
                } else {
                    // Only for truly unknown objects (not root)
                    newContextStack.append(.object(nil))
                    newObjectNode = nil  // Clear to disable parent's Trie
                    newTokenPath.reset()
                }
                newPendingKey = nil
            }
            
            if generatedText.contains("[") {
                if let key = newPendingKey,
                   let parentNode = newObjectNode,
                   let childNode = parentNode.properties[key],
                   childNode.kind == .array {
                    newContextStack.append(.array(childNode.items))
                } else {
                    newContextStack.append(.array(nil))
                }
                newPendingKey = nil
            }
            
            // Check for object/array exit
            if generatedText.contains("}") {
                // Pop object context
                while let last = newContextStack.last {
                    newContextStack.removeLast()
                    if case .object = last { break }
                }
                newObjectNode = nil
                for ctx in newContextStack.reversed() {
                    if case .object(let node) = ctx, let actualNode = node {
                        newObjectNode = actualNode
                        break
                    }
                }
                // Only reset if we have a valid node with a trie
                if let node = newObjectNode, let trie = schemaIndex?.trie(for: node) {
                    newTokenPath.reset(to: trie.root)
                } else {
                    newTokenPath.reset()  // Reset without trie
                }
            }
            
            if generatedText.contains("]") {
                while let last = newContextStack.last {
                    newContextStack.removeLast()
                    if case .array = last { break }
                }
                // Recompute current object node (mirror of '}' case)
                newObjectNode = nil
                for ctx in newContextStack.reversed() {
                    if case .object(let node) = ctx, let actualNode = node {
                        newObjectNode = actualNode
                        break
                    }
                }
                // Reset token path for the current object context
                if let node = newObjectNode, let trie = schemaIndex?.trie(for: node) {
                    newTokenPath.reset(to: trie.root)
                } else {
                    newTokenPath.reset()  // Reset without trie
                }
            }
            
            if case .inObject(.expectValueStart) = newPhase {
                let firstNonWS = generatedText.first { !$0.isWhitespace }
                if let char = firstNonWS,
                   char != "{" && char != "[" {
                    newPendingKey = nil
                }
            }
            
            let newSnapshot = ProcessorSnapshot(
                jsonPhase: newPhase,
                tokenPath: newTokenPath,
                isGenerating: prevSnapshot.isGenerating,
                tokenCount: prevSnapshot.tokenCount + 1,
                contextStack: newContextStack,
                pendingKey: newPendingKey,
                currentObjectNode: newObjectNode
            )
            lightweightState.withLock { $0 = newSnapshot }
            
        } catch let error as JSONGenerationError {
            errorState.withLock { $0 = error }
            Logger.error("[TokenTrieLogitProcessor] Token validation failed: \(error)")
        } catch {
            Logger.error("[TokenTrieLogitProcessor] Unexpected error: \(error)")
        }
    }
    
    
    public func getLastError() -> JSONGenerationError? {
        return errorState.withLock { $0 }
    }
    
    public func hasError() -> Bool {
        return errorState.withLock { $0 != nil }
    }
    
    public func clearError() {
        errorState.withLock { $0 = nil }
    }
    
    public func hasFatalError() -> Bool {
        return errorState.withLock { error in
            guard let error = error else { return false }
            switch error {
            case .noValidTokens, .invalidTokenSelected:
                return true
            case .emptyConstraints, .schemaViolation:
                return false
            }
        }
    }
    
    
    
    private func isInKeyState(phase: JSONStateMachine.Phase) -> Bool {
        if case .inString(let strPhase) = phase,
           case .body(let kind, _) = strPhase,
           kind == .key {
            return true
        }
        return false
    }
    
    /// GPU-optimized mask application
    private func applyHardMaskOptimized(to logits: MLXArray, allowedTokens: Set<Int32>) throws -> MLXArray {
        let vocab = logits.dim(logits.ndim - 1)
        let snap = lightweightState.withLock { $0 }

        // Only allow EOS when generation is complete
        var allow = allowedTokens
        if case .done = snap.jsonPhase, let eos = tokenizerAdapter.eosTokenId() {
            allow.insert(eos)
        }

        let indices = Array(allow.filter { $0 >= 0 && $0 < vocab })

        if indices.isEmpty {
            // Special handling for done phase with no EOS token
            if case .done = snap.jsonPhase {
                // No EOS token available - return original logits to allow natural termination
                return logits
            } else {
                // For other cases, apply safety constraints to encourage termination
                return applySafetyConstraints(logits)
            }
        }

        var mask = [Float](repeating: 0, count: vocab)
        for i in indices { mask[Int(i)] = 1 }
        let maskArray = MLXArray(mask)
        var shape = Array(repeating: 1, count: logits.ndim)
        shape[logits.ndim - 1] = vocab
        let reshaped = maskArray.reshaped(shape)
        let negInf = MLX.full(logits.shape, values: -Float.infinity)
        return MLX.where(reshaped .> 0, logits, negInf)
    }
    

    /// Dynamic quote candidates for tokenizers without standalone quote tokens
    private func dynamicQuoteCandidates(from logits: MLXArray, topK: Int = 30, fallback: Set<Int32>) -> Set<Int32> {
        // Use fallback if already available
        if !fallback.isEmpty { return fallback }
        
        // Use argPartition for O(V) complexity instead of O(V log V)
        let flat = logits.reshaped([-1])
        let n = flat.dim(0)
        let k = min(topK, n)
        
        // Partition to get top-k indices (kth largest at position n-k)
        let partitionedIndices = MLX.argPartition(flat, kth: n - k, axis: 0)
        
        // Get the top-k indices (they are at the end after partition)
        let topIndices = partitionedIndices[(n - k)..<n]
        
        // Extract indices and their values for sorting
        let indicesArray = topIndices.asArray(Int32.self)
        var indexValuePairs: [(Int32, Float)] = []
        for idx in indicesArray {
            let value = flat[Int(idx)].item(Float.self)
            indexValuePairs.append((idx, value))
        }
        
        // Sort by value descending (highest logits first)
        indexValuePairs.sort { $0.1 > $1.1 }
        
        var out = Set<Int32>()
        
        // Process from highest logit to lowest
        for (tid, _) in indexValuePairs {
            let piece = tokenizerAdapter.decodeToken(tid)
            if piece.contains("\"") { out.insert(tid) }
        }
        
        // Always allow escape sequences
        out.formUnion(specialTokens.backslashTokens)
        return out
    }
    
    
    
    public func debugState() -> String {
        let currentSnapshot = lightweightState.withLock { $0 }
        let error = errorState.withLock { $0 }
        
        let phaseDescription: String
        switch currentSnapshot.jsonPhase {
        case .root: phaseDescription = "root"
        case .inObject(let objPhase): phaseDescription = "inObject(\(objPhase))"
        case .inArray(let arrPhase): phaseDescription = "inArray(\(arrPhase))"
        case .inString(let strPhase): phaseDescription = "inString(\(strPhase))"
        case .inNumber(let numPhase): phaseDescription = "inNumber(\(numPhase))"
        case .inLiteral(let litPhase): phaseDescription = "inLiteral(\(litPhase))"
        case .done: phaseDescription = "done"
        case .error: phaseDescription = "error"
        }
        
        return """
        OptimizedTokenTrieLogitProcessor State:
        - JSON Phase: \(phaseDescription)
        - Token Path Length: \(currentSnapshot.tokenPath.tokens.count)
        - Is at Terminal: \(currentSnapshot.tokenPath.isAtTerminal())
        - Token Count: \(currentSnapshot.tokenCount)
        - Is Generating: \(currentSnapshot.isGenerating)
        - Last Error: \(error?.localizedDescription ?? "None")
        """
    }
}


extension TokenTrieLogitProcessor {
    /// Convenience initializer with default settings
    public convenience init(keys: [String], tokenizer: any Tokenizer) {
        let node = SchemaNode(
            kind: .object,
            properties: Dictionary(uniqueKeysWithValues: keys.map { ($0, SchemaNode.any) }),
            required: []
        )
        self.init(schema: node, tokenizer: tokenizer)
    }
    
    public func validateGenerated() -> Bool {
        return heavyState.withLock { state in
            let text = tokenizerAdapter.decode(state.generatedTokens)
            
            guard let data = text.data(using: String.Encoding.utf8),
                  let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else {
                return false
            }
            
            let jsonKeys = Set(json.keys)
            // Get current trie's keys
            let currentTrie = getCurrentTrie()
            let schemaKeys = currentTrie?.allKeys ?? Set()
            return jsonKeys.isSubset(of: schemaKeys)
        }
    }
}
