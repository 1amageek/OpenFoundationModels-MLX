import Foundation
@preconcurrency import MLX
import MLXLMCommon
import MLXLLM
@preconcurrency import Tokenizers
import Synchronization

/// DPDA × KeyTrie LogitProcessor
/// Implements deterministic pushdown automaton with KeyTrie for JSON generation
/// - Keys: Hard mask with TokenTrie constraints
/// - Values: Soft bias for type-specific tokens
/// - Error recovery: Closest key correction with Levenshtein distance
public final class DPDAKeyTrieLogitProcessor: LogitProcessor, @unchecked Sendable {
    
    // MARK: - Error Types
    
    public enum ProcessorError: Error, LocalizedError {
        case trieMismatch(partial: String)
        case emptyAllowedTokens
        case invalidPhase
        
        public var errorDescription: String? {
            switch self {
            case .trieMismatch(let partial):
                return "Trie mismatch for partial key: \(partial)"
            case .emptyAllowedTokens:
                return "No valid tokens allowed in current phase"
            case .invalidPhase:
                return "Invalid DPDA phase"
            }
        }
    }
    
    // MARK: - Properties
    
    private let schemaRoot: SchemaNode?
    private let schemaIndex: SchemaTrieIndex?
    private let tokenizer: any Tokenizer
    private let adapter: MLXLLMTokenizer
    private let specialTokens: MLXLLMTokenizer.SpecialTokens
    
    // DPDA state management
    private let dpda = JSONDPDA()
    
    // Schema context stack for nested objects and arrays
    private enum ContextFrame: Sendable {
        case object(node: SchemaNode?, allowedKeys: Set<String>)
        case array(itemSchema: SchemaNode?)
    }
    private let contextStack = Mutex<[ContextFrame]>([])
    private let currentNode = Mutex<SchemaNode?>(nil)
    private let currentArrayItemSchema = Mutex<SchemaNode?>(nil)
    
    // Key generation state
    private let triePath = Mutex<TokenTrie.Path>(TokenTrie.Path())
    private let keyBuffer = Mutex<String>("")
    private let confirmedKey = Mutex<String?>(nil)
    
    // Error tracking
    private let lastError = Mutex<ProcessorError?>(nil)
    
    // Performance optimizations
    private let tokenSearchCache = Mutex<[String: Set<Int32>]>([:])
    private let searchLimit = 50000  // Vocabulary search limit
    private let maxRetries = 3  // Max error recovery attempts
    private let retryCount = Mutex<Int>(0)
    
    // Soft bias configuration
    private let valueBias: Float = 2.5
    private let eosBias: Float = 3.0
    
    // MARK: - Initialization
    
    public init(
        schema: SchemaNode?,
        tokenizer: any Tokenizer,
        cachedIndex: SchemaTrieIndex? = nil
    ) {
        self.schemaRoot = schema
        self.tokenizer = tokenizer
        self.adapter = MLXLLMTokenizer(tokenizer: tokenizer)
        self.specialTokens = adapter.findSpecialTokens()
        
        // Initialize schema index
        if let cachedIndex = cachedIndex {
            self.schemaIndex = cachedIndex
            Logger.debug("[DPDAProcessor] Using cached SchemaTrieIndex")
        } else if let schema = schema {
            self.schemaIndex = SchemaTrieIndex(root: schema, tokenizer: adapter)
            Logger.debug("[DPDAProcessor] Created new SchemaTrieIndex")
        } else {
            self.schemaIndex = nil
        }
        
        // Initialize with root context
        currentNode.withLock { $0 = schema }
        if let trie = getCurrentTrie() {
            triePath.withLock { $0 = TokenTrie.Path(root: trie.root) }
        }
    }
    
    // MARK: - Helper Methods
    
    private func getCurrentTrie() -> TokenTrie? {
        let node = currentNode.withLock { $0 }
        guard let node = node, let index = schemaIndex else { return nil }
        return index.trie(for: node)
    }
    
    private func eosTokens() -> Set<Int32> {
        if let eos = adapter.eosTokenId() {
            return [eos]
        }
        return []
    }
    
    // MARK: - Token Search Cache
    
    private func findTokensContaining(_ substrings: [String]) -> Set<Int32> {
        tokenSearchCache.withLock { cache in
            var result = Set<Int32>()
            
            for substring in substrings {
                if let cached = cache[substring] {
                    result.formUnion(cached)
                    continue
                }
                
                var found = Set<Int32>()
                for tokenId in 0..<min(searchLimit, adapter.getVocabSize() ?? searchLimit) {
                    let id = Int32(tokenId)
                    let decoded = adapter.decodeToken(id)
                    if decoded.contains(substring) {
                        found.insert(id)
                    }
                }
                
                cache[substring] = found
                result.formUnion(found)
            }
            
            return result
        }
    }
    
    // MARK: - Closest Key Correction
    
    private func findClosestKey(_ input: String, in validKeys: Set<String>) -> String? {
        guard !validKeys.isEmpty else { return nil }
        
        let normalized = normalize(input)
        
        // Exact match (normalized)
        if let exact = validKeys.first(where: { normalize($0) == normalized }) {
            return exact
        }
        
        // Prefix match
        if let prefix = validKeys.first(where: { 
            normalize($0).hasPrefix(normalized) || normalized.hasPrefix(normalize($0))
        }) {
            return prefix
        }
        
        // Levenshtein distance (threshold: 2)
        var bestMatch: (key: String, distance: Int)?
        for key in validKeys {
            let distance = levenshteinDistance(normalized, normalize(key))
            if distance <= 2 {
                if bestMatch == nil || distance < bestMatch!.distance {
                    bestMatch = (key, distance)
                }
            }
        }
        
        return bestMatch?.key
    }
    
    private func normalize(_ s: String) -> String {
        s.lowercased()
            .replacingOccurrences(of: "_", with: "")
            .replacingOccurrences(of: "-", with: "")
            .trimmingCharacters(in: .whitespacesAndNewlines)
    }
    
    private func levenshteinDistance(_ s1: String, _ s2: String) -> Int {
        let a = Array(s1)
        let b = Array(s2)
        
        if a.isEmpty { return b.count }
        if b.isEmpty { return a.count }
        
        var matrix = Array(repeating: Array(repeating: 0, count: b.count + 1), count: a.count + 1)
        
        for i in 0...a.count { matrix[i][0] = i }
        for j in 0...b.count { matrix[0][j] = j }
        
        for i in 1...a.count {
            for j in 1...b.count {
                let cost = a[i-1] == b[j-1] ? 0 : 1
                matrix[i][j] = min(
                    matrix[i-1][j] + 1,      // deletion
                    matrix[i][j-1] + 1,      // insertion
                    matrix[i-1][j-1] + cost  // substitution
                )
            }
        }
        
        return matrix[a.count][b.count]
    }
    
    // MARK: - Soft Bias for Value Types
    
    private func getPreferredTokensForValueType(_ kind: SchemaNode.Kind) -> Set<Int32> {
        switch kind {
        case .string:
            return specialTokens.quoteTokens
            
        case .number:
            return findTokensContaining(["-", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9"])
            
        case .boolean:
            return findTokensContaining(["true", "false"])
            
        case .null:
            return findTokensContaining(["null"])
            
        case .object:
            return specialTokens.braceOpenTokens
            
        case .array:
            return specialTokens.bracketOpenTokens
            
        case .any:
            return []
        }
    }
    
    // MARK: - LogitProcessor Protocol
    
    public func prompt(_ prompt: MLXArray) {
        // Reset all state
        lastError.withLock { $0 = nil }
        dpda.reset()
        contextStack.withLock { $0.removeAll() }
        currentNode.withLock { $0 = schemaRoot }
        currentArrayItemSchema.withLock { $0 = nil }
        keyBuffer.withLock { $0 = "" }
        confirmedKey.withLock { $0 = nil }
        retryCount.withLock { $0 = 0 }
        
        // Reset trie path
        if let trie = getCurrentTrie() {
            triePath.withLock { $0 = TokenTrie.Path(root: trie.root) }
        } else {
            triePath.withLock { $0 = TokenTrie.Path() }
        }
        
        // Clear cache for new generation
        tokenSearchCache.withLock { $0.removeAll(keepingCapacity: true) }
    }
    
    public func process(logits: MLXArray) -> MLXArray {
        lastError.withLock { $0 = nil }
        
        var allowedTokens = Set<Int32>()
        var preferredTokens = Set<Int32>()
        
        let phase = dpda.phase
        
        switch phase {
        case .root:
            // JSON can start with {, [, or " (for simple values)
            allowedTokens.formUnion(specialTokens.braceOpenTokens)
            allowedTokens.formUnion(specialTokens.bracketOpenTokens)
            allowedTokens.formUnion(specialTokens.quoteTokens)
            
        case .inObject_expectKeyOrEnd:
            // Either start a key with " or close with }
            allowedTokens.formUnion(specialTokens.quoteTokens)
            allowedTokens.formUnion(specialTokens.braceCloseTokens)
            
        case .inObject_expectKey:
            // Must start key with "
            allowedTokens.formUnion(specialTokens.quoteTokens)
            
        case .inString(kind: .key):
            // Use TokenTrie for key constraints
            if let trie = getCurrentTrie() {
                let path = triePath.withLock { $0 }
                let trieTokens = trie.getAllowedTokens(for: path)
                allowedTokens.formUnion(trieTokens)
                
                // Always allow backslash for escaping
                allowedTokens.formUnion(specialTokens.backslashTokens)
                
                // If at terminal, allow closing quote
                if trie.canComplete(from: path) {
                    allowedTokens.formUnion(specialTokens.quoteTokens)
                }
                
                if allowedTokens.isEmpty {
                    // Fallback to quotes if no trie tokens
                    allowedTokens.formUnion(specialTokens.quoteTokens)
                    allowedTokens.formUnion(specialTokens.backslashTokens)
                }
            } else {
                // No schema, allow any string content
                allowedTokens.formUnion(specialTokens.quoteTokens)
                allowedTokens.formUnion(specialTokens.backslashTokens)
            }
            
        case .inObject_expectColon:
            // Must be colon after key
            allowedTokens.formUnion(specialTokens.colonTokens)
            
        case .inObject_expectValueStart:
            // Value can start with various tokens
            allowedTokens.formUnion(specialTokens.quoteTokens)
            allowedTokens.formUnion(specialTokens.braceOpenTokens)
            allowedTokens.formUnion(specialTokens.bracketOpenTokens)
            
            // Add soft bias for expected value type
            if let key = confirmedKey.withLock({ $0 }),
               let node = currentNode.withLock({ $0 }),
               let valueType = node.properties[key]?.kind {
                preferredTokens = getPreferredTokensForValueType(valueType)
            }
            
        case .inObject_afterValue:
            // Either comma for next key-value or close with }
            allowedTokens.formUnion(specialTokens.commaTokens)
            allowedTokens.formUnion(specialTokens.braceCloseTokens)
            
        case .inArray_expectValue:
            // Array values or close with ]
            allowedTokens.formUnion(specialTokens.bracketCloseTokens)
            allowedTokens.formUnion(specialTokens.quoteTokens)
            allowedTokens.formUnion(specialTokens.braceOpenTokens)
            allowedTokens.formUnion(specialTokens.bracketOpenTokens)
            
            // Add soft bias for array item type
            if let itemSchema = currentArrayItemSchema.withLock({ $0 }) {
                preferredTokens = getPreferredTokensForValueType(itemSchema.kind)
            }
            
        case .inArray_afterValue:
            // Either comma for next value or close with ]
            allowedTokens.formUnion(specialTokens.commaTokens)
            allowedTokens.formUnion(specialTokens.bracketCloseTokens)
            
        case .inString(kind: .value):
            // Value strings: no hard constraints, model handles content
            // Return with EOS safety bias only
            return applySafetyBias(logits)
            
        case .done:
            // Generation complete, boost EOS
            return applyEOSBoost(logits)
            
        case .error:
            // Error state, allow EOS to terminate
            lastError.withLock { $0 = .invalidPhase }
            return applyEOSBoost(logits)
        }
        
        // Apply hard mask for allowed tokens
        if allowedTokens.isEmpty {
            lastError.withLock { $0 = .emptyAllowedTokens }
            return applySafetyBias(logits)
        }
        
        var result = MLXUtils.applyLogitsMask(
            logits: logits,
            allowedTokens: allowedTokens,
            alwaysAllow: eosTokens()
        )
        
        // Apply soft bias for preferred tokens
        if !preferredTokens.isEmpty {
            result = MLXUtils.applySoftBias(
                logits: result,
                preferredTokens: preferredTokens,
                bias: valueBias
            )
        }
        
        return result
    }
    
    public func didSample(token: MLXArray) {
        let tokenId = Int32(token.item(Int.self))
        let text = adapter.decodeToken(tokenId)
        
        // Update key buffer if in key string
        if case .inString(kind: .key) = dpda.phase {
            if !specialTokens.quoteTokens.contains(tokenId) {
                let cleanText = text
                    .replacingOccurrences(of: "\"", with: "")
                    .replacingOccurrences(of: "\\", with: "")
                keyBuffer.withLock { $0 += cleanText }
                
                // Update trie path
                if let trie = getCurrentTrie() {
                    triePath.withLock { path in
                        let success = path.append(tokenId, in: trie)
                        if !success {
                            lastError.withLock { $0 = .trieMismatch(partial: keyBuffer.withLock { $0 }) }
                        }
                    }
                }
            }
        }
        
        // Advance DPDA with decoded text
        dpda.advance(with: text)
        
        // Handle phase transitions
        let newPhase = dpda.phase
        
        switch newPhase {
        case .inObject_expectColon:
            // Key ended, confirm or correct it
            let buffer = keyBuffer.withLock { $0 }
            
            if let trie = getCurrentTrie() {
                let path = triePath.withLock { $0 }
                if trie.canComplete(from: path), let keyName = path.getKeyName() {
                    confirmedKey.withLock { $0 = keyName }
                } else {
                    // Try to find closest valid key
                    let validKeys = trie.allKeys
                    confirmedKey.withLock { $0 = findClosestKey(buffer, in: validKeys) }
                }
            } else {
                confirmedKey.withLock { $0 = buffer }
            }
            
            // Reset for next key
            keyBuffer.withLock { $0 = "" }
            if let trie = getCurrentTrie() {
                triePath.withLock { $0 = TokenTrie.Path(root: trie.root) }
            }
            
        case .inObject_expectValueStart:
            // About to start a value
            break
            
        default:
            break
        }
        
        // Handle nested object/array entry
        if text.contains("{") {
            // Push context and enter nested object
            let key = confirmedKey.withLock { $0 }
            let node = currentNode.withLock { $0 }
            
            contextStack.withLock { stack in
                stack.append(.object(
                    node: node,
                    allowedKeys: Set(node?.objectKeys ?? [])
                ))
            }
            
            // Update current node to nested object
            if let key = key, let node = node {
                let childNode = node.properties[key]
                currentNode.withLock { $0 = childNode }
                currentArrayItemSchema.withLock { $0 = nil }  // Not in array
            } else {
                currentNode.withLock { $0 = nil }
                currentArrayItemSchema.withLock { $0 = nil }
            }
            
            // Reset trie path for new object
            if let trie = getCurrentTrie() {
                triePath.withLock { $0 = TokenTrie.Path(root: trie.root) }
            }
            
            confirmedKey.withLock { $0 = nil }
        }
        
        // Handle array entry
        if text.contains("[") {
            // Push context and enter array
            let key = confirmedKey.withLock { $0 }
            let node = currentNode.withLock { $0 }
            
            var itemSchema: SchemaNode? = nil
            if let key = key, let node = node {
                itemSchema = node.properties[key]?.items
            }
            
            contextStack.withLock { stack in
                stack.append(.array(itemSchema: itemSchema))
            }
            
            // Set array item schema
            currentArrayItemSchema.withLock { $0 = itemSchema }
            
            // Arrays don't have keys
            currentNode.withLock { $0 = nil }
            confirmedKey.withLock { $0 = nil }
        }
        
        // Handle object/array exit
        if text.contains("}") || text.contains("]") {
            // Pop context and restore parent
            contextStack.withLock { stack in
                if let frame = stack.popLast() {
                    switch frame {
                    case .object(let node, _):
                        currentNode.withLock { $0 = node }
                        currentArrayItemSchema.withLock { $0 = nil }
                    case .array(_):
                        // Restore from array context
                        currentArrayItemSchema.withLock { $0 = nil }
                        // Check if parent is object or array
                        if let parentFrame = stack.last {
                            switch parentFrame {
                            case .object(let node, _):
                                currentNode.withLock { $0 = node }
                            case .array(let schema):
                                currentArrayItemSchema.withLock { $0 = schema }
                            }
                        } else {
                            // Back to root
                            currentNode.withLock { $0 = schemaRoot }
                        }
                    }
                }
            }
            
            // Reset trie path for parent object
            if let trie = getCurrentTrie() {
                triePath.withLock { $0 = TokenTrie.Path(root: trie.root) }
            }
            
            confirmedKey.withLock { $0 = nil }
        }
    }
    
    // MARK: - Safety Biases
    
    private func applySafetyBias(_ logits: MLXArray) -> MLXArray {
        // Light EOS bias for safety
        guard let eos = adapter.eosTokenId() else { return logits }
        let vocabSize = logits.dim(logits.ndim - 1)
        guard eos >= 0 && eos < vocabSize else { return logits }
        
        let eosMask = MLXUtils.createVocabMask(vocabSize: vocabSize, tokens: [eos])
        var shape = Array(repeating: 1, count: logits.ndim)
        shape[logits.ndim - 1] = vocabSize
        
        return logits + eosMask.reshaped(shape) * eosBias
    }
    
    private func applyEOSBoost(_ logits: MLXArray) -> MLXArray {
        // Strong EOS bias for completion
        guard let eos = adapter.eosTokenId() else { return logits }
        let vocabSize = logits.dim(logits.ndim - 1)
        guard eos >= 0 && eos < vocabSize else { return logits }
        
        let eosMask = MLXUtils.createVocabMask(vocabSize: vocabSize, tokens: [eos])
        var shape = Array(repeating: 1, count: logits.ndim)
        shape[logits.ndim - 1] = vocabSize
        
        return logits + eosMask.reshaped(shape) * (eosBias * 2)
    }
}

// MARK: - ErrorCheckable Conformance

extension DPDAKeyTrieLogitProcessor: ErrorCheckable {
    public func hasError() -> Bool {
        lastError.withLock { $0 != nil }
    }
    
    public func hasFatalError() -> Bool {
        lastError.withLock { error in
            guard let error = error else { return false }
            switch error {
            case .trieMismatch, .emptyAllowedTokens:
                return true
            case .invalidPhase:
                return false
            }
        }
    }
    
    public func getLastError() -> JSONGenerationError? {
        lastError.withLock { error in
            guard let error = error else { return nil }
            switch error {
            case .trieMismatch(let partial):
                return .invalidTokenSelected(
                    token: -1,
                    partialKey: partial,
                    expectedTokens: []
                )
            case .emptyAllowedTokens:
                return .emptyConstraints
            case .invalidPhase:
                return .schemaViolation(reason: "Invalid DPDA phase")
            }
        }
    }
    
    public func clearError() {
        lastError.withLock { $0 = nil }
    }
}