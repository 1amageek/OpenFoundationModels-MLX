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

// Note: We can't make LogitProcessor methods throw, so we track errors internally

/// TokenTrieベースの制約をLogitProcessorとして実装
/// JSON生成時にスキーマに定義されたキーを原則ハードマスクし、
/// 値や一部フェーズはソフトヒント（バイアス）で誘導する
/// 最適化版: 原子操作 + 最小ロック + GPU並列化
public final class TokenTrieLogitProcessor: LogitProcessor, Sendable {
    
    // MARK: - Immutable Data (ロック不要)
    private let schemaRoot: SchemaNode?
    private let schemaIndex: SchemaTrieIndex?
    private let tokenizer: any Tokenizer
    private let tokenizerAdapter: MLXLLMTokenizer
    private let specialTokens: MLXLLMTokenizer.SpecialTokens
    private let maskHintGenerator: JSONMaskHintGenerator
    
    // For backward compatibility
    private let legacyTrie: TokenTrie?
    
    // MARK: - Context Stack for Nested Objects
    private enum Context: Sendable {
        case object(SchemaNode)
        case array(SchemaNode?)
    }
    
    // MARK: - 読み取り専用スナップショット（高速コピー）
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
    
    // MARK: - 最適化された状態管理
    // 軽量状態: Mutexで保護（読み取り専用操作を高速化）
    private let lightweightState: Mutex<ProcessorSnapshot>
    private let errorState = Mutex<JSONGenerationError?>(nil)
    
    // MARK: - 重い状態（最小限のロック使用）
    private struct HeavyState: Sendable {
        var jsonStateMachine: JSONStateMachine = JSONStateMachine()
        var generatedTokens: [Int32] = []
        var promptTokens: [Int32] = []
    }
    
    private let heavyState = Mutex<HeavyState>(HeavyState())
    
    // New init with SchemaNode for nested object support
    public init(schema: SchemaNode?, tokenizer: any Tokenizer) {
        // Immutableデータの初期化
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
            self.legacyTrie = nil
        } else {
            self.schemaIndex = nil
            self.legacyTrie = nil
        }
        
        // 軽量状態の初期化
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
        
        if let schema = schema {
            print("🚀 [TokenTrieLogitProcessor] Initialized with nested schema")
            print("📋 [TokenTrieLogitProcessor] Root keys: \(schema.objectKeys)")
        }
    }
    
    // Legacy init for backward compatibility
    public init(schema: SchemaMeta, tokenizer: any Tokenizer) {
        // Convert SchemaMeta to simple SchemaNode
        let node = SchemaNode(
            kind: .object,
            properties: Dictionary(uniqueKeysWithValues: schema.keys.map { ($0, SchemaNode.any) }),
            required: Set(schema.required)
        )
        
        // Also keep legacy trie for fallback
        let adapter = MLXLLMTokenizer(tokenizer: tokenizer)
        let legacyTrie = TokenTrieBuilder.buildCached(schema: schema, tokenizer: adapter)
        
        self.schemaRoot = node
        self.tokenizer = tokenizer
        self.tokenizerAdapter = adapter
        self.specialTokens = adapter.findSpecialTokens()
        self.maskHintGenerator = JSONMaskHintGenerator.forSchemaConstrainedDecoding(
            specialTokens: specialTokens,
            includeWhitespace: false
        )
        self.schemaIndex = SchemaTrieIndex(root: node, tokenizer: adapter)
        self.legacyTrie = legacyTrie
        
        let initialPath = TokenTrie.Path()
        let initialSnapshot = ProcessorSnapshot(
            tokenPath: initialPath,
            contextStack: [.object(node)],
            currentObjectNode: node
        )
        self.lightweightState = Mutex(initialSnapshot)
        
        print("🚀 [TokenTrieLogitProcessor] Initialized with legacy SchemaMeta")
        print("📋 [TokenTrieLogitProcessor] Schema keys: \(schema.keys)")
    }
    
    // MARK: - LogitProcessor Protocol
    
    public func prompt(_ prompt: MLXArray) {
        // 重い状態の更新（最小ロック）
        heavyState.withLock { state in
            let flat = prompt.reshaped([-1])
            let count = flat.dim(0)
            state.promptTokens = (0..<count).map { i in
                Int32(flat[i].item(Int.self))
            }
            state.jsonStateMachine.reset()
            state.generatedTokens.removeAll()
        }
        
        // 軽量状態の原子更新 - reset to root context
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
    
    // MARK: - Context Management Helpers
    
    private func getCurrentTrie() -> TokenTrie? {
        let snap = lightweightState.withLock { $0 }
        
        // Use legacy trie if available (backward compatibility)
        if let legacyTrie = legacyTrie {
            return legacyTrie
        }
        
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
                print("📦 [Context] Entering nested object for key '\(key)', new keys: \(childNode.objectKeys)")
            } else {
                // Unknown object structure
                let emptyNode = SchemaNode(kind: .object)
                newStack.append(.object(emptyNode))
                newObjectNode = emptyNode
                print("⚠️ [Context] Entering unknown object")
            }
            newPendingKey = nil
            
        case "[":
            // Entering array
            if let key = key,
               let parentNode = currentSnap.currentObjectNode,
               let childNode = parentNode.properties[key],
               childNode.kind == .array {
                newStack.append(.array(childNode.items))
                print("📦 [Context] Entering array for key '\(key)'")
            } else {
                newStack.append(.array(nil))
                print("⚠️ [Context] Entering unknown array")
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
                    if case .object(let node) = ctx {
                        newObjectNode = node
                        break
                    }
                }
                print("📦 [Context] Exited object, returning to parent")
                break
                
            case (.array, .array):
                print("📦 [Context] Exited array")
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
    
    // MARK: - 最適化されたメイン処理
    
    private func processOptimized(logits: MLXArray) throws -> MLXArray {
        // 1) スナップショット & JSON 状態取得
        let snap = lightweightState.withLock { $0 }
        let jsonState = heavyState.withLock { $0.jsonStateMachine }
        let isInKey = isInKeyState(phase: snap.jsonPhase)

        // 2) 生成状態フラグだけ更新
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
        
        // Get current trie based on context
        let currentTrie = getCurrentTrie()

        // 3) 構文ヒントを取得（現在のTrie/Path を渡す）
        let hint: JSONMaskHint? = currentTrie.flatMap { trie in
            maskHintGenerator.maskHint(
                for: jsonState,
                tokenTrie: trie,
                tokenPath: snap.tokenPath
            )
        }
        
        print("📊 [TokenTrieLogitProcessor] Phase: \(snap.jsonPhase), IsInKey: \(isInKey)")
        print("📊 [TokenTrieLogitProcessor] Token path: \(snap.tokenPath.tokens)")

        // 4) 許可集合の構築
        var allowed = Set<Int32>()
        var useHardMask = false

        if isInKey {
            // キー中: 現在のコンテキストのTrieの許可集合
            if let trie = currentTrie {
                allowed = trie.getAllowedTokens(for: snap.tokenPath)
                print("🔑 [TokenTrieLogitProcessor] In key state with context '\(snap.currentObjectNode?.objectKeys ?? [])', allowed tokens: \(allowed.count)")
                
                // 末端ならクォート（単体がなければ動的候補）とエスケープを許可
                if snap.tokenPath.isAtTerminal() {
                    let quoteCandidates = dynamicQuoteCandidates(from: logits, fallback: specialTokens.quoteTokens)
                    print("📝 [TokenTrieLogitProcessor] Terminal: adding \(quoteCandidates.count) quote candidates")
                    allowed.formUnion(quoteCandidates)
                    allowed.formUnion(specialTokens.backslashTokens)
                }
            } else {
                // No trie available - allow any tokens
                print("⚠️ [TokenTrieLogitProcessor] No trie for current context, allowing all tokens")
                allowed = Set(0..<Int32(logits.dim(logits.ndim - 1)))
            }

            // キー中はintersectionを行わない（動的クォート候補を潰さないため）
            // 構文側のhintは無視し、Trieの制約のみを信頼する
            useHardMask = true
            print("🔒 [TokenTrieLogitProcessor] Final allowed in key: \(allowed.count) tokens")

            // 途中で継続不能なら即エラー
            if allowed.isEmpty && !snap.tokenPath.isAtTerminal() {
                throw JSONGenerationError.noValidTokens(
                    partialKey: tokenizerAdapter.decode(snap.tokenPath.tokens),
                    position: snap.tokenPath.tokens.count
                )
            }
        } else {
            // キー外: 構文ヒントに従う（hard は物理マスク、soft は後段でバイアス）
            print("🔍 [TokenTrieLogitProcessor] Not in key state")
            if let h = hint {
                print("💡 [TokenTrieLogitProcessor] Hint mode: \(h.mode), allow: \(h.allow.count), prefer: \(h.prefer.count)")
                switch h.mode {
                case .hard:
                    allowed = h.allow
                    useHardMask = true
                    print("🔒 [TokenTrieLogitProcessor] Using hard mask with \(allowed.count) allowed tokens")
                case .soft:
                    // allowed は空のまま（素通し）→ prefer を後でバイアス
                    print("🔓 [TokenTrieLogitProcessor] Using soft bias")
                    break
                }
            } else {
                print("❓ [TokenTrieLogitProcessor] No hint available")
            }
        }

        // 5) マスク適用
        // hard指定時はallowedが空でもapplyする（.doneでEOSを追加するため）
        if useHardMask {
            return try applyHardMaskOptimized(to: logits, allowedTokens: allowed)
        }
        if let h = hint {
            if h.mode == .soft, !h.prefer.isEmpty {
                // ソフトヒントはバイアスのみ
                return MLXUtils.applySoftBias(logits: logits, preferredTokens: h.prefer, bias: 2.5)
            }
        }
        if isInKey && !allowed.isEmpty {
            // キー中は常にハードマスク
            return try applyHardMaskOptimized(to: logits, allowedTokens: allowed)
        }

        // 数値/リテラルなど制約不能な場面は素通し
        return logits
    }
    
    public func process(logits: MLXArray) -> MLXArray {
        do {
            // エラー状態をクリア（高速）
            errorState.withLock { $0 = nil }
            
            return try processOptimized(logits: logits)
        } catch let error as JSONGenerationError {
            // エラー状態を設定（高速）
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
            return logits * 0.7
        }
        
        let vocabSize = logits.dim(logits.ndim - 1)
        guard eosToken >= 0 && eosToken < vocabSize else {
            return logits * 0.7
        }
        
        let eosMask = MLXUtils.createVocabMask(vocabSize: vocabSize, tokens: [eosToken])
        var shape = Array(repeating: 1, count: logits.ndim)
        shape[logits.ndim - 1] = vocabSize
        
        let boost = eosMask.reshaped(shape) * 5.0
        let boostedLogits = logits + boost
        return boostedLogits * 0.8
    }
    
    public func didSample(token: MLXArray) {
        let tokenID = Int32(token.item(Int.self))
        
        do {
            // 重い状態更新（最小ロック）
            let (newPhase, generatedText) = heavyState.withLock { state in
                state.generatedTokens.append(tokenID)
                
                let text = tokenizerAdapter.decodeToken(tokenID)
                for char in text {
                    state.jsonStateMachine.processCharacter(char)
                }
                
                return (state.jsonStateMachine.phase, text)
            }
            
            // 直前スナップショットと新しいフェーズの両方を見る
            let prevSnapshot = lightweightState.withLock { $0 }
            var newTokenPath = prevSnapshot.tokenPath
            var newContextStack = prevSnapshot.contextStack
            var newPendingKey = prevSnapshot.pendingKey
            var newObjectNode = prevSnapshot.currentObjectNode
            
            let wasInKey = isInKeyState(phase: prevSnapshot.jsonPhase)
            let nowInKey = isInKeyState(phase: newPhase)
            
            print("🔄 [didSample] Phase transition: wasInKey=\(wasInKey), nowInKey=\(nowInKey), text='\(generatedText)'")
            
            // Get current trie for key tracking
            let currentTrie = getCurrentTrie()
            
            if wasInKey && nowInKey {
                // まだキー本文中 → Trie を前進
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
                // いまキーが閉じた → キー名を記録して次のキーに備える
                if prevSnapshot.tokenPath.isAtTerminal() {
                    newPendingKey = prevSnapshot.tokenPath.getKeyName()
                    print("🔑 [didSample] Key closed: '\(newPendingKey ?? "unknown")'、resetting path")
                }
                if let trie = currentTrie {
                    newTokenPath.reset(to: trie.root)
                } else {
                    newTokenPath.reset()
                }
            }
            
            // Handle context transitions based on JSON structure
            // Check for object/array entry
            if generatedText.contains("{") {
                if let key = newPendingKey,
                   let parentNode = newObjectNode,
                   let childNode = parentNode.properties[key],
                   childNode.kind == .object {
                    // Entering nested object with known schema
                    newContextStack.append(.object(childNode))
                    newObjectNode = childNode
                    print("📦 [didSample] Entering object '\(key)' with keys: \(childNode.objectKeys)")
                } else {
                    // Entering unknown object
                    let emptyNode = SchemaNode(kind: .object)
                    newContextStack.append(.object(emptyNode))
                    newObjectNode = emptyNode
                    print("⚠️ [didSample] Entering unknown object")
                }
                newPendingKey = nil
                // Reset path for new object context
                if let trie = schemaIndex?.trie(for: newObjectNode ?? SchemaNode(kind: .object)) {
                    newTokenPath.reset(to: trie.root)
                }
            }
            
            if generatedText.contains("[") {
                if let key = newPendingKey,
                   let parentNode = newObjectNode,
                   let childNode = parentNode.properties[key],
                   childNode.kind == .array {
                    newContextStack.append(.array(childNode.items))
                    print("📦 [didSample] Entering array '\(key)'")
                } else {
                    newContextStack.append(.array(nil))
                    print("⚠️ [didSample] Entering unknown array")
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
                // Update current object node
                newObjectNode = nil
                for ctx in newContextStack.reversed() {
                    if case .object(let node) = ctx {
                        newObjectNode = node
                        break
                    }
                }
                // Reset path for parent context
                if let node = newObjectNode,
                   let trie = schemaIndex?.trie(for: node) {
                    newTokenPath.reset(to: trie.root)
                }
                print("📦 [didSample] Exited object, returning to parent")
            }
            
            if generatedText.contains("]") {
                // Pop array context
                while let last = newContextStack.last {
                    newContextStack.removeLast()
                    if case .array = last { break }
                }
                print("📦 [didSample] Exited array")
            }
            
            // Clear pending key on value start (non-object/array)
            if case .inObject(.expectValueStart) = newPhase {
                let firstNonWS = generatedText.first { !$0.isWhitespace }
                if let char = firstNonWS,
                   char != "{" && char != "[" {
                    // Starting a primitive value
                    newPendingKey = nil
                }
            }
            
            // 新しいスナップショット（高速更新）
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
    
    // MARK: - エラー状態管理（原子操作）
    
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
    
    // MARK: - Private Methods
    
    // MARK: - 最適化されたヘルパーメソッド
    
    private func isInKeyState(phase: JSONStateMachine.Phase) -> Bool {
        if case .inString(let strPhase) = phase,
           case .body(let kind, _) = strPhase,
           kind == .key {
            return true
        }
        return false
    }
    
    /// GPU最適化されたマスク適用（完全にロック外）
    private func applyHardMaskOptimized(to logits: MLXArray, allowedTokens: Set<Int32>) throws -> MLXArray {
        let vocab = logits.dim(logits.ndim - 1)
        let snap = lightweightState.withLock { $0 }

        // EOS は「完了後」だけ許可（途中では許可しない）
        var allow = allowedTokens
        if case .done = snap.jsonPhase, let eos = tokenizerAdapter.eosTokenId() {
            allow.insert(eos)
        }

        // 許可が空でも .done + EOS 無しなどの特殊状況があり得るので投げない
        let indices = Array(allow.filter { $0 >= 0 && $0 < vocab })

        if indices.isEmpty {
            // 物理マスク不能：そのまま返すよりも少し鈍らせる（尻尾抑止）
            return logits * 0.9
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
    
    // Removed: applySoftBias - only hard constraints now
    // ↑ 上の processOptimized で soft ヒント時のみ Util のバイアスを使用

    // MARK: - Quote 終端の動的許可（単体 `"` が無いトークナイザ対策）
    private func dynamicQuoteCandidates(from logits: MLXArray, topK: Int = 256, fallback: Set<Int32>) -> Set<Int32> {
        // 単体 quote が既に検出できていればそれを使う
        if !fallback.isEmpty { return fallback }
        var out = Set<Int32>()
        for i in topKIndices(logits, k: topK) {
            let tid = Int32(i)
            let piece = tokenizerAdapter.decodeToken(tid)
            if piece.contains("\"") { out.insert(tid) }
        }
        // エスケープは常に許可（\" など）
        out.formUnion(specialTokens.backslashTokens)
        return out
    }

    private func topKIndices(_ logits: MLXArray, k: Int) -> [Int] {
        // 末端時のみ呼ばれる想定なので単純 CPU 実装で十分
        let flat = logits.reshaped([-1])
        let n = flat.dim(0)
        var arr: [(Float, Int)] = []
        arr.reserveCapacity(n)
        for i in 0..<n {
            arr.append((flat[i].item(Float.self), i))
        }
        arr.sort { $0.0 > $1.0 }
        return arr.prefix(k).map { $0.1 }
    }
    
    // MARK: - Debugging
    
    // MARK: - デバッグ情報
    
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

// MARK: - Extensions

extension TokenTrieLogitProcessor {
    /// Convenience initializer with default settings
    public convenience init(keys: [String], tokenizer: any Tokenizer) {
        let schema = SchemaMeta(keys: keys, required: [])
        self.init(schema: schema, tokenizer: tokenizer)
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
