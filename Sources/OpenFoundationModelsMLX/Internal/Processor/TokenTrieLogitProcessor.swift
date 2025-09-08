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
/// JSON生成時にスキーマに定義されたキーのみを物理的に許可する
/// 最適化版: 原子操作 + 最小ロック + GPU並列化
public final class TokenTrieLogitProcessor: LogitProcessor, Sendable {
    
    // MARK: - Immutable Data (ロック不要)
    private let tokenTrie: TokenTrie
    private let tokenizer: any Tokenizer
    private let tokenizerAdapter: MLXLLMTokenizer
    private let specialTokens: MLXLLMTokenizer.SpecialTokens
    private let maskHintGenerator: JSONMaskHintGenerator
    
    // MARK: - 読み取り専用スナップショット（高速コピー）
    private struct ProcessorSnapshot: Sendable {
        let jsonPhase: JSONStateMachine.Phase
        let tokenPath: TokenTrie.Path
        let isGenerating: Bool
        let tokenCount: Int
        
        init(jsonPhase: JSONStateMachine.Phase = .root, 
             tokenPath: TokenTrie.Path, 
             isGenerating: Bool = false, 
             tokenCount: Int = 0) {
            self.jsonPhase = jsonPhase
            self.tokenPath = tokenPath
            self.isGenerating = isGenerating
            self.tokenCount = tokenCount
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
    
    public init(schema: SchemaMeta, tokenizer: any Tokenizer) {
        // Immutableデータの初期化
        let adapter = MLXLLMTokenizer(tokenizer: tokenizer)
        self.tokenTrie = TokenTrieBuilder.buildCached(schema: schema, tokenizer: adapter)
        self.tokenizer = tokenizer
        self.tokenizerAdapter = adapter
        self.specialTokens = adapter.findSpecialTokens()
        self.maskHintGenerator = JSONMaskHintGenerator.forSchemaConstrainedDecoding(
            specialTokens: specialTokens,
            includeWhitespace: false
        )
        
        // 軽量状態の初期化
        let initialPath = TokenTrie.Path(root: tokenTrie.root)
        let initialSnapshot = ProcessorSnapshot(tokenPath: initialPath)
        self.lightweightState = Mutex(initialSnapshot)
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
        
        // 軽量状態の原子更新（ロックフリー）
        let newPath = TokenTrie.Path(root: tokenTrie.root)
        let newSnapshot = ProcessorSnapshot(
            jsonPhase: .root,
            tokenPath: newPath,
            isGenerating: false,
            tokenCount: 0
        )
        lightweightState.withLock { $0 = newSnapshot }
        errorState.withLock { $0 = nil }
    }
    
    // MARK: - 最適化されたメイン処理
    
    private func processOptimized(logits: MLXArray) throws -> MLXArray {
        // 1. 状態スナップショット取得（高速読み取り）
        let currentSnapshot = lightweightState.withLock { $0 }
        
        // 2. 生成状態更新（高速書き込み）
        let updatedSnapshot = ProcessorSnapshot(
            jsonPhase: currentSnapshot.jsonPhase,
            tokenPath: currentSnapshot.tokenPath,
            isGenerating: true,
            tokenCount: currentSnapshot.tokenCount
        )
        lightweightState.withLock { $0 = updatedSnapshot }
        
        // 3. キー状態判定（ロック不要）
        let isInKey = isInKeyState(phase: currentSnapshot.jsonPhase)
        guard isInKey else {
            return logits  // 制約なし、即座にリターン
        }
        
        // 4. 許可トークン計算（ロック不要、読み取り専用）
        let allowedTokens = tokenTrie.getAllowedTokens(for: currentSnapshot.tokenPath)
        
        // 5. 制約検証（ロック不要）
        if allowedTokens.isEmpty && !currentSnapshot.tokenPath.isAtTerminal() {
            throw JSONGenerationError.noValidTokens(
                partialKey: tokenizerAdapter.decode(currentSnapshot.tokenPath.tokens),
                position: currentSnapshot.tokenPath.tokens.count
            )
        }
        
        // 6. 有効トークン準備（ロック不要）
        var validTokens = allowedTokens
        if currentSnapshot.tokenPath.isAtTerminal() {
            if let quoteToken = specialTokens.quoteTokens.first {
                validTokens.insert(quoteToken)
            }
        }
        
        // 7. GPU処理（完全にロック外、最大並列性）
        return try applyHardMaskOptimized(to: logits, allowedTokens: validTokens)
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
            let (newPhase, _) = heavyState.withLock { state in
                state.generatedTokens.append(tokenID)
                
                let text = tokenizerAdapter.decodeToken(tokenID)
                for char in text {
                    state.jsonStateMachine.processCharacter(char)
                }
                
                return (state.jsonStateMachine.phase, text)
            }
            
            // 軽量状態の取得
            let currentSnapshot = lightweightState.withLock { $0 }
            var newTokenPath = currentSnapshot.tokenPath
            
            let isInKey = isInKeyState(phase: currentSnapshot.jsonPhase)
            if isInKey {
                let success = newTokenPath.append(tokenID, in: tokenTrie)
                if !success {
                    throw JSONGenerationError.invalidTokenSelected(
                        token: tokenID,
                        partialKey: tokenizerAdapter.decode(currentSnapshot.tokenPath.tokens),
                        expectedTokens: tokenTrie.getAllowedTokens(for: currentSnapshot.tokenPath)
                    )
                }
            } else {
                // キー状態外では、パスをリセット
                if newTokenPath.tokens.count > 0 {
                    newTokenPath.reset(to: tokenTrie.root)
                }
            }
            
            // 新しいスナップショット（高速更新）
            let newSnapshot = ProcessorSnapshot(
                jsonPhase: newPhase,
                tokenPath: newTokenPath,
                isGenerating: currentSnapshot.isGenerating,
                tokenCount: currentSnapshot.tokenCount + 1
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
        guard !allowedTokens.isEmpty else {
            throw JSONGenerationError.emptyConstraints
        }
        
        let actualVocabSize = logits.dim(logits.ndim - 1)
        
        // EOS許可判定（読み取り専用）
        let currentSnapshot = lightweightState.withLock { $0 }
        var allow = allowedTokens
        if !isInKeyState(phase: currentSnapshot.jsonPhase), 
           let eos = tokenizerAdapter.eosTokenId() {
            allow.insert(eos)
        }
        
        // GPU処理（並列実行）
        let allowedIndices = Array(allow.filter { $0 >= 0 && $0 < actualVocabSize })
        
        var mask = [Float](repeating: 0, count: actualVocabSize)
        for idx in allowedIndices {
            mask[Int(idx)] = 1
        }
        let maskArray = MLXArray(mask)
        
        var shape = Array(repeating: 1, count: logits.ndim)
        shape[logits.ndim - 1] = actualVocabSize
        let reshapedMask = maskArray.reshaped(shape)
        
        let negInf = MLX.full(logits.shape, values: -Float.infinity)
        return MLX.where(reshapedMask .> 0, logits, negInf)
    }
    
    // Removed: applySoftBias - only hard constraints now
    
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
            let text = tokenizer.decode(tokens: state.generatedTokens.map { Int($0) }, skipSpecialTokens: false)
            
            guard let data = text.data(using: String.Encoding.utf8),
                  let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else {
                return false
            }
            
            let jsonKeys = Set(json.keys)
            let schemaKeys = Set(tokenTrie.allKeys)
            return jsonKeys.isSubset(of: schemaKeys)
        }
    }
}
