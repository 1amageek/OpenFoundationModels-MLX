import Foundation
import MLX
import MLXLMCommon
import MLXLLM
@preconcurrency import Tokenizers

/// TokenTrieベースの制約をLogitProcessorとして実装
/// JSON生成時にスキーマに定義されたキーのみを物理的に許可する
public struct TokenTrieLogitProcessor: LogitProcessor {
    private let tokenTrie: TokenTrie
    private let tokenizer: any Tokenizer
    private let tokenizerAdapter: MLXLLMTokenizer
    private let specialTokens: MLXLLMTokenizer.SpecialTokens
    
    // 可変状態
    private var jsonState: EnhancedJSONStateMachine
    private var tokenPath: TokenTrie.Path
    private var promptTokens: [Int32] = []
    private var generatedTokens: [Int32] = []
    private var violationCount: Int = 0
    
    public init(
        schema: SchemaMeta,
        tokenizer: any Tokenizer
    ) {
        // TokenTrieを構築
        let adapter = MLXLLMTokenizer(tokenizer: tokenizer)
        self.tokenTrie = TokenTrieBuilder.build(
            from: schema,
            tokenizer: adapter
        )
        
        self.tokenizer = tokenizer
        self.tokenizerAdapter = adapter
        self.specialTokens = adapter.findSpecialTokens()
        self.jsonState = EnhancedJSONStateMachine()
        self.tokenPath = TokenTrie.Path(root: tokenTrie.root)
    }
    
    // MARK: - LogitProcessor Protocol
    
    public mutating func prompt(_ prompt: MLXArray) {
        // プロンプトトークンを保存
        let count = prompt.dim(0)
        promptTokens = (0..<count).map { i in
            Int32(prompt[i].item(Int.self))
        }
        
        // 状態をリセット
        jsonState.reset()
        tokenPath.reset(to: tokenTrie.root)
        generatedTokens.removeAll()
        violationCount = 0
    }
    
    public func process(logits: MLXArray) -> MLXArray {
        // 現在のJSON状態に基づいて制約を適用
        switch jsonState.phase {
        case .inKey:
            return processInKeyState(logits)
            
        case .expectColon:
            return processExpectColonState(logits)
            
        case .inValue, .expectCommaOrClose:
            // 値の部分では制約なし（プリミティブ性維持）
            return logits
            
        default:
            return logits
        }
    }
    
    public mutating func didSample(token: MLXArray) {
        let tokenID = Int32(token.item(Int.self))
        generatedTokens.append(tokenID)
        
        // JSON状態を更新 - 事前作成したアダプタを再利用
        jsonState.processTokenWithSpecialHandling(
            tokenID,
            tokenizer: tokenizerAdapter,
            specialTokens: specialTokens
        )
        
        // inKey状態の場合、TokenTrieパスを更新
        if jsonState.isInKeyEmission() {
            let success = tokenPath.append(tokenID, in: tokenTrie)
            if !success {
                // 無効なトークン - 違反カウントを増やす
                violationCount += 1
                if violationCount >= 2 {
                    // 連続違反でパスをリセット
                    tokenPath.reset(to: tokenTrie.root)
                    violationCount = 0
                }
            } else {
                violationCount = 0
            }
        } else if !jsonState.isInKeyEmission() {
            // キー出力が終了したらパスをリセット
            tokenPath.reset(to: tokenTrie.root)
            violationCount = 0
        }
    }
    
    // MARK: - Private Methods
    
    private func processInKeyState(_ logits: MLXArray) -> MLXArray {
        // TokenTrieから許可されたトークンを取得
        let allowedTokens = tokenTrie.getAllowedTokens(for: tokenPath)
        
        if allowedTokens.isEmpty {
            // 候補なし - すべてのトークンを-infに設定してリトライを誘発
            Logger.debug("[TokenTrie] No valid candidates at current path, triggering retry")
            return MLX.full(logits.shape, values: -Float.infinity)
        }
        
        // terminalノードの場合、引用符トークンも許可
        var finalAllowed = allowedTokens
        if tokenPath.isAtTerminal() {
            finalAllowed.formUnion(specialTokens.quoteTokens)
        }
        
        // GPU最適化されたマスクを適用
        return applyMask(to: logits, allowedTokens: finalAllowed)
    }
    
    private func processExpectColonState(_ logits: MLXArray) -> MLXArray {
        // コロントークンのみ許可
        if specialTokens.colonTokens.isEmpty {
            // コロントークンが見つからない場合は制約なし
            return logits
        }
        return applyMask(to: logits, allowedTokens: specialTokens.colonTokens)
    }
    
    private func applyMask(to logits: MLXArray, allowedTokens: Set<Int32>) -> MLXArray {
        // MLXのGPU最適化を活用したマスキング
        let vocabSize = logits.dim(logits.ndim - 1)
        
        // マスク配列を作成（0 = 禁止, 1 = 許可）
        var mask = [Float](repeating: 0, count: vocabSize)
        for tokenID in allowedTokens {
            if tokenID >= 0 && tokenID < vocabSize {
                mask[Int(tokenID)] = 1
            }
        }
        
        // MLXArrayに変換 - [V]の形状のまま使用
        let maskArray = MLXArray(mask)
        
        // マスクを適用: MLXのブロードキャストにより最後の次元で自動的に拡張される
        // 許可されたトークンは元の値、禁止は-inf
        let negInf = MLX.full(logits.shape, values: -Float.infinity)
        return MLX.where(maskArray .> 0, logits, negInf)
    }
    
    // MARK: - Debugging
    
    public func debugState() -> String {
        return """
        TokenTrieLogitProcessor State:
        - JSON Phase: \(jsonState.phase)
        - Token Path Length: \(tokenPath.tokens.count)
        - Is at Terminal: \(tokenPath.isAtTerminal())
        - Generated Tokens: \(generatedTokens.count)
        - Violation Count: \(violationCount)
        """
    }
}

// MARK: - Extensions

extension TokenTrieLogitProcessor {
    /// 簡易的な初期化（デフォルト設定）
    public init(keys: [String], tokenizer: any Tokenizer) {
        let schema = SchemaMeta(keys: keys, required: [])
        self.init(schema: schema, tokenizer: tokenizer)
    }
    
    /// スキーマ検証
    public func validateGenerated() -> Bool {
        // 生成されたテキストをデコード
        let text = tokenizer.decode(tokens: generatedTokens.map { Int($0) }, skipSpecialTokens: false)
        
        // JSONとしてパース可能か確認
        guard let data = text.data(using: String.Encoding.utf8),
              let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            return false
        }
        
        // スキーマのキーが含まれているか確認
        let jsonKeys = Set(json.keys)
        let schemaKeys = Set(tokenTrie.allKeys)
        
        // すべてのキーがスキーマに含まれているか
        return jsonKeys.isSubset(of: schemaKeys)
    }
}