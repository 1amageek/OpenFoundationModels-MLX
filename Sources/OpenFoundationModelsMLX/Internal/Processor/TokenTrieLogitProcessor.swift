import Foundation
import PRECISE
@preconcurrency import MLX
import MLXLMCommon
import MLXLLM
@preconcurrency import Tokenizers

/// TokenTrieベースの制約をLogitProcessorとして実装
/// JSON生成時にスキーマに定義されたキーのみを物理的に許可する
public final class TokenTrieLogitProcessor: LogitProcessor, @unchecked Sendable {
    private let tokenTrie: TokenTrie
    private let tokenizer: any Tokenizer
    private let tokenizerAdapter: MLXLLMTokenizer
    private let specialTokens: MLXLLMTokenizer.SpecialTokens
    private let maskHintGenerator: JSONMaskHintGenerator
    
    // 可変状態
    private var jsonState: JSONStateMachine
    private var tokenPath: TokenTrie.Path
    private var promptTokens: [Int32] = []
    private var generatedTokens: [Int32] = []
    private var violationCount: Int = 0
    
    
    // Soft constraint micro-bias value (small positive bias for schema-preferred tokens)
    private let microBias: Float
    
    public init(
        schema: SchemaMeta,
        tokenizer: any Tokenizer,
        microBias: Float = 0.2
    ) {
        // TokenTrieを構築（キャッシュ版を使用）
        let adapter = MLXLLMTokenizer(tokenizer: tokenizer)
        self.tokenTrie = TokenTrieBuilder.buildCached(
            schema: schema,
            tokenizer: adapter
        )
        
        self.tokenizer = tokenizer
        self.tokenizerAdapter = adapter
        self.specialTokens = adapter.findSpecialTokens()
        self.jsonState = JSONStateMachine()
        self.tokenPath = TokenTrie.Path(root: tokenTrie.root)
        self.maskHintGenerator = JSONMaskHintGenerator.forSchemaConstrainedDecoding(
            specialTokens: specialTokens,
            includeWhitespace: false
        )
        self.microBias = microBias
    }
    
    // MARK: - LogitProcessor Protocol
    
    public func prompt(_ prompt: MLXArray) {
        // フラット化して安全にトークンIDを取得
        let flat = prompt.reshaped([-1])
        let count = flat.dim(0)
        promptTokens = (0..<count).map { i in
            Int32(flat[i].item(Int.self))
        }
        
        // 状態をリセット
        jsonState.reset()
        tokenPath.reset(to: tokenTrie.root)
        generatedTokens.removeAll()
        violationCount = 0
    }
    
    public func process(logits: MLXArray) -> MLXArray {
        // Generate mask hint based on current JSON state
        guard let maskHint = maskHintGenerator.maskHint(
            for: jsonState,
            tokenTrie: tokenTrie,
            tokenPath: tokenPath
        ) else {
            // No constraints for this state
            return logits
        }
        
        // Apply mask based on hint mode
        switch maskHint.mode {
        case .hard:
            // Hard constraints - only allowed tokens
            return applyMask(to: logits, allowedTokens: maskHint.allow)
        case .soft:
            // Soft constraints - apply gentle micro-bias to preferred tokens
            // This provides guidance while maintaining generation flexibility
            return applySoftBias(to: logits, preferredTokens: maskHint.allow)
        }
    }
    
    public func didSample(token: MLXArray) {
        let tokenID = Int32(token.item(Int.self))
        generatedTokens.append(tokenID)
        
        // Decode token to character for state machine
        let tokenText = tokenizerAdapter.decodeToken(tokenID)
        
        // Update JSON state machine with each character
        for char in tokenText {
            jsonState.processCharacter(char)
        }
        
        // Update TokenTrie path when in key string state
        if case .inString(let strPhase) = jsonState.phase,
           case .body(let kind, _) = strPhase,
           kind == .key {
            // We're emitting a key - update path
            let success = tokenPath.append(tokenID, in: tokenTrie)
            if !success {
                // Invalid token - increase violation count
                violationCount += 1
                if violationCount >= 2 {
                    // Reset path after consecutive violations
                    tokenPath.reset(to: tokenTrie.root)
                    violationCount = 0
                }
            } else {
                violationCount = 0
            }
        } else {
            // Not in key emission - reset path if needed
            if tokenPath.tokens.count > 0 {
                tokenPath.reset(to: tokenTrie.root)
                violationCount = 0
            }
        }
    }
    
    // MARK: - Private Methods
    
    private func applyMask(to logits: MLXArray, allowedTokens: Set<Int32>) -> MLXArray {
        // 最後の次元を語彙サイズとして扱う（形状非依存）
        let actualVocabSize = logits.dim(logits.ndim - 1)
        
        // Always allow EOS token to prevent infinite generation
        var allow = allowedTokens
        if let eos = tokenizerAdapter.eosTokenId() {
            allow.insert(eos)
        }
        
        // Create mask using MLX-compatible operations
        let allowedIndices = Array(allow.filter { $0 >= 0 && $0 < actualVocabSize })
        
        // Create mask array - MLX will handle GPU optimization
        let maskArray: MLXArray
        if !allowedIndices.isEmpty {
            var mask = [Float](repeating: 0, count: actualVocabSize)
            for idx in allowedIndices {
                mask[Int(idx)] = 1
            }
            maskArray = MLXArray(mask)
        } else {
            maskArray = MLX.zeros([actualVocabSize])
        }
        
        // Reshape mask for broadcasting
        var shape = Array(repeating: 1, count: logits.ndim)
        shape[logits.ndim - 1] = actualVocabSize
        let reshapedMask = maskArray.reshaped(shape)
        
        // Apply mask using efficient GPU operations
        // Create -inf array for masked positions
        let negInf = MLX.full(logits.shape, values: -Float.infinity)
        return MLX.where(reshapedMask .> 0, logits, negInf)
    }
    
    /// Apply soft micro-bias to gently encourage preferred tokens without hard constraints
    /// This provides schema guidance while maintaining generation flexibility for numbers, literals, etc.
    private func applySoftBias(to logits: MLXArray, preferredTokens: Set<Int32>) -> MLXArray {
        guard !preferredTokens.isEmpty else {
            return logits
        }
        
        // 最後の次元を語彙サイズとして扱う（形状非依存）
        let actualVocabSize = logits.dim(logits.ndim - 1)
        
        // Create bias using MLX-compatible operations
        let preferredIndices = Array(preferredTokens.filter { $0 >= 0 && $0 < actualVocabSize })
        
        // Create bias array - MLX will handle GPU optimization
        let biasArray: MLXArray
        if !preferredIndices.isEmpty {
            var biasValues = [Float](repeating: 0, count: actualVocabSize)
            for idx in preferredIndices {
                biasValues[Int(idx)] = microBias
            }
            biasArray = MLXArray(biasValues)
        } else {
            biasArray = MLX.zeros([actualVocabSize])
        }
        
        // Reshape bias for broadcasting
        var shape = Array(repeating: 1, count: logits.ndim)
        shape[logits.ndim - 1] = actualVocabSize
        let reshapedBias = biasArray.reshaped(shape)
        
        // Apply micro-bias: logits + small_positive_bias
        return logits + reshapedBias
    }
    
    // MARK: - Debugging
    
    public func debugState() -> String {
        let phaseDescription: String
        switch jsonState.phase {
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
        TokenTrieLogitProcessor State:
        - JSON Phase: \(phaseDescription)
        - Token Path Length: \(tokenPath.tokens.count)
        - Is at Terminal: \(tokenPath.isAtTerminal())
        - Generated Tokens: \(generatedTokens.count)
        - Violation Count: \(violationCount)
        - Stack Depth: \(jsonState.stack.count)
        """
    }
}

// MARK: - Extensions

extension TokenTrieLogitProcessor {
    /// Convenience initializer with default settings
    public convenience init(keys: [String], tokenizer: any Tokenizer, microBias: Float = 0.2) {
        let schema = SchemaMeta(keys: keys, required: [])
        self.init(schema: schema, tokenizer: tokenizer, microBias: microBias)
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
