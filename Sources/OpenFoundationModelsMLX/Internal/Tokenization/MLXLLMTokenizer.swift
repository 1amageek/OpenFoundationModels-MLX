import Foundation
import Tokenizers

// TokenizerAdapter protocol for tokenization
public protocol TokenizerAdapter: Sendable {
    func encode(_ text: String) -> [Int32]
    func decode(_ ids: [Int32]) -> String
    func getVocabSize() -> Int?
    func fingerprint() -> String
}

// Simple tokenizer adapter for MLXLLM
public final class MLXLLMTokenizer: TokenizerAdapter, @unchecked Sendable {

    private let tokenizer: any Tokenizer

    public init(tokenizer: any Tokenizer) {
        self.tokenizer = tokenizer
    }

    // MARK: - TokenizerAdapter Protocol

    public func encode(_ text: String) -> [Int32] {
        let encoded = tokenizer.encode(text: text, addSpecialTokens: false)
        return encoded.map { Int32($0) }
    }

    public func decode(_ ids: [Int32]) -> String {
        let tokens = ids.map { Int($0) }
        return tokenizer.decode(tokens: tokens)
    }

    public func getVocabSize() -> Int? {
        // Most tokenizers don't expose vocab size directly
        return nil
    }

    public func fingerprint() -> String {
        // Generate a fingerprint based on tokenizer properties
        var fingerprint = "mlx-tokenizer"
        if let vocabSize = getVocabSize() {
            fingerprint += "-v\(vocabSize)"
        }
        if let eos = eosTokenId() {
            fingerprint += "-e\(eos)"
        }
        if let bos = bosTokenId() {
            fingerprint += "-b\(bos)"
        }
        return fingerprint
    }

    // MARK: - Utility Methods

    public func decodeToken(_ tokenID: Int32) -> String {
        return decode([tokenID])
    }

    public func eosTokenId() -> Int32? {
        return tokenizer.eosTokenId.map { Int32($0) }
    }

    public func bosTokenId() -> Int32? {
        return tokenizer.bosTokenId.map { Int32($0) }
    }

    public func unknownTokenId() -> Int32? {
        return tokenizer.unknownTokenId.map { Int32($0) }
    }
}
