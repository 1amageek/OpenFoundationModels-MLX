import Foundation
import Tokenizers

/// Adapter to bridge HuggingFace tokenizer to TokenizerAdapter protocol
public struct HuggingFaceTokenizerAdapter: TokenizerAdapter, @unchecked Sendable {
    private let tokenizer: Tokenizers.Tokenizer
    
    public init(tokenizer: Tokenizers.Tokenizer) {
        self.tokenizer = tokenizer
    }
    
    public func encode(_ text: String) -> [Int32] {
        let encoding = tokenizer.encode(text: text)
        return encoding.map { Int32($0) }
    }
    
    public func decode(_ ids: [Int32]) -> String {
        let intIds = ids.map { Int($0) }
        return tokenizer.decode(tokens: intIds)
    }
    
    public func getVocabSize() -> Int? {
        // Tokenizers package doesn't expose vocabulary size directly
        // Return a reasonable default
        return 50000
    }
    
    public func fingerprint() -> String {
        // Create a simple fingerprint
        return "huggingface-tokenizer"
    }
    
    public var eosTokenId: Int32 {
        // Try to get the actual EOS token ID from the tokenizer
        if let eosToken = tokenizer.eosToken {
            let encoding = tokenizer.encode(text: eosToken)
            if let firstId = encoding.first {
                return Int32(firstId)
            }
        }
        return 2 // Default EOS token ID
    }
    
    public var bosTokenId: Int32 {
        // Try to get the actual BOS token ID from the tokenizer
        if let bosToken = tokenizer.bosToken {
            let encoding = tokenizer.encode(text: bosToken)
            if let firstId = encoding.first {
                return Int32(firstId)
            }
        }
        return 1 // Default BOS token ID
    }
    
    public var unknownTokenId: Int32 {
        // Try to get the actual unknown token ID from the tokenizer
        if let unkToken = tokenizer.unknownToken {
            let encoding = tokenizer.encode(text: unkToken)
            if let firstId = encoding.first {
                return Int32(firstId)
            }
        }
        return 0 // Default unknown token ID
    }
    
    public func convertTokenToString(_ token: Int32) -> String? {
        return decode([token])
    }
}