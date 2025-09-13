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
    
    public struct SpecialTokens: Sendable {
        public let quoteTokens: Set<Int32>
        public let colonTokens: Set<Int32>
        public let braceOpenTokens: Set<Int32>
        public let braceCloseTokens: Set<Int32>
        public let commaTokens: Set<Int32>
        public let bracketOpenTokens: Set<Int32>
        public let bracketCloseTokens: Set<Int32>
        public let whitespaceTokens: Set<Int32>
        public let backslashTokens: Set<Int32>
        
        // Convenience properties
        public var quote: Int32? { quoteTokens.first }
        public var colon: Int32? { colonTokens.first }
        public var comma: Int32? { commaTokens.first }
        public var braceOpen: Int32? { braceOpenTokens.first }
        public var braceClose: Int32? { braceCloseTokens.first }
    }
    
    private let tokenizer: any Tokenizer
    private var _specialTokens: SpecialTokens?
    
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
    
    // MARK: - Special Token Detection
    
    public func findSpecialTokens() -> SpecialTokens {
        if let cached = _specialTokens {
            return cached
        }
        
        var quoteTokens = Set<Int32>()
        var colonTokens = Set<Int32>()
        var braceOpenTokens = Set<Int32>()
        var braceCloseTokens = Set<Int32>()
        var commaTokens = Set<Int32>()
        var bracketOpenTokens = Set<Int32>()
        var bracketCloseTokens = Set<Int32>()
        var whitespaceTokens = Set<Int32>()
        var backslashTokens = Set<Int32>()
        
        // Test common JSON symbols
        let jsonSymbols: [(String, String)] = [
            ("\"", "quote"),
            (":", "colon"),
            ("{", "braceOpen"),
            ("}", "braceClose"),
            (",", "comma"),
            ("[", "bracketOpen"),
            ("]", "bracketClose"),
            ("\\", "backslash")
        ]
        
        for (symbol, name) in jsonSymbols {
            let ids = encode(symbol)
            for id in ids {
                // Verify by decoding back
                let decoded = decode([id])
                if decoded == symbol {
                    switch name {
                    case "quote": quoteTokens.insert(id)
                    case "colon": colonTokens.insert(id)
                    case "braceOpen": braceOpenTokens.insert(id)
                    case "braceClose": braceCloseTokens.insert(id)
                    case "comma": commaTokens.insert(id)
                    case "bracketOpen": bracketOpenTokens.insert(id)
                    case "bracketClose": bracketCloseTokens.insert(id)
                    case "backslash": backslashTokens.insert(id)
                    default: break
                    }
                }
            }
        }
        
        // Test whitespace
        for ws in [" ", "\t", "\n", "\r"] {
            let ids = encode(ws)
            for id in ids {
                let decoded = decode([id])
                if decoded.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
                    whitespaceTokens.insert(id)
                }
            }
        }
        
        let specialTokens = SpecialTokens(
            quoteTokens: quoteTokens,
            colonTokens: colonTokens,
            braceOpenTokens: braceOpenTokens,
            braceCloseTokens: braceCloseTokens,
            commaTokens: commaTokens,
            bracketOpenTokens: bracketOpenTokens,
            bracketCloseTokens: bracketCloseTokens,
            whitespaceTokens: whitespaceTokens,
            backslashTokens: backslashTokens
        )
        
        _specialTokens = specialTokens
        return specialTokens
    }
    
    // MARK: - Utility Methods
    
    public func getSpecialTokens() -> SpecialTokens {
        return findSpecialTokens()
    }
    
    public func isSpecialToken(_ tokenID: Int32) -> Bool {
        let special = getSpecialTokens()
        return special.quoteTokens.contains(tokenID) ||
               special.colonTokens.contains(tokenID) ||
               special.braceOpenTokens.contains(tokenID) ||
               special.braceCloseTokens.contains(tokenID) ||
               special.commaTokens.contains(tokenID) ||
               special.bracketOpenTokens.contains(tokenID) ||
               special.bracketCloseTokens.contains(tokenID) ||
               special.backslashTokens.contains(tokenID)
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