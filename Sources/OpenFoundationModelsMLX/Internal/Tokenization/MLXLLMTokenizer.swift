import Foundation
import MLXLMCommon
import MLXLLM
import Tokenizers

// Concrete implementation of TokenizerAdapter for MLXLLM
// This wraps the MLXLMCommon Tokenizer to provide token encoding/decoding
// for Schema-Constrained Decoding (SCD)
public final class MLXLLMTokenizer: TokenizerAdapter, @unchecked Sendable {
    
    // Special token IDs for common JSON symbols
    public struct SpecialTokens: Sendable {
        public let quoteTokens: Set<Int32>      // Tokens containing '"'
        public let colonTokens: Set<Int32>      // Tokens containing ':'
        public let braceOpenTokens: Set<Int32>  // Tokens containing '{'
        public let braceCloseTokens: Set<Int32> // Tokens containing '}'
        public let commaTokens: Set<Int32>      // Tokens containing ','
    }
    
    private let tokenizer: any Tokenizer
    private var _specialTokens: SpecialTokens?
    
    public init(tokenizer: any Tokenizer) {
        self.tokenizer = tokenizer
        self._specialTokens = nil
    }
    
    // MARK: - TokenizerAdapter Protocol
    
    public func encode(_ text: String) -> [Int32] {
        let encoded = tokenizer.encode(text: text, addSpecialTokens: false)
        return encoded.map { Int32($0) }
    }
    
    public func decode(_ ids: [Int32]) -> String {
        let intIds = ids.map { Int($0) }
        return tokenizer.decode(tokens: intIds, skipSpecialTokens: false)
    }
    
    public func getVocabSize() -> Int? {
        // Unfortunately swift-transformers doesn't expose vocabulary size directly
        // We could potentially estimate it by trying to decode a range of token IDs
        // but that would be expensive. For now, return nil and let the caller
        // handle the absence of vocab size information.
        return nil
    }
    
    // MARK: - Extended API
    
    public func decodeToken(_ id: Int32) -> String {
        return decode([id])
    }
    
    public func getSpecialTokens() -> SpecialTokens? {
        return _specialTokens ?? findSpecialTokens()
    }
    
    // MARK: - Special Token Discovery
    
    public func findSpecialTokens() -> SpecialTokens {
        if let cached = _specialTokens {
            return cached
        }
        
        var quoteTokens = Set<Int32>()
        var colonTokens = Set<Int32>()
        var braceOpenTokens = Set<Int32>()
        var braceCloseTokens = Set<Int32>()
        var commaTokens = Set<Int32>()
        
        // Optimized approach: batch encode all symbols and decode only unique tokens
        let jsonSymbols = ["\"", ":", "{", "}", ","]
        let symbolsWithSpace = [" \"", "\" ", " :", ": ", " {", "{ ", " }", "} ", " ,", ", "]
        let allSymbols = jsonSymbols + symbolsWithSpace
        
        // Batch process all symbols - encode once
        var symbolToTokens: [String: [Int32]] = [:]
        for symbol in allSymbols {
            let tokenIDs = tokenizer.encode(text: symbol, addSpecialTokens: false)
            symbolToTokens[symbol] = tokenIDs.map { Int32($0) }
        }
        
        // Collect all unique token IDs
        let uniqueTokens = Set(symbolToTokens.values.flatMap { $0 })
        
        // Batch decode all unique tokens - decode once per unique token
        var tokenToText: [Int32: String] = [:]
        for tokenID in uniqueTokens {
            tokenToText[tokenID] = tokenizer.decode(tokens: [Int(tokenID)], skipSpecialTokens: false)
        }
        
        // Map tokens to categories based on decoded text
        for (symbol, tokenIDs) in symbolToTokens {
            let targetChar = symbol.trimmingCharacters(in: .whitespaces)
            
            for tokenID in tokenIDs {
                guard let decoded = tokenToText[tokenID],
                      decoded.contains(targetChar),
                      decoded.count <= 3 else { continue }
                
                switch targetChar {
                case "\"":
                    quoteTokens.insert(tokenID)
                case ":":
                    colonTokens.insert(tokenID)
                case "{":
                    braceOpenTokens.insert(tokenID)
                case "}":
                    braceCloseTokens.insert(tokenID)
                case ",":
                    commaTokens.insert(tokenID)
                default:
                    break
                }
            }
        }
        
        // Additional validation for exact matches
        for symbol in jsonSymbols {
            let tokenIDs = tokenizer.encode(text: symbol, addSpecialTokens: false)
            if tokenIDs.count == 1 {
                let tokenID = Int32(tokenIDs[0])
                let decoded = tokenizer.decode(tokens: tokenIDs, skipSpecialTokens: false)
                if decoded == symbol {
                    switch symbol {
                    case "\"":
                        quoteTokens.insert(tokenID)
                    case ":":
                        colonTokens.insert(tokenID)
                    case "{":
                        braceOpenTokens.insert(tokenID)
                    case "}":
                        braceCloseTokens.insert(tokenID)
                    case ",":
                        commaTokens.insert(tokenID)
                    default:
                        break
                    }
                }
            }
        }
        
        // Log discovery results
        print("[MLXLLMTokenizer] Special tokens discovered (optimized):")
        print("  Quotes: \(quoteTokens.count) tokens")
        print("  Colons: \(colonTokens.count) tokens")
        print("  Braces: \(braceOpenTokens.count) open, \(braceCloseTokens.count) close")
        print("  Commas: \(commaTokens.count) tokens")
        
        let specialTokens = SpecialTokens(
            quoteTokens: quoteTokens,
            colonTokens: colonTokens,
            braceOpenTokens: braceOpenTokens,
            braceCloseTokens: braceCloseTokens,
            commaTokens: commaTokens
        )
        
        _specialTokens = specialTokens
        return specialTokens
    }
    
    // MARK: - Token Validation
    
    /// Validates that tokens actually contain the expected character when decoded
    private func validateTokens(_ tokens: Set<Int32>, shouldContain char: String) -> Set<Int32> {
        return tokens.filter { tokenID in
            let decoded = tokenizer.decode(tokens: [Int(tokenID)])
            return decoded.contains(char) && decoded.count <= 3 // Reasonable length limit
        }
    }
    
    // MARK: - Utility Methods
    
    /// Check if a token ID represents a special JSON token
    public func isSpecialToken(_ tokenID: Int32) -> Bool {
        let special = getSpecialTokens() ?? findSpecialTokens()
        return special.quoteTokens.contains(tokenID) ||
               special.colonTokens.contains(tokenID) ||
               special.braceOpenTokens.contains(tokenID) ||
               special.braceCloseTokens.contains(tokenID) ||
               special.commaTokens.contains(tokenID)
    }
    
    /// Get the unknown token ID if available
    public func unknownTokenId() -> Int32? {
        return tokenizer.unknownTokenId.map { Int32($0) }
    }
    
    /// Get the end-of-sequence token ID if available  
    public func eosTokenId() -> Int32? {
        return tokenizer.eosTokenId.map { Int32($0) }
    }
}

// MARK: - Token Cache

// Cache tokenizer instances per model to avoid re-initialization
actor TokenizerCache {
    static let shared = TokenizerCache()
    private var cache: [String: MLXLLMTokenizer] = [:]
    
    func tokenizer(for modelId: String, tokenizer: Tokenizer) -> MLXLLMTokenizer {
        if let existing = cache[modelId] {
            return existing
        }
        let mlxTokenizer = MLXLLMTokenizer(tokenizer: tokenizer)
        cache[modelId] = mlxTokenizer
        return mlxTokenizer
    }
    
    func clear() {
        cache.removeAll()
    }
}