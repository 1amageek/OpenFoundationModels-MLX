import Foundation
import MLXLMCommon
import MLXLLM
import Tokenizers

// High-performance decode cache with thread-safe access
private final class DecodeCache {
    private var cache: [Int32: String] = [:]
    private let lock = NSLock()
    private let maxEntries: Int
    
    // Statistics for monitoring
    private(set) var hits: Int = 0
    private(set) var misses: Int = 0
    
    init(maxEntries: Int = 1000) {
        self.maxEntries = maxEntries
    }
    
    func get(_ tokenID: Int32) -> String? {
        lock.lock()
        defer { lock.unlock() }
        
        if let cached = cache[tokenID] {
            hits += 1
            return cached
        } else {
            misses += 1
            return nil
        }
    }
    
    func set(_ tokenID: Int32, _ text: String) {
        lock.lock()
        defer { lock.unlock() }
        
        // Simple eviction if cache is too large
        if cache.count >= maxEntries {
            // Remove ~10% of oldest entries (approximation using dictionary ordering)
            let toRemove = max(1, cache.count / 10)
            for _ in 0..<toRemove {
                if let firstKey = cache.keys.first {
                    cache.removeValue(forKey: firstKey)
                }
            }
        }
        
        cache[tokenID] = text
    }
    
    func clear() {
        lock.lock()
        defer { lock.unlock() }
        cache.removeAll()
        hits = 0
        misses = 0
    }
    
    var hitRate: Double {
        let total = hits + misses
        return total > 0 ? Double(hits) / Double(total) : 0.0
    }
}

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
        public let bracketOpenTokens: Set<Int32>  // Tokens containing '['
        public let bracketCloseTokens: Set<Int32> // Tokens containing ']'
        public let whitespaceTokens: Set<Int32>    // Pure whitespace tokens
        public let backslashTokens: Set<Int32>     // Tokens containing '\'
    }
    
    private let tokenizer: any Tokenizer
    private var _specialTokens: SpecialTokens?
    private var _tokenizerFingerprint: String?
    private let decodeCache: DecodeCache
    
    public init(tokenizer: any Tokenizer) {
        self.tokenizer = tokenizer
        self._specialTokens = nil
        self._tokenizerFingerprint = nil
        self.decodeCache = DecodeCache()
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
        // Check cache first for performance
        if let cached = decodeCache.get(id) {
            return cached
        }
        
        // Decode and cache the result
        let decoded = decode([id])
        decodeCache.set(id, decoded)
        
        // Log cache performance periodically
        if (decodeCache.hits + decodeCache.misses) % 1000 == 0 {
            Logger.debug("[MLXLLMTokenizer] Decode cache hit rate: \(String(format: "%.1f", decodeCache.hitRate * 100))%")
        }
        
        return decoded
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
        var bracketOpenTokens = Set<Int32>()
        var bracketCloseTokens = Set<Int32>()
        var whitespaceTokens = Set<Int32>()
        var backslashTokens = Set<Int32>()
        
        // Optimized approach: batch encode all symbols and decode only unique tokens
        let jsonSymbols = ["\"", ":", "{", "}", ",", "[", "]", "\\"]
        let symbolsWithSpace = [" \"", "\" ", " :", ": ", " {", "{ ", " }", "} ", " ,", ", ", " [", "[ ", " ]", "] ", " \\", "\\ "]
        let whitespaces = [" ", "  ", "   ", "\t", "\n", "\r\n"]
        let allSymbols = jsonSymbols + symbolsWithSpace + whitespaces
        
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
                guard let decoded = tokenToText[tokenID] else { continue }
                
                // Check for whitespace tokens
                if whitespaces.contains(symbol) {
                    let trimmed = decoded.trimmingCharacters(in: .whitespacesAndNewlines)
                    if trimmed.isEmpty && decoded.count <= 4 {
                        whitespaceTokens.insert(tokenID)
                    }
                    continue
                }
                
                // Check for symbol tokens
                guard decoded.contains(targetChar) else { continue }  // Removed length limit to catch composite tokens like "key
                
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
                case "[":
                    bracketOpenTokens.insert(tokenID)
                case "]":
                    bracketCloseTokens.insert(tokenID)
                case "\\":
                    backslashTokens.insert(tokenID)
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
                    case "[":
                        bracketOpenTokens.insert(tokenID)
                    case "]":
                        bracketCloseTokens.insert(tokenID)
                    case "\\":
                        backslashTokens.insert(tokenID)
                    default:
                        break
                    }
                }
            }
        }
        
        // Log discovery results
        Logger.debug("[MLXLLMTokenizer] Special tokens discovered: quotes=\(quoteTokens.count), colons=\(colonTokens.count), bracesOpen=\(braceOpenTokens.count), bracesClose=\(braceCloseTokens.count), commas=\(commaTokens.count), bracketsOpen=\(bracketOpenTokens.count), bracketsClose=\(bracketCloseTokens.count), whitespace=\(whitespaceTokens.count), backslash=\(backslashTokens.count)")
        
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
    
    /// Check if a token ID represents a special JSON token
    public func isSpecialToken(_ tokenID: Int32) -> Bool {
        let special = getSpecialTokens() ?? findSpecialTokens()
        return special.quoteTokens.contains(tokenID) ||
               special.colonTokens.contains(tokenID) ||
               special.braceOpenTokens.contains(tokenID) ||
               special.braceCloseTokens.contains(tokenID) ||
               special.commaTokens.contains(tokenID) ||
               special.bracketOpenTokens.contains(tokenID) ||
               special.bracketCloseTokens.contains(tokenID) ||
               special.whitespaceTokens.contains(tokenID) ||
               special.backslashTokens.contains(tokenID)
    }
    
    /// Get the unknown token ID if available
    public func unknownTokenId() -> Int32? {
        return tokenizer.unknownTokenId.map { Int32($0) }
    }
    
    /// Get the end-of-sequence token ID if available  
    public func eosTokenId() -> Int32? {
        return tokenizer.eosTokenId.map { Int32($0) }
    }
    
    // MARK: - Tokenizer Fingerprinting
    
    /// Generate a unique fingerprint for this tokenizer configuration
    /// Used to prevent cache collisions between different tokenizers
    public func getFingerprint() -> String {
        if let cached = _tokenizerFingerprint {
            return cached
        }
        
        // Compute fingerprint from tokenizer characteristics
        var components: [String] = []
        
        // 1. Tokenizer type (e.g., class name)
        components.append(String(describing: type(of: tokenizer)))
        
        // 2. Special token IDs
        if let eos = eosTokenId() {
            components.append("eos=\(eos)")
        }
        if let unk = unknownTokenId() {
            components.append("unk=\(unk)")
        }
        
        // 3. JSON symbol encodings (critical for constraint correctness)
        let jsonSymbols = [
            ("quote", "\""),
            ("colon", ":"),
            ("brace_open", "{"),
            ("brace_close", "}"),
            ("bracket_open", "["),
            ("bracket_close", "]"),
            ("comma", ","),
            ("backslash", "\\")
        ]
        
        for (name, symbol) in jsonSymbols {
            let encoded = encode(symbol)
            let encodingStr = encoded.map(String.init).joined(separator: ",")
            components.append("\(name)=[\(encodingStr)]")
        }
        
        // 4. Whitespace encoding (affects JSON formatting)
        let whitespaceEncodings = [
            ("space", " "),
            ("tab", "\t"),
            ("newline", "\n")
        ]
        
        for (name, ws) in whitespaceEncodings {
            let encoded = encode(ws)
            let encodingStr = encoded.map(String.init).joined(separator: ",")
            components.append("\(name)=[\(encodingStr)]")
        }
        
        // 5. Special token counts (for validation)
        let special = findSpecialTokens()
        components.append("quotes_count=\(special.quoteTokens.count)")
        components.append("colons_count=\(special.colonTokens.count)")
        
        let fingerprint = components.joined(separator: "|")
        _tokenizerFingerprint = fingerprint
        
        Logger.debug("[MLXLLMTokenizer] Generated fingerprint: \(fingerprint.prefix(100))...")
        return fingerprint
    }
}

