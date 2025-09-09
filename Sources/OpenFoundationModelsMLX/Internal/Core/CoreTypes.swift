import Foundation
import Synchronization

// Core types needed for schema-constrained decoding

/// Schema metadata for JSON generation with injection protection
public struct SchemaMeta: Sendable {
    public let keys: [String]
    public let required: [String]
    
    public init(keys: [String], required: [String] = []) {
        // Validate keys to prevent JSON injection
        let sanitizedKeys = keys.compactMap { key -> String? in
            // Check for potentially dangerous characters
            guard Self.isValidJSONKey(key) else {
                Logger.warning("[SchemaMeta] Rejected potentially dangerous key: '\(key)'")
                return nil
            }
            return key
        }
        
        self.keys = sanitizedKeys
        self.required = required.filter { sanitizedKeys.contains($0) }
        
        if sanitizedKeys.count < keys.count {
            Logger.warning("[SchemaMeta] Sanitized \(keys.count - sanitizedKeys.count) potentially dangerous keys")
        }
    }
    
    /// Validate a JSON key to prevent injection attacks
    private static func isValidJSONKey(_ key: String) -> Bool {
        // Reject empty keys
        guard !key.isEmpty else { return false }
        
        // Reject keys with control characters
        let controlCharacters = CharacterSet.controlCharacters
        if key.rangeOfCharacter(from: controlCharacters) != nil {
            return false
        }
        
        // Reject keys with special JSON characters that could break structure
        let dangerousChars = ["\"", "\\", "\n", "\r", "\t", "\0"]
        for char in dangerousChars {
            if key.contains(char) {
                return false
            }
        }
        
        // Reject excessively long keys (potential DoS)
        if key.count > 256 {
            return false
        }
        
        // Reject keys that look like code injection attempts
        let injectionPatterns = [
            "__proto__",      // Prototype pollution
            "constructor",    // Constructor injection
            "prototype",      // Prototype manipulation
            "$where",         // MongoDB injection
            "$regex",         // Regex injection
            "eval",          // Code execution
            "Function",      // Function constructor
            "<script",       // XSS attempt
            "javascript:",   // JavaScript protocol
            "onclick",       // Event handler injection
        ]
        
        let lowercased = key.lowercased()
        for pattern in injectionPatterns {
            if lowercased.contains(pattern) {
                return false
            }
        }
        
        return true
    }
}

// Token-level trie for schema keys
public struct TokenTrie: Sendable {
    public final class Node: @unchecked Sendable { 
        public var children: [Int32: Node] = [:]
        public var terminal = false 
        public var keyName: String?
    }
    
    public let root = Node()
    public var allKeys: Set<String> = []

    public init() {}

    public mutating func insert(tokenIDs: [Int32], keyName: String? = nil) {
        guard !tokenIDs.isEmpty else { return }
        var node = root
        for id in tokenIDs {
            if node.children[id] == nil { node.children[id] = Node() }
            node = node.children[id]!
        }
        node.terminal = true
        node.keyName = keyName
        print("🌳 [TokenTrie.insert] Marked node as terminal for key '\(keyName ?? "unknown")' at path \(tokenIDs)")
        if let key = keyName {
            allKeys.insert(key)
        }
    }

    public func node(for path: [Int32]) -> Node? {
        var node = root
        for id in path {
            guard let n = node.children[id] else { return nil }
            node = n
        }
        return node
    }

    public func allowedNext(from path: [Int32]) -> (ids: Set<Int32>, atTerminal: Bool)? {
        guard let node = node(for: path) else { return nil }
        return (Set(node.children.keys), node.terminal)
    }
    
    public func getAllowedTokens(for path: Path) -> Set<Int32> {
        guard let currentNode = path.currentNode ?? node(for: path.tokens) else {
            print("⚠️ [TokenTrie] No current node for path tokens: \(path.tokens)")
            return []
        }
        let allowed = Set(currentNode.children.keys)
        print("🔍 [TokenTrie] Allowed tokens for path \(path.tokens): \(allowed.prefix(5)) (total: \(allowed.count))")
        return allowed
    }
    
    public func canComplete(from path: Path) -> Bool {
        guard let currentNode = path.currentNode ?? node(for: path.tokens) else {
            return false
        }
        return currentNode.terminal
    }
    
    // Path tracker for maintaining state during generation
    public struct Path: Sendable {
        public private(set) var tokens: [Int32] = []
        public private(set) var currentNode: Node?
        
        public init() {
            self.currentNode = nil
        }
        
        public init(root: Node) {
            self.currentNode = root
        }
        
        public mutating func append(_ tokenID: Int32, in trie: TokenTrie) -> Bool {
            let nextNode: Node?
            if let current = currentNode {
                nextNode = current.children[tokenID]
            } else {
                nextNode = trie.root.children[tokenID]
            }
            
            guard let node = nextNode else {
                print("⚠️ [Path.append] No node found for token \(tokenID)")
                return false
            }
            
            tokens.append(tokenID)
            currentNode = node
            print("📍 [Path.append] Added token \(tokenID), path now: \(tokens), terminal: \(node.terminal), keyName: \(node.keyName ?? "nil")")
            return true
        }
        
        public mutating func reset(to root: Node? = nil) {
            tokens.removeAll(keepingCapacity: true)
            currentNode = root
        }
        
        public func isAtTerminal() -> Bool {
            let result = currentNode?.terminal ?? false
            print("🎯 [Path.isAtTerminal] Checking terminal: \(result), currentNode exists: \(currentNode != nil), keyName: \(currentNode?.keyName ?? "nil")")
            return result
        }
        
        public func getKeyName() -> String? {
            return currentNode?.keyName
        }
        
        public func isValid() -> Bool {
            return currentNode != nil
        }
    }
}

// Tokenizer adapter protocol
public protocol TokenizerAdapter: Sendable {
    func encode(_ text: String) -> [Int32]
    func decode(_ ids: [Int32]) -> String
    func getVocabSize() -> Int?
    
    /// Generate a unique identifier for this tokenizer
    /// Used for cache key generation to prevent cross-model contamination
    func fingerprint() -> String
}

// TokenTrie builder
public enum TokenTrieBuilder {
    // Thread-safe cache: NSCache is thread-safe, but we use Mutex for atomic get/set operations
    private final class TokenTrieBox: NSObject { 
        let value: TokenTrie
        init(_ v: TokenTrie) { self.value = v } 
    }
    nonisolated(unsafe) private static let trieCache = NSCache<NSString, TokenTrieBox>()
    private static let cacheMutex = Mutex(())
    
    public static func build(keys: [String], tokenizer: TokenizerAdapter) -> TokenTrie {
        var trie = TokenTrie()
        let uniqueKeys = Set(keys).filter { !$0.isEmpty }
        
        print("🔨 [TokenTrieBuilder] Building trie for keys: \(uniqueKeys)")
        
        for key in uniqueKeys {
            let ids = tokenizer.encode(key)
            print("🔑 [TokenTrieBuilder] Key '\(key)' encoded to tokens: \(ids)")
            trie.insert(tokenIDs: ids, keyName: key)
        }
        
        print("✅ [TokenTrieBuilder] Trie built with \(uniqueKeys.count) keys")
        return trie
    }
    
    public static func build(from schema: SchemaMeta, tokenizer: TokenizerAdapter) -> TokenTrie {
        return build(keys: schema.keys, tokenizer: tokenizer)
    }
    
    public static func buildCached(schema: SchemaMeta, tokenizer: TokenizerAdapter) -> TokenTrie {
        // Cache key = tokenizer fingerprint + schema keys (sorted)
        let tokenizerFingerprint = tokenizer.fingerprint()
        let schemaKey = schema.keys.sorted().joined(separator: "|")
        let cacheKey = "\(tokenizerFingerprint)|\(schemaKey)" as NSString

        // Use explicit closure to handle cache access safely
        let cachedTrie = cacheMutex.withLock { _ -> TokenTrie? in
            trieCache.object(forKey: cacheKey)?.value
        }
        
        if let cached = cachedTrie {
            return cached
        }
        
        // Build outside lock to avoid blocking other threads
        let trie = build(from: schema, tokenizer: tokenizer)
        
        // Store in cache
        cacheMutex.withLock { _ -> Void in
            trieCache.setObject(TokenTrieBox(trie), forKey: cacheKey)
        }
        
        return trie
    }
}