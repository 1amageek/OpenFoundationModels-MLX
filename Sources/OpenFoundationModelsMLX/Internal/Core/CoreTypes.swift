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
            return []
        }
        return Set(currentNode.children.keys)
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
                return false
            }
            
            tokens.append(tokenID)
            currentNode = node
            return true
        }
        
        public mutating func reset(to root: Node? = nil) {
            tokens.removeAll(keepingCapacity: true)
            currentNode = root
        }
        
        public func isAtTerminal() -> Bool {
            return currentNode?.terminal ?? false
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
}

// TokenTrie builder
public enum TokenTrieBuilder {
    public static func build(keys: [String], tokenizer: TokenizerAdapter) -> TokenTrie {
        var trie = TokenTrie()
        let uniqueKeys = Set(keys).filter { !$0.isEmpty }
        
        for key in uniqueKeys {
            let ids = tokenizer.encode(key)
            trie.insert(tokenIDs: ids, keyName: key)
        }
        return trie
    }
    
    public static func build(from schema: SchemaMeta, tokenizer: TokenizerAdapter) -> TokenTrie {
        return build(keys: schema.keys, tokenizer: tokenizer)
    }
    
    // Thread-safe cache using Mutex for synchronization
    // NSCache access is protected by cacheMutex
    nonisolated(unsafe) private static let trieCache = NSCache<NSString, AnyObject>()
    private static let cacheMutex = Mutex(())
    
    public static func buildCached(schema: SchemaMeta, tokenizer: TokenizerAdapter) -> TokenTrie {
        let cacheKey = schema.keys.sorted().joined(separator: "|") as NSString
        
        // Check cache first
        let cached = cacheMutex.withLock { _ in
            trieCache.object(forKey: cacheKey) as? TokenTrie
        }
        if let cached = cached {
            return cached
        }
        
        // Build outside of lock to avoid blocking other threads
        let trie = build(from: schema, tokenizer: tokenizer)
        
        // Store in cache
        cacheMutex.withLock { _ in
            trieCache.setObject(trie as AnyObject, forKey: cacheKey)
        }
        
        return trie
    }
}