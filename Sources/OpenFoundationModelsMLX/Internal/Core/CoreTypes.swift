import Foundation
import Synchronization
public struct TokenTrie: Sendable {
    public final class Node: @unchecked Sendable { 
        public var children: [Int32: Node] = [:]
        public var terminal = false 
        public var keyName: String?
    }
    
    public let root = Node()
    public var allKeys: Set<String> = []
    public var singleTokenMap: [Int32: String] = [:]  // Maps single-token IDs to key names

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
    
    /// Get all valid tokens for starting a key (includes single-token keys)
    public func getKeyStartTokens() -> Set<Int32> {
        var tokens = Set<Int32>()
        
        // Multi-token key prefixes
        for (tokenId, _) in root.children {
            tokens.insert(tokenId)
        }
        
        // Single-token keys
        for tokenId in singleTokenMap.keys {
            tokens.insert(tokenId)
        }
        
        print("🚀 [TokenTrie] Key start tokens: multi=\(root.children.keys.count), single=\(singleTokenMap.count), total=\(tokens.count)")
        return tokens
    }
    
    /// Check if a token ID represents a valid single-token key
    public func isValidSingleTokenKey(_ tokenId: Int32) -> Bool {
        return singleTokenMap[tokenId] != nil
    }
    
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

public protocol TokenizerAdapter: Sendable {
    func encode(_ text: String) -> [Int32]
    func decode(_ ids: [Int32]) -> String
    func getVocabSize() -> Int?
    func fingerprint() -> String
}

public enum TokenTrieBuilder {
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
            // Try different encoding patterns to detect single-token keys
            
            // Pattern 1: "\"key\"" - full quoted key
            let fullQuoted = "\"" + key + "\""
            let fullQuotedTokens = tokenizer.encode(fullQuoted)
            
            // Pattern 2: "\"key" - key with opening quote
            let openQuoted = "\"" + key
            let openQuotedTokens = tokenizer.encode(openQuoted)
            
            // Pattern 3: just the key
            let bareTokens = tokenizer.encode(key)
            
            print("🔑 [TokenTrieBuilder] Key '\(key)':")
            print("   Full quoted (\"\(key)\"): \(fullQuotedTokens)")
            print("   Open quoted (\"key): \(openQuotedTokens)")
            print("   Bare (key): \(bareTokens)")
            
            // Check if it's a single-token key
            if fullQuotedTokens.count == 1 {
                // Complete key with quotes is a single token
                trie.singleTokenMap[fullQuotedTokens[0]] = key
                print("   ✨ Single-token key detected: token \(fullQuotedTokens[0]) -> '\(key)'")
            } else if openQuotedTokens.count > 0 {
                // Use the open-quoted version for multi-token keys
                trie.insert(tokenIDs: openQuotedTokens, keyName: key)
                print("   📝 Multi-token key: \(openQuotedTokens)")
            } else {
                // Fallback to bare tokens
                trie.insert(tokenIDs: bareTokens, keyName: key)
                print("   📝 Fallback to bare tokens: \(bareTokens)")
            }
        }
        
        print("✅ [TokenTrieBuilder] Trie built with \(uniqueKeys.count) keys, \(trie.singleTokenMap.count) single-token")
        return trie
    }
    
}