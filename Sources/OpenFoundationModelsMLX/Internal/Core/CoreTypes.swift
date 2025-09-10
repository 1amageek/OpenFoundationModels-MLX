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
            let ids = tokenizer.encode(key)
            print("🔑 [TokenTrieBuilder] Key '\(key)' encoded to tokens: \(ids)")
            trie.insert(tokenIDs: ids, keyName: key)
        }
        
        print("✅ [TokenTrieBuilder] Trie built with \(uniqueKeys.count) keys")
        return trie
    }
    
}