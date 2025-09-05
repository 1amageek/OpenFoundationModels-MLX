import Foundation

// Token-level trie for schema keys. Uses token ids (Int32) provided by a
// tokenizer adapter. This enables strict next-token constraints during key
// emission.
struct TokenTrie: Sendable {
    // Note: Node is @unchecked Sendable because children are mutable during construction,
    // but immutable after the trie is built. This is safe as long as the trie is not
    // modified after construction (which is guaranteed by the struct's design).
    final class Node: @unchecked Sendable { 
        var children: [Int32: Node] = [:]
        var terminal = false 
        var keyName: String? // Store the original key name for debugging
    }
    
    let root = Node()  // Made internal for Path initialization
    var allKeys: Set<String> = []

    init() {}

    mutating func insert(tokenIDs: [Int32], keyName: String? = nil) {
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

    // Returns the node reached by following the given path; nil if no path.
    func node(for path: [Int32]) -> Node? {
        var node = root
        for id in path {
            guard let n = node.children[id] else { return nil }
            node = n
        }
        return node
    }

    func allowedNext(from path: [Int32]) -> (ids: Set<Int32>, atTerminal: Bool)? {
        guard let node = node(for: path) else { return nil }
        return (Set(node.children.keys), node.terminal)
    }
    
    // Get all allowed token IDs from current path
    func getAllowedTokens(for path: Path) -> Set<Int32> {
        guard let currentNode = path.currentNode ?? node(for: path.tokens) else {
            return []
        }
        return Set(currentNode.children.keys)
    }
    
    // Check if we can complete a key from current position
    func canComplete(from path: Path) -> Bool {
        guard let currentNode = path.currentNode ?? node(for: path.tokens) else {
            return false
        }
        return currentNode.terminal
    }
    
    // Path tracker for maintaining state during generation
    struct Path: Sendable {
        private(set) var tokens: [Int32] = []
        private(set) var currentNode: Node?
        
        init() {
            self.currentNode = nil
        }
        
        init(root: Node) {
            self.currentNode = root
        }
        
        mutating func append(_ tokenID: Int32, in trie: TokenTrie) -> Bool {
            let nextNode: Node?
            if let current = currentNode {
                nextNode = current.children[tokenID]
            } else {
                // Start from root if no current node
                nextNode = trie.root.children[tokenID]
            }
            
            guard let node = nextNode else {
                return false // Invalid token for current path
            }
            
            tokens.append(tokenID)
            currentNode = node
            return true
        }
        
        mutating func reset(to root: Node? = nil) {
            tokens.removeAll(keepingCapacity: true)
            currentNode = root
        }
        
        func isAtTerminal() -> Bool {
            return currentNode?.terminal ?? false
        }
        
        func getKeyName() -> String? {
            return currentNode?.keyName
        }
        
        func isValid() -> Bool {
            return currentNode != nil
        }
    }
}

// Minimal tokenizer adapter protocol for building token tries without binding to
// a specific backend. Implementations can wrap MLXLLM or any other tokenizer.
protocol TokenizerAdapter: Sendable {
    func encode(_ text: String) -> [Int32]
    func decode(_ ids: [Int32]) -> String
    func getVocabSize() -> Int?  // Optional vocab size for logits allocation
}

enum TokenTrieBuilder {
    static func build(keys: [String], tokenizer: TokenizerAdapter) -> TokenTrie {
        var trie = TokenTrie()
        // Remove duplicates and filter empty strings
        let uniqueKeys = Set(keys).filter { !$0.isEmpty }
        
        for key in uniqueKeys {
            let ids = tokenizer.encode(key)
            trie.insert(tokenIDs: ids, keyName: key)
        }
        return trie
    }
    
    // Build from schema metadata
    static func build(from schema: SchemaMeta, tokenizer: TokenizerAdapter) -> TokenTrie {
        return build(keys: schema.keys, tokenizer: tokenizer)
    }
    
    // Cache support for performance
    private static let cache = TrieCache()
    
    private final class TrieCache: @unchecked Sendable {
        private var storage: [String: TokenTrie] = [:]
        private let lock = NSLock()
        
        func get(_ key: String) -> TokenTrie? {
            lock.lock()
            defer { lock.unlock() }
            return storage[key]
        }
        
        func set(_ key: String, _ value: TokenTrie) {
            lock.lock()
            defer { lock.unlock() }
            storage[key] = value
        }
    }
    
    static func buildCached(schema: SchemaMeta, tokenizer: TokenizerAdapter, tokenizerID: String? = nil) -> TokenTrie {
        // Include tokenizer identity in the cache key to prevent cross-model Trie reuse
        let tokenizerFingerprint = tokenizerID ?? String(describing: type(of: tokenizer))
        let fingerprint = tokenizerFingerprint + "||" + schema.keys.sorted().joined(separator: "|")
        
        if let cached = cache.get(fingerprint) {
            return cached
        }
        
        let trie = build(from: schema, tokenizer: tokenizer)
        cache.set(fingerprint, trie)
        return trie
    }
}

