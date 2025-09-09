import Foundation
import Tokenizers

/// Index that maintains a TokenTrie for each object node in the schema hierarchy
/// This enables dynamic switching of allowed keys based on the current JSON context
public struct SchemaTrieIndex: Sendable {
    
    public let root: SchemaNode
    private let triesByNode: [ObjectIdentifier: TokenTrie]
    
    /// Build the index by creating a TokenTrie for each object node
    /// - Parameters:
    ///   - root: The root schema node
    ///   - tokenizer: Tokenizer adapter for encoding keys
    public init(root: SchemaNode, tokenizer: TokenizerAdapter) {
        self.root = root
        
        var triesMap: [ObjectIdentifier: TokenTrie] = [:]
        
        // Recursively build tries for all object nodes
        func buildTries(for node: SchemaNode, path: String = "root") {
            switch node.kind {
            case .object:
                let nodeId = ObjectIdentifier(node)
                
                // Build trie for this object's keys
                let keys = node.objectKeys
                if !keys.isEmpty {
                    print("🌳 [SchemaTrieIndex] Building trie for \(path) with keys: \(keys)")
                    let trie = TokenTrieBuilder.build(keys: keys, tokenizer: tokenizer)
                    triesMap[nodeId] = trie
                } else {
                    print("⚠️ [SchemaTrieIndex] Object at \(path) has no keys")
                }
                
                // Recursively process child properties
                for (key, child) in node.properties {
                    buildTries(for: child, path: "\(path).\(key)")
                }
                
            case .array:
                // Process array items if they exist
                if let items = node.items {
                    buildTries(for: items, path: "\(path)[]")
                }
                
            default:
                // Primitive types don't need tries
                break
            }
        }
        
        buildTries(for: root)
        self.triesByNode = triesMap
        
        print("✅ [SchemaTrieIndex] Built index with \(triesMap.count) tries")
    }
    
    /// Get the TokenTrie for a specific schema node
    /// - Parameter node: The schema node
    /// - Returns: TokenTrie if the node is an object with keys, nil otherwise
    public func trie(for node: SchemaNode) -> TokenTrie? {
        let nodeId = ObjectIdentifier(node)
        let result = triesByNode[nodeId]
        
        if result != nil {
            print("🔍 [SchemaTrieIndex] Found trie for node with keys: \(node.objectKeys)")
        } else if node.kind == .object {
            print("⚠️ [SchemaTrieIndex] No trie found for object node")
        }
        
        return result
    }
    
    /// Get all tries in the index (for debugging)
    public var allTries: [TokenTrie] {
        Array(triesByNode.values)
    }
    
    /// Check if a node has an associated trie
    public func hasTrie(for node: SchemaNode) -> Bool {
        triesByNode[ObjectIdentifier(node)] != nil
    }
}