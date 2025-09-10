import Foundation
import Tokenizers

/// Index that maintains a TokenTrie for each object node in the schema hierarchy
/// This enables dynamic switching of allowed keys based on the current JSON context
public struct SchemaTrieIndex: Sendable {
    
    public let root: SchemaNode
    private let triesByNode: [ObjectIdentifier: TokenTrie]
    private let triesByKeys: [String: TokenTrie]  // Fallback mapping by key signature
    
    /// Build the index by creating a TokenTrie for each object node
    /// - Parameters:
    ///   - root: The root schema node
    ///   - tokenizer: Tokenizer adapter for encoding keys
    public init(root: SchemaNode, tokenizer: TokenizerAdapter) {
        self.root = root
        
        var triesMap: [ObjectIdentifier: TokenTrie] = [:]
        var keyTriesMap: [String: TokenTrie] = [:]
        
        // Recursively build tries for all object nodes
        func buildTries(for node: SchemaNode, path: String = "root") {
            switch node.kind {
            case .object:
                let nodeId = ObjectIdentifier(node)
                
                // Build trie for this object's keys
                let keys = node.objectKeys
                if !keys.isEmpty {
                    print("🌳 ADAPT: Creating TokenTrie for object at path '\(path)' with keys: \(keys)")
                    let trie = TokenTrieBuilder.build(keys: keys, tokenizer: tokenizer)
                    triesMap[nodeId] = trie
                    
                    // Create fallback key signature and store trie
                    let keySignature = keys.joined(separator: ",")
                    keyTriesMap[keySignature] = trie
                    print("🌳 ADAPT: TokenTrie created with \(trie.allKeys.count) keys")
                } else {
                    print("🌳 ADAPT: Empty object at path '\(path)', no TokenTrie created")
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
        self.triesByKeys = keyTriesMap
        print("🌳 ADAPT: SchemaTrieIndex built with \(triesMap.count) TokenTries")
        
    }
    
    /// Get the TokenTrie for a specific schema node
    /// - Parameter node: The schema node
    /// - Returns: TokenTrie if the node is an object with keys, nil otherwise
    public func trie(for node: SchemaNode) -> TokenTrie? {
        let nodeId = ObjectIdentifier(node)
        
        // Try primary lookup by ObjectIdentifier
        if let result = triesByNode[nodeId] {
            print("🌳 ADAPT: Looking up Trie for node - found with \(result.allKeys.count) keys: \(result.allKeys)")
            return result
        }
        
        // Fallback to key signature lookup for object nodes
        if node.kind == .object {
            let keys = node.objectKeys
            if !keys.isEmpty {
                let keySignature = keys.joined(separator: ",")
                if let fallbackResult = triesByKeys[keySignature] {
                    print("🌳 ADAPT: Fallback lookup succeeded for keys: \(keys)")
                    return fallbackResult
                } else {
                    print("🌳 ADAPT: No Trie found for keys: \(keys)")
                }
            } else {
                // Silently return nil for empty objects
            }
        }
        
        return nil
    }
    
    /// Get all tries in the index (for debugging)
    public var allTries: [TokenTrie] {
        Array(triesByNode.values)
    }
    
    /// Check if a node has an associated trie
    public func hasTrie(for node: SchemaNode) -> Bool {
        let nodeId = ObjectIdentifier(node)
        
        // Try primary lookup
        if triesByNode[nodeId] != nil {
            return true
        }
        
        // Try fallback for object nodes
        if node.kind == .object {
            let keys = node.objectKeys
            if !keys.isEmpty {
                let keySignature = keys.joined(separator: ",")
                return triesByKeys[keySignature] != nil
            }
        }
        
        return false
    }
}