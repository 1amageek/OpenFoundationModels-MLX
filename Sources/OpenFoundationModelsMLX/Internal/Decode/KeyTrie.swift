import Foundation

// Normalized key trie for fast prefix checks (character-level)
struct KeyTrie: Sendable {
    // Note: Node is @unchecked Sendable because children are mutable during construction,
    // but immutable after the trie is built. This is safe as long as the trie is not
    // modified after construction (which is guaranteed by the struct's design).
    final class Node: @unchecked Sendable { var children: [Character: Node] = [:]; var terminal = false }
    private let root = Node()

    init(keys: [String]) {
        for k in keys.map(SchemaSnapParser.normalize) {
            guard !k.isEmpty else { continue }
            var node = root
            for ch in k {
                if node.children[ch] == nil { node.children[ch] = Node() }
                node = node.children[ch]!
            }
            node.terminal = true
        }
    }

    func hasPrefix(_ nk: String) -> Bool {
        var node = root
        for ch in nk {
            guard let n = node.children[ch] else { return false }
            node = n
        }
        return true
    }
}

