import Foundation
import Synchronization

/// Thread-safe LRU cache implementation
final class LRUCache<Key: Hashable & Sendable, Value: Sendable>: Sendable {
    private class Node {
        let key: Key?
        var value: Value?
        var prev: Node?
        var next: Node?
        
        init(key: Key? = nil, value: Value? = nil) {
            self.key = key
            self.value = value
        }
    }
    
    private struct State {
        var cache: [Key: Node] = [:]
        let head = Node()  // Dummy head
        let tail = Node()  // Dummy tail
        
        init() {
            head.next = tail
            tail.prev = head
        }
    }
    
    private let maxSize: Int
    private let mutex = Mutex<State>(.init())
    
    var count: Int {
        mutex.withLock { $0.cache.count }
    }
    
    init(maxSize: Int) {
        self.maxSize = maxSize
    }
    
    func get(_ key: Key) -> Value? {
        mutex.withLock { state in
            guard let node = state.cache[key] else { return nil }
            
            // Move to front (most recently used)
            removeNode(node, in: &state)
            addToFront(node, in: &state)
            
            return node.value!
        }
    }
    
    func set(_ key: Key, value: Value) {
        mutex.withLock { state in
            if let node = state.cache[key] {
                // Update existing
                node.value = value
                removeNode(node, in: &state)
                addToFront(node, in: &state)
            } else {
                // Add new
                let newNode = Node(key: key, value: value)
                state.cache[key] = newNode
                addToFront(newNode, in: &state)
                
                // Evict if needed
                if state.cache.count > maxSize {
                    if let lru = state.tail.prev, lru !== state.head, let lruKey = lru.key {
                        removeNode(lru, in: &state)
                        state.cache.removeValue(forKey: lruKey)
                    }
                }
            }
        }
    }
    
    func clear() {
        mutex.withLock { state in
            state.cache.removeAll()
            state.head.next = state.tail
            state.tail.prev = state.head
        }
    }
    
    private func removeNode(_ node: Node, in state: inout State) {
        node.prev?.next = node.next
        node.next?.prev = node.prev
    }
    
    private func addToFront(_ node: Node, in state: inout State) {
        node.prev = state.head
        node.next = state.head.next
        state.head.next?.prev = node
        state.head.next = node
    }
}