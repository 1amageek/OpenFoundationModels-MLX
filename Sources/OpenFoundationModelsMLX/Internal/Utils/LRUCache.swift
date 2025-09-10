import Foundation

/// Thread-safe LRU cache implementation
final class LRUCache<Key: Hashable, Value>: @unchecked Sendable {
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
    
    private let maxSize: Int
    private var cache: [Key: Node] = [:]
    private let head = Node()  // Dummy head
    private let tail = Node()  // Dummy tail
    private let lock = NSLock()
    
    var count: Int {
        lock.withLock { cache.count }
    }
    
    init(maxSize: Int) {
        self.maxSize = maxSize
        head.next = tail
        tail.prev = head
    }
    
    func get(_ key: Key) -> Value? {
        lock.withLock {
            guard let node = cache[key] else { return nil }
            
            // Move to front (most recently used)
            removeNode(node)
            addToFront(node)
            
            return node.value!
        }
    }
    
    func set(_ key: Key, value: Value) {
        lock.withLock {
            if let node = cache[key] {
                // Update existing
                node.value = value
                removeNode(node)
                addToFront(node)
            } else {
                // Add new
                let newNode = Node(key: key, value: value)
                cache[key] = newNode
                addToFront(newNode)
                
                // Evict if needed
                if cache.count > maxSize {
                    if let lru = tail.prev, lru !== head, let lruKey = lru.key {
                        removeNode(lru)
                        cache.removeValue(forKey: lruKey)
                    }
                }
            }
        }
    }
    
    func clear() {
        lock.withLock {
            cache.removeAll()
            head.next = tail
            tail.prev = head
        }
    }
    
    private func removeNode(_ node: Node) {
        node.prev?.next = node.next
        node.next?.prev = node.prev
    }
    
    private func addToFront(_ node: Node) {
        node.prev = head
        node.next = head.next
        head.next?.prev = node
        head.next = node
    }
}

extension NSLock {
    func withLock<T>(_ body: () throws -> T) rethrows -> T {
        lock()
        defer { unlock() }
        return try body()
    }
}