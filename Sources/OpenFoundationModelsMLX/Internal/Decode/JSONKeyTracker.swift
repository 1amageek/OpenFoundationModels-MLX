import Foundation

// Heuristic JSON key tracker used during streaming to detect when a partial
// key cannot possibly match any schema property name. This enables early abort
// and retry without exposing a broken attempt to the outer API.
struct JSONKeyTracker: Sendable {
    private let schemaKeys: [String]
    private let trie: KeyTrie
    private var lastNonWS: Character = "{"
    private(set) var readingKey = false
    private(set) var expectingColon = false
    private(set) var keyBuffer = ""
    private var escaped = false
    private var inString = false
    private(set) var violationCount = 0

    init(schemaKeys: [String]) {
        self.schemaKeys = schemaKeys
        self.trie = KeyTrie(keys: schemaKeys)
    }

    mutating func consume(_ chunk: String) {
        for ch in chunk {
            let isWS = ch.isWhitespace
            if !isWS { lastNonWS = ch }

            // handle string escapes
            if ch == "\\" && inString { escaped.toggle(); continue }

            if ch == "\"" && !escaped {
                inString.toggle()
                if !readingKey && (lastNonWS == "{" || lastNonWS == ",") {
                    // Potentially a key start
                    readingKey = true
                    keyBuffer.removeAll(keepingCapacity: true)
                    violationCount = 0
                } else if readingKey {
                    // End of key
                    readingKey = false
                    expectingColon = true
                }
                continue
            }

            if readingKey {
                keyBuffer.append(ch)
                let nk = SchemaSnapParser.normalize(keyBuffer)
                let anyPrefix = trie.hasPrefix(nk)
                if anyPrefix { violationCount = 0 } else { violationCount += 1 }
            } else if expectingColon {
                if ch == ":" { expectingColon = false }
            }

            escaped = false
        }
    }
}
