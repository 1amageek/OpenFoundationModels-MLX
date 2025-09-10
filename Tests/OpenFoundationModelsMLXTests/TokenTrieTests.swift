import Testing
import Foundation
@testable import OpenFoundationModelsMLX

@Suite("TokenTrie Tests")
struct TokenTrieTests {
    
    // MARK: - Mock Tokenizer
    
    struct MockTokenizer: TokenizerAdapter {
        func encode(_ text: String) -> [Int32] {
            // Simple ASCII encoding for testing
            return text.compactMap { char in
                guard let ascii = char.asciiValue else { return nil }
                return Int32(ascii)
            }
        }
        
        func decode(_ ids: [Int32]) -> String {
            return String(ids.compactMap { id in
                guard id > 0 && id <= 127 else { return nil }
                return Character(UnicodeScalar(Int(id))!)
            })
        }
        
        func getVocabSize() -> Int? {
            return 128  // ASCII range
        }
        
        func fingerprint() -> String {
            // Simple fingerprint for test tokenizer
            return "TokenTrieTests.MockTokenizer-ASCII-128"
        }
    }
    
    // MARK: - Basic Functionality Tests
    
    @Test("Initializes empty trie")
    func emptyTrie() {
        let trie = TokenTrie()
        // Root is always created, no need to check nil
        #expect(trie.allKeys.isEmpty)
    }
    
    @Test("Inserts single token sequence")
    func insertSingleSequence() {
        var trie = TokenTrie()
        let tokens: [Int32] = [65, 66, 67]  // "ABC"
        
        trie.insert(tokenIDs: tokens, keyName: "testKey")
        
        #expect(trie.allKeys == ["testKey"])
        let node = trie.node(for: tokens)
        #expect(node != nil)
        #expect(node?.terminal == true)
        #expect(node?.keyName == "testKey")
    }
    
    @Test("Inserts multiple sequences")
    func insertMultipleSequences() {
        var trie = TokenTrie()
        
        trie.insert(tokenIDs: [65, 66], keyName: "ab")
        trie.insert(tokenIDs: [65, 67], keyName: "ac")
        trie.insert(tokenIDs: [66, 67], keyName: "bc")
        
        #expect(trie.allKeys.sorted() == ["ab", "ac", "bc"])
        
        // Verify each path
        #expect(trie.node(for: [65, 66])?.terminal == true)
        #expect(trie.node(for: [65, 67])?.terminal == true)
        #expect(trie.node(for: [66, 67])?.terminal == true)
    }
    
    @Test("Handles overlapping sequences")
    func overlappingSequences() {
        var trie = TokenTrie()
        
        trie.insert(tokenIDs: [65, 66, 67], keyName: "abc")
        trie.insert(tokenIDs: [65, 66], keyName: "ab")
        
        // Both should be terminal
        let abNode = trie.node(for: [65, 66])
        let abcNode = trie.node(for: [65, 66, 67])
        
        #expect(abNode?.terminal == true)
        #expect(abNode?.keyName == "ab")
        #expect(abcNode?.terminal == true)
        #expect(abcNode?.keyName == "abc")
    }
    
    // MARK: - Path Management Tests
    
    @Test("Path initializes correctly")
    func pathInitialization() {
        let trie = TokenTrie()
        let path = TokenTrie.Path(root: trie.root)
        
        #expect(path.tokens.isEmpty)
        #expect(path.currentNode === trie.root)
    }
    
    @Test("Path appends valid tokens")
    func pathAppendValid() {
        var trie = TokenTrie()
        trie.insert(tokenIDs: [65, 66, 67], keyName: "abc")
        
        var path = TokenTrie.Path(root: trie.root)
        
        #expect(path.append(65, in: trie) == true)
        #expect(path.tokens == [65])
        
        #expect(path.append(66, in: trie) == true)
        #expect(path.tokens == [65, 66])
        
        #expect(path.append(67, in: trie) == true)
        #expect(path.tokens == [65, 66, 67])
        #expect(path.isAtTerminal() == true)
    }
    
    @Test("Path rejects invalid tokens")
    func pathRejectInvalid() {
        var trie = TokenTrie()
        trie.insert(tokenIDs: [65, 66], keyName: "ab")
        
        var path = TokenTrie.Path(root: trie.root)
        
        #expect(path.append(65, in: trie) == true)
        #expect(path.append(67, in: trie) == false)  // 67 not valid after 65
        #expect(path.tokens == [65])  // Should not add invalid token
    }
    
    @Test("Path reset functionality")
    func pathReset() {
        var trie = TokenTrie()
        trie.insert(tokenIDs: [65, 66, 67], keyName: "abc")
        
        var path = TokenTrie.Path(root: trie.root)
        _ = path.append(65, in: trie)
        _ = path.append(66, in: trie)
        
        #expect(path.tokens.count == 2)
        
        path.reset(to: trie.root)
        
        #expect(path.tokens.isEmpty)
        #expect(path.currentNode === trie.root)
    }
    
    @Test("Path terminal detection")
    func pathTerminalDetection() {
        var trie = TokenTrie()
        trie.insert(tokenIDs: [65, 66], keyName: "ab")
        trie.insert(tokenIDs: [65, 66, 67], keyName: "abc")
        
        var path = TokenTrie.Path(root: trie.root)
        
        _ = path.append(65, in: trie)
        #expect(path.isAtTerminal() == false)
        
        _ = path.append(66, in: trie)
        #expect(path.isAtTerminal() == true)  // "ab" is terminal
        
        _ = path.append(67, in: trie)
        #expect(path.isAtTerminal() == true)  // "abc" is also terminal
    }
    
    // MARK: - Allowed Tokens Tests
    
    @Test("Gets allowed tokens from root")
    func allowedTokensFromRoot() {
        var trie = TokenTrie()
        trie.insert(tokenIDs: [65, 66], keyName: "ab")
        trie.insert(tokenIDs: [67, 68], keyName: "cd")
        
        let result = trie.allowedNext(from: [])
        #expect(result != nil)
        #expect(result?.ids.contains(65) == true)
        #expect(result?.ids.contains(67) == true)
        #expect(result?.ids.contains(66) == false)  // Not from root
        #expect(result?.atTerminal == false)
    }
    
    @Test("Gets allowed tokens from partial path")
    func allowedTokensFromPartial() {
        var trie = TokenTrie()
        trie.insert(tokenIDs: [65, 66, 67], keyName: "abc")
        trie.insert(tokenIDs: [65, 66, 68], keyName: "abd")
        
        let result = trie.allowedNext(from: [65, 66])
        #expect(result != nil)
        #expect(result?.ids.contains(67) == true)
        #expect(result?.ids.contains(68) == true)
        #expect(result?.ids.count == 2)
        #expect(result?.atTerminal == false)
    }
    
    @Test("Returns nil for invalid path")
    func allowedTokensInvalidPath() {
        var trie = TokenTrie()
        trie.insert(tokenIDs: [65, 66], keyName: "ab")
        
        let result = trie.allowedNext(from: [67, 68])  // Invalid path
        #expect(result == nil)
    }
    
    // MARK: - Builder Tests
    
    @Test("Builds from schema")
    func buildFromSchema() {
        let keys = ["firstName", "lastName", "email"]
        let tokenizer = MockTokenizer()
        
        let trie = TokenTrieBuilder.build(keys: keys, tokenizer: tokenizer)
        
        #expect(trie.allKeys.sorted() == ["email", "firstName", "lastName"])
        
        // Verify each key is encoded and inserted
        let firstNameTokens = tokenizer.encode("firstName")
        let node = trie.node(for: firstNameTokens)
        #expect(node?.terminal == true)
        #expect(node?.keyName == "firstName")
    }
    
    @Test("Filters empty keys")
    func filterEmptyKeys() {
        let keys = ["valid", "", "  ", "another"]
        let tokenizer = MockTokenizer()
        
        let trie = TokenTrieBuilder.build(keys: keys, tokenizer: tokenizer)
        
        // Note: Only truly empty strings are filtered, not whitespace-only strings
        #expect(trie.allKeys.sorted() == ["  ", "another", "valid"])
    }
    
    @Test("Handles duplicate keys")
    func handleDuplicateKeys() {
        let keys = ["key1", "key1", "key2"]
        let tokenizer = MockTokenizer()
        
        let trie = TokenTrieBuilder.build(keys: keys, tokenizer: tokenizer)
        
        #expect(trie.allKeys.sorted() == ["key1", "key2"])  // Duplicates removed
    }
    
    // MARK: - Cache Tests
    
    @Test("Caches built tries")
    func cacheBuiltTries() {
        let keys = ["test"]
        let tokenizer = MockTokenizer()
        
        // Build multiple times
        let trie1 = TokenTrieBuilder.build(keys: keys, tokenizer: tokenizer)
        
        // Build again
        let trie2 = TokenTrieBuilder.build(keys: keys, tokenizer: tokenizer)
        
        // Should have same keys (caching works based on content)
        #expect(trie1.allKeys == trie2.allKeys)
    }
    
    @Test("Different schemas produce different tries")
    func differentSchemasDifferentTries() {
        let keys1 = ["key1"]
        let keys2 = ["key2"]
        let tokenizer = MockTokenizer()
        
        let trie1 = TokenTrieBuilder.build(keys: keys1, tokenizer: tokenizer)
        let trie2 = TokenTrieBuilder.build(keys: keys2, tokenizer: tokenizer)
        
        // Different schemas should produce different tries
        #expect(trie1.allKeys != trie2.allKeys)
        #expect(trie1.allKeys == ["key1"])
        #expect(trie2.allKeys == ["key2"])
    }
    
    // MARK: - Edge Cases
    
    @Test("Handles empty token sequence")
    func emptyTokenSequence() {
        var trie = TokenTrie()
        trie.insert(tokenIDs: [], keyName: "empty")
        
        // Empty sequence should not create any nodes
        #expect(trie.allKeys.isEmpty)
    }
    
    @Test("Handles very long sequences")
    func longSequences() {
        var trie = TokenTrie()
        let longTokens = Array<Int32>(1...1000)  // 1000 tokens
        
        trie.insert(tokenIDs: longTokens, keyName: "veryLong")
        
        #expect(trie.allKeys == ["veryLong"])
        let node = trie.node(for: longTokens)
        #expect(node?.terminal == true)
    }
    
    @Test("Handles nil key names")
    func nilKeyNames() {
        var trie = TokenTrie()
        trie.insert(tokenIDs: [65, 66], keyName: nil)
        
        let node = trie.node(for: [65, 66])
        #expect(node?.terminal == true)
        #expect(node?.keyName == nil)
        #expect(trie.allKeys.isEmpty)  // Nil keys not tracked
    }
    
    // MARK: - Performance Tests
    
    @Test("Handles many keys efficiently")
    func manyKeys() {
        var trie = TokenTrie()
        
        // Insert 100 different keys
        for i in 0..<100 {
            let tokens = [Int32(65 + (i % 26)), Int32(i)]
            trie.insert(tokenIDs: tokens, keyName: "key\(i)")
        }
        
        #expect(trie.allKeys.count == 100)
        
        // Verify random access is still fast
        // key50 was inserted with tokens [65 + (50 % 26), 50] = [89, 50]
        let testTokens = [Int32(89), Int32(50)]
        let node = trie.node(for: testTokens)
        #expect(node?.keyName == "key50")
    }
}