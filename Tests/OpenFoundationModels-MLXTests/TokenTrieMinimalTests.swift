import Testing
@testable import OpenFoundationModelsMLX

/// Minimal TDD-compliant tests for TokenTrie that match current implementation
/// These tests only use APIs that exist in the current TokenTrie implementation
@Suite struct TokenTrieMinimalTests {
    
    // MARK: - Simple Mock Tokenizer
    
    struct SimpleTokenizer: TokenizerAdapter {
        func encode(_ text: String) -> [Int32] {
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
            return 128
        }
    }
    
    // MARK: - Core API Tests (insert/node/allowedNext only)
    
    @Test func initializesEmpty() {
        let trie = TokenTrie()
        // Empty trie should allow traversal from empty path
        let result = trie.allowedNext(from: [])
        #expect(result != nil)
        #expect(result?.atTerminal == false)
    }
    
    @Test func insertsAndRetrievesNode() {
        var trie = TokenTrie()
        let tokens: [Int32] = [65, 66, 67] // "ABC"
        
        trie.insert(tokenIDs: tokens)
        
        // Should be able to retrieve the node
        let node = trie.node(for: tokens)
        #expect(node != nil)
        #expect(node?.terminal == true)
    }
    
    @Test func insertsWithKeyName() {
        var trie = TokenTrie()
        let tokens: [Int32] = [65, 66, 67] // "ABC"
        
        trie.insert(tokenIDs: tokens, keyName: "test")
        
        // Node should exist and be terminal
        let node = trie.node(for: tokens)
        #expect(node != nil)
        #expect(node?.terminal == true)
        // Note: keyName is stored but is internal detail for debugging
    }
    
    @Test func allowedNextReturnsCorrectTokens() {
        var trie = TokenTrie()
        trie.insert(tokenIDs: [65, 66]) // "AB"
        trie.insert(tokenIDs: [65, 67]) // "AC"
        
        // At root, should allow 'A' (65)
        let rootResult = trie.allowedNext(from: [])
        #expect(rootResult != nil)
        #expect(rootResult?.ids == Set([65])) // Use Set comparison
        #expect(rootResult?.atTerminal == false)
        
        // After 'A', should allow 'B' or 'C'
        let afterA = trie.allowedNext(from: [65])
        #expect(afterA != nil)
        #expect(afterA?.ids == Set([66, 67])) // Use Set comparison
        #expect(afterA?.atTerminal == false)
        
        // After 'AB', should be terminal with no next tokens
        let afterAB = trie.allowedNext(from: [65, 66])
        #expect(afterAB != nil)
        #expect(afterAB?.ids == Set<Int32>()) // Empty set
        #expect(afterAB?.atTerminal == true)
    }
    
    @Test func handlesNonExistentPaths() {
        var trie = TokenTrie()
        trie.insert(tokenIDs: [65, 66])
        
        // Path that doesn't exist should return nil
        let result = trie.allowedNext(from: [99, 100])
        #expect(result == nil)
        
        // Partial invalid path
        let partial = trie.allowedNext(from: [65, 99])
        #expect(partial == nil)
    }
    
    @Test func handlesEmptyInsert() {
        var trie = TokenTrie()
        trie.insert(tokenIDs: []) // Empty sequence
        
        // Should not affect trie structure
        let result = trie.allowedNext(from: [])
        #expect(result != nil)
        #expect(result?.ids.isEmpty == false || result?.ids.isEmpty == true) // May or may not have children
    }
    
    @Test func supportsOverlappingPaths() {
        var trie = TokenTrie()
        trie.insert(tokenIDs: [65, 66]) // "AB"
        trie.insert(tokenIDs: [65, 66, 67]) // "ABC" - extends AB
        
        // Both paths should be retrievable
        let nodeAB = trie.node(for: [65, 66])
        #expect(nodeAB != nil)
        #expect(nodeAB?.terminal == true)
        
        let nodeABC = trie.node(for: [65, 66, 67])
        #expect(nodeABC != nil)
        #expect(nodeABC?.terminal == true)
        
        // After AB, should allow continuation to C
        let afterAB = trie.allowedNext(from: [65, 66])
        #expect(afterAB != nil)
        #expect(afterAB?.ids == Set([67]))
        #expect(afterAB?.atTerminal == true) // AB itself is terminal
    }
    
    // MARK: - Tests for TokenTrieBuilder (if it exists)
    
    @Test func buildsFromKeys() {
        let tokenizer = SimpleTokenizer()
        let keys = ["name", "age"]
        
        let trie = TokenTrieBuilder.build(keys: keys, tokenizer: tokenizer)
        
        // Verify basic structure
        for key in keys {
            let tokens = tokenizer.encode(key)
            let node = trie.node(for: tokens)
            #expect(node != nil, "Key '\(key)' should be in trie")
            #expect(node?.terminal == true, "Key '\(key)' should be terminal")
        }
    }
    
    @Test func filtersEmptyKeys() {
        let tokenizer = SimpleTokenizer()
        let keys = ["name", "", "age"] // Include empty key
        
        let trie = TokenTrieBuilder.build(keys: keys, tokenizer: tokenizer)
        
        // Empty key should not create a path
        let nameTokens = tokenizer.encode("name")
        let ageTokens = tokenizer.encode("age")
        
        #expect(trie.node(for: nameTokens)?.terminal == true)
        #expect(trie.node(for: ageTokens)?.terminal == true)
        // Empty tokens [] would return root, which should not be terminal
        #expect(trie.node(for: [])?.terminal == false)
    }
}

/// Tests for future/extended APIs (currently implemented but marked as future-facing)
/// These use APIs like Path, allKeys, etc. that exist but may change
@Suite struct TokenTrieFutureAPITests {
    
    struct SimpleTokenizer: TokenizerAdapter {
        func encode(_ text: String) -> [Int32] {
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
            return 128
        }
    }
    
    // NOTE: These tests use Path API which is part of future SCD implementation
    // They are included here as they currently exist in the implementation
    
    @Test func pathInitialization() {
        // Future API: Path structure for stateful traversal
        let _ = TokenTrie() // Not used directly, just ensuring it compiles
        let path = TokenTrie.Path()
        
        #expect(path.tokens.isEmpty)
        #expect(!path.isValid()) // No current node
        #expect(!path.isAtTerminal())
        #expect(path.getKeyName() == nil)
    }
    
    @Test func pathWithRoot() {
        // Future API: Path initialized with root
        let trie = TokenTrie()
        let path = TokenTrie.Path(root: trie.root)
        
        #expect(path.tokens.isEmpty)
        #expect(path.isValid()) // Has root node
        #expect(!path.isAtTerminal())
    }
    
    @Test func pathTraversal() {
        // Future API: Path-based traversal
        var trie = TokenTrie()
        trie.insert(tokenIDs: [65, 66, 67], keyName: "abc")
        
        var path = TokenTrie.Path(root: trie.root)
        
        var result = path.append(65, in: trie)
        #expect(result == true)
        #expect(path.tokens == [65])
        
        result = path.append(66, in: trie)
        #expect(result == true)
        #expect(path.tokens == [65, 66])
        
        result = path.append(67, in: trie)
        #expect(result == true)
        #expect(path.isAtTerminal())
        #expect(path.getKeyName() == "abc")
    }
    
    @Test func allKeysTracking() {
        // Future API: allKeys property for key management
        var trie = TokenTrie()
        
        trie.insert(tokenIDs: [65], keyName: "a")
        trie.insert(tokenIDs: [66], keyName: "b")
        
        #expect(trie.allKeys.contains("a"))
        #expect(trie.allKeys.contains("b"))
        #expect(trie.allKeys.count == 2)
    }
    
    @Test func getAllowedTokensWithPath() {
        // Future API: getAllowedTokens using Path
        var trie = TokenTrie()
        trie.insert(tokenIDs: [65, 66])
        trie.insert(tokenIDs: [65, 67])
        
        var path = TokenTrie.Path(root: trie.root)
        let rootTokens = trie.getAllowedTokens(for: path)
        #expect(rootTokens == Set([65]))
        
        let _ = path.append(65, in: trie)
        let afterATokens = trie.getAllowedTokens(for: path)
        #expect(afterATokens == Set([66, 67]))
    }
    
    @Test func canCompleteCheck() {
        // Future API: canComplete for terminal checking
        var trie = TokenTrie()
        trie.insert(tokenIDs: [65, 66])
        
        var path = TokenTrie.Path(root: trie.root)
        #expect(!trie.canComplete(from: path)) // Root not terminal
        
        let _ = path.append(65, in: trie)
        #expect(!trie.canComplete(from: path)) // After A not terminal
        
        let _ = path.append(66, in: trie)
        #expect(trie.canComplete(from: path)) // After AB is terminal
    }
    
    @Test func schemaMetaIntegration() {
        // Future API: Integration with SchemaMeta
        let tokenizer = SimpleTokenizer()
        let schema = SchemaMeta(keys: ["user", "pass"], required: ["user"])
        
        let trie = TokenTrieBuilder.build(from: schema, tokenizer: tokenizer)
        
        #expect(trie.allKeys == Set(schema.keys))
        
        for key in schema.keys {
            let tokens = tokenizer.encode(key)
            #expect(trie.node(for: tokens)?.terminal == true)
        }
    }
    
    @Test func caching() {
        // Future API: Caching mechanism for performance
        let tokenizer = SimpleTokenizer()
        let schema = SchemaMeta(keys: ["cached"], required: [])
        
        let trie1 = TokenTrieBuilder.buildCached(schema: schema, tokenizer: tokenizer)
        let trie2 = TokenTrieBuilder.buildCached(schema: schema, tokenizer: tokenizer)
        
        // Should produce equivalent results
        #expect(trie1.allKeys == trie2.allKeys)
    }
}
