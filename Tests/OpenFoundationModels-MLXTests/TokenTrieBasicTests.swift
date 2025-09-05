import Testing
@testable import OpenFoundationModelsMLX

@Suite struct TokenTrieBasicTests {
    
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
    
    // MARK: - Basic Functionality Tests
    
    @Test func initializesEmpty() {
        let trie = TokenTrie()
        #expect(trie.allKeys.isEmpty)
        #expect(trie.allowedNext(from: []) != nil)
    }
    
    @Test func insertsBasicTokenSequence() {
        var trie = TokenTrie()
        let tokens: [Int32] = [65, 66, 67] // "ABC"
        
        trie.insert(tokenIDs: tokens, keyName: "test")
        
        #expect(trie.allKeys.contains("test"))
        #expect(trie.node(for: tokens) != nil)
        #expect(trie.node(for: tokens)?.terminal == true)
        #expect(trie.node(for: tokens)?.keyName == "test")
    }
    
    @Test func insertsMultipleSequences() {
        var trie = TokenTrie()
        let seq1: [Int32] = [65, 66] // "AB"
        let seq2: [Int32] = [67, 68] // "CD"
        
        trie.insert(tokenIDs: seq1, keyName: "key1")
        trie.insert(tokenIDs: seq2, keyName: "key2")
        
        #expect(trie.allKeys.count == 2)
        #expect(trie.allKeys.contains("key1"))
        #expect(trie.allKeys.contains("key2"))
        #expect(trie.node(for: seq1)?.terminal == true)
        #expect(trie.node(for: seq2)?.terminal == true)
    }
    
    @Test func handlesOverlappingSequences() {
        var trie = TokenTrie()
        let seq1: [Int32] = [65, 66] // "AB"
        let seq2: [Int32] = [65, 66, 67] // "ABC" - extends seq1
        
        trie.insert(tokenIDs: seq1, keyName: "short")
        trie.insert(tokenIDs: seq2, keyName: "long")
        
        #expect(trie.allKeys.count == 2)
        #expect(trie.node(for: seq1)?.terminal == true)
        #expect(trie.node(for: seq1)?.keyName == "short")
        #expect(trie.node(for: seq2)?.terminal == true)
        #expect(trie.node(for: seq2)?.keyName == "long")
    }
    
    @Test func retrievesAllowedTokens() {
        var trie = TokenTrie()
        trie.insert(tokenIDs: [65, 66], keyName: "ab")
        trie.insert(tokenIDs: [65, 67], keyName: "ac")
        
        // At root, should allow 'A' (65)
        let rootResult = trie.allowedNext(from: [])
        #expect(rootResult?.ids == [65])
        #expect(rootResult?.atTerminal == false)
        
        // After 'A', should allow 'B' (66) or 'C' (67)
        let afterA = trie.allowedNext(from: [65])
        #expect(afterA?.ids == [66, 67])
        #expect(afterA?.atTerminal == false)
        
        // After 'AB' or 'AC', should be terminal with no next tokens
        let afterAB = trie.allowedNext(from: [65, 66])
        #expect(afterAB?.ids.isEmpty == true)
        #expect(afterAB?.atTerminal == true)
    }
    
    @Test func handlesNonExistentPaths() {
        var trie = TokenTrie()
        trie.insert(tokenIDs: [65, 66], keyName: "ab")
        
        // Path that doesn't exist
        let result = trie.allowedNext(from: [99, 100])
        #expect(result == nil)
        
        // Partial path that doesn't exist
        let partialResult = trie.allowedNext(from: [65, 99])
        #expect(partialResult == nil)
    }
    
    // MARK: - Path Basic Tests
    
    @Test func initializesPathCorrectly() {
        let trie = TokenTrie()
        let path = TokenTrie.Path()
        
        #expect(path.tokens.isEmpty)
        #expect(!path.isValid())
        #expect(!path.isAtTerminal())
        #expect(path.getKeyName() == nil)
    }
    
    @Test func initializesPathWithRoot() {
        let trie = TokenTrie()
        let path = TokenTrie.Path(root: trie.root)
        
        #expect(path.tokens.isEmpty)
        #expect(path.isValid())
        #expect(!path.isAtTerminal())
    }
    
    @Test func appendsValidTokensSequentially() {
        var trie = TokenTrie()
        trie.insert(tokenIDs: [65, 66, 67], keyName: "abc")
        
        var path = TokenTrie.Path(root: trie.root)
        
        var result = path.append(65, in: trie)
        #expect(result == true)
        #expect(path.tokens == [65])
        #expect(path.isValid())
        
        result = path.append(66, in: trie)
        #expect(result == true)
        #expect(path.tokens == [65, 66])
        #expect(path.isValid())
        
        result = path.append(67, in: trie)
        #expect(result == true)
        #expect(path.tokens == [65, 66, 67])
        #expect(path.isValid())
        #expect(path.isAtTerminal())
        #expect(path.getKeyName() == "abc")
    }
    
    @Test func rejectsInvalidTokens() {
        var trie = TokenTrie()
        trie.insert(tokenIDs: [65, 66], keyName: "ab")
        
        var path = TokenTrie.Path(root: trie.root)
        
        var result = path.append(65, in: trie)
        #expect(result == true) // Valid: goes to 'A'
        
        result = path.append(99, in: trie)
        #expect(result == false) // Invalid: 'A' + 'c' not in trie
        
        // Path should remain in last valid state
        #expect(path.tokens == [65])
        #expect(path.isValid())
    }
    
    // MARK: - TokenTrieBuilder Tests
    
    @Test func buildsFromKeyArray() {
        let tokenizer = SimpleTokenizer()
        let keys = ["name", "age", "email"]
        
        let trie = TokenTrieBuilder.build(keys: keys, tokenizer: tokenizer)
        
        #expect(trie.allKeys == Set(keys))
        
        // Test that each key can be traversed
        for key in keys {
            let tokens = tokenizer.encode(key)
            #expect(trie.node(for: tokens)?.terminal == true)
            #expect(trie.node(for: tokens)?.keyName == key)
        }
    }
    
    @Test func filtersEmptyKeys() {
        let tokenizer = SimpleTokenizer()
        let keys = ["name", "", "age"] // Include empty key
        
        let trie = TokenTrieBuilder.build(keys: keys, tokenizer: tokenizer)
        
        // Should only include non-empty keys
        #expect(trie.allKeys.count == 2)
        #expect(trie.allKeys.contains("name"))
        #expect(trie.allKeys.contains("age"))
        #expect(!trie.allKeys.contains(""))
    }
    
    @Test func handlesDuplicateKeys() {
        let tokenizer = SimpleTokenizer()
        let keys = ["name", "age", "name", "email", "age"] // Duplicates
        
        let trie = TokenTrieBuilder.build(keys: keys, tokenizer: tokenizer)
        
        // Should deduplicate
        #expect(trie.allKeys.count == 3)
        #expect(trie.allKeys.contains("name"))
        #expect(trie.allKeys.contains("age"))
        #expect(trie.allKeys.contains("email"))
    }
    
    @Test func buildsFromSchemaMeta() {
        let tokenizer = SimpleTokenizer()
        let schema = SchemaMeta(keys: ["user", "pass"], required: ["user"])
        
        let trie = TokenTrieBuilder.build(from: schema, tokenizer: tokenizer)
        
        #expect(trie.allKeys == Set(schema.keys))
        
        // Verify both keys are properly encoded
        for key in schema.keys {
            let tokens = tokenizer.encode(key)
            #expect(trie.node(for: tokens)?.terminal == true)
            #expect(trie.node(for: tokens)?.keyName == key)
        }
    }
    
    // MARK: - Edge Cases
    
    @Test func handlesEmptyTokenSequence() {
        var trie = TokenTrie()
        trie.insert(tokenIDs: [], keyName: "empty")
        
        // Empty sequence should not be inserted
        #expect(!trie.allKeys.contains("empty"))
        #expect(trie.node(for: [])?.terminal == false)
    }
    
    @Test func handlesNilKeyName() {
        var trie = TokenTrie()
        trie.insert(tokenIDs: [65], keyName: nil)
        
        let node = trie.node(for: [65])
        #expect(node?.terminal == true)
        #expect(node?.keyName == nil)
        #expect(trie.allKeys.isEmpty)
    }
    
    @Test func handlesDuplicateInsertions() {
        var trie = TokenTrie()
        let tokens: [Int32] = [65, 66]
        
        trie.insert(tokenIDs: tokens, keyName: "first")
        trie.insert(tokenIDs: tokens, keyName: "second") // overwrites
        
        #expect(trie.allKeys.count == 2) // Both keys tracked
        #expect(trie.node(for: tokens)?.keyName == "second") // Latest wins
        #expect(trie.node(for: tokens)?.terminal == true)
    }
    
    // MARK: - JSON Constraint Tests
    
    @Test func constrainsJSONKeys() {
        let tokenizer = SimpleTokenizer()
        let jsonKeys = ["name", "age", "email"]
        let trie = TokenTrieBuilder.build(keys: jsonKeys, tokenizer: tokenizer)
        
        #expect(trie.allKeys == Set(jsonKeys))
        
        // Test that each key can be generated under JSON constraints
        for key in jsonKeys {
            let tokens = tokenizer.encode(key)
            #expect(!tokens.isEmpty, "Key '\(key)' produced empty tokens")
            
            let node = trie.node(for: tokens)
            #expect(node?.terminal == true, "Key '\(key)' is not terminal")
            #expect(node?.keyName == key, "Key name mismatch for '\(key)'")
        }
    }
    
    @Test func handlesSpecialCharacters() {
        let tokenizer = SimpleTokenizer()
        
        // Keys containing special characters (within ASCII range)
        let specialKeys = ["user_name", "user-id", "key:value"]
        let trie = TokenTrieBuilder.build(keys: specialKeys, tokenizer: tokenizer)
        
        for key in specialKeys {
            let tokens = tokenizer.encode(key)
            #expect(!tokens.isEmpty, "Special key '\(key)' encoded to empty tokens")
            
            // Verify round-trip
            let decoded = tokenizer.decode(tokens)
            #expect(decoded == key, "Round-trip failed: '\(key)' -> '\(decoded)'")
            
            // Verify trie contains the key
            let node = trie.node(for: tokens)
            #expect(node?.terminal == true, "Special key '\(key)' not found in trie")
        }
    }
    
    // MARK: - Schema Integration Tests
    
    @Test func enforcesRequiredKeys() {
        let tokenizer = SimpleTokenizer()
        
        let schema = SchemaMeta(
            keys: ["id", "name", "email", "age"],
            required: ["id", "name", "email"]
        )
        
        let trie = TokenTrieBuilder.build(from: schema, tokenizer: tokenizer)
        
        // All schema keys should be in trie
        for key in schema.keys {
            #expect(trie.allKeys.contains(key), "Schema key '\(key)' missing from trie")
        }
        
        // Required keys should be accessible
        for requiredKey in schema.required {
            let tokens = tokenizer.encode(requiredKey)
            let node = trie.node(for: tokens)
            #expect(node?.terminal == true, "Required key '\(requiredKey)' not terminal in trie")
        }
    }
    
    @Test func validatesCacheEfficiency() {
        let tokenizer = SimpleTokenizer()
        let schema = SchemaMeta(keys: ["cached_key"], required: [])
        
        let trie1 = TokenTrieBuilder.buildCached(schema: schema, tokenizer: tokenizer)
        let trie2 = TokenTrieBuilder.buildCached(schema: schema, tokenizer: tokenizer)
        
        #expect(trie1.allKeys == trie2.allKeys)
        
        // Verify functionality
        let tokens = tokenizer.encode("cached_key")
        #expect(trie1.node(for: tokens)?.terminal == true)
        #expect(trie2.node(for: tokens)?.terminal == true)
    }
}
