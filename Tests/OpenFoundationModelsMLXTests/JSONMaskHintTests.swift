import Testing
import Foundation
@testable import OpenFoundationModelsMLX
@testable import PRECISE

@Suite("JSON Mask Hint Tests")
struct JSONMaskHintTests {
    
    // MARK: - Test Setup
    
    struct TestSetup {
        let specialTokens: MLXLLMTokenizer.SpecialTokens
        let tokenTrie: TokenTrie
        let tokenPath: TokenTrie.Path
        
        init() {
            // Create mock special tokens
            specialTokens = MLXLLMTokenizer.SpecialTokens(
                quoteTokens: Set([34]),      // "
                colonTokens: Set([58]),      // :
                braceOpenTokens: Set([123]), // {
                braceCloseTokens: Set([125]), // }
                commaTokens: Set([44]),      // ,
                bracketOpenTokens: Set([91]), // [
                bracketCloseTokens: Set([93]), // ]
                whitespaceTokens: Set([32, 9, 10]), // space, tab, newline
                backslashTokens: Set([92])   // \
            )
            
            // Create a simple trie with test keys
            var trie = TokenTrie()
            trie.insert(tokenIDs: [110, 97, 109, 101], keyName: "name")  // "name"
            trie.insert(tokenIDs: [97, 103, 101], keyName: "age")        // "age"
            trie.insert(tokenIDs: [105, 100], keyName: "id")             // "id"
            self.tokenTrie = trie
            self.tokenPath = TokenTrie.Path(root: trie.root)
        }
    }
    
    // MARK: - Root Phase Tests
    
    @Test("Root phase allows value starts")
    func rootPhaseAllowsValues() {
        let setup = TestSetup()
        let generator = JSONMaskHintGenerator.forSchemaConstrainedDecoding(
            specialTokens: setup.specialTokens,
            includeWhitespace: false
        )
        
        let state = JSONStateMachine()
        
        let hint = generator.maskHint(
            for: state,
            tokenTrie: setup.tokenTrie,
            tokenPath: setup.tokenPath
        )
        
        #expect(hint != nil)
        #expect(hint?.mode == .soft)  // Root should be soft for numbers/literals
        #expect(hint?.allow.contains(34) == true)   // " for strings
        #expect(hint?.allow.contains(123) == true)  // { for objects
        #expect(hint?.allow.contains(91) == true)   // [ for arrays
    }
    
    // MARK: - Object Phase Tests
    
    @Test("Object expects key or close")
    func objectExpectsKeyOrClose() {
        let setup = TestSetup()
        let generator = JSONMaskHintGenerator.forSchemaConstrainedDecoding(
            specialTokens: setup.specialTokens,
            includeWhitespace: false
        )
        
        var state = JSONStateMachine()
        state.processCharacter("{")
        
        let hint = generator.maskHint(
            for: state,
            tokenTrie: setup.tokenTrie,
            tokenPath: setup.tokenPath
        )
        
        #expect(hint != nil)
        #expect(hint?.allow.contains(34) == true)   // " to start key
        #expect(hint?.allow.contains(125) == true)  // } to close empty object
    }
    
    @Test("Object key emission uses TokenTrie")
    func objectKeyEmissionUsesTrie() {
        let setup = TestSetup()
        let generator = JSONMaskHintGenerator.forSchemaConstrainedDecoding(
            specialTokens: setup.specialTokens,
            includeWhitespace: false
        )
        
        var state = JSONStateMachine()
        state.processCharacter("{")
        state.processCharacter("\"")
        
        // Now in key body - should use TokenTrie constraints
        let hint = generator.maskHint(
            for: state,
            tokenTrie: setup.tokenTrie,
            tokenPath: setup.tokenPath
        )
        
        #expect(hint != nil)
        #expect(hint?.mode == .hard)  // Key emission should be hard constrained
        
        // Should allow tokens that start valid keys
        // "name" starts with 'n' (110), "age" starts with 'a' (97), "id" starts with 'i' (105)
        #expect(hint?.allow.contains(110) == true)  // 'n'
        #expect(hint?.allow.contains(97) == true)   // 'a'
        #expect(hint?.allow.contains(105) == true)  // 'i'
    }
    
    @Test("Expects colon after key")
    func expectsColonAfterKey() {
        let setup = TestSetup()
        let generator = JSONMaskHintGenerator.forSchemaConstrainedDecoding(
            specialTokens: setup.specialTokens,
            includeWhitespace: false
        )
        
        var state = JSONStateMachine()
        state.processCharacter("{")
        state.processCharacter("\"")
        state.processCharacter("n")
        state.processCharacter("a")
        state.processCharacter("m")
        state.processCharacter("e")
        state.processCharacter("\"")
        
        // Should now expect colon
        let hint = generator.maskHint(
            for: state,
            tokenTrie: setup.tokenTrie,
            tokenPath: setup.tokenPath
        )
        
        #expect(hint != nil)
        #expect(hint?.allow.contains(58) == true)  // : colon
        #expect(hint?.allow.count == 1)  // Only colon allowed
    }
    
    // MARK: - Array Phase Tests
    
    @Test("Array expects value or close")
    func arrayExpectsValueOrClose() {
        let setup = TestSetup()
        let generator = JSONMaskHintGenerator.forSchemaConstrainedDecoding(
            specialTokens: setup.specialTokens,
            includeWhitespace: false
        )
        
        var state = JSONStateMachine()
        state.processCharacter("[")
        
        let hint = generator.maskHint(
            for: state,
            tokenTrie: setup.tokenTrie,
            tokenPath: setup.tokenPath
        )
        
        #expect(hint != nil)
        #expect(hint?.mode == .soft)  // Arrays allow any values
        #expect(hint?.allow.contains(34) == true)   // " for strings
        #expect(hint?.allow.contains(123) == true)  // { for objects
        #expect(hint?.allow.contains(91) == true)   // [ for nested arrays
        #expect(hint?.allow.contains(93) == true)   // ] to close empty array
    }
    
    @Test("Array expects comma or close after value")
    func arrayExpectsCommaOrClose() {
        let setup = TestSetup()
        let generator = JSONMaskHintGenerator.forSchemaConstrainedDecoding(
            specialTokens: setup.specialTokens,
            includeWhitespace: false
        )
        
        var state = JSONStateMachine()
        // Use a complete literal value instead of a number
        state.processCharacter("[")
        state.processCharacter("t")
        state.processCharacter("r")
        state.processCharacter("u")
        state.processCharacter("e")
        
        // After "true" literal completes, we should be in array expecting comma or close
        let hint = generator.maskHint(
            for: state,
            tokenTrie: setup.tokenTrie,
            tokenPath: setup.tokenPath
        )
        
        #expect(hint != nil)
        #expect(hint?.allow.contains(44) == true)  // , comma
        #expect(hint?.allow.contains(93) == true)  // ] close
    }
    
    // MARK: - String Phase Tests
    
    @Test("String value has no constraints")
    func stringValueNoConstraints() {
        let setup = TestSetup()
        let generator = JSONMaskHintGenerator.forSchemaConstrainedDecoding(
            specialTokens: setup.specialTokens,
            includeWhitespace: false
        )
        
        var state = JSONStateMachine()
        state.processCharacter("\"")  // Start string at root (value)
        
        let hint = generator.maskHint(
            for: state,
            tokenTrie: setup.tokenTrie,
            tokenPath: setup.tokenPath
        )
        
        #expect(hint == nil)  // No constraints for string values
    }
    
    // MARK: - Done Phase Tests
    
    @Test("Done phase only allows EOS")
    func donePhaseOnlyAllowsEOS() {
        let setup = TestSetup()
        let generator = JSONMaskHintGenerator.forSchemaConstrainedDecoding(
            specialTokens: setup.specialTokens,
            includeWhitespace: false
        )
        
        var state = JSONStateMachine()
        // Process a complete JSON
        for char in "\"test\"" {
            state.processCharacter(char)
        }
        
        #expect(state.phase == .done)
        
        let hint = generator.maskHint(
            for: state,
            tokenTrie: setup.tokenTrie,
            tokenPath: setup.tokenPath
        )
        
        #expect(hint != nil)
        #expect(hint?.mode == .hard)
        #expect(hint?.allow.isEmpty == true)  // Empty set - only EOS allowed (added by processor)
    }
    
    // MARK: - Error Phase Tests
    
    @Test("Error phase blocks all tokens")
    func errorPhaseBlocksAll() {
        let setup = TestSetup()
        let generator = JSONMaskHintGenerator.forSchemaConstrainedDecoding(
            specialTokens: setup.specialTokens,
            includeWhitespace: false
        )
        
        var state = JSONStateMachine()
        state.processCharacter("x")  // Invalid at root
        
        #expect(state.phase == .error)
        
        let hint = generator.maskHint(
            for: state,
            tokenTrie: setup.tokenTrie,
            tokenPath: setup.tokenPath
        )
        
        #expect(hint != nil)
        #expect(hint?.allow.isEmpty == true)  // No tokens allowed in error state
    }
    
    // MARK: - Whitespace Tests
    
    @Test("Includes whitespace when enabled")
    func includesWhitespace() {
        let setup = TestSetup()
        let generator = JSONMaskHintGenerator.forSchemaConstrainedDecoding(
            specialTokens: setup.specialTokens,
            includeWhitespace: true  // Enable whitespace
        )
        
        var state = JSONStateMachine()
        state.processCharacter("{")
        
        let hint = generator.maskHint(
            for: state,
            tokenTrie: setup.tokenTrie,
            tokenPath: setup.tokenPath
        )
        
        #expect(hint != nil)
        #expect(hint?.allow.contains(32) == true)  // space
        #expect(hint?.allow.contains(9) == true)   // tab
        #expect(hint?.allow.contains(10) == true)  // newline
    }
    
    @Test("Excludes whitespace when disabled")
    func excludesWhitespace() {
        let setup = TestSetup()
        let generator = JSONMaskHintGenerator.forSchemaConstrainedDecoding(
            specialTokens: setup.specialTokens,
            includeWhitespace: false  // Disable whitespace
        )
        
        var state = JSONStateMachine()
        state.processCharacter("{")
        
        let hint = generator.maskHint(
            for: state,
            tokenTrie: setup.tokenTrie,
            tokenPath: setup.tokenPath
        )
        
        #expect(hint != nil)
        #expect(hint?.allow.contains(32) == false)  // no space
        #expect(hint?.allow.contains(9) == false)   // no tab
        #expect(hint?.allow.contains(10) == false)  // no newline
    }
    
    // MARK: - Mode Tests
    
    @Test("Hard mode for structural tokens")
    func hardModeForStructural() {
        let setup = TestSetup()
        let generator = JSONMaskHintGenerator.forSchemaConstrainedDecoding(
            specialTokens: setup.specialTokens,
            includeWhitespace: false
        )
        
        var state = JSONStateMachine()
        state.processCharacter("{")
        state.processCharacter("\"")
        
        // In key body - should be hard mode
        let hint = generator.maskHint(
            for: state,
            tokenTrie: setup.tokenTrie,
            tokenPath: setup.tokenPath
        )
        
        #expect(hint?.mode == .hard)
    }
    
    @Test("Soft mode for value positions")
    func softModeForValues() {
        let setup = TestSetup()
        let generator = JSONMaskHintGenerator.forSchemaConstrainedDecoding(
            specialTokens: setup.specialTokens,
            includeWhitespace: false
        )
        
        let state = JSONStateMachine()
        
        // Root position - should be soft for numbers/literals
        let hint = generator.maskHint(
            for: state,
            tokenTrie: setup.tokenTrie,
            tokenPath: setup.tokenPath
        )
        
        #expect(hint?.mode == .soft)
    }
}