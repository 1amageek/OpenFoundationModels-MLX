import Testing
import Foundation
import Tokenizers
@testable import OpenFoundationModelsMLX

@Suite("MLXLLM Tokenizer Tests")
struct MLXLLMTokenizerTests {
    
    // MARK: - Mock Tokenizer for Testing
    
    class MockTokenizer: Tokenizer {
        typealias Message = [String: Any]
        typealias ToolSpec = [String: Any]
        
        var vocabulary: [String: Int] = [:]
        var bosToken: String? = "<s>"
        var bosTokenId: Int? = 0
        var eosToken: String? = "</s>"
        var eosTokenId: Int? = 2
        var unknownToken: String? = "<unk>"
        var unknownTokenId: Int? = 1
        var padTokenId: Int? = 3
        
        // Mock encoding mappings
        private let mockEncodings: [String: [Int]] = [
            "\"": [34],
            ":": [58],
            "{": [123],
            "}": [125],
            "[": [91],
            "]": [93],
            ",": [44],
            "\\": [92],
            " ": [32],
            "\t": [9],
            "\n": [10],
            "hello": [104, 101, 108, 108, 111],
            "world": [119, 111, 114, 108, 100],
            "test": [116, 101, 115, 116]
        ]
        
        func tokenize(text: String) -> [String] {
            // Simple tokenization - split into characters
            return text.map { String($0) }
        }
        
        func encode(text: String) -> [Int] {
            return encode(text: text, addSpecialTokens: false)
        }
        
        func encode(text: String, addSpecialTokens: Bool) -> [Int] {
            // Simple mock encoding
            if let encoded = mockEncodings[text] {
                return encoded
            }
            // Fallback to ASCII values for single characters
            if text.count == 1, let ascii = text.first?.asciiValue {
                return [Int(ascii)]
            }
            // Default for unknown
            return text.compactMap { Int($0.asciiValue ?? 0) }
        }
        
        func decode(tokens: [Int], skipSpecialTokens: Bool) -> String {
            // Simple ASCII decoding for testing
            let chars = tokens.compactMap { token -> Character? in
                guard token > 0 && token < 128 else { return nil }
                return Character(UnicodeScalar(token)!)
            }
            return String(chars)
        }
        
        func convertTokenToId(_ token: String) -> Int? {
            return vocabulary[token]
        }
        
        func convertIdToToken(_ id: Int) -> String? {
            return vocabulary.first { $0.value == id }?.key
        }
        
        // Required protocol methods with default implementations
        func applyChatTemplate(messages: [Message]) throws -> [Int] {
            return []
        }
        
        func applyChatTemplate(messages: [Message], tools: [ToolSpec]?) throws -> [Int] {
            return []
        }
        
        func applyChatTemplate(messages: [Message], tools: [ToolSpec]?, additionalContext: [String: Any]?) throws -> [Int] {
            return []
        }
        
        func applyChatTemplate(messages: [Message], chatTemplate: ChatTemplateArgument) throws -> [Int] {
            return []
        }
        
        func applyChatTemplate(messages: [Message], chatTemplate: String) throws -> [Int] {
            return []
        }
        
        func applyChatTemplate(
            messages: [Message],
            chatTemplate: ChatTemplateArgument?,
            addGenerationPrompt: Bool,
            truncation: Bool,
            maxLength: Int?,
            tools: [ToolSpec]?
        ) throws -> [Int] {
            return []
        }
        
        func applyChatTemplate(
            messages: [Message],
            chatTemplate: ChatTemplateArgument?,
            addGenerationPrompt: Bool,
            truncation: Bool,
            maxLength: Int?,
            tools: [ToolSpec]?,
            additionalContext: [String: Any]?
        ) throws -> [Int] {
            return []
        }
    }
    
    // MARK: - Basic Functionality Tests
    
    @Test("Initializes with tokenizer")
    func initialization() {
        let mockTokenizer = MockTokenizer()
        let tokenizer = MLXLLMTokenizer(tokenizer: mockTokenizer)
        
        #expect(tokenizer.eosTokenId() == 2)
        #expect(tokenizer.unknownTokenId() == 1)
    }
    
    @Test("Encodes text to token IDs")
    func encoding() {
        let mockTokenizer = MockTokenizer()
        let tokenizer = MLXLLMTokenizer(tokenizer: mockTokenizer)
        
        let encoded = tokenizer.encode("hello")
        #expect(encoded == [104, 101, 108, 108, 111])
        
        let jsonSymbol = tokenizer.encode("{")
        #expect(jsonSymbol == [123])
    }
    
    @Test("Decodes token IDs to text")
    func decoding() {
        let mockTokenizer = MockTokenizer()
        let tokenizer = MLXLLMTokenizer(tokenizer: mockTokenizer)
        
        let decoded = tokenizer.decode([104, 101, 108, 108, 111])
        #expect(decoded == "hello")
        
        let jsonDecoded = tokenizer.decode([123, 125])
        #expect(jsonDecoded == "{}")
    }
    
    @Test("Decodes single token")
    func singleTokenDecode() {
        let mockTokenizer = MockTokenizer()
        let tokenizer = MLXLLMTokenizer(tokenizer: mockTokenizer)
        
        let decoded = tokenizer.decodeToken(65)  // 'A'
        #expect(decoded == "A")
        
        let braceDecode = tokenizer.decodeToken(123)  // '{'
        #expect(braceDecode == "{")
    }
    
    // MARK: - Special Token Discovery Tests
    
    @Test("Discovers JSON special tokens")
    func specialTokenDiscovery() {
        let mockTokenizer = MockTokenizer()
        let tokenizer = MLXLLMTokenizer(tokenizer: mockTokenizer)
        
        let special = tokenizer.findSpecialTokens()
        
        #expect(special.quoteTokens.contains(34))  // '"'
        #expect(special.colonTokens.contains(58))  // ':'
        #expect(special.braceOpenTokens.contains(123))  // '{'
        #expect(special.braceCloseTokens.contains(125))  // '}'
        #expect(special.commaTokens.contains(44))  // ','
        #expect(special.bracketOpenTokens.contains(91))  // '['
        #expect(special.bracketCloseTokens.contains(93))  // ']'
        #expect(special.backslashTokens.contains(92))  // '\'
    }
    
    @Test("Caches special tokens")
    func specialTokenCaching() {
        let mockTokenizer = MockTokenizer()
        let tokenizer = MLXLLMTokenizer(tokenizer: mockTokenizer)
        
        let special1 = tokenizer.getSpecialTokens()
        let special2 = tokenizer.getSpecialTokens()
        
        // Should return same cached instance
        #expect(special1 != nil)
        #expect(special1?.quoteTokens == special2?.quoteTokens)
    }
    
    @Test("Identifies whitespace tokens")
    func whitespaceTokens() {
        let mockTokenizer = MockTokenizer()
        let tokenizer = MLXLLMTokenizer(tokenizer: mockTokenizer)
        
        let special = tokenizer.findSpecialTokens()
        
        #expect(special.whitespaceTokens.contains(32))  // ' '
        #expect(special.whitespaceTokens.contains(9))   // '\t'
        #expect(special.whitespaceTokens.contains(10))  // '\n'
    }
    
    @Test("Checks if token is special")
    func isSpecialToken() {
        let mockTokenizer = MockTokenizer()
        let tokenizer = MLXLLMTokenizer(tokenizer: mockTokenizer)
        
        #expect(tokenizer.isSpecialToken(34))   // '"' is special
        #expect(tokenizer.isSpecialToken(123))  // '{' is special
        #expect(!tokenizer.isSpecialToken(65))   // 'A' is not special
        #expect(!tokenizer.isSpecialToken(1000)) // Random high ID is not special
    }
    
    // MARK: - Tokenizer Fingerprint Tests
    
    @Test("Generates unique fingerprint")
    func fingerprintGeneration() {
        let mockTokenizer = MockTokenizer()
        let tokenizer = MLXLLMTokenizer(tokenizer: mockTokenizer)
        
        let fingerprint = tokenizer.getFingerprint()
        
        #expect(!fingerprint.isEmpty)
        #expect(fingerprint.contains("eos=2"))
        #expect(fingerprint.contains("unk=1"))
        #expect(fingerprint.contains("quote=[34]"))
        #expect(fingerprint.contains("colon=[58]"))
    }
    
    @Test("Caches fingerprint")
    func fingerprintCaching() {
        let mockTokenizer = MockTokenizer()
        let tokenizer = MLXLLMTokenizer(tokenizer: mockTokenizer)
        
        let fp1 = tokenizer.getFingerprint()
        let fp2 = tokenizer.getFingerprint()
        
        #expect(fp1 == fp2)  // Should return cached value
    }
    
    @Test("Different tokenizers produce different fingerprints")
    func uniqueFingerprints() {
        let mockTokenizer1 = MockTokenizer()
        mockTokenizer1.eosTokenId = 2
        let tokenizer1 = MLXLLMTokenizer(tokenizer: mockTokenizer1)
        
        let mockTokenizer2 = MockTokenizer()
        mockTokenizer2.eosTokenId = 5  // Different EOS
        let tokenizer2 = MLXLLMTokenizer(tokenizer: mockTokenizer2)
        
        let fp1 = tokenizer1.getFingerprint()
        let fp2 = tokenizer2.getFingerprint()
        
        #expect(fp1 != fp2)  // Different fingerprints
    }
    
    // MARK: - Decode Cache Tests
    
    @Test("Decode cache improves performance")
    func decodeCachePerformance() {
        let mockTokenizer = MockTokenizer()
        let tokenizer = MLXLLMTokenizer(tokenizer: mockTokenizer)
        
        // First call - cache miss
        let result1 = tokenizer.decodeToken(65)
        #expect(result1 == "A")
        
        // Second call - should hit cache
        let result2 = tokenizer.decodeToken(65)
        #expect(result2 == "A")
        
        // Different token - cache miss
        let result3 = tokenizer.decodeToken(66)
        #expect(result3 == "B")
        
        // Third call to first token - cache hit
        let result4 = tokenizer.decodeToken(65)
        #expect(result4 == "A")
    }
    
    @Test("Decode cache handles many tokens")
    func decodeCacheManyTokens() {
        let mockTokenizer = MockTokenizer()
        let tokenizer = MLXLLMTokenizer(tokenizer: mockTokenizer)
        
        // Decode many different tokens
        for i in Int32(65)..<Int32(90) {  // A-Z
            let decoded = tokenizer.decodeToken(i)
            #expect(!decoded.isEmpty)
        }
        
        // Decode same tokens again - should hit cache
        for i in Int32(65)..<Int32(90) {
            let decoded = tokenizer.decodeToken(i)
            #expect(!decoded.isEmpty)
        }
    }
    
    // MARK: - Vocabulary Size Tests
    
    @Test("Returns nil for vocabulary size")
    func vocabSize() {
        let mockTokenizer = MockTokenizer()
        let tokenizer = MLXLLMTokenizer(tokenizer: mockTokenizer)
        
        // Current implementation returns nil
        #expect(tokenizer.getVocabSize() == nil)
    }
    
    // MARK: - Edge Cases
    
    @Test("Handles empty string encoding")
    func emptyStringEncoding() {
        let mockTokenizer = MockTokenizer()
        let tokenizer = MLXLLMTokenizer(tokenizer: mockTokenizer)
        
        let encoded = tokenizer.encode("")
        #expect(encoded.isEmpty)
    }
    
    @Test("Handles empty array decoding")
    func emptyArrayDecoding() {
        let mockTokenizer = MockTokenizer()
        let tokenizer = MLXLLMTokenizer(tokenizer: mockTokenizer)
        
        let decoded = tokenizer.decode([])
        #expect(decoded == "")
    }
    
    @Test("Handles invalid token IDs in decode")
    func invalidTokenDecode() {
        let mockTokenizer = MockTokenizer()
        let tokenizer = MLXLLMTokenizer(tokenizer: mockTokenizer)
        
        let decoded = tokenizer.decode([999999, -1, 0])
        // Should skip invalid tokens
        #expect(decoded.isEmpty || decoded.count <= 1)
    }
}