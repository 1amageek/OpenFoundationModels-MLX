import Testing
import Foundation
import Tokenizers
@testable import OpenFoundationModelsMLX

@Suite("MLXLLM Tokenizer Tests")
struct MLXLLMTokenizerTests {

    // MARK: - Mock Tokenizer for Testing

    final class MockTokenizer: Tokenizer, @unchecked Sendable {
        typealias Message = [String: any Sendable]
        typealias ToolSpec = [String: any Sendable]

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
            return text.map { String($0) }
        }

        func encode(text: String) -> [Int] {
            return encode(text: text, addSpecialTokens: false)
        }

        func encode(text: String, addSpecialTokens: Bool) -> [Int] {
            if let encoded = mockEncodings[text] {
                return encoded
            }
            if text.count == 1, let ascii = text.first?.asciiValue {
                return [Int(ascii)]
            }
            return text.compactMap { Int($0.asciiValue ?? 0) }
        }

        func decode(tokens: [Int], skipSpecialTokens: Bool) -> String {
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

        func applyChatTemplate(messages: [Message]) throws -> [Int] { [] }
        func applyChatTemplate(messages: [Message], tools: [ToolSpec]?) throws -> [Int] { [] }
        func applyChatTemplate(messages: [Message], tools: [ToolSpec]?, additionalContext: [String: any Sendable]?) throws -> [Int] { [] }
        func applyChatTemplate(messages: [Message], chatTemplate: ChatTemplateArgument) throws -> [Int] { [] }
        func applyChatTemplate(messages: [Message], chatTemplate: String) throws -> [Int] { [] }
        func applyChatTemplate(
            messages: [Message],
            chatTemplate: ChatTemplateArgument?,
            addGenerationPrompt: Bool,
            truncation: Bool,
            maxLength: Int?,
            tools: [ToolSpec]?
        ) throws -> [Int] { [] }
        func applyChatTemplate(
            messages: [Message],
            chatTemplate: ChatTemplateArgument?,
            addGenerationPrompt: Bool,
            truncation: Bool,
            maxLength: Int?,
            tools: [ToolSpec]?,
            additionalContext: [String: any Sendable]?
        ) throws -> [Int] { [] }
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

    // MARK: - Tokenizer Fingerprint Tests

    @Test("Generates fingerprint")
    func fingerprintGeneration() {
        let mockTokenizer = MockTokenizer()
        let tokenizer = MLXLLMTokenizer(tokenizer: mockTokenizer)

        let fingerprint = tokenizer.fingerprint()

        #expect(!fingerprint.isEmpty)
        #expect(fingerprint.contains("mlx-tokenizer"))
        #expect(fingerprint.contains("-e2"))  // eos=2
        #expect(fingerprint.contains("-b0"))  // bos=0
    }

    // MARK: - Vocabulary Size Tests

    @Test("Returns nil for vocabulary size")
    func vocabSize() {
        let mockTokenizer = MockTokenizer()
        let tokenizer = MLXLLMTokenizer(tokenizer: mockTokenizer)

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
        #expect(decoded.isEmpty || decoded.count <= 1)
    }
}
