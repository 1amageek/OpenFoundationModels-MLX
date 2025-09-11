import Foundation
@testable import OpenFoundationModelsMLX
import MLX
import MLXLLM
import Tokenizers
import Hub

/// Mock processor for testing error detection and handling
final class MockLogitProcessor: ErrorCheckable, @unchecked Sendable {
    // Control properties for testing
    var shouldTriggerFatalError = false
    var shouldTriggerNonFatalError = false
    var fatalErrorAtToken: Int? = nil
    var currentTokenCount = 0
    
    // Error to return when triggered
    var mockError: JSONGenerationError?
    private var lastError: JSONGenerationError?
    
    init() {
    }
    
    func prompt(_ prompt: MLXArray) {
        currentTokenCount = 0
        clearError()
    }
    
    func process(logits: MLXArray) -> MLXArray {
        currentTokenCount += 1
        
        // Trigger error at specific token if configured
        if let errorToken = fatalErrorAtToken, currentTokenCount >= errorToken {
            shouldTriggerFatalError = true
        }
        
        let result = logits
        if shouldTriggerFatalError {
            mockError = mockError ?? .invalidTokenSelected(token: -1, partialKey: "test", expectedTokens: [])
            lastError = mockError
        } else if shouldTriggerNonFatalError {
            mockError = mockError ?? .emptyConstraints
            lastError = mockError
        }
        
        return result
    }
    
    func didSample(token: MLXArray) {
    }
    
    func hasError() -> Bool {
        return lastError != nil
    }
    
    func hasFatalError() -> Bool {
        guard let error = lastError else { return false }
        switch error {
        case .invalidTokenSelected:
            return true
        case .emptyConstraints, .schemaViolation, .unexpectedJSONStructure, .abortedDueToError:
            return false
        }
    }
    
    func getLastError() -> JSONGenerationError? {
        return lastError
    }
    
    func clearError() {
        shouldTriggerFatalError = false
        shouldTriggerNonFatalError = false
        mockError = nil
        lastError = nil
        currentTokenCount = 0
    }
}

/// Mock tokenizer for testing TokenizerAdapter
final class MockTokenizer: TokenizerAdapter {
    // Store some common JSON tokens for realistic testing
    private let jsonTokens: [String: Int32] = [
        "{": 100,
        "}": 101,
        "[": 102,
        "]": 103,
        "\"": 104,
        ":": 105,
        ",": 106,
        "true": 107,
        "false": 108,
        "null": 109
    ]
    
    private let reverseTokens: [Int32: String]
    
    init() {
        var reverse = [Int32: String]()
        for (token, id) in jsonTokens {
            reverse[id] = token
        }
        self.reverseTokens = reverse
    }
    
    func encode(_ text: String) -> [Int32] {
        if let jsonToken = jsonTokens[text] {
            return [jsonToken]
        }
        
        return text.unicodeScalars.map { Int32($0.value % 1000) }
    }
    
    func decode(_ ids: [Int32]) -> String {
        var result = ""
        for id in ids {
            if let jsonToken = reverseTokens[id] {
                result += jsonToken
            } else {
                result += String(UnicodeScalar(Int(id) % 128) ?? UnicodeScalar(65))
            }
        }
        return result
    }
    
    func decodeToken(_ id: Int32) -> String {
        if let jsonToken = reverseTokens[id] {
            return jsonToken
        }
        return String(UnicodeScalar(Int(id) % 128) ?? UnicodeScalar(65))
    }
    
    func getVocabSize() -> Int? {
        return 1000
    }
    
    func vocabSize() -> Int? {
        return 1000
    }
    
    func eosTokenId() -> Int32? {
        return 999
    }
    
    func bosTokenId() -> Int32? {
        return 1
    }
    
    func unknownTokenId() -> Int32? {
        return 0
    }
    
    func fingerprint() -> String {
        return "MockTokenizer-v1.0-vocab\(getVocabSize() ?? 0)"
    }
}

/// Mock implementation of swift-transformers Tokenizer for testing
final class MockSwiftTokenizer: Tokenizer, @unchecked Sendable {
    let vocabSize: Int
    let bosTokenID: Int?
    let eosTokenID: Int?
    let unknownTokenID: Int
    
    // Store some common JSON tokens for realistic testing
    private let jsonTokens: [String: Int] = [
        "{": 100,
        "}": 101,
        "[": 102,
        "]": 103,
        "\"": 104,
        ":": 105,
        ",": 106,
        "true": 107,
        "false": 108,
        "null": 109,
        " ": 110,  // Space token
        "\n": 111, // Newline token
        "\t": 112  // Tab token
    ]
    
    private let reverseTokens: [Int: String]
    
    init() {
        self.vocabSize = 1000
        self.bosTokenID = 1
        self.eosTokenID = 999
        self.unknownTokenID = 0
        
        var reverse = [Int: String]()
        for (token, id) in jsonTokens {
            reverse[id] = token
        }
        self.reverseTokens = reverse
    }
    
    func tokenize(text: String) -> [String] {
        var tokens: [String] = []
        var i = text.startIndex
        
        while i < text.endIndex {
            var matched = false
            for (token, _) in jsonTokens {
                if text[i...].hasPrefix(token) {
                    tokens.append(token)
                    i = text.index(i, offsetBy: token.count)
                    matched = true
                    break
                }
            }
            
            if !matched {
                tokens.append(String(text[i]))
                i = text.index(after: i)
            }
        }
        
        return tokens
    }
    
    func encode(text: String) -> [Int] {
        let tokens = tokenize(text: text)
        return tokens.map { token in
            if let id = jsonTokens[token] {
                return id
            }
            return Int(token.first?.unicodeScalars.first?.value ?? 0) % 1000
        }
    }
    
    func encode(text: String, addSpecialTokens: Bool) -> [Int] {
        var result = encode(text: text)
        if addSpecialTokens {
            if let bos = bosTokenID {
                result.insert(bos, at: 0)
            }
            if let eos = eosTokenID {
                result.append(eos)
            }
        }
        return result
    }
    
    func callAsFunction(_ text: String, addSpecialTokens: Bool) -> [Int] {
        return encode(text: text, addSpecialTokens: addSpecialTokens)
    }
    
    func decode(tokens: [Int]) -> String {
        return decode(tokens: tokens, skipSpecialTokens: false)
    }
    
    func decode(tokens: [Int], skipSpecialTokens: Bool) -> String {
        var result = ""
        for token in tokens {
            if skipSpecialTokens {
                if token == bosTokenID || token == eosTokenID || token == unknownTokenID {
                    continue
                }
            }
            
            // Check for JSON tokens first
            if let jsonToken = reverseTokens[token] {
                result += jsonToken
            } else {
                result += String(UnicodeScalar(token % 128) ?? UnicodeScalar(65))
            }
        }
        return result
    }
    
    func convertTokenToId(_ token: String) -> Int? {
        return token.first.map { Int($0.unicodeScalars.first?.value ?? 0) % 1000 }
    }
    
    func convertTokensToIds(_ tokens: [String]) -> [Int?] {
        return tokens.map { convertTokenToId($0) }
    }
    
    func convertIdToToken(_ id: Int) -> String? {
        return String(UnicodeScalar(id % 128) ?? UnicodeScalar(65))
    }
    
    func convertIdsToTokens(_ ids: [Int]) -> [String?] {
        return ids.map { convertIdToToken($0) }
    }
    
    var bosToken: String? { "<s>" }
    var bosTokenId: Int? { bosTokenID }
    var eosToken: String? { "</s>" }
    var eosTokenId: Int? { eosTokenID }
    var unknownToken: String? { "<unk>" }
    var unknownTokenId: Int? { unknownTokenID }
    var hasChatTemplate: Bool { false }
    
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