import Foundation
import OpenFoundationModels
import MLXLMCommon
@testable import OpenFoundationModelsMLX

// MARK: - Mock Tokenizer Implementations

/// Basic mock tokenizer that encodes text character by character
public final class MockTokenizer: TokenizerAdapter, @unchecked Sendable {
    private let tokenMap: [Int32: String]

    public init(tokenMap: [Int32: String] = [:]) {
        self.tokenMap = tokenMap
    }

    public func encode(_ text: String) -> [Int32] {
        return text.map { Int32($0.asciiValue ?? 0) }
    }

    public func decode(_ tokens: [Int32]) -> String {
        if let customText = tokenMap[tokens.first ?? -1] {
            return customText
        }
        return String(tokens.compactMap {
            if let scalar = UnicodeScalar(Int($0)) {
                return Character(scalar)
            }
            return nil
        })
    }

    public var eosTokenId: Int32? { 0 }
    public var bosTokenId: Int32? { 1 }
    public var unknownTokenId: Int32? { 2 }

    public func convertTokenToString(_ token: Int32) -> String? {
        if let scalar = UnicodeScalar(Int(token)) {
            return String(Character(scalar))
        }
        return nil
    }

    public func getVocabSize() -> Int? { 50000 }
    public func fingerprint() -> String { "mock-tokenizer" }
}

/// Character-based tokenizer that returns one character at a time
public final class CharacterTokenizer: TokenizerAdapter, @unchecked Sendable {
    public init() {}

    public func encode(_ text: String) -> [Int32] {
        return text.map { Int32($0.asciiValue ?? 0) }
    }

    public func decode(_ tokens: [Int32]) -> String {
        if let first = tokens.first,
           let scalar = UnicodeScalar(Int(first)) {
            return String(Character(scalar))
        }
        return ""
    }

    public var eosTokenId: Int32? { 0 }
    public var bosTokenId: Int32? { 1 }
    public var unknownTokenId: Int32? { 2 }

    public func convertTokenToString(_ token: Int32) -> String? {
        if let scalar = UnicodeScalar(Int(token)) {
            return String(Character(scalar))
        }
        return nil
    }

    public func getVocabSize() -> Int? { 50000 }
    public func fingerprint() -> String { "character-tokenizer" }
}

/// Configurable mock tokenizer for testing specific behaviors
public class ConfigurableTokenizer: TokenizerAdapter, @unchecked Sendable {
    public var nextDecodeResult: String = ""
    public var nextChar: String = ""

    public init() {}

    public func encode(_ text: String) -> [Int32] {
        return Array(repeating: Int32(0), count: text.count)
    }

    public func decode(_ tokens: [Int32]) -> String {
        return nextDecodeResult.isEmpty ? nextChar : nextDecodeResult
    }

    public var eosTokenId: Int32? { 0 }
    public var bosTokenId: Int32? { 1 }
    public var unknownTokenId: Int32? { 2 }

    public func convertTokenToString(_ token: Int32) -> String? {
        return nextDecodeResult.isEmpty ? nextChar : nextDecodeResult
    }

    public func getVocabSize() -> Int? { 50000 }
    public func fingerprint() -> String { "configurable-tokenizer" }
}


// MARK: - Common Test Schemas

public enum TestSchemas {
    /// Simple user schema
    nonisolated(unsafe) public static let userSchema: [String: Any] = [
        "type": "object",
        "properties": [
            "name": ["type": "string"],
            "email": ["type": "string"],
            "age": ["type": "integer"]
        ]
    ]

    /// Simple product schema
    nonisolated(unsafe) public static let productSchema: [String: Any] = [
        "type": "object",
        "properties": [
            "id": ["type": "integer"],
            "name": ["type": "string"],
            "price": ["type": "number"],
            "inStock": ["type": "boolean"]
        ]
    ]
}

// MARK: - Mock ModelCard

/// Simple mock ModelCard for testing
public struct MockModelCard: ModelCard {
    public let id: String

    public init(id: String = "mock-model") {
        self.id = id
    }

    public var params: GenerateParameters {
        GenerateParameters(maxTokens: 100, temperature: 0.7)
    }

    public func prompt(transcript: Transcript, options: GenerationOptions?) -> Prompt {
        return Prompt("Test prompt")
    }
}
