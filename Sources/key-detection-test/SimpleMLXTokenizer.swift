import Foundation
import OpenFoundationModelsMLX

/// Simple tokenizer adapter for testing KeyDetectionLogitProcessor
public final class SimpleMLXTokenizer: TokenizerAdapter, @unchecked Sendable {
    public func encode(_ text: String) -> [Int32] {
        return text.utf8.map { Int32($0) }
    }
    
    public func decode(_ tokens: [Int32]) -> String {
        let bytes = tokens.compactMap { token -> UInt8? in
            guard token >= 0 && token <= 127 else { return nil }
            return UInt8(token)
        }
        return String(bytes: bytes, encoding: .utf8) ?? ""
    }
    
    public func getVocabSize() -> Int? {
        return 128  // ASCII range
    }
    
    public func fingerprint() -> String {
        return "simple-mlx-tokenizer-v1"
    }
    
    public var eosTokenId: Int32? { return nil }
    public var bosTokenId: Int32? { return nil }
}