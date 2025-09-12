import Foundation
import MLXLMCommon
import Tokenizers

public final class NullConstraintEngine: ConstraintEngine, @unchecked Sendable {
    public let mode: ConstraintMode = .off
    
    public init() {}
    
    public func prepare(schema: SchemaNode?, tokenizer: any Tokenizer) async throws {
    }
    
    public func softPrompt(for schema: SchemaNode?) -> String? {
        return nil
    }
    
    public func logitProcessors() async -> [LogitProcessor] {
        return []
    }
    
    public func validator() -> (any JSONValidatorProtocol)? {
        return nil
    }
}