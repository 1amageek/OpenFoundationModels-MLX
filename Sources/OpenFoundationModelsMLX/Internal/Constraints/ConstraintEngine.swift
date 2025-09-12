import Foundation
import MLXLMCommon
import Tokenizers

protocol ConstraintEngine: Sendable {
    var mode: ConstraintMode { get }
    
    func prepare(schema: SchemaNode?, tokenizer: any Tokenizer) async throws
    
    func softPrompt(for schema: SchemaNode?) -> String?
    
    func logitProcessors() async -> [LogitProcessor]
    
    func validator() -> (any JSONValidatorProtocol)?
}

extension ConstraintEngine {
    func softPrompt(for schema: SchemaNode?) -> String? {
        return nil
    }
    
    func logitProcessors() async -> [LogitProcessor] {
        return []
    }
    
    func validator() -> (any JSONValidatorProtocol)? {
        return nil
    }
}