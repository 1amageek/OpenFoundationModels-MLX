import Foundation
import MLXLMCommon
import Tokenizers

protocol ConstraintEngine: Sendable {
    var mode: ConstraintMode { get }

    func prepare(schema: SchemaNode?, tokenizer: any Tokenizer, modelCard: (any ModelCard)?) async throws

    func softPrompt(for schema: SchemaNode?) -> String?

    func logitProcessors() async -> [LogitProcessor]
}

extension ConstraintEngine {
    func softPrompt(for schema: SchemaNode?) -> String? {
        return nil
    }

    func logitProcessors() async -> [LogitProcessor] {
        return []
    }
}