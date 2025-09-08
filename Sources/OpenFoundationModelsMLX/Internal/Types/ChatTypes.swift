import Foundation
import MLXLMCommon

// Internal chat-style types used by the MLXChatEngine.
// These types are not exported as public API; they support the internal
// core/engine design while keeping the OpenFoundationModels LanguageModel API intact.

// ChatRole and ChatMessage removed - no longer needed with prompt-based approach

enum ResponseFormatSpec: Sendable {
    case text
    case jsonSchema(schemaJSON: String)
    case jsonSchemaRef(typeName: String)
}

struct SamplingParameters: Codable, Sendable {
    var temperature: Double?
    var topP: Double?
    var topK: Int?
    var maxTokens: Int?
    var stop: [String]?
    var seed: Int?
}

struct ChatRequest: Sendable {
    let modelID: String
    let prompt: String  // The final rendered prompt from ModelCard.render
    let responseFormat: ResponseFormatSpec
    let sampling: SamplingParameters
    let schema: SchemaMeta?
    // If provided, backend should use these parameters as-is.
    let parameters: GenerateParameters?
}

struct ChatChoice: Sendable {
    let content: String  // Direct text content instead of ChatMessage
    let finishReason: String
}

struct ChatUsage: Sendable { let promptTokens: Int; let completionTokens: Int }

struct ChatResponseMeta: Sendable { }

struct ChatResponse: Sendable {
    let choices: [ChatChoice]
    let usage: ChatUsage
    let meta: ChatResponseMeta
}

struct ChatDelta: Sendable {
    let deltaText: String?
    let finishReason: String?
}

struct ChatChunk: Sendable {
    let deltas: [ChatDelta]
}
