import Foundation

// Internal chat-style types used by the MLXChatEngine.
// These types are not exported as public API; they support the internal
// core/engine design while keeping the OpenFoundationModels LanguageModel API intact.

enum ChatRole: String, Codable, Sendable {
    case system, user, assistant
}

struct ChatMessage: Codable, Sendable {
    let role: ChatRole
    let content: String
}

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

struct ChatPolicy: Codable, Sendable {
    var enableSCD: Bool = true // schema-constrained decoding
    var enableSnap: Bool = true // schema snap post-processing
    var retryMaxTries: Int = 2
}

struct ChatRequest: Sendable {
    let modelID: String
    let messages: [ChatMessage]
    let responseFormat: ResponseFormatSpec
    let sampling: SamplingParameters
    let policy: ChatPolicy
    let schema: SchemaMeta?
}

struct ChatChoice: Sendable {
    let message: ChatMessage
    let finishReason: String
}

struct ChatUsage: Sendable { let promptTokens: Int; let completionTokens: Int }

struct ChatResponseMeta: Sendable { let retries: Int }

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
    let tryIndex: Int
}

// Pre-parsed schema summary to drive constrained decoding and validation.
public struct SchemaMeta: Sendable {
    public let keys: [String]
    public let required: [String]
    
    public init(keys: [String], required: [String]) {
        self.keys = keys
        self.required = required
    }
}
