import Foundation
import MLXLMCommon
import OpenFoundationModels

/// Input for ModelCard rendering.
/// ModelCard receives this data and returns a fully rendered prompt string.
public struct ModelCardInput: Sendable {
    public struct Message: Sendable {
        /// Chat role for a message.
        public enum Role: String, Sendable { case system, user, assistant, tool }
        /// Role of the message.
        public let role: Role
        /// Message content in plain text.
        public let content: String
        /// Tool name when role == .tool
        public let toolName: String?
        public init(role: Role, content: String, toolName: String? = nil) {
            self.role = role
            self.content = content
            self.toolName = toolName
        }
    }
    public struct Tool: Sendable {
        /// Tool identifier exposed to the model.
        public let name: String
        public let description: String?
        public let parametersJSON: String?
        public init(name: String, description: String?, parametersJSON: String?) {
            self.name = name
            self.description = description
            self.parametersJSON = parametersJSON
        }
    }
    /// Current date string (e.g. 2025-09-07), ISO8601 full-date.
    public let currentDate: String
    /// Optional system instructions text.
    public let system: String?
    /// Ordered conversation messages (system/user/assistant/tool).
    public let messages: [Message]
    /// Declared tool definitions from transcript.
    public let tools: [Tool]
    public init(currentDate: String, system: String?, messages: [Message], tools: [Tool]) {
        self.currentDate = currentDate
        self.system = system
        self.messages = messages
        self.tools = tools
    }
}

/// Public model card interface.
/// Cards fully own prompt rendering and default generation parameters.
public protocol ModelCard: Identifiable, Sendable where ID == String {
    /// Backend model identifier (e.g., Hugging Face repo id or local id).
    var id: String { get }
    
    /// Generate prompt from transcript and options using OpenFoundationModels' PromptBuilder
    func prompt(transcript: Transcript, options: GenerationOptions?) -> Prompt
    
    /// Default generation parameters for this model (used as-is; no fallback/merge in MLX layer).
    var params: GenerateParameters { get }
}
