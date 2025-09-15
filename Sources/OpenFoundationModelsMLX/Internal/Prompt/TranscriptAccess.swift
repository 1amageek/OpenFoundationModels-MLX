import Foundation
import OpenFoundationModels
import OpenFoundationModelsExtra
import MLXLMCommon

// Extract information needed for prompt construction from Transcript
// using strongly-typed access via OpenFoundationModelsExtra.
enum TranscriptAccess {
    /// Simple message representation for internal use
    struct Message: Sendable {
        enum Role: String, Sendable { case system, user, assistant, tool }
        let role: Role
        let content: String
        let toolName: String?
    }

    struct Extracted: Sendable {
        var systemText: String?
        var messages: [Message]
        var schemaJSON: String?
        var toolDefs: [(name: String, description: String?, parametersJSON: String?)]
    }

    static func extract(from transcript: Transcript) -> Extracted {
        var out = Extracted(systemText: nil, messages: [], schemaJSON: nil, toolDefs: [])

        // 1) system (instructions)
        if let firstSystem = firstInstructions(transcript) {
            out.systemText = flattenTextSegments(firstSystem.segments)
            // tool definitions
            if let toolDefs = toolDefinitions(firstSystem) {
                out.toolDefs = toolDefs
            }
        }

        // 2) History (user/assistant) and schema (from most recent prompt)
        var lastPromptRF: Transcript.ResponseFormat? = nil
        for e in transcript {
            switch e {
            case .prompt(let p):
                let text = flattenTextSegments(p.segments)
                out.messages.append(.init(role: .user, content: text, toolName: nil))
                lastPromptRF = p.responseFormat
            case .response(let r):
                let text = flattenTextSegments(r.segments)
                if !text.isEmpty {
                    out.messages.append(.init(role: .assistant, content: text, toolName: nil))
                }
            default:
                continue
            }
        }
        if let rf = lastPromptRF, let schemaJSON = schemaJSONString(from: rf) { 
            out.schemaJSON = schemaJSON 
        }
        
        return out
    }

    // MARK: - Helpers using Transcript internals (via Extra)

    private static func firstInstructions(_ t: Transcript) -> Transcript.Instructions? {
        for e in t {
            if case .instructions(let inst) = e { return inst }
        }
        return nil
    }

    private static func flattenTextSegments(_ segments: [Transcript.Segment]) -> String {
        var pieces: [String] = []
        for s in segments {
            if case .text(let txt) = s { pieces.append(txt.content) }
        }
        return pieces.joined(separator: "\n")
    }

    private static func schemaJSONString(from responseFormat: Transcript.ResponseFormat?) -> String? {
        guard let responseFormat = responseFormat else { return nil }

        // Since schema is package-level, we need to use the Transcript encoding workaround
        // Create a minimal transcript to extract the schema
        do {
            let tempPrompt = Transcript.Prompt(
                segments: [],
                responseFormat: responseFormat
            )
            let tempTranscript = Transcript(entries: [.prompt(tempPrompt)])

            // Encode and extract schema
            let encoder = JSONEncoder()
            encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
            let jsonData = try encoder.encode(tempTranscript)
            let jsonObject = try JSONSerialization.jsonObject(with: jsonData) as? [String: Any]

            if let entries = jsonObject?["entries"] as? [[String: Any]],
               let firstEntry = entries.first,
               let responseFormatDict = firstEntry["responseFormat"] as? [String: Any],
               let schemaDict = responseFormatDict["schema"] as? [String: Any] {

                let schemaData = try JSONSerialization.data(
                    withJSONObject: schemaDict,
                    options: [.prettyPrinted, .sortedKeys]
                )
                return String(data: schemaData, encoding: .utf8)
            }
        } catch {
            return nil
        }

        return nil
    }

    private static func toolDefinitions(_ inst: Transcript.Instructions) -> [(String, String?, String?)]? {
        let defs = inst.toolDefinitions
        if defs.isEmpty { return nil }
        
        var out: [(String, String?, String?)] = []
        for d in defs {
            let name = d.name
            let desc = d.description
            let paramsJSON = extractParametersJSON(from: d)
            out.append((name, desc, paramsJSON))
        }
        return out
    }
    
    /// Extract parameters JSON from tool definition
    private static func extractParametersJSON(from toolDef: Transcript.ToolDefinition) -> String? {
        // ToolDefinition.parameters is a non-optional GenerationSchema (public access)
        let parameters = toolDef.parameters

        do {
            // Encode the GenerationSchema directly to JSON
            let encoder = JSONEncoder()
            encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
            let jsonData = try encoder.encode(parameters)
            return String(data: jsonData, encoding: .utf8)
        } catch {
            // Failed to encode parameters, return nil
            return nil
        }
    }
}