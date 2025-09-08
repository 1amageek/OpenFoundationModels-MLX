import Foundation
import OpenFoundationModels
import OpenFoundationModelsExtra

// Extract information needed for prompt construction from Transcript
// using strongly-typed access via OpenFoundationModelsExtra.
enum TranscriptAccess {
    struct Extracted: Sendable {
        var systemText: String?
        var messages: [ChatMessage] // ordered user/assistant history
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
                out.messages.append(.init(role: .user, content: text))
                lastPromptRF = p.responseFormat
            case .response(let r):
                let text = flattenTextSegments(r.segments)
                if !text.isEmpty { out.messages.append(.init(role: .assistant, content: text)) }
            default:
                continue
            }
        }
        if let rf = lastPromptRF, let schemaJSON = schemaJSONString(from: rf) { out.schemaJSON = schemaJSON }
        
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
        
        // Try to extract schema via JSON encoding/decoding approach
        do {
            // Create a temporary transcript with the response format
            let tempPrompt = Transcript.Prompt(
                segments: [.text(Transcript.TextSegment(content: "temp"))],
                responseFormat: responseFormat
            )
            let tempTranscript = Transcript(entries: [.prompt(tempPrompt)])
            
            // Encode to JSON and extract schema
            let jsonData = try JSONEncoder().encode(tempTranscript)
            let jsonObject = try JSONSerialization.jsonObject(with: jsonData) as? [String: Any]
            
            if let entries = jsonObject?["entries"] as? [[String: Any]],
               let firstEntry = entries.first,
               let responseFormatDict = firstEntry["responseFormat"] as? [String: Any],
               let schemaDict = responseFormatDict["schema"] as? [String: Any] {
                
                // Convert schema back to JSON string
                let schemaData = try JSONSerialization.data(withJSONObject: schemaDict)
                return String(data: schemaData, encoding: .utf8)
            }
        } catch {
            // Failed to extract schema, return nil
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
    
    /// Extract parameters JSON from tool definition via encoding/decoding
    private static func extractParametersJSON(from toolDef: Transcript.ToolDefinition) -> String? {
        do {
            // Create temporary transcript with instructions containing this tool definition
            let tempInst = Transcript.Instructions(segments: [], toolDefinitions: [toolDef])
            let tempTranscript = Transcript(entries: [.instructions(tempInst)])
            
            // Encode to JSON and extract parameters
            let jsonData = try JSONEncoder().encode(tempTranscript)
            let jsonObject = try JSONSerialization.jsonObject(with: jsonData) as? [String: Any]
            
            if let entries = jsonObject?["entries"] as? [[String: Any]],
               let firstEntry = entries.first,
               let toolDefinitions = firstEntry["toolDefinitions"] as? [[String: Any]],
               let firstTool = toolDefinitions.first,
               let parameters = firstTool["parameters"] {
                
                // Convert parameters back to JSON string
                let paramsData = try JSONSerialization.data(withJSONObject: parameters)
                return String(data: paramsData, encoding: .utf8)
            }
        } catch {
            // Failed to extract parameters, return nil
            return nil
        }
        
        return nil
    }
}
