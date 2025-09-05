import Foundation
import OpenFoundationModels

enum ToolCallDetector {
    static func entryIfPresent(_ text: String) -> Transcript.Entry? {
        // Try priority detection first: JSON objects that start with "tool_calls"
        if let priorityEntry = detectPriorityToolCalls(text) {
            return priorityEntry
        }
        
        // Fallback to general JSON object detection
        return detectGeneralToolCalls(text)
    }
    
    /// Detects tool_calls from JSON that starts with {"tool_calls": ...}
    /// This has priority to avoid conflicts with other JSON fragments
    private static func detectPriorityToolCalls(_ text: String) -> Transcript.Entry? {
        let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
        
        // Check multiple priority patterns for tool_calls
        let priorityPatterns = [
            "{\"tool_calls\":",
            "{ \"tool_calls\":",
            "{\n\"tool_calls\":",
            "{\n  \"tool_calls\":",
            "{\"tool_calls\" :"
        ]
        
        guard priorityPatterns.contains(where: { trimmed.hasPrefix($0) }) else {
            return nil
        }
        
        // Find the complete JSON object that starts with tool_calls
        guard let toolCallsJSON = extractToolCallsJSON(from: trimmed) else {
            return nil
        }
        
        return parseToolCallsJSON(toolCallsJSON)
    }
    
    /// General detection for any JSON object containing tool_calls
    private static func detectGeneralToolCalls(_ text: String) -> Transcript.Entry? {
        guard let obj = JSONUtils.firstTopLevelObject(in: text) else { return nil }
        guard let arr = obj["tool_calls"] as? [Any], !arr.isEmpty else { return nil }
        
        return buildToolCallsEntry(from: arr)
    }
    
    /// Extracts complete JSON object starting with tool_calls
    private static func extractToolCallsJSON(from text: String) -> String? {
        var braceCount = 0
        var inString = false
        var escapeNext = false
        var startIndex: String.Index?
        var endIndex: String.Index?
        
        // Guard against empty or too short strings
        guard text.count >= 2 else { return nil }
        
        for (index, char) in text.enumerated() {
            let stringIndex = text.index(text.startIndex, offsetBy: index)
            
            if escapeNext {
                escapeNext = false
                continue
            }
            
            switch char {
            case "\\":
                escapeNext = inString
            case "\"":
                inString.toggle()
            case "{" where !inString:
                if braceCount == 0 {
                    startIndex = stringIndex
                }
                braceCount += 1
            case "}" where !inString:
                braceCount -= 1
                if braceCount == 0 {
                    endIndex = text.index(after: stringIndex)
                    break
                }
            default:
                break
            }
            
            // Safety check: prevent infinite loops on malformed JSON
            if braceCount > 100 {
                return nil
            }
        }
        
        guard let start = startIndex, 
              let end = endIndex,
              start < end else {
            return nil
        }
        
        let extracted = String(text[start..<end])
        
        // Basic validation: ensure it's not just braces
        guard extracted.count > 2 else {
            return nil
        }
        
        return extracted
    }
    
    /// Parses validated tool_calls JSON
    private static func parseToolCallsJSON(_ json: String) -> Transcript.Entry? {
        guard let data = json.data(using: .utf8),
              let obj = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
              let arr = obj["tool_calls"] as? [Any], !arr.isEmpty else {
            return nil
        }
        
        return buildToolCallsEntry(from: arr)
    }
    
    /// Builds Transcript.Entry from tool calls array
    private static func buildToolCallsEntry(from toolCallsArray: [Any]) -> Transcript.Entry? {
        var calls: [Transcript.ToolCall] = []
        
        for item in toolCallsArray {
            guard let dict = item as? [String: Any] else { continue }
            guard let name = dict["name"] as? String, !name.isEmpty else { continue }
            
            // Handle both "arguments" and "parameters" keys for flexibility
            let argsObj = dict["arguments"] ?? dict["parameters"] ?? [:]
            
            do {
                let data = try JSONSerialization.data(withJSONObject: argsObj, options: [])
                guard let json = String(data: data, encoding: .utf8) else { continue }
                
                let gen = try GeneratedContent(json: json)
                
                // Use provided ID or generate new one
                let callID = (dict["id"] as? String) ?? UUID().uuidString
                let call = Transcript.ToolCall(id: callID, toolName: name, arguments: gen)
                calls.append(call)
            } catch {
                // Skip malformed tool call entries but continue processing others
                continue
            }
        }
        
        guard !calls.isEmpty else { return nil }
        let toolCalls = Transcript.ToolCalls(id: UUID().uuidString, calls)
        return .toolCalls(toolCalls)
    }
}

