import Foundation
import PRECISE
import OpenFoundationModels

enum ToolCallDetector {
    // Simpler regex pattern to find "tool_calls" key without catastrophic backtracking
    private static let toolCallsKeyPattern = #""tool_calls"\s*:\s*\["#
    
    private static let singleToolCallPattern = #"""
        \{\s*"(?:name|id|function|arguments|parameters)"[^{}]+?\}
        """#
    
    static func entryIfPresent(_ text: String) -> Transcript.Entry? {
        // Try JSON parsing first (most reliable for well-formed JSON)
        if let entry = detectWithJSONParsing(text) {
            return entry
        }
        
        // Fallback to regex-based detection for partial/malformed JSON
        return detectToolCallsWithRegex(text)
    }
    
    /// Primary detection using JSON parsing
    private static func detectWithJSONParsing(_ text: String) -> Transcript.Entry? {
        // Clean the text first
        let cleaned = cleanText(text)
        
        // Try to find JSON objects in the text
        let objects = JSONUtils.allTopLevelObjects(in: cleaned)
        
        // Check each object for tool_calls
        for obj in objects {
            if let arr = obj["tool_calls"] as? [Any], !arr.isEmpty {
                return buildToolCallsEntry(from: arr)
            }
        }
        
        return nil
    }
    
    /// Fallback regex-based detection (simplified to avoid backtracking)
    private static func detectToolCallsWithRegex(_ text: String) -> Transcript.Entry? {
        do {
            // First, just check if "tool_calls" exists
            let keyRegex = try NSRegularExpression(pattern: toolCallsKeyPattern, options: [.caseInsensitive])
            let cleaned = cleanText(text)
            let range = NSRange(cleaned.startIndex..<cleaned.endIndex, in: cleaned)
            
            guard keyRegex.firstMatch(in: cleaned, options: [], range: range) != nil else {
                return nil
            }
            
            // If tool_calls exists, try to extract the JSON object containing it
            // Use a simpler approach: find the opening { before tool_calls and matching }
            if let toolCallsRange = cleaned.range(of: "\"tool_calls\"") {
                // Find the enclosing object by counting braces
                var startIndex = cleaned.startIndex
                var openBraceCount = 0
                
                // Search backwards for opening brace
                var searchIndex = cleaned.index(before: toolCallsRange.lowerBound)
                while searchIndex >= cleaned.startIndex {
                    let char = cleaned[searchIndex]
                    if char == "{" {
                        openBraceCount += 1
                        if openBraceCount == 1 {
                            startIndex = searchIndex
                            break
                        }
                    } else if char == "}" {
                        openBraceCount -= 1
                    }
                    if searchIndex == cleaned.startIndex { break }
                    searchIndex = cleaned.index(before: searchIndex)
                }
                
                // Find matching closing brace
                openBraceCount = 0
                var endIndex = cleaned.endIndex
                searchIndex = startIndex
                while searchIndex < cleaned.endIndex {
                    let char = cleaned[searchIndex]
                    if char == "{" {
                        openBraceCount += 1
                    } else if char == "}" {
                        openBraceCount -= 1
                        if openBraceCount == 0 {
                            endIndex = cleaned.index(after: searchIndex)
                            break
                        }
                    }
                    searchIndex = cleaned.index(after: searchIndex)
                }
                
                // Extract and parse the JSON object
                let jsonString = String(cleaned[startIndex..<endIndex])
                return parseToolCallsJSON(jsonString)
            }
            
            // Final fallback: try to find individual tool calls
            return detectIndividualToolCalls(cleaned)
            
        } catch {
            // Regex compilation failed, use simple JSON detection
            Logger.warning("[ToolCallDetector] Regex compilation failed: \(error)")
            return detectSimpleToolCalls(text)
        }
    }
    
    /// Clean text by removing invisible characters and normalizing whitespace
    private static func cleanText(_ text: String) -> String {
        return text
            .replacingOccurrences(of: "\u{FEFF}", with: "")  // Remove BOM
            .replacingOccurrences(of: "\u{200B}", with: "")  // Remove zero-width space
            .replacingOccurrences(of: "\u{200C}", with: "")  // Remove zero-width non-joiner
            .replacingOccurrences(of: "\u{200D}", with: "")  // Remove zero-width joiner
    }
    
    /// Detect and reconstruct individual tool calls when full structure isn't found
    private static func detectIndividualToolCalls(_ text: String) -> Transcript.Entry? {
        do {
            let regex = try NSRegularExpression(pattern: singleToolCallPattern, options: [.caseInsensitive])
            let range = NSRange(text.startIndex..<text.endIndex, in: text)
            let matches = regex.matches(in: text, options: [], range: range)
            
            var calls: [Transcript.ToolCall] = []
            
            for match in matches {
                if let matchRange = Range(match.range, in: text) {
                    let callJSON = String(text[matchRange])
                    if let call = parseIndividualToolCall(callJSON) {
                        calls.append(call)
                    }
                }
            }
            
            guard !calls.isEmpty else { return nil }
            let toolCalls = Transcript.ToolCalls(id: UUID().uuidString, calls)
            return .toolCalls(toolCalls)
            
        } catch {
            Logger.warning("[ToolCallDetector] Individual tool call regex failed: \(error)")
            return nil
        }
    }
    
    /// Parse individual tool call JSON
    private static func parseIndividualToolCall(_ json: String) -> Transcript.ToolCall? {
        guard let data = json.data(using: .utf8),
              let dict = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            return nil
        }
        
        // Handle both "name" and "function" keys for tool name
        guard let name = (dict["name"] as? String) ?? (dict["function"] as? String),
              !name.isEmpty else {
            return nil
        }
        
        // Handle both "arguments" and "parameters" keys for flexibility
        let argsObj = dict["arguments"] ?? dict["parameters"] ?? [:]
        
        do {
            let argData = try JSONSerialization.data(withJSONObject: argsObj, options: [])
            guard let argJSON = String(data: argData, encoding: .utf8) else { return nil }
            
            let gen = try GeneratedContent(json: argJSON)
            let callID = (dict["id"] as? String) ?? UUID().uuidString
            return Transcript.ToolCall(id: callID, toolName: name, arguments: gen)
        } catch {
            Logger.warning("[ToolCallDetector] Failed to parse tool call arguments: \(error)")
            return nil
        }
    }
    
    /// Simple fallback detection method for when regex fails
    private static func detectSimpleToolCalls(_ text: String) -> Transcript.Entry? {
        // Get all top-level JSON objects using JSONUtils
        let objects = JSONUtils.allTopLevelObjects(in: text)
        
        // Check each object for tool_calls
        for obj in objects {
            if let arr = obj["tool_calls"] as? [Any], !arr.isEmpty {
                return buildToolCallsEntry(from: arr)
            }
        }
        
        return nil
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
            
            // Handle both "name" and "function" keys for tool name
            guard let name = (dict["name"] as? String) ?? (dict["function"] as? String),
                  !name.isEmpty else { continue }
            
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
