import Foundation
import OpenFoundationModels

enum ToolCallDetector {
    private static let toolCallsKeyPattern = #""tool_calls"\s*:\s*\["#
    
    private static let singleToolCallPattern = #"""
        \{\s*"(?:name|id|function|arguments|parameters)"[^{}]+?\}
        """#
    
    static func entryIfPresent(_ text: String) -> Transcript.Entry? {
        if let entry = detectWithJSONParsing(text) {
            return entry
        }
        
        return detectToolCallsWithRegex(text)
    }
    
    private static func detectWithJSONParsing(_ text: String) -> Transcript.Entry? {
        let cleaned = cleanText(text)
        
        let objects = JSONUtils.allTopLevelObjects(in: cleaned)
        
        for obj in objects {
            if let arr = obj["tool_calls"] as? [Any], !arr.isEmpty {
                return buildToolCallsEntry(from: arr)
            }
        }
        
        return nil
    }
    
    private static func detectToolCallsWithRegex(_ text: String) -> Transcript.Entry? {
        do {
            let keyRegex = try NSRegularExpression(pattern: toolCallsKeyPattern, options: [.caseInsensitive])
            let cleaned = cleanText(text)
            let range = NSRange(cleaned.startIndex..<cleaned.endIndex, in: cleaned)
            
            guard keyRegex.firstMatch(in: cleaned, options: [], range: range) != nil else {
                return nil
            }
            
            if let toolCallsRange = cleaned.range(of: "\"tool_calls\"") {
                var startIndex = cleaned.startIndex
                var openBraceCount = 0
                
                // Guard against boundary condition when tool_calls is at the start
                guard toolCallsRange.lowerBound > cleaned.startIndex else {
                    return nil
                }
                
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
                
                let jsonString = String(cleaned[startIndex..<endIndex])
                return parseToolCallsJSON(jsonString)
            }
            
            return detectIndividualToolCalls(cleaned)
            
        } catch {
            Logger.warning("[ToolCallDetector] Regex compilation failed: \(error)")
            return detectSimpleToolCalls(text)
        }
    }
    
    private static func cleanText(_ text: String) -> String {
        return text
            .replacingOccurrences(of: "\u{FEFF}", with: "")
            .replacingOccurrences(of: "\u{200B}", with: "")
            .replacingOccurrences(of: "\u{200C}", with: "")
            .replacingOccurrences(of: "\u{200D}", with: "")
    }
    
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
    
    private static func parseIndividualToolCall(_ json: String) -> Transcript.ToolCall? {
        guard let data = json.data(using: .utf8),
              let dict = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            return nil
        }
        
        guard let name = (dict["name"] as? String) ?? (dict["function"] as? String),
              !name.isEmpty else {
            return nil
        }
        
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
    
    private static func detectSimpleToolCalls(_ text: String) -> Transcript.Entry? {
        let objects = JSONUtils.allTopLevelObjects(in: text)
        
        for obj in objects {
            if let arr = obj["tool_calls"] as? [Any], !arr.isEmpty {
                return buildToolCallsEntry(from: arr)
            }
        }
        
        return nil
    }
    
    private static func parseToolCallsJSON(_ json: String) -> Transcript.Entry? {
        guard let data = json.data(using: .utf8),
              let obj = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
              let arr = obj["tool_calls"] as? [Any], !arr.isEmpty else {
            return nil
        }
        
        return buildToolCallsEntry(from: arr)
    }
    
    private static func buildToolCallsEntry(from toolCallsArray: [Any]) -> Transcript.Entry? {
        var calls: [Transcript.ToolCall] = []
        
        for item in toolCallsArray {
            guard let dict = item as? [String: Any] else { continue }
            
                guard let name = (dict["name"] as? String) ?? (dict["function"] as? String),
                  !name.isEmpty else { continue }
            
                let argsObj = dict["arguments"] ?? dict["parameters"] ?? [:]
            
            do {
                let data = try JSONSerialization.data(withJSONObject: argsObj, options: [])
                guard let json = String(data: data, encoding: .utf8) else { continue }
                
                let gen = try GeneratedContent(json: json)
                
                let callID = (dict["id"] as? String) ?? UUID().uuidString
                let call = Transcript.ToolCall(id: callID, toolName: name, arguments: gen)
                calls.append(call)
            } catch {
                continue
            }
        }
        
        guard !calls.isEmpty else { return nil }
        let toolCalls = Transcript.ToolCalls(id: UUID().uuidString, calls)
        return .toolCalls(toolCalls)
    }
}
