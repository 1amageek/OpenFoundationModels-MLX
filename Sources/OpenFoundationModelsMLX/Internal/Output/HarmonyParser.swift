import Foundation
import RegexBuilder
import OpenFoundationModelsCore

/// Parser for OpenAI Harmony format output
/// Handles channel-based output format with analysis, final, and commentary channels
struct HarmonyParser: Sendable {
    
    /// Parsed Harmony output with separated channels
    struct ParsedOutput: Sendable {
        let raw: String
        let final: String?
        let analysis: String?
        let commentary: String?
        
        /// Get the display content (final channel or fallback to raw)
        var displayContent: String {
            final ?? raw
        }
        
        /// Build metadata dictionary for non-final channels
        func metadata(includeAnalysis: Bool = false) -> [String: Any]? {
            var result: [String: Any] = [:]
            
            if includeAnalysis, let analysis = analysis {
                result["_analysis"] = analysis
            }
            
            if let commentary = commentary {
                result["_commentary"] = commentary
            }
            
            return result.isEmpty ? nil : result
        }
    }
    
    /// Parse raw Harmony format output into channels
    static func parse(_ raw: String) -> ParsedOutput {
        var channels: [String: String] = [:]
        
        // Debug logging
        Logger.info("[HarmonyParser] Parsing raw output (\(raw.count) chars)")
        if raw.count < 500 {
            Logger.info("[HarmonyParser] Raw content: \(raw)")
        }
        
        // Pattern 1: Channels WITHOUT <|start|>assistant prefix (e.g., analysis channel)
        // Using regex literal for simpler and more reliable matching
        let channelWithoutStartPattern = #/<\|channel\|>(\w+)(?:<\|constrain\|>\w+)?<\|message\|>(.*?)(?:<\|end\|>|<\|return\|>|<\|start\|>|$)/#
            .dotMatchesNewlines()
        
        // Pattern 2: Channels WITH <|start|>assistant prefix (e.g., final channel)
        // Using regex literal for simpler and more reliable matching
        let channelWithStartPattern = #/<\|start\|>assistant<\|channel\|>(\w+)(?:<\|constrain\|>\w+)?<\|message\|>(.*?)(?:<\|end\|>|<\|return\|>|<\|start\|>|$)/#
            .dotMatchesNewlines()
        
        // Also handle assistant messages without channel specification
        // Using regex literal for consistency
        let simpleAssistantPattern = #/<\|start\|>assistant(?!<\|channel\|>)(?:<\|constrain\|>\w+)?<\|message\|>(.*?)(?:<\|end\|>|<\|return\|>|<\|start\|>|$)/#
            .dotMatchesNewlines()
        
        // First, find channels WITH start marker (typically final channel)
        for match in raw.matches(of: channelWithStartPattern) {
            // match.output.1 is the channel name
            // match.output.2 is the content
            let channel = String(match.output.1)
            var content = String(match.output.2)
            
            Logger.info("[HarmonyParser] Found channel WITH start: '\(channel)' with \(content.count) chars")
            Logger.info("[HarmonyParser] Raw content for '\(channel)': \(content.prefix(100))...")
            
            // For final channel, extract just the JSON if present
            if channel == "final" {
                if let jsonContent = extractJSONFromContent(content) {
                    content = jsonContent
                }
            }
            
            channels[channel] = content.trimmingCharacters(in: .whitespacesAndNewlines)
        }
        
        // Then, find channels WITHOUT start marker (typically analysis channel)
        for match in raw.matches(of: channelWithoutStartPattern) {
            // match.output.1 is the channel name
            // match.output.2 is the content
            let channel = String(match.output.1)
            var content = String(match.output.2)
            
            // Skip if we already found this channel with start marker
            if channels[channel] != nil {
                continue
            }
            
            Logger.info("[HarmonyParser] Found channel WITHOUT start: '\(channel)' with \(content.count) chars")
            Logger.info("[HarmonyParser] Raw content for '\(channel)': \(content.prefix(100))...")
            
            // For final channel, extract just the JSON if present
            if channel == "final" {
                if let jsonContent = extractJSONFromContent(content) {
                    content = jsonContent
                }
            }
            
            channels[channel] = content.trimmingCharacters(in: .whitespacesAndNewlines)
        }
        
        // Debug logging for extracted channels
        Logger.info("[HarmonyParser] Extracted channels: \(channels.keys.sorted())")
        for (channel, content) in channels {
            Logger.info("[HarmonyParser] Channel '\(channel)': \(content.prefix(100))...")
        }
        
        // Also check for simple assistant messages without channel
        for match in raw.matches(of: simpleAssistantPattern) {
            // match.output.1 is the content (first capture group)
            let content = String(match.output.1)
            Logger.info("[HarmonyParser] Found simple assistant message with \(content.count) chars")
            channels["final"] = content.trimmingCharacters(in: .whitespacesAndNewlines)
        }
        
        // If no channels found but there's JSON, try to extract it
        if channels.isEmpty {
            if let json = extractJSON(from: raw) {
                channels["final"] = json
            }
        }
        
        let result = ParsedOutput(
            raw: raw,
            final: channels["final"],
            analysis: channels["analysis"],
            commentary: channels["commentary"]
        )
        
        Logger.info("[HarmonyParser] ParsedOutput - final: \(result.final?.prefix(50) ?? "nil"), analysis: \(result.analysis?.prefix(50) ?? "nil")")
        
        return result
    }
    
    /// Extract JSON from content by properly counting braces/brackets
    private static func extractJSONFromContent(_ content: String) -> String? {
        // Determine what type of JSON we have by finding the first JSON delimiter
        let objectStart = content.firstIndex(of: "{")
        let arrayStart = content.firstIndex(of: "[")
        
        // Check which comes first
        if let arrayIdx = arrayStart, (objectStart == nil || arrayIdx < objectStart!) {
            // Array comes first, extract array
            var bracketCount = 0
            var inString = false
            var escapeNext = false
            
            for (index, char) in content[arrayIdx...].enumerated() {
                let actualIndex = content.index(arrayIdx, offsetBy: index)
                
                if escapeNext {
                    escapeNext = false
                    continue
                }
                
                if char == "\\" {
                    escapeNext = true
                    continue
                }
                
                if char == "\"" && !escapeNext {
                    inString = !inString
                    continue
                }
                
                if !inString {
                    if char == "[" {
                        bracketCount += 1
                    } else if char == "]" {
                        bracketCount -= 1
                        if bracketCount == 0 {
                            // Found complete JSON array
                            let endIndex = content.index(actualIndex, offsetBy: 1)
                            let json = String(content[arrayIdx..<endIndex])
                            // Validate it's proper JSON
                            if let data = json.data(using: .utf8),
                               let _ = try? JSONSerialization.jsonObject(with: data) {
                                return json
                            }
                            break
                        }
                    }
                }
            }
        } else if let objectIdx = objectStart {
            // Object comes first, extract object
            var braceCount = 0
            var inString = false
            var escapeNext = false
            
            for (index, char) in content[objectIdx...].enumerated() {
                let actualIndex = content.index(objectIdx, offsetBy: index)
                
                if escapeNext {
                    escapeNext = false
                    continue
                }
                
                if char == "\\" {
                    escapeNext = true
                    continue
                }
                
                if char == "\"" && !escapeNext {
                    inString = !inString
                    continue
                }
                
                if !inString {
                    if char == "{" {
                        braceCount += 1
                    } else if char == "}" {
                        braceCount -= 1
                        if braceCount == 0 {
                            // Found complete JSON object
                            let endIndex = content.index(actualIndex, offsetBy: 1)
                            let json = String(content[objectIdx..<endIndex])
                            // Validate it's proper JSON
                            if let data = json.data(using: .utf8),
                               let _ = try? JSONSerialization.jsonObject(with: data) {
                                return json
                            }
                            break
                        }
                    }
                }
            }
        }
        
        // No JSON found with the new approach, keep the duplicate code below for backward compatibility
        // Try to extract JSON array (this is now redundant but kept for safety)
        if let startIndex = content.firstIndex(of: "[") {
            var bracketCount = 0
            var inString = false
            var escapeNext = false
            
            for (index, char) in content[startIndex...].enumerated() {
                let actualIndex = content.index(startIndex, offsetBy: index)
                
                if escapeNext {
                    escapeNext = false
                    continue
                }
                
                if char == "\\" {
                    escapeNext = true
                    continue
                }
                
                if char == "\"" && !escapeNext {
                    inString = !inString
                    continue
                }
                
                if !inString {
                    if char == "[" {
                        bracketCount += 1
                    } else if char == "]" {
                        bracketCount -= 1
                        if bracketCount == 0 {
                            // Found complete JSON array
                            let endIndex = content.index(actualIndex, offsetBy: 1)
                            let json = String(content[startIndex..<endIndex])
                            // Validate it's proper JSON
                            if let data = json.data(using: .utf8),
                               let _ = try? JSONSerialization.jsonObject(with: data) {
                                return json
                            }
                            break
                        }
                    }
                }
            }
        }
        
        return nil
    }
    
    /// Extract JSON object or array from text
    private static func extractJSON(from text: String) -> String? {
        // Simple pattern to find JSON objects
        // Note: This is a simplified approach - for production, use a proper JSON parser
        if let startIndex = text.firstIndex(of: "{"),
           let endIndex = text.lastIndex(of: "}") {
            let json = String(text[startIndex...endIndex])
            // Validate it's proper JSON
            if let data = json.data(using: .utf8),
               let _ = try? JSONSerialization.jsonObject(with: data) {
                return json
            }
        }
        
        // Try array pattern
        if let startIndex = text.firstIndex(of: "["),
           let endIndex = text.lastIndex(of: "]") {
            let json = String(text[startIndex...endIndex])
            // Validate it's proper JSON
            if let data = json.data(using: .utf8),
               let _ = try? JSONSerialization.jsonObject(with: data) {
                return json
            }
        }
        
        return nil
    }
    
    /// Stream-aware parser state for incremental parsing
    struct StreamState {
        private var buffer: String = ""
        private var inFinalChannel: Bool = false
        private var finalChannelStarted: Bool = false
        
        /// Process a new chunk and return any final channel content to stream
        mutating func processChunk(_ chunk: String) -> String? {
            buffer += chunk
            
            // Check if we've entered the final channel
            if !finalChannelStarted && buffer.contains("<|channel|>final") {
                finalChannelStarted = true
                inFinalChannel = true
                
                // Find where the message content starts
                if let messageStart = buffer.range(of: "<|message|>") {
                    // Clear buffer up to message start
                    buffer.removeSubrange(buffer.startIndex..<messageStart.upperBound)
                }
            }
            
            // If we're in final channel, stream the content
            if inFinalChannel {
                // Check for end markers
                if buffer.contains("<|end|>") || buffer.contains("<|start|>") {
                    inFinalChannel = false
                    // Extract content before end marker
                    if let endRange = buffer.range(of: "<|end|>") ?? buffer.range(of: "<|start|>") {
                        let content = String(buffer[..<endRange.lowerBound])
                        buffer.removeSubrange(..<endRange.upperBound)
                        return content
                    }
                }
                
                // Stream partial content if it's safe (not cutting in middle of special token)
                if !chunk.contains("<|") && buffer.count > 100 {
                    // Stream most of the buffer, keep last part for safety
                    let safeIndex = buffer.index(buffer.endIndex, offsetBy: -50, limitedBy: buffer.startIndex) ?? buffer.startIndex
                    let toStream = String(buffer[..<safeIndex])
                    buffer.removeSubrange(..<safeIndex)
                    return toStream.isEmpty ? nil : toStream
                }
            }
            
            return nil
        }
        
        /// Get any remaining buffered content
        mutating func flush() -> String? {
            guard !buffer.isEmpty else { return nil }
            let content = buffer
            buffer = ""
            return content
        }
    }
}