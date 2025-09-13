import Foundation
import RegexBuilder

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
        
        // Regex to match channel blocks
        // Pattern: <|start|>assistant<|channel|>{name}...<|message|>{content}<|end|>
        let channelPattern = Regex {
            "<|start|>assistant"
            
            // Optional channel specification
            Optionally {
                "<|channel|>"
                Capture {
                    OneOrMore(.word)
                }
            }
            
            // Optional constrain specification (for response_format)
            Optionally {
                "<|constrain|>"
                OneOrMore(.word)
            }
            
            // Message marker
            "<|message|>"
            
            // Content capture (non-greedy to stop at end markers)
            Capture {
                ZeroOrMore(.reluctant) {
                    CharacterClass.any
                }
            }
            
            // End marker (could be <|end|>, <|return|>, or start of next message)
            Optionally {
                ChoiceOf {
                    "<|end|>"
                    "<|return|>"
                    Lookahead { "<|start|>" }
                }
            }
        }
        
        // Find all channel matches
        for match in raw.matches(of: channelPattern) {
            // match.output.1 is the channel name (if present)
            // match.output.2 is the content
            if let channelName = match.output.1 {
                let channel = String(channelName)
                let content = String(match.output.2)
                channels[channel] = content.trimmingCharacters(in: .whitespacesAndNewlines)
            } else {
                // No channel specified, treat as final
                let content = String(match.output.2)
                channels["final"] = content.trimmingCharacters(in: .whitespacesAndNewlines)
            }
        }
        
        // If no channels found but there's JSON, try to extract it
        if channels.isEmpty {
            if let json = extractJSON(from: raw) {
                channels["final"] = json
            }
        }
        
        return ParsedOutput(
            raw: raw,
            final: channels["final"],
            analysis: channels["analysis"],
            commentary: channels["commentary"]
        )
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