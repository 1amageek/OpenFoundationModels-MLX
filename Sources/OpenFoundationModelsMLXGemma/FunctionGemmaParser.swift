import Foundation

/// Parser for FunctionGemma's function call output format
///
/// FunctionGemma outputs function calls in the format:
/// `<start_function_call>call:function_name{param:<escape>value<escape>}<end_function_call>`
public enum FunctionGemmaParser {

    /// Represents a parsed function call
    public struct FunctionCall: Sendable {
        public let name: String
        public let arguments: String  // JSON string of arguments

        public init(name: String, arguments: String) {
            self.name = name
            self.arguments = arguments
        }
    }

    /// Parse a function call from raw model output
    /// - Parameter text: Raw text that may contain a function call
    /// - Returns: Parsed FunctionCall if found, nil otherwise
    public static func parseFunctionCall(_ text: String) -> FunctionCall? {
        // Look for function call markers
        guard let startRange = text.range(of: "<start_function_call>"),
              let endRange = text.range(of: "<end_function_call>") else {
            return nil
        }

        // Extract content between markers
        let content = String(text[startRange.upperBound..<endRange.lowerBound])

        // Parse the call format: call:function_name{params}
        guard content.hasPrefix("call:") else {
            return nil
        }

        let afterCall = content.dropFirst(5)  // Remove "call:"

        // Find the function name (before the first '{')
        guard let braceIndex = afterCall.firstIndex(of: "{") else {
            // No parameters - just function name
            let functionName = String(afterCall).trimmingCharacters(in: .whitespaces)
            return FunctionCall(name: functionName, arguments: "{}")
        }

        let functionName = String(afterCall[..<braceIndex]).trimmingCharacters(in: .whitespaces)

        // Extract parameters between { and }
        guard let lastBrace = afterCall.lastIndex(of: "}") else {
            return nil
        }

        let paramsContent = String(afterCall[braceIndex...lastBrace])

        // Convert from FunctionGemma format to JSON
        let jsonArguments = convertToJSON(paramsContent)

        return FunctionCall(name: functionName, arguments: jsonArguments)
    }

    /// Convert FunctionGemma parameter format to JSON
    /// Input format: `{param:<escape>value<escape>}`
    /// Output format: `{"param": "value"}`
    private static func convertToJSON(_ params: String) -> String {
        // Remove outer braces
        var content = params
        if content.hasPrefix("{") { content.removeFirst() }
        if content.hasSuffix("}") { content.removeLast() }

        // If empty, return empty object
        if content.trimmingCharacters(in: .whitespaces).isEmpty {
            return "{}"
        }

        // Parse key:value pairs, respecting <escape> markers
        let pairs = splitByCommaOutsideEscape(content)
        var jsonParts: [String] = []

        for pair in pairs {
            let trimmed = pair.trimmingCharacters(in: .whitespaces)
            guard !trimmed.isEmpty else { continue }

            // Find the colon that separates key from value
            // The colon should be before the first <escape> marker
            guard let colonIndex = findKeyValueSeparator(in: trimmed) else { continue }

            let key = String(trimmed[..<colonIndex]).trimmingCharacters(in: .whitespaces)
            var value = String(trimmed[trimmed.index(after: colonIndex)...])
                .trimmingCharacters(in: .whitespaces)

            // Remove <escape> markers
            value = value.replacingOccurrences(of: "<escape>", with: "")

            // Escape quotes in value
            value = value.replacingOccurrences(of: "\"", with: "\\\"")

            jsonParts.append("\"\(key)\": \"\(value)\"")
        }

        return "{\(jsonParts.joined(separator: ", "))}"
    }

    /// Split string by commas that are outside of <escape>...</escape> markers
    private static func splitByCommaOutsideEscape(_ content: String) -> [String] {
        var pairs: [String] = []
        var current = ""
        var insideEscape = false
        var i = content.startIndex

        while i < content.endIndex {
            // Check for <escape> marker
            if content[i...].hasPrefix("<escape>") {
                insideEscape = !insideEscape
                current += "<escape>"
                i = content.index(i, offsetBy: 8)  // Length of "<escape>"
                continue
            }

            let char = content[i]

            if char == "," && !insideEscape {
                // Found a comma outside escape - end current pair
                pairs.append(current)
                current = ""
            } else {
                current.append(char)
            }

            i = content.index(after: i)
        }

        // Don't forget the last pair
        if !current.isEmpty {
            pairs.append(current)
        }

        return pairs
    }

    /// Find the colon that separates key from value (before any <escape> marker)
    private static func findKeyValueSeparator(in text: String) -> String.Index? {
        // The key:value separator should be the first colon before <escape>
        if let escapeRange = text.range(of: "<escape>") {
            // Find colon before the escape marker
            let beforeEscape = text[..<escapeRange.lowerBound]
            return beforeEscape.lastIndex(of: ":")
        } else {
            // No escape marker, use first colon
            return text.firstIndex(of: ":")
        }
    }

    /// Check if text contains a function call
    public static func containsFunctionCall(_ text: String) -> Bool {
        return text.contains("<start_function_call>") && text.contains("<end_function_call>")
    }

    /// Check if text is starting a function call (partial)
    public static func isStartingFunctionCall(_ text: String) -> Bool {
        return text.contains("<start_function_call>") && !text.contains("<end_function_call>")
    }

    /// Extract any text before the function call
    public static func textBeforeFunctionCall(_ text: String) -> String? {
        guard let startRange = text.range(of: "<start_function_call>") else {
            return nil
        }

        let before = String(text[..<startRange.lowerBound])
            .trimmingCharacters(in: .whitespacesAndNewlines)

        return before.isEmpty ? nil : before
    }
}
