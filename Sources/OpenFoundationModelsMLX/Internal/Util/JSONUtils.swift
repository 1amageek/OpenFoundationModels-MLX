import Foundation

enum JSONUtils {
    // Returns the first complete top-level JSON object found in text, or nil.
    static func firstTopLevelObject(in text: String) -> [String: Any]? {
        guard let start = text.firstIndex(of: "{") else { return nil }
        var depth = 0
        var endIndex: String.Index?
        var inString = false
        var escaped = false
        for i in text[start...].indices {
            let ch = text[i]
            if ch == "\\" && inString { escaped.toggle(); continue }
            if ch == "\"" && !escaped { inString.toggle() }
            if !inString {
                if ch == "{" { depth += 1 }
                if ch == "}" { depth -= 1; if depth == 0 { endIndex = i; break } }
            }
            escaped = false
        }
        guard let end = endIndex else { return nil }
        let jsonSlice = text[start...end]
        guard let data = String(jsonSlice).data(using: .utf8) else { return nil }
        return (try? JSONSerialization.jsonObject(with: data) as? [String: Any])
    }
}
