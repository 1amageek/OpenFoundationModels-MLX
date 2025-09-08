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
    
    // Returns ALL complete top-level JSON objects found in text
    static func allTopLevelObjects(in text: String) -> [[String: Any]] {
        var objects: [[String: Any]] = []
        var searchIndex = text.startIndex
        
        while searchIndex < text.endIndex {
            // Find next opening brace
            guard let start = text[searchIndex...].firstIndex(of: "{") else { break }
            
            // Find the complete object starting from this brace
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
                    if ch == "}" { 
                        depth -= 1
                        if depth == 0 { 
                            endIndex = i
                            break 
                        } 
                    }
                }
                escaped = false
            }
            
            if let end = endIndex {
                let jsonSlice = text[start...end]
                if let data = String(jsonSlice).data(using: .utf8),
                   let obj = try? JSONSerialization.jsonObject(with: data) as? [String: Any] {
                    objects.append(obj)
                }
                // Continue searching after this object
                searchIndex = text.index(after: end)
            } else {
                // No valid object found, stop searching
                break
            }
        }
        
        return objects
    }
}
