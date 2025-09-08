import Foundation

// Schema Snap (post-processing) — normalizes and, if unambiguous, corrects JSON
// object keys to the nearest schema property name. The true JSON parser hookup
// will be implemented alongside GeneratedContent integration; this utility
// provides the matching logic.
enum SchemaSnapParser {
    static func normalize(_ s: String) -> String {
        s.lowercased().replacingOccurrences(of: "_", with: "").replacingOccurrences(of: "-", with: "")
    }

    static func snapKey(_ key: String, against schemaKeys: [String], required: [String] = []) -> String? {
        let nk = normalize(key)
        
        // Build normalized map with collision handling
        var normalizedMap: [String: String] = [:]
        for original in schemaKeys {
            let normalized = normalize(original)
            
            if let existing = normalizedMap[normalized] {
                // Collision detected - apply resolution priority
                let isNewRequired = required.contains(original)
                let isExistingRequired = required.contains(existing)
                
                // Priority: required > shorter > alphabetical
                if isNewRequired && !isExistingRequired {
                    normalizedMap[normalized] = original
                } else if !isNewRequired && isExistingRequired {
                    // Keep existing required key
                    continue
                } else if original.count < existing.count {
                    // Prefer shorter key
                    normalizedMap[normalized] = original
                } else if original.count == existing.count && original < existing {
                    // Same length - use alphabetical order for consistency
                    normalizedMap[normalized] = original
                }
                // Otherwise keep existing
            } else {
                normalizedMap[normalized] = original
            }
        }
        
        if let exact = normalizedMap[nk] { return exact }

        // Distance-1 heuristic
        var candidate: (name: String, dist: Int)?
        for k in schemaKeys {
            let d = editDistance(nk, normalize(k))
            if d <= 1 { if candidate == nil || d < candidate!.dist { candidate = (k, d) } }
        }
        return candidate?.name
    }

    private static func editDistance(_ a: String, _ b: String) -> Int {
        let a = Array(a); let b = Array(b)
        var dp = Array(repeating: Array(repeating: 0, count: b.count + 1), count: a.count + 1)
        for i in 0...a.count { dp[i][0] = i }
        for j in 0...b.count { dp[0][j] = j }
        if a.isEmpty || b.isEmpty { return max(a.count, b.count) }
        for i in 1...a.count {
            for j in 1...b.count {
                let cost = (a[i-1] == b[j-1]) ? 0 : 1
                dp[i][j] = min(dp[i-1][j] + 1, dp[i][j-1] + 1, dp[i-1][j-1] + cost)
            }
        }
        return dp[a.count][b.count]
    }
}

