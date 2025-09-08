import Foundation

enum SnapUtils {
    static func snapObject(_ obj: [String: Any], schemaKeys: [String]) -> [String: Any] {
        var out: [String: Any] = [:]
        for (k, v) in obj { out[SchemaSnapParser.snapKey(k, against: schemaKeys) ?? k] = v }
        return out
    }
}

