import Foundation
import PRECISE
/// Unified JSON validation used by engine/backend.
/// - Extracts the first top-level JSON object from text
/// - Optionally applies Schema Snap
/// - Checks required keys and whether extra keys are allowed
struct JSONValidator: Sendable {
    let allowExtraKeys: Bool
    let enableSnap: Bool

    func validate(text: String, schema: SchemaMeta) -> Bool {
        guard var obj = JSONUtils.firstTopLevelObject(in: text) else { return false }
        if enableSnap { obj = SnapUtils.snapObject(obj, schemaKeys: schema.keys) }
        // required - check for both missing keys and null values
        for k in schema.required { 
            guard let value = obj[k] else { return false }
            // Check if value is NSNull (JSON null)
            if value is NSNull { return false }
        }
        // extras
        if !allowExtraKeys {
            let keySet = Set(obj.keys)
            if !keySet.isSubset(of: Set(schema.keys)) { return false }
        }
        return true
    }
}
