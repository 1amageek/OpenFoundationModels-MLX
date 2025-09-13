import Foundation

public enum ConstraintMode: String, CaseIterable, Sendable {
    case off
    case soft
    case hard
    case post
    
    public var description: String {
        switch self {
        case .off:
            return "No constraints applied"
        case .soft:
            return "Schema included in prompt"
        case .hard:
            return "Token-level logit masking"
        case .post:
            return "Post-generation validation and repair"
        }
    }
    
    public var requiresSchema: Bool {
        switch self {
        case .off:
            return false
        case .soft, .hard, .post:
            return true
        }
    }
}