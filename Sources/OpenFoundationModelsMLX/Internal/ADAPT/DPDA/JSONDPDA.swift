import Foundation

/// DPDA wrapper for existing JSONStateMachine
/// Provides simplified phase abstraction for DPDAKeyTrieLogitProcessor
public final class JSONDPDA: @unchecked Sendable {
    
    public enum StringKind: Sendable {
        case key
        case value
    }
    
    public enum Phase: Sendable, Equatable {
        case root
        case inObject_expectKeyOrEnd
        case inObject_expectKey
        case inObject_expectColon
        case inObject_expectValueStart
        case inObject_afterValue
        case inArray_expectValue
        case inArray_afterValue
        case inString(kind: StringKind)
        case done
        case error
    }
    
    private let stateMachine = JSONStateMachine()
    
    public init() {}
    
    public func reset() {
        stateMachine.reset()
    }
    
    public var phase: Phase {
        switch stateMachine.phase {
        case .root:
            return .root
            
        case .inObject(let objPhase):
            switch objPhase {
            case .expectKeyOrEnd:
                return .inObject_expectKeyOrEnd
            case .expectKey:
                return .inObject_expectKey
            case .expectColon:
                return .inObject_expectColon
            case .expectValueStart:
                return .inObject_expectValueStart
            case .afterValue:
                return .inObject_afterValue
            }
            
        case .inArray(let arrPhase):
            switch arrPhase {
            case .expectValue:
                return .inArray_expectValue
            case .afterValue:
                return .inArray_afterValue
            }
            
        case .inString(let strPhase):
            switch strPhase {
            case .body(kind: .key, _):
                return .inString(kind: .key)
            case .body(kind: .value, _):
                return .inString(kind: .value)
            default:
                // Handle start/end as part of body
                if case .start = strPhase {
                    return .inString(kind: .value)
                } else {
                    return .inObject_afterValue
                }
            }
            
        case .inNumber, .inLiteral:
            // These are value phases, treat as afterValue
            return .inObject_afterValue
            
        case .done:
            return .done
            
        case .error:
            return .error
        }
    }
    
    /// Process decoded text (single token) character by character
    public func advance(with decodedText: String) {
        for char in decodedText {
            stateMachine.processCharacter(char)
        }
    }
    
    public var isComplete: Bool {
        return stateMachine.isComplete()
    }
    
    public var isError: Bool {
        return stateMachine.isError()
    }
}