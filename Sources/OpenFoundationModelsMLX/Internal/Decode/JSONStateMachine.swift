import Foundation
import Synchronization

/// Simple JSON state machine for tracking position in JSON structure
/// Thread-safe implementation with Mutex for better performance
public final class JSONStateMachine: Sendable {
    
    public enum Phase: Sendable, Equatable {
        case root
        case inObject(ObjectPhase)
        case inArray(ArrayPhase)
        case inString(StringPhase)
        case inNumber(NumberPhase)
        case inLiteral(LiteralPhase)
        case done
        case error
    }
    
    public enum ObjectPhase: Sendable, Equatable {
        case expectKeyOrEnd      // After { - allows " or }
        case expectKey           // After , - only allows "
        case expectColon
        case expectValueStart
        case afterValue
    }
    
    public enum ArrayPhase: Sendable, Equatable {
        case expectValue
        case afterValue
    }
    
    public enum StringPhase: Sendable, Equatable {
        public enum StringKind: Sendable, Equatable {
            case key
            case value
        }
        
        case start
        case body(kind: StringKind, escaped: Bool)
        case end
    }
    
    public enum NumberPhase: Sendable, Equatable {
        case intStart
        case intZero
        case intNonZero
        case fracStart
        case frac
        case expStart
        case expSign
        case exp
    }
    
    public enum LiteralPhase: Sendable, Equatable {
        case t1, t2, t3  // 't', 'tr', 'tru'
        case f1, f2, f3, f4  // 'f', 'fa', 'fal', 'fals'
        case n1, n2, n3  // 'n', 'nu', 'nul'
        case done
    }
    
    public struct Frame: Sendable {
        let isObject: Bool
    }
    
    // State protected by Mutex
    private struct State: Sendable {
        var phase: Phase = .root
        var stack: [Frame] = []
        var depth: Int = 0
    }
    
    private let state = Mutex(State())
    
    // Public accessors
    public var phase: Phase {
        state.withLock { $0.phase }
    }
    
    public var stack: [Frame] {
        state.withLock { $0.stack }
    }
    
    public var depth: Int {
        state.withLock { $0.depth }
    }
    
    public init() {}
    
    public func reset() {
        state.withLock { state in
            state.phase = .root
            state.stack.removeAll()
            state.depth = 0
        }
    }
    
    public func isError() -> Bool {
        state.withLock { $0.phase == .error }
    }
    
    public func isComplete() -> Bool {
        state.withLock { $0.phase == .done }
    }
    
    public func processCharacter(_ char: Character) {
        state.withLock { state in
            let oldPhase = state.phase
            processCharacterInternal(char, &state)
            if oldPhase != state.phase {
                print("📊 ADAPT: Phase transition: \(oldPhase) → \(state.phase)")
            }
        }
    }
    
    private func processCharacterInternal(_ char: Character, _ state: inout State) {
        // Simplified state transitions
        switch state.phase {
        case .root:
            if char == "{" {
                state.phase = .inObject(.expectKeyOrEnd)
                state.depth += 1
                state.stack.append(Frame(isObject: true))
                print("📊 ADAPT: Context stack: push object (depth: \(state.depth))")
            } else if char == "[" {
                state.phase = .inArray(.expectValue)
                state.depth += 1
                state.stack.append(Frame(isObject: false))
                print("📊 ADAPT: Context stack: push array (depth: \(state.depth))")
            } else if char == "\"" {
                // Root-level string
                state.phase = .inString(.body(kind: .value, escaped: false))
            } else if char.isWholeNumber || char == "-" {
                // Root-level number
                state.phase = .inNumber(char == "0" ? .intZero : (char == "-" ? .intStart : .intNonZero))
            } else if char == "t" {
                // Root-level true
                state.phase = .inLiteral(.t1)
            } else if char == "f" {
                // Root-level false
                state.phase = .inLiteral(.f1)
            } else if char == "n" {
                // Root-level null
                state.phase = .inLiteral(.n1)
            } else if !char.isWhitespace {
                // Invalid character at root
                print("📊 ADAPT: Error detected at root with character: '\(char)'")
                state.phase = .error
            }
            
        case .inObject(let objPhase):
            switch objPhase {
            case .expectKeyOrEnd:
                if char == "\"" {
                    state.phase = .inString(.body(kind: .key, escaped: false))
                } else if char == "}" {
                    // Empty object is allowed after {
                    state.depth -= 1
                    _ = state.stack.popLast()
                    print("📊 ADAPT: Context stack: pop object (depth: \(state.depth))")
                    if state.stack.isEmpty {
                        state.phase = .done
                    } else if let frame = state.stack.last {
                        state.phase = frame.isObject ? .inObject(.afterValue) : .inArray(.afterValue)
                    }
                }
                
            case .expectKey:
                if char == "\"" {
                    state.phase = .inString(.body(kind: .key, escaped: false))
                } else if !char.isWhitespace {
                    // } is NOT allowed after comma - this would be a trailing comma
                    print("📊 ADAPT: Error - trailing comma detected (got '\(char)' after comma)")
                    state.phase = .error
                }
                
            case .expectColon:
                if char == ":" {
                    state.phase = .inObject(.expectValueStart)
                }
                
            case .expectValueStart:
                if char == "\"" {
                    state.phase = .inString(.body(kind: .value, escaped: false))
                } else if char == "{" {
                    state.phase = .inObject(.expectKeyOrEnd)
                    state.depth += 1
                    state.stack.append(Frame(isObject: true))
                } else if char == "[" {
                    state.phase = .inArray(.expectValue)
                    state.depth += 1
                    state.stack.append(Frame(isObject: false))
                } else if char.isWholeNumber || char == "-" {
                    state.phase = .inNumber(.intStart)
                } else if char == "t" {
                    state.phase = .inLiteral(.t1)
                } else if char == "f" {
                    state.phase = .inLiteral(.f1)
                } else if char == "n" {
                    state.phase = .inLiteral(.n1)
                }
                
            case .afterValue:
                if char == "," {
                    state.phase = .inObject(.expectKey)
                } else if char == "}" {
                    state.depth -= 1
                    _ = state.stack.popLast()
                    print("📊 ADAPT: Context stack: pop object (depth: \(state.depth))")
                    if state.stack.isEmpty {
                        state.phase = .done
                    } else if let frame = state.stack.last {
                        state.phase = frame.isObject ? .inObject(.afterValue) : .inArray(.afterValue)
                    }
                }
            }
            
        case .inString(let strPhase):
            switch strPhase {
            case .body(let kind, let escaped):
                if escaped {
                    state.phase = .inString(.body(kind: kind, escaped: false))
                } else if char == "\\" {
                    state.phase = .inString(.body(kind: kind, escaped: true))
                } else if char == "\"" {
                    // String ended
                    if kind == .key {
                        state.phase = .inObject(.expectColon)
                    } else if state.stack.isEmpty {
                        // Root-level string completed
                        state.phase = .done
                    } else if let frame = state.stack.last {
                        state.phase = frame.isObject ? .inObject(.afterValue) : .inArray(.afterValue)
                    }
                }
                
            default:
                break
            }
            
        case .inArray(let arrPhase):
            switch arrPhase {
            case .expectValue:
                if char == "]" {
                    // Empty array or end after previous value
                    state.depth -= 1
                    _ = state.stack.popLast()
                    if state.stack.isEmpty {
                        state.phase = .done
                    } else if let frame = state.stack.last {
                        state.phase = frame.isObject ? .inObject(.afterValue) : .inArray(.afterValue)
                    }
                } else if char == "\"" {
                    state.phase = .inString(.body(kind: .value, escaped: false))
                } else if char == "{" {
                    state.phase = .inObject(.expectKeyOrEnd)
                    state.depth += 1
                    state.stack.append(Frame(isObject: true))
                } else if char == "[" {
                    state.phase = .inArray(.expectValue)
                    state.depth += 1
                    state.stack.append(Frame(isObject: false))
                } else if char.isWholeNumber || char == "-" {
                    state.phase = .inNumber(char == "0" ? .intZero : (char == "-" ? .intStart : .intNonZero))
                } else if char == "t" {
                    state.phase = .inLiteral(.t1)
                } else if char == "f" {
                    state.phase = .inLiteral(.f1)
                } else if char == "n" {
                    state.phase = .inLiteral(.n1)
                }
                
            case .afterValue:
                if char == "," {
                    state.phase = .inArray(.expectValue)
                } else if char == "]" {
                    state.depth -= 1
                    _ = state.stack.popLast()
                    if state.stack.isEmpty {
                        state.phase = .done
                    } else if let frame = state.stack.last {
                        state.phase = frame.isObject ? .inObject(.afterValue) : .inArray(.afterValue)
                    }
                }
            }
            
        case .inNumber(let numPhase):
            // Handle number state transitions
            let isValueTerminator = (char == ",") || (char == "]") || (char == "}")
            
            if isValueTerminator {
                // Number ended, return to enclosing context
                if state.stack.isEmpty {
                    // Root-level number completed
                    state.phase = .done
                } else if let frame = state.stack.last {
                    state.phase = frame.isObject ? .inObject(.afterValue) : .inArray(.afterValue)
                    // Re-process the terminator in the new phase
                    processCharacterInternal(char, &state)
                }
                return
            }
            
            // Continue number parsing
            switch numPhase {
            case .intStart:
                if char == "0" {
                    state.phase = .inNumber(.intZero)
                } else if char.isWholeNumber {
                    state.phase = .inNumber(.intNonZero)
                }
            case .intZero:
                if char == "." {
                    state.phase = .inNumber(.fracStart)
                } else if char == "e" || char == "E" {
                    state.phase = .inNumber(.expStart)
                }
            case .intNonZero:
                if char.isWholeNumber {
                    // Stay in intNonZero
                    break
                } else if char == "." {
                    state.phase = .inNumber(.fracStart)
                } else if char == "e" || char == "E" {
                    state.phase = .inNumber(.expStart)
                }
            case .fracStart, .frac:
                if char.isWholeNumber {
                    state.phase = .inNumber(.frac)
                } else if char == "e" || char == "E" {
                    state.phase = .inNumber(.expStart)
                }
            case .expStart:
                if char == "+" || char == "-" {
                    state.phase = .inNumber(.expSign)
                } else if char.isWholeNumber {
                    state.phase = .inNumber(.exp)
                }
            case .expSign:
                if char.isWholeNumber {
                    state.phase = .inNumber(.exp)
                } else {
                    state.phase = .error
                }
            case .exp:
                if !char.isWholeNumber {
                    // Invalid character in exponent
                    state.phase = .error
                }
            }
            
        case .inLiteral(let litPhase):
            // Handle literal state transitions (true, false, null)
            switch litPhase {
            case .t1: state.phase = char == "r" ? .inLiteral(.t2) : .error
            case .t2: state.phase = char == "u" ? .inLiteral(.t3) : .error
            case .t3:
                if char == "e" {
                    // "true" complete
                    if state.stack.isEmpty {
                        state.phase = .done
                    } else if let frame = state.stack.last {
                        state.phase = frame.isObject ? .inObject(.afterValue) : .inArray(.afterValue)
                    }
                } else {
                    state.phase = .error
                }
            case .f1: state.phase = char == "a" ? .inLiteral(.f2) : .error
            case .f2: state.phase = char == "l" ? .inLiteral(.f3) : .error
            case .f3: state.phase = char == "s" ? .inLiteral(.f4) : .error
            case .f4:
                if char == "e" {
                    // "false" complete
                    if state.stack.isEmpty {
                        state.phase = .done
                    } else if let frame = state.stack.last {
                        state.phase = frame.isObject ? .inObject(.afterValue) : .inArray(.afterValue)
                    }
                } else {
                    state.phase = .error
                }
            case .n1: state.phase = char == "u" ? .inLiteral(.n2) : .error
            case .n2: state.phase = char == "l" ? .inLiteral(.n3) : .error
            case .n3:
                if char == "l" {
                    // "null" complete
                    if state.stack.isEmpty {
                        state.phase = .done
                    } else if let frame = state.stack.last {
                        state.phase = frame.isObject ? .inObject(.afterValue) : .inArray(.afterValue)
                    }
                } else {
                    state.phase = .error
                }
            default:
                state.phase = .error
            }
            
        case .done, .error:
            break
        }
    }
}