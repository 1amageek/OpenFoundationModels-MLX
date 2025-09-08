import Testing
import Foundation
@testable import OpenFoundationModelsMLX

@Suite("JSON State Machine Tests")
struct JSONStateMachineTests {
    
    // MARK: - Phase Transitions
    
    @Test("Initial state is root")
    func initialState() {
        let machine = JSONStateMachine()
        #expect(machine.phase == .root)
        #expect(!machine.isComplete())
        #expect(!machine.isError())
    }
    
    @Test("Transitions from root to object")
    func rootToObject() {
        var machine = JSONStateMachine()
        machine.processCharacter("{")
        
        if case .inObject(let phase) = machine.phase {
            #expect(phase == .expectKeyFirstQuote)
        } else {
            Issue.record("Expected inObject phase")
        }
    }
    
    @Test("Transitions from root to array")
    func rootToArray() {
        var machine = JSONStateMachine()
        machine.processCharacter("[")
        
        if case .inArray(let phase) = machine.phase {
            #expect(phase == .expectValue)
        } else {
            Issue.record("Expected inArray phase")
        }
    }
    
    @Test("Transitions from root to string")
    func rootToString() {
        var machine = JSONStateMachine()
        machine.processCharacter("\"")
        
        if case .inString(let phase) = machine.phase,
           case .body(let kind, _) = phase {
            #expect(kind == .value)
        } else {
            Issue.record("Expected inString phase with value kind")
        }
    }
    
    @Test("Transitions from root to number")
    func rootToNumber() {
        var machine = JSONStateMachine()
        machine.processCharacter("1")
        
        if case .inNumber(let phase) = machine.phase {
            #expect(phase == .intNonZero)  // 1-9 goes to intNonZero
        } else {
            Issue.record("Expected inNumber phase")
        }
    }
    
    @Test("Transitions from root to literal")
    func rootToLiteral() {
        var machine = JSONStateMachine()
        machine.processCharacter("t")
        
        if case .inLiteral(let phase) = machine.phase {
            #expect(phase == .t1)  // First character of "true"
        } else {
            Issue.record("Expected inLiteral phase")
        }
    }
    
    // MARK: - Object State Tests
    
    @Test("Object key-value sequence")
    func objectKeyValue() {
        var machine = JSONStateMachine()
        let json = "{\"name\":\"test\"}"
        
        for char in json {
            machine.processCharacter(char)
        }
        
        #expect(machine.phase == .done)
        #expect(machine.isComplete())
    }
    
    @Test("Empty object completion")
    func emptyObject() {
        var machine = JSONStateMachine()
        machine.processCharacter("{")
        machine.processCharacter("}")
        
        #expect(machine.phase == .done)
        #expect(machine.isComplete())
    }
    
    @Test("Multiple object properties")
    func multipleProperties() {
        var machine = JSONStateMachine()
        let json = "{\"a\":1,\"b\":2}"
        
        for char in json {
            machine.processCharacter(char)
        }
        
        #expect(machine.phase == .done)
        #expect(machine.isComplete())
    }
    
    // MARK: - Array State Tests
    
    @Test("Array value sequence")
    func arrayValues() {
        var machine = JSONStateMachine()
        let json = "[1,2,3]"
        
        for char in json {
            machine.processCharacter(char)
        }
        
        #expect(machine.phase == .done)
        #expect(machine.isComplete())
    }
    
    @Test("Empty array completion")
    func emptyArray() {
        var machine = JSONStateMachine()
        machine.processCharacter("[")
        machine.processCharacter("]")
        
        #expect(machine.phase == .done)
        #expect(machine.isComplete())
    }
    
    @Test("Nested arrays")
    func nestedArrays() {
        var machine = JSONStateMachine()
        let json = "[[1,2],[3,4]]"
        
        for char in json {
            machine.processCharacter(char)
        }
        
        #expect(machine.phase == .done)
        #expect(machine.isComplete())
    }
    
    // MARK: - String State Tests
    
    @Test("Simple string value")
    func simpleString() {
        var machine = JSONStateMachine()
        let json = "\"hello\""
        
        for char in json {
            machine.processCharacter(char)
        }
        
        #expect(machine.phase == .done)
        #expect(machine.isComplete())
    }
    
    @Test("String with escape sequences")
    func escapedString() {
        var machine = JSONStateMachine()
        let json = "\"hello\\\"world\\\"\""
        
        for char in json {
            machine.processCharacter(char)
        }
        
        #expect(machine.phase == .done)
        #expect(machine.isComplete())
    }
    
    @Test("String with Unicode escape")
    func unicodeEscape() {
        var machine = JSONStateMachine()
        let json = "\"\\u0041\""  // 'A'
        
        for char in json {
            machine.processCharacter(char)
        }
        
        #expect(machine.phase == .done)
        #expect(machine.isComplete())
    }
    
    // MARK: - Number State Tests
    
    @Test("Integer number")
    func integerNumber() {
        var machine = JSONStateMachine()
        let json = "42"
        
        for char in json {
            machine.processCharacter(char)
        }
        
        // Numbers at root level stay in their number phase
        if case .inNumber(let phase) = machine.phase {
            #expect(phase == .intNonZero)
        } else {
            Issue.record("Expected inNumber phase")
        }
        // Note: sawTopValue tracking removed
    }
    
    @Test("Decimal number")
    func decimalNumber() {
        var machine = JSONStateMachine()
        let json = "3.14"
        
        for char in json {
            machine.processCharacter(char)
        }
        
        // Decimal numbers stay in their frac phase
        if case .inNumber(let phase) = machine.phase {
            #expect(phase == .frac)
        } else {
            Issue.record("Expected inNumber phase")
        }
        // Note: sawTopValue tracking removed
    }
    
    @Test("Scientific notation")
    func scientificNotation() {
        var machine = JSONStateMachine()
        let json = "1.23e-4"
        
        for char in json {
            machine.processCharacter(char)
        }
        
        // Scientific notation stays in exp phase
        if case .inNumber(let phase) = machine.phase {
            #expect(phase == .exp)
        } else {
            Issue.record("Expected inNumber phase")
        }
        // Note: sawTopValue tracking removed
    }
    
    @Test("Negative number")
    func negativeNumber() {
        var machine = JSONStateMachine()
        let json = "-42"
        
        for char in json {
            machine.processCharacter(char)
        }
        
        // Negative numbers stay in their number phase
        if case .inNumber(let phase) = machine.phase {
            #expect(phase == .intNonZero)
        } else {
            Issue.record("Expected inNumber phase")
        }
        // Note: sawTopValue tracking removed
    }
    
    // MARK: - Literal State Tests
    
    @Test("True literal")
    func trueLiteral() {
        var machine = JSONStateMachine()
        let json = "true"
        
        for char in json {
            machine.processCharacter(char)
        }
        
        #expect(machine.phase == .done)
        #expect(machine.isComplete())
    }
    
    @Test("False literal")
    func falseLiteral() {
        var machine = JSONStateMachine()
        let json = "false"
        
        for char in json {
            machine.processCharacter(char)
        }
        
        #expect(machine.phase == .done)
        #expect(machine.isComplete())
    }
    
    @Test("Null literal")
    func nullLiteral() {
        var machine = JSONStateMachine()
        let json = "null"
        
        for char in json {
            machine.processCharacter(char)
        }
        
        #expect(machine.phase == .done)
        #expect(machine.isComplete())
    }
    
    // MARK: - Error State Tests
    
    @Test("Invalid character at root")
    func invalidRootCharacter() {
        var machine = JSONStateMachine()
        machine.processCharacter("x")
        
        #expect(machine.phase == .error)
        #expect(machine.isError())
    }
    
    @Test("Invalid literal character")
    func invalidLiteral() {
        var machine = JSONStateMachine()
        machine.processCharacter("t")
        machine.processCharacter("x")  // Invalid for "true"
        
        #expect(machine.phase == .error)
        #expect(machine.isError())
    }
    
    @Test("Unclosed string")
    func unclosedString() {
        var machine = JSONStateMachine()
        machine.processCharacter("\"")
        machine.processCharacter("t")
        machine.processCharacter("e")
        machine.processCharacter("s")
        machine.processCharacter("t")
        // Missing closing quote - not an error until we try to process beyond
        
        if case .inString = machine.phase {
            #expect(Bool(true))  // Still in string state
        } else {
            Issue.record("Should still be in string state")
        }
    }
    
    // MARK: - Stack Management Tests
    
    @Test("Nested object stack depth")
    func nestedObjectStack() {
        var machine = JSONStateMachine()
        let json = "{\"a\":{\"b\":{\"c\":1}}}"
        
        for char in json {
            machine.processCharacter(char)
            
            // Check stack depth at nested points
            if char == "c" {
                // When inside a string within nested objects, stack includes the string container
                #expect(machine.stack.count == 3)  // Two nested objects + string container
            }
        }
        
        #expect(machine.phase == .done)
        #expect(machine.stack.isEmpty)
    }
    
    @Test("Mixed nesting stack")
    func mixedNestingStack() {
        var machine = JSONStateMachine()
        let json = "{\"arr\":[{\"val\":42}]}"
        
        for char in json {
            machine.processCharacter(char)
        }
        
        #expect(machine.phase == .done)
        #expect(machine.isComplete())
        #expect(machine.stack.isEmpty)
    }
    
    // MARK: - Reset Tests
    
    @Test("Reset clears state")
    func resetClearsState() {
        var machine = JSONStateMachine()
        let json = "{\"test\":"
        
        for char in json {
            machine.processCharacter(char)
        }
        
        #expect(machine.phase != .root)
        #expect(!machine.stack.isEmpty)
        
        machine.reset()
        
        #expect(machine.phase == .root)
        #expect(machine.stack.isEmpty)
        #expect(!machine.isComplete())
        #expect(!machine.isError())
    }
    
    // MARK: - Complex JSON Tests
    
    @Test("Complex nested JSON")
    func complexJSON() {
        var machine = JSONStateMachine()
        let json = """
        {
            "name": "test",
            "age": 30,
            "active": true,
            "scores": [95, 87, 92],
            "address": {
                "street": "123 Main St",
                "city": "Tokyo",
                "coordinates": [35.6762, 139.6503]
            },
            "tags": null
        }
        """
        
        for char in json {
            machine.processCharacter(char)
        }
        
        #expect(machine.phase == .done)
        #expect(machine.isComplete())
    }
    
    @Test("Early completion detection")
    func earlyCompletion() {
        var machine = JSONStateMachine()
        let json = "{\"done\":true}"
        var completedAt = -1
        
        for (index, char) in json.enumerated() {
            machine.processCharacter(char)
            if machine.isComplete() && completedAt == -1 {
                completedAt = index
            }
        }
        
        #expect(completedAt == json.count - 1)  // Should complete at the closing brace
    }
}