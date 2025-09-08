import Testing
import Foundation
import OpenFoundationModels
@testable import OpenFoundationModelsMLX

@Suite("Tool Call Detector Tests")
struct ToolCallDetectorTests {
    
    // MARK: - Basic Detection Tests
    
    @Test("Detects simple tool call")
    func simpleToolCall() throws {
        let json = """
        {
            "tool_calls": [
                {
                    "name": "get_weather",
                    "arguments": {"location": "Tokyo"}
                }
            ]
        }
        """
        
        let entry = ToolCallDetector.entryIfPresent(json)
        
        #expect(entry != nil)
        if case .toolCalls(let toolCalls) = entry {
            // ToolCalls conforms to Collection, so we can check count and iterate
            #expect(toolCalls.count > 0)
            let firstCall = toolCalls[toolCalls.startIndex]
            #expect(firstCall.toolName == "get_weather")
            
            // Access GeneratedContent properties
            let props = try firstCall.arguments.properties()
            let location = try props["location"]?.value(String.self)
            #expect(location == "Tokyo")
        } else {
            Issue.record("Expected tool calls entry")
        }
    }
    
    @Test("Detects tool call with function field")
    func toolCallWithFunction() throws {
        let json = """
        {
            "tool_calls": [
                {
                    "function": "calculate_sum",
                    "arguments": {"a": 5, "b": 10}
                }
            ]
        }
        """
        
        let entry = ToolCallDetector.entryIfPresent(json)
        
        #expect(entry != nil)
        if case .toolCalls(let toolCalls) = entry {
            #expect(toolCalls.count > 0)
            let firstCall = toolCalls[toolCalls.startIndex]
            #expect(firstCall.toolName == "calculate_sum")
            
            let props = try firstCall.arguments.properties()
            let a = try props["a"]?.value(Double.self)  // JSON numbers are Double
            let b = try props["b"]?.value(Double.self)
            #expect(a == 5)
            #expect(b == 10)
        } else {
            Issue.record("Expected tool calls entry")
        }
    }
    
    @Test("Detects tool call with parameters field")
    func toolCallWithParameters() throws {
        let json = """
        {
            "tool_calls": [
                {
                    "name": "search",
                    "parameters": {"query": "Swift programming"}
                }
            ]
        }
        """
        
        let entry = ToolCallDetector.entryIfPresent(json)
        
        #expect(entry != nil)
        if case .toolCalls(let toolCalls) = entry {
            #expect(toolCalls.count > 0)
            let firstCall = toolCalls[toolCalls.startIndex]
            #expect(firstCall.toolName == "search")
            
            let props = try firstCall.arguments.properties()
            let query = try props["query"]?.value(String.self)
            #expect(query == "Swift programming")
        } else {
            Issue.record("Expected tool calls entry")
        }
    }
    
    // MARK: - Multiple Tool Calls Tests
    
    @Test("Detects multiple tool calls")
    func multipleToolCalls() throws {
        let json = """
        {
            "tool_calls": [
                {
                    "name": "first_tool",
                    "arguments": {"arg1": "value1"}
                },
                {
                    "name": "second_tool",
                    "arguments": {"arg2": "value2"}
                }
            ]
        }
        """
        
        let entry = ToolCallDetector.entryIfPresent(json)
        
        #expect(entry != nil)
        if case .toolCalls(let toolCalls) = entry {
            #expect(toolCalls.count == 2)
            
            // Check first tool
            let firstCall = toolCalls[toolCalls.startIndex]
            #expect(firstCall.toolName == "first_tool")
            let props1 = try firstCall.arguments.properties()
            let arg1 = try props1["arg1"]?.value(String.self)
            #expect(arg1 == "value1")
            
            // Check second tool
            let secondCall = toolCalls[toolCalls.index(after: toolCalls.startIndex)]
            #expect(secondCall.toolName == "second_tool")
            let props2 = try secondCall.arguments.properties()
            let arg2 = try props2["arg2"]?.value(String.self)
            #expect(arg2 == "value2")
        } else {
            Issue.record("Expected tool calls entry")
        }
    }
    
    // MARK: - Complex JSON Tests
    
    @Test("Detects tool call with nested arguments")
    func nestedArguments() throws {
        let json = """
        {
            "tool_calls": [
                {
                    "name": "complex_tool",
                    "arguments": {
                        "user": {
                            "name": "John",
                            "age": 30
                        },
                        "settings": {
                            "enabled": true,
                            "level": 5
                        }
                    }
                }
            ]
        }
        """
        
        let entry = ToolCallDetector.entryIfPresent(json)
        
        #expect(entry != nil)
        if case .toolCalls(let toolCalls) = entry {
            #expect(toolCalls.count > 0)
            let firstCall = toolCalls[toolCalls.startIndex]
            #expect(firstCall.toolName == "complex_tool")
            
            let props = try firstCall.arguments.properties()
            
            // Access nested user object
            if let userContent = props["user"] {
                let userProps = try userContent.properties()
                let userName = try userProps["name"]?.value(String.self)
                let userAge = try userProps["age"]?.value(Double.self)
                #expect(userName == "John")
                #expect(userAge == 30)
            } else {
                Issue.record("Expected user object in arguments")
            }
            
            // Access nested settings object
            if let settingsContent = props["settings"] {
                let settingsProps = try settingsContent.properties()
                let enabled = try settingsProps["enabled"]?.value(Bool.self)
                let level = try settingsProps["level"]?.value(Double.self)
                #expect(enabled == true)
                #expect(level == 5)
            } else {
                Issue.record("Expected settings object in arguments")
            }
        } else {
            Issue.record("Expected tool calls entry")
        }
    }
    
    @Test("Detects tool call with array arguments")
    func arrayArguments() throws {
        let json = """
        {
            "tool_calls": [
                {
                    "name": "batch_process",
                    "arguments": {
                        "items": ["item1", "item2", "item3"],
                        "options": [1, 2, 3]
                    }
                }
            ]
        }
        """
        
        let entry = ToolCallDetector.entryIfPresent(json)
        
        #expect(entry != nil)
        if case .toolCalls(let toolCalls) = entry {
            #expect(toolCalls.count > 0)
            let firstCall = toolCalls[toolCalls.startIndex]
            #expect(firstCall.toolName == "batch_process")
            
            let props = try firstCall.arguments.properties()
            
            // Access items array
            if let itemsContent = props["items"] {
                let items = try itemsContent.elements()
                let itemStrings = try items.map { try $0.value(String.self) }
                #expect(itemStrings == ["item1", "item2", "item3"])
            } else {
                Issue.record("Expected items array")
            }
            
            // Access options array
            if let optionsContent = props["options"] {
                let options = try optionsContent.elements()
                let optionNumbers = try options.map { try $0.value(Double.self) }
                #expect(optionNumbers == [1, 2, 3])
            } else {
                Issue.record("Expected options array")
            }
        } else {
            Issue.record("Expected tool calls entry")
        }
    }
    
    // MARK: - Edge Cases
    
    @Test("Returns nil for non-tool-call JSON")
    func nonToolCallJSON() {
        let json = """
        {
            "message": "This is a regular message",
            "status": "ok"
        }
        """
        
        let entry = ToolCallDetector.entryIfPresent(json)
        #expect(entry == nil)
    }
    
    @Test("Returns nil for invalid JSON")
    func invalidJSON() {
        let json = "This is not JSON at all"
        
        let entry = ToolCallDetector.entryIfPresent(json)
        #expect(entry == nil)
    }
    
    @Test("Returns nil for empty tool_calls array")
    func emptyToolCallsArray() {
        let json = """
        {
            "tool_calls": []
        }
        """
        
        let entry = ToolCallDetector.entryIfPresent(json)
        #expect(entry == nil)
    }
    
    @Test("Handles tool call with missing name")
    func missingName() {
        let json = """
        {
            "tool_calls": [
                {
                    "arguments": {"arg": "value"}
                }
            ]
        }
        """
        
        let entry = ToolCallDetector.entryIfPresent(json)
        // Should return nil as name is required
        #expect(entry == nil)
    }
    
    @Test("Handles tool call with empty arguments")
    func emptyArguments() throws {
        let json = """
        {
            "tool_calls": [
                {
                    "name": "simple_tool",
                    "arguments": {}
                }
            ]
        }
        """
        
        let entry = ToolCallDetector.entryIfPresent(json)
        
        #expect(entry != nil)
        if case .toolCalls(let toolCalls) = entry {
            #expect(toolCalls.count > 0)
            let firstCall = toolCalls[toolCalls.startIndex]
            #expect(firstCall.toolName == "simple_tool")
            
            let props = try firstCall.arguments.properties()
            #expect(props.isEmpty)
        } else {
            Issue.record("Expected tool calls entry")
        }
    }
    
    // MARK: - Text Cleaning Tests
    
    @Test("Detects tool call in messy text")
    func messyText() throws {
        let json = """
        Some text before
        {"tool_calls": [{"name": "test_tool", "arguments": {"key": "value"}}]}
        Some text after
        """
        
        let entry = ToolCallDetector.entryIfPresent(json)
        
        #expect(entry != nil)
        if case .toolCalls(let toolCalls) = entry {
            #expect(toolCalls.count > 0)
            let firstCall = toolCalls[toolCalls.startIndex]
            #expect(firstCall.toolName == "test_tool")
            
            let props = try firstCall.arguments.properties()
            let key = try props["key"]?.value(String.self)
            #expect(key == "value")
        } else {
            Issue.record("Expected tool calls entry")
        }
    }
    
    @Test("Detects tool call with extra whitespace")
    func extraWhitespace() throws {
        let json = """
        {
            "tool_calls"   :   [
                {
                    "name"  :  "whitespace_tool"  ,
                    "arguments"  :  {  "arg"  :  "val"  }
                }
            ]
        }
        """
        
        let entry = ToolCallDetector.entryIfPresent(json)
        
        #expect(entry != nil)
        if case .toolCalls(let toolCalls) = entry {
            #expect(toolCalls.count > 0)
            let firstCall = toolCalls[toolCalls.startIndex]
            #expect(firstCall.toolName == "whitespace_tool")
            
            let props = try firstCall.arguments.properties()
            let arg = try props["arg"]?.value(String.self)
            #expect(arg == "val")
        } else {
            Issue.record("Expected tool calls entry")
        }
    }
    
    // MARK: - Special Characters Tests
    
    @Test("Handles escaped characters in arguments")
    func escapedCharacters() throws {
        let json = """
        {
            "tool_calls": [
                {
                    "name": "string_tool",
                    "arguments": {
                        "text": "Line 1\\nLine 2",
                        "path": "C:\\\\Users\\\\file.txt",
                        "quote": "He said \\"Hello\\""
                    }
                }
            ]
        }
        """
        
        let entry = ToolCallDetector.entryIfPresent(json)
        
        #expect(entry != nil)
        if case .toolCalls(let toolCalls) = entry {
            #expect(toolCalls.count > 0)
            let firstCall = toolCalls[toolCalls.startIndex]
            #expect(firstCall.toolName == "string_tool")
            
            let props = try firstCall.arguments.properties()
            let text = try props["text"]?.value(String.self)
            let path = try props["path"]?.value(String.self)
            let quote = try props["quote"]?.value(String.self)
            
            // JSON parsing should handle escapes
            #expect(text == "Line 1\nLine 2")
            #expect(path == "C:\\Users\\file.txt")
            #expect(quote == #"He said "Hello""#)
        } else {
            Issue.record("Expected tool calls entry")
        }
    }
    
    @Test("Handles Unicode in tool names and arguments")
    func unicodeContent() throws {
        let json = """
        {
            "tool_calls": [
                {
                    "name": "translate_text",
                    "arguments": {
                        "text": "Hello 世界 🌍",
                        "language": "日本語"
                    }
                }
            ]
        }
        """
        
        let entry = ToolCallDetector.entryIfPresent(json)
        
        #expect(entry != nil)
        if case .toolCalls(let toolCalls) = entry {
            #expect(toolCalls.count > 0)
            let firstCall = toolCalls[toolCalls.startIndex]
            #expect(firstCall.toolName == "translate_text")
            
            let props = try firstCall.arguments.properties()
            let text = try props["text"]?.value(String.self)
            let language = try props["language"]?.value(String.self)
            
            #expect(text == "Hello 世界 🌍")
            #expect(language == "日本語")
        } else {
            Issue.record("Expected tool calls entry")
        }
    }
    
    // MARK: - Performance Tests
    
    @Test("Handles large tool call efficiently")
    func largeToolCall() throws {
        var arguments: [String: Any] = [:]
        for i in 0..<100 {
            arguments["field\(i)"] = "value\(i)"
        }
        
        let argumentsJSON = try JSONSerialization.data(withJSONObject: arguments)
        let argumentsString = String(data: argumentsJSON, encoding: .utf8)!
        
        let json = """
        {
            "tool_calls": [
                {
                    "name": "large_tool",
                    "arguments": \(argumentsString)
                }
            ]
        }
        """
        
        let entry = ToolCallDetector.entryIfPresent(json)
        
        #expect(entry != nil)
        if case .toolCalls(let toolCalls) = entry {
            #expect(toolCalls.count > 0)
            let firstCall = toolCalls[toolCalls.startIndex]
            #expect(firstCall.toolName == "large_tool")
            
            let props = try firstCall.arguments.properties()
            #expect(props.count == 100)
        } else {
            Issue.record("Expected tool calls entry")
        }
    }
}