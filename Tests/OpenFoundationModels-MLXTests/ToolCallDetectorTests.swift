import Testing
@testable import OpenFoundationModelsMLX

@Suite struct ToolCallDetectorTests {
    
    // MARK: - Positive Cases
    
    @Test func detectsSimpleToolCall() throws {
        let text = """
        {"tool_calls": [{"name": "get_weather", "arguments": {"city": "Tokyo"}}]}
        """
        let entry = ToolCallDetector.entryIfPresent(text)
        #expect(entry != nil, "Should detect tool_calls")
        
        // Verify it returns a toolCalls entry type
        if case .toolCalls = entry {
            // Success - we detected tool calls
            #expect(true)
        } else {
            #expect(Bool(false), "Should return toolCalls entry type")
        }
    }
    
    @Test func detectsMultipleToolCalls() throws {
        let text = """
        {"tool_calls": [
            {"name": "get_weather", "arguments": {"city": "Tokyo"}},
            {"name": "get_time", "arguments": {"timezone": "JST"}}
        ]}
        """
        let entry = ToolCallDetector.entryIfPresent(text)
        #expect(entry != nil, "Should detect multiple tool_calls")
        
        // Verify it returns a toolCalls entry type
        if case .toolCalls = entry {
            // Success - we detected tool calls
            #expect(true)
        } else {
            #expect(Bool(false), "Should return toolCalls entry type")
        }
    }
    
    // MARK: - Negative Cases
    
    @Test func returnsNilForNoToolCalls() throws {
        let text = """
        {"result": "success", "data": {"value": 42}}
        """
        let entry = ToolCallDetector.entryIfPresent(text)
        #expect(entry == nil, "Should return nil when no tool_calls key")
    }
    
    @Test func returnsNilForEmptyToolCalls() throws {
        let text = """
        {"tool_calls": []}
        """
        let entry = ToolCallDetector.entryIfPresent(text)
        #expect(entry == nil, "Should return nil for empty tool_calls array")
    }
    
    @Test func handlesInvalidJSON() throws {
        let text = """
        {"tool_calls": [{"name": "test", invalid json here
        """
        let entry = ToolCallDetector.entryIfPresent(text)
        #expect(entry == nil, "Should return nil for invalid JSON")
    }
    
    @Test func skipsMalformedToolCallEntries() throws {
        let text = """
        {"tool_calls": [
            {"name": "valid_tool", "arguments": {"key": "value"}},
            {"missing_name": true},
            {"name": "another_valid", "arguments": {}}
        ]}
        """
        let entry = ToolCallDetector.entryIfPresent(text)
        
        // Should still detect tool calls even with some malformed entries
        #expect(entry != nil, "Should still detect valid tool calls")
        
        if case .toolCalls = entry {
            // Success - valid entries were processed
            #expect(true)
        } else {
            #expect(Bool(false), "Should return toolCalls entry type")
        }
    }
    
    @Test func handlesToolCallsWithoutArguments() throws {
        let text = """
        {"tool_calls": [{"name": "simple_tool"}]}
        """
        let entry = ToolCallDetector.entryIfPresent(text)
        
        #expect(entry != nil, "Should handle tool calls without arguments")
        
        if case .toolCalls = entry {
            // Success - tool call detected even without arguments
            #expect(true)
        } else {
            #expect(Bool(false), "Should handle tool calls without arguments")
        }
    }
    
    // MARK: - Edge Cases
    
    @Test func detectsPriorityToolCallsPattern() throws {
        // Test the priority detection for tool_calls at start
        let text = #"{"tool_calls": [{"name": "test"}]}"#
        let entry = ToolCallDetector.entryIfPresent(text)
        #expect(entry != nil, "Should detect compact JSON starting with tool_calls")
    }
    
    @Test func handlesWhitespaceVariations() throws {
        let text = """
        {  "tool_calls"  :  [  {  "name"  :  "test"  }  ]  }
        """
        let entry = ToolCallDetector.entryIfPresent(text)
        #expect(entry != nil, "Should handle extra whitespace")
    }
}

