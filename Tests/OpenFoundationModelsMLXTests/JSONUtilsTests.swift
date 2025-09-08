import Testing
import Foundation
@testable import OpenFoundationModelsMLX

@Suite("JSONUtils Tests")
struct JSONUtilsTests {
    
    // MARK: - firstTopLevelObject Tests
    
    @Test("Extracts simple JSON object")
    func extractsSimpleObject() {
        let text = #"{"name": "John", "age": 30}"#
        let obj = JSONUtils.firstTopLevelObject(in: text)
        
        #expect(obj != nil)
        #expect(obj?["name"] as? String == "John")
        #expect(obj?["age"] as? Int == 30)
    }
    
    @Test("Extracts object with prefix text")
    func extractsWithPrefix() {
        let text = #"Some text before {"key": "value"} and after"#
        let obj = JSONUtils.firstTopLevelObject(in: text)
        
        #expect(obj != nil)
        #expect(obj?["key"] as? String == "value")
    }
    
    @Test("Extracts nested objects correctly")
    func extractsNestedObjects() {
        let text = #"{"outer": {"inner": "value"}, "count": 5}"#
        let obj = JSONUtils.firstTopLevelObject(in: text)
        
        #expect(obj != nil)
        #expect(obj?["count"] as? Int == 5)
        
        if let outer = obj?["outer"] as? [String: Any] {
            #expect(outer["inner"] as? String == "value")
        } else {
            Issue.record("Expected nested object")
        }
    }
    
    @Test("Handles escaped characters in strings")
    func handlesEscapedCharacters() {
        let text = #"{"quote": "He said \"Hello\"", "path": "C:\\Users\\file.txt"}"#
        let obj = JSONUtils.firstTopLevelObject(in: text)
        
        #expect(obj != nil)
        #expect(obj?["quote"] as? String == #"He said "Hello""#)
        #expect(obj?["path"] as? String == #"C:\Users\file.txt"#)
    }
    
    @Test("Handles braces in strings")
    func handlesBracesInStrings() {
        let text = #"{"text": "This has {braces} inside", "valid": true}"#
        let obj = JSONUtils.firstTopLevelObject(in: text)
        
        #expect(obj != nil)
        #expect(obj?["text"] as? String == "This has {braces} inside")
        #expect(obj?["valid"] as? Bool == true)
    }
    
    @Test("Returns nil for invalid JSON")
    func returnsNilForInvalid() {
        let invalid1 = "Not JSON at all"
        #expect(JSONUtils.firstTopLevelObject(in: invalid1) == nil)
        
        let invalid2 = #"{"unclosed": "object"#
        #expect(JSONUtils.firstTopLevelObject(in: invalid2) == nil)
        
        let invalid3 = #"{"invalid": undefined}"#
        #expect(JSONUtils.firstTopLevelObject(in: invalid3) == nil)
    }
    
    @Test("Returns nil for non-object JSON")
    func returnsNilForNonObject() {
        let array = #"[1, 2, 3]"#
        #expect(JSONUtils.firstTopLevelObject(in: array) == nil)
        
        let string = #""just a string""#
        #expect(JSONUtils.firstTopLevelObject(in: string) == nil)
        
        let number = "42"
        #expect(JSONUtils.firstTopLevelObject(in: number) == nil)
    }
    
    @Test("Handles empty object")
    func handlesEmptyObject() {
        let text = "{}"
        let obj = JSONUtils.firstTopLevelObject(in: text)
        
        #expect(obj != nil)
        #expect(obj?.isEmpty == true)
    }
    
    // MARK: - allTopLevelObjects Tests
    
    @Test("Extracts single object")
    func extractsSingleObjectInAll() {
        let text = #"{"id": 1, "name": "First"}"#
        let objects = JSONUtils.allTopLevelObjects(in: text)
        
        #expect(objects.count == 1)
        #expect(objects.first?["id"] as? Int == 1)
        #expect(objects.first?["name"] as? String == "First")
    }
    
    @Test("Extracts multiple objects")
    func extractsMultipleObjects() {
        let text = #"{"id": 1} some text {"id": 2} more text {"id": 3}"#
        let objects = JSONUtils.allTopLevelObjects(in: text)
        
        #expect(objects.count == 3)
        #expect(objects[0]["id"] as? Int == 1)
        #expect(objects[1]["id"] as? Int == 2)
        #expect(objects[2]["id"] as? Int == 3)
    }
    
    @Test("Ignores nested objects in all")
    func ignoresNestedInAll() {
        let text = #"{"outer": {"nested": 1}, "id": 1} {"id": 2}"#
        let objects = JSONUtils.allTopLevelObjects(in: text)
        
        #expect(objects.count == 2)
        #expect(objects[0]["id"] as? Int == 1)
        #expect(objects[1]["id"] as? Int == 2)
    }
    
    @Test("Handles mixed valid and invalid")
    func handlesMixedValidInvalid() {
        let text = #"{"valid": 1} {invalid json {"valid": 2}"#
        let objects = JSONUtils.allTopLevelObjects(in: text)
        
        // The algorithm will find the first valid object, then encounter the malformed
        // "{invalid json" which has no closing brace, causing it to stop searching
        #expect(objects.count == 1)
        #expect(objects[0]["valid"] as? Int == 1)
    }
    
    @Test("Returns empty array for no objects")
    func returnsEmptyForNoObjects() {
        let text = "No JSON objects here"
        let objects = JSONUtils.allTopLevelObjects(in: text)
        
        #expect(objects.isEmpty)
    }
    
    // MARK: - Complex JSON Tests
    
    @Test("Handles deeply nested structures")
    func deeplyNestedStructures() {
        let text = #"""
        {
            "level1": {
                "level2": {
                    "level3": {
                        "level4": {
                            "value": "deep"
                        }
                    }
                }
            }
        }
        """#
        
        let obj = JSONUtils.firstTopLevelObject(in: text)
        #expect(obj != nil)
        
        // Navigate through the nested structure
        if let level1 = obj?["level1"] as? [String: Any],
           let level2 = level1["level2"] as? [String: Any],
           let level3 = level2["level3"] as? [String: Any],
           let level4 = level3["level4"] as? [String: Any] {
            #expect(level4["value"] as? String == "deep")
        } else {
            Issue.record("Failed to navigate nested structure")
        }
    }
    
    @Test("Handles arrays in objects")
    func arraysInObjects() {
        let text = #"{"items": [1, 2, 3], "names": ["Alice", "Bob"]}"#
        let obj = JSONUtils.firstTopLevelObject(in: text)
        
        #expect(obj != nil)
        
        if let items = obj?["items"] as? [Int] {
            #expect(items == [1, 2, 3])
        } else {
            Issue.record("Expected items array")
        }
        
        if let names = obj?["names"] as? [String] {
            #expect(names == ["Alice", "Bob"])
        } else {
            Issue.record("Expected names array")
        }
    }
    
    @Test("Handles various data types")
    func variousDataTypes() {
        let text = #"""
        {
            "string": "text",
            "number": 42,
            "float": 3.14,
            "boolean": true,
            "null": null,
            "array": [1, 2, 3],
            "object": {"nested": "value"}
        }
        """#
        
        let obj = JSONUtils.firstTopLevelObject(in: text)
        
        #expect(obj != nil)
        #expect(obj?["string"] as? String == "text")
        #expect(obj?["number"] as? Int == 42)
        #expect(obj?["float"] as? Double == 3.14)
        #expect(obj?["boolean"] as? Bool == true)
        #expect(obj?["null"] is NSNull)
    }
    
    // MARK: - Edge Cases
    
    @Test("Handles Unicode characters")
    func unicodeCharacters() {
        let text = #"{"greeting": "こんにちは", "emoji": "👋", "name": "田中"}"#
        let obj = JSONUtils.firstTopLevelObject(in: text)
        
        #expect(obj != nil)
        #expect(obj?["greeting"] as? String == "こんにちは")
        #expect(obj?["emoji"] as? String == "👋")
        #expect(obj?["name"] as? String == "田中")
    }
    
    @Test("Handles very long strings")
    func veryLongStrings() {
        let longValue = String(repeating: "a", count: 10000)
        let text = #"{"long": "\#(longValue)"}"#
        let obj = JSONUtils.firstTopLevelObject(in: text)
        
        #expect(obj != nil)
        #expect(obj?["long"] as? String == longValue)
    }
    
    @Test("Handles whitespace variations")
    func whitespaceVariations() {
        let text = """
        {
            "key1"  :   "value1"  ,
            "key2":
            
            "value2"
        }
        """
        
        let obj = JSONUtils.firstTopLevelObject(in: text)
        
        #expect(obj != nil)
        #expect(obj?["key1"] as? String == "value1")
        #expect(obj?["key2"] as? String == "value2")
    }
    
    @Test("Handles single quotes in values")
    func singleQuotesInValues() {
        let text = #"{"text": "It's a test", "quote": "She said 'hello'"}"#
        let obj = JSONUtils.firstTopLevelObject(in: text)
        
        #expect(obj != nil)
        #expect(obj?["text"] as? String == "It's a test")
        #expect(obj?["quote"] as? String == "She said 'hello'")
    }
    
    @Test("Performance with many objects")
    func performanceWithManyObjects() {
        var text = ""
        for i in 0..<100 {
            text += #"{"id": \#(i)} "#
        }
        
        let objects = JSONUtils.allTopLevelObjects(in: text)
        
        #expect(objects.count == 100)
        #expect(objects[0]["id"] as? Int == 0)
        #expect(objects[99]["id"] as? Int == 99)
    }
}