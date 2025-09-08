import Testing
import Foundation
@testable import OpenFoundationModelsMLX

@Suite("JSONValidator Tests")
struct JSONValidatorTests {
    
    // MARK: - Basic Validation Tests
    
    @Test("Validates simple valid JSON")
    func simpleValidJSON() {
        let validator = JSONValidator(allowExtraKeys: false, enableSnap: false)
        let schema = SchemaMeta(keys: ["name", "age"], required: ["name"])
        
        let validJSON = #"{"name": "John", "age": 30}"#
        #expect(validator.validate(text: validJSON, schema: schema))
    }
    
    @Test("Rejects JSON missing required keys")
    func missingRequiredKeys() {
        let validator = JSONValidator(allowExtraKeys: false, enableSnap: false)
        let schema = SchemaMeta(keys: ["name", "age"], required: ["name"])
        
        let invalidJSON = #"{"age": 30}"#
        #expect(!validator.validate(text: invalidJSON, schema: schema))
    }
    
    @Test("Handles JSON with all required keys")
    func allRequiredKeys() {
        let validator = JSONValidator(allowExtraKeys: false, enableSnap: false)
        let schema = SchemaMeta(keys: ["name", "age", "email"], required: ["name", "email"])
        
        let validJSON = #"{"name": "John", "email": "john@example.com"}"#
        #expect(validator.validate(text: validJSON, schema: schema))
        
        let alsoValidJSON = #"{"name": "John", "age": 30, "email": "john@example.com"}"#
        #expect(validator.validate(text: alsoValidJSON, schema: schema))
    }
    
    // MARK: - Extra Keys Tests
    
    @Test("Rejects extra keys when not allowed")
    func rejectsExtraKeys() {
        let validator = JSONValidator(allowExtraKeys: false, enableSnap: false)
        let schema = SchemaMeta(keys: ["name", "age"], required: [])
        
        let jsonWithExtra = #"{"name": "John", "age": 30, "city": "Tokyo"}"#
        #expect(!validator.validate(text: jsonWithExtra, schema: schema))
    }
    
    @Test("Allows extra keys when enabled")
    func allowsExtraKeys() {
        let validator = JSONValidator(allowExtraKeys: true, enableSnap: false)
        let schema = SchemaMeta(keys: ["name", "age"], required: [])
        
        let jsonWithExtra = #"{"name": "John", "age": 30, "city": "Tokyo", "country": "Japan"}"#
        #expect(validator.validate(text: jsonWithExtra, schema: schema))
    }
    
    @Test("Validates with only extra keys when allowed")
    func onlyExtraKeysWhenAllowed() {
        let validator = JSONValidator(allowExtraKeys: true, enableSnap: false)
        let schema = SchemaMeta(keys: ["name"], required: [])
        
        let jsonOnlyExtra = #"{"city": "Tokyo", "country": "Japan"}"#
        #expect(validator.validate(text: jsonOnlyExtra, schema: schema))
    }
    
    // MARK: - Schema Snap Tests
    
    @Test("Applies snap to fix typos")
    func appliesSnapCorrection() {
        let validator = JSONValidator(allowExtraKeys: false, enableSnap: true)
        let schema = SchemaMeta(keys: ["firstName", "lastName"], required: [])
        
        // "firstname" should be corrected to "firstName"
        let jsonWithTypo = #"{"firstname": "John", "lastName": "Doe"}"#
        #expect(validator.validate(text: jsonWithTypo, schema: schema))
    }
    
    @Test("Snap doesn't affect valid keys")
    func snapPreservesValidKeys() {
        let validator = JSONValidator(allowExtraKeys: false, enableSnap: true)
        let schema = SchemaMeta(keys: ["name", "age", "email"], required: ["name"])
        
        let validJSON = #"{"name": "John", "age": 30, "email": "john@example.com"}"#
        #expect(validator.validate(text: validJSON, schema: schema))
    }
    
    @Test("Snap with required keys")
    func snapWithRequiredKeys() {
        let validator = JSONValidator(allowExtraKeys: false, enableSnap: true)
        let schema = SchemaMeta(keys: ["firstName", "lastName"], required: ["firstName"])
        
        // "firstname" should be corrected to "firstName" to satisfy required
        let jsonWithTypo = #"{"firstname": "John"}"#
        #expect(validator.validate(text: jsonWithTypo, schema: schema))
    }
    
    // MARK: - Invalid JSON Tests
    
    @Test("Rejects malformed JSON")
    func malformedJSON() {
        let validator = JSONValidator(allowExtraKeys: false, enableSnap: false)
        let schema = SchemaMeta(keys: ["name"], required: [])
        
        let malformed = #"{"name": "John""#  // Missing closing brace
        #expect(!validator.validate(text: malformed, schema: schema))
    }
    
    @Test("Rejects non-object JSON")
    func nonObjectJSON() {
        let validator = JSONValidator(allowExtraKeys: false, enableSnap: false)
        let schema = SchemaMeta(keys: ["name"], required: [])
        
        let arrayJSON = #"["John", "Doe"]"#
        #expect(!validator.validate(text: arrayJSON, schema: schema))
        
        let stringJSON = #""Just a string""#
        #expect(!validator.validate(text: stringJSON, schema: schema))
        
        let numberJSON = "42"
        #expect(!validator.validate(text: numberJSON, schema: schema))
    }
    
    @Test("Handles empty JSON object")
    func emptyJSONObject() {
        let validator = JSONValidator(allowExtraKeys: false, enableSnap: false)
        
        // No required keys - empty object is valid
        let schema1 = SchemaMeta(keys: ["name"], required: [])
        #expect(validator.validate(text: "{}", schema: schema1))
        
        // With required keys - empty object is invalid
        let schema2 = SchemaMeta(keys: ["name"], required: ["name"])
        #expect(!validator.validate(text: "{}", schema: schema2))
    }
    
    // MARK: - Text with Extra Content Tests
    
    @Test("Extracts JSON from text with prefix")
    func jsonWithPrefix() {
        let validator = JSONValidator(allowExtraKeys: false, enableSnap: false)
        let schema = SchemaMeta(keys: ["result"], required: ["result"])
        
        let textWithJSON = #"Here is the result: {"result": "success"}"#
        #expect(validator.validate(text: textWithJSON, schema: schema))
    }
    
    @Test("Extracts JSON from text with suffix")
    func jsonWithSuffix() {
        let validator = JSONValidator(allowExtraKeys: false, enableSnap: false)
        let schema = SchemaMeta(keys: ["status"], required: ["status"])
        
        let textWithJSON = #"{"status": "complete"} - Processing finished"#
        #expect(validator.validate(text: textWithJSON, schema: schema))
    }
    
    @Test("Extracts first JSON object from multiple")
    func multipleJSONObjects() {
        let validator = JSONValidator(allowExtraKeys: false, enableSnap: false)
        let schema = SchemaMeta(keys: ["id"], required: ["id"])
        
        let multiJSON = #"{"id": "1"} Some text {"id": "2"}"#
        // Should validate using first object
        #expect(validator.validate(text: multiJSON, schema: schema))
    }
    
    // MARK: - Complex Schema Tests
    
    @Test("Validates with many keys")
    func manyKeys() {
        let validator = JSONValidator(allowExtraKeys: false, enableSnap: false)
        let keys = (1...20).map { "field\($0)" }
        let required = ["field1", "field5", "field10"]
        let schema = SchemaMeta(keys: keys, required: required)
        
        var jsonDict: [String: Any] = [:]
        for key in required {
            jsonDict[key] = "value"
        }
        
        let jsonData = try! JSONSerialization.data(withJSONObject: jsonDict)
        let jsonString = String(data: jsonData, encoding: .utf8)!
        
        #expect(validator.validate(text: jsonString, schema: schema))
    }
    
    @Test("Validates nested objects")
    func nestedObjects() {
        let validator = JSONValidator(allowExtraKeys: false, enableSnap: false)
        let schema = SchemaMeta(keys: ["user", "settings"], required: ["user"])
        
        // Note: JSONValidator works on top-level keys only
        let nestedJSON = #"{"user": {"name": "John"}, "settings": {"theme": "dark"}}"#
        #expect(validator.validate(text: nestedJSON, schema: schema))
    }
    
    // MARK: - Edge Cases
    
    @Test("Handles keys with special characters")
    func specialCharacterKeys() {
        let validator = JSONValidator(allowExtraKeys: false, enableSnap: false)
        let schema = SchemaMeta(keys: ["user-name", "email_address", "phone.number"], required: [])
        
        let jsonWithSpecialKeys = #"{"user-name": "John", "email_address": "john@example.com"}"#
        #expect(validator.validate(text: jsonWithSpecialKeys, schema: schema))
    }
    
    @Test("Handles Unicode in values")
    func unicodeValues() {
        let validator = JSONValidator(allowExtraKeys: false, enableSnap: false)
        let schema = SchemaMeta(keys: ["name", "greeting"], required: ["name"])
        
        let unicodeJSON = #"{"name": "田中", "greeting": "こんにちは 👋"}"#
        #expect(validator.validate(text: unicodeJSON, schema: schema))
    }
    
    @Test("Handles null values")
    func nullValues() {
        let validator = JSONValidator(allowExtraKeys: false, enableSnap: false)
        let schema = SchemaMeta(keys: ["name", "age"], required: ["name"])
        
        // null value for required key should fail
        let nullRequired = #"{"name": null, "age": 30}"#
        #expect(!validator.validate(text: nullRequired, schema: schema))
        
        // null value for optional key should pass
        let nullOptional = #"{"name": "John", "age": null}"#
        #expect(validator.validate(text: nullOptional, schema: schema))
    }
}
