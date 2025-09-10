import Testing
@testable import OpenFoundationModelsMLX

struct JSONValidatorTests {
    
    @Test("Validates simple valid JSON")
    func simpleValidJSON() {
        let schema = SchemaNode(
            kind: .object,
            properties: [
                "name": SchemaNode.any,
                "age": SchemaNode.any
            ],
            required: ["name"]
        )
        
        let validJSON = #"{"name": "John", "age": 30}"#
        #expect(JSONValidator.validate(text: validJSON, schema: schema))
    }
    
    @Test("Rejects JSON missing required keys")
    func missingRequiredKeys() {
        let schema = SchemaNode(
            kind: .object,
            properties: [
                "name": SchemaNode.any,
                "age": SchemaNode.any
            ],
            required: ["name"]
        )
        
        let invalidJSON = #"{"age": 30}"#
        #expect(!JSONValidator.validate(text: invalidJSON, schema: schema))
    }
    
    @Test("Handles JSON with all required keys")
    func allRequiredKeys() {
        let schema = SchemaNode(
            kind: .object,
            properties: [
                "name": SchemaNode.any,
                "age": SchemaNode.any,
                "email": SchemaNode.any
            ],
            required: ["name", "email"]
        )
        
        let validJSON = #"{"name": "John", "email": "john@example.com"}"#
        #expect(JSONValidator.validate(text: validJSON, schema: schema))
        
        let alsoValidJSON = #"{"name": "John", "age": 30, "email": "john@example.com"}"#
        #expect(JSONValidator.validate(text: alsoValidJSON, schema: schema))
    }
    
    @Test("Allows extra keys (current implementation)")
    func allowsExtraKeys() {
        let schema = SchemaNode(
            kind: .object,
            properties: [
                "name": SchemaNode.any,
                "age": SchemaNode.any
            ],
            required: []
        )
        
        let jsonWithExtra = #"{"name": "John", "age": 30, "city": "Tokyo"}"#
        #expect(JSONValidator.validate(text: jsonWithExtra, schema: schema))
        
        let jsonWithMoreExtras = #"{"name": "John", "age": 30, "city": "Tokyo", "country": "Japan"}"#
        #expect(JSONValidator.validate(text: jsonWithMoreExtras, schema: schema))
    }
    
    @Test("Validates empty JSON when no required keys")
    func emptyJSONNoRequired() {
        let schema = SchemaNode(
            kind: .object,
            properties: ["name": SchemaNode.any],
            required: []
        )
        
        let emptyJSON = #"{}"#
        #expect(JSONValidator.validate(text: emptyJSON, schema: schema))
    }
    
    @Test("Validates string type")
    func stringTypeValidation() {
        let schema = SchemaNode(
            kind: .object,
            properties: ["name": SchemaNode(kind: .string)],
            required: ["name"]
        )
        
        let validJSON = #"{"name": "John"}"#
        #expect(JSONValidator.validate(text: validJSON, schema: schema))
        
        let invalidJSON = #"{"name": 123}"#
        #expect(!JSONValidator.validate(text: invalidJSON, schema: schema))
    }
    
    @Test("Validates number type")
    func numberTypeValidation() {
        let schema = SchemaNode(
            kind: .object,
            properties: ["age": SchemaNode(kind: .number)],
            required: ["age"]
        )
        
        let validJSON = #"{"age": 30}"#
        #expect(JSONValidator.validate(text: validJSON, schema: schema))
        
        let invalidJSON = #"{"age": "thirty"}"#
        #expect(!JSONValidator.validate(text: invalidJSON, schema: schema))
    }
    
    @Test("Validates boolean type")
    func booleanTypeValidation() {
        let schema = SchemaNode(
            kind: .object,
            properties: ["active": SchemaNode(kind: .boolean)],
            required: ["active"]
        )
        
        let validJSON = #"{"active": true}"#
        #expect(JSONValidator.validate(text: validJSON, schema: schema))
        
        let invalidJSON = #"{"active": "yes"}"#
        #expect(!JSONValidator.validate(text: invalidJSON, schema: schema))
    }
    
    @Test("Validates nested objects")
    func nestedObjectValidation() {
        let addressSchema = SchemaNode(
            kind: .object,
            properties: [
                "street": SchemaNode(kind: .string),
                "city": SchemaNode(kind: .string)
            ],
            required: ["city"]
        )
        
        let schema = SchemaNode(
            kind: .object,
            properties: [
                "name": SchemaNode(kind: .string),
                "address": addressSchema
            ],
            required: ["name"]
        )
        
        let validJSON = #"{"name": "John", "address": {"street": "123 Main", "city": "NYC"}}"#
        #expect(JSONValidator.validate(text: validJSON, schema: schema))
        
        let missingRequiredInNested = #"{"name": "John", "address": {"street": "123 Main"}}"#
        #expect(!JSONValidator.validate(text: missingRequiredInNested, schema: schema))
    }
    
    @Test("Validates arrays")
    func arrayValidation() {
        let schema = SchemaNode(
            kind: .object,
            properties: [
                "tags": SchemaNode(
                    kind: .array,
                    items: SchemaNode(kind: .string)
                )
            ],
            required: []
        )
        
        let validJSON = #"{"tags": ["swift", "ios", "coding"]}"#
        #expect(JSONValidator.validate(text: validJSON, schema: schema))
        
        let invalidJSON = #"{"tags": ["swift", 123, "coding"]}"#
        #expect(!JSONValidator.validate(text: invalidJSON, schema: schema))
    }
    
    @Test("Handles malformed JSON")
    func malformedJSON() {
        let schema = SchemaNode(
            kind: .object,
            properties: ["name": SchemaNode.any],
            required: []
        )
        
        let malformedJSON = #"{"name": "John""#
        #expect(!JSONValidator.validate(text: malformedJSON, schema: schema))
        
        let invalidJSON = #"not json at all"#
        #expect(!JSONValidator.validate(text: invalidJSON, schema: schema))
    }
    
    @Test("Validates null values")
    func nullValueValidation() {
        let schema = SchemaNode(
            kind: .object,
            properties: [
                "name": SchemaNode(kind: .string),
                "age": SchemaNode.any
            ],
            required: ["name"]
        )
        
        let validJSON = #"{"name": "John", "age": null}"#
        #expect(JSONValidator.validate(text: validJSON, schema: schema))
        
        let invalidJSON = #"{"name": null, "age": 30}"#
        #expect(!JSONValidator.validate(text: invalidJSON, schema: schema))
    }
    
    @Test("Validates empty arrays")
    func emptyArrayValidation() {
        let schema = SchemaNode(
            kind: .object,
            properties: [
                "items": SchemaNode(
                    kind: .array,
                    items: SchemaNode(kind: .string)
                )
            ],
            required: []
        )
        
        let emptyArray = #"{"items": []}"#
        #expect(JSONValidator.validate(text: emptyArray, schema: schema))
    }
}