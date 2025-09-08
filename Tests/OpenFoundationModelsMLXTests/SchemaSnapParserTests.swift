import Testing
import Foundation
@testable import OpenFoundationModelsMLX

@Suite("Schema Snap Parser Tests")
struct SchemaSnapParserTests {
    
    // MARK: - Normalization Tests
    
    @Test("Normalizes keys correctly")
    func normalization() {
        #expect(SchemaSnapParser.normalize("firstName") == "firstname")
        #expect(SchemaSnapParser.normalize("first_name") == "firstname")
        #expect(SchemaSnapParser.normalize("first-name") == "firstname")
        #expect(SchemaSnapParser.normalize("FIRST_NAME") == "firstname")
        #expect(SchemaSnapParser.normalize("First-Name") == "firstname")
    }
    
    @Test("Handles empty and special strings")
    func specialStrings() {
        #expect(SchemaSnapParser.normalize("") == "")
        #expect(SchemaSnapParser.normalize("_") == "")
        #expect(SchemaSnapParser.normalize("-") == "")
        #expect(SchemaSnapParser.normalize("___") == "")
        #expect(SchemaSnapParser.normalize("a_b_c_d") == "abcd")
    }
    
    // MARK: - Collision Handling Tests
    
    @Test("Handles normalization collisions without crash")
    func collisionHandling() {
        let schemaKeys = ["first_name", "firstName", "firstname"]
        
        // Should not crash despite all normalizing to "firstname"
        let result1 = SchemaSnapParser.snapKey("first_name", against: schemaKeys)
        #expect(result1 != nil)
        
        let result2 = SchemaSnapParser.snapKey("firstName", against: schemaKeys)
        #expect(result2 != nil)
        
        let result3 = SchemaSnapParser.snapKey("firstname", against: schemaKeys)
        #expect(result3 != nil)
    }
    
    @Test("Collision resolution prefers required keys")
    func collisionPrefersRequired() {
        let schemaKeys = ["first_name", "firstName"]
        let required = ["firstName"]
        
        let result = SchemaSnapParser.snapKey("firstname", against: schemaKeys, required: required)
        #expect(result == "firstName")  // Should prefer the required key
    }
    
    @Test("Collision resolution prefers shorter keys")
    func collisionPrefersShorter() {
        let schemaKeys = ["user_name", "userName"]  // Both normalize to "username"
        let required: [String] = []  // Neither is required
        
        let result = SchemaSnapParser.snapKey("username", against: schemaKeys, required: required)
        #expect(result == "userName")  // "userName" is shorter than "user_name"
    }
    
    @Test("Collision resolution uses alphabetical order for same length")
    func collisionAlphabetical() {
        let schemaKeys = ["userB", "userA"]  // Both normalize to same, same length
        let required: [String] = []
        
        let result = SchemaSnapParser.snapKey("usera", against: schemaKeys, required: required)
        #expect(result == "userA")  // "userA" comes before "userB" alphabetically
    }
    
    // MARK: - Exact Match Tests
    
    @Test("Returns exact normalized match")
    func exactMatch() {
        let schemaKeys = ["firstName", "lastName", "email"]
        
        #expect(SchemaSnapParser.snapKey("firstname", against: schemaKeys) == "firstName")
        #expect(SchemaSnapParser.snapKey("first_name", against: schemaKeys) == "firstName")
        #expect(SchemaSnapParser.snapKey("FIRSTNAME", against: schemaKeys) == "firstName")
        #expect(SchemaSnapParser.snapKey("lastName", against: schemaKeys) == "lastName")
        #expect(SchemaSnapParser.snapKey("email", against: schemaKeys) == "email")
    }
    
    // MARK: - Edit Distance Tests
    
    @Test("Finds keys within edit distance 1")
    func editDistance1() {
        let schemaKeys = ["firstName", "lastName", "email"]
        
        // One character difference after normalization
        #expect(SchemaSnapParser.snapKey("firstnam", against: schemaKeys) == "firstName")
        #expect(SchemaSnapParser.snapKey("firsname", against: schemaKeys) == "firstName")
        #expect(SchemaSnapParser.snapKey("firstnamen", against: schemaKeys) == "firstName")
        #expect(SchemaSnapParser.snapKey("emai", against: schemaKeys) == "email")
        // "emial" has edit distance 2 from "email" (swap), so it returns nil
        #expect(SchemaSnapParser.snapKey("emial", against: schemaKeys) == nil)
    }
    
    @Test("Returns nil for keys beyond edit distance 1")
    func beyondEditDistance() {
        let schemaKeys = ["firstName", "lastName", "email"]
        
        // Too many differences
        #expect(SchemaSnapParser.snapKey("frstnam", against: schemaKeys) == nil)
        #expect(SchemaSnapParser.snapKey("username", against: schemaKeys) == nil)
        #expect(SchemaSnapParser.snapKey("phone", against: schemaKeys) == nil)
    }
    
    @Test("Chooses closest match when multiple candidates")
    func closestMatch() {
        let schemaKeys = ["name", "game", "fame"]
        
        // "nam" is distance 1 from "name" (missing 'e')
        #expect(SchemaSnapParser.snapKey("nam", against: schemaKeys) == "name")
        
        // "ame" is distance 1 from all three (missing first char)
        // Should pick the first one found with distance 1
        let result = SchemaSnapParser.snapKey("ame", against: schemaKeys)
        #expect(result == "name" || result == "game" || result == "fame")
    }
    
    // MARK: - Edge Cases
    
    @Test("Handles empty schema keys")
    func emptySchemaKeys() {
        let schemaKeys: [String] = []
        
        #expect(SchemaSnapParser.snapKey("anything", against: schemaKeys) == nil)
    }
    
    @Test("Handles single schema key")
    func singleSchemaKey() {
        let schemaKeys = ["onlyKey"]
        
        #expect(SchemaSnapParser.snapKey("onlykey", against: schemaKeys) == "onlyKey")
        #expect(SchemaSnapParser.snapKey("only_key", against: schemaKeys) == "onlyKey")
        #expect(SchemaSnapParser.snapKey("onlyke", against: schemaKeys) == "onlyKey")  // Distance 1
        #expect(SchemaSnapParser.snapKey("different", against: schemaKeys) == nil)
    }
    
    @Test("Handles very long keys")
    func longKeys() {
        let schemaKeys = ["thisIsAVeryLongPropertyNameThatShouldStillWork"]
        
        let result = SchemaSnapParser.snapKey(
            "this_is_a_very_long_property_name_that_should_still_work",
            against: schemaKeys
        )
        #expect(result == "thisIsAVeryLongPropertyNameThatShouldStillWork")
    }
    
    // MARK: - Real-World Scenarios
    
    @Test("Common JSON property variations")
    func commonVariations() {
        let schemaKeys = ["createdAt", "updatedAt", "userId", "isActive"]
        
        #expect(SchemaSnapParser.snapKey("created_at", against: schemaKeys) == "createdAt")
        #expect(SchemaSnapParser.snapKey("updated-at", against: schemaKeys) == "updatedAt")
        #expect(SchemaSnapParser.snapKey("user_id", against: schemaKeys) == "userId")
        #expect(SchemaSnapParser.snapKey("is-active", against: schemaKeys) == "isActive")
        #expect(SchemaSnapParser.snapKey("USERID", against: schemaKeys) == "userId")
    }
    
    @Test("API response key mapping")
    func apiResponseMapping() {
        let schemaKeys = ["firstName", "lastName", "emailAddress", "phoneNumber"]
        let required = ["firstName", "lastName"]
        
        // Simulating keys from various API responses
        let apiKey1 = "first_name"
        let apiKey2 = "last_name"
        let apiKey3 = "email_address"
        let apiKey4 = "phone"
        
        #expect(SchemaSnapParser.snapKey(apiKey1, against: schemaKeys, required: required) == "firstName")
        #expect(SchemaSnapParser.snapKey(apiKey2, against: schemaKeys, required: required) == "lastName")
        #expect(SchemaSnapParser.snapKey(apiKey3, against: schemaKeys, required: required) == "emailAddress")
        #expect(SchemaSnapParser.snapKey(apiKey4, against: schemaKeys, required: required) == nil)  // Too different
    }
    
    // MARK: - Performance Tests
    
    @Test("Handles large schema efficiently")
    func largeSchema() {
        // Create a large schema with potential collisions
        var schemaKeys: [String] = []
        for i in 0..<100 {
            schemaKeys.append("field\(i)")
            schemaKeys.append("field_\(i)")
            schemaKeys.append("Field\(i)")
        }
        
        // Should handle without performance issues or crashes
        let result = SchemaSnapParser.snapKey("field_50", against: schemaKeys)
        #expect(result != nil)
        
        // Test with required keys
        let required = ["field50"]
        let resultWithRequired = SchemaSnapParser.snapKey("field50", against: schemaKeys, required: required)
        #expect(resultWithRequired == "field50")
    }
}