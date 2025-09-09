import XCTest
import OpenFoundationModels
import OpenFoundationModelsExtra
@testable import OpenFoundationModelsMLX

@Generable
struct NestedTestProfile {
    var name: String
    var age: Int
    var occupation: String
    var hobbies: [String]?
    var contact: ContactDetails?
    
    @Generable
    struct ContactDetails {
        var email: String
        var phone: String?
    }
}

class SchemaTrieIndexTest: XCTestCase {
    
    func testNestedTrieConstruction() throws {
        // Create a ResponseFormat from the nested Generable type
        let responseFormat = Transcript.ResponseFormat(type: NestedTestProfile.self)
        
        // Create a temporary transcript with this response format
        let prompt = Transcript.Prompt(
            segments: [.text(Transcript.TextSegment(content: "Generate a nested profile"))],
            responseFormat: responseFormat
        )
        let transcript = Transcript(entries: [.prompt(prompt)])
        
        // Extract schema JSON using TranscriptAccess
        let extracted = TranscriptAccess.extract(from: transcript)
        
        guard let schemaJSON = extracted.schemaJSON else {
            XCTFail("No schema JSON extracted")
            return
        }
        
        print("🔍 [SchemaTrieIndexTest] Schema JSON:")
        print(schemaJSON)
        
        // Convert to SchemaNode
        guard let data = schemaJSON.data(using: .utf8),
              let schemaDict = try JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            XCTFail("Failed to parse schema JSON")
            return
        }
        
        let schemaNode = SchemaBuilder.fromJSONSchema(schemaDict)
        print("\n🔨 [SchemaTrieIndexTest] Root SchemaNode created:")
        print("  Kind: \(schemaNode.kind)")
        print("  Keys: \(schemaNode.objectKeys)")
        print("  Required: \(schemaNode.required)")
        
        // Test contact node
        if let contactNode = schemaNode.properties["contact"] {
            print("\n📧 [SchemaTrieIndexTest] Contact SchemaNode:")
            print("  Kind: \(contactNode.kind)")
            print("  Keys: \(contactNode.objectKeys)")
            print("  Required: \(contactNode.required)")
        }
        
        // Create a mock tokenizer for testing
        let mockTokenizer = MockTokenizer()
        
        // Create SchemaTrieIndex
        print("\n🔧 [SchemaTrieIndexTest] Building SchemaTrieIndex...")
        let trieIndex = SchemaTrieIndex(root: schemaNode, tokenizer: mockTokenizer)
        
        // Check if we can get Tries for different contexts
        print("\n🌳 [SchemaTrieIndexTest] Testing Trie retrieval:")
        
        // Root Trie
        if let rootTrie = trieIndex.trie(for: schemaNode) {
            print("✅ Root Trie found with keys: \(rootTrie.allKeys)")
        } else {
            print("❌ Root Trie not found")
        }
        
        // Contact Trie
        if let contactNode = schemaNode.properties["contact"],
           let contactTrie = trieIndex.trie(for: contactNode) {
            print("✅ Contact Trie found with keys: \(contactTrie.allKeys)")
        } else {
            print("❌ Contact Trie not found")
        }
        
        // Verify the expected behavior: we should have at least 2 tries
        print("\n📊 [SchemaTrieIndexTest] Summary:")
        print("Expected: Root Trie (name, age, occupation, hobbies, contact) + Contact Trie (email, phone)")
        
        // Assert key expectations
        XCTAssertTrue(schemaNode.objectKeys.contains("contact"))
        XCTAssertTrue(schemaNode.objectKeys.contains("name"))
        
        if let contactNode = schemaNode.properties["contact"] {
            XCTAssertTrue(contactNode.objectKeys.contains("email"))
            XCTAssertTrue(contactNode.objectKeys.contains("phone"))
        } else {
            XCTFail("Contact node should exist")
        }
    }
}