import XCTest
import OpenFoundationModels
import OpenFoundationModelsExtra
@testable import OpenFoundationModelsMLX

@Generable
struct UserProfile {
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

class SchemaOutputTest: XCTestCase {
    
    func testSchemaGeneration() throws {
        // Create a ResponseFormat from the Generable type
        let responseFormat = Transcript.ResponseFormat(type: UserProfile.self)
        
        // Create a temporary transcript with this response format
        let prompt = Transcript.Prompt(
            segments: [.text(Transcript.TextSegment(content: "Generate a person profile"))],
            responseFormat: responseFormat
        )
        let transcript = Transcript(entries: [.prompt(prompt)])
        
        // Extract schema JSON using TranscriptAccess
        let extracted = TranscriptAccess.extract(from: transcript)
        
        print("🔍 Extracted Schema JSON:")
        if let schemaJSON = extracted.schemaJSON {
            print(schemaJSON)
            
            // Parse the JSON to verify structure
            guard let data = schemaJSON.data(using: .utf8),
                  let schemaDict = try JSONSerialization.jsonObject(with: data) as? [String: Any] else {
                XCTFail("Failed to parse schema JSON")
                return
            }
            
            print("\n📊 Schema Structure Analysis:")
            print("Root type: \(schemaDict["type"] ?? "unknown")")
            
            if let properties = schemaDict["properties"] as? [String: Any] {
                print("Root properties: \(Array(properties.keys).sorted())")
                
                // Check hobbies structure
                if let hobbies = properties["hobbies"] as? [String: Any] {
                    print("Hobbies type: \(hobbies["type"] ?? "unknown")")
                    if let items = hobbies["items"] {
                        print("Hobbies items: \(items)")
                    }
                }
                
                // Check contact structure
                if let contact = properties["contact"] as? [String: Any] {
                    print("Contact type: \(contact["type"] ?? "unknown")")
                    if let contactProperties = contact["properties"] as? [String: Any] {
                        print("Contact properties: \(Array(contactProperties.keys).sorted())")
                        
                        // Detailed analysis of nested properties
                        for (key, value) in contactProperties {
                            if let propDict = value as? [String: Any] {
                                print("  \(key): \(propDict["type"] ?? "unknown")")
                            }
                        }
                    } else {
                        print("❌ Contact is missing 'properties' field!")
                    }
                }
            }
            
            // Test SchemaBuilder conversion
            print("\n🔨 Testing SchemaBuilder conversion:")
            let schemaNode = SchemaBuilder.fromJSONSchema(schemaDict)
            print("SchemaNode kind: \(schemaNode.kind)")
            print("SchemaNode keys: \(schemaNode.objectKeys)")
            
            if let contactNode = schemaNode.properties["contact"] {
                print("Contact node kind: \(contactNode.kind)")
                print("Contact node keys: \(contactNode.objectKeys)")
            }
            
        } else {
            XCTFail("No schema JSON extracted from transcript")
        }
    }
}