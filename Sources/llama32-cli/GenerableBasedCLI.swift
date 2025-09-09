import Foundation
import OpenFoundationModels
import OpenFoundationModelsExtra
import OpenFoundationModelsMacros
import OpenFoundationModelsMLX
import MLXLMCommon

// Define Generable struct for Person profile
@Generable
struct PersonProfile {
    var name: String
    var age: Int
    var occupation: String
    var hobbies: [String]?
    var contact: ContactInfo?
    
    @Generable
    struct ContactInfo {
        var email: String
        var phone: String?
    }
}

@main
struct Llama32GenerableCLI {
    static func main() async {
        let model = "mlx-community/Llama-3.2-3B-Instruct"
        
        do {
            try await runGenerableJSONTest(model: model)
        } catch {
            print("❌ Error: \(error)")
        }
    }
    
    static func runGenerableJSONTest(model: String) async throws {
        print("🦙 Llama 3.2 Generable-based JSON Test")
        print("=======================================")
        print("Model: \(model)")
        print()
        
        // Load model
        print("Loading model...")
        let loader = ModelLoader()
        let progress = Progress(totalUnitCount: 100)
        
        // Simple progress monitoring
        Task {
            while progress.fractionCompleted < 1.0 {
                let percentage = Int(progress.fractionCompleted * 100)
                print("\rProgress: \(percentage)%", terminator: "")
                fflush(stdout)
                try await Task.sleep(nanoseconds: 200_000_000) // 0.2 seconds
            }
            print("\rModel ready!              ")
        }
        
        let modelContainer = try await loader.loadModel(model, progress: progress)
        
        // Create language model
        let card = LlamaModelCard(id: model)
        let languageModel = try await MLXLanguageModel(
            modelContainer: modelContainer,
            card: card
        )
        
        // Create LanguageModelSession with instructions
        let systemInstructions = """
        You are a helpful assistant that generates JSON data objects with actual values.
        
        IMPORTANT: 
        - Generate a JSON DATA OBJECT with real values, NOT a JSON schema
        - Do NOT include schema keywords like "type", "properties", "required"
        - Generate actual data values that match the field types
        - Your response must be valid JSON starting with { and ending with }
        - Do not include any text before or after the JSON object
        
        Expected structure for PersonProfile:
        - name (string): A person's full name
        - age (integer): Age between 0 and 150
        - occupation (string): Job or profession
        - hobbies (array of strings): 1 to 5 hobbies (optional)
        - contact (object): Contains email (required) and phone (optional)
        
        Example output:
        {
          "name": "Alice Smith",
          "age": 28,
          "occupation": "Software Engineer",
          "hobbies": ["reading", "coding", "hiking"],
          "contact": {
            "email": "alice@example.com",
            "phone": "+1-555-0123"
          }
        }
        """
        
        // Initialize session with the language model and instructions
        let session = LanguageModelSession(
            model: languageModel,
            instructions: systemInstructions
        )
        
        print("\n📋 Type-safe JSON Generation with @Generable")
        print("Schema: PersonProfile")
        print("Required fields: name, age, occupation")
        print("Optional fields: hobbies, contact")
        print("\n💬 Enter a prompt to generate structured data:\n")
        

        print("\n🔄 Generating structured response using @Generable...\n")
        
        // Debug: Print schema information
        print("🔍 [GenerableCLI] PersonProfile schema:")
        print("📋 [GenerableCLI] Schema: \(PersonProfile.generationSchema)")
        
        do {
            // Generate structured response using Generable type
            print("🚀 [GenerableCLI] Calling session.respond with PersonProfile.self...")
            let response = try await session.respond(
                to: "Generate a profile for a software engineer",
                generating: PersonProfile.self
            )
            print("✅ [GenerableCLI] Response received successfully")
            
            // Extract the generated PersonProfile
            let profile = response.content
            
            print("✅ Successfully generated PersonProfile:\n")
            print(response.content)
            
            // Display the structured data
            print("📊 Generated Data:")
            print(String(repeating: "-", count: 40))
            print("Name: \(profile.name)")
            print("Age: \(profile.age)")
            print("Occupation: \(profile.occupation)")
            
            if let hobbies = profile.hobbies {
                print("Hobbies: \(hobbies.joined(separator: ", "))")
            }
            
            if let contact = profile.contact {
                print("Contact:")
                print("  Email: \(contact.email)")
                if let phone = contact.phone {
                    print("  Phone: \(phone)")
                }
            }
            print(String(repeating: "-", count: 40))
            
            // Convert to JSON for display
            print("\n📄 JSON Representation:")
            let encoder = JSONEncoder()
            encoder.outputFormatting = .prettyPrinted
            if let jsonData = try? encoder.encode(profile),
               let jsonString = String(data: jsonData, encoding: .utf8) {
                print(jsonString)
            }
            
            // Validate the generated data
            validateProfile(profile)
            
            // Show session transcript for debugging
            print("\n📜 Session Transcript:")
            print("Total entries: \(session.transcript.count)")
            for (index, entry) in session.transcript.enumerated() {
                switch entry {
                case .instructions:
                    print("  \(index): [Instructions]")
                case .prompt(let prompt):
                    if let responseFormat = prompt.responseFormat {
                        print("  \(index): [Prompt with ResponseFormat: \(responseFormat.name)]")
                    } else {
                        print("  \(index): [Prompt]")
                    }
                case .response:
                    print("  \(index): [Response]")
                default:
                    print("  \(index): [Other]")
                }
            }
            
            // Show raw generated content for debugging
            print("\n🔍 Raw GeneratedContent:")
            print("Type: \(type(of: response.rawContent))")
            print("Content: \(response.rawContent)")
            
        } catch {
            print("❌ Error generating response: \(error)")
            print("Error type: \(type(of: error))")
            print("Error details: \(String(describing: error))")
            if let nsError = error as NSError? {
                print("Error domain: \(nsError.domain)")
                print("Error code: \(nsError.code)")
                print("Error userInfo: \(nsError.userInfo)")
            }
        }
        
        print("\n👋 Generable test completed!")
    }
    
    static func validateProfile(_ profile: PersonProfile) {
        print("\n📋 Data Validation:")
        
        var validationPassed = true
        
        // Validate name
        if !profile.name.isEmpty {
            print("  ✅ Name is valid: \"\(profile.name)\"")
        } else {
            print("  ❌ Name is empty")
            validationPassed = false
        }
        
        // Validate age
        if profile.age >= 0 && profile.age <= 150 {
            print("  ✅ Age is valid (0-150): \(profile.age)")
        } else {
            print("  ❌ Age out of range: \(profile.age)")
            validationPassed = false
        }
        
        // Validate occupation
        if !profile.occupation.isEmpty {
            print("  ✅ Occupation is valid: \"\(profile.occupation)\"")
        } else {
            print("  ❌ Occupation is empty")
            validationPassed = false
        }
        
        // Validate hobbies if present
        if let hobbies = profile.hobbies {
            if hobbies.count >= 1 && hobbies.count <= 5 {
                print("  ✅ Hobbies count valid (1-5): \(hobbies.count)")
            } else {
                print("  ⚠️ Hobbies count out of range: \(hobbies.count)")
            }
            
            if hobbies.allSatisfy({ !$0.isEmpty }) {
                print("  ✅ All hobbies are non-empty")
            } else {
                print("  ❌ Some hobbies are empty")
                validationPassed = false
            }
        }
        
        // Validate contact if present
        if let contact = profile.contact {
            if !contact.email.isEmpty {
                print("  ✅ Email is valid: \"\(contact.email)\"")
            } else {
                print("  ❌ Email is empty")
                validationPassed = false
            }
            
            if let phone = contact.phone, !phone.isEmpty {
                print("  ✅ Phone is valid: \"\(phone)\"")
            }
        }
        
        if validationPassed {
            print("\n✅ Validation PASSED - All required fields are valid")
        } else {
            print("\n❌ Validation FAILED - Some fields are invalid")
        }
    }
}

// Make PersonProfile conform to Encodable for JSON serialization
extension PersonProfile: Encodable {
    enum CodingKeys: String, CodingKey {
        case name, age, occupation, hobbies, contact
    }
    
    func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        try container.encode(name, forKey: .name)
        try container.encode(age, forKey: .age)
        try container.encode(occupation, forKey: .occupation)
        try container.encodeIfPresent(hobbies, forKey: .hobbies)
        try container.encodeIfPresent(contact, forKey: .contact)
    }
}

extension PersonProfile.ContactInfo: Encodable {
    enum CodingKeys: String, CodingKey {
        case email, phone
    }
    
    func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        try container.encode(email, forKey: .email)
        try container.encodeIfPresent(phone, forKey: .phone)
    }
}
