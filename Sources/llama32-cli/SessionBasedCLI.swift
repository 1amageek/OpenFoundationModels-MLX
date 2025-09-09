import Foundation
import OpenFoundationModels
import OpenFoundationModelsExtra
import OpenFoundationModelsMLX
import MLXLMCommon

@main
struct Llama32SessionCLI {
    static func main() async {
        let model = "mlx-community/Llama-3.2-3B-Instruct"
        
        do {
            try await runSessionJSONTest(model: model)
        } catch {
            print("❌ Error: \(error)")
        }
    }
    
    static func runSessionJSONTest(model: String) async throws {
        print("🦙 Llama 3.2 Session-based JSON Test")
        print("=====================================")
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
        
        Expected structure:
        - name (string): A person's full name
        - age (integer): Age between 0 and 150
        - occupation (string): Job or profession
        - hobbies (array of strings): 1 to 5 hobbies (optional)
        - contact (object): Contains email (required) and phone (optional)
        
        Example output:
        {
          "name": "Alice Smith",
          "age": 28,
          "occupation": "Engineer",
          "hobbies": ["reading", "coding"],
          "contact": {
            "email": "alice@example.com"
          }
        }
        """
        
        // Initialize session with the language model and instructions
        let session = LanguageModelSession(
            model: languageModel,
            instructions: systemInstructions
        )
        
        print("\n📋 JSON Generation with LanguageModelSession")
        print("Expected fields: name, age, occupation")
        print("Optional fields: hobbies, contact")
        print("\n💬 Enter a prompt to generate JSON:\n")
        
        print("You: ", terminator: "")
        fflush(stdout)
        
        guard let input = readLine() else { 
            print("No input received")
            return 
        }
        
        let trimmed = input.trimmingCharacters(in: .whitespacesAndNewlines)
        if trimmed.isEmpty { 
            print("Empty prompt")
            return 
        }
        
        print("\n🔄 Generating JSON response using Session...\n")
        
        do {
            // Generate response using session with a simple string prompt
            let response = try await session.respond(to: trimmed)
            
            // Extract the generated content
            let generatedText = response.content
            
            print("📝 Raw Response:")
            print(String(repeating: "=", count: 60))
            print(generatedText)
            print(String(repeating: "=", count: 60))
            print()
            
            // Try to extract and format JSON
            if let jsonStart = generatedText.firstIndex(of: "{"),
               let jsonEnd = generatedText.lastIndex(of: "}") {
                let jsonString = String(generatedText[jsonStart...jsonEnd])
                
                print("🔍 Extracted JSON String:")
                print(jsonString)
                print()
                
                // Pretty print and validate JSON
                if let data = jsonString.data(using: String.Encoding.utf8),
                   let jsonObject = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
                   let prettyData = try? JSONSerialization.data(withJSONObject: jsonObject, options: .prettyPrinted),
                   let prettyString = String(data: prettyData, encoding: .utf8) {
                    
                    print("📊 Formatted JSON:")
                    print(String(repeating: "-", count: 40))
                    print(prettyString)
                    print(String(repeating: "-", count: 40))
                    
                    // Validate against expected structure
                    validateJSON(jsonObject)
                }
            } else {
                print("⚠️ No JSON structure found in response")
            }
            
            // Show session transcript for debugging
            print("\n📜 Session Transcript:")
            print("Total entries: \(session.transcript.count)")
            for (index, entry) in session.transcript.enumerated() {
                switch entry {
                case .instructions:
                    print("  \(index): [Instructions]")
                case .prompt:
                    print("  \(index): [Prompt]")
                case .response:
                    print("  \(index): [Response]")
                default:
                    print("  \(index): [Other]")
                }
            }
            
        } catch {
            print("❌ Error generating response: \(error)")
        }
        
        print("\n👋 Session test completed!")
    }
    
    static func validateJSON(_ jsonObject: [String: Any]) {
        print("\n📋 Schema Validation:")
        
        // Check required fields
        let requiredFields = ["name", "age", "occupation"]
        var allRequiredPresent = true
        
        for field in requiredFields {
            if jsonObject[field] != nil {
                print("  ✅ Required field '\(field)': present")
            } else {
                print("  ❌ Required field '\(field)': MISSING")
                allRequiredPresent = false
            }
        }
        
        // Check field types
        if let name = jsonObject["name"] as? String {
            print("  ✅ 'name' is String: \"\(name)\"")
        } else if jsonObject["name"] != nil {
            print("  ❌ 'name' type error: expected String")
        }
        
        if let age = jsonObject["age"] as? Int {
            if age >= 0 && age <= 150 {
                print("  ✅ 'age' is Int (0-150): \(age)")
            } else {
                print("  ⚠️ 'age' out of range (0-150): \(age)")
            }
        } else if let ageDouble = jsonObject["age"] as? Double {
            let ageInt = Int(ageDouble)
            if ageInt >= 0 && ageInt <= 150 {
                print("  ✅ 'age' is Number (0-150): \(ageInt)")
            } else {
                print("  ⚠️ 'age' out of range (0-150): \(ageInt)")
            }
        } else if jsonObject["age"] != nil {
            print("  ❌ 'age' type error: expected Number")
        }
        
        if let occupation = jsonObject["occupation"] as? String {
            print("  ✅ 'occupation' is String: \"\(occupation)\"")
        } else if jsonObject["occupation"] != nil {
            print("  ❌ 'occupation' type error: expected String")
        }
        
        // Check optional fields
        if let hobbies = jsonObject["hobbies"] as? [String] {
            if hobbies.count >= 1 && hobbies.count <= 5 {
                print("  ✅ 'hobbies' is Array[String] (1-5 items): \(hobbies.count) items")
            } else {
                print("  ⚠️ 'hobbies' count out of range (1-5): \(hobbies.count) items")
            }
        }
        
        if let contact = jsonObject["contact"] as? [String: Any] {
            if contact["email"] != nil {
                print("  ✅ 'contact.email' is present")
            } else {
                print("  ❌ 'contact.email' is MISSING (required in contact)")
            }
        }
        
        if allRequiredPresent {
            print("\n✅ JSON validation PASSED - All required fields present")
        } else {
            print("\n❌ JSON validation FAILED - Missing required fields")
        }
    }
}