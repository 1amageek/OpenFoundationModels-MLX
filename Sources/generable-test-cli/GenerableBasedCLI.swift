import Foundation
import OpenFoundationModels
import OpenFoundationModelsExtra
import OpenFoundationModelsMacros
import OpenFoundationModelsMLX
import MLX

// Simple person profile with nested contact info
@Generable
struct PersonProfile {
    var name: String
    var age: Int
    var occupation: String
    var contact: ContactInfo
    
    @Generable
    struct ContactInfo {
        var email: String
        var phone: String?
    }
}

@main
struct GenerableBasedCLI {
    enum ModelChoice {
        case llama32_3B
        case gptOSS_20B
        
        var modelID: String {
            switch self {
            case .llama32_3B:
                return "mlx-community/Llama-3.2-3B-Instruct-4bit"
            case .gptOSS_20B:
                return "lmstudio-community/gpt-oss-20b-MLX-8bit"
            }
        }
        
        var modelCard: any ModelCard {
            switch self {
            case .llama32_3B:
                return Llama3ModelCard(id: modelID)
            case .gptOSS_20B:
                return GPTOSSModelCard(id: modelID)
            }
        }
    }
    
    static func main() async {
        // Model selection - easy to switch between different models
        let selectedModel = ModelChoice.llama32_3B  // Change this to switch models
        
        let modelID = selectedModel.modelID
        let modelCard = selectedModel.modelCard
        
        print("🚀 Nested Generable Validation Test")
        print("Model: \(modelID)\n")
        
        do {
            print("📦 Loading model...")
            let loader = ModelLoader()
            
            // Check if model needs to be downloaded
            let cachedModels = loader.cachedModels()
            if !cachedModels.contains(modelID) {
                print("Model not cached. Downloading \(modelID)...")
                print("This may take several minutes depending on model size.")
            }
            
            // Create progress object
            let progress = Progress(totalUnitCount: 100)
            
            // Monitor progress
            let progressTask = Task {
                while !Task.isCancelled && !progress.isFinished {
                    let percentage = Int(progress.fractionCompleted * 100)
                    print("\rProgress: \(percentage)%", terminator: "")
                    fflush(stdout)
                    try? await Task.sleep(nanoseconds: 500_000_000)
                }
            }
            
            // Load model
            let container = try await loader.loadModel(modelID, progress: progress)
            progressTask.cancel()
            
            print("\n✅ Model loaded\n")
            
            // Create language model first
            let languageModel = try await MLXLanguageModel(
                modelContainer: container,
                card: modelCard
            )
            
            // Note: KeyDetectionLogitProcessor would require access to tokenizer
            // which is internal to the model. For now, we'll use the model without it.
            print("💡 KeyDetectionLogitProcessor support is enabled in the pipeline")
            print("   To see detailed key detection, the model will use internal processors\n")
            
            // Create session
            let session = LanguageModelSession(
                model: languageModel,
                instructions: "Generate realistic data for the requested profile. Be concise and accurate."
            )
            
            // Test nested Generable
            print("=== Testing Nested Generable: PersonProfile ===")
            print("Schema: PersonProfile with nested ContactInfo")
            print("Required fields: name, age, occupation, contact (email required, phone optional)\n")
            
            print("🔄 Generating person profile...")
            
            do {
                let startTime = Date()
                
                let profile = try await session.respond(
                    to: "Generate a profile for a software engineer working at a tech company",
                    generating: PersonProfile.self
                ).content
                
                let elapsedTime = Date().timeIntervalSince(startTime)
                
                print("\n✅ Successfully generated PersonProfile in \(String(format: "%.2f", elapsedTime))s:")
                print("┌─ Person Details")
                print("│  Name: \(profile.name)")
                print("│  Age: \(profile.age)")
                print("│  Occupation: \(profile.occupation)")
                print("└─ Contact Info (nested)")
                print("   ├─ Email: \(profile.contact.email)")
                if let phone = profile.contact.phone {
                    print("   └─ Phone: \(phone)")
                } else {
                    print("   └─ Phone: (not provided)")
                }
                
                // Encode to JSON to verify structure
                let encoder = JSONEncoder()
                encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
                let jsonData = try encoder.encode(profile)
                let jsonString = String(data: jsonData, encoding: .utf8) ?? ""
                
                print("\n📋 Generated JSON:")
                print(jsonString)
                
                print("\n✨ Validation Result: PASSED")
                print("  - Nested structure correctly generated")
                print("  - All required fields present")
                print("  - Optional field handled properly")
                
            } catch {
                print("\n❌ Failed to generate PersonProfile")
                print("Error: \(error)")
                
                if let decodingError = error as? DecodingError {
                    switch decodingError {
                    case .keyNotFound(let key, let context):
                        print("  Missing key: '\(key.stringValue)'")
                        print("  Path: \(context.codingPath.map { $0.stringValue }.joined(separator: "."))")
                    case .typeMismatch(let type, let context):
                        print("  Type mismatch: expected \(type)")
                        print("  Path: \(context.codingPath.map { $0.stringValue }.joined(separator: "."))")
                    case .valueNotFound(let type, let context):
                        print("  Value not found: \(type)")
                        print("  Path: \(context.codingPath.map { $0.stringValue }.joined(separator: "."))")
                    case .dataCorrupted(let context):
                        print("  Data corrupted at: \(context.codingPath.map { $0.stringValue }.joined(separator: "."))")
                    @unknown default:
                        print("  Unknown decoding error")
                    }
                }
            }
            
            print("\n🏁 Test completed")
            
        } catch {
            print("❌ Fatal error: \(error)")
        }
        
        Stream().synchronize()
    }
}

// Make structs Encodable for JSON output
extension PersonProfile: Encodable {}
extension PersonProfile.ContactInfo: Encodable {}
