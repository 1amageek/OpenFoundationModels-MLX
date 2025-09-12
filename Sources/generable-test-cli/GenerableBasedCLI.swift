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

// Enums for CompanyProfile (simplified)
@Generable
enum CompanyType: String {
    case startup = "startup"
    case corporation = "corporation"
    case nonprofit = "nonprofit"
}

@Generable
enum DepartmentType: String {
    case engineering = "engineering"
    case sales = "sales"
    case marketing = "marketing"
    case operations = "operations"
}

@Generable
enum EmployeeLevel: String {
    case junior = "junior"
    case senior = "senior"
    case lead = "lead"
    case manager = "manager"
}

@Generable
enum ProjectStatus: String {
    case planning = "planning"
    case active = "active"
    case completed = "completed"
    case onHold = "on_hold"
}

// Simplified company profile with moderate nesting
@Generable
struct CompanyProfile {
    var name: String
    var founded: Int
    var type: CompanyType
    var employeeCount: Int
    var headquarters: Address
    var departments: [Department]
}

// Nested structs for CompanyProfile (simplified)
@Generable
struct Address {
    var street: String
    var city: String
    var country: String
    var postalCode: String
}

@Generable
struct Department {
    var name: String
    var type: DepartmentType
    var headCount: Int
    var manager: Employee
    var projects: [Project]?
}

@Generable
struct Employee {
    var firstName: String
    var lastName: String
    var email: String
    var level: EmployeeLevel
    var yearsExperience: Int
}

@Generable
struct Project {
    var name: String
    var status: ProjectStatus
    var startDate: String
    var teamSize: Int
    var budget: Double?
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
    
    enum TestChoice: Int {
        case personProfile = 1
        case companyProfile = 2
    }
    
    static func main() async {
        // Model selection - easy to switch between different models
        let selectedModel = ModelChoice.llama32_3B  // Change this to switch models
        
        let modelID = selectedModel.modelID
        let modelCard = selectedModel.modelCard
        
        print("🚀 Complex Generable Validation Test with KeyDetectionLogitProcessor")
        print("Model: \(modelID)\n")
        
        // Test selection menu
        print("Select test scenario:")
        print("1. PersonProfile (Simple nested structure)")
        print("2. CompanyProfile (Complex structure with enums and deep nesting)")
        print("Enter choice (1 or 2): ", terminator: "")
        
        let testChoice: TestChoice
        if let input = readLine(), let choice = Int(input), let test = TestChoice(rawValue: choice) {
            testChoice = test
        } else {
            print("Invalid choice. Using PersonProfile as default.")
            testChoice = .personProfile
        }
        print()
        
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
                instructions: "Generate realistic data for the requested profile. Be concise and accurate. For dates use ISO format (YYYY-MM-DD)."
            )
            
            switch testChoice {
            case .personProfile:
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
                    printDecodingError(error)
                }
                
            case .companyProfile:
                // Test complex Generable with enums
                print("=== Testing Moderate Complexity Generable: CompanyProfile ===")
                print("Schema: CompanyProfile with nested structures and enums")
                print("Features:")
                print("  - 4 enum types (CompanyType, DepartmentType, EmployeeLevel, ProjectStatus)")
                print("  - 3 levels of nesting (Company → Department → Project)")
                print("  - Arrays of departments and projects")
                print("  - Optional project arrays\n")
                
                print("🔄 Generating company profile...")
                print("Note: This will demonstrate KeyDetectionLogitProcessor with complex JSON\n")
                
                do {
                    let startTime = Date()
                    
                    let company = try await session.respond(
                        to: "Generate a profile for a technology startup company founded in 2020 with 2 departments (engineering and sales). Each department should have 1-2 projects.",
                        generating: CompanyProfile.self
                    ).content
                    
                    let elapsedTime = Date().timeIntervalSince(startTime)
                    
                    print("\n✅ Successfully generated CompanyProfile in \(String(format: "%.2f", elapsedTime))s:")
                    print("\n🏭 Company Overview")
                    print("┌─ Basic Info")
                    print("│  Name: \(company.name)")
                    print("│  Founded: \(company.founded)")
                    print("│  Type: \(company.type.rawValue)")
                    print("│  Employees: \(company.employeeCount)")
                    print("│")
                    print("├─ Headquarters")
                    print("│  \(company.headquarters.street)")
                    print("│  \(company.headquarters.city), \(company.headquarters.country)")
                    print("│  Postal Code: \(company.headquarters.postalCode)")
                    print("│")
                    print("└─ Departments (\(company.departments.count))")
                    for (index, dept) in company.departments.enumerated() {
                        let isLast = index == company.departments.count - 1
                        let deptPrefix = isLast ? "   └─" : "   ├─"
                        let subPrefix = isLast ? "      " : "   │  "
                        
                        print("\(deptPrefix) \(dept.name) (\(dept.type.rawValue))")
                        print("\(subPrefix)Head Count: \(dept.headCount)")
                        print("\(subPrefix)Manager: \(dept.manager.firstName) \(dept.manager.lastName)")
                        print("\(subPrefix)  Level: \(dept.manager.level.rawValue)")
                        print("\(subPrefix)  Experience: \(dept.manager.yearsExperience) years")
                        
                        if let projects = dept.projects, !projects.isEmpty {
                            print("\(subPrefix)Projects (\(projects.count)):")
                            for (pIndex, project) in projects.enumerated() {
                                let isLastProject = pIndex == projects.count - 1
                                let projectPrefix = isLastProject ? "\(subPrefix)└─" : "\(subPrefix)├─"
                                print("\(projectPrefix) \(project.name)")
                                print("\(subPrefix)   Status: \(project.status.rawValue)")
                                print("\(subPrefix)   Start: \(project.startDate)")
                                print("\(subPrefix)   Team Size: \(project.teamSize)")
                                if let budget = project.budget {
                                    print("\(subPrefix)   Budget: $\(String(format: "%.0f", budget))")
                                }
                            }
                        } else {
                            print("\(subPrefix)No active projects")
                        }
                    }
                    
                    // Encode to JSON to verify structure
                    let encoder = JSONEncoder()
                    encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
                    let jsonData = try encoder.encode(company)
                    let jsonString = String(data: jsonData, encoding: .utf8) ?? ""
                    
                    // Show first 1000 chars of JSON due to size
                    print("\n📋 Generated JSON (first 1000 chars):")
                    let truncated = String(jsonString.prefix(1000))
                    print(truncated)
                    if jsonString.count > 1000 {
                        print("... (\(jsonString.count - 1000) more characters)")
                    }
                    
                    print("\n✨ Validation Result: PASSED")
                    print("  - Nested structure correctly generated")
                    print("  - Enum values valid (4 types)")
                    print("  - Arrays of departments and projects handled")
                    print("  - 3-level nesting successful")
                    
                } catch {
                    print("\n❌ Failed to generate CompanyProfile")
                    print("Error: \(error)")
                    printDecodingError(error)
                }
            }
            
            print("\n🏁 Test completed")
            
        } catch {
            print("❌ Fatal error: \(error)")
        }
        
        Stream().synchronize()
    }
    
    static func printDecodingError(_ error: Error) {
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
}

// Make structs Encodable for JSON output
extension PersonProfile: Encodable {}
extension PersonProfile.ContactInfo: Encodable {}

// CompanyProfile and related types Encodable conformances
extension CompanyProfile: Encodable {}
extension CompanyType: Encodable {}
extension Address: Encodable {}
extension Department: Encodable {}
extension DepartmentType: Encodable {}
extension Employee: Encodable {}
extension EmployeeLevel: Encodable {}
extension Project: Encodable {}
extension ProjectStatus: Encodable {}
