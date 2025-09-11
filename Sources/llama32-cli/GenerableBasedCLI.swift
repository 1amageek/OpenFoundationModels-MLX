import Foundation
import OpenFoundationModels
import OpenFoundationModelsExtra
import OpenFoundationModelsMacros
import OpenFoundationModelsMLX
import MLXLMCommon
import MLX

// Complex hierarchical JSON structure for a company
@Generable
struct Company {
    var name: String
    var founded: Int
    var headquarters: Address
    var employees: Int
    var revenue: Revenue?
    var departments: [Department]
    var products: [Product]?
    var partnerships: [Partnership]?
    
    @Generable
    struct Address {
        var street: String
        var city: String
        var state: String
        var country: String
        var postalCode: String
        var coordinates: Coordinates?
        
        @Generable
        struct Coordinates {
            var latitude: Double
            var longitude: Double
        }
    }
    
    @Generable
    struct Revenue {
        var annual: Double
        var currency: String
        var fiscalYear: Int
        var quarters: [QuarterlyRevenue]?
        
        @Generable
        struct QuarterlyRevenue {
            var quarter: String
            var amount: Double
            var growth: Double?
        }
    }
    
    @Generable
    struct Department {
        var name: String
        var headCount: Int
        var budget: Double?
        var lead: Employee
        var teams: [Team]?
        
        @Generable
        struct Employee {
            var id: String
            var name: String
            var title: String
            var email: String
            var yearsAtCompany: Int?
            var skills: [String]?
        }
        
        @Generable
        struct Team {
            var name: String
            var size: Int
            var focus: String
            var projects: [String]?
        }
    }
    
    @Generable
    struct Product {
        var name: String
        var category: String
        var price: Double?
        var launched: String?
        var features: [String]?
        var metrics: ProductMetrics?
        
        @Generable
        struct ProductMetrics {
            var users: Int
            var rating: Double?
            var reviews: Int?
            var marketShare: Double?
        }
    }
    
    @Generable
    struct Partnership {
        var partnerName: String
        var type: String
        var since: String
        var value: Double?
        var status: String
    }
}

@main
struct ComplexJSONGeneratorCLI {
    static func main() async {
        let model = "mlx-community/Llama-3.2-3B-Instruct"
        
        print("🏢 Complex JSON Generation Test")
        print("=" * 50)
        print("Model: \(model)")
        print("Structure: Hierarchical Company Organization")
        print("=" * 50)
        print()
        
        do {
            try await runComplexJSONTest(model: model)
        } catch {
            print("❌ Fatal Error: \(error)")
            print("Error Type: \(type(of: error))")
            if let localizedError = error as? LocalizedError {
                if let description = localizedError.errorDescription {
                    print("Description: \(description)")
                }
                if let reason = localizedError.failureReason {
                    print("Reason: \(reason)")
                }
            }
        }
        
        // Always synchronize MLX tasks before program exit
        print("🏁 [DEBUG] main: About to call Stream().synchronize() at \(Date())")
        Stream().synchronize()
        print("🏁 [DEBUG] main: Stream().synchronize() completed")
        
        print("🏁 [DEBUG] main: About to exit program")
    }
    
    static func runComplexJSONTest(model: String) async throws {
        // Ensure MLX tasks are synchronized when exiting this function
        defer {
            print("🔄 [DEBUG] defer: About to call Stream().synchronize() at \(Date())")
            Stream().synchronize()
            print("🔄 [DEBUG] defer: Stream().synchronize() completed")
        }
        
        print("📦 Loading Model...")
        let loader = ModelLoader()
        let progress = Progress(totalUnitCount: 100)
        
        let progressTask = Task {
            var lastPercentage = 0
            while progress.fractionCompleted < 1.0 {
                let percentage = Int(progress.fractionCompleted * 100)
                if percentage != lastPercentage {
                    print("\r⏳ Progress: [\(String(repeating: "█", count: percentage/2))\(String(repeating: "░", count: 50-percentage/2))] \(percentage)%", terminator: "")
                    fflush(stdout)
                    lastPercentage = percentage
                }
                // Progress update removed - not essential
            }
            print("\r✅ Model Loaded Successfully!" + String(repeating: " ", count: 40))
        }
        
        let modelContainer = try await loader.loadModel(model, progress: progress)
        progressTask.cancel()
        
        let card = LlamaModelCard(id: model)
        let languageModel = try await MLXLanguageModel(
            modelContainer: modelContainer,
            card: card
        )
        
        let systemInstructions = """
        You are a data generation assistant that creates realistic and detailed JSON objects for companies.
        
        CRITICAL REQUIREMENTS:
        1. Generate a complete JSON DATA OBJECT with realistic values
        2. Do NOT generate JSON schemas or use schema keywords like "type", "properties", "required"
        3. All values must be realistic and internally consistent
        4. The JSON must be valid and properly formatted
        5. Start with { and end with } without any additional text
        
        STRUCTURE REQUIREMENTS:
        - Company must have multiple departments (at least 3)
        - Each department must have a lead employee with complete information
        - Include detailed address with coordinates
        - Revenue should include quarterly breakdowns
        - Products should have metrics
        - All optional fields should be populated when sensible
        
        VALUE GUIDELINES:
        - Use realistic company names (tech companies preferred)
        - Founded year: 1975-2023
        - Employee count: 100-100000
        - Revenue in millions/billions USD
        - Coordinates should be valid lat/long
        - Email addresses should follow standard format
        - Dates in ISO format (YYYY-MM-DD)
        """
        
        let session = LanguageModelSession(
            model: languageModel,
            instructions: systemInstructions
        )
        
        print("\n🎯 Test Configuration:")
        print("├── Structure: Company (with nested objects)")
        print("├── Depth: Up to 4 levels")
        print("├── Arrays: departments, products, teams, features")
        print("├── Optional fields: Multiple at each level")
        print("└── Total fields: ~40+ possible fields")
        print()
        
        // Generate multiple examples to test consistency
        let prompts = [
            "Generate a detailed company profile for a successful AI technology company based in Silicon Valley with at least 4 departments, multiple products, and recent partnerships.",
            "Create a comprehensive company profile for a renewable energy corporation with global operations, including financial data and organizational structure.",
            "Produce a complete company profile for a biotechnology firm with research departments, product pipeline, and strategic partnerships."
        ]
        
        for (index, prompt) in prompts.enumerated() {
            print("\n" + "=" * 60)
            print("📊 Test Case \(index + 1) of \(prompts.count)")
            print("=" * 60)
            print("Prompt: \(prompt)")
            print()
            
            do {
                // Measure generation time
                let startTime = Date()
                
                print("🔄 Generating complex JSON structure...")
                let response = try await session.respond(
                    to: prompt,
                    generating: Company.self
                )
                
                let generationTime = Date().timeIntervalSince(startTime)
                print("⏱️ Generation completed in \(String(format: "%.2f", generationTime)) seconds")
                
                let company = response.content
                
                // Display summary statistics
                print("\n📈 Generated Structure Summary:")
                print("├── Company: \(company.name)")
                print("├── Founded: \(company.founded)")
                print("├── Employees: \(company.employees)")
                print("├── Departments: \(company.departments.count)")
                if let products = company.products {
                    print("├── Products: \(products.count)")
                }
                if let partnerships = company.partnerships {
                    print("├── Partnerships: \(partnerships.count)")
                }
                print("└── Location: \(company.headquarters.city), \(company.headquarters.country)")
                
                // Detailed department breakdown
                print("\n🏢 Department Details:")
                for dept in company.departments {
                    print("  ├── \(dept.name)")
                    print("  │   ├── Head Count: \(dept.headCount)")
                    print("  │   ├── Lead: \(dept.lead.name) (\(dept.lead.title))")
                    if let teams = dept.teams {
                        print("  │   └── Teams: \(teams.count)")
                    }
                }
                
                // Convert to JSON and display
                print("\n📄 Generated JSON (truncated preview):")
                let encoder = JSONEncoder()
                encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
                if let jsonData = try? encoder.encode(company),
                   let jsonString = String(data: jsonData, encoding: .utf8) {
                    // Show first 1000 characters for preview
                    let preview = String(jsonString.prefix(1000))
                    print(preview)
                    if jsonString.count > 1000 {
                        print("... [\(jsonString.count - 1000) more characters]")
                    }
                    
                    // Calculate JSON size
                    let sizeKB = Double(jsonData.count) / 1024.0
                    print("\n📏 JSON Size: \(String(format: "%.2f", sizeKB)) KB")
                }
                
                // Validate the generated data
                let validationResult = validateCompany(company)
                print("\n✅ Validation: \(validationResult.passed ? "PASSED" : "FAILED")")
                if !validationResult.errors.isEmpty {
                    print("Issues found:")
                    for error in validationResult.errors {
                        print("  ⚠️ \(error)")
                    }
                }
                
            } catch {
                print("❌ [DEBUG] Generation failed at \(Date()): \(error)")
                print("❌ [DEBUG] About to call Stream().synchronize() after error")
                Stream().synchronize()
                print("❌ [DEBUG] Stream().synchronize() completed after error")
                continue
            }
        }
        
        print("\n" + "=" * 60)
        print("🎉 Complex JSON Generation Test Completed!")
        print("=" * 60)
        
        print("📍 [DEBUG] About to call final Stream().synchronize() at \(Date())")
        // Synchronize with MLX background tasks to prevent crash on exit
        // TokenIterator uses asyncEval() which may have tasks still executing
        Stream().synchronize()
        print("📍 [DEBUG] Final Stream().synchronize() completed")
    }
    
    static func validateCompany(_ company: Company) -> (passed: Bool, errors: [String]) {
        var errors: [String] = []
        
        // Basic field validation
        if company.name.isEmpty {
            errors.append("Company name is empty")
        }
        
        if company.founded < 1800 || company.founded > 2024 {
            errors.append("Founded year \(company.founded) is unrealistic")
        }
        
        if company.employees < 1 {
            errors.append("Employee count must be positive")
        }
        
        if company.departments.isEmpty {
            errors.append("Company must have at least one department")
        }
        
        for dept in company.departments {
            if dept.name.isEmpty {
                errors.append("Department name is empty")
            }
            if dept.headCount < 1 {
                errors.append("Department \(dept.name) has invalid headcount")
            }
            if dept.lead.email.isEmpty || !dept.lead.email.contains("@") {
                errors.append("Invalid email for \(dept.lead.name)")
            }
        }
        
        if company.headquarters.city.isEmpty || company.headquarters.country.isEmpty {
            errors.append("Incomplete headquarters address")
        }
        
        if let coords = company.headquarters.coordinates {
            if coords.latitude < -90 || coords.latitude > 90 {
                errors.append("Invalid latitude: \(coords.latitude)")
            }
            if coords.longitude < -180 || coords.longitude > 180 {
                errors.append("Invalid longitude: \(coords.longitude)")
            }
        }
        
        if let revenue = company.revenue {
            if revenue.annual <= 0 {
                errors.append("Annual revenue must be positive")
            }
            if let quarters = revenue.quarters {
                for quarter in quarters {
                    if quarter.amount < 0 {
                        errors.append("Negative quarterly revenue in \(quarter.quarter)")
                    }
                }
            }
        }
        
        return (errors.isEmpty, errors)
    }
}

extension String {
    static func *(lhs: String, rhs: Int) -> String {
        return String(repeating: lhs, count: rhs)
    }
}

extension Company: Encodable {}
extension Company.Address: Encodable {}
extension Company.Address.Coordinates: Encodable {}
extension Company.Revenue: Encodable {}
extension Company.Revenue.QuarterlyRevenue: Encodable {}
extension Company.Department: Encodable {}
extension Company.Department.Employee: Encodable {}
extension Company.Department.Team: Encodable {}
extension Company.Product: Encodable {}
extension Company.Product.ProductMetrics: Encodable {}
extension Company.Partnership: Encodable {}