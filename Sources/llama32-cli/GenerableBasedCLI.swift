import Foundation
import OpenFoundationModels
import OpenFoundationModelsExtra
import OpenFoundationModelsMacros
import OpenFoundationModelsMLX
import MLX

// Simple blog post with nested author
@Generable
struct BlogPost {
    var title: String
    var content: String
    var author: Author
    var tags: [String]
    
    @Generable
    struct Author {
        var name: String
        var email: String
    }
}

// Simple order with nested items
@Generable
struct Order {
    var orderId: Int
    var customerName: String
    var items: [OrderItem]
    var total: Double
    
    @Generable
    struct OrderItem {
        var name: String
        var price: Double
        var quantity: Int
    }
}

@main
struct GenerableBasedCLI {
    static func main() async {
        // Use lmstudio-community 8-bit version for better quality
        let modelID = "lmstudio-community/gpt-oss-20b-MLX-8bit"
        
        print("🚀 JSON Generation Demo")
        print("Model: \(modelID)\n")
        
        do {
            print("📦 Loading model...")
            let loader = ModelLoader()
            
            // Check if model needs to be downloaded
            let cachedModels = loader.cachedModels()
            if !cachedModels.contains(modelID) {
                print("Model not cached. Downloading \(modelID)...")
                print("This is a ~20GB download and may take several minutes.")
            }
            
            // Create progress object
            let progress = Progress(totalUnitCount: 100)
            
            // Start a task to monitor progress
            let progressTask = Task {
                while !Task.isCancelled && !progress.isFinished {
                    let percentage = Int(progress.fractionCompleted * 100)
                    let completed = progress.completedUnitCount
                    let total = progress.totalUnitCount
                    
                    if total > 0 {
                        print("\rProgress: \(percentage)% (\(completed)/\(total))", terminator: "")
                        fflush(stdout)
                    } else {
                        print("\rProgress: \(percentage)%", terminator: "")
                        fflush(stdout)
                    }
                    
                    try? await Task.sleep(nanoseconds: 500_000_000) // Update every 0.5 seconds
                }
            }
            
            // Load model with progress
            let container = try await loader.loadModel(modelID, progress: progress)
            
            // Cancel progress monitoring
            progressTask.cancel()
            
            print("\n✅ Model loaded\n")
            
            // Create language model
            let languageModel = try await MLXLanguageModel(
                modelContainer: container,
                card: GPTOSSModelCard(id: modelID)
            )
            
            // Create session
            let session = LanguageModelSession(
                model: languageModel,
                instructions: "Generate realistic JSON data. Be concise and accurate."
            )
            
            // Test 1: Blog post with nested author
            print("--- Test 1: Blog Post ---")
            print("Generating blog post...")
            
            let post = try await session.respond(
                to: "Generate a blog post about AI technology",
                generating: BlogPost.self
            ).content
            
            print("✅ Generated Blog Post:")
            print("  Title: \(post.title)")
            print("  Author: \(post.author.name) (\(post.author.email))")
            print("  Tags: \(post.tags.joined(separator: ", "))")
            print("  Content: \(String(post.content.prefix(100)))...")
            
            // Test 2: Order with nested items
            print("\n--- Test 2: Order ---")
            print("Generating order...")
            
            let order = try await session.respond(
                to: "Generate an order for electronics with 3 items",
                generating: Order.self
            ).content
            
            print("✅ Generated Order #\(order.orderId):")
            print("  Customer: \(order.customerName)")
            print("  Items:")
            for item in order.items {
                print("    - \(item.name): $\(item.price) x \(item.quantity)")
            }
            print("  Total: $\(order.total)")
            
            // Show JSON
            print("\n--- JSON Preview ---")
            let encoder = JSONEncoder()
            encoder.outputFormatting = .prettyPrinted
            if let jsonData = try? encoder.encode(post),
               let jsonString = String(data: jsonData, encoding: .utf8) {
                print(String(jsonString.prefix(300)))
                print("...")
            }
            
            print("\n🎉 Tests completed!")
            
        } catch {
            print("❌ Error: \(error)")
        }
        
        Stream().synchronize()
    }
}

// Make structs Encodable for JSON output
extension BlogPost: Encodable {}
extension BlogPost.Author: Encodable {}
extension Order: Encodable {}
extension Order.OrderItem: Encodable {}