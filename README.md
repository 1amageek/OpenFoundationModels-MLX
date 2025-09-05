# OpenFoundationModels-MLX

An MLX-backed adapter for [OpenFoundationModels](https://github.com/1amageek/OpenFoundationModels) that provides local inference capabilities while maintaining 100% API compatibility with Apple's LanguageModel protocol.

## Features

- **Full LanguageModel Protocol Compatibility**: Drop-in replacement for OpenFoundationModels implementations
- **TokenTrie-based Schema-Constrained Decoding (SCD)**: Ensures JSON generation strictly adheres to defined schemas at the token level
- **Local Inference**: Run language models on Apple Silicon using MLX
- **Structured Output Guarantee**: JSON outputs always conform to specified schemas
- **Tool Call Support**: Automatic detection and parsing of tool/function calls
- **Streaming Support**: Real-time text generation with AsyncStream

## Requirements

- macOS 14.0+ (Sonoma or later)
- Swift 6.2+ (Xcode 16.x or later)
- Apple Silicon Mac (M1/M2/M3)

## Installation

### Swift Package Manager

Add the following to your `Package.swift`:

```swift
dependencies: [
    .package(url: "https://github.com/1amageek/OpenFoundationModels-MLX.git", from: "1.0.0"),
    .package(url: "https://github.com/1amageek/OpenFoundationModels.git", from: "1.0.0")
]
```

Then add to your target:

```swift
.target(
    name: "YourTarget",
    dependencies: [
        .product(name: "OpenFoundationModels", package: "OpenFoundationModels"),
        .product(name: "OpenFoundationModelsMLX", package: "OpenFoundationModels-MLX")
    ]
)
```

## Usage

### Basic Text Generation

```swift
import OpenFoundationModels
import OpenFoundationModelsMLX

// Initialize the MLX language model
let model = MLXLanguageModel(
    modelName: "llama-3.2-1b-instruct",  // or your preferred model
    temperature: 0.7,
    maxTokens: 1000
)

// Create a transcript
var transcript = Transcript()
transcript.append(.user("What is the capital of France?"))

// Generate response
let response = try await model.generate(
    transcript: transcript,
    options: nil
)

print(response.content) // "The capital of France is Paris."
```

### Streaming Generation

```swift
// Stream responses token by token
for try await entry in model.stream(transcript: transcript, options: nil) {
    if case .assistant(let content, _) = entry {
        print(content, terminator: "") // Print as tokens arrive
    }
}
```

### Schema-Constrained JSON Generation

The key innovation of this library is **TokenTrie-based Schema-Constrained Decoding (SCD)**, which ensures JSON outputs always match your schema:

```swift
// Define your schema
struct PersonSchema: Codable {
    let firstName: String
    let lastName: String
    let age: Int
    let email: String?
}

// Configure options with schema
let options = LanguageModelOptions(
    schema: PersonSchema.self  // Automatically constrains output to match schema
)

// Add user message requesting structured data
transcript.append(.user("Generate information for a software engineer named John."))

// Generate with schema constraints
let response = try await model.generate(
    transcript: transcript,
    options: options
)

// Parse the guaranteed-valid JSON
if case .assistant(let content, _) = response {
    let decoder = JSONDecoder()
    let person = try decoder.decode(PersonSchema.self, from: content.data(using: .utf8)!)
    print("Name: \(person.firstName) \(person.lastName)")
    print("Age: \(person.age)")
}
```

### Tool/Function Calling

```swift
// Define available tools
let weatherTool = Tool(
    name: "get_weather",
    description: "Get the current weather for a location",
    parameters: [
        "location": "string",
        "units": "celsius or fahrenheit"
    ]
)

// Configure with tools
let options = LanguageModelOptions(
    tools: [weatherTool]
)

// Ask about weather
transcript.append(.user("What's the weather in Tokyo?"))

// Generate - automatically detects tool call intent
let response = try await model.generate(
    transcript: transcript,
    options: options
)

// Check for tool calls
if case .toolCall(let toolCall) = response {
    print("Tool: \(toolCall.name)")
    print("Arguments: \(toolCall.arguments)")
    // Execute tool and continue conversation...
}
```

### Advanced Configuration

```swift
// Full configuration example
let model = MLXLanguageModel(
    modelName: "llama-3.2-3b-instruct",
    temperature: 0.8,           // Creativity level (0.0-1.0)
    maxTokens: 2000,            // Maximum response length
    topP: 0.95,                 // Nucleus sampling
    topK: 50,                   // Top-K sampling
    repetitionPenalty: 1.1,     // Reduce repetition
    seed: 42                    // For reproducible outputs
)

// Configuration through options
let options = LanguageModelOptions(
    maxTokens: 1500,
    temperature: 0.9,
    schema: MySchema.self,      // Enable schema constraints
    tools: [myTool1, myTool2]   // Available tools
)

// Check model availability
if model.isAvailable {
    print("Model loaded and ready")
}

// Check locale support
if model.supports(locale: .init(identifier: "ja_JP")) {
    print("Model supports Japanese")
}
```

## How TokenTrie-based SCD Works

Traditional JSON generation often produces invalid syntax or missing fields. Our TokenTrie-based approach solves this by constraining generation at the token level:

1. **Schema Analysis**: Extracts all valid key names from your schema
2. **TokenTrie Construction**: Builds a trie of valid token sequences for keys
3. **Constrained Generation**: During generation, only allows tokens that form valid keys
4. **State Machine Validation**: Tracks JSON structure to apply constraints contextually

This ensures 100% schema compliance without post-processing or retries.

## Building from Source

```bash
# Clone the repository
git clone https://github.com/1amageek/OpenFoundationModels-MLX.git
cd OpenFoundationModels-MLX

# Build
swift build

# Run tests
swift test

# Build for release
swift build -c release
```

## Examples

### Chat Application

```swift
class ChatSession {
    let model = MLXLanguageModel(modelName: "llama-3.2-1b-instruct")
    var transcript = Transcript()
    
    func sendMessage(_ message: String) async throws -> String {
        transcript.append(.user(message))
        
        let response = try await model.generate(
            transcript: transcript,
            options: nil
        )
        
        transcript.append(response)
        
        if case .assistant(let content, _) = response {
            return content
        }
        return ""
    }
}
```

### Structured Data Extraction

```swift
struct ProductReview: Codable {
    let productName: String
    let rating: Int  // 1-5
    let pros: [String]
    let cons: [String]
    let recommendation: Bool
}

func extractReview(from text: String) async throws -> ProductReview {
    let model = MLXLanguageModel(modelName: "llama-3.2-3b-instruct")
    
    var transcript = Transcript()
    transcript.append(.user("""
        Extract structured review data from this text:
        \(text)
    """))
    
    let options = LanguageModelOptions(schema: ProductReview.self)
    let response = try await model.generate(transcript: transcript, options: options)
    
    if case .assistant(let content, _) = response {
        return try JSONDecoder().decode(ProductReview.self, from: content.data(using: .utf8)!)
    }
    
    throw ExtractionError.invalidResponse
}
```

## Performance Considerations

- **Model Loading**: First inference is slower due to model loading. Keep models in memory for subsequent calls.
- **Memory Usage**: Larger models require more RAM. Monitor memory usage with Activity Monitor.
- **GPU Utilization**: MLX automatically uses Apple Silicon GPU for acceleration.
- **Batch Processing**: Process multiple requests in parallel for better throughput.

## Troubleshooting

### Model Not Loading
- Ensure you have sufficient disk space for model downloads
- Check internet connection for initial model fetch
- Verify model name is correct

### Schema Constraints Not Working
- Check that your schema is Codable-compliant
- Verify JSON key names are valid identifiers
- Ensure the model supports structured generation

### Memory Issues
- Use smaller models for development/testing
- Implement proper model lifecycle management
- Consider quantized model variants

## Contributing

Contributions are welcome! Please feel free to submit issues and pull requests.

## Related Projects

- [OpenFoundationModels](https://github.com/1amageek/OpenFoundationModels) - The base framework providing the LanguageModel protocol
- [MLX Swift](https://github.com/ml-explore/mlx-swift) - Apple's machine learning framework for Apple Silicon

## License

MIT License - see LICENSE file for details.

## Author

Created by [@1amageek](https://x.com/1amageek)

## Acknowledgments

- Built on [OpenFoundationModels](https://github.com/1amageek/OpenFoundationModels) framework
- Powered by [MLX](https://github.com/ml-explore/mlx-swift) for efficient Apple Silicon inference
- Uses [swift-transformers](https://github.com/huggingface/swift-transformers) for tokenization