# OpenFoundationModels-MLX

An MLX-backed adapter for [OpenFoundationModels](https://github.com/1amageek/OpenFoundationModels) that provides local inference capabilities while maintaining 100% API compatibility with Apple's LanguageModel protocol.

## Features

- **Full LanguageModel Protocol Compatibility**: Drop-in replacement for OpenFoundationModels implementations
- **ADAPT System**: Adaptive Decoding with Advanced Processing Techniques for reliable structured output
- **Schema-Constrained Generation**: Ensures JSON outputs strictly conform to defined schemas
- **Local Inference**: Run language models on Apple Silicon using MLX
- **Multi-Model Support**: Built-in support for GPT and Llama model families
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
        .product(name: "OpenFoundationModelsMLX", package: "OpenFoundationModels-MLX"),
        .product(name: "OpenFoundationModelsMLXGPT", package: "OpenFoundationModels-MLX"),  // For GPT models
        .product(name: "OpenFoundationModelsMLXLlama", package: "OpenFoundationModels-MLX"), // For Llama models
        .product(name: "OpenFoundationModelsMLXUtils", package: "OpenFoundationModels-MLX")  // For ModelLoader
    ]
)
```

## Usage

### Basic Text Generation

```swift
import OpenFoundationModels
import OpenFoundationModelsMLX
import OpenFoundationModelsMLXGPT
import OpenFoundationModelsMLXUtils

// Load a GPT model
let loader = ModelLoader()
let modelContainer = try await loader.loadModel("lmstudio-community/gpt-oss-20b-MLX-8bit")

// Create model card for GPT
let card = GPTOSSModelCard(id: "lmstudio-community/gpt-oss-20b-MLX-8bit")

// Initialize the MLX language model
let model = try await MLXLanguageModel(
    modelContainer: modelContainer,
    card: card
)

// Create a transcript
var transcript = Transcript()
transcript.append(.user("What is the capital of France?"))

// Generate response
let response = try await model.generate(
    transcript: transcript,
    options: nil
)

if case .response(let response) = response {
    print(response.segments.first?.text?.content ?? "") // "The capital of France is Paris."
}
```

### Streaming Generation

```swift
// Stream responses token by token
for try await entry in model.stream(transcript: transcript, options: nil) {
    if case .response(let response) = entry {
        if let content = response.segments.first?.text?.content {
            print(content, terminator: "") // Print as tokens arrive
        }
    }
}
```

### Schema-Constrained JSON Generation

The ADAPT system ensures JSON outputs always match your schema through adaptive constraint application:

```swift
// Define your schema
struct PersonSchema: Codable {
    let firstName: String
    let lastName: String
    let age: Int
    let email: String?
}

// Configure options with schema
let options = GenerationOptions(
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
if case .response(let response) = response,
   let content = response.segments.first?.text?.content {
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
    parametersJSON: """
    {
        "type": "object",
        "properties": {
            "location": {"type": "string"},
            "units": {"type": "string", "enum": ["celsius", "fahrenheit"]}
        },
        "required": ["location"]
    }
    """
)

// Configure with tools
let options = GenerationOptions(
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
// Load model with custom parameters
let loader = ModelLoader()
let modelContainer = try await loader.loadModel("lmstudio-community/gpt-oss-20b-MLX-8bit")

// Create model card with custom parameters
let card = GPTOSSModelCard(id: "lmstudio-community/gpt-oss-20b-MLX-8bit")

// Add custom logit processors for additional constraints
let keyDetector = KeyDetectionLogitProcessor(validKeys: ["name", "age", "email"])

// Initialize with additional processors
let model = try await MLXLanguageModel(
    modelContainer: modelContainer,
    card: card,
    additionalProcessors: [keyDetector]
)

// Configuration through options
let options = GenerationOptions(
    maxTokens: 1500,
    temperature: 0.9,
    topP: 0.95,
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

## How ADAPT Works

The ADAPT (Adaptive Decoding with Advanced Processing Techniques) system ensures reliable structured output generation:

1. **Schema Analysis**: Extracts structure and constraints from your Codable types
2. **Context Detection**: Intelligently detects when JSON generation begins
3. **Key Validation**: Monitors and validates JSON keys during generation using TokenTrie
4. **State Machine Tracking**: Maintains JSON structure state for contextual constraints
5. **Adaptive Constraints**: Applies hard constraints, soft guidance, or observation mode based on context

This multi-layered approach ensures schema compliance while maintaining generation flexibility.

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
import OpenFoundationModels
import OpenFoundationModelsMLX
import OpenFoundationModelsMLXGPT
import OpenFoundationModelsMLXUtils

class ChatSession {
    let model: MLXLanguageModel
    var transcript = Transcript()

    init() async throws {
        let loader = ModelLoader()
        let container = try await loader.loadModel("lmstudio-community/gpt-oss-20b-MLX-8bit")
        let card = GPTOSSModelCard()
        self.model = try await MLXLanguageModel(modelContainer: container, card: card)
    }

    func sendMessage(_ message: String) async throws -> String {
        transcript.append(.user(message))

        let response = try await model.generate(
            transcript: transcript,
            options: nil
        )

        transcript.append(response)

        if case .response(let response) = response,
           let content = response.segments.first?.text?.content {
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
    // Initialize model with GPT
    let loader = ModelLoader()
    let container = try await loader.loadModel("lmstudio-community/gpt-oss-20b-MLX-8bit")
    let card = GPTOSSModelCard()
    let model = try await MLXLanguageModel(modelContainer: container, card: card)

    var transcript = Transcript()
    transcript.append(.user("""
        Extract structured review data from this text:
        \(text)
    """))

    let options = GenerationOptions(schema: ProductReview.self)
    let response = try await model.generate(transcript: transcript, options: options)

    if case .response(let response) = response,
       let content = response.segments.first?.text?.content {
        return try JSONDecoder().decode(ProductReview.self, from: content.data(using: .utf8)!)
    }

    throw ExtractionError.invalidResponse
}
```

## Supported Models

### GPT Models
- GPT-OSS models (via GPTOSSModelCard)
- Support for Harmony format output parsing
- Channel-aware constraint application

### Llama Models
- Llama 2 models (via LlamaModelCard)
- Llama 3.2 models (via Llama3ModelCard)
- Various quantization formats supported

## Performance Considerations

- **Model Loading**: First inference is slower due to model loading. Use ModelLoader's caching for better performance.
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