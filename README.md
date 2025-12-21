# OpenFoundationModels-MLX

An MLX-backed adapter for [OpenFoundationModels](https://github.com/1amageek/OpenFoundationModels) that provides local inference capabilities while maintaining 100% API compatibility with Apple's LanguageModel protocol.

## Features

- **Full LanguageModel Protocol Compatibility**: Drop-in replacement for OpenFoundationModels implementations
- **Local Inference**: Run language models on Apple Silicon using MLX
- **Multi-Model Support**: Built-in support for GPT, Llama, and Gemma model families
- **Tool Call Support**: Automatic detection and parsing of tool/function calls
- **Streaming Support**: Real-time text generation with AsyncStream

## Requirements

- macOS 15.0+ (Sequoia or later)
- iOS 18.0+
- Swift 6.2+ (Xcode 16.x or later)
- Apple Silicon Mac (M1/M2/M3/M4)

## Installation

### Swift Package Manager

Add the following to your `Package.swift`:

```swift
dependencies: [
    .package(url: "https://github.com/1amageek/OpenFoundationModels-MLX.git", branch: "main"),
    .package(url: "https://github.com/1amageek/OpenFoundationModels.git", branch: "main")
]
```

Then add to your target:

```swift
.target(
    name: "YourTarget",
    dependencies: [
        .product(name: "OpenFoundationModels", package: "OpenFoundationModels"),
        .product(name: "OpenFoundationModelsMLX", package: "OpenFoundationModels-MLX"),
        .product(name: "OpenFoundationModelsMLXGPT", package: "OpenFoundationModels-MLX"),   // For GPT models
        .product(name: "OpenFoundationModelsMLXLlama", package: "OpenFoundationModels-MLX"), // For Llama models
        .product(name: "OpenFoundationModelsMLXGemma", package: "OpenFoundationModels-MLX"), // For Gemma models
        .product(name: "OpenFoundationModelsMLXUtils", package: "OpenFoundationModels-MLX")  // For ModelLoader
    ]
)
```

## Usage

### Basic Text Generation

```swift
import OpenFoundationModels
import OpenFoundationModelsMLX
import OpenFoundationModelsMLXLlama
import OpenFoundationModelsMLXUtils

// Load a Llama model
let loader = ModelLoader()
let modelContainer = try await loader.loadModel("mlx-community/Llama-3.2-1B-Instruct-4bit")

// Create model card for Llama 3.2
let card = Llama3ModelCard(id: "mlx-community/Llama-3.2-1B-Instruct-4bit")

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
    print(response.segments.first?.text?.content ?? "")
}
```

### Streaming Generation

```swift
// Stream responses token by token
for try await entry in model.stream(transcript: transcript, options: nil) {
    if case .response(let response) = entry {
        if let content = response.segments.first?.text?.content {
            print(content, terminator: "")
        }
    }
}
```

### Using GPT Models

```swift
import OpenFoundationModelsMLXGPT

// Load a GPT model
let loader = ModelLoader()
let modelContainer = try await loader.loadModel("lmstudio-community/gpt-oss-20b-MLX-8bit")

// Create model card for GPT-OSS
let card = GPTOSSModelCard(id: "lmstudio-community/gpt-oss-20b-MLX-8bit")

let model = try await MLXLanguageModel(
    modelContainer: modelContainer,
    card: card
)
```

### Using FunctionGemma for Tool Calling

```swift
import OpenFoundationModelsMLXGemma

// Load FunctionGemma model
let loader = ModelLoader()
let modelContainer = try await loader.loadModel("mlx-community/functiongemma-270m-it-bf16")

// Create model card for FunctionGemma
let card = FunctionGemmaModelCard()

let model = try await MLXLanguageModel(
    modelContainer: modelContainer,
    card: card
)

// FunctionGemma is optimized for function calling
// Output format: <start_function_call>call:function_name{params}<end_function_call>
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
}
```

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

## Package Structure

The project is organized into multiple libraries:

| Library | Description |
|---------|-------------|
| `OpenFoundationModelsMLX` | Core MLX adapter implementing LanguageModel protocol |
| `OpenFoundationModelsMLXGPT` | GPT model cards and HarmonyParser |
| `OpenFoundationModelsMLXLlama` | Llama 2 and Llama 3.2 model cards |
| `OpenFoundationModelsMLXGemma` | FunctionGemma model cards |
| `OpenFoundationModelsMLXUtils` | ModelLoader and shared utilities |

## Supported Models

### GPT Models
- GPT-OSS models (via `GPTOSSModelCard`)
- Support for Harmony format output parsing

### Llama Models
- Llama 2 models (via `LlamaModelCard`)
- Llama 3.2 models (via `Llama3ModelCard`)
- Various quantization formats supported

### Gemma Models
- FunctionGemma (via `FunctionGemmaModelCard`)
- Optimized for function/tool calling

## Performance Considerations

- **Model Loading**: First inference is slower due to model loading. Use ModelLoader's caching for better performance.
- **Memory Usage**: Larger models require more RAM. Monitor memory usage with Activity Monitor.
- **GPU Utilization**: MLX automatically uses Apple Silicon GPU for acceleration.

## Troubleshooting

### Model Not Loading
- Ensure you have sufficient disk space for model downloads
- Check internet connection for initial model fetch
- Verify model name is correct

### Memory Issues
- Use smaller models for development/testing
- Implement proper model lifecycle management
- Consider quantized model variants (4bit, 8bit)

## Contributing

Contributions are welcome! Please feel free to submit issues and pull requests.

## Related Projects

- [OpenFoundationModels](https://github.com/1amageek/OpenFoundationModels) - The base framework providing the LanguageModel protocol
- [MLX Swift](https://github.com/ml-explore/mlx-swift) - Apple's machine learning framework for Apple Silicon
- [mlx-swift-lm](https://github.com/ml-explore/mlx-swift-lm) - MLX language model library

## License

MIT License - see LICENSE file for details.

## Author

Created by [@1amageek](https://x.com/1amageek)

## Acknowledgments

- Built on [OpenFoundationModels](https://github.com/1amageek/OpenFoundationModels) framework
- Powered by [MLX](https://github.com/ml-explore/mlx-swift) for efficient Apple Silicon inference
- Uses [swift-transformers](https://github.com/huggingface/swift-transformers) for tokenization
