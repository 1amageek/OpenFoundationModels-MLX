# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build Commands

```bash
# Build the project
swift build

# Build for release with optimizations
swift build -c release

# Run all tests
swift test

# Run specific test
swift test --filter TestClassName.testMethodName

# Clean build artifacts
swift package clean
```

## Package Structure

The project is organized into multiple libraries for better modularity:

### Libraries

**OpenFoundationModelsMLX** (Core)
- Core MLX adapter implementing Apple's LanguageModel protocol
- Internal utilities with `package` visibility for cross-library use
- Located in `Sources/OpenFoundationModelsMLX/`

**OpenFoundationModelsMLXGPT**
- GPT-specific model cards (GPTOSSModelCard)
- HarmonyParser for GPT output format parsing
- Located in `Sources/OpenFoundationModelsMLXGPT/`

**OpenFoundationModelsMLXLlama**
- Llama model cards (LlamaModelCard for Llama 2, Llama3ModelCard for Llama 3.2)
- Support for various Llama format variations
- Located in `Sources/OpenFoundationModelsMLXLlama/`

**OpenFoundationModelsMLXGemma**
- FunctionGemmaModelCard for Google's FunctionGemma models
- FunctionGemmaParser for parsing function call output format
- Specialized for function calling with `<start_function_call>` format
- Located in `Sources/OpenFoundationModelsMLXGemma/`

**OpenFoundationModelsMLXUtils**
- ModelLoader for downloading and caching models from Hugging Face
- Shared utilities independent of specific model types
- Located in `Sources/OpenFoundationModelsMLXUtils/`

## Architecture Overview

### Core Components

**MLXLanguageModel**: The main adapter implementing Apple's LanguageModel protocol, located in `Sources/OpenFoundationModelsMLX/Adapter/`. Provides drop-in compatibility with OpenFoundationModels.

**Generation Pipeline**: Located in `Internal/Pipeline/`:
- `GenerationPipeline`: Orchestrates the generation flow
- `GenerationOrchestrator`: Manages high-level generation coordination

**MLX Integration**: Located in `Internal/Engine/` and `Internal/Execution/`:
- `MLXBackend`: Interfaces with MLX Swift for model operations
- `MLXExecutor`: Handles actual model execution and token generation

### Key Design Patterns

**ModelCard Protocol**: Each model family implements a ModelCard that defines:
- Prompt formatting specific to the model
- Output parsing and processing
- Default generation parameters

**Streaming Support**: Both `generate()` and `stream()` methods are supported for synchronous and streaming generation.

## Testing Strategy

Tests are organized by component in `Tests/OpenFoundationModelsMLXTests/`:
- Unit tests for tokenizer and parser components
- Integration tests for model card functionality
- HarmonyParser tests for GPT-OSS output format
- FunctionGemma parser tests

## Model Support

Model cards define model-specific configurations and are organized by model family:

**GPT Models** (in `OpenFoundationModelsMLXGPT`):
- `GPTOSSModelCard`: Configuration for GPT-OSS models with Harmony format support

**Llama Models** (in `OpenFoundationModelsMLXLlama`):
- `LlamaModelCard`: Configuration for Llama 2 models
- `Llama3ModelCard`: Configuration for Llama 3.2 models

**Gemma Models** (in `OpenFoundationModelsMLXGemma`):
- `FunctionGemmaModelCard`: Configuration for FunctionGemma models
  - Default: `mlx-community/functiongemma-270m-it-bf16`
  - Specialized for function calling
  - Uses `<start_function_call>call:name{params}<end_function_call>` output format
  - Supports automatic tool call detection and parsing

Each card specifies tokenizer requirements, special tokens, and model-specific generation parameters.

## Development Notes

### Usage Example

```swift
import OpenFoundationModelsMLX
import OpenFoundationModelsMLXGPT   // For GPT models
import OpenFoundationModelsMLXLlama // For Llama models
import OpenFoundationModelsMLXGemma // For Gemma models
import OpenFoundationModelsMLXUtils // For ModelLoader

// Load a model
let loader = ModelLoader()
let container = try await loader.loadModel("model-id")

// Create a model card
let card = GPTOSSModelCard(id: "model-id")
// or
let card = Llama3ModelCard(id: "model-id")
// or
let card = FunctionGemmaModelCard()  // Uses default: mlx-community/functiongemma-270m-it-bf16

// Initialize the language model
let model = try await MLXLanguageModel(
    modelContainer: container,
    card: card
)
```

### Access Control

The project uses Swift's `package` access control for internal components that need to be shared between libraries but not exposed to external consumers:
- `TranscriptAccess`: Internal transcript processing utilities
- `Logger`: Internal logging utilities
- `HarmonyParser`: GPT-specific output parser (package-level in GPT library)

### Dependencies
- mlx-swift-lm: MLX language model library for Apple Silicon
- swift-transformers: Tokenization support from Hugging Face
- OpenFoundationModels: Base protocol definitions

### Performance Considerations
- Models are cached after first load for faster subsequent inference
- Token generation uses MLX's GPU acceleration on Apple Silicon

### Swift Executable Entry Points

**`@main` を使用する場合、ファイル名を `main.swift` 以外にすること**

`main.swift` はSwiftが自動的にトップレベルコードとして扱うため、`@main` 属性と競合する。

```
❌ main.swift + @main → コンパイルエラー
✅ App.swift + @main  → OK
```
