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

# Run the generable test CLI tool
swift run generable-test-cli

# Clean build artifacts
swift package clean
```

## Package Structure

The project is organized into multiple libraries for better modularity:

### Libraries

**OpenFoundationModelsMLX** (Core)
- Core MLX adapter implementing Apple's LanguageModel protocol
- ADAPT system for constraint-based generation
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

**OpenFoundationModelsMLXUtils**
- ModelLoader for downloading and caching models from Hugging Face
- Shared utilities independent of specific model types
- Located in `Sources/OpenFoundationModelsMLXUtils/`

## Architecture Overview

### Core Components

**MLXLanguageModel**: The main adapter implementing Apple's LanguageModel protocol, located in `Sources/OpenFoundationModelsMLX/Adapter/`. Provides drop-in compatibility with OpenFoundationModels.

**ADAPT System** (Adaptive Decoding with Advanced Processing Techniques): Located in `Internal/ADAPT/`, this includes:
- `KeyDetectionLogitProcessor`: Monitors and validates JSON key generation with schema awareness
- `JSONExtractor`: Extracts valid JSON from model outputs
- `JSONSchemaContextDetector`: Detects JSON generation context for constraint application
- `JSONStateMachine`: Tracks JSON structure state during generation

**Constraint System**: Located in `Internal/Constraints/`:
- `ConstraintEngine`: Protocol defining constraint application strategies
- `AdaptiveConstraintEngine`: Implements adaptive, context-aware constraints
- `JSONSchemaExtractor`: Extracts schemas from Codable types
- `SchemaNode`: Represents JSON schema structure for validation

**Generation Pipeline**: Located in `Internal/Pipeline/`:
- `GenerationPipeline`: Orchestrates the complete generation flow with constraints
- `GenerationOrchestrator`: Manages high-level generation coordination

**MLX Integration**: Located in `Internal/Engine/` and `Internal/Execution/`:
- `MLXBackend`: Interfaces with MLX Swift for model operations
- `MLXExecutor`: Handles actual model execution and token generation

### Key Design Patterns

**TokenTrie-based Schema-Constrained Decoding (SCD)**: The system builds tries of valid token sequences from JSON schemas and constrains generation at the token level to guarantee schema compliance.

**LogitProcessor Chain**: Multiple processors can be chained to apply different constraints (schema validation, key detection, tool detection) during generation.

**Adaptive Constraint Modes**: Supports three modes:
- Hard constraints: Strict token-level enforcement
- Soft constraints: Schema guidance via prompting
- Observation mode: Monitoring without interference

## Testing Strategy

Tests are organized by component in `Tests/OpenFoundationModelsMLXTests/`:
- Unit tests for individual components (e.g., `KeyDetectionLogitProcessorTests`)
- Integration tests for end-to-end scenarios (e.g., `KeyDetectionIntegrationTest`)
- Schema extraction and validation tests (e.g., `SchemaOutputTest`)

## Model Support

Model cards define model-specific configurations and are organized by model family:

**GPT Models** (in `OpenFoundationModelsMLXGPT`):
- `GPTOSSModelCard`: Configuration for GPT-OSS models with Harmony format support

**Llama Models** (in `OpenFoundationModelsMLXLlama`):
- `LlamaModelCard`: Configuration for Llama 2 models
- `Llama3ModelCard`: Configuration for Llama 3.2 models

Each card specifies tokenizer requirements, special tokens, and model-specific generation parameters.

## Development Notes

### Usage Example

```swift
import OpenFoundationModelsMLX
import OpenFoundationModelsMLXGPT  // For GPT models
import OpenFoundationModelsMLXLlama // For Llama models
import OpenFoundationModelsMLXUtils // For ModelLoader

// Load a model
let loader = ModelLoader()
let container = try await loader.loadModel("model-id")

// Create a model card
let card = GPTOSSModelCard(id: "model-id")
// or
let card = Llama3ModelCard(id: "model-id")

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
- MLX Swift (via mlx-swift-examples): Core ML framework for Apple Silicon
- swift-transformers: Tokenization support from Hugging Face
- OpenFoundationModels: Base protocol definitions

### Performance Considerations
- Models are cached after first load for faster subsequent inference
- Token generation uses MLX's GPU acceleration on Apple Silicon
- Constraint application is optimized to minimize overhead during generation