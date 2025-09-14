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

Model cards in `Public/ModelCards/` define model-specific configurations:
- `LlamaModelCard`: Configuration for Llama models
- `GPTOSSModelCard`: Configuration for GPT-OSS models

Each card specifies tokenizer requirements, special tokens, and model-specific generation parameters.

## Development Notes

### Dependencies
- MLX Swift (via mlx-swift-examples): Core ML framework for Apple Silicon
- swift-transformers: Tokenization support from Hugging Face
- OpenFoundationModels: Base protocol definitions

### Performance Considerations
- Models are cached after first load for faster subsequent inference
- Token generation uses MLX's GPU acceleration on Apple Silicon
- Constraint application is optimized to minimize overhead during generation