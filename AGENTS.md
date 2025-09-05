# Repository Guidelines

This repository provides an MLX‑backed adapter for OpenFoundationModels (SwiftPM). Keep changes small, focused, and consistent with the public API surface.

## Project Structure & Module Organization
- `Package.swift`: SwiftPM manifest.
- `Sources/OpenFoundationModels-MLX/` (module `OpenFoundationModelsMLX`):
  - `Adapter/`: `MLXLanguageModel` (conforms to `LanguageModel`; Apple‑compatible API, external surface stable).
  - `Internal/Engine/`: `MLXChatEngine`, `MLXBackend` (stub; run/retry control).
  - `Internal/Prompt/`: `PromptRenderer`, `TranscriptAccess`, `OptionsMapper`.
  - `Internal/Decode/`: SCD/Snap scaffolding (`KeyTrie`, `JSONKeyTracker`, token‑trie skeleton).
  - `Internal/Tooling/`: `ToolCallDetector`.
- `Tests/OpenFoundationModels-MLXTests/`: Swift Testing tests.

## Build, Test, and Development Commands
- `swift build` | `swift build -c release`: Build (debug/release).
- `swift package resolve` | `swift package update`: Resolve/update dependencies.
- `swift test --parallel` | `swift test --enable-code-coverage`: Run tests / with coverage.
- `xed .`: Open in Xcode (optional).

## Coding Style & Naming Conventions
- Swift 6.2, 4‑space indent, target ≤120 columns.
- Types/enums/protocols: PascalCase; methods/vars: lowerCamelCase.
- Public API must have `///` docs. Run `swift-format` if available.

## Testing Guidelines
- Use Swift Testing (`@Test`, `#expect(...)`).
- Name files like `<Feature>NameTests.swift`.
- Add minimal unit tests for changed logic (SCD/Snap/Tools).
- Example: `swift test --parallel` (fast), add `--enable-code-coverage` when needed.

## Architecture Notes
- Primitive policy: external API unchanged; retries only with identical settings; skip retry when `seed` is set.
- SCD: character‑level + Trie to detect key mismatches → internal retry. Token‑trie/logits‑mask ready once MLXLLM exposes step/logits.
- Snap: distance‑1 key correction; final validation requires `required` fields.
- Tool calls: inject STRICT‑JSON tool schema in system; when assistant emits `{"tool_calls":...}` return `Transcript.Entry.toolCalls` (not text).
- Transcript access via `OpenFoundationModelsExtra`; force schema/parameter JSON with `OFM_MLX_SCHEMA_JSON` if disabled.

## Commit & Pull Request Guidelines
- Commits: short, imperative, scoped (e.g., `feat(mlx): implement tool_calls detection`).
- PRs: state purpose, changes, tests, and impact. Ensure CI is green, run format/lint, and remove stray logs.

## Configuration & Known Limits
- Env: `OFM_MLX_RETRY_MAX` (max retries, default 2).
- Backend: `MLXBackend` is a stub; logits‑mask SCD will connect after MLXLLM publishes APIs.

