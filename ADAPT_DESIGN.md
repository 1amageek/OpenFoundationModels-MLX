# ADAPT: Adaptive Dynamic Assertion Protocol for Transformers

## System Overview

ADAPT (Adaptive Dynamic Assertion Protocol for Transformers) is a sophisticated constraint application system designed to ensure LLM-generated JSON conforms to specified schemas by dynamically repairing incorrect keys during generation.

## Core Problem

When LLMs generate JSON, they may produce keys that don't match the expected schema. ADAPT addresses this by:
1. **Detecting** when a key is being generated
2. **Analyzing** available valid keys from the schema
3. **Repairing** incorrect keys by modifying token probabilities
4. **Tracking** generation state for context-aware constraints

## Current Implementation Analysis

### Working Components

1. **JSON State Machine** (`JSONStateMachine`)
   - Accurately tracks JSON parsing state
   - Detects key/value boundaries
   - Handles nested structures

2. **JSON Extractor** (`JSONExtractor`)
   - Identifies JSON content in mixed text
   - Filters non-JSON tokens
   - Maintains parsing context

3. **Entropy Calculation**
   - Measures model confidence
   - Provides visual feedback (🟢 to ⚫ scale)
   - Works correctly during key generation

### Issues Identified

1. **Schema Not Being Used for Constraints**
   ```swift
   // Current: Only observes, doesn't constrain
   public func process(logits: MLXArray) -> MLXArray {
       // ... entropy calculation ...
       return logits  // Unmodified!
   }
   ```

2. **Empty Schema Problem**
   ```swift
   // When no schema provided:
   schemaNode = SchemaNode(kind: .object)  // Empty properties!
   // Results in: [No schema constraints]
   ```

3. **Missing Token-to-Key Mapping**
   - No mechanism to map partial tokens to valid keys
   - Cannot determine which tokens form valid key prefixes

## Proposed Architecture

### 1. Key Repair Pipeline

```
Token Stream → JSON Detection → Key Phase Detection → Schema Lookup →
Constraint Application → Modified Logits → Sampling
```

### 2. Component Responsibilities

#### KeyDetectionLogitProcessor (Enhanced)
- **Observe**: Track JSON structure and keys
- **Constrain**: Apply schema constraints to logits
- **Repair**: Replace invalid key tokens with valid ones

#### SchemaConstraintApplicator (New)
- Build token-to-key mappings
- Calculate key similarity scores
- Generate logit masks for constraints

#### TokenKeyMapper (New)
- Map token IDs to potential key completions
- Handle multi-token keys
- Support partial matching

### 3. Key Repair Algorithm

```swift
func applyKeyConstraints(logits: MLXArray, context: JSONContext) -> MLXArray {
    // 1. Get valid keys for current context
    let validKeys = schema.getValidKeys(at: context.path)

    // 2. Get current partial key
    let partialKey = context.currentPartialKey

    // 3. Find matching valid keys
    let matchingKeys = validKeys.filter { $0.hasPrefix(partialKey) }

    // 4. Build token mask
    var tokenMask = MLXArray.zeros(vocabSize)
    for key in matchingKeys {
        let nextTokens = getNextValidTokens(for: key, after: partialKey)
        for tokenId in nextTokens {
            tokenMask[tokenId] = 1.0
        }
    }

    // 5. Apply constraints
    if matchingKeys.isEmpty {
        // No valid continuations - find closest match
        let closestKey = findClosestKey(partialKey, in: validKeys)
        return redirectToKey(logits, target: closestKey)
    } else {
        // Apply soft constraints
        return applyMask(logits, mask: tokenMask, strength: 0.8)
    }
}
```

### 4. Token Mapping Strategy

#### Build Phase (Initialization)
```swift
struct TokenKeyMap {
    // Map from token sequences to valid keys
    var tokenToKeys: [TokenSequence: Set<String>]

    init(schema: SchemaNode, tokenizer: TokenizerAdapter) {
        // Pre-compute all valid key token sequences
        for key in schema.allKeys() {
            let tokens = tokenizer.encode(key)
            for i in 0..<tokens.count {
                let prefix = TokenSequence(tokens[0...i])
                tokenToKeys[prefix, default: []].insert(key)
            }
        }
    }
}
```

#### Lookup Phase (Runtime)
```swift
func getValidNextTokens(currentTokens: [Int32]) -> Set<Int32> {
    let sequence = TokenSequence(currentTokens)
    guard let possibleKeys = tokenToKeys[sequence] else {
        return []
    }

    var validTokens = Set<Int32>()
    for key in possibleKeys {
        let keyTokens = tokenizer.encode(key)
        if keyTokens.starts(with: currentTokens) &&
           currentTokens.count < keyTokens.count {
            validTokens.insert(keyTokens[currentTokens.count])
        }
    }
    return validTokens
}
```

### 5. Constraint Application Methods

#### Soft Constraints (Recommended)
- Boost probabilities of valid tokens
- Maintain some flexibility for model
- Temperature-based scaling

```swift
func applySoftConstraints(logits: MLXArray, validTokens: Set<Int32>) -> MLXArray {
    let temperature = 0.1  // Lower = stronger constraints
    var modifiedLogits = logits

    for tokenId in 0..<vocabSize {
        if !validTokens.contains(tokenId) {
            modifiedLogits[tokenId] *= temperature
        }
    }

    return modifiedLogits
}
```

#### Hard Constraints (Strict Mode)
- Completely mask invalid tokens
- Force valid key generation
- Zero probability for invalid tokens

```swift
func applyHardConstraints(logits: MLXArray, validTokens: Set<Int32>) -> MLXArray {
    var modifiedLogits = MLXArray.full(vocabSize, value: -Float.infinity)

    for tokenId in validTokens {
        modifiedLogits[tokenId] = logits[tokenId]
    }

    return modifiedLogits
}
```

### 6. Key Similarity and Repair

When an invalid key is detected, find the most similar valid key:

```swift
func findClosestKey(_ invalid: String, validKeys: [String]) -> String? {
    return validKeys
        .map { (key: $0, score: similarity(invalid, $0)) }
        .max(by: { $0.score < $1.score })?
        .key
}

func similarity(_ s1: String, _ s2: String) -> Float {
    // Combine multiple metrics:
    // 1. Levenshtein distance
    let editDistance = levenshteinDistance(s1, s2)

    // 2. Prefix match length
    let prefixLength = commonPrefixLength(s1, s2)

    // 3. Character overlap
    let overlap = characterOverlap(s1, s2)

    return Float(prefixLength) * 2.0 + overlap - Float(editDistance) * 0.5
}
```

## Implementation Plan

### Phase 1: Foundation (Immediate)
1. Fix schema initialization to always have properties
2. Add schema validation and warnings
3. Implement basic token-to-key mapping

### Phase 2: Constraint Application (Core)
1. Implement `applyKeyConstraints` method
2. Add soft constraint mode
3. Create similarity scoring system

### Phase 3: Advanced Features (Enhancement)
1. Multi-token lookahead
2. Context-aware constraint strength
3. Learning from corrections

### Phase 4: Optimization (Performance)
1. Cache token mappings
2. Optimize similarity calculations
3. Parallel constraint evaluation

## Configuration Options

```swift
struct ADAPTConfig {
    // Constraint strength (0.0 = none, 1.0 = strict)
    var constraintStrength: Float = 0.8

    // When to apply constraints
    var constraintMode: ConstraintMode = .duringKeyGeneration

    // Similarity threshold for key repair
    var similarityThreshold: Float = 0.6

    // Enable verbose logging
    var verbose: Bool = true

    // Show probability distributions
    var showProbabilities: Bool = true
}

enum ConstraintMode {
    case always
    case duringKeyGeneration
    case onlyWhenUncertain  // Based on entropy
    case never
}
```

## Success Metrics

1. **Key Accuracy**: % of generated keys matching schema
2. **Repair Rate**: % of invalid keys successfully corrected
3. **Generation Speed**: Impact on token/second
4. **Entropy Reduction**: Confidence improvement with constraints

## Example Usage

```swift
// Initialize with schema
let schema: [String: Any] = [
    "type": "object",
    "properties": [
        "name": ["type": "string"],
        "email": ["type": "string"],
        "age": ["type": "number"]
    ],
    "required": ["name", "email"]
]

// Create processor with ADAPT
let processor = KeyDetectionLogitProcessor(
    tokenizer: tokenizer,
    jsonSchema: schema,
    modelCard: modelCard,
    config: ADAPTConfig(
        constraintStrength: 0.9,
        constraintMode: .duringKeyGeneration
    )
)

// During generation, ADAPT will:
// 1. Detect when generating JSON keys
// 2. Apply constraints to ensure valid keys
// 3. Repair invalid keys automatically
// 4. Log all interventions for debugging
```

## Testing Strategy

### Unit Tests
- Token mapping accuracy
- Similarity scoring correctness
- Constraint application logic

### Integration Tests
- End-to-end JSON generation
- Schema compliance validation
- Performance benchmarks

### Test Cases
1. **Simple Object**: `{"name": "John", "age": 30}`
2. **Nested Objects**: Complex hierarchies
3. **Arrays**: Multiple items with consistent keys
4. **Invalid Keys**: Test repair mechanism
5. **Partial Tokens**: Multi-token key handling

## Future Enhancements

1. **Machine Learning Integration**
   - Learn from successful repairs
   - Adapt constraint strength based on model confidence
   - Personalized schema preferences

2. **Advanced Schema Support**
   - JSON Schema draft-07 full compliance
   - Custom validation rules
   - Conditional schemas

3. **Performance Optimizations**
   - GPU-accelerated similarity calculations
   - Batched constraint application
   - Incremental token mapping updates

## Conclusion

ADAPT represents a significant advancement in constrained generation, ensuring LLMs produce schema-compliant JSON while maintaining generation quality. By intercepting and modifying logits during key generation phases, ADAPT provides a robust solution for structured output generation.

The system's adaptive nature allows it to balance between strict schema enforcement and model flexibility, resulting in high-quality, valid JSON output that meets application requirements.