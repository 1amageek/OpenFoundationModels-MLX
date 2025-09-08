# OpenFoundationModels-MLX Tests

## Running Tests with GPT-OSS Model

### Prerequisites

1. **Swift 6.2+** and Xcode 16.x or later
2. **Sufficient disk space**: ~10GB for quantized model, ~40GB for full model
3. **Apple Silicon Mac** (M1/M2/M3) with sufficient memory

### Model Options

The tests support multiple model configurations:

- `mlx-community/gpt-oss-4bit` - Quantized 4-bit version (~10GB, recommended for testing)
- `openai/gpt-oss-20b` - Full precision model (~40GB)
- Custom models via `TEST_MODEL_ID` environment variable

### Running Tests

#### Basic Test Run
```bash
swift test
```

#### With Specific Model
```bash
TEST_MODEL_ID="mlx-community/gpt-oss-4bit" swift test
```

#### Run Specific Test Suite
```bash
swift test --filter GPTOSSIntegrationTests
```

#### Skip Model Tests in CI
Tests automatically skip if `CI` environment variable is set. To force run:
```bash
CI=1 ENABLE_MODEL_TESTS=1 swift test
```

### First Run

On first run, the model will be downloaded to:
```
~/Library/Caches/huggingface/hub/
```

Download progress will be displayed in the console. This is a one-time operation.

### Test Categories

#### Integration Tests (`GPTOSSIntegrationTests`)
- **Simple text generation**: Basic prompt-response validation
- **Greedy decoding**: Deterministic generation with temperature=0
- **JSON schema generation**: Schema-constrained decoding validation
- **Tool call detection**: Function calling capability tests
- **Streaming generation**: Async stream response validation

#### Performance Tests (`GPTOSSPerformanceTests`)
- **Generation performance**: Measures response time with timeout limits

#### Unit Tests (Existing)
- TokenTrie constraint computation
- JSON state machine transitions
- Schema validation and parsing
- Tool call detection accuracy

### Environment Variables

- `TEST_MODEL_ID`: Specify which model to use for tests
- `OFM_MLX_SCHEMA_JSON`: Override schema for constrained generation
- `OFM_MLX_RETRY_MAX`: Maximum retry attempts (default: 2)
- `OFM_MLX_ENABLE_SCD`: Enable/disable schema-constrained decoding
- `CI`: Set to skip model tests in CI environments
- `ENABLE_MODEL_TESTS`: Force model tests even in CI

### Debugging

Enable verbose logging:
```bash
swift test --enable-code-coverage
```

View test results in Xcode:
```bash
xed .
# Then run tests from Xcode's Test Navigator
```

### Troubleshooting

#### Out of Memory
- Use quantized model (`mlx-community/gpt-oss-4bit`)
- Close other applications
- Increase swap space if needed

#### Download Issues
- Check network connection
- Verify Hugging Face is accessible
- Clear cache: `rm -rf ~/Library/Caches/huggingface/hub/`

#### Slow Performance
- Ensure running on Apple Silicon (not Rosetta)
- Check Activity Monitor for memory pressure
- Use smaller `maximumResponseTokens` values

### Expected Test Output

Successful run shows:
```
📥 Loading model: mlx-community/gpt-oss-4bit
✅ Generated response: Paris is the capital of France...
✅ Greedy response: 4
✅ JSON response: {"city": "Paris", "country": "France"}
✅ Tool call detected: get_weather
📝 Chunk: 1...
📝 Chunk: 2...
✅ Full streamed response: 1 2 3 4 5
⏱️ Generation took 2.5 seconds
```

All tests passed.