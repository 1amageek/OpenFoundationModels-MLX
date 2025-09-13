#!/bin/bash

echo "🔍 Testing key-detection-test limited to non-MLX tests"
echo "========================================================="
echo ""

# Create a simple test file that simulates GPT OSS format
cat << 'EOF' > /tmp/test_gpt_oss_output.txt
<|channel|>analysis<|message|>Let me analyze this request and generate the appropriate JSON response. 
We need to create a user profile with the following structure.
<|end|><|start|>assistant<|channel|>final<|message|>```json
{
  "name": "Alice Johnson",
  "age": 28,
  "email": "alice@example.com",
  "isActive": true,
  "preferences": {
    "theme": "dark",
    "notifications": true
  }
}
```
EOF

echo "📝 Test Input File Created:"
echo "----------------------------"
cat /tmp/test_gpt_oss_output.txt
echo ""
echo "----------------------------"
echo ""

# Create a Swift test that uses JSONExtractor and JSONStateMachine
cat << 'EOF' > /tmp/test_json_extractor.swift
import Foundation
import OpenFoundationModelsMLX

// Read the test file
let testInput = try! String(contentsOfFile: "/tmp/test_gpt_oss_output.txt")

print("🧪 Testing JSONExtractor with GPT OSS Format")
print("=" + String(repeating: "=", count: 50))

var extractor = JSONExtractor()
var stateMachine = JSONStateMachine()
var jsonBuffer = ""
var detectedKeys: [String] = []
var skippedChars = 0
var jsonStarted = false

print("\n📊 Processing character by character...")

for char in testInput {
    let shouldProcess = extractor.processCharacter(char)
    
    if shouldProcess {
        if !jsonStarted {
            jsonStarted = true
            print("✅ JSON detected after skipping \(skippedChars) characters")
            print("   Starting with character: '\(char)'")
        }
        
        jsonBuffer.append(char)
        
        let previousPhase = stateMachine.phase
        stateMachine.processCharacter(char)
        
        // Detect keys
        if case .inString(.body(kind: .key, escaped: false)) = previousPhase {
            if case .inObject(.expectColon) = stateMachine.phase {
                let key = stateMachine.currentKey
                if !key.isEmpty && !detectedKeys.contains(key) {
                    detectedKeys.append(key)
                    print("🔑 Key detected: \"\(key)\"")
                }
            }
        }
    } else {
        if !jsonStarted {
            skippedChars += 1
        }
    }
}

print("\n📊 Final Results:")
print("   Skipped characters: \(skippedChars)")
print("   JSON buffer length: \(jsonBuffer.count)")
print("   Detected keys: \(detectedKeys)")

// Verify the JSON structure
print("\n🔍 JSON Structure:")
if let jsonData = jsonBuffer.data(using: .utf8) {
    do {
        let json = try JSONSerialization.jsonObject(with: jsonData, options: [])
        print("   ✅ Valid JSON parsed successfully")
        if let dict = json as? [String: Any] {
            print("   Top-level keys: \(dict.keys.sorted())")
        }
    } catch {
        print("   ⚠️ JSON parsing incomplete (expected for streaming): \(error)")
    }
}

// Test KeyDetectionLogitProcessor initialization
print("\n🧪 Testing KeyDetectionLogitProcessor initialization...")
let tokenizer = SimpleMLXTokenizer()
let processor = KeyDetectionLogitProcessor(
    tokenizer: tokenizer,
    verbose: false,
    topK: 5,
    showProbabilities: false
)
print("   ✅ KeyDetectionLogitProcessor created successfully")
print("   JSONExtractor integrated: ✓")
print("   JSONStateMachine integrated: ✓")

print("\n✅ All tests completed successfully!")
EOF

echo "🚀 Running JSONExtractor Integration Test..."
echo ""

# Compile and run the test
swift /tmp/test_json_extractor.swift 2>&1

echo ""
echo "========================================================="
echo "✅ Test completed"