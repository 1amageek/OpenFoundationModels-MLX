import Foundation
import OpenFoundationModelsMLX

print("🔍 JSONExtractor and JSONStateMachine Test")
print(String(repeating: "=", count: 60))

// Test 1: JSONExtractor with complete JSON
print("\n📝 PART 1: Complete JSON Tests")
print(String(repeating: "-", count: 50))
testJSONExtractor()

// Test 2: Integration tests
print("\n📝 PART 2: Integration Tests")
print(String(repeating: "-", count: 50))
testJSONExtractorIntegration()

// Test 3: Incomplete JSON (realistic LLM scenarios)
print("\n📝 PART 3: Incomplete/Streaming JSON Tests")
print(String(repeating: "-", count: 50))
testIncompleteJSON()

print("\n" + String(repeating: "=", count: 60))
print("✅ All tests completed")