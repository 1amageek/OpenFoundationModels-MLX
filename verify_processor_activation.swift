#!/usr/bin/env swift

// Simple script to verify processor activation logic

struct MockLogitProcessor {}

// Simulate GPTOSSModelCard behavior
func shouldActivateProcessor(_ raw: String, processor: Any) -> Bool {
    // Check if processor is KeyDetectionLogitProcessor (mock)
    let isKeyDetector = true // Simulate this check
    
    if isKeyDetector {
        // Find the last occurrence of "<|channel|>" 
        if let lastChannelRange = raw.range(of: "<|channel|>", options: .backwards) {
            let afterChannel = String(raw[lastChannelRange.upperBound...])
            if afterChannel.hasPrefix("final") {
                return true
            } else if afterChannel.hasPrefix("analysis") {
                return false
            }
        }
        // No channel found, don't activate
        return false
    }
    
    // For non-KeyDetectionLogitProcessor, always activate
    return true
}

// Test cases
let testCases: [(String, String, Bool)] = [
    ("Analysis channel only", 
     "<|start|>assistant<|channel|>analysis<|message|>Let me think about this", 
     false),
    
    ("Final channel only", 
     "<|start|>assistant<|channel|>final<|message|>{", 
     true),
    
    ("Mixed - final last", 
     """
     <|start|>assistant<|channel|>analysis<|message|>Thinking...
     <|end|>
     <|start|>assistant<|channel|>final<|message|>{
     """, 
     true),
    
    ("Mixed - analysis last", 
     """
     <|start|>assistant<|channel|>final<|message|>{}
     <|end|>
     <|start|>assistant<|channel|>analysis<|message|>More thinking
     """, 
     false),
    
    ("No channel", 
     "Just some text without channels", 
     false),
]

print("Testing GPTOSSModelCard processor activation logic:")
print("=" * 50)

var passed = 0
var failed = 0

for (description, input, expected) in testCases {
    let result = shouldActivateProcessor(input, processor: MockLogitProcessor())
    let status = result == expected ? "✅ PASS" : "❌ FAIL"
    
    if result == expected {
        passed += 1
    } else {
        failed += 1
    }
    
    print("\n\(status): \(description)")
    print("  Input: \(input.prefix(50))...")
    print("  Expected: \(expected), Got: \(result)")
}

print("\n" + "=" * 50)
print("Results: \(passed) passed, \(failed) failed")

if failed == 0 {
    print("✅ All tests passed!")
} else {
    print("❌ Some tests failed")
}