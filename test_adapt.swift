#!/usr/bin/env swift

import Foundation
import MLX
import MLXLMCommon
import OpenFoundationModelsMLX

// Create a simple test for ADAPT functionality
func testADAPT() {
    print("Testing ADAPT Key Constraint System...")

    // Create a schema with specific keys
    let schema: [String: Any] = [
        "type": "object",
        "properties": [
            "firstName": ["type": "string"],
            "lastName": ["type": "string"],
            "email": ["type": "string"]
        ]
    ]

    // Create a simple tokenizer
    let tokenizer = SimpleCharTokenizer()
    let modelCard = SimpleModelCard()

    let processor = KeyDetectionLogitProcessor(
        tokenizer: tokenizer,
        jsonSchema: schema,
        modelCard: modelCard,
        verbose: true,
        showProbabilities: true
    )

    // Initialize processor
    processor.prompt(MLX.zeros([1]))

    print("\n=== Simulating JSON generation ===")

    // Simulate generating: {"first
    let partialJSON = "{\"first"
    for char in partialJSON {
        let tokenId = Int32(char.asciiValue ?? 0)
        processor.didSample(token: MLXArray([tokenId]))
    }

    // Create mock logits
    let vocabSize = 256
    let logits = MLX.zeros([1, vocabSize])

    print("\n=== Processing logits with ADAPT ===")

    // Process logits - should apply constraints
    let modifiedLogits = processor.process(logits: logits)

    // Check if constraints were applied
    let originalArray = logits.asArray(Float.self)
    let modifiedArray = modifiedLogits.asArray(Float.self)

    // Check if 'N' token was boosted (for firstName)
    let nTokenId = Int(Character("N").asciiValue!)
    if modifiedArray[nTokenId] > originalArray[nTokenId] {
        print("✅ SUCCESS: Token 'N' was boosted for 'firstName' completion")
    } else {
        print("❌ FAILED: Token 'N' was not boosted")
    }

    print("\nTest completed!")
}

// Simple implementations for testing
final class SimpleCharTokenizer: TokenizerAdapter {
    func encode(_ text: String) -> [Int32] {
        return text.map { Int32($0.asciiValue ?? 0) }
    }

    func decode(_ ids: [Int32]) -> String {
        return String(ids.compactMap { id in
            guard id >= 0 && id < 128 else { return nil }
            return Character(UnicodeScalar(Int(id))!)
        })
    }

    func getVocabSize() -> Int? { return 256 }
    func fingerprint() -> String { return "simple-char" }
}

struct SimpleModelCard: ModelCard {
    let id = "test-model"

    func shouldActivateProcessor(_ raw: String, processor: any LogitProcessor) -> Bool {
        return true
    }

    func generate(from raw: String, options: GenerationOptions?) -> any MessageGenerator.Entry {
        return MessageGenerator.Response(segments: [.text(.init(content: raw))])
    }

    func prompt(transcript: Transcript, options: GenerationOptions?) -> Prompt {
        return Prompt(role: .user, content: transcript.toString())
    }
}

// Run the test
testADAPT()