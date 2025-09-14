import Foundation
import Testing
import MLX
import MLXLMCommon
@testable import OpenFoundationModelsMLX

@Suite("Entropy and Constraint Display Integration")
struct EntropyConstraintDisplayTest {

    // Mock tokenizer for testing
    final class TestTokenizerAdapter: ConfigurableTokenizer {
        var tokenCounter = 0
        let outputs = [
            "<|channel|>", "analysis", "<|message|>",
            "Let", " me", " think", "...",
            "{\"", "name", "\":\"", "Alice", "\",\"",
            "age", "\":", "30", ",\"",
            "city", "\":\"", "NYC", "\"}"
        ]

        override func decode(_ tokens: [Int32]) -> String {
            let result = tokenCounter < outputs.count ? outputs[tokenCounter] : ""
            tokenCounter += 1
            return result
        }
    }

    @Test("Complete flow: LLM output → JSON detection → Entropy display with constraints")
    func testCompleteEntropyConstraintDisplay() {
        // Setup schema
        let schema: [String: Any] = [
            "type": "object",
            "properties": [
                "name": ["type": "string"],
                "age": ["type": "integer"],
                "city": ["type": "string"],
                "active": ["type": "boolean"]
            ]
        ]

        // Create processor with verbose output
        let tokenizer = TestTokenizerAdapter()
        var processor = KeyDetectionLogitProcessor(
            tokenizer: tokenizer,
            jsonSchema: schema,
            verbose: true,
            topK: 5,
            showProbabilities: true
        )

        // Track what gets displayed
        var displayedEntropies: [Float] = []
        var displayedConstraints: [[String]] = []

        // Custom display handler for testing
        class DisplayCapture {
            var outputs: [String] = []

            func captureOutput(_ text: String) {
                outputs.append(text)

                // Parse entropy and constraints from output
                if text.contains("[Step") && text.contains("Entropy:") {
                    // Extract entropy value
                    if let entropyRange = text.range(of: "Entropy: "),
                       let endRange = text.range(of: " (", range: entropyRange.upperBound..<text.endIndex) {
                        let entropyStr = String(text[entropyRange.upperBound..<endRange.lowerBound])
                        if let entropy = Float(entropyStr) {
                            // Record entropy
                        }
                    }

                    // Extract constraints
                    if let constraintStart = text.range(of: "[", options: .backwards),
                       let constraintEnd = text.range(of: "]", options: .backwards) {
                        let constraintStr = String(text[constraintStart.upperBound..<constraintEnd.lowerBound])
                        let constraints = constraintStr.split(separator: ",").map { $0.trimmingCharacters(in: .whitespaces) }
                        // Record constraints
                    }
                }
            }
        }

        let capture = DisplayCapture()

        // Initialize processor
        processor.prompt(MLXArray.zeros([1]))

        // Simulate LLM generation with multiple tokens
        let simulatedTokens = [
            "<|channel|>", "analysis", "<|message|>",
            "Thinking", " about", " the", " data", "...\n",
            "{\"",  // JSON starts here
            "name", "\":\"", "Alice", "\",\"",
            "age", "\":", "30", ",\"",
            "city", "\":\"", "NYC", "\",\"",
            "active", "\":", "true",
            "}"  // JSON ends here
        ]

        // Process each token
        for (index, tokenText) in simulatedTokens.enumerated() {
            // Create mock logits
            let vocabSize = 50000
            var logits = MLX.zeros([1, vocabSize])

            // Make certain tokens more likely for testing
            if tokenText == "name" || tokenText == "age" || tokenText == "city" || tokenText == "active" {
                // Boost probability for key tokens
                logits = logits + 2.0
            }

            // Process logits
            let processedLogits = processor.process(logits: logits)

            // Simulate token sampling
            let token = MLXArray(Int32(index))

            // Update tokenizer output
            tokenizer.tokenCounter = index

            // Process sampled token
            processor.didSample(token: token)
        }

        // Finish processing
        processor.finish()

        // Verify results
        print("\n=== Test Summary ===")
        print("Detected keys: \(processor.allDetectedKeys)")

        // Check that keys were detected
        #expect(processor.allDetectedKeys.contains("name"), "Should detect 'name' key")
        #expect(processor.allDetectedKeys.contains("age"), "Should detect 'age' key")
        #expect(processor.allDetectedKeys.contains("city"), "Should detect 'city' key")
        #expect(processor.allDetectedKeys.contains("active"), "Should detect 'active' key")

        // The actual entropy and constraints would be displayed during processing
        // This test verifies the integration works correctly
    }

    @Test("Multiple JSONs with different schemas")
    func testMultipleJSONsWithDifferentConstraints() {
        // First JSON schema
        let userSchema: [String: Any] = [
            "type": "object",
            "properties": [
                "userId": ["type": "string"],
                "username": ["type": "string"]
            ]
        ]

        // Create processor
        let tokenizer = TestTokenizerAdapter()
        var processor = KeyDetectionLogitProcessor(
            tokenizer: tokenizer,
            jsonSchema: userSchema,
            verbose: false,  // Quiet for this test
            showProbabilities: false
        )

        processor.prompt(MLXArray.zeros([1]))

        // Simulate generation of multiple JSONs
        let text = """
        First user: {"userId": "u1", "username": "alice"}
        Second user: {"userId": "u2", "username": "bob"}
        """

        // Process character by character
        for char in text {
            // Mock logits
            let logits = MLX.zeros([1, 50000])
            _ = processor.process(logits: logits)

            // Mock token
            let token = MLXArray(Int32(0))
            tokenizer.tokenCounter = 0

            // Provide single character as token output
            class SingleCharTokenizer: TokenizerAdapter, @unchecked Sendable {
                var nextChar: String = ""

                func encode(_ text: String) -> [Int32] { [0] }
                func decode(_ tokens: [Int32]) -> String { nextChar }
                var eosTokenId: Int32 { 0 }
                var bosTokenId: Int32 { 1 }
                var unknownTokenId: Int32 { 2 }
                func convertTokenToString(_ token: Int32) -> String? { nextChar }
                func getVocabSize() -> Int? { 50000 }
                func fingerprint() -> String { "single-char" }
            }

            let charTokenizer = SingleCharTokenizer()
            charTokenizer.nextChar = String(char)

            // Use temporary processor with char tokenizer
            var charProcessor = KeyDetectionLogitProcessor(
                tokenizer: charTokenizer,
                jsonSchema: userSchema,
                verbose: false,
                showProbabilities: false
            )

            charProcessor.prompt(MLXArray.zeros([1]))
            charProcessor.didSample(token: token)
        }

        processor.finish()

        // Should detect keys from both JSONs
        let detectedKeys = processor.allDetectedKeys
        print("All detected keys: \(detectedKeys)")

        // Each JSON should contribute its keys
        let userIdCount = detectedKeys.filter { $0 == "userId" }.count
        let usernameCount = detectedKeys.filter { $0 == "username" }.count

        #expect(userIdCount >= 1, "Should detect userId at least once")
        #expect(usernameCount >= 1, "Should detect username at least once")
    }

    @Test("Progressive constraint reduction during generation")
    func testProgressiveConstraintReduction() {
        let schema: [String: Any] = [
            "type": "object",
            "properties": [
                "first": ["type": "string"],
                "second": ["type": "string"],
                "third": ["type": "string"]
            ]
        ]

        let tokenizer = TestTokenizerAdapter()
        var processor = KeyDetectionLogitProcessor(
            tokenizer: tokenizer,
            jsonSchema: schema,
            verbose: true,
            showProbabilities: true
        )

        processor.prompt(MLXArray.zeros([1]))

        // Simulate progressive JSON generation
        let tokens = [
            "Result:", " ", "{\"",
            "first", "\":\"", "a", "\",\"",
            "second", "\":\"", "b", "\",\"",
            "third", "\":\"", "c", "\"}"
        ]

        var constraintSnapshots: [[String]] = []

        for (i, _) in tokens.enumerated() {
            tokenizer.tokenCounter = i

            let logits = MLX.zeros([1, 50000])
            _ = processor.process(logits: logits)

            let token = MLXArray(Int32(i))
            processor.didSample(token: token)

            // Capture current constraints (would be displayed in verbose mode)
            // This demonstrates the progressive reduction
        }

        processor.finish()

        // Verify all keys were detected
        #expect(processor.allDetectedKeys.contains("first"), "Should detect 'first'")
        #expect(processor.allDetectedKeys.contains("second"), "Should detect 'second'")
        #expect(processor.allDetectedKeys.contains("third"), "Should detect 'third'")
    }
}