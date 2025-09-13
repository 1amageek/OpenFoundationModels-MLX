import Testing
import Foundation
import MLX
import MLXLMCommon
@testable import OpenFoundationModelsMLX

@Suite("Processor Activation Tests")
struct ProcessorActivationTests {
    
    @Test("GPTOSSModelCard activates KeyDetectionLogitProcessor only in final channel")
    func gptOSSActivation() {
        let card = GPTOSSModelCard(id: "test-model")
        
        // Create a mock processor for testing
        final class MockProcessor: LogitProcessor {
            func prompt(_ prompt: MLXArray) {}
            func process(logits: MLXArray) -> MLXArray { logits }
            func didSample(token: MLXArray) {}
        }
        
        // Create a mock tokenizer for KeyDetectionLogitProcessor
        final class MockTokenizer: TokenizerAdapter, @unchecked Sendable {
            func encode(_ text: String) -> [Int32] { [] }
            func decode(_ ids: [Int32]) -> String { "" }
            func getVocabSize() -> Int? { 50000 }
            func fingerprint() -> String { "mock" }
            var eosTokenId: Int32? { 0 }
            var bosTokenId: Int32? { 1 }
        }
        
        let mockProcessor = MockProcessor()
        let keyDetectionProcessor = KeyDetectionLogitProcessor(tokenizer: MockTokenizer(), verbose: false)
        
        // Test activation in analysis channel
        let analysisText = "<|start|>assistant<|channel|>analysis<|message|>Let me think about this"
        #expect(!card.shouldActivateProcessor(analysisText, processor: keyDetectionProcessor))
        
        // Test activation in final channel
        let finalText = "<|start|>assistant<|channel|>final<|message|>{"
        #expect(card.shouldActivateProcessor(finalText, processor: keyDetectionProcessor))
        
        // Test with mixed channels (should check last one)
        let mixedText = """
        <|start|>assistant<|channel|>analysis<|message|>Thinking...
        <|end|>
        <|start|>assistant<|channel|>final<|message|>{
        """
        #expect(card.shouldActivateProcessor(mixedText, processor: keyDetectionProcessor))
        
        // Test with analysis after final (should not activate)
        let reversedText = """
        <|start|>assistant<|channel|>final<|message|>{}
        <|end|>
        <|start|>assistant<|channel|>analysis<|message|>More thinking
        """
        #expect(!card.shouldActivateProcessor(reversedText, processor: keyDetectionProcessor))
        
        // Test activation for non-KeyDetectionLogitProcessor
        #expect(card.shouldActivateProcessor(analysisText, processor: mockProcessor))
        #expect(card.shouldActivateProcessor(finalText, processor: mockProcessor))
    }
    
    @Test("LlamaModelCard always activates processors")
    func llamaActivation() {
        let card = LlamaModelCard(id: "test-llama")
        
        final class MockProcessor: LogitProcessor {
            func prompt(_ prompt: MLXArray) {}
            func process(logits: MLXArray) -> MLXArray { logits }
            func didSample(token: MLXArray) {}
        }
        
        let processor = MockProcessor()
        
        // Should always return true for any text
        #expect(card.shouldActivateProcessor("", processor: processor))
        #expect(card.shouldActivateProcessor("Some text", processor: processor))
        #expect(card.shouldActivateProcessor("{\"key\": \"value\"}", processor: processor))
    }
    
    @Test("HarmonyParser correctly extracts channels")
    func harmonyParsing() {
        // HarmonyParser is a struct with static methods, not a class
        
        // Test single channel
        let singleChannel = "<|channel|>final<|message|>{\"key\": \"value\"}"
        let singleResult = HarmonyParser.parse(singleChannel)
        #expect(singleResult.final == "{\"key\": \"value\"}")
        #expect(singleResult.analysis == nil)
        
        // Test multiple channels
        let multiChannel = """
        <|channel|>analysis<|message|>Let me analyze this request.
        <|end|>
        <|start|>assistant<|channel|>final<|message|>{"result": "done"}
        """
        let multiResult = HarmonyParser.parse(multiChannel)
        #expect(multiResult.analysis == "Let me analyze this request.")
        #expect(multiResult.final == "{\"result\": \"done\"}")
        
        // Test with nested JSON
        let nestedJSON = """
        <|channel|>final<|message|>{"user": {"name": "John", "age": 30}}
        """
        let nestedResult = HarmonyParser.parse(nestedJSON)
        #expect(nestedResult.final == "{\"user\": {\"name\": \"John\", \"age\": 30}}")
    }
    
    @Test("GPTOSSModelCard generate method extracts final channel")
    func gptOSSGenerate() {
        let card = GPTOSSModelCard(id: "test-model")
        
        // Test with only final channel
        let onlyFinal = "<|channel|>final<|message|>Just a response"
        let entry1 = card.generate(from: onlyFinal, options: nil)
        if case .response(let response) = entry1,
           case .text(let segment) = response.segments.first {
            #expect(segment.content == "Just a response")
        } else {
            Issue.record("Expected text response")
        }
        
        // Test with analysis and final channels
        let withAnalysis = """
        <|channel|>analysis<|message|>Thinking about the problem...
        <|end|>
        <|start|>assistant<|channel|>final<|message|>Here's the answer
        """
        let entry2 = card.generate(from: withAnalysis, options: nil)
        if case .response(let response) = entry2,
           case .text(let segment) = response.segments.first {
            #expect(segment.content == "Here's the answer")
        } else {
            Issue.record("Expected text response from final channel")
        }
        
        // Test with JSON in final channel
        let withJSON = """
        <|channel|>analysis<|message|>Let me generate some JSON
        <|end|>
        <|start|>assistant<|channel|>final<|message|>{"key": "value", "number": 42}
        """
        let entry3 = card.generate(from: withJSON, options: nil)
        if case .response(let response) = entry3,
           case .text(let segment) = response.segments.first {
            #expect(segment.content == "{\"key\": \"value\", \"number\": 42}")
        } else {
            Issue.record("Expected JSON response from final channel")
        }
    }
}