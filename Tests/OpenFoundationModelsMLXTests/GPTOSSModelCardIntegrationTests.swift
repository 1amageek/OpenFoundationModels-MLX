import Testing
import Foundation
@testable import OpenFoundationModelsMLX
import OpenFoundationModelsMLXGPT

@Suite("GPTOSSModelCard Integration Tests")
struct GPTOSSModelCardIntegrationTests {

    @Test("HarmonyParser correctly extracts channels")
    func harmonyParsing() {
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
