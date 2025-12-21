import Testing
import Foundation
@testable import OpenFoundationModelsMLXGemma

@Suite("FunctionGemma Parser Tests")
struct FunctionGemmaParserTests {

    // MARK: - Function Call Parsing

    @Test("Parses simple function call")
    func parseSimpleFunctionCall() {
        let output = "<start_function_call>call:get_current_temperature{location:<escape>London<escape>}<end_function_call>"

        let result = FunctionGemmaParser.parseFunctionCall(output)

        #expect(result != nil)
        #expect(result?.name == "get_current_temperature")

        // Check JSON structure
        if let json = result?.arguments,
           let data = json.data(using: .utf8),
           let dict = try? JSONSerialization.jsonObject(with: data) as? [String: Any] {
            #expect(dict["location"] as? String == "London")
        } else {
            Issue.record("Failed to parse arguments as JSON")
        }
    }

    @Test("Parses function call with multiple parameters")
    func parseMultipleParams() {
        let output = "<start_function_call>call:book_flight{from:<escape>NYC<escape>,to:<escape>LAX<escape>,date:<escape>2024-01-15<escape>}<end_function_call>"

        let result = FunctionGemmaParser.parseFunctionCall(output)

        #expect(result != nil)
        #expect(result?.name == "book_flight")

        if let json = result?.arguments,
           let data = json.data(using: .utf8),
           let dict = try? JSONSerialization.jsonObject(with: data) as? [String: Any] {
            #expect(dict["from"] as? String == "NYC")
            #expect(dict["to"] as? String == "LAX")
            #expect(dict["date"] as? String == "2024-01-15")
        } else {
            Issue.record("Failed to parse arguments as JSON")
        }
    }

    @Test("Parses function call with no parameters")
    func parseNoParams() {
        let output = "<start_function_call>call:get_current_time{}<end_function_call>"

        let result = FunctionGemmaParser.parseFunctionCall(output)

        #expect(result != nil)
        #expect(result?.name == "get_current_time")
        #expect(result?.arguments == "{}")
    }

    @Test("Parses function call with text before")
    func parseWithPrecedingText() {
        let output = "Let me check the weather for you. <start_function_call>call:get_weather{city:<escape>Paris<escape>}<end_function_call>"

        let result = FunctionGemmaParser.parseFunctionCall(output)

        #expect(result != nil)
        #expect(result?.name == "get_weather")
    }

    @Test("Parses function call with text after")
    func parseWithFollowingText() {
        let output = "<start_function_call>call:search{query:<escape>restaurants<escape>}<end_function_call><end_of_turn>"

        let result = FunctionGemmaParser.parseFunctionCall(output)

        #expect(result != nil)
        #expect(result?.name == "search")
    }

    @Test("Returns nil for non-function output")
    func parseNonFunctionOutput() {
        let output = "I can help you with that question about the weather."

        let result = FunctionGemmaParser.parseFunctionCall(output)

        #expect(result == nil)
    }

    @Test("Returns nil for incomplete function call")
    func parseIncompleteFunctionCall() {
        let output = "<start_function_call>call:get_weather{city:<escape>Tokyo"

        let result = FunctionGemmaParser.parseFunctionCall(output)

        #expect(result == nil)
    }

    // MARK: - Edge Cases

    @Test("Parses value containing comma")
    func parseValueWithComma() {
        let output = "<start_function_call>call:search_location{address:<escape>123 Main St, New York, NY<escape>}<end_function_call>"

        let result = FunctionGemmaParser.parseFunctionCall(output)

        #expect(result != nil)
        #expect(result?.name == "search_location")

        if let json = result?.arguments,
           let data = json.data(using: .utf8),
           let dict = try? JSONSerialization.jsonObject(with: data) as? [String: Any] {
            #expect(dict["address"] as? String == "123 Main St, New York, NY")
        } else {
            Issue.record("Failed to parse arguments as JSON")
        }
    }

    @Test("Parses value containing colon")
    func parseValueWithColon() {
        let output = "<start_function_call>call:set_alarm{time:<escape>10:30 AM<escape>}<end_function_call>"

        let result = FunctionGemmaParser.parseFunctionCall(output)

        #expect(result != nil)
        #expect(result?.name == "set_alarm")

        if let json = result?.arguments,
           let data = json.data(using: .utf8),
           let dict = try? JSONSerialization.jsonObject(with: data) as? [String: Any] {
            #expect(dict["time"] as? String == "10:30 AM")
        } else {
            Issue.record("Failed to parse arguments as JSON")
        }
    }

    @Test("Parses multiple values with commas and colons")
    func parseMultipleComplexValues() {
        let output = "<start_function_call>call:create_event{title:<escape>Meeting: Project Review<escape>,location:<escape>123 Main St, Suite 100<escape>,time:<escape>14:00<escape>}<end_function_call>"

        let result = FunctionGemmaParser.parseFunctionCall(output)

        #expect(result != nil)
        #expect(result?.name == "create_event")

        if let json = result?.arguments,
           let data = json.data(using: .utf8),
           let dict = try? JSONSerialization.jsonObject(with: data) as? [String: Any] {
            #expect(dict["title"] as? String == "Meeting: Project Review")
            #expect(dict["location"] as? String == "123 Main St, Suite 100")
            #expect(dict["time"] as? String == "14:00")
        } else {
            Issue.record("Failed to parse arguments as JSON")
        }
    }

    // MARK: - Detection Methods

    @Test("Detects function call presence")
    func detectsFunctionCall() {
        let withCall = "<start_function_call>call:test{}<end_function_call>"
        let withoutCall = "Just some regular text"

        #expect(FunctionGemmaParser.containsFunctionCall(withCall) == true)
        #expect(FunctionGemmaParser.containsFunctionCall(withoutCall) == false)
    }

    @Test("Detects partial function call")
    func detectsPartialFunctionCall() {
        let partial = "<start_function_call>call:test{param:value"
        let complete = "<start_function_call>call:test{}<end_function_call>"

        #expect(FunctionGemmaParser.isStartingFunctionCall(partial) == true)
        #expect(FunctionGemmaParser.isStartingFunctionCall(complete) == false)
    }

    @Test("Extracts text before function call")
    func extractsTextBefore() {
        let output = "Here is your answer: <start_function_call>call:test{}<end_function_call>"

        let textBefore = FunctionGemmaParser.textBeforeFunctionCall(output)

        #expect(textBefore == "Here is your answer:")
    }

    @Test("Returns nil when no text before function call")
    func noTextBefore() {
        let output = "<start_function_call>call:test{}<end_function_call>"

        let textBefore = FunctionGemmaParser.textBeforeFunctionCall(output)

        #expect(textBefore == nil)
    }
}

@Suite("FunctionGemma ModelCard Tests")
struct FunctionGemmaModelCardTests {

    @Test("Creates prompt with tool definitions")
    func promptWithTools() {
        let card = FunctionGemmaModelCard(id: "test-model")

        // The prompt method requires a Transcript with tools
        // This test verifies the card initializes correctly
        #expect(card.id == "test-model")
        #expect(card.params.maxTokens == 128)
    }

    @Test("Default model ID is set")
    func defaultModelId() {
        let card = FunctionGemmaModelCard()

        #expect(card.id == "mlx-community/functiongemma-270m-it-bf16")
    }

    @Test("Generates response entry for plain text")
    func generatePlainText() {
        let card = FunctionGemmaModelCard()
        let raw = "Hello, I can help you with that.<end_of_turn>"

        let entry = card.generate(from: raw, options: nil)

        if case .response(let response) = entry,
           case .text(let segment) = response.segments.first {
            #expect(segment.content == "Hello, I can help you with that.")
        } else {
            Issue.record("Expected text response")
        }
    }

    @Test("Generates toolCalls entry for function call")
    func generateFunctionCall() {
        let card = FunctionGemmaModelCard()
        let raw = "<start_function_call>call:get_weather{city:<escape>Tokyo<escape>}<end_function_call>"

        let entry = card.generate(from: raw, options: nil)

        // Verify we get a toolCalls entry (calls property is package-level, so we can't check contents)
        if case .toolCalls(_) = entry {
            // Successfully detected as tool call
        } else {
            Issue.record("Expected toolCalls entry")
        }
    }
}
