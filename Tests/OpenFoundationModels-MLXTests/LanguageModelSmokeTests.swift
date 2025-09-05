import Testing
@testable import OpenFoundationModelsMLX

@Suite struct LanguageModelSmokeTests {
    @Test func availability() async throws {
        let lm = try await MLXLanguageModel(modelID: "mlx-community/Mistral-7B-Instruct-v0.3-4bit")
        #expect(lm.isAvailable == true)
    }
}

