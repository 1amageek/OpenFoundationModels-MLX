import Testing
@testable import OpenFoundationModelsMLX

@Suite struct ChatClientSmokeTests {
    @Test func createReturnsText() async throws {
        let client = MLXChatClient()
        let req = MLXChatClient.Request(
            modelID: "mlx-community/Mistral-7B-Instruct-v0.3-4bit",
            messages: [ChatMessage(role: .user, content: "hello")]
        )
        let res = try await client.create(req)
        #expect(!res.choices.isEmpty)
    }
}

