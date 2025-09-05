import Foundation

// SchemaConstrainedDecoder (SCD) — placeholder for a decoding-time constraint
// mechanism that ensures JSON keys adhere to a known schema. The concrete
// implementation will wire into the token sampling loop in MLXChatEngine.
enum SchemaConstrainedDecoder {
    struct State: Sendable {
        var inKey: Bool = false
        var candidateZeroCount: Int = 0
    }

    enum Decision: Sendable { case ok, noKeyCandidate }

    static func reset() -> State { .init() }

    static func onTextAppended(_ text: String, state: inout State) {
        // Minimal state transitions; real implementation will use a JSON
        // state-machine to track positions and key/value boundaries.
        if text.contains("\"") { state.inKey.toggle() }
    }

    static func decideForNextToken(
        allowedKeyTokens: Bool,
        state: inout State
    ) -> Decision {
        if allowedKeyTokens == false && state.inKey {
            state.candidateZeroCount += 1
            if state.candidateZeroCount >= 2 { return .noKeyCandidate }
        } else {
            state.candidateZeroCount = 0
        }
        return .ok
    }
}

