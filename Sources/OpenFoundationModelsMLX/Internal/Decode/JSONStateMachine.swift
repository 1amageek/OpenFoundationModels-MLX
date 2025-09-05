import Foundation

// JSONの鍵出力位置のみを扱う簡易状態機械（トークン単位版）。
// 将来的にtokenizerのpiece情報と連携して厳密化するが、
// ここではAPIスケルトンと最小限の更新点のみを持つ。
struct JSONStateMachine: Sendable {
    enum Phase: Sendable { case outside, inKey, expectColon }
    private(set) var phase: Phase = .outside

    mutating func onPieceEmitted(_ piece: String) {
        // 文字列ベースの素朴な遷移（将来、token pieceの種別で厳密化）
        switch phase {
        case .outside:
            if piece == "\"" { phase = .inKey }
        case .inKey:
            if piece == "\"" { phase = .expectColon }
        case .expectColon:
            if piece.trimmingCharacters(in: .whitespacesAndNewlines) == ":" { phase = .outside }
        }
    }
}

