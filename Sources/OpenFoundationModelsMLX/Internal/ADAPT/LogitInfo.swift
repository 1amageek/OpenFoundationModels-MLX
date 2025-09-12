import Foundation
import MLX

/// Structured representation of logit information at each generation step
public struct LogitInfo: Sendable {
    /// Information about a token candidate
    public struct Candidate: Sendable, Comparable {
        public let tokenId: Int32
        public let probability: Float
        public let logit: Float
        public let text: String?
        
        public static func < (lhs: Candidate, rhs: Candidate) -> Bool {
            lhs.probability < rhs.probability
        }
        
        /// Display-friendly text with escaped special characters
        public var displayText: String {
            (text ?? "<unknown>")
                .replacingOccurrences(of: "\n", with: "\\n")
                .replacingOccurrences(of: "\t", with: "\\t")
                .replacingOccurrences(of: "\r", with: "\\r")
        }
    }
    
    /// Generation step number
    public let step: Int
    
    /// Total vocabulary size
    public let vocabSize: Int
    
    /// Top candidate tokens sorted by probability (descending)
    public let topCandidates: [Candidate]
    
    /// Entropy of the probability distribution
    public let entropy: Float
    
    /// The token that was actually selected (set after sampling)
    public var selectedToken: Candidate?
    
    /// Find a candidate by token ID
    public func candidate(for tokenId: Int32) -> Candidate? {
        topCandidates.first { $0.tokenId == tokenId }
    }
    
    /// Format candidates for display
    public func formatCandidates(limit: Int = 5) -> String {
        let displayCandidates = topCandidates.prefix(limit)
        var output = ""
        
        for (index, candidate) in displayCandidates.enumerated() {
            output += String(format: "  %d. [%5d] '%-20s' (p=%.4f, logit=%.2f)\n",
                           index + 1,
                           candidate.tokenId,
                           String(candidate.displayText.prefix(20)),
                           candidate.probability,
                           candidate.logit)
        }
        
        return output
    }
    
    /// Get entropy interpretation
    public var entropyInterpretation: String {
        switch entropy {
        case ..<1.0:
            return "Very confident (low diversity)"
        case 1.0..<3.0:
            return "Moderately confident"
        case 3.0..<5.0:
            return "Uncertain (high diversity)"
        default:
            return "Very uncertain (very high diversity)"
        }
    }
}