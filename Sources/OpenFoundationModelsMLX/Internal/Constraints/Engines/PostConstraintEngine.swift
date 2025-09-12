import Foundation
import MLXLMCommon
import Tokenizers

public final class PostConstraintEngine: ConstraintEngine, @unchecked Sendable {
    public let mode: ConstraintMode = .post
    private let internalValidator: any JSONValidatorProtocol
    private let internalRepairer: (any JSONRepairer)?
    
    public init(
        validator: (any JSONValidatorProtocol)? = nil,
        repairer: (any JSONRepairer)? = nil
    ) {
        self.internalValidator = validator ?? SharedJSONValidator()
        self.internalRepairer = repairer
    }
    
    public func prepare(schema: SchemaNode?, tokenizer: any Tokenizer) async throws {
    }
    
    public func softPrompt(for schema: SchemaNode?) -> String? {
        return nil
    }
    
    public func logitProcessors() async -> [LogitProcessor] {
        return []
    }
    
    public func validator() -> (any JSONValidatorProtocol)? {
        return internalValidator
    }
    
    public func repairer() -> (any JSONRepairer)? {
        return internalRepairer
    }
}

public final class HeuristicJSONRepairer: JSONRepairer, @unchecked Sendable {
    public init() {}
    
    public func repair(_ invalidJSON: String, error: ValidationError, schema: SchemaNode?) async -> String? {
        var repaired = invalidJSON
        
        repaired = repaired.trimmingCharacters(in: .whitespacesAndNewlines)
        
        if !repaired.hasPrefix("{") && repaired.contains("{") {
            if let range = repaired.range(of: "{") {
                repaired = String(repaired[range.lowerBound...])
            }
        }
        
        if !repaired.hasSuffix("}") && repaired.contains("}") {
            if let range = repaired.range(of: "}", options: .backwards) {
                repaired = String(repaired[...range.upperBound])
            }
        }
        
        let openBraces = repaired.filter { $0 == "{" }.count
        let closeBraces = repaired.filter { $0 == "}" }.count
        if openBraces > closeBraces {
            repaired += String(repeating: "}", count: openBraces - closeBraces)
        }
        
        let openBrackets = repaired.filter { $0 == "[" }.count
        let closeBrackets = repaired.filter { $0 == "]" }.count
        if openBrackets > closeBrackets {
            repaired += String(repeating: "]", count: openBrackets - closeBrackets)
        }
        
        repaired = repaired.replacingOccurrences(of: ",}", with: "}")
        repaired = repaired.replacingOccurrences(of: ",]", with: "]")
        
        if let lastCommaRange = repaired.range(of: ",", options: .backwards),
           let nextChar = repaired[lastCommaRange.upperBound...].first,
           nextChar == "}" || nextChar == "]" {
            repaired.removeSubrange(lastCommaRange)
        }
        
        if let data = repaired.data(using: .utf8),
           (try? JSONSerialization.jsonObject(with: data, options: [])) != nil {
            return repaired
        }
        
        return nil
    }
    
    public func canRepair(_ error: ValidationError) -> Bool {
        return true
    }
}