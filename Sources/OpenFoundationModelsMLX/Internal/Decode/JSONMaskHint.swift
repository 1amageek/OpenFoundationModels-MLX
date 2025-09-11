import Foundation
public struct JSONMaskHint: Sendable {
    
    public enum Mode: Sendable {
        case hard
        case soft
    }
    
    public let allow: Set<Int32>
    public let prefer: Set<Int32>
    public let mode: Mode
    
    public init(allow: Set<Int32>, prefer: Set<Int32> = [], mode: Mode = .hard) {
        self.allow = allow
        self.prefer = prefer
        self.mode = mode
    }
}


public struct JSONMaskHintGenerator: Sendable {
    
    private let specialTokens: MLXLLMTokenizer.SpecialTokens
    private let includeWhitespace: Bool
    
    public init(specialTokens: MLXLLMTokenizer.SpecialTokens, includeWhitespace: Bool = true) {
        self.specialTokens = specialTokens
        self.includeWhitespace = includeWhitespace
    }
    
    func maskHint(
        for state: JSONStateMachine,
        tokenTrie: TokenTrie? = nil,
        tokenPath: TokenTrie.Path? = nil
    ) -> JSONMaskHint? {
        
        let phase = state.phase
        
        let ws = includeWhitespace ? specialTokens.whitespaceTokens : Set<Int32>()
        
        switch phase {
        case .root:
            let allowed = specialTokens.quoteTokens
                .union(specialTokens.braceOpenTokens)
                .union(specialTokens.bracketOpenTokens)
                .union(ws)
            return JSONMaskHint(allow: allowed, mode: .soft)
            
        case .inObject(let objPhase):
            return maskHintForObject(objPhase: objPhase, ws: ws, tokenTrie: tokenTrie, tokenPath: tokenPath)
            
        case .inArray(let arrPhase):
            return maskHintForArray(arrPhase: arrPhase, ws: ws)
            
        case .inString(let strPhase):
            return maskHintForString(strPhase: strPhase, tokenTrie: tokenTrie, tokenPath: tokenPath)
            
        case .inNumber, .inLiteral:
            return nil
            
        case .done:
            return JSONMaskHint(allow: [], mode: .hard)
            
        case .error:
            return JSONMaskHint(allow: [])
        }
    }
    
    
    private func maskHintForObject(
        objPhase: JSONStateMachine.ObjectPhase,
        ws: Set<Int32>,
        tokenTrie: TokenTrie?,
        tokenPath: TokenTrie.Path?
    ) -> JSONMaskHint? {
        
        switch objPhase {
        case .expectKeyOrEnd:
            let allowed = specialTokens.quoteTokens
                .union(specialTokens.braceCloseTokens)
                .union(ws)
            return JSONMaskHint(allow: allowed)
            
        case .expectKey:
            let allowed = specialTokens.quoteTokens
                .union(ws)
            return JSONMaskHint(allow: allowed)
            
        case .expectColon:
            let allowed = specialTokens.colonTokens.union(ws)
            return JSONMaskHint(allow: allowed)
            
        case .expectValueStart:
            let allowed = specialTokens.quoteTokens
                .union(specialTokens.braceOpenTokens)
                .union(specialTokens.bracketOpenTokens)
                .union(ws)
            return JSONMaskHint(allow: allowed, mode: .soft)
            
        case .afterValue:
            let allowed = specialTokens.commaTokens
                .union(specialTokens.braceCloseTokens)
                .union(ws)
            return JSONMaskHint(allow: allowed)
        }
    }
    
    
    private func maskHintForArray(
        arrPhase: JSONStateMachine.ArrayPhase,
        ws: Set<Int32>
    ) -> JSONMaskHint? {
        
        switch arrPhase {
        case .expectValue:
            let allowed = specialTokens.quoteTokens
                .union(specialTokens.braceOpenTokens)
                .union(specialTokens.bracketOpenTokens)
                .union(specialTokens.bracketCloseTokens)
                .union(ws)
            return JSONMaskHint(allow: allowed, mode: .soft)
            
        case .afterValue:
            let allowed = specialTokens.commaTokens
                .union(specialTokens.bracketCloseTokens)
                .union(ws)
            return JSONMaskHint(allow: allowed)
        }
    }
    
    
    private func maskHintForString(
        strPhase: JSONStateMachine.StringPhase,
        tokenTrie: TokenTrie?,
        tokenPath: TokenTrie.Path?
    ) -> JSONMaskHint? {
        
        switch strPhase {
        case .start, .end:
            return nil
        case .body(let kind, _):
            if kind == .key {
                if let trie = tokenTrie, let path = tokenPath {
                    var allowed = trie.getAllowedTokens(for: path)
                    allowed.formUnion(specialTokens.backslashTokens)
                    
                    if path.isAtTerminal() {
                        allowed.formUnion(specialTokens.quoteTokens)
                    }
                    
                    if allowed.isEmpty {
                        allowed = specialTokens.quoteTokens.union(specialTokens.backslashTokens)
                    }
                    
                    return JSONMaskHint(allow: allowed)
                }
            }
            return nil
        }
    }
}


extension JSONMaskHintGenerator {
    
    public static func forSchemaConstrainedDecoding(
        specialTokens: MLXLLMTokenizer.SpecialTokens,
        includeWhitespace: Bool = false
    ) -> JSONMaskHintGenerator {
        return JSONMaskHintGenerator(
            specialTokens: specialTokens,
            includeWhitespace: includeWhitespace
        )
    }
    
    func isViolation(
        tokenID: Int32,
        state: JSONStateMachine,
        tokenTrie: TokenTrie? = nil,
        tokenPath: TokenTrie.Path? = nil
    ) -> Bool {
        guard let hint = maskHint(for: state, tokenTrie: tokenTrie, tokenPath: tokenPath) else {
            return false
        }
        
        if hint.mode == .hard {
            return !hint.allow.contains(tokenID)
        } else {
            return false
        }
    }
}