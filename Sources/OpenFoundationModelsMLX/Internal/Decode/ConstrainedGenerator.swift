import Foundation

// Constrained generator that orchestrates Schema-Constrained Decoding
// using TokenTrie for strict JSON key generation
final class ConstrainedGenerator: @unchecked Sendable {
    struct Config: Sendable {
        let schema: SchemaMeta
        let tokenizer: TokenizerAdapter
        let tokenTrie: TokenTrie
        let specialTokens: MLXLLMTokenizer.SpecialTokens?
        let sampling: SamplingParameters
        
        // Fallback options
        let enableFallback: Bool
        let keyTrie: KeyTrie?
    }

    private let config: Config
    private var jsonState: EnhancedJSONStateMachine
    private var tokenPath: TokenTrie.Path
    private var sampler: ConstrainedSampler
    private var generatedTokens: [Int32] = []
    private var keyPrefix = ""

    init(config: Config) { 
        self.config = config
        self.jsonState = EnhancedJSONStateMachine()
        self.tokenPath = TokenTrie.Path(root: config.tokenTrie.root)
        self.sampler = ConstrainedSampler(
            tokenTrie: config.tokenTrie,
            specialTokens: config.specialTokens,
            keyTrie: config.keyTrie
        )
    }
    
    // MARK: - Core Generation Loop (when MLXLLM provides step/logits)
    
    #if canImport(MLXLLM)
    func generateConstrained(
        model: Any, // MLXLLM.Model
        prompt: String,
        maxTokens: Int
    ) async throws -> String {
        let promptTokens = config.tokenizer.encode(prompt)
        var context = promptTokens
        generatedTokens.removeAll(keepingCapacity: true)
        
        for _ in 0..<maxTokens {
            // Step 1: Get logits from model
            let logits = try await getLogits(model: model, context: context)
            
            // Step 2: Compute constraints based on current state
            let constraints = computeConstraints()
            
            // Step 3: Apply constraints to logits
            var maskedLogits = logits
            if case .ok(let allowed) = constraints {
                ConstrainedSampler.applyMask(logits: &maskedLogits, allowed: allowed)
            } else if case .noCandidate = constraints {
                // No valid candidates - trigger retry
                throw ConstraintError.noValidCandidates
            }
            
            // Step 4: Sample next token
            let nextToken = sample(from: maskedLogits)
            
            // Step 5: Update state
            updateState(with: nextToken)
            
            // Step 6: Check stop conditions
            if shouldStop(nextToken) {
                break
            }
            
            generatedTokens.append(nextToken)
            context.append(nextToken)
        }
        
        return config.tokenizer.decode(generatedTokens)
    }
    
    private func getLogits(model: Any, context: [Int32]) async throws -> [Float] {
        // TODO: Call MLXLLM model.forward(context) when API available
        return Array(repeating: 0.0, count: config.tokenizer.getVocabSize() ?? 32000)
    }
    
    private func sample(from logits: [Float]) -> Int32 {
        // TODO: Implement proper sampling (temperature, top-k, top-p)
        // For now, return argmax
        guard let maxIndex = logits.indices.max(by: { logits[$0] < logits[$1] }) else {
            return 0
        }
        return Int32(maxIndex)
    }
    #endif
    
    // MARK: - Constraint Computation
    
    func computeConstraints() -> ConstrainedSampler.Decision {
        return sampler.allowedTokens(
            state: jsonState.phase,
            tokenPath: tokenPath,
            keyPrefix: keyPrefix
        )
    }
    
    // Alternative: Use fallback character-based constraints
    func computeConstraintsFallback() -> ConstrainedSampler.Decision {
        // Convert enhanced state to simple state for fallback
        let simplePhase: JSONStateMachine.Phase
        switch jsonState.phase {
        case .inKey: simplePhase = .inKey
        case .expectColon: simplePhase = .expectColon
        default: simplePhase = .outside
        }
        
        return sampler.allowedTokensFallback(
            state: simplePhase,
            normalizedKeyPrefix: keyPrefix
        )
    }
    
    // MARK: - State Management
    
    func updateState(with tokenID: Int32) {
        let text = config.tokenizer.decode([tokenID])
        
        // Update JSON state machine
        jsonState.processToken(tokenID, tokenizer: config.tokenizer)
        
        // Update token path if in key
        if jsonState.isInKeyEmission() {
            let success = tokenPath.append(tokenID, in: config.tokenTrie)
            if !success {
                // Invalid token for current path
                tokenPath.reset(to: config.tokenTrie.root)
            }
            keyPrefix.append(text)
        } else if !jsonState.isInKeyEmission() {
            // Reset when not in key
            tokenPath.reset(to: config.tokenTrie.root)
            keyPrefix.removeAll(keepingCapacity: true)
        }
    }
    
    // For streaming generation
    func onEmittedPiece(_ piece: String, tokenID: Int32? = nil) {
        if let id = tokenID {
            updateState(with: id)
        } else {
            // Fallback to text-based update
            jsonState.processText(piece)
            if jsonState.isInKeyEmission() {
                keyPrefix.append(contentsOf: piece)
            } else {
                keyPrefix.removeAll(keepingCapacity: false)
            }
        }
    }
    
    // MARK: - Stop Conditions
    
    private func shouldStop(_ tokenID: Int32) -> Bool {
        // Check for EOS token or other stop conditions
        // TODO: Implement based on tokenizer's special tokens
        return false
    }
    
    // MARK: - Reset
    
    func reset() {
        jsonState.reset()
        tokenPath.reset(to: config.tokenTrie.root)
        keyPrefix.removeAll()
        generatedTokens.removeAll()
    }
}

// MARK: - Errors

enum ConstraintError: Error {
    case noValidCandidates
    case invalidTokenPath
    case schemaViolation(String)
}

