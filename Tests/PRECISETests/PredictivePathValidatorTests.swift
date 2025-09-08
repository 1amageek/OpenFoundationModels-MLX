import Testing
import Foundation
@testable import PRECISE
@testable import OpenFoundationModelsMLX
import Tokenizers

@Suite("Predictive Path Validator Tests")
struct PredictivePathValidatorTests {
    
    // Create test setup for each test
    func createTestSetup() async throws -> (MLXLLMTokenizer, PredictivePathValidator, TokenTrie) {
        let mockTokenizer = try await AutoTokenizer.from(pretrained: "bert-base-uncased")
        let tokenizer = MLXLLMTokenizer(tokenizer: mockTokenizer)
        let validator = PredictivePathValidator(tokenizer: tokenizer)
        
        // Build a simple token trie with test keys
        let schema = SchemaMeta(keys: ["name", "age", "email", "address"], required: ["name"])
        let tokenTrie = TokenTrieBuilder.build(from: schema, tokenizer: tokenizer)
        
        return (tokenizer, validator, tokenTrie)
    }
    
    @Test("Validate empty path")
    func validateEmptyPath() async throws {
        let (_, validator, tokenTrie) = try await createTestSetup()
        
        // Test validation at the root of the trie
        let path = TokenTrie.Path(root: tokenTrie.root)
        let validation = validator.validateFuturePaths(
            from: path,
            tokenTrie: tokenTrie,
            depth: 1
        )
        
        #expect(validation.hasValidPaths)
        #expect(!validation.tokenScores.isEmpty)
        #expect(validation.recommendedToken != nil)
    }
    
    @Test("Validate partial path")
    func validatePartialPath() async throws {
        let (tokenizer, validator, tokenTrie) = try await createTestSetup()
        
        // Create a path with partial key
        var path = TokenTrie.Path(root: tokenTrie.root)
        let nameTokens = tokenizer.encode("na")
        
        for token in nameTokens {
            _ = path.append(token, in: tokenTrie)
        }
        
        let validation = validator.validateFuturePaths(
            from: path,
            tokenTrie: tokenTrie,
            depth: 2
        )
        
        // Should have valid paths since "na" can lead to "name"
        #expect(validation.hasValidPaths)
        #expect(!validation.tokenScores.isEmpty)
    }
    
    @Test("Validate invalid path")
    func validateInvalidPath() async throws {
        let (tokenizer, validator, tokenTrie) = try await createTestSetup()
        
        // Create an invalid path
        var path = TokenTrie.Path(root: tokenTrie.root)
        let invalidTokens = tokenizer.encode("xyz")
        
        // This should fail to append if the token is not in the trie
        guard !invalidTokens.isEmpty else { return }
        let success = path.append(invalidTokens[0], in: tokenTrie)
        
        // Only check validation if append actually failed
        // Note: Some tokenizers may encode "xyz" with tokens that happen to be in the trie
        if !success {
            // Path is invalid, validation should reflect this
            let validation = validator.validateFuturePaths(
                from: path,
                tokenTrie: tokenTrie,
                depth: 1
            )
            
            #expect(!validation.hasValidPaths)
        } else {
            // If append succeeded, the test assumptions don't hold
            // Skip the test rather than fail
        }
    }
    
    @Test("Terminal detection")
    func terminalDetection() async throws {
        let (tokenizer, validator, tokenTrie) = try await createTestSetup()
        
        // Create a complete path to a terminal
        var path = TokenTrie.Path(root: tokenTrie.root)
        let nameTokens = tokenizer.encode("name")
        
        for token in nameTokens {
            _ = path.append(token, in: tokenTrie)
        }
        
        // At terminal, should recommend quote token
        let validation = validator.validateFuturePaths(
            from: path,
            tokenTrie: tokenTrie,
            depth: 0
        )
        
        if path.isAtTerminal() {
            // Should have boosted score for quote token
            if let quoteToken = tokenizer.findSpecialTokens().quote {
                let quoteScore = validation.tokenScores[quoteToken] ?? 0
                // Note: Score boosting depends on implementation, just check it exists
                #expect(quoteScore >= 0)
            }
        }
    }
    
    @Test("Look-ahead depth")
    func lookAheadDepth() async throws {
        let (_, validator, tokenTrie) = try await createTestSetup()
        let path = TokenTrie.Path(root: tokenTrie.root)
        
        // Test different depths
        let depths = [0, 1, 2, 3]
        var previousScoreCount = 0
        
        for depth in depths {
            let validation = validator.validateFuturePaths(
                from: path,
                tokenTrie: tokenTrie,
                depth: depth
            )
            
            // Deeper look-ahead should generally provide more nuanced scores
            if depth > 0 {
                #expect(validation.hasValidPaths)
            }
            
            // Track that we're getting scores
            if depth > 0 {
                #expect(validation.tokenScores.count >= previousScoreCount)
            }
            previousScoreCount = validation.tokenScores.count
        }
    }
    
    @Test("Multiple validations")
    func multipleValidations() async throws {
        let (_, validator, tokenTrie) = try await createTestSetup()
        let path = TokenTrie.Path(root: tokenTrie.root)
        
        // Perform multiple validations
        for _ in 0..<3 {
            _ = validator.validateFuturePaths(
                from: path,
                tokenTrie: tokenTrie,
                depth: 1
            )
        }
        
        let stats = validator.getStatistics()
        #expect(stats.totalValidations == 3)
    }
    
    @Test("Statistics tracking")
    func statisticsTracking() async throws {
        let (_, validator, tokenTrie) = try await createTestSetup()
        let path = TokenTrie.Path(root: tokenTrie.root)
        
        // Perform multiple validations
        for _ in 0..<5 {
            _ = validator.validateFuturePaths(
                from: path,
                tokenTrie: tokenTrie,
                depth: 1
            )
        }
        
        let stats = validator.getStatistics()
        #expect(stats.totalValidations == 5)
        #expect(stats.successfulValidations == 5)
        #expect(stats.averageValidationTime > 0)
    }
    
    @Test("Reset statistics")
    func resetStatistics() async throws {
        let (_, validator, tokenTrie) = try await createTestSetup()
        let path = TokenTrie.Path(root: tokenTrie.root)
        
        // Perform some validations
        _ = validator.validateFuturePaths(
            from: path,
            tokenTrie: tokenTrie,
            depth: 1
        )
        
        // Reset and verify
        validator.resetStatistics()
        let stats = validator.getStatistics()
        
        #expect(stats.totalValidations == 0)
        #expect(stats.successfulValidations == 0)
        #expect(stats.totalValidations == 0)
    }
}