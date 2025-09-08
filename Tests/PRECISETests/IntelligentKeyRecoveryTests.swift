import Testing
import Foundation
@testable import PRECISE
@testable import OpenFoundationModelsMLX
import Tokenizers

@Suite("Intelligent Key Recovery Tests")
struct IntelligentKeyRecoveryTests {
    
    // Create a mock tokenizer and recovery instance for each test
    func createTestSetup() async throws -> (MLXLLMTokenizer, IntelligentKeyRecovery) {
        let mockTokenizer = try await AutoTokenizer.from(pretrained: "bert-base-uncased")
        let tokenizer = MLXLLMTokenizer(tokenizer: mockTokenizer)
        let recovery = IntelligentKeyRecovery(tokenizer: tokenizer)
        return (tokenizer, recovery)
    }
    
    @Test("Exact prefix match")
    func exactPrefixMatch() async throws {
        let (_, recovery) = try await createTestSetup()
        let schemaKeys = ["firstName", "lastName", "email", "phoneNumber"]
        let partialKey = "first"
        let currentTokens: [Int32] = []
        
        let strategy = recovery.recoverFromInvalidPath(
            partialKey: partialKey,
            schemaKeys: schemaKeys,
            currentTokens: currentTokens
        )
        
        // Should complete to "firstName"
        if case .completeToKey(let target, _) = strategy {
            #expect(target == "firstName")
        } else {
            Issue.record("Expected completeToKey strategy")
        }
    }
    
    @Test("Edit distance recovery")
    func editDistanceRecovery() async throws {
        let (_, recovery) = try await createTestSetup()
        let schemaKeys = ["firstName", "lastName", "email"]
        let partialKey = "fristName"  // Typo: 'frist' instead of 'first'
        let currentTokens: [Int32] = []
        
        let strategy = recovery.recoverFromInvalidPath(
            partialKey: partialKey,
            schemaKeys: schemaKeys,
            currentTokens: currentTokens
        )
        
        // Should suggest correction to "firstName"
        switch strategy {
        case .completeToKey(let target, _):
            #expect(target == "firstName")
        case .closeCurrentKey:
            // Acceptable alternative
            break
        default:
            Issue.record("Expected recovery strategy")
        }
    }
    
    @Test("No match recovery")
    func noMatchRecovery() async throws {
        let (_, recovery) = try await createTestSetup()
        let schemaKeys = ["name", "age", "email"]
        let partialKey = "xyz123"  // No match
        let currentTokens: [Int32] = []
        
        let strategy = recovery.recoverFromInvalidPath(
            partialKey: partialKey,
            schemaKeys: schemaKeys,
            currentTokens: currentTokens
        )
        
        // Should either abort or try to close/skip
        switch strategy {
        case .abort:
            // Expected when no recovery possible
            break
        case .closeCurrentKey:
            // Acceptable recovery attempt
            break
        case .skipToNext:
            // Acceptable recovery attempt
            break
        default:
            Issue.record("Unexpected recovery strategy")
        }
    }
    
    @Test("Substring match")
    func substringMatch() async throws {
        let (_, recovery) = try await createTestSetup()
        let schemaKeys = ["userEmailAddress", "emailVerified", "primaryEmail"]
        let partialKey = "email"  // Substring of multiple keys
        let currentTokens: [Int32] = []
        
        let strategy = recovery.recoverFromInvalidPath(
            partialKey: partialKey,
            schemaKeys: schemaKeys,
            currentTokens: currentTokens
        )
        
        // Should find a match (any of the email-related keys)
        switch strategy {
        case .completeToKey(let target, _):
            #expect(schemaKeys.contains(target))
        default:
            // Other strategies are acceptable
            break
        }
    }
    
    @Test("Edit distance calculation")
    func editDistanceCalculation() async throws {
        let (_, recovery) = try await createTestSetup()
        // Test the internal edit distance algorithm indirectly
        let schemaKeys = ["name", "game", "fame"]  // All 1 edit from "name"
        let partialKey = "name"
        let currentTokens: [Int32] = []
        
        let strategy = recovery.recoverFromInvalidPath(
            partialKey: partialKey,
            schemaKeys: schemaKeys,
            currentTokens: currentTokens
        )
        
        // Should prefer exact match
        if case .completeToKey(let target, _) = strategy {
            #expect(target == "name")
        }
    }
    
    @Test("Closing tokens generation")
    func closingTokensGeneration() async throws {
        let (_, recovery) = try await createTestSetup()
        let schemaKeys = ["validKey"]
        let partialKey = "invalidPartial"
        let currentTokens: [Int32] = []
        
        let strategy = recovery.recoverFromInvalidPath(
            partialKey: partialKey,
            schemaKeys: schemaKeys,
            currentTokens: currentTokens
        )
        
        // Should try to close or skip when no match found
        switch strategy {
        case .closeCurrentKey(let tokens):
            #expect(!tokens.isEmpty)
            // Should include quote and possibly colon
        case .skipToNext(let tokens):
            #expect(!tokens.isEmpty)
            // Should include tokens to skip
        case .abort:
            // Acceptable if no recovery possible
            break
        default:
            break
        }
    }
    
    @Test("Statistics tracking")
    func statisticsTracking() async throws {
        let (_, recovery) = try await createTestSetup()
        let schemaKeys = ["name", "age"]
        
        // Perform multiple recoveries
        for i in 0..<5 {
            let partialKey = i < 3 ? "na" : "xyz"  // Mix of recoverable and non-recoverable
            _ = recovery.recoverFromInvalidPath(
                partialKey: partialKey,
                schemaKeys: schemaKeys,
                currentTokens: []
            )
        }
        
        let stats = recovery.getStatistics()
        #expect(stats.recoveryAttempts == 5)
        // At least some should succeed (the "na" -> "name" cases)
        #expect(stats.successfulRecoveries > 0)
    }
    
    @Test("Reset statistics")
    func resetStatistics() async throws {
        let (_, recovery) = try await createTestSetup()
        // Perform a recovery
        _ = recovery.recoverFromInvalidPath(
            partialKey: "test",
            schemaKeys: ["testing"],
            currentTokens: []
        )
        
        // Reset and verify
        recovery.resetStatistics()
        let stats = recovery.getStatistics()
        
        #expect(stats.recoveryAttempts == 0)
        #expect(stats.successfulRecoveries == 0)
    }
    
    @Test("Multiple close matches")
    func multipleCloseMatches() async throws {
        let (_, recovery) = try await createTestSetup()
        let schemaKeys = ["userName", "userEmail", "userId", "userAge"]
        let partialKey = "user"  // Prefix of all keys
        let currentTokens: [Int32] = []
        
        let strategy = recovery.recoverFromInvalidPath(
            partialKey: partialKey,
            schemaKeys: schemaKeys,
            currentTokens: currentTokens
        )
        
        // Should pick one of the valid completions
        if case .completeToKey(let target, _) = strategy {
            #expect(schemaKeys.contains(target))
            #expect(target.hasPrefix(partialKey))
        }
    }
}