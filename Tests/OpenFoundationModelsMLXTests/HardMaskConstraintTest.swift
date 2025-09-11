import XCTest
import OpenFoundationModels
import OpenFoundationModelsExtra
@testable import OpenFoundationModelsMLX
import MLX

@Generable
struct MaskTestProfile {
    var name: String
    var age: Int
    var contact: ContactInfo?
    
    @Generable
    struct ContactInfo {
        var email: String
        var phone: String?
    }
}

class HardMaskConstraintTest: XCTestCase {
    
    func testHardMaskApplication() throws {
        print("🧪 [HardMaskConstraintTest] Testing hard mask constraint application")
        
        // 1. Create proper schema node from @Generable
        let responseFormat = Transcript.ResponseFormat(type: MaskTestProfile.self)
        let prompt = Transcript.Prompt(
            segments: [.text(Transcript.TextSegment(content: "Generate profile"))],
            responseFormat: responseFormat
        )
        let transcript = Transcript(entries: [.prompt(prompt)])
        let extracted = TranscriptAccess.extract(from: transcript)
        
        guard let schemaJSON = extracted.schemaJSON,
              let data = schemaJSON.data(using: .utf8),
              let schemaDict = try JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            XCTFail("Failed to extract schema")
            return
        }
        
        let schemaNode = SchemaBuilder.fromJSONSchema(schemaDict)
        print("✅ Schema created with root keys: \(schemaNode.objectKeys)")
        
        // 2. Create processor with schema node
        let mockTokenizer = MockSwiftTokenizer()
        let processor = DPDAKeyTrieLogitProcessor(schema: schemaNode, tokenizer: mockTokenizer)
        
        print("✅ DPDAKeyTrieLogitProcessor created")
        
        // 3. Simulate the generation pipeline step by step
        
        // Create fake logits (vocab size 1000)
        let vocabSize = 1000
        let logits = MLX.ones([1, vocabSize]) * 5.0  // All equal probability initially
        
        print("\n🎯 [HardMaskConstraintTest] Simulating JSON generation pipeline...")
        
        // Simulate: { (opening object)
        print("\n1️⃣ Processing opening brace '{'")
        var currentLogits = try processor.process(logits: logits)
        processor.didSample(token: MLXArray(Int32(100))) // "{" token
        
        // Simulate: " (quote for first key)
        print("2️⃣ Processing quote for key '\"'")
        currentLogits = try processor.process(logits: logits)
        processor.didSample(token: MLXArray(Int32(104))) // "\"" token
        
        // Simulate: name (or any key)
        print("3️⃣ Processing key characters (should show hard constraints)")
        
        // This should trigger hard masking for key tokens only
        for i in 0..<5 { // Simulate a few character generations
            print("\n  📝 Key character \(i+1):")
            let keyLogits = try processor.process(logits: logits)
            
            // Check if logits were modified (indicating hard masking)
            let originalSum = MLX.sum(logits).item(Double.self)
            let processedSum = MLX.sum(keyLogits).item(Double.self)
            
            if abs(originalSum - processedSum) > 0.001 {
                print("    ✅ Hard mask applied - logits modified (original: \(originalSum), processed: \(processedSum))")
            } else {
                print("    ⚠️  No hard mask - logits unchanged")
            }
            
            // Sample a character (simulate 'n', 'a', 'm', 'e', or other key chars)
            let sampleToken = Int32(110 + i) // 'n', 'a', 'm', 'e', etc.
            processor.didSample(token: MLXArray(sampleToken))
        }
        
        // Simulate: " (closing quote for key)
        print("\n4️⃣ Processing closing quote for key")
        currentLogits = try processor.process(logits: logits)
        processor.didSample(token: MLXArray(Int32(104))) // "\"" token
        
        // Simulate: : (colon)
        print("5️⃣ Processing colon ':'")
        currentLogits = try processor.process(logits: logits)
        processor.didSample(token: MLXArray(Int32(105))) // ":" token
        
        print("\n📊 [HardMaskConstraintTest] Test completed")
        print("Expected behavior:")
        print("- Opening brace: No constraints (soft hints)")
        print("- Key quote: Hard mask to quote tokens only")
        print("- Key characters: Hard mask to Trie-allowed tokens")
        print("- Closing quote: Hard mask to quote tokens")
        print("- Colon: Hard mask to colon token")
    }
    
    func testNestedObjectConstraints() throws {
        print("\n🏗️ [HardMaskConstraintTest] Testing nested object constraints")
        
        // Create schema with nested structure
        let responseFormat = Transcript.ResponseFormat(type: MaskTestProfile.self)
        let prompt = Transcript.Prompt(
            segments: [.text(Transcript.TextSegment(content: "Generate profile"))],
            responseFormat: responseFormat
        )
        let transcript = Transcript(entries: [.prompt(prompt)])
        let extracted = TranscriptAccess.extract(from: transcript)
        
        guard let schemaJSON = extracted.schemaJSON,
              let data = schemaJSON.data(using: .utf8),
              let schemaDict = try JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            XCTFail("Failed to extract schema")
            return
        }
        
        let schemaNode = SchemaBuilder.fromJSONSchema(schemaDict)
        let mockTokenizer = MockSwiftTokenizer()
        let processor = DPDAKeyTrieLogitProcessor(schema: schemaNode, tokenizer: mockTokenizer)
        
        let logits = MLX.ones([1, 1000]) * 5.0
        
        // Simulate going into nested object:
        // {"contact": {
        print("Simulating nested object entry: {\"contact\": {")
        
        // {"
        processor.didSample(token: MLXArray(Int32(100))) // "{"
        var currentLogits = try processor.process(logits: logits)
        processor.didSample(token: MLXArray(Int32(104))) // "\""
        
        // Simulate "contact" key
        for char in "contact".utf8 {
            currentLogits = try processor.process(logits: logits)
            processor.didSample(token: MLXArray(Int32(char)))
        }
        
        // ":
        currentLogits = try processor.process(logits: logits)
        processor.didSample(token: MLXArray(Int32(104))) // "\""
        currentLogits = try processor.process(logits: logits)
        processor.didSample(token: MLXArray(Int32(105))) // ":"
        
        // {
        currentLogits = try processor.process(logits: logits)
        processor.didSample(token: MLXArray(Int32(100))) // "{"
        
        print("✅ Entered nested contact object")
        
        // Now inside contact object - should use contact Trie
        print("Testing constraints inside contact object...")
        
        // "
        currentLogits = try processor.process(logits: logits)
        processor.didSample(token: MLXArray(Int32(104))) // "\""
        
        // Key characters inside contact (should be constrained to "email" or "phone")
        print("Testing key constraints inside nested object (should allow email/phone keys only):")
        for i in 0..<3 {
            currentLogits = try processor.process(logits: logits)
            let originalSum = MLX.sum(logits).item(Double.self)
            let processedSum = MLX.sum(currentLogits).item(Double.self)
            
            if abs(originalSum - processedSum) > 0.001 {
                print("  ✅ Hard mask applied in nested context")
            } else {
                print("  ⚠️  No constraints in nested context")
            }
            
            processor.didSample(token: MLXArray(Int32(101 + i))) // Sample some chars
        }
        
        print("📊 Nested constraint test completed")
    }
}