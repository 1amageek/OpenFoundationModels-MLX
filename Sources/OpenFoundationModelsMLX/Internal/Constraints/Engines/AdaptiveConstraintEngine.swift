import Foundation
import MLX
import MLXLMCommon
import Tokenizers
import Synchronization

/// Adaptive constraint engine that selects appropriate constraints based on the request
/// This ensures all generation (Text/JSON) goes through the same pipeline with observable output
final class AdaptiveConstraintEngine: ConstraintEngine, Sendable {
    private let mutex = Mutex<State>(.init())
    
    private struct State: Sendable {
        var mode: ConstraintMode = .off
        var preparedSchema: SchemaNode?
        var keyDetectionProcessor: KeyDetectionLogitProcessor?
        var schemaProcessors: [any LogitProcessor] = []
    }
    
    init() {}
    
    var mode: ConstraintMode {
        mutex.withLock { $0.mode }
    }
    
    func prepare(schema: SchemaNode?, tokenizer: any Tokenizer, modelCard: (any ModelCard)?) async throws {
        // Create tokenizer adapter
        let tokenizerAdapter = MLXLLMTokenizer(tokenizer: tokenizer)
        
        // Debug: Check what schema we received
        if let schema = schema {
            print("[AdaptiveConstraintEngine] Schema received - Kind: \(schema.kind), Keys: \(schema.objectKeys), isEmpty: \(schema.isEmpty)")
        } else {
            print("[AdaptiveConstraintEngine] No schema received")
        }
        
        // Determine mode and processors based on schema
        if let schema = schema, !schema.isEmpty {
            print("[AdaptiveConstraintEngine] Preparing KeyDetectionLogitProcessor for JSON mode")
            
            // Extract schema keys for display
            let schemaKeys = schema.objectKeys.isEmpty ? nil : schema.objectKeys
            if let keys = schemaKeys {
                print("[AdaptiveConstraintEngine] Schema constraints: \(keys.joined(separator: ", "))")
            }
            
            // Extract nested schemas for proper context tracking
            var nestedSchemas: [String: [String]] = [:]
            for (key, propNode) in schema.properties {
                if propNode.kind == .object && !propNode.objectKeys.isEmpty {
                    nestedSchemas[key] = propNode.objectKeys
                    print("[AdaptiveConstraintEngine] Nested schema for '\(key)': \(propNode.objectKeys.joined(separator: ", "))")
                } else if propNode.kind == .array,
                          let itemsNode = propNode.items,
                          itemsNode.kind == .object && !itemsNode.objectKeys.isEmpty {
                    // For array items, use special key notation to indicate array context
                    let arrayKey = "\(key)[]"
                    nestedSchemas[arrayKey] = itemsNode.objectKeys
                    print("[AdaptiveConstraintEngine] Array item schema for '\(arrayKey)': \(itemsNode.objectKeys.joined(separator: ", "))")
                    
                    // Also check for nested objects within array items
                    for (itemPropKey, itemPropNode) in itemsNode.properties {
                        if itemPropNode.kind == .object && !itemPropNode.objectKeys.isEmpty {
                            let nestedArrayKey = "\(key)[].\(itemPropKey)"
                            nestedSchemas[nestedArrayKey] = itemPropNode.objectKeys
                            print("[AdaptiveConstraintEngine] Nested array item schema for '\(nestedArrayKey)': \(itemPropNode.objectKeys.joined(separator: ", "))")
                        }
                    }
                }
            }
            
            // JSON mode with schema constraints
            // Convert SchemaNode to JSON Schema format for KeyDetectionLogitProcessor
            let jsonSchema = convertSchemaNodeToJSONSchema(schema)

            print("[AdaptiveConstraintEngine] Converting SchemaNode to JSON Schema...")
            print("[AdaptiveConstraintEngine] Original SchemaNode keys: \(schema.objectKeys)")
            print("[AdaptiveConstraintEngine] Converted JSON Schema:")
            if let properties = jsonSchema["properties"] as? [String: Any] {
                print("[AdaptiveConstraintEngine] Root properties: \(properties.keys.sorted())")
                // Debug nested structures
                for (key, value) in properties {
                    if let objDict = value as? [String: Any],
                       let objProps = objDict["properties"] as? [String: Any] {
                        print("[AdaptiveConstraintEngine]   \(key) properties: \(objProps.keys.sorted())")
                    }
                }
            } else {
                print("[AdaptiveConstraintEngine] WARNING: No properties in converted JSON Schema!")
                print("[AdaptiveConstraintEngine] Full schema: \(jsonSchema)")
            }

            // Create key detection processor for JSON debugging with enhanced features
            let keyDetectionProcessor = KeyDetectionLogitProcessor(
                tokenizer: tokenizerAdapter,
                jsonSchema: jsonSchema,  // Pass JSON Schema format
                modelCard: modelCard,    // Pass modelCard for activation control
                verbose: true,           // Enable verbose output for key detection
                topK: 5,                 // Show top-5 candidates
                showProbabilities: true  // Show probability distributions
            )
            
            // Future: Add TokenTrieLogitProcessor or other ADAPT implementations here
            
            mutex.withLock {
                $0.mode = .hard
                $0.preparedSchema = schema
                $0.keyDetectionProcessor = keyDetectionProcessor
                $0.schemaProcessors = [] // Will be populated with ADAPT processors
            }
        } else {
            // Text mode - no processors
            mutex.withLock {
                $0.mode = .off
                $0.preparedSchema = nil
                $0.keyDetectionProcessor = nil
                $0.schemaProcessors = []
            }
        }
    }
    
    func softPrompt(for schema: SchemaNode?) -> String? {
        // No soft prompts needed for adaptive engine
        return nil
    }
    
    func logitProcessors() async -> [LogitProcessor] {
        return mutex.withLock { state in
            var processors: [any LogitProcessor] = []
            
            // Add key detection processor if in JSON mode
            if let keyDetection = state.keyDetectionProcessor {
                processors.append(keyDetection)
            }
            
            // Add schema-specific processors if in JSON mode
            processors.append(contentsOf: state.schemaProcessors)
            
            return processors
        }
    }

    // MARK: - Private Helpers

    /// Convert SchemaNode to JSON Schema format for JSONSchemaContextDetector
    private func convertSchemaNodeToJSONSchema(_ node: SchemaNode) -> [String: Any] {
        var result: [String: Any] = [:]

        // Set type based on node kind
        switch node.kind {
        case .object:
            result["type"] = "object"
            if !node.properties.isEmpty {
                var properties: [String: Any] = [:]
                for (key, propNode) in node.properties {
                    properties[key] = convertSchemaNodeToJSONSchema(propNode)
                }
                result["properties"] = properties
            }
            if !node.required.isEmpty {
                result["required"] = Array(node.required)
            }
        case .array:
            result["type"] = "array"
            if let items = node.items {
                result["items"] = convertSchemaNodeToJSONSchema(items)
            }
        case .string:
            result["type"] = "string"
        case .number:
            result["type"] = "number"
        case .boolean:
            result["type"] = "boolean"
        case .null:
            result["type"] = "null"
        case .any:
            // Don't set type for "any"
            break
        }

        return result
    }
}
