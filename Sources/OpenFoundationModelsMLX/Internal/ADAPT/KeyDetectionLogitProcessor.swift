import Foundation
import MLX
import MLXLMCommon

/// LogitProcessor that detects and logs JSON key generation
/// Uses JSONStateMachine to track parsing state and identify key tokens
public struct KeyDetectionLogitProcessor: LogitProcessor, @unchecked Sendable {
    
    // MARK: - Properties
    
    private let tokenizer: TokenizerAdapter
    private let verbose: Bool
    
    // Mutable state
    private var stateMachine = JSONStateMachine()
    private var generatedText = ""
    private var detectedKeys: [String] = []
    private var nestingStack: [String] = []  // Track nested object context
    
    // MARK: - Initialization
    
    /// Initialize the key detection processor
    /// - Parameters:
    ///   - tokenizer: Tokenizer for decoding tokens to text
    ///   - verbose: Whether to print detailed output (default: true)
    public init(
        tokenizer: TokenizerAdapter,
        verbose: Bool = true
    ) {
        self.tokenizer = tokenizer
        self.verbose = verbose
    }
    
    // MARK: - LogitProcessor Protocol
    
    public mutating func prompt(_ prompt: MLXArray) {
        // Reset state for new generation
        stateMachine.reset()
        generatedText = ""
        detectedKeys = []
        nestingStack = []
        
        if verbose {
            print("\n[KeyDetection] 🔍 Starting JSON key detection...")
        }
    }
    
    public func process(logits: MLXArray) -> MLXArray {
        // This processor only observes, doesn't modify logits
        return logits
    }
    
    public mutating func didSample(token: MLXArray) {
        let tokenId = token.item(Int32.self)
        let text = tokenizer.decode([tokenId])
        
        // Track generated text
        generatedText.append(text)
        
        // Process each character through the state machine
        for char in text {
            // Store previous phase for comparison
            let previousPhase = stateMachine.phase
            
            // Process the character
            stateMachine.processCharacter(char)
            
            // Check if we just exited key generation
            if case .inString(.body(kind: .key, escaped: false)) = previousPhase {
                if case .inObject(.expectColon) = stateMachine.phase {
                    // Key just completed - capture it immediately
                    let keyName = stateMachine.currentKey
                    if !keyName.isEmpty {
                        detectedKeys.append(keyName)
                        
                        if verbose {
                            let level = stateMachine.nestingLevel - 1
                            let indent = String(repeating: "  ", count: level)
                            
                            if level > 0 {
                                let context = nestingStack.last ?? "object"
                                print("[KeyDetection] \(indent)📍 Level \(level) key: \"\(keyName)\" (in \(context))")
                            } else {
                                print("[KeyDetection] 🔑 Root key: \"\(keyName)\"")
                            }
                        }
                    }
                }
            }
            
            // Check for context changes (entering/exiting objects)
            handleContextChanges(previousPhase: previousPhase)
            
            // Debug output for phase transitions
            if verbose && hasPhaseChanged(from: previousPhase, to: stateMachine.phase) {
                printPhaseTransition(from: previousPhase, to: stateMachine.phase)
            }
        }
    }
    
    // MARK: - Private Methods
    
    private mutating func handleContextChanges(previousPhase: JSONStateMachine.Phase) {
        // Check if we entered a new object
        if case .inObject(.expectValue) = previousPhase,
           case .inObject(.expectKeyOrEnd) = stateMachine.phase {
            // Entered a nested object after a key
            if let lastKey = detectedKeys.last {
                nestingStack.append(lastKey)
                if verbose {
                    print("[KeyDetection] 📂 Entering object: \(lastKey)")
                }
            }
        }
        
        // Check if we exited an object
        if stateMachine.nestingLevel < nestingStack.count {
            if let exitedContext = nestingStack.popLast() {
                if verbose {
                    print("[KeyDetection] 📁 Exiting object: \(exitedContext)")
                }
            }
        }
    }
    
    private func hasPhaseChanged(from oldPhase: JSONStateMachine.Phase, to newPhase: JSONStateMachine.Phase) -> Bool {
        // Simple equality check for phase changes
        switch (oldPhase, newPhase) {
        case (.root, .root),
             (.done, .done),
             (.error, .error):
            return false
        case (.inObject(let old), .inObject(let new)):
            return old != new
        case (.inArray(let old), .inArray(let new)):
            return old != new
        case (.inString(let old), .inString(let new)):
            return old != new
        case (.inNumber(let old), .inNumber(let new)):
            return old != new
        case (.inLiteral(let old), .inLiteral(let new)):
            return old != new
        default:
            return true
        }
    }
    
    private func printPhaseTransition(from oldPhase: JSONStateMachine.Phase, to newPhase: JSONStateMachine.Phase) {
        let oldDesc = phaseDescription(oldPhase)
        let newDesc = phaseDescription(newPhase)
        
        // Only print significant transitions
        if shouldPrintTransition(from: oldPhase, to: newPhase) {
            print("[KeyDetection] Phase: \(oldDesc) → \(newDesc)")
        }
    }
    
    private func shouldPrintTransition(from oldPhase: JSONStateMachine.Phase, to newPhase: JSONStateMachine.Phase) -> Bool {
        // Filter out noisy transitions
        switch (oldPhase, newPhase) {
        case (.inString, .inString):
            // Don't print character-by-character string updates
            return false
        case (.inNumber, .inNumber):
            // Don't print digit-by-digit number updates
            return false
        case (_, .inObject(.expectKeyFirstQuote)),
             (_, .inObject(.inKey)),
             (_, .inObject(.expectColon)):
            // These are key-related transitions worth showing
            return true
        default:
            // Show other significant transitions
            return true
        }
    }
    
    private func phaseDescription(_ phase: JSONStateMachine.Phase) -> String {
        switch phase {
        case .root:
            return "root"
        case .inObject(let objPhase):
            return "object.\(objectPhaseDescription(objPhase))"
        case .inArray(let arrPhase):
            return "array.\(arrayPhaseDescription(arrPhase))"
        case .inString(let strPhase):
            return "string.\(stringPhaseDescription(strPhase))"
        case .inNumber(let numPhase):
            return "number.\(numberPhaseDescription(numPhase))"
        case .inLiteral(let litPhase):
            if case .inProgress(let literal) = litPhase {
                return "literal(\(literal))"
            }
            return "literal"
        case .done:
            return "done"
        case .error:
            return "error"
        }
    }
    
    private func objectPhaseDescription(_ phase: JSONStateMachine.ObjectPhase) -> String {
        switch phase {
        case .expectKeyOrEnd: return "expectKeyOrEnd"
        case .expectKeyFirstQuote: return "expectKeyQuote"
        case .inKey: return "inKey"
        case .expectKeyEndQuote: return "expectEndQuote"
        case .expectColon: return "expectColon"
        case .expectValue: return "expectValue"
        case .expectCommaOrEnd: return "expectCommaOrEnd"
        }
    }
    
    private func arrayPhaseDescription(_ phase: JSONStateMachine.ArrayPhase) -> String {
        switch phase {
        case .expectValue: return "expectValue"
        case .expectCommaOrEnd: return "expectCommaOrEnd"
        }
    }
    
    private func stringPhaseDescription(_ phase: JSONStateMachine.StringPhase) -> String {
        switch phase {
        case .body(let kind, let escaped):
            let kindStr = kind == .key ? "key" : "value"
            let escStr = escaped ? ",escaped" : ""
            return "\(kindStr)\(escStr)"
        }
    }
    
    private func numberPhaseDescription(_ phase: JSONStateMachine.NumberPhase) -> String {
        switch phase {
        case .integer: return "integer"
        case .decimal: return "decimal"
        case .exponent: return "exponent"
        }
    }
    
    // MARK: - Public Accessors
    
    /// Get all detected keys in order
    public var allDetectedKeys: [String] {
        return detectedKeys
    }
    
    /// Get the current JSON parsing phase
    public var currentPhase: JSONStateMachine.Phase {
        return stateMachine.phase
    }
    
    /// Check if currently generating a key
    public var isGeneratingKey: Bool {
        return stateMachine.isInKeyGeneration
    }
}