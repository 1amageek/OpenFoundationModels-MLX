import Foundation
@preconcurrency import MLX

/// Utility functions for MLX operations used across the project
public enum MLXUtils {
    
    /// Apply logits mask to constrain token selection
    /// - Parameters:
    ///   - logits: The input logits array
    ///   - allowedTokens: Set of allowed token IDs
    ///   - alwaysAllow: Additional tokens to always allow (e.g., EOS)
    /// - Returns: Masked logits array with -inf for disallowed tokens
    public static func applyLogitsMask(
        logits: MLXArray,
        allowedTokens: Set<Int32>,
        alwaysAllow: Set<Int32> = []
    ) -> MLXArray {
        // Get vocabulary size from last dimension
        let vocabSize = logits.dim(logits.ndim - 1)
        
        // Combine allowed tokens
        let allAllowed = allowedTokens.union(alwaysAllow)
        
        // Create mask array
        var maskHost = [Float](repeating: 0, count: vocabSize)
        for tokenID in allAllowed {
            if tokenID >= 0 && tokenID < vocabSize {
                maskHost[Int(tokenID)] = 1
            }
        }
        let maskArray = MLXArray(maskHost)
        
        // Reshape mask for broadcasting
        var shape = Array(repeating: 1, count: logits.ndim)
        shape[logits.ndim - 1] = vocabSize
        let reshapedMask = maskArray.reshaped(shape)
        
        // Apply mask using MLX operations
        let negInf = MLX.full(logits.shape, values: -Float.infinity)
        return MLX.where(reshapedMask .> 0, logits, negInf)
    }
    
    /// Apply soft bias to preferred tokens
    /// - Parameters:
    ///   - logits: The input logits array
    ///   - preferredTokens: Set of preferred token IDs
    ///   - bias: Bias value to add to preferred tokens
    /// - Returns: Logits array with bias applied
    public static func applySoftBias(
        logits: MLXArray,
        preferredTokens: Set<Int32>,
        bias: Float
    ) -> MLXArray {
        guard !preferredTokens.isEmpty && bias != 0 else {
            return logits
        }
        
        // Get vocabulary size from last dimension
        let vocabSize = logits.dim(logits.ndim - 1)
        
        // Create bias array
        var biasHost = [Float](repeating: 0.0, count: vocabSize)
        for tokenID in preferredTokens {
            if tokenID >= 0 && tokenID < vocabSize {
                biasHost[Int(tokenID)] = bias
            }
        }
        let biasArray = MLXArray(biasHost)
        
        // Reshape bias for broadcasting
        var shape = Array(repeating: 1, count: logits.ndim)
        shape[logits.ndim - 1] = vocabSize
        let reshapedBias = biasArray.reshaped(shape)
        
        // Apply bias
        return logits + reshapedBias
    }
    
    /// Create a vocabulary mask for specific tokens
    /// - Parameters:
    ///   - vocabSize: Size of the vocabulary
    ///   - tokens: Token IDs to set to 1
    /// - Returns: MLXArray with 1s at token positions, 0s elsewhere
    public static func createVocabMask(
        vocabSize: Int,
        tokens: Set<Int32>
    ) -> MLXArray {
        var mask = [Float](repeating: 0, count: vocabSize)
        for tokenID in tokens {
            if tokenID >= 0 && tokenID < vocabSize {
                mask[Int(tokenID)] = 1
            }
        }
        return MLXArray(mask)
    }
}