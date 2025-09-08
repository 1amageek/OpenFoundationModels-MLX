import Foundation
import OpenFoundationModels

// Maps OpenFoundationModels.GenerationOptions into internal SamplingParameters.
enum OptionsMapper {
    static func map(_ options: GenerationOptions?) -> SamplingParameters {
        guard let options else { return SamplingParameters(temperature: nil, topP: nil, topK: nil, maxTokens: nil, stop: nil, seed: nil) }

        // Map all available parameters from GenerationOptions
        var sampling = SamplingParameters(
            temperature: options.temperature,
            topP: nil,  // Will be set based on sampling mode if available
            topK: nil,  // Will be set based on sampling mode if available  
            maxTokens: options.maximumResponseTokens,
            stop: nil,  // GenerationOptions doesn't expose stop sequences
            seed: nil  // Will be set based on sampling mode if available
        )

        // Extract sampling mode parameters
        // NOTE: SamplingMode.Kind is private, so we cannot directly compare mode types.
        // We infer greedy mode from temperature=0 as a safe fallback.
        if options.temperature == 0 {
            if sampling.temperature == nil { 
                sampling.temperature = 0.0 
            }
            sampling.topK = 1
        }

        return sampling
    }
}
