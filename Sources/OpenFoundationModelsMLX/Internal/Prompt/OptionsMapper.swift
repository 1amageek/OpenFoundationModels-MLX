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
        // Since SamplingMode.Kind is private, we need to check equality with static constructors
        if let mode = options.sampling {
            if mode == .greedy {
                sampling.temperature = 0.0
                sampling.topK = 1
            } else {
                // For random sampling modes, we can't extract the values directly
                // due to private Kind enum. We'll rely on the temperature field
                // from GenerationOptions as a fallback.
                // This is a limitation of the current API design.
                if options.temperature == 0 {
                    // Likely greedy mode requested via temperature
                    sampling.topK = 1
                }
            }
        }

        return sampling
    }
}
