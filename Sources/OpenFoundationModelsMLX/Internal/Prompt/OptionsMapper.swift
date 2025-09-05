import Foundation
import OpenFoundationModels

// Maps OpenFoundationModels.GenerationOptions into internal SamplingParameters.
enum OptionsMapper {
    static func map(_ options: GenerationOptions?) -> SamplingParameters {
        guard let options else { return SamplingParameters(temperature: nil, topP: nil, topK: nil, maxTokens: nil, stop: nil, seed: nil) }

        var sampling = SamplingParameters(
            temperature: options.temperature,
            topP: nil,
            topK: nil,
            maxTokens: options.maximumResponseTokens,
            stop: nil,
            seed: nil
        )

        if let mode = options.sampling, mode == .greedy {
            sampling.temperature = sampling.temperature ?? 0.0
            sampling.topK = 1
        }

        return sampling
    }
}
