import Foundation
import OpenFoundationModels
import MLXLMCommon

// Maps OpenFoundationModels.GenerationOptions into internal SamplingParameters.
enum OptionsMapper {
    static func map(_ options: GenerationOptions?, modelCard: (any ModelCard)? = nil) -> SamplingParameters {
        // When options is nil, use ModelCard params as the source
        guard let options else {
            if let modelCard = modelCard {
                // Use ModelCard's default parameters
                return SamplingParameters(
                    temperature: Double(modelCard.params.temperature),
                    topP: Double(modelCard.params.topP),
                    topK: nil,  // ModelCard params doesn't have topK
                    maxTokens: modelCard.params.maxTokens,
                    stop: nil,  // ModelCard params doesn't have stop
                    seed: nil   // ModelCard params doesn't have seed
                )
            } else {
                // No options and no ModelCard - return empty parameters
                return SamplingParameters(temperature: nil, topP: nil, topK: nil, maxTokens: nil, stop: nil, seed: nil)
            }
        }

        // Map all available parameters from GenerationOptions
        // Priority: GenerationOptions > ModelCard params > nil
        let fallbackTemp: Double? = modelCard != nil ? Double(modelCard!.params.temperature) : nil
        let fallbackTopP: Double? = modelCard != nil ? Double(modelCard!.params.topP) : nil
        
        var sampling = SamplingParameters(
            temperature: options.temperature ?? fallbackTemp,
            topP: nil,  // Will be set based on sampling mode if available
            topK: nil,  // Will be set based on sampling mode if available  
            maxTokens: options.maximumResponseTokens ?? modelCard?.params.maxTokens,
            stop: nil,  // GenerationOptions doesn't expose stop sequences
            seed: nil  // Will be set based on sampling mode if available
        )

        // Extract sampling mode parameters
        // NOTE: SamplingMode.Kind is private, so we cannot directly compare mode types.
        // Set temperature to 0.0 if not specified but temperature is 0
        if options.temperature == 0 {
            if sampling.temperature == nil { 
                sampling.temperature = 0.0 
            }
        }

        return sampling
    }
}
