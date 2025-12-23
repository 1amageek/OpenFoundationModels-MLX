import Foundation
import MLXLMCommon
import MLXLLM
import Tokenizers
import OpenFoundationModels

/// Errors specific to the generation pipeline
enum GenerationPipelineError: Error {
    case missingModelCard
}

struct GenerationPipeline: Sendable {

    let executor: MLXExecutor

    init(executor: MLXExecutor) {
        self.executor = executor
    }

    func run(
        prompt: String,
        parameters: GenerateParameters,
        modelCard: (any ModelCard)? = nil
    ) async throws -> String {
        // Execute generation
        do {
            var raw = ""
            let stream = await executor.executeStream(
                prompt: prompt,
                parameters: parameters,
                logitProcessor: nil
            )

            // Get stop tokens from ModelCard
            let stopTokens = modelCard?.stopTokens ?? []

            for try await chunk in stream {
                raw += chunk

                // Check for stop tokens
                if !stopTokens.isEmpty {
                    for stopToken in stopTokens {
                        if raw.contains(stopToken) {
                            Logger.info("[GenerationPipeline] Stop token detected: \(stopToken)")
                            // Truncate at stop token if needed (keep the stop token for parsing)
                            break
                        }
                    }
                    // If we found any stop token, break out of the loop
                    if stopTokens.contains(where: { raw.contains($0) }) {
                        break
                    }
                }
            }

            // Log the generated output
            Logger.info("[GenerationPipeline] Generated output (\(raw.count) characters):")
            Logger.info("========== START OUTPUT ==========")
            Logger.info(raw)
            Logger.info("========== END OUTPUT ==========")

            // ModelCard is required for proper output processing
            guard let card = modelCard else {
                throw GenerationPipelineError.missingModelCard
            }

            // Process output through ModelCard to extract final content
            let entry = card.generate(from: raw, options: nil)
            let output: String = {
                switch entry {
                case .response(let response):
                    return response.segments.compactMap { segment in
                        if case .text(let text) = segment {
                            return text.content
                        }
                        return nil
                    }.joined()
                default:
                    // For non-response entries, return raw
                    return raw
                }
            }()

            Logger.info("[GenerationPipeline] Extracted output from ModelCard: \(output)")

            return output

        } catch {
            throw error
        }
    }

    func stream(
        prompt: String,
        parameters: GenerateParameters,
        modelCard: (any ModelCard)? = nil
    ) -> AsyncThrowingStream<String, Error> {
        AsyncThrowingStream { continuation in
            Task {
                do {
                    // ModelCard is required for proper output processing
                    guard let card = modelCard else {
                        throw GenerationPipelineError.missingModelCard
                    }

                    let baseStream = await executor.executeStream(
                        prompt: prompt,
                        parameters: parameters,
                        logitProcessor: nil
                    )

                    // Get stop tokens from ModelCard
                    let stopTokens = card.stopTokens
                    var buffer = ""

                    for try await chunk in baseStream {
                        buffer += chunk
                        continuation.yield(chunk)

                        // Check for stop tokens
                        if !stopTokens.isEmpty {
                            if stopTokens.contains(where: { buffer.contains($0) }) {
                                Logger.info("[GenerationPipeline] Stream: Stop token detected")
                                break
                            }
                        }
                    }

                    continuation.finish()

                } catch {
                    continuation.finish(throwing: error)
                }
            }
        }
    }
}
