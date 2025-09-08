import Foundation
import PRECISE
import OpenFoundationModels
import OpenFoundationModelsExtra
import MLXLMCommon

// Builds a ChatRequest from a Transcript + options. At this stage, we keep
// logic minimal and safe; schema/tool injection will be expanded later.
enum PromptRenderer {
    static func buildRequest(
        card: any ModelCard,
        transcript: Transcript,
        options: GenerationOptions?
    ) throws -> ChatRequest {
        // Extract from transcript
        let ext = TranscriptAccess.extract(from: transcript)
        
        // Build ModelCardInput for rendering
        // NOTE: System is passed via ModelCardInput.system field only to avoid duplication.
        // The ModelCard is responsible for incorporating it into the final prompt.
        let dateFormatter = ISO8601DateFormatter()
        dateFormatter.formatOptions = [.withFullDate]
        let today = dateFormatter.string(from: Date())
        
        // Messages already come as ModelCardInput.Message from TranscriptAccess
        // Filter out system messages to avoid duplication since system is passed separately
        let filteredMessages = ext.messages.filter { $0.role != .system }
        
        let input = ModelCardInput(
            currentDate: today,
            system: ext.systemText,
            messages: filteredMessages,
            tools: ext.toolDefs.map { .init(name: $0.name, description: $0.description, parametersJSON: $0.parametersJSON) }
        )

        // SSOT: ModelCard.render output is the final prompt
        let prompt = try card.render(input: input)

        // Map transcript response format -> schema meta (no env fallback)
        let responseFormat: ResponseFormatSpec = {
            if let schemaJSON = ext.schemaJSON, !schemaJSON.isEmpty { 
                return .jsonSchema(schemaJSON: schemaJSON) 
            }
            return .text
        }()
        
        let schemaMeta: SchemaMeta? = {
            switch responseFormat {
            case .jsonSchema(let json):
                if let data = json.data(using: .utf8),
                   let dict = try? JSONSerialization.jsonObject(with: data) as? [String: Any] {
                    let keys = (dict["properties"] as? [String: Any])?.keys.map { String($0) } ?? []
                    let required = (dict["required"] as? [String]) ?? []
                    return SchemaMeta(keys: keys, required: required)
                }
                return nil
            default: 
                return nil
            }
        }()

        // No retry policy needed anymore
        let sampling = OptionsMapper.map(options)
        // Keep implementation simple: if caller provided GenerationOptions,
        // prefer sampling path; otherwise use card.params as-is.
        let directParams: GenerateParameters? = (options == nil) ? card.params : nil
        
        let req = ChatRequest(
            modelID: card.id,
            prompt: prompt,
            responseFormat: responseFormat,
            sampling: sampling,
            schema: schemaMeta,
            parameters: directParams
        )
        return req
    }
}