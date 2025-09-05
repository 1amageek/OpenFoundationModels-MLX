import Foundation
import OpenFoundationModels
import OpenFoundationModelsExtra

// Builds a ChatRequest from a Transcript + options. At this stage, we keep
// logic minimal and safe; schema/tool injection will be expanded later.
enum PromptRenderer {
    static func buildRequest(
        modelID: String,
        transcript: Transcript,
        options: GenerationOptions?
    ) -> ChatRequest {
        // JSONベースでTranscriptから必要情報を抽出（Codable準拠を活用）
        // Extra による強型アクセスで抽出
        let ext = TranscriptAccess.extract(from: transcript)
        var messages: [ChatMessage] = []
        if var sys = ext.systemText, !sys.isEmpty {
            // ツール定義をsystemに注入（存在する場合）。
            if !ext.toolDefs.isEmpty {
                sys += "\n\n[TOOLS]\nYou can call tools by returning STRICT JSON only, no prose."
                sys += "\nReturn: {\"tool_calls\":[{\"name\":string,\"arguments\":object}...]}"
                for t in ext.toolDefs {
                    sys += "\n- name: \(t.name)"
                    if let d = t.description { sys += "\n  description: \(d)" }
                    if let p = t.parametersJSON { sys += "\n  parameters: \(p)" }
                }
            }
            messages.append(.init(role: .system, content: sys))
        }
        // 履歴の取り込み（圧縮・削減は行わず、そのまま使用）
        for m in ext.messages { messages.append(m) }

        let sampling = OptionsMapper.map(options)

        // Response format: try env override for schema JSON; otherwise .text.
        let env = ProcessInfo.processInfo.environment
        let responseFormat: ResponseFormatSpec = {
            if let schemaJSON = ext.schemaJSON, !schemaJSON.isEmpty { return .jsonSchema(schemaJSON: schemaJSON) }
            if let schemaJSON = env["OFM_MLX_SCHEMA_JSON"], !schemaJSON.isEmpty { return .jsonSchema(schemaJSON: schemaJSON) }
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
            default: return nil
            }
        }()

        let retryMax = Int(env["OFM_MLX_RETRY_MAX"] ?? "2") ?? 2
        let req = ChatRequest(
            modelID: modelID,
            messages: messages,
            responseFormat: responseFormat,
            sampling: sampling,
            policy: .init(enableSCD: true, enableSnap: true, retryMaxTries: retryMax),
            schema: schemaMeta
        )
        return req
    }

    // 旧フォールバック関数は不要になったため削除
}
