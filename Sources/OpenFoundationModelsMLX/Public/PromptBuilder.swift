import Foundation

/// A lightweight Result Builder to help author Swift-native templates
/// in ModelCard implementations. Use standard Swift control flow
/// (if/for/switch) to assemble prompt pieces.
@resultBuilder
public enum PromptBuilder {
    public static func buildBlock(_ parts: String...) -> String { parts.joined() }
    public static func buildExpression(_ str: String) -> String { str }
    public static func buildOptional(_ part: String?) -> String { part ?? "" }
    public static func buildEither(first: String) -> String { first }
    public static func buildEither(second: String) -> String { second }
    public static func buildArray(_ parts: [String]) -> String { parts.joined() }
    public static func buildLimitedAvailability(_ part: String) -> String { part }
}

/// Render context passed to PromptTemplate closures.
public struct RenderContext: Sendable {
    public let input: ModelCardInput
    public init(input: ModelCardInput) { self.input = input }
}

/// A reusable prompt template.
public struct PromptTemplate: Sendable {
    public let render: @Sendable (RenderContext) -> String
    public init(render: @escaping @Sendable (RenderContext) -> String) { self.render = render }
}

/// Helper to create a PromptTemplate using the builder.
public func Template(@PromptBuilder _ body: @escaping @Sendable (RenderContext) -> String) -> PromptTemplate {
    PromptTemplate(render: body)
}
