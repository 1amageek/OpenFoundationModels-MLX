import Foundation

struct PipelineConfig: Sendable {
    let constraintMode: ConstraintMode
    let telemetryVerbose: Bool
    let extractJSON: Bool
    
    init(
        constraintMode: ConstraintMode = .off,
        telemetryVerbose: Bool = false,
        extractJSON: Bool = true
    ) {
        self.constraintMode = constraintMode
        self.telemetryVerbose = telemetryVerbose
        self.extractJSON = extractJSON
    }
    
    static let `default` = PipelineConfig()
    
    static let withHardConstraints = PipelineConfig(
        constraintMode: .hard
    )
    
    static let withSoftConstraints = PipelineConfig(
        constraintMode: .soft
    )
    
    static let withPostValidation = PipelineConfig(
        constraintMode: .post
    )
    
    func makeConstraintEngine() -> any ConstraintEngine {
        switch constraintMode {
        case .off:
            return NullConstraintEngine()
        case .soft:
            return SoftConstraintEngine()
        case .hard:
            return HardConstraintEngine()
        case .post:
            return PostConstraintEngine(
                repairer: HeuristicJSONRepairer()
            )
        }
    }
    
    func makeTelemetry() -> any Telemetry {
        return telemetryVerbose ? ConsoleTelemetry(verbose: true) : NoOpTelemetry()
    }
}