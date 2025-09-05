import Foundation

// Simple logging utility for OpenFoundationModels-MLX
// Controls verbose debug output based on environment variable or debug flag
enum Logger {
    enum Level: Int {
        case none = 0
        case error = 1
        case warning = 2
        case info = 3
        case debug = 4
        case verbose = 5
    }
    
    // Check environment variable for log level
    // Set OPENMODELS_LOG_LEVEL=debug for debug logging
    private static let logLevel: Level = {
        if let env = ProcessInfo.processInfo.environment["OPENMODELS_LOG_LEVEL"] {
            switch env.lowercased() {
            case "none": return .none
            case "error": return .error
            case "warning", "warn": return .warning
            case "info": return .info
            case "debug": return .debug
            case "verbose": return .verbose
            default: 
                #if DEBUG
                return .info  // Default to info in debug builds
                #else
                return .error // Only errors in release builds
                #endif
            }
        } else {
            #if DEBUG
            return .info  // Default to info in debug builds
            #else
            return .error // Only errors in release builds
            #endif
        }
    }()
    
    static func error(_ message: String, file: String = #file, function: String = #function) {
        if logLevel.rawValue >= Level.error.rawValue {
            print("[ERROR] \(message)")
        }
    }
    
    static func warning(_ message: String, file: String = #file, function: String = #function) {
        if logLevel.rawValue >= Level.warning.rawValue {
            print("[WARNING] \(message)")
        }
    }
    
    static func info(_ message: String, file: String = #file, function: String = #function) {
        if logLevel.rawValue >= Level.info.rawValue {
            print("[INFO] \(message)")
        }
    }
    
    static func debug(_ message: String, file: String = #file, function: String = #function) {
        if logLevel.rawValue >= Level.debug.rawValue {
            print("[DEBUG] \(message)")
        }
    }
    
    static func verbose(_ message: String, file: String = #file, function: String = #function) {
        if logLevel.rawValue >= Level.verbose.rawValue {
            print("[VERBOSE] \(message)")
        }
    }
}