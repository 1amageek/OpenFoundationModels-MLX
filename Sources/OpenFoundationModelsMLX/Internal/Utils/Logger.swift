import Foundation
import os

/// Simple logger for debugging
public enum Logger {
    private static let logger = os.Logger(subsystem: "com.openai.mlx", category: "MLX")
    
    public static func debug(_ message: String) {
        #if DEBUG
        logger.debug("\(message)")
        #endif
    }
    
    public static func info(_ message: String) {
        logger.info("\(message)")
    }
    
    public static func warning(_ message: String) {
        logger.warning("\(message)")
    }
    
    public static func error(_ message: String) {
        logger.error("\(message)")
    }
    
    public static func verbose(_ message: String) {
        #if DEBUG
        logger.debug("[VERBOSE] \(message)")
        #endif
    }
}