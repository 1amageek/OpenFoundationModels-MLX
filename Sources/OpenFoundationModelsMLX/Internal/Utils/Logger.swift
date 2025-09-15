import Foundation
import os

/// Simple logger for debugging
package enum Logger {
    private static let logger = os.Logger(subsystem: "com.openai.mlx", category: "MLX")

    package static func debug(_ message: String) {
        #if DEBUG
        logger.debug("\(message)")
        #endif
    }

    package static func info(_ message: String) {
        logger.info("\(message)")
    }

    package static func warning(_ message: String) {
        logger.warning("\(message)")
    }

    package static func error(_ message: String) {
        logger.error("\(message)")
    }

    package static func verbose(_ message: String) {
        #if DEBUG
        logger.debug("[VERBOSE] \(message)")
        #endif
    }
}