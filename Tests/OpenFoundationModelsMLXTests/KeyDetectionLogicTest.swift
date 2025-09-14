import Testing
import Foundation
@testable import OpenFoundationModelsMLX

@Suite("Key Detection Logic Tests")
struct KeyDetectionLogicTest {

    @Test("Key extraction from JSON text")
    func testKeyExtraction() {
        // Simple key extraction without complex state tracking
        var detectedKeys: [String] = []
        var currentKey = ""
        var inString = false
        var expectingKey = true
        var afterColon = false

        let json = #"{"user":{"name":"Alice","email":"alice@example.com"},"settings":{"theme":"dark"}}"#

        for char in json {
            switch char {
            case "\"":
                if !inString {
                    inString = true
                    if expectingKey {
                        currentKey = ""
                    }
                } else {
                    inString = false
                    if expectingKey && !currentKey.isEmpty {
                        detectedKeys.append(currentKey)
                        expectingKey = false
                    }
                }

            case ":":
                if !inString {
                    afterColon = true
                    expectingKey = false
                }

            case ",", "{":
                if !inString {
                    expectingKey = true
                    afterColon = false
                }

            case "}":
                if !inString {
                    expectingKey = true
                    afterColon = false
                }

            default:
                if inString && expectingKey {
                    currentKey.append(char)
                }
            }
        }

        print("Detected keys: \(detectedKeys)")

        // Verify we detected the correct keys
        #expect(detectedKeys.contains("user"))
        #expect(detectedKeys.contains("name"))
        #expect(detectedKeys.contains("email"))
        #expect(detectedKeys.contains("settings"))
        #expect(detectedKeys.contains("theme"))
    }

    @Test("Nested key context tracking")
    func testNestedContextTracking() {
        // Track nested context without using unused variables

        var currentContext: [String] = []
        var detectedKeys: [String] = []
        var currentKey = ""
        var depth = 0

        let json = #"{"user":{"name":"Alice"},"settings":{"theme":"dark"}}"#

        var inString = false
        var isKey = false

        for char in json {
            switch char {
            case "\"":
                inString = !inString
                if !inString && isKey {
                    // Key completed
                    detectedKeys.append(currentKey)

                    if depth == 1 {
                        currentContext = [currentKey]
                    } else if depth > 1 {
                        // Nested key
                        let fullPath = currentContext + [currentKey]
                        print("Nested key: \(fullPath.joined(separator: "."))")
                    }

                    currentKey = ""
                    isKey = false
                } else if inString && !isKey {
                    // Starting a potential key
                    isKey = true
                    currentKey = ""
                }

            case "{":
                if !inString {
                    depth += 1
                }

            case "}":
                if !inString {
                    depth -= 1
                    if !currentContext.isEmpty {
                        currentContext.removeLast()
                    }
                }

            case ":":
                if !inString {
                    isKey = false
                }

            case ",":
                if !inString {
                    // Ready for next key
                    isKey = false
                }

            default:
                if inString && isKey {
                    currentKey.append(char)
                }
            }
        }

        print("All detected keys: \(detectedKeys)")

        // Verify all keys were detected
        #expect(detectedKeys.contains("user"))
        #expect(detectedKeys.contains("name"))
        #expect(detectedKeys.contains("settings"))
        #expect(detectedKeys.contains("theme"))
    }
}