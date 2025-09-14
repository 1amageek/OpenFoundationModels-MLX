import Foundation
import Testing
@testable import OpenFoundationModelsMLX

@Suite("JSONContextTracker Tests")
struct JSONContextTrackerTests {

    @Test("Tracks root context correctly")
    func testRootContext() {
        let tracker = JSONContextTracker()

        #expect(tracker.getCurrentPath() == "")
        #expect(tracker.nestingDepth == 0)
        #expect(!tracker.isInArray())
    }

    @Test("Tracks simple object context")
    func testSimpleObjectContext() {
        let tracker = JSONContextTracker()

        // Detect "headquarters" key
        tracker.keyDetected("headquarters")
        // Enter object for headquarters
        tracker.enterObject()

        #expect(tracker.getCurrentPath() == "headquarters")
        #expect(tracker.nestingDepth == 1)
        #expect(!tracker.isInArray())
    }

    @Test("Tracks array context")
    func testArrayContext() {
        let tracker = JSONContextTracker()

        // Detect "departments" key
        tracker.keyDetected("departments")
        // Enter array for departments
        tracker.enterArray()

        #expect(tracker.getCurrentPath() == "departments[]")
        #expect(tracker.nestingDepth == 1)
        #expect(tracker.isInArray())
        #expect(tracker.getCurrentArrayContext() == "departments[]")
    }

    @Test("Tracks object within array")
    func testObjectWithinArray() {
        let tracker = JSONContextTracker()

        // Enter departments array
        tracker.keyDetected("departments")
        tracker.enterArray()

        // Enter object within array (array item)
        tracker.enterObject()

        // Path should still be departments[] for the array item
        #expect(tracker.getCurrentPath() == "departments[]")
        #expect(tracker.nestingDepth == 2)
        #expect(tracker.isInArray())
    }

    @Test("Tracks nested object within array item")
    func testNestedObjectWithinArrayItem() {
        let tracker = JSONContextTracker()

        // Enter departments array
        tracker.keyDetected("departments")
        tracker.enterArray()

        // Enter object within array (array item)
        tracker.enterObject()

        // Detect "manager" key within array item
        tracker.keyDetected("manager")
        // Enter manager object
        tracker.enterObject()

        // Path should be departments[].manager
        #expect(tracker.getCurrentPath() == "departments[].manager")
        #expect(tracker.nestingDepth == 3)
        #expect(tracker.isInArray())
    }

    @Test("Tracks nested array within array item")
    func testNestedArrayWithinArrayItem() {
        let tracker = JSONContextTracker()

        // Enter departments array
        tracker.keyDetected("departments")
        tracker.enterArray()

        // Enter object within array (array item)
        tracker.enterObject()

        // Detect "projects" key within array item
        tracker.keyDetected("projects")
        // Enter projects array
        tracker.enterArray()

        // Path should be departments[].projects[]
        #expect(tracker.getCurrentPath() == "departments[].projects[]")
        #expect(tracker.nestingDepth == 3)
        #expect(tracker.isInArray())
    }

    @Test("Handles context exit correctly")
    func testContextExit() {
        let tracker = JSONContextTracker()

        // Build nested context
        tracker.keyDetected("departments")
        tracker.enterArray()
        tracker.enterObject()
        tracker.keyDetected("manager")
        tracker.enterObject()

        #expect(tracker.getCurrentPath() == "departments[].manager")
        #expect(tracker.nestingDepth == 3)

        // Exit manager object
        tracker.exitContext()
        #expect(tracker.getCurrentPath() == "departments[]")
        #expect(tracker.nestingDepth == 2)

        // Exit array item object
        tracker.exitContext()
        #expect(tracker.getCurrentPath() == "departments[]")
        #expect(tracker.nestingDepth == 1)

        // Exit departments array
        tracker.exitContext()
        #expect(tracker.getCurrentPath() == "")
        #expect(tracker.nestingDepth == 0)
        #expect(!tracker.isInArray())
    }

    @Test("Gets correct context keys from schema")
    func testGetContextKeys() {
        let tracker = JSONContextTracker()

        let nestedSchemas = [
            "headquarters": ["city", "country", "postalCode", "street"],
            "departments[]": ["headCount", "manager", "name", "projects", "type"],
            "departments[].manager": ["email", "firstName", "lastName", "level", "yearsExperience"],
            "departments[].projects[]": ["budget", "name", "startDate", "status", "teamSize"]
        ]

        let rootKeys = ["departments", "employeeCount", "founded", "headquarters", "name", "type"]

        // Test root context
        var keys = tracker.getContextKeys(nestedSchemas: nestedSchemas, rootKeys: rootKeys)
        #expect(keys == rootKeys)

        // Test headquarters context
        tracker.keyDetected("headquarters")
        tracker.enterObject()
        keys = tracker.getContextKeys(nestedSchemas: nestedSchemas, rootKeys: rootKeys)
        #expect(keys == ["city", "country", "postalCode", "street"])

        // Reset and test departments array context
        tracker.reset()
        tracker.keyDetected("departments")
        tracker.enterArray()
        tracker.enterObject() // Array item
        keys = tracker.getContextKeys(nestedSchemas: nestedSchemas, rootKeys: rootKeys)
        #expect(keys == ["headCount", "manager", "name", "projects", "type"])

        // Test manager within departments
        tracker.keyDetected("manager")
        tracker.enterObject()
        keys = tracker.getContextKeys(nestedSchemas: nestedSchemas, rootKeys: rootKeys)
        #expect(keys == ["email", "firstName", "lastName", "level", "yearsExperience"])
    }

    @Test("Resets state correctly")
    func testReset() {
        let tracker = JSONContextTracker()

        // Build complex state
        tracker.keyDetected("departments")
        tracker.enterArray()
        tracker.enterObject()
        tracker.keyDetected("manager")
        tracker.enterObject()

        #expect(tracker.nestingDepth == 3)
        #expect(tracker.isInArray())

        // Reset
        tracker.reset()

        #expect(tracker.getCurrentPath() == "")
        #expect(tracker.nestingDepth == 0)
        #expect(!tracker.isInArray())
        #expect(tracker.getCurrentArrayContext() == nil)
    }
}