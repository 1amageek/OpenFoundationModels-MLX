import Foundation

// The exact JSON that should be extracted from the final channel
let jsonString = """
{"name":"NovaTech Labs","founded":2020,"type":"startup","employeeCount":45,"headquarters":{"street":"12 Innovation Drive","city":"San Francisco","country":"USA","postalCode":"94107"},"departments":[{"name":"Engineering","type":"engineering","headCount":25,"manager":{"firstName":"Alice","lastName":"Chen","email":"alice.chen@novatech.com","level":"lead","yearsExperience":8},"projects":[{"name":"Quantum AI Platform","status":"active","startDate":"2023-01-10","teamSize":12,"budget":500000.0},{"name":"Edge Computing SDK","status":"planning","startDate":"2024-02-01","teamSize":6,"budget":null}]},{"name":"Sales","type":"sales","headCount":20,"manager":{"firstName":"Mark","lastName":"Rogers","email":"mark.rogers@novatech.com","level":"manager","yearsExperience":10},"projects":[{"name":"Global Market Expansion","status":"active","startDate":"2023-06-15","teamSize":8,"budget":300000.0}]}]}
"""

print("Testing JSON parsing...")
print(String(repeating: "=", count: 60))

print("JSON string length: \(jsonString.count) characters")
print("\nFirst 100 characters:")
print(String(jsonString.prefix(100)))

if let data = jsonString.data(using: .utf8) {
    print("\nData size: \(data.count) bytes")
    
    do {
        let json = try JSONSerialization.jsonObject(with: data, options: []) as? [String: Any]
        print("\n✅ JSON parsed successfully!")
        print("Company name: \(json?["name"] as? String ?? "nil")")
        print("Founded: \(json?["founded"] as? Int ?? 0)")
        print("Type: \(json?["type"] as? String ?? "nil")")
        print("Employee count: \(json?["employeeCount"] as? Int ?? 0)")
        
        // Check departments
        if let departments = json?["departments"] as? [[String: Any]] {
            print("\nDepartments: \(departments.count)")
            for dept in departments {
                print("  - \(dept["name"] as? String ?? "unknown"): \(dept["headCount"] as? Int ?? 0) employees")
            }
        }
    } catch {
        print("\n❌ JSON parsing failed: \(error)")
        print("\nError details:")
        if let nsError = error as NSError? {
            print("  Domain: \(nsError.domain)")
            print("  Code: \(nsError.code)")
            print("  Description: \(nsError.localizedDescription)")
        }
    }
} else {
    print("❌ Failed to convert string to data")
}