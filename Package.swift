// swift-tools-version: 6.2
// The swift-tools-version declares the minimum version of Swift required to build this package.

import PackageDescription

let package = Package(
    name: "OpenFoundationModels-MLX",
    platforms: [.iOS(.v18), .macOS(.v15)],
    products: [
        // Products define the executables and libraries a package produces, making them visible to other packages.
        .library(
            name: "OpenFoundationModelsMLX",
            targets: ["OpenFoundationModelsMLX"]
        ),
        .executable(
            name: "generable-test",
            targets: ["generable-test-cli"]
        ),
    ],
    dependencies: [
        .package(url: "https://github.com/1amageek/OpenFoundationModels", branch: "main"),
        .package(url: "https://github.com/ml-explore/mlx-swift-examples.git", branch: "main"),
        .package(url: "https://github.com/huggingface/swift-transformers", from: "0.1.23"),
    ],
    targets: [
        // Main MLX adapter target
        .target(
            name: "OpenFoundationModelsMLX",
            dependencies: [
                .product(name: "OpenFoundationModelsExtra", package: "OpenFoundationModels"),
                .product(name: "OpenFoundationModels", package: "OpenFoundationModels"),
                .product(name: "MLXLLM", package: "mlx-swift-examples"),
                .product(name: "MLXLMCommon", package: "mlx-swift-examples"),
                .product(name: "Transformers", package: "swift-transformers"),
            ]
        ),
        // Generable test CLI executable
        .executableTarget(
            name: "generable-test-cli",
            dependencies: [
                "OpenFoundationModelsMLX",
                .product(name: "OpenFoundationModels", package: "OpenFoundationModels"),
                .product(name: "OpenFoundationModelsExtra", package: "OpenFoundationModels"),
                .product(name: "OpenFoundationModelsMacros", package: "OpenFoundationModels"),
                .product(name: "MLXLMCommon", package: "mlx-swift-examples"),
            ]
        ),
        // Main test target
        .testTarget(
            name: "OpenFoundationModelsMLXTests",
            dependencies: ["OpenFoundationModelsMLX"]
        ),
    ]
)
