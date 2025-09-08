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
    ],
    dependencies: [
        .package(url: "https://github.com/1amageek/OpenFoundationModels", branch: "main"),
        .package(url: "https://github.com/ml-explore/mlx-swift-examples.git", branch: "main"),
        .package(url: "https://github.com/huggingface/swift-transformers", .upToNextMinor(from: "0.1.23")),
    ],
    targets: [
        // PRECISE: Independent foundation module for constraint intelligence
        .target(
            name: "PRECISE",
            dependencies: [
                .product(name: "MLXLLM", package: "mlx-swift-examples"),
                .product(name: "MLXLMCommon", package: "mlx-swift-examples"),
                .product(name: "Transformers", package: "swift-transformers"),
            ],
            path: "Sources/PRECISE"
        ),
        // Main MLX adapter target that uses PRECISE
        .target(
            name: "OpenFoundationModelsMLX",
            dependencies: [
                "PRECISE",  // OpenFoundationModelsMLX uses PRECISE
                .product(name: "OpenFoundationModelsExtra", package: "OpenFoundationModels"),
                .product(name: "OpenFoundationModels", package: "OpenFoundationModels"),
                .product(name: "MLXLLM", package: "mlx-swift-examples"),
                .product(name: "MLXLMCommon", package: "mlx-swift-examples"),
                .product(name: "Transformers", package: "swift-transformers"),
            ]
        ),
        // PRECISE test target
        .testTarget(
            name: "PRECISETests",
            dependencies: ["PRECISE", "OpenFoundationModelsMLX"],
            path: "Tests/PRECISETests"
        ),
        // Main test target
        .testTarget(
            name: "OpenFoundationModelsMLXTests",
            dependencies: ["OpenFoundationModelsMLX"]
        ),
    ]
)
