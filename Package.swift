// swift-tools-version: 6.2
// The swift-tools-version declares the minimum version of Swift required to build this package.

import PackageDescription

let package = Package(
    name: "OpenFoundationModels-MLX",
    platforms: [.iOS(.v18), .macOS(.v15)],
    products: [
        // Core MLX adapter library
        .library(
            name: "OpenFoundationModelsMLX",
            targets: ["OpenFoundationModelsMLX"]
        ),
        // GPT model cards and parsers
        .library(
            name: "OpenFoundationModelsMLXGPT",
            targets: ["OpenFoundationModelsMLXGPT"]
        ),
        // Llama model cards
        .library(
            name: "OpenFoundationModelsMLXLlama",
            targets: ["OpenFoundationModelsMLXLlama"]
        ),
        // Gemma model cards (FunctionGemma)
        .library(
            name: "OpenFoundationModelsMLXGemma",
            targets: ["OpenFoundationModelsMLXGemma"]
        ),
        // Utilities (ModelLoader)
        .library(
            name: "OpenFoundationModelsMLXUtils",
            targets: ["OpenFoundationModelsMLXUtils"]
        ),
    ],
    dependencies: [
        .package(url: "https://github.com/1amageek/OpenFoundationModels", from: "1.0.0"),
        .package(url: "https://github.com/ml-explore/mlx-swift-lm.git", branch: "main"),
        .package(url: "https://github.com/huggingface/swift-transformers", from: "1.1.0"),
    ],
    targets: [
        // Main MLX adapter target
        .target(
            name: "OpenFoundationModelsMLX",
            dependencies: [
                .product(name: "OpenFoundationModelsExtra", package: "OpenFoundationModels"),
                .product(name: "OpenFoundationModels", package: "OpenFoundationModels"),
                .product(name: "MLXLLM", package: "mlx-swift-lm"),
                .product(name: "MLXLMCommon", package: "mlx-swift-lm"),
                .product(name: "Transformers", package: "swift-transformers"),
            ]
        ),
        // GPT model cards and parsers
        .target(
            name: "OpenFoundationModelsMLXGPT",
            dependencies: [
                "OpenFoundationModelsMLX",
                .product(name: "OpenFoundationModels", package: "OpenFoundationModels"),
                .product(name: "OpenFoundationModelsExtra", package: "OpenFoundationModels"),
                .product(name: "MLXLMCommon", package: "mlx-swift-lm"),
                .product(name: "MLXLLM", package: "mlx-swift-lm"),
            ]
        ),
        // Llama model cards
        .target(
            name: "OpenFoundationModelsMLXLlama",
            dependencies: [
                "OpenFoundationModelsMLX",
                .product(name: "OpenFoundationModels", package: "OpenFoundationModels"),
                .product(name: "OpenFoundationModelsExtra", package: "OpenFoundationModels"),
                .product(name: "MLXLMCommon", package: "mlx-swift-lm"),
            ]
        ),
        // Gemma model cards (FunctionGemma)
        .target(
            name: "OpenFoundationModelsMLXGemma",
            dependencies: [
                "OpenFoundationModelsMLX",
                .product(name: "OpenFoundationModels", package: "OpenFoundationModels"),
                .product(name: "OpenFoundationModelsExtra", package: "OpenFoundationModels"),
                .product(name: "MLXLMCommon", package: "mlx-swift-lm"),
            ]
        ),
        // Utilities (ModelLoader)
        .target(
            name: "OpenFoundationModelsMLXUtils",
            dependencies: [
                .product(name: "MLXLMCommon", package: "mlx-swift-lm"),
                .product(name: "MLXLLM", package: "mlx-swift-lm"),
            ]
        ),
        // Main test target
        .testTarget(
            name: "OpenFoundationModelsMLXTests",
            dependencies: [
                "OpenFoundationModelsMLX",
                "OpenFoundationModelsMLXGPT",
                "OpenFoundationModelsMLXLlama",
                "OpenFoundationModelsMLXGemma",
                "OpenFoundationModelsMLXUtils"
            ]
        ),
    ]
)
