// swift-tools-version: 6.0

import PackageDescription

var dependencies: [Package.Dependency] = []
var targets: [Target] = [
    .target(
        name: "MotifKit",
        path: "Sources/MotifKit"
    ),
    .executableTarget(
        name: "MotifChatApp",
        dependencies: ["MotifKit"],
        path: "Sources/MotifChatApp"
    ),
    .testTarget(
        name: "MotifKitTests",
        dependencies: ["MotifKit"],
        path: "Tests/MotifKitTests"
    ),
]

// Keep the default package lightweight and buildable on the current repo
// toolchain (Xcode 16.2 / Swift 6.0.3). Enable this overlay when actively
// porting Motif into MLX Swift:
//
//   MOTIFKIT_ENABLE_MLX=1 swift build --package-path swift --target MotifKitMLX
//
// The 2.30.6 pin is the newest mlx-swift-lm tag whose Package.swift is usable
// with Swift 6.0.x. Upgrade to 3.31.3+ once the repo toolchain is Xcode 16.3+
// / Swift 6.1+.
if Context.environment["MOTIFKIT_ENABLE_MLX"] == "1" {
    dependencies.append(contentsOf: [
        .package(url: "https://github.com/ml-explore/mlx-swift", .upToNextMinor(from: "0.30.6")),
        .package(url: "https://github.com/ml-explore/mlx-swift-lm", exact: "2.30.6"),
    ])
    targets.append(
        .target(
            name: "MotifKitMLX",
            dependencies: [
                "MotifKit",
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
                .product(name: "MLXLLM", package: "mlx-swift-lm"),
                .product(name: "MLXLMCommon", package: "mlx-swift-lm"),
            ],
            path: "Sources/MotifKitMLX"
        )
    )
}

let package = Package(
    name: "SwiftMotif",
    platforms: [
        .macOS(.v14)
    ],
    products: [
        .library(name: "MotifKit", targets: ["MotifKit"]),
        .executable(name: "MotifChatApp", targets: ["MotifChatApp"]),
    ],
    dependencies: dependencies,
    targets: targets
)

if Context.environment["MOTIFKIT_ENABLE_MLX"] == "1" {
    package.products.append(.library(name: "MotifKitMLX", targets: ["MotifKitMLX"]))
}
