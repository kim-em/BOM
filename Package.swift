// swift-tools-version: 5.9
import PackageDescription

let package = Package(
    name: "BOMRadar",
    platforms: [.macOS(.v13)],
    targets: [
        .executableTarget(name: "BOMRadar")
    ]
)
