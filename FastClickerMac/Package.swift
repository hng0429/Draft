// swift-tools-version:5.9
import PackageDescription

let package = Package(
    name: "FastClicker",
    platforms: [.macOS(.v13)],
    targets: [
        .executableTarget(
            name: "FastClicker",
            path: "Sources/FastClicker"
        ),
    ]
)
