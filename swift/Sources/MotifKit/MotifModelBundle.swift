import Foundation

public struct MotifModelBundle: Equatable, Sendable {
    public static let configFileName = "config.json"
    public static let generationConfigFileName = "generation_config.json"

    public var directoryURL: URL
    public var configuration: MotifModelConfiguration
    public var generationConfiguration: MotifGenerationConfiguration?

    public init(
        directoryURL: URL,
        configuration: MotifModelConfiguration,
        generationConfiguration: MotifGenerationConfiguration? = nil
    ) {
        self.directoryURL = directoryURL
        self.configuration = configuration
        self.generationConfiguration = generationConfiguration
    }

    public init(directoryURL: URL) throws {
        self = try MotifModelBundleLoader.loadMetadata(from: directoryURL)
    }

    public var extraEOSTokenIDs: [Int] {
        guard let generationConfiguration else { return [] }
        let configured = Set([configuration.eosTokenId].compactMap { $0 })
        return generationConfiguration.eosTokenIds.filter { !configured.contains($0) }
    }
}

public enum MotifModelBundleLoaderError: Error, LocalizedError, Equatable, Sendable {
    case missingConfig(URL)
    case unreadableConfig(URL, String)
    case unreadableGenerationConfig(URL, String)

    public var errorDescription: String? {
        switch self {
        case .missingConfig(let url):
            "Missing Motif config.json at \(url.path)"
        case .unreadableConfig(let url, let detail):
            "Could not read Motif config.json at \(url.path): \(detail)"
        case .unreadableGenerationConfig(let url, let detail):
            "Could not read Motif generation_config.json at \(url.path): \(detail)"
        }
    }
}

public enum MotifModelBundleLoader {
    public static func loadMetadata(from directoryURL: URL) throws -> MotifModelBundle {
        let configURL = directoryURL.appendingPathComponent(MotifModelBundle.configFileName)
        guard FileManager.default.fileExists(atPath: configURL.path) else {
            throw MotifModelBundleLoaderError.missingConfig(configURL)
        }

        let decoder = JSONDecoder()
        let configuration: MotifModelConfiguration
        do {
            configuration = try decoder.decode(
                MotifModelConfiguration.self,
                from: Data(contentsOf: configURL)
            )
        } catch {
            throw MotifModelBundleLoaderError.unreadableConfig(configURL, String(describing: error))
        }

        let generationURL = directoryURL.appendingPathComponent(MotifModelBundle.generationConfigFileName)
        let generationConfiguration: MotifGenerationConfiguration?
        if FileManager.default.fileExists(atPath: generationURL.path) {
            do {
                generationConfiguration = try decoder.decode(
                    MotifGenerationConfiguration.self,
                    from: Data(contentsOf: generationURL)
                )
            } catch {
                throw MotifModelBundleLoaderError.unreadableGenerationConfig(
                    generationURL,
                    String(describing: error)
                )
            }
        } else {
            generationConfiguration = nil
        }

        return MotifModelBundle(
            directoryURL: directoryURL,
            configuration: configuration,
            generationConfiguration: generationConfiguration
        )
    }
}
