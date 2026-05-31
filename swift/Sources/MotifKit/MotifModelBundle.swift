import Foundation

public struct MotifModelBundle: Equatable, Sendable {
    public static let configFileName = "config.json"
    public static let generationConfigFileName = "generation_config.json"
    public static let safetensorsFileName = "model.safetensors"
    public static let safetensorsIndexFileName = "model.safetensors.index.json"
    public static let tokenizerFileName = "tokenizer.json"
    public static let tokenizerConfigFileName = "tokenizer_config.json"
    public static let specialTokensMapFileName = "special_tokens_map.json"
    public static let chatTemplateFileName = "chat_template.jinja"

    public var directoryURL: URL
    public var configuration: MotifModelConfiguration
    public var generationConfiguration: MotifGenerationConfiguration?
    public var checkpointMetadata: MotifCheckpointMetadata
    public var tokenizerMetadata: MotifTokenizerMetadata
    public var directoryValidation: MotifModelDirectoryValidation

    public init(
        directoryURL: URL,
        configuration: MotifModelConfiguration,
        generationConfiguration: MotifGenerationConfiguration? = nil,
        checkpointMetadata: MotifCheckpointMetadata = .init(),
        tokenizerMetadata: MotifTokenizerMetadata = .init(),
        directoryValidation: MotifModelDirectoryValidation = .valid
    ) {
        self.directoryURL = directoryURL
        self.configuration = configuration
        self.generationConfiguration = generationConfiguration
        self.checkpointMetadata = checkpointMetadata
        self.tokenizerMetadata = tokenizerMetadata
        self.directoryValidation = directoryValidation
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

public struct MotifCheckpointMetadata: Equatable, Sendable {
    public var safetensorsFileURL: URL?
    public var indexFileURL: URL?
    public var shardFileNames: [String]
    public var weightMap: [String: String]
    public var indexMetadata: [String: MotifJSONValue]

    public var isSharded: Bool { indexFileURL != nil }
    public var hasCheckpointWeights: Bool { safetensorsFileURL != nil || !shardFileNames.isEmpty }
    public var tensorKeyCount: Int { weightMap.count }
    public func shardFileName(containingTensorKey tensorKey: String) -> String? {
        weightMap[tensorKey]
    }

    public init(
        safetensorsFileURL: URL? = nil,
        indexFileURL: URL? = nil,
        shardFileNames: [String] = [],
        weightMap: [String: String] = [:],
        indexMetadata: [String: MotifJSONValue] = [:]
    ) {
        self.safetensorsFileURL = safetensorsFileURL
        self.indexFileURL = indexFileURL
        self.shardFileNames = shardFileNames
        self.weightMap = weightMap
        self.indexMetadata = indexMetadata
    }
}

public struct MotifTokenizerMetadata: Equatable, Sendable {
    public var tokenizerJSONURL: URL?
    public var tokenizerConfigURL: URL?
    public var specialTokensMapURL: URL?
    public var chatTemplateURL: URL?
    public var tokenizerConfigChatTemplate: String?
    public var chatTemplateFileContents: String?
    public var generationChatTemplate: String?
    public var tokenizerConfigReadError: String?
    public var chatTemplateReadError: String?

    public var hasTokenizerFiles: Bool {
        tokenizerJSONURL != nil || (tokenizerConfigURL != nil && tokenizerConfigReadError == nil)
    }

    public var validationWarnings: [String] {
        [tokenizerConfigReadError, chatTemplateReadError].compactMap { $0 }
    }

    public var preferredChatTemplate: String? {
        generationChatTemplate ?? tokenizerConfigChatTemplate ?? chatTemplateFileContents
    }

    public var chatTemplateSourceFileName: String? {
        if generationChatTemplate != nil {
            return MotifModelBundle.generationConfigFileName
        }
        if tokenizerConfigChatTemplate != nil {
            return MotifModelBundle.tokenizerConfigFileName
        }
        if chatTemplateFileContents != nil {
            return MotifModelBundle.chatTemplateFileName
        }
        return nil
    }

    public init(
        tokenizerJSONURL: URL? = nil,
        tokenizerConfigURL: URL? = nil,
        specialTokensMapURL: URL? = nil,
        chatTemplateURL: URL? = nil,
        tokenizerConfigChatTemplate: String? = nil,
        chatTemplateFileContents: String? = nil,
        generationChatTemplate: String? = nil,
        tokenizerConfigReadError: String? = nil,
        chatTemplateReadError: String? = nil
    ) {
        self.tokenizerJSONURL = tokenizerJSONURL
        self.tokenizerConfigURL = tokenizerConfigURL
        self.specialTokensMapURL = specialTokensMapURL
        self.chatTemplateURL = chatTemplateURL
        self.tokenizerConfigChatTemplate = tokenizerConfigChatTemplate
        self.chatTemplateFileContents = chatTemplateFileContents
        self.generationChatTemplate = generationChatTemplate
        self.tokenizerConfigReadError = tokenizerConfigReadError
        self.chatTemplateReadError = chatTemplateReadError
    }
}

public struct MotifModelDirectoryValidation: Equatable, Sendable {
    public static let valid = MotifModelDirectoryValidation()

    public var missingRequiredFiles: [String]
    public var warnings: [String]

    public var isLoadableScaffold: Bool { missingRequiredFiles.isEmpty }

    public init(
        missingRequiredFiles: [String] = [],
        warnings: [String] = []
    ) {
        self.missingRequiredFiles = missingRequiredFiles
        self.warnings = warnings
    }

    public var summary: String? {
        guard !missingRequiredFiles.isEmpty || !warnings.isEmpty else { return nil }
        var parts: [String] = []
        if !missingRequiredFiles.isEmpty {
            parts.append("missing: \(missingRequiredFiles.joined(separator: ", "))")
        }
        if !warnings.isEmpty {
            parts.append("warnings: \(warnings.joined(separator: ", "))")
        }
        return parts.joined(separator: "; ")
    }

    public var blockingSummary: String? {
        guard !missingRequiredFiles.isEmpty else { return nil }
        return "missing: \(missingRequiredFiles.joined(separator: ", "))"
    }
}

public enum MotifModelBundleLoaderError: Error, LocalizedError, Equatable, Sendable {
    case missingConfig(URL)
    case unreadableConfig(URL, String)
    case unreadableGenerationConfig(URL, String)
    case unreadableCheckpointIndex(URL, String)
    case invalidCheckpointShardPath(URL, String)

    public var errorDescription: String? {
        switch self {
        case .missingConfig(let url):
            "Missing Motif config.json at \(url.path)"
        case .unreadableConfig(let url, let detail):
            "Could not read Motif config.json at \(url.path): \(detail)"
        case .unreadableGenerationConfig(let url, let detail):
            "Could not read Motif generation_config.json at \(url.path): \(detail)"
        case .unreadableCheckpointIndex(let url, let detail):
            "Could not read Motif safetensors index at \(url.path): \(detail)"
        case .invalidCheckpointShardPath(let url, let shardFileName):
            "Invalid safetensors shard path \"\(shardFileName)\" in \(url.path): shard paths must be file names inside the model directory"
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

        let checkpointMetadata = try loadCheckpointMetadata(from: directoryURL, decoder: decoder)
        let tokenizerMetadata = loadTokenizerMetadata(
            from: directoryURL,
            decoder: decoder,
            generationConfiguration: generationConfiguration
        )
        let directoryValidation = validateModelDirectory(
            checkpointMetadata: checkpointMetadata,
            tokenizerMetadata: tokenizerMetadata,
            generationConfiguration: generationConfiguration,
            fileManager: .default,
            directoryURL: directoryURL
        )

        return MotifModelBundle(
            directoryURL: directoryURL,
            configuration: configuration,
            generationConfiguration: generationConfiguration,
            checkpointMetadata: checkpointMetadata,
            tokenizerMetadata: tokenizerMetadata,
            directoryValidation: directoryValidation
        )
    }

    private static func loadCheckpointMetadata(
        from directoryURL: URL,
        decoder: JSONDecoder
    ) throws -> MotifCheckpointMetadata {
        let safetensorsURL = directoryURL.appendingPathComponent(MotifModelBundle.safetensorsFileName)
        let indexURL = directoryURL.appendingPathComponent(MotifModelBundle.safetensorsIndexFileName)
        let hasSingleFile = FileManager.default.fileExists(atPath: safetensorsURL.path)
        let hasIndex = FileManager.default.fileExists(atPath: indexURL.path)

        guard hasIndex else {
            return MotifCheckpointMetadata(
                safetensorsFileURL: hasSingleFile ? safetensorsURL : nil
            )
        }

        let index: SafetensorsIndexFile
        do {
            index = try decoder.decode(SafetensorsIndexFile.self, from: Data(contentsOf: indexURL))
        } catch {
            throw MotifModelBundleLoaderError.unreadableCheckpointIndex(
                indexURL,
                String(describing: error)
            )
        }

        let shardFileNames = try validatedShardFileNames(
            Array(Set(index.weightMap.values)).sorted(),
            indexURL: indexURL
        )

        return MotifCheckpointMetadata(
            safetensorsFileURL: hasSingleFile ? safetensorsURL : nil,
            indexFileURL: indexURL,
            shardFileNames: shardFileNames,
            weightMap: index.weightMap,
            indexMetadata: index.metadata ?? [:]
        )
    }

    private static func validatedShardFileNames(
        _ shardFileNames: [String],
        indexURL: URL
    ) throws -> [String] {
        try shardFileNames.map { shardFileName in
            guard isSafeShardFileName(shardFileName) else {
                throw MotifModelBundleLoaderError.invalidCheckpointShardPath(indexURL, shardFileName)
            }
            return shardFileName
        }
    }

    private static func isSafeShardFileName(_ shardFileName: String) -> Bool {
        guard !shardFileName.isEmpty else { return false }
        guard !shardFileName.hasPrefix("/") && !shardFileName.hasPrefix("~") else { return false }
        guard !shardFileName.contains("\\") else { return false }
        let components = shardFileName.split(separator: "/", omittingEmptySubsequences: false)
        guard components.count == 1 else { return false }
        guard !components.contains(where: { $0 == "." || $0 == ".." }) else { return false }
        return (shardFileName as NSString).lastPathComponent == shardFileName
    }

    private static func loadTokenizerMetadata(
        from directoryURL: URL,
        decoder: JSONDecoder,
        generationConfiguration: MotifGenerationConfiguration?
    ) -> MotifTokenizerMetadata {
        let tokenizerURL = existingURL(
            directoryURL.appendingPathComponent(MotifModelBundle.tokenizerFileName)
        )
        let tokenizerConfigURL = existingURL(
            directoryURL.appendingPathComponent(MotifModelBundle.tokenizerConfigFileName)
        )
        let specialTokensURL = existingURL(
            directoryURL.appendingPathComponent(MotifModelBundle.specialTokensMapFileName)
        )
        let chatTemplateURL = existingURL(
            directoryURL.appendingPathComponent(MotifModelBundle.chatTemplateFileName)
        )

        let tokenizerConfigChatTemplate: String?
        let tokenizerConfigReadError: String?
        if let tokenizerConfigURL {
            do {
                let data = try Data(contentsOf: tokenizerConfigURL)
                let config = try decoder.decode(TokenizerConfigFile.self, from: data)
                tokenizerConfigChatTemplate = config.chatTemplate
                tokenizerConfigReadError = nil
            } catch {
                tokenizerConfigChatTemplate = nil
                tokenizerConfigReadError = "\(MotifModelBundle.tokenizerConfigFileName) unreadable: \(String(describing: error))"
            }
        } else {
            tokenizerConfigChatTemplate = nil
            tokenizerConfigReadError = nil
        }

        let chatTemplateFileContents: String?
        let chatTemplateReadError: String?
        if let chatTemplateURL {
            do {
                chatTemplateFileContents = try String(contentsOf: chatTemplateURL, encoding: .utf8)
                chatTemplateReadError = nil
            } catch {
                chatTemplateFileContents = nil
                chatTemplateReadError = "\(MotifModelBundle.chatTemplateFileName) unreadable: \(String(describing: error))"
            }
        } else {
            chatTemplateFileContents = nil
            chatTemplateReadError = nil
        }

        return MotifTokenizerMetadata(
            tokenizerJSONURL: tokenizerURL,
            tokenizerConfigURL: tokenizerConfigURL,
            specialTokensMapURL: specialTokensURL,
            chatTemplateURL: chatTemplateURL,
            tokenizerConfigChatTemplate: tokenizerConfigChatTemplate,
            chatTemplateFileContents: chatTemplateFileContents,
            generationChatTemplate: generationConfiguration?.chatTemplate,
            tokenizerConfigReadError: tokenizerConfigReadError,
            chatTemplateReadError: chatTemplateReadError
        )
    }

    private static func validateModelDirectory(
        checkpointMetadata: MotifCheckpointMetadata,
        tokenizerMetadata: MotifTokenizerMetadata,
        generationConfiguration: MotifGenerationConfiguration?,
        fileManager: FileManager,
        directoryURL: URL
    ) -> MotifModelDirectoryValidation {
        var missingRequiredFiles: [String] = []
        var warnings: [String] = []

        if !checkpointMetadata.hasCheckpointWeights {
            missingRequiredFiles.append(
                "\(MotifModelBundle.safetensorsFileName) or \(MotifModelBundle.safetensorsIndexFileName)"
            )
        }

        for shardFileName in checkpointMetadata.shardFileNames {
            let shardURL = directoryURL.appendingPathComponent(shardFileName)
            if !fileManager.fileExists(atPath: shardURL.path) {
                missingRequiredFiles.append(shardFileName)
            }
        }

        if !tokenizerMetadata.hasTokenizerFiles {
            missingRequiredFiles.append(
                "\(MotifModelBundle.tokenizerFileName) or readable \(MotifModelBundle.tokenizerConfigFileName)"
            )
        }

        warnings.append(contentsOf: tokenizerMetadata.validationWarnings)

        if generationConfiguration == nil {
            warnings.append("\(MotifModelBundle.generationConfigFileName) not found")
        }

        if tokenizerMetadata.preferredChatTemplate == nil {
            warnings.append("chat template not found")
        }

        return MotifModelDirectoryValidation(
            missingRequiredFiles: missingRequiredFiles,
            warnings: warnings
        )
    }

    private static func existingURL(_ url: URL) -> URL? {
        FileManager.default.fileExists(atPath: url.path) ? url : nil
    }
}

private struct SafetensorsIndexFile: Decodable {
    var metadata: [String: MotifJSONValue]?
    var weightMap: [String: String]

    private enum CodingKeys: String, CodingKey {
        case metadata
        case weightMap = "weight_map"
    }
}

private struct TokenizerConfigFile: Decodable {
    var chatTemplate: String?

    private enum CodingKeys: String, CodingKey {
        case chatTemplate = "chat_template"
    }
}
