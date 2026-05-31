#if canImport(MLX) && canImport(MLXNN) && canImport(MLXLLM) && canImport(MLXLMCommon)
import Foundation
import MLX
import MLXLLM
import MLXLMCommon
import MLXNN
import MotifKit

public struct MotifMLXLayerPlan: Codable, Equatable, Sendable {
    public var configuration: MotifModelConfiguration
    public var attentionLayout: MotifAttentionLayout
    public var groupedAttentionReferencePlan: MotifGroupedAttentionReferencePlan?
    public var mlpLayout: MotifMLPLayout
    public var kernelNames: [String]
    public var decoderGraphPlan: MotifMLXDecoderGraphPlan

    public init(configuration: MotifModelConfiguration) throws {
        self.configuration = configuration
        self.attentionLayout = try MotifAttentionLayout(configuration: configuration)
        if attentionLayout.variant == .groupedDifferential {
            self.groupedAttentionReferencePlan = try MotifGroupedAttentionReferencePlan(
                configuration: configuration
            )
        } else {
            self.groupedAttentionReferencePlan = nil
        }
        self.mlpLayout = MotifMLPLayout(configuration: configuration)
        self.kernelNames = MotifMetalKernelRegistry.required.map(\.name)
        self.decoderGraphPlan = MotifMLXDecoderGraphPlan(
            configuration: configuration,
            attentionLayout: attentionLayout,
            cacheKind: attentionLayout.variant == .groupedDifferential ? .groupedFourSlot : .standard
        )
    }
}

public struct MotifMLXLoadPlan: Equatable, Sendable {
    public var modelType: String
    public var registryKey: String
    public var attentionVariant: MotifAttentionVariant
    public var featureFlags: MotifRuntimeFeatureFlags
    public var requiredKernelNames: [String]
    public var extraEOSTokenIDs: Set<Int>
    public var modelDirectory: URL?
    public var checkpointMetadata: MotifCheckpointMetadata?
    public var tokenizerMetadata: MotifTokenizerMetadata?
    public var directoryValidation: MotifModelDirectoryValidation?
    public var chatTemplate: String?
    public var mlxModelConfiguration: ModelConfiguration?
    public var layerPlan: MotifMLXLayerPlan?
    public var validationErrorDescription: String?

    public init(
        configuration: MotifModelConfiguration,
        featureFlags: MotifRuntimeFeatureFlags = .init(),
        modelDirectory: URL? = nil,
        extraEOSTokenIDs: [Int] = [],
        checkpointMetadata: MotifCheckpointMetadata? = nil,
        tokenizerMetadata: MotifTokenizerMetadata? = nil,
        directoryValidation: MotifModelDirectoryValidation? = nil
    ) {
        self.modelType = configuration.modelType
        self.registryKey = MotifMLXModelRegistry.registryKey(for: configuration)
        self.attentionVariant = configuration.attentionVariant
        self.featureFlags = featureFlags
        self.requiredKernelNames = configuration.requiredCustomKernelNames
        self.extraEOSTokenIDs = Set(extraEOSTokenIDs)
        self.modelDirectory = modelDirectory
        self.checkpointMetadata = checkpointMetadata
        self.tokenizerMetadata = tokenizerMetadata
        self.directoryValidation = directoryValidation
        self.chatTemplate = tokenizerMetadata?.preferredChatTemplate
        if let modelDirectory,
           directoryValidation?.isLoadableScaffold != false
        {
            self.mlxModelConfiguration = MotifMLXModelRegistry.modelConfiguration(
                directory: modelDirectory,
                bundleConfiguration: configuration,
                extraEOSTokenIDs: Set(extraEOSTokenIDs)
            )
        } else {
            self.mlxModelConfiguration = nil
        }

        do {
            self.layerPlan = try MotifMLXModelRegistry.layerPlan(for: configuration)
            self.validationErrorDescription = nil
        } catch {
            self.layerPlan = nil
            self.validationErrorDescription = String(describing: error)
        }
        if self.validationErrorDescription == nil {
            self.validationErrorDescription = directoryValidation?.blockingSummary
        }
    }

    public init(bundle: MotifModelBundle, featureFlags: MotifRuntimeFeatureFlags = .init()) {
        self.init(
            configuration: bundle.configuration,
            featureFlags: featureFlags,
            modelDirectory: bundle.directoryURL,
            extraEOSTokenIDs: bundle.extraEOSTokenIDs,
            checkpointMetadata: bundle.checkpointMetadata,
            tokenizerMetadata: bundle.tokenizerMetadata,
            directoryValidation: bundle.directoryValidation
        )
    }
}

public enum MotifMLXModelRegistry {
    public static let modelType = "motif"
    public static let defaultPrompt = "<|user|>Hello<|endofturn|><|assistant|><think>\n"

    public static func accepts(_ configuration: MotifModelConfiguration) -> Bool {
        configuration.modelType == modelType
    }

    public static func registryKey(for configuration: MotifModelConfiguration) -> String {
        accepts(configuration) ? modelType : configuration.modelType
    }

    public static func layerPlan(for configuration: MotifModelConfiguration) throws -> MotifMLXLayerPlan {
        try MotifMLXLayerPlan(configuration: configuration)
    }

    public static func loadPlan(
        for bundle: MotifModelBundle,
        featureFlags: MotifRuntimeFeatureFlags = .init()
    ) -> MotifMLXLoadPlan {
        MotifMLXLoadPlan(bundle: bundle, featureFlags: featureFlags)
    }

    public static func modelConfiguration(
        directory: URL,
        bundleConfiguration: MotifModelConfiguration,
        extraEOSTokenIDs: Set<Int>
    ) -> ModelConfiguration {
        var eosTokenIDs = extraEOSTokenIDs
        if let eosTokenID = bundleConfiguration.eosTokenId {
            eosTokenIDs.insert(eosTokenID)
        }
        return ModelConfiguration(
            directory: directory,
            defaultPrompt: defaultPrompt,
            eosTokenIds: eosTokenIDs
        )
    }
}
#endif
