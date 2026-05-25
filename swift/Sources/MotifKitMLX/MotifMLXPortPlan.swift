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
    public var mlxModelConfiguration: ModelConfiguration?
    public var layerPlan: MotifMLXLayerPlan?
    public var validationErrorDescription: String?

    public init(
        configuration: MotifModelConfiguration,
        featureFlags: MotifRuntimeFeatureFlags = .init(),
        modelDirectory: URL? = nil,
        extraEOSTokenIDs: [Int] = []
    ) {
        self.modelType = configuration.modelType
        self.registryKey = MotifMLXModelRegistry.registryKey(for: configuration)
        self.attentionVariant = configuration.attentionVariant
        self.featureFlags = featureFlags
        self.requiredKernelNames = configuration.requiredCustomKernelNames
        self.extraEOSTokenIDs = Set(extraEOSTokenIDs)
        self.modelDirectory = modelDirectory
        if let modelDirectory {
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
    }

    public init(bundle: MotifModelBundle, featureFlags: MotifRuntimeFeatureFlags = .init()) {
        self.init(
            configuration: bundle.configuration,
            featureFlags: featureFlags,
            modelDirectory: bundle.directoryURL,
            extraEOSTokenIDs: bundle.extraEOSTokenIDs
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
