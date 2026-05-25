#if canImport(MLX) && canImport(MLXNN) && canImport(MLXLLM) && canImport(MLXLMCommon)
import Foundation
import MLX
import MLXLLM
import MLXLMCommon
import MLXNN
import MotifKit

public actor MotifMLXBackend: MotifChatBackend {
    public nonisolated let configuration: MotifModelConfiguration?
    public nonisolated let featureFlags: MotifRuntimeFeatureFlags
    public nonisolated let loadPlan: MotifMLXLoadPlan?
    public nonisolated let layerPlan: MotifMLXLayerPlan?

    public init(
        configuration: MotifModelConfiguration? = nil,
        featureFlags: MotifRuntimeFeatureFlags = .init()
    ) {
        self.configuration = configuration
        self.featureFlags = featureFlags
        let plan = configuration.map {
            MotifMLXLoadPlan(configuration: $0, featureFlags: featureFlags)
        }
        self.loadPlan = plan
        self.layerPlan = plan?.layerPlan
    }

    public init(
        modelDirectory: URL,
        featureFlags: MotifRuntimeFeatureFlags = .init()
    ) throws {
        let bundle = try MotifModelBundle(directoryURL: modelDirectory)
        self.configuration = bundle.configuration
        self.featureFlags = featureFlags
        let plan = MotifMLXModelRegistry.loadPlan(for: bundle, featureFlags: featureFlags)
        self.loadPlan = plan
        self.layerPlan = plan.layerPlan
    }

    public nonisolated func streamResponse(
        messages: [MotifChatMessage],
        parameters: MotifGenerationParameters
    ) -> AsyncThrowingStream<MotifGenerationEvent, any Error> {
        AsyncThrowingStream { continuation in
            let planDetail = layerPlan.map {
                "Layer plan ready for \($0.configuration.modelType) (\($0.attentionLayout.variant.rawValue)); "
            } ?? ""
            continuation.finish(throwing: MotifBackendError.nativeBackendUnavailable(
                "\(planDetail)MLX Swift overlay has config, PolyNorm, and grouped attention/cache reference scaffolds, but full Motif generation is not implemented yet. Remaining port order: decoder layer wiring -> model registry/load -> sdpa_dual_v/gda_post_split Metal parity -> quantized cache kernels."
            ))
        }
    }
}

public enum MotifMLXPortStatus {
    public static let modelType = "motif"
    public static let requiresCustomKernels = MotifMetalKernels.descriptors.map(\.pythonSymbol)
    public static let customKernelDescriptors = MotifMetalKernels.descriptors
}
#endif
