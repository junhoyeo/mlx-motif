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
    public nonisolated let decoderGraphPlan: MotifMLXDecoderGraphPlan?

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
        self.decoderGraphPlan = plan?.layerPlan?.decoderGraphPlan
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
        self.decoderGraphPlan = plan.layerPlan?.decoderGraphPlan
    }

    public nonisolated var capabilityLabels: [MotifMLXCapabilityLabel] {
        decoderGraphPlan?.capabilityLabels ?? [.stillUnavailable]
    }

    public nonisolated func streamResponse(
        messages: [MotifChatMessage],
        parameters: MotifGenerationParameters
    ) -> AsyncThrowingStream<MotifGenerationEvent, any Error> {
        AsyncThrowingStream { continuation in
            let planDetail = layerPlan.map {
                "Layer plan ready for \($0.configuration.modelType) (\($0.attentionLayout.variant.rawValue)); "
            } ?? ""
            let capabilityDetail = "Capability labels: \(capabilityLabels.map(\.rawValue).joined(separator: ", ")); "
            let validationDetail = loadPlan?.validationErrorDescription.map {
                "Directory validation: \($0); "
            } ?? ""
            continuation.finish(throwing: MotifBackendError.nativeBackendUnavailable(
                "\(planDetail)\(validationDetail)\(capabilityDetail)MLX Swift overlay now has a buildable decoder graph scaffold (embeddings, decoder layers, final norm, lm head), but MotifMLXBackend does not claim runtime-generated output yet. Remaining port order: replace attention scaffold with full differential attention -> tokenizer/load integration -> sdpa_dual_v/gda_post_split Metal parity -> quantized cache kernels."
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
