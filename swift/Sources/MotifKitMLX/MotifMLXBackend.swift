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
    public nonisolated let modelDirectory: URL?

    private var nativeRuntime: MotifMLXNativeRuntime?

    public init(
        configuration: MotifModelConfiguration? = nil,
        featureFlags: MotifRuntimeFeatureFlags = .fromEnvironment()
    ) {
        self.configuration = configuration
        self.featureFlags = featureFlags
        self.modelDirectory = nil
        let plan = configuration.map {
            MotifMLXLoadPlan(configuration: $0, featureFlags: featureFlags)
        }
        self.loadPlan = plan
        self.layerPlan = plan?.layerPlan
        self.decoderGraphPlan = plan?.layerPlan?.decoderGraphPlan
    }

    public init(
        modelDirectory: URL,
        featureFlags: MotifRuntimeFeatureFlags = .fromEnvironment()
    ) throws {
        let bundle = try MotifModelBundle(directoryURL: modelDirectory)
        self.configuration = bundle.configuration
        self.featureFlags = featureFlags
        self.modelDirectory = modelDirectory
        let plan = MotifMLXModelRegistry.loadPlan(for: bundle, featureFlags: featureFlags)
        self.loadPlan = plan
        self.layerPlan = plan.layerPlan
        self.decoderGraphPlan = plan.layerPlan?.decoderGraphPlan
    }

    public nonisolated var capabilityLabels: [MotifMLXCapabilityLabel] {
        if modelDirectory != nil, loadPlan?.validationErrorDescription == nil {
            return [.buildableScaffold, .fixtureProvenSemanticParity, .runtimeGeneratedOutput]
        }
        return decoderGraphPlan?.capabilityLabels ?? [.stillUnavailable]
    }

    public nonisolated func streamResponse(
        messages: [MotifChatMessage],
        parameters: MotifGenerationParameters
    ) -> AsyncThrowingStream<MotifGenerationEvent, any Error> {
        guard let modelDirectory else {
            return unavailableStream(
                "MotifMLXBackend was created from configuration only. Provide a converted MLX model directory to load tokenizer, safetensors, and generation config before streaming native tokens."
            )
        }
        if let validationErrorDescription = loadPlan?.validationErrorDescription {
            return unavailableStream("Directory validation: \(validationErrorDescription)")
        }

        return AsyncThrowingStream { continuation in
            Task {
                do {
                    let runtime = try await self.runtime(modelDirectory: modelDirectory)
                    let stream = runtime.streamResponse(messages: messages, parameters: parameters)
                    for try await event in stream {
                        continuation.yield(event)
                    }
                    continuation.finish()
                } catch {
                    continuation.finish(throwing: error)
                }
            }
        }
    }

    private func runtime(modelDirectory: URL) async throws -> MotifMLXNativeRuntime {
        if let nativeRuntime { return nativeRuntime }
        let loaded = try await MotifMLXNativeRuntime.load(modelDirectory: modelDirectory, featureFlags: featureFlags)
        nativeRuntime = loaded
        return loaded
    }

    private nonisolated func unavailableStream(
        _ detail: String
    ) -> AsyncThrowingStream<MotifGenerationEvent, any Error> {
        AsyncThrowingStream { continuation in
            let planDetail = layerPlan.map {
                "Layer plan ready for \($0.configuration.modelType) (\($0.attentionLayout.variant.rawValue)); "
            } ?? ""
            let capabilityDetail = "Capability labels: \(capabilityLabels.map(\.rawValue).joined(separator: ", ")); "
            continuation.finish(throwing: MotifBackendError.nativeBackendUnavailable(
                "\(planDetail)\(capabilityDetail)\(detail)"
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
