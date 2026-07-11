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

    /// Keeps security-scoped directory access alive through deferred runtime
    /// loading and every subsequent backend read. Releasing the backend releases
    /// the lease; callers that do not need scoped access can continue omitting it.
    private let directoryAccessLease: NativeDirectoryAccessLease?
    private var nativeRuntime: MotifMLXNativeRuntime?
    private var nativeRuntimeLoad: (
        id: UUID,
        task: Task<MotifMLXNativeRuntime, Error>
    )?

    public init(
        configuration: MotifModelConfiguration? = nil,
        featureFlags: MotifRuntimeFeatureFlags = .fromEnvironment()
    ) {
        self.configuration = configuration
        self.featureFlags = featureFlags
        self.modelDirectory = nil
        self.directoryAccessLease = nil
        let plan = configuration.map {
            MotifMLXLoadPlan(configuration: $0, featureFlags: featureFlags)
        }
        self.loadPlan = plan
        self.layerPlan = plan?.layerPlan
        self.decoderGraphPlan = plan?.layerPlan?.decoderGraphPlan
    }

    public init(
        modelDirectory: URL,
        featureFlags: MotifRuntimeFeatureFlags = .fromEnvironment(),
        directoryAccessLease: NativeDirectoryAccessLease? = nil
    ) throws {
        let bundle = try MotifModelBundle(directoryURL: modelDirectory)
        self.configuration = bundle.configuration
        self.featureFlags = featureFlags
        self.modelDirectory = modelDirectory
        self.directoryAccessLease = directoryAccessLease
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
            let task = Task {
                do {
                    let runtime = try await self.runtime(modelDirectory: modelDirectory)
                    try Task.checkCancellation()
                    let stream = runtime.streamResponse(messages: messages, parameters: parameters)
                    for try await event in stream {
                        try Task.checkCancellation()
                        continuation.yield(event)
                    }
                    continuation.finish()
                } catch {
                    continuation.finish(throwing: error)
                }
            }
            // Forward consumer cancellation (e.g. the UI "stop" button) down to
            // this wrapper task. Cancelling it terminates the `for try await`
            // over the runtime stream, which fires that stream's own
            // onTermination and cancels the underlying GPU token loop. Without
            // this, an aborted turn keeps decoding to maxTokens/EOS.
            continuation.onTermination = { _ in task.cancel() }
        }
    }

    private func runtime(modelDirectory: URL) async throws -> MotifMLXNativeRuntime {
        if let nativeRuntime { return nativeRuntime }
        if let nativeRuntimeLoad {
            return try await finishRuntimeLoad(nativeRuntimeLoad)
        }

        // Runtime loading may outlive a cancelled response consumer. Keep one
        // producer task cached so a replacement turn joins the same filesystem
        // work instead of starting a second checkpoint load.
        let loadID = UUID()
        let featureFlags = featureFlags
        let loadTask = Task {
            try await MotifMLXNativeRuntime.load(
                modelDirectory: modelDirectory,
                featureFlags: featureFlags
            )
        }
        let load = (id: loadID, task: loadTask)
        nativeRuntimeLoad = load
        return try await finishRuntimeLoad(load)
    }

    private func finishRuntimeLoad(
        _ load: (id: UUID, task: Task<MotifMLXNativeRuntime, Error>)
    ) async throws -> MotifMLXNativeRuntime {
        do {
            let loaded = try await load.task.value
            if nativeRuntimeLoad?.id == load.id {
                nativeRuntime = loaded
                nativeRuntimeLoad = nil
            }
            return nativeRuntime ?? loaded
        } catch {
            if nativeRuntimeLoad?.id == load.id {
                nativeRuntimeLoad = nil
            }
            throw error
        }
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
