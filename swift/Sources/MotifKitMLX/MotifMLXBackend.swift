#if canImport(MLX) && canImport(MLXNN) && canImport(MLXLLM) && canImport(MLXLMCommon)
import Foundation
import MLX
import MLXLLM
import MLXLMCommon
import MLXNN
import MotifKit

public actor MotifMLXBackend: MotifChatBackend {
    public init() {}

    public nonisolated func streamResponse(
        messages: [MotifChatMessage],
        parameters: MotifGenerationParameters
    ) -> AsyncThrowingStream<MotifGenerationEvent, any Error> {
        AsyncThrowingStream { continuation in
            continuation.finish(throwing: MotifBackendError.nativeBackendUnavailable(
                "MLX Swift overlay is wired, but the Motif architecture port is not implemented yet. Port order: config -> PolyNorm -> MotifMLP -> GDA attention reference path -> KV cache -> sdpa_dual_v/gda_post_split kernels."
            ))
        }
    }
}

public enum MotifMLXPortStatus {
    public static let modelType = "motif"
    public static let requiresCustomKernels = [
        "polynorm",
        "gda_post_split",
        "sdpa_dual_v",
        "sdpa_dual_v_q4",
    ]
}
#endif
