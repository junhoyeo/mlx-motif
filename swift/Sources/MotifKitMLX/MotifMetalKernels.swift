#if canImport(MLX)
import Foundation
import MLX

public enum MotifMetalKernelExecutionMode: Sendable {
    /// Keep the native port on pure MLX Swift reference ops. This is the safe
    /// default until each kernel has golden-fixture parity with Python.
    case referenceOnly

    /// Allow the wrapped MLX custom Metal kernel for kernels whose Swift wrapper
    /// has parity coverage. Callers should opt in only from benchmark/parity runs.
    case experimentalMetal
}

public enum MotifMetalKernelStatus: String, Sendable {
    case referenceReady
    case wrapperScaffolded
    case parityPending
}

public enum MotifMetalKernelName: String, CaseIterable, Sendable {
    case polynorm
    case gdaPostSplit = "gda_post_split"
    case sdpaDualV = "sdpa_dual_v"
    case sdpaDualVQ4 = "sdpa_dual_v_q4"
}

public struct MotifMetalKernelDescriptor: Sendable, Equatable {
    public let name: MotifMetalKernelName
    public let pythonSymbol: String
    public let pythonSource: String
    public let swiftWrapper: String
    public let status: MotifMetalKernelStatus
    public let defaultEnabled: Bool
    public let parityFixture: String
    public let benchmarkShape: String

    public init(
        name: MotifMetalKernelName,
        pythonSymbol: String,
        pythonSource: String,
        swiftWrapper: String,
        status: MotifMetalKernelStatus,
        defaultEnabled: Bool,
        parityFixture: String,
        benchmarkShape: String
    ) {
        self.name = name
        self.pythonSymbol = pythonSymbol
        self.pythonSource = pythonSource
        self.swiftWrapper = swiftWrapper
        self.status = status
        self.defaultEnabled = defaultEnabled
        self.parityFixture = parityFixture
        self.benchmarkShape = benchmarkShape
    }
}

public enum MotifMetalKernels {
    public static let experimentalOptInEnvironmentVariable =
        "MOTIFKIT_ENABLE_EXPERIMENTAL_METAL_KERNELS"

    /// Swift-side manifest that mirrors `src/mlx_motif/kernels`. All entries
    /// stay disabled by default so the app cannot silently route through an
    /// unverified native Metal implementation.
    public static let descriptors: [MotifMetalKernelDescriptor] = [
        MotifMetalKernelDescriptor(
            name: .polynorm,
            pythonSymbol: "polynorm",
            pythonSource: "src/mlx_motif/kernels/mlp.py",
            swiftWrapper: "MotifPolynorm.apply",
            status: .wrapperScaffolded,
            defaultEnabled: false,
            parityFixture: "polynorm small-row + decode hidden-size golden tensors",
            benchmarkShape: "(..., D) with D in {128, 4096}"
        ),
        MotifMetalKernelDescriptor(
            name: .gdaPostSplit,
            pythonSymbol: "gda_post_split",
            pythonSource: "src/mlx_motif/kernels/gda.py",
            swiftWrapper: "planned MotifGDAPostSplit.apply",
            status: .parityPending,
            defaultEnabled: false,
            parityFixture: "attn_o/attn_n/subln/lambda fixture from Python gda_post_split_reference",
            benchmarkShape: "B=1, q_origin=32, q_groups=8, S in {1, 32}, channels=256"
        ),
        MotifMetalKernelDescriptor(
            name: .sdpaDualV,
            pythonSymbol: "sdpa_dual_v",
            pythonSource: "src/mlx_motif/kernels/attention.py",
            swiftWrapper: "planned MotifSDPADualV.apply",
            status: .parityPending,
            defaultEnabled: false,
            parityFixture: "q/k/v1/v2 decode fixture from Python sdpa_dual_v_reference",
            benchmarkShape: "B=1, Hq=40, Hkv=8, KV in {256, 1024}, D=128"
        ),
        MotifMetalKernelDescriptor(
            name: .sdpaDualVQ4,
            pythonSymbol: "sdpa_dual_v_q4",
            pythonSource: "src/mlx_motif/kernels/attention.py",
            swiftWrapper: "planned MotifSDPADualVQ4.apply",
            status: .parityPending,
            defaultEnabled: false,
            parityFixture: "packed uint32/scales/biases fixture shared with Python sdpa_dual_v_q4_reference",
            benchmarkShape: "B=1, Hq=40, Hkv=8, KV in {256, 1024}, D=128, bits in {4, 8}"
        ),
    ]

    public static var experimentalMetalRequested: Bool {
        let value = ProcessInfo.processInfo.environment[experimentalOptInEnvironmentVariable] ?? ""
        return ["1", "true", "TRUE", "yes", "YES"].contains(value)
    }

    public static func descriptor(for name: MotifMetalKernelName) -> MotifMetalKernelDescriptor {
        descriptors.first { $0.name == name }!
    }
}

public enum MotifPolynorm {
    private static let metalKernel = MLXFast.metalKernel(
        name: "motif_polynorm",
        inputNames: ["x", "weight", "bias", "eps_in"],
        outputNames: ["y"],
        source: motifPolynormMetalSource
    )

    /// PolyNorm reference path, matching Python `polynorm_reference`:
    ///
    /// `w0 * rms(x^3) + w1 * rms(x^2) + w2 * rms(x) + b`.
    public static func reference(
        _ x: MLXArray,
        weight: MLXArray,
        bias: MLXArray,
        eps: Float = 1e-6
    ) -> MLXArray {
        let x2 = x * x
        let x3 = x2 * x
        return weight[0] * rms(x3, eps: eps)
            + weight[1] * rms(x2, eps: eps)
            + weight[2] * rms(x, eps: eps)
            + bias
    }

    /// Safe entry point for the native port. The reference path is the default;
    /// the Metal wrapper is opt-in so app/runtime code cannot use it before
    /// parity fixtures and benchmark thresholds are added.
    public static func apply(
        _ x: MLXArray,
        weight: MLXArray,
        bias: MLXArray,
        eps: Float = 1e-6,
        executionMode: MotifMetalKernelExecutionMode = .referenceOnly
    ) -> MLXArray {
        guard executionMode == .experimentalMetal, MotifMetalKernels.experimentalMetalRequested else {
            return reference(x, weight: weight, bias: bias, eps: eps)
        }
        return metal(x, weight: weight, bias: bias, eps: eps)
    }

    private static func rms(_ x: MLXArray, eps: Float) -> MLXArray {
        x * rsqrt(mean(x * x, axis: -1, keepDims: true) + eps)
    }

    private static func metal(
        _ x: MLXArray,
        weight: MLXArray,
        bias: MLXArray,
        eps: Float
    ) -> MLXArray {
        let channels = x.shape.last ?? 0
        precondition(channels > 0, "PolyNorm requires a non-empty last dimension")

        let leadShape = Array(x.shape.dropLast())
        let rows = leadShape.reduce(1, *)
        if rows == 0 {
            return zeros(x.shape, dtype: x.dtype)
        }

        let xFlat = x.reshaped([rows, channels])
        let threadGroupSize = min(256, max(32, ((channels + 31) / 32) * 32))
        let epsArray = MLXArray([eps])
        let outputs = metalKernel(
            [xFlat, weight.asType(x.dtype), bias.asType(x.dtype), epsArray],
            template: [
                ("T", x.dtype),
                ("D", channels),
            ],
            grid: (rows * threadGroupSize, 1, 1),
            threadGroup: (threadGroupSize, 1, 1),
            outputShapes: [[rows, channels]],
            outputDTypes: [x.dtype]
        )
        return outputs[0].reshaped(x.shape)
    }
}

private let motifPolynormMetalSource = #"""
    // Each threadgroup handles one row of length D. Ported from
    // src/mlx_motif/kernels/mlp.py; keep source changes in lockstep with the
    // Python kernel until Swift golden fixtures own parity.
    uint row = threadgroup_position_in_grid.x;
    uint tid = thread_position_in_threadgroup.x;
    uint tgsize = threads_per_threadgroup.x;

    const device T* xrow = x + row * D;
    device T*       yrow = y + row * D;

    float s2 = 0.0f, s4 = 0.0f, s6 = 0.0f;
    for (uint i = tid; i < D; i += tgsize) {
        float v  = float(xrow[i]);
        float v2 = v * v;
        float v4 = v2 * v2;
        s2 += v2;
        s4 += v4;
        s6 += v4 * v2;
    }

    threadgroup float tg_s2[32];
    threadgroup float tg_s4[32];
    threadgroup float tg_s6[32];

    float r2 = simd_sum(s2);
    float r4 = simd_sum(s4);
    float r6 = simd_sum(s6);
    uint sg_id = simdgroup_index_in_threadgroup;
    uint lane  = thread_index_in_simdgroup;
    if (lane == 0) {
        tg_s2[sg_id] = r2;
        tg_s4[sg_id] = r4;
        tg_s6[sg_id] = r6;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (sg_id == 0) {
        uint n_sg = simdgroups_per_threadgroup;
        float v2 = (lane < n_sg) ? tg_s2[lane] : 0.0f;
        float v4 = (lane < n_sg) ? tg_s4[lane] : 0.0f;
        float v6 = (lane < n_sg) ? tg_s6[lane] : 0.0f;
        v2 = simd_sum(v2);
        v4 = simd_sum(v4);
        v6 = simd_sum(v6);
        if (lane == 0) {
            tg_s2[0] = v2;
            tg_s4[0] = v4;
            tg_s6[0] = v6;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float inv_d = 1.0f / float(D);
    float eps = float(eps_in[0]);
    float rs2 = metal::rsqrt(tg_s2[0] * inv_d + eps);
    float rs4 = metal::rsqrt(tg_s4[0] * inv_d + eps);
    float rs6 = metal::rsqrt(tg_s6[0] * inv_d + eps);

    float w0 = float(weight[0]);
    float w1 = float(weight[1]);
    float w2 = float(weight[2]);
    float b  = float(bias[0]);

    for (uint i = tid; i < D; i += tgsize) {
        float v  = float(xrow[i]);
        float v2 = v * v;
        float v3 = v2 * v;
        float out = w0 * (v3 * rs6) + w1 * (v2 * rs4) + w2 * (v * rs2) + b;
        yrow[i] = T(out);
    }
"""#
#endif
