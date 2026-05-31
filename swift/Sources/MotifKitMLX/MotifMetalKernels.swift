#if canImport(MLX)
import Dispatch
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
    case gdaPost = "gda_post"
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

public struct MotifMetalKernelParityCase: Sendable, Equatable {
    public let name: String
    public let kernel: MotifMetalKernelName
    public let sourceFixture: String
    public let shape: [Int]
    public let dtype: String
    public let relativeTolerance: Float
    public let absoluteTolerance: Float
    public let requiresRuntime: Bool
    public let requiresExperimentalMetal: Bool

    public init(
        name: String,
        kernel: MotifMetalKernelName,
        sourceFixture: String,
        shape: [Int],
        dtype: String,
        relativeTolerance: Float,
        absoluteTolerance: Float,
        requiresRuntime: Bool = true,
        requiresExperimentalMetal: Bool = false
    ) {
        self.name = name
        self.kernel = kernel
        self.sourceFixture = sourceFixture
        self.shape = shape
        self.dtype = dtype
        self.relativeTolerance = relativeTolerance
        self.absoluteTolerance = absoluteTolerance
        self.requiresRuntime = requiresRuntime
        self.requiresExperimentalMetal = requiresExperimentalMetal
    }
}

public struct MotifMetalKernelBenchmarkCase: Sendable, Equatable {
    public let name: String
    public let kernel: MotifMetalKernelName
    public let shape: [Int]
    public let baseline: String
    public let candidate: String
    public let warmupIterations: Int
    public let iterations: Int
    public let requiresRuntime: Bool
    public let requiresExperimentalMetal: Bool

    public init(
        name: String,
        kernel: MotifMetalKernelName,
        shape: [Int],
        baseline: String,
        candidate: String,
        warmupIterations: Int = 3,
        iterations: Int = 20,
        requiresRuntime: Bool = true,
        requiresExperimentalMetal: Bool = true
    ) {
        self.name = name
        self.kernel = kernel
        self.shape = shape
        self.baseline = baseline
        self.candidate = candidate
        self.warmupIterations = warmupIterations
        self.iterations = iterations
        self.requiresRuntime = requiresRuntime
        self.requiresExperimentalMetal = requiresExperimentalMetal
    }
}

public struct MotifMetalKernelParityResult: Sendable, Equatable {
    public let caseName: String
    public let maxAbsoluteError: Float
    public let maxRelativeError: Float
    public let passed: Bool

    public init(
        caseName: String,
        maxAbsoluteError: Float,
        maxRelativeError: Float,
        passed: Bool
    ) {
        self.caseName = caseName
        self.maxAbsoluteError = maxAbsoluteError
        self.maxRelativeError = maxRelativeError
        self.passed = passed
    }
}

public struct MotifMetalKernelBenchmarkResult: Sendable, Equatable {
    public let caseName: String
    public let iterations: Int
    public let referenceNanoseconds: UInt64
    public let candidateNanoseconds: UInt64

    public init(
        caseName: String,
        iterations: Int,
        referenceNanoseconds: UInt64,
        candidateNanoseconds: UInt64
    ) {
        self.caseName = caseName
        self.iterations = iterations
        self.referenceNanoseconds = referenceNanoseconds
        self.candidateNanoseconds = candidateNanoseconds
    }

    public var candidateSpeedup: Double {
        guard candidateNanoseconds > 0 else { return .infinity }
        return Double(referenceNanoseconds) / Double(candidateNanoseconds)
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
            name: .gdaPost,
            pythonSymbol: "gda_post",
            pythonSource: "src/mlx_motif/kernels/gda.py",
            swiftWrapper: "MotifGDAPost.apply",
            status: .referenceReady,
            defaultEnabled: false,
            parityFixture: "legacy grouped differential attention postprocess fixture",
            benchmarkShape: "B=1, q_origin=32, q_groups=8, S in {1, 32}, channels=256"
        ),
        MotifMetalKernelDescriptor(
            name: .gdaPostSplit,
            pythonSymbol: "gda_post_split",
            pythonSource: "src/mlx_motif/kernels/gda.py",
            swiftWrapper: "MotifGDAPostSplit.apply",
            status: .referenceReady,
            defaultEnabled: false,
            parityFixture: "attn_o/attn_n/subln/lambda fixture from Python gda_post_split_reference",
            benchmarkShape: "B=1, q_origin=32, q_groups=8, S in {1, 32}, channels=256"
        ),
        MotifMetalKernelDescriptor(
            name: .sdpaDualV,
            pythonSymbol: "sdpa_dual_v",
            pythonSource: "src/mlx_motif/kernels/attention.py",
            swiftWrapper: "MotifSDPADualV.apply",
            status: .referenceReady,
            defaultEnabled: false,
            parityFixture: "q/k/v1/v2 decode fixture from Python sdpa_dual_v_reference",
            benchmarkShape: "B=1, Hq=40, Hkv=8, KV in {256, 1024}, D=128"
        ),
        MotifMetalKernelDescriptor(
            name: .sdpaDualVQ4,
            pythonSymbol: "sdpa_dual_v_q4",
            pythonSource: "src/mlx_motif/kernels/attention.py",
            swiftWrapper: "MotifSDPADualVQ4.reference",
            status: .wrapperScaffolded,
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

public enum MotifMetalKernelHarness {
    public static let runtimeOptInEnvironmentVariable = "MOTIFKIT_RUN_MLX_RUNTIME_TESTS"

    public static let parityCases: [MotifMetalKernelParityCase] = [
        MotifMetalKernelParityCase(
            name: "polynorm_reference_2x3",
            kernel: .polynorm,
            sourceFixture: "tests/fixtures/motif_parity_cases.json#component_checks.polynorm",
            shape: [2, 3],
            dtype: "float32",
            relativeTolerance: 1e-6,
            absoluteTolerance: 1e-6,
            requiresExperimentalMetal: false
        ),
        MotifMetalKernelParityCase(
            name: "polynorm_decode_hidden_4096",
            kernel: .polynorm,
            sourceFixture: "deterministic Swift ramp tensor",
            shape: [1, 1, 4_096],
            dtype: "float32",
            relativeTolerance: 1e-5,
            absoluteTolerance: 1e-5,
            requiresExperimentalMetal: true
        ),
    ]

    public static let benchmarkCases: [MotifMetalKernelBenchmarkCase] = [
        MotifMetalKernelBenchmarkCase(
            name: "polynorm_decode_hidden_4096",
            kernel: .polynorm,
            shape: [1, 1, 4_096],
            baseline: "MotifPolynorm.reference",
            candidate: "MotifPolynorm.apply(.experimentalMetal)"
        ),
        MotifMetalKernelBenchmarkCase(
            name: "polynorm_prefill_128x4096",
            kernel: .polynorm,
            shape: [1, 128, 4_096],
            baseline: "MotifPolynorm.reference",
            candidate: "MotifPolynorm.apply(.experimentalMetal)",
            warmupIterations: 2,
            iterations: 10
        ),
    ]

    public static func parityCases(for kernel: MotifMetalKernelName) -> [MotifMetalKernelParityCase] {
        parityCases.filter { $0.kernel == kernel }
    }

    public static func benchmarkCases(for kernel: MotifMetalKernelName) -> [MotifMetalKernelBenchmarkCase] {
        benchmarkCases.filter { $0.kernel == kernel }
    }

    public static func deterministicInput(shape: [Int]) -> MLXArray {
        let count = shape.reduce(1, *)
        let values = (0..<count).map { index in
            Float((index % 17) - 8) / 4.0
        }
        return MLXArray(values, shape)
    }

    public static func checkPolynormParity(
        case parityCase: MotifMetalKernelParityCase,
        x: MLXArray,
        weight: MLXArray,
        bias: MLXArray,
        eps: Float = 1e-6,
        executionMode: MotifMetalKernelExecutionMode
    ) -> MotifMetalKernelParityResult {
        let reference = MotifPolynorm.reference(x, weight: weight, bias: bias, eps: eps)
        let candidate = MotifPolynorm.apply(
            x,
            weight: weight,
            bias: bias,
            eps: eps,
            executionMode: executionMode
        )
        eval(reference)
        eval(candidate)

        let referenceValues = reference.asArray(Float.self)
        let candidateValues = candidate.asArray(Float.self)
        precondition(referenceValues.count == candidateValues.count, "PolyNorm parity arrays must have equal length")

        var maxAbsoluteError: Float = 0
        var maxRelativeError: Float = 0
        for (referenceValue, candidateValue) in zip(referenceValues, candidateValues) {
            let absoluteError = abs(referenceValue - candidateValue)
            let denominator = max(abs(referenceValue), Float.leastNonzeroMagnitude)
            let relativeError = absoluteError / denominator
            maxAbsoluteError = max(maxAbsoluteError, absoluteError)
            maxRelativeError = max(maxRelativeError, relativeError)
        }

        return MotifMetalKernelParityResult(
            caseName: parityCase.name,
            maxAbsoluteError: maxAbsoluteError,
            maxRelativeError: maxRelativeError,
            passed: maxAbsoluteError <= parityCase.absoluteTolerance
                || maxRelativeError <= parityCase.relativeTolerance
        )
    }

    public static func benchmarkPolynorm(
        case benchmarkCase: MotifMetalKernelBenchmarkCase,
        weight: MLXArray = MLXArray([Float(0.4), 0.3, 0.3], [3]),
        bias: MLXArray = MLXArray([Float(0.05)], [1]),
        eps: Float = 1e-6,
        executionMode: MotifMetalKernelExecutionMode = .experimentalMetal
    ) -> MotifMetalKernelBenchmarkResult {
        precondition(benchmarkCase.kernel == .polynorm, "Only PolyNorm benchmark cases are implemented")
        precondition(benchmarkCase.iterations > 0, "Benchmark iterations must be positive")

        let x = deterministicInput(shape: benchmarkCase.shape)
        for _ in 0..<benchmarkCase.warmupIterations {
            eval(MotifPolynorm.reference(x, weight: weight, bias: bias, eps: eps))
            eval(MotifPolynorm.apply(x, weight: weight, bias: bias, eps: eps, executionMode: executionMode))
        }

        let referenceStart = DispatchTime.now().uptimeNanoseconds
        for _ in 0..<benchmarkCase.iterations {
            eval(MotifPolynorm.reference(x, weight: weight, bias: bias, eps: eps))
        }
        let referenceEnd = DispatchTime.now().uptimeNanoseconds

        let candidateStart = DispatchTime.now().uptimeNanoseconds
        for _ in 0..<benchmarkCase.iterations {
            eval(MotifPolynorm.apply(x, weight: weight, bias: bias, eps: eps, executionMode: executionMode))
        }
        let candidateEnd = DispatchTime.now().uptimeNanoseconds

        return MotifMetalKernelBenchmarkResult(
            caseName: benchmarkCase.name,
            iterations: benchmarkCase.iterations,
            referenceNanoseconds: referenceEnd - referenceStart,
            candidateNanoseconds: candidateEnd - candidateStart
        )
    }
}

public enum MotifSDPADualV {
    public static func reference(
        queries: MLXArray,
        keys: MLXArray,
        value1: MLXArray,
        value2: MLXArray,
        scale: Float,
        mask: MLXFast.ScaledDotProductAttentionMaskMode = .none
    ) -> MLXArray {
        var keys = keys
        var value1 = value1
        var value2 = value2
        let queryHeads = queries.dim(1)
        let keyHeads = keys.dim(1)
        if keyHeads > 0, queryHeads % keyHeads == 0, queryHeads != keyHeads {
            let repeatCount = queryHeads / keyHeads
            keys = repeated(keys, count: repeatCount, axis: 1)
            value1 = repeated(value1, count: repeatCount, axis: 1)
            value2 = repeated(value2, count: repeatCount, axis: 1)
        }
        let out1 = MLXFast.scaledDotProductAttention(
            queries: queries,
            keys: keys,
            values: value1,
            scale: scale,
            mask: mask
        )
        let out2 = MLXFast.scaledDotProductAttention(
            queries: queries,
            keys: keys,
            values: value2,
            scale: scale,
            mask: mask
        )
        return concatenated([out1, out2], axis: -1)
    }

    public static func apply(
        queries: MLXArray,
        keys: MLXArray,
        value1: MLXArray,
        value2: MLXArray,
        scale: Float,
        mask: MLXFast.ScaledDotProductAttentionMaskMode = .none,
        executionMode _: MotifMetalKernelExecutionMode = .referenceOnly
    ) -> MLXArray {
        // The Python Metal kernel is decode-only and disabled in Swift until
        // golden fixtures prove numerical parity. This callable wrapper gives
        // the model/server/bench paths the same routing surface today.
        reference(queries: queries, keys: keys, value1: value1, value2: value2, scale: scale, mask: mask)
    }
}

public enum MotifSDPADualVQ4 {
    public static func reference(
        queries: MLXArray,
        quantizedKeys: MotifQuantizedTuple,
        quantizedValue1: MotifQuantizedTuple,
        quantizedValue2: MotifQuantizedTuple,
        scale: Float,
        groupSize: Int,
        bits: Int,
        mode: QuantizationMode = .affine,
        dtype: DType? = nil
    ) -> MLXArray {
        let keys = dequantized(
            quantizedKeys.data,
            scales: quantizedKeys.scales,
            biases: quantizedKeys.biases,
            groupSize: groupSize,
            bits: bits,
            mode: mode,
            dtype: dtype
        )
        let value1 = dequantized(
            quantizedValue1.data,
            scales: quantizedValue1.scales,
            biases: quantizedValue1.biases,
            groupSize: groupSize,
            bits: bits,
            mode: mode,
            dtype: dtype
        )
        let value2 = dequantized(
            quantizedValue2.data,
            scales: quantizedValue2.scales,
            biases: quantizedValue2.biases,
            groupSize: groupSize,
            bits: bits,
            mode: mode,
            dtype: dtype
        )
        return MotifSDPADualV.reference(queries: queries, keys: keys, value1: value1, value2: value2, scale: scale)
    }
}

public enum MotifGDAPostSplit {
    public static func reference(
        attnOrigin: MLXArray,
        attnNoise: MLXArray,
        sublnWeight: MLXArray,
        lambda: MLXArray,
        lambdaInit: Double,
        groupedRatio: Int,
        eps: Float = 1e-5
    ) -> MLXArray {
        let noise = groupedRatio == 1 ? attnNoise : repeated(attnNoise, count: groupedRatio, axis: 1)
        let differential = attnOrigin - lambda.asType(attnOrigin.dtype) * noise
        let normalized = MLXFast.rmsNorm(
            differential,
            weight: sublnWeight.asType(differential.dtype),
            eps: eps
        )
        return normalized * Float(1.0 - lambdaInit)
    }

    public static func apply(
        attnOrigin: MLXArray,
        attnNoise: MLXArray,
        sublnWeight: MLXArray,
        lambda: MLXArray,
        lambdaInit: Double,
        groupedRatio: Int,
        eps: Float = 1e-5,
        executionMode _: MotifMetalKernelExecutionMode = .referenceOnly
    ) -> MLXArray {
        reference(
            attnOrigin: attnOrigin,
            attnNoise: attnNoise,
            sublnWeight: sublnWeight,
            lambda: lambda,
            lambdaInit: lambdaInit,
            groupedRatio: groupedRatio,
            eps: eps
        )
    }
}

public enum MotifGDAPost {
    public static func reference(
        merged: MLXArray,
        sublnWeight: MLXArray,
        lambda: MLXArray,
        lambdaInit: Double,
        queryGroups: Int,
        groupedRatio: Int,
        eps: Float = 1e-5
    ) -> MLXArray {
        let qOrigin = queryGroups * groupedRatio
        let pieces = merged.split(indices: [qOrigin], axis: 1)
        return MotifGDAPostSplit.reference(
            attnOrigin: pieces[0],
            attnNoise: pieces[1],
            sublnWeight: sublnWeight,
            lambda: lambda,
            lambdaInit: lambdaInit,
            groupedRatio: groupedRatio,
            eps: eps
        )
    }

    public static func apply(
        merged: MLXArray,
        sublnWeight: MLXArray,
        lambda: MLXArray,
        lambdaInit: Double,
        queryGroups: Int,
        groupedRatio: Int,
        eps: Float = 1e-5,
        executionMode _: MotifMetalKernelExecutionMode = .referenceOnly
    ) -> MLXArray {
        reference(
            merged: merged,
            sublnWeight: sublnWeight,
            lambda: lambda,
            lambdaInit: lambdaInit,
            queryGroups: queryGroups,
            groupedRatio: groupedRatio,
            eps: eps
        )
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
