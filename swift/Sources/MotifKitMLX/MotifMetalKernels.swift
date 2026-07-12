#if canImport(MLX)
import Dispatch
import Foundation
import MLX

public enum MotifMetalKernelExecutionMode: Sendable {
    /// Keep the native port on pure MLX Swift reference ops. This remains the
    /// debugging/escape-hatch path and is selected by `MLX_MOTIF_DISABLE_KERNELS=1`.
    case referenceOnly

    /// Prefer the direct Swift custom Metal kernel and fall back only for shapes
    /// outside a kernel's decode contract. This is the production default.
    case metalPreferred

    /// Backwards-compatible spelling for older parity harnesses. It now has the
    /// same behavior as `.metalPreferred`; the old env opt-in is intentionally no
    /// longer required because this PR makes custom kernels default-on.
    case experimentalMetal
}

public enum MotifMetalKernelStatus: String, Sendable {
    case referenceReady
    case wrapperScaffolded
    case metalReady
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
        requiresExperimentalMetal: Bool = false
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

/// Thread-safe, process-wide counter that records how many times a custom Metal
/// kernel wrapper fell back to its pure-MLX reference implementation.
///
/// The kernels intentionally fall back when an input is outside their decode
/// contract (wrong rank, non-affine quant, mismatched head packing, …). That
/// fallback is silent by design — it keeps numerics correct — but it also hides
/// regressions where the direct Metal path stops being selected. This telemetry
/// makes the fallback observable without changing any kernel behavior:
///
/// - Counts are bucketed by `MotifMetalKernelName` and are purely additive.
/// - Reads/writes are guarded by an `NSLock`, so concurrent kernel calls on
///   different threads cannot race. The lock is held only around small integer
///   mutations, so it does not change the async-ness of any call site.
/// - Set `MLX_MOTIF_LOG_FALLBACKS=1` to emit a one-line diagnostic to stderr on
///   each fallback. With the variable unset there is no output at all.
public enum MotifKernelFallbackTelemetry {
    public static let logEnvironmentVariable = "MLX_MOTIF_LOG_FALLBACKS"

    private static let lock = NSLock()
    nonisolated(unsafe) private static var counts: [MotifMetalKernelName: Int] = [:]

    /// Records a single fallback for `kernel` and, when logging is enabled,
    /// writes a diagnostic line to standard error.
    public static func recordFallback(_ kernel: MotifMetalKernelName) {
        let updated: Int = {
            lock.lock()
            defer { lock.unlock() }
            let next = (counts[kernel] ?? 0) + 1
            counts[kernel] = next
            return next
        }()
        logIfEnabled(kernel: kernel, total: updated)
    }

    /// Current fallback count for a single kernel.
    public static func fallbackCount(for kernel: MotifMetalKernelName) -> Int {
        lock.lock()
        defer { lock.unlock() }
        return counts[kernel] ?? 0
    }

    /// Snapshot of every recorded kernel's fallback count.
    public static func snapshot() -> [MotifMetalKernelName: Int] {
        lock.lock()
        defer { lock.unlock() }
        return counts
    }

    /// Clears all counters. Intended for test isolation.
    public static func reset() {
        lock.lock()
        defer { lock.unlock() }
        counts.removeAll(keepingCapacity: true)
    }

    private static var loggingEnabled: Bool {
        let value = ProcessInfo.processInfo.environment[logEnvironmentVariable]?
            .trimmingCharacters(in: .whitespacesAndNewlines)
            .lowercased()
        guard let value else { return false }
        return ["1", "true", "yes", "on"].contains(value)
    }

    private static func logIfEnabled(kernel: MotifMetalKernelName, total: Int) {
        guard loggingEnabled else { return }
        let line = "[mlx-motif] kernel fallback: \(kernel.rawValue) (total=\(total))\n"
        if let data = line.data(using: .utf8) {
            FileHandle.standardError.write(data)
        }
    }
}

/// Process-wide cache for the tiny constant scalar inputs (attention scale,
/// RMS eps, `1 - lambda_init`) that the kernel wrappers previously rebuilt as a
/// fresh 1-element `MLXArray` on every launch — several allocations plus graph
/// nodes per layer per decoded token. The values are per-model constants, so
/// the map stays a handful of entries deep for the process lifetime.
enum MotifKernelScalarCache {
    private static let lock = NSLock()
    nonisolated(unsafe) private static var floatScalars: [Float: MLXArray] = [:]

    /// Returns a shared `[1]`-shaped float32 array holding `value`.
    static func scalar(_ value: Float) -> MLXArray {
        lock.lock()
        defer { lock.unlock() }
        if let cached = floatScalars[value] {
            return cached
        }
        let array = MLXArray([value])
        floatScalars[value] = array
        return array
    }
}

public enum MotifMetalKernels {
    public static let disableEnvironmentVariable = "MLX_MOTIF_DISABLE_KERNELS"
    public static let legacyDisableEnvironmentVariable = "MOTIFKIT_DISABLE_CUSTOM_METAL_KERNELS"
    public static let experimentalOptInEnvironmentVariable =
        "MOTIFKIT_ENABLE_EXPERIMENTAL_METAL_KERNELS"

    /// Swift-side manifest that mirrors `src/mlx_motif/kernels`. All entries
    /// are enabled by default in the hard-parity branch; callers can force the
    /// reference path with `MLX_MOTIF_DISABLE_KERNELS=1`.
    public static let descriptors: [MotifMetalKernelDescriptor] = [
        MotifMetalKernelDescriptor(
            name: .polynorm,
            pythonSymbol: "polynorm",
            pythonSource: "src/mlx_motif/kernels/mlp.py",
            swiftWrapper: "MotifPolynorm.apply",
            status: .metalReady,
            defaultEnabled: true,
            parityFixture: "polynorm small-row + decode hidden-size golden tensors",
            benchmarkShape: "(..., D) with D in {128, 4096}"
        ),
        MotifMetalKernelDescriptor(
            name: .gdaPost,
            pythonSymbol: "gda_post",
            pythonSource: "src/mlx_motif/kernels/gda.py",
            swiftWrapper: "MotifGDAPost.apply",
            status: .metalReady,
            defaultEnabled: true,
            parityFixture: "legacy grouped differential attention postprocess fixture",
            benchmarkShape: "B=1, q_origin=32, q_groups=8, S in {1, 32}, channels=256"
        ),
        MotifMetalKernelDescriptor(
            name: .gdaPostSplit,
            pythonSymbol: "gda_post_split",
            pythonSource: "src/mlx_motif/kernels/gda.py",
            swiftWrapper: "MotifGDAPostSplit.apply",
            status: .metalReady,
            defaultEnabled: true,
            parityFixture: "attn_o/attn_n/subln/lambda fixture from Python gda_post_split_reference",
            benchmarkShape: "B=1, q_origin=32, q_groups=8, S in {1, 32}, channels=256"
        ),
        MotifMetalKernelDescriptor(
            name: .sdpaDualV,
            pythonSymbol: "sdpa_dual_v",
            pythonSource: "src/mlx_motif/kernels/attention.py",
            swiftWrapper: "MotifSDPADualV.apply",
            status: .metalReady,
            defaultEnabled: true,
            parityFixture: "q/k/v1/v2 decode fixture from Python sdpa_dual_v_reference",
            benchmarkShape: "B=1, Hq=40, Hkv=8, KV in {256, 1024}, D=128"
        ),
        MotifMetalKernelDescriptor(
            name: .sdpaDualVQ4,
            pythonSymbol: "sdpa_dual_v_q4",
            pythonSource: "src/mlx_motif/kernels/attention.py",
            swiftWrapper: "MotifSDPADualVQ4.apply",
            status: .metalReady,
            defaultEnabled: true,
            parityFixture: "packed uint32/scales/biases fixture shared with Python sdpa_dual_v_q4_reference",
            benchmarkShape: "B=1, Hq=40, Hkv=8, KV in {256, 1024}, D=128, bits in {4, 8}"
        ),
    ]

    public static var customMetalDisabled: Bool {
        isTruthy(ProcessInfo.processInfo.environment[disableEnvironmentVariable])
            || isTruthy(ProcessInfo.processInfo.environment[legacyDisableEnvironmentVariable])
    }

    public static var experimentalMetalRequested: Bool {
        let value = ProcessInfo.processInfo.environment[experimentalOptInEnvironmentVariable] ?? ""
        return ["1", "true", "TRUE", "yes", "YES"].contains(value)
    }

    public static func shouldUseMetal(executionMode: MotifMetalKernelExecutionMode) -> Bool {
        switch executionMode {
        case .referenceOnly:
            return false
        case .metalPreferred, .experimentalMetal:
            return !customMetalDisabled
        }
    }

    public static func descriptor(for name: MotifMetalKernelName) -> MotifMetalKernelDescriptor {
        descriptors.first { $0.name == name }!
    }

    private static func isTruthy(_ value: String?) -> Bool {
        guard let value = value?.trimmingCharacters(in: .whitespacesAndNewlines).lowercased() else {
            return false
        }
        return ["1", "true", "yes", "on"].contains(value)
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
            requiresExperimentalMetal: false
        ),
    ]

    public static let benchmarkCases: [MotifMetalKernelBenchmarkCase] = [
        MotifMetalKernelBenchmarkCase(
            name: "polynorm_decode_hidden_4096",
            kernel: .polynorm,
            shape: [1, 1, 4_096],
            baseline: "MotifPolynorm.reference",
            candidate: "MotifPolynorm.apply(.metalPreferred)"
        ),
        MotifMetalKernelBenchmarkCase(
            name: "polynorm_prefill_128x4096",
            kernel: .polynorm,
            shape: [1, 128, 4_096],
            baseline: "MotifPolynorm.reference",
            candidate: "MotifPolynorm.apply(.metalPreferred)",
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
        executionMode: MotifMetalKernelExecutionMode = .metalPreferred
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
    private static let metalKernel = MLXFast.metalKernel(
        name: "motif_sdpa_dual_v",
        inputNames: ["q", "k", "v1", "v2", "scale_in", "kv_len_in"],
        outputNames: ["y"],
        source: motifSDPADualVMetalSource
    )

    public static func reference(
        queries: MLXArray,
        keys: MLXArray,
        value1: MLXArray,
        value2: MLXArray,
        scale: Float,
        mask: MLXFast.ScaledDotProductAttentionMaskMode = .none
    ) -> MLXArray {
        // MLXFast SDPA broadcasts GQA natively (query head i reads kv head
        // i / (H_q / H_kv) — exactly repeat-interleave semantics), so
        // GQA-shaped keys/values are passed straight through instead of
        // materializing repeated full-sequence copies. This path runs on
        // every prefill, where the repeat used to cost three full-sequence
        // tensor materializations per layer.
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
        kvLen: Int? = nil,
        executionMode: MotifMetalKernelExecutionMode = .metalPreferred
    ) -> MLXArray {
        // `kvLen` is the number of live KV rows to attend over. It defaults to
        // the buffer length (`keys.dim(2)`); callers pass full step-padded
        // capacity buffers together with a smaller live length so the kernel
        // avoids a per-step `ensure_row_contiguous` copy of the live KV region
        // (see MotifGroupedKVCache.updateAndFetch4). Row strides come from each
        // buffer's own shape, so padded rows past `liveLength` are never read.
        // Mirrors the Python `sdpa_dual_v(..., kv_len=...)` wrapper.
        let liveLength = kvLen ?? keys.dim(2)
        guard MotifMetalKernels.shouldUseMetal(executionMode: executionMode),
              maskIsDecodeNoOp(mask),
              queries.dim(2) == 1,
              queries.dim(3) % 32 == 0,
              keys.dim(1) > 0,
              queries.dim(1) % keys.dim(1) == 0,
              liveLength >= 0,
              liveLength <= keys.dim(2),
              // Full K/V contract, mirroring the Python wrapper's asserts
              // (`v1.shape == v2.shape == (B, H_kv, KV, D)`, `k.shape[-1] == D`):
              // the kernel indexes K and both V slabs with strides derived from
              // queries/keys, so a mismatched value tensor would read
              // out-of-bounds device memory instead of falling back.
              keys.dim(0) == queries.dim(0),
              keys.dim(3) == queries.dim(3),
              value1.shape == keys.shape,
              value2.shape == keys.shape
        else {
            MotifKernelFallbackTelemetry.recordFallback(.sdpaDualV)
            // The reference (MLXFast SDPA) path has no runtime length bound, so
            // slice the buffers to the live region before the fallback attends
            // over them. Mirrors the Python reference wrapper, which slices
            // `k/v1/v2[..., :kv_len, :]` before falling back.
            return reference(
                queries: queries,
                keys: boundedRows(keys, to: liveLength),
                value1: boundedRows(value1, to: liveLength),
                value2: boundedRows(value2, to: liveLength),
                scale: scale,
                mask: mask
            )
        }
        return metal(queries: queries, keys: keys, value1: value1, value2: value2, scale: scale, kvLen: liveLength)
    }

    /// Slice a `(B, H, KV_cap, D)` buffer to its first `length` rows along the
    /// sequence axis. Returns the buffer unchanged when it is already exactly
    /// `length` long, so the exact-length caller path never builds a view.
    private static func boundedRows(_ buffer: MLXArray, to length: Int) -> MLXArray {
        guard length >= 0, length < buffer.dim(2) else { return buffer }
        return buffer[.ellipsis, ..<length, 0...]
    }

    /// The Metal kernel computes an unmasked softmax over every KV position.
    /// That is only valid when the supplied mask is a no-op for single-token
    /// decode: `.none`, or `.causal` with S_q == 1 (the sole query row is the
    /// last position and may attend to all cached keys). Explicit `.array` /
    /// `.arrays` masks must take the mask-honoring reference path instead of
    /// being silently dropped.
    private static func maskIsDecodeNoOp(_ mask: MLXFast.ScaledDotProductAttentionMaskMode) -> Bool {
        switch mask {
        case .none, .causal:
            return true
        default:
            return false
        }
    }

    private static func metal(
        queries: MLXArray,
        keys: MLXArray,
        value1: MLXArray,
        value2: MLXArray,
        scale: Float,
        kvLen: Int
    ) -> MLXArray {
        let batch = queries.dim(0)
        let queryHeads = queries.dim(1)
        let keyHeads = keys.dim(1)
        let kvSequenceLength = keys.dim(2)
        let headDim = queries.dim(3)
        guard keyHeads > 0, kvSequenceLength > 0 else {
            MotifKernelFallbackTelemetry.recordFallback(.sdpaDualV)
            return reference(queries: queries, keys: keys, value1: value1, value2: value2, scale: scale)
        }
        let gqaFactor = queryHeads / keyHeads
        let rows = batch * queryHeads
        let threads = 32 * 32
        let scaleArray = MotifKernelScalarCache.scalar(scale)
        // KV length is a runtime input rather than a template constant so a
        // growing decode context does not force a fresh Metal JIT compile
        // (one new pipeline per token) — the template set stays finite. The
        // kernel indexes K/V rows by each buffer's own capacity stride (from
        // its shape) and bounds the softmax loop by this live `kvLen`, so
        // callers may pass step-padded capacity buffers with kvLen < capacity.
        let kvLengthArray = MLXArray([Int32(kvLen)])
        let outputs = metalKernel(
            [queries, keys, value1, value2, scaleArray, kvLengthArray],
            template: [
                ("T", queries.dtype),
                ("D", headDim),
                ("GQA_FACTOR", gqaFactor),
            ],
            grid: (rows * threads, 1, 1),
            threadGroup: (threads, 1, 1),
            outputShapes: [[batch, queryHeads, 1, 2 * headDim]],
            outputDTypes: [queries.dtype]
        )
        return outputs[0]
    }
}

public enum MotifSDPADualVQ4 {
    /// Returns whether the scalar head-dimension contract required by the
    /// packed decode kernel is safe to evaluate. Keep the positivity check
    /// before every modulo so malformed group sizes route to the reference
    /// path instead of trapping in the wrapper guard.
    static func supportsPackedGeometry(headDim: Int, groupSize: Int, bits: Int) -> Bool {
        guard groupSize > 0,
              bits == 4 || bits == 8,
              headDim >= 32,
              headDim % 32 == 0,
              headDim % groupSize == 0
        else {
            return false
        }
        let qkPerThread = headDim / 32
        return (32 / bits) % qkPerThread == 0
            && groupSize % qkPerThread == 0
    }

    private static let metalKernel = MLXFast.metalKernel(
        name: "motif_sdpa_dual_v_q4",
        inputNames: [
            "q",
            "k_data",
            "k_scales",
            "k_biases",
            "v1_data",
            "v1_scales",
            "v1_biases",
            "v2_data",
            "v2_scales",
            "v2_biases",
            "scale_in",
            "kv_len_in",
        ],
        outputNames: ["y"],
        source: motifSDPADualVQ4MetalSource
    )

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

    public static func apply(
        queries: MLXArray,
        quantizedKeys: MotifQuantizedTuple,
        quantizedValue1: MotifQuantizedTuple,
        quantizedValue2: MotifQuantizedTuple,
        scale: Float,
        groupSize: Int,
        bits: Int,
        mode: QuantizationMode = .affine,
        dtype: DType? = nil,
        kvLen: Int? = nil,
        executionMode: MotifMetalKernelExecutionMode = .metalPreferred
    ) -> MLXArray {
        // Live KV rows to attend over (defaults to the packed buffer length).
        // Callers pass full step-padded capacity triples plus a smaller live
        // length so the kernel skips a per-step `ensure_row_contiguous` copy of
        // all nine live-region buffers. Mirrors Python `sdpa_dual_v_q4(kv_len=)`.
        let liveLength = kvLen ?? quantizedKeys.data.dim(2)
        guard MotifMetalKernels.shouldUseMetal(executionMode: executionMode),
              mode == .affine,
              queries.dim(2) == 1,
              supportsPackedGeometry(
                  headDim: queries.dim(3), groupSize: groupSize, bits: bits
              ),
              // Live-region bounds (from the full-capacity cache-fetch path):
              // callers pass step-padded capacity triples plus a smaller live
              // length, so bound it before the kernel indexes the packed buffers.
              liveLength >= 0,
              liveLength <= quantizedKeys.data.dim(2),
              // Packed-SDPA lane -> word -> group contract (SUFFICIENT, not just
              // necessary). Each lane owns a CONTIGUOUS block of qk_per_thread =
              // D/32 channels starting at base = sg_lid*qk_per_thread, and per
              // tensor per step loads exactly ONE packed uint32 (EL_PER_INT =
              // 32/bits channels/word) and ONE scale/bias (GROUP_SIZE channels/
              // group). The block is read correctly only if it stays inside a
              // single word AND a single quant group. Contiguous length-L blocks
              // at multiples of L fit inside every aligned size-W window iff L|W,
              // so the sufficient condition is qk_per_thread | EL_PER_INT (single
              // word; implies the old qk_per_thread <= EL_PER_INT) AND
              // qk_per_thread | GROUP_SIZE (single scale/bias group). The old
              // `<=` guard is only necessary: e.g. D=96/bits=8 gives
              // qk_per_thread=3, EL_PER_INT=4 (3<=4 passes) but lane 1's block
              // {3,4,5} straddles words 0/1 — channels 4,5 read the never-loaded
              // next word with shifts 32/40 (>=32, UB in Metal). Route every
              // out-of-contract shape to reference() instead. Mirrors the Python
              // asserts in sdpa_dual_v_q4.
              let keyBiases = quantizedKeys.biases,
              let value1Biases = quantizedValue1.biases,
              let value2Biases = quantizedValue2.biases,
              quantizedKeys.data.dim(1) > 0,
              queries.dim(1) % quantizedKeys.data.dim(1) == 0,
              // Full packed K/V contract, mirroring the fp16 wrapper and the
              // Python asserts: the kernel indexes all three quantized slabs
              // with strides derived from the K buffers, so mismatched value
              // shapes would read out-of-bounds memory instead of falling back.
              quantizedKeys.data.dim(0) == queries.dim(0),
              quantizedValue1.data.shape == quantizedKeys.data.shape,
              quantizedValue2.data.shape == quantizedKeys.data.shape,
              quantizedValue1.scales.shape == quantizedKeys.scales.shape,
              quantizedValue2.scales.shape == quantizedKeys.scales.shape,
              keyBiases.shape == quantizedKeys.scales.shape,
              value1Biases.shape == quantizedKeys.scales.shape,
              value2Biases.shape == quantizedKeys.scales.shape,
              quantizedKeys.scales.dim(0) == quantizedKeys.data.dim(0),
              quantizedKeys.scales.dim(1) == quantizedKeys.data.dim(1),
              quantizedKeys.scales.dim(2) == quantizedKeys.data.dim(2)
        else {
            MotifKernelFallbackTelemetry.recordFallback(.sdpaDualVQ4)
            // The dequant + MLXFast SDPA reference path has no runtime length
            // bound, so slice each packed triple to the live region first.
            // Mirrors the Python reference wrapper's `[..., :kv_len, :]` slice.
            return reference(
                queries: queries,
                quantizedKeys: boundedTriple(quantizedKeys, to: liveLength),
                quantizedValue1: boundedTriple(quantizedValue1, to: liveLength),
                quantizedValue2: boundedTriple(quantizedValue2, to: liveLength),
                scale: scale,
                groupSize: groupSize,
                bits: bits,
                mode: mode,
                dtype: dtype
            )
        }
        return metal(
            queries: queries,
            quantizedKeys: (quantizedKeys.data, quantizedKeys.scales, keyBiases),
            quantizedValue1: (quantizedValue1.data, quantizedValue1.scales, value1Biases),
            quantizedValue2: (quantizedValue2.data, quantizedValue2.scales, value2Biases),
            scale: scale,
            groupSize: groupSize,
            bits: bits,
            kvLen: liveLength
        )
    }

    /// Slice a packed `(data, scales, biases)` triple to its first `length`
    /// sequence rows. Returns the triple unchanged when it is already exactly
    /// `length` long so the exact-length caller path never builds a view.
    private static func boundedTriple(_ triple: MotifQuantizedTuple, to length: Int) -> MotifQuantizedTuple {
        guard length >= 0, length < triple.data.dim(2) else { return triple }
        return (
            triple.data[.ellipsis, ..<length, 0...],
            triple.scales[.ellipsis, ..<length, 0...],
            triple.biases.map { $0[.ellipsis, ..<length, 0...] }
        )
    }

    private static func metal(
        queries: MLXArray,
        quantizedKeys: (data: MLXArray, scales: MLXArray, biases: MLXArray),
        quantizedValue1: (data: MLXArray, scales: MLXArray, biases: MLXArray),
        quantizedValue2: (data: MLXArray, scales: MLXArray, biases: MLXArray),
        scale: Float,
        groupSize: Int,
        bits: Int,
        kvLen: Int
    ) -> MLXArray {
        let batch = queries.dim(0)
        let queryHeads = queries.dim(1)
        let keyHeads = quantizedKeys.data.dim(1)
        let kvSequenceLength = quantizedKeys.data.dim(2)
        let headDim = queries.dim(3)
        let elementsPerInt = 32 / bits
        guard keyHeads > 0, kvSequenceLength > 0, quantizedKeys.data.dim(3) * elementsPerInt == headDim else {
            MotifKernelFallbackTelemetry.recordFallback(.sdpaDualVQ4)
            return reference(
                queries: queries,
                quantizedKeys: (quantizedKeys.data, quantizedKeys.scales, quantizedKeys.biases),
                quantizedValue1: (quantizedValue1.data, quantizedValue1.scales, quantizedValue1.biases),
                quantizedValue2: (quantizedValue2.data, quantizedValue2.scales, quantizedValue2.biases),
                scale: scale,
                groupSize: groupSize,
                bits: bits
            )
        }
        let gqaFactor = queryHeads / keyHeads
        let rows = batch * queryHeads
        let threads = 32 * 32
        let scaleArray = MotifKernelScalarCache.scalar(scale)
        // Runtime KV length (see MotifSDPADualV.metal): the kernel indexes rows
        // by the packed K buffer's own capacity stride and bounds the softmax
        // loop by this live `kvLen`, so callers may pass step-padded capacity
        // triples with kvLen < capacity. Keeps the compiled pipeline set finite.
        let kvLengthArray = MLXArray([Int32(kvLen)])
        let outputs = metalKernel(
            [
                queries,
                quantizedKeys.data,
                quantizedKeys.scales,
                quantizedKeys.biases,
                quantizedValue1.data,
                quantizedValue1.scales,
                quantizedValue1.biases,
                quantizedValue2.data,
                quantizedValue2.scales,
                quantizedValue2.biases,
                scaleArray,
                kvLengthArray,
            ],
            template: [
                ("T", queries.dtype),
                ("D", headDim),
                ("GQA_FACTOR", gqaFactor),
                ("BITS", bits),
                ("GROUP_SIZE", groupSize),
            ],
            grid: (rows * threads, 1, 1),
            threadGroup: (threads, 1, 1),
            outputShapes: [[batch, queryHeads, 1, 2 * headDim]],
            outputDTypes: [queries.dtype]
        )
        return outputs[0]
    }
}

public enum MotifGDAPostSplit {
    private static let metalKernel = MLXFast.metalKernel(
        name: "motif_gda_post_split",
        inputNames: ["attn_o", "attn_n", "subln_w", "lambda_full", "scale_in", "eps_in"],
        outputNames: ["y"],
        source: motifGDAPostSplitMetalSource
    )

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
        executionMode: MotifMetalKernelExecutionMode = .metalPreferred
    ) -> MLXArray {
        guard MotifMetalKernels.shouldUseMetal(executionMode: executionMode),
              groupedRatio > 0,
              attnOrigin.dim(1) == attnNoise.dim(1) * groupedRatio
        else {
            MotifKernelFallbackTelemetry.recordFallback(.gdaPostSplit)
            return reference(
                attnOrigin: attnOrigin,
                attnNoise: attnNoise,
                sublnWeight: sublnWeight,
                lambda: lambda,
                lambdaInit: lambdaInit,
                groupedRatio: groupedRatio,
                eps: eps
            )
        }
        return metal(
            attnOrigin: attnOrigin,
            attnNoise: attnNoise,
            sublnWeight: sublnWeight,
            lambda: lambda,
            lambdaInit: lambdaInit,
            groupedRatio: groupedRatio,
            eps: eps
        )
    }

    private static func metal(
        attnOrigin: MLXArray,
        attnNoise: MLXArray,
        sublnWeight: MLXArray,
        lambda: MLXArray,
        lambdaInit: Double,
        groupedRatio: Int,
        eps: Float
    ) -> MLXArray {
        let batch = attnOrigin.dim(0)
        let queryOrigin = attnOrigin.dim(1)
        let sequenceLength = attnOrigin.dim(2)
        let channels = attnOrigin.dim(3)
        let queryGroups = attnNoise.dim(1)
        let rows = batch * queryOrigin * sequenceLength
        let threads = min(256, max(32, ((channels + 31) / 32) * 32))
        let scaleArray = MotifKernelScalarCache.scalar(Float(1.0 - lambdaInit))
        let epsArray = MotifKernelScalarCache.scalar(eps)
        let outputs = metalKernel(
            [
                attnOrigin,
                attnNoise,
                sublnWeight.asType(attnOrigin.dtype),
                lambda.asType(.float32),
                scaleArray,
                epsArray,
            ],
            template: [
                ("T", attnOrigin.dtype),
                ("Q_ORIGIN", queryOrigin),
                ("Q_GROUPS", queryGroups),
                ("GR", groupedRatio),
                ("S", sequenceLength),
                ("CHANNELS", channels),
            ],
            grid: (rows * threads, 1, 1),
            threadGroup: (threads, 1, 1),
            outputShapes: [[batch, queryOrigin, sequenceLength, channels]],
            outputDTypes: [attnOrigin.dtype]
        )
        return outputs[0]
    }
}

public enum MotifGDAPost {
    private static let metalKernel = MLXFast.metalKernel(
        name: "motif_gda_post",
        inputNames: ["merged", "subln_w", "lambda_full", "scale_in", "eps_in"],
        outputNames: ["y"],
        source: motifGDAPostMetalSource
    )

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
        executionMode: MotifMetalKernelExecutionMode = .metalPreferred
    ) -> MLXArray {
        guard MotifMetalKernels.shouldUseMetal(executionMode: executionMode),
              queryGroups > 0,
              groupedRatio > 0,
              merged.dim(1) == queryGroups * groupedRatio + queryGroups
        else {
            MotifKernelFallbackTelemetry.recordFallback(.gdaPost)
            return reference(
                merged: merged,
                sublnWeight: sublnWeight,
                lambda: lambda,
                lambdaInit: lambdaInit,
                queryGroups: queryGroups,
                groupedRatio: groupedRatio,
                eps: eps
            )
        }
        return metal(
            merged: merged,
            sublnWeight: sublnWeight,
            lambda: lambda,
            lambdaInit: lambdaInit,
            queryGroups: queryGroups,
            groupedRatio: groupedRatio,
            eps: eps
        )
    }

    private static func metal(
        merged: MLXArray,
        sublnWeight: MLXArray,
        lambda: MLXArray,
        lambdaInit: Double,
        queryGroups: Int,
        groupedRatio: Int,
        eps: Float
    ) -> MLXArray {
        let batch = merged.dim(0)
        let queryOrigin = queryGroups * groupedRatio
        let sequenceLength = merged.dim(2)
        let channels = merged.dim(3)
        let rows = batch * queryOrigin * sequenceLength
        let threads = min(256, max(32, ((channels + 31) / 32) * 32))
        let scaleArray = MotifKernelScalarCache.scalar(Float(1.0 - lambdaInit))
        let epsArray = MotifKernelScalarCache.scalar(eps)
        let outputs = metalKernel(
            [
                merged,
                sublnWeight.asType(merged.dtype),
                lambda.asType(.float32),
                scaleArray,
                epsArray,
            ],
            template: [
                ("T", merged.dtype),
                ("Q_ORIGIN", queryOrigin),
                ("Q_GROUPS", queryGroups),
                ("GR", groupedRatio),
                ("S", sequenceLength),
                ("CHANNELS", channels),
            ],
            grid: (rows * threads, 1, 1),
            threadGroup: (threads, 1, 1),
            outputShapes: [[batch, queryOrigin, sequenceLength, channels]],
            outputDTypes: [merged.dtype]
        )
        return outputs[0]
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

    /// Production entry point. The direct Metal wrapper is default-on; set
    /// `MLX_MOTIF_DISABLE_KERNELS=1` or pass `.referenceOnly` to force the
    /// reference path for debugging and differential checks.
    public static func apply(
        _ x: MLXArray,
        weight: MLXArray,
        bias: MLXArray,
        eps: Float = 1e-6,
        executionMode: MotifMetalKernelExecutionMode = .metalPreferred
    ) -> MLXArray {
        guard MotifMetalKernels.shouldUseMetal(executionMode: executionMode) else {
            MotifKernelFallbackTelemetry.recordFallback(.polynorm)
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
        let epsArray = MotifKernelScalarCache.scalar(eps)
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

private let motifSDPADualVMetalSource = #"""
    constexpr int BN = 32;
    constexpr int BD = 32;
    constexpr int qk_per_thread = D / BD;
    constexpr int v_per_thread  = D / BD;

    typedef float U;

    uint head_idx = threadgroup_position_in_grid.x;     // batch*Q_HEADS linear index
    uint sg_gid   = simdgroup_index_in_threadgroup;
    uint sg_lid   = thread_index_in_simdgroup;

    // GQA broadcast: each `GQA_FACTOR` consecutive Q heads share one (K, V*) head.
    // For GQA_FACTOR=1 this is the standard one-to-one mapping.
    uint kv_head_idx = head_idx / uint(GQA_FACTOR);

    // KV length is a runtime input (not a template constant) so decode does
    // not recompile the kernel for every new context length. Row strides are
    // derived from each buffer's own capacity (its shape), so callers may pass
    // padded-capacity buffers together with a smaller live kv_len.
    const int  kv_len    = int(kv_len_in[0]);
    const uint k_row_cap = uint(k_shape[2]);
    const uint v_row_cap = uint(v1_shape[2]);

    const device T* q_p  = q  + head_idx    * D                          + sg_lid * qk_per_thread;
    const device T* k_p  = k  + kv_head_idx * (k_row_cap * D) + sg_gid * D + sg_lid * qk_per_thread;
    const device T* v1_p = v1 + kv_head_idx * (v_row_cap * D) + sg_gid * D + sg_lid * v_per_thread;
    const device T* v2_p = v2 + kv_head_idx * (v_row_cap * D) + sg_gid * D + sg_lid * v_per_thread;
    device T*       y_p  = y  + head_idx    * (2 * D);

    float scale = float(scale_in[0]);

    // Per-lane Q (scaled once).
    thread U q_r[qk_per_thread];
    for (int j = 0; j < qk_per_thread; ++j) {
        q_r[j] = scale * U(q_p[j]);
    }

    // Per-lane V accumulators — two slabs, v_per_thread channels each.
    thread U o1[v_per_thread];
    thread U o2[v_per_thread];
    for (int j = 0; j < v_per_thread; ++j) { o1[j] = 0; o2[j] = 0; }

    // -1e30 instead of -INF so fast::exp(-INF - finite) doesn't return NaN
    // on M-series (lesson from commits 6922acb / acc2de7).
    U max_score = U(-1e30f);
    U sum_exp_score = U(0);

    int kv_stride = BN * D;
    for (int p = sg_gid; p < kv_len; p += BN) {
        // Load this lane's K slice.
        thread U k_r[qk_per_thread];
        for (int j = 0; j < qk_per_thread; ++j) k_r[j] = U(k_p[j]);

        // QK dot, then simd_sum across the 32-lane simdgroup.
        U score = 0;
        for (int j = 0; j < qk_per_thread; ++j) score += q_r[j] * k_r[j];
        score = simd_sum(score);

        // Online softmax update.
        U new_max = max(max_score, score);
        U fac = metal::fast::exp(max_score - new_max);
        U exp_score = metal::fast::exp(score - new_max);
        max_score = new_max;
        sum_exp_score = sum_exp_score * fac + exp_score;

        // Update BOTH V accumulators with the same softmax weight.
        for (int j = 0; j < v_per_thread; ++j) {
            o1[j] = o1[j] * fac + exp_score * U(v1_p[j]);
            o2[j] = o2[j] * fac + exp_score * U(v2_p[j]);
        }

        k_p  += kv_stride;
        v1_p += kv_stride;
        v2_p += kv_stride;
    }

    // Cross-simdgroup max/sum reduce — MLX SDPA pattern, with the two V slabs
    // staged in disjoint halves of the scratch buffer so each channel
    // iteration needs 2 threadgroup barriers instead of 4. Reduction order per
    // slab is unchanged, so the result is bit-identical to the serial form.
    threadgroup U max_scores[BN];
    threadgroup U sum_exp_scores[BN];
    threadgroup U outputs[2 * BN * BD];

    if (sg_lid == 0) {
        max_scores[sg_gid] = max_score;
        sum_exp_scores[sg_gid] = sum_exp_score;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    U lane_max = max_scores[sg_lid];
    U new_max = simd_max(lane_max);
    U lane_factor = metal::fast::exp(lane_max - new_max);
    U lane_sum = simd_sum(sum_exp_scores[sg_lid] * lane_factor);

    // Reduce both V slabs across simdgroups.
    for (int i = 0; i < v_per_thread; ++i) {
        outputs[sg_lid * BD + sg_gid] = o1[i];
        outputs[BN * BD + sg_lid * BD + sg_gid] = o2[i];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        o1[i] = simd_sum(outputs[sg_gid * BD + sg_lid] * lane_factor);
        o2[i] = simd_sum(outputs[BN * BD + sg_gid * BD + sg_lid] * lane_factor);
        o1[i] = (lane_sum == 0) ? o1[i] : (o1[i] / lane_sum);
        o2[i] = (lane_sum == 0) ? o2[i] : (o2[i] / lane_sum);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Lane 0 of each simdgroup writes its v_per_thread channels of each slab.
    if (sg_lid == 0) {
        for (int i = 0; i < v_per_thread; ++i) {
            y_p[sg_gid * v_per_thread + i] = T(o1[i]);
            y_p[D + sg_gid * v_per_thread + i] = T(o2[i]);
        }
    }
"""#

private let motifSDPADualVQ4MetalSource = #"""
    constexpr int  BN            = 32;
    constexpr int  BD            = 32;
    constexpr int  qk_per_thread = D / BD;          // 4 for D=128
    constexpr int  v_per_thread  = D / BD;          // 4
    constexpr int  EL_PER_INT    = 32 / BITS;
    constexpr uint MASK          = (1u << BITS) - 1u;
    constexpr int  U32_PER_ROW   = D / EL_PER_INT;
    constexpr int  SCB_PER_ROW   = D / GROUP_SIZE;

    typedef float U;

    uint head_idx = threadgroup_position_in_grid.x;     // batch*Q_HEADS linear index
    uint sg_gid   = simdgroup_index_in_threadgroup;
    uint sg_lid   = thread_index_in_simdgroup;
    uint kv_head_idx = head_idx / uint(GQA_FACTOR);

    // Per-lane channel slice — same uint32_idx, shift, group_idx for K, V1, V2.
    uint base         = uint(sg_lid) * uint(qk_per_thread);
    uint u32_idx_base = base / uint(EL_PER_INT);
    uint shift_base   = (base % uint(EL_PER_INT)) * uint(BITS);
    uint group_idx    = base / uint(GROUP_SIZE);

    // KV length is a runtime input (not a template constant) so decode does
    // not recompile the kernel for every new context length. Row strides use
    // the packed K buffer's own capacity (its shape); the wrapper guard pins
    // every quantized slab to that same shape.
    const int  kv_len     = int(kv_len_in[0]);
    const uint kv_row_cap = uint(k_data_shape[2]);

    // Per-tensor row offset for this kv_head (computed once).
    uint head_row_u32 = kv_head_idx * kv_row_cap * uint(U32_PER_ROW);
    uint head_row_scb = kv_head_idx * kv_row_cap * uint(SCB_PER_ROW);

    // Q (fp16/bf16): same indexing as sdpa_dual_v.
    const device T* q_p = q + head_idx * uint(D) + sg_lid * uint(qk_per_thread);

    float scale = float(scale_in[0]);

    // Per-lane Q with the MLX qdot trick: precompute q_pre[j] = scale * q[j] /
    // 2^(shift_base + j*BITS). Then in the kv loop, K's nibble at position j —
    // which appears in the packed word as `nibble * 2^(shift_base + j*BITS)`
    // when masked WITHOUT shifting — multiplied by q_pre[j] gives the
    // correctly-weighted partial dot. Saves one shift per channel per step.
    thread U q_pre[qk_per_thread];
    U q_sum = U(0);
    for (int j = 0; j < qk_per_thread; ++j) {
        U qj = scale * U(q_p[j]);
        q_sum += qj;
        // Inverse of the bit-position factor 2^(shift_base + j*BITS).
        U inv_factor = U(1.0f) / U(1u << (shift_base + uint(j * BITS)));
        q_pre[j] = qj * inv_factor;
    }

    // Per-lane V accumulators — two slabs.
    thread U o1[v_per_thread];
    thread U o2[v_per_thread];
    for (int j = 0; j < v_per_thread; ++j) { o1[j] = 0; o2[j] = 0; }

    // Per-lane masks for the in-place K mask-without-shift trick.
    thread uint k_masks[qk_per_thread];
    for (int j = 0; j < qk_per_thread; ++j) {
        k_masks[j] = MASK << (shift_base + uint(j * BITS));
    }

    // -1e30 sentinel (avoid -INF / NaN trap in fast::exp).
    U max_score = U(-1e30f);
    U sum_exp_score = U(0);

    for (int p = sg_gid; p < kv_len; p += BN) {
        uint row_u32 = head_row_u32 + uint(p) * uint(U32_PER_ROW) + u32_idx_base;
        uint row_scb = head_row_scb + uint(p) * uint(SCB_PER_ROW) + group_idx;

        // ---- K: in-place mask + qdot trick -----------------------------
        // dot(q, dequant(k)) = scale_k * sum_j(q[j] * raw[j]) + bias_k * sum_j(q[j])
        //                    = scale_k * sum_j(q_pre[j] * (k_pkd & k_masks[j])) + bias_k * q_sum
        uint32_t k_pkd = k_data[row_u32];
        U partial = 0;
        for (int j = 0; j < qk_per_thread; ++j) {
            partial += q_pre[j] * U(k_pkd & k_masks[j]);
        }
        U score = U(k_scales[row_scb]) * partial + U(k_biases[row_scb]) * q_sum;
        score = simd_sum(score);

        // ---- Online softmax update -------------------------------------
        U new_max = max(max_score, score);
        U fac = metal::fast::exp(max_score - new_max);
        U exp_score = metal::fast::exp(score - new_max);
        max_score = new_max;
        sum_exp_score = sum_exp_score * fac + exp_score;

        // ---- V1: bit-extract + accumulate ------------------------------
        {
            uint32_t v_pkd = v1_data[row_u32];
            U s = U(v1_scales[row_scb]);
            U b = U(v1_biases[row_scb]);
            for (int j = 0; j < v_per_thread; ++j) {
                uint raw = (v_pkd >> (shift_base + uint(j * BITS))) & MASK;
                o1[j] = o1[j] * fac + exp_score * (s * U(raw) + b);
            }
        }

        // ---- V2: bit-extract + accumulate ------------------------------
        {
            uint32_t v_pkd = v2_data[row_u32];
            U s = U(v2_scales[row_scb]);
            U b = U(v2_biases[row_scb]);
            for (int j = 0; j < v_per_thread; ++j) {
                uint raw = (v_pkd >> (shift_base + uint(j * BITS))) & MASK;
                o2[j] = o2[j] * fac + exp_score * (s * U(raw) + b);
            }
        }
    }

    device T* y_p = y + head_idx * uint(2 * D);

    // Cross-simdgroup max/sum reduce — MLX SDPA pattern, with the two V slabs
    // staged in disjoint halves of the scratch buffer so each channel
    // iteration needs 2 threadgroup barriers instead of 4. Reduction order per
    // slab is unchanged, so the result is bit-identical to the serial form.
    threadgroup U max_scores[BN];
    threadgroup U sum_exp_scores[BN];
    threadgroup U outputs[2 * BN * BD];

    if (sg_lid == 0) {
        max_scores[sg_gid] = max_score;
        sum_exp_scores[sg_gid] = sum_exp_score;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    U lane_max = max_scores[sg_lid];
    U new_max = simd_max(lane_max);
    U lane_factor = metal::fast::exp(lane_max - new_max);
    U lane_sum = simd_sum(sum_exp_scores[sg_lid] * lane_factor);

    for (int i = 0; i < v_per_thread; ++i) {
        outputs[sg_lid * BD + sg_gid] = o1[i];
        outputs[BN * BD + sg_lid * BD + sg_gid] = o2[i];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        o1[i] = simd_sum(outputs[sg_gid * BD + sg_lid] * lane_factor);
        o2[i] = simd_sum(outputs[BN * BD + sg_gid * BD + sg_lid] * lane_factor);
        o1[i] = (lane_sum == 0) ? o1[i] : (o1[i] / lane_sum);
        o2[i] = (lane_sum == 0) ? o2[i] : (o2[i] / lane_sum);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (sg_lid == 0) {
        for (int i = 0; i < v_per_thread; ++i) {
            y_p[sg_gid * v_per_thread + i] = T(o1[i]);
            y_p[D + sg_gid * v_per_thread + i] = T(o2[i]);
        }
    }
"""#

private let motifGDAPostMetalSource = #"""
    // Layout: one threadgroup per output row (b, h_o, s). Each row has
    // CHANNELS = 2 * head_dim contiguous channels.
    uint row     = threadgroup_position_in_grid.x;
    uint tid     = thread_position_in_threadgroup.x;
    uint tgsize  = threads_per_threadgroup.x;

    // Decompose row into (b, h_o, s) using contiguous (B, q_origin, S) layout.
    uint hs    = (Q_ORIGIN * S);
    uint b     = row / hs;
    uint hosx  = row - b * hs;
    uint h_o   = hosx / S;
    uint s     = hosx - h_o * S;
    uint h_n   = Q_ORIGIN + (h_o / GR);   // matching noise head

    // merged is (B, Q_HEADS, S, CHANNELS) where Q_HEADS = Q_ORIGIN + Q_GROUPS.
    uint q_heads_total = Q_ORIGIN + Q_GROUPS;
    const device T* row_o = merged + (((b * q_heads_total + h_o) * S) + s) * CHANNELS;
    const device T* row_n = merged + (((b * q_heads_total + h_n) * S) + s) * CHANNELS;
    device T*       row_y = y      + (((b * Q_ORIGIN     + h_o) * S) + s) * CHANNELS;

    float lam   = float(lambda_full[0]);
    float scale = float(scale_in[0]);
    float eps   = float(eps_in[0]);

    // Pass 1: compute the per-row sum of squares of the differential.
    float ssq = 0.0f;
    for (uint i = tid; i < CHANNELS; i += tgsize) {
        float d = float(row_o[i]) - lam * float(row_n[i]);
        ssq += d * d;
    }

    threadgroup float tg_partial[32];
    float r = simd_sum(ssq);
    uint sg_id = simdgroup_index_in_threadgroup;
    uint lane  = thread_index_in_simdgroup;
    if (lane == 0) tg_partial[sg_id] = r;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg_id == 0) {
        uint n_sg = simdgroups_per_threadgroup;
        float v = (lane < n_sg) ? tg_partial[lane] : 0.0f;
        v = simd_sum(v);
        if (lane == 0) tg_partial[0] = v;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float rms_inv = metal::rsqrt(tg_partial[0] / float(CHANNELS) + eps);

    // Pass 2: emit the SubLN-normalised, scaled output.
    for (uint i = tid; i < CHANNELS; i += tgsize) {
        float d  = float(row_o[i]) - lam * float(row_n[i]);
        float sw = float(subln_w[i]);
        row_y[i] = T(d * rms_inv * sw * scale);
    }
"""#

private let motifGDAPostSplitMetalSource = #"""
    // Same as gda_post but reads attn_o (B, q_origin, S, 2d) and
    // attn_n (B, q_groups, S, 2d) as separate buffers — saves the
    // (B, q_origin+q_groups, S, 2d) concat allocation upstream.
    uint row     = threadgroup_position_in_grid.x;
    uint tid     = thread_position_in_threadgroup.x;
    uint tgsize  = threads_per_threadgroup.x;

    uint hs    = (Q_ORIGIN * S);
    uint b     = row / hs;
    uint hosx  = row - b * hs;
    uint h_o   = hosx / S;
    uint s     = hosx - h_o * S;
    uint h_n   = h_o / GR;       // noise head index (0..q_groups-1)

    const device T* row_o = attn_o + (((b * Q_ORIGIN + h_o) * S) + s) * CHANNELS;
    const device T* row_n = attn_n + (((b * Q_GROUPS + h_n) * S) + s) * CHANNELS;
    device T*       row_y = y      + (((b * Q_ORIGIN + h_o) * S) + s) * CHANNELS;

    float lam   = float(lambda_full[0]);
    float scale = float(scale_in[0]);
    float eps   = float(eps_in[0]);

    float ssq = 0.0f;
    for (uint i = tid; i < CHANNELS; i += tgsize) {
        float d = float(row_o[i]) - lam * float(row_n[i]);
        ssq += d * d;
    }

    threadgroup float tg_partial[32];
    float r = simd_sum(ssq);
    uint sg_id = simdgroup_index_in_threadgroup;
    uint lane  = thread_index_in_simdgroup;
    if (lane == 0) tg_partial[sg_id] = r;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg_id == 0) {
        uint n_sg = simdgroups_per_threadgroup;
        float v = (lane < n_sg) ? tg_partial[lane] : 0.0f;
        v = simd_sum(v);
        if (lane == 0) tg_partial[0] = v;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float rms_inv = metal::rsqrt(tg_partial[0] / float(CHANNELS) + eps);

    for (uint i = tid; i < CHANNELS; i += tgsize) {
        float d  = float(row_o[i]) - lam * float(row_n[i]);
        float sw = float(subln_w[i]);
        row_y[i] = T(d * rms_inv * sw * scale);
    }
"""#

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
