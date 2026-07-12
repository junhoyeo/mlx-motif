#if canImport(MLX) && canImport(MLXNN) && canImport(MLXLLM) && canImport(MLXLMCommon)
import Foundation
import MLX
import MLXLLM
import MLXLMCommon
import MLXNN
import MotifKit

/// All four grouped-cache slots (kOrigin/kNoise/value1/value2) are allocated
/// with a single head count taken from kOrigin. With `kRatio > 1` the model
/// produces kOrigin with `keyGroups*kRatio` heads but kNoise/value1/value2 with
/// `keyGroups` heads, so storing them in equally-shaped slots would fail (or
/// silently corrupt) at write time. Fail loudly here so the unsupported
/// combination is unmistakable even when a caller constructs the cache directly
/// (bypassing `MotifMLXModel.newCache`, which guards the same case up front).
@inline(__always)
func motifAssertUniformSlotHeads(
    _ kOrigin: MLXArray,
    _ kNoise: MLXArray,
    _ value1: MLXArray,
    _ value2: MLXArray
) {
    let h = kOrigin.dim(1)
    precondition(
        kNoise.dim(1) == h && value1.dim(1) == h && value2.dim(1) == h,
        "4-slot grouped KV cache requires kOrigin/kNoise/value1/value2 to share a "
            + "head count; got kOrigin=\(h), kNoise=\(kNoise.dim(1)), "
            + "value1=\(value1.dim(1)), value2=\(value2.dim(1)). This happens with "
            + "kRatio > 1, which the 4-slot cache does not support."
    )
}

/// Validates an `offset` recovered from a serialized `metaState` against the
/// (already-restored) slot capacity. Returns an error message when the offset
/// is unusable, or `nil` when it is fine. A serialized cache is untrusted
/// input (a corrupted or crafted prompt-cache file); a negative or
/// beyond-capacity offset would silently mis-slice the live region on every
/// subsequent fetch. Exposed as a pure helper because the `metaState` setter
/// cannot throw, so the boundary check is unit-testable without tripping the
/// `preconditionFailure` in the setter. (Python mirror: the `meta_state`
/// setters in src/mlx_motif/cache.py.)
func motifRestoredOffsetError(offset: Int, capacity: Int?) -> String? {
    if offset < 0 {
        return "restored cache offset must be non-negative; got \(offset)"
    }
    if let capacity, offset > capacity {
        return "restored cache offset \(offset) exceeds the restored slot capacity \(capacity)"
    }
    return nil
}

public enum MotifGroupedAttentionReferenceError: Error, LocalizedError, Equatable, Sendable {
    case requiresGroupedDifferentialAttention
    case invalidShape(String)

    public var errorDescription: String? {
        switch self {
        case .requiresGroupedDifferentialAttention:
            "Motif grouped attention reference path requires num_noise_heads in config.json"
        case .invalidShape(let detail):
            "Invalid Motif grouped attention tensor shape: \(detail)"
        }
    }
}

/// Buildable Swift-side handoff for the Python grouped differential attention
/// path. It intentionally documents the reference-only limitations before the
/// native model starts routing real tokens through this code.
public struct MotifGroupedAttentionReferencePlan: Codable, Equatable, Sendable {
    public var layout: MotifAttentionLayout
    public var cacheKind: MotifKVCacheKind
    public var limitations: [String]
    public var parityHooks: [String]

    public init(
        configuration: MotifModelConfiguration,
        cacheKind: MotifKVCacheKind = .groupedFourSlot
    ) throws {
        let layout = try MotifAttentionLayout(configuration: configuration)
        guard layout.variant == .groupedDifferential else {
            throw MotifGroupedAttentionReferenceError.requiresGroupedDifferentialAttention
        }
        self.layout = layout
        self.cacheKind = cacheKind
        self.limitations = [
            "direct custom Metal dispatch is default-on for decode shapes, with MLX_MOTIF_DISABLE_KERNELS=1 as fallback",
            "expects q/k/v tensors after projection and RoPE, not a full decoder layer",
            "q4/q8 grouped cache exposes packed tuples to the direct sdpa_dual_v_q4 path; the dequantizing bridge remains diagnostic-only",
        ]
        self.parityHooks = [
            "Python MotifAttention._forward_grouped fallback fixture",
            "src/mlx_motif/kernels/gda.py:gda_post_split_reference",
            "src/mlx_motif/kernels/attention.py:sdpa_dual_v_reference",
            "tests/test_model.py grouped cache path cases",
        ]
    }
}

public struct MotifGroupedProjectionSlices {
    public var qOrigin: MLXArray
    public var qNoise: MLXArray
    public var kOrigin: MLXArray
    public var kNoise: MLXArray
    public var value1: MLXArray
    public var value2: MLXArray

    public init(
        qOrigin: MLXArray,
        qNoise: MLXArray,
        kOrigin: MLXArray,
        kNoise: MLXArray,
        value1: MLXArray,
        value2: MLXArray
    ) {
        self.qOrigin = qOrigin
        self.qNoise = qNoise
        self.kOrigin = kOrigin
        self.kNoise = kNoise
        self.value1 = value1
        self.value2 = value2
    }
}

public struct MotifGroupedKVCachedSlices {
    public var kOrigin: MLXArray
    public var kNoise: MLXArray
    public var value1: MLXArray
    public var value2: MLXArray

    public init(kOrigin: MLXArray, kNoise: MLXArray, value1: MLXArray, value2: MLXArray) {
        self.kOrigin = kOrigin
        self.kNoise = kNoise
        self.value1 = value1
        self.value2 = value2
    }
}

public enum MotifAttentionMaskPlanKind: String, Codable, Equatable, Sendable {
    case none
    case causal
    case materializedCausal
}

/// Shape-only attention-mask plan for the grouped reference path. Keeping this
/// separate from `MLXFast.ScaledDotProductAttentionMaskMode` makes RoPE/cache
/// parity testable without evaluating MLX runtime kernels.
public struct MotifAttentionMaskPlan: Codable, Equatable, Sendable {
    public var kind: MotifAttentionMaskPlanKind
    public var queryLength: Int
    public var keyValueLength: Int
    public var cacheOffset: Int
    public var windowSize: Int?
    public var materializedShape: [Int]?

    public init(
        kind: MotifAttentionMaskPlanKind,
        queryLength: Int,
        keyValueLength: Int,
        cacheOffset: Int,
        windowSize: Int? = nil,
        materializedShape: [Int]? = nil
    ) {
        self.kind = kind
        self.queryLength = queryLength
        self.keyValueLength = keyValueLength
        self.cacheOffset = cacheOffset
        self.windowSize = windowSize
        self.materializedShape = materializedShape
    }

    public var requiresMaterializedArray: Bool {
        kind == .materializedCausal
    }

    public func makeMask() -> MLXFast.ScaledDotProductAttentionMaskMode {
        switch kind {
        case .none:
            return .none
        case .causal:
            return .causal
        case .materializedCausal:
            return .array(createCausalMask(n: queryLength, offset: cacheOffset, windowSize: windowSize))
        }
    }
}

/// Deterministic shape contract for one grouped-differential attention step.
/// This mirrors the Python decode/prefill routing decisions without owning
/// MotifMLXBackend token generation.
public struct MotifGroupedAttentionStepPlan: Codable, Equatable, Sendable {
    public var batchSize: Int
    public var queryLength: Int
    public var cacheOffset: Int
    public var keyValueLength: Int
    public var ropePositionRange: [Int]
    public var qOriginShape: [Int]
    public var qNoiseShape: [Int]
    public var kOriginUpdateShape: [Int]
    public var kNoiseUpdateShape: [Int]
    public var valueUpdateShape: [Int]
    public var kOriginCachedShape: [Int]
    public var kNoiseCachedShape: [Int]
    public var valueCachedShape: [Int]
    public var outputShape: [Int]
    public var attentionPath: MotifAttentionPath
    public var maskPlan: MotifAttentionMaskPlan
}

/// Four-slot grouped KV cache matching the Python `MotifGroupedKVCache` split:
/// origin keys, noise keys, value slab 1, value slab 2. This is deliberately
/// fp16/bf16 reference storage; `MotifGroupedQuantizedKVCache` below owns the
/// opt-in q4/q8 packed-cache path.
public final class MotifGroupedKVCacheReference {
    public private(set) var offset: Int = 0

    private var kOrigin: MLXArray?
    private var kNoise: MLXArray?
    private var value1: MLXArray?
    private var value2: MLXArray?

    public init() {}

    public var isEmpty: Bool { offset == 0 }
    public var cachedLength: Int { offset }

    public func reset() {
        offset = 0
        kOrigin = nil
        kNoise = nil
        value1 = nil
        value2 = nil
    }

    public func updateAndFetch(
        kOrigin newKOrigin: MLXArray,
        kNoise newKNoise: MLXArray,
        value1 newValue1: MLXArray,
        value2 newValue2: MLXArray
    ) -> MotifGroupedKVCachedSlices {
        kOrigin = appendCached(kOrigin, newKOrigin)
        kNoise = appendCached(kNoise, newKNoise)
        value1 = appendCached(value1, newValue1)
        value2 = appendCached(value2, newValue2)
        offset = kOrigin?.shape[safe: 2] ?? 0

        return MotifGroupedKVCachedSlices(
            kOrigin: kOrigin!,
            kNoise: kNoise!,
            value1: value1!,
            value2: value2!
        )
    }

    public func fetch() -> MotifGroupedKVCachedSlices? {
        guard let kOrigin, let kNoise, let value1, let value2 else {
            return nil
        }
        return MotifGroupedKVCachedSlices(
            kOrigin: kOrigin,
            kNoise: kNoise,
            value1: value1,
            value2: value2
        )
    }

    /// Reorder cached batch rows for beam-search style reuse. This mirrors the
    /// batch-axis gather used by mlx-lm KV caches while leaving sequence offsets
    /// untouched.
    public func reorder(batchIndices: [Int]) {
        guard !batchIndices.isEmpty else { return }
        let indices = MLXArray(batchIndices.map(Int32.init), [batchIndices.count])
        kOrigin = kOrigin.map { take($0, indices, axis: 0) }
        kNoise = kNoise.map { take($0, indices, axis: 0) }
        value1 = value1.map { take($0, indices, axis: 0) }
        value2 = value2.map { take($0, indices, axis: 0) }
    }

    private func appendCached(_ cached: MLXArray?, _ next: MLXArray) -> MLXArray {
        guard let cached else { return next }
        return concatenated([cached, next], axis: 2)
    }
}

public typealias MotifQuantizedTuple = (data: MLXArray, scales: MLXArray, biases: MLXArray?)

public enum MotifGroupedKVCacheError: Error, LocalizedError, Equatable, Sendable {
    case unsupportedSingleSlotUpdate
    case invalidStateCount(Int)

    public var errorDescription: String? {
        switch self {
        case .unsupportedSingleSlotUpdate:
            "Motif grouped KV cache requires updateAndFetch4(kOrigin:kNoise:value1:value2:)."
        case .invalidStateCount(let count):
            "Motif grouped quantized KV cache expected 8 or 12 state arrays; got \(count)."
        }
    }
}

/// Production KVCache-compatible four-slot grouped cache. This mirrors the
/// Python `MotifGroupedKVCache`: k-origin, k-noise, value-1, and value-2 are
/// stored independently so grouped attention can avoid post-cache head splits.
public final class MotifGroupedKVCache: KVCache, CustomDebugStringConvertible {
    public var offset: Int = 0
    public var maxSize: Int? { nil }
    private var kOrigin: MLXArray?
    private var kNoise: MLXArray?
    private var value1: MLXArray?
    private var value2: MLXArray?
    public var step: Int

    public init(step: Int = 256) {
        self.step = step
    }

    public func innerState() -> [MLXArray] {
        [kOrigin, kNoise, value1, value2].compactMap { $0 }
    }

    public func update(keys _: MLXArray, values _: MLXArray) -> (MLXArray, MLXArray) {
        fatalError(MotifGroupedKVCacheError.unsupportedSingleSlotUpdate.localizedDescription)
    }

    /// Append the four incoming slices and return the FULL step-padded capacity
    /// buffers (axis-2 length is the allocated capacity, a multiple of `step`,
    /// NOT the live length `offset`). The buffers are row-contiguous, so feeding
    /// them straight to the `sdpa_dual_v` decode kernel (with `kvLen = offset`)
    /// avoids the per-step `ensure_row_contiguous` copy of the live KV region
    /// that an exact-length `[..., :offset, :]` view would force before every
    /// custom-kernel launch. Consumers that cannot take a runtime length bound
    /// (the reference / fallback path) must slice back to `[..., :offset, :]`.
    /// Mirrors the Python `MotifGroupedKVCache.update_and_fetch_4`
    /// (src/mlx_motif/cache.py).
    public func updateAndFetch4(
        kOrigin newKOrigin: MLXArray,
        kNoise newKNoise: MLXArray,
        value1 newValue1: MLXArray,
        value2 newValue2: MLXArray
    ) -> MotifGroupedKVCachedSlices {
        motifAssertUniformSlotHeads(newKOrigin, newKNoise, newValue1, newValue2)
        let previous = offset
        growIfNeeded(
            batch: newKOrigin.dim(0),
            heads: newKOrigin.dim(1),
            sequenceLength: newKOrigin.dim(2),
            headDim: newKOrigin.dim(3),
            dtype: newKOrigin.dtype
        )
        offset += newKOrigin.dim(2)

        kOrigin![.ellipsis, previous ..< offset, 0...] = newKOrigin
        kNoise![.ellipsis, previous ..< offset, 0...] = newKNoise
        value1![.ellipsis, previous ..< offset, 0...] = newValue1
        value2![.ellipsis, previous ..< offset, 0...] = newValue2

        return MotifGroupedKVCachedSlices(
            kOrigin: kOrigin!,
            kNoise: kNoise!,
            value1: value1!,
            value2: value2!
        )
    }

    private func growIfNeeded(batch: Int, heads: Int, sequenceLength: Int, headDim: Int, dtype: DType) {
        let previous = offset
        guard kOrigin == nil || previous + sequenceLength > kOrigin!.dim(2) else { return }
        let newSteps = ((step + sequenceLength - 1) / step) * step
        let freshShape = [batch, heads, newSteps, headDim]

        func fresh() -> MLXArray {
            MLXArray.zeros(freshShape, dtype: dtype)
        }

        if var kOrigin, var kNoise, var value1, var value2 {
            if previous % step != 0 {
                kOrigin = kOrigin[.ellipsis, ..<previous, 0...]
                kNoise = kNoise[.ellipsis, ..<previous, 0...]
                value1 = value1[.ellipsis, ..<previous, 0...]
                value2 = value2[.ellipsis, ..<previous, 0...]
            }
            self.kOrigin = concatenated([kOrigin, fresh()], axis: 2)
            self.kNoise = concatenated([kNoise, fresh()], axis: 2)
            self.value1 = concatenated([value1, fresh()], axis: 2)
            self.value2 = concatenated([value2, fresh()], axis: 2)
        } else {
            kOrigin = fresh()
            kNoise = fresh()
            value1 = fresh()
            value2 = fresh()
        }
    }

    public var state: [MLXArray] {
        get {
            guard let kOrigin, let kNoise, let value1, let value2 else { return [] }
            return [
                kOrigin[.ellipsis, ..<offset, 0...],
                kNoise[.ellipsis, ..<offset, 0...],
                value1[.ellipsis, ..<offset, 0...],
                value2[.ellipsis, ..<offset, 0...],
            ]
        }
        set {
            guard newValue.count == 4 else {
                fatalError("MotifGroupedKVCache state must have exactly 4 arrays")
            }
            kOrigin = newValue[0]
            kNoise = newValue[1]
            value1 = newValue[2]
            value2 = newValue[3]
            offset = newValue[0].dim(2)
        }
    }

    public var metaState: [String] {
        get { [String(offset), String(step)] }
        set {
            guard !newValue.isEmpty else { return }
            let restored = Int(newValue[0]) ?? 0
            if let reason = motifRestoredOffsetError(offset: restored, capacity: kOrigin?.dim(2)) {
                preconditionFailure(reason)
            }
            offset = restored
            if newValue.count > 1 {
                step = Int(newValue[1]) ?? step
            }
        }
    }

    public var isTrimmable: Bool { true }

    @discardableResult
    public func trim(_ n: Int) -> Int {
        let trimmed = max(0, min(offset, n))
        offset -= trimmed
        return trimmed
    }

    public func makeMask(
        n: Int,
        windowSize: Int?,
        returnArray: Bool
    ) -> MLXFast.ScaledDotProductAttentionMaskMode {
        if n == 1 { return .none }
        if returnArray || (windowSize != nil && n > windowSize!) {
            return .array(createCausalMask(n: n, offset: offset, windowSize: windowSize))
        }
        return .causal
    }

    public var debugDescription: String {
        "MotifGroupedKVCache(offset: \(offset), step: \(step))"
    }
}

/// Quantized four-slot grouped cache. It exposes the packed tuples for the
/// future `sdpa_dual_v_q4` Metal path and also provides a dequantized bridge
/// so the native Swift model can use q4/q8 cache modes today.
public final class MotifGroupedQuantizedKVCache: KVCache, CustomDebugStringConvertible {
    public var offset: Int = 0
    public var maxSize: Int? { nil }
    private var kOrigin: MotifQuantizedTuple?
    private var kNoise: MotifQuantizedTuple?
    private var value1: MotifQuantizedTuple?
    private var value2: MotifQuantizedTuple?
    public var step: Int
    public let groupSize: Int
    public let bits: Int
    public let mode: QuantizationMode

    public init(groupSize: Int = 64, bits: Int = 4, mode: QuantizationMode = .affine, step: Int = 256) {
        self.groupSize = groupSize
        self.bits = bits
        self.mode = mode
        self.step = step
    }

    public func innerState() -> [MLXArray] {
        state
    }

    public func update(keys _: MLXArray, values _: MLXArray) -> (MLXArray, MLXArray) {
        fatalError(MotifGroupedKVCacheError.unsupportedSingleSlotUpdate.localizedDescription)
    }

    /// Dequantizing bridge for the fp16/bf16 attention fallbacks. Returns
    /// EXACT-length (`[..., :offset, :]`) dequantized slices, since those
    /// consumers (the reference / prefill path) have no runtime length bound.
    /// The bandwidth-saving `sdpa_dual_v_q4` decode path uses
    /// `updateAndFetch4Quantized` instead. Mirrors the Python
    /// `MotifGroupedQuantizedKVCache.update_and_fetch_4` dequant bridge.
    public func updateAndFetch4(
        kOrigin newKOrigin: MLXArray,
        kNoise newKNoise: MLXArray,
        value1 newValue1: MLXArray,
        value2 newValue2: MLXArray
    ) -> MotifGroupedKVCachedSlices {
        let packed = updateAndFetch4Quantized(
            kOrigin: newKOrigin,
            kNoise: newKNoise,
            value1: newValue1,
            value2: newValue2
        )
        // `updateAndFetch4Quantized` now returns FULL step-padded capacity
        // triples, so bound each to the live region (`live(...)`) before
        // dequantizing for the exact-length fp16 fallback consumers.
        return MotifGroupedKVCachedSlices(
            kOrigin: dequantize(live(packed.kOrigin), dtype: newKOrigin.dtype),
            kNoise: dequantize(live(packed.kNoise), dtype: newKNoise.dtype),
            value1: dequantize(live(packed.value1), dtype: newValue1.dtype),
            value2: dequantize(live(packed.value2), dtype: newValue2.dtype)
        )
    }

    /// Append the four incoming slices and return each slot's FULL step-padded
    /// capacity triple `(data, scales, biases)` — axis-2 length is the allocated
    /// capacity, NOT the live length `offset`. The triples are row-contiguous so
    /// `sdpa_dual_v_q4` can read packed memory directly (with `kvLen = offset`)
    /// without the per-step `ensure_row_contiguous` copy of all nine live-region
    /// buffers that exact-length `[..., :offset, :]` views would force. Mirrors
    /// the Python `MotifGroupedQuantizedKVCache.update_and_fetch_4_quantized`.
    public func updateAndFetch4Quantized(
        kOrigin newKOrigin: MLXArray,
        kNoise newKNoise: MLXArray,
        value1 newValue1: MLXArray,
        value2 newValue2: MLXArray
    ) -> (
        kOrigin: MotifQuantizedTuple,
        kNoise: MotifQuantizedTuple,
        value1: MotifQuantizedTuple,
        value2: MotifQuantizedTuple
    ) {
        motifAssertUniformSlotHeads(newKOrigin, newKNoise, newValue1, newValue2)
        let previous = offset
        growIfNeeded(
            batch: newKOrigin.dim(0),
            heads: newKOrigin.dim(1),
            sequenceLength: newKOrigin.dim(2),
            headDim: newKOrigin.dim(3),
            dtype: newKOrigin.dtype
        )
        offset += newKOrigin.dim(2)

        writeQuantized(newKOrigin, into: &kOrigin, range: previous ..< offset)
        writeQuantized(newKNoise, into: &kNoise, range: previous ..< offset)
        writeQuantized(newValue1, into: &value1, range: previous ..< offset)
        writeQuantized(newValue2, into: &value2, range: previous ..< offset)

        return (
            kOrigin!,
            kNoise!,
            value1!,
            value2!
        )
    }

    private func initQuantizedStorage(batch: Int, heads: Int, sequenceLength: Int, headDim: Int, dtype: DType)
        -> MotifQuantizedTuple
    {
        // A head_dim that doesn't divide into whole quantization groups (or
        // whole packed uint32 words) would silently truncate `packedLast` /
        // `groups` below and then fail opaquely inside `quantized(...)` at the
        // first cache write. The model-level guard
        // (`MotifMLXModel.fourSlotCacheUnsupportedReason`) rejects this before
        // a cache is ever built; this precondition keeps direct constructions
        // honest. (Python mirror: `MotifGroupedQuantizedKVCache
        // ._validate_head_dim` in src/mlx_motif/cache.py.)
        precondition(
            groupSize > 0 && bits > 0 && headDim % groupSize == 0 && (headDim * bits) % 32 == 0,
            "Quantized 4-slot KV cache requires head_dim divisible by group_size and "
                + "by the packed uint32 word; got head_dim=\(headDim), "
                + "groupSize=\(groupSize), bits=\(bits)"
        )
        // Allocate the packed storage directly instead of materializing a full
        // fp zeros slab and running the quantize kernel over it just to obtain
        // correctly-shaped buffers. Quantizing zeros produces all-zero packed
        // words / scales / biases, so plain zeros of the packed shapes are
        // bit-identical — and skip a throwaway slab plus a quantize dispatch on
        // every growth step. Mirrors the Python reference `_init_storage`
        // (src/mlx_motif/cache.py). Only the default affine layout is safe to
        // synthesize by hand; other modes fall back to quantizing zeros.
        if mode == .affine {
            let packedLast = headDim * bits / 32
            let groups = headDim / groupSize
            let data = MLXArray.zeros([batch, heads, sequenceLength, packedLast], dtype: .uint32)
            let scales = MLXArray.zeros([batch, heads, sequenceLength, groups], dtype: dtype)
            let biases = MLXArray.zeros([batch, heads, sequenceLength, groups], dtype: dtype)
            return (data, scales, biases)
        }
        let zeros = MLXArray.zeros([batch, heads, sequenceLength, headDim], dtype: dtype)
        let q = quantized(zeros, groupSize: groupSize, bits: bits, mode: mode)
        return (q.wq, q.scales, q.biases)
    }

    private func growIfNeeded(batch: Int, heads: Int, sequenceLength: Int, headDim: Int, dtype: DType) {
        let previous = offset
        guard kOrigin == nil || previous + sequenceLength > kOrigin!.data.dim(2) else { return }
        let newSteps = ((step + sequenceLength - 1) / step) * step
        func fresh() -> MotifQuantizedTuple {
            initQuantizedStorage(batch: batch, heads: heads, sequenceLength: newSteps, headDim: headDim, dtype: dtype)
        }
        func expand(_ tuple: MotifQuantizedTuple) -> MotifQuantizedTuple {
            let trimmed = previous % step == 0 ? tuple : live(tuple)
            let add = fresh()
            return (
                concatenated([trimmed.data, add.data], axis: 2),
                concatenated([trimmed.scales, add.scales], axis: 2),
                optionalConcatenate(trimmed.biases, add.biases, axis: 2)
            )
        }
        if let kOrigin, let kNoise, let value1, let value2 {
            self.kOrigin = expand(kOrigin)
            self.kNoise = expand(kNoise)
            self.value1 = expand(value1)
            self.value2 = expand(value2)
        } else {
            kOrigin = fresh()
            kNoise = fresh()
            value1 = fresh()
            value2 = fresh()
        }
    }

    private func writeQuantized(_ fresh: MLXArray, into slot: inout MotifQuantizedTuple?, range: Range<Int>) {
        let q = quantized(fresh, groupSize: groupSize, bits: bits, mode: mode)
        guard let current = slot else {
            fatalError("Quantized cache storage not initialized")
        }
        current.data[.ellipsis, range, 0...] = q.wq
        current.scales[.ellipsis, range, 0...] = q.scales
        if let biases = q.biases {
            current.biases![.ellipsis, range, 0...] = biases
        }
        slot = current
    }

    private func live(_ tuple: MotifQuantizedTuple) -> MotifQuantizedTuple {
        (
            tuple.data[.ellipsis, ..<offset, 0...],
            tuple.scales[.ellipsis, ..<offset, 0...],
            tuple.biases?[.ellipsis, ..<offset, 0...]
        )
    }

    public func dequantize(_ tuple: MotifQuantizedTuple, dtype: DType? = nil) -> MLXArray {
        dequantized(
            tuple.data,
            scales: tuple.scales,
            biases: tuple.biases,
            groupSize: groupSize,
            bits: bits,
            mode: mode,
            dtype: dtype
        )
    }

    private func optionalConcatenate(_ lhs: MLXArray?, _ rhs: MLXArray?, axis: Int) -> MLXArray? {
        guard let lhs, let rhs else { return nil }
        return concatenated([lhs, rhs], axis: axis)
    }

    public var state: [MLXArray] {
        get {
            guard let kOrigin, let kNoise, let value1, let value2 else { return [] }
            let tuples = [live(kOrigin), live(kNoise), live(value1), live(value2)]
            return tuples.flatMap { [$0.data, $0.scales, $0.biases].compactMap { $0 } }
        }
        set {
            switch newValue.count {
            case 8:
                kOrigin = (newValue[0], newValue[1], nil)
                kNoise = (newValue[2], newValue[3], nil)
                value1 = (newValue[4], newValue[5], nil)
                value2 = (newValue[6], newValue[7], nil)
            case 12:
                kOrigin = (newValue[0], newValue[1], newValue[2])
                kNoise = (newValue[3], newValue[4], newValue[5])
                value1 = (newValue[6], newValue[7], newValue[8])
                value2 = (newValue[9], newValue[10], newValue[11])
            default:
                fatalError(MotifGroupedKVCacheError.invalidStateCount(newValue.count).localizedDescription)
            }
            offset = kOrigin?.data.dim(2) ?? 0
        }
    }

    public var metaState: [String] {
        get { [String(offset), String(groupSize), String(bits), String(step)] }
        set {
            guard !newValue.isEmpty else { return }
            // groupSize/bits are immutable `let`s here, so — unlike the Python
            // mirror — the restore path cannot corrupt them; only offset needs
            // the boundary check (capacity lives on the packed triple's data).
            let restored = Int(newValue[0]) ?? 0
            if let reason = motifRestoredOffsetError(offset: restored, capacity: kOrigin?.data.dim(2)) {
                preconditionFailure(reason)
            }
            offset = restored
            if newValue.count > 3 {
                step = Int(newValue[3]) ?? step
            }
        }
    }

    public var isTrimmable: Bool { true }

    @discardableResult
    public func trim(_ n: Int) -> Int {
        let trimmed = max(0, min(offset, n))
        offset -= trimmed
        return trimmed
    }

    public func makeMask(
        n: Int,
        windowSize: Int?,
        returnArray: Bool
    ) -> MLXFast.ScaledDotProductAttentionMaskMode {
        if n == 1 { return .none }
        if returnArray || (windowSize != nil && n > windowSize!) {
            return .array(createCausalMask(n: n, offset: offset, windowSize: windowSize))
        }
        return .causal
    }

    public var debugDescription: String {
        "MotifGroupedQuantizedKVCache(offset: \(offset), groupSize: \(groupSize), bits: \(bits))"
    }
}

public enum MotifGroupedDifferentialAttentionReference {
    public static func planStep(
        layout: MotifAttentionLayout,
        batchSize: Int,
        sequenceLength: Int,
        cacheOffset: Int = 0,
        cacheKind: MotifKVCacheKind = .groupedFourSlot,
        fusedRope: Bool = false,
        featureFlags: MotifRuntimeFeatureFlags = .init(),
        forceMaterializedMask: Bool = false,
        slidingWindow: Int? = nil
    ) throws -> MotifGroupedAttentionStepPlan {
        guard layout.variant == .groupedDifferential else {
            throw MotifGroupedAttentionReferenceError.requiresGroupedDifferentialAttention
        }
        guard batchSize > 0 else {
            throw MotifGroupedAttentionReferenceError.invalidShape("batchSize must be positive")
        }
        guard sequenceLength > 0 else {
            throw MotifGroupedAttentionReferenceError.invalidShape("sequenceLength must be positive")
        }
        guard cacheOffset >= 0 else {
            throw MotifGroupedAttentionReferenceError.invalidShape("cacheOffset must be non-negative")
        }

        let noiseHeads = layout.requiredNoiseHeads
        let originHeads = noiseHeads * layout.groupedRatio
        let keyNoiseHeads = layout.requiredKeyNoiseHeads
        let keyValueLength = cacheOffset + sequenceLength
        let maskPlan = makeMaskPlan(
            sequenceLength: sequenceLength,
            cacheOffset: cacheOffset,
            forceMaterialized: forceMaterializedMask,
            slidingWindow: slidingWindow
        )
        let attentionPath = MotifAttentionPathResolver.resolve(
            cacheKind: cacheKind,
            sequenceLength: sequenceLength,
            keyValueRepeat: layout.keyValueRepeat,
            keyRatio: layout.keyRatio,
            fusedRope: fusedRope,
            featureFlags: featureFlags
        )

        return MotifGroupedAttentionStepPlan(
            batchSize: batchSize,
            queryLength: sequenceLength,
            cacheOffset: cacheOffset,
            keyValueLength: keyValueLength,
            ropePositionRange: [cacheOffset, keyValueLength],
            qOriginShape: [batchSize, originHeads, sequenceLength, layout.headDim],
            qNoiseShape: [batchSize, noiseHeads, sequenceLength, layout.headDim],
            kOriginUpdateShape: [batchSize, keyNoiseHeads * layout.keyRatio, sequenceLength, layout.headDim],
            kNoiseUpdateShape: [batchSize, keyNoiseHeads, sequenceLength, layout.headDim],
            valueUpdateShape: [batchSize, keyNoiseHeads, sequenceLength, layout.headDim],
            kOriginCachedShape: [batchSize, keyNoiseHeads * layout.keyRatio, keyValueLength, layout.headDim],
            kNoiseCachedShape: [batchSize, keyNoiseHeads, keyValueLength, layout.headDim],
            valueCachedShape: [batchSize, keyNoiseHeads, keyValueLength, layout.headDim],
            outputShape: [batchSize, originHeads, sequenceLength, 2 * layout.headDim],
            attentionPath: attentionPath,
            maskPlan: maskPlan
        )
    }

    public static func makeMaskPlan(
        sequenceLength: Int,
        cacheOffset: Int = 0,
        forceMaterialized: Bool = false,
        slidingWindow: Int? = nil
    ) -> MotifAttentionMaskPlan {
        let keyValueLength = cacheOffset + sequenceLength
        if sequenceLength == 1 {
            return MotifAttentionMaskPlan(
                kind: .none,
                queryLength: sequenceLength,
                keyValueLength: keyValueLength,
                cacheOffset: cacheOffset,
                windowSize: slidingWindow
            )
        }

        if forceMaterialized || (slidingWindow != nil && sequenceLength > slidingWindow!) {
            return MotifAttentionMaskPlan(
                kind: .materializedCausal,
                queryLength: sequenceLength,
                keyValueLength: keyValueLength,
                cacheOffset: cacheOffset,
                windowSize: slidingWindow,
                materializedShape: [sequenceLength, keyValueLength]
            )
        }

        return MotifAttentionMaskPlan(
            kind: .causal,
            queryLength: sequenceLength,
            keyValueLength: keyValueLength,
            cacheOffset: cacheOffset,
            windowSize: slidingWindow
        )
    }

    /// Split projected grouped-attention tensors into the six logical streams
    /// consumed by the reference attention/cache path. Inputs mirror Python
    /// `_forward_grouped` immediately after q/k/v projection and before cache
    /// append: flat tensors have shape `[B, S, projection_width]`.
    public static func splitProjectedTensors(
        qFlat: MLXArray,
        kFlat: MLXArray,
        vFlat: MLXArray,
        layout: MotifAttentionLayout,
        qOriginFirst: Bool = false
    ) throws -> MotifGroupedProjectionSlices {
        guard layout.variant == .groupedDifferential else {
            throw MotifGroupedAttentionReferenceError.requiresGroupedDifferentialAttention
        }
        guard let batch = qFlat.shape[safe: 0], let sequenceLength = qFlat.shape[safe: 1] else {
            throw MotifGroupedAttentionReferenceError.invalidShape("qFlat must be [B, S, projection_width]")
        }

        let q = qFlat
            .reshaped([batch, sequenceLength, layout.queryHeads, layout.headDim])
            .transposed(0, 2, 1, 3)
        let k = kFlat
            .reshaped([batch, sequenceLength, layout.keyValueHeads, layout.headDim])
            .transposed(0, 2, 1, 3)
        let v = vFlat
            .reshaped([batch, sequenceLength, 2 * layout.requiredKeyNoiseHeads, layout.headDim])
            .transposed(0, 2, 1, 3)

        return try splitPreparedTensors(
            q: q,
            k: k,
            v: v,
            layout: layout,
            qOriginFirst: qOriginFirst
        )
    }

    /// Split prepared `[B, heads, S, D]` tensors. Use this after RoPE has been
    /// applied to q/k in a full model implementation.
    public static func splitPreparedTensors(
        q: MLXArray,
        k: MLXArray,
        v: MLXArray,
        layout: MotifAttentionLayout,
        qOriginFirst: Bool = false
    ) throws -> MotifGroupedProjectionSlices {
        guard let batch = q.shape[safe: 0], let sequenceLength = q.shape[safe: 2] else {
            throw MotifGroupedAttentionReferenceError.invalidShape("q must be [B, H, S, D]")
        }
        let noiseHeads = layout.requiredNoiseHeads
        let originHeads = noiseHeads * layout.groupedRatio

        let qOrigin: MLXArray
        let qNoise: MLXArray
        if qOriginFirst {
            let pieces = q.split(indices: [originHeads], axis: 1)
            qOrigin = pieces[0]
            qNoise = pieces[1]
        } else {
            let qGrouped = q.reshaped([
                batch,
                noiseHeads,
                layout.groupedRatio + 1,
                sequenceLength,
                layout.headDim,
            ])
            let pieces = qGrouped.split(indices: [layout.groupedRatio], axis: 2)
            qOrigin = pieces[0].reshaped([batch, originHeads, sequenceLength, layout.headDim])
            qNoise = pieces[1].reshaped([batch, noiseHeads, sequenceLength, layout.headDim])
        }

        let keyGroups = layout.requiredKeyNoiseHeads
        let keyGrouped = k.reshaped([
            batch,
            keyGroups,
            layout.keyRatio + 1,
            k.shape[safe: 2] ?? sequenceLength,
            layout.headDim,
        ])
        let keyPieces = keyGrouped.split(indices: [layout.keyRatio], axis: 2)
        let kOrigin = keyPieces[0].reshaped([
            batch,
            keyGroups * layout.keyRatio,
            k.shape[safe: 2] ?? sequenceLength,
            layout.headDim,
        ])
        let kNoise = keyPieces[1].reshaped([
            batch,
            keyGroups,
            k.shape[safe: 2] ?? sequenceLength,
            layout.headDim,
        ])

        let valueGrouped = v.reshaped([
            batch,
            keyGroups,
            2,
            v.shape[safe: 2] ?? sequenceLength,
            layout.headDim,
        ])
        let valuePieces = valueGrouped.split(indices: [1], axis: 2)
        let value1 = valuePieces[0].reshaped([
            batch,
            keyGroups,
            v.shape[safe: 2] ?? sequenceLength,
            layout.headDim,
        ])
        let value2 = valuePieces[1].reshaped([
            batch,
            keyGroups,
            v.shape[safe: 2] ?? sequenceLength,
            layout.headDim,
        ])

        return MotifGroupedProjectionSlices(
            qOrigin: qOrigin,
            qNoise: qNoise,
            kOrigin: kOrigin,
            kNoise: kNoise,
            value1: value1,
            value2: value2
        )
    }

    /// Pure MLX Swift grouped differential attention reference. It mirrors the
    /// Python fallback math: two dual-value SDPAs followed by differential
    /// subtract, SubLN/RMSNorm, and `(1 - lambda_init)` scaling.
    public static func apply(
        qOrigin: MLXArray,
        qNoise: MLXArray,
        kOrigin: MLXArray,
        kNoise: MLXArray,
        value1: MLXArray,
        value2: MLXArray,
        sublnWeight: MLXArray,
        lambda: MLXArray,
        lambdaInit: Double,
        groupedRatio: Int,
        scale: Float,
        mask: MLXFast.ScaledDotProductAttentionMaskMode = .none,
        eps: Float = 1e-5
    ) -> MLXArray {
        // Both downstream paths broadcast GQA natively — the Metal kernel via
        // `kv_head_idx = head_idx / GQA_FACTOR` and MLXFast SDPA via its
        // built-in grouped-query support (identical repeat-interleave head
        // mapping) — so the origin slabs are passed unrepeated instead of
        // materializing groupedRatio× full-sequence copies per layer.
        let attnOrigin = dualValueAttention(
            queries: qOrigin,
            keys: kOrigin,
            value1: value1,
            value2: value2,
            scale: scale,
            mask: mask
        )
        let attnNoise = dualValueAttention(
            queries: qNoise,
            keys: kNoise,
            value1: value1,
            value2: value2,
            scale: scale,
            mask: mask
        )
        return postSplit(
            attnOrigin: attnOrigin,
            attnNoise: attnNoise,
            sublnWeight: sublnWeight,
            lambda: lambda,
            lambdaInit: lambdaInit,
            groupedRatio: groupedRatio,
            eps: eps
        )
    }

    public static func dualValueAttention(
        queries: MLXArray,
        keys: MLXArray,
        value1: MLXArray,
        value2: MLXArray,
        scale: Float,
        mask: MLXFast.ScaledDotProductAttentionMaskMode = .none
    ) -> MLXArray {
        MotifSDPADualV.apply(
            queries: queries,
            keys: keys,
            value1: value1,
            value2: value2,
            scale: scale,
            mask: mask
        )
    }

    public static func postSplit(
        attnOrigin: MLXArray,
        attnNoise: MLXArray,
        sublnWeight: MLXArray,
        lambda: MLXArray,
        lambdaInit: Double,
        groupedRatio: Int,
        eps: Float = 1e-5
    ) -> MLXArray {
        MotifGDAPostSplit.apply(
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

private extension MotifAttentionLayout {
    var requiredNoiseHeads: Int {
        queryHeads / (groupedRatio + 1)
    }

    var requiredKeyNoiseHeads: Int {
        keyNoiseHeads ?? keyValueHeads
    }
}

private extension Array where Element == Int {
    subscript(safe index: Int) -> Int? {
        indices.contains(index) ? self[index] : nil
    }
}
#endif
