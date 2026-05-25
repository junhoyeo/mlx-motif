#if canImport(MLX) && canImport(MLXNN) && canImport(MLXLLM) && canImport(MLXLMCommon)
import Foundation
import MLX
import MLXLLM
import MLXLMCommon
import MLXNN
import MotifKit

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
            "reference-only MLX Swift ops; no custom Metal dispatch by default",
            "expects q/k/v tensors after projection and RoPE, not a full decoder layer",
            "quantized grouped cache is a parity hook only until sdpa_dual_v_q4 is ported",
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
/// fp16/bf16 reference storage; quantized cache support stays represented by
/// `MotifGroupedAttentionReferencePlan.parityHooks` until packed q4/q8 kernels
/// have Swift fixtures.
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
        let attnOrigin = dualValueAttention(
            queries: qOrigin,
            keys: repeatHeadsIfNeeded(kOrigin, count: groupedRatio),
            value1: repeatHeadsIfNeeded(value1, count: groupedRatio),
            value2: repeatHeadsIfNeeded(value2, count: groupedRatio),
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
        let values = concatenated([value1, value2], axis: -1)
        return MLXFast.scaledDotProductAttention(
            queries: queries,
            keys: keys,
            values: values,
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
        let noise = repeatHeadsIfNeeded(attnNoise, count: groupedRatio)
        let differential = attnOrigin - lambda.asType(attnOrigin.dtype) * noise
        let normalized = MLXFast.rmsNorm(
            differential,
            weight: sublnWeight.asType(differential.dtype),
            eps: eps
        )
        return normalized * Float(1.0 - lambdaInit)
    }

    private static func repeatHeadsIfNeeded(_ x: MLXArray, count: Int) -> MLXArray {
        count == 1 ? x : repeated(x, count: count, axis: 1)
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
