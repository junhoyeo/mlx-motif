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

    private func appendCached(_ cached: MLXArray?, _ next: MLXArray) -> MLXArray {
        guard let cached else { return next }
        return concatenated([cached, next], axis: 2)
    }
}

public enum MotifGroupedDifferentialAttentionReference {
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
