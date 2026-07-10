#if canImport(MLX) && canImport(MLXNN) && canImport(MLXLLM) && canImport(MLXLMCommon)
import Foundation
import MLX
import MLXLLM
import MLXLMCommon
import MLXNN
import MotifKit
import Tokenizers

public enum MotifMLXCapabilityLabel: String, Codable, Equatable, Sendable {
    case buildableScaffold = "buildable scaffold"
    case fixtureProvenSemanticParity = "fixture-proven semantic parity"
    case runtimeGeneratedOutput = "runtime-generated output"
    case stillUnavailable = "still unavailable"
}

public struct MotifMLXDecoderLayerPlan: Codable, Equatable, Sendable {
    public var layerIndex: Int
    public var lambdaInit: Double
    public var attentionProjectionShapes: [String: [Int]]
    public var inputLayerNormShape: [Int]
    public var postAttentionLayerNormShape: [Int]
    public var attentionSubLayerNormShape: [Int]
    public var mlpLayout: MotifMLPLayout
    public var cacheKind: MotifKVCacheKind

    public init(
        configuration: MotifModelConfiguration,
        attentionLayout: MotifAttentionLayout,
        layerIndex: Int,
        cacheKind: MotifKVCacheKind
    ) {
        self.layerIndex = layerIndex
        self.lambdaInit = MotifAttentionLayout.lambdaInit(layerIndex: layerIndex)
        self.attentionProjectionShapes = [
            "q_proj": [attentionLayout.qProjectionSize, configuration.hiddenSize],
            "k_proj": [attentionLayout.kProjectionSize, configuration.hiddenSize],
            "v_proj": [attentionLayout.vProjectionSize, configuration.hiddenSize],
            "o_proj": [configuration.hiddenSize, attentionLayout.outputProjectionInputSize],
        ]
        self.inputLayerNormShape = [configuration.hiddenSize]
        self.postAttentionLayerNormShape = [configuration.hiddenSize]
        self.attentionSubLayerNormShape = [2 * attentionLayout.headDim]
        self.mlpLayout = MotifMLPLayout(configuration: configuration)
        self.cacheKind = cacheKind
    }
}

public struct MotifMLXDecoderGraphPlan: Codable, Equatable, Sendable {
    public var capabilityLabels: [MotifMLXCapabilityLabel]
    public var embeddingShape: [Int]
    public var decoderLayerCount: Int
    public var firstDecoderLayer: MotifMLXDecoderLayerPlan
    public var finalNormShape: [Int]
    public var lmHeadShape: [Int]?
    public var tiedEmbeddingLMHead: Bool
    public var backendReadiness: String

    public init(
        configuration: MotifModelConfiguration,
        attentionLayout: MotifAttentionLayout,
        cacheKind: MotifKVCacheKind
    ) {
        self.capabilityLabels = [
            .buildableScaffold,
            .fixtureProvenSemanticParity,
        ]
        self.embeddingShape = [configuration.vocabSize, configuration.hiddenSize]
        self.decoderLayerCount = configuration.numHiddenLayers
        self.firstDecoderLayer = MotifMLXDecoderLayerPlan(
            configuration: configuration,
            attentionLayout: attentionLayout,
            layerIndex: 0,
            cacheKind: cacheKind
        )
        self.finalNormShape = [configuration.hiddenSize]
        self.tiedEmbeddingLMHead = configuration.tieWordEmbeddings
        self.lmHeadShape = configuration.tieWordEmbeddings
            ? nil
            : [configuration.vocabSize, configuration.hiddenSize]
        self.backendReadiness = "reference decoder graph can run through MLX Swift once real Motif weights/tokenizer are loaded; custom Metal and q4 fast paths remain gated by parity/benchmark evidence"
    }
}

private func motifStringOrNumberMap(
    from scaling: MotifRopeScalingConfiguration?
) -> [String: StringOrNumber]? {
    guard let scaling else { return nil }
    var values: [String: StringOrNumber] = [:]
    if let type = scaling.type { values["type"] = .string(type) }
    if let ropeType = scaling.ropeType { values["rope_type"] = .string(ropeType) }
    if let factor = scaling.factor { values["factor"] = .float(Float(factor)) }
    if let original = scaling.originalMaxPositionEmbeddings {
        values["original_max_position_embeddings"] = .int(original)
    }
    if let low = scaling.lowFreqFactor { values["low_freq_factor"] = .float(Float(low)) }
    if let high = scaling.highFreqFactor { values["high_freq_factor"] = .float(Float(high)) }
    return values.isEmpty ? nil : values
}

public final class MotifMLXAttentionScaffold: Module {
    @ModuleInfo(key: "q_proj") public var queryProjection: Linear
    @ModuleInfo(key: "k_proj") public var keyProjection: Linear
    @ModuleInfo(key: "v_proj") public var valueProjection: Linear
    @ModuleInfo(key: "o_proj") public var outputProjection: Linear
    @ModuleInfo(key: "subln") public var subLayerNorm: RMSNorm
    @ParameterInfo(key: "lambda_q1") public var lambdaQ1: MLXArray
    @ParameterInfo(key: "lambda_k1") public var lambdaK1: MLXArray
    @ParameterInfo(key: "lambda_q2") public var lambdaQ2: MLXArray
    @ParameterInfo(key: "lambda_k2") public var lambdaK2: MLXArray

    public let layout: MotifAttentionLayout
    public let lambdaInit: Float
    public let layerIndex: Int
    public let capabilityLabels: [MotifMLXCapabilityLabel] = [.buildableScaffold, .fixtureProvenSemanticParity]

    private let scale: Float
    private let rope: any OffsetLayer
    private let configuration: MotifModelConfiguration
    private let runtimeFeatures: MotifRuntimeFeatureFlags
    private var fusedQueryKeyValueProjection: Linear?
    private var fusedProjectionUsesOriginFirstQueries = false

    public init(
        configuration: MotifModelConfiguration,
        layerIndex: Int,
        runtimeFeatures: MotifRuntimeFeatureFlags = .fromEnvironment()
    ) throws {
        let layout = try MotifAttentionLayout(configuration: configuration)
        self.layout = layout
        self.configuration = configuration
        self.runtimeFeatures = runtimeFeatures
        self.layerIndex = layerIndex
        self.lambdaInit = Float(MotifAttentionLayout.lambdaInit(layerIndex: layerIndex))
        self.scale = pow(Float(layout.headDim), -0.5)
        self.rope = initializeRope(
            dims: layout.headDim,
            base: Float(configuration.ropeTheta),
            traditional: false,
            scalingConfig: motifStringOrNumberMap(from: configuration.ropeScaling),
            maxPositionEmbeddings: configuration.maxPositionEmbeddings
        )
        self._queryProjection.wrappedValue = Linear(
            configuration.hiddenSize,
            layout.qProjectionSize,
            bias: configuration.useBias
        )
        self._keyProjection.wrappedValue = Linear(
            configuration.hiddenSize,
            layout.kProjectionSize,
            bias: configuration.useBias
        )
        self._valueProjection.wrappedValue = Linear(
            configuration.hiddenSize,
            layout.vProjectionSize,
            bias: configuration.useBias
        )
        self._outputProjection.wrappedValue = Linear(
            layout.outputProjectionInputSize,
            configuration.hiddenSize,
            bias: configuration.useBias
        )
        self._subLayerNorm.wrappedValue = RMSNorm(
            dimensions: 2 * layout.headDim,
            eps: Float(configuration.attnRMSNormEps)
        )
        self._lambdaQ1.wrappedValue = MLXArray.zeros([layout.headDim], dtype: .float32)
        self._lambdaK1.wrappedValue = MLXArray.zeros([layout.headDim], dtype: .float32)
        self._lambdaQ2.wrappedValue = MLXArray.zeros([layout.headDim], dtype: .float32)
        self._lambdaK2.wrappedValue = MLXArray.zeros([layout.headDim], dtype: .float32)
    }

    /// Motif attention path in MLX Swift. Decode-time grouped attention now
    /// prefers the direct custom Metal kernels, while reference paths remain
    /// available through `MLX_MOTIF_DISABLE_KERNELS=1`.
    public func callAsFunction(
        _ x: MLXArray,
        mask: MLXFast.ScaledDotProductAttentionMaskMode,
        cache: KVCache?
    ) -> MLXArray {
        switch layout.variant {
        case .groupedDifferential:
            return forwardGrouped(x, mask: mask, cache: cache)
        case .vanillaDifferential:
            return forwardVanilla(x, mask: mask, cache: cache)
        }
    }

    /// Reference box for the materialized fp32 lambda scalar. Held as a plain
    /// Swift class (not an `MLXArray`/`Module`) so MLXNN parameter reflection
    /// never treats the cache as a model parameter.
    private final class LambdaScalarCache {
        var fp32: MLXArray?
    }

    private let lambdaScalarCache = LambdaScalarCache()

    /// Lambda scalar for the differential subtraction, computed in fp32 to match
    /// Python's `_lambda_full(mx.float32)` — the `exp`/`sub` is numerically
    /// sensitive, so lambda is kept in fp32 before any downcast (see model.py:255
    /// "lambda parameters — kept fp32 for numerical stability of exp/sub").
    ///
    /// `lambdaQ1/K1/Q2/K2` are frozen at inference, so the fp32 scalar is
    /// materialized once and cached; callers requesting another dtype get a cheap
    /// cast of the cached value instead of rebuilding the exp/sum graph per layer
    /// per decoded token. The cache is only valid while the lambda parameters do
    /// not change (inference / `training == false`).
    func lambdaFull(dtype: DType) -> MLXArray {
        let base = cachedLambdaFullFP32()
        return dtype == .float32 ? base : base.asType(dtype)
    }

    private func cachedLambdaFullFP32() -> MLXArray {
        if let cached = lambdaScalarCache.fp32 {
            return cached
        }
        let l1 = exp(sum(lambdaQ1 * lambdaK1, axis: -1).asType(.float32))
        let l2 = exp(sum(lambdaQ2 * lambdaK2, axis: -1).asType(.float32))
        let value = (l1 - l2 + lambdaInit).asType(.float32)
        // Materialize once so subsequent forwards reference stored data rather
        // than re-executing the reduction/exp kernels per decoded token.
        eval(value)
        lambdaScalarCache.fp32 = value
        return value
    }

    private func forwardGrouped(
        _ x: MLXArray,
        mask: MLXFast.ScaledDotProductAttentionMaskMode,
        cache: KVCache?
    ) -> MLXArray {
        let batch = x.dim(0)
        let sequenceLength = x.dim(1)
        let keyGroups = layout.keyNoiseHeads ?? layout.keyValueHeads
        let offset = cache?.offset ?? 0
        let cacheKind = groupedCacheKind(cache)
        let attentionPath = MotifAttentionPathResolver.resolve(
            cacheKind: cacheKind,
            sequenceLength: sequenceLength,
            keyValueRepeat: layout.keyValueRepeat,
            keyRatio: layout.keyRatio,
            fusedRope: configuration.fusedRope,
            featureFlags: runtimeFeatures
        )

        let qFlat: MLXArray
        let kFlat: MLXArray
        let vFlat: MLXArray
        let qOriginFirst: Bool
        if let fusedQueryKeyValueProjection {
            let qkv = fusedQueryKeyValueProjection(x)
            qFlat = qkv[0..., 0..., ..<layout.qProjectionSize]
            kFlat = qkv[
                0...,
                0...,
                layout.qProjectionSize ..< (layout.qProjectionSize + layout.kProjectionSize)
            ]
            vFlat = qkv[0..., 0..., (layout.qProjectionSize + layout.kProjectionSize)...]
            qOriginFirst = fusedProjectionUsesOriginFirstQueries
        } else {
            qFlat = queryProjection(x)
            kFlat = keyProjection(x)
            vFlat = valueProjection(x)
            qOriginFirst = false
        }

        var q = qFlat
            .reshaped([batch, sequenceLength, layout.queryHeads, layout.headDim])
            .transposed(0, 2, 1, 3)
        var k = kFlat
            .reshaped([batch, sequenceLength, layout.keyValueHeads, layout.headDim])
            .transposed(0, 2, 1, 3)
        var v = vFlat
            .reshaped([batch, sequenceLength, 2 * keyGroups, layout.headDim])
            .transposed(0, 2, 1, 3)

        q = rope(q, offset: offset)
        let slices: MotifGroupedProjectionSlices
        var quantizedSlices: (
            kOrigin: MotifQuantizedTuple,
            kNoise: MotifQuantizedTuple,
            value1: MotifQuantizedTuple,
            value2: MotifQuantizedTuple
        )?

        if let grouped = cache as? MotifGroupedQuantizedKVCache {
            let prepared = try! MotifGroupedDifferentialAttentionReference.splitPreparedTensors(
                q: q,
                k: k,
                v: v,
                layout: layout,
                qOriginFirst: qOriginFirst
            )
            let kOrigin = rope(prepared.kOrigin, offset: offset)
            let kNoise = rope(prepared.kNoise, offset: offset)
            if attentionPath == .quantizedSDPA {
                let packed = grouped.updateAndFetch4Quantized(
                    kOrigin: kOrigin,
                    kNoise: kNoise,
                    value1: prepared.value1,
                    value2: prepared.value2
                )
                quantizedSlices = packed
                slices = MotifGroupedProjectionSlices(
                    qOrigin: prepared.qOrigin,
                    qNoise: prepared.qNoise,
                    // In the optimized q4 route the cached K/V slabs stay packed
                    // and are consumed directly by `MotifSDPADualVQ4.apply`.
                    // These per-token tensors are not used by that branch; keeping
                    // them here avoids materializing a full dequantized cache.
                    kOrigin: kOrigin,
                    kNoise: kNoise,
                    value1: prepared.value1,
                    value2: prepared.value2
                )
            } else {
                let cached = grouped.updateAndFetch4(
                    kOrigin: kOrigin,
                    kNoise: kNoise,
                    value1: prepared.value1,
                    value2: prepared.value2
                )
                slices = MotifGroupedProjectionSlices(
                    qOrigin: prepared.qOrigin,
                    qNoise: prepared.qNoise,
                    kOrigin: cached.kOrigin,
                    kNoise: cached.kNoise,
                    value1: cached.value1,
                    value2: cached.value2
                )
            }
        } else if let grouped = cache as? MotifGroupedKVCache {
            let prepared = try! MotifGroupedDifferentialAttentionReference.splitPreparedTensors(
                q: q,
                k: k,
                v: v,
                layout: layout,
                qOriginFirst: qOriginFirst
            )
            let cached = grouped.updateAndFetch4(
                kOrigin: rope(prepared.kOrigin, offset: offset),
                kNoise: rope(prepared.kNoise, offset: offset),
                value1: prepared.value1,
                value2: prepared.value2
            )
            slices = MotifGroupedProjectionSlices(
                qOrigin: prepared.qOrigin,
                qNoise: prepared.qNoise,
                kOrigin: cached.kOrigin,
                kNoise: cached.kNoise,
                value1: cached.value1,
                value2: cached.value2
            )
        } else {
            k = rope(k, offset: offset)
            if let cache {
                (k, v) = cache.update(keys: k, values: v)
            }
            slices = try! MotifGroupedDifferentialAttentionReference.splitPreparedTensors(
                q: q,
                k: k,
                v: v,
                layout: layout,
                qOriginFirst: qOriginFirst
            )
        }

        // Match Python's grouped path (`self._lambda_full(mx.float32)`, model.py:517):
        // keep lambda in fp32 into the kernels. Casting to the activation dtype
        // here would round lambda before the wrappers upcast it back to fp32,
        // losing mantissa bits and diverging from Python in every grouped layer.
        let lambda = lambdaFull(dtype: .float32).reshaped([1])
        let kernelExecutionMode: MotifMetalKernelExecutionMode =
            runtimeFeatures.disableCustomKernels ? .referenceOnly : .metalPreferred
        let out: MLXArray
        if attentionPath == .quantizedSDPA, let quantizedSlices, let grouped = cache as? MotifGroupedQuantizedKVCache {
            let attnOrigin = MotifSDPADualVQ4.apply(
                queries: slices.qOrigin,
                quantizedKeys: quantizedSlices.kOrigin,
                quantizedValue1: quantizedSlices.value1,
                quantizedValue2: quantizedSlices.value2,
                scale: scale,
                groupSize: grouped.groupSize,
                bits: grouped.bits,
                mode: grouped.mode,
                dtype: x.dtype,
                executionMode: kernelExecutionMode
            )
            let attnNoise = MotifSDPADualVQ4.apply(
                queries: slices.qNoise,
                quantizedKeys: quantizedSlices.kNoise,
                quantizedValue1: quantizedSlices.value1,
                quantizedValue2: quantizedSlices.value2,
                scale: scale,
                groupSize: grouped.groupSize,
                bits: grouped.bits,
                mode: grouped.mode,
                dtype: x.dtype,
                executionMode: kernelExecutionMode
            )
            out = MotifGDAPostSplit.apply(
                attnOrigin: attnOrigin,
                attnNoise: attnNoise,
                sublnWeight: subLayerNorm.weight,
                lambda: lambda,
                lambdaInit: Double(lambdaInit),
                groupedRatio: layout.groupedRatio,
                eps: Float(configuration.attnRMSNormEps),
                executionMode: kernelExecutionMode
            )
        } else if attentionPath == .dualV {
            // The kernel broadcasts GQA natively (`kv_head_idx = head_idx /
            // GQA_FACTOR`, i.e. repeat-interleave semantics), matching Python
            // model.py's `sdpa_dual_v(q1, k1, v1, v2, scale)`: pass the cached
            // slabs unrepeated instead of materializing groupedRatio× copies
            // of the full KV history per layer per decoded token.
            let attnOrigin = MotifSDPADualV.apply(
                queries: slices.qOrigin,
                keys: slices.kOrigin,
                value1: slices.value1,
                value2: slices.value2,
                scale: scale,
                mask: mask,
                executionMode: kernelExecutionMode
            )
            let attnNoise = MotifSDPADualV.apply(
                queries: slices.qNoise,
                keys: slices.kNoise,
                value1: slices.value1,
                value2: slices.value2,
                scale: scale,
                mask: mask,
                executionMode: kernelExecutionMode
            )
            out = MotifGDAPostSplit.apply(
                attnOrigin: attnOrigin,
                attnNoise: attnNoise,
                sublnWeight: subLayerNorm.weight,
                lambda: lambda,
                lambdaInit: Double(lambdaInit),
                groupedRatio: layout.groupedRatio,
                eps: Float(configuration.attnRMSNormEps),
                executionMode: kernelExecutionMode
            )
        } else {
            out = MotifGroupedDifferentialAttentionReference.apply(
                qOrigin: slices.qOrigin,
                qNoise: slices.qNoise,
                kOrigin: slices.kOrigin,
                kNoise: slices.kNoise,
                value1: slices.value1,
                value2: slices.value2,
                sublnWeight: subLayerNorm.weight,
                lambda: lambda,
                lambdaInit: Double(lambdaInit),
                groupedRatio: layout.groupedRatio,
                scale: scale,
                mask: mask,
                eps: Float(configuration.attnRMSNormEps)
            )
        }
        return outputProjection(out.transposed(0, 2, 1, 3).reshaped([batch, sequenceLength, -1]))
    }

    private func groupedCacheKind(_ cache: KVCache?) -> MotifKVCacheKind {
        switch cache {
        case is MotifGroupedQuantizedKVCache:
            if let quantized = cache as? MotifGroupedQuantizedKVCache {
                return quantized.bits == 8 ? .groupedQuantized8Bit : .groupedQuantized4Bit
            }
            return .groupedQuantized4Bit
        case is MotifGroupedKVCache:
            return .groupedFourSlot
        default:
            return .standard
        }
    }

    // V-cache head-ordering invariant (Swift vs Python divergence):
    //
    // Unlike the Python reference (`_forward_vanilla` in src/mlx_motif/model.py),
    // this Swift path stores V in the KV cache in the projection's *paired* head
    // order [v0_a, v0_b, v1_a, v1_b, ...] — the raw `valueProjection` output is
    // written straight to `cache.update`. The reshape → transpose → reshape into
    // a 2*headDim-wide per-KV-head V happens AFTER fetch, on the full cached V.
    // So the Swift vanilla V cache is HF-ordered and does NOT carry the Python
    // side's *slab* head-order trap; no ordering marker is needed here, and the
    // `KVCache` type is correct as-is.
    //
    // DIRECTIVE: if this path is ever reworked to the Python-style per-slab SDPA
    // (which stores V slab-ordered [va_0..va_{Hk-1}, vb_0..vb_{Hk-1}] at
    // projection time), mirror the Python `MotifVanillaKVCache` marker/accessor
    // here so the non-standard V ordering cannot leak silently.
    private func forwardVanilla(
        _ x: MLXArray,
        mask: MLXFast.ScaledDotProductAttentionMaskMode,
        cache: KVCache?
    ) -> MLXArray {
        let batch = x.dim(0)
        let sequenceLength = x.dim(1)
        let effectiveHeads = layout.queryHeads
        let effectiveKVHeads = layout.keyValueHeads
        let offset = cache?.offset ?? 0

        var q = queryProjection(x)
            .reshaped([batch, sequenceLength, 2 * effectiveHeads, layout.headDim])
            .transposed(0, 2, 1, 3)
        var k = keyProjection(x)
            .reshaped([batch, sequenceLength, 2 * effectiveKVHeads, layout.headDim])
            .transposed(0, 2, 1, 3)
        var v = valueProjection(x)
            .reshaped([batch, sequenceLength, 2 * effectiveKVHeads, layout.headDim])
            .transposed(0, 2, 1, 3)

        q = rope(q, offset: offset)
        k = rope(k, offset: offset)
        if let cache {
            (k, v) = cache.update(keys: k, values: v)
        }
        let kvSequenceLength = k.dim(2)
        v = v.reshaped([batch, effectiveKVHeads, 2, kvSequenceLength, layout.headDim])
            .transposed(0, 1, 3, 2, 4)
            .reshaped([batch, effectiveKVHeads, kvSequenceLength, 2 * layout.headDim])
        v = repeatHeadsIfNeeded(v, count: layout.keyValueRepeat)

        let qPairs = q.reshaped([batch, effectiveHeads, 2, sequenceLength, layout.headDim])
        let kPairs = k.reshaped([batch, effectiveKVHeads, 2, kvSequenceLength, layout.headDim])
        let q1 = qPairs[0..., 0..., 0, 0..., 0...]
        let q2 = qPairs[0..., 0..., 1, 0..., 0...]
        let k1 = repeatHeadsIfNeeded(kPairs[0..., 0..., 0, 0..., 0...], count: layout.keyValueRepeat)
        let k2 = repeatHeadsIfNeeded(kPairs[0..., 0..., 1, 0..., 0...], count: layout.keyValueRepeat)

        let out1 = MLXFast.scaledDotProductAttention(
            queries: q1,
            keys: k1,
            values: v,
            scale: scale,
            mask: mask
        )
        let out2 = MLXFast.scaledDotProductAttention(
            queries: q2,
            keys: k2,
            values: v,
            scale: scale,
            mask: mask
        )
        let differential = out1 - lambdaFull(dtype: out1.dtype) * out2
        let normalized = MLXFast.rmsNorm(
            differential,
            weight: subLayerNorm.weight.asType(differential.dtype),
            eps: Float(configuration.attnRMSNormEps)
        ) * Float(1.0 - lambdaInit)
        return outputProjection(
            normalized.transposed(0, 2, 1, 3).reshaped([batch, sequenceLength, -1])
        )
    }

    private func repeatHeadsIfNeeded(_ x: MLXArray, count: Int) -> MLXArray {
        count == 1 ? x : repeated(x, count: count, axis: 1)
    }

    @discardableResult
    public func fuseQueryKeyValueProjectionIfPossible() -> Bool {
        guard runtimeFeatures.fuseQueryKeyValue,
              layout.variant == .groupedDifferential,
              fusedQueryKeyValueProjection == nil
        else {
            return false
        }
        guard !configuration.useBias,
              queryProjection.bias == nil,
              keyProjection.bias == nil,
              valueProjection.bias == nil
        else {
            return false
        }

        let permutation = Self.originFirstQueryPermutation(
            noiseGroups: layout.queryHeads / (layout.groupedRatio + 1),
            groupedRatio: layout.groupedRatio,
            headDim: layout.headDim
        )

        if let query = queryProjection as? QuantizedLinear {
            guard let key = keyProjection as? QuantizedLinear,
                  let value = valueProjection as? QuantizedLinear,
                  query.bits == key.bits,
                  query.bits == value.bits,
                  query.groupSize == key.groupSize,
                  query.groupSize == value.groupSize,
                  query.mode == key.mode,
                  query.mode == value.mode
            else {
                return false
            }
            let qWeight = take(query.weight, permutation, axis: 0)
            let qScales = take(query.scales, permutation, axis: 0)
            let qBiases = query.biases.map { take($0, permutation, axis: 0) }
            fusedQueryKeyValueProjection = QuantizedLinear(
                weight: concatenated([qWeight, key.weight, value.weight], axis: 0),
                bias: nil,
                scales: concatenated([qScales, key.scales, value.scales], axis: 0),
                biases: concatenateOptional([qBiases, key.biases, value.biases], axis: 0),
                groupSize: query.groupSize,
                bits: query.bits,
                mode: query.mode
            )
        } else {
            guard !(keyProjection is QuantizedLinear), !(valueProjection is QuantizedLinear) else {
                return false
            }
            fusedQueryKeyValueProjection = Linear(
                weight: concatenated([
                    take(queryProjection.weight, permutation, axis: 0),
                    keyProjection.weight,
                    valueProjection.weight,
                ], axis: 0),
                bias: nil
            )
        }

        fusedProjectionUsesOriginFirstQueries = true
        if let fused = fusedQueryKeyValueProjection as? QuantizedLinear {
            eval([fused.weight, fused.scales] + [fused.biases].compactMap { $0 })
        } else if let fusedQueryKeyValueProjection {
            eval(fusedQueryKeyValueProjection.weight)
        }

        // Release the original q/k/v projection weights now that the fused copy
        // is materialized. Fusion is gated to `.groupedDifferential`, whose
        // `callAsFunction` always takes the fused branch, so the originals are
        // dead weight — keeping them doubles resident attention-projection
        // memory for the model's lifetime. Replacing them with empty-weight
        // placeholders drops the last reference to the full buffers. Mirrors the
        // Python reference, which sets q_proj/k_proj/v_proj = Identity() after
        // fusing (src/mlx_motif/model.py fuse_qkv()). @ModuleInfo forbids direct
        // property mutation after init, so route through update(modules:).
        let released: [(String, Module)] = [
            ("q_proj", Linear(weight: MLXArray.zeros([1, 1]), bias: nil)),
            ("k_proj", Linear(weight: MLXArray.zeros([1, 1]), bias: nil)),
            ("v_proj", Linear(weight: MLXArray.zeros([1, 1]), bias: nil)),
        ]
        update(modules: ModuleChildren.unflattened(released))
        return true
    }

    private static func originFirstQueryPermutation(noiseGroups: Int, groupedRatio: Int, headDim: Int) -> MLXArray {
        var rows: [Int32] = []
        rows.reserveCapacity(noiseGroups * (groupedRatio + 1) * headDim)
        for group in 0 ..< noiseGroups {
            for origin in 0 ..< groupedRatio {
                let head = group * (groupedRatio + 1) + origin
                for row in 0 ..< headDim {
                    rows.append(Int32(head * headDim + row))
                }
            }
        }
        for group in 0 ..< noiseGroups {
            let head = group * (groupedRatio + 1) + groupedRatio
            for row in 0 ..< headDim {
                rows.append(Int32(head * headDim + row))
            }
        }
        return MLXArray(rows, [rows.count])
    }

    private func concatenateOptional(_ arrays: [MLXArray?], axis: Int) -> MLXArray? {
        let present = arrays.compactMap { $0 }
        guard present.count == arrays.count else {
            return nil
        }
        return concatenated(present, axis: axis)
    }
}

public final class MotifMLXDecoderLayer: Module {
    @ModuleInfo(key: "self_attn") public var attention: MotifMLXAttentionScaffold
    @ModuleInfo(key: "mlp") public var mlp: MotifMLXMLP
    @ModuleInfo(key: "input_layernorm") public var inputLayerNorm: MotifMLXRMSNorm
    @ModuleInfo(key: "post_attention_layernorm") public var postAttentionLayerNorm: MotifMLXRMSNorm

    public let layerIndex: Int

    public init(
        configuration: MotifModelConfiguration,
        layerIndex: Int,
        runtimeFeatures: MotifRuntimeFeatureFlags = .fromEnvironment()
    ) throws {
        self.layerIndex = layerIndex
        self._attention.wrappedValue = try MotifMLXAttentionScaffold(
            configuration: configuration,
            layerIndex: layerIndex,
            runtimeFeatures: runtimeFeatures
        )
        self._mlp.wrappedValue = try MotifMLXMLP(configuration: configuration)
        self._inputLayerNorm.wrappedValue = MotifMLXRMSNorm(configuration: configuration)
        self._postAttentionLayerNorm.wrappedValue = MotifMLXRMSNorm(configuration: configuration)
    }

    public func callAsFunction(
        _ x: MLXArray,
        mask: MLXFast.ScaledDotProductAttentionMaskMode,
        cache: KVCache?
    ) -> MLXArray {
        let h = x + attention(inputLayerNorm(x), mask: mask, cache: cache)
        return h + mlp(postAttentionLayerNorm(h))
    }
}

public final class MotifMLXModelInner: Module {
    @ModuleInfo(key: "embed_tokens") public var embedTokens: Embedding
    @ModuleInfo(key: "norm") public var norm: MotifMLXRMSNorm

    public let configuration: MotifModelConfiguration
    public let runtimeFeatures: MotifRuntimeFeatureFlags
    public let layers: [MotifMLXDecoderLayer]

    public init(
        configuration: MotifModelConfiguration,
        runtimeFeatures: MotifRuntimeFeatureFlags = .fromEnvironment()
    ) throws {
        self.configuration = configuration
        self.runtimeFeatures = runtimeFeatures
        self._embedTokens.wrappedValue = Embedding(
            embeddingCount: configuration.vocabSize,
            dimensions: configuration.hiddenSize
        )
        var decoderLayers: [MotifMLXDecoderLayer] = []
        decoderLayers.reserveCapacity(configuration.numHiddenLayers)
        for index in 0 ..< configuration.numHiddenLayers {
            decoderLayers.append(try MotifMLXDecoderLayer(
                configuration: configuration,
                layerIndex: index,
                runtimeFeatures: runtimeFeatures
            ))
        }
        self.layers = decoderLayers
        self._norm.wrappedValue = MotifMLXRMSNorm(configuration: configuration)
    }

    public func callAsFunction(_ inputs: MLXArray, cache: [KVCache]? = nil) -> MLXArray {
        var hidden = embedTokens(inputs)
        let mask = createAttentionMask(h: hidden, cache: cache?.first)
        for (index, layer) in layers.enumerated() {
            let layerCache = cache.flatMap { index < $0.count ? $0[index] : nil }
            hidden = layer(hidden, mask: mask, cache: layerCache)
        }
        return norm(hidden)
    }
}

public final class MotifMLXModel: Module, LLMModel, KVCacheDimensionProvider {
    public let vocabularySize: Int
    public let kvHeads: [Int]
    public let configuration: MotifModelConfiguration
    public let runtimeFeatures: MotifRuntimeFeatureFlags
    public let graphPlan: MotifMLXDecoderGraphPlan
    public let model: MotifMLXModelInner

    @ModuleInfo(key: "lm_head") public var lmHead: Linear?

    public init(
        configuration: MotifModelConfiguration,
        runtimeFeatures: MotifRuntimeFeatureFlags = .fromEnvironment()
    ) throws {
        self.configuration = configuration
        self.runtimeFeatures = runtimeFeatures
        self.vocabularySize = configuration.vocabSize
        self.kvHeads = (0 ..< configuration.numHiddenLayers).map { _ in configuration.numKeyValueHeads }
        let attentionLayout = try MotifAttentionLayout(configuration: configuration)
        self.graphPlan = MotifMLXDecoderGraphPlan(
            configuration: configuration,
            attentionLayout: attentionLayout,
            cacheKind: configuration.isGroupedDifferentialAttention ? .groupedFourSlot : .standard
        )
        self.model = try MotifMLXModelInner(configuration: configuration, runtimeFeatures: runtimeFeatures)
        if !configuration.tieWordEmbeddings {
            self._lmHead.wrappedValue = Linear(
                configuration.hiddenSize,
                configuration.vocabSize,
                bias: false
            )
        }
    }

    public var loraLayers: [Module] {
        model.layers
    }

    public func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        let hidden = model(inputs, cache: cache)
        if configuration.tieWordEmbeddings {
            return model.embedTokens.asLinear(hidden)
        }
        if let lmHead {
            return lmHead(hidden)
        }
        fatalError("MotifMLXModel missing lm_head for untied embeddings")
    }

    public func newCache(parameters: GenerateParameters?) -> [KVCache] {
        guard configuration.isGroupedDifferentialAttention else {
            return Self.defaultCaches(count: configuration.numHiddenLayers, parameters: parameters)
        }
        // Fail-fast: the 4-slot grouped caches (MotifGroupedKVCache /
        // MotifGroupedQuantizedKVCache) allocate all four slots with a single
        // head count taken from kOrigin, but with kRatio > 1 the kOrigin slot
        // has keyGroups*kRatio heads while kNoise/value1/value2 have keyGroups
        // heads — the mismatched slot writes fail (or silently corrupt)
        // mid-forward at cache-write time. No shipped config uses kRatio > 1, so
        // guard the unsupported combination loudly here. (Python mirror:
        // Model.make_cache.)
        if let reason = Self.fourSlotCacheUnsupportedReason(
            mode: runtimeFeatures.fourSlotCacheMode,
            kRatio: configuration.kRatio
        ) {
            preconditionFailure(reason)
        }
        switch runtimeFeatures.fourSlotCacheMode {
        case .disabled:
            return Self.defaultCaches(count: configuration.numHiddenLayers, parameters: parameters)
        case .fp:
            return (0 ..< configuration.numHiddenLayers).map { _ in MotifGroupedKVCache() }
        case .q4:
            return (0 ..< configuration.numHiddenLayers).map { _ in
                MotifGroupedQuantizedKVCache(groupSize: parameters?.kvGroupSize ?? 64, bits: 4)
            }
        case .q8:
            return (0 ..< configuration.numHiddenLayers).map { _ in
                MotifGroupedQuantizedKVCache(groupSize: parameters?.kvGroupSize ?? 64, bits: 8)
            }
        }
    }

    /// Returns a human-readable reason when the requested four-slot cache
    /// configuration is unsupported, or `nil` when it is fine. Exposed
    /// (non-crashing) so the fail-fast guard in `newCache` is unit-testable.
    ///
    /// The four-slot grouped caches allocate every slot with kOrigin's head
    /// count; with `kRatio > 1` the kOrigin slot has `keyGroups*kRatio` heads
    /// while kNoise/value1/value2 have `keyGroups` heads, so the combination is
    /// unsupported. (Python mirror: `Model.make_cache`.)
    static func fourSlotCacheUnsupportedReason(
        mode: MotifRuntimeFeatureFlags.FourSlotCacheMode,
        kRatio: Int
    ) -> String? {
        guard mode != .disabled, kRatio != 1 else { return nil }
        return "Four-slot grouped KV cache is not supported with kRatio > 1 "
            + "(got kRatio=\(kRatio)): every slot is allocated with kOrigin's head "
            + "count, which differs from kNoise/value1/value2 when kRatio > 1. "
            + "Disable the four-slot cache for this configuration."
    }

    private static func defaultCaches(count: Int, parameters: GenerateParameters?) -> [KVCache] {
        if let maxKVSize = parameters?.maxKVSize {
            return (0 ..< count).map { _ in RotatingKVCache(maxSize: maxKVSize, keep: 4) }
        }
        if let bits = parameters?.kvBits {
            return (0 ..< count).map { _ in
                QuantizedKVCache(groupSize: parameters?.kvGroupSize ?? 64, bits: bits)
            }
        }
        return (0 ..< count).map { _ in KVCacheSimple() }
    }

    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        weights.filter {
            !$0.key.contains("rotary_emb.inv_freq") && !$0.key.contains("rope.inv_freq")
        }
    }

    @discardableResult
    public func fuseQueryKeyValueProjectionsIfPossible() -> Int {
        model.layers.reduce(0) { count, layer in
            count + (layer.attention.fuseQueryKeyValueProjectionIfPossible() ? 1 : 0)
        }
    }

    public func messageGenerator(tokenizer: any Tokenizer) -> any MessageGenerator {
        DefaultMessageGenerator()
    }
}
#endif
