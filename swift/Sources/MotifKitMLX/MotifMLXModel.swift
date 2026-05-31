#if canImport(MLX) && canImport(MLXNN) && canImport(MLXLLM) && canImport(MLXLMCommon)
import Foundation
import MLX
import MLXLLM
import MLXLMCommon
import MLXNN
import MotifKit

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
        self.capabilityLabels = [.buildableScaffold, .stillUnavailable]
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
        self.backendReadiness = "buildable scaffold; MotifMLXBackend still unavailable until tokenizer/load and attention runtime emit verified tokens"
    }
}

public final class MotifMLXAttentionScaffold: Module {
    @ModuleInfo(key: "q_proj") public var queryProjection: Linear
    @ModuleInfo(key: "k_proj") public var keyProjection: Linear
    @ModuleInfo(key: "v_proj") public var valueProjection: Linear
    @ModuleInfo(key: "o_proj") public var outputProjection: Linear
    @ModuleInfo(key: "subln") public var subLayerNorm: RMSNorm

    public let layout: MotifAttentionLayout
    public let lambdaInit: Float
    public let layerIndex: Int
    public let capabilityLabels: [MotifMLXCapabilityLabel] = [.buildableScaffold, .stillUnavailable]

    public init(configuration: MotifModelConfiguration, layerIndex: Int) throws {
        let layout = try MotifAttentionLayout(configuration: configuration)
        self.layout = layout
        self.layerIndex = layerIndex
        self.lambdaInit = Float(MotifAttentionLayout.lambdaInit(layerIndex: layerIndex))
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
    }

    /// Buildable decoder-graph placeholder. It wires all Motif attention
    /// parameters into the module tree and validates projection shapes without
    /// claiming runtime-generated output. The backend remains unavailable until
    /// this method is replaced by the full differential-attention implementation.
    public func callAsFunction(
        _ x: MLXArray,
        mask _: MLXFast.ScaledDotProductAttentionMaskMode,
        cache _: KVCache?
    ) -> MLXArray {
        let qFlat = queryProjection(x)
        _ = keyProjection(x)
        _ = valueProjection(x)

        let batch = qFlat.dim(0)
        let sequenceLength = qFlat.dim(1)
        let attentionOutput = zeros(
            [batch, sequenceLength, layout.outputProjectionInputSize],
            dtype: x.dtype
        )
        return outputProjection(attentionOutput)
    }
}

public final class MotifMLXDecoderLayer: Module {
    @ModuleInfo(key: "self_attn") public var attention: MotifMLXAttentionScaffold
    @ModuleInfo(key: "mlp") public var mlp: MotifMLXMLP
    @ModuleInfo(key: "input_layernorm") public var inputLayerNorm: MotifMLXRMSNorm
    @ModuleInfo(key: "post_attention_layernorm") public var postAttentionLayerNorm: MotifMLXRMSNorm

    public let layerIndex: Int

    public init(configuration: MotifModelConfiguration, layerIndex: Int) throws {
        self.layerIndex = layerIndex
        self._attention.wrappedValue = try MotifMLXAttentionScaffold(
            configuration: configuration,
            layerIndex: layerIndex
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
    public let layers: [MotifMLXDecoderLayer]

    public init(configuration: MotifModelConfiguration) throws {
        self.configuration = configuration
        self._embedTokens.wrappedValue = Embedding(
            embeddingCount: configuration.vocabSize,
            dimensions: configuration.hiddenSize
        )
        var decoderLayers: [MotifMLXDecoderLayer] = []
        decoderLayers.reserveCapacity(configuration.numHiddenLayers)
        for index in 0 ..< configuration.numHiddenLayers {
            decoderLayers.append(try MotifMLXDecoderLayer(configuration: configuration, layerIndex: index))
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

public final class MotifMLXModel: Module, KVCacheDimensionProvider {
    public let vocabularySize: Int
    public let kvHeads: [Int]
    public let configuration: MotifModelConfiguration
    public let graphPlan: MotifMLXDecoderGraphPlan
    public let model: MotifMLXModelInner

    @ModuleInfo(key: "lm_head") public var lmHead: Linear?

    public init(configuration: MotifModelConfiguration) throws {
        self.configuration = configuration
        self.vocabularySize = configuration.vocabSize
        self.kvHeads = (0 ..< configuration.numHiddenLayers).map { _ in configuration.numKeyValueHeads }
        let attentionLayout = try MotifAttentionLayout(configuration: configuration)
        self.graphPlan = MotifMLXDecoderGraphPlan(
            configuration: configuration,
            attentionLayout: attentionLayout,
            cacheKind: configuration.isGroupedDifferentialAttention ? .groupedFourSlot : .standard
        )
        self.model = try MotifMLXModelInner(configuration: configuration)
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

    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        weights.filter {
            !$0.key.contains("rotary_emb.inv_freq") && !$0.key.contains("rope.inv_freq")
        }
    }
}
#endif
