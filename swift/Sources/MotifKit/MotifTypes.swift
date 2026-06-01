import Foundation

public enum MotifRole: String, Codable, Sendable, CaseIterable {
    case system
    case user
    case assistant
}

public struct MotifChatMessage: Identifiable, Codable, Equatable, Sendable {
    public var id: UUID
    public var role: MotifRole
    public var content: String
    public var createdAt: Date

    public init(
        id: UUID = UUID(),
        role: MotifRole,
        content: String,
        createdAt: Date = Date()
    ) {
        self.id = id
        self.role = role
        self.content = content
        self.createdAt = createdAt
    }

    public static func system(_ content: String) -> Self { .init(role: .system, content: content) }
    public static func user(_ content: String) -> Self { .init(role: .user, content: content) }
    public static func assistant(_ content: String) -> Self { .init(role: .assistant, content: content) }
}

public enum MotifThinkMode: String, Codable, Sendable, CaseIterable {
    case visible
    case hidden
    case captured
}

public struct MotifGenerationParameters: Equatable, Sendable {
    public var model: String
    public var maxTokens: Int
    public var temperature: Double
    public var thinkMode: MotifThinkMode

    public init(
        model: String = "motif",
        maxTokens: Int = 512,
        temperature: Double = 0.6,
        thinkMode: MotifThinkMode = .hidden
    ) {
        self.model = model
        self.maxTokens = maxTokens
        self.temperature = temperature
        self.thinkMode = thinkMode
    }
}

/// Token-usage accounting surfaced on the terminal generation event.
///
/// Mirrors the OpenAI `usage` object (`prompt_tokens` / `completion_tokens` /
/// `total_tokens`) so HTTP servers built on `MotifGenerationEvent` can report
/// real counts instead of zeros. The underlying MLX `generate(...)` `.info`
/// completion carries these figures; this struct threads them out of the
/// runtime to consumers (e.g. `MotifNativeServe`).
public struct MotifGenerationUsage: Equatable, Sendable {
    public var promptTokens: Int
    public var completionTokens: Int

    public var totalTokens: Int { promptTokens + completionTokens }

    public init(promptTokens: Int, completionTokens: Int) {
        self.promptTokens = promptTokens
        self.completionTokens = completionTokens
    }
}

public enum MotifGenerationEvent: Equatable, Sendable {
    case text(String)
    case reasoning(String)
    /// Terminal event. Carries token-usage accounting when the backend can
    /// surface it (the native MLX runtime does); `nil` for backends that do
    /// not report counts (e.g. the remote OpenAI-compatible SSE client, whose
    /// upstream stream omits `usage`). Existing `case .completed:` patterns
    /// continue to match and may ignore the payload.
    case completed(usage: MotifGenerationUsage?)
}

public protocol MotifChatBackend: Sendable {
    func streamResponse(
        messages: [MotifChatMessage],
        parameters: MotifGenerationParameters
    ) -> AsyncThrowingStream<MotifGenerationEvent, any Error>
}

public enum MotifBackendError: Error, LocalizedError, Equatable, Sendable {
    case nativeBackendUnavailable(String)
    case invalidEndpoint(URL)
    case httpStatus(Int)
    case malformedServerEvent(String)

    public var errorDescription: String? {
        switch self {
        case .nativeBackendUnavailable(let detail):
            "Native Motif MLX backend is not available yet: \(detail)"
        case .invalidEndpoint(let url):
            "Invalid Motif endpoint: \(url.absoluteString)"
        case .httpStatus(let status):
            "Motif endpoint returned HTTP \(status)"
        case .malformedServerEvent(let event):
            "Malformed streaming event from Motif endpoint: \(event)"
        }
    }
}

public enum MotifAttentionVariant: String, Codable, Equatable, Sendable {
    case vanillaDifferentialAttention = "vanilla_differential_attention"
    case groupedDifferentialAttention = "grouped_differential_attention"
}

public indirect enum MotifJSONValue: Codable, Equatable, Sendable {
    case string(String)
    case int(Int)
    case double(Double)
    case bool(Bool)
    case object([String: MotifJSONValue])
    case array([MotifJSONValue])
    case null

    public init(from decoder: any Decoder) throws {
        let container = try decoder.singleValueContainer()
        if container.decodeNil() {
            self = .null
        } else if let value = try? container.decode(Bool.self) {
            self = .bool(value)
        } else if let value = try? container.decode(Int.self) {
            self = .int(value)
        } else if let value = try? container.decode(Double.self) {
            self = .double(value)
        } else if let value = try? container.decode(String.self) {
            self = .string(value)
        } else if let value = try? container.decode([MotifJSONValue].self) {
            self = .array(value)
        } else if let value = try? container.decode([String: MotifJSONValue].self) {
            self = .object(value)
        } else {
            throw DecodingError.dataCorruptedError(
                in: container,
                debugDescription: "Unsupported JSON value in Motif configuration"
            )
        }
    }

    public func encode(to encoder: any Encoder) throws {
        var container = encoder.singleValueContainer()
        switch self {
        case .string(let value):
            try container.encode(value)
        case .int(let value):
            try container.encode(value)
        case .double(let value):
            try container.encode(value)
        case .bool(let value):
            try container.encode(value)
        case .object(let value):
            try container.encode(value)
        case .array(let value):
            try container.encode(value)
        case .null:
            try container.encodeNil()
        }
    }
}

public enum MotifModelConfigurationError: Error, LocalizedError, Equatable, Sendable {
    case unsupportedModelType(String)
    case nonPositiveField(String, Int)
    case unsupportedHiddenActivation(String)
    case invalidAttentionShape(String)

    public var errorDescription: String? {
        switch self {
        case .unsupportedModelType(let modelType):
            "Unsupported Motif model_type: \(modelType)"
        case .nonPositiveField(let field, let value):
            "Motif config field \(field) must be positive; got \(value)"
        case .unsupportedHiddenActivation(let activation):
            "Unsupported Motif hidden_act: \(activation)"
        case .invalidAttentionShape(let detail):
            "Invalid Motif attention shape: \(detail)"
        }
    }
}

public struct MotifRopeScalingConfiguration: Codable, Equatable, Sendable {
    public var type: String?
    public var ropeType: String?
    public var factor: Double?
    public var originalMaxPositionEmbeddings: Int?
    public var lowFreqFactor: Double?
    public var highFreqFactor: Double?

    public init(
        type: String? = nil,
        ropeType: String? = nil,
        factor: Double? = nil,
        originalMaxPositionEmbeddings: Int? = nil,
        lowFreqFactor: Double? = nil,
        highFreqFactor: Double? = nil
    ) {
        self.type = type
        self.ropeType = ropeType
        self.factor = factor
        self.originalMaxPositionEmbeddings = originalMaxPositionEmbeddings
        self.lowFreqFactor = lowFreqFactor
        self.highFreqFactor = highFreqFactor
    }

    private enum CodingKeys: String, CodingKey {
        case type
        case ropeType = "rope_type"
        case factor
        case originalMaxPositionEmbeddings = "original_max_position_embeddings"
        case lowFreqFactor = "low_freq_factor"
        case highFreqFactor = "high_freq_factor"
    }
}

public struct MotifGenerationConfiguration: Codable, Equatable, Sendable {
    public var bosTokenId: Int?
    public var eosTokenIds: [Int]
    public var padTokenId: Int?
    public var maxNewTokens: Int?
    public var temperature: Double?
    public var doSample: Bool?
    public var chatTemplate: String?

    public var primaryEOSTokenId: Int? { eosTokenIds.first }
    public var bosTokenID: Int? {
        get { bosTokenId }
        set { bosTokenId = newValue }
    }
    public var eosTokenIDs: [Int] {
        get { eosTokenIds }
        set { eosTokenIds = newValue }
    }
    public var padTokenID: Int? {
        get { padTokenId }
        set { padTokenId = newValue }
    }

    public init(
        bosTokenId: Int? = nil,
        eosTokenIds: [Int] = [],
        padTokenId: Int? = nil,
        maxNewTokens: Int? = nil,
        temperature: Double? = nil,
        doSample: Bool? = nil,
        chatTemplate: String? = nil
    ) {
        self.bosTokenId = bosTokenId
        self.eosTokenIds = eosTokenIds
        self.padTokenId = padTokenId
        self.maxNewTokens = maxNewTokens
        self.temperature = temperature
        self.doSample = doSample
        self.chatTemplate = chatTemplate
    }

    private enum CodingKeys: String, CodingKey {
        case bosTokenId = "bos_token_id"
        case eosTokenId = "eos_token_id"
        case padTokenId = "pad_token_id"
        case maxNewTokens = "max_new_tokens"
        case temperature
        case doSample = "do_sample"
        case chatTemplate = "chat_template"
    }

    public init(from decoder: any Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        bosTokenId = try container.decodeIfPresent(Int.self, forKey: .bosTokenId)
        if let eosTokenId = try? container.decode(Int.self, forKey: .eosTokenId) {
            eosTokenIds = [eosTokenId]
        } else {
            eosTokenIds = (try? container.decode([Int].self, forKey: .eosTokenId)) ?? []
        }
        padTokenId = try container.decodeIfPresent(Int.self, forKey: .padTokenId)
        maxNewTokens = try container.decodeIfPresent(Int.self, forKey: .maxNewTokens)
        temperature = try container.decodeIfPresent(Double.self, forKey: .temperature)
        doSample = try container.decodeIfPresent(Bool.self, forKey: .doSample)
        chatTemplate = try container.decodeIfPresent(String.self, forKey: .chatTemplate)
    }

    public func encode(to encoder: any Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        try container.encodeIfPresent(bosTokenId, forKey: .bosTokenId)
        try container.encode(eosTokenIds, forKey: .eosTokenId)
        try container.encodeIfPresent(padTokenId, forKey: .padTokenId)
        try container.encodeIfPresent(maxNewTokens, forKey: .maxNewTokens)
        try container.encodeIfPresent(temperature, forKey: .temperature)
        try container.encodeIfPresent(doSample, forKey: .doSample)
        try container.encodeIfPresent(chatTemplate, forKey: .chatTemplate)
    }
}

public struct MotifModelConfiguration: Codable, Equatable, Sendable {
    public static let canonicalModelType = "motif"

    public var modelType: String
    public var hiddenSize: Int
    public var numHiddenLayers: Int
    public var intermediateSize: Int
    public var numAttentionHeads: Int
    public var numKeyValueHeads: Int
    public var vocabSize: Int
    public var rmsNormEps: Double
    public var ropeTheta: Double
    public var maxPositionEmbeddings: Int
    public var headDim: Int?
    public var numNoiseHeads: Int?
    public var kRatio: Int
    public var attnRMSNormEps: Double
    public var tieWordEmbeddings: Bool
    public var ropeScaling: MotifRopeScalingConfiguration?
    public var hiddenActivation: String
    public var useBias: Bool
    public var expanded: Bool
    public var slidingWindow: Int?
    public var useSlidingWindow: Bool
    public var maxWindowLayers: Int?
    public var fusedRope: Bool
    public var bosTokenId: Int?
    public var eosTokenId: Int?
    public var quantization: [String: MotifJSONValue]?

    public var isGroupedDifferentialAttention: Bool { numNoiseHeads != nil }
    public var effectiveHeadDim: Int { headDim ?? hiddenSize / numAttentionHeads }
    public var attentionVariant: MotifAttentionVariant {
        isGroupedDifferentialAttention ? .groupedDifferentialAttention : .vanillaDifferentialAttention
    }
    public var groupedRatio: Int? {
        guard let numNoiseHeads else { return nil }
        return (numAttentionHeads - numNoiseHeads) / numNoiseHeads
    }
    public var keyNoiseHeads: Int? {
        guard isGroupedDifferentialAttention else { return nil }
        return numKeyValueHeads / (kRatio + 1)
    }
    public var requiredCustomKernelNames: [String] {
        if isGroupedDifferentialAttention {
            return MotifMetalKernelRegistry.required.map(\.name)
        }
        return ["polynorm"]
    }
    public var bosTokenID: Int? {
        get { bosTokenId }
        set { bosTokenId = newValue }
    }
    public var eosTokenID: Int? {
        get { eosTokenId }
        set { eosTokenId = newValue }
    }

    public init(
        modelType: String = Self.canonicalModelType,
        hiddenSize: Int,
        numHiddenLayers: Int,
        intermediateSize: Int,
        numAttentionHeads: Int,
        numKeyValueHeads: Int,
        vocabSize: Int,
        rmsNormEps: Double = 1e-6,
        ropeTheta: Double = 10_000,
        maxPositionEmbeddings: Int = 8_192,
        headDim: Int? = nil,
        numNoiseHeads: Int? = nil,
        kRatio: Int = 1,
        attnRMSNormEps: Double = 1e-5,
        tieWordEmbeddings: Bool = false,
        ropeScaling: MotifRopeScalingConfiguration? = nil,
        hiddenActivation: String = "poly_norm",
        useBias: Bool = false,
        expanded: Bool = false,
        slidingWindow: Int? = nil,
        useSlidingWindow: Bool = false,
        maxWindowLayers: Int? = nil,
        fusedRope: Bool = false,
        bosTokenId: Int? = nil,
        eosTokenId: Int? = nil,
        quantization: [String: MotifJSONValue]? = nil
    ) {
        self.modelType = modelType
        self.hiddenSize = hiddenSize
        self.numHiddenLayers = numHiddenLayers
        self.intermediateSize = intermediateSize
        self.numAttentionHeads = numAttentionHeads
        self.numKeyValueHeads = numKeyValueHeads
        self.vocabSize = vocabSize
        self.rmsNormEps = rmsNormEps
        self.ropeTheta = ropeTheta
        self.maxPositionEmbeddings = maxPositionEmbeddings
        self.headDim = headDim
        self.numNoiseHeads = numNoiseHeads
        self.kRatio = kRatio
        self.attnRMSNormEps = attnRMSNormEps
        self.tieWordEmbeddings = tieWordEmbeddings
        self.ropeScaling = ropeScaling
        self.hiddenActivation = hiddenActivation
        self.useBias = useBias
        self.expanded = expanded
        self.slidingWindow = slidingWindow
        self.useSlidingWindow = useSlidingWindow
        self.maxWindowLayers = maxWindowLayers
        self.fusedRope = fusedRope
        self.bosTokenId = bosTokenId
        self.eosTokenId = eosTokenId
        self.quantization = quantization
    }

    private enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case hiddenSize = "hidden_size"
        case numHiddenLayers = "num_hidden_layers"
        case intermediateSize = "intermediate_size"
        case numAttentionHeads = "num_attention_heads"
        case numKeyValueHeads = "num_key_value_heads"
        case vocabSize = "vocab_size"
        case rmsNormEps = "rms_norm_eps"
        case ropeTheta = "rope_theta"
        case maxPositionEmbeddings = "max_position_embeddings"
        case headDim = "head_dim"
        case numNoiseHeads = "num_noise_heads"
        case kRatio = "k_ratio"
        case attnRMSNormEps = "attn_rms_norm_eps"
        case tieWordEmbeddings = "tie_word_embeddings"
        case ropeScaling = "rope_scaling"
        case hiddenActivation = "hidden_act"
        case useBias = "use_bias"
        case expanded
        case slidingWindow = "sliding_window"
        case useSlidingWindow = "use_sliding_window"
        case maxWindowLayers = "max_window_layers"
        case fusedRope = "fused_rope"
        case bosTokenId = "bos_token_id"
        case eosTokenId = "eos_token_id"
        case quantization
    }

    public init(from decoder: any Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        self.init(
            modelType: try container.decodeIfPresent(String.self, forKey: .modelType) ?? "motif",
            hiddenSize: try container.decode(Int.self, forKey: .hiddenSize),
            numHiddenLayers: try container.decode(Int.self, forKey: .numHiddenLayers),
            intermediateSize: try container.decode(Int.self, forKey: .intermediateSize),
            numAttentionHeads: try container.decode(Int.self, forKey: .numAttentionHeads),
            numKeyValueHeads: try container.decode(Int.self, forKey: .numKeyValueHeads),
            vocabSize: try container.decode(Int.self, forKey: .vocabSize),
            rmsNormEps: try container.decodeIfPresent(Double.self, forKey: .rmsNormEps) ?? 1e-6,
            ropeTheta: try container.decodeIfPresent(Double.self, forKey: .ropeTheta) ?? 10_000,
            maxPositionEmbeddings: try container.decodeIfPresent(
                Int.self,
                forKey: .maxPositionEmbeddings
            ) ?? 8_192,
            headDim: try container.decodeIfPresent(Int.self, forKey: .headDim),
            numNoiseHeads: try container.decodeIfPresent(Int.self, forKey: .numNoiseHeads),
            kRatio: try container.decodeIfPresent(Int.self, forKey: .kRatio) ?? 1,
            attnRMSNormEps: try container.decodeIfPresent(
                Double.self,
                forKey: .attnRMSNormEps
            ) ?? 1e-5,
            tieWordEmbeddings: try container.decodeIfPresent(
                Bool.self,
                forKey: .tieWordEmbeddings
            ) ?? false,
            ropeScaling: try container.decodeIfPresent(
                MotifRopeScalingConfiguration.self,
                forKey: .ropeScaling
            ),
            hiddenActivation: try container.decodeIfPresent(
                String.self,
                forKey: .hiddenActivation
            ) ?? "poly_norm",
            useBias: try container.decodeIfPresent(Bool.self, forKey: .useBias) ?? false,
            expanded: try container.decodeIfPresent(Bool.self, forKey: .expanded) ?? false,
            slidingWindow: try container.decodeIfPresent(Int.self, forKey: .slidingWindow),
            useSlidingWindow: try container.decodeIfPresent(
                Bool.self,
                forKey: .useSlidingWindow
            ) ?? false,
            maxWindowLayers: try container.decodeIfPresent(Int.self, forKey: .maxWindowLayers),
            fusedRope: try container.decodeIfPresent(Bool.self, forKey: .fusedRope) ?? false,
            bosTokenId: try container.decodeIfPresent(Int.self, forKey: .bosTokenId),
            eosTokenId: try container.decodeIfPresent(Int.self, forKey: .eosTokenId),
            quantization: try container.decodeIfPresent(
                [String: MotifJSONValue].self,
                forKey: .quantization
            )
        )
    }
}

public struct MotifRuntimeFeatureFlags: Equatable, Sendable {
    public var dualVAttention: Bool
    public var fourSlotCache: String?
    public var quantizedSDPA: Bool
    public var disableCustomKernels: Bool
    public var fuseQueryKeyValue: Bool

    public init(
        dualVAttention: Bool = true,
        fourSlotCache: String? = nil,
        quantizedSDPA: Bool = true,
        disableCustomKernels: Bool = false,
        fuseQueryKeyValue: Bool = false
    ) {
        self.dualVAttention = dualVAttention
        self.fourSlotCache = fourSlotCache
        self.quantizedSDPA = quantizedSDPA
        self.disableCustomKernels = disableCustomKernels
        self.fuseQueryKeyValue = fuseQueryKeyValue
    }

    public enum FourSlotCacheMode: String, Codable, Equatable, Sendable {
        case disabled
        case fp = "1"
        case q4
        case q8

        public var cacheKind: MotifKVCacheKind {
            switch self {
            case .disabled:
                .standard
            case .fp:
                .groupedFourSlot
            case .q4:
                .groupedQuantized4Bit
            case .q8:
                .groupedQuantized8Bit
            }
        }
    }

    public var fourSlotCacheMode: FourSlotCacheMode {
        Self.parseFourSlotCacheMode(fourSlotCache)
    }

    public static func fromEnvironment(
        _ environment: [String: String] = ProcessInfo.processInfo.environment
    ) -> Self {
        MotifRuntimeFeatureFlags(
            dualVAttention: !isFalsy(environment["MLX_MOTIF_DUAL_V"], defaultValue: true),
            fourSlotCache: environment["MLX_MOTIF_4SLOT_CACHE"],
            quantizedSDPA: !isFalsy(environment["MLX_MOTIF_QUANT_SDPA"], defaultValue: true),
            disableCustomKernels: !isFalsy(environment["MLX_MOTIF_DISABLE_KERNELS"], defaultValue: false),
            // QKV fusion defaults ON for the grouped q4 decode path: the
            // synthetic decode micro-benchmark (MotifDecodeBench, q4 gs=64, B=1,
            // S=1) shows ~12-20% lower median ms/step at the 12.7B per-layer
            // shape with fusion enabled. Fused == unfused numerical equivalence
            // (incl. the q4 path) is gated by MotifQKVFusionParityTests under
            // MOTIFKIT_RUN_MLX_RUNTIME_TESTS=1. Opt out with MLX_MOTIF_FUSE_QKV=0.
            fuseQueryKeyValue: !isFalsy(environment["MLX_MOTIF_FUSE_QKV"], defaultValue: true)
        )
    }

    public static func parseFourSlotCacheMode(_ value: String?) -> FourSlotCacheMode {
        guard let normalized = value?.trimmingCharacters(in: .whitespacesAndNewlines).lowercased(),
              !["", "0", "false", "off", "no"].contains(normalized)
        else {
            return .disabled
        }
        switch normalized {
        case "q4", "4":
            return .q4
        case "q8", "8":
            return .q8
        default:
            return .fp
        }
    }

    private static func isFalsy(_ value: String?, defaultValue: Bool) -> Bool {
        guard let normalized = value?.trimmingCharacters(in: .whitespacesAndNewlines) else {
            return !defaultValue
        }
        return ["", "0", "false", "False", "off", "OFF", "no", "NO"].contains(normalized)
    }
}
