import Foundation

public enum MotifArchitectureError: Error, LocalizedError, Equatable, Sendable {
    case invalidGroupedAttention(String)
    case invalidVanillaAttention(String)
    case unsupportedActivation(String)

    public var errorDescription: String? {
        switch self {
        case .invalidGroupedAttention(let detail):
            "Invalid Motif grouped attention configuration: \(detail)"
        case .invalidVanillaAttention(let detail):
            "Invalid Motif vanilla attention configuration: \(detail)"
        case .unsupportedActivation(let activation):
            "Unsupported Motif activation: \(activation)"
        }
    }
}

public struct MotifAttentionLayout: Codable, Equatable, Sendable {
    public enum Variant: String, Codable, Sendable {
        case vanillaDifferential = "vanilla_differential"
        case groupedDifferential = "grouped_differential"
    }

    public var variant: Variant
    public var headDim: Int
    public var queryHeads: Int
    public var keyValueHeads: Int
    public var keyNoiseHeads: Int?
    public var groupedRatio: Int
    public var keyRatio: Int
    public var keyValueRepeat: Int
    public var qProjectionSize: Int
    public var kProjectionSize: Int
    public var vProjectionSize: Int
    public var outputProjectionInputSize: Int

    public init(configuration: MotifModelConfiguration) throws {
        let headDim = configuration.effectiveHeadDim
        if let noiseHeads = configuration.numNoiseHeads {
            let originHeads = configuration.numAttentionHeads - noiseHeads
            guard noiseHeads > 0, originHeads > 0, originHeads.isMultiple(of: noiseHeads) else {
                throw MotifArchitectureError.invalidGroupedAttention(
                    "num_attention_heads - num_noise_heads must be a positive multiple of num_noise_heads"
                )
            }
            let groupedRatio = originHeads / noiseHeads
            let keyRatio = configuration.kRatio
            guard keyRatio >= 1 else {
                throw MotifArchitectureError.invalidGroupedAttention("k_ratio must be >= 1")
            }
            let keySlots = keyRatio + 1
            guard configuration.numKeyValueHeads.isMultiple(of: keySlots) else {
                throw MotifArchitectureError.invalidGroupedAttention(
                    "num_key_value_heads must be divisible by k_ratio + 1"
                )
            }
            let keyNoiseHeads = configuration.numKeyValueHeads / keySlots
            guard keyNoiseHeads > 0, noiseHeads.isMultiple(of: keyNoiseHeads) else {
                throw MotifArchitectureError.invalidGroupedAttention(
                    "num_noise_heads must be divisible by derived key noise heads"
                )
            }

            self.variant = .groupedDifferential
            self.headDim = headDim
            self.queryHeads = (groupedRatio + 1) * noiseHeads
            self.keyValueHeads = configuration.numKeyValueHeads
            self.keyNoiseHeads = keyNoiseHeads
            self.groupedRatio = groupedRatio
            self.keyRatio = keyRatio
            self.keyValueRepeat = noiseHeads / keyNoiseHeads
            self.qProjectionSize = queryHeads * headDim
            self.kProjectionSize = configuration.numKeyValueHeads * headDim
            self.vProjectionSize = 2 * keyNoiseHeads * headDim
            self.outputProjectionInputSize = 2 * groupedRatio * noiseHeads * headDim
        } else {
            guard configuration.numAttentionHeads.isMultiple(of: 2),
                  configuration.numKeyValueHeads.isMultiple(of: 2)
            else {
                throw MotifArchitectureError.invalidVanillaAttention(
                    "num_attention_heads and num_key_value_heads must be even for vanilla differential attention"
                )
            }
            let effectiveHeads = configuration.numAttentionHeads / 2
            let effectiveKVHeads = configuration.numKeyValueHeads / 2
            guard effectiveKVHeads > 0, effectiveHeads.isMultiple(of: effectiveKVHeads) else {
                throw MotifArchitectureError.invalidVanillaAttention(
                    "effective attention heads must be divisible by effective KV heads"
                )
            }
            let repeatCount = effectiveHeads / effectiveKVHeads

            self.variant = .vanillaDifferential
            self.headDim = headDim
            self.queryHeads = effectiveHeads
            self.keyValueHeads = effectiveKVHeads
            self.keyNoiseHeads = nil
            self.groupedRatio = 1
            self.keyRatio = 1
            self.keyValueRepeat = repeatCount
            self.qProjectionSize = configuration.hiddenSize
            self.kProjectionSize = configuration.hiddenSize / repeatCount
            self.vProjectionSize = configuration.hiddenSize / repeatCount
            self.outputProjectionInputSize = configuration.hiddenSize
        }
    }

    public static func lambdaInit(layerIndex: Int) -> Double {
        0.8 - 0.6 * exp(-0.3 * Double(layerIndex - 1))
    }
}

public enum MotifKVCacheKind: String, Codable, Equatable, Sendable {
    case standard
    case groupedFourSlot = "grouped_4slot"
    case groupedQuantized4Bit = "grouped_q4"
    case groupedQuantized8Bit = "grouped_q8"

    public var isGroupedFourSlot: Bool {
        switch self {
        case .groupedFourSlot, .groupedQuantized4Bit, .groupedQuantized8Bit:
            true
        case .standard:
            false
        }
    }

    public var isQuantized: Bool {
        switch self {
        case .groupedQuantized4Bit, .groupedQuantized8Bit:
            true
        case .standard, .groupedFourSlot:
            false
        }
    }
}

public enum MotifAttentionPath: String, Codable, Equatable, Sendable {
    case quantizedSDPA = "sdpa_dual_v_q4"
    case dualV = "sdpa_dual_v"
    case fallback = "stacked_sdpa"
}

public enum MotifAttentionPathResolver {
    public static func resolve(
        cacheKind: MotifKVCacheKind?,
        sequenceLength: Int,
        keyValueRepeat: Int,
        keyRatio: Int,
        fusedRope: Bool,
        featureFlags: MotifRuntimeFeatureFlags = .init()
    ) -> MotifAttentionPath {
        let decodeShapeOK = sequenceLength == 1 && keyValueRepeat == 1 && keyRatio == 1
        guard decodeShapeOK, !fusedRope else { return .fallback }

        if cacheKind?.isQuantized == true, featureFlags.quantizedSDPA {
            return .quantizedSDPA
        }
        if featureFlags.dualVAttention {
            return .dualV
        }
        return .fallback
    }
}

public struct MotifPolyNormCoefficients: Codable, Equatable, Sendable {
    public var weight: [Double]
    public var bias: Double
    public var epsilon: Double

    public init(weight: [Double] = [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0], bias: Double = 0, epsilon: Double = 1e-6) {
        self.weight = weight
        self.bias = bias
        self.epsilon = epsilon
    }
}

public enum MotifReferenceMath {
    public static func polyNorm(row: [Double], coefficients: MotifPolyNormCoefficients = .init()) throws -> [Double] {
        guard coefficients.weight.count == 3 else {
            throw MotifArchitectureError.unsupportedActivation("poly_norm requires exactly three weights")
        }
        guard !row.isEmpty else { return [] }

        func rmsNormalized(_ values: [Double]) -> [Double] {
            let meanSquare = values.reduce(0) { $0 + $1 * $1 } / Double(values.count)
            let scale = 1.0 / sqrt(meanSquare + coefficients.epsilon)
            return values.map { $0 * scale }
        }

        let squared = row.map { $0 * $0 }
        let cubed = zip(squared, row).map { $0 * $1 }
        let normCubed = rmsNormalized(cubed)
        let normSquared = rmsNormalized(squared)
        let normInput = rmsNormalized(row)

        return row.indices.map { index in
            coefficients.weight[0] * normCubed[index]
                + coefficients.weight[1] * normSquared[index]
                + coefficients.weight[2] * normInput[index]
                + coefficients.bias
        }
    }
}

public struct MotifMLPLayout: Codable, Equatable, Sendable {
    public var hiddenSize: Int
    public var intermediateSize: Int
    public var activationName: String
    public var usesBias: Bool
    public var gateProjectionShape: [Int]
    public var upProjectionShape: [Int]
    public var downProjectionShape: [Int]

    public init(configuration: MotifModelConfiguration) {
        self.hiddenSize = configuration.hiddenSize
        self.intermediateSize = configuration.intermediateSize
        self.activationName = configuration.hiddenActivation
        self.usesBias = configuration.useBias
        self.gateProjectionShape = [configuration.intermediateSize, configuration.hiddenSize]
        self.upProjectionShape = [configuration.intermediateSize, configuration.hiddenSize]
        self.downProjectionShape = [configuration.hiddenSize, configuration.intermediateSize]
    }
}

public struct MotifMetalKernelDescriptor: Codable, Equatable, Sendable {
    public var name: String
    public var domain: String
    public var defaultEnabled: Bool
    public var referenceSymbol: String
    public var shapeContract: String

    public init(
        name: String,
        domain: String,
        defaultEnabled: Bool,
        referenceSymbol: String,
        shapeContract: String
    ) {
        self.name = name
        self.domain = domain
        self.defaultEnabled = defaultEnabled
        self.referenceSymbol = referenceSymbol
        self.shapeContract = shapeContract
    }
}

public enum MotifMetalKernelRegistry {
    public static let required: [MotifMetalKernelDescriptor] = [
        .init(
            name: "polynorm",
            domain: "mlp",
            defaultEnabled: true,
            referenceSymbol: "polynorm_reference",
            shapeContract: "x (..., D), weight (3), bias (1) -> (..., D)"
        ),
        .init(
            name: "gda_post",
            domain: "attention",
            defaultEnabled: true,
            referenceSymbol: "gda_post_reference",
            shapeContract: "merged (B, q_groups*gr + q_groups, S, 2*head_dim) -> (B, q_groups*gr, S, 2*head_dim)"
        ),
        .init(
            name: "gda_post_split",
            domain: "attention",
            defaultEnabled: true,
            referenceSymbol: "gda_post_split_reference",
            shapeContract: "attn_o (B, q_groups*gr, S, 2*head_dim), attn_n (B, q_groups, S, 2*head_dim) -> attn_o shape"
        ),
        .init(
            name: "sdpa_dual_v",
            domain: "attention",
            defaultEnabled: true,
            referenceSymbol: "sdpa_dual_v_reference",
            shapeContract: "q (B, Hq, 1, D), k/v1/v2 (B, Hkv, K, D) -> (B, Hq, 1, 2*D)"
        ),
        .init(
            name: "sdpa_dual_v_q4",
            domain: "attention",
            defaultEnabled: true,
            referenceSymbol: "sdpa_dual_v_q4_reference",
            shapeContract: "q (B, Hq, 1, D), quantized k/v1/v2 triples -> (B, Hq, 1, 2*D)"
        ),
    ]
}
