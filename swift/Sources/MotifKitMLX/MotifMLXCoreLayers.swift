#if canImport(MLX) && canImport(MLXNN) && canImport(MLXLLM) && canImport(MLXLMCommon)
import Foundation
import MLX
import MLXLLM
import MLXLMCommon
import MLXNN
import MotifKit

public enum MotifMLXCoreLayerError: Error, LocalizedError, Equatable, Sendable {
    case unsupportedActivation(String)

    public var errorDescription: String? {
        switch self {
        case .unsupportedActivation(let activation):
            "Unsupported Motif MLX activation: \(activation)"
        }
    }
}

public enum MotifMLXActivationName: String, Codable, Equatable, Sendable {
    case polyNorm = "poly_norm"
    case silu
    case gelu

    public init(configurationValue: String) throws {
        guard let value = Self(rawValue: configurationValue) else {
            throw MotifMLXCoreLayerError.unsupportedActivation(configurationValue)
        }
        self = value
    }
}

public final class MotifMLXPolyNorm: Module, UnaryLayer {
    public let weight: MLXArray
    public let bias: MLXArray
    public let epsilon: Float

    public init(
        weight: [Float] = [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0],
        bias: Float = 0,
        epsilon: Float = 1e-6
    ) {
        precondition(weight.count == 3, "Motif PolyNorm requires exactly three polynomial weights")
        self.weight = MLXArray(weight, [3])
        self.bias = MLXArray([bias], [1])
        self.epsilon = epsilon
        super.init()
    }

    public convenience init(coefficients: MotifPolyNormCoefficients) {
        self.init(
            weight: coefficients.weight.map(Float.init),
            bias: Float(coefficients.bias),
            epsilon: Float(coefficients.epsilon)
        )
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        func rmsNormalized(_ value: MLXArray) -> MLXArray {
            value * rsqrt(mean(value * value, axis: -1, keepDims: true) + epsilon)
        }

        let squared = x * x
        let cubed = squared * x
        return weight[0] * rmsNormalized(cubed)
            + weight[1] * rmsNormalized(squared)
            + weight[2] * rmsNormalized(x)
            + bias
    }
}

public final class MotifMLXRMSNorm: RMSNorm {
    public init(configuration: MotifModelConfiguration) {
        super.init(dimensions: configuration.hiddenSize, eps: Float(configuration.rmsNormEps))
    }

    public override init(dimensions: Int, eps: Float = 1e-5) {
        super.init(dimensions: dimensions, eps: eps)
    }
}

public final class MotifMLXMLP: Module, UnaryLayer {
    @ModuleInfo(key: "gate_proj") public var gateProjection: Linear
    @ModuleInfo(key: "up_proj") public var upProjection: Linear
    @ModuleInfo(key: "down_proj") public var downProjection: Linear
    @ModuleInfo(key: "act_fn") public var polyNorm: MotifMLXPolyNorm

    public let activationName: MotifMLXActivationName

    public init(
        hiddenSize: Int,
        intermediateSize: Int,
        activationName: MotifMLXActivationName = .polyNorm,
        useBias: Bool = false,
        polyNormCoefficients: MotifPolyNormCoefficients = .init()
    ) {
        self.activationName = activationName
        self._gateProjection.wrappedValue = Linear(hiddenSize, intermediateSize, bias: useBias)
        self._upProjection.wrappedValue = Linear(hiddenSize, intermediateSize, bias: useBias)
        self._downProjection.wrappedValue = Linear(intermediateSize, hiddenSize, bias: useBias)
        self._polyNorm.wrappedValue = MotifMLXPolyNorm(coefficients: polyNormCoefficients)
    }

    public convenience init(configuration: MotifModelConfiguration) throws {
        try self.init(
            hiddenSize: configuration.hiddenSize,
            intermediateSize: configuration.intermediateSize,
            activationName: MotifMLXActivationName(configurationValue: configuration.hiddenActivation),
            useBias: configuration.useBias
        )
    }

    public init(
        gateProjection: Linear,
        upProjection: Linear,
        downProjection: Linear,
        activationName: MotifMLXActivationName = .polyNorm,
        polyNormCoefficients: MotifPolyNormCoefficients = .init()
    ) {
        self.activationName = activationName
        self._gateProjection.wrappedValue = gateProjection
        self._upProjection.wrappedValue = upProjection
        self._downProjection.wrappedValue = downProjection
        self._polyNorm.wrappedValue = MotifMLXPolyNorm(coefficients: polyNormCoefficients)
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        downProjection(activate(gateProjection(x)) * upProjection(x))
    }

    private func activate(_ x: MLXArray) -> MLXArray {
        switch activationName {
        case .polyNorm:
            polyNorm(x)
        case .silu:
            silu(x)
        case .gelu:
            gelu(x)
        }
    }
}
#endif
