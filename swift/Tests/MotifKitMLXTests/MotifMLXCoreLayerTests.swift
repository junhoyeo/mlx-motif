import Foundation
import MLX
import MLXNN
import MotifKit
@testable import MotifKitMLX
import XCTest

final class MotifMLXCoreLayerTests: XCTestCase {
    override func setUpWithError() throws {
        try XCTSkipUnless(
            ProcessInfo.processInfo.environment["MOTIFKIT_RUN_MLX_RUNTIME_TESTS"] == "1",
            "MLX runtime ops require the default metallib; build-only CI still verifies layer type-checking."
        )
    }

    func testPolyNormMatchesMotifReferenceMath() throws {
        try requireMLXRuntime()
        let input = [1.0, -2.0, 0.5, -0.25]
        let coefficients = MotifPolyNormCoefficients(
            weight: [0.4, 0.3, 0.3],
            bias: 0.05,
            epsilon: 1e-6
        )
        let layer = MotifMLXPolyNorm(coefficients: coefficients)
        let x = MLXArray(converting: input, [1, input.count])

        let output = layer(x)
        eval(output)

        let expected = try MotifReferenceMath.polyNorm(row: input, coefficients: coefficients)
        let got = output.asArray(Float.self).map(Double.init)
        XCTAssertEqual(got.count, expected.count)
        for (actual, expected) in zip(got, expected) {
            XCTAssertEqual(actual, expected, accuracy: 1e-5)
        }
    }

    func testRMSNormMatchesReferenceFormula() throws {
        try requireMLXRuntime()
        let epsilon: Float = 1e-6
        let layer = MotifMLXRMSNorm(dimensions: 4, eps: epsilon)
        let values = [1.0, -2.0, 0.5, -0.25]
        let x = MLXArray(converting: values, [1, values.count])

        let output = layer(x)
        eval(output)

        let scale = 1.0 / sqrt(values.reduce(0) { $0 + $1 * $1 } / Double(values.count) + Double(epsilon))
        let expected = values.map { Float($0 * scale) }
        for (actual, expected) in zip(output.asArray(Float.self), expected) {
            XCTAssertEqual(actual, expected, accuracy: 1e-5)
        }
    }

    func testMotifMLPWithIdentityProjectionsUsesPolyNormGate() throws {
        try requireMLXRuntime()
        let coefficients = MotifPolyNormCoefficients(
            weight: [0.4, 0.3, 0.3],
            bias: 0.05,
            epsilon: 1e-6
        )
        let identity = MLXArray(converting: [
            1.0, 0, 0, 0,
            0, 1.0, 0, 0,
            0, 0, 1.0, 0,
            0, 0, 0, 1.0,
        ], [4, 4])
        let mlp = MotifMLXMLP(
            gateProjection: Linear(weight: identity),
            upProjection: Linear(weight: identity),
            downProjection: Linear(weight: identity),
            polyNormCoefficients: coefficients
        )
        let values = [1.0, -2.0, 0.5, -0.25]
        let x = MLXArray(converting: values, [1, 1, values.count])

        let output = mlp(x)
        eval(output)

        let gate = try MotifReferenceMath.polyNorm(row: values, coefficients: coefficients)
        let expected = zip(gate, values).map { Float($0 * $1) }
        XCTAssertEqual(output.shape, [1, 1, 4])
        for (actual, expected) in zip(output.asArray(Float.self), expected) {
            XCTAssertEqual(actual, expected, accuracy: 1e-5)
        }
    }

    func testMotifMLPBuildsFromConfiguration() throws {
        try requireMLXRuntime()
        let configuration = MotifModelConfiguration(
            hiddenSize: 4,
            numHiddenLayers: 1,
            intermediateSize: 8,
            numAttentionHeads: 2,
            numKeyValueHeads: 2,
            vocabSize: 16,
            hiddenActivation: "poly_norm"
        )
        let mlp = try MotifMLXMLP(configuration: configuration)
        let x = MLXArray(converting: [Double](repeating: 0.25, count: 4), [1, 1, 4])

        let output = mlp(x)
        eval(output)

        XCTAssertEqual(output.shape, [1, 1, 4])
        XCTAssertEqual(mlp.activationName, .polyNorm)
    }

    private func requireMLXRuntime(file: StaticString = #filePath, line: UInt = #line) throws {
        try XCTSkipUnless(
            ProcessInfo.processInfo.environment["MOTIFKIT_RUN_MLX_RUNTIME_TESTS"] == "1",
            "MLX runtime ops require the default metallib; build-only CI still verifies MotifKitMLX type-checking.",
            file: file,
            line: line
        )
    }
}
