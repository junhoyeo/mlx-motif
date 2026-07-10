import MLX
import MLXNN
import MotifKit
@testable import MotifKitMLX
import XCTest

/// Numerics-parity gate for the grouped-attention lambda scalar.
///
/// Python's grouped forward keeps lambda in fp32 all the way into the kernels
/// (`self._lambda_full(mx.float32)`, model.py:517), because `lambda =
/// exp(q1·k1) - exp(q2·k2) + lambda_init` is numerically sensitive (model.py:255:
/// "kept fp32 for numerical stability of exp/sub"). The Swift grouped path must
/// do the same: `lambdaFull(dtype: .float32)`. Casting to the fp16/bf16 activation
/// dtype first would round lambda before the Metal wrappers upcast it back, so the
/// upcast could not recover the lost mantissa bits and every grouped layer would
/// diverge from Python.
///
/// These tests also cover the inference-time cache: lambda is frozen after weight
/// load, so `lambdaFull` materializes the fp32 scalar once and reuses it.
final class MotifLambdaFP32ParityTests: XCTestCase {
    private func groupedConfiguration() -> MotifModelConfiguration {
        // Grouped-differential config (q heads = (groupedRatio+1) * noiseHeads),
        // headDim 64 to mirror the fusion-parity fixtures.
        MotifModelConfiguration(
            hiddenSize: 128,
            numHiddenLayers: 2,
            intermediateSize: 256,
            numAttentionHeads: 12,
            numKeyValueHeads: 4,
            vocabSize: 64,
            headDim: 64,
            numNoiseHeads: 4,
            kRatio: 1,
            hiddenActivation: "poly_norm"
        )
    }

    /// Populate the lambda projection vectors so the resulting scalar is NOT
    /// exactly representable in fp16 — this is what makes fp32 retention observable.
    private func makeScaffold(layerIndex: Int) throws -> MotifMLXAttentionScaffold {
        let configuration = groupedConfiguration()
        let scaffold = try MotifMLXAttentionScaffold(
            configuration: configuration,
            layerIndex: layerIndex
        )
        let headDim = scaffold.layout.headDim
        func constant(_ value: Float) -> MLXArray {
            (MLXArray.ones([headDim], dtype: .float32) * value).asType(.float32)
        }
        scaffold.update(parameters: ModuleParameters.unflattened([
            ("lambda_q1", constant(0.10)),
            ("lambda_k1", constant(0.10)),
            ("lambda_q2", constant(0.05)),
            ("lambda_k2", constant(0.05)),
        ]))
        return scaffold
    }

    /// Reference fp32 lambda computed exactly as Python's `_lambda_full(mx.float32)`.
    private func referenceLambdaFP32(_ scaffold: MotifMLXAttentionScaffold) -> MLXArray {
        let l1 = exp(sum(scaffold.lambdaQ1 * scaffold.lambdaK1, axis: -1).asType(.float32))
        let l2 = exp(sum(scaffold.lambdaQ2 * scaffold.lambdaK2, axis: -1).asType(.float32))
        return (l1 - l2 + scaffold.lambdaInit).asType(.float32)
    }

    func testGroupedLambdaIsComputedInFloat32() throws {
        try requireMLXRuntime()
        let scaffold = try makeScaffold(layerIndex: 5)

        let expected = referenceLambdaFP32(scaffold)
        let actual = scaffold.lambdaFull(dtype: .float32)
        eval(expected, actual)

        XCTAssertEqual(actual.dtype, .float32, "grouped lambda must stay fp32 into the kernels")
        XCTAssertEqual(
            actual.item(Float.self),
            expected.item(Float.self),
            accuracy: 0.0,
            "fp32 lambda must match Python's _lambda_full(mx.float32) exactly"
        )
    }

    func testFloat32LambdaDiffersFromFloat16Rounding() throws {
        try requireMLXRuntime()
        let scaffold = try makeScaffold(layerIndex: 5)

        let fp32 = scaffold.lambdaFull(dtype: .float32)
        // What the old code produced: lambda rounded to the fp16 activation dtype
        // before the wrappers upcast it back to fp32.
        let fp16RoundTrip = fp32.asType(.float16).asType(.float32)
        eval(fp32, fp16RoundTrip)

        let delta = abs(fp32.item(Float.self) - fp16RoundTrip.item(Float.self))
        XCTAssertGreaterThan(
            delta,
            1e-6,
            "test fixture must exercise a lambda value that fp16 cannot represent exactly"
        )
    }

    func testLambdaFullIsCachedAndStableAcrossCalls() throws {
        try requireMLXRuntime()
        let scaffold = try makeScaffold(layerIndex: 3)

        let first = scaffold.lambdaFull(dtype: .float32)
        let second = scaffold.lambdaFull(dtype: .float32)
        eval(first, second)

        // The cache must be bit-identical across calls (no parity drift).
        XCTAssertEqual(
            first.item(Float.self),
            second.item(Float.self),
            accuracy: 0.0,
            "cached fp32 lambda must be identical across forward calls"
        )

        // A non-fp32 request is a cheap cast of the same cached fp32 base.
        let asHalf = scaffold.lambdaFull(dtype: .float16)
        let expectedHalf = first.asType(.float16)
        eval(asHalf, expectedHalf)
        XCTAssertEqual(asHalf.dtype, .float16)
        XCTAssertEqual(
            asHalf.item(Float.self),
            expectedHalf.item(Float.self),
            accuracy: 0.0,
            "non-fp32 lambda must be a direct cast of the cached fp32 value"
        )
    }

    private func requireMLXRuntime(file: StaticString = #filePath, line: UInt = #line) throws {
        try XCTSkipUnless(
            ProcessInfo.processInfo.environment["MOTIFKIT_RUN_MLX_RUNTIME_TESTS"] == "1",
            "MLX runtime ops require the default metallib; build-only CI still verifies type-checking.",
            file: file,
            line: line
        )
    }
}
