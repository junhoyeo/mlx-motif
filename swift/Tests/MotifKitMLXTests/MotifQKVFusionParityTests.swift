import MLX
import MLXNN
import MotifKit
@testable import MotifKitMLX
import XCTest

/// Numerical-equivalence gate for QKV-projection fusion.
///
/// Fusion (`MLX_MOTIF_FUSE_QKV`, default ON) concatenates the q/k/v projections
/// into a single matmul and applies an origin-first query permutation. These
/// tests prove the fused decoder produces the SAME logits as the unfused one
/// for identical weights — the equivalence the default-on flip relies on.
///
/// Both runs use one model instance (identical random weights); fusion only
/// activates after `fuseQueryKeyValueProjectionsIfPossible()`, so the first
/// forward exercises the unfused path and the second the fused path.
final class MotifQKVFusionParityTests: XCTestCase {
    private func groupedConfiguration() -> MotifModelConfiguration {
        // Small grouped-differential config (q heads = (groupedRatio+1) * noiseHeads),
        // headDim 64 so q4 group size 64 divides it.
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

    private func maxAbsDiff(_ a: MLXArray, _ b: MLXArray) -> Float {
        eval(a, b)
        return abs(a - b).max().item(Float.self)
    }

    func testFusedDecoderMatchesUnfusedLogits() throws {
        try requireMLXRuntime()
        let configuration = groupedConfiguration()
        let flags = MotifRuntimeFeatureFlags(fuseQueryKeyValue: true)
        let model = try MotifMLXModel(configuration: configuration, runtimeFeatures: flags)
        eval(model.parameters())

        let tokens = MLXArray((0 ..< 3).map { Int32($0 % configuration.vocabSize) }, [1, 3])

        // First forward: fused projection not built yet -> unfused path.
        let unfused = model(tokens, cache: nil)
        eval(unfused)

        let fusedCount = model.fuseQueryKeyValueProjectionsIfPossible()
        XCTAssertEqual(
            fusedCount,
            configuration.numHiddenLayers,
            "every grouped layer should fuse when MLX_MOTIF_FUSE_QKV is on and projections are bias-free"
        )

        // Second forward: same weights, now via the fused projection.
        let fused = model(tokens, cache: nil)
        eval(fused)

        XCTAssertEqual(unfused.shape, fused.shape)
        XCTAssertLessThan(
            maxAbsDiff(unfused, fused),
            1e-4,
            "fused QKV decode logits must match the unfused path"
        )
    }

    func testFusedQuantizedDecoderMatchesUnfusedLogits() throws {
        try requireMLXRuntime()
        let configuration = groupedConfiguration()
        let flags = MotifRuntimeFeatureFlags(fourSlotCache: "q4", fuseQueryKeyValue: true)
        let model = try MotifMLXModel(configuration: configuration, runtimeFeatures: flags)

        // Quantize Linear/Embedding to q4 (group size 64) to exercise the
        // QuantizedLinear fusion branch — the actual default decode path.
        quantize(model: model, groupSize: 64, bits: 4) { _, module in
            module is Linear || module is Embedding
        }
        eval(model.parameters())

        let tokens = MLXArray((0 ..< 3).map { Int32($0 % configuration.vocabSize) }, [1, 3])

        let unfused = model(tokens, cache: nil)
        eval(unfused)

        let fusedCount = model.fuseQueryKeyValueProjectionsIfPossible()
        XCTAssertEqual(fusedCount, configuration.numHiddenLayers)

        let fused = model(tokens, cache: nil)
        eval(fused)

        XCTAssertEqual(unfused.shape, fused.shape)
        XCTAssertLessThan(
            maxAbsDiff(unfused, fused),
            1e-4,
            "fused q4 QKV decode logits must match the unfused quantized path"
        )
    }

    func testFusionRespectsOptOut() throws {
        try requireMLXRuntime()
        let configuration = groupedConfiguration()
        let flags = MotifRuntimeFeatureFlags(fuseQueryKeyValue: false)
        let model = try MotifMLXModel(configuration: configuration, runtimeFeatures: flags)
        eval(model.parameters())

        // With fusion opted out, no layer should fuse.
        XCTAssertEqual(model.fuseQueryKeyValueProjectionsIfPossible(), 0)
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
