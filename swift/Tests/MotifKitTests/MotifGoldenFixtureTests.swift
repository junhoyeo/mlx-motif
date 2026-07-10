import XCTest
@testable import MotifKit

private struct GoldenFixture: Decodable {
    var modelConfigs: [String: MotifModelConfiguration]
    var attentionPathCases: [AttentionPathCase]
    var polynorm: PolyNormCase
    var requiredKernels: [String]

    enum CodingKeys: String, CodingKey {
        case modelConfigs = "model_configs"
        case attentionPathCases = "attention_path_cases"
        case polynorm
        case requiredKernels = "required_kernels"
    }
}

private struct AttentionPathCase: Decodable {
    var name: String
    var cacheKind: MotifKVCacheKind
    var sequenceLength: Int
    var keyValueRepeat: Int
    var keyRatio: Int
    var fusedRope: Bool
    var dualVAttention: Bool
    var quantizedSDPA: Bool
    var expected: MotifAttentionPath

    enum CodingKeys: String, CodingKey {
        case name
        case cacheKind = "cache_kind"
        case sequenceLength = "sequence_length"
        case keyValueRepeat = "key_value_repeat"
        case keyRatio = "key_ratio"
        case fusedRope = "fused_rope"
        case dualVAttention = "dual_v_attention"
        case quantizedSDPA = "quantized_sdpa"
        case expected
    }
}

private struct PolyNormCase: Decodable {
    var input: [Double]
    var weight: [Double]
    var bias: Double
    var epsilon: Double
    var expected: [Double]
}

private func loadGoldenFixture(file: StaticString = #filePath, line: UInt = #line) throws -> GoldenFixture {
    let url = try XCTUnwrap(
        Bundle.module.url(forResource: "motif_golden_config", withExtension: "json"),
        "Missing golden fixture resource",
        file: file,
        line: line
    )
    let data = try Data(contentsOf: url)
    let decoder = JSONDecoder()
    return try decoder.decode(GoldenFixture.self, from: data)
}

final class MotifGoldenFixtureTests: XCTestCase {
    func testPythonStyleModelConfigsDecodeWithDefaults() throws {
        let fixture = try loadGoldenFixture()
        let vanilla = try XCTUnwrap(fixture.modelConfigs["vanilla_26b"])
        XCTAssertEqual(vanilla.modelType, "motif")
        XCTAssertEqual(vanilla.hiddenActivation, "poly_norm")
        XCTAssertFalse(vanilla.isGroupedDifferentialAttention)
        XCTAssertEqual(vanilla.effectiveHeadDim, 128)
        XCTAssertEqual(vanilla.attnRMSNormEps, 1e-5)

        let grouped = try XCTUnwrap(fixture.modelConfigs["grouped_127b"])
        XCTAssertTrue(grouped.isGroupedDifferentialAttention)
        XCTAssertEqual(grouped.numNoiseHeads, 8)
        XCTAssertEqual(grouped.headDim, 128)
        XCTAssertEqual(grouped.kRatio, 1)
    }

    func testAttentionLayoutMatchesPythonShapeDerivations() throws {
        let fixture = try loadGoldenFixture()
        let vanilla = try MotifAttentionLayout(configuration: XCTUnwrap(fixture.modelConfigs["vanilla_26b"]))
        XCTAssertEqual(vanilla.variant, .vanillaDifferential)
        XCTAssertEqual(vanilla.queryHeads, 8)
        XCTAssertEqual(vanilla.keyValueHeads, 2)
        XCTAssertEqual(vanilla.keyValueRepeat, 4)
        XCTAssertEqual(vanilla.qProjectionSize, 2048)
        XCTAssertEqual(vanilla.kProjectionSize, 512)
        XCTAssertEqual(vanilla.vProjectionSize, 512)
        XCTAssertEqual(vanilla.outputProjectionInputSize, 2048)

        let grouped = try MotifAttentionLayout(configuration: XCTUnwrap(fixture.modelConfigs["grouped_127b"]))
        XCTAssertEqual(grouped.variant, .groupedDifferential)
        XCTAssertEqual(grouped.queryHeads, 40)
        XCTAssertEqual(grouped.keyValueHeads, 16)
        XCTAssertEqual(grouped.keyNoiseHeads, 8)
        XCTAssertEqual(grouped.groupedRatio, 4)
        XCTAssertEqual(grouped.keyValueRepeat, 1)
        XCTAssertEqual(grouped.qProjectionSize, 5120)
        XCTAssertEqual(grouped.kProjectionSize, 2048)
        XCTAssertEqual(grouped.vProjectionSize, 2048)
        XCTAssertEqual(grouped.outputProjectionInputSize, 8192)
    }

    func testAttentionPathResolverMatchesGoldenCases() throws {
        let fixture = try loadGoldenFixture()
        for testCase in fixture.attentionPathCases {
            let flags = MotifRuntimeFeatureFlags(
                dualVAttention: testCase.dualVAttention,
                quantizedSDPA: testCase.quantizedSDPA
            )
            let resolved = MotifAttentionPathResolver.resolve(
                cacheKind: testCase.cacheKind,
                sequenceLength: testCase.sequenceLength,
                keyValueRepeat: testCase.keyValueRepeat,
                keyRatio: testCase.keyRatio,
                fusedRope: testCase.fusedRope,
                featureFlags: flags
            )
            XCTAssertEqual(resolved, testCase.expected, testCase.name)
        }
    }

    func testRuntimeFeatureFlagsParsePythonCompatibleEnvironment() {
        let q4 = MotifRuntimeFeatureFlags.fromEnvironment([
            "MLX_MOTIF_4SLOT_CACHE": "q4",
            "MLX_MOTIF_DUAL_V": "1",
            "MLX_MOTIF_QUANT_SDPA": "1",
        ])
        XCTAssertEqual(q4.fourSlotCacheMode, .q4)
        XCTAssertEqual(q4.fourSlotCacheMode.cacheKind, .groupedQuantized4Bit)
        XCTAssertTrue(q4.dualVAttention)
        XCTAssertTrue(q4.quantizedSDPA)

        let disabled = MotifRuntimeFeatureFlags.fromEnvironment([
            "MLX_MOTIF_4SLOT_CACHE": "0",
            "MLX_MOTIF_DUAL_V": "false",
            "MLX_MOTIF_QUANT_SDPA": "off",
            "MLX_MOTIF_DISABLE_KERNELS": "1",
        ])
        XCTAssertEqual(disabled.fourSlotCacheMode, .disabled)
        XCTAssertFalse(disabled.dualVAttention)
        XCTAssertFalse(disabled.quantizedSDPA)
        XCTAssertTrue(disabled.disableCustomKernels)

        // Unset defaults ON (fp16 4-slot) — the measured-fastest decode
        // configuration; parity with Python's Model.make_cache default.
        let defaults = MotifRuntimeFeatureFlags.fromEnvironment([:])
        XCTAssertEqual(defaults.fourSlotCacheMode, .fp)
        XCTAssertEqual(defaults.fourSlotCacheMode.cacheKind, .groupedFourSlot)
        // QKV fusion now defaults ON for the grouped q4 decode path (measured
        // ~12-20% lower median ms/step at the 12.7B per-layer shape in the
        // synthetic decode micro-benchmark). It remains explicitly opt-out.
        XCTAssertTrue(MotifRuntimeFeatureFlags.fromEnvironment([:]).fuseQueryKeyValue)
        XCTAssertTrue(MotifRuntimeFeatureFlags.fromEnvironment(["MLX_MOTIF_FUSE_QKV": "1"]).fuseQueryKeyValue)
        XCTAssertFalse(MotifRuntimeFeatureFlags.fromEnvironment(["MLX_MOTIF_FUSE_QKV": "0"]).fuseQueryKeyValue)
        XCTAssertFalse(MotifRuntimeFeatureFlags.fromEnvironment(["MLX_MOTIF_FUSE_QKV": "off"]).fuseQueryKeyValue)
    }

    func testPolyNormReferenceMatchesGoldenFixture() throws {
        let fixture = try loadGoldenFixture()
        let coefficients = MotifPolyNormCoefficients(
            weight: fixture.polynorm.weight,
            bias: fixture.polynorm.bias,
            epsilon: fixture.polynorm.epsilon
        )
        let got = try MotifReferenceMath.polyNorm(row: fixture.polynorm.input, coefficients: coefficients)
        XCTAssertEqual(got.count, fixture.polynorm.expected.count)
        for (actual, expected) in zip(got, fixture.polynorm.expected) {
            XCTAssertEqual(actual, expected, accuracy: 1e-12)
        }
    }

    func testKernelRegistryCoversPythonCustomKernelSurface() throws {
        let fixture = try loadGoldenFixture()
        let names = Set(MotifMetalKernelRegistry.required.map(\.name))
        XCTAssertEqual(names, Set(fixture.requiredKernels))
    }


    func testModelBundleLoaderReadsConfigAndExtraEOSMetadata() throws {
        let bundleURL = try XCTUnwrap(Bundle.module.resourceURL, "Missing test resource directory")
        XCTAssertTrue(FileManager.default.fileExists(atPath: bundleURL.appendingPathComponent("config.json").path))
        let bundle = try MotifModelBundleLoader.loadMetadata(from: bundleURL)
        XCTAssertEqual(bundle.configuration.modelType, "motif")
        XCTAssertEqual(bundle.configuration.numNoiseHeads, 8)
        XCTAssertEqual(bundle.generationConfiguration?.eosTokenIDs, [2, 100257])
        XCTAssertEqual(bundle.extraEOSTokenIDs, [2, 100257])
    }
}
