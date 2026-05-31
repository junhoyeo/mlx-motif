#if canImport(MLX)
import Foundation
import MLX
@testable import MotifKitMLX
import XCTest

final class MotifMetalKernelsTests: XCTestCase {
    func testManifestEnablesHardParityKernelsByDefault() {
        let descriptors = MotifMetalKernels.descriptors
        XCTAssertEqual(descriptors.map(\.name), MotifMetalKernelName.allCases)
        XCTAssertTrue(descriptors.allSatisfy(\.defaultEnabled))
        XCTAssertTrue(descriptors.allSatisfy { $0.status == .metalReady })
        XCTAssertEqual(MotifMetalKernels.descriptor(for: .gdaPost).pythonSymbol, "gda_post")
        XCTAssertFalse(MotifMetalKernels.customMetalDisabled)
    }

    func testParityHarnessDocumentsPolyNormRuntimeCases() {
        let cases = MotifMetalKernelHarness.parityCases(for: .polynorm)

        XCTAssertEqual(cases.map(\.name), [
            "polynorm_reference_2x3",
            "polynorm_decode_hidden_4096",
        ])
        XCTAssertTrue(cases.allSatisfy { $0.requiresRuntime })
        XCTAssertFalse(cases.contains { $0.requiresExperimentalMetal })
        XCTAssertTrue(cases.allSatisfy { $0.sourceFixture.contains("fixture") || $0.sourceFixture.contains("Swift") })
    }

    func testBenchmarkHarnessDocumentsPolyNormReferenceVersusDefaultMetal() {
        let cases = MotifMetalKernelHarness.benchmarkCases(for: .polynorm)

        XCTAssertEqual(cases.map(\.name), [
            "polynorm_decode_hidden_4096",
            "polynorm_prefill_128x4096",
        ])
        XCTAssertTrue(cases.allSatisfy { $0.requiresRuntime })
        XCTAssertFalse(cases.contains { $0.requiresExperimentalMetal })
        XCTAssertTrue(cases.allSatisfy { $0.baseline == "MotifPolynorm.reference" })
        XCTAssertTrue(cases.allSatisfy { $0.candidate == "MotifPolynorm.apply(.metalPreferred)" })
        XCTAssertTrue(cases.allSatisfy { $0.iterations > 0 })
    }

    func testPolynormReferenceOnlyEscapeHatchMatchesReference() throws {
        try requireMLXRuntime()
        let x = MLXArray([Float(1.0), -2.0, 3.0, 4.0, -5.0, 6.0], [2, 3])
        let weight = MLXArray([Float(0.4), 0.3, 0.3])
        let bias = MLXArray([Float(0.05)])

        let reference = MotifPolynorm.reference(x, weight: weight, bias: bias, eps: 1e-6)
        let safeDefault = MotifPolynorm.apply(
            x,
            weight: weight,
            bias: bias,
            eps: 1e-6,
            executionMode: .referenceOnly
        )

        XCTAssertEqual(safeDefault.shape, x.shape)
        XCTAssertTrue(allClose(reference, safeDefault, rtol: 1e-6, atol: 1e-6).item(Bool.self))
    }

    func testPolynormParityHarnessMatchesFixtureCase() throws {
        try requireMLXRuntime()
        let parityCase = try XCTUnwrap(
            MotifMetalKernelHarness.parityCases(for: .polynorm).first { $0.name == "polynorm_reference_2x3" }
        )
        let x = MLXArray(
            [Float(-1.5), -0.5, 0.0, 0.5, 1.0, 2.0],
            parityCase.shape
        )
        let weight = MLXArray([Float(0.2), 0.3, 0.5])
        let bias = MLXArray([Float(0.125)])

        let result = MotifMetalKernelHarness.checkPolynormParity(
            case: parityCase,
            x: x,
            weight: weight,
            bias: bias,
            eps: 1e-6,
            executionMode: .referenceOnly
        )

        XCTAssertTrue(result.passed)
        XCTAssertEqual(result.maxAbsoluteError, 0, accuracy: 1e-7)
        XCTAssertEqual(result.maxRelativeError, 0, accuracy: 1e-7)
    }

    func testDualVMetalMatchesReferenceForDecodeFixture() throws {
        try requireMLXRuntime()
        let q = MotifMetalKernelHarness.deterministicInput(shape: [1, 4, 1, 32])
        let k = MotifMetalKernelHarness.deterministicInput(shape: [1, 2, 3, 32])
        let v1 = MotifMetalKernelHarness.deterministicInput(shape: [1, 2, 3, 32]) * 0.5
        let v2 = MotifMetalKernelHarness.deterministicInput(shape: [1, 2, 3, 32]) * -0.25

        let reference = MotifSDPADualV.reference(queries: q, keys: k, value1: v1, value2: v2, scale: 0.17677669)
        let candidate = MotifSDPADualV.apply(
            queries: q,
            keys: k,
            value1: v1,
            value2: v2,
            scale: 0.17677669,
            executionMode: .metalPreferred
        )

        XCTAssertEqual(candidate.shape, reference.shape)
        XCTAssertTrue(allClose(reference, candidate, rtol: 1e-3, atol: 1e-3).item(Bool.self))
    }

    func testPackedQ4DualVMetalMatchesDequantBridgeForDecodeFixture() throws {
        try requireMLXRuntime()
        let q = MotifMetalKernelHarness.deterministicInput(shape: [1, 4, 1, 32])
        let cache = MotifGroupedQuantizedKVCache(groupSize: 32, bits: 4)
        let packed = cache.updateAndFetch4Quantized(
            kOrigin: MotifMetalKernelHarness.deterministicInput(shape: [1, 2, 3, 32]),
            kNoise: MotifMetalKernelHarness.deterministicInput(shape: [1, 2, 3, 32]) * -0.5,
            value1: MotifMetalKernelHarness.deterministicInput(shape: [1, 2, 3, 32]) * 0.25,
            value2: MotifMetalKernelHarness.deterministicInput(shape: [1, 2, 3, 32]) * -0.125
        )

        let reference = MotifSDPADualVQ4.reference(
            queries: q,
            quantizedKeys: packed.kOrigin,
            quantizedValue1: packed.value1,
            quantizedValue2: packed.value2,
            scale: 0.17677669,
            groupSize: cache.groupSize,
            bits: cache.bits,
            mode: cache.mode,
            dtype: q.dtype
        )
        let candidate = MotifSDPADualVQ4.apply(
            queries: q,
            quantizedKeys: packed.kOrigin,
            quantizedValue1: packed.value1,
            quantizedValue2: packed.value2,
            scale: 0.17677669,
            groupSize: cache.groupSize,
            bits: cache.bits,
            mode: cache.mode,
            dtype: q.dtype,
            executionMode: .metalPreferred
        )

        XCTAssertEqual(candidate.shape, reference.shape)
        XCTAssertTrue(allClose(reference, candidate, rtol: 1e-3, atol: 1e-3).item(Bool.self))
    }

    func testPolynormBenchmarkHarnessRunsWithDefaultMetal() throws {
        try requireMLXRuntime()
        let benchmarkCase = try XCTUnwrap(
            MotifMetalKernelHarness.benchmarkCases(for: .polynorm).first { $0.name == "polynorm_decode_hidden_4096" }
        )

        let result = MotifMetalKernelHarness.benchmarkPolynorm(case: benchmarkCase)

        XCTAssertEqual(result.caseName, benchmarkCase.name)
        XCTAssertEqual(result.iterations, benchmarkCase.iterations)
        XCTAssertGreaterThan(result.referenceNanoseconds, 0)
        XCTAssertGreaterThan(result.candidateNanoseconds, 0)
        XCTAssertGreaterThan(result.candidateSpeedup, 0)
    }

    /// Proves the telemetry counter stays at zero for a known-good q4 decode,
    /// i.e. the direct Metal kernel actually ran and did NOT silently fall back
    /// to the dequant reference bridge. Runtime-gated: it early-returns (skips)
    /// when `MOTIFKIT_RUN_MLX_RUNTIME_TESTS != 1`, so it is a no-op in build-only
    /// CI and sandboxes that lack the Metal runtime.
    func testPackedQ4DualVDecodeRunsMetalWithoutFallback() throws {
        try requireMLXRuntime()
        MotifKernelFallbackTelemetry.reset()
        XCTAssertEqual(MotifKernelFallbackTelemetry.fallbackCount(for: .sdpaDualVQ4), 0)

        let q = MotifMetalKernelHarness.deterministicInput(shape: [1, 4, 1, 32])
        let cache = MotifGroupedQuantizedKVCache(groupSize: 32, bits: 4)
        let packed = cache.updateAndFetch4Quantized(
            kOrigin: MotifMetalKernelHarness.deterministicInput(shape: [1, 2, 3, 32]),
            kNoise: MotifMetalKernelHarness.deterministicInput(shape: [1, 2, 3, 32]) * -0.5,
            value1: MotifMetalKernelHarness.deterministicInput(shape: [1, 2, 3, 32]) * 0.25,
            value2: MotifMetalKernelHarness.deterministicInput(shape: [1, 2, 3, 32]) * -0.125
        )

        let candidate = MotifSDPADualVQ4.apply(
            queries: q,
            quantizedKeys: packed.kOrigin,
            quantizedValue1: packed.value1,
            quantizedValue2: packed.value2,
            scale: 0.17677669,
            groupSize: cache.groupSize,
            bits: cache.bits,
            mode: cache.mode,
            dtype: q.dtype,
            executionMode: .metalPreferred
        )
        // Force evaluation so the kernel (or its fallback) actually executes.
        eval(candidate)

        XCTAssertEqual(
            MotifKernelFallbackTelemetry.fallbackCount(for: .sdpaDualVQ4),
            0,
            "q4 decode fixture fell back to the reference bridge instead of running the Metal kernel"
        )
    }

    private func requireMLXRuntime(file: StaticString = #filePath, line: UInt = #line) throws {
        try XCTSkipUnless(
            ProcessInfo.processInfo.environment["MOTIFKIT_RUN_MLX_RUNTIME_TESTS"] == "1",
            "MLX runtime ops require the default metallib; build-only CI still verifies wrapper type-checking.",
            file: file,
            line: line
        )
    }

}
#endif
