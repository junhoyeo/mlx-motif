#if canImport(MLX)
import Foundation
import MLX
@testable import MotifKitMLX
import XCTest

final class MotifMetalKernelsTests: XCTestCase {
    func testManifestKeepsExperimentalKernelsDisabledByDefault() {
        let descriptors = MotifMetalKernels.descriptors
        XCTAssertEqual(descriptors.map(\.name), MotifMetalKernelName.allCases)
        XCTAssertTrue(descriptors.allSatisfy { !$0.defaultEnabled })
        XCTAssertEqual(MotifMetalKernels.descriptor(for: .polynorm).status, .wrapperScaffolded)
        XCTAssertEqual(MotifMetalKernels.descriptor(for: .gdaPost).status, .parityPending)
        XCTAssertEqual(MotifMetalKernels.descriptor(for: .gdaPost).pythonSymbol, "gda_post")
    }

    func testParityHarnessDocumentsPolyNormRuntimeCases() {
        let cases = MotifMetalKernelHarness.parityCases(for: .polynorm)

        XCTAssertEqual(cases.map(\.name), [
            "polynorm_reference_2x3",
            "polynorm_decode_hidden_4096",
        ])
        XCTAssertTrue(cases.allSatisfy { $0.requiresRuntime })
        XCTAssertTrue(cases.contains { $0.requiresExperimentalMetal })
        XCTAssertTrue(cases.allSatisfy { $0.sourceFixture.contains("fixture") || $0.sourceFixture.contains("Swift") })
    }

    func testBenchmarkHarnessDocumentsPolyNormReferenceVersusExperimentalMetal() {
        let cases = MotifMetalKernelHarness.benchmarkCases(for: .polynorm)

        XCTAssertEqual(cases.map(\.name), [
            "polynorm_decode_hidden_4096",
            "polynorm_prefill_128x4096",
        ])
        XCTAssertTrue(cases.allSatisfy { $0.requiresRuntime })
        XCTAssertTrue(cases.allSatisfy { $0.requiresExperimentalMetal })
        XCTAssertTrue(cases.allSatisfy { $0.baseline == "MotifPolynorm.reference" })
        XCTAssertTrue(cases.allSatisfy { $0.candidate == "MotifPolynorm.apply(.experimentalMetal)" })
        XCTAssertTrue(cases.allSatisfy { $0.iterations > 0 })
    }

    func testPolynormSafeEntryPointUsesReferenceByDefault() throws {
        try XCTSkipUnless(
            ProcessInfo.processInfo.environment["MOTIFKIT_RUN_MLX_RUNTIME_TESTS"] == "1",
            "MLX runtime ops require the default metallib; build-only CI still verifies wrapper type-checking."
        )
        let x = MLXArray([Float(1.0), -2.0, 3.0, 4.0, -5.0, 6.0], [2, 3])
        let weight = MLXArray([Float(0.4), 0.3, 0.3])
        let bias = MLXArray([Float(0.05)])

        let reference = MotifPolynorm.reference(x, weight: weight, bias: bias, eps: 1e-6)
        let safeDefault = MotifPolynorm.apply(
            x,
            weight: weight,
            bias: bias,
            eps: 1e-6,
            executionMode: .experimentalMetal
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

    func testPolynormBenchmarkHarnessRunsWhenExperimentalMetalIsEnabled() throws {
        try requireExperimentalMetalRuntime()
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

    private func requireMLXRuntime(file: StaticString = #filePath, line: UInt = #line) throws {
        try XCTSkipUnless(
            ProcessInfo.processInfo.environment["MOTIFKIT_RUN_MLX_RUNTIME_TESTS"] == "1",
            "MLX runtime ops require the default metallib; build-only CI still verifies wrapper type-checking.",
            file: file,
            line: line
        )
    }

    private func requireExperimentalMetalRuntime(file: StaticString = #filePath, line: UInt = #line) throws {
        try requireMLXRuntime(file: file, line: line)
        try XCTSkipUnless(
            MotifMetalKernels.experimentalMetalRequested,
            "Experimental Metal kernel benchmarks require MOTIFKIT_ENABLE_EXPERIMENTAL_METAL_KERNELS=1.",
            file: file,
            line: line
        )
    }
}
#endif
