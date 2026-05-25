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
}
#endif
