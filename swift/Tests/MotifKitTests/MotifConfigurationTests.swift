import XCTest
@testable import MotifKit

final class MotifConfigurationTests: XCTestCase {
    func testGroupedAttentionFlagTracksNoiseHeads() {
        let vanilla = MotifModelConfiguration(
            hiddenSize: 2048,
            numHiddenLayers: 24,
            intermediateSize: 8192,
            numAttentionHeads: 16,
            numKeyValueHeads: 4,
            vocabSize: 128_000
        )
        XCTAssertFalse(vanilla.isGroupedDifferentialAttention)

        let grouped = MotifModelConfiguration(
            hiddenSize: 4096,
            numHiddenLayers: 40,
            intermediateSize: 16384,
            numAttentionHeads: 40,
            numKeyValueHeads: 16,
            vocabSize: 128_000,
            headDim: 128,
            numNoiseHeads: 8
        )
        XCTAssertTrue(grouped.isGroupedDifferentialAttention)
    }
}
