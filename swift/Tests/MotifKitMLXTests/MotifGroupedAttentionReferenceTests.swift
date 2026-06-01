import MLX
import MotifKit
@testable import MotifKitMLX
import XCTest

final class MotifGroupedAttentionReferenceTests: XCTestCase {
    func testCachedDecodePlanMatchesGroupedAttentionShapeContract() throws {
        let layout = try MotifAttentionLayout(configuration: makeGroupedConfiguration())

        let plan = try MotifGroupedDifferentialAttentionReference.planStep(
            layout: layout,
            batchSize: 2,
            sequenceLength: 1,
            cacheOffset: 3,
            cacheKind: .groupedQuantized4Bit
        )

        XCTAssertEqual(plan.keyValueLength, 4)
        XCTAssertEqual(plan.ropePositionRange, [3, 4])
        XCTAssertEqual(plan.qOriginShape, [2, 8, 1, 16])
        XCTAssertEqual(plan.qNoiseShape, [2, 2, 1, 16])
        XCTAssertEqual(plan.kOriginUpdateShape, [2, 2, 1, 16])
        XCTAssertEqual(plan.kNoiseUpdateShape, [2, 2, 1, 16])
        XCTAssertEqual(plan.valueUpdateShape, [2, 2, 1, 16])
        XCTAssertEqual(plan.kOriginCachedShape, [2, 2, 4, 16])
        XCTAssertEqual(plan.kNoiseCachedShape, [2, 2, 4, 16])
        XCTAssertEqual(plan.valueCachedShape, [2, 2, 4, 16])
        XCTAssertEqual(plan.outputShape, [2, 8, 1, 32])
        XCTAssertEqual(plan.attentionPath, .quantizedSDPA)
        XCTAssertEqual(plan.maskPlan.kind, .none)
        XCTAssertEqual(plan.maskPlan.materializedShape, nil)
    }

    func testPrefillPlanMaterializesSlidingWindowMaskAndFallsBackFromDecodeKernel() throws {
        let layout = try MotifAttentionLayout(configuration: makeGroupedConfiguration())

        let plan = try MotifGroupedDifferentialAttentionReference.planStep(
            layout: layout,
            batchSize: 1,
            sequenceLength: 4,
            cacheOffset: 2,
            cacheKind: .groupedFourSlot,
            slidingWindow: 2
        )

        XCTAssertEqual(plan.keyValueLength, 6)
        XCTAssertEqual(plan.ropePositionRange, [2, 6])
        XCTAssertEqual(plan.maskPlan.kind, .materializedCausal)
        XCTAssertEqual(plan.maskPlan.materializedShape, [4, 6])
        XCTAssertEqual(plan.maskPlan.windowSize, 2)
        XCTAssertEqual(plan.attentionPath, .fallback)
    }

    func testSplitPreparedTensorsProducePythonGroupedStreamShapes() throws {
        try requireMLXRuntime()
        let layout = try MotifAttentionLayout(configuration: makeGroupedConfiguration())
        let q = zeros([1, 10, 2, 16])
        let k = zeros([1, 4, 2, 16])
        let v = zeros([1, 4, 2, 16])

        let slices = try MotifGroupedDifferentialAttentionReference.splitPreparedTensors(
            q: q,
            k: k,
            v: v,
            layout: layout
        )

        XCTAssertEqual(slices.qOrigin.shape, [1, 8, 2, 16])
        XCTAssertEqual(slices.qNoise.shape, [1, 2, 2, 16])
        XCTAssertEqual(slices.kOrigin.shape, [1, 2, 2, 16])
        XCTAssertEqual(slices.kNoise.shape, [1, 2, 2, 16])
        XCTAssertEqual(slices.value1.shape, [1, 2, 2, 16])
        XCTAssertEqual(slices.value2.shape, [1, 2, 2, 16])
    }

    func testGroupedKVCacheUpdateFetchReorderAndResetPreserveShapeContract() throws {
        try requireMLXRuntime()
        let cache = MotifGroupedKVCacheReference()
        XCTAssertTrue(cache.isEmpty)

        _ = cache.updateAndFetch(
            kOrigin: zeros([2, 2, 1, 4]),
            kNoise: zeros([2, 2, 1, 4]),
            value1: zeros([2, 2, 1, 4]),
            value2: zeros([2, 2, 1, 4])
        )
        let second = cache.updateAndFetch(
            kOrigin: zeros([2, 2, 1, 4]),
            kNoise: zeros([2, 2, 1, 4]),
            value1: zeros([2, 2, 1, 4]),
            value2: zeros([2, 2, 1, 4])
        )

        XCTAssertEqual(cache.cachedLength, 2)
        XCTAssertEqual(second.kOrigin.shape, [2, 2, 2, 4])
        XCTAssertEqual(second.kNoise.shape, [2, 2, 2, 4])
        XCTAssertEqual(second.value1.shape, [2, 2, 2, 4])
        XCTAssertEqual(second.value2.shape, [2, 2, 2, 4])

        cache.reorder(batchIndices: [1, 0])
        let reordered = cache.fetch()
        XCTAssertEqual(reordered?.kOrigin.shape, [2, 2, 2, 4])
        XCTAssertEqual(reordered?.kNoise.shape, [2, 2, 2, 4])
        XCTAssertEqual(cache.cachedLength, 2)

        cache.reset()
        XCTAssertTrue(cache.isEmpty)
        XCTAssertNil(cache.fetch())
    }

    func testProductionGroupedKVCacheStateAndTrimShapeContract() throws {
        try requireMLXRuntime()
        let cache = MotifGroupedKVCache()
        let k1 = MLXArray.ones([1, 2, 1, 4])
        let k2 = MLXArray.ones([1, 2, 1, 4]) * 2
        let v1 = MLXArray.ones([1, 2, 1, 4]) * 3
        let v2 = MLXArray.ones([1, 2, 1, 4]) * 4

        let cached = cache.updateAndFetch4(kOrigin: k1, kNoise: k2, value1: v1, value2: v2)

        XCTAssertEqual(cache.offset, 1)
        XCTAssertEqual(cached.kOrigin.shape, [1, 2, 1, 4])
        XCTAssertEqual(cache.state.count, 4)
        XCTAssertEqual(cache.metaState.first, "1")
        XCTAssertEqual(cache.trim(1), 1)
        XCTAssertEqual(cache.offset, 0)
    }

    func testQuantizedGroupedKVCacheProvidesPackedAndDequantizedPaths() throws {
        try requireMLXRuntime()
        // group size must be one of MLX's supported sizes (32/64/128) and divide
        // the quantized (head) dimension; mirror the production default (64).
        let cache = MotifGroupedQuantizedKVCache(groupSize: 32, bits: 4)
        let k1 = MLXArray.ones([1, 2, 1, 32])
        let k2 = MLXArray.ones([1, 2, 1, 32]) * 2
        let v1 = MLXArray.ones([1, 2, 1, 32]) * 3
        let v2 = MLXArray.ones([1, 2, 1, 32]) * 4

        let packed = cache.updateAndFetch4Quantized(kOrigin: k1, kNoise: k2, value1: v1, value2: v2)
        XCTAssertEqual(cache.offset, 1)
        XCTAssertEqual(Array(packed.kOrigin.data.shape[0...2]), [1, 2, 1])
        XCTAssertTrue([8, 12].contains(cache.state.count))

        let dequantized = cache.updateAndFetch4(kOrigin: k1, kNoise: k2, value1: v1, value2: v2)
        XCTAssertEqual(cache.offset, 2)
        XCTAssertEqual(dequantized.value2.shape, [1, 2, 2, 32])
    }

    func testInvalidPlanInputsThrowDeterministicErrors() throws {
        let layout = try MotifAttentionLayout(configuration: makeGroupedConfiguration())

        XCTAssertThrowsError(
            try MotifGroupedDifferentialAttentionReference.planStep(
                layout: layout,
                batchSize: 1,
                sequenceLength: 0
            )
        ) { error in
            XCTAssertEqual(
                error as? MotifGroupedAttentionReferenceError,
                .invalidShape("sequenceLength must be positive")
            )
        }
    }

    private func makeGroupedConfiguration() -> MotifModelConfiguration {
        MotifModelConfiguration(
            hiddenSize: 64,
            numHiddenLayers: 2,
            intermediateSize: 128,
            numAttentionHeads: 10,
            numKeyValueHeads: 4,
            vocabSize: 128,
            headDim: 16,
            numNoiseHeads: 2,
            kRatio: 1,
            hiddenActivation: "poly_norm"
        )
    }

    private func requireMLXRuntime(file: StaticString = #filePath, line: UInt = #line) throws {
        try XCTSkipUnless(
            ProcessInfo.processInfo.environment["MOTIFKIT_RUN_MLX_RUNTIME_TESTS"] == "1",
            "MLX runtime ops require the default metallib; build-only CI still verifies grouped attention planning type-checking.",
            file: file,
            line: line
        )
    }
}
