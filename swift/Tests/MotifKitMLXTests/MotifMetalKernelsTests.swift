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

    /// Locks the GQA-broadcast contract the callers now rely on: passing
    /// unrepeated GQA-shaped K/V1/V2 must be bit-identical to the legacy
    /// caller behavior of pre-repeating the slabs to full query-head count
    /// (the kernel maps `kv_head_idx = head_idx / GQA_FACTOR`, which is
    /// exactly repeat-interleave).
    func testDualVMetalGQABroadcastMatchesPreRepeatedInputsBitExact() throws {
        try requireMLXRuntime()
        let q = MotifMetalKernelHarness.deterministicInput(shape: [1, 4, 1, 32])
        let k = MotifMetalKernelHarness.deterministicInput(shape: [1, 2, 3, 32])
        let v1 = MotifMetalKernelHarness.deterministicInput(shape: [1, 2, 3, 32]) * 0.5
        let v2 = MotifMetalKernelHarness.deterministicInput(shape: [1, 2, 3, 32]) * -0.25
        let scale: Float = 0.17677669

        let broadcast = MotifSDPADualV.apply(
            queries: q,
            keys: k,
            value1: v1,
            value2: v2,
            scale: scale,
            executionMode: .metalPreferred
        )
        let preRepeated = MotifSDPADualV.apply(
            queries: q,
            keys: repeated(k, count: 2, axis: 1),
            value1: repeated(v1, count: 2, axis: 1),
            value2: repeated(v2, count: 2, axis: 1),
            scale: scale,
            executionMode: .metalPreferred
        )
        eval(broadcast, preRepeated)

        XCTAssertEqual(broadcast.shape, preRepeated.shape)
        XCTAssertEqual(
            abs(broadcast - preRepeated).max().item(Float.self),
            0,
            "native GQA broadcast must be bit-identical to pre-repeated inputs"
        )
    }

    /// An explicit `.array` mask must be honored (via the reference fallback)
    /// instead of being silently dropped by the mask-less Metal kernel.
    func testDualVApplyHonorsExplicitArrayMaskAtDecode() throws {
        try requireMLXRuntime()
        let q = MotifMetalKernelHarness.deterministicInput(shape: [1, 4, 1, 32])
        let k = MotifMetalKernelHarness.deterministicInput(shape: [1, 2, 3, 32])
        let v1 = MotifMetalKernelHarness.deterministicInput(shape: [1, 2, 3, 32]) * 0.5
        let v2 = MotifMetalKernelHarness.deterministicInput(shape: [1, 2, 3, 32]) * -0.25
        let scale: Float = 0.17677669
        // Additive mask blocking the final KV position — NOT a decode no-op.
        let mask = MLXArray([Float(0), 0, -1e9], [1, 1, 1, 3])

        MotifKernelFallbackTelemetry.reset()
        let masked = MotifSDPADualV.apply(
            queries: q,
            keys: k,
            value1: v1,
            value2: v2,
            scale: scale,
            mask: .array(mask),
            executionMode: .metalPreferred
        )
        let reference = MotifSDPADualV.reference(
            queries: q,
            keys: k,
            value1: v1,
            value2: v2,
            scale: scale,
            mask: .array(mask)
        )
        let unmasked = MotifSDPADualV.apply(
            queries: q,
            keys: k,
            value1: v1,
            value2: v2,
            scale: scale,
            executionMode: .metalPreferred
        )
        eval(masked, reference, unmasked)

        XCTAssertGreaterThan(
            MotifKernelFallbackTelemetry.fallbackCount(for: .sdpaDualV),
            0,
            "a non-trivial mask must route to the mask-honoring reference path"
        )
        XCTAssertEqual(
            abs(masked - reference).max().item(Float.self),
            0,
            "masked apply must match the mask-honoring reference exactly"
        )
        XCTAssertGreaterThan(
            abs(masked - unmasked).max().item(Float.self),
            0,
            "the mask must actually change the result for this fixture"
        )
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

    /// Item 1 acceptance: exercise `kvLen < capacity` through the PUBLIC fp16
    /// wrapper. Full step-padded capacity buffers whose padding rows are POISON
    /// (values large enough to blow up an unbounded softmax) must, when the live
    /// length is passed as `kvLen`, produce output bit-close to the exact-length
    /// reference computed over only the live region — proving the kernel indexes
    /// by capacity stride but never reads past `kvLen`.
    func testDualVApplyBoundsPaddedCapacityBuffersByKVLen() throws {
        try requireMLXRuntime()
        let scale: Float = 0.17677669
        let liveLength = 3
        let paddingRows = 5  // capacity = liveLength + paddingRows = 8 > liveLength
        let q = MotifMetalKernelHarness.deterministicInput(shape: [1, 4, 1, 32])
        let kLive = MotifMetalKernelHarness.deterministicInput(shape: [1, 2, liveLength, 32])
        let v1Live = MotifMetalKernelHarness.deterministicInput(shape: [1, 2, liveLength, 32]) * 0.5
        let v2Live = MotifMetalKernelHarness.deterministicInput(shape: [1, 2, liveLength, 32]) * -0.25

        // Poison the padding region: the padding KEYS repeat a real live key
        // row (so they attend with an ordinary, finite softmax weight rather
        // than being pushed to ~0 by an extreme score), while the padding VALUES
        // are huge — so any unbounded read swamps the accumulator with poison V.
        let keyPoison = repeated(kLive[.ellipsis, 0 ..< 1, 0...], count: paddingRows, axis: 2)
        let valuePoison = MLXArray.ones([1, 2, paddingRows, 32]) * 1e3
        let kFull = concatenated([kLive, keyPoison], axis: 2)
        let v1Full = concatenated([v1Live, valuePoison], axis: 2)
        let v2Full = concatenated([v2Live, valuePoison], axis: 2)

        let reference = MotifSDPADualV.reference(
            queries: q, keys: kLive, value1: v1Live, value2: v2Live, scale: scale
        )
        let bounded = MotifSDPADualV.apply(
            queries: q,
            keys: kFull,
            value1: v1Full,
            value2: v2Full,
            scale: scale,
            kvLen: liveLength,
            executionMode: .metalPreferred
        )
        let unbounded = MotifSDPADualV.apply(
            queries: q,
            keys: kFull,
            value1: v1Full,
            value2: v2Full,
            scale: scale,
            executionMode: .metalPreferred
        )
        eval(reference, bounded, unbounded)

        XCTAssertEqual(bounded.shape, reference.shape)
        XCTAssertTrue(
            allClose(reference, bounded, rtol: 1e-3, atol: 1e-3).item(Bool.self),
            "kvLen must bound the kernel to the live region; poisoned padding must be ignored"
        )
        XCTAssertGreaterThan(
            abs(unbounded - reference).max().item(Float.self),
            1e-2,
            "without the kvLen bound the poisoned padding must corrupt the output"
        )
    }

    /// Item 1 acceptance for the packed q4 wrapper: full step-padded capacity
    /// triples whose padding rows are poison must, when bounded by `kvLen`,
    /// match the exact-length reference over only the live region.
    func testPackedQ4DualVApplyBoundsPaddedCapacityTriplesByKVLen() throws {
        try requireMLXRuntime()
        let scale: Float = 0.17677669
        let liveLength = 3
        let paddingRows = 5
        let q = MotifMetalKernelHarness.deterministicInput(shape: [1, 4, 1, 32])
        let kLive = MotifMetalKernelHarness.deterministicInput(shape: [1, 2, liveLength, 32])
        let v1Live = MotifMetalKernelHarness.deterministicInput(shape: [1, 2, liveLength, 32]) * 0.25
        let v2Live = MotifMetalKernelHarness.deterministicInput(shape: [1, 2, liveLength, 32]) * -0.125
        // Padding keys repeat a real live key row (ordinary softmax weight);
        // padding values are huge, so any unbounded read is swamped by poison V.
        let keyPoison = repeated(kLive[.ellipsis, 0 ..< 1, 0...], count: paddingRows, axis: 2)
        let valuePoison = MLXArray.ones([1, 2, paddingRows, 32]) * 1e3
        let kFull = concatenated([kLive, keyPoison], axis: 2)
        let v1Full = concatenated([v1Live, valuePoison], axis: 2)
        let v2Full = concatenated([v2Live, valuePoison], axis: 2)

        // Quantization groups along the last (channel) axis per row, so the live
        // rows quantize bit-identically whether or not padding is appended.
        func packed(_ x: MLXArray) -> MotifQuantizedTuple {
            let q = quantized(x, groupSize: 32, bits: 4, mode: .affine)
            return (q.wq, q.scales, q.biases)
        }

        // Exact-length reference: dequant + dual-V SDPA over the live rows only.
        let reference = MotifSDPADualVQ4.reference(
            queries: q,
            quantizedKeys: packed(kLive),
            quantizedValue1: packed(v1Live),
            quantizedValue2: packed(v2Live),
            scale: scale,
            groupSize: 32,
            bits: 4,
            mode: .affine,
            dtype: q.dtype
        )
        let bounded = MotifSDPADualVQ4.apply(
            queries: q,
            quantizedKeys: packed(kFull),
            quantizedValue1: packed(v1Full),
            quantizedValue2: packed(v2Full),
            scale: scale,
            groupSize: 32,
            bits: 4,
            mode: .affine,
            dtype: q.dtype,
            kvLen: liveLength,
            executionMode: .metalPreferred
        )
        let unbounded = MotifSDPADualVQ4.apply(
            queries: q,
            quantizedKeys: packed(kFull),
            quantizedValue1: packed(v1Full),
            quantizedValue2: packed(v2Full),
            scale: scale,
            groupSize: 32,
            bits: 4,
            mode: .affine,
            dtype: q.dtype,
            executionMode: .metalPreferred
        )
        eval(reference, bounded, unbounded)

        XCTAssertEqual(bounded.shape, reference.shape)
        XCTAssertTrue(
            allClose(reference, bounded, rtol: 1e-3, atol: 1e-3).item(Bool.self),
            "kvLen must bound the packed kernel to the live region; poisoned padding must be ignored"
        )
        XCTAssertGreaterThan(
            abs(unbounded - reference).max().item(Float.self),
            1e-2,
            "without the kvLen bound the poisoned padding must corrupt the packed output"
        )
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

    /// A head dim that violates the packed word contract (qk_per_thread = D/32
    /// must be <= EL_PER_INT = 32/bits) must route to the dequant reference
    /// bridge, never the packed Metal kernel. For D=256/bits=8, qk_per_thread=8
    /// > EL_PER_INT=4: the kernel would shift a uint32 by up to 56 bits (UB in
    /// Metal) and read half the channels as garbage. This asserts the guard in
    /// MotifSDPADualVQ4.apply forces the fallback, matching the Python assert.
    /// Runtime-gated so `shouldUseMetal` is true — the ONLY reason to fall back
    /// here is the head-dim/bits guard, not the absence of a Metal runtime.
    func testPackedQ4DualVFallsBackWhenHeadDimExceedsPackedWordContract() throws {
        try requireMLXRuntime()
        MotifKernelFallbackTelemetry.reset()
        XCTAssertEqual(MotifKernelFallbackTelemetry.fallbackCount(for: .sdpaDualVQ4), 0)

        // D=256 with bits=8 violates D/32 <= 32/bits (8 > 4).
        let q = MotifMetalKernelHarness.deterministicInput(shape: [1, 4, 1, 256])
        let cache = MotifGroupedQuantizedKVCache(groupSize: 64, bits: 8)
        let packed = cache.updateAndFetch4Quantized(
            kOrigin: MotifMetalKernelHarness.deterministicInput(shape: [1, 2, 3, 256]),
            kNoise: MotifMetalKernelHarness.deterministicInput(shape: [1, 2, 3, 256]) * -0.5,
            value1: MotifMetalKernelHarness.deterministicInput(shape: [1, 2, 3, 256]) * 0.25,
            value2: MotifMetalKernelHarness.deterministicInput(shape: [1, 2, 3, 256]) * -0.125
        )

        // reference() dequantizes and runs the generic dual-V SDPA, so the
        // out-of-contract shape still produces a correct, evaluatable result.
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
        eval(candidate)

        XCTAssertGreaterThan(
            MotifKernelFallbackTelemetry.fallbackCount(for: .sdpaDualVQ4),
            0,
            "out-of-contract D=256/bits=8 shape must fall back to reference(), not run the packed kernel"
        )
        // The forced fallback must equal the dequant reference bridge exactly.
        XCTAssertEqual(candidate.shape, reference.shape)
        XCTAssertTrue(allClose(reference, candidate, rtol: 1e-3, atol: 1e-3).item(Bool.self))
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
