import Foundation
import XCTest
@testable import MotifKit

/// CI-compiled assertions for `MotifAttentionPathResolver.resolve`.
///
/// This resolver is pure logic (no MLX), so it builds and runs in plain CI
/// without the MLX overlay or a Metal toolchain. The point of these tests is to
/// pin the contract that decides whether a decode step is dispatched to the
/// direct custom Metal path (`.quantizedSDPA` / `.dualV`) or to the safe
/// stacked-SDPA reference (`.fallback`). If a future change silently downgrades
/// a known-good decode shape to `.fallback`, one of these assertions breaks.
final class MotifAttentionPathResolverTests: XCTestCase {
    /// A q4 cache + single-token decode + kv_repeat == 1 + no fused rope must
    /// select the quantized Metal SDPA path and never silently fall back.
    func testQuantizedDecodeSelectsQuantizedSDPA() {
        let path = MotifAttentionPathResolver.resolve(
            cacheKind: .groupedQuantized4Bit,
            sequenceLength: 1,
            keyValueRepeat: 1,
            keyRatio: 1,
            fusedRope: false
        )

        XCTAssertEqual(path, .quantizedSDPA)
        XCTAssertNotEqual(path, .fallback, "q4 decode must not be silently downgraded to the reference path")
    }

    /// Fused rope is outside the decode kernel contract and must force the
    /// reference path even for an otherwise-valid q4 decode shape.
    func testFusedRopeForcesFallback() {
        let path = MotifAttentionPathResolver.resolve(
            cacheKind: .groupedQuantized4Bit,
            sequenceLength: 1,
            keyValueRepeat: 1,
            keyRatio: 1,
            fusedRope: true
        )

        XCTAssertEqual(path, .fallback)
    }

    /// Prefill (sequence length > 1) is outside the single-token decode contract
    /// and must resolve to the reference path regardless of cache kind.
    func testPrefillResolvesToFallback() {
        let path = MotifAttentionPathResolver.resolve(
            cacheKind: .groupedQuantized4Bit,
            sequenceLength: 4,
            keyValueRepeat: 1,
            keyRatio: 1,
            fusedRope: false
        )

        XCTAssertEqual(path, .fallback)
    }

    /// A non-quantized 4-slot decode cache should select the (non-quant) dual-V
    /// Metal path, not the quantized one and not the reference path.
    func testDualVDecodeSelectsDualV() {
        let path = MotifAttentionPathResolver.resolve(
            cacheKind: .groupedFourSlot,
            sequenceLength: 1,
            keyValueRepeat: 1,
            keyRatio: 1,
            fusedRope: false
        )

        XCTAssertEqual(path, .dualV)
        XCTAssertNotEqual(path, .fallback, "non-quant 4-slot decode must not be silently downgraded to the reference path")
    }

    /// A kv_repeat (GQA replication) other than 1 is outside the decode kernel
    /// contract and must resolve to the reference path.
    func testNonUnitKeyValueRepeatForcesFallback() {
        let path = MotifAttentionPathResolver.resolve(
            cacheKind: .groupedQuantized4Bit,
            sequenceLength: 1,
            keyValueRepeat: 2,
            keyRatio: 1,
            fusedRope: false
        )

        XCTAssertEqual(path, .fallback)
    }

    /// When the quantized-SDPA feature flag is disabled, a q4 decode falls back
    /// to the next eligible path (dual-V) rather than the quantized kernel.
    func testQuantizedFlagDisabledDropsToDualV() {
        let path = MotifAttentionPathResolver.resolve(
            cacheKind: .groupedQuantized4Bit,
            sequenceLength: 1,
            keyValueRepeat: 1,
            keyRatio: 1,
            fusedRope: false,
            featureFlags: MotifRuntimeFeatureFlags(dualVAttention: true, quantizedSDPA: false)
        )

        XCTAssertEqual(path, .dualV)
    }
}
