import MotifKit
@testable import MotifKitMLX
import XCTest

/// Guards the fail-fast contract for the four-slot grouped KV cache when
/// `kRatio > 1`. The caches allocate every slot (kOrigin/kNoise/value1/value2)
/// with kOrigin's head count, but with `kRatio > 1` the kOrigin slot carries
/// `keyGroups*kRatio` heads while the other three carry `keyGroups` heads — an
/// unsupported combination that must be rejected up front rather than surfacing
/// as an opaque shape error mid-forward. (Python mirror: `Model.make_cache` /
/// `tests/test_model.py::test_make_cache_rejects_4slot_with_k_ratio_gt_1`.)
///
/// The guard in `newCache` uses `preconditionFailure`, which cannot be caught
/// by XCTest, so the decision logic is factored into the pure, non-crashing
/// helper `fourSlotCacheUnsupportedReason` that these tests exercise directly.
final class MotifFourSlotCacheGuardTests: XCTestCase {
    func testKRatio1IsSupportedForEveryMode() {
        for mode in [
            MotifRuntimeFeatureFlags.FourSlotCacheMode.disabled,
            .fp,
            .q4,
            .q8,
        ] {
            XCTAssertNil(
                MotifMLXModel.fourSlotCacheUnsupportedReason(mode: mode, kRatio: 1),
                "kRatio == 1 must be supported for mode \(mode)"
            )
        }
    }

    func testKRatioGreaterThan1IsRejectedForEnabledModes() {
        for mode in [
            MotifRuntimeFeatureFlags.FourSlotCacheMode.fp,
            .q4,
            .q8,
        ] {
            let reason = MotifMLXModel.fourSlotCacheUnsupportedReason(mode: mode, kRatio: 2)
            XCTAssertNotNil(reason, "kRatio > 1 must be rejected for mode \(mode)")
            XCTAssertTrue(
                reason?.contains("kRatio") ?? false,
                "reason should mention kRatio: \(reason ?? "nil")"
            )
        }
    }

    func testKRatioGreaterThan1IsAllowedWhenFourSlotDisabled() {
        // With the four-slot cache disabled the kRatio > 1 model falls back to
        // the stock single-slot cache and must not be rejected.
        XCTAssertNil(
            MotifMLXModel.fourSlotCacheUnsupportedReason(mode: .disabled, kRatio: 2)
        )
    }
}
