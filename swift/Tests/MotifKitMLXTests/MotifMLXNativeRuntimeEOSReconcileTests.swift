#if canImport(MLX) && canImport(MLXNN) && canImport(MLXLLM) && canImport(MLXLMCommon)
import Foundation
import MLX
import MLXLMCommon
@testable import MotifKitMLX
import XCTest

/// Minimal `KVCache` test double whose trimmability is configurable. Used to
/// exercise the EOS surplus reconciliation guard (Item 7) without a real model
/// — the production Motif caches are always trimmable, so a double is the only
/// way to drive the non-trimmable branch.
private final class FakeKVCache: KVCache {
    var offset: Int
    let trimmable: Bool
    private(set) var trimCallCount = 0
    private(set) var lastTrimmedBy = 0

    init(offset: Int, trimmable: Bool) {
        self.offset = offset
        self.trimmable = trimmable
    }

    var maxSize: Int? { nil }
    var state: [MLXArray] {
        get { [] }
        set { _ = newValue }
    }

    var metaState: [String] {
        get { [] }
        set { _ = newValue }
    }

    var isTrimmable: Bool { trimmable }

    func update(keys: MLXArray, values: MLXArray) -> (MLXArray, MLXArray) {
        (keys, values)
    }

    @discardableResult
    func trim(_ n: Int) -> Int {
        let trimmed = min(offset, n)
        offset -= trimmed
        trimCallCount += 1
        lastTrimmedBy = trimmed
        return trimmed
    }

    func makeMask(
        n: Int, windowSize: Int?, returnArray: Bool
    ) -> MLXFast.ScaledDotProductAttentionMaskMode {
        .none
    }

    func innerState() -> [MLXArray] { [] }
}

final class MotifMLXNativeRuntimeEOSReconcileTests: XCTestCase {
    override func tearDown() {
        MotifMLXNativeRuntime.assertsOnUnreconcilableEOSCache = true
        super.tearDown()
    }

    func testZeroSurplusIsReusableAndDoesNotTrim() {
        let cache = [FakeKVCache(offset: 5, trimmable: true), FakeKVCache(offset: 5, trimmable: true)]
        let reusable = MotifMLXNativeRuntime.reconcileEOSCacheSurplus(cache, surplus: 0)
        XCTAssertTrue(reusable)
        XCTAssertTrue(cache.allSatisfy { $0.trimCallCount == 0 })
        XCTAssertTrue(cache.allSatisfy { $0.offset == 5 })
    }

    func testTrimmableSurplusTrimsEveryLayerAndStaysReusable() {
        let cache = [FakeKVCache(offset: 6, trimmable: true), FakeKVCache(offset: 6, trimmable: true)]
        let reusable = MotifMLXNativeRuntime.reconcileEOSCacheSurplus(cache, surplus: 1)
        XCTAssertTrue(reusable)
        XCTAssertTrue(cache.allSatisfy { $0.trimCallCount == 1 })
        XCTAssertTrue(cache.allSatisfy { $0.lastTrimmedBy == 1 })
        XCTAssertTrue(cache.allSatisfy { $0.offset == 5 })
    }

    func testNonTrimmableSurplusInvalidatesReuseAndDoesNotTrim() {
        // Disable the debug assertionFailure so the release-fallback (log +
        // invalidate) path is exercised without trapping the test process.
        MotifMLXNativeRuntime.assertsOnUnreconcilableEOSCache = false
        let cache: [KVCache] = [
            FakeKVCache(offset: 6, trimmable: true),
            FakeKVCache(offset: 6, trimmable: false),
        ]
        let reusable = MotifMLXNativeRuntime.reconcileEOSCacheSurplus(cache, surplus: 1)
        XCTAssertFalse(reusable, "a non-trimmable layer must invalidate cross-turn reuse")
        // No layer is trimmed when the guard fires — the whole cache is dropped.
        for layer in cache {
            if let fake = layer as? FakeKVCache {
                XCTAssertEqual(fake.trimCallCount, 0)
                XCTAssertEqual(fake.offset, 6)
            }
        }
    }
}
#endif
