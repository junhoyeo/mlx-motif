import XCTest
@testable import MotifKit

/// Tests the decision/fallback logic for acquiring access to the native model
/// checkpoint directory. The resolver's Foundation/sandbox operations are
/// injected, so these run headless without a real security-scoped directory:
/// we assert the *wiring* (which closures fire, in what order, and which typed
/// error surfaces) rather than real sandbox behavior.
final class NativeModelDirectoryResolverTests: XCTestCase {
    private let bookmarkURL = URL(fileURLWithPath: "/tmp/motif-bookmark-dir", isDirectory: true)
    private let pathURL = URL(fileURLWithPath: "/tmp/motif-path-dir", isDirectory: true)
    private let sampleBookmark = Data([0x01, 0x02, 0x03])

    // MARK: - Bookmark create -> resolve round-trip is wired

    func testBookmarkPresentResolvesAndStartsScopedAccess() throws {
        var didStartAccess = false
        var didResolvePath = false
        var persistedURLs: [URL] = []

        let resolver = NativeModelDirectoryResolver(
            loadBookmark: { self.sampleBookmark },
            resolveBookmark: { data, isStale in
                XCTAssertEqual(data, self.sampleBookmark)
                isStale = false
                return self.bookmarkURL
            },
            startAccess: { url in
                XCTAssertEqual(url, self.bookmarkURL)
                didStartAccess = true
                return true
            },
            stopAccess: { _ in },
            persistBookmark: { persistedURLs.append($0) },
            resolvePath: {
                didResolvePath = true
                return self.pathURL
            }
        )

        let grant = try resolver.acquire()

        XCTAssertEqual(grant.url, bookmarkURL, "Bookmark URL should win over the path fallback")
        XCTAssertTrue(didStartAccess, "Scoped access must be started for a resolved bookmark")
        XCTAssertFalse(didResolvePath, "Path fallback must not run when a bookmark resolves")
        XCTAssertTrue(persistedURLs.isEmpty, "A non-stale bookmark must not be re-persisted")
    }

    func testGrantStopReleasesScopedAccessForTheResolvedURL() throws {
        var stoppedURLs: [URL] = []
        let resolver = makeResolver(
            loadBookmark: { self.sampleBookmark },
            resolveBookmark: { _, isStale in isStale = false; return self.bookmarkURL },
            startAccess: { _ in true },
            stopAccess: { stoppedURLs.append($0) }
        )

        let grant = try resolver.acquire()
        XCTAssertTrue(stoppedURLs.isEmpty, "stop must not fire until the caller releases it")
        grant.stop()
        XCTAssertEqual(stoppedURLs, [bookmarkURL], "stop must release the resolved bookmark URL")
    }

    func testLeaseExplicitStopReleasesOwnedGrant() {
        let recorder = StopRecorder()
        var lease: NativeDirectoryAccessLease? = NativeDirectoryAccessLease(
            grant: NativeDirectoryAccessGrant(url: bookmarkURL) {
                recorder.recordStop()
            }
        )

        XCTAssertEqual(lease?.url, bookmarkURL)
        XCTAssertEqual(recorder.count, 0)
        lease?.stop()
        XCTAssertEqual(recorder.count, 1)

        lease = nil
        XCTAssertEqual(recorder.count, 1, "deinit after explicit stop must not release twice")
    }

    func testLeaseStopIsIdempotentUnderConcurrentCallers() {
        let recorder = StopRecorder()
        let lease = NativeDirectoryAccessLease(
            grant: NativeDirectoryAccessGrant(url: bookmarkURL) {
                recorder.recordStop()
            }
        )

        DispatchQueue.concurrentPerform(iterations: 64) { _ in
            lease.stop()
        }
        lease.stop()
        XCTAssertEqual(recorder.count, 1)
    }

    func testLeaseDeinitReleasesOwnedGrant() {
        let recorder = StopRecorder()
        weak var weakLease: NativeDirectoryAccessLease?
        var lease: NativeDirectoryAccessLease? = NativeDirectoryAccessLease(
            grant: NativeDirectoryAccessGrant(url: bookmarkURL) {
                recorder.recordStop()
            }
        )
        weakLease = lease

        lease = nil

        XCTAssertNil(weakLease)
        XCTAssertEqual(recorder.count, 1)
    }

    // MARK: - Missing-bookmark fallback to path-based behavior

    func testMissingBookmarkFallsBackToPathWithNoOpStop() throws {
        var didStartAccess = false
        let resolver = makeResolver(
            loadBookmark: { nil },
            startAccess: { _ in didStartAccess = true; return true },
            resolvePath: { self.pathURL }
        )

        let grant = try resolver.acquire()

        XCTAssertEqual(grant.url, pathURL, "With no bookmark we must use the path fallback")
        XCTAssertFalse(didStartAccess, "Path fallback must not start scoped access")
        // stop is a no-op for the unsandboxed fallback; calling it must not crash.
        grant.stop()
    }

    func testPathFallbackPropagatesEmptyPathError() {
        let resolver = makeResolver(
            loadBookmark: { nil },
            resolvePath: { throw NativeModelDirectoryError.emptyPath }
        )

        XCTAssertThrowsError(try resolver.acquire()) { error in
            XCTAssertEqual(error as? NativeModelDirectoryError, .emptyPath)
        }
    }

    // MARK: - Stale-bookmark re-resolve path

    func testStaleBookmarkReResolvesAndRePersists() throws {
        var persistedURLs: [URL] = []
        let resolver = makeResolver(
            loadBookmark: { self.sampleBookmark },
            resolveBookmark: { _, isStale in
                isStale = true
                return self.bookmarkURL
            },
            startAccess: { _ in true },
            persistBookmark: { persistedURLs.append($0) }
        )

        let grant = try resolver.acquire()

        XCTAssertEqual(grant.url, bookmarkURL)
        XCTAssertEqual(
            persistedURLs, [bookmarkURL],
            "A stale bookmark must be re-persisted for the freshly resolved URL"
        )
    }

    // MARK: - Resolution / access failures surface typed errors (no crash)

    func testResolutionFailureSurfacesUnresolvableBookmark() {
        struct Boom: Error, LocalizedError { var errorDescription: String? { "boom" } }
        var didStartAccess = false
        let resolver = makeResolver(
            loadBookmark: { self.sampleBookmark },
            resolveBookmark: { _, _ in throw Boom() },
            startAccess: { _ in didStartAccess = true; return true }
        )

        XCTAssertThrowsError(try resolver.acquire()) { error in
            XCTAssertEqual(error as? NativeModelDirectoryError, .unresolvableBookmark("boom"))
        }
        XCTAssertFalse(didStartAccess, "Access must not be started when resolution fails")
    }

    func testAccessStartFailureSurfacesInaccessibleError() {
        var didPersist = false
        let resolver = makeResolver(
            loadBookmark: { self.sampleBookmark },
            resolveBookmark: { _, isStale in isStale = true; return self.bookmarkURL },
            startAccess: { _ in false },
            persistBookmark: { _ in didPersist = true }
        )

        XCTAssertThrowsError(try resolver.acquire()) { error in
            XCTAssertEqual(error as? NativeModelDirectoryError, .inaccessibleScopedResource)
        }
        XCTAssertFalse(didPersist, "A stale bookmark must not be re-persisted if access cannot start")
    }

    // MARK: - Typed error descriptions

    func testTypedErrorsProvideUserFacingDescriptions() {
        XCTAssertEqual(
            (NativeModelDirectoryError.emptyPath as LocalizedError).errorDescription,
            "Native MLX model directory cannot be empty."
        )
        XCTAssertTrue(
            NativeModelDirectoryError.unresolvableBookmark("detail").errorDescription?
                .contains("detail") == true
        )
        XCTAssertNotNil(NativeModelDirectoryError.inaccessibleScopedResource.errorDescription)
    }

    // MARK: - Helpers

    /// Builds a resolver with sensible defaults so each test overrides only the
    /// closures it cares about.
    private func makeResolver(
        loadBookmark: @escaping () -> Data? = { nil },
        resolveBookmark: @escaping (Data, inout Bool) throws -> URL = { _, isStale in
            isStale = false
            return URL(fileURLWithPath: "/tmp/unused")
        },
        startAccess: @escaping (URL) -> Bool = { _ in true },
        stopAccess: @escaping (URL) -> Void = { _ in },
        persistBookmark: @escaping (URL) -> Void = { _ in },
        resolvePath: @escaping () throws -> URL = { URL(fileURLWithPath: "/tmp/motif-path-dir") }
    ) -> NativeModelDirectoryResolver {
        NativeModelDirectoryResolver(
            loadBookmark: loadBookmark,
            resolveBookmark: resolveBookmark,
            startAccess: startAccess,
            stopAccess: stopAccess,
            persistBookmark: persistBookmark,
            resolvePath: resolvePath
        )
    }
}

private final class StopRecorder: @unchecked Sendable {
    private let lock = NSLock()
    private var stopCount = 0

    var count: Int {
        lock.lock()
        defer { lock.unlock() }
        return stopCount
    }

    func recordStop() {
        lock.lock()
        stopCount += 1
        lock.unlock()
    }
}
