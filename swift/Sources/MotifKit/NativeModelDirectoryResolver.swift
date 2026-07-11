import Foundation

/// Typed errors surfaced while resolving the native model directory. These are
/// intentionally distinct so the UI can tell "re-select it" (resolution/access
/// failure) apart from "you never picked one" (empty path).
public enum NativeModelDirectoryError: Error, Equatable, Sendable {
    /// No directory has been configured (empty/whitespace path and no bookmark).
    case emptyPath
    /// A bookmark was stored but could not be resolved to a URL.
    case unresolvableBookmark(String)
    /// A bookmark resolved to a URL but security-scoped access could not be started.
    case inaccessibleScopedResource
}

extension NativeModelDirectoryError: LocalizedError {
    public var errorDescription: String? {
        switch self {
        case .emptyPath:
            return "Native MLX model directory cannot be empty."
        case .unresolvableBookmark(let detail):
            return "Could not resolve the saved native model directory. Re-select it. (\(detail))"
        case .inaccessibleScopedResource:
            return "Could not obtain access to the saved native model directory. Re-select it."
        }
    }
}

/// An active (possibly no-op) access grant to the native model directory.
///
/// `stop` must be invoked exactly once, after every file read for the load has
/// completed. For the unsandboxed path-based fallback `stop` is a no-op because
/// no scoped access was started.
public struct NativeDirectoryAccessGrant {
    public let url: URL
    public let stop: () -> Void

    public init(url: URL, stop: @escaping () -> Void) {
        self.url = url
        self.stop = stop
    }
}

/// Owns a native-directory access grant for the lifetime of a lazy-loading
/// backend. Explicit and deinitialization-driven release share the same
/// exactly-once path, so callers may stop the lease defensively without risking
/// an unbalanced security-scope release.
public final class NativeDirectoryAccessLease: @unchecked Sendable {
    public let url: URL

    private let lock = NSLock()
    private var grant: NativeDirectoryAccessGrant?

    public init(grant: NativeDirectoryAccessGrant) {
        self.url = grant.url
        self.grant = grant
    }

    /// Releases the owned grant at most once. The callback runs outside the
    /// lock so an injected stop implementation cannot deadlock by re-entering
    /// lease-owned code.
    public func stop() {
        let grantToStop: NativeDirectoryAccessGrant?
        lock.lock()
        grantToStop = grant
        grant = nil
        lock.unlock()

        grantToStop?.stop()
    }

    deinit {
        stop()
    }
}

/// Pure decision logic for acquiring access to the native model checkpoint
/// directory.
///
/// The actual Foundation security-scoped-bookmark calls and `UserDefaults`
/// persistence are injected as closures so the *decision and fallback* behavior
/// can be unit-tested headless, without a real sandboxed directory:
///
///   - When a bookmark is stored it is preferred (works under App Sandbox and
///     across relaunch); scoped access is started and the returned grant's
///     `stop` releases it.
///   - When resolution reports the bookmark is stale, a fresh bookmark is
///     re-persisted; the just-started access still covers the current load.
///   - When no bookmark is stored, the resolver falls back to the plain path
///     (unsandboxed/dev use) and returns a no-op grant.
///   - Resolution/access failures surface a typed `NativeModelDirectoryError`
///     instead of crashing.
public struct NativeModelDirectoryResolver {
    /// Loads the persisted bookmark data, or `nil` when none is stored.
    public var loadBookmark: () -> Data?
    /// Resolves bookmark data to a URL, reporting staleness via the inout flag.
    /// Throws when the bookmark cannot be resolved at all.
    public var resolveBookmark: (Data, inout Bool) throws -> URL
    /// Starts security-scoped access for the resolved URL; returns `false` on
    /// failure. Paired with `stopAccess`.
    public var startAccess: (URL) -> Bool
    /// Stops security-scoped access previously started for the URL.
    public var stopAccess: (URL) -> Void
    /// Re-persists a fresh bookmark for the URL (used when the prior one was stale).
    public var persistBookmark: (URL) -> Void
    /// Resolves the plain on-disk path fallback when no bookmark is available.
    public var resolvePath: () throws -> URL

    public init(
        loadBookmark: @escaping () -> Data?,
        resolveBookmark: @escaping (Data, inout Bool) throws -> URL,
        startAccess: @escaping (URL) -> Bool,
        stopAccess: @escaping (URL) -> Void,
        persistBookmark: @escaping (URL) -> Void,
        resolvePath: @escaping () throws -> URL
    ) {
        self.loadBookmark = loadBookmark
        self.resolveBookmark = resolveBookmark
        self.startAccess = startAccess
        self.stopAccess = stopAccess
        self.persistBookmark = persistBookmark
        self.resolvePath = resolvePath
    }

    /// Acquires access to the native model directory, preferring a stored
    /// security-scoped bookmark and falling back to the plain path.
    public func acquire() throws -> NativeDirectoryAccessGrant {
        if let grant = try scopedAccessFromBookmark() {
            return grant
        }
        // No bookmark stored (non-sandboxed/dev use, or reset state): fall back
        // to the path. No scoped access is needed when the app is unsandboxed,
        // so the grant's `stop` is a no-op.
        return NativeDirectoryAccessGrant(url: try resolvePath(), stop: {})
    }

    private func scopedAccessFromBookmark() throws -> NativeDirectoryAccessGrant? {
        guard let bookmark = loadBookmark() else {
            return nil
        }

        var isStale = false
        let url: URL
        do {
            url = try resolveBookmark(bookmark, &isStale)
        } catch {
            throw NativeModelDirectoryError.unresolvableBookmark(error.localizedDescription)
        }

        guard startAccess(url) else {
            throw NativeModelDirectoryError.inaccessibleScopedResource
        }

        if isStale {
            // Re-persist a fresh bookmark for the resolved URL; failure is
            // non-fatal because the just-started access still covers this load.
            persistBookmark(url)
        }

        return NativeDirectoryAccessGrant(url: url, stop: { stopAccess(url) })
    }
}
