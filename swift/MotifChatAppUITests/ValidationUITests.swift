import XCTest

/// Rows 8, 9 of docs/swift-app-smoke.md — backend-construction validation
/// errors. These run in the lightweight (no-MLX) build with no fake backend, so
/// the real error paths in ChatStore.buildBackend() are exercised.
@MainActor
final class ValidationUITests: XCTestCase {
    override func setUp() {
        continueAfterFailure = false
    }

    /// Row 8: with the app built WITHOUT MOTIFKIT_ENABLE_MLX, selecting Native
    /// MLX and sending surfaces the "not compiled" error. This is the canonical
    /// deterministic test for the lightweight build.
    func testNativeMLXNotCompiledShowsError() {
        let app = XCUIApplication().launchForUITest(
            defaultsSuite: "io.junho.motif.uitest.validation.native",
            resetDefaults: true
        )

        openRuntime(app)
        selectPickerValue(app, pickerId: A11y.backendPicker, title: "Native MLX checkpoint")

        openChat(app)
        sendPrompt(app, "Hello")

        waitForLabel(app, id: A11y.error, contains: "not compiled into this app build")
    }

    /// Row 9: an invalid OpenAI-compatible endpoint surfaces the
    /// invalid-endpoint error before any network call.
    ///
    /// `validatedEndpointURL` requires an absolute http/https URL with a host,
    /// so plain garbage like "not a url" (no scheme) is rejected app-side even
    /// though macOS 14+ Foundation's lenient URL parser would percent-encode it
    /// into a parseable URL.
    func testInvalidEndpointURLShowsError() {
        // Reset defaults so the backend starts at its default (OpenAI-compatible)
        // — otherwise a persisted nativeMLX mode from a prior run hides the
        // endpoint field entirely.
        let app = XCUIApplication().launchForUITest(
            defaultsSuite: "io.junho.motif.uitest.validation.endpoint",
            resetDefaults: true
        )

        // Edit the endpoint via the Runtime pane's Form field (the toolbar copy
        // can collapse into an overflow menu at narrow widths).
        openRuntime(app)
        let endpoint = element(app, A11y.runtimeEndpoint)
        requireExists(endpoint, "runtime endpoint field")
        endpoint.clearAndType("not a url")

        openChat(app)
        sendPrompt(app, "Hi")

        waitForLabel(app, id: A11y.error, contains: "Endpoint must be an absolute HTTP or HTTPS URL")
    }
}
