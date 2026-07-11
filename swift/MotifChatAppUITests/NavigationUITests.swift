import XCTest

/// Rows 1, 2, 15 of docs/swift-app-smoke.md — launch, sidebar navigation, and
/// settings persistence. None of these need a model or network.
@MainActor
final class NavigationUITests: XCTestCase {
    override func setUp() {
        continueAfterFailure = false
    }

    /// Row 1: launching shows the Chat/Runtime sidebar and an empty chat with the
    /// "Ask Motif…" input field.
    func testLaunchShowsSidebarAndInput() {
        let app = XCUIApplication().launchForUITest(
            defaultsSuite: "io.junho.motif.uitest.nav.launch",
            resetDefaults: true
        )

        // The sidebar shows Runtime + conversation history; the chat pane
        // itself is the default detail at launch.
        requireExists(element(app, A11y.sidebarRuntime), "sidebar Runtime row")
        requireExists(element(app, A11y.input), "chat input field")
    }

    /// Row 2: selecting "Native MLX checkpoint" in the Runtime backend picker
    /// reveals the converted-model-directory field and its Choose… button.
    func testSelectingNativeMLXRevealsDirectoryControls() {
        let app = XCUIApplication().launchForUITest(
            defaultsSuite: "io.junho.motif.uitest.nav.native",
            resetDefaults: true
        )

        openRuntime(app)
        selectPickerValue(app, pickerId: A11y.backendPicker, title: "Native MLX checkpoint")

        // The Choose… button only exists in the nativeMLX branch of the Backend
        // section, so its presence confirms the picker switched the sub-form.
        requireExists(app.buttons["Choose…"], "Choose… directory button")
    }

    /// Row 15: non-default settings survive quit + relaunch. Uses an isolated
    /// UserDefaults suite so the developer's real preferences are never touched.
    func testSettingsPersistAcrossRelaunch() {
        let suite = "io.junho.motif.uitest.persistence"
        let app = XCUIApplication()

        // First launch: start from a clean suite, then flip Thinking to a
        // non-default value (default is Captured).
        app.launchForUITest(defaultsSuite: suite, resetDefaults: true)
        openRuntime(app)
        selectPickerValue(app, pickerId: A11y.thinkPicker, title: "Visible")

        let thinkAfterChange = element(app, A11y.thinkPicker)
        requireExists(thinkAfterChange, "Thinking picker")
        XCTAssertEqual((thinkAfterChange.value as? String) ?? "", "Visible")

        app.terminate()

        // Relaunch against the same suite WITHOUT reset — the change must persist.
        app.launchForUITest(defaultsSuite: suite, resetDefaults: false)
        openRuntime(app)

        let thinkAfterRelaunch = element(app, A11y.thinkPicker)
        requireExists(thinkAfterRelaunch, "Thinking picker after relaunch")
        XCTAssertEqual(
            (thinkAfterRelaunch.value as? String) ?? "",
            "Visible",
            "Thinking mode should persist across relaunch"
        )
    }
}
