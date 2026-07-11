import XCTest

/// LIVE end-to-end tool round-trip through the GUI against the real MLX native
/// backend and a converted checkpoint: model -> app parses tool_call -> app
/// executes the builtin calculator -> tool result fed back -> model continues.
///
/// Opt-in and slow: requires the MLX build and ~/.models/motif-2.6b-mlx-q4, and
/// runs only under the MotifChatMLX scheme (see
/// scripts/ui_test_swift_chat_app_mlx.sh). The 2.6B checkpoint is used because a
/// tool round-trip is a few seconds; the 12.7B reasoning model spends minutes
/// per turn and is not practical to drive from XCUITest.
///
/// The backend and demo-tools state are preset via launch flags rather than
/// driven through the Runtime pane's Picker and the tools toggle. Driving a
/// SwiftUI pop-up menu is unreliable on a developer's live desktop: when another
/// app (Telegram, Slack, …) holds macOS key-window focus, the Picker's NSMenu
/// opens and instantly closes, so the menu item is never hittable. Presetting
/// keeps this test about the *model round-trip*, not window-manager luck.
@MainActor
final class LiveToolRoundTripUITests: XCTestCase {
    func testCalculatorToolRoundTripWithRealModel() {
        // Launch straight into the native MLX backend (checkpoint defaults to
        // ~/.models/motif-2.6b-mlx-q4) with the demo tools already enabled. The
        // chat pane is the default detail, so no navigation is needed.
        let app = XCUIApplication().launchForUITest(
            defaultsSuite: "io.junho.motif.uitest.mlx.tools",
            resetDefaults: true,
            fakeConfig: ["MOTIF_UITEST_DEBUG_LOG": "/tmp/motif_live_toolloop.log"],
            extraArguments: [
                LaunchFlag.backendMode, "nativeMLX",
                LaunchFlag.enableTools, "1",
            ]
        )
        // Bring our window frontmost so keyboard input reaches the chat field
        // even if another app launched in front of the test runner.
        app.activate()

        // Ask something that needs the calculator so the model emits a tool_call.
        sendPrompt(app, "What is 37 * 41? Use the calculator tool, then state the result.")

        // The real model emits a tool_call, the app executes the safe calculator
        // (37 * 41 = 1517), and the result lands in the transcript as a
        // tool-result card. Generous timeout: the first turn loads the checkpoint.
        waitForAnyText(app, contains: "1517", timeout: 300)

        // The loop then continues into a final answer and the turn settles.
        requireExists(app.buttons[A11y.send], "Send returns after the tool round-trip", timeout: 180)
    }
}
