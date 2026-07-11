import XCTest

/// Rows 12, 13, 14 of docs/swift-app-smoke.md — think-mode routing. The fake
/// backend mirrors the real backends' contract: captured → `.reasoning` events
/// (disclosure), visible → inline `<think>` in the answer text, hidden → nothing.
///
/// Each test starts from an isolated, reset defaults suite so the transcript is
/// empty (one assistant bubble) and settings start at their defaults.
@MainActor
final class ThinkModeUITests: XCTestCase {
    override func setUp() {
        continueAfterFailure = false
    }

    /// Row 13: Thinking = hidden shows only the final answer — no reasoning
    /// disclosure appears.
    func testHiddenThinkingShowsNoReasoning() {
        let app = XCUIApplication().launchForUITest(
            fakeBackend: true,
            defaultsSuite: "io.junho.motif.uitest.think.hidden",
            resetDefaults: true,
            fakeConfig: [
                "MOTIF_UITEST_FAKE_CHUNKS": "8",
                "MOTIF_UITEST_FAKE_DELAY_MS": "30",
                "MOTIF_UITEST_FAKE_ANSWER": "HIDDENANSWER ",
                "MOTIF_UITEST_FAKE_REASONING": "SHOULDNOTAPPEAR",
            ]
        )

        openRuntime(app)
        selectPickerValue(app, pickerId: A11y.thinkPicker, title: "Hidden")

        openChat(app)
        sendPrompt(app, "Hello")

        waitForAnyLabel(app, contains: "HIDDENANSWER")
        XCTAssertTrue(
            expectAbsent(app, id: A11y.reasoning),
            "No captured-reasoning disclosure should appear in hidden mode"
        )
    }

    /// Row 12: Thinking = visible streams the `<think>` block inline in the
    /// answer bubble (not the captured disclosure).
    func testVisibleThinkingInlinesReasoning() {
        let app = XCUIApplication().launchForUITest(
            fakeBackend: true,
            defaultsSuite: "io.junho.motif.uitest.think.visible",
            resetDefaults: true,
            fakeConfig: [
                "MOTIF_UITEST_FAKE_CHUNKS": "8",
                "MOTIF_UITEST_FAKE_DELAY_MS": "30",
                "MOTIF_UITEST_FAKE_ANSWER": "VISIBLEANSWER ",
                "MOTIF_UITEST_FAKE_REASONING": "VISIBLEREASONINGSENTINEL",
            ]
        )

        openRuntime(app)
        selectPickerValue(app, pickerId: A11y.thinkPicker, title: "Visible")

        openChat(app)
        sendPrompt(app, "Hello")

        // Reasoning + answer both land inline in the message body (label)...
        waitForAnyLabel(app, contains: "VISIBLEREASONINGSENTINEL")
        waitForAnyLabel(app, contains: "VISIBLEANSWER")
        // ...and NOT via the captured-reasoning disclosure.
        XCTAssertTrue(
            expectAbsent(app, id: A11y.reasoning),
            "Visible mode should not use the captured-reasoning disclosure"
        )
    }

    /// Row 14: Thinking = captured routes reasoning into the "Captured
    /// reasoning" disclosure — auto-expanded (raw text visible) while the model
    /// is thinking, collapsed to a summary once the answer streams, and still
    /// present afterwards.
    ///
    /// The disclosure is asserted during its auto-expanded thinking window
    /// (via the fake backend's preface delay) rather than by clicking it open
    /// afterwards: XCUITest clicks on this SwiftUI DisclosureGroup's label do
    /// not reliably toggle it (verified — its AX value stays 0 across repeated
    /// clicks), so the manual expand-after-the-fact gesture stays a manual
    /// checklist step.
    func testCapturedThinkingShowsReasoningDisclosure() {
        let app = XCUIApplication().launchForUITest(
            fakeBackend: true,
            defaultsSuite: "io.junho.motif.uitest.think.captured",
            resetDefaults: true,
            fakeConfig: [
                "MOTIF_UITEST_FAKE_CHUNKS": "6",
                "MOTIF_UITEST_FAKE_DELAY_MS": "30",
                // 4s thinking window between the reasoning event and the first
                // answer chunk, during which the disclosure is auto-expanded.
                "MOTIF_UITEST_FAKE_PREFACE_DELAY_MS": "4000",
                "MOTIF_UITEST_FAKE_ANSWER": "CAPTUREDANSWER ",
                "MOTIF_UITEST_FAKE_REASONING": "CAPTUREDREASONINGSENTINEL",
            ]
        )

        openRuntime(app)
        selectPickerValue(app, pickerId: A11y.thinkPicker, title: "Captured")

        openChat(app)
        sendPrompt(app, "Hello")

        // While thinking, the disclosure is auto-expanded and the raw reasoning
        // text is on screen.
        let disclosure = element(app, A11y.reasoning)
        requireExists(disclosure, "captured-reasoning disclosure while thinking")
        waitForAnyText(app, contains: "CAPTUREDREASONINGSENTINEL", timeout: 10)

        // The answer then streams into the bubble (reasoning is NOT inline).
        waitForAnyLabel(app, contains: "CAPTUREDANSWER")

        // After the turn ends the disclosure collapses to its summary but stays
        // in the transcript.
        requireExists(app.buttons[A11y.send], "Send button after completion", timeout: 15)
        requireExists(element(app, A11y.reasoning), "collapsed disclosure persists")
    }

    /// Captured reasoning is persisted with the conversation, so the disclosure
    /// is still present after quitting and relaunching (the disclosure only
    /// renders when reasoning for that message is non-empty).
    func testCapturedReasoningSurvivesRelaunch() {
        let suite = "io.junho.motif.uitest.think.persist"
        let config = [
            "MOTIF_UITEST_FAKE_CHUNKS": "6",
            "MOTIF_UITEST_FAKE_DELAY_MS": "20",
            "MOTIF_UITEST_FAKE_ANSWER": "PERSISTANSWER ",
            "MOTIF_UITEST_FAKE_REASONING": "PERSISTREASONINGSENTINEL",
        ]
        let app = XCUIApplication()

        // First launch: fresh suite, capture a reasoning turn.
        app.launchForUITest(fakeBackend: true, defaultsSuite: suite, resetDefaults: true, fakeConfig: config)
        openRuntime(app)
        selectPickerValue(app, pickerId: A11y.thinkPicker, title: "Captured")
        openChat(app)
        sendPrompt(app, "Hello")
        waitForAnyLabel(app, contains: "PERSISTANSWER")
        requireExists(element(app, A11y.reasoning), "reasoning disclosure after generation")
        requireExists(app.buttons[A11y.send], "Send returns before relaunch", timeout: 15)

        app.terminate()

        // Relaunch the same suite WITHOUT reset and WITHOUT generating again: the
        // disclosure must reappear purely from persisted reasoning.
        app.launchForUITest(defaultsSuite: suite, resetDefaults: false)
        waitForAnyLabel(app, contains: "PERSISTANSWER")
        requireExists(element(app, A11y.reasoning), "captured reasoning survives relaunch")
    }
}
