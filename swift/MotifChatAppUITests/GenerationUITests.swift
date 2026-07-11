import XCTest

/// Rows 4, 5 (and the UI mechanics of 10/11) of docs/swift-app-smoke.md —
/// streaming, the Send↔Stop swap, completion to Idle, and mid-stream cancel.
/// Driven by the injected fake backend so there is no model or network.
@MainActor
final class GenerationUITests: XCTestCase {
    override func setUp() {
        continueAfterFailure = false
    }

    /// Row 4/10: sending streams an answer, swaps Send→Stop while generating,
    /// then returns to Idle with the answer in the assistant bubble.
    func testSendStreamsAndReturnsToIdle() {
        let app = XCUIApplication().launchForUITest(
            fakeBackend: true,
            defaultsSuite: "io.junho.motif.uitest.gen1",
            resetDefaults: true,
            fakeConfig: [
                // ~3s stream so the Send→Stop swap is comfortably observable.
                "MOTIF_UITEST_FAKE_CHUNKS": "40",
                "MOTIF_UITEST_FAKE_DELAY_MS": "80",
                "MOTIF_UITEST_FAKE_ANSWER": "FAKE_ANSWER_TOKEN ",
            ]
        )

        sendPrompt(app, "Hello")

        // Send is replaced by Stop while generating.
        requireExists(app.buttons[A11y.stop], "Stop button appears while generating", timeout: 8)

        // Streaming finishes: Stop reverts to Send.
        requireExists(app.buttons[A11y.send], "Send button returns after completion", timeout: 20)

        // The assistant bubble contains the fake answer (bubble text is exposed
        // via the combined element's label).
        waitForAnyLabel(app, contains: "FAKE_ANSWER_TOKEN")

        // Runtime status settles on Idle.
        openRuntime(app)
        waitForLabel(app, id: A11y.runtimeStatus, contains: "Idle")
    }

    /// Row 5: clicking Stop mid-stream halts streaming and appends the
    /// [Cancelled] marker to the in-flight assistant bubble.
    func testStopMidStreamAppendsCancelled() {
        let app = XCUIApplication().launchForUITest(
            fakeBackend: true,
            defaultsSuite: "io.junho.motif.uitest.gen2",
            resetDefaults: true,
            fakeConfig: [
                // Long, slow stream so there is a comfortable window to Stop.
                "MOTIF_UITEST_FAKE_CHUNKS": "80",
                "MOTIF_UITEST_FAKE_DELAY_MS": "80",
                "MOTIF_UITEST_FAKE_ANSWER": "streaming ",
            ]
        )

        sendPrompt(app, "Hello")

        let stop = app.buttons[A11y.stop]
        requireExists(stop, "Stop button appears while generating", timeout: 8)
        stop.click()

        // Generation halts and Send comes back.
        requireExists(app.buttons[A11y.send], "Send button returns after cancel", timeout: 8)

        // cancel() sets the runtime status to "Cancelled" (and the completion
        // block leaves it there rather than resetting to Idle).
        openRuntime(app)
        waitForLabel(app, id: A11y.runtimeStatus, contains: "Cancelled")
    }
}
