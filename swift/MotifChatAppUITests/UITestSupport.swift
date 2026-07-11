import XCTest

/// Accessibility identifiers exposed by the app for UI automation. Keep in sync
/// with the `.accessibilityIdentifier(...)` modifiers in ContentView.swift.
enum A11y {
    // The sidebar has no "Chat" row: the chat pane is the default detail and
    // File > New Chat (⌘N) navigates back to it — see openChat(_:).
    static let sidebarRuntime = "motif.sidebar.runtime"
    static let input = "motif.chat.input"
    static let send = "motif.chat.send"
    static let stop = "motif.chat.stop"
    static let generating = "motif.chat.generating"
    static let error = "motif.chat.error"
    static let bubbleAssistant = "motif.chat.bubble.assistant"
    static let messageText = "motif.chat.message.text"
    static let reasoning = "motif.chat.reasoning"
    static let runtimeEndpoint = "motif.runtime.endpoint"
    static let backendPicker = "motif.runtime.backendPicker"
    static let thinkPicker = "motif.runtime.thinkPicker"
    static let runtimeStatus = "motif.runtime.status"
    static let maxTokens = "motif.runtime.maxTokens"
}

/// Launch arguments/environment understood by the app in UI-test builds.
/// Contract documented in docs/swift-app-smoke.md.
enum LaunchFlag {
    static let fakeBackend = "-UITestFakeBackend"
    static let defaultsSuite = "-UITestDefaultsSuite"
    static let resetDefaults = "-UITestResetDefaults"
}

@MainActor
extension XCUIApplication {
    /// Configures and launches the app for a deterministic UI test.
    ///
    /// - fakeBackend: swap in the in-process deterministic streaming backend
    ///   (no model, no network).
    /// - defaultsSuite / resetDefaults: isolate persisted settings from the
    ///   developer's real preferences (see ChatStore.defaults).
    /// - fakeConfig: MOTIF_UITEST_FAKE_* environment overrides for the fake
    ///   backend's pacing/content.
    @discardableResult
    func launchForUITest(
        fakeBackend: Bool = false,
        defaultsSuite: String? = nil,
        resetDefaults: Bool = false,
        fakeConfig: [String: String] = [:]
    ) -> XCUIApplication {
        var args: [String] = []
        if fakeBackend { args.append(LaunchFlag.fakeBackend) }
        if let suite = defaultsSuite {
            args.append(LaunchFlag.defaultsSuite)
            args.append(suite)
            if resetDefaults { args.append(LaunchFlag.resetDefaults) }
        }
        launchArguments = args
        for (k, v) in fakeConfig { launchEnvironment[k] = v }
        launch()
        // Ensure the window is actually up before any query, so the first
        // sidebar/element lookup isn't racing app startup.
        _ = wait(for: .runningForeground, timeout: 30)
        return self
    }
}

@MainActor
extension XCUIElement {
    /// Focus a macOS text field, clear it, and type new text.
    func clearAndType(_ text: String) {
        clear()
        typeText(text)
    }

    /// Focus a macOS text field and clear its contents.
    func clear() {
        click()
        typeKey("a", modifierFlags: .command)
        typeKey(.delete, modifierFlags: [])
    }

    /// Focus and type without clearing.
    func focusAndType(_ text: String) {
        click()
        typeText(text)
    }
}

@MainActor
extension XCTestCase {
    /// Returns the first element with `id` of any type (SwiftUI on macOS maps a
    /// given modifier to varying XCUIElement types, so query broadly).
    func element(_ app: XCUIApplication, _ id: String) -> XCUIElement {
        app.descendants(matching: .any).matching(identifier: id).firstMatch
    }

    /// Assert an element exists within `timeout`, failing with a clear message.
    @discardableResult
    func requireExists(
        _ e: XCUIElement,
        _ label: String,
        timeout: TimeInterval = 10,
        file: StaticString = #filePath,
        line: UInt = #line
    ) -> Bool {
        let ok = e.waitForExistence(timeout: timeout)
        XCTAssertTrue(ok, "Expected element to exist: \(label)", file: file, line: line)
        return ok
    }

    /// Wait until at least one element with `id` has a label containing `substring`.
    /// Polls the accessibility tree via a predicate expectation — no fixed sleeps.
    @discardableResult
    func waitForLabel(
        _ app: XCUIApplication,
        id: String,
        contains substring: String,
        timeout: TimeInterval = 15,
        file: StaticString = #filePath,
        line: UInt = #line
    ) -> Bool {
        // SwiftUI exposes text content via `value` on some elements and `label`
        // on others, so match either.
        let query = app.descendants(matching: .any)
            .matching(NSPredicate(
                format: "identifier == %@ AND (label CONTAINS %@ OR value CONTAINS %@)",
                id, substring, substring
            ))
        let exp = expectation(
            for: NSPredicate(format: "count > 0"),
            evaluatedWith: query,
            handler: nil
        )
        let result = XCTWaiter().wait(for: [exp], timeout: timeout)
        let ok = result == .completed
        XCTAssertTrue(
            ok,
            "Expected an element id=\(id) whose label contains \"\(substring)\"",
            file: file,
            line: line
        )
        return ok
    }

    /// Wait until at least one element ANYWHERE has a **label** containing
    /// `substring`. Used for message-bubble content: SwiftUI's
    /// `.accessibilityElement(children: .combine)` concatenates the bubble text
    /// into the element's `label`. Matches on `label` only — `value CONTAINS`
    /// over `.any` crashes the automation session on numeric-valued elements
    /// (e.g. progress indicators), whereas `label` is always a string.
    @discardableResult
    func waitForAnyLabel(
        _ app: XCUIApplication,
        contains substring: String,
        timeout: TimeInterval = 15,
        file: StaticString = #filePath,
        line: UInt = #line
    ) -> Bool {
        let query = app.descendants(matching: .any)
            .matching(NSPredicate(format: "label CONTAINS %@", substring))
        let exp = expectation(for: NSPredicate(format: "count > 0"), evaluatedWith: query)
        let ok = XCTWaiter().wait(for: [exp], timeout: timeout) == .completed
        XCTAssertTrue(ok, "Expected some element whose label contains \"\(substring)\"", file: file, line: line)
        return ok
    }

    /// Wait until at least one **static text** has a `value` containing
    /// `substring`. Plain SwiftUI `Text` (e.g. the expanded reasoning-disclosure
    /// body) exposes its content via `value`, not `label`. Restricted to
    /// `staticTexts` so `value CONTAINS` is only ever run against string values.
    @discardableResult
    func waitForStaticTextValue(
        _ app: XCUIApplication,
        contains substring: String,
        timeout: TimeInterval = 15,
        file: StaticString = #filePath,
        line: UInt = #line
    ) -> Bool {
        let query = app.staticTexts
            .matching(NSPredicate(format: "value CONTAINS %@", substring))
        let exp = expectation(for: NSPredicate(format: "count > 0"), evaluatedWith: query)
        let ok = XCTWaiter().wait(for: [exp], timeout: timeout) == .completed
        XCTAssertTrue(ok, "Expected a static text whose value contains \"\(substring)\"", file: file, line: line)
        return ok
    }

    /// Wait until text containing `substring` appears ANYWHERE, checking every
    /// exposure route SwiftUI uses: static-text `value`, text-view `value`, and
    /// combined-element `label`. Each route is individually crash-safe (`value
    /// CONTAINS` only ever runs against string-valued element types).
    @discardableResult
    func waitForAnyText(
        _ app: XCUIApplication,
        contains substring: String,
        timeout: TimeInterval = 15,
        file: StaticString = #filePath,
        line: UInt = #line
    ) -> Bool {
        let valuePred = NSPredicate(format: "value CONTAINS %@", substring)
        let labelPred = NSPredicate(format: "label CONTAINS %@", substring)
        let appears = NSPredicate { _, _ in
            if app.staticTexts.matching(valuePred).count > 0 { return true }
            if app.textViews.matching(valuePred).count > 0 { return true }
            return app.descendants(matching: .any).matching(labelPred).count > 0
        }
        let exp = XCTNSPredicateExpectation(predicate: appears, object: app)
        let ok = XCTWaiter().wait(for: [exp], timeout: timeout) == .completed
        XCTAssertTrue(ok, "Expected text containing \"\(substring)\" to appear", file: file, line: line)
        return ok
    }

    /// Expand a SwiftUI DisclosureGroup. The element is a DisclosureTriangle
    /// whose accessibility `value` is 0 (collapsed) / 1 (expanded); a single
    /// center-click can land on the label without toggling, so click-and-verify
    /// with retries until the value flips.
    func expandDisclosure(
        _ disclosure: XCUIElement,
        file: StaticString = #filePath,
        line: UInt = #line
    ) {
        let isExpanded = NSPredicate { obj, _ in
            guard let el = obj as? XCUIElement else { return false }
            return "\(el.value ?? "")" == "1"
        }
        for _ in 0..<4 {
            if "\(disclosure.value ?? "")" == "1" { return }
            disclosure.click()
            let exp = XCTNSPredicateExpectation(predicate: isExpanded, object: disclosure)
            if XCTWaiter().wait(for: [exp], timeout: 2) == .completed { return }
        }
        XCTFail("Could not expand disclosure \(disclosure.identifier)", file: file, line: line)
    }

    /// Wait until NO element with `id` exists (e.g. an error or disclosure that
    /// must be absent). Returns true if the condition holds within `timeout`.
    func expectAbsent(
        _ app: XCUIApplication,
        id: String,
        timeout: TimeInterval = 5
    ) -> Bool {
        let query = app.descendants(matching: .any).matching(identifier: id)
        let exp = expectation(for: NSPredicate(format: "count == 0"), evaluatedWith: query)
        return XCTWaiter().wait(for: [exp], timeout: timeout) == .completed
    }

    /// Navigate the sidebar to the Runtime pane.
    func openRuntime(_ app: XCUIApplication, file: StaticString = #filePath, line: UInt = #line) {
        let runtime = element(app, A11y.sidebarRuntime)
        requireExists(runtime, "sidebar Runtime row", file: file, line: line)
        runtime.click()
    }

    /// Navigate to the Chat pane. The sidebar has no "Chat" row — the chat pane
    /// is the default detail, and File > New Chat (⌘N) creates/selects a
    /// conversation, which deterministically routes the detail back to ChatView.
    func openChat(_ app: XCUIApplication, file: StaticString = #filePath, line: UInt = #line) {
        app.typeKey("n", modifierFlags: .command)
        requireExists(element(app, A11y.input), "chat input after ⌘N", file: file, line: line)
    }

    /// Type a prompt in the Chat input and click Send.
    func sendPrompt(_ app: XCUIApplication, _ text: String, file: StaticString = #filePath, line: UInt = #line) {
        let input = element(app, A11y.input)
        requireExists(input, "chat input field", file: file, line: line)
        input.focusAndType(text)
        let send = app.buttons[A11y.send]
        requireExists(send, "Send button", file: file, line: line)
        XCTAssertTrue(send.isEnabled, "Send should be enabled once the prompt is non-empty", file: file, line: line)
        send.click()
    }

    /// Select a value in a macOS SwiftUI Picker (pop-up button) by its visible title.
    func selectPickerValue(
        _ app: XCUIApplication,
        pickerId: String,
        title: String,
        file: StaticString = #filePath,
        line: UInt = #line
    ) {
        let picker = element(app, pickerId)
        requireExists(picker, "picker \(pickerId)", file: file, line: line)
        picker.click()
        let item = app.menuItems[title]
        if requireExists(item, "menu item \"\(title)\"", timeout: 5, file: file, line: line) {
            item.click()
        }
    }
}
