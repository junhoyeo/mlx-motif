import Foundation
import MotifKit
@testable import MotifChatApp
import XCTest

final class ChatInteractionPolicyTests: XCTestCase {
    func testStaleGenerationCannotFinishNewerGeneration() {
        var lifecycle = ChatGenerationLifecycle()
        let older = lifecycle.begin()
        let newer = lifecycle.begin()

        XCTAssertFalse(lifecycle.finish(id: older))
        XCTAssertEqual(lifecycle.activeID, newer)
        XCTAssertTrue(lifecycle.finish(id: newer))
        XCTAssertNil(lifecycle.activeID)
    }

    func testEndpointValidationRequiresAbsoluteHTTPURLWithHost() {
        XCTAssertEqual(
            ChatStore.validatedEndpointURL("  http://127.0.0.1:8080/v1  ")?.absoluteString,
            "http://127.0.0.1:8080/v1"
        )
        XCTAssertEqual(
            ChatStore.validatedEndpointURL("https://example.com/api")?.absoluteString,
            "https://example.com/api"
        )

        for invalid in ["", "not a url", "/v1", "localhost:8080/v1", "ftp://example.com/v1", "http:///v1"] {
            XCTAssertNil(ChatStore.validatedEndpointURL(invalid), "Expected invalid endpoint: \(invalid)")
        }
    }

    func testBookmarkLookupIsBoundToSelectedCanonicalPath() {
        let directoryA = URL(fileURLWithPath: "/tmp/models/A", isDirectory: true)
        let directoryB = URL(fileURLWithPath: "/tmp/models/B", isDirectory: true)
        let bookmarkA = Data([0x0A])
        let bookmarkB = Data([0x0B])
        let bookmarks = [
            ChatStore.canonicalModelDirectoryPath(directoryA): bookmarkA.base64EncodedString(),
            ChatStore.canonicalModelDirectoryPath(directoryB): bookmarkB.base64EncodedString(),
        ]

        XCTAssertEqual(ChatStore.bookmarkData(for: directoryB, in: bookmarks), bookmarkB)
        XCTAssertNil(
            ChatStore.bookmarkData(
                for: directoryB,
                in: [ChatStore.canonicalModelDirectoryPath(directoryA): bookmarkA.base64EncodedString()]
            ),
            "Selecting B must never reuse A's security-scoped bookmark"
        )
        XCTAssertFalse(ChatStore.bookmarkURL(directoryA, matches: directoryB))
        XCTAssertTrue(ChatStore.bookmarkURL(directoryB.standardizedFileURL, matches: directoryB))
    }

    func testTranscriptFollowPolicyCoalescesAndHonorsManualDisengagement() throws {
        let first = try XCTUnwrap(TranscriptScrollPolicy.request(
            isNearBottom: true,
            forced: false,
            id: UUID()
        ))
        let newer = try XCTUnwrap(TranscriptScrollPolicy.request(
            isNearBottom: true,
            forced: false,
            id: UUID()
        ))

        XCTAssertTrue(TranscriptScrollPolicy.shouldPerform(first, pending: first, isNearBottom: true))
        XCTAssertFalse(
            TranscriptScrollPolicy.shouldPerform(first, pending: newer, isNearBottom: true),
            "A stale queued token update must be coalesced behind the newest request"
        )
        XCTAssertFalse(
            TranscriptScrollPolicy.shouldPerform(first, pending: first, isNearBottom: false),
            "Manual scrolling away must cancel an already-queued automatic follow"
        )
        XCTAssertNil(TranscriptScrollPolicy.request(isNearBottom: false, forced: false))

        let forced = try XCTUnwrap(TranscriptScrollPolicy.request(
            isNearBottom: false,
            forced: true
        ))
        XCTAssertEqual(
            TranscriptScrollPolicy.prioritized(pending: forced, new: newer),
            forced,
            "An automatic layout update must not replace a queued conversation-change jump"
        )
        XCTAssertTrue(
            TranscriptScrollPolicy.shouldPerform(forced, pending: forced, isNearBottom: false),
            "Jump to Latest is an explicit forced request"
        )
    }

    func testResponseRangeIncludesCompleteAssistantToolBundleButPreservesPrompt() throws {
        let system = MotifChatMessage.system("system")
        let firstUser = MotifChatMessage.user("calculate")
        let toolCall = MotifChatMessage.assistant("tool call")
        let toolResult = MotifChatMessage.tool("42", name: "calculator")
        let finalAnswer = MotifChatMessage.assistant("The answer is 42.")
        let secondUser = MotifChatMessage.user("continue")
        let secondAnswer = MotifChatMessage.assistant("Sure.")
        let messages = [system, firstUser, toolCall, toolResult, finalAnswer, secondUser, secondAnswer]

        let range = try XCTUnwrap(ChatStore.responseRange(containing: toolResult.id, in: messages))
        XCTAssertEqual(range, 2..<5)
        XCTAssertEqual(Array(messages[range]).map(\.id), [toolCall.id, toolResult.id, finalAnswer.id])
        XCTAssertFalse(range.contains(1), "The originating user prompt must be preserved")
        XCTAssertEqual(ChatStore.responseRange(containing: finalAnswer.id, in: messages), range)
        XCTAssertNil(ChatStore.responseRange(containing: firstUser.id, in: messages))
    }

    func testMarkdownSegmentIdentitySurvivesStreamingAppends() {
        let initial = MarkdownSegment.identifiedSegments(from: "Hello")
        let extended = MarkdownSegment.identifiedSegments(from: "Hello world")
        XCTAssertEqual(initial.map(\.id), [0])
        XCTAssertEqual(extended.map(\.id), initial.map(\.id))

        let fenced = MarkdownSegment.identifiedSegments(from: "Hello\n```swift\nlet value = 4")
        let streamed = MarkdownSegment.identifiedSegments(from: "Hello\n```swift\nlet value = 42")
        XCTAssertEqual(fenced.map(\.id), [0, 1])
        XCTAssertEqual(streamed.map(\.id), fenced.map(\.id))
        XCTAssertEqual(streamed.last?.segment, .code("let value = 42", language: "swift"))
    }

    func testMarkdownBlocksPreserveBoundedParagraphRhythmAndGroupLists() throws {
        let blocks = MarkdownProseBlock.blocks(
            from: "First line\ncontinues here\n\n\n- one\n- two\n\n## Heading"
        )

        XCTAssertEqual(blocks.map(\.id), [0, 4, 7])
        XCTAssertFalse(blocks[0].hasParagraphBreakBefore)
        XCTAssertTrue(blocks[1].hasParagraphBreakBefore)
        XCTAssertTrue(blocks[2].hasParagraphBreakBefore)

        guard case .paragraph(let paragraph) = blocks[0].kind else {
            return XCTFail("Expected paragraph block")
        }
        XCTAssertEqual(paragraph, "First line\ncontinues here")

        guard case .list(style: .bulleted, let items) = blocks[1].kind else {
            return XCTFail("Expected grouped bulleted list")
        }
        XCTAssertEqual(items.map(\.text), ["one", "two"])

        guard case .heading(level: 2, text: "Heading") = blocks[2].kind else {
            return XCTFail("Expected level-two heading")
        }
    }

    func testMarkdownParserHandlesClosedFenceAndTrailingProse() {
        let segments = MarkdownSegment.identifiedSegments(
            from: "Intro\n```swift\nlet value = 42\n```\nAfter"
        )

        XCTAssertEqual(segments.map(\.id), [0, 1, 4])
        XCTAssertEqual(segments.map(\.segment), [
            .prose("Intro"),
            .code("let value = 42", language: "swift"),
            .prose("After"),
        ])
    }

    func testMarkdownParserKeepsMixedQuoteListParagraphAndDividerBoundaries() throws {
        let blocks = MarkdownProseBlock.blocks(
            from: "> quoted\n\n1. first\n2) second\n\nParagraph\n---"
        )

        XCTAssertEqual(blocks.map(\.id), [0, 2, 5, 6])
        guard case .quote("quoted") = blocks[0].kind else {
            return XCTFail("Expected quote")
        }
        guard case .list(style: .numbered, let items) = blocks[1].kind else {
            return XCTFail("Expected numbered list")
        }
        XCTAssertEqual(items.map(\.marker), ["1", "2"])
        XCTAssertEqual(items.map(\.text), ["first", "second"])
        guard case .paragraph("Paragraph") = blocks[2].kind else {
            return XCTFail("Expected paragraph")
        }
        guard case .divider = blocks[3].kind else {
            return XCTFail("Expected divider")
        }
    }

    @MainActor
    func testCancellationStaysGeneratingUntilOwnedTaskCleansUp() async {
        let store = ChatStore(
            backendOverride: HangingBackend(),
            loadsPersistedConversations: false
        )
        store.prompt = "Hello"
        store.send()

        XCTAssertTrue(store.isGenerating)
        store.cancel()
        XCTAssertTrue(store.isGenerating, "A replacement turn must remain blocked while cancellation settles")
        XCTAssertEqual(store.runtimeStatus, "Stopping…")

        await waitUntil { !store.isGenerating }
        XCTAssertFalse(store.isGenerating)
        XCTAssertEqual(store.runtimeStatus, "Cancelled")
    }

    @MainActor
    func testDeletingActiveConversationCancelsItsGenerationAndReturnsNewChatToIdle() async throws {
        let store = ChatStore(
            backendOverride: HangingBackend(),
            loadsPersistedConversations: false
        )
        let deletedConversationID = try XCTUnwrap(store.activeConversationID)
        store.prompt = "Hello"
        store.send()

        store.deleteConversation(deletedConversationID)

        XCTAssertNotEqual(store.activeConversationID, deletedConversationID)
        // Deleting the active conversation releases the composer immediately —
        // every callback of the dying stream no-ops once its conversation is
        // gone, so there is nothing for the replacement chat to wait on.
        XCTAssertFalse(store.isGenerating)
        XCTAssertEqual(store.runtimeStatus, "Idle")
        // Once the cancelled stream settles, its typed outcome lands without
        // bleeding status into the replacement conversation.
        await waitUntil { store.lastGenerationOutcome == .cancelled }
        XCTAssertFalse(store.isGenerating)
        XCTAssertEqual(store.runtimeStatus, "Idle")
    }

    @MainActor
    func testConversationSwitchPreservesStoppingStateAndTypedCancellationOutcome() async throws {
        let store = ChatStore(
            backendOverride: HangingBackend(),
            loadsPersistedConversations: false
        )
        let firstConversationID = try XCTUnwrap(store.activeConversationID)
        store.newChat()
        let secondConversationID = try XCTUnwrap(store.activeConversationID)
        store.selectConversation(firstConversationID)
        store.prompt = "Hello"
        store.send()

        store.selectConversation(secondConversationID)

        XCTAssertTrue(store.isGenerating)
        XCTAssertEqual(store.runtimeStatus, "Stopping…")
        await waitUntil { !store.isGenerating }
        XCTAssertEqual(store.lastGenerationOutcome, .cancelled)
        XCTAssertEqual(store.runtimeStatus, "Idle")
    }

    @MainActor
    func testRegenerateRestoresResponseWhenBackendSetupFailsSynchronously() {
        let store = ChatStore(loadsPersistedConversations: false)
        let system = MotifChatMessage.system("system")
        let user = MotifChatMessage.user("question")
        let answer = MotifChatMessage.assistant("valuable answer")
        let originalMessages = [system, user, answer]
        store.messages = originalMessages
        store.reasoningByMessage[answer.id] = "reasoning"
        store.backendMode = .openAICompatible
        store.endpoint = "not a url"

        store.regenerateLast()

        XCTAssertEqual(store.messages, originalMessages)
        XCTAssertEqual(store.reasoningByMessage[answer.id], "reasoning")
        XCTAssertFalse(store.isGenerating)
        XCTAssertEqual(store.runtimeStatus, "Error")
        XCTAssertNotNil(store.lastError)
    }

    @MainActor
    func testRegenerateRestoresResponseAfterAsynchronousBackendFailure() async {
        let store = ChatStore(
            backendOverride: DelayedFailingBackend(),
            loadsPersistedConversations: false
        )
        let system = MotifChatMessage.system("system")
        let user = MotifChatMessage.user("question")
        let answer = MotifChatMessage.assistant("valuable answer")
        let originalMessages = [system, user, answer]
        store.messages = originalMessages
        store.reasoningByMessage[answer.id] = "reasoning"

        store.regenerateLast()
        await waitUntil { !store.isGenerating }

        XCTAssertEqual(store.messages, originalMessages)
        XCTAssertEqual(store.reasoningByMessage[answer.id], "reasoning")
        XCTAssertEqual(store.lastGenerationOutcome, .failed)
        XCTAssertEqual(store.runtimeStatus, "Error")
        XCTAssertEqual(store.lastError, "delayed backend failure")
    }

    @MainActor
    func testRegenerateRestoresResponseAfterCancellation() async {
        let store = ChatStore(
            backendOverride: HangingBackend(),
            loadsPersistedConversations: false
        )
        let system = MotifChatMessage.system("system")
        let user = MotifChatMessage.user("question")
        let answer = MotifChatMessage.assistant("valuable answer")
        let originalMessages = [system, user, answer]
        store.messages = originalMessages
        store.reasoningByMessage[answer.id] = "reasoning"

        store.regenerateLast()
        store.cancel()
        await waitUntil { !store.isGenerating }

        XCTAssertEqual(store.messages, originalMessages)
        XCTAssertEqual(store.reasoningByMessage[answer.id], "reasoning")
        XCTAssertEqual(store.lastGenerationOutcome, .cancelled)
        XCTAssertEqual(store.runtimeStatus, "Cancelled")
    }

    @MainActor
    func testRegenerationAutoContinuationKeepsTurnSnapshotAndSameBubble() async throws {
        let backend = ScriptedBackend([
            .events([
                .text("replacement part one"),
                .completed(usage: nil, finishReason: .length),
                .completed(usage: nil, finishReason: .unknown),
            ]),
            .events([
                .text(" and part two"),
                .completed(usage: nil, finishReason: .stop),
            ]),
        ])
        let store = ChatStore(
            backendOverride: backend,
            loadsPersistedConversations: false
        )
        let system = MotifChatMessage.system("system")
        let user = MotifChatMessage.user("question")
        let oldAnswer = MotifChatMessage.assistant("old answer")
        store.messages = [system, user, oldAnswer]
        store.model = "snapshotted-model"
        store.maxTokens = 321

        store.regenerateLast()
        store.model = "later-model"
        store.maxTokens = 999
        await waitUntil { backend.requests.count == 2 && !store.isGenerating }

        XCTAssertEqual(store.messages.count, 3)
        XCTAssertEqual(store.messages.last?.content, "replacement part one and part two")
        XCTAssertEqual(backend.requests.map(\.parameters.model), ["snapshotted-model", "snapshotted-model"])
        XCTAssertEqual(backend.requests.map(\.parameters.maxTokens), [321, 321])
        XCTAssertTrue(
            try XCTUnwrap(backend.requests.last?.messages.last?.content)
                .contains("previous response was cut off")
        )
        XCTAssertTrue(store.truncatedMessages.isEmpty)
        XCTAssertEqual(store.runtimeStatus, "Idle")
    }

    @MainActor
    func testRegenerationRestoresTruncationStateWhenContinuationFails() async {
        let backend = ScriptedBackend([
            .events([
                .text("replacement"),
                .completed(usage: nil, finishReason: .length),
            ]),
            .failure(after: [.text(" unfinished continuation")]),
        ])
        let store = ChatStore(
            backendOverride: backend,
            loadsPersistedConversations: false
        )
        let system = MotifChatMessage.system("system")
        let user = MotifChatMessage.user("question")
        let oldAnswer = MotifChatMessage.assistant("valuable but truncated answer")
        let originalMessages = [system, user, oldAnswer]
        store.messages = originalMessages
        store.truncatedMessages.insert(oldAnswer.id)

        store.regenerateLast()
        await waitUntil { backend.requests.count == 2 && !store.isGenerating }

        XCTAssertEqual(store.messages, originalMessages)
        XCTAssertEqual(store.truncatedMessages, Set([oldAnswer.id]))
        XCTAssertEqual(store.lastGenerationOutcome, .failed)
        XCTAssertEqual(store.runtimeStatus, "Error")
        XCTAssertEqual(store.lastError, "scripted backend failure")
    }

    @MainActor
    func testDeleteResponseRemovesBundleAndTransientStateButKeepsUserPrompt() {
        let store = ChatStore(loadsPersistedConversations: false)
        let system = MotifChatMessage.system("system")
        let user = MotifChatMessage.user("calculate")
        let toolCall = MotifChatMessage.assistant("tool call")
        let toolResult = MotifChatMessage.tool("42", name: "calculator")
        let finalAnswer = MotifChatMessage.assistant("The answer is 42.")
        store.messages = [system, user, toolCall, toolResult, finalAnswer]
        store.reasoningByMessage[finalAnswer.id] = "reasoning"
        store.metrics[finalAnswer.id] = MessageMetrics(
            promptTokens: 2,
            completionTokens: 4,
            tokensPerSecond: 12,
            timeToFirstToken: 0.2,
            elapsed: 0.5,
            estimated: false
        )

        store.deleteResponse(containing: toolCall.id)

        XCTAssertEqual(store.messages, [system, user])
        XCTAssertNil(store.reasoningByMessage[finalAnswer.id])
        XCTAssertNil(store.metrics[finalAnswer.id])
    }

    @MainActor
    func testGenerationUsesConfigurationSnapshot() async {
        let backend = RecordingBackend()
        let store = ChatStore(
            backendOverride: backend,
            loadsPersistedConversations: false
        )
        store.model = "original-model"
        store.maxTokens = 321
        store.temperature = 0.25
        store.thinkMode = .captured
        store.toolsEnabled = true
        store.prompt = "Hello"

        store.send()
        store.model = "replacement-model"
        store.maxTokens = 999
        store.temperature = 1.5
        store.thinkMode = .hidden
        store.toolsEnabled = false

        await waitUntil { backend.request != nil && !store.isGenerating }
        let request = backend.request
        XCTAssertEqual(request?.parameters.model, "original-model")
        XCTAssertEqual(request?.parameters.maxTokens, 321)
        XCTAssertEqual(request?.parameters.temperature, 0.25)
        XCTAssertEqual(request?.parameters.thinkMode, .captured)
        XCTAssertTrue(request?.messages.first?.content.contains("get_current_time") == true)
    }

    @MainActor
    func testCapturedReasoningPersistsAcrossConversationSwitch() async throws {
        let backend = ScriptedBackend([
            .events([
                .reasoning("First, recall the speed formula. "),
                .reasoning("60 / 1.5 = 40."),
                .text("The average speed is 40 mph."),
                .completed(usage: nil, finishReason: .stop),
            ]),
        ])
        let store = ChatStore(backendOverride: backend, loadsPersistedConversations: false)
        let firstID = try XCTUnwrap(store.activeConversationID)

        store.prompt = "What is the average speed?"
        store.send()
        await waitUntil { !store.isGenerating }

        let answerID = try XCTUnwrap(store.messages.last(where: { $0.role == .assistant })?.id)
        XCTAssertEqual(
            store.reasoningByMessage[answerID],
            "First, recall the speed formula. 60 / 1.5 = 40."
        )

        // Switching away clears the live reasoning map...
        store.newChat()
        XCTAssertNil(store.reasoningByMessage[answerID])

        // ...and reopening the original conversation restores it (the persistence
        // path other than UserDefaults: the in-memory `conversations` array that
        // is what gets JSON-encoded to disk).
        store.selectConversation(firstID)
        XCTAssertEqual(
            store.reasoningByMessage[answerID],
            "First, recall the speed formula. 60 / 1.5 = 40."
        )
    }

    func testConversationCodableRoundTripsReasoningAndDecodesLegacyPayload() throws {
        let answerID = UUID()
        let conversation = MotifConversation(
            title: "t",
            messages: [.user("q"), .init(id: answerID, role: .assistant, content: "a")],
            reasoningByMessage: [answerID.uuidString: "captured reasoning"]
        )
        let data = try JSONEncoder().encode(conversation)
        let decoded = try JSONDecoder().decode(MotifConversation.self, from: data)
        XCTAssertEqual(decoded.reasoningByMessage?[answerID.uuidString], "captured reasoning")

        // A conversation persisted before this field (no `reasoningByMessage`
        // key) must still decode, with reasoning defaulting to nil.
        let legacy = #"{"id":"\#(UUID().uuidString)","title":"t","messages":[],"createdAt":0,"updatedAt":0}"#
        let legacyDecoded = try JSONDecoder().decode(MotifConversation.self, from: Data(legacy.utf8))
        XCTAssertNil(legacyDecoded.reasoningByMessage)
    }

    @MainActor
    private func waitUntil(
        attempts: Int = 200,
        _ predicate: @escaping @MainActor () -> Bool
    ) async {
        for _ in 0..<attempts {
            if predicate() { return }
            await Task.yield()
        }
        XCTFail("Timed out waiting for asynchronous state")
    }
}

private struct HangingBackend: MotifChatBackend {
    func streamResponse(
        messages: [MotifChatMessage],
        parameters: MotifGenerationParameters
    ) -> AsyncThrowingStream<MotifGenerationEvent, any Error> {
        AsyncThrowingStream { continuation in
            continuation.yield(.text("partial"))
        }
    }
}

private struct DelayedFailingBackend: MotifChatBackend {
    private struct Failure: LocalizedError, Sendable {
        var errorDescription: String? { "delayed backend failure" }
    }

    func streamResponse(
        messages: [MotifChatMessage],
        parameters: MotifGenerationParameters
    ) -> AsyncThrowingStream<MotifGenerationEvent, any Error> {
        AsyncThrowingStream { continuation in
            let task = Task {
                await Task.yield()
                continuation.yield(.text("partial replacement"))
                continuation.finish(throwing: Failure())
            }
            continuation.onTermination = { _ in task.cancel() }
        }
    }
}

private final class ScriptedBackend: MotifChatBackend, @unchecked Sendable {
    enum Step: Sendable {
        case events([MotifGenerationEvent])
        case failure(after: [MotifGenerationEvent])
    }

    struct Request: Sendable {
        let messages: [MotifChatMessage]
        let parameters: MotifGenerationParameters
    }

    private struct Failure: LocalizedError, Sendable {
        var errorDescription: String? { "scripted backend failure" }
    }

    private let lock = NSLock()
    private let steps: [Step]
    private var storedRequests: [Request] = []

    init(_ steps: [Step]) {
        self.steps = steps
    }

    var requests: [Request] {
        lock.withLock { storedRequests }
    }

    func streamResponse(
        messages: [MotifChatMessage],
        parameters: MotifGenerationParameters
    ) -> AsyncThrowingStream<MotifGenerationEvent, any Error> {
        let step = lock.withLock { () -> Step in
            storedRequests.append(Request(messages: messages, parameters: parameters))
            return steps[min(storedRequests.count - 1, steps.count - 1)]
        }
        return AsyncThrowingStream { continuation in
            switch step {
            case .events(let events):
                for event in events { continuation.yield(event) }
                continuation.finish()
            case .failure(let events):
                for event in events { continuation.yield(event) }
                continuation.finish(throwing: Failure())
            }
        }
    }
}

private final class RecordingBackend: MotifChatBackend, @unchecked Sendable {
    struct Request: Sendable {
        let messages: [MotifChatMessage]
        let parameters: MotifGenerationParameters
    }

    private let lock = NSLock()
    private var storedRequest: Request?

    var request: Request? {
        lock.withLock { storedRequest }
    }

    func streamResponse(
        messages: [MotifChatMessage],
        parameters: MotifGenerationParameters
    ) -> AsyncThrowingStream<MotifGenerationEvent, any Error> {
        lock.withLock {
            storedRequest = Request(messages: messages, parameters: parameters)
        }
        return AsyncThrowingStream { continuation in
            continuation.yield(.completed(usage: nil, finishReason: .stop))
            continuation.finish()
        }
    }
}
