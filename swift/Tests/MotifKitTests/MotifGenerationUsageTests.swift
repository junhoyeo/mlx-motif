import XCTest

@testable import MotifKit

/// Unit coverage for the token-usage accounting threaded through the terminal
/// generation event. This is the MotifKit half of the cross-module `usage` fix
/// that lets `MotifNativeServe` report real `prompt_tokens` / `completion_tokens`
/// / `total_tokens` (matching the Python reference server) instead of zeros.
final class MotifGenerationUsageTests: XCTestCase {
    func testTotalTokensIsPromptPlusCompletion() {
        let usage = MotifGenerationUsage(promptTokens: 7, completionTokens: 2)
        XCTAssertEqual(usage.promptTokens, 7)
        XCTAssertEqual(usage.completionTokens, 2)
        // total_tokens == prompt_tokens + completion_tokens — the OpenAI `usage`
        // shape this mirrors and the Python server emits.
        XCTAssertEqual(usage.totalTokens, 9)
    }

    func testCompletedEventCarriesUsagePayload() {
        let usage = MotifGenerationUsage(promptTokens: 5, completionTokens: 3)
        let event = MotifGenerationEvent.completed(usage: usage, finishReason: .stop)
        // The payload must be reachable by consumers (e.g. the server's `usage`
        // field), not just present on the case.
        if case .completed(let surfaced, let reason) = event {
            XCTAssertEqual(surfaced?.promptTokens, 5)
            XCTAssertEqual(surfaced?.completionTokens, 3)
            XCTAssertEqual(surfaced?.totalTokens, 8)
            XCTAssertEqual(reason, .stop)
        } else {
            XCTFail("expected .completed event to carry usage")
        }
    }

    func testCompletedEventEquatableAcrossUsage() {
        let a = MotifGenerationEvent.completed(usage: MotifGenerationUsage(promptTokens: 1, completionTokens: 1), finishReason: .stop)
        let b = MotifGenerationEvent.completed(usage: MotifGenerationUsage(promptTokens: 1, completionTokens: 1), finishReason: .stop)
        let none = MotifGenerationEvent.completed(usage: nil, finishReason: .stop)
        XCTAssertEqual(a, b)
        XCTAssertNotEqual(a, none)
        // Backends that cannot report counts (the remote SSE client) construct
        // `.completed(usage: nil, ...)`; that must remain a distinct, valid value.
        XCTAssertEqual(none, .completed(usage: nil, finishReason: .stop))
    }

    func testFinishReasonDistinguishesTruncation() {
        // A truncated turn (.length) must not compare equal to a natural stop —
        // the UI's continue affordance keys off exactly this distinction.
        let truncated = MotifGenerationEvent.completed(usage: nil, finishReason: .length)
        let stopped = MotifGenerationEvent.completed(usage: nil, finishReason: .stop)
        XCTAssertNotEqual(truncated, stopped)
    }
}
