import XCTest
@testable import MotifKit

final class ThinkStreamFilterTests: XCTestCase {
    func testVisibleModePassesThroughTagsAndContent() {
        var filter = ThinkStreamFilter(mode: .visible)
        XCTAssertEqual(filter.feed("A <think>secret</think> B"), "A <think>secret</think> B")
        XCTAssertEqual(filter.capturedReasoning, "")
    }

    func testHiddenModeDropsThinkBlockAcrossChunkBoundaries() {
        var filter = ThinkStreamFilter(mode: .hidden)
        XCTAssertEqual(filter.feed("Hello <thi"), "Hello ")
        XCTAssertEqual(filter.feed("nk>secret"), "")
        XCTAssertEqual(filter.feed("</thi"), "")
        XCTAssertEqual(filter.feed("nk> world"), " world")
        XCTAssertEqual(filter.finish(), "")
        XCTAssertEqual(filter.capturedReasoning, "")
    }

    func testCapturedModeStoresReasoningAndEmitsVisibleText() {
        var filter = ThinkStreamFilter(mode: .captured)
        XCTAssertEqual(filter.feed("A<think>one"), "A")
        XCTAssertEqual(filter.feed(" two</think>B"), "B")
        XCTAssertEqual(filter.capturedReasoning, "one two")
    }

    func testPromptCanStartInsideThinkBlock() {
        var filter = ThinkStreamFilter(mode: .hidden, startsInsideThinkBlock: true)
        XCTAssertEqual(filter.feed("hidden</think>answer"), "answer")
    }

    func testPromptTailDetectionIgnoresEarlierThinkText() {
        XCTAssertFalse(promptStartsInsideThinkBlock("user mentioned <think> but assistant starts now"))
        XCTAssertTrue(promptStartsInsideThinkBlock("<|assistant|><think>\n"))
    }
}
