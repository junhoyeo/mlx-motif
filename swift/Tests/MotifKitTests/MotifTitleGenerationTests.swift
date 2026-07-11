import XCTest
@testable import MotifKit

final class MotifTitleGenerationTests: XCTestCase {
    func testSystemPromptAsksForShortSameLanguageTitle() {
        let prompt = MotifTitleGeneration.systemPrompt()
        XCTAssertTrue(prompt.contains("title generator"))
        XCTAssertTrue(prompt.contains("SAME language"))
        XCTAssertTrue(prompt.contains("6 words or fewer"))
    }

    func testSanitizeTakesFirstNonEmptyLine() {
        let out = MotifTitleGeneration.sanitize("\n  Refactor the auth service  \nextra\n", fallback: "fb")
        XCTAssertEqual(out, "Refactor the auth service")
    }

    func testSanitizeStripsThinkBlock() {
        let raw = "<think>the user wants a calc</think>Multiply 37 by 41"
        XCTAssertEqual(MotifTitleGeneration.sanitize(raw, fallback: "fb"), "Multiply 37 by 41")
    }

    func testSanitizeStripsUnclosedThinkTail() {
        // A dangling <think> with no close: everything after it is reasoning, so
        // only the text before it is the title.
        let raw = "Debug 500 errors\n<think>hmm let me reconsider and keep going"
        XCTAssertEqual(MotifTitleGeneration.sanitize(raw, fallback: "fb"), "Debug 500 errors")
    }

    func testSanitizeStripsWrappingQuotesAndTrailingPunctuation() {
        XCTAssertEqual(MotifTitleGeneration.sanitize("\"Connect Postgres to API.\"", fallback: "fb"),
                       "Connect Postgres to API")
        XCTAssertEqual(MotifTitleGeneration.sanitize("`parser.ts bug fix`", fallback: "fb"),
                       "parser.ts bug fix")
    }

    func testSanitizeCutsColonQuotedRestatement() {
        // A small model sometimes appends a quoted restatement after a colon.
        let raw = "Connect Postgres to API: \"Establish Postgres Connection for API\""
        XCTAssertEqual(MotifTitleGeneration.sanitize(raw, fallback: "fb"), "Connect Postgres to API")
    }

    func testSanitizeKeepsLegitimateColon() {
        // A real title colon (not followed by an opening quote) is preserved.
        XCTAssertEqual(MotifTitleGeneration.sanitize("auth.ts: refresh token", fallback: "fb"),
                       "auth.ts: refresh token")
    }

    func testSanitizeStripsEchoedTitleLabel() {
        XCTAssertEqual(MotifTitleGeneration.sanitize("title: Debug production 500 errors", fallback: "fb"),
                       "Debug production 500 errors")
    }

    func testSanitizePreservesNonLatinScript() {
        // Title language mirrors the input; sanitizer must not mangle it.
        XCTAssertEqual(MotifTitleGeneration.sanitize("오늘 날씨 문의", fallback: "fb"), "오늘 날씨 문의")
    }

    func testSanitizeFallsBackWhenEmptyOrOnlyReasoning() {
        XCTAssertEqual(MotifTitleGeneration.sanitize("   \n  ", fallback: "First prompt"), "First prompt")
        XCTAssertEqual(MotifTitleGeneration.sanitize("<think>only reasoning</think>", fallback: "First prompt"),
                       "First prompt")
    }

    func testSanitizeCapsLength() {
        let long = String(repeating: "word ", count: 40)
        let out = MotifTitleGeneration.sanitize(long, fallback: "fb", maxLength: 48)
        XCTAssertLessThanOrEqual(out.count, 49) // 48 + the ellipsis
        XCTAssertTrue(out.hasSuffix("…"))
    }
}
