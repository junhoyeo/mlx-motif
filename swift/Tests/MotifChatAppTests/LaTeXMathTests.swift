import Foundation
@testable import MotifChatApp
import XCTest

/// Covers the pure (non-view) parts of the LaTeX math renderer: the parser, the
/// inline `\(…\)` / `$…$` splitter, and display-math block segmentation.
final class LaTeXMathTests: XCTestCase {

    // MARK: Parser

    func testParsesVariablesUprightDigitsAndMappedSymbols() {
        XCTAssertEqual(LaTeXMath.parse("x"), [.run("x", italic: true)])
        XCTAssertEqual(LaTeXMath.parse("5"), [.run("5", italic: false)])
        XCTAssertEqual(LaTeXMath.parse("-"), [.run("\u{2212}", italic: false)])
        XCTAssertEqual(LaTeXMath.parse("\\alpha"), [.run("\u{03B1}", italic: false)])
        XCTAssertEqual(LaTeXMath.parse("\\to"), [.run("\u{2192}", italic: false)])
        XCTAssertEqual(LaTeXMath.parse("\\cdot"), [.run("\u{22C5}", italic: false)])
    }

    func testParsesFunctionNamesAsUprightText() {
        XCTAssertEqual(LaTeXMath.parse("\\sin"), [.text("sin")])
        XCTAssertEqual(LaTeXMath.parse("\\ln"), [.text("ln")])
    }

    func testParsesFractionWithGroupedNumeratorAndDenominator() {
        XCTAssertEqual(
            LaTeXMath.parse("\\frac{a}{b}"),
            [.fraction(numerator: [.run("a", italic: true)], denominator: [.run("b", italic: true)])]
        )
    }

    func testParsesSuperscriptAndSubscript() {
        XCTAssertEqual(
            LaTeXMath.parse("x^2"),
            [.scripted(base: .run("x", italic: true), sub: nil, sup: [.run("2", italic: false)], bigOperator: false)]
        )
        XCTAssertEqual(
            LaTeXMath.parse("a_i"),
            [.scripted(base: .run("a", italic: true), sub: [.run("i", italic: true)], sup: nil, bigOperator: false)]
        )
    }

    func testLimitOperatorIsMarkedBigOperatorForUnderscripts() {
        XCTAssertEqual(
            LaTeXMath.parse("\\lim_{x}"),
            [.scripted(base: .text("lim"), sub: [.run("x", italic: true)], sup: nil, bigOperator: true)]
        )
    }

    func testParsesBoxedSqrtAndText() {
        XCTAssertEqual(LaTeXMath.parse("\\boxed{5}"), [.boxed([.run("5", italic: false)])])
        XCTAssertEqual(LaTeXMath.parse("\\sqrt{2}"), [.sqrt([.run("2", italic: false)])])
        XCTAssertEqual(LaTeXMath.parse("\\text{hi}"), [.text("hi")])
    }

    func testUnknownCommandFallsBackToItsNameWithoutCrashing() {
        XCTAssertEqual(LaTeXMath.parse("\\wobble"), [.text("wobble")])
    }

    func testLeftRightDelimitersAreKeptWithoutTheCommand() {
        // \left( x \right) → "(" x ")"
        XCTAssertEqual(
            LaTeXMath.parse("\\left(x\\right)"),
            [.run("(", italic: false), .run("x", italic: true), .run(")", italic: false)]
        )
    }

    // MARK: Inline splitting

    func testSplitsParenthesisDelimitedInlineMath() {
        XCTAssertEqual(
            MathInline.spans("a \\(x\\) b"),
            [.text("a "), .math("x"), .text(" b")]
        )
        XCTAssertTrue(MathInline.hasMath("a \\(x\\) b"))
    }

    func testDollarMathRequiresMathLikeContent() {
        XCTAssertEqual(MathInline.spans("value $x^2$ end"), [.text("value "), .math("x^2"), .text(" end")])
        // Currency must not be mistaken for math.
        XCTAssertFalse(MathInline.hasMath("cost $5 and $10 today"))
        XCTAssertEqual(MathInline.spans("cost $5 and $10 today"), [.text("cost $5 and $10 today")])
    }

    func testUnterminatedInlineMathStaysLiteralForStreaming() {
        let line = "start \\(x + 1"
        XCTAssertFalse(MathInline.hasMath(line))
        XCTAssertEqual(MathInline.spans(line), [.text(line)])
    }

    func testTokensPreserveOrderOfWordsAndMath() {
        let tokens = MathLineToken.tokens(from: "a \\(x\\) b")
        let kinds: [String] = tokens.map { token in
            switch token.kind {
            case .word(let w): return "word:\(String(w.characters))"
            case .math(let s): return "math:\(s)"
            }
        }
        // The space after the closing "\)" is kept as its own token so the gap
        // between math and following prose survives wrapping.
        XCTAssertEqual(kinds, ["word:a ", "math:x", "word: ", "word:b"])
    }

    // MARK: Display-math segmentation

    func testMultilineDisplayMathBecomesItsOwnSegment() {
        let segments = MarkdownSegment.identifiedSegments(
            from: "Intro\n\\[\na+b\n\\]\nAfter"
        )
        XCTAssertEqual(segments.map(\.segment), [
            .prose("Intro"),
            .displayMath("a+b"),
            .prose("After"),
        ])
        XCTAssertEqual(segments.map(\.id), [0, 1, 4])
    }

    func testSingleLineDisplayMathDelimiters() {
        XCTAssertEqual(
            MarkdownSegment.identifiedSegments(from: "\\[ x = 1 \\]").map(\.segment),
            [.displayMath("x = 1")]
        )
        XCTAssertEqual(
            MarkdownSegment.identifiedSegments(from: "$$a=b$$").map(\.segment),
            [.displayMath("a=b")]
        )
    }

    func testUnterminatedDisplayMathStillRendersPartialForStreaming() {
        let segments = MarkdownSegment.identifiedSegments(from: "\\[\na+b")
        XCTAssertEqual(segments.map(\.segment), [.displayMath("a+b")])
    }

    func testInlineMathDoesNotTriggerBlockSegmentation() {
        // A line with only inline \(…\) has no block delimiters, so it stays a
        // single prose segment (inline math is handled at render time).
        let segments = MarkdownSegment.identifiedSegments(from: "text \\(x\\) more")
        XCTAssertEqual(segments.map(\.segment), [.prose("text \\(x\\) more")])
        XCTAssertEqual(segments.map(\.id), [0])
    }

    func testDisplayMathSegmentIdentityIsStableWhileStreaming() {
        let partial = MarkdownSegment.identifiedSegments(from: "Intro\n\\[\na+b")
        let complete = MarkdownSegment.identifiedSegments(from: "Intro\n\\[\na+b\n\\]")
        XCTAssertEqual(partial.map(\.id), complete.map(\.id))
    }

    // MARK: Regressions from adversarial review

    func testSameLineDisplayMathAndTrailingProseGetUniqueIDs() {
        let segments = MarkdownSegment.identifiedSegments(from: "$$x$$ trailing")
        XCTAssertEqual(segments.map(\.segment), [.displayMath("x"), .prose("trailing")])
        XCTAssertEqual(Set(segments.map(\.id)).count, segments.count, "Segment IDs must be unique for ForEach")
    }

    func testInlineCodeSpanSuppressesMathDelimiters() {
        XCTAssertFalse(MathInline.hasMath("use `$x$` literally"))
        XCTAssertEqual(MathInline.spans("use `$x$` literally"), [.text("use `$x$` literally")])
        // Math outside the code span still parses.
        XCTAssertTrue(MathInline.hasMath("code `$x$` then \\(y\\)"))
    }

    func testTextCommandPreservesInteriorWhitespace() {
        XCTAssertEqual(LaTeXMath.parse("\\text{for all}"), [.text("for all")])
        XCTAssertEqual(LaTeXMath.parse("\\operatorname{arg max}"), [.text("arg max")])
    }

    func testMarkdownEmphasisSurvivesAcrossInlineMath() {
        let tokens = MathLineToken.tokens(from: "**energy \\(E\\) is conserved**")

        // No literal emphasis markers leak into any word token.
        for token in tokens {
            if case .word(let w) = token.kind {
                XCTAssertFalse(String(w.characters).contains("*"), "Unmatched markdown marker leaked")
            }
        }
        // The math atom is preserved in place.
        XCTAssertTrue(tokens.contains { if case .math("E") = $0.kind { return true } else { return false } })
        // Emphasis is applied to the surrounding words (single parse context).
        let anyStrong = tokens.contains { token in
            if case .word(let w) = token.kind {
                return w.runs.contains { $0.inlinePresentationIntent?.contains(.stronglyEmphasized) == true }
            }
            return false
        }
        XCTAssertTrue(anyStrong, "Bold spanning the math boundary should survive")
    }
}
