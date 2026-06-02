import XCTest
@testable import MotifKit

/// Covers the pure sliding-window trimming used by the app's context guard.
/// These run headless against `motifTrimMessagesToBudget`, which the app's
/// `ChatStore` calls when assembling request messages.
final class MotifContextBudgetTests: XCTestCase {
  func testEmptyKeepsAll() {
    let result = motifTrimMessagesToBudget([], budgetTokens: 1000)
    XCTAssertTrue(result.kept.isEmpty)
    XCTAssertEqual(result.dropped, 0)
  }

  func testBelowBudgetKeepsAll() {
    let messages: [MotifChatMessage] = [
      .system("You are Motif."),
      .user("Hello"),
      .assistant("Hi there"),
    ]
    let result = motifTrimMessagesToBudget(messages, budgetTokens: 1000)
    XCTAssertEqual(result.kept, messages)
    XCTAssertEqual(result.dropped, 0)
  }

  func testOverBudgetDropsOldestNonSystemFirst() {
    // Each content is 40 chars => ~10 tokens. Budget of 25 fits ~2 turns.
    let chunk = String(repeating: "a", count: 40)
    let messages: [MotifChatMessage] = [
      .system("sys"),
      .user(chunk),       // oldest -> dropped first
      .assistant(chunk),
      .user(chunk),       // latest -> always kept
    ]
    let result = motifTrimMessagesToBudget(messages, budgetTokens: 25)

    XCTAssertGreaterThan(result.dropped, 0)
    // System always retained as first.
    XCTAssertEqual(result.kept.first?.role, .system)
    // The dropped one is the oldest user message; recent turns survive.
    XCTAssertFalse(result.kept.contains(messages[1]))
    XCTAssertTrue(result.kept.contains(messages[3]))
  }

  func testSystemMessageAlwaysRetained() {
    let huge = String(repeating: "z", count: 10_000)
    let messages: [MotifChatMessage] = [
      .system("system prompt"),
      .user(huge),
      .assistant(huge),
      .user("final question"),
    ]
    let result = motifTrimMessagesToBudget(messages, budgetTokens: 10)
    XCTAssertEqual(result.kept.first?.role, .system)
    XCTAssertEqual(result.kept.first?.content, "system prompt")
  }

  func testNeverDropsLatestUserMessage() {
    let huge = String(repeating: "q", count: 100_000)
    let messages: [MotifChatMessage] = [
      .system("sys"),
      .user("old"),
      .user(huge), // latest, alone exceeds budget
    ]
    let result = motifTrimMessagesToBudget(messages, budgetTokens: 5)
    XCTAssertEqual(result.kept.last, messages.last)
  }

  func testNoLeadingSystemStillTrims() {
    let chunk = String(repeating: "b", count: 40) // ~10 tokens
    let messages: [MotifChatMessage] = [
      .user(chunk),
      .assistant(chunk),
      .user(chunk),
    ]
    let result = motifTrimMessagesToBudget(messages, budgetTokens: 15)
    XCTAssertGreaterThan(result.dropped, 0)
    XCTAssertEqual(result.kept.last, messages.last)
  }
}
