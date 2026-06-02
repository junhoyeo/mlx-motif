import Foundation

/// Conservative char-based token estimate for context-budget accounting.
///
/// There is no tokenizer available in the app layer, so we approximate token
/// count as `characters / 4`, which over-counts slightly for typical English
/// prose (a safe direction for a budget guard). The estimate is intentionally
/// simple and deterministic so the trimming logic stays unit-testable.
public func motifEstimatedTokenCount(_ text: String) -> Int {
  // Round up so even a 1–3 character message costs at least one token.
  (text.count + 3) / 4
}

/// Trims a chat transcript so its estimated token count fits `budgetTokens`,
/// using a sliding-window policy:
///
/// - The leading `.system` message (if present as the first message) is always
///   retained and never counted out of the conversation.
/// - The most recent messages are kept; oldest non-system messages are dropped
///   first until the cumulative estimate fits the budget.
/// - The latest message is always retained even if it alone exceeds the budget,
///   so the active user turn is never dropped.
///
/// This is pure, allocation-light, and deterministic so it can be exercised
/// headlessly in unit tests.
///
/// - Returns: the kept messages in original order plus the number dropped.
public func motifTrimMessagesToBudget(
  _ messages: [MotifChatMessage],
  budgetTokens: Int
) -> (kept: [MotifChatMessage], dropped: Int) {
  guard !messages.isEmpty else { return (messages, 0) }

  // Preserve a leading system message verbatim; budget only the rest.
  let hasLeadingSystem = messages.first?.role == .system
  let systemMessage = hasLeadingSystem ? messages.first : nil
  let conversation = hasLeadingSystem ? Array(messages.dropFirst()) : messages

  guard !conversation.isEmpty else { return (messages, 0) }

  let systemCost = systemMessage.map { motifEstimatedTokenCount($0.content) } ?? 0
  let effectiveBudget = max(0, budgetTokens - systemCost)

  // Walk newest -> oldest, accumulating cost; stop before the budget overflows.
  var keptReversed: [MotifChatMessage] = []
  var runningCost = 0
  for message in conversation.reversed() {
    let cost = motifEstimatedTokenCount(message.content)
    if keptReversed.isEmpty {
      // Always keep the latest message, even if it alone exceeds the budget.
      keptReversed.append(message)
      runningCost = cost
      continue
    }
    if runningCost + cost > effectiveBudget {
      break
    }
    keptReversed.append(message)
    runningCost += cost
  }

  let keptConversation = Array(keptReversed.reversed())
  let dropped = conversation.count - keptConversation.count

  var kept: [MotifChatMessage] = []
  if let systemMessage { kept.append(systemMessage) }
  kept.append(contentsOf: keptConversation)
  return (kept, dropped)
}
