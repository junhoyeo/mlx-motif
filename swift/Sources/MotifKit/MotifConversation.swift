import Foundation

/// A persisted chat conversation: an ordered list of messages plus metadata for
/// the history sidebar. Codable so the app can JSON-encode an array of these
/// into `UserDefaults`.
public struct MotifConversation: Identifiable, Codable, Equatable, Sendable {
  public var id: UUID
  public var title: String
  public var messages: [MotifChatMessage]
  public var createdAt: Date
  public var updatedAt: Date
  /// Captured `<think>` reasoning for assistant turns, keyed by the message id's
  /// `uuidString`. Persisted alongside `messages` so the "Captured reasoning"
  /// disclosure survives a conversation switch or app relaunch — reasoning is
  /// display-only and never replayed into the model request. Optional so
  /// conversations persisted before this field still decode.
  public var reasoningByMessage: [String: String]?
  /// True once a title has been produced by the lightweight title model, which
  /// makes it sticky: `deriveTitle` (the transient first-prompt title) stops
  /// overwriting it as the conversation grows. Optional so conversations
  /// persisted before this field still decode (treated as not-yet-generated).
  public var titleGenerated: Bool?

  public init(
    id: UUID = UUID(),
    title: String,
    messages: [MotifChatMessage],
    createdAt: Date = Date(),
    updatedAt: Date = Date(),
    reasoningByMessage: [String: String]? = nil,
    titleGenerated: Bool? = nil
  ) {
    self.id = id
    self.title = title
    self.messages = messages
    self.createdAt = createdAt
    self.updatedAt = updatedAt
    self.reasoningByMessage = reasoningByMessage
    self.titleGenerated = titleGenerated
  }

  /// Placeholder shown until the first user message names the conversation.
  public static let untitledTitle = "New conversation"

  /// Derives a short title from the first user message, falling back to a
  /// placeholder when the conversation only has a system prompt so far.
  public static func deriveTitle(
    from messages: [MotifChatMessage],
    maxLength: Int = 48
  ) -> String {
    guard
      let firstUser = messages.first(where: { $0.role == .user }),
      case let trimmed = firstUser.content.trimmingCharacters(in: .whitespacesAndNewlines),
      !trimmed.isEmpty
    else {
      return untitledTitle
    }
    // Collapse to a single line so titles stay tidy in the sidebar.
    let singleLine = trimmed
      .split(whereSeparator: \.isNewline)
      .joined(separator: " ")
    if singleLine.count <= maxLength {
      return singleLine
    }
    let prefix = singleLine.prefix(maxLength).trimmingCharacters(in: .whitespaces)
    return prefix + "…"
  }
}
