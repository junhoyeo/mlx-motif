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

  public init(
    id: UUID = UUID(),
    title: String,
    messages: [MotifChatMessage],
    createdAt: Date = Date(),
    updatedAt: Date = Date()
  ) {
    self.id = id
    self.title = title
    self.messages = messages
    self.createdAt = createdAt
    self.updatedAt = updatedAt
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
