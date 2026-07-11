import SwiftUI

/// Renders chat message content as a mix of Markdown prose and fenced code
/// blocks. Splits the raw content on ``` fences: prose segments render as
/// Markdown (inline syntax preserved, falling back to plain text on parse
/// failure) and code segments render in a monospaced block with a Copy button.
///
/// Robust to partial/streaming content: an unterminated final fence is treated
/// as an in-progress code block so incremental tokens still render sensibly.
struct MarkdownMessageView: View {
  let content: String

  var body: some View {
    VStack(alignment: .leading, spacing: 8) {
      ForEach(MarkdownSegment.identifiedSegments(from: content)) { identifiedSegment in
        switch identifiedSegment.segment {
        case .prose(let text):
          ProseView(text: text)
        case .code(let code, let language):
          CodeBlockView(code: code, language: language)
        }
      }
    }
    // Left-align blocks via the stack's alignment guide, but let the content hug
    // its natural width (up to the enclosing bubble's max-width cap) rather than
    // greedily filling — otherwise every bubble stretches to the cap regardless
    // of how little it contains. (vstack skill.)
  }
}

/// Markdown prose. SwiftUI's `Text(AttributedString)` only renders *inline*
/// markdown (bold/italic/code) — it flattens block constructs, so a `- item`
/// list shows as literal text. To get real bullet/numbered lists we split the
/// prose into block-level lines here and render list items as proper rows,
/// applying inline markdown within each line.
private struct ProseView: View {
  let text: String

  private var blocks: [MarkdownProseBlock] { MarkdownProseBlock.blocks(from: text) }

  var body: some View {
    VStack(alignment: .leading, spacing: 4) {
      ForEach(blocks) { block in
        blockView(block)
          .padding(.top, topPadding(for: block))
      }
    }
  }

  @ViewBuilder
  private func blockView(_ block: MarkdownProseBlock) -> some View {
    switch block.kind {
    case .paragraph(let text):
      inlineText(text)
    case .heading(let level, let text):
      inlineText(text)
        .font(headingFont(level))
        .fontWeight(.semibold)
        .accessibilityHeading(accessibilityHeadingLevel(level))
    case .divider:
      Divider()
        .padding(.vertical, 2)
        .accessibilityHidden(true)
    case .quote(let text):
      HStack(alignment: .top, spacing: 8) {
        RoundedRectangle(cornerRadius: 1.5)
          .fill(.tertiary)
          .frame(width: 3)
          .accessibilityHidden(true)
        inlineText(text)
          .foregroundStyle(.secondary)
      }
      .fixedSize(horizontal: false, vertical: true)
      .accessibilityElement(children: .combine)
      .accessibilityLabel(Text("Quote: \(plainAccessibilityText(text))"))
    case .list(let style, let items):
      VStack(alignment: .leading, spacing: 3) {
        ForEach(items) { item in
          HStack(alignment: .firstTextBaseline, spacing: 8) {
            switch style {
            case .bulleted:
              Text("•")
                .foregroundStyle(.secondary)
                .frame(minWidth: 14, alignment: .trailing)
            case .numbered:
              Text("\(item.marker ?? "").")
                .foregroundStyle(.secondary)
                .monospacedDigit()
                .frame(minWidth: 20, alignment: .trailing)
            }
            inlineText(item.text)
          }
          .accessibilityElement(children: .combine)
          .accessibilityLabel(listItemAccessibilityLabel(item, style: style))
        }
      }
      .accessibilityElement(children: .contain)
      .accessibilityLabel(
        Text("\(style == .bulleted ? "Bulleted" : "Numbered") list, \(items.count) items")
      )
    }
  }

  /// Blank lines add one bounded paragraph gap. Repeated blank lines are
  /// intentionally collapsed so model output cannot create giant dead zones.
  private func topPadding(for block: MarkdownProseBlock) -> CGFloat {
    switch block.kind {
    case .heading:
      block.hasParagraphBreakBefore ? 8 : 4
    default:
      block.hasParagraphBreakBefore ? 4 : 0
    }
  }

  private func listItemAccessibilityLabel(
    _ item: MarkdownListItem,
    style: MarkdownListStyle
  ) -> Text {
    let text = plainAccessibilityText(item.text)
    switch style {
    case .bulleted:
      return Text("List item: \(text)")
    case .numbered:
      return Text("Item \(item.marker ?? ""): \(text)")
    }
  }

  private func accessibilityHeadingLevel(_ level: Int) -> AccessibilityHeadingLevel {
    switch level {
    case 1: .h1
    case 2: .h2
    case 3: .h3
    case 4: .h4
    case 5: .h5
    default: .h6
    }
  }

  private func plainAccessibilityText(_ line: String) -> String {
    String(inlineAttributedString(line).characters)
  }

  private func inlineAttributedString(_ line: String) -> AttributedString {
    (try? AttributedString(
      markdown: line,
      options: .init(interpretedSyntax: .inlineOnlyPreservingWhitespace)
    )) ?? AttributedString(line)
  }

  /// Heading sizes step down with level; H4+ clamp to headline-weight body.
  private func headingFont(_ level: Int) -> Font {
    switch level {
    case 1: .title2
    case 2: .title3
    default: .headline
    }
  }

  /// One line with inline markdown (bold/italic/inline-code) applied.
  @ViewBuilder
  private func inlineText(_ line: String) -> some View {
    Text(inlineAttributedString(line))
      .textSelection(.enabled)
      .fixedSize(horizontal: false, vertical: true)
  }
}

/// A deterministic, testable block model for Markdown prose. Source line
/// positions provide stable SwiftUI identity while an append-only response is
/// streaming, without tying identity to text that changes on every token.
struct MarkdownProseBlock: Identifiable, Equatable {
  enum Kind: Equatable {
    case paragraph(String)
    case heading(level: Int, text: String)
    case divider
    case quote(String)
    case list(style: MarkdownListStyle, items: [MarkdownListItem])
  }

  let id: Int
  var kind: Kind
  let hasParagraphBreakBefore: Bool

  static func blocks(from text: String) -> [MarkdownProseBlock] {
    var blocks: [MarkdownProseBlock] = []
    var paragraphLines: [String] = []
    var paragraphStartLine: Int?
    var paragraphHasBreakBefore = false
    var pendingParagraphBreak = false

    func flushParagraph() {
      guard let startLine = paragraphStartLine else { return }
      blocks.append(
        MarkdownProseBlock(
          id: startLine,
          kind: .paragraph(paragraphLines.joined(separator: "\n")),
          hasParagraphBreakBefore: paragraphHasBreakBefore
        ))
      paragraphLines.removeAll(keepingCapacity: true)
      paragraphStartLine = nil
      paragraphHasBreakBefore = false
    }

    func appendStructural(_ kind: Kind, at line: Int) {
      flushParagraph()
      blocks.append(
        MarkdownProseBlock(
          id: line,
          kind: kind,
          hasParagraphBreakBefore: pendingParagraphBreak && !blocks.isEmpty
        ))
      pendingParagraphBreak = false
    }

    func appendListItem(_ item: MarkdownListItem, style: MarkdownListStyle) {
      flushParagraph()
      if !pendingParagraphBreak,
        let lastIndex = blocks.indices.last,
        case .list(let existingStyle, var items) = blocks[lastIndex].kind,
        existingStyle == style
      {
        items.append(item)
        blocks[lastIndex].kind = .list(style: style, items: items)
      } else {
        blocks.append(
          MarkdownProseBlock(
            id: item.id,
            kind: .list(style: style, items: [item]),
            hasParagraphBreakBefore: pendingParagraphBreak && !blocks.isEmpty
          ))
      }
      pendingParagraphBreak = false
    }

    for (lineNumber, rawLine) in text.components(separatedBy: "\n").enumerated() {
      let trimmed = rawLine.trimmingCharacters(in: .whitespaces)
      if trimmed.isEmpty {
        flushParagraph()
        pendingParagraphBreak = !blocks.isEmpty
        continue
      }

      // Horizontal rule: a line that is only 3+ dashes/asterisks/underscores.
      if trimmed.count >= 3,
        trimmed.allSatisfy({ $0 == "-" || $0 == "*" || $0 == "_" }),
        Set(trimmed).count == 1
      {
        appendStructural(.divider, at: lineNumber)
        continue
      }

      // Heading: 1–6 leading #s followed by a space ("### Title").
      if trimmed.hasPrefix("#") {
        let hashes = trimmed.prefix(while: { $0 == "#" })
        let rest = trimmed.dropFirst(hashes.count)
        if (1...6).contains(hashes.count), rest.first == " " {
          appendStructural(
            .heading(
              level: hashes.count,
              text: rest.trimmingCharacters(in: .whitespaces)
            ),
            at: lineNumber
          )
          continue
        }
      }

      // Blockquote: "> " prefix (single level; nested markers collapse).
      if trimmed.hasPrefix(">") {
        let body = trimmed.drop(while: { $0 == ">" || $0 == " " })
        appendStructural(.quote(String(body)), at: lineNumber)
        continue
      }

      // Bullet: -, *, or • followed by a space.
      if let marker = ["- ", "* ", "• "].first(where: { trimmed.hasPrefix($0) }) {
        appendListItem(
          MarkdownListItem(
            id: lineNumber,
            marker: nil,
            text: String(trimmed.dropFirst(marker.count))
          ),
          style: .bulleted
        )
        continue
      }

      // Numbered: "1. ", "2) " etc.
      if let markerRange = trimmed.range(of: #"^\d+[.)]\s+"#, options: .regularExpression) {
        let marker = trimmed[trimmed.startIndex..<markerRange.upperBound]
          .trimmingCharacters(in: CharacterSet(charactersIn: ".) "))
        appendListItem(
          MarkdownListItem(
            id: lineNumber,
            marker: marker,
            text: String(trimmed[markerRange.upperBound...])
          ),
          style: .numbered
        )
        continue
      }

      if paragraphStartLine == nil {
        paragraphStartLine = lineNumber
        paragraphHasBreakBefore = pendingParagraphBreak && !blocks.isEmpty
        pendingParagraphBreak = false
      }
      paragraphLines.append(trimmed)
    }

    flushParagraph()
    return blocks
  }
}

enum MarkdownListStyle: Equatable {
  case bulleted
  case numbered
}

struct MarkdownListItem: Identifiable, Equatable {
  let id: Int
  let marker: String?
  let text: String
}

/// A fenced code block: monospaced, subtle background, with a Copy button.
private struct CodeBlockView: View {
  let code: String
  let language: String?

  @State private var didCopy = false

  var body: some View {
    VStack(alignment: .leading, spacing: 0) {
      HStack {
        if let language, !language.isEmpty {
          Text(language)
            .font(.caption2)
            .foregroundStyle(.secondary)
        }
        Spacer()
        Button {
          copyToPasteboard(code)
          didCopy = true
          DispatchQueue.main.asyncAfter(deadline: .now() + 1.2) { didCopy = false }
        } label: {
          Label(didCopy ? "Copied" : "Copy", systemImage: didCopy ? "checkmark" : "doc.on.doc")
            .font(.caption2)
        }
        .buttonStyle(.borderless)
        .accessibilityLabel(Text(didCopy ? "Code copied" : "Copy code"))
        .accessibilityHint(Text("Copies this code block to the clipboard"))
      }
      .padding(.horizontal, 10)
      .padding(.vertical, 6)

      Divider()

      ScrollView(.horizontal, showsIndicators: true) {
        Text(code)
          // One step below body — code blocks read better slightly smaller
          // than prose (mono stays: this is literal code content).
          .font(.system(.callout, design: .monospaced))
          .textSelection(.enabled)
          .padding(10)
          .frame(maxWidth: .infinity, alignment: .leading)
      }
      .focusable()
      .accessibilityLabel(Text(codeAccessibilityLabel))
      .accessibilityHint(Text("Scroll horizontally with the arrow keys to view long lines"))
    }
    .background(.thinMaterial, in: RoundedRectangle(cornerRadius: 8, style: .continuous))
    .overlay(
      RoundedRectangle(cornerRadius: 8, style: .continuous)
        .stroke(.quaternary, lineWidth: 1)
    )
  }

  private var codeAccessibilityLabel: String {
    guard let language, !language.isEmpty else { return "Code block" }
    return "\(language) code block"
  }
}

/// One parsed segment of message content.
enum MarkdownSegment: Equatable {
  case prose(String)
  case code(String, language: String?)

  /// Returns segments with source-line identities that remain stable as new
  /// lines and tokens are appended during streaming.
  static func identifiedSegments(from content: String) -> [IdentifiedMarkdownSegment] {
    guard content.contains("```") else {
      let trimmed = content.trimmingCharacters(in: .whitespacesAndNewlines)
      return trimmed.isEmpty ? [] : [.init(id: 0, segment: .prose(content))]
    }

    var segments: [IdentifiedMarkdownSegment] = []
    let lines = content.components(separatedBy: "\n")
    var proseBuffer: [String] = []
    var proseStartLine: Int?
    var codeBuffer: [String] = []
    var codeStartLine: Int?
    var codeLanguage: String?
    var inCode = false

    func flushProse() {
      let joined = proseBuffer.joined(separator: "\n")
      if !joined.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
        segments.append(.init(id: proseStartLine ?? 0, segment: .prose(joined)))
      }
      proseBuffer.removeAll(keepingCapacity: true)
      proseStartLine = nil
    }

    func flushCode() {
      segments.append(
        .init(
          id: codeStartLine ?? 0,
          segment: .code(codeBuffer.joined(separator: "\n"), language: codeLanguage)
        ))
      codeBuffer.removeAll(keepingCapacity: true)
      codeStartLine = nil
      codeLanguage = nil
    }

    for (lineNumber, line) in lines.enumerated() {
      if line.hasPrefix("```") {
        if inCode {
          // Closing fence.
          flushCode()
          inCode = false
        } else {
          // Opening fence — capture optional language tag.
          flushProse()
          codeStartLine = lineNumber
          let lang = line.dropFirst(3).trimmingCharacters(in: .whitespaces)
          codeLanguage = lang.isEmpty ? nil : lang
          inCode = true
        }
        continue
      }
      if inCode {
        codeBuffer.append(line)
      } else {
        if proseStartLine == nil { proseStartLine = lineNumber }
        proseBuffer.append(line)
      }
    }

    // Streaming: an unterminated code fence still renders its partial content.
    if inCode {
      flushCode()
    } else {
      flushProse()
    }

    return segments
  }

  /// Splits raw message content on ``` fences. Handles an unterminated trailing
  /// fence (streaming) by emitting whatever code has arrived so far.
  static func segments(from content: String) -> [MarkdownSegment] {
    identifiedSegments(from: content).map(\.segment)
  }
}

struct IdentifiedMarkdownSegment: Identifiable, Equatable {
  let id: Int
  let segment: MarkdownSegment
}

/// Cross-platform-ish pasteboard copy (macOS app target).
func copyToPasteboard(_ string: String) {
  #if canImport(AppKit)
    NSPasteboard.general.clearContents()
    NSPasteboard.general.setString(string, forType: .string)
  #endif
}
