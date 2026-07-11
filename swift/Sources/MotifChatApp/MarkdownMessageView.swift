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
      ForEach(Array(MarkdownSegment.segments(from: content).enumerated()), id: \.offset) { _, segment in
        switch segment {
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

  private struct Block: Identifiable {
    let id = UUID()
    let kind: Kind
    let text: String
    enum Kind { case paragraph, bullet, numbered(String), heading(Int) }
  }

  private var blocks: [Block] {
    var result: [Block] = []
    for rawLine in text.components(separatedBy: "\n") {
      let line = rawLine
      let trimmed = line.trimmingCharacters(in: .whitespaces)
      if trimmed.isEmpty { continue }

      // Heading: 1–6 leading #s followed by a space ("### Title").
      if trimmed.hasPrefix("#") {
        let hashes = trimmed.prefix(while: { $0 == "#" })
        let rest = trimmed.dropFirst(hashes.count)
        if (1...6).contains(hashes.count), rest.first == " " {
          result.append(.init(
            kind: .heading(hashes.count),
            text: rest.trimmingCharacters(in: .whitespaces)
          ))
          continue
        }
      }
      // Bullet: -, *, or • followed by a space.
      if let marker = ["- ", "* ", "• "].first(where: { trimmed.hasPrefix($0) }) {
        result.append(.init(kind: .bullet, text: String(trimmed.dropFirst(marker.count))))
        continue
      }
      // Numbered: "1. ", "2) " etc.
      if let dotRange = trimmed.range(of: #"^\d+[.)]\s+"#, options: .regularExpression) {
        let number = trimmed[trimmed.startIndex..<dotRange.upperBound]
          .trimmingCharacters(in: CharacterSet(charactersIn: ".) "))
        result.append(.init(kind: .numbered(number), text: String(trimmed[dotRange.upperBound...])))
        continue
      }
      result.append(.init(kind: .paragraph, text: trimmed))
    }
    return result
  }

  var body: some View {
    VStack(alignment: .leading, spacing: 4) {
      ForEach(blocks) { block in
        switch block.kind {
        case .paragraph:
          inlineText(block.text)
        case .heading(let level):
          inlineText(block.text)
            .font(headingFont(level))
            .fontWeight(.semibold)
            // Extra air above a heading separates sections without needing
            // markdown's blank-line semantics (blank lines are dropped here).
            .padding(.top, 6)
        case .bullet:
          HStack(alignment: .firstTextBaseline, spacing: 8) {
            Text("•").foregroundStyle(.secondary)
            inlineText(block.text)
          }
        case .numbered(let n):
          HStack(alignment: .firstTextBaseline, spacing: 8) {
            Text("\(n).").foregroundStyle(.secondary).monospacedDigit()
            inlineText(block.text)
          }
        }
      }
    }
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
    let attributed = (try? AttributedString(
      markdown: line,
      options: .init(interpretedSyntax: .inlineOnlyPreservingWhitespace)
    )) ?? AttributedString(line)
    Text(attributed)
      .textSelection(.enabled)
      .fixedSize(horizontal: false, vertical: true)
  }
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
      }
      .padding(.horizontal, 10)
      .padding(.vertical, 6)

      Divider()

      ScrollView(.horizontal, showsIndicators: false) {
        Text(code)
          // One step below body — code blocks read better slightly smaller
          // than prose (mono stays: this is literal code content).
          .font(.system(.callout, design: .monospaced))
          .textSelection(.enabled)
          .padding(10)
          .frame(maxWidth: .infinity, alignment: .leading)
      }
    }
    .background(.black.opacity(0.06), in: RoundedRectangle(cornerRadius: 8, style: .continuous))
    .overlay(
      RoundedRectangle(cornerRadius: 8, style: .continuous)
        .stroke(.quaternary, lineWidth: 1)
    )
  }
}

/// One parsed segment of message content.
enum MarkdownSegment: Equatable {
  case prose(String)
  case code(String, language: String?)

  /// Splits raw message content on ``` fences. Handles an unterminated trailing
  /// fence (streaming) by emitting whatever code has arrived so far.
  static func segments(from content: String) -> [MarkdownSegment] {
    guard content.contains("```") else {
      let trimmed = content.trimmingCharacters(in: .whitespacesAndNewlines)
      return trimmed.isEmpty ? [] : [.prose(content)]
    }

    var segments: [MarkdownSegment] = []
    let lines = content.components(separatedBy: "\n")
    var proseBuffer: [String] = []
    var codeBuffer: [String] = []
    var codeLanguage: String?
    var inCode = false

    func flushProse() {
      let joined = proseBuffer.joined(separator: "\n")
      if !joined.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
        segments.append(.prose(joined))
      }
      proseBuffer.removeAll()
    }

    func flushCode() {
      segments.append(.code(codeBuffer.joined(separator: "\n"), language: codeLanguage))
      codeBuffer.removeAll()
      codeLanguage = nil
    }

    for line in lines {
      if line.hasPrefix("```") {
        if inCode {
          // Closing fence.
          flushCode()
          inCode = false
        } else {
          // Opening fence — capture optional language tag.
          flushProse()
          let lang = line.dropFirst(3).trimmingCharacters(in: .whitespaces)
          codeLanguage = lang.isEmpty ? nil : lang
          inCode = true
        }
        continue
      }
      if inCode {
        codeBuffer.append(line)
      } else {
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
}

/// Cross-platform-ish pasteboard copy (macOS app target).
func copyToPasteboard(_ string: String) {
  #if canImport(AppKit)
  NSPasteboard.general.clearContents()
  NSPasteboard.general.setString(string, forType: .string)
  #endif
}
