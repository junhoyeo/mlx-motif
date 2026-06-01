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
    .frame(maxWidth: .infinity, alignment: .leading)
  }
}

/// Markdown prose, parsed inline so paragraphs/lists/bold/inline-code survive,
/// with a plain-text fallback if parsing throws.
private struct ProseView: View {
  let text: String

  var body: some View {
    if let attributed = try? AttributedString(
      markdown: text,
      options: .init(interpretedSyntax: .inlineOnlyPreservingWhitespace)
    ) {
      Text(attributed)
        .textSelection(.enabled)
        .frame(maxWidth: .infinity, alignment: .leading)
    } else {
      Text(text)
        .textSelection(.enabled)
        .frame(maxWidth: .infinity, alignment: .leading)
    }
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
          .font(.system(.body, design: .monospaced))
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
