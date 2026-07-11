import Foundation

/// Deterministic prompt + output sanitizer for model-generated conversation
/// titles. Kept pure (no model, no I/O) so the exact prompt and the
/// cleanup rules are unit-testable; the app runs this against the lightweight
/// native checkpoint and feeds the first user message in. Mirrors the shape of
/// opencode's title agent (a small model, one-shot, first user turn only) but
/// titles in the user's own language rather than forcing one.
public enum MotifTitleGeneration {
    /// System prompt for the title turn. The user's first message is supplied as
    /// the user turn; the model must reply with only the title.
    public static func systemPrompt() -> String {
        [
            "You are a title generator. Output ONLY a short conversation title and nothing else.",
            "",
            "Rules:",
            "- 6 words or fewer, on a single line.",
            "- Write the title in the SAME language the user wrote in.",
            "- No surrounding quotes, no trailing punctuation, no explanation, no preamble.",
            "- Capture the main topic or request so the user can find the chat later.",
            "- Keep technical terms, filenames, numbers, and error codes exact.",
            "- NEVER answer the message or hold a conversation — only title it.",
            "- Never refuse or comment on the input; always produce a meaningful title.",
            "",
            // Few-shot examples anchor the "title, don't answer" behaviour. A
            // small model otherwise starts answering non-English inputs instead
            // of titling them; pairing inputs with titles in several languages
            // fixes that while honoring "same language as the user".
            "Examples:",
            "message: debug 500 errors in production",
            "title: Debug production 500 errors",
            "message: can you add refresh token support to auth.ts",
            "title: Add refresh token to auth.ts",
            "message: 쿠버네티스 파드가 자꾸 죽어요 왜죠",
            "title: 쿠버네티스 파드 크래시 조사",
            "message: 今日の天気はどう",
            "title: 今日の天気の問い合わせ",
            "message: hello",
            "title: Casual greeting",
        ].joined(separator: "\n")
    }

    /// Cleans raw model output into a single-line title. Strips any `<think>`
    /// reasoning block, takes the first non-empty line, removes wrapping quotes
    /// and trailing punctuation, and caps the length. Returns `fallback` when the
    /// cleaned result is empty (e.g. the model emitted only reasoning or blanks),
    /// so a title is always produced.
    public static func sanitize(_ raw: String, fallback: String, maxLength: Int = 48) -> String {
        // Drop any captured reasoning the model may have emitted first.
        var text = raw
        while let open = text.range(of: "<think>"),
              let close = text.range(of: "</think>", range: open.upperBound..<text.endIndex) {
            text.removeSubrange(open.lowerBound..<close.upperBound)
        }
        // A dangling, unclosed <think> means everything after it is reasoning.
        if let open = text.range(of: "<think>") {
            text.removeSubrange(open.lowerBound..<text.endIndex)
        }

        guard var line = text
            .split(whereSeparator: \.isNewline)
            .map({ $0.trimmingCharacters(in: .whitespaces) })
            .first(where: { !$0.isEmpty })
        else { return fallback }

        // Drop a leading "title:" / "제목:" style label the model sometimes echoes
        // from the few-shot format.
        for label in ["title:", "Title:", "TITLE:"] where line.hasPrefix(label) {
            line = String(line.dropFirst(label.count)).trimmingCharacters(in: .whitespaces)
        }
        // A small model sometimes appends a quoted restatement after a colon
        // ("Connect Postgres to API: \"Establish ...\""). A legitimate title
        // colon is never followed by an opening quote, so cut there.
        if let range = line.range(of: ": \"") ?? line.range(of: ": “") {
            line = String(line[line.startIndex..<range.lowerBound]).trimmingCharacters(in: .whitespaces)
        }

        // Strip a single layer of wrapping quotes/backticks the model likes to add.
        for quote in ["\"", "'", "`", "“", "”"] where line.hasPrefix(quote) && line.hasSuffix(quote) && line.count >= 2 {
            line = String(line.dropFirst().dropLast()).trimmingCharacters(in: .whitespaces)
        }
        // Trim trailing sentence punctuation ("... debugging." -> "... debugging").
        line = line.trimmingCharacters(in: CharacterSet(charactersIn: " .,:;·"))

        guard !line.isEmpty else { return fallback }
        if line.count <= maxLength { return line }
        return line.prefix(maxLength).trimmingCharacters(in: .whitespaces) + "…"
    }
}
