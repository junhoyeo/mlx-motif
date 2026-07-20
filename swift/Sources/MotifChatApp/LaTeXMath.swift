import SwiftUI

// Dependency-free LaTeX math rendering for chat messages.
//
// The default MotifChatApp target is intentionally kept free of external
// packages (see Package.swift), so instead of pulling in a MathJax/KaTeX/
// SwiftMath dependency this renders the common subset of LaTeX that language
// models actually emit — fractions, sub/superscripts, radicals, \boxed, Greek
// letters and operators — using native SwiftUI layout.
//
// Two layers:
//   1. `LaTeXMath.parse` — a pure, testable recursive-descent parser turning a
//      math string into a `[MathNode]` tree (no SwiftUI involved).
//   2. `MathView` — lays the tree out with real stacked fractions, side/under
//      scripts, and radicals.
// Inline vs display splitting (`\(..\)`, `$..$`, `\[..\]`, `$$..$$`) lives in
// `MarkdownMessageView`.

// MARK: - Model

/// A parsed LaTeX math node. Kept separate from the view layer so parsing is
/// deterministic and unit-testable without a running view host.
indirect enum MathNode: Equatable {
  /// A rendered glyph run already mapped to Unicode. `italic` marks single
  /// math variables (rendered slanted) apart from digits/operators (upright).
  case run(String, italic: Bool)
  /// Upright multi-letter text: function names, `\text{…}`, `\mathrm{…}`.
  case text(String)
  /// A horizontal sequence (a `{…}` group).
  case row([MathNode])
  /// A stacked fraction.
  case fraction(numerator: [MathNode], denominator: [MathNode])
  /// A base carrying an optional subscript and/or superscript. `bigOperator`
  /// marks limit-like operators (∑, ∏, ∫, lim) whose scripts stack above and
  /// below in display mode.
  case scripted(base: MathNode, sub: [MathNode]?, sup: [MathNode]?, bigOperator: Bool)
  /// A square root (radicand only; an explicit degree is dropped).
  case sqrt([MathNode])
  /// `\boxed{…}` — content inside a thin rule.
  case boxed([MathNode])
  /// Explicit horizontal spacing (`\,`, `\;`, `\quad`), width in points at the
  /// base font size.
  case space(CGFloat)
}

// MARK: - Parser

enum LaTeXMath {
  /// Parses a LaTeX math-mode string into a node tree. Never throws: anything
  /// unrecognised falls back to a literal glyph run so partial/streaming input
  /// still renders sensibly.
  static func parse(_ source: String) -> [MathNode] {
    var parser = MathParser(Array(source))
    return parser.parseRow(until: nil)
  }
}

private struct MathParser {
  private let chars: [Character]
  private var i = 0

  init(_ chars: [Character]) { self.chars = chars }

  private var current: Character? { i < chars.count ? chars[i] : nil }

  /// Parses nodes until `stop` (e.g. `}`) or end of input.
  mutating func parseRow(until stop: Character?) -> [MathNode] {
    var nodes: [MathNode] = []
    while let c = current {
      if let stop, c == stop { break }
      if c == " " || c == "\n" || c == "\t" { i += 1; continue }
      guard let atom = parseAtom() else { break }
      nodes.append(attachScripts(to: atom.node, bigOperator: atom.bigOperator))
    }
    return nodes
  }

  /// One atom plus a flag marking limit-like operators.
  private mutating func parseAtom() -> (node: MathNode, bigOperator: Bool)? {
    guard let c = current else { return nil }
    switch c {
    case "\\":
      return parseCommand()
    case "{":
      i += 1
      let inner = parseRow(until: "}")
      if current == "}" { i += 1 }
      return (.row(inner), false)
    case "}":
      return nil
    case "^", "_":
      // A dangling script with no base — render the raw marker.
      i += 1
      return (.run(String(c), italic: false), false)
    default:
      i += 1
      return (glyph(for: c), false)
    }
  }

  /// Consumes `_`/`^` following an atom and wraps it in a `.scripted` node.
  private mutating func attachScripts(to base: MathNode, bigOperator: Bool) -> MathNode {
    var sub: [MathNode]?
    var sup: [MathNode]?
    while let c = current, c == "_" || c == "^" {
      i += 1
      let arg = parseScriptArgument()
      if c == "_" { sub = arg } else { sup = arg }
    }
    if sub == nil && sup == nil { return base }
    return .scripted(base: base, sub: sub, sup: sup, bigOperator: bigOperator)
  }

  /// A script argument: a `{…}` group or a single following atom.
  private mutating func parseScriptArgument() -> [MathNode] {
    while let c = current, c == " " { i += 1 }
    guard let c = current else { return [] }
    if c == "{" {
      i += 1
      let inner = parseRow(until: "}")
      if current == "}" { i += 1 }
      return inner
    }
    guard let atom = parseAtom() else { return [] }
    return [atom.node]
  }

  private mutating func parseCommand() -> (node: MathNode, bigOperator: Bool)? {
    i += 1  // consume "\"
    guard let c = current else { return (.run("\\", italic: false), false) }

    // Control symbols: "\" + a single non-letter (\, \; \{ \\ …).
    if !c.isLetter {
      i += 1
      switch c {
      case ",", ":", ";", " ": return (.space(4), false)
      case "!": return (.space(0), false)
      case "\\": return (.space(0), false)  // row break in matrices — ignored
      default: return (.run(String(c), italic: false), false)
      }
    }

    // Command name: "\" + letters.
    var name = ""
    while let ch = current, ch.isLetter { name.append(ch); i += 1 }

    switch name {
    case "frac", "dfrac", "tfrac", "cfrac":
      let num = parseGroupArgument()
      let den = parseGroupArgument()
      return (.fraction(numerator: num, denominator: den), false)
    case "sqrt":
      // Drop an optional degree `[n]`.
      if current == "[" {
        while let ch = current, ch != "]" { i += 1 }
        if current == "]" { i += 1 }
      }
      return (.sqrt(parseGroupArgument()), false)
    case "boxed", "fbox":
      return (.boxed(parseGroupArgument()), false)
    case "text", "mathrm", "operatorname", "mathbf", "mathsf", "mathtt", "mathit", "textbf", "textit", "mathcal", "mathbb":
      // Text-like commands hold ordinary prose ("for all"), so read the brace
      // body verbatim rather than through the math-row parser (which drops
      // whitespace between atoms).
      return (.text(parseRawGroup()), false)
    case "left", "right":
      // Keep the delimiter that follows; `.` means an invisible fence.
      if let d = current {
        i += 1
        if d == "\\" {  // \left\{  etc.
          if let d2 = current { i += 1; return (.run(String(d2), italic: false), false) }
        }
        if d == "." { return (.space(0), false) }
        return (.run(delimiterGlyph(d), italic: false), false)
      }
      return (.space(0), false)
    case "quad": return (.space(10), false)
    case "qquad": return (.space(18), false)
    default:
      if MathSymbols.bigOperators.contains(name) {
        return (.text(MathSymbols.functions[name] ?? name), true)
      }
      if MathSymbols.functions.keys.contains(name) {
        return (.text(MathSymbols.functions[name]!), false)
      }
      if let sym = MathSymbols.symbols[name] {
        return (.run(sym, italic: false), false)
      }
      // Unknown command — show its name upright rather than a stray backslash.
      return (.text(name), false)
    }
  }

  private mutating func parseGroupArgument() -> [MathNode] {
    while let c = current, c == " " { i += 1 }
    guard let c = current else { return [] }
    if c == "{" {
      i += 1
      let inner = parseRow(until: "}")
      if current == "}" { i += 1 }
      return inner
    }
    if c == "\\" {
      if let atom = parseCommand() { return [atom.node] }
      return []
    }
    i += 1
    return [glyph(for: c)]
  }

  /// Maps a bare character to a glyph node (variables slanted, digits upright).
  private func glyph(for c: Character) -> MathNode {
    if c.isLetter { return .run(String(c), italic: true) }
    switch c {
    case "-": return .run("\u{2212}", italic: false)  // minus sign
    case "*": return .run("\u{2217}", italic: false)  // asterisk operator
    default: return .run(String(c), italic: false)
    }
  }

  private func delimiterGlyph(_ c: Character) -> String {
    switch c {
    case "|": return "|"
    case "<": return "\u{27E8}"  // ⟨
    case ">": return "\u{27E9}"  // ⟩
    default: return String(c)
    }
  }

  /// Reads a `{…}` brace body verbatim (whitespace preserved, nested braces
  /// balanced, `\{`/`\}` unescaped), for text-like commands. Falls back to a
  /// single following character when there is no brace group.
  private mutating func parseRawGroup() -> String {
    while let c = current, c == " " { i += 1 }
    guard current == "{" else {
      if let c = current { i += 1; return String(c) }
      return ""
    }
    i += 1  // consume "{"
    var depth = 1
    var out = ""
    while let c = current {
      if c == "\\" {
        i += 1
        if let escaped = current { out.append(escaped); i += 1 }
        continue
      }
      if c == "{" { depth += 1; out.append(c); i += 1; continue }
      if c == "}" {
        depth -= 1
        i += 1
        if depth == 0 { break }
        out.append(c)
        continue
      }
      out.append(c)
      i += 1
    }
    return out
  }
}

// MARK: - Symbol tables

enum MathSymbols {
  /// Limit-like operators whose scripts stack above/below in display mode.
  static let bigOperators: Set<String> = [
    "sum", "prod", "coprod", "int", "iint", "iiint", "oint",
    "lim", "limsup", "liminf", "max", "min", "sup", "inf",
    "argmax", "argmin", "bigcup", "bigcap",
  ]

  /// Commands rendered as upright multi-letter text (function names + the big
  /// operators that have a glyph).
  static let functions: [String: String] = [
    "sin": "sin", "cos": "cos", "tan": "tan", "cot": "cot", "sec": "sec",
    "csc": "csc", "sinh": "sinh", "cosh": "cosh", "tanh": "tanh",
    "arcsin": "arcsin", "arccos": "arccos", "arctan": "arctan",
    "log": "log", "ln": "ln", "lg": "lg", "exp": "exp",
    "det": "det", "dim": "dim", "deg": "deg", "arg": "arg", "gcd": "gcd",
    "ker": "ker", "hom": "hom", "Pr": "Pr", "mod": "mod", "bmod": "mod",
    "lim": "lim", "limsup": "lim sup", "liminf": "lim inf",
    "max": "max", "min": "min", "sup": "sup", "inf": "inf",
    "argmax": "arg max", "argmin": "arg min",
    "sum": "\u{2211}", "prod": "\u{220F}", "coprod": "\u{2210}",
    "int": "\u{222B}", "iint": "\u{222C}", "iiint": "\u{222D}", "oint": "\u{222E}",
    "bigcup": "\u{22C3}", "bigcap": "\u{22C2}",
  ]

  /// Single-glyph LaTeX commands → Unicode.
  static let symbols: [String: String] = [
    // Lowercase Greek
    "alpha": "\u{03B1}", "beta": "\u{03B2}", "gamma": "\u{03B3}", "delta": "\u{03B4}",
    "epsilon": "\u{03B5}", "varepsilon": "\u{03B5}", "zeta": "\u{03B6}", "eta": "\u{03B7}",
    "theta": "\u{03B8}", "vartheta": "\u{03D1}", "iota": "\u{03B9}", "kappa": "\u{03BA}",
    "lambda": "\u{03BB}", "mu": "\u{03BC}", "nu": "\u{03BD}", "xi": "\u{03BE}",
    "pi": "\u{03C0}", "varpi": "\u{03D6}", "rho": "\u{03C1}", "varrho": "\u{03F1}",
    "sigma": "\u{03C3}", "varsigma": "\u{03C2}", "tau": "\u{03C4}", "upsilon": "\u{03C5}",
    "phi": "\u{03C6}", "varphi": "\u{03D5}", "chi": "\u{03C7}", "psi": "\u{03C8}",
    "omega": "\u{03C9}",
    // Uppercase Greek
    "Gamma": "\u{0393}", "Delta": "\u{0394}", "Theta": "\u{0398}", "Lambda": "\u{039B}",
    "Xi": "\u{039E}", "Pi": "\u{03A0}", "Sigma": "\u{03A3}", "Upsilon": "\u{03A5}",
    "Phi": "\u{03A6}", "Psi": "\u{03A8}", "Omega": "\u{03A9}",
    // Relations
    "leq": "\u{2264}", "le": "\u{2264}", "geq": "\u{2265}", "ge": "\u{2265}",
    "neq": "\u{2260}", "ne": "\u{2260}", "approx": "\u{2248}", "equiv": "\u{2261}",
    "sim": "\u{223C}", "simeq": "\u{2243}", "cong": "\u{2245}", "propto": "\u{221D}",
    "ll": "\u{226A}", "gg": "\u{226B}", "prec": "\u{227A}", "succ": "\u{227B}",
    "subset": "\u{2282}", "supset": "\u{2283}", "subseteq": "\u{2286}", "supseteq": "\u{2287}",
    "in": "\u{2208}", "notin": "\u{2209}", "ni": "\u{220B}",
    // Operators
    "times": "\u{00D7}", "div": "\u{00F7}", "cdot": "\u{22C5}", "ast": "\u{2217}",
    "pm": "\u{00B1}", "mp": "\u{2213}", "star": "\u{22C6}", "circ": "\u{2218}",
    "bullet": "\u{2219}", "oplus": "\u{2295}", "otimes": "\u{2297}",
    "cup": "\u{222A}", "cap": "\u{2229}", "setminus": "\u{2216}",
    "wedge": "\u{2227}", "vee": "\u{2228}", "land": "\u{2227}", "lor": "\u{2228}",
    // Arrows
    "to": "\u{2192}", "rightarrow": "\u{2192}", "leftarrow": "\u{2190}",
    "Rightarrow": "\u{21D2}", "Leftarrow": "\u{21D0}", "leftrightarrow": "\u{2194}",
    "Leftrightarrow": "\u{21D4}", "mapsto": "\u{21A6}", "implies": "\u{27F9}",
    "iff": "\u{27FA}", "uparrow": "\u{2191}", "downarrow": "\u{2193}",
    // Misc
    "infty": "\u{221E}", "partial": "\u{2202}", "nabla": "\u{2207}",
    "forall": "\u{2200}", "exists": "\u{2203}", "nexists": "\u{2204}",
    "emptyset": "\u{2205}", "varnothing": "\u{2205}", "aleph": "\u{2135}",
    "hbar": "\u{210F}", "ell": "\u{2113}", "Re": "\u{211C}", "Im": "\u{2111}",
    "angle": "\u{2220}", "triangle": "\u{25B3}", "square": "\u{25A1}",
    "dagger": "\u{2020}", "prime": "\u{2032}", "degree": "\u{00B0}",
    "neg": "\u{00AC}", "lnot": "\u{00AC}", "surd": "\u{221A}",
    "ldots": "\u{2026}", "cdots": "\u{22EF}", "vdots": "\u{22EE}", "ddots": "\u{22F1}",
    "dots": "\u{2026}", "dot": "\u{02D9}",
    "perp": "\u{22A5}", "parallel": "\u{2225}", "nparallel": "\u{2226}",
    "therefore": "\u{2234}", "because": "\u{2235}",
    "lfloor": "\u{230A}", "rfloor": "\u{230B}", "lceil": "\u{2308}", "rceil": "\u{2309}",
    "langle": "\u{27E8}", "rangle": "\u{27E9}",
    "backslash": "\\", "vert": "|", "Vert": "\u{2016}",
  ]
}

// MARK: - Layout

/// Base point size for rendered math. Chosen to sit close to the app's body
/// text so inline math lines up with surrounding prose.
enum MathMetrics {
  static let base: CGFloat = 13
}

/// Renders a parsed `[MathNode]` tree. `display` enables limit-style under/over
/// scripts on big operators.
struct MathView: View {
  let nodes: [MathNode]
  var display: Bool = false

  var body: some View {
    MathRowView(nodes: nodes, size: MathMetrics.base, display: display)
      .fixedSize()
  }
}

private struct MathRowView: View {
  let nodes: [MathNode]
  let size: CGFloat
  var display: Bool = false

  var body: some View {
    HStack(alignment: .center, spacing: size * 0.06) {
      ForEach(Array(nodes.enumerated()), id: \.offset) { _, node in
        MathNodeView(node: node, size: size, display: display)
      }
    }
  }
}

private struct MathNodeView: View {
  let node: MathNode
  let size: CGFloat
  var display: Bool = false

  var body: some View {
    switch node {
    case .run(let s, let italic):
      Text(s)
        .font(.system(size: size, design: .serif))
        .italic(italic)
    case .text(let s):
      Text(s)
        .font(.system(size: size, design: .serif))
    case .row(let inner):
      MathRowView(nodes: inner, size: size, display: display)
    case .space(let w):
      Color.clear.frame(width: w / 13 * size, height: 1)
    case .fraction(let num, let den):
      FractionView(numerator: num, denominator: den, size: size, display: display)
    case .scripted(let base, let sub, let sup, let bigOp):
      ScriptedView(base: base, sub: sub, sup: sup, bigOperator: bigOp, size: size, display: display)
    case .sqrt(let rad):
      SqrtView(radicand: rad, size: size, display: display)
    case .boxed(let inner):
      MathRowView(nodes: inner, size: size, display: display)
        .padding(.horizontal, size * 0.28)
        .padding(.vertical, size * 0.16)
        .overlay(
          RoundedRectangle(cornerRadius: 3)
            .stroke(.primary.opacity(0.55), lineWidth: 1)
        )
    }
  }
}

private struct FractionView: View {
  let numerator: [MathNode]
  let denominator: [MathNode]
  let size: CGFloat
  var display: Bool

  var body: some View {
    let f = size * 0.94
    VStack(spacing: size * 0.14) {
      MathRowView(nodes: numerator, size: f, display: display)
      Rectangle().frame(height: max(1, size * 0.055))
      MathRowView(nodes: denominator, size: f, display: display)
    }
    .fixedSize()
    .padding(.horizontal, size * 0.12)
  }
}

private struct ScriptedView: View {
  let base: MathNode
  let sub: [MathNode]?
  let sup: [MathNode]?
  let bigOperator: Bool
  let size: CGFloat
  var display: Bool

  var body: some View {
    let s = max(8, size * 0.72)
    if bigOperator && display {
      VStack(spacing: size * 0.02) {
        if let sup { MathRowView(nodes: sup, size: s, display: false) }
        MathNodeView(node: base, size: size, display: display)
        if let sub { MathRowView(nodes: sub, size: s, display: false) }
      }
    } else {
      HStack(alignment: .center, spacing: size * 0.02) {
        MathNodeView(node: base, size: size, display: display)
        VStack(alignment: .leading, spacing: 0) {
          if let sup {
            MathRowView(nodes: sup, size: s, display: false)
          } else {
            Color.clear.frame(width: 0, height: s * 0.55)
          }
          if let sub {
            MathRowView(nodes: sub, size: s, display: false)
          } else {
            Color.clear.frame(width: 0, height: s * 0.55)
          }
        }
      }
    }
  }
}

private struct SqrtView: View {
  let radicand: [MathNode]
  let size: CGFloat
  var display: Bool

  var body: some View {
    HStack(alignment: .center, spacing: 1) {
      Text("\u{221A}")
        .font(.system(size: size * 1.15, design: .serif))
      MathRowView(nodes: radicand, size: size, display: display)
        .padding(.top, max(1, size * 0.1))
        .overlay(alignment: .top) {
          Rectangle().frame(height: max(1, size * 0.055))
        }
        .padding(.trailing, size * 0.1)
    }
  }
}

// MARK: - Wrapping flow layout

/// A simple left-to-right flow layout that wraps to the next line when a
/// subview would overflow the proposed width. Used to interleave prose words
/// (`Text`) with inline `MathView`s while still wrapping naturally. Items on a
/// line are vertically centered so prose sits on the math axis of taller atoms
/// (fractions, radicals) rather than clinging to their tops.
struct WrapLayout: Layout {
  var spacing: CGFloat = 0
  var lineSpacing: CGFloat = 3

  private struct Line {
    var indices: [Int] = []
    var height: CGFloat = 0
  }

  private func layoutLines(_ subviews: Subviews, maxWidth: CGFloat) -> (lines: [Line], sizes: [CGSize], width: CGFloat) {
    var lines: [Line] = []
    var sizes: [CGSize] = []
    var line = Line()
    var x: CGFloat = 0
    var widest: CGFloat = 0

    for (index, subview) in subviews.enumerated() {
      let sz = subview.sizeThatFits(.unspecified)
      sizes.append(sz)
      if !line.indices.isEmpty && x + sz.width > maxWidth {
        lines.append(line)
        line = Line()
        x = 0
      }
      line.indices.append(index)
      line.height = max(line.height, sz.height)
      x += sz.width + spacing
      widest = max(widest, x)
    }
    if !line.indices.isEmpty { lines.append(line) }
    return (lines, sizes, widest)
  }

  func sizeThatFits(proposal: ProposedViewSize, subviews: Subviews, cache: inout ()) -> CGSize {
    let maxWidth = proposal.width ?? .infinity
    let (lines, _, widest) = layoutLines(subviews, maxWidth: maxWidth)
    let height = lines.reduce(0) { $0 + $1.height } + CGFloat(max(0, lines.count - 1)) * lineSpacing
    let width = maxWidth.isFinite ? min(maxWidth, widest) : widest
    return CGSize(width: width, height: height)
  }

  func placeSubviews(in bounds: CGRect, proposal: ProposedViewSize, subviews: Subviews, cache: inout ()) {
    let (lines, sizes, _) = layoutLines(subviews, maxWidth: bounds.width)
    var y = bounds.minY
    for line in lines {
      var x = bounds.minX
      for index in line.indices {
        let sz = sizes[index]
        subviews[index].place(
          at: CGPoint(x: x, y: y + (line.height - sz.height) / 2),
          anchor: .topLeading,
          proposal: ProposedViewSize(sz)
        )
        x += sz.width + spacing
      }
      y += line.height + lineSpacing
    }
  }
}

// MARK: - Inline delimiter splitting

/// A run within a single prose line: literal text or inline math source.
enum InlineMathSpan: Equatable {
  case text(String)
  case math(String)
}

/// Splits a prose line into text/inline-math runs on `\(…\)` and `$…$`
/// delimiters. Unterminated openers stay literal so streaming input never
/// renders half-parsed math.
enum MathInline {
  static func hasMath(_ line: String) -> Bool {
    spans(line).contains { if case .math = $0 { return true } else { return false } }
  }

  static func spans(_ line: String) -> [InlineMathSpan] {
    let chars = Array(line)
    var spans: [InlineMathSpan] = []
    var text = ""
    var i = 0

    func flushText() {
      if !text.isEmpty { spans.append(.text(text)); text = "" }
    }

    while i < chars.count {
      let c = chars[i]

      // Markdown inline code spans win over math: a `…` run keeps its content
      // literal so `` `$x$` `` renders as code, not an equation. Multi-backtick
      // fences (``code``) are matched by run length.
      if c == "`" {
        var runLength = 0
        var j = i
        while j < chars.count, chars[j] == "`" { runLength += 1; j += 1 }
        if let closeStart = findBacktickRun(chars, from: j, length: runLength) {
          let end = closeStart + runLength
          text.append(contentsOf: chars[i..<end])
          i = end
          continue
        } else {
          // Unterminated fence — keep the backticks literal and move on.
          text.append(contentsOf: chars[i..<j])
          i = j
          continue
        }
      }

      // \( … \)  and inline \[ … \]  (unambiguous LaTeX delimiters).
      if c == "\\", i + 1 < chars.count, chars[i + 1] == "(" || chars[i + 1] == "[" {
        let closer: Character = chars[i + 1] == "(" ? ")" : "]"
        if let closeStart = findEscapedDelimiter(chars, from: i + 2, closer: closer) {
          flushText()
          spans.append(.math(String(chars[(i + 2)..<closeStart])))
          i = closeStart + 2
          continue
        }
      }

      // $ … $  (single dollar, never $$; heuristic-guarded against currency).
      if c == "$", !(i + 1 < chars.count && chars[i + 1] == "$"),
        !(i > 0 && chars[i - 1] == "\\") {
        if let close = findDollar(chars, from: i + 1) {
          let inner = String(chars[(i + 1)..<close])
          if isLikelyMath(inner) {
            flushText()
            spans.append(.math(inner))
            i = close + 1
            continue
          }
        }
      }

      text.append(c)
      i += 1
    }

    flushText()
    return spans
  }

  /// Finds the start index of a backtick run of exactly `length` (the closing
  /// fence of an inline code span). A longer run does not close it.
  private static func findBacktickRun(_ chars: [Character], from: Int, length: Int) -> Int? {
    var i = from
    while i < chars.count {
      if chars[i] == "`" {
        var run = 0
        var j = i
        while j < chars.count, chars[j] == "`" { run += 1; j += 1 }
        if run == length { return i }
        i = j
      } else {
        i += 1
      }
    }
    return nil
  }

  /// Finds the start index of a `\<closer>` escaped delimiter (e.g. `\)`).
  private static func findEscapedDelimiter(_ chars: [Character], from: Int, closer: Character) -> Int? {
    var i = from
    while i + 1 < chars.count {
      if chars[i] == "\\" && chars[i + 1] == closer { return i }
      i += 1
    }
    return nil
  }

  /// Finds the next unescaped single `$` (not part of `$$`).
  private static func findDollar(_ chars: [Character], from: Int) -> Int? {
    var i = from
    while i < chars.count {
      if chars[i] == "$", chars[i - 1] != "\\" {
        if i + 1 < chars.count, chars[i + 1] == "$" { return nil }  // $$ → display, bail
        return i
      }
      i += 1
    }
    return nil
  }

  /// Guards `$…$` against ordinary prose/currency ("$5 and $10"). Accepts a
  /// LaTeX command/brace/script, a bare symbol token, or a short operator
  /// expression.
  static func isLikelyMath(_ inner: String) -> Bool {
    let t = inner.trimmingCharacters(in: .whitespaces)
    guard !t.isEmpty else { return false }
    if t.contains(where: { "\\^_{}".contains($0) }) { return true }
    if t.allSatisfy({ $0.isLetter || $0.isNumber }) { return true }
    let operators = Set("+-=*/<>")
    if t.count <= 24, t.contains(where: { operators.contains($0) }), !t.contains(", ") {
      return true
    }
    return false
  }
}

/// A wrapping-flow token: a prose word (carrying its trailing space, with inline
/// markdown attributes preserved) or an inline math atom.
struct MathLineToken: Identifiable, Equatable {
  enum Kind: Equatable {
    case word(AttributedString)
    case math(String)
  }

  let id: Int
  let kind: Kind

  /// Sentinel (Unicode private-use) standing in for a math span while the whole
  /// line is parsed as Markdown once, so emphasis that brackets math (e.g.
  /// `**energy $E$ is conserved**`) resolves instead of leaving unmatched
  /// markers on either side of the split.
  private static let mathPlaceholder: Character = "\u{E000}"

  static func tokens(from line: String) -> [MathLineToken] {
    // Replace each math span with a placeholder, parse the combined line as
    // Markdown once, then split back into words + math atoms.
    var combined = ""
    var mathSources: [String] = []
    for span in MathInline.spans(line) {
      switch span {
      case .text(let text):
        combined += text
      case .math(let source):
        combined.append(mathPlaceholder)
        mathSources.append(source)
      }
    }

    let attr = inlineMarkdownAttributedString(combined)
    var out: [MathLineToken] = []
    var id = 0
    var mathIndex = 0
    var current = AttributedString()

    func flushWord() {
      if !current.characters.isEmpty {
        out.append(.init(id: id, kind: .word(current)))
        id += 1
        current = AttributedString()
      }
    }

    let characters = attr.characters
    var idx = characters.startIndex
    while idx < characters.endIndex {
      let next = characters.index(after: idx)
      let ch = characters[idx]
      if ch == mathPlaceholder {
        flushWord()
        if mathIndex < mathSources.count {
          out.append(.init(id: id, kind: .math(mathSources[mathIndex])))
          id += 1
          mathIndex += 1
        }
        idx = next
        continue
      }
      current.append(AttributedString(attr[idx..<next]))
      // Break word tokens on spaces so wrapping stays word-granular; each token
      // keeps its trailing space and its Markdown attributes.
      if ch == " " { flushWord() }
      idx = next
    }
    flushWord()
    return out
  }
}

/// Renders one prose line that mixes text and inline math, wrapping naturally.
struct MathInlineView: View {
  let line: String

  private var tokens: [MathLineToken] { MathLineToken.tokens(from: line) }

  var body: some View {
    WrapLayout(spacing: 0, lineSpacing: 3) {
      ForEach(tokens) { token in
        switch token.kind {
        case .word(let word):
          Text(word)
            .font(.system(size: MathMetrics.base))
            .textSelection(.enabled)
        case .math(let source):
          MathView(nodes: LaTeXMath.parse(source))
        }
      }
    }
  }
}
