public struct ThinkStreamFilter: Sendable {
    public static let openTag = "<think>"
    public static let closeTag = "</think>"

    public var mode: MotifThinkMode
    public private(set) var isInsideThinkBlock: Bool
    public private(set) var bufferedPartialTag: String
    public private(set) var capturedReasoning: String

    public init(mode: MotifThinkMode = .visible, startsInsideThinkBlock: Bool = false) {
        self.mode = mode
        self.isInsideThinkBlock = startsInsideThinkBlock
        self.bufferedPartialTag = ""
        self.capturedReasoning = ""
    }

    public mutating func feed(_ text: String) -> String {
        guard mode != .visible else { return text }

        bufferedPartialTag += text
        var output = ""

        while !bufferedPartialTag.isEmpty {
            if isInsideThinkBlock {
                guard let closeRange = bufferedPartialTag.range(of: Self.closeTag) else {
                    let keep = Self.partialTagSuffixLength(bufferedPartialTag, tag: Self.closeTag)
                    let consumeEnd = bufferedPartialTag.index(bufferedPartialTag.endIndex, offsetBy: -keep)
                    let consumed = String(bufferedPartialTag[..<consumeEnd])
                    if mode == .captured { capturedReasoning += consumed }
                    bufferedPartialTag = String(bufferedPartialTag[consumeEnd...])
                    return output
                }

                let inside = String(bufferedPartialTag[..<closeRange.lowerBound])
                if mode == .captured { capturedReasoning += inside }
                isInsideThinkBlock = false
                bufferedPartialTag = String(bufferedPartialTag[closeRange.upperBound...])
            } else {
                guard let openRange = bufferedPartialTag.range(of: Self.openTag) else {
                    let keep = Self.partialTagSuffixLength(bufferedPartialTag, tag: Self.openTag)
                    let emitEnd = bufferedPartialTag.index(bufferedPartialTag.endIndex, offsetBy: -keep)
                    output += String(bufferedPartialTag[..<emitEnd])
                    bufferedPartialTag = String(bufferedPartialTag[emitEnd...])
                    return output
                }

                output += String(bufferedPartialTag[..<openRange.lowerBound])
                isInsideThinkBlock = true
                bufferedPartialTag = String(bufferedPartialTag[openRange.upperBound...])
            }
        }

        return output
    }

    public mutating func finish() -> String {
        defer { bufferedPartialTag = "" }
        guard mode != .visible else { return bufferedPartialTag }
        if isInsideThinkBlock {
            if mode == .captured { capturedReasoning += bufferedPartialTag }
            return ""
        }
        return bufferedPartialTag
    }

    static func partialTagSuffixLength(_ buffer: String, tag: String) -> Int {
        guard !buffer.isEmpty, tag.count > 1 else { return 0 }
        let maxCount = min(buffer.count, tag.count - 1)
        guard maxCount > 0 else { return 0 }

        for count in stride(from: maxCount, through: 1, by: -1) {
            let suffix = String(buffer.suffix(count))
            if tag.hasPrefix(suffix) { return count }
        }
        return 0
    }
}

public func promptStartsInsideThinkBlock(_ prompt: String) -> Bool {
    prompt.trimmingCharacters(in: .whitespacesAndNewlines).hasSuffix(ThinkStreamFilter.openTag)
}
