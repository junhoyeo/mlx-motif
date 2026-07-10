import Foundation

/// Prompt-based tool/function calling for the native Swift Motif server.
///
/// This is the Swift port of `mlx_motif.tool_calls` (Python reference). It is
/// intentionally a faithful port so both servers behave identically:
///
///   * ``buildToolsPreamble(_:)`` produces a **byte-identical** system preamble
///     to the Python `build_tools_preamble` for the same tools JSON. The
///     cross-language golden test (`MotifToolCallingTests`) pins this against
///     the shared fixture `tests/fixtures/tool_preamble_cases.json`.
///   * ``parseToolCall(text:tools:)`` ports `parse_tool_call`: it scans the raw
///     model output left-to-right for the FIRST balanced `{...}` that decodes to
///     a recognised tool-call shape (`{"tool_call": {...}}` or the bare
///     `{"name","arguments"}` OpenAI form), tolerating the small model's looping
///     and surrounding prose.
///
/// Execution stays client-side for the server (matching OpenAI semantics); the
/// execution *loop* is the Python-CLI/app-level feature.
public enum MotifToolCalling {
    // MARK: - Preamble

    /// Build the deterministic tools system preamble.
    ///
    /// `tools` is the OpenAI `tools` array as decoded from JSON
    /// (`[[String: Any]]`, each `{"type","function":{...}}`). The output matches
    /// Python's `build_tools_preamble` exactly, including the sorted-key,
    /// space-separated JSON rendering of each tool's `parameters` schema.
    public static func buildToolsPreamble(_ tools: [Any]) -> String {
        var lines: [String] = [
            "You have access to the following tools. When a tool is needed to "
                + "answer the user, respond with a SINGLE JSON object on its own and "
                + "nothing else, in this exact form:",
            "{\"tool_call\": {\"name\": \"<tool name>\", \"arguments\": {<json arguments>}}}",
            "Do not wrap the JSON in markdown fences. Emit it exactly once. If no "
                + "tool is needed, answer the user normally in plain text.",
            "",
            "Available tools:",
        ]
        for tool in tools {
            guard let toolDict = tool as? [String: Any],
                  let fn = toolDict["function"] as? [String: Any]
            else { continue }
            guard let name = fn["name"] as? String, !name.isEmpty else { continue }
            let description = fn["description"] as? String ?? ""
            // Python: `fn.get("parameters", {})` -> default empty object.
            let parameters = fn["parameters"] ?? [String: Any]()
            // Python: `f"- {name}: {description}".rstrip()`.
            lines.append(rstrip("- \(name): \(description)"))
            lines.append("  parameters schema: \(CanonicalJSON.serialize(parameters))")
        }
        return lines.joined(separator: "\n")
    }

    /// Strip trailing whitespace, matching Python `str.rstrip()` (spaces, tabs,
    /// newlines, carriage returns, form feeds, vertical tabs).
    private static func rstrip(_ s: String) -> String {
        var scalars = Array(s.unicodeScalars)
        let ws: Set<Unicode.Scalar> = [" ", "\t", "\n", "\r", "\u{0B}", "\u{0C}"]
        while let last = scalars.last, ws.contains(last) { scalars.removeLast() }
        var view = String.UnicodeScalarView()
        view.append(contentsOf: scalars)
        return String(view)
    }

    // MARK: - Parsing

    /// A parsed tool call: the tool `name` and its decoded `arguments` object.
    public struct ParsedToolCall: Equatable, Sendable {
        public let name: String
        public let arguments: [String: MotifJSONValue]

        public init(name: String, arguments: [String: MotifJSONValue]) {
            self.name = name
            self.arguments = arguments
        }
    }

    /// Extract the FIRST valid tool call from raw model `text`.
    ///
    /// Returns `nil` when no recognised tool call is present. When `toolNames`
    /// is non-nil, the call's name must be one of them (unknown names are
    /// skipped and scanning continues) — mirroring the Python `tools` filter.
    public static func parseToolCall(text: String, toolNames: Set<String>?) -> ParsedToolCall? {
        if text.isEmpty { return nil }
        for candidate in balancedJSONObjects(in: text) {
            guard let data = candidate.data(using: .utf8),
                  let obj = try? JSONSerialization.jsonObject(with: data) as? [String: Any]
            else { continue }
            guard let call = normaliseToolCall(obj) else { continue }
            if let toolNames, !toolNames.contains(call.name) { continue }
            return call
        }
        return nil
    }

    /// Convenience: derive the allowed tool-name set from a decoded `tools`
    /// array (or `nil` to accept any name).
    public static func toolNames(from tools: [Any]?) -> Set<String>? {
        guard let tools else { return nil }
        var names = Set<String>()
        for tool in tools {
            if let toolDict = tool as? [String: Any],
               let fn = toolDict["function"] as? [String: Any],
               let name = fn["name"] as? String {
                names.insert(name)
            }
        }
        return names
    }

    private static func normaliseToolCall(_ obj: [String: Any]) -> ParsedToolCall? {
        // Accept `{"tool_call": {...}}` or the bare object itself.
        let inner: [String: Any]
        if let wrapped = obj["tool_call"] as? [String: Any] {
            inner = wrapped
        } else {
            inner = obj
        }
        guard let name = inner["name"] as? String, !name.isEmpty else { return nil }
        // `arguments` defaults to `{}` when omitted; a present non-object is
        // rejected (parity with the Python `isinstance(arguments, dict)` check).
        let argumentsAny: Any = inner["arguments"] ?? [String: Any]()
        guard let argumentsDict = argumentsAny as? [String: Any] else { return nil }
        return ParsedToolCall(name: name, arguments: MotifJSONValue.object(from: argumentsDict))
    }

    /// Yield each top-level balanced `{...}` substring in `text`, left to right.
    ///
    /// String-aware (ignores braces inside JSON strings, respects backslash
    /// escapes). Ports `_iter_json_objects` — including its early stop when an
    /// opener has no balanced closer.
    private static func balancedJSONObjects(in text: String) -> [String] {
        let scalars = Array(text.unicodeScalars)
        let n = scalars.count
        var results: [String] = []
        var i = 0
        while i < n {
            if scalars[i] != "{" {
                i += 1
                continue
            }
            var depth = 0
            var inString = false
            var escape = false
            var end = -1
            var j = i
            while j < n {
                let c = scalars[j]
                if inString {
                    if escape {
                        escape = false
                    } else if c == "\\" {
                        escape = true
                    } else if c == "\"" {
                        inString = false
                    }
                    j += 1
                    continue
                }
                if c == "\"" {
                    inString = true
                } else if c == "{" {
                    depth += 1
                } else if c == "}" {
                    depth -= 1
                    if depth == 0 {
                        end = j
                        break
                    }
                }
                j += 1
            }
            if end == -1 {
                // No balanced closer for this opener; nothing further can form a
                // complete object starting at or before here.
                break
            }
            var view = String.UnicodeScalarView()
            view.append(contentsOf: scalars[i...end])
            results.append(String(view))
            i = end + 1
        }
        return results
    }
}

extension MotifJSONValue {
    /// Convert a `JSONSerialization`-decoded value into a `MotifJSONValue`.
    static func from(_ value: Any) -> MotifJSONValue {
        if value is NSNull { return .null }
        if let number = value as? NSNumber {
            if CFGetTypeID(number) == CFBooleanGetTypeID() { return .bool(number.boolValue) }
            switch CFNumberGetType(number) {
            case .float32Type, .float64Type, .floatType, .doubleType, .cgFloatType:
                return .double(number.doubleValue)
            default:
                return .int(number.intValue)
            }
        }
        if let s = value as? String { return .string(s) }
        if let array = value as? [Any] { return .array(array.map { from($0) }) }
        if let dict = value as? [String: Any] { return .object(object(from: dict)) }
        return .null
    }

    /// Convert a `[String: Any]` object into `[String: MotifJSONValue]`.
    static func object(from dict: [String: Any]) -> [String: MotifJSONValue] {
        var out: [String: MotifJSONValue] = [:]
        for (key, value) in dict { out[key] = from(value) }
        return out
    }

    /// Bridge back to a `JSONSerialization`-compatible `Any` value.
    public var anyValue: Any {
        switch self {
        case .string(let value): return value
        case .int(let value): return value
        case .double(let value): return value
        case .bool(let value): return value
        case .object(let value): return value.mapValues { $0.anyValue }
        case .array(let value): return value.map { $0.anyValue }
        case .null: return NSNull()
        }
    }
}

/// Canonical JSON serializer matching Python `json.dumps(obj, sort_keys=True)`
/// with its default options (`ensure_ascii=True`, item separator `", "`, key
/// separator `": "`). Used to render tool `parameters` schemas byte-identically
/// to the Python preamble builder.
enum CanonicalJSON {
    static func serialize(_ value: Any) -> String {
        if value is NSNull { return "null" }

        // Bool must be checked before the general NSNumber path because
        // `true`/`false` bridge to NSNumber.
        if let number = value as? NSNumber, isBool(number) {
            return number.boolValue ? "true" : "false"
        }
        if let number = value as? NSNumber {
            if isFloat(number) {
                return serializeDouble(number.doubleValue)
            }
            return String(number.int64Value)
        }
        if let b = value as? Bool { return b ? "true" : "false" }
        if let i = value as? Int { return String(i) }
        if let d = value as? Double { return serializeDouble(d) }
        if let s = value as? String { return serializeString(s) }
        if let array = value as? [Any] {
            return "[" + array.map { serialize($0) }.joined(separator: ", ") + "]"
        }
        if let dict = value as? [String: Any] {
            let pairs = dict.keys.sorted().map { key in
                serializeString(key) + ": " + serialize(dict[key]!)
            }
            return "{" + pairs.joined(separator: ", ") + "}"
        }
        // Unknown types fall back to a JSON null rather than crashing.
        return "null"
    }

    private static func isBool(_ number: NSNumber) -> Bool {
        CFGetTypeID(number) == CFBooleanGetTypeID()
    }

    private static func isFloat(_ number: NSNumber) -> Bool {
        let type = CFNumberGetType(number)
        switch type {
        case .float32Type, .float64Type, .floatType, .doubleType, .cgFloatType:
            return true
        default:
            return false
        }
    }

    private static func serializeDouble(_ d: Double) -> String {
        if d == d.rounded() && abs(d) < 1e16 {
            // Python prints integral floats as `10.0`.
            return String(format: "%.1f", d)
        }
        return String(d)
    }

    /// Escape a string like Python `json.dumps` with `ensure_ascii=True`.
    static func serializeString(_ s: String) -> String {
        var out = "\""
        for scalar in s.unicodeScalars {
            switch scalar {
            case "\"": out += "\\\""
            case "\\": out += "\\\\"
            case "\u{08}": out += "\\b"
            case "\u{0C}": out += "\\f"
            case "\n": out += "\\n"
            case "\r": out += "\\r"
            case "\t": out += "\\t"
            default:
                let v = scalar.value
                if v < 0x20 {
                    out += String(format: "\\u%04x", v)
                } else if v < 0x80 {
                    out.unicodeScalars.append(scalar)
                } else if v <= 0xFFFF {
                    out += String(format: "\\u%04x", v)
                } else {
                    // Encode as a UTF-16 surrogate pair (Python does the same).
                    let adjusted = v - 0x10000
                    let high = 0xD800 + (adjusted >> 10)
                    let low = 0xDC00 + (adjusted & 0x3FF)
                    out += String(format: "\\u%04x\\u%04x", high, low)
                }
            }
        }
        out += "\""
        return out
    }
}
