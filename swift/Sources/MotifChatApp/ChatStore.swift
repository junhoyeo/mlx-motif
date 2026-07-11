import Foundation
import MotifKit
#if MOTIFKIT_ENABLE_MLX
import MotifKitMLX
#endif

enum MotifChatBackendMode: String, CaseIterable, Identifiable {
    case openAICompatible
    case nativeMLX

    var id: String { rawValue }

    var label: String {
        switch self {
        case .openAICompatible:
            "OpenAI-compatible endpoint"
        case .nativeMLX:
            "Native MLX checkpoint"
        }
    }

    var systemImage: String {
        switch self {
        case .openAICompatible:
            "network"
        case .nativeMLX:
            "cpu"
        }
    }
}

/// Strategy for keeping a conversation within the model context window. Only
/// `.slidingWindow` is implemented today; `.summarize` is reserved so the UI and
/// persistence can grow into it without another migration.
enum MotifCompactionMode: String, CaseIterable, Identifiable {
    case slidingWindow

    var id: String { rawValue }

    var label: String {
        switch self {
        case .slidingWindow:
            "Sliding window (drop oldest)"
        }
    }
}

/// Per-assistant-turn decode metrics, surfaced in the UI (dev-tool readout).
/// Populated from the terminal `.completed(usage:)` event when the backend
/// reports real token counts (the native MLX runtime does); the completion
/// count falls back to a char-based estimate for backends that omit usage.
struct MessageMetrics: Equatable {
    var promptTokens: Int
    var completionTokens: Int
    var tokensPerSecond: Double
    var timeToFirstToken: Double
    var elapsed: Double
    /// True when completionTokens came from the char-based estimate rather than
    /// the backend's real usage counts (so the UI can mark it as approximate).
    var estimated: Bool
}

/// Two SAFE built-in demo tools, described as OpenAI-style tool dicts, used to
/// exercise the prompt-based tool-calling path in the app. No tool is executed
/// here — the app renders the model's emitted `{"tool_call": …}` as a card.
enum MotifChatDemoTools {
    static var tools: [Any] {
        [
        [
            "type": "function",
            "function": [
                "name": "get_current_time",
                "description": "Get the current date and time.",
                "parameters": ["type": "object", "properties": [String: Any]()],
            ] as [String: Any],
        ] as [String: Any],
        [
            "type": "function",
            "function": [
                "name": "calculator",
                "description": "Evaluate a basic arithmetic expression.",
                "parameters": [
                    "type": "object",
                    "properties": [
                        "expression": [
                            "type": "string",
                            "description": "e.g. 37 * 41",
                        ] as [String: Any]
                    ] as [String: Any],
                    "required": ["expression"],
                ] as [String: Any],
            ] as [String: Any],
        ] as [String: Any],
        ]
    }

    static let names: Set<String> = MotifToolCalling.toolNames(from: tools) ?? []

    /// Executes one of the two SAFE builtin demo tools. Errors are returned as
    /// strings (never thrown) so they feed back to the model as a tool result
    /// rather than crashing the turn. No eval/exec anywhere — the calculator is
    /// a closed-form recursive-descent parser over numbers and + - * / ( ).
    static func execute(_ call: MotifToolCalling.ParsedToolCall) -> String {
        switch call.name {
        case "get_current_time":
            let fmt = DateFormatter()
            fmt.dateFormat = "yyyy-MM-dd HH:mm:ss zzz"
            return fmt.string(from: Date())
        case "calculator":
            guard let expr = call.arguments["expression"]?.anyValue as? String else {
                return "error: missing 'expression' argument"
            }
            guard let value = SafeArithmetic.evaluate(expr) else {
                return "error: invalid arithmetic expression"
            }
            // Render integers without a trailing .0 (37 * 41 -> 1517).
            return value == value.rounded() && abs(value) < 1e15
                ? String(Int64(value)) : String(value)
        default:
            return "error: unknown tool '\(call.name)'"
        }
    }
}

/// Minimal arithmetic evaluator for the calculator demo tool: numbers,
/// + - * / and parentheses with unary minus. A hand-rolled recursive-descent
/// parser (the Swift mirror of Python's AST-whitelisted `safe_arithmetic`) —
/// deliberately NOT NSExpression, which can raise ObjC exceptions and evaluate
/// more than arithmetic.
enum SafeArithmetic {
    static func evaluate(_ expression: String) -> Double? {
        var parser = Parser(Array(expression.unicodeScalars))
        guard let value = parser.parseExpression(), parser.atEnd, value.isFinite else {
            return nil
        }
        return value
    }

    private struct Parser {
        let chars: [Unicode.Scalar]
        var pos = 0
        init(_ chars: [Unicode.Scalar]) { self.chars = chars }

        var atEnd: Bool {
            var p = pos
            while p < chars.count, chars[p] == " " { p += 1 }
            return p == chars.count
        }

        mutating func skipSpaces() { while pos < chars.count, chars[pos] == " " { pos += 1 } }

        mutating func parseExpression() -> Double? {
            guard var lhs = parseTerm() else { return nil }
            while true {
                skipSpaces()
                guard pos < chars.count, chars[pos] == "+" || chars[pos] == "-" else { return lhs }
                let op = chars[pos]; pos += 1
                guard let rhs = parseTerm() else { return nil }
                lhs = op == "+" ? lhs + rhs : lhs - rhs
            }
        }

        mutating func parseTerm() -> Double? {
            guard var lhs = parseFactor() else { return nil }
            while true {
                skipSpaces()
                guard pos < chars.count, chars[pos] == "*" || chars[pos] == "/" else { return lhs }
                let op = chars[pos]; pos += 1
                guard let rhs = parseFactor() else { return nil }
                lhs = op == "*" ? lhs * rhs : lhs / rhs
            }
        }

        mutating func parseFactor() -> Double? {
            skipSpaces()
            guard pos < chars.count else { return nil }
            if chars[pos] == "-" { pos += 1; return parseFactor().map { -$0 } }
            if chars[pos] == "(" {
                pos += 1
                guard let inner = parseExpression() else { return nil }
                skipSpaces()
                guard pos < chars.count, chars[pos] == ")" else { return nil }
                pos += 1
                return inner
            }
            var digits = ""
            while pos < chars.count,
                  chars[pos] == "." || (chars[pos].value >= 48 && chars[pos].value <= 57) {
                digits.unicodeScalars.append(chars[pos]); pos += 1
            }
            return digits.isEmpty ? nil : Double(digits)
        }
    }
}

/// Small value-type coordinator that makes generation ownership explicit. A
/// completion may clear shared UI state only when it still owns the active ID;
/// this keeps a late callback from an older task from finishing a newer one.
struct ChatGenerationLifecycle: Equatable {
    private(set) var activeID: UUID?

    @discardableResult
    mutating func begin(id: UUID = UUID()) -> UUID {
        activeID = id
        return id
    }

    func owns(_ id: UUID) -> Bool {
        activeID == id
    }

    @discardableResult
    mutating func finish(id: UUID) -> Bool {
        guard owns(id) else { return false }
        activeID = nil
        return true
    }
}

/// Typed terminal result for the most recently settled generation. UI
/// announcements use this instead of inferring an outcome from display copy
/// such as `runtimeStatus`, which may legitimately return to Idle after the
/// user switches conversations while cancellation is settling.
enum ChatGenerationOutcome: Equatable, Sendable {
    case succeeded
    case cancelled
    case failed
}

@MainActor
final class ChatStore: ObservableObject {
    @Published var backendMode: MotifChatBackendMode = ChatStore.storedBackendMode() {
        didSet { ChatStore.defaults.set(backendMode.rawValue, forKey: DefaultsKey.backendMode.rawValue) }
    }
    @Published var endpoint = ChatStore.storedString(.endpoint, defaultValue: "http://127.0.0.1:8080/v1") {
        didSet { ChatStore.defaults.set(endpoint, forKey: DefaultsKey.endpoint.rawValue) }
    }
    @Published var model = ChatStore.storedString(.model, defaultValue: "motif") {
        didSet { ChatStore.defaults.set(model, forKey: DefaultsKey.model.rawValue) }
    }
    @Published var nativeModelDirectory = ChatStore.storedNativeModelDirectory() {
        didSet { ChatStore.defaults.set(nativeModelDirectory, forKey: DefaultsKey.nativeModelDirectory.rawValue) }
    }
    @Published var thinkMode: MotifThinkMode = ChatStore.storedThinkMode() {
        didSet { ChatStore.defaults.set(thinkMode.rawValue, forKey: DefaultsKey.thinkMode.rawValue) }
    }
    @Published var maxTokens = ChatStore.storedInt(.maxTokens, defaultValue: 4096) {
        didSet { ChatStore.defaults.set(maxTokens, forKey: DefaultsKey.maxTokens.rawValue) }
    }
    @Published var temperature = ChatStore.storedDouble(.temperature, defaultValue: 0.6) {
        didSet { ChatStore.defaults.set(temperature, forKey: DefaultsKey.temperature.rawValue) }
    }
    @Published var contextTokenBudget = ChatStore.storedInt(.contextTokenBudget, defaultValue: 12000) {
        didSet { ChatStore.defaults.set(contextTokenBudget, forKey: DefaultsKey.contextTokenBudget.rawValue) }
    }
    @Published var compactionMode: MotifCompactionMode = .slidingWindow
    @Published var prompt = ""
    @Published var messages: [MotifChatMessage] = [
        .system(ChatStore.defaultSystemPrompt)
    ] {
        didSet { syncActiveConversation() }
    }
    @Published var isGenerating = false
    @Published private(set) var lastGenerationOutcome: ChatGenerationOutcome?
    @Published var lastError: String?
    @Published var runtimeStatus = "Idle"
    /// Live captured reasoning (`<think>` content) keyed by the assistant
    /// message it belongs to, so it streams ABOVE that message's bubble.
    /// Session-only, like `metrics`/`toolCalls` (not persisted across relaunch).
    @Published var reasoningByMessage: [UUID: String] = [:]
    /// Bumps on every reasoning token so the transcript can follow the stream
    /// to the bottom before any answer text arrives — answer text drives
    /// `liveTokenEstimate`, but reasoning arrives first and separately.
    @Published private(set) var reasoningCharCount = 0
    /// Non-error note surfaced when the context guard trims earlier messages.
    @Published var contextNotice: String?
    @Published var conversations: [MotifConversation] = []
    @Published var activeConversationID: UUID?

    // Dev-tool surfaces --------------------------------------------------------
    /// Per-assistant-message decode metrics (tok/s, token counts, TTFT).
    @Published var metrics: [UUID: MessageMetrics] = [:]
    /// Parsed tool call per assistant message id, when the turn emitted one.
    @Published var toolCalls: [UUID: MotifToolCalling.ParsedToolCall] = [:]
    /// Assistant messages truncated at the output-token limit (finish reason
    /// `.length`). The UI offers a Continue affordance for these; cleared once
    /// a continued turn reaches a natural stop. Session-only.
    @Published var truncatedMessages: Set<UUID> = []
    /// Tool-execution rounds consumed by the CURRENT user turn (reset on every
    /// send/regenerate). Bounds the execute→continue loop like Python's
    /// run_tool_loop max_rounds so a looping model can't spin forever.
    private var toolRoundsThisTurn = 0
    private static let maxToolRounds = 3
    /// Automatic continuations consumed by the CURRENT user turn (reset on every
    /// send/regenerate). Bounds the auto-resume-on-truncation loop like the
    /// Claude Code harness's `MAX_OUTPUT_TOKENS_RECOVERY_LIMIT` so a runaway
    /// generation can't resume forever.
    private var continuationsThisTurn = 0
    private static let maxContinuations = 3
    /// When enabled, the demo tools preamble is injected for each turn.
    @Published var toolsEnabled: Bool = ChatStore.storedBool(.toolsEnabled, defaultValue: false) {
        didSet { ChatStore.defaults.set(toolsEnabled, forKey: DefaultsKey.toolsEnabled.rawValue) }
    }
    /// Recently-selected native checkpoint directories (most-recent first),
    /// persisted so a folder picked via the header stays in the quick list even
    /// when it lives outside ~/.models. Seeded from ~/.models on first launch.
    @Published var recentNativeModelDirectories: [String] = ChatStore.storedRecentNativeModelDirectories() {
        didSet {
            ChatStore.defaults.set(recentNativeModelDirectories, forKey: DefaultsKey.recentNativeModelDirectories.rawValue)
        }
    }
    /// Live decode readout while a turn streams (updated as text arrives).
    @Published var liveTokensPerSecond: Double = 0
    @Published var liveTokenEstimate: Int = 0

    private struct GenerationConfiguration: Sendable {
        let backendMode: MotifChatBackendMode
        let endpoint: String
        let nativeModelDirectory: String
        let parameters: MotifGenerationParameters
        let contextTokenBudget: Int
        let toolsEnabled: Bool

        var backendSignature: String {
            switch backendMode {
            case .openAICompatible:
                "openai:\(endpoint):\(parameters.model)"
            case .nativeMLX:
                "native:\(nativeModelDirectory)"
            }
        }
    }

    private struct RegenerationBackup {
        let conversationID: UUID
        let insertionIndex: Int
        let messages: [MotifChatMessage]
        let reasoning: [UUID: String]
        let metrics: [UUID: MessageMetrics]
        let toolCalls: [UUID: MotifToolCalling.ParsedToolCall]
        let truncatedMessageIDs: Set<UUID>
        let reasoningCharCount: Int
        let toolRounds: Int
        let continuations: Int
    }

    private var generationLifecycle = ChatGenerationLifecycle()
    private var generationTask: (id: UUID, task: Task<Void, Never>)?
    private var pendingRegenerationBackup: RegenerationBackup?
    // Streaming-metrics scratch (reset per turn).
    private var genFirstTokenAt: Date?
    private var genCharCount: Int = 0
    /// Backend reused across turns so the native runtime (and its KV-cache) is
    /// not rebuilt every message — that reuse is what lets the runtime skip
    /// re-prefilling the shared conversation prefix. Keyed on the backend config
    /// signature; rebuilt when the config changes or after a backend error.
    private var cachedBackend: (any MotifChatBackend)?
    private var cachedBackendSignature: String?
    private let backendOverride: (any MotifChatBackend)?
    private let persistsConversations: Bool
    /// Suppresses conversation auto-save while we are loading messages into the
    /// view (selecting/restoring), so a load doesn't bump `updatedAt` or reorder.
    private var isLoadingConversation = false
    private static let defaults = UserDefaults.standard
    private static let defaultSystemPrompt = "You are Motif accessed through the configured local runtime. Be concise and helpful."

    var nativeMLXCompiledIn: Bool {
        #if MOTIFKIT_ENABLE_MLX
        true
        #else
        false
        #endif
    }

    /// Estimated token count of the full current transcript (context meter).
    var currentContextTokens: Int {
        messages.reduce(0) { $0 + motifEstimatedTokenCount($1.content) }
    }

    /// Fraction of the context budget currently used (0…1+).
    var contextUsageFraction: Double {
        guard contextTokenBudget > 0 else { return 0 }
        return Double(currentContextTokens) / Double(contextTokenBudget)
    }

    /// Quick-pick checkpoint directories for the header's native-model menu:
    /// the current selection first, then the persisted recently-used list, then
    /// any ~/.models folders that contain a config.json (deduped, order-stable).
    var discoveredModelDirectories: [String] {
        var seen = Set<String>()
        var dirs: [String] = []
        func add(_ path: String) {
            let trimmed = path.trimmingCharacters(in: .whitespacesAndNewlines)
            guard !trimmed.isEmpty, seen.insert(trimmed).inserted else { return }
            dirs.append(trimmed)
        }

        add(nativeModelDirectory)
        for recent in recentNativeModelDirectories { add(recent) }
        for path in Self.modelsDirectoryCheckpoints() { add(path) }
        return dirs
    }

    /// Folders under ~/.models that contain a config.json (a converted checkpoint).
    private static func modelsDirectoryCheckpoints() -> [String] {
        let fm = FileManager.default
        let root = fm.homeDirectoryForCurrentUser.appendingPathComponent(".models")
        guard let entries = try? fm.contentsOfDirectory(
            at: root, includingPropertiesForKeys: [.isDirectoryKey]
        ) else { return [] }
        return entries
            .sorted { $0.lastPathComponent < $1.lastPathComponent }
            .filter { fm.fileExists(atPath: $0.appendingPathComponent("config.json").path) }
            .map(\.path)
    }

    /// Records a directory at the head of the recently-used list (deduped, capped).
    private func rememberNativeModelDirectory(_ path: String) {
        let trimmed = path.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return }
        var updated = recentNativeModelDirectories.filter { $0 != trimmed }
        updated.insert(trimmed, at: 0)
        if updated.count > 8 { updated = Array(updated.prefix(8)) }
        recentNativeModelDirectories = updated
    }

    /// Short label for the active backend, shown in the header selector.
    var backendDisplayName: String {
        switch backendMode {
        case .nativeMLX:
            let name = (nativeModelDirectory as NSString).lastPathComponent
            return name.isEmpty ? "native MLX" : name
        case .openAICompatible:
            return model
        }
    }

    init(
        backendOverride: (any MotifChatBackend)? = nil,
        loadsPersistedConversations: Bool = true
    ) {
        self.backendOverride = backendOverride
        self.persistsConversations = loadsPersistedConversations
        conversations = loadsPersistedConversations ? Self.loadConversations() : []
        // Open the most recently updated conversation if one exists, otherwise
        // start a fresh chat so the active conversation is always tracked.
        if let mostRecent = conversations.max(by: { $0.updatedAt < $1.updatedAt }) {
            loadConversation(mostRecent)
        } else {
            startFreshConversation()
        }
    }

    func newChat() {
        cancel()
        prompt = ""
        reasoningByMessage.removeAll()
        truncatedMessages.removeAll()
        reasoningCharCount = 0
        contextNotice = nil
        lastError = nil
        if !isGenerating {
            runtimeStatus = "Idle"
        }
        startFreshConversation()
    }

    /// Creates a brand-new conversation, registers it, and makes it active.
    private func startFreshConversation() {
        let conversation = MotifConversation(
            title: MotifConversation.untitledTitle,
            messages: [.system(Self.defaultSystemPrompt)]
        )
        conversations.insert(conversation, at: 0)
        activeConversationID = conversation.id
        isLoadingConversation = true
        messages = conversation.messages
        isLoadingConversation = false
        saveConversations()
    }

    func selectConversation(_ id: UUID) {
        guard id != activeConversationID,
              let conversation = conversations.first(where: { $0.id == id }) else { return }
        cancel()
        prompt = ""
        reasoningByMessage.removeAll()
        truncatedMessages.removeAll()
        reasoningCharCount = 0
        contextNotice = nil
        lastError = nil
        if !isGenerating {
            runtimeStatus = "Idle"
        }
        loadConversation(conversation)
    }

    func deleteConversation(_ id: UUID) {
        // A stream owns message and security-scope state for its conversation;
        // request cancellation before removing that conversation from storage.
        if activeConversationID == id {
            cancel()
        }
        conversations.removeAll { $0.id == id }
        if activeConversationID == id {
            if let next = conversations.max(by: { $0.updatedAt < $1.updatedAt }) {
                loadConversation(next)
            } else {
                startFreshConversation()
            }
        }
        saveConversations()
    }

    private func loadConversation(_ conversation: MotifConversation) {
        activeConversationID = conversation.id
        isLoadingConversation = true
        messages = conversation.messages
        isLoadingConversation = false
    }

    /// Mirrors the live `messages` back into the active conversation and persists.
    /// Called from `messages.didSet`; skipped while loading a conversation.
    private func syncActiveConversation() {
        guard !isLoadingConversation, let id = activeConversationID,
              let index = conversations.firstIndex(where: { $0.id == id }) else { return }
        conversations[index].messages = messages
        conversations[index].title = MotifConversation.deriveTitle(from: messages)
        conversations[index].updatedAt = Date()
        saveConversations()
    }

    private func saveConversations() {
        guard persistsConversations else { return }
        guard let data = try? JSONEncoder().encode(conversations) else { return }
        ChatStore.defaults.set(data, forKey: DefaultsKey.conversations.rawValue)
    }

    private static func loadConversations() -> [MotifConversation] {
        guard let data = defaults.data(forKey: DefaultsKey.conversations.rawValue),
              let decoded = try? JSONDecoder().decode([MotifConversation].self, from: data) else {
            return []
        }
        return decoded
    }

    /// Builds the request transcript, applying the context-budget guard so long
    /// conversations don't overflow the model window. Sets `contextNotice` when
    /// earlier messages were trimmed (a non-error status note).
    func messagesWithinBudget() -> [MotifChatMessage] {
        messagesWithinBudget(tokenBudget: contextTokenBudget)
    }

    private func messagesWithinBudget(tokenBudget: Int) -> [MotifChatMessage] {
        switch compactionMode {
        case .slidingWindow:
            let result = motifTrimMessagesToBudget(messages, budgetTokens: tokenBudget)
            if result.dropped > 0 {
                contextNotice = "Trimmed \(result.dropped) earlier message\(result.dropped == 1 ? "" : "s") to fit context"
            } else {
                contextNotice = nil
            }
            return result.kept
        }
    }

    func cancel() {
        guard let generationTask,
              generationLifecycle.owns(generationTask.id) else { return }
        generationTask.task.cancel()
        runtimeStatus = "Stopping…"
    }

    func send() {
        let trimmed = prompt.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty, !isGenerating else { return }

        reasoningCharCount = 0
        prompt = ""
        toolRoundsThisTurn = 0
        continuationsThisTurn = 0
        messages.append(.user(trimmed))
        _ = startGeneration(configuration: generationConfiguration())
    }

    /// Drops the latest response bundle and re-runs its user prompt. The complete
    /// prior response and its session-only presentation state are restored if
    /// setup or streaming fails, or if the replacement is cancelled.
    func regenerateLast() {
        guard !isGenerating,
              let conversationID = activeConversationID,
              let lastID = messages.last?.id,
              let responseRange = Self.responseRange(containing: lastID, in: messages),
              responseRange.upperBound == messages.endIndex else { return }

        let removedMessages = Array(messages[responseRange])
        let removedIDs = Set(removedMessages.map(\.id))
        let removedReasoning = reasoningByMessage.filter { removedIDs.contains($0.key) }
        let removedMetrics = metrics.filter { removedIDs.contains($0.key) }
        let removedToolCalls = toolCalls.filter { removedIDs.contains($0.key) }
        let removedTruncations = truncatedMessages.intersection(removedIDs)
        let previousReasoningCharCount = reasoningCharCount
        let previousToolRounds = toolRoundsThisTurn
        let previousContinuations = continuationsThisTurn
        let configuration = generationConfiguration()

        pendingRegenerationBackup = RegenerationBackup(
            conversationID: conversationID,
            insertionIndex: responseRange.lowerBound,
            messages: removedMessages,
            reasoning: removedReasoning,
            metrics: removedMetrics,
            toolCalls: removedToolCalls,
            truncatedMessageIDs: removedTruncations,
            reasoningCharCount: previousReasoningCharCount,
            toolRounds: previousToolRounds,
            continuations: previousContinuations
        )

        removeTransientState(for: removedIDs)
        messages.removeSubrange(responseRange)
        reasoningCharCount = 0
        toolRoundsThisTurn = 0
        continuationsThisTurn = 0
        guard !startGeneration(configuration: configuration) else { return }
        restorePendingRegenerationBackup()
    }

    /// Returns the complete assistant/tool response bundle for the user turn
    /// containing `id`. The originating user prompt is deliberately excluded.
    /// This pure helper is internal so response-boundary behavior is testable.
    nonisolated static func responseRange(
        containing id: UUID,
        in messages: [MotifChatMessage]
    ) -> Range<Int>? {
        guard let messageIndex = messages.firstIndex(where: { $0.id == id }),
              messages[messageIndex].role == .assistant || messages[messageIndex].role == .tool,
              let userIndex = messages[...messageIndex].lastIndex(where: { $0.role == .user }) else {
            return nil
        }

        let responseStart = userIndex + 1
        let responseEnd = messages.indices.first(where: {
            $0 > responseStart && (messages[$0].role == .user || messages[$0].role == .system)
        }) ?? messages.endIndex
        guard responseStart < responseEnd, responseStart..<responseEnd ~= messageIndex else {
            return nil
        }
        return responseStart..<responseEnd
    }

    /// Removes the whole response for a turn: tool call, tool result, follow-up
    /// rounds, and final assistant answer. The user prompt remains available for
    /// editing or regeneration.
    func deleteResponse(containing id: UUID) {
        guard !isGenerating,
              let responseRange = Self.responseRange(containing: id, in: messages) else { return }
        let removedIDs = Set(messages[responseRange].map(\.id))
        removeTransientState(for: removedIDs)
        messages.removeSubrange(responseRange)
    }

    /// Compatibility alias for existing message-row callers. Deletion is now
    /// response-scoped rather than single-message-scoped to prevent orphaned
    /// tool results or final answers.
    func deleteMessage(_ id: UUID) {
        deleteResponse(containing: id)
    }

    private func removeTransientState(for messageIDs: Set<UUID>) {
        truncatedMessages.subtract(messageIDs)
        for id in messageIDs {
            reasoningByMessage.removeValue(forKey: id)
            metrics.removeValue(forKey: id)
            toolCalls.removeValue(forKey: id)
        }
    }

    /// Restores the response replaced by an in-flight regeneration. The backup
    /// survives every tool round and is resolved only when the full regenerated
    /// turn succeeds, fails, or is cancelled.
    @discardableResult
    private func restorePendingRegenerationBackup() -> Bool {
        guard let backup = pendingRegenerationBackup else { return false }
        pendingRegenerationBackup = nil

        if activeConversationID == backup.conversationID {
            let insertionIndex = min(backup.insertionIndex, messages.endIndex)
            let generatedIDs = Set(messages[insertionIndex...].map(\.id))
            removeTransientState(for: generatedIDs)
            messages.replaceSubrange(insertionIndex..., with: backup.messages)
            reasoningCharCount = backup.reasoningCharCount
            toolRoundsThisTurn = backup.toolRounds
            continuationsThisTurn = backup.continuations
            reasoningByMessage.merge(backup.reasoning) { _, restored in restored }
            metrics.merge(backup.metrics) { _, restored in restored }
            toolCalls.merge(backup.toolCalls) { _, restored in restored }
            truncatedMessages.formUnion(backup.truncatedMessageIDs)
        } else if let index = conversations.firstIndex(where: { $0.id == backup.conversationID }) {
            var conversation = conversations[index]
            let insertionIndex = min(backup.insertionIndex, conversation.messages.endIndex)
            let generatedIDs = Set(conversation.messages[insertionIndex...].map(\.id))
            removeTransientState(for: generatedIDs)
            conversation.messages.replaceSubrange(insertionIndex..., with: backup.messages)
            conversation.title = MotifConversation.deriveTitle(from: conversation.messages)
            conversation.updatedAt = Date()
            conversations[index] = conversation
            reasoningByMessage.merge(backup.reasoning) { _, restored in restored }
            metrics.merge(backup.metrics) { _, restored in restored }
            toolCalls.merge(backup.toolCalls) { _, restored in restored }
            truncatedMessages.formUnion(backup.truncatedMessageIDs)
            saveConversations()
        } else {
            // The conversation itself was intentionally deleted while stopping.
            return false
        }
        return true
    }

    private func generationConfiguration() -> GenerationConfiguration {
        GenerationConfiguration(
            backendMode: backendMode,
            endpoint: endpoint,
            nativeModelDirectory: nativeModelDirectory,
            parameters: MotifGenerationParameters(
                model: model,
                maxTokens: maxTokens,
                temperature: temperature,
                thinkMode: thinkMode
            ),
            contextTokenBudget: contextTokenBudget,
            toolsEnabled: toolsEnabled
        )
    }

    /// Shared streaming machinery for `send()`/`regenerateLast()`. Assumes the
    /// latest user turn is already the last message; builds the budgeted request
    /// transcript, appends a streaming assistant placeholder, and drives the
    /// backend stream into it.
    /// Request-only instruction (never persisted) that continues a response cut
    /// off at the output-token limit. Adapted from the Claude Code harness's
    /// max-tokens recovery message.
    private static let resumeInstruction =
        "Your previous response was cut off at the output-token limit. "
        + "Continue exactly where you left off — no apology, no recap, no "
        + "repetition of what you already wrote. Pick up mid-sentence if that "
        + "is where the cut happened."

    /// Drives one snapshotted generation round. When `continuationOf` is set,
    /// the run resumes a truncated assistant turn in the existing bubble while
    /// retaining the original turn's backend, model, parameters, and tool mode.
    @discardableResult
    private func startGeneration(
        configuration: GenerationConfiguration,
        continuationOf continuationID: UUID? = nil
    ) -> Bool {
        guard generationLifecycle.activeID == nil else { return false }
        let generationID = UUID()
        let backend: any MotifChatBackend
        do {
            backend = try makeBackend(configuration: configuration)
        } catch {
            lastError = error.localizedDescription
            runtimeStatus = "Error"
            return false
        }

        lastGenerationOutcome = nil
        lastError = nil
        runtimeStatus = configuration.backendMode == .nativeMLX
            ? "Loading native checkpoint…" : "Connecting to endpoint…"
        // Build the request transcript before adding the streaming placeholder,
        // applying the context-budget guard so long chats don't overflow. When
        // tools are enabled, inject the demo-tools preamble into the leading
        // system message for this turn only (the stored transcript is untouched).
        var requestMessages = messagesWithinBudget(tokenBudget: configuration.contextTokenBudget)
        // Parity with Python run_tool_loop: each assistant turn whose tool call
        // actually EXECUTED (marked by the tool-result turn right after it —
        // a persisted marker, unlike the session-only `toolCalls` dict) is
        // sent to the model as the single canonical {"tool_call": ...} JSON,
        // not the raw output (which may repeat the JSON or carry think
        // blocks). Display keeps the raw content.
        for index in requestMessages.indices where requestMessages[index].role == .assistant {
            guard index + 1 < requestMessages.count,
                  requestMessages[index + 1].role == .tool else { continue }
            let call = toolCalls[requestMessages[index].id]
                ?? MotifToolCalling.parseToolCall(
                    text: requestMessages[index].content,
                    toolNames: MotifChatDemoTools.names
                )
            if let call {
                requestMessages[index].content = MotifToolCalling.canonicalToolCallJSON(call)
            }
        }
        if configuration.toolsEnabled {
            let preamble = MotifToolCalling.buildToolsPreamble(MotifChatDemoTools.tools)
            if let first = requestMessages.first, first.role == .system {
                requestMessages[0].content += "\n\n" + preamble
            } else {
                requestMessages.insert(.system(preamble), at: 0)
            }
        }
        let assistantID: UUID
        if let continuationID {
            // Resume: keep streaming into the truncated message; add the
            // request-only resume instruction as the final turn so the model
            // continues after it. Not persisted, so future turns stay clean.
            requestMessages.append(.user(Self.resumeInstruction))
            assistantID = continuationID
            truncatedMessages.remove(continuationID)
        } else {
            messages.append(.assistant(""))
            assistantID = messages[messages.count - 1].id
        }
        let conversationID = activeConversationID
        generationLifecycle.begin(id: generationID)
        isGenerating = true

        // Reset per-turn decode-metrics scratch.
        let started = Date()
        genFirstTokenAt = nil
        genCharCount = 0
        liveTokensPerSecond = 0
        liveTokenEstimate = 0
        let promptEstimate = requestMessages.reduce(0) { $0 + motifEstimatedTokenCount($1.content) }

        let task = Task { [weak self] in
            guard let self else { return }
            var outcome = ChatGenerationOutcome.succeeded
            do {
                for try await event in backend.streamResponse(
                    messages: Array(requestMessages),
                    parameters: configuration.parameters
                ) {
                    try Task.checkCancellation()
                    await MainActor.run {
                        guard self.generationLifecycle.owns(generationID),
                              self.activeConversationID == conversationID else { return }
                        if self.runtimeStatus != "Generating…" {
                            self.runtimeStatus = "Generating…"
                        }
                        switch event {
                        case .text(let text):
                            if self.genFirstTokenAt == nil { self.genFirstTokenAt = Date() }
                            self.genCharCount += text.count
                            self.liveTokenEstimate = (self.genCharCount + 3) / 4
                            if let first = self.genFirstTokenAt {
                                let dt = Date().timeIntervalSince(first)
                                if dt > 0 { self.liveTokensPerSecond = Double(self.liveTokenEstimate) / dt }
                            }
                            self.append(text, toAssistantMessage: assistantID)
                        case .reasoning(let reasoning):
                            self.reasoningByMessage[assistantID, default: ""] += reasoning
                            self.reasoningCharCount += reasoning.count
                        case .completed(let usage, let finishReason):
                            let now = Date()
                            let content = self.messages.first(where: { $0.id == assistantID })?.content ?? ""
                            let completion = usage?.completionTokens ?? motifEstimatedTokenCount(content)
                            let decodeStart = self.genFirstTokenAt ?? started
                            let decodeElapsed = max(now.timeIntervalSince(decodeStart), 1e-6)
                            self.metrics[assistantID] = MessageMetrics(
                                promptTokens: usage?.promptTokens ?? promptEstimate,
                                completionTokens: completion,
                                tokensPerSecond: Double(completion) / decodeElapsed,
                                timeToFirstToken: (self.genFirstTokenAt ?? now).timeIntervalSince(started),
                                elapsed: now.timeIntervalSince(started),
                                estimated: usage == nil
                            )
                            // Record truncation so the UI can offer to continue
                            // the turn. A prior truncation is cleared once the
                            // model reaches a natural stop.
                            switch finishReason {
                            case .length:
                                self.truncatedMessages.insert(assistantID)
                            case .stop, .cancelled:
                                self.truncatedMessages.remove(assistantID)
                            case .unknown:
                                // Some SSE servers send a meaningful terminal
                                // reason followed by a bare [DONE]. Do not let
                                // that trailing unknown erase `.length`.
                                break
                            }
                        }
                    }
                }
                try Task.checkCancellation()
            } catch is CancellationError {
                outcome = .cancelled
                await MainActor.run {
                    guard self.generationLifecycle.owns(generationID),
                          self.activeConversationID == conversationID else { return }
                    self.runtimeStatus = "Cancelled"
                    self.append("\n\n[Cancelled]", toAssistantMessage: assistantID)
                }
            } catch {
                if Task.isCancelled {
                    outcome = .cancelled
                    await MainActor.run {
                        guard self.generationLifecycle.owns(generationID),
                              self.activeConversationID == conversationID else { return }
                        self.runtimeStatus = "Cancelled"
                        self.append("\n\n[Cancelled]", toAssistantMessage: assistantID)
                    }
                    // Some async backends surface cancellation as their own
                    // transport error rather than `CancellationError`.
                    // Treat it as cancellation and keep the backend reusable.
                } else {
                    outcome = .failed
                    await MainActor.run {
                        guard self.generationLifecycle.owns(generationID) else { return }
                        self.invalidateBackend(matching: configuration.backendSignature)
                        guard self.activeConversationID == conversationID else { return }
                        self.runtimeStatus = "Error"
                        self.lastError = error.localizedDescription
                        self.append("\n\n[Error: \(error.localizedDescription)]", toAssistantMessage: assistantID)
                    }
                }
            }

            await MainActor.run {
                guard self.generationLifecycle.finish(id: generationID) else { return }
                self.lastGenerationOutcome = outcome
                self.isGenerating = false
                if self.generationTask?.id == generationID {
                    self.generationTask = nil
                }
                self.liveTokensPerSecond = 0
                self.liveTokenEstimate = 0
                if outcome != .succeeded, self.restorePendingRegenerationBackup() {
                    if self.activeConversationID != conversationID {
                        self.runtimeStatus = "Idle"
                    }
                    return
                }
                guard self.activeConversationID == conversationID else {
                    if self.runtimeStatus == "Stopping…" {
                        self.runtimeStatus = "Idle"
                    }
                    return
                }
                // Resume a response cut off at the output-token limit before
                // interpreting it as a tool call. The same turn configuration
                // and regeneration backup survive every bounded continuation.
                if outcome == .succeeded,
                   self.truncatedMessages.contains(assistantID),
                   self.continuationsThisTurn < Self.maxContinuations {
                    self.continuationsThisTurn += 1
                    self.runtimeStatus = "Continuing (\(self.continuationsThisTurn)/\(Self.maxContinuations))…"
                    if !self.startGeneration(
                        configuration: configuration,
                        continuationOf: assistantID
                    ) {
                        self.restorePendingRegenerationBackup()
                    }
                    return
                }
                // If this turn emitted a tool call, capture it so the transcript
                // renders it as a card instead of raw JSON — then EXECUTE the
                // builtin, append the result as a tool turn, and continue
                // generating so the model can answer with the result (bounded;
                // mirrors Python's run_tool_loop).
                let content = self.messages.first(where: { $0.id == assistantID })?.content ?? ""
                if let call = MotifToolCalling.parseToolCall(text: content, toolNames: MotifChatDemoTools.names) {
                    self.toolCalls[assistantID] = call
                    let canExecute = configuration.toolsEnabled && outcome == .succeeded
                    if canExecute, self.toolRoundsThisTurn < Self.maxToolRounds {
                        self.toolRoundsThisTurn += 1
                        let result = MotifChatDemoTools.execute(call)
                        // Bare result as a real `tool` turn (rendered <|tool|>
                        // by the template) — parity with Python run_tool_loop.
                        // The "Tool result (name):" framing is display-only.
                        self.messages.append(.tool(result, name: call.name))
                        self.runtimeStatus = "Running tool round \(self.toolRoundsThisTurn)…"
                        if !self.startGeneration(configuration: configuration) {
                            self.restorePendingRegenerationBackup()
                        }
                        return
                    } else if canExecute {
                        // Budget exhausted while the model keeps calling tools.
                        // Python's run_tool_loop returns final text with
                        // stopped_reason="max_rounds"; here we surface a visible
                        // note so the transcript never dead-ends on a bare
                        // tool-call card with no resolution (the original bug,
                        // deferred to the round cap).
                        self.messages.append(.assistant(
                            "_Stopped after \(Self.maxToolRounds) tool rounds without a final answer._"))
                    }
                }
                if outcome == .succeeded {
                    self.pendingRegenerationBackup = nil
                    self.runtimeStatus = "Idle"
                }
            }
        }
        generationTask = (generationID, task)
        return true
    }

    func selectNativeModelDirectory(_ url: URL) {
        nativeModelDirectory = url.path
        rememberNativeModelDirectory(url.path)
        persistSecurityScopedBookmark(for: url)
        backendMode = .nativeMLX
        runtimeStatus = "Native checkpoint selected"
        lastError = nil
    }

    /// Selects an already-known checkpoint path (from the header's quick list),
    /// promoting it in the recently-used list. No file picker / bookmark here —
    /// ~/.models entries are readable without a scoped bookmark in dev builds.
    func selectNativeModelDirectoryPath(_ path: String) {
        nativeModelDirectory = path
        rememberNativeModelDirectory(path)
        backendMode = .nativeMLX
        runtimeStatus = "Native checkpoint selected"
        lastError = nil
    }

    /// Persists a security-scoped bookmark for the selected directory so the app
    /// can re-acquire scoped access after relaunch and under App Sandbox. Failures
    /// are non-fatal: the path-based fallback keeps non-sandboxed/dev use working.
    private func persistSecurityScopedBookmark(for url: URL) {
        #if os(macOS)
        let options: URL.BookmarkCreationOptions = [.withSecurityScope]
        #else
        let options: URL.BookmarkCreationOptions = []
        #endif
        do {
            let bookmark = try url.bookmarkData(
                options: options,
                includingResourceValuesForKeys: nil,
                relativeTo: nil
            )
            var bookmarks = Self.storedNativeModelDirectoryBookmarks()
            bookmarks[Self.canonicalModelDirectoryPath(url)] = bookmark.base64EncodedString()
            ChatStore.defaults.set(bookmarks, forKey: DefaultsKey.nativeModelDirectoryBookmarks.rawValue)
        } catch {
            // Drop only this path's invalid entry; bookmarks for other imported
            // model directories remain valid.
            var bookmarks = Self.storedNativeModelDirectoryBookmarks()
            bookmarks.removeValue(forKey: Self.canonicalModelDirectoryPath(url))
            ChatStore.defaults.set(bookmarks, forKey: DefaultsKey.nativeModelDirectoryBookmarks.rawValue)
        }
    }

    func resetRuntimeSettings() {
        // Reset runtime/generation settings but preserve saved conversation
        // history (its own delete affordance handles that).
        for key in DefaultsKey.allCases where key != .conversations {
            ChatStore.defaults.removeObject(forKey: key.rawValue)
        }
        backendMode = .openAICompatible
        endpoint = "http://127.0.0.1:8080/v1"
        model = "motif"
        nativeModelDirectory = ChatStore.defaultNativeModelDirectory
        thinkMode = .captured
        maxTokens = 4096
        temperature = 0.6
        contextTokenBudget = 12000
        toolsEnabled = false
        recentNativeModelDirectories = Self.modelsDirectoryCheckpoints()
        runtimeStatus = "Settings reset"
        lastError = nil
    }

    private func append(_ text: String, toAssistantMessage id: UUID) {
        guard let index = messages.firstIndex(where: { $0.id == id }) else { return }
        messages[index].content += text
    }

    /// Discards the cached backend so the next turn rebuilds it (and resets the
    /// native KV-cache). Call when the backend config changes or after an error.
    private func invalidateBackend() {
        cachedBackend = nil
        cachedBackendSignature = nil
    }

    /// Invalidates only the backend used by the failing generation. A late
    /// failure from an older task must not discard a newer task's backend.
    private func invalidateBackend(matching signature: String) {
        guard cachedBackendSignature == signature else { return }
        invalidateBackend()
    }

    private func makeBackend(
        configuration: GenerationConfiguration
    ) throws -> any MotifChatBackend {
        if let backendOverride { return backendOverride }
        // Reuse the backend across turns while its config is unchanged, so the
        // native runtime's KV-cache survives between messages.
        if let cachedBackend, cachedBackendSignature == configuration.backendSignature {
            return cachedBackend
        }
        invalidateBackend()

        let backend = try buildBackend(configuration: configuration)
        cachedBackend = backend
        cachedBackendSignature = configuration.backendSignature
        return backend
    }

    private func buildBackend(
        configuration: GenerationConfiguration
    ) throws -> any MotifChatBackend {
        switch configuration.backendMode {
        case .openAICompatible:
            guard let baseURL = Self.validatedEndpointURL(configuration.endpoint) else {
                throw MotifChatStoreError.invalidEndpoint
            }
            return OpenAICompatibleMotifBackend(baseURL: baseURL)

        case .nativeMLX:
            #if MOTIFKIT_ENABLE_MLX
            let access = try acquireNativeModelDirectoryAccess(
                modelDirectory: configuration.nativeModelDirectory
            )
            // Loading is lazy and the backend is cached across turns, so the
            // backend—not the first consumer task—must own scoped directory
            // access. The lease releases exactly once when that backend is
            // invalidated and all in-flight producer work has let it go.
            let lease = NativeDirectoryAccessLease(grant: access)
            do {
                return try MotifMLXBackend(
                    modelDirectory: lease.url,
                    directoryAccessLease: lease
                )
            } catch {
                lease.stop()
                throw error
            }
            #else
            throw MotifChatStoreError.nativeMLXNotCompiled
            #endif
        }
    }

    /// Accept only absolute HTTP(S) endpoints with a real authority host.
    /// Kept internal and pure so URL edge cases can be regression-tested.
    nonisolated static func validatedEndpointURL(_ endpoint: String) -> URL? {
        let trimmed = endpoint.trimmingCharacters(in: .whitespacesAndNewlines)
        guard let components = URLComponents(string: trimmed),
              let scheme = components.scheme?.lowercased(),
              scheme == "http" || scheme == "https",
              let host = components.host,
              !host.isEmpty,
              let url = components.url else {
            return nil
        }
        return url
    }

    /// Resolves the native model directory, preferring a security-scoped bookmark so
    /// reads succeed under App Sandbox and after relaunch. Bookmarks are selected by
    /// canonical model path; the returned grant is transferred to the cached native
    /// backend so access spans its lazy load. Falls back to the plain path when no
    /// matching bookmark exists.
    ///
    /// The decision/fallback logic lives in `NativeModelDirectoryResolver` (in
    /// MotifKit) so it is unit-testable headless; this method only wires the real
    /// Foundation bookmark / `UserDefaults` operations into that resolver.
    private func acquireNativeModelDirectoryAccess(
        modelDirectory: String
    ) throws -> NativeDirectoryAccessGrant {
        let selectedURL = try resolvedNativeModelDirectoryFromPath(modelDirectory)
        let selectedBookmark = bookmarkDataForSelectedDirectory(selectedURL)
        let resolver = NativeModelDirectoryResolver(
            loadBookmark: { selectedBookmark },
            resolveBookmark: { bookmark, isStale in
                #if os(macOS)
                let options: URL.BookmarkResolutionOptions = [.withSecurityScope]
                #else
                let options: URL.BookmarkResolutionOptions = []
                #endif
                let resolvedURL = try URL(
                    resolvingBookmarkData: bookmark,
                    options: options,
                    relativeTo: nil,
                    bookmarkDataIsStale: &isStale
                )
                guard Self.bookmarkURL(resolvedURL, matches: selectedURL) else {
                    throw MotifChatStoreError.nativeBookmarkPathMismatch
                }
                return resolvedURL
            },
            startAccess: { $0.startAccessingSecurityScopedResource() },
            stopAccess: { $0.stopAccessingSecurityScopedResource() },
            persistBookmark: { [weak self] url in self?.persistSecurityScopedBookmark(for: url) },
            resolvePath: { selectedURL }
        )
        return try resolver.acquire()
    }

    /// Retrieves only the bookmark associated with `selectedURL`. The legacy
    /// single-bookmark key is accepted once when it resolves to the same path,
    /// then migrated into the path-keyed dictionary. A legacy bookmark for A is
    /// never allowed to shadow a selected path B.
    private func bookmarkDataForSelectedDirectory(_ selectedURL: URL) -> Data? {
        var bookmarks = Self.storedNativeModelDirectoryBookmarks()
        if let bookmark = Self.bookmarkData(for: selectedURL, in: bookmarks) {
            return bookmark
        }

        guard let legacyEncoded = ChatStore.defaults.string(
            forKey: DefaultsKey.nativeModelDirectoryBookmark.rawValue
        ), let legacyBookmark = Data(base64Encoded: legacyEncoded) else {
            return nil
        }

        var isStale = false
        #if os(macOS)
        let options: URL.BookmarkResolutionOptions = [.withSecurityScope]
        #else
        let options: URL.BookmarkResolutionOptions = []
        #endif
        guard let legacyURL = try? URL(
            resolvingBookmarkData: legacyBookmark,
            options: options,
            relativeTo: nil,
            bookmarkDataIsStale: &isStale
        ), Self.bookmarkURL(legacyURL, matches: selectedURL) else {
            return nil
        }

        bookmarks[Self.canonicalModelDirectoryPath(selectedURL)] = legacyEncoded
        ChatStore.defaults.set(bookmarks, forKey: DefaultsKey.nativeModelDirectoryBookmarks.rawValue)
        return legacyBookmark
    }

    /// Stable dictionary key for a selected model directory.
    nonisolated static func canonicalModelDirectoryPath(_ url: URL) -> String {
        url.standardizedFileURL.path
    }

    /// Pure lookup seam used by tests to prove that selecting B cannot reuse
    /// the stored security bookmark for A.
    nonisolated static func bookmarkData(
        for selectedURL: URL,
        in encodedBookmarks: [String: String]
    ) -> Data? {
        guard let encoded = encodedBookmarks[canonicalModelDirectoryPath(selectedURL)] else {
            return nil
        }
        return Data(base64Encoded: encoded)
    }

    nonisolated static func bookmarkURL(_ resolvedURL: URL, matches selectedURL: URL) -> Bool {
        canonicalModelDirectoryPath(resolvedURL) == canonicalModelDirectoryPath(selectedURL)
    }

    private static func storedNativeModelDirectoryBookmarks() -> [String: String] {
        guard let stored = defaults.dictionary(
            forKey: DefaultsKey.nativeModelDirectoryBookmarks.rawValue
        ) else { return [:] }
        return stored.compactMapValues { $0 as? String }
    }

    private func resolvedNativeModelDirectoryFromPath(_ modelDirectory: String) throws -> URL {
        let trimmed = modelDirectory.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { throw NativeModelDirectoryError.emptyPath }

        let expanded: String
        if trimmed == "~" {
            expanded = FileManager.default.homeDirectoryForCurrentUser.path
        } else if trimmed.hasPrefix("~/") {
            expanded = FileManager.default.homeDirectoryForCurrentUser
                .appendingPathComponent(String(trimmed.dropFirst(2)))
                .path
        } else {
            expanded = trimmed
        }

        if expanded.hasPrefix("/") {
            return URL(fileURLWithPath: expanded, isDirectory: true)
        }
        return URL(fileURLWithPath: FileManager.default.currentDirectoryPath)
            .appendingPathComponent(expanded, isDirectory: true)
    }

    private static var defaultNativeModelDirectory: String {
        ProcessInfo.processInfo.environment["MOTIF_MODEL_DIR"] ?? "~/.models/motif-2.6b-mlx-q4"
    }

    private static func storedBackendMode() -> MotifChatBackendMode {
        guard let rawValue = defaults.string(forKey: DefaultsKey.backendMode.rawValue) else {
            return .openAICompatible
        }
        return MotifChatBackendMode(rawValue: rawValue) ?? .openAICompatible
    }

    private static func storedThinkMode() -> MotifThinkMode {
        // Default to `.captured`: reasoning models stream their `<think>` block
        // into a live disclosure above the answer, which is the most useful
        // default for a dev tool inspecting model behaviour.
        guard let rawValue = defaults.string(forKey: DefaultsKey.thinkMode.rawValue) else {
            return .captured
        }
        return MotifThinkMode(rawValue: rawValue) ?? .captured
    }

    private static func storedNativeModelDirectory() -> String {
        storedString(.nativeModelDirectory, defaultValue: defaultNativeModelDirectory)
    }

    /// Loads the persisted recently-used list, seeding it from ~/.models on first
    /// launch so the quick list is populated before any manual selection.
    private static func storedRecentNativeModelDirectories() -> [String] {
        if let stored = defaults.stringArray(forKey: DefaultsKey.recentNativeModelDirectories.rawValue) {
            return stored
        }
        return modelsDirectoryCheckpoints()
    }

    private static func storedString(_ key: DefaultsKey, defaultValue: String) -> String {
        defaults.string(forKey: key.rawValue) ?? defaultValue
    }

    private static func storedInt(_ key: DefaultsKey, defaultValue: Int) -> Int {
        if defaults.object(forKey: key.rawValue) == nil { return defaultValue }
        return defaults.integer(forKey: key.rawValue)
    }

    private static func storedDouble(_ key: DefaultsKey, defaultValue: Double) -> Double {
        if defaults.object(forKey: key.rawValue) == nil { return defaultValue }
        return defaults.double(forKey: key.rawValue)
    }

    private static func storedBool(_ key: DefaultsKey, defaultValue: Bool) -> Bool {
        if defaults.object(forKey: key.rawValue) == nil { return defaultValue }
        return defaults.bool(forKey: key.rawValue)
    }
}

private enum DefaultsKey: String, CaseIterable {
    case backendMode = "motif.chat.backendMode"
    case endpoint = "motif.chat.endpoint"
    case model = "motif.chat.model"
    case nativeModelDirectory = "motif.chat.nativeModelDirectory"
    case nativeModelDirectoryBookmark = "motif.chat.nativeModelDirectoryBookmark"
    case nativeModelDirectoryBookmarks = "motif.chat.nativeModelDirectoryBookmarks"
    case thinkMode = "motif.chat.thinkMode"
    case maxTokens = "motif.chat.maxTokens"
    case temperature = "motif.chat.temperature"
    case contextTokenBudget = "motif.chat.contextTokenBudget"
    case toolsEnabled = "motif.chat.toolsEnabled"
    case recentNativeModelDirectories = "motif.chat.recentNativeModelDirectories"
    case conversations = "motif.chat.conversations"
}

private enum MotifChatStoreError: Error, LocalizedError {
    case invalidEndpoint
    case nativeBookmarkPathMismatch
    case nativeMLXNotCompiled

    var errorDescription: String? {
        switch self {
        case .invalidEndpoint:
            "Endpoint must be an absolute HTTP or HTTPS URL with a host."
        case .nativeBookmarkPathMismatch:
            "The saved folder access does not match the selected native model directory. Re-select the folder."
        case .nativeMLXNotCompiled:
            "Native MLX chat is not compiled into this app build. Rebuild with MOTIFKIT_ENABLE_MLX=1."
        }
    }
}
