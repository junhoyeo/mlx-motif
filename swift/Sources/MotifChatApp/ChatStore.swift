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
    @Published var maxTokens = ChatStore.storedInt(.maxTokens, defaultValue: 512) {
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
    @Published var lastError: String?
    @Published var runtimeStatus = "Idle"
    @Published var capturedReasoning = ""
    /// Non-error note surfaced when the context guard trims earlier messages.
    @Published var contextNotice: String?
    @Published var conversations: [MotifConversation] = []
    @Published var activeConversationID: UUID?

    // Dev-tool surfaces --------------------------------------------------------
    /// Per-assistant-message decode metrics (tok/s, token counts, TTFT).
    @Published var metrics: [UUID: MessageMetrics] = [:]
    /// Parsed tool call per assistant message id, when the turn emitted one.
    @Published var toolCalls: [UUID: MotifToolCalling.ParsedToolCall] = [:]
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

    private var generationTask: Task<Void, Never>?
    // Streaming-metrics scratch (reset per turn).
    private var genFirstTokenAt: Date?
    private var genCharCount: Int = 0
    private var nativeDirectoryAccess: NativeDirectoryAccessGrant?
    /// Backend reused across turns so the native runtime (and its KV-cache) is
    /// not rebuilt every message — that reuse is what lets the runtime skip
    /// re-prefilling the shared conversation prefix. Keyed on the backend config
    /// signature; rebuilt when the config changes or after a backend error.
    private var cachedBackend: (any MotifChatBackend)?
    private var cachedBackendSignature: String?
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

    init() {
        conversations = Self.loadConversations()
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
        capturedReasoning = ""
        contextNotice = nil
        lastError = nil
        runtimeStatus = "Idle"
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
        capturedReasoning = ""
        contextNotice = nil
        lastError = nil
        runtimeStatus = "Idle"
        loadConversation(conversation)
    }

    func deleteConversation(_ id: UUID) {
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
        switch compactionMode {
        case .slidingWindow:
            let result = motifTrimMessagesToBudget(messages, budgetTokens: contextTokenBudget)
            if result.dropped > 0 {
                contextNotice = "Trimmed \(result.dropped) earlier message\(result.dropped == 1 ? "" : "s") to fit context"
            } else {
                contextNotice = nil
            }
            return result.kept
        }
    }

    func cancel() {
        generationTask?.cancel()
        generationTask = nil
        isGenerating = false
        if runtimeStatus != "Idle" {
            runtimeStatus = "Cancelled"
        }
    }

    func send() {
        let trimmed = prompt.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty, !isGenerating else { return }

        capturedReasoning = ""
        prompt = ""
        messages.append(.user(trimmed))
        startGeneration()
    }

    /// Drops the trailing assistant reply (if any) and re-runs the latest user
    /// turn through the same streaming path as `send()`. No-op while generating
    /// or when there is no user message to replay.
    func regenerateLast() {
        guard !isGenerating else { return }
        // Remove a trailing assistant message so the latest turn is the user's.
        if let last = messages.last, last.role == .assistant {
            messages.removeLast()
        }
        guard messages.last?.role == .user else { return }
        capturedReasoning = ""
        startGeneration()
    }

    /// Removes a single message by id. Disabled mid-stream so a delete can't race
    /// the streaming placeholder it is appending into.
    func deleteMessage(_ id: UUID) {
        guard !isGenerating else { return }
        messages.removeAll { $0.id == id }
    }

    /// Shared streaming machinery for `send()`/`regenerateLast()`. Assumes the
    /// latest user turn is already the last message; builds the budgeted request
    /// transcript, appends a streaming assistant placeholder, and drives the
    /// backend stream into it.
    private func startGeneration() {
        let backend: any MotifChatBackend
        do {
            backend = try makeBackend()
        } catch {
            lastError = error.localizedDescription
            runtimeStatus = "Error"
            return
        }

        lastError = nil
        runtimeStatus = backendMode == .nativeMLX ? "Loading native checkpoint…" : "Connecting to endpoint…"
        // Build the request transcript before adding the streaming placeholder,
        // applying the context-budget guard so long chats don't overflow. When
        // tools are enabled, inject the demo-tools preamble into the leading
        // system message for this turn only (the stored transcript is untouched).
        var requestMessages = messagesWithinBudget()
        if toolsEnabled {
            let preamble = MotifToolCalling.buildToolsPreamble(MotifChatDemoTools.tools)
            if let first = requestMessages.first, first.role == .system {
                requestMessages[0].content += "\n\n" + preamble
            } else {
                requestMessages.insert(.system(preamble), at: 0)
            }
        }
        messages.append(.assistant(""))
        let assistantID = messages[messages.count - 1].id
        isGenerating = true

        // Reset per-turn decode-metrics scratch.
        let started = Date()
        genFirstTokenAt = nil
        genCharCount = 0
        liveTokensPerSecond = 0
        liveTokenEstimate = 0
        let promptEstimate = requestMessages.reduce(0) { $0 + motifEstimatedTokenCount($1.content) }

        let parameters = MotifGenerationParameters(
            model: model,
            maxTokens: maxTokens,
            temperature: temperature,
            thinkMode: thinkMode
        )

        generationTask = Task { [weak self] in
            guard let self else { return }
            do {
                for try await event in backend.streamResponse(
                    messages: Array(requestMessages),
                    parameters: parameters
                ) {
                    try Task.checkCancellation()
                    await MainActor.run {
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
                            self.capturedReasoning += reasoning
                        case .completed(let usage):
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
                        }
                    }
                }
            } catch is CancellationError {
                await MainActor.run {
                    self.runtimeStatus = "Cancelled"
                    self.append("\n\n[Cancelled]", toAssistantMessage: assistantID)
                }
            } catch {
                await MainActor.run {
                    self.runtimeStatus = "Error"
                    self.lastError = error.localizedDescription
                    self.append("\n\n[Error: \(error.localizedDescription)]", toAssistantMessage: assistantID)
                    // Drop the (possibly half-loaded) backend so the next turn
                    // rebuilds cleanly and the KV-cache starts fresh.
                    self.invalidateBackend()
                }
            }

            await MainActor.run {
                // Release scoped access only after every file read for this load is
                // done — this block runs on all completion paths (success/error/cancel).
                self.releaseNativeDirectoryAccess()
                self.isGenerating = false
                self.generationTask = nil
                self.liveTokensPerSecond = 0
                self.liveTokenEstimate = 0
                // If this turn emitted a tool call, capture it so the transcript
                // renders it as a card instead of raw JSON.
                let content = self.messages.first(where: { $0.id == assistantID })?.content ?? ""
                if let call = MotifToolCalling.parseToolCall(text: content, toolNames: MotifChatDemoTools.names) {
                    self.toolCalls[assistantID] = call
                }
                if self.lastError == nil, self.runtimeStatus != "Cancelled" {
                    self.runtimeStatus = "Idle"
                }
            }
        }
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
            ChatStore.defaults.set(
                bookmark.base64EncodedString(),
                forKey: DefaultsKey.nativeModelDirectoryBookmark.rawValue
            )
        } catch {
            // Drop any stale bookmark so resolution falls back to the path cleanly.
            ChatStore.defaults.removeObject(forKey: DefaultsKey.nativeModelDirectoryBookmark.rawValue)
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
        thinkMode = .hidden
        maxTokens = 512
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

    /// Config signature: when this is unchanged across turns we can safely reuse
    /// the same backend instance (and its loaded runtime + KV-cache).
    private var backendSignature: String {
        switch backendMode {
        case .openAICompatible:
            return "openai:\(endpoint):\(model)"
        case .nativeMLX:
            return "native:\(nativeModelDirectory)"
        }
    }

    /// Discards the cached backend so the next turn rebuilds it (and resets the
    /// native KV-cache). Call when the backend config changes or after an error.
    private func invalidateBackend() {
        cachedBackend = nil
        cachedBackendSignature = nil
    }

    private func makeBackend() throws -> any MotifChatBackend {
        // Reuse the backend across turns while its config is unchanged, so the
        // native runtime's KV-cache survives between messages.
        if let cachedBackend, cachedBackendSignature == backendSignature {
            return cachedBackend
        }
        invalidateBackend()

        let backend = try buildBackend()
        cachedBackend = backend
        cachedBackendSignature = backendSignature
        return backend
    }

    private func buildBackend() throws -> any MotifChatBackend {
        switch backendMode {
        case .openAICompatible:
            guard let baseURL = URL(string: endpoint) else {
                throw MotifChatStoreError.invalidEndpoint
            }
            return OpenAICompatibleMotifBackend(baseURL: baseURL)

        case .nativeMLX:
            #if MOTIFKIT_ENABLE_MLX
            // The access started here is released in `releaseNativeDirectoryAccess()`
            // once generation finishes, so it spans the actual file reads that the
            // backend performs lazily during streaming — not just backend init.
            let access = try acquireNativeModelDirectoryAccess()
            nativeDirectoryAccess = access
            do {
                return try MotifMLXBackend(modelDirectory: access.url)
            } catch {
                releaseNativeDirectoryAccess()
                throw error
            }
            #else
            throw MotifChatStoreError.nativeMLXNotCompiled
            #endif
        }
    }

    /// Releases any active security-scoped access to the native model directory.
    /// Safe to call when no access is held.
    private func releaseNativeDirectoryAccess() {
        nativeDirectoryAccess?.stop()
        nativeDirectoryAccess = nil
    }

    /// Resolves the native model directory, preferring a security-scoped bookmark so
    /// reads succeed under App Sandbox and after relaunch. Starts scoped access when a
    /// bookmark is available; the returned token's `stop` must be invoked once the
    /// checkpoint load completes. Falls back to the plain path when no bookmark exists.
    ///
    /// The decision/fallback logic lives in `NativeModelDirectoryResolver` (in
    /// MotifKit) so it is unit-testable headless; this method only wires the real
    /// Foundation bookmark / `UserDefaults` operations into that resolver.
    private func acquireNativeModelDirectoryAccess() throws -> NativeDirectoryAccessGrant {
        let resolver = NativeModelDirectoryResolver(
            loadBookmark: {
                guard
                    let encoded = ChatStore.defaults.string(
                        forKey: DefaultsKey.nativeModelDirectoryBookmark.rawValue
                    ),
                    let bookmark = Data(base64Encoded: encoded)
                else {
                    return nil
                }
                return bookmark
            },
            resolveBookmark: { bookmark, isStale in
                #if os(macOS)
                let options: URL.BookmarkResolutionOptions = [.withSecurityScope]
                #else
                let options: URL.BookmarkResolutionOptions = []
                #endif
                return try URL(
                    resolvingBookmarkData: bookmark,
                    options: options,
                    relativeTo: nil,
                    bookmarkDataIsStale: &isStale
                )
            },
            startAccess: { $0.startAccessingSecurityScopedResource() },
            stopAccess: { $0.stopAccessingSecurityScopedResource() },
            persistBookmark: { [weak self] url in self?.persistSecurityScopedBookmark(for: url) },
            resolvePath: { try self.resolvedNativeModelDirectoryFromPath() }
        )
        return try resolver.acquire()
    }

    private func resolvedNativeModelDirectoryFromPath() throws -> URL {
        let trimmed = nativeModelDirectory.trimmingCharacters(in: .whitespacesAndNewlines)
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
        guard let rawValue = defaults.string(forKey: DefaultsKey.thinkMode.rawValue) else {
            return .hidden
        }
        return MotifThinkMode(rawValue: rawValue) ?? .hidden
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
    case nativeMLXNotCompiled

    var errorDescription: String? {
        switch self {
        case .invalidEndpoint:
            "Endpoint must be a valid URL."
        case .nativeMLXNotCompiled:
            "Native MLX chat is not compiled into this app build. Rebuild with MOTIFKIT_ENABLE_MLX=1."
        }
    }
}
