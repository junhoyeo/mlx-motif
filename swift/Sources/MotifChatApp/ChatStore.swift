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

    private var generationTask: Task<Void, Never>?
    private var nativeDirectoryAccess: NativeDirectoryAccessGrant?
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
        capturedReasoning = ""
        prompt = ""
        messages.append(.user(trimmed))
        // Build the request transcript before adding the streaming placeholder,
        // applying the context-budget guard so long chats don't overflow.
        let requestMessages = messagesWithinBudget()
        messages.append(.assistant(""))
        let assistantID = messages[messages.count - 1].id
        isGenerating = true

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
                            self.append(text, toAssistantMessage: assistantID)
                        case .reasoning(let reasoning):
                            self.capturedReasoning += reasoning
                        case .completed:
                            break
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
                }
            }

            await MainActor.run {
                // Release scoped access only after every file read for this load is
                // done — this block runs on all completion paths (success/error/cancel).
                self.releaseNativeDirectoryAccess()
                self.isGenerating = false
                self.generationTask = nil
                if self.lastError == nil, self.runtimeStatus != "Cancelled" {
                    self.runtimeStatus = "Idle"
                }
            }
        }
    }

    func selectNativeModelDirectory(_ url: URL) {
        nativeModelDirectory = url.path
        persistSecurityScopedBookmark(for: url)
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
        runtimeStatus = "Settings reset"
        lastError = nil
    }

    private func append(_ text: String, toAssistantMessage id: UUID) {
        guard let index = messages.firstIndex(where: { $0.id == id }) else { return }
        messages[index].content += text
    }

    private func makeBackend() throws -> any MotifChatBackend {
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
