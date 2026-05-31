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
    @Published var prompt = ""
    @Published var messages: [MotifChatMessage] = [
        .system(ChatStore.defaultSystemPrompt)
    ]
    @Published var isGenerating = false
    @Published var lastError: String?
    @Published var runtimeStatus = "Idle"
    @Published var capturedReasoning = ""

    private var generationTask: Task<Void, Never>?
    private static let defaults = UserDefaults.standard
    private static let defaultSystemPrompt = "You are Motif accessed through the configured local runtime. Be concise and helpful."

    var nativeMLXCompiledIn: Bool {
        #if MOTIFKIT_ENABLE_MLX
        true
        #else
        false
        #endif
    }

    func newChat() {
        cancel()
        prompt = ""
        capturedReasoning = ""
        lastError = nil
        runtimeStatus = "Idle"
        messages = [
            .system(Self.defaultSystemPrompt)
        ]
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
        messages.append(.assistant(""))
        let assistantID = messages[messages.count - 1].id
        isGenerating = true

        let parameters = MotifGenerationParameters(
            model: model,
            maxTokens: maxTokens,
            temperature: temperature,
            thinkMode: thinkMode
        )
        let requestMessages = messages.dropLast()

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
        backendMode = .nativeMLX
        runtimeStatus = "Native checkpoint selected"
        lastError = nil
    }

    func resetRuntimeSettings() {
        for key in DefaultsKey.allCases {
            ChatStore.defaults.removeObject(forKey: key.rawValue)
        }
        backendMode = .openAICompatible
        endpoint = "http://127.0.0.1:8080/v1"
        model = "motif"
        nativeModelDirectory = ChatStore.defaultNativeModelDirectory
        thinkMode = .hidden
        maxTokens = 512
        temperature = 0.6
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
            return try MotifMLXBackend(modelDirectory: resolvedNativeModelDirectory())
            #else
            throw MotifChatStoreError.nativeMLXNotCompiled
            #endif
        }
    }

    private func resolvedNativeModelDirectory() throws -> URL {
        let trimmed = nativeModelDirectory.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { throw MotifChatStoreError.emptyNativeModelDirectory }

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
    case thinkMode = "motif.chat.thinkMode"
    case maxTokens = "motif.chat.maxTokens"
    case temperature = "motif.chat.temperature"
}

private enum MotifChatStoreError: Error, LocalizedError {
    case invalidEndpoint
    case nativeMLXNotCompiled
    case emptyNativeModelDirectory

    var errorDescription: String? {
        switch self {
        case .invalidEndpoint:
            "Endpoint must be a valid URL."
        case .nativeMLXNotCompiled:
            "Native MLX chat is not compiled into this app build. Rebuild with MOTIFKIT_ENABLE_MLX=1."
        case .emptyNativeModelDirectory:
            "Native MLX model directory cannot be empty."
        }
    }
}
