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
    @Published var backendMode: MotifChatBackendMode = .openAICompatible
    @Published var endpoint = "http://127.0.0.1:8080/v1"
    @Published var model = "motif"
    @Published var nativeModelDirectory = ProcessInfo.processInfo.environment["MOTIF_MODEL_DIR"] ?? "~/.models/motif-2.6b-mlx-q4"
    @Published var thinkMode: MotifThinkMode = .hidden
    @Published var maxTokens = 512
    @Published var temperature = 0.6
    @Published var prompt = ""
    @Published var messages: [MotifChatMessage] = [
        .system(ChatStore.defaultSystemPrompt)
    ]
    @Published var isGenerating = false
    @Published var lastError: String?
    @Published var capturedReasoning = ""

    private var generationTask: Task<Void, Never>?
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
        messages = [
            .system(Self.defaultSystemPrompt)
        ]
    }

    func cancel() {
        generationTask?.cancel()
        generationTask = nil
        isGenerating = false
    }

    func send() {
        let trimmed = prompt.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty, !isGenerating else { return }

        let backend: any MotifChatBackend
        do {
            backend = try makeBackend()
        } catch {
            lastError = error.localizedDescription
            return
        }

        lastError = nil
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
                    self.append("\n\n[Cancelled]", toAssistantMessage: assistantID)
                }
            } catch {
                await MainActor.run {
                    self.lastError = error.localizedDescription
                    self.append("\n\n[Error: \(error.localizedDescription)]", toAssistantMessage: assistantID)
                }
            }

            await MainActor.run {
                self.isGenerating = false
                self.generationTask = nil
            }
        }
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
