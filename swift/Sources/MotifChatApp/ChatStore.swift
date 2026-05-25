import Foundation
import MotifKit

@MainActor
final class ChatStore: ObservableObject {
    @Published var endpoint = "http://127.0.0.1:8080/v1"
    @Published var model = "motif"
    @Published var thinkMode: MotifThinkMode = .hidden
    @Published var maxTokens = 512
    @Published var temperature = 0.6
    @Published var prompt = ""
    @Published var messages: [MotifChatMessage] = [
        .system("You are Motif running locally on Apple Silicon. Be concise and helpful.")
    ]
    @Published var isGenerating = false
    @Published var lastError: String?
    @Published var capturedReasoning = ""

    private var generationTask: Task<Void, Never>?

    func newChat() {
        cancel()
        prompt = ""
        capturedReasoning = ""
        lastError = nil
        messages = [
            .system("You are Motif running locally on Apple Silicon. Be concise and helpful.")
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
        guard let baseURL = URL(string: endpoint) else {
            lastError = "Endpoint must be a valid URL."
            return
        }

        lastError = nil
        capturedReasoning = ""
        prompt = ""
        messages.append(.user(trimmed))
        messages.append(.assistant(""))
        let assistantID = messages[messages.count - 1].id
        isGenerating = true

        let backend = OpenAICompatibleMotifBackend(baseURL: baseURL)
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
}
