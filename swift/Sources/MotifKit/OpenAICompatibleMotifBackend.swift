import Foundation

public struct OpenAICompatibleMotifBackend: MotifChatBackend {
    public var baseURL: URL
    public var apiKey: String
    public var session: URLSession

    public init(
        baseURL: URL = URL(string: "http://127.0.0.1:8080/v1")!,
        apiKey: String = "ignored",
        session: URLSession = .shared
    ) {
        self.baseURL = baseURL
        self.apiKey = apiKey
        self.session = session
    }

    public func streamResponse(
        messages: [MotifChatMessage],
        parameters: MotifGenerationParameters
    ) -> AsyncThrowingStream<MotifGenerationEvent, any Error> {
        AsyncThrowingStream { continuation in
            let task = Task {
                do {
                    let url = baseURL.appendingPathComponent("chat/completions")
                    var request = URLRequest(url: url)
                    request.httpMethod = "POST"
                    request.setValue("application/json", forHTTPHeaderField: "Content-Type")
                    request.setValue("Bearer \(apiKey)", forHTTPHeaderField: "Authorization")
                    request.httpBody = try JSONSerialization.data(withJSONObject: requestBody(
                        messages: messages,
                        parameters: parameters
                    ))

                    let (bytes, response) = try await session.bytes(for: request)
                    if let http = response as? HTTPURLResponse, !(200..<300).contains(http.statusCode) {
                        throw MotifBackendError.httpStatus(http.statusCode)
                    }

                    for try await line in bytes.lines {
                        try Task.checkCancellation()
                        guard line.hasPrefix("data:") else { continue }
                        let payload = line.dropFirst("data:".count).trimmingCharacters(in: .whitespaces)
                        if payload == "[DONE]" {
                            continuation.yield(.completed(usage: nil))
                            continuation.finish()
                            return
                        }
                        try emit(payload: payload, continuation: continuation)
                    }

                    continuation.yield(.completed(usage: nil))
                    continuation.finish()
                } catch is CancellationError {
                    continuation.finish()
                } catch {
                    continuation.finish(throwing: error)
                }
            }

            continuation.onTermination = { _ in task.cancel() }
        }
    }

    private func requestBody(
        messages: [MotifChatMessage],
        parameters: MotifGenerationParameters
    ) -> [String: Any] {
        [
            "model": parameters.model,
            "stream": true,
            "max_tokens": parameters.maxTokens,
            "temperature": parameters.temperature,
            "think_mode": parameters.thinkMode.rawValue,
            "messages": messages.map { message in
                ["role": message.role.rawValue, "content": message.content]
            },
        ]
    }

    private func emit(
        payload: String,
        continuation: AsyncThrowingStream<MotifGenerationEvent, any Error>.Continuation
    ) throws {
        guard let data = payload.data(using: .utf8),
              let object = try JSONSerialization.jsonObject(with: data) as? [String: Any]
        else {
            throw MotifBackendError.malformedServerEvent(payload)
        }

        if let reasoning = object["reasoning"] as? String, !reasoning.isEmpty {
            continuation.yield(.reasoning(reasoning))
        }

        guard let choices = object["choices"] as? [[String: Any]],
              let firstChoice = choices.first
        else { return }

        if let delta = firstChoice["delta"] as? [String: Any],
           let content = delta["content"] as? String,
           !content.isEmpty {
            continuation.yield(.text(content))
        }

        if let finishReason = firstChoice["finish_reason"] as? String,
           !finishReason.isEmpty,
           finishReason != "null" {
            continuation.yield(.completed(usage: nil))
        }
    }
}
