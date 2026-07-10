#if canImport(MotifKitMLX)
import Foundation
import MotifKit
import MotifKitMLX
import Network

@main
struct MotifNativeServeCommand {
    static func main() async {
        do {
            try await run()
        } catch {
            FileHandle.standardError.write(Data("error: \(error.localizedDescription)\n".utf8))
            Foundation.exit(1)
        }
    }

    private static func run() async throws {
        let arguments = Array(CommandLine.arguments.dropFirst())
        guard let modelIndex = arguments.firstIndex(of: "--model"), arguments.indices.contains(modelIndex + 1) else {
            FileHandle.standardError.write(Data("usage: MotifNativeServe --model <converted-mlx-dir> [--host 127.0.0.1] [--port 8080] [--model-id motif] [--think-mode visible|hidden|captured]\n".utf8))
            Foundation.exit(2)
        }
        let modelDirectory = URL(fileURLWithPath: arguments[modelIndex + 1])
        let host = value(after: "--host", in: arguments) ?? "127.0.0.1"
        let port = UInt16(value(after: "--port", in: arguments) ?? "8080") ?? 8080
        let modelID = value(after: "--model-id", in: arguments) ?? "motif"
        let thinkMode = MotifThinkMode(rawValue: value(after: "--think-mode", in: arguments) ?? "visible") ?? .visible

        FileHandle.standardError.write(Data("Loading native Swift Motif runtime from \(modelDirectory.path) …\n".utf8))
        let runtime = try await MotifMLXNativeRuntime.load(modelDirectory: modelDirectory)
        let server = NativeOpenAIServer(runtime: runtime, host: host, port: port, modelID: modelID, defaultThinkMode: thinkMode)
        try server.start()
        FileHandle.standardError.write(Data("Serving \(modelID) on http://\(host):\(port)/v1\n".utf8))
        await server.waitUntilCancelled()
    }

    private static func value(after flag: String, in arguments: [String]) -> String? {
        guard let index = arguments.firstIndex(of: flag), arguments.indices.contains(index + 1) else { return nil }
        return arguments[index + 1]
    }
}

private final class NativeOpenAIServer: @unchecked Sendable {
    let runtime: MotifMLXNativeRuntime
    let host: String
    let port: UInt16
    let modelID: String
    let defaultThinkMode: MotifThinkMode
    private var listener: NWListener?

    init(runtime: MotifMLXNativeRuntime, host: String, port: UInt16, modelID: String, defaultThinkMode: MotifThinkMode) {
        self.runtime = runtime
        self.host = host
        self.port = port
        self.modelID = modelID
        self.defaultThinkMode = defaultThinkMode
    }

    func start() throws {
        let listener = try NWListener(using: .tcp, on: NWEndpoint.Port(rawValue: port)!)
        listener.service = nil
        listener.newConnectionHandler = { [weak self] connection in
            guard let self else { return }
            connection.start(queue: .global())
            self.receive(connection: connection, accumulated: Data())
        }
        listener.start(queue: .global())
        self.listener = listener
    }

    func waitUntilCancelled() async {
        // Bridge both SIGTERM and Swift Task cancellation into a single
        // continuation so cleanup (listener?.cancel()) is always reached.
        //
        // Bug fixed: the original `try await Task.sleep` loop would throw
        // CancellationError on structured-concurrency cancellation, which
        // propagated out of the `throws` function *before* reaching
        // `listener?.cancel()`.  SIGTERM also never set Task.isCancelled, so
        // the listener was never cancelled on normal external termination.
        final class Box: @unchecked Sendable {
            var continuation: CheckedContinuation<Void, Never>?
            var resumed = false
            func resume() {
                guard !resumed else { return }
                resumed = true
                continuation?.resume()
            }
        }
        let box = Box()
        // Suppress the default SIGTERM disposition so our handler runs instead.
        signal(SIGTERM, SIG_IGN)
        let src = DispatchSource.makeSignalSource(signal: SIGTERM, queue: .main)
        src.setEventHandler { box.resume() }
        src.resume()
        await withTaskCancellationHandler {
            await withCheckedContinuation { (c: CheckedContinuation<Void, Never>) in
                box.continuation = c
                // If the Task was already cancelled before we stored the
                // continuation, resume immediately.
                if Task.isCancelled { box.resume() }
            }
        } onCancel: {
            box.resume()
        }
        src.cancel()
        signal(SIGTERM, SIG_DFL)
        listener?.cancel()
    }

    private func receive(connection: NWConnection, accumulated: Data) {
        connection.receive(minimumIncompleteLength: 1, maximumLength: 1 << 20) { [weak self] data, _, isComplete, error in
            guard let self else { return }
            var buffer = accumulated
            if let data { buffer.append(data) }
            if error != nil || isComplete {
                connection.cancel()
                return
            }
            if let request = HTTPRequest(data: buffer), request.isComplete {
                Task { await self.handle(request: request, connection: connection) }
            } else {
                self.receive(connection: connection, accumulated: buffer)
            }
        }
    }

    private func handle(request: HTTPRequest, connection: NWConnection) async {
        do {
            if request.method == "GET", request.path == "/v1/models" {
                try await sendJSON(connection, status: 200, payload: [
                    "object": "list",
                    "data": [["id": modelID, "object": "model", "created": Int(Date().timeIntervalSince1970)]],
                ])
                connection.cancel()
                return
            }
            guard request.method == "POST", request.path == "/v1/chat/completions" else {
                try await sendJSON(connection, status: 404, payload: ["error": ["message": "not found"]])
                connection.cancel()
                return
            }
            // Parity with the Python server: a malformed JSON body is a client
            // error (400 `bad json: …`), NOT a 500. Catch the decode failure
            // specifically so genuine runtime faults below still surface as 500.
            let chat: ChatCompletionRequest
            do {
                chat = try JSONDecoder().decode(ChatCompletionRequest.self, from: request.body)
            } catch {
                try await sendJSON(connection, status: 400, payload: ["error": ["message": "bad json: \(error)"]])
                connection.cancel()
                return
            }
            guard !chat.messages.isEmpty else {
                try await sendJSON(connection, status: 400, payload: ["error": ["message": "messages required"]])
                connection.cancel()
                return
            }
            // OpenAI-style prompt-based tool calling (parity with the Python
            // server): the raw `tools` array and `tool_choice` are re-read from
            // the body (ChatCompletionRequest omits them for brevity). When
            // tools are declared and not suppressed, we inject the same
            // deterministic preamble MotifToolCalling.buildToolsPreamble
            // produces (byte-identical to Python) as a leading system message.
            // Execution stays CLIENT-SIDE, matching OpenAI semantics — the
            // server only surfaces the parsed tool_calls.
            let tools = Self.extractTools(from: request.body)
            var messages = chat.motifMessages
            if !tools.isEmpty {
                let preamble = MotifToolCalling.buildToolsPreamble(tools)
                messages.insert(MotifChatMessage(role: .system, content: preamble), at: 0)
            }
            let parameters = chat.parameters(defaultThinkMode: defaultThinkMode)
            if chat.stream == true {
                try await streamChat(messages: messages, parameters: parameters, tools: tools, includeUsage: chat.includeUsage, connection: connection)
            } else {
                try await completeChat(messages: messages, parameters: parameters, tools: tools, connection: connection)
            }
            connection.cancel()
        } catch {
            try? await sendJSON(connection, status: 500, payload: ["error": ["message": String(describing: error)]])
            connection.cancel()
        }
    }

    /// Extract the OpenAI `tools` array from the raw request body, honouring
    /// `tool_choice: "none"` (which suppresses tools entirely, matching the
    /// Python server). Returns the tool dictionaries as `[[String: Any]]`.
    private static func extractTools(from body: Data) -> [Any] {
        guard let root = try? JSONSerialization.jsonObject(with: body) as? [String: Any] else { return [] }
        if let choice = root["tool_choice"] as? String, choice == "none" { return [] }
        guard let tools = root["tools"] as? [Any] else { return [] }
        return tools
    }

    private func streamChat(messages: [MotifChatMessage], parameters: MotifGenerationParameters, tools: [Any], includeUsage: Bool, connection: NWConnection) async throws {
        try await sendHeader(connection, status: 200, contentType: "text/event-stream", extraHeaders: ["Cache-Control": "no-cache"])
        let requestID = "chatcmpl-" + UUID().uuidString.replacingOccurrences(of: "-", with: "").prefix(24)
        let created = Int(Date().timeIntervalSince1970)
        let stream = runtime.streamResponse(messages: messages, parameters: parameters)
        // Prompt-based tool calling needs the FULL output to extract the first
        // JSON tool-call (the small model loops), so when tools are declared we
        // buffer instead of streaming content deltas, then emit a single
        // tool_calls chunk at the end (parity with the Python server).
        let toolNames = MotifToolCalling.toolNames(from: tools)
        var toolBuffer: String? = tools.isEmpty ? nil : ""
        // Parity with the Python server: captured reasoning is attached to the
        // terminal stop chunk rather than emitted as a separate intermediate
        // SSE event. We buffer it here and merge it into the `.completed` chunk.
        // The Swift SSE client (`OpenAICompatibleMotifBackend.emit`) reads the
        // top-level `reasoning` field off whatever chunk carries it, so the
        // client-visible event order is unchanged by this move.
        var capturedReasoning: String?
        // Authoritative token usage carried by the terminal `.completed` event.
        // Used only when the client explicitly requests OpenAI's trailing
        // streaming usage chunk (`stream_options.include_usage`).
        var terminalUsage: MotifGenerationUsage?
        for try await event in stream {
            switch event {
            case .text(let text):
                if toolBuffer != nil {
                    // Buffer instead of streaming: we need the full text to
                    // extract the first tool call.
                    toolBuffer! += text
                    continue
                }
                let payload: [String: Any] = [
                    "id": String(requestID), "object": "chat.completion.chunk", "created": created, "model": modelID,
                    "choices": [["index": 0, "delta": ["content": text], "finish_reason": NSNull()]],
                ]
                try await sendSSE(connection, payload: payload)
            case .reasoning(let reasoning):
                capturedReasoning = reasoning
            case .completed(let usage):
                terminalUsage = usage
                // Tools declared: parse the buffered output. If a tool call is
                // found, emit a tool_calls delta chunk and finish with
                // finish_reason "tool_calls"; otherwise flush the buffered text.
                var finishReason = "stop"
                if let buffered = toolBuffer {
                    if let call = MotifToolCalling.parseToolCall(text: buffered, toolNames: toolNames) {
                        let arguments = argumentsJSONString(call.arguments)
                        let toolChunk: [String: Any] = [
                            "id": String(requestID), "object": "chat.completion.chunk", "created": created, "model": modelID,
                            "choices": [[
                                "index": 0,
                                "delta": ["tool_calls": [[
                                    "index": 0,
                                    "id": "call_" + UUID().uuidString.replacingOccurrences(of: "-", with: "").prefix(24),
                                    "type": "function",
                                    "function": ["name": call.name, "arguments": arguments],
                                ]]],
                                "finish_reason": NSNull(),
                            ]],
                        ]
                        try await sendSSE(connection, payload: toolChunk)
                        finishReason = "tool_calls"
                    } else if !buffered.isEmpty {
                        let payload: [String: Any] = [
                            "id": String(requestID), "object": "chat.completion.chunk", "created": created, "model": modelID,
                            "choices": [["index": 0, "delta": ["content": buffered], "finish_reason": NSNull()]],
                        ]
                        try await sendSSE(connection, payload: payload)
                    }
                }
                var final: [String: Any] = [
                    "id": String(requestID), "object": "chat.completion.chunk", "created": created, "model": modelID,
                    "choices": [["index": 0, "delta": [:], "finish_reason": finishReason]],
                ]
                if let capturedReasoning, !capturedReasoning.isEmpty {
                    final["reasoning"] = capturedReasoning
                }
                try await sendSSE(connection, payload: final)
            }
        }
        // OpenAI-compatible `stream_options: {include_usage: true}`: after the
        // terminal stop chunk and before `[DONE]`, emit one extra chunk with
        // empty `choices` and a `usage` object. All fields come from the
        // authoritative terminal `MotifGenerationUsage` surfaced by the runtime
        // — the same source used by the non-streaming response. If a backend
        // cannot report usage, omit the chunk instead of emitting wrong counts.
        if includeUsage, let usage = terminalUsage {
            let usageChunk: [String: Any] = [
                "id": String(requestID), "object": "chat.completion.chunk", "created": created, "model": modelID,
                "choices": [],
                "usage": [
                    "prompt_tokens": usage.promptTokens,
                    "completion_tokens": usage.completionTokens,
                    "total_tokens": usage.totalTokens,
                ],
            ]
            try await sendSSE(connection, payload: usageChunk)
        }
        try await sendRaw(connection, Data("data: [DONE]\n\n".utf8))
    }

    private func completeChat(messages: [MotifChatMessage], parameters: MotifGenerationParameters, tools: [Any], connection: NWConnection) async throws {
        var content = ""
        var reasoning: String?
        var usage: MotifGenerationUsage?
        let stream = runtime.streamResponse(messages: messages, parameters: parameters)
        for try await event in stream {
            switch event {
            case .text(let text): content += text
            case .reasoning(let text): reasoning = text
            // Parity with the Python server: `usage` is populated from the
            // terminal generation result. The native MLX runtime surfaces the
            // real prompt/generation token counts on the `.completed` event
            // (sourced from MLX's `generate(...)` `.info` completion). Match the
            // Python `usage` shape exactly (prompt_tokens / completion_tokens /
            // total_tokens). See docs/server-parity.md.
            case .completed(let completedUsage): usage = completedUsage
            }
        }
        let resolvedUsage = usage ?? MotifGenerationUsage(promptTokens: 0, completionTokens: 0)

        // Prompt-based tool calling (parity with Python): when tools were
        // declared, try to extract the first JSON tool-call from the (possibly
        // looping) output and shape it as OpenAI `tool_calls` with
        // finish_reason "tool_calls". Falls back to a normal content response.
        var message: [String: Any] = ["role": "assistant", "content": content]
        var finishReason = "stop"
        if !tools.isEmpty,
           let call = MotifToolCalling.parseToolCall(text: content, toolNames: MotifToolCalling.toolNames(from: tools)) {
            message = [
                "role": "assistant",
                "content": NSNull(),
                "tool_calls": [[
                    "id": "call_" + UUID().uuidString.replacingOccurrences(of: "-", with: "").prefix(24),
                    "type": "function",
                    "function": ["name": call.name, "arguments": argumentsJSONString(call.arguments)],
                ]],
            ]
            finishReason = "tool_calls"
        }

        var payload: [String: Any] = [
            "id": "chatcmpl-" + UUID().uuidString.replacingOccurrences(of: "-", with: "").prefix(24),
            "object": "chat.completion",
            "created": Int(Date().timeIntervalSince1970),
            "model": modelID,
            "choices": [["index": 0, "message": message, "finish_reason": finishReason]],
            "usage": [
                "prompt_tokens": resolvedUsage.promptTokens,
                "completion_tokens": resolvedUsage.completionTokens,
                "total_tokens": resolvedUsage.totalTokens,
            ],
        ]
        if let reasoning { payload["reasoning"] = reasoning }
        try await sendJSON(connection, status: 200, payload: payload)
    }
}

/// Serialize parsed tool-call arguments back into a JSON string (OpenAI places
/// `arguments` as a JSON-encoded string). Keys are sorted for determinism.
private func argumentsJSONString(_ arguments: [String: MotifJSONValue]) -> String {
    let object: [String: Any] = arguments.mapValues { $0.anyValue }
    if let data = try? JSONSerialization.data(withJSONObject: object, options: [.sortedKeys]),
       let string = String(data: data, encoding: .utf8) {
        return string
    }
    return "{}"
}

private struct HTTPRequest {
    var method: String
    var path: String
    var headers: [String: String]
    var body: Data
    var isComplete: Bool

    init?(data: Data) {
        guard let marker = data.range(of: Data("\r\n\r\n".utf8)) else { return nil }
        let head = data[..<marker.lowerBound]
        guard let headText = String(data: head, encoding: .utf8) else { return nil }
        let lines = headText.components(separatedBy: "\r\n")
        guard let requestLine = lines.first else { return nil }
        let parts = requestLine.split(separator: " ")
        guard parts.count >= 2 else { return nil }
        method = String(parts[0])
        path = String(parts[1])
        headers = [:]
        for line in lines.dropFirst() {
            let pieces = line.split(separator: ":", maxSplits: 1).map(String.init)
            if pieces.count == 2 { headers[pieces[0].lowercased()] = pieces[1].trimmingCharacters(in: .whitespaces) }
        }
        let bodyStart = marker.upperBound
        let contentLength = Int(headers["content-length"] ?? "0") ?? 0
        body = data[bodyStart...]
        isComplete = body.count >= contentLength
        if body.count > contentLength { body = body.prefix(contentLength) }
    }
}

private struct ChatCompletionRequest: Decodable {
    var messages: [Message]
    var stream: Bool?
    var maxTokens: Int?
    var temperature: Double?
    var thinkMode: MotifThinkMode?
    var streamOptions: StreamOptions?

    /// Whether to emit OpenAI's trailing streaming usage chunk.
    var includeUsage: Bool { streamOptions?.includeUsage == true }

    enum CodingKeys: String, CodingKey {
        case messages, stream, temperature
        case maxTokens = "max_tokens"
        case thinkMode = "think_mode"
        case streamOptions = "stream_options"
    }

    struct StreamOptions: Decodable {
        var includeUsage: Bool?

        enum CodingKeys: String, CodingKey {
            case includeUsage = "include_usage"
        }
    }

    struct Message: Decodable {
        var role: String
        var content: String
    }

    var motifMessages: [MotifChatMessage] {
        messages.map { message in
            MotifChatMessage(role: MotifRole(rawValue: message.role) ?? .user, content: message.content)
        }
    }

    func parameters(defaultThinkMode: MotifThinkMode) -> MotifGenerationParameters {
        MotifGenerationParameters(
            maxTokens: maxTokens ?? 256,
            temperature: temperature ?? 0,
            thinkMode: thinkMode ?? defaultThinkMode
        )
    }
}

/// Map an HTTP status code to its standard reason phrase.
///
/// The previous implementation hardcoded `"ERROR"` for every non-200 status,
/// producing wire lines like `HTTP/1.1 404 ERROR`. That diverges from the
/// Python server (and from RFC 7231), which emit proper reason phrases.
private func reasonPhrase(for status: Int) -> String {
    switch status {
    case 200: return "OK"
    case 400: return "Bad Request"
    case 404: return "Not Found"
    case 500: return "Internal Server Error"
    default: return "OK"
    }
}

private func sendHeader(_ connection: NWConnection, status: Int, contentType: String, extraHeaders: [String: String] = [:]) async throws {
    var header = "HTTP/1.1 \(status) \(reasonPhrase(for: status))\r\nContent-Type: \(contentType)\r\nConnection: close\r\n"
    for (key, value) in extraHeaders { header += "\(key): \(value)\r\n" }
    header += "\r\n"
    try await sendRaw(connection, Data(header.utf8))
}

private func sendJSON(_ connection: NWConnection, status: Int, payload: [String: Any]) async throws {
    let body = try JSONSerialization.data(withJSONObject: payload)
    let header = "HTTP/1.1 \(status) \(reasonPhrase(for: status))\r\nContent-Type: application/json\r\nContent-Length: \(body.count)\r\nConnection: close\r\n\r\n"
    var data = Data(header.utf8)
    data.append(body)
    try await sendRaw(connection, data)
}

private func sendSSE(_ connection: NWConnection, payload: [String: Any]) async throws {
    let data = try JSONSerialization.data(withJSONObject: payload)
    try await sendRaw(connection, Data("data: ".utf8) + data + Data("\n\n".utf8))
}

private func sendRaw(_ connection: NWConnection, _ data: Data) async throws {
    try await withCheckedThrowingContinuation { (continuation: CheckedContinuation<Void, any Error>) in
        connection.send(content: data, completion: .contentProcessed { error in
            if let error { continuation.resume(throwing: error) } else { continuation.resume() }
        })
    }
}
#else
@main
struct MotifNativeServeCommand {
    static func main() {
        print("MotifNativeServe requires MOTIFKIT_ENABLE_MLX=1")
    }
}
#endif
