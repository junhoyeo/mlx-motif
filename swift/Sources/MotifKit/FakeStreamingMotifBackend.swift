import Foundation

/// Deterministic in-process `MotifChatBackend` used to drive the SwiftUI app
/// from XCUITest with **no model and no network**.
///
/// It streams a canned answer as a series of `.text` chunks with a configurable
/// per-chunk delay so a UI test has a real window to click **Stop** mid-stream,
/// honours cooperative cancellation (so the app's `CancellationError` path runs
/// and appends `[Cancelled]`), and mirrors the real backends' think-mode
/// contract by routing reasoning through `.reasoning` / inline `<think>` /
/// nothing for `.captured` / `.visible` / `.hidden` respectively.
///
/// The backend is always compiled in (it has no MLX dependency); it is only
/// *reached* when `ChatStore.buildBackend()` sees the `-UITestFakeBackend`
/// launch argument, so it never affects `swift run` / production behaviour.
public struct FakeStreamingMotifBackend: MotifChatBackend {
    public struct Config: Sendable {
        public var chunks: Int
        public var delay: Duration
        /// Pause between the reasoning phase and the first answer chunk. While
        /// no answer text has arrived the app treats the turn as "thinking", so
        /// a UI test can observe thinking-phase UI (e.g. the auto-expanded
        /// reasoning disclosure) for this long.
        public var prefaceDelay: Duration
        public var answer: String
        public var reasoning: String
        public var finishReason: MotifFinishReason

        public init(
            chunks: Int,
            delay: Duration,
            prefaceDelay: Duration = .zero,
            answer: String,
            reasoning: String,
            finishReason: MotifFinishReason
        ) {
            self.chunks = chunks
            self.delay = delay
            self.prefaceDelay = prefaceDelay
            self.answer = answer
            self.reasoning = reasoning
            self.finishReason = finishReason
        }

        /// Builds a config from the app's launch environment so a UI test can
        /// tune pacing/content without a rebuild:
        ///   MOTIF_UITEST_FAKE_CHUNKS           (Int,   default 24)
        ///   MOTIF_UITEST_FAKE_DELAY_MS         (Int,   default 60)   — 24 * 60ms ≈ 1.4s stream
        ///   MOTIF_UITEST_FAKE_PREFACE_DELAY_MS (Int,   default 0)    — thinking-phase window
        ///   MOTIF_UITEST_FAKE_ANSWER           (String)
        ///   MOTIF_UITEST_FAKE_REASONING        (String)
        ///   MOTIF_UITEST_FAKE_FINISH           ("stop" | "length", default "stop")
        public static func fromEnvironment(
            _ env: [String: String] = ProcessInfo.processInfo.environment
        ) -> Config {
            let chunks = env["MOTIF_UITEST_FAKE_CHUNKS"].flatMap(Int.init) ?? 24
            let delayMS = env["MOTIF_UITEST_FAKE_DELAY_MS"].flatMap(Int.init) ?? 60
            let prefaceMS = env["MOTIF_UITEST_FAKE_PREFACE_DELAY_MS"].flatMap(Int.init) ?? 0
            let answer = env["MOTIF_UITEST_FAKE_ANSWER"]
                ?? "Hello from the fake Motif backend. "
            let reasoning = env["MOTIF_UITEST_FAKE_REASONING"]
                ?? "Thinking step by step about the request."
            let finish: MotifFinishReason =
                (env["MOTIF_UITEST_FAKE_FINISH"] == "length") ? .length : .stop
            return Config(
                chunks: max(1, chunks),
                delay: .milliseconds(max(0, delayMS)),
                prefaceDelay: .milliseconds(max(0, prefaceMS)),
                answer: answer,
                reasoning: reasoning,
                finishReason: finish
            )
        }
    }

    private let config: Config

    public init(config: Config = .fromEnvironment()) {
        self.config = config
    }

    public func streamResponse(
        messages: [MotifChatMessage],
        parameters: MotifGenerationParameters
    ) -> AsyncThrowingStream<MotifGenerationEvent, any Error> {
        let config = self.config
        let thinkMode = parameters.thinkMode
        return AsyncThrowingStream { continuation in
            let task = Task {
                do {
                    // Reasoning first, routed by think-mode exactly like the real
                    // backends: captured -> .reasoning events; visible -> inline
                    // <think> text; hidden -> nothing.
                    switch thinkMode {
                    case .captured:
                        continuation.yield(.reasoning(config.reasoning))
                    case .visible:
                        continuation.yield(.text("<think>\(config.reasoning)</think>\n"))
                    case .hidden:
                        break
                    }
                    if config.prefaceDelay != .zero {
                        try await Task.sleep(for: config.prefaceDelay)
                    }

                    for _ in 0..<config.chunks {
                        try Task.checkCancellation()
                        continuation.yield(.text(config.answer))
                        if config.delay != .zero {
                            try await Task.sleep(for: config.delay)
                        }
                    }
                    continuation.yield(
                        .completed(usage: nil, finishReason: config.finishReason)
                    )
                    continuation.finish()
                } catch {
                    // Cancellation (or any error) ends the stream; the app's
                    // consuming loop surfaces the cancel via its own
                    // Task.checkCancellation().
                    continuation.finish(throwing: error)
                }
            }
            continuation.onTermination = { _ in task.cancel() }
        }
    }
}
