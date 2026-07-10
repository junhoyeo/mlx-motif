#if canImport(MLX) && canImport(MLXNN) && canImport(MLXLLM) && canImport(MLXLMCommon)
import Foundation
import MLX
import MLXLLM
@preconcurrency import MLXLMCommon
import MLXNN
import MotifKit
import Tokenizers

public enum MotifMLXNativeRuntimeError: Error, LocalizedError {
    case modelDirectoryNotLoadable(String)
    case missingCheckpointWeights(URL)
    case missingTokenizer(URL)
    case notEnoughTokensForPerplexity(Int)

    public var errorDescription: String? {
        switch self {
        case .modelDirectoryNotLoadable(let detail):
            "Motif model directory is not loadable by the native Swift MLX runtime: \(detail)"
        case .missingCheckpointWeights(let url):
            "Motif model directory has no safetensors checkpoint weights: \(url.path)"
        case .missingTokenizer(let url):
            "Motif model directory has no tokenizer files usable by swift-tokenizers: \(url.path)"
        case .notEnoughTokensForPerplexity(let count):
            "Need at least 2 tokens to compute perplexity; got \(count)."
        }
    }
}

public struct MotifMLXChatInputProcessor: UserInputProcessor {
    public let tokenizer: any Tokenizer
    public let messageGenerator: any MessageGenerator
    public let chatTemplate: String?

    public init(
        tokenizer: any Tokenizer,
        messageGenerator: any MessageGenerator = DefaultMessageGenerator(),
        chatTemplate: String? = nil
    ) {
        self.tokenizer = tokenizer
        self.messageGenerator = messageGenerator
        self.chatTemplate = chatTemplate
    }

    public func prepare(input: UserInput) async throws -> LMInput {
        let messages = messageGenerator.generate(from: input)
        let tokenIDs = try tokenIDs(for: messages, tools: input.tools)
        return LMInput(tokens: MLXArray(tokenIDs))
    }

    public func promptText(for messages: [MotifChatMessage]) throws -> String {
        let generated = messageGenerator.generate(messages: messages.map(Self.chatMessage(from:)))
        let tokenIDs = try tokenIDs(for: generated, tools: nil)
        return tokenizer.decode(tokens: tokenIDs)
    }

    private func tokenIDs(for messages: [Message], tools: [ToolSpec]?) throws -> [Int] {
        do {
            if let chatTemplate {
                return try tokenizer.applyChatTemplate(
                    messages: messages,
                    chatTemplate: .literal(chatTemplate),
                    addGenerationPrompt: true,
                    truncation: false,
                    maxLength: nil,
                    tools: tools
                )
            }
            return try tokenizer.applyChatTemplate(
                messages: messages,
                chatTemplate: nil,
                addGenerationPrompt: true,
                truncation: false,
                maxLength: nil,
                tools: tools
            )
        } catch {
            guard let chatTemplate else { throw error }
            return tokenizer.encode(text: fallbackPrompt(messages: messages, chatTemplate: chatTemplate))
        }
    }

    private func fallbackPrompt(messages: [Message], chatTemplate: String) -> String {
        if chatTemplate.contains("<|beginoftext|>") {
            return fallbackReasoningPrompt(messages: messages)
        }
        var prompt = ""
        for message in messages {
            let role = stringValue(message["role"]) ?? "user"
            let content = stringValue(message["content"]) ?? ""
            prompt += "<|startofturn|><|\(role)|>\n\n\(content)<|endofturn|>"
        }
        prompt += "<|startofturn|><|assistant|>\n\n"
        return prompt
    }

    private func fallbackReasoningPrompt(messages: [Message]) -> String {
        var prompt = "<|beginoftext|><startofturn|>\n"
        let systemMessages = messages.filter { stringValue($0["role"]) == "system" }
        if !systemMessages.isEmpty {
            prompt += "<|system|>\n"
            for message in systemMessages {
                prompt += (stringValue(message["content"]) ?? "") + "\n"
            }
            prompt += "<|endofturn|>"
        }
        for message in messages where stringValue(message["role"]) != "system" {
            let role = stringValue(message["role"]) ?? "user"
            let content = stringValue(message["content"]) ?? ""
            prompt += "<|startofturn|>\n<|\(role)|>\n\(content)\n<|endofturn|>"
        }
        prompt += "<|assistant|><think>\n"
        return prompt
    }

    private func stringValue(_ value: (any Sendable)?) -> String? {
        value as? String
    }

    public static func userInput(from messages: [MotifChatMessage]) -> UserInput {
        UserInput(chat: messages.map(chatMessage(from:)))
    }

    public static func chatMessage(from message: MotifChatMessage) -> Chat.Message {
        switch message.role {
        case .system:
            .system(message.content)
        case .user:
            .user(message.content)
        case .assistant:
            .assistant(message.content)
        }
    }
}

public final class MotifMLXNativeRuntime: @unchecked Sendable {
    public let bundle: MotifModelBundle
    public let modelConfiguration: ModelConfiguration
    public let generationContext: ModelContext
    public let inputProcessor: MotifMLXChatInputProcessor

    /// Persistent KV-cache carried across turns of a single conversation.
    ///
    /// `streamResponse` re-prefilled the entire conversation on every turn,
    /// which is O(n²) as the chat grows. When the new prompt is an exact
    /// token-prefix extension of what this cache already holds, we feed only
    /// the new suffix and reuse the cached prefix instead of re-prefilling it.
    /// `nil` means "no reusable cache" — the next turn starts fresh.
    ///
    /// Reuse feeds the exact KV the model already computed — but only once the
    /// cache offset has been reconciled with `cachedTokenIDs`. `generateTokens`
    /// steps the stop token through the model before deciding to stop, so an
    /// EOS-terminated turn leaves the cache one position ahead of the tokens we
    /// recorded; `streamResponse` trims that surplus before carrying the cache
    /// forward (otherwise the next reused suffix would be appended at the wrong
    /// offset and shift every RoPE position by +1). With that reconciliation the
    /// reused prefix is bit-for-bit the KV computed for the recorded tokens.
    /// On the quantized (q4) checkpoint the resulting greedy output is
    /// a valid decode but is not guaranteed byte-identical to the legacy
    /// "re-prefill everything" path, because a single-token decode forward
    /// (`[1, 1]`) and a batched prefill forward (`[1, n]`) over the same tokens
    /// round slightly differently in the quantized matmul kernels. Reuse carries
    /// the KV built incrementally during generation; the legacy path rebuilds
    /// that region with a batched prefill, so the two can diverge by a few
    /// tokens. This is a model/kernel property, not a reuse bug — see the
    /// `MotifMLXNativeRuntimeCacheReuseTests` correctness gate.
    private var reuseCache: [KVCache]?

    /// Token IDs whose attention state is currently captured in `reuseCache`
    /// (prompt + generated tokens of the previous turn). Used to compute the
    /// longest common prefix against the next turn's prompt.
    private var cachedTokenIDs: [Int]?

    /// Test/diagnostic escape hatch: when true the runtime always resets the
    /// cache (behaving exactly like the original, cache-less implementation).
    /// Used by the correctness gate to produce the no-reuse baseline.
    private var reuseDisabled = false

    /// Token IDs generated during the most recent `streamResponse` turn.
    /// Exposed for the cross-turn correctness gate (reuse-vs-no-reuse parity).
    public private(set) var lastGeneratedTokenIDs: [Int] = []

    /// Whether the most recent `streamResponse` turn reused the carried cache
    /// (true) or prefilled the full prompt (false). Diagnostic only.
    public private(set) var lastTurnReusedCache = false

    /// Offset of the currently carried reuse cache, or `nil` when no cache is
    /// carried. Diagnostic only: the cache-reuse correctness gate asserts this
    /// equals ``cachedTokenCount`` after every turn (including EOS-terminated
    /// turns) so the reused suffix is always appended at the right position.
    public var reuseCacheOffset: Int? { reuseCache?.first?.offset }

    /// Number of token IDs the carried reuse cache claims to hold (prompt +
    /// generated tokens of the previous turn), or `nil` when no cache is
    /// carried. Diagnostic only; pairs with ``reuseCacheOffset``.
    public var cachedTokenCount: Int? { cachedTokenIDs?.count }

    public static func load(
        modelDirectory: URL,
        featureFlags: MotifRuntimeFeatureFlags = .fromEnvironment()
    ) async throws -> MotifMLXNativeRuntime {
        let bundle = try MotifModelBundle(directoryURL: modelDirectory)
        if let blocking = bundle.directoryValidation.blockingSummary {
            throw MotifMLXNativeRuntimeError.modelDirectoryNotLoadable(blocking)
        }
        guard bundle.checkpointMetadata.hasCheckpointWeights else {
            throw MotifMLXNativeRuntimeError.missingCheckpointWeights(modelDirectory)
        }
        guard bundle.tokenizerMetadata.hasTokenizerFiles else {
            throw MotifMLXNativeRuntimeError.missingTokenizer(modelDirectory)
        }

        let model = try MotifMLXModel(configuration: bundle.configuration, runtimeFeatures: featureFlags)
        let baseConfiguration: BaseConfiguration
        do {
            baseConfiguration = try JSONDecoder().decode(
                BaseConfiguration.self,
                from: Data(contentsOf: modelDirectory.appendingPathComponent(MotifModelBundle.configFileName))
            )
        } catch {
            throw MotifMLXNativeRuntimeError.modelDirectoryNotLoadable(
                "BaseConfiguration decode failed: \(String(reflecting: error))"
            )
        }
        try loadWeights(
            modelDirectory: modelDirectory,
            model: model,
            quantization: nil,
            perLayerQuantization: baseConfiguration.perLayerQuantization
        )
        model.fuseQueryKeyValueProjectionsIfPossible()

        let eosTokenIDs = Set(
            ([bundle.configuration.eosTokenId].compactMap { $0 })
                + (bundle.generationConfiguration?.eosTokenIds ?? [])
        )
        let modelConfiguration = ModelConfiguration(
            directory: modelDirectory,
            overrideTokenizer: "PreTrainedTokenizer",
            defaultPrompt: "Hello",
            eosTokenIds: eosTokenIDs
        )
        let tokenizer: any Tokenizer
        do {
            tokenizer = try await loadTokenizer(configuration: modelConfiguration, hub: defaultHubApi)
        } catch {
            throw MotifMLXNativeRuntimeError.modelDirectoryNotLoadable(
                "loadTokenizer failed: \(String(reflecting: error))"
            )
        }
        let inputProcessor = MotifMLXChatInputProcessor(
            tokenizer: tokenizer,
            messageGenerator: model.messageGenerator(tokenizer: tokenizer),
            chatTemplate: bundle.tokenizerMetadata.preferredChatTemplate
        )
        let context = ModelContext(
            configuration: modelConfiguration,
            model: model,
            processor: inputProcessor,
            tokenizer: tokenizer
        )
        return MotifMLXNativeRuntime(
            bundle: bundle,
            modelConfiguration: modelConfiguration,
            generationContext: context,
            inputProcessor: inputProcessor
        )
    }

    public init(
        bundle: MotifModelBundle,
        modelConfiguration: ModelConfiguration,
        generationContext: ModelContext,
        inputProcessor: MotifMLXChatInputProcessor
    ) {
        self.bundle = bundle
        self.modelConfiguration = modelConfiguration
        self.generationContext = generationContext
        self.inputProcessor = inputProcessor
    }

    /// Drop the carried KV-cache so the next `streamResponse` prefills the full
    /// prompt from scratch. Call when a new conversation begins or whenever
    /// cross-turn reuse must not apply.
    public func resetCache() {
        reuseCache = nil
        cachedTokenIDs = nil
    }

    /// Force the always-reset path (no cross-turn reuse). Intended for the
    /// correctness gate to compare reuse output against a cache-less baseline.
    public func setReuseEnabled(_ enabled: Bool) {
        reuseDisabled = !enabled
        if reuseDisabled {
            resetCache()
        }
    }

    /// Length of the longest common prefix of two token-ID sequences.
    private static func commonPrefixLength(_ lhs: [Int], _ rhs: [Int]) -> Int {
        var length = 0
        let limit = min(lhs.count, rhs.count)
        while length < limit, lhs[length] == rhs[length] {
            length += 1
        }
        return length
    }

    public func streamResponse(
        messages: [MotifChatMessage],
        parameters: MotifGenerationParameters
    ) -> AsyncThrowingStream<MotifGenerationEvent, any Error> {
        AsyncThrowingStream { continuation in
            Task {
                do {
                    var thinkFilter = ThinkStreamFilter(
                        mode: parameters.thinkMode,
                        startsInsideThinkBlock: try inputProcessor.promptText(for: messages).trimmingCharacters(in: .whitespacesAndNewlines).hasSuffix(ThinkStreamFilter.openTag)
                    )
                    let input = MotifMLXChatInputProcessor.userInput(from: messages)
                    let lmInput = try await inputProcessor.prepare(input: input)
                    let generationParameters = GenerateParameters(
                        maxTokens: parameters.maxTokens,
                        temperature: Float(parameters.temperature)
                    )

                    // Full prompt token IDs for this turn (used for prefix
                    // matching and to report the authoritative prompt-token
                    // count regardless of how many tokens we actually prefill).
                    let promptTokens = lmInput.text.tokens.asArray(Int.self)

                    // Decide whether the carried cache can be reused. Reuse is
                    // only valid when the cache holds a clean prefix of the new
                    // prompt — i.e. the previous turn's tokens are a strict
                    // prefix of this turn's tokens. Any divergence (user edited /
                    // deleted / regenerated) forces a full fresh prefill, which
                    // is byte-for-byte the original cache-less behavior.
                    let cache: [KVCache]
                    let prefillInput: LMInput
                    var reusedCache = false
                    if !reuseDisabled,
                        let existingCache = reuseCache,
                        let existingTokens = cachedTokenIDs {
                        let prefix = Self.commonPrefixLength(existingTokens, promptTokens)
                        if prefix == existingTokens.count, prefix > 0, prefix < promptTokens.count {
                            // Cache holds an exact prefix; prefill only the new
                            // suffix. The KVCache appends at its current offset,
                            // reusing the exact KV already computed for the
                            // prefix. (On q4 this is a valid greedy decode but
                            // may differ by a few tokens from a full re-prefill;
                            // see reuseCache doc and the correctness gate.)
                            cache = existingCache
                            prefillInput = LMInput(tokens: MLXArray(Array(promptTokens[prefix...])))
                            reusedCache = true
                        } else {
                            cache = generationContext.model.newCache(parameters: generationParameters)
                            prefillInput = lmInput
                        }
                    } else {
                        cache = generationContext.model.newCache(parameters: generationParameters)
                        prefillInput = lmInput
                    }
                    lastTurnReusedCache = reusedCache

                    var generatedTokenIDs: [Int] = []
                    var detokenizer = NaiveStreamingDetokenizer(tokenizer: generationContext.tokenizer)
                    let stream = try generateTokens(
                        input: prefillInput,
                        cache: cache,
                        parameters: generationParameters,
                        context: generationContext
                    )
                    for await generation in stream {
                        switch generation {
                        case .token(let token):
                            generatedTokenIDs.append(token)
                            detokenizer.append(token: token)
                            if let chunk = detokenizer.next() {
                                let visible = thinkFilter.feed(chunk)
                                if !visible.isEmpty {
                                    continuation.yield(.text(visible))
                                }
                            }
                        case .info(let completion):
                            let tail = thinkFilter.finish()
                            if !tail.isEmpty {
                                continuation.yield(.text(tail))
                            }
                            if parameters.thinkMode == .captured, !thinkFilter.capturedReasoning.isEmpty {
                                continuation.yield(.reasoning(thinkFilter.capturedReasoning))
                            }
                            // Carry the cache forward for the next turn. The
                            // cache now holds the full prompt plus everything we
                            // generated this turn.
                            lastGeneratedTokenIDs = generatedTokenIDs
                            if reuseDisabled {
                                resetCache()
                            } else {
                                // Reconcile the cache offset with the tokens we
                                // actually recorded before carrying it forward.
                                // `generateTokens` runs each token through the
                                // model *before* yielding it, and on an
                                // EOS-terminated turn the stop token is stepped
                                // (its KV appended, offset advanced) but never
                                // yielded — so the cache ends up one position
                                // ahead of `generatedTokenIDs`. Left unfixed, the
                                // next reused turn appends its suffix at the wrong
                                // offset on top of the stale stop-token KV,
                                // shifting every RoPE position by +1. Trim the
                                // surplus so offset == cachedTokenIDs.count and
                                // reuse is truly lossless (see the
                                // MotifMLXNativeRuntimeCacheReuseTests gate).
                                let recordedTokenCount = promptTokens.count + generatedTokenIDs.count
                                let extra = (cache.first?.offset ?? recordedTokenCount) - recordedTokenCount
                                if extra > 0 {
                                    for kvCache in cache where kvCache.isTrimmable {
                                        kvCache.trim(extra)
                                    }
                                }
                                reuseCache = cache
                                cachedTokenIDs = promptTokens + generatedTokenIDs
                            }
                            // Report the authoritative prompt-token count (the
                            // FULL prompt, not just the suffix we prefilled) so
                            // the usage contract is identical whether or not the
                            // cache was reused.
                            let usage = MotifGenerationUsage(
                                promptTokens: promptTokens.count,
                                completionTokens: completion.generationTokenCount
                            )
                            continuation.yield(.completed(usage: usage))
                        }
                    }
                    continuation.finish()
                } catch {
                    // Generation aborted before producing terminal info; the
                    // in-place cache may be partially advanced and no longer
                    // matches `cachedTokenIDs`, so drop it to force a clean
                    // prefill next turn.
                    resetCache()
                    continuation.finish(throwing: error)
                }
            }
        }
    }

    public func perplexity(
        text: String,
        chunkSize: Int = 512,
        maxTokens: Int = 2_048
    ) throws -> MotifMLXPerplexityResult {
        var tokenIDs = generationContext.tokenizer.encode(text: text)
        if maxTokens > 0, tokenIDs.count > maxTokens {
            tokenIDs = Array(tokenIDs.prefix(maxTokens))
        }
        guard tokenIDs.count >= 2 else {
            throw MotifMLXNativeRuntimeError.notEnoughTokensForPerplexity(tokenIDs.count)
        }

        let started = Date()
        var totalNLL: Double = 0
        var totalTokens = 0
        let stride = max(1, chunkSize)
        var chunks = 0

        var start = 0
        while start < tokenIDs.count - 1 {
            let end = min(tokenIDs.count, start + stride + 1)
            let chunk = Array(tokenIDs[start..<end])
            if chunk.count >= 2 {
                let inputIDs = Array(chunk.dropLast())
                let targetIDs = Array(chunk.dropFirst())
                let x = MLXArray(inputIDs, [1, inputIDs.count])
                let y = MLXArray(targetIDs, [1, targetIDs.count])
                let logits = generationContext.model(x, cache: nil).asType(.float32)
                let logProbabilities = log(softmax(logits, axis: -1))
                let gathered = takeAlong(logProbabilities, y.reshaped([1, targetIDs.count, 1]), axis: -1)
                let nll = -sum(gathered).item(Float.self)
                totalNLL += Double(nll)
                totalTokens += y.size
                chunks += 1
            }
            start += stride
        }

        let elapsed = Date().timeIntervalSince(started)
        let nllPerToken = totalNLL / Double(max(1, totalTokens))
        return MotifMLXPerplexityResult(
            perplexity: exp(nllPerToken),
            nllPerToken: nllPerToken,
            tokens: totalTokens,
            chunks: chunks,
            elapsedSeconds: elapsed,
            tokensPerSecond: elapsed > 0 ? Double(totalTokens) / elapsed : 0
        )
    }

    public func logitSnapshot(
        text: String,
        maxTokens: Int = 512,
        topK: Int = 10
    ) throws -> MotifMLXLogitSnapshot {
        var tokenIDs = generationContext.tokenizer.encode(text: text)
        if maxTokens > 0, tokenIDs.count > maxTokens {
            tokenIDs = Array(tokenIDs.prefix(maxTokens))
        }
        guard !tokenIDs.isEmpty else {
            throw MotifMLXNativeRuntimeError.notEnoughTokensForPerplexity(tokenIDs.count)
        }
        let started = Date()
        let logits = generationContext.model(MLXArray(tokenIDs, [1, tokenIDs.count]), cache: nil)
            .asType(.float32)
        let last = logits[0, tokenIDs.count - 1, 0...]
        eval(last)
        let values = last.asArray(Float.self)
        let top = values.enumerated()
            .sorted { lhs, rhs in lhs.element > rhs.element }
            .prefix(max(1, topK))
            .map { MotifMLXLogitEntry(token: $0.offset, logit: Double($0.element)) }
        let checksum = values.reduce(Double(0)) { partial, value in partial + Double(value) }
        return MotifMLXLogitSnapshot(
            promptTokens: tokenIDs.count,
            vocabularySize: values.count,
            checksum: checksum,
            topK: Array(top),
            elapsedSeconds: Date().timeIntervalSince(started)
        )
    }

    public func benchmarkGeneration(
        messages: [MotifChatMessage],
        maxTokens: Int = 64,
        temperature: Double = 0
    ) async throws -> MotifMLXGenerationBenchmark {
        let started = Date()
        let input = MotifMLXChatInputProcessor.userInput(from: messages)
        let lmInput = try await inputProcessor.prepare(input: input)
        let stream = try generateTokens(
            input: lmInput,
            parameters: GenerateParameters(maxTokens: maxTokens, temperature: Float(temperature)),
            context: generationContext,
            includeStopToken: false
        )
        var tokens: [Int] = []
        var info: MotifMLXGenerationInfo?
        for await event in stream {
            switch event {
            case .token(let token):
                tokens.append(token)
            case .info(let completion):
                info = MotifMLXGenerationInfo(
                    promptTokenCount: completion.promptTokenCount,
                    generationTokenCount: completion.generationTokenCount,
                    promptTime: completion.promptTime,
                    generateTime: completion.generateTime,
                    promptTokensPerSecond: completion.promptTokensPerSecond,
                    tokensPerSecond: completion.tokensPerSecond
                )
            }
        }
        let text = generationContext.tokenizer.decode(tokens: tokens)
        return MotifMLXGenerationBenchmark(
            promptCharacters: try inputProcessor.promptText(for: messages).count,
            promptTokens: lmInput.text.tokens.size,
            generatedTokens: tokens.count,
            elapsedSeconds: Date().timeIntervalSince(started),
            outputCharacters: text.count,
            generationInfo: info,
            output: text
        )
    }
}

public struct MotifMLXPerplexityResult: Codable, Equatable, Sendable {
    public var perplexity: Double
    public var nllPerToken: Double
    public var tokens: Int
    public var chunks: Int
    public var elapsedSeconds: Double
    public var tokensPerSecond: Double
}

public struct MotifMLXLogitEntry: Codable, Equatable, Sendable {
    public var token: Int
    public var logit: Double
}

public struct MotifMLXLogitSnapshot: Codable, Equatable, Sendable {
    public var promptTokens: Int
    public var vocabularySize: Int
    public var checksum: Double
    public var topK: [MotifMLXLogitEntry]
    public var elapsedSeconds: Double
}

public struct MotifMLXGenerationInfo: Codable, Equatable, Sendable {
    public var promptTokenCount: Int
    public var generationTokenCount: Int
    public var promptTime: Double
    public var generateTime: Double
    public var promptTokensPerSecond: Double
    public var tokensPerSecond: Double
}

public struct MotifMLXGenerationBenchmark: Codable, Equatable, Sendable {
    public var promptCharacters: Int
    public var promptTokens: Int
    public var generatedTokens: Int
    public var elapsedSeconds: Double
    public var outputCharacters: Int
    public var generationInfo: MotifMLXGenerationInfo?
    public var output: String
}
#endif
