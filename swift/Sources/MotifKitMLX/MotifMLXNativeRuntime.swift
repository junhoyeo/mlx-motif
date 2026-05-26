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
        let tokenIDs: [Int]
        if let chatTemplate {
            tokenIDs = try tokenizer.applyChatTemplate(
                messages: messages,
                chatTemplate: .literal(chatTemplate),
                addGenerationPrompt: true,
                truncation: false,
                maxLength: nil,
                tools: input.tools
            )
        } else {
            tokenIDs = try tokenizer.applyChatTemplate(
                messages: messages,
                chatTemplate: nil,
                addGenerationPrompt: true,
                truncation: false,
                maxLength: nil,
                tools: input.tools
            )
        }
        return LMInput(tokens: MLXArray(tokenIDs))
    }

    public func promptText(for messages: [MotifChatMessage]) throws -> String {
        let generated = messageGenerator.generate(messages: messages.map(Self.chatMessage(from:)))
        let tokenIDs: [Int]
        if let chatTemplate {
            tokenIDs = try tokenizer.applyChatTemplate(
                messages: generated,
                chatTemplate: .literal(chatTemplate),
                addGenerationPrompt: true,
                truncation: false,
                maxLength: nil,
                tools: nil
            )
        } else {
            tokenIDs = try tokenizer.applyChatTemplate(
                messages: generated,
                chatTemplate: nil,
                addGenerationPrompt: true,
                truncation: false,
                maxLength: nil,
                tools: nil
            )
        }
        return tokenizer.decode(tokens: tokenIDs)
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
        let baseConfiguration = try JSONDecoder().decode(
            BaseConfiguration.self,
            from: Data(contentsOf: modelDirectory.appendingPathComponent(MotifModelBundle.configFileName))
        )
        try loadWeights(
            modelDirectory: modelDirectory,
            model: model,
            quantization: nil,
            perLayerQuantization: baseConfiguration.perLayerQuantization
        )

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
        let tokenizer = try await loadTokenizer(configuration: modelConfiguration, hub: defaultHubApi)
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
                    let stream = try generate(
                        input: lmInput,
                        parameters: generationParameters,
                        context: generationContext
                    )
                    for await generation in stream {
                        switch generation {
                        case .chunk(let text):
                            let visible = thinkFilter.feed(text)
                            if !visible.isEmpty {
                                continuation.yield(.text(visible))
                            }
                        case .info:
                            let tail = thinkFilter.finish()
                            if !tail.isEmpty {
                                continuation.yield(.text(tail))
                            }
                            if parameters.thinkMode == .captured, !thinkFilter.capturedReasoning.isEmpty {
                                continuation.yield(.reasoning(thinkFilter.capturedReasoning))
                            }
                            continuation.yield(.completed)
                        case .toolCall:
                            break
                        }
                    }
                    continuation.finish()
                } catch {
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
