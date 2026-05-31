#if canImport(MLX) && canImport(MLXNN) && canImport(MLXLLM) && canImport(MLXLMCommon)
import Foundation
import MLX
import MLXLMCommon
import MotifKit

public struct MotifSpeculativeDecodingParameters: Codable, Equatable, Sendable {
    public var draftTokens: Int
    public var includeRawTokens: Bool

    public init(draftTokens: Int = 4, includeRawTokens: Bool = false) {
        self.draftTokens = max(1, draftTokens)
        self.includeRawTokens = includeRawTokens
    }
}

public struct MotifSpeculativeDecodingMetrics: Codable, Equatable, Sendable {
    public var targetTokens: Int
    public var proposedDraftTokens: Int
    public var acceptedDraftTokens: Int
    public var rejectedDraftTokens: Int
    public var targetModelSteps: Int
    public var draftModelRuns: Int
    public var elapsedSeconds: Double

    public var acceptanceRate: Double {
        guard proposedDraftTokens > 0 else { return 0 }
        return Double(acceptedDraftTokens) / Double(proposedDraftTokens)
    }

    public var tokensPerSecond: Double {
        guard elapsedSeconds > 0 else { return 0 }
        return Double(targetTokens) / elapsedSeconds
    }
}

public struct MotifSpeculativeGenerationResult: Codable, Equatable, Sendable {
    public var text: String
    public var tokens: [Int]?
    public var metrics: MotifSpeculativeDecodingMetrics
}

extension MotifMLXNativeRuntime {
    public func speculativeGenerate(
        messages: [MotifChatMessage],
        draftRuntime: MotifMLXNativeRuntime,
        parameters: MotifGenerationParameters,
        speculativeParameters: MotifSpeculativeDecodingParameters = .init()
    ) async throws -> MotifSpeculativeGenerationResult {
        let started = Date()
        let input = MotifMLXChatInputProcessor.userInput(from: messages)
        let targetInput = try await inputProcessor.prepare(input: input)
        let promptTokens = targetInput.text.tokens.asArray(Int.self)
        let stopTokenIDs = speculativeStopTokenIDs()
        let generationParameters = GenerateParameters(
            maxTokens: parameters.maxTokens,
            temperature: Float(parameters.temperature)
        )
        let targetStream = try generateTokens(
            input: targetInput,
            parameters: generationParameters,
            context: generationContext,
            includeStopToken: true
        )
        var targetIterator = targetStream.makeAsyncIterator()
        var generated: [Int] = []
        var proposedDraftTokens = 0
        var acceptedDraftTokens = 0
        var rejectedDraftTokens = 0
        var targetModelSteps = 0
        var draftModelRuns = 0

        func nextTargetToken() async -> Int? {
            while let event = await targetIterator.next() {
                switch event {
                case .token(let token):
                    return token
                case .info:
                    return nil
                }
            }
            return nil
        }

        while generated.count < parameters.maxTokens {
            let remaining = parameters.maxTokens - generated.count
            let proposalCount = min(speculativeParameters.draftTokens, remaining)
            let proposals = try await draftRuntime.proposeDraftTokens(
                prefixTokens: promptTokens + generated,
                maxTokens: proposalCount,
                temperature: Float(parameters.temperature)
            )
            draftModelRuns += 1
            proposedDraftTokens += proposals.count

            if proposals.isEmpty {
                guard let target = await nextTargetToken() else { break }
                targetModelSteps += 1
                if stopTokenIDs.contains(target) { break }
                generated.append(target)
                continue
            }

            var rejected = false
            for proposal in proposals {
                guard generated.count < parameters.maxTokens, let target = await nextTargetToken() else {
                    rejected = true
                    break
                }
                targetModelSteps += 1
                if stopTokenIDs.contains(target) { rejected = true; break }
                if target == proposal {
                    acceptedDraftTokens += 1
                    generated.append(proposal)
                } else {
                    rejectedDraftTokens += 1
                    generated.append(target)
                    rejected = true
                    break
                }
            }

            if !rejected, generated.count < parameters.maxTokens {
                // In classic speculative decoding the target model also supplies
                // one bonus token after accepting the whole draft block. Pull it
                // here so metrics distinguish all-accepted cycles from stalls.
                if let target = await nextTargetToken() {
                    targetModelSteps += 1
                    if stopTokenIDs.contains(target) { break }
                    generated.append(target)
                }
            }
        }

        let elapsed = Date().timeIntervalSince(started)
        let text = generationContext.tokenizer.decode(tokens: generated)
        return MotifSpeculativeGenerationResult(
            text: text,
            tokens: speculativeParameters.includeRawTokens ? generated : nil,
            metrics: MotifSpeculativeDecodingMetrics(
                targetTokens: generated.count,
                proposedDraftTokens: proposedDraftTokens,
                acceptedDraftTokens: acceptedDraftTokens,
                rejectedDraftTokens: rejectedDraftTokens,
                targetModelSteps: targetModelSteps,
                draftModelRuns: draftModelRuns,
                elapsedSeconds: elapsed
            )
        )
    }

    private func proposeDraftTokens(
        prefixTokens: [Int],
        maxTokens: Int,
        temperature: Float
    ) async throws -> [Int] {
        guard maxTokens > 0 else { return [] }
        let input = LMInput(tokens: MLXArray(prefixTokens, [prefixTokens.count]))
        let stream = try generateTokens(
            input: input,
            parameters: GenerateParameters(maxTokens: maxTokens, temperature: temperature),
            context: generationContext,
            includeStopToken: false
        )
        var tokens: [Int] = []
        for await event in stream {
            switch event {
            case .token(let token):
                tokens.append(token)
                if tokens.count >= maxTokens { return tokens }
            case .info:
                return tokens
            }
        }
        return tokens
    }

    private func speculativeStopTokenIDs() -> Set<Int> {
        var ids = modelConfiguration.eosTokenIds
        if let eos = generationContext.tokenizer.eosTokenId { ids.insert(eos) }
        if let unknown = generationContext.tokenizer.unknownTokenId { ids.insert(unknown) }
        return ids
    }
}
#endif
