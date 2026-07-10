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

public enum MotifSpeculativeDecodingError: Error, LocalizedError, Equatable {
    /// Lossless speculative sampling at temperature > 0 requires rejection
    /// sampling against the full target distribution, which is not implemented.
    /// Only greedy (temperature == 0) decoding is supported.
    case greedyOnly(Double)
    /// Rejected draft tokens must be trimmed out of the KV caches, so both the
    /// target and draft caches must be trimmable.
    case cacheNotTrimmable
    case emptyPrompt

    public var errorDescription: String? {
        switch self {
        case .greedyOnly(let temperature):
            "Speculative decoding supports greedy decoding only (temperature 0); got \(temperature). "
                + "Lossless sampling at temperature > 0 needs rejection sampling, which is not implemented."
        case .cacheNotTrimmable:
            "Speculative decoding requires trimmable KV caches (rejected draft rows are trimmed in place)."
        case .emptyPrompt:
            "Speculative decoding requires a non-empty prompt."
        }
    }
}

public struct MotifSpeculativeDecodingMetrics: Codable, Equatable, Sendable {
    /// Tokens emitted by the decoder (the target model's greedy continuation).
    public var targetTokens: Int
    public var proposedDraftTokens: Int
    public var acceptedDraftTokens: Int
    public var rejectedDraftTokens: Int
    /// Number of target-model forward passes (the initial prefill-and-verify
    /// plus one batched verification per cycle). Strictly less than
    /// `targetTokens` whenever any draft token is accepted — this is the
    /// mechanism of the speculative speedup.
    public var targetModelSteps: Int
    /// Number of draft proposal cycles.
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

struct MotifSpeculativeEngineOutcome {
    var tokens: [Int] = []
    var proposedDraftTokens = 0
    var acceptedDraftTokens = 0
    var rejectedDraftTokens = 0
    var targetModelSteps = 0
    var draftModelRuns = 0
}

/// Greedy speculative decoding with batched target verification.
///
/// Per cycle the draft model proposes `k` tokens autoregressively from its own
/// persistent KV cache, then the target model verifies the whole block with a
/// single batched `[1, k + 1]` forward against its persistent KV cache (the
/// `+ 1` is the last already-emitted token, whose row yields the target's
/// prediction for the first proposal position). The longest prefix of
/// proposals matching the target argmaxes is accepted, plus the target's own
/// next token (correction on mismatch, bonus on full acceptance). Rows past
/// the accepted prefix are trimmed from both caches.
///
/// Greedy accept rule: a draft token is accepted iff it equals the target
/// argmax at its position, so the emitted sequence is exactly the target
/// model's greedy continuation regardless of the draft model — the draft only
/// changes how many target forwards it takes to produce it.
enum MotifSpeculativeEngine {
    static func decode(
        targetModel: any LanguageModel,
        draftModel: any LanguageModel,
        targetCache: [KVCache],
        draftCache: [KVCache],
        promptTokens: [Int],
        maxTokens: Int,
        draftTokenCount: Int,
        stopTokenIDs: Set<Int>
    ) throws -> MotifSpeculativeEngineOutcome {
        guard !promptTokens.isEmpty else {
            throw MotifSpeculativeDecodingError.emptyPrompt
        }
        guard targetCache.allSatisfy(\.isTrimmable), draftCache.allSatisfy(\.isTrimmable) else {
            throw MotifSpeculativeDecodingError.cacheNotTrimmable
        }

        var outcome = MotifSpeculativeEngineOutcome()
        guard maxTokens > 0 else { return outcome }
        let draftTokenCount = max(1, draftTokenCount)

        /// Feed `tokens` through `model` (appending to `cache`) and return the
        /// argmax token IDs of the last `lastCount` positions.
        func greedyForward(
            _ model: any LanguageModel, tokens: [Int], cache: [KVCache], lastCount: Int
        ) -> [Int] {
            let logits = model(MLXArray(tokens.map(Int32.init), [1, tokens.count]), cache: cache)
            let length = logits.dim(1)
            let tail = logits[0, (length - lastCount) ..< length, 0...]
            return argMax(tail, axis: -1).asArray(Int32.self).map(Int.init)
        }

        func trim(_ caches: [KVCache], by amount: Int) {
            guard amount > 0 else { return }
            for cache in caches {
                cache.trim(amount)
            }
        }

        // Loop invariants (checked at the top of each iteration):
        // - `context` == promptTokens + emitted tokens.
        // - `targetCache` holds KV for exactly `context[0 ..< targetCacheCount]`
        //   with targetCacheCount <= context.count - 1 (the final context token
        //   is pending; its logits come from the next verification forward).
        // - `draftCache` holds KV for exactly `context[0 ..< draftCacheCount]`
        //   with draftCacheCount <= context.count - 1.
        var context = promptTokens
        var targetCacheCount = 0
        var draftCacheCount = 0

        while outcome.tokens.count < maxTokens {
            let remaining = maxTokens - outcome.tokens.count
            // Proposing more than remaining - 1 is wasted work: the cycle emits
            // at most accepted + 1 tokens. remaining == 1 degenerates to a plain
            // greedy step (k == 0, verify forward of just the pending token).
            let k = min(draftTokenCount, remaining - 1)

            // 1. Draft proposes k tokens greedily from its persistent cache.
            //    The first forward feeds whatever context suffix the draft cache
            //    is missing (at minimum the last emitted token); the remaining
            //    k - 1 proposals are single-token decode steps.
            var proposals: [Int] = []
            if k > 0 {
                let missing = Array(context[draftCacheCount...])
                var next = greedyForward(draftModel, tokens: missing, cache: draftCache, lastCount: 1)[0]
                draftCacheCount = context.count
                proposals.append(next)
                while proposals.count < k {
                    next = greedyForward(draftModel, tokens: [next], cache: draftCache, lastCount: 1)[0]
                    draftCacheCount += 1
                    proposals.append(next)
                }
                outcome.draftModelRuns += 1
                outcome.proposedDraftTokens += k
            }

            // 2. Single batched target forward verifies the whole block: input
            //    is the pending context suffix plus all k proposals ([1, k + 1]
            //    in steady state; the first cycle folds the prompt prefill in).
            //    The last k + 1 rows are the target argmaxes t_0 ... t_k.
            let verifyTokens = Array(context[targetCacheCount...]) + proposals
            let predictions = greedyForward(
                targetModel, tokens: verifyTokens, cache: targetCache, lastCount: k + 1)
            targetCacheCount = context.count + k
            outcome.targetModelSteps += 1

            // 3. Accept the longest matching prefix; emit it plus the target's
            //    next token (correction or bonus), honoring stop tokens and the
            //    maxTokens budget.
            var accepted = 0
            while accepted < k, proposals[accepted] == predictions[accepted] {
                accepted += 1
            }
            outcome.acceptedDraftTokens += accepted
            outcome.rejectedDraftTokens += k - accepted

            let previousCount = context.count
            var stopped = false
            for prediction in predictions[0 ... accepted] {
                if stopTokenIDs.contains(prediction) {
                    stopped = true
                    break
                }
                outcome.tokens.append(prediction)
                context.append(prediction)
                if outcome.tokens.count >= maxTokens { break }
            }
            if stopped || outcome.tokens.count >= maxTokens { break }

            // 4. The loop continues, so the full block (accepted + 1 tokens)
            //    was emitted. Trim rows past the accepted prefix so both caches
            //    again hold exactly context[0 ..< count - 1]. (k >= 1 here:
            //    k == 0 implies remaining == 1, which always exits above.)
            let targetValid = context.count - 1  // == previousCount + accepted
            trim(targetCache, by: targetCacheCount - targetValid)
            targetCacheCount = targetValid
            let draftValid = previousCount + min(accepted, k - 1)
            trim(draftCache, by: draftCacheCount - draftValid)
            draftCacheCount = draftValid
        }

        return outcome
    }
}

extension MotifMLXNativeRuntime {
    /// Speculative decoding: `draftRuntime` proposes
    /// `speculativeParameters.draftTokens` tokens per cycle from a persistent
    /// draft KV cache, and the target model verifies each block with a single
    /// batched `[1, k + 1]` forward against a persistent target KV cache,
    /// trimming rejected rows from both caches. Accepted-block cycles advance
    /// the output by up to `draftTokens + 1` tokens per target forward.
    ///
    /// Greedy only: throws ``MotifSpeculativeDecodingError/greedyOnly(_:)``
    /// when `parameters.temperature != 0`. The emitted tokens follow the greedy
    /// accept rule (a draft token is accepted iff it equals the target argmax
    /// at that position), so the output is the target model's own greedy
    /// continuation; the draft model only reduces the number of target
    /// forwards. The draft runtime must share the target's tokenizer and
    /// vocabulary — proposals are compared as token IDs.
    ///
    /// Exactness caveat: verification uses batched forwards while plain
    /// generation uses `[1, 1]` decode steps. On quantized (q4/q8) checkpoints
    /// the two can round differently on near-tie logits, so the output is a
    /// valid greedy decode but not guaranteed byte-identical to the sequential
    /// path (same property documented for KV-cache reuse in
    /// `MotifMLXNativeRuntimeCacheReuseTests`). On fp16/fp32 weights the
    /// equivalence is exact in practice and is asserted token-for-token by
    /// `MotifSpeculativeDecodingTests`.
    public func speculativeGenerate(
        messages: [MotifChatMessage],
        draftRuntime: MotifMLXNativeRuntime,
        parameters: MotifGenerationParameters,
        speculativeParameters: MotifSpeculativeDecodingParameters = .init()
    ) async throws -> MotifSpeculativeGenerationResult {
        guard parameters.temperature == 0 else {
            throw MotifSpeculativeDecodingError.greedyOnly(parameters.temperature)
        }
        let started = Date()
        let input = MotifMLXChatInputProcessor.userInput(from: messages)
        let targetInput = try await inputProcessor.prepare(input: input)
        let promptTokens = targetInput.text.tokens.asArray(Int.self)
        let generationParameters = GenerateParameters(
            maxTokens: parameters.maxTokens,
            temperature: 0
        )
        let outcome = try MotifSpeculativeEngine.decode(
            targetModel: generationContext.model,
            draftModel: draftRuntime.generationContext.model,
            targetCache: generationContext.model.newCache(parameters: generationParameters),
            draftCache: draftRuntime.generationContext.model.newCache(parameters: generationParameters),
            promptTokens: promptTokens,
            maxTokens: parameters.maxTokens,
            draftTokenCount: speculativeParameters.draftTokens,
            stopTokenIDs: speculativeStopTokenIDs()
        )
        let elapsed = Date().timeIntervalSince(started)
        let text = generationContext.tokenizer.decode(tokens: outcome.tokens)
        return MotifSpeculativeGenerationResult(
            text: text,
            tokens: speculativeParameters.includeRawTokens ? outcome.tokens : nil,
            metrics: MotifSpeculativeDecodingMetrics(
                targetTokens: outcome.tokens.count,
                proposedDraftTokens: outcome.proposedDraftTokens,
                acceptedDraftTokens: outcome.acceptedDraftTokens,
                rejectedDraftTokens: outcome.rejectedDraftTokens,
                targetModelSteps: outcome.targetModelSteps,
                draftModelRuns: outcome.draftModelRuns,
                elapsedSeconds: elapsed
            )
        )
    }

    private func speculativeStopTokenIDs() -> Set<Int> {
        var ids = modelConfiguration.eosTokenIds
        if let eos = generationContext.tokenizer.eosTokenId { ids.insert(eos) }
        if let unknown = generationContext.tokenizer.unknownTokenId { ids.insert(unknown) }
        return ids
    }
}
#endif
