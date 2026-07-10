#if canImport(MLX) && canImport(MLXNN) && canImport(MLXLLM) && canImport(MLXLMCommon)
import Foundation
import MLX
import MLXLMCommon
import MLXRandom
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
    /// Retained for API/source stability. Temperature > 0 is now supported via
    /// lossless rejection sampling (see `MotifSpeculativeEngine.decodeSampling`),
    /// so this is no longer thrown by `speculativeGenerate`.
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

/// Speculative decoding with batched target verification.
///
/// Per cycle the draft model proposes `k` tokens autoregressively from its own
/// persistent KV cache, then the target model verifies the whole block with a
/// single batched `[1, k + 1]` forward against its persistent KV cache (the
/// `+ 1` is the last already-emitted token, whose row yields the target's
/// prediction for the first proposal position). Accepted proposals plus one
/// correction/bonus token are emitted; rows past the accepted prefix are
/// trimmed from both caches.
///
/// Two lossless accept rules, selected by `temperature`:
///
/// * **Greedy (temperature == 0):** a draft token is accepted iff it equals the
///   target argmax at its position, so the emitted sequence is exactly the
///   target model's greedy continuation regardless of the draft model.
/// * **Sampling (temperature > 0):** standard rejection sampling
///   (Leviathan et al. 2023 / Chen et al. 2023). The draft proposes token `x`
///   with probability `q(x)`; it is accepted with probability `min(1, p(x)/q(x))`
///   where `p` is the target distribution; on rejection the correction token is
///   drawn from the normalized residual `max(0, p - q)`, and on full acceptance
///   a bonus token is drawn from `p`. The emitted distribution equals plain
///   target sampling exactly.
///
/// In both cases the draft only changes how many target forwards it takes to
/// produce the output, never the output's distribution. Sampling draws use the
/// MLX global RNG, so `MLXRandom.seed(_:)` makes a run reproducible.
enum MotifSpeculativeEngine {
    static func decode(
        targetModel: any LanguageModel,
        draftModel: any LanguageModel,
        targetCache: [KVCache],
        draftCache: [KVCache],
        promptTokens: [Int],
        maxTokens: Int,
        draftTokenCount: Int,
        stopTokenIDs: Set<Int>,
        temperature: Float = 0
    ) throws -> MotifSpeculativeEngineOutcome {
        guard !promptTokens.isEmpty else {
            throw MotifSpeculativeDecodingError.emptyPrompt
        }
        guard targetCache.allSatisfy(\.isTrimmable), draftCache.allSatisfy(\.isTrimmable) else {
            throw MotifSpeculativeDecodingError.cacheNotTrimmable
        }

        if temperature > 0 {
            return decodeSampling(
                targetModel: targetModel,
                draftModel: draftModel,
                targetCache: targetCache,
                draftCache: draftCache,
                promptTokens: promptTokens,
                maxTokens: maxTokens,
                draftTokenCount: draftTokenCount,
                stopTokenIDs: stopTokenIDs,
                temperature: temperature
            )
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

    /// Lossless speculative SAMPLING via rejection sampling (temperature > 0).
    ///
    /// Cache mechanics (proposal, batched verify, per-cycle trim to
    /// `context[0 ..< count - 1]`) are identical to the greedy path — only the
    /// token selection and the accept rule differ. All sampling draws go through
    /// the MLX global RNG (`MLXRandom.uniform`), so seeding with
    /// `MLXRandom.seed(_:)` makes the whole run reproducible.
    private static func decodeSampling(
        targetModel: any LanguageModel,
        draftModel: any LanguageModel,
        targetCache: [KVCache],
        draftCache: [KVCache],
        promptTokens: [Int],
        maxTokens: Int,
        draftTokenCount: Int,
        stopTokenIDs: Set<Int>,
        temperature: Float
    ) -> MotifSpeculativeEngineOutcome {
        var outcome = MotifSpeculativeEngineOutcome()
        guard maxTokens > 0 else { return outcome }
        let draftTokenCount = max(1, draftTokenCount)
        let inverseTemperature = 1 / temperature

        /// Feed `tokens` through `model` (appending to `cache`) and return the
        /// logits of the last `lastCount` positions, shape `[lastCount, vocab]`.
        func forwardTail(
            _ model: any LanguageModel, tokens: [Int], cache: [KVCache], lastCount: Int
        ) -> MLXArray {
            let logits = model(MLXArray(tokens.map(Int32.init), [1, tokens.count]), cache: cache)
            let length = logits.dim(1)
            return logits[0, (length - lastCount) ..< length, 0...]
        }

        /// Temperature-scaled softmax of one `[vocab]` logits row.
        func probabilities(_ logitsRow: MLXArray) -> MLXArray {
            softmax(logitsRow * inverseTemperature, axis: -1)
        }

        /// Inverse-CDF sample of one index from a normalized `[vocab]`
        /// probability row, using a single global-RNG uniform draw (so the whole
        /// decode is reproducible under `MLXRandom.seed`).
        func sampleIndex(_ probabilities: MLXArray) -> Int {
            let u = MLXRandom.uniform(Float(0) ..< Float(1), [1]).item(Float.self)
            let cdf = cumsum(probabilities, axis: -1)
            // Number of CDF entries <= u is the index of the first entry > u.
            let index = sum((cdf .<= u).asType(.int32)).item(Int.self)
            return min(max(index, 0), probabilities.dim(0) - 1)
        }

        func trim(_ caches: [KVCache], by amount: Int) {
            guard amount > 0 else { return }
            for cache in caches {
                cache.trim(amount)
            }
        }

        var context = promptTokens
        var targetCacheCount = 0
        var draftCacheCount = 0

        while outcome.tokens.count < maxTokens {
            let remaining = maxTokens - outcome.tokens.count
            let k = min(draftTokenCount, remaining - 1)

            // 1. Draft proposes k tokens autoregressively, SAMPLING each from its
            //    own distribution q_i and recording q_i (needed for the accept
            //    test and the residual on rejection).
            var proposals: [Int] = []
            var proposalProbability: [Float] = []  // q_i(x_i)
            var draftDistributions: [MLXArray] = []  // full q_i, for the residual
            if k > 0 {
                let missing = Array(context[draftCacheCount...])
                var qRow = probabilities(forwardTail(draftModel, tokens: missing, cache: draftCache, lastCount: 1)[0])
                draftCacheCount = context.count
                var next = sampleIndex(qRow)
                proposals.append(next)
                draftDistributions.append(qRow)
                proposalProbability.append(qRow[next].item(Float.self))
                while proposals.count < k {
                    qRow = probabilities(forwardTail(draftModel, tokens: [next], cache: draftCache, lastCount: 1)[0])
                    draftCacheCount += 1
                    next = sampleIndex(qRow)
                    proposals.append(next)
                    draftDistributions.append(qRow)
                    proposalProbability.append(qRow[next].item(Float.self))
                }
                outcome.draftModelRuns += 1
                outcome.proposedDraftTokens += k
            }

            // 2. Single batched target forward verifies the block. The last
            //    k + 1 rows are the target distributions p_0 ... p_k.
            let verifyTokens = Array(context[targetCacheCount...]) + proposals
            let targetTail = forwardTail(targetModel, tokens: verifyTokens, cache: targetCache, lastCount: k + 1)
            targetCacheCount = context.count + k
            outcome.targetModelSteps += 1
            var targetDistributions: [MLXArray] = []
            targetDistributions.reserveCapacity(k + 1)
            for position in 0 ... k {
                targetDistributions.append(probabilities(targetTail[position]))
            }

            // 3. Rejection sampling: accept proposal i with prob min(1, p/q); on
            //    reject draw the correction from the normalized residual and stop
            //    the block; on full acceptance draw a bonus token from p_k.
            var accepted = 0
            var correctionToken = 0
            var rejected = false
            while accepted < k {
                let proposal = proposals[accepted]
                let targetProbability = targetDistributions[accepted][proposal].item(Float.self)
                let draftProbability = proposalProbability[accepted]
                let acceptProbability = draftProbability > 0
                    ? min(Float(1), targetProbability / draftProbability)
                    : Float(1)
                let u = MLXRandom.uniform(Float(0) ..< Float(1), [1]).item(Float.self)
                if u < acceptProbability {
                    accepted += 1
                } else {
                    let residual = maximum(targetDistributions[accepted] - draftDistributions[accepted], 0)
                    // Residual mass is > 0 whenever a rejection occurs (p != q).
                    let normalized = residual / sum(residual)
                    correctionToken = sampleIndex(normalized)
                    rejected = true
                    break
                }
            }
            if !rejected {
                // All k proposals accepted (or k == 0): bonus token from p_k.
                correctionToken = sampleIndex(targetDistributions[k])
            }
            outcome.acceptedDraftTokens += accepted
            outcome.rejectedDraftTokens += k - accepted

            // 4. Emit accepted proposals + the correction/bonus token, honoring
            //    stop tokens and the maxTokens budget.
            let previousCount = context.count
            var stopped = false
            let emitted = Array(proposals[0 ..< accepted]) + [correctionToken]
            for token in emitted {
                if stopTokenIDs.contains(token) {
                    stopped = true
                    break
                }
                outcome.tokens.append(token)
                context.append(token)
                if outcome.tokens.count >= maxTokens { break }
            }
            if stopped || outcome.tokens.count >= maxTokens { break }

            // 5. Trim rows past the accepted prefix so both caches again hold
            //    exactly context[0 ..< count - 1] (identical to the greedy path).
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
    /// Two lossless modes, selected by `parameters.temperature`:
    ///
    /// * **Greedy (temperature == 0):** a draft token is accepted iff it equals
    ///   the target argmax at that position, so the output is the target model's
    ///   own greedy continuation.
    /// * **Sampling (temperature > 0):** lossless rejection sampling
    ///   (Leviathan/Chen) — the emitted distribution equals plain target
    ///   sampling exactly. Draws use the MLX global RNG, so `MLXRandom.seed(_:)`
    ///   makes a run reproducible.
    ///
    /// In both modes the draft only reduces the number of target forwards, never
    /// changes the output's distribution. The draft runtime must share the
    /// target's tokenizer and vocabulary — proposals are compared as token IDs.
    ///
    /// Exactness caveat: verification uses batched forwards while plain
    /// generation uses `[1, 1]` decode steps. On quantized (q4/q8) checkpoints
    /// the two can round differently on near-tie logits, so the output is a
    /// valid decode but not guaranteed byte-identical to the sequential path
    /// (same property documented for KV-cache reuse in
    /// `MotifMLXNativeRuntimeCacheReuseTests`). On fp16/fp32 weights the greedy
    /// equivalence is exact in practice and is asserted token-for-token by
    /// `MotifSpeculativeDecodingTests`.
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
        let generationParameters = GenerateParameters(
            maxTokens: parameters.maxTokens,
            temperature: Float(parameters.temperature)
        )
        let outcome = try MotifSpeculativeEngine.decode(
            targetModel: generationContext.model,
            draftModel: draftRuntime.generationContext.model,
            targetCache: generationContext.model.newCache(parameters: generationParameters),
            draftCache: draftRuntime.generationContext.model.newCache(parameters: generationParameters),
            promptTokens: promptTokens,
            maxTokens: parameters.maxTokens,
            draftTokenCount: speculativeParameters.draftTokens,
            stopTokenIDs: speculativeStopTokenIDs(),
            temperature: Float(parameters.temperature)
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
