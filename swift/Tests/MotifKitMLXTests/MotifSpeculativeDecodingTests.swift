#if canImport(MLX) && canImport(MLXNN) && canImport(MLXLLM) && canImport(MLXLMCommon)
import MLX
import MLXLMCommon
import MLXRandom
import MotifKit
@testable import MotifKitMLX
import XCTest

/// Correctness gate for speculative decoding.
///
/// The load-bearing invariant: speculative output token IDs are EXACTLY the
/// plain greedy decode of the target model, regardless of the draft model —
/// the draft only reduces the number of target forward passes. Both properties
/// are asserted here on tiny random-weight grouped-differential models (fp32,
/// so batched-vs-sequential forwards agree bit-for-bit in practice).
///
/// Gated on `MOTIFKIT_RUN_MLX_RUNTIME_TESTS=1` like the other MLX-runtime
/// suites (MLX ops require the default metallib).
final class MotifSpeculativeDecodingTests: XCTestCase {
    private func requireMLXRuntime(file: StaticString = #filePath, line: UInt = #line) throws {
        try XCTSkipUnless(
            ProcessInfo.processInfo.environment["MOTIFKIT_RUN_MLX_RUNTIME_TESTS"] == "1",
            "MLX runtime ops require the default metallib; build-only CI still verifies type-checking.",
            file: file,
            line: line
        )
    }

    /// Same tiny grouped-differential shape used by MotifQKVFusionParityTests.
    private func tinyConfiguration() -> MotifModelConfiguration {
        MotifModelConfiguration(
            hiddenSize: 128,
            numHiddenLayers: 2,
            intermediateSize: 256,
            numAttentionHeads: 12,
            numKeyValueHeads: 4,
            vocabSize: 64,
            headDim: 64,
            numNoiseHeads: 4,
            kRatio: 1,
            hiddenActivation: "poly_norm"
        )
    }

    private func makeModel(seed: UInt64) throws -> MotifMLXModel {
        MLXRandom.seed(seed)
        let model = try MotifMLXModel(
            configuration: tinyConfiguration(),
            runtimeFeatures: MotifRuntimeFeatureFlags()
        )
        eval(model.parameters())
        return model
    }

    private let prompt = [1, 5, 9, 2, 7, 3, 11, 4]

    /// Plain sequential greedy decode: batched prompt prefill, then one [1, 1]
    /// target forward per token. This is the baseline speculative decoding must
    /// reproduce token-for-token.
    private func plainGreedy(
        model: MotifMLXModel,
        prompt: [Int],
        maxTokens: Int,
        stopTokenIDs: Set<Int> = []
    ) -> [Int] {
        let cache = model.newCache(parameters: GenerateParameters(temperature: 0))
        var logits = model(MLXArray(prompt.map(Int32.init), [1, prompt.count]), cache: cache)
        var out: [Int] = []
        while out.count < maxTokens {
            let token = argMax(logits[0, -1, 0...], axis: -1).item(Int.self)
            if stopTokenIDs.contains(token) { break }
            out.append(token)
            if out.count == maxTokens { break }
            logits = model(MLXArray([Int32(token)], [1, 1]), cache: cache)
        }
        return out
    }

    private func speculative(
        target: MotifMLXModel,
        draft: MotifMLXModel,
        prompt: [Int],
        maxTokens: Int,
        draftTokenCount: Int = 4,
        stopTokenIDs: Set<Int> = []
    ) throws -> MotifSpeculativeEngineOutcome {
        try MotifSpeculativeEngine.decode(
            targetModel: target,
            draftModel: draft,
            targetCache: target.newCache(parameters: GenerateParameters(temperature: 0)),
            draftCache: draft.newCache(parameters: GenerateParameters(temperature: 0)),
            promptTokens: prompt,
            maxTokens: maxTokens,
            draftTokenCount: draftTokenCount,
            stopTokenIDs: stopTokenIDs
        )
    }

    /// HARD GATE: with the draft sharing the target's weights (perfect drafter),
    /// speculative output must equal plain greedy decode exactly AND the target
    /// must run strictly fewer forward passes than tokens generated — the
    /// structural source of the speculative speedup the old implementation
    /// could never produce.
    func testSelfDraftMatchesPlainGreedyWithFewerTargetForwards() throws {
        try requireMLXRuntime()
        let model = try makeModel(seed: 7)
        let maxTokens = 24

        let baseline = plainGreedy(model: model, prompt: prompt, maxTokens: maxTokens)
        XCTAssertEqual(baseline.count, maxTokens)

        let outcome = try speculative(
            target: model, draft: model, prompt: prompt, maxTokens: maxTokens)

        XCTAssertEqual(
            outcome.tokens, baseline,
            "speculative output must be exactly the target's plain greedy decode")
        XCTAssertLessThan(
            outcome.targetModelSteps, outcome.tokens.count,
            "batched verification must use fewer target forwards than tokens generated")
        XCTAssertGreaterThan(outcome.acceptedDraftTokens, 0)
        print(
            "[speculative gate] self-draft: \(outcome.tokens.count) tokens with "
                + "\(outcome.targetModelSteps) target forwards, "
                + "accepted \(outcome.acceptedDraftTokens)/\(outcome.proposedDraftTokens) drafts")
    }

    /// A draft with unrelated random weights (frequent rejections, trims every
    /// cycle) must still yield exactly the target's plain greedy decode — the
    /// draft can never change the output, only the forward count.
    func testDivergentDraftStillMatchesPlainGreedy() throws {
        try requireMLXRuntime()
        let target = try makeModel(seed: 7)
        let draft = try makeModel(seed: 99)
        let maxTokens = 24

        let baseline = plainGreedy(model: target, prompt: prompt, maxTokens: maxTokens)
        let outcome = try speculative(
            target: target, draft: draft, prompt: prompt, maxTokens: maxTokens)

        XCTAssertEqual(
            outcome.tokens, baseline,
            "a divergent draft must not change the emitted tokens (rejection + trim path)")
        XCTAssertEqual(
            outcome.acceptedDraftTokens + outcome.rejectedDraftTokens,
            outcome.proposedDraftTokens)
        print(
            "[speculative gate] divergent draft: \(outcome.tokens.count) tokens with "
                + "\(outcome.targetModelSteps) target forwards, "
                + "accepted \(outcome.acceptedDraftTokens)/\(outcome.proposedDraftTokens) drafts")
    }

    /// Stop tokens terminate speculative decoding at exactly the same point as
    /// plain greedy decode (stop token not emitted), even when the stop token
    /// is produced mid-draft-block.
    func testStopTokenTerminatesLikePlainGreedy() throws {
        try requireMLXRuntime()
        let model = try makeModel(seed: 7)
        let reference = plainGreedy(model: model, prompt: prompt, maxTokens: 24)
        // Choose a token the greedy continuation actually produces mid-stream
        // and treat it as the stop token.
        let stop = reference[8]
        let stopSet: Set<Int> = [stop]

        let baseline = plainGreedy(
            model: model, prompt: prompt, maxTokens: 24, stopTokenIDs: stopSet)
        XCTAssertFalse(baseline.contains(stop))

        for draftSeed in [UInt64(7), UInt64(99)] {
            let draft = draftSeed == 7 ? model : try makeModel(seed: draftSeed)
            let outcome = try speculative(
                target: model, draft: draft, prompt: prompt, maxTokens: 24,
                stopTokenIDs: stopSet)
            XCTAssertEqual(
                outcome.tokens, baseline,
                "stop-token termination must match plain greedy (draft seed \(draftSeed))")
        }
    }

    /// The maxTokens budget truncates mid-block identically to plain greedy.
    func testMaxTokensTruncatesMidBlock() throws {
        try requireMLXRuntime()
        let model = try makeModel(seed: 7)
        let baseline = plainGreedy(model: model, prompt: prompt, maxTokens: 24)

        for budget in [1, 2, 5, 7] {
            let outcome = try speculative(
                target: model, draft: model, prompt: prompt, maxTokens: budget)
            XCTAssertEqual(
                outcome.tokens, Array(baseline.prefix(budget)),
                "maxTokens=\(budget) must emit exactly the first \(budget) greedy tokens")
        }
    }

    /// The empty-prompt guard still throws (shared by both greedy and sampling).
    func testEngineRejectsEmptyPromptAndGuardsAreThrown() throws {
        try requireMLXRuntime()
        let model = try makeModel(seed: 7)
        XCTAssertThrowsError(
            try speculative(target: model, draft: model, prompt: [], maxTokens: 4)
        ) { error in
            XCTAssertEqual(error as? MotifSpeculativeDecodingError, .emptyPrompt)
        }
    }

    // MARK: - Speculative sampling (temperature > 0)

    private func speculativeSampling(
        target: MotifMLXModel,
        draft: MotifMLXModel,
        prompt: [Int],
        maxTokens: Int,
        temperature: Float,
        draftTokenCount: Int = 4,
        stopTokenIDs: Set<Int> = []
    ) throws -> MotifSpeculativeEngineOutcome {
        try MotifSpeculativeEngine.decode(
            targetModel: target,
            draftModel: draft,
            targetCache: target.newCache(parameters: GenerateParameters(temperature: temperature)),
            draftCache: draft.newCache(parameters: GenerateParameters(temperature: temperature)),
            promptTokens: prompt,
            maxTokens: maxTokens,
            draftTokenCount: draftTokenCount,
            stopTokenIDs: stopTokenIDs,
            temperature: temperature
        )
    }

    /// (i) Temperature > 0 no longer throws — rejection sampling is implemented.
    func testSamplingTemperatureDoesNotThrow() throws {
        try requireMLXRuntime()
        let model = try makeModel(seed: 7)
        MLXRandom.seed(42)
        XCTAssertNoThrow(
            try speculativeSampling(
                target: model, draft: model, prompt: prompt, maxTokens: 8, temperature: 0.8)
        )
        let outcome = try speculativeSampling(
            target: model, draft: model, prompt: prompt, maxTokens: 8, temperature: 0.8)
        XCTAssertEqual(outcome.tokens.count, 8, "sampling must still honor the maxTokens budget")
    }

    func testNearZeroResidualFallsBackToTargetArgmax() throws {
        try requireMLXRuntime()

        let target = MLXArray([Float(0.1), 0.2, 0.6, 0.1])
        let nearTieDraft = MLXArray([Float(0.1) - Float(1e-8), 0.2, 0.6, 0.1])
        var nearTieSamplerCalled = false
        let fallbackToken = MotifSpeculativeEngine.residualCorrectionToken(
            target: target,
            draft: nearTieDraft,
            sampleIndex: { _ in
                nearTieSamplerCalled = true
                return 0
            }
        )

        XCTAssertEqual(fallbackToken, 2)
        XCTAssertFalse(nearTieSamplerCalled, "near-zero residuals must not be normalized and sampled")

        let ordinaryDraft = MLXArray([Float(0.05), 0.25, 0.6, 0.1])
        var ordinarySamplerCalled = false
        let sampledToken = MotifSpeculativeEngine.residualCorrectionToken(
            target: target,
            draft: ordinaryDraft,
            sampleIndex: { _ in
                ordinarySamplerCalled = true
                return 3
            }
        )

        XCTAssertEqual(sampledToken, 3)
        XCTAssertTrue(ordinarySamplerCalled, "meaningful residuals must use the sampler")
    }

    /// (ii) Self-draft (draft == target) means q == p at every position, so the
    /// accept probability min(1, p/q) is 1 and essentially every proposal is
    /// accepted. Assert acceptance rate > 0.95 over a run.
    func testSelfDraftSamplingAcceptanceRateNearOne() throws {
        try requireMLXRuntime()
        let model = try makeModel(seed: 7)
        MLXRandom.seed(123)
        let outcome = try speculativeSampling(
            target: model, draft: model, prompt: prompt, maxTokens: 48, temperature: 0.8)
        XCTAssertGreaterThan(outcome.proposedDraftTokens, 0)
        let acceptanceRate = Double(outcome.acceptedDraftTokens) / Double(outcome.proposedDraftTokens)
        XCTAssertGreaterThan(
            acceptanceRate, 0.95,
            "self-draft rejection sampling (p == q) must accept nearly all proposals; "
                + "got \(outcome.acceptedDraftTokens)/\(outcome.proposedDraftTokens)")
        print(
            "[speculative sampling gate] self-draft acceptance "
                + "\(outcome.acceptedDraftTokens)/\(outcome.proposedDraftTokens) = \(acceptanceRate)")
    }

    /// (iii) Determinism: the same MLXRandom seed reproduces the sampled output
    /// exactly. Runs the decode twice with the same seed and asserts equality.
    func testSeededSamplingIsDeterministic() throws {
        try requireMLXRuntime()
        let target = try makeModel(seed: 7)
        let draft = try makeModel(seed: 99)

        func run() throws -> [Int] {
            MLXRandom.seed(2024)
            return try speculativeSampling(
                target: target, draft: draft, prompt: prompt, maxTokens: 24, temperature: 0.9
            ).tokens
        }

        let first = try run()
        let second = try run()
        XCTAssertFalse(first.isEmpty)
        XCTAssertEqual(
            first, second,
            "the same MLXRandom seed must reproduce the sampled speculative output exactly")
    }
}
#endif
