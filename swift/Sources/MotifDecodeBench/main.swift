#if canImport(MotifKitMLX)
import Foundation
import MLX
import MLXNN
import MotifKit
import MotifKitMLX

// MotifDecodeBench
// ----------------
// Synthetic micro-benchmark for the Swift grouped-differential-attention q4
// DECODE path. There are NO model checkpoints in this environment, so this
// harness builds the real Motif decoder graph with RANDOM weights at a real
// per-layer config shape, quantizes the projections to q4 (group size 64),
// prefills a q4 grouped KV cache to a chosen context length, then times the
// single-token (sequenceLength == 1) decode step.
//
// HONESTY: every number printed here is a synthetic-weights, per-decode-step
// micro-benchmark on ONE machine. It is NOT an end-to-end tok/s measurement.
// End-to-end validation on real checkpoints is still required.

struct BenchConfig {
    let name: String
    let hiddenSize: Int
    let numHiddenLayers: Int
    let intermediateSize: Int
    let numAttentionHeads: Int
    let numKeyValueHeads: Int
    let headDim: Int
    let numNoiseHeads: Int
    let kRatio: Int
    let vocabSize: Int

    func modelConfiguration() -> MotifModelConfiguration {
        MotifModelConfiguration(
            hiddenSize: hiddenSize,
            numHiddenLayers: numHiddenLayers,
            intermediateSize: intermediateSize,
            numAttentionHeads: numAttentionHeads,
            numKeyValueHeads: numKeyValueHeads,
            vocabSize: vocabSize,
            headDim: headDim,
            numNoiseHeads: numNoiseHeads,
            kRatio: kRatio,
            hiddenActivation: "poly_norm"
        )
    }
}

// Real Motif per-layer config shapes. numHiddenLayers is intentionally reduced
// so the synthetic graph fits in memory and so the benchmark isolates the
// per-layer decode cost (the per-step cost scales ~linearly with layer count).
// Per-layer attention/MLP dims match the published Motif layouts.
let configs: [BenchConfig] = [
    // Motif 2-12.7B-Reasoning per-layer layout:
    //   hidden 4096, 40 q heads, 16 kv heads, headDim 128, 8 noise heads, k_ratio 1
    //   intermediate 16384, grouped differential attention.
    BenchConfig(
        name: "12.7B-perlayer",
        hiddenSize: 4096,
        numHiddenLayers: 4,
        intermediateSize: 16384,
        numAttentionHeads: 40,
        numKeyValueHeads: 16,
        headDim: 128,
        numNoiseHeads: 8,
        kRatio: 1,
        vocabSize: 4096
    ),
    // Motif 2-2.6B-ish grouped per-layer layout (smaller hidden/heads).
    BenchConfig(
        name: "2.6B-perlayer",
        hiddenSize: 2048,
        numHiddenLayers: 4,
        intermediateSize: 8192,
        numAttentionHeads: 20,
        numKeyValueHeads: 8,
        headDim: 128,
        numNoiseHeads: 4,
        kRatio: 1,
        vocabSize: 4096
    ),
]

func arg(_ name: String, default def: String) -> String {
    let args = CommandLine.arguments
    if let i = args.firstIndex(of: name), args.indices.contains(i + 1) { return args[i + 1] }
    return def
}

let warmup = Int(arg("--warmup", default: "10")) ?? 10
let timed = Int(arg("--timed", default: "50")) ?? 50
let contextLengths = arg("--contexts", default: "512,3000")
    .split(separator: ",").compactMap { Int($0) }
let fuseEnv = ProcessInfo.processInfo.environment["MLX_MOTIF_FUSE_QKV"]
let configFilter = arg("--config", default: "all")

func median(_ xs: [Double]) -> Double {
    let s = xs.sorted()
    guard !s.isEmpty else { return 0 }
    let m = s.count / 2
    return s.count % 2 == 0 ? (s[m - 1] + s[m]) / 2 : s[m]
}

// Build a model with random weights at q4, prefill the q4 cache, time decode.
func benchOne(config: BenchConfig, context: Int, fuse: Bool) throws -> (median: Double, min: Double) {
    let modelConfig = config.modelConfiguration()
    let flags = MotifRuntimeFeatureFlags(
        fourSlotCache: "q4",
        fuseQueryKeyValue: fuse
    )
    let model = try MotifMLXModel(configuration: modelConfig, runtimeFeatures: flags)

    // Quantize the projection / MLP / lm_head Linear layers to q4 group size 64,
    // matching the real q4 decode path. Embedding stays as-is. This exercises the
    // QuantizedLinear concatenation in the fused path.
    quantize(model: model, groupSize: 64, bits: 4) { _, module in
        module is Linear || module is Embedding
    }
    eval(model.parameters())

    if fuse {
        let fused = model.fuseQueryKeyValueProjectionsIfPossible()
        precondition(fused == modelConfig.numHiddenLayers, "expected all layers fused, got \(fused)")
    }

    // q4 four-slot cache mode is selected via the feature flags above; newCache
    // defaults kvGroupSize to 64 when parameters is nil.
    let cache = model.newCache(parameters: nil)

    // Prefill: push `context` tokens through the model to populate the q4 cache.
    // Done in chunks to keep prefill memory bounded; correctness of prefill is
    // not what we measure (the runtime parity tests cover that) — we just need a
    // realistically populated cache before timing single-token decode.
    let prefillChunk = 256
    var done = 0
    while done < context {
        let n = min(prefillChunk, context - done)
        let tokens = MLXArray((0 ..< n).map { Int32(($0 + done) % config.vocabSize) }, [1, n])
        let out = model(tokens, cache: cache)
        eval(out)
        done += n
    }

    var nextToken = Int32(context % config.vocabSize)

    // Warmup decode steps (excluded from timing).
    for _ in 0 ..< warmup {
        let t = MLXArray([nextToken], [1, 1])
        let out = model(t, cache: cache)
        eval(out)
        nextToken = Int32((Int(nextToken) + 1) % config.vocabSize)
    }

    // Timed decode steps. Each step is sequenceLength == 1 and we eval() the
    // output to force materialization (MLX is lazy).
    var samples: [Double] = []
    samples.reserveCapacity(timed)
    for _ in 0 ..< timed {
        let t = MLXArray([nextToken], [1, 1])
        let start = DispatchTime.now()
        let out = model(t, cache: cache)
        eval(out)
        let end = DispatchTime.now()
        samples.append(Double(end.uptimeNanoseconds - start.uptimeNanoseconds) / 1_000_000.0)
        nextToken = Int32((Int(nextToken) + 1) % config.vocabSize)
    }
    return (median(samples), samples.min() ?? 0)
}

// One-time global warmup: the FIRST MLX graph evaluated in a process pays a
// one-shot Metal kernel JIT / allocator cost. Without absorbing it here it
// leaks into whichever config/fuse combination happens to run first and
// inflates that single number. Run a small model end-to-end first to pay it.
func globalWarmup() throws {
    let tiny = BenchConfig(
        name: "warmup", hiddenSize: 512, numHiddenLayers: 2, intermediateSize: 1024,
        numAttentionHeads: 10, numKeyValueHeads: 4, headDim: 128, numNoiseHeads: 2,
        kRatio: 1, vocabSize: 256
    )
    for fuse in [false, true] {
        _ = try benchOne(config: tiny, context: 64, fuse: fuse)
    }
}

print("MotifDecodeBench — synthetic q4 grouped-attention DECODE micro-benchmark")
print("warmup=\(warmup) timed=\(timed) (median ms/step, batch=1, S=1, cache=q4 gs=64)")
print("NOTE: synthetic random weights, per-decode-step, one machine. NOT end-to-end tok/s.")
if let fuseEnv { print("MLX_MOTIF_FUSE_QKV (env) = \(fuseEnv)") }
print("")
func pad(_ s: String, _ w: Int) -> String {
    s.count >= w ? s : s + String(repeating: " ", count: w - s.count)
}

try globalWarmup()

print(pad("config", 18) + pad("context", 9) + pad("fuse", 8)
    + pad("ms/step(med)", 14) + "ms/step(min)")
print(String(repeating: "-", count: 60))

for config in configs where configFilter == "all" || configFilter == config.name {
    for context in contextLengths {
        for fuse in [false, true] {
            let r = try benchOne(config: config, context: context, fuse: fuse)
            print(pad(config.name, 18) + pad("\(context)", 9)
                + pad(fuse ? "ON" : "OFF", 8)
                + pad(String(format: "%.4f", r.median), 14)
                + String(format: "%.4f", r.min))
        }
    }
}
#else
import Foundation
print("MotifDecodeBench requires MOTIFKIT_ENABLE_MLX=1 build.")
exit(1)
#endif
