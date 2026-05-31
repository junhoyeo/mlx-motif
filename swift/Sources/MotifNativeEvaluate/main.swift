#if canImport(MotifKitMLX)
import Foundation
import MotifKit
import MotifKitMLX

@main
struct MotifNativeEvaluateCommand {
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
            FileHandle.standardError.write(Data("usage: MotifNativeEvaluate --model <converted-mlx-dir> [--mode perplexity|bench|bench-suite|logits] [--text text|--text-file path] [--chunk n] [--max-tokens n] [--prompt text|--prompt-file path] [--suite-json path] [--top-k n]\n".utf8))
            Foundation.exit(2)
        }
        let mode = value(after: "--mode", in: arguments) ?? "perplexity"
        guard mode == "perplexity" || mode == "bench" || mode == "bench-suite" || mode == "logits" else {
            throw CommandError.invalidMode(mode)
        }
        let modelDirectory = URL(fileURLWithPath: arguments[modelIndex + 1])
        let runtime = try await MotifMLXNativeRuntime.load(modelDirectory: modelDirectory)

        switch mode {
        case "perplexity":
            let text = try inputText(arguments: arguments)
            let chunk = Int(value(after: "--chunk", in: arguments) ?? "512") ?? 512
            let maxTokens = Int(value(after: "--max-tokens", in: arguments) ?? "2048") ?? 2048
            let result = try runtime.perplexity(text: text, chunkSize: chunk, maxTokens: maxTokens)
            let data = try JSONEncoder.pretty.encode(result)
            FileHandle.standardOutput.write(data)
            FileHandle.standardOutput.write(Data("\n".utf8))
        case "bench":
            let prompt = try benchmarkPrompt(arguments: arguments)
            let maxTokens = Int(value(after: "--max-tokens", in: arguments) ?? "64") ?? 64
            let temperature = Double(value(after: "--temperature", in: arguments) ?? "0") ?? 0
            let result = try await runtime.benchmarkGeneration(
                messages: [.user(prompt)],
                maxTokens: maxTokens,
                temperature: temperature
            )
            let data = try JSONEncoder.pretty.encode(result)
            FileHandle.standardOutput.write(data)
            FileHandle.standardOutput.write(Data("\n".utf8))
        case "bench-suite":
            let suite = try BenchmarkSuite.load(path: value(after: "--suite-json", in: arguments))
            let started = Date()
            var caseResults: [BenchmarkSuiteCaseResult] = []
            for benchmarkCase in suite.cases {
                let repeats = max(1, benchmarkCase.repeats ?? suite.repeats ?? 1)
                var runs: [MotifMLXGenerationBenchmark] = []
                runs.reserveCapacity(repeats)
                for _ in 0..<repeats {
                    let result = try await runtime.benchmarkGeneration(
                        messages: [.user(benchmarkCase.prompt)],
                        maxTokens: benchmarkCase.maxTokens ?? suite.maxTokens ?? 64,
                        temperature: benchmarkCase.temperature ?? suite.temperature ?? 0
                    )
                    runs.append(result)
                }
                caseResults.append(BenchmarkSuiteCaseResult(
                    name: benchmarkCase.name,
                    requestedPromptTokens: benchmarkCase.requestedPromptTokens,
                    repeats: repeats,
                    runs: runs
                ))
            }
            let result = BenchmarkSuiteResult(
                modelDirectory: modelDirectory.path,
                elapsedSeconds: Date().timeIntervalSince(started),
                cases: caseResults
            )
            let data = try JSONEncoder.pretty.encode(result)
            FileHandle.standardOutput.write(data)
            FileHandle.standardOutput.write(Data("\n".utf8))
        case "logits":
            let text = try inputText(arguments: arguments)
            let maxTokens = Int(value(after: "--max-tokens", in: arguments) ?? "512") ?? 512
            let topK = Int(value(after: "--top-k", in: arguments) ?? "10") ?? 10
            let result = try runtime.logitSnapshot(text: text, maxTokens: maxTokens, topK: topK)
            let data = try JSONEncoder.pretty.encode(result)
            FileHandle.standardOutput.write(data)
            FileHandle.standardOutput.write(Data("\n".utf8))
        default:
            throw CommandError.invalidMode(mode)
        }
    }

    private static func benchmarkPrompt(arguments: [String]) throws -> String {
        if let prompt = value(after: "--prompt", in: arguments) { return prompt }
        if let path = value(after: "--prompt-file", in: arguments) {
            return try String(contentsOfFile: path, encoding: .utf8)
        }
        return "Explain grouped differential attention in one sentence."
    }

    private static func inputText(arguments: [String]) throws -> String {
        if let text = value(after: "--text", in: arguments) { return text }
        if let path = value(after: "--text-file", in: arguments) {
            return try String(contentsOfFile: path, encoding: .utf8)
        }
        return """
        The differential transformer computes attention as the difference between two softmax distributions weighted by a learnable scalar. Grouped differential attention shares noise heads across several origin heads, reducing projection cost while preserving the denoising effect. PolyNorm applies normalized linear, quadratic, and cubic transforms as a trainable activation.
        """
    }

    private static func value(after flag: String, in arguments: [String]) -> String? {
        guard let index = arguments.firstIndex(of: flag), arguments.indices.contains(index + 1) else {
            return nil
        }
        return arguments[index + 1]
    }
}

private struct BenchmarkSuite: Decodable {
    var repeats: Int?
    var maxTokens: Int?
    var temperature: Double?
    var cases: [BenchmarkSuiteCase]

    static func load(path: String?) throws -> BenchmarkSuite {
        guard let path else {
            throw CommandError.missingSuiteJSON
        }
        let url = URL(fileURLWithPath: path)
        return try JSONDecoder().decode(BenchmarkSuite.self, from: Data(contentsOf: url))
    }
}

private struct BenchmarkSuiteCase: Decodable {
    var name: String
    var prompt: String
    var requestedPromptTokens: Int?
    var repeats: Int?
    var maxTokens: Int?
    var temperature: Double?
}

private struct BenchmarkSuiteResult: Encodable {
    var modelDirectory: String
    var elapsedSeconds: Double
    var cases: [BenchmarkSuiteCaseResult]
}

private struct BenchmarkSuiteCaseResult: Encodable {
    var name: String
    var requestedPromptTokens: Int?
    var repeats: Int
    var runs: [MotifMLXGenerationBenchmark]
}

private enum CommandError: Error, LocalizedError {
    case invalidMode(String)
    case missingSuiteJSON

    var errorDescription: String? {
        switch self {
        case .invalidMode(let mode):
            "Unsupported mode: \(mode). Use perplexity, bench, bench-suite, or logits."
        case .missingSuiteJSON:
            "bench-suite mode requires --suite-json <path>."
        }
    }
}

private extension JSONEncoder {
    static var pretty: JSONEncoder {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        return encoder
    }
}
#else
@main
struct MotifNativeEvaluateCommand {
    static func main() {
        print("MotifNativeEvaluate requires MOTIFKIT_ENABLE_MLX=1")
    }
}
#endif
