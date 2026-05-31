#if canImport(MotifKitMLX)
import Foundation
import MotifKit
import MotifKitMLX

@main
struct MotifNativeGenerateCommand {
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
            FileHandle.standardError.write(Data("usage: MotifNativeGenerate --model <converted-mlx-dir> [--prompt text] [--max-tokens n] [--temperature t] [--think-mode visible|hidden|captured] [--speculative] [--speculative-draft-model dir] [--speculative-draft-tokens n] [--json]\n".utf8))
            Foundation.exit(2)
        }
        let modelDirectory = URL(fileURLWithPath: arguments[modelIndex + 1])
        let prompt = value(after: "--prompt", in: arguments) ?? "Hello"
        let maxTokens = Int(value(after: "--max-tokens", in: arguments) ?? "128") ?? 128
        let temperature = Double(value(after: "--temperature", in: arguments) ?? "0") ?? 0
        let thinkMode = MotifThinkMode(rawValue: value(after: "--think-mode", in: arguments) ?? "hidden") ?? .hidden
        let useSpeculative = arguments.contains("--speculative") || value(after: "--speculative-draft-model", in: arguments) != nil
        let jsonOutput = arguments.contains("--json")

        if useSpeculative {
            let draftDirectory = URL(
                fileURLWithPath: value(after: "--speculative-draft-model", in: arguments) ?? modelDirectory.path
            )
            let draftTokens = Int(value(after: "--speculative-draft-tokens", in: arguments) ?? "4") ?? 4
            let runtime = try await MotifMLXNativeRuntime.load(modelDirectory: modelDirectory)
            let draftRuntime = draftDirectory == modelDirectory
                ? runtime
                : try await MotifMLXNativeRuntime.load(modelDirectory: draftDirectory)
            let result = try await runtime.speculativeGenerate(
                messages: [.user(prompt)],
                draftRuntime: draftRuntime,
                parameters: MotifGenerationParameters(
                    maxTokens: maxTokens,
                    temperature: temperature,
                    thinkMode: thinkMode
                ),
                speculativeParameters: MotifSpeculativeDecodingParameters(
                    draftTokens: draftTokens,
                    includeRawTokens: jsonOutput
                )
            )
            if jsonOutput {
                let data = try JSONEncoder.pretty.encode(result)
                FileHandle.standardOutput.write(data)
                FileHandle.standardOutput.write(Data("\n".utf8))
            } else {
                print(result.text)
                let data = try JSONEncoder.pretty.encode(result.metrics)
                FileHandle.standardError.write(Data("\n[speculative_metrics]\n".utf8))
                FileHandle.standardError.write(data)
                FileHandle.standardError.write(Data("\n".utf8))
            }
            return
        }

        let backend = try MotifMLXBackend(modelDirectory: modelDirectory)
        let stream = backend.streamResponse(
            messages: [.user(prompt)],
            parameters: MotifGenerationParameters(
                maxTokens: maxTokens,
                temperature: temperature,
                thinkMode: thinkMode
            )
        )
        for try await event in stream {
            switch event {
            case .text(let text):
                print(text, terminator: "")
                fflush(stdout)
            case .reasoning(let reasoning):
                FileHandle.standardError.write(Data("\n[reasoning]\n\(reasoning)\n".utf8))
            case .completed:
                print("")
            }
        }
    }

    private static func value(after flag: String, in arguments: [String]) -> String? {
        guard let index = arguments.firstIndex(of: flag), arguments.indices.contains(index + 1) else {
            return nil
        }
        return arguments[index + 1]
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
struct MotifNativeGenerateCommand {
    static func main() {
        print("MotifNativeGenerate requires MOTIFKIT_ENABLE_MLX=1")
    }
}
#endif
