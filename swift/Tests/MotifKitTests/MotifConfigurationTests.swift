import Foundation
import XCTest
@testable import MotifKit

final class MotifConfigurationTests: XCTestCase {
    private func fixtureURL(_ name: String) -> URL {
        URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .appendingPathComponent("Fixtures")
            .appendingPathComponent(name)
    }

    private func decodeFixture<T: Decodable>(_ name: String, as type: T.Type = T.self) throws -> T {
        let data = try Data(contentsOf: fixtureURL(name))
        return try JSONDecoder().decode(T.self, from: data)
    }

    func testGroupedAttentionFlagTracksNoiseHeads() {
        let vanilla = MotifModelConfiguration(
            hiddenSize: 2048,
            numHiddenLayers: 24,
            intermediateSize: 8192,
            numAttentionHeads: 16,
            numKeyValueHeads: 4,
            vocabSize: 128_000
        )
        XCTAssertFalse(vanilla.isGroupedDifferentialAttention)

        let grouped = MotifModelConfiguration(
            hiddenSize: 4096,
            numHiddenLayers: 40,
            intermediateSize: 16384,
            numAttentionHeads: 40,
            numKeyValueHeads: 16,
            vocabSize: 128_000,
            headDim: 128,
            numNoiseHeads: 8
        )
        XCTAssertTrue(grouped.isGroupedDifferentialAttention)
    }

    func testDecodesVanillaMotifConfigFixture() throws {
        let config: MotifModelConfiguration = try decodeFixture("motif_config_vanilla.json")

        XCTAssertEqual(config.modelType, "motif")
        XCTAssertEqual(config.hiddenSize, 64)
        XCTAssertEqual(config.numHiddenLayers, 2)
        XCTAssertEqual(config.intermediateSize, 128)
        XCTAssertEqual(config.numAttentionHeads, 4)
        XCTAssertEqual(config.numKeyValueHeads, 4)
        XCTAssertEqual(config.vocabSize, 128)
        XCTAssertEqual(config.rmsNormEps, 1e-6)
        XCTAssertEqual(config.ropeTheta, 10_000)
        XCTAssertEqual(config.maxPositionEmbeddings, 64)
        XCTAssertEqual(config.headDim, 16)
        XCTAssertNil(config.numNoiseHeads)
        XCTAssertEqual(config.kRatio, 1)
        XCTAssertEqual(config.attnRMSNormEps, 1e-5)
        XCTAssertTrue(config.tieWordEmbeddings)
        XCTAssertNil(config.ropeScaling)
        XCTAssertEqual(config.hiddenActivation, "poly_norm")
        XCTAssertFalse(config.useBias)
        XCTAssertFalse(config.expanded)
        XCTAssertNil(config.slidingWindow)
        XCTAssertFalse(config.useSlidingWindow)
        XCTAssertNil(config.maxWindowLayers)
        XCTAssertFalse(config.fusedRope)
        XCTAssertEqual(config.bosTokenId, 1)
        XCTAssertEqual(config.eosTokenId, 2)
        XCTAssertFalse(config.isGroupedDifferentialAttention)
    }

    func testDecodesGroupedMotifConfigFixture() throws {
        let config: MotifModelConfiguration = try decodeFixture("motif_config_grouped.json")

        XCTAssertEqual(config.numAttentionHeads, 10)
        XCTAssertEqual(config.numKeyValueHeads, 4)
        XCTAssertEqual(config.numNoiseHeads, 2)
        XCTAssertEqual(config.kRatio, 1)
        XCTAssertFalse(config.tieWordEmbeddings)
        XCTAssertEqual(config.ropeScaling?.ropeType, "linear")
        XCTAssertEqual(config.ropeScaling?.factor, 2.0)
        XCTAssertEqual(config.ropeScaling?.originalMaxPositionEmbeddings, 64)
        XCTAssertTrue(config.expanded)
        XCTAssertEqual(config.slidingWindow, 32)
        XCTAssertTrue(config.useSlidingWindow)
        XCTAssertEqual(config.maxWindowLayers, 1)
        XCTAssertTrue(config.fusedRope)
        XCTAssertTrue(config.isGroupedDifferentialAttention)
    }

    func testGenerationConfigPreservesMultipleEOSTokenIDs() throws {
        let config: MotifGenerationConfiguration = try decodeFixture(
            "generation_config_multi_eos.json"
        )

        XCTAssertEqual(config.bosTokenId, 1)
        XCTAssertEqual(config.eosTokenIds, [2, 128_009])
        XCTAssertEqual(config.primaryEOSTokenId, 2)
        XCTAssertEqual(config.padTokenId, 0)
        XCTAssertEqual(config.maxNewTokens, 256)
        XCTAssertEqual(config.temperature, 0.6)
        XCTAssertEqual(config.doSample, true)
        XCTAssertEqual(config.chatTemplate, "{{ bos_token }}{{ messages }}")
    }

    func testGenerationConfigAcceptsSingleEOSTokenID() throws {
        let config: MotifGenerationConfiguration = try decodeFixture(
            "generation_config_single_eos.json"
        )

        XCTAssertEqual(config.eosTokenIds, [2])
        XCTAssertEqual(config.primaryEOSTokenId, 2)
        XCTAssertEqual(config.maxNewTokens, 128)
        XCTAssertEqual(config.temperature, 0.0)
        XCTAssertEqual(config.doSample, false)
    }

    func testModelBundleExtraEOSOmitsConfiguredPrimaryEOS() {
        let configuration = MotifModelConfiguration(
            hiddenSize: 64,
            numHiddenLayers: 2,
            intermediateSize: 128,
            numAttentionHeads: 4,
            numKeyValueHeads: 4,
            vocabSize: 128,
            eosTokenId: 2
        )
        let generationConfiguration = MotifGenerationConfiguration(eosTokenIds: [2, 100_257])
        let bundle = MotifModelBundle(
            directoryURL: URL(fileURLWithPath: "/tmp/motif"),
            configuration: configuration,
            generationConfiguration: generationConfiguration
        )

        XCTAssertEqual(bundle.extraEOSTokenIDs, [100_257])
    }

    func testModelBundleWithoutGenerationConfigHasNoExtraEOS() {
        let configuration = MotifModelConfiguration(
            hiddenSize: 64,
            numHiddenLayers: 2,
            intermediateSize: 128,
            numAttentionHeads: 4,
            numKeyValueHeads: 4,
            vocabSize: 128,
            eosTokenId: 2
        )
        let bundle = MotifModelBundle(
            directoryURL: URL(fileURLWithPath: "/tmp/motif"),
            configuration: configuration
        )

        XCTAssertEqual(bundle.extraEOSTokenIDs, [])
    }

    func testModelBundleLoaderReportsMissingConfig() throws {
        let directory = try makeTemporaryDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }

        XCTAssertThrowsError(try MotifModelBundleLoader.loadMetadata(from: directory)) { error in
            guard case MotifModelBundleLoaderError.missingConfig = error else {
                return XCTFail("Expected missingConfig, got \(error)")
            }
        }
    }

    func testModelBundleLoaderReportsMalformedConfig() throws {
        let directory = try makeTemporaryDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        try "{".write(
            to: directory.appendingPathComponent(MotifModelBundle.configFileName),
            atomically: true,
            encoding: .utf8
        )

        XCTAssertThrowsError(try MotifModelBundleLoader.loadMetadata(from: directory)) { error in
            guard case MotifModelBundleLoaderError.unreadableConfig = error else {
                return XCTFail("Expected unreadableConfig, got \(error)")
            }
        }
    }

    func testModelBundleLoaderReportsMalformedGenerationConfig() throws {
        let directory = try makeTemporaryDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        let configData = try Data(contentsOf: fixtureURL("motif_config_grouped.json"))
        try configData.write(to: directory.appendingPathComponent(MotifModelBundle.configFileName))
        try "{".write(
            to: directory.appendingPathComponent(MotifModelBundle.generationConfigFileName),
            atomically: true,
            encoding: .utf8
        )

        XCTAssertThrowsError(try MotifModelBundleLoader.loadMetadata(from: directory)) { error in
            guard case MotifModelBundleLoaderError.unreadableGenerationConfig = error else {
                return XCTFail("Expected unreadableGenerationConfig, got \(error)")
            }
        }
    }

    func testModelBundleLoaderReadsCheckpointTokenizerAndValidationMetadata() throws {
        let directory = try makeTemporaryDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        try writeMinimalConfig(to: directory)
        try """
        {
          "eos_token_id": [2, 100257],
          "chat_template": "generation-template"
        }
        """.write(
            to: directory.appendingPathComponent(MotifModelBundle.generationConfigFileName),
            atomically: true,
            encoding: .utf8
        )
        try """
        {
          "metadata": {"total_size": 1234},
          "weight_map": {
            "model.embed_tokens.weight": "model-00001-of-00002.safetensors",
            "model.layers.0.self_attn.q_proj.weight": "model-00002-of-00002.safetensors"
          }
        }
        """.write(
            to: directory.appendingPathComponent(MotifModelBundle.safetensorsIndexFileName),
            atomically: true,
            encoding: .utf8
        )
        try Data().write(to: directory.appendingPathComponent("model-00001-of-00002.safetensors"))
        try Data().write(to: directory.appendingPathComponent("model-00002-of-00002.safetensors"))
        try "{}".write(
            to: directory.appendingPathComponent(MotifModelBundle.tokenizerFileName),
            atomically: true,
            encoding: .utf8
        )
        try """
        {"chat_template": "tokenizer-config-template"}
        """.write(
            to: directory.appendingPathComponent(MotifModelBundle.tokenizerConfigFileName),
            atomically: true,
            encoding: .utf8
        )
        try "file-template".write(
            to: directory.appendingPathComponent(MotifModelBundle.chatTemplateFileName),
            atomically: true,
            encoding: .utf8
        )

        let bundle = try MotifModelBundleLoader.loadMetadata(from: directory)

        XCTAssertTrue(bundle.checkpointMetadata.isSharded)
        XCTAssertEqual(bundle.checkpointMetadata.tensorKeyCount, 2)
        XCTAssertEqual(
            bundle.checkpointMetadata.shardFileName(
                containingTensorKey: "model.layers.0.self_attn.q_proj.weight"
            ),
            "model-00002-of-00002.safetensors"
        )
        XCTAssertEqual(
            bundle.checkpointMetadata.shardFileNames,
            ["model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors"]
        )
        XCTAssertEqual(bundle.checkpointMetadata.indexMetadata["total_size"], .int(1234))
        XCTAssertEqual(bundle.tokenizerMetadata.preferredChatTemplate, "generation-template")
        XCTAssertEqual(
            bundle.tokenizerMetadata.chatTemplateSourceFileName,
            MotifModelBundle.generationConfigFileName
        )
        XCTAssertTrue(bundle.directoryValidation.isLoadableScaffold)
        XCTAssertNil(bundle.directoryValidation.summary)
    }

    func testModelBundleLoaderValidatesIncompleteModelDirectory() throws {
        let directory = try makeTemporaryDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        try writeMinimalConfig(to: directory)

        let bundle = try MotifModelBundleLoader.loadMetadata(from: directory)

        XCTAssertFalse(bundle.directoryValidation.isLoadableScaffold)
        XCTAssertEqual(
            bundle.directoryValidation.missingRequiredFiles,
            [
                "model.safetensors or model.safetensors.index.json",
                "tokenizer.json or readable tokenizer_config.json",
            ]
        )
        XCTAssertEqual(
            bundle.directoryValidation.warnings,
            ["generation_config.json not found", "chat template not found"]
        )
        XCTAssertEqual(
            bundle.directoryValidation.blockingSummary,
            "missing: model.safetensors or model.safetensors.index.json, tokenizer.json or readable tokenizer_config.json"
        )
    }

    func testModelBundleLoaderReportsMalformedCheckpointIndex() throws {
        let directory = try makeTemporaryDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        try writeMinimalConfig(to: directory)
        try "{".write(
            to: directory.appendingPathComponent(MotifModelBundle.safetensorsIndexFileName),
            atomically: true,
            encoding: .utf8
        )

        XCTAssertThrowsError(try MotifModelBundleLoader.loadMetadata(from: directory)) { error in
            guard case MotifModelBundleLoaderError.unreadableCheckpointIndex = error else {
                return XCTFail("Expected unreadableCheckpointIndex, got \(error)")
            }
        }
    }

    func testModelBundleLoaderRejectsCheckpointShardPathsEscapingBundle() throws {
        for shardFileName in ["../outside.safetensors", "/tmp/outside.safetensors"] {
            let directory = try makeTemporaryDirectory()
            defer { try? FileManager.default.removeItem(at: directory) }
            try writeMinimalConfig(to: directory)
            try """
            {
              "weight_map": {
                "model.embed_tokens.weight": "\(shardFileName)"
              }
            }
            """.write(
                to: directory.appendingPathComponent(MotifModelBundle.safetensorsIndexFileName),
                atomically: true,
                encoding: .utf8
            )

            XCTAssertThrowsError(try MotifModelBundleLoader.loadMetadata(from: directory)) { error in
                guard case MotifModelBundleLoaderError.invalidCheckpointShardPath(_, let invalidShardFileName) = error else {
                    return XCTFail("Expected invalidCheckpointShardPath, got \(error)")
                }
                XCTAssertEqual(invalidShardFileName, shardFileName)
            }
        }
    }

    func testModelBundleLoaderTreatsMalformedTokenizerConfigAsBlockingWhenNoTokenizerJSON() throws {
        let directory = try makeTemporaryDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        try writeMinimalConfig(to: directory)
        try Data().write(to: directory.appendingPathComponent(MotifModelBundle.safetensorsFileName))
        try "{".write(
            to: directory.appendingPathComponent(MotifModelBundle.tokenizerConfigFileName),
            atomically: true,
            encoding: .utf8
        )

        let bundle = try MotifModelBundleLoader.loadMetadata(from: directory)

        XCTAssertFalse(bundle.tokenizerMetadata.hasTokenizerFiles)
        XCTAssertNil(bundle.tokenizerMetadata.tokenizerConfigChatTemplate)
        XCTAssertTrue(bundle.tokenizerMetadata.tokenizerConfigReadError?.contains("tokenizer_config.json unreadable") == true)
        XCTAssertEqual(
            bundle.directoryValidation.missingRequiredFiles,
            ["tokenizer.json or readable tokenizer_config.json"]
        )
        XCTAssertTrue(
            bundle.directoryValidation.warnings.contains {
                $0.contains("tokenizer_config.json unreadable")
            }
        )
    }

    func testModelBundleLoaderReportsUnreadableChatTemplateFile() throws {
        let directory = try makeTemporaryDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        try writeMinimalConfig(to: directory)
        try Data().write(to: directory.appendingPathComponent(MotifModelBundle.safetensorsFileName))
        try "{}".write(
            to: directory.appendingPathComponent(MotifModelBundle.tokenizerFileName),
            atomically: true,
            encoding: .utf8
        )
        try Data([0xff, 0xfe]).write(to: directory.appendingPathComponent(MotifModelBundle.chatTemplateFileName))

        let bundle = try MotifModelBundleLoader.loadMetadata(from: directory)

        XCTAssertTrue(bundle.tokenizerMetadata.hasTokenizerFiles)
        XCTAssertNil(bundle.tokenizerMetadata.chatTemplateFileContents)
        XCTAssertTrue(bundle.tokenizerMetadata.chatTemplateReadError?.contains("chat_template.jinja unreadable") == true)
        XCTAssertTrue(bundle.directoryValidation.isLoadableScaffold)
        XCTAssertTrue(
            bundle.directoryValidation.warnings.contains {
                $0.contains("chat_template.jinja unreadable")
            }
        )
    }

    private func writeMinimalConfig(to directory: URL) throws {
        let configData = try Data(contentsOf: fixtureURL("motif_config_grouped.json"))
        try configData.write(to: directory.appendingPathComponent(MotifModelBundle.configFileName))
    }

    private func makeTemporaryDirectory() throws -> URL {
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent("MotifKitTests-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        return directory
    }
}
