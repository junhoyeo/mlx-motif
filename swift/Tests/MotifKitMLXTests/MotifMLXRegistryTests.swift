import Foundation
import MotifKit
@testable import MotifKitMLX
import XCTest

final class MotifMLXRegistryTests: XCTestCase {
    func testRegistryAcceptsMotifAndBuildsLayerPlan() throws {
        let configuration = makeGroupedConfiguration()

        XCTAssertTrue(MotifMLXModelRegistry.accepts(configuration))
        let plan = MotifMLXLoadPlan(configuration: configuration)

        XCTAssertEqual(plan.modelType, "motif")
        XCTAssertEqual(plan.registryKey, MotifMLXModelRegistry.modelType)
        XCTAssertEqual(plan.attentionVariant, .groupedDifferentialAttention)
        XCTAssertEqual(plan.requiredKernelNames, MotifMetalKernelRegistry.required.map(\.name))
        XCTAssertNil(plan.validationErrorDescription)

        let layerPlan = try XCTUnwrap(plan.layerPlan)
        XCTAssertEqual(layerPlan.attentionLayout.variant, .groupedDifferential)
        XCTAssertEqual(layerPlan.attentionLayout.queryHeads, 40)
        XCTAssertEqual(layerPlan.mlpLayout.activationName, "poly_norm")
        XCTAssertNil(plan.mlxModelConfiguration)
    }

    func testBundleLoadPlanCarriesMLXModelConfigurationAndExtraEOS() throws {
        let directory = try makeModelDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        let bundle = try MotifModelBundle(directoryURL: directory)

        let plan = MotifMLXModelRegistry.loadPlan(for: bundle)

        XCTAssertEqual(plan.modelDirectory, directory)
        XCTAssertEqual(plan.extraEOSTokenIDs, Set([100_257]))
        XCTAssertEqual(plan.mlxModelConfiguration?.id, .directory(directory))
        XCTAssertEqual(plan.mlxModelConfiguration?.eosTokenIds, Set([2, 100_257]))
        XCTAssertEqual(plan.checkpointMetadata?.tensorKeyCount, 1)
        XCTAssertEqual(plan.tokenizerMetadata?.preferredChatTemplate, "generation-template")
        XCTAssertTrue(plan.directoryValidation?.isLoadableScaffold == true)
        XCTAssertEqual(plan.chatTemplate, "generation-template")
        XCTAssertEqual(plan.layerPlan?.attentionLayout.outputProjectionInputSize, 8_192)
    }

    func testBackendInitializesLoadPlanFromBundleDirectory() throws {
        let directory = try makeModelDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }

        let backend = try MotifMLXBackend(modelDirectory: directory)

        XCTAssertEqual(backend.configuration?.modelType, "motif")
        XCTAssertEqual(backend.loadPlan?.registryKey, MotifMLXModelRegistry.modelType)
        XCTAssertEqual(backend.loadPlan?.extraEOSTokenIDs, Set([100_257]))
        XCTAssertEqual(backend.loadPlan?.checkpointMetadata?.shardFileNames, ["model.safetensors"])
        XCTAssertEqual(backend.loadPlan?.tokenizerMetadata?.chatTemplateSourceFileName, "generation_config.json")
        XCTAssertNil(backend.loadPlan?.validationErrorDescription)
    }

    func testLayerPlanIncludesDecoderGraphScaffoldWithoutRuntimeClaim() throws {
        let configuration = makeGroupedConfiguration()

        let plan = MotifMLXLoadPlan(configuration: configuration)
        let graph = try XCTUnwrap(plan.layerPlan?.decoderGraphPlan)

        XCTAssertEqual(graph.capabilityLabels, [.buildableScaffold, .stillUnavailable])
        XCTAssertFalse(graph.capabilityLabels.contains(.runtimeGeneratedOutput))
        XCTAssertEqual(graph.embeddingShape, [128_000, 4_096])
        XCTAssertEqual(graph.decoderLayerCount, 40)
        XCTAssertEqual(graph.firstDecoderLayer.layerIndex, 0)
        XCTAssertEqual(graph.firstDecoderLayer.attentionProjectionShapes["q_proj"], [5_120, 4_096])
        XCTAssertEqual(graph.firstDecoderLayer.attentionProjectionShapes["k_proj"], [2_048, 4_096])
        XCTAssertEqual(graph.firstDecoderLayer.attentionProjectionShapes["v_proj"], [2_048, 4_096])
        XCTAssertEqual(graph.firstDecoderLayer.attentionProjectionShapes["o_proj"], [4_096, 8_192])
        XCTAssertEqual(graph.firstDecoderLayer.cacheKind, .groupedFourSlot)
        XCTAssertEqual(graph.finalNormShape, [4_096])
        XCTAssertEqual(graph.lmHeadShape, [128_000, 4_096])
        XCTAssertFalse(graph.tiedEmbeddingLMHead)
        XCTAssertTrue(graph.backendReadiness.contains("buildable scaffold"))
    }

    func testMotifMLXModelBuildsDecoderGraphModuleTree() throws {
        try XCTSkipUnless(
            ProcessInfo.processInfo.environment["MOTIFKIT_RUN_MLX_RUNTIME_TESTS"] == "1",
            "MotifMLXModel allocates MLX arrays; runtime ops require the default metallib."
        )
        let configuration = MotifModelConfiguration(
            hiddenSize: 8,
            numHiddenLayers: 2,
            intermediateSize: 16,
            numAttentionHeads: 4,
            numKeyValueHeads: 2,
            vocabSize: 32,
            headDim: 2,
            numNoiseHeads: 1,
            kRatio: 1,
            hiddenActivation: "poly_norm"
        )

        let model = try MotifMLXModel(configuration: configuration)

        XCTAssertEqual(model.vocabularySize, 32)
        XCTAssertEqual(model.kvHeads, [2, 2])
        XCTAssertEqual(model.model.layers.count, 2)
        XCTAssertEqual(model.graphPlan.capabilityLabels, [.buildableScaffold, .stillUnavailable])
        XCTAssertEqual(model.graphPlan.firstDecoderLayer.attentionProjectionShapes["q_proj"], [8, 8])
        XCTAssertEqual(model.graphPlan.firstDecoderLayer.attentionProjectionShapes["v_proj"], [4, 8])
        XCTAssertNotNil(model.lmHead)
        XCTAssertEqual(model.model.layers[0].attention.layout.variant, .groupedDifferential)
        XCTAssertEqual(model.model.layers[0].attention.lambdaInit, Float(MotifAttentionLayout.lambdaInit(layerIndex: 0)))
        XCTAssertEqual(model.loraLayers.count, 2)
    }

    func testBackendKeepsRuntimeGeneratedOutputUnavailable() async throws {
        let backend = MotifMLXBackend(configuration: makeGroupedConfiguration())

        XCTAssertEqual(backend.capabilityLabels, [.buildableScaffold, .stillUnavailable])
        XCTAssertFalse(backend.capabilityLabels.contains(.runtimeGeneratedOutput))

        var iterator = backend.streamResponse(
            messages: [.user("Hello")],
            parameters: MotifGenerationParameters(maxTokens: 1)
        ).makeAsyncIterator()

        do {
            _ = try await iterator.next()
            XCTFail("MotifMLXBackend should not claim runtime-generated output yet")
        } catch MotifBackendError.nativeBackendUnavailable(let detail) {
            XCTAssertTrue(detail.contains("Capability labels: buildable scaffold, still unavailable"))
            XCTAssertTrue(detail.contains("does not claim runtime-generated output yet"))
        } catch {
            XCTFail("Expected nativeBackendUnavailable, got \(error)")
        }
    }

    private func makeGroupedConfiguration() -> MotifModelConfiguration {
        MotifModelConfiguration(
            hiddenSize: 4_096,
            numHiddenLayers: 40,
            intermediateSize: 16_384,
            numAttentionHeads: 40,
            numKeyValueHeads: 16,
            vocabSize: 128_000,
            headDim: 128,
            numNoiseHeads: 8,
            kRatio: 1,
            hiddenActivation: "poly_norm",
            eosTokenId: 2
        )
    }

    private func makeModelDirectory() throws -> URL {
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent("MotifKitMLXTests-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)

        try """
        {
          "model_type": "motif",
          "hidden_size": 4096,
          "num_hidden_layers": 40,
          "intermediate_size": 16384,
          "num_attention_heads": 40,
          "num_key_value_heads": 16,
          "vocab_size": 128000,
          "head_dim": 128,
          "num_noise_heads": 8,
          "k_ratio": 1,
          "hidden_act": "poly_norm",
          "eos_token_id": 2
        }
        """.write(
            to: directory.appendingPathComponent(MotifModelBundle.configFileName),
            atomically: true,
            encoding: .utf8
        )
        try """
        {
          "eos_token_id": [2, 100257],
          "pad_token_id": 0,
          "chat_template": "generation-template"
        }
        """.write(
            to: directory.appendingPathComponent(MotifModelBundle.generationConfigFileName),
            atomically: true,
            encoding: .utf8
        )
        try """
        {
          "metadata": {"total_size": 64},
          "weight_map": {
            "model.embed_tokens.weight": "model.safetensors"
          }
        }
        """.write(
            to: directory.appendingPathComponent(MotifModelBundle.safetensorsIndexFileName),
            atomically: true,
            encoding: .utf8
        )
        try Data().write(to: directory.appendingPathComponent(MotifModelBundle.safetensorsFileName))
        try "{}".write(
            to: directory.appendingPathComponent(MotifModelBundle.tokenizerFileName),
            atomically: true,
            encoding: .utf8
        )
        return directory
    }
}
