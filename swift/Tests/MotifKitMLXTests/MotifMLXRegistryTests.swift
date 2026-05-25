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
        XCTAssertEqual(plan.layerPlan?.attentionLayout.outputProjectionInputSize, 8_192)
    }

    func testBackendInitializesLoadPlanFromBundleDirectory() throws {
        let directory = try makeModelDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }

        let backend = try MotifMLXBackend(modelDirectory: directory)

        XCTAssertEqual(backend.configuration?.modelType, "motif")
        XCTAssertEqual(backend.loadPlan?.registryKey, MotifMLXModelRegistry.modelType)
        XCTAssertEqual(backend.loadPlan?.extraEOSTokenIDs, Set([100_257]))
        XCTAssertNil(backend.loadPlan?.validationErrorDescription)
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
          "pad_token_id": 0
        }
        """.write(
            to: directory.appendingPathComponent(MotifModelBundle.generationConfigFileName),
            atomically: true,
            encoding: .utf8
        )
        return directory
    }
}
