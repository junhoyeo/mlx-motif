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

    private func makeTemporaryDirectory() throws -> URL {
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent("MotifKitTests-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        return directory
    }
}
