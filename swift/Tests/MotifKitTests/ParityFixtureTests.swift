import Foundation
import XCTest
@testable import MotifKit

final class ParityFixtureTests: XCTestCase {
    func testMotifConfigFixturesMapIntoSwiftConfiguration() throws {
        let root = try loadFixtureRoot()
        let configs = try array(root["configs"])

        for rawCase in configs {
            let caseDictionary = try dictionary(rawCase)
            let caseName = try string(caseDictionary, "name")
            let python = try dictionary(caseDictionary["python"])
            let expectations = try dictionary(caseDictionary["expectations"])

            let configuration = MotifModelConfiguration(
                modelType: try string(python, "model_type"),
                hiddenSize: try int(python, "hidden_size"),
                numHiddenLayers: try int(python, "num_hidden_layers"),
                intermediateSize: try int(python, "intermediate_size"),
                numAttentionHeads: try int(python, "num_attention_heads"),
                numKeyValueHeads: try int(python, "num_key_value_heads"),
                vocabSize: try int(python, "vocab_size"),
                rmsNormEps: try double(python, "rms_norm_eps"),
                ropeTheta: try double(python, "rope_theta"),
                maxPositionEmbeddings: try int(python, "max_position_embeddings"),
                headDim: try optionalInt(python, "head_dim"),
                numNoiseHeads: try optionalInt(python, "num_noise_heads"),
                kRatio: try int(python, "k_ratio"),
                attnRMSNormEps: try double(python, "attn_rms_norm_eps"),
                hiddenActivation: try string(python, "hidden_act"),
                useBias: try bool(python, "use_bias"),
                fusedRope: try bool(python, "fused_rope")
            )

            XCTAssertEqual(
                configuration.isGroupedDifferentialAttention,
                try bool(expectations, "is_grouped"),
                caseName
            )
            XCTAssertEqual(configuration.hiddenActivation, "poly_norm")
            XCTAssertEqual(configuration.headDim, try int(expectations, "effective_head_dim"))

            if configuration.isGroupedDifferentialAttention {
                let noiseHeads = try XCTUnwrap(configuration.numNoiseHeads)
                XCTAssertEqual(
                    configuration.numAttentionHeads / noiseHeads,
                    try int(expectations, "origin_heads_per_noise_head")
                )
                XCTAssertEqual(
                    configuration.numKeyValueHeads / noiseHeads,
                    try int(expectations, "kv_heads_per_group")
                )
            }
        }
    }

    func testThinkFilterFixturesMatchSwiftBehavior() throws {
        let root = try loadFixtureRoot()
        let cases = try array(root["think_filter_cases"])

        for rawCase in cases {
            let caseDictionary = try dictionary(rawCase)
            let caseName = try string(caseDictionary, "name")
            let mode = try XCTUnwrap(MotifThinkMode(rawValue: try string(caseDictionary, "mode")))
            var filter = ThinkStreamFilter(
                mode: mode,
                startsInsideThinkBlock: try bool(caseDictionary, "start_in_think")
            )

            let emitted = try array(caseDictionary["chunks"])
                .map { try string($0) }
                .map { filter.feed($0) }
                .joined()

            XCTAssertEqual(emitted, try string(caseDictionary, "expected_emitted"), caseName)
            XCTAssertEqual(
                filter.capturedReasoning,
                try string(caseDictionary, "expected_captured"),
                caseName
            )
        }
    }

    func testPolyNormFixtureMatchesPureSwiftReferenceFormula() throws {
        let root = try loadFixtureRoot()
        let componentChecks = try array(root["component_checks"])
        let rawCase = try XCTUnwrap(componentChecks.first { rawCase in
            guard let dictionary = rawCase as? [String: Any] else { return false }
            return dictionary["kind"] as? String == "polynorm"
        })
        let caseDictionary = try dictionary(rawCase)

        let eps = try double(caseDictionary, "eps")
        let weight = try doubleArray(caseDictionary["weight"])
        let bias = try doubleArray(caseDictionary["bias"])[0]
        let input = try nestedDoubleArray(caseDictionary["input"])
        let expected = try nestedDoubleArray(caseDictionary["expected"])

        let actual = input.map { row in
            let meanX2 = row.map { pow($0, 2) }.reduce(0, +) / Double(row.count)
            let meanX4 = row.map { pow($0, 4) }.reduce(0, +) / Double(row.count)
            let meanX6 = row.map { pow($0, 6) }.reduce(0, +) / Double(row.count)
            let invRMSX = 1 / sqrt(meanX2 + eps)
            let invRMSX2 = 1 / sqrt(meanX4 + eps)
            let invRMSX3 = 1 / sqrt(meanX6 + eps)

            return row.map { value in
                weight[0] * (pow(value, 3) * invRMSX3)
                    + weight[1] * (pow(value, 2) * invRMSX2)
                    + weight[2] * (value * invRMSX)
                    + bias
            }
        }

        XCTAssertEqual(actual.count, expected.count)
        for (actualRow, expectedRow) in zip(actual, expected) {
            XCTAssertEqual(actualRow.count, expectedRow.count)
            for (actualValue, expectedValue) in zip(actualRow, expectedRow) {
                XCTAssertEqual(actualValue, expectedValue, accuracy: 1e-12)
            }
        }
    }
}

private func fixtureURL() -> URL {
    URL(fileURLWithPath: #filePath)
        .deletingLastPathComponent()
        .deletingLastPathComponent()
        .deletingLastPathComponent()
        .deletingLastPathComponent()
        .appendingPathComponent("tests/fixtures/motif_parity_cases.json")
}

private func loadFixtureRoot() throws -> [String: Any] {
    let data = try Data(contentsOf: fixtureURL())
    let root = try JSONSerialization.jsonObject(with: data)
    return try dictionary(root)
}

private func dictionary(_ value: Any?) throws -> [String: Any] {
    try XCTUnwrap(value as? [String: Any])
}

private func array(_ value: Any?) throws -> [Any] {
    try XCTUnwrap(value as? [Any])
}

private func string(_ dictionary: [String: Any], _ key: String) throws -> String {
    try string(dictionary[key])
}

private func string(_ value: Any?) throws -> String {
    try XCTUnwrap(value as? String)
}

private func int(_ dictionary: [String: Any], _ key: String) throws -> Int {
    let value = try XCTUnwrap(dictionary[key])
    if let number = value as? NSNumber { return number.intValue }
    return try XCTUnwrap(value as? Int)
}

private func optionalInt(_ dictionary: [String: Any], _ key: String) throws -> Int? {
    guard let value = dictionary[key], !(value is NSNull) else { return nil }
    if let number = value as? NSNumber { return number.intValue }
    return try XCTUnwrap(value as? Int)
}

private func double(_ dictionary: [String: Any], _ key: String) throws -> Double {
    let value = try XCTUnwrap(dictionary[key])
    if let number = value as? NSNumber { return number.doubleValue }
    return try XCTUnwrap(value as? Double)
}

private func bool(_ dictionary: [String: Any], _ key: String) throws -> Bool {
    let value = try XCTUnwrap(dictionary[key])
    if let bool = value as? Bool { return bool }
    if let number = value as? NSNumber { return number.boolValue }
    return try XCTUnwrap(value as? Bool)
}

private func doubleArray(_ value: Any?) throws -> [Double] {
    try array(value).map { item in
        if let number = item as? NSNumber { return number.doubleValue }
        return try XCTUnwrap(item as? Double)
    }
}

private func nestedDoubleArray(_ value: Any?) throws -> [[Double]] {
    try array(value).map { try doubleArray($0) }
}
