import Foundation
import XCTest
@testable import MotifKit

final class MotifToolCallingTests: XCTestCase {
    // MARK: - Cross-language golden preamble parity

    /// The shared fixture's `expected` string is the authoritative Python
    /// output of `build_tools_preamble`. The Swift port must reproduce it
    /// byte-for-byte for the same tools JSON.
    func testPreambleMatchesCrossLanguageGoldenFixture() throws {
        let cases = try loadPreambleCases()
        XCTAssertFalse(cases.isEmpty, "fixture must contain at least one case")
        for rawCase in cases {
            let caseDict = try dictionary(rawCase)
            let name = try string(caseDict, "name")
            let tools = try array(caseDict["tools"])
            let expected = try string(caseDict, "expected")
            XCTAssertEqual(MotifToolCalling.buildToolsPreamble(tools), expected, name)
        }
    }

    // MARK: - parseToolCall semantics (ported from tests/test_tool_calls.py)

    private let weatherToolNames: Set<String> = ["get_weather"]

    func testValidSingleToolCallObject() {
        let text = #"{"tool_call": {"name": "get_weather", "arguments": {"city": "Tokyo"}}}"#
        let call = MotifToolCalling.parseToolCall(text: text, toolNames: weatherToolNames)
        XCTAssertEqual(call?.name, "get_weather")
        XCTAssertEqual(call?.arguments["city"], .string("Tokyo"))
    }

    func testStandardOpenAIShapeWithoutWrapper() {
        let text = #"{"name": "get_weather", "arguments": {"city": "Paris"}}"#
        let call = MotifToolCalling.parseToolCall(text: text, toolNames: weatherToolNames)
        XCTAssertEqual(call?.name, "get_weather")
        XCTAssertEqual(call?.arguments["city"], .string("Paris"))
    }

    func testRepeatedObjectsReturnsFirst() {
        let one = #"{"tool_call": {"name": "get_weather", "arguments": {"city": "Tokyo"}}}"#
        let two = #"{"tool_call": {"name": "get_weather", "arguments": {"city": "Osaka"}}}"#
        let text = "\(one)\n\(two)\n\(one)"
        let call = MotifToolCalling.parseToolCall(text: text, toolNames: weatherToolNames)
        XCTAssertEqual(call?.arguments["city"], .string("Tokyo"))
    }

    func testSurroundingProseIsTolerated() {
        let text = "Sure, let me check that for you.\n"
            + #"{"tool_call": {"name": "get_weather", "arguments": {"city": "Berlin"}}}"#
            + "\nI will get back to you."
        let call = MotifToolCalling.parseToolCall(text: text, toolNames: weatherToolNames)
        XCTAssertEqual(call?.arguments["city"], .string("Berlin"))
    }

    func testNoToolCallReturnsNil() {
        XCTAssertNil(MotifToolCalling.parseToolCall(text: "The weather in Tokyo is sunny.", toolNames: weatherToolNames))
    }

    func testMalformedJSONReturnsNil() {
        let text = #"{"tool_call": {"name": "get_weather", "arguments": {city: Tokyo}}"#
        XCTAssertNil(MotifToolCalling.parseToolCall(text: text, toolNames: weatherToolNames))
    }

    func testNonToolJSONReturnsNil() {
        XCTAssertNil(MotifToolCalling.parseToolCall(text: #"{"foo": "bar", "n": 1}"#, toolNames: weatherToolNames))
    }

    func testUnknownToolNameSkippedWhenNamesGiven() {
        let text = #"{"tool_call": {"name": "send_email", "arguments": {"to": "a@b.c"}}}"#
        XCTAssertNil(MotifToolCalling.parseToolCall(text: text, toolNames: weatherToolNames))
    }

    func testUnknownToolNameAcceptedWhenNoFilter() {
        let text = #"{"tool_call": {"name": "send_email", "arguments": {"to": "a@b.c"}}}"#
        let call = MotifToolCalling.parseToolCall(text: text, toolNames: nil)
        XCTAssertEqual(call?.name, "send_email")
    }

    func testArgumentsDefaultToEmptyWhenOmitted() {
        let text = #"{"tool_call": {"name": "get_weather"}}"#
        let call = MotifToolCalling.parseToolCall(text: text, toolNames: weatherToolNames)
        XCTAssertEqual(call?.name, "get_weather")
        XCTAssertEqual(call?.arguments.isEmpty, true)
    }

    func testNonDictArgumentsRejected() {
        let text = #"{"tool_call": {"name": "get_weather", "arguments": "Tokyo"}}"#
        XCTAssertNil(MotifToolCalling.parseToolCall(text: text, toolNames: weatherToolNames))
    }

    func testBracesInStringValueDoNotBreakMatching() {
        let text = #"{"tool_call": {"name": "get_weather", "arguments": {"city": "Tokyo {special}"}}}"#
        let call = MotifToolCalling.parseToolCall(text: text, toolNames: weatherToolNames)
        XCTAssertEqual(call?.arguments["city"], .string("Tokyo {special}"))
    }

    func testEmptyTextReturnsNil() {
        XCTAssertNil(MotifToolCalling.parseToolCall(text: "", toolNames: weatherToolNames))
    }

    // MARK: - Fixture loading + helpers

    private func loadPreambleCases() throws -> [Any] {
        let url = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .appendingPathComponent("tests/fixtures/tool_preamble_cases.json")
        let data = try Data(contentsOf: url)
        let root = try dictionary(try JSONSerialization.jsonObject(with: data))
        return try array(root["cases"])
    }

    private func dictionary(_ value: Any?) throws -> [String: Any] {
        try XCTUnwrap(value as? [String: Any])
    }

    private func array(_ value: Any?) throws -> [Any] {
        try XCTUnwrap(value as? [Any])
    }

    private func string(_ dict: [String: Any], _ key: String) throws -> String {
        try XCTUnwrap(dict[key] as? String)
    }
}
