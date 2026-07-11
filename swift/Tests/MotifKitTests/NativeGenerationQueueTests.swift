import Foundation
@testable import MotifKit
import XCTest

final class NativeGenerationQueueTests: XCTestCase {
    func testCancellationCleanupFinishesBeforeReplacementProducerStarts() async throws {
        let queue = NativeGenerationQueue()
        let recorder = GenerationActivityRecorder()

        let first = Task {
            try await queue.run {
                await recorder.start("first")
                do {
                    try await waitForCancellation()
                } catch {
                    await nonCancellableYieldPass()
                    await recorder.finish("first")
                    throw error
                }
            }
        }
        await waitUntil { await recorder.activeCount == 1 }

        first.cancel()
        let replacement = Task {
            try await queue.run {
                await recorder.start("replacement")
                await recorder.finish("replacement")
            }
        }

        do {
            _ = try await first.value
            XCTFail("Expected the first producer to be cancelled")
        } catch is CancellationError {
            // Expected.
        }
        try await replacement.value

        let snapshot = await recorder.snapshot
        XCTAssertEqual(snapshot.maximumActive, 1)
        XCTAssertEqual(
            snapshot.events,
            ["start:first", "finish:first", "start:replacement", "finish:replacement"]
        )
    }

    private func waitUntil(
        attempts: Int = 2_000,
        _ predicate: @escaping @Sendable () async -> Bool
    ) async {
        for _ in 0..<attempts {
            if await predicate() { return }
            await Task.yield()
        }
        XCTFail("Timed out waiting for asynchronous state")
    }
}

private actor GenerationActivityRecorder {
    private var active = 0
    private var maximum = 0
    private var recordedEvents: [String] = []

    var activeCount: Int { active }

    var snapshot: (maximumActive: Int, events: [String]) {
        (maximum, recordedEvents)
    }

    func start(_ name: String) {
        active += 1
        maximum = max(maximum, active)
        recordedEvents.append("start:\(name)")
    }

    func finish(_ name: String) {
        recordedEvents.append("finish:\(name)")
        active -= 1
    }
}

private func nonCancellableYieldPass() async {
    for _ in 0..<100 {
        await Task.yield()
    }
}

private func waitForCancellation() async throws {
    while true {
        try Task.checkCancellation()
        await Task.yield()
    }
}
