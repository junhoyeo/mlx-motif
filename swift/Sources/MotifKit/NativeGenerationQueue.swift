import Foundation

/// Serializes asynchronous native-generation producers, including their
/// cancellation cleanup. A replacement operation is queued behind the prior
/// operation's *actual* completion rather than merely behind its consumer
/// stream ending.
public actor NativeGenerationQueue {
    private var tail: (id: UUID, task: Task<Void, Never>)?

    public init() {}

    public func run<T: Sendable>(
        _ operation: @escaping @Sendable () async throws -> T
    ) async throws -> T {
        let previous = tail?.task
        let operationID = UUID()
        let resultTask = Task<Result<T, any Error>, Never> {
            if let previous {
                await previous.value
            }
            do {
                try Task.checkCancellation()
                return .success(try await operation())
            } catch {
                return .failure(error)
            }
        }
        let completionTask = Task<Void, Never> {
            _ = await resultTask.value
        }
        tail = (operationID, completionTask)

        let result = await withTaskCancellationHandler {
            await resultTask.value
        } onCancel: {
            resultTask.cancel()
        }

        if tail?.id == operationID {
            tail = nil
        }
        return try result.get()
    }
}
