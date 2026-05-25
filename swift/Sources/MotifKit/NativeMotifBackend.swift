public struct NativeMotifBackend: MotifChatBackend {
    public init() {}

    public func streamResponse(
        messages: [MotifChatMessage],
        parameters: MotifGenerationParameters
    ) -> AsyncThrowingStream<MotifGenerationEvent, any Error> {
        AsyncThrowingStream { continuation in
            continuation.finish(throwing: MotifBackendError.nativeBackendUnavailable(
                "MotifKitMLX must port Model, grouped differential attention, KV cache, and custom Metal kernels before native generation can run."
            ))
        }
    }
}
