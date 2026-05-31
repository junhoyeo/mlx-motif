public struct NativeMotifBackend: MotifChatBackend {
    public var configuration: MotifModelConfiguration?
    public var featureFlags: MotifRuntimeFeatureFlags

    public init(
        configuration: MotifModelConfiguration? = nil,
        featureFlags: MotifRuntimeFeatureFlags = .init()
    ) {
        self.configuration = configuration
        self.featureFlags = featureFlags
    }

    public func streamResponse(
        messages: [MotifChatMessage],
        parameters: MotifGenerationParameters
    ) -> AsyncThrowingStream<MotifGenerationEvent, any Error> {
        AsyncThrowingStream { continuation in
            let configDetail = configuration.map {
                "Decoded \($0.modelType) config (\($0.attentionVariant.rawValue), hidden_size=\($0.hiddenSize), layers=\($0.numHiddenLayers)); "
            } ?? ""
            continuation.finish(throwing: MotifBackendError.nativeBackendUnavailable(
                "\(configDetail)MotifKitMLX must port Model, grouped differential attention, KV cache, and custom Metal kernels before native generation can run."
            ))
        }
    }
}
