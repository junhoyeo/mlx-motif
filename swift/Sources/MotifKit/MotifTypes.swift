import Foundation

public enum MotifRole: String, Codable, Sendable, CaseIterable {
    case system
    case user
    case assistant
}

public struct MotifChatMessage: Identifiable, Codable, Equatable, Sendable {
    public var id: UUID
    public var role: MotifRole
    public var content: String
    public var createdAt: Date

    public init(
        id: UUID = UUID(),
        role: MotifRole,
        content: String,
        createdAt: Date = Date()
    ) {
        self.id = id
        self.role = role
        self.content = content
        self.createdAt = createdAt
    }

    public static func system(_ content: String) -> Self { .init(role: .system, content: content) }
    public static func user(_ content: String) -> Self { .init(role: .user, content: content) }
    public static func assistant(_ content: String) -> Self { .init(role: .assistant, content: content) }
}

public enum MotifThinkMode: String, Codable, Sendable, CaseIterable {
    case visible
    case hidden
    case captured
}

public struct MotifGenerationParameters: Equatable, Sendable {
    public var model: String
    public var maxTokens: Int
    public var temperature: Double
    public var thinkMode: MotifThinkMode

    public init(
        model: String = "motif",
        maxTokens: Int = 512,
        temperature: Double = 0.6,
        thinkMode: MotifThinkMode = .hidden
    ) {
        self.model = model
        self.maxTokens = maxTokens
        self.temperature = temperature
        self.thinkMode = thinkMode
    }
}

public enum MotifGenerationEvent: Equatable, Sendable {
    case text(String)
    case reasoning(String)
    case completed
}

public protocol MotifChatBackend: Sendable {
    func streamResponse(
        messages: [MotifChatMessage],
        parameters: MotifGenerationParameters
    ) -> AsyncThrowingStream<MotifGenerationEvent, any Error>
}

public enum MotifBackendError: Error, LocalizedError, Equatable, Sendable {
    case nativeBackendUnavailable(String)
    case invalidEndpoint(URL)
    case httpStatus(Int)
    case malformedServerEvent(String)

    public var errorDescription: String? {
        switch self {
        case .nativeBackendUnavailable(let detail):
            "Native Motif MLX backend is not available yet: \(detail)"
        case .invalidEndpoint(let url):
            "Invalid Motif endpoint: \(url.absoluteString)"
        case .httpStatus(let status):
            "Motif endpoint returned HTTP \(status)"
        case .malformedServerEvent(let event):
            "Malformed streaming event from Motif endpoint: \(event)"
        }
    }
}

public struct MotifModelConfiguration: Codable, Equatable, Sendable {
    public var modelType: String
    public var hiddenSize: Int
    public var numHiddenLayers: Int
    public var intermediateSize: Int
    public var numAttentionHeads: Int
    public var numKeyValueHeads: Int
    public var vocabSize: Int
    public var rmsNormEps: Double
    public var ropeTheta: Double
    public var maxPositionEmbeddings: Int
    public var headDim: Int?
    public var numNoiseHeads: Int?
    public var kRatio: Int
    public var attnRMSNormEps: Double
    public var hiddenActivation: String
    public var useBias: Bool
    public var fusedRope: Bool

    public var isGroupedDifferentialAttention: Bool { numNoiseHeads != nil }

    public init(
        modelType: String = "motif",
        hiddenSize: Int,
        numHiddenLayers: Int,
        intermediateSize: Int,
        numAttentionHeads: Int,
        numKeyValueHeads: Int,
        vocabSize: Int,
        rmsNormEps: Double = 1e-6,
        ropeTheta: Double = 10_000,
        maxPositionEmbeddings: Int = 8_192,
        headDim: Int? = nil,
        numNoiseHeads: Int? = nil,
        kRatio: Int = 1,
        attnRMSNormEps: Double = 1e-5,
        hiddenActivation: String = "poly_norm",
        useBias: Bool = false,
        fusedRope: Bool = false
    ) {
        self.modelType = modelType
        self.hiddenSize = hiddenSize
        self.numHiddenLayers = numHiddenLayers
        self.intermediateSize = intermediateSize
        self.numAttentionHeads = numAttentionHeads
        self.numKeyValueHeads = numKeyValueHeads
        self.vocabSize = vocabSize
        self.rmsNormEps = rmsNormEps
        self.ropeTheta = ropeTheta
        self.maxPositionEmbeddings = maxPositionEmbeddings
        self.headDim = headDim
        self.numNoiseHeads = numNoiseHeads
        self.kRatio = kRatio
        self.attnRMSNormEps = attnRMSNormEps
        self.hiddenActivation = hiddenActivation
        self.useBias = useBias
        self.fusedRope = fusedRope
    }
}

public struct MotifRuntimeFeatureFlags: Equatable, Sendable {
    public var dualVAttention: Bool
    public var fourSlotCache: String?
    public var quantizedSDPA: Bool
    public var disableCustomKernels: Bool

    public init(
        dualVAttention: Bool = true,
        fourSlotCache: String? = nil,
        quantizedSDPA: Bool = true,
        disableCustomKernels: Bool = false
    ) {
        self.dualVAttention = dualVAttention
        self.fourSlotCache = fourSlotCache
        self.quantizedSDPA = quantizedSDPA
        self.disableCustomKernels = disableCustomKernels
    }
}
