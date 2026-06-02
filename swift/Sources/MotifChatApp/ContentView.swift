import MotifKit
import SwiftUI
import UniformTypeIdentifiers

struct ContentView: View {
    @StateObject private var store = ChatStore()
    @State private var selectedPanel: SidebarPanel? = .chat

    var body: some View {
        NavigationSplitView {
            SidebarView(store: store, selectedPanel: $selectedPanel)
                .navigationTitle("Motif")
                .toolbar {
                    Button(action: store.newChat) {
                        Label("New Chat", systemImage: "square.and.pencil")
                    }
                }
        } detail: {
            switch selectedPanel {
            case .runtime:
                RuntimeView(store: store)
            case .chat, .none:
                ChatView(store: store)
            }
        }
        .frame(minWidth: 860, minHeight: 620)
        // Translucent window: the desktop behind the window shows through and
        // blurs (NSVisualEffectView .behindWindow), giving the Liquid Glass
        // chrome real, live content to refract — no fake gradient.
        .background(VisualEffectBackground().ignoresSafeArea())
        .onReceive(NotificationCenter.default.publisher(for: .newMotifChatRequested)) { _ in
            store.newChat()
            selectedPanel = .chat
        }
    }
}

private enum SidebarPanel: Hashable {
    case chat
    case runtime
}

// MARK: - Sidebar (Part 4: conversation history)

private struct SidebarView: View {
    @ObservedObject var store: ChatStore
    @Binding var selectedPanel: SidebarPanel?

    var body: some View {
        List(selection: $selectedPanel) {
            Section {
                Label("Chat", systemImage: "bubble.left.and.bubble.right")
                    .tag(SidebarPanel.chat)
                Label("Runtime", systemImage: "speedometer")
                    .tag(SidebarPanel.runtime)
            }

            Section {
                if store.conversations.isEmpty {
                    Text("No saved conversations yet")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                } else {
                    ForEach(sortedConversations) { conversation in
                        ConversationRow(
                            conversation: conversation,
                            isActive: conversation.id == store.activeConversationID
                        )
                        .contentShape(Rectangle())
                        .onTapGesture {
                            store.selectConversation(conversation.id)
                            selectedPanel = .chat
                        }
                        .contextMenu {
                            Button(role: .destructive) {
                                store.deleteConversation(conversation.id)
                            } label: {
                                Label("Delete", systemImage: "trash")
                            }
                        }
                    }
                    .onDelete { offsets in
                        for index in offsets {
                            store.deleteConversation(sortedConversations[index].id)
                        }
                    }
                }
            } header: {
                HStack {
                    Text("History")
                    Spacer()
                    Button {
                        store.newChat()
                        selectedPanel = .chat
                    } label: {
                        Label("New", systemImage: "plus")
                            .labelStyle(.iconOnly)
                    }
                    .buttonStyle(.borderless)
                }
            }
        }
    }

    private var sortedConversations: [MotifConversation] {
        store.conversations.sorted { $0.updatedAt > $1.updatedAt }
    }
}

private struct ConversationRow: View {
    let conversation: MotifConversation
    let isActive: Bool

    var body: some View {
        VStack(alignment: .leading, spacing: 2) {
            Text(conversation.title)
                .lineLimit(1)
                .fontWeight(isActive ? .semibold : .regular)
            Text(conversation.updatedAt, format: .relative(presentation: .named))
                .font(.caption2)
                .foregroundStyle(.secondary)
        }
        .padding(.vertical, 2)
    }
}

// MARK: - Chat

private struct ChatView: View {
    @ObservedObject var store: ChatStore

    private var visibleMessages: [MotifChatMessage] {
        store.messages.filter { $0.role != .system }
    }

    var body: some View {
        ZStack(alignment: .bottom) {
            ScrollViewReader { proxy in
                ScrollView {
                    if visibleMessages.isEmpty {
                        EmptyChatState()
                            .frame(maxWidth: .infinity, minHeight: 420)
                    } else {
                        LazyVStack(alignment: .leading, spacing: 16) {
                            ForEach(visibleMessages) { message in
                                MessageBubble(
                                    message: message,
                                    isStreaming: store.isGenerating
                                        && message.id == visibleMessages.last?.id
                                        && message.role == .assistant
                                )
                                .id(message.id)
                            }

                            if !store.capturedReasoning.isEmpty {
                                ReasoningDisclosure(text: store.capturedReasoning)
                            }
                        }
                        .padding(20)
                        // leave room so the floating glass input bar doesn't cover
                        // the last message.
                        .padding(.bottom, 96)
                    }
                }
                // Let the behind-window vibrancy show through the transcript.
                .scrollContentBackground(.hidden)
                .onChange(of: store.messages) { _, messages in
                    if let last = messages.last {
                        withAnimation(.easeOut(duration: 0.2)) {
                            proxy.scrollTo(last.id, anchor: .bottom)
                        }
                    }
                }
            }

            // The input bar floats over the transcript so the Liquid Glass has
            // real (scrolling, tinted) content behind it to refract — glass over
            // a flat solid color is nearly invisible by design.
            InputBar(store: store)
        }
        .navigationTitle("Motif Chat")
    }
}

private struct EmptyChatState: View {
    var body: some View {
        VStack(spacing: 12) {
            Image(systemName: "sparkles")
                .font(.system(size: 40))
                .foregroundStyle(.secondary)
            Text("Ask Motif anything")
                .font(.title2)
                .fontWeight(.medium)
            Text("Start a conversation below. Your chats are saved in the sidebar.")
                .font(.callout)
                .foregroundStyle(.secondary)
                .multilineTextAlignment(.center)
        }
        .padding()
    }
}

private struct ReasoningDisclosure: View {
    let text: String

    var body: some View {
        DisclosureGroup {
            Text(text)
                .font(.system(.body, design: .monospaced))
                .textSelection(.enabled)
                .frame(maxWidth: .infinity, alignment: .leading)
                .padding(.top, 6)
        } label: {
            Label("Captured reasoning", systemImage: "brain")
                .font(.callout)
        }
        .padding(12)
        .background(.ultraThinMaterial, in: RoundedRectangle(cornerRadius: 12, style: .continuous))
        .overlay(
            RoundedRectangle(cornerRadius: 12, style: .continuous)
                .stroke(.quaternary, lineWidth: 1)
        )
    }
}

private struct InputBar: View {
    @ObservedObject var store: ChatStore

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            if let notice = store.contextNotice {
                Label(notice, systemImage: "scissors")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
            if let lastError = store.lastError {
                Text(lastError)
                    .font(.caption)
                    .foregroundStyle(.red)
            }

            HStack(alignment: .bottom, spacing: 10) {
                TextField("Ask Motif…", text: $store.prompt, axis: .vertical)
                    .textFieldStyle(.plain)
                    .lineLimit(2...8)
                    .onSubmit { store.send() }
                    .accessibilityLabel("Message Motif")
                    .accessibilityIdentifier("motif.chat.input")

                if store.isGenerating {
                    Button(role: .cancel, action: store.cancel) {
                        Label("Stop", systemImage: "stop.fill").labelStyle(.iconOnly)
                    }
                    .motifGlassButton()
                    .tint(.red)
                    .keyboardShortcut(".", modifiers: [.command])
                    .accessibilityIdentifier("motif.chat.stop")
                } else {
                    Button(action: store.send) {
                        Label("Send", systemImage: "arrow.up").labelStyle(.iconOnly)
                    }
                    .motifGlassButton()
                    .tint(.accentColor)
                    .keyboardShortcut(.return, modifiers: [.command])
                    .disabled(store.prompt.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)
                    .accessibilityIdentifier("motif.chat.send")
                }
            }
            // Floating glass chrome (Part 5) — the input bar is container chrome,
            // not content. Inset padding + a rounded surface let the glass read
            // as a distinct floating bar over the scrolling transcript behind it.
            .padding(.horizontal, 14)
            .padding(.vertical, 10)
            .motifGlassSurface(cornerRadius: 22)
            .motifGlassGroup()
        }
        .padding(16)
    }
}

// MARK: - Message bubble (Part 1)

private struct MessageBubble: View {
    let message: MotifChatMessage
    let isStreaming: Bool

    @State private var isHovering = false
    @State private var didCopy = false

    private var isUser: Bool { message.role == .user }

    var body: some View {
        HStack(alignment: .top) {
            if isUser { Spacer(minLength: 80) }

            VStack(alignment: .leading, spacing: 6) {
                HStack(spacing: 6) {
                    Text(message.role.rawValue.capitalized)
                        .font(.caption)
                        .foregroundStyle(.secondary)
                    if isHovering {
                        Button {
                            copyToPasteboard(message.content)
                            didCopy = true
                            DispatchQueue.main.asyncAfter(deadline: .now() + 1.2) { didCopy = false }
                        } label: {
                            Image(systemName: didCopy ? "checkmark" : "doc.on.doc")
                                .font(.caption2)
                        }
                        .buttonStyle(.borderless)
                        .help("Copy message")
                    }
                }

                if message.content.isEmpty && !isStreaming {
                    Text("…")
                        .foregroundStyle(.secondary)
                } else {
                    HStack(alignment: .bottom, spacing: 4) {
                        if isStreaming {
                            StreamingCaret()
                        }
                        MarkdownMessageView(content: message.content)
                    }
                }
            }
            .padding(12)
            .background(bubbleBackground, in: RoundedRectangle(cornerRadius: 14, style: .continuous))
            .frame(maxWidth: 680, alignment: isUser ? .trailing : .leading)

            if message.role == .assistant { Spacer(minLength: 80) }
        }
        .frame(maxWidth: .infinity, alignment: isUser ? .trailing : .leading)
        .onHover { isHovering = $0 }
    }

    private var bubbleBackground: AnyShapeStyle {
        // Materials only (per Apple, never glass on message content).
        isUser ? AnyShapeStyle(.blue.opacity(0.16)) : AnyShapeStyle(.ultraThinMaterial)
    }
}

/// Subtle blinking caret shown at the end of the streaming assistant message.
private struct StreamingCaret: View {
    @State private var visible = true

    var body: some View {
        Text("▌")
            .foregroundStyle(.secondary)
            .opacity(visible ? 1 : 0.15)
            .onAppear {
                withAnimation(.easeInOut(duration: 0.6).repeatForever(autoreverses: true)) {
                    visible.toggle()
                }
            }
    }
}

// MARK: - Runtime

private struct RuntimeView: View {
    @ObservedObject var store: ChatStore
    @State private var showingModelDirectoryImporter = false

    var body: some View {
        Form {
            Section("Backend") {
                Picker("Chat path", selection: $store.backendMode) {
                    ForEach(MotifChatBackendMode.allCases) { mode in
                        Label(mode.label, systemImage: mode.systemImage).tag(mode)
                    }
                }

                switch store.backendMode {
                case .openAICompatible:
                    TextField("OpenAI-compatible endpoint", text: $store.endpoint)
                        .textFieldStyle(.roundedBorder)
                    TextField("Model ID", text: $store.model)
                        .textFieldStyle(.roundedBorder)

                case .nativeMLX:
                    HStack {
                        TextField("Converted MLX model directory", text: $store.nativeModelDirectory)
                            .textFieldStyle(.roundedBorder)
                        Button("Choose…") {
                            showingModelDirectoryImporter = true
                        }
                    }
                    Text("Build with `MOTIFKIT_ENABLE_MLX=1` and point this at an HF→MLX converted Motif checkpoint directory.")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }

                Picker("Thinking", selection: $store.thinkMode) {
                    ForEach(MotifThinkMode.allCases, id: \.self) { mode in
                        Text(mode.rawValue.capitalized).tag(mode)
                    }
                }
            }

            Section("App status") {
                Label("Active chat path: \(store.backendMode.label)", systemImage: store.backendMode.systemImage)
                Label("Runtime status: \(store.runtimeStatus)", systemImage: statusIcon)
                Label(
                    store.nativeMLXCompiledIn
                        ? "Native in-process MLX generation is compiled into this build"
                        : "Native in-process MLX generation requires MOTIFKIT_ENABLE_MLX=1",
                    systemImage: store.nativeMLXCompiledIn ? "checkmark.seal" : "shippingbox"
                )
                Text("Use the OpenAI-compatible endpoint for a Python or Swift local server, or rebuild the app with the MotifKitMLX overlay to stream directly from a converted checkpoint without a server hop.")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }

            Section("Generation") {
                Stepper("Max tokens: \(store.maxTokens)", value: $store.maxTokens, in: 1...8192, step: 64)
                Slider(value: $store.temperature, in: 0...2) {
                    Text("Temperature")
                }
                Text("Temperature: \(store.temperature, specifier: "%.2f")")
                    .foregroundStyle(.secondary)
            }

            Section("Context") {
                Picker("Compaction", selection: $store.compactionMode) {
                    ForEach(MotifCompactionMode.allCases) { mode in
                        Text(mode.label).tag(mode)
                    }
                }
                Stepper(
                    "Context budget: \(store.contextTokenBudget) tokens",
                    value: $store.contextTokenBudget,
                    in: 1000...131072,
                    step: 1000
                )
                Text("Approx. budget guard (≈ characters ÷ 4). When a conversation exceeds this, the oldest turns are dropped while the system prompt and latest message are always kept.")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }

            Section("Native MLX status") {
                Label("MotifKitMLX overlay is gated behind MOTIFKIT_ENABLE_MLX=1 for lightweight default builds", systemImage: "shippingbox")
                Label("Native path: tokenizer/chat template, checkpoint loading, q4 cache, and custom Metal", systemImage: "checklist")
                Label("Remaining: speculative decoding", systemImage: "wrench.and.screwdriver")
                Label("Evidence: docs/benchmarks/swift-python-hard-parity-20260526T091532Z.md", systemImage: "speedometer")
            }

            Section("Settings") {
                Button("Reset runtime settings", role: .destructive) {
                    store.resetRuntimeSettings()
                }
            }
        }
        .formStyle(.grouped)
        .padding()
        .navigationTitle("Runtime")
        .fileImporter(
            isPresented: $showingModelDirectoryImporter,
            allowedContentTypes: [.folder],
            allowsMultipleSelection: false
        ) { result in
            switch result {
            case .success(let urls):
                guard let url = urls.first else { return }
                // Scoped access here covers only the synchronous bookmark *creation*
                // inside `selectNativeModelDirectory`. The checkpoint *load* later
                // re-acquires its own scoped access by resolving that bookmark, so it
                // is intentionally not held open past this callback.
                let isSecurityScoped = url.startAccessingSecurityScopedResource()
                defer {
                    if isSecurityScoped { url.stopAccessingSecurityScopedResource() }
                }
                store.selectNativeModelDirectory(url)
            case .failure(let error):
                store.lastError = error.localizedDescription
            }
        }
    }

    private var statusIcon: String {
        switch store.runtimeStatus {
        case "Idle":
            "checkmark.circle"
        case "Error":
            "exclamationmark.triangle"
        case "Cancelled":
            "stop.circle"
        default:
            "hourglass"
        }
    }
}
