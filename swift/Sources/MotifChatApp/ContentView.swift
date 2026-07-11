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
                // Make the toolbar chrome read as a distinct glass/material
                // surface over the translucent window (on macOS 26 the toolbar
                // adopts Liquid Glass; older OSes get a bar material).
                .toolbarBackground(.visible, for: .windowToolbar)
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
        // `.hudWindow` reads more translucent than `.underWindowBackground`, so
        // the desktop shows through the chrome (the glass has content to refract).
        .background(VisualEffectBackground(material: .hudWindow).ignoresSafeArea())
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
                    // Keep the + off the sidebar's right edge.
                    .padding(.trailing, 8)
                }
            }
        }
        // Let the glass chrome behind the list show through (the List's own
        // opaque background would otherwise hide it).
        .scrollContentBackground(.hidden)
        // Sidebar is structural chrome, not message content, so glass is allowed
        // here (Apple's rule). Grouped so it blends with the toolbar glass above.
        .background(EmptyView().motifGlassChrome().ignoresSafeArea())
        .motifGlassGroup()
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
    @State private var showingModelDirImporter = false

    private var visibleMessages: [MotifChatMessage] {
        store.messages.filter { $0.role != .system }
    }

    var body: some View {
        chatSurface
            .navigationTitle("Motif Chat")
            // The backend selector + live readout live in the unified window
            // toolbar (one top bar) instead of a second header row below it —
            // otherwise the visible window toolbar reserves an empty strip above
            // a separate header bar, which reads as stray top padding.
            .toolbar {
                // The backend/model selector identifies *what you're talking to*
                // — the view's context — so per the HIG (Toolbars) it leads,
                // like Xcode's scheme selector, rather than sitting centered.
                ToolbarItem(placement: .navigation) {
                    BackendMenu(store: store, showingImporter: $showingModelDirImporter)
                }
                // Separate ToolbarItems (NOT one ToolbarItemGroup): a group merges
                // its children into a single shared capsule, which broke the
                // interior padding whenever the tok/s readout appeared next to
                // the meter. One item per pill, each with matched padding.
                // Order: context gauge first, live speed to its RIGHT.
                ToolbarItem(placement: .primaryAction) {
                    ContextMeter(store: store)
                }
                if store.isGenerating {
                    ToolbarItem(placement: .primaryAction) {
                        HStack(spacing: 5) {
                            Image(systemName: "bolt.fill").font(.caption2).foregroundStyle(.yellow)
                            Text("\(store.liveTokensPerSecond, specifier: "%.0f") tok/s · \(store.liveTokenEstimate) tok")
                                .font(.caption)
                                .foregroundStyle(.secondary)
                        }
                        .padding(.horizontal, 16)
                        .padding(.vertical, 5)
                        .accessibilityIdentifier("motif.chat.tokrate")
                    }
                }
            }
            // Give the unified toolbar a visible (glass on macOS 26) background so
            // transcript content scrolling up behind it is masked, not bleeding
            // through as raw text above the header controls.
            .toolbarBackground(.visible, for: .windowToolbar)
            .fileImporter(
                isPresented: $showingModelDirImporter,
                allowedContentTypes: [.folder],
                allowsMultipleSelection: false
            ) { result in
                switch result {
                case .success(let urls):
                    guard let url = urls.first else { return }
                    let scoped = url.startAccessingSecurityScopedResource()
                    defer { if scoped { url.stopAccessingSecurityScopedResource() } }
                    store.selectNativeModelDirectory(url)
                case .failure(let error):
                    store.lastError = error.localizedDescription
                }
            }
    }

    // Stable id for an invisible element pinned to the very bottom of the
    // transcript. Scrolling to it (rather than to the last message) keeps the
    // view pinned even as trailing content — a streaming message or the captured
    // reasoning disclosure — grows.
    private static let bottomAnchor = "motif.chat.bottomAnchor"

    private var chatSurface: some View {
        ScrollViewReader { proxy in
            ScrollView {
                if visibleMessages.isEmpty {
                    EmptyChatState(store: store)
                        .frame(maxWidth: .infinity, minHeight: 420)
                } else {
                    LazyVStack(alignment: .leading, spacing: 16) {
                        ForEach(visibleMessages) { message in
                            MessageBubble(
                                store: store,
                                message: message,
                                isStreaming: store.isGenerating
                                    && message.id == visibleMessages.last?.id
                                    && message.role == .assistant,
                                isLast: message.id == visibleMessages.last?.id
                            )
                            .id(message.id)
                        }

                        if !store.capturedReasoning.isEmpty {
                            ReasoningDisclosure(text: store.capturedReasoning)
                        }

                        // Bottom scroll target. The input bar's height is handled
                        // by `.safeAreaInset` below, so no magic bottom padding is
                        // needed here — content is measured, never guessed.
                        Color.clear
                            .frame(height: 1)
                            .id(Self.bottomAnchor)
                    }
                    .padding(20)
                }
            }
            // Let the behind-window vibrancy show through the transcript.
            .scrollContentBackground(.hidden)
            // Start pinned to the newest message (scrollview skill: Controlling
            // Scroll Position).
            .defaultScrollAnchor(.bottom)
            // Fade transcript content cleanly under the toolbar / input bar glass
            // (Liquid Glass: scrollEdgeEffectStyle).
            .motifScrollEdgeSoft()
            // Follow the transcript to the bottom both when a new message is
            // appended AND while a single assistant message streams token by
            // token — `liveTokenEstimate` ticks on every decoded token, so the
            // view tracks the growing text instead of only reacting to the
            // messages array changing.
            .onChange(of: store.messages) { _, _ in
                scrollToBottom(proxy, animated: true)
            }
            .onChange(of: store.liveTokenEstimate) { _, _ in
                scrollToBottom(proxy, animated: false)
            }
            // The input bar is a real bottom inset of the scroll view: content is
            // inset by the bar's measured height (never hidden under it), yet the
            // scroll content still passes *behind* the floating glass bar so the
            // Liquid Glass has live, scrolling content to refract.
            .safeAreaInset(edge: .bottom, spacing: 0) {
                InputBar(store: store)
            }
        }
    }

    /// Pin the transcript to its bottom anchor. Animated for discrete message
    /// changes; unanimated while streaming so rapid per-token updates don't stack
    /// competing animations (which reads as jitter).
    private func scrollToBottom(_ proxy: ScrollViewProxy, animated: Bool) {
        guard !visibleMessages.isEmpty else { return }
        if animated {
            withAnimation(.easeOut(duration: 0.18)) {
                proxy.scrollTo(Self.bottomAnchor, anchor: .bottom)
            }
        } else {
            proxy.scrollTo(Self.bottomAnchor, anchor: .bottom)
        }
    }
}

private struct EmptyChatState: View {
    @ObservedObject var store: ChatStore

    private let examples = [
        "Think step by step: why is the sky blue?",
        "What is 37 × 41?",
        "Write a Swift function that reverses a linked list.",
    ]

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

            VStack(spacing: 8) {
                ForEach(examples, id: \.self) { example in
                    Button {
                        store.prompt = example
                        store.send()
                    } label: {
                        HStack {
                            Text(example).lineLimit(1)
                            Spacer(minLength: 0)
                            Image(systemName: "arrow.up.right").font(.caption2).foregroundStyle(.tertiary)
                        }
                        .frame(maxWidth: 380, alignment: .leading)
                    }
                    .buttonStyle(.bordered)
                    .controlSize(.large)
                    .disabled(store.isGenerating)
                }
            }
            .padding(.top, 10)
            .accessibilityIdentifier("motif.chat.examples")
        }
        .padding()
    }
}

private struct ReasoningDisclosure: View {
    let text: String

    var body: some View {
        DisclosureGroup {
            Text(text)
                .font(.body)
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
            if store.isGenerating {
                HStack(spacing: 6) {
                    ProgressView()
                        .controlSize(.small)
                    Text("Generating…")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
                .accessibilityIdentifier("motif.chat.generating")
            }
            if let notice = store.contextNotice {
                HStack(spacing: 6) {
                    Label(notice, systemImage: "scissors")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                    Spacer()
                    Text("Budget \(store.contextTokenBudget) tok")
                        .font(.caption2)
                        .foregroundStyle(.tertiary)
                }
            }
            if let lastError = store.lastError {
                Text(lastError)
                    .font(.caption)
                    .foregroundStyle(.red)
            }

            // Claude-style layout: the text field spans the full bar width, and
            // the controls live in a row UNDER it — mode icons on the left,
            // send/stop on the right — instead of crowding the field inline.
            VStack(alignment: .leading, spacing: 10) {
                TextField("Ask Motif…", text: $store.prompt, axis: .vertical)
                    .textFieldStyle(.plain)
                    .lineLimit(1...8)
                    .onSubmit { store.send() }
                    .accessibilityLabel("Message Motif")
                    .accessibilityIdentifier("motif.chat.input")

                HStack(spacing: 14) {
                    ThinkModeMenu(store: store)
                    ToolsToggle(store: store)

                    Spacer()

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
            }
            // Floating glass chrome (Part 5) — the input bar is container chrome,
            // not content. Inset padding + a rounded surface let the glass read
            // as a distinct floating bar over the scrolling transcript behind it.
            .padding(.horizontal, 16)
            .padding(.vertical, 12)
            .motifGlassSurface(cornerRadius: 22)
            .motifGlassGroup()
        }
        .padding(16)
    }
}

// MARK: - Message bubble (Part 1)

private struct MessageBubble: View {
    @ObservedObject var store: ChatStore
    let message: MotifChatMessage
    let isStreaming: Bool
    let isLast: Bool

    @State private var isHovering = false
    @State private var didCopy = false

    private var isUser: Bool { message.role == .user }
    private var isAssistant: Bool { message.role == .assistant }

    var body: some View {
        // ONE alignment mechanism (hstack/vstack + ui-patterns skills): the
        // bubble hugs its content up to a consistent max width, then a single
        // trailing `.frame(maxWidth: .infinity, alignment:)` positions it on the
        // correct side of the transcript — no HStack+Spacer *and* outer frame
        // fighting each other.
        VStack(alignment: bubbleAlignment, spacing: 6) {
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

            if let call = store.toolCalls[message.id] {
                ToolCallCard(call: call)
            } else if message.role == .tool {
                // Executed-tool result feeding the next round — compact and
                // distinct from prose so the loop is visible at a glance.
                HStack(alignment: .firstTextBaseline, spacing: 6) {
                    Image(systemName: "checkmark.circle.fill")
                        .foregroundStyle(.green)
                        .font(.callout)
                    Text(message.content)
                        .font(.callout)
                        .textSelection(.enabled)
                }
                .accessibilityIdentifier("motif.chat.toolresult")
            } else if isStreaming && message.content.contains("\"tool_call\"") {
                HStack(spacing: 6) {
                    Image(systemName: "wrench.and.screwdriver.fill")
                        .foregroundStyle(.orange)
                    Text("calling a tool…")
                        .foregroundStyle(.secondary)
                }
                .font(.callout)
            } else if message.content.isEmpty && !isStreaming {
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

            // Dev-tool decode metrics under finished assistant turns.
            if isAssistant, let m = store.metrics[message.id] {
                MetricsLine(metrics: m)
            }

            // Per-message actions on assistant turns. Revealed on hover (kept
            // mounted but hidden so layout doesn't jump) and never while the
            // message is mid-stream. Materials/plain controls only — no glass
            // on message content.
            if isAssistant && !isStreaming {
                actionsRow
                    .opacity(isHovering ? 1 : 0)
                    .allowsHitTesting(isHovering)
            }
        }
        .padding(12)
        .background(bubbleBackground, in: RoundedRectangle(cornerRadius: 14, style: .continuous))
        // Consistent max bubble width; the bubble hugs shorter content.
        .frame(maxWidth: 680, alignment: isUser ? .trailing : .leading)
        // Single alignment step: push the bubble to its side of the transcript.
        .frame(maxWidth: .infinity, alignment: isUser ? .trailing : .leading)
        .onHover { isHovering = $0 }
    }

    /// Content alignment inside the bubble: user turns hang from the trailing
    /// edge, assistant turns from the leading edge.
    private var bubbleAlignment: HorizontalAlignment { isUser ? .trailing : .leading }

    private var actionsRow: some View {
        HStack(spacing: 12) {
            Button {
                copyToPasteboard(message.content)
                didCopy = true
                DispatchQueue.main.asyncAfter(deadline: .now() + 1.2) { didCopy = false }
            } label: {
                Label(didCopy ? "Copied" : "Copy", systemImage: didCopy ? "checkmark" : "doc.on.doc")
                    .font(.caption2)
            }
            .help("Copy message")

            // Regenerate only applies to the latest assistant turn — re-running an
            // older turn would discard everything after it.
            if isLast {
                Button {
                    store.regenerateLast()
                } label: {
                    Label("Regenerate", systemImage: "arrow.clockwise")
                        .font(.caption2)
                }
                .help("Re-run the last prompt")
            }

            Button(role: .destructive) {
                store.deleteMessage(message.id)
            } label: {
                Label("Delete", systemImage: "trash")
                    .font(.caption2)
            }
            .help("Delete this message")
        }
        .buttonStyle(.borderless)
        .labelStyle(.titleAndIcon)
        .foregroundStyle(.secondary)
        // Actions are disabled mid-generation to match ChatStore's guards.
        .disabled(store.isGenerating)
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
        // Kill the grouped form's own top inset: stacked under the reserved
        // toolbar strip it read as a large dead band above 'Backend'. With a
        // zero top margin the first section sits flush under the toolbar,
        // matching the chat pane's vertical rhythm.
        .contentMargins(.top, 0, for: .scrollContent)
        // Let the sections float on the translucent window like the chat pane.
        .scrollContentBackground(.hidden)
        .navigationTitle("Runtime")
        .toolbar {
            // Give the otherwise-empty toolbar strip a purpose on this tab —
            // a leading title pill where the chat tab has its model selector.
            ToolbarItem(placement: .navigation) {
                Label("Runtime", systemImage: "speedometer")
                    // Toolbar labels default to icon-only; keep the title.
                    .labelStyle(.titleAndIcon)
                    .font(.subheadline).fontWeight(.semibold)
                    .padding(.horizontal, 16)
                    .padding(.vertical, 5)
            }
        }
        // Mask the toolbar strip like the chat tab — without this the empty
        // unified-toolbar band above the form reads as dead gray space.
        .toolbarBackground(.visible, for: .windowToolbar)
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

// MARK: - Dev-tool chrome (header selector, meters, tool cards)

/// Header above the transcript: backend/model selector on the left (dev-tool —
/// which model you're talking to is never hidden), live decode readout and the
/// context meter on the right.
private struct BackendMenu: View {
    @ObservedObject var store: ChatStore
    @Binding var showingImporter: Bool

    var body: some View {
        HStack(spacing: 12) {
            Menu {
                Section("Backend") {
                    Button {
                        store.backendMode = .nativeMLX
                    } label: {
                        Label(
                            "Native MLX checkpoint",
                            systemImage: store.backendMode == .nativeMLX ? "checkmark" : "cpu"
                        )
                    }
                    Button {
                        store.backendMode = .openAICompatible
                    } label: {
                        Label(
                            "Endpoint · \(store.endpoint)",
                            systemImage: store.backendMode == .openAICompatible ? "checkmark" : "network"
                        )
                    }
                }
                if store.backendMode == .nativeMLX {
                    Section("Checkpoint") {
                        ForEach(store.discoveredModelDirectories, id: \.self) { dir in
                            Button {
                                store.selectNativeModelDirectoryPath(dir)
                            } label: {
                                Label(
                                    (dir as NSString).lastPathComponent,
                                    systemImage: dir == store.nativeModelDirectory ? "checkmark" : "folder"
                                )
                            }
                        }
                        Button {
                            showingImporter = true
                        } label: {
                            Label("Choose…", systemImage: "folder.badge.plus")
                        }
                    }
                }
            } label: {
                HStack(spacing: 8) {
                    StatusDot(store: store)
                    VStack(alignment: .leading, spacing: 1) {
                        Text(store.backendDisplayName)
                            .font(.subheadline).fontWeight(.semibold).lineLimit(1)
                        Text(store.backendMode == .nativeMLX ? "native MLX" : "endpoint")
                            .font(.caption2).foregroundStyle(.secondary)
                    }
                    Image(systemName: "chevron.down").font(.caption2).foregroundStyle(.secondary)
                }
            }
            .menuStyle(.borderlessButton)
            .menuIndicator(.hidden)
            .fixedSize()
            .accessibilityIdentifier("motif.chat.model")

            // Endpoint mode edits its host + model id inline in the header — the
            // active target is a dev-tool detail that shouldn't require opening
            // the Runtime panel.
            if store.backendMode == .openAICompatible {
                HStack(spacing: 6) {
                    Image(systemName: "network").font(.caption2).foregroundStyle(.secondary)
                    TextField("endpoint", text: $store.endpoint)
                        .textFieldStyle(.plain)
                        .font(.caption)
                        .frame(minWidth: 150, maxWidth: 240)
                        .accessibilityIdentifier("motif.chat.endpoint")
                    Divider().frame(height: 12)
                    TextField("model", text: $store.model)
                        .textFieldStyle(.plain)
                        .font(.caption)
                        .frame(minWidth: 60, maxWidth: 120)
                        .accessibilityIdentifier("motif.chat.modelid")
                }
                .padding(.horizontal, 8)
                .padding(.vertical, 4)
                .background(.quaternary.opacity(0.5), in: Capsule())
            }

        }
        // Breathing room inside the toolbar's auto glass pill. Applied at the
        // ITEM ROOT (not on the Menu, not inside its label — the capsule wraps
        // the item's bounds and ignores label padding) so BOTH modes stay
        // uniformly padded: native (menu only) and endpoint (menu + inline
        // fields, which previously sat flush against the capsule's right edge).
        .padding(.horizontal, 16)
        .padding(.vertical, 5)
    }
}

/// Green idle / yellow generating / red on error.
private struct StatusDot: View {
    @ObservedObject var store: ChatStore
    var body: some View {
        Circle().fill(color).frame(width: 8, height: 8)
            .help(store.runtimeStatus)
    }
    private var color: Color {
        if store.lastError != nil { return .red }
        if store.isGenerating { return .yellow }
        return .green
    }
}

/// Thin context-usage bar: estimated transcript tokens vs the budget; amber ≥80%.
private struct ContextMeter: View {
    @ObservedObject var store: ChatStore
    var body: some View {
        let fraction = min(store.contextUsageFraction, 1.0)
        let amber = store.contextUsageFraction >= 0.8
        HStack(spacing: 6) {
            Image(systemName: "gauge.with.dots.needle.33percent")
                .font(.caption2).foregroundStyle(.secondary)
            ProgressView(value: fraction)
                .progressViewStyle(.linear)
                .frame(width: 56)
                .tint(amber ? .orange : .accentColor)
            Text("\(compact(store.currentContextTokens)) / \(compact(store.contextTokenBudget))")
                .font(.caption2)
                .foregroundStyle(amber ? .orange : .secondary)
        }
        // Breathing room inside the toolbar's auto glass pill (matches the
        // model-selector pill).
        .padding(.horizontal, 16)
        .padding(.vertical, 5)
        .help("Context usage (estimated tokens) vs budget")
        .accessibilityIdentifier("motif.chat.context")
    }
    private func compact(_ n: Int) -> String {
        n >= 1000 ? String(format: "%.1fk", Double(n) / 1000) : "\(n)"
    }
}

/// Brain-icon menu that picks the reasoning-trace mode (visible/hidden/captured).
private struct ThinkModeMenu: View {
    @ObservedObject var store: ChatStore
    var body: some View {
        Menu {
            Picker("Thinking", selection: $store.thinkMode) {
                ForEach(MotifThinkMode.allCases, id: \.self) { mode in
                    Text(mode.rawValue.capitalized).tag(mode)
                }
            }
        } label: {
            Image(systemName: "brain")
                .foregroundStyle(store.thinkMode == .hidden ? Color.secondary : Color.accentColor)
        }
        .menuStyle(.borderlessButton)
        .menuIndicator(.hidden)
        .fixedSize()
        .help("Reasoning trace: \(store.thinkMode.rawValue)")
        .accessibilityIdentifier("motif.chat.thinkmode")
    }
}

/// Toggles the two safe demo tools (get_current_time, calculator) for the turn.
private struct ToolsToggle: View {
    @ObservedObject var store: ChatStore
    var body: some View {
        Button {
            store.toolsEnabled.toggle()
        } label: {
            Image(systemName: "wrench.and.screwdriver")
                .foregroundStyle(store.toolsEnabled ? Color.accentColor : Color.secondary)
        }
        .buttonStyle(.borderless)
        .help(store.toolsEnabled
            ? "Demo tools ON (get_current_time, calculator)"
            : "Enable demo tools")
        .accessibilityIdentifier("motif.chat.tools")
    }
}

/// One-line decode readout under a finished assistant turn.
private struct MetricsLine: View {
    let metrics: MessageMetrics
    var body: some View {
        HStack(spacing: 6) {
            Image(systemName: "gauge.with.needle").font(.caption2)
            Text("\(metrics.tokensPerSecond, specifier: "%.1f") tok/s")
            Text("·")
            Text("\(metrics.completionTokens) tok")
            Text("·")
            Text("TTFT \(metrics.timeToFirstToken, specifier: "%.2f")s")
            if metrics.estimated {
                Text("(est.)").foregroundStyle(.tertiary)
            }
        }
        .font(.caption2)
        .foregroundStyle(.secondary)
        .accessibilityIdentifier("motif.chat.metrics")
    }
}

/// Renders a parsed tool call as a distinct card (dev-tool: name + args + raw JSON).
private struct ToolCallCard: View {
    let call: MotifToolCalling.ParsedToolCall
    @State private var showRaw = false

    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            HStack(spacing: 6) {
                Image(systemName: "wrench.and.screwdriver.fill").foregroundStyle(.orange)
                Text("Tool call").font(.caption).foregroundStyle(.secondary)
                Text(call.name).font(.callout).fontWeight(.semibold)
            }
            ForEach(call.arguments.keys.sorted(), id: \.self) { key in
                HStack(alignment: .top, spacing: 6) {
                    Text(key + ":")
                        .font(.caption).foregroundStyle(.secondary)
                    Text(displayValue(call.arguments[key]))
                        .font(.system(.caption, design: .monospaced)).textSelection(.enabled)
                }
            }
            DisclosureGroup(isExpanded: $showRaw) {
                Text(rawJSON)
                    .font(.caption2)
                    .textSelection(.enabled)
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .padding(.top, 4)
            } label: {
                Text("raw JSON").font(.caption2).foregroundStyle(.secondary)
            }
        }
        .padding(10)
        .frame(maxWidth: 420, alignment: .leading)
        .background(.orange.opacity(0.10), in: RoundedRectangle(cornerRadius: 10, style: .continuous))
        .overlay(
            RoundedRectangle(cornerRadius: 10, style: .continuous)
                .stroke(.orange.opacity(0.35), lineWidth: 1)
        )
        .accessibilityIdentifier("motif.chat.toolcard")
    }

    private func displayValue(_ value: MotifJSONValue?) -> String {
        guard let value else { return "" }
        switch value {
        case .string(let s): return s
        case .int(let i): return String(i)
        case .double(let d): return String(d)
        case .bool(let b): return String(b)
        case .null: return "null"
        default:
            if let data = try? JSONSerialization.data(withJSONObject: value.anyValue),
               let s = String(data: data, encoding: .utf8) { return s }
            return ""
        }
    }

    private var rawJSON: String {
        var args: [String: Any] = [:]
        for (key, value) in call.arguments { args[key] = value.anyValue }
        let object: [String: Any] = ["tool_call": ["name": call.name, "arguments": args]]
        if let data = try? JSONSerialization.data(
            withJSONObject: object, options: [.prettyPrinted, .sortedKeys]
        ), let s = String(data: data, encoding: .utf8) {
            return s
        }
        return ""
    }
}
