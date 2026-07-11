import Accessibility
import MotifKit
import SwiftUI
import UniformTypeIdentifiers

struct ContentView: View {
    @StateObject private var store = ChatStore()
    @State private var selection: SidebarDestination?

    var body: some View {
        NavigationSplitView {
            SidebarView(store: store, selection: $selection)
                .navigationTitle("Motif")
                .toolbar {
                    Button(action: startNewChat) {
                        Label("New Chat", systemImage: "square.and.pencil")
                    }
                    .disabled(store.isGenerating)
                    .help(store.isGenerating ? "Stop the current response before starting a new chat" : "New chat")
                }
                // Make the toolbar chrome read as a distinct glass/material
                // surface over the translucent window (on macOS 26 the toolbar
                // adopts Liquid Glass; older OSes get a bar material).
                .toolbarBackground(.visible, for: .windowToolbar)
        } detail: {
            switch selection {
            case .runtime:
                RuntimeView(store: store)
            case .conversation, .none:
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
            guard !store.isGenerating else { return }
            startNewChat()
        }
        .onAppear {
            if let id = store.activeConversationID {
                selection = .conversation(id)
            }
        }
        .onChange(of: selection) { _, destination in
            switch destination {
            case .conversation(let id):
                store.selectConversation(id)
            case .runtime, .none:
                break
            }
        }
        .onChange(of: store.activeConversationID) { _, id in
            guard selection != .runtime, let id else { return }
            selection = .conversation(id)
        }
    }

    private func startNewChat() {
        store.newChat()
        if let id = store.activeConversationID {
            selection = .conversation(id)
        } else {
            selection = nil
        }
    }
}

private enum SidebarDestination: Hashable {
    case runtime
    case conversation(UUID)
}

/// Pure scheduling rules for transcript auto-follow. The queued callback must
/// still be the newest request and the reader must still be near the bottom;
/// explicit "Jump to Latest" actions are the only forced requests.
struct TranscriptScrollRequest: Equatable {
    let id: UUID
    let forced: Bool
}

enum TranscriptScrollPolicy {
    static func request(
        isNearBottom: Bool,
        forced: Bool,
        id: UUID = UUID()
    ) -> TranscriptScrollRequest? {
        guard forced || isNearBottom else { return nil }
        return TranscriptScrollRequest(id: id, forced: forced)
    }

    static func shouldPerform(
        _ request: TranscriptScrollRequest,
        pending: TranscriptScrollRequest?,
        isNearBottom: Bool
    ) -> Bool {
        pending == request && (request.forced || isNearBottom)
    }

    static func prioritized(
        pending: TranscriptScrollRequest?,
        new request: TranscriptScrollRequest
    ) -> TranscriptScrollRequest {
        if let pending, pending.forced, !request.forced {
            return pending
        }
        return request
    }
}

// MARK: - Sidebar (Part 4: conversation history)

private struct SidebarView: View {
    @ObservedObject var store: ChatStore
    @Binding var selection: SidebarDestination?
    @State private var conversationsPendingDeletion: [MotifConversation] = []

    var body: some View {
        List(selection: $selection) {
            Section {
                NavigationLink(value: SidebarDestination.runtime) {
                    Label("Runtime", systemImage: "speedometer")
                        .accessibilityIdentifier("motif.sidebar.runtime")
                }
            }

            Section {
                if store.conversations.isEmpty {
                    Text("No saved conversations yet")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                } else {
                    ForEach(sortedConversations) { conversation in
                        NavigationLink(value: SidebarDestination.conversation(conversation.id)) {
                            ConversationRow(
                                conversation: conversation,
                                isActive: conversation.id == store.activeConversationID
                            )
                        }
                        .contextMenu {
                            Button(role: .destructive) {
                                conversationsPendingDeletion = [conversation]
                            } label: {
                                Label("Delete", systemImage: "trash")
                            }
                            .disabled(store.isGenerating)
                        }
                    }
                    .onDelete { offsets in
                        guard !store.isGenerating else { return }
                        conversationsPendingDeletion = offsets.map { sortedConversations[$0] }
                    }
                }
            } header: {
                HStack {
                    Text("History")
                    Spacer()
                    Button {
                        store.newChat()
                        if let id = store.activeConversationID {
                            selection = .conversation(id)
                        }
                    } label: {
                        Label("New", systemImage: "plus")
                            .labelStyle(.iconOnly)
                    }
                    .buttonStyle(.borderless)
                    .disabled(store.isGenerating)
                    .help(store.isGenerating ? "Stop the current response before starting a new chat" : "New chat")
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
        .alert(
            conversationsPendingDeletion.count == 1 ? "Delete conversation?" : "Delete conversations?",
            isPresented: deleteConfirmationPresented
        ) {
            Button("Cancel", role: .cancel) {
                conversationsPendingDeletion.removeAll()
            }
            Button(
                conversationsPendingDeletion.count == 1 ? "Delete" : "Delete All",
                role: .destructive
            ) {
                let ids = conversationsPendingDeletion.map(\.id)
                conversationsPendingDeletion.removeAll()
                for id in ids {
                    store.deleteConversation(id)
                }
            }
        } message: {
            Text(deleteConfirmationMessage)
        }
    }

    private var sortedConversations: [MotifConversation] {
        store.conversations.sorted { $0.updatedAt > $1.updatedAt }
    }

    private var deleteConfirmationPresented: Binding<Bool> {
        Binding(
            get: { !conversationsPendingDeletion.isEmpty },
            set: { if !$0 { conversationsPendingDeletion.removeAll() } }
        )
    }

    private var deleteConfirmationMessage: String {
        guard conversationsPendingDeletion.count == 1,
              let conversation = conversationsPendingDeletion.first else {
            return "This permanently removes the selected conversations and their messages."
        }
        return "\u{201c}\(conversation.title)\u{201d} and all of its messages will be permanently removed."
    }
}

private struct ConversationRow: View {
    let conversation: MotifConversation
    let isActive: Bool

    var body: some View {
        HStack(spacing: 8) {
            VStack(alignment: .leading, spacing: 2) {
                Text(conversation.title)
                    .lineLimit(1)
                    .fontWeight(isActive ? .semibold : .regular)
                Text(conversation.updatedAt, format: .relative(presentation: .named))
                    .font(.caption2)
                    .foregroundStyle(.secondary)
            }
            Spacer(minLength: 4)
            if isActive {
                Image(systemName: "checkmark.circle.fill")
                    .font(.caption)
                    .foregroundStyle(Color.accentColor)
                    .accessibilityHidden(true)
            }
        }
        .padding(.vertical, 2)
        .accessibilityElement(children: .ignore)
        .accessibilityLabel(conversation.title)
        .accessibilityValue(
            "\(isActive ? "Current conversation, " : "")updated \(conversation.updatedAt.formatted(.relative(presentation: .named)))"
        )
        .help(conversation.title)
    }
}

// MARK: - Chat

private struct ChatView: View {
    @ObservedObject var store: ChatStore
    @Environment(\.accessibilityReduceMotion) private var reduceMotion
    @State private var showingModelDirImporter = false
    @State private var isNearBottom = true
    @State private var bottomInsetHeight: CGFloat = 0
    @State private var pendingScrollRequest: TranscriptScrollRequest?

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
                        .disabled(store.isGenerating)
                        .help(store.isGenerating ? "Backend settings are locked for the current response" : "Choose backend and model")
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
                        .accessibilityElement(children: .ignore)
                        .accessibilityLabel("Generation speed")
                        .accessibilityValue(
                            "\(String(format: "%.0f", store.liveTokensPerSecond)) tokens per second, \(store.liveTokenEstimate) tokens generated"
                        )
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
            .onChange(of: store.isGenerating) { wasGenerating, isGenerating in
                let announcement: String
                if isGenerating {
                    announcement = "Motif is generating a response"
                } else if wasGenerating {
                    switch store.lastGenerationOutcome {
                    case .cancelled:
                        announcement = "Response stopped"
                    case .failed:
                        announcement = "Response failed"
                    case .succeeded, .none:
                        announcement = "Response complete"
                    }
                } else {
                    return
                }
                AccessibilityNotification.Announcement(announcement).post()
            }
            .onChange(of: store.lastError) { _, error in
                guard let error, !error.isEmpty else { return }
                AccessibilityNotification.Announcement("Generation error: \(error)").post()
            }
    }

    // Stable id for an invisible element pinned to the very bottom of the
    // transcript. Scrolling to it (rather than to the last message) keeps the
    // view pinned even as trailing content — a streaming message or the captured
    // reasoning disclosure — grows.
    private static let bottomAnchor = "motif.chat.bottomAnchor"
    private static let scrollCoordinateSpace = "motif.chat.transcriptScroll"
    private static let transcriptMaxWidth: CGFloat = 880

    private var chatSurface: some View {
        GeometryReader { viewport in
            ScrollViewReader { proxy in
                ScrollView {
                    if visibleMessages.isEmpty {
                        EmptyChatState(store: store)
                            .frame(maxWidth: Self.transcriptMaxWidth, minHeight: 420)
                            .frame(maxWidth: .infinity)
                    } else {
                        LazyVStack(alignment: .leading, spacing: 16) {
                            ForEach(Array(visibleMessages.enumerated()), id: \.element.id) { index, message in
                                TranscriptRow(
                                    store: store,
                                    message: message,
                                    isLast: message.id == visibleMessages.last?.id,
                                    executedToolFollows: index + 1 < visibleMessages.count
                                        && visibleMessages[index + 1].role == .tool
                                )
                                .id(message.id)
                            }

                            Color.clear
                                .frame(height: 1)
                                .background {
                                    GeometryReader { anchor in
                                        Color.clear.preference(
                                            key: TranscriptBottomPreferenceKey.self,
                                            value: anchor.frame(in: .named(Self.scrollCoordinateSpace)).maxY
                                        )
                                    }
                                }
                                .id(Self.bottomAnchor)
                        }
                        .frame(maxWidth: Self.transcriptMaxWidth)
                        .frame(maxWidth: .infinity)
                        .padding(.horizontal, 24)
                        .padding(.vertical, 20)
                    }
                }
                .coordinateSpace(name: Self.scrollCoordinateSpace)
                .scrollContentBackground(.hidden)
                .defaultScrollAnchor(.bottom)
                .motifScrollEdgeSoft()
                .onPreferenceChange(TranscriptBottomPreferenceKey.self) { anchorY in
                    guard anchorY.isFinite else { return }
                    let scrollViewportHeight = max(0, viewport.size.height - bottomInsetHeight)
                    let nearBottom = anchorY >= 0 && anchorY <= scrollViewportHeight + 72
                    if nearBottom != isNearBottom {
                        isNearBottom = nearBottom
                    }
                }
                .onPreferenceChange(TranscriptBottomInsetPreferenceKey.self) { height in
                    bottomInsetHeight = height
                }
                .onChange(of: store.activeConversationID) { _, _ in
                    isNearBottom = true
                    requestScrollToBottom(proxy, animated: false, forced: true)
                }
                .onChange(of: store.messages) { _, _ in
                    followLatestIfNeeded(proxy, animated: !store.isGenerating)
                }
                .onChange(of: store.reasoningCharCount) { _, _ in
                    followLatestIfNeeded(proxy, animated: false)
                }
                .safeAreaInset(edge: .bottom, spacing: 0) {
                    VStack(spacing: 0) {
                        if !isNearBottom && !visibleMessages.isEmpty {
                            HStack {
                                Spacer()
                                Button {
                                    isNearBottom = true
                                    requestScrollToBottom(proxy, animated: true, forced: true)
                                } label: {
                                    Label("Jump to Latest", systemImage: "arrow.down")
                                }
                                .buttonStyle(.bordered)
                                .controlSize(.small)
                                .help("Return to the newest message and resume automatic scrolling")
                                .accessibilityHint("Resumes following the response as it streams")
                                .padding(.horizontal, 24)
                                .padding(.bottom, 8)
                            }
                            .frame(maxWidth: Self.transcriptMaxWidth)
                            .frame(maxWidth: .infinity)
                        }
                        InputBar(store: store, maxWidth: Self.transcriptMaxWidth)
                    }
                    .background {
                        GeometryReader { inset in
                            Color.clear.preference(
                                key: TranscriptBottomInsetPreferenceKey.self,
                                value: inset.size.height
                            )
                        }
                    }
                }
            }
        }
    }

    private func followLatestIfNeeded(_ proxy: ScrollViewProxy, animated: Bool) {
        requestScrollToBottom(proxy, animated: animated, forced: false)
    }

    /// Coalesce queued follow requests and re-check the current follow state at
    /// execution time. This prevents an already-queued token update from pulling
    /// the reader back down after they manually scroll away from the bottom.
    private func requestScrollToBottom(
        _ proxy: ScrollViewProxy,
        animated: Bool,
        forced: Bool
    ) {
        guard !visibleMessages.isEmpty else { return }
        guard let request = TranscriptScrollPolicy.request(
            isNearBottom: isNearBottom,
            forced: forced
        ) else { return }
        guard TranscriptScrollPolicy.prioritized(
            pending: pendingScrollRequest,
            new: request
        ) == request else {
            return
        }
        pendingScrollRequest = request

        DispatchQueue.main.async {
            guard TranscriptScrollPolicy.shouldPerform(
                request,
                pending: pendingScrollRequest,
                isNearBottom: isNearBottom
            ) else {
                if pendingScrollRequest == request {
                    pendingScrollRequest = nil
                }
                return
            }
            pendingScrollRequest = nil
            if animated && !reduceMotion {
                withAnimation(.easeOut(duration: 0.18)) {
                    proxy.scrollTo(Self.bottomAnchor, anchor: .bottom)
                }
            } else {
                proxy.scrollTo(Self.bottomAnchor, anchor: .bottom)
            }
        }
    }
}

private struct TranscriptBottomPreferenceKey: PreferenceKey {
    static let defaultValue: CGFloat = .infinity

    static func reduce(value: inout CGFloat, nextValue: () -> CGFloat) {
        value = nextValue()
    }
}

private struct TranscriptBottomInsetPreferenceKey: PreferenceKey {
    static let defaultValue: CGFloat = 0

    static func reduce(value: inout CGFloat, nextValue: () -> CGFloat) {
        value = nextValue()
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
    var isStreaming: Bool = false
    @Environment(\.accessibilityReduceMotion) private var reduceMotion

    // Auto-expanded while the model is thinking so the stream is visible, then
    // collapsed to a tidy summary once the answer starts. The user can still
    // toggle it manually at any point.
    @State private var expanded = false

    var body: some View {
        DisclosureGroup(isExpanded: $expanded) {
            Text(text)
                .font(.body)
                .textSelection(.enabled)
                .frame(maxWidth: .infinity, alignment: .leading)
                .padding(.top, 6)
        } label: {
            HStack(spacing: 6) {
                Label(isStreaming ? "Reasoning…" : "Captured reasoning", systemImage: "brain")
                    .font(.callout)
                if isStreaming {
                    ProgressView().controlSize(.mini)
                }
            }
        }
        .accessibilityIdentifier("motif.chat.reasoning")
        .padding(12)
        .background(.ultraThinMaterial, in: RoundedRectangle(cornerRadius: 12, style: .continuous))
        .overlay(
            RoundedRectangle(cornerRadius: 12, style: .continuous)
                .stroke(.quaternary, lineWidth: 1)
        )
        .onAppear { expanded = isStreaming }
        .onChange(of: isStreaming) { _, streaming in
            if reduceMotion {
                expanded = streaming
            } else {
                withAnimation(.easeInOut(duration: 0.2)) { expanded = streaming }
            }
        }
    }
}

private struct InputBar: View {
    @ObservedObject var store: ChatStore
    let maxWidth: CGFloat
    @FocusState private var composerFocused: Bool

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            if store.isGenerating {
                HStack(spacing: 6) {
                    ProgressView()
                        .controlSize(.small)
                    Text(store.runtimeStatus)
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
                .accessibilityElement(children: .ignore)
                .accessibilityLabel("Response status")
                .accessibilityValue(store.runtimeStatus)
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
                    .accessibilityLabel("Generation error")
                    .accessibilityValue(lastError)
                    .accessibilityIdentifier("motif.chat.error")
            }

            // Claude-style layout: the text field spans the full bar width, and
            // the controls live in a row UNDER it — mode icons on the left,
            // send/stop on the right — instead of crowding the field inline.
            VStack(alignment: .leading, spacing: 10) {
                TextField("Ask Motif…", text: $store.prompt, axis: .vertical)
                    .textFieldStyle(.plain)
                    .lineLimit(1...8)
                    .onSubmit { store.send() }
                    .focused($composerFocused)
                    .accessibilityLabel("Message Motif")
                    .accessibilityHint("Press Command Return to send")
                    .accessibilityIdentifier("motif.chat.input")

                HStack(spacing: 14) {
                    ThinkModeMenu(store: store)
                    ToolsToggle(store: store)

                    Spacer()

                    if store.isGenerating {
                        Button(role: .cancel, action: store.cancel) {
                            Label(
                                store.runtimeStatus == "Stopping…" ? "Stopping" : "Stop",
                                systemImage: "stop.fill"
                            )
                            .labelStyle(.iconOnly)
                        }
                        .motifGlassButton()
                        .tint(.red)
                        .keyboardShortcut(".", modifiers: [.command])
                        .disabled(store.runtimeStatus == "Stopping…")
                        .help(store.runtimeStatus == "Stopping…" ? "Stopping response" : "Stop response")
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
        .frame(maxWidth: maxWidth)
        .frame(maxWidth: .infinity)
        .padding(.horizontal, 24)
        .padding(.vertical, 16)
        .onAppear { composerFocused = true }
        .onChange(of: store.activeConversationID) { _, _ in
            guard !store.isGenerating else { return }
            composerFocused = true
        }
        .onChange(of: store.isGenerating) { wasGenerating, isGenerating in
            if wasGenerating && !isGenerating {
                composerFocused = true
            }
        }
    }
}

// MARK: - Transcript row

/// One transcript entry: an assistant turn's captured reasoning (streamed live
/// above the answer) followed by the message bubble. Extracted from the
/// `ForEach` body so the per-row derivations stay out of the view builder's
/// type-inference (which times out when inlined).
private struct TranscriptRow: View {
    @ObservedObject var store: ChatStore
    let message: MotifChatMessage
    let isLast: Bool
    let executedToolFollows: Bool

    private var reasoning: String {
        message.role == .assistant ? (store.reasoningByMessage[message.id] ?? "") : ""
    }

    /// Reasoning is "live" while this is the last message, the turn is
    /// generating, and no answer text has arrived yet (Motif emits the whole
    /// `<think>` block before the answer). Once answer text starts, the card
    /// collapses to a tidy summary.
    private var reasoningStreaming: Bool {
        store.isGenerating && isLast && message.content.isEmpty
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            if !reasoning.isEmpty {
                ReasoningDisclosure(text: reasoning, isStreaming: reasoningStreaming)
            }
            MessageBubble(
                store: store,
                message: message,
                isStreaming: store.isGenerating && isLast && message.role == .assistant,
                isLast: isLast,
                executedToolFollows: executedToolFollows
            )
        }
    }
}

// MARK: - Message bubble (Part 1)

private struct MessageBubble: View {
    @ObservedObject var store: ChatStore
    let message: MotifChatMessage
    let isStreaming: Bool
    let isLast: Bool
    /// True when the next transcript message is a `.tool` result — the
    /// persisted marker that this assistant turn's tool call actually ran.
    let executedToolFollows: Bool

    @State private var didCopy = false
    @State private var pendingDestructiveAction: MessageDestructiveAction?

    private var isUser: Bool { message.role == .user }
    private var isAssistant: Bool { message.role == .assistant }

    /// Parsed tool call to render as a card. The in-memory `store.toolCalls`
    /// covers the live session; after a relaunch that dict is empty, so a
    /// restored executed-call turn (marked by its following tool result)
    /// re-parses its content on the fly.
    private var displayedToolCall: MotifToolCalling.ParsedToolCall? {
        store.toolCalls[message.id]
            ?? (executedToolFollows
                ? MotifToolCalling.parseToolCall(
                    text: message.content, toolNames: MotifChatDemoTools.names)
                : nil)
    }

    var body: some View {
        // ONE alignment mechanism (hstack/vstack + ui-patterns skills): the
        // bubble hugs its content up to a consistent max width, then a single
        // trailing `.frame(maxWidth: .infinity, alignment:)` positions it on the
        // correct side of the transcript — no HStack+Spacer *and* outer frame
        // fighting each other.
        VStack(alignment: bubbleAlignment, spacing: 6) {
            // Role label only — copy lives in the always-visible actions row
            // below (one copy affordance per message, not two).
            Text(message.role.rawValue.capitalized)
                .font(.caption)
                .foregroundStyle(.secondary)

            if let call = displayedToolCall {
                ToolCallCard(call: call)
            } else if message.role == .tool {
                // Executed-tool result feeding the next round — compact and
                // distinct from prose so the loop is visible at a glance.
                HStack(alignment: .firstTextBaseline, spacing: 6) {
                    Image(systemName: "checkmark.circle.fill")
                        .foregroundStyle(.green)
                        .font(.callout)
                    // Content is the bare result exactly as the model sees it;
                    // the "Tool result (name)" framing is display-only.
                    Text("Tool result (\(message.name ?? "tool")): \(message.content)")
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
                    MarkdownMessageView(content: message.content)
                        .accessibilityElement(children: .combine)
                        .accessibilityIdentifier("motif.chat.message.text")
                    if isStreaming {
                        StreamingCaret()
                    }
                }
            }

            // Dev-tool decode metrics under finished assistant turns.
            if isAssistant, let m = store.metrics[message.id] {
                MetricsLine(metrics: m)
            }

            // A truncated turn auto-resumes into this same bubble (no user
            // action). If the flag still shows once generation is idle, the
            // auto-continue budget was exhausted — surface a terminal note so
            // the cut-off isn't silent.
            if isAssistant && !isStreaming && store.truncatedMessages.contains(message.id) {
                HStack(spacing: 6) {
                    Image(systemName: "scissors")
                        .foregroundStyle(.orange)
                    Text("Still cut off after auto-continuing — raise Max tokens in Runtime")
                        .foregroundStyle(.secondary)
                }
                .font(.caption)
                .accessibilityIdentifier("motif.chat.truncated")
            }

            // Per-message actions on assistant turns — always visible (no
            // hover reveal), never while the message is mid-stream.
            // Materials/plain controls only — no glass on message content.
            if isAssistant && !isStreaming {
                actionsRow
            }
        }
        .padding(12)
        .background(bubbleBackground, in: RoundedRectangle(cornerRadius: 14, style: .continuous))
        // Consistent max bubble width; the bubble hugs shorter content.
        .frame(maxWidth: 680, alignment: isUser ? .trailing : .leading)
        // Single alignment step: push the bubble to its side of the transcript.
        .frame(maxWidth: .infinity, alignment: isUser ? .trailing : .leading)
        .alert(
            pendingDestructiveAction == .regenerate ? "Regenerate response?" : "Delete response?",
            isPresented: destructiveConfirmationPresented
        ) {
            Button("Cancel", role: .cancel) {
                pendingDestructiveAction = nil
            }
            switch pendingDestructiveAction {
            case .regenerate:
                Button("Regenerate", role: .destructive) {
                    pendingDestructiveAction = nil
                    store.regenerateLast()
                }
            case .delete:
                Button("Delete", role: .destructive) {
                    pendingDestructiveAction = nil
                    store.deleteMessage(message.id)
                }
            case .none:
                EmptyView()
            }
        } message: {
            Text(
                pendingDestructiveAction == .regenerate
                    ? "Motif will retry the latest prompt and replace this response."
                    : "This permanently removes the response from the conversation."
            )
        }
        .accessibilityIdentifier(isUser ? "motif.chat.bubble.user" : "motif.chat.bubble.assistant")
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
                    pendingDestructiveAction = .regenerate
                } label: {
                    Label("Regenerate", systemImage: "arrow.clockwise")
                        .font(.caption2)
                }
                .help("Re-run the last prompt")
            }

            Button(role: .destructive) {
                pendingDestructiveAction = .delete
            } label: {
                Label("Delete", systemImage: "trash")
                    .font(.caption2)
            }
            .help("Delete this response")
        }
        .buttonStyle(.borderless)
        .labelStyle(.titleAndIcon)
        .foregroundStyle(.secondary)
        // Actions are disabled mid-generation to match ChatStore's guards.
        .disabled(store.isGenerating)
    }

    private var destructiveConfirmationPresented: Binding<Bool> {
        Binding(
            get: { pendingDestructiveAction != nil },
            set: { if !$0 { pendingDestructiveAction = nil } }
        )
    }

    private var bubbleBackground: AnyShapeStyle {
        // Materials only (per Apple, never glass on message content).
        isUser ? AnyShapeStyle(.blue.opacity(0.16)) : AnyShapeStyle(.ultraThinMaterial)
    }
}

private enum MessageDestructiveAction {
    case regenerate
    case delete
}

/// Subtle blinking caret shown at the end of the streaming assistant message.
private struct StreamingCaret: View {
    @Environment(\.accessibilityReduceMotion) private var reduceMotion
    @State private var visible = true

    var body: some View {
        Text("▌")
            .foregroundStyle(.secondary)
            .opacity(reduceMotion || visible ? 1 : 0.15)
            .accessibilityHidden(true)
            .onAppear {
                guard !reduceMotion else { return }
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
    @State private var showingResetConfirmation = false

    var body: some View {
        Form {
            if store.isGenerating {
                Section {
                    Label(
                        "Settings are locked until the current response finishes",
                        systemImage: "lock.fill"
                    )
                    .font(.callout)
                    .foregroundStyle(.secondary)
                    .accessibilityLabel("Runtime settings locked")
                    .accessibilityValue("The current response is using the settings shown below")
                }
            }

            Section("Backend") {
                Picker("Chat path", selection: $store.backendMode) {
                    ForEach(MotifChatBackendMode.allCases) { mode in
                        Label(mode.label, systemImage: mode.systemImage).tag(mode)
                    }
                }
                .accessibilityIdentifier("motif.runtime.backendPicker")

                switch store.backendMode {
                case .openAICompatible:
                    TextField("OpenAI-compatible endpoint", text: $store.endpoint)
                        .textFieldStyle(.roundedBorder)
                        .accessibilityIdentifier("motif.runtime.endpoint")
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
                .accessibilityIdentifier("motif.runtime.thinkPicker")
            }
            .disabled(store.isGenerating)

            Section("App status") {
                Label("Active chat path: \(store.backendMode.label)", systemImage: store.backendMode.systemImage)
                Label("Runtime status: \(store.runtimeStatus)", systemImage: statusIcon)
                    .accessibilityIdentifier("motif.runtime.status")
                if let lastError = store.lastError {
                    Label(lastError, systemImage: "exclamationmark.triangle.fill")
                        .foregroundStyle(.red)
                        .textSelection(.enabled)
                        .accessibilityLabel("Runtime error")
                        .accessibilityValue(lastError)
                }
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
                    .accessibilityIdentifier("motif.runtime.maxTokens")
                Slider(value: $store.temperature, in: 0...2) {
                    Text("Temperature")
                }
                Text("Temperature: \(store.temperature, specifier: "%.2f")")
                    .foregroundStyle(.secondary)
            }
            .disabled(store.isGenerating)

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
            .disabled(store.isGenerating)

            Section("Native MLX status") {
                Label("MotifKitMLX overlay is gated behind MOTIFKIT_ENABLE_MLX=1 for lightweight default builds", systemImage: "shippingbox")
                Label("Native path: tokenizer/chat template, checkpoint loading, q4 cache, and custom Metal", systemImage: "checklist")
                Label("Speculative decoding is implemented; practical speedup requires a smaller draft model", systemImage: "checkmark.seal")
                Label("Evidence: docs/benchmarks/swift-python-hard-parity-20260526T091532Z.md", systemImage: "speedometer")
            }

            Section("Settings") {
                Button("Reset runtime settings", role: .destructive) {
                    showingResetConfirmation = true
                }
                .disabled(store.isGenerating)
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
        .alert("Reset runtime settings?", isPresented: $showingResetConfirmation) {
            Button("Cancel", role: .cancel) {}
            Button("Reset", role: .destructive) {
                store.resetRuntimeSettings()
            }
        } message: {
            Text("Backend, model, generation, context, thinking, and tool settings will return to their defaults. Conversation history is preserved.")
        }
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
            "circle"
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
        .accessibilityElement(children: .ignore)
        .accessibilityLabel("Backend and model")
        .accessibilityValue("\(store.backendDisplayName), \(store.backendMode.label), runtime \(store.runtimeStatus)")
        .accessibilityIdentifier("motif.chat.model")
        // 16pt to match every other toolbar pill (context meter, tok/s readout,
        // Runtime title) — "one item per pill, each with matched padding".
        .padding(.horizontal, 16)
        .padding(.vertical, 5)
    }
}

/// A shape-and-color runtime indicator. Idle stays neutral because a configured
/// backend is not validated until the user sends a request.
private struct StatusDot: View {
    @ObservedObject var store: ChatStore

    var body: some View {
        Image(systemName: icon)
            .font(.caption)
            .foregroundStyle(color)
            .help(accessibilityValue)
            .accessibilityLabel("Runtime status")
            .accessibilityValue(accessibilityValue)
    }

    private var icon: String {
        if store.lastError != nil || store.runtimeStatus == "Error" {
            return "exclamationmark.circle.fill"
        }
        if store.isGenerating {
            return "ellipsis.circle.fill"
        }
        if store.runtimeStatus == "Cancelled" {
            return "stop.circle"
        }
        if store.runtimeStatus == "Idle" {
            return "circle"
        }
        return "info.circle.fill"
    }

    private var color: Color {
        if store.lastError != nil || store.runtimeStatus == "Error" { return .red }
        if store.isGenerating { return .orange }
        if store.runtimeStatus == "Idle" || store.runtimeStatus == "Cancelled" { return .secondary }
        return .accentColor
    }

    private var accessibilityValue: String {
        store.runtimeStatus == "Idle"
            ? "Idle; backend will be checked when a message is sent"
            : store.runtimeStatus
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
        .accessibilityElement(children: .ignore)
        .accessibilityLabel("Context usage")
        .accessibilityValue(
            "\(store.currentContextTokens) of \(store.contextTokenBudget) estimated tokens, \(Int(fraction * 100)) percent"
        )
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
        .disabled(store.isGenerating)
        .accessibilityLabel("Reasoning trace")
        .accessibilityValue(store.thinkMode.rawValue.capitalized)
        .accessibilityHint("Applies to the next response")
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
        .disabled(store.isGenerating)
        .accessibilityLabel("Demo tools")
        .accessibilityValue(store.toolsEnabled ? "Enabled" : "Disabled")
        .accessibilityHint("Applies to the next response")
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
