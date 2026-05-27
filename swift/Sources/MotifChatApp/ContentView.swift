import MotifKit
import SwiftUI
import UniformTypeIdentifiers

struct ContentView: View {
    @StateObject private var store = ChatStore()
    @State private var selectedPanel: SidebarPanel? = .chat

    var body: some View {
        NavigationSplitView {
            List(selection: $selectedPanel) {
                Label("Chat", systemImage: "bubble.left.and.bubble.right")
                    .tag(SidebarPanel.chat)
                Label("Runtime", systemImage: "speedometer")
                    .tag(SidebarPanel.runtime)
            }
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
        .onReceive(NotificationCenter.default.publisher(for: .newMotifChatRequested)) { _ in
            store.newChat()
        }
    }
}

private enum SidebarPanel: Hashable {
    case chat
    case runtime
}

private struct ChatView: View {
    @ObservedObject var store: ChatStore

    var body: some View {
        VStack(spacing: 0) {
            ScrollViewReader { proxy in
                ScrollView {
                    LazyVStack(alignment: .leading, spacing: 14) {
                        ForEach(store.messages.filter { $0.role != .system }) { message in
                            MessageBubble(message: message)
                                .id(message.id)
                        }

                        if !store.capturedReasoning.isEmpty {
                            DisclosureGroup("Captured reasoning") {
                                Text(store.capturedReasoning)
                                    .font(.system(.body, design: .monospaced))
                                    .textSelection(.enabled)
                                    .frame(maxWidth: .infinity, alignment: .leading)
                                    .padding(.top, 6)
                            }
                            .padding()
                            .background(.yellow.opacity(0.12), in: RoundedRectangle(cornerRadius: 12))
                        }
                    }
                    .padding(20)
                }
                .onChange(of: store.messages) { _, messages in
                    if let last = messages.last {
                        withAnimation(.easeOut(duration: 0.2)) {
                            proxy.scrollTo(last.id, anchor: .bottom)
                        }
                    }
                }
            }

            Divider()

            VStack(alignment: .leading, spacing: 8) {
                if let lastError = store.lastError {
                    Text(lastError)
                        .font(.caption)
                        .foregroundStyle(.red)
                }

                HStack(alignment: .bottom, spacing: 10) {
                    TextField("Ask Motif…", text: $store.prompt, axis: .vertical)
                        .textFieldStyle(.roundedBorder)
                        .lineLimit(2...8)
                        .onSubmit { store.send() }

                    if store.isGenerating {
                        Button("Stop", role: .cancel, action: store.cancel)
                            .keyboardShortcut(".", modifiers: [.command])
                    } else {
                        Button("Send", action: store.send)
                            .keyboardShortcut(.return, modifiers: [.command])
                            .disabled(store.prompt.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)
                    }
                }
            }
            .padding()
            .background(.regularMaterial)
        }
        .navigationTitle("Motif Chat")
    }
}

private struct MessageBubble: View {
    let message: MotifChatMessage

    var body: some View {
        HStack(alignment: .top) {
            if message.role == .user { Spacer(minLength: 80) }

            VStack(alignment: .leading, spacing: 6) {
                Text(message.role.rawValue.capitalized)
                    .font(.caption)
                    .foregroundStyle(.secondary)
                Text(LocalizedStringKey(message.content.isEmpty ? "…" : message.content))
                    .textSelection(.enabled)
            }
            .padding(12)
            .background(background, in: RoundedRectangle(cornerRadius: 14, style: .continuous))
            .frame(maxWidth: 680, alignment: message.role == .user ? .trailing : .leading)

            if message.role == .assistant { Spacer(minLength: 80) }
        }
        .frame(maxWidth: .infinity, alignment: message.role == .user ? .trailing : .leading)
    }

    private var background: some ShapeStyle {
        message.role == .user ? AnyShapeStyle(.blue.opacity(0.16)) : AnyShapeStyle(.quaternary)
    }
}

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

            Section("Native MLX status") {
                Label("MotifKitMLX overlay is gated behind MOTIFKIT_ENABLE_MLX=1 for lightweight default builds", systemImage: "shippingbox")
                Label("Native path: tokenizer/chat template, checkpoint loading, q4 cache, custom Metal, and speculative decoding", systemImage: "checklist")
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
