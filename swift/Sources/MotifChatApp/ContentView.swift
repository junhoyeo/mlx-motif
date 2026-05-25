import MotifKit
import SwiftUI

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

    var body: some View {
        Form {
            Section("Backend") {
                TextField("OpenAI-compatible endpoint", text: $store.endpoint)
                    .textFieldStyle(.roundedBorder)
                TextField("Model ID", text: $store.model)
                    .textFieldStyle(.roundedBorder)
                Picker("Thinking", selection: $store.thinkMode) {
                    ForEach(MotifThinkMode.allCases, id: \.self) { mode in
                        Text(mode.rawValue.capitalized).tag(mode)
                    }
                }
            }

            Section("App status") {
                Label("Active chat path: OpenAI-compatible streaming endpoint", systemImage: "network")
                Label("Native in-process MLX generation: optional package overlay", systemImage: "cpu")
                Text("Run `mlx-motif serve` and point this app at `/v1` for the default local chat path. The optional MotifKitMLX overlay now has a native reference generation CLI for converted checkpoints, while this lightweight app target remains server-backed by default.")
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
                Label("MotifKitMLX overlay is gated behind MOTIFKIT_ENABLE_MLX=1", systemImage: "shippingbox")
                Label("Current native evidence: reference load/generation wiring plus fixture parity", systemImage: "checklist")
                Label("Remaining native path: q4 cache, custom Metal kernel parity, speculative decoding, and same-machine benchmarks", systemImage: "wrench.and.screwdriver")
            }
        }
        .formStyle(.grouped)
        .padding()
        .navigationTitle("Runtime")
    }
}
