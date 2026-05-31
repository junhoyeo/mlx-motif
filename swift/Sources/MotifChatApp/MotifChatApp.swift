import MotifKit
import SwiftUI

@main
struct MotifChatApp: App {
    var body: some Scene {
        WindowGroup {
            ContentView()
        }
        .windowStyle(.titleBar)
        .commands {
            CommandGroup(replacing: .newItem) {
                Button("New Chat") {
                    NotificationCenter.default.post(name: .newMotifChatRequested, object: nil)
                }
                .keyboardShortcut("n", modifiers: [.command])
            }
        }
    }
}

extension Notification.Name {
    static let newMotifChatRequested = Notification.Name("newMotifChatRequested")
}
