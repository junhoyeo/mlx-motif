import MotifKit
import SwiftUI

@main
struct MotifChatApp: App {
    var body: some Scene {
        WindowGroup {
            ContentView()
        }
        // Hidden title bar + behind-window vibrancy (set in ContentView) make the
        // window translucent so the desktop shows through and the glass refracts it.
        .windowStyle(.hiddenTitleBar)
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
