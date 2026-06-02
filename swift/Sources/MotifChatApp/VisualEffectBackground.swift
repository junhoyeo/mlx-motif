import SwiftUI

// MARK: - Behind-window vibrancy background
//
// SwiftUI has no built-in way to make a macOS window background translucent so
// the DESKTOP (whatever is behind the window) shows through and blurs. The
// canonical approach is to bridge `NSVisualEffectView` with
// `blendingMode = .behindWindow`, which samples and blurs the content behind the
// window — real macOS vibrancy, not a fake gradient.
//
// Refs:
// - Apple Developer Forums: "Transparent window in SwiftUI macOS application"
// - philz.blog: "Vibrancy, NSAppearance, and Visual Effects in Modern AppKit"
// - zachwaugh.com: "Creating a blurred window background with SwiftUI on macOS"
//
// Pair this with `.windowStyle(.hiddenTitleBar)` and `.ignoresSafeArea()` so the
// effect goes edge-to-edge and the window itself is non-opaque. The Liquid Glass
// chrome (input bar / buttons) then refracts this live, blurred desktop content.

struct VisualEffectBackground: NSViewRepresentable {
  var material: NSVisualEffectView.Material = .underWindowBackground
  var blendingMode: NSVisualEffectView.BlendingMode = .behindWindow

  func makeNSView(context: Context) -> NSVisualEffectView {
    let view = NSVisualEffectView()
    view.material = material
    view.blendingMode = blendingMode
    view.state = .active
    // Ensure the window itself is non-opaque so .behindWindow can sample the
    // desktop rather than a black backing.
    DispatchQueue.main.async { [weak view] in
      view?.window?.isOpaque = false
      view?.window?.backgroundColor = .clear
    }
    return view
  }

  func updateNSView(_ view: NSVisualEffectView, context: Context) {
    view.material = material
    view.blendingMode = blendingMode
    view.state = .active
  }
}
