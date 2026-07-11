import SwiftUI

// MARK: - Liquid Glass styling (gated)
//
// Liquid Glass APIs (`.glassEffect`, `GlassEffectContainer`,
// `.buttonStyle(.glass)` / `.glassProminent`) ship in the macOS 26 SDK
// (Xcode 26 / Swift 6.2+). They do not exist in older SDKs, so every call is
// guarded by `#if compiler(>=6.2)` (compile-time) plus `if #available(macOS
// 26.0, *)` (runtime), each with a `.regularMaterial` / bordered fallback.
//
// Glass is applied ONLY to chrome (input bar, send button) — never to message
// content, per Apple's "never apply glass to content" guidance. To be visible,
// glass needs (a) a non-zero rounded shape so it reads as a distinct floating
// surface and (b) content behind it to refract — the input bar floats inset
// over the scrolling transcript, and the Send button uses the prominent
// (tinted) style.

extension View {
  /// Floating glass surface for container chrome (the input bar). Falls back to
  /// a rounded material on toolchains/OSes without Liquid Glass.
  @ViewBuilder
  func motifGlassSurface(cornerRadius: CGFloat = 22) -> some View {
    #if compiler(>=6.2)
    if #available(macOS 26.0, *) {
      // `.regular` (frosted) — the input bar is a standard control that floats
      // over the *scrolling transcript*, so it must stay legible: `.clear` let
      // the text behind bleed through and collide with the field's placeholder.
      // Per the Liquid Glass guidance, `.clear` is for controls over media;
      // `.regular` is the variant for toolbars/nav bars/standard controls.
      self.glassEffect(.regular, in: .rect(cornerRadius: cornerRadius))
    } else {
      self.background(
        .regularMaterial,
        in: RoundedRectangle(cornerRadius: cornerRadius, style: .continuous)
      )
    }
    #else
    self.background(
      .regularMaterial,
      in: RoundedRectangle(cornerRadius: cornerRadius, style: .continuous)
    )
    #endif
  }

  /// Prominent (tinted) glass treatment for the send/stop button. This is the
  /// visible glass variant; falls back to bordered-prominent elsewhere.
  @ViewBuilder
  func motifGlassButton() -> some View {
    #if compiler(>=6.2)
    if #available(macOS 26.0, *) {
      self.buttonStyle(.glassProminent)
    } else {
      self.buttonStyle(.borderedProminent)
    }
    #else
    self.buttonStyle(.borderedProminent)
    #endif
  }

  /// Glass background for structural chrome that spans an edge — the sidebar
  /// and toolbar. Unlike `motifGlassSurface` (a floating inset bar using the
  /// high-transparency `.clear` glass), this uses the `.regular` glass so the
  /// large surface stays legible behind list/toolbar content. Falls back to a
  /// `.bar` material on toolchains/OSes without Liquid Glass. Edge-to-edge, so
  /// no corner radius — it reads as a panel, not a floating pill.
  @ViewBuilder
  func motifGlassChrome() -> some View {
    #if compiler(>=6.2)
    if #available(macOS 26.0, *) {
      self.glassEffect(.regular, in: .rect(cornerRadius: 0))
    } else {
      self.background(.bar)
    }
    #else
    self.background(.bar)
    #endif
  }

  /// Scroll-edge effect (macOS 26): as transcript content scrolls beneath the
  /// window toolbar and the floating input bar, the system fades/blurs it at the
  /// edges so the glass chrome stays legible and content doesn't hard-clip
  /// against it. Per the HIG (Toolbars → "use a ScrollEdgeEffectStyle … to
  /// distinguish the toolbar area from the content area") and the
  /// `scrollEdgeEffectStyle(_:for:)` API. `.soft` is a gradual blur transition;
  /// no-op fallback on older toolchains/OSes.
  @ViewBuilder
  func motifScrollEdgeSoft() -> some View {
    #if compiler(>=6.2)
    if #available(macOS 26.0, *) {
      // `.hard` at the top gives a firm blur cutoff so transcript content is
      // masked where it meets the toolbar (a `.soft` gradient let sharp text
      // bleed through above the header). `.soft` at the bottom keeps a gentle
      // fade into the floating input bar.
      self
        .scrollEdgeEffectStyle(.hard, for: .top)
        .scrollEdgeEffectStyle(.soft, for: .bottom)
    } else {
      self
    }
    #else
    self
    #endif
  }

  /// Wraps grouped glass chrome in a `GlassEffectContainer` so nearby glass
  /// elements blend/refract consistently (Apple requires glass to share a
  /// container to sample correctly). No-op fallback otherwise.
  @ViewBuilder
  func motifGlassGroup(spacing: CGFloat = 10) -> some View {
    #if compiler(>=6.2)
    if #available(macOS 26.0, *) {
      GlassEffectContainer(spacing: spacing) { self }
    } else {
      self
    }
    #else
    self
    #endif
  }
}
