import AppIntents
import SwiftUI

struct LLMFarmShortcutsProvider: AppShortcutsProvider {
  static var appShortcuts: [AppShortcut] {
    AppShortcut(
      intent: LLMQueryIntent(),
      phrases: [
        "Ask Local LLM"
      ],
      shortTitle: "Create query to local LLM model",
      systemImageName: "brain.filled.head.profile"
    )
  }
}
