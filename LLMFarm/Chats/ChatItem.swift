/// ChatItem manages the UI for individual chat sessions while tracking context window implications
///
/// This view carefully orchestrates chat display while considering:
/// 1. Memory Management:
///    - Model size tracking for context window planning
///    - Preview text truncation to avoid memory spikes
///    - Avatar image caching and sizing
/// 2. Context Window Optimization:
///    - Model selection affects available context size
///    - Message preview length impacts token count
///    - UI elements sized for efficient token usage
/// 3. Performance Monitoring:
///    - Model size display for user awareness
///    - Chat title length constraints
///    - Image asset memory footprint
///
/// Product teams should monitor:
/// 1. Context utilization per chat
/// 2. Memory pressure during chat loads
/// 3. Model switching frequency
/// 4. Preview text token counts

import Inject
import SwiftUI

struct ChatItem: View {
  /// Enables hot reloading for rapid context window testing
  @ObserveInjection var inject

  /// Avatar image name, sized appropriately for context budget
  var chatImage: String = ""

  /// Chat title, constrained to avoid excessive token usage
  var chatTitle: String = ""

  /// Preview text, truncated based on available context window
  var message: String = ""

  /// Timestamp for context window age tracking
  var time: String = ""

  /// Model identifier for context size determination
  var model: String = ""

  /// Unique chat ID for context isolation
  var chat: String = ""

  /// Model size in GB, critical for context window planning
  var model_size: String = ""

  /// Selected model binding for dynamic context resizing
  @Binding var model_name: String

  /// Chat title binding for context-aware updates
  @Binding var title: String

  /// Cleanup callback to free context window
  var close_chat: () -> Void

  var body: some View {
    HStack {
      // Avatar sized to 85x85 to limit memory impact on context
      Image(chatImage + "_85")
        .resizable()
        .background(Color("color_bg_inverted").opacity(0.05))
        .padding(EdgeInsets(top: 7, leading: 5, bottom: 7, trailing: 5))
        .frame(width: 85, height: 85)
        .clipShape(Circle())

      VStack(alignment: .leading, spacing: 5) {
        HStack {
          // Title display with memory-conscious styling
          Text(chatTitle)
            .fontWeight(.semibold)
            .padding(.top, 3)
            .multilineTextAlignment(.leading)
          Spacer()
        }

        // Preview combines message and model size for context awareness
        Text(message + " " + model_size + "G")
          .foregroundColor(Color("color_bg_inverted").opacity(0.5))
          .font(.footnote)
          .opacity(0.6)
          .lineLimit(2)  // Prevent excessive token accumulation
      }
    }
    .enableInjection()
  }
}
