//
//  MessageView.swift
//  LLMFarm
//
//  A view that displays a single chat message with sender info, content, and status
//

import Inject
import MarkdownUI
import SwiftUI

/// Displays a single chat message with sender info, content and status
struct MessageView: View {
  @ObserveInjection var inject  // Enables hot reloading during development

  var message: Message  // The message to display
  @Binding var chatStyle: String  // Style to use for markdown rendering
  @State var status: String?  // Optional status text to display

  /// Displays the sender name/type for a message
  private struct SenderView: View {
    var sender: Message.Sender
    var current_model = "LLM"  // Name of the current AI model

    var body: some View {
      switch sender {
      case .user:
        Text("You")
          .font(.caption)
          .foregroundColor(.accentColor)
      case .user_rag:
        Text("RAG")  // Indicates message uses RAG context
          .font(.caption)
          .foregroundColor(.accentColor)
      case .system:
        Text(current_model)
          .font(.caption)
          .foregroundColor(.accentColor)
      }
    }
  }

  /// Displays the actual message content based on its state
  private struct MessageContentView: View {
    var message: Message
    @Binding var chatStyle: String
    @Binding var status: String?
    var sender: Message.Sender
    @State var showRag = false  // Controls visibility of RAG context

    var body: some View {
      switch message.state {
      case .none:  // Initial loading state
        VStack(alignment: .leading) {
          ProgressView()
          if status != nil {
            Text(status!)
              .font(.footnote)
          }
        }

      case .error:  // Error state shows red text
        Text(message.text)
          .foregroundColor(Color.red)
          .textSelection(.enabled)

      case .typed:  // Regular message state
        VStack(alignment: .leading) {
          if message.header != "" {
            Text(message.header)
              .font(.footnote)
              .foregroundColor(Color.gray)
              .textSelection(.enabled)
          }
          MessageImage(message: message)
          if sender == .user_rag {
            VStack {
              Button(
                action: {
                  showRag = !showRag
                },
                label: {
                  if showRag {
                    Text("Hide")
                      .font(.footnote)
                  } else {
                    Text("Show text")
                      .font(.footnote)
                  }
                }
              )
              .buttonStyle(.borderless)
              if showRag {
                Text(LocalizedStringKey(message.text)).font(.footnote).textSelection(.enabled)
              }
            }.textSelection(.enabled)
          } else {
            Text(LocalizedStringKey(message.text))
              .textSelection(.enabled)
          }
        }.textSelection(.enabled)

      case .predicting:  // Shows loading indicator while generating response
        HStack {
          Text(message.text).textSelection(.enabled)
          ProgressView()
            .padding(.leading, 3.0)
            .frame(maxHeight: .infinity, alignment: .bottom)
        }.textSelection(.enabled)

      case .predicted(let totalSecond):  // Shows completed message with timing stats
        VStack(alignment: .leading) {
          switch chatStyle {
          case "DocC":
            Markdown(message.text).markdownTheme(.docC).textSelection(.enabled)
          case "Basic":
            Markdown(message.text).markdownTheme(.basic).textSelection(.enabled)
          case "GitHub":
            Markdown(message.text).markdownTheme(.gitHub).textSelection(.enabled)
          default:
            Text(message.text).textSelection(.enabled).textSelection(.enabled)
          }
          Text(String(format: "%.2f ses, %.2f t/s", totalSecond, message.tok_sec))
            .font(.footnote)
            .foregroundColor(Color.gray)
        }.textSelection(.enabled)
      }
    }
  }

  var body: some View {
    HStack {
      // Right-align user messages, left-align system messages
      if message.sender == .user {
        Spacer()
      }

      VStack(alignment: .leading, spacing: 6.0) {
        SenderView(sender: message.sender)
        MessageContentView(
          message: message,
          chatStyle: $chatStyle,
          status: $status,
          sender: message.sender
        )
        .padding(12.0)
        .background(Color.secondary.opacity(0.2))
        .cornerRadius(12.0)
      }

      if message.sender == .system {
        Spacer()
      }
    }
    .enableInjection()
  }
}

// Preview provider commented out but available for testing
// struct MessageView_Previews: PreviewProvider {
//     static var previews: some View {
//         VStack {
//             MessageView(message: Message(sender: .user, state: .none, text: "none", tok_sec: 0))
//             MessageView(message: Message(sender: .user, state: .error, text: "error", tok_sec: 0))
//             MessageView(message: Message(sender: .user, state: .predicting, text: "predicting", tok_sec: 0))
//             MessageView(message: Message(sender: .user, state: .predicted(totalSecond: 3.1415), text: "predicted", tok_sec: 0))
//         }
//     }
// }
