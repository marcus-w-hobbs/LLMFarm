//
//  ChatItem.swift
//  ChatUI
//
//  Created by Shezad Ahamed on 6/08/21.
//

import Inject
import SwiftUI

/// A view that represents a single chat item in a list or grid
struct ChatItem: View {
  /// Property wrapper for hot reloading support
  @ObserveInjection var inject

  /// Name of the chat avatar image (without size suffix)
  var chatImage: String = ""

  /// Title/name of the chat
  var chatTitle: String = ""

  /// Preview message or description text
  var message: String = ""

  /// Timestamp of last message (currently unused)
  var time: String = ""

  /// Name of the AI model used
  var model: String = ""

  /// Chat identifier
  var chat: String = ""

  /// Size of the model in gigabytes
  var model_size: String = ""

  /// Two-way binding for selected model name
  @Binding var model_name: String

  /// Two-way binding for chat title
  @Binding var title: String

  /// Callback to close the chat
  var close_chat: () -> Void

  var body: some View {
    HStack {
      // Chat avatar image
      Image(chatImage + "_85")
        .resizable()
        .background(Color("color_bg_inverted").opacity(0.05))
        .padding(EdgeInsets(top: 7, leading: 5, bottom: 7, trailing: 5))
        .frame(width: 85, height: 85)
        .clipShape(Circle())

      VStack(alignment: .leading, spacing: 5) {
        HStack {
          // Chat title
          Text(chatTitle)
            .fontWeight(.semibold)
            .padding(.top, 3)
            .multilineTextAlignment(.leading)
          Spacer()
        }

        // Model info and preview message
        Text(message + " " + model_size + "G")
          .foregroundColor(Color("color_bg_inverted").opacity(0.5))
          .font(.footnote)
          .opacity(0.6)
          .lineLimit(2)
      }
    }
    .enableInjection()
  }
}
