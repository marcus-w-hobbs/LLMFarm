//
//  Message.swift
//  AlpacaChatApp
//
//  Created by Yoshimasa Niwa on 3/20/23.
//

import Foundation

/// Represents a single message in a chat conversation
struct Message: Identifiable {
  /// Represents the current state of a message
  enum State: Equatable {
    /// Initial state
    case none
    /// Message encountered an error
    case error
    /// Message has been typed/entered
    case typed
    /// Message is currently being predicted by the AI
    case predicting
    /// Message prediction completed with timing information
    case predicted(totalSecond: Double)
  }

  /// Identifies who sent the message
  enum Sender {
    /// Message sent by the user
    case user
    /// Message sent by user with RAG (Retrieval Augmented Generation) context
    case user_rag
    /// Message sent by the system/AI
    case system
  }

  /// Unique identifier for the message
  var id = UUID()
  /// Who sent the message (user, system, etc)
  var sender: Sender
  /// Current state of the message
  var state: State = .none
  /// The actual message content
  var text: String
  /// Tokens per second processing speed
  var tok_sec: Double
  /// Optional header text for the message
  var header: String = ""
  /// Optional file attachment path/URL
  var attachment: String? = nil
  /// Type of attachment (if any)
  var attachment_type: String? = nil
  /// Whether the message content is markdown formatted
  var is_markdown: Bool = false
}
