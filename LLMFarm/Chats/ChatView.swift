//
//  ChatView.swift
//
//  Main view for displaying and interacting with a chat conversation
//  Created by Guinmoon
//

import Inject
import SwiftUI

/// Main view for displaying chat messages and handling user interactions
struct ChatView: View {
  // MARK: - Properties

  @ObserveInjection var inject  // Enables hot reloading during development
  @EnvironmentObject var aiChatModel: AIChatModel  // Manages chat state and AI interactions
  @EnvironmentObject var orientationInfo: OrientationInfo  // Tracks device orientation

  // Default placeholder text for input field
  // #if os(iOS)
  @State var placeholderString: String = "Type your message..."
  @State private var inputText: String = "Type your message..."
  // #else
  //     @State var placeholderString: String = ""
  //     @State private var inputText: String = ""
  // #endif

  // MARK: - Bindings

  @Binding var modelName: String  // Name of the AI model being used
  @Binding var chatSelection: [String: String]?  // Currently selected chat
  @Binding var title: String  // Chat title
  var CloseChat: () -> Void  // Callback to close the chat
  @Binding var AfterChatEdit: () -> Void  // Callback after editing chat settings
  @Binding var addChatDialog: Bool  // Controls add chat dialog visibility
  @Binding var editChatDialog: Bool  // Controls edit chat dialog visibility

  // MARK: - View State

  @State var chatStyle: String = "None"  // Visual style of chat bubbles
  @State private var reloadButtonIcon: String = "arrow.counterclockwise.circle"
  @State private var clearChatButtonIcon: String = "eraser.line.dashed.fill"
  @State private var scrollProxy: ScrollViewProxy? = nil  // For programmatic scrolling
  @State private var scrollTarget: Int?  // Target message ID to scroll to
  @State private var toggleEditChat = false  // Controls edit chat sheet visibility
  @State private var clearChatAlert = false  // Controls clear chat alert visibility
  @State private var autoScroll = true  // Whether to auto-scroll to new messages
  @State private var enableRAG = false  // Enables Retrieval Augmented Generation
  @FocusState var focusedField: Field?  // Currently focused input field
  @Namespace var bottomID  // Namespace for scroll animations

  @FocusState private var isInputFieldFocused: Bool

  // MARK: - Methods

  /// Scrolls the chat to the bottom
  /// - Parameter with_animation: Whether to animate the scroll
  func scrollToBottom(with_animation: Bool = false) {
    var scroll_bug = true
    #if os(macOS)
      scroll_bug = false
    #else
      if #available(iOS 16.4, *) {
        scroll_bug = false
      }
    #endif
    if scroll_bug {
      return
    }
    if !autoScroll {
      return
    }
    let last_msg = aiChatModel.messages.last
    if last_msg != nil && last_msg?.id != nil && scrollProxy != nil {
      if with_animation {
        withAnimation {
          scrollProxy?.scrollTo("latest")
        }
      } else {
        scrollProxy?.scrollTo("latest")
      }
    }
  }

  /// Reloads the current chat data
  func reload() async {
    if chatSelection == nil {
      return
    }
    print(chatSelection!)
    print("\nreload\n")
    aiChatModel.reload_chat(chatSelection!)
  }

  /// Forces a complete reload of the chat
  func hard_reload_chat() {
    self.aiChatModel.hard_reload_chat()
  }

  // MARK: - Subviews

  /// Overlay button to scroll to bottom of chat
  private var scrollDownOverlay: some View {
    Button {
      Task {
        autoScroll = true
        scrollToBottom()
      }
    } label: {
      Image(systemName: "arrow.down.circle")
        .resizable()
        .foregroundColor(.white)
        .frame(width: 25, height: 25)
        .padding([.bottom, .trailing], 15)
        .opacity(0.4)
    }
    .buttonStyle(BorderlessButtonStyle())
  }

  /// Debug overlay showing current AI state
  private var debugOverlay: some View {
    Text(String(describing: aiChatModel.state))
      .foregroundColor(.white)
      .frame(width: 185, height: 25)
      .opacity(0.4)
  }

  // MARK: - Body

  var body: some View {
    VStack {
      // Loading state indicator
      VStack {
        if aiChatModel.state == .loading || aiChatModel.state == .ragIndexLoading
          || aiChatModel.state == .ragSearch
        {
          VStack {
            HStack {
              Text(String(describing: aiChatModel.state))
                .foregroundColor(.accentColor)
                .frame(width: 200)
                .opacity(0.4)
                .offset(x: -75, y: 8)
                .frame(alignment: .leading)
                .font(.footnote)
              ProgressView(value: aiChatModel.load_progress)
                .padding(.leading, -195)
                .offset(x: 0, y: -4)
            }
          }
        }
      }

      // Main chat message list
      ScrollViewReader { scrollView in
        VStack {
          List {
            ForEach(aiChatModel.messages, id: \.id) { message in
              MessageView(message: message, chatStyle: $chatStyle, status: nil).id(message.id)
                .textSelection(.enabled)
            }
            .listRowSeparator(.hidden)
            Text("").id("latest")
          }
          .textSelection(.enabled)
          .listStyle(PlainListStyle())
          .overlay(scrollDownOverlay, alignment: .bottomTrailing)
        }
        .textSelection(.enabled)
        .onChange(of: aiChatModel.AI_typing) { ai_typing in
          scrollToBottom(with_animation: false)
        }
        .disabled(chatSelection == nil)
        .onAppear {
          scrollProxy = scrollView
          scrollToBottom(with_animation: false)
        }
      }
      .textSelection(.enabled)
      .frame(maxHeight: .infinity)
      .disabled(aiChatModel.state == .loading)
      .onChange(of: chatSelection) { selection in
        Task {
          if selection == nil {
            CloseChat()
          } else {
            print(selection!)
            chatStyle = selection!["chat_style"] as String? ?? "none"
            await self.reload()
          }
        }
      }
      .onTapGesture { location in
        print("Tapped at \(location)")
        focusedField = nil
        autoScroll = false
      }

      // Toolbar buttons
      .toolbar {
        Button {
          Task {
            clearChatAlert = true
          }
        } label: {
          Image(systemName: clearChatButtonIcon)
        }
        .alert(
          "Are you sure?", isPresented: $clearChatAlert,
          actions: {
            Button("Cancel", role: .cancel, action: {})
            Button(
              "Clear", role: .destructive,
              action: {
                aiChatModel.messages = []
                save_chat_history(aiChatModel.messages, aiChatModel.chat_name + ".json")
                clearChatButtonIcon = "checkmark"
                hard_reload_chat()
                run_after_delay(
                  delay: 1200, function: { clearChatButtonIcon = "eraser.line.dashed.fill" })
              })
          },
          message: {
            Text("The message history will be cleared")
          })
        Button {
          Task {
            hard_reload_chat()
            reloadButtonIcon = "checkmark"
            run_after_delay(
              delay: 1200, function: { reloadButtonIcon = "arrow.counterclockwise.circle" })
          }
        } label: {
          Image(systemName: reloadButtonIcon)
        }
        .disabled(aiChatModel.predicting)
        Button {
          Task {
            toggleEditChat = true
            editChatDialog = true
          }
        } label: {
          Image(systemName: "slider.horizontal.3")
        }
      }
      .navigationTitle(aiChatModel.Title)

      // Text input field
      LLMTextInput(
        messagePlaceholder: placeholderString,
        show_attachment_btn: self.aiChatModel.is_mmodal,
        focusedField: $focusedField,
        auto_scroll: $autoScroll,
        enableRAG: $enableRAG
      ).environmentObject(aiChatModel)
        .disabled(self.aiChatModel.chat_name == "")
    }

    // Chat settings sheet
    .sheet(isPresented: $toggleEditChat) {
      ChatSettingsView(
        add_chat_dialog: $toggleEditChat,
        edit_chat_dialog: $editChatDialog,
        chat_name: aiChatModel.chat_name,
        after_chat_edit: $AfterChatEdit,
        toggleSettings: .constant(false)
      ).environmentObject(aiChatModel)
        #if os(macOS)
          .frame(minWidth: 400, minHeight: 600)
        #endif
    }
    .textSelection(.enabled)
    .preferredColorScheme(.light)
    .enableInjection()
  }
}

// MARK: - Preview Provider

//struct ChatView_Previews: PreviewProvider {
//    static var previews: some View {
//        ChatView(chat_selected: .constant(true),model_name: .constant(""),chat_name:.constant(""),title: .constant("Title"),close_chat: {},add_chat_dialog:.constant(false),edit_chat_dialog:.constant(false))
//    }
//}
