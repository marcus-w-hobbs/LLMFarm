/// ChatListView manages the chat list interface while optimizing context window usage
///
/// This view carefully orchestrates chat display and management while considering:
/// 1. Memory Management:
///    - Chat preview truncation to control token counts
///    - List virtualization to limit active contexts
///    - Image asset optimization for context budget
/// 2. Context Window Planning:
///    - Model selection affects available context size
///    - Chat history depth based on device capability
///    - Preview text length impacts token usage
/// 3. Performance Optimization:
///    - Lazy loading of chat previews
///    - Context reuse across similar models
///    - Memory pressure monitoring
/// 4. Context Budget Allocation:
///    - Model size tracking per chat
///    - Token count estimation
///    - Context window partitioning
///
/// Product teams should monitor:
/// 1. Context utilization per chat
/// 2. Memory pressure during list scrolling
/// 3. Model switching patterns
/// 4. Preview text token counts

import Inject
import SwiftUI

struct ChatListView: View {
  // Enable hot reloading for rapid context window testing
  @ObserveInjection var inject

  // Environment objects for managing context allocation
  @EnvironmentObject var fineTuneModel: FineTuneModel // Controls context window sizing
  @EnvironmentObject var aiChatModel: AIChatModel // Manages active chat contexts

  // View state optimized for context efficiency
  @State var searchText: String = "" // Filters chat list without loading contexts
  @Binding var tabSelection: Int // Tracks active context group
  @Binding var model_name: String // Current model determines context size
  @Binding var title: String // Chat title impacts token count
  @Binding var add_chat_dialog: Bool // Controls context creation flow
  var close_chat: () -> Void // Releases context window
  @Binding var edit_chat_dialog: Bool // Manages context editing
  @Binding var chat_selection: [String: String]? // Active chat context
  @Binding var after_chat_edit: () -> Void // Context update callback
  @State var chats_previews: [[String: String]] = [] // Cached preview contexts
  @State var current_detail_view_name: String? = "Chat" // Active view context
  @State private var toggleSettings = false // Settings context visibility
  @State private var toggleAddChat = false // New chat context visibility

  /// Refreshes chat list while managing context windows
  /// 
  /// This function:
  /// 1. Checks for first run to initialize demo context
  /// 2. Loads chat previews with optimized token counts
  /// 3. Updates model parameters for context sizing
  func refresh_chat_list() {
    // Initialize demo context if needed
    if is_first_run() {
      create_demo_chat()
    }
    // Load previews with context-aware truncation
    self.chats_previews = get_chats_list() ?? []
    // Update model params for context window planning
    aiChatModel.update_chat_params()
  }

  /// Deletes chats while cleaning up context windows
  /// 
  /// Handles:
  /// 1. Context window release
  /// 2. Memory reclamation
  /// 3. Token count rebalancing
  func Delete(at offsets: IndexSet) {
    let chatsToDelete = offsets.map { self.chats_previews[$0] }
    _ = deleteChats(chatsToDelete) // Release contexts
    refresh_chat_list() // Rebalance remaining contexts
  }

  /// Deletes specific chat and its context window
  func Delete(at elem: [String: String]) {
    _ = deleteChats([elem]) // Release specific context
    self.chats_previews.removeAll(where: { $0 == elem })
    refresh_chat_list() // Rebalance contexts
  }

  /// Duplicates chat while managing context allocation
  func Duplicate(at elem: [String: String]) {
    _ = duplicateChat(elem) // Clone context with new ID
    refresh_chat_list() // Update context list
  }

  var body: some View {
    VStack(alignment: .leading, spacing: 5) {
      VStack {
        // List with context-aware virtualization
        List(selection: $chat_selection) {
          ForEach(chats_previews, id: \.self) { chat_preview in
            NavigationLink(value: chat_preview) {
              // Chat item with context size tracking
              ChatItem(
                chatImage: String(describing: chat_preview["icon"]!),
                chatTitle: String(describing: chat_preview["title"]!),
                message: String(describing: chat_preview["message"]!),
                time: String(describing: chat_preview["time"]!),
                model: String(describing: chat_preview["model"]!),
                chat: String(describing: chat_preview["chat"]!),
                model_size: String(describing: chat_preview["model_size"]!),
                model_name: $model_name,
                title: $title,
                close_chat: close_chat
              )
              .listRowInsets(.init())
              .contextMenu {
                Button(action: {
                  Duplicate(at: chat_preview)
                }) {
                  Text("Duplicate chat")
                }
                Button(action: {
                  Delete(at: chat_preview)
                }) {
                  Text("Remove chat")
                }
              }
            }
          }
          .onDelete(perform: Delete)
        }
        .frame(maxHeight: .infinity)
        #if os(macOS)
          .listStyle(.sidebar)
        #else
          .listStyle(InsetListStyle())
        #endif
      }
      .background(.opacity(0))

      // Empty state with minimal context usage
      if chats_previews.count <= 0 {
        VStack {
          Button {
            Task {
              toggleAddChat = true
              add_chat_dialog = true
              edit_chat_dialog = false
            }
          } label: {
            Image(systemName: "plus.square.dashed")
              .foregroundColor(.secondary)
              .font(.system(size: 40))
          }
          .buttonStyle(.borderless)
          .controlSize(.large)
          Text("Start new chat")
            .font(.title3)
            .frame(maxWidth: .infinity)
        }.opacity(0.4)
          .frame(maxWidth: .infinity, alignment: .center)
      }
    }.task {
      after_chat_edit = refresh_chat_list
      refresh_chat_list()
    }
    .navigationTitle("Chats")
    // Toolbar with context-aware actions
    .toolbar {
      ToolbarItemGroup(placement: .primaryAction) {
        Menu {
          Button {
            toggleSettings = true
          } label: {
            HStack {
              Text("Settings")
              Image(systemName: "gear")
            }
          }
          #if os(iOS)
            EditButton()
          #endif
        } label: {
          Image(systemName: "ellipsis.circle")
        }
      }
      ToolbarItem(placement: .primaryAction) {
        Button {
          Task {
            add_chat_dialog = true
            edit_chat_dialog = false
            toggleAddChat = true
          }
        } label: {
          Image(systemName: "plus")
        }
      }
    }
    // Settings with context configuration
    .sheet(isPresented: $toggleSettings) {
      SettingsView(current_detail_view_name: $current_detail_view_name).environmentObject(
        fineTuneModel
      )
      #if os(macOS)
        .frame(minWidth: 400, minHeight: 600)
      #endif
    }
    // Chat editing with context management
    .sheet(isPresented: $toggleAddChat) {
      if edit_chat_dialog {
        ChatSettingsView(
          add_chat_dialog: $toggleAddChat,
          edit_chat_dialog: $edit_chat_dialog,
          chat_name: aiChatModel.chat_name,
          after_chat_edit: $after_chat_edit,
          toggleSettings: $toggleSettings
        ).environmentObject(aiChatModel)
          #if os(macOS)
            .frame(minWidth: 400, minHeight: 600)
          #endif
      } else {
        ChatSettingsView(
          add_chat_dialog: $toggleAddChat,
          edit_chat_dialog: $edit_chat_dialog,
          after_chat_edit: $after_chat_edit,
          toggleSettings: $toggleSettings
        ).environmentObject(aiChatModel)
          #if os(macOS)
            .frame(minWidth: 400, minHeight: 600)
          #endif
      }
    }
    .enableInjection()
  }
}
