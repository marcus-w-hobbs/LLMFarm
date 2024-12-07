//
//  ChatListView.swift
//  ChatUI
//
//  Created by Shezad Ahamed on 05/08/21.
//
// A view that displays a list of chats and provides functionality to manage them

import Inject
import SwiftUI

struct ChatListView: View {
  // Enable hot reloading during development
  @ObserveInjection var inject

  // Environment objects for model state
  @EnvironmentObject var fineTuneModel: FineTuneModel // Manages fine-tuning settings
  @EnvironmentObject var aiChatModel: AIChatModel // Manages chat state and AI interactions

  // View state
  @State var searchText: String = "" // Search filter text
  @Binding var tabSelection: Int // Currently selected tab
  @Binding var model_name: String // Name of the selected AI model
  @Binding var title: String // Title of the current chat
  @Binding var add_chat_dialog: Bool // Controls visibility of add chat dialog
  var close_chat: () -> Void // Callback to close current chat
  @Binding var edit_chat_dialog: Bool // Controls visibility of edit chat dialog
  @Binding var chat_selection: [String: String]? // Currently selected chat
  @Binding var after_chat_edit: () -> Void // Callback after editing a chat
  @State var chats_previews: [[String: String]] = [] // List of chat previews
  @State var current_detail_view_name: String? = "Chat" // Name of current detail view
  @State private var toggleSettings = false // Controls settings sheet visibility
  @State private var toggleAddChat = false // Controls add chat sheet visibility

  // Refreshes the list of chats from storage
  func refresh_chat_list() {
    if is_first_run() {
      create_demo_chat()
    }
    self.chats_previews = get_chats_list() ?? []
    aiChatModel.update_chat_params()
  }

  // Deletes chats at specified offsets in the list
  func Delete(at offsets: IndexSet) {
    let chatsToDelete = offsets.map { self.chats_previews[$0] }
    _ = deleteChats(chatsToDelete)
    refresh_chat_list()
  }

  // Deletes a specific chat
  func Delete(at elem: [String: String]) {
    _ = deleteChats([elem])
    self.chats_previews.removeAll(where: { $0 == elem })
    refresh_chat_list()
  }

  // Creates a duplicate of a chat
  func Duplicate(at elem: [String: String]) {
    _ = duplicateChat(elem)
    refresh_chat_list()
  }

  var body: some View {
    VStack(alignment: .leading, spacing: 5) {
      VStack {
        // List of chat items
        List(selection: $chat_selection) {
          ForEach(chats_previews, id: \.self) { chat_preview in
            NavigationLink(value: chat_preview) {
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

      // Empty state view when no chats exist
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
    // Toolbar with settings and add chat buttons
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
    // Settings sheet
    .sheet(isPresented: $toggleSettings) {
      SettingsView(current_detail_view_name: $current_detail_view_name).environmentObject(
        fineTuneModel
      )
      #if os(macOS)
        .frame(minWidth: 400, minHeight: 600)
      #endif
    }
    // Add/Edit chat sheet
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

// Preview commented out for brevity
