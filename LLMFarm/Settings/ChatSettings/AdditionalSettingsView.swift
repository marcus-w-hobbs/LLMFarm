//
//  AdditionalSettingsView.swift
//  LLMFarm
//
//  Created by guinmoon on 22.06.2024.
//

import Inject
import SwiftUI

/// A view that provides additional chat settings like template saving and chat style selection
struct AdditionalSettingsView: View {
  @ObserveInjection var inject  // Enables hot reloading during development

  @Binding var save_load_state: Bool  // Whether to save/load chat state
  @Binding var save_as_template_name: String  // Name for saving current settings as template
  @Binding var chat_style: String  // Selected chat display style
  @Binding var chat_styles: [String]  // Available chat display styles

  /// Callback to get current chat options as dictionary
  /// - Parameter includeState: Whether to include state in options
  /// - Returns: Dictionary of chat options
  var get_chat_options_dict: (Bool) -> [String: Any]
  
  /// Callback to refresh the list of available templates
  var refresh_templates: () -> Void

  var body: some View {
    Group {
      // Template saving section
      VStack {
        Text("Save as new template:")
          .frame(maxWidth: .infinity, alignment: .leading)
          .padding(.horizontal, 5)
        HStack {
          // Platform-specific text field implementation
          #if os(macOS)
            // macOS uses custom text field that reports when editing ends
            DidEndEditingTextField(text: $save_as_template_name, didEndEditing: { newName in })
              .frame(maxWidth: .infinity, alignment: .leading)
          #else
            // iOS uses standard text field
            TextField("New template name...", text: $save_as_template_name)
              .frame(maxWidth: .infinity, alignment: .leading)
              .textFieldStyle(.plain)
          #endif
          
          // Save template button
          Button {
            Task {
              let options = get_chat_options_dict(true)
              _ = CreateChat(
                options, edit_chat_dialog: true, chat_name: save_as_template_name + ".json",
                save_as_template: true)
              refresh_templates()
            }
          } label: {
            Image(systemName: "doc.badge.plus")
          }
          .frame(alignment: .trailing)
        }
        .frame(maxWidth: .infinity)
        .padding(.horizontal, 5)
      }
      .padding(.top)

      // Save/Load state toggle
      HStack {
        Toggle("Save/Load State", isOn: $save_load_state)
          .frame(maxWidth: 220, alignment: .leading)
        Spacer()
      }
      .padding(.top, 5)
      .padding(.horizontal, 5)
      .padding(.bottom, 4)

      // Chat style picker
      HStack {
        Text("Chat Style:")
          .frame(maxWidth: .infinity, alignment: .leading)
        Picker("", selection: $chat_style) {
          ForEach(chat_styles, id: \.self) {
            Text($0)
          }
        }
        .pickerStyle(.menu)
      }
      .padding(.horizontal, 5)
      .padding(.top, 8)
    }
    .enableInjection()
  }
}

// Preview provider commented out but available for testing
//#Preview {
//    AdditionalSettingsView()
//}
