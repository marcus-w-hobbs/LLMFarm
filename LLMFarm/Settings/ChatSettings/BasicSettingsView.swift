//
//  BasicSettingsView.swift
//  LLMFarm
//
//  Created by guinmoon on 22.06.2024.
//

import Inject
import SwiftUI

/// A view that provides basic configuration options for a chat session, including:
/// - Chat title and icon selection
/// - Model settings template selection
struct BasicSettingsView: View {
  @ObserveInjection var inject  // Enables hot reloading during development

  // MARK: - View State
  @Binding var chat_title: String  // Title displayed for the chat
  @Binding var model_icon: String  // Currently selected model icon
  @Binding var model_icons: [String]  // Available model icons to choose from
  @Binding var model_inferences: [String]  // Available inference backends
  @Binding var ggjt_v3_inferences: [String]  // GGJT v3 specific inference options
  @Binding var model_inference: String  // Selected inference backend
  @Binding var ggjt_v3_inference: String  // Selected GGJT v3 inference option
  @Binding var model_inference_inner: String  // Internal inference selection state
  @Binding var model_settings_template: ChatSettingsTemplate  // Currently selected settings template
  @Binding var model_setting_templates: [ChatSettingsTemplate]  // Available settings templates
  @Binding var applying_template: Bool  // Whether a template is currently being applied

  // MARK: - Actions
  /// Callback to apply a selected settings template
  var apply_setting_template: (ChatSettingsTemplate) -> Void

  var body: some View {
    // MARK: - Title and Icon Selection
    HStack {
      // Icon picker showing available model icons in a menu
      Picker("", selection: $model_icon) {
        ForEach(model_icons, id: \.self) { img in
          Image(img + "_48")
            .resizable()
            .background(Color("color_bg_inverted").opacity(0.05))
            .padding(EdgeInsets(top: 7, leading: 5, bottom: 7, trailing: 5))
            .frame(width: 48, height: 48)
            .clipShape(Circle())
        }
      }
      .pickerStyle(.menu)
      .frame(maxWidth: 80, alignment: .leading)
      .frame(height: 48)

      // Platform-specific text field for chat title
      #if os(macOS)
        // macOS uses custom text field that reports when editing ends
        DidEndEditingTextField(text: $chat_title, didEndEditing: { newName in })
          .frame(maxWidth: .infinity, alignment: .leading)
      #else
        // iOS uses standard text field
        TextField("Title...", text: $chat_title)
          .frame(maxWidth: .infinity, alignment: .leading)
          .textFieldStyle(.plain)
      #endif
    }
    .padding([.top])

    // MARK: - Settings Template Selection
    HStack {
      Text("Settings template:")
        .frame(maxWidth: .infinity, alignment: .leading)
      Picker("", selection: $model_settings_template) {
        ForEach(model_setting_templates, id: \.self) { template in
          Text(template.template_name).tag(template)
        }
      }
      .onChange(of: model_settings_template) { tmpl in
        applying_template = true
        apply_setting_template(model_settings_template)
      }
      .pickerStyle(.menu)
    }
    .padding(.horizontal, 5)
    .padding(.top, 8)
    .enableInjection()

    // Note: Inference selection UI is currently commented out but preserved for future use
    // when supporting multiple inference backends like Minicpm, Bunny etc.
  }
}

// Preview provider commented out but available for testing
//#Preview {
//    BasicSettingsView()
//}
