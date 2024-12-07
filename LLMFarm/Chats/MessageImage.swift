//
//  MessageImage.swift
//  LLMFarm
//
//  Created by guinmoon on 16.03.2024.
//

import Inject
import PhotosUI
import SwiftUI

/// A view that displays an image attachment from a chat message
struct MessageImage: View {
  // Enable hot reloading during development
  @ObserveInjection var inject

  /// The message containing the image attachment to display
  var message: Message

  /// Maximum width constraint for the image
  var maxWidth: CGFloat = 300

  /// Maximum height constraint for the image
  var maxHeight: CGFloat = 300

  /// Helper function for debug printing
  /// - Parameter str: String to print
  func print_(_ str: String) {
    print(str)
  }

  var body: some View {
    Group {
      // Only show image if message has an image attachment
      if message.attachment != nil && message.attachment_type != nil
        && message.attachment_type == "img"
      {
        // Get full path to cached image
        let img_path = get_path_by_short_name(message.attachment!, dest: "cache/images")
        if img_path != nil {
          #if os(macOS)
            // Load and display image on macOS
            let ns_img = NSImage(contentsOfFile: img_path!)
            if ns_img != nil {
              // Calculate width while maintaining aspect ratio
              let w = CGFloat(ns_img!.width * maxHeight / ns_img!.height)
              Image(nsImage: ns_img!)  //переделать
                .resizable()
                .aspectRatio(contentMode: .fit)
                .frame(maxWidth: w, maxHeight: maxHeight)
            }
          #elseif os(iOS)
            // Load and display image on iOS
            let ui_img = UIImage(contentsOfFile: img_path!)
            if ui_img != nil {
              // Calculate width while maintaining aspect ratio
              let w = CGFloat(ui_img!.cgImage!.width * Int(maxHeight) / ui_img!.cgImage!.height)
              Image(uiImage: ui_img!.fixedOrientation)
                .resizable()
                .aspectRatio(contentMode: .fit)
                .frame(maxWidth: w, maxHeight: maxHeight)
            }
          #endif
        }
      }
    }.enableInjection()
  }
}
