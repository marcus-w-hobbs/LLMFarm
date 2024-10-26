//
//  SummaryIntent.swift
//  LLMFarm
//
//  Created by guinmoon on 16.10.2024.
//

import Foundation
import AppIntents
import llmfarm_core_cpp

struct LLMSummaryIntent: AppIntent {
    static let title: LocalizedStringResource = "Summarize text"
    static let description: LocalizedStringResource = "Create text summary with LLM"
    
    /// Launch your app when the system triggers this intent.
    //    static let openAppWhenRun: Bool = true
    
    
    @Parameter(title: "Token Limit", default: 150)
    var token_limit: Int

    @Parameter(title: "Use history", default: false)
    var use_history: Bool
    
    @Parameter(title: "Chat")
    var chat: ShortcutsChatEntity?
    
    @Parameter(title: "Query")
    var query: String?
    
//    @Parameter(
//        title: "Image",
//        description: "Image to Multimodal",
//        supportedTypeIdentifiers: ["public.image"],
//        inputConnectionBehavior: .connectToPreviousIntentResult
//    )
//    var imageUrls: [IntentFile]?
    
    /// Define the method that the system calls when it triggers this event.
    @MainActor
    func perform() async throws -> some IntentResult & ReturnsValue<String> {

        var img_path:String? = nil
        
//        if let imageUrls = imageUrls?.compactMap({ $0.data }), !imageUrls.isEmpty {
//            #if os(iOS)
//            let ui_img = UIImage(data: imageUrls[0])?.fixedOrientation
//            #else
//            let ui_img = UIImage(data: imageUrls[0])
//            #endif
//            if ui_img != nil {
//                img_path = save_image_from_library_to_cache(ui_img)
//                if img_path != nil{
//                    img_path = get_path_by_short_name(img_path!,dest: "cache/images")
////                    print(img_path)
//                }
//            }
//        }
        if (query == nil){
            return .result(value: "Query is empty.")
        }
        if (chat == nil){
            return .result(value: "Please select chat.")
        }
        
        let trimmedQuery = query!.trimmingCharacters(in: .whitespacesAndNewlines) + "\n\n## Summary:\n"
        
        let res = one_short_query(trimmedQuery,chat!.chat,token_limit,img_path:img_path,use_history: use_history)
        return .result(value: res)
        
    }
    
}