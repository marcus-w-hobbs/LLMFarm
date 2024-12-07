/// DownloadModelsView manages the model download UI while carefully tracking memory and context window implications
///
/// This view orchestrates the critical task of downloading and managing large language models while providing:
/// 1. Model Selection Phase:
///    - Lists available models with metadata like context size
///    - Shows download status and memory requirements
///    - Enables importing custom models
/// 2. Download Management Phase:
///    - Tracks download progress and memory usage
///    - Provides cancellation and cleanup
///    - Validates model compatibility
/// 3. Model Loading Phase:
///    - Initializes context window based on model size
///    - Manages memory allocation for inference
///    - Configures optimal batch sizes
///
/// Context Window Considerations:
/// - Each model defines max context size (e.g. 4K, 8K, 32K tokens)
/// - Larger contexts require more GPU/CPU memory
/// - Memory usage scales quadratically with context size
/// - Product teams should monitor:
///   1. Memory pressure during loads
///   2. Context window utilization
///   3. Inference latency vs context size
///   4. Download completion rates
struct DownloadModelsView: View {
  // Search state for filtering models by context size, memory requirements
  @State var searchText: String = ""
  
  // Available models with metadata about context windows and memory needs
  @State var models_info: [DownloadModelInfo] =
    get_downloadble_models("downloadable_models.json") ?? []
    
  // Currently selected model for detailed context/memory analysis
  @State var model_selection: String?
  
  // Import flow state management
  @State private var isImporting: Bool = false
  @State private var modelImported: Bool = false
  
  // Supported model formats and their typical context sizes
  let bin_type = UTType(tag: "bin", tagClass: .filenameExtension, conformingTo: nil)  // Legacy format
  let gguf_type = UTType(tag: "gguf", tagClass: .filenameExtension, conformingTo: nil) // Modern format with flexible context
  
  // Model file management for context validation
  @State private var model_file_url: URL = URL(filePath: "")
  @State private var model_file_name: String = ""
  @State private var model_file_path: String = "select model"
  @State private var add_button_icon: String = "plus.app"

  // Helper to reset UI state after model operations
  private func delayIconChange() {
    DispatchQueue.main.asyncAfter(deadline: .now() + 1.5) {
      add_button_icon = "plus.app"
    }
  }

  var body: some View {
    ZStack {
      VStack {
        // Main model list with context window metadata
        VStack(spacing: 5) {
          List(selection: $model_selection) {
            ForEach(models_info, id: \.self) { model_info in
              // Each row shows model details including context size
              ModelDownloadItem(modelInfo: model_info)
            }
          }
          #if os(macOS)
            .listStyle(.sidebar)
          #else
            .listStyle(InsetListStyle())
          #endif
        }
        
        // Empty state with import option
        if models_info.count <= 0 {
          VStack {
            Button {
              Task {
                isImporting.toggle()
              }
            } label: {
              Image(systemName: "plus.square.dashed")
                .foregroundColor(.secondary)
                .font(.system(size: 40))
            }
            .buttonStyle(.borderless)
            .controlSize(.large)
            Text("Add model")
              .font(.title3)
              .frame(maxWidth: .infinity)
          }.opacity(0.4)
            .frame(maxWidth: .infinity, alignment: .center)
        }
      }
    }
    .toolbar {
    }
    .navigationTitle("Download models")
  }
}
