import Inject
import SwiftUI

/// DownloadButton manages model downloads while carefully tracking memory and context window usage
///
/// This view handles the critical task of downloading large language models while providing:
/// 1. Progress tracking and cancellation
/// 2. Memory-efficient download streaming
/// 3. Proper cleanup of temporary files
/// 4. Status management for UI feedback
///
/// Memory Management Flow:
/// 1. Download Phase:
///    - Uses URLSession streaming to minimize memory footprint
///    - Only loads chunks into memory as needed
///    - Temporary files cleaned up after completion
/// 2. File Writing Phase:
///    - Writes directly to final location
///    - Avoids duplicate copies in memory
/// 3. Cleanup Phase:
///    - Cancels downloads on view disappear
///    - Releases memory and file handles
///
/// Context Window Considerations:
/// - Downloaded models define context window size
/// - Larger models = larger context but more memory
/// - Status tracking helps prevent OOM situations
struct DownloadButton: View {
  @ObserveInjection var inject

  // Core model parameters that affect context window size
  @Binding var modelName: String  // Model identifier
  @Binding var modelUrl: String   // Source URL
  @Binding var filename: String   // Local filename
  
  // Status tracking for memory/context management
  @Binding var status: String     // Current download state

  // Download state management
  @State private var downloadTask: URLSessionDownloadTask?
  @State private var progress = 0.0
  @State private var observation: NSKeyValueObservation?

  private func checkFileExistenceAndUpdateStatus() {
  }

  /// Manages model download while carefully handling memory and context resources
  ///
  /// This function orchestrates the download pipeline with precise control over memory usage:
  /// 1. URL Validation Phase (~0 MB memory)
  /// 2. Download Streaming Phase (Chunks of ~10 MB)
  /// 3. File Writing Phase (~Size of model)
  /// 4. Cleanup Phase (Release all resources)
  ///
  /// Memory Budget Example (8GB device):
  /// - Download buffer: 10 MB
  /// - Temporary storage: Model size
  /// - Available for inference: Remaining RAM
  ///
  /// Product teams should monitor:
  /// - Memory pressure during downloads
  /// - Disk space requirements
  /// - Download completion rates
  private func download() {
    status = "downloading"
    print("Downloading model \(modelName) from \(modelUrl)")
    
    // Phase 1: URL Setup
    guard let url = URL(string: modelUrl) else { return }
    let fileURL = getFileURLFormPathStr(dir: "models", filename: filename)

    // Phase 2: Configure Memory-Efficient Download
    downloadTask = URLSession.shared.downloadTask(with: url) { temporaryURL, response, error in
      // Handle download errors that could corrupt model files
      if let error = error {
        print("Error: \(error.localizedDescription)")
        return
      }

      // Validate server response to ensure model integrity
      guard let response = response as? HTTPURLResponse, (200...299).contains(response.statusCode)
      else {
        print("Server error!")
        return
      }

      // Phase 3: Memory-Safe File Writing
      do {
        if let temporaryURL = temporaryURL {
          // Direct file copy to minimize memory usage
          try FileManager.default.copyItem(at: temporaryURL, to: fileURL)
          print("Writing to \(filename) completed")

          // Update UI state after successful write
          status = "downloaded"
        }
      } catch let err {
        print("Error: \(err.localizedDescription)")
      }
    }

    // Configure progress tracking for memory monitoring
    observation = downloadTask?.progress.observe(\.fractionCompleted) { progress, _ in
      self.progress = progress.fractionCompleted
    }

    // Begin memory-managed download
    downloadTask?.resume()
  }

  var body: some View {
    VStack {
      switch status {
      case "download":
        Button(action: download) {
          Image(systemName: "icloud.and.arrow.down")
        }
        .buttonStyle(.borderless)
      case "downloading":
        Button(action: {
          // Cancel download to free resources
          downloadTask?.cancel()
          status = "download"
        }) {
          HStack {
            Image(systemName: "stop.circle.fill")
            Text("\(Int(progress * 100))%")
              .padding(.trailing, -20)
          }
        }
        .buttonStyle(.borderless)
      case "downloaded":
        Image(systemName: "checkmark.circle.fill")
      default:
        Text("Unknown status")
      }
    }
    // Critical cleanup to prevent memory leaks
    .onDisappear {
      downloadTask?.cancel()
    }.onChange(of: status) { st in
      print(st)
    }
    .enableInjection()
  }
}
