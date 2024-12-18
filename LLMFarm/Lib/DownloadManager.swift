//
//  DownloadManager.swift
//  LLMFarm
//
//  Created by guinmoon on 08.09.2023.
//

import Foundation

/// Manages downloading, deleting and checking status of files
@available(iOS 17.0, *)
@Observable final class DownloadManager {
  /// Whether a download is currently in progress
  var isDownloading = false
  
  /// Whether the target file has been downloaded
  var isDownloaded = false

  /// Downloads a video file from a remote URL to the documents directory
  /// - Note: If the file already exists locally, the download will be skipped
  func downloadFile() {
    print("downloadFile")
    isDownloading = true

    let docsUrl = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first

    let destinationUrl = docsUrl?.appendingPathComponent("myVideo.mp4")
    if let destinationUrl = destinationUrl {
      if FileManager().fileExists(atPath: destinationUrl.path) {
        print("File already exists")
        isDownloading = false
      } else {
        let urlRequest = URLRequest(
          url: URL(
            string:
              "https://file-examples-com.github.io/uploads/2017/04/file_example_MP4_480_1_5MG.mp4")!
        )

        let dataTask = URLSession.shared.dataTask(with: urlRequest) { (data, response, error) in

          if let error = error {
            print("Request error: ", error)
            self.isDownloading = false
            return
          }

          guard let response = response as? HTTPURLResponse else { return }

          if response.statusCode == 200 {
            guard let data = data else {
              self.isDownloading = false
              return
            }
            DispatchQueue.main.async {
              do {
                try data.write(to: destinationUrl, options: Data.WritingOptions.atomic)

                DispatchQueue.main.async {
                  self.isDownloading = false
                  self.isDownloaded = true
                }
              } catch let error {
                print("Error decoding: ", error)
                self.isDownloading = false
              }
            }
          }
        }
        dataTask.resume()
      }
    }
  }

  /// Deletes the downloaded video file from the documents directory
  /// - Note: If the file doesn't exist, this operation is a no-op
  func deleteFile() {
    let docsUrl = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first

    let destinationUrl = docsUrl?.appendingPathComponent("myVideo.mp4")
    if let destinationUrl = destinationUrl {
      guard FileManager().fileExists(atPath: destinationUrl.path) else { return }
      do {
        try FileManager().removeItem(atPath: destinationUrl.path)
        print("File deleted successfully")
        isDownloaded = false
      } catch let error {
        print("Error while deleting video file: ", error)
      }
    }
  }

  /// Checks if the video file exists in the documents directory and updates `isDownloaded` accordingly
  func checkFileExists() {
    let docsUrl = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first

    let destinationUrl = docsUrl?.appendingPathComponent("myVideo.mp4")
    if let destinationUrl = destinationUrl {
      if FileManager().fileExists(atPath: destinationUrl.path) {
        isDownloaded = true
      } else {
        isDownloaded = false
      }
    } else {
      isDownloaded = false
    }
  }

  // Commented out AVPlayer functionality
  //    func getVideoFileAsset() -> AVPlayerItem? {
  //        let docsUrl = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first
  //
  //        let destinationUrl = docsUrl?.appendingPathComponent("myVideo.mp4")
  //        if let destinationUrl = destinationUrl {
  //            if (FileManager().fileExists(atPath: destinationUrl.path)) {
  //                let avAssest = AVAsset(url: destinationUrl)
  //                return AVPlayerItem(asset: avAssest)
  //            } else {
  //                return nil
  //            }
  //        } else {
  //            return nil
  //        }
  //    }
}
