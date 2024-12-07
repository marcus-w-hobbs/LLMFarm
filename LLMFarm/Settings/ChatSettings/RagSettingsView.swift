//
//  RagSettingsView.swift
//  LLMFarm
//
//  Created by guinmoon on 20.10.2024.
//

// RagSettingsView provides a comprehensive interface for configuring and managing RAG (Retrieval Augmented Generation)
// settings in the LLM inference pipeline. It allows fine-grained control over:
//
// - Text chunking parameters for optimal context window utilization
// - Embedding model selection for semantic search
// - Similarity metrics for retrieval accuracy
// - Text splitting strategies for context preservation
// - Real-time index building and testing
//
// The view is designed to help product engineers optimize the RAG pipeline by providing:
// 1. Direct control over chunk size/overlap to manage token usage
// 2. Multiple embedding models to balance speed vs accuracy
// 3. Various similarity metrics to tune retrieval precision
// 4. Debug tools to test and validate the RAG setup
// 5. Real-time search testing to verify retrieval quality

import SimilaritySearchKit
import SimilaritySearchKitDistilbert
import SimilaritySearchKitMiniLMAll
import SimilaritySearchKitMiniLMMultiQA
import SwiftUI

struct RagSettingsView: View {
  // Directory path for storing RAG index and documents
  @State var ragDir: String

  // Debug search input and results
  @State var inputText: String = ""
  var searchUrl: URL  // URL for document storage
  var ragUrl: URL  // URL for index storage
  var searchResultsCount: Int = 3  // Number of results to return in debug mode
  @State var loadIndexResult: String = ""  // Status message for index operations
  @State var searchResults: String = ""  // Debug search results

  // Core RAG configuration parameters
  @Binding private var chunkSize: Int  // Size of text chunks in tokens
  @Binding private var chunkOverlap: Int  // Overlap between chunks to preserve context
  @Binding private var currentModel: EmbeddingModelType  // Model for semantic embeddings
  @Binding private var comparisonAlgorithm: SimilarityMetricType  // Similarity calculation method
  @Binding private var chunkMethod: TextSplitterType  // Text splitting strategy
  @Binding private var ragTop: Int  // Max number of relevant chunks to retrieve

  // Initialize view with all required RAG parameters
  init(
    ragDir: String,
    chunkSize: Binding<Int>,
    chunkOverlap: Binding<Int>,
    currentModel: Binding<EmbeddingModelType>,
    comparisonAlgorithm: Binding<SimilarityMetricType>,
    chunkMethod: Binding<TextSplitterType>,
    ragTop: Binding<Int>
  ) {
    self.ragDir = ragDir

    // Configure storage paths for documents and index
    self.ragUrl =
      FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first?
      .appendingPathComponent(ragDir) ?? URL(fileURLWithPath: "")
    self.searchUrl =
      FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first?
      .appendingPathComponent(ragDir + "/docs") ?? URL(fileURLWithPath: "")

    // Bind all RAG configuration parameters
    self._chunkSize = chunkSize
    self._chunkOverlap = chunkOverlap
    self._currentModel = currentModel
    self._comparisonAlgorithm = comparisonAlgorithm
    self._chunkMethod = chunkMethod
    self._ragTop = ragTop
  }

  var body: some View {
    ScrollView(showsIndicators: false) {
      VStack {
        GroupBox(
          label:
            Text("RAG Settings")
        ) {
          // Chunk size control - critical for managing token usage
          HStack {
            Text("Chunk Size:")
              .frame(maxWidth: 100, alignment: .leading)
            TextField("size..", value: $chunkSize, format: .number)
              .frame(alignment: .leading)
              .multilineTextAlignment(.trailing)
              .textFieldStyle(.plain)
              #if os(iOS)
                .keyboardType(.numbersAndPunctuation)
              #endif
          }

          // Chunk overlap - ensures context preservation between chunks
          HStack {
            Text("Chunk Overlap:")
              .frame(maxWidth: 100, alignment: .leading)
            TextField("size..", value: $chunkOverlap, format: .number)
              .frame(alignment: .leading)
              .multilineTextAlignment(.trailing)
              .textFieldStyle(.plain)
              #if os(iOS)
                .keyboardType(.numbersAndPunctuation)
              #endif
          }

          // Embedding model selection for semantic search
          HStack {
            Text("Embedding Model:")
              .frame(maxWidth: 100, alignment: .leading)
            Picker("", selection: $currentModel) {
              ForEach(SimilarityIndex.EmbeddingModelType.allCases, id: \.self) { option in
                Text(String(describing: option))
              }
            }
            .frame(maxWidth: .infinity, alignment: .trailing)
            .pickerStyle(.menu)
          }

          // Similarity metric for tuning retrieval precision
          HStack {
            Text("Similarity Metric:")
              .frame(maxWidth: 120, alignment: .leading)
            Picker("", selection: $comparisonAlgorithm) {
              ForEach(SimilarityIndex.SimilarityMetricType.allCases, id: \.self) { option in
                Text(String(describing: option))
              }
            }
            .frame(maxWidth: .infinity, alignment: .trailing)
            .pickerStyle(.menu)
          }

          // Text splitting strategy selection
          HStack {
            Text("Text Splitter:")
              .frame(maxWidth: 120, alignment: .leading)
            Picker("", selection: $chunkMethod) {
              ForEach(TextSplitterType.allCases, id: \.self) { option in
                Text(String(describing: option))
              }
            }
            .frame(maxWidth: .infinity, alignment: .trailing)
            .pickerStyle(.menu)
          }

          // Maximum number of chunks to retrieve
          HStack {
            Text("Max RAG answers count:")
              .frame(maxWidth: 100, alignment: .leading)
            TextField("count..", value: $ragTop, format: .number)
              .frame(alignment: .leading)
              .multilineTextAlignment(.trailing)
              .textFieldStyle(.plain)
              #if os(iOS)
                .keyboardType(.numbersAndPunctuation)
              #endif
          }

        }

        // Debug tools for testing and validation
        GroupBox(
          label:
            Text("RAG Debug")
        ) {
          HStack {
            Button(
              action: {
                Task {
                  await BuildIndex(ragURL: ragUrl)
                }
              },
              label: {
                Text("Rebuild index")
                  .font(.title2)
              }
            )
            .padding()

            Button(
              action: {
                Task {
                  await LoadIndex(ragURL: ragUrl)
                }
              },
              label: {
                Text("Load index")
                  .font(.title2)
              }
            )
            .padding()
          }

          Text(loadIndexResult)

          // Search input field for testing retrieval
          TextField("Search text", text: $inputText, axis: .vertical)
            .onSubmit {
              Task {
                await Search()
              }
            }
            .textFieldStyle(.plain)
            .frame(maxWidth: .infinity)
            .padding(.horizontal, 12)
            .padding(.vertical, 8)
            .background {
              RoundedRectangle(cornerRadius: 20)
                #if os(macOS)
                  .stroke(Color(NSColor.systemGray), lineWidth: 0.2)
                #else
                  .stroke(Color(UIColor.systemGray2), lineWidth: 0.2)
                #endif
                .background {
                  RoundedRectangle(cornerRadius: 20)
                    .fill(.white.opacity(0.1))
                }
                .padding(.trailing, 2)

            }
            .lineLimit(1...5)

          Button(
            action: {
              Task {
                await GeneratePrompt()
              }
            },
            label: {
              Text("Search and Generate Prompt")
                .font(.title2)
            }
          )

          Text(searchResults)
            .padding()
            .textSelection(.enabled)
        }

      }
    }
  }

  // Rebuilds the RAG index with current settings and measures performance
  func BuildIndex(ragURL: URL) async {
    let start = DispatchTime.now()
    updateIndexComponents(
      currentModel: currentModel, comparisonAlgorithm: comparisonAlgorithm, chunkMethod: chunkMethod
    )
    await BuildNewIndex(
      searchUrl: searchUrl,
      chunkSize: chunkSize,
      chunkOverlap: chunkOverlap)
    let end = DispatchTime.now()
    let nanoTime = end.uptimeNanoseconds - start.uptimeNanoseconds
    let timeInterval = Double(nanoTime) / 1_000_000_000
    loadIndexResult = String(timeInterval) + " sec"
    saveIndex(url: ragURL, name: "RAG_index")
  }

  // Loads existing index with current similarity settings
  func LoadIndex(ragURL: URL) async {
    updateIndexComponents(
      currentModel: currentModel, comparisonAlgorithm: comparisonAlgorithm, chunkMethod: chunkMethod
    )
    await loadExistingIndex(url: ragURL, name: "RAG_index")
    loadIndexResult = "Loaded"
  }

  // Performs test search and measures retrieval time
  func Search() async {
    let start = DispatchTime.now()
    let results = await searchIndexWithQuery(query: inputText, top: searchResultsCount)
    let end = DispatchTime.now()
    let nanoTime = end.uptimeNanoseconds - start.uptimeNanoseconds
    let timeInterval = Double(nanoTime) / 1_000_000_000

    searchResults = String(describing: results)
    print(results)

    print("Search time: \(timeInterval) sec")
  }

  // Generates LLM prompt from retrieved chunks for testing
  func GeneratePrompt() async {
    let start = DispatchTime.now()
    let results = await searchIndexWithQuery(query: inputText, top: searchResultsCount)
    let end = DispatchTime.now()
    let nanoTime = end.uptimeNanoseconds - start.uptimeNanoseconds
    let timeInterval = Double(nanoTime) / 1_000_000_000

    if results == nil {
      return
    }

    let llmPrompt = SimilarityIndex.exportLLMPrompt(query: inputText, results: results!)

    searchResults = llmPrompt
    print(llmPrompt)

    print("Search time: \(timeInterval) sec")
  }
}

//#Preview {
//    RagSettingsView()
//}
