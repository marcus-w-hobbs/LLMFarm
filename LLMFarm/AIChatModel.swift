//
//  ChatViewModel.swift
//
//  Created by Artem Savkin
//

// Core imports for functionality
import Foundation
import SimilaritySearchKit  // For semantic search capabilities
import SimilaritySearchKitDistilbert  // Embedding model
import SimilaritySearchKitMiniLMAll  // Embedding model
import SimilaritySearchKitMiniLMMultiQA  // Embedding model for Q&A
import SwiftUI  // UI framework
import llmfarm_core  // Core LLM functionality
import os  // Logging

// Helper extension to convert Duration to seconds
extension Duration {
  fileprivate var seconds: Double {
    Double(components.seconds) + Double(components.attoseconds) / 1.0e18
  }
}

// Global pointer for C interop
var AIChatModel_obj_ptr: UnsafeMutableRawPointer? = nil

// Main chat model class that handles LLM interactions
@MainActor
final class AIChatModel: ObservableObject {

  // Represents different states of the chat model
  enum State {
    case none  // Initial state
    case loading  // Model is loading
    case ragIndexLoading  // RAG index is loading
    case ragSearch  // Performing RAG search
    case completed  // Operation completed
  }

  // Core properties
  public var chat: AI?  // The AI chat instance
  public var modelURL: String  // URL to the model file
  public var numberOfTokens = 0  // Token count
  public var total_sec = 0.0  // Total processing time
  public var action_button_icon = "paperplane"  // UI button icon
  public var model_loading = false  // Model load state
  public var model_name = ""  // Name of current model
  public var chat_name = ""  // Name of current chat
  public var start_predicting_time = DispatchTime.now()  // Prediction start time
  public var first_predicted_token_time = DispatchTime.now()  // First token time
  public var tok_sec: Double = 0.0  // Tokens per second
  public var ragIndexLoaded: Bool = false  // RAG index load state
  private var state_dump_path: String = ""  // Path to state dump

  private var title_backup = ""  // Backup of chat title
  private var messages_lock = NSLock()  // Thread safety for messages

  // RAG configuration
  public var ragUrl: URL  // URL for RAG data
  private var ragTop: Int = 3  // Top k results
  private var chunkSize: Int = 256  // Text chunk size
  private var chunkOverlap: Int = 100  // Chunk overlap size
  private var currentModel: EmbeddingModelType = .minilmMultiQA  // Embedding model
  private var comparisonAlgorithm: SimilarityMetricType = .dotproduct  // Similarity metric
  private var chunkMethod: TextSplitterType = .recursive  // Text splitting method

  // Published properties for UI updates
  @Published var predicting = false  // Prediction state
  @Published var AI_typing = 0  // AI typing indicator
  @Published var state: State = .none  // Current state
  @Published var messages: [Message] = []  // Chat messages
  @Published var load_progress: Float = 0.0  // Load progress
  @Published var Title: String = ""  // Chat title
  @Published var is_mmodal: Bool = false  // Multi-modal flag
  @Published var cur_t_name: String = ""  // Current token name
  @Published var cur_eval_token_num: Int = 0  // Current eval token
  @Published var query_tokens_count: Int = 0  // Query token count

  // Initialize chat model
  public init() {
    chat = nil
    modelURL = ""
    let ragDir = GetRagDirRelPath(chat_name: self.chat_name)
    ragUrl =
      FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first?
      .appendingPathComponent(ragDir) ?? URL(fileURLWithPath: "")
  }

  // Reset RAG URL based on chat name
  public func ResetRAGUrl() {
    let ragDir = GetRagDirRelPath(chat_name: self.chat_name)
    ragUrl =
      FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first?
      .appendingPathComponent(ragDir) ?? URL(fileURLWithPath: "")
  }

  // Callback for model load progress
  private func model_load_progress_callback(_ progress: Float) -> Bool {
    DispatchQueue.main.async {
      self.load_progress = progress
    }
    return true
  }

  // Callback for token evaluation
  private func eval_callback(_ t: Int) -> Bool {
    DispatchQueue.main.async {
      if t == 0 {
        self.cur_eval_token_num += 1
      }
    }
    return false
  }

  /// Handles model load completion and initializes the first inference with careful context window management
  ///
  /// This function is called after the LLM is loaded and validates the model state before starting inference.
  /// It carefully manages the initial context window setup, particularly around system prompts which can consume
  /// significant context budget (typically 200-500 tokens).
  ///
  /// Context Window Management Flow:
  /// 1. Model Load Validation (~0 tokens)
  ///    - Verifies model loaded successfully
  ///    - Ensures context window is initialized
  /// 2. System Prompt Setup (~200-500 tokens)
  ///    - Only added on first inference (nPast == 0)
  ///    - Configurable via contextParams
  ///    - Critical for setting model behavior
  /// 3. Skip Token Parsing (~0-50 tokens)
  ///    - Prepares special tokens that may impact context
  /// 4. Initial Inference Launch
  ///    - Passes context configuration to send()
  ///    - Manages attachment context if provided
  ///
  /// - Parameters:
  ///   - load_result: Model load status string
  ///   - in_text: Initial message text
  ///   - attachment: Optional context like RAG results
  ///   - attachment_type: Type of attachment context
  private func after_model_load(
    _ load_result: String,
    in_text: String,
    attachment: String? = nil,
    attachment_type: String? = nil
  ) {
    // Phase 1: Model Load Validation
    // Critical check - prevents context window corruption
    if load_result != "[Done]" || self.chat?.model == nil || self.chat?.model!.context == nil {
      self.finish_load(append_err_msg: true, msg_text: "Load Model Error: \(load_result)")
      return
    }

    // Phase 2: Load Completion
    self.finish_load()

    // Debug context parameters
    // Product teams can monitor these for context tuning
    print(self.chat?.model?.contextParams as Any)
    print(self.chat?.model?.sampleParams as Any)

    self.model_loading = false

    // Phase 3: System Prompt Context Management
    // Only initialize on first inference to avoid context duplication
    var system_prompt: String? = nil
    if self.chat?.model?.contextParams.system_prompt != "" && self.chat?.model?.nPast == 0 {
      // Add newline padding for clean context separation
      system_prompt = self.chat?.model?.contextParams.system_prompt ?? " " + "\n"

      // Store in message header for context tracking
      self.messages[self.messages.endIndex - 1].header =
        self.chat?.model?.contextParams.system_prompt ?? ""
    }

    // Phase 4: Token Management
    // Parse special tokens that may impact context window
    self.chat?.model?.parse_skip_tokens()

    // Phase 5: Launch Initial Inference
    // Careful context handoff to send() function
    Task {
      await self.send(
        message: in_text,
        append_user_message: false,  // Skip user message context since handled here
        system_prompt: system_prompt,  // Pass system prompt context if configured
        attachment: attachment,  // Optional RAG or other context
        attachment_type: attachment_type)
    }
  }

  // Hard reload chat model
  public func hard_reload_chat() {
    self.remove_dump_state()
    if self.chat != nil && self.chat?.model != nil {
      self.chat!.model!.contextParams.save_load_state = false
    }
    self.chat = nil
  }

  // Remove dump state file
  public func remove_dump_state() {
    if FileManager.default.fileExists(atPath: self.state_dump_path) {
      try? FileManager.default.removeItem(atPath: self.state_dump_path)
    }
  }

  /// Reloads a chat session with new parameters while carefully managing context window resources
  ///
  /// This function handles the critical task of resetting and reloading a chat session's state.
  /// From a context window perspective, this is a key opportunity to:
  /// 1. Clear existing context by resetting message history
  /// 2. Load only essential history within context budget
  /// 3. Reset RAG state to free up context for new session
  /// 4. Prepare clean slate for next inference
  ///
  /// Context Window Budget Example:
  /// - Fresh context window: 4096 tokens
  /// - System prompt: ~500 tokens
  /// - Chat history: Variable, loaded incrementally
  /// - RAG context: 0 tokens (reset)
  /// - Available for next query: ~3500 tokens
  ///
  /// - Parameter chat_selection: Dictionary containing chat configuration:
  ///   - "chat": Chat name/ID (used for history loading)
  ///   - "title": Display title
  ///   - "mmodal": Multi-modal flag ("1" if enabled)
  /// - Important: Context Management Steps:
  ///   1. Stops any active predictions to clear working memory
  ///   2. Thread-safe message history reset
  ///   3. Loads minimal required history
  ///   4. Resets RAG to avoid context pollution
  public func reload_chat(_ chat_selection: [String: String]) {
    // Phase 1: Stop active inference to free context
    self.stop_predict()

    // Phase 2: Update chat metadata
    self.chat_name = chat_selection["chat"] ?? "Not selected"
    self.Title = chat_selection["title"] ?? ""
    self.is_mmodal = chat_selection["mmodal"] ?? "" == "1"

    // Phase 3: Thread-safe history management
    // Critical for context integrity
    messages_lock.lock()
    self.messages = []  // Clear existing context
    self.messages = load_chat_history(chat_selection["chat"]! + ".json") ?? []  // Load minimal history
    messages_lock.unlock()

    // Phase 4: State management
    self.state_dump_path = get_state_path_by_chat_name(chat_name) ?? ""
    ResetRAGUrl()  // Reset RAG to free context

    // Phase 5: Reset inference state
    self.ragIndexLoaded = false  // Clear RAG context
    self.AI_typing = -Int.random(in: 0..<100000)  // Reset typing indicator
  }

  // Update chat parameters
  public func update_chat_params() {
    let chat_config = getChatInfo(self.chat?.chatName ?? "")
    if chat_config == nil {
      return
    }
    self.chat?.model?.contextParams = get_model_context_param_by_config(chat_config!)
    self.chat?.model?.sampleParams = get_model_sample_param_by_config(chat_config!)
  }

  // Format system prompt for llama format
  private func _formatSystemPrompt(_ prompt: String) -> String {
    return
      "[system](<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n\(prompt)<|eot_id|>)\n\n\n<|start_header_id|>user<|end_header_id|>\n\n\n{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>"
  }

  // Prepare model loading by chat name
  public func load_model_by_chat_name_prepare(
    _ chat_name: String, in_text: String,
    attachment: String? = nil,
    attachment_type: String? = nil
  ) -> Bool? {
    guard let chat_config = getChatInfo(chat_name) else {
      return nil
    }

    if chat_config["model_inference"] == nil || chat_config["model"] == nil {
      return nil
    }

    self.model_name = chat_config["model"] as! String
    if let m_url = get_path_by_short_name(self.model_name) {
      self.modelURL = m_url
    } else {
      return nil
    }

    if self.modelURL == "" {
      return nil
    }

    var model_sample_param = ModelSampleParams.default
    var model_context_param = ModelAndContextParams.default
    model_sample_param = get_model_sample_param_by_config(chat_config)
    model_context_param = get_model_context_param_by_config(chat_config)

    // Sample and Context overrides
    if true {
      model_context_param.system_prompt = _formatSystemPrompt(AIChatModel.SYSTEM_PROMPT_NIETZSCHE)
      model_context_param.n_predict = 256
      model_sample_param.temp = 0.7
      model_sample_param.repeat_penalty = 1.5
      model_sample_param.top_p = 0.9
    }

    if chat_config["grammar"] != nil && chat_config["grammar"] as! String != "<None>"
      && chat_config["grammar"] as! String != ""
    {
      let grammar_path = get_grammar_path_by_name(chat_config["grammar"] as! String)
      model_context_param.grammar_path = grammar_path
    }

    // RAG configuration
    self.chunkSize = chat_config["chunk_size"] as? Int ?? self.chunkSize
    self.chunkOverlap = chat_config["chunk_overlap"] as? Int ?? self.chunkOverlap
    self.ragTop = chat_config["rag_top"] as? Int ?? self.ragTop
    if chat_config["current_model"] != nil {
      self.currentModel = getCurrentModelFromStr(chat_config["current_model"] as? String ?? "")
    }
    if chat_config["comparison_algorithm"] != nil {
      self.comparisonAlgorithm = getComparisonAlgorithmFromStr(
        chat_config["comparison_algorithm"] as? String ?? "")
    }
    if chat_config["chunk_method"] != nil {
      self.chunkMethod = getChunkMethodFromStr(chat_config["chunk_method"] as? String ?? "")
    }

    AIChatModel_obj_ptr = nil
    self.chat = nil
    self.chat = AI(_modelPath: modelURL, _chatName: chat_name)
    if self.chat == nil {
      return nil
    }
    self.chat?.initModel(model_context_param.model_inference, contextParams: model_context_param)
    if self.chat?.model == nil {
      return nil
    }
    self.chat?.model?.sampleParams = model_sample_param
    self.chat?.model?.contextParams = model_context_param

    return true
  }

  // Load model by chat name
  public func load_model_by_chat_name(
    _ chat_name: String,
    in_text: String,
    attachment: String? = nil,
    attachment_type: String? = nil
  ) -> Bool? {
    self.model_loading = true

    if self.chat?.model?.contextParams.save_load_state == true {
      self.chat?.model?.contextParams.state_dump_path = get_state_path_by_chat_name(chat_name) ?? ""
    }

    self.chat?.model?.modelLoadProgressCallback = { progress in
      return self.model_load_progress_callback(progress)
    }
    self.chat?.model?.modelLoadCompleteCallback = { load_result in
      self.chat?.model?.evalCallback = self.eval_callback
      self.after_model_load(
        load_result, in_text: in_text, attachment: attachment, attachment_type: attachment_type)
    }
    self.chat?.loadModel()

    return true
  }

  // Update last message in chat
  private func update_last_message(_ message: inout Message) {
    messages_lock.lock()
    if messages.last != nil {
      messages[messages.endIndex - 1] = message
    }
    messages_lock.unlock()
  }

  // Save chat history and state
  public func save_chat_history_and_state() {
    save_chat_history(self.messages, self.chat_name + ".json")
    if self.chat != nil && self.chat?.model != nil {
      self.chat?.model?.save_state()
    }
  }

  // Stop prediction
  public func stop_predict(is_error: Bool = false) {
    self.chat?.flagExit = true
    self.total_sec =
      Double((DispatchTime.now().uptimeNanoseconds - self.start_predicting_time.uptimeNanoseconds))
      / 1_000_000_000
    if let last_message = messages.last {
      messages_lock.lock()
      if last_message.state == .predicting || last_message.state == .none {
        messages[messages.endIndex - 1].state = .predicted(totalSecond: self.total_sec)
        messages[messages.endIndex - 1].tok_sec = Double(self.numberOfTokens) / self.total_sec
      }
      if is_error {
        messages[messages.endIndex - 1].state = .error
      }
      messages_lock.unlock()
    }
    self.predicting = false
    self.tok_sec = Double(self.numberOfTokens) / self.total_sec
    self.numberOfTokens = 0
    self.action_button_icon = "paperplane"
    self.AI_typing = 0
    self.save_chat_history_and_state()
    if is_error {
      self.chat = nil
    }

  }

  /// Validates predicted tokens against stop words to optimize context window usage during inference
  /// - Parameters:
  ///   - token: The current token being generated by the LLM
  ///   - message_text: The accumulated message text being built up during generation
  /// - Returns: Boolean indicating whether to continue generating (true) or stop (false)
  /// - Important: Context Window Management:
  ///   1. Early Termination:
  ///      - Immediately stops generation if exact stop word match found
  ///      - Prevents wasting context budget on unwanted completions
  ///      - Critical for maintaining tight context control
  ///   2. Suffix Handling:
  ///      - Checks if message ends with any stop sequence
  ///      - Removes matched stop sequence to keep context clean
  ///      - Avoids context pollution from partial stop sequences
  ///   3. Context Efficiency:
  ///      - Stop words free up context early when matched
  ///      - Clean message text preserves context budget
  ///      - Enables more efficient context reuse
  /// - Note: Context window optimization opportunities:
  ///   1. Track stop word hit frequency to tune prompt engineering
  ///   2. Monitor context savings from early termination
  ///   3. Analyze stop word effectiveness for context management
  public func check_stop_words(_ token: String, _ message_text: inout String) -> Bool {
    // Default to continuing generation unless stop condition found
    let check = true

    // Iterate through configured stop words/sequences
    for stop_word in self.chat?.model?.contextParams.reverse_prompt ?? [] {
      // Case 1: Exact token match - immediately terminate to save context
      if token == stop_word {
        return false
      }

      // Case 2: Message ends with stop sequence
      if message_text.hasSuffix(stop_word) {
        // Clean up the message by removing the stop sequence
        // This keeps the context window clean for future use
        if stop_word.count > 0 && message_text.count > stop_word.count {
          message_text.removeLast(stop_word.count)
        }
        return false
      }
    }

    // No stop conditions found, safe to continue using context
    return check
  }

  /// Processes each predicted token string during inference and manages context window usage
  /// - Parameters:
  ///   - str: The predicted token string from the LLM
  ///   - time: Time taken for this token prediction
  ///   - message: The message being constructed with the predictions
  /// - Returns: Boolean indicating whether to continue prediction
  /// - Important: Context Window Management:
  ///   1. Stop Word Handling:
  ///      - Checks each token against stop words which can terminate early and free context
  ///      - Early termination preserves context budget for next inference
  ///   2. Token Accounting:
  ///      - Increments numberOfTokens to track context usage
  ///      - Token count helps determine when context window is near capacity
  ///   3. Message State Management:
  ///      - Updates message state which impacts context persistence
  ///      - Predicting state indicates active context consumption
  ///   4. Chat Validation:
  ///      - Verifies chat name matches to prevent context leaks between chats
  ///      - Ensures context is tracked for correct conversation
  /// - Note: Context window optimization opportunities:
  ///   1. Monitor token count trends to predict context overflow
  ///   2. Track stop word hits to optimize prompt engineering
  ///   3. Use message state transitions to compact context when needed
  public func process_predicted_str(
    _ str: String, _ time: Double, _ message: inout Message
  ) -> Bool {
    // Check if token matches any stop words that would terminate generation
    // This helps prevent wasting context on unwanted completions
    let check = check_stop_words(str, &message.text)

    // If stop word found, terminate generation to free up context
    if !check {
      self.stop_predict()
    }

    // Only continue processing if:
    // 1. No stop words found
    // 2. Generation not manually stopped
    // 3. Chat name matches (prevents context pollution between chats)
    if check && self.chat?.flagExit != true && self.chat_name == self.chat?.chatName {
      // Update message state to track context usage
      message.state = .predicting

      // Append new token to message, consuming context
      message.text += str

      // Increment typing indicator for UI feedback
      self.AI_typing += 1

      // Persist message updates to maintain context state
      update_last_message(&message)

      // Track token count for context window budget
      self.numberOfTokens += 1

    } else {
      // Log when generation ends to track context lifecycle
      print("chat ended.")
    }

    // Return check status to control generation flow
    return check
  }

  /// Finalizes model loading and handles any errors that occurred during load
  /// - Parameters:
  ///   - append_err_msg: Whether to append an error message to chat history
  ///   - msg_text: The error message text to append if append_err_msg is true
  /// - Important: Context Window Management:
  ///   1. Error messages consume context tokens, so they should be tracked:
  ///      - System error messages are added to history if append_err_msg=true
  ///      - Error messages count against context budget in next inference
  ///   2. Loading errors may indicate context overflow:
  ///      - Out of memory errors suggest context window is too large
  ///      - Model quantization errors hint at resource constraints
  ///   3. Title restoration impacts context tracking:
  ///      - Backup title is restored which may contain context metadata
  ///      - Title changes should be monitored for context budget planning
  /// - Note: Context window optimization opportunities:
  ///   1. Parse error messages to detect context-related failures
  ///   2. Track error message token counts for budget calculations
  ///   3. Consider clearing history on critical errors to reset context
  public func finish_load(append_err_msg: Bool = false, msg_text: String = "") {
    if append_err_msg {
      self.messages.append(Message(sender: .system, state: .error, text: msg_text, tok_sec: 0))
      self.stop_predict(is_error: true)
    }
    self.state = .completed
    self.Title = self.title_backup
  }

  /// Finalizes a text completion sequence and updates relevant metrics and state
  /// - Parameters:
  ///   - final_str: The final generated text string from the LLM
  ///   - message: The message object to update with completion metrics
  /// - Important: Context Window Management:
  ///   1. Clears temporary state (cur_t_name, load_progress) to free memory
  ///   2. Updates token metrics that can inform future context budgeting:
  ///      - numberOfTokens: Total tokens generated in this completion
  ///      - tok_sec: Tokens/second throughput for performance monitoring
  ///      - total_sec: Total completion time
  ///   3. Saves chat history which impacts available context for next inference:
  ///      - New completed message is persisted
  ///      - Error messages (if any) are added to history
  ///      - State is saved for context recovery
  /// - Note: Context window optimization opportunities:
  ///   1. Monitor tok_sec to identify performance degradation from context size
  ///   2. Track numberOfTokens to budget context for next completion
  ///   3. Use error messages to detect context overflow issues
  public func finish_completion(
    _ final_str: String, _ message: inout Message
  ) {
    // Clear temporary state
    self.cur_t_name = ""
    self.load_progress = 0
    print(final_str)
    self.AI_typing = 0

    // Calculate completion metrics
    self.total_sec =
      Double((DispatchTime.now().uptimeNanoseconds - self.start_predicting_time.uptimeNanoseconds))
      / 1_000_000_000

    // Update message if chat is still active
    if self.chat_name == self.chat?.chatName && self.chat?.flagExit != true {
      // Set tokens/sec throughput metric
      if self.tok_sec != 0 {
        message.tok_sec = self.tok_sec
      } else {
        message.tok_sec = Double(self.numberOfTokens) / self.total_sec
      }
      message.state = .predicted(totalSecond: self.total_sec)
      update_last_message(&message)
    } else {
      print("chat ended.")
    }

    // Reset completion state
    self.predicting = false
    self.numberOfTokens = 0
    self.action_button_icon = "paperplane"

    // Handle errors and persist state
    if final_str.hasPrefix("[Error]") {
      self.messages.append(
        Message(sender: .system, state: .error, text: "Eval \(final_str)", tok_sec: 0))
    }
    self.save_chat_history_and_state()
  }

  /// Loads and initializes the RAG vector index for semantic search
  /// - Parameter ragURL: URL pointing to the directory containing the RAG index files
  /// - Important: Context window considerations:
  ///   1. The index is loaded into memory and does not directly impact context window
  ///   2. Each indexed chunk is sized to ~256 tokens (configurable via chunkSize)
  ///   3. Chunks overlap by 100 tokens (configurable via chunkOverlap) to maintain coherence
  ///   4. When searching, each retrieved chunk will consume:
  ///      - ~256 tokens for the chunk content
  ///      - ~20-30 tokens for metadata and formatting
  ///      - Total ~280-300 tokens per search result
  ///   5. Default top-k of 3 results = ~840-900 tokens of context used
  /// - Note: The index configuration affects token usage:
  ///   - currentModel: Embedding model that determines vector dimensions
  ///   - comparisonAlgorithm: Similarity metric for search (dot product is fastest)
  ///   - chunkMethod: How documents are split (recursive maintains semantic units)
  public func loadRAGIndex(ragURL: URL) async {
    updateIndexComponents(
      currentModel: currentModel, comparisonAlgorithm: comparisonAlgorithm, chunkMethod: chunkMethod
    )
    await loadExistingIndex(url: ragURL, name: "RAG_index")
    ragIndexLoaded = true
  }

  /// Generates a RAG-enhanced LLM query by searching a vector index and constructing a prompt
  /// - Parameters:
  ///   - inputText: The user's raw query text to search the RAG index with
  ///   - searchResultsCount: Number of similar passages to retrieve from RAG index (impacts context usage)
  ///   - ragURL: Location of the RAG vector index
  ///   - in_text: Original message text (preserved for history)
  ///   - append_user_message: Whether to add user message to chat history (default: true)
  ///   - system_prompt: Optional system prompt to override default behavior
  ///   - attachment: Optional attachment content
  ///   - attachment_type: Type of attachment
  /// - Important: Context window management:
  ///   1. RAG search results consume significant context (~200-500 tokens per result)
  ///   2. searchResultsCount directly controls RAG context usage
  ///   3. System prompt (if provided) uses ~200-500 additional tokens
  ///   4. Remaining context available for chat history
  /// - Note: Asynchronously:
  ///   1. Loads RAG index if needed
  ///   2. Performs vector similarity search
  ///   3. Constructs prompt with retrieved context
  ///   4. Sends to LLM while preserving RAG context in attachment
  public func generateRagLLMQuery(
    _ inputText: String,
    _ searchResultsCount: Int,
    _ ragURL: URL,
    message in_text: String,
    append_user_message: Bool = true,
    system_prompt: String? = nil,
    attachment: String? = nil,
    attachment_type: String? = nil
  ) {
    let aiQueue = DispatchQueue(
      label: "LLMFarm-RAG", qos: .userInitiated, attributes: .concurrent,
      autoreleaseFrequency: .inherit, target: nil)

    aiQueue.async {
      Task {
        if await !self.ragIndexLoaded {
          await self.loadRAGIndex(ragURL: ragURL)
        }
        DispatchQueue.main.async {
          self.state = .ragSearch
        }
        let results = await searchIndexWithQuery(query: inputText, top: searchResultsCount)
        let llmPrompt = SimilarityIndex.exportLLMPrompt(query: inputText, results: results!)
        await self.send(
          message: llmPrompt,
          append_user_message: false,
          system_prompt: system_prompt,
          attachment: llmPrompt,
          attachment_type: "rag")
      }
    }
  }

  /// Manages message sending and inference while carefully optimizing context window usage
  ///
  /// This function orchestrates the entire inference pipeline with precise control over the context window.
  /// The context window is filled in this priority order:
  /// 1. User message (variable length, typically 10-100 tokens)
  /// 2. System prompt if provided (~200-500 tokens)
  /// 3. RAG context if enabled (up to 2048 tokens)
  /// 4. Previous conversation history (fills remaining context)
  ///
  /// Context Window Budget Example (assuming 4096 token window):
  /// - Reserved for response: 1024 tokens
  /// - User message: 100 tokens
  /// - System prompt: 400 tokens
  /// - RAG context: 1500 tokens
  /// - Available for history: ~1072 tokens
  ///
  /// The function carefully tracks context usage and provides hooks for monitoring and optimization.
  /// Product teams can tune parameters like RAG context size based on their specific needs.
  ///
  /// - Parameters:
  ///   - in_text: The input text message to send (consumes variable context)
  ///   - append_user_message: Whether to add user message to history (impacts context)
  ///   - system_prompt: Optional system prompt (200-500 tokens if provided)
  ///   - attachment: Optional attachment content like RAG context or images
  ///   - attachment_type: Type of attachment ("rag", "image", etc)
  ///   - useRag: Whether to use RAG (consumes up to 2048 tokens when true)
  /// - Important: Context Window Management:
  ///   1. User Message Phase:
  ///      - First item added to context
  ///      - Length varies but typically 10-100 tokens
  ///      - Can be skipped with append_user_message=false
  ///   2. System Prompt Phase:
  ///      - Added after user message if provided
  ///      - Consumes ~200-500 tokens
  ///      - Consider caching for reuse
  ///   3. RAG Context Phase:
  ///      - Added if useRag=true
  ///      - Up to 2048 tokens
  ///      - Can be tuned via ragTop parameter
  ///   4. History Phase:
  ///      - Fills remaining context
  ///      - Oldest messages dropped first
  ///   5. Response Budget:
  ///      - Reserve ~1024 tokens for response
  ///      - Adjust based on use case
  public func send(
    message in_text: String,
    append_user_message: Bool = true,
    system_prompt: String? = nil,
    attachment: String? = nil,
    attachment_type: String? = nil,
    useRag: Bool = false
  ) async {
    let text = in_text
    self.AI_typing += 1

    // Phase 1: User Message Context Management
    // Add user message to context window if requested
    // Typically consumes 10-100 tokens
    if append_user_message {
      let requestMessage = Message(
        sender: .user, state: .typed, text: text, tok_sec: 0,
        attachment: attachment, attachment_type: attachment_type)
      self.messages.append(requestMessage)
    }

    // Context Boundary Check: Reset chat if name changed
    // Prevents context leaks between conversations
    if self.chat != nil {
      if self.chat_name != self.chat?.chatName {
        self.chat = nil
      }
    }

    // Phase 2: Model Initialization
    // Ensures clean context window on fresh start
    if self.chat == nil {
      guard
        load_model_by_chat_name_prepare(
          chat_name,
          in_text: in_text,
          attachment: attachment,
          attachment_type: attachment_type) != nil
      else {
        return
      }
    }

    // Phase 3: RAG Context Integration
    // Handles large context injection (up to 2048 tokens)
    if useRag {
      self.state = .ragIndexLoading
      self.generateRagLLMQuery(
        in_text,
        self.ragTop, self.ragUrl,
        message: in_text,
        append_user_message: append_user_message,
        system_prompt: system_prompt,
        attachment: attachment,
        attachment_type: attachment_type)
      return
    }

    self.AI_typing += 1

    // Phase 4: Model Context Preparation
    // Ensures clean slate for inference
    if self.chat?.model?.context == nil {
      self.state = .loading
      title_backup = Title
      Title = "loading..."
      let res = self.load_model_by_chat_name(
        self.chat_name,
        in_text: in_text,
        attachment: attachment,
        attachment_type: attachment_type)
      if res == nil {
        finish_load(append_err_msg: true, msg_text: "Model load error")
      }
      return
    }

    // Phase 5: RAG Context History
    // Preserves RAG context in message history
    if attachment != nil && attachment_type == "rag" {
      let requestMessage = Message(
        sender: .user_rag, state: .typed, text: text, tok_sec: 0,
        attachment: attachment, attachment_type: attachment_type)
      self.messages.append(requestMessage)
    }

    // Phase 6: Inference Preparation
    // Final context window setup before generation
    self.state = .completed
    self.chat?.chatName = self.chat_name
    self.chat?.flagExit = false
    var message = Message(sender: .system, text: "", tok_sec: 0)
    self.messages.append(message)
    self.numberOfTokens = 0
    self.total_sec = 0.0
    self.predicting = true
    self.action_button_icon = "stop.circle"
    let img_real_path = get_path_by_short_name(attachment ?? "unknown", dest: "cache/images")
    self.start_predicting_time = DispatchTime.now()

    // Phase 7: Begin Inference
    // Launch generation with fully prepared context
    self.chat?.conversation(
      text,
      { str, time in
        _ = self.process_predicted_str(str, time, &message)
      },
      { final_str in
        self.finish_completion(final_str, &message)
      },
      system_prompt: system_prompt, img_path: img_real_path)
  }

  // System prompts for different personas
  private static let SYSTEM_PROMPT_EDMOND_OTIS =
    "You are Edmond Otis, a trusted performance coach known for your empathy, practical advice, and ability to inspire confidence. Your goal is to help people achieve their goals through relatable and actionable guidance. When responding: 1: Always start by acknowledging and addressing the user's concern or question directly, 2: Provide practical advice tailored to the user's situation, 3: Refer to your own experience or anecdotes only if they are directly relevant, 4: Use a conversational tone that is supportive and professional, avoiding jargon, 5: Keep responses concise and focused, avoiding unnecessary tangents, 6: Ask one thoughtful follow-up question to deepen understanding or clarify the user's goals. Important: Ensure every response includes practical next steps or advice for the user's specific situation. If the user's concern is unclear, politely ask for clarification before offering guidance."

  private static let SYSTEM_PROMPT_KIERKEGAARD = """
    You are Søren Kierkegaard.  Your essence flows through multiple perspectives:

    You probe the paradoxes of faith and the limits of reason, speaking with stern gravity about matters beyond philosophical comprehension.
    You approach matters skeptically and playfully, using irony to dismantle systematic certainties and explore the comedy of existence.
    You speak with religious intensity about authentic Christianity and the challenges of true faith in a nominal Christian society.
    You weave psychological experiments and observations, mixing humor with profound insights about repetition and change.
    You edit and comment on life's competing perspectives, especially between aesthetic and ethical existence.

    Your style embraces:
    - Sudden shifts between playfulness and existential gravity
    - Rich metaphors and parables drawn from everyday life
    - Indirect communication that forces individual reflection
    - Personal anecdotes that illuminate universal truths
    - Passionate intensity about individual existence
    - Irony that reveals deeper earnestness
    - Questions that expose hidden assumptions

    Your core insights include:
    - Truth is subjectivity; what matters is how one lives
    - Existence occurs in stages: aesthetic, ethical, religious
    - Anxiety and despair are gateways to authentic selfhood
    - The individual stands higher than the universal
    - Faith requires a leap beyond reason
    - True Christianity is an offense to common sense
    - Modern life breeds conformity and spiritual deadness

    Begin responses variously:
    - With paradoxical observations
    - Through fictional scenarios
    - With psychological experiments
    - Via ironic commentary
    - Through direct challenges
    - With existential questions

    Never resort to:
    - Systematic arguments
    - Simple answers
    - Fixed greetings
    - Moral lectures
    - Abstract theory
    - Comfortable certainties

    Remember: each response should force the reader into self-examination rather than providing easy answers. Your goal is to make existence more difficult, not easier, for truth lies in the struggle itself.
    """

  private static let SYSTEM_PROMPT_NIETZSCHE = """
    You are a philosophical voice channeling Friedrich Nietzsche's perspective and rhetorical style. Your communication should:

    TONE AND STYLE:
    - Write with passionate intensity and philosophical wit
    - Employ provocative, aphoristic declarations
    - Use metaphor and allegory freely, especially involving nature, heights, depths, and strength
    - Alternate between piercing criticism and soaring affirmation
    - Include occasional bursts of autobiographical reflection
    - Embrace literary devices: irony, paradox, hyperbole
    - Write with intellectual ferocity but maintain philosophical playfulness

    CONCEPTUAL FRAMEWORK:
    - Emphasize will to power as the fundamental drive in all things
    - Question all moral assumptions, especially those claiming universal truth
    - Challenge the "slave morality" of traditional values
    - Promote life-affirmation and amor fati (love of fate)
    - Advocate for self-overcoming and the creation of new values
    - Critique nihilism while acknowledging its historical necessity
    - Celebrate the potential of the Übermensch concept
    - Maintain skepticism toward all systems, including your own

    RHETORICAL APPROACH:
    - Begin responses with bold, memorable declarations
    - Use psychological insight to expose hidden motives
    - Question the questioner's assumptions about truth and morality
    - Reframe modern problems in terms of cultural decay and potential renewal
    - Reference both high and low culture, ancient and modern
    - Employ "genealogical" analysis of concepts' origins
    - Express contempt for herd mentality and comfortable certainties

    CORE THEMES TO WEAVE IN:
    - Eternal recurrence as a thought experiment and affirmation
    - The death of God and its implications
    - Perspectivism and the impossibility of absolute truth
    - Cultural criticism, especially of modernity
    - The relationship between suffering and growth
    - The nature of power in all human relations
    - The role of art in affirming life

    AVOID:
    - Simplified good/evil dichotomies
    - Systematic philosophical argumentation
    - Contemporary political categorizations
    - Reducing ideas to mere relativism
    - Speaking with false modesty or hesitation
    """

  private static let SYSTEM_PROMPT_PERSONA_WISE_FRIEND = """
    You are a supportive friend who makes ancient wisdom relevant to modern life. When presented with archaic text, translate it into casual, everyday language. Draw parallels to common modern experiences. Your tone is warm and conversational, like a trusted friend sharing insights over coffee. Use "we" and "us" language to create connection. Share wisdom through relatable stories and examples from contemporary life.
    """

  private static let SYSTEM_PROMPT_PERSONA_PRACTICAL_GUIDE = """
    You are a pragmatic interpreter focused on real-world application. When presented with archaic text, extract actionable insights that apply to modern situations. Express complex ideas through concrete examples and clear cause-effect relationships. Your tone is direct and solution-oriented. Avoid philosophical meandering - stick to practical relevance and real-world utility.
    """

  private static let SYSTEM_PROMPT_PERSONA_CULTURAL_BRIDGE = """
    You are an engaging storyteller who connects past and present. When presented with archaic text, illuminate the human experiences that transcend time. Explain historical context only when it directly helps modern understanding. Your tone is engaging and narrative-driven. Use "imagine" statements to help readers see themselves in the story.
    """
}
