//
//  ModelSettingsTemplate.swift
//  LLMFarm
//
//  Created by guinmoon on 17.07.2023.
//

import Foundation

/// Template for configuring chat model inference settings
struct ChatSettingsTemplate: Hashable {
  /// Name of this settings template
  var template_name: String = "Custom"
  /// Inference backend to use (e.g. "llama")
  var inference: String = "llama"
  /// Context window size in tokens
  var context: Int32 = 1024
  /// Batch size for inference
  var n_batch: Int32 = 512
  /// Temperature for sampling (higher = more random)
  var temp: Float = 0.9
  /// Top-k sampling parameter
  var top_k: Int32 = 40
  /// Top-p (nucleus) sampling parameter
  var top_p: Float = 0.95
  /// Number of tokens to look back for repetition penalty
  var repeat_last_n: Int32 = 64
  /// Penalty factor for repeated tokens
  var repeat_penalty: Float = 1.1
  /// Format string for constructing prompts
  var prompt_format: String = "{{prompt}}"
  /// String that triggers the model to stop generating
  var reverse_prompt: String = ""
  /// Whether to use Metal GPU acceleration
  var use_metal: Bool = false
  /// Whether to use Metal for CLIP model
  var use_clip_metal: Bool = false
  /// Mirostat sampling algorithm version (0 = disabled)
  var mirostat: Int32 = 0
  /// Mirostat target entropy
  var mirostat_tau: Float = 5
  /// Mirostat learning rate
  var mirostat_eta: Float = 0.1
  /// Grammar to constrain generation
  var grammar: String = "<None>"
  /// Number of CPU threads to use (0 = auto)
  var numberOfThreads: Int32 = 0
  /// Whether to add beginning-of-sequence token
  var add_bos_token: Bool = true
  /// Whether to add end-of-sequence token
  var add_eos_token: Bool = false
  /// Whether to parse special tokens in the text
  var parse_special_tokens = true
  /// Whether to memory map model files
  var mmap: Bool = true
  /// Whether to lock model in memory
  var mlock: Bool = false
  /// Whether to use Flash Attention optimization
  var flash_attn: Bool = false
  /// Tail free sampling parameter
  var tfs_z: Float = 1
  /// Typical sampling parameter
  var typical_p: Float = 1
  /// Tokens to skip during generation
  var skip_tokens: String = ""

  /// Hashes the template based on its name
  func hash(into hasher: inout Hasher) {
    hasher.combine(template_name)
  }

  /// Compares templates based on name equality
  static func == (lhs: ChatSettingsTemplate, rhs: ChatSettingsTemplate) -> Bool {
    return lhs.template_name == rhs.template_name
  }
}
