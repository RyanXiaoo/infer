// model_config.h — parsed config.json for Llama-family models (Qwen2/2.5, Llama 3).
//
// Portability rule: everything is derived from config.json — no model-specific
// hardcoding in the engine. The one wrinkle: Qwen2's config.json does NOT carry
// attention_bias (its HF config class hardcodes bias=true for QKV), while
// Llama's does (default false). So: read the flag if present, else default by
// the `architectures` field.

#pragma once

#include <cstdint>
#include <string>

namespace llm {

struct ModelConfig {
    std::string architecture;      // e.g. "Qwen2ForCausalLM", "LlamaForCausalLM"
    int64_t hidden_size = 0;
    int64_t num_hidden_layers = 0;
    int64_t num_attention_heads = 0;
    int64_t num_key_value_heads = 0;
    int64_t head_dim = 0;          // derived hidden/heads when absent from config.json
    int64_t intermediate_size = 0;
    int64_t vocab_size = 0;
    double rope_theta = 0.0;
    double rms_norm_eps = 0.0;
    bool tie_word_embeddings = false;
    bool attention_bias = false;   // Qwen2: true (QKV only, never o_proj); Llama: false

    static ModelConfig from_file(const std::string& config_json_path);
};

} // namespace llm
