// model.h — weight wiring for a Llama-family decoder (Qwen2/2.5, Llama 3).
//
// Pure name-resolution, no math: every tensor the forward pass needs is looked
// up ONCE at load time and held as a zero-copy view into the safetensors mmap.
// Resolving by name up front means a missing/renamed tensor fails loudly at
// startup with the tensor's name, not mid-forward as a garbage read.
//
// All dimensions come from ModelConfig (portability rule: nothing hardcoded).
// Tied LM head: there is no lm_head tensor — logits reuse embed_tokens.

#pragma once

#include "loader.h"
#include "model_config.h"

#include <string>
#include <vector>

namespace llm {

struct LayerWeights {
    // Attention. Biases exist only when cfg.attention_bias (Qwen2: QKV yes,
    // o_proj never); absent biases are null — Linear treats null as "no bias".
    const Tensor* q_w = nullptr;
    const Tensor* q_b = nullptr;
    const Tensor* k_w = nullptr;
    const Tensor* k_b = nullptr;
    const Tensor* v_w = nullptr;
    const Tensor* v_b = nullptr;
    const Tensor* o_w = nullptr;
    // MLP (SwiGLU): down(silu(gate(x)) * up(x)), no biases in this family.
    const Tensor* gate_w = nullptr;
    const Tensor* up_w = nullptr;
    const Tensor* down_w = nullptr;
    // Pre-norm weights.
    const Tensor* input_ln = nullptr;
    const Tensor* post_attn_ln = nullptr;
};

class Model {
public:
    // Loads config.json + model.safetensors from a HF model directory and
    // resolves every weight view. Throws std::runtime_error on anything absent
    // or shape-inconsistent with the config.
    void load(const std::string& model_dir);

    ModelConfig cfg;
    const Tensor* embed_tokens = nullptr;   // [vocab, hidden]; also the tied LM head
    const Tensor* final_norm = nullptr;     // model.norm.weight
    std::vector<LayerWeights> layers;

private:
    SafetensorsFile file_;   // owns the mmap; views above live as long as this
    const Tensor* get(const std::string& name);
    const Tensor* get(const std::string& name, int64_t rows, int64_t cols);  // shape-checked
};

} // namespace llm
