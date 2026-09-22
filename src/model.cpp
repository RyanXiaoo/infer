// model.cpp — see model.h. Name resolution + shape verification only.

#include "model.h"

#include <fstream>
#include <set>
#include <stdexcept>

#include "../third_party/nlohmann/json.hpp"

namespace llm {

const Tensor* Model::get(const std::string& name) {
    for (auto& f : files_) if (f->has(name)) return &f->get(name);
    throw std::runtime_error("tensor not found in any shard: " + name);
}

const Tensor* Model::get(const std::string& name, int64_t rows, int64_t cols) {
    const Tensor* t = get(name);
    // cols < 0 means "1-D of length rows" (biases, norm weights).
    bool ok = (cols < 0) ? (t->shape == std::vector<int64_t>{rows})
                         : (t->shape == std::vector<int64_t>{rows, cols});
    if (!ok) {
        throw std::runtime_error("shape mismatch for " + name +
                                 ": config expects [" + std::to_string(rows) +
                                 (cols < 0 ? "" : ", " + std::to_string(cols)) + "]");
    }
    return t;
}

void Model::load(const std::string& model_dir) {
    cfg = ModelConfig::from_file(model_dir + "/config.json");
    // Sharded checkpoint: the index lists which file holds each tensor; open
    // every distinct shard. Otherwise the single model.safetensors.
    files_.clear();
    std::ifstream idx(model_dir + "/model.safetensors.index.json");
    if (idx) {
        const auto j = nlohmann::json::parse(idx);
        std::set<std::string> shards;
        for (auto& [k, v] : j.at("weight_map").items()) shards.insert(v.get<std::string>());
        for (const std::string& s : shards) {
            files_.push_back(std::make_unique<SafetensorsFile>());
            files_.back()->open(model_dir + "/" + s);
        }
    } else {
        files_.push_back(std::make_unique<SafetensorsFile>());
        files_.back()->open(model_dir + "/model.safetensors");
    }

    const int64_t h = cfg.hidden_size;
    const int64_t q_out = cfg.num_attention_heads * cfg.head_dim;
    const int64_t kv_out = cfg.num_key_value_heads * cfg.head_dim;

    embed_tokens = get("model.embed_tokens.weight", cfg.vocab_size, h);
    final_norm = get("model.norm.weight", h, -1);
    // Tied: the embedding table is the LM head. Untied (7B): its own matrix.
    lm_head = cfg.tie_word_embeddings ? nullptr : get("lm_head.weight", cfg.vocab_size, h);

    layers.resize(cfg.num_hidden_layers);
    for (int64_t i = 0; i < cfg.num_hidden_layers; i++) {
        const std::string p = "model.layers." + std::to_string(i) + ".";
        LayerWeights& L = layers[i];
        L.q_w = get(p + "self_attn.q_proj.weight", q_out, h);
        L.k_w = get(p + "self_attn.k_proj.weight", kv_out, h);
        L.v_w = get(p + "self_attn.v_proj.weight", kv_out, h);
        L.o_w = get(p + "self_attn.o_proj.weight", h, q_out);
        if (cfg.attention_bias) {
            L.q_b = get(p + "self_attn.q_proj.bias", q_out, -1);
            L.k_b = get(p + "self_attn.k_proj.bias", kv_out, -1);
            L.v_b = get(p + "self_attn.v_proj.bias", kv_out, -1);
        }
        L.gate_w = get(p + "mlp.gate_proj.weight", cfg.intermediate_size, h);
        L.up_w = get(p + "mlp.up_proj.weight", cfg.intermediate_size, h);
        L.down_w = get(p + "mlp.down_proj.weight", h, cfg.intermediate_size);
        L.input_ln = get(p + "input_layernorm.weight", h, -1);
        L.post_attn_ln = get(p + "post_attention_layernorm.weight", h, -1);
    }
}

} // namespace llm
