#include "model_config.h"

#include <fstream>
#include <stdexcept>

#include "../third_party/nlohmann/json.hpp"

using nlohmann::json;

namespace llm {

ModelConfig ModelConfig::from_file(const std::string& path) {
    std::ifstream f(path);
    if (!f) throw std::runtime_error("cannot open " + path);
    json j = json::parse(f);

    ModelConfig c;
    auto archs = j.at("architectures").get<std::vector<std::string>>();
    if (archs.empty()) throw std::runtime_error("empty architectures in " + path);
    c.architecture = archs[0];

    c.hidden_size = j.at("hidden_size").get<int64_t>();
    c.num_hidden_layers = j.at("num_hidden_layers").get<int64_t>();
    c.num_attention_heads = j.at("num_attention_heads").get<int64_t>();
    c.num_key_value_heads = j.at("num_key_value_heads").get<int64_t>();
    c.intermediate_size = j.at("intermediate_size").get<int64_t>();
    c.vocab_size = j.at("vocab_size").get<int64_t>();
    c.rope_theta = j.at("rope_theta").get<double>();
    c.rms_norm_eps = j.at("rms_norm_eps").get<double>();
    c.tie_word_embeddings = j.value("tie_word_embeddings", false);

    // head_dim is often absent; derive and, when present, assert consistency.
    int64_t derived = c.hidden_size / c.num_attention_heads;
    c.head_dim = j.value("head_dim", derived);
    if (c.head_dim != derived)
        throw std::runtime_error("head_dim inconsistent with hidden_size/num_heads");

    // attention_bias: explicit flag wins; otherwise default by architecture.
    if (j.contains("attention_bias")) {
        c.attention_bias = j["attention_bias"].get<bool>();
    } else if (c.architecture == "Qwen2ForCausalLM") {
        c.attention_bias = true;    // Qwen2's HF config class hardcodes QKV bias
    } else {
        c.attention_bias = false;
    }

    if (c.num_attention_heads % c.num_key_value_heads != 0)
        throw std::runtime_error("num_heads not divisible by num_kv_heads (GQA requires it)");

    return c;
}

} // namespace llm
