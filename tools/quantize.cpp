// quantize.cpp — offline weight quantiser (Stage 10).
//
// Reads a bf16 safetensors model through the Stage 1 loader and writes
// <model_dir>/model.q8.llmq or model.q4.llmq: every linear matrix and the
// embedding table quantised group-wise (src/quant.h), norms and biases copied
// as bf16. Reports bytes before/after and the worst per-group relative error.
//
// Usage: quantize <model_dir> int8|int4

#include "../src/f16.h"
#include "../src/model.h"
#include "../src/quant.h"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <string>

int main(int argc, char** argv) {
    if (argc != 3) { std::fprintf(stderr, "usage: %s <model_dir> int8|int4\n", argv[0]); return 2; }
    const std::string dir = argv[1], kind_s = argv[2];
    const llm::QKind kind = kind_s == "int4" ? llm::QKind::kInt4 : llm::QKind::kInt8;
    llm::Model m;
    m.load(dir);
    llm::QuantWriter w;
    size_t bytes_in = 0, bytes_out = 0;
    double worst = 0.0;
    const auto t0 = std::chrono::steady_clock::now();

    auto put_mat = [&](const std::string& name, const llm::Tensor* t) {
        const int64_t rows = t->shape[0], cols = t->shape[1];
        llm::QMatrix q = llm::quantize(t->u16(), rows, cols, kind);
        // Worst absolute error relative to the group's absmax, sampled rows.
        const llm::QTensor v = q.view();
        for (int64_t r = 0; r < rows; r += std::max<int64_t>(1, rows / 64)) {
            for (int64_t g = 0; g < v.groups(); g++) {
                float amax = 0, err = 0;
                for (int64_t c = g * llm::kGroup; c < std::min(cols, (g + 1) * llm::kGroup); c++) {
                    const float x = f16::bf16_to_f32(t->u16()[r * cols + c]);
                    amax = std::max(amax, std::fabs(x));
                    err = std::max(err, std::fabs(x - llm::dequant(v, r, c)));
                }
                if (amax > 0) worst = std::max(worst, double(err / amax));
            }
        }
        bytes_in += size_t(rows) * cols * 2;
        bytes_out += q.q.size() + q.scales.size() * 2 + q.zeros.size();
        w.add(name, q);
    };
    auto put_vec = [&](const std::string& name, const llm::Tensor* t) {
        if (!t) return;
        w.add_bf16(name, t->u16(), t->shape);
        bytes_in += t->nbytes; bytes_out += t->nbytes;
    };

    put_mat("model.embed_tokens.weight", m.embed_tokens);
    if (m.lm_head) put_mat("lm_head.weight", m.lm_head);
    put_vec("model.norm.weight", m.final_norm);
    for (size_t i = 0; i < m.layers.size(); i++) {
        const std::string p = "model.layers." + std::to_string(i) + ".";
        const llm::LayerWeights& L = m.layers[i];
        put_mat(p + "self_attn.q_proj.weight", L.q_w);
        put_mat(p + "self_attn.k_proj.weight", L.k_w);
        put_mat(p + "self_attn.v_proj.weight", L.v_w);
        put_mat(p + "self_attn.o_proj.weight", L.o_w);
        put_mat(p + "mlp.gate_proj.weight", L.gate_w);
        put_mat(p + "mlp.up_proj.weight", L.up_w);
        put_mat(p + "mlp.down_proj.weight", L.down_w);
        put_vec(p + "input_layernorm.weight", L.input_ln);
        put_vec(p + "post_attention_layernorm.weight", L.post_attn_ln);
        put_vec(p + "self_attn.q_proj.bias", L.q_b);
        put_vec(p + "self_attn.k_proj.bias", L.k_b);
        put_vec(p + "self_attn.v_proj.bias", L.v_b);
        std::fprintf(stderr, "\rlayer %zu/%zu", i + 1, m.layers.size());
    }
    const std::string out = llm::QuantModel::file_for(dir, kind);
    w.write(out);
    const double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    std::printf("\nwrote %s: %.2f GB -> %.2f GB (%.2fx), worst group error %.2f%% of absmax, %.1f s\n",
                out.c_str(), bytes_in / 1e9, bytes_out / 1e9, double(bytes_in) / double(bytes_out),
                100 * worst, s);
    return 0;
}
