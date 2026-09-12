// session.cpp — see session.h. Reuses the forward-pass primitives directly
// (llm::detail); the only genuinely new code is decode_one's single-query
// attention over the cache.

#include "session.h"
#include "forward_internal.h"
#include "f16.h"

#include <cmath>
#include <limits>
#include <stdexcept>

namespace llm {

using namespace llm::detail;

Session::Session(const Model& m, int64_t max_seq) : m_(m), max_seq_(max_seq) {
    const auto& c = m.cfg;
    const int64_t kv_dim = c.num_key_value_heads * c.head_dim;
    k_cache_.assign(c.num_hidden_layers, std::vector<float>(max_seq * kv_dim));
    v_cache_.assign(c.num_hidden_layers, std::vector<float>(max_seq * kv_dim));
}

std::vector<float> Session::prefill(const std::vector<int64_t>& ids) {
    const auto& cfg = m_.cfg;
    const int64_t T = int64_t(ids.size());
    const int64_t H = cfg.hidden_size, hd = cfg.head_dim;
    const int64_t nh = cfg.num_attention_heads, nkv = cfg.num_key_value_heads;
    const int64_t q_out = nh * hd, kv_out = nkv * hd, I = cfg.intermediate_size;
    if (T == 0) throw std::runtime_error("prefill: empty prompt");
    if (T > max_seq_) throw std::runtime_error("prefill: prompt exceeds max_seq");

    std::vector<float> h(T * H);
    for (int64_t t = 0; t < T; t++) {
        if (ids[t] < 0 || ids[t] >= cfg.vocab_size)
            throw std::runtime_error("token id out of range");
        const uint16_t* row = m_.embed_tokens->u16() + ids[t] * H;
        for (int64_t i = 0; i < H; i++) h[t * H + i] = f16::bf16_to_f32(row[i]);
    }

    std::vector<float> cos_t, sin_t;
    rope_tables(cfg.rope_theta, hd, /*pos0=*/0, T, cos_t, sin_t);

    std::vector<float> normed(T * H), q(T * q_out), k(T * kv_out), v(T * kv_out);
    std::vector<float> ctx(T * q_out), attn_out(T * H);
    std::vector<float> gate(T * I), up(T * I), mlp_out(T * H);

    for (int64_t li = 0; li < cfg.num_hidden_layers; li++) {
        const LayerWeights& L = m_.layers[li];
        rmsnorm(h.data(), L.input_ln, float(cfg.rms_norm_eps), T, H, normed.data());
        linear(normed.data(), L.q_w, L.q_b, T, H, q_out, q.data());
        linear(normed.data(), L.k_w, L.k_b, T, H, kv_out, k.data());
        linear(normed.data(), L.v_w, L.v_b, T, H, kv_out, v.data());
        apply_rope(q.data(), T, nh, hd, cos_t, sin_t);
        apply_rope(k.data(), T, nkv, hd, cos_t, sin_t);

        // Fill the cache with this layer's post-RoPE K and V for positions 0..T-1.
        std::copy(k.begin(), k.end(), k_cache_[li].begin());
        std::copy(v.begin(), v.end(), v_cache_[li].begin());

        attention(q.data(), k.data(), v.data(), T, nh, nkv, hd, ctx.data());
        linear(ctx.data(), L.o_w, nullptr, T, q_out, H, attn_out.data());
        for (int64_t i = 0; i < T * H; i++) h[i] += attn_out[i];

        rmsnorm(h.data(), L.post_attn_ln, float(cfg.rms_norm_eps), T, H, normed.data());
        linear(normed.data(), L.gate_w, nullptr, T, H, I, gate.data());
        linear(normed.data(), L.up_w, nullptr, T, H, I, up.data());
        for (int64_t i = 0; i < T * I; i++) gate[i] = silu(gate[i]) * up[i];
        linear(gate.data(), L.down_w, nullptr, T, I, H, mlp_out.data());
        for (int64_t i = 0; i < T * H; i++) h[i] += mlp_out[i];
    }
    pos_ = T;

    // Final norm + tied LM head for the last position only.
    std::vector<float> normed_last(H);
    rmsnorm(h.data() + (T - 1) * H, m_.final_norm, float(cfg.rms_norm_eps), 1, H,
            normed_last.data());
    std::vector<float> logits(cfg.vocab_size);
    linear(normed_last.data(), m_.embed_tokens, nullptr, 1, H, cfg.vocab_size,
           logits.data());
    return logits;
}

std::vector<float> Session::decode_one(int64_t id) {
    const auto& cfg = m_.cfg;
    const int64_t H = cfg.hidden_size, hd = cfg.head_dim;
    const int64_t nh = cfg.num_attention_heads, nkv = cfg.num_key_value_heads;
    const int64_t q_out = nh * hd, kv_out = nkv * hd, I = cfg.intermediate_size;
    const int64_t kv_dim = kv_out;
    if (pos_ >= max_seq_) throw std::runtime_error("decode_one: cache full (max_seq)");
    if (id < 0 || id >= cfg.vocab_size) throw std::runtime_error("token id out of range");
    const int64_t cache_len = pos_ + 1;   // positions 0..pos_ after appending

    // Embed the single new token.
    h_.assign(H, 0.0f);
    const uint16_t* row = m_.embed_tokens->u16() + id * H;
    for (int64_t i = 0; i < H; i++) h_[i] = f16::bf16_to_f32(row[i]);

    // RoPE tables for exactly one position: pos_ (its true place in the sequence).
    std::vector<float> cos_t, sin_t;
    rope_tables(cfg.rope_theta, hd, /*pos0=*/pos_, 1, cos_t, sin_t);

    normed_.assign(H, 0.0f);
    q_.assign(q_out, 0.0f);
    k_.assign(kv_out, 0.0f);
    v_.assign(kv_out, 0.0f);
    ctx_.assign(q_out, 0.0f);
    delta_.assign(H, 0.0f);
    gate_.assign(I, 0.0f);
    up_.assign(I, 0.0f);

    const int64_t group = nh / nkv;
    const float scale = 1.0f / std::sqrt(float(hd));

    for (int64_t li = 0; li < cfg.num_hidden_layers; li++) {
        const LayerWeights& L = m_.layers[li];
        rmsnorm(h_.data(), L.input_ln, float(cfg.rms_norm_eps), 1, H, normed_.data());
        linear(normed_.data(), L.q_w, L.q_b, 1, H, q_out, q_.data());
        linear(normed_.data(), L.k_w, L.k_b, 1, H, kv_out, k_.data());
        linear(normed_.data(), L.v_w, L.v_b, 1, H, kv_out, v_.data());
        apply_rope(q_.data(), 1, nh, hd, cos_t, sin_t);
        apply_rope(k_.data(), 1, nkv, hd, cos_t, sin_t);

        // Append this token's K/V at row pos_.
        float* kc = k_cache_[li].data() + pos_ * kv_dim;
        float* vc = v_cache_[li].data() + pos_ * kv_dim;
        std::copy(k_.begin(), k_.end(), kc);
        std::copy(v_.begin(), v_.end(), vc);

        // Single-query GQA attention over the whole cache [0..pos_]. Mirror of
        // detail::attention with T fixed to 1 and K/V drawn from the cache.
        std::vector<float> scores(cache_len);
        for (int64_t hh = 0; hh < nh; hh++) {
            const int64_t g = hh / group;
            const float* qr = q_.data() + hh * hd;
            float maxs = -std::numeric_limits<float>::infinity();
            for (int64_t s = 0; s < cache_len; s++) {
                const float* kr = k_cache_[li].data() + s * kv_dim + g * hd;
                float acc = 0.0f;
                for (int64_t d = 0; d < hd; d++) acc += qr[d] * kr[d];
                scores[s] = acc * scale;
                if (scores[s] > maxs) maxs = scores[s];
            }
            float denom = 0.0f;
            for (int64_t s = 0; s < cache_len; s++) {
                scores[s] = std::exp(scores[s] - maxs);
                denom += scores[s];
            }
            float* out = ctx_.data() + hh * hd;
            for (int64_t d = 0; d < hd; d++) out[d] = 0.0f;
            for (int64_t s = 0; s < cache_len; s++) {
                float p = scores[s] / denom;
                const float* vr = v_cache_[li].data() + s * kv_dim + g * hd;
                for (int64_t d = 0; d < hd; d++) out[d] += p * vr[d];
            }
        }

        linear(ctx_.data(), L.o_w, nullptr, 1, q_out, H, delta_.data());
        for (int64_t i = 0; i < H; i++) h_[i] += delta_[i];

        rmsnorm(h_.data(), L.post_attn_ln, float(cfg.rms_norm_eps), 1, H, normed_.data());
        linear(normed_.data(), L.gate_w, nullptr, 1, H, I, gate_.data());
        linear(normed_.data(), L.up_w, nullptr, 1, H, I, up_.data());
        for (int64_t i = 0; i < I; i++) gate_[i] = silu(gate_[i]) * up_[i];
        linear(gate_.data(), L.down_w, nullptr, 1, I, H, delta_.data());
        for (int64_t i = 0; i < H; i++) h_[i] += delta_[i];
    }
    pos_++;

    rmsnorm(h_.data(), m_.final_norm, float(cfg.rms_norm_eps), 1, H, normed_.data());
    std::vector<float> logits(cfg.vocab_size);
    linear(normed_.data(), m_.embed_tokens, nullptr, 1, H, cfg.vocab_size, logits.data());
    return logits;
}

std::vector<int64_t> greedy_decode_cached(const Model& m,
                                          const std::vector<int64_t>& ids, int n_new,
                                          int64_t max_seq) {
    Session s(m, max_seq);
    std::vector<float> logits = s.prefill(ids);
    std::vector<int64_t> out;
    const int64_t V = m.cfg.vocab_size;
    for (int i = 0; i < n_new; i++) {
        int64_t best = 0;
        for (int64_t v = 1; v < V; v++)
            if (logits[v] > logits[best]) best = v;
        out.push_back(best);
        logits = s.decode_one(best);
    }
    return out;
}

} // namespace llm
