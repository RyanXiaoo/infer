// forward.cpp — see forward.h. Plain scalar C++; correctness only.

#include "forward.h"
#include "forward_internal.h"
#include "f16.h"

#include <cassert>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>

namespace llm::detail {

// ---------------------------------------------------------------- primitives
// Shared with the KV-cache session (see forward_internal.h for contracts).

// y = x / sqrt(mean(x^2) + eps) * w, per row. Mean over the hidden dim — no
// mean-subtraction, no bias (that would be LayerNorm, the wrong norm here).
// Double accumulator: free on CPU, removes one source of drift while debugging.
void rmsnorm(const float* x, const Tensor* w, float eps, int64_t T, int64_t H,
             float* out) {
    for (int64_t t = 0; t < T; t++) {
        const float* row = x + t * H;
        double ss = 0.0;
        for (int64_t i = 0; i < H; i++) ss += double(row[i]) * double(row[i]);
        float scale = 1.0f / std::sqrt(float(ss / double(H)) + eps);
        for (int64_t i = 0; i < H; i++)
            out[t * H + i] = row[i] * scale * f16::bf16_to_f32(w->u16()[i]);
    }
}

// y[t,o] = sum_i W[o,i]*x[t,i] (+ b[o]). W is bf16 row-major [out, in] exactly
// as stored in the file (element (o,i) at o*in+i — the Stage 1 spot-check fact).
void linear(const float* x, const Tensor* W, const Tensor* b, int64_t T,
            int64_t in, int64_t out, float* y) {
    for (int64_t t = 0; t < T; t++) {
        const float* xr = x + t * in;
        for (int64_t o = 0; o < out; o++) {
            const uint16_t* wr = W->u16() + o * in;
            float acc = b ? f16::bf16_to_f32(b->u16()[o]) : 0.0f;
            for (int64_t i = 0; i < in; i++)
                acc += f16::bf16_to_f32(wr[i]) * xr[i];
            y[t * out + o] = acc;
        }
    }
}

// RoPE tables, HF layout: inv_freq in fp32 (HF computes it fp32 regardless of
// model dtype), row [pos] = [cos(pos*f_0)..cos(pos*f_{h/2-1}) | same again] —
// the table is duplicated halves, NOT interleaved.
void rope_tables(double theta, int64_t head_dim, int64_t pos0, int64_t T,
                 std::vector<float>& cos_t, std::vector<float>& sin_t) {
    const int64_t half = head_dim / 2;
    std::vector<float> inv_freq(half);
    for (int64_t j = 0; j < half; j++)
        inv_freq[j] = 1.0f / std::pow(float(theta), float(2 * j) / float(head_dim));
    cos_t.resize(T * head_dim);
    sin_t.resize(T * head_dim);
    for (int64_t t = 0; t < T; t++) {
        for (int64_t j = 0; j < half; j++) {
            // Absolute position pos0+t: a decoded token is rotated at its true
            // place in the sequence, not at 0 (the KV-cache correctness trap).
            float a = float(pos0 + t) * inv_freq[j];
            float c = std::cos(a), s = std::sin(a);
            cos_t[t * head_dim + j] = c;
            cos_t[t * head_dim + half + j] = c;
            sin_t[t * head_dim + j] = s;
            sin_t[t * head_dim + half + j] = s;
        }
    }
}

// rotate_half: for x = [x1 | x2] (halves of head_dim),
//   x' = x*cos + [-x2 | x1]*sin
// This is HF/Llama's half-split rotation — NOT NeoX pairwise interleaving,
// which produces plausible text and subtly wrong logits (the classic bug).
// Applied in place, per head, to Q and K only (never V).
void apply_rope(float* x, int64_t T, int64_t n_heads, int64_t head_dim,
                const std::vector<float>& cos_t, const std::vector<float>& sin_t) {
    const int64_t half = head_dim / 2;
    const int64_t stride = n_heads * head_dim;
    for (int64_t t = 0; t < T; t++) {
        const float* c = cos_t.data() + t * head_dim;
        const float* s = sin_t.data() + t * head_dim;
        for (int64_t h = 0; h < n_heads; h++) {
            float* v = x + t * stride + h * head_dim;
            for (int64_t j = 0; j < half; j++) {
                float x1 = v[j], x2 = v[half + j];
                v[j] = x1 * c[j] - x2 * s[j];
                v[half + j] = x2 * c[half + j] + x1 * s[half + j];
            }
        }
    }
}

// GQA causal attention over the full sequence (prefill / recompute path).
// q: [T, n_heads*hd], k/v: [T, n_kv*hd] (all post-RoPE where applicable).
// Q head h reads KV head h/group (group = n_heads/n_kv: 14/2 -> 7 for Qwen-0.5B).
// Output ctx: [T, n_heads*hd] — the concat-of-heads o_proj input.
void attention(const float* q, const float* k, const float* v, int64_t T,
               int64_t n_heads, int64_t n_kv, int64_t hd, float* ctx) {
    const int64_t group = n_heads / n_kv;
    const int64_t qs = n_heads * hd, ks = n_kv * hd;
    const float scale = 1.0f / std::sqrt(float(hd));
    std::vector<float> scores(T);
    for (int64_t h = 0; h < n_heads; h++) {
        const int64_t g = h / group;
        for (int64_t t = 0; t < T; t++) {
            const float* qr = q + t * qs + h * hd;
            // Causal: position t attends to s <= t only. Future scores are never
            // computed at all — equivalent to masking them to -inf pre-softmax.
            float maxs = -std::numeric_limits<float>::infinity();
            for (int64_t s = 0; s <= t; s++) {
                const float* kr = k + s * ks + g * hd;
                float acc = 0.0f;
                for (int64_t d = 0; d < hd; d++) acc += qr[d] * kr[d];
                scores[s] = acc * scale;
                if (scores[s] > maxs) maxs = scores[s];
            }
            // Softmax with max-subtraction (overflow discipline), fp32.
            float denom = 0.0f;
            for (int64_t s = 0; s <= t; s++) {
                scores[s] = std::exp(scores[s] - maxs);
                denom += scores[s];
            }
            float* out = ctx + t * qs + h * hd;
            for (int64_t d = 0; d < hd; d++) out[d] = 0.0f;
            for (int64_t s = 0; s <= t; s++) {
                float p = scores[s] / denom;
                const float* vr = v + s * ks + g * hd;
                for (int64_t d = 0; d < hd; d++) out[d] += p * vr[d];
            }
        }
    }
}

} // namespace llm::detail

namespace llm {

using namespace llm::detail;

namespace {

// --------------------------------------------------------------- tap helpers

void tap(const TapFn* fn, const std::string& name, const std::vector<float>& x,
         std::vector<int64_t> shape) {
    if (fn && *fn) (*fn)(name, x.data(), shape);
}

// Goldens store post-RoPE Q/K in HF's transposed view [n_heads, T, hd]; ours
// lives as [T, n_heads*hd]. Transpose only when a hook is actually installed.
void tap_heads(const TapFn* fn, const std::string& name, const std::vector<float>& x,
               int64_t T, int64_t n_heads, int64_t hd) {
    if (!fn || !*fn) return;
    std::vector<float> tr(n_heads * T * hd);
    for (int64_t h = 0; h < n_heads; h++)
        for (int64_t t = 0; t < T; t++)
            for (int64_t d = 0; d < hd; d++)
                tr[(h * T + t) * hd + d] = x[t * n_heads * hd + h * hd + d];
    (*fn)(name, tr.data(), {n_heads, T, hd});
}

} // namespace

// ---------------------------------------------------------------- forward

std::vector<float> forward(const Model& m, const std::vector<int64_t>& ids,
                           const TapFn* taps, bool last_only) {
    const auto& cfg = m.cfg;
    const int64_t T = int64_t(ids.size());
    const int64_t H = cfg.hidden_size;
    const int64_t hd = cfg.head_dim;
    const int64_t nh = cfg.num_attention_heads, nkv = cfg.num_key_value_heads;
    const int64_t q_out = nh * hd, kv_out = nkv * hd;
    const int64_t I = cfg.intermediate_size;

    // Embedding lookup: row gather from bf16 embed_tokens.
    std::vector<float> h(T * H);
    for (int64_t t = 0; t < T; t++) {
        if (ids[t] < 0 || ids[t] >= cfg.vocab_size)
            throw std::runtime_error("token id out of range: " + std::to_string(ids[t]));
        const uint16_t* row = m.embed_tokens->u16() + ids[t] * H;
        for (int64_t i = 0; i < H; i++) h[t * H + i] = f16::bf16_to_f32(row[i]);
    }
    tap(taps, "hidden_state_0", h, {1, T, H});

    std::vector<float> cos_t, sin_t;
    rope_tables(cfg.rope_theta, hd, /*pos0=*/0, T, cos_t, sin_t);
    tap(taps, "rope_cos", cos_t, {1, T, hd});
    tap(taps, "rope_sin", sin_t, {1, T, hd});

    // Scratch reused across layers.
    std::vector<float> normed(T * H), q(T * q_out), k(T * kv_out), v(T * kv_out);
    std::vector<float> ctx(T * q_out), attn_out(T * H);
    std::vector<float> gate(T * I), up(T * I), mlp_out(T * H);

    for (int64_t li = 0; li < cfg.num_hidden_layers; li++) {
        const LayerWeights& L = m.layers[li];
        const std::string p = "layer" + std::to_string(li) + ".";

        // h = h + attn(input_layernorm(h))   (pre-norm residual stream)
        rmsnorm(h.data(), L.input_ln, float(cfg.rms_norm_eps), T, H, normed.data());
        tap(taps, p + "post_input_layernorm", normed, {1, T, H});

        linear(normed.data(), L.q_w, L.q_b, T, H, q_out, q.data());
        linear(normed.data(), L.k_w, L.k_b, T, H, kv_out, k.data());
        linear(normed.data(), L.v_w, L.v_b, T, H, kv_out, v.data());
        tap(taps, p + "q_proj", q, {1, T, q_out});
        tap(taps, p + "k_proj", k, {1, T, kv_out});
        tap(taps, p + "v_proj", v, {1, T, kv_out});

        apply_rope(q.data(), T, nh, hd, cos_t, sin_t);
        apply_rope(k.data(), T, nkv, hd, cos_t, sin_t);
        tap_heads(taps, p + "q_post_rope", q, T, nh, hd);
        tap_heads(taps, p + "k_post_rope", k, T, nkv, hd);

        attention(q.data(), k.data(), v.data(), T, nh, nkv, hd, ctx.data());
        tap(taps, p + "attn_out_pre_o_proj", ctx, {1, T, q_out});

        linear(ctx.data(), L.o_w, nullptr, T, q_out, H, attn_out.data());
        for (int64_t i = 0; i < T * H; i++) h[i] += attn_out[i];
        tap(taps, p + "post_attn_residual", h, {1, T, H});

        // h = h + mlp(post_attention_layernorm(h))
        rmsnorm(h.data(), L.post_attn_ln, float(cfg.rms_norm_eps), T, H, normed.data());
        tap(taps, p + "post_post_attention_layernorm", normed, {1, T, H});

        linear(normed.data(), L.gate_w, nullptr, T, H, I, gate.data());
        linear(normed.data(), L.up_w, nullptr, T, H, I, up.data());
        tap(taps, p + "mlp_gate", gate, {1, T, I});
        tap(taps, p + "mlp_up", up, {1, T, I});
        for (int64_t i = 0; i < T * I; i++) gate[i] = silu(gate[i]) * up[i];
        linear(gate.data(), L.down_w, nullptr, T, I, H, mlp_out.data());
        tap(taps, p + "mlp_down", mlp_out, {1, T, H});
        for (int64_t i = 0; i < T * H; i++) h[i] += mlp_out[i];

        // HF's hidden_state_{li+1} is this residual boundary — except the last,
        // which HF stores with the final norm already applied (tapped below).
        if (li + 1 < cfg.num_hidden_layers)
            tap(taps, "hidden_state_" + std::to_string(li + 1), h, {1, T, H});
    }

    rmsnorm(h.data(), m.final_norm, float(cfg.rms_norm_eps), T, H, normed.data());
    tap(taps, "hidden_state_" + std::to_string(cfg.num_hidden_layers), normed, {1, T, H});

    // Tied LM head: logits = final_hidden @ embed_tokens^T. embed_tokens is
    // [vocab, hidden] row-major — exactly linear()'s weight layout, no bias.
    const int64_t t0 = last_only ? T - 1 : 0;
    const int64_t rows = T - t0;
    std::vector<float> logits(rows * cfg.vocab_size);
    linear(normed.data() + t0 * H, m.embed_tokens, nullptr, rows, H,
           cfg.vocab_size, logits.data());
    if (!last_only) tap(taps, "logits", logits, {1, T, cfg.vocab_size});
    return logits;
}

std::vector<int64_t> greedy_decode(const Model& m, std::vector<int64_t> ids, int n_new) {
    std::vector<int64_t> out;
    for (int i = 0; i < n_new; i++) {
        std::vector<float> logits = forward(m, ids, nullptr, /*last_only=*/true);
        int64_t best = 0;
        for (int64_t v = 1; v < m.cfg.vocab_size; v++)
            if (logits[v] > logits[best]) best = v;
        out.push_back(best);
        ids.push_back(best);
    }
    return out;
}

} // namespace llm
