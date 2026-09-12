// forward_internal.h — the forward pass's primitive ops, shared between the
// full recompute path (forward.cpp) and the KV-cache session (session.cpp).
//
// Internal header: engine implementation files only. Same contracts as always:
// bf16 weights converted at use-time, fp32 activations and accumulation.

#pragma once

#include "model.h"

#include <cstdint>
#include <vector>

namespace llm::detail {

// y = x / sqrt(mean(x^2) + eps) * w, per row of x [T, H].
void rmsnorm(const float* x, const Tensor* w, float eps, int64_t T, int64_t H,
             float* out);

// y[t,o] = sum_i W[o,i]*x[t,i] (+ b[o]); W bf16 row-major [out, in].
void linear(const float* x, const Tensor* W, const Tensor* b, int64_t T,
            int64_t in, int64_t out, float* y);

// HF-layout RoPE tables (duplicated halves, fp32 inv_freq) for positions
// [pos0, pos0+T): row r of the output corresponds to absolute position pos0+r.
// pos0 is what makes the KV cache correct — a decoded token's tables are built
// at its true position (the cache length), never position 0.
void rope_tables(double theta, int64_t head_dim, int64_t pos0, int64_t T,
                 std::vector<float>& cos_t, std::vector<float>& sin_t);

// Half-split rotate_half applied in place to x [T, n_heads*head_dim], using
// table rows 0..T-1 (which already encode absolute positions via pos0 above).
void apply_rope(float* x, int64_t T, int64_t n_heads, int64_t head_dim,
                const std::vector<float>& cos_t, const std::vector<float>& sin_t);

// Full-sequence GQA causal attention (prefill / recompute path).
void attention(const float* q, const float* k, const float* v, int64_t T,
               int64_t n_heads, int64_t n_kv, int64_t hd, float* ctx);

// silu(z) = z * sigmoid(z)
inline float silu(float z) { return z / (1.0f + std::exp(-z)); }

} // namespace llm::detail
