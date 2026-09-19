// ops.cuh — launcher declarations for the Stage 3 naive kernels.
//
// One launcher per op, all operating on device pointers: activations fp32,
// weights bf16 exactly as stored in the file (converted to fp32 in registers).
// Every launcher enqueues on the default stream and does NOT synchronize —
// the forward pass is a single in-order stream of launches, so correctness
// needs no syncs until results are copied back.
//
// CUDA-facing only (included from .cu files); the boundary-rule-safe public
// interface lives in kernels/model_gpu.h.

#pragma once

#include <cuda_bf16.h>
#include <cstdint>

namespace llm::gpu {

// out[t,i] = f32(table[ids[t]*H + i])
void launch_embedding(const __nv_bfloat16* table, const int64_t* ids, int64_t T,
                      int64_t H, float* out);

// Per row: y = x / sqrt(mean(x^2)+eps) * f32(w). One block per token.
void launch_rmsnorm(const float* x, const __nv_bfloat16* w, float eps, int64_t T,
                    int64_t H, float* out);

// y[t,o] = sum_i f32(W[o,i]) * x[t,i] (+ f32(b[o])). One thread per output
// element, serial dot product — deliberately naive (Stage 5 optimizes this).
void launch_linear_mine(const float* x, const __nv_bfloat16* W,
                        const __nv_bfloat16* b, int64_t T, int64_t in,
                        int64_t out, float* y);

// Stage 5 decode GEMV: one block per output row, `threads` threads (power of
// two, <= 1024; 0 = auto) split the row's dot product and tree-reduce it.
// `interleaved` picks strided vs contiguous element assignment (see linear.cu).
// T != 1 falls back to launch_linear_mine.
void launch_gemv_rowpar(const float* x, const __nv_bfloat16* W, const __nv_bfloat16* b,
                        int64_t T, int64_t in, int64_t out, float* y, int threads = 0,
                        bool interleaved = true);

// Elementwise bf16 -> fp32 (builds the cuBLAS path's fp32 weight mirrors).
void launch_bf16_to_f32(const __nv_bfloat16* in, int64_t n, float* out);

// y[t,o] += f32(b[o])  (bias epilogue for the cuBLAS path)
void launch_add_bias(const __nv_bfloat16* b, int64_t T, int64_t out, float* y);

// Half-split rotate_half applied in place to [T, n_heads*hd], using the
// duplicated-halves cos/sin tables ([T, hd], fp32, computed host-side).
void launch_rope(float* x, const float* cos_t, const float* sin_t, int64_t T,
                 int64_t n_heads, int64_t hd);

// GQA causal attention, one thread per (head, query position) — each thread
// owns one score row in the global scratch buffer scores[nh*T*T] and mirrors
// the CPU implementation exactly (scale, max-subtracted softmax, weighted mix).
void launch_attention(const float* q, const float* k, const float* v,
                      float* scores_scratch, int64_t T, int64_t n_heads,
                      int64_t n_kv, int64_t hd, float* ctx);

// gate[i] = silu(gate[i]) * up[i]
void launch_swiglu(float* gate, const float* up, int64_t n);

// h[i] += delta[i]
void launch_residual_add(float* h, const float* delta, int64_t n);

// Copy T rows of freshly-computed post-RoPE K and V into the caches at row
// `pos` (prefill: T rows from 0; decode: 1 row at the current length).
void launch_cache_append(const float* k_src, const float* v_src, float* k_cache,
                         float* v_cache, int64_t T, int64_t kv_dim, int64_t pos);

// Single-query GQA attention over the cache: q is [n_heads*hd] (one token),
// k_cache/v_cache hold `cache_len` rows of [n_kv*hd]. One thread per head.
// Naive on purpose (Stage 8 fuses it); mirror of the CPU decode loop.
void launch_attention_cached(const float* q, const float* k_cache,
                             const float* v_cache, int64_t cache_len,
                             int64_t n_heads, int64_t n_kv, int64_t hd, float* ctx);

// Stage 5 variants of launch_attention_cached. Both keep each head's scores in
// `scores` (device scratch, at least n_heads * cache_len floats) instead of
// recomputing them per pass.
//   stored: still one thread per head (isolates the cost of the recomputation).
//   par:    one block per head, `threads` threads splitting the cache positions
//           (power of two, <= 1024; 0 = choose from hd and cache_len).
void launch_attention_cached_stored(const float* q, const float* k_cache,
                                    const float* v_cache, float* scores,
                                    int64_t cache_len, int64_t n_heads, int64_t n_kv,
                                    int64_t hd, float* ctx);
void launch_attention_cached_par(const float* q, const float* k_cache,
                                 const float* v_cache, float* scores, int64_t cache_len,
                                 int64_t n_heads, int64_t n_kv, int64_t hd, float* ctx,
                                 int threads = 0);

} // namespace llm::gpu
