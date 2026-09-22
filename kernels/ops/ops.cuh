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

#ifdef __CUDACC__
namespace llm { namespace gpu {
// Counter-based RNG (Stage 9): a hash of (seed, step, index) gives a uniform
// in (0, 1) with no state; the same triple always gives the same number.
// Two splitmix64 finalizer rounds, the second after folding in the index:
// the Stage 9 version XORed the index into the low bits before a single
// multiply, which made the noises within one row a lattice (x0*K + d*K)
// rather than independent draws and biased peaked distributions (found by
// the Stage 11 distribution test, tests/test_ops_gpu.cu spec/accept).
__device__ __forceinline__ uint64_t mix64(uint64_t z) {
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    return z ^ (z >> 31);
}
__device__ __forceinline__ float hash_uniform(uint64_t seed, uint64_t step, uint64_t v) {
    uint64_t x = mix64(seed + 0x9E3779B97F4A7C15ull * (step + 1));
    x = mix64(x ^ (v + 0x632BE59BD9B4E019ull));
    return (float((x >> 40) & 0xFFFFFF) + 0.5f) / 16777216.0f;   // top 24 bits, never 0
}
__device__ __forceinline__ float gumbel_noise(uint64_t seed, uint64_t step, uint64_t v) {
    return -logf(-logf(hash_uniform(seed, step, v)));
}
}} // namespace llm::gpu
#endif

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

// Stage 5 Phase 3 decode-step fusions (T = 1). Bit-identical to the launches
// they replace; they exist to cut host launch count, not GPU time.
//   add_rmsnorm:     h += delta; out = rmsnorm(h) * w        (was 2 launches)
//   rope_qk_append:  rope(q) in place; rope(k) -> k_cache_row; v -> v_cache_row
//                    (was 3 launches). cos_row/sin_row: this position's [hd] rows.
void launch_add_rmsnorm(float* h, const float* delta, const __nv_bfloat16* w, float eps,
                        int64_t H, float* out);
void launch_rope_qk_append(float* q, const float* k, const float* v, const float* cos_row,
                           const float* sin_row, int64_t n_heads, int64_t n_kv, int64_t hd,
                           float* k_cache_row, float* v_cache_row);

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
int attention_par_threads(int64_t hd, int64_t len);
void launch_attention_cached_stored(const float* q, const float* k_cache,
                                    const float* v_cache, float* scores,
                                    int64_t cache_len, int64_t n_heads, int64_t n_kv,
                                    int64_t hd, float* ctx);
void launch_attention_cached_par(const float* q, const float* k_cache,
                                 const float* v_cache, float* scores, int64_t cache_len,
                                 int64_t n_heads, int64_t n_kv, int64_t hd, float* ctx,
                                 int threads = 0);

// Stage 6/7 batched decode step (kernels/ops/batch.cu). Rows b < B carry
// per-row device arrays slot[b] and pos[b]. The KV cache is PAGED: per layer a
// pool [n_blocks x 16 x kv_dim] for K and for V, addressed through a block
// table [n_slots x max_blocks] (row s of slot = table[slot][s/16]*16 + s%16).
// A contiguous cache is an identity table. B <= 32.
constexpr int kKvBlockRows = 16;
// x is [kMaxRows x in] scratch (rows >= B are read, so keep them finite).
void launch_gemv_batched(const float* x, const __nv_bfloat16* W, const __nv_bfloat16* bias,
                         int B, int64_t in, int64_t out, float* y);
void launch_gemv_batched_v1(const float* x, const __nv_bfloat16* W, const __nv_bfloat16* bias,
                            int B, int64_t in, int64_t out, float* y);
void launch_rope_qk_append_batched(float* q, const float* k, const float* v,
                                   const float* cos_tab, const float* sin_tab,
                                   const int* slot, const int* pos, int B, int64_t n_heads,
                                   int64_t n_kv, int64_t hd, const int* table, int max_blocks,
                                   float* k_cache, float* v_cache);
void launch_attention_cached_batched(const float* q, const float* k_cache,
                                     const float* v_cache, float* scores, const int* slot,
                                     const int* pos, int B, int64_t max_cache_len,
                                     int64_t n_heads, int64_t n_kv, int64_t hd,
                                     const int* table, int max_blocks, int64_t scores_stride,
                                     float* ctx, int threads = 0);
// Stage 8 flash-decoding: positions of each (row, head) split across `splits`
// blocks (= attention_splits(max_seq), fixed per session) with online-softmax
// partials part_m/part_l [B x n_heads x splits] and part_o [.. x hd], merged
// by a combine kernel. Same inputs as launch_attention_cached_batched.
int attention_splits(int64_t max_seq);
void launch_attention_split(const float* q, const float* k_cache, const float* v_cache,
                            const int* slot, const int* pos, int B, int64_t n_heads,
                            int64_t n_kv, int64_t hd, const int* table, int max_blocks,
                            int splits, float* part_m, float* part_l, float* part_o, float* ctx);
// Copy-on-write: block src -> dst in one layer's K and V pools.
void launch_block_copy(const float* k_pool, const float* v_pool, int src, int dst,
                       int64_t kv_dim, float* k_out, float* v_out);
// Stage 10 quantised weights (kernels/ops/quant.cu). Layout per src/quant.h:
// groups of 128 columns share a bf16 scale (and an int4 zero point).
enum class QuantKind { kInt8, kInt4 };
struct QuantView {
    QuantKind kind;
    int64_t rows, cols, groups;
    const uint8_t* q;         // int8 bytes, or packed int4 nibbles (low first)
    const uint16_t* scales;   // bf16 bits, rows * groups
    const uint8_t* zeros;     // int4: rows * groups; int8: nullptr
};
void launch_gemv_batched_q(const float* x, const QuantView& W, const __nv_bfloat16* bias, int B,
                           int64_t in, int64_t out, float* y);
void launch_gemm_tiled_q(const float* x, const QuantView& W, const __nv_bfloat16* bias, int64_t T,
                         int64_t in, int64_t out, float* y);
void launch_embedding_q(const QuantView& table, const int64_t* ids, int64_t T, int64_t H,
                        float* out);

// Stage 8 prefill GEMM (kernels/ops/gemm.cu): y[T x out] = x[T x in] * W^T (+ bias),
// 64x64 shared-memory tiles, fp32 accumulate. Any T, in, out.
void launch_gemm_tiled(const float* x, const __nv_bfloat16* W, const __nv_bfloat16* bias,
                       int64_t T, int64_t in, int64_t out, float* y);

// out[b] = sample from softmax(logits[b] / temperature[b]) via Gumbel-max with a
// counter-based RNG keyed by (seed[b], step[b]); temperature 0 = exact argmax.
// Speculative accept/resample (Stage 11), one block per verify group g:
//   target logits rows trow[g] + i (stride V) are p_i (after candidate i);
//   draft logits for step i, draft row drow[g], live at draft + (i*maxB + drow[g])*V.
// For i in 0..k-1 accept drafts[g*k+i] with probability min(1, p_i(d)/q_i(d))
// at temperature temp[g]; on the first rejection sample from the residual
// max(0, p_i - q_i) normalised; if all k accepted sample from p_k. Writes the
// accepted count out_a[g] (0..k) and the emitted token out_tok[g].
void launch_spec_accept(const float* target, const float* draft, int64_t V, int64_t V_eff, int k, int maxB,
                        const int* trow, const int* drow, const int64_t* pos0, const float* temp,
                        const uint64_t* seed, const int64_t* drafts, int G, int* out_a, int64_t* out_tok);
// Rows have stride V; only ids < V_eff are candidates.
void launch_sample_rows(const float* logits, int B, int64_t V, int64_t V_eff, const float* temperature,
                        const uint64_t* seed, const int64_t* step, int64_t* out);
// out[b] = -log softmax(logits[b])[target[b]]
void launch_nll_rows(const float* logits, int B, int64_t V, const int64_t* target, float* out);
// out[b] = argmax over logits[b*V .. b*V+V) (lowest index on ties).
void launch_argmax_rows(const float* logits, int B, int64_t V, int64_t* out);

} // namespace llm::gpu
