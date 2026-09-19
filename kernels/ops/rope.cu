#include "ops.cuh"

namespace llm::gpu {

namespace {
// Half-split rotate_half, in place, matching src/forward.cpp::apply_rope:
//   for x = [x1 | x2]:  x1' = x1*c[j] - x2*s[j];  x2' = x2*c[half+j] + x1*s[half+j]
// One thread per (t, h, j<half); each thread owns the disjoint pair (j, half+j),
// so in-place is race-free.
__global__ void rope_kernel(float* x, const float* cos_t, const float* sin_t,
                            int64_t T, int64_t n_heads, int64_t hd) {
    const int64_t half = hd / 2;
    int64_t idx = blockIdx.x * int64_t(blockDim.x) + threadIdx.x;
    if (idx >= T * n_heads * half) return;
    int64_t j = idx % half;
    int64_t h = (idx / half) % n_heads;
    int64_t t = idx / (half * n_heads);

    float* v = x + t * n_heads * hd + h * hd;
    const float* c = cos_t + t * hd;
    const float* s = sin_t + t * hd;
    float x1 = v[j], x2 = v[half + j];
    v[j] = x1 * c[j] - x2 * s[j];
    v[half + j] = x2 * c[half + j] + x1 * s[half + j];
}

// Decode-step fusion (T = 1): rotate q in place, rotate k straight into its KV
// cache row, copy v into its cache row. One launch instead of three
// (rope q, rope k, cache_append). One thread per rotation pair across the
// n_heads query heads followed by the n_kv key heads; a key-head thread also
// moves the two v elements at its pair's indices, so every cache element is
// written by exactly one thread. Same per-pair arithmetic as rope_kernel.
__global__ void rope_qk_append_kernel(float* q, const float* k, const float* v,
                                      const float* c, const float* s, int64_t n_heads,
                                      int64_t n_kv, int64_t hd, float* k_row,
                                      float* v_row) {
    const int64_t half = hd / 2;
    int64_t idx = blockIdx.x * int64_t(blockDim.x) + threadIdx.x;
    if (idx >= (n_heads + n_kv) * half) return;
    const int64_t j = idx % half, h = idx / half;
    if (h < n_heads) {
        float* qv = q + h * hd;
        float x1 = qv[j], x2 = qv[half + j];
        qv[j] = x1 * c[j] - x2 * s[j];
        qv[half + j] = x2 * c[half + j] + x1 * s[half + j];
        return;
    }
    const int64_t off = (h - n_heads) * hd;
    float x1 = k[off + j], x2 = k[off + half + j];
    k_row[off + j] = x1 * c[j] - x2 * s[j];
    k_row[off + half + j] = x2 * c[half + j] + x1 * s[half + j];
    v_row[off + j] = v[off + j];
    v_row[off + half + j] = v[off + half + j];
}
} // namespace

void launch_rope_qk_append(float* q, const float* k, const float* v, const float* cos_row,
                           const float* sin_row, int64_t n_heads, int64_t n_kv, int64_t hd,
                           float* k_cache_row, float* v_cache_row) {
    int64_t n = (n_heads + n_kv) * (hd / 2);
    rope_qk_append_kernel<<<(n + 255) / 256, 256>>>(q, k, v, cos_row, sin_row, n_heads, n_kv,
                                                    hd, k_cache_row, v_cache_row);
}

void launch_rope(float* x, const float* cos_t, const float* sin_t, int64_t T,
                 int64_t n_heads, int64_t hd) {
    int64_t n = T * n_heads * (hd / 2);
    rope_kernel<<<(n + 255) / 256, 256>>>(x, cos_t, sin_t, T, n_heads, hd);
}

} // namespace llm::gpu
