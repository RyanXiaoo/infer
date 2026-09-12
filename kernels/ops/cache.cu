#include "ops.cuh"

namespace llm::gpu {

namespace {
__global__ void cache_append_kernel(const float* k_src, const float* v_src,
                                    float* k_cache, float* v_cache, int64_t T,
                                    int64_t kv_dim, int64_t pos) {
    int64_t idx = blockIdx.x * int64_t(blockDim.x) + threadIdx.x;
    if (idx >= T * kv_dim) return;
    int64_t dst = pos * kv_dim + idx;   // rows pos..pos+T-1
    k_cache[dst] = k_src[idx];
    v_cache[dst] = v_src[idx];
}

// One thread per query head. Single query vector over cache_len cached rows;
// max-subtracted softmax then weighted sum, exactly like the CPU decode path.
__global__ void attention_cached_kernel(const float* q, const float* k_cache,
                                        const float* v_cache, int64_t cache_len,
                                        int64_t n_heads, int64_t n_kv, int64_t hd,
                                        float* ctx) {
    int64_t h = blockIdx.x * int64_t(blockDim.x) + threadIdx.x;
    if (h >= n_heads) return;
    const int64_t group = n_heads / n_kv, g = h / group;
    const int64_t kv_dim = n_kv * hd;
    const float scale = rsqrtf(float(hd));
    const float* qr = q + h * hd;

    // Two passes over the cache to avoid a per-thread scratch array: pass 1 for
    // max and denom (online-softmax style), pass 2 for the weighted V sum.
    float maxs = -INFINITY;
    for (int64_t s = 0; s < cache_len; s++) {
        const float* kr = k_cache + s * kv_dim + g * hd;
        float acc = 0.0f;
        for (int64_t d = 0; d < hd; d++) acc += qr[d] * kr[d];
        acc *= scale;
        if (acc > maxs) maxs = acc;
    }
    float denom = 0.0f;
    for (int64_t s = 0; s < cache_len; s++) {
        const float* kr = k_cache + s * kv_dim + g * hd;
        float acc = 0.0f;
        for (int64_t d = 0; d < hd; d++) acc += qr[d] * kr[d];
        denom += expf(acc * scale - maxs);
    }
    float* out = ctx + h * hd;
    for (int64_t d = 0; d < hd; d++) out[d] = 0.0f;
    for (int64_t s = 0; s < cache_len; s++) {
        const float* kr = k_cache + s * kv_dim + g * hd;
        float acc = 0.0f;
        for (int64_t d = 0; d < hd; d++) acc += qr[d] * kr[d];
        float p = expf(acc * scale - maxs) / denom;
        const float* vr = v_cache + s * kv_dim + g * hd;
        for (int64_t d = 0; d < hd; d++) out[d] += p * vr[d];
    }
}
} // namespace

void launch_cache_append(const float* k_src, const float* v_src, float* k_cache,
                         float* v_cache, int64_t T, int64_t kv_dim, int64_t pos) {
    int64_t n = T * kv_dim;
    cache_append_kernel<<<(n + 255) / 256, 256>>>(k_src, v_src, k_cache, v_cache, T,
                                                  kv_dim, pos);
}

void launch_attention_cached(const float* q, const float* k_cache,
                             const float* v_cache, int64_t cache_len,
                             int64_t n_heads, int64_t n_kv, int64_t hd, float* ctx) {
    attention_cached_kernel<<<(n_heads + 63) / 64, 64>>>(q, k_cache, v_cache, cache_len,
                                                         n_heads, n_kv, hd, ctx);
}

} // namespace llm::gpu
