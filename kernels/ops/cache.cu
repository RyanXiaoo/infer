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

// ---------------------------------------------------------------------------
// Stage 5 Phase 1: the same single-query attention, parallel inside each head.
//
// The naive kernel above gives one thread a whole head: ~3 * cache_len * hd
// global-memory loads executed one after another, so a launch lasts as long as
// that loop (419 us at cache_len ~45) while the rest of the GPU idles. Here a
// head is one BLOCK and its threads split the cache positions, so the longest
// chain any thread walks shrinks by about the thread count.
//
// This is a parallelized naive attention, not the fused flash-style kernel:
// that one is written once, at Stage 8, against the paged KV layout. To keep
// the layout swappable, every K/V address comes from kv_row() below and nothing
// else in the kernels knows how the cache is laid out.

// Row `s` of kv head `g` in a contiguous [max_seq, n_kv*hd] cache.
// Stage 7 replaces this body with a block-table lookup.
__device__ inline const float* kv_row(const float* cache, int64_t s, int64_t g,
                                      int64_t kv_dim, int64_t hd) {
    return cache + s * kv_dim + g * hd;
}

// Rung A: still one thread per head, but each score is computed once and kept
// in `scores` ([n_heads, cache_len]) instead of being recomputed in all three
// passes. Isolates how much of the naive cost was the 3x recomputation.
__global__ void attention_cached_stored_kernel(const float* q, const float* k_cache,
                                               const float* v_cache, float* scores,
                                               int64_t cache_len, int64_t n_heads,
                                               int64_t n_kv, int64_t hd, float* ctx) {
    int64_t h = blockIdx.x * int64_t(blockDim.x) + threadIdx.x;
    if (h >= n_heads) return;
    const int64_t g = h / (n_heads / n_kv), kv_dim = n_kv * hd;
    const float scale = rsqrtf(float(hd));
    const float* qr = q + h * hd;
    float* sc = scores + h * cache_len;

    float maxs = -INFINITY;
    for (int64_t s = 0; s < cache_len; s++) {
        const float* kr = kv_row(k_cache, s, g, kv_dim, hd);
        float acc = 0.0f;
        for (int64_t d = 0; d < hd; d++) acc += qr[d] * kr[d];
        sc[s] = acc * scale;
        if (sc[s] > maxs) maxs = sc[s];
    }
    float denom = 0.0f;
    for (int64_t s = 0; s < cache_len; s++) denom += (sc[s] = expf(sc[s] - maxs));
    float* out = ctx + h * hd;
    for (int64_t d = 0; d < hd; d++) out[d] = 0.0f;
    for (int64_t s = 0; s < cache_len; s++) {
        const float p = sc[s] / denom;
        const float* vr = kv_row(v_cache, s, g, kv_dim, hd);
        for (int64_t d = 0; d < hd; d++) out[d] += p * vr[d];
    }
}

// Rung B: one block per query head, B = blockDim.x threads (a power of two).
// Three phases; every thread of the block must reach every __syncthreads().
//
//   1. scores. Thread i takes positions i, i+B, i+2B, ...: one q.k dot product
//      each, written to `scores`, tracking its own max. Tree-reduce -> head max.
//   2. softmax numerators. Same positions: scores[s] = exp(scores[s] - max), own
//      sum. Tree-reduce -> denominator. (Scores are stored, not streamed: for one
//      query the row is only cache_len floats. Online softmax solves prefill's
//      T x T problem, which is Stage 8's.)
//   3. weighted V sum, split over BOTH axes. out[d] = sum_s p[s] * v[s][d] is hd
//      independent sums of length cache_len. With C = B / hd "classes", thread
//      i = c*hd + d owns output dim d and positions c, c+C, c+2C, ...; then a
//      C-way reduce per dim. Chain length: 2 * cache_len / C loads.
//      If hd > B there are no spare threads: thread i owns dims i, i+B, ... over
//      all positions and writes ctx directly (C = 1, nothing to reduce).
//
// `part` is block shared memory, B floats, reused by all three reductions.
__global__ void attention_cached_par_kernel(const float* q, const float* k_cache,
                                            const float* v_cache, float* scores,
                                            int64_t cache_len, int64_t n_heads,
                                            int64_t n_kv, int64_t hd, int classes,
                                            float* ctx) {
    extern __shared__ float part[];
    const int B = blockDim.x, i = threadIdx.x;
    const int64_t h = blockIdx.x;
    const int64_t g = h / (n_heads / n_kv), kv_dim = n_kv * hd;
    const float scale = rsqrtf(float(hd));
    const float* qr = q + h * hd;
    float* sc = scores + h * cache_len;

    // ---- phase 1: scores + max
    float m = -INFINITY;
    for (int64_t s = i; s < cache_len; s += B) {
        const float* kr = kv_row(k_cache, s, g, kv_dim, hd);
        float acc = 0.0f;
        for (int64_t d = 0; d < hd; d++) acc += qr[d] * kr[d];
        sc[s] = acc * scale;
        if (sc[s] > m) m = sc[s];
    }
    part[i] = m;
    __syncthreads();
    for (int stride = B / 2; stride > 0; stride >>= 1) {
        if (i < stride && part[i + stride] > part[i]) part[i] = part[i + stride];
        __syncthreads();
    }
    const float maxs = part[0];
    __syncthreads();   // everyone has read part[0] before phase 2 overwrites it

    // ---- phase 2: exp + denominator
    float sum = 0.0f;
    for (int64_t s = i; s < cache_len; s += B) sum += (sc[s] = expf(sc[s] - maxs));
    part[i] = sum;
    __syncthreads();
    for (int stride = B / 2; stride > 0; stride >>= 1) {
        if (i < stride) part[i] += part[i + stride];
        __syncthreads();
    }
    const float inv_denom = 1.0f / part[0];
    __syncthreads();

    // ---- phase 3: weighted V sum
    float* out = ctx + h * hd;
    if (hd > B) {
        for (int64_t d = i; d < hd; d += B) {
            float acc = 0.0f;
            for (int64_t s = 0; s < cache_len; s++)
                acc += sc[s] * kv_row(v_cache, s, g, kv_dim, hd)[d];
            out[d] = acc * inv_denom;
        }
        return;
    }
    const int c = i / int(hd), d = i % int(hd);
    float acc = 0.0f;
    if (c < classes)
        for (int64_t s = c; s < cache_len; s += classes)
            acc += sc[s] * kv_row(v_cache, s, g, kv_dim, hd)[d];
    part[i] = acc;
    __syncthreads();
    for (int stride = classes / 2; stride > 0; stride >>= 1) {
        if (c < stride) part[i] += part[i + stride * int(hd)];
        __syncthreads();
    }
    if (c == 0) out[d] = part[i] * inv_denom;
}

int pow2_floor(int64_t v) {
    int p = 1;
    while (int64_t(p) * 2 <= v) p *= 2;
    return p;
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

void launch_attention_cached_stored(const float* q, const float* k_cache,
                                    const float* v_cache, float* scores,
                                    int64_t cache_len, int64_t n_heads, int64_t n_kv,
                                    int64_t hd, float* ctx) {
    attention_cached_stored_kernel<<<(n_heads + 63) / 64, 64>>>(
        q, k_cache, v_cache, scores, cache_len, n_heads, n_kv, hd, ctx);
}

void launch_attention_cached_par(const float* q, const float* k_cache,
                                 const float* v_cache, float* scores, int64_t cache_len,
                                 int64_t n_heads, int64_t n_kv, int64_t hd, float* ctx,
                                 int threads) {
    // Phases 1-2 want ~one position per thread; phase 3 wants threads >= hd so
    // each output dim has its own thread, and more than that to split positions.
    if (threads <= 0) {
        // From attn_bench: 256 vs 1024 threads is a wash up to a few hundred
        // positions; past that, and at hd=128 (where 256 threads leave only two
        // position classes in phase 3), the larger block wins.
        threads = 256;
        while (threads < 1024 && (threads < 8 * hd || threads * 2 < cache_len)) threads *= 2;
    }
    const int classes = hd > threads ? 1 : pow2_floor(threads / hd);
    attention_cached_par_kernel<<<n_heads, threads, threads * sizeof(float)>>>(
        q, k_cache, v_cache, scores, cache_len, n_heads, n_kv, hd, classes, ctx);
}

} // namespace llm::gpu
