#include "ops.cuh"

namespace llm::gpu {

namespace {
// GQA causal attention, one thread per (head, query position) — a direct port
// of src/forward.cpp::attention. Each thread serially computes its score row
// (into its private slice of the global scratch), does the max-subtracted
// softmax, and accumulates the weighted mix of values. Only ~n_heads*T threads
// of parallelism — deliberately naive; the fused rewrite is Stage 8's job.
__global__ void attention_kernel(const float* q, const float* k, const float* v,
                                 float* scores, int64_t T, int64_t n_heads,
                                 int64_t n_kv, int64_t hd, float* ctx) {
    int64_t idx = blockIdx.x * int64_t(blockDim.x) + threadIdx.x;
    if (idx >= n_heads * T) return;
    const int64_t h = idx / T, t = idx % T;
    const int64_t group = n_heads / n_kv, g = h / group;
    const int64_t qs = n_heads * hd, ks = n_kv * hd;
    const float scale = rsqrtf(float(hd));

    const float* qr = q + t * qs + h * hd;
    float* sc = scores + idx * T;   // this thread's private score row

    // Causal: scores for s <= t only (futures never computed = masked to -inf).
    float maxs = -INFINITY;
    for (int64_t s = 0; s <= t; s++) {
        const float* kr = k + s * ks + g * hd;
        float acc = 0.0f;
        for (int64_t d = 0; d < hd; d++) acc += qr[d] * kr[d];
        acc *= scale;
        sc[s] = acc;
        if (acc > maxs) maxs = acc;
    }
    float denom = 0.0f;
    for (int64_t s = 0; s <= t; s++) {
        sc[s] = expf(sc[s] - maxs);
        denom += sc[s];
    }
    float* out = ctx + t * qs + h * hd;
    for (int64_t d = 0; d < hd; d++) out[d] = 0.0f;
    for (int64_t s = 0; s <= t; s++) {
        float p = sc[s] / denom;
        const float* vr = v + s * ks + g * hd;
        for (int64_t d = 0; d < hd; d++) out[d] += p * vr[d];
    }
}
} // namespace

void launch_attention(const float* q, const float* k, const float* v,
                      float* scores_scratch, int64_t T, int64_t n_heads,
                      int64_t n_kv, int64_t hd, float* ctx) {
    int64_t n = n_heads * T;
    attention_kernel<<<(n + 63) / 64, 64>>>(q, k, v, scores_scratch, T, n_heads,
                                            n_kv, hd, ctx);
}

} // namespace llm::gpu
