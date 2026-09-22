// spec.cu — sampled speculative decoding: accept coins and residual
// resampling on the device (Stage 11).
//
// Leviathan et al. / Chen et al. 2023: with the draft token d_i drawn from
// q_i and the target distribution p_i at the same position, accepting d_i
// with probability min(1, p_i(d_i) / q_i(d_i)) and otherwise emitting a
// sample from norm(max(0, p_i - q_i)) yields a token distributed exactly as
// p_i. Applied left to right until the first rejection; if every draft token
// is accepted the k+1-th position samples from p_k directly.
//
// One block per verify group. Each position needs the log-normalisers of
// p_i and q_i (a max pass and a sum pass over the vocabulary for each), the
// coin, and on rejection one Gumbel-max pass over the residual. The coin and
// the residual noise use seeds derived from the request's seed so they never
// share a hash triple with the draft's own Gumbel noise at the same position.

#include "ops.cuh"

#include <cmath>

namespace llm { namespace gpu {

namespace {

constexpr int kThreads = 1024;
constexpr uint64_t kCoinSeed = 0x5EEDC01Full, kResidualSeed = 0x5EEDBEEFull;

__device__ float block_max(float v, float* red) {
    red[threadIdx.x] = v;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) red[threadIdx.x] = fmaxf(red[threadIdx.x], red[threadIdx.x + s]);
        __syncthreads();
    }
    const float r = red[0];
    __syncthreads();
    return r;
}
__device__ float block_sum(float v, float* red) {
    red[threadIdx.x] = v;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) red[threadIdx.x] += red[threadIdx.x + s];
        __syncthreads();
    }
    const float r = red[0];
    __syncthreads();
    return r;
}
// argmax with lowest index on ties; returns index (or -1 if every value is -inf)
__device__ int64_t block_argmax(float best, int64_t besti, float* red, int64_t* redi) {
    red[threadIdx.x] = best;
    redi[threadIdx.x] = besti;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            const float ov = red[threadIdx.x + s];
            const int64_t oi = redi[threadIdx.x + s];
            if (ov > red[threadIdx.x] || (ov == red[threadIdx.x] && oi < redi[threadIdx.x])) {
                red[threadIdx.x] = ov;
                redi[threadIdx.x] = oi;
            }
        }
        __syncthreads();
    }
    const int64_t r = red[0] == -INFINITY ? -1 : redi[0];
    __syncthreads();
    return r;
}
// log sum_v exp(row[v] / T) over v < V_eff, as (max, log-sum) so callers can
// form log p(v) = row[v] / T - max - logsum.
__device__ void row_lse(const float* row, int64_t V_eff, float T, float* red, float& m, float& ls) {
    float lm = -INFINITY;
    for (int64_t v = threadIdx.x; v < V_eff; v += blockDim.x) lm = fmaxf(lm, row[v] / T);
    m = block_max(lm, red);
    float sum = 0.0f;
    for (int64_t v = threadIdx.x; v < V_eff; v += blockDim.x) sum += expf(row[v] / T - m);
    ls = logf(block_sum(sum, red));
}
// Gumbel-max sample from softmax(row / T)
__device__ int64_t row_sample(const float* row, int64_t V_eff, float T, uint64_t seed, uint64_t step,
                              float* red, int64_t* redi) {
    float best = -INFINITY;
    int64_t besti = 0;
    for (int64_t v = threadIdx.x; v < V_eff; v += blockDim.x) {
        const float x = row[v] / T + gumbel_noise(seed, step, uint64_t(v));
        if (x > best) { best = x; besti = v; }
    }
    return block_argmax(best, besti, red, redi);
}

__global__ void spec_accept_kernel(const float* target, const float* draft, int64_t V, int64_t V_eff, int k,
                                   int maxB, const int* trow, const int* drow, const int64_t* pos0, const float* temp,
                                   const uint64_t* seed, const int64_t* drafts, int* out_a, int64_t* out_tok) {
    __shared__ float red[kThreads];
    __shared__ int64_t redi[kThreads];
    const int g = blockIdx.x;
    const float T = temp[g];
    const uint64_t s = seed[g];
    for (int i = 0; i < k; i++) {
        const float* p = target + (int64_t(trow[g]) + i) * V;
        const float* q = draft + (int64_t(i) * maxB + drow[g]) * V;
        const int64_t d = drafts[int64_t(g) * k + i];
        const uint64_t step = uint64_t(pos0[g] + i);
        float pm, pls, qm, qls;
        row_lse(p, V_eff, T, red, pm, pls);
        row_lse(q, V_eff, T, red, qm, qls);
        const float logp = p[d] / T - pm - pls;
        const float logq = q[d] / T - qm - qls;
        const float u = hash_uniform(s ^ kCoinSeed, step, 0);
        if (logf(u) < logp - logq) continue;   // accept d_i
        // reject: sample from the residual max(0, p - q)
        float best = -INFINITY;
        int64_t besti = 0;
        for (int64_t v = threadIdx.x; v < V_eff; v += blockDim.x) {
            const float r = expf(p[v] / T - pm - pls) - expf(q[v] / T - qm - qls);
            if (r > 0.0f) {
                const float x = logf(r) + gumbel_noise(s ^ kResidualSeed, step, uint64_t(v));
                if (x > best) { best = x; besti = v; }
            }
        }
        int64_t tok = block_argmax(best, besti, red, redi);
        if (tok < 0) tok = row_sample(p, V_eff, T, s, step, red, redi);   // p == q numerically: plain sample
        if (threadIdx.x == 0) { out_a[g] = i; out_tok[g] = tok; }
        return;
    }
    // every draft token accepted: the bonus token is a plain sample from p_k
    const float* pk = target + (int64_t(trow[g]) + k) * V;
    const int64_t tok = row_sample(pk, V_eff, T, s, uint64_t(pos0[g] + k), red, redi);
    if (threadIdx.x == 0) { out_a[g] = k; out_tok[g] = tok; }
}

} // namespace

void launch_spec_accept(const float* target, const float* draft, int64_t V, int64_t V_eff, int k, int maxB,
                        const int* trow, const int* drow, const int64_t* pos0, const float* temp,
                        const uint64_t* seed, const int64_t* drafts, int G, int* out_a, int64_t* out_tok) {
    spec_accept_kernel<<<G, kThreads>>>(target, draft, V, V_eff, k, maxB, trow, drow, pos0, temp, seed, drafts,
                                        out_a, out_tok);
}

}} // namespace llm::gpu
