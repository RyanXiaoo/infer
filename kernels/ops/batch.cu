// batch.cu — Stage 6 kernels for the batched decode step: B sequences, one
// new token each, in one pass. Every kernel here takes per-row arrays on the
// device (slot, position) because rows belong to different sequences at
// different lengths.
//
// The reason batching pays: decode is bound by reading the weights, and the
// weights are the same for every sequence. gemv_batched reads each weight row
// ONCE and multiplies it with all B input vectors, so the weight traffic per
// token drops by B. Attention does not share (each sequence has its own cache)
// and stays proportional to B.

#include "ops.cuh"

namespace llm::gpu {

namespace {
constexpr int kMaxBatch = 32;

// y[b][o] = sum_i W[o][i] * x[b][i] (+ bias[o]) for b < B.
//
// v1 (kept for the bench as gemv_batched_v1): one block per output row, each
// thread B runtime-indexed accumulators. Measured step time grew almost
// linearly with B: the accumulators spill to local memory, and every weight
// load is paired with B activation loads through L1, so activation traffic
// (out * B * in * 4 bytes) dwarfs the weight traffic (out * in * 2) it was
// meant to amortize.
//
// v2: RW output rows per block (one warp each) share one activation chunk
// staged in shared memory, xs[BT][kChunk]; activation traffic drops by RW.
// BT is a compile-time batch width (B rounded up to 1,2,4,8,16,32) so the
// accumulators live in registers. Each warp walks its row over the chunk with
// interleaved lanes (coalesced weight loads), then reduces each accumulator
// across the warp with shuffles. Rows of x beyond B are read but not written;
// the caller keeps its scratch zero-initialised so they are finite.
constexpr int kChunk = 256;
constexpr int kRowsPerBlock = 8;

template <int BT>
__global__ void gemv_batched_kernel(const float* x, const __nv_bfloat16* W,
                                    const __nv_bfloat16* bias, int B, int64_t in,
                                    float* y, int64_t out) {
    __shared__ float xs[BT][kChunk];
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int64_t o = int64_t(blockIdx.x) * kRowsPerBlock + warp;
    const __nv_bfloat16* wr = W + (o < out ? o : 0) * in;

    float acc[BT];
#pragma unroll
    for (int b = 0; b < BT; b++) acc[b] = 0.0f;

    for (int64_t c0 = 0; c0 < in; c0 += kChunk) {
        const int len = int(in - c0 < kChunk ? in - c0 : kChunk);
        for (int t = threadIdx.x; t < BT * kChunk; t += blockDim.x) {
            const int b = t / kChunk, j = t % kChunk;
            xs[b][j] = j < len ? x[int64_t(b) * in + c0 + j] : 0.0f;
        }
        __syncthreads();
        if (o < out) {
            for (int j = lane; j < len; j += 32) {
                const float w = __bfloat162float(wr[c0 + j]);
#pragma unroll
                for (int b = 0; b < BT; b++) acc[b] += w * xs[b][j];
            }
        }
        __syncthreads();
    }
    if (o >= out) return;
#pragma unroll
    for (int b = 0; b < BT; b++)
        for (int off = 16; off > 0; off >>= 1)
            acc[b] += __shfl_down_sync(0xffffffffu, acc[b], off);
    if (lane == 0) {
        const float bv = bias ? __bfloat162float(bias[o]) : 0.0f;
        for (int b = 0; b < B; b++) y[int64_t(b) * out + o] = acc[b] + bv;
    }
}

// v1, for the before/after bench.
__global__ void gemv_batched_v1_kernel(const float* x, const __nv_bfloat16* W,
                                       const __nv_bfloat16* bias, int B, int64_t in,
                                       float* y, int64_t out) {
    extern __shared__ float part[];   // [B][blockDim.x]
    const int T = blockDim.x, j = threadIdx.x;
    const int64_t o = blockIdx.x;
    const __nv_bfloat16* wr = W + o * in;
    float acc[kMaxBatch];
    for (int b = 0; b < B; b++) acc[b] = 0.0f;
    for (int64_t i = j; i < in; i += T) {
        const float w = __bfloat162float(wr[i]);
        for (int b = 0; b < B; b++) acc[b] += w * x[b * in + i];
    }
    for (int b = 0; b < B; b++) part[b * T + j] = acc[b];
    __syncthreads();
    for (int stride = T / 2; stride > 0; stride >>= 1) {
        if (j < stride)
            for (int b = 0; b < B; b++) part[b * T + j] += part[b * T + j + stride];
        __syncthreads();
    }
    if (j == 0) {
        const float bv = bias ? __bfloat162float(bias[o]) : 0.0f;
        for (int b = 0; b < B; b++) y[b * out + o] = part[b * T] + bv;
    }
}

// Paged KV addressing (Stage 7). The cache is a pool of blocks of kBlockRows
// positions ([n_blocks x kBlockRows x kv_dim] per layer); sequence `slot`'s
// positions live in the blocks listed in its row of the table
// ([n_slots x max_blocks]). Row s of slot: block = table[slot][s / 16],
// pool row = block * 16 + s % 16. A contiguous cache is the special case of an
// identity table, so there is one code path.
constexpr int kBlockRows = 16;

__device__ inline int64_t pool_row(const int* table, int max_blocks, int slot, int64_t s) {
    return int64_t(table[slot * max_blocks + int(s / kBlockRows)]) * kBlockRows + s % kBlockRows;
}

// Per row b: rotate q[b] in place at position pos[b], rotate k[b] into slot
// slot[b]'s cache row pos[b], copy v[b] there. cos/sin tables are [max_seq x hd].
// One thread per (row, rotation pair), q heads then k heads as in the T=1 kernel.
__global__ void rope_qk_append_batched_kernel(float* q, const float* k, const float* v,
                                              const float* cos_tab, const float* sin_tab,
                                              const int* slot, const int* pos, int B,
                                              int64_t n_heads, int64_t n_kv, int64_t hd,
                                              const int* table, int max_blocks,
                                              float* k_cache, float* v_cache) {
    const int64_t half = hd / 2, per_row = (n_heads + n_kv) * half;
    int64_t idx = blockIdx.x * int64_t(blockDim.x) + threadIdx.x;
    if (idx >= B * per_row) return;
    const int b = int(idx / per_row);
    const int64_t r = idx % per_row, j = r % half, h = r / half;
    const float* c = cos_tab + int64_t(pos[b]) * hd;
    const float* s = sin_tab + int64_t(pos[b]) * hd;
    const int64_t kv_dim = n_kv * hd;
    if (h < n_heads) {
        float* qv = q + b * n_heads * hd + h * hd;
        float x1 = qv[j], x2 = qv[half + j];
        qv[j] = x1 * c[j] - x2 * s[j];
        qv[half + j] = x2 * c[half + j] + x1 * s[half + j];
        return;
    }
    const int64_t off = (h - n_heads) * hd;
    const float* kr = k + b * kv_dim;
    const float* vr = v + b * kv_dim;
    const int64_t row = pool_row(table, max_blocks, slot[b], pos[b]) * kv_dim;
    float x1 = kr[off + j], x2 = kr[off + half + j];
    k_cache[row + off + j] = x1 * c[j] - x2 * s[j];
    k_cache[row + off + half + j] = x2 * c[half + j] + x1 * s[half + j];
    v_cache[row + off + j] = vr[off + j];
    v_cache[row + off + half + j] = vr[off + half + j];
}

struct KvView {   // one sequence's K or V rows, addressed through its block-table row
    const float* pool;
    const int* row_table;   // this slot's row of the table
    int64_t kv_dim, hd;
    __device__ inline const float* row(int64_t s, int64_t g) const {
        return pool + (int64_t(row_table[s / kBlockRows]) * kBlockRows + s % kBlockRows) * kv_dim + g * hd;
    }
};

// Same three-phase kernel as attention_cached_par_kernel; block (b, h) reads
// slot[b]'s rows over cache_len = pos[b] + 1. scores: [B x n_heads x scores_stride].
__global__ void attention_cached_batched_kernel(const float* q, const float* k_cache,
                                                const float* v_cache, float* scores,
                                                const int* slot, const int* pos,
                                                int64_t n_heads, int64_t n_kv, int64_t hd,
                                                const int* table, int max_blocks,
                                                int64_t scores_stride, int classes, float* ctx) {
    extern __shared__ float part[];
    const int Bt = blockDim.x, i = threadIdx.x;
    const int b = blockIdx.y;
    const int64_t h = blockIdx.x;
    const int64_t g = h / (n_heads / n_kv), kv_dim = n_kv * hd;
    const int64_t cache_len = pos[b] + 1;
    const KvView K{k_cache, table + slot[b] * max_blocks, kv_dim, hd};
    const KvView V{v_cache, table + slot[b] * max_blocks, kv_dim, hd};
    const float scale = rsqrtf(float(hd));
    const float* qr = q + b * n_heads * hd + h * hd;
    float* sc = scores + (int64_t(b) * n_heads + h) * scores_stride;

    float m = -INFINITY;
    for (int64_t s = i; s < cache_len; s += Bt) {
        const float* kr = K.row(s, g);
        float acc = 0.0f;
        for (int64_t d = 0; d < hd; d++) acc += qr[d] * kr[d];
        sc[s] = acc * scale;
        if (sc[s] > m) m = sc[s];
    }
    part[i] = m;
    __syncthreads();
    for (int stride = Bt / 2; stride > 0; stride >>= 1) {
        if (i < stride && part[i + stride] > part[i]) part[i] = part[i + stride];
        __syncthreads();
    }
    const float maxs = part[0];
    __syncthreads();

    float sum = 0.0f;
    for (int64_t s = i; s < cache_len; s += Bt) sum += (sc[s] = expf(sc[s] - maxs));
    part[i] = sum;
    __syncthreads();
    for (int stride = Bt / 2; stride > 0; stride >>= 1) {
        if (i < stride) part[i] += part[i + stride];
        __syncthreads();
    }
    const float inv_denom = 1.0f / part[0];
    __syncthreads();

    float* out = ctx + b * n_heads * hd + h * hd;
    if (hd > Bt) {
        for (int64_t d = i; d < hd; d += Bt) {
            float acc = 0.0f;
            for (int64_t s = 0; s < cache_len; s++) acc += sc[s] * V.row(s, g)[d];
            out[d] = acc * inv_denom;
        }
        return;
    }
    const int c = i / int(hd), d = i % int(hd);
    float acc = 0.0f;
    if (c < classes)
        for (int64_t s = c; s < cache_len; s += classes) acc += sc[s] * V.row(s, g)[d];
    part[i] = acc;
    __syncthreads();
    for (int stride = classes / 2; stride > 0; stride >>= 1) {
        if (c < stride) part[i] += part[i + stride * int(hd)];
        __syncthreads();
    }
    if (c == 0) out[d] = part[i] * inv_denom;
}

// Copy-on-write: block `src` -> block `dst` in one layer's K and V pools.
__global__ void block_copy_kernel(const float* k_pool, const float* v_pool, int src, int dst,
                                  int64_t kv_dim, float* k_out, float* v_out) {
    const int64_t n = int64_t(kBlockRows) * kv_dim;
    int64_t i = blockIdx.x * int64_t(blockDim.x) + threadIdx.x;
    if (i >= n) return;
    k_out[int64_t(dst) * n + i] = k_pool[int64_t(src) * n + i];
    v_out[int64_t(dst) * n + i] = v_pool[int64_t(src) * n + i];
}

// ---------------------------------------------------------------------------
// Stage 8: flash-decoding. One block per (row, head) cannot hide the latency
// of walking thousands of cached positions (measured: 610 us per launch at
// ~3000 positions, 146 ms per single-sequence step). Split each (row, head)'s
// positions across kSplits blocks: block s handles positions
// [s*chunk, (s+1)*chunk) with its own running max m_s, sum l_s and partial
// output o_s (online-softmax partials in registers and 1 KB of shared memory,
// no scores buffer), then a small combine kernel merges the partials:
//   M = max_s m_s;  L = sum_s l_s * exp(m_s - M);  out = sum_s o_s * exp(m_s - M) / L.
// The grid is fixed per batch width (kSplits from max_seq), which keeps the
// launch capturable into a CUDA graph; blocks past a row's length write empty
// partials (m = -inf, l = 0).
constexpr int kSplitChunk = 256;    // positions per block
constexpr int kSplitThreads = 256;

__global__ void attention_split_kernel(const float* q, const float* k_cache, const float* v_cache,
                                       const int* slot, const int* pos, int64_t n_heads,
                                       int64_t n_kv, int64_t hd, const int* table, int max_blocks,
                                       int splits, float* part_m, float* part_l, float* part_o) {
    __shared__ float sc[kSplitChunk];
    __shared__ float red[kSplitThreads];
    const int i = threadIdx.x, T = blockDim.x;
    const int64_t h = blockIdx.x;
    const int b = blockIdx.y, s_id = blockIdx.z;
    const int64_t g = h / (n_heads / n_kv), kv_dim = n_kv * hd;
    const int64_t cache_len = pos[b] + 1;
    const int64_t chunk = (cache_len + splits - 1) / splits;
    const int64_t s0 = int64_t(s_id) * chunk, s1 = min(cache_len, s0 + chunk);
    const int64_t pidx = (int64_t(b) * n_heads + h) * splits + s_id;
    float* o = part_o + pidx * hd;
    if (s0 >= s1) {   // empty split
        if (i == 0) { part_m[pidx] = -INFINITY; part_l[pidx] = 0.0f; }
        for (int64_t d = i; d < hd; d += T) o[d] = 0.0f;
        return;
    }
    const KvView K{k_cache, table + slot[b] * max_blocks, kv_dim, hd};
    const KvView V{v_cache, table + slot[b] * max_blocks, kv_dim, hd};
    const float scale = rsqrtf(float(hd));
    const float* qr = q + b * n_heads * hd + h * hd;
    const int n = int(s1 - s0);

    // phase 1: scores for this split, block max
    float m = -INFINITY;
    for (int j = i; j < n; j += T) {
        const float* kr = K.row(s0 + j, g);
        float acc = 0.0f;
        for (int64_t d = 0; d < hd; d++) acc += qr[d] * kr[d];
        sc[j] = acc * scale;
        if (sc[j] > m) m = sc[j];
    }
    red[i] = m;
    __syncthreads();
    for (int stride = T / 2; stride > 0; stride >>= 1) {
        if (i < stride && red[i + stride] > red[i]) red[i] = red[i + stride];
        __syncthreads();
    }
    const float ms = red[0];
    __syncthreads();
    // phase 2: exp and sum
    float sum = 0.0f;
    for (int j = i; j < n; j += T) sum += (sc[j] = expf(sc[j] - ms));
    red[i] = sum;
    __syncthreads();
    for (int stride = T / 2; stride > 0; stride >>= 1) {
        if (i < stride) red[i] += red[i + stride];
        __syncthreads();
    }
    if (i == 0) { part_m[pidx] = ms; part_l[pidx] = red[0]; }
    __syncthreads();
    // phase 3: unnormalised weighted V sum. Thread c*hd + d owns dim d and
    // every C-th position of the split; C-way reduce per dim.
    const int classes = T >= hd ? T / int(hd) : 1;
    if (T < hd) {
        for (int64_t d = i; d < hd; d += T) {
            float acc = 0.0f;
            for (int j = 0; j < n; j++) acc += sc[j] * V.row(s0 + j, g)[d];
            o[d] = acc;
        }
        return;
    }
    const int c = i / int(hd), d = i % int(hd);
    float acc = 0.0f;
    if (c < classes) {
#pragma unroll 4
        for (int j = c; j < n; j += classes) acc += sc[j] * V.row(s0 + j, g)[d];
    }
    red[i] = acc;
    __syncthreads();
    for (int stride = classes / 2; stride > 0; stride >>= 1) {
        if (c < stride) red[i] += red[i + stride * int(hd)];
        __syncthreads();
    }
    if (c == 0) o[d] = red[i];
}

// Combine: one block per (row, head); thread per output dim.
__global__ void attention_combine_kernel(const float* part_m, const float* part_l,
                                         const float* part_o, int64_t n_heads, int64_t hd,
                                         int splits, float* ctx) {
    const int64_t h = blockIdx.x;
    const int b = blockIdx.y;
    const int64_t base = (int64_t(b) * n_heads + h) * splits;
    float M = -INFINITY;
    for (int s = 0; s < splits; s++) M = fmaxf(M, part_m[base + s]);
    float L = 0.0f;
    for (int s = 0; s < splits; s++) L += part_l[base + s] * expf(part_m[base + s] - M);
    const float inv = 1.0f / L;
    for (int64_t d = threadIdx.x; d < hd; d += blockDim.x) {
        float acc = 0.0f;
        for (int s = 0; s < splits; s++)
            acc += part_o[(base + s) * hd + d] * expf(part_m[base + s] - M);
        ctx[(int64_t(b) * n_heads + h) * hd + d] = acc * inv;
    }
}

// argmax over each row of logits// argmax over each row of logits [B x V]: one block per row, strided scan +
// block reduce on (value, index); ties -> lowest index, matching the CPU loop.
__global__ void argmax_rows_kernel(const float* logits, int64_t V, int64_t* out) {
    extern __shared__ float sv[];
    int64_t* si = reinterpret_cast<int64_t*>(sv + blockDim.x);
    const float* row = logits + blockIdx.x * V;
    float best = -INFINITY;
    int64_t besti = 0;
    for (int64_t v = threadIdx.x; v < V; v += blockDim.x)
        if (row[v] > best) { best = row[v]; besti = v; }
    sv[threadIdx.x] = best;
    si[threadIdx.x] = besti;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            const float ov = sv[threadIdx.x + stride];
            const int64_t oi = si[threadIdx.x + stride];
            if (ov > sv[threadIdx.x] || (ov == sv[threadIdx.x] && oi < si[threadIdx.x])) {
                sv[threadIdx.x] = ov;
                si[threadIdx.x] = oi;
            }
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) out[blockIdx.x] = si[0];
}

int pow2_floor_b(int64_t v) {
    int p = 1;
    while (int64_t(p) * 2 <= v) p *= 2;
    return p;
}
} // namespace

void launch_gemv_batched(const float* x, const __nv_bfloat16* W, const __nv_bfloat16* bias,
                         int B, int64_t in, int64_t out, float* y) {
    const unsigned grid = unsigned((out + kRowsPerBlock - 1) / kRowsPerBlock);
    const int threads = kRowsPerBlock * 32;
#define LAUNCH(BT) gemv_batched_kernel<BT><<<grid, threads>>>(x, W, bias, B, in, y, out)
    if (B <= 1) LAUNCH(1);
    else if (B <= 2) LAUNCH(2);
    else if (B <= 4) LAUNCH(4);
    else if (B <= 8) LAUNCH(8);
    else if (B <= 16) LAUNCH(16);
    else LAUNCH(32);
#undef LAUNCH
}

void launch_gemv_batched_v1(const float* x, const __nv_bfloat16* W, const __nv_bfloat16* bias,
                            int B, int64_t in, int64_t out, float* y) {
    const int threads = 64;
    gemv_batched_v1_kernel<<<out, threads, size_t(B) * threads * sizeof(float)>>>(
        x, W, bias, B, in, y, out);
}

void launch_rope_qk_append_batched(float* q, const float* k, const float* v,
                                   const float* cos_tab, const float* sin_tab,
                                   const int* slot, const int* pos, int B, int64_t n_heads,
                                   int64_t n_kv, int64_t hd, const int* table, int max_blocks,
                                   float* k_cache, float* v_cache) {
    const int64_t n = int64_t(B) * (n_heads + n_kv) * (hd / 2);
    rope_qk_append_batched_kernel<<<(n + 255) / 256, 256>>>(
        q, k, v, cos_tab, sin_tab, slot, pos, B, n_heads, n_kv, hd, table, max_blocks, k_cache,
        v_cache);
}

void launch_block_copy(const float* k_pool, const float* v_pool, int src, int dst,
                       int64_t kv_dim, float* k_out, float* v_out) {
    const int64_t n = int64_t(kBlockRows) * kv_dim;
    block_copy_kernel<<<(n + 255) / 256, 256>>>(k_pool, v_pool, src, dst, kv_dim, k_out, v_out);
}

void launch_attention_cached_batched(const float* q, const float* k_cache,
                                     const float* v_cache, float* scores, const int* slot,
                                     const int* pos, int B, int64_t max_cache_len,
                                     int64_t n_heads, int64_t n_kv, int64_t hd,
                                     const int* table, int max_blocks, int64_t scores_stride,
                                     float* ctx, int threads) {
    if (threads <= 0) threads = attention_par_threads(hd, max_cache_len);
    const int classes = hd > threads ? 1 : pow2_floor_b(threads / hd);
    const dim3 grid(unsigned(n_heads), unsigned(B), 1u);
    attention_cached_batched_kernel<<<grid, threads, threads * sizeof(float)>>>(
        q, k_cache, v_cache, scores, slot, pos, n_heads, n_kv, hd, table, max_blocks,
        scores_stride, classes, ctx);
}

int attention_splits(int64_t max_seq) {
    int s = int((max_seq + kSplitChunk - 1) / kSplitChunk);
    return s < 1 ? 1 : s > 64 ? 64 : s;
}

void launch_attention_split(const float* q, const float* k_cache, const float* v_cache,
                            const int* slot, const int* pos, int B, int64_t n_heads,
                            int64_t n_kv, int64_t hd, const int* table, int max_blocks,
                            int splits, float* part_m, float* part_l, float* part_o, float* ctx) {
    const dim3 grid{unsigned(n_heads), unsigned(B), unsigned(splits)};
    attention_split_kernel<<<grid, kSplitThreads>>>(q, k_cache, v_cache, slot, pos, n_heads, n_kv,
                                                    hd, table, max_blocks, splits, part_m, part_l,
                                                    part_o);
    const dim3 cgrid{unsigned(n_heads), unsigned(B), 1u};
    attention_combine_kernel<<<cgrid, 128>>>(part_m, part_l, part_o, n_heads, hd, splits, ctx);
}

void launch_argmax_rows(const float* logits, int B, int64_t V, int64_t* out) {
    const int threads = 256;
    argmax_rows_kernel<<<B, threads, threads * (sizeof(float) + sizeof(int64_t))>>>(logits, V, out);
}

} // namespace llm::gpu
