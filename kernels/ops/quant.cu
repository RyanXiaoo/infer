// quant.cu — kernels over group-quantised weights (Stage 10). Layout and
// arithmetic are defined once in src/quant.h; these kernels must reproduce
// dequant() there exactly (int8: s * q; int4: s * (q - z); s a bf16 scale per
// group of 128 columns).
//
// The decode GEMV keeps the Stage 6 v2 structure (8 output rows per block,
// activation chunk in shared memory, compile-time batch width) and only
// changes what a lane loads from the weight row: 4 consecutive int8 bytes or
// 8 consecutive int4 nibbles per step, which keeps a warp's loads contiguous
// (32 lanes x 4 bytes = 128 bytes). Weight bytes per token drop 2x / 4x; the
// multiplies are unchanged, and the dequantisation is a few integer ops per
// weight that hide under the memory traffic.

#include "ops.cuh"

namespace llm::gpu {

namespace {
constexpr int kRowsPerBlock = 8;
constexpr int kGroup = 128;

__device__ inline float bf16_bits(uint16_t b) { return __uint_as_float(uint32_t(b) << 16); }

// Lane-level weight fetch: K consecutive columns starting at c (c % K == 0,
// all inside one group) as fp32.
template <QuantKind KIND>
struct Fetch;
// 4 bytes at a possibly unaligned address: one 32-bit load when aligned (every
// real model: widths are multiples of 128), byte loads otherwise (test shapes).
__device__ inline uint32_t load4(const uint8_t* p) {
    if ((reinterpret_cast<uintptr_t>(p) & 3) == 0) return *reinterpret_cast<const uint32_t*>(p);
    return uint32_t(p[0]) | (uint32_t(p[1]) << 8) | (uint32_t(p[2]) << 16) | (uint32_t(p[3]) << 24);
}
template <>
struct Fetch<QuantKind::kInt8> {
    static constexpr int K = 4;
    __device__ static void load(const QuantView& W, int64_t row, int64_t c, float s, float* w) {
        const uint32_t v = load4(W.q + row * W.cols + c);
#pragma unroll
        for (int i = 0; i < 4; i++) w[i] = s * float(int8_t((v >> (8 * i)) & 0xFF));
    }
};
template <>
struct Fetch<QuantKind::kInt4> {
    static constexpr int K = 8;
    __device__ static void load(const QuantView& W, int64_t row, int64_t c, float s, float* w) {
        const uint32_t v = load4(W.q + (row * W.cols + c) / 2);
        const int z = W.zeros[row * W.groups + c / kGroup];
#pragma unroll
        for (int i = 0; i < 8; i++) w[i] = s * float(int((v >> (4 * i)) & 0xF) - z);
    }
};

// Scalar dequant for the tile loader and the embedding lookup.
__device__ inline float dequant_at(const QuantView& W, int64_t r, int64_t c) {
    const float s = bf16_bits(W.scales[r * W.groups + c / kGroup]);
    if (W.kind == QuantKind::kInt8) return s * float(int8_t(W.q[r * W.cols + c]));
    const uint8_t byte = W.q[(r * W.cols + c) / 2];
    const int v = (c & 1) ? (byte >> 4) : (byte & 0xF);
    return s * float(v - int(W.zeros[r * W.groups + c / kGroup]));
}

// Small batch (BT <= 4): no shared-memory staging at all. Each warp streams
// its own row; activations are read straight from global memory as float4
// (a few KB per row, L1-resident after the first warp touches them). The
// staged version copied the whole activation row into shared memory once per
// block of 8 rows: with int4 weights that copy was as many bytes as the
// weights themselves, and the two barriers per chunk serialised the warps.
template <int BT, QuantKind KIND, bool VEC>
__global__ void gemv_q_direct_kernel(const float* __restrict__ x, QuantView W,
                                     const __nv_bfloat16* __restrict__ bias, int B, int64_t in,
                                     float* __restrict__ y, int64_t out) {
    constexpr int K = Fetch<KIND>::K;
    constexpr int V4 = K / 4;
    constexpr int STEPS = 4;                  // weight words in flight per lane
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int64_t o = int64_t(blockIdx.x) * kRowsPerBlock + warp;
    if (o >= out) return;
    const uint16_t* __restrict__ row_scales = W.scales + o * W.groups;
    const uint8_t* __restrict__ row_q = W.q + (KIND == QuantKind::kInt4 ? o * W.cols / 2 : o * W.cols);
    const uint8_t* __restrict__ row_z = KIND == QuantKind::kInt4 ? W.zeros + o * W.groups : nullptr;

    float acc[BT];
#pragma unroll
    for (int b = 0; b < BT; b++) acc[b] = 0.0f;

    const int64_t full = in / (32 * K) * (32 * K);   // columns covered by whole steps
    for (int64_t c0 = 0; c0 < full; c0 += int64_t(STEPS) * 32 * K) {
        uint32_t wv[STEPS];
        float sc[STEPS];
        int z[STEPS];
#pragma unroll
        for (int st = 0; st < STEPS; st++) {
            const int64_t c = c0 + int64_t(st) * 32 * K + lane * K;
            const bool ok = c < full;
            wv[st] = ok ? load4(row_q + (KIND == QuantKind::kInt4 ? c / 2 : c)) : 0u;
            sc[st] = ok ? bf16_bits(row_scales[c / kGroup]) : 0.0f;
            z[st] = (ok && KIND == QuantKind::kInt4) ? row_z[c / kGroup] : 0;
        }
#pragma unroll
        for (int st = 0; st < STEPS; st++) {
            const int64_t c = c0 + int64_t(st) * 32 * K + lane * K;
            if (c >= full) break;
            float w[K];
            if (KIND == QuantKind::kInt8) {
#pragma unroll
                for (int i = 0; i < 4; i++) w[i] = sc[st] * float(int8_t((wv[st] >> (8 * i)) & 0xFF));
            } else {
#pragma unroll
                for (int i = 0; i < 8; i++) w[i] = sc[st] * float(int((wv[st] >> (4 * i)) & 0xF) - z[st]);
            }
#pragma unroll
            for (int b = 0; b < BT; b++) {
                if (b >= B) break;
                const float* xr = x + int64_t(b) * in + c;
                if (VEC) {   // compile-time: rows are 16-byte aligned
                    const float4* xv = reinterpret_cast<const float4*>(xr);
#pragma unroll
                    for (int v = 0; v < V4; v++) {
                        const float4 f = __ldg(xv + v);
                        acc[b] += w[4 * v] * f.x + w[4 * v + 1] * f.y + w[4 * v + 2] * f.z + w[4 * v + 3] * f.w;
                    }
                } else {   // rows not 16-byte aligned (test shapes): scalar
#pragma unroll
                    for (int i = 0; i < K; i++) acc[b] += w[i] * __ldg(xr + i);
                }
            }
        }
    }
    // Ragged tail (in not a multiple of 32*K): element-wise.
    for (int64_t c = full + lane; c < in; c += 32) {
        const float w = dequant_at(W, o, c);
#pragma unroll
        for (int b = 0; b < BT; b++) if (b < B) acc[b] += w * x[int64_t(b) * in + c];
    }
#pragma unroll
    for (int b = 0; b < BT; b++)
        for (int off = 16; off > 0; off >>= 1)
            acc[b] += __shfl_down_sync(0xffffffffu, acc[b], off);
    if (lane == 0) {
        const float bv = bias ? __bfloat162float(bias[o]) : 0.0f;
        for (int b = 0; b < B; b++) y[int64_t(b) * out + o] = acc[b] + bv;
    }
}

template <int BT, QuantKind KIND>
__global__ void gemv_batched_q_kernel(const float* __restrict__ x, QuantView W,
                                      const __nv_bfloat16* __restrict__ bias, int B, int64_t in,
                                      float* __restrict__ y, int64_t out) {
    // Serves the larger batch widths (int8 above 4 rows, int4 above 16); the
    // direct kernel below serves the rest. 256-column chunks measured best
    // here; wider chunks cost registers and lost at batch 16.
    constexpr int CH = 256;
    constexpr int K = Fetch<KIND>::K;
    constexpr int V4 = K / 4;
    // A lane's activation reads are K consecutive floats, done as float4 loads
    // (conflict-free per quarter-warp; scalar reads at stride K were a K-way
    // bank conflict that made this kernel slower than bf16 at batch 16).
    __shared__ __align__(16) float xs[BT][CH];
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int64_t o = int64_t(blockIdx.x) * kRowsPerBlock + warp;
    const int64_t row = o < out ? o : 0;
    const uint16_t* __restrict__ row_scales = W.scales + row * W.groups;
    const uint8_t* __restrict__ row_q = W.q + (KIND == QuantKind::kInt4 ? row * W.cols / 2 : row * W.cols);

    float acc[BT];
#pragma unroll
    for (int b = 0; b < BT; b++) acc[b] = 0.0f;

    for (int64_t c0 = 0; c0 < in; c0 += CH) {
        const int len = int(in - c0 < CH ? in - c0 : CH);
        for (int t = threadIdx.x; t < BT * CH; t += blockDim.x) {
            const int b = t / CH, j = t % CH;
            xs[b][j] = j < len ? x[int64_t(b) * in + c0 + j] : 0.0f;
        }
        __syncthreads();
        if (o < out) {
            for (int j = lane * K; j < len; j += 32 * K) {
                const int64_t c = c0 + j;
                float w[K];
                if (j + K <= len) {
                    const float sc = bf16_bits(row_scales[c / kGroup]);
                    const uint32_t v = load4(row_q + (KIND == QuantKind::kInt4 ? c / 2 : c));
                    if (KIND == QuantKind::kInt8) {
#pragma unroll
                        for (int i = 0; i < 4; i++) w[i] = sc * float(int8_t((v >> (8 * i)) & 0xFF));
                    } else {
                        const int z = W.zeros[row * W.groups + c / kGroup];
#pragma unroll
                        for (int i = 0; i < 8; i++) w[i] = sc * float(int((v >> (4 * i)) & 0xF) - z);
                    }
                } else {   // ragged tail of the last chunk: element-wise, never past the row
#pragma unroll
                    for (int i = 0; i < K; i++) w[i] = j + i < len ? dequant_at(W, row, c + i) : 0.0f;
                }
#pragma unroll
                for (int b = 0; b < BT; b++) {
                    const float4* xv = reinterpret_cast<const float4*>(&xs[b][j]);
#pragma unroll
                    for (int v4 = 0; v4 < V4; v4++) {
                        const float4 f = xv[v4];
                        acc[b] += w[4 * v4] * f.x + w[4 * v4 + 1] * f.y + w[4 * v4 + 2] * f.z + w[4 * v4 + 3] * f.w;
                    }
                }
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

// Prefill GEMM: same tiling as gemm.cu; the weight slice is dequantised on
// its way into shared memory.
constexpr int TM = 64, TN = 64, TK = 16, kThreads = 256;
__global__ void gemm_tiled_q_kernel(const float* __restrict__ x, QuantView W,
                                    const __nv_bfloat16* __restrict__ bias, int64_t T, int64_t in,
                                    int64_t out, float* __restrict__ y) {
    __shared__ float xs[TM][TK + 1];
    __shared__ float ws[TN][TK + 1];
    const int tid = threadIdx.x, tr = tid / 16, tc = tid % 16;
    const int64_t t0 = int64_t(blockIdx.y) * TM, o0 = int64_t(blockIdx.x) * TN;
    float acc[4][4];
#pragma unroll
    for (int i = 0; i < 4; i++)
#pragma unroll
        for (int j = 0; j < 4; j++) acc[i][j] = 0.0f;
    for (int64_t k0 = 0; k0 < in; k0 += TK) {
        for (int e = tid; e < TM * TK; e += kThreads) {
            const int r = e / TK, k = e % TK;
            const int64_t t = t0 + r, i = k0 + k, o = o0 + r;
            xs[r][k] = (t < T && i < in) ? x[t * in + i] : 0.0f;
            ws[r][k] = (o < out && i < in) ? dequant_at(W, o, i) : 0.0f;
        }
        __syncthreads();
#pragma unroll
        for (int k = 0; k < TK; k++) {
            float xv[4], wv[4];
#pragma unroll
            for (int i = 0; i < 4; i++) xv[i] = xs[tr * 4 + i][k];
#pragma unroll
            for (int j = 0; j < 4; j++) wv[j] = ws[tc * 4 + j][k];
#pragma unroll
            for (int i = 0; i < 4; i++)
#pragma unroll
                for (int j = 0; j < 4; j++) acc[i][j] += xv[i] * wv[j];
        }
        __syncthreads();
    }
#pragma unroll
    for (int i = 0; i < 4; i++) {
        const int64_t t = t0 + tr * 4 + i;
        if (t >= T) continue;
#pragma unroll
        for (int j = 0; j < 4; j++) {
            const int64_t o = o0 + tc * 4 + j;
            if (o >= out) continue;
            y[t * out + o] = acc[i][j] + (bias ? __bfloat162float(bias[o]) : 0.0f);
        }
    }
}

__global__ void embedding_q_kernel(QuantView table, const int64_t* ids, int64_t T, int64_t H,
                                   float* out) {
    int64_t idx = blockIdx.x * int64_t(blockDim.x) + threadIdx.x;
    if (idx >= T * H) return;
    out[idx] = dequant_at(table, ids[idx / H], idx % H);
}
} // namespace

void launch_gemv_batched_q(const float* x, const QuantView& W, const __nv_bfloat16* bias, int B,
                           int64_t in, int64_t out, float* y) {
    const unsigned grid = unsigned((out + kRowsPerBlock - 1) / kRowsPerBlock);
    const int threads = kRowsPerBlock * 32;
#define LAUNCH(BT, KIND) gemv_batched_q_kernel<BT, KIND><<<grid, threads>>>(x, W, bias, B, in, y, out)
#define DIRECT(BT, KIND)                                                                    \
    do {                                                                                    \
        if ((in & 3) == 0) gemv_q_direct_kernel<BT, KIND, true><<<grid, threads>>>(x, W, bias, B, in, y, out);  \
        else gemv_q_direct_kernel<BT, KIND, false><<<grid, threads>>>(x, W, bias, B, in, y, out);               \
    } while (0)
    // Measured on 1.5B: the direct kernel wins for int4 up to batch 16 and for
    // int8 up to batch 4; beyond that the staged kernel (activations shared by
    // 8 rows) is faster.
#define DISPATCH(KIND, DIRECT_MAX)                                          \
    do {                                                                    \
        if (B <= 1) DIRECT(1, KIND);                                        \
        else if (B <= 2) DIRECT(2, KIND);                                   \
        else if (B <= 4) DIRECT(4, KIND);                                   \
        else if (B <= 8) { if (DIRECT_MAX >= 8) DIRECT(8, KIND); else LAUNCH(8, KIND); }      \
        else if (B <= 16) { if (DIRECT_MAX >= 16) DIRECT(16, KIND); else LAUNCH(16, KIND); }  \
        else LAUNCH(32, KIND);                                              \
    } while (0)
    if (W.kind == QuantKind::kInt8) DISPATCH(QuantKind::kInt8, 4);
    else DISPATCH(QuantKind::kInt4, 16);
#undef DISPATCH
#undef DIRECT
#undef LAUNCH
}

void launch_gemm_tiled_q(const float* x, const QuantView& W, const __nv_bfloat16* bias, int64_t T,
                         int64_t in, int64_t out, float* y) {
    const dim3 grid{unsigned((out + TN - 1) / TN), unsigned((T + TM - 1) / TM), 1u};
    gemm_tiled_q_kernel<<<grid, kThreads>>>(x, W, bias, T, in, out, y);
}

void launch_embedding_q(const QuantView& table, const int64_t* ids, int64_t T, int64_t H,
                        float* out) {
    const int64_t n = T * H;
    embedding_q_kernel<<<(n + 255) / 256, 256>>>(table, ids, T, H, out);
}

} // namespace llm::gpu
