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
constexpr int kChunk = 256;        // must be a multiple of the group (128)
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

template <int BT, QuantKind KIND>
__global__ void gemv_batched_q_kernel(const float* x, QuantView W, const __nv_bfloat16* bias,
                                      int B, int64_t in, float* y, int64_t out) {
    __shared__ float xs[BT][kChunk];
    constexpr int K = Fetch<KIND>::K;
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int64_t o = int64_t(blockIdx.x) * kRowsPerBlock + warp;
    const int64_t row = o < out ? o : 0;

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
            for (int j = lane * K; j < len; j += 32 * K) {
                const int64_t c = c0 + j;
                const float s = bf16_bits(W.scales[row * W.groups + c / kGroup]);
                float w[K];
                if (j + K <= len) {
                    Fetch<KIND>::load(W, row, c, s, w);
                } else {   // ragged tail: element-wise, never past the row
#pragma unroll
                    for (int i = 0; i < K; i++) w[i] = j + i < len ? dequant_at(W, row, c + i) : 0.0f;
                }
#pragma unroll
                for (int i = 0; i < K; i++) {
                    if (j + i < len) {
#pragma unroll
                        for (int b = 0; b < BT; b++) acc[b] += w[i] * xs[b][j + i];
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
#define DISPATCH(KIND)                                    \
    do {                                                  \
        if (B <= 1) LAUNCH(1, KIND);                      \
        else if (B <= 2) LAUNCH(2, KIND);                 \
        else if (B <= 4) LAUNCH(4, KIND);                 \
        else if (B <= 8) LAUNCH(8, KIND);                 \
        else if (B <= 16) LAUNCH(16, KIND);               \
        else LAUNCH(32, KIND);                            \
    } while (0)
    if (W.kind == QuantKind::kInt8) DISPATCH(QuantKind::kInt8);
    else DISPATCH(QuantKind::kInt4);
#undef DISPATCH
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
