// gemm.cu — tiled GEMM for prefill (Stage 8): y[T x out] = x[T x in] * W[out x in]^T.
//
// Decode is a matrix-VECTOR product and is bound by reading the weights once
// (Stage 5). Prefill of a T-token prompt is a matrix-MATRIX product: the same
// weights serve T rows, so weight traffic can be amortised over T instead of
// re-read per row. The Stage 6 batched GEMV amortises over at most 32 rows and
// a 1024-token prompt reads every weight 32 times (measured: 84% of GPU time
// on a long-prompt workload).
//
// Classic shared-memory tiling (the Stage 0 warm-up kernel, grown up): a block
// computes a 64 x 64 output tile with 256 threads, each thread a 4 x 4
// register sub-tile; the K dimension (`in`) streams through shared memory in
// slices of 16. Per slice the block loads 64x16 of x and 64x16 of W (bf16 ->
// fp32 on load) once and uses each value 64 times from shared memory. fp32
// accumulation, no tensor cores.

#include "ops.cuh"

namespace llm::gpu {

namespace {
constexpr int TM = 64, TN = 64, TK = 16;   // tile: rows of x, rows of W, slice of in
constexpr int kThreads = 256;              // 16 x 16 threads, 4 x 4 outputs each

__global__ void gemm_tiled_kernel(const float* __restrict__ x, const __nv_bfloat16* __restrict__ W,
                                  const __nv_bfloat16* __restrict__ bias, int64_t T, int64_t in,
                                  int64_t out, float* __restrict__ y) {
    __shared__ float xs[TM][TK + 1];   // +1: no bank conflicts on the column reads below
    __shared__ float ws[TN][TK + 1];
    const int tid = threadIdx.x, tr = tid / 16, tc = tid % 16;
    const int64_t t0 = int64_t(blockIdx.y) * TM, o0 = int64_t(blockIdx.x) * TN;

    float acc[4][4];
#pragma unroll
    for (int i = 0; i < 4; i++)
#pragma unroll
        for (int j = 0; j < 4; j++) acc[i][j] = 0.0f;

    for (int64_t k0 = 0; k0 < in; k0 += TK) {
        // 64 x 16 = 1024 elements per operand, 4 per thread; consecutive threads
        // take consecutive k so each row's loads coalesce.
        for (int e = tid; e < TM * TK; e += kThreads) {
            const int r = e / TK, k = e % TK;
            const int64_t t = t0 + r, i = k0 + k;
            xs[r][k] = (t < T && i < in) ? x[t * in + i] : 0.0f;
            const int64_t o = o0 + r;
            ws[r][k] = (o < out && i < in) ? __bfloat162float(W[o * in + i]) : 0.0f;
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
} // namespace

void launch_gemm_tiled(const float* x, const __nv_bfloat16* W, const __nv_bfloat16* bias,
                       int64_t T, int64_t in, int64_t out, float* y) {
    const dim3 grid(unsigned((out + TN - 1) / TN), unsigned((T + TM - 1) / TM), 1u);
    gemm_tiled_kernel<<<grid, kThreads>>>(x, W, bias, T, in, out, y);
}

} // namespace llm::gpu
