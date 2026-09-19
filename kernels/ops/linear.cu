#include "ops.cuh"

namespace llm::gpu {

namespace {
// One thread per output element (t, o); serial fp32 dot product over `in`.
// Zero data reuse on purpose — this is the Stage 5 optimization target, and
// its %-of-cuBLAS number is part of this stage's deliverables.
__global__ void linear_kernel(const float* x, const __nv_bfloat16* W,
                              const __nv_bfloat16* b, int64_t T, int64_t in,
                              int64_t out, float* y) {
    int64_t idx = blockIdx.x * int64_t(blockDim.x) + threadIdx.x;
    if (idx >= T * out) return;
    int64_t t = idx / out, o = idx % out;
    const float* xr = x + t * in;
    const __nv_bfloat16* wr = W + o * in;   // row-major [out, in], as stored
    float acc = b ? __bfloat162float(b[o]) : 0.0f;
    for (int64_t i = 0; i < in; i++) acc += __bfloat162float(wr[i]) * xr[i];
    y[idx] = acc;
}

// Stage 5 Phase 2: decode GEMV (T = 1), parallel inside each output row.
//
// Measured on the kernel above: a launch takes ~110 ns * in, whatever `out` is
// (896x896 and 4864x896 both 0.10 ms; 896x4864 0.53 ms). Every thread walks its
// row's `in` elements one after another and all rows run at once, so the launch
// lasts as long as ONE row's loop. The limit is that serial chain, not memory
// bandwidth. Same disease as the Stage 4 attention kernel, same cure:
//
// one BLOCK per output row, B = blockDim.x threads (power of two) split the dot
// product, then a tree reduce in block shared memory (B floats). Chain length
// drops from `in` to in/B + log2(B).
//
// Interleaved = true:  thread j takes elements j, j+B, j+2B, ...  At any step
//   the block's threads are reading B ADJACENT weights (and B adjacent x's).
// Interleaved = false: thread j takes the contiguous chunk [j*len, (j+1)*len).
//   Each thread streams its own region; neighbours are len elements apart.
// Both are kept so the bench can show which access pattern the memory system
// prefers; the launcher uses the measured winner.
template <bool Interleaved>
__global__ void gemv_rowpar_kernel(const float* x, const __nv_bfloat16* W,
                                   const __nv_bfloat16* b, int64_t in, float* y) {
    extern __shared__ float part[];
    const int B = blockDim.x, j = threadIdx.x;
    const int64_t o = blockIdx.x;
    const __nv_bfloat16* wr = W + o * in;

    float acc = 0.0f;
    if (Interleaved) {
        for (int64_t i = j; i < in; i += B) acc += __bfloat162float(wr[i]) * x[i];
    } else {
        const int64_t len = (in + B - 1) / B;
        const int64_t lo = j * len, hi = lo + len < in ? lo + len : in;
        for (int64_t i = lo; i < hi; i++) acc += __bfloat162float(wr[i]) * x[i];
    }
    part[j] = acc;
    __syncthreads();
    for (int stride = B / 2; stride > 0; stride >>= 1) {
        if (j < stride) part[j] += part[j + stride];
        __syncthreads();
    }
    if (j == 0) y[o] = part[0] + (b ? __bfloat162float(b[o]) : 0.0f);
}

__global__ void bf16_to_f32_kernel(const __nv_bfloat16* in, int64_t n, float* out) {
    int64_t i = blockIdx.x * int64_t(blockDim.x) + threadIdx.x;
    if (i < n) out[i] = __bfloat162float(in[i]);
}

__global__ void add_bias_kernel(const __nv_bfloat16* b, int64_t T, int64_t out,
                                float* y) {
    int64_t idx = blockIdx.x * int64_t(blockDim.x) + threadIdx.x;
    if (idx < T * out) y[idx] += __bfloat162float(b[idx % out]);
}
// Threads per row, from the gemv_bench sweep over all ten decode shapes of the
// 0.5B and 1.5B models (32..1024 threads, interleaved and contiguous): 64
// interleaved is best or within noise of best on every shape. Fewer threads
// leave the chain long at in = 4864/8960 (32 threads: 3x slower on 896x4864);
// more threads multiply launch size and reduction steps for no shorter chain
// (1024 threads: 12x slower on the LM head). Contiguous chunks lose 2.4-4x on
// the wide-input shapes, where interleaving lets a block's loads coalesce.
int gemv_auto_threads(int64_t /*in*/, int64_t /*out*/) { return 64; }
} // namespace

void launch_linear_mine(const float* x, const __nv_bfloat16* W,
                        const __nv_bfloat16* b, int64_t T, int64_t in,
                        int64_t out, float* y) {
    int64_t n = T * out;
    linear_kernel<<<(n + 255) / 256, 256>>>(x, W, b, T, in, out, y);
}

void launch_gemv_rowpar(const float* x, const __nv_bfloat16* W, const __nv_bfloat16* b,
                        int64_t T, int64_t in, int64_t out, float* y, int threads,
                        bool interleaved) {
    if (T != 1) {   // prefill GEMM is 0.2% of decode-time profile: stays naive
        launch_linear_mine(x, W, b, T, in, out, y);
        return;
    }
    if (threads <= 0) threads = gemv_auto_threads(in, out);
    if (interleaved)
        gemv_rowpar_kernel<true><<<out, threads, threads * sizeof(float)>>>(x, W, b, in, y);
    else
        gemv_rowpar_kernel<false><<<out, threads, threads * sizeof(float)>>>(x, W, b, in, y);
}

void launch_bf16_to_f32(const __nv_bfloat16* in, int64_t n, float* out) {
    bf16_to_f32_kernel<<<(n + 255) / 256, 256>>>(in, n, out);
}

void launch_add_bias(const __nv_bfloat16* b, int64_t T, int64_t out, float* y) {
    int64_t n = T * out;
    add_bias_kernel<<<(n + 255) / 256, 256>>>(b, T, out, y);
}

} // namespace llm::gpu
