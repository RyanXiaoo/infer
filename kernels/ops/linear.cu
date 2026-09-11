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

__global__ void bf16_to_f32_kernel(const __nv_bfloat16* in, int64_t n, float* out) {
    int64_t i = blockIdx.x * int64_t(blockDim.x) + threadIdx.x;
    if (i < n) out[i] = __bfloat162float(in[i]);
}

__global__ void add_bias_kernel(const __nv_bfloat16* b, int64_t T, int64_t out,
                                float* y) {
    int64_t idx = blockIdx.x * int64_t(blockDim.x) + threadIdx.x;
    if (idx < T * out) y[idx] += __bfloat162float(b[idx % out]);
}
} // namespace

void launch_linear_mine(const float* x, const __nv_bfloat16* W,
                        const __nv_bfloat16* b, int64_t T, int64_t in,
                        int64_t out, float* y) {
    int64_t n = T * out;
    linear_kernel<<<(n + 255) / 256, 256>>>(x, W, b, T, in, out, y);
}

void launch_bf16_to_f32(const __nv_bfloat16* in, int64_t n, float* out) {
    bf16_to_f32_kernel<<<(n + 255) / 256, 256>>>(in, n, out);
}

void launch_add_bias(const __nv_bfloat16* b, int64_t T, int64_t out, float* y) {
    int64_t n = T * out;
    add_bias_kernel<<<(n + 255) / 256, 256>>>(b, T, out, y);
}

} // namespace llm::gpu
