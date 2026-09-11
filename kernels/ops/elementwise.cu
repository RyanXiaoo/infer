#include "ops.cuh"

namespace llm::gpu {

namespace {
__global__ void swiglu_kernel(float* gate, const float* up, int64_t n) {
    int64_t i = blockIdx.x * int64_t(blockDim.x) + threadIdx.x;
    if (i >= n) return;
    float g = gate[i];
    gate[i] = g / (1.0f + expf(-g)) * up[i];   // silu(g) * up
}

__global__ void residual_add_kernel(float* h, const float* delta, int64_t n) {
    int64_t i = blockIdx.x * int64_t(blockDim.x) + threadIdx.x;
    if (i < n) h[i] += delta[i];
}
} // namespace

void launch_swiglu(float* gate, const float* up, int64_t n) {
    swiglu_kernel<<<(n + 255) / 256, 256>>>(gate, up, n);
}

void launch_residual_add(float* h, const float* delta, int64_t n) {
    residual_add_kernel<<<(n + 255) / 256, 256>>>(h, delta, n);
}

} // namespace llm::gpu
