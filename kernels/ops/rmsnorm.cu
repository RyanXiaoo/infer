#include "ops.cuh"

namespace llm::gpu {

namespace {
// One block per token row. Threads accumulate partial sums of squares, reduce
// through shared memory, then all threads scale their strided elements.
// (The __syncthreads() pair here is exactly what racecheck polices.)
__global__ void rmsnorm_kernel(const float* x, const __nv_bfloat16* w, float eps,
                               int64_t H, float* out) {
    extern __shared__ float partial[];
    const float* row = x + blockIdx.x * H;
    float* orow = out + blockIdx.x * H;

    float ss = 0.0f;
    for (int64_t i = threadIdx.x; i < H; i += blockDim.x) ss += row[i] * row[i];
    partial[threadIdx.x] = ss;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) partial[threadIdx.x] += partial[threadIdx.x + stride];
        __syncthreads();
    }

    float scale = rsqrtf(partial[0] / float(H) + eps);
    for (int64_t i = threadIdx.x; i < H; i += blockDim.x)
        orow[i] = row[i] * scale * __bfloat162float(w[i]);
}
} // namespace

void launch_rmsnorm(const float* x, const __nv_bfloat16* w, float eps, int64_t T,
                    int64_t H, float* out) {
    const int threads = 256;
    rmsnorm_kernel<<<T, threads, threads * sizeof(float)>>>(x, w, eps, H, out);
}

} // namespace llm::gpu
