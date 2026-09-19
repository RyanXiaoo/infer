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

// Decode-step fusion (T = 1): h += delta, then out = rmsnorm(h) * w, in one
// launch instead of two. Same arithmetic in the same order as residual_add
// followed by rmsnorm_kernel (same strided partial sums, same reduction tree),
// so the result is bit-identical. The point is not GPU time (both kernels are
// microseconds): every launch costs the host ~6-15 us, and there are two of
// these pairs per layer.
__global__ void add_rmsnorm_kernel(float* h, const float* delta, const __nv_bfloat16* w,
                                   float eps, int64_t H, float* out) {
    extern __shared__ float partial[];
    float ss = 0.0f;
    for (int64_t i = threadIdx.x; i < H; i += blockDim.x) {
        h[i] += delta[i];
        ss += h[i] * h[i];
    }
    partial[threadIdx.x] = ss;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) partial[threadIdx.x] += partial[threadIdx.x + stride];
        __syncthreads();
    }

    float scale = rsqrtf(partial[0] / float(H) + eps);
    for (int64_t i = threadIdx.x; i < H; i += blockDim.x)
        out[i] = h[i] * scale * __bfloat162float(w[i]);
}
} // namespace

void launch_add_rmsnorm(float* h, const float* delta, const __nv_bfloat16* w, float eps,
                        int64_t H, float* out) {
    const int threads = 256;
    add_rmsnorm_kernel<<<1, threads, threads * sizeof(float)>>>(h, delta, w, eps, H, out);
}

void launch_rmsnorm(const float* x, const __nv_bfloat16* w, float eps, int64_t T,
                    int64_t H, float* out) {
    const int threads = 256;
    rmsnorm_kernel<<<T, threads, threads * sizeof(float)>>>(x, w, eps, H, out);
}

} // namespace llm::gpu
