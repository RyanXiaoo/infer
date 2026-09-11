#include "ops.cuh"

namespace llm::gpu {

namespace {
// Half-split rotate_half, in place, matching src/forward.cpp::apply_rope:
//   for x = [x1 | x2]:  x1' = x1*c[j] - x2*s[j];  x2' = x2*c[half+j] + x1*s[half+j]
// One thread per (t, h, j<half); each thread owns the disjoint pair (j, half+j),
// so in-place is race-free.
__global__ void rope_kernel(float* x, const float* cos_t, const float* sin_t,
                            int64_t T, int64_t n_heads, int64_t hd) {
    const int64_t half = hd / 2;
    int64_t idx = blockIdx.x * int64_t(blockDim.x) + threadIdx.x;
    if (idx >= T * n_heads * half) return;
    int64_t j = idx % half;
    int64_t h = (idx / half) % n_heads;
    int64_t t = idx / (half * n_heads);

    float* v = x + t * n_heads * hd + h * hd;
    const float* c = cos_t + t * hd;
    const float* s = sin_t + t * hd;
    float x1 = v[j], x2 = v[half + j];
    v[j] = x1 * c[j] - x2 * s[j];
    v[half + j] = x2 * c[half + j] + x1 * s[half + j];
}
} // namespace

void launch_rope(float* x, const float* cos_t, const float* sin_t, int64_t T,
                 int64_t n_heads, int64_t hd) {
    int64_t n = T * n_heads * (hd / 2);
    rope_kernel<<<(n + 255) / 256, 256>>>(x, cos_t, sin_t, T, n_heads, hd);
}

} // namespace llm::gpu
