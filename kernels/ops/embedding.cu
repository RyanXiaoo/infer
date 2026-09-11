#include "ops.cuh"

namespace llm::gpu {

namespace {
__global__ void embedding_kernel(const __nv_bfloat16* table, const int64_t* ids,
                                 int64_t T, int64_t H, float* out) {
    int64_t idx = blockIdx.x * int64_t(blockDim.x) + threadIdx.x;
    if (idx >= T * H) return;
    int64_t t = idx / H, i = idx % H;
    out[idx] = __bfloat162float(table[ids[t] * H + i]);
}
} // namespace

void launch_embedding(const __nv_bfloat16* table, const int64_t* ids, int64_t T,
                      int64_t H, float* out) {
    int64_t n = T * H;
    embedding_kernel<<<(n + 255) / 256, 256>>>(table, ids, T, H, out);
}

} // namespace llm::gpu
