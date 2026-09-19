// profile_one.cu — launch ONE kernel variant a few times at one shape, for ncu.
//
// ncu replays each profiled kernel launch several times to collect its counter
// sets, so pointing it at a bench that launches thousands of kernels is
// unusable. This runs exactly one variant at one representative shape:
//
//   profile_one gemv naive|rowpar <out> <in>
//   profile_one attn naive|stored|par <n_heads> <n_kv> <hd> <cache_len>
//
//   ncu -k gemv_rowpar_kernel -s 2 -c 1 --set full -f -o bench/raw/gemv_rowpar_896x4864 \
//       ./build/profile_one gemv rowpar 896 4864
//   ncu -i bench/raw/gemv_rowpar_896x4864.ncu-rep --page details
//
// (tools/ncu_stage5.sh runs the whole Stage 5 before/after set.)

#include "../common/cuda_check.cuh"
#include "../ops/ops.cuh"

#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

namespace {
constexpr int kLaunches = 4;   // ncu: -s 2 skips warm launches, -c 1 profiles the next

template <typename T>
T* device_fill(size_t n, T value) {
    std::vector<T> h(n, value);
    T* d = nullptr;
    CUDA_CHECK(cudaMalloc(&d, n * sizeof(T)));
    CUDA_CHECK(cudaMemcpy(d, h.data(), n * sizeof(T), cudaMemcpyHostToDevice));
    return d;
}
} // namespace

int main(int argc, char** argv) {
    const std::string family = argc > 1 ? argv[1] : "", variant = argc > 2 ? argv[2] : "";
    if (family == "gemv" && argc == 5) {
        const int64_t out = std::atoll(argv[3]), in = std::atoll(argv[4]);
        auto* W = reinterpret_cast<__nv_bfloat16*>(device_fill<uint16_t>(size_t(out) * in, 0x3C80));
        float* x = device_fill<float>(size_t(in), 0.5f);
        float* y = device_fill<float>(size_t(out), 0.0f);
        for (int i = 0; i < kLaunches; i++) {
            if (variant == "naive") llm::gpu::launch_linear_mine(x, W, nullptr, 1, in, out, y);
            else llm::gpu::launch_gemv_rowpar(x, W, nullptr, 1, in, out, y);
            CUDA_CHECK(cudaDeviceSynchronize());
        }
        return 0;
    }
    if (family == "attn" && argc == 7) {
        const int64_t nh = std::atoll(argv[3]), nkv = std::atoll(argv[4]),
                      hd = std::atoll(argv[5]), len = std::atoll(argv[6]);
        float* q = device_fill<float>(size_t(nh) * hd, 0.01f);
        float* k = device_fill<float>(size_t(len) * nkv * hd, 0.02f);
        float* v = device_fill<float>(size_t(len) * nkv * hd, 0.03f);
        float* scores = device_fill<float>(size_t(nh) * len, 0.0f);
        float* ctx = device_fill<float>(size_t(nh) * hd, 0.0f);
        for (int i = 0; i < kLaunches; i++) {
            if (variant == "naive")
                llm::gpu::launch_attention_cached(q, k, v, len, nh, nkv, hd, ctx);
            else if (variant == "stored")
                llm::gpu::launch_attention_cached_stored(q, k, v, scores, len, nh, nkv, hd, ctx);
            else
                llm::gpu::launch_attention_cached_par(q, k, v, scores, len, nh, nkv, hd, ctx);
            CUDA_CHECK(cudaDeviceSynchronize());
        }
        return 0;
    }
    std::fprintf(stderr, "usage: %s gemv naive|rowpar <out> <in>\n"
                         "       %s attn naive|stored|par <n_heads> <n_kv> <hd> <cache_len>\n",
                 argv[0], argv[0]);
    return 2;
}
