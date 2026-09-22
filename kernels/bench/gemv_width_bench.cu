// gemv_width_bench.cu — batched decode GEMV cost vs batch width (Stage 11).
//
// Speculative decoding verifies k+1 rows in one forward and only pays if that
// forward costs about as much as a one-row step. This bench times the batched
// GEMV kernels (bf16, int8, int4) for B = 1..16 on the 1.5B and 7B shapes and
// prints the time relative to B = 1: a memory-bound kernel would stay near
// 1.0x because the weight bytes are the same at every width.
//
// Usage: gemv_width_bench [achievable_GBps]

#include "../common/bench.cuh"
#include "../common/cuda_check.cuh"
#include "../ops/ops.cuh"

#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

namespace {

struct Shape { const char* model; const char* layer; int64_t out, in; };
const Shape kShapes[] = {
    {"1.5B", "mlp gate/up", 8960, 1536},
    {"1.5B", "mlp down", 1536, 8960},
    {"1.5B", "lm_head", 151936, 1536},
    {"7B", "mlp gate/up", 18944, 3584},
    {"7B", "mlp down", 3584, 18944},
};
const int kWidths[] = {1, 2, 3, 4, 5, 6, 8, 12, 16};
constexpr int64_t kGroup = 128;

std::vector<uint8_t> fill_bytes(size_t n) {
    std::vector<uint8_t> v(n);
    uint32_t s = 0x9E3779B9u;
    for (size_t i = 0; i < n; i++) { s = s * 1664525u + 1013904223u; v[i] = uint8_t(s >> 24); }
    return v;
}
std::vector<uint16_t> fill_bf16_small(size_t n) {   // ~[-0.06, 0.06]
    std::vector<uint16_t> w(n);
    uint32_t s = 0x9E3779B9u;
    for (size_t i = 0; i < n; i++) {
        s = s * 1664525u + 1013904223u;
        w[i] = uint16_t(((s >> 31) << 15) | ((0x78u + (s >> 8) % 3u) << 7) | ((s >> 16) & 0x7Fu));
    }
    return w;
}

} // namespace

int main(int argc, char** argv) {
    const double achievable = argc > 1 ? std::atof(argv[1]) : 784.0;
    std::printf("batched decode GEMV vs batch width; GB/s of weight bytes (100%% = %.0f)\n", achievable);
    std::printf("%-5s %-12s %-5s %3s %9s %8s %6s %7s\n", "model", "layer", "dtype", "B", "ms", "GB/s", "%achv", "x B=1");
    for (const Shape& sh : kShapes) {
        const size_t n = size_t(sh.out) * sh.in;
        const int64_t groups = (sh.in + kGroup - 1) / kGroup;
        std::vector<uint16_t> w_host = fill_bf16_small(n);
        std::vector<uint8_t> q8 = fill_bytes(n), q4 = fill_bytes(n / 2), z4 = fill_bytes(size_t(sh.out) * groups);
        std::vector<uint16_t> sc = fill_bf16_small(size_t(sh.out) * groups);
        std::vector<float> x_host(size_t(32) * sh.in, 0.5f);
        __nv_bfloat16* W = nullptr;
        uint8_t *dq8 = nullptr, *dq4 = nullptr, *dz4 = nullptr;
        uint16_t* dsc = nullptr;
        float *x = nullptr, *y = nullptr;
        CUDA_CHECK(cudaMalloc(&W, n * 2));
        CUDA_CHECK(cudaMalloc(&dq8, n));
        CUDA_CHECK(cudaMalloc(&dq4, n / 2));
        CUDA_CHECK(cudaMalloc(&dz4, z4.size()));
        CUDA_CHECK(cudaMalloc(&dsc, sc.size() * 2));
        CUDA_CHECK(cudaMalloc(&x, x_host.size() * 4));
        CUDA_CHECK(cudaMalloc(&y, size_t(32) * sh.out * 4));
        CUDA_CHECK(cudaMemcpy(W, w_host.data(), n * 2, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dq8, q8.data(), n, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dq4, q4.data(), n / 2, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dz4, z4.data(), z4.size(), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dsc, sc.data(), sc.size() * 2, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(x, x_host.data(), x_host.size() * 4, cudaMemcpyHostToDevice));
        llm::gpu::QuantView v8{llm::gpu::QuantKind::kInt8, sh.out, sh.in, groups, dq8, dsc, nullptr};
        llm::gpu::QuantView v4{llm::gpu::QuantKind::kInt4, sh.out, sh.in, groups, dq4, dsc, dz4};
        struct Dt { const char* name; double bytes; int kind; } dts[] = {
            {"bf16", double(n) * 2, 0}, {"int8", double(n) + double(sc.size()) * 2, 1},
            {"int4", double(n) / 2 + double(sc.size()) * 2 + double(z4.size()), 2}};
        for (const Dt& dt : dts) {
            double base = 0;
            for (int B : kWidths) {
                bench::Result r = bench::run("gemv_width", "", [&] {
                    if (dt.kind == 0) llm::gpu::launch_gemv_batched(x, W, nullptr, B, sh.in, sh.out, y);
                    else if (dt.kind == 1) llm::gpu::launch_gemv_batched_q(x, v8, nullptr, B, sh.in, sh.out, y);
                    else llm::gpu::launch_gemv_batched_q(x, v4, nullptr, B, sh.in, sh.out, y);
                    CUDA_CHECK_LAUNCH();
                }, 5, 30, "mine");
                if (B == 1) base = r.median_ms;
                const double g = dt.bytes / (r.median_ms / 1e3) / 1e9;
                std::printf("%-5s %-12s %-5s %3d %9.4f %8.1f %5.1f%% %6.2fx\n", sh.model, sh.layer, dt.name, B,
                            r.median_ms, g, 100 * g / achievable, r.median_ms / base);
            }
        }
        CUDA_CHECK(cudaFree(W)); CUDA_CHECK(cudaFree(dq8)); CUDA_CHECK(cudaFree(dq4)); CUDA_CHECK(cudaFree(dz4));
        CUDA_CHECK(cudaFree(dsc)); CUDA_CHECK(cudaFree(x)); CUDA_CHECK(cudaFree(y));
    }
    return 0;
}
