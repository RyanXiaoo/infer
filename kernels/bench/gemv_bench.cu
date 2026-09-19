// gemv_bench.cu — per-shape decode GEMV microbench (Stage 5).
//
// Decode runs every linear layer at T=1, so each one is a matrix-vector product
// whose cost is reading the weight matrix once: the op is bound by memory
// bandwidth, not arithmetic. The honest unit is therefore effective GB/s
// (weight bytes read / kernel time), compared with what this card's memory can
// actually deliver, and only then with cuBLAS.
//
// Two things to keep straight when reading the table:
//   * my kernel reads bf16 weights (2 bytes each); the cuBLAS path reads the
//     fp32 mirrors (4 bytes each). At equal GB/s mine finishes in half the time,
//     so "% of cuBLAS time" and "% of cuBLAS bandwidth" differ by 2x.
//   * every distinct decode shape of the 0.5B dev model and the 1.5B measurement
//     model is here. Per token each layer shape runs n_layers times (q/o twice,
//     k/v twice, gate/up twice), the LM head once.
//
// Small shapes finish in tens of microseconds, below what one CUDA-event pair
// resolves under WSL2/WDDM (work is submitted in batches, so a lone short kernel
// mostly measures submission latency). Those shapes are timed as kBurst
// back-to-back launches per rep; the record's dims carry the "_x20" suffix and
// its median_ms is for the whole burst. The table prints per-launch time.
// Caveat: a burst rereads one matrix, so a matrix smaller than L2 (~50 MB) is
// served from cache after the first launch and can show more GB/s than DRAM
// delivers. In the model every matrix is read once per token out of a >1 GB
// working set. Read small-shape rows as launch-floor numbers, not bandwidth.
//
// Weights are random (timing does not depend on the values), so no model files
// are needed. One JSON record per (implementation, shape) lands in bench/.
//
// Usage: gemv_bench [achievable_GBps]
//   achievable_GBps: measured memory bandwidth used as the 100% mark. Defaults
//   to the Stage 0 vector_add measurement until the STREAM bench replaces it.

#include "../common/bench.cuh"
#include "../common/cuda_check.cuh"
#include "../ops/ops.cuh"

#include <cublas_v2.h>

#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

namespace {

constexpr double kStage0AchievableGBps = 747.0;   // bench/vector_add_n16M_20260819
constexpr int kBurst = 20;                         // launches per timed rep, small shapes
constexpr size_t kBurstBelowBytes = 64u << 20;     // bf16 weight bytes

struct Shape { const char* model; const char* layer; int64_t out, in; };
const Shape kShapes[] = {
    {"0.5B", "attn q/o", 896, 896},
    {"0.5B", "attn k/v", 128, 896},
    {"0.5B", "mlp gate/up", 4864, 896},
    {"0.5B", "mlp down", 896, 4864},
    {"0.5B", "lm_head", 151936, 896},
    {"1.5B", "attn q/o", 1536, 1536},
    {"1.5B", "attn k/v", 256, 1536},
    {"1.5B", "mlp gate/up", 8960, 1536},
    {"1.5B", "mlp down", 1536, 8960},
    {"1.5B", "lm_head", 151936, 1536},
};

struct GemvImpl {
    const char* name;
    void (*fn)(const float* x, const __nv_bfloat16* W, const __nv_bfloat16* b, int64_t T,
               int64_t in, int64_t out, float* y);
};
template <int Threads, bool Interleaved>
void rowpar(const float* x, const __nv_bfloat16* W, const __nv_bfloat16* b, int64_t T,
            int64_t in, int64_t out, float* y) {
    llm::gpu::launch_gemv_rowpar(x, W, b, T, in, out, y, Threads, Interleaved);
}
// rowpar_N = N threads per output row, interleaved elements; _c = contiguous chunks.
const GemvImpl kImpls[] = {
    {"gemv_naive", llm::gpu::launch_linear_mine},
    {"rowpar_32", rowpar<32, true>},     {"rowpar_64", rowpar<64, true>},
    {"rowpar_128", rowpar<128, true>},   {"rowpar_256", rowpar<256, true>},
    {"rowpar_512", rowpar<512, true>},   {"rowpar_1024", rowpar<1024, true>},
    {"rowpar_64c", rowpar<64, false>},   {"rowpar_256c", rowpar<256, false>},
    {"rowpar_auto", rowpar<0, true>},
};

// Cheap deterministic fill: bf16 patterns for values in roughly [-0.06, 0.06].
std::vector<uint16_t> fill_bf16(size_t n) {
    std::vector<uint16_t> w(n);
    uint32_t s = 0x9E3779B9u;
    for (size_t i = 0; i < n; i++) {
        s = s * 1664525u + 1013904223u;
        // sign | exponent 0x78..0x7A (2^-7..2^-5) | 7 mantissa bits
        w[i] = uint16_t(((s >> 31) << 15) | ((0x78u + (s >> 8) % 3u) << 7) | ((s >> 16) & 0x7Fu));
    }
    return w;
}

double gbps(double bytes, double ms) { return bytes / (ms / 1e3) / 1e9; }

} // namespace

int main(int argc, char** argv) {
    const double achievable = argc > 1 ? std::atof(argv[1]) : kStage0AchievableGBps;

    cublasHandle_t cublas = nullptr;
    if (cublasCreate(&cublas) != CUBLAS_STATUS_SUCCESS) {
        std::fprintf(stderr, "cublasCreate failed\n");
        return 1;
    }

    std::printf("decode GEMV (T=1), 100%% = %.0f GB/s measured achievable bandwidth\n\n", achievable);
    std::printf("%-5s %-12s %-12s %-11s %9s %8s %7s   %s\n", "model", "layer", "out x in",
                "impl", "median ms", "GB/s", "%achv", "vs cuBLAS fp32 (time)");

    for (const Shape& sh : kShapes) {
        const size_t n = size_t(sh.out) * sh.in;
        const int burst = n * 2 < kBurstBelowBytes ? kBurst : 1;
        const std::string shape = std::to_string(sh.out) + "x" + std::to_string(sh.in);
        const std::string dims = burst > 1 ? shape + "_x" + std::to_string(burst) : shape;

        std::vector<uint16_t> w_host = fill_bf16(n);
        std::vector<float> x_host(sh.in, 0.5f);
        __nv_bfloat16* W = nullptr;
        float *Wf = nullptr, *x = nullptr, *y = nullptr;
        CUDA_CHECK(cudaMalloc(&W, n * 2));
        CUDA_CHECK(cudaMalloc(&Wf, n * 4));
        CUDA_CHECK(cudaMalloc(&x, size_t(sh.in) * 4));
        CUDA_CHECK(cudaMalloc(&y, size_t(sh.out) * 4));
        CUDA_CHECK(cudaMemcpy(W, w_host.data(), n * 2, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(x, x_host.data(), size_t(sh.in) * 4, cudaMemcpyHostToDevice));
        llm::gpu::launch_bf16_to_f32(W, int64_t(n), Wf);   // same mirror the engine builds
        CUDA_CHECK(cudaDeviceSynchronize());

        // The yardstick: exactly the call GpuModel::Impl::linear makes on --gemm=cublas.
        const float alpha = 1.0f, beta = 0.0f;
        bench::Result rc = bench::run("gemv_cublas", dims, [&] {
            for (int i = 0; i < burst; i++)
                cublasSgemm(cublas, CUBLAS_OP_T, CUBLAS_OP_N, int(sh.out), 1, int(sh.in),
                            &alpha, Wf, int(sh.in), x, int(sh.in), &beta, y, int(sh.out));
        }, 10, 50, "cublas");
        bench::write_record(rc);
        const double cublas_ms = rc.median_ms / burst;

        for (const GemvImpl& impl : kImpls) {
            bench::Result r = bench::run(impl.name, dims, [&] {
                for (int i = 0; i < burst; i++) impl.fn(x, W, nullptr, 1, sh.in, sh.out, y);
                CUDA_CHECK_LAUNCH();
            }, 10, 50, "mine");
            bench::write_record(r);
            const double ms = r.median_ms / burst;
            const double g = gbps(double(n) * 2, ms);
            std::printf("%-5s %-12s %-12s %-11s %9.4f %8.1f %6.1f%%   %.1f%% of cuBLAS speed\n",
                        sh.model, sh.layer, shape.c_str(), impl.name, ms, g,
                        100 * g / achievable, 100 * cublas_ms / ms);
        }
        const double gc = gbps(double(n) * 4, cublas_ms);
        std::printf("%-5s %-12s %-12s %-11s %9.4f %8.1f %6.1f%%   (reads 2x the bytes)\n\n",
                    sh.model, sh.layer, shape.c_str(), "cublas fp32", cublas_ms, gc,
                    100 * gc / achievable);

        CUDA_CHECK(cudaFree(W));
        CUDA_CHECK(cudaFree(Wf));
        CUDA_CHECK(cudaFree(x));
        CUDA_CHECK(cudaFree(y));
    }
    cublasDestroy(cublas);
    return 0;
}
