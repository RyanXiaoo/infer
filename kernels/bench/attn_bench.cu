// attn_bench.cu — single-query cached attention microbench (Stage 5 Phase 1).
//
// Decode attention for one token: one query vector per head against cache_len
// cached K/V rows. The arithmetic is tiny (14 heads * 64 positions * 64 dims is
// ~57k multiply-adds), so what this measures is how long the longest chain of
// one-after-another memory loads in any thread is, and how that grows with
// cache_len. Variants:
//   naive     one thread per head, every q.k dot product computed three times
//   stored    one thread per head, scores computed once and kept      (rung A)
//   par/N     one block of N threads per head splitting the positions (rung B)
//   par/auto  rung B with the launcher's own thread-count choice
//
// Every row is timed as kBurst back-to-back launches per rep (see gemv_bench.cu:
// a lone short kernel mostly measures WSL2/WDDM submission latency). Record dims
// carry "_x20"; the table prints per-launch microseconds. K/V are random: timing
// does not depend on the values.

#include "../common/bench.cuh"
#include "../common/cuda_check.cuh"
#include "../ops/ops.cuh"

#include <cstdio>
#include <functional>
#include <string>
#include <vector>

namespace {

constexpr int kBurst = 20;

struct Layout { const char* model; int64_t n_heads, n_kv, hd; };
const Layout kLayouts[] = {{"0.5B", 14, 2, 64}, {"1.5B", 12, 2, 128}};
const int64_t kLens[] = {16, 64, 256, 1024, 4096};

struct Bufs { float *q, *k, *v, *scores, *ctx; };
using Launch = std::function<void(const Bufs&, int64_t len, const Layout&)>;
struct Impl { const char* name; Launch fn; };

Impl par(const char* name, int threads) {
    return {name, [threads](const Bufs& b, int64_t len, const Layout& L) {
        llm::gpu::launch_attention_cached_par(b.q, b.k, b.v, b.scores, len, L.n_heads,
                                              L.n_kv, L.hd, b.ctx, threads);
    }};
}
const std::vector<Impl> kImpls = {
    {"attn_naive", [](const Bufs& b, int64_t len, const Layout& L) {
        llm::gpu::launch_attention_cached(b.q, b.k, b.v, len, L.n_heads, L.n_kv, L.hd, b.ctx);
    }},
    {"attn_stored", [](const Bufs& b, int64_t len, const Layout& L) {
        llm::gpu::launch_attention_cached_stored(b.q, b.k, b.v, b.scores, len, L.n_heads,
                                                 L.n_kv, L.hd, b.ctx);
    }},
    par("attn_par256", 256),
    par("attn_par1024", 1024),
    par("attn_par_auto", 0),
};

float* device_random(size_t n) {
    std::vector<float> h(n);
    uint32_t s = 0x2545F491u;
    for (float& f : h) {
        s = s * 1664525u + 1013904223u;
        f = float(int32_t(s >> 8) % 2001 - 1000) / 1000.0f;
    }
    float* d = nullptr;
    CUDA_CHECK(cudaMalloc(&d, n * 4));
    CUDA_CHECK(cudaMemcpy(d, h.data(), n * 4, cudaMemcpyHostToDevice));
    return d;
}

} // namespace

int main() {
    std::printf("cached decode attention, microseconds per launch (median of 50 x %d-launch bursts)\n\n",
                kBurst);
    std::printf("%-5s %9s", "model", "cache_len");
    for (const Impl& impl : kImpls) std::printf(" %14s", impl.name);
    std::printf("   best vs naive\n");

    for (const Layout& L : kLayouts) {
        const int64_t max_len = kLens[sizeof(kLens) / sizeof(kLens[0]) - 1];
        const int64_t kv_dim = L.n_kv * L.hd;
        Bufs b{device_random(L.n_heads * L.hd), device_random(max_len * kv_dim),
               device_random(max_len * kv_dim), device_random(L.n_heads * max_len),
               device_random(L.n_heads * L.hd)};
        for (int64_t len : kLens) {
            const std::string dims = std::string(L.model) + "_len" + std::to_string(len) +
                                     "_x" + std::to_string(kBurst);
            std::printf("%-5s %9lld", L.model, (long long)len);
            double naive_us = 0, best_us = 1e30;
            for (const Impl& impl : kImpls) {
                bench::Result r = bench::run(impl.name, dims, [&] {
                    for (int i = 0; i < kBurst; i++) impl.fn(b, len, L);
                    CUDA_CHECK_LAUNCH();
                });
                bench::write_record(r);
                const double us = r.median_ms * 1e3 / kBurst;
                if (&impl == &kImpls[0]) naive_us = us; else if (us < best_us) best_us = us;
                std::printf(" %14.1f", us);
            }
            std::printf("   %6.1fx\n", naive_us / best_us);
        }
        std::printf("\n");
        for (float* p : {b.q, b.k, b.v, b.scores, b.ctx}) CUDA_CHECK(cudaFree(p));
    }
    return 0;
}
