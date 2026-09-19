// test_ops_gpu.cu — kernel-level correctness for the two Stage 5 hot kernels:
// the T=1 GEMV (linear) and the single-query cached attention.
//
// Why this exists next to test_forward_gpu / test_kv_cache: those run real
// prompts, which only ever exercise the model's own shapes and cache lengths
// up to ~270. Parallel reductions break at the edges instead: a length of 1, a
// length one past a block or warp boundary (33, 257), a width that is not a
// multiple of the thread count. So this test sweeps exactly those sizes on
// random data, per kernel, in seconds.
//
// Reference = the same math on the host in double precision, so every
// implementation (the naive kernels today, the optimized ones as they land) is
// judged against the true value rather than against another fp32 summation
// order. To add an implementation, append one line to gemv_impls / attn_impls.
//
// Traps built into the data:
//   * output buffers carry canary words past the end (catches overrun writes);
//   * cache rows past cache_len are NaN (catches reading beyond the live rows);
//   * "spike" attention cases put nearly all softmax weight on one position
//     (first / middle / last), so silently dropping that position is a gross
//     error instead of a 1/cache_len one.
//
// BENCH_SMOKE=1 (set by tools/sanitize.sh) drops the large shapes: racecheck
// runs 10-100x slower and the edge sizes are all small anyway.
//
// Usage: test_ops_gpu [--verbose] [--selftest]
//   --verbose   print the worst error of every check (how much tolerance is left)
//   --selftest  swap in deliberately broken kernels (one input term dropped, the
//               last cache position dropped) and require the sweep to REJECT
//               them. A test that cannot fail proves nothing; this shows the
//               tolerances are tight enough to see a single missing element.

#include "../kernels/common/cuda_check.cuh"
#include "../kernels/ops/ops.cuh"
#include "../src/f16.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <random>
#include <string>
#include <vector>

namespace {

using GemvFn = void (*)(const float* x, const __nv_bfloat16* W, const __nv_bfloat16* b,
                        int64_t T, int64_t in, int64_t out, float* y);
using AttnFn = void (*)(const float* q, const float* k_cache, const float* v_cache,
                        int64_t cache_len, int64_t n_heads, int64_t n_kv, int64_t hd,
                        float* ctx);

struct GemvImpl { const char* name; GemvFn fn; };
struct AttnImpl { const char* name; AttnFn fn; };

// The Stage 5 attention kernels take a scores scratch buffer; the test's common
// signature does not, so these wrappers own one (NaN-filled before every call:
// a kernel that reads a score it never wrote poisons its output).
template <typename Launch>
void with_scores_scratch(int64_t n_heads, int64_t cache_len, Launch launch) {
    const size_t n = size_t(n_heads) * cache_len;
    float* scores = nullptr;
    CUDA_CHECK(cudaMalloc(&scores, n * 4));
    CUDA_CHECK(cudaMemset(scores, 0xFF, n * 4));   // all-ones bit pattern = NaN
    launch(scores);
    CUDA_CHECK(cudaDeviceSynchronize());
    cudaFree(scores);
}
void attn_stored(const float* q, const float* k, const float* v, int64_t len, int64_t nh,
                 int64_t nkv, int64_t hd, float* ctx) {
    with_scores_scratch(nh, len, [&](float* sc) {
        llm::gpu::launch_attention_cached_stored(q, k, v, sc, len, nh, nkv, hd, ctx);
    });
}
template <int Threads>
void attn_par(const float* q, const float* k, const float* v, int64_t len, int64_t nh,
              int64_t nkv, int64_t hd, float* ctx) {
    with_scores_scratch(nh, len, [&](float* sc) {
        llm::gpu::launch_attention_cached_par(q, k, v, sc, len, nh, nkv, hd, ctx, Threads);
    });
}

template <int Threads, bool Interleaved>
void gemv_rowpar(const float* x, const __nv_bfloat16* W, const __nv_bfloat16* b, int64_t T,
                 int64_t in, int64_t out, float* y) {
    llm::gpu::launch_gemv_rowpar(x, W, b, T, in, out, y, Threads, Interleaved);
}

std::vector<GemvImpl> gemv_impls = {
    {"naive", llm::gpu::launch_linear_mine},
    {"rowpar/auto", gemv_rowpar<0, true>},
    {"rowpar/32", gemv_rowpar<32, true>},
    {"rowpar/1024", gemv_rowpar<1024, true>},       // more threads than elements at small in
    {"rowpar/64-contig", gemv_rowpar<64, false>},
    {"rowpar/1024-contig", gemv_rowpar<1024, false>},
};
std::vector<AttnImpl> attn_impls = {
    {"naive", llm::gpu::launch_attention_cached},
    {"stored", attn_stored},
    {"par/auto", attn_par<0>},
    {"par/32", attn_par<32>},       // fewer threads than hd: the hd > B path
    {"par/64", attn_par<64>},       // threads == hd (hd 64): one class
    {"par/1024", attn_par<1024>},   // the largest block
};

// --selftest stand-ins. Each behaves like a kernel with one classic reduction
// bug: the final element never makes it into the sum.
void broken_gemv_drops_last_term(const float* x, const __nv_bfloat16* W,
                                 const __nv_bfloat16* b, int64_t T, int64_t in,
                                 int64_t out, float* y) {
    float* xc = nullptr;
    CUDA_CHECK(cudaMalloc(&xc, size_t(T) * in * 4));
    CUDA_CHECK(cudaMemcpy(xc, x, size_t(T) * in * 4, cudaMemcpyDeviceToDevice));
    for (int64_t t = 0; t < T; t++)
        CUDA_CHECK(cudaMemset(xc + t * in + (in - 1), 0, 4));
    llm::gpu::launch_linear_mine(xc, W, b, T, in, out, y);
    CUDA_CHECK(cudaDeviceSynchronize());
    cudaFree(xc);
}
void broken_attn_drops_last_position(const float* q, const float* k_cache,
                                     const float* v_cache, int64_t cache_len,
                                     int64_t n_heads, int64_t n_kv, int64_t hd,
                                     float* ctx) {
    llm::gpu::launch_attention_cached(q, k_cache, v_cache, cache_len - 1, n_heads, n_kv,
                                      hd, ctx);
}

constexpr int kCanary = 16;
constexpr float kCanaryValue = -12345.5f;

int failures = 0;
int checks = 0;
bool smoke = false;
bool verbose = false;
bool selftest = false;   // failures are the expected outcome: keep them quiet

// Device buffer holding a host vector's contents.
template <typename T>
struct Dev {
    T* p = nullptr;
    size_t n = 0;
    explicit Dev(const std::vector<T>& h) : n(h.size()) {
        CUDA_CHECK(cudaMalloc(&p, n * sizeof(T)));
        CUDA_CHECK(cudaMemcpy(p, h.data(), n * sizeof(T), cudaMemcpyHostToDevice));
    }
    ~Dev() { cudaFree(p); }
    Dev(const Dev&) = delete;
    Dev& operator=(const Dev&) = delete;
    std::vector<T> download() const {
        std::vector<T> h(n);
        CUDA_CHECK(cudaMemcpy(h.data(), p, n * sizeof(T), cudaMemcpyDeviceToHost));
        return h;
    }
};

std::vector<float> randn(std::mt19937& rng, size_t n, float stddev = 1.0f) {
    std::normal_distribution<float> d(0.0f, stddev);
    std::vector<float> v(n);
    for (float& f : v) f = d(rng);
    return v;
}

// Random bf16 values. Returns the raw 16-bit patterns; `as_f32` receives the
// value each pattern actually decodes to (what the kernel will compute with).
std::vector<uint16_t> rand_bf16(std::mt19937& rng, size_t n, float stddev,
                                std::vector<float>& as_f32) {
    std::vector<float> f = randn(rng, n, stddev);
    std::vector<uint16_t> bits(n);
    as_f32.resize(n);
    for (size_t i = 0; i < n; i++) {
        bits[i] = f16::f32_to_bf16(f[i]);
        as_f32[i] = f16::bf16_to_f32(bits[i]);
    }
    return bits;
}

bool canary_intact(const std::vector<float>& buf, size_t payload) {
    for (size_t i = payload; i < buf.size(); i++)
        if (buf[i] != kCanaryValue) return false;
    return true;
}

void report(bool ok, const std::string& what, double worst, double limit,
            const char* detail = "") {
    checks++;
    if (ok) {
        if (verbose && limit > 0) std::printf("ok   %-58s %.3g\n", what.c_str(), worst);
        return;
    }
    failures++;
    if (selftest && !verbose) return;
    std::printf("FAIL %s: worst error %.3g of allowed %.3g %s\n", what.c_str(), worst,
                limit, detail);
}

// ------------------------------------------------------------------ GEMV
//
// Error is measured against the conditioning of each dot product,
// sum_i |w_i * x_i| (+ |b|): fp32 rounding scales with that, not with the
// (possibly cancelling) result. Measured on the naive kernel the worst case is
// 4e-7 (151936x896); 5e-6 leaves 10x room for a different summation order and
// is still 20x below 1/in at in=8960, the size of one dropped term.
constexpr double kGemvTol = 5e-6;

void test_gemv(std::mt19937& rng, int64_t T, int64_t in, int64_t out, bool bias) {
    std::vector<float> w_f, b_f;
    std::vector<uint16_t> w_bits = rand_bf16(rng, size_t(out) * in, 0.05f, w_f);
    std::vector<uint16_t> b_bits = rand_bf16(rng, size_t(out), 0.5f, b_f);
    std::vector<float> x = randn(rng, size_t(T) * in);

    std::vector<double> ref(size_t(T) * out), cond(size_t(T) * out);
    for (int64_t t = 0; t < T; t++)
        for (int64_t o = 0; o < out; o++) {
            double acc = bias ? double(b_f[o]) : 0.0;
            double mag = std::abs(acc);
            const float* wr = &w_f[size_t(o) * in];
            const float* xr = &x[size_t(t) * in];
            for (int64_t i = 0; i < in; i++) {
                double p = double(wr[i]) * double(xr[i]);
                acc += p;
                mag += std::abs(p);
            }
            ref[t * out + o] = acc;
            cond[t * out + o] = mag;
        }

    Dev<uint16_t> dW(w_bits), db(b_bits);
    Dev<float> dx(x);
    for (const GemvImpl& impl : gemv_impls) {
        Dev<float> dy(std::vector<float>(size_t(T) * out + kCanary, kCanaryValue));
        impl.fn(dx.p, reinterpret_cast<const __nv_bfloat16*>(dW.p),
                bias ? reinterpret_cast<const __nv_bfloat16*>(db.p) : nullptr, T, in, out,
                dy.p);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        std::vector<float> y = dy.download();

        double worst = 0.0;
        bool finite = true;
        for (size_t j = 0; j < ref.size(); j++) {
            if (!std::isfinite(y[j])) { finite = false; break; }
            double scale = cond[j] > 0 ? cond[j] : 1.0;
            worst = std::max(worst, std::abs(double(y[j]) - ref[j]) / scale);
        }
        std::string what = std::string("gemv/") + impl.name + " T=" + std::to_string(T) +
                           " " + std::to_string(out) + "x" + std::to_string(in) +
                           (bias ? " +bias" : "");
        if (!finite) report(false, what, INFINITY, kGemvTol, "(non-finite output)");
        else report(worst <= kGemvTol, what, worst, kGemvTol);
        report(canary_intact(y, ref.size()), what + " canary", 1, 0, "(wrote past y)");
    }
}

// ------------------------------------------------------------- attention
//
// Each output is a softmax-weighted average of value rows, so its error is
// measured against max|v|. The naive kernel's serial fp32 sums reach 2.2e-5 at
// cache_len 4097 (3e-6 or less up to 1024), so 5e-5 is the tightest bound that
// holds there. It is still below 1/cache_len = 2.4e-4, the weight of one dropped
// position under near-uniform attention; the spike cases cover the rest.
constexpr double kAttnTol = 5e-5;
constexpr int64_t kPoisonRows = 7;

enum class Spike { kNone, kFirst, kMiddle, kLast };

void test_attention(std::mt19937& rng, int64_t n_heads, int64_t n_kv, int64_t hd,
                    int64_t cache_len, Spike spike) {
    const int64_t kv_dim = n_kv * hd, group = n_heads / n_kv;
    const int64_t rows = cache_len + kPoisonRows;
    std::vector<float> q = randn(rng, size_t(n_heads) * hd);
    std::vector<float> k = randn(rng, size_t(rows) * kv_dim);
    std::vector<float> v = randn(rng, size_t(rows) * kv_dim);

    if (spike != Spike::kNone) {
        // Make one cached key (in every kv head) parallel to the group's first
        // query head: its score dominates that head's softmax.
        const int64_t s = spike == Spike::kFirst ? 0
                        : spike == Spike::kLast  ? cache_len - 1 : cache_len / 2;
        for (int64_t g = 0; g < n_kv; g++)
            for (int64_t d = 0; d < hd; d++)
                k[s * kv_dim + g * hd + d] = 3.0f * q[(g * group) * hd + d];
    }
    const float nan = std::numeric_limits<float>::quiet_NaN();
    for (int64_t i = cache_len * kv_dim; i < rows * kv_dim; i++) k[i] = v[i] = nan;

    double vmax = 0.0;
    for (int64_t i = 0; i < cache_len * kv_dim; i++) vmax = std::max(vmax, double(std::abs(v[i])));

    std::vector<double> ref(size_t(n_heads) * hd, 0.0), sc(cache_len);
    const double scale = 1.0 / std::sqrt(double(hd));
    for (int64_t h = 0; h < n_heads; h++) {
        const int64_t g = h / group;
        double maxs = -INFINITY;
        for (int64_t s = 0; s < cache_len; s++) {
            double acc = 0.0;
            for (int64_t d = 0; d < hd; d++)
                acc += double(q[h * hd + d]) * double(k[s * kv_dim + g * hd + d]);
            sc[s] = acc * scale;
            maxs = std::max(maxs, sc[s]);
        }
        double denom = 0.0;
        for (int64_t s = 0; s < cache_len; s++) denom += (sc[s] = std::exp(sc[s] - maxs));
        for (int64_t s = 0; s < cache_len; s++)
            for (int64_t d = 0; d < hd; d++)
                ref[h * hd + d] += sc[s] / denom * double(v[s * kv_dim + g * hd + d]);
    }

    Dev<float> dq(q), dk(k), dv(v);
    for (const AttnImpl& impl : attn_impls) {
        Dev<float> dctx(std::vector<float>(ref.size() + kCanary, kCanaryValue));
        impl.fn(dq.p, dk.p, dv.p, cache_len, n_heads, n_kv, hd, dctx.p);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        std::vector<float> ctx = dctx.download();

        double worst = 0.0;
        bool finite = true;
        for (size_t j = 0; j < ref.size(); j++) {
            if (!std::isfinite(ctx[j])) { finite = false; break; }
            worst = std::max(worst, std::abs(double(ctx[j]) - ref[j]) / vmax);
        }
        static const char* spike_name[] = {"", " spike=first", " spike=middle", " spike=last"};
        std::string what = std::string("attn/") + impl.name + " heads=" +
                           std::to_string(n_heads) + "/" + std::to_string(n_kv) + " hd=" +
                           std::to_string(hd) + " len=" + std::to_string(cache_len) +
                           spike_name[int(spike)];
        if (!finite) report(false, what, INFINITY, kAttnTol, "(non-finite: read past cache_len?)");
        else report(worst <= kAttnTol, what, worst, kAttnTol);
        report(canary_intact(ctx, ref.size()), what + " canary", 1, 0, "(wrote past ctx)");
    }
}

// ------------------------------------------------------ decode-step fusions
//
// The Phase 3 fused kernels claim to be bit-identical to the launches they
// replace (same arithmetic, same order), so they are held to exact equality
// against the unfused sequence rather than to a tolerance.
void test_add_rmsnorm(std::mt19937& rng, int64_t H) {
    std::vector<float> w_f;
    std::vector<uint16_t> w_bits = rand_bf16(rng, size_t(H), 1.0f, w_f);
    std::vector<float> h = randn(rng, size_t(H)), delta = randn(rng, size_t(H));
    Dev<uint16_t> dw(w_bits);
    Dev<float> dd(delta), h_ref(h), h_fus(h);
    Dev<float> o_ref(std::vector<float>(size_t(H) + kCanary, kCanaryValue));
    Dev<float> o_fus(std::vector<float>(size_t(H) + kCanary, kCanaryValue));
    const auto* w = reinterpret_cast<const __nv_bfloat16*>(dw.p);

    llm::gpu::launch_residual_add(h_ref.p, dd.p, H);
    llm::gpu::launch_rmsnorm(h_ref.p, w, 1e-6f, 1, H, o_ref.p);
    llm::gpu::launch_add_rmsnorm(h_fus.p, dd.p, w, 1e-6f, H, o_fus.p);
    CUDA_CHECK(cudaDeviceSynchronize());

    const std::string what = "fused/add_rmsnorm H=" + std::to_string(H);
    report(h_ref.download() == h_fus.download(), what + " residual", 1, 0, "(h differs)");
    report(o_ref.download() == o_fus.download(), what + " output", 1, 0, "(out differs)");
}

void test_rope_qk_append(std::mt19937& rng, int64_t n_heads, int64_t n_kv, int64_t hd,
                         int64_t pos) {
    const int64_t kv_dim = n_kv * hd, rows = pos + 3;
    std::vector<float> q = randn(rng, size_t(n_heads) * hd), k = randn(rng, size_t(kv_dim)),
                       v = randn(rng, size_t(kv_dim)), c = randn(rng, size_t(hd)),
                       s = randn(rng, size_t(hd));
    std::vector<float> cache0(size_t(rows) * kv_dim, kCanaryValue);   // untouched rows = canaries
    Dev<float> dc(c), ds(s), dv(v);
    Dev<float> q_ref(q), k_ref(k), kc_ref(cache0), vc_ref(cache0);
    Dev<float> q_fus(q), k_fus(k), kc_fus(cache0), vc_fus(cache0);

    llm::gpu::launch_rope(q_ref.p, dc.p, ds.p, 1, n_heads, hd);
    llm::gpu::launch_rope(k_ref.p, dc.p, ds.p, 1, n_kv, hd);
    llm::gpu::launch_cache_append(k_ref.p, dv.p, kc_ref.p, vc_ref.p, 1, kv_dim, pos);
    llm::gpu::launch_rope_qk_append(q_fus.p, k_fus.p, dv.p, dc.p, ds.p, n_heads, n_kv, hd,
                                    kc_fus.p + pos * kv_dim, vc_fus.p + pos * kv_dim);
    CUDA_CHECK(cudaDeviceSynchronize());

    const std::string what = "fused/rope_qk_append heads=" + std::to_string(n_heads) + "/" +
                             std::to_string(n_kv) + " hd=" + std::to_string(hd) +
                             " pos=" + std::to_string(pos);
    report(q_ref.download() == q_fus.download(), what + " q", 1, 0, "(q differs)");
    report(kc_ref.download() == kc_fus.download(), what + " k cache", 1, 0, "(k cache differs)");
    report(vc_ref.download() == vc_fus.download(), what + " v cache", 1, 0, "(v cache differs)");
}

} // namespace

int main(int argc, char** argv) {
    if (const char* s = std::getenv("BENCH_SMOKE"); s && s[0] == '1') smoke = true;
    for (int i = 1; i < argc; i++) {
        if (std::string(argv[i]) == "--verbose") verbose = true;
        else if (std::string(argv[i]) == "--selftest") selftest = true;
        else { std::fprintf(stderr, "usage: %s [--verbose] [--selftest]\n", argv[0]); return 2; }
    }
    if (selftest) {
        gemv_impls = {{"BROKEN-drops-last-term", broken_gemv_drops_last_term}};
        attn_impls = {{"BROKEN-drops-last-position", broken_attn_drops_last_position}};
    }
    std::mt19937 rng(20260919);

    // GEMV edge grid: widths around the warp size and the models' real widths,
    // against tiny and non-multiple-of-anything output counts.
    const std::vector<int64_t> ins = {1, 31, 32, 33, 63, 65, 896, 1536, 4864, 8960};
    const std::vector<int64_t> outs = {1, 127, 128};
    for (int64_t in : ins)
        for (int64_t out : outs) {
            test_gemv(rng, 1, in, out, /*bias=*/false);
            test_gemv(rng, 1, in, out, /*bias=*/true);
        }
    test_gemv(rng, 3, 896, 128, true);      // T>1 stays on whatever path serves prefill
    if (!smoke) {
        // The shapes that carry ~90% of decode GEMV work (out x in), both models.
        const int64_t real[][2] = {{4864, 896},  {896, 4864},  {151936, 896},
                                   {8960, 1536}, {1536, 8960}, {151936, 1536}};
        for (auto& s : real) test_gemv(rng, 1, s[1], s[0], false);
    }

    // Attention: both models' head layouts plus MHA and MQA extremes.
    const int64_t layouts[][3] = {{14, 2, 64}, {12, 2, 128}, {4, 4, 8}, {8, 1, 16}};
    std::vector<int64_t> lens = {1, 2, 3, 31, 32, 33, 63, 64, 65, 255, 256, 257};
    if (!smoke) { lens.push_back(1000); lens.push_back(1024); lens.push_back(4097); }
    for (auto& L : layouts)
        for (int64_t len : lens) {
            test_attention(rng, L[0], L[1], L[2], len, Spike::kNone);
            test_attention(rng, L[0], L[1], L[2], len, Spike::kLast);
            if (len > 2) {
                test_attention(rng, L[0], L[1], L[2], len, Spike::kFirst);
                test_attention(rng, L[0], L[1], L[2], len, Spike::kMiddle);
            }
        }

    if (!selftest) {
        for (int64_t H : {1, 255, 256, 257, 896, 1536}) test_add_rmsnorm(rng, H);
        const int64_t rope_layouts[][3] = {{14, 2, 64}, {12, 2, 128}, {4, 4, 8}, {8, 1, 16}};
        for (auto& L : rope_layouts)
            for (int64_t pos : {0, 1, 77}) test_rope_qk_append(rng, L[0], L[1], L[2], pos);
    }

    if (selftest) {
        // Every accuracy check must fail; the canary checks (the other half) pass.
        const int accuracy_checks = checks / 2;
        const bool rejected_all = failures == accuracy_checks;
        std::printf("test_ops_gpu --selftest: broken kernels rejected in %d of %d cases%s\n",
                    failures, accuracy_checks, rejected_all ? "" : "  <-- tolerances too loose");
        return rejected_all ? 0 : 1;
    }
    if (failures == 0) {
        std::printf("test_ops_gpu: %d checks passed (%zu gemv impl, %zu attention impl%s)\n",
                    checks, gemv_impls.size(), attn_impls.size(), smoke ? ", smoke" : "");
        return 0;
    }
    std::printf("test_ops_gpu: %d of %d checks FAILED\n", failures, checks);
    return 1;
}
