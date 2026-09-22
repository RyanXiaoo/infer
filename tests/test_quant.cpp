// test_quant.cpp — quantiser round trip and container (Stage 10), Mac-only.

#include "../src/f16.h"
#include "../src/quant.h"

#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

namespace {
int failures = 0;
#define CHECK(cond, ...) \
    do { if (!(cond)) { failures++; std::printf("FAIL %s:%d: ", __FILE__, __LINE__); \
         std::printf(__VA_ARGS__); std::printf("\n"); } } while (0)

std::vector<uint16_t> random_bf16(std::mt19937& rng, size_t n, float scale) {
    std::normal_distribution<float> d(0.0f, scale);
    std::vector<uint16_t> v(n);
    for (auto& x : v) x = f16::f32_to_bf16(d(rng));
    return v;
}
} // namespace

int main() {
    using namespace llm;
    std::mt19937 rng(10);
    for (QKind kind : {QKind::kInt8, QKind::kInt4}) {
        for (int64_t cols : {128, 130, 896, 1536}) {   // 130: a partial trailing group
            const int64_t rows = 37;
            std::vector<uint16_t> w = random_bf16(rng, size_t(rows) * cols, 0.05f);
            QMatrix q = quantize(w.data(), rows, cols, kind);
            const QTensor v = q.view();
            double worst = 0.0, sum_err = 0.0, sum_abs = 0.0;
            for (int64_t r = 0; r < rows; r++)
                for (int64_t g = 0; g < v.groups(); g++) {
                    float amax = 0, err = 0;
                    for (int64_t c = g * kGroup; c < std::min(cols, (g + 1) * kGroup); c++) {
                        const float x = f16::bf16_to_f32(w[size_t(r * cols + c)]);
                        const float e = std::fabs(x - dequant(v, r, c));
                        amax = std::max(amax, std::fabs(x)); err = std::max(err, e);
                        sum_err += e; sum_abs += std::fabs(x);
                    }
                    worst = std::max(worst, double(err / amax));
                }
            // Bounds: half a step of the stored scale, plus bf16 rounding of the scale (~0.4%).
            const double bound = kind == QKind::kInt8 ? 0.5 / 127 + 0.004 : 0.5 / 15 * 2 + 0.01;
            CHECK(worst <= bound, "%s cols=%lld: worst group error %.4f > %.4f", qkind_name(kind),
                  (long long)cols, worst, bound);
            std::printf("%s cols=%lld: worst %.4f of absmax, mean |err|/mean |w| = %.4f, bytes %zu -> %zu\n",
                        qkind_name(kind), (long long)cols, worst, sum_err / sum_abs, w.size() * 2,
                        q.q.size() + q.scales.size() * 2 + q.zeros.size());
        }
    }
    {   // int4 packing: nibbles land where dequant() reads them
        std::vector<uint16_t> w(256);
        for (int i = 0; i < 256; i++) w[size_t(i)] = f16::f32_to_bf16(float(i % 16) - 7.5f);
        QMatrix q = quantize(w.data(), 2, 128, QKind::kInt4);
        const QTensor v = q.view();
        bool ok = true;
        // Values sit exactly on half-steps of the scale (1.0), so the error is s/2.
        for (int64_t c = 0; c < 128; c++) ok &= std::fabs(dequant(v, 0, c) - (float(c % 16) - 7.5f)) <= 0.51f;
        CHECK(ok, "int4 ramp does not round-trip");
    }
    {   // container: write, read back, identical bytes and views
        std::vector<uint16_t> w = random_bf16(rng, 8 * 256, 0.1f), n = random_bf16(rng, 8, 1.0f);
        QMatrix q = quantize(w.data(), 8, 256, QKind::kInt8);
        QuantWriter wr;
        wr.add("m", q);
        wr.add_bf16("n", n.data(), {8});
        const std::string path = "/tmp/test_quant.llmq";
        wr.write(path);
        QuantFile f;
        f.open(path);
        CHECK(f.has("m") && f.has("n") && f.kind() == QKind::kInt8, "container names/kind");
        const QTensor& m = f.get("m");
        bool same = m.rows == 8 && m.cols == 256;
        for (int64_t r = 0; r < 8 && same; r++)
            for (int64_t c = 0; c < 256; c++)
                if (dequant(m, r, c) != dequant(q.view(), r, c)) { same = false; break; }
        CHECK(same, "int8 tensor differs after read-back");
        const QTensor& nv = f.get("n");
        CHECK(nv.kind == QKind::kBf16 && nv.rows == 8 && nv.bf16[3] == n[3], "bf16 tensor read-back");
    }
    if (failures == 0) { std::printf("test_quant: all checks passed\n"); return 0; }
    std::printf("test_quant: %d FAILURES\n", failures);
    return 1;
}
