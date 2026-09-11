// test_npy.cpp — proves the C++ npz reader can load the goldens Stage 2 will
// depend on: shapes coherent, dtypes right, and a value cross-check against
// the manifest-independent structure (logits vocab dim, hidden dim).

#include "../src/npy.h"

#include <cmath>
#include <cstdio>
#include <fstream>

static int failures = 0;

#define CHECK(cond, ...)                                                     \
    do {                                                                     \
        if (!(cond)) {                                                       \
            std::printf("FAIL %d: ", __LINE__);                              \
            std::printf(__VA_ARGS__);                                        \
            std::printf("\n");                                               \
            failures++;                                                      \
        }                                                                    \
    } while (0)

int main() {
    const std::string path = std::string(MODEL_ROOT) + "/tests/golden/prompt0_fp32.npz";
    if (!std::ifstream(path)) {
        std::printf("SKIP: %s not found — run tools/dump_logits.py first\n", path.c_str());
        return 0;
    }

    auto arrs = llm::npy::load_npz(path);
    CHECK(arrs.count("token_ids"), "token_ids missing");
    CHECK(arrs.count("logits"), "logits missing");
    CHECK(arrs.count("hidden_state_0"), "hidden_state_0 missing");
    CHECK(arrs.count("rope_cos"), "rope_cos missing");
    CHECK(arrs.count("layer0.q_proj"), "layer0.q_proj tap missing");

    const auto& ids = arrs["token_ids"];
    CHECK(ids.dtype == llm::npy::Dtype::I64, "token_ids not i64");
    CHECK(ids.shape.size() == 2 && ids.shape[0] == 1, "token_ids shape");
    int64_t seq = ids.shape[1];
    CHECK(seq > 0 && seq < 100, "unreasonable seq_len %lld", (long long)seq);
    for (int64_t i = 0; i < seq; i++)
        CHECK(ids.i64()[i] >= 0 && ids.i64()[i] < 151936, "token id out of vocab range");

    const auto& logits = arrs["logits"];
    CHECK(logits.dtype == llm::npy::Dtype::F32, "logits not f32");
    CHECK(logits.shape == std::vector<int64_t>({1, seq, 151936}), "logits shape");
    // Sanity: finite, not all zero.
    double sum = 0;
    bool finite = true;
    for (int64_t i = 0; i < logits.numel(); i++) {
        float v = logits.f32()[i];
        if (!std::isfinite(v)) finite = false;
        sum += std::fabs(v);
    }
    CHECK(finite, "logits contain non-finite values");
    CHECK(sum > 0, "logits all zero");

    const auto& h0 = arrs["hidden_state_0"];
    CHECK(h0.shape == std::vector<int64_t>({1, seq, 896}), "hidden_state_0 shape");

    if (failures == 0) {
        std::printf("test_npy: all checks passed (%zu arrays, seq_len=%lld)\n",
                    arrs.size(), (long long)seq);
        return 0;
    }
    std::printf("test_npy: %d FAILURES\n", failures);
    return 1;
}
