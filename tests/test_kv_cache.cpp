// test_kv_cache.cpp — the KV cache must not change the answer.
//
// Gate: cached greedy decoding produces the IDENTICAL token stream as the
// Stage 3 full-recompute greedy, on every golden prompt. That is the hard
// correctness claim; the HF goldens still gate the underlying forward math
// (test_forward), so this test targets the cache logic specifically.
//
// Also prints per-step logit agreement for the first few steps as diagnostic
// context. Bit-equality of logits is NOT expected or asserted: decode_one's
// single-query attention sums over the cache in the same order as the full
// pass, so agreement is near-exact, but any tiny reassociation is fine as long
// as the argmax (hence the token) matches. Token equality is the real gate.

#include "../src/forward.h"
#include "../src/session.h"
#include "../src/model.h"
#include "../src/npy.h"

#ifdef HAVE_GPU
#include "../kernels/model_gpu.h"
#include "../kernels/session_gpu.h"
#endif

#include <cmath>
#include <cstdio>
#include <fstream>
#include <string>
#include <vector>

#include "../third_party/nlohmann/json.hpp"

static int failures = 0;
static constexpr int N = 32;

// --gpu-only skips the CPU half (minutes of single-threaded recompute) when
// iterating on GPU kernels; ctest always runs the full test.
int main(int argc, char** argv) {
    const bool gpu_only = argc > 1 && std::string(argv[1]) == "--gpu-only";
    const std::string root = MODEL_ROOT;
    const std::string golden = root + "/tests/golden";

    std::ifstream mf(golden + "/manifest.json");
    if (!mf) { std::printf("SKIP: goldens not found\n"); return 0; }
    nlohmann::json manifest = nlohmann::json::parse(mf);
    int n_prompts = int(manifest["prompts"].size());

    llm::Model model;
    model.load(root + "/models/Qwen2.5-0.5B-Instruct");

    for (int pi = 0; pi < n_prompts && !gpu_only; pi++) {
        const std::string tag = "prompt" + std::to_string(pi);
        auto g = llm::npy::load_npz(golden + "/" + tag + "_bf16.npz");
        const auto& ids_arr = g.at("token_ids");
        std::vector<int64_t> ids(ids_arr.i64(), ids_arr.i64() + ids_arr.numel());

        std::vector<int64_t> recompute = llm::greedy_decode(model, ids, N);
        std::vector<int64_t> cached = llm::greedy_decode_cached(model, ids, N);

        if (recompute != cached) {
            std::printf("FAIL %s: token streams differ\n  recompute:", tag.c_str());
            for (auto v : recompute) std::printf(" %lld", (long long)v);
            std::printf("\n  cached:   ");
            for (auto v : cached) std::printf(" %lld", (long long)v);
            std::printf("\n");
            failures++;
            continue;
        }

        // Diagnostic: per-step logit agreement for the first 8 steps.
        llm::Session s(model, 512);
        std::vector<float> clog = s.prefill(ids);
        std::vector<int64_t> seq = ids;
        double worst = 0.0;
        for (int step = 0; step < 8 && step < N; step++) {
            std::vector<float> rlog = llm::forward(model, seq, nullptr, /*last_only=*/true);
            const int64_t V = model.cfg.vocab_size;
            double m = 0.0;
            for (int64_t v = 0; v < V; v++)
                m = std::max(m, std::abs(double(clog[v]) - double(rlog[v])));
            worst = std::max(worst, m);
            int64_t best = 0;
            for (int64_t v = 1; v < V; v++) if (clog[v] > clog[best]) best = v;
            seq.push_back(best);
            clog = s.decode_one(best);
        }
        std::printf("%s: streams identical (%d tokens); worst per-step logit diff %.4g\n",
                    tag.c_str(), N, worst);
    }

#ifdef HAVE_GPU
    // GPU: cached vs recompute must produce identical token streams for every
    // combination of GEMM path, decode-attention kernel and decode-step sequence.
    // Recompute runs T > 1 (Stage 3 kernels); cached runs the Stage 5 T = 1 kernels.
    llm::GpuModel gpu(model);
    for (llm::GemmPath gemm : {llm::GemmPath::kMineNaive, llm::GemmPath::kMine, llm::GemmPath::kCublas})
        for (int pi = 0; pi < n_prompts; pi++) {
            auto g = llm::npy::load_npz(golden + "/prompt" + std::to_string(pi) + "_bf16.npz");
            const auto& ia = g.at("token_ids");
            std::vector<int64_t> ids(ia.i64(), ia.i64() + ia.numel());
            std::vector<int64_t> rec = llm::greedy_decode_gpu(gpu, ids, N, gemm);
            for (llm::AttnPath attn : {llm::AttnPath::kNaive, llm::AttnPath::kParallel})
                for (llm::StepPath step : {llm::StepPath::kUnfused, llm::StepPath::kFused}) {
                    const std::string name = std::string("gpu/") + llm::gemm_path_name(gemm) +
                        (attn == llm::AttnPath::kNaive ? "/attn-naive" : "/attn-par") +
                        (step == llm::StepPath::kUnfused ? "/unfused" : "/fused") +
                        " prompt" + std::to_string(pi);
                    std::vector<int64_t> cac =
                        llm::greedy_decode_cached_gpu(gpu, ids, N, 512, gemm, attn, step);
                    if (rec != cac) {
                        std::printf("FAIL %s: token streams differ\n", name.c_str());
                        failures++;
                    } else {
                        std::printf("%s: streams identical (%d tokens)\n", name.c_str(), N);
                    }
                }
        }
#endif

    if (failures == 0) {
        std::printf("test_kv_cache: cached == recompute on all prompts\n");
        return 0;
    }
    std::printf("test_kv_cache: %d FAILURES\n", failures);
    return 1;
}
