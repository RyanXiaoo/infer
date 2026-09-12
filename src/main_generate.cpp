// main_generate.cpp — generation demo + baselines.
//
// No tokenizer until Stage 4 Phase C, so prompts are golden token IDs by index
// (tests/golden/prompt<i>_bf16.npz). Greedy-decodes n tokens, printing IDs,
// prefill time, and decode tokens/sec.
//
// Usage: main_generate [prompt=0] [n_new=16] [device=cpu|gpu] [gemm=mine|cublas] [cache=on|off]
//   cache=off is the Stage 3 full-recompute path; cache=on is the Stage 4
//   KV-cache path. Both produce identical tokens; the gap is the whole point,
//   and it GROWS with n_new (recompute is O(T) per token, cache is O(1) model work).

#include "forward.h"
#include "session.h"
#include "model.h"
#include "npy.h"

#ifdef HAVE_GPU
#include "../kernels/model_gpu.h"
#include "../kernels/session_gpu.h"
#endif

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

using Clock = std::chrono::steady_clock;
static double since(Clock::time_point t0) {
    return std::chrono::duration<double>(Clock::now() - t0).count();
}

int main(int argc, char** argv) {
    const std::string root = MODEL_ROOT;
    const int prompt_idx = argc > 1 ? std::atoi(argv[1]) : 0;
    const int n_new = argc > 2 ? std::atoi(argv[2]) : 16;
    const std::string device = argc > 3 ? argv[3] : "cpu";
    const std::string gemm = argc > 4 ? argv[4] : "mine";
    const bool cache = argc > 5 ? (std::string(argv[5]) != "off") : true;

    llm::Model model;
    model.load(root + "/models/Qwen2.5-0.5B-Instruct");

    auto g = llm::npy::load_npz(root + "/tests/golden/prompt" +
                                std::to_string(prompt_idx) + "_bf16.npz");
    const auto& ids_arr = g.at("token_ids");
    std::vector<int64_t> ids(ids_arr.i64(), ids_arr.i64() + ids_arr.numel());

    std::printf("prompt %d (%zu tokens), device=%s gemm=%s cache=%s:\n", prompt_idx,
                ids.size(), device.c_str(), gemm.c_str(), cache ? "on" : "off");

    double prefill_s = 0, decode_s = 0;
    std::vector<int64_t> out;

    if (device == "cpu") {
        if (cache) {
            llm::Session s(model, /*max_seq=*/n_new + int(ids.size()) + 8);
            auto t0 = Clock::now();
            std::vector<float> logits = s.prefill(ids);
            prefill_s = since(t0);
            t0 = Clock::now();
            const int64_t V = model.cfg.vocab_size;
            for (int i = 0; i < n_new; i++) {
                int64_t best = 0;
                for (int64_t v = 1; v < V; v++) if (logits[v] > logits[best]) best = v;
                out.push_back(best);
                logits = s.decode_one(best);
            }
            decode_s = since(t0);
        } else {
            auto t0 = Clock::now();
            llm::forward(model, ids, nullptr, /*last_only=*/true);
            prefill_s = since(t0);
            t0 = Clock::now();
            out = llm::greedy_decode(model, ids, n_new);
            decode_s = since(t0);
        }
    } else if (device == "gpu") {
#ifdef HAVE_GPU
        llm::GemmPath path = gemm == "cublas" ? llm::GemmPath::kCublas : llm::GemmPath::kMine;
        llm::GpuModel gpu(model);
        if (cache) {
            const int64_t ms = n_new + int(ids.size()) + 8;
            llm::GpuSession warm(gpu, ms, path);
            warm.prefill(ids);   // warm-up (context, cublas mirrors)
            llm::GpuSession s(gpu, ms, path);
            auto t0 = Clock::now();
            std::vector<float> logits = s.prefill(ids);
            prefill_s = since(t0);
            t0 = Clock::now();
            const int64_t V = model.cfg.vocab_size;
            for (int i = 0; i < n_new; i++) {
                int64_t best = 0;
                for (int64_t v = 1; v < V; v++) if (logits[v] > logits[best]) best = v;
                out.push_back(best);
                logits = s.decode_one(best);
            }
            decode_s = since(t0);
        } else {
            llm::forward_gpu(gpu, ids, nullptr, true, path);   // warm-up
            auto t0 = Clock::now();
            llm::forward_gpu(gpu, ids, nullptr, /*last_only=*/true, path);
            prefill_s = since(t0);
            t0 = Clock::now();
            out = llm::greedy_decode_gpu(gpu, ids, n_new, path);
            decode_s = since(t0);
        }
#else
        std::fprintf(stderr, "this build has no GPU support\n");
        return 1;
#endif
    } else {
        std::fprintf(stderr, "unknown device '%s' (cpu|gpu)\n", device.c_str());
        return 1;
    }

    std::printf("generated:");
    for (auto v : out) std::printf(" %lld", (long long)v);
    std::printf("\nprefill: %.3f s (%zu tokens)\ndecode:  %.3f s for %d tokens = "
                "%.3f tok/s\n", prefill_s, ids.size(), decode_s, n_new, n_new / decode_s);
    return 0;
}
