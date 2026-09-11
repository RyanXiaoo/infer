// main_generate.cpp — Stage 2 demo + the honest baseline.
//
// No tokenizer until Stage 4, so prompts are golden token IDs by index
// (tests/golden/prompt<i>_bf16.npz). Greedy-decodes n tokens, printing IDs,
// prefill time, and decode tokens/sec — the "before" number every later
// speedup (CUDA port, KV cache, fused kernels) is measured against. Decode
// recomputes the full sequence per token ON PURPOSE; do not fix the slowness
// here, measure it.
//
// Usage: main_generate [prompt_index=0] [n_new=16]

#include "forward.h"
#include "model.h"
#include "npy.h"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

using Clock = std::chrono::steady_clock;

static double seconds_since(Clock::time_point t0) {
    return std::chrono::duration<double>(Clock::now() - t0).count();
}

int main(int argc, char** argv) {
    const std::string root = MODEL_ROOT;
    const int prompt_idx = argc > 1 ? std::atoi(argv[1]) : 0;
    const int n_new = argc > 2 ? std::atoi(argv[2]) : 16;

    llm::Model model;
    model.load(root + "/models/Qwen2.5-0.5B-Instruct");

    auto g = llm::npy::load_npz(root + "/tests/golden/prompt" +
                                std::to_string(prompt_idx) + "_bf16.npz");
    const auto& ids_arr = g.at("token_ids");
    std::vector<int64_t> ids(ids_arr.i64(), ids_arr.i64() + ids_arr.numel());

    std::printf("prompt %d (%zu tokens):", prompt_idx, ids.size());
    for (auto v : ids) std::printf(" %lld", (long long)v);
    std::printf("\n");

    // Prefill = one full forward over the prompt (logits for the last position).
    auto t0 = Clock::now();
    std::vector<float> logits = llm::forward(model, ids, nullptr, /*last_only=*/true);
    double prefill_s = seconds_since(t0);

    t0 = Clock::now();
    std::vector<int64_t> out = llm::greedy_decode(model, ids, n_new);
    double decode_s = seconds_since(t0);

    std::printf("generated:");
    for (auto v : out) std::printf(" %lld", (long long)v);
    std::printf("\nprefill: %.3f s (%zu tokens)\ndecode:  %.3f s for %d tokens = "
                "%.3f tok/s\n", prefill_s, ids.size(), decode_s, n_new,
                n_new / decode_s);
    return 0;
}
