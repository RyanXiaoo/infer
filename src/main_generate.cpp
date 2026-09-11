// main_generate.cpp — generation demo + the honest baselines.
//
// No tokenizer until Stage 4, so prompts are golden token IDs by index
// (tests/golden/prompt<i>_bf16.npz). Greedy-decodes n tokens, printing IDs,
// prefill time, and decode tokens/sec. Decode recomputes the full sequence
// per token ON PURPOSE (no KV cache until Stage 4); measure it, don't fix it.
//
// Usage: main_generate [prompt_index=0] [n_new=16] [device=cpu|gpu] [gemm=mine|cublas]
// The gpu device exists only in the CUDA build (the Mac build prints an error).

#include "forward.h"
#include "model.h"
#include "npy.h"

#ifdef HAVE_GPU
#include "../kernels/model_gpu.h"
#endif

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
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
    const std::string device = argc > 3 ? argv[3] : "cpu";
    const std::string gemm = argc > 4 ? argv[4] : "mine";

    llm::Model model;
    model.load(root + "/models/Qwen2.5-0.5B-Instruct");

    auto g = llm::npy::load_npz(root + "/tests/golden/prompt" +
                                std::to_string(prompt_idx) + "_bf16.npz");
    const auto& ids_arr = g.at("token_ids");
    std::vector<int64_t> ids(ids_arr.i64(), ids_arr.i64() + ids_arr.numel());

    std::printf("prompt %d (%zu tokens), device=%s gemm=%s:", prompt_idx, ids.size(),
                device.c_str(), gemm.c_str());
    for (auto v : ids) std::printf(" %lld", (long long)v);
    std::printf("\n");

    double prefill_s = 0, decode_s = 0;
    std::vector<int64_t> out;

    if (device == "cpu") {
        auto t0 = Clock::now();
        llm::forward(model, ids, nullptr, /*last_only=*/true);
        prefill_s = seconds_since(t0);
        t0 = Clock::now();
        out = llm::greedy_decode(model, ids, n_new);
        decode_s = seconds_since(t0);
    } else if (device == "gpu") {
#ifdef HAVE_GPU
        llm::GemmPath path = gemm == "cublas" ? llm::GemmPath::kCublas
                                              : llm::GemmPath::kMine;
        llm::GpuModel gpu(model);   // upload not timed: one-time startup cost
        // Warm-up pass (context init, cuBLAS workspace, fp32 mirrors if cublas).
        llm::forward_gpu(gpu, ids, nullptr, true, path);
        auto t0 = Clock::now();
        llm::forward_gpu(gpu, ids, nullptr, /*last_only=*/true, path);
        prefill_s = seconds_since(t0);
        t0 = Clock::now();
        out = llm::greedy_decode_gpu(gpu, ids, n_new, path);
        decode_s = seconds_since(t0);
#else
        std::fprintf(stderr, "this build has no GPU support (CPU-only machine)\n");
        return 1;
#endif
    } else {
        std::fprintf(stderr, "unknown device '%s' (cpu|gpu)\n", device.c_str());
        return 1;
    }

    std::printf("generated:");
    for (auto v : out) std::printf(" %lld", (long long)v);
    std::printf("\nprefill: %.3f s (%zu tokens)\ndecode:  %.3f s for %d tokens = "
                "%.3f tok/s\n", prefill_s, ids.size(), decode_s, n_new,
                n_new / decode_s);
    return 0;
}
