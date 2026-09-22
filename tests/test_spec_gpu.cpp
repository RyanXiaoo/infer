// test_spec_gpu.cpp — greedy speculative decoding must equal the target's own
// greedy decoding, token for token (Stage 11).
//
// Draft: Qwen2.5-0.5B, target: Qwen2.5-1.5B (same tokenizer; the golden
// prompts' ids are shared). For k in {1, 2, 4, 8}: every prompt's speculative
// stream == greedy_decode_cached_gpu(target). Then two slots speculating
// together through the scheduler == each alone. Any bug in truncation, verify
// or the accept rule shows up here as a token mismatch. Acceptance rates are
// reported.

#include "../kernels/batch_gpu.h"
#include "../kernels/model_gpu.h"
#include "../kernels/session_gpu.h"
#include "../kernels/spec_engine.h"
#include "../src/model.h"
#include "../src/model_select.h"
#include "../src/npy.h"
#include "../src/scheduler.h"

#include <cstdio>
#include <fstream>
#include <map>
#include <string>
#include <vector>

#include "../third_party/nlohmann/json.hpp"

static int failures = 0;
static constexpr int N = 32;
static constexpr int64_t kMaxSeq = 160;

int main() {
    const std::string root = MODEL_ROOT;
    const std::string target_name = "Qwen2.5-1.5B-Instruct", draft_name = "Qwen2.5-0.5B-Instruct";
    const std::string golden = root + "/tests/golden/" + target_name;
    std::ifstream mf(golden + "/manifest.json");
    if (!mf) { std::printf("SKIP: goldens for %s not found\n", target_name.c_str()); return 0; }
    const int n_prompts = int(nlohmann::json::parse(mf)["prompts"].size());

    llm::Model tm, dm;
    tm.load(root + "/models/" + target_name);
    dm.load(root + "/models/" + draft_name);
    llm::GpuModel tgpu(tm), dgpu(dm);

    std::vector<std::vector<int64_t>> prompts, ref;
    for (int pi = 0; pi < n_prompts; pi++) {
        auto g = llm::npy::load_npz(golden + "/prompt" + std::to_string(pi) + "_bf16.npz");
        const auto& ia = g.at("token_ids");
        prompts.emplace_back(ia.i64(), ia.i64() + ia.numel());
        ref.push_back(llm::greedy_decode_cached_gpu(tgpu, prompts.back(), N, kMaxSeq, llm::GemmPath::kMine));
    }

    for (int k : {1, 2, 4, 8}) {
        llm::GpuBatch target(tgpu, 1, kMaxSeq, llm::GemmPath::kMine, 0, false, true);
        llm::GpuBatch draft(dgpu, 1, kMaxSeq, llm::GemmPath::kMine, 0, false, true);
        target.set_vocab_limit(std::min(tm.cfg.vocab_size, dm.cfg.vocab_size));
        draft.set_vocab_limit(std::min(tm.cfg.vocab_size, dm.cfg.vocab_size));
        llm::SpecEngine spec(target, draft, k);
        int ok = 0;
        for (int pi = 0; pi < n_prompts; pi++) {
            std::vector<int64_t> out{spec.prefill(0, prompts[pi])};
            while (int(out.size()) < N) {
                auto m = spec.step_multi({{0, out.back(), {}}});
                for (int64_t t : m[0]) if (int(out.size()) < N) out.push_back(t);
            }
            spec.release(0);
            if (out == ref[pi]) ok++;
            else { failures++; std::printf("FAIL k=%d prompt%d: speculative stream differs\n", k, pi); }
        }
        const llm::SpecStats& st = spec.stats();
        std::printf("k=%d: %d/%d prompts identical to target greedy; acceptance %.0f%%, %.2f tokens per round\n",
                    k, ok, n_prompts, 100 * st.acceptance(), st.tokens_per_round());
    }

    {   // batched: 2 slots through the scheduler, k = 4
        llm::GpuBatch target(tgpu, 2, kMaxSeq, llm::GemmPath::kMine, 0, false, true);
        llm::GpuBatch draft(dgpu, 2, kMaxSeq, llm::GemmPath::kMine, 0, false, true);
        target.set_vocab_limit(std::min(tm.cfg.vocab_size, dm.cfg.vocab_size));
        draft.set_vocab_limit(std::min(tm.cfg.vocab_size, dm.cfg.vocab_size));
        llm::SpecEngine spec(target, draft, 4);
        llm::SchedulerConfig cfg;
        cfg.eos_id = -1;
        llm::Scheduler sch(spec, cfg);
        for (int pi = 0; pi < n_prompts; pi++) { llm::Request r; r.id = pi; r.prompt = prompts[pi]; r.max_new = N; sch.submit(r); }
        sch.run_until_idle();
        std::map<int64_t, std::vector<int64_t>> got;
        for (const auto& c : sch.completed()) got[c.id] = c.tokens;
        int ok = 0;
        for (int pi = 0; pi < n_prompts; pi++) {
            if (got[pi] == ref[pi]) ok++;
            else { failures++; std::printf("FAIL batched prompt%d: stream differs (%zu tokens)\n", pi, got[pi].size()); }
        }
        std::printf("batched 2 slots, k=4: %d/%d identical, %lld scheduler steps, acceptance %.0f%%\n",
                    ok, n_prompts, (long long)sch.steps(), 100 * spec.stats().acceptance());
    }

    if (failures == 0) { std::printf("test_spec_gpu: speculative == target greedy\n"); return 0; }
    std::printf("test_spec_gpu: %d FAILURES\n", failures);
    return 1;
}
