// test_batch_gpu.cpp — batched decode produces the same tokens as one
// sequence at a time (Stage 6 gate).
//
// (a) The five golden prompts decoded together in one GpuBatch, N tokens each,
//     must equal each prompt decoded alone through GpuSession.
// (b) The same prompts pushed through the Scheduler with fewer slots than
//     requests, so prompts join and leave a running batch at different steps,
//     must give the same per-prompt streams. Batch composition must not leak
//     into any row's result.
// Both on gemm=mine (batched GEMV) and gemm=cublas (T = B Sgemm).

#include "../kernels/batch_gpu.h"
#include "../kernels/model_gpu.h"
#include "../kernels/session_gpu.h"
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
static constexpr int N = 24;
static constexpr int64_t kMaxSeq = 128;

int main() {
    const std::string root = MODEL_ROOT;
    const std::string golden = llm::golden_dir(root);
    std::ifstream mf(golden + "/manifest.json");
    if (!mf) { std::printf("SKIP: goldens not found\n"); return 0; }
    const int n_prompts = int(nlohmann::json::parse(mf)["prompts"].size());

    llm::Model model;
    model.load(llm::model_dir(root));
    llm::GpuModel gpu(model);

    std::vector<std::vector<int64_t>> prompts;
    for (int pi = 0; pi < n_prompts; pi++) {
        auto g = llm::npy::load_npz(golden + "/prompt" + std::to_string(pi) + "_bf16.npz");
        const auto& ia = g.at("token_ids");
        prompts.emplace_back(ia.i64(), ia.i64() + ia.numel());
    }

    for (llm::GemmPath gemm : {llm::GemmPath::kMine, llm::GemmPath::kCublas}) {
        const std::string gname = llm::gemm_path_name(gemm);
        std::vector<std::vector<int64_t>> ref;
        for (const auto& p : prompts)
            ref.push_back(llm::greedy_decode_cached_gpu(gpu, p, N, kMaxSeq, gemm));

        // (a) all prompts in one batch, lockstep.
        {
            llm::GpuBatch batch(gpu, n_prompts, kMaxSeq, gemm);
            std::vector<std::vector<int64_t>> out(n_prompts);
            std::vector<llm::StepRow> rows;
            for (int s = 0; s < n_prompts; s++) {
                out[s].push_back(batch.prefill(s, prompts[s]));
                rows.push_back({s, out[s].back()});
            }
            for (int t = 1; t < N; t++) {
                std::vector<int64_t> next = batch.step(rows);
                for (int s = 0; s < n_prompts; s++) { out[s].push_back(next[s]); rows[s].token = next[s]; }
            }
            for (int s = 0; s < n_prompts; s++) {
                if (out[s] != ref[s]) {
                    failures++;
                    std::printf("FAIL %s lockstep prompt%d: batched stream differs\n", gname.c_str(), s);
                } else {
                    std::printf("%s lockstep prompt%d: identical (%d tokens)\n", gname.c_str(), s, N);
                }
            }
        }

        // (b) staggered through the scheduler: 2 slots, 5 requests, mixed lengths
        // so retirements and admissions interleave.
        {
            llm::GpuBatch batch(gpu, 2, kMaxSeq, gemm);
            llm::SchedulerConfig cfg;
            cfg.eos_id = -1;
            llm::Scheduler sch(batch, cfg);
            for (int s = 0; s < n_prompts; s++) {
                llm::Request r;
                r.id = s;
                r.prompt = prompts[s];
                r.max_new = N - (s % 3) * 5;   // 24, 19, 14, 24, 19
                sch.submit(r);
            }
            sch.run_until_idle();
            std::map<int64_t, std::vector<int64_t>> got;
            for (const auto& c : sch.completed()) got[c.id] = c.tokens;
            for (int s = 0; s < n_prompts; s++) {
                std::vector<int64_t> want(ref[s].begin(), ref[s].begin() + (N - (s % 3) * 5));
                if (got[s] != want) {
                    failures++;
                    std::printf("FAIL %s staggered prompt%d: stream differs\n", gname.c_str(), s);
                } else {
                    std::printf("%s staggered prompt%d: identical (%zu tokens)\n", gname.c_str(), s, want.size());
                }
            }
            std::printf("%s staggered: %lld scheduler steps, %zu events\n", gname.c_str(),
                        (long long)sch.steps(), sch.events().size());
        }
    }

    if (failures == 0) { std::printf("test_batch_gpu: batched == single on all prompts\n"); return 0; }
    std::printf("test_batch_gpu: %d FAILURES\n", failures);
    return 1;
}
