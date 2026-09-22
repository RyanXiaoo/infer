// test_batch_gpu.cpp — batched decode produces the same tokens as one
// sequence at a time (Stage 6 gate).
//
// (a) The five golden prompts decoded together in one GpuBatch, N tokens each,
//     must equal each prompt decoded alone through GpuSession.
// (b) The same prompts pushed through the Scheduler with fewer slots than
//     requests, so prompts join and leave a running batch at different steps,
//     must give the same per-prompt streams. Batch composition must not leak
//     into any row's result.
// (c) Stage 7: the same requests under a KV budget too small to hold them all
//     (forces preemption + recompute) and with a shared 32-token prefix
//     (prefix cache + copy-on-write) must still give the same streams.
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
            for (bool split : {false, true}) {
            llm::GpuBatch batch(gpu, n_prompts, kMaxSeq, gemm, 0, true, false, split);
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
                    std::printf("FAIL %s lockstep attn=%s prompt%d: batched stream differs\n", gname.c_str(), split ? "split" : "par", s);
                } else {
                    std::printf("%s lockstep attn=%s prompt%d: identical (%d tokens)\n", gname.c_str(), split ? "split" : "par", s, N);
                }
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

    // (d) Stage 8: the same lockstep + staggered runs with CUDA graphs must
    // give the same tokens (graphs replay the identical kernel sequence).
    {
        const llm::GemmPath gemm = llm::GemmPath::kMine;
        std::vector<std::vector<int64_t>> ref;
        for (const auto& p : prompts)
            ref.push_back(llm::greedy_decode_cached_gpu(gpu, p, N, kMaxSeq, gemm));
        llm::GpuBatch batch(gpu, 2, kMaxSeq, gemm, 0, true, /*use_graphs=*/true);
        llm::SchedulerConfig cfg;
        cfg.eos_id = -1;
        llm::Scheduler sch(batch, cfg);
        for (int s = 0; s < n_prompts; s++) {
            llm::Request r; r.id = s; r.prompt = prompts[s]; r.max_new = N - (s % 3) * 5;
            sch.submit(r);
        }
        sch.run_until_idle();
        std::map<int64_t, std::vector<int64_t>> got;
        for (const auto& c : sch.completed()) got[c.id] = c.tokens;
        for (int s = 0; s < n_prompts; s++) {
            std::vector<int64_t> want(ref[s].begin(), ref[s].begin() + (N - (s % 3) * 5));
            if (got[s] != want) { failures++; std::printf("FAIL graphs prompt%d: stream differs\n", s); }
        }
        std::printf("graphs: %d batch widths captured, %lld steps, tokens %s\n",
                    batch.graphs_captured(), (long long)sch.steps(),
                    failures ? "DIFFER" : "identical");
    }

    // (c) paged-specific: tiny budget -> preemption; shared prefix -> COW.
    {
        const llm::GemmPath gemm = llm::GemmPath::kMine;
        std::vector<std::vector<int64_t>> ref;
        for (const auto& p : prompts)
            ref.push_back(llm::greedy_decode_cached_gpu(gpu, p, N, kMaxSeq, gemm));

        // Budget: 3 blocks. Two prompts are admitted (1 block each, admission
        // needs 2 free), then growth past position 16 needs a third block for
        // each: the second one cannot get it and is preempted, later resumed.
        {
            llm::GpuBatch probe(gpu, 1, kMaxSeq, gemm);
            const int64_t budget = 3 * probe.bytes_per_block();
            llm::GpuBatch batch(gpu, 2, kMaxSeq, gemm, budget);
            llm::SchedulerConfig cfg;
            cfg.eos_id = -1;
            llm::Scheduler sch(batch, cfg);
            for (int s = 0; s < n_prompts; s++) {
                llm::Request r; r.id = s; r.prompt = prompts[s]; r.max_new = N;
                sch.submit(r);
            }
            sch.run_until_idle();
            std::map<int64_t, std::vector<int64_t>> got;
            for (const auto& c : sch.completed()) got[c.id] = c.tokens;
            for (int s = 0; s < n_prompts; s++)
                if (got[s] != ref[s]) { failures++; std::printf("FAIL budget prompt%d: stream differs\n", s); }
            std::printf("budget %d blocks: %d preemptions, %lld steps, blocks in use after: %d%s\n",
                        batch.blocks_total(), sch.preemptions(), (long long)sch.steps(),
                        batch.blocks_in_use(), sch.preemptions() > 0 ? "" : "  <-- expected preemptions");
            if (sch.preemptions() == 0) failures++;
            if (batch.blocks_in_use() != 0) { failures++; std::printf("FAIL: blocks leaked\n"); }
        }

        // Shared prefix: prompt = 32 tokens of prompt3 repeated, then each golden
        // prompt. The second and later requests must reuse the first's blocks.
        {
            std::vector<int64_t> prefix;
            while (prefix.size() < 32) prefix.insert(prefix.end(), prompts[3].begin(), prompts[3].end());
            prefix.resize(32);
            std::vector<std::vector<int64_t>> ps, refs;
            for (int s = 0; s < n_prompts; s++) {
                std::vector<int64_t> p = prefix;
                p.insert(p.end(), prompts[s].begin(), prompts[s].end());
                ps.push_back(p);
                refs.push_back(llm::greedy_decode_cached_gpu(gpu, p, N, kMaxSeq, gemm));
            }
            llm::GpuBatch batch(gpu, n_prompts, kMaxSeq, gemm, 0, /*prefix_cache=*/true);
            std::vector<std::vector<int64_t>> out(n_prompts);
            std::vector<llm::StepRow> rows;
            int64_t reused_total = 0;
            for (int s = 0; s < n_prompts; s++) {
                out[s].push_back(batch.prefill(s, ps[s]));
                reused_total += batch.last_prefill_reused();
                rows.push_back({s, out[s].back()});
            }
            for (int t = 1; t < N; t++) {
                std::vector<int64_t> next = batch.step(rows);
                for (int s = 0; s < n_prompts; s++) { out[s].push_back(next[s]); rows[s].token = next[s]; }
            }
            for (int s = 0; s < n_prompts; s++)
                if (out[s] != refs[s]) { failures++; std::printf("FAIL prefix prompt%d: stream differs\n", s); }
            // Prompts 1..4 share block 0 (positions 0-15) of prompt 0; block 1 holds
            // positions 16-31 and is the last full block only when T % 16 == 0.
            std::printf("shared prefix: %lld positions served from the prefix cache across %d prompts\n",
                        (long long)reused_total, n_prompts);
            if (reused_total < 16 * (n_prompts - 1)) { failures++; std::printf("FAIL: prefix cache unused\n"); }
        }
    }

    // (e) Stage 8: a long prompt (300 tokens) runs through the tiled-GEMM prefill
    // path (chunks of 256 rows, logits for the last row) and must match the
    // single-sequence session; with graphs on as well.
    {
        const llm::GemmPath gemm = llm::GemmPath::kMine;
        std::vector<int64_t> longp;
        while (longp.size() < 300) longp.insert(longp.end(), prompts[3].begin(), prompts[3].end());
        longp.resize(300);
        const std::vector<int64_t> ref = llm::greedy_decode_cached_gpu(gpu, longp, 16, 400, gemm);
        for (bool graphs : {false, true}) {
            llm::GpuBatch batch(gpu, 1, 400, gemm, 0, false, graphs, /*attn_split=*/true);
            std::vector<int64_t> out{batch.prefill(0, longp)};
            for (int t = 1; t < 16; t++) out.push_back(batch.step({{0, out.back()}})[0]);
            if (out != ref) { failures++; std::printf("FAIL long prompt (graphs %d): stream differs\n", int(graphs)); }
            else std::printf("long prompt 300 tokens (graphs %s): identical (16 tokens)\n", graphs ? "on" : "off");
        }
    }

    if (failures == 0) { std::printf("test_batch_gpu: batched == single on all prompts\n"); return 0; }
    std::printf("test_batch_gpu: %d FAILURES\n", failures);
    return 1;
}
