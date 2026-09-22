// main_serve_bench.cpp — throughput vs latency across batch sizes (Stage 6).
//
// A fixed set of requests (the golden prompts, cycled, with a spread of
// max_new lengths so requests finish at different times) is pushed through
// the scheduler once per slot count. Reported per configuration:
//   throughput      generated tokens / s over the whole run
//   step time       mean ms per scheduler step (one batched decode)
//   token latency   mean ms between a request's consecutive tokens
//   request latency p50 / p95 ms from submission (t=0) to completion,
//                   which includes time spent waiting in the queue
//   TTFT            p50 ms from submission to first token
// Continuous vs static batching is the second axis. All requests arrive at
// t=0 (closed-loop); eos is disabled so lengths are deterministic.
//
// Usage: main_serve_bench [key=value ...]
//   n=64            requests            gemm=mine|cublas
//   slots=1,2,4,8,16,32                 max_new=48 (spread max_new/2 .. 3*max_new/2)
//   max_seq=192     per-sequence cap    budget_mb=0 (0 = reserve slots x max_seq)
//   prefix=0        shared prompt prefix length prepended to every request
//   prefix_cache=1  reuse the shared prefix's KV blocks across requests
//   modes=both|continuous|static
//   graphs=0|1      capture the batched step into a CUDA graph per batch width (gemm=mine)
//   attn=split|par  flash-decoding (split positions across blocks) or one block per (row, head)
//   events=<path>   write the LAST configuration's scheduler event log (JSON lines)
//                   for viz/index.html (default: bench/events_<model>_<gemm>_slots<N>.jsonl
//                   for the largest continuous run)
// Writes bench/stage6_serve_<model>_<gemm>_<timestamp>.json.

#include "../kernels/batch_gpu.h"
#include "../kernels/model_gpu.h"
#include "../kernels/spec_engine.h"
#include "model.h"
#include "model_select.h"
#include "npy.h"
#include "scheduler.h"

#include <algorithm>
#include <map>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <fstream>
#include <sstream>
#include <memory>
#include <string>
#include <vector>

namespace {

std::vector<int> parse_list(const std::string& s) {
    std::vector<int> v;
    std::stringstream ss(s);
    std::string item;
    while (std::getline(ss, item, ',')) v.push_back(std::atoi(item.c_str()));
    return v;
}

double percentile(std::vector<double> v, double p) {
    if (v.empty()) return 0.0;
    std::sort(v.begin(), v.end());
    const size_t i = size_t(p * double(v.size() - 1) + 0.5);
    return v[std::min(i, v.size() - 1)];
}

struct Row {
    int slots; bool continuous; double wall_ms, tok_s, step_ms, token_lat_ms,
    req_p50, req_p95, ttft_p50; int64_t steps, tokens; int blocks_total, blocks_peak, preemptions;
    double acceptance;   // speculative rounds: accepted / drafted (0 when not speculating)
};

} // namespace

int main(int argc, char** argv) {
    const std::string root = MODEL_ROOT;
    std::map<std::string, std::string> kv;
    for (int i = 1; i < argc; i++) {
        const std::string a = argv[i];
        const size_t eq = a.find('=');
        if (eq == std::string::npos) { std::fprintf(stderr, "expected key=value, got %s\n", argv[i]); return 2; }
        kv[a.substr(0, eq)] = a.substr(eq + 1);
    }
    auto get = [&](const char* k, const std::string& d) { return kv.count(k) ? kv[k] : d; };
    const int n_requests = std::atoi(get("n", "64").c_str());
    const std::string gemm_s = get("gemm", "mine");
    const std::vector<int> slot_list = parse_list(get("slots", "1,2,4,8,16,32"));
    const int max_new_base = std::atoi(get("max_new", "48").c_str());
    const int64_t max_seq = std::atoll(get("max_seq", "192").c_str());
    const int64_t budget = std::atoll(get("budget_mb", "0").c_str()) << 20;
    const int prefix_len = std::atoi(get("prefix", "0").c_str());
    const bool prefix_cache = get("prefix_cache", "1") != "0";
    const std::string modes = get("modes", "both");
    const std::string events_path = get("events", "");
    const bool graphs = get("graphs", "0") == "1";
    const bool attn_split = get("attn", "split") != "par";
    const llm::GemmPath gemm = llm::gemm_path_from(gemm_s);
    // Stage 11: LLM_DRAFT=<model> plus spec_k=N wrap the engine in speculative decoding.
    const int spec_k = std::atoi(get("spec_k", "4").c_str());
    const std::string draft = llm::draft_name();

    // Stage 10: LLM_QUANT=int8|int4 loads model.q8/q4.llmq instead of the bf16 weights.
    llm::Model model;
    llm::QuantModel qmodel;
    const std::string quant = llm::quant_name();
    if (quant.empty()) model.load(llm::model_dir(root));
    else qmodel.load(llm::model_dir(root), quant == "int4" ? llm::QKind::kInt4 : llm::QKind::kInt8);
    std::unique_ptr<llm::GpuModel> gpu_p = quant.empty() ? std::make_unique<llm::GpuModel>(model)
                                                         : std::make_unique<llm::GpuModel>(qmodel);
    llm::GpuModel& gpu = *gpu_p;
    llm::Model dmodel;
    llm::QuantModel dqmodel;
    std::unique_ptr<llm::GpuModel> dgpu;
    const std::string dquant = llm::draft_quant_name();
    if (!draft.empty() && dquant.empty()) { dmodel.load(root + "/models/" + draft); dgpu = std::make_unique<llm::GpuModel>(dmodel); }
    if (!draft.empty() && !dquant.empty()) {
        dqmodel.load(root + "/models/" + draft, dquant == "int4" ? llm::QKind::kInt4 : llm::QKind::kInt8);
        dgpu = std::make_unique<llm::GpuModel>(dqmodel);
    }
    const std::string golden = llm::golden_dir(root);
    std::vector<std::vector<int64_t>> prompts;
    for (int pi = 0; pi < 5; pi++) {
        auto g = llm::npy::load_npz(golden + "/prompt" + std::to_string(pi) + "_bf16.npz");
        const auto& ia = g.at("token_ids");
        prompts.emplace_back(ia.i64(), ia.i64() + ia.numel());
    }
    // Optional shared prefix (a "system prompt"): prompt3 repeated to length.
    std::vector<int64_t> prefix;
    while (int(prefix.size()) < prefix_len) prefix.insert(prefix.end(), prompts[3].begin(), prompts[3].end());
    prefix.resize(size_t(prefix_len));
    // Request set: prompts cycled, max_new spread over base/2 .. base*3/2.
    std::vector<llm::Request> requests;
    int64_t total_tokens = 0;
    for (int i = 0; i < n_requests; i++) {
        llm::Request r;
        r.id = i;
        r.prompt = prefix;
        const auto& body = prompts[size_t(i) % prompts.size()];
        r.prompt.insert(r.prompt.end(), body.begin(), body.end());
        r.max_new = max_new_base / 2 + (i * 7) % (max_new_base + 1);
        total_tokens += r.max_new;
        requests.push_back(r);
    }
    std::printf("model %s gemm=%s: %d requests, %lld tokens to generate, max_seq %lld, "
                "budget %s, prefix %d (cache %s), graphs %s, attn %s, draft %s k=%d\n",
                llm::model_name().c_str(), gemm_s.c_str(), n_requests, (long long)total_tokens,
                (long long)max_seq, budget ? (std::to_string(budget >> 20) + " MB").c_str() : "reserve",
                prefix_len, prefix_cache ? "on" : "off", graphs ? "on" : "off",
                attn_split ? "split" : "par", draft.empty() ? "none" : draft.c_str(), spec_k);
    std::printf("%5s %10s %9s %8s %10s %9s %9s %8s %6s %7s %7s %7s\n", "slots", "mode", "tok/s",
                "step ms", "tok lat ms", "req p50", "req p95", "ttft p50", "steps", "blocks",
                "peak", "preempt");

    std::vector<Row> rows;
    for (int slots : slot_list) {
        for (bool continuous : {true, false}) {
            if (modes == "continuous" && !continuous) continue;
            if (modes == "static" && continuous) continue;
            llm::GpuBatch batch(gpu, slots, max_seq, gemm, budget, prefix_cache, graphs, attn_split);
            std::unique_ptr<llm::GpuBatch> dbatch;
            std::unique_ptr<llm::SpecEngine> spec;
            if (dgpu) {
                dbatch = std::make_unique<llm::GpuBatch>(*dgpu, slots, max_seq, gemm, 0, prefix_cache, graphs, attn_split);
                const int64_t v = std::min(gpu.cfg().vocab_size, dgpu->cfg().vocab_size);
                batch.set_vocab_limit(v); dbatch->set_vocab_limit(v);
                spec = std::make_unique<llm::SpecEngine>(batch, *dbatch, spec_k);
            }
            llm::BatchEngine& eng = spec ? static_cast<llm::BatchEngine&>(*spec) : batch;
            // Warm-up: one short request so first-launch costs stay out of the timing.
            { llm::Request w = requests[0]; w.max_new = 4; llm::SchedulerConfig c; llm::Scheduler s(eng, c); s.submit(w); s.run_until_idle(); }
            llm::SchedulerConfig cfg;
            cfg.continuous = continuous;
            cfg.eos_id = -1;
            llm::Scheduler sch(eng, cfg);
            for (const auto& r : requests) sch.submit(r);
            const auto t0 = std::chrono::steady_clock::now();
            sch.run_until_idle();
            const double wall_ms = std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - t0).count();

            std::vector<double> req_lat, ttft, tok_lat;
            int64_t tokens = 0;
            for (const auto& c : sch.completed()) {
                req_lat.push_back(c.finished_ms);   // submitted at t=0
                ttft.push_back(c.first_token_ms);
                if (c.tokens.size() > 1)
                    tok_lat.push_back((c.finished_ms - c.first_token_ms) / double(c.tokens.size() - 1));
                tokens += int64_t(c.tokens.size());
            }
            double tl = 0; for (double x : tok_lat) tl += x; tl /= double(std::max<size_t>(1, tok_lat.size()));
            int peak = 0;
            for (size_t i = 0; i < sch.events().size(); i++)
                peak = std::max(peak, sch.events().at(i).pool_in_use);
            Row row{slots, continuous, wall_ms, tokens / (wall_ms / 1e3), wall_ms / double(sch.steps()),
                    tl, percentile(req_lat, 0.5), percentile(req_lat, 0.95), percentile(ttft, 0.5),
                    sch.steps(), tokens, batch.blocks_total(), peak, sch.preemptions(),
                    spec ? spec->stats().acceptance() : 0.0};
            rows.push_back(row);
            std::printf("%5d %10s %9.1f %8.2f %10.2f %9.0f %9.0f %8.0f %6lld %7d %7d %7d", slots,
                        continuous ? "continuous" : "static", row.tok_s, row.step_ms, row.token_lat_ms,
                        row.req_p50, row.req_p95, row.ttft_p50, (long long)row.steps, row.blocks_total,
                        row.blocks_peak, row.preemptions);
            if (spec) std::printf("  accept %.0f%% %.2f tok/round", 100 * row.acceptance, spec->stats().tokens_per_round());
            std::printf("\n");
            if (slots == slot_list.back() && (continuous || modes == "static"))
                sch.events().write_jsonl(!events_path.empty() ? events_path
                    : root + "/bench/events_" + llm::model_name() + "_" + gemm_s + "_slots" +
                          std::to_string(slots) + ".jsonl");
        }
    }

    char ts[32];
    std::time_t now = std::time(nullptr);
    std::strftime(ts, sizeof ts, "%Y%m%d-%H%M%S", std::localtime(&now));
    const std::string tag = get("tag", "");
    const std::string path = root + "/bench/serve_" + llm::model_name() + "_" + gemm_s + (tag.empty() ? "" : "_" + tag) + "_" + ts + ".json";
    std::ofstream f(path);
    f << "{\n \"what\": \"Stage 6 serving sweep: throughput vs latency over batch size, continuous vs static batching\",\n"
      << " \"model\": \"" << llm::model_name() << "\", \"gemm\": \"" << gemm_s << "\",\n"
      << " \"requests\": " << n_requests << ", \"tokens_generated\": " << total_tokens
      << ", \"max_new_base\": " << max_new_base << ", \"max_seq\": " << max_seq
      << ", \"budget_mb\": " << (budget >> 20) << ", \"bytes_per_block\": " << llm::GpuBatch(gpu, 1, 16, gemm).bytes_per_block()
      << ", \"prefix\": " << prefix_len << ", \"prefix_cache\": " << (prefix_cache ? "true" : "false")
      << ", \"graphs\": " << (graphs ? "true" : "false") << ", \"attn\": \"" << (attn_split ? "split" : "par") << "\",\n"
      << " \"draft\": \"" << draft << "\", \"draft_quant\": \"" << dquant << "\", \"spec_k\": " << (draft.empty() ? 0 : spec_k) << ",\n"
      << " \"prompts\": \"golden prompts 0-4 cycled, all submitted at t=0, eos disabled\",\n \"rows\": [\n";
    for (size_t i = 0; i < rows.size(); i++) {
        const Row& r = rows[i];
        f << "  {\"slots\": " << r.slots << ", \"mode\": \"" << (r.continuous ? "continuous" : "static")
          << "\", \"tok_s\": " << r.tok_s << ", \"wall_ms\": " << r.wall_ms << ", \"step_ms\": " << r.step_ms
          << ", \"token_latency_ms\": " << r.token_lat_ms << ", \"request_p50_ms\": " << r.req_p50
          << ", \"request_p95_ms\": " << r.req_p95 << ", \"ttft_p50_ms\": " << r.ttft_p50
          << ", \"steps\": " << r.steps << ", \"blocks_total\": " << r.blocks_total
          << ", \"blocks_peak\": " << r.blocks_peak << ", \"preemptions\": " << r.preemptions
          << ", \"acceptance\": " << r.acceptance << "}" << (i + 1 < rows.size() ? ",\n" : "\n");
    }
    f << " ]\n}\n";
    std::printf("wrote %s\n", path.c_str());
    return 0;
}
