// main_server.cpp — HTTP front end (Stage 9): an OpenAI-compatible subset over
// the serving core in src/serve.h.
//
//   POST /v1/chat/completions   {"messages":[{"role","content"}...], "max_tokens",
//                                "temperature", "seed", "stream"}
//   POST /v1/completions        {"prompt", ...same fields}
//   GET  /metrics               JSON: queue depth, active, tokens/s, TTFT and
//                               latency percentiles, KV blocks, preemptions,
//                               cancellations
//   GET  /health
//
// stream=true answers with server-sent events, one "data:" chunk per token
// (UTF-8-safe: bytes that do not yet form a whole code point are held back,
// as in main_chat). A client that disconnects mid-stream is detected when the
// next chunk fails to write; the request is cancelled and its KV blocks come
// back on the scheduler's next step.
//
// One process, one GPU, one engine thread; HTTP handlers run on httplib's
// thread pool and only touch the serving core through request handles.
//
// Usage: main_server [--port 8080] [--slots 16] [--max-seq 2048] [--kv-budget-mb 0]
//                    [--gemm mine|cublas] [--graphs 0|1] [--model <name>]
//                    [--draft <name>] [--spec-k 4]     (Stage 11: speculative decoding)

#include "../kernels/batch_gpu.h"
#include "../kernels/model_gpu.h"
#include "../kernels/spec_engine.h"
#include "model.h"
#include "model_select.h"
#include "serve.h"
#include "tokenizer.h"

#include "../third_party/httplib/httplib.h"
#include "../third_party/nlohmann/json.hpp"

#include <algorithm>
#include <atomic>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <string>

using json = nlohmann::json;

namespace {

std::atomic<httplib::Server*> g_server{nullptr};
void on_signal(int) { if (auto* s = g_server.load()) s->stop(); }

// Hold back an incomplete trailing UTF-8 sequence so every chunk is valid text.
std::string flush_utf8(std::string& pending) {
    size_t keep = 0;
    for (size_t i = pending.size(); i > 0 && keep < 4; i--) {
        const unsigned char c = (unsigned char)pending[i - 1];
        if ((c & 0xC0) != 0x80) {   // lead byte or ASCII
            const size_t need = c < 0x80 ? 1 : c < 0xE0 ? 2 : c < 0xF0 ? 3 : 4;
            if (pending.size() - (i - 1) < need) keep = pending.size() - (i - 1);
            break;
        }
        keep++;
    }
    if (keep >= pending.size()) return "";
    std::string out = pending.substr(0, pending.size() - keep);
    pending.erase(0, out.size());
    return out;
}

std::string reason_name(llm::StopReason r) {
    switch (r) {
        case llm::StopReason::kEos: return "stop";
        case llm::StopReason::kMaxNew: return "length";
        case llm::StopReason::kCacheFull: return "length";
        case llm::StopReason::kCancelled: return "cancelled";
    }
    return "stop";
}

} // namespace

int main(int argc, char** argv) {
    int port = 8080, slots = 16;
    int64_t max_seq = 2048, budget_mb = 0;
    std::string gemm = "mine";
    bool graphs = true;
    int spec_k = 4;
    for (int i = 1; i + 1 < argc; i += 2) {
        const std::string k = argv[i], v = argv[i + 1];
        if (k == "--port") port = std::atoi(v.c_str());
        else if (k == "--slots") slots = std::atoi(v.c_str());
        else if (k == "--max-seq") max_seq = std::atoll(v.c_str());
        else if (k == "--kv-budget-mb") budget_mb = std::atoll(v.c_str());
        else if (k == "--gemm") gemm = v;
        else if (k == "--graphs") graphs = v != "0";
        else if (k == "--model") setenv("LLM_MODEL", v.c_str(), 1);
        else if (k == "--draft") setenv("LLM_DRAFT", v.c_str(), 1);
        else if (k == "--spec-k") spec_k = std::atoi(v.c_str());
        else { std::fprintf(stderr, "unknown flag %s\n", k.c_str()); return 2; }
    }
    const std::string root = MODEL_ROOT;
    // Stage 10: LLM_QUANT=int8|int4 loads model.q8/q4.llmq instead of the bf16 weights.
    llm::Model model;
    llm::QuantModel qmodel;
    const std::string quant = llm::quant_name();
    if (quant.empty()) model.load(llm::model_dir(root));
    else qmodel.load(llm::model_dir(root), quant == "int4" ? llm::QKind::kInt4 : llm::QKind::kInt8);
    llm::Tokenizer tok;
    tok.load(llm::model_dir(root) + "/tokenizer.json");
    std::unique_ptr<llm::GpuModel> gpu_p = quant.empty() ? std::make_unique<llm::GpuModel>(model)
                                                         : std::make_unique<llm::GpuModel>(qmodel);
    llm::GpuModel& gpu = *gpu_p;
    llm::GpuBatch engine(gpu, slots, max_seq, llm::gemm_path_from(gemm), budget_mb << 20, true, graphs);
    // Stage 11: --draft <model> (or LLM_DRAFT) wraps the engine in speculative decoding.
    llm::Model dmodel;
    llm::QuantModel dqmodel;
    std::unique_ptr<llm::GpuModel> dgpu;
    std::unique_ptr<llm::GpuBatch> dbatch;
    std::unique_ptr<llm::SpecEngine> spec;
    const std::string draft = llm::draft_name();
    if (!draft.empty()) {
        const std::string dquant = llm::draft_quant_name();
        if (dquant.empty()) { dmodel.load(root + "/models/" + draft); dgpu = std::make_unique<llm::GpuModel>(dmodel); }
        else {
            dqmodel.load(root + "/models/" + draft, dquant == "int4" ? llm::QKind::kInt4 : llm::QKind::kInt8);
            dgpu = std::make_unique<llm::GpuModel>(dqmodel);
        }
        dbatch = std::make_unique<llm::GpuBatch>(*dgpu, slots, max_seq, llm::gemm_path_from(gemm), 0, true, graphs);
        const int64_t v = std::min(gpu.cfg().vocab_size, dgpu->cfg().vocab_size);
        engine.set_vocab_limit(v); dbatch->set_vocab_limit(v);
        spec = std::make_unique<llm::SpecEngine>(engine, *dbatch, spec_k);
    }
    llm::BatchEngine& eng = spec ? static_cast<llm::BatchEngine&>(*spec) : engine;
    llm::SchedulerConfig cfg;
    cfg.eos_id = tok.eos_id();
    llm::Server core(eng, cfg);
    std::fprintf(stderr, "model %s, %d slots, max_seq %lld, %d KV blocks, port %d, draft %s k=%d\n",
                 llm::model_name().c_str(), slots, (long long)max_seq, engine.blocks_total(), port,
                 draft.empty() ? "none" : draft.c_str(), spec_k);

    httplib::Server http;
    g_server = &http;
    std::signal(SIGINT, on_signal);
    std::signal(SIGTERM, on_signal);

    http.Get("/health", [](const httplib::Request&, httplib::Response& res) {
        res.set_content("{\"status\":\"ok\"}", "application/json");
    });
    http.Get("/metrics", [&](const httplib::Request&, httplib::Response& res) {
        const llm::Metrics m = core.metrics();
        json j{{"queued", m.queued}, {"active", m.active}, {"tokens_per_s", m.tokens_per_s},
               {"ttft_p50_ms", m.ttft_p50_ms}, {"ttft_p95_ms", m.ttft_p95_ms},
               {"latency_p50_ms", m.latency_p50_ms}, {"latency_p95_ms", m.latency_p95_ms},
               {"requests_done", m.requests_done}, {"tokens_total", m.tokens_total},
               {"preemptions", m.preemptions}, {"cancellations", m.cancellations},
               {"kv_blocks_in_use", m.blocks_in_use}, {"kv_blocks_total", m.blocks_total},
               {"steps", m.steps}};
        res.set_content(j.dump(), "application/json");
    });

    // Shared handler: builds the prompt ids, submits, streams or collects.
    auto handle = [&](const httplib::Request& req, httplib::Response& res, bool chat) {
        json body;
        try { body = json::parse(req.body); } catch (...) {
            res.status = 400; res.set_content("{\"error\":\"bad json\"}", "application/json"); return;
        }
        std::vector<int64_t> ids;
        if (chat) {
            std::string system = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant.";
            std::string convo;
            for (const auto& m : body.value("messages", json::array())) {
                const std::string role = m.value("role", "user"), content = m.value("content", "");
                if (role == "system") { system = content; continue; }
                convo += "<|im_start|>" + role + "\n" + content + "<|im_end|>\n";
            }
            ids = tok.encode("<|im_start|>system\n" + system + "<|im_end|>\n" + convo + "<|im_start|>assistant\n");
        } else {
            ids = tok.encode(body.value("prompt", ""));
        }
        llm::Request r;
        r.prompt = ids;
        r.max_new = body.value("max_tokens", 128);
        r.sample.temperature = body.value("temperature", 0.0f);
        r.sample.seed = body.value("seed", uint64_t(0));
        if (ids.empty() || int64_t(ids.size()) + r.max_new > max_seq) {
            res.status = 400;
            res.set_content("{\"error\":\"prompt + max_tokens exceeds max_seq\"}", "application/json");
            return;
        }
        auto h = core.submit(std::move(r));
        const std::string id = "req-" + std::to_string(h->id());
        const std::string object = chat ? "chat.completion" : "text_completion";
        const bool stream = body.value("stream", false);

        if (!stream) {
            std::string text;
            int64_t t;
            int n = 0;
            while (h->next(t)) { if (t != tok.eos_id()) text += tok.id_to_bytes(t); n++; }
            json choice = chat ? json{{"index", 0}, {"message", {{"role", "assistant"}, {"content", text}}},
                                      {"finish_reason", reason_name(h->reason())}}
                               : json{{"index", 0}, {"text", text}, {"finish_reason", reason_name(h->reason())}};
            json j{{"id", id}, {"object", object}, {"choices", json::array({choice})},
                   {"usage", {{"prompt_tokens", ids.size()}, {"completion_tokens", n}}}};
            res.set_content(j.dump(), "application/json");
            return;
        }

        // SSE: one event per token, [DONE] at the end. A failed write means
        // the client went away: cancel and stop.
        auto pending = std::make_shared<std::string>();
        auto finished = std::make_shared<bool>(false);
        res.set_chunked_content_provider("text/event-stream",
            [&, h, id, object, chat, pending, finished](size_t, httplib::DataSink& sink) {
                if (*finished) { sink.done(); return true; }
                int64_t t;
                if (!h->next(t)) {
                    *finished = true;
                    std::string tail = *pending;
                    json delta = chat ? json{{"role", "assistant"}, {"content", tail}} : json{{"text", tail}};
                    json j{{"id", id}, {"object", object + ".chunk"},
                           {"choices", json::array({{{"index", 0}, {"delta", delta},
                                                     {"finish_reason", reason_name(h->reason())}}})}};
                    const std::string ev = "data: " + j.dump() + "\n\ndata: [DONE]\n\n";
                    if (!sink.write(ev.data(), ev.size())) return false;
                    sink.done();
                    return true;
                }
                if (t != tok.eos_id()) *pending += tok.id_to_bytes(t);
                const std::string text = flush_utf8(*pending);
                if (text.empty()) return true;   // wait for a complete code point
                json delta = chat ? json{{"content", text}} : json{{"text", text}};
                json j{{"id", id}, {"object", object + ".chunk"},
                       {"choices", json::array({{{"index", 0}, {"delta", delta}, {"finish_reason", nullptr}}})}};
                const std::string ev = "data: " + j.dump() + "\n\n";
                if (!sink.write(ev.data(), ev.size())) { h->cancel(); return false; }
                return true;
            },
            [h](bool success) { if (!success) h->cancel(); });
    };
    http.Post("/v1/chat/completions", [&](const httplib::Request& req, httplib::Response& res) { handle(req, res, true); });
    http.Post("/v1/completions", [&](const httplib::Request& req, httplib::Response& res) { handle(req, res, false); });

    std::fprintf(stderr, "listening on http://0.0.0.0:%d\n", port);
    http.listen("0.0.0.0", port);
    std::fprintf(stderr, "shutting down\n");
    core.shutdown();
    return 0;
}
