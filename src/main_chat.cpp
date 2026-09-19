// main_chat.cpp — talk to the engine.
//
// Reads a line from stdin, wraps the running conversation in Qwen's chat
// template, prefills the KV-cache session, then samples token by token,
// streaming the detokenized reply until <|im_end|> or a length cap. Keeps the
// full conversation so turns have memory (each turn re-prefills the history;
// cross-turn cache reuse is a later optimization).
//
// Usage: main_chat [device=cpu|gpu] [gemm=mine|cublas]
//                  [--temp T] [--top-k K] [--top-p P] [--seed S] [--max N]

#include "forward.h"
#include "model_select.h"
#include "model.h"
#include "session.h"
#include "sampler.h"
#include "tokenizer.h"

#ifdef HAVE_GPU
#include "../kernels/model_gpu.h"
#include "../kernels/session_gpu.h"
#endif

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

namespace {

// Print the longest UTF-8-complete prefix of `pending`, keep the incomplete
// tail (byte-level BPE tokens can end mid-codepoint).
void flush_utf8(std::string& pending) {
    size_t safe = pending.size();
    // Walk back over trailing continuation bytes to find the last lead byte.
    size_t i = pending.size();
    while (i > 0 && (static_cast<unsigned char>(pending[i - 1]) & 0xC0) == 0x80) i--;
    if (i > 0) {
        unsigned char lead = pending[i - 1];
        size_t need = lead < 0x80 ? 1 : lead < 0xE0 ? 2 : lead < 0xF0 ? 3 : 4;
        size_t have = pending.size() - (i - 1);
        if (have < need) safe = i - 1;   // last codepoint incomplete: hold it
    }
    std::fwrite(pending.data(), 1, safe, stdout);
    std::fflush(stdout);
    pending.erase(0, safe);
}

}  // namespace

int main(int argc, char** argv) {
    const std::string root = MODEL_ROOT;
    std::string device = "gpu", gemm = "cublas";
    float temp = 0.7f, top_p = 0.9f;
    int top_k = 40, max_new = 256;
    uint64_t seed = 0;

    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        auto next = [&]() { return std::string(i + 1 < argc ? argv[++i] : ""); };
        if (a == "cpu" || a == "gpu") device = a;
        else if (a == "mine" || a == "cublas" || a == "naive") gemm = a;
        else if (a == "--temp") temp = std::stof(next());
        else if (a == "--top-k") top_k = std::stoi(next());
        else if (a == "--top-p") top_p = std::stof(next());
        else if (a == "--seed") seed = std::stoull(next());
        else if (a == "--max") max_new = std::stoi(next());
    }

#ifndef HAVE_GPU
    if (device == "gpu") { std::fprintf(stderr, "no GPU in this build; using cpu\n"); device = "cpu"; }
#endif

    llm::Model model;
    model.load(llm::model_dir(root));
    llm::Tokenizer tok;
    tok.load(llm::model_dir(root) + "/tokenizer.json");

#ifdef HAVE_GPU
    std::unique_ptr<llm::GpuModel> gpu;
    llm::GemmPath path = llm::gemm_path_from(gemm);
    if (device == "gpu") gpu = std::make_unique<llm::GpuModel>(model);
#endif

    const std::string system =
        "You are Qwen, created by Alibaba Cloud. You are a helpful assistant.";
    std::string convo = "<|im_start|>system\n" + system + "<|im_end|>\n";
    const int64_t eos = tok.eos_id();

    std::fprintf(stderr, "chat ready (device=%s gemm=%s temp=%.2f). Empty line to quit.\n",
                 device.c_str(), gemm.c_str(), temp);

    std::string line;
    while (true) {
        std::fprintf(stderr, "\n> ");
        if (!std::getline(std::cin, line) || line.empty()) break;

        convo += "<|im_start|>user\n" + line + "<|im_end|>\n<|im_start|>assistant\n";
        std::vector<int64_t> ids = tok.encode(convo);
        const int64_t max_seq = int64_t(ids.size()) + max_new + 8;

        llm::Sampler sampler(seed);
        sampler.temperature = temp;
        sampler.top_k = top_k;
        sampler.top_p = top_p;

        std::string reply, pending;
        auto step = [&](std::vector<float>& logits, auto&& decode_next) {
            for (int n = 0; n < max_new; n++) {
                int64_t id = sampler.sample(logits);
                if (id == eos) break;
                std::string bytes = tok.id_to_bytes(id);
                reply += bytes;
                pending += bytes;
                flush_utf8(pending);
                logits = decode_next(id);
            }
        };

        if (device == "cpu") {
            llm::Session s(model, max_seq);
            std::vector<float> logits = s.prefill(ids);
            step(logits, [&](int64_t id) { return s.decode_one(id); });
        }
#ifdef HAVE_GPU
        else {
            llm::GpuSession s(*gpu, max_seq, path);
            std::vector<float> logits = s.prefill(ids);
            step(logits, [&](int64_t id) { return s.decode_one(id); });
        }
#endif
        std::fwrite(pending.data(), 1, pending.size(), stdout);  // final flush
        std::fflush(stdout);
        convo += reply + "<|im_end|>\n";
    }
    return 0;
}
