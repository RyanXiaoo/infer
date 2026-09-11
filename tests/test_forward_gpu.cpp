// test_forward_gpu.cpp — the Stage 2 validation ladder, pointed at the GPU
// forward pass, run once per GEMM path (mine, then cuBLAS).
//
// Identical discipline to test_forward.cpp: taps compare in execution order
// against BOTH goldens (fp32 = same-arithmetic reference, gate at cosine >=
// 0.999; bf16 printed as drift context on failure), first failing rung aborts
// the prompt, argmax equality at every position, then greedy continuation.
// A failure on kMine that vanishes on kCublas isolates the bug to my GEMV —
// that's the flag's whole purpose.

#include "../kernels/model_gpu.h"
#include "../src/model.h"
#include "../src/npy.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <map>
#include <string>
#include <vector>

#include "../third_party/nlohmann/json.hpp"

using llm::npy::Array;

static int failures = 0;

struct Cmp {
    double max_abs = 0.0;
    double cosine = 1.0;
};

static Cmp compare(const float* a, const float* b, int64_t n) {
    Cmp c;
    double dot = 0, na = 0, nb = 0;
    for (int64_t i = 0; i < n; i++) {
        double d = std::abs(double(a[i]) - double(b[i]));
        if (d > c.max_abs) c.max_abs = d;
        dot += double(a[i]) * double(b[i]);
        na += double(a[i]) * double(a[i]);
        nb += double(b[i]) * double(b[i]);
    }
    c.cosine = (na > 0 && nb > 0) ? dot / (std::sqrt(na) * std::sqrt(nb)) : 1.0;
    return c;
}

static bool check_tap(const std::string& tag, const std::string& name,
                      const std::vector<float>& ours,
                      const std::map<std::string, Array>& g32,
                      const std::map<std::string, Array>& g16) {
    auto it32 = g32.find(name);
    if (it32 == g32.end()) {
        std::printf("FAIL %s/%s: tap missing from golden npz\n", tag.c_str(), name.c_str());
        failures++;
        return false;
    }
    const Array& ref32 = it32->second;
    if (int64_t(ours.size()) != ref32.numel()) {
        std::printf("FAIL %s/%s: numel %zu vs golden %lld\n", tag.c_str(), name.c_str(),
                    ours.size(), (long long)ref32.numel());
        failures++;
        return false;
    }
    Cmp c32 = compare(ours.data(), ref32.f32(), ref32.numel());
    if (c32.cosine >= 0.999) return true;

    Cmp c16{};
    auto it16 = g16.find(name);
    if (it16 != g16.end()) c16 = compare(ours.data(), it16->second.f32(), ref32.numel());
    std::printf("FAIL %s/%s: cosine vs fp32 %.6f (max_abs %.4g) | vs bf16 %.6f "
                "(max_abs %.4g)\n", tag.c_str(), name.c_str(), c32.cosine, c32.max_abs,
                c16.cosine, c16.max_abs);
    failures++;
    return false;
}

int main() {
    const std::string root = MODEL_ROOT;
    const std::string golden = root + "/tests/golden";

    std::ifstream mf(golden + "/manifest.json");
    if (!mf) {
        std::printf("SKIP: goldens not found\n");
        return 0;
    }
    nlohmann::json manifest = nlohmann::json::parse(mf);
    int n_prompts = int(manifest["prompts"].size());
    const bool smoke = std::getenv("FORWARD_SMOKE") != nullptr;
    if (smoke) n_prompts = 1;

    llm::Model model;
    model.load(root + "/models/Qwen2.5-0.5B-Instruct");
    llm::GpuModel gpu(model);

    nlohmann::json greedy;
    {
        std::ifstream gf(golden + "/greedy.json");
        if (gf) greedy = nlohmann::json::parse(gf);
    }

    for (llm::GemmPath gemm : {llm::GemmPath::kMine, llm::GemmPath::kCublas}) {
        const std::string gname = gemm == llm::GemmPath::kMine ? "mine" : "cublas";

        for (int pi = 0; pi < n_prompts; pi++) {
            const std::string tag = "prompt" + std::to_string(pi) + "/" + gname;
            auto g32 = llm::npy::load_npz(golden + "/prompt" + std::to_string(pi) + "_fp32.npz");
            auto g16 = llm::npy::load_npz(golden + "/prompt" + std::to_string(pi) + "_bf16.npz");

            const Array& ids_arr = g16.at("token_ids");
            std::vector<int64_t> ids(ids_arr.i64(), ids_arr.i64() + ids_arr.numel());
            const int64_t T = int64_t(ids.size());

            std::map<std::string, std::vector<float>> taps;
            llm::TapFn hook = [&](const std::string& name, const float* data,
                                  const std::vector<int64_t>& shape) {
                int64_t n = 1;
                for (int64_t d : shape) n *= d;
                taps[name] = std::vector<float>(data, data + n);
            };
            std::vector<float> logits = llm::forward_gpu(gpu, ids, &hook, false, gemm);

            std::vector<std::string> ladder = {"hidden_state_0"};
            for (int li : {0, 12}) {
                const std::string p = "layer" + std::to_string(li) + ".";
                for (const char* op : {"post_input_layernorm", "q_proj", "k_proj", "v_proj"})
                    ladder.push_back(p + op);
                if (li == 0) { ladder.push_back("rope_cos"); ladder.push_back("rope_sin"); }
                for (const char* op : {"q_post_rope", "k_post_rope", "attn_out_pre_o_proj",
                                       "post_attn_residual", "post_post_attention_layernorm",
                                       "mlp_gate", "mlp_up", "mlp_down"})
                    ladder.push_back(p + op);
                ladder.push_back("hidden_state_" + std::to_string(li + 1));
            }
            for (int64_t l = 2; l <= model.cfg.num_hidden_layers; l++)
                if (l != 13) ladder.push_back("hidden_state_" + std::to_string(l));
            ladder.push_back("logits");

            bool ok = true;
            for (const auto& name : ladder)
                if (!check_tap(tag, name, taps.at(name), g32, g16)) { ok = false; break; }
            if (!ok) continue;

            const Array& glog = g32.at("logits");
            const int64_t V = model.cfg.vocab_size;
            for (int64_t t = 0; t < T; t++) {
                const float* ours = logits.data() + t * V;
                const float* ref = glog.f32() + t * V;
                int64_t oa = 0, ra = 0;
                for (int64_t v = 1; v < V; v++) {
                    if (ours[v] > ours[oa]) oa = v;
                    if (ref[v] > ref[ra]) ra = v;
                }
                if (oa != ra) {
                    std::printf("FAIL %s: argmax mismatch at pos %lld: %lld vs %lld\n",
                                tag.c_str(), (long long)t, (long long)oa, (long long)ra);
                    failures++;
                    ok = false;
                }
            }
            if (!ok) continue;

            // Milestone E on the kMine path only (cublas already proved A-D;
            // greedy re-runs the full forward per token and would double the
            // slowest part of the suite for no isolation value).
            if (gemm == llm::GemmPath::kMine && !smoke && !greedy.is_null()) {
                std::vector<int64_t> want =
                    greedy["prompts"][pi]["greedy_continuation"].get<std::vector<int64_t>>();
                std::vector<int64_t> got = llm::greedy_decode_gpu(gpu, ids, int(want.size()), gemm);
                if (got != want) {
                    std::printf("FAIL %s: greedy continuation mismatch\n", tag.c_str());
                    failures++;
                    continue;
                }
                std::printf("%s: full ladder A-E passed (T=%lld)\n", tag.c_str(), (long long)T);
            } else {
                std::printf("%s: ladder A-D passed (T=%lld)\n", tag.c_str(), (long long)T);
            }
        }
    }

    if (failures == 0) {
        std::printf("test_forward_gpu: all prompts passed on both GEMM paths\n");
        return 0;
    }
    std::printf("test_forward_gpu: %d FAILURES\n", failures);
    return 1;
}
