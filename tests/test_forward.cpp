// test_forward.cpp — the Stage 2 validation ladder: our forward pass vs the
// HF goldens, tap by tap inside layer 0, then every layer boundary, then
// logits, then greedy decode. Comparison discipline (why two goldens exist):
//
//   * The checkpoint stores ONLY bf16 weights, so HF's "fp32" run is bf16
//     weights upcast + fp32 activations — arithmetically the SAME setup as
//     this engine (bf16 weights, fp32 math). The fp32 golden is therefore the
//     tight reference: healthy is cosine >= 0.999 vs it (in practice ~1.0;
//     only accumulation order differs). First observed empirically: attention
//     matched fp32 to 2e-6 while sitting at 0.99 cosine vs bf16.
//   * The bf16 golden (bf16 activations end to end) shows what full-bf16
//     drift looks like — diagnostic context, printed on failure, not a gate.
//     Far from BOTH goldens -> bug in the op.
//   * Taps compare in execution order and the ladder stops a prompt at its
//     first failing rung — a failure names ONE op, not "somewhere in 24 layers".

#include "../src/forward.h"
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
    int64_t n = 0;
};

static Cmp compare(const float* a, const float* b, int64_t n) {
    Cmp c;
    c.n = n;
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

// One rung: our tap vs BOTH goldens. Pass = cosine >= 0.999 vs the fp32 golden
// (the same-arithmetic reference; see header). On failure both comparisons
// print, so "bug" vs "precision drift" is visible immediately.
static bool check_tap(const std::string& prompt_tag, const std::string& name,
                      const std::vector<float>& ours,
                      const std::map<std::string, Array>& g32,
                      const std::map<std::string, Array>& g16) {
    auto it32 = g32.find(name), it16 = g16.find(name);
    if (it32 == g32.end()) {
        std::printf("FAIL %s/%s: tap missing from golden npz\n", prompt_tag.c_str(),
                    name.c_str());
        failures++;
        return false;
    }
    const Array& ref32 = it32->second;
    if (int64_t(ours.size()) != ref32.numel()) {
        std::printf("FAIL %s/%s: numel %zu vs golden %lld\n", prompt_tag.c_str(),
                    name.c_str(), ours.size(), (long long)ref32.numel());
        failures++;
        return false;
    }
    Cmp c32 = compare(ours.data(), ref32.f32(), ref32.numel());
    if (c32.cosine >= 0.999) return true;

    Cmp c16{};
    if (it16 != g16.end()) c16 = compare(ours.data(), it16->second.f32(), ref32.numel());
    std::printf("FAIL %s/%s: cosine vs fp32 %.6f (max_abs %.4g) | vs bf16 %.6f "
                "(max_abs %.4g) — far from both means bug in this op\n",
                prompt_tag.c_str(), name.c_str(), c32.cosine, c32.max_abs,
                c16.cosine, c16.max_abs);
    failures++;
    return false;
}

int main() {
    const std::string root = MODEL_ROOT;
    const std::string golden = root + "/tests/golden";

    std::ifstream mf(golden + "/manifest.json");
    if (!mf) {
        std::printf("SKIP: goldens not found — run tools/dump_logits.py first\n");
        return 0;
    }
    nlohmann::json manifest = nlohmann::json::parse(mf);
    int n_prompts = int(manifest["prompts"].size());

    // Project convention (see tools/sanitize.sh): smoke mode for sanitizer runs.
    // One prompt covers every code path; ASan's ~10x slowdown on 5 prompts plus
    // greedy decode buys no extra coverage.
    const bool smoke = std::getenv("FORWARD_SMOKE") != nullptr;
    if (smoke) n_prompts = 1;

    llm::Model model;
    model.load(root + "/models/Qwen2.5-0.5B-Instruct");

    // Optional milestone E golden (tools/dump_logits.py --greedy).
    nlohmann::json greedy;
    {
        std::ifstream gf(golden + "/greedy.json");
        if (gf) greedy = nlohmann::json::parse(gf);
    }

    for (int pi = 0; pi < n_prompts; pi++) {
        const std::string tag = "prompt" + std::to_string(pi);
        auto g32 = llm::npy::load_npz(golden + "/" + tag + "_fp32.npz");
        auto g16 = llm::npy::load_npz(golden + "/" + tag + "_bf16.npz");

        const Array& ids_arr = g16.at("token_ids");
        std::vector<int64_t> ids(ids_arr.i64(), ids_arr.i64() + ids_arr.numel());
        const int64_t T = int64_t(ids.size());

        // Run once, capturing every tap.
        std::map<std::string, std::vector<float>> taps;
        llm::TapFn hook = [&](const std::string& name, const float* data,
                              const std::vector<int64_t>& shape) {
            int64_t n = 1;
            for (int64_t d : shape) n *= d;
            taps[name] = std::vector<float>(data, data + n);
        };
        std::vector<float> logits = llm::forward(model, ids, &hook);

        // The ladder, in execution order. First failure aborts this prompt:
        // everything downstream of a broken op would fail noisily and add nothing.
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
        for (const auto& name : ladder) {
            if (!check_tap(tag, name, taps.at(name), g32, g16)) { ok = false; break; }
        }
        if (!ok) continue;

        // Milestone D strict gate: argmax equality vs the fp32 golden (the
        // same-arithmetic reference) at EVERY position. On failure print the
        // golden's top-2 margin — a microscopic margin means a coin-flip logit
        // pair, not a broken op.
        const Array& glog = g32.at("logits");
        const int64_t V = model.cfg.vocab_size;
        for (int64_t t = 0; t < T; t++) {
            const float* ours = logits.data() + t * V;
            const float* ref = glog.f32() + t * V;
            int64_t oa = 0, ra = 0, r2 = -1;
            for (int64_t v = 1; v < V; v++) {
                if (ours[v] > ours[oa]) oa = v;
                if (ref[v] > ref[ra]) { r2 = ra; ra = v; }
                else if (r2 < 0 || ref[v] > ref[r2]) r2 = v;
            }
            if (oa != ra) {
                std::printf("FAIL %s: argmax mismatch at pos %lld: ours %lld vs golden "
                            "%lld (golden top-2 margin %.4g)\n", tag.c_str(),
                            (long long)t, (long long)oa, (long long)ra,
                            ref[ra] - ref[r2]);
                failures++;
                ok = false;
            }
        }
        if (!ok) continue;

        // Milestone E: greedy continuation vs the HF oracle loop.
        if (smoke) {
            std::printf("%s: A-D passed; E skipped (smoke mode)\n", tag.c_str());
            continue;
        }
        if (greedy.is_null()) {
            std::printf("%s: A-D passed; E SKIPPED (no greedy.json — run "
                        "tools/dump_logits.py --greedy)\n", tag.c_str());
            continue;
        }
        const auto& entry = greedy["prompts"][pi];
        std::vector<int64_t> want = entry["greedy_continuation"].get<std::vector<int64_t>>();
        std::vector<int64_t> got = llm::greedy_decode(model, ids, int(want.size()));
        if (got != want) {
            std::printf("FAIL %s: greedy continuation mismatch\n  ours:  ", tag.c_str());
            for (auto v : got) std::printf("%lld ", (long long)v);
            std::printf("\n  golden:");
            for (auto v : want) std::printf(" %lld", (long long)v);
            std::printf("\n");
            failures++;
            continue;
        }
        std::printf("%s: full ladder A-E passed (T=%lld, %zu taps, greedy %zu tokens)\n",
                    tag.c_str(), (long long)T, taps.size(), want.size());
    }

    if (failures == 0) {
        std::printf("test_forward: all prompts passed the full ladder\n");
        return 0;
    }
    std::printf("test_forward: %d FAILURES\n", failures);
    return 1;
}
