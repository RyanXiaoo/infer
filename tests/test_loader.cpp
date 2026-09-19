// test_loader.cpp — constructive verification of the safetensors loader against
// the oracle manifest (tests/golden/manifest.json, produced by tools/dump_logits.py).
//
// The key assertion is EXACT SET EQUALITY between the manifest's expected tensor
// set (derived from the HF module tree) and what the file contains: one check
// that catches tied embeddings (no lm_head.weight), the Qwen2 QKV biases, stray
// buffers, and truncated/wrong-variant downloads. Spot values are compared as
// raw bf16 BITS at asymmetric (row, col) indices — bit compare catches
// offset/byte-order bugs tolerance would hide; strided indices catch
// transposed-layout bugs element (0,0) can't.

#include "../src/loader.h"
#include "../src/model_select.h"
#include "../src/model_config.h"

#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <set>
#include <string>

#include "../third_party/nlohmann/json.hpp"

using nlohmann::json;

static int failures = 0;

#define CHECK(cond, ...)                                                     \
    do {                                                                     \
        if (!(cond)) {                                                       \
            std::printf("FAIL %d: ", __LINE__);                              \
            std::printf(__VA_ARGS__);                                        \
            std::printf("\n");                                               \
            failures++;                                                      \
        }                                                                    \
    } while (0)

// sha256 via the system tool (test-only; macOS ships shasum, Linux sha256sum).
static std::string sha256_file(const std::string& path) {
    std::string cmds[] = {"shasum -a 256 \"" + path + "\" 2>/dev/null",
                          "sha256sum \"" + path + "\" 2>/dev/null"};
    for (const auto& cmd : cmds) {
        if (FILE* p = popen(cmd.c_str(), "r")) {
            char buf[160] = {};
            if (fgets(buf, sizeof(buf), p)) {
                pclose(p);
                std::string s(buf);
                return s.substr(0, s.find_first_of(" \t"));
            }
            pclose(p);
        }
    }
    return "";
}

int main() {
    const std::string root = MODEL_ROOT;   // injected by CMake: repo root
    const std::string model_dir = llm::model_dir(root);
    const std::string st_path = model_dir + "/model.safetensors";
    const std::string manifest_path = llm::golden_dir(root) + "/manifest.json";

    std::ifstream mf(manifest_path);
    if (!mf) {
        std::printf("SKIP: %s not found — run tools/download_model.py and "
                    "tools/dump_logits.py first\n", manifest_path.c_str());
        return 0;
    }
    json manifest = json::parse(mf);

    // --- sha256: the oracle and this test must be looking at the same bytes ---
    std::string sha = sha256_file(st_path);
    CHECK(sha == manifest["sha256_model_safetensors"].get<std::string>(),
          "sha256 mismatch: file %s vs manifest %s — regenerate goldens",
          sha.c_str(), manifest["sha256_model_safetensors"].get<std::string>().c_str());

    // --- load ---
    llm::SafetensorsFile st;
    st.open(st_path);

    // --- exact set equality with the module-tree-derived expected set ---
    std::set<std::string> expected(manifest["expected_tensors"].begin(),
                                   manifest["expected_tensors"].end());
    auto names_v = st.names();
    std::set<std::string> actual(names_v.begin(), names_v.end());
    for (const auto& n : expected)
        CHECK(actual.count(n), "missing from file: %s", n.c_str());
    for (const auto& n : actual)
        CHECK(expected.count(n), "unexpected in file: %s", n.c_str());
    CHECK(!actual.count("lm_head.weight"),
          "lm_head.weight present but embeddings should be tied");

    // --- shapes and dtypes match the manifest ---
    for (auto& [name, info] : manifest["tensor_shapes"].items()) {
        if (!st.has(name)) continue;   // already reported above
        const auto& t = st.get(name);
        auto shape = info["shape"].get<std::vector<int64_t>>();
        CHECK(t.shape == shape, "shape mismatch for %s", name.c_str());
        CHECK(std::string(llm::dtype_name(t.dtype)) == info["dtype"].get<std::string>(),
              "dtype mismatch for %s: %s vs %s", name.c_str(), llm::dtype_name(t.dtype),
              info["dtype"].get<std::string>().c_str());
    }

    // --- offsets: non-overlapping, in-bounds; gaps allowed (alignment) ---
    {
        std::vector<std::pair<size_t, size_t>> spans;
        for (const auto& n : names_v) spans.push_back(st.data_offsets(n));
        std::sort(spans.begin(), spans.end());
        size_t covered = 0;
        for (size_t i = 0; i < spans.size(); i++) {
            CHECK(spans[i].second <= st.data_section_size(),
                  "span out of bounds at %zu", spans[i].second);
            if (i > 0)
                CHECK(spans[i].first >= spans[i - 1].second,
                      "overlapping tensor spans at %zu", spans[i].first);
            covered += spans[i].second - spans[i].first;
        }
        // Coverage-with-small-gaps: safetensors permits alignment padding.
        double frac = double(covered) / double(st.data_section_size());
        CHECK(frac > 0.999, "tensors cover only %.4f of data section", frac);
    }

    // --- strided bit-exact spot checks ---
    for (const auto& sc : manifest["spot_checks"]) {
        std::string name = sc["tensor"].get<std::string>();
        int64_t row = sc["row"].get<int64_t>();
        const auto& t = st.get(name);
        CHECK(t.dtype == llm::Dtype::BF16, "spot-check tensor %s not bf16", name.c_str());
        int64_t idx;
        if (sc["col"].is_null()) {
            idx = row;                                     // 1-D tensor
        } else {
            int64_t col = sc["col"].get<int64_t>();
            idx = row * t.shape[1] + col;                  // row-major [out, in]
        }
        uint16_t got = t.u16()[idx];
        uint16_t want = static_cast<uint16_t>(sc["bf16_bits"].get<int>());
        CHECK(got == want, "spot value mismatch %s[%lld]: 0x%04x vs 0x%04x",
              name.c_str(), (long long)idx, got, want);
    }

    // --- ModelConfig against facts known independently of our parser (the model
    // cards), per checkpoint. A model without an entry still gets every check
    // above; only this cross-check is skipped.
    struct Facts { const char* model; int64_t hidden, layers, heads, kv_heads, head_dim,
                   intermediate, vocab; double rope_theta; bool tied, attn_bias; };
    static const Facts kFacts[] = {
        {"Qwen2.5-0.5B-Instruct", 896, 24, 14, 2, 64, 4864, 151936, 1000000.0, true, true},
        {"Qwen2.5-1.5B-Instruct", 1536, 28, 12, 2, 128, 8960, 151936, 1000000.0, true, true},
    };
    auto cfg = llm::ModelConfig::from_file(model_dir + "/config.json");
    const Facts* f = nullptr;
    for (const Facts& k : kFacts)
        if (llm::model_name() == k.model) f = &k;
    if (!f) {
        std::printf("note: no config facts recorded for %s, cross-check skipped\n",
                    llm::model_name().c_str());
    } else {
        CHECK(cfg.architecture == "Qwen2ForCausalLM", "architecture: %s", cfg.architecture.c_str());
        CHECK(cfg.hidden_size == f->hidden, "hidden_size %lld", (long long)cfg.hidden_size);
        CHECK(cfg.num_hidden_layers == f->layers, "layers %lld", (long long)cfg.num_hidden_layers);
        CHECK(cfg.num_attention_heads == f->heads, "heads %lld", (long long)cfg.num_attention_heads);
        CHECK(cfg.num_key_value_heads == f->kv_heads, "kv heads %lld", (long long)cfg.num_key_value_heads);
        CHECK(cfg.head_dim == f->head_dim, "head_dim %lld", (long long)cfg.head_dim);
        CHECK(cfg.intermediate_size == f->intermediate, "intermediate %lld", (long long)cfg.intermediate_size);
        CHECK(cfg.vocab_size == f->vocab, "vocab %lld", (long long)cfg.vocab_size);
        CHECK(cfg.rope_theta == f->rope_theta, "rope_theta %f", cfg.rope_theta);
        CHECK(cfg.tie_word_embeddings == f->tied, "tie_word_embeddings");
        CHECK(cfg.attention_bias == f->attn_bias, "attention_bias");
    }

    if (failures == 0) {
        std::printf("test_loader: all checks passed (%zu tensors, set-equal, "
                    "offsets sane, spots bit-exact)\n", actual.size());
        return 0;
    }
    std::printf("test_loader: %d FAILURES\n", failures);
    return 1;
}
