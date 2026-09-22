// test_tokenizer.cpp — encode must match HF ids exactly on an ordinary-text
// corpus (English, code, punctuation, digits, whitespace), decode must
// round-trip, and the chat template must match HF's apply_chat_template.

#include "../src/tokenizer.h"

#include <cstdio>
#include <fstream>
#include <string>
#include <vector>

#include "../third_party/nlohmann/json.hpp"

using nlohmann::json;
static int failures = 0;

static void show(const char* what, const std::string& text,
                 const std::vector<int64_t>& got, const std::vector<int64_t>& want) {
    std::printf("FAIL %s: %s\n  ours:  ", what, text.c_str());
    for (auto v : got) std::printf("%lld ", (long long)v);
    std::printf("\n  golden:");
    for (auto v : want) std::printf(" %lld", (long long)v);
    std::printf("\n");
}

int main() {
    const std::string root = MODEL_ROOT;
    std::ifstream tf(root + "/tests/golden/tokenizer_tests.json");
    if (!tf) { std::printf("SKIP: run tools/dump_tokenizer_tests.py first\n"); return 0; }
    json tests = json::parse(tf);

    llm::Tokenizer tok;
    tok.load(root + "/models/Qwen2.5-0.5B-Instruct/tokenizer.json");

    // Encode + round-trip over the corpus.
    for (const auto& c : tests["corpus"]) {
        std::string text = c["text"].get<std::string>();
        std::vector<int64_t> want = c["ids"].get<std::vector<int64_t>>();
        std::vector<int64_t> got = tok.encode(text);
        if (got != want) { show("encode", text, got, want); failures++; continue; }
        std::string round = tok.decode(got);
        if (round != text) {
            std::printf("FAIL decode round-trip: %s\n  got: %s\n", text.c_str(), round.c_str());
            failures++;
        }
    }

    // Adversarial corpus (Stage 9): exact-match rate per category. Every
    // category must match HF exactly now that the pre-tokenizer uses the full
    // Unicode letter/number/space classes; round-trip must always hold.
    if (tests.contains("adversarial")) {
        int total = 0, matched = 0;
        for (auto it = tests["adversarial"].begin(); it != tests["adversarial"].end(); ++it) {
            int n = 0, ok = 0;
            for (const auto& c : it.value()) {
                std::string text = c["text"].get<std::string>();
                std::vector<int64_t> want = c["ids"].get<std::vector<int64_t>>();
                std::vector<int64_t> got = tok.encode(text);
                n++;
                if (got == want) ok++;
                else show(("encode[" + it.key() + "]").c_str(), text, got, want);
                // Round-trip recovers the NFC form (HF does the same: the
                // normalizer runs before tokenization).
                if (tok.decode(got) != tok.normalize(text)) { std::printf("FAIL round-trip [%s]: %s\n", it.key().c_str(), text.c_str()); failures++; }
            }
            std::printf("adversarial %-16s %d/%d exact\n", it.key().c_str(), ok, n);
            total += n; matched += ok;
            failures += n - ok;
        }
        std::printf("adversarial total: %d/%d exact\n", matched, total);
    }

    // Chat template.
    for (const auto& c : tests["chats"]) {
        std::vector<int64_t> want = c["ids"].get<std::vector<int64_t>>();
        // Only the single-user and system+user shapes the C++ chat_wrap supports.
        const auto& msgs = c["messages"];
        std::string user, system;
        bool has_system = false;
        for (const auto& m : msgs) {
            if (m["role"] == "user") user = m["content"].get<std::string>();
            else if (m["role"] == "system") { system = m["content"].get<std::string>(); has_system = true; }
        }
        std::vector<int64_t> got = has_system ? tok.chat_wrap(user, system) : tok.chat_wrap(user);
        if (got != want) { show("chat_wrap", user, got, want); failures++; }
    }

    // eos sanity.
    if (tok.eos_id() != tests["special"]["eos_id"].get<int64_t>()) {
        std::printf("FAIL eos_id: %lld vs %lld\n", (long long)tok.eos_id(),
                    (long long)tests["special"]["eos_id"].get<int64_t>());
        failures++;
    }

    if (failures == 0) { std::printf("test_tokenizer: all encode/decode/chat checks passed\n"); return 0; }
    std::printf("test_tokenizer: %d FAILURES\n", failures);
    return 1;
}
