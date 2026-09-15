// tokenizer.cpp — see tokenizer.h.

#include "tokenizer.h"

#include <algorithm>
#include <array>
#include <fstream>
#include <stdexcept>

#include "../third_party/nlohmann/json.hpp"

using nlohmann::json;

namespace llm {

namespace {

// GPT-2 byte-level alphabet: the 256 bytes are mapped to printable Unicode
// codepoints so BPE operates on text. Bytes that are already printable ASCII-ish
// map to themselves; the rest map to U+0100.. in order. Returns codepoint per byte.
std::array<int, 256> byte_to_codepoint() {
    std::array<int, 256> cp{};
    std::vector<int> printable;
    for (int b = int('!'); b <= int('~'); b++) printable.push_back(b);
    for (int b = 0xA1; b <= 0xAC; b++) printable.push_back(b);
    for (int b = 0xAE; b <= 0xFF; b++) printable.push_back(b);
    std::array<bool, 256> in{};
    for (int b : printable) in[b] = true;
    for (int b = 0; b < 256; b++) cp[b] = b;   // provisional
    int n = 0;
    for (int b = 0; b < 256; b++) {
        if (in[b]) {
            cp[b] = b;
        } else {
            cp[b] = 256 + n;   // map to U+0100+n
            n++;
        }
    }
    return cp;
}

std::string cp_to_utf8(int c) {
    std::string s;
    if (c < 0x80) {
        s += char(c);
    } else if (c < 0x800) {
        s += char(0xC0 | (c >> 6));
        s += char(0x80 | (c & 0x3F));
    } else {
        s += char(0xE0 | (c >> 12));
        s += char(0x80 | ((c >> 6) & 0x3F));
        s += char(0x80 | (c & 0x3F));
    }
    return s;
}

// Decode one UTF-8 codepoint from s at i; advances i. Returns -1 on end.
int next_codepoint(const std::string& s, size_t& i) {
    if (i >= s.size()) return -1;
    unsigned char c = s[i];
    if (c < 0x80) { i += 1; return c; }
    if ((c >> 5) == 0x6 && i + 1 < s.size()) {
        int cp = ((c & 0x1F) << 6) | (s[i + 1] & 0x3F);
        i += 2;
        return cp;
    }
    if ((c >> 4) == 0xE && i + 2 < s.size()) {
        int cp = ((c & 0x0F) << 12) | ((s[i + 1] & 0x3F) << 6) | (s[i + 2] & 0x3F);
        i += 3;
        return cp;
    }
    if ((c >> 3) == 0x1E && i + 3 < s.size()) {
        int cp = ((c & 0x07) << 18) | ((s[i + 1] & 0x3F) << 12) |
                 ((s[i + 2] & 0x3F) << 6) | (s[i + 3] & 0x3F);
        i += 4;
        return cp;
    }
    i += 1;   // malformed byte, consume one
    return c;
}

// Codepoint classification for the GPT-2/Qwen pre-tokenizer regex.
// ASCII-exact; any non-ASCII codepoint that is not obviously space/punct is
// treated as a letter (the documented ASCII-exact scope).
bool is_ws(int c) { return c == ' ' || c == '\t' || c == '\n' || c == '\r' ||
                           c == '\f' || c == '\v'; }
bool is_letter(int c) {
    if ((c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z')) return true;
    if (c < 0x80) return false;      // ASCII non-letter
    return !is_ws(c);                 // non-ASCII: treat as letter unless whitespace
}
bool is_digit(int c) { return c >= '0' && c <= '9'; }

} // namespace

void Tokenizer::load(const std::string& path) {
    std::ifstream f(path);
    if (!f) throw std::runtime_error("cannot open " + path);
    json j = json::parse(f);

    // Byte-level alphabet.
    auto cp = byte_to_codepoint();
    for (int b = 0; b < 256; b++) {
        byte_to_uni_[b] = cp_to_utf8(cp[b]);
        uni_to_byte_[byte_to_uni_[b]] = b;
    }

    // Vocab: token string -> id.
    const auto& vocab = j.at("model").at("vocab");
    int64_t max_id = 0;
    for (auto it = vocab.begin(); it != vocab.end(); ++it) {
        int64_t id = it.value().get<int64_t>();
        vocab_[it.key()] = id;
        max_id = std::max(max_id, id);
    }

    // Merges: rank = position in the list. Entries are "A B" (or ["A","B"]).
    const auto& merges = j.at("model").at("merges");
    int32_t rank = 0;
    for (const auto& m : merges) {
        std::string key;
        if (m.is_string()) {
            key = m.get<std::string>();
        } else {  // array form
            key = m[0].get<std::string>() + " " + m[1].get<std::string>();
        }
        merge_rank_[key] = rank++;
    }

    // Added/special tokens: literal content -> id.
    if (j.contains("added_tokens")) {
        for (const auto& a : j.at("added_tokens")) {
            std::string content = a.at("content").get<std::string>();
            int64_t id = a.at("id").get<int64_t>();
            specials_[content] = id;
            special_texts_.push_back(content);
            vocab_[content] = id;   // so id_to_token has them
            max_id = std::max(max_id, id);
        }
    }
    // Longest-first so <|im_start|> matches before any prefix.
    std::sort(special_texts_.begin(), special_texts_.end(),
              [](const std::string& a, const std::string& b) { return a.size() > b.size(); });

    id_to_token_.assign(max_id + 1, "");
    for (const auto& [tok, id] : vocab_) id_to_token_[id] = tok;

    eos_id_ = specials_.count("<|im_end|>") ? specials_["<|im_end|>"] : 0;
    im_start_ = specials_.count("<|im_start|>") ? specials_["<|im_start|>"] : 0;
    im_end_ = eos_id_;
}

// GPT-2/Qwen pre-tokenizer, ASCII-exact. The regex alternatives, in order:
//   '(?i:'s|'t|'re|'ve|'m|'ll|'d)   contractions
//   [^\r\n\p{L}\p{N}]?\p{L}+        optional non-L/N lead + letter run
//   \p{N}                           a SINGLE digit (no +)
//    ?[^\s\p{L}\p{N}]+[\r\n]*       optional space + punct run + trailing newlines
//   \s*[\r\n]+                      newline runs
//   \s+(?!\S) | \s+                 trailing/other whitespace
std::vector<std::string> Tokenizer::pretokenize(const std::string& text) const {
    std::vector<std::string> out;
    // Work on codepoints with byte offsets so pieces are byte-accurate.
    std::vector<int> cps;
    std::vector<size_t> starts;   // byte offset of each codepoint
    for (size_t i = 0; i < text.size();) {
        starts.push_back(i);
        cps.push_back(next_codepoint(text, i));
    }
    starts.push_back(text.size());
    const size_t N = cps.size();

    auto piece = [&](size_t a, size_t b) {
        out.push_back(text.substr(starts[a], starts[b] - starts[a]));
    };
    auto lower = [](int c) { return (c >= 'A' && c <= 'Z') ? c + 32 : c; };

    size_t p = 0;
    while (p < N) {
        int c = cps[p];

        // Contractions: 's 't 're 've 'm 'll 'd (case-insensitive).
        if (c == '\'' && p + 1 < N) {
            int n1 = lower(cps[p + 1]);
            if (n1 == 's' || n1 == 't' || n1 == 'm' || n1 == 'd') { piece(p, p + 2); p += 2; continue; }
            if (p + 2 < N) {
                int n2 = lower(cps[p + 2]);
                if ((n1 == 'r' && n2 == 'e') || (n1 == 'v' && n2 == 'e') ||
                    (n1 == 'l' && n2 == 'l')) { piece(p, p + 3); p += 3; continue; }
            }
        }

        // [^\r\n\p{L}\p{N}]? \p{L}+ : optional single non-letter/non-digit
        // (but not \r\n) lead, then one or more letters.
        {
            size_t q = p;
            if (!is_letter(c) && !is_digit(c) && c != '\r' && c != '\n' &&
                q + 1 < N && is_letter(cps[q + 1])) {
                q++;  // consume the lead char
            }
            if (is_letter(cps[q])) {
                size_t start = p;
                while (q < N && is_letter(cps[q])) q++;
                piece(start, q);
                p = q;
                continue;
            }
        }

        // \p{N} : a single digit.
        if (is_digit(c)) { piece(p, p + 1); p += 1; continue; }

        //  ?[^\s\p{L}\p{N}]+[\r\n]* : optional leading space, punct run, trailing newlines.
        {
            size_t q = p;
            size_t start = p;
            if (c == ' ' && q + 1 < N && !is_ws(cps[q + 1]) &&
                !is_letter(cps[q + 1]) && !is_digit(cps[q + 1])) {
                q++;  // leading space belongs to the punct run
            }
            if (q < N && !is_ws(cps[q]) && !is_letter(cps[q]) && !is_digit(cps[q])) {
                while (q < N && !is_ws(cps[q]) && !is_letter(cps[q]) && !is_digit(cps[q])) q++;
                while (q < N && (cps[q] == '\r' || cps[q] == '\n')) q++;
                piece(start, q);
                p = q;
                continue;
            }
        }

        // Whitespace runs (rules 5-7). A maximal run [p,q). If the run's last
        // char is a plain space AND the next token is a letter or punct (never a
        // digit), that one space is "donated" to the next token via the ` ?`
        // lead in rules 2/4; we leave it behind. The rest is emitted as a
        // whitespace piece. (Mixed space+newline+space runs are an ASCII-scope
        // edge case; the golden test covers the ordinary ones.)
        if (is_ws(c)) {
            size_t q = p;
            while (q < N && is_ws(cps[q])) q++;
            size_t run_end = q;
            if (q < N && cps[q - 1] == ' ' && !is_digit(cps[q]) && !is_ws(cps[q]))
                run_end = q - 1;   // donate the last space to the next token
            if (run_end > p) piece(p, run_end);
            p = run_end;
            if (p == q) continue;
            // One space left: consumed as the next rule's lead. Guard progress.
            if (p < N) { /* next iteration's letter/punct rule takes it */ }
            continue;
        }

        // Fallback: emit the single codepoint (shouldn't normally hit).
        piece(p, p + 1);
        p += 1;
    }
    return out;
}

std::vector<int64_t> Tokenizer::bpe(const std::string& piece_bytes) const {
    // Map raw bytes to the byte-level alphabet, one symbol per byte.
    std::vector<std::string> syms;
    for (unsigned char b : piece_bytes) syms.push_back(byte_to_uni_[b]);
    if (syms.empty()) return {};

    // Repeatedly merge the adjacent pair with the lowest rank.
    while (syms.size() >= 2) {
        int best_rank = INT32_MAX;
        size_t best_i = 0;
        for (size_t i = 0; i + 1 < syms.size(); i++) {
            auto it = merge_rank_.find(syms[i] + " " + syms[i + 1]);
            if (it != merge_rank_.end() && it->second < best_rank) {
                best_rank = it->second;
                best_i = i;
            }
        }
        if (best_rank == INT32_MAX) break;
        syms[best_i] = syms[best_i] + syms[best_i + 1];
        syms.erase(syms.begin() + best_i + 1);
    }

    std::vector<int64_t> ids;
    for (const auto& s : syms) {
        auto it = vocab_.find(s);
        if (it == vocab_.end())
            throw std::runtime_error("bpe: symbol not in vocab: " + s);
        ids.push_back(it->second);
    }
    return ids;
}

std::vector<int64_t> Tokenizer::encode_ordinary(const std::string& text) const {
    std::vector<int64_t> ids;
    for (const auto& piece : pretokenize(text)) {
        auto part = bpe(piece);
        ids.insert(ids.end(), part.begin(), part.end());
    }
    return ids;
}

std::vector<int64_t> Tokenizer::encode(const std::string& text) const {
    // Split around special-token literals (longest-first), which map directly.
    std::vector<int64_t> ids;
    size_t pos = 0;
    while (pos < text.size()) {
        size_t best = std::string::npos;
        size_t best_len = 0;
        std::string best_tok;
        for (const auto& sp : special_texts_) {
            size_t at = text.find(sp, pos);
            if (at != std::string::npos && (at < best || (at == best && sp.size() > best_len))) {
                best = at;
                best_len = sp.size();
                best_tok = sp;
            }
        }
        if (best == std::string::npos) {
            auto part = encode_ordinary(text.substr(pos));
            ids.insert(ids.end(), part.begin(), part.end());
            break;
        }
        if (best > pos) {
            auto part = encode_ordinary(text.substr(pos, best - pos));
            ids.insert(ids.end(), part.begin(), part.end());
        }
        ids.push_back(specials_.at(best_tok));
        pos = best + best_len;
    }
    return ids;
}

std::string Tokenizer::id_to_bytes(int64_t id) const {
    if (id < 0 || id >= int64_t(id_to_token_.size())) return "";
    const std::string& tok = id_to_token_[id];
    if (specials_.count(tok)) return tok;   // specials render literally (or "")
    // Reverse the byte-level alphabet back to raw bytes.
    std::string out;
    for (size_t i = 0; i < tok.size();) {
        // byte-level chars are 1-3 UTF-8 bytes; try longest match.
        bool matched = false;
        for (int len = 3; len >= 1; len--) {
            if (i + len <= tok.size()) {
                auto it = uni_to_byte_.find(tok.substr(i, len));
                if (it != uni_to_byte_.end()) {
                    out += char(it->second);
                    i += len;
                    matched = true;
                    break;
                }
            }
        }
        if (!matched) i++;   // shouldn't happen
    }
    return out;
}

std::string Tokenizer::decode(const std::vector<int64_t>& ids) const {
    std::string out;
    for (int64_t id : ids) out += id_to_bytes(id);
    return out;
}

std::vector<int64_t> Tokenizer::chat_wrap(const std::string& user,
                                          const std::string& system) const {
    std::string s = "<|im_start|>system\n" + system + "<|im_end|>\n" +
                    "<|im_start|>user\n" + user + "<|im_end|>\n" +
                    "<|im_start|>assistant\n";
    return encode(s);
}

} // namespace llm
