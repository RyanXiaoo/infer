// tokenizer.h — byte-level BPE tokenizer for Qwen2.5 (GPT-2 family).
//
// Loaded from tokenizer.json. Encode: split out special tokens, pre-tokenize
// with the GPT-2/Qwen regex (ASCII-exact; see tokenizer.cpp), map bytes to the
// byte-level alphabet, apply rank-priority merges, look up ids. Decode reverses.
// chat_wrap builds Qwen's <|im_start|>…<|im_end|> turn format.
//
// Plain C++ (boundary rule). ASCII-exact pre-tokenization matches HuggingFace
// on English/code/punctuation/digits; full Unicode-property parity is deferred
// (Stage 9). See tests/test_tokenizer.cpp.

#pragma once

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

namespace llm {

class Tokenizer {
public:
    void load(const std::string& tokenizer_json_path);

    std::string normalize(const std::string& text) const;          // NFC, as tokenizer.json specifies
    std::vector<int64_t> encode(const std::string& text) const;   // normalize, then split + merge
    std::string decode(const std::vector<int64_t>& ids) const;

    // Qwen chat turn: system + user, ending with the assistant generation prompt.
    std::vector<int64_t> chat_wrap(const std::string& user,
                                   const std::string& system =
                                       "You are Qwen, created by Alibaba Cloud. "
                                       "You are a helpful assistant.") const;

    int64_t eos_id() const { return eos_id_; }
    // decode helper for streaming: id -> its raw byte string (pre-UTF8-join).
    std::string id_to_bytes(int64_t id) const;

private:
    // vocab: byte-level-encoded token string -> id, and the inverse.
    std::unordered_map<std::string, int64_t> vocab_;
    std::vector<std::string> id_to_token_;   // index = id
    // merge rank: "A B" pair key -> rank (lower applies first).
    std::unordered_map<std::string, int32_t> merge_rank_;
    // special tokens: literal text -> id (matched before the regex).
    std::unordered_map<std::string, int64_t> specials_;
    std::vector<std::string> special_texts_;  // for longest-first matching

    // byte <-> byte-level-alphabet char (GPT-2 mapping), built in load().
    std::string byte_to_uni_[256];            // one byte -> 1-3 UTF-8 bytes
    std::unordered_map<std::string, int> uni_to_byte_;

    int64_t eos_id_ = 0;
    int64_t im_start_ = 0, im_end_ = 0;

    std::vector<int64_t> encode_ordinary(const std::string& text) const;
    std::vector<std::string> pretokenize(const std::string& text) const;
    std::vector<int64_t> bpe(const std::string& piece_bytes) const;
};

} // namespace llm
