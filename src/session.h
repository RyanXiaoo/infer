// session.h — the CPU DecodeSession: a KV cache plus the two operations that
// use it (Stage 4).
//
// The invariant this rests on: with causal attention, an old token's K and V
// never change once computed (later tokens never influenced them), so they are
// computed once and kept. prefill() runs the full forward over the prompt and
// fills the cache; decode_one() runs a single-token forward that reads the
// cache instead of recomputing the past. Keys are cached ALREADY ROTATED, and
// each new token's Q/K rotate at its true position (the current cache length).
//
// Contract: greedy decoding through a session produces the identical token
// stream to the Stage 3 full-recompute engine (tests/test_kv_cache.cpp).

#pragma once

#include "model.h"

#include <cstdint>
#include <vector>

namespace llm {

class Session {
public:
    // Preallocates the cache: per layer, K and V of [max_seq x n_kv*head_dim]
    // fp32 (24 x 2 x 512 x 128 x 4B = 12MB at the defaults). max_seq is this
    // session's context capacity — prompt plus generated tokens.
    explicit Session(const Model& m, int64_t max_seq = 512);

    // Full forward over the prompt; fills the cache; returns [vocab] logits
    // for the last position. Must be called first, exactly once.
    std::vector<float> prefill(const std::vector<int64_t>& ids);

    // Single-token forward against the cache; appends the token's K/V;
    // returns [vocab] logits. Throws when the cache is full.
    std::vector<float> decode_one(int64_t id);

    int64_t position() const { return pos_; }

private:
    const Model& m_;
    const int64_t max_seq_;
    int64_t pos_ = 0;
    // Per layer, [max_seq x kv_dim] row-major; row = absolute position.
    std::vector<std::vector<float>> k_cache_, v_cache_;
    // Reusable T=1 scratch (allocated once; decode_one is called in a loop).
    std::vector<float> h_, normed_, q_, k_, v_, ctx_, delta_, gate_, up_;
};

// prefill + argmax/decode_one loop. Returns the n_new generated ids.
std::vector<int64_t> greedy_decode_cached(const Model& m,
                                          const std::vector<int64_t>& ids,
                                          int n_new, int64_t max_seq = 512);

} // namespace llm
