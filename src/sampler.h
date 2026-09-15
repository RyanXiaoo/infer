// sampler.h — turns a logits vector into the next token id.
//
// temperature = 0 is exact greedy (argmax), preserving the deterministic path.
// Otherwise: scale by 1/temp, optional top-k truncation, optional top-p
// (nucleus) truncation, softmax, sample with a seeded RNG. Runs on the host
// over the logits the session already returns, so it is device-agnostic.

#pragma once

#include <cstdint>
#include <random>
#include <vector>

namespace llm {

struct Sampler {
    float temperature = 0.0f;   // 0 => greedy
    int top_k = 0;              // 0 => disabled
    float top_p = 1.0f;        // 1 => disabled
    std::mt19937 rng;

    explicit Sampler(uint64_t seed = 0) : rng(seed) {}

    int64_t sample(const std::vector<float>& logits);
};

} // namespace llm
