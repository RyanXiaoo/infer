// forward.h — the CPU forward pass (Stage 2's deliverable).
//
// Deliberately slow reference implementation: no KV cache, no SIMD, no
// threading. Weights are read as bf16 and converted per-element at use-time
// (f16::bf16_to_f32); activations and accumulators are fp32 throughout. Every
// optimized version that follows is validated against this one.
//
// The tap hook exists so the parity test can capture intermediates (named to
// match the golden npz keys from tools/dump_logits.py) without the forward API
// growing test-only return values. With no hook installed, no copies happen.

#pragma once

#include "model.h"

#include <cstdint>
#include <functional>
#include <string>
#include <vector>

namespace llm {

// (name, row-major data, shape) — shapes mirror the goldens' layouts, e.g.
// q_post_rope arrives as [n_heads, T, head_dim] like HF's transposed view.
using TapFn = std::function<void(const std::string& name, const float* data,
                                 const std::vector<int64_t>& shape)>;

// Full prefill over token_ids. Returns logits:
//   last_only=false: [T, vocab]  (what the parity test compares)
//   last_only=true:  [1, vocab]  (greedy decode only needs the last position;
//                                 skipping the other rows changes no values)
std::vector<float> forward(const Model& m, const std::vector<int64_t>& token_ids,
                           const TapFn* taps = nullptr, bool last_only = false);

// Greedy continuation: append argmax(next-token logits) n_new times, re-running
// the full forward each step (the O(T^2)-per-token baseline — no cache until
// Stage 4). Returns just the generated ids.
std::vector<int64_t> greedy_decode(const Model& m, std::vector<int64_t> token_ids,
                                   int n_new);

} // namespace llm
