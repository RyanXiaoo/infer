// spec_engine.h — speculative decoding over two GpuBatch engines (Stage 11).
//
// A small DRAFT model proposes k tokens with k cheap steps; the TARGET model
// checks all k+1 positions in one forward (GpuBatch::verify), which costs
// about one target step because the weights are read once either way. Every
// accepted draft token is a target token that skipped a target step.
//
// Greedy: accept d_i while the target's argmax after d_{i-1} equals d_i; the
// token emitted at the first mismatch is the target's own argmax (or, if all
// k were accepted, the target's prediction after d_k, a free extra token).
// The output equals the target decoding alone, token for token.
//
// Both caches then roll back to the accepted length (GpuBatch::truncate).
// The draft's cache lags one token behind when everything was accepted; it
// catches up with one extra draft step before the next round.
//
// Rows with temperature > 0 fall back to a plain target step for now (sampled
// speculation is the next commit). Plain C++ header; uses only GpuBatch.

#pragma once

#include "batch_gpu.h"

#include <cstdint>
#include <memory>
#include <vector>

namespace llm {

struct SpecStats {
    int64_t rounds = 0, drafted = 0, accepted = 0, emitted = 0;
    double acceptance() const { return drafted ? double(accepted) / double(drafted) : 0.0; }
    double tokens_per_round() const { return rounds ? double(emitted) / double(rounds) : 0.0; }
};

class SpecEngine : public BatchEngine {
public:
    // target and draft must have the same slot count and tokenizer; k <= 8.
    // Verification runs all slots' candidates in as few target forwards as
    // fit 32 rows each (k = 4 -> 6 slots per forward).
    SpecEngine(GpuBatch& target, GpuBatch& draft, int k);

    int slots() const override { return target_.slots(); }
    int64_t max_seq() const override { return target_.max_seq(); }
    bool has_room(int64_t prompt_len) const override;
    int64_t prefill(int slot, const std::vector<int64_t>& prompt, SampleParams sample = {}) override;
    std::vector<int64_t> step(const std::vector<StepRow>& rows) override;
    std::vector<std::vector<int64_t>> step_multi(const std::vector<StepRow>& rows) override;
    void release(int slot) override;
    int blocks_in_use() const override { return target_.blocks_in_use(); }
    int blocks_total() const override { return target_.blocks_total(); }

    const SpecStats& stats() const { return stats_; }
    int k() const { return k_; }

private:
    GpuBatch& target_;
    GpuBatch& draft_;
    const int k_;
    SpecStats stats_;
    // Per slot: a draft token the draft's cache still has to ingest (after a
    // round where every draft token was accepted), or -1.
    std::vector<int64_t> draft_owes_;
};

} // namespace llm
