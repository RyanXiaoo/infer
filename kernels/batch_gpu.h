// batch_gpu.h — GpuBatch: the GPU BatchEngine (Stage 6).
//
// S slots, each with its own contiguous KV cache region ([max_seq x kv_dim]
// per layer; paging is Stage 7). prefill() runs one prompt into one slot;
// step() runs ONE decode step for up to 32 rows at once: every weight matrix
// is read once per step and multiplied with all rows' activations, which is
// where batching's throughput comes from.
//
// Greedy only: the next token is chosen on the device (argmax per row) and
// only B token ids come back, not B x vocab logits. Per-request sampling in
// the batched path is a listed follow-up.
//
// Plain C++ header (boundary rule).

#pragma once

#include "model_gpu.h"
#include "../src/scheduler.h"

#include <cstdint>
#include <memory>
#include <vector>

namespace llm {

class GpuBatch : public BatchEngine {
public:
    GpuBatch(GpuModel& m, int n_slots, int64_t max_seq, GemmPath gemm = GemmPath::kMine);
    ~GpuBatch() override;
    GpuBatch(const GpuBatch&) = delete;
    GpuBatch& operator=(const GpuBatch&) = delete;

    int slots() const override;
    int64_t max_seq() const override;
    int64_t prefill(int slot, const std::vector<int64_t>& prompt) override;
    std::vector<int64_t> step(const std::vector<StepRow>& rows) override;

    int64_t position(int slot) const;   // tokens in the slot's cache
    static constexpr int kMaxRows = 32;

    struct Impl;

private:
    std::unique_ptr<Impl> impl_;
};

} // namespace llm
