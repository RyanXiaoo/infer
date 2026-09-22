// batch_gpu.h — GpuBatch: the GPU BatchEngine (Stage 6, paged in Stage 7).
//
// S slots share a paged KV cache: per layer, a pool of 16-position blocks;
// each slot owns a block table (src/block_pool.h decides which blocks, this
// class moves the bytes). The pool is sized by a byte budget, so memory in use
// follows the tokens present rather than slots x max_seq.
//
// One code path serves prefill and decode: forward_rows() runs up to 32
// (slot, position, token) rows through the model, writing each row's K/V into
// its slot's blocks and reading attention through the table. A prompt is T
// rows of one slot at consecutive positions (chunked by 32); a decode step is
// one row per active slot. Prefix cache: full prompt blocks already computed
// by an earlier request are shared, and only the tail is run.
//
// Greedy only: the next token is chosen on the device (argmax per row).
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
    // kv_budget_bytes = 0 -> enough blocks for every slot to reach max_seq
    // (the Stage 6 reservation). Otherwise the pool holds budget / bytes_per_block
    // blocks and the scheduler preempts when they run out.
    // use_graphs: capture the step's kernel sequence once per batch width and
    // replay it with one launch (Stage 8). Only for gemm=mine.
    // attn_split: flash-decoding attention (positions split across blocks with
    // online-softmax partials) instead of one block per (row, head).
    GpuBatch(GpuModel& m, int n_slots, int64_t max_seq, GemmPath gemm = GemmPath::kMine,
             int64_t kv_budget_bytes = 0, bool prefix_cache = true, bool use_graphs = false,
             bool attn_split = true);
    ~GpuBatch() override;
    GpuBatch(const GpuBatch&) = delete;
    GpuBatch& operator=(const GpuBatch&) = delete;

    int slots() const override;
    int64_t max_seq() const override;
    bool has_room(int64_t prompt_len) const override;
    int64_t prefill(int slot, const std::vector<int64_t>& prompt, SampleParams sample = {}) override;
    std::vector<int64_t> step(const std::vector<StepRow>& rows) override;
    void release(int slot) override;
    int blocks_in_use() const override;
    int blocks_total() const override;

    int64_t position(int slot) const;       // tokens in the slot's cache
    int64_t bytes_per_block() const;        // K + V, all layers
    int64_t last_prefill_reused() const;    // positions served by the prefix cache
    int graphs_captured() const;            // distinct batch widths captured so far
    static constexpr int kMaxRows = 32;

    struct Impl;

private:
    std::unique_ptr<Impl> impl_;
};

} // namespace llm
