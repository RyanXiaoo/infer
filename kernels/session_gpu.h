// session_gpu.h — GPU DecodeSession, the device-resident twin of src/session.h.
//
// Plain C++ header (boundary rule). Same contract as the CPU Session: prefill
// once, decode_one per token, cached greedy matches the recompute engine.

#pragma once

#include "model_gpu.h"

#include <cstdint>
#include <memory>
#include <vector>

namespace llm {

class GpuSession {
public:
    GpuSession(GpuModel& m, int64_t max_seq = 512, GemmPath gemm = GemmPath::kMine,
               AttnPath attn = AttnPath::kParallel);
    ~GpuSession();
    GpuSession(const GpuSession&) = delete;
    GpuSession& operator=(const GpuSession&) = delete;

    std::vector<float> prefill(const std::vector<int64_t>& ids);
    std::vector<float> decode_one(int64_t id);
    int64_t position() const;

    struct Impl;

private:
    std::unique_ptr<Impl> impl_;
};

std::vector<int64_t> greedy_decode_cached_gpu(GpuModel& m,
                                              const std::vector<int64_t>& ids,
                                              int n_new, int64_t max_seq = 512,
                                              GemmPath gemm = GemmPath::kMine,
                                              AttnPath attn = AttnPath::kParallel);

} // namespace llm
