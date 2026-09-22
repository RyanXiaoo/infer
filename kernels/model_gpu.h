// model_gpu.h — public interface to the GPU forward pass (Stage 3).
//
// Plain C++ header (boundary rule: no CUDA types — safe to include from host
// code; the Mac build never compiles the implementation behind it).
//
// Mirrors src/forward.h: same TapFn, same tap names and shapes, same
// last_only semantics — so the same validation ladder drives both paths.

#pragma once

#include "../src/forward.h"   // Model, TapFn
#include "../src/model.h"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace llm {

// Which implementation serves the linear/GEMV ops. kMine is the deliverable;
// kCublas is the bisection tool (ladder fails on mine but passes on cublas ->
// the bug is in my GEMV; fails on both -> it's in RoPE/attention/norm/glue)
// and the honesty yardstick for Stage 5.
//
// kMine serves decode (T = 1) with the Stage 5 row-parallel GEMV and prefill
// (T > 1) with the Stage 3 kernel. kMineNaive is the Stage 3 one-thread-per-
// output kernel everywhere: the correctness reference and the "before" number.
enum class GemmPath { kMine, kCublas, kMineNaive };

inline GemmPath gemm_path_from(const std::string& s) {
    return s == "cublas" ? GemmPath::kCublas : s == "naive" ? GemmPath::kMineNaive
                                                            : GemmPath::kMine;
}
inline const char* gemm_path_name(GemmPath p) {
    return p == GemmPath::kCublas ? "cublas" : p == GemmPath::kMineNaive ? "naive" : "mine";
}

// How GpuSession::decode_one issues a layer's kernels. kFused (Stage 5 Phase 3)
// cuts launches per layer from 16 to 9 with bit-identical results: residual-add
// folded into the following RMSNorm, rope(q)+rope(k)+cache-append in one kernel,
// and, on GemmPath::kMine only, q|k|v and gate|up each as one GEMV over
// concatenated weights. kUnfused is the Stage 4 sequence (reference, "before").
enum class StepPath { kUnfused, kFused };

// Which kernel serves single-query (decode) attention over the KV cache.
// kNaive is the Stage 4 one-thread-per-head kernel, kept as the reference;
// kParallel is the Stage 5 one-block-per-head kernel. Independent of GemmPath:
// attention is always my kernel.
enum class AttnPath { kNaive, kParallel };

class GpuModel {
public:
    // Uploads every weight tensor (bf16, byte-identical to the file) to the
    // device once. `m` must outlive nothing — weights are copied, not viewed.
    explicit GpuModel(const Model& m);
    ~GpuModel();
    GpuModel(const GpuModel&) = delete;
    GpuModel& operator=(const GpuModel&) = delete;

    const ModelConfig& cfg() const;

    struct Impl;

private:
    std::unique_ptr<Impl> impl_;
    friend std::vector<float> forward_gpu(GpuModel&, const std::vector<int64_t>&,
                                          const TapFn*, bool, GemmPath);
    friend class GpuSession;   // shares the device weights + linear dispatch
    friend class GpuBatch;     // Stage 6 batched decode
};

// Same contract as llm::forward (src/forward.h), executed on the GPU.
std::vector<float> forward_gpu(GpuModel& m, const std::vector<int64_t>& token_ids,
                               const TapFn* taps = nullptr, bool last_only = false,
                               GemmPath gemm = GemmPath::kMine);

// Same contract as llm::greedy_decode: full recompute per token (no cache).
std::vector<int64_t> greedy_decode_gpu(GpuModel& m, std::vector<int64_t> token_ids,
                                       int n_new, GemmPath gemm = GemmPath::kMine);

} // namespace llm
