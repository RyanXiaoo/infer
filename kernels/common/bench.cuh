// bench.cuh — the permanent timing harness.
//
// Rules this encodes (see plans/stage-00-setup.md):
//   * CUDA events, never wall clock: wall clock around an async launch times the
//     launch, not the kernel.
//   * Warmup iterations first (first launch pays context/module load), then N
//     timed reps; we report the median and the spread (min..max).
//   * Clocks are read via NVML before and after the run. If the SM clock drifted,
//     the record is stamped clock_stable=false — a locked card should not drift.
//   * GPU temperature is recorded: a "regression" on a warm card isn't a regression.
//   * Every result is written as one JSON file in bench/ with full provenance:
//     commit (+dirty flag), clocks, temp, driver/toolkit versions. Records from a
//     dirty tree are poison for six-month comparisons, so the dirty bit is loud.
#pragma once

#include <functional>
#include <string>
#include <vector>

namespace bench {

struct GpuState {
    unsigned sm_clock_mhz = 0;
    unsigned mem_clock_mhz = 0;
    unsigned temp_c = 0;
    std::string driver_version;
    std::string gpu_name;
};

struct Result {
    std::string kernel;      // e.g. "matmul_tiled"
    std::string dims;        // e.g. "N=4096"
    std::string gemm_path;   // "mine" | "cublas" | "n/a"
    int warmup_iters = 0;
    int timed_iters = 0;
    double median_ms = 0.0;
    double min_ms = 0.0;
    double max_ms = 0.0;
    GpuState before, after;
    bool clock_stable = true;
};

// Read current clocks/temp/driver via NVML. Safe to call repeatedly.
GpuState query_gpu_state();

// Time `launch` (which must enqueue exactly one iteration's work on the default
// stream) with CUDA events: `warmup` untimed reps, then `iters` timed reps.
Result run(const std::string& kernel, const std::string& dims,
           const std::function<void()>& launch, int warmup = 10, int iters = 50,
           const std::string& gemm_path = "n/a");

// Write the result as bench/<kernel>_<dims>_<timestamp>.json, stamped with
// commit hash, dirty flag, and toolkit version. Returns the path written.
std::string write_record(const Result& r, const std::string& out_dir = "bench");

} // namespace bench
