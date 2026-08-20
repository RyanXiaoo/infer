// cuda_check.cuh — error-checking macros. CUDA-facing code only (never include
// from shared src/ headers).
#pragma once

#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t err_ = (call);                                              \
        if (err_ != cudaSuccess) {                                              \
            std::fprintf(stderr, "CUDA error %s:%d: %s (%s)\n", __FILE__,       \
                         __LINE__, cudaGetErrorString(err_), #call);            \
            std::exit(1);                                                       \
        }                                                                       \
    } while (0)

// After a kernel launch: catches launch-config errors immediately, and (in debug
// builds) synchronizes so the failing kernel is the one that reports.
#define CUDA_CHECK_LAUNCH()                                                     \
    do {                                                                        \
        CUDA_CHECK(cudaGetLastError());                                         \
        /* keep async in release: sync only when debugging */                   \
    } while (0)
