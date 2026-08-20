cmake_minimum_required(VERSION 3.24)
project(llm-inference-engine LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# ---------------------------------------------------------------- CPU targets
# These must always build on the Mac (no CUDA anywhere in their include graph —
# that's the abstraction boundary; see plans/stage-00-setup.md).

add_executable(test_f16 tests/test_f16.cpp)
enable_testing()
add_test(NAME f16 COMMAND test_f16)

# ---------------------------------------------------------------- CUDA targets
# Only configured where a CUDA compiler exists (the WSL2 box). sm_120 SASS for
# the 5070 Ti plus PTX fallback ("120-real;120-virtual") so the binary can JIT
# on a different card (rented A100/H100) instead of failing to load.

include(CheckLanguage)
check_language(CUDA)
if(CMAKE_CUDA_COMPILER)
  enable_language(CUDA)
  set(CMAKE_CUDA_ARCHITECTURES "120-real;120-virtual")
  set(CMAKE_CUDA_STANDARD 20)

  find_package(CUDAToolkit REQUIRED)   # for NVML (nvidia-ml) used by the harness

  add_library(bench_harness STATIC kernels/common/bench.cu)
  target_link_libraries(bench_harness PUBLIC CUDA::nvml)

  add_executable(warmup_bench kernels/warmup/warmup_bench.cu)
  target_link_libraries(warmup_bench PRIVATE bench_harness)
else()
  message(STATUS "No CUDA compiler — building CPU-only targets (expected on the Mac)")
endif()
