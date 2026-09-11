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

add_library(engine_core STATIC
  src/loader.cpp
  src/model_config.cpp
  src/npy.cpp
  src/model.cpp
  src/forward.cpp)
target_include_directories(engine_core PUBLIC third_party)

add_executable(test_loader tests/test_loader.cpp)
target_link_libraries(test_loader PRIVATE engine_core)
target_compile_definitions(test_loader PRIVATE MODEL_ROOT="${CMAKE_SOURCE_DIR}")
add_test(NAME loader COMMAND test_loader)

add_executable(test_npy tests/test_npy.cpp)
target_link_libraries(test_npy PRIVATE engine_core)
target_compile_definitions(test_npy PRIVATE MODEL_ROOT="${CMAKE_SOURCE_DIR}")
add_test(NAME npy COMMAND test_npy)

add_executable(test_forward tests/test_forward.cpp)
target_link_libraries(test_forward PRIVATE engine_core)
target_compile_definitions(test_forward PRIVATE MODEL_ROOT="${CMAKE_SOURCE_DIR}")
add_test(NAME forward COMMAND test_forward)

add_executable(main_generate src/main_generate.cpp)
target_link_libraries(main_generate PRIVATE engine_core)
target_compile_definitions(main_generate PRIVATE MODEL_ROOT="${CMAKE_SOURCE_DIR}")

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
