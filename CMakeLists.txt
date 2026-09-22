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
  src/forward.cpp
  src/session.cpp
  src/tokenizer.cpp
  src/sampler.cpp
  src/scheduler.cpp
  src/block_pool.cpp
  src/serve.cpp
  src/quant.cpp)
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

add_executable(test_kv_cache tests/test_kv_cache.cpp)
target_link_libraries(test_kv_cache PRIVATE engine_core)
target_compile_definitions(test_kv_cache PRIVATE MODEL_ROOT="${CMAKE_SOURCE_DIR}")
add_test(NAME kv_cache COMMAND test_kv_cache)

add_executable(test_tokenizer tests/test_tokenizer.cpp)
target_link_libraries(test_tokenizer PRIVATE engine_core)
target_compile_definitions(test_tokenizer PRIVATE MODEL_ROOT="${CMAKE_SOURCE_DIR}")
add_test(NAME tokenizer COMMAND test_tokenizer)

add_executable(test_serve tests/test_serve.cpp)
target_link_libraries(test_serve PRIVATE engine_core)
add_test(NAME serve COMMAND test_serve)

add_executable(test_quant tests/test_quant.cpp)
target_link_libraries(test_quant PRIVATE engine_core)
add_test(NAME quant COMMAND test_quant)

add_executable(quantize tools/quantize.cpp)
target_link_libraries(quantize PRIVATE engine_core)

add_executable(test_block_pool tests/test_block_pool.cpp)
target_link_libraries(test_block_pool PRIVATE engine_core)
add_test(NAME block_pool COMMAND test_block_pool)

add_executable(test_scheduler tests/test_scheduler.cpp)
target_link_libraries(test_scheduler PRIVATE engine_core)
add_test(NAME scheduler COMMAND test_scheduler)

add_executable(test_sampler tests/test_sampler.cpp)
target_link_libraries(test_sampler PRIVATE engine_core)
add_test(NAME sampler COMMAND test_sampler)

add_executable(main_generate src/main_generate.cpp)
target_link_libraries(main_generate PRIVATE engine_core)
target_compile_definitions(main_generate PRIVATE MODEL_ROOT="${CMAKE_SOURCE_DIR}")

add_executable(main_chat src/main_chat.cpp)
target_link_libraries(main_chat PRIVATE engine_core)
target_compile_definitions(main_chat PRIVATE MODEL_ROOT="${CMAKE_SOURCE_DIR}")

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
  # Per-thread default stream: lets GpuBatch capture its plain <<<>>> launch
  # sequence into a CUDA graph (the legacy default stream cannot be captured).
  # cuBLAS is pointed at the same stream in GpuModel's constructor.
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --default-stream per-thread")

  find_package(CUDAToolkit REQUIRED)   # for NVML (nvidia-ml) used by the harness

  add_library(bench_harness STATIC kernels/common/bench.cu)
  target_link_libraries(bench_harness PUBLIC CUDA::nvml)

  add_executable(warmup_bench kernels/warmup/warmup_bench.cu)
  target_link_libraries(warmup_bench PRIVATE bench_harness)

  # Stage 3: the GPU forward pass (naive kernels + cuBLAS bisection path).
  add_library(engine_gpu STATIC
    kernels/model_gpu.cu
    kernels/session_gpu.cu
    kernels/ops/embedding.cu
    kernels/ops/rmsnorm.cu
    kernels/ops/linear.cu
    kernels/ops/rope.cu
    kernels/ops/attention.cu
    kernels/ops/elementwise.cu
    kernels/ops/cache.cu
    kernels/ops/batch.cu
    kernels/ops/gemm.cu
    kernels/ops/quant.cu
    kernels/batch_gpu.cu)
  target_link_libraries(engine_gpu PUBLIC engine_core CUDA::cudart CUDA::cublas)

  # Stage 5: per-shape decode GEMV microbench (mine vs cuBLAS, GB/s vs achievable).
  add_executable(gemv_bench kernels/bench/gemv_bench.cu)
  target_link_libraries(gemv_bench PRIVATE bench_harness engine_gpu)

  # Stage 5: measured achievable memory bandwidth (the GEMV tables' 100% mark).
  add_executable(stream_bench kernels/bench/stream_bench.cu)
  target_link_libraries(stream_bench PRIVATE bench_harness)

  # Stage 5: single-variant, single-shape driver for ncu counter collection.
  add_executable(profile_one kernels/bench/profile_one.cu)
  target_link_libraries(profile_one PRIVATE engine_gpu)

  # Stage 5: cached decode attention microbench (us/launch vs cache_len, per variant).
  add_executable(attn_bench kernels/bench/attn_bench.cu)
  target_link_libraries(attn_bench PRIVATE bench_harness engine_gpu)

  add_executable(test_forward_gpu tests/test_forward_gpu.cpp)
  target_link_libraries(test_forward_gpu PRIVATE engine_gpu)
  target_compile_definitions(test_forward_gpu PRIVATE MODEL_ROOT="${CMAKE_SOURCE_DIR}")
  add_test(NAME forward_gpu COMMAND test_forward_gpu)

  # Stage 5: per-kernel edge-size sweep vs a host double-precision reference.
  add_executable(test_ops_gpu tests/test_ops_gpu.cu)
  target_link_libraries(test_ops_gpu PRIVATE engine_gpu)
  add_test(NAME ops_gpu COMMAND test_ops_gpu)
  add_test(NAME ops_gpu_selftest COMMAND test_ops_gpu --selftest)

  # Stage 6: batched decode == single-sequence decode, lockstep and staggered.
  add_executable(test_batch_gpu tests/test_batch_gpu.cpp)
  target_link_libraries(test_batch_gpu PRIVATE engine_gpu)
  target_compile_definitions(test_batch_gpu PRIVATE MODEL_ROOT="${CMAKE_SOURCE_DIR}")
  add_test(NAME batch_gpu COMMAND test_batch_gpu)

  # Stage 6: throughput / latency sweep over batch size, continuous vs static.
  add_executable(main_serve_bench src/main_serve_bench.cpp)
  target_link_libraries(main_serve_bench PRIVATE engine_gpu)
  target_compile_definitions(main_serve_bench PRIVATE MODEL_ROOT="${CMAKE_SOURCE_DIR}")

  # Stage 9: HTTP server (cpp-httplib, vendored) over the serving core.
  find_package(Threads REQUIRED)
  add_executable(main_server src/main_server.cpp)
  target_link_libraries(main_server PRIVATE engine_gpu Threads::Threads)
  target_compile_definitions(main_server PRIVATE MODEL_ROOT="${CMAKE_SOURCE_DIR}" CPPHTTPLIB_NO_EXCEPTIONS=0)

  # KV cache test gains its GPU half in CUDA builds.
  target_compile_definitions(test_kv_cache PRIVATE HAVE_GPU)
  target_link_libraries(test_kv_cache PRIVATE engine_gpu)

  # main_generate + main_chat grow the gpu device in CUDA builds only.
  target_compile_definitions(main_generate PRIVATE HAVE_GPU)
  target_link_libraries(main_generate PRIVATE engine_gpu)
  target_compile_definitions(main_chat PRIVATE HAVE_GPU)
  target_link_libraries(main_chat PRIVATE engine_gpu)
else()
  message(STATUS "No CUDA compiler — building CPU-only targets (expected on the Mac)")
endif()
