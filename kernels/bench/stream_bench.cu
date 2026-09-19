// stream_bench.cu — measured achievable memory bandwidth (STREAM-style).
//
// Decode GEMV is bound by how fast weights can be read out of device memory,
// so "how good is my kernel" needs a denominator measured on THIS card at THIS
// memory clock, not the spec sheet (896 GB/s for the 5070 Ti) and not Stage 0's
// vector_add (which mixes reads and writes and ran at a different clock).
//
// The four classic STREAM kernels (fp32, arrays far larger than L2), counting
// every byte moved, plus two read-dominated kernels shaped like what GEMV
// actually does: each thread reads K consecutive elements and writes one sum.
//   copy   c = a            1 read + 1 write per element
//   scale  b = s*c          1 + 1
//   add    c = a + b        2 + 1
//   triad  a = b + s*c      2 + 1
//   read_f32 / read_bf16    K reads + 1 write per thread (K = 64)
// The best GB/s across them is the number the GEMV tables use as 100%.

#include "../common/bench.cuh"
#include "../common/cuda_check.cuh"

#include <cuda_bf16.h>

#include <algorithm>
#include <cstdio>
#include <string>
#include <vector>

namespace {

constexpr int64_t kN = int64_t(1) << 26;   // 64M elements: 256 MB per fp32 array
constexpr int64_t kK = 64;                 // elements summed per thread in the read tests

__global__ void k_copy(const float* a, float* c, int64_t n) {
    int64_t i = blockIdx.x * int64_t(blockDim.x) + threadIdx.x;
    if (i < n) c[i] = a[i];
}
__global__ void k_scale(float* b, const float* c, float s, int64_t n) {
    int64_t i = blockIdx.x * int64_t(blockDim.x) + threadIdx.x;
    if (i < n) b[i] = s * c[i];
}
__global__ void k_add(const float* a, const float* b, float* c, int64_t n) {
    int64_t i = blockIdx.x * int64_t(blockDim.x) + threadIdx.x;
    if (i < n) c[i] = a[i] + b[i];
}
__global__ void k_triad(float* a, const float* b, const float* c, float s, int64_t n) {
    int64_t i = blockIdx.x * int64_t(blockDim.x) + threadIdx.x;
    if (i < n) a[i] = b[i] + s * c[i];
}
template <typename T>
__global__ void k_read(const T* a, float* sums, int64_t n_sums) {
    int64_t i = blockIdx.x * int64_t(blockDim.x) + threadIdx.x;
    if (i >= n_sums) return;
    const T* p = a + i * kK;
    float acc = 0.0f;
    for (int64_t j = 0; j < kK; j++) acc += float(p[j]);
    sums[i] = acc;
}

struct Row { const char* name; double bytes; double ms; };

} // namespace

int main() {
    float *a = nullptr, *b = nullptr, *c = nullptr;
    CUDA_CHECK(cudaMalloc(&a, kN * 4));
    CUDA_CHECK(cudaMalloc(&b, kN * 4));
    CUDA_CHECK(cudaMalloc(&c, kN * 4));
    std::vector<float> ones(kN, 1.0f);
    for (float* p : {a, b, c})
        CUDA_CHECK(cudaMemcpy(p, ones.data(), kN * 4, cudaMemcpyHostToDevice));
    // The bf16 read test reinterprets `a`'s bytes (0x3F80 / 0x0000 patterns: finite).
    const auto* a_bf16 = reinterpret_cast<const __nv_bfloat16*>(a);
    const int64_t n_sums = kN / kK;
    const int64_t grid = (kN + 255) / 256, grid_sums = (n_sums + 255) / 256;
    const std::string dims = "n64M";

    std::vector<Row> rows;
    auto time = [&](const char* name, double bytes, const std::function<void()>& launch) {
        bench::Result r = bench::run(std::string("stream_") + name, dims, launch);
        bench::write_record(r);
        rows.push_back({name, bytes, r.median_ms});
    };
    time("copy", 2.0 * kN * 4, [&] { k_copy<<<grid, 256>>>(a, c, kN); CUDA_CHECK_LAUNCH(); });
    time("scale", 2.0 * kN * 4, [&] { k_scale<<<grid, 256>>>(b, c, 3.0f, kN); CUDA_CHECK_LAUNCH(); });
    time("add", 3.0 * kN * 4, [&] { k_add<<<grid, 256>>>(a, b, c, kN); CUDA_CHECK_LAUNCH(); });
    time("triad", 3.0 * kN * 4, [&] { k_triad<<<grid, 256>>>(a, b, c, 3.0f, kN); CUDA_CHECK_LAUNCH(); });
    time("read_f32", kN * 4.0 + n_sums * 4.0,
         [&] { k_read<float><<<grid_sums, 256>>>(a, c, n_sums); CUDA_CHECK_LAUNCH(); });
    // bf16: the same 256 MB buffer holds 2x as many 2-byte elements.
    time("read_bf16", kN * 4.0 + 2 * n_sums * 4.0, [&] {
        k_read<__nv_bfloat16><<<2 * grid_sums, 256>>>(a_bf16, c, 2 * n_sums);
        CUDA_CHECK_LAUNCH();
    });

    bench::GpuState st = bench::query_gpu_state();
    std::printf("\nSTREAM-style bandwidth, 256 MB arrays (SM %u MHz, mem %u MHz, %u C)\n",
                st.sm_clock_mhz, st.mem_clock_mhz, st.temp_c);
    double best = 0;
    for (const Row& r : rows) {
        const double g = r.bytes / (r.ms / 1e3) / 1e9;
        best = std::max(best, g);
        std::printf("  %-10s %8.3f ms  %7.1f GB/s\n", r.name, r.ms, g);
    }
    std::printf("achievable (best of the above): %.0f GB/s\n", best);

    for (float* p : {a, b, c}) CUDA_CHECK(cudaFree(p));
    return 0;
}
