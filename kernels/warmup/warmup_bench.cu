// warmup_bench.cu — Stage 0 throwaway kernels, real harness.
//
// Three kernels to learn the execution/memory model on: vector add, naive matmul
// (one thread per output element, all traffic to global memory), and tiled matmul
// (32x32 shared-memory tiles). fp32 everywhere — these never see model weights.
//
// The interesting output is the L2 crossover sweep: at sizes where the working
// set (3*N^2*4 bytes) fits in L2, naive and tiled show nearly identical DRAM
// traffic (L2 absorbs naive's redundant reads) and the difference lives in
// L2->SM traffic. Past L2, the DRAM-bytes gap opens. Sizes below bracket the
// 5070 Ti's L2.

#include "../common/bench.cuh"
#include "../common/cuda_check.cuh"

#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

constexpr int TILE = 32;

__global__ void vector_add(const float* a, const float* b, float* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) c[i] = a[i] + b[i];
}

// C[i,j] = sum_k A[i,k] * B[k,j]; row-major. Each thread reads a full row of A
// and column of B from global memory — N reads of each, no reuse across threads
// except what L2 catches.
__global__ void matmul_naive(const float* A, const float* B, float* C, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= N || col >= N) return;
    float acc = 0.0f;
    for (int k = 0; k < N; k++) acc += A[row * N + k] * B[k * N + col];
    C[row * N + col] = acc;
}

// Same math, but each 32x32 block stages tiles of A and B through shared memory:
// every loaded value is reused 32 times instead of once. Global traffic drops by
// ~TILE; the __syncthreads() pair is what racecheck exists to police.
__global__ void matmul_tiled(const float* A, const float* B, float* C, int N) {
    __shared__ float As[TILE][TILE];
    __shared__ float Bs[TILE][TILE];

    int row = blockIdx.y * TILE + threadIdx.y;
    int col = blockIdx.x * TILE + threadIdx.x;
    float acc = 0.0f;

    for (int t = 0; t < N; t += TILE) {
        As[threadIdx.y][threadIdx.x] = (row < N && t + threadIdx.x < N)
                                           ? A[row * N + t + threadIdx.x] : 0.0f;
        Bs[threadIdx.y][threadIdx.x] = (col < N && t + threadIdx.y < N)
                                           ? B[(t + threadIdx.y) * N + col] : 0.0f;
        __syncthreads();
        for (int k = 0; k < TILE; k++) acc += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        __syncthreads();
    }
    if (row < N && col < N) C[row * N + col] = acc;
}

// ---------------------------------------------------------------- validation

// Full CPU reference at small N; at large N checking every element would take
// minutes on CPU, so we spot-check: random output elements recomputed as single
// dot products (cheap regardless of N).
static void spot_check(const std::vector<float>& A, const std::vector<float>& B,
                       const std::vector<float>& C, int N, int samples,
                       const char* name) {
    std::mt19937 rng(1234);
    std::uniform_int_distribution<int> d(0, N - 1);
    for (int s = 0; s < samples; s++) {
        int i = d(rng), j = d(rng);
        double ref = 0.0;
        for (int k = 0; k < N; k++) ref += (double)A[i * N + k] * B[k * N + j];
        float got = C[i * N + j];
        // fp32 accumulation order differs GPU vs CPU; tolerance scales with N.
        double tol = 1e-5 * N;
        if (std::abs(ref - got) > tol * std::max(1.0, std::abs(ref))) {
            std::fprintf(stderr, "%s N=%d MISMATCH at (%d,%d): ref=%f got=%f\n",
                         name, N, i, j, ref, got);
            std::exit(1);
        }
    }
}

static void bench_matmuls(int N) {
    size_t bytes = (size_t)N * N * sizeof(float);
    std::vector<float> hA((size_t)N * N), hB((size_t)N * N), hC((size_t)N * N);
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (auto& x : hA) x = dist(rng);
    for (auto& x : hB) x = dist(rng);

    float *dA, *dB, *dC;
    CUDA_CHECK(cudaMalloc(&dA, bytes));
    CUDA_CHECK(cudaMalloc(&dB, bytes));
    CUDA_CHECK(cudaMalloc(&dC, bytes));
    CUDA_CHECK(cudaMemcpy(dA, hA.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dB, hB.data(), bytes, cudaMemcpyHostToDevice));

    dim3 block(TILE, TILE);
    dim3 grid((N + TILE - 1) / TILE, (N + TILE - 1) / TILE);
    char dims[32];
    std::snprintf(dims, sizeof(dims), "N%d", N);

    double gflop = 2.0 * N * N * N / 1e9;

    auto report = [&](const bench::Result& r) {
        std::printf("  %-14s %-6s median %8.3f ms  (%7.2f GFLOP/s)  spread [%.3f, %.3f]"
                    "  clk %u->%u MHz  %u°C\n",
                    r.kernel.c_str(), dims, r.median_ms, gflop / (r.median_ms / 1e3),
                    r.min_ms, r.max_ms, r.before.sm_clock_mhz, r.after.sm_clock_mhz,
                    r.after.temp_c);
        std::string p = bench::write_record(r);
        std::printf("    -> %s\n", p.c_str());
    };

    // Naive
    auto r1 = bench::run("matmul_naive", dims, [&] {
        matmul_naive<<<grid, block>>>(dA, dB, dC, N);
        CUDA_CHECK_LAUNCH();
    });
    CUDA_CHECK(cudaMemcpy(hC.data(), dC, bytes, cudaMemcpyDeviceToHost));
    spot_check(hA, hB, hC, N, 32, "matmul_naive");
    report(r1);

    // Tiled
    auto r2 = bench::run("matmul_tiled", dims, [&] {
        matmul_tiled<<<grid, block>>>(dA, dB, dC, N);
        CUDA_CHECK_LAUNCH();
    });
    CUDA_CHECK(cudaMemcpy(hC.data(), dC, bytes, cudaMemcpyDeviceToHost));
    spot_check(hA, hB, hC, N, 32, "matmul_tiled");
    report(r2);

    std::printf("  tiled/naive speedup: %.2fx\n\n", r1.median_ms / r2.median_ms);

    CUDA_CHECK(cudaFree(dA));
    CUDA_CHECK(cudaFree(dB));
    CUDA_CHECK(cudaFree(dC));
}

int main(int argc, char** argv) {
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    std::printf("%s, sm_%d%d, %zu MB global, %d KB L2\n\n", prop.name, prop.major,
                prop.minor, prop.totalGlobalMem >> 20, prop.l2CacheSize >> 10);

    // Vector add: correctness + a first taste of the harness.
    {
        int n = 1 << 24;
        size_t bytes = (size_t)n * sizeof(float);
        std::vector<float> a(n, 1.5f), b(n, 2.25f), c(n);
        float *da, *db, *dc;
        CUDA_CHECK(cudaMalloc(&da, bytes));
        CUDA_CHECK(cudaMalloc(&db, bytes));
        CUDA_CHECK(cudaMalloc(&dc, bytes));
        CUDA_CHECK(cudaMemcpy(da, a.data(), bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(db, b.data(), bytes, cudaMemcpyHostToDevice));
        auto r = bench::run("vector_add", "n16M", [&] {
            vector_add<<<(n + 255) / 256, 256>>>(da, db, dc, n);
            CUDA_CHECK_LAUNCH();
        });
        CUDA_CHECK(cudaMemcpy(c.data(), dc, bytes, cudaMemcpyDeviceToHost));
        for (int i = 0; i < n; i += 999983)
            if (c[i] != 3.75f) { std::fprintf(stderr, "vector_add wrong\n"); return 1; }
        // 3 arrays touched once: effective bandwidth = 3*bytes / time.
        std::printf("  vector_add     n16M   median %8.3f ms  (%7.1f GB/s effective)\n",
                    r.median_ms, 3.0 * bytes / (r.median_ms / 1e3) / 1e9);
        std::printf("    -> %s\n\n", bench::write_record(r).c_str());
        CUDA_CHECK(cudaFree(da));
        CUDA_CHECK(cudaFree(db));
        CUDA_CHECK(cudaFree(dc));
    }

    // L2 crossover sweep. Working set 3*N^2*4B: N=512 -> 3MB (inside L2),
    // N=2048 -> 50MB (~at L2 on this card), N=4096 -> 201MB (well past).
    std::vector<int> sizes = {512, 1024, 2048, 4096};
    if (argc > 1) { sizes = {std::atoi(argv[1])}; }   // single size for ncu runs
    for (int N : sizes) bench_matmuls(N);

    std::printf("done. records in bench/\n");
    return 0;
}
