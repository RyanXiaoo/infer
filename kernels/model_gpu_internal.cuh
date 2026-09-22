// model_gpu_internal.cuh — device-side helpers shared by the GPU forward pass
// (model_gpu.cu) and the GPU KV-cache session (session_gpu.cu).
//
// CUDA-facing (.cu includers only). Holds the RAII device buffer, the uploaded
// weight representation, the per-layer weight bundle, the GpuModel::Impl
// definition (device weights + linear dispatch), and the host-side RoPE tables.

#pragma once

#include "model_gpu.h"
#include "common/cuda_check.cuh"
#include "ops/ops.cuh"
#include "../src/quant.h"

#include <cublas_v2.h>

#include <cmath>
#include <stdexcept>
#include <cstdio>
#include <cstdlib>
#include <vector>

#define CUBLAS_CHECK(call)                                                      \
    do {                                                                        \
        cublasStatus_t st_ = (call);                                            \
        if (st_ != CUBLAS_STATUS_SUCCESS) {                                     \
            std::fprintf(stderr, "cuBLAS error %d at %s:%d (%s)\n", int(st_),   \
                         __FILE__, __LINE__, #call);                            \
            std::exit(1);                                                       \
        }                                                                       \
    } while (0)

namespace llm {

// RAII device buffer.
struct DevBuf {
    void* p = nullptr;
    DevBuf() = default;
    explicit DevBuf(size_t bytes) { CUDA_CHECK(cudaMalloc(&p, bytes)); }
    ~DevBuf() { if (p) cudaFree(p); }
    DevBuf(DevBuf&& o) noexcept : p(o.p) { o.p = nullptr; }
    DevBuf& operator=(DevBuf&& o) noexcept {
        if (p) cudaFree(p);
        p = o.p;
        o.p = nullptr;
        return *this;
    }
    DevBuf(const DevBuf&) = delete;
    DevBuf& operator=(const DevBuf&) = delete;
    float* f() const { return static_cast<float*>(p); }
    __nv_bfloat16* bf() const { return static_cast<__nv_bfloat16*>(p); }
    int64_t* i64() const { return static_cast<int64_t*>(p); }
};

// One uploaded weight: bf16 always; fp32 mirror built on first cuBLAS use.
struct DevTensor {
    DevBuf bf16;
    const __nv_bfloat16* view = nullptr;   // set instead of bf16 by upload_into
    DevBuf f32;          // empty until ensure_f32
    int64_t numel = 0;
    // Stage 10: quantised matrices live here instead of in bf16.
    bool quantized = false;
    DevBuf qdata, qscales, qzeros;
    gpu::QuantView qview{};

    void upload_q(const QTensor& t) {
        quantized = true;
        numel = t.rows * t.cols;
        const size_t nq = t.q_bytes(), ns = size_t(t.rows) * t.groups();
        qdata = DevBuf(nq);
        CUDA_CHECK(cudaMemcpy(qdata.p, t.q, nq, cudaMemcpyHostToDevice));
        qscales = DevBuf(ns * 2);
        CUDA_CHECK(cudaMemcpy(qscales.p, t.scales, ns * 2, cudaMemcpyHostToDevice));
        if (t.zeros) {
            qzeros = DevBuf(ns);
            CUDA_CHECK(cudaMemcpy(qzeros.p, t.zeros, ns, cudaMemcpyHostToDevice));
        }
        qview = gpu::QuantView{t.kind == QKind::kInt4 ? gpu::QuantKind::kInt4 : gpu::QuantKind::kInt8,
                               t.rows, t.cols, t.groups(), static_cast<const uint8_t*>(qdata.p),
                               static_cast<const uint16_t*>(qscales.p),
                               t.zeros ? static_cast<const uint8_t*>(qzeros.p) : nullptr};
    }
    // bf16 vector (norm, bias) from a .llmq file.
    void upload_bf16_bits(const uint16_t* data, int64_t n) {
        numel = n;
        bf16 = DevBuf(size_t(n) * 2);
        CUDA_CHECK(cudaMemcpy(bf16.p, data, size_t(n) * 2, cudaMemcpyHostToDevice));
    }

    // Upload into a slice of a buffer owned by someone else. Used so q|k|v and
    // gate|up sit back to back in device memory: row-major [out, in] matrices
    // with the same `in` concatenate into one [sum(out), in] matrix, which the
    // row-parallel GEMV (one block per output row) serves in a single launch.
    void upload_into(const Tensor* t, __nv_bfloat16* dst) {
        numel = t->numel();
        CUDA_CHECK(cudaMemcpy(dst, t->u16(), size_t(numel) * 2, cudaMemcpyHostToDevice));
        view = dst;
    }

    void upload(const Tensor* t) {
        numel = t->numel();
        bf16 = DevBuf(size_t(numel) * 2);
        CUDA_CHECK(cudaMemcpy(bf16.p, t->u16(), size_t(numel) * 2,
                              cudaMemcpyHostToDevice));
    }
    const __nv_bfloat16* bf() const { return view ? view : bf16.bf(); }
    const float* ensure_f32() {
        if (quantized) throw std::runtime_error("cuBLAS path needs bf16 weights (gemm=mine for quantised models)");
        if (!f32.p) {
            f32 = DevBuf(size_t(numel) * 4);
            gpu::launch_bf16_to_f32(bf(), numel, f32.f());
        }
        return f32.f();
    }
};

struct DevLayer {
    DevTensor q_w, q_b, k_w, k_b, v_w, v_b, o_w;
    DevTensor gate_w, up_w, down_w;
    DevTensor input_ln, post_attn_ln;
    bool has_bias = false;
    // Backing storage for the views: [q|k|v] weights, [q|k|v] biases, [gate|up].
    // q_w.bf() / q_b.bf() / gate_w.bf() double as the fused matrices' bases.
    DevBuf qkv_w_buf, qkv_b_buf, gate_up_buf;
};

// Host-side RoPE tables (duplicated-halves layout, fp32 inv_freq) for absolute
// positions pos0..pos0+T. The full pass passes pos0=0; decode passes the
// token's true position so cached keys are rotated correctly.
inline void rope_tables_host(double theta, int64_t hd, int64_t pos0, int64_t T,
                             std::vector<float>& c, std::vector<float>& s) {
    const int64_t half = hd / 2;
    std::vector<float> inv_freq(half);
    for (int64_t j = 0; j < half; j++)
        inv_freq[j] = 1.0f / std::pow(float(theta), float(2 * j) / float(hd));
    c.resize(T * hd);
    s.resize(T * hd);
    for (int64_t t = 0; t < T; t++)
        for (int64_t j = 0; j < half; j++) {
            float a = float(pos0 + t) * inv_freq[j];
            c[t * hd + j] = c[t * hd + half + j] = std::cos(a);
            s[t * hd + j] = s[t * hd + half + j] = std::sin(a);
        }
}

struct GpuModel::Impl {
    ModelConfig cfg;
    DevTensor embed_tokens, final_norm;
    DevTensor lm_head;                       // untied models only
    bool untied = false;
    DevTensor& head() { return untied ? lm_head : embed_tokens; }
    std::vector<DevLayer> layers;
    cublasHandle_t cublas = nullptr;

    // linear dispatch: mine = row-parallel GEMV at T=1 (naive kernel for T>1),
    // naive = the Stage 3 kernel always, cublas = Sgemm on fp32 mirrors.
    void linear(GemmPath path, const float* x, DevTensor& W, DevTensor* b,
                int64_t T, int64_t in, int64_t out, float* y) {
        if (W.quantized) throw std::runtime_error("this path has no quantised kernels; use GpuBatch");
        if (path == GemmPath::kMineNaive) {
            gpu::launch_linear_mine(x, W.bf(), b ? b->bf() : nullptr, T, in, out, y);
            return;
        }
        if (path == GemmPath::kMine) {
            gpu::launch_gemv_rowpar(x, W.bf(), b ? b->bf() : nullptr, T, in, out, y);
            return;
        }
        // Row-major y[T,out] = x[T,in] * W^T. In cuBLAS column-major terms:
        // W row-major [out,in] IS W^T column-major [in,out]; y_cm[out,T].
        const float* Wf = W.ensure_f32();
        const float alpha = 1.0f, beta = 0.0f;
        CUBLAS_CHECK(cublasSgemm(cublas, CUBLAS_OP_T, CUBLAS_OP_N, int(out), int(T),
                                 int(in), &alpha, Wf, int(in), x, int(in), &beta, y,
                                 int(out)));
        if (b) gpu::launch_add_bias(b->bf(), T, out, y);
    }
};

} // namespace llm
