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

#include <cublas_v2.h>

#include <cmath>
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
    DevBuf f32;          // empty until ensure_f32
    int64_t numel = 0;

    void upload(const Tensor* t) {
        numel = t->numel();
        bf16 = DevBuf(size_t(numel) * 2);
        CUDA_CHECK(cudaMemcpy(bf16.p, t->u16(), size_t(numel) * 2,
                              cudaMemcpyHostToDevice));
    }
    const __nv_bfloat16* bf() const { return bf16.bf(); }
    const float* ensure_f32() {
        if (!f32.p) {
            f32 = DevBuf(size_t(numel) * 4);
            gpu::launch_bf16_to_f32(bf16.bf(), numel, f32.f());
        }
        return f32.f();
    }
};

struct DevLayer {
    DevTensor q_w, q_b, k_w, k_b, v_w, v_b, o_w;
    DevTensor gate_w, up_w, down_w;
    DevTensor input_ln, post_attn_ln;
    bool has_bias = false;
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
    std::vector<DevLayer> layers;
    cublasHandle_t cublas = nullptr;

    // linear dispatch: mine = the naive kernel; cublas = Sgemm on fp32 mirrors.
    void linear(GemmPath path, const float* x, DevTensor& W, DevTensor* b,
                int64_t T, int64_t in, int64_t out, float* y) {
        if (path == GemmPath::kMine) {
            gpu::launch_linear_mine(x, W.bf(), b ? b->bf() : nullptr, T, in, out, y);
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
