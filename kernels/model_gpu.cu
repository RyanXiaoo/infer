// model_gpu.cu — weight upload + the GPU forward pass (Stage 3).
//
// A line-for-line mirror of src/forward.cpp: same op order, same tap names and
// shapes, same "bf16 weights, fp32 math" semantics — so the same golden ladder
// validates both. Everything stays on-device between the token-id upload and
// the logits download; taps (test-only) copy intermediates back per checkpoint.
//
// cuBLAS path: fp32 mirrors of the weight matrices are built lazily on first
// kCublas use (~2x weight memory, fine in 16GB) and cublasSgemm serves every
// linear. Numerically it upcasts the same bf16 values, so golden tolerances
// hold on both paths.

#include "model_gpu.h"
#include "common/cuda_check.cuh"
#include "ops/ops.cuh"

#include <cublas_v2.h>

#include <cmath>
#include <cstring>
#include <map>
#include <stdexcept>
#include <string>

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

namespace {

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

// Host-side RoPE tables — same code as the CPU path (duplicated-halves layout,
// fp32 inv_freq), uploaded per forward.
void rope_tables_host(double theta, int64_t hd, int64_t T, std::vector<float>& c,
                      std::vector<float>& s) {
    const int64_t half = hd / 2;
    std::vector<float> inv_freq(half);
    for (int64_t j = 0; j < half; j++)
        inv_freq[j] = 1.0f / std::pow(float(theta), float(2 * j) / float(hd));
    c.resize(T * hd);
    s.resize(T * hd);
    for (int64_t t = 0; t < T; t++)
        for (int64_t j = 0; j < half; j++) {
            float a = float(t) * inv_freq[j];
            c[t * hd + j] = c[t * hd + half + j] = std::cos(a);
            s[t * hd + j] = s[t * hd + half + j] = std::sin(a);
        }
}

} // namespace

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
        // W row-major [out,in] IS W^T column-major [in,out], x row-major [T,in]
        // IS x^T column-major [in,T]; y_cm[out,T] = (W_cm)^T * x_cm.
        const float* Wf = W.ensure_f32();
        const float alpha = 1.0f, beta = 0.0f;
        CUBLAS_CHECK(cublasSgemm(cublas, CUBLAS_OP_T, CUBLAS_OP_N, int(out), int(T),
                                 int(in), &alpha, Wf, int(in), x, int(in), &beta, y,
                                 int(out)));
        if (b) gpu::launch_add_bias(b->bf(), T, out, y);
    }
};

GpuModel::GpuModel(const Model& m) : impl_(new Impl) {
    impl_->cfg = m.cfg;
    impl_->embed_tokens.upload(m.embed_tokens);
    impl_->final_norm.upload(m.final_norm);
    impl_->layers.resize(m.layers.size());
    for (size_t i = 0; i < m.layers.size(); i++) {
        const LayerWeights& L = m.layers[i];
        DevLayer& D = impl_->layers[i];
        D.q_w.upload(L.q_w);
        D.k_w.upload(L.k_w);
        D.v_w.upload(L.v_w);
        D.o_w.upload(L.o_w);
        D.gate_w.upload(L.gate_w);
        D.up_w.upload(L.up_w);
        D.down_w.upload(L.down_w);
        D.input_ln.upload(L.input_ln);
        D.post_attn_ln.upload(L.post_attn_ln);
        D.has_bias = L.q_b != nullptr;
        if (D.has_bias) {
            D.q_b.upload(L.q_b);
            D.k_b.upload(L.k_b);
            D.v_b.upload(L.v_b);
        }
    }
    CUBLAS_CHECK(cublasCreate(&impl_->cublas));
    CUDA_CHECK(cudaDeviceSynchronize());
}

GpuModel::~GpuModel() {
    if (impl_ && impl_->cublas) cublasDestroy(impl_->cublas);
}

const ModelConfig& GpuModel::cfg() const { return impl_->cfg; }

namespace {

// Tap helpers: copy a device buffer back and hand it to the hook. Test-only
// path — when no hook is installed these are never called.
void tap_dev(const TapFn* fn, const std::string& name, const float* dev, int64_t n,
             std::vector<int64_t> shape) {
    if (!fn || !*fn) return;
    std::vector<float> h(n);
    CUDA_CHECK(cudaMemcpy(h.data(), dev, size_t(n) * 4, cudaMemcpyDeviceToHost));
    (*fn)(name, h.data(), shape);
}

// Post-RoPE Q/K goldens use HF's [n_heads, T, hd] view; ours is [T, n_heads*hd].
void tap_dev_heads(const TapFn* fn, const std::string& name, const float* dev,
                   int64_t T, int64_t n_heads, int64_t hd) {
    if (!fn || !*fn) return;
    std::vector<float> h(T * n_heads * hd), tr(n_heads * T * hd);
    CUDA_CHECK(cudaMemcpy(h.data(), dev, h.size() * 4, cudaMemcpyDeviceToHost));
    for (int64_t hh = 0; hh < n_heads; hh++)
        for (int64_t t = 0; t < T; t++)
            for (int64_t d = 0; d < hd; d++)
                tr[(hh * T + t) * hd + d] = h[t * n_heads * hd + hh * hd + d];
    (*fn)(name, tr.data(), {n_heads, T, hd});
}

} // namespace

std::vector<float> forward_gpu(GpuModel& gm, const std::vector<int64_t>& ids,
                               const TapFn* taps, bool last_only, GemmPath gemm) {
    GpuModel::Impl& M = *gm.impl_;
    const auto& cfg = M.cfg;
    const int64_t T = int64_t(ids.size());
    const int64_t H = cfg.hidden_size, hd = cfg.head_dim;
    const int64_t nh = cfg.num_attention_heads, nkv = cfg.num_key_value_heads;
    const int64_t q_out = nh * hd, kv_out = nkv * hd;
    const int64_t I = cfg.intermediate_size, V = cfg.vocab_size;

    for (int64_t id : ids)
        if (id < 0 || id >= V)
            throw std::runtime_error("token id out of range: " + std::to_string(id));

    // Device scratch (allocated per call — naive stage; pooling comes later).
    DevBuf d_ids(size_t(T) * 8);
    CUDA_CHECK(cudaMemcpy(d_ids.p, ids.data(), size_t(T) * 8, cudaMemcpyHostToDevice));
    DevBuf h_(T * H * 4), normed(T * H * 4);
    DevBuf q(T * q_out * 4), k(T * kv_out * 4), v(T * kv_out * 4);
    DevBuf ctx(T * q_out * 4), delta(T * H * 4);
    DevBuf gate(T * I * 4), up(T * I * 4);
    DevBuf scores(size_t(nh) * T * T * 4);
    DevBuf d_cos(T * hd * 4), d_sin(T * hd * 4);

    gpu::launch_embedding(M.embed_tokens.bf(), d_ids.i64(), T, H, h_.f());
    tap_dev(taps, "hidden_state_0", h_.f(), T * H, {1, T, H});

    std::vector<float> cos_h, sin_h;
    rope_tables_host(cfg.rope_theta, hd, T, cos_h, sin_h);
    CUDA_CHECK(cudaMemcpy(d_cos.p, cos_h.data(), cos_h.size() * 4, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_sin.p, sin_h.data(), sin_h.size() * 4, cudaMemcpyHostToDevice));
    if (taps && *taps) {
        (*taps)("rope_cos", cos_h.data(), {1, T, hd});
        (*taps)("rope_sin", sin_h.data(), {1, T, hd});
    }

    const float eps = float(cfg.rms_norm_eps);
    for (int64_t li = 0; li < cfg.num_hidden_layers; li++) {
        DevLayer& L = M.layers[li];
        const std::string p = "layer" + std::to_string(li) + ".";

        gpu::launch_rmsnorm(h_.f(), L.input_ln.bf(), eps, T, H, normed.f());
        tap_dev(taps, p + "post_input_layernorm", normed.f(), T * H, {1, T, H});

        M.linear(gemm, normed.f(), L.q_w, L.has_bias ? &L.q_b : nullptr, T, H, q_out, q.f());
        M.linear(gemm, normed.f(), L.k_w, L.has_bias ? &L.k_b : nullptr, T, H, kv_out, k.f());
        M.linear(gemm, normed.f(), L.v_w, L.has_bias ? &L.v_b : nullptr, T, H, kv_out, v.f());
        tap_dev(taps, p + "q_proj", q.f(), T * q_out, {1, T, q_out});
        tap_dev(taps, p + "k_proj", k.f(), T * kv_out, {1, T, kv_out});
        tap_dev(taps, p + "v_proj", v.f(), T * kv_out, {1, T, kv_out});

        gpu::launch_rope(q.f(), d_cos.f(), d_sin.f(), T, nh, hd);
        gpu::launch_rope(k.f(), d_cos.f(), d_sin.f(), T, nkv, hd);
        tap_dev_heads(taps, p + "q_post_rope", q.f(), T, nh, hd);
        tap_dev_heads(taps, p + "k_post_rope", k.f(), T, nkv, hd);

        gpu::launch_attention(q.f(), k.f(), v.f(), scores.f(), T, nh, nkv, hd, ctx.f());
        tap_dev(taps, p + "attn_out_pre_o_proj", ctx.f(), T * q_out, {1, T, q_out});

        M.linear(gemm, ctx.f(), L.o_w, nullptr, T, q_out, H, delta.f());
        gpu::launch_residual_add(h_.f(), delta.f(), T * H);
        tap_dev(taps, p + "post_attn_residual", h_.f(), T * H, {1, T, H});

        gpu::launch_rmsnorm(h_.f(), L.post_attn_ln.bf(), eps, T, H, normed.f());
        tap_dev(taps, p + "post_post_attention_layernorm", normed.f(), T * H, {1, T, H});

        M.linear(gemm, normed.f(), L.gate_w, nullptr, T, H, I, gate.f());
        M.linear(gemm, normed.f(), L.up_w, nullptr, T, H, I, up.f());
        tap_dev(taps, p + "mlp_gate", gate.f(), T * I, {1, T, I});
        tap_dev(taps, p + "mlp_up", up.f(), T * I, {1, T, I});
        gpu::launch_swiglu(gate.f(), up.f(), T * I);
        M.linear(gemm, gate.f(), L.down_w, nullptr, T, I, H, delta.f());
        tap_dev(taps, p + "mlp_down", delta.f(), T * H, {1, T, H});
        gpu::launch_residual_add(h_.f(), delta.f(), T * H);

        if (li + 1 < cfg.num_hidden_layers)
            tap_dev(taps, "hidden_state_" + std::to_string(li + 1), h_.f(), T * H,
                    {1, T, H});
    }

    gpu::launch_rmsnorm(h_.f(), M.final_norm.bf(), eps, T, H, normed.f());
    tap_dev(taps, "hidden_state_" + std::to_string(cfg.num_hidden_layers), normed.f(),
            T * H, {1, T, H});

    // Tied LM head: logits = final_hidden @ embed_tokens^T.
    const int64_t t0 = last_only ? T - 1 : 0;
    const int64_t rows = T - t0;
    DevBuf d_logits(size_t(rows) * V * 4);
    M.linear(gemm, normed.f() + t0 * H, M.embed_tokens, nullptr, rows, H, V,
             d_logits.f());

    std::vector<float> logits(size_t(rows) * V);
    CUDA_CHECK(cudaMemcpy(logits.data(), d_logits.p, logits.size() * 4,
                          cudaMemcpyDeviceToHost));
    if (!last_only && taps && *taps) (*taps)("logits", logits.data(), {1, T, V});
    return logits;
}

std::vector<int64_t> greedy_decode_gpu(GpuModel& gm, std::vector<int64_t> ids,
                                       int n_new, GemmPath gemm) {
    std::vector<int64_t> out;
    const int64_t V = gm.cfg().vocab_size;
    for (int i = 0; i < n_new; i++) {
        std::vector<float> logits = forward_gpu(gm, ids, nullptr, true, gemm);
        int64_t best = 0;
        for (int64_t v = 1; v < V; v++)
            if (logits[v] > logits[best]) best = v;
        out.push_back(best);
        ids.push_back(best);
    }
    return out;
}

} // namespace llm
