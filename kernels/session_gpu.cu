// session_gpu.cu — GPU DecodeSession (Stage 4). Device-resident twin of
// src/session.cpp: prefill runs the full forward and fills device caches;
// decode_one runs a T=1 pipeline against them. Reuses the model's device
// weights, linear dispatch, and op launchers via model_gpu_internal.cuh.

#include "session_gpu.h"
#include "model_gpu_internal.cuh"

#include <stdexcept>
#include <vector>

namespace llm {

struct GpuSession::Impl {
    GpuModel& gm;
    GpuModel::Impl& M;
    const int64_t max_seq;
    const GemmPath gemm;
    const AttnPath attn;
    int64_t pos = 0;

    // Per-layer device caches, [max_seq x kv_dim] fp32.
    std::vector<DevBuf> k_cache, v_cache;
    int64_t kv_dim;

    // Reused decode scratch (T=1).
    DevBuf d_id, h_, normed_, q_, k_, v_, ctx_, delta_, gate_, up_, d_cos, d_sin;
    // Decode attention scores, [n_heads x max_seq]. One buffer serves every
    // layer: a layer's scores are dead once its ctx is written.
    DevBuf attn_scores_;

    Impl(GpuModel& g, int64_t ms, GemmPath gm_path, AttnPath attn_path)
        : gm(g), M(*g.impl_), max_seq(ms), gemm(gm_path), attn(attn_path) {
        const auto& c = M.cfg;
        kv_dim = c.num_key_value_heads * c.head_dim;
        const int64_t H = c.hidden_size, hd = c.head_dim, I = c.intermediate_size;
        const int64_t q_out = c.num_attention_heads * hd;
        k_cache.reserve(c.num_hidden_layers);
        v_cache.reserve(c.num_hidden_layers);
        for (int64_t i = 0; i < c.num_hidden_layers; i++) {
            k_cache.emplace_back(size_t(max_seq) * kv_dim * 4);
            v_cache.emplace_back(size_t(max_seq) * kv_dim * 4);
        }
        d_id = DevBuf(8);
        h_ = DevBuf(H * 4);
        normed_ = DevBuf(H * 4);
        q_ = DevBuf(q_out * 4);
        k_ = DevBuf(kv_dim * 4);
        v_ = DevBuf(kv_dim * 4);
        ctx_ = DevBuf(q_out * 4);
        delta_ = DevBuf(H * 4);
        gate_ = DevBuf(I * 4);
        up_ = DevBuf(I * 4);
        d_cos = DevBuf(hd * 4);
        d_sin = DevBuf(hd * 4);
        attn_scores_ = DevBuf(size_t(c.num_attention_heads) * max_seq * 4);
    }

    std::vector<float> prefill(const std::vector<int64_t>& ids) {
        const auto& cfg = M.cfg;
        const int64_t T = int64_t(ids.size());
        const int64_t H = cfg.hidden_size, hd = cfg.head_dim;
        const int64_t nh = cfg.num_attention_heads, nkv = cfg.num_key_value_heads;
        const int64_t q_out = nh * hd, kv_out = nkv * hd, I = cfg.intermediate_size;
        const int64_t V = cfg.vocab_size;
        if (T == 0 || T > max_seq) throw std::runtime_error("prefill: bad prompt length");
        for (int64_t id : ids)
            if (id < 0 || id >= V) throw std::runtime_error("token id out of range");

        DevBuf d_ids(size_t(T) * 8);
        CUDA_CHECK(cudaMemcpy(d_ids.p, ids.data(), size_t(T) * 8, cudaMemcpyHostToDevice));
        DevBuf h(T * H * 4), normed(T * H * 4);
        DevBuf q(T * q_out * 4), k(T * kv_out * 4), v(T * kv_out * 4);
        DevBuf ctx(T * q_out * 4), delta(T * H * 4);
        DevBuf gate(T * I * 4), up(T * I * 4);
        DevBuf scores(size_t(nh) * T * T * 4);
        DevBuf dcos(T * hd * 4), dsin(T * hd * 4);

        gpu::launch_embedding(M.embed_tokens.bf(), d_ids.i64(), T, H, h.f());
        std::vector<float> cos_h, sin_h;
        rope_tables_host(cfg.rope_theta, hd, /*pos0=*/0, T, cos_h, sin_h);
        CUDA_CHECK(cudaMemcpy(dcos.p, cos_h.data(), cos_h.size() * 4, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dsin.p, sin_h.data(), sin_h.size() * 4, cudaMemcpyHostToDevice));

        for (int64_t li = 0; li < cfg.num_hidden_layers; li++) {
            DevLayer& L = M.layers[li];
            gpu::launch_rmsnorm(h.f(), L.input_ln.bf(), float(cfg.rms_norm_eps), T, H, normed.f());
            M.linear(gemm, normed.f(), L.q_w, L.has_bias ? &L.q_b : nullptr, T, H, q_out, q.f());
            M.linear(gemm, normed.f(), L.k_w, L.has_bias ? &L.k_b : nullptr, T, H, kv_out, k.f());
            M.linear(gemm, normed.f(), L.v_w, L.has_bias ? &L.v_b : nullptr, T, H, kv_out, v.f());
            gpu::launch_rope(q.f(), dcos.f(), dsin.f(), T, nh, hd);
            gpu::launch_rope(k.f(), dcos.f(), dsin.f(), T, nkv, hd);
            // Fill the cache with all T post-RoPE K/V rows.
            gpu::launch_cache_append(k.f(), v.f(), k_cache[li].f(), v_cache[li].f(), T,
                                     kv_dim, 0);
            gpu::launch_attention(q.f(), k.f(), v.f(), scores.f(), T, nh, nkv, hd, ctx.f());
            M.linear(gemm, ctx.f(), L.o_w, nullptr, T, q_out, H, delta.f());
            gpu::launch_residual_add(h.f(), delta.f(), T * H);
            gpu::launch_rmsnorm(h.f(), L.post_attn_ln.bf(), float(cfg.rms_norm_eps), T, H, normed.f());
            M.linear(gemm, normed.f(), L.gate_w, nullptr, T, H, I, gate.f());
            M.linear(gemm, normed.f(), L.up_w, nullptr, T, H, I, up.f());
            gpu::launch_swiglu(gate.f(), up.f(), T * I);
            M.linear(gemm, gate.f(), L.down_w, nullptr, T, I, H, delta.f());
            gpu::launch_residual_add(h.f(), delta.f(), T * H);
        }
        pos = T;

        // Final norm + tied LM head, last position only.
        gpu::launch_rmsnorm(h.f() + (T - 1) * H, M.final_norm.bf(),
                            float(cfg.rms_norm_eps), 1, H, normed_.f());
        DevBuf d_logits(size_t(V) * 4);
        M.linear(gemm, normed_.f(), M.embed_tokens, nullptr, 1, H, V, d_logits.f());
        std::vector<float> logits(V);
        CUDA_CHECK(cudaMemcpy(logits.data(), d_logits.p, size_t(V) * 4, cudaMemcpyDeviceToHost));
        return logits;
    }

    std::vector<float> decode_one(int64_t id) {
        const auto& cfg = M.cfg;
        const int64_t H = cfg.hidden_size, hd = cfg.head_dim;
        const int64_t nh = cfg.num_attention_heads, nkv = cfg.num_key_value_heads;
        const int64_t q_out = nh * hd, kv_out = nkv * hd, I = cfg.intermediate_size;
        const int64_t V = cfg.vocab_size;
        if (pos >= max_seq) throw std::runtime_error("decode_one: cache full (max_seq)");
        if (id < 0 || id >= V) throw std::runtime_error("token id out of range");
        const int64_t cache_len = pos + 1;

        int64_t id64 = id;
        CUDA_CHECK(cudaMemcpy(d_id.p, &id64, 8, cudaMemcpyHostToDevice));
        gpu::launch_embedding(M.embed_tokens.bf(), d_id.i64(), 1, H, h_.f());

        std::vector<float> cos_h, sin_h;
        rope_tables_host(cfg.rope_theta, hd, /*pos0=*/pos, 1, cos_h, sin_h);
        CUDA_CHECK(cudaMemcpy(d_cos.p, cos_h.data(), cos_h.size() * 4, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_sin.p, sin_h.data(), sin_h.size() * 4, cudaMemcpyHostToDevice));

        for (int64_t li = 0; li < cfg.num_hidden_layers; li++) {
            DevLayer& L = M.layers[li];
            gpu::launch_rmsnorm(h_.f(), L.input_ln.bf(), float(cfg.rms_norm_eps), 1, H, normed_.f());
            M.linear(gemm, normed_.f(), L.q_w, L.has_bias ? &L.q_b : nullptr, 1, H, q_out, q_.f());
            M.linear(gemm, normed_.f(), L.k_w, L.has_bias ? &L.k_b : nullptr, 1, H, kv_out, k_.f());
            M.linear(gemm, normed_.f(), L.v_w, L.has_bias ? &L.v_b : nullptr, 1, H, kv_out, v_.f());
            gpu::launch_rope(q_.f(), d_cos.f(), d_sin.f(), 1, nh, hd);
            gpu::launch_rope(k_.f(), d_cos.f(), d_sin.f(), 1, nkv, hd);
            gpu::launch_cache_append(k_.f(), v_.f(), k_cache[li].f(), v_cache[li].f(), 1,
                                     kv_dim, pos);
            if (attn == AttnPath::kParallel)
                gpu::launch_attention_cached_par(q_.f(), k_cache[li].f(), v_cache[li].f(),
                                                 attn_scores_.f(), cache_len, nh, nkv, hd,
                                                 ctx_.f());
            else
                gpu::launch_attention_cached(q_.f(), k_cache[li].f(), v_cache[li].f(),
                                             cache_len, nh, nkv, hd, ctx_.f());
            M.linear(gemm, ctx_.f(), L.o_w, nullptr, 1, q_out, H, delta_.f());
            gpu::launch_residual_add(h_.f(), delta_.f(), H);
            gpu::launch_rmsnorm(h_.f(), L.post_attn_ln.bf(), float(cfg.rms_norm_eps), 1, H, normed_.f());
            M.linear(gemm, normed_.f(), L.gate_w, nullptr, 1, H, I, gate_.f());
            M.linear(gemm, normed_.f(), L.up_w, nullptr, 1, H, I, up_.f());
            gpu::launch_swiglu(gate_.f(), up_.f(), I);
            M.linear(gemm, gate_.f(), L.down_w, nullptr, 1, I, H, delta_.f());
            gpu::launch_residual_add(h_.f(), delta_.f(), H);
        }
        pos++;

        gpu::launch_rmsnorm(h_.f(), M.final_norm.bf(), float(cfg.rms_norm_eps), 1, H, normed_.f());
        DevBuf d_logits(size_t(V) * 4);
        M.linear(gemm, normed_.f(), M.embed_tokens, nullptr, 1, H, V, d_logits.f());
        std::vector<float> logits(V);
        CUDA_CHECK(cudaMemcpy(logits.data(), d_logits.p, size_t(V) * 4, cudaMemcpyDeviceToHost));
        return logits;
    }
};

GpuSession::GpuSession(GpuModel& m, int64_t max_seq, GemmPath gemm, AttnPath attn)
    : impl_(new Impl(m, max_seq, gemm, attn)) {}
GpuSession::~GpuSession() = default;
std::vector<float> GpuSession::prefill(const std::vector<int64_t>& ids) {
    return impl_->prefill(ids);
}
std::vector<float> GpuSession::decode_one(int64_t id) { return impl_->decode_one(id); }
int64_t GpuSession::position() const { return impl_->pos; }

std::vector<int64_t> greedy_decode_cached_gpu(GpuModel& m,
                                              const std::vector<int64_t>& ids, int n_new,
                                              int64_t max_seq, GemmPath gemm,
                                              AttnPath attn) {
    GpuSession s(m, max_seq, gemm, attn);
    std::vector<float> logits = s.prefill(ids);
    std::vector<int64_t> out;
    const int64_t V = m.cfg().vocab_size;
    for (int i = 0; i < n_new; i++) {
        int64_t best = 0;
        for (int64_t v = 1; v < V; v++)
            if (logits[v] > logits[best]) best = v;
        out.push_back(best);
        logits = s.decode_one(best);
    }
    return out;
}

} // namespace llm
