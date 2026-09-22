// batch_gpu.cu — see batch_gpu.h. Mirrors session_gpu.cu's decode step with
// B rows; kernels come from kernels/ops/batch.cu.

#include "batch_gpu.h"
#include "model_gpu_internal.cuh"

#include <algorithm>
#include <stdexcept>
#include <vector>

namespace llm {

struct GpuBatch::Impl {
    GpuModel::Impl& M;
    const int n_slots;
    const int64_t max_seq;
    const GemmPath gemm;
    int attn_threads;
    std::vector<int64_t> pos;   // per slot: rows filled in its cache

    // Per layer: [n_slots x max_seq x kv_dim] fp32.
    std::vector<DevBuf> k_cache, v_cache;
    int64_t kv_dim, H, hd, nh, nkv, q_out, kv_out, I, V;

    // Decode scratch for up to kMaxRows rows.
    DevBuf d_ids, d_slot, d_pos, d_out;
    DevBuf h_, normed_, q_, k_, v_, ctx_, delta_, gate_, up_, scores_, d_logits_;
    DevBuf d_cos_tab, d_sin_tab;
    // Prefill scratch is allocated per call (prompt length varies; prefill is
    // not the hot path).

    Impl(GpuModel& g, int slots, int64_t ms, GemmPath gm)
        : M(*g.impl_), n_slots(slots), max_seq(ms), gemm(gm), pos(size_t(slots), 0) {
        if (slots < 1 || slots > kMaxRows) throw std::runtime_error("GpuBatch: 1..32 slots");
        const auto& c = M.cfg;
        H = c.hidden_size; hd = c.head_dim; nh = c.num_attention_heads;
        nkv = c.num_key_value_heads; I = c.intermediate_size; V = c.vocab_size;
        kv_dim = nkv * hd; q_out = nh * hd; kv_out = kv_dim;
        attn_threads = gpu::attention_par_threads(hd, max_seq);
        for (int64_t l = 0; l < c.num_hidden_layers; l++) {
            k_cache.emplace_back(size_t(slots) * max_seq * kv_dim * 4);
            v_cache.emplace_back(size_t(slots) * max_seq * kv_dim * 4);
        }
        const int R = kMaxRows;
        d_ids = DevBuf(R * 8); d_slot = DevBuf(R * 4); d_pos = DevBuf(R * 4); d_out = DevBuf(R * 8);
        h_ = DevBuf(R * H * 4); normed_ = DevBuf(R * H * 4);
        q_ = DevBuf(R * q_out * 4); k_ = DevBuf(R * kv_out * 4); v_ = DevBuf(R * kv_out * 4);
        ctx_ = DevBuf(R * q_out * 4); delta_ = DevBuf(R * H * 4);
        gate_ = DevBuf(R * I * 4); up_ = DevBuf(R * I * 4);
        scores_ = DevBuf(size_t(R) * nh * max_seq * 4);
        d_logits_ = DevBuf(size_t(R) * V * 4);
        // The batched GEMV reads all kMaxRows activation rows (compile-time
        // batch width); rows beyond the live batch must be finite.
        for (DevBuf* b : {&normed_, &ctx_, &gate_})
            CUDA_CHECK(cudaMemset(b->p, 0, size_t(R) * (b == &ctx_ ? q_out : b == &gate_ ? I : H) * 4));
        std::vector<float> cos_h, sin_h;
        rope_tables_host(c.rope_theta, hd, 0, max_seq, cos_h, sin_h);
        d_cos_tab = DevBuf(cos_h.size() * 4);
        d_sin_tab = DevBuf(sin_h.size() * 4);
        CUDA_CHECK(cudaMemcpy(d_cos_tab.p, cos_h.data(), cos_h.size() * 4, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_sin_tab.p, sin_h.data(), sin_h.size() * 4, cudaMemcpyHostToDevice));
    }

    // T-row linear: my batched GEMV reads each weight once for up to 32 rows;
    // longer inputs (prefill) go in chunks of 32. The reference paths run their
    // T-row forms. x must have rows padded to a multiple of 32 (the kernel
    // reads the whole compile-time batch width).
    void linear_b(const float* x, DevTensor& W, DevTensor* b, int T, int64_t in, int64_t out,
                  float* y) {
        if (gemm != GemmPath::kMine) { M.linear(gemm, x, W, b, T, in, out, y); return; }
        for (int r0 = 0; r0 < T; r0 += kMaxRows)
            gpu::launch_gemv_batched(x + int64_t(r0) * in, W.bf(), b ? b->bf() : nullptr,
                                     std::min(kMaxRows, T - r0), in, out, y + int64_t(r0) * out);
    }

    float* kc(int64_t layer, int slot) { return k_cache[layer].f() + size_t(slot) * max_seq * kv_dim; }
    float* vc(int64_t layer, int slot) { return v_cache[layer].f() + size_t(slot) * max_seq * kv_dim; }

    int64_t prefill(int slot, const std::vector<int64_t>& ids) {
        const auto& cfg = M.cfg;
        const int64_t T = int64_t(ids.size());
        if (slot < 0 || slot >= n_slots) throw std::runtime_error("prefill: bad slot");
        if (T == 0 || T > max_seq) throw std::runtime_error("prefill: bad prompt length");
        for (int64_t id : ids)
            if (id < 0 || id >= V) throw std::runtime_error("token id out of range");
        const float eps = float(cfg.rms_norm_eps);

        DevBuf d_pids(size_t(T) * 8);
        CUDA_CHECK(cudaMemcpy(d_pids.p, ids.data(), size_t(T) * 8, cudaMemcpyHostToDevice));
        // Row count padded to a multiple of 32 and zero-filled: linear_b's
        // kernel reads whole 32-row chunks.
        const int64_t Tp = (T + kMaxRows - 1) / kMaxRows * kMaxRows;
        DevBuf h(Tp * H * 4), normed(Tp * H * 4), q(Tp * q_out * 4), k(Tp * kv_out * 4),
               v(Tp * kv_out * 4), ctx(Tp * q_out * 4), delta(Tp * H * 4), gate(Tp * I * 4),
               up(Tp * I * 4), sc(size_t(nh) * T * T * 4);
        for (DevBuf* b : {&normed, &ctx, &gate})
            CUDA_CHECK(cudaMemset(b->p, 0, size_t(Tp) * (b == &ctx ? q_out : b == &gate ? I : H) * 4));

        gpu::launch_embedding(M.embed_tokens.bf(), d_pids.i64(), T, H, h.f());
        for (int64_t li = 0; li < cfg.num_hidden_layers; li++) {
            DevLayer& L = M.layers[li];
            gpu::launch_rmsnorm(h.f(), L.input_ln.bf(), eps, T, H, normed.f());
            linear_b(normed.f(), L.q_w, L.has_bias ? &L.q_b : nullptr, int(T), H, q_out, q.f());
            linear_b(normed.f(), L.k_w, L.has_bias ? &L.k_b : nullptr, int(T), H, kv_out, k.f());
            linear_b(normed.f(), L.v_w, L.has_bias ? &L.v_b : nullptr, int(T), H, kv_out, v.f());
            gpu::launch_rope(q.f(), d_cos_tab.f(), d_sin_tab.f(), T, nh, hd);
            gpu::launch_rope(k.f(), d_cos_tab.f(), d_sin_tab.f(), T, nkv, hd);
            gpu::launch_cache_append(k.f(), v.f(), kc(li, slot), vc(li, slot), T, kv_dim, 0);
            gpu::launch_attention(q.f(), k.f(), v.f(), sc.f(), T, nh, nkv, hd, ctx.f());
            linear_b(ctx.f(), L.o_w, nullptr, int(T), q_out, H, delta.f());
            gpu::launch_residual_add(h.f(), delta.f(), T * H);
            gpu::launch_rmsnorm(h.f(), L.post_attn_ln.bf(), eps, T, H, normed.f());
            linear_b(normed.f(), L.gate_w, nullptr, int(T), H, I, gate.f());
            linear_b(normed.f(), L.up_w, nullptr, int(T), H, I, up.f());
            gpu::launch_swiglu(gate.f(), up.f(), T * I);
            linear_b(gate.f(), L.down_w, nullptr, int(T), I, H, delta.f());
            gpu::launch_residual_add(h.f(), delta.f(), T * H);
        }
        pos[slot] = T;

        gpu::launch_rmsnorm(h.f() + (T - 1) * H, M.final_norm.bf(), eps, 1, H, normed_.f());
        // Same T=1 kernel the single-sequence session uses for its first token.
        M.linear(gemm, normed_.f(), M.embed_tokens, nullptr, 1, H, V, d_logits_.f());
        gpu::launch_argmax_rows(d_logits_.f(), 1, V, d_out.i64());
        int64_t first = 0;
        CUDA_CHECK(cudaMemcpy(&first, d_out.p, 8, cudaMemcpyDeviceToHost));
        return first;
    }

    std::vector<int64_t> step(const std::vector<StepRow>& rows) {
        const auto& cfg = M.cfg;
        const int B = int(rows.size());
        if (B == 0) return {};
        if (B > kMaxRows) throw std::runtime_error("step: more than 32 rows");
        std::vector<int64_t> ids(B);
        std::vector<int> slot(B), p(B);
        int64_t max_len = 0;
        for (int b = 0; b < B; b++) {
            const int s = rows[b].slot;
            if (s < 0 || s >= n_slots) throw std::runtime_error("step: bad slot");
            if (pos[s] >= max_seq) throw std::runtime_error("step: slot cache full");
            if (rows[b].token < 0 || rows[b].token >= V) throw std::runtime_error("token id out of range");
            ids[b] = rows[b].token; slot[b] = s; p[b] = int(pos[s]);
            max_len = std::max(max_len, pos[s] + 1);
        }
        CUDA_CHECK(cudaMemcpy(d_ids.p, ids.data(), B * 8, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_slot.p, slot.data(), B * 4, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_pos.p, p.data(), B * 4, cudaMemcpyHostToDevice));
        const float eps = float(cfg.rms_norm_eps);
        const int* dslot = static_cast<const int*>(d_slot.p);
        const int* dpos = static_cast<const int*>(d_pos.p);

        gpu::launch_embedding(M.embed_tokens.bf(), d_ids.i64(), B, H, h_.f());
        for (int64_t li = 0; li < cfg.num_hidden_layers; li++) {
            DevLayer& L = M.layers[li];
            gpu::launch_rmsnorm(h_.f(), L.input_ln.bf(), eps, B, H, normed_.f());
            linear_b(normed_.f(), L.q_w, L.has_bias ? &L.q_b : nullptr, B, H, q_out, q_.f());
            linear_b(normed_.f(), L.k_w, L.has_bias ? &L.k_b : nullptr, B, H, kv_out, k_.f());
            linear_b(normed_.f(), L.v_w, L.has_bias ? &L.v_b : nullptr, B, H, kv_out, v_.f());
            gpu::launch_rope_qk_append_batched(q_.f(), k_.f(), v_.f(), d_cos_tab.f(), d_sin_tab.f(),
                                               dslot, dpos, B, nh, nkv, hd, max_seq,
                                               k_cache[li].f(), v_cache[li].f());
            gpu::launch_attention_cached_batched(q_.f(), k_cache[li].f(), v_cache[li].f(),
                                                 scores_.f(), dslot, dpos, B, max_len, nh, nkv,
                                                 hd, max_seq, ctx_.f(), attn_threads);
            linear_b(ctx_.f(), L.o_w, nullptr, B, q_out, H, delta_.f());
            gpu::launch_residual_add(h_.f(), delta_.f(), int64_t(B) * H);
            gpu::launch_rmsnorm(h_.f(), L.post_attn_ln.bf(), eps, B, H, normed_.f());
            linear_b(normed_.f(), L.gate_w, nullptr, B, H, I, gate_.f());
            linear_b(normed_.f(), L.up_w, nullptr, B, H, I, up_.f());
            gpu::launch_swiglu(gate_.f(), up_.f(), int64_t(B) * I);
            linear_b(gate_.f(), L.down_w, nullptr, B, I, H, delta_.f());
            gpu::launch_residual_add(h_.f(), delta_.f(), int64_t(B) * H);
        }
        gpu::launch_rmsnorm(h_.f(), M.final_norm.bf(), eps, B, H, normed_.f());
        linear_b(normed_.f(), M.embed_tokens, nullptr, B, H, V, d_logits_.f());
        gpu::launch_argmax_rows(d_logits_.f(), B, V, d_out.i64());
        std::vector<int64_t> out(B);
        CUDA_CHECK(cudaMemcpy(out.data(), d_out.p, B * 8, cudaMemcpyDeviceToHost));
        for (int b = 0; b < B; b++) pos[slot[b]]++;
        return out;
    }
};

GpuBatch::GpuBatch(GpuModel& m, int n_slots, int64_t max_seq, GemmPath gemm)
    : impl_(new Impl(m, n_slots, max_seq, gemm)) {}
GpuBatch::~GpuBatch() = default;
int GpuBatch::slots() const { return impl_->n_slots; }
int64_t GpuBatch::max_seq() const { return impl_->max_seq; }
int64_t GpuBatch::prefill(int slot, const std::vector<int64_t>& prompt) {
    return impl_->prefill(slot, prompt);
}
std::vector<int64_t> GpuBatch::step(const std::vector<StepRow>& rows) { return impl_->step(rows); }
int64_t GpuBatch::position(int slot) const { return impl_->pos.at(size_t(slot)); }

} // namespace llm
