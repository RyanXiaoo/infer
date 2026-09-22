// batch_gpu.cu — see batch_gpu.h. Kernels come from kernels/ops/batch.cu;
// block bookkeeping from src/block_pool.h.

#include "batch_gpu.h"
#include "model_gpu_internal.cuh"
#include "../src/block_pool.h"

#include <algorithm>
#include <stdexcept>
#include <vector>

namespace llm {

namespace {
struct Row { int slot; int64_t pos; int64_t token; };
} // namespace

struct GpuBatch::Impl {
    GpuModel::Impl& M;
    const int n_slots;
    const int64_t max_seq;
    const GemmPath gemm;
    const bool prefix_cache;
    const int max_blocks;         // per slot: ceil(max_seq / 16)
    int attn_threads;
    int64_t kv_dim, H, hd, nh, nkv, q_out, kv_out, I, V, n_layers;
    int64_t last_reused = 0;

    BlockPool pool;
    std::vector<BlockTable> tables;     // per slot
    std::vector<int> h_table;           // [n_slots x max_blocks], mirror of the device table
    bool table_dirty = true;
    DevBuf d_table;

    std::vector<DevBuf> k_pool, v_pool;  // per layer: [n_blocks x 16 x kv_dim]

    // Row scratch for up to kMaxRows rows.
    DevBuf d_ids, d_slot, d_pos, d_out;
    DevBuf h_, normed_, q_, k_, v_, ctx_, delta_, gate_, up_, scores_, d_logits_;
    DevBuf d_cos_tab, d_sin_tab;

    static int64_t block_bytes(const ModelConfig& c) {
        return int64_t(gpu::kKvBlockRows) * c.num_key_value_heads * c.head_dim * 4 * 2 * c.num_hidden_layers;
    }
    static int blocks_for(int slots, int64_t ms, int64_t budget, const ModelConfig& c) {
        const int per_slot = int((ms + gpu::kKvBlockRows - 1) / gpu::kKvBlockRows);
        return budget > 0 ? int(budget / block_bytes(c)) : slots * per_slot;
    }

    Impl(GpuModel& g, int slots, int64_t ms, GemmPath gm, int64_t budget, bool pc)
        : M(*g.impl_), n_slots(slots), max_seq(ms), gemm(gm), prefix_cache(pc),
          max_blocks(int((ms + gpu::kKvBlockRows - 1) / gpu::kKvBlockRows)),
          pool(blocks_for(slots, ms, budget, g.impl_->cfg)), tables(size_t(slots)) {
        if (slots < 1 || slots > kMaxRows) throw std::runtime_error("GpuBatch: 1..32 slots");
        if (pool.n_blocks() < 1) throw std::runtime_error("GpuBatch: budget below one block");
        const auto& c = M.cfg;
        H = c.hidden_size; hd = c.head_dim; nh = c.num_attention_heads;
        nkv = c.num_key_value_heads; I = c.intermediate_size; V = c.vocab_size;
        n_layers = c.num_hidden_layers;
        kv_dim = nkv * hd; q_out = nh * hd; kv_out = kv_dim;
        attn_threads = gpu::attention_par_threads(hd, max_seq);
        const size_t pool_bytes = size_t(pool.n_blocks()) * gpu::kKvBlockRows * kv_dim * 4;
        for (int64_t l = 0; l < n_layers; l++) {
            k_pool.emplace_back(pool_bytes);
            v_pool.emplace_back(pool_bytes);
        }
        h_table.assign(size_t(slots) * max_blocks, 0);
        d_table = DevBuf(h_table.size() * 4);

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

    void linear_b(const float* x, DevTensor& W, DevTensor* b, int B, int64_t in, int64_t out,
                  float* y) {
        if (gemm != GemmPath::kMine) { M.linear(gemm, x, W, b, B, in, out, y); return; }
        gpu::launch_gemv_batched(x, W.bf(), b ? b->bf() : nullptr, B, in, out, y);
    }

    void sync_table() {
        for (int s = 0; s < n_slots; s++) {
            const auto& blocks = tables[size_t(s)].blocks;
            for (size_t j = 0; j < blocks.size(); j++) h_table[size_t(s) * max_blocks + j] = blocks[j];
        }
        CUDA_CHECK(cudaMemcpy(d_table.p, h_table.data(), h_table.size() * 4, cudaMemcpyHostToDevice));
        table_dirty = false;
    }

    // Give every row a writable block for its position; `failed` marks rows
    // the pool could not serve (their state is untouched).
    void plan_rows(const std::vector<Row>& rows, std::vector<bool>& failed) {
        failed.assign(rows.size(), false);
        for (size_t i = 0; i < rows.size(); i++) {
            WritePlan w = plan_write(pool, tables[size_t(rows[i].slot)], rows[i].pos);
            if (!w.ok) { failed[i] = true; continue; }
            if (w.cow) {
                for (int64_t l = 0; l < n_layers; l++)
                    gpu::launch_block_copy(k_pool[l].f(), v_pool[l].f(), w.cow_from, w.block,
                                           kv_dim, k_pool[l].f(), v_pool[l].f());
                table_dirty = true;
            } else if (rows[i].pos % gpu::kKvBlockRows == 0) {
                table_dirty = true;   // a new block entered the table
            }
        }
    }

    // Runs `rows` (<= kMaxRows) through the model; out[i] = argmax of row i's
    // logits. Every row's block must already be planned.
    void forward_rows(const std::vector<Row>& rows, int64_t* out) {
        const auto& cfg = M.cfg;
        const int B = int(rows.size());
        std::vector<int64_t> ids(B);
        std::vector<int> slot(B), p(B);
        int64_t max_len = 0;
        for (int b = 0; b < B; b++) {
            ids[b] = rows[b].token; slot[b] = rows[b].slot; p[b] = int(rows[b].pos);
            max_len = std::max(max_len, rows[b].pos + 1);
        }
        if (table_dirty) sync_table();
        CUDA_CHECK(cudaMemcpy(d_ids.p, ids.data(), B * 8, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_slot.p, slot.data(), B * 4, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_pos.p, p.data(), B * 4, cudaMemcpyHostToDevice));
        const float eps = float(cfg.rms_norm_eps);
        const int* dslot = static_cast<const int*>(d_slot.p);
        const int* dpos = static_cast<const int*>(d_pos.p);
        const int* dtab = static_cast<const int*>(d_table.p);

        gpu::launch_embedding(M.embed_tokens.bf(), d_ids.i64(), B, H, h_.f());
        for (int64_t li = 0; li < n_layers; li++) {
            DevLayer& L = M.layers[li];
            gpu::launch_rmsnorm(h_.f(), L.input_ln.bf(), eps, B, H, normed_.f());
            linear_b(normed_.f(), L.q_w, L.has_bias ? &L.q_b : nullptr, B, H, q_out, q_.f());
            linear_b(normed_.f(), L.k_w, L.has_bias ? &L.k_b : nullptr, B, H, kv_out, k_.f());
            linear_b(normed_.f(), L.v_w, L.has_bias ? &L.v_b : nullptr, B, H, kv_out, v_.f());
            gpu::launch_rope_qk_append_batched(q_.f(), k_.f(), v_.f(), d_cos_tab.f(), d_sin_tab.f(),
                                               dslot, dpos, B, nh, nkv, hd, dtab, max_blocks,
                                               k_pool[li].f(), v_pool[li].f());
            gpu::launch_attention_cached_batched(q_.f(), k_pool[li].f(), v_pool[li].f(),
                                                 scores_.f(), dslot, dpos, B, max_len, nh, nkv,
                                                 hd, dtab, max_blocks, max_seq, ctx_.f(),
                                                 attn_threads);
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
        CUDA_CHECK(cudaMemcpy(out, d_out.p, B * 8, cudaMemcpyDeviceToHost));
    }

    bool has_room(int64_t prompt_len) const {
        return pool.n_free() >= BlockTable::blocks_needed(prompt_len) + 1;
    }

    int64_t prefill(int slot, const std::vector<int64_t>& ids) {
        const int64_t T = int64_t(ids.size());
        if (slot < 0 || slot >= n_slots) throw std::runtime_error("prefill: bad slot");
        if (T == 0 || T > max_seq) throw std::runtime_error("prefill: bad prompt length");
        for (int64_t id : ids)
            if (id < 0 || id >= V) throw std::runtime_error("token id out of range");
        BlockTable& t = tables[size_t(slot)];
        if (!t.blocks.empty()) release_all(pool, t);
        const int64_t reused = plan_prefill(pool, t, ids, prefix_cache);
        if (reused < 0) return -1;
        last_reused = reused;
        table_dirty = true;

        // Run positions [reused, T) as rows of this slot, 32 at a time. Rows in
        // one chunk see each other's K/V (written before attention) and each
        // attends over [0, pos], so causality holds within a chunk.
        int64_t first = -1;
        for (int64_t r0 = reused; r0 < T; r0 += kMaxRows) {
            std::vector<Row> rows;
            for (int64_t p = r0; p < std::min(T, r0 + int64_t(kMaxRows)); p++)
                rows.push_back({slot, p, ids[size_t(p)]});
            int64_t outs[kMaxRows];
            forward_rows(rows, outs);
            first = outs[rows.size() - 1];
        }
        register_prefix(pool, t, ids);
        return first;
    }

    std::vector<int64_t> step(const std::vector<StepRow>& srows) {
        const int B = int(srows.size());
        if (B == 0) return {};
        if (B > kMaxRows) throw std::runtime_error("step: more than 32 rows");
        std::vector<Row> rows;
        std::vector<int64_t> out(size_t(B), -1);
        std::vector<size_t> idx;
        for (int b = 0; b < B; b++) {
            const int s = srows[b].slot;
            if (s < 0 || s >= n_slots) throw std::runtime_error("step: bad slot");
            if (srows[b].token < 0 || srows[b].token >= V) throw std::runtime_error("token id out of range");
            BlockTable& t = tables[size_t(s)];
            if (t.length >= max_seq) continue;   // stays -1: cache full for this row
            rows.push_back({s, t.length, srows[b].token});
            idx.push_back(size_t(b));
        }
        std::vector<bool> failed;
        plan_rows(rows, failed);
        std::vector<Row> go;
        std::vector<size_t> go_idx;
        for (size_t i = 0; i < rows.size(); i++)
            if (!failed[i]) { go.push_back(rows[i]); go_idx.push_back(idx[i]); }
        if (!go.empty()) {
            int64_t outs[kMaxRows];
            forward_rows(go, outs);
            for (size_t i = 0; i < go.size(); i++) {
                out[go_idx[i]] = outs[i];
                tables[size_t(go[i].slot)].length++;
            }
        }
        return out;
    }

    void release(int slot) {
        release_all(pool, tables[size_t(slot)]);
        table_dirty = true;
    }
};

GpuBatch::GpuBatch(GpuModel& m, int n_slots, int64_t max_seq, GemmPath gemm,
                   int64_t kv_budget_bytes, bool prefix_cache)
    : impl_(new Impl(m, n_slots, max_seq, gemm, kv_budget_bytes, prefix_cache)) {}
GpuBatch::~GpuBatch() = default;
int GpuBatch::slots() const { return impl_->n_slots; }
int64_t GpuBatch::max_seq() const { return impl_->max_seq; }
bool GpuBatch::has_room(int64_t prompt_len) const { return impl_->has_room(prompt_len); }
int64_t GpuBatch::prefill(int slot, const std::vector<int64_t>& prompt) { return impl_->prefill(slot, prompt); }
std::vector<int64_t> GpuBatch::step(const std::vector<StepRow>& rows) { return impl_->step(rows); }
void GpuBatch::release(int slot) { impl_->release(slot); }
int GpuBatch::blocks_in_use() const { return impl_->pool.n_in_use(); }
int GpuBatch::blocks_total() const { return impl_->pool.n_blocks(); }
int64_t GpuBatch::position(int slot) const { return impl_->tables.at(size_t(slot)).length; }
int64_t GpuBatch::bytes_per_block() const { return Impl::block_bytes(impl_->M.cfg); }
int64_t GpuBatch::last_prefill_reused() const { return impl_->last_reused; }

} // namespace llm
