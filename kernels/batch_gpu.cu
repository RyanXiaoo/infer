// batch_gpu.cu — see batch_gpu.h. Kernels come from kernels/ops/batch.cu and
// kernels/ops/gemm.cu; block bookkeeping from src/block_pool.h.

#include "batch_gpu.h"
#include "model_gpu_internal.cuh"
#include "../src/block_pool.h"

#include <algorithm>
#include <stdexcept>
#include <vector>

namespace llm {

namespace {
struct Row { int slot; int64_t pos; int64_t token; };

// Row scratch for one forward of up to `rows` rows. Decode uses a 32-row
// scratch (kept, and captured into graphs); prefill uses a 256-row scratch so
// a long prompt runs in few chunks through the tiled GEMM.
struct Scratch {
    int rows = 0;
    bool all_logits = true;   // false: prefill, only the last row's logits are wanted
    DevBuf d_ids, d_slot, d_pos, d_out;
    DevBuf h, normed, q, k, v, ctx, delta, gate, up, scores, logits;
    DevBuf part_m, part_l, part_o;   // flash-decoding partials [R x nh x splits (x hd)]
    void alloc(int R, bool all, int64_t H, int64_t q_out, int64_t kv_out, int64_t I, int64_t nh,
               int64_t hd, int64_t max_seq, int64_t V, int splits) {
        rows = R; all_logits = all;
        part_m = DevBuf(size_t(R) * nh * splits * 4);
        part_l = DevBuf(size_t(R) * nh * splits * 4);
        part_o = DevBuf(size_t(R) * nh * splits * hd * 4);
        // The T=1 LM head runs the batched GEMV over a 32-row window, so the
        // logits buffer always holds 32 rows even when one is wanted.
        const int LR = all ? R : 32;
        d_ids = DevBuf(size_t(R) * 8); d_slot = DevBuf(size_t(R) * 4);
        d_pos = DevBuf(size_t(R) * 4); d_out = DevBuf(size_t(R) * 8);
        h = DevBuf(size_t(R) * H * 4); normed = DevBuf(size_t(R) * H * 4);
        q = DevBuf(size_t(R) * q_out * 4); k = DevBuf(size_t(R) * kv_out * 4);
        v = DevBuf(size_t(R) * kv_out * 4); ctx = DevBuf(size_t(R) * q_out * 4);
        delta = DevBuf(size_t(R) * H * 4); gate = DevBuf(size_t(R) * I * 4);
        up = DevBuf(size_t(R) * I * 4);
        scores = DevBuf(size_t(R) * nh * max_seq * 4);
        logits = DevBuf(size_t(LR) * V * 4);
        // The batched GEMV reads whole 32-row groups of its input; rows beyond
        // the live batch must be finite.
        CUDA_CHECK(cudaMemset(normed.p, 0, size_t(R) * H * 4));
        CUDA_CHECK(cudaMemset(ctx.p, 0, size_t(R) * q_out * 4));
        CUDA_CHECK(cudaMemset(gate.p, 0, size_t(R) * I * 4));
    }
};
} // namespace

struct GpuBatch::Impl {
    GpuModel::Impl& M;
    const int n_slots;
    const int64_t max_seq;
    const GemmPath gemm;
    const bool prefix_cache;
    const bool use_graphs;
    const bool attn_split;
    const int max_blocks;         // per slot: ceil(max_seq / 16)
    int attn_threads, splits;
    int64_t kv_dim, H, hd, nh, nkv, q_out, kv_out, I, V, n_layers;
    int64_t last_reused = 0;
    static constexpr int kPrefillRows = 256;

    BlockPool pool;
    std::vector<BlockTable> tables;     // per slot
    std::vector<int> h_table;           // [n_slots x max_blocks], mirror of the device table
    bool table_dirty = true;
    DevBuf d_table;
    std::vector<DevBuf> k_pool, v_pool;  // per layer: [n_blocks x 16 x kv_dim]
    DevBuf d_cos_tab, d_sin_tab;

    Scratch dec;   // 32 rows, logits for every row
    Scratch pre;   // 256 rows, logits for the last row only (allocated on first prefill > 32)

    // One executable graph per batch width B (decode scratch only). Everything
    // inside a graph is fixed for a given B: grid sizes, the GEMV template,
    // the attention thread count (from max_seq), every pointer. What changes
    // per step lives in device memory the graph reads: ids/slot/pos, the block
    // table, the caches.
    std::vector<cudaGraphExec_t> graphs;
    int n_captured = 0;

    static int64_t block_bytes(const ModelConfig& c) {
        return int64_t(gpu::kKvBlockRows) * c.num_key_value_heads * c.head_dim * 4 * 2 * c.num_hidden_layers;
    }
    static int blocks_for(int slots, int64_t ms, int64_t budget, const ModelConfig& c) {
        const int per_slot = int((ms + gpu::kKvBlockRows - 1) / gpu::kKvBlockRows);
        return budget > 0 ? int(budget / block_bytes(c)) : slots * per_slot;
    }

    Impl(GpuModel& g, int slots, int64_t ms, GemmPath gm, int64_t budget, bool pc, bool graphs_on,
         bool split)
        : M(*g.impl_), n_slots(slots), max_seq(ms), gemm(gm), prefix_cache(pc),
          use_graphs(graphs_on && gm == GemmPath::kMine), attn_split(split),
          max_blocks(int((ms + gpu::kKvBlockRows - 1) / gpu::kKvBlockRows)),
          pool(blocks_for(slots, ms, budget, g.impl_->cfg)), tables(size_t(slots)),
          graphs(size_t(kMaxRows) + 1, nullptr) {
        if (slots < 1 || slots > kMaxRows) throw std::runtime_error("GpuBatch: 1..32 slots");
        if (pool.n_blocks() < 1) throw std::runtime_error("GpuBatch: budget below one block");
        const auto& c = M.cfg;
        H = c.hidden_size; hd = c.head_dim; nh = c.num_attention_heads;
        nkv = c.num_key_value_heads; I = c.intermediate_size; V = c.vocab_size;
        n_layers = c.num_hidden_layers;
        kv_dim = nkv * hd; q_out = nh * hd; kv_out = kv_dim;
        attn_threads = gpu::attention_par_threads(hd, max_seq);
        splits = gpu::attention_splits(max_seq);
        const size_t pool_bytes = size_t(pool.n_blocks()) * gpu::kKvBlockRows * kv_dim * 4;
        for (int64_t l = 0; l < n_layers; l++) {
            k_pool.emplace_back(pool_bytes);
            v_pool.emplace_back(pool_bytes);
        }
        h_table.assign(size_t(slots) * max_blocks, 0);
        d_table = DevBuf(h_table.size() * 4);
        dec.alloc(kMaxRows, true, H, q_out, kv_out, I, nh, hd, max_seq, V, splits);
        std::vector<float> cos_h, sin_h;
        rope_tables_host(c.rope_theta, hd, 0, max_seq, cos_h, sin_h);
        d_cos_tab = DevBuf(cos_h.size() * 4);
        d_sin_tab = DevBuf(sin_h.size() * 4);
        CUDA_CHECK(cudaMemcpy(d_cos_tab.p, cos_h.data(), cos_h.size() * 4, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_sin_tab.p, sin_h.data(), sin_h.size() * 4, cudaMemcpyHostToDevice));
    }

    ~Impl() {
        for (cudaGraphExec_t g : graphs) if (g) cudaGraphExecDestroy(g);
    }

    // T-row linear. Up to 32 rows: the batched GEMV (weights read once for the
    // rows). More: the tiled GEMM (weights read once per 64-row tile). The
    // reference paths run their own T-row forms.
    void linear_b(const float* x, DevTensor& W, DevTensor* b, int T, int64_t in, int64_t out,
                  float* y) {
        if (gemm != GemmPath::kMine) { M.linear(gemm, x, W, b, T, in, out, y); return; }
        if (T <= kMaxRows) gpu::launch_gemv_batched(x, W.bf(), b ? b->bf() : nullptr, T, in, out, y);
        else gpu::launch_gemm_tiled(x, W.bf(), b ? b->bf() : nullptr, T, in, out, y);
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

    // The kernel sequence of one forward over B rows in scratch S: no host
    // work, no synchronisation, per-step inputs read from device arrays. Logits
    // (and the argmax) for all B rows when S.all_logits, else for the last row
    // only (prefill wants just the next token).
    void launch_sequence(Scratch& S, int B) {
        const auto& cfg = M.cfg;
        const float eps = float(cfg.rms_norm_eps);
        const int* dslot = static_cast<const int*>(S.d_slot.p);
        const int* dpos = static_cast<const int*>(S.d_pos.p);
        const int* dtab = static_cast<const int*>(d_table.p);

        gpu::launch_embedding(M.embed_tokens.bf(), S.d_ids.i64(), B, H, S.h.f());
        for (int64_t li = 0; li < n_layers; li++) {
            DevLayer& L = M.layers[li];
            gpu::launch_rmsnorm(S.h.f(), L.input_ln.bf(), eps, B, H, S.normed.f());
            linear_b(S.normed.f(), L.q_w, L.has_bias ? &L.q_b : nullptr, B, H, q_out, S.q.f());
            linear_b(S.normed.f(), L.k_w, L.has_bias ? &L.k_b : nullptr, B, H, kv_out, S.k.f());
            linear_b(S.normed.f(), L.v_w, L.has_bias ? &L.v_b : nullptr, B, H, kv_out, S.v.f());
            gpu::launch_rope_qk_append_batched(S.q.f(), S.k.f(), S.v.f(), d_cos_tab.f(), d_sin_tab.f(),
                                               dslot, dpos, B, nh, nkv, hd, dtab, max_blocks,
                                               k_pool[li].f(), v_pool[li].f());
            if (attn_split)
                gpu::launch_attention_split(S.q.f(), k_pool[li].f(), v_pool[li].f(), dslot, dpos, B,
                                            nh, nkv, hd, dtab, max_blocks, splits, S.part_m.f(),
                                            S.part_l.f(), S.part_o.f(), S.ctx.f());
            else
                gpu::launch_attention_cached_batched(S.q.f(), k_pool[li].f(), v_pool[li].f(),
                                                     S.scores.f(), dslot, dpos, B, max_seq, nh, nkv,
                                                     hd, dtab, max_blocks, max_seq, S.ctx.f(),
                                                     attn_threads);
            linear_b(S.ctx.f(), L.o_w, nullptr, B, q_out, H, S.delta.f());
            gpu::launch_residual_add(S.h.f(), S.delta.f(), int64_t(B) * H);
            gpu::launch_rmsnorm(S.h.f(), L.post_attn_ln.bf(), eps, B, H, S.normed.f());
            linear_b(S.normed.f(), L.gate_w, nullptr, B, H, I, S.gate.f());
            linear_b(S.normed.f(), L.up_w, nullptr, B, H, I, S.up.f());
            gpu::launch_swiglu(S.gate.f(), S.up.f(), int64_t(B) * I);
            linear_b(S.gate.f(), L.down_w, nullptr, B, I, H, S.delta.f());
            gpu::launch_residual_add(S.h.f(), S.delta.f(), int64_t(B) * H);
        }
        if (S.all_logits) {
            gpu::launch_rmsnorm(S.h.f(), M.final_norm.bf(), eps, B, H, S.normed.f());
            linear_b(S.normed.f(), M.embed_tokens, nullptr, B, H, V, S.logits.f());
            gpu::launch_argmax_rows(S.logits.f(), B, V, S.d_out.i64());
        } else {
            // Last row only. The batched GEMV reads a whole 32-row group of its
            // input, so run it over the 32-row window ending at the last row
            // (earlier rows are live, finite data) and take the window's last
            // output row.
            const int64_t last = B - 1, start = std::max<int64_t>(0, last - (kMaxRows - 1));
            gpu::launch_rmsnorm(S.h.f() + last * H, M.final_norm.bf(), eps, 1, H, S.normed.f() + last * H);
            const int Bw = int(last - start + 1);
            linear_b(S.normed.f() + start * H, M.embed_tokens, nullptr, Bw, H, V, S.logits.f());
            gpu::launch_argmax_rows(S.logits.f() + (Bw - 1) * V, 1, V, S.d_out.i64());
        }
    }

    // Runs `rows` (<= S.rows) through the model in scratch S. out receives
    // S.logits_rows argmax ids (all rows, or just the last).
    void forward_rows(Scratch& S, const std::vector<Row>& rows, int64_t* out) {
        const int B = int(rows.size());
        std::vector<int64_t> ids(B);
        std::vector<int> slot(B), p(B);
        for (int b = 0; b < B; b++) { ids[b] = rows[b].token; slot[b] = rows[b].slot; p[b] = int(rows[b].pos); }
        if (table_dirty) sync_table();
        CUDA_CHECK(cudaMemcpy(S.d_ids.p, ids.data(), B * 8, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(S.d_slot.p, slot.data(), B * 4, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(S.d_pos.p, p.data(), B * 4, cudaMemcpyHostToDevice));
        const bool graph = use_graphs && &S == &dec;
        if (graph) {
            cudaGraphExec_t& exec = graphs[size_t(B)];
            if (!exec) {
                // Capture on the per-thread default stream (the launchers use
                // plain <<<>>>). Kernels are recorded, not run.
                cudaGraph_t g = nullptr;
                CUDA_CHECK(cudaStreamBeginCapture(cudaStreamPerThread, cudaStreamCaptureModeThreadLocal));
                launch_sequence(S, B);
                CUDA_CHECK(cudaStreamEndCapture(cudaStreamPerThread, &g));
                CUDA_CHECK(cudaGraphInstantiate(&exec, g, 0));
                CUDA_CHECK(cudaGraphDestroy(g));
                n_captured++;
            }
            CUDA_CHECK(cudaGraphLaunch(exec, cudaStreamPerThread));
        } else {
            launch_sequence(S, B);
        }
        const int n_out = S.all_logits ? B : 1;
        CUDA_CHECK(cudaMemcpy(out, S.d_out.p, size_t(n_out) * 8, cudaMemcpyDeviceToHost));
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

        // Run positions [reused, T) as rows of this slot. Rows in one chunk see
        // each other's K/V (written before attention) and each attends over
        // [0, pos], so causality holds within a chunk. Chunks > 32 rows go
        // through the tiled GEMM in the prefill scratch.
        const int64_t L = T - reused;
        Scratch* S = &dec;
        if (L > kMaxRows) {
            if (pre.rows == 0) pre.alloc(kPrefillRows, false, H, q_out, kv_out, I, nh, hd, max_seq, V, splits);
            S = &pre;
        }
        int64_t first = -1;
        for (int64_t r0 = reused; r0 < T; r0 += S->rows) {
            std::vector<Row> rows;
            for (int64_t p = r0; p < std::min(T, r0 + int64_t(S->rows)); p++)
                rows.push_back({slot, p, ids[size_t(p)]});
            int64_t outs[kMaxRows];
            forward_rows(*S, rows, outs);
            first = S == &dec ? outs[rows.size() - 1] : outs[0];
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
            forward_rows(dec, go, outs);
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
                   int64_t kv_budget_bytes, bool prefix_cache, bool use_graphs, bool attn_split)
    : impl_(new Impl(m, n_slots, max_seq, gemm, kv_budget_bytes, prefix_cache, use_graphs,
                     attn_split)) {}
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
int GpuBatch::graphs_captured() const { return impl_->n_captured; }

} // namespace llm
