// spec_engine.cu — see spec_engine.h.

#include "spec_engine.h"
#include "model_gpu_internal.cuh"
#include "ops/ops.cuh"

#include <algorithm>
#include <stdexcept>

namespace llm {

// Device side of the sampled path: the draft's logits of every draft step
// (the accept kernel needs q_i(d_i) and the residual), and small per-group
// argument arrays for one accept launch.
struct SpecEngine::Dev {
    int k, maxB;
    int64_t V;
    DevBuf stash;                                   // [k x maxB x V] fp32
    DevBuf trow, drow, pos0, temp, seed, drafts;    // per group inputs
    DevBuf out_a, out_tok;
    Dev(int k_, int maxB_, int64_t V_)
        : k(k_), maxB(maxB_), V(V_), stash(size_t(k_) * maxB_ * V_ * 4),
          trow(size_t(maxB_) * 4), drow(size_t(maxB_) * 4), pos0(size_t(maxB_) * 8), temp(size_t(maxB_) * 4),
          seed(size_t(maxB_) * 8), drafts(size_t(maxB_) * k_ * 8), out_a(size_t(maxB_) * 4), out_tok(size_t(maxB_) * 8) {}
    float* stash_row(int step) { return stash.f() + size_t(step) * maxB * V; }
};

SpecEngine::SpecEngine(GpuBatch& target, GpuBatch& draft, int k)
    : target_(target), draft_(draft), k_(k), draft_owes_(size_t(target.slots()), -1) {
    if (target.slots() != draft.slots()) throw std::runtime_error("SpecEngine: slot counts differ");
    if (k < 1 || k > 8) throw std::runtime_error("SpecEngine: k must be 1..8");
    if (target.max_seq() > draft.max_seq()) throw std::runtime_error("SpecEngine: draft max_seq too small");
    if (target.vocab() != draft.vocab() && target.vocab_limit() > std::min(target.vocab(), draft.vocab()))
        throw std::runtime_error("SpecEngine: set_vocab_limit(min vocab) on both engines first");
}

SpecEngine::~SpecEngine() = default;

bool SpecEngine::has_room(int64_t prompt_len) const {
    return target_.has_room(prompt_len) && draft_.has_room(prompt_len);
}

int64_t SpecEngine::prefill(int slot, const std::vector<int64_t>& prompt, SampleParams sample) {
    draft_owes_[size_t(slot)] = -1;
    const int64_t first = target_.prefill(slot, prompt, sample);
    if (first < 0) return -1;
    if (draft_.prefill(slot, prompt, sample) < 0) { target_.release(slot); return -1; }
    return first;
}

void SpecEngine::release(int slot) {
    target_.release(slot);
    draft_.release(slot);
    draft_owes_[size_t(slot)] = -1;
}

std::vector<int64_t> SpecEngine::step(const std::vector<StepRow>& rows) {
    std::vector<std::vector<int64_t>> multi = step_multi(rows);
    std::vector<int64_t> out(rows.size(), -1);
    for (size_t i = 0; i < rows.size(); i++) if (!multi[i].empty()) out[i] = multi[i][0];
    return out;
}

// Runs the accept kernel for the sampled groups of one verify chunk (target
// logits still hold that chunk). sampled_j: indices j (into spec_idx) of the
// chunk's sampled groups; a_out / tok_out receive their accepted counts and
// emitted tokens in the same order.
void SpecEngine::accept_sampled(const std::vector<StepRow>& rows, const std::vector<size_t>& spec_idx,
                                const std::vector<std::vector<int64_t>>& drafts, const std::vector<int64_t>& len0,
                                size_t j0, const std::vector<size_t>& sampled_j, std::vector<int>& a_out,
                                std::vector<int64_t>& tok_out) {
    const int G = int(sampled_j.size());
    std::vector<int> trow(G), drow(G);
    std::vector<int64_t> pos0(G), dr(size_t(G) * k_);
    std::vector<float> temp(G);
    std::vector<uint64_t> seed(G);
    for (int g = 0; g < G; g++) {
        const size_t j = sampled_j[size_t(g)];
        const StepRow& r = rows[spec_idx[j]];
        trow[g] = int(j - j0) * (k_ + 1);   // groups are packed in j order within the chunk
        drow[g] = int(j);                   // draft step rows follow spec_idx order
        pos0[g] = len0[j];
        temp[g] = r.sample.temperature;
        seed[g] = r.sample.seed;
        for (int i = 0; i < k_; i++) dr[size_t(g) * k_ + i] = drafts[j][size_t(i)];
    }
    Dev& d = *dev_;
    CUDA_CHECK(cudaMemcpy(d.trow.p, trow.data(), size_t(G) * 4, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d.drow.p, drow.data(), size_t(G) * 4, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d.pos0.p, pos0.data(), size_t(G) * 8, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d.temp.p, temp.data(), size_t(G) * 4, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d.seed.p, seed.data(), size_t(G) * 8, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d.drafts.p, dr.data(), dr.size() * 8, cudaMemcpyHostToDevice));
    gpu::launch_spec_accept(target_.device_logits(), d.stash.f(), d.V, target_.vocab_limit(), k_, d.maxB,
                            d.trow.i32(), d.drow.i32(), d.pos0.i64(), d.temp.f(),
                            static_cast<const uint64_t*>(d.seed.p), d.drafts.i64(), G, d.out_a.i32(), d.out_tok.i64());
    CUDA_CHECK(cudaGetLastError());
    a_out.resize(size_t(G));
    tok_out.resize(size_t(G));
    CUDA_CHECK(cudaMemcpy(a_out.data(), d.out_a.p, size_t(G) * 4, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(tok_out.data(), d.out_tok.p, size_t(G) * 8, cudaMemcpyDeviceToHost));
}

std::vector<std::vector<int64_t>> SpecEngine::step_multi(const std::vector<StepRow>& rows) {
    const size_t B = rows.size();
    std::vector<std::vector<int64_t>> out(B);
    if (B == 0) return out;
    std::vector<size_t> spec_idx(B);
    for (size_t i = 0; i < B; i++) spec_idx[i] = i;
    bool any_sampled = false;
    for (const StepRow& r : rows) any_sampled |= r.sample.temperature > 0.0f;
    if (any_sampled && !dev_) dev_ = std::make_unique<Dev>(k_, target_.slots(), target_.vocab());
    if (any_sampled && int(B) > dev_->maxB) throw std::runtime_error("SpecEngine: more rows than slots");

    // 1. Draft catches up where it still owes a token, then drafts k tokens
    //    for every row with k batched steps (sampled rows draw d_i ~ q_i with
    //    the request's temperature and seed; their logits are stashed for the
    //    accept kernel).
    std::vector<StepRow> owe;
    for (size_t i : spec_idx) {
        const int s = rows[i].slot;
        if (draft_owes_[size_t(s)] >= 0) { owe.push_back({s, draft_owes_[size_t(s)], {}}); draft_owes_[size_t(s)] = -1; }
    }
    if (!owe.empty()) draft_.step(owe);

    std::vector<StepRow> drows;
    for (size_t i : spec_idx) drows.push_back({rows[i].slot, rows[i].token, rows[i].sample});
    std::vector<std::vector<int64_t>> drafts(spec_idx.size());
    std::vector<int64_t> len0(spec_idx.size());   // target length before this round
    for (size_t j = 0; j < spec_idx.size(); j++) len0[j] = target_.position(rows[spec_idx[j]].slot);
    for (int step = 0; step < k_; step++) {
        std::vector<int64_t> d = draft_.step(drows);
        for (size_t j = 0; j < drows.size(); j++) {
            if (d[j] < 0) throw std::runtime_error("speculative: draft out of cache memory");
            drafts[j].push_back(d[j]);
            drows[j].token = d[j];
        }
        if (any_sampled)
            CUDA_CHECK(cudaMemcpy(dev_->stash_row(step), draft_.device_logits(), drows.size() * size_t(dev_->V) * 4,
                                  cudaMemcpyDeviceToDevice));
    }
    // The draft's cache now holds k positions past len0 (last, d_1 .. d_{k-1}).

    // 2. Target verifies [last, d_1 .. d_k] for every slot, packing as many
    //    slots as fit one 32-row forward; greedy groups compare argmaxes on the
    //    host, sampled groups run the accept kernel while the logits are live.
    std::vector<int> acc(spec_idx.size());
    std::vector<int64_t> emit_tok(spec_idx.size());
    {
        const size_t per = size_t(GpuBatch::kMaxRows / (k_ + 1));
        for (size_t j0 = 0; j0 < spec_idx.size(); j0 += per) {
            const size_t j1 = std::min(spec_idx.size(), j0 + per);
            std::vector<GpuBatch::VerifyGroup> groups;
            std::vector<size_t> sampled_j;
            for (size_t j = j0; j < j1; j++) {
                const StepRow& r = rows[spec_idx[j]];
                std::vector<int64_t> cand{r.token};
                cand.insert(cand.end(), drafts[j].begin(), drafts[j].end());
                groups.push_back({r.slot, cand, r.sample});
                if (r.sample.temperature > 0.0f) sampled_j.push_back(j);
            }
            std::vector<std::vector<int64_t>> t = target_.verify(groups);   // t[i] = argmax after cand[i]
            for (size_t j = j0; j < j1; j++) {
                if (rows[spec_idx[j]].sample.temperature > 0.0f) continue;
                int a = 0;
                while (a < k_ && t[j - j0][size_t(a)] == drafts[j][size_t(a)]) a++;
                acc[j] = a;
                emit_tok[j] = t[j - j0][size_t(a)];   // correction, or the free extra token when a == k
            }
            if (!sampled_j.empty()) {
                std::vector<int> a_out;
                std::vector<int64_t> tok_out;
                accept_sampled(rows, spec_idx, drafts, len0, j0, sampled_j, a_out, tok_out);
                for (size_t g = 0; g < sampled_j.size(); g++) { acc[sampled_j[g]] = a_out[g]; emit_tok[sampled_j[g]] = tok_out[g]; }
            }
        }
    }

    // 3. Emit accepted drafts + the correction/bonus token; roll both caches
    //    back to the accepted length: positions [0, len0 + a + 1) hold
    //    last, d_1 .. d_a.
    for (size_t j = 0; j < spec_idx.size(); j++) {
        const StepRow& r = rows[spec_idx[j]];
        const int a = acc[j];
        std::vector<int64_t>& emit = out[spec_idx[j]];
        for (int i = 0; i < a; i++) emit.push_back(drafts[j][size_t(i)]);
        emit.push_back(emit_tok[j]);
        target_.truncate(r.slot, len0[j] + a + 1);
        if (a < k_) {
            draft_.truncate(r.slot, len0[j] + a + 1);
        } else {
            // draft cache ends after d_{k-1}; it must ingest d_k before the next round
            draft_owes_[size_t(r.slot)] = drafts[j][size_t(k_ - 1)];
        }
        stats_.rounds++;
        stats_.drafted += k_;
        stats_.accepted += a;
        stats_.emitted += a + 1;
    }
    return out;
}

} // namespace llm
