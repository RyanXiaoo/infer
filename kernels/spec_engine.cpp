// spec_engine.cpp — see spec_engine.h.

#include "spec_engine.h"

#include <algorithm>
#include <stdexcept>

namespace llm {

SpecEngine::SpecEngine(GpuBatch& target, GpuBatch& draft, int k)
    : target_(target), draft_(draft), k_(k), draft_owes_(size_t(target.slots()), -1) {
    if (target.slots() != draft.slots()) throw std::runtime_error("SpecEngine: slot counts differ");
    if (k < 1 || k > 8) throw std::runtime_error("SpecEngine: k must be 1..8");
    if (target.max_seq() > draft.max_seq()) throw std::runtime_error("SpecEngine: draft max_seq too small");
}

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

std::vector<std::vector<int64_t>> SpecEngine::step_multi(const std::vector<StepRow>& rows) {
    const size_t B = rows.size();
    std::vector<std::vector<int64_t>> out(B);
    if (B == 0) return out;

    // Rows that sample (temperature > 0) take a plain target step this commit.
    std::vector<StepRow> plain;
    std::vector<size_t> plain_idx, spec_idx;
    for (size_t i = 0; i < B; i++) {
        if (rows[i].sample.temperature > 0.0f) { plain.push_back(rows[i]); plain_idx.push_back(i); }
        else spec_idx.push_back(i);
    }
    if (!plain.empty()) {
        std::vector<int64_t> t = target_.step(plain);
        for (size_t j = 0; j < plain.size(); j++) if (t[j] >= 0) out[plain_idx[j]] = {t[j]};
        // keep the draft in sync for these rows (it must see the same tokens)
        draft_.step(plain);
    }
    if (spec_idx.empty()) return out;

    // 1. Draft catches up where it still owes a token, then drafts k tokens
    //    for every speculating row with k batched steps.
    std::vector<StepRow> owe;
    for (size_t i : spec_idx) {
        const int s = rows[i].slot;
        if (draft_owes_[size_t(s)] >= 0) { owe.push_back({s, draft_owes_[size_t(s)], {}}); draft_owes_[size_t(s)] = -1; }
    }
    if (!owe.empty()) draft_.step(owe);

    std::vector<StepRow> drows;
    for (size_t i : spec_idx) drows.push_back({rows[i].slot, rows[i].token, {}});
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
    }
    // The draft's cache now holds k positions past len0 (last, d_1 .. d_{k-1}).

    // 2. Target verifies [last, d_1 .. d_k] for every slot, packing as many
    //    slots as fit one 32-row forward.
    std::vector<std::vector<int64_t>> targ(spec_idx.size());
    {
        const size_t per = size_t(GpuBatch::kMaxRows / (k_ + 1));
        for (size_t j0 = 0; j0 < spec_idx.size(); j0 += per) {
            std::vector<GpuBatch::VerifyGroup> groups;
            for (size_t j = j0; j < std::min(spec_idx.size(), j0 + per); j++) {
                const StepRow& r = rows[spec_idx[j]];
                std::vector<int64_t> cand{r.token};
                cand.insert(cand.end(), drafts[j].begin(), drafts[j].end());
                groups.push_back({r.slot, cand, r.sample});
            }
            std::vector<std::vector<int64_t>> t = target_.verify(groups);
            for (size_t j = j0; j < j0 + t.size(); j++) targ[j] = t[j - j0];
        }
    }
    for (size_t j = 0; j < spec_idx.size(); j++) {
        const StepRow& r = rows[spec_idx[j]];
        const std::vector<int64_t>& t = targ[j];   // t[i] = argmax after cand[i]

        // 3. Accept the longest prefix where the draft matched the target.
        int a = 0;
        while (a < k_ && t[size_t(a)] == drafts[j][size_t(a)]) a++;
        std::vector<int64_t>& emit = out[spec_idx[j]];
        for (int i = 0; i < a; i++) emit.push_back(drafts[j][size_t(i)]);
        emit.push_back(t[size_t(a)]);   // correction, or the free extra token when a == k

        // 4. Roll both caches back to the accepted length: positions
        //    [0, len0 + a + 1) hold last, d_1 .. d_a.
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
