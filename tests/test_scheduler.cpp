// test_scheduler.cpp — the continuous batching scheduler against a fake engine.
//
// The fake engine generates a deterministic token stream per request (its id
// repeated, then eos after `len` tokens) and records every step's row set, so
// the test can assert WHEN requests were admitted and retired, that outputs
// are per-request correct even as the batch composition changes under them,
// and that static and continuous batching differ by exactly the expected
// number of steps.

#include "../src/scheduler.h"

#include <cstdio>
#include <map>
#include <string>
#include <vector>

namespace {

int failures = 0;
#define CHECK(cond, ...)                                                        \
    do {                                                                        \
        if (!(cond)) {                                                          \
            failures++;                                                         \
            std::printf("FAIL %s:%d: ", __FILE__, __LINE__);                    \
            std::printf(__VA_ARGS__);                                           \
            std::printf("\n");                                                  \
        }                                                                       \
    } while (0)

constexpr int64_t kEos = 999;

// Request id N generates tokens N, N, N, ... and eos after `len[N]` tokens.
// Memory is modelled as one "block" per token, `budget` blocks in total (0 =
// unlimited). A preempted request comes back with its generated tokens inside
// the prompt (prompt = [id, 1, 2, id, id, ...]); emitted is recovered from that.
struct FakeEngine : llm::BatchEngine {
    int n_slots;
    int64_t cap;
    int budget = 0;
    std::map<int64_t, int> len;                 // request id -> tokens before eos
    struct SlotState { int64_t id = -1; int emitted = 0; int tokens = 0; bool used = false; };
    std::vector<SlotState> state;
    std::vector<std::vector<int>> step_rows;    // per step(): the slots decoded

    FakeEngine(int slots, int64_t max_seq) : n_slots(slots), cap(max_seq), state(slots) {}
    int slots() const override { return n_slots; }
    int64_t max_seq() const override { return cap; }
    int used_blocks() const { int n = 0; for (auto& s : state) n += s.tokens; return n; }
    bool has_room(int64_t prompt_len) const override {
        return budget == 0 || used_blocks() + prompt_len + 1 <= budget;
    }
    int64_t next(int slot) {
        SlotState& s = state[slot];
        s.emitted++;
        return s.emitted > len.at(s.id) ? kEos : s.id;
    }
    int64_t prefill(int slot, const std::vector<int64_t>& prompt) override {
        if (!has_room(int64_t(prompt.size()))) return -1;
        state[slot] = SlotState{prompt.at(0), int(prompt.size()) - 3, int(prompt.size()) + 1, true};
        return next(slot);
    }
    std::vector<int64_t> step(const std::vector<llm::StepRow>& rows) override {
        std::vector<int> ss;
        std::vector<int64_t> out;
        for (const auto& r : rows) {
            CHECK(r.token == (state[r.slot].emitted > len.at(state[r.slot].id) ? kEos : state[r.slot].id),
                  "fed token is not the slot's last output");
            if (budget && used_blocks() + 1 > budget) { out.push_back(-1); continue; }
            state[r.slot].tokens++;
            ss.push_back(r.slot);
            out.push_back(next(r.slot));
        }
        step_rows.push_back(ss);
        return out;
    }
    void release(int slot) override { state[slot] = SlotState{}; }
    int blocks_in_use() const override { return used_blocks(); }
    int blocks_total() const override { return budget; }
};

llm::Request req(int64_t id, int max_new, int64_t arrival = 0) {
    llm::Request r;
    r.id = id;
    r.prompt = {id, 1, 2};
    r.max_new = max_new;
    r.arrival_step = arrival;
    return r;
}

std::vector<int64_t> expect_tokens(int64_t id, int len, int max_new) {
    std::vector<int64_t> v(size_t(std::min(len, max_new)), id);
    if (len < max_new) v.push_back(kEos);
    return v;
}

void test_outputs_and_timing(bool continuous) {
    FakeEngine eng(2, 256);
    eng.len = {{10, 3}, {20, 6}, {30, 2}, {40, 4}};
    llm::SchedulerConfig cfg;
    cfg.continuous = continuous;
    cfg.eos_id = kEos;
    llm::Scheduler sch(eng, cfg);
    for (int64_t id : {10, 20, 30, 40}) sch.submit(req(id, 100));
    sch.run_until_idle();

    const char* mode = continuous ? "continuous" : "static";
    CHECK(sch.completed().size() == 4, "%s: %zu completed", mode, sch.completed().size());
    for (const auto& c : sch.completed())
        CHECK(c.tokens == expect_tokens(c.id, eng.len.at(c.id), 100),
              "%s: request %lld tokens wrong", mode, (long long)c.id);
    std::map<int64_t, const llm::Completed*> by_id;
    for (const auto& c : sch.completed()) by_id[c.id] = &c;

    if (continuous) {
        // Slots: 10 and 20 start at step 0. 10 emits 3 tokens + eos: prefill gives
        // token 1 at step 0, steps 0,1,2 give tokens 2,3,eos -> retires at step 3,
        // and 30 takes its slot at step 3. 30 (len 2) frees it again at step 5,
        // so 40 is admitted at 5, before 20 finishes at step 6.
        CHECK(by_id[10]->finished_step == 3, "10 finished at %lld", (long long)by_id[10]->finished_step);
        CHECK(by_id[30]->admitted_step == 3, "30 admitted at %lld", (long long)by_id[30]->admitted_step);
        CHECK(by_id[20]->finished_step == 6, "20 finished at %lld", (long long)by_id[20]->finished_step);
        CHECK(by_id[40]->admitted_step == 5, "40 admitted at %lld", (long long)by_id[40]->admitted_step);
        // The batch stayed at 2 rows from step 0 until 30 finished (step 5), then 1-2 rows.
        CHECK(eng.step_rows[3].size() == 2, "step 3 had %zu rows", eng.step_rows[3].size());
    } else {
        // Static: 30 waits until BOTH 10 and 20 are done (step 6).
        CHECK(by_id[30]->admitted_step == 6, "static: 30 admitted at %lld",
              (long long)by_id[30]->admitted_step);
        CHECK(by_id[40]->admitted_step == 6, "static: 40 admitted at %lld",
              (long long)by_id[40]->admitted_step);
    }
    std::printf("%s: 4 requests in %lld steps, %zu decode calls\n", mode,
                (long long)sch.steps(), eng.step_rows.size());
}

void test_arrivals_and_max_new() {
    FakeEngine eng(3, 64);
    eng.len = {{1, 100}, {2, 100}};   // never hit eos: max_new decides
    llm::SchedulerConfig cfg;
    cfg.eos_id = kEos;
    llm::Scheduler sch(eng, cfg);
    sch.submit(req(1, 5, /*arrival=*/0));
    sch.submit(req(2, 3, /*arrival=*/4));
    sch.run_until_idle();
    std::map<int64_t, const llm::Completed*> by_id;
    for (const auto& c : sch.completed()) by_id[c.id] = &c;
    CHECK(by_id[1]->tokens.size() == 5, "max_new: %zu tokens", by_id[1]->tokens.size());
    CHECK(by_id[2]->admitted_step == 4, "arrival: admitted at %lld", (long long)by_id[2]->admitted_step);
    CHECK(by_id[2]->tokens.size() == 3, "max_new: %zu tokens", by_id[2]->tokens.size());
    // Steps 1..3 ran with one row (request 1 alone) while 2 had not arrived.
    CHECK(eng.step_rows[2].size() == 1, "step 2 rows %zu", eng.step_rows[2].size());
}

void test_preemption() {
    // Same workload as test_outputs_and_timing but the fake engine holds only
    // 14 token-blocks: two 3-token prompts (4 blocks each incl. the first output)
    // fit, growth forces evictions. Outputs must be unchanged; preemptions > 0.
    FakeEngine eng(2, 256);
    eng.budget = 14;
    eng.len = {{10, 3}, {20, 6}, {30, 2}, {40, 4}};
    llm::SchedulerConfig cfg;
    cfg.eos_id = kEos;
    llm::Scheduler sch(eng, cfg);
    for (int64_t id : {10, 20, 30, 40}) sch.submit(req(id, 100));
    sch.run_until_idle();
    CHECK(sch.completed().size() == 4, "preempt: %zu completed", sch.completed().size());
    for (const auto& c : sch.completed())
        CHECK(c.tokens == expect_tokens(c.id, eng.len.at(c.id), 100),
              "preempt: request %lld tokens wrong", (long long)c.id);
    CHECK(sch.preemptions() > 0, "expected at least one preemption");
    CHECK(eng.used_blocks() == 0, "leak: %d blocks after idle", eng.used_blocks());
    int preempt_events = 0;
    for (size_t i = 0; i < sch.events().size(); i++)
        if (sch.events().at(i).kind == llm::EventKind::kPreempt) preempt_events++;
    CHECK(preempt_events == sch.preemptions(), "preempt events %d vs %d", preempt_events, sch.preemptions());
    std::printf("preemption: 4 requests, %d preemptions, %lld steps\n", sch.preemptions(),
                (long long)sch.steps());
}

void test_event_ring() {
    FakeEngine eng(1, 256);
    eng.len = {{7, 2}};
    llm::SchedulerConfig cfg;
    cfg.eos_id = kEos;
    cfg.event_capacity = 4;   // tiny: forces wraparound
    llm::Scheduler sch(eng, cfg);
    sch.submit(req(7, 100));
    sch.run_until_idle();
    // Events: admit, step0, step1, retire, step2(0 rows) = 5 > 4 -> 1 dropped.
    CHECK(sch.events().size() == 4, "ring holds %zu", sch.events().size());
    CHECK(sch.events().dropped() == 1, "dropped %zu", sch.events().dropped());
    CHECK(sch.events().at(3).kind == llm::EventKind::kStep, "last event is the empty step");
    CHECK(sch.events().at(2).kind == llm::EventKind::kRetire, "retire kept");
}

} // namespace

int main() {
    test_outputs_and_timing(true);
    test_outputs_and_timing(false);
    test_arrivals_and_max_new();
    test_preemption();
    test_event_ring();
    if (failures == 0) { std::printf("test_scheduler: all checks passed\n"); return 0; }
    std::printf("test_scheduler: %d FAILURES\n", failures);
    return 1;
}
