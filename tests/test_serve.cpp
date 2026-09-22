// test_serve.cpp — the serving core against a fake engine (Stage 9).
//
// The fake engine: request id N emits token N forever (eos never comes from
// the model), one "block" per token, an optional budget. Clients run on their
// own threads. What is checked: tokens stream in order and stop at max_new or
// the per-request eos; cancel mid-stream frees the slot within one step and
// returns memory; cancel-while-queued never prefills; 100 requests with random
// cancels leave zero blocks in use and no thread stuck; shutdown ends
// in-flight requests as cancelled.

#include "../src/serve.h"

#include <atomic>
#include <chrono>
#include <cstdio>
#include <map>
#include <random>
#include <thread>
#include <vector>

namespace {
int failures = 0;
#define CHECK(cond, ...) \
    do { if (!(cond)) { failures++; std::printf("FAIL %s:%d: ", __FILE__, __LINE__); \
         std::printf(__VA_ARGS__); std::printf("\n"); } } while (0)

constexpr int64_t kEos = 999;

struct FakeEngine : llm::BatchEngine {
    int n_slots; int64_t cap; int budget;
    struct S { int64_t id = -1; int tokens = 0; bool used = false; int64_t eos_at = -1; int emitted = 0; };
    std::vector<S> st;
    std::atomic<int> prefills{0}, steps{0};
    std::map<int64_t, int> eos_after;   // request id -> emit eos after this many tokens
    FakeEngine(int slots, int64_t max_seq, int budget_ = 0) : n_slots(slots), cap(max_seq), budget(budget_), st(slots) {}
    int slots() const override { return n_slots; }
    int64_t max_seq() const override { return cap; }
    int used() const { int n = 0; for (auto& s : st) n += s.tokens; return n; }
    bool has_room(int64_t p) const override { return budget == 0 || used() + p + 1 <= budget; }
    int64_t next(int slot) {
        S& s = st[slot]; s.emitted++;
        auto it = eos_after.find(s.id);
        return (it != eos_after.end() && s.emitted > it->second) ? kEos : s.id;
    }
    int64_t prefill(int slot, const std::vector<int64_t>& prompt, llm::SampleParams) override {
        if (!has_room(int64_t(prompt.size()))) return -1;
        prefills++;
        st[slot] = S{prompt.at(0), int(prompt.size()) + 1, true, -1, int(prompt.size()) - 3};
        std::this_thread::sleep_for(std::chrono::microseconds(200));
        return next(slot);
    }
    std::vector<int64_t> step(const std::vector<llm::StepRow>& rows) override {
        steps++;
        std::vector<int64_t> out;
        for (auto& r : rows) {
            if (budget && used() + 1 > budget) { out.push_back(-1); continue; }
            st[r.slot].tokens++;
            out.push_back(next(r.slot));
        }
        std::this_thread::sleep_for(std::chrono::microseconds(200));
        return out;
    }
    void release(int slot) override { st[slot] = S{}; }
    int blocks_in_use() const override { return used(); }
    int blocks_total() const override { return budget; }
};

llm::Request req(int64_t id, int max_new) {
    llm::Request r; r.id = id; r.prompt = {id, 1, 2}; r.max_new = max_new; return r;
}

std::vector<int64_t> drain(llm::RequestHandle& h) {
    std::vector<int64_t> v; int64_t t;
    while (h.next(t)) v.push_back(t);
    return v;
}
} // namespace

int main() {
    using namespace llm;
    {   // streaming, ordering, max_new, per-request eos
        FakeEngine eng(2, 256);
        eng.eos_after[20] = 3;
        SchedulerConfig cfg; cfg.eos_id = kEos;
        Server srv(eng, cfg);
        auto h1 = srv.submit(req(10, 5));
        Request r2 = req(20, 50); r2.eos_id = kEos;
        auto h2 = srv.submit(r2);
        std::vector<int64_t> a = drain(*h1), b = drain(*h2);
        CHECK(a == std::vector<int64_t>(5, 10), "max_new stream: %zu tokens", a.size());
        CHECK(b == std::vector<int64_t>({20, 20, 20, kEos}), "eos stream: %zu tokens", b.size());
        CHECK(h1->reason() == StopReason::kMaxNew && h2->reason() == StopReason::kEos, "reasons");
        CHECK(eng.used() == 0, "blocks after: %d", eng.used());
        Metrics m = srv.metrics();
        CHECK(m.requests_done == 2 && m.tokens_total == 9 && m.active == 0, "metrics: done %lld tokens %lld",
              (long long)m.requests_done, (long long)m.tokens_total);
    }
    {   // cancel mid-stream frees the slot; a queued request takes it
        FakeEngine eng(1, 1 << 20);
        SchedulerConfig cfg; cfg.eos_id = kEos;
        Server srv(eng, cfg);
        auto h1 = srv.submit(req(10, 1000));
        auto h2 = srv.submit(req(20, 3));
        int64_t t; int n = 0;
        while (n < 4 && h1->next(t)) n++;
        h1->cancel();
        std::vector<int64_t> rest = drain(*h1);
        CHECK(h1->reason() == StopReason::kCancelled, "cancel reason");
        CHECK(rest.size() <= 2, "at most one step of tokens after cancel, got %zu", rest.size());
        std::vector<int64_t> b = drain(*h2);
        CHECK(b == std::vector<int64_t>(3, 20), "queued request ran after the cancel");
        CHECK(eng.used() == 0, "blocks after: %d", eng.used());
        CHECK(srv.metrics().cancellations == 1, "cancellations");
    }
    {   // cancel while queued: never prefilled
        FakeEngine eng(1, 256);
        SchedulerConfig cfg; cfg.eos_id = kEos;
        Server srv(eng, cfg);
        auto h1 = srv.submit(req(10, 20));
        auto h2 = srv.submit(req(20, 20));
        h2->cancel();
        std::vector<int64_t> b = drain(*h2);
        std::vector<int64_t> a = drain(*h1);
        CHECK(b.empty() && h2->reason() == StopReason::kCancelled, "queued cancel: %zu tokens", b.size());
        CHECK(a.size() == 20, "other request unaffected: %zu", a.size());
        CHECK(eng.prefills == 1, "prefills: %d (cancelled request must not prefill)", eng.prefills.load());
    }
    {   // load: 100 requests, 4 slots, budget that forces preemption, random cancels
        FakeEngine eng(4, 256, /*budget=*/60);
        SchedulerConfig cfg; cfg.eos_id = kEos;
        Server srv(eng, cfg);
        std::vector<std::thread> clients;
        std::atomic<int> ok{0}, cancelled{0};
        for (int i = 0; i < 100; i++) {
            clients.emplace_back([&, i] {
                std::mt19937 rng{uint32_t(i)};
                auto h = srv.submit(req(1000 + i, 5 + int(rng() % 20)));
                const bool will_cancel = rng() % 3 == 0;
                int64_t t; int n = 0;
                while (h->next(t)) { n++; if (will_cancel && n == 3) h->cancel(); }
                if (h->reason() == StopReason::kCancelled) cancelled++; else ok++;
            });
        }
        for (auto& c : clients) c.join();
        CHECK(ok + cancelled == 100, "all clients returned: %d + %d", ok.load(), cancelled.load());
        CHECK(cancelled > 0, "some cancels happened");
        CHECK(eng.used() == 0, "leak: %d blocks in use after load", eng.used());
        Metrics m = srv.metrics();
        CHECK(m.requests_done == 100 && m.active == 0 && m.queued == 0, "metrics after load: done %lld active %d queued %d",
              (long long)m.requests_done, m.active, m.queued);
        std::printf("load: 100 requests, %d cancelled, %lld preemptions, %lld steps\n", cancelled.load(),
                    (long long)m.preemptions, (long long)m.steps);
    }
    {   // shutdown ends in-flight requests as cancelled and joins the thread
        FakeEngine eng(1, 1 << 20);
        SchedulerConfig cfg; cfg.eos_id = kEos;
        auto* srv = new Server(eng, cfg);
        auto h = srv->submit(req(10, 100000));
        int64_t t; h->next(t);
        delete srv;
        drain(*h);
        CHECK(h->done() && h->reason() == StopReason::kCancelled, "shutdown reason");
    }
    if (failures == 0) { std::printf("test_serve: all checks passed\n"); return 0; }
    std::printf("test_serve: %d FAILURES\n", failures);
    return 1;
}
