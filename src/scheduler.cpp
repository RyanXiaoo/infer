// scheduler.cpp — see scheduler.h.

#include "scheduler.h"

#include <chrono>
#include <cstdio>
#include <stdexcept>

namespace llm {

namespace {
double wall_ms() {
    using namespace std::chrono;
    return duration<double, std::milli>(steady_clock::now().time_since_epoch()).count();
}
} // namespace

void EventRing::write_jsonl(const std::string& path) const {
    FILE* f = std::fopen(path.c_str(), "w");
    if (!f) throw std::runtime_error("cannot write " + path);
    static const char* names[] = {"", "admit", "step", "retire", "preempt"};
    for (size_t i = 0; i < size(); i++) {
        const Event& e = at(i);
        std::fprintf(f,
                     "{\"kind\":\"%s\",\"step\":%lld,\"t_ms\":%.3f,\"slot\":%d,"
                     "\"request\":%lld,\"batch\":%u,\"tokens\":%lld,\"dur_ms\":%.3f,"
                     "\"reason\":%u,\"blocks_in_use\":%d,\"blocks_total\":%d}\n",
                     names[int(e.kind)], (long long)e.step, e.t_ms, e.slot,
                     (long long)e.request_id, unsigned(e.batch_size), (long long)e.tokens,
                     e.dur_ms, unsigned(e.reason), e.pool_in_use, e.pool_total);
    }
    std::fclose(f);
}

Scheduler::Scheduler(BatchEngine& engine, SchedulerConfig cfg)
    : eng_(engine), cfg_(cfg), slots_(size_t(engine.slots())), events_(cfg.event_capacity),
      t0_ms_(wall_ms()) {}

double Scheduler::now_ms() const { return wall_ms() - t0_ms_; }

void Scheduler::submit(Request r) { queue_.push_back(std::move(r)); }

bool Scheduler::idle() const { return queue_.empty() && active_count_ == 0; }

Event Scheduler::ev(EventKind k, uint8_t reason, int slot, int64_t req, int64_t tokens,
                    double t, double dur) const {
    return Event{k, reason, uint16_t(active_count_), slot, step_, req, tokens, t, dur,
                 eng_.blocks_in_use(), eng_.blocks_total()};
}

void Scheduler::admit() {
    // Static batching: only refill once the whole batch has drained.
    if (!cfg_.continuous && active_count_ > 0) return;
    for (int s = 0; s < int(slots_.size()) && !queue_.empty(); s++) {
        if (slots_[s].used) continue;
        if (queue_.front().arrival_step > step_) break;   // FIFO: nothing later is due
        if (int64_t(queue_.front().prompt.size()) + queue_.front().max_new > eng_.max_seq())
            throw std::runtime_error("request does not fit the cache (prompt + max_new > max_seq)");
        if (!eng_.has_room(int64_t(queue_.front().prompt.size()))) break;   // wait for memory
        Request r = std::move(queue_.front());
        queue_.pop_front();

        Active& a = slots_[s];
        const double t_admit = now_ms();
        const int64_t first = eng_.prefill(s, r.prompt);
        if (first < 0) {   // out of memory mid-prefill: put it back, try next step
            queue_.push_front(std::move(r));
            break;
        }
        a.used = true;
        a.req = std::move(r);
        a.out = a.req.generated_so_far;
        a.preemptions = int(a.req.generated_so_far.empty() ? 0 : 1);
        a.prompt_len = int64_t(a.req.prompt.size()) - int64_t(a.out.size());
        a.admitted_step = step_;
        a.admitted_ms = t_admit;
        a.last_token = first;
        a.out.push_back(first);
        a.first_token_ms = now_ms();
        active_count_++;
        events_.push(ev(EventKind::kAdmit, 0, s, a.req.id, int64_t(a.req.prompt.size()),
                        a.admitted_ms, a.first_token_ms - a.admitted_ms));
    }
}

void Scheduler::retire(int s, uint8_t reason) {
    Active& a = slots_[s];
    Completed c;
    c.id = a.req.id;
    c.tokens = std::move(a.out);
    c.admitted_step = a.admitted_step;
    c.finished_step = step_;
    c.admitted_ms = a.admitted_ms;
    c.first_token_ms = a.first_token_ms;
    c.finished_ms = now_ms();
    c.preemptions = a.preemptions;
    const int64_t n = int64_t(c.tokens.size());
    completed_.push_back(std::move(c));
    eng_.release(s);
    a = Active{};
    active_count_--;
    events_.push(ev(EventKind::kRetire, reason, s, completed_.back().id, n,
                    completed_.back().finished_ms, 0.0));
}

int Scheduler::preempt_one() {
    int victim = -1;
    for (int s = 0; s < int(slots_.size()); s++)
        if (slots_[s].used && (victim < 0 || slots_[s].admitted_step > slots_[victim].admitted_step))
            victim = s;
    if (victim < 0) return -1;
    Active& a = slots_[victim];
    // A victim that has already finished (eos / max_new just produced) is
    // retired, which frees its memory just the same.
    if (a.last_token == cfg_.eos_id) { retire(victim, 0); return victim; }
    if (int(a.out.size()) >= a.req.max_new) { retire(victim, 1); return victim; }
    Request r = std::move(a.req);
    // Resume later from prompt + everything generated so far (recompute).
    r.prompt.insert(r.prompt.end(), a.out.begin() + int64_t(r.generated_so_far.size()), a.out.end());
    r.generated_so_far = std::move(a.out);
    r.arrival_step = 0;
    const int64_t id = r.id;
    queue_.push_front(std::move(r));
    eng_.release(victim);
    a = Active{};
    active_count_--;
    preemptions_++;
    events_.push(ev(EventKind::kPreempt, 0, victim, id, 0, now_ms(), 0.0));
    return victim;
}

int Scheduler::step() {
    const double t_start = now_ms();
    // Order matters: retire first so a slot freed this step is refilled this
    // step (continuous batching's whole point), then admit, then decode.
    for (int s = 0; s < int(slots_.size()); s++) {
        Active& a = slots_[s];
        if (!a.used) continue;
        if (a.last_token == cfg_.eos_id) retire(s, 0);
        else if (int(a.out.size()) >= a.req.max_new) retire(s, 1);
        else if (a.prompt_len + int64_t(a.out.size()) >= eng_.max_seq()) retire(s, 2);
    }
    admit();

    std::vector<StepRow> rows;
    for (int s = 0; s < int(slots_.size()); s++)
        if (slots_[s].used) rows.push_back({s, slots_[s].last_token});

    int decoded = 0;
    // Rows that cannot get cache memory are retried after evicting the most
    // recently admitted request, until every remaining row proceeds.
    while (!rows.empty()) {
        std::vector<int64_t> next = eng_.step(rows);
        std::vector<StepRow> retry;
        for (size_t i = 0; i < rows.size(); i++) {
            if (next[i] < 0) { retry.push_back(rows[i]); continue; }
            Active& a = slots_[rows[i].slot];
            a.last_token = next[i];
            a.out.push_back(next[i]);
            decoded++;
        }
        if (retry.empty()) break;
        const int victim = preempt_one();
        if (victim < 0) throw std::runtime_error("scheduler: out of KV memory with nothing to preempt");
        rows.clear();
        for (const StepRow& r : retry)
            if (r.slot != victim) rows.push_back(r);
    }
    events_.push(Event{EventKind::kStep, 0, uint16_t(decoded), -1, step_, -1, int64_t(decoded),
                       t_start, now_ms() - t_start, eng_.blocks_in_use(), eng_.blocks_total()});
    step_++;
    return decoded;
}

} // namespace llm
