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
    static const char* names[] = {"", "admit", "step", "retire"};
    for (size_t i = 0; i < size(); i++) {
        const Event& e = at(i);
        std::fprintf(f,
                     "{\"kind\":\"%s\",\"step\":%lld,\"t_ms\":%.3f,\"slot\":%d,"
                     "\"request\":%lld,\"batch\":%u,\"tokens\":%lld,\"dur_ms\":%.3f,"
                     "\"reason\":%u}\n",
                     names[int(e.kind)], (long long)e.step, e.t_ms, e.slot,
                     (long long)e.request_id, unsigned(e.batch_size), (long long)e.tokens,
                     e.dur_ms, unsigned(e.reason));
    }
    std::fclose(f);
}

Scheduler::Scheduler(BatchEngine& engine, SchedulerConfig cfg)
    : eng_(engine), cfg_(cfg), slots_(size_t(engine.slots())), events_(cfg.event_capacity),
      t0_ms_(wall_ms()) {}

double Scheduler::now_ms() const { return wall_ms() - t0_ms_; }

void Scheduler::submit(Request r) { queue_.push_back(std::move(r)); }

bool Scheduler::idle() const { return queue_.empty() && active_count_ == 0; }

void Scheduler::admit() {
    // Static batching: only refill once the whole batch has drained.
    if (!cfg_.continuous && active_count_ > 0) return;
    for (int s = 0; s < int(slots_.size()) && !queue_.empty(); s++) {
        if (slots_[s].used) continue;
        if (queue_.front().arrival_step > step_) break;   // FIFO: nothing later is due
        Request r = std::move(queue_.front());
        queue_.pop_front();
        if (int64_t(r.prompt.size()) + r.max_new > eng_.max_seq())
            throw std::runtime_error("request does not fit the cache (prompt + max_new > max_seq)");

        Active& a = slots_[s];
        a.used = true;
        a.req = std::move(r);
        a.out.clear();
        a.prompt_len = int64_t(a.req.prompt.size());
        a.admitted_step = step_;
        a.admitted_ms = now_ms();
        a.last_token = eng_.prefill(s, a.req.prompt);
        a.out.push_back(a.last_token);
        a.first_token_ms = now_ms();
        active_count_++;
        events_.push(Event{EventKind::kAdmit, 0, uint16_t(active_count_), s, step_,
                           a.req.id, a.prompt_len, a.admitted_ms, a.first_token_ms - a.admitted_ms});
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
    const int64_t n = int64_t(c.tokens.size());
    completed_.push_back(std::move(c));
    a = Active{};
    active_count_--;
    events_.push(Event{EventKind::kRetire, reason, uint16_t(active_count_), s, step_,
                       completed_.back().id, n, completed_.back().finished_ms, 0.0});
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

    if (!rows.empty()) {
        std::vector<int64_t> next = eng_.step(rows);
        for (size_t i = 0; i < rows.size(); i++) {
            Active& a = slots_[rows[i].slot];
            a.last_token = next[i];
            a.out.push_back(next[i]);
        }
    }
    events_.push(Event{EventKind::kStep, 0, uint16_t(rows.size()), -1, step_, -1,
                       int64_t(rows.size()), t_start, now_ms() - t_start});
    step_++;
    return int(rows.size());
}

} // namespace llm
