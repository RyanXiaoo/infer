// scheduler.h — continuous batching scheduler (Stage 6).
//
// The engine underneath serves S slots: prefill(slot, prompt) starts a
// sequence in a slot and returns its first generated token; step(rows) runs
// ONE decode step for every listed slot at once and returns each slot's next
// token. Decode cost is dominated by reading the weights, which the rows
// share, so a step with 8 rows costs little more than a step with 1: the
// scheduler's job is to keep the batch full.
//
// Continuous batching (Orca / vLLM): at every step, requests that have
// finished leave their slot and queued requests take the free slots
// immediately. Static batching waits for the whole batch to finish before
// admitting the next one; it is kept as the comparison baseline.
//
// Plain C++ (boundary rule): BatchEngine is an interface, so the scheduler is
// unit-tested on the Mac with a fake engine and driven on the GPU by GpuBatch.
//
// Event log: every admit, step and retire is written as a fixed-size struct
// into a preallocated ring buffer (no allocation on the hot path) and drained
// to a JSON-lines file afterwards. This is the visualizer's input (Stage 8).

#pragma once

#include <cstdint>
#include <deque>
#include <functional>
#include <mutex>
#include <string>
#include <unordered_set>
#include <vector>

namespace llm {

// Per-request sampling. temperature 0 = greedy (exact argmax); otherwise the
// engine samples logits / temperature with a per-request seeded RNG.
struct SampleParams {
    float temperature = 0.0f;
    uint64_t seed = 0;
};

struct StepRow {
    int slot;
    int64_t token;   // the token to feed (the previous output)
    SampleParams sample;
};

class BatchEngine {
public:
    virtual ~BatchEngine() = default;
    virtual int slots() const = 0;
    virtual int64_t max_seq() const = 0;
    // KV memory: can a prompt of this length be admitted right now?
    virtual bool has_room(int64_t prompt_len) const = 0;
    // Runs the prompt into `slot`'s cache; returns the first generated token,
    // or -1 if the cache is out of memory (nothing is kept).
    virtual int64_t prefill(int slot, const std::vector<int64_t>& prompt,
                            SampleParams sample = {}) = 0;
    // One decode step for all rows; out[i] is rows[i]'s next token, or -1 if
    // that row could not get cache memory (its state is unchanged).
    virtual std::vector<int64_t> step(const std::vector<StepRow>& rows) = 0;
    // Engines that can emit several tokens per step (speculative decoding)
    // override this; out[i] is empty when the row could not proceed.
    virtual std::vector<std::vector<int64_t>> step_multi(const std::vector<StepRow>& rows) {
        std::vector<int64_t> one = step(rows);
        std::vector<std::vector<int64_t>> out(one.size());
        for (size_t i = 0; i < one.size(); i++) if (one[i] >= 0) out[i] = {one[i]};
        return out;
    }
    // Frees the slot's cache memory.
    virtual void release(int slot) = 0;
    virtual int blocks_in_use() const = 0;
    virtual int blocks_total() const = 0;
};

struct Request {
    int64_t id = 0;
    std::vector<int64_t> prompt;
    int max_new = 32;
    int64_t arrival_step = 0;   // scheduler step at which it may be admitted
    SampleParams sample;
    // Per-request stop token; kUseDefault = the scheduler config's eos_id.
    static constexpr int64_t kUseDefault = INT64_MIN;
    int64_t eos_id = kUseDefault;
    // Set by preemption: tokens generated before eviction. On re-admission the
    // prompt already includes them and generation resumes after them.
    std::vector<int64_t> generated_so_far;
};

enum class StopReason : uint8_t { kEos = 0, kMaxNew = 1, kCacheFull = 2, kCancelled = 3 };

struct Completed {
    int64_t id = 0;
    StopReason reason = StopReason::kEos;
    std::vector<int64_t> tokens;   // generated ids, including eos if hit
    int64_t admitted_step = 0, finished_step = 0;
    double admitted_ms = 0, first_token_ms = 0, finished_ms = 0;   // wall clock
    int preemptions = 0;
};

enum class EventKind : uint8_t { kAdmit = 1, kStep = 2, kRetire = 3, kPreempt = 4 };

// Fixed-size, POD: written into the ring buffer without allocating.
struct Event {
    EventKind kind;
    uint8_t reason;        // retire: 0 = eos, 1 = max_new, 2 = cache full
    uint16_t batch_size;   // step: rows decoded; admit/retire: batch size after
    int32_t slot;
    int64_t step;
    int64_t request_id;    // admit/retire
    int64_t tokens;        // retire: generated count; step: rows*1
    double t_ms;           // wall clock since scheduler construction
    double dur_ms;         // step: prefill+decode time of this step
    int32_t pool_in_use;   // KV blocks in use after the event
    int32_t pool_total;
};

class EventRing {
public:
    explicit EventRing(size_t capacity) : buf_(capacity) {}
    void push(const Event& e) {
        buf_[head_ % buf_.size()] = e;
        head_++;
        if (head_ - tail_ > buf_.size()) { tail_ = head_ - buf_.size(); dropped_++; }
    }
    size_t size() const { return head_ - tail_; }
    size_t dropped() const { return dropped_; }
    const Event& at(size_t i) const { return buf_[(tail_ + i) % buf_.size()]; }
    // JSON lines, one event per line.
    void write_jsonl(const std::string& path) const;

private:
    std::vector<Event> buf_;
    size_t head_ = 0, tail_ = 0, dropped_ = 0;
};

struct SchedulerConfig {
    bool continuous = true;    // false = static batching baseline
    int64_t eos_id = -1;       // -1 = never stop on eos
    size_t event_capacity = 1 << 16;
    // Called from step() on the scheduler's thread: every generated token
    // (including the prefill's first token) and every completion.
    std::function<void(int64_t request_id, int64_t token)> on_token;
    std::function<void(const Completed&)> on_done;
};

class Scheduler {
public:
    Scheduler(BatchEngine& engine, SchedulerConfig cfg);

    void submit(Request r);            // enqueue (FIFO); safe from any thread
    void cancel(int64_t request_id);   // safe from any thread; takes effect next step
    bool idle() const;                 // no queued and no active requests
    // Runs retire -> admit -> decode once. Returns the number of rows decoded.
    int step();
    void run_until_idle() { while (!idle()) step(); }

    const std::vector<Completed>& completed() const { return completed_; }
    int preemptions() const { return preemptions_; }
    const EventRing& events() const { return events_; }
    int64_t steps() const { return step_; }
    int active() const { return active_count_; }

private:
    struct Active {
        bool used = false;
        Request req;
        std::vector<int64_t> out;
        int64_t last_token = 0;
        int64_t prompt_len = 0;
        int64_t admitted_step = 0;
        double admitted_ms = 0, first_token_ms = 0;
        int preemptions = 0;
    };
    BatchEngine& eng_;
    SchedulerConfig cfg_;
    mutable std::mutex mu_;            // guards queue_ and cancelled_
    std::deque<Request> queue_;
    std::unordered_set<int64_t> cancelled_;
    std::vector<Active> slots_;
    std::vector<Completed> completed_;
    EventRing events_;
    int64_t step_ = 0;
    int active_count_ = 0;
    int preemptions_ = 0;
    double t0_ms_;

    double now_ms() const;
    void admit();
    void retire(int slot, StopReason reason);
    void finish_unstarted(Request r, StopReason reason);
    // Evicts the most recently admitted active request back to the queue
    // front (recompute-style preemption). Returns its slot, or -1 if none.
    int preempt_one();
    Event ev(EventKind k, uint8_t reason, int slot, int64_t req, int64_t tokens, double t,
             double dur) const;
};

} // namespace llm
