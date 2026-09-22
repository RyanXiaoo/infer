// serve.h — the serving core (Stage 9): a thread that runs the scheduler
// forever, request handles that stream tokens out, cancellation that gives KV
// blocks back mid-generation, and metrics. No HTTP here: main_server.cpp maps
// HTTP onto this; tests drive it with the fake engine on the Mac.
//
// Threads: clients call submit()/cancel()/next() from any thread. One engine
// thread owns the scheduler and the GPU: it drains nothing itself (the
// scheduler's queue is thread-safe), steps while anything is active, and
// sleeps on a condition variable when idle. Tokens reach a handle through the
// scheduler's on_token callback on the engine thread; next() waits on the
// handle's own condition variable.
//
// Cancellation: handle->cancel() marks the request in the scheduler; the next
// step's retire pass frees its slot and blocks (at most one step late). A
// request cancelled while still queued is never prefilled.

#pragma once

#include "scheduler.h"

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

namespace llm {

class RequestHandle {
public:
    // Blocks until a token is available; false once the request is finished
    // and all tokens were consumed.
    bool next(int64_t& token);
    void cancel();
    bool done() const;
    StopReason reason() const;
    int64_t id() const { return id_; }

private:
    friend class Server;
    int64_t id_ = 0;
    std::function<void(int64_t)> cancel_fn_;
    mutable std::mutex mu_;
    std::condition_variable cv_;
    std::deque<int64_t> tokens_;
    bool done_ = false;
    StopReason reason_ = StopReason::kEos;
    void push(int64_t token);
    void finish(StopReason r);
};

struct Metrics {
    int queued = 0, active = 0;
    double tokens_per_s = 0;          // generated tokens over the last second
    double ttft_p50_ms = 0, ttft_p95_ms = 0;
    double latency_p50_ms = 0, latency_p95_ms = 0;
    int64_t requests_done = 0, tokens_total = 0, preemptions = 0, cancellations = 0;
    int blocks_in_use = 0, blocks_total = 0;
    int64_t steps = 0;
};

class Server {
public:
    Server(BatchEngine& engine, SchedulerConfig cfg);
    ~Server();   // shutdown()
    Server(const Server&) = delete;
    Server& operator=(const Server&) = delete;

    // The request's id is assigned here (r.id is overwritten).
    std::shared_ptr<RequestHandle> submit(Request r);
    Metrics metrics() const;
    void shutdown();   // stops the engine thread; in-flight requests finish as cancelled

private:
    BatchEngine& eng_;
    std::unique_ptr<Scheduler> sch_;
    std::thread loop_;
    std::atomic<bool> running_{true};
    mutable std::mutex mu_;                    // handles_, idle wakeup, metrics windows
    std::condition_variable wake_;
    std::vector<std::pair<int64_t, std::shared_ptr<RequestHandle>>> handles_;
    std::atomic<int64_t> next_id_{1};
    std::atomic<int> queued_{0};
    // metrics
    std::deque<double> token_times_ms_;      // last second's token timestamps
    std::deque<double> ttft_ms_, latency_ms_; // last 256 completions
    int64_t done_ = 0, tokens_ = 0, cancellations_ = 0;
    double t0_ms_;

    void loop();
    void on_token(int64_t id, int64_t token);
    void on_done(const Completed& c);
    std::shared_ptr<RequestHandle> find(int64_t id);
    double now_ms() const;
};

} // namespace llm
