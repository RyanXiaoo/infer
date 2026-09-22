// serve.cpp — see serve.h.

#include "serve.h"

#include <algorithm>
#include <chrono>

namespace llm {

namespace {
double wall_ms() {
    using namespace std::chrono;
    return duration<double, std::milli>(steady_clock::now().time_since_epoch()).count();
}
double pct(std::deque<double> v, double p) {
    if (v.empty()) return 0.0;
    std::sort(v.begin(), v.end());
    return v[std::min(v.size() - 1, size_t(p * double(v.size() - 1) + 0.5))];
}
} // namespace

// ---------------------------------------------------------------- handle
bool RequestHandle::next(int64_t& token) {
    std::unique_lock<std::mutex> lock(mu_);
    cv_.wait(lock, [&] { return !tokens_.empty() || done_; });
    if (tokens_.empty()) return false;
    token = tokens_.front();
    tokens_.pop_front();
    return true;
}
void RequestHandle::cancel() { if (cancel_fn_) cancel_fn_(id_); }
bool RequestHandle::done() const { std::lock_guard<std::mutex> lock(mu_); return done_; }
StopReason RequestHandle::reason() const { std::lock_guard<std::mutex> lock(mu_); return reason_; }
void RequestHandle::push(int64_t token) {
    { std::lock_guard<std::mutex> lock(mu_); tokens_.push_back(token); }
    cv_.notify_all();
}
void RequestHandle::finish(StopReason r) {
    { std::lock_guard<std::mutex> lock(mu_); done_ = true; reason_ = r; }
    cv_.notify_all();
}

// ---------------------------------------------------------------- server
Server::Server(BatchEngine& engine, SchedulerConfig cfg) : eng_(engine), t0_ms_(wall_ms()) {
    cfg.on_token = [this](int64_t id, int64_t tok) { on_token(id, tok); };
    cfg.on_done = [this](const Completed& c) { on_done(c); };
    sch_ = std::make_unique<Scheduler>(engine, cfg);
    loop_ = std::thread([this] { loop(); });
}

Server::~Server() { shutdown(); }

double Server::now_ms() const { return wall_ms() - t0_ms_; }

std::shared_ptr<RequestHandle> Server::submit(Request r) {
    auto h = std::make_shared<RequestHandle>();
    h->id_ = r.id = next_id_++;
    h->cancel_fn_ = [this](int64_t id) { sch_->cancel(id); wake_.notify_all(); };
    {
        std::lock_guard<std::mutex> lock(mu_);
        handles_.emplace_back(h->id_, h);
    }
    queued_++;
    sch_->submit(std::move(r));
    wake_.notify_all();
    return h;
}

std::shared_ptr<RequestHandle> Server::find(int64_t id) {
    std::lock_guard<std::mutex> lock(mu_);
    for (auto& [hid, h] : handles_) if (hid == id) return h;
    return nullptr;
}

void Server::on_token(int64_t id, int64_t token) {
    if (auto h = find(id)) {
        // The prefill's first token also marks admission for the metrics.
        h->push(token);
    }
    std::lock_guard<std::mutex> lock(mu_);
    const double t = now_ms();
    token_times_ms_.push_back(t);
    while (!token_times_ms_.empty() && token_times_ms_.front() < t - 1000.0) token_times_ms_.pop_front();
    tokens_++;
}

void Server::on_done(const Completed& c) {
    if (auto h = find(c.id)) h->finish(c.reason);
    std::lock_guard<std::mutex> lock(mu_);
    handles_.erase(std::remove_if(handles_.begin(), handles_.end(),
                                  [&](auto& p) { return p.first == c.id; }), handles_.end());
    queued_--;
    done_++;
    if (c.reason == StopReason::kCancelled) cancellations_++;
    if (!c.tokens.empty()) {
        // admitted_ms is when prefill started; the request was submitted at or
        // before that, so this is a lower bound on true TTFT (queue wait is
        // in latency_ms_ via finished - submitted only when known).
        ttft_ms_.push_back(c.first_token_ms - c.admitted_ms);
        latency_ms_.push_back(c.finished_ms - c.admitted_ms);
        while (ttft_ms_.size() > 256) ttft_ms_.pop_front();
        while (latency_ms_.size() > 256) latency_ms_.pop_front();
    }
}

void Server::loop() {
    while (running_) {
        if (sch_->idle()) {
            std::unique_lock<std::mutex> lock(mu_);
            wake_.wait_for(lock, std::chrono::milliseconds(5), [&] { return !running_ || !sch_->idle(); });
            continue;
        }
        sch_->step();
    }
    // Shutdown: everything still in flight ends as cancelled.
    std::vector<std::shared_ptr<RequestHandle>> left;
    { std::lock_guard<std::mutex> lock(mu_); for (auto& p : handles_) left.push_back(p.second); handles_.clear(); }
    for (auto& h : left) h->finish(StopReason::kCancelled);
}

void Server::shutdown() {
    if (!running_.exchange(false)) return;
    wake_.notify_all();
    if (loop_.joinable()) loop_.join();
}

Metrics Server::metrics() const {
    Metrics m;
    std::lock_guard<std::mutex> lock(mu_);
    m.active = sch_->active();
    m.queued = std::max(0, queued_.load() - m.active);
    m.tokens_per_s = double(token_times_ms_.size());
    m.ttft_p50_ms = pct(ttft_ms_, 0.5); m.ttft_p95_ms = pct(ttft_ms_, 0.95);
    m.latency_p50_ms = pct(latency_ms_, 0.5); m.latency_p95_ms = pct(latency_ms_, 0.95);
    m.requests_done = done_; m.tokens_total = tokens_; m.cancellations = cancellations_;
    m.preemptions = sch_->preemptions();
    m.blocks_in_use = eng_.blocks_in_use(); m.blocks_total = eng_.blocks_total();
    m.steps = sch_->steps();
    return m;
}

} // namespace llm
