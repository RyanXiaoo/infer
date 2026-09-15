#include "sampler.h"

#include <algorithm>
#include <cmath>

namespace llm {

int64_t Sampler::sample(const std::vector<float>& logits) {
    const int64_t V = int64_t(logits.size());

    // Greedy fast path (also what temperature 0 means).
    if (temperature <= 0.0f) {
        int64_t best = 0;
        for (int64_t v = 1; v < V; v++)
            if (logits[v] > logits[best]) best = v;
        return best;
    }

    // Candidate set as (logit, id), scaled by temperature.
    std::vector<std::pair<float, int64_t>> c;
    c.reserve(V);
    for (int64_t v = 0; v < V; v++) c.emplace_back(logits[v] / temperature, v);

    // top-k: keep the k highest-logit candidates.
    if (top_k > 0 && top_k < int(c.size())) {
        std::nth_element(c.begin(), c.begin() + top_k, c.end(),
                         [](auto& a, auto& b) { return a.first > b.first; });
        c.resize(top_k);
    }

    // Softmax over the candidates (max-subtracted).
    float mx = c[0].first;
    for (auto& e : c) mx = std::max(mx, e.first);
    double sum = 0.0;
    for (auto& e : c) { e.first = std::exp(e.first - mx); sum += e.first; }
    for (auto& e : c) e.first = float(e.first / sum);

    // top-p (nucleus): sort desc, keep the smallest prefix whose mass >= top_p.
    if (top_p < 1.0f) {
        std::sort(c.begin(), c.end(), [](auto& a, auto& b) { return a.first > b.first; });
        double cum = 0.0;
        size_t keep = 0;
        for (; keep < c.size(); keep++) {
            cum += c[keep].first;
            if (cum >= top_p) { keep++; break; }
        }
        c.resize(keep);
        // renormalize
        double s = 0.0;
        for (auto& e : c) s += e.first;
        for (auto& e : c) e.first = float(e.first / s);
    }

    // Sample from the (renormalized) categorical distribution.
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    float r = dist(rng);
    double cum = 0.0;
    for (auto& e : c) {
        cum += e.first;
        if (r <= cum) return e.second;
    }
    return c.back().second;   // float rounding fallback
}

} // namespace llm
