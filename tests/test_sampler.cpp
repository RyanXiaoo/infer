// test_sampler.cpp — temperature 0 is argmax; a fixed seed is reproducible;
// top-k and top-p restrict the support set as expected.

#include "../src/sampler.h"

#include <cstdio>
#include <set>
#include <vector>

static int failures = 0;
#define CHECK(c, ...) do { if(!(c)){ std::printf("FAIL %d: ",__LINE__); std::printf(__VA_ARGS__); std::printf("\n"); failures++; } } while(0)

int main() {
    std::vector<float> logits = {1.0f, 3.0f, 2.0f, 0.5f, 5.0f, 4.0f};  // argmax = index 4

    // temperature 0 -> argmax
    {
        llm::Sampler s(123);
        s.temperature = 0.0f;
        CHECK(s.sample(logits) == 4, "temp0 not argmax");
    }

    // fixed seed -> reproducible sequence
    {
        llm::Sampler a(42), b(42);
        a.temperature = b.temperature = 1.0f;
        for (int i = 0; i < 20; i++)
            CHECK(a.sample(logits) == b.sample(logits), "same seed diverged at %d", i);
    }

    // top_k = 1 -> always the argmax regardless of temperature
    {
        llm::Sampler s(7);
        s.temperature = 2.0f;
        s.top_k = 1;
        for (int i = 0; i < 50; i++) CHECK(s.sample(logits) == 4, "top_k=1 not argmax");
    }

    // top_k = 2 -> only the two highest-logit ids (4 and 5) ever appear
    {
        llm::Sampler s(7);
        s.temperature = 2.0f;
        s.top_k = 2;
        std::set<int64_t> seen;
        for (int i = 0; i < 500; i++) seen.insert(s.sample(logits));
        for (int64_t id : seen) CHECK(id == 4 || id == 5, "top_k=2 produced id %lld", (long long)id);
    }

    // top_p very small -> only the argmax (its mass alone exceeds a tiny p)
    {
        llm::Sampler s(7);
        s.temperature = 1.0f;
        s.top_p = 0.01f;
        for (int i = 0; i < 50; i++) CHECK(s.sample(logits) == 4, "tiny top_p not argmax");
    }

    if (failures == 0) { std::printf("test_sampler: all checks passed\n"); return 0; }
    std::printf("test_sampler: %d FAILURES\n", failures);
    return 1;
}
