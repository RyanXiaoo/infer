// test_block_pool.cpp — paged KV bookkeeping (Stage 7), Mac-only logic.

#include "../src/block_pool.h"

#include <cstdio>
#include <numeric>
#include <vector>

namespace {
int failures = 0;
#define CHECK(cond, ...)                                                        \
    do { if (!(cond)) { failures++; std::printf("FAIL %s:%d: ", __FILE__, __LINE__); \
         std::printf(__VA_ARGS__); std::printf("\n"); } } while (0)

std::vector<int64_t> ids(int64_t n, int64_t base) {
    std::vector<int64_t> v(static_cast<size_t>(n), 0);
    std::iota(v.begin(), v.end(), base);
    return v;
}
} // namespace

int main() {
    using namespace llm;

    {   // alloc / release / reuse
        BlockPool pool(3);
        int a = pool.alloc(), b = pool.alloc(), c = pool.alloc();
        CHECK(a == 0 && b == 1 && c == 2, "ids %d %d %d", a, b, c);
        CHECK(pool.alloc() == -1, "pool should be empty");
        CHECK(pool.n_in_use() == 3 && pool.peak_in_use() == 3, "in use %d", pool.n_in_use());
        pool.release(b);
        CHECK(pool.n_free() == 1 && pool.alloc() == 1, "freed block reused");
    }

    {   // per-position writes: new block every 16 positions, no COW when unshared
        BlockPool pool(8);
        BlockTable t;
        for (int64_t p = 0; p < 40; p++) {
            WritePlan w = plan_write(pool, t, p);
            CHECK(w.ok && !w.cow, "pos %lld plan", (long long)p);
            CHECK(w.block == t.blocks[size_t(p / kBlockSize)], "pos %lld block", (long long)p);
            t.length = p + 1;
        }
        CHECK(t.blocks.size() == 3 && pool.n_in_use() == 3, "3 blocks for 40 positions");
        release_all(pool, t);
        CHECK(pool.n_in_use() == 0, "leak after release_all");
    }

    {   // prefix cache: identical 32-token prefix shared; partial block never shared
        BlockPool pool(16);
        BlockTable a, b;
        std::vector<int64_t> prompt_a = ids(40, 100);      // 2 full blocks + 8
        std::vector<int64_t> prompt_b = ids(32, 100);      // same first 32 ids
        prompt_b.insert(prompt_b.end(), {7, 8, 9});
        CHECK(plan_prefill(pool, a, prompt_a, true) == 0, "first prefill reuses nothing");
        register_prefix(pool, a, prompt_a);
        CHECK(a.hashes.size() == 2, "2 full-block hashes");
        const int64_t reused = plan_prefill(pool, b, prompt_b, true);
        CHECK(reused == 32, "b reused %lld positions", (long long)reused);
        CHECK(b.blocks[0] == a.blocks[0] && b.blocks[1] == a.blocks[1], "b shares a's full blocks");
        CHECK(b.blocks[2] != a.blocks[2], "partial block is private");
        CHECK(pool.refcount(a.blocks[0]) == 2, "refcount 2 on shared block");
        CHECK(pool.n_in_use() == 4, "4 blocks: 2 shared + 2 partial, got %d", pool.n_in_use());

        // a different prefix reuses nothing, even if a LATER block would match
        BlockTable c;
        std::vector<int64_t> prompt_c = ids(32, 500);
        CHECK(plan_prefill(pool, c, prompt_c, true) == 0, "different prefix: no reuse");
        // an exact 32-token repeat of a's prefix shares only block 0: the last
        // block is always recomputed so prefill has a position to run
        BlockTable e;
        CHECK(plan_prefill(pool, e, ids(32, 100), true) == 16, "last block never shared");
        release_all(pool, e);

        // COW: b writes into position 16 (inside shared block 1) -> private copy
        b.length = 35;
        // first, appending at 35 lands in b's own partial block: no COW
        WritePlan w = plan_write(pool, b, 35);
        CHECK(w.ok && !w.cow && w.block == b.blocks[2], "append into private partial block");
        // now simulate a rewrite of a shared position
        WritePlan cw = plan_write(pool, b, 20);
        CHECK(cw.ok && cw.cow && cw.cow_from == a.blocks[1], "COW from shared block");
        CHECK(b.blocks[1] == cw.block && b.blocks[1] != a.blocks[1], "b now owns a copy");
        CHECK(pool.refcount(a.blocks[1]) == 1, "shared block back to refcount 1");
        CHECK(b.hashes.size() == 1, "b's hash chain truncated at the copied block");

        release_all(pool, a);
        CHECK(pool.refcount(b.blocks[0]) == 1, "b still holds block 0");
        release_all(pool, b);
        release_all(pool, c);
        CHECK(pool.n_in_use() == 0, "leak after everything released: %d", pool.n_in_use());
        // the prefix entries died with their blocks
        BlockTable d;
        CHECK(plan_prefill(pool, d, prompt_a, true) == 0, "cache entries gone after release");
        release_all(pool, d);
    }

    {   // out of blocks: prefill releases what it took; write reports failure
        BlockPool pool(2);
        BlockTable t;
        CHECK(plan_prefill(pool, t, ids(40, 0), false) == -1, "40 positions need 3 blocks");
        CHECK(pool.n_in_use() == 0 && t.blocks.empty(), "nothing leaked on failure");
        CHECK(plan_prefill(pool, t, ids(32, 0), false) == 0, "32 positions fit");
        WritePlan w = plan_write(pool, t, 32);
        CHECK(!w.ok, "third block unavailable");
        release_all(pool, t);
    }

    if (failures == 0) { std::printf("test_block_pool: all checks passed\n"); return 0; }
    std::printf("test_block_pool: %d FAILURES\n", failures);
    return 1;
}
