// block_pool.cpp — see block_pool.h.

#include "block_pool.h"

#include <stdexcept>

namespace llm {

BlockPool::BlockPool(int n_blocks) : refcount_(size_t(n_blocks), 0), hash_of_(size_t(n_blocks), 0) {
    free_.reserve(size_t(n_blocks));
    for (int b = n_blocks - 1; b >= 0; b--) free_.push_back(b);   // alloc() hands out 0 first
}

int BlockPool::alloc() {
    if (free_.empty()) return -1;
    const int b = free_.back();
    free_.pop_back();
    refcount_[size_t(b)] = 1;
    if (n_in_use() > peak_) peak_ = n_in_use();
    return b;
}

void BlockPool::share(int block) {
    if (refcount_[size_t(block)] <= 0) throw std::logic_error("share of a free block");
    refcount_[size_t(block)]++;
}

void BlockPool::release(int block) {
    int& rc = refcount_[size_t(block)];
    if (rc <= 0) throw std::logic_error("release of a free block");
    if (--rc == 0) {
        forget(block);
        free_.push_back(block);
    }
}

uint64_t BlockPool::chain(uint64_t prev, const int64_t* ids, int n) {
    // FNV-1a over the previous hash and the ids; never returns 0 (0 = unset).
    uint64_t h = 1469598103934665603ull ^ prev;
    auto mix = [&](uint64_t v) {
        for (int i = 0; i < 8; i++) { h ^= (v >> (8 * i)) & 0xff; h *= 1099511628211ull; }
    };
    mix(prev);
    for (int i = 0; i < n; i++) mix(uint64_t(ids[i]));
    return h ? h : 1;
}

int BlockPool::lookup(uint64_t hash) const {
    auto it = prefix_.find(hash);
    return it == prefix_.end() ? -1 : it->second;
}

void BlockPool::insert(uint64_t hash, int block) {
    if (hash_of_[size_t(block)] == hash) return;
    prefix_[hash] = block;
    hash_of_[size_t(block)] = hash;
}

void BlockPool::forget(int block) {
    uint64_t& h = hash_of_[size_t(block)];
    if (h == 0) return;
    auto it = prefix_.find(h);
    if (it != prefix_.end() && it->second == block) prefix_.erase(it);
    h = 0;
}

WritePlan plan_write(BlockPool& pool, BlockTable& t, int64_t pos) {
    WritePlan p;
    const size_t bi = size_t(pos / kBlockSize);
    if (bi == t.blocks.size()) {
        const int b = pool.alloc();
        if (b < 0) { p.ok = false; return p; }
        t.blocks.push_back(b);
        p.block = b;
        return p;
    }
    if (bi > t.blocks.size()) throw std::logic_error("plan_write: position skips a block");
    const int b = t.blocks[bi];
    if (pool.refcount(b) > 1) {
        const int fresh = pool.alloc();
        if (fresh < 0) { p.ok = false; return p; }
        pool.release(b);
        t.blocks[bi] = fresh;
        if (bi < t.hashes.size()) t.hashes.resize(bi);   // no longer a cached full block
        p.cow = true;
        p.cow_from = b;
        p.block = fresh;
        return p;
    }
    p.block = b;
    return p;
}

int64_t plan_prefill(BlockPool& pool, BlockTable& t, const std::vector<int64_t>& ids,
                     bool use_prefix_cache) {
    const int64_t T = int64_t(ids.size());
    // Blocks eligible for sharing: full blocks that are not the LAST block of
    // the prompt. The last block is always computed locally so prefill has at
    // least one position to run (its logits give the first generated token).
    const int64_t full = (T - 1) / kBlockSize;
    int64_t reused = 0;
    uint64_t h = 0;
    bool chain_alive = use_prefix_cache;
    for (int64_t bi = 0; bi < full; bi++) {
        h = BlockPool::chain(h, ids.data() + bi * kBlockSize, kBlockSize);
        int b = chain_alive ? pool.lookup(h) : -1;
        if (b >= 0) {
            pool.share(b);
            reused += kBlockSize;
        } else {
            chain_alive = false;   // a miss breaks the prefix; later blocks cannot match
            b = pool.alloc();
            if (b < 0) { release_all(pool, t); return -1; }
        }
        t.blocks.push_back(b);
        t.hashes.push_back(h);
    }
    // The remaining positions (1..kBlockSize of them) get a private block.
    const int b = pool.alloc();
    if (b < 0) { release_all(pool, t); return -1; }
    t.blocks.push_back(b);
    t.length = T;
    return reused;
}

void register_prefix(BlockPool& pool, BlockTable& t, const std::vector<int64_t>& ids) {
    const int64_t full = int64_t(ids.size()) / kBlockSize;
    uint64_t h = 0;
    for (int64_t bi = 0; bi < full; bi++) {
        h = BlockPool::chain(h, ids.data() + bi * kBlockSize, kBlockSize);
        if (bi < int64_t(t.hashes.size())) t.hashes[size_t(bi)] = h; else t.hashes.push_back(h);
        if (pool.lookup(h) < 0) pool.insert(h, t.blocks[size_t(bi)]);
    }
}

void release_all(BlockPool& pool, BlockTable& t) {
    for (int b : t.blocks) pool.release(b);
    t.blocks.clear();
    t.hashes.clear();
    t.length = 0;
}

} // namespace llm
