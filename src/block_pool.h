// block_pool.h — paged KV cache bookkeeping (Stage 7). Plain C++, no CUDA.
//
// The KV cache is a pool of fixed-size blocks (kBlockSize positions each, in
// every layer). A sequence owns an ordered list of block ids (its block table)
// and takes a new block only when it fills the last one, so memory in use
// tracks the tokens actually present instead of a worst-case reservation.
//
// Blocks carry a reference count so two sequences can share the blocks that
// hold an identical prefix (prefix cache, keyed by a hash chain over the
// block's token ids). Writing into a shared block first gives the writer a
// private copy (copy-on-write); the copy itself is a device operation the GPU
// engine performs when this class says so.
//
// Nothing here touches device memory: BlockPool decides WHICH block, the
// engine moves bytes. That keeps the policy testable on the Mac.

#pragma once

#include <cstdint>
#include <unordered_map>
#include <vector>

namespace llm {

constexpr int kBlockSize = 16;

class BlockPool {
public:
    explicit BlockPool(int n_blocks);

    int n_blocks() const { return int(refcount_.size()); }
    int n_free() const { return int(free_.size()); }
    int n_in_use() const { return n_blocks() - n_free(); }
    int peak_in_use() const { return peak_; }
    int refcount(int block) const { return refcount_[size_t(block)]; }

    int alloc();                 // -1 when the pool is empty
    void share(int block);       // refcount++
    void release(int block);     // refcount--; back on the free list at zero

    // Prefix cache: full blocks whose contents are determined by (the chain of)
    // token ids can be reused by any sequence with the same prefix.
    // hash = chain(prev_hash, ids[kBlockSize]); lookup returns -1 on a miss.
    static uint64_t chain(uint64_t prev, const int64_t* ids, int n);
    int lookup(uint64_t hash) const;
    void insert(uint64_t hash, int block);
    void forget(int block);      // called when a block's refcount hits zero

private:
    std::vector<int> refcount_;
    std::vector<int> free_;
    std::unordered_map<uint64_t, int> prefix_;
    std::vector<uint64_t> hash_of_;   // block -> hash it is registered under (0 = none)
    int peak_ = 0;
};

// One sequence's view of the pool.
struct BlockTable {
    std::vector<int> blocks;     // block id per kBlockSize positions
    std::vector<uint64_t> hashes; // hash chain per FULL block (size <= blocks.size())
    int64_t length = 0;          // positions written

    int block_of(int64_t pos) const { return blocks[size_t(pos / kBlockSize)]; }
    static int64_t blocks_needed(int64_t positions) {
        return (positions + kBlockSize - 1) / kBlockSize;
    }
};

// What the engine must do before a position can be written. Returned by
// BlockPool-level helpers below so the engine stays a straight-line executor.
struct WritePlan {
    bool ok = true;              // false: out of blocks (caller preempts)
    bool cow = false;            // copy `cow_from` into `block` first
    int cow_from = -1;
    int block = -1;              // block that will hold the position
};

// Ensure `t.blocks` covers position `pos` and that its block is writable by
// this sequence alone. Allocates a new block at a block boundary; on a shared
// block, allocates a private copy and reports the COW. Does not change length.
WritePlan plan_write(BlockPool& pool, BlockTable& t, int64_t pos);

// Prefix cache for prefill: for every full block of `ids` except the last one,
// reuse a cached block when the hash chain matches, else allocate. Returns the
// number of positions covered by reused blocks (a multiple of kBlockSize); the
// caller computes only from there, so at least one position is always
// computed. The trailing block is allocated fresh (never shared). On
// out-of-blocks, releases what it took and returns -1.
int64_t plan_prefill(BlockPool& pool, BlockTable& t, const std::vector<int64_t>& ids,
                     bool use_prefix_cache);

// Register the sequence's full blocks in the prefix cache (after prefill
// computed them) and release everything on retirement.
void register_prefix(BlockPool& pool, BlockTable& t, const std::vector<int64_t>& ids);
void release_all(BlockPool& pool, BlockTable& t);

} // namespace llm
