# LLM inference engine from scratch (C++ / CUDA)

A from-scratch inference engine for Llama-style models (Qwen2.5 0.5B and 1.5B today), built stage
by stage and documented as a blog series. No PyTorch, no cuDNN: the model loader, the transformer
forward pass, the tokenizer, the KV cache, every CUDA kernel and the batching scheduler are written
here. cuBLAS exists in the tree only behind a `--gemm=cublas` flag, as a bisection tool and a
yardstick; every headline number comes from kernels in this repo.

Hardware: one RTX 5070 Ti (16 GB, sm_120) on WSL2. Correctness is checked against HuggingFace
reference logits on every stage; performance is measured under locked clocks with provenance
(commit, clocks, temperature) stamped into every record in `bench/`.

## Results

Decode speed, greedy, one sequence, tokens per second (median of repeated runs, locked clocks):

| | Qwen2.5-0.5B, 256 tokens | Qwen2.5-1.5B, 256 tokens |
|---|---|---|
| Stage 3: naive CUDA kernels, full recompute per token | 2.9 | |
| Stage 4: + KV cache | 15.7 | 7.0 |
| Stage 5: + parallel attention | 36.5 | 17.0 |
| Stage 5: + row-parallel GEMV | 204.7 | 126.8 |
| Stage 5: + fused decode step, hoisted host work | **361.4** | **178.5** |
| same engine, matmuls routed through cuBLAS (fp32) | 164.9 | 94.3 |

Serving throughput with batching (Stage 6), 64 requests of 24 to 72 tokens, continuous batching,
generated tokens per second:

| slots | 1 | 2 | 4 | 8 | 16 | 32 |
|---|---|---|---|---|---|---|
| Qwen2.5-1.5B | 142 | 209 | 387 | 535 | **615** | 549 |
| Qwen2.5-0.5B | 223 | 375 | 692 | 1105 | 1326 | **1343** |

Step time on 1.5B goes from 7.2 ms at 1 row to 13.9 ms at 8 rows: the weights are read once per
step whatever the batch size, which is the whole reason batching works. Continuous batching beats
static batching by 10 to 22% on the same request set at 4 to 16 slots (`bench/stage6_serve_*.json`).

Paged KV cache (Stage 7), 1.5B, 16 slots, max_seq 4096: a 512 MB block pool serves the same
request set at the same 639 tok/s as the reserved layout, with a peak of 46 blocks (41 MB) in use
where reservation would hold 3.8 GB. A shared 64-token prompt prefix served from the prefix cache
takes throughput from 285 to 624 tok/s and halves blocks in use; at a 32 MB budget the scheduler
preempts (47 evictions) and still serves 442 tok/s (`bench/stage7_serve_*.json`).

On the 1.5B model the single-sequence decode GEMV reads weights at 791 GB/s against a measured achievable
784 GB/s (STREAM-style, `kernels/bench/stream_bench.cu`): the kernel sits at the memory-bandwidth
roofline. The theoretical single-sequence ceiling is 254 tok/s; the engine reaches 178, and the gap
is host overhead (kernel launches, the logits copy, CPU argmax), not kernel time. Getting past that
ceiling needs more tokens per weight read, which is what the batching above does.

Per-kernel evidence (Nsight Compute, `bench/stage5_ncu_counters.json`), naive kernel -> new kernel:

| kernel | compute pipelines busy | memory bandwidth used |
|---|---|---|
| GEMV 1536 x 8960 (1.5B MLP down) | 0.33% -> 16.7% | 3.1% -> 85% |
| GEMV 151936 x 1536 (LM head) | 3.2% -> 27.1% | 30% -> 93% |
| cached attention, 1024 positions | 0.02% -> 2.1% | 11.6 ms -> 45 us per launch |

## What is built

| Stage | What | Where |
|---|---|---|
| 0 | CMake dual-target (CPU-only Mac / CUDA), CUDA-event timing harness with NVML clock and temperature capture, JSON bench records with commit + dirty-tree provenance, compute-sanitizer four-pass script, locked-clock script | `kernels/common/`, `tools/` |
| 1 | safetensors loader (mmap, zero-copy tensor views), `config.json` parser, HF oracle script dumping goldens with per-layer taps, loader test asserting the exact tensor set and bit-exact spot values | `src/loader.*`, `src/model*.{h,cpp}`, `tools/dump_logits.py` |
| 2 | CPU forward pass: RMSNorm, RoPE, GQA attention, SwiGLU, tied LM head; validated layer by layer against HF taps | `src/forward.cpp` |
| 3 | Naive CUDA port, one kernel per op, `--gemm=mine\|cublas` bisection flag at every matmul call site | `kernels/ops/`, `kernels/model_gpu.cu` |
| 4 | KV cache with prefill/decode split; byte-level BPE tokenizer matching HF ids exactly on an English/code corpus; temperature / top-k / top-p sampling; streaming UTF-8-safe chat CLI | `src/session.cpp`, `kernels/session_gpu.cu`, `src/tokenizer.cpp`, `src/main_chat.cpp` |
| 5 | Profile-driven kernel optimization: parallel cached attention (one block per head), row-parallel GEMV (one block per output row, interleaved loads, tree reduce), fused decode step (16 -> 9 launches per layer, bit-identical), hoisted per-token host work; STREAM bandwidth bench; roofline; ncu counters; second model (1.5B) with zero engine changes | `kernels/ops/cache.cu`, `kernels/ops/linear.cu`, `kernels/bench/`, `tools/nsys_kernel_summary.py`, `tools/ncu_stage5.sh` |
| 6 | Batched decode (weights read once per step for up to 32 sequences) and a continuous-batching scheduler with a preallocated event ring; throughput-vs-latency sweep | `kernels/ops/batch.cu`, `kernels/batch_gpu.cu`, `src/scheduler.cpp`, `src/main_serve_bench.cpp` |
| 7 | Paged KV cache: 16-position blocks, per-sequence block tables, free-list allocator with refcounts, prefix cache with copy-on-write, recompute-style preemption under a byte budget | `src/block_pool.cpp`, `kernels/ops/batch.cu`, `kernels/batch_gpu.cu` |

Planned: fused flash-style attention over the paged layout
and CUDA graphs (8), HTTP serving with cancellation (9), int8/int4 quantization (10), speculative
decoding (11).

## Correctness discipline

- `tests/test_forward`, `tests/test_forward_gpu`: the golden ladder. Every intermediate (post-norm,
  Q/K/V, post-RoPE, attention output, MLP, residual stream per layer, logits) is compared in
  execution order against HuggingFace dumps; the first failing rung names the broken op. Greedy
  continuations must match token for token.
- `tests/test_kv_cache`: cached decode must produce the identical token stream to full recompute,
  for every combination of GEMM path, attention kernel and decode-step variant.
- `tests/test_ops_gpu`: each optimized kernel against a double-precision host reference at edge sizes
  (lengths 1, 33, 257, 4097; widths not divisible by the thread count), with NaN poison past the
  live cache rows, canary words past every output, and a `--selftest` mode that swaps in
  deliberately broken kernels and requires the sweep to reject them.
- `tests/test_batch_gpu`: batched decode must equal single-sequence decode per row, both in
  lockstep and with requests joining and leaving a running batch; also under a KV budget that
  forces preemption, and with a shared prefix served from the prefix cache.
- `tests/test_block_pool`, `tests/test_scheduler`: allocator, prefix cache, copy-on-write and
  scheduler policy (admission, retirement, preemption) on the Mac against a fake engine.
- `tools/sanitize.sh`: memcheck, racecheck, initcheck, synccheck, clean before a stage is called done.
- Bisection: a golden-ladder failure on `--gemm=mine` that disappears on `--gemm=cublas` is a GEMV
  bug; one that persists is in RoPE, attention, norms or glue.

## Blog series

"LLM Inference Engine from scratch" on Medium: https://medium.com/@ryan___

1. Loading the model
2. Forward pass
3. Engine meets GPU
4. KV cache
5. Tokenization
6. Kernel optimization (to be written)
7. What limits speed once the kernels are fast (to be written)
8. Batching and continuous batching (to be written)
9. Paged KV cache (to be written)

## Building and running

```
# Mac (CPU-only targets) or the CUDA box (all targets):
cmake -S . -B build && cmake --build build -j
ctest --test-dir build

# Models and goldens (offline HF tooling, once):
python tools/download_model.py Qwen/Qwen2.5-0.5B-Instruct
python tools/dump_logits.py && python tools/dump_logits.py --greedy
python tools/dump_tokenizer_tests.py

# Chat (GPU, my kernels):
./build/main_chat gpu mine

# Decode benchmark:  prompt n_new device gemm cache attn step
./build/main_generate 3 256 gpu mine on par fused

# Second model: every test and tool honours LLM_MODEL.
LLM_MODEL=Qwen2.5-1.5B-Instruct ./build/test_forward_gpu

# Serving sweep (Stages 6-7): key=value args
./build/main_serve_bench slots=1,2,4,8,16,32
./build/main_serve_bench slots=16 max_seq=4096 budget_mb=64 prefix=64
```

Model weights, goldens and profiler reports are not committed; `bench/*.json` records are.
