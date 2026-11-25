# infer

An LLM inference engine written from scratch in C++ and CUDA, for Llama-style models
(Qwen2.5 0.5B, 1.5B and, quantized, 7B). No PyTorch, no cuDNN. The safetensors loader, the
forward pass, the BPE tokenizer, every CUDA kernel, the paged KV cache, the continuous-batching
scheduler, the HTTP server, the quantizer and the speculative decoder are all in this repo.
cuBLAS is present only behind a `--gemm=cublas` flag as a comparison path for bisection; every
number below comes from kernels written here.

Hardware: one RTX 5070 Ti (16 GB, sm_120) under WSL2. Correctness is checked against HuggingFace
logits at every layer. Performance is measured under locked clocks, and every record in `bench/`
carries the commit, clocks and temperature it was taken at.

Blog series: [LLM Inference Engine from scratch](#blog-series), one post per stage.

## Highlights

- **Single-sequence decode on Qwen2.5-1.5B went from 7 to 178 tok/s in bf16, 316 at int4.**
  The decode GEMV streams weights at 791 GB/s against a measured 784 GB/s achievable, so it sits
  at the memory roofline. What remains is host overhead, not kernel time.
- **Batched serving reaches 1339 tok/s at 16 concurrent sequences on 1.5B** with a paged KV
  cache (prefix sharing, copy-on-write, preemption under a byte budget), CUDA-graph replay of the
  decode step and flash-decoding attention.
- **OpenAI-compatible HTTP server** with SSE streaming, on-device sampling and cancellation that
  frees a dropped client's KV blocks within one scheduler step. 1148 tok/s over HTTP at 16
  clients, 53 ms time to first token.
- **int8 is lossless and Qwen2.5-7B fits on the 16 GB card.** Perplexity 8.339 to 8.345 at int8;
  7B runs at int4 (4.0 GB of weights, 126 tok/s) and int8 (271 tok/s at 16 slots).
- **Speculative decoding with a 0.5B draft** takes 7B int8 from 82 to 120 tok/s single-sequence,
  with output identical to the target alone.
- **Tokenizer exact against HuggingFace** on an adversarial corpus (CJK, emoji, combining marks,
  mixed scripts).
- Every kernel has a host reference test at edge sizes with NaN poison and canaries, a
  `--selftest` mode that must reject deliberately broken kernels, and a compute-sanitizer pass.

## Results

All numbers: RTX 5070 Ti, locked clocks, greedy unless stated, from the records in `bench/`.

### The progression on Qwen2.5-1.5B

Each row re-runs the decode path that stage introduced, from the final binaries, on one model
and one card. Tokens per second, from `bench/final_progression.json` (`tools/bench_all.sh`).

| decode path                                          | 1 sequence | 16 sequences |
| ---------------------------------------------------- | ---------- | ------------ |
| naive CUDA kernels, full recompute per token         | 2.9        |              |
| + KV cache                                           | 12.7       |              |
| + row-parallel GEMV, parallel attention, fused step  | 183        |              |
| + continuous batching over a paged KV cache          | 131        | 1176         |
| + CUDA-graph replay and flash-decoding attention     | 176        | 1308         |
| + int8 weights                                       | 241        | 823          |
| + int4 weights                                       | 316        |              |
| + speculative decoding (0.5B int8 draft, k=1)        | 224        |              |
| Qwen2.5-7B int4 / int8                               | 126 / 82   | 284 (int8)   |
| Qwen2.5-7B int8 + speculative decoding               | 121        |              |

The same engine with its matmuls routed through cuBLAS fp32 decodes at 94 tok/s single-sequence
(fp32 reads twice the bytes). On 0.5B the single-sequence figure is 361 tok/s bf16.

### Serving throughput by concurrent sequences

Continuous batching, generated tokens per second, from the per-stage serving sweeps.

| model / weights                        | 1   | 4    | 8    | 16       | 32       |
| -------------------------------------- | --- | ---- | ---- | -------- | -------- |
| 1.5B bf16                              | 178 | 554  | 892  | 1339     | **1361** |
| 1.5B bf16, over HTTP (SSE, 16 clients) |     |      |      | 1148     |          |
| 1.5B int8                              | 245 | 614  | 540  | 827      | **1083** |
| 1.5B int4                              | 311 | 676  | 756  | **899**  | 845      |
| 0.5B bf16                              | 368 | 1099 | 1743 | 2492     | **2785** |
| 7B int4                                | 126 | 188  | 252  | 243      |          |
| 7B int8                                | 82  | 179  | 213  | **271**  |          |

Throughput scales because the weights are read once per step whatever the batch width. The
batched GEMV holds the memory roofline through 8 rows and 1.3x of it at 16, by register-tiling
four output rows per warp so they share each activation load. Before that fix 16 rows cost 3.8x a
single row and the 1.5B bf16 line topped out at 683 tok/s. The int8 dip at 4 to 8 rows is a known
gap: both quantized kernels are slow at that width on the small 1.5B MLP shapes.

Long context and paged KV, 1.5B bf16:

- A 3000-token context costs 6.6 ms per decoded token with the tiled prefill GEMM, CUDA graphs
  and flash-decoding attention (146 ms before).
- A 512 MB block pool serves 16 sequences at max_seq 4096 at full throughput, peaking at 41 MB in
  use where up-front reservation would hold 3.8 GB.
- A shared 64-token prefix served from the prefix cache takes throughput from 285 to 624 tok/s.
- At a 32 MB budget the scheduler preempts 47 times and still serves 442 tok/s.

### Speculative decoding

One sequence, 0.5B draft, tok/s, acceptance rate in parentheses.

| target    | alone | k=1       | k=2 | k=3       | k=4 | best gain |
| --------- | ----- | --------- | --- | --------- | --- | --------- |
| 7B int8   | 82    | 120 (86%) | 107 | 111 (72%) | 84  | 1.46x     |
| 1.5B bf16 | 178   | 224 (84%) | 215 | 218 (70%) | 208 | 1.26x     |
| 7B int4   | 126   | 136 (82%) | 99  | 109 (63%) | 87  | 1.07x     |

k=1 wins on this card. A 0.5B draft step is launch-bound at 2.1 to 2.8 ms whatever its dtype, and
the target's k+1-row verify costs about one single step for bf16 but 1.5 to 2x for the
compute-bound int8 and int4 kernels at three or more rows. Speculation beats plain batching at 1
to 2 concurrent sequences; from 4 sequences the same weight read is spent more cheaply on real
requests (1.5B: 461 vs 554 tok/s at 4 slots).

### Quantization accuracy

Teacher-forced perplexity on 8k tokens.

| model | bf16  | int8  | int4 (round-to-nearest, groups of 128) |
| ----- | ----- | ----- | -------------------------------------- |
| 0.5B  | 16.83 | 16.87 | 22.21                                  |
| 1.5B  | 8.339 | 8.345 | 11.25                                  |
| 7B    |       | 1.83  | 2.54                                   |

int8 is free. int4 without calibration costs 35%, which is the gap GPTQ and AWQ close.

### Kernel evidence

Nsight Compute, naive kernel to optimized kernel.

| kernel                           | compute pipelines busy | memory bandwidth used     |
| -------------------------------- | ---------------------- | ------------------------- |
| GEMV 1536 x 8960 (1.5B MLP down) | 0.33% to 16.7%         | 3.1% to 85%               |
| GEMV 151936 x 1536 (LM head)     | 3.2% to 27.1%          | 30% to 93%                |
| cached attention, 1024 positions | 0.02% to 2.1%          | 11.6 ms to 45 us a launch |

## Reproducing the numbers

Profiler reports are not committed, but the summary of each one is. Every profile table in the
blog posts can be re-printed from a committed record without a GPU:

```
git clone https://github.com/RyanXiaoo/infer
cd infer
python3 tools/nsys_kernel_summary.py --from-json bench/stage5_baseline_profile_mine.json
```

The script checks that the kernel rows sum to the GPU busy time and that the busy time fits inside
the decode window, and fails if they do not. The Stage 5 records are
`bench/stage5_{baseline,phase1,phase2,phase3,final}_profile_*.json`. The decode-speed chart from
the Stage 5 write-up is drawn by `tools/plot_attention_fix.py` from
`bench/stage5_decode_ladder.json`.

Everything else needs the GPU box. `tools/bench_all.sh` re-runs the whole progression table from
the built binaries, and `tools/lock_clocks.sh lock` sets the clocks before any timing run.

## How it works

**Model loading.** `src/loader.cpp` mmaps safetensors (single file or sharded) and exposes
zero-copy bf16 tensor views. `src/model.cpp` resolves the Qwen and Llama tensor names from
`config.json`, so every dimension comes from the checkpoint and nothing in the engine names a
model. A `.llmq` container holds group-quantized weights (`src/quant.cpp`).

**Forward pass.** RMSNorm, RoPE, grouped-query attention, SwiGLU, tied or untied LM head. First
on the CPU (`src/forward.cpp`, the reference), then on the GPU (`kernels/model_gpu.cu`) with fp32
activations and KV. `tools/dump_logits.py` writes HuggingFace goldens with per-layer taps that
the golden-ladder tests compare in execution order.

**Kernels** (`kernels/ops/`, one file per family):

- `linear.cu`: row-parallel decode GEMV with interleaved coalesced loads and a warp tree reduce.
- `gemm.cu`: 64x64 tiled prefill GEMM. `attention.cu`: prefill attention.
- `cache.cu`: KV append and the single-sequence cached attention, one block per head.
- `batch.cu`: the batched engine's kernels. A batched GEMV that stages an activation chunk in
  shared memory and register-tiles four output rows per warp, flash-decoding attention that
  splits positions across blocks and combines online-softmax partials, and Gumbel-max sampling
  with a counter-based hash RNG.
- `rmsnorm.cu`, `rope.cu`, `elementwise.cu`: fused residual+norm, fused RoPE+cache-append, SwiGLU.
- `quant.cu`: int8 and int4 dequantizing GEMV, GEMM and embedding. `embedding.cu`: bf16 embedding.
- `spec.cu`: the speculative accept and residual-sampling kernel.

Each kernel is profiled with nsys and ncu (`tools/nsys_kernel_summary.py`, `tools/ncu_summary.py`)
and microbenched (`kernels/bench/`).

**KV cache** (`src/block_pool.cpp`, `kernels/batch_gpu.cu`). A pool of 16-position blocks with
per-sequence block tables, refcounts, a hash-chained prefix cache with copy-on-write, and
recompute-style preemption under a byte budget. Kernels address the cache through the table, so
a contiguous cache is the special case of one sequence.

**Scheduler and serving** (`src/scheduler.cpp`, `src/serve.cpp`, `src/main_server.cpp`).
Continuous batching (retire, admit, decode each step), per-request sampling parameters and stop
reasons, an engine thread with streaming request handles and cancellation, and an
OpenAI-compatible HTTP layer (vendored cpp-httplib) with SSE. The decode step is captured into a
CUDA graph per batch width. An event ring records every step, and `viz/index.html` replays it
(slot timeline, batch size, blocks in use).

**Speculative decoding** (`kernels/spec_engine.cu`). A draft and a target engine behind the same
`BatchEngine` interface. Each round the draft takes k steps, the target verifies k+1 rows per
slot in one forward with all slots packed, the longest matching prefix is accepted (or the
rejection-sampling kernel runs for sampled requests), and both caches are truncated to the
accepted length.

## Correctness

- `test_forward`, `test_forward_gpu`: the golden ladder. Every intermediate (post-norm, Q/K/V,
  post-RoPE, attention output, MLP, residual stream, logits) is compared in execution order
  against HuggingFace dumps, so the first failing rung names the broken op. Greedy continuations
  must match token for token.
- `test_kv_cache`: cached decode must produce the identical token stream to full recompute, for
  every combination of GEMM path, attention kernel and decode-step variant.
- `test_ops_gpu`: each optimized kernel against a double-precision host reference at edge sizes
  (lengths 1, 33, 257, 4097; widths not divisible by the thread count), with NaN poison past the
  live cache rows, canary words past every output, and a `--selftest` mode that swaps in broken
  kernels and requires the sweep to reject them. The sampling and speculative accept kernels are
  checked against the distribution they claim to sample (32768 trials, total variation under
  0.03). This test found a bias in the original sampler's hash.
- `test_batch_gpu`: batched decode must equal single-sequence decode per row, in lockstep and
  with requests joining and leaving a running batch, under a KV budget that forces preemption,
  with a shared prefix from the prefix cache, and with quantized weights.
- `test_spec_gpu`: speculative decoding must equal the target's greedy stream token for token for
  k in {1, 2, 4, 8}, batched through the scheduler, and near-greedy when sampled.
- `test_block_pool`, `test_scheduler`, `test_serve`: allocator, prefix cache, copy-on-write,
  scheduler policy and the serving lifecycle (100 concurrent clients with random cancels must
  leave zero blocks in use), runnable on a machine without a GPU against a fake engine.
- `test_tokenizer`: exact HuggingFace ids on an ordinary corpus and an adversarial one, plus the
  chat template.
- `tools/sanitize.sh`: memcheck, racecheck, initcheck and synccheck, clean before a feature is
  called done.
- Bisection: a golden-ladder failure on `--gemm=mine` that disappears on `--gemm=cublas` is a
  GEMV bug; one that persists is in RoPE, attention, norms or glue.

## Building and running

```
# Mac (CPU-only targets) or the CUDA box (all targets)
cmake -S . -B build && cmake --build build -j
ctest --test-dir build

# Models and goldens, once (offline HuggingFace tooling)
python tools/download_model.py Qwen/Qwen2.5-0.5B-Instruct
python tools/dump_logits.py && python tools/dump_logits.py --greedy
python tools/dump_tokenizer_tests.py
```

Every test and tool honours `LLM_MODEL` (default Qwen2.5-0.5B-Instruct) and `LLM_QUANT=int8|int4`.

```
# Chat on the GPU with my kernels
./build/main_chat gpu mine

# Single-sequence decode benchmark: prompt n_new device gemm cache attn step
./build/main_generate 3 256 gpu mine on par fused

# Serving sweep (key=value args); then open viz/index.html and drop bench/events_*.jsonl
./build/main_serve_bench slots=1,2,4,8,16,32 graphs=1
./build/main_serve_bench slots=16 max_seq=4096 budget_mb=64 prefix=64 graphs=1

# Quantize, then run any tool on the quantized weights
./build/quantize models/Qwen2.5-1.5B-Instruct int4
LLM_MODEL=Qwen2.5-1.5B-Instruct LLM_QUANT=int4 ./build/ppl 512 8192

# Speculative decoding, in the bench or the server
LLM_MODEL=Qwen2.5-7B-Instruct LLM_QUANT=int8 LLM_DRAFT=Qwen2.5-0.5B-Instruct LLM_DRAFT_QUANT=int8 \
  ./build/main_serve_bench slots=1 spec_k=1 graphs=1 max_seq=512
./build/main_server --model Qwen2.5-1.5B-Instruct --draft Qwen2.5-0.5B-Instruct --spec-k 1

# Serve and load it
./build/main_server --port 8080 --slots 16 --max-seq 2048 --model Qwen2.5-1.5B-Instruct
curl -N localhost:8080/v1/chat/completions -d '{"messages":[{"role":"user","content":"Hi"}],"stream":true}'
tools/load_gen.py --url http://localhost:8080 --requests 64 --concurrency 16

# The whole progression from the built binaries, into bench/final_progression.json
tools/bench_all.sh
```

Model weights, goldens and profiler reports are not committed. The `bench/*.json` records are.

## Repository layout

```
src/            CPU side: loader, model, forward pass (reference), tokenizer, sampler,
                block pool, scheduler, serving core, HTTP server, quantizer library
kernels/        CUDA side: ops/ (one file per kernel family), model_gpu (GPU forward),
                session_gpu (single-sequence fused decode), batch_gpu (batched/paged engine),
                spec_engine (speculative decoding), bench/ (microbenchmarks), common/ (timing)
tests/          golden-ladder tests, kernel tests, KV/batch/scheduler/serve/tokenizer tests;
                golden/ holds the HuggingFace dumps (not committed)
tools/          golden dumpers, quantizer and perplexity tools, load generator, profiler
                summaries and plots, sanitizer and clock scripts, the progression script
bench/          every measurement as a JSON record with commit and clock provenance
viz/            self-contained scheduler replay visualizer
third_party/    cpp-httplib, nlohmann/json (vendored, header-only)
```

## Not built

int4 calibration (GPTQ or AWQ style), tensor-core prefill attention and GEMM, tree or
multi-candidate speculation, KV-cache quantization, top-k and top-p in the batched path.

## Blog series

"LLM Inference Engine from scratch" on Medium, one post per stage.

1. [Loading the model](https://medium.com/@ryan___/llm-inference-engine-from-scratch-loading-the-model-538d27af6867)
2. [Forward pass](https://medium.com/@ryan___/llm-inference-engine-from-scratch-forward-pass-b967334212c8)
3. [Engine meets GPU](https://medium.com/@ryan___/llm-inference-engine-engine-meets-gpu-440a3d9ed75e)
4. [KV cache](https://medium.com/@ryan___/llm-inference-engine-from-scratch-kv-cache-750765be3018)
5. [Tokenization](https://medium.com/@ryan___/llm-inference-engine-from-scratch-tokenization-d708756399bd)
6. [Kernel optimization](https://medium.com/@ryan___/llm-inference-engine-from-scratch-part-6-kernel-optimization-ac0604b26790)
7. [CPU optimization and the hardware limit](https://medium.com/@ryan___/llm-inference-engine-from-scratch-part-7-cpu-optimization-and-the-hardware-limit-6f3a9760f0ae)
8. Batching and continuous batching (to be written)
9. Paged KV cache (to be written)
10. Long context, CUDA graphs and the replay visualizer (to be written)
11. Serving and cancellation (to be written)
12. Quantization and a 7B model (to be written)
13. Speculative decoding and the final numbers (to be written)
