# infer: an LLM inference engine from scratch (C++ / CUDA)

A from-scratch inference engine for Llama-style models (Qwen2.5 0.5B, 1.5B and, quantized, 7B).
No PyTorch, no cuDNN: the safetensors loader, the transformer forward pass, the byte-level BPE
tokenizer, the paged KV cache, every CUDA kernel, the continuous-batching scheduler, the HTTP
server, the quantizer and the speculative decoder are all written here. cuBLAS exists in the tree
only behind a `--gemm=cublas` flag, as a bisection tool and a yardstick; every headline number
comes from kernels in this repo.

Hardware: one RTX 5070 Ti (16 GB, sm_120) on WSL2. Correctness is checked against HuggingFace
reference logits at every layer; performance is measured under locked clocks with provenance
(commit, clocks, temperature) stamped into every record in `bench/`.

## Highlights

- **Single-sequence decode on Qwen2.5-1.5B: 7 -> 178 tok/s (bf16), 313 tok/s at int4.** The decode
  GEMV streams weights at 791 GB/s against a measured 784 GB/s achievable (STREAM-style): it sits
  at the memory-bandwidth roofline, and the remaining gap to the 254 tok/s theoretical ceiling is
  host overhead, not kernel time.
- **Batched serving: 1339 tok/s at 16 concurrent sequences on 1.5B, 2492 tok/s on 0.5B**, with a
  paged KV cache (16-position blocks, prefix sharing with copy-on-write, preemption under a byte
  budget), CUDA-graph replay of the decode step and flash-decoding attention.
- **OpenAI-compatible HTTP server** with SSE streaming, seeded temperature sampling on the device,
  and cancellation that returns a disconnected client's KV blocks within one scheduler step
  (623 tok/s over HTTP at 16 clients; half the streams dropped mid-flight leaves 0 blocks in use).
- **Quantization: int8 is lossless (perplexity 8.339 -> 8.345), and Qwen2.5-7B runs on the 16 GB
  card** at int4 (4.0 GB of weights, 126 tok/s single-sequence) and int8 (271 tok/s at 16 slots).
- **Speculative decoding** with a 0.5B draft: 7B int8 goes 82 -> 120 tok/s single-sequence, with
  output identical to the target alone (exactly when greedy, in distribution when sampled: the
  rejection-sampling kernel is checked by a distribution test).
- **Tokenizer exact against HuggingFace** on an adversarial corpus (CJK, emoji, combining marks,
  mixed scripts): Unicode classes and NFC normalization from generated tables.
- Every kernel has a host reference test at edge sizes with NaN poison and canaries, a
  `--selftest` mode that must reject deliberately broken kernels, and a four-pass
  compute-sanitizer script; every engine path must reproduce the single-sequence token stream.

## Results

All numbers: RTX 5070 Ti, locked clocks, greedy unless stated, from the records in `bench/`.

### The progression, Qwen2.5-1.5B (tokens per second, `tools/bench_all.sh`)

Each row re-runs the decode path that step introduced, from the committed binaries, on one
model and one card (`bench/final_progression.json`):

| decode path                                                      | 1 sequence | 16 sequences |
| ---------------------------------------------------------------- | ---------- | ------------ |
| naive CUDA kernels, full recompute per token                     | 2.9        |              |
| + KV cache                                                       | 12.7       |              |
| + coalesced row-parallel GEMV, one-block-per-head attention, fused decode step | 183 |     |
| + batched decode and continuous batching (paged KV cache)        | 131        | 1176         |
| + CUDA-graph replay and flash-decoding attention                 | 176        | 1308         |
| + int8 weights (group-wise, dequantized in registers)            | 241        | 823          |
| + int4 weights                                                   | 316        |              |
| + speculative decoding (0.5B int8 draft, k=1, bf16 target)       | 224        |              |
| Qwen2.5-7B int4 / int8                                           | 126 / 82   | 284 (int8)   |
| Qwen2.5-7B int8 + speculative decoding                           | 121        |              |

The same engine with its matmuls routed through cuBLAS fp32 decodes at 94 tok/s single-sequence
(the yardstick reads twice the bytes). On 0.5B the single-sequence figure is 361 tok/s bf16 and
2785 tok/s at 32 sequences.

### Serving throughput by concurrent sequences (continuous batching, generated tokens per second)

| model / weights   | 1   | 4    | 8    | 16       | 32       |
| ----------------- | --- | ---- | ---- | -------- | -------- |
| 1.5B bf16         | 178 | 554  | 892  | 1339     | **1361** |
| 1.5B int8         | 245 | 614  | 540  | 827      | **1083** |
| 1.5B int4         | 311 | 676  | 756  | **899**  | 845      |
| 0.5B bf16         | 368 | 1099 | 1743 | 2492     | **2785** |
| 7B int4           | 126 | 188  | 252  | 243      |          |
| 7B int8           | 82  | 179  | 213  | **271**  |          |

The weights are read once per step whatever the batch width, which is why throughput scales; the
batched GEMV holds the memory roofline through 8 rows and 1.3x of it at 16 (register tiling: four
output rows per warp reuse each activation float4; before that fix 16 rows cost 3.8x a single row
and the 1.5B bf16 line topped out at 683 tok/s). int8 dips at 5 to 8 rows: both quantised kernels
are slow at that width on the small 1.5B MLP shapes, a known gap.

Long context: with the tiled prefill GEMM, CUDA graphs and flash-decoding attention a 3000-token
context costs 6.6 ms per decoded token (146 ms before), and 1024-token prompts run at 72 tok/s at
16 slots. Paged KV: a 512 MB block pool serves 16 sequences at max_seq 4096 at full throughput
with a peak of 41 MB in use where reservation would hold 3.8 GB; a shared 64-token prefix served
from the prefix cache takes throughput from 285 to 624 tok/s; at a 32 MB budget the scheduler
preempts 47 times and still serves 442 tok/s.

### Speculative decoding (1 sequence, 0.5B draft, tok/s; acceptance in parentheses)

| target        | alone | k=1       | k=2 | k=3       | k=4 | best gain |
| ------------- | ----- | --------- | --- | --------- | --- | --------- |
| 7B int8       | 82    | 120 (86%) | 107 | 111 (72%) | 84  | 1.46x     |
| 1.5B bf16     | 178   | 224 (84%) | 215 | 218 (70%) | 208 | 1.26x     |
| 7B int4       | 126   | 136 (82%) | 99  | 109 (63%) | 87  | 1.07x     |

k=1 wins on this hardware: a 0.5B draft step is launch-bound at 2.1 to 2.8 ms whatever its dtype
(20% of a 7B int8 step, 45% of a 1.5B step), and the target's k+1-row verify costs 1.0x a single
step for bf16 but 1.5 to 2x for the compute-bound int8/int4 kernels at 3+ rows. Speculation beats
plain batching at 1 to 2 concurrent sequences; from 4 sequences the same weight read is spent
more cheaply on real requests (1.5B: 461 vs 554 tok/s at 4 slots).

### Quantization accuracy (teacher-forced perplexity on 8k tokens)

| model | bf16  | int8  | int4 (round-to-nearest, groups of 128) |
| ----- | ----- | ----- | -------------------------------------- |
| 0.5B  | 16.83 | 16.87 | 22.21                                  |
| 1.5B  | 8.339 | 8.345 | 11.25                                  |
| 7B    |       | 1.83  | 2.54                                   |

int8 is free; int4 without calibration costs 35%, which is the gap GPTQ/AWQ-style methods close.

### Kernel evidence (Nsight Compute, naive kernel -> optimized kernel)

| kernel                            | compute pipelines busy | memory bandwidth used   |
| --------------------------------- | ---------------------- | ----------------------- |
| GEMV 1536 x 8960 (1.5B MLP down)  | 0.33% -> 16.7%         | 3.1% -> 85%             |
| GEMV 151936 x 1536 (LM head)      | 3.2% -> 27.1%          | 30% -> 93%              |
| cached attention, 1024 positions  | 0.02% -> 2.1%          | 11.6 ms -> 45 us/launch |

## How it works

**Model loading.** `src/loader.cpp` mmaps safetensors (single file or sharded via the index
json) and exposes zero-copy bf16 tensor views; `src/model.cpp` resolves the Qwen/Llama tensor
names from `config.json` so every dimension comes from the checkpoint and nothing in the engine
names a model. A `.llmq` container holds group-quantized weights (`src/quant.cpp`).

**Forward pass.** RMSNorm, RoPE, grouped-query attention, SwiGLU, tied or untied LM head, first
on the CPU (`src/forward.cpp`, the reference) and then on the GPU (`kernels/model_gpu.cu`) with
fp32 activations and KV. `tools/dump_logits.py` writes HuggingFace goldens with per-layer taps
that the golden-ladder tests compare in execution order.

**Kernels** (`kernels/ops/`): row-parallel decode GEMV with interleaved coalesced loads and a
warp tree reduce; a batched GEMV that stages an activation chunk in shared memory and register-
tiles four output rows per warp; a 64x64 tiled prefill GEMM; one-block-per-head cached attention
and a flash-decoding variant that splits positions across blocks and combines online-softmax
partials; fused residual+norm and RoPE+cache-append; int8/int4 dequantizing GEMV, GEMM and
embedding; Gumbel-max sampling with a counter-based hash RNG; the speculative accept/residual
kernel. Each is profiled with nsys/ncu (`tools/nsys_kernel_summary.py`, `tools/ncu_summary.py`)
and microbenched (`kernels/bench/`).

**KV cache** (`src/block_pool.cpp`, `kernels/batch_gpu.cu`): a pool of 16-position blocks with
per-sequence block tables, refcounts, a hash-chained prefix cache with copy-on-write, and
recompute-style preemption under a byte budget. Kernels address the cache through the table, so a
contiguous cache is just the special case of one sequence.

**Scheduler and serving** (`src/scheduler.cpp`, `src/serve.cpp`, `src/main_server.cpp`):
continuous batching (retire, admit, decode each step), per-request sampling parameters and stop
reasons, an engine thread with streaming request handles and cancellation, an event ring that
`viz/index.html` replays (slot timeline, batch size, blocks in use), and an OpenAI-compatible
HTTP layer (vendored cpp-httplib) with SSE. The decode step is captured into a CUDA graph per
batch width.

**Speculative decoding** (`kernels/spec_engine.cu`): a draft and a target engine behind the
same `BatchEngine` interface. Per round the draft takes k steps, the target verifies k+1 rows
per slot in one forward (all slots packed), the longest matching prefix is accepted (or the
rejection-sampling kernel runs for sampled requests), and both caches are truncated to the
accepted length. The scheduler consumes multi-token steps.

## Repository layout

```
src/            CPU side: loader, model, forward pass (reference), tokenizer, sampler,
                block pool, scheduler, serving core, HTTP server, quantizer library
kernels/        CUDA side: ops/ (one file per kernel family), model_gpu (GPU forward),
                session_gpu (single-sequence fused decode), batch_gpu (batched/paged engine),
                spec_engine (speculative decoding), bench/ (microbenchmarks), common/ (timing)
tests/          golden-ladder tests, kernel tests, KV/batch/scheduler/serve/tokenizer tests;
                golden/ holds HF dumps (not committed)
tools/          HF golden dumpers, quantizer and perplexity tools, load generator, profiler
                summaries, sanitizer and clock scripts, the final progression script
bench/          every measurement as a JSON record with commit and clock provenance
viz/            self-contained scheduler replay visualizer
third_party/    cpp-httplib, nlohmann/json (vendored, header-only)
```

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
  deliberately broken kernels and requires the sweep to reject them. The sampling and speculative
  accept kernels are checked against the distribution they claim to sample (32768 trials, total
  variation < 0.03); this test found a bias in the original sampler's hash.
- `tests/test_batch_gpu`: batched decode must equal single-sequence decode per row, both in
  lockstep and with requests joining and leaving a running batch; also under a KV budget that
  forces preemption, with a shared prefix served from the prefix cache, and with quantized weights.
- `tests/test_spec_gpu`: speculative decoding must equal the target's greedy stream token for
  token for k in {1, 2, 4, 8}, batched through the scheduler, and near-greedy when sampled.
- `tests/test_block_pool`, `tests/test_scheduler`, `tests/test_serve`: allocator, prefix cache,
  copy-on-write, scheduler policy (admission, retirement, preemption, cancellation) and the
  serving lifecycle (100 concurrent clients with random cancels must leave zero blocks in use),
  on the Mac against a fake engine.
- `tests/test_tokenizer`: exact HF ids on an ordinary corpus and on an adversarial one (CJK,
  emoji, combining marks, mixed scripts, whitespace, code), plus the chat template.
- `tools/sanitize.sh`: memcheck, racecheck, initcheck, synccheck, clean before a feature is called done.
- Bisection: a golden-ladder failure on `--gemm=mine` that disappears on `--gemm=cublas` is a GEMV
  bug; one that persists is in RoPE, attention, norms or glue.

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

# Single-sequence decode benchmark:  prompt n_new device gemm cache attn step
./build/main_generate 3 256 gpu mine on par fused

# Any model: every test and tool honours LLM_MODEL (and LLM_QUANT=int8|int4).
LLM_MODEL=Qwen2.5-1.5B-Instruct ./build/test_forward_gpu

# Serving sweep (key=value args)
./build/main_serve_bench slots=1,2,4,8,16,32 graphs=1
./build/main_serve_bench slots=16 max_seq=4096 budget_mb=64 prefix=64 graphs=1
# then open viz/index.html and drop bench/events_*.jsonl

# Quantize and run any tool on the quantized weights:
./build/quantize models/Qwen2.5-1.5B-Instruct int4
LLM_MODEL=Qwen2.5-1.5B-Instruct LLM_QUANT=int4 ./build/ppl 512 8192

# Speculative decoding: a draft model, in the bench or the server
LLM_MODEL=Qwen2.5-7B-Instruct LLM_QUANT=int8 LLM_DRAFT=Qwen2.5-0.5B-Instruct LLM_DRAFT_QUANT=int8 \
  ./build/main_serve_bench slots=1 spec_k=1 graphs=1 max_seq=512
./build/main_server --model Qwen2.5-1.5B-Instruct --draft Qwen2.5-0.5B-Instruct --spec-k 1

# Serve and load it:
./build/main_server --port 8080 --slots 16 --max-seq 2048 --model Qwen2.5-1.5B-Instruct
curl -N localhost:8080/v1/chat/completions -d '{"messages":[{"role":"user","content":"Hi"}],"stream":true}'
tools/load_gen.py --url http://localhost:8080 --requests 64 --concurrency 16

# The whole progression from the committed binaries -> bench/final_progression.json
tools/bench_all.sh
```

Model weights, goldens and profiler reports are not committed; `bench/*.json` records are.

## Not built

int4 calibration (GPTQ/AWQ-style); tensor-core (bf16 mma) prefill attention and GEMM;
tree/multi-candidate speculation; KV-cache quantization; top-k/top-p in the batched path.

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
10. Long context, CUDA graphs and the replay visualizer (to be written)
11. Serving and cancellation (to be written)
12. Quantization and a 7B model (to be written)
13. Speculative decoding and the final numbers (to be written)
