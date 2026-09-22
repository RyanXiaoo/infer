#!/usr/bin/env bash
# bench_all.sh — the final progression: every stage's decode path, from the
# committed binaries, on one model (Stage 11 capstone).
#
# Run ON the GPU box from the repo root after `cmake --build build`, with
# clocks locked (tools/lock_clocks.sh lock from the Mac). Each row re-runs the
# path that stage introduced, so the table shows what each stage bought:
#
#   Stage 3  naive GPU kernels, full recompute each token   main_generate cache=off gemm=naive
#   Stage 4  KV cache                                       main_generate cache=on  gemm=naive attn=naive step=unfused
#   Stage 5  row-parallel GEMV, parallel attention, fusion  main_generate cache=on  gemm=mine  attn=par   step=fused
#   Stage 6  continuous batching (paged since Stage 7)      main_serve_bench slots=1,16 graphs=0
#   Stage 8  CUDA graphs + flash-decoding attention         main_serve_bench slots=1,16 graphs=1
#   Stage 10 int8 / int4 weights                            LLM_QUANT=int8|int4
#   Stage 11 speculative decoding (0.5B int4 draft)         LLM_DRAFT=... spec_k=K
#   plus the 7B int4/int8 rows that only exist because of Stage 10.
#
# Writes bench/final_progression.json and prints the README table.
#
# Usage: tools/bench_all.sh [model]      (default Qwen2.5-1.5B-Instruct)

set -euo pipefail
MODEL=${1:-Qwen2.5-1.5B-Instruct}
DRAFT=Qwen2.5-0.5B-Instruct
N_NEW=64
OUT=bench/final_progression.json
TMP=bench/raw/final; mkdir -p "$TMP"
ROWS=()

gen() {   # gen <label> <main_generate args...>  -> single-sequence tok/s
    local label=$1; shift
    local toks
    toks=$(LLM_MODEL=$MODEL ./build/main_generate prompt=0 n_new=$N_NEW device=gpu "$@" 2>/dev/null \
           | sed -n 's/.*= \([0-9.]*\) tok\/s/\1/p')
    printf '%-44s %8s tok/s (1 seq)\n' "$label" "$toks"
    ROWS+=("{\"stage\": \"$label\", \"slots\": 1, \"tok_s\": $toks, \"how\": \"main_generate $*\"}")
}
serve() {   # serve <label> <slots> <env...> -- <bench args...>  -> tok/s from the JSON row
    local label=$1 slots=$2; shift 2
    local envs=()
    while [ "$1" != "--" ]; do envs+=("$1"); shift; done; shift
    local tag; tag=final_$(echo "$label" | tr -c 'a-zA-Z0-9\n' '_')
    rm -f bench/serve_*_"$tag"_*.json
    env LLM_MODEL=$MODEL "${envs[@]}" ./build/main_serve_bench n=$((slots * 4 > 12 ? slots * 4 : 12)) \
        slots=$slots max_new=$N_NEW modes=continuous tag="$tag" "$@" >/dev/null 2>&1
    local f; f=$(ls bench/serve_*_"$tag"_*.json | tail -1)
    local toks acc
    toks=$(python3 -c "import json,sys; r=json.load(open(sys.argv[1]))['rows'][0]; print(round(r['tok_s'],1))" "$f")
    acc=$(python3 -c "import json,sys; r=json.load(open(sys.argv[1]))['rows'][0]; print(round(100*r.get('acceptance',0)))" "$f")
    printf '%-44s %8s tok/s (%d slots)%s\n' "$label" "$toks" "$slots" "$([ "$acc" != 0 ] && echo "  accept $acc%")"
    ROWS+=("{\"stage\": \"$label\", \"slots\": $slots, \"tok_s\": $toks, \"acceptance\": $acc, \"how\": \"${envs[*]} main_serve_bench slots=$slots $*\"}")
    mv "$f" "$TMP/"
}

echo "model $MODEL, $N_NEW new tokens per sequence"
gen   "Stage 3  naive kernels, no cache"            cache=off gemm=naive
gen   "Stage 4  KV cache"                           cache=on gemm=naive attn=naive step=unfused
gen   "Stage 5  kernels + fusion"                   cache=on gemm=mine attn=par step=fused
serve "Stage 6  batching, 1 slot"        1  -- graphs=0
serve "Stage 6  batching, 16 slots"      16 -- graphs=0
serve "Stage 8  graphs + split attn, 1"  1  -- graphs=1
serve "Stage 8  graphs + split attn, 16" 16 -- graphs=1
serve "Stage 10 int8, 1"                 1  LLM_QUANT=int8 -- graphs=1
serve "Stage 10 int8, 16"                16 LLM_QUANT=int8 -- graphs=1
serve "Stage 10 int4, 1"                 1  LLM_QUANT=int4 -- graphs=1
serve "Stage 11 speculative int4 target, k=2, 1" 1 LLM_QUANT=int4 LLM_DRAFT=$DRAFT LLM_DRAFT_QUANT=int4 -- graphs=1 spec_k=2
if [ -f models/Qwen2.5-7B-Instruct/model.q4.llmq ]; then
    MODEL=Qwen2.5-7B-Instruct
    serve "7B int4, 1"                    1  LLM_QUANT=int4 -- graphs=1 max_seq=512
    serve "7B int4 speculative k=4, 1"    1  LLM_QUANT=int4 LLM_DRAFT=$DRAFT LLM_DRAFT_QUANT=int4 -- graphs=1 max_seq=512 spec_k=4
    serve "7B int8, 16"                   16 LLM_QUANT=int8 -- graphs=1 max_seq=512
fi

{
    echo "{\"what\": \"Final progression: each stage's decode path from the committed binaries, $N_NEW new tokens, locked clocks\","
    echo " \"date\": \"$(date +%F)\", \"rows\": ["
    for i in "${!ROWS[@]}"; do echo "  ${ROWS[$i]}$([ $i -lt $((${#ROWS[@]} - 1)) ] && echo ,)"; done
    echo " ]}"
} > "$OUT"
echo "wrote $OUT"
