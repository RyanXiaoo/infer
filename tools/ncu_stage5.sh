#!/usr/bin/env bash
# ncu_stage5.sh — ncu counter evidence for the Stage 5 kernels, before vs after.
#
# Run ON the GPU box from the repo root (needs build/profile_one). For each case
# ncu profiles ONE launch (-s 2 skips two warm launches, -c 1 profiles the next)
# with the full section set, then tools/ncu_summary.py turns the reports into a
# table + bench/stage5_ncu_counters.json. Reports (*.ncu-rep) stay in bench/raw/
# (gitignored); the JSON is what gets committed.
#
# Usage: tools/ncu_stage5.sh

set -euo pipefail
OUT=bench/raw/ncu
mkdir -p "$OUT"

run() {   # run <report-name> <kernel-name> <profile_one args...>
    local name=$1 kernel=$2; shift 2
    echo "== $name"
    ncu -k "$kernel" -s 2 -c 1 --set full -f -o "$OUT/$name" ./build/profile_one "$@" \
        2>&1 | grep -E "Profiling|Error|ERR" || true
}

# GEMV: mlp down of both models (the serial-depth worst case) and the 1.5B LM head.
for shape in "896 4864" "1536 8960" "151936 1536"; do
    set -- $shape
    run "gemv_naive_${1}x${2}"  linear_kernel       gemv naive  "$1" "$2"
    run "gemv_rowpar_${1}x${2}" gemv_rowpar_kernel  gemv rowpar "$1" "$2"
done

# Cached attention: 0.5B layout at a short and a long context, 1.5B layout long.
for cfg in "14 2 64 64" "14 2 64 1024" "12 2 128 1024"; do
    set -- $cfg
    run "attn_naive_h${1}_hd${3}_len${4}" attention_cached_kernel     attn naive "$@"
    run "attn_par_h${1}_hd${3}_len${4}"   attention_cached_par_kernel attn par   "$@"
done

python3 tools/ncu_summary.py "$OUT" --json bench/stage5_ncu_counters.json
