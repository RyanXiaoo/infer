#!/usr/bin/env bash
# sanitize.sh — run compute-sanitizer over a binary.
#
# The four tools are mutually exclusive MODES, not flags on one run, so this is
# four sequential invocations:
#   memcheck  — OOB/misaligned global accesses, leaks
#   racecheck — shared-memory races (missing __syncthreads(); the tiled-matmul
#               and fused-attention bug class). 10-100x runtime.
#   initcheck — reads of uninitialized global memory (the canonical KV-cache bug)
#   synccheck — invalid/divergent synchronization
#
# --quick skips racecheck for fast iteration. The full four-pass run is the
# pre-stage-done gate; a stage isn't finished until it's clean.
#
# Usage: sanitize.sh [--quick] <binary> [args...]

set -uo pipefail

QUICK=0
if [[ "${1:-}" == "--quick" ]]; then QUICK=1; shift; fi
if [[ $# -lt 1 ]]; then echo "usage: $0 [--quick] <binary> [args...]" >&2; exit 1; fi

TOOLS=(memcheck initcheck synccheck)
if [[ $QUICK -eq 0 ]]; then TOOLS=(memcheck racecheck initcheck synccheck); fi

# Project convention: binaries using the bench harness honor BENCH_SMOKE=1
# (one warmup + one rep per kernel). Timings are meaningless under a sanitizer,
# and racecheck's 10-100x overhead on a full 60-rep suite can run for minutes
# or get OOM-killed inside the WSL VM.
export BENCH_SMOKE=1

FAILED=0
for tool in "${TOOLS[@]}"; do
    echo "=== compute-sanitizer --tool $tool $* ==="
    if ! compute-sanitizer --tool "$tool" --error-exitcode 1 "$@"; then
        echo "!!! $tool FAILED"
        FAILED=1
    fi
    echo
done

if [[ $FAILED -eq 1 ]]; then
    echo "sanitize: FAILURES (see above)"
    exit 1
fi
echo "sanitize: all passes clean (${TOOLS[*]})"
