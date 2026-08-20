#!/usr/bin/env bash
# lock_clocks.sh — lock/unlock/query GPU clocks for reproducible benchmarks.
#
# Run FROM THE MAC (or anywhere with `ssh pc` configured). Under WSL2,
# nvidia-smi is mostly a read-only passthrough, so clock control has to happen
# on the Windows side — this script drives it over `ssh pc`.
#
# Lock target: ~82% of max boost. Locking at the top of the boost range just
# thermal-throttles away from the locked value; below-boost is sustainable
# indefinitely. Slower absolute numbers, stable comparisons.
#
# Memory clocks: --lock-memory-clocks is attempted but frequently unsupported
# on GeForce. If it fails we say so and move on — the bench harness records the
# memory clock per-run rather than assuming it's pinned.
#
# Usage: lock_clocks.sh lock | unlock | status

set -euo pipefail

SMI='& "C:\Program Files\NVIDIA Corporation\NVSMI\nvidia-smi.exe"'
# Newer drivers install nvidia-smi on PATH for PowerShell; try plain name first.
run_smi() {
    ssh pc "nvidia-smi $*" 2>/dev/null || ssh pc "$SMI $*"
}

case "${1:-status}" in
  lock)
    MAX=$(run_smi --query-gpu=clocks.max.sm --format=csv,noheader,nounits | tr -d '[:space:]')
    if [[ -z "$MAX" ]]; then echo "could not query max SM clock" >&2; exit 1; fi
    TARGET=$(( MAX * 82 / 100 ))
    echo "max boost ${MAX} MHz -> locking SM at ${TARGET} MHz (82%)"
    run_smi "--lock-gpu-clocks=${TARGET},${TARGET}"
    MAXMEM=$(run_smi --query-gpu=clocks.max.memory --format=csv,noheader,nounits | tr -d '[:space:]')
    if run_smi "--lock-memory-clocks=${MAXMEM},${MAXMEM}"; then
      echo "memory clock locked at ${MAXMEM} MHz"
    else
      echo "NOTE: memory clock lock unsupported on this GPU (expected on GeForce)."
      echo "      DRAM throughput is NOT pinned — harness records mem clock per run."
    fi
    "$0" status
    ;;
  unlock)
    run_smi --reset-gpu-clocks || true
    run_smi --reset-memory-clocks || true
    echo "clocks reset to default behavior"
    ;;
  status)
    echo "current clocks / temp:"
    run_smi --query-gpu=clocks.sm,clocks.mem,temperature.gpu,name --format=csv
    ;;
  *)
    echo "usage: $0 lock|unlock|status" >&2
    exit 1
    ;;
esac
