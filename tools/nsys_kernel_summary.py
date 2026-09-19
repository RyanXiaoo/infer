#!/usr/bin/env python3
"""nsys_kernel_summary.py — where does decode time go, from one nsys capture.

Input is the .sqlite export of an nsys report:
    nsys profile -o bench/decode_mine ./build/main_generate 3 64 gpu mine on
    nsys export --type sqlite bench/decode_mine.nsys-rep
    tools/nsys_kernel_summary.py bench/decode_mine.sqlite --tokens 64

Three tables, all restricted to the DECODE WINDOW, so prefill, warm-up, weight
upload and cuBLAS init never leak into per-token numbers. Every decode step
ends with one device-to-host copy of the logits, so the window is the last
--tokens DtoH copies: it opens when the copy before them finishes (the prefill
logits) and closes when the last one finishes.
  1. GPU time per kernel name: total, % of GPU time, launches, avg, ms/token.
  2. Host CUDA API time per call name (launch cost, blocking copies, malloc/free).
  3. The window itself: span, GPU busy time, idle time, launches per token.

The check that matters: the kernel rows must sum to "GPU busy", and GPU busy
must be <= the window span. If a number quoted somewhere does not fit inside
those two totals, it was misread.

--json writes the same numbers as a bench/ record (the .sqlite/.nsys-rep
files are gitignored; the JSON is what gets committed).
"""

import argparse
import json
import re
import sqlite3
import subprocess
import sys
from pathlib import Path

KERNELS = "CUPTI_ACTIVITY_KIND_KERNEL"
RUNTIME = "CUPTI_ACTIVITY_KIND_RUNTIME"
MEMCPY = "CUPTI_ACTIVITY_KIND_MEMCPY"
MINE_NAMESPACE = "llm::gpu::"   # my kernels demangle into this namespace


def has_table(db, name):
    q = "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?"
    return db.execute(q, (name,)).fetchone() is not None


def decode_window(db, tokens, marker):
    """(t0, t1) in ns spanning exactly `tokens` decode steps (see module doc).

    Falls back to "first launch of `marker` -> last kernel" when the capture has
    no usable memcpy trace; that window starts a few kernels into step one.
    """
    if has_table(db, MEMCPY):
        ends = [r[0] for r in db.execute(
            f"SELECT end FROM {MEMCPY} WHERE copyKind = 2 ORDER BY end")]
        if len(ends) > tokens:
            return ends[-tokens - 1], ends[-1]
    t0 = db.execute(
        f"SELECT MIN(k.start) FROM {KERNELS} k JOIN StringIds s ON s.id = k.shortName "
        "WHERE s.value = ?", (marker,)).fetchone()[0]
    if t0 is None:
        sys.exit(f"marker kernel '{marker}' not found; pass --marker or --whole-run")
    t1 = db.execute(f"SELECT MAX(end) FROM {KERNELS}").fetchone()[0]
    return t0, t1


def kernel_rows(db, t0, t1):
    # Library kernels often share a useless shortName ("kernel"), so group on the
    # demangled name and keep the short one for display.
    q = (f"SELECT s.value, d.value, COUNT(*), SUM(k.end - k.start) "
         f"FROM {KERNELS} k JOIN StringIds s ON s.id = k.shortName "
         f"JOIN StringIds d ON d.id = k.demangledName "
         f"WHERE k.start >= ? AND k.end <= ? GROUP BY d.value ORDER BY 4 DESC")
    rows = []
    for short, demangled, n, total_ns in db.execute(q, (t0, t1)):
        mine = MINE_NAMESPACE in demangled
        name = short if mine else display_name(short, demangled)
        rows.append({"name": name, "mine": mine, "launches": n, "total_ns": total_ns})
    return merge_same_name(rows)


def display_name(short, demangled):
    if short != "kernel":
        return short
    # "std::enable_if<..>::type internal::gemvx::kernel<int, ...>(...)" -> "gemvx::kernel"
    scoped = re.findall(r"(\w+::kernel)\s*<", demangled)
    return scoped[-1] if scoped else short


def merge_same_name(rows):
    merged = {}
    for r in rows:
        m = merged.setdefault(r["name"], {**r, "launches": 0, "total_ns": 0})
        m["launches"] += r["launches"]
        m["total_ns"] += r["total_ns"]
    return sorted(merged.values(), key=lambda r: -r["total_ns"])


def api_rows(db, t0, t1):
    if not has_table(db, RUNTIME):
        return []
    q = (f"SELECT s.value, COUNT(*), SUM(r.end - r.start) FROM {RUNTIME} r "
         f"JOIN StringIds s ON s.id = r.nameId WHERE r.start >= ? AND r.start <= ? "
         f"GROUP BY s.value ORDER BY 3 DESC")
    return [{"name": n.split("_v")[0], "calls": c, "total_ns": t}
            for n, c, t in db.execute(q, (t0, t1))]


def memcpy_rows(db, t0, t1):
    if not has_table(db, MEMCPY):
        return []
    kinds = {1: "HtoD", 2: "DtoH", 8: "DtoD"}
    q = (f"SELECT copyKind, COUNT(*), SUM(end - start), SUM(bytes) FROM {MEMCPY} "
         f"WHERE start >= ? AND start <= ? GROUP BY copyKind")
    return [{"kind": kinds.get(k, str(k)), "copies": c, "total_ns": t, "bytes": b}
            for k, c, t, b in db.execute(q, (t0, t1))]


def git_commit():
    try:
        rev = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                             text=True, check=True).stdout.strip()
        dirty = bool(subprocess.run(["git", "status", "--porcelain"], capture_output=True,
                                    text=True, check=True).stdout.strip())
        return rev, dirty
    except (OSError, subprocess.CalledProcessError):
        return None, None


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("sqlite", type=Path, help="nsys sqlite export")
    ap.add_argument("--tokens", type=int, required=True,
                    help="decode steps in the capture (n_new), for per-token columns")
    ap.add_argument("--marker", default="attention_cached_kernel",
                    help="fallback window start when there is no memcpy trace: "
                         "a kernel that only runs during decode")
    ap.add_argument("--whole-run", action="store_true",
                    help="no window: first kernel to last kernel")
    ap.add_argument("--top", type=int, default=12, help="rows per table")
    ap.add_argument("--json", type=Path, help="also write a bench/ record here")
    ap.add_argument("--label", default="", help="free text stored in the record "
                    "(model, prompt, gemm path, flags)")
    args = ap.parse_args()

    db = sqlite3.connect(f"file:{args.sqlite}?mode=ro", uri=True)
    if args.whole_run:
        t0, t1 = db.execute(f"SELECT MIN(start), MAX(end) FROM {KERNELS}").fetchone()
    else:
        t0, t1 = decode_window(db, args.tokens, args.marker)

    kernels = kernel_rows(db, t0, t1)
    apis = api_rows(db, t0, t1)
    copies = memcpy_rows(db, t0, t1)

    n = args.tokens
    span_ns = t1 - t0
    busy_ns = sum(k["total_ns"] for k in kernels)
    launches = sum(k["launches"] for k in kernels)
    ms = lambda ns: ns / 1e6

    print(f"{args.sqlite.name}   decode window {span_ns / 1e9:.3f} s, {n} tokens"
          f"{'  [' + args.label + ']' if args.label else ''}\n")

    print(f"{'GPU kernel':34s} {'total s':>8s} {'%GPU':>6s} {'launches':>9s} "
          f"{'avg us':>8s} {'ms/token':>9s}")
    for k in kernels[:args.top]:
        tag = "" if k["mine"] else " (lib)"
        print(f"{(k['name'] + tag)[:34]:34s} {k['total_ns'] / 1e9:8.4f} "
              f"{100 * k['total_ns'] / busy_ns:6.1f} {k['launches']:9d} "
              f"{k['total_ns'] / k['launches'] / 1e3:8.1f} {ms(k['total_ns']) / n:9.3f}")
    rest = kernels[args.top:]
    if rest:
        rest_ns = sum(k["total_ns"] for k in rest)
        print(f"{'(' + str(len(rest)) + ' more)':34s} {rest_ns / 1e9:8.4f} "
              f"{100 * rest_ns / busy_ns:6.1f}")
    mine_ns = sum(k["total_ns"] for k in kernels if k["mine"])
    print(f"{'  my kernels':34s} {mine_ns / 1e9:8.4f} {100 * mine_ns / busy_ns:6.1f}")
    print(f"{'  library kernels':34s} {(busy_ns - mine_ns) / 1e9:8.4f} "
          f"{100 * (busy_ns - mine_ns) / busy_ns:6.1f}")

    if apis:
        print(f"\n{'host CUDA API':34s} {'total s':>8s} {'calls':>9s} {'avg us':>8s} "
              f"{'ms/token':>9s}")
        for a in apis[:6]:
            print(f"{a['name'][:34]:34s} {a['total_ns'] / 1e9:8.4f} {a['calls']:9d} "
                  f"{a['total_ns'] / a['calls'] / 1e3:8.1f} {ms(a['total_ns']) / n:9.3f}")
        print("  (blocking calls such as cudaMemcpy include time spent waiting for the GPU)")

    for c in copies:
        print(f"memcpy {c['kind']}: {c['copies']} copies, {c['bytes'] / 1e6:.1f} MB, "
              f"{c['total_ns'] / 1e9:.4f} s")

    print(f"\nper token:  window {ms(span_ns) / n:7.3f} ms   GPU busy {ms(busy_ns) / n:7.3f} ms"
          f"   GPU idle {ms(span_ns - busy_ns) / n:7.3f} ms   launches {launches / n:6.1f}")
    print(f"ceilings:   {1e9 * n / span_ns:6.1f} tok/s from the window, "
          f"{1e9 * n / busy_ns:6.1f} tok/s if the GPU never waited")

    if args.json:
        commit, dirty = git_commit()
        record = {
            "what": "nsys decode-window summary (tools/nsys_kernel_summary.py)",
            "source": args.sqlite.name,
            "label": args.label,
            "tokens": n,
            "window": "whole run" if args.whole_run else f"last {n} logits copies",
            "summarized_at_commit": commit,
            "summarized_at_dirty": dirty,
            "per_token_ms": {"window": ms(span_ns) / n, "gpu_busy": ms(busy_ns) / n,
                             "gpu_idle": ms(span_ns - busy_ns) / n},
            "launches_per_token": launches / n,
            "kernels": [{"name": k["name"], "mine": k["mine"], "launches": k["launches"],
                         "total_s": k["total_ns"] / 1e9,
                         "pct_gpu": 100 * k["total_ns"] / busy_ns,
                         "avg_us": k["total_ns"] / k["launches"] / 1e3,
                         "ms_per_token": ms(k["total_ns"]) / n} for k in kernels],
            "host_api": [{"name": a["name"], "calls": a["calls"],
                          "total_s": a["total_ns"] / 1e9,
                          "ms_per_token": ms(a["total_ns"]) / n} for a in apis[:8]],
            "memcpy": [{"kind": c["kind"], "copies": c["copies"], "mb": c["bytes"] / 1e6,
                        "total_s": c["total_ns"] / 1e9} for c in copies],
        }
        args.json.write_text(json.dumps(record, indent=2) + "\n")
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
