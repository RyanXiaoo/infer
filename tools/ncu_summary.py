#!/usr/bin/env python3
"""ncu_summary.py — pull the counters that matter out of a folder of ncu reports.

Reads every *.ncu-rep in the folder through `ncu -i <rep> --page details`, keeps
a fixed set of metrics, prints one row per report and optionally writes JSON.

The metrics, and what each one says about a bandwidth- or latency-bound kernel:
  Duration                  one launch, microseconds (under the profiler)
  SM Busy                   % of compute pipelines doing work
  Memory Throughput         GB/s actually moved through the memory system
  Max Bandwidth             that, as % of what the memory system can carry
  DRAM Throughput           % of DRAM bandwidth used
  L1/TEX Hit Rate, L2 Hit Rate   where the loads were served from
  notes                     ncu's own OPT findings, first line each
"""

import argparse
import json
import re
import subprocess
from pathlib import Path

METRICS = {
    "Duration": None,                   # unit varies: normalised to microseconds
    "SM Busy": "sm_busy_pct",
    "Memory Throughput": None,          # appears twice (% and Gbyte/s): handled by unit
    "Max Bandwidth": "max_bandwidth_pct",
    "DRAM Throughput": "dram_throughput_pct",
    "L1/TEX Hit Rate": "l1_hit_pct",
    "L2 Hit Rate": "l2_hit_pct",
    "Compute (SM) Throughput": "compute_throughput_pct",
}
ROW = re.compile(r"^\s{4}(\S.*?)\s{2,}(\S+)?\s+(-?[\d.]+(?:e[+-]?\d+)?)\s*$")


# ncu auto-scales units per row (us/ms/s, Kbyte/s..Gbyte/s); normalise them.
TIME_TO_US = {"ns": 1e-3, "us": 1.0, "ms": 1e3, "s": 1e6}
RATE_TO_GBPS = {"byte/s": 1e-9, "kbyte/s": 1e-6, "mbyte/s": 1e-3, "gbyte/s": 1.0}


def parse(text):
    out, notes = {}, []
    lines = text.splitlines()
    for i, line in enumerate(lines):
        m = ROW.match(line)
        if m:
            name, unit, value = m.group(1).strip(), (m.group(2) or "").lower(), float(m.group(3))
            if name == "Memory Throughput":
                if unit in RATE_TO_GBPS:
                    out.setdefault("memory_throughput_gbps", value * RATE_TO_GBPS[unit])
                else:
                    out.setdefault("memory_busy_pct", value)
            elif name == "Duration":
                out.setdefault("duration_us", value * TIME_TO_US.get(unit, 1.0))
            elif name in METRICS and METRICS[name]:
                out.setdefault(METRICS[name], value)
        elif line.strip().startswith("OPT"):
            first = line.strip()[3:].strip()
            nxt = lines[i + 1].strip() if i + 1 < len(lines) else ""
            notes.append((first + " " + nxt).strip()[:200])
    out["notes"] = notes
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("folder", type=Path)
    ap.add_argument("--json", type=Path)
    args = ap.parse_args()

    rows = {}
    for rep in sorted(args.folder.glob("*.ncu-rep")):
        text = subprocess.run(["ncu", "-i", str(rep), "--page", "details"],
                              capture_output=True, text=True).stdout
        rows[rep.stem] = parse(text)

    cols = [("duration_us", "dur us"), ("sm_busy_pct", "SM busy%"),
            ("memory_throughput_gbps", "mem GB/s"), ("max_bandwidth_pct", "maxBW%"),
            ("dram_throughput_pct", "DRAM%"), ("l1_hit_pct", "L1 hit%"), ("l2_hit_pct", "L2 hit%")]
    print(f"{'report':34s}" + "".join(f"{h:>10s}" for _, h in cols))
    for name, r in rows.items():
        print(f"{name:34s}" + "".join(
            f"{r[k]:10.3g}" if k in r else f"{'-':>10s}" for k, _ in cols))

    if args.json:
        args.json.write_text(json.dumps({
            "what": "ncu counters, one profiled launch per report (--set full), Stage 5 before/after",
            "reports": rows}, indent=1) + "\n")
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
