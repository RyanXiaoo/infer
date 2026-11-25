#!/usr/bin/env python3
"""plot_attention_fix.py — decode speed vs reply length, before and after the
parallel attention kernel (Stage 5 phase 1), on Qwen2.5-0.5B.

The naive cached-attention kernel's cost grows with the number of cached tokens,
so decode slowed down as the reply got longer on both gemm paths. The parallel
kernel removed that: the "after" lines are flat. GEMV was still the naive kernel
in every row here, which is why the "my kernels" lines barely move up.

Sources: bench/stage5_decode_ladder.json (medians, locked clocks) for three of
the four series. The cuBLAS "after" pair is from single runs taken at the phase 1
commit, recorded in the Stage 5 notes; once attention is fast that path is
CPU-bound under WSL2 and run-to-run noise is large, so treat it as "no longer
falls", not as a precise number.

Usage: tools/plot_attention_fix.py [out.png]
"""

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
LADDER = ROOT / "bench" / "stage5_decode_ladder.json"

# categorical slots: one hue per gemm path; before/after is line style, not shade
BLUE, ORANGE = "#2a78d6", "#eb6834"
SURFACE, INK, INK_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e1"


def ladder(key_32, key_256):
    tok = json.loads(LADDER.read_text())["tok_s"]
    return tok[key_32]["median"], tok[key_256]["median"]


def main():
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "bench" / "stage5_attention_fix_0.5B.png"
    series = [
        # label, (tok/s at 32, tok/s at 256), color, linestyle
        ("my kernels, before", ladder("0.5B gemm=naive attn=naive step=unfused n=32",
                                      "0.5B gemm=naive attn=naive step=unfused n=256"), BLUE, "--"),
        ("my kernels, attention fixed", ladder("0.5B gemm=naive attn=par step=unfused n=32",
                                               "0.5B gemm=naive attn=par step=unfused n=256"), BLUE, "-"),
        ("cuBLAS path, before", ladder("0.5B gemm=cublas attn=naive step=unfused n=32",
                                       "0.5B gemm=cublas attn=naive step=unfused n=256"), ORANGE, "--"),
        ("cuBLAS path, attention fixed", (98.7, 129.9), ORANGE, "-"),   # single runs, see docstring
    ]

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11})
    fig, ax = plt.subplots(figsize=(7.2, 4.6), dpi=200)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)

    x = [0, 1]
    for label, (y32, y256), color, ls in series:
        before = ls == "--"
        ax.plot(x, [y32, y256], color=color, lw=2, ls=ls, marker="o", ms=7,
                markerfacecolor=SURFACE if before else color, markeredgecolor=color,
                markeredgewidth=1.8, label=label, zorder=3)
        ax.annotate(f"{y32:.0f}", (0, y32), xytext=(-8, 0), textcoords="offset points",
                    ha="right", va="center", color=INK_2, fontsize=10)
        ax.annotate(f"{y256:.0f}", (1, y256), xytext=(8, 0), textcoords="offset points",
                    ha="left", va="center", color=INK_2, fontsize=10)

    ax.set_xlim(-0.25, 1.25)
    ax.set_xticks(x)
    ax.set_xticklabels(["32-token reply", "256-token reply"], color=INK)
    ax.set_ylim(0, 145)
    ax.set_ylabel("average decode speed, tok/s", color=INK)
    ax.yaxis.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(axis="y", colors=INK_2, length=0)
    ax.tick_params(axis="x", colors=INK, length=0)

    ax.set_title("Only the attention kernel changed. GEMV is still the naive kernel in every line.",
                 loc="left", color=INK, fontsize=11, pad=12)
    leg = ax.legend(loc="upper left", frameon=False, fontsize=9.5, labelcolor=INK_2, ncol=2,
                    handlelength=2.2, columnspacing=1.5)
    for t in leg.get_texts():
        t.set_color(INK_2)

    fig.text(0.01, 0.01, "Qwen2.5-0.5B, 12-token prompt, locked clocks. cuBLAS after fix: single runs, "
             "CPU-bound under WSL2.", color=INK_2, fontsize=8)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, facecolor=SURFACE)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
