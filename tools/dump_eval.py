#!/usr/bin/env python3
"""Evaluation tokens for perplexity (Stage 10): a public-domain text, tokenised
with the model's HF tokenizer, saved as tests/golden/<model>/eval_tokens.npy
(int64, the first --tokens ids). Default text: Alice's Adventures in
Wonderland from Project Gutenberg (cached at tests/golden/eval_text.txt).

Usage: tools/dump_eval.py [--model Qwen2.5-0.5B-Instruct] [--tokens 8192]
"""

import sys
import urllib.request
from pathlib import Path

import numpy as np
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parent.parent
URL = "https://www.gutenberg.org/cache/epub/11/pg11.txt"


def arg(name, default):
    return sys.argv[sys.argv.index(name) + 1] if name in sys.argv else default


def main():
    model = arg("--model", "Qwen2.5-0.5B-Instruct")
    n = int(arg("--tokens", "8192"))
    text_path = ROOT / "tests" / "golden" / "eval_text.txt"
    if not text_path.exists():
        raw = urllib.request.urlopen(URL, timeout=60).read().decode("utf-8", "replace")
        # strip the Gutenberg header/footer
        a, b = raw.find("CHAPTER I."), raw.rfind("*** END OF THE PROJECT GUTENBERG")
        text_path.write_text(raw[a if a > 0 else 0:b if b > 0 else len(raw)])
    text = text_path.read_text()
    tok = AutoTokenizer.from_pretrained(ROOT / "models" / model)
    ids = tok(text, add_special_tokens=False)["input_ids"][:n]
    out_dir = ROOT / "tests" / "golden" / ("" if model == "Qwen2.5-0.5B-Instruct" else model)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "eval_tokens.npy", np.asarray(ids, dtype=np.int64))
    print(f"wrote {out_dir / 'eval_tokens.npy'}: {len(ids)} tokens")


if __name__ == "__main__":
    main()
