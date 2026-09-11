#!/usr/bin/env python3
"""Download a model checkpoint from HuggingFace into models/<name>/.

Offline tooling only — never part of the serving path.

Usage: python tools/download_model.py [Qwen/Qwen2.5-0.5B-Instruct]
"""

import sys
from pathlib import Path

from huggingface_hub import snapshot_download

DEFAULT = "Qwen/Qwen2.5-0.5B-Instruct"

# Everything the engine (and Stage 4's tokenizer) needs; skips .bin duplicates etc.
PATTERNS = ["*.safetensors", "config.json", "generation_config.json",
            "tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt"]


def main() -> None:
    repo = sys.argv[1] if len(sys.argv) > 1 else DEFAULT
    dest = Path(__file__).resolve().parent.parent / "models" / repo.split("/")[-1]
    dest.mkdir(parents=True, exist_ok=True)
    path = snapshot_download(repo, allow_patterns=PATTERNS, local_dir=dest)
    print(f"downloaded {repo} -> {path}")
    for f in sorted(Path(path).glob("*")):
        print(f"  {f.name}  {f.stat().st_size:,} bytes")


if __name__ == "__main__":
    main()
