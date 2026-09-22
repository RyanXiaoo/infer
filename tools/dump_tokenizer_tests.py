#!/usr/bin/env python3
"""Golden tokenizer tests: HF encodings of a corpus, for exact-match testing
of the C++ tokenizer. Writes tests/golden/tokenizer_tests.json.

Offline oracle only, like dump_logits.py.
"""

import json
from pathlib import Path

from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = ROOT / "models" / "Qwen2.5-0.5B-Instruct"
OUT = ROOT / "tests" / "golden" / "tokenizer_tests.json"

# Ordinary text the ASCII-exact pre-tokenizer must match exactly: English,
# punctuation, digits (each its own token in Qwen), code, contractions,
# whitespace runs, leading/trailing spaces, newlines.
CORPUS = [
    "The capital of France is",
    "Hello, world!",
    "def fibonacci(n): return n if n < 2 else fibonacci(n-1) + fibonacci(n-2)",
    "It's a test, isn't it? We'll see... 42 apples and 1000 oranges.",
    "  leading and   multiple   spaces  ",
    "line one\nline two\n\nline four",
    "Numbers: 0 1 23 456 7890",
    "Mix3d alpha123numeric tokens",
    "Punctuation!!! ??? ... ---> <=== {[()]}",
    "CamelCase snake_case kebab-case SCREAMING_CASE",
    "a",
    " ",
    "",
    "Tabs\tand\ttabs",
    "The quick brown fox jumps over the lazy dog.",
]

# Adversarial corpus (Stage 9), per category, for the match-rate report.
ADVERSARIAL = {
    "unicode_letters": ["Grüße aus München", "naïve café résumé", "Ελληνικά και ελληνικά",
                        "Русский текст здесь", "עברית ועוד", "العربية لغة"],
    "cjk": ["日本語のテキストです", "中文文本示例。", "한국어 문장입니다", "混合 mixed 文本 text"],
    "emoji": ["Hello 👋 world 🌍!", "🚀🚀🚀 launch", "thumbs 👍🏽 up", "family 👨‍👩‍👧‍👦 emoji", "🇫🇷 flag"],
    "combining_marks": ["e\u0301 (e + acute)", "Zalgo t\u0361\u035ce\u0300x\u0334t", "ñ vs n\u0303"],
    "unicode_numbers": ["²³ superscripts", "½ ¼ fractions", "٣٤٥ arabic-indic", "Ⅻ roman"],
    "digit_runs": ["1234567890", "3.14159265358979", "phone 555-0123-4567", "year 2026, 1e10"],
    "whitespace": ["a\u00a0b (nbsp)", "tab\t\ttab", "trailing   \n\n  ", "\u2003em space", "mixed \t \n \r\n end",
                   "  x  ", "\n\n\n"],
    "code": ["for (int i = 0; i < n; i++) {\n\tsum += a[i];\n}", "x = {'k': [1, 2, 3]}  # dict",
             "SELECT * FROM t WHERE id >= 10;", "s = \"escaped \\\"quote\\\"\"", "<div class=\"a\">&amp;</div>"],
    "punct_runs": ["!!!???...", "--- *** ===", "«quotes» “curly” ‘single’", "a…b—c–d", "¿Qué? ¡Sí!"],
    "mixed_scripts": ["Tokyo東京Tokyo", "abc123абв", "αβγ+xyz=42", "Ürün №5", "café123"],
}

# Chat-template cases (the exact structure main_chat must reproduce).
CHATS = [
    [{"role": "user", "content": "Hi"}],
    [{"role": "user", "content": "What is the capital of France?"}],
    [{"role": "system", "content": "You are a helpful assistant."},
     {"role": "user", "content": "Explain recursion briefly."}],
]


def main() -> None:
    tok = AutoTokenizer.from_pretrained(MODEL_DIR)
    out = {
        "corpus": [{"text": s, "ids": tok(s, add_special_tokens=False)["input_ids"]}
                   for s in CORPUS],
        "chats": [{"messages": m,
                   "ids": tok.apply_chat_template(m, add_generation_prompt=True,
                                                  return_dict=True)["input_ids"]}
                  for m in CHATS],
        "adversarial": {cat: [{"text": s, "ids": tok(s, add_special_tokens=False)["input_ids"]} for s in texts]
                        for cat, texts in ADVERSARIAL.items()},
        "special": {
            "eos": tok.eos_token, "eos_id": tok.eos_token_id,
            "im_start": tok.convert_tokens_to_ids("<|im_start|>"),
            "im_end": tok.convert_tokens_to_ids("<|im_end|>"),
        },
    }
    OUT.write_text(json.dumps(out, indent=1))
    print(f"wrote {OUT}: {len(CORPUS)} corpus, {len(CHATS)} chats, "
          f"{sum(len(v) for v in ADVERSARIAL.values())} adversarial in {len(ADVERSARIAL)} categories")
    for c in out["corpus"][:3]:
        print(f"  {c['text']!r} -> {c['ids']}")


if __name__ == "__main__":
    main()
