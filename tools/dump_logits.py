#!/usr/bin/env python3
"""The oracle: dump golden reference data from HuggingFace for the C++ engine.

Produces, in tests/golden/:
  manifest.json            — expected tensor set (from the HF module tree), shapes/dtypes,
                             strided spot values as raw bf16 bits, sha256 of the
                             safetensors file, library versions, prompt list
  prompt<i>_fp32.npz       — goldens computed with dtype=float32 (the definition of correct)
  prompt<i>_bf16.npz       — goldens computed with dtype=bfloat16 (≈ what a bf16-weights/
                             fp32-accum engine should produce; separates bug from drift)

Each .npz contains: token_ids, logits (full seq x vocab, fp32), hidden_state_<L> for every
residual-stream boundary (NOTE: hf index 0 = embedding output, i = after layer i-1, last one
has the final norm already applied), intra-layer taps for layers 0 and 12, RoPE pins
(position_ids, cos, sin), and the attention mask actually used.

Conventions the C++ side relies on:
  * np.savez (uncompressed) — the C++ npz reader doesn't do deflate
  * all float arrays saved as float32; raw-bits arrays as uint16; ids as int64
  * attn_implementation="eager" — matches what the CPU engine mirrors (SDPA can differ
    in the last bits)
"""

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb

ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = ROOT / "models" / "Qwen2.5-0.5B-Instruct"
OUT = ROOT / "tests" / "golden"

PROMPTS = [
    "The capital of France is",
    "def fibonacci(n):",
    "1 + 1 =",
    "Once upon a time, in a village by the sea,",
    "The three laws of thermodynamics are",
]

TAP_LAYERS = [0, 12]


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def to_np(t: torch.Tensor) -> np.ndarray:
    """numpy has no bfloat16: values always go out as float32."""
    return t.detach().float().cpu().numpy().astype(np.float32)


def expected_tensor_set(model) -> list[str]:
    """The tensor names the safetensors file must contain, derived from the module
    tree: deduplicated parameters (drops tied lm_head.weight, matching what
    safetensors serialization does) plus persistent buffers."""
    names = {n for n, _ in model.named_parameters(remove_duplicate=True)}
    names |= {n for n, _ in model.named_buffers() if n in dict(model.state_dict())}
    return sorted(names)


def spot_checks(model_path: Path) -> list[dict]:
    """Raw bf16 bits at deliberately asymmetric (row, col) indices of non-square
    matrices — catches transposed-layout bugs that element (0,0) can't."""
    from safetensors import safe_open

    picks = [
        ("model.embed_tokens.weight", 12345, 678),
        ("model.layers.0.self_attn.k_proj.weight", 3, 7),      # [128, 896]: non-square
        ("model.layers.0.self_attn.q_proj.bias", 11, None),    # 1-D: bias coverage
        ("model.layers.5.mlp.gate_proj.weight", 100, 800),     # [4864, 896]
        ("model.layers.23.post_attention_layernorm.weight", 42, None),
    ]
    out = []
    with safe_open(model_path, framework="pt") as f:
        for name, row, col in picks:
            t = f.get_tensor(name)
            bits = t.view(torch.uint16)
            v = bits[row] if col is None else bits[row, col]
            out.append({"tensor": name, "row": row, "col": col, "bf16_bits": int(v)})
    return out


def run_model(dtype: torch.dtype, token_ids: torch.Tensor):
    """One forward pass with taps hooked. Returns dict of golden arrays."""
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_DIR, dtype=dtype, attn_implementation="eager"
    )
    model.eval()

    taps: dict[str, np.ndarray] = {}
    hooks = []

    def save_out(key):
        def fn(_m, _inp, out):
            taps[key] = to_np(out[0] if isinstance(out, tuple) else out)
        return fn

    def save_in(key):
        def fn(_m, inp):
            taps[key] = to_np(inp[0])
        return fn

    for li in TAP_LAYERS:
        layer = model.model.layers[li]
        p = f"layer{li}."
        hooks += [
            layer.input_layernorm.register_forward_hook(save_out(p + "post_input_layernorm")),
            layer.self_attn.q_proj.register_forward_hook(save_out(p + "q_proj")),
            layer.self_attn.k_proj.register_forward_hook(save_out(p + "k_proj")),
            layer.self_attn.v_proj.register_forward_hook(save_out(p + "v_proj")),
            layer.self_attn.o_proj.register_forward_pre_hook(save_in(p + "attn_out_pre_o_proj")),
            layer.post_attention_layernorm.register_forward_pre_hook(
                save_in(p + "post_attn_residual")),
            layer.post_attention_layernorm.register_forward_hook(
                save_out(p + "post_post_attention_layernorm")),
            layer.mlp.gate_proj.register_forward_hook(save_out(p + "mlp_gate")),
            layer.mlp.up_proj.register_forward_hook(save_out(p + "mlp_up")),
            layer.mlp.down_proj.register_forward_hook(save_out(p + "mlp_down")),
        ]

    # RoPE pins: cos/sin come from the model's own rotary module (fp32 inv_freq
    # regardless of model dtype — the CPU path must match that).
    rope: dict[str, np.ndarray] = {}

    def rope_hook(_m, _inp, out):
        cos, sin = out
        rope["cos"] = to_np(cos)
        rope["sin"] = to_np(sin)

    hooks.append(model.model.rotary_emb.register_forward_hook(rope_hook))

    # Attention mask actually used (eager path passes it into each attention module).
    mask_holder: dict[str, np.ndarray] = {}

    def mask_hook(_m, _args, kwargs):
        am = kwargs.get("attention_mask")
        if am is not None:
            mask_holder["attention_mask"] = to_np(am)
        return None

    hooks.append(model.model.layers[0].self_attn.register_forward_pre_hook(
        mask_hook, with_kwargs=True))

    with torch.no_grad():
        out = model(token_ids, output_hidden_states=True, use_cache=False)

    for h in hooks:
        h.remove()

    seq_len = token_ids.shape[1]
    position_ids = np.arange(seq_len, dtype=np.int64)[None, :]

    # Q/K after RoPE, computed with HF's own function (half-split rotate_half,
    # NOT NeoX interleaving) — golden by construction.
    cfg = model.config
    n_heads = cfg.num_attention_heads
    n_kv = cfg.num_key_value_heads
    hd = cfg.hidden_size // n_heads
    for li in TAP_LAYERS:
        p = f"layer{li}."
        q = torch.from_numpy(taps[p + "q_proj"]).view(1, seq_len, n_heads, hd).transpose(1, 2)
        k = torch.from_numpy(taps[p + "k_proj"]).view(1, seq_len, n_kv, hd).transpose(1, 2)
        cos = torch.from_numpy(rope["cos"])
        sin = torch.from_numpy(rope["sin"])
        q_rot, k_rot = apply_rotary_pos_emb(q, k, cos, sin)
        taps[p + "q_post_rope"] = to_np(q_rot)
        taps[p + "k_post_rope"] = to_np(k_rot)

    arrays = {
        "token_ids": token_ids.numpy().astype(np.int64),
        "position_ids": position_ids,
        "logits": to_np(out.logits),
        "rope_cos": rope["cos"],
        "rope_sin": rope["sin"],
        **{f"hidden_state_{i}": to_np(h) for i, h in enumerate(out.hidden_states)},
        **taps,
    }
    if "attention_mask" in mask_holder:
        arrays["attention_mask"] = mask_holder["attention_mask"]
    return arrays, model


def main() -> None:
    if not MODEL_DIR.exists():
        sys.exit(f"model not found at {MODEL_DIR}; run tools/download_model.py first")
    OUT.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(0)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    st_path = MODEL_DIR / "model.safetensors"

    model_for_manifest = None
    for i, prompt in enumerate(PROMPTS):
        ids = tokenizer(prompt, return_tensors="pt").input_ids
        for dtype, tag in [(torch.float32, "fp32"), (torch.bfloat16, "bf16")]:
            arrays, model = run_model(dtype, ids)
            np.savez(OUT / f"prompt{i}_{tag}.npz", **arrays)   # uncompressed on purpose
            print(f"prompt{i}_{tag}.npz: seq_len={ids.shape[1]}, {len(arrays)} arrays")
            if model_for_manifest is None:
                model_for_manifest = model

    # Expected tensor set from the module tree must match the file exactly —
    # asserted here at oracle time, then re-asserted by the C++ test against
    # the manifest. Catches stray buffers / wrong-variant downloads early.
    from safetensors import safe_open
    expected = expected_tensor_set(model_for_manifest)
    with safe_open(st_path, framework="pt") as f:
        in_file = sorted(f.keys())
    assert expected == in_file, (
        f"module tree vs file mismatch:\n only in tree: {set(expected) - set(in_file)}\n"
        f" only in file: {set(in_file) - set(expected)}")

    shapes = {}
    with safe_open(st_path, framework="pt") as f:
        for name in in_file:
            t = f.get_slice(name)
            shapes[name] = {"shape": list(t.get_shape()), "dtype": t.get_dtype()}

    manifest = {
        "model": "Qwen/Qwen2.5-0.5B-Instruct",
        "sha256_model_safetensors": sha256_of(st_path),
        "versions": {"transformers": transformers.__version__,
                     "torch": torch.__version__,
                     "numpy": np.__version__},
        "prompts": PROMPTS,
        "tap_layers": TAP_LAYERS,
        "expected_tensors": expected,
        "tensor_shapes": shapes,
        "spot_checks": spot_checks(st_path),
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=1))
    print(f"manifest.json: {len(expected)} tensors, sha256 {manifest['sha256_model_safetensors'][:16]}…")


if __name__ == "__main__":
    main()
