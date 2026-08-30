"""M0 sanity check: does core/memory.py's KV-per-token formula match what a
real inference engine actually allocates?

Loads a small model into vLLM on whatever GPU is available, reads back the
engine's own KV cache accounting, and compares it against
core.memory.kv_per_token_bytes() for the same model/dtype. No cloud spend —
this is meant to run on a laptop GPU before M3's paid calibration.

Usage (inside the venv created for this check):
    python3 calibrate/scripts/m0_kv_check.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from core.memory import ModelSpec, kv_per_token_bytes  # noqa: E402

MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
MODEL_SPEC_NAME = "qwen2.5-0.5b"
KV_DTYPE = "fp16"
ENGINE_DTYPE = "float16"


def find_cache_config(llm):
    """vLLM has moved cache_config around across versions (V0 vs V1 engine).
    Try the known locations instead of hardcoding one path."""
    candidates = [
        lambda: llm.llm_engine.cache_config,
        lambda: llm.llm_engine.engine_core.cache_config,
        lambda: llm.llm_engine.vllm_config.cache_config,
    ]
    for get in candidates:
        try:
            return get()
        except AttributeError:
            continue
    raise RuntimeError("could not locate cache_config on this vLLM version — inspect `llm.llm_engine` manually")


def find_model_config(llm):
    candidates = [
        lambda: llm.llm_engine.model_config,
        lambda: llm.llm_engine.vllm_config.model_config,
    ]
    for get in candidates:
        try:
            return get()
        except AttributeError:
            continue
    raise RuntimeError("could not locate model_config on this vLLM version")


def main():
    from vllm import LLM

    print(f"loading {MODEL_ID} into vLLM (dtype={ENGINE_DTYPE})...")
    llm = LLM(
        model=MODEL_ID,
        dtype=ENGINE_DTYPE,
        gpu_memory_utilization=0.5,
        max_model_len=4096,
        enforce_eager=True,
    )

    cache_config = find_cache_config(llm)
    model_config = find_model_config(llm)

    num_gpu_blocks = cache_config.num_gpu_blocks
    block_size = cache_config.block_size
    num_layers = model_config.get_num_layers(llm.llm_engine.vllm_config.parallel_config) if hasattr(model_config, "get_num_layers") else None
    num_kv_heads = model_config.get_num_kv_heads(llm.llm_engine.vllm_config.parallel_config) if hasattr(model_config, "get_num_kv_heads") else None
    head_size = model_config.get_head_size() if hasattr(model_config, "get_head_size") else None

    print("\n--- raw vLLM cache_config / model_config ---")
    print(f"num_gpu_blocks = {num_gpu_blocks}")
    print(f"block_size     = {block_size} tokens/block")
    print(f"num_layers     = {num_layers}")
    print(f"num_kv_heads   = {num_kv_heads}")
    print(f"head_size      = {head_size}")
    print(f"cache_dtype    = {cache_config.cache_dtype}")

    # vLLM's own per-token KV byte accounting, derived from its own reported
    # architecture numbers (not ours) — this is the ground truth to compare against.
    kv_dtype_bytes = 2  # fp16/bf16; vLLM's cache_config.cache_dtype confirms this above
    vllm_bytes_per_token = 2 * num_layers * num_kv_heads * head_size * kv_dtype_bytes
    vllm_total_kv_bytes = num_gpu_blocks * block_size * vllm_bytes_per_token

    model = ModelSpec.load(MODEL_SPEC_NAME)
    our_bytes_per_token = kv_per_token_bytes(model, KV_DTYPE, tp=1)

    print("\n--- comparison: bytes per token in KV cache ---")
    print(f"vLLM (from its own model_config):  {vllm_bytes_per_token} bytes/token")
    print(f"ours (core.memory formula 2):      {our_bytes_per_token} bytes/token")
    if vllm_bytes_per_token != our_bytes_per_token:
        err = abs(our_bytes_per_token - vllm_bytes_per_token) / vllm_bytes_per_token * 100
        print(f"MISMATCH: {err:.1f}% error — check specs/models/{MODEL_SPEC_NAME}.yaml against the model's real config")
    else:
        print("MATCH — formula 2 (kv_per_token_bytes) is exact for this architecture.")

    print("\n--- vLLM's actual allocated KV capacity ---")
    print(f"num_gpu_blocks x block_size = {num_gpu_blocks * block_size} tokens of KV capacity")
    print(f"= {vllm_total_kv_bytes / 1e9:.3f} GB usable KV at gpu_memory_utilization=0.5")


if __name__ == "__main__":
    main()
