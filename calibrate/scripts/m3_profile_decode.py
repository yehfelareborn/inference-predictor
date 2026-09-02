"""Prototype / standalone diagnostic for the CPU-vs-CUDA split that
m3_calibrate.py's decode calibration now uses by default (torch.profiler
op-attribution into mbu_weights/mbu_kv/overhead_per_iter, replacing an
earlier wall-clock batch-sweep regression that was structurally unable to
separate weight-read and KV-read bandwidth efficiency -- see
calibrate/error-table.md). This script predates that change and was how the
method was first tried out; kept as a standalone tool rather than folded in
or deleted because of one capability m3_calibrate.py doesn't have: the
M3_ENFORCE_EAGER=0 toggle below, which A/B tests eager mode against CUDA
graphs (`python3 -m calibrate.scripts.m3_profile_decode` with
M3_ENFORCE_EAGER=0) -- that comparison is what first confirmed
kernel-launch/dispatch overhead is real and large (2.2-2.75x on this model),
not just noise, and isn't something the calibration script itself needs to
do on every run.

Why this needs a special script instead of just wrapping the existing
generate() calls: vLLM's V1 engine runs its actual model-execution loop
(EngineCore) in a SEPARATE PROCESS by default (VLLM_ENABLE_V1_MULTIPROCESSING
defaults to True) -- torch.profiler in the calling process would see none of
the real CUDA kernels, only IPC plumbing, and silently report near-zero CUDA
time. That would look exactly like "confirms CPU overhead dominates" while
actually meaning "we didn't measure the GPU at all". Forcing
VLLM_ENABLE_V1_MULTIPROCESSING=0 below makes EngineCore run in-process so the
profiler can actually see its kernels -- verified via `vllm/envs.py` on this
installed version (0.28.0) before writing this script.

Scope: two batch sizes (1 and 12 -- the endpoints of the range
m3_calibrate.py sweeps), same 0.5B model on the same RTX 3070 laptop.
Aggregate CPU-vs-CUDA totals only, not the weight/kv/overhead op-name
breakdown m3_calibrate.py now computes -- see that script for the version of
this method that actually produces calibration coefficients.

Usage (inside .venv):
    python3 calibrate/scripts/m3_profile_decode.py
    M3_ENFORCE_EAGER=0 python3 calibrate/scripts/m3_profile_decode.py   # CUDA graphs on
"""
from __future__ import annotations

import os

# Must be set before vllm constructs its engine (read at LLM() construction
# time, not at import time, but setting it this early removes any doubt).
os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")

import random
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from core.memory import HardwareSpec, ModelSpec, kv_per_token_bytes  # noqa: E402

MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
MODEL_SPEC_NAME = "qwen2.5-0.5b"
HW_SPEC_NAME = "rtx3070-laptop"
ENGINE_DTYPE = "float16"
KV_DTYPE = "fp16"
GPU_MEM_UTIL = 0.5
DECODE_CONTEXT_LEN = 1024
DECODE_OUTPUT_TOKENS = 24
BATCHES = [1, 12]  # endpoints of m3_calibrate.py's DECODE_BATCHES range
N_TRIALS = 3  # profiler instrumentation itself adds overhead per run -- fewer, not more, trials than the wall-clock script

# Toggle: does CUDA-graph capture (eliminating per-kernel launch overhead)
# close the gap the eager-mode run found? M3_ENFORCE_EAGER=0 to test.
# Capture sizes restricted to exactly BATCHES -- this 8GB card can't afford
# vLLM's full default capture-size sweep on top of the model + KV cache.
ENFORCE_EAGER = os.environ.get("M3_ENFORCE_EAGER", "1") != "0"


def profiled_median(fn, n_trials: int):
    """Runs fn() n_trials times, each inside its own fresh torch.profiler
    session (not one session spanning all trials, to avoid accumulating
    events across calls). Returns (median_cpu_s, median_cuda_s, last_profiler)."""
    from torch.profiler import ProfilerActivity, profile

    cpu_times, cuda_times = [], []
    prof = None
    for _ in range(n_trials):
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
            fn()
        events = prof.key_averages()
        cpu_times.append(sum(e.self_cpu_time_total for e in events) / 1e6)  # us -> s
        cuda_times.append(sum(e.self_device_time_total for e in events) / 1e6)
    return statistics.median(cpu_times), statistics.median(cuda_times), prof


def main():
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt

    rng = random.Random(0)
    vocab_size = 151643  # Qwen2.5 tokenizer vocab_size

    def make_prompts(n_tokens: int, batch: int):
        return [TokensPrompt(prompt_token_ids=[rng.randrange(vocab_size) for _ in range(n_tokens)]) for _ in range(batch)]

    print(f"VLLM_ENABLE_V1_MULTIPROCESSING={os.environ['VLLM_ENABLE_V1_MULTIPROCESSING']}")
    print(f"loading {MODEL_ID} on {HW_SPEC_NAME} (enforce_eager={ENFORCE_EAGER})...")
    llm = LLM(
        model=MODEL_ID, dtype=ENGINE_DTYPE, gpu_memory_utilization=GPU_MEM_UTIL,
        max_model_len=4096, enforce_eager=ENFORCE_EAGER, enable_prefix_caching=False,
        compilation_config=(None if ENFORCE_EAGER else {"cudagraph_capture_sizes": BATCHES}),
    )

    model = ModelSpec.load(MODEL_SPEC_NAME)
    hw = HardwareSpec.load(HW_SPEC_NAME)
    bandwidth = hw.memory_bandwidth_gbs * 1e9
    kv_bpt = kv_per_token_bytes(model, KV_DTYPE, tp=1)

    print("warmup...")
    llm.generate(make_prompts(64, 1), SamplingParams(max_tokens=4, temperature=0), use_tqdm=False)

    print(
        f"\n{'batch':>5} {'cpu_pre':>9} {'cuda_pre':>9} {'cpu_tot':>9} {'cuda_tot':>9} "
        f"{'cpu/iter':>9} {'cuda/iter':>10} {'naive_mbu':>9}"
    )
    results = []
    for b in BATCHES:
        prefill_prompts = make_prompts(DECODE_CONTEXT_LEN, b)
        cpu_p, cuda_p, _ = profiled_median(
            lambda p=prefill_prompts: llm.generate(p, SamplingParams(max_tokens=1, temperature=0), use_tqdm=False), N_TRIALS
        )
        decode_prompts = make_prompts(DECODE_CONTEXT_LEN, b)
        cpu_t, cuda_t, prof = profiled_median(
            lambda p=decode_prompts: llm.generate(p, SamplingParams(max_tokens=DECODE_OUTPUT_TOKENS, temperature=0), use_tqdm=False),
            N_TRIALS,
        )
        n_iters = DECODE_OUTPUT_TOKENS - 1
        cpu_per_iter = (cpu_t - cpu_p) / n_iters
        cuda_per_iter = (cuda_t - cuda_p) / n_iters
        bytes_moved = DECODE_CONTEXT_LEN * kv_bpt * b
        # NOT a real mbu_kv measurement -- treats ALL cuda_per_iter as if it
        # were pure KV-bandwidth-bound, which it isn't (most of it is
        # GEMM/weight-matmul time, per the op tables printed below). Kept as
        # a cheap sanity-check number only; m3_calibrate.py's op-name
        # attribution is what actually separates these out correctly.
        naive_mbu = bytes_moved / (bandwidth * cuda_per_iter) if cuda_per_iter > 0 else float("nan")
        results.append((b, cpu_per_iter, cuda_per_iter, naive_mbu))

        print(
            f"{b:5d} {cpu_p*1e3:8.2f}m {cuda_p*1e3:8.2f}m {cpu_t*1e3:8.2f}m {cuda_t*1e3:8.2f}m "
            f"{cpu_per_iter*1e3:8.3f}m {cuda_per_iter*1e3:9.3f}m {naive_mbu:9.4f}"
        )

        print(f"\n  --- top ops by device time, batch={b} (last of {N_TRIALS} trials, full decode call) ---")
        print(prof.key_averages().table(sort_by="self_device_time_total", row_limit=10))

    print("\n--- summary ---")
    for b, cpu_i, cuda_i, naive_mbu in results:
        total = cpu_i + cuda_i
        print(
            f"batch={b:3d}  cpu/iter={cpu_i*1e3:7.3f}ms ({cpu_i/total*100:5.1f}%)  "
            f"cuda/iter={cuda_i*1e3:7.3f}ms ({cuda_i/total*100:5.1f}%)  naive_mbu={naive_mbu:.4f}"
        )


if __name__ == "__main__":
    main()
