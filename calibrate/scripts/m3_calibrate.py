"""M3: local calibration harness. Fits mfu_prefill, mbu_weights, mbu_kv,
overhead_per_iter, and kv_overhead against a real vLLM instance on whatever
GPU is available on this rig (RTX 3070 Laptop 8GB, Ampere, no native FP8 --
see specs/hardware/rtx3070-laptop.yaml). $0 cost -- the whole point of
running this before renting anything, per the roadmap.

tp_efficiency is NOT measured here -- needs >=2 GPUs, this rig has one.
Left null in the output engine spec until that's measured on rented
multi-GPU hardware.

Methodology notes (read before trusting the numbers):
  - mfu_prefill: wall-clock around llm.generate() calls, not vLLM's internal
    RequestOutput.metrics -- that field is None by default on this vLLM
    version (0.28.0, V1 engine) without additional engine config this
    project hasn't needed to figure out.
  - mbu_weights / mbu_kv / overhead_per_iter: an EARLIER version of this
    script derived these from ONE linear regression over a wall-clock
    batch sweep (iter_time = overhead_per_iter + slope*batch) -- slope gave
    a single "mbu_decode", intercept gave overhead_per_iter. That method
    was found to be structurally broken (see calibrate/error-table.md,
    "wrong functional form" section): it forced weight-reads and
    KV-cache-reads to share one bandwidth-efficiency constant, but they
    measurably don't -- weights are one large contiguous read, KV cache is
    vLLM's paged, non-contiguous per-sequence blocks. Sharing one constant
    produced a negative (physically impossible) overhead_per_iter in every
    configuration tried: this model, this rig, in eager mode AND with CUDA
    graphs enabled, and even on a 7B model on a rented A100/H100.
    This version instead uses torch.profiler to directly attribute measured
    CUDA kernel time to weight-matmul kernels vs attention/KV-read kernels
    by op name (see classify_kernel below), giving mbu_weights and mbu_kv
    as two independent direct measurements instead of one regression-
    derived (and, it turns out, meaningless) blended value.
    overhead_per_iter is the leftover: every other kernel (RMSNorm, RoPE,
    activation, etc.) plus all CPU-side dispatch time, which is real and
    substantial (a CUDA-graph A/B test cut total decode latency 2.2-2.75x
    on this model, confirming kernel-launch overhead is large, not noise).
  - Requires VLLM_ENABLE_V1_MULTIPROCESSING=0 (set below, before importing
    vllm): vLLM's V1 engine runs its actual model execution (EngineCore) in
    a SEPARATE PROCESS by default, which torch.profiler in this process
    cannot see -- it would silently report near-zero CUDA time, which would
    look exactly like "confirms CPU overhead dominates" while actually
    meaning "profiled the wrong process". Confirmed present in this vLLM
    version via its own vllm/envs.py before relying on it.
  - enforce_eager=True (no CUDA graphs), same choice m0_kv_check.py made,
    for reproducibility and because CUDA-graph capture at multiple batch
    sizes eats into this card's 8GB. These coefficients are therefore for
    eager-mode execution specifically -- likely understate what a
    CUDA-graph-optimized production deployment would achieve (measured
    directly: calibrate/scripts/m3_profile_decode.py's eager-vs-CUDA-graph
    A/B test).
  - The weight-matmul / attention-kernel classification in classify_kernel
    is a keyword heuristic based on the op names actually observed for this
    model + FlashAttention backend on this vLLM version -- not a guaranteed-
    universal classifier. Documented, not hidden.
  - kv_overhead here conflates true block-allocation overhead with
    workspace-size estimation error (WORKSPACE_RESERVE_GB below is a round
    guess, not read from vLLM's own reported peak-activation number) --
    documented, not hidden.
  - Only tested on a tiny (0.5B) model on a tiny (8GB) card. These
    coefficients are not claimed to hold for the 7B/32B-on-H100 regime the
    roadmap's own error-table example targets -- that needs the paid M3
    stage (real GPU rental) -- see calibrate/scripts/m3_calibrate_modal.py.

Usage (inside .venv):
    python3 calibrate/scripts/m3_calibrate.py
"""
from __future__ import annotations

import os

# Must be set before vllm constructs its engine -- see methodology notes above.
os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")

import random
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from core.memory import HardwareSpec, ModelSpec, kv_per_token_bytes, weights_gb  # noqa: E402
from core.roofline import prefill_flops  # noqa: E402

MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
MODEL_SPEC_NAME = "qwen2.5-0.5b"
HW_SPEC_NAME = "rtx3070-laptop"
WEIGHT_DTYPE = "bf16"  # hardware.yaml's peak_flops key; engine runs fp16 numerically, same tensor-core throughput
ENGINE_DTYPE = "float16"
KV_DTYPE = "fp16"
TP = 1
GPU_MEM_UTIL = 0.5
WORKSPACE_RESERVE_GB = 0.3  # round estimate, not read from vLLM's own peak-activation report -- see module docstring

PREFILL_LENGTHS = [256, 512, 1024, 2048]
PREFILL_N_TRIALS = 7

# torch.profiler instrumentation adds real overhead per call, and each point
# here is now an independent direct measurement (not a regression sample --
# no line needs fitting through noise), so fewer trials than a wall-clock
# sweep needs. Kept the same batch range as before for trend visibility.
DECODE_PROFILE_BATCHES = [1, 2, 4, 6, 8, 12]
DECODE_CONTEXT_LEN = 1024
DECODE_OUTPUT_TOKENS = 24
DECODE_N_TRIALS = 3


def median_time(fn, n_trials: int) -> float:
    times = []
    for _ in range(n_trials):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    return statistics.median(times)


# Keyword substrings (checked case-insensitively) that identify which real
# physical work a CUDA kernel is doing. Anything that matches neither falls
# into "overhead" -- RMSNorm, RoPE, activation kernels, and all CPU-side
# dispatch time all belong there: real but not modeled by either bandwidth
# term, and the reason a single mbu couldn't absorb everything (see module
# docstring).
_WEIGHT_KERNEL_KEYWORDS = ("mm", "gemm", "gemv", "xmma", "cutlass")
_KV_KERNEL_KEYWORDS = ("flash", "varlen", "paged", "attn")


def classify_kernel(name: str) -> str:
    lname = name.lower()
    if any(kw in lname for kw in _WEIGHT_KERNEL_KEYWORDS):
        return "weight"
    if any(kw in lname for kw in _KV_KERNEL_KEYWORDS):
        return "kv"
    return "overhead"


def profile_call(fn) -> dict:
    """Runs fn() once inside torch.profiler. Returns seconds per category:
    weight/kv from real CUDA device time (classified by op name), overhead
    from every other CUDA kernel PLUS all CPU-side time (dispatch,
    scheduling, tokenization -- real cost, just not bandwidth-modeled)."""
    from torch.profiler import ProfilerActivity, profile

    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        fn()
    events = prof.key_averages()
    cats = {"weight": 0.0, "kv": 0.0, "overhead": 0.0}
    for e in events:
        cats[classify_kernel(e.key)] += e.self_device_time_total / 1e6  # us -> s
    cats["overhead"] += sum(e.self_cpu_time_total for e in events) / 1e6
    return cats


def profiled_median_categories(fn, n_trials: int) -> dict:
    samples = [profile_call(fn) for _ in range(n_trials)]
    return {cat: statistics.median(s[cat] for s in samples) for cat in ("weight", "kv", "overhead")}


def find_cache_config(llm):
    """vLLM has moved cache_config around across versions (V0 vs V1 engine).
    Try the known locations instead of hardcoding one path -- same approach
    m0_kv_check.py already uses."""
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
    raise RuntimeError("could not locate cache_config on this vLLM version")


def main():
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt

    rng = random.Random(0)
    vocab_size = 151643  # Qwen2.5 tokenizer vocab_size, avoids importing the tokenizer just for this

    def make_prompts(n_tokens: int, batch: int):
        # Random token IDs, freshly sampled every call -- vLLM's automatic
        # prefix caching (on by default) would otherwise turn every repeat
        # of the same fixed content into a near-instant cache hit after the
        # first trial, which is exactly what happened on the first run of
        # this script: it made prefill look many times faster than the
        # hardware's own peak FLOPs could physically allow (mfu > 1, and
        # growing with n_tokens instead of converging). Also disabled
        # outright via enable_prefix_caching=False below, belt-and-suspenders.
        return [TokensPrompt(prompt_token_ids=[rng.randrange(vocab_size) for _ in range(n_tokens)]) for _ in range(batch)]

    print(f"loading {MODEL_ID} on {HW_SPEC_NAME} (dtype={ENGINE_DTYPE})...")
    llm = LLM(
        model=MODEL_ID, dtype=ENGINE_DTYPE, gpu_memory_utilization=GPU_MEM_UTIL,
        max_model_len=4096, enforce_eager=True, enable_prefix_caching=False,
    )

    model = ModelSpec.load(MODEL_SPEC_NAME)
    hw = HardwareSpec.load(HW_SPEC_NAME)
    peak_flops = hw.peak_flops[WEIGHT_DTYPE]
    bandwidth = hw.memory_bandwidth_gbs * 1e9
    w_bytes = weights_gb(model, WEIGHT_DTYPE, TP) * 1e9
    kv_bpt = kv_per_token_bytes(model, KV_DTYPE, TP)

    print("warmup...")
    llm.generate(make_prompts(64, 1), SamplingParams(max_tokens=4, temperature=0), use_tqdm=False)

    # ---- mfu_prefill ----
    print("\n--- prefill timing ---")
    mfu_samples = []
    for n in PREFILL_LENGTHS:
        prompts = make_prompts(n, 1)
        t = median_time(lambda p=prompts: llm.generate(p, SamplingParams(max_tokens=1, temperature=0), use_tqdm=False), PREFILL_N_TRIALS)
        flops = prefill_flops(model, n)["total_flops"]
        mfu = flops / (peak_flops * t)
        mfu_samples.append(mfu)
        print(f"  n_tokens={n:5d}  measured={t * 1e3:8.2f}ms  implied mfu_prefill={mfu:.4f}")

    # ---- mbu_weights / mbu_kv / overhead_per_iter: torch.profiler op-attribution ----
    print("\n--- decode timing (torch.profiler op-attribution, not batch-sweep regression) ---")
    mbu_w_samples, mbu_kv_samples, overhead_samples = [], [], []
    for b in DECODE_PROFILE_BATCHES:
        prefill_prompts = make_prompts(DECODE_CONTEXT_LEN, b)
        cats_p = profiled_median_categories(
            lambda p=prefill_prompts: llm.generate(p, SamplingParams(max_tokens=1, temperature=0), use_tqdm=False), DECODE_N_TRIALS
        )
        decode_prompts = make_prompts(DECODE_CONTEXT_LEN, b)
        cats_t = profiled_median_categories(
            lambda p=decode_prompts: llm.generate(p, SamplingParams(max_tokens=DECODE_OUTPUT_TOKENS, temperature=0), use_tqdm=False),
            DECODE_N_TRIALS,
        )
        n_iters = DECODE_OUTPUT_TOKENS - 1
        weight_per_iter = (cats_t["weight"] - cats_p["weight"]) / n_iters
        kv_per_iter = (cats_t["kv"] - cats_p["kv"]) / n_iters
        overhead_per_iter_b = (cats_t["overhead"] - cats_p["overhead"]) / n_iters

        mbu_w_b = w_bytes / (bandwidth * weight_per_iter) if weight_per_iter > 0 else float("nan")
        mbu_kv_b = (DECODE_CONTEXT_LEN * kv_bpt * b) / (bandwidth * kv_per_iter) if kv_per_iter > 0 else float("nan")
        mbu_w_samples.append(mbu_w_b)
        mbu_kv_samples.append(mbu_kv_b)
        overhead_samples.append(overhead_per_iter_b)
        print(
            f"  batch={b:3d}  weight/iter={weight_per_iter * 1e3:7.3f}ms (mbu_weights={mbu_w_b:.4f})  "
            f"kv/iter={kv_per_iter * 1e3:7.3f}ms (mbu_kv={mbu_kv_b:.4f})  overhead/iter={overhead_per_iter_b * 1e3:7.3f}ms"
        )

    mbu_weights = statistics.median(mbu_w_samples)
    # mbu_kv, unlike mbu_weights, is unreliable at small batch: at batch=1 the
    # actual KV bytes moved (~12KB) are tiny enough that the measured kv/iter
    # time is dominated by the attention kernel's own launch floor, not real
    # achieved bandwidth (mirrors mfu_prefill_best's same reasoning below --
    # small-signal points are the most overhead-contaminated). Use the
    # largest-batch (most KV bytes moved, least launch-floor-dominated)
    # sample as the best available estimate instead of a median that would
    # mix in the least-reliable points.
    mbu_kv = mbu_kv_samples[-1]
    overhead_per_iter = statistics.median(overhead_samples)

    print(f"\nimplied mbu_weights (median across batches)         = {mbu_weights:.4f}")
    print(f"implied mbu_kv (largest-batch estimate, batch={DECODE_PROFILE_BATCHES[-1]}) = {mbu_kv:.4f}")
    print(f"implied overhead_per_iter (median across batches)   = {overhead_per_iter * 1e3:.4f}ms")

    # Validated per-coefficient, not as one all-or-nothing bundle -- these are
    # three independent measurements now (unlike the old regression, where a
    # single failure meant the whole slope/intercept pair was meaningless).
    # A real run surfaced why this matters: on Modal H100 with a 7B model,
    # mbu_weights and overhead_per_iter came out perfectly sensible while
    # mbu_kv alone was garbage (grew past 1.0 with batch -- KV bytes at
    # batch<=12/context=1024 never escaped the attention kernel's own launch
    # floor on hardware this fast, so kv/iter stayed roughly flat instead of
    # scaling with bytes). Bundling all three into one gate would have thrown
    # away two perfectly good numbers because of the third.
    mbu_weights_valid = 0 < mbu_weights <= 1.0
    mbu_kv_valid = 0 < mbu_kv <= 1.0
    overhead_valid = overhead_per_iter >= 0
    if not mbu_weights_valid:
        print(f"WARNING: mbu_weights={mbu_weights:.4f} out of (0, 1] -- writing null.")
    if not mbu_kv_valid:
        print(
            f"WARNING: mbu_kv={mbu_kv:.4f} out of (0, 1] -- writing null. Likely cause: KV bytes moved at "
            f"this batch/context range never escaped the attention kernel's own launch-time floor, so "
            f"kv/iter didn't scale with bytes the way the ratio assumes -- try a larger batch range, not "
            f"more trials at this one."
        )
    if not overhead_valid:
        print(f"WARNING: overhead_per_iter={overhead_per_iter*1e3:.4f}ms is negative -- writing null.")

    # ---- kv_overhead ----
    print("\n--- kv_overhead ---")
    cache_config = find_cache_config(llm)
    num_gpu_blocks = cache_config.num_gpu_blocks
    block_size = cache_config.block_size
    usable_actual_gb = num_gpu_blocks * block_size * kv_bpt / 1e9
    usable_predicted_gb = GPU_MEM_UTIL * hw.memory_gb - w_bytes / 1e9 - WORKSPACE_RESERVE_GB
    kv_overhead = usable_actual_gb / usable_predicted_gb
    print(f"  predicted usable KV = {usable_predicted_gb:.3f} GB")
    print(f"  actual usable KV    = {usable_actual_gb:.3f} GB  ({num_gpu_blocks} blocks x {block_size} tokens/block)")
    print(f"  implied kv_overhead = {kv_overhead:.4f}")

    # mfu_prefill hadn't converged across PREFILL_LENGTHS (rose from 0.37 at
    # n=256 to 0.59+ at n=2048 -- small-n points are the most overhead-
    # contaminated). Use the largest-n sample as the best available estimate
    # rather than a median that mixes in the least-converged points.
    mfu_prefill_best = mfu_samples[-1]

    # ---- write specs/engines/vllm-<version>.yaml ----
    import vllm

    out_path = Path(__file__).resolve().parent.parent.parent / "specs" / "engines" / f"vllm-{vllm.__version__}.yaml"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    mbu_w_line = f"mbu_weights: {mbu_weights:.4f}" if mbu_weights_valid else "mbu_weights: null"
    mbu_kv_line = f"mbu_kv: {mbu_kv:.4f}" if mbu_kv_valid else "mbu_kv: null"
    overhead_line = f"overhead_per_iter_ms: {overhead_per_iter * 1e3:.4f}" if overhead_valid else "overhead_per_iter_ms: null"
    invalid_notes = []
    if not mbu_weights_valid:
        invalid_notes.append("# mbu_weights: out of (0, 1] -- see calibrate/error-table.md.")
    if not mbu_kv_valid:
        invalid_notes.append(
            "# mbu_kv: out of (0, 1] -- KV bytes at this batch/context range likely never escaped\n"
            "# the attention kernel's own launch-time floor on this hardware. See calibrate/error-table.md."
        )
    if not overhead_valid:
        invalid_notes.append("# overhead_per_iter: came out negative -- see calibrate/error-table.md.")
    decode_note = ("\n".join(invalid_notes) + "\n") if invalid_notes else ""

    content = f"""\
name: vllm-{vllm.__version__}
# M3 calibration coefficients. Fit against a SINGLE data point: {MODEL_ID}
# ({MODEL_SPEC_NAME}) on {HW_SPEC_NAME}, {ENGINE_DTYPE} weights, {KV_DTYPE} KV,
# tp={TP}, gpu_memory_utilization={GPU_MEM_UTIL}, enforce_eager=True.
# NOT validated on a 7B/32B-on-H100 regime -- that's the paid part of M3,
# not yet done. See calibrate/scripts/m3_calibrate.py for full methodology
# and caveats (torch.profiler op-attribution for decode, eager mode,
# kv_overhead conflates workspace-estimation error with true block overhead).
calibrated_on:
  model: {MODEL_SPEC_NAME}
  hardware: {HW_SPEC_NAME}
  vllm_version: "{vllm.__version__}"

# mfu_prefill rose with prompt length across the tested range (not fully
# converged even by n=2048: 0.37 @256, 0.45-0.51 @512-1024, {mfu_prefill_best:.4f} @2048) --
# small-n points are more overhead-contaminated, so this uses the largest-n
# (most-converged) sample as the best available estimate, not a median that
# would mix in the least-reliable points. Likely still an underestimate of
# the true asymptotic ceiling; larger prompts or a bigger model would refine it.
mfu_prefill: {mfu_prefill_best:.4f}

# mbu_weights/mbu_kv/overhead_per_iter measured via torch.profiler op-name
# attribution (see classify_kernel in this script), NOT a batch-sweep linear
# regression -- an earlier version of this script tried that and it was
# structurally unable to separate weight-read and KV-read bandwidth
# efficiency when they genuinely differ. See calibrate/error-table.md.
{decode_note}{mbu_w_line}
{mbu_kv_line}
{overhead_line}
kv_overhead: {kv_overhead:.4f}
tp_efficiency: null  # needs >=2 GPUs -- not measurable on this rig, deferred
"""
    out_path.write_text(content)
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
