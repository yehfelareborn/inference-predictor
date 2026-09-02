"""M3 cloud calibration, via Modal (https://modal.com). Runs the 4 "clean"
coefficients (mfu_prefill, mbu_weights, overhead_per_iter, kv_overhead) --
NOT mbu_kv -- across multiple rented GPU types for the same model, to see
how weight-bandwidth efficiency and fixed overhead vary by hardware.

mbu_kv is deliberately excluded from this pass. It isn't just "harder to
measure" -- the investigation in calibrate/error-table.md found it doesn't
behave like a hardware-independent constant at the batch range tested
(RTX3070 needed batch~12 to converge; H100 hadn't converged even at
batch=96, and diverged further with wider batches instead of settling).
Bundling a still-exploratory, wide-batch-range investigation into this
"get 4 known-good numbers across N cards" pass would slow and complicate
both. mbu_kv gets its own dedicated follow-up.

Model: Qwen/Qwen3.8-27B (specs/models/qwen3.8-27b.yaml) -- a hybrid
Gated-DeltaNet/full-attention architecture, dense (not MoE), 27.78B params,
~55.6GB in bf16. Chosen over this project's earlier Qwen2.5-7B passes
because it's already a real, independently-sourced spec in this project
(unlike DeepSeek-V4-Flash-0731, which was considered and rejected for this
round -- 304B params needs multi-GPU TP, which is blocked by the same
flashinfer bug as tp_efficiency below, plus several undocumented novel
mechanisms (DSpark, indexed attention, compress_ratios) that would need the
paper, not just config.json, to model correctly).

Known caveat carried into this pass, not fixed: Gated DeltaNet's recurrent-
state kernels aren't recognized by classify_kernel's GEMM/attention keyword
matching (see calibrate/scripts/m3_calibrate.py) -- their real compute cost
lands in the "overhead" bucket instead of being separated out. This model's
overhead_per_iter is therefore NOT directly comparable to a pure-dense
model's overhead_per_iter -- it's real cost, not a bug, but it conflates
generic kernel-launch dispatch with genuine GDN state computation. Same
kind of documented-not-hidden conflation kv_overhead already carries
(workspace-estimation error mixed into "block overhead").

Hardware: a100-80g, h100-sxm, b200, b300 -- all real on-Modal GPU types
(confirmed via `modal billing rates`), all already have specs/hardware/
entries. l40s is deliberately excluded: 55.6GB of bf16 weights doesn't fit
its 48GB, and switching to fp8 to fit isn't a clean alternative either --
a100-80g has no native fp8 tensor core (peak_flops.fp8: null in its own
yaml), so no single dtype covers all 4 target cards plus l40s. Per this
project's own hardware inventory, l40s is mostly running smaller POC-stage
models anyway, so this isn't a representative loss.

Run: python3 -m modal run calibrate/scripts/m3_calibrate_modal.py
Writes specs/engines/vllm-<version>-<hw_spec_name>.yaml PER hardware --
not one shared vllm-<version>.yaml, which only made sense when this
project had one model/hardware combination at a time (see
calibrate/scripts/m3_calibrate.py, which still uses the unsuffixed name
for its single local RTX3070 pass).

Kept deliberately bounded (small trial counts, bf16-only, 6 decode batch
points not the wide mbu_kv-hunting sweep) to bound cost against whatever's
left of Modal's free monthly credit -- if it stops mid-run, that's an
expected outcome (out of credit), not a bug to chase.
"""
import modal

MODEL_ID = "Qwen/Qwen3.8-27B"
MODEL_SPEC_NAME = "qwen3.8-27b"
WEIGHT_DTYPE = "bf16"
ENGINE_DTYPE = "bfloat16"
KV_DTYPE = "fp16"
TP = 1
GPU_MEM_UTIL = 0.85
WORKSPACE_RESERVE_GB = 2.0  # round estimate for a much bigger model (hidden_size=5120 vs 0.5B's much
                            # smaller) on datacenter cards -- NOT sourced from a real peak-activation
                            # report, same honesty caveat as m3_calibrate.py's WORKSPACE_RESERVE_GB=0.3
VOCAB_SIZE = 248320  # from specs/models/qwen3.8-27b.yaml

PREFILL_LENGTHS = [256, 512, 1024, 2048]
PREFILL_N_TRIALS = 7
DECODE_BATCHES = [1, 2, 4, 6, 8, 12]  # mbu_weights/overhead_per_iter only -- see module docstring for why mbu_kv is excluded
DECODE_CONTEXT_LEN = 1024
DECODE_OUTPUT_TOKENS = 24
DECODE_N_TRIALS = 3  # torch.profiler instrumentation adds real overhead per call

HARDWARE = [
    ("a100-80g", "A100-80GB"),
    ("h100-sxm", "H100"),
    ("b200", "B200"),
    ("b300", "B300"),
]
# NOTE: a100-80g and h100-sxm already completed successfully in an earlier
# run of this sweep (before the FLASH_ATTN fix above was added -- Hopper and
# older cards never hit the bug it fixes, so those two results are still
# valid) -- temporarily narrowed to just the two that crashed, to avoid
# re-paying for a re-run of the two that already succeeded. Restore the
# full HARDWARE list above for the next from-scratch run.
HARDWARE = [hw for hw in HARDWARE if hw[0] in ("b200", "b300")]

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("vllm==0.28.0", "pyyaml")
    # See calibrate/scripts/m3_calibrate.py's module docstring for both of
    # these: flashinfer's CUDA-JIT sampler can't find a matching nvcc on this
    # bare image, and vLLM V1's EngineCore runs in a separate process by
    # default, which torch.profiler in the calling process can't see into.
    .env({"VLLM_USE_FLASHINFER_SAMPLER": "0", "VLLM_ENABLE_V1_MULTIPROCESSING": "0"})
    .add_local_dir("core", remote_path="/root/core")
    .add_local_dir("specs", remote_path="/root/specs")
)

# A THIRD, separate nvcc dependency -- found the hard way on the first B200
# run of this sweep: vLLM auto-selects attention backends per-GPU
# (vllm/platforms/cuda.py), and for Blackwell (compute capability major==10)
# it prefers FLASHINFER's TensorRT-LLM fused-attention kernel *first*, ahead
# of FlashAttention. That kernel JIT-compiles via nvcc on first use --
# same missing-nvcc failure as the sampler, just a different code path the
# VLLM_USE_FLASHINFER_SAMPLER=0 fix above doesn't reach (A100/H100 never hit
# this because Hopper's backend priority list puts FLASH_ATTN first).
# Forcing FLASH_ATTN explicitly avoids it everywhere, not just Blackwell --
# it's in every GPU's candidate list per cuda.py, so this is a safe default
# across the whole HARDWARE sweep, not a per-card special case.
ATTENTION_CONFIG = {"backend": "FLASH_ATTN"}

app = modal.App("inference-predictor-m3-sweep", image=image)


def _median_time(fn, n_trials: int) -> float:
    import statistics
    import time

    times = []
    for _ in range(n_trials):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    return statistics.median(times)


# Same classification + profiling approach as m3_calibrate.py (duplicated,
# not imported -- this file's own module gets mounted into the Modal
# container automatically, calibrate/ as a package is not).
_WEIGHT_KERNEL_KEYWORDS = ("mm", "gemm", "gemv", "xmma", "cutlass")
_KV_KERNEL_KEYWORDS = ("flash", "varlen", "paged", "attn")


def _classify_kernel(name: str) -> str:
    lname = name.lower()
    if any(kw in lname for kw in _WEIGHT_KERNEL_KEYWORDS):
        return "weight"
    if any(kw in lname for kw in _KV_KERNEL_KEYWORDS):
        return "kv"
    return "overhead"


def _profile_call(fn) -> dict:
    from torch.profiler import ProfilerActivity, profile

    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        fn()
    events = prof.key_averages()
    cats = {"weight": 0.0, "kv": 0.0, "overhead": 0.0}
    for e in events:
        cats[_classify_kernel(e.key)] += e.self_device_time_total / 1e6
    cats["overhead"] += sum(e.self_cpu_time_total for e in events) / 1e6
    return cats


def _profiled_median_categories(fn, n_trials: int) -> dict:
    import statistics

    samples = [_profile_call(fn) for _ in range(n_trials)]
    return {cat: statistics.median(s[cat] for s in samples) for cat in ("weight", "kv", "overhead")}


def _find_cache_config(llm):
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


@app.function(gpu="A100-80GB", timeout=1800)
def calibrate_all(hw_spec_name: str) -> dict:
    import os

    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"  # belt-and-suspenders on top of the image-level .env

    import random
    import statistics
    import sys

    sys.path.insert(0, "/root")
    from core.memory import HardwareSpec, ModelSpec, kv_per_token_bytes, weights_gb
    from core.roofline import prefill_flops
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt

    model = ModelSpec.load(MODEL_SPEC_NAME)
    hw = HardwareSpec.load(hw_spec_name)
    peak_flops = hw.peak_flops[WEIGHT_DTYPE]
    bandwidth = hw.memory_bandwidth_gbs * 1e9
    kv_bpt = kv_per_token_bytes(model, KV_DTYPE, TP)
    w_bytes = weights_gb(model, WEIGHT_DTYPE, TP) * 1e9

    rng = random.Random(0)

    def make_prompts(n_tokens, batch):
        return [TokensPrompt(prompt_token_ids=[rng.randrange(VOCAB_SIZE) for _ in range(n_tokens)]) for _ in range(batch)]

    print(f"loading {MODEL_ID} on {hw_spec_name}...")
    llm = LLM(
        model=MODEL_ID, dtype=ENGINE_DTYPE, gpu_memory_utilization=GPU_MEM_UTIL,
        max_model_len=4096, enforce_eager=True, enable_prefix_caching=False,
        attention_config=ATTENTION_CONFIG,
    )
    llm.generate(make_prompts(64, 1), SamplingParams(max_tokens=4, temperature=0), use_tqdm=False)

    # ---- mfu_prefill ----
    print("\n--- prefill timing ---")
    mfu_samples = []
    for n in PREFILL_LENGTHS:
        prompts = make_prompts(n, 1)
        t = _median_time(lambda p=prompts: llm.generate(p, SamplingParams(max_tokens=1, temperature=0), use_tqdm=False), PREFILL_N_TRIALS)
        flops = prefill_flops(model, n)["total_flops"]
        mfu = flops / (peak_flops * t)
        mfu_samples.append(mfu)
        print(f"  n_tokens={n:5d}  measured={t * 1e3:8.2f}ms  implied mfu_prefill={mfu:.4f}")
    mfu_prefill_best = mfu_samples[-1]  # largest-n = most-converged, same reasoning as m3_calibrate.py

    # ---- mbu_weights / overhead_per_iter: torch.profiler op-attribution (mbu_kv skipped, see module docstring) ----
    print("\n--- decode timing (torch.profiler op-attribution) ---")
    mbu_w_samples, overhead_samples = [], []
    for b in DECODE_BATCHES:
        prefill_prompts = make_prompts(DECODE_CONTEXT_LEN, b)
        cats_p = _profiled_median_categories(
            lambda p=prefill_prompts: llm.generate(p, SamplingParams(max_tokens=1, temperature=0), use_tqdm=False), DECODE_N_TRIALS
        )
        decode_prompts = make_prompts(DECODE_CONTEXT_LEN, b)
        cats_t = _profiled_median_categories(
            lambda p=decode_prompts: llm.generate(p, SamplingParams(max_tokens=DECODE_OUTPUT_TOKENS, temperature=0), use_tqdm=False),
            DECODE_N_TRIALS,
        )
        n_iters = DECODE_OUTPUT_TOKENS - 1
        weight_per_iter = (cats_t["weight"] - cats_p["weight"]) / n_iters
        overhead_per_iter_b = (cats_t["overhead"] - cats_p["overhead"]) / n_iters
        mbu_w_b = w_bytes / (bandwidth * weight_per_iter) if weight_per_iter > 0 else float("nan")
        mbu_w_samples.append(mbu_w_b)
        overhead_samples.append(overhead_per_iter_b)
        print(f"  batch={b:3d}  weight/iter={weight_per_iter*1e3:7.3f}ms (mbu_weights={mbu_w_b:.4f})  overhead/iter={overhead_per_iter_b*1e3:7.3f}ms")

    mbu_weights = statistics.median(mbu_w_samples)
    overhead_per_iter = statistics.median(overhead_samples)
    mbu_weights_valid = 0 < mbu_weights <= 1.0
    overhead_valid = overhead_per_iter >= 0
    print(f"\nimplied mbu_weights (median)      = {mbu_weights:.4f}  (valid: {mbu_weights_valid})")
    print(f"implied overhead_per_iter (median) = {overhead_per_iter*1e3:.4f}ms  (valid: {overhead_valid})")

    # ---- kv_overhead ----
    print("\n--- kv_overhead ---")
    cache_config = _find_cache_config(llm)
    num_gpu_blocks = cache_config.num_gpu_blocks
    block_size = cache_config.block_size
    usable_actual_gb = num_gpu_blocks * block_size * kv_bpt / 1e9
    usable_predicted_gb = GPU_MEM_UTIL * hw.memory_gb - w_bytes / 1e9 - WORKSPACE_RESERVE_GB
    kv_overhead = usable_actual_gb / usable_predicted_gb
    print(f"  predicted usable KV = {usable_predicted_gb:.3f} GB")
    print(f"  actual usable KV    = {usable_actual_gb:.3f} GB  ({num_gpu_blocks} blocks x {block_size} tokens/block)")
    print(f"  implied kv_overhead = {kv_overhead:.4f}")

    return {
        "hw_spec_name": hw_spec_name,
        "mfu_prefill": mfu_prefill_best,
        "mfu_samples": mfu_samples,
        "mbu_weights": mbu_weights if mbu_weights_valid else None,
        "overhead_per_iter_ms": overhead_per_iter * 1e3 if overhead_valid else None,
        "kv_overhead": kv_overhead,
        "num_gpu_blocks": num_gpu_blocks,
        "block_size": block_size,
    }


@app.function(gpu="A100-80GB:2", timeout=1200)
def calibrate_tp_efficiency():
    import random
    import sys

    sys.path.insert(0, "/root")
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt

    rng = random.Random(0)
    context_len = 1024
    batch = 16
    output_tokens = 16
    n_trials = 3

    def make_prompts():
        return [TokensPrompt(prompt_token_ids=[rng.randrange(VOCAB_SIZE) for _ in range(context_len)]) for _ in range(batch)]

    results = {}
    for tp in (1, 2):
        print(f"\n--- tp={tp} ---")
        llm = LLM(model=MODEL_ID, dtype=ENGINE_DTYPE, gpu_memory_utilization=GPU_MEM_UTIL,
                  max_model_len=4096, enforce_eager=True, enable_prefix_caching=False, tensor_parallel_size=tp,
                  attention_config=ATTENTION_CONFIG)
        llm.generate(make_prompts(), SamplingParams(max_tokens=4, temperature=0), use_tqdm=False)

        t_prefill = _median_time(
            lambda: llm.generate(make_prompts(), SamplingParams(max_tokens=1, temperature=0), use_tqdm=False), n_trials
        )
        t_total = _median_time(
            lambda: llm.generate(make_prompts(), SamplingParams(max_tokens=output_tokens, temperature=0), use_tqdm=False), n_trials
        )
        iter_time = (t_total - t_prefill) / (output_tokens - 1)
        results[tp] = iter_time
        print(f"  tp={tp}  per-iter decode time = {iter_time*1e3:.3f}ms")
        del llm

    ideal_speedup = 2.0
    actual_speedup = results[1] / results[2]
    tp_efficiency = actual_speedup / ideal_speedup
    print(f"\nactual speedup tp=1->tp=2: {actual_speedup:.3f}x (ideal: {ideal_speedup}x)")
    print(f"implied tp_efficiency: {tp_efficiency:.4f}")

    return {"tp1_iter_ms": results[1] * 1e3, "tp2_iter_ms": results[2] * 1e3, "tp_efficiency": tp_efficiency}


def _write_yaml(result: dict, vllm_version: str) -> None:
    from pathlib import Path

    hw_spec_name = result["hw_spec_name"]
    out_path = Path(__file__).resolve().parent.parent.parent / "specs" / "engines" / f"vllm-{vllm_version}-{hw_spec_name}.yaml"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    mbu_w_line = f"mbu_weights: {result['mbu_weights']:.4f}" if result["mbu_weights"] is not None else "mbu_weights: null"
    overhead_line = (
        f"overhead_per_iter_ms: {result['overhead_per_iter_ms']:.4f}"
        if result["overhead_per_iter_ms"] is not None else "overhead_per_iter_ms: null"
    )
    mfu_samples_str = ", ".join(f"{v:.4f}" for v in result["mfu_samples"])

    content = f"""\
name: vllm-{vllm_version}-{hw_spec_name}
# M3 calibration coefficients. Fit against: {MODEL_ID} ({MODEL_SPEC_NAME}) on
# {hw_spec_name}, {ENGINE_DTYPE} weights, {KV_DTYPE} KV, tp={TP},
# gpu_memory_utilization={GPU_MEM_UTIL}, enforce_eager=True. See
# calibrate/scripts/m3_calibrate_modal.py for full methodology and caveats
# -- notably: mbu_kv NOT attempted here (see that script's module
# docstring), and overhead_per_iter for this hybrid-architecture model
# conflates genuine Gated-DeltaNet state computation with generic
# kernel-launch dispatch (classify_kernel doesn't recognize GDN kernels).
calibrated_on:
  model: {MODEL_SPEC_NAME}
  hardware: {hw_spec_name}
  vllm_version: "{vllm_version}"

# mfu_prefill samples across n_tokens={PREFILL_LENGTHS}: [{mfu_samples_str}]
# -- uses the largest-n (most-converged) sample, not a median, same
# reasoning as m3_calibrate.py's local pass.
mfu_prefill: {result['mfu_prefill']:.4f}
{mbu_w_line}
mbu_kv: null  # not attempted in this pass -- see module docstring
{overhead_line}
kv_overhead: {result['kv_overhead']:.4f}
tp_efficiency: null  # still blocked on the flashinfer array.array[int] bug
"""
    out_path.write_text(content)
    print(f"wrote {out_path}")


@app.local_entrypoint()
def main():
    # Hardcoded to match the version this file's own image pip_installs
    # (see `image = ...pip_install("vllm==0.28.0", ...)` above) -- needed
    # here only for the output yaml filename, not to actually import vllm.
    vllm_version = "0.28.0"

    results = []
    for hw_spec_name, gpu in HARDWARE:
        print(f"\n=== {hw_spec_name} ({gpu}) ===")
        result = calibrate_all.with_options(gpu=gpu).remote(hw_spec_name)
        print(result)
        results.append(result)
        _write_yaml(result, vllm_version)

    print("\n=== summary ===")
    for r in results:
        print(
            f"{r['hw_spec_name']:12s}  mfu_prefill={r['mfu_prefill']:.4f}  "
            f"mbu_weights={r['mbu_weights']}  overhead_per_iter_ms={r['overhead_per_iter_ms']}  "
            f"kv_overhead={r['kv_overhead']:.4f}"
        )

    # tp_efficiency still blocked on the flashinfer array.array[int] bug --
    # skipped here to avoid paying for a 2-GPU call that's known to crash.
    # Re-enable once that's worked around.
    return
    print("\n=== tp_efficiency (2 GPUs) ===")
    tp_result = calibrate_tp_efficiency.remote()
    print(tp_result)
