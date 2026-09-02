"""M1: roofline latency prediction. Pure analytical FLOPs/bytes -> predicted
time, no real hardware measurement (that's M3's job).

MFU (compute efficiency) and MBU (memory-bandwidth efficiency) here are
user-supplied assumptions, not calibrated coefficients -- the roadmap's M3
stage is exactly where these get replaced with numbers fitted to real
hardware runs. Defaults (mfu=0.5, mbu=0.8) are common rule-of-thumb figures
for reasonably-optimized inference, not measurements from this project.

Two known simplifications, documented rather than silently baked in:
  - TP all-reduce communication time is added on top of max(compute, memory)
    rather than modeled as overlapping with either. Real implementations can
    overlap comm with compute for earlier layers; this conservative model
    does not credit that overlap.
  - Prefill's memory-bound time only counts reading the weights once (not KV
    cache writes). KV write bandwidth is a second-order cost the roadmap's
    formula list doesn't ask for.

Decode's memory time is split into two ADDITIVE terms -- weights_bytes/(bw*
mbu_weights) + (kv_bytes+state_bytes)/(bw*mbu_kv) -- not one combined
bytes/(bw*mbu) term. This was a real bug in an earlier version of this
model, found empirically during M3 calibration (see calibrate/error-table.md):
weight reads (one large contiguous buffer, read once per step) and KV-cache
reads (vLLM's paged, non-contiguous per-sequence blocks, read once per
sequence in the batch) measurably achieve different real bandwidth
efficiency on real hardware. Sharing one mbu constant between them forced
`overhead_per_iter` negative in every calibration attempt (0.5B/RTX3070
eager and CUDA-graph, 7B/A100, 7B/H100) -- not a measurement-noise problem,
a wrong-functional-form problem. Additive (not max'd) because the two reads
happen in sequential kernels within a decode step (weight-matmul kernels,
then attention/KV-read kernels), same reasoning already applied to TP comm
time above -- this is a conservative, no-overlap-credited model, consistent
with the rest of this module.

`overhead_per_iter_s` is a genuine fixed per-iteration cost (kernel-launch/
dispatch overhead, confirmed real and large via a torch.profiler diagnostic
-- CUDA graphs cut measured decode latency 2.2-2.75x on a 0.5B model)
added on top of max(compute_time, memory_time), not folded into either
term. Defaults to 0.0 (uncalibrated) -- M3's job is to measure it, ideally
via profiler op-level attribution (sum CUDA time for weight-matmul kernels
vs attention/KV-read kernels separately) rather than the batch-sweep linear
regression this project's calibration scripts used, which cannot identify
three unknowns (mbu_weights, mbu_kv, overhead_per_iter) from a sweep along
one variable (batch, which only ever moves kv_bytes).
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Optional

from core.memory import DTYPE_BYTES, HardwareSpec, ModelSpec, kv_per_token_bytes, weights_gb


def interconnect_bandwidth_gbs(hw: HardwareSpec) -> float:
    """NVLink bandwidth if the card has it, else fall back to PCIe -- e.g.
    l40s.yaml has no NVLink, so multi-GPU traffic goes over PCIe instead."""
    ic = hw.interconnect or {}
    if ic.get("type") not in (None, "none") and ic.get("bandwidth_gbs", 0) > 0:
        return ic["bandwidth_gbs"]
    return hw.pcie_gbs


def ridge_point(hw: HardwareSpec, dtype: str) -> Optional[float]:
    """FLOP/byte. A hardware property -- independent of MFU/MBU, which are
    efficiency derates applied after you know which side of the ridge you're
    on. Returns None if this hardware has no published peak FLOPs for dtype
    (e.g. A100 + fp8 -- Ampere has no native FP8 tensor core)."""
    peak = hw.peak_flops.get(dtype)
    if peak is None:
        return None
    return peak / (hw.memory_bandwidth_gbs * 1e9)


def tp_comm_bytes(tokens: int, hidden_size: int, dtype_bytes: int, tp: int, layers: int) -> float:
    """2 all-reduces per layer (post-attention, post-MLP). Ring all-reduce
    communication volume per GPU is 2*(tp-1)/tp times the tensor size."""
    if tp <= 1:
        return 0.0
    data_size = tokens * hidden_size * dtype_bytes
    per_layer = 2 * data_size * 2 * (tp - 1) / tp
    return layers * per_layer


def _comm_time_s(comm_bytes: float, hw: HardwareSpec) -> float:
    """Turns comm_bytes into seconds using whatever interconnect is
    available. Raises rather than silently treating an unconfigured
    interconnect as free/instant -- no current spec hits this (l40s is the
    only card with no NVLink, and it has a real pcie_gbs), but a future
    hardware spec with both interconnect.type: none and pcie_gbs: null
    would otherwise get a silently-wrong zero communication cost."""
    if comm_bytes <= 0:
        return 0.0
    bw = interconnect_bandwidth_gbs(hw)
    if not bw:
        raise ValueError(f"{hw.name} has no usable interconnect or PCIe bandwidth figure for TP communication")
    return comm_bytes / (bw * 1e9)


def _require_peak_flops(hw: HardwareSpec, dtype: str) -> float:
    peak = hw.peak_flops.get(dtype)
    if peak is None:
        raise ValueError(
            f"{hw.name} has no published peak FLOPs for dtype '{dtype}' "
            f"(see its yaml in specs/hardware/) -- pick a supported dtype instead of guessing."
        )
    return peak


# ---------------------------------------------------------------------------
# Prefill
# ---------------------------------------------------------------------------


def prefill_flops(model: ModelSpec, tokens: int) -> dict:
    """linear = 2 x active_params x tokens (standard forward-pass
    FLOPs-per-token approximation; uses active_params rather than params so
    MoE models aren't charged compute for experts that didn't run -- see
    ModelSpec.active_params). attention = 4 x L x tokens^2 x hidden_size --
    full dense s x s attention matrix cost (QK^T + softmax(QK^T)V), the same
    convention used in Kaplan et al. 2020's scaling-law FLOP counting. This
    ignores the ~2x saving causal masking gives you; it's the roadmap's
    stated formula.
    """
    linear = 2 * model.active_params() * tokens
    attention = 4 * model.layers * (tokens**2) * model.hidden_size
    return {"linear_flops": linear, "attention_flops": attention, "total_flops": linear + attention}


@dataclass
class PrefillResult:
    tokens: int
    linear_flops: float
    attention_flops: float
    total_flops: float
    weights_bytes: float
    tp_comm_time_s: float
    compute_time_s: float
    memory_time_s: float
    predicted_time_s: float
    status: str  # COMPUTE-BOUND or MEMORY-BOUND
    mfu: float
    mbu: float


def compute_prefill(
    model: ModelSpec,
    hw: HardwareSpec,
    tokens: int,
    weight_dtype: str = "fp8",
    comm_dtype: str = "bf16",
    tp: int = 1,
    mfu: float = 0.5,
    mbu: float = 0.8,
) -> PrefillResult:
    flops = prefill_flops(model, tokens)
    peak = _require_peak_flops(hw, weight_dtype)
    w_bytes = weights_gb(model, weight_dtype, tp) * 1e9

    comm_bytes = tp_comm_bytes(tokens, model.hidden_size, DTYPE_BYTES[comm_dtype], tp, model.layers)
    comm_time = _comm_time_s(comm_bytes, hw)

    # TP shards the weight matrices, so each GPU does ~1/tp of the matmul
    # FLOPs -- same reasoning weights_gb() already applies to memory. The raw
    # totals in `flops` stay undivided (they describe the whole workload, and
    # the linear/attention split percentage is tp-invariant either way).
    compute_time = flops["total_flops"] / tp / (peak * mfu)
    memory_time = w_bytes / (hw.memory_bandwidth_gbs * 1e9 * mbu)
    status = "COMPUTE-BOUND" if compute_time >= memory_time else "MEMORY-BOUND"
    predicted = max(compute_time, memory_time) + comm_time

    return PrefillResult(
        tokens=tokens,
        linear_flops=flops["linear_flops"],
        attention_flops=flops["attention_flops"],
        total_flops=flops["total_flops"],
        weights_bytes=w_bytes,
        tp_comm_time_s=comm_time,
        compute_time_s=compute_time,
        memory_time_s=memory_time,
        predicted_time_s=predicted,
        status=status,
        mfu=mfu,
        mbu=mbu,
    )


# ---------------------------------------------------------------------------
# Decode
# ---------------------------------------------------------------------------


def decode_flops(model: ModelSpec, batch: int) -> float:
    """One new token per sequence in the batch, one forward pass each.
    active_params, not params -- see ModelSpec.active_params."""
    return 2 * model.active_params() * batch


def decode_bytes(
    model: ModelSpec, hw: HardwareSpec, batch: int, context_length: int, weight_dtype: str, kv_dtype: str, tp: int
) -> dict:
    """Weights are read once per decode step, shared across the whole batch.
    KV cache is per-sequence: each of the `batch` sequences reads its own
    `context_length` tokens worth of KV. `state_bytes` is the extra
    per-sequence read for hybrid architectures' recurrent-state layers
    (Mamba-2, Gated DeltaNet, ...) -- fixed size, independent of
    context_length, 0 for standard transformers. See ModelSpec.fixed_state_bytes.
    """
    w_bytes = weights_gb(model, weight_dtype, tp) * 1e9
    kv_bpt = kv_per_token_bytes(model, kv_dtype, tp)
    kv_bytes = batch * context_length * kv_bpt
    state_bytes = batch * model.fixed_state_bytes(tp, kv_dtype)
    return {
        "weights_bytes": w_bytes,
        "kv_bytes": kv_bytes,
        "state_bytes": state_bytes,
        "total_bytes": w_bytes + kv_bytes + state_bytes,
        "kv_per_token_bytes": kv_bpt,
    }


def crossover_batch(
    model: ModelSpec, hw: HardwareSpec, context_length: int, weight_dtype: str, kv_dtype: str, tp: int
) -> Optional[float]:
    """Solve arithmetic_intensity(B) == ridge_point for B: the batch size at
    which decode flips from memory-bound to compute-bound at a fixed context
    length. Returns None if it's not solvable in the usual sense -- either
    this hardware has no peak-FLOPs figure for weight_dtype, or the KV+state
    term grows faster than the compute term per unit of batch (long enough
    context, or large enough fixed hybrid-arch state, that decode never
    becomes compute-bound no matter how large batch gets).

    Deliberately takes no mfu/mbu_weights/mbu_kv -- this answers a pure
    hardware/workload question (where do the raw FLOP and byte counts cross
    the hardware's own ridge point), not "where does compute_decode's
    max(compute_time, memory_time) actually flip" under some assumed
    efficiency. Those two crossovers only coincide if mfu and both mbu
    values happen to be equal; this function does not (and after the
    weights/kv mbu split, could not cleanly) assume that. Unaffected by the
    mbu_weights/mbu_kv split above -- it never depended on mbu in the first
    place.
    """
    ridge = ridge_point(hw, weight_dtype)
    if ridge is None:
        return None
    w_bytes = weights_gb(model, weight_dtype, tp) * 1e9
    kv_bpt = kv_per_token_bytes(model, kv_dtype, tp)
    state_bytes_per_seq = model.fixed_state_bytes(tp, kv_dtype)
    denom = 2 * model.active_params() / tp - ridge * (context_length * kv_bpt + state_bytes_per_seq)
    if denom <= 0:
        return None
    return ridge * w_bytes / denom


@dataclass
class DecodeResult:
    batch: int
    context_length: int
    weights_bytes: float
    kv_bytes: float
    state_bytes: float
    total_bytes: float
    kv_per_token_bytes: float
    total_flops: float
    arithmetic_intensity: float
    ridge_point: Optional[float]
    crossover_batch: Optional[float]
    tp_comm_time_s: float
    compute_time_s: float
    memory_time_weights_s: float
    memory_time_kv_s: float
    memory_time_s: float
    overhead_per_iter_s: float
    predicted_time_s: float
    status: str
    throughput_tokens_s: float
    mfu: float
    mbu_weights: float
    mbu_kv: float


def compute_decode(
    model: ModelSpec,
    hw: HardwareSpec,
    batch: int,
    context_length: int,
    weight_dtype: str = "fp8",
    kv_dtype: str = "fp16",
    comm_dtype: str = "bf16",
    tp: int = 1,
    mfu: float = 0.5,
    mbu_weights: float = 0.8,
    mbu_kv: float = 0.8,
    overhead_per_iter_s: float = 0.0,
) -> DecodeResult:
    peak = _require_peak_flops(hw, weight_dtype)
    b = decode_bytes(model, hw, batch, context_length, weight_dtype, kv_dtype, tp)
    flops = decode_flops(model, batch)
    ai = flops / tp / b["total_bytes"]  # per-GPU FLOPs over per-GPU bytes, same reasoning as compute_time below
    ridge = ridge_point(hw, weight_dtype)
    xover = crossover_batch(model, hw, context_length, weight_dtype, kv_dtype, tp)

    comm_bytes = tp_comm_bytes(batch, model.hidden_size, DTYPE_BYTES[comm_dtype], tp, model.layers)
    comm_time = _comm_time_s(comm_bytes, hw)

    bandwidth = hw.memory_bandwidth_gbs * 1e9
    # TP shards the weight matmul, so each GPU does ~1/tp of the FLOPs.
    compute_time = flops / tp / (peak * mfu)
    # Weights (one large contiguous read, once per step) and KV+state (vLLM's
    # paged, per-sequence blocks, once per sequence in the batch) achieve
    # different real bandwidth efficiency -- see module docstring. Additive,
    # not combined into one bytes/(bw*mbu) term, and not max'd against each
    # other either: they run in sequential kernels within a decode step.
    memory_time_weights = b["weights_bytes"] / (bandwidth * mbu_weights)
    memory_time_kv = (b["kv_bytes"] + b["state_bytes"]) / (bandwidth * mbu_kv)
    memory_time = memory_time_weights + memory_time_kv
    status = "COMPUTE-BOUND" if compute_time >= memory_time else "MEMORY-BOUND"
    predicted = max(compute_time, memory_time) + comm_time + overhead_per_iter_s
    throughput = batch / predicted if predicted > 0 else 0.0

    return DecodeResult(
        batch=batch,
        context_length=context_length,
        weights_bytes=b["weights_bytes"],
        kv_bytes=b["kv_bytes"],
        state_bytes=b["state_bytes"],
        total_bytes=b["total_bytes"],
        kv_per_token_bytes=b["kv_per_token_bytes"],
        total_flops=flops,
        arithmetic_intensity=ai,
        ridge_point=ridge,
        crossover_batch=xover,
        tp_comm_time_s=comm_time,
        compute_time_s=compute_time,
        memory_time_weights_s=memory_time_weights,
        memory_time_kv_s=memory_time_kv,
        memory_time_s=memory_time,
        overhead_per_iter_s=overhead_per_iter_s,
        predicted_time_s=predicted,
        status=status,
        throughput_tokens_s=throughput,
        mfu=mfu,
        mbu_weights=mbu_weights,
        mbu_kv=mbu_kv,
    )


def cost_per_million_tokens(price_usd_hr: float, tokens_per_sec: float) -> float:
    """$/hr / (tokens/s x 3600) x 1e6 -- $ per 1M output tokens."""
    return price_usd_hr / (tokens_per_sec * 3600) * 1e6


# ---------------------------------------------------------------------------
# Reports
# ---------------------------------------------------------------------------


def format_prefill_report(model: ModelSpec, hw: HardwareSpec, r: PrefillResult) -> str:
    attn_pct = r.attention_flops / r.total_flops * 100
    linear_pct = r.linear_flops / r.total_flops * 100
    lines = [
        f"model: {model.name}  hardware: {hw.name}",
        "",
        f"Prefill ({r.tokens} tokens)",
        f"  linear layer FLOPs   {r.linear_flops:.3e}  ({linear_pct:.1f}%)",
        f"  attention FLOPs      {r.attention_flops:.3e}  ({attn_pct:.1f}%)",
        f"  predicted TTFT       {r.predicted_time_s * 1e3:.1f} ms   (MFU {r.mfu})",
        f"  status               {r.status}",
    ]
    if r.tp_comm_time_s > 0:
        lines.append(f"  TP comm time         {r.tp_comm_time_s * 1e3:.1f} ms")
    return "\n".join(lines)


def format_decode_report(model: ModelSpec, hw: HardwareSpec, r: DecodeResult) -> str:
    lines = [
        f"model: {model.name}  hardware: {hw.name}",
        "",
        f"Decode (batch={r.batch}, context={r.context_length})",
        f"  weights read/step    {r.weights_bytes / 1e9:.1f} GB",
        f"  KV read/step         {r.kv_bytes / 1e9:.2f} GB",
    ]
    if r.state_bytes > 0:
        lines.append(f"  state read/step      {r.state_bytes / 1e9:.2f} GB  (hybrid arch: Mamba/linear-attention layers)")
    lines += [
        f"  predicted TPOT       {r.predicted_time_s * 1e3:.2f} ms   (MBU weights={r.mbu_weights}, kv={r.mbu_kv})",
        f"  status               {r.status}",
        f"  throughput           {r.throughput_tokens_s:.1f} tokens/s",
        f"  arithmetic intensity {r.arithmetic_intensity:.1f} FLOP/byte",
    ]
    if r.ridge_point is not None:
        lines.append(f"  machine ridge point  {r.ridge_point:.1f} FLOP/byte")
    if r.crossover_batch is not None:
        lines.append(f"  -> need batch ~= {r.crossover_batch:.0f} to become compute-bound at this context length")
    elif r.ridge_point is not None:
        lines.append("  -> context is long enough that decode stays memory-bound at any batch size")
    if r.tp_comm_time_s > 0:
        lines.append(f"  TP comm time         {r.tp_comm_time_s * 1e3:.2f} ms")
    if r.overhead_per_iter_s > 0:
        lines.append(f"  fixed overhead/iter  {r.overhead_per_iter_s * 1e3:.2f} ms  (kernel-launch/dispatch, M3-calibrated)")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(prog="roofline")
    sub = parser.add_subparsers(dest="command", required=True)

    pf = sub.add_parser("prefill", help="prefill FLOPs + predicted TTFT")
    pf.add_argument("--model", required=True)
    pf.add_argument("--hw", required=True)
    pf.add_argument("--tokens", type=int, default=4096)
    pf.add_argument("--dtype", default="fp8", choices=DTYPE_BYTES.keys(), help="weight precision")
    pf.add_argument("--comm-dtype", default="bf16", choices=DTYPE_BYTES.keys(), help="TP all-reduce precision")
    pf.add_argument("--tp", type=int, default=1)
    pf.add_argument("--mfu", type=float, default=0.5, help="assumed compute efficiency, 0-1")
    pf.add_argument("--mbu", type=float, default=0.8, help="assumed memory-bandwidth efficiency, 0-1")

    dc = sub.add_parser("decode", help="decode bytes + predicted TPOT")
    dc.add_argument("--model", required=True)
    dc.add_argument("--hw", required=True)
    dc.add_argument("--batch", type=int, default=32)
    dc.add_argument("--context-length", type=int, default=4096)
    dc.add_argument("--dtype", default="fp8", choices=DTYPE_BYTES.keys(), help="weight precision")
    dc.add_argument("--kv-dtype", default="fp16", choices=DTYPE_BYTES.keys())
    dc.add_argument("--comm-dtype", default="bf16", choices=DTYPE_BYTES.keys(), help="TP all-reduce precision")
    dc.add_argument("--tp", type=int, default=1)
    dc.add_argument("--mfu", type=float, default=0.5, help="assumed compute efficiency, 0-1")
    dc.add_argument("--mbu-weights", type=float, default=0.8, help="assumed weight-read bandwidth efficiency, 0-1")
    dc.add_argument("--mbu-kv", type=float, default=0.8, help="assumed KV-cache-read bandwidth efficiency, 0-1")
    dc.add_argument("--overhead-per-iter-ms", type=float, default=0.0, help="fixed per-iteration kernel-launch/dispatch overhead, ms")
    dc.add_argument("--price-usd-hr", type=float, default=None, help="optional: print $/1M output tokens")

    args = parser.parse_args()
    model = ModelSpec.load(args.model)
    hw = HardwareSpec.load(args.hw)

    from core.attribution import attribute, format_attribution  # deferred: attribution imports from this module

    if args.command == "prefill":
        r = compute_prefill(
            model, hw, args.tokens, weight_dtype=args.dtype, comm_dtype=args.comm_dtype, tp=args.tp, mfu=args.mfu, mbu=args.mbu
        )
        print(format_prefill_report(model, hw, r))
        print()
        print(format_attribution(attribute(r)))
    elif args.command == "decode":
        r = compute_decode(
            model,
            hw,
            args.batch,
            args.context_length,
            weight_dtype=args.dtype,
            kv_dtype=args.kv_dtype,
            comm_dtype=args.comm_dtype,
            tp=args.tp,
            mfu=args.mfu,
            mbu_weights=args.mbu_weights,
            mbu_kv=args.mbu_kv,
            overhead_per_iter_s=args.overhead_per_iter_ms / 1e3,
        )
        print(format_decode_report(model, hw, r))
        if args.price_usd_hr is not None:
            cost = cost_per_million_tokens(args.price_usd_hr, r.throughput_tokens_s)
            print(f"  cost                 ${cost:.2f} / 1M output tokens  (@ ${args.price_usd_hr}/hr)")
        print()
        print(format_attribution(attribute(r)))


if __name__ == "__main__":
    main()
