"""M1 bottleneck attribution: breaks a roofline prediction into where the
predicted time actually went, and suggests the single most effective lever.
This is the project's stated core differentiator -- a roofline number alone
tells you "how fast"; this tells you "why", and what to change about it.

Percentages are shares of `predicted_time_s`. Decode's HBM weight read and
KV cache read are broken out using DecodeResult's own separately-computed
memory_time_weights_s/memory_time_kv_s (each uses its own calibrated mbu --
see core/roofline.py's module docstring for why weights and KV cache don't
share one bandwidth-efficiency constant), not a byte-count-proportional
split of one combined memory_time_s -- those two no longer move together
once the two mbu values differ. Kernel-launch/dispatch overhead
(`overhead_per_iter_s`) is attributed directly when a caller has supplied a
calibrated (nonzero) value; left out of the breakdown when it's the
uncalibrated default of 0.0 rather than shown as a misleading 0.0% line.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple, Union

from core.roofline import DecodeResult, PrefillResult

Result = Union[PrefillResult, DecodeResult]


@dataclass
class Attribution:
    breakdown: List[Tuple[str, float]]  # (label, fraction of predicted_time_s)
    lever: str


def attribute_prefill(r: PrefillResult) -> Attribution:
    breakdown: List[Tuple[str, float]] = []
    if r.status == "COMPUTE-BOUND":
        linear_time = r.linear_flops / r.total_flops * r.compute_time_s
        attn_time = r.attention_flops / r.total_flops * r.compute_time_s
        breakdown.append(("linear layer compute", linear_time / r.predicted_time_s))
        breakdown.append(("attention compute", attn_time / r.predicted_time_s))
    else:
        breakdown.append(("HBM weight read (compute underutilized)", r.memory_time_s / r.predicted_time_s))
    if r.tp_comm_time_s > 0:
        breakdown.append(("TP all-reduce", r.tp_comm_time_s / r.predicted_time_s))

    attn_share = r.attention_flops / r.total_flops
    comm_share = r.tp_comm_time_s / r.predicted_time_s
    if r.status == "MEMORY-BOUND":
        lever = "prompt too short to saturate compute -- batch multiple prefills together, or accept this as a latency floor"
    elif comm_share > 0.2:
        lever = "TP all-reduce is a large share of TTFT -- reduce tp degree, or use a card/topology with faster interconnect"
    elif attn_share > 0.3:
        lever = "attention FLOPs dominate (long context) -- shorten context, or use context/sequence parallelism"
    else:
        lever = "linear-layer compute dominates -- lower weight precision (e.g. fp8), or use a smaller model"
    return Attribution(breakdown=breakdown, lever=lever)


def attribute_decode(r: DecodeResult) -> Attribution:
    breakdown: List[Tuple[str, float]] = []
    if r.status == "MEMORY-BOUND":
        # kv_time here covers kv_bytes + state_bytes together (both share
        # mbu_kv and both live in memory_time_kv_s -- see compute_decode).
        # Split state back out proportionally by bytes just for display;
        # they're not independently measurable from memory_time_kv_s alone.
        kv_and_state_time = r.memory_time_kv_s
        state_time = 0.0
        if r.state_bytes > 0 and (r.kv_bytes + r.state_bytes) > 0:
            state_time = r.state_bytes / (r.kv_bytes + r.state_bytes) * kv_and_state_time
        kv_time = kv_and_state_time - state_time
        breakdown.append(("HBM weight read", r.memory_time_weights_s / r.predicted_time_s))
        breakdown.append(("KV cache read", kv_time / r.predicted_time_s))
        if state_time > 0:
            breakdown.append(("recurrent state read (hybrid arch)", state_time / r.predicted_time_s))
    else:
        breakdown.append(("compute (batch large enough to be compute-bound)", r.compute_time_s / r.predicted_time_s))
    if r.tp_comm_time_s > 0:
        breakdown.append(("TP all-reduce", r.tp_comm_time_s / r.predicted_time_s))
    if r.overhead_per_iter_s > 0:
        breakdown.append(("kernel-launch/dispatch overhead", r.overhead_per_iter_s / r.predicted_time_s))

    # Time shares, not byte shares -- weights and KV+state no longer move
    # together once mbu_weights != mbu_kv, so a byte-count fraction of
    # total_bytes would misjudge which one actually dominates predicted_time_s.
    kv_share = kv_time / r.predicted_time_s if r.status == "MEMORY-BOUND" else 0.0
    state_share = state_time / r.predicted_time_s if r.status == "MEMORY-BOUND" else 0.0
    comm_share = r.tp_comm_time_s / r.predicted_time_s
    if r.status == "COMPUTE-BOUND":
        lever = "already compute-bound -- lower weight precision, or reduce batch if latency (not throughput) is the priority"
    elif comm_share > 0.2:
        lever = "TP all-reduce is a large share of TPOT -- reduce tp degree, or use a card/topology with faster interconnect"
    elif state_share > 0.5:
        lever = (
            "fixed recurrent-state read dominates -- this doesn't shrink with batch or context (it's constant "
            "per sequence), only with tp or a different model; raising batch size helps less here than usual"
        )
    elif kv_share > 0.5:
        lever = "KV cache read dominates -- shorten context, lean harder on GQA, or use KV cache offload"
    else:
        lever = "HBM weight read dominates -- raise batch size, or lower weight precision"
    return Attribution(breakdown=breakdown, lever=lever)


def attribute(r: Result) -> Attribution:
    # Duck-typed rather than isinstance: `python3 -m core.roofline` loads that
    # module as `__main__`, so a PrefillResult/DecodeResult constructed there
    # is a different class object than the one this module imported as
    # `core.roofline` -- isinstance would spuriously fail across that split.
    if hasattr(r, "linear_flops"):
        return attribute_prefill(r)  # type: ignore[arg-type]
    if hasattr(r, "weights_bytes"):
        return attribute_decode(r)  # type: ignore[arg-type]
    raise TypeError(f"don't know how to attribute {type(r)}")


def format_attribution(a: Attribution) -> str:
    lines = ["bottleneck attribution"]
    for label, frac in sorted(a.breakdown, key=lambda item: -item[1]):
        lines.append(f"  {frac * 100:5.1f}%  {label}")
    lines.append(f"  most effective lever: {a.lever}")
    return "\n".join(lines)
