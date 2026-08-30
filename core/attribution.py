"""M1 bottleneck attribution: breaks a roofline prediction into where the
predicted time actually went, and suggests the single most effective lever.
This is the project's stated core differentiator -- a roofline number alone
tells you "how fast"; this tells you "why", and what to change about it.

Percentages are shares of `predicted_time_s`. Kernel-launch and other small
fixed per-iteration overheads (the roadmap's M3 calibration target
`overhead_per_iter`) are NOT modeled here -- there's no measured value to
attribute time to yet at this ($0 GPU cost) analytical stage, so it's left
out rather than filled in with an invented percentage.
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
        w_time = r.weights_bytes / r.total_bytes * r.memory_time_s
        kv_time = r.kv_bytes / r.total_bytes * r.memory_time_s
        breakdown.append(("HBM weight read", w_time / r.predicted_time_s))
        breakdown.append(("KV cache read", kv_time / r.predicted_time_s))
        if r.state_bytes > 0:
            state_time = r.state_bytes / r.total_bytes * r.memory_time_s
            breakdown.append(("recurrent state read (hybrid arch)", state_time / r.predicted_time_s))
    else:
        breakdown.append(("compute (batch large enough to be compute-bound)", r.compute_time_s / r.predicted_time_s))
    if r.tp_comm_time_s > 0:
        breakdown.append(("TP all-reduce", r.tp_comm_time_s / r.predicted_time_s))

    kv_share = r.kv_bytes / r.total_bytes if r.total_bytes else 0.0
    state_share = r.state_bytes / r.total_bytes if r.total_bytes else 0.0
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
