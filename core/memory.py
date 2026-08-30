"""M0: capacity calculator. Pure memory accounting, no timing.

Answers exactly one question: does this model fit on this card, and how many
concurrent sequences can it hold.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import yaml

SPECS_DIR = Path(__file__).resolve().parent.parent / "specs"

# bytes per element; dtype names follow the strings used in the specs
DTYPE_BYTES = {
    "fp32": 4,
    "fp16": 2,
    "bf16": 2,
    "fp8": 1,
    "int8": 1,
}


@dataclass
class ModelSpec:
    name: str
    params: float
    layers: int
    hidden_size: int
    n_heads: int
    n_kv_heads: int
    head_dim: int
    vocab_size: int
    moe: Optional[dict] = None
    # Hybrid archs only (Mamba-2, Gated DeltaNet, ...): {count, state_elements_per_seq, state_dtype}.
    # state_dtype is "fixed_fp32" (Mamba-2: state is always kept in fp32
    # regardless of model dtype -- a numerical-stability convention, not a
    # serving choice) or "kv_dtype" (Gated DeltaNet: state is stored at
    # whatever dtype the model runs in -- approximated here by reusing
    # --kv-dtype, the closest lever this project exposes, though a real
    # serving engine may or may not actually tie the two together).
    # See fixed_state_bytes().
    state_layers: Optional[dict] = None

    @classmethod
    def load(cls, name: str) -> "ModelSpec":
        return cls(**_load_yaml(SPECS_DIR / "models" / f"{name}.yaml"))

    def fixed_state_bytes(self, tp: int, kv_dtype: str) -> float:
        """Extra per-sequence memory from recurrent-state layers (Mamba-2,
        Gated DeltaNet, ...) that kv_per_token_bytes() doesn't capture,
        because it's a FIXED size independent of context length -- unlike a
        real KV cache, which grows with every token. `layers`/`n_kv_heads`/
        `head_dim` on this spec represent only the true full-attention
        layers (established convention: see nemotron-nano-9b.yaml and
        qwen3.8-27b.yaml); this covers the rest.

        Returns 0.0 for models with no `state_layers` block (dense/MoE
        transformers) -- capacity formulas that add this in are unchanged
        for every model this project modeled before hybrid architectures
        showed up.

        Raises if `state_layers` is present but `state_elements_per_seq` is
        still null (state size not yet researched/confirmed) -- this
        project's convention is to error on an unfilled number rather than
        silently treat it as 0, which would just reproduce the old
        known-underestimate silently instead of visibly.
        """
        if not self.state_layers:
            return 0.0
        elements = self.state_layers.get("state_elements_per_seq")
        if elements is None:
            raise ValueError(
                f"{self.name} has state_layers but state_elements_per_seq is still null "
                f"(recurrent-state size not yet confirmed) -- capacity for this model "
                f"can't be computed until that's filled in, see its yaml comments."
            )
        dtype_bytes = 4 if self.state_layers.get("state_dtype") == "fixed_fp32" else DTYPE_BYTES[kv_dtype]
        return self.state_layers["count"] * elements * dtype_bytes / tp

    def active_params(self) -> float:
        """Params actually multiplied-through per token -- what FLOPs scale
        with. Equal to `params` for dense models. For MoE, only `n_active` of
        `n_experts` are run per token, so FLOPs scale with a fraction of
        `params`, even though *memory* (weights_gb) still needs all of them
        resident. Approximated as `params * n_active/n_experts`: exact if
        every parameter lives inside a routed expert, an overestimate of the
        active fraction to the extent attention/embedding/router params
        (never duplicated per-expert) are a non-trivial share of the total --
        no field in this schema separates dense-backbone params from
        expert-FFN params, so this is a documented approximation, not exact.
        """
        if not self.moe:
            return self.params
        return self.params * self.moe["n_active"] / self.moe["n_experts"]


@dataclass
class HardwareSpec:
    name: str
    memory_gb: float
    memory_bandwidth_gbs: float
    peak_flops: dict = field(default_factory=dict)
    interconnect: dict = field(default_factory=dict)
    pcie_gbs: float = 0.0

    @classmethod
    def load(cls, name: str) -> "HardwareSpec":
        return cls(**_load_yaml(SPECS_DIR / "hardware" / f"{name}.yaml"))


def _coerce_numeric(obj):
    """PyYAML's default resolver won't parse scientific notation without an
    explicit sign (e.g. `32.8e9`) as a float — it stays a string. The specs
    follow the roadmap's notation, so coerce on load instead of rewriting
    every yaml file to `32.8e+9`."""
    if isinstance(obj, dict):
        return {k: _coerce_numeric(v) for k, v in obj.items()}
    if isinstance(obj, str):
        try:
            return float(obj)
        except ValueError:
            return obj
    return obj


def _load_yaml(path: Path) -> dict:
    with open(path) as f:
        return _coerce_numeric(yaml.safe_load(f))


@dataclass
class CapacityResult:
    weights_gb: float
    workspace_gb: float
    usable_kv_gb: float
    kv_per_token_bytes: float
    fixed_state_bytes: float  # 0 for standard transformers, >0 for hybrid archs
    concurrency_at: dict  # seq_len -> max concurrent sequences


def weights_gb(model: ModelSpec, weight_dtype: str, tp: int) -> float:
    """Formula 1: weights = params x bytes_per_param (divide by tp after TP)

    A MoE model's `params` is assumed to already count all experts (weights
    occupy memory regardless of how many experts are active).
    """
    return model.params * DTYPE_BYTES[weight_dtype] / tp / 1e9


def kv_per_token_bytes(model: ModelSpec, kv_dtype: str, tp: int) -> float:
    """Formula 2: kv_per_token = 2 x L x n_kv_heads x head_dim x kv_bytes (divide by tp after TP)

    The leading 2 is one copy each for K and V. Using n_kv_heads instead of
    n_heads is what makes GQA cheaper on KV cache.
    """
    return 2 * model.layers * model.n_kv_heads * model.head_dim * DTYPE_BYTES[kv_dtype] / tp


def usable_kv_gb(hw: HardwareSpec, weights_gb_: float, workspace_reserve_gb: float) -> float:
    """Formula 4: usable KV = memory - weights - workspace"""
    return hw.memory_gb - weights_gb_ - workspace_reserve_gb


def max_concurrency(
    usable_kv_gb_: float, kv_per_token_bytes_: float, seq_len: int, fixed_state_bytes: float = 0.0
) -> int:
    """Formula 5: concurrency = usable KV / (kv_per_token x seq_len + fixed_state_bytes)

    fixed_state_bytes is the per-sequence memory that does NOT grow with
    context length -- non-zero only for hybrid architectures with
    recurrent-state layers (Mamba-2, Gated DeltaNet, ...). 0 for every
    standard transformer, which reduces this to the original formula
    exactly -- this is a generalization, not a behavior change, for models
    without state_layers.
    """
    if usable_kv_gb_ <= 0:
        return 0
    usable_kv_bytes = usable_kv_gb_ * 1e9
    per_seq_bytes = kv_per_token_bytes_ * seq_len + fixed_state_bytes
    if per_seq_bytes <= 0:
        return 0
    return int(usable_kv_bytes // per_seq_bytes)


def compute_capacity(
    model: ModelSpec,
    hw: HardwareSpec,
    weight_dtype: str = "fp8",
    kv_dtype: str = "fp16",
    tp: int = 1,
    workspace_reserve_gb: float = 4.0,
    context_lengths=(8192, 32768, 131072),
) -> CapacityResult:
    w_gb = weights_gb(model, weight_dtype, tp)
    kv_bpt = kv_per_token_bytes(model, kv_dtype, tp)
    fixed_state = model.fixed_state_bytes(tp, kv_dtype)
    kv_gb = usable_kv_gb(hw, w_gb, workspace_reserve_gb)
    concurrency = {s: max_concurrency(kv_gb, kv_bpt, s, fixed_state) for s in context_lengths}
    return CapacityResult(
        weights_gb=w_gb,
        workspace_gb=workspace_reserve_gb,
        usable_kv_gb=kv_gb,
        kv_per_token_bytes=kv_bpt,
        fixed_state_bytes=fixed_state,
        concurrency_at=concurrency,
    )


def format_report(model: ModelSpec, hw: HardwareSpec, result: CapacityResult) -> str:
    lines = [
        f"model: {model.name}  hardware: {hw.name}",
        "",
        "memory breakdown (per GPU)",
        f"  weights               {result.weights_gb:6.1f} GB",
        f"  workspace/activation  {result.workspace_gb:6.1f} GB",
        f"  usable KV             {result.usable_kv_gb:6.1f} GB",
        "  ─────────────────────────",
        f"  KV per token          {result.kv_per_token_bytes/1024:6.1f} KB",
    ]
    if result.fixed_state_bytes > 0:
        lines.append(f"  + fixed state/seq     {result.fixed_state_bytes/1024:6.1f} KB  (hybrid arch: Mamba/linear-attention layers)")
    lines.append("")
    lines.append("capacity")
    for seq_len, n in sorted(result.concurrency_at.items()):
        per_seq_gb = (result.kv_per_token_bytes * seq_len + result.fixed_state_bytes) / 1e9
        label = f"{seq_len // 1024}K" if seq_len >= 1024 else str(seq_len)
        lines.append(f"  {label:>6} context, 1 seq  {per_seq_gb:6.1f} GB  -> fits {n} concurrent")
    lines.append("")
    zero_at = [s for s, n in sorted(result.concurrency_at.items()) if n == 0]
    if result.usable_kv_gb <= 0:
        lines.append("verdict: does not fit. weights + workspace already exceed the card's memory.")
    elif zero_at:
        labels = ", ".join(f"{s // 1024}K" if s >= 1024 else str(s) for s in zero_at)
        lines.append(
            f"verdict: deployable, but {labels} context can't hold even 1 sequence"
            " (needs shorter context, higher tp, or lower precision). bottleneck: KV capacity."
        )
    else:
        lines.append("verdict: deployable. bottleneck: KV capacity.")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(prog="predict")
    sub = parser.add_subparsers(dest="command", required=True)

    cap = sub.add_parser("capacity", help="capacity check: does it fit, how many concurrent sequences")
    cap.add_argument("--model", required=True, help="specs/models/<name>.yaml")
    cap.add_argument("--hw", required=True, help="specs/hardware/<name>.yaml")
    cap.add_argument("--dtype", default="fp8", choices=DTYPE_BYTES.keys(), help="weight precision")
    cap.add_argument("--kv-dtype", default="fp16", choices=DTYPE_BYTES.keys())
    cap.add_argument("--tp", type=int, default=1)
    cap.add_argument("--workspace-reserve-gb", type=float, default=4.0)

    args = parser.parse_args()

    if args.command == "capacity":
        model = ModelSpec.load(args.model)
        hw = HardwareSpec.load(args.hw)
        result = compute_capacity(
            model,
            hw,
            weight_dtype=args.dtype,
            kv_dtype=args.kv_dtype,
            tp=args.tp,
            workspace_reserve_gb=args.workspace_reserve_gb,
        )
        print(format_report(model, hw, result))


if __name__ == "__main__":
    main()
