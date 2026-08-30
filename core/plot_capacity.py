"""M0's showcase chart: concurrency vs context length.

One plot answers "at different context lengths, how many concurrent
requests can this deployment hold."
"""
from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
import numpy as np

from core.memory import DTYPE_BYTES, HardwareSpec, ModelSpec, kv_per_token_bytes, max_concurrency, weights_gb


def concurrency_curve(model, hw, weight_dtype, kv_dtype, tp, workspace_reserve_gb, seq_lens):
    w_gb = weights_gb(model, weight_dtype, tp)
    kv_bpt = kv_per_token_bytes(model, kv_dtype, tp)
    fixed_state = model.fixed_state_bytes(tp, kv_dtype)
    usable_gb = hw.memory_gb - w_gb - workspace_reserve_gb
    concurrency = [max_concurrency(usable_gb, kv_bpt, s, fixed_state) for s in seq_lens]
    return concurrency, w_gb, usable_gb


def plot(model_names, hw_name, weight_dtype, kv_dtype, tp, workspace_reserve_gb, out_path):
    hw = HardwareSpec.load(hw_name)
    seq_lens = np.array([2**k for k in range(10, 18)])  # 1K ~ 128K

    fig, ax = plt.subplots(figsize=(8, 5))
    for model_name in model_names:
        model = ModelSpec.load(model_name)
        concurrency, w_gb, usable_gb = concurrency_curve(
            model, hw, weight_dtype, kv_dtype, tp, workspace_reserve_gb, seq_lens
        )
        concurrency = np.array(concurrency)
        label = f"{model.name} (weights {w_gb:.1f}GB, KV budget {usable_gb:.1f}GB)"
        # A log-scale y-axis can't render 0. Letting the line silently vanish
        # reads as missing data rather than "genuinely doesn't fit here", so
        # swap 0 for NaN to break the line and mark the cutoff explicitly.
        plotted = np.where(concurrency == 0, np.nan, concurrency)
        (line,) = ax.plot(seq_lens, plotted, marker="o", label=label)
        zero_mask = concurrency == 0
        if zero_mask.any():
            cutoff = seq_lens[zero_mask][0]
            cutoff_label = f"{cutoff // 1024}K" if cutoff >= 1024 else str(cutoff)
            ax.axvline(cutoff, color=line.get_color(), ls=":", alpha=0.5)
            ax.annotate(
                f"{model.name}: 0 capacity ≥ {cutoff_label}",
                xy=(cutoff, 1),
                xytext=(4, 4),
                textcoords="offset points",
                color=line.get_color(),
                fontsize=8,
            )

    ax.set_xscale("log", base=2)
    ax.set_yscale("log", base=2)
    ax.set_xlabel("context length (tokens)")
    ax.set_ylabel("max concurrent requests")
    ax.set_title(f"Concurrency vs context length — {hw.name}, weight={weight_dtype}, kv={kv_dtype}, tp={tp}")
    ax.grid(True, which="both", ls="--", alpha=0.4)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"wrote {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", action="append", required=True, help="repeatable for multiple models")
    parser.add_argument("--hw", required=True)
    parser.add_argument("--dtype", default="fp8", choices=DTYPE_BYTES.keys())
    parser.add_argument("--kv-dtype", default="fp16", choices=DTYPE_BYTES.keys())
    parser.add_argument("--tp", type=int, default=1)
    parser.add_argument("--workspace-reserve-gb", type=float, default=4.0)
    parser.add_argument("--out", default="concurrency.png")
    args = parser.parse_args()

    plot(args.model, args.hw, args.dtype, args.kv_dtype, args.tp, args.workspace_reserve_gb, args.out)


if __name__ == "__main__":
    main()
