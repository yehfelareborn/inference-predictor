"""M1's showcase chart: decode batch size vs (TPOT, throughput), with the
memory-bound -> compute-bound crossover marked. Answers "how much batching
headroom is left before this deployment stops being memory-bound."
"""
from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
import numpy as np

from core.memory import DTYPE_BYTES, HardwareSpec, ModelSpec
from core.roofline import compute_decode


def sweep(model, hw, weight_dtype, kv_dtype, comm_dtype, tp, context_length, mfu, mbu_weights, mbu_kv, batches):
    tpot_ms, throughput, statuses = [], [], []
    for b in batches:
        r = compute_decode(
            model, hw, int(b), context_length,
            weight_dtype=weight_dtype, kv_dtype=kv_dtype, comm_dtype=comm_dtype, tp=tp,
            mfu=mfu, mbu_weights=mbu_weights, mbu_kv=mbu_kv,
        )
        tpot_ms.append(r.predicted_time_s * 1e3)
        throughput.append(r.throughput_tokens_s)
        statuses.append(r.status)
    return np.array(tpot_ms), np.array(throughput), statuses


def plot(model_name, hw_name, weight_dtype, kv_dtype, comm_dtype, tp, context_length, mfu, mbu_weights, mbu_kv, out_path):
    model = ModelSpec.load(model_name)
    hw = HardwareSpec.load(hw_name)
    batches = np.array([2**k for k in range(0, 13)])  # 1 ~ 4096

    tpot_ms, throughput, statuses = sweep(
        model, hw, weight_dtype, kv_dtype, comm_dtype, tp, context_length, mfu, mbu_weights, mbu_kv, batches
    )

    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax2 = ax1.twinx()

    l1, = ax1.plot(batches, tpot_ms, marker="o", color="tab:red", label="TPOT (ms)")
    l2, = ax2.plot(batches, throughput, marker="s", color="tab:blue", label="throughput (tokens/s)")

    ax1.set_xscale("log", base=2)
    ax1.set_yscale("log")
    ax2.set_yscale("log")
    ax1.set_xlabel("batch size")
    ax1.set_ylabel("TPOT (ms)", color="tab:red")
    ax2.set_ylabel("throughput (tokens/s)", color="tab:blue")
    ax1.tick_params(axis="y", labelcolor="tab:red")
    ax2.tick_params(axis="y", labelcolor="tab:blue")

    # Mark where status flips from MEMORY-BOUND to COMPUTE-BOUND in the sweep.
    flip_idx = next((i for i in range(1, len(statuses)) if statuses[i] != statuses[i - 1]), None)
    if flip_idx is not None:
        flip_batch = batches[flip_idx]
        ax1.axvline(flip_batch, color="gray", ls=":", alpha=0.6)
        ax1.annotate(
            f"-> COMPUTE-BOUND at batch>={flip_batch}",
            xy=(flip_batch, tpot_ms[flip_idx]),
            xytext=(4, 0),
            textcoords="offset points",
            fontsize=8,
            color="gray",
        )
    else:
        note = f"stays {statuses[0]} across this whole batch range"
        ax1.annotate(
            note, xy=(0.98, 0.05), xycoords="axes fraction", fontsize=8, color="gray", ha="right", va="bottom"
        )

    ax1.grid(True, which="both", ls="--", alpha=0.3)
    ax1.set_title(
        f"Decode: batch vs TPOT/throughput — {model.name} on {hw.name}\n"
        f"context={context_length}, weight={weight_dtype}, kv={kv_dtype}, tp={tp}"
    )
    fig.legend(handles=[l1, l2], loc="upper left", bbox_to_anchor=(0.1, 0.88), fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"wrote {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--hw", required=True)
    parser.add_argument("--dtype", default="fp8", choices=DTYPE_BYTES.keys())
    parser.add_argument("--kv-dtype", default="fp16", choices=DTYPE_BYTES.keys())
    parser.add_argument("--comm-dtype", default="bf16", choices=DTYPE_BYTES.keys())
    parser.add_argument("--tp", type=int, default=1)
    parser.add_argument("--context-length", type=int, default=4096)
    parser.add_argument("--mfu", type=float, default=0.5)
    parser.add_argument("--mbu-weights", type=float, default=0.8)
    parser.add_argument("--mbu-kv", type=float, default=0.8)
    parser.add_argument("--out", default="roofline-decode.png")
    args = parser.parse_args()

    plot(
        args.model, args.hw, args.dtype, args.kv_dtype, args.comm_dtype, args.tp, args.context_length,
        args.mfu, args.mbu_weights, args.mbu_kv, args.out,
    )


if __name__ == "__main__":
    main()
