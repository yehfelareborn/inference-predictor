"""M2's showcase chart: request rate (QPS) vs throughput/latency, with the
knee marked -- the point where added load stops turning into added
throughput and instead turns into queueing delay.
"""
from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
import numpy as np

from core.memory import HardwareSpec, ModelSpec
from core.scheduler import Simulator, WorkloadSpec, _percentile


def sweep(model, hw, workload_name, dtype, kv_dtype, tp, sim_seconds, seed, qps_values):
    throughput, p95_e2e, hit_safety_valve = [], [], []
    for qps in qps_values:
        w = WorkloadSpec.load(workload_name)
        w.qps = qps
        sim = Simulator(model, hw, w, weight_dtype=dtype, kv_dtype=kv_dtype, tp=tp, sim_seconds=sim_seconds, seed=seed)
        r = sim.run()
        throughput.append(r.total_output_tokens / r.sim_duration_s)
        p95_e2e.append(_percentile(r.e2e_samples, 0.95) * 1e3 if r.e2e_samples else float("nan"))
        hit_safety_valve.append(r.hit_safety_valve)
    return np.array(throughput), np.array(p95_e2e), np.array(hit_safety_valve)


def find_knee(qps_values, throughput, hit_safety_valve):
    """Knee = the QPS step with the largest drop in marginal throughput gain
    per unit QPS increase -- i.e. where the throughput curve bends over.
    Points that hit the safety valve are excluded first: their throughput is
    a potential undercount (queue never fully drained), and including them
    can make this heuristic latch onto a sampling artifact instead of a real
    saturation point -- exactly what happened testing this against GLM-4.5,
    which is why this filter exists."""
    qps_values = np.asarray(qps_values)
    valid = ~hit_safety_valve
    qps_clean, throughput_clean = qps_values[valid], throughput[valid]
    if len(qps_clean) < 3:
        return None
    gains = np.diff(throughput_clean) / np.diff(qps_clean)
    drop = -np.diff(gains)  # how much the marginal gain fell, step to step
    idx = int(np.argmax(drop)) + 1  # +1: drop[i] compares gains[i] and gains[i+1]
    return qps_clean[idx]


def plot(model_name, hw_name, workload_name, dtype, kv_dtype, tp, sim_seconds, seed, qps_values, out_path):
    model = ModelSpec.load(model_name)
    hw = HardwareSpec.load(hw_name)
    throughput, p95_e2e, hit_safety_valve = sweep(model, hw, workload_name, dtype, kv_dtype, tp, sim_seconds, seed, qps_values)
    knee = find_knee(qps_values, throughput, hit_safety_valve)

    if hit_safety_valve.any():
        bad_qps = ", ".join(f"{q:g}" for q, bad in zip(qps_values, hit_safety_valve) if bad)
        print(
            f"WARNING: queue never fully drained at QPS = {bad_qps} (marked hollow on the chart) -- "
            f"throughput there is a likely undercount, not a real dip. Rerun with a larger --sim-seconds "
            f"if you need those points to be trustworthy; knee-finding already ignores them."
        )

    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax2 = ax1.twinx()
    l1, = ax1.plot(qps_values, throughput, marker="o", color="tab:blue", label="throughput (tok/s)")
    l2, = ax2.plot(qps_values, p95_e2e, marker="s", color="tab:red", label="E2E p95 (ms)")
    ax2.set_yscale("log")

    handles = [l1, l2]
    if hit_safety_valve.any():
        qps_arr = np.asarray(qps_values)
        marker = ax1.scatter(qps_arr[hit_safety_valve], throughput[hit_safety_valve], facecolors="none",
                              edgecolors="tab:blue", s=120, linewidths=1.5, zorder=5,
                              label="undercount (queue didn't drain)")
        handles.append(marker)

    ax1.set_xlabel("request rate (QPS)")
    ax1.set_ylabel("throughput (tok/s)", color="tab:blue")
    ax2.set_ylabel("E2E latency p95 (ms)", color="tab:red")
    ax1.tick_params(axis="y", labelcolor="tab:blue")
    ax2.tick_params(axis="y", labelcolor="tab:red")

    if knee is not None:
        ax1.axvline(knee, color="gray", ls=":", alpha=0.6)
        ax1.annotate(f"knee ~= {knee:.1f} QPS", xy=(knee, 0), xycoords=("data", "axes fraction"),
                     xytext=(4, 6), textcoords="offset points", fontsize=8, color="gray")

    ax1.grid(True, ls="--", alpha=0.3)
    ax1.set_title(f"Request rate vs throughput/latency — {model.name} on {hw.name}\nworkload={workload_name}, tp={tp}")
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.12, 0.88), fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"wrote {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="llama3.1-8b")
    parser.add_argument("--hw", default="h100-sxm")
    parser.add_argument("--workload", default="steady-agentic")
    parser.add_argument("--dtype", default="fp8")
    parser.add_argument("--kv-dtype", default="fp16")
    parser.add_argument("--tp", type=int, default=1)
    parser.add_argument("--sim-seconds", type=float, default=30.0)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--qps", type=float, nargs="+", default=[2, 4, 8, 16, 24, 32, 48, 64])
    parser.add_argument("--out", default="scheduler-knee.png")
    args = parser.parse_args()

    plot(args.model, args.hw, args.workload, args.dtype, args.kv_dtype, args.tp, args.sim_seconds, args.seed, args.qps, args.out)


if __name__ == "__main__":
    main()
