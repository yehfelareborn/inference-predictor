"""M2 completion-definition check #1: reproduce the known qualitative finding
that uniform-length synthetic traffic underestimates continuous batching's
value, because it hides the heavy-tailed length distribution that causes
head-of-line blocking under static batching.

Runs both scheduler policies (continuous_batching, static_batching) against
both workloads (steady-agentic.yaml: heavy-tailed lognormal lengths,
uniform-baseline.yaml: same means, sigma=0) and compares continuous
batching's throughput improvement over static batching under each. The claim
holds if that improvement is larger under the heavy-tailed workload.
"""
from __future__ import annotations

import argparse
import copy

from core.memory import HardwareSpec, ModelSpec
from core.scheduler import Simulator, WorkloadSpec


def run_one(model, hw, workload_name, policy, dtype, kv_dtype, tp, sim_seconds, seed):
    w = WorkloadSpec.load(workload_name)
    w.scheduler = copy.deepcopy(w.scheduler)
    w.scheduler.policy = policy
    sim = Simulator(model, hw, w, weight_dtype=dtype, kv_dtype=kv_dtype, tp=tp, sim_seconds=sim_seconds, seed=seed)
    return sim.run()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="llama3.1-8b")
    parser.add_argument("--hw", default="h100-sxm")
    parser.add_argument("--dtype", default="fp8")
    parser.add_argument("--kv-dtype", default="fp16")
    parser.add_argument("--tp", type=int, default=1)
    parser.add_argument("--sim-seconds", type=float, default=30.0)
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()

    model = ModelSpec.load(args.model)
    hw = HardwareSpec.load(args.hw)

    rows = []
    for workload_name in ("uniform-baseline", "steady-agentic"):
        results = {}
        for policy in ("static_batching", "continuous_batching"):
            r = run_one(model, hw, workload_name, policy, args.dtype, args.kv_dtype, args.tp, args.sim_seconds, args.seed)
            results[policy] = r.total_output_tokens / r.sim_duration_s
        improvement = (results["continuous_batching"] / results["static_batching"] - 1) * 100
        rows.append((workload_name, results["static_batching"], results["continuous_batching"], improvement))

    print(f"model={model.name}  hw={hw.name}  dtype={args.dtype}  seed={args.seed}\n")
    print(f"{'workload':<18} {'static tok/s':>14} {'continuous tok/s':>18} {'improvement':>13}")
    for name, static_tps, cont_tps, imp in rows:
        print(f"{name:<18} {static_tps:>14.1f} {cont_tps:>18.1f} {imp:>12.1f}%")

    uniform_imp, agentic_imp = rows[0][3], rows[1][3]
    print()
    if agentic_imp > uniform_imp:
        print(
            f"CONFIRMED: heavy-tailed traffic shows a larger continuous-batching improvement "
            f"({agentic_imp:.1f}%) than uniform traffic ({uniform_imp:.1f}%) -- uniform synthetic "
            f"benchmarks understate continuous batching's value, as the roadmap's cited finding predicts."
        )
    else:
        print(
            f"NOT REPRODUCED this run: uniform improvement ({uniform_imp:.1f}%) >= "
            f"heavy-tailed improvement ({agentic_imp:.1f}%). Try a different seed or higher qps."
        )


if __name__ == "__main__":
    main()
