# Inference Predictor

> Know whether it's worth renting the GPU — before you rent it.

An analytical-first LLM inference performance predictor: model config + hardware spec + deployment parameters in, latency/capacity predictions and **bottleneck attribution** out — calibrated with a small amount of real measurement instead of requiring full profiling on the target hardware.

## Why this exists

Worth being upfront about existing work here rather than setting up a strawman. Checked each of these against its own paper/repo rather than assuming:

| | [Vidur](https://arxiv.org/abs/2405.05465) | [LLMServingSim](https://arxiv.org/abs/2408.05499) | [TokenSim](https://arxiv.org/abs/2503.08415) | This project |
|---|---|---|---|---|
| Method | profile real HW → random-forest interpolation → simulate | profile real HW (vLLM-based layerwise profiler) → co-simulate compute + network | pluggable analytical compute simulators (e.g. [GenZ](https://arxiv.org/abs/2406.01698), [LLMCompass](https://ieeexplore.ieee.org/document/10609604/)) → event-driven simulation; can also fold in real profiling | spec sheet → closed-form analytical formulas → light real-world calibration |
| Needs the actual card | Yes — interpolates within a GPU it has profiled, not shown to extrapolate to unprofiled hardware | Yes — takes real per-hardware profiling data as input | No, in its analytical mode — built to evaluate hardware it doesn't physically have | No |
| Reported accuracy | <9% error (LLaMA2-7B/70B, InternLM-20B, Qwen-72B on A100/H100) | <14.7% error (paper); ~1% in a narrower single-GPU benchmark (RTX 4090, per its README) | ~0.1–0.6% vs. real vLLM (LLaMA2-7B on A100, ShareGPT traffic) | not yet calibrated against real hardware (M3, in progress) |
| Primary output | latency/throughput numbers + optimal deployment config | latency/throughput numbers across heterogeneous HW pools | latency/throughput numbers, HW/SW design-space exploration | **bottleneck attribution & decision boundaries** |

So the honest pitch isn't "nobody else can predict hardware you don't have" — TokenSim, and the analytical compute simulators it wraps (GenZ: 5.82% geomean error, LLMCompass: ~4.1% error for LLM inference), already do that, and do it well. What this project actually bets on is being small enough to read end to end in one sitting — a handful of closed-form formulas over a yaml spec, no compute-simulator backend to wire up — and treating *why a deployment is slow*, not just how slow, as the primary output rather than something you'd have to dig out of a trace yourself.

## Status

| Milestone | What it answers | Status |
|---|---|---|
| **M0** — capacity | Does this model fit on this card? How many concurrent requests? | ✅ done |
| **M1** — roofline + attribution | How fast is one prefill/decode step, and where does the time go? | ✅ done |
| **M2** — scheduling simulator | Under real traffic, what do p50/p95/p99 actually look like? | ✅ done |
| **M3** — calibration + error table | How wrong is the analytical model, and why? | 🚧 in progress |
| **M4** — decision-boundary scan + findings | What concrete deployment decisions can this answer? | not started |

M0–M2 run entirely locally at $0 GPU cost — every number is analytical. M3 is the only stage that costs money: it fits coefficients (MFU, MBU, per-iteration overhead, TP communication efficiency) against real measurements on rented hardware.

## Quick start

```bash
git clone https://github.com/yehfelareborn/inference-predictor
cd inference-predictor

# Does Qwen3-32B fit on an H100 at fp8, and how many concurrent 32K-context requests?
python3 -m core.memory capacity --model qwen3-32b --hw h100-sxm --dtype fp8 --kv-dtype fp16

# How fast is decode at batch=32, and what's the bottleneck?
python3 -m core.roofline decode --model qwen3-32b --hw h100-sxm --batch 32 --context-length 4096 --dtype fp8 --kv-dtype fp16

# Under real traffic (Poisson arrivals, heavy-tailed lengths), what's p95 latency?
python3 -m core.scheduler --model qwen3-32b --hw h100-sxm --workload steady-agentic --dtype fp8 --kv-dtype fp16
```

Full usage, every flag, and how to read the output: [`USAGE.md`](USAGE.md).

## What's actually been validated

- **The KV-cache formula matches vLLM's own internal accounting exactly.** Ran a real vLLM instance locally (Qwen2.5-0.5B on a consumer RTX 3070), read vLLM's own `cache_config`, and compared: 12,288 bytes/token predicted vs. 12,288 bytes/token actual. See `calibrate/scripts/m0_kv_check.py`.
- **Tensor-parallel scaling behaves correctly.** The crossover batch size (where decode flips from memory-bound to compute-bound) is mathematically invariant to `tp` — checked numerically across `tp=1,2,4`, not just asserted.
- **Continuous batching's value is reproduced quantitatively.** Heavy-tailed synthetic traffic shows continuous batching beating static batching by ~250–290%, vs. only ~120–155% under uniform-length traffic — confirming the known result that uniform benchmarks understate continuous batching's benefit, consistent across 4 random seeds.

## Repo structure

```
core/            M0-M2 implementation (memory.py, roofline.py, attribution.py, scheduler.py, plus plotting)
specs/
  models/        Model architecture specs — dense, MoE, and hybrid Mamba-2 / linear-attention
  hardware/      GPU specs, including unreleased/unrentable cards (B200, Rubin)
  workloads/     M2 traffic-shape configs (arrival process, length distributions, scheduler policy)
  engines/       M3 calibration coefficients (empty until M3 lands)
findings/        The actual point of this project — see below
calibrate/       Local validation scripts and their raw output
web/             Interactive capacity-explorer page (static, no build step — just serve the directory)
```

## Findings

The tool is instrumentation; **the findings are the product**. See [`findings/`](findings/) — currently one entry, on why MoE architectures are more prone to cross-hardware output divergence in agentic workflows than dense models are. More land as M3/M4 produce them.

## Known limitations

- Hybrid architectures (Mamba-2, Gated DeltaNet) model recurrent-state memory with a documented approximation — see the warning comments in `specs/models/nemotron-nano-9b.yaml` / `qwen3.8-27b.yaml`.
- MoE FLOPs use `active_params ≈ params × n_active/n_experts`, which slightly overestimates active compute — attention/embedding params aren't duplicated per-expert, and the schema doesn't separate them out from expert-FFN params.
- MFU/MBU are user-supplied assumptions (defaults 0.5/0.8) until M3 replaces them with measured coefficients.
- No real-hardware calibration yet beyond the M0 vLLM cross-check above — everything else is analytical, not measured against production traffic.
