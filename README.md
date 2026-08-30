# Inference Predictor

> Know whether it's worth renting the GPU — before you rent it.

An analytical-first LLM inference performance predictor: model config + hardware spec + deployment parameters in, latency/capacity predictions and **bottleneck attribution** out — calibrated with a small amount of real measurement instead of requiring full profiling on the target hardware.

## Why this exists

Existing tools (Vidur, LLMServingSim, TokenSim) all need to profile on the actual target hardware before they can predict anything:

| | Vidur / LLMServingSim / TokenSim | This project |
|---|---|---|
| Method | profile on target hardware → ML interpolation → simulate | spec sheet → analytical derivation → light real-world calibration |
| Precondition | must have the actual card | just the spec sheet |
| Accuracy | high (Vidur reports <9% error) | lower, but honestly bounded |
| Can predict hardware you don't have | No | Yes — B200, Rubin, unreleased cards |
| Primary output | numbers + optimal config | **bottleneck attribution & decision boundaries** |

The trade: accuracy for coverage and explainability. This won't out-predict a real profiler on hardware you can already rent — it's for the hardware you can't (yet), and for understanding *why* a deployment is slow, not just how slow.

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
