# Usage: M0–M2

Three milestones, each answering a different question, each building on the last:

- **M0** (`core/memory.py`) — does this model fit on this card, and how many concurrent requests can it hold?
- **M1** (`core/roofline.py` + `core/attribution.py`) — how fast is a single prefill/decode step, and where does the time actually go?
- **M2** (`core/scheduler.py`) — under real traffic (arrivals, batching, preemption), what do p50/p95/p99 latency and throughput look like?

All three read model/hardware specs from `specs/models/*.yaml` and `specs/hardware/*.yaml` — pass the filename without `.yaml` as `--model`/`--hw`. Run everything from the repo root with `python3 -m core.<module>`.

---

## M0 — capacity

```bash
python3 -m core.memory capacity --model qwen3-32b --hw h100-sxm --dtype fp8 --kv-dtype fp16 --tp 1
```

- `--dtype` is weight precision, `--kv-dtype` is KV cache precision, `--tp` is tensor-parallel degree.
- `--workspace-reserve-gb` (default 4.0) is memory set aside for activations/workspace, subtracted before computing usable KV.

Output is a memory breakdown (weights / workspace / usable KV) plus concurrency at 8K/32K/128K context, ending in a verdict: `does not fit`, `deployable, but <ctx> can't hold even 1 sequence`, or `deployable`.

**Chart** (concurrency vs context length, multi-model overlay):
```bash
python3 -m core.plot_capacity --model qwen3-32b --model llama3.1-8b --hw h100-sxm \
  --dtype fp8 --kv-dtype fp16 --tp 1 --out concurrency.png
```
`--model` is repeatable. Zero-capacity points break the line and get an explicit cutoff annotation rather than silently vanishing.

**Hybrid architectures** (Mamba-2, Gated DeltaNet — see `specs/models/nemotron-nano-9b.yaml`, `qwen3.8-27b.yaml`): capacity output shows an extra `+ fixed state/seq` line. This is real per-sequence memory that doesn't grow with context length, on top of the normal KV term — it can dominate at short context, which is why these models' concurrency curves flatten out earlier than a standard transformer's would.

---

## M1 — roofline latency + attribution

```bash
python3 -m core.roofline prefill --model qwen3-32b --hw h100-sxm --tokens 4096 --dtype fp8 --tp 1
python3 -m core.roofline decode  --model qwen3-32b --hw h100-sxm --batch 32 --context-length 4096 \
  --dtype fp8 --kv-dtype fp16 --tp 1 --price-usd-hr 2.5
```

- `prefill --tokens N` is prompt length; `decode --batch N --context-length N` is concurrent sequences and current context depth.
- `--mfu`/`--mbu` (defaults 0.5/0.8) are assumed efficiency factors — analytical placeholders, not measured (that's M3's job).
- `--price-usd-hr` on `decode` prints `$/1M output tokens`.

Every call prints a **bottleneck attribution** block after the main numbers: time-share breakdown (linear compute / attention compute / TP comm for prefill; HBM weight read / KV read / TP comm for decode) plus a one-line "most effective lever" recommendation. This is the point of M1 — a number alone doesn't tell you what to change.

Switching hardware to answer "what if we used L40S instead" needs no L40S — just change `--hw`. `crossover_batch` (shown in the decode report) tells you what batch size would flip decode from memory-bound to compute-bound at the given context length — usually `None` ("stays memory-bound at any batch size") for realistic models on modern hardware, which is itself the expected, correct answer, not a bug.

**Chart** (decode batch size vs TPOT/throughput, single model+hardware):
```bash
python3 -m core.plot_roofline --model qwen3-32b --hw h100-sxm --dtype fp8 --kv-dtype fp16 \
  --context-length 4096 --tp 1 --out roofline-decode.png
```

**Hybrid architectures**: decode report gets an extra `state read/step` line, and attribution gets a "recurrent state read" bucket. Mamba-2's state is always fp32 regardless of `--kv-dtype` (a numerical-stability constant); Gated DeltaNet's state precision is approximated as following `--kv-dtype`. Switching `--kv-dtype` will visibly change one but not the other — that's correct, not a bug.

---

## M2 — scheduling simulator

Traffic isn't a CLI flag — it's a yaml file in `specs/workloads/`, since there are several nested knobs:

```yaml
arrival: poisson
qps: 8
input_len:  {dist: lognormal, mean: 2048, sigma: 0.8}   # sigma: 0 = uniform length, no tail
output_len: {dist: lognormal, mean: 512,  sigma: 1.2}
prefix_share: 0.35      # probability a request is a full prefix-cache hit (simplified model)

scheduler:
  policy: continuous_batching   # or static_batching (pre-continuous-batching baseline, for comparison)
  chunked_prefill: true
  max_num_batched_tokens: 8192
  max_num_seqs: 256
  preemption: recompute          # or swap
```
Two ready-made examples: `steady-agentic` (heavy-tailed lengths) and `uniform-baseline` (same means, no tail).

```bash
python3 -m core.scheduler --model llama3.1-8b --hw h100-sxm --workload steady-agentic \
  --dtype fp8 --kv-dtype fp16 --tp 1 --sim-seconds 30 --seed 1
```

`--sim-seconds` is how long to generate arrivals for; the sim then drains whatever's left in-flight before stopping. `--seed` fixes the RNG for reproducibility.

Report fields:
- `TTFT`/`TPOT`/`E2E` at p50/p95/p99 — the whole point of M2 over M0/M1, which only give averages.
- `queueing share of E2E` — how much of end-to-end latency is queueing, not computing. High (>50%) means the bottleneck is scheduling, not hardware.
- `preemption rate` — fraction of requests evicted mid-flight at least once. Near 0% is healthy; climbing into double digits means the deployment is thrashing.
- `KV utilization (p95)` — near 100% is what triggers preemption.
- **`WARNING: queue never fully drained...`** — appears when the run hit its internal safety valve (3x `--sim-seconds`) before the backlog cleared. When this fires, `throughput` is a likely **undercount** (in-flight work isn't credited yet) — it does not necessarily mean the system is slow, just that `--sim-seconds` was too short for this model/hardware/tp/capacity combination to fully drain in the tested window. Rerun with a larger `--sim-seconds` before trusting the number. Bigger models / smaller `tp` / tighter KV budgets need more simulated seconds to drain a given QPS.

**Compare continuous vs static batching** (reproduces the roadmap's cited finding that uniform synthetic traffic underestimates continuous batching's value):
```bash
python3 -m core.validate_scheduler --model llama3.1-8b --hw h100-sxm --dtype fp8 --kv-dtype fp16 --sim-seconds 30 --seed 1
```

**Find the safe QPS ceiling** (throughput/latency vs request rate, with the knee marked):
```bash
python3 -m core.plot_scheduler --model llama3.1-8b --hw h100-sxm --workload steady-agentic \
  --dtype fp8 --kv-dtype fp16 --tp 1 --sim-seconds 60 --seed 1 \
  --qps 2 4 8 16 24 32 48 64 --out knee.png
```
Points that hit the safety valve are drawn as hollow circles and excluded from knee-finding automatically — the console also lists which QPS values were affected. If most/all points come back hollow, `--sim-seconds` is too short for this configuration; raise it and rerun. `find_knee()` looks for the single largest drop in marginal throughput gain, so it can occasionally misfire on a noisy middle point — sanity-check that the curve actually flattens (and KV utilization climbs toward 100%) at or after the marked knee before trusting it, especially with fewer than ~5 clean data points.

Large/MoE models on smaller cards often need `--tp` raised before M2 will even construct (`weights_gb(...) + workspace` must fit under `hw.memory_gb` — same "does not fit" check as M0).

**Hybrid architectures**: report prints an extra line up front (`hybrid arch: N blocks/seq reserved...`) — every admitted sequence pays this fixed block cost on top of its own growing KV blocks, and every iteration pays a fixed state-read bandwidth cost per active sequence, regardless of chunk size.

---

## Common gotchas

- **A100 has no native FP8** (`peak_flops.fp8: null` in its spec) — use `--dtype bf16` there. Any dtype a card doesn't support raises a clear error rather than a wrong number.
- **`--tp` needed for big/MoE models on smaller cards** — e.g. GLM-4.5 (355B) needs `--tp 8` on an 80GB card. All three milestones use the same fits-or-doesn't check, so the error is consistent everywhere.
- **M2's `--sim-seconds` should scale with how slow the deployment is** — a small dense model on a fast card drains fast; a large/MoE model on a tightly-tp'd card can need `--sim-seconds` several times larger at the same QPS before the safety-valve warning stops firing.
- **Hybrid-architecture models** (currently `nemotron-nano-9b`, `qwen3.8-27b`) are real, sourced models — not placeholders — but their `layers`/`n_kv_heads`/`head_dim` represent only the true full-attention layers; the rest of their memory/bandwidth cost comes from `state_layers` in the yaml, which all three milestones already account for.
