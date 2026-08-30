"""M2: iteration-based scheduling simulator. M0/M1 only give averages; this
adds queueing, batching, and preemption dynamics to get p50/p95/p99.

Time advances one *iteration* (engine step) at a time, not via a
continuous-time event queue -- this matches how a real inference engine
(e.g. vLLM's scheduler loop) actually works: every step, decide what to run,
run it, advance the clock by however long that step took, repeat. Per-
iteration cost reuses core.roofline's compute/memory roofline model, applied
to whatever mix of decode tokens + prefill-chunk tokens got batched together
that step.

Known simplifications, documented rather than silently baked in (consistent
with core/roofline.py's own documented simplifications):
  - No TP communication cost -- M2 is about scheduling dynamics for a fixed,
    already-chosen deployment, not a place to re-derive M1's TP tradeoffs.
  - KV cache write bandwidth isn't modeled, same choice core/roofline.py
    made for prefill.
  - Prefix caching is the roadmap's own stated simplification: a per-request
    coin flip (probability = prefix_share) for a full cache hit, not a
    fractional/partial-overlap model.
  - Swap-preemption's PCIe transfer-back cost delays only the swapped
    request's own resumption, not the whole batch's iteration time (models
    swap I/O as overlappable with other requests' compute, which is roughly
    how real async CPU<->GPU transfers behave).
"""
from __future__ import annotations

import argparse
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from core.memory import HardwareSpec, ModelSpec, kv_per_token_bytes, usable_kv_gb, weights_gb
from core.memory import DTYPE_BYTES, _load_yaml
from core.roofline import _require_peak_flops

SPECS_DIR = Path(__file__).resolve().parent.parent / "specs"
BLOCK_SIZE = 16  # tokens per KV block, vLLM's own default


@dataclass
class LenDist:
    dist: str
    mean: float
    sigma: float

    def sample(self, rng: random.Random) -> int:
        if self.sigma <= 0:
            return max(1, round(self.mean))
        # Want E[X] = mean for the *lognormal*, not for the underlying
        # normal -- lognormal mean is exp(mu + sigma^2/2), so back out mu.
        mu = math.log(self.mean) - self.sigma**2 / 2
        return max(1, round(rng.lognormvariate(mu, self.sigma)))


@dataclass
class SchedulerConfig:
    policy: str  # continuous_batching or static_batching
    chunked_prefill: bool
    max_num_batched_tokens: int
    max_num_seqs: int
    preemption: str  # recompute or swap


@dataclass
class WorkloadSpec:
    name: str
    arrival: str
    qps: float
    input_len: LenDist
    output_len: LenDist
    prefix_share: float
    scheduler: SchedulerConfig

    @classmethod
    def load(cls, name: str) -> "WorkloadSpec":
        data = _load_yaml(SPECS_DIR / "workloads" / f"{name}.yaml")
        data["input_len"] = LenDist(**data["input_len"])
        data["output_len"] = LenDist(**data["output_len"])
        data["scheduler"] = SchedulerConfig(**data["scheduler"])
        return cls(**data)


# ---------------------------------------------------------------------------
# Requests and KV block allocation
# ---------------------------------------------------------------------------


@dataclass
class Request:
    id: int
    arrival_time: float
    input_len: int
    output_len: int
    prefix_hit: bool

    state: str = "WAITING"  # WAITING, PREFILLING, DECODING, DONE
    tokens_processed: int = 0  # prefill progress
    output_tokens_generated: int = 0
    blocks_held: int = 0

    first_token_time: Optional[float] = None
    completion_time: Optional[float] = None
    queueing_time: float = 0.0
    last_state_change: float = 0.0
    preemptions: int = 0
    swap_available_at: Optional[float] = None  # None = not swapped out

    def context_len(self) -> int:
        """Tokens whose KV must be resident: prompt + generated so far."""
        return self.tokens_processed + self.output_tokens_generated

    def prefill_remaining(self) -> int:
        return self.input_len - self.tokens_processed


class KVBlockAllocator:
    def __init__(self, total_blocks: int, block_size: int = BLOCK_SIZE):
        self.total_blocks = total_blocks
        self.block_size = block_size
        self.free_blocks = total_blocks

    def blocks_needed(self, num_tokens: int) -> int:
        return -(-num_tokens // self.block_size)  # ceil div

    def try_allocate(self, num_blocks: int) -> bool:
        if num_blocks <= self.free_blocks:
            self.free_blocks -= num_blocks
            return True
        return False

    def free(self, num_blocks: int) -> None:
        self.free_blocks += num_blocks
        assert self.free_blocks <= self.total_blocks

    def utilization(self) -> float:
        return 1 - self.free_blocks / self.total_blocks


# ---------------------------------------------------------------------------
# Simulator
# ---------------------------------------------------------------------------


@dataclass
class SimResult:
    ttft_samples: List[float]
    tpot_samples: List[float]
    e2e_samples: List[float]
    queueing_ratio_samples: List[float]
    kv_util_samples: List[float]
    total_output_tokens: int
    sim_duration_s: float
    num_completed: int
    num_preemption_events: int
    num_requests_ever_preempted: int
    num_rejected: int
    state_blocks_per_seq: int  # 0 for standard transformers, >0 for hybrid archs
    hit_safety_valve: bool  # True if the run was cut off before the queue fully drained -- throughput may be an undercount


def _percentile(samples: List[float], p: float) -> float:
    if not samples:
        return float("nan")
    s = sorted(samples)
    k = (len(s) - 1) * p
    lo, hi = math.floor(k), math.ceil(k)
    if lo == hi:
        return s[lo]
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


class Simulator:
    def __init__(
        self,
        model: ModelSpec,
        hw: HardwareSpec,
        workload: WorkloadSpec,
        weight_dtype: str = "fp8",
        kv_dtype: str = "fp16",
        tp: int = 1,
        mfu: float = 0.5,
        mbu: float = 0.8,
        workspace_reserve_gb: float = 4.0,
        seed: int = 0,
        sim_seconds: float = 30.0,
    ):
        self.model = model
        self.hw = hw
        self.workload = workload
        self.sched = workload.scheduler
        self.weight_dtype = weight_dtype
        self.kv_dtype = kv_dtype
        self.tp = tp
        self.mfu = mfu
        self.mbu = mbu
        self.sim_seconds = sim_seconds
        self.rng = random.Random(seed)

        self.peak_flops = _require_peak_flops(hw, weight_dtype)
        self.w_bytes = weights_gb(model, weight_dtype, tp) * 1e9
        self.kv_bpt = kv_per_token_bytes(model, kv_dtype, tp)
        usable_gb = usable_kv_gb(hw, weights_gb(model, weight_dtype, tp), workspace_reserve_gb)
        if usable_gb <= 0:
            raise ValueError(
                f"{model.name} weights ({weights_gb(model, weight_dtype, tp):.1f}GB) + workspace "
                f"({workspace_reserve_gb:.1f}GB) already exceed {hw.name}'s {hw.memory_gb}GB -- this is "
                f"an M0 'does not fit' case (see core.memory), not something M2 can schedule around. "
                f"Raise tp, or pick a smaller model/dtype/card."
            )
        bytes_per_block = BLOCK_SIZE * self.kv_bpt
        self.allocator = KVBlockAllocator(int(usable_gb * 1e9 // bytes_per_block))

        # Hybrid architectures (Mamba-2, Gated DeltaNet, ...) carry a fixed
        # per-sequence state on top of the growing attention KV cache -- see
        # ModelSpec.fixed_state_bytes. Charged as a flat block reservation
        # every admitted sequence pays up front (on top of its own growing
        # KV blocks), reusing the same block-granularity accounting M2
        # already uses for KV, rather than a parallel byte-exact tracker.
        # 0 for every model without state_layers -- no behavior change there.
        self.state_bytes_per_seq = model.fixed_state_bytes(tp, kv_dtype)
        self.state_blocks = math.ceil(self.state_bytes_per_seq / bytes_per_block) if self.state_bytes_per_seq > 0 else 0

        self.clock = 0.0
        self.next_id = 0
        self.next_arrival = self._sample_interarrival()
        self.waiting: List[Request] = []
        self.running: List[Request] = []
        self.done: List[Request] = []
        self.rejected: List[Request] = []

        self.tpot_samples: List[float] = []
        self.kv_util_samples: List[float] = []
        self.num_preemption_events = 0
        self.preempted_ids = set()

    def _sample_interarrival(self) -> float:
        return self.rng.expovariate(self.workload.qps)

    def _spawn_arrivals_up_to(self, t: float) -> None:
        while self.next_arrival <= t:
            r = Request(
                id=self.next_id,
                arrival_time=self.next_arrival,
                input_len=self.workload.input_len.sample(self.rng),
                output_len=self.workload.output_len.sample(self.rng),
                prefix_hit=self.rng.random() < self.workload.prefix_share,
                last_state_change=self.next_arrival,
            )
            self.next_id += 1
            self.waiting.append(r)
            self.next_arrival += self._sample_interarrival()

    def _iteration_cost_s(self, batch: List[tuple]) -> float:
        """batch: list of (request, chunk_tokens). chunk_tokens==1 with the
        request already in DECODING means a decode step; otherwise it's a
        prefill chunk."""
        total_tokens = sum(c for _, c in batch)
        linear_flops = 2 * self.model.active_params() * total_tokens
        attn_flops = 0.0
        kv_read_bytes = 0.0
        # Hybrid archs' recurrent-state layers are read+updated once per
        # active sequence per iteration regardless of chunk size -- that's
        # their whole point (O(1) state access vs attention's O(context)),
        # so this is charged per sequence in the batch, not per token.
        state_read_bytes = len(batch) * self.state_bytes_per_seq
        for r, c in batch:
            if r.state == "DECODING":
                attn_flops += 4 * r.context_len() * self.model.hidden_size
                kv_read_bytes += r.context_len() * self.kv_bpt
            else:
                already = r.tokens_processed
                attn_flops += 4 * c * (already + c) * self.model.hidden_size
        compute_time = (linear_flops + attn_flops) / self.tp / (self.peak_flops * self.mfu)
        memory_time = (self.w_bytes + kv_read_bytes + state_read_bytes) / (self.hw.memory_bandwidth_gbs * 1e9 * self.mbu)
        return max(compute_time, memory_time)

    def _try_admit(self, r: Request) -> bool:
        """Attempt to allocate blocks and start (or resume) a waiting request.
        Returns False if there isn't room -- caller decides whether to
        preempt something to make room."""
        blocks_needed = self._needed_blocks(r) - r.blocks_held
        if blocks_needed > 0 and not self.allocator.try_allocate(blocks_needed):
            return False
        r.blocks_held += max(blocks_needed, 0)
        if r.prefix_hit and r.tokens_processed == 0:
            r.tokens_processed = r.input_len  # simplified: full cache hit skips prefill entirely
        r.state = "DECODING" if r.tokens_processed >= r.input_len else "PREFILLING"
        self.running.append(r)
        return True

    def _preempt_one(self, now: float) -> bool:
        """LIFO victim selection (vLLM's own default): evict the most
        recently admitted running request."""
        if not self.running:
            return False
        victim = self.running.pop()
        self.allocator.free(victim.blocks_held)
        self.num_preemption_events += 1
        self.preempted_ids.add(victim.id)
        victim.preemptions += 1
        victim.blocks_held = 0
        if self.sched.preemption == "recompute":
            # Losing the KV cache means re-processing prompt + everything
            # generated so far as a fresh prefill when it resumes.
            victim.input_len = victim.input_len + victim.output_tokens_generated
            victim.output_tokens_generated = 0
            victim.tokens_processed = 0
            victim.state = "WAITING"
        else:  # swap
            swap_bytes = victim.context_len() * self.kv_bpt
            bw = self.hw.pcie_gbs or self.hw.memory_bandwidth_gbs
            victim.swap_available_at = now + swap_bytes / (bw * 1e9)
            victim.state = "WAITING"
        victim.last_state_change = now
        # Insert right *after* whatever the caller is currently trying to
        # admit (always self.waiting[0] at call time), not at index 0 --
        # otherwise a freshly-preempted victim jumps the queue ahead of the
        # request the preemption was done to make room for, and (worse, under
        # recompute) a victim whose own re-admission need just grew can end
        # up perpetually cutting in front of itself.
        self.waiting.insert(1, victim)
        return True

    def _needed_blocks(self, r: Request) -> int:
        tokens = r.context_len() if r.context_len() > 0 else r.input_len
        return self.allocator.blocks_needed(max(tokens, 1)) + self.state_blocks

    def _schedule_iteration(self, now: float) -> List[tuple]:
        budget = self.sched.max_num_batched_tokens
        batch: List[tuple] = []

        decoding = [r for r in self.running if r.state == "DECODING"]
        prefilling = [r for r in self.running if r.state == "PREFILLING"]

        for r in decoding:
            if len(batch) >= self.sched.max_num_seqs or budget <= 0:
                break
            batch.append((r, 1))
            budget -= 1

        for r in prefilling:
            if budget <= 0:
                break
            chunk = min(budget, r.prefill_remaining()) if self.sched.chunked_prefill else r.prefill_remaining()
            if not self.sched.chunked_prefill and chunk > budget:
                continue  # unchunked: must fit whole prefill in one shot, else wait
            batch.append((r, chunk))
            budget -= chunk

        admitted_this_step = 0
        preemptions_this_step = 0
        # Hard cap on preemptions per iteration. Without this, two similarly-
        # sized requests can bounce in and out of `running` forever: admitting
        # r evicts victim, victim is requeued right behind r, admitting
        # victim next evicts r (LIFO picks whatever was *just* admitted), and
        # so on -- neither ever reaches `batch`/`budget`, so the loop's own
        # exit conditions never trigger and the other waiting requests never
        # get a turn. Real schedulers cap per-tick scheduling work too; once
        # hit, stop for this iteration and let the next one retry.
        max_preemptions_per_iter = 2 * self.sched.max_num_seqs
        while self.waiting and budget > 0 and len(batch) < self.sched.max_num_seqs:
            r = self.waiting[0]
            if r.swap_available_at is not None and now < r.swap_available_at:
                break  # head-of-line: still mid-transfer: real schedulers would look further, kept simple
            if self.sched.policy == "static_batching" and self.running and admitted_this_step == 0 and any(
                x.state != "DONE" for x in self.running
            ):
                break  # no new admissions until the whole running batch drains
            if preemptions_this_step >= max_preemptions_per_iter:
                break
            if self._needed_blocks(r) > self.allocator.total_blocks:
                # Can never fit even with the entire running batch evicted --
                # e.g. a lognormal-tail input_len sample, or (under recompute
                # preemption) a request whose re-admission need grew past
                # capacity across repeated eviction cycles. Drop it rather
                # than loop preempting everything forever for a lost cause.
                self.waiting.pop(0)
                r.state = "DONE"
                r.completion_time = None
                self.rejected.append(r)
                continue
            first_chunk = r.prefill_remaining() if r.tokens_processed or r.prefix_hit else r.input_len
            if not self.sched.chunked_prefill and first_chunk > budget:
                break
            first_chunk = min(first_chunk, budget) if self.sched.chunked_prefill else first_chunk
            r.queueing_time += now - r.last_state_change
            if not self._try_admit(r):
                if not self._preempt_one(now):
                    break
                preemptions_this_step += 1
                continue
            self.waiting.pop(0)
            r.last_state_change = now
            admitted_this_step += 1
            if r.state == "DECODING":
                continue  # cache hit: ready to decode next iteration, not this one
            batch.append((r, first_chunk))
            budget -= first_chunk

        return batch

    def _apply_batch(self, batch: List[tuple], now: float, iter_time: float) -> None:
        for r, c in batch:
            if r.state == "DECODING":
                r.output_tokens_generated += 1
                if r.first_token_time is None:
                    r.first_token_time = now + iter_time
                self.tpot_samples.append(iter_time)
                if r.output_tokens_generated >= r.output_len:
                    r.state = "DONE"
                    r.completion_time = now + iter_time
                    self.allocator.free(r.blocks_held)
                    r.blocks_held = 0
            else:
                r.tokens_processed += c
                if r.tokens_processed >= r.input_len:
                    r.state = "DECODING"
                    if r.first_token_time is None:
                        # first decode token actually happens next iteration;
                        # approximate TTFT as end of the last prefill chunk.
                        pass
        still_running = []
        for r in self.running:
            (self.done if r.state == "DONE" else still_running).append(r)
        self.running = still_running

    def run(self) -> SimResult:
        max_iters = 2_000_000
        hit_safety_valve = False
        for _ in range(max_iters):
            if self.clock >= self.sim_seconds and not self.running and not self.waiting:
                break
            if self.clock >= self.sim_seconds * 3 and (self.running or self.waiting):
                # System hasn't drained by 3x sim_seconds -- either genuinely
                # overloaded, or sim_seconds was just too short for this
                # model/hardware/capacity combination (both look the same
                # from here: the queue never emptied). Stop anyway rather
                # than run forever, but flag it -- total_output_tokens is
                # counted only from requests that actually finished, so
                # throughput = total_output_tokens / sim_duration_s can look
                # artificially low when this fires, not because the system
                # is slow, but because in-flight work isn't credited yet.
                hit_safety_valve = True
                break
            self._spawn_arrivals_up_to(min(self.clock, self.sim_seconds))
            batch = self._schedule_iteration(self.clock)
            self.kv_util_samples.append(self.allocator.utilization())
            if not batch:
                if not self.waiting and not self.running:
                    self.clock = self.next_arrival
                    continue
                iter_time = 1e-3  # idle tick: waiting on blocks/admission, not doing work
            else:
                iter_time = self._iteration_cost_s(batch)
            self._apply_batch(batch, self.clock, iter_time)
            self.clock += iter_time
        else:
            hit_safety_valve = True  # exhausted max_iters without draining -- same practical consequence

        ttft = [r.first_token_time - r.arrival_time for r in self.done if r.first_token_time is not None]
        e2e = [r.completion_time - r.arrival_time for r in self.done if r.completion_time is not None]
        qratio = [
            r.queueing_time / (r.completion_time - r.arrival_time)
            for r in self.done
            if r.completion_time and (r.completion_time - r.arrival_time) > 0
        ]
        total_output = sum(r.output_tokens_generated for r in self.done)
        return SimResult(
            ttft_samples=ttft,
            tpot_samples=self.tpot_samples,
            e2e_samples=e2e,
            queueing_ratio_samples=qratio,
            kv_util_samples=self.kv_util_samples,
            total_output_tokens=total_output,
            sim_duration_s=self.clock,
            num_completed=len(self.done),
            num_preemption_events=self.num_preemption_events,
            num_requests_ever_preempted=len(self.preempted_ids),
            num_rejected=len(self.rejected),
            state_blocks_per_seq=self.state_blocks,
            hit_safety_valve=hit_safety_valve,
        )


def format_report(workload: WorkloadSpec, r: SimResult) -> str:
    def pct(samples, p):
        return _percentile(samples, p) * 1e3  # ms

    lines = [
        f"workload: {workload.name}  ({workload.arrival}, qps={workload.qps}, policy={workload.scheduler.policy})",
        f"completed: {r.num_completed} requests over {r.sim_duration_s:.1f}s (sim)"
        + (f"  [{r.num_rejected} rejected: needed more blocks than total KV capacity]" if r.num_rejected else ""),
    ]
    if r.state_blocks_per_seq:
        lines.append(
            f"hybrid arch: {r.state_blocks_per_seq} blocks/seq reserved up front for recurrent state "
            f"(Mamba/linear-attention layers), on top of each sequence's own growing KV blocks"
        )
    lines += [
        "",
        f"TTFT   p50 {pct(r.ttft_samples, 0.5):.0f}ms  p95 {pct(r.ttft_samples, 0.95):.0f}ms  p99 {pct(r.ttft_samples, 0.99):.0f}ms",
        f"TPOT   p50 {pct(r.tpot_samples, 0.5):.1f}ms  p95 {pct(r.tpot_samples, 0.95):.1f}ms  p99 {pct(r.tpot_samples, 0.99):.1f}ms",
        f"E2E    p50 {pct(r.e2e_samples, 0.5):.0f}ms  p95 {pct(r.e2e_samples, 0.95):.0f}ms  p99 {pct(r.e2e_samples, 0.99):.0f}ms",
        f"throughput            {r.total_output_tokens / r.sim_duration_s:.1f} tok/s",
        f"queueing share of E2E {_percentile(r.queueing_ratio_samples, 0.5) * 100:.1f}%  (median)",
        f"preemption rate       {r.num_requests_ever_preempted / max(r.num_completed, 1) * 100:.1f}%"
        f"  ({r.num_preemption_events} events)",
        f"KV utilization (p95)  {_percentile(r.kv_util_samples, 0.95) * 100:.1f}%",
    ]
    if r.hit_safety_valve:
        lines.append(
            "\nWARNING: queue never fully drained within 3x --sim-seconds -- throughput above is likely an "
            "UNDERCOUNT (in-flight requests aren't credited). Either the system is genuinely overloaded at this "
            "QPS, or --sim-seconds is just too short for this model/hardware/capacity combination -- rerun with "
            "a larger --sim-seconds and check whether the number changes before trusting it."
        )
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(prog="scheduler")
    parser.add_argument("--model", required=True)
    parser.add_argument("--hw", required=True)
    parser.add_argument("--workload", required=True, help="specs/workloads/<name>.yaml")
    parser.add_argument("--dtype", default="fp8", choices=DTYPE_BYTES.keys())
    parser.add_argument("--kv-dtype", default="fp16", choices=DTYPE_BYTES.keys())
    parser.add_argument("--tp", type=int, default=1)
    parser.add_argument("--mfu", type=float, default=0.5)
    parser.add_argument("--mbu", type=float, default=0.8)
    parser.add_argument("--workspace-reserve-gb", type=float, default=4.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--sim-seconds", type=float, default=30.0)
    args = parser.parse_args()

    model = ModelSpec.load(args.model)
    hw = HardwareSpec.load(args.hw)
    workload = WorkloadSpec.load(args.workload)

    sim = Simulator(
        model, hw, workload,
        weight_dtype=args.dtype, kv_dtype=args.kv_dtype, tp=args.tp,
        mfu=args.mfu, mbu=args.mbu, workspace_reserve_gb=args.workspace_reserve_gb,
        seed=args.seed, sim_seconds=args.sim_seconds,
    )
    result = sim.run()
    print(format_report(workload, result))


if __name__ == "__main__":
    main()
