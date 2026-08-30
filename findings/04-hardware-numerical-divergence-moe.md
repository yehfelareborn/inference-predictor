# Finding 04: Why MoE models are more likely to diverge across hardware in agentic workflows

**Status: hypothesis, grounded in published work, not yet independently tested in this project.** This finding is different in kind from 01–03: those follow predict → measure → explain. This one predicts and cites external evidence for the mechanism, but this project has not run the cross-hardware experiment itself. Section 5 spells out exactly what that experiment would need. Don't read this as validated until that section is closed out.

## 1. The question, and why it matters

Same checkpoint, same dtype, same sampling parameters, deployed on two different accelerators (say H100 and B200) — should the outputs ever differ?

The capacity/latency framing in this repo (M0–M4) implicitly assumes the answer is no: hardware changes *how fast* you get an answer, not *what* the answer is. That assumption is what lets M1's roofline model treat correctness as a solved, hardware-independent given and only model speed.

In practice, teams running agentic workflows (multi-step tool use, plan → call → observe → replan loops) sometimes report that migrating the same model to different hardware, or even just a different inference engine version, changes behavior — degraded response quality, or the agent getting stuck repeating the same failed action. If that's real and not just anecdote/confounded by other changes, it's a gap in the "hardware only affects speed" assumption that the rest of this project rests on, and it hits MoE models (DeepSeek-family included) harder than dense ones. Worth naming the mechanism precisely instead of leaving it as a vague "sometimes it just behaves differently."

## 2. The mechanism (analytical prediction)

The causal chain, step by step:

1. **IEEE 754 floating-point addition is not associative.** `(a+b)+c` and `a+(b+c)` are only approximately equal, not identical. This is a hardware/math fact, not a bug.
2. **Different GPUs — and even the same GPU at different batch sizes — invoke different kernels** for matmul, RMSNorm, and attention (different tiling, different reduction order, different Tensor Core MMA shapes per architecture generation). Each choice implies a different order of floating-point reduction.
3. **This produces tiny per-logit differences** (relative error roughly 1e-3 to 1e-6) between hardware/kernel combinations. Negligible for a continuous-valued metric, but LLM decoding is full of hard decision boundaries: greedy decoding is an argmax over the vocab, and top-k/top-p sampling still requires a stable ranking before the randomness is applied.
4. **Dense models have exactly one hard decision per generated token**: the final vocab-logit argmax. Every token is processed by the same fixed set of weights, so tiny numerical jitter has one place to matter.
5. **MoE models add one hard decision per token per MoE layer, on top of that.** The router's top-k expert selection is itself an argmax/top-k over router logits. With load-balancing objectives during training pushing router logits toward closer, more evenly-distributed scores across experts (that's the whole point of load balancing — avoid starving experts), many tokens sit close to a routing boundary between two experts. A model with `L` MoE layers and top-`k` routing has on the order of `L × k` additional hard decisions per token, compared to a dense model's one.
6. **Flipping a routing decision is not a small perturbation — it swaps in a different learned function.** Token routed to expert A vs expert B doesn't shift the hidden state by a little; it applies a different FFN entirely. The output can diverge by far more than the tiny numerical nudge that caused the flip. This is amplification, not just propagation.
7. **Autoregressive generation compounds this over a sequence, and agentic loops compound it further.** One flipped token early in a tool-calling chain changes all downstream context (the generated token becomes part of the next step's input). A multi-step agent (plan → call tool → observe → replan) has many more autoregressive steps than a single Q&A turn, so there are more chances for one flipped decision to push the trajectory somewhere the model has no good recovery path from — which looks, from the outside, like "got stuck in a loop."

Net prediction: **MoE models should show measurably more cross-hardware / cross-kernel output divergence than comparably-sized dense models, and the gap should widen with sequence length and number of autoregressive steps** — exactly the shape of an agentic workflow.

## 3. Why DeepSeek specifically

DeepSeek-V3/R1-class models use fine-grained MoE — on the order of 256 routed experts with top-8 activation, plus shared experts, at every layer. The sheer number of routing decisions (256-way top-8, per layer, per token) multiplies the number of close-call boundaries relative to a coarser MoE like Mixtral's 8-way top-2. More experts competing for a fixed top-k slot means, statistically, more near-ties — more surface area for a numerical nudge to flip something.

This isn't just this project's inference from first principles. DeepSeek's own team has apparently run into a stronger version of this same problem and had to design around it. Per public reporting on the DeepSeek-V4 technical report: when multiple SMs from different expert-parallel ranks concurrently write to the same receiving buffer, the order in which writes land is itself nondeterministic — a distinct, lower-level source of nondeterminism than the kernel-reduction-order issue in step 2 above, but the same *category* of problem (nondeterministic numerics → routing/accumulation order changes → different result). Their fix was a token-order pre-processing step per rank plus buffer isolation across ranks, specifically to make expert-parallel dispatch and MoE backward-pass accumulation deterministic. That the DeepSeek team built dedicated infrastructure for this is strong independent evidence the problem is real at their scale, not a theoretical curiosity.

Separately, a recent paper studying this directly — "From Expert Reduction to Behavioral Divergence: Tracing Numerical State through Sparse MoE Inference" — reports that perturbations to expert-reduction order produce measurable "routing and semantic bifurcations" in MoE inference. That paper's own framing is important, though: it establishes that such perturbations *can* produce bifurcation, not how often a real GPU/NPU/distributed runtime naturally realizes each permutation in production. That's exactly the gap this project would need to close to turn this from "plausible mechanism" into "measured effect."

On the general nondeterminism side, Thinking Machines Lab's "Defeating Nondeterminism in LLM Inference" work (and the accompanying `batch_invariant_ops` implementation) is the most concrete public demonstration of the root cause: they show that LLM inference is nondeterministic even at temperature 0 because kernel reduction strategy depends on batch size ("batch invariance"), and that constraining kernels to a single, batch-size-independent reduction strategy gets bit-identical outputs across 1,000 repeated runs of Qwen3-8B — at a real cost (61.5% throughput hit for their baseline batch-invariant kernels, reduced to ~34% overhead once SGLang integrated them with CUDA graphs). That's a dense model, not MoE, and it's the *same-hardware, different-batch-composition* version of this problem rather than the *cross-hardware* version this finding is about — but it's the cleanest available evidence that (a) the underlying nondeterminism is real and measurable, not just theoretical, and (b) fixing it is possible but not free.

## 4. Testable claims

If the mechanism above is right, all of the following should hold:

- **Claim A**: Running the same MoE checkpoint on two different GPU architectures (or the same GPU with different attention backends/kernel flags) at temperature 0 produces token-level divergence at a rate higher than a comparably-sized dense model under an identical test.
- **Claim B**: Divergence rate correlates with routing granularity — more experts / lower top-k fraction (DeepSeek-style fine-grained MoE) diverges more than coarser MoE (Mixtral-style) under the same hardware-swap test.
- **Claim C**: Divergence probability grows super-linearly with sequence length / number of autoregressive steps, not linearly — consistent with compounding rather than independent per-token error.
- **Claim D**: Multi-turn agentic (tool-calling) benchmarks show a higher rate of degenerate/looping behavior under a hardware or kernel-backend swap than single-turn benchmarks do, for the same model.

## 5. What this project hasn't done yet

No experiment has been run here. To turn this from a literature-grounded hypothesis into an actual finding with a real/predicted-vs-measured section, at minimum:

- Two GPU architectures (or, cheaper: one GPU with two different forced attention backends/kernel configs — vLLM already lets you pin this) running the **same** MoE checkpoint, same dtype, same sampling config (temperature 0).
- A small-enough MoE model to run this affordably — Mixtral-8x7B (already in `specs/models/mixtral-8x7b.yaml`) is a reasonable starting point before attempting anything DeepSeek-scale, which is well beyond this project's `~$40` total GPU budget.
- A comparison harness measuring token-level edit distance / first-divergence-point between generation trajectories across the two configs, over a batch of prompts, at increasing generation lengths — to test Claim C directly.
- Ideally, a small agentic benchmark (a fixed multi-step tool-use task) run under both configs, measuring loop/failure rate — to test Claim D, which is the part that actually matters for the motivating question in Section 1.

This is a real gap in this project's current scope (M0's `core/memory.py` and the planned M1 roofline model both implicitly assume output is hardware-independent). Whether it's worth pursuing as a fourth empirical finding, on top of the roadmap's original three, is an open question — flagging it here rather than deciding it here.

## 6. When this would not hold

- **Batch-invariant / deterministic inference kernels.** If the serving stack uses batch-invariant kernels throughout (the Thinking Machines approach, or any equivalent), the reduction-order source of divergence in step 2–3 goes away by construction. This is real, shipping work, not hypothetical — the mechanism in this finding specifically does not apply to a deployment built this way, at the cost of the throughput hit noted above.
- **Well-separated router logits.** If a MoE model's router was trained with a large enough confidence margin between the top-k and (k+1)-th expert for most tokens, small numerical jitter isn't enough to flip the decision. This is model/training-dependent, not universal to all MoE architectures — a MoE model with a strongly peaked router is much less exposed to this than one trained under aggressive load-balancing pressure toward uniform routing.
- **Non-agentic, short, single-turn use.** The compounding argument in step 7 needs autoregressive length to bite. A single short completion has little room for one flipped token to snowball; the effect this finding is about is specifically an agentic/long-horizon phenomenon, not a general "MoE models are unreliable" claim.
- **Sampling instead of greedy decoding doesn't save you as much as it sounds like it should.** Token *sampling* being stochastic doesn't make MoE *routing* stochastic — routing is (almost always) a deterministic top-k over router logits regardless of decoding temperature. Turning up temperature reduces sensitivity to logit jitter at the final vocab argmax, but does nothing for the routing argmax inside each MoE layer.

## Sources

- [Defeating Nondeterminism in LLM Inference — Thinking Machines Lab](https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/)
- [thinking-machines-lab/batch_invariant_ops (GitHub)](https://github.com/thinking-machines-lab/batch_invariant_ops)
- [From Expert Reduction to Behavioral Divergence: Tracing Numerical State through Sparse MoE Inference (arXiv 2607.28097)](https://arxiv.org/html/2607.28097)
- [DeepSeek-V4: Towards Highly Efficient Million-Token Context Intelligence (arXiv 2606.19348)](https://arxiv.org/pdf/2606.19348)
- [Deploying DeepSeek with PD Disaggregation and Large-Scale Expert Parallelism on 96 H100 GPUs — LMSYS Org](https://www.lmsys.org/blog/2025-05-05-large-scale-ep/)
