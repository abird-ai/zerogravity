# Numerical correctness, model-quality evaluation and benchmark methodology

**26 Oct 2026 — all-model correctness policy:**
[pinned vLLM and llama.cpp practice](../plans/numerics-and-context/CORRECTNESS_POLICY.md)
governs operator accuracy, model comparisons and separate determinism claims
across CPU/CUDA/ROCm/Vulkan. Shared comparison tools reject incomplete evidence;
their output alone grants no model/backend qualification. Existing exact source,
state and physical-ownership contracts remain binding.

**15 Oct 2026 — mandatory vLLM parity-and-beyond qualification:** [36 concurrency/source tests and fair benchmark methodology](../research/VLLM_PARITY_AND_COMBINATORY_SPECIALIZATION_ACCEPTANCE_2026-10-15.md) require pinned vLLM V1/vLLM-Omni version/config and identical original or derived quant artifact, tokenizer/template, task/numeric policy and comparable device/OS/cache state. Report cold/warm, admission and rejection, per-tenant fairness, p50/p95/p99 TTFT/ITL, accepted **committed** SLO goodput, actual bytes/driver and power, vLLM current continuous/hybrid/batch/disagg capability. Quant-model fidelity and target speculation exactness are distinct. “Outclasses vLLM” cannot be claimed on a different quant, cherry-picked prompt or from the existence of planned design.


**15 Oct specialization evidence clarification:** [derived quantization source qualification](DERIVED_QUANTIZATION_CONTRACT_2026-10-15.md) and [original extreme-specialization evidence/method](../research/SPECIALIZATION_SOURCE_ATLAS_2026-10-15.md) require **three independent comparisons**: original FP8/BF16 model vs lossy derived quant; exact kernel/source math against the *same derived pack*; and target-only vs MTP/DFlash speculative acceptance/output distribution. Source reports Qwen derived quant top1 ~93.81% vs official FP8 yet lossless speculation relative to quantized target—no contradiction. PPL/argmax-only tests can miss invalid scratch/FP16 overflow, and parallel batch reduction may change near ties. Benchmark matched checkpoint/source/quant/prompt/acceptance/context/driver/power, repeated interleaved runs and negative candidates; never promote single prompt speed or confuse pack size with speed.


> **v4 governing contract (2026-10-12):** Detailed technical mechanisms in this supporting spec remain useful, but the [minimal architecture](MODULAR_AI_SYSTEMS_REFERENCE_ARCHITECTURE.md), [corrected composition interfaces](COMPOSITION_AND_EXTENSION_CONTRACTS.md), [execution/state protocol](MINIMAL_EXECUTION_SPINE_AND_PROTOCOL_V4.md) and [task/compiler refinements](TASK_PROGRAM_AND_COMPILER_REFINEMENT_V4.md) **supersede older sketches** whenever they conflict. In particular: **StateDomains only Prepare**; TaskCoordinator performs one PublishRoot; CIR only reports readiness; global resource admission is coarse with device-local allocation; TaskDriver is trusted minimal stage control; RIR/KIR are transient compiler views; GPU-only fallback and numeric/safety gates remain. No implementation is asserted.


This is the authority for deciding whether a kernel or execution optimization is valid and whether a performance claim is supported. The user requests peak performance *without losing general compatibility and fidelity*. No quality promise may be inferred from model loading or plausible answers.

## 1. Four independent notions of correctness

**Semantic formula correctness:** operation implements exact released architecture (e.g. KDA vs GDN, MLA NoPE, sparse top-k, mHC, time compression).  
**Numerical operator agreement:** tensor outputs match an independent reference to documented tolerance under chosen precision/reduction policy.  
**Model quality:** aggregate behavior stays acceptable versus the chosen source checkpoint/quantization and official or independent task/reference corpus.  
**Reproducible state/sampling:** same execution plan, seed, tokens and scheduler produce compatible states and outputs; exact target sampling distribution preserved where advertised.

One may pass while another fails. A 90/100 greedy first-token match against FP8 doesn't prove 90% overall reasoning equivalence; exact logits against the *previous optimized version* do not guarantee parity to original BF16 source.

## 2. Reference layers

- **A: scalar/FP64 mathematical oracle** for attention score/softmax/selection, routing and recurrence where calculable; use sufficiently stable range/reference for quantized integer dot exactness.
- **B: native reference model** (pinned HF Transformers or official source, with exact model and template) for teacher-forced layer activations/logits, sampling and image embeddings. Official reference may use different numerical reductions on GPU; document this.
- **C: zerogravity portable Rust CPU implementation** which implements exact chosen model formulas and quant layout; must agree with A and B by declared tolerance.
- **D: optimized device kernels/graph** compared with C and B under matching precision and chunk mode.
- **E: full serving lifecycle** state, prefix, tools, serialization and sampled/greedy outputs, concurrency and failure.

Never use our own optimized GPU kernel as the **sole** reference for itself. Reference kernels and corpus fixtures must be versioned independently.

## 3. Numeric error metrics

Given reference `r` and candidate `c`, record:
- max absolute `max |c_i-r_i|`;
- max relative with epsilon for near-zero denominators;
- RMS error and relative L2;
- cosine similarity for embedding vectors;
- KL divergence between logits-derived distributions and top-k agreement;
- max/mean error per tensor/layer/context/frontier;
- FP64 sparse attention error, selected-key IDs and score ties;
- first-token greedy agreement and longest matching continuation prefix;
- negative log likelihood / perplexity on checkpoint-matched corpus;
- model-specific official task quality and tool/structured output success;
- sampled distribution statistical tests where exact speculation is promised.

**Numerical tolerance values are not decided globally.** They must depend on op, dtype, quant, downstream sensitivity and reference. Keep stringent bounds for masking, index IDs, integer quant decode, positional state and sampler; allow bounded floating reduction differences only with high-level model quality proof.

## 4. Critical operator regression cases

| Operator | Required edge cases |
|---|---|
| Norm + RMS/Layer | tiny variance, zeros, large vector, BF16/F16 rounding, odd widths |
| RoPE/MRoPE | position 0, long context, extrapolation, multi-axis images, model scale |
| Attention softmax | all masked/empty, extreme logits, causal edges, head tails, ragged chunks |
| Sliding/KV | ring wrap, overwrite after branch, context eviction, page COW |
| Sparse QSA/DSA | selected width, top-k ties, negative scores, index masks, compression boundaries, 32K/128K |
| MLA | absorbed vs expanded compressed latent, NoPE, FP16 cache and head scaling |
| GDN | scalar decay variants, conv kernel 4, chunk vs step, state matrix snapshots |
| KDA | per-channel/fine-grained gate, lower bound, recurrent state, 34-layer geometry |
| mHC | Sinkhorn iterations, 4-stream mixing, normalization and rounding, phase offsets |
| MoE | router bias/scoring/top-k, duplicate experts, shared + routed order, 288/512 experts, quant tails |
| Quant matrix | all sign bits, block padding, scales, exact dequant, strange strides, type safety |
| Engram/n-gram | hashed token history, repeated IDs, invalid table row, truncated file, state restore |
| MTP/DSpark | verifier row grouping, 0..N accepted prefix, RNG rollback, tool boundary, EOS |
| Vision/audio | preprocessing, image resize, position maps, encoder cosine and ordering |

Tests must cover one-token decode, repeated 2/4/8 verification, 256/1024/2048+ prefill and continuation after deep prefixes. The best path for one shape may be slower/wrong for another.

## 5. Quantization and quality comparison

Keep three artifacts conceptually separate:
- original BF16/FP8 model reference and official model family;
- a low-bit checkpoint produced by particular imatrix/calibration recipe, with quality/precision fingerprint;
- a lossless device repack of the second checkpoint that should preserve semantic dequant output.

Comparing BF16 model vs Q2 changes quality even if source code same. Evaluate per-tensor sensitivity and per-family quality on a fixed corpus before accepting new quant. Mixed quant may retain routers, attention and dense always-active projections at higher precision while aggressively compressing routed expert matrices; document weights/bytes per category.

An optimization that changes sparse top-k, cache KV quant or sampler algorithm is not merely layout/performance; it requires its own numerical/quality contract.

## 6. Speculative decoding sampling rigor

For greedy target decoding, proposals are accepted only when matching target decisions under same reference/plan arithmetic; near-tied logits can legitimately diverge across FP precision or batch reduction. Record first divergence and logit delta, not only final answer.

For stochastic sampling, correct target-preserving speculation requires mathematically correct acceptance probabilities and residual redistribution. Verify proposed source `q`, target `p`, RNG consumption, filtering/top-p/top-k/min-p order, penalties and corrections using exact or high-precision CPU probability reference. Property tests over tiny categorical vocab exhaust probabilities and distributions; Monte Carlo validates empirical distribution (with confidence limits). Tests for zero-probability proposals and small `q(x)`, EOS/stop and batch index boundaries.

If an "opportunistic" mode accepts matching greedy proposals at nonzero temperature, label it **not distribution-preserving**, user opt-in only (proposed default policy). Record result differences, don't market as exact target.

## 7. Hybrid state replay equivalence

For a fixed model and reference plan:
- one full prefill vs chunks 1,31,256,1024,2048,non-even;
- decode after initial/continued prefill at 4K/8K/32K/128K and sparse thresholds;
- fork same immutable prefix into N branches, mutate one and ensure others identical;
- serialize/restore snapshot and compare logits/history position;
- run speculative proposals and commit every prefix length 0..d, compare to fresh token-by-token control;
- cancel between GPU events and re-run from committed prefix;
- tool-mode boundary and image/text history.

If output differs only because the execution plan uses different floating reduction order, classify separate "same-plan exact" and "cross-plan tolerance" test. Numerical drift should be bounded and model quality verified; no hidden byte-identity promise across Metal/CUDA/HIP.

## 8. Benchmark taxonomy

**Isolated kernel:** per op/shape latency, occupancy, memory throughput, numerical diff. Useful for diagnosis, not serving ranking.

**Model inference:** prefill tokens per second, target autoregressive decode tokens/sec, speculative **committed** tokens/sec, context depth and batch. Distinguish new prompt chunk length from existing cached prefix depth.

**Service benchmark:** client-observed TTFT P50/P95, inter-token latency P50/P95, total request latency and aggregate tokens/sec under concurrency and mixed arrivals, token correctness and cancellation.

**System benchmark:** cold/warm load, physical RAM peak and GTT vs VRAM, SSD read rate/hit rate, contention, network bytes and tokens per joule; no OOM/swap storms.

**Quality benchmark:** official/checkpoint-matched completions and tasks, numerical metrics and input modality. Never report speed improvement when outputs are silently lower quality.

## 9. Benchmark comparability matrix

Always match:
- **exact model family & checkpoint SHA**, quant bytes/recipe, tokenizer/template, enabled MTP support and auxiliary tables;
- GPU/CPU device model, DRAM/VRAM, PCIe/NUMA/GTT, kernel/runtime/compiler driver versions, memory clocks and power;
- prefill new-token count, existing depth, output tokens, stop policy, cancellation, beginning vs continued prefix, cached state;
- concurrency, warmup, explicit use of graph capture, batch/gpu prefill time and cache condition;
- sampling temperature/top-p/min-p, speculative exact/opportunistic mode, acceptance distribution and draft depth;
- timing method; host vs device timer, throughput vs end-to-end latency, repeat count and quality checks.

When matched comparison isn't feasible, report two separate numbers with visible caveat; do not compute misleading "engine X 10x better than engine Y" from different models.

## 10. Recorded external author baselines (research only)

- Gufo `gfx1151` Qwen3.8 Flash-Next UD-Q4_K_XL: reported pp2048 1700.52 and tg128 25.95 AR at d0; at 131K ~1478.66 pp / 24.92 tg; one-user repetitive MTP up to ~60.39 vs mixed ~31.32; C8 AR aggregate ~108.68. Project's source documents describe pinned llama.cpp comparator and one warmed sample.
- A specialized `gfx1151` Qwen3.8 native-quant engine: ~1584 pp 8K, ~1517 pp 131K, ~37.6 short-context serial, ~55–56 coding-agent speculation; different checkpoint/quant and benchmark methodology from Gufo.
- Maya GLM5.3 Flash Q2 on 128GB Strix: ~210–217 pp and ~17.8 tg when experts resident; dual-gpu AMD MTP ~33 tg with different machine.
- DwarfStar Strix DeepSeek V4 Flash Q2: ~187.26 to 292.86 pp after kernel tuning, context around 4K; model differs from Qwen/GLM.
- Prior-art evidence shows model- and operator-dependent performance gaps even when the host language is Rust; source-specific numbers belong in the research atlas, not as a universal design target.

Numbers show **mechanisms** worth reproducing, not targets achieved or ranked. Reinspect source bench documents and attach per-commit evidence before using them as acceptance thresholds.

## 11. Statistical method

Prefer at least several stable independent repeats with alternated A/B order (ABBA or randomized block), power/thermal stabilization and significance/uncertainty. Use bootstrap interval or robust median CI, show raw result distribution and context-specific outliers. For p95 need enough requests for meaningful estimates; 10 token samples do not give reliable tail percentiles. Time-of-day/OS load/SSD page cache should be controlled, and explicit test classes distinguish cold from warm and cache-hit from miss.

A win on individual kernel might be rejected if whole-model throughput worsens or P95 rises. Small single-run percentage differences should be reported as inconclusive.

## 12. Recording results

Record the command, relevant model/reference/device/toolchain profile, measured
quality/performance, actual result and remaining failures in the plan TODO.
Model content identities and independent numerical oracles remain required where
byte/source equivalence matters. Keep a bounded diagnostic capture only when it
helps investigate or assess that result; no blanket log inventory or receipt
schema is required. See [the TODO workflow](../../AGENTS.md),
[experiments](../plans/EXPERIMENT_PROGRAM.md) and
[release gates](../plans/RELEASE_ACCEPTANCE_GATES.md). Never commit model weights,
private prompts or credentials.

## 13. Quality gates and acceptance policy

A new specialized kernel only becomes default if:
1. required shape and semantic tests pass;
2. state rollback/snapshot/continuation tests pass;
3. no unauthorized change of quantization, sparse selected set or sampling distribution;
4. model quality remains within documented tolerance for the exact checkpoint;
5. matched end-to-end benchmark demonstrates gain beyond noise or a justified throughput/latency tradeoff;
6. memory and device failure/stress tests pass;
7. fallback tested when kernel disabled and compiler/driver cache invalidated.

If data incomplete, keep current simplest correct kernel; optional experimental flag permitted only with clear label. The team's optimization lab must preserve negative experimental findings to avoid rediscovering them.

## 14. Long context and quality

A model's declared max context is not proof of useful retrieval or stable latency at that depth. Exercise indexed sparse-attention threshold, latent compression and recurrent state, 8K/32K/64K/128K/256K/1M where hardware allows. Use needle/structured retrieval, code references and official continuation logprobs. Attention selector quality should be measured separately from overall model quant quality.

## 15. Acceptance report example (pseudocode)

```text
Experiment: gfx1151-quantized-moe-prefill
Model: exact GGUF SHA / official reference / quant
Hardware: AMD gfx1151, 128 GB, ROCm build, fixed power
A: kernel version 1, B: specialized version 2
Shapes: expert count, active top-k, 256/1024/2048 new rows
Correctness: maxabs/rms, selected ids, state parity, full logits
Quality: official corpus reference, greedy divergence and NLL
Performance: ABBA repeats, medians/CI, TTFT/P95, cache/SSD
Result: pass/neutral/fail/inconclusive
Next action: keep/revert/iterate, commit and source links
```

All fields filled from actual evidence, not from qualitative architectural reasoning.

## 2026-10-10 integrated v2 addendum — defensible Pareto performance and correction ledger

The long-term objective is to approach or improve the **measured Pareto frontier**, not claim faster-than-everything by architecture. Report per (model checkpoint, quant recipe, device, driver, context, batch/concurrency, acceptance, sampler, power) cells: throughput/TTFT/P95, RAM/cost/power, independent numerical fidelity/quality and availability. A win for C1 decode cannot imply a C32 serving win; repetitive MTP speed is not unconstrained reasoning t/s. External prior-art engine numbers (Gufo/Maya/DwarfStar and similar) are author reports and quant/workload-specific.

**Serving-mode-aware tactic selection:** tune candidate kernels against **actual eager vs captured graph execution**, not isolated GPU kernel timer alone. Offline schedule search demonstrates possible speedups but also a compilation-time explosion risk; use bounded experiments. Bound compile time/memory and warmup; ship measured plan, with reproducible default and no unchecked runtime JIT inside token loop.

Preserve research's detailed `S1–S12` GPU/compiler and CPU-oracle ABBA acceptance, but split tests:
- strict same-plan exact/bitwise on controlled exactly-representable cases;
- FP32/BF16/fused GPU numerical agreement at declared tolerance;
- exact target **sampling distribution** with speculative rejection/correction;
- checkpoint-level quality and long context/task tests;
- cross-plan/backends independent quality contract rather than universal bit identity.

**Source corrections:** the research's 8×L40S ~126 tok/s DwarfStar benchmark is a different model/workload and cannot stand as matched GLM-5.3 evidence. Historical 75–125 engineer-month and ~93 engineer-day Phase0/4–6wk estimates are staffing *hypotheses*, not guarantees. Research `D-14` MIT/Apache and old CUDA kernel versions are council recommendations, not user approvals.

**New test classes:** CIR lease/event failure, compiler effect rewrite legality, 3+ hybrid state group prefix share, model-version/LoRA tenant cache isolation, GPU cold/host memory/GTT and NVMe critical path, new backend codegen generated ISA. Publish and retain negative experiments; any unproven gain stays opt-in.


## 2026-10-12 integration — NPU BFP compute and performance comparability

The Gufo XDNA2 source converts Q8_0 model weights and GPU F16 activations into v8bfp16ebs8 block-float (8 signed mantissas and 1 shared exponent per nine bytes). This is a **compute-arithmetic change** rather than a new source GGUF checkpoint. Its 30 matching greedy prompts are a preliminary smoke test, not proof of equal logits, stochastic distribution, robust coding quality or GDN/QSA state. Require full tensor/logit/quality and model-continuation parity, attention selected-key identity and all-state/speculative rollback.

The claimed 16K prefill GPU-only 1698 → GPU+NPU 1821 tokens/sec was measured with held fabric clock and performance-mode NPU, so benchmark comparisons must hold clock/power/temperature constant in both arms, and measure GPU-only default vs held-clock as a separate experiment. Include pipeline first-consumer stalls, NPU BO sync, pinned RAM, energy/idle power, C1/C8 concurrency and long context. On hardware at risk of hard hang, do not conduct uncontrolled GPU-NPU load tests. Source numbers are author reports, **not** zerogravity achievements.

## v4 refinement — four distinct truth claims and task-relative units

Clearly distinguish: (1) **semantic model correctness** including trained routing/positions and state effects; (2) **numeric compute agreement** at a declared rounding/reassociation/temporary quant tolerance; (3) **checkpoint/task quality** under model+input+context/phase; (4) **distribution/state exactness** of sampled speculation and target verifier under supported modes. A faster stored quant or altered expert top-k/attention indexer is a **changed model/approximation**, not an automatic optimization. Strict bitwise replay may be a fixed-plan test; no blanket cross-driver GPU bit identity without proof.

For non-token tasks compare appropriately: ASR transcript/WER/timecodes, TTS audio output and latency, image/video sampled quality/completion, embedding/rerank accuracy, and optional future training loss/gradient. Use `WorkUnit` and `OutputUnit`, not fake t/s across all AI execution. Benchmark actual whole-task P95, time-to-first-valid-output, peak physical memory and energy in addition to operator kernels, under matched source checkpoint/quant/driver/power. Source-author claims and hypothetical architecture wins remain unverified.

## v4.1 — cross-accelerator precision and program validity

Qualify compute arithmetic and task quality per `RegionExecutionProvider`, not one generic “GPU supports FP4” flag. At minimum distinguish **native FP4 E2M1 with FP8 scale per 16 values and per-tensor scale**, power-of-two scaled microscale FP4 across 32 values, FP8/BF16 formats, integer/GGUF block encodings and device-specific compute accumulation. A temporary device compute format differs from source checkpoint quantization and a deliberately requantized checkpoint; **do not change model routing, trained attention or final sampling silently to fit a TPU/wafer/Blackwell backend**.

For graph-program backends, correctness must include **whole closed subgraph values, dynamic shape guards, device-managed recurrent/KV/indexer state, provisional snapshot/rollback or explicit stateless restriction**, and exact temporal task frontiers. A program that compiles but cannot reconstruct correct state after cancellation/spec rejection is not functionally qualified for general decoding. Published vendor tokens/s or TPU training peak does not establish same-model inference performance; compare exact checkpoint/quant/graph compiler/driver and cold compile amortization, full TaskDriver P95/quality/energy, topology/link and per-chip useful outputs. Temporary/simulated graph-only backend passes architecture proof **only**, not physical performance.

Read [accelerator backend contract](ACCELERATOR_EXECUTION_FAMILIES_AND_BACKEND_CONTRACT.md) and research [platform evidence](../prior-art/ACCELERATOR_PRIOR_ART_AND_PLATFORM_RESEARCH_2026-10-12.md).

## v4.2 — distributed mathematical parity and transport-aware benchmarks

Three correctness classes require **different** references: (a) contiguous layer parallel with per-layer KV/SSM co-location must match unpartitioned full model state and logits; (b) attention/FFN compute/state separation must preserve the exact source attention/FFN and numeric mode with explicit cross-network activations; (c) sequence-range KV sharding may merge per-shard attention summaries only if **the operator's PartitionAlgebra** reproduces mathematically valid causal/masked softmax and data-dependent selected-indexer semantics. Empty all-masked shard and floating overflow/underflow/tie cases need numerical reference tests. **No generic GDN/KDA recurrence sharding assumption.**

Metric matrix: C1 and C8/32 latency, TTFT/p95 ITL and committed t/s, prefill/verify and long context 4K/32K/128K; same source model/quant and device capacity, network direct LAN vs encrypted overlay direct vs peer-relayed vs high-RTT WAN, tunnel encryption CPU/MTU/loss and idle power. Measure **capacity enabling** separately from raw speed. KV state transferred must include hybrid recurrent/indexer/ngram/modal/grammar when required; a text-prefix hit without that state is invalid.

Fault/quality tests: wrong remote model hash/quant/adapter, corrupted/partial transfer, stale cache hint, remote replica serving wrong checkpoint, coordinator epoch loss/split brain, remote old-epoch StateShard generation and event/DMA quiescence, old root recover/replay. A physical network test needs actual machines/links; simulated cost doesn't prove performance. [Network research](../prior-art/DISTRIBUTED_PRIOR_ART_AND_TUNNEL_RESEARCH_2026-10-12.md).

## v4.3 exact-model routing and multi-model performance truthfulness

Routing a request to the wrong checkpoint, different stored quant, LoRA adapter generation, model-native tokenizer/template or sampler is **a semantic correctness regression**, even if output is plausible and the destination replica produces more tokens/sec. Default exact alias routes only among model/TaskDriver/numerical-compatible replicas. Explicitly configured `auto` source selection is a **changed target model policy** and reports actual resolved source, not an equivalent kernel optimization. Model profile revision/hot reload can't change live source StateRoot.

Report three distinct measurements: (1) one model `run` direct/in-process vs `serve` through ModelRouter under same artifact/task/sampler with negligible/per-request route work and **no per-token proxy tax**; (2) several resident source models/replicas with physical RAM/session KV and native batching, C1/C8 TTFT/P95, fair queue; (3) model-loading/unloading/cold start/anti-thrash and source quality/availability. “Many registered models” isn't a claim they all run simultaneously. A multi-model total output tokens/sec can be misleading if compared across different checkpoints/context/numeric or tasks; report per-source and per-tenant workload weights.

API conformance is a separate quality gate: tested SDK and endpoint versions, chat vs Responses SSE schemas, usage, model ID, stop/JSON/tool data, Unicode and cancellation, unsupported parameter errors, no assumed stateful Response/tool execution; no silent downgrade to different model or quant when overloaded. [Router](MODEL_ROUTING_RESIDENCY_AND_MULTI_MODEL_SERVING_CONTRACT.md), [CLI/API](ZERO_CONFIG_CLI_CONFIG_AND_OPENAI_API_CONTRACT.md).

## U-048 — separate NPU-twin parity, source quality and route performance

[heterogeneous pipeline qualification](HETEROGENEOUS_PREFILL_AND_RETAINED_REPLAY_CONTRACT_2026-10-15.md) distinguishes exact W8/grid-scale/BF16
CPU↔GPU-twin↔NPU execution from quality versus the selected immutable source
quantization and from target-lossless speculation. Greedy text equality alone
proves neither logits nor sampled law. Paired/tail arithmetic and selected family
cuts are bound to existing numeric/state provenance or a proved compatibility
class. H5/H6 source-bit gates remain intact; incompatible exact-mode candidates
are refused. A qualified tolerant/derived encoding remains explicit policy, never
a silent tolerance increase. Timing needs matched artifacts, quality, routes,
state/KV modes, prompt/sampling/context, power and repeated stationary comparisons;
all 40 new HP gates are planned, with no new performance result here.

## Original Qwen3.5 source arithmetic profile

This technical summary retains implementation constraints formerly described in
retired phase reports. The detailed executable contract is the existing CPU/GPU
source helpers, independent reference clients and regression tests. Accepted
results and open work are in [the numerics TODO](../plans/numerics-and-context/TODO.md).

- Pin native reference revision, CPU/F32 KV/nonflash mode and physical
  microbatch/chunk history. Raw teacher token IDs avoid incomplete UTF-8 HTTP
  fragments. Functional8K thresholds do not permit source-bit operator changes.
- Quantized projections preserve their source block and reduction schedules:
  Q4_0 uses Q8_0 activation blocks and the optimized eight-row32-integer-block
  FP32 FMA order; Q4_1 adds its separate zero-point path; Q8_0 retains eight
  accumulators; Q6_K uses signed-six/four-product/eight-lane order and contiguous16
  correction. Original F32 uses four FMA accumulators and its reduction tree.
  A standalone one-row dot is not automatically the same source schedule.
- FP32 products, recurrence iterations and reduction intermediates must round
  where the source rounds. Explicit stored buffers provide required device
  boundaries; a local cast need not prevent contraction or reassociation.
  GDN order is decay, prediction, beta, state update, readout and scaling.
  RMS uses rounded squares, ordered F64 sum, then the specified FP32 stages;
  Q/K L2 retains its RMS(epsilon/width) and scale schedule.
- WideF64 GEMV uses bounded F64 multiply/add then F32 conversion, with no
  residual buffers; its measured source qualification is not general IEEE FMA
  emulation. Separately, RoPE/QK/value and other qualified FMA consumers retain
  stored FastTwoSum residual correction and midpoint regressions because a
  generic F64 multiply/add then F32 conversion can double-round.
- Preserve softmax's two-step F64 reciprocal then F32 conversion. For example,
  denominator `0x3ff0000018000024` gives F64 `0x3fefffffd0000000`, then F32
  `0x3f7ffffe`; direct exact-real-to-F32 rounding is a different operation.
- Attention value accumulation includes exactly ceil(valid/256)*256 source
  FMAs with canonical future operands +0. Skipping padding or adding capacity
  blocks changes zero signs. Rewind/import must restore canonical future KV;
  capacity changes stride, not the arithmetic prefix.
- Partial IMRoPE pairs half-head axes with iterative FP32 theta/source FMA.
  Do not replace it with a closed-form FP64 phase or unqualified device trig.
  GGML polynomial SiLU differs from scalar-libm attention sigmoid/GDN gates;
  keep each original exp/trig, contraction, underflow and signed-zero policy.
- Activation quantization preserves Q8_K first signed maximum, independent
  rounding and Q8_1 sum from unrounded d before half conversion. Source-defined
  tiny blocks can contain permitted Inf/NaN scratch and saturating NaN-to-zero
  code conversion. Validate consumed scales/status/final values according to
  the operator; this is no exception for observable nonfinite model outputs.
- Greedy selection rejects nonfinite logits and preserves finite f32 total_cmp:
  +0 outranks -0; equal-bit maxima select the lowest token ID. Preserve source
  bits and subnormals. Never silently widen zero-sign or numerical policies.
