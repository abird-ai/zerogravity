# Vision, product scope, constraints and engineering philosophy

> **v4 governing contract (2026-10-12):** Detailed technical mechanisms in this supporting spec remain useful, but the [minimal architecture](MODULAR_AI_SYSTEMS_REFERENCE_ARCHITECTURE.md), [corrected composition interfaces](COMPOSITION_AND_EXTENSION_CONTRACTS.md), [execution/state protocol](MINIMAL_EXECUTION_SPINE_AND_PROTOCOL_V4.md) and [task/compiler refinements](TASK_PROGRAM_AND_COMPILER_REFINEMENT_V4.md) **supersede older sketches** whenever they conflict. In particular: **StateDomains only Prepare**; TaskCoordinator performs one PublishRoot; CIR only reports readiness; global resource admission is coarse with device-local allocation; TaskDriver is trusted minimal stage control; RIR/KIR are transient compiler views; GPU-only fallback and numeric/safety gates remain. No implementation is asserted.


**zerogravity** is a Rust-native inference compiler+runtime. Its promise is not that it can parse any file or always exceed every competing engine, but that for **each supported model** there is a verified correct portable execution path, and that the same model can use progressively more specialized execution on capable hardware without losing maintainability or exposing multiple incompatible runtimes.

## Directly expressed user requirements

- Build **zerogravity**, in Rust, with a comprehensive handoff archive.
- Combine portable model execution, token-budget continuous batching, prefix-aware serving, fused model-region specialization, tiered sparse weights, hybrid state and Rust-owned device kernels into one original system.
- Include everyday modern models: Llama, Gemma 4, Qwen 3.5+, dense/MoE and new families.
- Include unusually large models: Qwen3.8-Flash-Next ~125B language backbone plus auxiliary n-gram/MTP, GLM-5.3-Flash ~321B total, on AMD Strix Halo Ryzen AI Max+395 / Radeon 8060S specifically.
- Strive for best peak performance **on eligible model+hardware**, and general fallback elsewhere.
- Aspire to native high-throughput serving, portable model usability and extreme hardware-specific inference performance, measured without changing model quality, plus future composable AI tasks.
- Formal Semantic IR is the chosen architectural direction; precise IR design and kernel framework remain proposals pending proof.

## Foundational contract

Supported execution means all of the following: checkpoint and tokenizer parsed unambiguously; semantic graph defined; quantization/numerics declared; device backend lowers every required operator; memory available or supported streaming plan; state and sampling semantics preserved across prefill/decode/branch/cancel; observed outputs validated against independent references. **Not** merely model name recognized or a model file downloaded.

For each feature declare stages: `PLANNED`, `EXPERIMENTAL`, `FUNCTIONAL`, `VERIFIED`, `PERFORMANCE_QUALIFIED`. A model can be `VERIFIED` on CPU and experimental on ROCm without contradiction. A specific quant variant can be unsupported while another variant is supported. Modality support is independent of text support.

## Three plan levels

1. Portable/reference execution: stable semantics, robust fallbacks, correct state.
2. Optimized device execution: tuned kernels, batching, graph capture, efficient memory.
3. Extreme specialization: named model-family/hardware tensor shapes, target instructions, sparse/recurrent/MoE fusion; zero forced semantic graph overhead in hot path.

A specialized plan may encompass an entire layer or model region, not just an isolated op. But all inputs, effects and outputs must satisfy the verified region contract. A fast path can be bypassed without restarting the entire process; a state-layout change may require a safe checkpoint/conversion boundary.

## What makes zerogravity qualitatively different (hypotheses, not claims)

- Compile-time mathematical graph and state effects separated from runtime request decisions.
- One resource arbiter coordinating expert cache, prefix state, scratch, n-gram/Engram disk IO and speculation, instead of independent competing allocators.
- Plan candidates optimized for **real concurrency and route overlap**, not only isolated GPU GEMM.
- Full hybrid-state prefix sharing (KV + recurrent and sparse indexer) to reduce repeated agent-context prefill.
- Adaptive speculative policy aware of *verification and memory cost*, not acceptance alone.
- One stable interface for portable, tuned and extreme **Rust-authored** GPU kernels.
- Honest, published negative experiments; automatic selection only after whole-model validation.

## Non-goals

- Arbitrary execution of remote HF model Python code; open format != arbitrary semantics.
- Unlimited context at fixed performance; context/memory geometry matters.
- Magical 321B model in insufficient RAM: SSD tier can help capacity but IO determines usable speed.
- Drop precision silently to reach a throughput marketing metric.
- Force one CUDA/ROCm/Metal implementation for all GPU families.
- Claim speed dominance based on unrelated model/quant numbers.
- Implement a full GPU driver/OS stack in Rust; exact acceptable external vendor runtime interaction is O-003.
- Give client code low-level device pointers or require users to understand expert-cache mechanics.

## User journeys (product specs)

**Local developer:** downloads supported GGUF/safetensors; `zerogravity run model.gguf`; runs text or an explicitly supported multimodal model; sees automatic hardware selection, fit forecast and descriptive unsupported-op failure; no Python inference server dependency. Output streaming and Ctrl-C cancellation preserve integrity.

**Coding agent:** connects via local OpenAI/Responses-compatible endpoint; long system/tool prefixes reused, tool schema/rendered bytes preserved, concurrent sessions isolated, snapshots safely persisted/invalidated and incremental generation begins promptly.

**Strix Halo owner:** model supports `gfx1151`; runtime detects GTT vs physical RAM, computes safe model/context/scratch fit, selects quantized fused GDN/KDA/attention/MoE plan or routes missing experts to IO, displays performance/capacity tradeoffs. Never requires unsafe kernel parameters to function; documented optional tuning is separate.

**GPU server operator:** serves many users, continuous batching with low P95, fairness, quotas, prefix caching and metrics; separate tuning modes for low latency and aggregate throughput; kernel + model + drivers pinned for reproducibility.

**Model contributor:** provides tokenizer/config adapter and semantic ops; CPU reference passes first; portable GPU follows; extreme specialization optional. Does not touch HTTP routing/session lifecycle to add another attention type.

## Product experience requirements

- One coherent executable and Rust SDK (names/flags in other docs are illustrative).
- Clear `inspect` output: model, checkpoint identity, supported modalities, backend chosen, op/region fallback reasons, memory layout/budget, disk IO and speculative mode.
- Appropriate errors: unknown model op, wrong tensor name/shape, unsupported quant, corrupted shard, insufficient memory, unusable GPU, untrusted remote code, incompatible checkpoint, invalid prefix snapshot.
- Deterministic and auditable `--quality`/reference configuration for tests; explicit `--fast` policy cannot silently change model format.
- Cold/warm startup and memory reservation displayed honestly.
- Conservative default API bind localhost and bounded input/remote asset access.

## Quality constraints and optimization order

1. Semantic and state correctness.
2. Quality equivalence relative to chosen checkpoint + quant policy.
3. Safety, memory lifetime, crash handling.
4. Practical compatibility/user clarity.
5. Stable latency and throughput.
6. Peak specialized speed.
7. Experimental complexity only when proven helpful.

Do not invert this hierarchy by prioritizing a kernel headline over exact model state or system stability.

## Integration boundaries

- Frontend owns HTTP/SDK and prompt rendering; it does not own GPU state.
- Model registry owns mathematical definitions/weights and checkpoint provenance.
- Compiler owns execution region legality and plan generation.
- Runtime owns session frontier, scheduling and speculation.
- Resource manager owns buffer leases/IO/prefetch/eviction and physical budgets.
- Device backend owns actual async compute and events.
- Validator owns independent references and qualification evidence.
- No module may mutate another module's invariants through raw pointers without a versioned narrow contract.

## User decisions not inferred

No license, supported OS release priority, first model order, kernel compiler selection, precise numerical tolerance, context/concurrency targets, remote distribution policy, public milestones or deployment dates were selected. They remain explicit in DECISION_REGISTER; phase plan uses recommendations.

## Success definition

Achieve credible model breadth and convenience *plus* narrow performance-class wins where hardware specialization is valuable. Publish full quality/performance matrix including losing cells. The ultimate goal is a design that can improve across models and devices without accumulating incompatible model-specific servers.

## 2026-10-10 integrated v2 addendum — research synthesis and product boundaries

The unified product vision is **a semantic, effect-aware, hardware-adaptive inference compiler + Rust serving engine**, not three independently maintained "personalities". A single model recipe produces reference execution, tuned generic and extreme region variants; all feed the same model/session/worker state contracts. The added async resource-command layer enables optimizing copies, GPU graphs, expert miss I/O, KV sharing and scratch simultaneously without a graph interpreter in token hot path.

"Outclass all projects" is a **measured Pareto ambition**, not an all-model/per-hardware guarantee. Benchmark quality and exact sampling distribution separately from throughput, TTFT/p95, memory/energy, startup and ability to accept model families; publish losing cells and unsupported paths. A correct general fallback is guaranteed for *registered semantic operators* on an actually capable backend with sufficient memory, not arbitrary unknown HF custom code or GGUF metadata.

The prior-art corpus examines budgeted multi-request serving, precise portable formats, extreme quantized kernels, stateful hybrid inference, tiered sparse parameter execution, compiler rewrites and serving-mode-aware autotuning. Its named sources and licenses are preserved in `PRIOR_ART_LICENSE_AND_PROVENANCE_POLICY.md` and historical research. The system defines original SIR/RIR/KIR/CIR and typed TaskProgram ownership instead of relying on any named upstream project.

**Product coverage expanded explicitly** to encoder-only embeddings/rerank, LoRA/adapters, grammar-constrained decoding, content-addressed ModelStore and multimodal encoder budgets, in addition to user-required modern text models. Each capability has a distinct acceptance/runner type, not a blanket promise of v1 support. No mandatory dynamic third-party executable plugin ecosystem or deeply duplicated model-specific server.


## 2026-10-12 capability expansion — Gufo NPU and task pipelines

User decision U-016 explicitly adds the **mikealanni Gufo fork's partial same-SoC XDNA2 NPU+iGPU cooperative prefill** (Qwen3.8 Flash-Next) and upstream Gufo ASR/TTS/image/video architecture research to zerogravity's design. This extends the product's hardware adaptability beyond GPU/CPU/dGPU+UMA expert tiers: a device may own **only a subset of a pure operator's output rows** while the main accelerator concurrently performs independent work, with CIR first-use barrier and typed effect/state commit. The model mathematics stays identical; missing NPU gets GPU-only fallback.

New TaskProgram contract covers noncausal encoder, speech/video output and iterative image/diffusion/vocoder stages without twisting them into the LM autoregressive scheduler. All such tasks are planned extensions with independent acceptance, not functional now. **Severe caveat:** source author reports hard GPU+NPU host freezes when fabric clocks change; NPU hardware qualification and explicit opt-in are essential, never root system tuning or silent auto-selection. A Rust-authored XDNA AIE kernel is not yet proven; the supplied fork's Python/C++ kernel cannot be imported as if it satisfied Rust-only implementation. See two new NPU/Gufo docs under this folder.

## v4 refined product scope and minimality principle

**Primary goal:** one extremely capable Rust-owned AI computation execution spine that permits specialization of closed mathematical regions and *trusted task-specific stage control* without duplicating model loaders, allocators, sampler state, server or device backends. Usable inference for standard known models first, then hybrid/sparse/MoE and multiple task modalities (speech, image, video, embeddings), with **future optional** gradient/parameter-update tasks. Do not build a generic workflow/agent permission runtime or a mandatory training/autograd compiler before a real model vertical slice.

**Minimality test:** only SemanticProgram and compiled ExecutableBundle need enduring execution representations; optional RIR/KIR views and CIR events are local compiler/plan work. Global physical budget with device-local fast pools; one session/shard writer publishes prepared state; GPU-native inner steps reduce host sync where exact semantics allow; one library+binary initial packaging. Complexity earns its place only if a correctness proof or measured whole-task performance/operational need requires it.

**Evidence:** matched model/quant/driver/context/task quality, TTFT/P95/energy/physical memory, source-license integrity and cross-device safety; one faster microkernel isn't proof of a better runtime. `ARCHITECTURE_V4_CRITICAL_REVIEW_2026-10-12.md` contains exhaustive risks and source research, whereas normative interfaces are original project-neutral contracts.
