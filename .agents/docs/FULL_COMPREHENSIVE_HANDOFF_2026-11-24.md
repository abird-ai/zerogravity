# ZeroGravity full comprehensive handoff — 24 Nov 2026

This is the canonical takeover pack for /home/pvl/spaces/abird/src/zerogravity as of **24 Nov 2026**, after the resident HIP projection profiling campaign and after the previously concurrent CUDA FlashInfer + paired-GDN slice was committed and pushed.

It is intended to let a new agent become productive **without reconstructing context from chat history**. It combines:

- the original product and architecture intent;
- the architectural invariants that survived implementation;
- the current repository, host and build operating contract;
- the major implementation and optimization chronology from the beginning;
- current selected paths versus retained dormant primitives;
- accepted and rejected performance decisions and why;
- current benchmark authority;
- live pending work and immediate next steps;
- exact links to every plan, spec and historical handoff;
- a generated inventory of all .agents/docs and .agents/plans material;
- a generated inventory of every currently open TODO item;
- the complete Git commit ledger for the project period.

**Authority rule:** live source, tests, current plan TODOs and executed results outrank dated prose. This file is the cross-window entry point, not permission to ignore current source. If source and this document differ, inspect source and the owning TODO, reconcile the discrepancy, then update this handoff.

Historical authority is chained, not discarded:

1. this file — current 24 Nov takeover state;
2. CURRENT_HANDOFF.md — maintained detailed current map;
3. FULL_COMPREHENSIVE_HANDOFF_2026-11-23.md — complete post-22-Nov product/build/provider delta;
4. FULL_COMPREHENSIVE_HANDOFF_2026-11-22.md — very large complete 22 Nov implementation/evidence archive;
5. the 21/20/18/15/14 Nov packs preserve earlier implementation detail;
6. the October master/specs and Git history preserve the initial design and every superseded implementation checkpoint.

---

## 0. Executive takeover snapshot

### Exact Git state at handoff creation

- Branch: main.
- Local HEAD: e1929762a87b1e34de099d9f40b85bcff385c27d — docs: record projection traversal A/B.
- Tracked origin/main: exact same e1929762a87b1e34de099d9f40b85bcff385c27d.
- Worktree: clean at handoff start.
- Commit signing is mandatory; current signatures use ED25519 key SHA256:4aB+NvRb1shCx4xjtZDr2Y7FzfHGph80eFtpF+pHuCI.
- The local environment lacks the optional allowed_signers principal lookup; git log --show-signature still reports the cryptographic signature as Good.
- Push is never implied by implementation/commit requests. Push only on explicit user authorization.
- HTTPS GitHub push may fail on this host because the local CA chain is absent; the already-authorized SSH path works with normal host-key verification using StrictHostKeyChecking=accept-new in an isolated connector call.

### Host/source authority

- **L5 (dotlink---l5) is the only source-authoring host.**
- Canonical repo: /home/pvl/spaces/abird/src/zerogravity.
- main is the only persistent local branch/worktree.
- Do not create durable side worktrees or branches for normal implementation, profiling or review. Bounded external scratch directories are fine; remove them at checkpoint.
- **A1 (dotlink---a1) is execution/testing only.** Current exact source at this handoff is e192976….
  - Radeon 890M / gfx1150 / wave32 HIP path.
  - RTX 4060 Laptop SM89, still not a qualified production CUDA target.
  - canonical 0.8B Qwen cache.
  - 2B cache payload incomplete; do not transfer gigabytes merely for a canary.
- **X2 (dotlink---x2) is execution/testing only** for large-VRAM AMD/ROCm. Revalidate its exact checkout, KFD owners and current VRAM before any physical work. It may be shared with unrelated workloads; never kill/reclaim other owners.
- Move exact L5 commits to A1/X2 by **verified Git bundle** for testing. Do not push merely as a transport mechanism.

### Build/runtime defaults

Use, unless a concrete resource constraint requires less:

~~~
export CARGO_BUILD_JOBS=8
export RUSTC_WRAPPER=/run/current-system/sw/bin/kache
~~~

Physical GPU commands go through:

~~~
python3 -B scripts/native-gpu.py -- <command>
~~~

When the owning gate requires network isolation, preserve the accepted private network envelope and --require-private-network; do not weaken sandbox, TLS, SSH, device or ownership rules to make a gate pass.

Use python3 -B / PYTHONDONTWRITEBYTECODE=1 for read-only Python probes.

### Current source composition

The previously concurrent CUDA dirty slice is no longer dirty. At explicit owner request it was committed and pushed as:

**6d5dc72d3711cd3b0e14db0a3c007851984a0745 — feat(cuda): integrate flashinfer decode and paired gdn state**

It includes:

- AttentionProvider::FlashInferDecode support;
- CubeCL CUDA build/runtime bridge for FlashInfer;
- prepared attention composition intended to use FA2/prefill and FlashInfer/decode;
- paired GDN state import/export primitives state_pair_in.cu and state_pair_out_certified.cu;
- associated resident CUDA forward wiring and Nix/Cargo feature dependencies.

**Important:** this conversation committed the whole owner slice because the user explicitly asked to commit all and push. It did **not** run a fresh matched CUDA physical qualification campaign afterward. Treat the source as landed, not as a new accepted performance claim. Revalidate the owning CUDA correctness/provider cells before using it as benchmark authority.

### Immediate current HIP state

The selected gfx1150 resident Qwen path is intentionally conservative:

- c5b3fad — qualified gfx1150 native module allowance, enabling resident weights under the exact 14 GiB envelope;
- 4070610 — singleton projection writes directly to final destination;
- per-matrix projection input packing remains selected;
- canonical rows1 exact CPU64-style BF16 projection remains selected;
- ffd5ff8 — residual materialization fused into RMS first squares pass;
- a5c57c — cached GDN Q/K normalization uses one plane32 workgroup per value head;
- recurrent/private-state experimental cohort primitives are retained but not selected where their A/B lost;
- all newer projection alternatives listed below are dormant.

A1 has **no matched vLLM baseline**. Do not compare raw A1 latency against the X2 or L5 vLLM numbers as if they were matched hardware.

### Immediate next implementation question

Before another HIP projection micro-optimization:

1. inspect the authenticated source / resident matrix preparation path;
2. determine whether it **already proves the exact resident BF16 weight payload is finite**;
3. if and only if that proof already exists and is tied to the same source/matrix identity consumed by projection, add a reusable exact **weight-certified GEMM sibling** that removes only the repeated weight-exponent check from the K64 hot loop;
4. preserve every FMA chain, reduction order, output rounding and status contract;
5. direct-time 3584x1024 and 6144x1024 first;
6. only run output16 whole-model A/B if device time materially wins;
7. if no existing weight-finiteness proof exists, **do not create a new verifier** merely for this optimization; move to simple packed-word address/index hoisting.

---

## 1. Read first — exact bootstrap order

A new agent should read in this order:

1. AGENTS.md
2. .agents/plans/README.md
3. this file
4. .agents/docs/CURRENT_HANDOFF.md
5. .agents/docs/CURRENT_BASELINE.md
6. .agents/plans/TODO.md
7. .agents/plans/model-execution/TODO.md
8. .agents/plans/serving-and-speculation/TODO.md
9. .agents/plans/numerics-and-context/CORRECTNESS_POLICY.md
10. .agents/plans/numerics-and-context/TODO.md
11. .agents/plans/memory-and-streaming/TODO.md
12. .agents/plans/pipeline-composition/SHARED_EXECUTION_ARCHITECTURE.md
13. .agents/plans/pipeline-composition/PROFILE_POLICY_AND_COMPONENT_REUSE.md
14. .agents/plans/pipeline-composition/TODO.md
15. .agents/plans/cpu-gpu-optimization/TODO.md
16. .agents/plans/ZEROGRAVITY_MASTER_PLAN_2026-10-10.md
17. the subsystem specs linked from §15 and Appendix A as needed.

For a question involving a specific historical experiment, search the owning TODO and the dated handoff chain before rerunning it. The huge 21–22 Nov handoffs are archives of detailed evidence and decisions; they are intentionally not copied verbatim into this file.

---

## 2. Owner direction / working style

These are durable owner preferences and should be treated as engineering rules:

- favor **elegant, simple, clear, intuitive** designs;
- ZeroGravity is a composable Rust library/runtime, not a single-model benchmark;
- preserve reusable, model-neutral primitives even when one production selector loses an A/B;
- reject the **consumer/selector**, not a correct reusable kernel/layout/transport;
- delete a primitive only when it is wrong/unsafe, violates numerical/ownership/resource contracts, or is strictly superseded by an equivalent retained primitive;
- favor zero-cost abstractions and compile-time specialization where clean;
- optimize aggressively where the fast design is also simple and maintainable;
- otherwise prefer clean/simple first, with targeted optimization in genuinely hot paths;
- do not add architecture merely to reduce a benchmark counter;
- do not treat lower launch count as performance proof;
- do not overengineer test/evidence granularity, fixture matrices, SHA pinning or per-helper acceptance administration;
- when implementation is complete enough to compare against the real baseline, prefer fast real-model/vLLM feedback over another layer of test scaffolding;
- use existing tests/profilers where possible; temporary probes belong outside the tracked tree and should be removed after the decision;
- implementation turns should aim for a coherent checkpoint rather than endless micro-iterations; historically the owner asked to keep each model turn around ~30 minutes and pause at the next coherent checkpoint;
- do not push unless explicitly asked;
- preserve unrelated work and concurrent ownership;
- signed commits only.

The durable experiment-retention rule is especially important:

> A negative A/B on one model/shape/backend/device rejects that production selection. It does not justify throwing away a clean reusable primitive that may win on another model, backend, geometry or future device.

---

## 3. Original mission and design baseline

The original mission in .agents/plans/ZEROGRAVITY_MASTER_PLAN_2026-10-10.md is still the governing product argument:

> Build the best practical Rust-owned modular AI execution runtime across model/task breadth, source/trained math fidelity, latency/throughput, physical resource efficiency, security, operations and evolvability.

The central design is **one fast execution spine**, not a family of disconnected engines:

- model mathematics are defined once with explicit effects and reference semantics;
- tasks define stage control and output, not device-specific execution;
- a compiler/provider layer substitutes closed semantically legal regions with qualified kernels, captured graphs, native device programs or dataflow regions;
- physical capabilities/resources are explicit;
- every task session has one logical publication authority;
- model loader, scheduler, service, output delivery and resource ownership are shared across specializations.

### Enduring high-level pipeline

~~~
Trusted source/artifacts + ModelDefinition
                 |
          Semantic model plan
       math + typed state effects
                 |
     legal region/provider selection
                 |
      physical command/resource plan
                 |
       prepared executable program
                 |
  ResourceArbiter + device-local owners
                 |
  Task/Resident coordinator + workers
                 |
     provisional results + readiness
                 |
       prepare all required domains
                 |
          publish one StateRoot
                 |
       committed result / stream
~~~

Two conceptual representations remain the right mental model:

1. semantic/model/task intent;
2. prepared physical execution.

Intermediate transformation/provider views are tools, not independent runtime ecosystems.

The scope is inference-first but task/modality general: autoregressive text, encoders/rerankers, ASR/TTS, image/video iterative generation, multimodal programs, with future training seams only where they do not bloat the inference hot path.

---

## 4. Non-negotiable architecture

### 4.1 One logical publication authority

StateRoot is the semantic visibility boundary.

Device work can be ready, failed, cancelled or completion-unknown. It does not independently publish session-visible state. Required domains prepare provisional updates; one coordinator publishes the new root at a legal frontier.

This applies to:

- KV;
- recurrent/GDN state;
- sparse/indexer state;
- RNG/sampling/grammar;
- adapters;
- media/latent/timestep state;
- future stateful domains.

Cancellation may discard a completed candidate but must not fabricate completion. Unknown physical completion must retain/quarantine ownership until safe.

### 4.2 Physical ownership is separate from logical state

ResourceArbiter, ResidentService, resident leases and canonical physical identity own physical resource safety.

DeviceBufferIdentity / canonical backing identity is the physical ownership truth; backend pointer geometry is not a substitute for ownership identity.

Exact handle/view ranges are still legitimate for kernel binding geometry.

Never recycle storage merely because the host timed out if the device may still touch it.

### 4.3 One scheduler / one state engine

Do not introduce per-provider or per-model schedulers.

Current serving concepts include:

- fair admission/work queues;
- active populations;
- refill between decode rounds;
- physical cohorts B1/B2/B4/B8 where qualified;
- partial regrouping such as B2;
- same-frontier/ragged admission rules where qualified;
- independent session/root ownership even when physical work is fused.

A wider physical cohort is not a new semantic batch identity.

### 4.4 Source identity and numerical contracts are explicit

Never infer support from tensor shapes alone.

Source/checkpoint, tensor role, codec, quantization, numerical profile/contract, position/adapter/session compatibility and backend capability remain explicit.

There is no universal numerical epsilon. Each operator/profile owns its contract.

Changed arithmetic/reassociation/precision is a new qualification scope, not a silent optimization.

### 4.5 Prepared execution is not semantic authority

Prepared kernels/native modules/provider selections may be immutable and cached, but they do not own:

- model identity;
- session identity;
- state publication;
- cancellation truth;
- source semantics;
- root/version authority.

Frontend/config/CLI state is never runtime truth merely because it selected a prepared implementation.

### 4.6 Paged/COW/cache correctness

Paged state, prefix reuse and COW must preserve:

- exact source/profile/domain/schema identity;
- committed causal frontier;
- immutable prior snapshots;
- correct future-zero tails where required;
- no stale-generation / ABA reuse;
- event/reader lifetime;
- exact root publication.

Paged transport optimizations may change physical movement, never publication authority.

### 4.7 Production vs qualification separation

Production libraries must not depend on test helpers or benchmark fixtures.

Keep reusable execution primitives in shared code; keep fixture prompts, expected outputs, comparison matrices and fault orchestration in test/qualification support.

A feature flag is not enough to make test-only architecture production-safe.

---

## 5. Current Rust workspace and ownership boundaries

Top-level crates:

- zerogravity-core — shared semantic/resource/state contracts;
- zerogravity-artifacts — authenticated source/cache/artifact handling;
- zerogravity-model-qwen35 — Qwen3.5 source/model semantics;
- zerogravity-model-gemma4 — Gemma 4 semantics;
- zerogravity-cpu — CPU execution primitives;
- zerogravity-gpu — GPU backends, source primitives, resident execution, CUDA/HIP/Vulkan providers and physical ownership;
- zerogravity-engine — scheduling/runtime coordination;
- zerogravity-serving — serving protocol/runtime layer;
- zerogravity — public SDK/facade/product composition;
- zerogravity-cli — single-binary CLI product surface.

The workspace is intentionally split where unsafe/toolchain/feature boundaries justify it; this is not an invitation to fragment the runtime into duplicated engines.

Model crates own model semantics. Shared arithmetic/transport/state/resource primitives remain model-neutral where possible.

---

## 6. Product and interface direction

The product direction established in the October master plan and refined on 23 Nov remains:

- one normal CLI;
- one model/source resolver;
- native model lifecycle/download/cache management;
- typed config/profile documents;
- one task/runtime facade;
- OpenAI-compatible serving where semantics are actually implemented;
- TUI and SolidJS Web control surfaces over the same runtime/config state;
- no separate hidden server/runtime per model;
- no opaque provider silently changing source model, quant, adapter or numerical mode.

The canonical user-facing goal is simple while the internals stay explicit:

~~~
zg run <model> ...
zg serve ...
zg model ...
zg device ...
zg config ...
zg profile ...
~~~

Exact commands and current implementation status live in:

- .agents/plans/cli-ux/NEXT_ITERATION_CLI_UX_AND_MODEL_LIFECYCLE.md
- .agents/plans/cli-device-routing/README.md
- .agents/docs/ZERO_CONFIG_CLI_CONFIG_AND_OPENAI_API_CONTRACT.md
- .agents/docs/RUNTIME_SERVING_AND_UX.md
- .agents/docs/MODEL_ROUTING_RESIDENCY_AND_MULTI_MODEL_SERVING_CONTRACT.md

Design intent is not implementation evidence. Unsupported product paths must refuse truthfully.

---

## 7. Model/backend scope and discipline

### Qwen

Qwen is the main implementation/performance target because it exercises dense projection, GDN/recurrent state, full attention, KV, exact source math, state publication, batching/cohorts and 0.8B / 2B / 27B memory regimes.

Current target families include Qwen3.5+, Qwen3.8 and later family work, but model names must not leak into shared kernel/runtime abstractions.

### Gemma

Gemma 4 source/model contracts and E2B/other qualification work exist, but do not infer parity from Qwen. The current handoff/plans keep actual trained-source and backend acceptance scopes distinct.

### CUDA

L5 RTX 3060 remains the main current CUDA performance/correctness host.

The 22 Nov baseline says C1 is accepted for cross-provider comparison while fused C4+ identical-request lane invariance was still a correctness blocker. The newly landed 6d5dc72 provider/state slice may alter relevant code. Revalidate the current source before declaring that blocker fixed or obsolete.

### HIP/ROCm

- X2 gfx1151 is the matched large-VRAM ROCm/vLLM host.
- A1 gfx1150 is the fast current resident-kernel experimentation host.
- gfx1150 has a separately qualified native module opaque allowance.
- Do **not** apply gfx1150 accounting to gfx1151 or the generic fallback without an independent qualification.

### Vulkan

Vulkan remains a planned/partially implemented backend whose parity and release gates are tracked separately. CUDA/HIP success does not establish Vulkan support.

### NPU / distributed / multimodal

The architecture intentionally leaves seams for NPU, distributed placement, ASR/TTS/image/video and external task services, but those are separately gated. Do not convert architectural compatibility into a support claim.

---

## 8. Architecture evolution from the beginning

This section is the human-readable chronology. Appendix C contains the complete commit ledger so no implementation step disappears.

### 8.1 10–13 Oct — architecture consolidation

The project converged on the v4/v4.3 design:

- one semantic execution spine;
- explicit model/task/source/state contracts;
- one logical StateRoot publication;
- physical resource ownership separated from semantic state;
- region/provider substitution rather than per-hardware engines;
- CPU/reference semantics as correctness authority;
- future modality/task breadth without building a generic privileged workflow VM.

The master architecture, minimal execution spine, task/compiler refinement, resource/memory, routing, API and security specs were written during this phase.

### 8.2 13–18 Oct — immutable source identity, artifacts and model semantics

Key implementation themes:

- immutable source identity;
- authenticated cache/snapshot handling;
- resumable/pinned acquisition;
- safetensors/source shard binding;
- Qwen3.5 model config/tensor/tokenizer semantics;
- Gemma 4 model family semantics;
- portable CPU reference math;
- source-bound state and forward plans;
- early typed multimodal/pipeline scaffolding.

The important design boundary established here still holds: **recognizing a file format is not evidence that a model/operator is supported.**

### 8.3 19–22 Oct — serving, native GPU work and state/resource ownership

The runtime gained:

- bounded OpenAI CPU routing;
- resident GPU requests and fair waves;
- native multi-row CUDA primitives;
- batched attention/GDN/KV/elementwise work;
- state-cohort publication;
- quantized projections;
- device-side greedy feedback;
- B2/B4/B8 serving experiments;
- typed source operators;
- exact CPU/source numerical qualification;
- graph/replay experiments;
- explicit physical resource and state ownership.

This period established the durable distinction between **semantic correctness**, **physical readiness**, and **publication**.

### 8.4 23–29 Oct — ordinary source execution and repository normalization

Major transitions:

- one normal CLI/device/source routing surface;
- Linux physical PCI/device-node identity;
- Qwen ordinary CUDA path generalized away from finite captured recipes;
- reusable BF16 GEMM and resident matrix ownership;
- source-exact rotary/attention/state transport;
- ordinary cancellation/context admission;
- exact status/ownership qualification;
- human cache aliases and crash consistency;
- captured routes moved back to qualification support;
- terminology/module ownership normalized.

The 28 Oct ordinary CUDA effort is the bridge from qualification-specific GPU code to the current source-derived resident execution architecture.

### 8.5 29 Oct–3 Nov — Fabric vNext transition

The runtime was simplified around shared ownership:

- source/state owners unified;
- CPU runtime and serving wired through the same engine;
- one shared CPU/GPU scheduler;
- SDK owns run/artifact orchestration;
- numerical/layout contracts moved to shared core;
- catalog and serving load path unified;
- preparation bound to source owners;
- run configuration policy centralized;
- full feedback packets and cold Gemma lifecycle qualification.

Fabric transition source architecture is considered functionally accepted inside its named scopes; unsupported model/provider cells remain explicit.

### 8.6 5–8 Nov — populated HIP and shared source execution

The ordinary Qwen HIP source path became physically runnable, populated serving was qualified, source chunks were reused, output reduction boundaries were fixed, and shared source execution was composed across backends.

This established the current HIP optimization substrate rather than a second HIP model engine.

### 8.7 9–13 Nov — CUDA performance campaign

The CUDA path moved from C1 toward real B2/B4/B8 execution:

- device greedy feedback;
- decode GEMV specialization;
- batched source decode;
- cached GDN conv/recurrent;
- full attention;
- certified BF16 stores;
- output-head weight reuse;
- batched aligned prompt prefill;
- B8 cohorts;
- larger K tiles;
- small-context qualification;
- partial regrouping;
- chunked/ragged prefill;
- mutual cohort compatibility;
- exact private KV/state transitions;
- row-split GDN;
- inner-major B2/B4/B8 head paths;
- stateless row/RMS/gate/QK/rotary primitives;
- private B8 recurrent/KV state;
- B8 private-KV CTA256.

A long series of alternatives were measured and rejected while preserving reusable primitives. This is where the current “retain mechanism, reject selector” rule became especially important.

### 8.8 14–20 Nov — HIP/X2, paged state, graph and serving optimization

Major work included:

- restoring HIP source feature isolation;
- X2 HIP health and 27B resident/host-load analysis;
- paged/COW state implementation and transport optimization;
- refill/interleaving/goodput;
- graph capture/resource lifecycle and private selector experiments;
- matched profile discipline;
- memory accounting and source transport optimization.

Public graph serving remained Native where graph gains did not justify lifecycle complexity.

### 8.9 21–23 Nov — architecture cleanup, build convergence and matched baselines

The project completed broad simplification and convergence work:

- one production feature algebra across CPU/CUDA/HIP/Vulkan/model/serve;
- single public GPU run surface;
- terminology cleanup;
- product CLI/TUI/Web lifecycle design;
- OpenAI-compatible server design;
- matched vLLM baselines on L5/X2;
- CUDA attention provider abstraction + optional FA2;
- fused Q/K normalization+rotary prefill;
- fused recurrent export certification;
- B1 reusable CUDA primitives;
- A1 public HIP canary;
- HIP decode cohort and recurrent experiments.

The 23 Nov comprehensive handoff is the best narrative source for this transition.

### 8.10 23–24 Nov — A1 gfx1150 resident HIP optimization

The 14 GiB gfx1150 accounting breakthrough changed the performance regime.

#### Native module accounting

c5b3fad qualified a **1 GiB** gfx1150 module opaque allowance rather than the generic 8 GiB fallback, based on physical measurement and exact environment revalidation.

That moved Qwen0.8B weights from streamed to resident on A1.

Observed consequences under the same 14 GiB envelope:

- 47+3 dispatches: 17,653 -> 9,817;
- stable 47+3 wall time around 0.53–0.56 s instead of ~6.36 s streamed;
- output16: 73,384 -> 31,592 dispatches;
- output16 warm mean around 0.962 s instead of ~31.35 s streamed.

Do not generalize this allowance to gfx1151 or other HIP devices.

#### Resident hot-path relocalization

Once resident, the old streamed “projection is ~75% by launches” story became obsolete.

Initial resident dispatch localization showed:

- projection 561 launches/token;
- RMS/residual-RMS 489;
- GDN 360;
- QK/rotary 150.

ffd5ff8 fused residual materialization into RMS's first squares pass and improved C4/output16 wall time ~1.1% while preserving the standalone residual primitive.

Broader RMS launch fusions reduced launches substantially but lost wall time:

- scalar_cascade retained dormant;
- fused normalize/weight/store tail retained dormant.

This established a durable rule: **launch count is a poor performance proxy on resident gfx1150.**

#### GDN normalization

a5c57c selected a plane32 cached-GDN Q/K normalization kernel: one 32-lane workgroup per value head, preserving the source reduction structure.

#### Actual GPU-time localization

Temporary serialized HIP device-event profiling ranked real resident device time:

steady cached decode approximately:

- projection: **~19.8–21.5 ms/token (~60–62%)**;
- GDN: **~6.6–7.5 ms (~20–22%)**;
- RMS: **~3.0–3.4 ms (~9–10%)**;
- QK/rotary and each remaining family: roughly ~2% or less.

Prefill had the same order:

- projection ~296 ms;
- GDN ~156 ms.

Projection sub-profile:

- pack ~22.2 ms;
- exact GEMM body ~561.1 ms;
- status ~21.0 ms;
- GEMM body ~92.9% of projection time.

Dominant output16 shapes:

- 3584x1024;
- 6144x1024;
- output head 248320x1024;
- 1024x3584;
- 1024x2048.

#### Projection alternatives retained but not selected

The following are all intentionally reusable/dormant on current gfx1150:

- per-flight prepacked input path;
- coalesced projection status;
- certified in-kernel projection status;
- rows4 shared-input projection;
- rows2 shared-input projection;
- exact wave32 projection;
- paired packed-word lane-shuffle projection;
- input-certified pack + GEMM composition.

Exact recent results:

**Wave32 (072b791 / b4fc231)**

- exact arithmetic tree;
- serialized GEMM 561.1 -> 577.7 ms (~3% slower);
- clean output16 timed mean ~0.955 -> 1.101 s;
- reject selector, retain primitive.

**Rows2 (253a565 / 6629ee2)**

- exact two-row independent 64-partial trees;
- 3584x1024 flipped sign between repeats;
- 1024x3584 ~24% slower;
- head ~4.4% slower;
- other expansions only ~2.6–2.9% better;
- no stable geometry rule; reject selector.

**Paired packed load (55477b4 / 49e5149)**

- adjacent lanes share packed u32 through plane_shuffle_xor;
- same 64 partials/reduction;
- 3584x1024: 65.20 -> 153.73 ms (~2.36x slower);
- 6144x1024: 42.82 -> 87.91 ms (~2.05x slower);
- reject selector.

**Input-certified pack/GEMM (ba8c5ea / 49e5149)**

- projection-specific pack publishes final shared input status once;
- GEMM skips only repeated activation nonfinite exponent check;
- exact FMAs/reduction/output checks retained;
- paired same-commit timing:
  - 3584x1024: 61.786 -> 61.422 ms (~0.6% faster);
  - 6144x1024: 38.045 -> 45.533 ms (~19.7% slower);
- reject selector; no whole-model A/B warranted.

Current canonical rows1 projection remains selected.

---

## 9. Current CUDA state

Before the 24 Nov catch-up commit, the selected ordinary CUDA path included:

- independent exact source reduction contracts;
- B2/B4/B8 physical cohorts;
- same-frontier chunked/ragged prefill;
- K64 Ordered/Rounded/Interleaved projection families;
- exact row-split GDN;
- canonical-boundary private B8 recurrent core;
- ragged private B8 KV;
- B8 private-KV CTA256;
- accepted B4 GDN/attention composition;
- full BF16 head + greedy;
- canonical state/root publication.

Recent 23 Nov work additionally retained/selected:

- Q/K normalization+rotary prefill fusion;
- recurrent export certification;
- B1 direct GDN conv primitive;
- B1 fused head/greedy primitives.

6d5dc72 then landed the previously concurrent source for:

- FlashInfer decode provider bridge;
- FA2/FlashInfer provider composition;
- paired GDN state import/export.

### CUDA correctness/performance caution

CURRENT_BASELINE.md still says:

- C1 is accepted for cross-provider performance comparison;
- fused C4+ identical-request lane invariance was not yet accepted;
- C4+ rows are diagnostic until correctness is fixed.

Because 6d5dc72 landed after that baseline, the next CUDA agent must inspect and run the smallest current reproducer before deciding whether the blocker still exists. Do not silently assume the new provider work fixed it.

Matched vLLM baseline on L5 RTX3060 remains the cross-provider authority unless a new matched campaign supersedes it.

---

## 10. Current HIP/ROCm state

### X2 matched baseline authority

CURRENT_BASELINE.md is still the active matched ROCm/vLLM source:

- matched 24 GiB 0.8B/2B;
- matched 75 GiB 27B;
- exact greedy IDs;
- current baseline table and host conditions.

The older X2 HIP path in that baseline is much slower than vLLM and was still largely physical width1. Do not substitute A1 numbers for X2 matched results.

### A1 gfx1150 optimization authority

A1 is useful for rapid current resident Qwen0.8B physical A/Bs.

Current public canary is green on exact e192976…:

~~~
cargo build --locked --release -p zerogravity-cli \
  --no-default-features --features hip,model-qwen35

python3 -B scripts/native-gpu.py -- \
  target/release/zerogravity run-rocm qwen3.5-0.8b \
    --prompt hello --max-tokens 2 --context 512 --explain
~~~

The KFD owner set must be recorded before and after; the ZeroGravity run must add no lingering owner.

Recent canaries returned to the exact pre-run owner set. Numeric PIDs are not stable across time and are not an identity contract.

### No matched A1 vLLM claim

Do not say A1 “beats” or “loses to” the X2/L5 vLLM table. Different hardware and memory policy make that comparison invalid.

---

## 11. Current performance baseline and interpretation

Read CURRENT_BASELINE.md in full.

Key authority:

### L5 CUDA / vLLM

Matched 0.8B and 2B C1..C64 tables exist.

Only accepted correctness cells should be used for performance claims.

### X2 ROCm / vLLM

Matched small-model 24 GiB and 27B 75 GiB tables exist.

### L5 27B streamed mode

This is a **memory-constrained streaming mode**, not a resident arithmetic baseline.

The ~306 s 3-token result is useful for:

- SSD/source streaming;
- overlap;
- source transport;
- memory admission.

It is not useful as a judgment of resident GPU arithmetic.

### A1 resident output16 local baseline

For the current selected gfx1150 path after residency/GDN normalization and before the dormant projection alternatives:

- clean output16 timed waves around 1.002 s and 0.908 s;
- mean ~0.955 s;
- exact/replay-stable.

Use this only as a local A1 optimization baseline, not a matched vLLM result.

---

## 12. Reusable primitives that must not be lost

This is not exhaustive; search owning TODOs and Git for exact variants.

### Projection

Retain:

- canonical exact CPU64 BF16 projection;
- direct singleton destination path;
- prepacked input alternate;
- coalesced-status alternate;
- certified projection alternate;
- rows2;
- rows4;
- wave32;
- paired packed-word lane-sharing;
- projection-specific input-certified pack;
- input-certified GEMM sibling.

Current Qwen/gfx1150 selects only the canonical rows1/per-matrix-pack path plus direct singleton destination.

### RMS

Retain:

- standalone residual;
- residual_squares;
- shared exact scalar helper;
- four-stage scalar cascade;
- normalize/weight shared helpers;
- fused vector tail alternate.

Current selection uses residual first-pass fusion but not broad cascade/tail fusion.

### GDN/state

Retain:

- canonical state transports;
- private recurrent kernels;
- grouped recurrent transports;
- cached GDN gated primitives;
- paired GDN state import/export;
- Conv4/grouped conv primitives;
- exact-root derived-state cache;
- recurrent cohort alternatives.

### CUDA projection/attention

Retain:

- K64 Ordered/Rounded/Interleaved families;
- inner-major B2/B4/B8 paths;
- shared-input projection pair;
- private ragged attention;
- native FA2 provider;
- FlashInfer decode bridge/provider;
- full head/greedy and retained fused variants.

### KV / paging / scheduling

Retain:

- B4/B8 packed-BF16 KV;
- ragged private KV;
- paged/COW state and paired-page transports;
- refill/regroup/fair-interleaving machinery;
- dormant graph lifecycle/selector primitives where still present.

A production rejection is not a deletion request.

---

## 13. Important rejected or held patterns

Do not casually repeat these without a materially different mechanism or hardware:

- CUDA B16 cohort widening;
- unequal-frontier admission without a new per-lane source qualification;
- persistent derived/private GDN representation that lost whole-model A/B;
- grouped recurrent transport consumer where the primitive win did not translate;
- broad launch-count-driven RMS fusions on resident gfx1150;
- rows4/rows2/wave32 projection selectors on gfx1150;
- cross-lane packed-word shuffle on gfx1150;
- input-certified projection selector on gfx1150;
- graph serving promotion when B4/B8 gains do not justify retained lifecycle cost;
- old eager-vLLM / Podman comparison paths that are not the maintained matched baseline;
- silently generalizing gfx1150 native-memory accounting to gfx1151/other HIP devices;
- debug-profile timing as release-performance evidence;
- extra permanent profiler/evidence frameworks for one optimization question.

---

## 14. Evidence and benchmark discipline

Use the smallest real check that can answer the current question.

Preferred optimization loop:

1. inspect current source and exact selected consumer;
2. identify a real device-time owner;
3. implement a clean reusable primitive;
4. compile once;
5. direct-time the dominant real shapes;
6. if it materially wins, run one real model A/B;
7. preserve exact tokens/state/root/retirement;
8. retain the primitive if correct even when selector loses;
9. remove temporary selector/profiler/runner;
10. update owning TODO/handoff;
11. checkpoint.

Do not:

- infer speed from dispatch count;
- add a fixture matrix before there is a concrete gap;
- turn transient profiling into production architecture;
- compare unmatched devices as if they were one baseline;
- retag incomplete/interrupted runs as passes;
- let cleanup success prove an earlier failed case;
- promote a provider based on microkernel timing alone.

When vLLM is the comparison target, match:

- exact model/checkpoint;
- quant/precision;
- context;
- concurrency;
- memory envelope;
- task/greedy semantics;
- hardware;
- optimized build/runtime profile.

---

## 15. Canonical design/spec navigation

### Core architecture

- VISION_SCOPE_AND_PRINCIPLES.md
- MODULAR_AI_SYSTEMS_REFERENCE_ARCHITECTURE.md
- MINIMAL_EXECUTION_SPINE_AND_PROTOCOL_V4.md
- TASK_PROGRAM_AND_COMPILER_REFINEMENT_V4.md
- SEMANTIC_IR_AND_COMPILER.md
- ENGINE_CORE_EXECUTION_AND_PROTOCOL.md
- COMPOSITION_AND_EXTENSION_CONTRACTS.md

### State / memory / ownership

- STATE_CACHE_AND_TRANSACTIONS.md
- MEMORY_RESIDENCY_AND_IO.md
- UNIFIED_VIRTUAL_TENSOR_RESIDENCY_AND_SSD_STREAMING_2026-10-15.md
- ASYNC_RESOURCE_COMMAND_IR.md
- INVARIANTS_AND_FAILURE_SEMANTICS.md

### Device / providers / performance

- ACCELERATOR_EXECUTION_FAMILIES_AND_BACKEND_CONTRACT.md
- DEVICE_KERNEL_AND_CODEGEN.md
- OPERATOR_AND_FUSION_CATALOG.md
- COST_MODEL_AND_AUTOTUNING.md
- PERFORMANCE_HYPOTHESES_AND_RISKS.md
- NUMERICS_QUALITY_AND_BENCHMARKS.md

### Model/source/quantization

- MODEL_SEMANTICS_AND_COMPATIBILITY.md
- FORMATS_QUANTIZATION_AND_LOADING.md
- DERIVED_QUANTIZATION_CONTRACT_2026-10-15.md
- VERIFIED_MODEL_ASSEMBLIES_AND_SPECIALIZATION_V5_1_2026-10-15.md

### Serving / routing / API

- MODEL_ROUTING_RESIDENCY_AND_MULTI_MODEL_SERVING_CONTRACT.md
- SCHEDULER_BATCHING_AND_SPECULATION.md
- RUNTIME_SERVING_AND_UX.md
- ZERO_CONFIG_CLI_CONFIG_AND_OPENAI_API_CONTRACT.md

### Distributed / heterogeneous / multimodal

- DISTRIBUTED_PLACEMENT_AND_NETWORK_EXECUTION_CONTRACT.md
- REMOTE_PEER_TRANSPORT_AND_STATE_HANDOFF_PROTOCOL.md
- DISTRIBUTED_PARALLELISM.md
- NPU_GPU_COOPERATIVE_EXECUTION_AND_TASK_PIPELINES.md
- UNIFIED_MULTIMODAL_EXECUTION_CONTRACT_V5_2026-10-14.md
- UNIVERSAL_MULTIMODAL_EXECUTION_ARCHITECTURE_V5_1_2026-10-15.md
- REALTIME_MULTIMODAL_SERVING_AND_QOS_V5_1_2026-10-15.md
- EXTREME_HYBRID_MODELS_AND_SPECULATIVE_DRAFTER_SEMANTICS_2026-10-15.md

### Live implementation plans

- .agents/plans/TODO.md
- .agents/plans/model-execution/TODO.md
- .agents/plans/serving-and-speculation/TODO.md
- .agents/plans/numerics-and-context/TODO.md
- .agents/plans/memory-and-streaming/TODO.md
- .agents/plans/cpu-gpu-optimization/TODO.md
- .agents/plans/pipeline-composition/TODO.md
- .agents/plans/model-acquisition/TODO.md
- .agents/plans/cli-device-routing/TODO.md
- .agents/plans/platform-and-release/TODO.md
- .agents/plans/distributed-execution/TODO.md
- .agents/plans/multimodal/TODO.md

Appendix A inventories every current file so no plan/spec is silently omitted.

---

## 16. Current live work map

The shared plan queue is authoritative. At handoff creation the top-level active work still includes major ongoing efforts such as:

- serving-and-speculation;
- model execution;
- numerical/context closure;
- memory/streaming;
- CPU/GPU optimization;
- pipeline composition;
- model acquisition;
- CLI/device routing;
- platform/release;
- distributed execution;
- multimodal work.

Do not infer priority from this paragraph alone. Read .agents/plans/TODO.md and the owning plan TODO.

Appendix B contains every currently unchecked [ ], active [~] and blocked [!] TODO line across the plan tree with file+line location. It is generated at this handoff so unfinished work is not lost.

---

## 17. Immediate next-agent procedure

When taking over:

1. revalidate git status, HEAD, origin/main;
2. verify no other process/agent is writing the target files;
3. read AGENTS.md, this file, current handoff, baseline and owning TODO;
4. check A1/X2 exact source before executing there;
5. for the current HIP projection continuation:
   - inspect VerifiedMatrix, Bf16MatrixSource, ResidentMatrix preparation/upload;
   - determine whether finiteness is already authenticated for the exact BF16 weight payload;
   - do **not** add a verifier if none exists;
   - if proof exists, add a dormant reusable weight-certified projection sibling;
   - direct-time 3584x1024 / 6144x1024;
   - only then consider output16 A/B;
6. separately, before CUDA performance claims, revalidate the landed 6d5dc72 FlashInfer/paired-GDN source and the current C4+ identical-lane correctness cell;
7. preserve all dormant reusable primitives;
8. use A1/X2 only for execution;
9. checkpoint a coherent source+plan update;
10. do not push unless the user explicitly asks.

---

## 18. Do not redo without new evidence

Do not spend another turn rediscovering that:

- gfx1150 weights can be resident under the qualified 1 GiB native-module allowance;
- broad launch-count reduction is not a reliable resident-performance predictor;
- wave32 exact projection loses;
- rows2/rows4 grouping does not have a robust gfx1150 selector;
- lane-shuffle packed-word sharing is much slower;
- moving input nonfinite certification out of GEMM is not a useful general selector;
- A1 has no matched vLLM baseline;
- X2/L5 matched vLLM numbers are different hardware scopes;
- the recurrent-cohort selector lost its prior gfx1150 A/B even though primitives are retained;
- CUDA C4+ baseline correctness must be revalidated before performance promotion;
- the 27B L5 streamed path is a streaming/memory mode, not resident arithmetic evidence.

---

## 19. Starting prompt for the next agent

Copy/paste this prompt into the next agent:

~~~
You are taking over ZeroGravity at /home/pvl/spaces/abird/src/zerogravity.

Read, in order:
1. AGENTS.md
2. .agents/plans/README.md
3. .agents/docs/FULL_COMPREHENSIVE_HANDOFF_2026-11-24.md
4. .agents/docs/CURRENT_HANDOFF.md
5. .agents/docs/CURRENT_BASELINE.md
6. .agents/plans/TODO.md
7. .agents/plans/model-execution/TODO.md
8. .agents/plans/serving-and-speculation/TODO.md
9. the owning specs linked from the handoff.

Revalidate actual Git/source state before editing. L5 is the only source-authoring
host; main is the only persistent worktree/branch. A1 and X2 are execution/testing
only. Move exact L5 commits by verified Git bundle. Do not push unless I explicitly
ask.

Engineering rules:
- elegant, simple, clear, intuitive Rust;
- zero-cost abstractions where practical;
- optimize real hot paths when the fast design stays maintainable;
- otherwise prefer clean/simple first;
- no overengineered fixture/evidence/pinning bureaucracy;
- prefer real-model and matched-vLLM feedback once implementation is complete enough;
- preserve exact source/numerical/state/ownership/resource semantics;
- preserve reusable model-neutral primitives even when one selector loses;
- reject losing consumer/selector, not a correct reusable primitive;
- remove temporary probes/selectors/runners after the decision.

Current exact handoff HEAD is e1929762a87b1e34de099d9f40b85bcff385c27d,
which is also origin/main at handoff creation.

Immediate HIP continuation:
- current gfx1150 rows1 projection remains selected;
- projection is ~60–62% of steady resident decode device time and the exact GEMM
  body is ~92.9% of projection time;
- wave32, rows2, rows4, paired packed-load, prepacked/coalesced/certified and
  input-certified alternatives are retained dormant;
- inspect existing authenticated source/resident weight preparation and determine
  whether it already proves BF16 weight finiteness for the exact resident matrix;
- only if that proof already exists, add a reusable weight-certified GEMM sibling
  that removes only the repeated weight exponent check while preserving all64 FMA
  chains/reduction/rounding/status semantics;
- direct-time 3584x1024 and 6144x1024 first;
- only run output16 whole-model A/B if device time materially wins;
- if no existing proof exists, do not add a new verifier just for this optimization;
  move to simple packed-word address/index hoisting.

CUDA caution:
6d5dc72 landed the previously concurrent FlashInfer-decode + paired-GDN-state
source and was explicitly committed/pushed as a whole. This handoff does not claim
fresh physical CUDA acceptance after that landing. Revalidate the smallest current
provider/correctness cells, including the C4+ identical-lane issue from
CURRENT_BASELINE, before making new CUDA performance claims.

Aim for one coherent implementation/checkpoint this turn and pause there.
~~~

---

## 20. Handoff maintenance rules

When the next substantial phase lands:

- update the owning plan TODO first;
- keep CURRENT_HANDOFF.md as the maintained current map;
- if another full dated takeover is warranted, chain it to this file rather than copying multi-megabyte historical bodies;
- keep historical packs intact;
- do not create a parallel evidence ledger;
- let Git remain the exact implementation chronology;
- append new reusable/rejected primitive decisions so future agents do not repeat dead experiments.

---

## Appendix A — exhaustive .agents/docs / .agents/plans inventory

Generated from the repository at handoff creation. Listing a file does not make
dated prose more authoritative than live source/current TODOs; it guarantees
the design/work item remains discoverable.

| File | Bytes | H1 / purpose hint |
| --- | ---: | --- |
| `.agents/docs/ACCELERATOR_EXECUTION_FAMILIES_AND_BACKEND_CONTRACT.md` | 23581 | zerogravity — Universal region-execution backend contract (v4.1) |
| `.agents/docs/ASYNC_RESOURCE_COMMAND_IR.md` | 19879 | zerogravity — Async Command/Resource IR (CIR), ownership and compiled execution |
| `.agents/docs/COMPOSITION_AND_EXTENSION_CONTRACTS.md` | 23822 | zerogravity — v4.1 typed composition and region execution contracts |
| `.agents/docs/COST_MODEL_AND_AUTOTUNING.md` | 24473 | Joint cost model, online adaptation and compiler autotuning |
| `.agents/docs/CURRENT_BASELINE.md` | 5718 | Current performance baseline — 22 Nov 2026 |
| `.agents/docs/CURRENT_HANDOFF.md` | 181924 | ZeroGravity current comprehensive handoff — 24 Nov 2026 |
| `.agents/docs/DERIVED_QUANTIZATION_CONTRACT_2026-10-15.md` | 14326 | Zerogravity — EXL3/QTIP and Source-Qualified Derived Quantization Contract |
| `.agents/docs/DEVICE_KERNEL_AND_CODEGEN.md` | 23101 | Device abstraction, Rust GPU kernels, code generation and backend specialization |
| `.agents/docs/DISTRIBUTED_PARALLELISM.md` | 7780 | Distributed and heterogeneous execution — v4.2 authoritative index |
| `.agents/docs/DISTRIBUTED_PLACEMENT_AND_NETWORK_EXECUTION_CONTRACT.md` | 37796 | zerogravity — Distributed placement, network and state execution contract (v4.2) |
| `.agents/docs/ENGINE_CORE_EXECUTION_AND_PROTOCOL.md` | 23782 | zerogravity — Native Engine Core, Worker Protocol, KV policies and request lifecycles |
| `.agents/docs/EXAMPLE_EXECUTION_TRACES.md` | 19718 | Design walkthroughs — concrete execution and fallback traces |
| `.agents/docs/EXTREME_HYBRID_MODELS_AND_SPECULATIVE_DRAFTER_SEMANTICS_2026-10-15.md` | 17081 | Zerogravity — Extreme Hybrid Model Recipes, Native MTP and DFlash Source Contracts |
| `.agents/docs/FORMATS_QUANTIZATION_AND_LOADING.md` | 15167 | Model artifacts, quantization, tokenizer/template parity and safe loading |
| `.agents/docs/FULL_COMPREHENSIVE_HANDOFF_2026-11-14.md` | 86952 | ZeroGravity full comprehensive handoff — 14 Nov 2026 |
| `.agents/docs/FULL_COMPREHENSIVE_HANDOFF_2026-11-15.md` | 131347 | ZeroGravity full comprehensive handoff — 15 Nov 2026 |
| `.agents/docs/FULL_COMPREHENSIVE_HANDOFF_2026-11-18.md` | 265825 | ZeroGravity full comprehensive handoff — 18 Nov 2026 |
| `.agents/docs/FULL_COMPREHENSIVE_HANDOFF_2026-11-20.md` | 401084 | ZeroGravity full comprehensive handoff — 20 Nov 2026 |
| `.agents/docs/FULL_COMPREHENSIVE_HANDOFF_2026-11-21.md` | 1570993 | ZeroGravity full comprehensive handoff — 21 Nov 2026 |
| `.agents/docs/FULL_COMPREHENSIVE_HANDOFF_2026-11-22.md` | 2471529 | ZeroGravity full comprehensive handoff — 22 Nov 2026 |
| `.agents/docs/FULL_COMPREHENSIVE_HANDOFF_2026-11-23.md` | 40354 | ZeroGravity full comprehensive handoff — 23 Nov 2026 |
| `.agents/docs/FULL_COMPREHENSIVE_HANDOFF_2026-11-24.md` | 223761 | ZeroGravity full comprehensive handoff — 24 Nov 2026 |
| `.agents/docs/GFX1151_ROCM_KERNEL_COST_AND_UNIFIED_MEMORY_SAFETY_CONTRACT_2026-10-15.md` | 20317 | Zerogravity — gfx1151 ROCm Kernel Specialization, MoE Scheduling and Unified-Memory Safety Contract |
| `.agents/docs/HETEROGENEOUS_PREFILL_AND_RETAINED_REPLAY_CONTRACT_2026-10-15.md` | 18090 | Zerogravity — Heterogeneous prefill and retained replay contract |
| `.agents/docs/INVARIANTS_AND_FAILURE_SEMANTICS.md` | 20079 | Cross-cutting invariants, state lifetimes and failure semantics |
| `.agents/docs/JOINT_CONCURRENT_SERVING_OPTIMIZATION_AND_GOODPUT_CONTRACT_2026-10-15.md` | 22925 | Zerogravity — Joint Concurrent Serving, Goodput and Combinatory Specialization Contract |
| `.agents/docs/LINUX_DEV_NODE_SELECTORS_2026-10-25.md` | 2406 | Linux GPU device-node selectors |
| `.agents/docs/MEMORY_RESIDENCY_AND_IO.md` | 29375 | Global memory, expert residency, storage I/O and resource admission |
| `.agents/docs/MINIMAL_EXECUTION_SPINE_AND_PROTOCOL_V4.md` | 27943 | Minimal execution spine v4 — state, asynchronous workers, resource ownership and hot-path performance |
| `.agents/docs/MODEL_ROUTING_RESIDENCY_AND_MULTI_MODEL_SERVING_CONTRACT.md` | 33807 | zerogravity — Built-in model router, multi-model residency and serving contract (v4.3) |
| `.agents/docs/MODEL_SEMANTICS_AND_COMPATIBILITY.md` | 19641 | Model semantics, model adapters and compatibility qualification |
| `.agents/docs/MODULAR_AI_SYSTEMS_REFERENCE_ARCHITECTURE.md` | 24661 | zerogravity — modular AI execution reference architecture v4.1 |
| `.agents/docs/NPU_GPU_COOPERATIVE_EXECUTION_AND_TASK_PIPELINES.md` | 25980 | zerogravity — NPU/GPU cooperative execution and multimodal TaskProgram design |
| `.agents/docs/NUMERICS_QUALITY_AND_BENCHMARKS.md` | 31132 | Numerical correctness, model-quality evaluation and benchmark methodology |
| `.agents/docs/OPERATOR_AND_FUSION_CATALOG.md` | 19814 | Operator/region contracts, fusion patterns and model execution catalog |
| `.agents/docs/PERFORMANCE_HYPOTHESES_AND_RISKS.md` | 18664 | Architectural risks, opportunities, trade-offs and falsification criteria |
| `.agents/docs/PRODUCT_MODEL_PIPELINE_AND_EXTENSIONS.md` | 20666 | zerogravity — Product, model loading and extension contract across the full pipeline |
| `.agents/docs/REALTIME_MULTIMODAL_SERVING_AND_QOS_V5_1_2026-10-15.md` | 31781 | Zerogravity Real-Time Multimodal Serving, Sessions, QoS and API Contract — v5.1 |
| `.agents/docs/REMOTE_PEER_TRANSPORT_AND_STATE_HANDOFF_PROTOCOL.md` | 16163 | zerogravity — Remote peer transport, state handoff and failure protocol (v4.2) |
| `.agents/docs/RUNTIME_SERVING_AND_UX.md` | 18788 | Runtime, APIs, developer experience and observability |
| `.agents/docs/RUST_WORKSPACE_AND_ENGINEERING.md` | 25936 | Rust workspace, interfaces, implementation practices and contributor workflow |
| `.agents/docs/SCHEDULER_BATCHING_AND_SPECULATION.md` | 24878 | Scheduler, continuous batching, prefix reuse and speculative execution |
| `.agents/docs/SECURITY_AND_RELIABILITY.md` | 23521 | Security, reliability, resilience and safe defaults |
| `.agents/docs/SEMANTIC_IR_AND_COMPILER.md` | 23564 | Semantic IR, state-effect system and ahead-of-time/JIT execution planning |
| `.agents/docs/STATE_CACHE_AND_TRANSACTIONS.md` | 23890 | Session state, prefix caching, snapshots and transactional semantics |
| `.agents/docs/TASK_PROGRAM_AND_COMPILER_REFINEMENT_V4.md` | 24409 | v4 TaskProgram and compiler contracts — high throughput with a small semantic core |
| `.agents/docs/TERMINOLOGY.md` | 7577 | ZeroGravity terminology |
| `.agents/docs/UNIFIED_MULTIMODAL_EXECUTION_CONTRACT_V5_2026-10-14.md` | 21748 | Zerogravity — Unified multimodal execution contracts (v5 design proposal) |
| `.agents/docs/UNIFIED_VIRTUAL_TENSOR_RESIDENCY_AND_SSD_STREAMING_2026-10-15.md` | 22533 | Zerogravity — First-Class Virtual Tensor Residency and High-Performance SSD Streaming |
| `.agents/docs/UNIVERSAL_MULTIMODAL_EXECUTION_ARCHITECTURE_V5_1_2026-10-15.md` | 28589 | Zerogravity Universal Multimodal Execution Architecture — v5.1 Design Specification |
| `.agents/docs/VERIFIED_MODEL_ASSEMBLIES_AND_SPECIALIZATION_V5_1_2026-10-15.md` | 19704 | Zerogravity Verified Model Assemblies, Source Codecs and Legal Specialization — v5.1 |
| `.agents/docs/VISION_SCOPE_AND_PRINCIPLES.md` | 14140 | Vision, product scope, constraints and engineering philosophy |
| `.agents/docs/ZERO_CONFIG_CLI_CONFIG_AND_OPENAI_API_CONTRACT.md` | 21399 | zerogravity — One-command CLI, progressive configuration and OpenAI-compatible APIs (v4.3) |
| `.agents/plans/CURRENT_STATUS_AND_TRACKING.md` | 23641 | Current status and tracking |
| `.agents/plans/EXPERIMENT_PROGRAM.md` | 43912 | zerogravity — Project-neutral experimental program, H01–H43 (v4 reviewed) |
| `.agents/plans/IMMUTABLE_SHARDED_SOURCE_IMPLEMENTATION_SPEC.md` | 12539 | Large immutable source loading: bounded basic-unit proposal |
| `.agents/plans/IMPLEMENTATION_ROADMAP.md` | 41396 | Zerogravity — architecture roadmap P00–P23 |
| `.agents/plans/INTEGRATED_PHASE_0_AND_DELIVERY_GATES.md` | 28363 | zerogravity — Independent Stage-0 feasibility and delivery gates (v4 reviewed) |
| `.agents/plans/MODEL_ACQUISITION_IMPLEMENTATION_SPEC.md` | 15319 | Acquisition next unit — bounded cache first, then HF/GitHub vertical slice |
| `.agents/plans/README.md` | 4451 | Plans |
| `.agents/plans/RELEASE_ACCEPTANCE_GATES.md` | 27291 | zerogravity — Model-neutral release, quality and composability gates (v4 reviewed) |
| `.agents/plans/REQUIREMENTS_AND_TEST_TRACEABILITY.md` | 8791 | Requirements and test traceability |
| `.agents/plans/TODO.md` | 41593 | Plan queue |
| `.agents/plans/ZEROGRAVITY_MASTER_PLAN_2026-10-10.md` | 55266 | zerogravity — master architecture v4.3: composable AI execution, model routing and simple user experience |
| `.agents/plans/cli-device-routing/README.md` | 44480 | CLI and device routing |
| `.agents/plans/cli-device-routing/TODO.md` | 18691 | CLI and device routing |
| `.agents/plans/cli-ux/NEXT_ITERATION_CLI_UX_AND_MODEL_LIFECYCLE.md` | 82634 | Next implementation iteration — CLI, model lifecycle and terminal UX |
| `.agents/plans/cpu-gpu-optimization/BACKEND_PARITY.md` | 12325 | Vulkan / CUDA / HIP parity |
| `.agents/plans/cpu-gpu-optimization/BENCHMARK_CONTAINER_TERMINATION.md` | 6694 | l5 benchmark container termination |
| `.agents/plans/cpu-gpu-optimization/NEXT_ITERATION_BACKEND_PARITY_AND_EXECUTION_PERFORMANCE.md` | 41289 | Next implementation iteration — backend parity and execution-performance architecture |
| `.agents/plans/cpu-gpu-optimization/SHARED_MEMORY_AND_BOUNDARIES.md` | 16407 | Shared storage and CPU–GPU boundaries |
| `.agents/plans/cpu-gpu-optimization/SHORT_HF_RESULTS.md` | 21808 | Short HF performance — current source21 measurement |
| `.agents/plans/cpu-gpu-optimization/TODO.md` | 107428 | Cpu gpu optimization |
| `.agents/plans/distributed-execution/TODO.md` | 3821 | Distributed execution |
| `.agents/plans/fabric-vnext-transition/ARCHITECTURE_AND_CRATE_CONTRACTS.md` | 19736 | Fabric vNext transition — complete target crate/module architecture and contracts |
| `.agents/plans/fabric-vnext-transition/CURRENT_SOURCE_RECONCILIATION.md` | 33484 | Fabric vNext transition — current-source reconciliation |
| `.agents/plans/fabric-vnext-transition/DECISIONS_RISKS_AND_ACCEPTANCE.md` | 14609 | Fabric vNext transition — settled decisions, risks, unresolved choices and acceptance law |
| `.agents/plans/fabric-vnext-transition/FEATURE_PARITY_MATRIX.md` | 33311 | Fabric vNext — comprehensive current feature and behavior preservation matrix |
| `.agents/plans/fabric-vnext-transition/HANDOFF.md` | 14180 | Fabric vNext transition — architecture and continuation |
| `.agents/plans/fabric-vnext-transition/PHASED_IMPLEMENTATION.md` | 26622 | Fabric vNext — precise implementation phases, logical units and signed-commit program |
| `.agents/plans/fabric-vnext-transition/QUALIFICATION_AND_COMMIT_POLICY.md` | 16534 | Fabric vNext — qualification, testing cadence, bugfix policy and signed commits |
| `.agents/plans/fabric-vnext-transition/QUANTIZATION_AND_NUMERICS.md` | 17295 | Fabric vNext transition — quantization, codecs, numerical equivalence and ownership |
| `.agents/plans/fabric-vnext-transition/README.md` | 20062 | zerogravity Fabric vNext — transition architecture and implementation plan |
| `.agents/plans/fabric-vnext-transition/ROADMAP_AFTER_TRANSITION.md` | 17831 | Fabric vNext — feature-expansion roadmap AFTER structural/functional transition |
| `.agents/plans/fabric-vnext-transition/STRUCTURAL_SKELETON.md` | 13839 | Fabric vNext transition — FIRST full Rust skeleton and complete code relocation |
| `.agents/plans/fabric-vnext-transition/TODO.md` | 41132 | Fabric vNext transition — active implementation queue |
| `.agents/plans/fabric-vnext/DECISIONS_AND_OPEN_QUESTIONS.md` | 21019 | Fabric vNext — decision ledger, rejected alternatives, open questions and conversation provenance |
| `.agents/plans/fabric-vnext/HANDOFF.md` | 7316 | Fabric vNext — architecture design context and handoff |
| `.agents/plans/fabric-vnext/IMPLEMENTATION_AND_ACCEPTANCE.md` | 39617 | Fabric vNext — migration blueprint, acceptance gates and source-change crosswalk |
| `.agents/plans/fabric-vnext/PRIOR_ART_AND_REVALIDATION.md` | 45401 | Fabric vNext — full comparative research and zerogravity revalidation |
| `.agents/plans/fabric-vnext/PUBLIC_LIB_AND_SERVING.md` | 25666 | Fabric vNext — public zerogravity Rust library and reusable serving library |
| `.agents/plans/fabric-vnext/README.md` | 45806 | zerogravity Fabric vNext — complete architecture reference |
| `.agents/plans/fabric-vnext/TODO.md` | 8032 | Fabric vNext — held future plan queue |
| `.agents/plans/memory-and-streaming/README.md` | 3106 | Source-preserving SSD residency |
| `.agents/plans/memory-and-streaming/TODO.md` | 35283 | Memory and streaming |
| `.agents/plans/model-acquisition/README.md` | 95265 | Model acquisition |
| `.agents/plans/model-acquisition/TODO.md` | 45445 | TODO — model acquisition |
| `.agents/plans/model-execution/README.md` | 24867 | Zerogravity: source-real model execution (CPU-first) |
| `.agents/plans/model-execution/TODO.md` | 336845 | Model execution |
| `.agents/plans/model-execution/archive/populated-hip-serving/TODO.md` | 15479 | Populated Qwen HIP serving |
| `.agents/plans/multimodal/TODO.md` | 7625 | Multimodal |
| `.agents/plans/numerics-and-context/CORRECTNESS_FIRST_VLLM_PLAN_2026-10-20.md` | 45470 | Correctness first, then measured vLLM performance |
| `.agents/plans/numerics-and-context/CORRECTNESS_POLICY.md` | 7481 | Correctness policy — all models and backends |
| `.agents/plans/numerics-and-context/TODO.md` | 145717 | Numerics and context |
| `.agents/plans/pipeline-composition/FORMATS.md` | 7672 | Source formats and precision homes |
| `.agents/plans/pipeline-composition/PROFILE_POLICY_AND_COMPONENT_REUSE.md` | 11926 | Profile policy, replaceable components and native reuse |
| `.agents/plans/pipeline-composition/README.md` | 40567 | Zerogravity — one fabric, single-model and multi-model pipelines |
| `.agents/plans/pipeline-composition/REGIONS_AND_PROFILES.md` | 49881 | Recursive pipelines and composable optimization profiles |
| `.agents/plans/pipeline-composition/SHARED_EXECUTION_ARCHITECTURE.md` | 56237 | Pre-0.1 execution fabric: source-grounded decisions |
| `.agents/plans/pipeline-composition/STRUCTURE_AND_NAMING.md` | 14133 | Structure and naming |
| `.agents/plans/pipeline-composition/TODO.md` | 53516 | TODO — Single-process model and pipeline composition |
| `.agents/plans/platform-and-release/TODO.md` | 7527 | Platform and release |
| `.agents/plans/serving-and-speculation/BATCHING_AND_PAGED_STATE.md` | 73106 | Batching, paged attention and hybrid prefix state |
| `.agents/plans/serving-and-speculation/BENCHMARK_CAMPAIGN_2026-10-27.md` | 9621 | Qwen matched serving campaign — 27–30 Oct 2026 |
| `.agents/plans/serving-and-speculation/FUSED_GPU_BATCHING.md` | 60236 | P12 — native fused GPU batch acceptance |
| `.agents/plans/serving-and-speculation/NEXT_ITERATION_OPENAI_COMPATIBLE_SERVING.md` | 31645 | Next implementation iteration — full OpenAI-compatible serving |
| `.agents/plans/serving-and-speculation/README.md` | 13805 | Native serving and request scheduling |
| `.agents/plans/serving-and-speculation/TODO.md` | 128559 | Serving and speculation |

## Appendix B — generated current open-work inventory

Generated from every .agents/plans/**/TODO.md. The owning TODO remains the
authority because surrounding context and ordering may matter.

<details>
<summary>Open / active / blocked TODO lines</summary>

### `.agents/plans/TODO.md`

**Active**
- L479: - [ ] P1 serving-and-speculation — @pvl — l5 CUDA `.8B`/`2B`, x2 ROCm `.8B`/`2B`/`27B`: source-real correctness/paged/prefix/fused 2/4/8-row cohorts accepted; multi-token prefill + autoregressive refill/regroup now fast under matched opt1. Public CUDA serving passes4K/8K/16K on both small models,0.8B32K has one full pass, and paged2B32K passes32766 tokens in224.80s. Same32K-capacity1024-token A/B is283.22s contiguous-streamed vs28.56s paged-resident (~9.9x). Generic cold-start smart config now jointly resolves Qwen `state_policy` + `cohort_width` from fresh runtime admission: Exclusive=ignore user knobs, Enabled(default)=user values override Auto per knob, Disabled=no Auto; score=resident weights > wider 8/4-row cohort > Contiguous tie. Default2B32K/1024 Auto-selects paged and passes in26.51s exact. Existing `active=8` SDK with cohort width unspecified Auto-selects8 for both Contiguous/user-Paged halves and publishes `{8}` at135.979/133.004 tok/s exact. Nix-realized vLLM0.33.0/cu130 correctness now passes exact local0.8B/2B on L5 without Podman; the frozen wheel graph is store-realized and an optimized CUDA-graph smoke passes, so the retained concurrency=1/4/8/16/32/64 paired matrix is next on L5 before the exact-commit X2 ROCm campaign. Independent populated all-state-domain closure remains open; X2 ROCm remains environment-gated — [details](serving-and-speculation/TODO.md)
- L480: - [~] P1 numerics-and-context — @pvl — exact prefill 47 + decode 48–50, lifecycle/capacity and final gate pass; active-context/2B reference inputs missing — [details](numerics-and-context/TODO.md)
- L481: - [~] P1 memory-and-streaming — @pvl — 32K storage/execution policy is materially closed:0.8B32K has one full public-SDK pass;2B32K passes the full32766-token retained cell with Paged64/64, and canonical default startup Auto-selects paged when contiguous would stream. Final CUDA admission independently clamps provider capacity to original-actor live free VRAM while preserving configured headroom. CUDA-forward source-page transfer now retains the verified `TensorWindow` itself through the existing checkpoint and uses the original actor's bounded owned-window H2D directly, removing the superseded pinned/Bytes copy chain. Matched 2B/32K prompt48 streamed feedback is38.46s vs39.36s at `b839920`; cache-promotion41.00s, moved-pinned44.88s and direct-SSD/cache-bypass41.77s variants were rejected. A same-tree one-page CUDA retirement-ring Qwen A/B is rejected at39.281s vs38.323s serial (~2.5% slower); production remains serial at that boundary and default smart config avoids it when paged state enables resident weights. The model-neutral primitive remains dormant/reusable across backends: CUDA `OwnedCudaEvent` and HIP `OwnedHipTransferRetirement` implement `RetirementMark`, with zero-cost CUDA/HIP ring aliases over the shared `TransferRetirementRing`. Shared completed-root reuse, allocation-free common-frontier reconciliation, exact partial-tail paged COW ownership and canonical physical allocation+range identity are active; CUDA-forward prepared sets, streamed-head transfer binding and production HIP resident-text workspace/root binding use canonical ledger identity. Remaining work is additional real identity consumers plus cache-aware direct/async IO and broader authenticated streaming tiers — [details](memory-and-streaming/TODO.md)
- L482: - [~] P1 cpu-gpu-optimization — @pvl — forward kernel/transport checks pass; exact whole-model parity before measured tuning and the matched benchmark campaign — [details](cpu-gpu-optimization/TODO.md)
- L483: - [~] P1 pipeline-composition — @pvl — recursive primitives/pipelines/profiles; shared execution adapters active, reuse current adapters in Fabric; Qwen expansion deferred — [details](pipeline-composition/TODO.md)
- L484: - [~] P1 cli-device-routing — @pvl — unified run/config/source/device selection; preserve behavior through Fabric SDK/CLI migration — [details](cli-device-routing/TODO.md)
- L485: - [~] P2 model-execution — @pvl — current Qwen CUDA forward owns source-real fused 2/4/8-row contiguous+paged transport, complete-domain prefix reuse, event-backed/multi-flight primitives, retained stateless + multi-token attention/GDN prefill, reduced 4/8-row GDN layout selection and bounded paired-page KV gather with fused inactive-tail fill. Latest matched opt1 short-wave is135.236 tok/s contiguous/132.949 paged (~1.7% wall-time gap); current68-token refill is169.837/160.919 tok/s. Rejected A/B mechanics are retained dormant rather than deleted: lane regroup, copy+tail-zero,17 BF16 GEMM strategies (including restored terms64 shared-row, paired-boundary and K64 weight-cache variants), recurrent tiled transpose and GDN occupancy variants, with zero production consumers. The restoration sweep also repairs default/qualification/CUDA-native test cfg drift; the largest native-qualification lane is external-SIGKILL incomplete. Public graph serving deliberately remains Native. Large same-contract CUDA numerical acceptance still depends on approved references; populated small HIP and resident27B work remain separate. X2 is shared: preserve other owners and verify only ZeroGravity's incremental resource delta retires — [details](model-execution/TODO.md)
- L486: - [~] P2 model-acquisition — @pvl — [details](model-acquisition/TODO.md)
**Icebox**
- L490: - [ ] multimodal — [details](multimodal/TODO.md)
- L491: - [ ] distributed-execution — [details](distributed-execution/TODO.md)
- L492: - [ ] platform-and-release — [details](platform-and-release/TODO.md)

### `.agents/plans/cli-device-routing/TODO.md`

**Uncommitted**
- L88: - [ ] Shared pre-Client transactional admission remains open and explicitly
- L94: - [ ] Recheck live source, ownership and accepted model/backend evidence before
- L99: - [ ] Accept local paths, catalog/model names, `hf:`, `gh:`, `http:` and `https:`
- L103: - [ ] Resolve already acquired immutable exact sources, reject ambiguous variants
- L107: - [ ] Every new CLI semantic option has a typed v2 field. Unit1 implementation
- L117: - [ ] Normalize `run`, backend aliases and common task flags into one library
- L119: - [ ] Extend the canonical pipeline document/compiler to proposed v2 with one
- L122: - [ ] Implement unambiguous positional model/config and explicit selectors;
- L125: - [ ] Implement CLI > selected config > built-in resolution with per-field
- L128: - [ ] Migrate model metadata to `--model-config`; cover every existing task,
- L130: - [ ] Retain bounded strict YAML/graph validation; check the complete merged
- L133: - [ ] Resolve exact local source and bounded task/model requirements through
- L135: - [ ] Add read-only inventory and stable physical identity resolution; derive
- L137: - [ ] Implement strict ordered allowlists, CPU-terminal semantics, local-only
- L139: - [ ] Implement current-memory observation with explicit provenance/age/unknown
- L142: - [ ] Resolve a safe default operator-ceiling/bootstrap policy once for a new
- L145: - [ ] Integrate complete cold/prefill/decode/native/compiler/state/output costs
- L148: - [ ] Implement configurable headroom, all-GPU full-fit scan and qualified CPU
- L150: - [ ] Revalidate and admit through original owners; bound safe startup replanning,
- L152: - [ ] Provide `devices list`, `--dry-run`, `--explain` and structured refusals
- L154: - [ ] Validate the assembled unit on approved source/hardware/backend tuples;
- L164: - [ ] Define source-specific indivisible groups and embedding/head/shared-weight/
- L166: - [ ] Extend original arbiter with all-or-none multi-domain admission if needed;
- L168: - [ ] Integrate original state shards, typed activations and actual event/reader
- L171: - [ ] Prove same-backend multi-GPU and GPU↔CPU compositions separately before
- L173: - [ ] Verify full-logit/complete-state following-token parity, cancellation,
- L175: - [ ] Compare full CPU and each feasible single/split route with complete cold/
- L182: - [ ] Introduce semantic stage-role constraints without model-family CLI routing
- L184: - [ ] Qualify actual trained draft/MTP source and recursive state semantics;
- L186: - [ ] Add an NPU provider only through existing region/state/lifetime boundaries;
- L189: - [ ] Resolve explicitly enrolled peer aliases to authenticated identities and
- L191: - [ ] Coordinate bounded all-or-none holds through original per-host ledgers;
- L193: - [ ] Qualify first two-host contiguous layer/local-state composition on approved
- L195: - [ ] Keep remote inference APIs that hide region/state contracts outside native
**Tests and verification**
- L204: - [ ] **Syntax equivalence:** canonical/alias/repeated flags normalize identically;
- L207: - [ ] **Default config:** present/missing/unreadable/invalid default; absolute,
- L210: - [ ] **Config discrimination:** YAML suffix, extensionless explicit --config,
- L214: - [ ] **Config/CLI parity:** direct YAML and flags yield equal typed requests;
- L218: - [ ] **Precedence/provenance:** CLI scalar wins, alias equals canonical backend,
- L222: - [ ] **Path bases:** config-origin source/sidecar/audit paths use selected config
- L225: - [ ] **Config safety/entrypoint:** unknown/duplicate fields, expansion/depth/file
- L229: - [ ] **Identity:** CUDA/Vulkan aliases of one card count once; AMD discrete and
- L232: - [ ] **Forced scope:** CUDA+AMD rejects before allocation/compiler activity;
- L235: - [ ] **Full fit:** both vendors fit→NVIDIA; NVIDIA insufficient/occupied and AMD
- L238: - [ ] **CPU fallback:** no GPUs, missing GPU build, unqualified GPU model and no
- L241: - [ ] **Current pressure:** large total VRAM with low free bytes does not fit;
- L244: - [ ] **Accounting:** original leases plus actual backing plus unbacked promises;
- L247: - [ ] **UMA:** CPU/GPU/NPU alias one memory domain; cross-device views/transfers
- L249: - [ ] **Complete peaks:** weights fit but prefill/KV/output/compiler/native/capture/
- L252: - [ ] **Headroom:** percentage/bytes/default/floor/overflow validation; resolved
- L256: - [ ] **TOCTOU:** external allocation and concurrent internal admission between
- L259: - [ ] **Unsafe preparation/flight:** unknown module load/unload, transfer/capture/
- L262: - [ ] **Split feasibility:** largest legal prefix leaves a feasible remainder;
- L265: - [ ] **Runtime equivalence:** real full affected logits/tensors, every required
- L268: - [ ] **Fairness/lifetime:** overlapping requests/models preserve independent
- L271: - [ ] **Diagnostics:** list/dry-run leave model, GPU context/compiler, resource
- L275: - [ ] **NPU role, future:** qualified NPU still loses when transfer/verification/
- L280: - [ ] **Remote, future:** unknown enrollment, identity/epoch mismatch, denied
- L283: - [ ] **Performance:** authorized paired whole-request comparisons, cold/hot
- L286: - [ ] **Final assembled gate:** after applicable unit qualification, run

### `.agents/plans/cpu-gpu-optimization/TODO.md`

**Uncommitted**
- L30: - [ ] After correctness: optimize CUDA `.8B`/`2B` on l5 and ROCm `.8B`/`2B`/`27B`
- L33: - [ ] Beat the matched, pinned vLLM baseline in every required feasible run;
- L37: - [ ] x2 compute/memory/storage use is authorized without routine requests;
- L55: - [ ] Run current-source `.8B`/`2B` model, cached, fault, shutdown and fitted-context
- L60: - [ ] Run the one serial campaign specified in
- L67: - [~] [Shared vLLM/llama.cpp correctness policy](../numerics-and-context/CORRECTNESS_POLICY.md)
- L71: - [ ] Preserve case5 FAIL101 and quarantine. Original default vLLM N1/B4 also
- L78: - [ ] Current B4 → fresh optimized matched C16 (.8B +2B) → x2 original27B,
- L265: - [~] User extends both-model sequence to C16, thenC32 andC64; NO C2. Implement
- L279: - [~] C16 both executed on frozen21. Clean2B: ZG78.7859tok/s,1.62466s;
- L505: - [ ] Shared compiler lifetime/readiness: fix checked all-exit raw NVRTC cleanup,
- L512: - [ ] After current C8/refill and source113 qualification: shared prepared matrix/
- L649: - [ ] **Parallel implementation, third merge — genuine BF16/HF request batching
- L662: - [ ] Lift reusable source identity, bounded immutable reads, prepared bindings,
- L666: - [ ] Shared resident-backing pool + immutable layout/access metadata + bounded
- L673: - [ ] Demonstrate superior matched .8B C1 AND C4, then run C8. Only after all
- L679: - [ ] Merge Qwen27B/Gemma contained units as ready in parallel. Their later GPU
- L681: - [ ] Bring Vulkan implementations to CUDA parity independently; AMD uses HIP,
- L688: - [ ] **OPT-01 matched short baseline:** after parity, established same-source
- L698: - [ ] **OPT-02 reusable bindings:** profile then qualify stable leased row/
- L737: - [ ] **OPT-03 prepared optimized regions:** close existing production graph
- L742: - [ ] **OPT-04 temporal/fair work:** extend existing native sequence batches
- L747: - [ ] **OPT-05 state/residency composition:** full-domain compatible prefix
- L751: - [ ] **OPT-06 larger dense source:** after long-context prerequisites, original
- L755: - [ ] **OPT-07 optional proposals:** target verifier/all-prefix rollback first,
- L760: - [ ] **OPT-08 nested profiles:** reconcile primitives→recursive pipelines→
- L765: - [ ] **OPT-09 deferred capabilities:** typed sparse selection, grouped experts/
- L769: - [ ] **OPT-10 promotion:** source/full-logit/all-state/natural-output, actual
- L779: - [ ] Qualify [cross-stage residency/fusion](../pipeline-composition/REGIONS_AND_PROFILES.md#resident-values-and-cross-stage-lowering--owner21-oct)
- L785: - [ ] Complete the dedicated [ExLlamaV3 investigation](../fabric-vnext/PRIOR_ART_AND_REVALIDATION.md):
- L792: - [ ] Implement native kernel-contract comparison/fixture support, reusing or
- L798: - [ ] Implement the useful kernels/components identified by these investigations
- L802: - [ ] Specialize first for the NVIDIA cards on both existing machines, then
- L808: - [ ] Bench and hyper-optimize in strict user order:
- L817: - [ ] Keep the local Renoir/gfx90c hold: 2B initialization aborted and automatic resets wedged it; no further launch/reset/recovery or Drop-retirement proof is implied.
- L818: - [ ] Establish controlled cold/warm request baselines across all five GPU/three CPU optimization targets; accepted five backend profiles do not mean five cards or fleet-wide maximal capability use.
- L819: - [ ] Qualify capability-selected AVX2/AVX512/VNNI quantized kernels, cache/thread/topology policies and source-derived repacking while preserving reduction/rounding/zero/subnormal rules.
- L820: - [ ] Measure native/Vulkan tiling, subgroup/vector loads, packing, source-exact fusion/prepared dispatch; gate tensor/MMA/WMMA/integer-dot use by exact arithmetic, not advertised support.
- L821: - [ ] Evaluate explicit NativeF32 projection policy only after physical FMA edge tests, all-codec trained vectors and whole-model/cancellation/replay gates; WideF64 stays default meanwhile.
- L822: - [ ] Profile then qualify captured graphs/replay, bounded device loops/persistent scheduling, CPU/GPU overlap and exact optional GPU tokenization/detokenization; prepared plans alone are not graph capture.
- L823: - [ ] Keep separate LLVM code-generation paths and broader accelerator backends unqualified until independently built and physically tested; accepted native paths use CPP NVRTC/HIPRTC.
- L824: - [ ] Qualify optional EXL3/QTIP/trellis/codebook/low-bit derived packs and calibration identity/quality separately from original source semantics. No hidden precision substitution or default steering.
- L825: - [ ] Promote only repeated complete-request gains with cold/warm cost, tails, memory, power, source correctness, negative results and qualified capability/version guards.
- L827: - [ ] Gate future compiler/driver/libm and fast-arithmetic profiles with scalar fingerprints, physical edge/stress and trained full-logit/state tests. WideF64 is qualified on measured inputs, not a universal IEEE FMA emulation.
- L828: - [ ] Qualify real deadline wakeups, concurrent/multi-stream scheduling and actual driver failure/recovery before broader worker claims; queue deadlines currently need scheduling/poll opportunities.
- L829: - [ ] Keep the local gfx90c hold. If a healthy/recovered profile is separately authorized, localize retained allocation/H2D/final-sync initialization before changing kernels; natural SIGABRT establishes neither Rust Drop quarantine nor reclaimed storage.
- L830: - [ ] Consider read-mostly publication only after representative contention and ownership evidence. If left-right fits, test deterministic dual replay without IO/RNG/admission/state/device effects, one writer, persistent non-Sync readers, short guards without await/load/device work and explicit publish/drain/shutdown. Account for both indexes/logs while sharing one payload; do not reopen the rejected cold-catalog migration.
- L832: - [ ] Resolve applicable SYK-011..016 source/derived-quality/LLVM/SLO/pressure/steering choices from the [specialization reference](../../research/EXTREME_MODEL_SPECIALIZATION_INTEGRATION_AND_ACCEPTANCE_2026-10-15.md) before that slice; proposed SYK-002..010 are not frozen user approvals. Preserve negative candidates and the default disabled steering policy.
**Committed**
- L849: - [~] Shared queue checkpoint preserves application-owned HOST admission,
- L894: - [~] Direct short-HF route lifetime correction: preliminary tokenizer drops
- L904: - [~] CUDA B4 HOST requirement needs correction before fresh acceptance.
- L928: - [~] Five-lane source is assembled in MAIN; the final affected batch completes
**Tests and verification**
- L1242: - [ ] Assembled .8B host batch: authenticate immutable114-weight cohort once
- L1253: - [ ] Native harness passes positive fixtures and rejects all named planted
- L1255: - [ ] Reused/ported implementation passes exact/approximate declared-contract,
- L1258: - [ ] QTIP-01..14: derived source codec/pack identity, transforms, held-out quality and physical kernel correctness pass independently of original model semantics.
- L1259: - [ ] HYB-01..18: exact source routing/draft/tap/mask/reduction and target-authoritative verification pass, including cases where a wrong draft is rejected or deeper drafting slows work.
- L1260: - [ ] GFX-01..18: WMMA/LDS caller/shape coverage, driver/compiler numeric gates and coherent-memory physical headroom pass; require matched complete-request gains. See the [50-test register](../../research/EXTREME_MODEL_SPECIALIZATION_INTEGRATION_AND_ACCEPTANCE_2026-10-15.md#4-acceptance-register-50-planned-tests-not-implemented).
- L1266: - [ ] Complete current15 fresh matched API/HTTP for .8B and2B; ROOT1254 running. Then quiet C1/C4 and C8 only after both rate+wall wins, no hot compiler activity; previous .8B measurements remain historical.
- L1267: - [ ] Full assembled check.sh phase gate after CLI/source reconciliation; source12 gate18 remains the last accepted whole gate.
- L1273: - [ ] Real current-source qualification remains mandatory before production routing: .8B natural/current-root parity including cached 47→50, physical fault/cancellation/retirement and real-model shutdown, then .8B 512/1K/2K/4K/8K and 16K/32K. Admit 2B contexts only where the exact two-root plan fits.
- L1274: - [ ] Current physical blocker: the approved .8B source is now visible at `/home/pvl/.cache/abird/zerogravity/external-baselines/models/Qwen--Qwen3.5-0.8B/2fc06364715b967f1860aea9cf38778875588b17`; its 1,746,942,600-byte safetensor SHA256 is exactly `04b1c301231dd422b8860db31311ab2721511346a32cb1e079c4c4e5f1fe4696`. The transient `.agents/runs` native bundle was intentionally purged and no `cuda-kernels.json`/legacy `short-hf-native.json` bundle remains under the repo or user cache. Reacquire/regenerate the qualified native/reference bundle through the maintained capture path; do not restore deleted run directories or substitute another model source.
- L1289: - [ ] These are fit/admission results only. Physical correctness, continuation,
- L1308: - [ ] Real physical after-launch cancellation/fault and real-model shutdown
- L1332: - [ ] Execute both phases serially as soon as the approved .8B safetensor shard
- L1359: - [ ] Physical execution is currently blocked by provider identity, not code:
- L1486: - [ ] When the approved source tree, pinned native bundle payloads and pinned

### `.agents/plans/distributed-execution/TODO.md`

**Uncommitted**
- L21: - [ ] Evaluate layer, tensor/head, context and expert partitions as independent
- L25: - [ ] Admit shared collective/IPC/RDMA backing, staging, graph and encoded-reader
- L28: - [ ] Later MoE EP/replicas/EPLB/communication overlap: preserve logical expert
- L32: - [ ] Later produced-state P/D/encoder separation and remote prefix hydrate:
- L36: - [ ] Compare placement against local batching and capacity needs using actual
- L41: - [ ] Preserve zero-network standalone execution; add capability/transport/placement seams through the same model/task/coordinator/resource authorities.
- L42: - [ ] Qualify optional local CPU/GPU/NPU split projection, expert ownership and prefill/decode placement with actual event/memory/power-fabric safety; Stage N is independent of CPU/GPU release.
- L43: - [ ] Begin authorized two-node layer-group plus local-state execution and immutable full-state hydration; then evaluate P/D and attention+state versus FFN/expert disaggregation.
- L44: - [ ] Extend tensor/head/pipeline/data/context/expert/sequence parallelism and collectives only where exact state partition algebra, source numerics and transport cost support them.
- L45: - [ ] Authenticate/version source/state/tenant/session epochs and complete transfers; loss/timeout/stale peers cannot publish partial roots or release unknown in-flight storage.
- L46: - [ ] Qualify topology changes only at committed frontiers and compare collocation against fast/slow/tunneled links, including real transfer/copy/crypto cost and fairness.
- L47: - [ ] Add a peer/transport/placement without rewriting model, task, coordinator or API; fleet SSH qualification alone is not distributed inference.

### `.agents/plans/fabric-vnext/TODO.md`

**Uncommitted — execution HOLD**
- L28: - [ ] Confirm the user has expressly initiated the **read-only resync/revalidation phase**.
- L29: - [ ] Read the full frozen Fabric vNext pack and literal user amendments, then AGENTS.md and current owner plans.
- L30: - [ ] Inspect current live HEAD, branch, checkout, recent commits, in-flight agents and true Cargo feature/dependency graph. Protect third-party work.
- L31: - [ ] Reconcile current source/architecture docs/active TODOs with historical 26-Oct analysis; classify old path anchors obsolete/new/already done.
- L32: - [ ] Reconfirm current in-source model families, generic semantics, numerical policies, physical provider, queues, state publication, resource services and source ownership.
- L33: - [ ] Reconfirm current model-neutral engine/SDK/CLI/serving responsibilities and look specifically for newly added server/engine library work.
- L34: - [ ] Re-evaluate public zerogravity facade + standalone zerogravity-serving crate + thin CLI against exact new code; revise module names/ownership for simplicity without weakening the newest user decision.
- L35: - [ ] Validate distinction among mathematical region, native ModelAssembly and multi-model TaskPipeline; complete-domain state and typed media clocks.
- L36: - [ ] Crosswalk every source/design gap with the current active P00–P23 workstreams, including source/quality/physical and model/backend holds.
- L37: - [ ] Produce source-grounded diagram, gap table, revised implementation plan, test evidence ledger and explicit remaining decisions. **Stop without coding.**
- L41: - [ ] Receive explicit user permission to start implementation after the source-grounded resync.
- L42: - [ ] Freeze a reproducible implementation/test baseline without corrupting active work.
- L43: - [ ] Introduce minimal generic SemanticProgram/closed-region and prepare/bundle paths from existing ForwardPlan; avoid second IR.
- L44: - [ ] Remove family math and source-specific source/ABI recipes from generic core, artifact, GPU and serving implementations.
- L45: - [ ] Refactor model family crates and reverse dependencies, no GPU → Qwen/Gemma import.
- L46: - [ ] Extract one model-neutral zerogravity-engine or equally enforceable cohesive engine boundary; one TaskCoordinator/StateRoot and physical ResourceArbiter.
- L47: - [ ] Provide source-qualified model instance/session/TaskDriver interfaces and cold built-in registration, not model-name switches in runtime.
- L48: - [ ] Implement public zerogravity Rust library for ALL functionality (run, inspect, load/pull/cache, admin, device planning, pipelines, metrics and serving).
- L49: - [ ] Create independent, embeddable **zerogravity-serving** Rust library for HTTP/OpenAI/Responses/stream/media service protocols and lifecycle; no CLI/model-kernel imports.
- L50: - [ ] Make zerogravity-cli ONLY call exported zerogravity library and own args/env/terminal/signal/output.
- L51: - [ ] Prove separate library embed and server start/stop, direct run and HTTP share same model/resource/state; strict auth/stream/errors.
- L52: - [ ] Prove Qwen3.5 0.8B/2B source-real CPU and selected GPU state/numeric/regression gates survive changes.
- L53: - [ ] Add Gemma source-real model through same engine and server without source-specific allocator/HTTP/scheduler branching.
- L54: - [ ] Record accepted and failed/unsupported paths per actual profile/driver/model/corpus and explicitly leave unrun hardware cases open.
- L58: - [ ] Persistent GPU slots, incremental row metadata, actual async bounded step submission and retirement.
- L59: - [ ] Native source-qualified continuous batching, real temporal prefill/decode overlap and fairness/ageing.
- L60: - [ ] Paged and complete-domain hybrid state/COW/prefix checkpoint reconciliation including sparse and recurrent state.
- L61: - [ ] Region fusion/capture/device-native feedback; optional speculation and draft/target full state semantics.
- L62: - [ ] Unified physical residency and SSD streaming on one source identity with byte-budgeted cache and real completion.
- L63: - [ ] Qwen3.8 27B exact source/CPU/selected GPU, Gemma E2B→26B-A4B→31B independent functionality; then richer sparse/quant targets.
- L64: - [ ] First real non-token task and multi-model typed pipeline via same public library/engine/serving.
- L65: - [ ] Qualified CPU+GPU+NPU, local split and authorized remote region workers only with actual source/state/physical/latency proof.
- L66: - [ ] Matched vLLM/llama.cpp complete-request/quality/TTFT/ITL/P95/goodput/memory/energy benchmarks and negative findings.
- L67: - [ ] Minimality/extension drills for new model, novel stateful op, backend, task, transport with measured shared source change radius.
- L68: - [ ] Consolidate normative docs/public whitepaper/current status after actual implementation acceptance.
**Tests and verification — maintained design set**
- L72: - [ ] Keep this plan set internally linked and consistent with the canonical current handoff and four-file planning spine.
- L73: - [ ] Revalidate source-dependent statements against current main before implementation; historical SHAs remain provenance only.
- L74: - [ ] Keep the next-agent prompt source-first: read the canonical handoff, current status/traceability, then this plan; do not reconstruct deleted archive snapshots.
- L75: - [ ] Later: enforce Cargo acyclic layering including serving→engine and facade→serving and CLI→facade.
- L76: - [ ] Later: public zerogravity-only embedded Rust direct run/server examples compile without CLI, with one shared model/ResourceArbiter and exact source states.
- L77: - [ ] Later: source/semantic/quality/physical/serving/multimodal/distributed acceptance gates defined in IMPLEMENTATION_AND_ACCEPTANCE.md and original project owners.
- L78: - [ ] Later: no vLLM/llama.cpp performance parity claim absent true matched workload/driver/profile/quality evidence.

### `.agents/plans/fabric-vnext-transition/TODO.md`

**Future expansion AFTER transition, not initial acceptance**
- L538: - [ ] E01 original-HF general Qwen3.5/2B and real 27B; independent fresh47–50/long context CPU/CUDA/HIP/Vulkan.
- L539: - [ ] E02 vLLM continuous batching, persistent row slots, temporal prefill, all-domain paged/COW state, goodput; benchmark.
- L540: - [ ] E03 shared multi-model residency, replicas and hot reload.
- L541: - [ ] E04 true large dense SSD authenticated windows and physical streaming.
- L542: - [ ] E05 Gemma4 E2B/26B-A4B/31B complete native qualification, then new families.
- L543: - [ ] E06 Q4_K source kernel, Q1/Q2/Q3/NVFP4/EXL3/source derived quant and quality evidence.
- L544: - [ ] E07 sparse/QSA/GDN/MLA/Engram/MoE/MTP draft state and expert IO.
- L545: - [ ] E08 native ASR/TTS/codec/image/video/duplex and typed task pipelines.
- L546: - [ ] E09 NPU/graph/wafer/remote distributed placement and state bridge.
- L547: - [ ] E10 calibrated cost/autotune, matched vLLM/llama.cpp task benchmarks and release.

### `.agents/plans/memory-and-streaming/TODO.md`

**Uncommitted**
- L22: - [~] Physically qualify each admitted context independently. Preserve current
- L50: - [~] Preserve complete recurrent/conv/KV state history, chunk continuation,
- L63: - [~] Bounded completed-root backing reuse is adopted in the shared resident layer.
- L78: - [~] The first CUDA-forward source-page transfer cleanup now uses the existing
- L110: - [~] Canonical physical buffer identity now exists in the model-neutral device
- L149: - [ ] Add measured lifetime-aware scratch reuse/phase lending only when no live
- L152: - [ ] Protect demand/decode hotsets from transient prefill and unused predicted
- L156: - [ ] Separate known source lookup IDs, dense next-layer windows, produced-state
- L160: - [~] Common-frontier prefix reconciliation is now implemented model-neutrally in
- L176: - [ ] For dense27B, compare layer-major/row-cohort source-window reuse against
- L181: - [ ] **Next SSD source execution:** integrate accepted original-index
- L187: - [ ] Shared one-process pipeline residency: source-exact immutable
- L193: - [ ] Prioritize Qwen3.5 2B resident then Qwen3.8 27B SSD
- L201: - [ ] Route legacy/new model allocations, CPU state/scratch/staging and APU/GPU shared backing into the same configured physical pools with OS/cgroup headroom; never sum overlapping UMA/VRAM/GTT counters.
- L202: - [ ] **Paged hybrid state allocator:** extend original leases/StateRoot with
- L321: - [ ] **Complete-domain prefix reuse/COW:** the first physical fork substrate is
- L368: - [ ] Qualify committed snapshot persistence/restore, corruption/security/compatibility UX and restore-versus-recompute; diagnostic portable checkpoints are not product persistence.
- L369: - [ ] Add immutable source extents and bounded coalesced IO/materialization with stable logical tensor identity, genuine copy readiness and original IO/device reader leases.
- L370: - [ ] Protect encoded-but-unsubmitted/captured expert reads and generation ABA; support mixed-codec expert size classes under one byte cap, source-exact routing and hot-cache preservation.
- L371: - [ ] Add bounded PLE/Engram lookup rows and separately governed disk state/media tiers; fully resident execution must incur no mandatory IO or host cache dispatch.
- L372: - [ ] Test source-exact resident/streamed parity, short/corrupt IO, cancellation, safe victim refusal, SSD QoS and actual physical lifetime before eviction/free-space claims.
- L373: - [ ] Optional direct/async/platform IO needs equivalent semantics and explicit support; large memory-pressure/SSD experiments require approved hardware/source scope.
- L375: - [ ] Distinguish logical envelope retirement from driver allocator pages/pools, fragmentation, cross-process use and actual OS release. Driver VRAM/GTT usage/max counters are not reservations or immediately free bytes.
- L376: - [ ] Prove trusted teardown/recovery before reclaiming unknown-work handles or clearing a fence. Diagnostic service sync/readback permits inspection/exit only and cannot synthesize Ready, retire unknown events or authorize reuse.
**Tests and verification**
- L453: - [ ] SSD-01..32: run the [full source/residency suite](../../research/SSD_STREAMING_IMPLEMENTATION_ROADMAP_AND_ACCEPTANCE_2026-10-15.md) for each claimed backing/backend; mock tiers cannot qualify physical IO or trained streaming.
- L454: - [ ] SSD-14/15: encoded-but-unsubmitted, queued and captured readers prevent generation-ABA slot reuse through actual GPU completion/cancellation.
- L455: - [ ] SSD-21/22/24: mixed Q2/Q4 expert sizes share one byte cap with safe victims and fair progress; hot-cache retention cannot bypass admission.
- L456: - [ ] SSD-32: an authorized original trained streamed model matches complete resident logits/StateRoot; measure cold/warm IO, staging, OS headroom and genuine resource lifetime.

### `.agents/plans/model-acquisition/TODO.md`

**Uncommitted**
- L302: - [ ] Canonical source-pull execution order (18 Oct decision): Qwen3.5 .8B/
- L314: - [ ] Extend the new bounded HF/GitHub Auxiliary-only --bundle to
- L322: - [ ] Extend cloud profiles beyond the narrow qualified AWS FULL_OBJECT SHA256 object route: broader S3 partitions/compatible vendors/credentials/checksum algorithms, Azure latest/SAS/managed chains and per-object digest-map prefixes. Keep explicit version, caller-SHA and conditional FirstSeen contracts distinct; never infer immutable versions or trusted raw hashes from unqualified validators or multipart COMPOSITE checksums.
- L323: - [ ] Extend ModelScope only through separately qualified alias authority and redirect/CDN/mirror/credential profiles. Selected missing sizes now use exact pinned HEAD; no alias-to-commit authority was established from revision echoes, timestamps or file last-modified IDs. The current profile does not close broader parity or independently prove server revision/membership behavior.
- L324: - [ ] Measure and qualify native Xet fetch coalescing/chunk caching and any persistent native-prefix protocol. The current bounded per-intersection complete-signed-set reader has counted overfetch/refetch and no performance claim; native process interruption restarts a full transfer. Keep per-term/partial-xorb proof limits, fixed CAS scope and existing elapsed-time/HTTP-trailer boundaries explicit.
- L325: - [ ] Close remaining finite convenience/compatibility gaps: required sidecar/adapter/sparse-quant policy, bounded auxiliary/presets, mirror/endpoint/env precedence and gate/license diagnostics. Preserve explicit source authority/no silent numeric fallback and separate loader/runtime plans.
- L327: - [ ] Qualify any future v1 migration only after separate authorization and proven reader/process quiescence; current removal supports originally protected roots only.
- L328: - [ ] Extend cleanup only through separately qualified exact ownership/recovery protocols for dead retained staging/general orphans and any later eviction policy. Current exact-snapshot operation deliberately retains them and every stable lock inode.
- L329: - [ ] Any additional lazy/raw-path/mmap consumer must adopt managed descriptor/open ownership or remain externally quiescent during removal. Observed named cache charge is not a process-global final-close/physical-free-space attestation.
- L330: - [ ] Qualify additional filesystem pairs before extending the explicit allowlist. Different st_dev values alone do not prove independent backing objects for Btrfs subvolumes, overlay/FUSE or virtualized views; retain branch-specific root device/mount checks and complete inode/ancestry exclusion. No unsupported pair gains acceptance from generic mocks.
- L331: - [ ] Separately authorize any needed namespace-only hidden-bind/subvolume/overlay filesystem fixtures before mount operations. Real target-root/unselected inode aliases and hidden bind aliases in both directions remain NOT RUN; no host mounts, block devices, new privileges or live cache actions are implied. Current tests use already existing distinct filesystems only.
- L332: - [ ] Independently qualify any broader local layouts, renamed/unnamespaced roots, missing mount identities or new destination-root creation before exposing support. Keep explicit selection, retained descriptors, bounded topology work and existing cancellation/rollback limits; real import still needs user-selected source/repository/commit/files and copying authorization.
- L333: - [ ] Finish the finite provider/loader parity ledger beyond qualified HF/GitHub/HTTPS/OCI/GCS/versioned-S3/Azure/ModelScope and explicit local-HF import slices: remaining external-cache layouts/filesystems, additional model hubs, unversioned/broader S3/Azure/compatible profiles, native Xet and broader OCI profiles. Opaque serialization/plugin loaders remain separate runtime work.
- L334: - [ ] Continue explicit source/license authorization and trusted-digest versus first-seen provenance through additional providers/loaders; acquisition grants no license or loading support.
**Tests and verification**
- L338: - [ ] ACQ-A12: complete the finite provider/loader parity inventory and independently qualify remaining supported paths.
- L351: - [ ] Each remaining provider phase requires focused API/CLI/protocol/auth/version/range/offline/fault gates, full frozen isolated check.sh, relevant source validation, actual source-integrity and documentation/dependency/root README checks before signing.
**Deferred**
- L355: - [ ] Unsupported plans fail closed until qualified: unknown-length required/model-closure files, capped resumable transfers, unqualified external-cache layouts/filesystems and remaining cloud/OCI profiles, legacy-root removal and unmanaged-reader concurrent reclamation remain explicit above. Independent full HTTP trailer validation is unqualified under the documented decoder boundary, rather than a reason these payloads universally refuse. Independent component data grants no additional loader eligibility.
- L356: - [ ] Linux local-filesystem semantics are qualified first. HTTP cancellation obeys configured whole-request bounds; old retained attempts consume real quota and arbitrary same-UID mutation remains outside descriptor isolation.

### `.agents/plans/model-execution/TODO.md`

**Uncommitted**
- L393: - [ ] Establish independent large-CUDA distribution/state accuracy (session task44)
- L466: - [ ] Localize ROCm2B C4 warmup and C16/32/64 warmup/timed divergence (task39),
- L468: - [~] Close ordinary27B HIP HTTP504 and populated1K/2K references/serving (task11).
- L482: - [ ] Continue genuine shared inter-session batching beyond current B8 decode/chunked
- L511: - [ ] Optimize measured CUDA2B/HIP hot paths and run controlled comparisons after
- L964: - [ ] Resolve ROCm2B batch-dependent output mismatch. Retained
- L1012: - [ ] Complete actual27B ordinary HIP/SDK/serve, not the finite two-position route.
- L1016: - [ ] Qualify bounded weight streaming and source/physical final readers.
- L1022: - [ ] Independently audit the original CPU capture before native comparison.
- L1055: - [ ] Run actual-source admission, full-vocabulary/state parity, observer-free
- L1063: - [ ] Finish additive BF16/F32 GDN source storage qualification. Canonical host
- L1147: - [ ] Complete independent CUDA references on l5; Main owns implementation and
- L1290: - [ ] Extend populated contexts beyond accepted0.8B/2B HIP scopes to remaining
- L1295: - [ ] Extend/qualify serving concurrency through C1..C64 and general27B/CUDA;
- L1298: - [ ] Measure matched vLLM workloads as each implemented path passes its relevant
- L1305: - [ ] Owner requirement: independently configure queue depth, active admission,
- L1323: - [ ] Owner30Oct explicitly resumes Qwen ROCm (HIP) on `pvl-x2` independently
- L1330: - [ ] Implement ordinary HIP through shared operators/physical owners, not a
- L1337: - [ ] Generalize actual geometry/backend execution for27B. Finite27B still
- L1363: - [ ] Qualify `.8B` then `2B` full model, cached continuation, faults, shutdown
- L1369: - [~] Every existing/future registry target follows the shared
- L1374: - [ ] Implement/qualify missing full-model/context/backend cells, beginning
- L1403: - [~] Three passes executed on pvl-x2.
- L1419: - [~] Qwen natural2-position independent original comparison and Gemma learned
- L1428: - [ ] Gemma PLE twofixture source draft and prepared controller need corrective
- L1436: - [ ] Close isolated source113 as one checked model unit, then actual x2 source/
- L1447: - [ ] Reconcile frozen models113/P1 hooks and Gemma final-feedback fix with current
- L1450: - [ ] Qwen27: qualify natural and temporal decode/prefill, residency/admitted
- L1456: - [ ] Gemma E2B final-reader ownership: source466 review finds Session/Snapshot
- L1501: - [ ] Gemma E2B/E4B: dependency-based shared-tail fast prefill, PLE row reuse,
- L1505: - [ ] Add functional Gemma E4B and12B routes;12B uses Unified geometry/MM mapping,
- L1507: - [ ] Gemma26B-A4B: learned router/grouped experts/dense+expert composition, then
- L1514: - [ ] Both families: shared continuous serving/temporal work, paged KV/COW,
- L1518: - [ ] Later explicit codec/distributed/MM/LoRA profiles: do not substitute lossy
- L1610: - [ ] Continue Qwen27B/Gemma source/runtime integration independently; merge
- L1616: - [ ] Gemma follow-on: qualify tokenizer/ordinary learned route and broader
- L1619: - [ ] Independent follow-on units continue while short-HF validation runs:
- L1623: - [ ] Gemma trained Rust execution requires exact learned BF16 arithmetic and
- L1634: - [ ] Dense27B: bind full trained source/state through bounded row/layer windows;
- L1638: - [ ] Gemma4 variants: source-real PLE, normalization/logit caps, SWA/global
- L1643: - [ ] Later sparse Qwen: stable finite top-k/tie/block ordering, source-selected
- L1647: - [ ] Later DeepSeek: exact CSA/HCA/MLA/indexer/compressor/mHC/YOCO/Engram and
- L1652: - [ ] Later GLM: distinguish KDA from GDN, MLA/DSA index/open-tail state, mHC and
- L1657: - [ ] **Independent trained BF16/FLA checkpoint gate:** original
- L1712: - [ ] **Next trained reference gate:** qualify exact original HF BF16
- L1720: - [ ] **Following ME-P0 real trained integration:** fuse the
- L1729: - [ ] **ME-P0 next after diagnostics:** wire the analytically verified
- L1737: - [ ] **ME-P0 next numerical stage:** source-exact original Qwen
- L1745: - [ ] **ME-P0 next:** bind source-approved original Qwen3.8 27B
- L1754: - [ ] **ACTIVE NEXT: ME-P0 Qwen3.5 2B then Qwen3.8 27B.**
- L1766: - [ ] **NEXT after Qwen: ME-P1 Gemma4 E2B→26B-A4B→31B**, independently
- L1775: - [ ] Preserve direct one-model-per-process execution and one
- L1782: - [ ] Enforce new owner priority: Qwen3.5 .8B regression, Qwen3.5
- L1790: - [ ] Treat single-model local GGUF execution and multi-model
- L1797: - [ ] After the accepted source-acquisition v1 cutover, use the existing
- L1814: - [ ] Acquire/bind authorized original config/index/tokenizer/checkpoint identities and choose a capable admitted host. Large BF16 sources exceed current local RAM; current executable source limit stays 4 GiB.
- L1815: - [ ] Qualify real large (>4GiB) original trained shard payload on an
- L1822: - [ ] Qwen3.8 27B: original HF grouped-head ordering and Q/K replication 16→48, explicit converted-profile permutations, untied head and source-derived norm/RoPE geometry.
- L1823: - [ ] Gemma E2B/26B: shared GELU/multiply/scale/soft-cap, proportional RoPE, value norm/sliding/shared KV, E2B PLE/doubled FFN and 26B K=V/MoE branches; choose and qualify the exact reference-framework numerical schedule.
- L1824: - [ ] Qualify actual trained CPU operators/full vocabulary/state/nonzero replay, official tokenization/templates/reasoning/stop behavior, sessions, rollback/checkpoints and resource refusal.
- L1825: - [ ] Add the same source-bound forward plans and full physical GPU qualification per eligible device, preserving all state/actual-completion/cancellation guarantees.
- L1826: - [ ] Keep distinct future heavy/hybrid recipes (Qwen Flash-Next, GLM, MiMo and source-defined QSA/MLA/KDA/indexers/hyperconnections), and broaden dense/encoder/pooling/reranking/LoRA support only through independent model/codec/backend gates.
- L1827: - [ ] Preserve CPU-first family priorities; already-authorized parallel basics do not authorize model/SDK downloads or unsupported whole-model claims.
- L1829: - [ ] Bind explicit original HF versus converted Qwen source profiles. Conversion permutes grouped/tiled V projections/constants/convolution/output, uses -exp(A_log) and adds1 to non-GDN norms; GDN swish is SiLU while full-attention gating remains sigmoid.
- L1830: - [ ] Choose and independently qualify Gemma26B reference-framework routing/top-k/reduction/scale placement before production execution. Five analytic test-local routing cases are not trained source parity.
**Deferred — after Qwen/Gemma and Flash execution**
- L3401: - [ ] Qualify independently selected NanoJev-like calibrated decision
- L3414: - [ ] Real current-source qualification remains mandatory before production routing: .8B natural/current-root parity including cached 47→50, physical fault/cancellation/retirement and real-model shutdown, then .8B 512/1K/2K/4K/8K and 16K/32K. Admit 2B contexts only where the exact two-root plan fits.
- L3415: - [ ] Current physical blocker: the approved .8B source is now visible at `/home/pvl/.cache/abird/zerogravity/external-baselines/models/Qwen--Qwen3.5-0.8B/2fc06364715b967f1860aea9cf38778875588b17`; its 1,746,942,600-byte safetensor SHA256 is exactly `04b1c301231dd422b8860db31311ab2721511346a32cb1e079c4c4e5f1fe4696`. The transient `.agents/runs` native bundle was intentionally purged and no `cuda-kernels.json`/legacy `short-hf-native.json` bundle remains under the repo or user cache. Reacquire/regenerate the qualified native/reference bundle through the maintained capture path; do not restore deleted run directories or substitute another model source.
- L3431: - [ ] These are fit/admission results only. Physical correctness, continuation,
- L3450: - [ ] Real physical after-launch cancellation/fault and real-model shutdown
- L3466: - [ ] This provides the committed-root comparison surface for the real .8B
- L3489: - [ ] Execute both phases serially as soon as the approved .8B safetensor shard
- L3518: - [ ] Physical execution is currently blocked by provider identity, not code:
- L3645: - [ ] When the approved source tree, pinned native bundle payloads and pinned
- L4620: - [ ] X2 remains physically dirty and the blocker is now the host KFD/process-

### `.agents/plans/multimodal/TODO.md`

**Uncommitted**
- L12: - [ ] Opt-in, source-granted context-filter outputs as bounded,
- L24: - [ ] Model each ASR (Qwen ASR/Whisper/Canary/Parakeet),
- L32: - [ ] Enable native progressive image output only at complete
- L38: - [ ] Add typed sample/frame/codebook/latent/decision values, rational clocks, checked conversions and complete source-bound component/codec/processor assemblies.
- L39: - [ ] Prove a small real non-token encoder/decision task and bounded iterative task through the existing engine before broad media claims.
- L40: - [ ] Qualify CPU-first original ASR and non-AR TTS (Whisper/Parakeet and Kokoro candidates); then actual backend execution and measured streaming/audio quality.
- L41: - [ ] Add depth-AR/RVQ speech, dual-AR codebook TTS and genuine codec-based duplex neural audio (including Qwen-Omni) where source/license/resources permit; mock codecs or text cascades do not establish native support.
- L42: - [ ] Qualify seeded source-exact image flow/editing conditioner+VAE pipelines, then video temporal/3D-VAE/expert switching and synchronized AV commit groups; candidate families remain separately gated.
- L43: - [ ] Keep ternary PTQ/Hadamard codecs, schema/calibration decision heads and specialist encoders as independent codec/model/task gates, not aliases of Qwen.
- L44: - [ ] Implement RealtimeSession lanes/epochs, bounded media buffers, VAD/barge-in, presentation/ack/playback frontiers and transactional rollback without speculative/partial media leakage.
- L45: - [ ] Add per-model transport capabilities and authorized hosted TaskService adapters with privacy/cost/idempotency/Unknown reconciliation; no duplicate local runtime or guessed realtime guarantees.
- L46: - [ ] Complete MM-01..24 plus model-specific ASR/TTS/image/video/decision quality/clock/resource suites; no real media-model support follows from source-free mocks.
**Tests and verification**
- L59: - [ ] MM-D1/MM-01/MM-15: checked rational sample/frame/codec clocks, 44.1/48 kHz to 16/24 kHz and 24000/1001 or 30000/1001 FPS, long-duration remainder, discontinuity, overflow and bounded parser/buffer refusal.
- L60: - [ ] MM-D2/MM-02: complete source/processor/codec/license assemblies; wrong conditioner/shape/mask, missing source transform and cross-session mutable leakage refuse before use.
- L61: - [ ] MM-D3/MM-03..09: seeded solver/RNG and direct finite decision state, CTC/RNNT/TDT distinction, complete RVQ depth and joint AV commit; interrupted/stale device work never publishes or exposes media.
- L62: - [ ] MM-ASR-01..12: actual frontend/window/blank/predictor/duration/language/timestamp/diarization source fixtures; offline-only checkpoint refuses streaming, malicious audio bounded before allocation, sessions isolated.
- L63: - [ ] MM-TTS-01..12: non-AR phoneme/duration/vocoder and exact PCM count; incomplete 9/10 RVQ frame unplayable, codec-rate swap refused, overlap/voice generation and duplex/barge-in/state/lease boundaries correct.
- L64: - [ ] MM-IMG-01..12: source conditioner/solver/CFG/seed/mask/VAE/color and generation-entrypoint traces; invalid source stage/crop/processor fails, cancel retains old root and unknown physical handles, source/license/hosted status truthful.
- L65: - [ ] MM-VID-01..15: temporal VAE/latent crop/frame count, Wan timestep experts, source tile masks/rectangular attention, AV cross-attention/joint clocks, invalid GOP/backpressure, hot reload, cancellation and whole-assembly admission.
- L66: - [ ] MM-DEC-01..06: finite and joint schema head options/order/null/refusal/calibration against original source; generated text is not source-equivalent, external egress refused without authorization.
- L67: - [ ] MM-BON-01..08: PTQ1_0/PQ2_0 packed source scales/signs/padding plus mandatory 1024 Hadamard activation transform; wrong codec or plain-Q2 path refused, original scalar/CPU/device source parity separately qualified.
- L68: - [ ] MM-13..17/MM-19: real per-session/turn/transport epochs; stale ACK ignored, duplicate reconnect cursor idempotent/authenticated/bounded, playback progress remains uncertain when missing, slow clients do not release device leases or grow queues indefinitely.
- L69: - [ ] MM-18/MM-22/MM-23: ten speech/five video mixed jobs, fair shared physical admission and legal solver/frame yield; no per-packet catalog/router/global snapshot or silent precision/step downgrade, coupled output stays unplayable until both tracks valid.
- L70: - [ ] MM-SRV-01..12/MM-20/21/24: exact versioned SDK/protocol/field/task conformance, hosted accepted-timeout Unknown/idempotent reconciliation, callback/signature/TTL/MIME bounds, malformed media/unauthorized egress/cross-tenant replay refused before side effects.
- L71: - [ ] MM-I0 change-radius drill: small encoder/finite task and iterative mock use existing source/state/resource owners; no central router/worker rewrite or duplicate runtime; original Qwen gates stay unchanged.
- L72: - [ ] Eligible source-real CPU→physical full-model→playable-stream→soft-QoS gates: exact licensed checkpoints/processors/codecs and original quality/state oracles, real negative cancellation/admission/replay tests and measured cold/warm tails/memory. Mock/reference/region/hosted results never promote another tier.
- L73: - [ ] Resolve realtime section18 policy choices explicitly (media numeric policy, first sources, SLO/clock uncertainty, mid-turn changes, persistence, biometric/asset consent, hosted cost/egress, transport packaging and safe physical recovery). User examples do not establish released model IDs or grant execution/provider/download authority.

### `.agents/plans/numerics-and-context/TODO.md`

**Uncommitted**
- L534: - [ ] Implement genuine source-profile multi-row math and call-free lowering.
- L543: - [ ] Owner30Oct expands completion to `pvl-l5` CUDA `.8B`/`2B` and `pvl-x2`
- L553: - [ ] Recover the exact additional2B reference `TextCutlassBf16Sm80` image
- L559: - [ ] Obtain independently trusted active-context comparison inputs. The retained
- L565: - [ ] Finish ROCm on `pvl-x2`, explicitly authorized30Oct without waiting for
- L591: - [ ] Broader CUDA active-context/2B numerical coverage remains open; the CUDA
- L608: - [ ] Resolve ordinary cuBLAS ownership before model math. Fresh phase 2 fails
- L615: - [ ] Resume fresh ordinary comparison using the independently retained digest;
- L652: - [ ] Requalify cached47→48→49→50, affected faults/cancellation/retirement and
- L655: - [ ] Run the final isolated `scripts/check.sh`, then the single serial matched
- L669: - [ ] Remove remaining shared rotary fault assertions/fixed-prompt qualifier
- L677: - [~] Sequential continuation after serving shutdown: fresh .8B prompt47/48/49/50
- L692: - [~] Apply [all-model correctness policy](CORRECTNESS_POLICY.md), based on
- L695: - [ ] .8B initial47/48/49/50 and cached47→50: actual independent original and
- L701: - [ ] Genuine .8B512/1024/2048/4096/8192, then16K/32K context/state/feedback;
- L707: - [ ] **Next one short-HF unit, then matched benchmarks:** both approved .8B/2B
- L715: - [ ] Actualcanonical47projectiondiscovery1 completed original PLAIN+OBSERVED/source/ownedcleanup, but exactcommon state.recurrent.after call0 differs. NewOBS cumsum/KKT/H/output selections differacceptedreference1; selectionvariance is verified, specificcause unproved. Independent actual metadata identity/layout gate now passes separately; no fullmodel/packedargs/native/perf acceptance.
- L716: - [ ] Qualify actual canonical short47input/max8output/EOS248046 and ragged57/62/67/72 prompts; exact source/profile/numerical schedule and original all24/all42/fullV/actual sampler. Extend genuine rotary/ABI/control coverage as required, then original worker/HTTP and C1/2/4/8. Old GGUF diagnostic shapes do not grant HF parity.
- L717: - [ ] Distinct native2B source opening/2048wide ordinary+dualRMS/projection+6144MLP/head/970window tied table+4096B source-row actor. Actual2B prefill PTX differs in cumsum, scaled-dot KKT, recompute W/U and chunk output; matching GDN geometry and different cache keys are not an execution permit. Preserve actual reference3/source/relative5GiB boundary; derive physical fit, then same fullmodel/natural/fault/serving closure. Matched short benchmarks, long contexts and dense27B follow actual parity. End-to-end benchmarks remain unexecuted.
- L718: - [ ] Investigate actual2B discovery2 PLAIN outlier: both lanes finish/source/ownedcleanup pass, OBSERVED matches acceptedreference3, but PLAIN first differs at call2/layer8 recurrent state and branch/residual, then consumed downstream/fullV. Conv.after is only its last3 cache rows; no proof all upstream conv outputs match. Ordinary layer0 RMS10boundaries/565248B match independently. Fullmodel discovery remains unaccepted; source audit confirms timing-based autotuning but no specific cause; unchanged PLAINrepeat1 completed with exact reproduction. Independent RMS leaf reference accepted after complete source/IR/byte gates (565248B boundaries/sourceembedding,4096B gamma); native2048 component physical comparisons/faults/fullgate20 now pass separately, explicit parent-failure boundary retained. Unchanged PLAINrepeat1 reproduces discovery2PLAIN exactly across113713152all42+11168652common+6117576final+565248RMS B, still548/21/3different records versusOBSERVED/reference3. Python-only actual selection2 diagnostic now completes18records/lane and strict source/common/final/prior/all42/RMS gates PASS despite four differing selected stageconfigs. Different configuration alone is insufficient; the earlier reproducible outlier remains unresolved.
- L721: - [ ] Native continuation: genuine47 current-root/position/provider/cache coverage, cached2B histories and whole-model admission remain open. Isolated2B QKVZ/BA phase51 and actual six-node source prefix/GDN phase52 are accepted below; initial1/64 stops before GdnOut. Original full24 permit stays guarded.
- L722: - [ ] Current original2B KKT causal leaf1:29frozenchildren/23pure guards, actual original source/ownedcleanup and ordinaryRMS565248B pass. Four KKT configs/two repeats produce bit-identical A/F32+Ai/BF16 (all72comparisons0), but every solve uses the SAME2warps/4stages selected kernel: this does not compare solve variants. Strict original model reference still FAILS; current run reproduces prior PLAIN outlier byte-for-byte131564628B/27files. No tuner override or failure waiver; KKT config alone does not explain this case. Physical allocator peak remains unknown.
- L723: - [ ] Canonical47 failure narrowed in parallel: three separate PLAIN runs share the same all42 record differences; their OBSERVED lanes match acceptedreference1. Original late-H pair1 now reproduces oldbad PLAIN/good OBSERVED actual state bytes under healthy source/owned cleanup; original raw comparison remains FAIL. Inputs to H already differ: W59BF16words, U9BF16words (token34/head4); K/G/H0/history exact, Vnew=U, finalHT1002F32words differ. Later failed-pair solve-counterfactual1 directly captures equal operands/F32A and first solve.Ai differences, below. This earlier late-H capture alone did not localize the cause.
- L724: - [ ] Post-recompute47 investigation: attempt1 observer refused copied callable-grid assumption; original caller uses literal(1,16), failed attempt/cleanup retained. Narrowv2/28guards/peer review +48files/33parents/21runtime/3historypins PASS; actual attempt2 source/owned cleanup and original pair/common/fullV/all42 gates PASS. Seven actual K/V/beta/G/Ai/W/U records872320B/lane all equal good35; neither lane reproduces badW/U, so no historical cause/fix. Same-input original solve/recompute counterfactual proposed to test the actual observed warp2/warp4 solve selections without production tuner mutation.
- L725: - [ ] Repair47 exact arithmetic binding at existing provider/preparation seam; authenticate declared candidate and actual original code/ABI/launch histories, qualify finite owned solve variants, then connected full47+cached histories. Broad geometry/source equality cannot establish byte equivalence across two proven unequal solve implementations. No tuner forcing, expected-word selection, tolerance relaxation or failed-parent waiver; all18 GDN layer/eight-call identity coverage and fullmodel gates remain open.
- L727: - [ ] Retired2B postAttention1 parent gate remains CLOSED: bothlanes healthy/newlayer0pairedboundaries/source/native14op PASS. Fresh L8 pair3 source/lifetimes/full37paired archives verified after22observer+5host-schema guards: dual/RMS/QKVZ/BA/fullConv/preparedQKV/coefficients/cumulativeG/KKT A exact; FIRST solveAi14BF16differences, W96/U17/H2048. Actualsolve configs2warps/3stages versus4/4 and fullPTX differ; currentHT hashes reproduce prior same-lane split. Its original fullV/all42 failures remain fatal, RMSPASS. Separate new original MLP/ACT reference success and fixedOBS-only component successor do not accept this failedparent. Retired same-input originalsolve/cross-recompute leaf3 PASS: same actual794640B inputs; solve2warps/3stages reproducesPLAIN and4/4 reproducesOBS, Ai14/W96/U17 word split exact under BOTH recompute8/3 and8/4, twice; all12 returned completePTX/cachehashes equal actualoriginal variants. This isolates that split to solve configuration; it does not repair the original parent or prove earlier uncaptured inputs. Previous72 comparisons did not vary solve configuration.
- L728: - [ ] Next: all24/all42/fullV/feedback/cached47 histories and workerHTTP parity. New original2B MLP+activationcapture1 passes strict original common/fullV/prior/all42/RMS/postAttention/MLP reference gates and captures two6144activation packets; ROOTpostvalidation/native2B path/fullmodel remainpending. Earlier failed postAttention parents remainfailed; predeclared fixedOBSpost1 successor target unchanged. MatchedC1/2/4/8 benchmarks follow parity, then longcontexts and27B.
- L729: - [ ] Post-parity overhead follow-up: consider immutable GdnArithmetic same-object pointer shortcut before structural fallback; source-valid review only, no measured speedup or code change.
- L885: - [ ] Qualify original-row Q4_0_8x8 QKV/Z using existing source projection math
- L891: - [ ] Nonzero cached-prior, live producer, full temporal/default-bulk route/
- L897: - [ ] MAIN full formatting still flags only the preserved SiLU draft
- L903: - [ ] **vLLM parity remains open:** implement and physically qualify its own
- L910: - [ ] **Required three-model correctness gate (owner,20 Oct):** resolve
- L920: - [ ] Extend genuine resident GPU context/corpus qualification beyond the currently accepted 512 consumed tokens, including capacity/window/padding/replay boundaries.
- L921: - [ ] Run genuine whole-model 16K/32K CPU/GPU gates only after prerequisites; functional 8K success is not independent long-context numerical closure.
- L922: - [ ] Keep internal full-logit/all-state/trace bit parity distinct from independent native-source stage/top20 comparisons, including the known 2B 56 signed-zero distinctions.
- L923: - [ ] Broaden prompts/tokenizers/precision profiles and sampling quality only with independent oracles, exact history and explicit qualified ranges.
- L925: - [ ] Diagnose identical teacher-forced prefixes512/1024/2048/4096/8192 against pinned llama.cpp0.6.0/build11429 `d812350`, CPU/F32 KV and Flash Attention off. Fix the physical microbatch/chunk/arithmetic schedule and compare per-head GDN state, gain/beta, quantization and AR-versus-bulk recurrence; distinguish replay correctness from arithmetic.
- L926: - [ ] Compare native ubatch1/64/256/512 only within bounded feasible runs; the historical full ubatch1 capture did not finish. Use raw token IDs to avoid incomplete UTF-8 HTTP rows. Recheck short corpus, layers, state and actual8K after any fix; retain original functional0.45 shared-top20-logprob/0.05 top-token-probability limits and strict source-bit operator gates.
**Committed**
- L1286: - [ ] Naturalfeedback/current-root publication and serving integration through originalSession/ResidentProgram. Forced353 diagnostics remainfrontier0/unpublished; candidate IDs[198,599,8,72,18] were not fed back. Separate source-staged per-call binding/publication unit in progress. No serving/performance/fullmodel-default-route acceptance; newimplementation remainsISOonly.
- L1288: - [ ] Native2B original-source all-state/natural-feedback qualification and serving integration. Reference success does not establish Zerogravity inference, a driver hard cap or unknown transient highwater; actual baseline is separate from requested relative memory. Benchmarks/longcontexts/27B runtime remain OPEN.
- L1303: - [ ] Real current-source qualification remains mandatory before production routing: .8B natural/current-root parity including cached 47→50, physical fault/cancellation/retirement and real-model shutdown, then .8B 512/1K/2K/4K/8K and 16K/32K. Admit 2B contexts only where the exact two-root plan fits.
- L1304: - [ ] Current physical blocker: the approved .8B source is now visible at `/home/pvl/.cache/abird/zerogravity/external-baselines/models/Qwen--Qwen3.5-0.8B/2fc06364715b967f1860aea9cf38778875588b17`; its 1,746,942,600-byte safetensor SHA256 is exactly `04b1c301231dd422b8860db31311ab2721511346a32cb1e079c4c4e5f1fe4696`. The transient `.agents/runs` native bundle was intentionally purged and no `cuda-kernels.json`/legacy `short-hf-native.json` bundle remains under the repo or user cache. Reacquire/regenerate the qualified native/reference bundle through the maintained capture path; do not restore deleted run directories or substitute another model source.
- L1319: - [ ] These are fit/admission results only. Physical correctness, continuation,
- L1338: - [ ] Real physical after-launch cancellation/fault and real-model shutdown
- L1354: - [ ] This provides the committed-root comparison surface for the real .8B
- L1377: - [ ] Execute both phases serially as soon as the approved .8B safetensor shard
- L1404: - [ ] Physical execution is currently blocked by provider identity, not code:
- L1531: - [ ] When the approved source tree, pinned native bundle payloads and pinned

### `.agents/plans/pipeline-composition/TODO.md`

**Uncommitted**
- L142: - [ ] Finish remaining qualifier/fault and captured testcase dispatch boundaries
- L154: - [ ] Extend source-bound .8B qualification to fresh47–50-token prompts, genuine
- L157: - [ ] After source-bound correctness and legitimate source access: current
- L166: - [ ] Natural independent-reference and explicit Gemma row-hit/transfer scopes;
- L175: - [ ] Map each applicable strategy to existing primitive→model operation→family→
- L181: - [ ] Preserve distinct numerical strategies: source113 CPU-order BF16 versus
- L192: - [ ] Use temporal/ragged pipeline liveness for Gemma E2B/E4B shared-tail fast
- L195: - [ ] Adopt source-bound phase selection for2B cachedB4 through the existing
- L226: - [ ] Implement the Astra next-native redesign
- L240: - [ ] Extend the original HIP actor's bounded prepared-set lifecycle for multiple
- L286: - [ ] Record Qwen27B's one family traversal through the existing backend/plan
- L291: - [ ] Qualify the integrated `TextCallGeometry` extraction and actual Gemma
- L294: - [ ] Move remaining closed family builders/geometry out of core after generic
- L297: - [ ] Use one prepared-model serving adapter for validation/session/accepted
- L299: - [ ] Implement sealed NativeCohort in the SAME node loop: authentic perrow
- L306: - [ ] Share immutable recipe/topology/SSA/storage/native candidates inside the
- L329: - [ ] Before Gemma HIP merge: retain authentic source/snapshot lifecycle through
- L339: - [ ] Demonstrate one forced ordinary versus optimized region/profile slice
- L342: - [ ] Later multidevice/MoE reuse: source/range/codec identity separate from
- L358: - [ ] Carry eligible stage outputs through typed leased device bindings with
- L362: - [ ] Prepare cross-stage resident handoff, device chains/graphs, exact fused
- L366: - [ ] Plan joint placement/layout/liveness and admitted scratch/graph/staging
- L370: - [ ] Keep qualified intermediates, model state and compact generation feedback
- L375: - [ ] Explain and attribute each required materialization/copy/readback/wait and
- L378: - [ ] Prove zero intermediate H2D/D2H payload traffic and no stage-only host wait
- L390: - [ ] Support explicitly approximate profiles beyond derived weight precision:
- L394: - [ ] Permit interchangeable allocator/pool, provider, codec, scheduling,
- L399: - [ ] Resolve one shared versioned configuration visible to participating
- L403: - [ ] Apply audited MIT/Apache static-native reuse first to every implementation
- L407: - [ ] Consolidate profile/phase/backend equivalence acceptance: solo/concurrent,
- L411: - [ ] Exercise the first useful region/profile with one reused or ported
- L423: - [ ] Define primitive semantics separately from their kernel implementations;
- L425: - [ ] Define a pipeline node that may reference a bounded subpipeline: one
- L428: - [ ] Define immutable profiles combining named pipelines and permitted source/
- L432: - [ ] Compile hierarchical views into existing semantic/physical plans; legal
- L436: - [ ] Specify child-region effects/exports, alias/liveness after fusion, late
- L441: - [ ] Profile examples: resident interactive, concurrent throughput, long/shared
- L444: - [ ] Compare minimal efficient kernel paths with ordinary decomposition and
- L447: - [ ] After parity and matched baseline, measure source-only preparation
- L451: - [ ] Review extension drills for a new primitive/model/provider/task without
**Deferred**
- L767: - [ ] OSS release/distribution naming and "latest" pointers remain
- L769: - [ ] Remote routed pipelines/peers, encrypted hosted gateways and
- L784: - [ ] Connect authenticated application/conversation history providers to
- L792: - [ ] Source execution: bind actual learned model, tokenizer, processor
- L796: - [ ] Single-process multi-model scheduler: immutable source residency,
- L801: - [ ] Pipeline runtime/hot reload: per-session model identity, graph generation,
- L805: - [ ] API serving: explicit auth/bind/route config and tested OpenAI
- L809: - [ ] Context-aware source-real voice pipeline: stage-specific ASR prompt/
- L819: - [ ] Actual staged ASR + diarization (speaker-tagged timestamps), ASR
- L824: - [ ] Optimize model execution: Qwen3.5 2B, Qwen3.8 27B
- L829: - [ ] Complete remaining exact Q2/Q3.x/Q4 quant source pipelines,
- L834: - [ ] Trained and streaming multi-model parity, real media correctness,
- L841: - [!] HIPv4 first healthy57492 refused before child/model/GPU at old-Q import receipt schema; actual source/binary/owner closure preserved, no inference result. Controlv5 real metadata-only preflight accepted both actual receipt schemas but independent review found missing finalizer import alarm_until; distinct control-only successor required before fresh actual serial cases. No source rebuild/import/copy/download needed.

### `.agents/plans/platform-and-release/TODO.md`

**Uncommitted**
- L12: - [ ] Owner23Oct: bring Vulkan execution to CUDA feature/correctness parity;
- L21: - [ ] Apply the owner-approved [static-native reuse/direct-port rule](../pipeline-composition/PROFILE_POLICY_AND_COMPONENT_REUSE.md#reuse-first-across-all-implementations-and-harnesses)
- L25: - [ ] Prove actual final binary linkage for selected reused components across
- L30: - [ ] Verify shared effective-configuration explanation/revision handling and
- L33: - [ ] Generalize the working typed forward plan into verified effectful semantic regions/task control, closed provider substitution, bounded shape/kernel caches and explicit command/resource legality; no new persistent IR/runtime without a real need.
- L34: - [ ] Prove artifact→model→compiled task→coordinator→output and explain-plan for supported tasks, including source-qualified preselected fallback and no selected-GPU-error CPU substitution.
- L35: - [ ] Run extension drills for one stateful semantic op, device/provider, non-token TaskProgram, quant/storage/API adapter and optional ParameterVersion/TrainStep seam; future training is not thereby implemented.
- L36: - [ ] Finish per-file source/model/license/NOTICE/SBOM policy, malformed-input fuzzing and ownership/security review, authenticated APIs/private caches and qualified driver/device safety.
- L37: - [ ] Verify Nix/module/native/cross packages and installer/release behavior on actual Linux/Windows/macOS target access; packaging definitions alone do not prove platform support.
- L38: - [ ] Maintain minimal/default/composed CPU-free feature isolation, one lock/version authority and truthful unsupported capability behavior.
- L39: - [ ] Gate bounded compiler/autotune/AOT caches by source/device/driver/shape and retain negative candidates; no unbounded first-request search.
- L40: - [ ] Complete reproducible/signing/release policy and fair model/task/quality/performance matrices with cold/warm, p50/p95/p99, memory/energy and uncertainty. Push/tag/release remain separately authorized actions.
- L41: - [ ] Demonstrate compatibility/invalidation of operator/task/state/plan caches and bounded change radius before broader platform claims.
- L43: - [ ] Before a true zero-copy strided View is enabled, validate contiguity against stored value provenance and test public PlanValue retargeting. Current View physically materializes dense output; retain cached-element and dynamic range guards.
- L44: - [ ] Preserve one GPU-library service/queue/quarantine owner and cold generic-plan admission through packaging changes; verify actual final downstream links, not only compiler/type checks.
**Tests and verification**
- L99: - [ ] Qualify all six [CPU release definitions](../../../docs/DEVELOPMENT.md#cpu-cross-release-definitions) with actual target execution, installer checksum/version/refusal cases and platform signing/authenticity policy. Build definitions and same-host checks do not establish other-target support.
- L100: - [ ] Test enabled NixOS service lifecycle and runtime-secret handling after serving is implemented; the currently disabled service definition and module evaluation are not a running-server claim.

### `.agents/plans/serving-and-speculation/TODO.md`

**Uncommitted**
- L158: - [ ] Next resident projection unit: inspect existing authenticated source/resident
- L166: - [ ] Parallel CUDA correctness unit: localize and fix the deterministic C4/B8
- L169: - [ ] After each coherent fix: rerun only the smallest real-model cells needed to
- L236: - [ ] Treat X2 as a **shared physical resource**, not an exclusive test card.
- L248: - [ ] L5 remains the source/repository authority. X2 is execution/testing only and
- L254: - [ ] Finish ordinary correctness on `pvl-l5` CUDA `.8B`/`2B` through genuine
- L257: - [ ] Qualify real C1/C4/C8/C16/C32/C64 scheduling, independent state, faults
- L259: - [ ] Then optimize and execute every feasible matched vLLM cell under the
- L262: - [ ] Use x2 resources as needed; continue without routine permission prompts.
- L272: - [ ] Owner resumed. Owned GEMM/RMS/rotary/attention now pass exact ordinary `.8B` cached47→50: full logits/all42/natural feedback and normal shutdown. Normal/fault lifecycle and final gate pass; long-active-context/2B numerical scopes need independent reference inputs; see [numerics](../numerics-and-context/TODO.md). Original cuBLAS private interface remains correctly refused; no alternate allocator or relaxed reference.
- L273: - [ ] Then exact ordinary cached47→50 comparison, real after-launch cancellation/fault/retirement/shutdown, admitted context correctness and assembled gate. Reference capture alone closes none of those gates.
- L274: - [ ] Owner30Oct: continue through Qwen ROCm (HIP) on `pvl-x2` now; do not wait on wider CUDA reference gaps. Needed approved2B/9B/27B downloads may use canonical XDG cache. No subagents or benchmark detour; serving remains disabled.
- L290: - [ ] Extend retained usage/timing for exact context prompt tokens and explicit
- L293: - [ ] After ordinary `.8B`/`2B` correctness, cached/fault/shutdown, long-context
- L300: - [ ] Replay applicable original-model/backend shutdown and fault scopes under
- L496: - [ ] Profile and fix the .8B gap first, then beat the same matched vLLM C1/C4
- L532: - [ ] Remaining .8B C4 optimization: source-safe HF physical coalescing/batching.
- L543: - [ ] After source/state/natural-serving parity, execute established short
- L548: - [ ] After the already authorized matched benchmarks complete, implement
- L558: - [ ] Compose stable leased rows/diff metadata, compact readout, qualified
- L563: - [ ] Extend full-state branch verification before any proposer: prefix0..K,
- L568: - [ ] Retain profile/shape/layout/source/context-grant compatibility at every
- L585: - [ ] Integrate both accepted GGUF/reference and newly qualified vLLM numerical
- L588: - [ ] Implement original-plan paged KV append and block-table QK/softmax/AV,
- L593: - [~] Extend bounded refill/regroup into fair aged chunked prefill/decode
- L956: - [ ] Run profile-matched batching/paging/prefix-hit/overlap ablations and genuine
- L983: - [ ] **P12 ordered SiLU integration:** private CUDA B1/B2/B4/B8 operator
- L998: - [ ] **Stock2B loader and matched benchmark:** genuine pinned stock2B
- L1029: - [ ] **Production CUDA graph admission:** signed `5bf440d` first made the
- L1073: - [~] One later repeat observed a transient post-capture total resident-byte
- L1144: - [ ] **P12 fused continuous GPU serving:** extend the accepted CUDA B2/B4/B8
- L1168: - [ ] Connect accepted core `ContextOwner`/`ContextReader` snapshot
- L1180: - [ ] One router must snapshot each enabled, source-granted context
- L1193: - [ ] Make YAML/repeatable-flag compiled pipeline configs a first-class
- L1202: - [ ] Distinguish model-trained MTP weights and Qwen Flash-Next
- L1208: - [ ] Implement canonical TaskRequest/coordinator workers and CLI/SDK/API integration, model routing/residency lifecycle, explicit supported OpenAI/Anthropic interfaces, streaming/tools/reasoning/schema/sampling and truthful capabilities.
- L1209: - [ ] Enforce separate API/tenant payload, queued-prefill/residency and physical active-step admission; bounded overload, cancellation, disconnect, stale epochs and output backpressure.
- L1210: - [ ] Add real compatible continuous/grouped batches, aged chunked prefill, decode/media fairness and finite joint batch/draft/expert/kernel/placement plans; no per-token catalog/JSON/global-lock work.
- L1211: - [ ] Finish Qwen basics before optimizations; enumerate the complete pinned llama.cpp/vLLM union, public and internal surfaces. Preserve 105 provisional groups/145 document seeds and CLI/API frontiers as unreviewed inputs, not exhaustive parity.
- L1212: - [ ] Implement every applicable sampler/logits-processor/RNG/logprob/grammar/tool/stop/reasoning composition and its independently verified outputs.
- L1213: - [ ] Qualify target-authoritative MTP, draft model, n-gram/suffix, EAGLE, DFlash/DFlash2, DSpark and related registered strategies; DSpark2 exact primary source/version remains unresolved.
- L1214: - [ ] Test all accepted-prefix lengths, target-distribution correction, target/draft RNG/grammar ownership, full KV/recurrent/index/media rollback, physical branch admission and authentic cancellation/quarantine.
- L1215: - [ ] Jointly benchmark material combinations, adaptive depth/backoff and all offered/rejected/canceled work: committed SLO-goodput, TTFT/ITL tails, fairness, memory and energy versus equivalent pinned vLLM/V1/Omni.
- L1216: - [ ] Preserve zero untriaged frontier/disposition requirements before any full-parity claim; synthetic acceptance, fewer launches and competitor measurements do not establish local parity or speed.

</details>

## Appendix C — complete Git chronology since 10 Oct 2026

Git is the exact source-history authority. This generated ledger includes every
commit reachable from current HEAD in the project period, including reverted or
superseded experiments. Read the owning diff/TODO before treating a commit as selected.

<details>
<summary>Full commit ledger</summary>

~~~text
be95424837deb7919adf34503db71d27d3ad3adb	2026-10-13T12:00:00+08:00	docs(design): product, architecture, prior-art and research plans
1e10b5cc1d5cf34832477bc83ef39117a765c6a0	2026-10-13T12:00:00+08:00	feat(core): immutable source identity, state roots and physical resource ownership
af9a5d3969c9a87951990223b6d289aba69c63fb	2026-10-14T12:00:00+08:00	feat(artifacts): sharded source artifacts and model identity
b276fd01e421f1b7e416ae773610b5823b9da36a	2026-10-14T12:00:00+08:00	test(artifacts): cache transactions
586d440b144727bdd9acea7455afb27f6c38ae25	2026-10-15T12:00:00+08:00	test(artifacts): qwen text contract
782e6fc48f0924cc055232d56338da62b68b1a79	2026-10-15T12:00:00+08:00	feat(cpu): portable CPU math and weights
1fd12da7840151ba47ab43368b4e4f435aac3d3b	2026-10-16T12:00:00+08:00	feat(model-qwen35): source-verified Qwen3.5 forward and tokenizer
901ba9ae2f2ed531d992a6d657a37dc362518bd0	2026-10-16T12:00:00+08:00	test(model-qwen35): source layout contract
e6ecf02bb7d951e01e66b430a24553c690b4ac5b	2026-10-17T12:00:00+08:00	feat(model-gemma4): Gemma contract, MoE and 26B variant
81642a9af4d88d928327a1012af1594168704956	2026-10-17T12:00:00+08:00	test(model-gemma4): e2b contract
7aa19c268026069ad916f6a69440dab602aaa94a	2026-10-18T12:00:00+08:00	test(model-gemma4): family contract
65b026f902501e85afad5fd31f3ca53968a3e836	2026-10-18T12:00:00+08:00	test(model-gemma4): variant 26b contract
b03c16451d65c33a05b3c495c1205240db4ccac0	2026-10-19T12:00:00+08:00	feat(data): vendored CubeCL license and provenance
f491e34fe55e67eed434ac577c32d44eca51430e	2026-10-19T12:00:00+08:00	feat(data): vendor CubeCL cubecl-ir 0.11.0
627c58ec80ff7b33fa1ce5086a729946c5b1cd44	2026-10-20T12:00:00+08:00	feat(data): vendor CubeCL cubecl-cpp 0.11.0
3664f975e79242d8b1a12bc4eefe886f81a33c2f	2026-10-20T12:00:00+08:00	feat(data): vendor CubeCL cubecl-cuda 0.11.0
f77ae37ec0317f39a2e1ce4466d82c645cd344da	2026-10-21T12:00:00+08:00	feat(data): vendor CubeCL cubecl-hip 0.11.0
7d768a080a84b6132595f7cadbc84b75b69106e4	2026-10-21T12:00:00+08:00	feat(gpu): source-exact resident GPU execution and physical ownership
a9723bceb8ad4aa4df639dbe660efcae6c22711b	2026-10-22T12:00:00+08:00	feat(runtime): facade, semantic program, task drivers and state publication
c3d0d0b06fb1548c8bf3db10f2965e5af52be9bc	2026-10-22T12:00:00+08:00	test(runtime): gemma contract
4add50a023d4fb5dd989989cab75ac4298d46d29	2026-10-23T12:00:00+08:00	test(runtime): layout api
b4a36b829031fba498e45bebb71a6cd8f01e7ef9	2026-10-23T12:00:00+08:00	test(runtime): multimodal mm0
a691e21c040d2dcf46a52f2cbcf0e0eb934c8369	2026-10-24T12:00:00+08:00	test(runtime): qwen real model
25ec3e27e3c208158e9763944026003005456461	2026-10-24T12:00:00+08:00	test(runtime): qwen shared model
047e3b70bbcbd7acbb5176712208cc4bcc02dd29	2026-10-25T12:00:00+08:00	feat(cli): single-binary inspect and run
52e37313bff57c2a7e831f50e9236fa983e13d02	2026-10-25T12:00:00+08:00	test(cli): CLI contract
22e0cc5abb4a067a95324206c2881de9e3327427	2026-10-26T12:00:00+08:00	feat(tooling): workspace GPU qualification crate
4d82c6704dad47875fa42bdec65f9a65de79722e	2026-10-26T12:00:00+08:00	test(tooling): independent reference oracle llama_reference_layers
a2d6a02f1577554ea8dba669f77ba7b903c1bb29	2026-10-27T12:00:00+08:00	test(tooling): independent reference oracle llama_reference_logits
a8ea8926b53d7b4a9c6d00f3d9be04906f8d64bb	2026-10-27T12:00:00+08:00	test(tooling): independent reference oracle llama_reference_qk
6a0079eb167d1f1593fe2a9372039138cd949770	2026-10-28T12:00:00+08:00	test(tooling): independent reference oracle llama_reference_recurrent
be88e30733f9459eda5f150bdbd70d9d0f8c6684	2026-10-28T12:00:00+08:00	test(tooling): independent reference oracle llama_reference_rope
77778a2c258751c64c56c4ffd3dba2ff70f7296c	2026-10-29T12:00:00+08:00	test(tooling): independent reference oracle llama_reference_softmax
6015241a5f5c2bb7ee65715a998988a026ff800c	2026-10-29T12:00:00+08:00	test(tooling): independent reference oracle llama_reference_value
e190aff1fdd392696e8086010a7bbc1f12d3df4f	2026-10-30T12:00:00+08:00	test(tooling): independent reference oracle llama_reference_value_padding
0aea8de66e3eb3b704e7d26687731a3e5b07d3bb	2026-10-30T12:00:00+08:00	test(tooling): build-release-artifacts.sh
057987203d5de45ad268ba02cfc5084f76f48229	2026-10-31T12:00:00+08:00	test(tooling): check.sh
c74450a27b1e3080e9576b0f95922f17c7cdca8b	2026-10-31T12:00:00+08:00	test(tooling): compare-attention-value.py
9f54d2435fe6af15b3eaf28980bc1039ad02b559	2026-11-01T12:00:00+08:00	test(tooling): compare-qwen-layers.py
7458ffc286f7e72c70ac18acfd3f776186aa204a	2026-11-01T12:00:00+08:00	test(tooling): cpu-qualification-gate.py
73cb6e55e1795e2c13533f60621dc29619cf5e2c	2026-11-02T12:00:00+08:00	test(tooling): native-gpu.py
80145a7f91724fb34a6f67e9972850a279982bf3	2026-11-02T12:00:00+08:00	test(tooling): test-install.sh
ca1138b8cf866c81bbfb276358d03a62576df16a	2026-11-03T12:00:00+08:00	test(tooling): test-native-gpu.py
5a0636c3f9c744c2f979ded40943a269ba5a1176	2026-11-03T12:00:00+08:00	test(tooling): test-numeric-resume.py
9a6c504ebef037e09bd187d65e2bec7e2e456eef	2026-11-05T12:00:00+08:00	chore(workspace): restore the locked full workspace manifest
57a9ad7fe78bf7c5e64d3870a6ba176ceba4cc07	2026-11-05T12:00:00+08:00	test(tooling): independent reference probe notes
1af2ac434279cb5f4b635a51884c936f19b54aa3	2026-10-16T21:39:45+08:00	fix(nix): give module check a complete base system
0452c594d99df6d3b4c972ceaae8df6ea56925e3	2026-10-16T23:45:32+08:00	docs(plan): record module eval check fix
2afa32aa74aff7f35ec1b52e4caa7079a5f68980	2026-10-16T23:45:32+08:00	docs(readme): reorder engineering and FAQ sections
a054d396e1ec12fc5e9b8e2ca60e8060954870c5	2026-10-17T08:33:07+08:00	docs(agents): document plan layout and TODO spec
cc3cab27a2713fdb64bb3f1a42c8138c060c5dce	2026-10-17T08:33:07+08:00	docs(contributing): align guide and AI disclosure
7e61cb63efd0c1429aa96044f2f7d28fda1e59aa	2026-10-17T08:41:24+08:00	docs(agents): gate pushes and destructive actions
9239719c25128028fd78a6a60f817ce6b626517f	2026-10-17T09:54:54+08:00	docs: retire historical archive bodies
6deebf22228315863f563723b5e6761ac6b51c5c	2026-10-17T10:11:15+08:00	refactor(cache): namespace model cache under abird/zerogravity
fa6affa425089fdf9a15bfa10b330e03f738fec8	2026-10-17T10:30:09+08:00	feat(artifacts): add pinned model pulls
3c97c52075563f65928c564818ce73502168778f	2026-10-17T13:44:29+08:00	feat(artifacts): resume interrupted pulls
baa9d8818061958a63eabeb61a7239a103a738f0	2026-10-17T14:12:22+08:00	feat(artifacts): load pinned cache components
20f449d20320b6f2912698c7b97cfdfd82472ba2	2026-10-17T14:37:44+08:00	feat(artifacts): recover missing pull sizes
f44264c017856a015584a094566a077638939e55	2026-10-17T15:03:41+08:00	feat(cache): add protected snapshot removal
ea700f67fca7ff7871613044a989e03b8ad6f9d7	2026-10-17T15:23:55+08:00	feat(pull)!: add capped auxiliary streaming
03b69482a5fb10b8c1cc7032e7cbdf39e2e5fead	2026-10-17T15:41:45+08:00	feat(pull): pin multi-asset release components
e44f4ee8020d5661dd8167cb07dfcb085906fcb7	2026-10-17T16:16:36+08:00	feat(cache): import local HF snapshots
fce1353e649df2eb73fe6a39437043fc8551aef5	2026-10-17T16:33:28+08:00	feat(cache): qualify cross-filesystem imports
e0073a5593e5641494c0f48e8d2dcc9e9e61a119	2026-10-17T17:05:07+08:00	feat(pull): add pinned HTTPS objects
e82d76b8f8e91f7d4b49c211e26baa081ff523e5	2026-10-17T17:14:19+08:00	feat(pull): add pinned OCI GGUF layers
f74a525366e6280a9806ecaabd7e1f4df46a6a97	2026-10-17T17:26:28+08:00	feat(pull): add version-pinned GCS objects
833d68f716783c255b24691552a081a1d2b88e76	2026-10-17T17:51:00+08:00	feat(pull): add versioned S3 and SigV4
e62222276e0dac62ed554a3fec2f2ae86edad497	2026-10-17T18:13:08+08:00	feat(pull): add pinned Azure blobs
87f6454fb7c4832262d36104a806baad9a9734a7	2026-10-17T18:24:38+08:00	feat(pull): add pinned ModelScope files
223ca794c9d45a0e246ad1cd9ba8613f2f203cd4	2026-10-17T18:33:14+08:00	feat(pull): add Xet hash and chunk primitives
a774ded154ed066c8f4b134ce7f5cba32ccdd09c	2026-10-17T18:45:39+08:00	feat(pull): add cloud content contracts
837e77f5080c0c2c07fee90b2ef9fb791068a7d3	2026-10-17T18:55:23+08:00	fix(pull): enforce offline acquisition rules
52e42fcdf0e7bead1ebefe55d8103501c7bd4f64	2026-10-17T19:15:13+08:00	feat(pull): add native Xet reconstruction
46cf5a5830e7e8d8096234341ef51fafdf63cc16	2026-10-17T19:24:22+08:00	feat(pull): add conditional cloud reads
4db5efbcce973d21bf1c8ca750471e6b2ab956ab	2026-10-17T19:34:10+08:00	feat(pull): recover pinned ModelScope sizes
f22f75972e63ea70771c59fbce065e371d5f29d4	2026-10-17T19:43:00+08:00	feat(pull): list pinned file candidates
f197c9cf4a44e06a3c1cf3c532855545bed945fa	2026-10-17T19:52:21+08:00	feat(pull): select explicit GGUF quant families
1a22597d9fda50ad71c43ae08f8660fb5b014218	2026-10-17T17:44:06Z	docs: preserve complete October 17 takeover and paused sources
04363c3e83368517475b6dccd4894ff437639927	2026-10-17T17:44:24Z	feat(pull): qualify bounded full-path filters and exclusion closure
8195c18e43bd2aba51d727c03ebe03df6d1fd166	2026-10-17T18:12:21Z	feat(pull): reconstruct verified required native Xet indexes
f56b57400eb913240d6e2b2eeea5d60d9ab6f35e	2026-10-17T19:30:00Z	feat(pull): authenticate versioned AWS FULL_OBJECT SHA256
d5aa5cb0ba10d2603ae34488e54755c5495f63b3	2026-10-17T19:48:40Z	feat(pull): finish v1 original-model companion closure
fee0bb2e2d690956069a0de151770c21168bea63	2026-10-17T19:57:15Z	feat(pull): close official Qwen and Gemma source metadata
4bf624e300254f845e9e047fcb344026c9252d5d	2026-10-17T21:31:11Z	feat(pipeline): scaffold shared multimodel routing and opt-in chat context
a9984a47052baaf1aeda558154256d361dcc64af	2026-10-18T09:26:30+08:00	chore(gitignore): share one canonical ignore block
ac44e836b46f6c5aff44afa298bdb342ff65433e	2026-10-18T02:23:14Z	feat(pipeline): compose source-scoped context filters and typed decisions
f28a62ceb47229279b70283f4e88747d1719e2ed	2026-10-18T03:23:06Z	feat(context): snapshot authorized committed history for pipelines
f0746d1d7e22b94b55c93cb734a8c4ad3b2ab67a	2026-10-18T04:38:18Z	feat(pipeline): finish YAML-first typed multimodal scaffolding
93a562288f0e00240da92c37c9dd7f4b846d62d3	2026-10-18T05:21:30Z	feat(source): bind verified immutable safetensors shards
b796a2e17a4f20086c4db59743aea9558ae1a892	2026-10-18T06:27:12Z	feat(qwen): bind original source tensors to shared CPU value operators
b11e58cb9ef7dd0d607f301d90c1870897cebd34	2026-10-18T07:23:47Z	feat(qwen): qualify original hybrid layer CPU proposals
1b761119c2de2690723b9531b1d43660a66c1001	2026-10-18T15:29:41+08:00	docs(agents): unify shared conventions and plan model
631e59fd614fa6576985e8ff8bf5503d88f99359	2026-10-18T15:30:21+08:00	docs(plans): migrate historical plans to the unified model
6329cffbb3e1b93811235867fa697f5f76f3d27a	2026-10-18T08:18:40Z	feat(qwen): admit source-bound original layer states atomically
e43ba4c63106bab9804ff0f581457ce4f370741e	2026-10-18T09:16:57Z	feat(qwen): compose original full-model FP32 diagnostic forward
28e47f5a4728679e33659298db043da728f225c1	2026-10-18T17:17:44+08:00	docs(agents): add terse-writing guidance
7e328411a8de3c0cc2b9cb9690f2e4a0aa9a54c9	2026-10-18T17:24:36+08:00	docs(agents): strengthen ownership and commit-scope rules
aaf9ba6ec94a9d1bbd4e09b747d9338637fc9e87	2026-10-18T10:09:54Z	feat(qwen): qualify source-bound BF16 dtype diagnostics
1c0bb41f2e7b2f635b255d5285040ed15b9c25b6	2026-10-18T11:26:23Z	feat(qwen): bind pinned source bundles to numerical reference traces
e90ff8868c46f321ff8bd0bf4c2154a71606aab2	2026-10-18T19:38:06+08:00	docs(agents): add terse-output guidance
35b90932f9bdf93127059df426e7d7eeaee54cf5	2026-10-19T00:19:08Z	feat(serve): add bounded OpenAI CPU router and verified SSD extents
9790fa2ef6c4993be0b93014ad1e63d5db5cabbb	2026-10-19T02:24:07Z	feat(serve): admit resident GPU requests with fair wave scheduling
114bba566bb2d150afe5f8a0fb5701880cae713f	2026-10-19T03:59:32Z	feat(gpu): qualify native multi-row F32 projection on CUDA
2869f8784d986dd80973da03afafde89da34c947	2026-10-19T05:11:09Z	feat(gpu): qualify batched attention GDN KV and elementwise operators
08be9c64ea60b5dcf4391c54cc183d63666e4a70	2026-10-19T07:43:08Z	feat(gpu): qualify resident B2-B8 state cohort publication
3ec431a2df62bbcae5091a048f285801f151da7e	2026-10-19T08:04:35Z	feat(gpu): qualify native batched GGUF quantized projections
c85ad250334f590a87c99710e95c71a3deb28323	2026-10-19T08:41:20Z	docs(agents): preserve complete Zerogravity history and P12 handoff
35ebc5b343fd82201eac768edf86ba7a81c41846	2026-10-19T18:08:46+08:00	feat(gpu): fuse native resident model batches
904a190cf39aca74e06c979dbaddb0db7f4404f1	2026-10-19T18:09:43+08:00	docs(agents): record native batch acceptance
54fd580815dbcfcf0916e4e737beaf21728818b2	2026-10-19T19:51:39+08:00	feat(gpu): keep batch greedy feedback on device
570d865696ab398f0c5c37f08c5c951595eadd75	2026-10-19T19:52:16+08:00	docs(agents): record batch feedback acceptance
2677d8b664c78b657c26e7bf46c835317b003913	2026-10-19T22:33:08+08:00	feat(serve): run native CUDA B2 waves
7fc9ad475a30a3d2120f307a7da2ff75a80e337a	2026-10-19T22:33:38+08:00	docs(serve): bound native B2 acceptance
f26232e8b03bf5ab3c13e00ed641eb0d602360d2	2026-10-20T00:28:17+08:00	fix(qwen): honor tied original output heads
6bfcdbe72f0133a6c79e2e7de169c3de74b3da8a	2026-10-20T00:30:04+08:00	feat(bench): account for offered serving work
d9fa829ceba092c7f8a3fa9c2746a11a0906eaba	2026-10-20T00:40:08+08:00	feat(serve): qualify native CUDA rows 4 and 8
9815504a6975ce766766fc8089729be8121df3ac	2026-10-20T01:13:36+08:00	feat(qwen): bind original HF diagnostic profiles
dbb5f4f4cc42e28c91b12e906f9091ef76510c03	2026-10-20T01:53:12+08:00	feat(serve): fuse unequal decode frontiers
cadcbd9d1216b00ca4b69cae9bc04a409e347193	2026-10-20T02:07:20+08:00	feat(qwen): qualify pinned CPU HF operators
1b599df06d8d7c8c4912838836b272e28e04aca1	2026-10-20T02:08:49+08:00	feat(bench): isolate owned provider lifecycles
d41163432fd6233c7bd88f07b64fb71b1086a902	2026-10-20T02:11:18+08:00	feat(artifacts): own verified tensor windows
2a32aff615fc4f7f6718f14a15187825730d6897	2026-10-20T02:43:37+08:00	feat(qwen): compose typed source operators
c5eb1f8e717c79146937e585a145a0052eff747b	2026-10-20T03:22:04+08:00	feat(cpu): qualify source sigmoid and RMS
d0a60844217fc106fe8c70bb3a99073eef0387c8	2026-10-20T04:19:25+08:00	perf(gpu): fuse source activation preparation
5350a4d40b535106240d48fec8aedd459d5cd17a	2026-10-20T04:56:45+08:00	feat(cpu): qualify typed source windows
f78892957f09a84838246aede4fef2be040a0d18	2026-10-20T06:13:20+08:00	feat(gpu): fuse native teacher-prefix serving
bfe62df68d763b9d5c6b87c891be6a62b3593de9	2026-10-20T06:35:48+08:00	feat(cpu): execute leased source-window plans
fe2a34213009cc160f4d82068bc7dd1f49d05c6c	2026-10-20T08:12:19+08:00	feat(cpu): qualify original attention fragments
9a199017f18f1bcb1a8053685be3c3c29236ec90	2026-10-20T08:21:04+08:00	feat(gpu): qualify batched graph replay
fff1842594f8011b761d08fcc1690311b72543e3	2026-10-20T09:20:27+08:00	feat: qualify trained BF16 operator slices
fb68d8fa165de16e21d1bd37afef3790b80da971	2026-10-20T13:27:39+08:00	perf: combine GPU state scatter certification
bb089823653f89d64bc97935cd7b11b7c3e66fd3	2026-10-24T16:55:25+08:00	feat(core): add run policy request validation
bd277cf7667381333d54ff3c7f249dae92b8efa6	2026-10-24T16:55:25+08:00	feat(core): add reserved domains and workspace
35c9962d8633599a8b2393b29e1fa26f02677272	2026-10-24T16:55:26+08:00	feat(core): admit device backing views
eb49fa0e945bec802db28472deb50ab94b715284	2026-10-24T16:55:26+08:00	feat(core): record plan topology and source programs
ec30569bc6b921bfc09af888978d22d23e957944	2026-10-24T16:55:26+08:00	refactor(cpu): share reference math lanes
9dd1ebfb18623e27cb5ef07940f5e6839f1ecaed	2026-10-24T16:55:26+08:00	feat(cpu): add original torch CPU operators
a3716fd0de505b5bcf1e7e43ff3da7154c1ac8ff	2026-10-24T16:55:27+08:00	feat(artifacts): select cached sources by locator
77c48b315feec501082c2609077033331cc9fcb7	2026-10-24T16:55:27+08:00	feat(artifacts): authenticate reusable source access
7ba9278b0a1ab97514f9511fc7a72ae90eeccf1a	2026-10-24T16:55:27+08:00	feat(cpu): evaluate reusable text source windows
dbd836351d40289eb8d99f7e1f648c139530de34	2026-10-24T16:55:27+08:00	feat(cpu): add haswell and zen bulk kernels
b026d0f9d5024d06f5b83768191f70258c1aa306	2026-10-24T16:55:28+08:00	feat(tools): add qwen27b output-tail reference
85d86b3c5863d9d78b6bed0b7db6e36ed28686cf	2026-10-24T16:55:28+08:00	feat(tools): add qwen27b attention-tail reference
0c7a4a9770f64d3de1e3c920a438c88e088668c3	2026-10-24T16:55:28+08:00	feat(tools): add qwen27b full-window reference
af5aa8498550fb98e69734c18769cd2c84299278	2026-10-24T16:55:29+08:00	feat(tools): add qwen27b layer9-prefix reference
6dd5970776b7b5839ebc6b60c3694049d31f406b	2026-10-24T16:55:29+08:00	feat(tools): add qwen27b initial-core reference
2c70ccb66c780874998e77bbdb2c9e33387bdf2f	2026-10-24T16:55:29+08:00	feat(tools): add gemma4 tokenizer reference
c870725898c9a0595609f53f7310182c5bec4fac	2026-10-24T16:55:30+08:00	feat(tools): add gemma4-e2b text reference
8b0cc32c951027123ffa2a9dde1e5f7dd6c2a57a	2026-10-24T16:55:30+08:00	feat(tools): add gemma4-e2b CPU matrix
283765cff8bdbb69d4d6674d7fd28360238ea693	2026-10-24T16:55:30+08:00	feat(tools): add portable BF16 slice reference
b5015e7a77e37172236c602b65a25c74a3caead4	2026-10-24T16:55:31+08:00	feat(gemma4): admit dense 31B text metadata
bebfeaf2d81722ebca4c11613e8a40d5834e18df	2026-10-24T16:55:31+08:00	feat(gemma4): add E2B text reference and tokenizer
3002089997fcee9e43abc3dc60404cbeba1f52ab	2026-10-24T16:55:32+08:00	feat(gemma4): add E2B learned-source diagnostics
225fc5e1a520637b45c3c76f36e4b58963c1de08	2026-10-24T16:55:32+08:00	feat(qwen35): record window, bulk and CPU tail sources
9af407bf3ff5763c20f4e9fddbf8061fa16f8179	2026-10-24T16:55:32+08:00	feat(qwen35): add qwen27b window replays
c4cbff17e6c431ee1d60e67cd725b3bf0f2fc4a7	2026-10-24T16:55:32+08:00	feat(cuda): observe device metadata
d9d4f8b9bb500dcf632b37efae50c2926152adf3	2026-10-24T16:55:33+08:00	feat(cuda): own compiler, source and cubin actor paths
4bc7c429d585d720600a9a33f0da25da4baceff4	2026-10-24T16:55:33+08:00	feat(hip): expose availability and prepared native sets
248bcec54c6e3d94992e379b509d84fd5864cbbc	2026-10-24T16:55:33+08:00	build(gpu): add qualification feature surface
68c2f5c050136978f5e7c01958fc08e6f2504d60	2026-10-24T16:55:33+08:00	feat(gpu): add shared kernel source primitives
abca70fdebda8e395b513133e79f938a9b2d84cd	2026-10-24T16:55:33+08:00	feat(gpu): add original source arithmetic kernels
63583681ae8c3f850f1b6e00dad4de8a7f421c56	2026-10-24T16:55:33+08:00	feat(gpu): add native policy and physical identity
fe95ecdde7800a6a9079f70ff7aa90f6780cad15	2026-10-24T16:55:33+08:00	feat(gpu): add vLLM input RMS qualification kernels
405e2462ff310c6ceae1e43ee54c54e2b13907c0	2026-10-24T16:55:34+08:00	feat(gpu): add vLLM GDN instruction kernels
71c59964149f280787359063f1c1e17eb52fef98	2026-10-24T16:55:34+08:00	feat(gpu): add Qwen2B GDN instruction assets
00d4bbabde0491f6f0be513f0223558bc813edd6	2026-10-24T16:55:34+08:00	feat(gpu): add .8B short-HF GDN call catalog
d031166b726a1ab07ef88ea8c1401e01df8285ee	2026-10-24T16:55:35+08:00	feat(gpu): add vLLM CUDA qualification scaffolding
8dbe361c1e53db00738f1e25ab11dca44b5f5840	2026-10-24T16:55:35+08:00	feat(gpu): add layer, attention and GDN cubin checks
13ae2ffb7d0510073fd36dfc137693a9de2f3bb8	2026-10-24T16:55:35+08:00	feat(gpu): add canonical47 model qualifications
bbafb1de8c769c0ad1d158ffc138b7db0144f319	2026-10-24T16:55:35+08:00	feat(gpu): add Qwen2B qualification leaves
f619f7eed8377b8c8bf9838a642b86d049881180	2026-10-24T16:55:35+08:00	feat(gpu): add short-HF and full-model checks
3e89f1919993081535b5fe2820da05ea9d840b82	2026-10-24T16:55:36+08:00	feat(gpu): wire resident leases, program and native sets
136a4a8692d9aefe19e1d7ef25eab4e2fc48d561	2026-10-24T16:55:37+08:00	feat(facade): expose CPU routing preparation
08b30b571e4b3557be78318b66365030c1bf5bb2	2026-10-24T16:55:37+08:00	feat(facade): add canonical47 original-HF loader
bc66207ce9c5a1ca96e609e975151541579c4390	2026-10-24T16:55:37+08:00	feat(facade): add short-HF serving source loader
e44a53f75320e21df9642f5c3fe711efe8fccc04	2026-10-24T16:55:37+08:00	test(facade): add short-HF physical suite
08fc5935e124f6af33c5010fc4309648539e1641	2026-10-24T16:55:37+08:00	test(facade): add short-HF cohort suite
a4a5b6322f343004d1ecdf13b264654da466424f	2026-10-24T16:55:37+08:00	feat(facade): add Gemma4 E2B CPU candidate
2058dcb1ea3f8053b6906561d0698df392c91731	2026-10-24T16:55:37+08:00	feat(facade): add qwen27b HIP teacher source
71f496b190dc2c7ced495f9d4c1f5c5da9495f33	2026-10-24T16:55:37+08:00	feat(facade): add Gemma4 HIP drivers
752642361cb4ad2f769b868f071a3307e3b01651	2026-10-24T16:55:37+08:00	feat(facade): add shared HIP lifecycle driver
2bd8167db2c1d8817d38e4851dd45f9c2c9aa6e4	2026-10-24T16:55:37+08:00	build(cli): gate finite HF serving and run features
b3c5a532d72a266fc3d66e13c5c1a0f7ad1bd084	2026-10-24T16:55:37+08:00	feat(cli): route run requests and add gemma4 candidate
45570bdb3d7f21f531b8cbc840633b8d2eb83e4c	2026-10-24T16:55:38+08:00	perf(serve): plan atomic source prefill waves
e6b07efdbb2d6ec56d979ff09b025c8071038f06	2026-10-24T16:55:38+08:00	feat(serve): route and serve canonical47 sources
9999adb1a66d646f76cd1152c32da51b22c9eded	2026-10-24T16:55:38+08:00	feat(serve): serve short-HF BF16 B4 cohorts
ef9cffd413dfe3a02d8ed54e4a2ff3a5e317e10b	2026-10-24T16:55:38+08:00	chore(scripts): extend gate for HF features
2225af34b82e0e60da0f50dd4c5ec76eb827ca48	2026-10-24T17:01:02+08:00	docs(agents): codify parallel execution cadence
abb6b3b3b233e6887e182acb7c95ed26718601e4	2026-10-24T17:01:02+08:00	docs(research): compare optimization mechanisms
4b44d313bedecc6b82ecd44e317d786db2e08cd5	2026-10-24T17:01:02+08:00	docs(policy): adopt reuse-first provenance rule
46996b66e39e832f05d0174419e9c8f69367128f	2026-10-24T17:01:02+08:00	docs(plans): design composition and batch state
6ef74b048e830f1075d4d5ca4296cc801b8b6871	2026-10-24T17:01:02+08:00	docs(cpu-gpu): record shared memory and parity
fb7ea9a9a77b95543544ae131fb44da26522e7a7	2026-10-24T17:01:02+08:00	docs(numerics): record F32 closure checkpoints
424c672768b388f1738187baffafba7973c4cd08	2026-10-24T17:01:02+08:00	docs(models): record source20 and Qwen audit
da28f851448b8cebdd8b5c132f1a7371a6b39380	2026-10-24T17:01:03+08:00	docs(acquisition): record Gemma E2B acquisition
0c966799d7830c6b0a3d6b354fcf5318d36add9d	2026-10-24T17:01:03+08:00	docs(cpu-gpu): record current checkpoints
c57648ac54ff02a5370c3a26e4dc0c1634cee7da	2026-10-24T17:01:03+08:00	docs(memory): adopt shared pools and paged state
b552e2198d66400c94fbb56d4dc5deeb7604115b	2026-10-24T17:01:03+08:00	docs(composition): record shared HIP checkpoints
6075a4386589ca5d6e0d857d14f3ef4c6a4879b7	2026-10-24T17:01:03+08:00	docs(serving): certify transport and add batching
3b9da530d922814e8e5dea75574c2d4fdbc11462	2026-10-24T17:01:03+08:00	docs(distributed): defer mechanism reconciliation
38e3d77eba85875f0a3b2131a74c0de42933690b	2026-10-24T17:01:03+08:00	docs(platform): add Vulkan parity and reuse gates
a765f221279c9f803f0dacbee314c6ad4126905f	2026-10-24T17:01:03+08:00	docs(cli): add device routing plan
309b17d81bfbe7878fc8734adee424cb7785f533	2026-10-24T17:01:03+08:00	docs(handoff): archive history and run notes
ac17d49c07b2cb0e4653db19dc8621c6f0591279	2026-10-24T17:01:03+08:00	docs(handoff): preserve paused continuation text
7115754b639aa8bfbaab3a9231b97458fe945df5	2026-10-24T17:01:04+08:00	docs(handoff): preserve draft patches
11a47c74ecc55d92749dfd581d90254f02defdd5	2026-10-24T17:01:04+08:00	docs(handoff): preserve takeover and audit reports
b5d814f8fcab784514b4c6d9bfb1a11a7fa635d2	2026-10-24T17:01:04+08:00	docs(handoff): preserve canonical sourcebooks
e11fb986091f8fef71abf188e15177242084e596	2026-10-24T17:01:04+08:00	docs(handoff): index the 20 Oct handoff packet
4bf36731cfc4f66e02aff2f63fc32b653f6e363c	2026-10-24T17:01:04+08:00	docs(plans): activate CLI, composition and compute
8f01635305af920eadba96a7d4f1d6b9e8f588cf	2026-10-25T10:00:19+08:00	feat(core): portable admission and thread-control roots
5d40cf507a2e8f5f159b664ddc83876d73745607	2026-10-25T10:00:23+08:00	feat(cli): shared pending queue and ingress credit
52f76a0e0dc3ce8e09476af1460193a03a551227	2026-10-25T10:00:27+08:00	fix(cli): short-HF tokenizer credit and HOST8
efd7a34220ce168b5a5c851226b400e07e95a095	2026-10-25T10:00:32+08:00	fix(gpu): CUDA fragment inventory and host bound
19ff0598dd79d12ade182d90a855ecf9c60b1f59	2026-10-25T10:00:38+08:00	feat(gpu): Vulkan native catalogue via cubecl-wgpu
29a98c1c7c086d6367308ad9468c427046008cc5	2026-10-25T10:00:44+08:00	feat(gpu): HIP source-page events and witness
e8db111158991339f76e5b06ed3acb8b370f302a	2026-10-25T10:00:48+08:00	feat(examples): Qwen27B HIP matrix, natural tools
332038ea78ec8b1e69a2a29d3ca09370d712ef6c	2026-10-25T10:00:52+08:00	feat(examples): Gemma4 HIP final-owner and matrix
68e13a8114e130ab89ac4c55ee9ad14172faa42f	2026-10-25T10:00:52+08:00	chore(cargo): lock assembled five-lane graph
74d0c71998709de48115db16507ea0109b23f65b	2026-10-25T10:02:44+08:00	docs(plans): record assembly, C16 and handoff
1269b8eae908a40e79d747e923e29b708d84ba12	2026-10-25T10:04:00+08:00	docs(plans): retire five landed lane trees
06b89da455655ef039be2a87fd1e29917ce4d2c3	2026-10-25T10:28:19+08:00	fix(gpu): HIP probe witness page id typing
c217c35d480eb33b04eb981496d552a816b169e9	2026-10-25T10:28:24+08:00	fix(qualification): V state base parents, attempt-2 bases
3941a3c9fe0ee5dec0b9f836a86e992161f5f8ba	2026-10-25T11:02:16+08:00	fix(qualification): Gemma resident report-phase envelope 4GiB
e6ab074e3118a7200ea0a620a92b3a0990bbcac5	2026-10-25T11:33:24+08:00	fix(qualification): H/G source census skips build dirs
1565683442e80f529d810d98c63ca2583c5b01d7	2026-10-25T11:51:52+08:00	docs(plans): record queue GPU acceptance results
2304f8b4d2079c5312a8020f14af49c2c3832ab3	2026-10-25T12:03:16+08:00	fix(cli): precedence, ambiguity, pre-client hold
d3d21a35c5b75272d82947ccd9f351bc7888e3c3	2026-10-25T12:07:48+08:00	feat(gpu): B4 grouped transport and certification
6f1e0947cdf7673cbe4626b983fe894fbad1744d	2026-10-25T12:40:33+08:00	docs(plans): record main checkpoint and B4 merge
9fbe15e9e45c0d9c71ce00e73dfbdde343a77513	2026-10-25T13:08:47+08:00	docs(handoff): record checkpoint, HGV pass 3, successor queue
5749b732dc755c24d2179ba7dd7e8abc9380805e	2026-10-25T13:28:53+08:00	docs(handoff): record Qwen27B capture lane yield detail
488e0ea2c21e2301c2fdeaf82dab6de32789ea08	2026-10-25T08:14:56Z	fix: repair native qualification contracts and GPU feature isolation
93b611acaa657b6999a19239e4596d4de47b4de3	2026-10-25T08:26:58Z	fix: prevent Gemma HIP source-control payload shadowing
6edc6e2c99d9a2353d16a85ac689e76af3628be5	2026-10-25T08:33:50Z	docs: pin refreshed H G packages and completed GPU-free controls
9e67b74000bc085a844d8844d954d82f8e196493	2026-10-25T08:59:57Z	docs(audit): independently validate post-4bf GPU boundaries and resume holds
03f2e662f219797da524d0e4d6e2aabef979471b	2026-10-25T11:08:08Z	fix(gpu): certify grouped B4 CUDA ABI and exact fixture retirement
3d9866dfd6cf66009c9aa6efe0967e4a134e4e41	2026-10-25T11:09:42Z	docs(audit): distinguish 23 model-dependent B4 scopes
a39ca967ea2029400649284285ca18620544152c	2026-10-25T12:34:50Z	fix(gpu): qualify grouped transport from independent fresh state and seal KV cache views
7362699b7023b30c815482a71900af40b0a0688f	2026-10-25T12:35:00Z	fix(serve): preserve prepublication cancellations and bound decode context
5229e53d7b46c0fd322217b216b2e11a8b94a433	2026-10-25T12:37:00Z	docs(review): record architectural and correctness findings and qualification gates
83a31747679385cd5951487e6bcdfb9a2f2d1f24	2026-10-25T13:51:15Z	refactor(gpu): one typed closed CUDA ABI for metadata and bound handles
095bf8a4b97dbd7a0561c7ea4316ef7ff8167bdd	2026-10-25T13:53:06Z	refactor(core): collocate Qwen3.5 recipes without changing type identity
2a47d25181f1a8d91a9489d9f2d621d3807bb329	2026-10-25T13:53:16Z	feat(cli): concise run placement aliases and observed-device inventory
d9c5872c73ee4d7b2bbbe1079446ac7634f8d919	2026-10-25T13:53:41Z	docs: scope ABI family and run UX refactor acceptance
5399ab93929eab8e6448078ae8126144b147b238	2026-10-25T14:31:01Z	refactor(cli): use device selection without redundant on alias
d84853337bbd2830f50a9c4e704d266990fd2ce4	2026-10-25T14:59:13Z	feat(cli): resolve Linux GPU device nodes to verified physical PCI identities
d0fff261904fcb7836c0f71d0682028a12fd53b4	2026-10-25T16:26:13Z	docs(handoff): preserve complete ZeroGravity history, open work and takeover prompt
a857d45ddea827457fb59823ea6597f3eca03d22	2026-10-26T02:55:15+08:00	test(gpu): repair B4 qualification fixtures
6610b6195505d7d8fb17640660461310fa1c3d41	2026-10-26T07:57:31+08:00	docs: record B4 and C16 outcomes
6d855599531e8f67ec6cd1e1e00f7e1b186ff7f3	2026-10-26T08:47:53+08:00	test: add shared model correctness checks
195099ca583ace766d075682927f590360c823a5	2026-10-26T08:48:41+08:00	docs: set all-model correctness policy
405a3e6358c85917506f76f780094de89eaec01b	2026-10-26T13:39:06+08:00	refactor(formats): group codecs and format homes
d9a8ecf6187216c8b909c60e44927b06f256afdc	2026-10-26T13:40:43+08:00	refactor(correctness): name input geometry
fa844a663ff9ab4f701146be61bf2353cf580385	2026-10-26T13:41:20+08:00	docs: define module ownership and naming
a5ed7de9e0f0682d86aca09d0ec73336a5f2ecf1	2026-10-26T15:07:57+08:00	refactor(models): contain source plan ownership
c90b8cd7e2fc8b05784d1e27c907347ea4319f97	2026-10-26T15:30:04+08:00	refactor(gpu): separate providers and bindings
41e93370b7b3b9edc45cd9575eaa6b9d6845bed2	2026-10-26T15:41:21+08:00	fix(gpu): preserve plan and capture guards
fc734571e6e342b376f406aac7fc2ef4e42e69f2	2026-10-26T15:53:31+08:00	refactor(app): compose model execution routes
4a0f4552354ad8f5931b7cb45bd85d34a0026eeb	2026-10-26T16:01:43+08:00	docs: record completed source checkpoint
5c2fe532c68e98a0c750b5ff27969ae965ec3f9d	2026-10-26T17:13:15+08:00	test(cuda): restore source admission checks
9c2c51ab291580a5c47d9528f8ff2ed87bd8228a	2026-10-26T17:15:22+08:00	docs: record side-worktree retirement
fe1762568c96eead3893f338d0181b8634a98303	2026-10-26T18:26:57+08:00	refactor(cpu): own shared GDN convolution
263ff791d04eb50fd1aeca85249b0f7aed3cb1ad	2026-10-26T18:27:45+08:00	refactor(gpu): remove model dependency
e7fc3150927104b2f22dd020f63efb8817a75d82	2026-10-26T18:31:38+08:00	docs: explain benchmark container termination
50abaf71121a2ffbb6a649f16aafe185650c1388	2026-10-26T19:20:42+08:00	fix(serve): join workers on process shutdown
c84b98744955c43f7a6a992289c7d8a977a0ee9e	2026-10-26T19:21:14+08:00	fix(bench): retain owned stop evidence
4f427dd814f734afdad798b053141946c998347a	2026-10-26T19:28:49+08:00	docs: record serving shutdown validation
cd4159d26048b0395c40235b320b7d48df6ce53d	2026-10-26T21:58:07+08:00	fix(lifecycle): retain owned command streams
5a18b5bbe6a80976b9255813070c09d2cae61a3c	2026-10-26T21:58:36+08:00	refactor(cuda): derive capture scope ownership
61f095cf760ee6b8d1f015b71b13ac46835a4917	2026-10-26T21:59:12+08:00	feat(reference): capture finite Qwen prompts
3612ecb7077d18685e926c698b812e4e2e60375f	2026-10-26T23:14:16+08:00	docs: define production and test boundaries
1f08e5525b5ede7124054156083e5a4ff67d151c	2026-10-26T23:23:45+08:00	refactor: separate rotary and test preparation
d74f59a18c6345dbea963861acd519c1282a26fa	2026-10-26T23:24:36+08:00	feat(reference): emit captured CUDA recipes
5f4051ecbfeaa1350055db47fcd281fbeb81843b	2026-10-27T00:05:41+08:00	docs: preserve comprehensive agent handoff
a1c51f9d478ec8025c8da09e58a8a2626c4cfe32	2026-10-27T00:34:05+08:00	refactor: separate fixed rotary preparation
91a29586a292a7b8204e85cf4ed21cf03968b718	2026-10-27T00:56:52+08:00	fix(cuda): seal rotary production routes
9ad9aac42e0ea63ab78e8ce9a7836cc1d770ab62	2026-10-27T00:36:42+08:00	parameterize CUDA GDN prefill convolution
05baab5abb2e02702ab565d267b44474e9362e66	2026-10-27T00:37:11+08:00	fix: separate prompt semantics from CUDA qualification
d087012fa5f4d604069a015a7773defed5ae185c	2026-10-27T01:31:53+08:00	fix(cuda): correct parameterized Conv identity
0c63d89a54ee8a45a870cc383cef144acfb09579	2026-10-27T02:46:41+08:00	fix(qwen35-cuda): isolate finite qualification routes
ae936a8ab3102ebac682c947a63dbf105d91418b	2026-10-27T04:32:55+08:00	fix(qwen35-cuda): enforce qualification boundaries
7a910d9aab0f7f09f788ff983c678ec3fad25413	2026-10-27T05:04:07+08:00	fix(qwen35-cuda): make route selection declarative
ca0d2bb0f4323f7584a764e0e162219c47561b52	2026-10-27T05:05:50+08:00	test(qwen35-cuda): cover source-kind neutrality
f61a3ca618a6b1463a185231a86f879f13019f21	2026-10-27T05:32:39+08:00	fix(qwen35-cuda): restrict finite route targets
85b272476e2b61d65d0d5236abad3294b29ca117	2026-10-27T06:10:23+08:00	feat(cuda): parameterize GDN prefill sequence setup
1fe615113e1c859d665957632166bea96095f5fa	2026-10-27T06:40:08+08:00	cuda: parameterize remaining GDN stages
2e3bbeccf97b360ea3a5ab228623996e58228b77	2026-10-27T07:12:15+08:00	feat(cuda-gdn): configure remaining prefill stages
1b898ffdb9ac333b535233ca49c1f9da1c290c4c	2026-10-27T08:14:15+08:00	fix(cuda-gdn): seal explicit implementation selection
c0bdbb183e6d9112ee2c9d8cc1c5281cc35c30f4	2026-10-27T09:13:23+08:00	refactor(cuda-gdn): remove recipe execution bridge
eb38dd5e892f2120ac29cc2fb79b2dbe137a37fe	2026-10-27T12:25:58+08:00	feat(cuda-gdn): add fresh-48 output variant
ba7fb0389978d82e66d1832f50feb32238717eef	2026-10-27T13:50:22+08:00	feat(cuda): add qualification fresh-prefill capture
fadf2b1123309a2118725f2cc005d3edc7ea39f2	2026-10-27T14:35:30+08:00	fix(cuda): use fresh prefill reference layout
ae9a84394d3aa08d77e0edfe4285dc09b7862040	2026-10-27T15:30:30+08:00	fix(cuda): decouple GDN parameters from text kernels
532a35c437d4b941ea9931d9c1e3ca16d9b7d080	2026-10-27T15:46:05+08:00	fix(cuda): use fresh reference prompts
54d7ea7a3012b70f3b92ec8c45800a913e6dc811	2026-10-27T16:00:06+08:00	fix(cuda): admit fresh .8B row extents
e99c76e3e644b612a173a029230d44dd4e781fc2	2026-10-27T16:07:02+08:00	fix(cuda): account fixture host baseline
ad82feb5222d81a5f19fe1311744387b8b9959bf	2026-10-27T17:51:44+08:00	feat(cuda): add call-free Qwen plans
f7ff4f96be5110a9a064e9c78643897212a65537	2026-10-27T18:38:40+08:00	fix(cuda): retain ordinary plan source owner
1921db16695d8b9166399cd10eff793a0ab8e9cb	2026-10-27T17:30:30+08:00	feat(bench): expand bounded comparison matrices
1d278edeaa875aa8d85936566516388943d9227e	2026-10-27T17:47:51+08:00	fix(bench): retain complete matrix cell results
fe228abda89c1388efe9524439aa8d04c4d37c07	2026-10-27T18:36:55+08:00	feat(cuda-gdn): add ordinary chunk runtime API
52bfb519a245b114de90e6e2d5cb1de8a80497ca	2026-10-27T19:34:25+08:00	fix(cuda-gdn): complete ordinary runtime schedule
901d23eb518c4b9167707f8f716aebb50f3cac27	2026-10-28T02:27:44+08:00	docs(handoff): record 27 Oct checkpoint
ea66b3ab3096d08bafe0f28cfa9a5f0c12307e45	2026-10-27T17:55:20+08:00	Add dynamic CUDA ordinary text primitives
68393e5a644d661bf5e101d47cebd2290c1faff2	2026-10-27T18:58:38+08:00	fix(cuda): constrain ordinary text candidate
6b4a2ec7a8f81fcf1995f6aa3063dc2c152a7a5b	2026-10-27T20:38:45+08:00	fix(cuda): close ordinary Qwen review blockers
441fa64f52be6d3d8636e6dc8241d241e6b55889	2026-10-27T19:18:09Z	fix(cuda): close ordinary primitive review gaps
ee67aa848f2603bb342029c811692361ec10b1e0	2026-10-28T01:06:15Z	docs: close ordinary CUDA primitive review
c2494ac1665508f4eabf9aab08885db5adbfec62	2026-10-28T01:14:59Z	core: bind ordinary source invocations
301e912f4046a6ed5d4e0e41c275ae2423a6dfb2	2026-10-28T01:37:46Z	gpu: generalize owned BF16 GEMM geometry
dcfbd74f7ec060e640e734cb3765b0b585226bd0	2026-10-28T01:40:19Z	gpu: admit verified BF16 matrix upload on CUDA
b116ccdbaba8ff4fa4657bd93e017cbb9d80a9e4	2026-10-28T01:43:13Z	gpu: expose owned cuBLAS to ordinary CUDA
b4a926303ccffbea6c21bf1fac5b42a313ff8021	2026-10-28T02:45:47Z	gpu: add ordinary CUDA projection owner
40c8721c2c1a257371526e6aad9cabf18a7c652a	2026-10-28T02:53:04Z	gpu: bind ordinary primitives to owner status
71cd73779a39cc74ba558efa52c7b4dc5055a13a	2026-10-28T03:00:47Z	gpu: add source-exact ordinary CUDA attention
e5f148e04c4ced0b0fda61efadace115b11c6017	2026-10-28T03:14:12Z	gpu: reuse ordinary CUDA source projections
18579ffc25818c99f9e8d884daf9fadc5a0c4dac	2026-10-28T03:14:12Z	gpu: make resident state precision explicit
9abd1d973d1097681d859d65b1cc0c50d4ed850f	2026-10-28T03:15:33Z	gpu: validate typed resident state layouts
4fde7574e75274a8971d82d0a07f1b0d54c8dfa1	2026-10-28T03:22:41Z	gpu: share canonical GDN state transport
cba4f5e2ceb689818892bee641ad55d8f4398c94	2026-10-28T03:25:29Z	gpu: partition ordinary GDN runtime schedule
8e964cef1d1df61f68ff540131476441e8c72694	2026-10-28T03:28:50Z	gpu: expose neutral ordinary GDN primitives
8d5035dc051c86107c35ba4da4dad13284bdbc41	2026-10-28T03:53:51Z	gpu: add ordinary CUDA parameter transport
57379b3452c15d4cbd1782fabe1244c54e8333d9	2026-10-28T04:02:12Z	gpu: derive ordinary CUDA resource layout
685e9be69162c0480302ca6daf609f2926714276	2026-10-28T04:04:13Z	gpu: reuse ordinary CUDA matrix packs
6f5d43d2bb9b92723df5c0deff4260a41e74fcc3	2026-10-28T04:08:42Z	gpu: prepare ordinary CUDA cold source owner
11aaf2bb19ceb11bed752945fb5a9973d4372834	2026-10-28T04:12:11Z	gpu: bind ordinary CUDA resident body
81318adeb432ec1fda08ef8fa930478253b01ddd	2026-10-28T04:12:33Z	gpu: expose ordinary CUDA cold values to body
00ee779cb096ee567be920991d4e7d3f3b4c5ee5	2026-10-28T04:16:42Z	gpu: preserve ordinary residual and GDN scratch ownership
3865ffd1ee31b530ab74e3e09184b1949c932ad6	2026-10-28T04:29:42Z	gpu: bound ordinary attention to active KV
c7b1962b60ceb2f00ff204954c0a78f0aed17559	2026-10-28T04:34:06Z	gpu: retain ordinary GDN local states
a8eef412f2b27020dbdc559ce2b6b074bccc7445	2026-10-28T04:36:01Z	gpu: add ordinary CUDA active view transport
d0f557a47e2f1359a08c30822c5b3bb3fce5942e	2026-10-28T04:44:15Z	gpu: execute ordinary CUDA through resident owner
9edece37532ed6eaaa09153885d32c3faad4a73a	2026-10-28T04:48:26Z	gpu: separate ordinary CUDA owned and execution accesses
be5362e5f26ed4e74db75341892bad9e0f5f2907	2026-10-28T05:00:42Z	gpu: load ordinary CUDA resident model
1f69da6fe186a8959a2f8bc20616ae8838f761ae	2026-10-28T05:04:45Z	gpu: execute ordinary CUDA as one resident chunk
b379a76963f60893f5197ac3d882f58778de27c5	2026-10-28T05:13:14Z	qwen: add ordinary CUDA qualification adapter
8c2459929a81256760e337283eb25b73e152b8ae	2026-10-28T05:47:28Z	gpu: close ordinary qualification build edges
c680b6c5b6902a87851a3073e7360586e2b09fa5	2026-10-28T05:54:59Z	gpu: add ordinary CUDA explicit shutdown
eeb38bc60f33741d8d3afd17b92648fb2a201a2c	2026-10-28T05:57:51Z	docs: mark ordinary CUDA qualification-only
5e29118c2f781ea8dba5b4d85fe04c570284c988	2026-10-28T06:03:39Z	gpu: qualify ordinary CUDA context admission
ec44357c57704420731d2a74f3ad888cb45cfb6e	2026-10-28T06:07:58Z	gpu: add ordinary CUDA request cancellation
f2603b3d90dcb70dbd668b5b2d23beb07814f140	2026-10-28T06:16:15Z	gpu: canonicalize ordinary KV root snapshots
e93e7b42dda9bc9ed2f6ac92c34ae7f369428220	2026-10-28T07:03:53Z	qwen: add serial ordinary CUDA parity harness
014b77385211a3431ecc90f4e2eda86659917cfd	2026-10-28T07:07:33Z	qwen: add runnable ordinary parity qualification
ceae923844d08dbc4741cdcbc2789034d51f0f64	2026-10-28T07:12:39Z	gpu: harden ordinary BLAS retirement boundary
9e566438d6a428ffb7aa363fcc05029fe1a9c59d	2026-10-28T07:15:28Z	gpu: harden ordinary BLAS activation failures
487a796a52b65e1170a961ba9a1e23eb5f426844	2026-10-28T07:39:21Z	qwen: isolate ordinary parity across processes
ee70ec6aa01e1da846ea385afcd64f7acf280cd4	2026-10-28T08:25:59Z	gpu: use source-exact ordinary rotary
78426672edaccbb669254b87a8e47f759586049b	2026-10-28T08:26:14Z	docs: record ordinary rotary qualification
ea4b67b3fc8ce8ed56f0c8f367ea7b0575881cc3	2026-10-28T08:32:39Z	gpu: refuse legacy ordinary RoPE contract
62ec2cb6fdcb8488b2e0b385ff995d903dcc7c89	2026-10-28T08:45:56Z	gpu: isolate ordinary CUDA feature boundaries
13703a4552919f795dc6b747ef2e9547c174a275	2026-10-28T09:23:09Z	gpu: add ordinary owned BLAS qualification
9ae5e7ebee018695361ee31246df44d23a676d8e	2026-10-28T09:23:10Z	docs: record ordinary provider qualification blocker
ccebfb05c7d5efb4127051059de7fe50eed227ed	2026-10-28T09:40:14Z	gpu: qualify ordinary primitive numerics
f6edcbb9a878f9d72ea6e1eeb5fbf2893a07fdc5	2026-10-28T09:40:14Z	docs: record ordinary primitive parity
642e24529111ed966a35e4825dee7bac501f3663	2026-10-28T09:45:58Z	gpu: qualify ordinary residual RMS
ff512af4a3a28b71c8861501528d04b5f61f1c45	2026-10-28T09:45:58Z	docs: record ordinary residual RMS parity
77dcafa4934a02457bf09701e1116ea8a9494181	2026-10-28T10:16:27Z	gpu: qualify ordinary multiply and KV append
107e301fe437180b083d8c889345ab956662da65	2026-10-28T10:16:27Z	docs: record ordinary multiply and KV qualification
04fad2a20f9c5992c4ee723859c4dddea743b113	2026-10-28T10:19:14Z	gpu: qualify ordinary embedding
4274b192e0793b838956f8ef7c4b2b6ccf7009d4	2026-10-28T10:19:14Z	docs: record ordinary embedding qualification
f17540b053fe0fa94fa99243a6f3b6bd65b7cb20	2026-10-28T10:24:38Z	gpu: qualify ordinary attention wrapper
7c67dd8f2c6caf9422c36d7644554a2c345065fe	2026-10-28T10:24:39Z	docs: record ordinary attention wrapper qualification
3eb3f74150066c56d38b9f18ffa4aa492dd9c157	2026-10-28T10:28:16Z	gpu: qualify ordinary canonical state transport
a6743c9f1fe7638340639065b5e27fa1bdfe6d9c	2026-10-28T10:28:16Z	docs: record ordinary state transport qualification
8f4450c06aee8979f18cc68510152b5bef981366	2026-10-28T10:38:32Z	gpu: fix cached GDN gated launch grid
8dc7cdffabbae94b2c8a4d68dea28aa66cd2f975	2026-10-28T10:38:32Z	docs: record generic ordinary GDN qualification
ee5dfe10966ae7ee7be47b1904e35657fb07db66	2026-10-28T11:17:10Z	scripts: add serial ordinary cached47 parity runner
c4f54ebd0a92d8fa21a420d63c56b1f16d3173ae	2026-10-28T11:17:10Z	docs: record serial ordinary parity runbook
b5c353ceef5c44d92511396933a1494b8faf0bb5	2026-10-28T11:22:31Z	qwen: harden ordinary parity restart
d03bb3a94c431b9a50a7767a9da3cfe177bebc7c	2026-10-28T11:22:31Z	docs: record ordinary parity restart semantics
3ed372de521553bc30d6e12159cf368dfcf198b2	2026-10-28T11:38:17Z	docs: add comprehensive 28 Oct handoff
563ad344a0e56ac6500afa19096d5acdcc93cb44	2026-10-28T13:40:57Z	feat(cache): add human by-name snapshot aliases
be1e81a69482b9838ba6f2f943fc6fd001dc3d85	2026-10-28T14:03:47Z	refactor(cache): group by-name aliases by model
35a025c6562a36342552f62245a3c75cca8b66c4	2026-10-28T14:45:17Z	fix(cache): stabilize human snapshot aliases
76d1561824e98139dd8ce07e352bee4ee1fcd11a	2026-10-28T16:17:15Z	fix(cache): make human aliases crash-consistent
8bdd54b0f13aed240efac150afc9bb3a8c0747d2	2026-10-28T18:56:39Z	fix(qwen35-cuda): authenticate parity evidence
f6209d923a5f1e3d0cf967a93d887528b5cf166f	2026-10-28T19:54:19Z	refactor(qwen35-cuda): simplify parity evidence
768b724a45921e3fe2c3fcf305c3fdac2d67817b	2026-10-29T01:58:37Z	test(serve): align qwen cuda qualification boundary
932f56499a24ce2a8f38ff5ad203fcb2716c831e	2026-10-29T01:58:42Z	refactor(cuda): move gdn recipes to capture tooling
734d29f52ed5e83abc069fdd7ca3902cb6803e47	2026-10-29T01:58:48Z	refactor(qwen35): retire obsolete qwen27 replay harnesses
2ff3d49e70a1b80fe708c9e4ae0d8ff741990548	2026-10-29T01:59:07Z	docs: add fabric vnext transition design
15f5bd3ae26891f67380edeb1a52f4afb2cae16b	2026-10-29T01:59:14Z	docs: record retired worktree and run cleanup
9abd9365d1d9334f6cbaf8a348278e5d977384dc	2026-10-29T02:10:35Z	test: retire machine-local capture dependencies
ace934b4a402278392e30ea63e2cfade0f5208df	2026-10-29T02:12:38Z	docs: prune superseded handoff archives
7b5e50e926b0724b7ad0e630afd58c6a0075644c	2026-10-29T02:15:10Z	docs: normalize current planning provenance
379d91d42aa247b0548b02211505c3aa9743658e	2026-10-29T02:51:16Z	test(serve): make pid1 shutdown control capability-aware
ae8cd89d29f1b143f0fa8a7ecffacc9a125e5128	2026-10-29T02:51:16Z	test: cover qwen27 layer9 reference contract
ff89b87cd3d288e6dc83a2e4f423e19045142d68	2026-10-29T03:00:59Z	docs: add canonical current handoff
11664d02f740b60aad2d9bdee69f9cfd3dc42593	2026-10-29T23:54:30+08:00	test(cache): make private fixtures umask-safe
60907d12861bc3338d5e20d2eef749510aa8cdb6	2026-10-29T23:55:02+08:00	fix(core): report residency refusal extents
97c52ec8ff7530e7f7cc44adc5ea19a5e3df4a05	2026-10-29T23:56:18+08:00	fix(cuda): restore reference capture profile
440768d58869bf672943c64b055cd56cf2091287	2026-10-30T00:36:51+08:00	docs(cuda): record provider boundary findings
2983bb72c5832c6efe5d9117281afa33a31517a2	2026-10-30T21:23:06+08:00	feat(cuda): own BF16 Qwen execution
b262d2e9e5d0b29672c2438e72f72b6754bb9e90	2026-10-30T21:33:20+08:00	docs(cuda): record execution qualification
99f0232d066c8cef7d8302adb1015ad7dce05643	2026-10-31T05:10:36+08:00	fix(gpu): preserve declared parameter rank
825ade8aca3940a838d372823696743683beb88c	2026-10-31T05:20:53+08:00	fix(hip): preserve natural diagnostic ownership
8ca41db024886018afdf0975e34649d4f5a7f082	2026-10-31T05:24:29+08:00	docs: reconcile Qwen qualification campaign
d4d5df4cf771c2981a667de92f526771064359f3	2026-10-31T07:32:12+08:00	fix(cpu): preserve original attention PV order
f6fa4b993687f8d8c5c93566602182fcb8d5bdf9	2026-10-31T07:33:34+08:00	refactor(qwen): share source execution
87ba2e2edd6b6d55c6d24adce89942e5e20f9f87	2026-10-31T07:35:06+08:00	docs: record source attention qualification
8a1c6b1456a6921f1fbec950ad58fc6d54bfd36f	2026-10-31T11:39:56+08:00	fix(gdn): retain initial empty-state addition
1f29a42c656be1540db8996df5f198e174eb2bc5	2026-10-31T11:42:37+08:00	feat(hip): run source-bound Qwen models
85f33c77cbfe5babaa21e008dfec13070d2a80c0	2026-10-31T13:25:24+08:00	fix(hip): retain canonical source ownership
55536ce31e57049c2df378e085969ff141f16578	2026-10-31T15:42:51+08:00	feat(cpu): implement source chunk GDN
e2559b6063b42ac65a331eab2652a079f82258c5	2026-10-31T17:36:26+08:00	feat(gpu): implement source chunk GDN
1c35172b3c22f027c6f2b72fa0d177fe205797a1	2026-10-31T17:39:13+08:00	docs: record verified checkpoint and pause
7396984887e1acd1a1ebfa9b61e09fd100d2ec63	2026-10-31T18:36:28+08:00	feat(fabric): add compilable base skeleton
c853ebb6594868a332462a6343a55a671c3ef797	2026-10-31T18:39:50+08:00	docs(fabric): reconcile plans and recovery
7fc5d17316d16a8edea0df9008e60f91e62aa81e	2026-11-01T12:01:52+08:00	refactor(fabric): unify source and state owners
8591b5221a0e09e22b146b22a9041ab7cc2a716b	2026-11-01T12:02:49+08:00	feat(fabric): wire CPU runtime and serving
27ac27892abe2abb28a08e2303d4db39d3f9f95b	2026-11-01T12:04:59+08:00	docs(fabric): record working key-path checkpoint
d78eb5537e8002823e17191b6ad5bb4c6f91319e	2026-11-01T13:12:18+08:00	fix(engine): reject runtime self-shutdown
ae7ec111114a678d4474e575b3da064cfe72190a	2026-11-01T16:15:56+08:00	refactor(engine)!: share CPU/GPU scheduling
72622ce6ae85db9c181434c55afbdc1e2d04faa5	2026-11-01T16:20:30+08:00	docs(fabric): reconcile scheduler checkpoint
d4e94b1d436b586aaab84cb4d21c795d16bb029c	2026-11-01T20:02:38+08:00	refactor(sdk): own run and artifact orchestration
bab4390c59bf07420270080bff7cb9f898f25d01	2026-11-01T20:20:59+08:00	docs(fabric): reconcile SDK ownership checkpoint
83060e5dd99d6a7aadd92befb9ea271dfd1891a1	2026-11-01T21:33:23+08:00	refactor(core)!: bind numerical/layout contracts
2f192240784d532a673463e6f53a50517e577692	2026-11-01T21:34:07+08:00	refactor(sdk)!: unify catalog and serve loading
51cd294d7c9616cf0e26ef6f9514aa0b44b8ae70	2026-11-01T21:36:41+08:00	docs(fabric): reconcile contract checkpoints
860ce389b5de888604a3edbc56d07dc70b21dfcb	2026-11-01T23:11:03+08:00	feat(engine)!: bind preparation to source owners
d0e03f5f77ea8ce6934fe7cfc7579b8eca7a41ec	2026-11-01T23:15:16+08:00	docs(fabric): reconcile preparation checkpoint
fef7123394d4d200c677edd077d38e7c39462749	2026-11-02T09:42:43+08:00	refactor(sdk)!: own run configuration policy
1f40880546dd1d7ce4b54250124055bee8482742	2026-11-02T09:45:16+08:00	docs(fabric): record source parity checkpoint
8d761c45e07cefbdcafc7d95f551efac83d21ec0	2026-11-03T02:36:41+08:00	fix(gpu): bind full greedy feedback packets
8c43844e0d214c7b9d3f74b772089973ed40c7e3	2026-11-03T02:47:09+08:00	test(gpu): qualify cold Gemma SDK lifecycle
9defb6c6e46b7579f6406305201c3f4c54107989	2026-11-03T03:43:03+08:00	docs(fabric): close scoped transition acceptance
1c999298dde1e4f7fd8d0812cd401681914aabeb	2026-11-03T03:52:24+08:00	docs(fabric): count supervised cases accurately
bb400c42136a798e177be82d3a1e6ca40b3b8431	2026-11-03T18:49:17+08:00	feat(hip): run ordinary Qwen source chunks
8ffb5113900a79c0b99d41bc5999d9892b1c4604	2026-11-05T09:49:52+08:00	fix(hip): qualify populated Qwen serving
b9d8355a67914f65577cd5b9d92928131aafda37	2026-11-07T00:26:35+08:00	perf(artifacts): reuse verified source chunks
a57590e2351beefc637f68129558d277fd8a0aa8	2026-11-07T00:29:41+08:00	fix(cuda): preserve output reduction boundary
4f1e8275f8270387b237223cf269be2cdf47ac79	2026-11-08T08:13:06+08:00	feat(gpu): compose shared source execution
d16f0968655c0775c327025bc9388a8a2b312b23	2026-11-08T10:09:34+08:00	feat(hip): admit explicit total device budgets
b364cbeabdde7f49647eefcd93467d38b1207969	2026-11-08T12:10:15+08:00	docs: reconcile checkpoint handoff
9d73d467e702ff31a66460fe345ff8b7de6575a2	2026-11-09T17:19:28Z	perf(cuda): keep greedy feedback on device
9f45471a3cb7d4c8bfce4a87f36a54ea64c90264	2026-11-09T17:45:26Z	perf(cuda): specialize decode gemv reductions
5836cb7da52abd55fdd65336b2a9a339d73a545b	2026-11-09T17:46:15Z	docs: record repeated CUDA C1 feedback
55e195c640bbea14cc251139fa763fb063319974	2026-11-10T00:41:12Z	perf(cuda): batch ordinary source decode
eea45fad75e7938dd6d88151b6635c25a61e3a0b	2026-11-10T02:06:43Z	perf(cuda): batch cached GDN convolution
98976bebca69cbbaceef3bc0ac28a798462f795a	2026-11-10T02:17:41Z	perf(cuda): batch cached GDN recurrent
03267124e9fa02bfd8fc39364f5fbc1e6452928d	2026-11-10T02:25:53Z	perf(cuda): batch cached full attention
f50645e7c39f701bf915a5f41f12d45f288b4ea1	2026-11-10T02:58:31Z	perf(cuda): certify BF16 GEMM stores
1e4d76ee6d7f1d43453936feb095548658b4d89e	2026-11-10T03:49:21Z	perf(cuda): reuse output-head weights across B4
f28d280a4efeffd701d01f3bb852201a65d3ef75	2026-11-10T05:37:33Z	perf(cuda): batch aligned full-prompt prefill
ccf441d46a64fd407217e33cbf5360046d2402d9	2026-11-10T09:25:38Z	perf(cuda): widen native source cohorts to B8
67586ebf9244c873d26c4fa49a0e8fab4aba9f92	2026-11-11T00:01:10Z	perf(cuda): widen large-K ordered prefill tiles
77ffc23826c2cf393f8341fbeb360f359b03218f	2026-11-11T00:11:13Z	perf(cuda): widen ordered B8 decode K tiles
74f642237341f61df9e62f57b5a85da7eb05c02b	2026-11-11T07:31:21Z	test(cuda): qualify populated small contexts
5a75a704b9ccb3854956e59a9ab2022810cc4e13	2026-11-11T08:57:06Z	perf(cuda): widen rounded B8 K tiles
597da3f25021a711c0a30ab1fc3a54e12fa075e1	2026-11-11T09:14:15Z	test(cuda): measure large cross-profile state accuracy
b95acdc8568efc317ab0875d231ab209fde1544f	2026-11-11T13:58:31Z	perf(cuda): regroup partial native cohorts
2b8e90fff0df7c3f7ef96971ebaa8754a111ae85	2026-11-11T16:15:46Z	perf(cuda): widen interleaved B8 K tiles
fa703fa9af2ea86cd6af5ef0220359bc6b55cf13	2026-11-11T16:49:30Z	perf(cuda): batch chunked ragged prefill
0d9a56da48b5330d5d19394f1b44bd5b7c881aaa	2026-11-11T16:54:21Z	fix(engine): require pairwise cohort compatibility
6a4e2c9312a7943ddba6b6a270225811b205b337	2026-11-11T16:58:18Z	docs(cuda): record chunked prefill boundaries
d927fda0d7c23fc5e2fdcc4149528f774f97ab2d	2026-11-11T17:04:56Z	docs(cuda): record queued B8 scaling
96a7ac74152e6e619faabb17aa2d42bf9a812146	2026-11-11T17:11:37Z	fix(engine): require mutual cohort compatibility
8f8b4648d64f011b75af4c7e379e9ead1dd4e780	2026-11-11T18:01:12Z	docs(cuda): record grouped transport rejection
e22c442e8f8aab5f6303a4af261ef19652db4b6c	2026-11-11T18:11:52Z	docs(cuda): record B8 attention rejection
3a14c48203969c0a2a98b2c01596f7ad239381ee	2026-11-11T19:49:49Z	docs(cuda): record M128 tile rejection
af7f34c35ebf848e4f0ef14ffdcf3db8345f9d3d	2026-11-11T19:57:27Z	docs(cuda): record B8 row-group rejection
ea9b83b55b80b3a4cec602e55436bd143aaa47c2	2026-11-11T20:06:24Z	docs(cuda): record ordered N16 rejection
51c776b81d7745e3c7565ab0257b6904abb427d6	2026-11-11T20:11:26Z	docs(cuda): record B8 head cache rejection
54ea36859cff72dd4dd84504f9517de80ef2dd94	2026-11-11T21:08:36Z	feat(gpu): retain qualified private B4 KV transition
9197cfa4e5116c18e3f77ffb57c2c27ddeb05ffc	2026-11-11T21:37:32Z	feat(gpu): retain qualified batched source rotary
083cc6d966253c9604c6d3113a597742535e627c	2026-11-11T21:45:22Z	docs(cuda): record paired-column head rejection
5d7dcb4bb1fa1134795464a40a99a6a428955c76	2026-11-11T21:52:55Z	feat(gpu): retain ragged private B4 attention
da591acf4131c9c3d99afe584ab7078b8de62166	2026-11-11T22:13:58Z	perf(cuda): admit ragged native decode cohorts
1361c1385864b3971793f65a14a7d2031fe14b4f	2026-11-11T22:22:29Z	feat(cuda): retain certified BF16 row split
d1ec7236345745424920ded084b9314a41775fc4	2026-11-11T22:51:17Z	perf(cuda): batch GDN row splitting
59ae8499621644d870bb72498edd29612ceefb8c	2026-11-11T23:22:11Z	feat(cuda): add exact sigmoid multiply primitive
0f0af042e08b8293a3d8601e7666c3b82af982a8	2026-11-12T04:43:35Z	perf(cuda): pack B8 head inputs inner-major
265e970051370ac747895e93c9af216a0c663f66	2026-11-12T04:52:15Z	feat(gpu): retain scalar-position rotary batching
78ef7f279b52465be3133fb7b3b4083877cec55c	2026-11-12T04:54:17Z	test(cuda): keep strict GPU clippy clean
be1cd9e5a29e0a7e4a5d21c1350d70fea993f94e	2026-11-12T07:37:12Z	perf(cuda): batch ragged prefill cohorts
9204d75e3984d65d0165ce8bdb654001867f5b19	2026-11-12T07:56:21Z	docs(cuda): record unequal-frontier prefill rejection
790d3de0724ca499b9529fc878316177a49fbd2e	2026-11-12T08:25:46Z	perf(cuda): pack B4 head inputs inner-major
154483f96c9923357a45688c55ab9689f6fa5a4b	2026-11-12T08:36:10Z	perf(cuda): pack B2 head inputs inner-major
0136b9193244e268a409f44d4d3226af04f14d48	2026-11-12T10:19:17Z	perf(cuda): gather ragged head rows inner-major
801fb382bbeb91d2dd2701713ed81eb043a15d3d	2026-11-12T11:28:15Z	perf(cuda): batch ordinary rotary spans
cffd78e4d4dde13be548bd7ecd510db5360b7130	2026-11-12T11:39:30Z	docs(cuda): record explicit-position rotary rejection
f92ed9a1eec64973907ccc279fb7d9ae3ecb5f98	2026-11-12T13:05:51Z	feat(cuda): retain qualified stateless row spans
c911c2c5917ef9f9aaa3a04853833ab472b2ca99	2026-11-12T14:20:39Z	fix(cuda): share finite compiler diagnostics
c86e6166645c5b27658abdd22f097effa06e3c3f	2026-11-12T14:32:32Z	perf(cuda): batch compatible RMS spans
884ab441036d141703176e117087a1ffe6704825	2026-11-12T14:52:53Z	feat(cuda): retain stateless gate primitives
43ab4c6f4191d647f398e30e5855d10724c25545	2026-11-12T15:07:35Z	feat(cuda): retain stateless QK RMS primitives
b191fc60492e5c6e54c858fe60b50c73ee8e0653	2026-11-12T17:07:57Z	fix(cuda): preserve decode reduction contracts
1dce1b0d17fc2b0b15c0252550d76a8f6c8f1ba9	2026-11-13T01:40:15Z	feat(cuda): retain cached GDN B8 gated primitive
c6f3c90b71242fbeec43500128c199549a53b94a	2026-11-13T01:47:59Z	docs(cuda): record lane-blocked layout rejection
19296300f595588f1a2a7e29ab50fecb867690a6	2026-11-13T02:07:56Z	docs(cuda): record decode projection profiling
7331ef1ec3be8ccf8f8cc15220f9dfaa87f4211a	2026-11-13T02:18:11Z	docs(cuda): record decode transport and unpack rejections
0951bec114c0b317fb2f9d441f2acea57cd50477	2026-11-13T03:03:23Z	docs(cuda): bound remaining decode hot path
fe3b122e1e4a5521b775b6cb5fed9d552bf9c0bb	2026-11-13T03:57:02Z	test(cuda): strengthen state transport oracles
0cdfaad87b5f9e1df07946567706e1d768acab92	2026-11-13T06:36:43Z	feat(cuda): retain ragged private B8 attention
d196c1b956b6055d10603e7dc5334539736eaedd	2026-11-13T07:22:08Z	feat(cuda): qualify private GDN state binding
bad471387db5016eb2b752c311c6eff85b41cd20	2026-11-13T11:53:47Z	test(cuda): qualify in-place private GDN state
16fba0ee49a7c14e970b3892c4d4857a11e2c1b3	2026-11-13T13:10:45Z	feat(gpu): retain exact-root derived state cache
2f651b7e62819abe6c5ba6606bf67d7b56c670e7	2026-11-13T13:54:34Z	fix(gpu): harden resident session root binding
60b048d47b97c60e802e64b86444d0a43cf54913	2026-11-13T14:59:26Z	perf(cuda): batch canonical-boundary GDN recurrent core
775dfa6051aaf11571a8b47a253826935248cf44	2026-11-13T15:57:51Z	perf(cuda): batch ragged B8 KV transitions
c12acab60c357e0742ad4ecd2f00295cd9b6bb0a	2026-11-13T17:17:08Z	feat(cuda): retain grouped B8 recurrent transport
85f458f554edf9be7bac491309e1844f27e76351	2026-11-13T18:14:16Z	feat(cuda): retain packed BF16 batch greedy
f303bca53286a7c96ae1ab78b95532244dd756ef	2026-11-13T18:40:01Z	feat(cuda): retain exact head greedy partials
68ac8109bd47b6ace8f006ce109bac043f595893	2026-11-13T19:02:53Z	feat(cuda): retain grouped ConvB4 transport
da3eae5e8f5ce2d27d167e84690db5e27a703d3d	2026-11-13T19:20:50Z	feat(cuda): retain dual QK head RMS
ee347d93bc7b2f7e362b8b07b03cf0e9f9523ea9	2026-11-13T19:43:50Z	feat(cuda): retain certified recurrent export
399e43a0a5d0c22927bdc28e1d35775be0393c96	2026-11-13T20:08:14Z	perf(cuda): widen B8 private KV CTA
de2eed8cd1eabd059f377d5a18f269f5ef39fc8c	2026-11-13T20:44:50Z	docs(cuda): record post-KV optimization probes
cb9e349946c6daceff7d58ff28df1005dfa3ece4	2026-11-13T21:55:57Z	docs(cuda): record projection cache probes
6c601f9131ef6ac33c2dcd15deb37daad35432a8	2026-11-13T22:13:13Z	docs: refresh comprehensive handoff
c282f20ea10c16680b033586bf11d0e5d62c0189	2026-11-13T23:11:19Z	fix(hip): restore source profile compilation
874e67db0b6e737e6d9de31e44ade597b3160143	2026-11-13T23:18:11Z	feat(cuda): retain shared-input projection pair
2caf304e89bc12c77308f993df267356113baac7	2026-11-14T03:09:35Z	docs: refresh handoff after HIP and CUDA checkpoints
52c8a16187c7d89977c5308ebddfe094aeda5e70	2026-11-14T03:33:12Z	docs: record X2 HIP physical health
134a571dc080ac8e2d0f243d3e4aac0ff4cfa5dc	2026-11-14T03:56:08Z	fix(hip): isolate CUDA private KV qualification
ca6e99f9b54bf880943f72752350a3cb105c21e9	2026-11-14T04:07:41Z	docs: localize 27B X2 host-load blocker
b0cf0b21e5a72a9e89c0431f8437afa2a635982e	2026-11-14T04:43:00Z	fix(cache): avoid duplicate Qwen snapshot hashing
f5d2a88ffc8cb82c2a6686ad4e913768186b3b88	2026-11-14T04:45:37Z	docs: record Qwen source-open localization
d68b649031ac89ddfb88f6880cb7185208b25c09	2026-11-14T05:04:48Z	docs: record X2 27B physical allocation blocker
5c8f14375d4ee46afe3cc02a95d710fe813eae6f	2026-11-14T05:24:29Z	docs: localize stranded X2 KFD owner
3cadc3b2ac8eadc95a8e1b19003b922e355c339d	2026-11-14T05:57:30Z	docs: reject shared-pair Qwen serving consumer
5bf440d7f891111feac63f50d04a4333b40286bb	2026-11-14T06:17:11Z	refactor(cuda): admit graph substrate in native builds
bc9a1b3485776240513ff8be08500881def4054e	2026-11-14T06:18:28Z	docs: record production CUDA graph substrate
db7be604addc4f908efb8172eb51306ec436104f	2026-11-14T06:48:17Z	feat(cuda): admit captured graph resources
cbf13841b56985c0be0642fbe8de8fbc2022e296	2026-11-14T06:49:29Z	docs: record captured graph resource admission
6a5c6101a6979d4c56c20de4c2065f44bffe8463	2026-11-14T09:32:38Z	fix(cuda): make nonpersistent NVRTC driver-safe
9dca4b84e8899bfff61716fe3f11ba9c6f23051f	2026-11-14T09:39:32Z	docs: record physical graph resource acceptance
81b7c71324e6c3ba99d4d77bff7e4e14b2ddcf28	2026-11-14T09:52:04Z	test(cuda): witness graph resource lifetime
b12e8f5395af84bda5a5c0eab960efef2399f6c7	2026-11-14T09:53:10Z	docs: record graph lifetime repeatability witness
f1ce532d49c4121324eb5f5a401f85ea900f1251	2026-11-14T10:16:07Z	feat(cuda): bind graph replay to resident flight
9c86c0cd8396abd023d4504c6f2a559249d331fd	2026-11-14T10:18:27Z	docs: record graph resident-flight authority
bc0e2673bc45e42379a40c23bc26b0d132069b0a	2026-11-14T10:30:06Z	docs: add full comprehensive continuation handoff
63ee14e440d73beedab6e2acb96838926968eed9	2026-11-14T12:47:34Z	feat(cuda): own resident graph lifecycle cache
4502f5678e8cb1641848be62cd0fb5068bae1d09	2026-11-14T12:49:43Z	docs: record resident graph lifecycle checkpoint
9d6613cb0bef84147a7d30453e5afb82ca83411b	2026-11-14T13:52:02Z	feat(cuda): select compatible resident graph cache
bc76312a915b1f0f20f20505443d21682dd18724	2026-11-14T14:08:34Z	docs: record resident graph selector checkpoint
599cb28945bee79ef70dc48ea35566c526a7f049	2026-11-14T18:19:51Z	test(cuda): measure resident graph selector widths
4be26f29a4b43cfa915a89b45839e875bdfd71fd	2026-11-14T18:21:32Z	docs: record graph selector width decision
b19bdb481b3bbc194eb4d021ed40112197a92e75	2026-11-14T19:12:58Z	docs: record cold prefix reconstruction checkpoint
f4fb04b6c0dec311bb52affd10abfae755a01a54	2026-11-14T22:26:55Z	docs: close reconstructed prefix scheduling benchmark
7dea460a80b420d0ce49a60f6c860b3502c67c38	2026-11-14T22:55:14Z	feat(serving): age prefill behind decode
ba7e3768197c417e42eecd7315f6043f1d5e9464	2026-11-14T23:18:10Z	docs: accept fair serving interleaving
9dab0043719e435ef3914f86f3966079deabc5ae	2026-11-15T00:12:55Z	feat(cuda): add temporal fused prefill primitive
1e0f4ebfc5b9650341814ad669d4101aea717623	2026-11-15T00:56:53Z	feat(serving): enable qualified temporal B2 prefill
b599db32dec2620014f2445b73d378aab8299d56	2026-11-15T01:08:50Z	docs: refresh comprehensive continuation handoff
12f784083f4057dcf36fd343e408d5a7ed482589	2026-11-15T01:49:20Z	test(cuda): qualify temporal B4/B8 prefill constituents
80409ac30a12c03f9ce4585068faa44aab791460	2026-11-15T01:49:37Z	docs: checkpoint temporal B4/B8 qualification
330150fe23dd97be1fecac42de39b9058e2140e2	2026-11-15T01:50:37Z	docs: record post-qualification GPU baseline
36253481d5e26d27ffb577187b21c338a211beaf	2026-11-15T02:06:37Z	test(cuda): bound temporal qualifier host state
4d74d7f92bc4dd69b18bf947286dcb6a537755ff	2026-11-15T02:08:24Z	docs: close temporal B4/B8 B1 qualification
eaec62d8a40c6179baa49e62be7a358b4b800128	2026-11-15T02:59:51Z	feat(serving): enable qualified temporal B4 prefill
9347f5b838e389f68b4eb38f6d51b664b844704d	2026-11-15T03:00:34Z	docs: accept temporal B4 production
8e6f14d3defc0aa2e542c26ffb9ccc7c736f1fcb	2026-11-15T03:09:51Z	feat(serving): enable qualified temporal B8 prefill
fc9ba86567522b1a9171715ce520b820d7cad60c	2026-11-15T03:11:54Z	docs: complete temporal serving foundation
0027c282fb28fcf846c141f032ea0e4e4fdee599	2026-11-15T03:24:04Z	feat(state): add paged COW block-table primitive
fc99e9c3354500d1a8f946ad1c19d1f2e0b80d31	2026-11-15T03:24:54Z	docs: checkpoint paged state COW foundation
54340a6d848ff5f19bb19333f550b1c057328881	2026-11-15T03:43:39Z	feat(state): bind paged tables to resident pages
818cbb67a55518e3b44ad5c6393e85c67311b887	2026-11-15T03:44:22Z	docs: accept resident paged ownership
1eb6e0f669f7bf6f31e4a0b7cdbdffa14ef73b7c	2026-11-15T05:33:43Z	feat(state): materialize paged COW appends
2c908aef374d817825acd7cf4909358cfe8ec7ad	2026-11-15T05:34:39Z	docs: accept paged COW materialization
3ac205991086e4d9e599993bc7960010d2f8706a	2026-11-15T05:56:22Z	feat(state): add paged KV gather representation
5e6b391e8b6f417882affb54875439fc0fb19094	2026-11-15T05:58:00Z	docs: accept paged KV gather boundary
dc6a188bad8d30a28d6871f0798ae98c4a6803bd	2026-11-15T07:47:05Z	feat(state): add paged KV staging scatter
db3c155f52d6c2c16927208ff0010be03dd9d447	2026-11-15T07:47:59Z	docs: checkpoint paged KV staging bridge
53336b3d961f2ffc59c0b845f876ff4e8e0033c9	2026-11-15T08:27:03Z	feat(state): admit paged KV session domains
4e5aaade5bd83700f907d4f44e380dbbec20bbd7	2026-11-15T08:28:45Z	docs: checkpoint paged KV domain publication
ef9e4d4c4364df423d4a6afad108a1c5b562d967	2026-11-15T08:48:09Z	feat(cuda): add in-place paged KV staging mode
240e0f23def6d0993c9022a8c43d482cdeda58ff	2026-11-15T08:50:08Z	docs: accept in-place paged KV staging
93468b1f5062804f009f817c6f5dd50bc7d0516e	2026-11-15T09:12:34Z	feat(state): validate hybrid paged root semantics
4bafad794e36cb2c4b9647d8807bfc0f8818b4c4	2026-11-15T09:13:28Z	docs: accept hybrid paged root semantics
cb57e1d8bd4d33b530c4efc771e19450df1ef55e	2026-11-15T09:42:21Z	feat(state): construct hybrid paged state pools
9815c2abc63b81cf7ecbbc74041356d9e8a0f10e	2026-11-15T09:43:21Z	docs: accept physical hybrid state pools
b9b04aafd9d10b0325ec5daac45cb3f6be8fedf0	2026-11-15T10:12:58Z	feat(state): execute hybrid paged candidates
d944eae24cb0686d0d9eb21979f93984bac9c09f	2026-11-15T10:15:33Z	docs: accept hybrid paged candidate execution
1413f729ac5db55131b8379752b8cdf0a7b98263	2026-11-15T11:02:06Z	feat(state): add explicit paged source policy
919538ef25d338decfba1d5db54c5a2c3a5801e8	2026-11-15T11:05:53Z	docs: checkpoint explicit paged source policy
398e4461fc3bd30a84e050951dbdf204b05f0cdd	2026-11-15T11:40:07Z	test(state): qualify paged source lifecycle
3a1596db12d0f2e5d104ed1980f57af9ae27a513	2026-11-15T11:40:54Z	docs: checkpoint paged source lifecycle qualifier
30414571c5998880d99c7f685227b83434e6c03c	2026-11-17T14:51:42Z	test(state): fix paged restore qualification
30b311c1a0ca717a87d53a184b88fad5f29341fc	2026-11-17T14:53:44Z	docs: accept 0.8b paged source parity
e6aac539edcf631199bb51628c7f9e361e65057f	2026-11-17T15:55:32Z	test(state): keep paged lifecycle cold-only
cb35ce220edc1d69fa17d80050992f630dec997a	2026-11-17T15:58:43Z	docs: accept cold 0.8b paged lifecycle
a21ef0fecdc9727bcd1e8f8dbd742975cb3b4a5a	2026-11-17T16:27:22Z	docs: correct dotlink signal attribution
28890d1526538352423ff376f53042ad4636462e	2026-11-17T16:39:21Z	docs: record 2b paged sandbox blocker
816e123c3fb68b2322aa3f23896a936a5d3c358f	2026-11-18T05:27:19Z	docs: record unrestricted 2b qualification route
8b65a8ddfb55e20d948d700fe487ed03e94011fa	2026-11-18T16:30:38Z	docs: accept 2b paged source lifecycle
c6ba1c658e77dee6315c77124ff9d15e8df284cd	2026-11-18T17:30:00Z	docs: correct heartbeat harness attribution
6b920a0a91e534eae86f81e603cf419dea3b207b	2026-11-18T17:50:59Z	fix(harness): reuse existing sandbox isolation
2e0f5f8cd35e1e2896bb5b87fdae77915e566907	2026-11-18T17:54:11Z	docs: checkpoint native gpu harness isolation
e65f00a88f8ee92e2b175e3db96f932df59d2a0f	2026-11-18T18:26:26Z	docs: create full comprehensive handoff 2026-11-18
cfe734af1dc7e478c826e70469d917812d36dd9a	2026-11-18T19:18:20Z	feat(cuda): retire in-flight paged cancellation
3e8c4e71c38f23be304756734726f6ef9e9283a6	2026-11-19T03:18:25Z	feat(cuda): expose explicit paged state policy
7c13aa362f10553b8858c15a9dbd8a60f8b5a28e	2026-11-19T04:01:55Z	feat(cuda): add paged native cohort transport
3d4de835e1bddaaafe622634789b329b6e5d7ddd	2026-11-19T04:25:26Z	test(cuda): qualify paged native sdk
4688cbae9e28838d3b21ff1674f4c48dec519cd9	2026-11-19T04:36:53Z	fix(harness): recognize isolated child netns
2452eee189581d7ecb608ac5b937499222b4577d	2026-11-19T04:54:02Z	fix(cuda): restore native feature boundary
ca8fa69ecd8e58fe7f3dd0720839bdf58b91ae8a	2026-11-19T04:54:31Z	test(cuda): qualify paged native http
30aa1ae7903ea772f12beef8535dadc784f6de17	2026-11-19T11:05:36Z	feat(cuda): fork complete prefix snapshots
8c8affa9a3cdf7f2a9b9b0eec13a0392f8809be0	2026-11-19T11:11:47Z	feat(engine): add bounded prefix cache
c24246ddbc4aa285a84f93be35ccd8336119b875	2026-11-19T11:54:16Z	feat(engine): enable scoped prefix reuse
6303db8ee38f1c7ab5742d17a1c7e05138a70137	2026-11-19T12:10:15Z	feat(serving): scope HTTP prefix reuse
f83bcb9a7ca8e7a683b02ad9770114873709c34b	2026-11-19T12:28:19Z	feat(engine): bound prefix cache retention
c0d0e69d792169340b64e7e013013911d2332086	2026-11-19T13:17:31Z	test(cuda): qualify paged pool pressure
c347fda06f5c267b1ae1c7106447c9e9b66de8e8	2026-11-19T14:01:02Z	feat(cuda): add owned secondary stream events
064b99bc0531e0d02b8c63b27e1fe82c670b44ce	2026-11-20T02:46:23Z	feat(cuda): retain resident event-backed flights
e1080c0a17a059070089edc2eccf9468053a797a	2026-11-20T05:25:51Z	feat(gpu): support bounded resident multi-flight
9147b3fbfaf37266097e14631597f56f7b98d6f0	2026-11-20T06:11:58Z	test(cuda): profile native B8 serving
19c012f924856950401c706ad23286fd4a8a02b7	2026-11-20T10:18:42Z	docs: correct Qwen B8 prefill diagnosis
05fe44da6af13e44e38d4ed5b3dea6570b5cb5f1	2026-11-20T10:40:17Z	feat(cuda): retain stateless prefill cohort lowering
f0f06402c78d4511dea1559c73d4295f88790f93	2026-11-20T13:41:21Z	feat(cuda): batch multi-token prefill attention
1222f8a11f0c773fba41a502b84ce0862d30442c	2026-11-20T15:06:56Z	feat(cuda): batch multi-token GDN prefill
60ba956c3057019ad615c5b25ede3a9f7167d955	2026-11-20T15:30:39Z	docs: align serving perf feedback profile
3afb9902f674d9cfe2fc4488e628b61b61cae395	2026-11-20T16:23:56Z	fix(cuda): retain GDN layouts for reduced bands
7beb1be30312680720eaafc9abb10eca23648e51	2026-11-20T16:41:50Z	docs: create full comprehensive handoff 2026-11-20
0dac5abfd3a8edcd82d955a6a125c1f2c5426ec7	2026-11-20T17:02:54Z	perf(cuda): zero paged state on device
0680b0b8e757ebeb7003f315f08433d7c8bf299f	2026-11-20T17:28:55Z	perf(cuda): batch paged KV gathers
e72030cd04f71ad7a3cd0b11d7d771875a3f9d88	2026-11-20T17:43:45Z	perf(cuda): fuse paged gather tail zero
dfc3e396c9e969c48a1a55bd6d35cb9a8ced1924	2026-11-20T17:57:54Z	docs: record continuous refill goodput
4ec3a0cb36f0a2affd5b092d1262952614428219	2026-11-20T18:25:56Z	docs: record x2 ROCm runtime blocker
5a95da56316b9e8757175ef01a97472bc656491e	2026-11-20T23:48:38Z	docs: reject mixed prefill regrouping
9122d1e0d4b6b04d0d99b6d5d7ccc083015237d5	2026-11-21T00:04:22Z	feat(gpu): retain cross-model perf candidates
29a6238b6c2b59f019928672b3d169cc318111fd	2026-11-21T04:11:14Z	feat(gpu): restore dormant kernel candidates
468a36054f6004eaa42d1d7dc14dc52df029af3e	2026-11-21T04:27:11Z	test(cuda): qualify populated 4k 8k serving
127dae431a7db55d9f38dd305b87b94e5af88401	2026-11-21T04:59:39Z	docs: record populated 16k serving
5466577d3976d2ab56471a27c614debe8ccb2425	2026-11-21T05:14:01Z	test(cuda): retain current 32k context qualifier
e79814ce0c0a17a93e412b45d7dd66418e6acfd6	2026-11-21T05:49:19Z	refactor(errors): preserve resource exhaustion context
603844ba0556be3491828b9495535f8c9d1dd183	2026-11-21T06:19:23Z	feat(cuda): expose exact source admission inventory
973aca4adb7f2ad2dd7246ba1fc51a23c8e44826	2026-11-21T06:44:57Z	docs: record persistent external gpu sigkills
f86538345fe4ce412df81864542c5ec183446460	2026-11-21T07:24:08Z	docs: record 32k trace pass and external kill
f1cfd26a2f64e04f6d9ba6f7cf313b15480e47f3	2026-11-21T08:53:02Z	feat(engine): reconcile exact prefix frontiers
763eb99d58c815bcf1fe22eefb710b9cf8ef6da0	2026-11-21T09:04:46Z	feat(core): expose physical buffer identity
82d137ec8002af161cdcfeb3f680ba2608c1df54	2026-11-21T09:14:14Z	fix(core): scope physical buffer identities
3802b9de1d7d6157165d4da76d07a0d76e03de33	2026-11-21T10:05:28Z	refactor(gpu): centralize resident alias checks
e8c36fb7d5065d63b8e31a221bb6f2a7e4f75d31	2026-11-21T10:32:36Z	docs: localize 2b 32k response timeout
9896a85832c3db9338cdb9e114653e1e27ae3bc2	2026-11-21T10:58:57Z	perf(cuda): drop unused attention reserve
be02ade8e81ff356c9bca5cbba1796d08ec692ed	2026-11-21T11:21:05Z	perf(cuda): price reachable paged state
0dd844044497ced882c59595a49dff8cdf4ffd86	2026-11-21T12:02:59Z	feat(runtime): add smart model startup config
28459c231764277c141f7fb078b29dc9ee0d403a	2026-11-21T12:19:20Z	fix(cuda): honor live provider headroom
3b5f191eb94b027179604d04ebac7a09b6b1050b	2026-11-21T12:37:27Z	feat(runtime): smart-select native cohorts
849cb87d1f295c8acf6dcf4d8bf7638e830ed117	2026-11-21T12:49:59Z	docs: close exact partial-tail COW ownership
6c68a361e843c73107a86aaedea5ca697a30258c	2026-11-21T13:08:28Z	refactor(cuda): use ledger identity for prepared leases
f9acd9e49a5eaf29e02d89b2c6f31ec468871ef4	2026-11-21T13:43:28Z	docs: consolidate 21 nov comprehensive handoff
b616429a41d26a4fefac209bc487dcfa9191c9f1	2026-11-21T13:44:56Z	docs: make handoff checkpoint self-stable
f42d7722eb4f4f0f392c9b6e6348555162eb2f7a	2026-11-21T14:04:50Z	refactor(cuda): use ledger identity for streamed head
968a3682492e2675e7837d13c767905f895fb2ae	2026-11-21T14:35:20Z	refactor(hip): use ledger identity for source calls
b839920ec0553c4a27dc753c076487f0ac2c482d	2026-11-21T15:27:46Z	perf(cuda): reuse pinned source staging
8c96a886cf8746ed2ad49208c2c43aaa94e4fd7c	2026-11-21T16:04:37Z	perf(cuda): upload source pages from retained owners
b88d3b3728b4eea2e249f4680be9c83edfeeb034	2026-11-21T16:41:16Z	refactor: clarify model and cohort terminology
f3f000055cb949f7db367b3d1bc427c472faceb0	2026-11-21T16:58:54Z	refactor: name qwen source execution forward
001b4c6fbee3b70770d4065ca4cd8a124982b93b	2026-11-21T17:51:39Z	refactor: clarify text invocation terminology
e8cf4d89feaa4fdb6f02ea9191620c25053be9c0	2026-11-21T18:54:28Z	refactor: clarify autoregressive terminology
d2adbaf3cbc04dd41c27fd56b90222b786c25dcb	2026-11-21T19:01:21Z	refactor: name complete state coverage semantically
ff099e626882201fbded10de3a2f4cbf7242497d	2026-11-21T19:13:42Z	refactor: name model-wide coverage semantically
3fb731ffab12d01ed3f58ffe7e6a4c15c5e664ba	2026-11-21T19:20:17Z	refactor: name fixed prefill trajectory semantically
c0cc636c10e5378174750896c61602ddc0855348	2026-11-21T19:36:35Z	refactor: clarify fixed-prompt workload profiles
4e03edb9a08e9a17772bfd2074620448956c5c57	2026-11-21T19:44:47Z	refactor: clarify cohort and concurrency terminology
2a87e3dfc77dd8aab894f719a8a54b89d944a96f	2026-11-21T20:10:22Z	refactor: name zero-centered rms explicitly
fb90d9ee199c373d5354101079e3394cb7b52694	2026-11-21T20:40:06Z	docs: record rejected cuda source-page runahead
37ec699c4ba67a5cbad4b5088b2a17902669e997	2026-11-21T20:56:11Z	refactor(gpu): retain backend transfer retirement primitives
fdcec908987184a2e2248b4da60b30c22472a040	2026-11-21T21:14:02Z	feat(cuda): restore reusable gemm candidates
8dfaaf05560e39e2c9916b7ec27ee3dd90db80b8	2026-11-21T21:52:15Z	feat(bench): add nix-native vllm cuda provider
4201095e9bb4b9704a486a151145c31adaa7a326	2026-11-22T03:01:51Z	docs: consolidate 22 nov comprehensive handoff
15f5c4fde2bc6659f47a822e4b42844a8077f21f	2026-11-22T03:19:53Z	feat(bench): realize vllm provider with nix
461ec52698e1907258408dd377a229e07084da03	2026-11-22T03:45:05Z	test(bench): add public cuda population benchmark
169a92a519afacb374fe2570cfc72b01544d042f	2026-11-22T03:51:52Z	feat(bench): support raw completion workloads
277a27915e5392e6fe950961e2bbf94c4ab4025f	2026-11-22T04:49:14Z	test(bench): add hip population benchmark
93d80f3686801bad417f19ff0372f75328046bd6	2026-11-22T07:05:54Z	feat(bench): add nix-native vllm rocm provider
4d55d231cf765f4e4448c81cb8709b5f5a81df43	2026-11-22T07:18:42Z	docs: record nix-native rocm smoke
c8b1bed7762b1c28262ab2254a7c87686cca4eed	2026-11-22T07:24:52Z	fix(bench): provide rocm triton host compiler
6e2382fd2ab53ef23a426ce0816e8da778fec75e	2026-11-22T07:33:55Z	docs: record rocm vllm c1 smoke
2a595be84bc7c5dccf7509e35e4f9b3a341662a2	2026-11-22T07:49:18Z	docs: record matched rocm vllm small models
b34c025d36d0e7eaca6d559504e498f998552e19	2026-11-22T13:10:28Z	test(bench): establish clean current baseline
94a7e2d02db72b3f0d172e46b14eeca7da05e33a	2026-11-22T14:45:15Z	feat(hip): batch source decode projections
48d3d944dc003b181c915cc013348ca93f0f137f	2026-11-22T14:55:03Z	fix(hip): size decode cohort head scratch
79de45a3ff637ffe096e18a7f202a701edd982b9	2026-11-22T15:50:27Z	test(bench): observe hip decode cohorts
cb923cc2b7e79b1d40df705c78989c4bf2abed83	2026-11-22T16:00:35Z	fix(hip): price decode cohort projection scratch
189c7187601115b7915ac092ca23d595607ac546	2026-11-22T16:06:15Z	test(hip): accept fused cohort observations
b6c60ef3b1aad3ce828519801b748dfe8d4f15f5	2026-11-22T16:10:40Z	docs: record hip decode cohort checkpoint
99dcbbee20b50b8e588f167cf2c0ec40b31782c7	2026-11-22T16:25:56Z	perf(hip): batch row-wise decode operators
e35e650c6bebeeb814c770f8aee1a031a819415a	2026-11-22T16:28:18Z	fix(hip): compare decode cohort semantics structurally
537e5e8542f6d07588dddcbf52a5131f1fc5a6e3	2026-11-22T16:31:31Z	docs: advance hip decode cohort baseline
9ee877c7ba87ad233228ef71a2803ed4063996ec	2026-11-22T16:39:02Z	perf(hip): batch aligned decode rotary prep
a5ec4c31eb39d480bee499bda6b7757fbf9b4d74	2026-11-22T16:41:48Z	fix(hip): seal batched decode rotary variants
a3908378fd217124fdf1581763002df190133b75	2026-11-22T16:44:56Z	docs: advance hip aligned decode baseline
1de99ef6210941a662bc0fc5155092e7f366fddc	2026-11-22T16:53:57Z	perf(hip): batch independent decode conv state
c33016d649e850d33b13e9f5cdce254e35924353	2026-11-22T16:56:59Z	fix(hip): preserve conv coefficient input contract
57cc384c4ff55f5576ccc9a321fe9600cf70ce24	2026-11-22T17:01:24Z	docs: record stateful hip conv cohort
491cf9957d83bc46100cc863b0b062909459fbd6	2026-11-22T18:48:50Z	feat: unify production GPU build and run paths
6f8f07111fb38eb910c051063cefe76f8c1b75bf	2026-11-22T19:09:38Z	refactor(gpu): separate attention semantics from providers
64c06e2fc313d3b24c0a5abe0065a085879f0e90	2026-11-23T00:41:27Z	feat(cuda): add opt-in FlashAttention provider
ed0382a192ac882be0fb8e2346e23a4bab252284	2026-11-23T02:00:40Z	perf(cuda): keep FlashAttention off decode path
0a35f175629f0841b98d78fc156c0c30eaf88df3	2026-11-23T02:13:58Z	perf(cuda): fuse qk normalization and rotary prefill
2805c37577a185d7065fcb9397f57827d1010d90	2026-11-23T02:20:17Z	perf(cuda): enable fused qk rotary prefill
179bb1528339c258b7cb125086ac6fc692698d18	2026-11-23T03:01:57Z	perf(cuda): fuse GDN recurrent export certification
c829aa9855f3eebfc0a6635122d6448c781d27e5	2026-11-23T08:27:11Z	feat(cuda): retain B1 direct GDN conv primitive
f3f872cb567a6885918ae2b4866ee6d535d736d8	2026-11-23T08:44:57Z	feat(cuda): retain B1 fused head greedy primitives
fb20345bb7b5ce0b3285470c77564e492331254b	2026-11-23T09:48:25Z	docs: consolidate 23 nov comprehensive handoff
0c4d95e6a1fce0e162927ef2b55647fac7d1fba1	2026-11-23T10:35:52Z	docs: revalidate production feature convergence
6f4418ba8262baff0dc023923865ea8bcfb7d63c	2026-11-23T15:25:54Z	fix(qwen): allow tokenizer-owned chat prompts
2d16cd5aab95949cd295070a8a5df84172409f29	2026-11-23T15:28:47Z	docs: accept a1 hip public canary
11f692bf7b2871e1dac974eb96bb27f3968ee316	2026-11-23T15:48:53Z	perf(hip): batch recurrent GDN decode state
7fcc9c4739b9bceedb298d4835a55ad5b64c1a84	2026-11-23T15:56:11Z	perf(hip): retain recurrent cohort primitives dormant
31eaf496385890005f04a7c920ea5b9ee367404d	2026-11-23T15:59:38Z	docs: reject hip recurrent cohort selector
69630ec3426d2ca523cdbe5e2183b01f68b93f49	2026-11-23T16:10:31Z	perf(hip): avoid dense recurrent decay scratch
0d201ff0bac05aa92d48dc8faa4e4cbfa2cce7d5	2026-11-23T16:33:50Z	perf(hip): fuse cached GDN coefficient exp
879bee798a29f5da5a0fa4d2812840604f0bad76	2026-11-23T20:12:58Z	perf(hip): keep singleton recurrent primitives dormant
4070610433a3068943e844b2135d9776beffaa47	2026-11-23T20:27:47Z	perf(hip): write singleton projections directly
fdc5a8ec86aa9861576fbc33bfbfb4a4cc4c2ad4	2026-11-23T20:36:10Z	perf(hip): reuse packed singleton projection slots
baf4c9f090e4106cc4a35439a30854328adfaea3	2026-11-23T20:39:17Z	perf(hip): keep projection alias contract narrow
ec6fb57f08b948c804f869787a5a17f52e038846	2026-11-23T20:43:39Z	perf(hip): coalesce singleton projection status
d15b02f6973e36baedb4b6a374058b48e2e76795	2026-11-23T20:51:33Z	perf(hip): retain projection status coalescing dormant
e606586dd3cd7ef76c2ec832bb5cb76c3b9c87af	2026-11-23T20:53:29Z	docs: record hip singleton projection A/B
e1701e707de5b23589e24509b5b401cc1f138720	2026-11-24T00:43:51Z	perf(hip): reuse singleton projection input pack
1d49ec7c4760e0f00c9af2f6d0c7babf250a2530	2026-11-24T00:47:46Z	perf(hip): retain prepacked projection path dormant
5d51896532574a08d22912b44da127a6564e3cec	2026-11-24T00:50:48Z	perf(hip): certify singleton projection in-kernel
90537b6c2da8f626e08e7665311d49e249c4bad6	2026-11-24T00:58:05Z	perf(hip): retain certified projection primitive dormant
9e91a0d26a7b77456411b15074e77098f7b754c3	2026-11-24T01:05:40Z	docs: record hip projection and native accounting
c5b3fadd214d9d89a84e6c6d7da8fc756bf5fc76	2026-11-24T01:29:29Z	perf(hip): qualify gfx1150 native module allowance
fd4611e12fc1c2f173b2d80d4e8d2105fe601327	2026-11-24T01:37:07Z	docs: accept gfx1150 resident hip policy
ffd5ff8f951c5db9b113865c6d5a22a954e282e7	2026-11-24T01:56:10Z	perf(hip): fuse residual rms first pass
dfc57183f8fe36f00e065e9b7c1482b6480488ec	2026-11-24T02:03:19Z	docs: record resident hip hot path
b44a55c868e72e04f68d7e900ee32b822cc89c32	2026-11-24T02:09:10Z	perf(hip): fuse rms scalar cascade
11e61f850557c4303a95c2686a8afb6af5de3591	2026-11-24T02:15:26Z	perf(hip): retain rms scalar cascade dormant
9040492544f0ea6e114fd749232cc2b185aa8a65	2026-11-24T02:20:20Z	perf(hip): fuse rms vector tail
654adbb17a35d66922ba918477f9b872b704b605	2026-11-24T02:25:24Z	perf(hip): retain rms vector tail dormant
288e978666463148475900b9db167f41ee5af99d	2026-11-24T02:26:15Z	docs: record resident rms fusion A/B
7ff8784c8d454af634a143d30d228e8164d78527	2026-11-24T02:51:20Z	perf(hip): reuse resident singleton projection pack
36c318db569916ac8bf264e33b4d9dc25dfd08dd	2026-11-24T02:55:01Z	perf(hip): keep resident projection packing selected
1b25fc30cab76b93e1bb02f3b09e1f6b20bc5ef6	2026-11-24T03:02:07Z	perf(hip): share singleton projection input across four rows
5e4cac97fa78eefb1f4efc721bfee1ae0c3317a3	2026-11-24T03:06:20Z	perf(hip): retain four-row projection primitive dormant
a5c57c13308cf86b2daa0c2cce4e463f1fb4c447	2026-11-24T03:14:31Z	perf(hip): parallelize cached gdn qk normalization
072b79128163e2c18a0bfa946099875b13fd9122	2026-11-24T03:30:33Z	perf(hip): retain exact wave32 projection primitive
b4fc23188a10583588a610d3a6657a37fee7cde6	2026-11-24T03:39:52Z	perf(hip): keep wave32 projection primitive dormant
d6043af75b9f0e557afee142176df3f3454b16e0	2026-11-24T03:39:58Z	docs: record resident projection device profile
253a565b543447833bd113309098f46ee113155e	2026-11-24T03:54:46Z	perf(hip): retain exact rows2 projection primitive
6629ee26a9179ac2b3ec62a1e6ef5996cc6e6b04	2026-11-24T04:05:42Z	perf(hip): keep rows2 projection primitive dormant
a8c92a2984c6f1de04fe901e367b174c1319eea3	2026-11-24T04:05:55Z	docs: record rows2 projection A/B
6d5dc72d3711cd3b0e14db0a3c007851984a0745	2026-11-24T04:37:53Z	feat(cuda): integrate flashinfer decode and paired gdn state
55477b4b8b6e52db3b11450cb8fdc68fffbb5cbb	2026-11-24T05:09:21Z	perf(hip): retain paired packed-load projection primitive
ba8c5ea7ac6fb8575f102607cdefd0155de510da	2026-11-24T05:17:53Z	perf(hip): retain input-certified projection primitives
49e5149887a6687733b7dfb9420eb8f5575043d6	2026-11-24T05:28:26Z	perf(hip): keep projection traversal alternates dormant
e1929762a87b1e34de099d9f40b85bcff385c27d	2026-11-24T05:28:32Z	docs: record projection traversal A/B
~~~

</details>

## Appendix D — dated comprehensive handoff chain

- FULL_COMPREHENSIVE_HANDOFF_2026-11-14.md — 86,952 bytes
- FULL_COMPREHENSIVE_HANDOFF_2026-11-15.md — 131,347 bytes
- FULL_COMPREHENSIVE_HANDOFF_2026-11-18.md — 265,825 bytes
- FULL_COMPREHENSIVE_HANDOFF_2026-11-20.md — 401,084 bytes
- FULL_COMPREHENSIVE_HANDOFF_2026-11-21.md — 1,570,993 bytes
- FULL_COMPREHENSIVE_HANDOFF_2026-11-22.md — 2,471,529 bytes
- FULL_COMPREHENSIVE_HANDOFF_2026-11-23.md — 40,354 bytes
- FULL_COMPREHENSIVE_HANDOFF_2026-11-24.md — 223,761 bytes
