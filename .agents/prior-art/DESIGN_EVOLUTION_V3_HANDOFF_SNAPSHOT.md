# Design evolution archive — complete v3 agent handoff

**Non-normative historical snapshot archived 2026-10-12**, preserving the comprehensive original v3 takeover text with decisions, prior-art research and older execution contracts before the fresh v4 critical review. Current authority: `MODULAR_AI_SYSTEMS_REFERENCE_ARCHITECTURE.md`, `COMPOSITION_AND_EXTENSION_CONTRACTS.md`, `MINIMAL_EXECUTION_SPINE_AND_PROTOCOL_V4.md`, `TASK_PROGRAM_AND_COMPILER_REFINEMENT_V4.md`, current v4 master and decision register. In particular v3 StateDomain::commit/TaskNode/4-enduring-IR and single-owner physical-allocation ideas are superseded by v4.

---

# ZEROGRAVITY — FULL COMPREHENSIVE TAKEOVER HANDOFF v3 (2026-10-12)

**Project:** zerogravity — an original Rust-authored **modular AI task compiler, execution engine and serving runtime**.  
**Canonical root:** `/home/pvl/spaces/abird/src/zerogravity/`.  
**Evidence corpus:** `/home/pvl/spaces/abird/research/inference/`, **read-only** unless the user separately authorizes edits.  
**State:** COMPLETE architectural research/planning only; **no Rust engine, Cargo workspace, compiled GPU/NPU device code, model execution, physical test, benchmark, server, Git repository/commit or release exists**. All ZG-P00…P23 implementation phases are NOT_STARTED.  
**The latest user directive U-017:** express all normative plans in original, project-neutral capability terms; prior projects are explicitly attributed **prior art**, with source-specific direct references outside research retained only as needed for verified MIT-licensed sources. Do not turn someone else's implementation into the engine's parent architecture.

**Read this handoff in full before editing code.** Historic source-labeled takeover handoff (including all original research decisions and summaries) is preserved in `PRIOR_ART_HISTORICAL_MASTER_HANDOFF_V2.md` and is *not* the normative v3 design.

---

## 1. Precise user requirements and their evolution

The conversation began with a request for a new Rust-native runtime that could combine **general model support and serving** with **extreme hardware/model specialization**, particularly on AMD Strix Halo `gfx1151`. User targeted broad families including Llama, Gemma 4, Qwen3.5+, Qwen3.8-Flash-Next, GLM-5.3-Flash and later models, and wanted to learn from leading inference engines without maintaining incompatible model-specific runtimes.

The user explicitly endorsed **Semantic IR** as an architecture direction, subject to reality: it enables transformation and portability, not automatic performance superiority. They named the project **zerogravity**, requested exhaustive plans/handoffs, corrected canonical path to `/home/pvl/spaces/abird/src/zerogravity/`, and asked us to compare and combine an extensive existing source-analysis corpus with the new designs.

Subsequently the user demanded comprehensive fresh research on **long-term adaptability at every pipeline stage** (models, codegen, memory, scheduling, runtime, application, hardware) and incorporated new source demonstrations of (a) asymmetric compute/weight ownership across discrete and integrated accelerators and (b) **independent same-SoC GPU+NPU output-row GEMM cooperation** with strict hardware-safety constraints. They also requested speech, media and iterative AI pipelines to compose without using fake text-decoder loops.

Finally, **U-017** directs us to generalize all plans away from external project names unless those projects are MIT licensed; source research must remain cited and properly licensed, and the **architectural expression must be ours**. They want a composable AI system that can add new specializations *without writing new engines each time*.

User decision register `DECISION_REGISTER.md` is authoritative for exact historic IDs (U-001…U-017); design recommendations are P-xxx and unresolved product/technological decisions O-xxx. Do not infer that an old source project's license, API, compiler version, speed threshold or kernel language was user-approved.

## 2. Required reading and source-of-truth order

1. the agent playbook and **`DECISION_REGISTER.md`**.
2. This handoff, the rewritten **`ZEROGRAVITY_MASTER_PLAN_2026-10-10.md` v3**.
3. **`../docs/MODULAR_AI_SYSTEMS_REFERENCE_ARCHITECTURE.md`** and **`../docs/COMPOSITION_AND_EXTENSION_CONTRACTS.md`**.
4. `PRIOR_ART_LICENSE_AND_PROVENANCE_POLICY.md`: verified repository root license/NOTICE caveats and prior-art taxonomy; names/data used only in research, not core types.
5. `INTEGRATED_PHASE_0_AND_DELIVERY_GATES.md`, `IMPLEMENTATION_ROADMAP.md`, `EXPERIMENT_PROGRAM.md`, `RELEASE_ACCEPTANCE_GATES.md`.
6. Detailed subsystem specs for the current task in `../docs`, especially `SEMANTIC_IR_AND_COMPILER.md`, `ASYNC_RESOURCE_COMMAND_IR.md`, `ENGINE_CORE_EXECUTION_AND_PROTOCOL.md`, `STATE_CACHE_AND_TRANSACTIONS.md`, `MEMORY_RESIDENCY_AND_IO.md`, `DEVICE_KERNEL_AND_CODEGEN.md`, `PRODUCT_MODEL_PIPELINE_AND_EXTENSIONS.md`, `SECURITY_AND_RELIABILITY.md`, `INVARIANTS_AND_FAILURE_SEMANTICS.md` and `NUMERICS_QUALITY_AND_BENCHMARKS.md`.
7. For an external algorithm/benchmark, search the **research-only prior-art** files and source revision URLs or the read-only `research/inference` tree, not the normative architecture for an external project's type names.

**If the current checked code and measured tests conflict with a plan**, validate the live code/model and update the plan—never assert a phase completed based on a historic roadmap sentence. Do not automatically run `git status` in a folder with no Git; check existence first.

## 3. Original, composable system architecture

**One execution spine**:

```text
Source Manifest / Artifact / Tokenizer / Media Encoder
                       |
                ModelDefinition
                       |
             Semantic Model IR (SIR)
              typed math + state effects
                       |
             Optimized Region IR (RIR)
           legal fusion/packing/row splitting
                       |              \
                       |             Kernel Providers (KIR)
                       |              portable to extreme Rust code
                       |             /
             Command/Resource IR (CIR)
            async buffers/leases/IO/streams/events
                       |
             Compiled ExecutableBundle
                       |
    Topology/Device Capabilities + Physical Resource Arbiter
                       |
  Typed TaskProgram → TaskCoordinator/EngineCore → StepPlan
                       |
               Device/IO Workers
                       |
             Provisional state/events
                       |
      Atomic TaskSession transaction and result/stream
```

**Independent dimensions of capability composition**:
- **Model mathematics:** typed source operators, tensor/quant/position, recurrence/attention/router semantics, full reference oracle.
- **Task control:** text autoregression/speculation, encoder-only scores, ASR, TTS, image/video iterative stages, multimodal/agent workflows with different committed frontiers.
- **Device kernels/codegen:** portable CPU/GPU and optional hardware-specialized whole-region kernels; selection by op/shape/precision/safety/evidence, not model names.
- **Resource placement/IO:** immutable weights, expert slots, KV/recurrent/latent state, n-gram lookup rows, GPU/NPU pinned buffers, SSD/remote; all leased under one physical budget.
- **Runtime/orchestration:** TaskCoordinator owns sessions and scheduling; StepPlan/WorkerDelta are work commands, not whole model data; output only after commit.
- **Product/service:** canonical TaskRequest/TaskResult/ModeReceipt, independent SDK/HTTP/CLI/media output adapters, model store and permissioned external actions.

An optimized implementation may replace any **closed semantically equivalent execution region** with a direct Rust-authored fused device kernel or captured graph if full values, state effects, numeric precision and resource dependencies are validated. A missing optimized kernel uses portable/generic region execution or canonical state conversion/replay; a new kernel must never require a copy of the whole server. Unknown mathematical operators are not implicitly executable from a checkpoint format.

**Do not conflate** `KernelPlan` (validated candidates), `ExecutionPolicy` (device/resource/quality constraints), `ExecutableBundle` (compiled commands) and `StepPlan` (actual mutable scheduling cycle). One massive object binding all four would impair maintainability.

## 4. Why task composition is a core requirement

AI does not mean only autoregressive next-token models:
- Language generation has causal token frontier, KV/recurrent/indexer state, sampler and potential draft verifier.
- Embeddings and rerank models have batched encoder/pooling/score and no language decoder.
- ASR has acoustic sample/window/stream state, preprocessing, encoder and transcription.
- TTS has text conditioning, acoustic frames/tokens and vocoder PCM with speaker/voice state.
- Image/editing has conditioning and iterative diffusion/flow/denoiser latent steps before image output.
- Video/audio uses temporal/frame state and codec/output-mux behavior, potentially multiple model programs.
- Compound RAG and agent tasks compose permitted retrieval/ranking/generation/other TaskPrograms, with a security boundary around external tools.

Every task exposes typed `TaskProgram` stages (`Sequence`, `Parallel`, `Iterate`, `Select`, `Window`, `VerifyAndCommit`, `Emit` and optional permissioned `ExternalAction`). They share model Semantics/KIR/CIR/memory/DeviceWorker but may use different scheduler **work units and stream commit frontiers**. Implementing ASR/TTS/image/video is optional until a model/task quality matrix passes; the design supports them independently without pretending they already work.

## 5. Specialized parallelism without engine forks

- **Region fusion/layout:** normal/quant/projection, sparse selector+attention, recurrent update, routed expert gate/up/down, LM head/sampler, encoder/vocoder/denoiser as legal.
- **Routed expert ownership:** tensor identity/route top-k unchanged; hot vs cold experts computed on different GPU/CPU/IO owners; activations and partial results transferred under bounded leases and event ordering.
- **Intra-operator row partition:** independent projection outputs can run on GPU+NPU/other eligible devices while the GPU progresses on unrelated work; CIR first-use join and mathematical row/quant coverage are mandatory. NPU block-float arithmetic may change numerical results, so source numeric/quality policy applies independently.
- **Model/device/phase placement:** single CPU/GPU, partial offload, layer/tensor/expert parallel, prefill/decode split, future heterogeneous remote topology with authentication and state consistency.
- **Speculative target execution:** trained/native or independent draft and prefix lookup, exact target-distribution correction by default where implemented; full state/RNG/grammar rollback for rejected proposals.
- **Prefix reuse:** full hybrid-domain state snapshots across 3+ different attention/recurrent/indexer/media state types; one common legal resume frontier and immutable sharing/COW.

Every physical accelerator profile includes compute capability, memory/links and **safety/firmware/fabric**. A potential GPU+NPU whole-host hang is a hard qualification veto until isolated physical testing; no automatic root/sysfs/clock/BIOS/IOMMU changes. GPU-only reference remains usable. Driver runtime FFI and owner Rust AIE/kernel source are separate policy questions; wrapping foreign source is not automatically a Rust kernel.

## 6. Source formats, provenance and licenses

Safe GGUF/safetensors, tokenizer/chats/template/structured tools/grammar, content-addressed immutable artifact store, optional packer/quant/calibration with separate source and compute-precision identity, adapters, sidecars, model/vision/audio manifests. Format support requires verified loader **and** registered semantics, sufficient resources and task output support.

**Source naming and reuse policy:** no architecture type or phase depends on an external engine's project name, even if its MIT license permits study/reuse. Verified MIT source is a *candidate* to reuse after per-file header/NOTICE, model data/device executable and Rust-source policy checks. Non-MIT source retains name only as prior-art evidence unless explicitly authorized. Proprietary/nonpublic code cannot be copied. Licenses for model weights, trained derivative drafters, metrics/input datasets and GPU/NPU device programs are distinct from engine repository root licenses.

**Source evidence preserved** (do not erase): `PRIOR_ART_LICENSE_AND_PROVENANCE_POLICY.md` with verified license-text blob SHAs, `PRIOR_ART_HISTORICAL_MASTER_PLAN_V2.md`, `PRIOR_ART_HISTORICAL_ROADMAP_V2.md`, `PRIOR_ART_HISTORICAL_STAGE0_V2.md`, `PRIOR_ART_HISTORICAL_EXPERIMENTS_V2.md`, `PRIOR_ART_HISTORICAL_ACCEPTANCE_GATES_V2.md`, `PRIOR_ART_HISTORICAL_MASTER_HANDOFF_V2.md`, and dedicated source/hardware research analyses. External research tree stays read-only.

## 7. Physical memory, safety and async ownership

One `PhysicalResourceArbiter`, specialized subpools for immutable always-on weights, routed expert slots, paged or contiguous KV/state, frozen prefixes, lookup rows, NPU/GPU pinned staging, scratch, encoder/media and optional remote buffers. One global budget and OS/cgroup headroom, not a universal cache eviction policy.

Every async device/IO/peer operation reads/writes through a versioned BufferLease and timepoint, with explicit alias and ownership. A failed or timed-out DMA may still be accessing memory; quarantine until guaranteed quiescence or device reset. No reuse based only on a software timer. GPU addressability/unified virtual GTT is not free physical RAM.

One session owner commits all state effects atomically; device worker never publishes externally visible text/audio/image/video before TaskProgram's output validity rule. Cancellation leaves last complete emitted frontier and aborts uncommitted work. Distributed peer partial computations cannot masquerade as successful model state.

## 8. Correctness and performance expectations

Benchmark model/task-specific **semantics, numeric agreement, task quality and state transactions separately**. For source model checkpoint/quant/source hardware/driver and workload match:
- logs of teacher-forced layer activations/full-vocab logits and effects;
- target stochastic distribution and draft rollback where available;
- task-specific output quality (text, embedding/rerank scores, ASR audio alignment, TTS, image/video);
- physical resource headroom, cold/warm IO/expert cache and GPU/NPU event/overlap;
- TTFT, P50/P95 latency, committed useful outputs/sec (correct unit), energy and concurrency;
- compiler/driver/library/shape/quant/numeric plan provenance, actual ISA occupancy;
- ABBA/BAAB and raw uncertainty with a portable baseline and external peer when genuinely comparable.

**Never** claim universal speed lead based on architectural elegance, a microkernel, incomparable model quant, reduced trained MoE top-k, altered sampling, or unmatched power clocks. A complex scheduler/kernel candidate that loses whole-task latency or quality is disabled; negative experiments remain recorded.

## 9. Roadmap and change control

Stable project phases **P00–P23** in the rewritten `IMPLEMENTATION_ROADMAP.md`. Stage-0 S0-00…S0-14 plus optional NPU S0-NPU-A…F in `INTEGRATED_PHASE_0_AND_DELIVERY_GATES.md`; experiments H01–H31 in `EXPERIMENT_PROGRAM.md`; detailed release criteria in `RELEASE_ACCEPTANCE_GATES.md`.

**Start smallest full vertical slice:**
1. Lean Rust CPU build, trusted artifact parser and tokenizer/quant reference;
2. basic math/state SIR and CPU oracle, effect verifier and TaskProgram loop;
3. minimal RIR region/legal fallback, simple KIR kernel provider;
4. CIR buffer/event/IO simulator proving leases and aborts;
5. one real physical GPU device backend, direct compiled bundle and actual model result through TaskCoordinator;
6. one synthetic non-token task and second operator/device extension drill proving no duplicate engine;
7. only then advanced MoE/quant/sparse/heterogeneous/spec/graph capture/distributed and media tasks as requested/feasible.

NPU hardware safety/source-language feasibility is optional; no delay to general CPU/GPU release if blocked. No 25 empty crates or general-purpose optimizer before the vertical slice. User specifically asked for **plans** here, not code or benchmarks.

Each code phase completion must append actual changed files/test commands/driver/device/quant/source SHA, numerical/task quality, physical memory, P95 and known failures to `CURRENT_STATUS_AND_TRACKING.md`, and update decision register, relevant specs and checksum manifest. Never rewrite historical decisions to make an experiment seem accepted.

## 10. Open policies that require future evidence/user input

Project license and binary name, exact Rust GPU/NPU source-language/vendor math FFI boundaries, mandatory first device/model/task/quant release matrix, public SDK/HTTP transport and authentication, distributed trust default, C ABI/plugin model, deployment OS/CI hardware, default SLO and numerical tolerance, device clock/fabric safety qualification, proprietary device program rights, model adapter DSL choice and artifact/cache serialization upgrades. Detailed O-xxx entries in `DECISION_REGISTER.md`; don't fill silently.

No need to re-ask the user about project name/location, Rust-owned core, semantic IR, broad modern model families, integration of prior art, or modular composition goal—they are explicit already.

## 11. Agent startup prompt for direct reuse

```text
You are taking over the zerogravity Rust modular AI execution engine at
/home/pvl/spaces/abird/src/zerogravity.

Read in full:
- .agents/plans/AGENT_IMPLEMENTATION_PLAYBOOK.md
- .agents/plans/DECISION_REGISTER.md (especially U-017)
- .agents/plans/ZEROGRAVITY_MASTER_PLAN_2026-10-10.md (v3)
- .agents/docs/MODULAR_AI_SYSTEMS_REFERENCE_ARCHITECTURE.md
- .agents/docs/COMPOSITION_AND_EXTENSION_CONTRACTS.md
- .agents/prior-art/PRIOR_ART_LICENSE_AND_PROVENANCE_POLICY.md
- .agents/plans/INTEGRATED_PHASE_0_AND_DELIVERY_GATES.md
- .agents/plans/IMPLEMENTATION_ROADMAP.md
- .agents/plans/EXPERIMENT_PROGRAM.md
- .agents/plans/RELEASE_ACCEPTANCE_GATES.md
and all relevant state/kernel/CIR/task/resource/security/numeric subsystem specs.

Inspect the live project before relying on any completion status; at creation
there is no Rust implementation or Git repository. Treat the separate
/home/pvl/spaces/abird/research/inference directory as READ-ONLY prior art.

Architecture is ORIGINAL, project-neutral and capability-composable:
Artifact/ModelDefinition -> effect-aware Semantic IR -> legal Region IR ->
Rust KernelProviders -> async Command/Resource IR -> immutable direct
ExecutableBundle -> single TaskCoordinator/EngineCore + physical resource
arbiter -> typed TaskProgram/session transactions and streaming results.

Add a new model/operator/device/quant/media TaskProgram through its own
validated interface, never by creating another specialized inference server.
Preserve reference math, full mutable state, resource leases, numeric/sampling
quality and safe fallback. Names of non-MIT external projects appear only in
prior-art research/provenance, not architecture types; MIT code reuse still
needs file-level notice and Rust-source review.

If the current user request is design-only, modify docs/decisions/manifest
only. For implementation work start a tiny Stage-0 real model/task vertical
slice, test physical device+quality+memory and record actual evidence.
Never claim performance or a phase completed merely from documents.
```

## 12. Closure of this handoff

This is the complete operational context for any subsequent autonomous engineering agent. It identifies the original product goals, strict source/licensing distinction, model/task/semantic/kernel/resource/serving interfaces, prior research location, phase gates, physical memory and hardware safety, experiment methodology, pending decisions, and exact next-step discipline.

**Our defining promise is not "we copied the best engine"; it is "we can add the best verified implementation of any supported semantic region, on any qualified hardware, without duplicating the rest of the AI system."**