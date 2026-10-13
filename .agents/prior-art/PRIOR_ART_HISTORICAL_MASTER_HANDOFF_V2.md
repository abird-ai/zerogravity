# Prior-art archive — historical complete v2/v2.1 takeover handoff

**NON-NORMATIVE historic source and architecture provenance, archived 2026-10-12.** The prior original handoff names research projects, codepaths and source benchmarks; all are retained for traceability, but the current project-neutral architecture, live roadmap and next-agent prompt control design and implementation. Do not adopt its earlier source-project-labeled interfaces or benchmark parameters as user choices. See `MODULAR_AI_SYSTEMS_REFERENCE_ARCHITECTURE.md`, `COMPOSITION_AND_EXTENSION_CONTRACTS.md` and `PRIOR_ART_LICENSE_AND_PROVENANCE_POLICY.md` for the operative design and safe source usage rules.

---

# ZEROGRAVITY — LATEST TAKEOVER UPDATE (v2.1 · 2026-10-12)

**Canonical project root:** `/home/pvl/spaces/abird/src/zerogravity/`. **Scope of this update:** user explicitly requested architectural and execution-pipeline inclusion of [mikealanni/gufo](https://github.com/mikealanni/gufo) and the [Oct 11 Strix Halo NPU prefill report](https://www.reddit.com/r/StrixHalo/comments/1x3gtdh/qwen38_flashnext_prefill_past_1800_toks_on_strix/). Existing **upstream `gufo-org/gufo`** is a different source reference. This section supplements and takes precedence over the v2 historical takeover below if details conflict. No source from prior-art projects was copied unless directly vendored (for example, CubeCL (vendored)), which retains its upstream license; the Zerogravity engine is a from-scratch rebuild informed by prior-art concepts, learnings and research.

**Status:** all implementation phases P00–P23 are STILL **NOT_STARTED**; no functioning Rust engine, AIE program, XRT/NPU backend, model runs, performance or hardware safety validation exists. No Git repository was initialized. Nothing in these files means GPU+NPU is automatically enabled. The user's separate `research/inference` corpus is preserved read-only; modifications restricted to current `.agents/docs` and `.agents/plans`.

## Essential reading additions (must read before NPU/TaskProgram implementation)

- `GUFO_FORK_AND_XDNA2_PREFILL_SOURCE_RESEARCH_2026-10-12.md` — inspect exact fork and upstream; primary source at fork `fork-enhancements`, commit **`8e0fb1dedbc06c2b9d2ed6cc429cc394b840dee7`** (GitHub app); Reddit author report date **2026-10-11**. Paths include NPU GEMM source, XRT BO bridge, HIP staging/first-use join, AIE source and compiled program.
- `../docs/NPU_GPU_COOPERATIVE_EXECUTION_AND_TASK_PIPELINES.md` — **authoritative integration design** for projection-row compute ownership across RDNA GPU+XDNA2 NPU, CIR events, resource manager, Rust source policy, safety/fallback and TaskProgram media stage variants.
- `DECISION_REGISTER.md`: user decision **U-016**, working choices **P-026..030**, open **O-023..026**. Older user decisions U-001..U-015 preserved; no unapproved source language, license or device policy overridden.
- `ZEROGRAVITY_MASTER_PLAN_2026-10-10.md` **§17**: latest v2.1 architecture and all domains.
- `INTEGRATED_PHASE_0_AND_DELIVERY_GATES.md`: new **S0-NPU-A…F** optional track; doesn't block CPU/GPU vertical slice; `IMPLEMENTATION_ROADMAP.md`: phase mapping P01/P04/P05/P07/P10/P11/P13/P16/P17/P18/P20/P21/P22/P23.
- `EXPERIMENT_PROGRAM.md`: H26–H31 tests (source precision, Rust-owned XDNA2 codegen, cross-API event/BO safety, same-clock ABBA, whole-model quality, TaskProgram extension). `RELEASE_ACCEPTANCE_GATES.md` makes hardware safety and numerical qualification mandatory.

## Critical user-requested architecture — WHAT IS DIFFERENT

The specific `mikealanni/gufo` fork is **not** another ordinary Gufo benchmark nor a discrete-GPU+iGPU expert-ownership approach. It offloads pure projection **output row subsets** from `gfx1151` GPU to the **on-SoC XDNA2 NPU** while GPU computes independent rows/ops in parallel. Source-demonstrated cases:
1. Gated DeltaNet SSM input **z** rows;
2. Qwen Sparse Attention query/gate rows for selected heads;
3. shared expert gate/up GEMM (NOT routed experts themselves).

One fixed NPU GEMM candidate is **M=4096 prompt tokens, K=2560 input, N=1024 output** with **v8bfp16ebs8** (one exponent plus eight 8-bit mantissas per nine-byte group), AIE 32-core program and XRT long-lived hardware context. GPU encodes inputs, NPU consumes HIP-mapped pinned/XRT-imported BOs, C output joins at its *first actual consumer* through callback/event ordering. Its input is weight/activation slices of the *same model forward pass*, not a second NPU drafter or independent model.

Author reports on **128GB Strix Halo GMKtec EVO-X2, UD-IQ4_XS** at 16K prefill: **1698 GPU-only → 1821 GPU+NPU tok/s, +7.2%**, 32K NPU 1783, 40K cold served 1637; 30 greedy prompts identical is a limited smoke test, not full-logit parity or independent benchmark. Do not confuse with upstream Gufo UD-Q4_K_XL GPU-only source measurements, Qwen 27B DFlash2, or its DeepSeek/R9700 results.

**Main design:** no new Gufo model runtime. ModelRecipe/SIR stays exactly mathematical, RIR optionally creates `ProjectionRowSlice` with disjoint row maps and selected compute owners, KIR emits capable NPU/GPU provider kernels/encodings, CIR plans mapped BO leases, external XRT enqueue/wait, concurrent GPU progress and first-use joins, while **one EngineCore** commits all recurrent/attention/sampler state. `ResourceArbiter` accounts every pinned weight/input/output and clock/power domain capacity. Missing NPU, unsupported shapes, incompatible precision or unqualified hardware => **GPU-only fallback**. XRT midflight failure aborts provisional transaction and quarantines DMA buffers.

**Safety is a hard veto for automatic NPU activation:** the Reddit poster/fork author reports **hard system hangs/machine checks** when GPU fabric frequency changes while NPU is working. Holding fabric clock at max, NPU power mode and suspend/keepalive changes were the author's environment-specific workaround; 12 clean runs do NOT establish general reliability and pinned clocks can raise idle power. **Never automatically run sudo/sysfs/fabric-clock/systemd/IOMMU/firmware tuning.** An independent GPU-only path is mandatory; NPU stays OFF unless firmware/AGESA/ROCm/XRT/power/driver/compiled program is controlled, source-policy compatible and physically qualified on isolated hardware. A software watchdog cannot recover a frozen SoC.

**Rust-only caveat:** fork's AIE kernel and compiler generator are **C++ plus Python MLIR-AIE**, hosted by C++ XRT, with compiled `.xclbin` and instruction binaries. A Rust FFI wrapper does not satisfy the stated Rust-authored custom kernel requirement. S0-NPU-B must prove genuine Rust-owned AIE/MLIR kernel codegen or mark feature blocked without taking over Gufo source. Driver/runtime FFI remains an explicit user policy question, and NPU build is optional.

## Other upstream Gufo pipelines (future model/task categories)

The **upstream gufo-org/gufo** repository now includes model guides and runnable work for Qwen3.8-27B with DFlash2 and vision, Qwen3.8 Flash-Next MTP, DeepSeek V4 DSpark, Qwen3-ASR, Qwen3-TTS; Qwen-Image-2.1 and MiniMax H3 text/video/audio pipelines are in progress. These **do not mean zerogravity supports any one of them**. Implement task-specific stage/loop contracts:
- LM decode / draft verifier: causal token frontier;
- ASR: audio input encoder/chunks and transcript;
- TTS: acoustic generation + vocoder/media stream;
- image edit/generate: conditioning + repeated denoise/flow steps;
- video/audio: frame/temporal-stage and media output contract;
- embeddings/rerank: encoder+pool/score without LM KV.

All share ModelStore, exact SIR operator reference, RIR/KIR candidate backend, CIR state/async resources and EngineCore/DeviceWorkers, but **TaskProgram** owns stage control, iteration frontiers and output type so no media pipeline is shoehorned into next-token sampling. A new NPU encoder offload is *unproven* until actual kernel/quality/stability experiments.

## Handoff/final safety instruction

For **design-only** follow-up, edit docs/decisions/manifest only. For **authorized implementation**, start CPU reference+generic GPU and small RIR/CIR fake device slice test; NPU fork offload remains a separate experimental S0-NPU-A…F and can be blocked without blocking portable inference. Never announce "seamless NPU enabled" before real Rust AIE codegen, cross-XRT/HIP DMA correctness, matched full logits, controlled same-clock/energy A/B and safe repeated physical hardware qualification. Last accepted model semantics, numerical precision, sampling distribution and durable state always outrank peak t/s.

---

## PREVIOUS UNIFIED v2 TAKEOVER (preserved, subordinate to the latest update)

### Previous title: ZEROGRAVITY — CURRENT COMPREHENSIVE MASTER TAKEOVER (Integrated v2 · 2026-10-10)

**Authoritative root**: `/home/pvl/spaces/abird/src/zerogravity/`. Evidence and pinned source analysis: `/home/pvl/spaces/abird/research/inference/` (**read-only during this merger**). Tool: dotlink l5. **Status:** complete v2 architecture/research **documentation only**; no Rust engine/source implementation, Cargo workspace, Git repo, GPU benchmark, model downloads or release was created. Every ZG-P00…P23 implementation phase is NOT_STARTED.

**This section supersedes earlier v1 handoff text wherever inconsistent.** The earlier handoff remains below under a historical heading to preserve the original user decisions and research discussion. Latest user decisions U-014/U-015 explicitly request merging the best of both plans and expanding research for the strongest possible future-proof inference design, while preserving U-001…U-013 (project Rust/name/models/Strix/canonical path).

## Read-first authority

1. the agent playbook and `DECISION_REGISTER.md`: confirmed user decisions versus assistant/research working choices and open questions; **research council "RESOLVED" values are not automatically approved by user**.
2. `ZEROGRAVITY_MASTER_PLAN_2026-10-10.md`: **rewritten v2** complete architecture.
3. `ARCHITECTURE_RECONCILIATION_AND_AUTHORITY.md`: explicit D-01…D-20 ruling and research defects.
4. `COMPARATIVE_COMPILER_AND_RUNTIME_RESEARCH_2026-10-10.md`: source-backed comprehensive new research across compiler/serving/IO/kernel/model evolution.
5. `../docs/SEMANTIC_IR_AND_COMPILER.md`, `../docs/ASYNC_RESOURCE_COMMAND_IR.md`, `../docs/ENGINE_CORE_EXECUTION_AND_PROTOCOL.md`: runtime/IR foundations.
6. `../docs/PRODUCT_MODEL_PIPELINE_AND_EXTENSIONS.md`: GGUF/safetensors/quant/tokenizer/grammar/model store/LoRA/embedding/multimodal.
7. `INTEGRATED_PHASE_0_AND_DELIVERY_GATES.md`, `IMPLEMENTATION_ROADMAP.md`, `EXPERIMENT_PROGRAM.md`, `RELEASE_ACCEPTANCE_GATES.md`: real dependency, 24 phases, experimental/quality gates.
8. All other subsystem docs including state/KV/cache/MoE/scheduling/kernel/Strix/distributed/numerics/security/Rust engineering, status and refreshed SHA-256 manifest.
9. Read the relevant pinned-revision manifest, synthesis/innovation research and platform/machine docs; verify latest upstream before coding. **Do not modify research corpus in this task.**

## Current architecture (mandatory conceptual distinction)

```text
ModelStore / GGUF/safetensors / tokenizers + template + mm / adapters
    -> verified ArchDef/ModelRecipe -> Semantic Model IR (SIR)
       mathematical model operator graph + explicit SSA state effects
    -> legal optimized Region IR (RIR): fusion/layout/precision/shape
                          \
                           Kernel IR/compiler providers (KIR):
                           Rust CPU + CubeCL + target Rust HIP +
                           NVIDIA Rust tile / SIMT-PTX compiler candidates
                          /
    -> Async Command/Resource IR (CIR):
       Buffers+leases+timepoints+GPU streams+SSD+peer+commit edges
    -> immutable ExecutableBundle: direct kernels/captured regions
                      |
HardwareProbe -> pure HardwarePolicy + ResourceArbiter (physical budget)
                      |
Edge/API -> EngineCore(single logical session owner) -> StepPlan
           -> WorkerBackend GPU/CPU direct launch -> WorkerDelta
           -> target verification + exact all-state frontier commit
           -> streaming committed output / ModeReceipt / metrics
```

**A kernel plan is not a semantic model**: `KernelPlan` stores tested candidate implementations and provenance; `HardwarePolicy` sets safe resident/SSD/KV placement; `ExecutableBundle` holds compiled code+resource graph; `StepPlan` assigns real request rows to handles. Do not create a giant object called ExecutionPlan that conflates these. Model-specific `ModelSpec::kernel_plan(hw)` from historical research is superseded by model recipe with exact math plus compiler `KernelProvider` target selection.

**RIR/KIR/CIR build-time only**, no per-token IR graph interpreter. A specialized whole-layer/whole-model Rust GPU kernel may replace any semantically closed region if quality, state-effect, device capacity/alias and resource contracts validate. If a fast candidate unavailable, fall back to generic/ref region at a *compatible state/layout boundary*, or replay from canonical prefix if conversion required. Unknown new model op still requires a reference implementation; file format alone doesn't grant arbitrary model coverage.

## Best of the research preserved

- **vLLM/SGLang serving**: token-budget scheduling, paged KV, chained prefix hash, optional SGLang radix index over same allocation pool, async prefill/decode/verify, latency priority and per-user p95.
- **Detailed native engine**: single owner Rust `EngineCore`, bounded `EngineMsg/EngineEvent` and worker `StepPlan/WorkerDelta`, persistent `InputBatch` deltas, no global per-token lock; logical owner and device workers distinct.
- **KV choice by workload**: paged generic and contiguous extreme reserve-all, both under one physical resource arbiter; v1 design can choose only proven subset. Generalized full hybrid state-group prefix coordinator works across 3+ types, not only ordinary KV.
- **Maya/Strata/DwarfStar giant MoE**: SSD/RAM/UMA expert tiers, fixed safe slots, router device pointer table, asynchronous bounded miss request, pinned staging, hotlists/recency/LFU, CPU lane calibrated; n-gram/Engram huge table reads are separate tiny row-lookup workload.
- **Gufo performance path**: model-specific gfx1151 Qwen GDN/QSA/MTP and grouped low-bit MoE; GLM KDA/DSA/MLA/mHC and per-layer expert caches; attention selection and draft rollback must stay device native where feasible.
- **DwarfStar frontier mechanics**: `CommitPlan{FullDirect,Prefix(n),Replay(n)}`, frozen boundary snapshots, trained DSpark/MTP verification; wrapped inside full SIR state transaction (KV/GDN/KDA/indexer/token history/grammar/RNG/modality).
- **llama.cpp product**: GGUF parser, CPU SIMD, quant/dialect, tokenizer pretype, Jinja model prompt, grammar (a grammar library/GBNF), tool parser and easy single-binary CLI.
- **Ollama model store+calibration**: content-addressed SHA256 blobs, manifest list/GC/sidecars, offline download, optional acceleration pack and imatrix/quality overlays; adapters LoRA, embedding/rerank PoolingSpec and multimodal separate encoder scheduler.
- **2026 compiler innovation**: an ahead-of-time ML compiler's Flow/Stream/HAL explicit async resource, a tensor compiler's Relax/TIR/MetaSchedule, MLIR side effects, bounded graph-search, 2026 kernel-dispatch eager/graph aware tuning; NVIDIA Rust tile/SIMT-PTX compilers and CubeCL alternatives. These are research evidence, not bundled dependencies.
- **Quality and operations**: independent official/CPU numeric oracle, ABBA where exact applicable, target distribution correctness, bounded memory, source pinned experiments, compiler+model critical path profiling, source licenses/security and phase gates.

## Original programs' faults explicitly corrected

1. Old research "any model any runtime runs" is overbroad: **only supported model semantics and backend/resources** may run, otherwise typed refusal.
2. Disk-KV research calls header 48 bytes while specified fields total **54**; five-character magic erroneously `u16`. **New versioned binary schema required**, no original copy.
3. DwarfStar 8×L40S DeepSeek multi-session benchmark appears in a GLM table: not a matched GLM performance baseline.
4. Old research 6GB RTX 3060/Cezanne 27GB machine and purchase of 128GB Strix are historical assumptions: probe current reachable devices; user hardware info cannot be inferred reliably from old research.
5. `cubecl=0.11`, exact Rust MSRV, `zg`, dual MIT/Apache-2.0 and C++/HIP vendor escape are research council preferences; **remain open** until measured or user approved.
6. Research "two concurrent requests = native batch" and "frontier rewind O(1)" need implementation caveats; device grouped rows and physical state rollback measured.
7. Prior project 3-layer SIR/EIR/DPI omitted explicit async resource/command optimizer, and product lacked grammar/ModelStore/LoRA/embedding/rerank detail.
8. 93 engineer-days research Phase0 cannot honestly be equated to 4–6 weeks for two engineers without major scope cuts or added staffing. Stage0 now splits hard gate vs optional.
9. Historical source repositories/performance claims are dated and from authors, not zerogravity benchmarks. No source-license-free copying of specialized engine code.

## New Stage 0 and phases

**Mandatory earliest evidence** from `INTEGRATED_PHASE_0_AND_DELIVERY_GATES.md`:
- S0-00 reproducible lean Rust workspace/source pins/decision log.
- S0-01 true gfx1151 Rust kernel ISA/quality and performance feasibility; do not infer all GPU support from CubeCL README.
- S0-03 GGUF/tokenizer + CPU quant/attention math oracle, first real dense model; optional CUDA Rust tile compiler / Rust SIMT-PTX compiler as second target.
- S0-04 minimal SIR explicit KV+GDN effects (full indexer soon).
- S0-05 RIR legal fusion + KIR kernel candidate and missing-specialization safe fallback.
- S0-06 CIR lease/timepoint/IO/error simulation, no alias/early release.
- S0-07 EngineCore/StepPlan/WorkerDelta and paged KV/frontier simulator.
- S0-08 speculative all-state/RNG rollback and reference quality.
- **S0-12** complete artifact → SIR→RIR→KIR/CIR→CPU & GPU direct executable→tokens through same EngineCore, with fallback disabled kernel tested.

Other experiments (grammar/embedding/LoRA/MTP/various GPUs and long-term async cache innovation) are independent tracks. **No need to implement a 25-crate scaffold or general search optimizer to complete the vertical slice**. All implementation phases ZG-P00…ZG-P23 remain NOT_STARTED now; this document update does not count as actual implementation.

Release goals G1 portable runner, G2 serving, G3 extreme Qwen/GLM/DeepSeek, G4 breadth/parallel all need real source-matched quality/latency/performance evidence. Huge model fit requires OS headroom and resource capacity; do not trust ROCm GTT as free RAM.

## Existing user decisions and open policies

Consult decision IDs **U-001…U-015**. User approved consolidation and comprehensive research, **not** specific libraries, license, vendor binding, benchmark percentages, release backend sequence or buying hardware. The reconciliation proposals are P-018…P-025. Open O-001…O-022 include FFI and non-Rust kernel source, compiler selection, product executable/license, v1 model/quant and OS matrix, disk snapshot schema, user-visible CLI and external plugin trust.

## Extension/maintainability acceptance

For every new model semantic op: official source/checkpoint geometry and tokenizer, Rust CPU independent oracle, typed operator/effect/snapshot/replay, generic provider, optional tuned KIR, full logits/quality at long context and forced fallback. Adding op shouldn't change HTTP/ModelStore/EngineCore except genuinely novel capability. New GPU should add DeviceCapabilities + KernelProvider with independent benchmarks, not alter every model recipe. New I/O tier extends CIR resources and ResourceArbiter, not model math.

**Release claim rule:** only matched model/checkpoint/quant/context/sampler/concurrency/device/driver/power, TTFT/p95 and benchmark repeats can establish superiority. Negative results and unsupported cells must be published. Every compiled optimization remains optional and can be disabled when worse.

## Next-agent operating instructions

1. Use the current live canonical root; dotlink l5 launch root `/home/pvl/spaces/abird`. The research corpus is read-only reference.
2. Read master+decision+reconciliation+research+all pertinent subsystem docs; verify actual files and Git if initialized (was absent at consolidation).
3. Obey **current user's requested scope**. For design request edit docs only. For implementation take one small Stage0/phase subset, don't decide unapproved policy choices silently.
4. Inspect exact source pins/hardware and independent official model reference. Preserve SIR effects and physical resource/event safety.
5. Test and record source SHA, model/quant/hardware, kernel generated ISA, numerical parity, prefix/spec rollback, quality, measured p95 and full-model throughput; mark incomplete GPU/model tests SKIPPED/BLOCKED honestly.
6. Update roadmap/decision/status/handoff/manifest on completion. Never claim faster than any engine without a controlled comparison.
7. No closed-source engine copying, no implicit target precision change, no arbitrary HF remote code execution, no unsafe memory overcommit.

**Current completion:** comprehensive research/plan merger; no engine code/release. The following historical snapshot is retained as provenance and is subordinate to this current v2 section.

---

## HISTORICAL INITIAL HANDOFF SNAPSHOT (v1, preserved for context)

### Prior title: ZEROGRAVITY — FULL MASTER HANDOFF AND TAKEOVER AUTHORITY (2026-10-10)

> **Purpose:** external LLMs and engineering agents must be able to take over zerogravity from disk alone, without the originating ChatGPT conversation. This is a **design/research handoff**, not a code completion report. Read the master plan and subsystem specs; this document makes the objectives, user decisions, research and next-action boundaries self-contained.

Project: **zerogravity**  
Creation date: **Saturday, 2026-10-10 (Asia/Singapore)**  
Working directory observed through **dotlink l5**: `/home/pvl/spaces/abird`  
Project directory (user-corrected canonical location): `/home/pvl/spaces/abird/src/zerogravity`  
Documentation: `zerogravity/.agents/docs`, `zerogravity/.agents/plans`.  
**Implementation state at handoff:** design/archive only. No Cargo workspace, Rust engine, GPU kernels, benchmarks, downloaded model weights, CI, Git repository, or releases were created in this request. Never mistake external source-project performance measurements for zerogravity results.

---

## 1. What the user explicitly wants

The user wants to build one entirely Rust-authored inference engine called **zerogravity** that runs:
- common models (Llama, Gemma 4, Qwen 3.5+ dense/MoE/hybrid and future compatible architecture families);
- exceptionally large modern models, especially **Qwen3.8-Flash-Next (~125B language backbone, auxiliary n-gram lookup/MTP)** and **GLM-5.3-Flash (~321B total)**;
- common CPU/GPU hardware as a normal portable engine, but **aggressively specializes for eligible hardware**, especially **AMD Ryzen AI Max+ 395 / Radeon 8060S / Strix Halo `gfx1151`**, to compete with highly tuned engines;
- high server concurrency/throughput like vLLM/SGLang; model support and usability like llama.cpp; model/hardware specialization like Gufo; expert memory tiering like Maya/Strata/DwarfStar;
- seamless correctness-preserving **fallback to a general model runtime** when a specialized kernel/whole region is absent, rather than refusing all other models or duplicating runtimes.

The user specifically requested a **Semantic IR-based architecture**, supported the idea of pushing it beyond existing inference runtimes when possible, and demanded that the docs preserve every design decision/unknown for external agents. They asked for research, then a refined design, then a comprehensive persistent documentation/handoff archive through **dotlink l5**. They have **not** asked in this turn for Rust implementation, repo initialization or model benchmarking.

User decisions are verbatim-intent summarized with IDs **U-001..U-012** in `DECISION_REGISTER.md`. **Do not upgrade any assistant proposal into an explicit user choice.**

### Key user questions that shaped the project

1. Research vLLM, SGLang, llama.cpp, specialized gfx1151 engines, two Strix Halo Reddit experiments (Maya GLM and Qwen Flash-Next) and additional optimized engines; first goal research only.
2. Include `https://github.com/antirez/ds4` (DwarfStar) in architectural research.
3. Combine the best concepts to build a **refined architecture reaching each model's peak possible performance**, potentially exceeding specialized projects, while **retaining a general runtime fallback**.
4. Ask whether Semantic IR gives an ability to outclass existing runtimes; answer was **credible architectural opportunity, not speed guarantee**. The key is mathematical/state definitions compiled to hardware-specialized direct GPU plans, not runtime IR interpretation.
5. Name project **zerogravity** and write exhaustive authoritative plans/docs/handoff at the initially requested relative `./zerogravity/.agents/{docs,plans}` via `dotlink l5`.
6. **Later explicit user correction (U-013):** move the complete project to `/home/pvl/spaces/abird/src/zerogravity/`, update documentation paths and revalidate. This corrects placement only, preserving all technical decisions.

This chronology matters: no claim that source research proved performance, and no implementation phase has been approved/started by this request.

---

## 2. Authority and reading order

1. `../plans/AGENT_IMPLEMENTATION_PLAYBOOK.md` — navigation/agent contract.
2. `../plans/DECISION_REGISTER.md` — user's choices versus assistant proposals/open decisions.
3. `../plans/ZEROGRAVITY_MASTER_PLAN_2026-10-10.md` — complete system charter and architecture.
4. This handoff — history, decisions and takeover steps.
5. `../plans/IMPLEMENTATION_ROADMAP.md` — 24 phases P00–P23 with gates and dependency order.
6. `../plans/EXPERIMENT_PROGRAM.md` and `RELEASE_ACCEPTANCE_GATES.md` — methodology/closure conditions.
7. The original per-file SHA-256 integrity manifest — to be refreshed after deliberate modifications.
7. All `docs/` specs that relate to the area of work, plus always `INVARIANTS_AND_FAILURE_SEMANTICS.md`, `NUMERICS_QUALITY_AND_BENCHMARKS.md` and `SECURITY_AND_RELIABILITY.md`.

**Live source beats historical prose:** before claiming any progress on future turns, inspect the actual files and Git state if repo exists. No project Git repository had been initialized at initial documentation creation or at directory relocation. Source upstream GitHub heads change; pin specific revisions before importing/benchmarking.

### Subsystem docs and what they must answer

- `VISION_SCOPE_AND_PRINCIPLES.md`: supported product experiences, goals/non-goals, layering, generality.
- `RESEARCH_SOURCE_ATLAS.md`: all studied upstream projects, concrete repo/source paths, author benchmarks, limitations and license caveats.
- `MODEL_SEMANTICS_AND_COMPATIBILITY.md`: model architectures, operator registry, adapter protocol, status matrix.
- `OPERATOR_AND_FUSION_CATALOG.md`: operator-level math/effects/quant/shape, region fusion and safe fallback.
- `SEMANTIC_IR_AND_COMPILER.md`: typed state-effect SSA, SIR→EIR→DPI, verifier, lowering, plan cache and legalization.
- `COST_MODEL_AND_AUTOTUNING.md`: measured GPU compute/memory/IO/queue cost, plan variants, runtime adaptation and hypothesis testing.
- `DEVICE_KERNEL_AND_CODEGEN.md`: Rust GPU compiler decision/FFI, kernel families, `gfx1151`, CUDA/Metal/Vulkan/CPU, graph capture.
- `STATE_CACHE_AND_TRANSACTIONS.md`: KV/GDN/KDA/indexer/lookup state, immutable prefixes, COW, speculative commit, disk snapshots.
- `MEMORY_RESIDENCY_AND_IO.md`: physical RAM/GTT, expert slots/leases, fixed slabs, RAM/SSD tiers, disk-only n-gram and scratch.
- `SCHEDULER_BATCHING_AND_SPECULATION.md`: unified token budget, prefill/decode/verify batching, GPU MoE grouping, draft policies and exact sampling.
- `DISTRIBUTED_PARALLELISM.md`: tensor/expert/layer/P-D parallelism, topology, handshake, transaction, fault recovery.
- `FORMATS_QUANTIZATION_AND_LOADING.md`: GGUF/HF safetensors, tokenizer/template, quant provenance, safe parsing, lossless packing.
- `RUNTIME_SERVING_AND_UX.md`: CLI/SDK/server APIs, streaming, tools, introspection, metrics and resource clarity.
- `NUMERICS_QUALITY_AND_BENCHMARKS.md`: reference/quant/quality differences, real benchmark protocols, statistics and tests.
- `RUST_WORKSPACE_AND_ENGINEERING.md`: Cargo workspace and Rust source/FFI, NixOS/toolchains, CI, safety/contributor rules.
- `SECURITY_AND_RELIABILITY.md`: trust boundaries, GPU memory, untrusted model files, HTTP auth, private snapshots, distributed failures.
- `STRIX_HALO_OPTIMIZATION_DOSSIER.md`: Qwen/GLM/DeepSeek on `gfx1151`, memory tiers and kernel experiments.
- `PERFORMANCE_HYPOTHESES_AND_RISKS.md`: why it could be better, why it could fail, alternatives and kill criteria.
- `EXAMPLE_EXECUTION_TRACES.md`: end-to-end traces of fallback, Qwen resident, GLM streamed, coding agents, remote failure.

The documents intentionally overlap where necessary so individual agents can work from a focused spec without losing critical correctness or context. In a conflict, user decisions and independent tested mathematical source override a prose recommendation; register the conflict in a new decision record.

---

## 3. Research history — each project's actual contribution

### vLLM (https://github.com/vllm-project/vllm)
Token-budget continuous batching V1 scheduler, paged KV, prefix reuse, optimized attention and GPU graphs, distributed parallelism. Understand scheduling from computed tokens and available memory. Its strongest quality is high-throughput service architecture. **Do not copy Python/Torch dependency architecture** just to gain V1 scheduling ideas.

### SGLang (https://github.com/sgl-project/sglang)
RadixAttention/prefix reuse, HiCache GPU/RAM/storage, overlap scheduling, speculation (MTP/EAGLE/DFlash etc.), expert parallel and P/D disaggregation. Key: scheduling must know cached work and overlap CPU/GPU/IO, but HiCache stores **context state** whereas Maya stores **model expert weights**. These require separate caches sharing one budget.

### llama.cpp / ggml (https://github.com/ggml-org/llama.cpp)
GGUF, CPU/GPU native graph model adapters, many quant types, partial offload/mmap, simple CLI/server and broad device support. Key: portability/compatibility and model assets. Pitfall: generic graph/kernel gaps on new GDN/DSA/MoE shapes and CPU fallback for some GPU operations can hurt long-context speed. The new engine should be easy to run, not depend on llama.cpp as a hidden runtime.

### Specialized gfx1151 engine
Specialized Qwen3.8-Flash-Next `gfx1151` engine with deeply tuned kernels, quant precision split, resident core weights and paged n-gram table, trained speculation. Reference author reports ~1584 prefill tokens/s 8K, ~1517 at 131K, ~37.6 short serial generation and ~55–56 coding speculation on 128GB Strix (different versions/conditions). Native checkpoint ~62.1 GiB plus ~47.7 GiB paged n-gram. **Core engine source not published; EULA limits source reuse.** Treat benchmarks and public descriptions as independent behavior research. Cannot copy closed source.

### Gufo (https://github.com/gufo-org/gufo)
Inspectable ROCm `gfx1151` Qwen3.8 Flash-Next extreme kernels and GPU session state/rollback, long-context sparse attention, native batching and transparent rejected experiments. Published Oct 8–9 2026 single-user UD-Q4_K_XL AR ~1700.52 prefill/~25.95 decode at depth 0 and ~1478.66/~24.92 at 128K; repetitive MTP ~60.39 while mixed MTP ~31.32; C8 AR aggregate ~108.68. One warmed sample/point, specific comparator llama.cpp revisions. Deep value: accurate per-session GDN/indexer/MTP replay and benchmarking. **Not apples-to-apples versus that engine's quant and model layout.**

### Project Maya (https://github.com/mw00/project-maya)
GLM-5.3-Flash specialized 321B (18B active) with `src/core/glm_fast_path.cu`, GPU device expert pointer table, host-mapped miss ring, pinned DMA staging, cache slots protected per-layer/in-flight, CPU lane, ROCm `gfx1151` experimental. GLM has 45 layers, 34 KDA+11 sparse MLA/DSA, 42 routed layers × 288 = 12,096 expert blobs, top8, 4-stream mHC and trained MTP. Maya-S quant ~96.5 GiB; Strix 128GB author reports ~210–217 prefill, ~17.8 generation and ~99.9% expert cache hits when pool holds virtually all experts. Two discrete AMD GPU+MTP result ~33 tok/s different hardware/acceptance. Core transferable insight: **cache migration and physical memory topology** are first-class execution decisions. Maya service is not vLLM-equivalent high-concurrency native batching.

### Strata (https://github.com/mach10x/strata)
Prior Qwen3.8-Flash-Next (qwen4exp) specialized giant MoE with GPU/RAM/NVMe expert tier, GPU/CPU overlap and n-gram lookups. Distinguish large lookup rows vs expert matrix blobs. Learn placement policies, pipeline/layer split and speculation. Not portable model runtime.

### DwarfStar / antirez/ds4 (https://github.com/antirez/ds4)
Native C/ObjC/CUDA/HIP model-specific runtime, permissive MIT source with GGML attribution. Inspect `ds4.h`, `ds4_rocm_compat.cu`, `rocm/ds4_rocm_runtime.cuh`, `ds4_ssd.c`, `ds4_engram.c`, `ds4_tp.h`, docs/QA. Visible GitHub connector head `fc80bd695da76ee14bcb86820bf900a5cdfda806` dated 2026-10-07; reverify actual checkout. Model focus DeepSeek V4/V4.1, GLM 5.2/5.3/5.3 Flash, Qwen3.8 Flash-Next. Compressed V4 attention uses 128-token raw window with ratio4/indexed and ratio128 compressed older history. Disk-only Qwen n-gram ~95.37 GiB with Q2 main/MTP ~41.73 GiB, V4.1 Engram ~189 GiB. Dynamic SSD expert slabs, safe memory guards, DSpark (5 candidate tokens), true/exact sampling option, TP RDMA and layer PP, agent/HTTP/integration QA. Strix DeepSeek Q2 author measured prefill 187.26→292.86 after tuned IQ2/rocWMMA kernels. **Important limitations:** Qwen3.8 path Metal/CUDA but not ROCm in inspected docs; ROCm decode-island graph capture stub returns unsupported in `ds4_rocm_compat.cu`; certain concurrency is ordered fallback, not native batching. Learn both the strengths and the limits.

### Others
- a CPU/GPU heterogeneous engine: x86 AVX-512 expert computation + GPUs and layerwise prefill, high-precision massive checkpoints.
- a quant-aware inference project: quantized expert decode GEMV vs grouped prefill GEMM by routing shape.
- a kernel-dispatch library: attention/MLA/MoE kernel dispatch and autotuning patterns.
- a vendor graph compiler: AutoDeploy/model graph transformation and device execution compiler.
- a unified-memory tensor framework: unified-memory semantics; Apple and AMD hardware not identical.
- An independent Rust serving project: broad Rust serving with a Rust tensor library, competitive dense results and some large MoE gaps, proving Rust language alone doesn't solve kernel quality.
- A Rust tensor library: Rust tensors/model reference.
- CubeCL: Rust-authored GPU source and codegen; alpha/evolving, `gfx1151` compiler performance must be proven.

## 4. Hardware/model facts to carry into planning

| Model | Research shape | State/weight distinctions | Must test |
|---|---|---|---|
| Llama dense | GQA/RoPE/FFN decoder | ordinary paged KV | CPU/GPU parity, launch/throughput |
| Gemma 4 | sliding/global and MoE for 26B-A4B | different attention windows and 128 top8+shared experts in cited variant | template/precision/attention and grouped MoE |
| Qwen 3.5+ | hybrid GDN/full attention, optionally MoE | recurrent state + KV | chunk parity, state snapshots |
| Qwen3.8 Flash-Next | 48 layers, GDN/QSA 36+12, 512 top10, MTP, large ngram | resident compute weights vs disk lookup | `gfx1151` deep attention, MTP, memory, quant quality |
| GLM5.3 Flash | 45 layers, 34 KDA+11 sparse MLA, 288 top8 on 42 MoE layers, mHC | massive expert pool and recurrent/latent state | 92 vs 128 GB memory, cold/warm cache, KDA/DSA |
| DeepSeek V4 Flash | compressed time-axis attention + MoE, DSpark | 128 local window, 4:1 vs 128:1 historical | correct compression/indexing, spec economics |

The user targets older and newer family variants too; exact geometry MUST come from the pinned model config/checkpoint, not this coarse model-family table. `Qwen 3.5+` is not synonym for `Qwen3.8-Flash-Next`.

Physical device support is **model/quant/operation dependent**. A `gfx1151` with 92/96GB is not guaranteed to fit a 128GB reference model profile; model dense/KV/KDA/scratch/draft plus OS headroom matters. GPU-visible GTT does not equal available RAM.

## 5. The architecture — authoritative synthesis

```
frontends: CLI / API / Rust SDK / agents
    -> artifact loader / tokenizer & template
    -> MODEL SEMANTIC IR (typed math + explicit state transitions)
    -> OPTIMIZED EXECUTION IR (fusion/shape/layout/quant/placement)
    -> DEVICE PLAN (kernels/streams/graph/events/leases)
    -> runtime scheduler: token budget + prefill/decode/verify
                |                |
       prefix-state cache    global ResourceArbiter
                |            / | experts, KV, lookup IO, scratch
                v           /
          model/device executor (CPU, ROCm, CUDA, Metal, Vulkan)
          + online measurements, offline tuning, numerical oracles
```

### How a generic model runs
Known adapter maps checkpoint to typed semantic operators; portable CPU reference exists, and generic GPU/backend lowers all supported ops where available. Runtime plans and executes direct kernels without interpreting semantic graph per token. Unsupported novel op gives explicit error. Availability of GGUF alone cannot guarantee model execution.

### How an extreme model runs
Compiler recognizes exact legal model operator regions, quant, shape and architecture (e.g. Qwen GDN/QSA/MoE on `gfx1151`), resolves specialized Rust-authored kernels, preallocates leases/scratch and captures graphs where viable. Scheduler chooses a plan based on batch, context, memory and numeric policy. Resident GPU expert routing avoids host round-trips; disk misses enter safe async handoff. Non-specialized regions use valid fallback at canonical tensor/state cut points. State correctness survives switching.

### Why semantic IR can help but does not guarantee speed
Semantic IR exposes optimization legality, shared model definition and backend independence. Peak performance still requires exceptional hardware kernels, correctly priced memory/IO, efficient scheduling, reduced launch overhead, controlled fusion and robust compilation. A handwritten specialized/Gufo kernel can outrun generic IR output. The architecture **must permit handwritten Rust target kernels/whole-layer graphs** without any abstraction performance ceiling.

## 6. Key subsystem decisions and invariants

### Semantic IR
Three levels SIR→EIR→DPI. SIR includes SSA tensor data **and ordered effects** for KV/GDN/KDA/indexer/history/RNG. Compiler reorders pure ops only when state/effects and numerical policy permit. Device plans directly execute precompiled kernels; no per-token interpreted IR. Repacking quant layout differs from requantizing model.

### State and prefix
Commit only a consistent token frontier across all mutable state. A speculative rejected token cannot leak changes to KV/SSM/indexer/lookup/sampler. Prefix snapshots must include all model-specific state, not just KV. COW/frozen immutable prefixes shared among many agents, mutable tails private. Disk snapshot schema versioned with exact model/quant/position/renderer/bytes and checksums. State fallback may require explicit conversion/replay.

### Memory
One global budget, different cache policies for dense weights, routed experts, n-gram rows, KV and recurrent state. Strix physical RAM/GTT separated; avoid unsafe allocations. Cache slabs/leases/generations and GPU route pointer table; host-mapped request ring only for misses, never miss on every hit. CPU expert lanes only when measured favorable; UMA CPU/GPU share bandwidth. Prefill scratch may borrow optional expert-cache capacity only through safe eviction/lease.

### Scheduler
vLLM-like unified token budgets; SGLang-like prefix-aware admission; native batching and route grouping for MoE where profitable; prefill/decode/verify distinct kernel regimes. Multiple HTTP clients are not synonymous with native batching. Fairness/TTFT/P95 and throughput objective separated; no starvation or silent session evictions.

### Speculation
Target verifier authority, MTP/DSpark/EAGLE/small-model/prompt-lookup options when compatible; automatic controller may select depth/backoff by committed tokens per **complete** draft/verify/rollback/memory cost. Proposed default is distribution-preserving stochastic sampling, approximate/opportunistic mode explicit, pending policy confirmation. Exact same-plan greedy may be tested, cross-plan bitwise identity not automatically possible.

### Device kernels
All own source Rust, through candidate CubeCL or custom backend intrinsics when peak ISA needs it. Vendor runtime FFI policy OPEN; don't commit to a full C++ inference engine. Separate prefill and decode kernels, target-specific WMMA/MFMA, sparse selector, GDN/KDA, grouped MoE and full model graph. Generated ISA and physical GPU measurements determine whether any compiler is adequate.

### Distributed
Single-device default. Pipeline, tensor/expert parallelism, optional P/D; topology/memory-aware placement and frontier atomicity. Wrong model/hash or partial collective fail closed, rebuild from durable common prefix. No unauthenticated remote exposure by default.

### Quant/quality
Input GGUF/safetensors, known tokenizer/template/architecture, strict manifest and hashes, no HF remote code runtime execution. Automatic kernel/packing change may be lossless; lower precision always a different explicitly requested artifact. Use independent CPU/FP64 oracle, official checkpoint-based model evaluation and controlled A/B benchmarks; no fabricated improvement percentages.

### UX
One clean local `run/serve/inspect/bench` experience (example commands, not decided), OpenAI/Anthropic compatible APIs where actually tested, tool/reasoning/template correctness, visible backend/fallback/memory plan, local-only bind by default, no surprise model download or code execution.

## 7. Proposed development phases, all still NOT_STARTED

The road map `IMPLEMENTATION_ROADMAP.md` defines P00 through P23:
- **P00–P05 foundations:** repo/toolchains; Rust GPU feasibility; independent CPU oracle; artifact formats; typed SIR; EIR compiler and lowering.
- **P06–P09 usable general engine:** CPU executor, GPU backends, first complete Llama-style dense model and local CLI/server.
- **P10–P12 state/memory/hybrid/MoE:** resource allocator and prefix-state cache, GDN/KDA/sparse attention, expert tiering and grouped MoE.
- **P13/P14 extreme hardware targets:** Qwen3.8 and GLM5.3 on `gfx1151`, full quality/bench evidence versus reference projects.
- **P15/P16 speculation and high-throughput server:** exact target verifier, draft algorithms, adaptive policy, continuous mixed batching, multiuser MoE grouping.
- **P17–P19 breadth:** Gemma 4/Qwen3.5+/DeepSeek and newer, multimodal, distributed parallelism.
- **P20–P23 finish:** autotuner and performance lab, security/release, benchmark qualification, next model compatibility.

Each phase lists prerequisites, tasks, exit criteria and failure/stop strategy. Do not implement all phases in one unreviewable change; begin with P00/P01 feasibility and P02 reference state correctness after explicit user instruction.

Milestones:
M0 architecture/ISA proof, M1 portable useful engine, M2 general hybrid/MoE, M3 Strix specialist, M4 throughput, M5 distributed+quality, M6 extensibility.

## 8. Research/benchmark discipline and important caveats

Performance is a *benchmarking claim*, not guaranteed by having Semantic IR. External measurements used different models, quantizations, ROCm, prefill chunk/context, CPU/GPU memory and speculation acceptance. Always match checkpoint hash/quant and quality, device/driver/power, new prefill tokens vs cached context depth, output length, warm/cold cache, concurrency, sampling and draft accept rate. Report TTFT and P95, aggregate committed tokens/s and resident bytes, not only peak t/s. Use several alternating A/B repeats, retain raw artifacts and both winning and losing results.

Examples of critical distinctions:
- A native-quant specialized engine versus Gufo Unsloth Q4 versus DwarfStar Q2 **not** same checkpoint/precision.
- ~60 tok/s repetitive MTP may drop to ~31 tok/s mixed text; not serial decode speed.
- Maya all-resident GLM performance on 128GB may degrade sharply when 92GB forces SSD expert reads.
- vLLM/SGLang enterprise throughput on H100 does not by itself predict Strix Halo on ROCm.
- `hipMemGetInfo` GTT capacity can be virtual, not usable RAM.
- Graph capture availability must be tested per backend; DwarfStar ROCm eager path is not proof HIP cannot graph capture.

## 9. What has actually been created, and what has not

**Created and relocated:** the full Markdown design corpus under `/home/pvl/spaces/abird/src/zerogravity/.agents/docs` and `.agents/plans`, plus project README. The user explicitly corrected the initial directory placement and the archive was moved. The files contain user decision IDs, original research source URLs, architecture and subsystem specs, risk/cost models, detailed phases, experiment and acceptance gates.

**Not created:** Rust code, Cargo workspace, kernel compiler implementation, release, benchmark output, CI, models, Git commit/init or tool modifications to unrelated repositories. No promise of a faster-than-specialized-engine result has been experimentally validated. If a future agent sees any implementation after this handoff, inspect and update status from **live tree**, do not claim it was part of the original documentation delivery.

## 10. How next agent should begin

1. Connect to the actual available dotlink connector; this creation used **dotlink l5**. Connector launch directory observed `/home/pvl/spaces/abird`.
2. Confirm `zerogravity/` exists, list files, read `START_HERE`, this handoff, `DECISION_REGISTER`, `ZEROGRAVITY_MASTER_PLAN`, roadmap and relevant subsystem docs **in full**.
3. Inspect `git status --short --branch`, `git rev-parse HEAD`, `git worktree list` **only if** a Git repository has since been initialized. At creation no project .git.
4. Decide the requested scope. If user asked design update, edit docs only. If coding/phase requested, take the next dependency-approved bounded phase and write validation plan first.
5. Pin and inspect live official source/repo snapshots before assuming current model geometry or upstream benchmarks. Honor source licenses.
6. For P00/P01, preserve future extensibility but keep code small. First meaningful GPU experiment is Rust-authored gfx1151 Q8/IQ2 MMVQ/WMMA and one stateful GDN step with CPU oracle. No giant serving framework before this proof.
7. After code, test CPU and physical relevant GPU, run performance and quality controls, update roadmap status with commit/test/evidence, decision register, affected specs and handoff.
8. Never report phase complete without fresh live-tree evidence.

## 11. High-priority open decisions

Explicitly unresolved, don't invent a user's selection:
- O-001 project license;
- O-002 production hardware target order and v1 scope;
- O-003 strict interpretation of Rust-only around vendor driver/math APIs;
- O-004 primary GPU Rust compiler/CubeCL versus custom low-level backend;
- O-005 first model release acceptance target and quant checkpoint;
- O-006 exact quant/format support matrix;
- O-007 cross-backend determinism policy;
- O-008 public SDK/API ABI;
- O-009 remote transport/auth policy;
- O-010 OS/toolchains/release CI support;
- O-011 model-quality thresholds;
- O-012 SLO and concurrency defaults;
- O-013 declarative vs Rust model adapter interface;
- O-014 model/kernel plugin trust;
- O-015 when to initialize Git/release;
- O-016 CLI branding and public flags.

Some can be resolved by benchmark before asking preference; explicit product and policy choices should be presented to user when they materially block implementation. Do not re-ask user facts already provided.

## 12. Source/decision preservation policy

New architecture decision record must include date, source/user instruction, old assumption, rejected alternatives, changed invariants, source/benchmark proof, next actions and superseded decision ID. Documentation can evolve, but historical reasoning and negative experiments should not disappear. External references must retain URL, commit or dated snapshot and source license. Model weights and private prompts are not checked into this project by default.

The long-term engine must remain both **maintainable and optimizable**: no abstraction may impose a hot-path performance ceiling, but optimized kernels must be lawful under the same model semantics and state contract. Code-level specialization can always replace a whole IR region if it is independently proven correct and efficient.

---

## 13. Self-contained next-agent starting prompt

```text
You are taking over the zerogravity Rust inference-engine project in
/home/pvl/spaces/abird/src/zerogravity using the available dotlink connector.

Before any implementation, read IN FULL:
- .agents/plans/AGENT_IMPLEMENTATION_PLAYBOOK.md
- .agents/plans/DECISION_REGISTER.md
- .agents/plans/ZEROGRAVITY_MASTER_PLAN_2026-10-10.md
- .agents/plans/IMPLEMENTATION_ROADMAP.md
- .agents/plans/EXPERIMENT_PROGRAM.md
- .agents/plans/RELEASE_ACCEPTANCE_GATES.md

Then read all relevant .agents/docs, especially:
SEMANTIC_IR_AND_COMPILER, STATE_CACHE_AND_TRANSACTIONS,
DEVICE_KERNEL_AND_CODEGEN, MEMORY_RESIDENCY_AND_IO,
SCHEDULER_BATCHING_AND_SPECULATION, MODEL_SEMANTICS_AND_COMPATIBILITY,
INVARIANTS_AND_FAILURE_SEMANTICS, NUMERICS_QUALITY_AND_BENCHMARKS,
RUST_WORKSPACE_AND_ENGINEERING, SECURITY_AND_RELIABILITY.

Inspect the live directory, repository state (if Git exists), and all source
before trusting any documented phase completion. All implementation phases
were NOT_STARTED at creation of these plans.

Preserve the user's requirements: name zerogravity; Rust-authored engine
and owned kernels; general semantic portable fallback; vLLM/SGLang-class
serving ambition; llama.cpp-like ease/model breadth; maximal eligible
hardware specialization for Qwen3.8-Flash-Next, GLM-5.3-Flash, Gemma 4,
Qwen3.5+, Llama and others, especially AMD gfx1151/Strix Halo.

Keep proposals/open decisions distinct from explicit user approval.
Do not claim benchmark wins without physical tests; do not silently change
precision or sampling; do not copy proprietary engine code.

Follow the narrow dependency-ordered roadmap and write rigorous numerical,
state, memory, security and end-to-end performance evidence. If the current
request is design-only, work on docs and do NOT begin Rust code.
```

## 14. Closure of THIS handoff

The right next action depends on the *next* user instruction: either review/refine these specs or begin P00/P01 once explicitly requested. Everything necessary to orient an engineering agent—the user goals, historical research, architecture, subsystem contracts, phase dependencies, risks, validation criteria and open decisions—has been deliberately preserved on disk.