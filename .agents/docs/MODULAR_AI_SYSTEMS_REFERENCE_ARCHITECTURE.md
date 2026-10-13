# zerogravity — modular AI execution reference architecture v4.1

**2026-10-12 · authoritative v4.1, design only.** This is the **smallest architectural contract** for a high-performance, composable Rust AI runtime, including native kernels and compiler-centric/device-graph/dataflow execution families. Exact formal execution/state details are in `MINIMAL_EXECUTION_SPINE_AND_PROTOCOL_V4.md`; model/task/compiler details in `TASK_PROGRAM_AND_COMPILER_REFINEMENT_V4.md`; critical review and decisions in `ARCHITECTURE_V4_CRITICAL_REVIEW_2026-10-12.md`. The previous v3 text is preserved in `DESIGN_EVOLUTION_V3_SNAPSHOT.md`. No code or benchmark has been implemented.

## 1. Prime rule — compose execution capabilities, never new engines

**A model describes mathematics; a task describes control; a provider supplies an implementation; a plan schedules computation and physical resources; a coordinator owns visible state.** No new model, task, quantization, accelerator or specialization is allowed to require a second independent model server, allocator, output service or task-session authority.

We target a practical **Pareto frontier** of mathematical/task quality, latency/P95, throughput, energy, memory capacity, ease of deployment and cost of extension. A general design does not automatically beat a specialized engine on one model/hardware workload; the ability to **substitute a complete hand-tuned region** while reusing the same runtime gives us a path to compete.

The runtime is Rust-owned. Vendor drivers/toolchains are optional devices with explicit language/FFI/license policy. It must always be possible to build a useful CPU-only reference path without any GPU/NPU compiler or external model server. Unknown mathematical ops are not magically runnable from a recognized checkpoint.

## 2. The minimal system vocabulary

| Canonical concept | Authority | Not responsible for |
|---|---|---|
| **Artifact/ModelDefinition** | immutable model weights, config, tokenizer/quant, exact semantic program/registered state effects | selecting a vendor kernel or serving requests |
| **SemanticProgram (SIR)** | mathematical operations + shapes/precision/position + typed mutable effect dependencies | physical device buffers |
| **TaskDriver** | task-specific stage/iteration control, stop/frontier/output type; one driver per task session | hardware kernel codegen and external privileged agent workflows |
| **RegionExecutionProvider** | one **closed semantic region** lowered to a native kernel, captured graph, ahead-of-time device graph or static dataflow program; native KernelProvider is one option | mutating session roots, duplicating a model runtime, using unknown opaque inference APIs as verified execution |
| **ExecutableBundle** | immutable prepared kernel handles + **CIR** command/resource DAG + guarded variants/fallback | live request queue or plan search on every output |
| **DeviceBackend/IOBackend** | physical execution, memory subpool, async stream/events/transfers, runtime capability | model semantics or user-visible results |
| **PhysicalResourceArbiter** | coarse global **admission/reservation/headroom** and quotas across physical tiers | per-kernel allocation lock or setting user's firmware |
| **TaskCoordinator** | one logical mutable writer **per TaskSession/shard**; commit/stream ordering, fair scheduling | direct unsynchronized device memory mutation |
| **StateRoot/StateTransaction** | immutable version tuple across **all** state domains; prepare/validate → one atomic logical root publication | magically atomically writing all GPU device memory |
| **ModeReceipt/Evidence** | exact supported backend/shape/precision/path, performance/quality and fallback provenance | claiming unmeasured performance |

These concepts can start as **modules within one Rust library plus binary**, not nine crates and a plugin ecosystem. The exact accelerator micro-schedule can be owned by its device/compiler; Rust retains model/task/state/reservation authority and a single program submit/ready boundary. Optional device providers can later be separated into a few crates when target/compiler build constraints justify it.

## 3. Two enduring representations; compiler-internal optimization views

```text
Artifacts + ModelDefinition            Trusted task request / TaskDriver
             |                                        |
             +--------------> SemanticProgram (SIR) <+
                           tensor math, typed effects
                                     |
                   transient RIR legal-region transformations
                     + provider candidates: tile, whole graph, static dataflow
                                     |
               physical CIR command/resource/timepoint lowering
                  (kernel, graph, fabric program or supported remote program)
                                     |
                           ExecutableBundle
               guarded shape bands + code handles + memory DAG
                                     |
                PhysicalResourceArbiter (coarse reserves)
                                     |
                TaskCoordinator -> StepPlan -> DeviceWorkers
                                     |
                   Prepared StateRoot + EventCompletion
                                     |
                  PublishRoot by single session owner
                                     |
                 Ordered committed task outputs/ModeReceipt
```

**Only the semantic program and physical executable plan have durable/architecturally stable identity.** SIR is the canonical executable mathematical reference; RIR and KIR are bounded compiler-local *views*, and CIR lives within the physical plan. These stages are helpful engineering vocabulary, **not a mandate for four independently serialized dialects or runtimes**.

At steady operation, the device worker submits **precompiled regions/kernels/graphs/static dataflow programs directly**, allowing a vendor accelerator compiler to own its micro-schedule without a host-side loop per tensor operation. It does not interpret the SIR or run optimization passes per language token/audio frame/denoise timestep. Dynamic router IDs and sampling can remain in device buffers where the target mathematical/grammar contract allows, avoiding mandatory host roundtrips.

A new specialized whole-layer or whole-model code path is simply a `RegionCandidate` satisfying a closed semantic region contract (same inputs, outputs, mutable state effects, numerical/correctness mode and bounded resources). The portable decomposition remains valid and can be forced on in tests. Fallback between incompatible physical state layouts occurs at a canonical committed checkpoint with conversion/replay, not by pointer reinterpretation.

## 4. Task control without an internal workflow platform

Use a **small trusted Rust `TaskDriver`** to issue stages and transitions for a task kind, not a general user-supplied DSL for arbitrary agent steps. Stage-level control can express a bounded branch/window/loop in the TaskDriver's implementation, but device-native `StepProgram` executes repeated mathematical steps where possible without frequent host wakeups.

Primary task types:
- autoregressive text with exact sampling/optional trained speculation and token frontier;
- embedding/encoder and rerank scoring without decoder-specific KV;
- ASR and streaming speech recognition with audio-window/sample frontier;
- TTS/voice acoustic generation + vocoder with complete PCM/audio chunks;
- image generation/edit with validated denoiser/flow/timestep loop and latent state;
- video/audio with frame/window/temporal state and synchronized outputs;
- compound **AI computation** with trusted stage composition, not arbitrary tool execution by default;
- **future optional** gradient/parameter-update training task, whose mutable model version/optimizer state are isolated from running inference sessions.

External RAG, tool use, agent/cross-product workflows and privileged actions belong in an **application/orchestrator above** the inference TaskRequest boundary, with authorization/sandbox. The inference runtime is a compute engine, not an unrestricted automation system.

Every task declares its own `WorkUnit`, `CommittedFrontier`, `StreamContract`, `CancellationPolicy`, resources and numerical/quality checks. A video frame isn't an LLM token. Model/task support remains status-gated by actual checkpoint and code implementation; listing a kind is not claiming support.

## 5. State and output transaction, exactly one logical publish

The earlier `StateDomain::commit()` was insufficient. v4 requires:
1. **Begin** from pinned immutable `StateRoot{session_epoch,frontier,domains}`.
2. **Execute** into private/COW provisional state; device/IO workers return events and candidate state.
3. **Prepare** each domain under one target frontier, verify shape/schema/quant, sampler/RNG/grammar/adapter and actual device event readiness; failure → abort and quarantine buffers where DMA still live.
4. **PublishRoot** exactly once by sole logical session writer when all required domains ready. No domain independently publishes committed state and no physical CIR command has that authority.
5. **Emit** ordered verified results identified by `(session,epoch,output_seq)`; live transport promises ordering/at-most-once *attempt*, not impossible exactly-once receipt across lost network ACK. Optional authenticated cursor replay can support resumable delivery.

This is atomic **logical root visibility**; physical GPU state updates use immutable/COW/private tail or safe undo/recompute. Distributed workers require versioned epochs/fences/readiness agreement; never claim instant distributed memory atomicity or durable exactly-once semantics without a real distributed log protocol.

Full-state prefix cache (KV, recurrence, sparse indexer, n-gram, sampler/grammar, adapters, modality) reuses the last **common compatible** frontier across all domains, not simply token text; immutable shared snapshot identity prevents cross-tenant/model/quant/position leakage.

## 6. Fast scheduling and resource ownership

One session/shard writer owns current TaskCursor and committed state, but not every hardware allocation in a global mutex. **Global admission is coarse; device-local subpools allocate fast under reservations** with generation-tagged BufferLeases and async retire; all permanent/temporary model, KV, recurrent, experts, encoder/media scratch, NPU staging and remote IO count against real physical RAM/VRAM/OS headroom.

A `StepPlan` binds compiled `ExecutableBundleId` with actual request/stage rows, device slot generations, runtime `ShapeGuard`, reserved leases and target state epoch. Persistent worker batches apply diffs; precompiled device-native step islands can route/specialize/advance dependent operations without host sync as allowed. Prefill, decode1, verify, encoder, denoise and vocoder are separate shape/cost regimes. A safe host-loop plan remains for features whose grammar/recurrent effects prohibit device-only progression.

Scheduler policy uses **typed stage cost/resource budgets** and fairness deadlines, not one universal token budget. Coarse stage coordination can overlap GPU and CPU work, route through sparse expert/SSD tiers and use NPU or remote accelerators after actual availability/numeric/fabric safety qualification. A slow client cannot accumulate infinite uncommitted GPU/PCM/frame output buffers; backpressure and cancellation are required.

**Physical safety is conservative, not omniscient:** OS memory changes independently of the engine. Probe and reserve headroom, monitor pressure and fail/evict/replan safely; don't promise zero OOM on an unrestricted multiprocess OS. GPU-visible GTT/virtual addressing isn't free physical memory.

## 7. Shape, numeric, cost and specialization guards

Every KernelCandidate/ExecutableBundle variant is bound to:
- semantic/StateEffect identity, numeric policy and source artifact/quant;
- symbolic **fixed rank + bounded dynamic dimension constraints**, alignment, ragged rows, supported batch/context/step band and output layout;
- device ISA/driver/compiler/program and optional graph/eager/safe-power profile;
- scratch/physical memory leases, availability/first-use fences and state-layout compatibility;
- numerical/quality evidence and measured stage/full-task performance.

Dynamic router IDs are **runtime data**, not a per-token compiled kernel variant; unsupported shapes always have a generic runtime guard/fallback if that operator is semantically supported. Compile/tune a small set of hot shape bands, not a codegen artifact for every context length. Offline bounded search; online only select approved variants at safe epochs with hysteresis. Compiled GPU plans are expendable caches invalidated on toolchain/driver/model changes; **never persist raw driver handles** inside state snapshots.

Qualify performance under identical model/quant/context/task/power/driver and exact numeric policy, with TTFT/P95, committed output units, quality, physical memory, energy and source-proven baseline. A mathematically model-changing approximation (reduced expert top-k/mask/altered diffusion schedule) requires explicit user opt-in, never automatic "optimization."

## 8. Minimal extension seams and evolution

| Adding | Extend | Must remain unchanged |
|---|---|---|
| New model with known ops | ModelDefinition/config/weights and independent reference fixtures | TaskCoordinator, DeviceBackend, ResourceArbiter |
| Novel recurrent or sparse op | semantic op + exact StateDomain + portable reference | unrelated models, CLI/server |
| New kernel/device or graph/dataflow accelerator | RegionExecutionProvider and compatible native kernel/graph compiler/runtime adapter, state bridge, physical/topology/shape and safety evidence | model mathematics, task frontiers and user-visible session authority |
| New quant or layout | checked codec/pack and numerical contract | generic session service |
| New AI task | TaskDriver stages/output/stream frontier and needed op semantics | device/resource/other TaskDrivers |
| New storage/peer link | CIR IO/transport and ResourceTier with lease/event semantics | tensor math and TaskDriver |
| New speculative algorithm | DraftSource/target verifier + full state transaction | ordinary model operations |
| New compiler optimizer | verified legal region transformation and bounded search | model source, user tools |
| New service adapter | canonical TaskRequest/Result mapper, auth and privacy | kernel providers, state effects |
| Future training | optional GradientProgram and ParameterVersion publication separate from inference | existing immutable inference model/session semantics |

The **modularity test** is practical: implement one new model operator, one accelerator provider and one non-token TaskDriver without editing unrelated API, memory, GPU driver or other task code; measure code churn/compile/load overhead and quality.

## 9. Deferred complexity and engineering discipline

**Initial implementation:** one library and one binary (or 2–4 cohesive crates if GPU toolchain demands it); static built-in providers, one model+TaskDriver vertical slice and device event/lease checker. Small versioned types, direct prepared execution, independent reference. Avoid bloated trait hierarchies and duplicated cache ownership.

**Defer:** generic plugin ABI, unrestricted workflow DSL, online distributed training optimizer, automatic dynamic model JIT every shape, equality-saturation in steady serving, all hardware types, media endpoints without reference checkpoints, distributed durable transaction WAL, third-party executable source code loading.

**Security:** no untrusted model remote-code execution or implicit tool privilege; no automatic `sudo`, BIOS/IOMMU/fabric-clock/suspend modification to speed an experimental GPU/NPU path; no cross-session private prefix/cache leak; executable device binaries/trusted code require independently reviewed source and precise driver capability. Rust-owned kernel source requirement persists, vendor driver FFI policy open.

## 10. Evidence, scope and authority

**Primary current docs:** `COMPOSITION_AND_EXTENSION_CONTRACTS.md` (corrected illustrative interfaces), `MINIMAL_EXECUTION_SPINE_AND_PROTOCOL_V4.md` (state and async executor), `TASK_PROGRAM_AND_COMPILER_REFINEMENT_V4.md` (IR and TaskDriver), `ARCHITECTURE_V4_CRITICAL_REVIEW_2026-10-12.md` (review and prior-art research), v4 master and decision register. All earlier subgroup documents are supporting details, subordinated where in tension with this v4 design.

**Release proof:** G0 real model and fake async event/shape/state faults, G1 CPU/GPU usable path, G2 multi-session and hybrid-state service, G3 extreme model/device specialization, G4 non-token model/task breadth and distributed, G5 reproducible release, G6 extension drill and sustainable evolution. All remain **NOT_STARTED** until code/tests/hardware evidence exists. `research/inference` is read-only evidence.

**Long-term win condition:** new semantic models, kernel hardware, state machines, iterated tasks and storage tiers attach without rewriting the runtime. When a fully specialized closed region is fastest, use it there; when it loses/unavailable/unsafe, preserve the portable model and a truthful receipt.

## 11. v4.1 execution granularity — native GPU, graph-only and dataflow accelerators

The single `RegionExecutionProvider` seam supports **multiple physical execution kinds** without adding another model or TaskDriver runtime:
- **NativeKernel** and **CapturedCommandGraph**: prepared direct CPU/GPU/edge accelerator computation;
- **AheadOfTimeDeviceGraph**: an entire legally closed model region/task stage compiled by a graph-centric accelerator, with shape, state, runtime, firmware and quant qualification;
- **StaticallyScheduledDataflow**: device compiler places entire operation pipelines/collectives and manages SRAM/on-chip program/weight streaming, preserving a large closed region boundary;
- **ManagedDeviceSession**: only when driver/SDK exposes enough program-state/snapshot/rollback/epoch and cost semantics for our StateRoot contract; opaque model-serving APIs are *application TaskService adapters*, not lowerable backend implementations.

The provider determines minimum efficient **region granularity** (tensor op, fused region, stage, whole model, multi-device partition) and micro-schedule owner. **Rust TaskCoordinator still owns model/task admission, one StateRoot PublishRoot, output sequence and global physical reservations**; a vendor graph compiler may own only internal device scheduling and memory layout. TPU-like graph programs need bounded dynamic shape groups and nontrivial compilation/warmup budget; wafer-scale/static pipelines need programmed fabric, streamed weight sources and stable on-chip residency; GPU rack domains need actual peer links/collective topology and fault fencing; coherent local SoCs need real physical UMA not assumed HBM.

A source compiler/runtime inaccessible or forbidden under Rust-source/FFI policy produces `CAN_MODEL_BUT_NOT_COMPILE` rather than a false `SUPPORTED` status. A device graph with opaque **in-place state** but no safe snapshot/recompute cannot legally replace a general stateful speculative region. Tiny host kernel launches should not be used to emulate the static pipeline; match the *target's efficient native granularity*. **All of this remains within one SemanticProgram→ExecutableBundle execution spine.**

**Detailed contract:** [accelerator execution families](ACCELERATOR_EXECUTION_FAMILIES_AND_BACKEND_CONTRACT.md). **External prior art:** [accelerator platform research](../prior-art/ACCELERATOR_PRIOR_ART_AND_PLATFORM_RESEARCH_2026-10-12.md) (research only).

## 12. v4.2 — distributed machine and state placement as a core execution capability

**The universal runtime is local-first but network-composable.** Independent maps for compute, immutable model parameters, mutable per-layer/per-session state, and reusable immutable caches combine into **one PlacementPlan** within ExecutableBundle/CIR, with a costed TransferPlan and TopologyEpoch. This does not add a third enduring IR or an independent cluster inference engine. The same RegionExecutionProvider, DeviceWorker/RemoteRegionWorker, TaskDriver, ResourceArbiter, StateRoot Prepare/PublishRoot and output sequence apply on every machine.

Capabilities span single-host colocated, per-layer compute+KV colocated on remote peers, compute separated from attention+KV, dense/FFN/experts on other workers, prefill/decode state handoff, remote prefix hydrate, attention token-range shards, tensor/head/expert collectives, and different TaskDriver encoder/denoiser/vocoder stages. **Remote storage, remote stateful compute, and full remote state reads are different physical candidates.** Dynamic movement only at a safe committed frontier, and SIR operator-specific PartitionAlgebra is necessary before token-range attention splitting. Caches can be stale hints; authoritative mutable shard references require versioned leases and epoch fencing.

**Peer connectivity:** explicit authorized peer enrollment; control over authenticated direct/tunneled links; measured separate bulk data transport over physically eligible RDMA/peer or bounded encrypted streams. Tunnels are connectivity, not RDMA or low P95 latency guarantees; relayed WAN often favors whole layer+cache placement over chatty per-layer splits. Remote failure returns Prepared/Failed, never another session writer; a partition stops or replays safely, without claiming transparent leader failover/exactly-once output.

**Full specs:** [placement and physical cost](DISTRIBUTED_PLACEMENT_AND_NETWORK_EXECUTION_CONTRACT.md), [peer/tunnel and state protocol](REMOTE_PEER_TRANSPORT_AND_STATE_HANDOFF_PROTOCOL.md), and [subsystem entry](DISTRIBUTED_PARALLELISM.md). Source names/benchmark caveats remain [research-only](../prior-art/DISTRIBUTED_PRIOR_ART_AND_TUNNEL_RESEARCH_2026-10-12.md).

## 13. v4.3 — built-in multi-model request routing without another runtime

The project now has **one small ModelCatalog/ModelRouter/ResidencyManager control plane** integrated with the same TaskCoordinator, ResourceArbiter, RegionExecutionProvider/ExecutableBundle and DeviceWorkers used by single-model direct `run`. **Router request admission is not a hot-per-token execution path**. Standalone local model inference can run in-process without an HTTP server or daemon, and a long-lived `serve` instance exposes the same library engine across multiple registered models.

Separate `RegisteredModel` immutable checkpoint/quant/ModelDefinition, versioned `ModelAlias/ModelProfile`, actual resource-admitted `ResidentModel/Replica` (potentially one sharded DeploymentPlan across nodes), and `TaskSession` with its own full StateRoot. Registered model count isn't resident model count; replicas of same checkpoint aren't multiple distinct model sources; target/draft/encoder different models in one TaskProgram aren't replicas. One alias revision resolves to an exact immutable source/adapter/template/numeric identity for the session duration. **No automatic different-model or lower-quant substitute** under memory/queue pressure.

The router chooses compatible warm replicas using measured queue/cold load/physical capacity/prefix affinity and authorized device/peer placement; stateful sessions remain sticky unless full compatible StateRoot handoff at a committed frontier. Model residency (verify→admit→load→warm→ready→idle/drain→unload) is safe under physical reservations, per-device local pools and lease/drain gates. Pin/keepalive/scale-to-zero are optional. A configured `auto` alias may deliberately choose a different target model under an explicit user/model-quality policy and reports the actual source.

**Product surface:** proposed simple `zerogravity run <model>` and `zerogravity serve`, native `models list/ps/inspect/pull`, versioned model profiles and one loopback OpenAI-compatible `/v1` API subset with honest SDK/streaming/usage/tool-output conformance. Native model-management/admin rights are separate from public generation. Full reference: [model router and residency](MODEL_ROUTING_RESIDENCY_AND_MULTI_MODEL_SERVING_CONTRACT.md), [CLI/config/API design](ZERO_CONFIG_CLI_CONFIG_AND_OPENAI_API_CONTRACT.md). Source project comparisons in [non-normative research](../prior-art/MULTI_MODEL_ROUTER_AND_API_PRIOR_ART_RESEARCH_2026-10-12.md).
