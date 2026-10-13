# zerogravity — v4 critical architectural review, challenges and resolution record

**Review date:** 2026-10-12. **Scope:** a fresh cross-cutting engineering review of the complete v3 Markdown architecture, subsystem specs, 24-phase implementation roadmap, 31 experiments, task/device API sketches, state/memory/safety contracts, and the prior-art research corpus. **Classification:** design review and adjudicated architecture proposals, **not code or benchmark results**. New authoritative execution rules live in `MINIMAL_EXECUTION_SPINE_AND_PROTOCOL_V4.md` and `TASK_PROGRAM_AND_COMPILER_REFINEMENT_V4.md`. Legacy v3 core is preserved in `DESIGN_EVOLUTION_V3_SNAPSHOT.md`. The current master/decision register controls status.

**User objective:** achieve the highest plausible performance/quality/productivity Pareto frontier for **composable AI execution**, retaining a very small mandatory core, clean future model/operator/device/quant/task extensibility and no new model-specific inference engine for specializations. Performance supremacy is *unverified*, not an architectural guarantee.

## 1. Research methodology and genuinely material new evidence

The review used the existing **read-only** `research/inference` source investigation, v3 live project specs and independent first-party public design/implementation documentation. Source names are **deliberately confined to this research record** and the existing prior-art/license atlas; normative v4 design uses independent mechanism names.

- **Asynchronous inference hot path:** [vLLM Model Runner V2, March 24 2026](https://vllm.ai/blog/2026-03-24-mrv2) separates generic model logic from GPU-native input preparation, device-side rejection sampling, CPU/GPU overlap, and asynchronous output stream. It explicitly calls out host synchronization and CPU-producer/GPU-consumer buffer race risks. The lesson is **avoid mandatory CPU round-trips between GPU steps while retaining a logical owner and causally ordered state**; do not blindly reproduce one library's runner API.
- **Explicit resource lifetimes:** a streaming-resource dialect of a known compiler IR models async work placement, timepoints, allocation/liveness and target-specific resources. The lesson is **model device work and resources separately from mathematical graph**. Do not implement the entire external compiler as zerogravity's runtime.
- **Portable semantic schema and shape dynamism:** [StableHLO compatibility](https://github.com/openxla/stablehlo/blob/main/docs/compatibility.md), [dynamic dimensions](https://github.com/openxla/stablehlo/blob/main/docs/dynamism.md) and [spec](https://openxla.org/stablehlo/spec) demonstrate versioned operator contracts, static-rank/dynamic-dimension handling, runtime guards and effect tokens. The lesson is **explicit shape predicates, schema version compatibility and effect verification**, not source project dialect inheritance. This document does **not** propose embedding StableHLO C++ in the core or claiming its exact compatibility window for our own artifact.
- **Serving fairness:** [Fairness-aware chunked-prefill research (June 2026)](https://arxiv.org/abs/2606.09061) identifies latency jitter/head-of-line blocking from rigid token budgets and studies stage time constraints. The lesson is **scheduler budgets as target-time/resource/fairness tradeoffs** rather than unconditional 128/2048-token chunks or only aggregate throughput. The paper's author results are not independently reproduced here.
- **Technical risk:** wide model/inference source corpora show that per-model fused kernels, hetero expert ownership and GPU+NPU projection slicing can be fast under particular shapes/precision; they should be optional providers with references, typed state boundaries, XRT/GPU event ownership and physical hardware safety qualification, not the new core.

The review does **not** claim other systems lack some technique or that new 2026 software has been compiled/tested on our own devices.

## 2. Severity-ranked findings: concrete faults in v3

Severity indicates *cost of freezing the wrong interface*, not a discovered executable bug—**no executable engine exists yet**.

| ID / priority | v3 design problem | Consequence if coded literally | v4 correction |
|---|---|---|---|
| **R4-01 Critical** | `StateDomain::commit(candidate,...)` can mutate/publish individual domains before the promised all-domain atomic `StateTransaction`. | partial logical KV/recurrent/RNG state on failure; impossible true all-domain commit | domain **Prepare**/rollback, immutable state-root assembly, one **PublishRoot** by single session writer, retire old root after device-event quiescence |
| **R4-02 Critical** | CIR includes `Commit{transaction}`, while TaskCoordinator/EngineCore independently owns commit. | two commit authorities and ambiguous device/host visibility | CIR signals `Ready/Failed` provisional work; **only coordinator publishes logical state/output**; device command completion never means user-visible commit |
| **R4-03 High** | Logical single owner is sometimes conflated with physical owner for all request/session queues, KV ledger and allocations. | one process-global lock/queue bottleneck and synchronous allocations under C32+ | single writer **per session/shard**, independently concurrent model/device workers, hierarchical resource reservations; fast device-local allocator owns physical suballocations |
| **R4-04 High** | Four IR names SIR/RIR/KIR/CIR are presented as separate, always mandatory compiler data formats. | compiler framework complexity, version explosion and per-step overhead | only **two durable intermediate contracts** initially: semantic program and physical executable plan; RIR/KIR are **transient pass/candidate views**, CIR is the typed command/resource portion of executable plan; no general compiler framework dependency |
| **R4-05 High** | `TaskNode::{Sequence,Parallel,Select,Iterate,ExternalAction...}` effectively includes a generic app/agent workflow DSL in the hot runtime. | scope creep, memory/security complexity, task graph interpretation every token | **small trusted Rust TaskDriver** with `poll/prepare/step/commit/emit` and optional small bounded stage DAG; external tools and agent workflows in a **separate permissioned application layer** |
| **R4-06 High** | v3 says “no per-token IR interpreter” but does not define how dynamic routed experts, streaming loops, shape changes and draft accept/reject avoid CPU sync. | fast low-level kernel nullified by CPU copy/graph breaks and old-input races | prepared GPU-native **StepProgram** with index/generation buffers, mutable parameter slots, device routing/sampling where valid, host control only at true inter-stage/commit boundaries |
| **R4-07 High** | Multiple identities (`KernelPlan`, DevicePlan, `ExecutableBundle`, `ExecutionPolicy`, `StepPlan`) but unclear what persists/changes and what is in cache key. | recompile storms, stale device pointers, graph-cache explosion and misuse after driver/model upgrades | identity matrix + canonical plan fingerprints; immutable compiled variants, mutable bounded parameter slots; runtime step only binds IDs/shape guards/state epoch. LRU and cost of cold JIT explicitly budgeted |
| **R4-08 High** | Physical memory budget is described as a universal exact guarantee despite OS/cgroup/GTT changes and kernel/driver allocations. | false no-OOM promise, reserve bottleneck or unsafe peak scratch | global **admission reservations + safety headroom**, per-device memory pool, runtime pressure/reclaim/watchdog, fail-closed allocation; cannot guarantee OS never changes memory unexpectedly |
| **R4-09 High** | Dynamic dimension, ragged batch, sparse route and chunk shape bounds aren't a first-class typed guard. | treating `M=4096`/exact quant/shape as generic; wrong kernel call or endless recompilation | bounded symbolic `ShapeEnv` / runtime guard contract, segmented ragged rows and shape bands; decompose/fallback on shape miss, JIT off critical path |
| **R4-10 High** | Numerical `exact sampled distribution`, “bitwise full logits”, “reference” and “validated tolerant” mix distinct guarantees. | impossible cross-backend bit-identical promises and accepting numerically changed router top-k | explicit math, numerical/quality and stochastic sampling **separate contracts**; same-plan deterministic replay distinct from cross-plan tolerance; intentional approximation only opt-in |
| **R4-11 High** | Distributed “atomic same frontier” underspecifies partial network failure, lost ACK and external streaming. | dual ranks committing different state, duplicate/phantom output on retries | epoch fencing, prepare/ready/commit protocol at logical coordinator, idempotent retry/replay, outputs after durable or volatile commit according to declared guarantees; no magical exactly-once network semantics |
| **R4-12 High** | Output streaming tied implicitly to state publication with no explicit transport delivery contract. | duplicate token/audio chunk on disconnect/retry, exposed uncommitted media | ordered `(session,commit_epoch,output_seq)` emission with at-most-once live stream, optional durable replay/ACK if requested; exact-once end-to-end **not guaranteed** |
| **R4-13 High** | Training/backprop and mutable model weights absent despite “all AI execution”, while requirements could imply immediate support. | either unnecessary initial optimizer/training bloat or architectural barriers later | keep inference-first **read-only model core**; reserve `ParameterVersion + GradientProgram` extension seam for training/fine-tuning with separate task loop and versioned optimizer state, **not default release commitment** |
| **R4-14 High** | TaskProgram's `ExternalAction` appears inside core while security forbids arbitrary tools. | inference engine becomes privileged workflow executor | move actions to app/agent integration boundary with authenticated intent and sandbox; core only accepts trusted typed task/data requests |
| **R4-15 Medium** | Optional plugins and 8–12 crates appear mandatory in roadmap examples. | unnatural trait dispatch/many dependency seams and long compilation | start **one library + one binary** or 2–4 cohesive crates; statically linked built-in providers, expand only with measured compile/security/release boundary |
| **R4-16 Medium** | “one semantic op” can be a giant fused black box or tiny primitive without guidance. | can't optimize/fallback at useful layers; wrong effect hiding or variant explosion | two-level op vocabulary: high-level **semantic regions** retain attention/recurrent/task state identity; bounded low-level pure tensor primitives/transformations inside region; only required equations model-specific |
| **R4-17 Medium** | Specialized cache and state snapshot/effect checker don't specify immutable source artifact vs mutable LoRA/weight generation placement. | cross-adapter cache contamination, drafter mismatch and executable invalidation | separate immutable checkpoint/packing, adapter generation and parameter version; central `CompatibilityKey` includes source+task+state/numeric/geometry when needed; version changes invalidate unsafe state/plan reuse |
| **R4-18 Medium** | All optimizer candidates assumed to be searched offline before serving; cold model/shape support unclear. | long load time, surprise cold compile and missing all-purpose baseline | always provide safe portable kernel/reference; load only needed shape bands, deferred background/JIT only if explicitly requested/controlled; no runtime work waiting for expensive search |
| **R4-19 Medium** | Benchmark claims use “committed tokens” as universal unit, despite audio/image/video tasks. | false comparability of throughput | define `WorkUnit` and `OutputUnit` for each TaskDriver; normalize energy/latency per **meaningful committed task output** with optional device utilization, not fake token conversions |
| **R4-20 Medium** | Broad device/topology/NPU safety modeled, but qualification/unsafe transition has no invalidation and rollback semantics. | stale “safe” program after firmware/power change and dangerous auto run | device attestation fingerprint and runtime safety gate; on change disable/re-evaluate profile, preserve CPU/GPU-only; no automatic privileged mitigation |
| **R4-21 Medium** | Data/model representation has no clear stop at “inference runtime” boundary vs tool-rich agent platform. | duplicative security sandbox/state engines, misses specialization priorities | core provides typed programs/effects/outputs; external apps can orchestrate multiple calls and tools via independent permissioned adapters; no embedded unrestricted agent automation |
| **R4-22 Medium** | No mandatory "negative architecture test" when adding future model types. | creeping model-name conditionals and duplicated runtime | explicit change-radius budget, second model/new device/non-token task/fault injection, architecture acceptance must fail if new specialized server/allocator required |
| **R4-23 Medium** | Version policy lists many keys but lacks canonical wire/snapshot/compiled cache separation. | serializing raw pointers, breaking upgrades, stale implementation state | specify persisted artifact/semantic/state schemas separately; **never persist executable pointers**. Compiled bundle caches expendable and invalidated by driver/compiler; snapshot format versioned and verified independently |
| **R4-24 Medium** | Precision and performance optimization objective lacks explicit counterexample: reduced top-k is faster but changed trained math. | accidental quality/sampling change | mark optimization as `SemanticallyEquivalent`, `NumericallyDifferentButValidated` or `ModelChangingApproximation`; only first two auto-eligible under configured numeric policy; latter only explicit opt-in |

## 3. Primary adjudications — fewer concepts, not more mandatory infrastructure

**D4-A — Minimum runtime ownership:** three core authorities only: **Model/Task program ownership (immutable)**, **per-session mutable TaskCoordinator logical state**, **PhysicalResourceArbiter admission and per-device leases**. Compiler and backend providers are pure/adaptable; they cannot independently mutate session state. TaskDriver is a trusted control policy, not yet a DSL runtime.

**D4-B — IR minimization:** SIR persists as canonical mathematics/effects; RIR/KIR are *compiler-internal transient views* (no new universal serialized formats); CIR is a compiled **command and resource DAG within executable plan**. Runtime only consumes direct plan handles and bounded parameter slots. Keep terms to communicate phases but don't force the codebase into four separate interpreters or crate registries.

**D4-C — Safe session state publication:** StateDomains do **not** expose a public mutating `commit()`; instead return immutable candidate versions, with `prepare` and rollback/quarantine, and a single atomic root publication when all required device events are ready. Old root stays visible on aborted speculative branch, IO failure or model repair. Streaming output only after publication.

**D4-D — Async-first hot loops:** host TaskCoordinator must schedule and own commit logically; a kernel/graph can compute multiple dependent token steps or verify candidates entirely device-side if verified, yielding a *single prepared frontier summary and output slice* to host after event completion. No CPU per-token router/grammar/quantize barrier unless model exactness or user-visible output requirement demands it.

**D4-E — Executor modularity:** KernelProvider capabilities include op/region semantics/shape/quant/numeric/device regime. A provider can be a hand-tuned whole model region, but cannot own its own API, state authority, store, allocator or task loop. Device programs and XRT/GPU driver FFI are independently reviewed; GPU-only path unchanged without experimental NPU.

**D4-F — Hierarchical memory:** central budget and admissions; local fast memory pool/lease/liveness obey global reservations and pressure. Don't allocate persistent weights/device graph per token. Prefix cache and expert tier have specialized eviction, one shared physical ledger.

**D4-G — AI breadth only through typed tasks:** a small TaskDriver and task-specific stage transitions handle text, embedding, ASR, TTS, image/video and **optional future gradients/training**. The memory/compiler core does not become a model training optimizer, workflow orchestrator or arbitrary external-tool host in first release. External tool calls belong above core API.

**D4-H — Portable first / specialized always optional:** source checkpoint/ModelDefinition and reference mathematics are authoritative; new hardware or model attaches via narrow capability interface. Runtime can decline unqualified kernel/shape/power and still function on generic backend where all registered semantic ops and physical memory suffice.

## 4. Performance budget and hot-path invariants

Performance-oriented design must be stated as measurable *budget categories* rather than invented universal microsecond numbers:

- **Hot dispatch:** cached `ExecutableBundle` maps shape/phase to compiled code handles and device-resident argument/index buffers; cannot deserialize arbitrary model/task/IR graph on each token. Count CPU instructions/host sync, allocation and enqueue per step.
- **GPU native work:** router count/selected IDs, grouped experts, state update and trained draft verification occur device-side when model/provider permits; host does not stage tiny expert-ID tensors every token in a guaranteed resident fast path.
- **Scheduling:** bounded StepPlan/diff update, pipeline CPU prepare and GPU execute, output transfer asynchronously where exact state dependencies allow; StepProgram can batch multiple device-native microsteps but remains transactional and cancellation-safe.
- **Memory:** no steady per-token host/device allocation for KV/slabs, no per-operator global arbiter synchronization; pre-reserve, reuse typed slots, deferred retire.
- **Shape:** symbolic rank/dimension bounds plus validated shape guards; avoid one kernel variant for every integer context length, bucket by cost and verified numeric range.
- **Compilation:** on cold model, compile only immediately necessary portable kernel/shape plan; optimize/tune hot regions asynchronously/explicitly after validation; no guaranteed instant compilation of exotic models.
- **Fused regions:** huge register/LDS kernels allowed but promote by full-task elapsed time and quality, not source elegance/launch count.
- **Concurrency:** evaluate C1/2/4/8/16/32 and task mix including audio/media with P95 and fairness; no blanket faster-on-every-model claim.
- **Energy/physical safety:** GPU+NPU shared LPDDR/power and SSD latency must be measured under same-clock ABBA; hardware unsafe profile disables optimization, never modifies privileged settings.

## 5. Minimality budget / architectural complexity constraints

**Required in first real vertical slice:** one semantic program representation, bounded verified state-effect graph, reference execution, one task driver, one kernel registry for a handful of ops, one device provider, a small command DAG/event lease verifier, one resource-admission manager, and one session owner/worker protocol. They can live in **2–4 Rust crates initially** (or one library + binary), not 9 mandatory crates.

**Explicitly deferred until tests justify it:** general workflow/agent DSL and external-action runtime; dynamic library plugin ABI; arbitrary offline equality saturation; distributed transaction WAL; automatic topology co-optimizer; every quant format; media modalities without model test fixtures; online training engine; JIT per new token shape; persistent cross-driver graph cache serialization.

**Minimum schema count:** one SIR semantic schema; one versioned state snapshot schema; one compiled executable plan representation whose cache can be dropped/rebuilt; one canonical TaskRequest/Result schema. RIR/KIR/CIR are logical compiler stages/views, not necessarily persisted standalone formats. External model artifacts are their original formats plus separately versioned optional derived packs.

**Software boundary:** all portability lies in Rust logical semantics and device/IO backend contracts. Never require an upstream model server; a 1-off fastest kernel should be expressible as a region candidate under existing runtime.

**No dynamic plugin system by default:** built-in static providers first; capability registry can be a Rust enum/table with function pointers/traits resolved once at model load, not a per-kernel dynamic dispatch zoo.

## 6. New required proof/counterexample program

| Test | What it proves | Rejection signal |
|---|---|---|
| **T4-01 Atomic state root** | fake KV+GDN+RNG branches prepare then one PublishRoot; failures leave original root intact | one domain mutates visible root while another fails |
| **T4-02 Device-native second step** | StepProgram executes two dependent GPU-native iterations without host forcing sampled token roundtrip when legal | scheduler must transfer every sampled token just to continue |
| **T4-03 Hierarchical lease** | central model/session reserve with device-local fast suballocator and async retire; no global allocation lock in hot path | buffer ABA or 1 global lock per op |
| **T4-04 Shape guard** | ragged/chunk inputs choose valid compiled shape variant or portable safe fallback | invalid kernel receives unsupported shape, infinite JIT churn |
| **T4-05 Non-token task reuse** | one lightweight TaskDriver handles encoder/iterative stage using same executor; no giant DSL | every stage requires a separate server or must report fake tokens |
| **T4-06 Training future seam** | synthetic parameter-update loop uses mutable ParameterVersion and gradient/optimizer state without editing inference kernel/session ownership | adding parameter effects forces rewrite of entire inference engine |
| **T4-07 Distributed fencing** | coordinator fails after rank prepared and before commit; old root/outputs remain valid, recovery/epoch fencing | ranks emit divergent user-visible results or stale retries commit |
| **T4-08 Output reconnect** | disconnect/retry with optional output seq cursor and zero uncommitted media/text leakage | duplicate unidentifiable emitted chunks or leaked speculative tails |
| **T4-09 Unfused path dominance** | zero-specialization path is fully functional and one extreme fused kernel replaces legal region without owning server | fused kernel forces copying sampler/resource manager |
| **T4-10 Compilation/cache stability** | shape/toolchain upgrades invalidate only necessary bundles and don't blow cold-start memory | stale graph pointers, hundreds of unused variants |
| **T4-11 Cross-run numeric fidelity** | source quant/compute precision and deliberate approximation tracked independently | route/top-k silently reduced for t/s |
| **T4-12 External tool boundary** | untrusted task graph cannot issue arbitrary process/network/file action through inference kernel or IR op | model tensor can trigger privileged tool calls |

These tests must become executable only when user authorizes implementation. Add to experiment and gates without inventing passing results.

## 7. Risks and unresolved policy, retained deliberately

- **Model scope:** “all things related to AI execution” means *composable supported AI programs*, not unrestricted execution of arbitrary unknown mathematics. Training/autograd/fine-tuning is a **future optional task capability**; the inference-first optimizer/recurrent/quant core must not prepay its design cost. Check user priority before treating training as an MVP deliverable.
- **Rust source:** backend kernels must be Rust-authored under stated user intent; whether external CUDA/ROCm/XRT driver, vendor math or non-Rust machine-code generator is allowed is open. C++/Python AIE kernel source remains research, not silently imported.
- **Lifetime:** no universal strict “host never OOMs” proof against unrelated system allocations. Guarantee admission under a measured conservative snapshot and best-effort pressure fallback, with clean device allocation failure handling.
- **Distributed correctness:** volatile single-coordinator publication is not durable exactly-once network delivery; don't promise durable distributed transactions until implementing proper log/ack/recovery.
- **Numerical:** bitwise replay can be guaranteed only under a *particular fixed plan/mode* where verified. Cross-GPU reductions can differ; the numeric contract specifies acceptable tolerances and quality, not silent approximation.
- **Pipeline universality:** task-specific loops and live reactive workflows can require new semantic op/control effects; stable primitives reduce but don't eliminate needed new code.
- **Legal scope:** project-neutral design and provenance remain separate. Verified MIT source may be considered for concrete source reuse with file-level notices and Rust-source approval; non-MIT upstream code is cited research only by user's project policy.
- **Performance:** even the best designed modular architecture may trail a hand-coded specialist on one workload. Specialized-region substitution is the mechanism to compete **without duplicating engine ownership**, not a promise of guaranteed outperformance.

## 8. v4 migration of authoritative specs

- **Rewrite** `MODULAR_AI_SYSTEMS_REFERENCE_ARCHITECTURE.md` as one-page-of-concepts normative top-level, linking deep contracts and drawing the correct 2-durable-representation architecture.
- **Rewrite** `COMPOSITION_AND_EXTENSION_CONTRACTS.md` to remove per-domain mutating `commit` and mandate prepared root publication, plus task driver/shape guard/provider and physical plan identity.
- **Revise** master plan, START_HERE, NEXT_AGENT_PROMPT, handoff to v4 authority/read order.
- **Add** precise execution/state protocol `MINIMAL_EXECUTION_SPINE_AND_PROTOCOL_V4.md`, and task/compiler `TASK_PROGRAM_AND_COMPILER_REFINEMENT_V4.md`.
- **Update** roadmap/Stage0/experiment/release gates and relevant subsystem specs with direct v4 proof tests and late-bound physical safety/precision.
- **Preserve** v3 originals in `DESIGN_EVOLUTION_V3_SNAPSHOT.md`, older v2 archive, all prior-art and license evidence. Research library remains read-only; update SHA-256 manifest **last**.
- **No Git initialization, kernel build or benchmark** is part of this architecture review.

## 9. Review verdict

The most consequential correction is **not another IR dialect or optimizer**. It is to make session state publication and CPU/device asynchronous execution *mechanically precise*, while reducing the number of **persistent abstractions** that runtime and future agents must maintain.

The architecture should ultimately have **one semantic program, one task driver, one executable plan with validated resource commands, one logical session-state publication authority, one physical budget authority, and many independent kernels/devices**. Fast paths are chosen at model/phase/shape boundaries and can cover entire fused regions, but cannot override mathematical or state ownership.

This increases the *chance* of achieving high performance **and** future modularity by allowing specialization to become a small replaceable component, not an independently administered runtime.
