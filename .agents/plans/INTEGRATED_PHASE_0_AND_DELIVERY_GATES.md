# zerogravity — Independent Stage-0 feasibility and delivery gates (v4 reviewed)

**2026-10-12 · NOT STARTED.** This is the project-neutral execution plan for architecture feasibility. It retains stable IDs **S0-00…S0-14** and optional NPU IDs **S0-NPU-A…F**; all remain PLANNED/NOT_STARTED. Normative math/task/kernel/resource contracts are in `../docs/MODULAR_AI_SYSTEMS_REFERENCE_ARCHITECTURE.md` and `../docs/COMPOSITION_AND_EXTENSION_CONTRACTS.md`.

## 1. Hard problem and mitigation

Two dangerous extremes are (a) a collection of fixed per-model GPU kernels plus a separate model-specific server that cannot grow to new algorithms, and (b) a large general IR/optimizer/framework that has never executed a real model efficiently. Our proof process deliberately combines **physical kernel feasibility**, **independent CPU reference and state truth**, a *minimal effect-aware graph compiler*, a *small async lease/event executor* and one **complete model+TaskProgram→output vertical slice**.

Do not freeze dozens of crate APIs or an exclusive kernel compiler before demonstrating the vertical slice. Do not defer kernel codegen until an enormous generic framework exists. Optional experimental accelerators may fail without preventing portable CPU/GPU success.

## 2. Stage-0 work packages

| ID | Work | Smallest concrete artifact | Independent pass gate |
|---|---|---|---|
| **S0-00** | Reproducible Rust baseline | lean Cargo CPU build, pinned dev shell/toolchain, source/provenance/license ledger, hardware probe | clean CPU tests, no unreviewed dependencies, correct device/memory inventory |
| **S0-01** | Physical native GPU kernel viability | Rust-authored CPU oracle and simple low-bit GEMV, prefill GEMM/WMMA and state update on target GPU | physical result, generated ISA, numeric/quality proof and measured gap to reference |
| **S0-02** | Second accelerator/codegen path | same shape/reference through different device/compiler kernel provider | compile+physical parity or honest missing/blocked capability, no model recipe edits |
| **S0-03** | Model artifact and source truth | GGUF/safetensors checked reader, tokenizer and quant codec, small real checkpoint | official/reference token, tensor and logits agreement; parser fuzz rejects bad files |
| **S0-04** | Small typed SIR with mutable effects | pure SSA + KV/one recurrent state operator, verify/Fork/Commit/Abort | illegal effect reorder rejected; canonical state output passes |
| **S0-05** | Minimal RIR and KIR registry | closed region fuse or row split, kernel candidate fallback, layout conversion | same math/state with optimized and generic variants, accurate error |
| **S0-06** | Async CIR and leases | fake CPU/GPU/IO executor with Acquire/Dispatch/Copy/Wait/Commit and timed events | invalid alias, stale generation, premature free, lost event/short IO fail closed |
| **S0-07** | Coordinator/worker protocol and state pools | EngineMsg/Event, StepPlan/WorkerDelta, paged KV and simulation | block conservation, no state cross-talk or global hot mutex, admission budget correct |
| **S0-08** | Model/task quality and transactional recovery | teacher-forced full logits, draft accept/abort, RNG and state replay reference | exact/tolerant modes differentiated; rejected speculative tail leaves no effects |
| **S0-09** | Rendering, grammar, service contracts | exact token/template/structured output and mock TaskRequest, streaming, cancellation | independent role/token vectors; no unverified provisional emission |
| **S0-10** | Immutable ExecutableBundle and plan cache | versioned direct kernel handles, stable graph/eager modes and capture proof | no stale captured address, no per-token SIR interpreter, valid forced fallback |
| **S0-11** | Sparse parameter placement simulator | layer/route/skew/hit/miss trace, optional cache/prefetch candidates and global headroom | no unsafe overcommit/eviction; measured cost rather than guessed optimal |
| **S0-12** | **Real end-to-end model vertical slice** | one checkpoint→ModelDefinition→SIR→RIR/KIR/CIR→CPU+GPU worker→committed streamed result | real model numeric/reference/quality, full state, CLI or SDK, forced missing fast kernel |
| **S0-13** | Multi-domain prefix and task composition | three different state domains, synthetic non-token TaskProgram stage loop and stage-specific output | exact common resume frontier, safe branches, no forced text token loop |
| **S0-14** | Optional additional portability and draft | alternative backend and optional zero-weight draft lookup/fallback | correct output/model state on supported backend; measured benefit or disable |

**Hard gates before freezing architecture:** S0-00, S0-01 (or documented physical specialization blocker), S0-03/04/05/06/07/08/12. **Composition should also be proved by S0-13** before claiming the system is generally modular. S0-02/09/10/11/14 may be staged when relevant, with all skips clearly recorded. The numbers are IDs, **not guaranteed calendar or engineer estimates**.

## 3. Optional Stage N — shared-SoC GPU+NPU and partial-operator execution

A specific coaccelerator may have its own device program format, memory import rules and **unresolved hardware/firmware power safety**. The NPU track never blocks GPU-only runtime.

| ID | Work | Pass / blocked condition |
|---|---|---|
| **S0-NPU-A** | Model-independent split-projection math proof and fake dual-device CIR | disjoint row spans, complete output order, quant/bias, first-use barrier, correct generic fallback |
| **S0-NPU-B** | Rust-owned NPU/AIE program/compiler and safe driver FFI | source-language and license policy satisfied plus compiled/validated shape/encoding; otherwise BLOCKED_BY_POLICY |
| **S0-NPU-C** | Physical SoC concurrency safety | dedicated isolated hardware and exact firmware/driver/fabric/clock evidence, normal boot/suspend/resume and long soak; hard hang blocks automatic enablement |
| **S0-NPU-D** | Whole-model/quality proof | full official model/logits/state, CPU/GPU-only vs GPU+NPU split and compute-precision error, exact sampling where claimed |
| **S0-NPU-E** | Matched end-to-end benchmarks | same model/checkpoint/quant and held physical clocks ON/OFF ABBA; TTFT/P95/memory/energy/idle/power, beneficial critical-path overlap |
| **S0-NPU-F** | New model/task/device extension | add second eligible projection operator and distinct encoder/task stage without touching model service and resource runtime; unsupported/unsafe defaults GPU-only |

An upstream-author prefill result is a **test target**, not proof. **No automatic privileged fabric clock, BIOS/IOMMU, NPU power-mode, runtime suspend or system service modifications.** A whole-machine hang is not recoverable by a software timer and the NPU path stays disabled without actual platform qualification. It is permissible and correct to ship GPU-only if the Rust source compiler or hardware safety gate cannot be passed.

## 4. Gates by architectural responsibility

### G0-A — Semantic and model foundations
- A real checkpoint's tensor/quant/tokenizer/config/template is parsed safely.
- A reference operator and independent model logit/continuation fixture prove correctness; novel op can't be silently guessed.
- SIR carries exact value shape, model position, quant/computation dtype and mutation effects; illegal reorders rejected.
- TaskProgram emits one correct output and supports a different synthetic bounded task stage with independent frontier.
- Unknown format/operator and malicious file rejected clearly without OOM or arbitrary code execution.

### G0-B — Kernel codegen and resource execution
- At least one physical GPU backend executes correct Rust-owned kernel(s); generated ISA/occupancy and build provenance stored.
- RIR produces one legal variant and fully correct fallback region from identical semantic model.
- CIR manages lease generations, async events/IO, alias and failure before committing state; graphical memory safety simulated and hardware smoke.
- Compiled ExecutableBundle is direct call/graph, no token-hot SIR/RIR/CIR interpretation or unnecessary memory copy.
- NPU/peer optional backend cannot change reference model, API or default GPU runtime when absent.

### G0-C — Functional end-to-end vertical slice
- Real model file/quant and tokenizer→typed ModelDefinition and TaskProgram→legal regions→portable CPU plus one validated GPU→user result, with exact state and fixed plan.
- Forced disabling a fast kernel does not cause another serving engine or broken session state; performance tradeoff recorded.
- TaskCoordinator owns request/commit frontier and DeviceWorker owns device events; no premature output.
- Physical resource headroom accounts OS/cgroup, GPU virtual address vs real RAM, scratch and in-flight buffers.

### G0-D — Modularity proof
- Add second model/novel op by extending reference op and model config only, not rewrite runtime service or allocator.
- Add second hypothetical accelerator provider with identity/shape/capability and fallback without modifying SIR model math.
- Add non-token task (e.g. encode→pool or mocked denoise loop) through same resource/worker/TaskProgram infrastructure without a decoder-only workaround.
- Code churn and test coverage recorded; new model-specific duplicated server is a **FAIL**.

## 5. Later release gates (capabilities, not promised dates)

**G1 Portable useful:** model support matrix, CPU+one GPU, safe local service/tokenizer/tool/grammar, complete output, source math validated and clear fallback/refusal. **G2 Production scheduler/state:** physical resource arbiter, multi-session token/other stage budgets, 3+ state-group prefixes, no use-after-free, P95 under load. **G3 Specialized/heterogeneous:** complex recurrent/sparse/MoE, expert tiers, optional NPU/two-GPU work, source quality and safety gates. **G4 Broader AI tasks/distribution:** embeddings/rerank, ASR/TTS/image/video and other TaskPrograms per model, topology/transport fail-closed, long-run reliability/security and independent benchmarks. Each model/task/hardware cell must be individually qualified.

## 6. Evidence and kill/pivot criteria

Per experiment/phase record commit/source, actual machine topology (CPU/GPU/NPU/physical RAM/storage), checkpoint+quant, tokenizer/semantic schema, compiler/device program source and flags, driver/power/FCLK for shared SoC, input/output task geometry, numerical/state tests, failures, per-request p95, energy and retry/abort correctness. User-specific output privacy preserved.

**Pivot/stop:**
1. Semantic IR cannot represent a recurrent/media loop without hidden mutation → fix model/TaskProgram effects before kernels.
2. Rust GPU/NPU kernel backend cannot emit correct/competitive instructions → defer extreme plan, keep portable model functionality.
3. CIR causes per-token interpreter overhead → compile static direct commands and shrink abstraction.
4. Physical memory/lease/IO/clock unsafe → prevent serving affected profile.
5. Specialized plan changes source model route/top-k/quant/sampling without opt-in → reject variant.
6. An extension requires rewriting unrelated server/allocator/model core → fix capability boundary.
7. Optimization gives no repeated whole-task gain or worsens P95/quality → retain simpler policy.
8. Source is non-MIT and named-project-dependent, or file/license rights unverified → preserve idea as attributed prior art only, no automatic source import.

**Closure:** no phase or gate becomes VERIFIED from documents. Update status/decision/handoff/manifest only after actual code, tests and physical evidence. The prior-art archive preserves older experiments and named sources, but the canonical acceptance criteria are these independently expressed gates.

## v4 correction and dedicated proof gates — 2026-10-12

**Authority:** [Critical review](../prior-art/ARCHITECTURE_V4_CRITICAL_REVIEW_2026-10-12.md), [execution/state](../docs/MINIMAL_EXECUTION_SPINE_AND_PROTOCOL_V4.md), [compiler/task](../docs/TASK_PROGRAM_AND_COMPILER_REFINEMENT_V4.md). Stable S0-00…S0-14 and S0-NPU-A…F remain PLANNED; **S0-V4-A…J** are *review closure tests*, not extra compulsory crates or a new entire implementation phase. Fold their smallest proofs into P04–P09 during the first vertical slice; later tests are deferred appropriately.

| Review gate | Why | Minimum convincing proof |
|---|---|---|
| **S0-V4-A — single root publication** | Fix per-domain state commit leak | fake KV+GDN+RNG all Prepare then **one** PublishRoot, failure in second domain leaves old root fully readable, no client output |
| **S0-V4-B — device-native hot loop** | Avoid mandatory host sync | on supported GPU run two causally dependent provisional target steps/rows with device routing/sampler index buffers, same reference state, no forced sampled token readback; record host wait cost |
| **S0-V4-C — hierarchical allocator** | Remove global per-token allocator lock | global reserve→local device subpool→generation lease→event retire, fault/stale generation tests; O(1) frequent allocations, no central mutex per op |
| **S0-V4-D — symbolic shape guards** | Prevent invalid kernels/JIT explosion | rank fixed with bounded dynamic dims, ragged offsets and sizes 1,7,128,4096; eligible kernel or deterministic safe fallback without out-of-bounds |
| **S0-V4-E — non-token TaskDriver** | Prevent embedded workflow DSL | trusted built-in encoder/pool or mock denoise loop through existing TaskCoordinator/DeviceWorker, output frontier not token |
| **S0-V4-F — specialization containment** | Prove no parallel engine | closed whole-layer region kernel can be enabled/disabled with identical model/task/server/state/physical budget ownership |
| **S0-V4-G — output delivery semantics** | Avoid promise of exactly-once network | output sequence IDs after root publish, slow consumer credits, disconnect/resume optional replay/dedup test, no provisional leak |
| **S0-V4-H — distributed epoch fencing** | Fail closed after partial rank prepare | fake remote peer timeout/lost ACK/stale epoch; coordinator does not publish new root until correctly prepared and fenced; no false durability guarantee |
| **S0-V4-I — optional training upgrade seam** | Broad AI execution without core training bloat | synthetic ParameterVersion+GradientProgram/optimizer effect can be represented in SIR/TaskDriver without mutating pinned inference sessions; real training deferred |
| **S0-V4-J — compilation/cache minimality** | Avoid four durable IR formats and cold compile storms | SIR+ExecutableBundle authoritative, transient region/kernel views, no raw driver pointers persisted, cache LRU/driver invalidation and generic shape fallback |

**Architecture freeze minimum:** V4-A/C/D/F/J plus core S0-00/03/04/05/06/07/12 should accompany the first real model. V4-B/E should be demonstrated before announcing high-performance asynchronous and cross-task composition as fully proven. V4-G/H/I can be implemented at service, distribution and future-learning phases respectively, without holding up the first GPU reference engine. **All are NOT_STARTED**.

**Known constraints:** GPU memory pressure cannot be perfectly predicted on a multiprocess OS; an OS OOM due to unrelated work is not proof that strict physical safety was guaranteed. Device loss may leave DMA live; quarantine, don't recycle. Cross-rank logical root publication and network output ACK are *not* automatic durable distributed transactions.

**Stable read order:** the old subphase gate text remains valuable background for quantitative tests, but where it says independent state domain `commit` or forces every TaskProgram into a huge DSL, these v4 gates and their owner contracts supersede it.

## v4.1 optional Stage Fabric — graph-only, wafer/static dataflow and heterogeneous cluster execution

**Not a prerequisite for the portable CPU/GPU S0 vertical slice.** Existing S0-00…S0-14, S0-NPU-A…F and S0-V4-A…J stay unchanged. The new **S0-FABRIC-A…H** are design-proof/future-hardware gates for one **RegionExecutionProvider** capable of different *minimum efficient region scales*, from native single kernel to static compiled wafer-scale program. All are NOT_STARTED.

| Gate | Smallest falsifiable proof | Pass vs blocked |
|---|---|---|
| **S0-FABRIC-A — graph-only provider** | fake target that accepts only closed whole-model stages, returns one AOT compiled executable and event; same source SemanticProgram/TaskDriver/StateRoot | correctness and no new server; actual TPU/AOT qualification remains blocked without vendor compiler/hardware |
| **S0-FABRIC-B — static dataflow fabric** | simulator for multi-core statically scheduled SRAM pipeline + streamed immutable model weights; compile/load cost, one completion event | tensor/reference parity, real weight/memory critical path, no host issue-per-op; actual wafer performance unverified |
| **S0-FABRIC-C — safe state bridge** | fake graph provider with private provisional KV/recurrence/adapter state, Prepare+PublishRoot, fork/snapshot/recompute; also opaque in-place state without rollback | first passes, second **must be rejected** for speculative stateful reuse (stateless encoder allowed) |
| **S0-FABRIC-D — shape/program lifecycle** | static and bounded dynamic shape guards, compilation/cache/version fingerprints, cold source fallback and no token-by-token graph recompilation | cache bounded, source semantics/quant correct, no stale program handles |
| **S0-FABRIC-E — topology/fabric** | mock 2×GPU / compiler-managed interchip collective / programmed fabric with link cost and fault/lost ACK | correct placement/ready/rank fencing, cannot publish mismatched root or assume aggregate memory pointer |
| **S0-FABRIC-F — coherent vs HBM and low precision** | device capability/quant mock for coherent local ARM+GPU, discrete HBM and rack topology; distinct NVFP4/MXFP4/other block layout | math/codec/shape and physical resident/OS budget correct; real device benchmarks separate |
| **S0-FABRIC-G — opaque service boundary** | a fake text-only remote model API with no tensor/state bridge and untrusted checkpoint provenance | cannot masquerade as native RegionExecutionProvider, only explicit permissioned TaskServiceAdapter with limited guarantees |
| **S0-FABRIC-H — extension and Rust/toolchain policy** | attach two new provider kinds and program liveness variants while existing ModelDefinition/TaskDriver/TaskCoordinator/ResourceArbiter remain unchanged | no new server, vendor compiler authorization/license evidenced; missing compiler reported unsupported not qualified |

**Release/maturity ladder:** `ARCHITECTURE_COMPATIBLE → SDK_FEASIBLE → COMPILES → FUNCTIONAL → STATE_SAFE → PERFORMANCE_QUALIFIED → PRODUCTION_QUALIFIED`, each tied to a specific region/model/task/quant/driver/shape/topology. Mock backend qualifies **architecture only**, never a real TPU/wafer or Blackwell program. Native GB10/B200 GPU paths can be real P01/P07 device qualification if hardware available; graph/wafer/Trainium style requires authorized vendor compiler plus state/quant/shape tests.

**Main gating rule:** stateful full-model program must provide privately prepared versions or safe snapshot/recompute. A compiler that only exposes opaque in-place state, or an API that returns text, is not an interchangeable region backend. No source-unknown kernel copied, no unapproved foreign-language compiler/FFI, no new runtime servers. See `../docs/ACCELERATOR_EXECUTION_FAMILIES_AND_BACKEND_CONTRACT.md` and separately sourced `../prior-art/ACCELERATOR_PRIOR_ART_AND_PLATFORM_RESEARCH_2026-10-12.md`.

**Phase mapping:** P01/P05 for RegionExecutionProvider + lowerer, P07 DeviceBackend program lifecycle, P08 POC graph fake, P10 memory/resident program, P16 cost+shape, P17–18 compiled task stages, P19 topology/collective/fabric, P20 compiler program cache/tuning, P21 vendor/trust/privacy, P22 physical performance, P23 new-accelerator extension drill. Existing task/runtime ownership remains unchanged.

## v4.2 optional Stage Distributed — topology, remote state and tunnel execution

**S0-DIST-A…L are NOT_STARTED architecture/transport proof gates, not additional mandatory crates or blockers for the first CPU/GPU end-to-end model.** They complement original S0-00…14, optional NPU A…F, v4 A…J and Fabric A…H. The authoritative rules are [distributed placement](../docs/DISTRIBUTED_PLACEMENT_AND_NETWORK_EXECUTION_CONTRACT.md) and [remote peer/state handoff](../docs/REMOTE_PEER_TRANSPORT_AND_STATE_HANDOFF_PROTOCOL.md). Names from current research remain [prior-art only](../prior-art/DISTRIBUTED_PRIOR_ART_AND_TUNNEL_RESEARCH_2026-10-12.md).

| Gate | Objective proof | Honest limit |
|---|---|---|
| **S0-DIST-A — topology and resources** | fake 3 heterogeneous nodes, measured-link object with directional bandwidth/RTT/P95/direct-vs-relay, separate compute/parameters/mutable state/immutable cache, no new IR | simulator doesn't qualify physical tunnel |
| **S0-DIST-B — layer state colocation** | two layer groups on different fake workers, each owning correct per-layer KV/recurrent state, stage activation only; C1 serial cost vs prefill pipeline | no universal multi-node speedup |
| **S0-DIST-C — independent attention/FFN and cache** | remote attention executes on KV owner, FFN elsewhere; compare one-time cache hydration and naive remote KV reads with modeled bandwidth and state correctness | slow tunnel may reject plan |
| **S0-DIST-D — remote prefix and full hybrid state** | immutable cache hint verified by source/quant/position/tenant and all state domains, transfer/recompute choice and old snapshot root correctness | directory hint isn't mutable owner |
| **S0-DIST-E — P/D handoff** | prefill worker complete KV/GDN/indexer/grammar/adapter state to decode worker after resource reserve; failure rollback and TTFT link cost | ordinary KV-only handoff insufficient |
| **S0-DIST-F — sequence KV partition algebra** | ordinary masked/causal attention shard max/Z/weighted numerator merge, all-masked/numerical edge tests; selected sparse and GDN refuse without op-specific proof | no universal recurrence split |
| **S0-DIST-G — trusted peer enrollment** | no unauthorized peer executes/receives models/prompts/weights, no remote shell, scoped ID/tenant/model/lease; program integrity/rights checked | auto-discovery ≠ consent |
| **S0-DIST-H — transport paths** | fake direct LAN vs encrypted tunnel direct/peer-relay/relay RTT, bandwidth/crypto/packet loss; rejects impossible GPUDirect over overlay | physical tests later |
| **S0-DIST-I — remote coordinator fencing** | duplicate/out-of-order Ready, wrong checkpoint/epoch, old coordinator after partition; exactly one writer/root output or fail-stop | no durable leader election assumed |
| **S0-DIST-J — state shard migration** | reserve/hydrate complete compatible state → fence old owner → one root placement epoch change, or old root preserved on failure; live DMA quarantined | not arbitrary mid-GPU step migration |
| **S0-DIST-K — mixed task stage split** | encoder/ASR/image stage plus text task across fake peers with task-specific output frontier, correct stream/privacy, no separate server | real codecs/model quality separately |
| **S0-DIST-L — real two trusted machines** | physically run one reference-checked model over enrolled direct authenticated link, then explicitly tested private tunnel; record model/quant/state, P95, memory and failure | BLOCKED until user authorizes devices/network and code |

**Initial distributed implementation order:** simulator A/B/I/G, optional one real D-LAN layer split L, then D/E remote-prefix+P/D, then C/F remote attention/sequence shards, and multi-peer topology search only after measured need. A 2-node private cluster is sufficient proof; no mandatory DHT, container orchestrator, cluster-wide consensus DB or generic workflow engine.

**Safety:** do not connect or enroll user's machines without explicit authorization. Tunnel doesn't confer user data permission or hardware RDMA. Peer cache replicas are immutable location hints; mutable StateShardOwner carries lease+epoch. Any network partition during a provisional commit must fail closed or reconstruct from safe committed history. Simulated success is **ARCHITECTURE_SIMULATED**, not physical state/performance qualification.

## v4.3 optional Stage Serve — minimal CLI, model router and API conformance

**S0-SERVE-A…J are NOT_STARTED proof gates.** They add no new implementation phase IDs and do not block the original single-model CPU/GPU vertical slice. They validate that built-in multi-model serving can be added through existing TaskCoordinator/ResourceArbiter/RegionExecutionProvider with *no extra inference engine*. Source-attributed comparisons in `../prior-art/MULTI_MODEL_ROUTER_AND_API_PRIOR_ART_RESEARCH_2026-10-12.md`; normative [router](../docs/MODEL_ROUTING_RESIDENCY_AND_MULTI_MODEL_SERVING_CONTRACT.md), [CLI/config/API](../docs/ZERO_CONFIG_CLI_CONFIG_AND_OPENAI_API_CONTRACT.md).

| Gate | Minimum convincing test | Unqualified/incorrect if |
|---|---|---|
| **S0-SERVE-A — zero-config local `run`** | one real supported local checkpoint, no config/server/cloud/account, trusted artifact/quant and committed terminal output | mandatory daemon, arbitrary source download or a second engine |
| **S0-SERVE-B — catalog and source binding** | two exact immutable model source hashes, alias/model-profile revisions, tensor/adapter/tokenizer/task validation | alias update changes a live StateRoot |
| **S0-SERVE-C — safe on-demand load** | fake model load with actual reservation/ready lifecycle, 100 concurrent same-model requests deduplicated, bounded waiter queue and timeouts | 100 duplicate GPU allocations or infinite waits |
| **S0-SERVE-D — multiple physical residents** | 2–4 fake models/real small compatible models share one physical arbiter; context/SSM scratch counted, idle unload/event retire safe | virtual GTT counted as RAM or unsnapshotable active session evicted |
| **S0-SERVE-E — equivalent replica routing** | 2 same-artifact replicas, prefix-affine sticky session, queue/fairness, resource/deadline and safe full StateRoot migration | request routed to different source quant/model or stale state |
| **S0-SERVE-F — per-model and request config** | global, model, alias, CLI, authorized request and hard policies resolved deterministically; effective provenance and conflicts shown; invalid keys rejected | request overrides security/quant/resource or silent ignored setting |
| **S0-SERVE-G — OpenAI SDK chat/model** | one loopback `GET /v1/models`, chat nonstream and SSE/usage/finish/tool data via pinned current Python+JS SDK; exact model identity and semantics | wrong stream schema or fake compatibility |
| **S0-SERVE-H — task/API subsets** | real supported embeddings model/pooling and stateless Responses events as separate conformance cells when implemented; stateful store/tool cloud fields explicitly refuse until qualified | claim endpoints work with no qualified TaskDriver |
| **S0-SERVE-I — local vs external security** | loopback default, no automatic external bind, admin model management restricted, public auth/TLS required and scoped model visibility | unauthenticated remote admin or arbitrary model/media/tool code |
| **S0-SERVE-J — distributed model/replica distinction** | one model sharded across two fake peers, two replicas of exact same source and two distinct source models; router admission and placement/state/tenant budget correct | model shard mistaken as replica, unauthorized remote transfer, second model server |

**Early core proof:** A+B+C+F+G can be integrated during P08/P09 once one model genuinely runs; D+E at concurrency/residency P10/P16, H with TaskDriver breadth P17/P18, J after distributed mock v4.2. Local `run` shall have no mandatory HTTP router/proxy startup or per-token JSON overhead. **All gates are design tests only until actual Rust model/device/API code exists.**

**Safety:** no downloaded model/source, public listener, service manager, account/tunnel setup or network permission without user instruction. Profile and model-selection changes cannot reduce original checkpoint/quant/target sampling quality implicitly.
