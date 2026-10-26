Author: ROOT documentation coordinator; design/context lane
Co-authors: GPT-6.1 Sol xhigh
Owner: @pvl; ROOT reconciles this handoff

# Design, decisions and original-source reading catalog — 26 October 2026

Documentation-only continuation. This chapter preserves the project's full
design trajectory and tells a successor where the original bodies live. It is
not a replacement implementation queue, a claim that every design has shipped,
or a new mandatory archive/receipt workflow. No runtime, build, model, device or
benchmark was executed for this chapter.

The current [AGENTS](../../../AGENTS.md), [plan rules](../../plans/README.md),
[queue](../../plans/TODO.md), live source and actually executed results govern.
Historical status sentences and embedded commands retain their original scope.
The coordinator's main handoff and companion chapters supply current source,
worktree, verification, user-decision and open-work facts.

## 1. Complete reading route and authority

Read in this order; follow complete linked bodies when a selected task depends
on them. An index or a truncated tool response cannot substitute for the source.

1. Current AGENTS → plan rules/queue → the owning plan's TODO. Current user
   instructions outrank old prompts, model assignments and process ceremonies.
2. [Master v4.3](../../plans/ZEROGRAVITY_MASTER_PLAN_2026-10-10.md),
   [whitepaper](../../../docs/WHITEPAPER.md) and
   [engineering specification ES-00…15](../../../docs/ENGINEERING_SPECIFICATION.md).
   The master filename is dated10Oct, but its body is the later v4.3 synthesis.
   Whitepaper/ES initial “unimplemented” paragraphs are manuscript-date status,
   not a current all-project status statement.
3. Actual archived [decision register](../../plans/archive/history/DECISION_REGISTER.md),
   [traceability](../../plans/archive/history/REQUIREMENTS_AND_TEST_TRACEABILITY.md),
   [P00…23 playbook](../../plans/archive/history/AGENT_IMPLEMENTATION_PLAYBOOK.md),
   [roadmap](../../plans/IMPLEMENTATION_ROADMAP.md),
   [Stage0](../../plans/INTEGRATED_PHASE_0_AND_DELIVERY_GATES.md),
   [experiments](../../plans/EXPERIMENT_PROGRAM.md) and
   [release gates](../../plans/RELEASE_ACCEPTANCE_GATES.md).
4. [Modular architecture](../MODULAR_AI_SYSTEMS_REFERENCE_ARCHITECTURE.md),
   [minimal spine](../MINIMAL_EXECUTION_SPINE_AND_PROTOCOL_V4.md),
   [composition contracts](../COMPOSITION_AND_EXTENSION_CONTRACTS.md) and
   [task/compiler refinement](../TASK_PROGRAM_AND_COMPILER_REFINEMENT_V4.md).
   They resolve older ambiguous commit authority and permanent-IR sketches.
5. Current [shared ForwardPlan architecture](../../plans/pipeline-composition/SHARED_EXECUTION_ARCHITECTURE.md),
   [regions/profiles](../../plans/pipeline-composition/REGIONS_AND_PROFILES.md),
   [profile/reuse policy](../../plans/pipeline-composition/PROFILE_POLICY_AND_COMPONENT_REUSE.md),
   [structure/naming](../../plans/pipeline-composition/STRUCTURE_AND_NAMING.md),
   [formats](../../plans/pipeline-composition/FORMATS.md) and
   [all-model correctness policy](../../plans/numerics-and-context/CORRECTNESS_POLICY.md).
6. Read the subsystem bodies in §6 and their own source TODOs. Use §7/§8 to
   connect integrated ES contracts and original phases to those current owners.
7. For why the design changed, read [17Oct](../FULL_COMPREHENSIVE_AGENT_HANDOFF_2026-10-17.md),
   [19Oct](../FULL_COMPREHENSIVE_AGENT_HANDOFF_2026-10-19.md),
   [20Oct](../FULL_COMPREHENSIVE_AGENT_HANDOFF_2026-10-20.md),
   [25Oct AM](../FULL_COMPREHENSIVE_AGENT_HANDOFF_2026-10-25.md),
   [25Oct PM](../FULL_COMPREHENSIVE_AGENT_HANDOFF_2026-10-25-PM.md) and
   [25Oct NIGHT](../FULL_COMPREHENSIVE_AGENT_HANDOFF_2026-10-25-NIGHT.md).
   Their differing status is a dated chronology, not interchangeable acceptance.
8. The [20Oct context catalog](../handoff-2026-10-20/CONTEXT_CATALOG.md) locates
   every body inside the existing sourcebooks and recovered-history volumes.
   [History/design audit](../handoff-2026-10-20/HISTORY_DESIGN_AND_COVERAGE_AUDIT.md)
   describes positive recovery and honest limits. The present source inventory
   is in §11; the coordinator's `SOURCE_CATALOG.md` supplies preservation mapping.

Original bodies outside the two requested agent roots remain authoritative:
top-level `docs/`, `.agents/research/` and `.agents/prior-art/`. For this explicit
handoff request, ROOT preserves byte-identical dated copies under
`reference-sources/publication/`, `reference-sources/research/` and
`reference-sources/prior-art/` in this directory. Those are immutable snapshots,
not additional independent authorities or live work queues. Leave originals,
sealed source packages, existing archives and the user-owned root README intact.

## 2. Original ambition — retain the whole product, constrain claims

Zerogravity is intended to be one Rust-owned AI execution fabric and simple
native binary for text, image, video, audio, real-time and combined multimodal
inference. It seeks both broad useful execution and extreme source/hardware
specializations. The user wanted vLLM/SGLang-class serving efficiency with
accessible local use, followed by measured superiority where achievable.
There is no approved universal speed percentage or deadline. A source model
name, folder, parser, feature or architectural diagram proves no inference cell.

| Ambition | Contract and boundary |
|---|---|
| Broad model families | Llama/dense, Qwen3.5+ dense/hybrid/MoE, Gemma4; original later priorities include Qwen3.8-27B and Gemma4 E2B/26B-A4B, other Gemma variants, Qwen Flash-Next~125B, GLM Flash~321B, DeepSeek and further task families. Audit the exact released architecture/revision; conversational names/approximate sizes are not tensor geometry. |
| Portable plus specialized execution | CPU reference and tuned generic operations; native CUDA, ROCm(HIP), Vulkan and eligible future Metal/NPU/ASIC paths; whole-layer/model specialization is legal through a closed region and existing owners. Missing semantics cannot acquire a fallback by reading a container. |
| Hardware breadth | Coherent APU/CPU-GPU, discrete VRAM, rack HBM/collectives, GPU+NPU, AOT graph and static-dataflow/wafer devices at natural granularity. Presence, SDK feasibility, compilation, physical safety and production support are distinct. |
| Efficient serving | One library for direct CLI/SDK/API; exact aliases and compatible replicas; fair continuous batching, chunked prompt processing, full-state prefix reuse, speculative target verification, bounded active/output resources and useful complete-request goodput/p95. Request batching and temporal prefill are different mechanisms. |
| Composed pipelines | YAML typed stages and nested subpipelines; ASR/diarization→LLM/decision/filter→TTS, encoder/rerank, image/video and coupled AV. Existing graph/config/context scaffolds are not native media inference. Application tools and side effects remain permissioned above inference. |
| Tiered memory and large models | Immutable GPU/UMA/RAM/SSD tensor backing, routed experts/lookup rows, private state/COW and admitted prefetch. Source identity survives materialization/eviction. SSD may enable capacity while losing latency. Addressability/GTT is not free physical RAM. |
| Multiple devices and machines | Compute, weights, mutable state and immutable caches can have independent placement. Transfers, topology/epochs, rights and real cost are explicit. An authenticated tunnel is not RDMA; capacity benefit is not C1 throughput benefit. |
| Simple UX | Zero mandatory daemon/config/cloud for a supported local model; `run`, `serve`, inspect/explain, explicit device constraints, XDG paths, optional advanced config. No silent source/quant/model/backend change or implicit download. |
| Product/release | Rust/Nix/crane, CPU-minimal builds, portable installers, signed/provenance-aware releases and a tested OS/hardware/model matrix. Six cross-build definitions or a Linux binary do not qualify six native platforms. |
| Future training | Optional GradientProgram/TrainStep/ParameterVersion seam; inference remains bound to immutable model versions. A mandatory trainer/autograd/cluster stack was deliberately deferred. |

Canonical intent: [vision](../VISION_SCOPE_AND_PRINCIPLES.md), master §§1–15,
register U-001…U-022 and whitepaper §§1/11–13. Full source/quant/tokenizer/task,
operator/model accuracy, state/physical safety, serving and performance each
need their own actual evidence. Finite proofs, synthetic fixtures and partial
adapters must stay labeled as such.

## 3. Design chronology and why each refinement exists

Dates below describe document lineage. Some imported Git timestamps are skewed;
use real ancestry and source IDs for implementation chronology. This table
does not manufacture execution dates or reconstruct missing Git objects.

| Stage | Original concern | Retained result / supersession | Complete source |
|---|---|---|---|
| V1,10Oct | Rust-native semantic engine, ordinary and extreme models, general fallback, source formats/state, accessible CLI plus peak serving. | Mathematics must survive optimized lowering. Broad support and faster-than ambition began as research, not achieved benchmarks. | Register U-001…U-015; [17Oct original-vision chapter](../FULL_COMPREHENSIVE_AGENT_HANDOFF_2026-10-17.md); recovered V1 inside [RECOVERED_HISTORY_01](../handoff-2026-10-20/RECOVERED_HISTORY_01.md). |
| V2 | Combine separate research/compiler programs; SIR math/effects, region transformations, kernel candidates and asynchronous resources. | One model recipe; hardware policy, compiler and per-cycle StepPlan are different responsibilities. Old model-owned `kernel_plan(hw)` is not the final placement authority. | [Reconciliation](../../prior-art/ARCHITECTURE_RECONCILIATION_AND_AUTHORITY.md), [compiler research](../../prior-art/COMPARATIVE_COMPILER_AND_RUNTIME_RESEARCH_2026-10-10.md), register P-018…P-025. |
| V2.1,11–12Oct | Same-SoC GPU+NPU row-split prefill and multiple task kinds. | Independent projection output rows may be placed cooperatively; source versus temporary compute encoding, join semantics, and real power/fabric hang risk stay explicit. Different from dGPU/iGPU expert placement. | [NPU/task contract](../NPU_GPU_COOPERATIVE_EXECUTION_AND_TASK_PIPELINES.md), [Gufo/XDNA2 research](../../prior-art/GUFO_FORK_AND_XDNA2_PREFILL_SOURCE_RESEARCH_2026-10-12.md), P-026…P-030. |
| V3,12Oct | Original project-neutral model/op/task/region/device/resource interfaces instead of upstream-named engines. | External work remains attributed research; no separate per-model server. Earlier domain commit, workflow and IR proliferation sketches were narrowed next. | Register U-017/U-018/P-031…P-035; recovered V3 bodies in [RECOVERED_HISTORY_01](../handoff-2026-10-20/RECOVERED_HISTORY_01.md). |
| V4 | Reduce the architecture to a minimal fast spine. | Two enduring representations; five owners; domains prepare, coordinator alone publishes; coarse physical admission/local pools; trusted TaskDriver; real event retirement. RIR/KIR are transient, CIR belongs in physical execution. | [Minimal spine](../MINIMAL_EXECUTION_SPINE_AND_PROTOCOL_V4.md), [task/compiler](../TASK_PROGRAM_AND_COMPILER_REFINEMENT_V4.md), [critical review R4-01…24](../../prior-art/ARCHITECTURE_V4_CRITICAL_REVIEW_2026-10-12.md), P-036…P-045. |
| V4.1 | Kernel-only provider model cannot express TPU/AOT/static fabrics efficiently. | One RegionExecutionProvider accepts kernel, captured graph, AOT graph or static dataflow at natural closed-region size. Opaque hosted inference is a separate service adapter. StateBridge is a hard gate. | [Accelerator contract](../ACCELERATOR_EXECUTION_FAMILIES_AND_BACKEND_CONTRACT.md), master §12, U-019/P-046…P-055. |
| V4.2 | First-class distributed execution, not merely remote API routing. | Four orthogonal placement maps plus transfers, authenticated peers, bounded credits, epochs and one root publisher. No instantaneous physical transaction, auto-enrollment, WAN speed illusion or transparent HA. | [Placement](../DISTRIBUTED_PLACEMENT_AND_NETWORK_EXECUTION_CONTRACT.md), [peer protocol](../REMOTE_PEER_TRANSPORT_AND_STATE_HANDOFF_PROTOCOL.md), master §13, U-020/P-056…P-070. |
| V4.3 | One-command local use and native many-model routing. | RegisteredModel vs Alias/Profile vs ResidentReplica vs TaskSession; exact source binding at admission, equivalent-replica balancing, independent roots and one execution engine. No per-token router/proxy. | [Router](../MODEL_ROUTING_RESIDENCY_AND_MULTI_MODEL_SERVING_CONTRACT.md), [CLI/API](../ZERO_CONFIG_CLI_CONFIG_AND_OPENAI_API_CONTRACT.md), master §14, U-021/P-071…P-086. |
| Publication dossier,12Oct | Coherent public design plus exact agent implementation contract. | Whitepaper is an argument/proposal; ES-00…15 and INV-01…25 provide acceptance vocabulary. APIs/snippets are illustrative, not frozen crate/ABI commitments. | [Whitepaper](../../../docs/WHITEPAPER.md), [ES](../../../docs/ENGINEERING_SPECIFICATION.md), register U-022/P-087…P-091. |
| Packaging→reference→GPU,12–17Oct | Small real source model first, then exact native regions/residency and one resident graph. | P00 bootstrap is distinct from inference; host/mock/physical/full-model/serving proof levels never collapse. U-027 deferred16K/32K-before-GPU while preserving applicable independent8K gates. | Archived [accepted baseline](../../plans/archive/history/accepted-baseline/TODO.md), U-023…U-056; original phase bodies in existing recovered volumes. |
| V5,14Oct | Text-only type/frontier assumptions fail on media and direct decisions. | Typed media clocks/values, verified component assemblies, distinct CTC/transducer/RVQ/flow/decision control, synchronized complete outputs, orthogonal codecs. Mock MM0 checks are interface proof. | [V5 contract MM-01…11](../UNIFIED_MULTIMODAL_EXECUTION_CONTRACT_V5_2026-10-14.md), [family atlas](../../research/MULTIMODAL_MODEL_FAMILY_SOURCE_ATLAS_AND_IMPLEMENTATION_2026-10-14.md). |
| V5.1,15Oct | Real-time/duplex, detailed assemblies, media QoS, source-specialized hybrid/SSD work. | Model commit, transport delivery, ACK and playback differ; session/turn/transport/physical epochs differ. SSD is a storage policy under one tensor/resource fabric. Approximation remains explicit. | [Universal V5.1](../UNIVERSAL_MULTIMODAL_EXECUTION_ARCHITECTURE_V5_1_2026-10-15.md), [realtime](../REALTIME_MULTIMODAL_SERVING_AND_QOS_V5_1_2026-10-15.md), [assemblies](../VERIFIED_MODEL_ASSEMBLIES_AND_SPECIALIZATION_V5_1_2026-10-15.md), [virtual tensor/SSD](../UNIFIED_VIRTUAL_TENSOR_RESIDENCY_AND_SSD_STREAMING_2026-10-15.md). |
| Layout/concurrency/acquisition,16–18Oct | Shared immutable models with independent sessions, clean workspace owners, explicit XDG acquisition. | Reusable probes moved to crates/tools; no second product. Cold-catalog left-right migration rejected absent real contention. Provider/cache publication is not ModelReady or runnable inference. | U-057…U-068; archived [register](../../plans/archive/history/DECISION_REGISTER.md), [acquisition spec](../../plans/MODEL_ACQUISITION_IMPLEMENTATION_SPEC.md), live acquisition TODO/archive. |
| 19–20Oct serving/numerics | Same-engine HTTP/SSE, native request batching, source-specific HF/GGUF profiles and deeper original-reference investigation. | Concurrent requests, physical multi-row kernels, temporal prefill and fully qualified serving remain distinct. GGUF AR1, default bulk, Torch CPU and vLLM CUDA are distinct numerical profiles. Later first-gap observations supersede simplistic GDN-only causality. | [19Oct handoff](../FULL_COMPREHENSIVE_AGENT_HANDOFF_2026-10-19.md), [20Oct numerical continuation](../handoff-2026-10-20/NUMERICAL_AND_MODEL_CONTINUATION.md), [GDN audit](../handoff-2026-10-20/GDN_VLLM_ARITHMETIC_AUDIT.md), [correctness-first plan](../../plans/numerics-and-context/CORRECTNESS_FIRST_VLLM_PLAN_2026-10-20.md). |
| 21Oct recursive composition/reuse | Primitives→nested regions/tasks→workload profiles; efficient resident stage edges and reusable components. | Nesting is metadata/composition, not another interpreter. Explicit approximate profiles; replaceable pools/allocators preserve one accounting authority. Reuse suitable audited MIT/Apache Rust/static C ABI/native components, else faithful attributed ports. | [Regions/profiles](../../plans/pipeline-composition/REGIONS_AND_PROFILES.md), [profile/reuse](../../plans/pipeline-composition/PROFILE_POLICY_AND_COMPONENT_REUSE.md), [updated provenance policy](../../prior-art/PRIOR_ART_LICENSE_AND_PROVENANCE_POLICY.md). |
| 23Oct shared ForwardPlan review | Family recipes in shared core, source-specific heads, ownership duplication and finite backend eligibility. | One family mathematical traversal records/evaluates common operations. Share real source/window/head/storage/transport/physical ownership; bind request authority late. Profiles cannot mint kernel permission. | [Shared execution architecture](../../plans/pipeline-composition/SHARED_EXECUTION_ARCHITECTURE.md), [memory boundaries](../../plans/cpu-gpu-optimization/SHARED_MEMORY_AND_BOUNDARIES.md). |
| 24–25Oct CLI/backend/B4 | Actual selection policy, physical device identity, source/ABI/refactor and B4 certification/C16/x2 sequence. | Parser/selector support does not create a backend/model route. Driver support nodes are not selected GPU identities. New ABI/oracle/source changes need their affected evidence; old diagnostics do not qualify current source. | [CLI design](../../plans/cli-device-routing/README.md), [ABI/family refactor](../ABI_FAMILY_CLI_REFACTOR_2026-10-25.md), [Linux nodes](../LINUX_DEV_NODE_SELECTORS_2026-10-25.md), [NIGHT](../FULL_COMPREHENSIVE_AGENT_HANDOFF_2026-10-25-NIGHT.md). |
| 26Oct current rules | Model-owned semantics, meaningful names, actual removal of production qualification/testcase coupling; common correctness approach. | Keep tests/cases/comparison data in Rust test support or explicit diagnostic targets. Ordinary execution cannot depend on qualification tables. Do not relabel a whitelist as capability or add placement-enforcement tests. Accuracy, semantics/safety and determinism are separate. | Current [AGENTS](../../../AGENTS.md), [structure](../../plans/pipeline-composition/STRUCTURE_AND_NAMING.md), [formats](../../plans/pipeline-composition/FORMATS.md), [correctness](../../plans/numerics-and-context/CORRECTNESS_POLICY.md); current ROOT handoff/user-decision chapter. |

## 4. Decision taxonomy, complete ID continuity and supersessions

The archived [register](../../plans/archive/history/DECISION_REGISTER.md) retains
U-001…U-070, P-001…P-096, O-001…O-074 and A-001…A-006. `USER`, `PROPOSAL`,
working technical baseline, `OPEN` and assumption are different kinds of record.
Do not promote an assistant recommendation to user authorization. A supersession
changes the stated dimension, not every older source/math/ownership requirement.

| IDs | What must survive |
|---|---|
| U-001…U-015 | Lowercase name, Rust-owned engine, broad/extreme models, safe legal fallback, semantic IR, all-context records, no fabricated percentage/deadline, corrected canonical root and unified maintainable architecture. U-013 changes U-002's path. |
| U-016…U-022 | Cooperative prefill, project-neutral ownership/attribution, critical minimality review, natural accelerator granularity, distribution, built-in router/simple UX and publishable design. These requests did not themselves run hardware. |
| U-023…U-029 | Packaging-only first; explicit later implementation; .8B/2B CPU→GPU practical order; independent numerical/context truth;16K/32K prerequisite deferral; parallel GPU/numerics/media streams. |
| U-030…U-040 | Mixed source codecs, persistent residency, static source/dynamic step split, async readback, uncertain-Future retention, global physical admission/fair workers, real Vulkan arithmetic and narrow review repairs, shared immutable model/independent sessions. |
| U-041…U-049 | Additive V5.1 scope, repository reconciliation, user-owned README, source GDN/attention progression, comprehensive history, retained replay/cooperative prefill and neutralized implementation narrative. |
| U-050…U-056 | Exact GDN nonlinear/RoPE/QK/softmax/value and remaining full-GPU continuation. Each operator stage retained incomplete whole-model boundaries; later bounded full-model acceptance does not relax the original math. |
| U-057…U-064 | First fleet, independent lanes/worktrees/coordinator ownership, structure and measured-contention concurrency pass, XDG acquisition, new-family source/CPU-first work, pinned Qwen parity/advanced decoding frontier, rejected cold-catalog migration. |
| U-065…U-070 | Action-specific publication authorization; rewritten historical labels vs reachable objects; simple plan/run TODOs replace mandatory archives/hashes; wrap/pause and explicit comprehensive documentation request. Later explicit resumes supersede only the old pause. |
| P-001…P-017 | Original portable/tuned/extreme, compiler/effects/state/cache/format/UX proposals. |
| P-018…P-030 | Unified compiler/execution, Stage0, heterogeneous compute/task research. |
| P-031…P-045 | Neutral V3 then minimal V4; two durable representations, one prepared root publisher, coarse resource admission, trusted control, shape guards and optional future training. |
| P-046…P-070 | Accelerator V4.1 then distributed V4.2 provider/state/transport/cost/qualification contracts. |
| P-071…P-086 | Router V4.3 exact identities/residency/config/API/minimality. |
| P-087…P-096 | Publication/agent delivery and lean packaging/release foundation. Former mandatory hash/archive process in P-091 is superseded. |
| O-001…O-022 | Original license/platform/FFI/toolchain/model/quality/API/config and product questions; later choices settle only the actual answered question. |
| O-023…O-041 | NPU/source/compiler/hang/task/reuse/training/delivery/allocator/plugin and graph/wafer/SDK/state/privacy/ambiguous accelerator-name questions. |
| O-042…O-066 | Remote topology/enrollment/privacy/transport/recovery/partition/rights/SLA and CLI/catalog/replica/profile/admin/residency/multi-role questions. |
| O-067…O-074 | Publication/byline/legal/channel and real platform/service/release qualification. Old “no measurements/no Git/no service” wording belongs to its date. |
| A-001…A-006 | A-001 path superseded; dated source research requires refreshed pins; external author measurements are not our results; fallback requires real semantics/resources; no ambient account/secret/publication authority. |

Important later rulings beside older prose:

- **MIT selected15Oct** for project source/docs, independently of imported
  files, models, codecs, datasets and SDK licenses. Old O-001/O-068 ambiguity
  must not re-open the selected project license or erase third-party rights.
- **YAML chosen later** for pipeline/serving config. Master/ES proposed TOML is
  historical, not current config authority.
- **Reuse-first21Oct** explicitly supersedes older from-scratch and Rust-only
  foreign-kernel restrictions. Suitable audited MIT/Apache components may use
  Rust or static native C ABI, including a narrow C++ shim; otherwise port the
  smallest useful implementation/fixtures faithfully. No second permission is
  required merely because a suitable component is externally authored.
  Production stays one native binary without Python/PyTorch/foreign server.
- **One owner per responsibility**, not a frozen crate count, allocator
  algorithm or single process-global scheduler thread. Cohesive crate splits,
  alternative pools and concrete native providers are allowed when they enforce
  actual boundaries. No extra independent resource or root authority.
- **Contention before concurrency machinery.** Functional left-right tests do
  not establish useful catalog contention or a speedup; that migration was
  rejected. Reuse old accepted evidence only for its unchanged applicable scope.
- **TODO simplification16–17Oct** supersedes generated handoff/SHA/receipt duties.
  This explicit comprehensive request authorizes dated documentation retention,
  not permanent per-helper administrative gates. Actual model/source/content
  integrity and numerical acceptance stay required.
- **Current26Oct accuracy policy** preserves exact declared contracts while
  separating operator/model accuracy from same-plan, batch, scheduling and
  cross-hardware determinism. Default upstream batching may differ numerically.
  Preserve strict historical failures; do not rename them into success.
- **Current26Oct qualification separation** is a real dependency/caller cleanup,
  not cosmetic names or a feature flag still enabled by production. Keep actual
  source geometry, kernel/image/ABI and physical ownership checks in execution;
  move assertions, cases, expected outputs and reference catalogs to tests or
  an explicit diagnostic. A finite captured route remains qualification support
  until general runtime support exists. No new convention/enforcement tests.
- **Current delegation** is GPT-6.1 Sol xhigh for coordination, architecture,
  implementation, execution and review. Historic Astra/Pi/GLM role settings and
  launch recipes do not override that. Batch substantial implementations and
  validate the assembled result; complete `scripts/check.sh` last. The current
  refactor waiver for full Qwen27B/Gemma suites grants no new qualification.
- Root README remains user-authored/byte-preserved. Current AGENTS still governs
  scoped signed commits, separate push/publication authorization, approved
  model/SDK/device actions and local Renoir/gfx90c hold. Old broad repository
  ownership or earlier cleanup approvals are not permission to overwrite peers.

Identifier namespaces must remain attached to their documents:

- Reconciliation **D-01…D-20** records council/research dispositions, not a new
  user signature on each option.
- Critical-review **R4-01…R4-24**, multimodal **DMM-001…032/OMM-001…022** and
  specialization **SYK** decisions/questions have their own status.
- Multimodal-local historical U-040/U-041 collide numerically with different
  canonical register rows. Cite the original ledger, never merge the meanings.
- Roadmap **ZG-P12** means routed sparse parameters; later serving **P12** means
  native batching. **H01…H86** design hypotheses differ from implementation
  H3/H7/H12/H25 stages. Latest user “3→4→2” is a work order, not P03/P04/P02.

## 5. Surviving execution contracts and current ownership

The architecture keeps five responsibilities: immutable source semantics,
immutable prepared execution, logical session publication, physical resource
admission, and actual device/IO execution. Task/service policies compose them.

```text
authenticated source/config + family-owned mathematical traversal
    → one ForwardPlan: values, exact operations/numerics and state effects
    → prepared compatible kernels/regions/graphs + storage/transfer lifetimes
    → late original attempt/root/frontier/row/lease bindings
    → original fair worker + physical ResourceArbiter + actual flight/events
    → Ready or failure/retained Unknown/quarantine
    → original SessionAuthority publishes one complete logical StateRoot
    → committed typed output; transport/ACK/playback are separate
```

| Contract | Practical consequence / original owner |
|---|---|
| Distinct identities | Immutable artifact/config/tokenizer, mathematical/numerical contract, prepared kernel/image/ABI, actual physical device and per-attempt authority are different truths. Shape, dtype, source hash or byte count alone cannot replace the other joins. |
| One mathematical engine | Families interpret source roles/ties/config and traverse math once, evaluating or recording existing operations. A fast provider replaces a closed declared region; it cannot import a second model traversal/server/scheduler. |
| One root publisher | Domains/workers/peers only prepare versions and return real completion. Every required domain/epoch/frontier and expected old root validates before publication. Private/COW/replay protects prior state; all fallible sibling transitions precede grouped swaps. |
| Physical ownership | Coarse original reservations include source, native code/context, scratch, candidate state, staging and output; fast local pools allocate under those reservations. Same physical PCI domain is counted once across backend views within the process. |
| Real lifetime | Encoded, captured, queued, executing and escaped/snapshot readers retain exact backing/generations. Future drop, timeout, refcount or socket completion is not GPU/DMA cancellation. Unknown work retains owners/charges and fences reuse/publication. |
| Legal fallback | Only before submission or at a proven compatible conversion/replay boundary. Unknown submitted work cannot silently retry another backend or source. Forced GPU selection never becomes CPU neural spill. |
| Full-state compatibility | Prefix/restore uses all model-required KV, convolution/recurrent/indexer, history/RNG/grammar/adapter/media domains at the common causal frontier. A KV-only/text hit cannot manufacture a complete hybrid root. |
| Bounded preparation | Prepare immutable topology, source joins, semantic schedule and finite implementation candidates once; bind actual inputs/state/epochs late. Runtime uses IDs/arrays, not per-token model catalogs, graph optimization or uncontrolled JIT. |
| Separate profiles | Source/storage encoding, numerical schedule, workload policy, explicit approximation and placement are independent. Approximation declares changed math/state/distribution and measured quality; it cannot waive safety/ownership. |
| Typed non-token control | Source media clocks, codec groups, solver iterations and finite decisions retain real units and output finality. A neural latent, incomplete RVQ group, provisional transcript or sent packet is not necessarily playable committed media. |

Verified-present code navigation, not fresh source/behavior acceptance:

- [core program](../../../crates/zerogravity-core/src/accelerator/program.rs),
  [source](../../../crates/zerogravity-core/src/source.rs),
  [session root](../../../crates/zerogravity-core/src/session_root.rs),
  [resource arbiter](../../../crates/zerogravity-core/src/resource_arbiter.rs).
- [Qwen traversal](../../../crates/zerogravity-model-qwen35/src/forward.rs),
  family crates `zerogravity-model-qwen35` and `zerogravity-model-gemma4` for
  current source/config/state and optional execution adapters. Source facts
  outrank historical file names in the23Oct review.
- [GPU resident lowering](../../../crates/zerogravity-gpu/src/resident_program.rs),
  [original resident/session path](../../../crates/zerogravity-gpu/src/resident_inference.rs),
  [matrix source join](../../../crates/zerogravity-gpu/src/resident_leases/resident_matrix/source.rs).
- [CLI request parser](../../../crates/zerogravity-cli/src/run_cli.rs),
  [serving route](../../../crates/zerogravity-cli/src/serve/gpu_engine/route.rs),
  [pure pipeline config](../../../crates/zerogravity-core/src/pipeline.rs).

The current [structure](../../plans/pipeline-composition/STRUCTURE_AND_NAMING.md)
distinguishes code dependency, mathematical/task graph and physical placement.
A module boundary alone forces no copy/wait, source size alone chooses no GPU,
and shared immutable weights never imply shared mutable session state.

## 6. Annotated subsystem reading catalog

All links lead to complete original bodies. Initial status examples and proposed
Rust types are dated; current TODO/source/tests determine what actually exists.

### Core/compiler/physical/state

| Body | Read for / dependency |
|---|---|
| [Vision/scope](../VISION_SCOPE_AND_PRINCIPLES.md) | User requirements, foundational contract, three planning levels and explicit non-goals. |
| [Modular reference](../MODULAR_AI_SYSTEMS_REFERENCE_ARCHITECTURE.md) | Small ontology; one fabric/two representations; extension seams and natural provider granularity. |
| [Minimal spine](../MINIMAL_EXECUTION_SPINE_AND_PROTOCOL_V4.md) | Five owners; prepare→publish; WorkEnvelope/worker credits; hierarchy, failure matrix and honest delivery. |
| [Composition interfaces](../COMPOSITION_AND_EXTENSION_CONTRACTS.md) | Exact source/shape/effect/region/state bridges; illustrative interfaces, lifecycle and invalidation. |
| [Task/compiler refinement](../TASK_PROGRAM_AND_COMPILER_REFINEMENT_V4.md) | Semantic operations vs pure internals, dynamic guards, trusted task frontiers, prepared StepProgram and optional gradients. |
| [Semantic IR/compiler](../SEMANTIC_IR_AND_COMPILER.md) | Typed effects and legal transforms; later V4 addenda supersede initial three/four durable-IR phrasing. |
| [Async resource commands](../ASYNC_RESOURCE_COMMAND_IR.md) | Resource/lease/timepoint effects, static-vs-dynamic bindings, first-use waits and failure dependencies; CIR has no logical root publication. |
| [Device kernels/codegen](../DEVICE_KERNEL_AND_CODEGEN.md) | Actual kernel contracts, ISA/codegen, numerical stores/reductions and prepared source/driver identity. Apply current reuse policy to old source-language wording. |
| [Operator/fusion catalog](../OPERATOR_AND_FUSION_CATALOG.md) | Projection/embedding, attention, recurrent, experts, quant and fusion legality; semantic identity before physical fusion. |
| [Engine protocol](../ENGINE_CORE_EXECUTION_AND_PROTOCOL.md) | Compact bounded scheduler/worker messages, persistent input batches, frontier/StepPlan and no copied model graph per step. |
| [State/cache transactions](../STATE_CACHE_AND_TRANSACTIONS.md) | Required mutable domains, prefix-common frontier, snapshots/COW and speculative rollback. V4 prepare-only publication supersedes old domain commits. |
| [Invariants/failure](../INVARIANTS_AND_FAILURE_SEMANTICS.md) | Source/identity, effect order, alias/memory and typed failure state; cancellation does not imply safe reclamation. |
| [Memory/residency/IO](../MEMORY_RESIDENCY_AND_IO.md) | Physical vs virtual capacity, tier priorities, expert slots, source IO and global budget/local policy split. |
| [Virtual tensor/SSD](../UNIFIED_VIRTUAL_TENSOR_RESIDENCY_AND_SSD_STREAMING_2026-10-15.md) | Stable source extents versus materialized generations, resident/miss fast paths, encoded readers, mixed expert sizes, four SSD uses and real cancellation. |
| [Cost/autotuning](../COST_MODEL_AND_AUTOTUNING.md) | Whole critical path rather than sum/FLOPs; planning timescales, finite caches, cold/warm and bounded offline tuning. |
| [Risk/hypothesis register](../PERFORMANCE_HYPOTHESES_AND_RISKS.md) | Falsifiable gains, uncertainty, tradeoffs and rejected reasoning; hypotheses are not measurements. |
| [Execution traces](../EXAMPLE_EXECUTION_TRACES.md) | Worked dense/extreme/prefix/future-op examples; illustrative trace is not a real executed checkpoint. |

### Model/source/quantization/acquisition

| Body | Read for / dependency |
|---|---|
| [Model semantics](../MODEL_SEMANTICS_AND_COMPATIBILITY.md) | Exact registered architecture/op support; candidate families and metadata profiles. Unknown topology never becomes executable from a name. |
| [Formats/loading](../FORMATS_QUANTIZATION_AND_LOADING.md) | Bounded GGUF/HF, tokenizers/templates, stored vs compute encoding, malformed/large sources and optional conversion. |
| [Product artifact pipeline](../PRODUCT_MODEL_PIPELINE_AND_EXTENSIONS.md) | ModelStore, source manifests, quant/tokenizer/grammar and generation-vs-encoder roles. |
| [Immutable sharded-source spec](../../plans/IMMUTABLE_SHARDED_SOURCE_IMPLEMENTATION_SPEC.md) | Actual retained immutable bytes/shards/header/index/shape/range identity and ownership; readonly path/mmap does not seal mutable external files. |
| [Acquisition implementation spec](../../plans/MODEL_ACQUISITION_IMPLEMENTATION_SPEC.md) | Explicit HF/GitHub/provider acquisition, complete required closure, offline policy, atomic publication/removal and loader bridge. |
| [Derived quant contract](../DERIVED_QUANTIZATION_CONTRACT_2026-10-15.md) | Lossless repack, temporary compute transform and new lossy derived artifact differ; calibration/source quality and speed must both be measured. |
| [Extreme hybrid/drafters](../EXTREME_HYBRID_MODELS_AND_SPECULATIVE_DRAFTER_SEMANTICS_2026-10-15.md) | Separate Qwen/GLM/MiMo recipes, real MTP/DFlash semantics and target authority; name/paired model cannot be guessed. |
| [Heterogeneous prefill/replay](../HETEROGENEOUS_PREFILL_AND_RETAINED_REPLAY_CONTRACT_2026-10-15.md) | Row splits versus expert placement, realized numerical schedule, ready/unknown signals and retained replay ownership. |
| [gfx1151 kernel/UMA contract](../GFX1151_ROCM_KERNEL_COST_AND_UNIFIED_MEMORY_SAFETY_CONTRACT_2026-10-15.md) | Wave/LDS and source kernel costs, routed rows, coherent RAM headroom, clock/fabric safety and physical qualification. |
| [NPU/GPU task pipelines](../NPU_GPU_COOPERATIVE_EXECUTION_AND_TASK_PIPELINES.md) | Three hardware parallelism types, source-exact projection joins, experimental power/fabric safety and task control. |
| [Current format homes](../../plans/pipeline-composition/FORMATS.md) | Exact Q4_0/Q4_1/Q4_K/Q6_K/Q8_0 versus unspecified Q1/Q2/Q3/NVFP4; BF16 is precision. Scalar Q4_K decoding does not grant a CPU projection kernel; backend bodies do not grant complete models. |
| [Gemma family research](../../research/GEMMA4_FAMILY_CONTRACTS_2026-10-16.md) | Variant source config, PLE/doubled FFN, proportional RoPE, shared/sliding KV, raw K=V and sparse routing; source/analytic contracts are not trained MoE execution. |

### Serving/composition/multimodal/distribution/release

| Body | Read for / dependency |
|---|---|
| [Scheduler/speculation](../SCHEDULER_BATCHING_AND_SPECULATION.md) | Unified real work budgets, routing/cache-aware batches, exact target verifier and complete state rollback. |
| [Joint serving/goodput](../JOINT_CONCURRENT_SERVING_OPTIMIZATION_AND_GOODPUT_CONTRACT_2026-10-15.md) | Three admission valves, full-state frontier meet, bounded Pareto choices, fair complete-request goodput and negative cases. |
| [Router/residency](../MODEL_ROUTING_RESIDENCY_AND_MULTI_MODEL_SERVING_CONTRACT.md) | Four identities, exact binding vs explicit different-model selection, cold load deduplication, compatible replicas and safe unload. |
| [CLI/config/API](../ZERO_CONFIG_CLI_CONFIG_AND_OPENAI_API_CONTRACT.md) | Progressive UX, effective provenance, official SDK/field/stream subset, typed unsupported semantics and loopback/admin boundary. |
| [Runtime UX](../RUNTIME_SERVING_AND_UX.md) | Library/CLI/protocol adaptation, stop/UTF-8/tool-stream semantics and truthful user errors. |
| [Current routing design](../../plans/cli-device-routing/README.md) | Prefer all qualified single-GPU fits before splitting; forced backend on every neural stage; current resources, source schemes, YAML and inspect/dry-run. |
| [Batching/paged state](../../plans/serving-and-speculation/BATCHING_AND_PAGED_STATE.md) | Request batching vs temporal prefill; paged KV vs recurrent domains; all-domain prefix/COW, readers, fairness and full numerical/physical/HTTP gates. Old C2 examples yield to explicit later no-C2 order. |
| [Recursive regions/profiles](../../plans/pipeline-composition/REGIONS_AND_PROFILES.md) | Nested reusable scope metadata, deterministic flattening, closed effect/observer contract and post-fusion liveness. Mathematical and task ports remain typed differently. |
| [Profile/reuse policy](../../plans/pipeline-composition/PROFILE_POLICY_AND_COMPONENT_REUSE.md) | Explicit approximation, replaceable component implementations, shared immutable config/observations, audited static native reuse/ports and meaningful correctness harnesses. |
| [V5 multimodal](../UNIFIED_MULTIMODAL_EXECUTION_CONTRACT_V5_2026-10-14.md) | MM-01…11 media clocks/assemblies/control/frontiers/codecs/finite decisions/native-vs-hosted contract; initial mock status stays bounded. |
| [V5.1 universal](../UNIVERSAL_MULTIMODAL_EXECUTION_ARCHITECTURE_V5_1_2026-10-15.md) | Typed values, fifteen task families, rational clocks, multitrack root, source transforms, specialization axes and actual capability/status. |
| [V5.1 verified assemblies](../VERIFIED_MODEL_ASSEMBLIES_AND_SPECIALIZATION_V5_1_2026-10-15.md) | One source model with independently placed verified components differs from replicas or ASR→LLM→TTS workflow. Exact processor/codec/tensor/rights identity. |
| [V5.1 realtime/QoS](../REALTIME_MULTIMODAL_SERVING_AND_QOS_V5_1_2026-10-15.md) | Capture/perception/response/output lanes; committed/delivered/ACK/played frontiers, barge-in/reconnect epochs, bounded buffering and soft deadlines. |
| [Accelerator families](../ACCELERATOR_EXECUTION_FAMILIES_AND_BACKEND_CONTRACT.md) | Natural granularity, compile/load/warm/submit/retire, state bridge and truthful native-vs-opaque service distinction. |
| [Distributed overview](../DISTRIBUTED_PARALLELISM.md) | Entry map to four placements, authority, required topology and zero standalone cluster tax. |
| [Distributed placement](../DISTRIBUTED_PLACEMENT_AND_NETWORK_EXECUTION_CONTRACT.md) | Source mathematical/task/dependency graphs versus physical plan; actual topology, token-range algebra and transfer/cold/failure costs. |
| [Peer/state handoff](../REMOTE_PEER_TRANSPORT_AND_STATE_HANDOFF_PROTOCOL.md) | Bounded authenticated control/bulk, peer/model/tenant rights, authoritative state shards vs cache hints, migration/fencing/fail-stop. |
| [Numerical quality](../NUMERICS_QUALITY_AND_BENCHMARKS.md) | Independent reference layers, declared error and quality measures, rounding/reduction/casts and matched source benchmarks. Current all-model policy refines determinism interpretation. |
| [Security/reliability](../SECURITY_AND_RELIABILITY.md) | Untrusted artifacts/programs, physical lifetime, listener/admission/privacy, remote rights and operational faults. |
| [Rust engineering](../RUST_WORKSPACE_AND_ENGINEERING.md) | Cohesive ownership/build boundaries and hot-path discipline. Historical crate counts/names are illustrative; current structure owns migration. |
| [Release gates](../../plans/RELEASE_ACCEPTANCE_GATES.md) and [publication checklist](../../../docs/PUBLICATION_AND_REVIEW_CHECKLIST.md) | Real model/task/backend/platform/quality/state/security and release acceptance versus manuscript publication. Source packages/feature checks/installer definitions remain narrower than a release. |

## 7. Engineering specification ES-00…15 crosswalk

The full [engineering specification](../../../docs/ENGINEERING_SPECIFICATION.md)
is preserved; this table is a reading guide, not an implementation checklist.

| ES | Exact purpose | Later live owner / refinement |
|---|---|---|
| ES-00 | Authority, scope, proof ladder, internal/public-format stability. | AGENTS + plan rules; current source/results override initial manuscript status. |
| ES-01 | Dependency boundaries, artifact/semantic/task/model/session/placement/runtime identities. | Structure/naming; family-independent core/backends; one application registration boundary. |
| ES-02 | Trusted artifacts, exact registered source mathematics and four correctness questions. | Acquisition/source specs, model crates, all-model correctness policy. |
| ES-03 | Trusted TaskDriver, task-specific outputs/frontiers, external application actions. | Composition/multimodal plans; test/config scaffolds remain distinct from actual tasks. |
| ES-04 | One semantic program/compiled plan, closed regions/providers, guards, bounded caches, CIR. | Shared ForwardPlan, recursive scope metadata and prepared native implementations. |
| ES-05 | Coarse physical admission, local pools, bounded WorkEnvelope and real completion. | Original ResourceArbiter/worker/lease/flight; shared memory boundaries and separate backend qualification. |
| ES-06 | All-domain prepare→one root; speculative rollback, prefix compatibility and output sequencing. | Original SessionAuthority path; complete-domain paged/cache/COW work. |
| ES-07 | Four serving entities, exact route/replica, cold load, lifecycle and alias revision. | Model loading/application adapters and same-engine serving plan. |
| ES-08 | Independent distributed placement, attention algebra, network cost and remote epochs. | Distributed icebox; actual two-host numerical/state proof remains separate. |
| ES-09 | Zero-config local run, same-engine SDK/API, config provenance and conformance subset. | YAML/current CLI-device plan; forced backend and actual capability checks. |
| ES-10 | Threat boundaries, typed errors, observability and license/rights. | Current MIT/native-reuse policy and runtime ownership; external/remote permissions remain scoped. |
| ES-11 | INV-01…25 plus deliberate counterexamples. | Relevant actual tests and review; not a mandate for policy-placement test machinery or fixture-dependent production. |
| ES-12 | Local CLI, multi-model service, remote layers/state, graph/media and failed verifier walkthroughs. | Illustrative control flow; each current implemented route needs actual source/results. |
| ES-13 | Real vertical slice first; extension next; stable P00…23 and capability definition of done. | Current substantial end-to-end units, assembled-batch validation and gate last. |
| ES-14 | Low change radius for model/op/kernel/quant/task/peer/API; reject copied authorities. | Current model-owned traversal/shared-operator refactor and honest extension drills. |
| ES-15 | Normative source map and glossary, no frozen pseudocode ABI. | Original canonical sources plus current supersessions in this chapter. |

## 8. Original P00…P23 capability map and dependency spine

Use the original [roadmap](../../plans/IMPLEMENTATION_ROADMAP.md) and
[playbook](../../plans/archive/history/AGENT_IMPLEMENTATION_PLAYBOOK.md) for
full deliver/pass/refuse detail. These IDs are stable capability coverage, not
proof that all phases are done or instructions to replay every earlier gate.

| Phase | Original capability | Current primary plan(s); remaining proof principle |
|---|---|---|
| P00 | Lean reproducible Rust/toolchain/source/release foundation. | platform-and-release; packaging scaffolding differs from real platform/release acceptance. |
| P01 | Physical native accelerator source/codegen feasibility. | cpu-gpu-optimization; compiles/mock/device probe is not full-model support. |
| P02 | Independent mathematical and state-transition oracle. | numerics-and-context/model-execution; original reference and exact profile, not self-comparison. |
| P03 | Trusted artifacts, quant codecs, tokenizer and rendering. | model-acquisition/model-execution; complete closure and source identity, no executable remote code. |
| P04 | Effect-aware semantic program and trusted task control. | pipeline-composition; one ForwardPlan and real effects/frontiers. |
| P05 | Legal region optimization, candidates/guards and async compiled plan. | pipeline-composition/cpu-gpu-optimization; no new permanent IR/interpreter or guard-free shape widening. |
| P06 | Useful portable source-correct CPU inference. | model-execution/numerics-and-context; real learned model outputs, tokenizer/state and applicable contexts. |
| P07 | Physical GPU workers, streams, pools, leases and completion. | cpu-gpu-optimization/memory-and-streaming; separate native backend/source/driver/physical proof. |
| P08 | First complete trained model/task vertical slice. | model-execution/cli-device-routing; actual source→math→device→root→user output, same engine. |
| P09 | TaskCoordinator, worker protocol, router and API/SDK. | serving-and-speculation/cli-device-routing; real endpoint/field/stream subset, no second model loop. |
| P10 | Global physical residency and complete-domain prefix/snapshots. | memory-and-streaming/serving-and-speculation; live roots/readers/candidates/COW, not KV-only hit. |
| P11 | Complex attention/recurrent/hybrid source operators. | model-execution/numerics-and-context; exact source geometry/reduction/state and genuine contexts. |
| P12 | Routed sparse parameters, expert cache and tiered IO. | memory-and-streaming/model-execution; actual trained routing/top-k, generations/readers and bounded misses. |
| P13 | First extreme large hybrid specialization. | model-execution/cpu-gpu-optimization; full source/task/state/performance scope, not analytic metadata. |
| P14 | Substantially different second heavy family. | model-execution/pipeline-composition; same owners, real source execution and low change radius. |
| P15 | Target-authoritative MTP/draft/speculative verification. | serving-and-speculation; target distribution, full rejected state/RNG rollback and natural feedback. |
| P16 | High concurrency, continuous batching and joint cost/SLO. | serving-and-speculation/cpu-gpu-optimization; full requests/HTTP/fairness/p95 and exact matched baseline. |
| P17 | Broader model roles, adapters, embeddings and reranking. | model-execution/pipeline-composition; independent family/task/adapter proof. |
| P18 | ASR/TTS/image/video/non-token and iterative tasks. | multimodal/pipeline-composition; playable/seeded/source-correct media and actual clocks/codecs. |
| P19 | Multiple devices, collectives, authenticated remote placement. | distributed-execution; actual source/state/peer/tunnel/cost/fault proof, not SSH fleet discovery. |
| P20 | Bounded offline tuning and negative experiments. | cpu-gpu-optimization/pipeline-composition; real whole-request benefit, finite cache/compile cost. |
| P21 | Security/source rights and signed reproducible release. | platform-and-release; parser/API/privacy/NOTICE/fuzz/OS/hardware and actual artifacts. |
| P22 | Release-level numerical/task/performance matrix. | all affected plans/platform-and-release; scoped wins/ties/losses, real complete useful outputs. |
| P23 | New model/op/kernel/task/quant/IO/peer without framework growth. | pipeline-composition/platform-and-release; no duplicate scheduler/allocator/publisher; future training seam remains optional. |

Original dependency spine: P00 alongside independent P01 feasibility and
P02/P03 source truth → P04 effects/task → P05 prepared execution → P06/P07 →
P08 useful model → P09/P10 shared serving/residency → P11–P14 complex/sparse/
different families → P15/P16 speculation/batching → P17/P18 breadth → P19
distribution → P20–P22 full optimization/security/release → repeated P23
extension proof. Independent ready source/parser/device/design work can proceed
in parallel; this is not a waterfall or a reason to block safe unrelated lanes.

Supporting falsification namespaces:

- Stage0: S0-00…14; optional NPU-A…F, V4-A…J, FABRIC-A…H,
  DIST-A…L and SERVE-A…J. Tiny mock graph/dataflow/peer tests establish
  interfaces, not physical accelerator/network support.
- H01…14: kernels/cache/speculation/serving/memory/topology/tasks;
  H15…25: effects/async/capture/search/frontier/modularity/input/runtime cost;
  H26…31: cooperative NPU/safety/source quality/non-token tasks;
  H32…43: V4 publication/device progression/resources/shapes/minimality;
  H44…55: graph/dataflow/source-policy extension;
  H56…70: real distributed state/transport/failure/capacity;
  H71…86: local UX/catalog/residency/replica/profiles/API/security/change radius.
- Modality acceptance: MM, ASR, TTS, IMG, VID, DEC, BON and SRV IDs in the
  preserved multimodal roadmap/dossiers. SSD-01…32 and SERVOPT gates retain
  their original source/reader/quality/cost/fault semantics.

Choose applicable checks for the changed end-to-end behavior. No instruction
here requires all86 hypotheses before useful implementation, a per-helper gate,
new policy tests, or repetition of accepted unchanged coverage.

## 9. Latest shared-architecture implications

The23Oct shared architecture is a source review plus explicit design direction;
its earlier file/bug observations can be superseded by subsequent source. Its
lasting requirements, strengthened by26Oct:

- Family config/tensor-role joins/ties/geometry and mathematical traversal live
  with the family. Generic artifacts read authenticated bytes and containers;
  shared core/operators do not own Qwen/Gemma recipes or testcase ordinals.
- Source-specific reduction/FMA/store/rounding is actual math. Removing family
  leakage must preserve those numerical/source identities, not turn them into
  an unconstrained dtype enum or a global epsilon.
- Shared prepared head/matrix/page transport must retain exact source issuer,
  tensor/member/range/axes/codec and actual producer. Equal source+shape alone
  cannot authenticate another same-shaped matrix.
- Share actual context/envelope/queue/ledger/fence, backend compiler/module
  owners, storage/rings and transfers. Model sub-admissions are independent;
  attaching a model cannot increase physical caps or spend the whole remainder.
- Distinguish prepared immutable topology from request invocation/root/phase.
  No first-row plan can authorize siblings. Cohorts need every row's genuine
  source/prior/candidate/position/range and complete cost before first submit.
- Backend native context/compiler/module lifetimes differ. Successful module
  unload alone cannot prove driver context/allocator memory is gone. Unknown
  load/launch/sync/unload/drop retains applicable owners/charges and fences.
- B4/B8 arithmetic does not grant B2. Request width, source-prefill population,
  temporal rows, cached frontiers and natural feedback are explicit data. A
  captured source state can be a labeled diagnostic operand, never a production
  initial root that repairs missing Rust inference.
- Existing analytic MoE, synchronous exclusive cache, config-only pipeline and
  transport tests are not asynchronous trained expert inference, a recursive
  execution engine or distributed/media product.
- Fewer launches/copies/types are mechanisms. Measure complete admitted
  requests and tail latency under declared resources/quality; fusion can increase
  liveness/occupancy cost. Preserve forcible ordinary decomposition.
- Source/compile feature boundaries plus review protect architecture. Current
  production qualification/table coupling must be removed from actual callers,
  exports, includes and enabled dependencies; do not conceal it behind names,
  feature gates or renamed fixture capabilities.

Current26Oct [correctness policy](../../plans/numerics-and-context/CORRECTNESS_POLICY.md)
requires semantics/safety, independent operator accuracy, matched model/task
accuracy and separately advertised determinism. For .8B, fresh47/48/49/50-token
prompts differ from47-token prompt plus cached continuation. Original N1/B4
width-dependent BF16 observations do not invalidate each width's independent
accuracy, but do preserve the historical strict batch-equality failure. Genuine
512/1K/2K/4K/8K and deferred16K/32K require real full-model prefixes and
continuations; high position IDs or empty padding cannot qualify long context.

## 10. Historical recovery, stale links and limits

The17Oct design/history chapter and20Oct catalog explicitly corrected a coarse
“early reports lost” account: an older complete94-source implementation handoff
through U-040/P07-G3b survives nested inside the recovered multimodal aggregate.
Missing standalone paths do not imply missing embedded text. The existing
sourcebooks preserve those actual bodies; this handoff need not recreate them.

| Historical material | Canonical surviving reading path / limit |
|---|---|
| V2.1/V2 handoff embedding original V1; V2 master/roadmap/Stage0/experiments/gates; V3 design/handoff snapshots; pre-V4.2 distributed snapshot | [RECOVERED_HISTORY_01.md](../handoff-2026-10-20/RECOVERED_HISTORY_01.md); exact original path/ref/body ranges in [CONTEXT_CATALOG](../handoff-2026-10-20/CONTEXT_CATALOG.md). |
| Complete older multimodal aggregate containing94-source early implementation/handoff records and source proposals | [RECOVERED_HISTORY_02.md](../handoff-2026-10-20/RECOVERED_HISTORY_02.md). The recorded2,796,909-byte/23,979-line body is historical text, not a current binary/reference runtime. |
| Retired modality agent briefs, SSD wrapper and extreme-specialization wrapper | [RECOVERED_HISTORY_03.md](../handoff-2026-10-20/RECOVERED_HISTORY_03.md), with original references in the catalog. |
| Actual pre-product README body recovered from recorded be95424 provenance | [RECOVERED_HISTORY_04.md](../handoff-2026-10-20/RECOVERED_HISTORY_04.md). Current research pointer is a different file; neither authorizes changing root README. |
| All19Oct context including17Oct, former plan snapshots and old next prompt | [19Oct handoff](../FULL_COMPREHENSIVE_AGENT_HANDOFF_2026-10-19.md). Its Appendix A contains17Oct; B former TODOs; C outside-root catalog; D old prompt. |
|20Oct sourcebooks, all then-retained run Markdown, selected paused sources/patches/licenses | [CONTEXT_CATALOG](../handoff-2026-10-20/CONTEXT_CATALOG.md), `ARCHIVE_CONTENTS.json`, `HANDOFF_CONTENTS.json`. Exact source-to-volume ranges are recorded there. Sourcebooks are dated copies; patches remain unaccepted unless later source proves integration. |
| Original33-node graph, later129/191-source aggregate editions, full70-report set and every H4+ verbatim report | Earlier recovery audits did not recover these in full. Some early report bodies survive inside94-source archive. Current summaries/approval labels do not manufacture unavailable originals. |
| Old6d74f76/bfe672c approval labels | Preserved as historical accepted-baseline/cleanup labels. Earlier rewritten-graph audits found their objects unresolvable; a printed SHA does not establish an available object. This lane did not rerun Git-object recovery. |

Canonical remaps for old links:

| Former path | Present source |
|---|---|
| `.agents/plans/DECISION_REGISTER.md` | [archive/history/DECISION_REGISTER.md](../../plans/archive/history/DECISION_REGISTER.md) |
| `.agents/plans/AGENT_IMPLEMENTATION_PLAYBOOK.md` | [archive/history/AGENT_IMPLEMENTATION_PLAYBOOK.md](../../plans/archive/history/AGENT_IMPLEMENTATION_PLAYBOOK.md) |
| `.agents/plans/REQUIREMENTS_AND_TEST_TRACEABILITY.md` | [archive/history/REQUIREMENTS_AND_TEST_TRACEABILITY.md](../../plans/archive/history/REQUIREMENTS_AND_TEST_TRACEABILITY.md) |
| `.agents/plans/CURRENT_STATUS_AND_TRACKING.md` | [archive/history/CURRENT_STATUS_AND_TRACKING.md](../../plans/archive/history/CURRENT_STATUS_AND_TRACKING.md), historical only; use live queue/TODOs for status. |
| `.agents/plans/OPEN_CONTRIBUTION_AI_DEVELOPMENT_AND_MIT_POLICY_2026-10-15.md` | [archive/history/open-contribution policy](../../plans/archive/history/OPEN_CONTRIBUTION_AI_DEVELOPMENT_AND_MIT_POLICY_2026-10-15.md) |
| `.agents/plans/{accepted-baseline,handoff-cleanup,historical-records}/TODO.md` | Corresponding [archive/history](../../plans/archive/history/) directories. |
| `.agents/plans/TODO-SPEC.md` | Current [AGENTS](../../../AGENTS.md) and [plan rules](../../plans/README.md). |
| Master's unqualified `FINAL_ARCHITECTURE_REVIEW_2026-10-12.md` | [.agents/prior-art/final review](../../prior-art/FINAL_ARCHITECTURE_REVIEW_2026-10-12.md). |

Scope of this lane's review: complete original NIGHT, v4.3 master, whitepaper,
ES-00…15, archived decision register and current shared ForwardPlan/structure/
format/correctness bodies; original historical design/history core chapters;
linked subsystem and research catalog/selected normative detail. The exhaustive
file inventory below proves navigation coverage, not independent line-by-line
requalification of every nested sourcebook/patch/research claim. Existing
catalogs preserve complete bodies and exact historical ranges. No fresh model,
hardware, test, benchmark, external-web or Git-recovery assertion is made here.

The external read-only research library
`/home/pvl/spaces/abird/research/inference/{analysis,refs/COMMITS.txt}` remains a
source-provenance dependency identified by reconciliation. This lane did not
read/copy the complete external source tree, model caches, SDKs, containers or
raw tensor/binary payloads. Dated Markdown does not reproduce their environment.
Inspect named run paths only when needed; do not broadly recurse giant runs.

## 11. Complete current original-file inventory

This inventory enumerates files existing under `.agents/docs`, `.agents/plans`,
`.agents/research`, `.agents/prior-art` and top-level `docs` at this lane's
inspection, excluding the newly authored26Oct handoff and its copies. It is a
one-time navigation snapshot, not another hash/receipt requirement. Exact body
snapshots/mirror correspondence belong to ROOT's `SOURCE_CATALOG.md`.

Annotations distinguish live queues/contracts, dated source reviews, proposals
and preservation volumes. Numbered volumes contain multiple historical bodies;
resolve their embedded links against each original path per the20Oct catalog.

At inspection: **264 original files** (257 Markdown, 7 other text/catalog files). All are linked below.

### 11.1 Core contracts, dated reviews and top-level handoffs

56 files.

| Complete original path | Reading purpose / status |
|---|---|
| [.agents/docs/ABI_FAMILY_CLI_REFACTOR_2026-10-25.md](../ABI_FAMILY_CLI_REFACTOR_2026-10-25.md) | Dated source review/refactor/continuation; actual code/results and later fixes supersede earlier findings. |
| [.agents/docs/ACCELERATOR_EXECUTION_FAMILIES_AND_BACKEND_CONTRACT.md](../ACCELERATOR_EXECUTION_FAMILIES_AND_BACKEND_CONTRACT.md) | Original detailed contract/context body; see annotated reading catalog and current owning TODO for authority. |
| [.agents/docs/ARCHITECTURE_CORRECTNESS_REVIEW_2026-10-25.md](../ARCHITECTURE_CORRECTNESS_REVIEW_2026-10-25.md) | Dated source review/refactor/continuation; actual code/results and later fixes supersede earlier findings. |
| [.agents/docs/ASYNC_RESOURCE_COMMAND_IR.md](../ASYNC_RESOURCE_COMMAND_IR.md) | Original detailed contract/context body; see annotated reading catalog and current owning TODO for authority. |
| [.agents/docs/B4_C16_EXECUTION_CONTINUATION_2026-10-25.md](../B4_C16_EXECUTION_CONTINUATION_2026-10-25.md) | Dated source review/refactor/continuation; actual code/results and later fixes supersede earlier findings. |
| [.agents/docs/COMPOSITION_AND_EXTENSION_CONTRACTS.md](../COMPOSITION_AND_EXTENSION_CONTRACTS.md) | Original detailed contract/context body; see annotated reading catalog and current owning TODO for authority. |
| [.agents/docs/COST_MODEL_AND_AUTOTUNING.md](../COST_MODEL_AND_AUTOTUNING.md) | Original detailed contract/context body; see annotated reading catalog and current owning TODO for authority. |
| [.agents/docs/DERIVED_QUANTIZATION_CONTRACT_2026-10-15.md](../DERIVED_QUANTIZATION_CONTRACT_2026-10-15.md) | Original detailed contract/context body; see annotated reading catalog and current owning TODO for authority. |
| [.agents/docs/DEVICE_KERNEL_AND_CODEGEN.md](../DEVICE_KERNEL_AND_CODEGEN.md) | Original detailed contract/context body; see annotated reading catalog and current owning TODO for authority. |
| [.agents/docs/DISTRIBUTED_PARALLELISM.md](../DISTRIBUTED_PARALLELISM.md) | Original detailed contract/context body; see annotated reading catalog and current owning TODO for authority. |
| [.agents/docs/DISTRIBUTED_PLACEMENT_AND_NETWORK_EXECUTION_CONTRACT.md](../DISTRIBUTED_PLACEMENT_AND_NETWORK_EXECUTION_CONTRACT.md) | Original detailed contract/context body; see annotated reading catalog and current owning TODO for authority. |
| [.agents/docs/ENGINE_CORE_EXECUTION_AND_PROTOCOL.md](../ENGINE_CORE_EXECUTION_AND_PROTOCOL.md) | Original detailed contract/context body; see annotated reading catalog and current owning TODO for authority. |
| [.agents/docs/EXAMPLE_EXECUTION_TRACES.md](../EXAMPLE_EXECUTION_TRACES.md) | Original detailed contract/context body; see annotated reading catalog and current owning TODO for authority. |
| [.agents/docs/EXTREME_HYBRID_MODELS_AND_SPECULATIVE_DRAFTER_SEMANTICS_2026-10-15.md](../EXTREME_HYBRID_MODELS_AND_SPECULATIVE_DRAFTER_SEMANTICS_2026-10-15.md) | Original detailed contract/context body; see annotated reading catalog and current owning TODO for authority. |
| [.agents/docs/FORMATS_QUANTIZATION_AND_LOADING.md](../FORMATS_QUANTIZATION_AND_LOADING.md) | Original detailed contract/context body; see annotated reading catalog and current owning TODO for authority. |
| [.agents/docs/FULL_COMPREHENSIVE_AGENT_HANDOFF_2026-10-17.md](../FULL_COMPREHENSIVE_AGENT_HANDOFF_2026-10-17.md) | Dated comprehensive historical handoff; chronology/approval record, current source and decisions supersede status. |
| [.agents/docs/FULL_COMPREHENSIVE_AGENT_HANDOFF_2026-10-19.md](../FULL_COMPREHENSIVE_AGENT_HANDOFF_2026-10-19.md) | Dated comprehensive historical handoff; chronology/approval record, current source and decisions supersede status. |
| [.agents/docs/FULL_COMPREHENSIVE_AGENT_HANDOFF_2026-10-20.md](../FULL_COMPREHENSIVE_AGENT_HANDOFF_2026-10-20.md) | Dated comprehensive historical handoff; chronology/approval record, current source and decisions supersede status. |
| [.agents/docs/FULL_COMPREHENSIVE_AGENT_HANDOFF_2026-10-25-NIGHT.md](../FULL_COMPREHENSIVE_AGENT_HANDOFF_2026-10-25-NIGHT.md) | Dated comprehensive historical handoff; chronology/approval record, current source and decisions supersede status. |
| [.agents/docs/FULL_COMPREHENSIVE_AGENT_HANDOFF_2026-10-25-PM.md](../FULL_COMPREHENSIVE_AGENT_HANDOFF_2026-10-25-PM.md) | Dated comprehensive historical handoff; chronology/approval record, current source and decisions supersede status. |
| [.agents/docs/FULL_COMPREHENSIVE_AGENT_HANDOFF_2026-10-25.md](../FULL_COMPREHENSIVE_AGENT_HANDOFF_2026-10-25.md) | Dated comprehensive historical handoff; chronology/approval record, current source and decisions supersede status. |
| [.agents/docs/GFX1151_ROCM_KERNEL_COST_AND_UNIFIED_MEMORY_SAFETY_CONTRACT_2026-10-15.md](../GFX1151_ROCM_KERNEL_COST_AND_UNIFIED_MEMORY_SAFETY_CONTRACT_2026-10-15.md) | Original detailed contract/context body; see annotated reading catalog and current owning TODO for authority. |
| [.agents/docs/HANDOFF_ALL_COMMIT_LINEAGE_2026-10-25.md](../HANDOFF_ALL_COMMIT_LINEAGE_2026-10-25.md) | Dated25Oct source/commit/open-item coverage index; descriptive snapshot, not live queue. |
| [.agents/docs/HANDOFF_ALL_DESIGN_AND_SPEC_SOURCES_2026-10-25.md](../HANDOFF_ALL_DESIGN_AND_SPEC_SOURCES_2026-10-25.md) | Dated25Oct source/commit/open-item coverage index; descriptive snapshot, not live queue. |
| [.agents/docs/HANDOFF_ALL_OPEN_PLAN_ITEMS_2026-10-25.md](../HANDOFF_ALL_OPEN_PLAN_ITEMS_2026-10-25.md) | Dated25Oct source/commit/open-item coverage index; descriptive snapshot, not live queue. |
| [.agents/docs/HETEROGENEOUS_PREFILL_AND_RETAINED_REPLAY_CONTRACT_2026-10-15.md](../HETEROGENEOUS_PREFILL_AND_RETAINED_REPLAY_CONTRACT_2026-10-15.md) | Original detailed contract/context body; see annotated reading catalog and current owning TODO for authority. |
| [.agents/docs/INVARIANTS_AND_FAILURE_SEMANTICS.md](../INVARIANTS_AND_FAILURE_SEMANTICS.md) | Original detailed contract/context body; see annotated reading catalog and current owning TODO for authority. |
| [.agents/docs/JOINT_CONCURRENT_SERVING_OPTIMIZATION_AND_GOODPUT_CONTRACT_2026-10-15.md](../JOINT_CONCURRENT_SERVING_OPTIMIZATION_AND_GOODPUT_CONTRACT_2026-10-15.md) | Original detailed contract/context body; see annotated reading catalog and current owning TODO for authority. |
| [.agents/docs/LINUX_DEV_NODE_SELECTORS_2026-10-25.md](../LINUX_DEV_NODE_SELECTORS_2026-10-25.md) | Dated source review/refactor/continuation; actual code/results and later fixes supersede earlier findings. |
| [.agents/docs/MEMORY_RESIDENCY_AND_IO.md](../MEMORY_RESIDENCY_AND_IO.md) | Original detailed contract/context body; see annotated reading catalog and current owning TODO for authority. |
| [.agents/docs/MINIMAL_EXECUTION_SPINE_AND_PROTOCOL_V4.md](../MINIMAL_EXECUTION_SPINE_AND_PROTOCOL_V4.md) | Original detailed contract/context body; see annotated reading catalog and current owning TODO for authority. |
| [.agents/docs/MODEL_ROUTING_RESIDENCY_AND_MULTI_MODEL_SERVING_CONTRACT.md](../MODEL_ROUTING_RESIDENCY_AND_MULTI_MODEL_SERVING_CONTRACT.md) | Original detailed contract/context body; see annotated reading catalog and current owning TODO for authority. |
| [.agents/docs/MODEL_SEMANTICS_AND_COMPATIBILITY.md](../MODEL_SEMANTICS_AND_COMPATIBILITY.md) | Original detailed contract/context body; see annotated reading catalog and current owning TODO for authority. |
| [.agents/docs/MODULAR_AI_SYSTEMS_REFERENCE_ARCHITECTURE.md](../MODULAR_AI_SYSTEMS_REFERENCE_ARCHITECTURE.md) | Original detailed contract/context body; see annotated reading catalog and current owning TODO for authority. |
| [.agents/docs/NPU_GPU_COOPERATIVE_EXECUTION_AND_TASK_PIPELINES.md](../NPU_GPU_COOPERATIVE_EXECUTION_AND_TASK_PIPELINES.md) | Original detailed contract/context body; see annotated reading catalog and current owning TODO for authority. |
| [.agents/docs/NUMERICS_QUALITY_AND_BENCHMARKS.md](../NUMERICS_QUALITY_AND_BENCHMARKS.md) | Original detailed contract/context body; see annotated reading catalog and current owning TODO for authority. |
| [.agents/docs/OPERATOR_AND_FUSION_CATALOG.md](../OPERATOR_AND_FUSION_CATALOG.md) | Original detailed contract/context body; see annotated reading catalog and current owning TODO for authority. |
| [.agents/docs/PERFORMANCE_HYPOTHESES_AND_RISKS.md](../PERFORMANCE_HYPOTHESES_AND_RISKS.md) | Original detailed contract/context body; see annotated reading catalog and current owning TODO for authority. |
| [.agents/docs/POST_4BF3673_AUDIT_AND_RESUME_2026-10-25.md](../POST_4BF3673_AUDIT_AND_RESUME_2026-10-25.md) | Dated source review/refactor/continuation; actual code/results and later fixes supersede earlier findings. |
| [.agents/docs/PRODUCT_MODEL_PIPELINE_AND_EXTENSIONS.md](../PRODUCT_MODEL_PIPELINE_AND_EXTENSIONS.md) | Original detailed contract/context body; see annotated reading catalog and current owning TODO for authority. |
| [.agents/docs/REALTIME_MULTIMODAL_SERVING_AND_QOS_V5_1_2026-10-15.md](../REALTIME_MULTIMODAL_SERVING_AND_QOS_V5_1_2026-10-15.md) | Original detailed contract/context body; see annotated reading catalog and current owning TODO for authority. |
| [.agents/docs/REMOTE_PEER_TRANSPORT_AND_STATE_HANDOFF_PROTOCOL.md](../REMOTE_PEER_TRANSPORT_AND_STATE_HANDOFF_PROTOCOL.md) | Original detailed contract/context body; see annotated reading catalog and current owning TODO for authority. |
| [.agents/docs/RUNTIME_SERVING_AND_UX.md](../RUNTIME_SERVING_AND_UX.md) | Original detailed contract/context body; see annotated reading catalog and current owning TODO for authority. |
| [.agents/docs/RUST_WORKSPACE_AND_ENGINEERING.md](../RUST_WORKSPACE_AND_ENGINEERING.md) | Original detailed contract/context body; see annotated reading catalog and current owning TODO for authority. |
| [.agents/docs/SCHEDULER_BATCHING_AND_SPECULATION.md](../SCHEDULER_BATCHING_AND_SPECULATION.md) | Original detailed contract/context body; see annotated reading catalog and current owning TODO for authority. |
| [.agents/docs/SECURITY_AND_RELIABILITY.md](../SECURITY_AND_RELIABILITY.md) | Original detailed contract/context body; see annotated reading catalog and current owning TODO for authority. |
| [.agents/docs/SEMANTIC_IR_AND_COMPILER.md](../SEMANTIC_IR_AND_COMPILER.md) | Original detailed contract/context body; see annotated reading catalog and current owning TODO for authority. |
| [.agents/docs/STATE_CACHE_AND_TRANSACTIONS.md](../STATE_CACHE_AND_TRANSACTIONS.md) | Original detailed contract/context body; see annotated reading catalog and current owning TODO for authority. |
| [.agents/docs/TAKEOVER_6EDC6E2_INDEPENDENT_AUDIT_2026-10-25.md](../TAKEOVER_6EDC6E2_INDEPENDENT_AUDIT_2026-10-25.md) | Dated source review/refactor/continuation; actual code/results and later fixes supersede earlier findings. |
| [.agents/docs/TASK_PROGRAM_AND_COMPILER_REFINEMENT_V4.md](../TASK_PROGRAM_AND_COMPILER_REFINEMENT_V4.md) | Original detailed contract/context body; see annotated reading catalog and current owning TODO for authority. |
| [.agents/docs/UNIFIED_MULTIMODAL_EXECUTION_CONTRACT_V5_2026-10-14.md](../UNIFIED_MULTIMODAL_EXECUTION_CONTRACT_V5_2026-10-14.md) | Original detailed contract/context body; see annotated reading catalog and current owning TODO for authority. |
| [.agents/docs/UNIFIED_VIRTUAL_TENSOR_RESIDENCY_AND_SSD_STREAMING_2026-10-15.md](../UNIFIED_VIRTUAL_TENSOR_RESIDENCY_AND_SSD_STREAMING_2026-10-15.md) | Original detailed contract/context body; see annotated reading catalog and current owning TODO for authority. |
| [.agents/docs/UNIVERSAL_MULTIMODAL_EXECUTION_ARCHITECTURE_V5_1_2026-10-15.md](../UNIVERSAL_MULTIMODAL_EXECUTION_ARCHITECTURE_V5_1_2026-10-15.md) | Original detailed contract/context body; see annotated reading catalog and current owning TODO for authority. |
| [.agents/docs/VERIFIED_MODEL_ASSEMBLIES_AND_SPECIALIZATION_V5_1_2026-10-15.md](../VERIFIED_MODEL_ASSEMBLIES_AND_SPECIALIZATION_V5_1_2026-10-15.md) | Original detailed contract/context body; see annotated reading catalog and current owning TODO for authority. |
| [.agents/docs/VISION_SCOPE_AND_PRINCIPLES.md](../VISION_SCOPE_AND_PRINCIPLES.md) | Original detailed contract/context body; see annotated reading catalog and current owning TODO for authority. |
| [.agents/docs/ZERO_CONFIG_CLI_CONFIG_AND_OPENAI_API_CONTRACT.md](../ZERO_CONFIG_CLI_CONFIG_AND_OPENAI_API_CONTRACT.md) | Original detailed contract/context body; see annotated reading catalog and current owning TODO for authority. |

### 11.2 Historical handoff chapters and preservation volumes

93 files.

| Complete original path | Reading purpose / status |
|---|---|
| [.agents/docs/handoff-2026-10-17/PAUSED_FILTERS_CONTEXT.md](../handoff-2026-10-17/PAUSED_FILTERS_CONTEXT.md) | Historical17Oct paused acquisition/source context; current implementation must be independently established. |
| [.agents/docs/handoff-2026-10-17/PAUSED_XET_INDEX_CONTEXT.md](../handoff-2026-10-17/PAUSED_XET_INDEX_CONTEXT.md) | Historical17Oct paused acquisition/source context; current implementation must be independently established. |
| [.agents/docs/handoff-2026-10-17/XET_INDEX_DESIGN.md](../handoff-2026-10-17/XET_INDEX_DESIGN.md) | Historical17Oct paused acquisition/source context; current implementation must be independently established. |
| [.agents/docs/handoff-2026-10-17/paused-filters-oracle.py.txt](../handoff-2026-10-17/paused-filters-oracle.py.txt) | Preserved acquisition prototype/fixture; source-only draft, not current executable support. |
| [.agents/docs/handoff-2026-10-17/paused-filters.patch](../handoff-2026-10-17/paused-filters.patch) | Preserved acquisition prototype/fixture; source-only draft, not current executable support. |
| [.agents/docs/handoff-2026-10-17/paused-xet-index.patch](../handoff-2026-10-17/paused-xet-index.patch) | Preserved acquisition prototype/fixture; source-only draft, not current executable support. |
| [.agents/docs/handoff-2026-10-20/ARCHIVE_CONTENTS.json](../handoff-2026-10-20/ARCHIVE_CONTENTS.json) | Dated archive/body location metadata; preservation integrity is not execution proof. |
| [.agents/docs/handoff-2026-10-20/CANONICAL_SOURCEBOOK_01.md](../handoff-2026-10-20/CANONICAL_SOURCEBOOK_01.md) | Dated20Oct complete original spec/plan/research bodies; use CONTEXT_CATALOG source-to-volume map. |
| [.agents/docs/handoff-2026-10-20/CANONICAL_SOURCEBOOK_02.md](../handoff-2026-10-20/CANONICAL_SOURCEBOOK_02.md) | Dated20Oct complete original spec/plan/research bodies; use CONTEXT_CATALOG source-to-volume map. |
| [.agents/docs/handoff-2026-10-20/CANONICAL_SOURCEBOOK_03.md](../handoff-2026-10-20/CANONICAL_SOURCEBOOK_03.md) | Dated20Oct complete original spec/plan/research bodies; use CONTEXT_CATALOG source-to-volume map. |
| [.agents/docs/handoff-2026-10-20/CANONICAL_SOURCEBOOK_04.md](../handoff-2026-10-20/CANONICAL_SOURCEBOOK_04.md) | Dated20Oct complete original spec/plan/research bodies; use CONTEXT_CATALOG source-to-volume map. |
| [.agents/docs/handoff-2026-10-20/CANONICAL_SOURCEBOOK_05.md](../handoff-2026-10-20/CANONICAL_SOURCEBOOK_05.md) | Dated20Oct complete original spec/plan/research bodies; use CONTEXT_CATALOG source-to-volume map. |
| [.agents/docs/handoff-2026-10-20/CANONICAL_SOURCEBOOK_06.md](../handoff-2026-10-20/CANONICAL_SOURCEBOOK_06.md) | Dated20Oct complete original spec/plan/research bodies; use CONTEXT_CATALOG source-to-volume map. |
| [.agents/docs/handoff-2026-10-20/CANONICAL_SOURCEBOOK_07.md](../handoff-2026-10-20/CANONICAL_SOURCEBOOK_07.md) | Dated20Oct complete original spec/plan/research bodies; use CONTEXT_CATALOG source-to-volume map. |
| [.agents/docs/handoff-2026-10-20/CONTEXT_CATALOG.md](../handoff-2026-10-20/CONTEXT_CATALOG.md) | Dated20Oct standalone continuation/catalog/audit; full original context remains in its companion volumes. |
| [.agents/docs/handoff-2026-10-20/CURRENT_TAKEOVER.md](../handoff-2026-10-20/CURRENT_TAKEOVER.md) | Dated20Oct standalone continuation/catalog/audit; full original context remains in its companion volumes. |
| [.agents/docs/handoff-2026-10-20/DRAFT_PATCHES_01.md](../handoff-2026-10-20/DRAFT_PATCHES_01.md) | Dated preserved patch candidates; unaccepted status remains source-specific; never auto-apply. |
| [.agents/docs/handoff-2026-10-20/DRAFT_PATCHES_02.md](../handoff-2026-10-20/DRAFT_PATCHES_02.md) | Dated preserved patch candidates; unaccepted status remains source-specific; never auto-apply. |
| [.agents/docs/handoff-2026-10-20/DRAFT_PATCHES_03.md](../handoff-2026-10-20/DRAFT_PATCHES_03.md) | Dated preserved patch candidates; unaccepted status remains source-specific; never auto-apply. |
| [.agents/docs/handoff-2026-10-20/DRAFT_PATCHES_04.md](../handoff-2026-10-20/DRAFT_PATCHES_04.md) | Dated preserved patch candidates; unaccepted status remains source-specific; never auto-apply. |
| [.agents/docs/handoff-2026-10-20/DRAFT_PATCHES_05.md](../handoff-2026-10-20/DRAFT_PATCHES_05.md) | Dated preserved patch candidates; unaccepted status remains source-specific; never auto-apply. |
| [.agents/docs/handoff-2026-10-20/DRAFT_PATCHES_06.md](../handoff-2026-10-20/DRAFT_PATCHES_06.md) | Dated preserved patch candidates; unaccepted status remains source-specific; never auto-apply. |
| [.agents/docs/handoff-2026-10-20/DRAFT_PATCHES_07.md](../handoff-2026-10-20/DRAFT_PATCHES_07.md) | Dated preserved patch candidates; unaccepted status remains source-specific; never auto-apply. |
| [.agents/docs/handoff-2026-10-20/DRAFT_PATCHES_08.md](../handoff-2026-10-20/DRAFT_PATCHES_08.md) | Dated preserved patch candidates; unaccepted status remains source-specific; never auto-apply. |
| [.agents/docs/handoff-2026-10-20/DRAFT_PATCHES_09.md](../handoff-2026-10-20/DRAFT_PATCHES_09.md) | Dated preserved patch candidates; unaccepted status remains source-specific; never auto-apply. |
| [.agents/docs/handoff-2026-10-20/DRAFT_PATCHES_10.md](../handoff-2026-10-20/DRAFT_PATCHES_10.md) | Dated preserved patch candidates; unaccepted status remains source-specific; never auto-apply. |
| [.agents/docs/handoff-2026-10-20/DRAFT_PATCHES_11.md](../handoff-2026-10-20/DRAFT_PATCHES_11.md) | Dated preserved patch candidates; unaccepted status remains source-specific; never auto-apply. |
| [.agents/docs/handoff-2026-10-20/DRAFT_PATCHES_12.md](../handoff-2026-10-20/DRAFT_PATCHES_12.md) | Dated preserved patch candidates; unaccepted status remains source-specific; never auto-apply. |
| [.agents/docs/handoff-2026-10-20/DRAFT_PATCHES_13.md](../handoff-2026-10-20/DRAFT_PATCHES_13.md) | Dated preserved patch candidates; unaccepted status remains source-specific; never auto-apply. |
| [.agents/docs/handoff-2026-10-20/DRAFT_PATCHES_14.md](../handoff-2026-10-20/DRAFT_PATCHES_14.md) | Dated preserved patch candidates; unaccepted status remains source-specific; never auto-apply. |
| [.agents/docs/handoff-2026-10-20/DRAFT_PATCHES_15.md](../handoff-2026-10-20/DRAFT_PATCHES_15.md) | Dated preserved patch candidates; unaccepted status remains source-specific; never auto-apply. |
| [.agents/docs/handoff-2026-10-20/DRAFT_PATCHES_16.md](../handoff-2026-10-20/DRAFT_PATCHES_16.md) | Dated preserved patch candidates; unaccepted status remains source-specific; never auto-apply. |
| [.agents/docs/handoff-2026-10-20/DRAFT_PATCHES_17.md](../handoff-2026-10-20/DRAFT_PATCHES_17.md) | Dated preserved patch candidates; unaccepted status remains source-specific; never auto-apply. |
| [.agents/docs/handoff-2026-10-20/DRAFT_PATCHES_18.md](../handoff-2026-10-20/DRAFT_PATCHES_18.md) | Dated preserved patch candidates; unaccepted status remains source-specific; never auto-apply. |
| [.agents/docs/handoff-2026-10-20/DRAFT_PATCHES_19.md](../handoff-2026-10-20/DRAFT_PATCHES_19.md) | Dated preserved patch candidates; unaccepted status remains source-specific; never auto-apply. |
| [.agents/docs/handoff-2026-10-20/DRAFT_PATCHES_20.md](../handoff-2026-10-20/DRAFT_PATCHES_20.md) | Dated preserved patch candidates; unaccepted status remains source-specific; never auto-apply. |
| [.agents/docs/handoff-2026-10-20/DRAFT_PATCHES_21.md](../handoff-2026-10-20/DRAFT_PATCHES_21.md) | Dated preserved patch candidates; unaccepted status remains source-specific; never auto-apply. |
| [.agents/docs/handoff-2026-10-20/DRAFT_PATCHES_22.md](../handoff-2026-10-20/DRAFT_PATCHES_22.md) | Dated preserved patch candidates; unaccepted status remains source-specific; never auto-apply. |
| [.agents/docs/handoff-2026-10-20/DRAFT_PATCHES_23.md](../handoff-2026-10-20/DRAFT_PATCHES_23.md) | Dated preserved patch candidates; unaccepted status remains source-specific; never auto-apply. |
| [.agents/docs/handoff-2026-10-20/DRAFT_PATCHES_24.md](../handoff-2026-10-20/DRAFT_PATCHES_24.md) | Dated preserved patch candidates; unaccepted status remains source-specific; never auto-apply. |
| [.agents/docs/handoff-2026-10-20/DRAFT_PATCHES_25.md](../handoff-2026-10-20/DRAFT_PATCHES_25.md) | Dated preserved patch candidates; unaccepted status remains source-specific; never auto-apply. |
| [.agents/docs/handoff-2026-10-20/DRAFT_PATCHES_26.md](../handoff-2026-10-20/DRAFT_PATCHES_26.md) | Dated preserved patch candidates; unaccepted status remains source-specific; never auto-apply. |
| [.agents/docs/handoff-2026-10-20/DRAFT_PATCHES_27.md](../handoff-2026-10-20/DRAFT_PATCHES_27.md) | Dated preserved patch candidates; unaccepted status remains source-specific; never auto-apply. |
| [.agents/docs/handoff-2026-10-20/DRAFT_PATCHES_28.md](../handoff-2026-10-20/DRAFT_PATCHES_28.md) | Dated preserved patch candidates; unaccepted status remains source-specific; never auto-apply. |
| [.agents/docs/handoff-2026-10-20/DRAFT_PATCHES_29.md](../handoff-2026-10-20/DRAFT_PATCHES_29.md) | Dated preserved patch candidates; unaccepted status remains source-specific; never auto-apply. |
| [.agents/docs/handoff-2026-10-20/GDN_VLLM_ARITHMETIC_AUDIT.md](../handoff-2026-10-20/GDN_VLLM_ARITHMETIC_AUDIT.md) | Dated20Oct standalone continuation/catalog/audit; full original context remains in its companion volumes. |
| [.agents/docs/handoff-2026-10-20/HANDOFF_CONTENTS.json](../handoff-2026-10-20/HANDOFF_CONTENTS.json) | Dated archive/body location metadata; preservation integrity is not execution proof. |
| [.agents/docs/handoff-2026-10-20/HISTORY_DESIGN_AND_COVERAGE_AUDIT.md](../handoff-2026-10-20/HISTORY_DESIGN_AND_COVERAGE_AUDIT.md) | Dated20Oct standalone continuation/catalog/audit; full original context remains in its companion volumes. |
| [.agents/docs/handoff-2026-10-20/NUMERICAL_AND_MODEL_CONTINUATION.md](../handoff-2026-10-20/NUMERICAL_AND_MODEL_CONTINUATION.md) | Dated20Oct standalone continuation/catalog/audit; full original context remains in its companion volumes. |
| [.agents/docs/handoff-2026-10-20/PAUSED_CONTINUATION_TEXT_01.md](../handoff-2026-10-20/PAUSED_CONTINUATION_TEXT_01.md) | Dated selected paused source/harness/metadata text; not compiled or qualified by preservation. |
| [.agents/docs/handoff-2026-10-20/PAUSED_CONTINUATION_TEXT_02.md](../handoff-2026-10-20/PAUSED_CONTINUATION_TEXT_02.md) | Dated selected paused source/harness/metadata text; not compiled or qualified by preservation. |
| [.agents/docs/handoff-2026-10-20/PAUSED_CONTINUATION_TEXT_03.md](../handoff-2026-10-20/PAUSED_CONTINUATION_TEXT_03.md) | Dated selected paused source/harness/metadata text; not compiled or qualified by preservation. |
| [.agents/docs/handoff-2026-10-20/PAUSED_CONTINUATION_TEXT_04.md](../handoff-2026-10-20/PAUSED_CONTINUATION_TEXT_04.md) | Dated selected paused source/harness/metadata text; not compiled or qualified by preservation. |
| [.agents/docs/handoff-2026-10-20/PAUSED_CONTINUATION_TEXT_05.md](../handoff-2026-10-20/PAUSED_CONTINUATION_TEXT_05.md) | Dated selected paused source/harness/metadata text; not compiled or qualified by preservation. |
| [.agents/docs/handoff-2026-10-20/PAUSED_CONTINUATION_TEXT_06.md](../handoff-2026-10-20/PAUSED_CONTINUATION_TEXT_06.md) | Dated selected paused source/harness/metadata text; not compiled or qualified by preservation. |
| [.agents/docs/handoff-2026-10-20/PAUSED_CONTINUATION_TEXT_07.md](../handoff-2026-10-20/PAUSED_CONTINUATION_TEXT_07.md) | Dated selected paused source/harness/metadata text; not compiled or qualified by preservation. |
| [.agents/docs/handoff-2026-10-20/PAUSED_CONTINUATION_TEXT_08.md](../handoff-2026-10-20/PAUSED_CONTINUATION_TEXT_08.md) | Dated selected paused source/harness/metadata text; not compiled or qualified by preservation. |
| [.agents/docs/handoff-2026-10-20/PAUSED_CONTINUATION_TEXT_09.md](../handoff-2026-10-20/PAUSED_CONTINUATION_TEXT_09.md) | Dated selected paused source/harness/metadata text; not compiled or qualified by preservation. |
| [.agents/docs/handoff-2026-10-20/PAUSED_CONTINUATION_TEXT_10.md](../handoff-2026-10-20/PAUSED_CONTINUATION_TEXT_10.md) | Dated selected paused source/harness/metadata text; not compiled or qualified by preservation. |
| [.agents/docs/handoff-2026-10-20/PAUSED_CONTINUATION_TEXT_11.md](../handoff-2026-10-20/PAUSED_CONTINUATION_TEXT_11.md) | Dated selected paused source/harness/metadata text; not compiled or qualified by preservation. |
| [.agents/docs/handoff-2026-10-20/PAUSED_CONTINUATION_TEXT_12.md](../handoff-2026-10-20/PAUSED_CONTINUATION_TEXT_12.md) | Dated selected paused source/harness/metadata text; not compiled or qualified by preservation. |
| [.agents/docs/handoff-2026-10-20/PAUSED_CONTINUATION_TEXT_13.md](../handoff-2026-10-20/PAUSED_CONTINUATION_TEXT_13.md) | Dated selected paused source/harness/metadata text; not compiled or qualified by preservation. |
| [.agents/docs/handoff-2026-10-20/PAUSED_CONTINUATION_TEXT_14.md](../handoff-2026-10-20/PAUSED_CONTINUATION_TEXT_14.md) | Dated selected paused source/harness/metadata text; not compiled or qualified by preservation. |
| [.agents/docs/handoff-2026-10-20/PAUSED_CONTINUATION_TEXT_15.md](../handoff-2026-10-20/PAUSED_CONTINUATION_TEXT_15.md) | Dated selected paused source/harness/metadata text; not compiled or qualified by preservation. |
| [.agents/docs/handoff-2026-10-20/PAUSED_CONTINUATION_TEXT_16.md](../handoff-2026-10-20/PAUSED_CONTINUATION_TEXT_16.md) | Dated selected paused source/harness/metadata text; not compiled or qualified by preservation. |
| [.agents/docs/handoff-2026-10-20/PAUSED_CONTINUATION_TEXT_17.md](../handoff-2026-10-20/PAUSED_CONTINUATION_TEXT_17.md) | Dated selected paused source/harness/metadata text; not compiled or qualified by preservation. |
| [.agents/docs/handoff-2026-10-20/PAUSED_CONTINUATION_TEXT_18.md](../handoff-2026-10-20/PAUSED_CONTINUATION_TEXT_18.md) | Dated selected paused source/harness/metadata text; not compiled or qualified by preservation. |
| [.agents/docs/handoff-2026-10-20/PAUSED_CONTINUATION_TEXT_19.md](../handoff-2026-10-20/PAUSED_CONTINUATION_TEXT_19.md) | Dated selected paused source/harness/metadata text; not compiled or qualified by preservation. |
| [.agents/docs/handoff-2026-10-20/PAUSED_CONTINUATION_TEXT_20.md](../handoff-2026-10-20/PAUSED_CONTINUATION_TEXT_20.md) | Dated selected paused source/harness/metadata text; not compiled or qualified by preservation. |
| [.agents/docs/handoff-2026-10-20/PAUSED_CONTINUATION_TEXT_21.md](../handoff-2026-10-20/PAUSED_CONTINUATION_TEXT_21.md) | Dated selected paused source/harness/metadata text; not compiled or qualified by preservation. |
| [.agents/docs/handoff-2026-10-20/PAUSED_CONTINUATION_TEXT_22.md](../handoff-2026-10-20/PAUSED_CONTINUATION_TEXT_22.md) | Dated selected paused source/harness/metadata text; not compiled or qualified by preservation. |
| [.agents/docs/handoff-2026-10-20/PAUSED_CONTINUATION_TEXT_23.md](../handoff-2026-10-20/PAUSED_CONTINUATION_TEXT_23.md) | Dated selected paused source/harness/metadata text; not compiled or qualified by preservation. |
| [.agents/docs/handoff-2026-10-20/PAUSED_CONTINUATION_TEXT_24.md](../handoff-2026-10-20/PAUSED_CONTINUATION_TEXT_24.md) | Dated selected paused source/harness/metadata text; not compiled or qualified by preservation. |
| [.agents/docs/handoff-2026-10-20/PAUSED_CONTINUATION_TEXT_25.md](../handoff-2026-10-20/PAUSED_CONTINUATION_TEXT_25.md) | Dated selected paused source/harness/metadata text; not compiled or qualified by preservation. |
| [.agents/docs/handoff-2026-10-20/PAUSED_CONTINUATION_TEXT_26.md](../handoff-2026-10-20/PAUSED_CONTINUATION_TEXT_26.md) | Dated selected paused source/harness/metadata text; not compiled or qualified by preservation. |
| [.agents/docs/handoff-2026-10-20/PAUSED_CONTINUATION_TEXT_27.md](../handoff-2026-10-20/PAUSED_CONTINUATION_TEXT_27.md) | Dated selected paused source/harness/metadata text; not compiled or qualified by preservation. |
| [.agents/docs/handoff-2026-10-20/PAUSED_CONTINUATION_TEXT_28.md](../handoff-2026-10-20/PAUSED_CONTINUATION_TEXT_28.md) | Dated selected paused source/harness/metadata text; not compiled or qualified by preservation. |
| [.agents/docs/handoff-2026-10-20/PAUSED_CONTINUATION_TEXT_29.md](../handoff-2026-10-20/PAUSED_CONTINUATION_TEXT_29.md) | Dated selected paused source/harness/metadata text; not compiled or qualified by preservation. |
| [.agents/docs/handoff-2026-10-20/PAUSED_LICENSE_AND_CUDA_SOURCE.md](../handoff-2026-10-20/PAUSED_LICENSE_AND_CUDA_SOURCE.md) | Dated20Oct standalone continuation/catalog/audit; full original context remains in its companion volumes. |
| [.agents/docs/handoff-2026-10-20/RECOVERED_HISTORY_01.md](../handoff-2026-10-20/RECOVERED_HISTORY_01.md) | Recovered original historical bodies with recorded source refs; see recovery map above. |
| [.agents/docs/handoff-2026-10-20/RECOVERED_HISTORY_02.md](../handoff-2026-10-20/RECOVERED_HISTORY_02.md) | Recovered original historical bodies with recorded source refs; see recovery map above. |
| [.agents/docs/handoff-2026-10-20/RECOVERED_HISTORY_03.md](../handoff-2026-10-20/RECOVERED_HISTORY_03.md) | Recovered original historical bodies with recorded source refs; see recovery map above. |
| [.agents/docs/handoff-2026-10-20/RECOVERED_HISTORY_04.md](../handoff-2026-10-20/RECOVERED_HISTORY_04.md) | Recovered original historical bodies with recorded source refs; see recovery map above. |
| [.agents/docs/handoff-2026-10-20/RUN_NOTES_01.md](../handoff-2026-10-20/RUN_NOTES_01.md) | Dated retained run Markdown; each case keeps its original failed/interrupted/accepted scope. |
| [.agents/docs/handoff-2026-10-20/RUN_NOTES_02.md](../handoff-2026-10-20/RUN_NOTES_02.md) | Dated retained run Markdown; each case keeps its original failed/interrupted/accepted scope. |
| [.agents/docs/handoff-2026-10-20/RUN_NOTES_03.md](../handoff-2026-10-20/RUN_NOTES_03.md) | Dated retained run Markdown; each case keeps its original failed/interrupted/accepted scope. |
| [.agents/docs/handoff-2026-10-20/RUN_NOTES_04.md](../handoff-2026-10-20/RUN_NOTES_04.md) | Dated retained run Markdown; each case keeps its original failed/interrupted/accepted scope. |
| [.agents/docs/handoff-2026-10-20/RUN_NOTES_05.md](../handoff-2026-10-20/RUN_NOTES_05.md) | Dated retained run Markdown; each case keeps its original failed/interrupted/accepted scope. |
| [.agents/docs/handoff-2026-10-20/RUN_NOTES_06.md](../handoff-2026-10-20/RUN_NOTES_06.md) | Dated retained run Markdown; each case keeps its original failed/interrupted/accepted scope. |
| [.agents/docs/handoff-2026-10-20/RUN_NOTES_07.md](../handoff-2026-10-20/RUN_NOTES_07.md) | Dated retained run Markdown; each case keeps its original failed/interrupted/accepted scope. |
| [.agents/docs/handoff-2026-10-20/SERVING_PERFORMANCE_AND_PAUSED_GPU_WORK.md](../handoff-2026-10-20/SERVING_PERFORMANCE_AND_PAUSED_GPU_WORK.md) | Dated20Oct standalone continuation/catalog/audit; full original context remains in its companion volumes. |
| [.agents/docs/handoff-2026-10-20/VALIDATION.md](../handoff-2026-10-20/VALIDATION.md) | Dated20Oct standalone continuation/catalog/audit; full original context remains in its companion volumes. |
| [.agents/docs/handoff-2026-10-20/WORKSPACE_SNAPSHOT.md](../handoff-2026-10-20/WORKSPACE_SNAPSHOT.md) | Dated20Oct standalone continuation/catalog/audit; full original context remains in its companion volumes. |

### 11.3 Current plans, original phase contracts and archived decisions

56 files.

| Complete original path | Reading purpose / status |
|---|---|
| [.agents/plans/EXPERIMENT_PROGRAM.md](../../plans/EXPERIMENT_PROGRAM.md) | Original architecture/capability/dependency/acceptance body; initial NOT_STARTED or planned labels are dated. |
| [.agents/plans/IMMUTABLE_SHARDED_SOURCE_IMPLEMENTATION_SPEC.md](../../plans/IMMUTABLE_SHARDED_SOURCE_IMPLEMENTATION_SPEC.md) | Original detailed contract/context body; see annotated reading catalog and current owning TODO for authority. |
| [.agents/plans/IMPLEMENTATION_ROADMAP.md](../../plans/IMPLEMENTATION_ROADMAP.md) | Original architecture/capability/dependency/acceptance body; initial NOT_STARTED or planned labels are dated. |
| [.agents/plans/INTEGRATED_PHASE_0_AND_DELIVERY_GATES.md](../../plans/INTEGRATED_PHASE_0_AND_DELIVERY_GATES.md) | Original architecture/capability/dependency/acceptance body; initial NOT_STARTED or planned labels are dated. |
| [.agents/plans/MODEL_ACQUISITION_IMPLEMENTATION_SPEC.md](../../plans/MODEL_ACQUISITION_IMPLEMENTATION_SPEC.md) | Original detailed contract/context body; see annotated reading catalog and current owning TODO for authority. |
| [.agents/plans/NEXT_AGENT_PROMPT_2026-10-17.md](../../plans/NEXT_AGENT_PROMPT_2026-10-17.md) | Historical ready-to-copy prompt; old scope, role, command and permission assumptions require current instructions. |
| [.agents/plans/NEXT_AGENT_PROMPT_2026-10-19.md](../../plans/NEXT_AGENT_PROMPT_2026-10-19.md) | Historical ready-to-copy prompt; old scope, role, command and permission assumptions require current instructions. |
| [.agents/plans/NEXT_AGENT_PROMPT_2026-10-20.md](../../plans/NEXT_AGENT_PROMPT_2026-10-20.md) | Historical ready-to-copy prompt; old scope, role, command and permission assumptions require current instructions. |
| [.agents/plans/NEXT_AGENT_PROMPT_2026-10-25-NIGHT.md](../../plans/NEXT_AGENT_PROMPT_2026-10-25-NIGHT.md) | Historical ready-to-copy prompt; old scope, role, command and permission assumptions require current instructions. |
| [.agents/plans/NEXT_AGENT_PROMPT_2026-10-25.md](../../plans/NEXT_AGENT_PROMPT_2026-10-25.md) | Historical ready-to-copy prompt; old scope, role, command and permission assumptions require current instructions. |
| [.agents/plans/README.md](../../plans/README.md) | Current planning layout/ownership/tracking rulebook; latest explicit user model settings override old launch recipes. |
| [.agents/plans/RELEASE_ACCEPTANCE_GATES.md](../../plans/RELEASE_ACCEPTANCE_GATES.md) | Original architecture/capability/dependency/acceptance body; initial NOT_STARTED or planned labels are dated. |
| [.agents/plans/TODO.md](../../plans/TODO.md) | Live single plan queue; coordinator alone reconciles status/ownership. |
| [.agents/plans/ZEROGRAVITY_MASTER_PLAN_2026-10-10.md](../../plans/ZEROGRAVITY_MASTER_PLAN_2026-10-10.md) | Original architecture/capability/dependency/acceptance body; initial NOT_STARTED or planned labels are dated. |
| [.agents/plans/archive/history/AGENT_IMPLEMENTATION_PLAYBOOK.md](../../plans/archive/history/AGENT_IMPLEMENTATION_PLAYBOOK.md) | Original archived policy/status/decision/phase reference; apply current supersessions and path remaps. |
| [.agents/plans/archive/history/CURRENT_STATUS_AND_TRACKING.md](../../plans/archive/history/CURRENT_STATUS_AND_TRACKING.md) | Original archived policy/status/decision/phase reference; apply current supersessions and path remaps. |
| [.agents/plans/archive/history/DECISION_REGISTER.md](../../plans/archive/history/DECISION_REGISTER.md) | Original archived policy/status/decision/phase reference; apply current supersessions and path remaps. |
| [.agents/plans/archive/history/OPEN_CONTRIBUTION_AI_DEVELOPMENT_AND_MIT_POLICY_2026-10-15.md](../../plans/archive/history/OPEN_CONTRIBUTION_AI_DEVELOPMENT_AND_MIT_POLICY_2026-10-15.md) | Original archived policy/status/decision/phase reference; apply current supersessions and path remaps. |
| [.agents/plans/archive/history/REQUIREMENTS_AND_TEST_TRACEABILITY.md](../../plans/archive/history/REQUIREMENTS_AND_TEST_TRACEABILITY.md) | Original archived policy/status/decision/phase reference; apply current supersessions and path remaps. |
| [.agents/plans/archive/history/accepted-baseline/TODO.md](../../plans/archive/history/accepted-baseline/TODO.md) | Archived completed/provenance and historical open/check scope; current owning TODO/source outranks dated status. |
| [.agents/plans/archive/history/handoff-cleanup/TODO.md](../../plans/archive/history/handoff-cleanup/TODO.md) | Archived completed/provenance and historical open/check scope; current owning TODO/source outranks dated status. |
| [.agents/plans/archive/history/historical-records/TODO.md](../../plans/archive/history/historical-records/TODO.md) | Archived completed/provenance and historical open/check scope; current owning TODO/source outranks dated status. |
| [.agents/plans/cli-device-routing/README.md](../../plans/cli-device-routing/README.md) | Owning plan design and linked contracts; current TODO/source establish implementation status. |
| [.agents/plans/cli-device-routing/TODO.md](../../plans/cli-device-routing/TODO.md) | Live owning plan implementation/test/provenance and unfinished work; read complete multiline entries. |
| [.agents/plans/cpu-gpu-optimization/BACKEND_PARITY.md](../../plans/cpu-gpu-optimization/BACKEND_PARITY.md) | Separate physical source/transport/numerical/backend evidence cells; no cross-backend inference equivalence. |
| [.agents/plans/cpu-gpu-optimization/BENCHMARK_CONTAINER_TERMINATION.md](../../plans/cpu-gpu-optimization/BENCHMARK_CONTAINER_TERMINATION.md) | Benchmark/process/container EOF and termination investigation; observed exit status is not an inferred cause. |
| [.agents/plans/cpu-gpu-optimization/SHARED_MEMORY_AND_BOUNDARIES.md](../../plans/cpu-gpu-optimization/SHARED_MEMORY_AND_BOUNDARIES.md) | Current pool/ring/source-reader/preparation/retirement contract under original physical owners. |
| [.agents/plans/cpu-gpu-optimization/SHORT_HF_RESULTS.md](../../plans/cpu-gpu-optimization/SHORT_HF_RESULTS.md) | Historical bounded source/profile benchmark observations and losses; not a current universal performance claim. |
| [.agents/plans/cpu-gpu-optimization/TODO.md](../../plans/cpu-gpu-optimization/TODO.md) | Live owning plan implementation/test/provenance and unfinished work; read complete multiline entries. |
| [.agents/plans/distributed-execution/TODO.md](../../plans/distributed-execution/TODO.md) | Live owning plan implementation/test/provenance and unfinished work; read complete multiline entries. |
| [.agents/plans/memory-and-streaming/README.md](../../plans/memory-and-streaming/README.md) | Owning plan design and linked contracts; current TODO/source establish implementation status. |
| [.agents/plans/memory-and-streaming/TODO.md](../../plans/memory-and-streaming/TODO.md) | Live owning plan implementation/test/provenance and unfinished work; read complete multiline entries. |
| [.agents/plans/model-acquisition/README.md](../../plans/model-acquisition/README.md) | Owning plan design and linked contracts; current TODO/source establish implementation status. |
| [.agents/plans/model-acquisition/TODO.md](../../plans/model-acquisition/TODO.md) | Live owning plan implementation/test/provenance and unfinished work; read complete multiline entries. |
| [.agents/plans/model-acquisition/archive/through-a774ded/TODO.md](../../plans/model-acquisition/archive/through-a774ded/TODO.md) | Archived completed/provenance and historical open/check scope; current owning TODO/source outranks dated status. |
| [.agents/plans/model-acquisition/archive/through-fce1353/TODO.md](../../plans/model-acquisition/archive/through-fce1353/TODO.md) | Archived completed/provenance and historical open/check scope; current owning TODO/source outranks dated status. |
| [.agents/plans/model-execution/README.md](../../plans/model-execution/README.md) | Owning plan design and linked contracts; current TODO/source establish implementation status. |
| [.agents/plans/model-execution/TODO.md](../../plans/model-execution/TODO.md) | Live owning plan implementation/test/provenance and unfinished work; read complete multiline entries. |
| [.agents/plans/multimodal/TODO.md](../../plans/multimodal/TODO.md) | Live owning plan implementation/test/provenance and unfinished work; read complete multiline entries. |
| [.agents/plans/numerics-and-context/CORRECTNESS_FIRST_VLLM_PLAN_2026-10-20.md](../../plans/numerics-and-context/CORRECTNESS_FIRST_VLLM_PLAN_2026-10-20.md) | Pinned-profile numerical plan; preserve GGUF/AR1/default-bulk/Torch/vLLM distinctions and actual failures. |
| [.agents/plans/numerics-and-context/CORRECTNESS_POLICY.md](../../plans/numerics-and-context/CORRECTNESS_POLICY.md) | Current26Oct all-model semantics/safety, independent operator/model accuracy and separate determinism policy. |
| [.agents/plans/numerics-and-context/TODO.md](../../plans/numerics-and-context/TODO.md) | Live owning plan implementation/test/provenance and unfinished work; read complete multiline entries. |
| [.agents/plans/pipeline-composition/FORMATS.md](../../plans/pipeline-composition/FORMATS.md) | Current26Oct exact codec/precision homes, implemented variants, explicit unsupported cells and required qualification. |
| [.agents/plans/pipeline-composition/PROFILE_POLICY_AND_COMPONENT_REUSE.md](../../plans/pipeline-composition/PROFILE_POLICY_AND_COMPONENT_REUSE.md) | Explicit approximation, replaceable components, shared policy context and native reuse/port correctness. |
| [.agents/plans/pipeline-composition/README.md](../../plans/pipeline-composition/README.md) | Owning plan design and linked contracts; current TODO/source establish implementation status. |
| [.agents/plans/pipeline-composition/REGIONS_AND_PROFILES.md](../../plans/pipeline-composition/REGIONS_AND_PROFILES.md) | Recursive semantic/task scope and workload-profile design over existing two representations; not a new runtime. |
| [.agents/plans/pipeline-composition/SHARED_EXECUTION_ARCHITECTURE.md](../../plans/pipeline-composition/SHARED_EXECUTION_ARCHITECTURE.md) | Current shared ForwardPlan ownership design and dated source reviews; finite candidates never mint arbitrary support. |
| [.agents/plans/pipeline-composition/STRUCTURE_AND_NAMING.md](../../plans/pipeline-composition/STRUCTURE_AND_NAMING.md) | Current26Oct actual owner/dependency/naming boundaries and family/caller migration; no qualification from organization. |
| [.agents/plans/pipeline-composition/TODO.md](../../plans/pipeline-composition/TODO.md) | Live owning plan implementation/test/provenance and unfinished work; read complete multiline entries. |
| [.agents/plans/platform-and-release/TODO.md](../../plans/platform-and-release/TODO.md) | Live owning plan implementation/test/provenance and unfinished work; read complete multiline entries. |
| [.agents/plans/qwen-parity/API_ROUTE_FRONTIER.csv](../../plans/qwen-parity/API_ROUTE_FRONTIER.csv) | Pinned public/internal Qwen compatibility frontier data; inventory is not implementation/parity acceptance. |
| [.agents/plans/qwen-parity/CLI_OPTION_FRONTIER.csv](../../plans/qwen-parity/CLI_OPTION_FRONTIER.csv) | Pinned public/internal Qwen compatibility frontier data; inventory is not implementation/parity acceptance. |
| [.agents/plans/serving-and-speculation/BATCHING_AND_PAGED_STATE.md](../../plans/serving-and-speculation/BATCHING_AND_PAGED_STATE.md) | Serving batching/paging contract and dated finite acceptance; read current owning TODO and later source. |
| [.agents/plans/serving-and-speculation/FUSED_GPU_BATCHING.md](../../plans/serving-and-speculation/FUSED_GPU_BATCHING.md) | Serving batching/paging contract and dated finite acceptance; read current owning TODO and later source. |
| [.agents/plans/serving-and-speculation/README.md](../../plans/serving-and-speculation/README.md) | Owning plan design and linked contracts; current TODO/source establish implementation status. |
| [.agents/plans/serving-and-speculation/TODO.md](../../plans/serving-and-speculation/TODO.md) | Live owning plan implementation/test/provenance and unfinished work; read complete multiline entries. |

### 11.4 Research/model/modality/optimization dossiers outside docs/plans

41 files.

| Complete original path | Reading purpose / status |
|---|---|
| [.agents/research/EXTREME_MODEL_SPECIALIZATION_INTEGRATION_AND_ACCEPTANCE_2026-10-15.md](../../research/EXTREME_MODEL_SPECIALIZATION_INTEGRATION_AND_ACCEPTANCE_2026-10-15.md) | Dated research/source/acceptance proposal; preserve exact source/namespace and distinguish planned from executed. |
| [.agents/research/GEMMA4_FAMILY_CONTRACTS_2026-10-16.md](../../research/GEMMA4_FAMILY_CONTRACTS_2026-10-16.md) | Dated research/source/acceptance proposal; preserve exact source/namespace and distinguish planned from executed. |
| [.agents/research/GPU_TEXT_FRONTEND_AND_DEVICE_ORCHESTRATION_ROADMAP_2026-10-16.md](../../research/GPU_TEXT_FRONTEND_AND_DEVICE_ORCHESTRATION_ROADMAP_2026-10-16.md) | Dated research/source/acceptance proposal; preserve exact source/namespace and distinguish planned from executed. |
| [.agents/research/HETEROGENEOUS_PREFILL_AND_REPLAY_ACCEPTANCE_2026-10-15.md](../../research/HETEROGENEOUS_PREFILL_AND_REPLAY_ACCEPTANCE_2026-10-15.md) | Dated research/source/acceptance proposal; preserve exact source/namespace and distinguish planned from executed. |
| [.agents/research/INITIAL_HARDWARE_TARGETS_AND_OPTIMIZATION_2026-10-16.md](../../research/INITIAL_HARDWARE_TARGETS_AND_OPTIMIZATION_2026-10-16.md) | Dated research/source/acceptance proposal; preserve exact source/namespace and distinguish planned from executed. |
| [.agents/research/MODEL_PULL_XDG_CACHE_AND_SOURCE_PARITY_2026-10-16.md](../../research/MODEL_PULL_XDG_CACHE_AND_SOURCE_PARITY_2026-10-16.md) | Dated research/source/acceptance proposal; preserve exact source/namespace and distinguish planned from executed. |
| [.agents/research/MULTIMODAL_MODEL_FAMILY_SOURCE_ATLAS_AND_IMPLEMENTATION_2026-10-14.md](../../research/MULTIMODAL_MODEL_FAMILY_SOURCE_ATLAS_AND_IMPLEMENTATION_2026-10-14.md) | Dated research/source/acceptance proposal; preserve exact source/namespace and distinguish planned from executed. |
| [.agents/research/MULTIMODAL_V5_1_ACCEPTANCE_AND_ROADMAP_2026-10-15.md](../../research/MULTIMODAL_V5_1_ACCEPTANCE_AND_ROADMAP_2026-10-15.md) | Dated research/source/acceptance proposal; preserve exact source/namespace and distinguish planned from executed. |
| [.agents/research/MULTIMODAL_V5_1_DECISIONS_AND_PROVENANCE_2026-10-15.md](../../research/MULTIMODAL_V5_1_DECISIONS_AND_PROVENANCE_2026-10-15.md) | Dated research/source/acceptance proposal; preserve exact source/namespace and distinguish planned from executed. |
| [.agents/research/NATIVE_FP32_FMA_QUALIFICATION_PROPOSAL_2026-10-16.md](../../research/NATIVE_FP32_FMA_QUALIFICATION_PROPOSAL_2026-10-16.md) | Dated research/source/acceptance proposal; preserve exact source/namespace and distinguish planned from executed. |
| [.agents/research/PULL_A1_LOCAL_XDG_CACHE_2026-10-16.md](../../research/PULL_A1_LOCAL_XDG_CACHE_2026-10-16.md) | Dated research/source/acceptance proposal; preserve exact source/namespace and distinguish planned from executed. |
| [.agents/research/QWEN_BASICS_PARITY_AND_ADVANCED_DECODING_ROADMAP_2026-10-16.md](../../research/QWEN_BASICS_PARITY_AND_ADVANCED_DECODING_ROADMAP_2026-10-16.md) | Dated research/source/acceptance proposal; preserve exact source/namespace and distinguish planned from executed. |
| [.agents/research/README_PRE_PRODUCT_REWRITE_SNAPSHOT_2026-10-15.md](../../research/README_PRE_PRODUCT_REWRITE_SNAPSHOT_2026-10-15.md) | Retained historical README pointer/context; root README remains immutable to agents. |
| [.agents/research/SPECIALIZATION_SOURCE_ATLAS_2026-10-15.md](../../research/SPECIALIZATION_SOURCE_ATLAS_2026-10-15.md) | Dated research/source/acceptance proposal; preserve exact source/namespace and distinguish planned from executed. |
| [.agents/research/SSD_STREAMING_IMPLEMENTATION_ROADMAP_AND_ACCEPTANCE_2026-10-15.md](../../research/SSD_STREAMING_IMPLEMENTATION_ROADMAP_AND_ACCEPTANCE_2026-10-15.md) | Dated research/source/acceptance proposal; preserve exact source/namespace and distinguish planned from executed. |
| [.agents/research/STRIX_HALO_OPTIMIZATION_DOSSIER.md](../../research/STRIX_HALO_OPTIMIZATION_DOSSIER.md) | Dated research/source/acceptance proposal; preserve exact source/namespace and distinguish planned from executed. |
| [.agents/research/VLLM_PARITY_AND_COMBINATORY_SPECIALIZATION_ACCEPTANCE_2026-10-15.md](../../research/VLLM_PARITY_AND_COMBINATORY_SPECIALIZATION_ACCEPTANCE_2026-10-15.md) | Dated research/source/acceptance proposal; preserve exact source/namespace and distinguish planned from executed. |
| [.agents/research/engine-optimization-comparison-2026-10-21/README.md](../../research/engine-optimization-comparison-2026-10-21/README.md) | Pinned optimization-by-optimization source research/adoption accounting; no automatic runtime or benchmark claim. |
| [.agents/research/engine-optimization-comparison-2026-10-21/composition-review.md](../../research/engine-optimization-comparison-2026-10-21/composition-review.md) | Pinned optimization-by-optimization source research/adoption accounting; no automatic runtime or benchmark claim. |
| [.agents/research/engine-optimization-comparison-2026-10-21/deepinfer-inventory.md](../../research/engine-optimization-comparison-2026-10-21/deepinfer-inventory.md) | Pinned optimization-by-optimization source research/adoption accounting; no automatic runtime or benchmark claim. |
| [.agents/research/engine-optimization-comparison-2026-10-21/deepinfer.md](../../research/engine-optimization-comparison-2026-10-21/deepinfer.md) | Pinned optimization-by-optimization source research/adoption accounting; no automatic runtime or benchmark claim. |
| [.agents/research/engine-optimization-comparison-2026-10-21/dwarfstar.md](../../research/engine-optimization-comparison-2026-10-21/dwarfstar.md) | Pinned optimization-by-optimization source research/adoption accounting; no automatic runtime or benchmark claim. |
| [.agents/research/engine-optimization-comparison-2026-10-21/exllamav3-adoption-scope.md](../../research/engine-optimization-comparison-2026-10-21/exllamav3-adoption-scope.md) | Pinned optimization-by-optimization source research/adoption accounting; no automatic runtime or benchmark claim. |
| [.agents/research/engine-optimization-comparison-2026-10-21/halogen-inventory.md](../../research/engine-optimization-comparison-2026-10-21/halogen-inventory.md) | Pinned optimization-by-optimization source research/adoption accounting; no automatic runtime or benchmark claim. |
| [.agents/research/engine-optimization-comparison-2026-10-21/halogen.md](../../research/engine-optimization-comparison-2026-10-21/halogen.md) | Pinned optimization-by-optimization source research/adoption accounting; no automatic runtime or benchmark claim. |
| [.agents/research/engine-optimization-comparison-2026-10-21/llamacpp.md](../../research/engine-optimization-comparison-2026-10-21/llamacpp.md) | Pinned optimization-by-optimization source research/adoption accounting; no automatic runtime or benchmark claim. |
| [.agents/research/engine-optimization-comparison-2026-10-21/maya-inventory.md](../../research/engine-optimization-comparison-2026-10-21/maya-inventory.md) | Pinned optimization-by-optimization source research/adoption accounting; no automatic runtime or benchmark claim. |
| [.agents/research/engine-optimization-comparison-2026-10-21/maya.md](../../research/engine-optimization-comparison-2026-10-21/maya.md) | Pinned optimization-by-optimization source research/adoption accounting; no automatic runtime or benchmark claim. |
| [.agents/research/engine-optimization-comparison-2026-10-21/nvidia-guide-inventory.md](../../research/engine-optimization-comparison-2026-10-21/nvidia-guide-inventory.md) | Pinned optimization-by-optimization source research/adoption accounting; no automatic runtime or benchmark claim. |
| [.agents/research/engine-optimization-comparison-2026-10-21/nvidia-guide.md](../../research/engine-optimization-comparison-2026-10-21/nvidia-guide.md) | Pinned optimization-by-optimization source research/adoption accounting; no automatic runtime or benchmark claim. |
| [.agents/research/engine-optimization-comparison-2026-10-21/profile-coverage.md](../../research/engine-optimization-comparison-2026-10-21/profile-coverage.md) | Pinned optimization-by-optimization source research/adoption accounting; no automatic runtime or benchmark claim. |
| [.agents/research/engine-optimization-comparison-2026-10-21/tensorfold-inventory.md](../../research/engine-optimization-comparison-2026-10-21/tensorfold-inventory.md) | Pinned optimization-by-optimization source research/adoption accounting; no automatic runtime or benchmark claim. |
| [.agents/research/engine-optimization-comparison-2026-10-21/tensorfold.md](../../research/engine-optimization-comparison-2026-10-21/tensorfold.md) | Pinned optimization-by-optimization source research/adoption accounting; no automatic runtime or benchmark claim. |
| [.agents/research/engine-optimization-comparison-2026-10-21/vllm.md](../../research/engine-optimization-comparison-2026-10-21/vllm.md) | Pinned optimization-by-optimization source research/adoption accounting; no automatic runtime or benchmark claim. |
| [.agents/research/multimodal/ASR_AND_STREAMING_SPEECH_UNDERSTANDING_2026-10-15.md](../../research/multimodal/ASR_AND_STREAMING_SPEECH_UNDERSTANDING_2026-10-15.md) | Detailed source modality dossier and negative cases; declared models/quality/codecs/rights require actual proof. |
| [.agents/research/multimodal/CROSS_STREAM_ARCHITECTURE_RECONCILIATION_2026-10-15.md](../../research/multimodal/CROSS_STREAM_ARCHITECTURE_RECONCILIATION_2026-10-15.md) | Detailed source modality dossier and negative cases; declared models/quality/codecs/rights require actual proof. |
| [.agents/research/multimodal/DECISION_MODELS_TERNARY_AND_SPECIALIST_TEXT_2026-10-15.md](../../research/multimodal/DECISION_MODELS_TERNARY_AND_SPECIALIST_TEXT_2026-10-15.md) | Detailed source modality dossier and negative cases; declared models/quality/codecs/rights require actual proof. |
| [.agents/research/multimodal/IMAGE_GENERATION_EDITING_AND_FLOW_MODELS_2026-10-15.md](../../research/multimodal/IMAGE_GENERATION_EDITING_AND_FLOW_MODELS_2026-10-15.md) | Detailed source modality dossier and negative cases; declared models/quality/codecs/rights require actual proof. |
| [.agents/research/multimodal/SERVING_API_AND_REALTIME_PRIOR_ART_2026-10-15.md](../../research/multimodal/SERVING_API_AND_REALTIME_PRIOR_ART_2026-10-15.md) | Detailed source modality dossier and negative cases; declared models/quality/codecs/rights require actual proof. |
| [.agents/research/multimodal/TTS_VOICE_CODECS_AND_FULL_DUPLEX_2026-10-15.md](../../research/multimodal/TTS_VOICE_CODECS_AND_FULL_DUPLEX_2026-10-15.md) | Detailed source modality dossier and negative cases; declared models/quality/codecs/rights require actual proof. |
| [.agents/research/multimodal/VIDEO_TEMPORAL_MODELS_AND_SYNCHRONIZED_AV_2026-10-15.md](../../research/multimodal/VIDEO_TEMPORAL_MODELS_AND_SYNCHRONIZED_AV_2026-10-15.md) | Detailed source modality dossier and negative cases; declared models/quality/codecs/rights require actual proof. |

### 11.5 Prior-art, neutrality/license and review bodies outside docs/plans

12 files.

| Complete original path | Reading purpose / status |
|---|---|
| [.agents/prior-art/ACCELERATOR_PRIOR_ART_AND_PLATFORM_RESEARCH_2026-10-12.md](../../prior-art/ACCELERATOR_PRIOR_ART_AND_PLATFORM_RESEARCH_2026-10-12.md) | Attributed pinned prior art/research, not an engine dependency or independent user approval. |
| [.agents/prior-art/ARCHITECTURE_RECONCILIATION_AND_AUTHORITY.md](../../prior-art/ARCHITECTURE_RECONCILIATION_AND_AUTHORITY.md) | Attributed pinned prior art/research, not an engine dependency or independent user approval. |
| [.agents/prior-art/ARCHITECTURE_V4_CRITICAL_REVIEW_2026-10-12.md](../../prior-art/ARCHITECTURE_V4_CRITICAL_REVIEW_2026-10-12.md) | Original critical/final review rationale and unresolved risks; proposals/status retain original date. |
| [.agents/prior-art/COMPARATIVE_COMPILER_AND_RUNTIME_RESEARCH_2026-10-10.md](../../prior-art/COMPARATIVE_COMPILER_AND_RUNTIME_RESEARCH_2026-10-10.md) | Attributed pinned prior art/research, not an engine dependency or independent user approval. |
| [.agents/prior-art/DISTRIBUTED_PRIOR_ART_AND_TUNNEL_RESEARCH_2026-10-12.md](../../prior-art/DISTRIBUTED_PRIOR_ART_AND_TUNNEL_RESEARCH_2026-10-12.md) | Attributed pinned prior art/research, not an engine dependency or independent user approval. |
| [.agents/prior-art/FINAL_ARCHITECTURE_REVIEW_2026-10-12.md](../../prior-art/FINAL_ARCHITECTURE_REVIEW_2026-10-12.md) | Original critical/final review rationale and unresolved risks; proposals/status retain original date. |
| [.agents/prior-art/GUFO_FORK_AND_XDNA2_PREFILL_SOURCE_RESEARCH_2026-10-12.md](../../prior-art/GUFO_FORK_AND_XDNA2_PREFILL_SOURCE_RESEARCH_2026-10-12.md) | Attributed pinned prior art/research, not an engine dependency or independent user approval. |
| [.agents/prior-art/MULTI_MODEL_ROUTER_AND_API_PRIOR_ART_RESEARCH_2026-10-12.md](../../prior-art/MULTI_MODEL_ROUTER_AND_API_PRIOR_ART_RESEARCH_2026-10-12.md) | Attributed pinned prior art/research, not an engine dependency or independent user approval. |
| [.agents/prior-art/PRIOR_ART_LICENSE_AND_PROVENANCE_POLICY.md](../../prior-art/PRIOR_ART_LICENSE_AND_PROVENANCE_POLICY.md) | Current neutrality/license policy with21Oct reuse-first supersession; source/file/NOTICE and model rights remain independent. |
| [.agents/prior-art/RESEARCH_SOURCE_ATLAS.md](../../prior-art/RESEARCH_SOURCE_ATLAS.md) | Attributed pinned prior art/research, not an engine dependency or independent user approval. |
| [.agents/prior-art/SSD_STREAMING_AND_RELATED_PRIOR_ART_2026-10-15.md](../../prior-art/SSD_STREAMING_AND_RELATED_PRIOR_ART_2026-10-15.md) | Attributed pinned prior art/research, not an engine dependency or independent user approval. |
| [.agents/prior-art/VLLM_CONCURRENCY_PRIOR_ART_AND_ARCH_RECONCILIATION_2026-10-15.md](../../prior-art/VLLM_CONCURRENCY_PRIOR_ART_AND_ARCH_RECONCILIATION_2026-10-15.md) | Attributed pinned prior art/research, not an engine dependency or independent user approval. |

### 11.6 Public engineering/publication bodies outside agent roots

6 files.

| Complete original path | Reading purpose / status |
|---|---|
| [docs/DEVELOPMENT.md](../../../docs/DEVELOPMENT.md) | Developer build/release/tooling guidance; actual executed/platform evidence controls claims. |
| [docs/ENGINEERING_SPECIFICATION.md](../../../docs/ENGINEERING_SPECIFICATION.md) | Original integrated ES-00…15/INV-01…25 design; pseudocode is not frozen public ABI. |
| [docs/PUBLICATION_AND_REVIEW_CHECKLIST.md](../../../docs/PUBLICATION_AND_REVIEW_CHECKLIST.md) | Claims/byline/legal/review/publication and future-results acceptance, not a release result. |
| [docs/README.md](../../../docs/README.md) | Audience/authority public index; apply stale-link remaps above. |
| [docs/REFERENCES.md](../../../docs/REFERENCES.md) | Primary/academic bibliography; citation does not grant source-code or model redistribution rights. |
| [docs/WHITEPAPER.md](../../../docs/WHITEPAPER.md) | Original public technical argument/proposal; initial implementation status is manuscript-date history. |

No original was moved or altered by this lane. The tables deliberately preserve old bodies and their later supersessions instead of treating a source catalog, fixture list or design title as executable capability.
