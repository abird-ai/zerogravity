# zerogravity — master architecture v4.3: composable AI execution, model routing and simple user experience


**Design baseline:** reviewed and extended 2026-10-12 (v4.3 local-first UX, native multi-model router, residency, profiles and API compatibility). This is an architectural specification, not an implementation-status ledger. [The plan index](TODO.md) and [accepted baseline](accepted-baseline/TODO.md) govern current completion; [numerics](numerics-and-context/TODO.md), [model execution](model-execution/TODO.md) and [CPU/GPU optimization](cpu-gpu-optimization/TODO.md) retain their unfinished gates. The October 13 progress prefixes have been retired rather than presented as live status.

**Authority:** v4.3 supersedes the older v2/v3 sketches where they differ. The [critical review](../prior-art/ARCHITECTURE_V4_CRITICAL_REVIEW_2026-10-12.md) retains the rejected-design rationale; earlier document versions remain in Git. Work tracking follows [AGENTS.md](../../AGENTS.md) and [TODO-SPEC.md](TODO-SPEC.md), not archived handoff or manifest procedures.

**Latest integrated, publication-facing authority:** `../../docs/ENGINEERING_SPECIFICATION.md` (ES-00…ES-15 model/task/compiler/state/resource/distribution/router/API implementation contracts), `../../docs/WHITEPAPER.md` (technical argument and limitations), `../../docs/README.md` (documentation categories and authority), and [the current workflow](../../AGENTS.md) with [TODO-SPEC.md](TODO-SPEC.md) (implementation and verification tracking). These do **not** invalidate the exact mathematical/operator/protocol detail below; they resolve cross-document ambiguity and establish the shortest implementation reading path.

**Read-first detailed subsystem references:**
1. `../docs/MODULAR_AI_SYSTEMS_REFERENCE_ARCHITECTURE.md` — short governing ontology and invariants.
2. `../docs/COMPOSITION_AND_EXTENSION_CONTRACTS.md` — illustrative minimal Rust API contracts.
3. `../docs/ACCELERATOR_EXECUTION_FAMILIES_AND_BACKEND_CONTRACT.md` — universal native-kernel, AOT graph and static-dataflow region provider, lifecycle/state bridges and hardware topology.
4. `../docs/DISTRIBUTED_PLACEMENT_AND_NETWORK_EXECUTION_CONTRACT.md` — independent compute/weight/mutable state/cache placement across heterogeneous local and networked machines.
5. `../docs/REMOTE_PEER_TRANSPORT_AND_STATE_HANDOFF_PROTOCOL.md` — authorized tunnel peering, transport, remote state lease/epoch and migration/failure semantics.
6. `../docs/MODEL_ROUTING_RESIDENCY_AND_MULTI_MODEL_SERVING_CONTRACT.md` — built-in catalog/alias/model profile, multi-model residency and exact replica/TaskSession routing.
7. `../docs/ZERO_CONFIG_CLI_CONFIG_AND_OPENAI_API_CONTRACT.md` — one-command CLI, typed configuration, single API endpoint and tested OpenAI-compatible subset.
8. `../docs/MINIMAL_EXECUTION_SPINE_AND_PROTOCOL_V4.md` — precise state publication, async workers, hierarchical memory ownership and delivery semantics.
9. `../docs/TASK_PROGRAM_AND_COMPILER_REFINEMENT_V4.md` — two owning representations, dynamic shape/region guards, TaskDriver, optional gradients, performance.
10. `../prior-art/ARCHITECTURE_V4_CRITICAL_REVIEW_2026-10-12.md` — review findings R4-01…R4-24 and independently researched external prior art.
11. `DECISION_REGISTER.md`, `INTEGRATED_PHASE_0_AND_DELIVERY_GATES.md`, `IMPLEMENTATION_ROADMAP.md`, `EXPERIMENT_PROGRAM.md`, `RELEASE_ACCEPTANCE_GATES.md` — user requirements and practical work/evidence.
12. The other subsystem specifications, with v4 supersession note, and separate `../prior-art/PRIOR_ART_LICENSE_AND_PROVENANCE_POLICY.md`.

## 1. Mission, scope and central design argument

Build the **best practical Rust-owned modular AI execution runtime**, in the combined sense of useful model/task breadth, source/trained math fidelity, low-latency peak/aggregate throughput, physical resource efficiency, security, reliable operations and cost of evolving new model/device technologies.

The differentiator is **reuse of a single fast execution spine**:
- a mathematical op is defined **once**, with effects and a CPU/reference oracle;
- a task defines stage control/output and exposes no particular device/quantization;
- the compiler selects **closed semantically legal regions** and can replace a tensor/layer/model/partition region with a qualified Rust-native kernel, captured graph, externally compiled device program or statically scheduled dataflow plan, while retaining generic fallback;
- device/resource capabilities and async work are explicitly planned and measured;
- a task session has **one logical state publication authority**, independently of physical device workers;
- the service, output delivery, model loader, scheduler and physical budget **are not cloned for every specialization**.

**Scope:** inference-first but **modality- and task-general** execution: text generation, model encoders and rerank, speech recognition/synthesis, image/video iterative generation and multimodal model programs, plus optional future gradient/fine-tuning and agent-app orchestration **above** the trusted compute boundary. “All AI execution” is a *pluggable semantics and TaskDriver goal*, not instant support for arbitrary unknown model files or every optimizer/training algorithm.

**Performance ambition:** compare at equal source checkpoint/quant/semantic mode/hardware/context/batch/power/task quality; specialize where measured better. No architecture alone proves universal supremacy; lower TTFT, P95 and committed output cost must be demonstrated on actual devices.

## 2. Six composable architectural concerns (not six plugin engines)

```text
   Trusted artifacts and typed ModelDefinition         TaskRequest/TaskDriver
                 |                                             |
                 +---------------- SemanticProgram (SIR) -----+
                              math + typed state effects
                                         |
                   RIR pure-region legality / KIR candidates
                                  (transient compiler views)
                                         |
                  CIR physical resource and command lowering
                                         |
                   Immutable compiled ExecutableBundle
                (shape guards, direct kernels, graph islands)
                                         |
              PhysicalResourceArbiter / Device/TopologyProfile
              coarse admission → fast device-local suballocators
                                         |
             TaskCoordinator(session writer) → StepPlan → Workers
                                         |
                  Provisional domain results + readiness
                                         |
                Prepare all → Publish one StateRoot
                                         |
                Typed committed TaskResult/stream + ModeReceipt
```

**Only two enduring compiler representations:** (A) `SemanticProgram` for exact model math/effects, (B) `ExecutableBundle` for physical execution—including direct kernels and external/graph/dataflow program handles—whose command/resource DAG is CIR. `RIR` and `KIR` are useful internal transformations/provider candidates, **not mandatory independent serialized IR ecosystems or runtime interpreters**. This reduces surface, keeps model-load compile bounded and eliminates semantic IR walks in the hot output loop.

**Separate task control from model math.** Use one minimal **trusted Rust TaskDriver** to issue stage intents/loop frontier/output, while repeated per-token/per-timestep mathematical work can be a **device-native StepProgram** without host synchronization. Do not create a generic user-supplied workflow language or embedded privileged agent/tool executor inside inference.

## 3. Correct model math and mutable-state transactions

### 3.1 Model and operator support

ModelDefinition checks trained checkpoint config/tensor roles/position/quant/architecture and emits high-level operators for attention (full/sliding/sparse/latent/compressed), recurrence/delta gates/conv, MoE router/shared experts, n-gram/history, multi-stream residual, modality encoders, acoustic/vocoder/denoise as each is actually defined. Pure local math may lower into lower-level arithmetic/quant/matmul/gather/reduction inside a closed region, but op source semantics and state effects remain provable.

Model names/versions (e.g. Llama, Gemma 4, Qwen3.5+, Qwen3.8 Flash-Next, GLM-5.3 Flash, DeepSeek V4 variants) are **artifact/test targets**, not names of core Rust IR types or kernel schedulers. New checkpoint using known ops adds model definition/fixture; genuinely new math needs operator schema+reference code. A file format parser recognizing tensors is not evidence any unknown op can run.

### 3.2 One atomic logical StateRoot publication

**v4 corrects an important v3 type bug.** Old `StateDomain::commit` should **never** directly mutate visible session data. Each domain computes provisional updates and prepares a version; a TaskCoordinator owns the **single** logical writer for a TaskSession/shard. It checks all required domain versions/events at the same frontier and performs a one-root publication. No CIR `Command::Commit` can independently publish user-visible state; device events signal **ready** or **failed**, not semantic commit.

This is atomic *logical metadata visibility* across KV, recurrence, sparse index, n-gram, RNG/sampler, grammar, adapters, latent/timestep/media state. Physical GPU buffers require private/COW/undo or safe checkpoint replay until publish; rollback may cost more than O(1). Distributed peer state uses epoch fencing/ready agreement/recovery, not an unsubstantiated instantaneous physical transaction.

**Output streaming:** committed root precedes `(session,epoch,output_seq)` output event. Default live transport offers ordered, at-most-once **attempted** delivery, not exactly-once receipt across disconnect/lost ACK. Resumable cursor/idempotency is an optional separate protocol with bounded/durable log and privacy policy. Cancellation cannot leak speculative tokens or partially decoded media state.

### 3.3 Full-state prefix/cache compatibility

A shared prefix can be resumed only when each required state group has compatible checkpoint/schema/position/quant/adapter/media/grammar history at the same legal causal frontier. Multi-domain finalization alignment may force a shorter prefix and suffix recompute. Immutable snapshots + COW support agent forks; active in-flight version/generation leases prevent ABA, wrong lookup or cross-tenant data reuse.

## 4. Specialization and fast-path architecture

### 4.1 Closed-region substitution

Every `RegionCandidate` (including a native `KernelCandidate` implementation) declares exact mathematical region, inputs/outputs/effects, source quant vs compute precision, supported symbolic shape/ragged band, target ISA/compiler/driver, physical buffer layout/scratch/event timing, graph/eager mode and independent quality evidence. Compiler can choose a whole attention layer/entire region without dictating its internal implementation. Fallback remains reference/generic at a proven boundary or canonical state replay.

**Fastest region need not be fastest task.** Performance qualification evaluates full-task TTFT/P95/throughput/energy/physical memory and state correctness, not only microkernel FLOPs. Reassociation, BF8/FP8 transforms and quant arithmetic are explicitly numerically qualified; intentional trained top-k/mask/diffusion schedule changes require explicit changed-model opt-in, never silent optimization.

### 4.2 Shape, graph and hot scheduling discipline

Typed ShapeEnv and runtime guards support bounded dynamic dimensions, ragged batches/route offsets, strides/alignment, prefill vs decode vs verifier vs encoder/denoiser/vocoder bands and per-device ISA. Dynamic expert IDs and sampled tokens can stay device-side; do not compile one variant per token context length or expert ID.

Prepared `ExecutableBundle` and `StepProgram` use persistent device input/parameter slots and direct launches/captured graphs; no SIR/RIR/CIR interpretation per token. A task driver is polled at **stage boundaries**, not necessarily after every mathematical operation or emitted token. GPU-native routing, accepted speculative target results, sampling/grammar where supported, and state continuation avoid CPU synchronization; other exact task/control modes can fall back to honest host steps.

On resource/shape miss, use a **prepared general kernel**; optional background optimization never blocks the current request or silently changes numeric mode. Kernel/search cache bounded by physical/source/shape key and LRU; invalidated by code/driver/op schema changes.

## 5. Coarse resource admission, local physical execution

**Global ResourceArbiter:** conservative OS/cgroup/UMA/VRAM/GTT/SSD and device memory budget with headroom, session/model reservations, fairness and optional evictable caches. It does **not** serialize every kernel temporary alloc. **DeviceBackend:** fast pre-reserved local slab/page/pool and stream/event infrastructure issuing generation-tagged BufferLeases, with asynchronous retire once all readers quiesce. Device workers own GPU/NPU contexts/programs/captured graph handles; TaskCoordinator owns only logical state publication.

Resources include immutable weights, draft/adapters, routed experts, full/sliding KV and GDN/KDA/indexer history, frozen prefix snapshots, model code/graph programs, scratch/quant staging, pinned GPU/NPU BO, encoders/vocoders and media streams, SSD lookup row buffers and collectives. Different cache eviction strategies share **one physical admission ledger**, not one universal cache policy.

**OS safety:** no general guarantee physical memory stays constant after probing. Engine conservatively reserves, watches pressure and fails safely; CPU/GPU/NPU kernel allocation errors are first-class. Virtual GPU addressability is not necessarily physical free RAM. A timed-out external DMA may still touch memory: do not recycle until quiesced or quarantine/reset.

## 6. Heterogeneous device and source-language strategy

**Capability-based physical topology:** CPU SIMD, integrated/discrete GPU, NPU and future accelerator, pinned host/system RAM, storage/remote, copy and event bridges and shared power/firmware domains. `SplitProjectionCompute` may assign independent output rows to GPU+NPU; `AsymmetricExpertOwnership` may assign top-k routed experts across fast/small and slow/large owners. Both are **physical region placement alternatives under the same model semantics**, not second model engines.

A device declared present is not automatically **safe or numerically qualified**. A particular shared SoC NPU/GPU power/fabric behavior may hard hang a host; default unqualified offload OFF, no privileged system clock/BIOS/IOMMU/suspend changes. Rust-authored kernel source remains user preference; XRT/CUDA/ROCm vendor runtime FFI vs foreign kernel language is a separate unresolved product policy. An absent or unsafe NPU cannot block CPU/GPU execution of a supported model.

**Distributed:** peers handshake exact checkpoint/semantic/state schema and attempt epoch, stage provisional results, coordinator fences lost ranks/epochs before publish. No claim of physical distributed transaction magic or end-to-end exactly-once network delivery. Remote resource/collective is optional and should not tax single device.

## 7. Task kinds and optional training compatibility

TaskDrivers for:
- text autoregression, target verified draft and structured outputs;
- embedding/pool and cross-encoder rerank;
- speech recognition, acoustic frames and TTS/vocoder;
- image flow/diffusion generation/edit and video/audio temporal stage pipelines;
- compound trusted **compute** stages;
- **future optional** `TrainStep` with gradient/optimizer/ParameterVersion state, independent of default inference build.

Agent tools/RAG/external workflows are above the permissioned TaskRequest layer, not core GPU commands. A new model's Tensor semantics can be reused in any TaskDriver but requires task-specific validation. “All AI execution” means **extensible supporting mathematics/control/device**, not guaranteed released training/ASR/video on day one.

**Training seam:** ParameterVersion and optimizer state can be a future transactional state domain; running inference sessions pin one immutable model+adapter version and only switch on explicit safe boundary. Activation checkpointing, gradients, distributed optimizer and training durability are deferred until proven needed; do not bloat the inference hot path for them now.

## 8. Minimal Rust modules and release/process strategy

**Keep cohesive ownership boundaries**, splitting packages only when backend toolchain, unsafe boundary, compile profile or feature isolation justifies it. The original one-library/one-binary bootstrap advice was not a permanent crate limit; the accepted workspace layout is recorded in the [baseline](accepted-baseline/TODO.md). Organize internal modules for artifacts+models, semantics+state, task drivers, compiler+kernel registry, device+resource, runtime/coordinator, optional service/validation. Avoid mandatory dynamic plugins, giant abstract factories, generalized agent workflows or per-model servers. Compile-time feature flags isolate unavailable CUDA/ROCm/XRT drivers while CPU tests run on ordinary hosts.

**Future developer test:** independently add one new stateful op/model, another kernel device, an unusual quant, a non-token TaskDriver and an alternate IO tier. Record unrelated files/modules touched, compile time and hot-path overhead. If the change duplicates scheduler/model loader/allocator or requires broad type switches throughout unrelated code, **the architecture fails maintainability**.

**Focus implementation in a vertical slice:** one real source-verified model and TaskDriver from artifact through SIR/reference, legal region, KIR candidate, compiled CIR plan, CPU/one GPU backend and state/output publication; forcibly disable fast kernel and verify general fallback. Pair with mock three-domain state transactions, dynamic shape/async event lease and one non-token synthetic task. Only then expand optimized MoE, exotic models, experimental NPU, speech/media, speculative tuning and distributed breadth.

## 9. Quality and performance governance

Separate (1) semantic formula/state, (2) reference-to-kernel numerical tolerance/precision, (3) checkpoint-level task quality, (4) stochastic target sampling/distribution and valid committed output. A limited generated-string comparison is not a full quality proof; cross-backend bit-identical output isn't universal unless contract/test says so.

Benchmark relevant device/task:
- kernel ISA/register/occupancy, bytes/time and exact quality;
- full program prefill/AR target t/s, TTFT, P50/P95/P99 under native C1/2/4/8/16/32+ work;
- task-specific units (ASR WER/timestamps, TTS audio, image/video sampling latency/quality, embeddings/rerank, future gradients);
- real memory/IO/cache/quant/prepacked weights/prefix reuse and energy;
- source checkpoint/quant/driver/compiler/fabric/power and before/after methods, with raw evidence;
- physical hardware safety, failure injection, source/license rights and mode receipt honesty.

Optimization can be promoted only inside measured capability cell and safety/quality constraints. A no-gain policy should be disabled; no universal "fastest on all models" assertion. Publish known losses/skips to avoid overfitting benchmarks.

## 10. Staged delivery and source/research separation

**Stable phases ZG-P00–P23** are maintained; v4 review introduces additional **S0-V4** protocol and minimality proof gates and new **H32–H43** falsifiable experiments (see integrated program). Their presence is **planning**, not implementation completion.

**Prior art:** retained source research records external URLs/revisions, licenses, benchmark author claims and limitations; superseded v2/v3 document bodies are recoverable from Git, not mandatory live archives; core architecture and type/module names are **original project-neutral contracts**. Verified MIT source may be considered only after file-level notice/weight/device code licensing and Rust ownership review. Non-MIT projects remain attributable research; no copied source without explicit approval. The external research corpus remains read-only.

**Developer operating rule:** start with [the plan index](TODO.md), its relevant TODO and linked contracts. Use the decision register and this master for design context, inspect live source/tests, and record implementation and verification under actual commits using [the TODO rules](TODO-SPEC.md). No generated handoff, raw-evidence archive or SHA inventory is a delivery requirement.

## 11. Success criterion and practical limitations

**Correct model mathematics and full mutable state + a tiny effective runtime that can execute independently authored high-performance specializations across tasks and devices without multiplying engines.**

If v4 cannot (A) avoid host synchronization on supported device-native steps, (B) guard dynamic shape/precision correctly, (C) publish all-state roots atomically, (D) keep per-device allocations out of a global lock, and (E) add a second model/device/TaskDriver without unrelated rewrites, then we must **simplify or revise the interface before building breadth**.

This master states the **design ambition**, not a verified fastest-runtime claim. The accepted implementation is narrower; outstanding extension, quality and physical matched-benchmark gates remain in the current plans.

## 12. v4.1 — universal region execution across GPU, TPU and static dataflow fabrics

**Latest hardware-composability request U-019:** account for local coherent Blackwell SoCs, discrete/rack-scale FP4/HBM multi-GPUs, compiler-centric TPU/ASIC graph devices, wafer-scale/static dataflow accelerators, specialized sparse/model execution systems, and future device generations. Preserve the minimal **one semantic program → one compiled ExecutableBundle → one TaskCoordinator/StateRoot** core. Do not require a new model runtime per hardware vendor.

### 12.1 The decisive abstraction is *minimum efficient execution granularity*

Old `KernelProvider::compile(one kernel)` models ordinary CPU/GPU dispatch and some NPUs. **It does not by itself model an accelerator whose natural submitted object is a compiled whole model graph, pipelined static fabric program or wafer-scale weight stream.** Generalize it to a single **`RegionExecutionProvider`** whose candidate may cover:
- **NativeKernel:** one operation or fused Rust-authored region;
- **CapturedCommandGraph:** pre-recorded direct GPU/CPU/NPU commands;
- **AheadOfTimeDeviceGraph:** graph compiler lowering a verified SIR subgraph/model stage to an executable program;
- **StaticallyScheduledDataflow:** a large placed accelerator/fabric program with compile-time communication and explicit weight residency/streaming;
- **ManagedDeviceSession:** only when device SDK exposes enough source math, provisional state, snapshot/fork/replay, epoch and numeric/control semantics to qualify as actual lowerable work. A vendor serving endpoint with opaque text output belongs to a **separate TaskServiceAdapter**, not a fake native backend.

This is **one candidate/capability interface, not five new engines**. Native `KernelProvider` is its simplest implementation form, not an independently maintained runtime. No source model definition or TaskDriver should mention a hardware vendor. The provider's micro-kernel/program schedule belongs to the device/backend compiler, while the Rust coordinator owns task admission, logical StateRoot publication, output and resource quota.

### 12.2 Platform categories this represents

| Hardware category | Major physical distinction | Candidate mode and proof |
|---|---|---|
| Local coherent CPU+GPU SoCs (including small Arm/Blackwell and x86/RDNA hosts) | LPDDR shared physical RAM and CPU↔GPU coherence, limited bandwidth/OS headroom | Native/Captured graph with shared-memory lease, actual safe OS reserve and Arm64 or x86 ISA |
| Discrete GPU workstation and datacenter HBM GPU | external VRAM/HBM, tensor low-bit formats, PCIe and native graph execution | Rust native precision-specific kernels and capture, exact source quant/matrix scale proof |
| Multi-GPU rack fabric | peer-to-peer, switch/interconnect/collectives, sharded HBM | Partitioned graph/collective with actual topology costs, epoch fencing and all-domain state |
| Compiler-centric cloud TPU/accelerator | requires ahead-of-time graph lowering and often static/bounded dynamic shapes | AOTGraph, graph program lifecycle, shape bands, persistent device state bridge and full model verification |
| Wafer-scale and deterministic dataflow ASIC | static on-chip SRAM/core placement, stream weights/data and device-owned schedule | StaticDataflow, large closed stage/model regions, compiler source rights, controlled program load/weight streams |
| Heterogeneous GPU + NPU / GPU + specialized processor | different compute owners, memory formats, power/fabric and link bandwidth | one CIR graph with device-owned subprograms and explicit first consumer/IO readiness, no unsafe auto offload |
| Managed external model service | no mathematical intermediate program or trusted KV/RNG state API | **External TaskServiceAdapter** with independently declared model/session/privacy semantics, never native backend equivalence |

Examples and verified first-party URLs—**NVIDIA DGX Spark/GB10; Blackwell RTX, B200/B300, GB200/GB300; Google TPU v6e/v7x Ironwood; Cerebras CS-3/WSE-3; AWS Trainium/Inferentia; Groq LPU; AMD Instinct CDNA; Intel Gaudi; Tenstorrent; SambaNova and others**—are catalogued exclusively as source-attributed **prior art**, not mandatory architecture types: `../prior-art/ACCELERATOR_PRIOR_ART_AND_PLATFORM_RESEARCH_2026-10-12.md`. The user said “cerberus”; this is likely Cerebras but a distinct Cerberus parallel-decoding algorithm also exists, which fits the existing `DraftSource/TargetVerifier` seam and requires algorithm-specific quality proof.

### 12.3 New required capability detail

`RegionExecutionProvider` must expose region scope, op/shape/quant/numeric/fabric support, **Compile→Load/Upload→Warm→Submit→Ready/Fail→Retire**, first-use memory and links, **who controls internal micro-scheduling**, optional device loops, collectives and parameter/weight residency. The compiled program may be costly to rebuild or warm: retain bounded graph/program cache and shape variants, provide cold portable fallback, never JIT a whole TPU model on every generated token.

**State bridge is a hard gate:** a graph/wafer provider with hidden in-place KV/recurrence state that cannot fork/prepare/snapshot/recompute is not automatically safe for speculative decoding, cancellation or hybrid prefix caching. Stateless encoders may be eligible earlier; stateful program must return **private provisional state + real ready events**, while TaskCoordinator alone publishes the new StateRoot. External hosting/serving API is not a substitute for that contract.

**Model precision:** Blackwell NVFP4 (16-value E4M3 microblocks plus per-tensor scale), MXFP4 (different block scaling) and CPU/TPU/AMD custom formats are *different stored/compute/physical encodings*. No silent quant downgrade or inferred compatibility. Physical topology includes coherent UMA vs HBM/VRAM, on-chip SRAM/wafer weight streams, rack interconnect, cloud access control and egress cost.

### 12.4 Rust-first and feasibility truth

The **semantic/task/resource/worker engine remains Rust-owned**. Vendor device compilers/runtime/firmware and proprietary graph IR lowering may require external binaries/FFI not yet approved by the user. A Rust host adapter that submits a vendor-compiled graph does **not** imply we authored that vendor's device compiler or kernel in Rust; classify support transparently as `architecture-compatible`, `SDK/toolchain-feasible`, `compiles`, `functional`, `state-safe`, `performance-qualified`, or `production-qualified` **for each model/task/quant/shape/device**.

Never claim TPU, wafer-scale, Groq/LPU, Trainium or other ASICs actually run zerogravity merely because the abstraction can describe their program. Provider SDK access, source-license policy, numerical oracle, device/state failure behavior and physical hardware are independent qualification steps.

### 12.5 Proof program and scope

New optional **S0-FABRIC-A…H** fake graph/dataflow backend tests and **H44…H55** extension experiments cover graph program lifecycle, topology, state snapshots/refusal, external service misclassification, per-quant source correctness, cold compile/shape bands, rack collectives and source+vendor compiler policy. These **do not delay CPU/GPU Stage-0 vertical slice**. Existing P00–P23 unchanged; P01/P05/P07/P08/P10/P16/P19/P20/P21/P22/P23 now explicitly carry these backend family acceptance gates in the roadmap.

**Reference:** `../docs/ACCELERATOR_EXECUTION_FAMILIES_AND_BACKEND_CONTRACT.md` (normative, precise execution model), `../prior-art/ACCELERATOR_PRIOR_ART_AND_PLATFORM_RESEARCH_2026-10-12.md` (source-specific, not normative). Existing v4 state, TaskDriver, shape guard, CIR and physical allocator contracts remain authoritative.

**Success test:** a second backend that requires a **whole static compiled model graph** and one stateless **streamed-weight dataflow program** can run source-verified regions through the same TaskCoordinator, StateRoot/lease/admission and TaskDriver, without forking source model math or service. A device lacking a compiler or suitable state bridge stays explicitly unsupported rather than being falsely represented as “seamless.”


## 13. v4.2 — One distributed AI execution fabric across hardware, machines and tunnels

**Latest user requirement U-020** extends the project beyond peak single-device runtime: *independently place mathematical execution, model layers, their parameters, live KV/recurrent/indexer state, reusable prefix caches and task stages across heterogeneous local and networked machines, including explicitly authorized tunnels*. Prefer the best topology for **each checkpoint/quant/task/context/concurrency/physical network**, while retaining a truly minimal one-machine fast path. No model-specific distributed inference engines.

**New definitive references:** [Distributed placement and network execution](../docs/DISTRIBUTED_PLACEMENT_AND_NETWORK_EXECUTION_CONTRACT.md) (full architecture and performance/correctness), [remote peer transport and state handoff](../docs/REMOTE_PEER_TRANSPORT_AND_STATE_HANDOFF_PROTOCOL.md) (trusted tunnel, leases and failure protocol), [distributed subsystem entrypoint](../docs/DISTRIBUTED_PARALLELISM.md), and [distributed prior-art source research](../prior-art/DISTRIBUTED_PRIOR_ART_AND_TUNNEL_RESEARCH_2026-10-12.md) (attributed references only). The placement and peer contracts supersede earlier rank-local commit sketches: peers prepare and report actual readiness; only the session coordinator publishes the logical root. Earlier text remains available in Git, not as an independent live specification.

### 13.1 Four independent placement axes and one derived transfer schedule

- **ComputePlacement:** direct kernel, full graph, layer groups, core attention, FFN/MoE, tensor/head, token-range attention shards and TaskDriver/media stages; legal closed SIR regions only.
- **ParameterPlacement:** immutable model weight tensor shards/replicas, sparse experts, adapters, program images and external model/lookup sources, independently of compute location.
- **StatePlacement:** each session/layer/state-domain mutable KV, recurrence GDN/KDA, sparse indexer, n-gram/RNG/grammar/sampler and modality/iterative state has one versioned authoritative physical state-shard owner under a logical TaskSession root.
- **CachePlacement:** immutable shareable prefix/checkpoint/encoder-feature caches on GPU/CPU/SSD/other machines, with validated content hash/schema/tenant and locality directory **hints only**.
- **TransferPlan:** physical movement of activations, state snapshots, compact attention/expert partials, model weights and collectives, annotated with actual link RTT/P95 throughput, encryption/transport, bandwidth, buffer leases, source rights and ready events.

A *single* `PlacementPlan` is compiled from these into the existing `ExecutableBundle`/CIR **within the current topology and placement epoch**. No third durable IR, new service kernel runtime or different mathematical ModelDefinition. A `RegionExecutionProvider` may own a whole multi-machine/static graph program, with model/task/root publication remaining ours.

### 13.2 The placement topologies that must be representable

| Placement | Core execution pattern | When considered |
|---|---|---|
| **Colocated baseline** | all model layers and live state on one device/host | default latency/perf comparison, no cluster overhead |
| **Layer+state colocated per peer** | model layer groups on machines A/B/C, each holds their own layer KV/SSM; only activation boundaries sent | run larger-than-one-host model, LAN/tunnel capacity mode |
| **Separate layer compute and cache storage** | full compatible prefix hydrate once or run operator near cache with query/activation exchange | remote cache useful but full KV transfer per token expensive |
| **Attention/KV vs FFN/experts split** | stateful attention executed with cache on bandwidth/capacity-rich nodes, FFN/MoE on compute-rich nodes | long context/batches and fast links where frequent layer crossings worthwhile |
| **Sequence attention cache sharding** | split context by token ranges, send query to state owners, combine compact stable softmax summaries | very long context on qualified ordinary softmax operators and faster fabric |
| **P/D split** | prefill vs decode machines/pools, full hybrid state handoff at committed frontier | protect TTFT/ITL via phase separation when state transfer affordable |
| **Tensor/head or expert sharding** | collectives/partial results across local/rack ranks | low latency/high bandwidth and acceptable numeric reductions |
| **Remote immutable cache and model source** | share prefix or model tensor blobs across enrolled peers; hydrate/recompute | avoid recomputing long repeated prompts/duplicating parameters |
| **Task-stage partition** | audio encoder, text decode, vocoder, denoiser/decoder on different workers | specialize task workloads when transferring outputs is cheaper than moving parameters |
| **Heterogeneous local + tunneled network** | trusted CPU/GPU/NPU or AOT graph workers on local and remote private peer group | capacity or throughput subject to real RTT, data permissions and partial failure |

These are **eligible plan candidates**, not simultaneous mandatory modes; mixes are legal only if the exact state-domain partition and real network resource/delay budget permit them. A “cache on machine B, attention on A” topology is *not* semantically/physically equivalent to “attention + KV both on B”; compute beside KV can replace full remote KV reads with compact query and result, while a remote prefix cache may be best **hydrated once**.

### 13.3 Topology transport and trusted private tunnels

**One optional PeerAgent capability** reuses the same binary/library, RegionExecutionProvider and local ResourceArbiter to advertise verified devices/resources and execute a scoped closed region, not another chat server. Enroll peers explicitly. Control/metadata can travel through an authorized encrypted private tunnel, while the data plane chooses an available trusted direct interconnect, RDMA or authenticated reliable stream. **A tunnel cannot turn TCP/QUIC into GPUDirect or make relayed WAN latency disappear.** Link capabilities must distinguish direct UDP, peer relay and remote relay, loss/MTU, effective bandwidth/P95 RTT, copy/crypto overhead, egress and link permission.

Model weights, private prompts, hidden activations, KV and modalities cannot be sent to unknown peers merely because they advertise free GPU memory. Remote code/agent shell execution is not part of joining the cluster. Authenticate/authorize node, tenant, model and each sensitive data class; remote memory keys and state-owner leases are scoped/fenced. Start with a **small manually enrolled cluster**; peer auto-discovery, leader election, distributed metadata DB and autoscaling only if product measurements justify them.

### 13.4 One logical StateRoot across remote state shards

A SessionEpoch has exactly **one TaskCoordinator writer**. `StateRoot` can contain remote state-shard references, but every remote worker returns **Prepared/Ready/Failed** with buffer leases, StateShardGeneration and matching placement epoch; the coordinator **alone** validates/publishes the new logical root and output sequence. Device worker or peer does not independently publish a sample/new token. Losing a rank during prefill/collective aborts or safely replays from last compatible committed snapshot; a speculative branch cannot leave rejected KV/GDN/indexer/RNG state on any peer.

**Replanning at committed task frontier only:** reserve/warm new compute owner, copy/verify compatible state version, fence old writers/in-flight DMA, publish new placement/ownership epoch in a single root update; hold/quarantine invalid memory until physical completion. Peer departure or network partition cannot create two independent session writers. Without a durable lease/quorum/checkpoint protocol, **fail-stop rather than promise automatic transparent failover**. User output delivery remains ordered attempt with optional authenticated cursor replay; no claim of network exactly-once receipt.

Immutable remote prefix and model cache directories can be eventually updated and stale; they are revalidated on use and never override mutable state authority.

### 13.5 Topology-aware mathematical and cost legality

- Source SIR effects decide which attention, recurrent, latent/sparse indexer and media states can move or be partitioned.
- **Softmax attention token-range sharding** is algebraically composable using local logit max/denominator/weighted-value numerator and numerically stable global rescale. Correct positional/mask/routing schema mandatory. Unknown compressed, global selected-top-k and recurrent GDN/KDA do **not** automatically inherit this partition; add an operator-specific `PartitionAlgebra` with reference proof first.
- **Attention–FFN split** can reduce compute/memory coupling but adds per-layer network RTT. On a relayed tunnel it can easily be much slower than a colocated layer group. More nodes can improve *model capacity*, not necessarily C1 decode speed.
- A costed placement uses actual device compute and resident bytes, per-boundary activation/state/collective size, measured direct/relay link throughput/P95, phase/shape/cold warmup, batching and fairness, privacy and failure recovery. Candidate selection optimizes the **whole critical path of valid committed task outputs**, with capacity/energy modes separate from latency/throughput.
- Remote full KV reads per token across ordinary WAN/tunnel should normally be rejected; prefer local state migration, remote attention beside state, compact partial attention, recompute or simpler pipeline. Fast RDMA must be physically verified, never inferred from a VPN interface.

### 13.6 Delivery and proof without multiplying engine complexity

**Do not block P00–P08 local CPU/GPU vertical slice.** Expand existing P19 distributed phase incrementally:
1. simulated 2–3 peer layer placement with per-layer state, correctness/failure;
2. actual trusted two-node layer split via bounded encrypted stream with exact model/quant and slow-link tests;
3. remote prefix and full hybrid state handoff, including P/D and cache hit;
4. independent attention/FFN ownership and sequence-sharded KV where qualified fast link and operator math justify;
5. automatic measured placement across mixed devices and tunnel topologies, optional rack compiled collectives, graceful rebalancing and larger cluster management.

Optional **S0-DIST-A…L** design/transport/failure gates and **H56…H70** research experiments now track these capabilities; all are **PLANNED, not run**. Existing stable ZG-P00…P23 stay unchanged. Higher-performance / production claims require actual source-verified checkpoint, task, network direct/relay profile, numerics/full state, P95, physical memory and recovery evidence.

**Architecture acceptance:** adding a new private peer, transport link, remote state cache provider or layer/attention placement requires only `NodeCapabilities`, scoped `PeerTransport`, `PlacementPlan`/state-domain partition capability and actual measured tests; must **not** require cloning ModelDefinition, TaskDriver, TaskCoordinator, any API server or global allocator.

**This becomes a core zerogravity capability goal alongside peak single-device performance, not a side project or external model-serving integration.**


## 14. v4.3 — one-command local UX, built-in multi-model router and OpenAI-compatible serving

**Latest explicit user requirement U-021:** zerogravity must be both an extremely capable **hardware/model-specialized execution system** and a **joyfully simple local model tool**, with a built-in router/load balancer that can manage and serve many different model families and per-model settings through **one OpenAI-compatible API**. From a user's viewpoint, one `run model` command should work with sane defaults; every choice can be inspected and overridden safely when needed.

**Authoritative design files:** [model router and residency](../docs/MODEL_ROUTING_RESIDENCY_AND_MULTI_MODEL_SERVING_CONTRACT.md) and [zero-config CLI/config/API](../docs/ZERO_CONFIG_CLI_CONFIG_AND_OPENAI_API_CONTRACT.md). Names, source project implementations and current API features remain in [dated prior-art research](../prior-art/MULTI_MODEL_ROUTER_AND_API_PRIOR_ART_RESEARCH_2026-10-12.md) (non-normative).

### 14.1 One engine across CLI, SDK, API and many models

```text
 CLI run / Rust SDK / API compatibility / native task endpoints
                         |
                 Auth + TaskRequest
                         |
     ModelCatalog + versioned exact aliases & ModelProfiles
                         |
               ModelRouter (request admission)
                         |
          ModelResidencyManager (load/warm/evict)
                         |
       existing ResourceArbiter + PlacementPlan selector
                         |
             TaskCoordinator / TaskDriver
                         |
             ExecutableBundle / DeviceWorkers
                         |
                Prepared all-domain state
                         |
                PublishRoot once + stream
```

**No extra inference server**: ModelRouter does not run its own token sampler, device allocator or task-output stream protocol. It resolves and admits work at **request/model/session/phase boundaries**, while the existing TaskCoordinator batches and executes directly. A single model bound to in-process `run` has no mandatory HTTP/daemon/network hop; `serve` exposes the same engine plus API adapters. Optional trusted same-user local socket reuse is a convenience, not a required service or automatic root/systemd change.

### 14.2 Four entities must remain distinct

1. **RegisteredModel:** immutable content-addressed artifact/checkpoint/quant/semantic model and allowed tasks. A catalog can describe thousands without allocating their weights.
2. **ModelAlias + ModelProfile:** human-friendly versioned name, role and safe per-model settings (context, sampling, planned device/perf/cache/resident policy). Alias revision changes affect **new requests only**.
3. **ResidentModel / Replica:** actually loaded/warmed physical checkpoint/adapter/program under resource reservation. **One sharded deployment across several GPUs/machines is one replica; multiple independent deployments of an equivalent logical model are multiple replicas**; shared base weights/adapters may be physically deduplicated where safe.
4. **TaskSession:** user/task-specific mutable KV/SSM/indexer/RNG/grammar/media state, pinned source/quant/adapter and current StateRoot; replica/session affinity and future migration governed by identical v4.2 StateCompatKey/PlacementEpoch rules.

Many models can be **registered**, while simultaneous resident/active models/replicas are limited by actual physical RAM/HBM/UMA, model/adapter weights, compiled code, scratch, context/state, device queues, remote reservations and OS margin. On-demand loading, warming, idle eviction and configurable keepalive/pins/TTL happen under the same ResourceArbiter—**no unbounded resource claim**.

### 14.3 Exact-model routing versus model choice

**Default exact:** `run coding` or an OpenAI `model:"coding"` resolves deterministic alias/revision to one **precise checkpoint/quant/adapter/TaskDriver** and may load-balance among semantically/numerically compatible replicas based on queue, p95, cache affinity, device/remote safety and tenant policy. Stateful sessions remain sticky or migrate only after verified full StateRoot handoff at committed frontier. A router must never silently answer using another language model or lower quant because an exact model is busy.

**Explicit selector:** a configured `auto` alias or named pool may choose a different target source by an authorized policy, but must report the actual selected model and meet task/privacy/quality conditions. This is **not equivalent** to replica balancing and is not default for exact models.

**Multi-model TaskProgram:** target+draft, encoder+language model, embedding+reranker or TTS+vocoder can load several separately pinned artifacts under one coordinated task and budget. They are **not replicas**. Each may be placed locally or on trusted remote workers through v4.2, with source rights and full-state effects preserved.

### 14.4 Resident lifecycle and model switching

A prepared model moves through `registered→verified→admitting→loading→warming→ready→busy/idle/draining→unloaded`, with typed failed/unsupported states. Deduplicate concurrent loads of same binding, reserve before GPU allocations, continue serving other warm models while cold load runs, bound waiter queues and timeouts and avoid unload/reload thrashing. **Never evict active unsnapshotable KV or in-flight GPU/DMA.** Allow optional keepalive, preload/pin, idle scale-to-zero and safe lease-drained model unload, with actual physical capacity authoritative.

A hot alias/config update publishes a new immutable revision for future routes; old sessions remain on old source/template/adapter/plan until completion or explicitly legal migration. A source artifact deletion cannot destroy active session resources; model store GC respects references. A model cannot silently swap to a different source due to alias reuse mid-generation.

### 14.5 Configuration without upfront complexity

**Level 0:** `zerogravity run <model>`, `zerogravity serve` (illustrative final spelling). No mandatory daemon, config file, external service, account/cloud or manual hardware flags for local supported file.

**Advanced:** typed human-editable global config, per-model profile, alias/deployment policy, per-session and allowlisted request overrides, physical resources/graph/kernel/quant, sampling/grammar, model pin/warm/eviction, authorized multi-machine sharding, routing strategy/replicas and API/TLS/auth. **Hard security/semantic/memory constraints always outrank overrides.**

Users can inspect **effective values and why** via `inspect --explain-plan` and `config show --effective --model <name>`, including model source/quant, profile revision, resident replicas, task type, device/state/cache placement and performance evidence. No silent quant downgrade or unknown model pulls; remote fetch only with explicit trusted source/license/size/authorization. Proposed config is small versioned TOML; exact keys/path/port are open user/implementation decisions.

### 14.6 APIs, security and complete streaming semantics

Recommended first interoperable subset: **`GET /v1/models`**, **`POST /v1/chat/completions`** with correct model-template rendering, sampler, SSE deltas/finish, usage/tool-call **generation** and **`POST /v1/embeddings`** when real encoder qualified. Then qualified completions and `POST /v1/responses` with a **separately versioned stateless subset** first, stateful response storage/previous-response and typed Responses stream events only when genuine StateRoot/session retention and conformance exist. No claim every cloud built-in tool/search feature works; tools are only output data until an external authorized agent/workflow adapter executes them.

**Default local-only:** loopback bind, no auth required solely for same-user local trusted CLI unless configured; external bind requires explicit authentication/TLS and separate restricted native admin model-management API. Tenant/allowed model aliases and peer permissions do not transfer from any arbitrary OpenAI API token. Private source/model weights/media/prefix/remote state and log data stay protected. No privileged driver tuning or public remote server implied by `run`.

**Compatibility is a tested matrix** (endpoint × task/model × field × streaming × SDK version × numeric/usage behavior), not a boolean. Unsupported features return typed errors, not HTTP 200 with silently changed semantics.

### 14.7 Performance and long-term minimality

Router selection uses warm replica cost, physical model residency, cache locality, per-task context/StateRoot, queue fairness/SLO, peer/hardware/authorization and cold compilation/startup. **No router in per-token GPU hot loop, no per-output HTTP proxy tax, no global allocation mutex per kernel**. Multiple models have distinct queues and fairness; within a model, already existing native batching handles compatible sessions. Model cold load, on-demand eviction and warm pins are coarse physical-budget decisions, with measured P95/TTFT/cold startup/energy and safe backpressure.

**Proof:** one zero-config in-process model, one shared multi-model service, a cold+warm model concurrent load, safe idle evict without StateRoot corruption, two aliases of same source, equivalent replica balancing with sticky state, one explicit model-choice alias, model/task multi-role pipeline and correct official OpenAI SDK nonstream/stream tests. Additional S0-SERVE-A…J and H71…H86 are optional acceptance/evidence tracks; all NOT RUN. ZG-P00…P23 remain unchanged.

**Non-goals:** copying an existing inference gateway, starting independent model servers for every loaded artifact, automatically downloading unknown models, silently switching quant/source on overload, pretending all models fit simultaneously, running arbitrary tool actions inside an API handler, or implementing every provider's proprietary cloud-only Responses behavior.

**Final UX success:** users see **one command and one endpoint**; expert developers see composable and controllable model profiles, replica placements, multi-hardware routing, real source fidelity, full model-state safety and transparent performance. Both experiences run the same original zerogravity engine.


## 15. Final editorial and implementation authority (publication dossier 0.1, U-022)

**The public project documents are now structured by audience rather than chronological appended notes.** Source architecture remains v4.3; the October 12, 2026 publication dossier adds *no new runtime subsystem or experimental result*. The user asked for a complete whitepaper manuscript and full agent-deliverable design specification.

**Start here:**
- `../../docs/WHITEPAPER.md`: externally readable manuscript, attributed prior art and falsifiable hypotheses, **design only**.
- `../../docs/ENGINEERING_SPECIFICATION.md`: integrated normative implementation acceptance **ES-00…ES-15**, spanning precise source semantics, region execution, all-domain prepared state, hierarchical resources, distributed placement, multi-model router, CLI/API, source/security and implementation tests.
- `../../docs/REFERENCES.md`: numbered academic and primary source citations, no presumed code-copy permission.
- `../../docs/README.md`: documentation authority/navigation map, separating active contracts from non-normative historical/source reports.
- [AGENTS.md](../../AGENTS.md) and [TODO-SPEC.md](TODO-SPEC.md): bounded implementation, appropriate source/test/hardware/security gates and commit-grouped TODOs. Historical playbook templates impose no extra tracking duties.
- `REQUIREMENTS_AND_TEST_TRACEABILITY.md`: explicit U-001…U-022 user intents mapped to ES/INV/phase/test obligations.
- `../../docs/PUBLICATION_AND_REVIEW_CHECKLIST.md`: claims, author/license/public release signoff and future physical benchmark/reproducibility.
- `FINAL_ARCHITECTURE_REVIEW_2026-10-12.md`: final editorial/technical consistency and residual risk ledger.

**Current architecture invariants do not change:** one SemanticProgram and compiled ExecutableBundle; one trusted TaskDriver; one RegionExecutionProvider for native/graph/static programs; one physical admission authority with device-local pools; full StateDomains Prepare and one per-session TaskCoordinator PublishRoot; optional costed separate compute/parameter/live-state/immutable-cache distributed placements and authenticated peer transport; one exact-model catalog/router/residency layer and same-engine in-process CLI/loopback HTTP API.

**Evolve the working Rust model/task execution spine without building a framework for every research paper.** P00…P23 remain coverage references, not a blanket NOT_STARTED status: completed slices and genuine unfinished work are tracked in [the active plans](TODO.md). Stage0/Experiment/Release documents supply applicable acceptance criteria, not a demand to rerun completed historical phases. Design type sketches do not override tested source APIs.

**Publication policy:** manuscript is eligible for external *technical design review* but not experimental-results publication; named author/byline, original code/document license, stable publisher/DOI, independent reviewer and final public release must be approved separately (O-067…O-071). No PDF was generated or source/model/hardware/network action taken during this documentation pass. Any future performance edition must bring raw matched checkpoint/model/task/device/network data and negative evidence.

**Agent delivery success:** a new agent can identify exact authority, select one bounded phase, implement source-verified math and a compiled legal region with safe state/resources and fallback, prove a real task result and report exact tests/blocked issues without rereading a previous chat. A source-project-branded interface or cloned model runtime is evidence of architectural regression, not of innovation.

