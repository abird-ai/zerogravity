Author: Project owner
Co-authors: GPT-6, architectural research and revalidation sessions (26–27 October 2026)
Owner: @pvl; future implementation coordinator to be appointed only after resynchronization

# zerogravity Fabric vNext — complete architecture reference

**Design reference frozen:** 2026-10-27.
**Status:** REFERENCE ONLY. NOT APPROVED FOR IMPLEMENTATION OR REBASE. NOT an executable capability declaration.
**User's latest architectural instructions:** anything may change before v1; serving MUST be a reusable Rust library separate from the CLI; ALL useful SDK, serving, model, pipeline, engine, and administration functionality MUST be accessible through the public zerogravity Rust library; the CLI MUST be only a consumer of that library and own parsing, terminal presentation, and OS command interaction. Save the design separately. On the next explicit phase, FIRST resynchronize with the changed repository, review changes and revise this design; do not implement now.

This file is the normative *proposed* architecture reference for the next conversation, not a retroactive change to existing implementation requirements. The companion reports deliberately retain preceding research, counterexamples, original options and exact user decisions. The entire folder is self-contained for an agent without chat history.

Read [HANDOFF.md](HANDOFF.md) first for the next-agent workflow, [PRIOR_ART_AND_REVALIDATION.md](PRIOR_ART_AND_REVALIDATION.md) for source-level llama.cpp/vLLM/current-zerogravity evidence, [PUBLIC_LIB_AND_SERVING.md](PUBLIC_LIB_AND_SERVING.md) for the 27 October public-library amendments, [IMPLEMENTATION_AND_ACCEPTANCE.md](IMPLEMENTATION_AND_ACCEPTANCE.md) for staged migration, [DECISIONS_AND_OPEN_QUESTIONS.md](DECISIONS_AND_OPEN_QUESTIONS.md) for commitments versus options, and [TODO.md](TODO.md) for held work. The canonical current handoff is `.agents/docs/CURRENT_HANDOFF.md`; it points into the maintained navigation spine and this plan set. Git history retains superseded snapshots.

## Current handoff method

Use `.agents/docs/CURRENT_HANDOFF.md` as the only cross-window entrypoint. It links to the four maintained planning/status files and to this directory. Do not recreate monolithic archives, documentation SHA catalogues or frozen copies; Git history is the historical record.

## 0. Mission, scope and fidelity

**Positioning:** zerogravity is a **semantic, task-aware, hardware-adaptive Rust AI execution fabric**: one native execution substrate spanning local inference, performance-specialized GPU execution, model routing, concurrent serving, model composition, multimodal generation/perception, memory/storage tiers, and eventually trusted heterogeneous/distributed placement. It is not three independent "portable", "vLLM-like", and "extreme" inference engines.

Preserve these project statements in meaning:

> Run any *supported and independently qualified* model locally with the single-binary Rust execution fabric for text, image, video, audio, real-time, and combined multimodal inference. Model-specific mathematics, hardware kernels, memory placement, scheduling, and serving compose under one engine—not a separate runtime for every model or modality.

The phrase "any model" is an ambition, not permission to execute unknown mathematics or arbitrary downloaded HF Python. A filename, model registry entry, tokenizer or parsed weight file is not proof of functional model support. User-facing output must state the exact model+checkpoint+quant+task+shape+device+numeric+quality status. A fast path must never silently substitute a different checkpoint/approximation.

The project was previously designed around a semantic compiler, state-effect system, native Rust kernels, global physical resource authority, CPU/GPU/NPU/peer workers, source-trusted assembly, and a TaskCoordinator that publishes only source-valid state. That direction remains, but pre-v1 every internal API, module, name, crate layout, registry and implementation is permitted to change if simplicity, soundness, composability, performance or maintainability improve. Retain *proven behavior and test evidence*, not awkward compatibility facades.

**Scope inherited from the complete preceding discussion and designs:**
- Dense, sparse, MoE, GQA/SWA/MLA, GDN/KDA/SSM, n-gram/Engram, mHC, learned draft/MTP, explicit speculative inference and independent state restoration.
- Reference CPU execution plus CUDA, ROCm/HIP, Vulkan, future Metal, NPUs such as XDNA2, graph-only/AOT accelerators, statically scheduled fabrics and permitted remote peers.
- Original GGUF and safetensors/other verified sources; multiple model component assemblies; exact codecs and optional user-authorized derived low-bit/approximate sources.
- GPU VRAM, discrete HBM, coherent UMA, ordinary host RAM, pinned RAM, page cache, SSD and remote storage; physical capacity and real transfer coherency matter, not virtual address space claims.
- High throughput: continuous batching, chunked prefill/decode coexistence, persistent request slots, paged state, prefix reuse, fused/captured/model-wide specialization, dynamic MoE hotsets, optional speculation, tail latency/fairness, quality-preserving performance.
- Image generation/editing, video, embeddings/reranking, typed finite decisions, ASR, TTS/neural audio, native full-duplex and synchronized real-time audio/video; distinct task drivers and typed media clocks.
- One-process multiple model instances and replicas; one model can be partitioned across devices; many typed task pipelines can share immutable source/model instances; original state remains session-private.
- CLI, stable embedded Rust API, SDK integration, HTTP OpenAI-compatible APIs, Responses, streaming, WebSocket/WebRTC/media eventually, model admin/control surfaces, trustworthy observability. **Every programmatic function must be reachable through the public zerogravity library; serving is a separate reusable library.**
- Local-first runtime with optional remote placement under explicit user trust/authorization, cost, link and epoch proofs; no mandatory coordinator RPC for local execution.
- Source/semantic/numeric/state/device/transport licensing and safety first; publish neutral/negative benchmark findings and unsupported capability cells.

**Not goals for the initial proof:** running arbitrary Python/C++ from model repositories, shipping a new GPU driver, mandatory heavyweight training/autograd, arbitrarily extensible agent workflow DSL, separate optimizer daemon, public unstable plugin ABI, hidden hosted inference pretending native kernels, or forcing distributed execution into the local hot path.

### 0.1 Design ordering

1. Semantic and task correctness, exact source identity.
2. Numerical/quality equivalence under declared exact or selected approximate policy.
3. Real physical resource lifetime, asynchronous failure safety, one logical committed state.
4. User-facing correctness, privacy, usability and clear refusals.
5. Sustainable extension boundaries and reproducibility.
6. Whole-request/service goodput, TTFT/ITL/P95/P99, memory/energy.
7. Peak specialized performance on proven compatible model/device/workload.
8. Complexity only with a correctness proof or measured benefit.

The target is NOT to implement the most architecture nouns, nor to win an unrelated tokens/s benchmark. It is to execute useful source-valid AI work with the lowest defensible end-to-end cost.

## 1. Research synthesis and what changes from the preceding design

**llama.cpp's reference strength:** tensor graphs, per-model architecture builders, small public inference API, GGML backend interface/scheduler, CPU/GPU portability, backends selected below model mathematics, and a mature simple local executable experience. Avoid a global proliferation of hard-coded architecture enum/metadata switch edits and a GGUF-conversion prerequisite for all sources. Its multi-model router can manage model child processes, but that is not the same as zerogravity's shared resource, one-process, task-aware fabric.

**vLLM's reference strength:** V1 EngineCore separates request scheduling/cache planning from model execution; explicit scheduler outputs; KV blocks and prefix reuse; persistent batches, async workers and GPU model runners; reusable neural layers, model registry and out-of-tree model extension; mature OpenAI serving. Model Runner V2 redesign recognizes overly complex old model runners and targets stable slots, incremental GPU state/metadata, async dispatch and a minimal shared runner. Avoid assuming PyTorch/CUDA is the one foundation, treating request-slot state as the sole kind of state, or importing an ever-growing model-specific branch hierarchy into serving.

**vLLM-Omni's additional lesson:** modality-specific stages, coders/decoders, diffusion and other tasks require their own work units, control loops and timing. Reuse scheduling/resource/transport infrastructure but do not pretend all work is the next LM token. A logical multi-stage pipeline does not require a separate model server per stage.

**zerogravity's distinctive design strength:** exact source/numerical profile, semantic effects, two enduring compiler representations, a generalized physical region provider with optional model-wide fusion, one resource authority, multi-domain StateRoot publication, independently placeable compute/weights/state/cache and typed media frontiers. This goes beyond either reference architecture as a unified abstraction but is only an architectural hypothesis until tests and workloads demonstrate it.

**Corrections to the earlier assistant proposal:**
- Original file-tree plan retained crates by default; user's later pre-v1 permission authorizes breaking any layer or crate. Recommend a **model-neutral engine crate** enforcing strong dependency direction.
- The original vNext proposal left HTTP/server code near the CLI. **This is superseded**: introduce a reusable **zerogravity-serving library**; make zerogravity CLI only argument/config entry, invocation and presentation.
- Original vNext proposal said library/facade should assemble capabilities but did not clearly require all functions through it. **This is superseded**: public **zerogravity library is the single programmatic SDK and stable conceptual entry point**. Every capability including serving configuration, lifecycle, requests, events, introspection, model administration and future media/distributed APIs goes through it. Advanced native crate APIs can remain implementation details; no Rust consumer should need CLI imports or fork.
- Make **global scheduling and physical resource budgeting** foundational, not retrofitted after text CPU reference.
- Generic state is not just paged KV. Model states include recurrent, sparse/indexer, sampler/RNG, grammar, media clocks and codec/solver states.
- Distinguish mathematical graph composition, one verified multimodal model assembly, and a task pipeline connecting separate models. All can be typed, but they do not share a recursive interpreter or logical state publisher.
- No blanket one global token scheduling algorithm. One global policy/admission system and stage-local work-forming algorithms.
- Source-specific whole-model GPU specializations can be first-class closed-region implementations; their math and eligibility live in families, actual kernels/compilation/physical lifetime below shared providers.
- One physical plan with four independent placements; do not add four separate placement schedulers or a new distributed engine.
- Exactly two architecturally enduring execution representations: SemanticProgram and ExecutableBundle. RIR/KIR/CIR are optimization views/embedded command DAG, not parallel serialized runtimes.
- Explicitly preserve library-first API and delivery output-state separation, including services started from embedded programs, not only shell.

## 2. The architecture — one engine, five authority domains

Conceptual user-to-execution flow:

~~~text
               CLIENTS / PROGRAMS
  Rust application  CLI  OpenAI HTTP  Responses  WebSocket  Realtime
         \           |           |            /
                     |
                 zerogravity
      public SDK, built-in registry, typed builders
                     |
            zerogravity-serving
     reusable optional protocol servers / client sessions
                     |
   RESOLVED REQUEST + EXACT REGISTERED MODEL/TASK
                     |
       MODEL CATALOG / ROUTER / ADMISSION
                     |
       TASK COORDINATOR (session authority)
      TaskDriver | scheduling policy | StateRoot
          /                           \
  VERIFIED MODEL ASSEMBLY        CURRENT RESOURCE
  + original artifacts           and topology facts
          \                           /
            SemanticProgram (typed exact meaning)
                     |
       REGIONS / QUALIFIED COMPILER CANDIDATES
                     |
               ExecutableBundle
     guarded code+graphs+resource/transfer templates
                     |
         SCHEDULER + PHYSICAL RESOURCE ARBITER
                     |
             WORK ENVELOPES / SLOTS
                     |
 CPU | CUDA | HIP | Vulkan | NPU | IO | trusted remote workers
                     |
  ACTUAL READY/FAIL/UNKNOWN + PREPARED PRIVATE STATE
                     |
         SESSION AUTHORITY PUBLISHES ONCE
                     |
    COMMITTED OUTPUT => BOUNDED DELIVERY + ACK/PLAYBACK
~~~

This is a logical contract flow. Static validation/preparation occurs on source admission, cold model load or off-critical-path tuning, not every generated token. Logical authorities are NOT one thread/process/trait/crate each.

### Five primary authorities

1. **Model/semantic authority.** Owns exact original artifact/assembly, model family config, tensor roles/ties, tokenizer/processor, mathematical definitions, state-domain schemas and source-qualified conversions. It does NOT own device memory, HTTP, routing, task publication or user process privileges.
2. **Compilation/plan authority.** Owns semantic legality, pure-region fusion/rewrites, guarded region candidate selection, executable preparation, shape bands, code cache and physical plan generation. It does NOT own mutable session frontier or pretend compilation is physical success.
3. **Logical session authority.** Per TaskSession or authorized shard one coordinator/writer owns TaskDriver progression, expected StateRoot, epoch, cancellation, provisional → committed transition and committed output sequence. No kernel, transport or individual StateDomain may publish the whole session independently.
4. **Physical resource authority.** One global physical ResourceArbiter accounts for shared addressable/physical memory domains, committed and future liabilities, DMA/IO/graph/software/OS headroom, quotas and coarse all-or-none reservations. Device-local pools allocate fast under prepaid envelopes, not under a global lock per op.
5. **Device/IO execution authority.** DeviceWorkers, IOWorkers and qualified RemoteRegionWorkers own driver contexts, code images, local pools, stream/event/transfer submission and true Ready/Failed/Unknown completion, fencing and retirement. They return private candidate state and receipts, never directly a user-visible TaskRoot.

Other components are coordinated roles, NOT competing owners: ModelRouter chooses an exact immutable target and compatible replica at request admission, a scheduler selects work from already-admitted sessions, TaskDrivers select next legal stage, serving protocol adapters implement wire semantics, delivery sessions track external transport progress.

**Hard invariants:** a model addition may not create another allocator/scheduler/state publisher/API server; a device addition may not change mathematical source truth; a new task may not rewrite the physical worker; a specialized whole region remains an optional provider candidate with a canonical decomposable baseline.

## 3. Three different graph topologies, never conflated

A) **Software dependency graph.** Cargo crates and internal modules; conceptual owner and access constraints.

B) **Mathematical/task graph.** Typed tensors/media values, semantic operations, state effects, nested mathematical scopes, verified model assembly and trusted task-stage dependencies. This defines meaning and legal composition.

C) **Physical placement/command graph.** Compute regions, parameter extents, state shards, reusable caches, device-local buffers, transfers, event dependencies, captured programs and remote link costs. It defines one bound physical execution.

One structure does not force another. A Rust crate edge does not require data copy; a model boundary does not require a server/process; a type-correct model port does not prove backend placement or quality; a Ready device event does not mean task state was published or packet played.

### Distinguish three senses of pipeline

| Composition | Example | Authority |
| --- | --- | --- |
| Mathematical region | RMS → attention → FFN, nested fused subregions | SemanticProgram / compiler |
| Single verified model assembly | image text encoder + transformer + scheduler + VAE, speech encoder+codec | ModelDefinition/Assembly |
| Task workflow/pipeline | separately registered ASR → LLM → TTS | TaskCoordinator/TaskDriver, compound typed stages |

Nested mathematical regions are compiler structure, not dynamically recursive execution. A verified model assembly is one source-identified machine-learned model with independently placeable components. A multi-model task pipeline coordinates *different* committed sessions with explicit stage or compound transaction boundaries; it is not identical to a single model's internal stages. External tools/actions remain above core behind permissions, not arbitrary GPU TaskDriver callbacks.

## 4. Two enduring representations

**SemanticProgram (SIR).** One canonical device-independent typed mathematical program: source-bound parameters; logical values/rank/axes/shapes; positions, attention variants/masks; exact operator formulas; cast/reduction, nonfinite and numerical profile; explicit immutable reads; typed mutable state effects; valid stage/frontier outputs. Pure operations can be rewritten only under declared algebraic/numeric legality. Stateful operations cannot cross causal/state boundaries unless a validated effect transformation exists.

**ExecutableBundle.** One immutable prepared physical plan for a versioned semantic program, workload regime, topological capabilities and chosen admissible candidates. Contains direct kernels, captured programs, AOT graph handles, guarded shape bands, resource/transfer/dependency commands, storage lifetime/leases, state bridge, fallback cut, code/compiler/driver identity and bound mutable parameter slots. Live request epochs/slots/physical leases bind **late**, not into the compiled identity.

Compiler-local representations are useful but NOT additional permanent standards: RIR = transient legal region/layout/fusion options; KIR = transient kernel/codegen candidate; CIR = command/resource/event DAG contained inside the physical bundle. A StepProgram or WorkEnvelope is a prepared execution handle/bindings, not a third semantic IR. Compile/load/tune outside the request hot loop; bounded prevalidated variants only in the immediate dispatch loop. A shape miss uses a correct proven baseline or typed refusal, not uncontrolled dynamic codegen.

**Closed-region specialization:** a physical provider may replace an op, fused block, model stage, whole-model mode or partitioned program only if **all** visible inputs/outputs/state effects, original artifact/semantic/numerical rules, provenance, shape guards, source access, physical reader/lease lifetime, real completion, cancellation/fork/replay and required observations are honored. If state layout changes, transition at a proven committed frontier with snapshot/conversion/recompute; never retag pointers or publish partial state. Whole-model optimized programs are first-class, but **not standalone model inference engines**.

## 5. Extensible execution providers, not a trait zoo

A single **RegionExecutionProvider** conceptual seam chooses and prepares qualified closed semantic regions. Native CPU/GPU kernels are one execution family; others are captured GPU graphs, AOT device graphs, statically scheduled dataflow/fabric programs, and qualified remote multi-device compiled regions. An opaque inference text API is an application-level TaskServiceAdapter, not an admitted device-region provider.

Provider advertises actual capabilities: semantic op/schema, executable granularity, tensor/encoding/precision/shape/raggedness, device control/loop support, state bridge, memory topology, transfers/coherency, code image/compiler/driver identity, failure/cancel/fencing, test evidence and source/license policy.

No generic requirement to emulate wafer/TPU internals through host-issued scalar kernels. Providers can own large device-native programs with bounded loops; actual task/session authority stays in Rust. Optional Rust-owned GPU kernels supported where feasible; approved vendor runtime FFI remains distinct from importing unreviewed third-party compiled kernels.

Do NOT require one trait per codec, operator, kernel, buffer or platform. Prefer static enum dispatch and concrete prepared structs until two real consumers justify a trait. No heap virtual dispatch per tensor element, no vtable in a hot projection loop; coarse per-prepared executable dispatch is fine.

### Execution/granularity table

| Candidate | Suitable job | Limitation |
| --- | --- | --- |
| Portable CPU | canonical reference and source-supported local execution | throughput may be limited |
| Native GPU op | GEMV/GEMM, attention tile, packed quant kernel | too many launches if forced on static accelerators |
| Fused native region | norm+projection, recurrence, grouped MoE, full layer | numerical/effect/observer legality must be proven |
| Captured command graph | repeated shape-bounded GPU stage | pointer generations/encoded readers/capture lifetime |
| AOT device graph | encoder, denoiser, layer, whole model graph | compile/load cost, shape bands and state export restrictions |
| Static dataflow | fabric/wafer/graph accelerators | device-owned schedule and state bridge |
| Partitioned program | multi GPU/NPU/remote layers/heads/experts | complete transfer/state/fencing/cost model |
| External hosted inference | explicit TaskServiceAdapter | NEVER counted as native region execution |

## 6. Task drivers and work policies

One small trusted **TaskDriver** describes high-level stage sequencing, iteration/frontier legality, model-specific stop/valid-output rules and stage-dependent state; it never becomes a general agent workflow DSL, task subprocess manager or provider allocator. Stages can be model-wide closed graph islands and bounded device-native loops, not a callback per token.

Tasks require *different work units*: AR token/target verification rows; encoder sequence/window; audio frontend samples/chunks; ASR hypothesis window; TTS acoustic frames and RVQ codebook groups; denoiser/flow timestep and spatial/temporal tile; image and video output boundaries; typed score/decision units. A token budget cannot assign meaning to a PCM sample or image solver step.

One common scheduler/service makes global fairness, budgets and admission decisions; stage-local policies generate compatible execution work:
- Decode/prefill/target-verifier token or row budgets; chunked prefill with protected ITL and aging to prevent starvation.
- Big-batch encoder throughput and bounded memory.
- Denoising/video spatial tiles, allowed timestep schedule and temporal dependencies.
- Live acoustic input and complete codec output under real-time soft deadline and jitter constraints.
- Finite/schema-specific decision scoring and atomic answer sets.
- Optional distributed stage and remote queue only when authorized and profitable.

Never mathematical-batch incompatible models because their tensor shapes happen to match. Different tasks/models can be *co-scheduled* on shared devices and physical pools; within one qualified model/adapter/numerical plan, compatible sessions may form native batches.

## 7. High-performance serving is an engine concern, serving protocols are a library concern

vLLM-class baseline preserved: continuous batching, prefill/decode coexistence, persistent stable slots and incremental GPU metadata, paged/hybrid cache, prefix sharing, async worker stages, optional speculation with target verifier, GPU-native sampling/feedback, admission backpressure, QoS, fairness, prefix/placement affinity and comparable latency/goodput measurement.

**Three admission layers:** ingress request/payload/tenant/deadline credits; model/session/full-state/source and uncomputed work capacity; device StepPlan rows/scratch/transfers/active submission. One ResourceArbiter owns physical bytes; these are three control checkpoints, not duplicate independently spendable capacities.

**Host hot-path contract:** accept compact already-validated step inputs, choose among finite prepared plan candidates, bind slot and lease generations, submit and retire; no JSON, no catalog search, no dynamic compiler search, no global allocation lock per kernel and no mandatory device→host→device roundtrip when validated device-native feedback is available. GPU execution may overlap host prep, IO, peer transfer and another legal stage while preserving real dependency/event fences.

**Persistent batching:** stable request slots and per-session versions; incremental token/mask/position/row pointer metadata; opaque prepared executor receives bounded deltas; compatible cohorts never pad inactive rows into visible state/sample. A batch is physically real only when its native kernels share work, not merely when the server loops N independent single-request calls.

**Bounded candidate selection:** joint optimize prefill chunk, active rows, speculation depth (including K=0), MoE expert union/padding, source encoding, shape-compatible kernel, hot cache/SSD transfer, topology and request QoS. Candidate set is prequalified and bounded; online change only at legal state/placement boundaries with hysteresis, fallback and truthful mode receipt. Never use an approximation without explicit policy and quality proof.

**Streaming:** immutable committed outputs are exposed via SDK events, then optionally serialized by serving adapters. Client delivery, ACK and media playback are separate credits/epochs; slow output transport must not retain GPU scratch forever or force other session stalls.

The new **zerogravity-serving library** owns protocol adapters, network lifecycle, supported HTTP/OpenAI/Responses semantics, streaming framing, access control at transport level, TLS/reverse proxy integration policy, metrics endpoints, request validation mapping, timeouts, listener/reload options and bounded connection buffers. It delegates all inference scheduling/model loading/state/admission to the engine through zerogravity's public runtime API. It is reusable from a Rust executable embedding an API server; CLI just configures and calls it. Neither CLI nor the serving crate is allowed to implement an inference engine.

Full detailed public API, crate dependencies, server lifecycle and conformance is in [PUBLIC_LIB_AND_SERVING.md](PUBLIC_LIB_AND_SERVING.md).

## 8. State beyond paged KV

**Logical StateRoot** is an immutable committed version per TaskSession (or explicitly authorized shard), containing full required domain version handles and a typed TaskFrontier. Domains include full/sliding KV, GDN/KDA/SSM convolution and matrices, sparse/MLA/QSA indexer, n-gram/Engram and target/draft taps, grammar/RNG/decoder history, audio encoder frontend overlap, acoustic codec/duration/RVQ state, image/video solver/latent state and typed decision output. Physical policies vary; the transaction protocol does not.

Mandatory sequence:

~~~text
Begin(expected_root, epoch)
    -> private/COW provisional state and admitted resources
    -> prepared execution and actual device/IO completion events
    -> each required domain PREPARES a compatible version
    -> coordinator validates complete frontier/effects/epoch/old root
    -> exactly one PublishRoot
    -> ordered source-valid committed output events
    -> retire old readers after actual physical quiescence
~~~

No independent StateDomain::commit, no device-visible Command::Commit that publishes logical state, no unverified token emission, no hidden model mutation on cancellation. Private device buffers cannot overwrite the old committed state unless proven undo/recompute safe. A logical O(1) root swap is not O(1) recurrent state copying.

**Hybrid prefix reuse** requires a common restorable frontier intersection across *all required* model/task state domains, with source/checkpoint/quant/processor/tokenizer/template/position/numeric/state schema/tenant authorization. A KV hit without matching GDN or indexer checkpoint cannot be reported as a complete prefix hit; back off to the largest jointly restorable frontier or recompute. Compatible COW pages protect queued, encoded, captured and in-flight readers through true completion.

Speculation keeps independent draft and target state, target-authoritative verification of candidate tokens and sampler distribution, complete committed acceptance, and rollback/recompute for ALL effects including RNG/grammar, recurrence, sparse indices and media. Model-correct output from a broken always-rejected drafter does not prove the drafter correct.

## 9. Resource, memory and placement

Physical resources: GPU/CPU/NPU computation, model weights, ephemeral activations, per-session state, shared prefix snapshots, draft/MTP weights and state, read-only expert pool, pinned DMA windows, CPU threads, native modules/graphs/code images, SSD staging, OS/cgroup reserves and network/link credits.

The **single global PhysicalResourceArbiter** admits coarse reservations at model/session/step boundaries; device-local slabs/rings/pools allocate and retire only from prepaid sub-envelopes with generation-tagged real backing leases. Reserve **all necessary domains atomically** for a multi-GPU+host+pinned/IO plan; alias overlapping UMA domains and count backing once. A host/GPU/IO observation is advisory and can be stale; there is no impossible OOM guarantee against OS/driver changes.

**Four independently representable placements** within one PlacementPlan:
1. ComputePlacement: op/layer/model stage, partitioned outputs, experts, prefill/decode roles.
2. ParameterPlacement: immutable source weights/derived packs, adapter/MTP/lookup and graph images.
3. StatePlacement: per-session authoritative live KV/recurrent/sparse/decoder/media state.
4. CachePlacement: immutable/recomputable prefix snapshots, encoder features, expert caches and other retained assets.
A derived **TransferPlan** connects mismatched owners with exact source/version/tenant/layout, first-use dependency and real readiness, never a separate distributed scheduler.

Default local placement colocates work/parameters/state unless measured benefit and valid bridge justify separation. One logical tensor can be materialized in local VRAM, unified RAM, ordinary RAM, SSD or authorized remote peer; its source and semantic identity remain unchanged. Fully resident fast path incurs NO mandatory virtual tensor lookup/cache miss branch per projection. A real SSD miss or device transfer uses bounded asynchronous work with event fences, cancellation, source hash and lease retention; caches cannot free buffers that a recorded/captured/queued GPU command has already encoded as a reader.

UMA: GTT/VRAM addressable limits are NOT physical free RAM. Observe MemAvailable, actual driver memory, page cache, OS/cgroup headroom, pinned memory, external pressure and overlapping domains. Avoid unsafe clock/firmware/system tuning, especially for GPU+XDNA2 NPU cooperation. A model bigger than RAM may be correct but SSD-bound; do not claim high-speed without cold/hot IO measurements.

Cache policies must be specialized under one global budget:
- immutable model weights and source snapshots share only exact authenticated backing;
- paged KV needs page tables, refcounts, valid tails and typed state resume;
- recurrent state needs bounded checkpoints/COW/recompute;
- MoE experts need actual source-selected IDs, size classes and decode-hotset protection;
- lookup tables need bounded row IO;
- scratch needs liveness and aliasing under actual graph/capture readers;
- disk staging must have true async readiness and source mutation/short-read refusal.
A universal LRU is wrong; independent unaccounted caches are equally wrong.

## 10. Multimodal and realtime design

Typed value descriptors include Tensor, TokenSequence, CodebookSequence/RVQ, AudioPCM, AcousticFeatures, VideoFrames (PTS/DTS), ImagePixels/color metadata, MediaLatents/VAE identity, EncodedPackets, Transcript with speaker/timing/finality, DecisionScores with honest calibration, Structured(schema). ValueRef has content and lease identity, bounded extents, correct clocks. No implicit conversions based on equal byte shapes or guessed sample rates.

**ModelAssembly** binds a whole source-authored composite checkpoint (e.g. text encoder + denoiser + VAE; vision encoder + projector + text decoder; acoustic encoder + codec), required processors and state schemas. It is NOT replica group nor task pipeline across unrelated independently registered models.

TaskDrivers cover causal text, encode/pool/rank, Whisper-like encoder-decoder ASR, CTC, RNN-T/TDT, deterministic and codec-AR TTS, acoustic-flow TTS, native duplex audio, latent image flow, image AR codebook, temporal video, joint AV, finite typed decisions. Native stage math and codecs must be source-qualified; a fake textual transcript or model-name metadata never implies runnable media.

Typed task pipelines may nest descriptions, but compile stage structure and typed ports ahead of execution; no recursive per-token/stage interpreter. Allow device-resident stage edges with compatible producer/consumer, real events, generation-safe leases; materialize/convert only where needed. Default compound task behavior is legal staged commits, not fictitious atomic rollback across prior ASR/audio playback/external tools. Optional explicit joint multi-model transaction needs separate acceptance.

Media has distinct Prepared, Committed, Delivered, Acknowledged, Presented frontiers, plus received/decoded/consumed ingress. ClockDomainId, rational rates, exact conversion residue, track/lane IDs, generations and source delay prevent drift and stale playback. Turn/OutputGeneration barge-in invalidates future uncommitted playback without resetting physical card epoch or rewriting prior committed/root history. Transport reconnect never grants raw neural state. Playback/ACK cannot prove the human actually heard sound.

## 11. Distributed and heterogeneous evolution

Support CPU/GPU/NPU combined compute, expert placement, legal output-row projection slices, multi-GPU layer/tensor/expert sharding, optional remote closed-region workers, disaggregated prefill/decode and graph/static devices as *physical candidate plans*, not new model servers.

A new provider must expose accurate device/topology/compiler/runtime capabilities, explicit coherency, typed state bridge (private provisional versions, verified snapshot/restore, or safe recompute), real completion/failure/fencing and program lifecycle/memory cost. Opaque-unrecoverable state is ineligible for stateful region replacement. A vendor text API is an external hosted task service and never masquerades as a region provider.

Compute/parameter/state/cache placement may vary independently, with an exact TransferPlan. State migration or topology change occurs only at a verified committed frontier after reserve/transfer/check/fence. One authorized logical session publisher remains responsible, even if remote workers own provisional state shards. Remote peers require explicit enrollment/data rights, authentication, tenant scoping, link/latency measurement and failure epoch protection; a tunnel grants connectivity, not RDMA, privilege or low RTT. No consensus/transparent exactly-once network failover claim without an actual protocol.

NPU/GPU cooperative kernels need explicit power/fabric safety; documented host hard-hang reports forbid automatic unsafe tuning. Rust-hosted SDK bindings do not magically make vendor-written device code Rust-authored.

## 12. Model onboarding and correctness

**Target model contributor workflow:**
1. Add source/processor/format recognizer and exact component manifest with hashes/license.
2. Define family config, roles/ties, source tensor validation, one mathematical layer/forward traversal, state schema and tokenizer/template/media processor as appropriate.
3. Record/evaluate into canonical SemanticProgram; implement new *shared* operators only if mathematics genuinely requires them.
4. Prove CPU mathematical reference, original-source numerical fidelity, complete trained model logits/state and task-quality gates at declared scope.
5. Reuse already qualified generic provider/capabilities; register independently qualified specialized region/kernel/graph only when beneficial.
6. Register family/variant+task capability in the single cold-built catalog; do not claim broader GPU/quant/context/model support from metadata.
7. Reuse SAME engine state/scheduler/router/serving/public API. No Qwen/Gemma/http-specific request branches.
8. Benchmark real cold/warm workloads, including quality and negative results.

Extensibility drills: add (a) another model using all known ops; (b) a truly novel stateful op; (c) a fused region disabled without changing semantic output; (d) a new device with real event/lease behavior; (e) an encoder/ASR/denoiser task with non-token frontier; (f) a mock remote state bridge. Measure changed shared production files, LoC and model-specific branches—not merely number of traits.

**Four independent quality claims:** (1) semantic and source safety; (2) per-op declared numerical accuracy; (3) full-model task/distribution quality versus pinned reference; (4) determinism in explicitly advertised run/batch/hardware modes. Do not replace these by top1, perplexity, matching forced teacher trajectories, hash equality or an upstream tolerance imported without context. Keep GGUF and original HF, BF16/FP8 and quant, C1/C4 and CPU/CUDA/HIP/Vulkan evidence independent. A different source quant is a different model artifact. Performance follows correctness and complete resource/state tests.

Capability status ladder: recognized metadata, semantic/reference, compiled, full-functional, state-safe, quality-validated, performance-qualified, production-qualified, or explicit blocked/unsupported. Each cell qualified per exact source, task, shape, mode, device, compiler/driver and numeric.

## 13. Deployment, packaging and stable public API goals

The user should be able to:
- import one public Rust crate "zerogravity", configure/load/inspect/run any supported model and task, and stream events, with no CLI dependency;
- construct a library runtime with models, caching, routing, pipelines, memory/device policies, telemetry, qualified provider selection and lifecycle;
- build the **same** local daemon/server programmatically, without invoking the zerogravity CLI or reimplementing HTTP/model routing; use the **zerogravity-serving** library through public reexports/builders in zerogravity;
- use CLI as the thinnest interface: parsing, local config/presentation/signal dispatch, error display, invocation of library functions; no proprietary source/memory/scheduler logic in CLI;
- run binary "zerogravity run model.gguf" without compulsory daemon/config, "zerogravity serve ..." with same engine, introspect genuine capabilities/fit/refusals, obtain a clear dry run;
- embed without Python model runtime or vendor GPU SDK when using CPU only; optional backends feature-gated and accurately advertised.

**Library must own ALL functionality** the CLI exposes, including model pull/cache, inspect, devices list, model catalog/alias/loaded list, direct run, serve start/stop/configure, request processing, output streams, pipeline compile/check/execute, snapshots, support matrix, generated metrics, status and explanations. Supporting crates can implement it; facade exposes coherent public typed APIs. Do not make the facade one giant implementation crate; reexport a curated capability-oriented surface, private modules for complexity, versioning/semantic stability only when decided.

**Serving library** is an independent package with HTTP/stream protocols, connection lifecycle, auth/transport policy, routing protocol payloads to engine Requests and committed Events, bounded ingress/egress, SSE/chunk framing, cancellation and structured client conformance. It must work from ANY embedded Rust application, not only zerogravity-cli. It depends on model-neutral engine contracts and shared types, not on CLI, model-family concrete code or GPU specifics. The facade exposes its builders/types so ordinary users depend only on zerogravity. The dependency DAG must remain acyclic.

See [PUBLIC_LIB_AND_SERVING.md](PUBLIC_LIB_AND_SERVING.md) for exact intended examples, API matrix, feature/dependency boundaries and acceptance.

## 14. Initial accepted design priority

**At any implementation start, first revalidate against current repository HEAD/source**, assess recent refactors/commits and plan crosswalk, then follow the user's latest explicit authorization. CURRENT_HANDOFF.md is the cross-window authority for that decision. This 27 October archive deliberately does not claim the 26 October code inventory remains current. The previous inspected live tree (2026-10-26) was HEAD 195099ca and very dirty; by 2026-10-27 on dotlink-l5 HEAD has advanced to eb38dd5e and was clean at initial read. This reflects moving work, not a verified migration. Never transplant old file-line findings without rechecking.

After explicit implementation authorization, focus: make baseline reproducible; enforce source→semantic→physical ownership and Cargo boundaries; extract one common engine and robust library facade; split serving into standalone crate; have CLI depend exclusively on facade; qualify Qwen3.5 0.8B/2B source CPU and selected GPU with unchanged numerical evidence; register Gemma without copying runtime; then vLLM-class temporal serving and complete-domain state; then 27B SSD/dense, Gemma variants, broader tasks/distribution in accepted user sequence.

**Design acceptance gate:** simultaneously satisfy:
- Adding a model without a new op touches only its family, one registration point and tests.
- Adding a device touches only provider/backend and its capabilities/gates.
- Adding a task touches its TaskDriver/source/codec and typed ports, not a model-named token runner.
- Changing a cache strategy cannot mint new physical capacity, skip source checks, publish state or free live encoded readers.
- A new protocol endpoint does not create an inference engine.
- Every functionality is consumable via public zerogravity Rust library; separate zerogravity-serving crate can start a server via programmatic calls; CLI contains no engine/serving implementation.
- One complete second model or non-token task validates reuse with real source/fault/quality and no hidden per-model conditional explosion.
- Full-request performance and honest failure/support receipts replace isolated kernel-only success.

## 15. Cross-document map and handoff

Use this pack:
- [PRIOR_ART_AND_REVALIDATION.md](PRIOR_ART_AND_REVALIDATION.md): in-depth actual llama.cpp, vLLM and historic zerogravity file trees; design comparison and source pin caveats; architectural documents and outstanding gaps.
- [PUBLIC_LIB_AND_SERVING.md](PUBLIC_LIB_AND_SERVING.md): precise 27 October requirements: library API, separate serving library, application assembly, CLI ownership, protocol conformance, lifetime/cancellation and API examples.
- [IMPLEMENTATION_AND_ACCEPTANCE.md](IMPLEMENTATION_AND_ACCEPTANCE.md): source-to-target file migration map, phased zero-runtime-change plan, gates, performance studies and risk reduction.
- [DECISIONS_AND_OPEN_QUESTIONS.md](DECISIONS_AND_OPEN_QUESTIONS.md): original/user amendments, settled architectural invariants, rejected alternatives, explicitly undecided policy choices.
- [HANDOFF.md](HANDOFF.md): mandatory read order, next conversation starting prompt, resync-only instructions and no-edit requirement.
- [TODO.md](TODO.md): held project-specific checklist, separate from existing .agents/plans/TODO.md owner queue.
- Existing reference docs: ../pipeline-composition/STRUCTURE_AND_NAMING.md, ../pipeline-composition/REGIONS_AND_PROFILES.md, ../../docs/MODULAR_AI_SYSTEMS_REFERENCE_ARCHITECTURE.md, ../../docs/TASK_PROGRAM_AND_COMPILER_REFINEMENT_V4.md, and the full project documentation tree.

Do not edit source, active TODOs, prior qualification receipts, existing plans, firmware/device settings, model assets, HEAD, branch or push based on this archive. It is a **side reference** created to preserve future design context pending explicit later resynchronization.
