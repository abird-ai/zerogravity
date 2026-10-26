# Zerogravity
## A composable execution architecture for AI models, heterogeneous accelerators, and distributed state

**Technical whitepaper · Design proposal · Version 0.1 · 12 October 2026**

**Status:** proposed architecture, not a report of an implemented or experimentally validated runtime. The concrete engineering contracts are specified in the accompanying [Engineering Specification](ENGINEERING_SPECIFICATION.md). The source-linked bibliography is in [References](REFERENCES.md).

---

## Abstract

AI execution systems increasingly optimize different layers of the same problem: model operator semantics, low-bit arithmetic, memory residency, batching, compilation, accelerator placement, state reuse, distributed processing and user-facing serving. Many optimizations are valuable only for particular model architectures, shapes, hardware or request regimes. Combining them often requires parallel serving engines, duplicated state managers, incompatible device runtimes or model-specific integrations. Generality, meanwhile, can impose unnecessary dispatch, conversion and synchronization costs.

**Zerogravity proposes a single Rust-owned, composition-oriented AI execution architecture** that separates mathematical semantics, task control, physical implementation, resource residency and session state. An immutable SemanticProgram expresses model computations and versioned effects; a compiled ExecutableBundle contains closed-region implementations and asynchronous resource commands. One RegionExecutionProvider boundary accommodates direct kernels, captured graphs, ahead-of-time accelerator graphs and statically scheduled fabric programs. A small TaskDriver controls different AI tasks without imposing an autoregressive token loop on encoders, speech, diffusion or future training extensions. A per-session coordinator publishes one consistent logical StateRoot only after every required state domain and physical event is prepared. A resource authority admits model, task and device memory under real physical budgets, while fast device-local pools avoid a central per-kernel allocation bottleneck.

The same execution spine also supports independently placed compute, model parameters, mutable session state and immutable caches across local or authenticated remote machines. Built-in multi-model routing separates model identity from warm physical replicas, enabling one-command local use and a versioned OpenAI-compatible API subset without another inference engine.

The proposal targets a measurable Pareto frontier of **semantic fidelity, latency, throughput, memory efficiency, energy, developer effort and composability**. It does **not** claim unprecedented performance, universal model support or functional accelerator backends today. We present the architecture, invariants, trade-offs, falsifiable hypotheses and staged validation needed to determine whether it can deliver those goals.

**Keywords:** AI inference; semantic compilation; model serving; operator specialization; heterogeneous accelerators; asynchronous execution; KV and recurrent state; distributed placement; model routing; Rust.

---

## Contents

- [1. Introduction: why a unified execution system?](#1-introduction-why-a-unified-execution-system)
- [2. Architecture principles](#2-architecture-principles)
- [3. The minimal execution spine](#3-the-minimal-execution-spine)
- [4. AI tasks as distinct control policies over one executor](#4-ai-tasks-as-distinct-control-policies-over-one-executor)
- [5. State consistency is a primary execution contract](#5-state-consistency-is-a-primary-execution-contract)
- [6. Resource ownership and efficient execution](#6-resource-ownership-and-efficient-execution)
- [7. Hardware universality without pretending devices are equivalent](#7-hardware-universality-without-pretending-devices-are-equivalent)
- [8. Distributed execution: place computation, parameters, state and caches independently](#8-distributed-execution-place-computation-parameters-state-and-caches-independently)
- [9. Built-in multi-model service and local UX](#9-built-in-multi-model-service-and-local-ux)
- [10. Quality and security model](#10-quality-and-security-model)
- [11. A falsifiable evaluation program](#11-a-falsifiable-evaluation-program)
- [12. Implementation and evolution strategy](#12-implementation-and-evolution-strategy)
- [13. Limitations and open engineering questions](#13-limitations-and-open-engineering-questions)
- [14. Conclusion](#14-conclusion)
- [References and reproducibility](#references-and-reproducibility)

---

## 1. Introduction: why a unified execution system?

A contemporary AI model may contain dense projections, multiple attention varieties, conditional experts, learned state compression, recurrent matrices, audio/vision encoders and external lookup structures. Its request may require millions of relatively regular prefill operations followed by many latency-sensitive incremental steps. Another model may execute an encoder pass, diffusion iteration or vocoder stage instead of autoregressive text generation.

Hardware diversity makes the challenge harder. A consumer APU shares physical RAM between CPU and GPU; a discrete accelerator has its own memory and expensive interconnect; a rack of GPUs coordinates collectives; a TPU-like or wafer-scale system compiles much larger regions into device-owned programs. A second machine might offer capacity but be separated by slow or relayed network links. A model may need dense compute on one processor while its attention state or sparse experts live elsewhere.

Existing research demonstrates benefits of attention tiling [R02](REFERENCES.md#r02), paged state management [R01](REFERENCES.md#r01), prefix reuse [R04](REFERENCES.md#r04), speculative execution [R03](REFERENCES.md#r03), compiler search [R07](REFERENCES.md#r07), phase disaggregation [R05](REFERENCES.md#r05) and remote tensor transfer [R14](REFERENCES.md#r14). **These are prior-art mechanisms, not original inventions claimed here.** Zerogravity asks a systems question: can their useful *classes of optimization* coexist under one set of mathematical and state/resource contracts, without paying the cost of a generalized interpreted workflow engine at execution time?

The resulting design has two outward-facing requirements. At the developer level, new models, mathematical operators, kernels, device types, media tasks and topologies should be added through narrow interfaces. At the user level, a supported local model should be usable with a single command, and many named models should be served through one predictable interface, with complex details visible only when requested.

### 1.1 Problem statement

We seek a runtime architecture capable of minimizing cost for a supported task over a given physical topology, subject to a correctness and security contract:

```text
Select an admissible execution plan P for model M, task T and topology H
that optimizes latency, useful throughput, memory and energy,
subject to:
  exact registered model/operator/state semantics;
  caller-selected numerical and sampling policy;
  physical resource capacity, data access and device qualification;
  valid state transitions, cancellation and output ordering;
  bounded compilation/runtime overhead and recovery behavior.
```

No single plan is optimal for all inputs. A speed-oriented GPU kernel may be slower at decode batch size one than at prefill. A distributed model might fit in aggregate memory but lose interactive speed to network round trips. The goal is **plan selection and specialization under explicit truth constraints**, not a universal assertion that more parallelism always helps.

### 1.2 Scope

**In scope, architecturally:** typed model semantics, source validation, reference execution, specialized compiled regions, task-specific loops, CPU/GPU/NPU and graph accelerators, multi-model routing, stateful serving, remote execution and storage, authenticated network transport, and resource/quality governance.

**In scope as later optional extensions:** training/fine-tuning through versioned parameter/gradient tasks, additional accelerator architectures, large cluster orchestration, specialized media pipelines and broader API standards.

**Not implied:** execution of arbitrary unknown checkpoint architectures, automatic compilation of proprietary device kernels in Rust, compatibility with all vendor cloud models, unrestricted agent tool execution, or demonstrated performance leadership.

---

## 2. Architecture principles

The design is guided by seven constraints.

**Axiom A — mathematical semantics precede hardware.** A trained operator must have an exact registered meaning, shape, position convention, source quantization and effect contract. A file parser recognizing a tensor type does not make an unknown model executable.

**Axiom B — task control is not model math.** Sampling one token, scoring a document, transcribing audio and iterating a denoiser have different stopping conditions, state frontiers and output validity. A shared executor should not require a fake next-token interface for every task.

**Axiom C — specialization replaces closed regions, not entire engines.** A provider may implement a GEMM tile, a fused layer, a model stage or a whole compiled graph if its externally visible values/effects match the original semantic region. It cannot introduce an independent model registry, session writer, allocator or HTTP server.

**Axiom D — state publication has one authority.** Workers can execute and prepare provisional state; a TaskCoordinator publishes a new logical session root only after the required domains and events agree. Output becomes visible only after the corresponding root publication.

**Axiom E — resources are physical and time-dependent.** Memory addresses are not necessarily free RAM, GPU allocations have asynchronous lifetimes, and data resident on another machine is not zero-cost local state. Admission and device-local allocation are distinct responsibilities.

**Axiom F — distributed placement is optional, measured and trusted.** More devices and machines can increase capacity without increasing single-request speed. Remote execution is chosen only when the workload, network, source rights and recovery contract permit it; a private tunnel is not equivalent to RDMA.

**Axiom G — simple experience, explainable decisions.** Users can run or serve a named local model without learning a compiler IR, manually distributing layers or starting a cluster. Every meaningful nonsecurity configuration can be examined and overridden at a valid lifecycle boundary.

These principles yield a specific thesis: **composability is achieved by fixing what each component may own, not by introducing a plugin abstraction for every internal implementation detail.**

---

## 3. The minimal execution spine

Figure 1 summarizes the proposed system. Arrows represent logical lowering and control; the diagram does not imply a host round trip for every operator or token.

```text
  Local CLI / Rust SDK / API
              |
  Request validation + ModelCatalog / exact alias / ModelRouter
              |
  Artifact / ModelDefinition ---------- Trusted TaskDriver
              |                               |
              +------ SemanticProgram --------+
                       (SIR: math + state effects)
                            |
             legal region transforms, provider selection
                     (temporary compiler views)
                            |
                  ExecutableBundle
    (compiled kernels/graphs + async commands + shape guards)
                            |
         Topology / PlacementPlan / ResourceArbiter
          (coarse admissions, local device subpools)
                            |
          TaskCoordinator / StepPlan / RegionWorkers
          (GPU, CPU, NPU, AOT graph, network peers)
                            |
       prepared domain versions + true ready events
                            |
                 one PublishRoot
                            |
       committed typed outputs + ModeReceipt
```

*Figure 1 — Zerogravity's proposed architectural ownership and compilation flow.*

### 3.1 Two enduring representations

The first representation, **SemanticProgram**, records *what* the model computes: typed tensor and structured values, precise mathematical operators, dimensions, positions, numerical policy and explicit state effects. It supports a reference evaluator and is the basis for correctness tests.

The second, **ExecutableBundle**, records *how and where* a legal closed semantic region executes: prepared kernel or accelerator program handles, shape guards, resource lifetimes, copies, events, collectives and allowed fallbacks. Drivers and topology may require several compiled variants; those are managed as bounded caches.

The conceptual optimizer stages remain useful but need not be separate enduring formats:
- **Region transformation** finds legal fusion, tiling, layout changes, expert grouping or independent output-row partitions.
- **Kernel/program selection** checks shape, precision, driver, ISA, topology, state and evidence for eligible implementations.
- **Physical command planning** lays out asynchronous copies, reservations, submissions, producer-consumer fences and completion events *within* the compiled bundle.

This separation draws general inspiration from extensible compiler research [R06](REFERENCES.md#r06), explicit asynchronous resource modeling [R13](REFERENCES.md#r13) and versioned semantic operations [R12](REFERENCES.md#r12). It does not require embedding an external compiler runtime. The concrete design intent is to remove interpreted semantic graph traversal from latency-sensitive steady-state execution.

### 3.2 Closed-region substitution

A **closed region** has declared logical inputs, outputs, mutable effects, numerical conditions and a compatible fallback boundary. The compiler can ask several implementation providers to prepare it. One may supply a portable CPU implementation; another may supply a fused GPU graph; another may consume a whole model stage in an AOT accelerator compiler.

An aggressively optimized provider is eligible only if it preserves the semantic contract, accepts its shape/dtype/quant, has a valid state bridge and resource plan, and meets safety/quality policy. Some alternative implementations produce slightly different floating-point reductions; equivalence of *mathematical operator* does not automatically mean bit-identical output across hardware. Numerically different but qualified execution is separate from changing the trained model itself.

**Fallback is a composability test.** Disabling a specialized implementation must not require a different model server. If physical state layouts are incompatible, a fallback may require snapshot conversion or replay at a committed boundary rather than an unsafe pointer cast.

### 3.3 Shape-bounded compilation

Accelerator code often specializes batch size, sequence length, quant block, tensor strides, route capacity and attention regime. ExecutableBundle therefore carries shape predicates and bands, not a new kernel for every observed context length. On a guard miss, the runtime selects a prepared generic region, compiles a legal variant outside the latency-critical path when allowed, or reports a precise unsupported shape. It never launches code on an unvalidated shape.

**Prediction:** a bounded set of common shape regimes plus direct device handles can reduce compilation churn and CPU dispatch overhead while retaining rare-shape correctness. This must be measured in a real model, not inferred from the representation alone.

---

## 4. AI tasks as distinct control policies over one executor

A **TaskDriver** owns coarse, trusted task control: which stage executes next, the loop boundary, an output unit, the currently committed frontier and cancellation. It does not implement a general permissionless workflow language.

For language generation, it renders and validates prompt tokens, performs prefill, then iterates decode or target-verified speculative steps. For embeddings it encodes and pools; for speech recognition it handles audio windows and transcript boundaries; for TTS it chains acoustic and vocoder stages; for diffusion it advances latent/timestep state until a complete output can be decoded. Video and multimodal tasks may interleave stages, maintaining their own temporal validity.

A task may use multiple different models, such as encoder plus reranker or target plus draft. Each logical model remains separately identified and admitted. **Multiple models in a task are not replicas of one model.**

**Multimodal model-family stress test (v5 design proposal, 14 October 2026):** The fundamental ownership contracts above accommodate text, speech, image, video, structured decisions and compound models, but the v4 prose alone does not fully specify their execution semantics. A proposed additive contract introduces typed media values and rational timebases, source-verified component assemblies, codec/quantization transforms, non-token decision heads, denoising/flow iteration, and multi-track synchronized StateRoot frontiers. These prevent incompatible assumptions such as assuming a large image-generation checkpoint's generation is either purely AR or purely DiT without auditing its checkpoint-specific source path, mapping CTC/Transducer speech to a text token loop, or treating synchronized audiovisual generation as independent video plus TTS. The same SemanticProgram, ExecutableBundle, TaskDriver and sole PublishRoot authority remain; the execution spine is **not** being replaced. See the [proposed unified multimodal execution contract](../.agents/docs/UNIFIED_MULTIMODAL_EXECUTION_CONTRACT_V5_2026-10-14.md) and [author-source model research atlas](../.agents/research/MULTIMODAL_MODEL_FAMILY_SOURCE_ATLAS_AND_IMPLEMENTATION_2026-10-14.md). Native execution of these additional named checkpoints is not implemented.

The TaskDriver calls compiled model regions through the common device/resource system. Where correct, a **device-native StepProgram** can perform multiple causally dependent operations internally—including routing, sampling or verification—before host inspection, avoiding a mandatory CPU readback at every token. This cannot be promised for every model, grammar or media loop. Unsupported modes retain a correct host-controlled path.

Externally authorized retrieval, tools and multi-agent workflows belong to an **application orchestration boundary above** TaskRequest, not to tensor operators or a hidden shell inside the TaskDriver. This preserves inference-engine safety and keeps the mandatory runtime small.

A future optional **GradientProgram** could express backward operators and transactional ParameterVersion updates. Ordinary inference sessions would remain bound to an immutable checkpoint/adapter generation. This is an extensibility seam, **not** a claim to support training today.

---

## 5. State consistency is a primary execution contract

Many serving optimizations depend on modifying state: KV pages, sliding caches, compressed attention, recurrent matrices, sparse indexers, token-history lookups, sampler/RNG and grammar state. Treating these as implicit mutable buffers makes speculative rejection, prefix forks, cancellation, migration and distributed execution difficult to reason about.

Zerogravity proposes a **logical StateRoot** containing a versioned reference to every required state domain at the same committed task frontier.

### 5.1 Prepare, then publish once

The transition is:

```text
Begin(epoch, committed_root)
  -> execute into private / COW / rollback-safe provisional state
  -> Prepare each required domain and validate common frontier
  -> wait for actual device, copy, IO and peer-ready events
  -> TaskCoordinator checks expected old root and session epoch
  -> PublishRoot(new complete root) exactly once
  -> release committed output events in order
  -> retire old physical resources after their true last-use events
```

**No state domain or device command publishes visible session state independently.** If the second domain fails preparation, the original root remains authoritative; private mutations are aborted or reconstructed. A physical buffer may still be active after a host timeout: its lease must remain held or quarantined until safely quiescent.

This is **atomic logical visibility**, not a claim that all GPU memory writes occur atomically across devices, nor a free distributed durability guarantee. An implementation needs private/COW tails, compatible checkpoint replay or carefully proven undo operations to ensure the published root never references half-mutated state.

### 5.2 Prefix sharing and speculative verification

A prefix-cache hit is valid only if the *entire required model state* can resume from the same causal frontier under an exact source/quant/adapter/template/position/tenant/grammar identity. Matching visible text alone is insufficient. Copy-on-write snapshots allow efficient compatible forks, but cache eviction cannot invalidate a reader holding a live lease.

Speculative decoding proposes candidate tokens, checks them against the authoritative target and commits only the accepted corrected prefix. When advertised as target-distribution preserving, sampling and RNG state must follow a valid verifier algorithm [R03](REFERENCES.md#r03). Approximate alternatives require a distinct explicit quality mode; better speed from altered trained top-k or reduced indexer semantics is **a changed model**, not a transparent optimization.

### 5.3 Streaming and remote failures

Committed output carries a monotonically ordered per-session sequence. A client may receive an event and disconnect before acknowledging it: the runtime can promise ordered **attempted** delivery and an optional authenticated replay cursor, but not exactly-once reception without a separate durable protocol.

Across remote machines, workers prepare versioned state shards under coordinator and placement epochs. The session coordinator alone publishes the new root after all required participants are ready. A lost node, peer lease or old coordinator must be fenced; without durable consensus and checkpoint recovery, the safe behavior may be to stop or replay. **Automatic transparent failover is not an inherent property of a private tunnel.**

---

## 6. Resource ownership and efficient execution

Physical admission and fast allocation operate at different timescales. A global **PhysicalResourceArbiter** accounts for mandatory model weights, adapter/draft/encoder models, KV/recurrent and other mutable state, expert slots, prefix caches, compiler program residency, scratch, DMA staging, SSD/remote IO, OS headroom and multiple concurrent tasks. It decides whether a model/session/placement can be admitted and coordinates evictable capacity.

A **DeviceBackend local pool** performs frequent allocation and reuse from approved reservations with generation-tagged BufferLeases. Retire events track outstanding GPU/NPU/IO readers. A single global mutex on every matmul scratch allocation or token would defeat the high-concurrency design. Conversely, allocating directly from each backend without a shared physical ledger would overcommit a multi-model or shared-memory system.

These ideas follow the general problem of dynamic attention state management described in [R01](REFERENCES.md#r01), but the architecture extends state accounting to hybrid recurrent domains, multiple models and heterogeneous physical memory classes.

**Resource caveat:** no userspace process can guarantee unrelated processes will never change available OS memory. The enforceable goals are conservative admission, pressure observation, bounded buffers, safe failure and recovery—not an absolute claim that the machine can never OOM.

---

## 7. Hardware universality without pretending devices are equivalent

The provider interface adapts to the **minimum efficient execution granularity of the hardware**:

| Executable kind | Native physical unit | Typical specialization scope |
|---|---|---|
| Direct kernel | CPU/GPU/NPU kernel or tile | one op or fused region |
| Captured command graph | prepared device submissions | repeated shapes, decoder or denoiser island |
| Ahead-of-time device graph | compiler-generated closed subgraph | whole layer/model stage |
| Statically scheduled dataflow | compiler-placed on-chip program/fabric | full pipeline or partitioned model |
| Managed device session | opaque physical state under a *verified* state bridge | only qualified programs; often restricted |

A CPU/GPU kernel can run within a simple device worker. A TPU-like graph backend may compile a whole semantic stage into a program with shape guards and its own device-internal schedule [R15](REFERENCES.md#r15). A wafer-scale system may place a dataflow pipeline across on-chip compute and stream weights from elsewhere [R16](REFERENCES.md#r16). Zerogravity should not emulate every microoperation of such hardware with a host RPC or claim native kernel source access when only a vendor compiler is available.

Supported compiler and driver APIs, executable source rights, device program state snapshots, compile/upload/warmup costs, internal memory and collectives are all **capabilities to verify**, not presumed from a product name. A remote text-generation service that exposes only input/output and no trustworthy mathematical state bridge is an **external TaskServiceAdapter**, not an equivalent native implementation of arbitrary closed semantic regions.

### 7.1 Hardware topology matters

A coherent CPU/GPU workstation shares physical RAM and bandwidth [R18](REFERENCES.md#r18); a datacenter GPU uses independent HBM and fast peer links; a TPU pod uses its own interchip topology [R15](REFERENCES.md#r15). GPU virtual addressability is not necessarily spare physical memory. Numerical formats must be identified precisely: differently scaled FP4 formats or temporary low-precision compute are **not** interchangeable model storage artifacts [R17](REFERENCES.md#r17).

The system therefore treats compute domains, memory tiers, links, compiler domains, driver versions and power/safety constraints as separate topology properties. Backend selection must be based on actual qualified capabilities and measured task costs.

---

## 8. Distributed execution: place computation, parameters, state and caches independently

Most distributed systems begin with a single model partition: layers, tensors, experts or pipeline stages. Zerogravity proposes a more general **PlacementPlan** with four orthogonal maps:

1. **ComputePlacement** locates semantic operators, layer ranges, attention cores, FFN/experts, encoder/denoiser stages or graph programs.
2. **ParameterPlacement** locates trained dense/shared/expert weights, adapters, immutable lookup tables and packed artifacts.
3. **StatePlacement** locates each mutable session/layer KV, recurrence, indexer, grammar or media state shard under versioned leases.
4. **CachePlacement** locates optional immutable reusable prefix snapshots and derived artifacts. A presence hint does not grant authority to mutate live session state.

A **TransferPlan** joins the maps with explicit activations, state snapshots, partial reductions, copies, collectives, memory reservations and authenticated transport events. These are views within ExecutableBundle—not another durable IR or independent cluster inference engine.

### 8.1 Representative topologies

**Colocated execution** keeps layers, weights and state on one machine and remains the latency baseline.

**Layer pipeline** partitions model layer groups over machines, typically with each layer's KV/recurrent state colocated with that layer's computation. It transfers activations at stage boundaries rather than the full cache every token. It may enable a model too large for one node, while adding serial per-token hop latency.

**Attention/FFN disaggregation** keeps attention and its potentially large KV state on a memory-rich worker, with projections/experts on another compute-rich worker. It exchanges compact queries, hidden activations and results; this can be better than moving large caches but may suffer repeated cross-network calls. Operator-level and P/D disaggregation are extensively explored in prior work [R05](REFERENCES.md#r05) [R10](REFERENCES.md#r10).

**Remote prefix cache** hydrates one immutable compatible snapshot from a cache worker, then computes locally. Recompute may be cheaper than transferring gigabytes over a constrained link.

**Sequence-sharded attention** places disjoint token-range KV shards on attention workers and merges partial softmax outputs. For ordinary softmax attention each shard returns a local maximum score, shifted exponential denominator and weighted-value numerator; global rescaling reconstructs the attention result in real arithmetic. Positions, causal masks, finite precision, global selection and state updates must be correct. The same algebra does **not** automatically partition recurrent or learned sparse indexer state [R08](REFERENCES.md#r08) [R10](REFERENCES.md#r10).

**Prefill/decode separation** places prompt processing and token generation on different workers, transferring the *full compatible hybrid state* at a safe committed frontier, not merely plain transformer KV [R05](REFERENCES.md#r05).

**Tensor/head/expert parallelism** uses collective communication or compact routed rows on physically suitable links. A bandwidth-starved tunnel usually cannot sustain per-layer allreduce at datacenter-like performance.

### 8.2 Network and trust

An opt-in **PeerAgent** on another authorized machine reuses the same compiled-region executor and physical admission system; it is not a separate public model server. The control plane handles node identity, capabilities, authorization, placement epochs and lease/failure events. The data plane chooses actual eligible local DMA/RDMA or bounded reliable encrypted transfers, with source/destination leases and flow control [R14](REFERENCES.md#r14).

An encrypted overlay can connect machines through direct peers or relays, but does not create GPU-RDMA, universal coherence or low-latency bandwidth. Effective link rate, RTT/P95 jitter, copy/crypto costs, loss and egress are inputs to placement [R22](REFERENCES.md#r22). Private prompt, weight and state export requires explicit user/tenant/model rights; the existence of a peer does not grant permission to send it all model inputs.

Relocation takes place only at a compatible committed TaskFrontier: reserve and warm the destination, transfer/validate all required state, fence the old writer and publish a new placement epoch. A lost coordinator or network partition may require fail-stop/replay rather than magical live failover.

### 8.3 The performance condition for distribution

Distribution is worthwhile only when increased compute capacity, memory availability or overlapped work outweighs additional communication and recovery costs. A useful *rough model*, not a scheduler formula, is:

```text
T_task(P) ≈ longest critical dependency path of:
    region compute
  + transfers (effective bytes / measured bandwidth)
  + link RTT and synchronization
  + queue, cache hydrate and program warmup
  + state publication and output delivery
subject to correctness, physical capacity, trust and latency constraints.
```

Legal parallel execution reduces the critical path only where independent device/transport work overlaps. A large activation transferred once at a layer boundary is different from a remote read of growing KV **on every token**. A slow-link distributed plan can be a useful *capacity mode* even when single-request generation speed decreases. This is an honest and important product distinction.

---

## 9. Built-in multi-model service and local UX

Users should not have to operate a cluster scheduler to use one local model. The proposed terminal surface begins with:

```sh
zerogravity run ./model.gguf
zerogravity run coding
zerogravity serve
zerogravity models list
zerogravity inspect coding --explain-plan
```

*Illustrative, not implemented or a frozen public interface.*

The built-in model router manages **four distinct identities**: an immutable RegisteredModel checkpoint/quant/adapter; a versioned alias/profile carrying a friendly name and configuration; one or more physically admitted ResidentModel replicas/deployments; and per-request mutable TaskSessions/StateRoots.

A catalog may contain many model definitions without allocating their tensors. Concurrent residency is bounded by real physical resource capacity. The manager deduplicates cold loads, reserves before loading, warms an executable when possible, serves other resident models concurrently and releases idle instances only after active sessions, device events and state leases are safe.

**Exact-model routing is the default.** Requesting a named model never silently selects another checkpoint, quantization or adapter because the original is cold or overloaded. Balancing among *equivalent* replicas is a different operation, preserving session/cache affinity and the selected numerical contract. A policy alias may intentionally select a different model only when explicitly configured and disclosed. One sharded deployment spanning several machines is not the same as several replicas.

The server exposes one model-aware API surface over the same executor as the CLI. An initial **OpenAI-compatible subset** would include model listing and chat completions with correct request/stream/finish/error semantics; embeddings need a validated encoder TaskDriver; Responses has distinct event and state-store requirements [R21](REFERENCES.md#r21). A local endpoint should not claim all cloud-specific features, built-in tools or conversation retention merely because a path exists.

Configuration is progressive: safe automatic defaults for `run`, model management and inspection for daily use, then typed global/model/alias/CLI/request profiles, placement, routing, numeric, hardware and security controls when needed. Hard model semantics, physical safety and tenant authorization always take precedence over ordinary overrides.

---

## 10. Quality and security model

The architecture distinguishes four kinds of claims:

| Claim | Required evidence |
|---|---|
| **Semantic equivalence** | exact source model formula, shapes, positions, attention/router and all state effects |
| **Numerical agreement** | kernel/source dtype and quant conditions, reference tensors/logits, justified error tolerances |
| **Task quality** | independent model/task fixtures under realistic prompts, contexts, audio, image/video or scores |
| **Sampling/distribution fidelity** | exact target verifier algorithm where promised; RNG/grammar state and rejected branch rollback |

A faster model with reduced expert top-k, altered sparse selection or new lossy checkpoint is not the same model merely because its responses look plausible. Alternative numeric quality modes must be visible and testable.

Security follows the same ownership architecture. Checkpoint files are untrusted data, not arbitrary executable code. Model-generated tool calls are output data, not shell permissions. Compiled device programs require source/toolchain trust. Remote workers require explicit enrollment and data-access rights. Local API listeners default to loopback; external exposure needs intentional authentication/TLS and restricted model-management controls. Prefix caches and metrics must not leak tenant context or hidden model inputs.

**No absolute hardware fault prevention is claimed.** Unsafe shared power/fabric configurations and unknown device program sources must be disabled until specifically qualified; device DMA can outlive a host timeout and requires proper quarantine or reset.

---

## 11. A falsifiable evaluation program

The architecture is a proposal. Demonstrating its value requires an implementation and independent measurements. In particular, **composability is a testable property**, not just a diagram.

### 11.1 Functional milestones

1. **Reference semantics:** parse one real trusted checkpoint and tokenizer; validate exact target logits/state and produce a correct CPU result.
2. **Compiled physical execution:** run the same model on a qualified GPU using a prepared executable, safe buffer/event lifetime and a forced generic fallback.
3. **Atomic state:** inject a failure in one of several state domains and prove no partially published root or speculative output.
4. **Extension drill:** add a second model operator and a new kernel provider without changing unrelated HTTP, model registry, allocator or session owner.
5. **Non-token task:** run an encoder or mocked iterative media driver through the same executor without fake autoregressive state.
6. **Multi-model product:** one-command direct run, same-engine router server, exact aliases, safe concurrent residency/replica balancing and official client compatibility tests.
7. **Distribution:** reproduce a single model across two authorized workers, per-layer state colocation, safe transfer/migration and peer failure behavior; separately test optional remote cache and attention partition.

### 11.2 Performance measurements

For each supported model/artifact/quant/device/task combination, capture:

- time to first **valid committed** output, total task completion and P50/P95/P99 tail latencies;
- useful throughput at concurrency 1, 2, 4, 8, 16 and beyond, with native batching and fairness;
- program compilation/warmup and cold versus warm model load;
- kernel/device occupancy, source/numeric agreement and device/host synchronization overhead;
- actual peak physical DRAM/VRAM/HBM, model+KV+scratch/graph cache, cache hit/miss, SSD/network bytes;
- energy, CPU host overhead, thermal/clock conditions, SDK/compiler/driver versions and exact topology;
- load-balancing behavior for multiple models, replicas, output streaming and queue contention;
- network direct, encrypted direct and relayed path performance, capacity gains **separate from** generation speed.

A benchmark claim should be associated with exact source checkpoint/hash, model/adapter, prompt/context, sampler/output type, execution plan, driver/hardware and full task quality. Baselines must use matched semantics. Model-specific work may outperform a generic path and still lose to an external specialist; publishing both positive and negative outcomes is essential.

### 11.3 Explicit hypotheses rather than results

**H-A:** prepared, device-native execution reduces CPU synchronization and dispatch overhead for eligible models.

**H-B:** closed-region substitution allows extreme kernel performance without additional model-serving engines.

**H-C:** coordinated memory admission plus device-local allocation improves multi-model concurrency and reduces preventable resource errors.

**H-D:** independent compute/state/cache placement enables some previously infeasible model/context sizes and improves SLOs on suitable fast networks.

**H-E:** a small, versioned task API can add another model, accelerator or task with limited change radius and without breaking prior workloads.

Each hypothesis can fail under particular hardware or models. Success is determined by reproduced end-to-end results, not chosen vocabulary or source-author benchmark numbers.

---

## 12. Implementation and evolution strategy

A usable system should precede a large framework. The initial implementation should begin as **one Rust library and one CLI binary**, split into additional crates only where target compiler, security or build constraints justify it. The first vertical slice is a real supported model, trusted artifact parser, CPU reference, minimal semantic/effect representation, one task driver, simple legal region/provider, asynchronous buffer/command checker, one physical GPU and a committed output.

The second slice proves **architecture reuse**: another model/op, a missing fast-kernel fallback, a non-token task and controlled physical resource sharing. Multi-model routing and one loopback API then use the same engine; distributed execution progresses from a simulated remote worker and an authorized two-node layer split to more demanding attention/state and collective topologies.

Full TPU/static-dataflow, advanced media pipelines, high-availability clustering, native accelerator compiler toolchains and gradient training remain **separately qualified extensions**. They should not inflate the startup or runtime cost of a single CPU/GPU text request.

The [Engineering Specification](ENGINEERING_SPECIFICATION.md) states exact component contracts, model/task/device/placement interfaces and failure tests. The [Implementation Playbook](../.agents/plans/AGENT_IMPLEMENTATION_PLAYBOOK.md) organizes the preserved 24-phase roadmap into bounded, independently verifiable work packages. Prior-art licenses and historical decisions remain auditable.

---

## 13. Limitations and open engineering questions

This proposal cannot guarantee universal inference superiority. Known risks include low-bit GPU/accelerator code-generation efficiency, dynamic shape compile cost, numerical behavior of fused kernels, high-overhead abstraction interfaces, failure recovery of opaque device-managed state, transient pressure in unified-memory systems, poor network/tunnel links, and inconsistent model-family/checkpoint semantics. A narrow hand-tuned implementation can outperform a general system on its favorite shape; our claim is that such an implementation *should be integrable as a candidate* without duplicating the rest of the serving runtime.

Some device vendors expose a compatible graph compiler or an opaque model API but no independently implementable Rust device kernel target. A Rust host cannot honestly claim a fully Rust-authored native accelerator compiler on that basis. Storage/snapshot and model weight redistribution rights can constrain distributed placement. Network-exactly-once results and automatic coordinator failover require additional durable protocols and are not included by default.

Training/fine-tuning is intentionally a future parameter-version/gradient task seam rather than a hidden default requirement. The project may later decide to specialize for selected AI modalities or accelerator families based on actual demand. Source registry choices, stable API levels, ABI/compiler policy, default model, model downloads, security deployment and service configuration are not all frozen.

Finally, publication of a design is **not** a substitute for implementation artifacts. This whitepaper is ready for technical review as a proposed system design; any future experimental-results edition must add real measurements, reproducible sources and independent quality evaluations.

---

## 14. Conclusion

Zerogravity proposes a **single composable AI execution spine** that spans model semantics, optimized regions, accelerator programs, asynchronous resources, state transactions, model serving and distributed placement. The organizing decision is to keep ownership narrow: one canonical mathematical program, one prepared physical execution plan, task-specific control, authoritative session-state publication, global resource admission with local fast allocators, and interchangeable qualified providers.

A new mathematical model changes its model definition and genuinely novel operators. A new accelerator changes region lowering and its physical capability adapter. A new task supplies a task driver and validated output state. A new placement changes a costed physical plan and transport. A new API changes the typed ingress/egress adapter. **None should require a new model-serving engine.**

The corresponding user promise is equally important: a supported model can be started with one command, while every meaningful specialization and distributed decision can be inspected and controlled when needed.

This is a concrete and falsifiable architecture for evolving AI execution systems. Whether it becomes a leading runtime will depend on carefully implemented kernels, trusted state/resource mechanics, benchmark discipline and the practical cost of adding the next generation of models and hardware.

---

## References and reproducibility

Full bibliography: [References](REFERENCES.md). The academic and primary-source material covers memory-paged serving, IO-aware attention, target-authoritative speculation, structured serving, phase disaggregation, typed compilers, accelerator and transport architectures, local model UX and API contracts. **Names and code licenses are retained solely as attributed prior art; zerogravity's component names and executable design are independent.**

**Documentation release:** architecture v4.3, manuscript version 0.1 (12 October 2026). **Implementation status:** a packaging-only Rust binary and release/CI scaffolding partially advance P00; the **inference engine and P01–P23 remain NOT_STARTED**, and no user hardware/model/tunnel benchmark is represented as performed here. Figures are architectural illustrations, not screenshots of running software.
