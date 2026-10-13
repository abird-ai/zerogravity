# zerogravity — Built-in model router, multi-model residency and serving contract (v4.3)

Design requirements below. Implemented/tested scope and unfinished work are
tracked only in [the plan TODO index](../plans/TODO.md).


**15 Oct 2026 — P09 three-stage admission and multi-model scheduling:** [Joint serving contract](JOINT_CONCURRENT_SERVING_OPTIMIZATION_AND_GOODPUT_CONTRACT_2026-10-15.md) distinguishes (1) bounded API/tenant queued request and input-byte credits, (2) per-model admitted sessions, **queued prefill token work** and actual immutable source/private state resource envelopes, and (3) device active sequence/verification row/scratch/IO budget. Do not allocate full KV for queued requests, confuse model registration with physical simultaneous residence, or silently change requested quant/model to fit. Per-tenant fairness and stable model session binding with typed overload/latency receipts precede new work. One ModelRouter resolves exact model and legal replica **once at admission**, not every token/audio frame; compatible models may time-share one physical worker but must never be fused into the same tensor GEMM without source proof. SERVOPT-01..07/22/31 test these boundaries.


**15 October 2026 — RESIDENCY POLICY IS NOT MODEL IDENTITY:** [First-class SSD tiered materialization](UNIFIED_VIRTUAL_TENSOR_RESIDENCY_AND_SSD_STREAMING_2026-10-15.md) allows one registered verified model to run from physical GPU/UMA/RAM/SSD backing without a second model alias or extra service. The built-in Router binds exact ModelAssembly/derived quant and numeric policy; the storage planner automatically prefers full residency when mandatory weights+state fit with OS headroom, else source/backend-qualified streamed experts/lookup rows, else typed resource/latency refusal. Immutable source tensors may share loaded/streamed cache across sessions under physical leases; user KV/draft/recurrent state remains separate. A model too large for RAM can be *capacity-feasible* yet fail its interactive latency target—report effective cache/SSD QoS and no hidden model downgrade. SSD support remains backend/format/physical evidence scoped; tests SSD-01..32.

**AUTHORITATIVE DESIGN · 2026-10-12 · NOT IMPLEMENTED.** The model router is a **core capability of one zerogravity runtime**, not another inference engine or a mandatory front proxy/process. Read alongside [minimal semantic/task architecture](MODULAR_AI_SYSTEMS_REFERENCE_ARCHITECTURE.md), [execution/state protocol](MINIMAL_EXECUTION_SPINE_AND_PROTOCOL_V4.md), [full physical budget](MEMORY_RESIDENCY_AND_IO.md), [distributed placement](DISTRIBUTED_PLACEMENT_AND_NETWORK_EXECUTION_CONTRACT.md) and [local CLI/API/config contract](ZERO_CONFIG_CLI_CONFIG_AND_OPENAI_API_CONTRACT.md). Named source-project research is isolated in [attributed prior-art research](../prior-art/MULTI_MODEL_ROUTER_AND_API_PRIOR_ART_RESEARCH_2026-10-12.md).

**User goal U-021:** a blissfully simple `run <model>` and `serve` experience, **plus** a built-in load-balancing multi-model router, many registered/available models, multiple simultaneously resident/active model instances where physical resources permit, per-model and per-request configuration, native local/distributed placement, an OpenAI-compatible interface, full control through progressive disclosure. Simple defaults are a product requirement, not permission for silent model/quant/quality change.

## 1. One engine, four separable entities

**A. RegisteredModel** — an immutable validated `ModelDefinition` and `ArtifactKey` with checkpoint/shards, stored quant, tokenizer/template, adapters/sidecars/model license, supported TaskKinds/semantic op schemas and default safe config. It may have *zero* loaded weights or compiled programs.

**B. ModelAlias/ModelProfile** — a human-readable versioned name/tag and optional **configuration overlay** referencing one RegisteredModel/adapter tuple or a separately named policy-defined model group. `chat`, `code`, `vision`, `embedding` can be friendly aliases, but exact identity and capability are always inspectable. An alias is **not** a physical model or source checkpoint. Defaults may include context, sampler, speed/quality, persistent resident policy, compatible draft/adapter, per-tenant access and placement preferences; never mutate source model data silently.

**C. ResidentModel / Replica** — an independently admitted, loaded and warmed physical execution instance of one resolved model and source/adapter identity, using an `ExecutableBundle` and a `PlacementPlan` across qualified devices/peers. Several replicas of an **equivalent logical model** may exist for load balancing or capacity. A single replica may be **sharded across several machines** (one distributed instance), and multiple replicas may share immutable weights/cache storage where physical identity and leases permit.

**D. TaskSession / Request** — per-user/task control and mutable full model state, current `StateRoot`, output stream and workload requirements. Multiple TaskSessions share one ResidentModel/weights/compiled plan but **cannot** mutate each other's KV/SSM/sampler/grammar state. The same TaskDriver/TaskCoordinator and physical ResourceArbiter handles CLI and HTTP clients.

**Do not conflate:**
- many **registered** models vs many **resident** models vs many **concurrent sessions**;
- one model **split across peer devices** vs multiple **replicas of the whole model**;
- a **model alias/profile** vs `auto` **model-choice policy**;
- a **model stage graph with multiple compatible models** (e.g. encoder/target/drafter) vs **routing one request to a different target model**;
- a **compiled KernelPlan/ExecutableBundle** vs model artifacts or session state.

## 2. Minimal architecture — routing above shared execution, never hot-per-token proxy

```text
CLI run / native SDK / OpenAI-compatible API / typed media API
                          |
              Parse + Auth + Request validation
                          |
                 ModelCatalog + ModelAliases
            (versioned artifact/profile/capabilities)
                          |
                ModelRouter / Admission
          exact identity or explicitly configured policy
                          |
            ResidencyManager + ResourceArbiter
           resolve/load/warm/replica/evict/memory
                          |
               RequestPlacementSelector
          (task/model/phase/shape/cache/peers)
                          |
            TaskCoordinator + TaskDriver
                (same as ordinary run)
                          |
           ExecutableBundle + RegionWorkers
            local and/or authorized remote peers
                          |
      Prepared full-state versions → one PublishRoot
                          |
      Canonical committed TaskResult / stream / receipt
                          |
             CLI, SDK or API adapter formatting
```

There is **no** second token scheduler, duplicate model-weight allocator, model-specific HTTP process or proxy-in-proxy requirement. The router runs at **request admission and safe model/session/placement boundaries**, not once for every token/attention kernel. The device-native StepProgram's hot path remains intact and protected by the same physical state/lease semantics.

For a small embedded use, `run` may create the engine **in-process**, with no daemon. A configured long-running `serve` process can reuse the same engine objects. An already-running **same-user trusted local service** may be reused by CLI only after an explicit/verified local socket/process identity handshake, not by blindly connecting to an unrelated HTTP listener. No auto-install systemd/launchd service or privileged background daemon.

## 3. Typed identities, versioning and naming

Illustrative, not frozen Rust API:
```rust
struct RegisteredModelId(ArtifactAndSemanticKey);
struct AliasId(String);                // human-facing name, tenant scoped
struct AliasRevision(u64);            // atomic config generation
struct ProfileId(String);
struct ModelInstanceId(u128);
struct ReplicaGroupId(u128);
struct ModelGeneration(u64);          // resolves immutable artifact and overlays
struct DeploymentId(u128);            // one physical PlacementPlan (may span peers)
struct SessionRouteId(u128);

struct ModelBinding {
    registered: RegisteredModelId,
    profile_revision: AliasRevision,
    adapter_key: Option<AdapterGenerationId>,
    tokenizer_and_template: SourceTextRenderingId,
    allowed_numerical_mode: NumericContractId,
}
```

Aliases resolve under **one atomic catalog revision snapshot at request admission**. Subsequent alias updates never change an active session model/adapter/template/quant or invalidate its current StateRoot without explicit new session/replay. Hot reload/prewarm uses a newly published alias version; old active sessions continue on their old immutable source/runtime handles until drained or migrated by a **compatible, qualified state transformation**.

**Model naming constraints:** path files, offline content-addressed hashes, optional verified registry references and friendly names are separately parsed. User names must not bypass filesystem allowed roots, tenant scope or pull-source/license policy. A model alias cannot point to an arbitrary external API host without a separately configured trusted external TaskServiceAdapter.

## 4. Distinct route decisions: source identity, replica, phase and application model selection

### 4.1 Exact-model request (default, no substitution)

If the user issues `run mistral-small` or an API request `{"model":"coding"}`, resolve `coding@revision` to its **exact RegisteredModel checkpoint/quant/adapters/profile**. Then select an eligible *execution replica/placement* implementing **the same semantic/numerical contract**. A router must not silently change Llama↔Qwen, Q4↔Q8, a LoRA overlay, tokenizer roles or JSON-schema semantics because one is faster/loaded.

Aliases can be modified by their authorized owner; stable per-session bindings and `ModeReceipt` show the resolved immutable source identity and actual backend.

### 4.2 Replica load balancing (same logical model)

Choose among **equivalent** loaded/qualified replicas using measured:
- request task/operator/shape/context and requested output/mode;
- queue lengths, workload estimate and fair-share budget, predicted TTFT/P95/service time, source cache/prefix affinity;
- available model-weight resident bytes and per-session KV/state reservation, graph warm status;
- device temperature/power/safety/driver qualification and physical topology direct/relay links;
- tenant/data locality/allowed host, source-license rights;
- current session's state location, previous root/replica affinity and migration cost.

Prefer the replica holding the **compatible committed session StateRoot**. Stateless requests can route dynamically across replicas; stateful continuation is sticky unless a safe checkpoint/state migration is prepared and verified. Random or round-robin replica routing is a **baseline candidate**, not the right default for expensive stateful KV and long prompts. Hysteresis avoids ping-pong; no unbounded optimization cost per request.

Replicas may have different compiled execution tactics/accelerators, but only if both satisfy **the selected model/numerical/state compatibility policy**. A numerical-mode change that affects client-visible determinism must be opted into or pinned, not implicit because different node available. Strict deterministic requests can pin a specific qualified numerical plan.

### 4.3 Phase/task and multi-model pipeline routing

A single TaskProgram may explicitly bind several models: target plus trained MTP/draft, embedding model plus reranker, vision encoder and language model, speech encoder and decoder, or TTS conditioning and vocoder. These are **distinct named logical roles with separate artifacts and lifetimes**, all reserved under one TaskSession/application budget. The router schedules their stages/locations according to exact declared compatibility; it does not treat auxiliary model weights as an accidental replica of target.

Prefill/decode disaggregation or attention↔FFN placement is a **placement decision for one source model**, not request router silently changing target. The existing distributed PlacementPlan and ModelRouting manager must use the **same PhysicalResourceArbiter** to avoid capacity promises that conflict.

### 4.4 Optional intelligent target-model selector (explicitly configured)

A policy alias such as `auto` or `fast-private` may select among **different RegisteredModels**, but only if caller/admin config explicitly names allowed candidate pool and task/quality/privacy/latency conditions. Its result includes actual model selected and numeric/quality details in ModeReceipt; strict `exact` requests cannot fall back to another model or lower quant. This is an *application-level routing choice* and is **not** the same as load balancing equivalent replicas.

Policies cannot silently:
- choose a source model with different tokenizer/template/role semantics for an **ongoing session**;
- route sensitive input or weights to unauthorized external/cloud peers;
- decrease user-selected numeric/sampling quality or trained top-k;
- claim compatible token-level stream state across unrelated models;
- use remote proxy API as if it were a genuine local RegionExecutionProvider.

A failed exact model request gets typed insufficient-resources/no-compatible-replica error or an explicit user alternative suggestion, not wrong model output.

## 5. Model residency lifecycle, cold-start and scale-to-zero

```text
REGISTERED ───────► RESOLVING/VERIFYING ─► PREPARING/COMPILING
                     │                         |
                     ▼                         ▼
                   FAILED               ADMITTING MEMORY
                                                |
                                          LOADING WEIGHTS
                                                |
                                           WARMING PLAN
                                                |
                                              READY
                                             /   |   \
                                          BUSY IDLE DRAINING
                                             \   |   /
                                            COOLING
                                                |
                                            UNLOADING
                                                |
                                             UNLOADED
```

States are bounded, versioned and reported; optional side tasks `FETCHING`, `CONVERTING`, `UNSUPPORTED`, `BLOCKED_BY_POLICY` are **not** silently mapped to READY. Source downloads require explicit trusted source and user authorization (interactive consent or a user-requested pull). Loading means tensor/program verification and memory reservation before actual program execution. `READY` means enough validated devices/kernels/weights and model/task capabilities to accept the advertised request class; a cold compiled backend may be `PREPARING` while a portable variant is `READY`.

On-demand model access:
1. resolve alias/source and effective profile against hard security/numerical/device limits;
2. check catalog and residency of exact compatible artifact/adapter and TaskKind;
3. if not loaded, deduplicate concurrent load attempts by `ModelBinding`, reserve safe physical model+KV/graph+scratch resources, prepare ProgramLifecycle and load/warm once;
4. queue bounded waiting requests and let other resident models keep serving while load runs asynchronously; fail with useful retry/load state if deadline insufficient;
5. when ready, place requests on existing/new compatible replica and produce committed TaskResult;
6. when idle and no active pinned/snapshot-unreplayable session lease, eligible instance can cool/unload after configurable TTL or pressure; immutable catalog record remains.

**As many models as needed** means unbounded/large catalog and *as many concurrent resident instances as actual capacity policy supports*. ResourceArbiter checks **physical RAM, GTT versus true system memory, GPU HBM/VRAM, pinned buffers, per-context KV/SSM, graph/code cache, encoder/draft/adapter, distributed peers and OS reserve**. Do not promise infinite resident models. Saturation triggers safe queueing, idle replica unloading, allowed cache eviction, alternative qualified placement or explicit rejection—not host/system OOM.

### 5.1 Warm/keepalive policies

- `on_demand`: first admitted request loads; idle unload when beneficial.
- `keep_warm_for(duration)`: retain eligible weights/program after last request if resources permit; policy may be preempted by pressure unless explicitly pinned.
- `pin`: reservation for frequently used model/group, subject to owner permissions and hard headroom; cannot pin beyond safe physical budget.
- `preload/on_startup`: optional admin-selected only; don't automatically load every registered model.
- `scale_to_zero`: no resident weights after idle, with retained immutable source and any compatible **committed** state snapshots only under explicit session preservation/retention policy; otherwise warn that volatile sessions are lost.
- `max_resident_models`: optional administrative **safety/fairness bound**, not fixed universal 3/4/unlimited. Actual physical capacity is always the primary cap.

**Eviction must not corrupt active sessions or in-flight DMA**. The residency manager cannot unload an instance while TaskCoordinator has unresolved prepared state, unquiesced device graph or client-active unsnapshotable session. If a model is idle but its sessions have safely serialized/resumable state elsewhere, it may be evicted after owner policy and version/schema verification. On memory pressure, release optional cache/unused instances first; admit/reject safely rather than sneakily quantizing active weights. Avoid unload/reload thrashing for alternating large models via hysteresis, recent-load cost and fair-share.

### 5.2 Model replicas, sharding and placement

- **One deployment** may shard a model's tensor/layer/expert/state across several peers; router knows this is **one** replica's cohesive PlacementPlan.
- **Another replica** may independently load the same exact semantic checkpoint elsewhere with separate physical state roots; load balancer can choose it for new sessions.
- Source immutable tensors/adapters might be shared via content-addressed cache within a single host, but physical weight copies and driver allocations are budgeted separately.
- Local, trusted tunnel and rack nodes have independent physical reservation ledgers; router must not use a candidate placement that violates explicit model license/tenant/data locality.
- A new model can be loaded on a node already running another model if headroom, graph/kernel, memory and scheduling cost qualify. Scheduling data-parallel independent models vs sharding a single huge model are **different plan choices**.

## 6. One global physical budget, two scheduling timescales

**Model/replica level (coarse):** resource admission, warm/pin/unload, model placement, task/model pool quotas, prefetch/prefix/locality, idle budgets and power/temperature. The router must not hold a **global lock on every token** or perform per-token model alias resolution.

**Within an admitted model/replica:** existing TaskCoordinator and DeviceWorker schedule request rows with native batching, KV context and cross-session shared immutable weights, speculative verifier, async IO and fairness. Compatible request stage groups batch physically where exact source/task/adapters/numerics/shape permit; unrelated models may execute concurrently on separate streams/devices if resources allow. **Do not presume arbitrary same-GPU model kernels can batch as one mathematical model**, and don't serialize them through one global HTTP request mutex.

**Cross-model scheduler** may optionally amortize shared immutable base weights for compatible adapters and route compatible tasks to a warmed model. Tiny models with small contexts may coexist; giant models can use entire device/UMA/SSD+remote sharding. The `PhysicalResourceArbiter` remains source of truth for memory; a router-side “free VRAM estimate” is advisory, never authority to overcommit.

Queue fairness:
- per-model and global limits for pending bytes/tokens/tasks, separate model-load waiters vs active inference requests;
- per-tenant quotas and load-shedding policy, fairness/aging and deadline/SLO tags (interactive vs batch);
- **not** one huge FIFO that starves loaded interactive models while an unrelated giant model swaps in;
- bounded prefill/cold-load work and model-specific context/parallel reservation; avoid denial of service through model-switch thrashing;
- respond with known 429/503/429 retry metadata where appropriate, not hang forever.

## 7. Routing snapshot and request consistency

A canonical request is **resolved and pinned** once at admission:
```rust
struct ResolvedRoute {
    alias: AliasId,
    alias_rev: AliasRevision,
    binding: ModelBinding,
    task: TaskKind,
    generation_options: ValidatedRequestOptions,
    route_policy: RouterPolicyId,
    admitted_instance: Option<ModelInstanceId>,
    placement: Option<PlacementPlanId>,
    state_root: Option<StateRootId>,
    scope: TenantAuthorizationScope,
}
```

The `ResolvedRoute` is a **logical request control record**, not a per-token payload. Subsequent routing to another replica or backend during session continuation requires exact source/quant/adapter/state/schema compatibility and prepared state transfer under a new placement epoch at **committed TaskFrontier**, as specified in [distributed placement](DISTRIBUTED_PLACEMENT_AND_NETWORK_EXECUTION_CONTRACT.md). Streaming output model ID and usage correspond to the pinned resolved model, not an alias remapped halfway through generation.

Alias updates are transactional:
- create/edit new immutable revision, validate target source/template/TaskKind and profile config against policy;
- atomically publish alias map revision for *future* requests;
- existing streams/sessions continue under old binding or explicit safe migration;
- delete/unload waits on references/leases; alias removal doesn't destroy immutable model blobs that others reference;
- restart recovery reconstructs catalog from versioned manifest, not from transient warmed GPU pointers.

## 8. Per-model settings: explicit scope and safety

Every setting declares `Scope`, `Type`, `Validation`, `MutableWhen`, `SecurityFloor`, `CacheInvalidation` and `EffectiveSource`.

| Configuration class | Examples | When it can change |
|---|---|---|
| **Immutable source identity** | checkpoint SHA, tokenizer, tensor stored quant, model formula, fixed trained top-k | different RegisteredModel/alias revision; never silent runtime override |
| **Model-load plan** | devices, memory class, per-layer placement, quant pack, GPU fractions, offload, graph/kernel preferences, tensor/expert shards | before loading/warm; change requires new deployment or safe drain |
| **Resident instance** | min/target replicas, preload/pin/TTL/eviction, max concurrent models, device headroom, energy budget | admission policy at safe boundaries |
| **Per-session state** | context limit, KV pool, prefix cache, grammar/sampler source, LoRA adapter selection | at session start or only when compatible snapshot/replay proves legal |
| **Per-request generation** | temperature, top_p, top_k (where supported), max output, stop, seed, response_format, tool choice, reasoning visibility | validated per request; cannot break active session semantics or hard auth/physical limits |
| **Task pipeline** | encoder model, draft model, reranker, speech/vision sidecars, stage output codec | task admission or new TaskProgram revision |
| **Router policy** | exact model aliases, replica pool, sticky session, cache affinity, SLO, model selector opt-in, queue limits | policy revision; active routes remain bound |
| **Topology/network** | allowed peers, strict locality, preferred devices, transport, state/weight/cache placements | safe new PlacementEpoch only, explicit remote authorization |
| **Service/security** | listen bind, TLS/mTLS, keys, allowed registry sources, auth scopes, log privacy | admin config/restart or safe hot reload; **never downgraded by request** |

**No implicit source quant downgrade or cross-model fallback.** A profile can intentionally point to an explicit quantized derived artifact; router cannot create/use it without source/quality identity and user authorization. Existing task/memory hard limits cannot be defeated by per-request settings.

## 9. Multi-model coexistence and cache compatibility

- **Shared immutable weights:** a model plus compatible adapter generation may share base source/device weight buffers; different adapters can share base if exact math/activation implementation supports request-specific low-rank contribution and tenant isolation.
- **KV/recurrent/session state:** never shared across unrelated model artifacts/adapter/tokenizer/position/grammar/tenant. Prefix cache hit must pass existing full StateCompatKey and all relevant state-domain frontier—not just same human model alias or text.
- **Compiled bundles:** may reuse code for different model instances if SemanticKey, physical shape/quant, driver/compiler/capture and state layout match, and parameter slots correctly bind weight addresses; never persist raw pointers or leave stale captured graphs after unload.
- **Cross-model routing:** one request's target verifier can't use a draft trained for incompatible model without explicit verifiable compatibility. Encoder–reranker or target–draft pipelines require separate admitted model resources and cost budget, not silent placement.
- **Replicas in remote cluster:** shared prefix/snapshot migration only among authorized peers with exact model binding and a valid state bridge; if no safe transfer, stick to original instance or begin a new session.

## 10. Failure model and human-facing behavior

| Condition | Correct runtime behavior |
|---|---|
| Requested model registered but not loaded | automatic **local** authorized load once, bounded queue/progress and safe memory reservation |
| Requested name unknown | clear `ModelNotFound`, suggestions/list; **no arbitrary remote download** |
| Model available in registry but not downloaded | explain size/license/source; explicit `pull` or interactive consent; offline mode refuses gracefully |
| Artifact recognized but semantic op unsupported | explicit `UnsupportedModelOp`; not a random alternative model |
| Model fits only via optional peer | ask/require configured explicit remote policy; otherwise report placement options, no private data exfiltration |
| Multiple models exceed RAM | unload safely idle/unpinned, evict optional cache, queue or reject; no OS-wide OOM, no unapproved quant change |
| Model loads but compiling preferred fast graph fails | select known valid portable/generic candidate if semantics supported, explain fallback |
| Model already loaded under same source profile | share resident weights/compiled plan; create fresh TaskSession state |
| Alias edited while request streaming | pinned old model/profile revision until end; no mixed tokenizer/KV |
| Model stops/unloads during active request | graceful drain by default; forced cancel only explicit admin and correct last committed frontier |
| Replica/peer dies | failover only with proven compatible StateRoot snapshot/epoch fencing; otherwise typed interruption, never silent new model |
| Cache affinity route worse than fresh load | measured cost selector chooses qualified alternative replica; no network state transfer per token by default |
| User requests unavailable API option | typed `UnsupportedFeature/UnsupportedParameter`; not silently ignored or model-substituted |
| Client disconnect or slow reader | bounded backpressure, cancellation policy and preserved committed output sequence; no leaked provisional tokens |

A `ModeReceipt` reports resolved model source/quant/adapter, effective profile revision, actual replica/deployment, numerical/TaskDriver mode, load/warm/queue time, cache and fallback reasons, physical resources, and optional privacy-safe placement. Default CLI/API output remains concise, with diagnostics only on explicit inspect/explain or authorized headers.

## 11. Security, remote scope and API boundaries

**Router metadata is sensitive:** available models may reveal private research, custom weights/tenants, device inventory or internal peer topology. Native model admin/control API is **not** the same as public inference API, and exposed model IDs are filtered by tenant/role. Localhost default avoids accidental WAN binding; external bind requires auth/TLS strategy and explicit consent. Registry pull and model artifact shipment must respect license, private credentials and dataset/data policy. No model-supplied remote code or tool execution in model router.

Router is not a mandatory **arbitrary upstream cloud provider proxy**. Optional TaskServiceAdapters for approved external model APIs may exist at the application level, with explicit trust/credentials/quality/data egress, but cannot pretend their text-only response is a native zerogravity model `RegionExecutionProvider`.

Avoid an API control-plane call in the hot inference loop. User request body may contain arbitrary tool schemas/JSON/media URLs: tokenize/validate with limits and safe allowlisted fetch policies; executing tools is an independent permissioned workflow above TaskRequest.

## 12. Observability and perf admission proofs

Track at distinct levels:
- **Catalog:** registered aliases, pinned artifact and profile revisions, known tasks/qualifications; no private IDs exposed to unauthorized clients.
- **Residency:** per model replica warm/loaded/idled/loading/evicting, actual weight/program bytes by device, load/init graph time, adapter/drafter memory, pins and eviction attempts.
- **Router:** resolved alias→artifact and replica, queue delay / load wait, cold start/p95, load thrash rate, cache affinity hit/miss, model switch and per-tenant fairness.
- **Execution:** RequestPlacement/StepPlan, batching/graph mode, KV prefix state, per-phase TTFT/decode, real energy/device/IO, task output units, fallback reason.
- **Network:** peer direct/relay, resource reservations, state hydration and transfer bytes/time, retry/epoch failure, forbidden remote policy.
- **Quality:** exact source/quant/LoRA, sampler/grammar, task output/reference; no silent model substitution, tier downgrade or hidden tool invocations.

**Benchmarks:**
1. one-model single process `run` versus `serve` with no extra proxy/per-token dispatch overhead;
2. 1, 2, 4, 8 independent small models resident, with realistic context and physical memory and concurrent requests, C1/C8 fairness/P95;
3. giant model + two tiny models under memory pressure, idle eviction/thrash guard without corrupting active KV;
4. replica routing same exact model with prefix locality and sticky sessions; compare naive round robin, per-replica queues and measured critical path;
5. 100 concurrent requests for cold model deduplicated to one load and bounded waiting; no accidental extra copies;
6. hot alias/profile update mid-stream and concurrent streaming under old revision; no wrong model ID/state;
7. distributed one model sharded across two peers versus two replicas and two source models on same peer; no mistaken replica/cluster accounting;
8. error handling on untrusted or unknown model/quant, no silent fetch or quant downgrade; lost peer, cancellation and worker DMA fences;
9. OpenAI SDK chat/responses/embeddings smoke tests and exact streaming conformance, including UnsupportedFeature/unknown params;
10. model/task-specific parameters allowed/denied by profile policy and resolved provenance `config explain`.

An optimization can earn a default only after real model/quant/hardware tests; first proof is functional accuracy and safe resident scheduling with **no separate inference engine**.

## 13. Minimal implementation staging

**The first model vertical slice stays mandatory.** After a usable CPU/GPU reference:
- add `ModelCatalog` of content-addressed manifests and immutable alias bindings;
- add **one** `ResidentModel` lifecycle + safe on-demand load under existing ResourceArbiter;
- add bounded `ModelRouter` that chooses exact source binding and eligible resident replica/placement, reusing existing TaskCoordinator;
- expose `run` and `serve` with same internal engine, plus `ls/ps/inspect/pull` progressively;
- add equivalent replica routing, hot/idle eviction and per-model settings, then multi-model batch fairness/cold-start policies;
- only then model-choice policy aliases, distributed placement across peers and non-text TaskDriver pools if physical/model tests justify complexity.

**Do not** add a generic service mesh, separate gateway daemon, distributed registry DB, model-specific worker subprocess per model or hardcoded maximum loaded model count as an architectural requirement. A model process isolation mode may be a later security/fault tradeoff, still speaking the same engine worker protocol.

**Closure requirement:** a user can launch one supported model with no config and get a correct response, or expose several models and manage their residency through one API, without ever needing to understand CUDA tiles, RIR/KIR/CIR, sharding epochs or routing internals. Every hidden choice can be examined and overridden safely when explicitly requested.
