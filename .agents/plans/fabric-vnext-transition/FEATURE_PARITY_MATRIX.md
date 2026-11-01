# Fabric vNext — comprehensive current feature and behavior preservation matrix

**Historical source baseline:** HEAD fadf2b1 on dotlink-l5, 27 Oct 2026. The functional rows remain a recovery checklist for F4/Q0/Q2, but current source/tests and CURRENT_SOURCE_RECONCILIATION.md determine today's implementation state. Break/delete pre-v1 implementation and API names freely, but do not lose actually supported model, artifact, device, service or safety behavior. Current stubs/metadata only MUST remain explicitly unsupported until genuinely implemented.

## 31-Oct preservation delta (supersedes historical scope cells)

| Existing accepted scope | Preserve during migration | Still unqualified/deferred |
|---|---|---|
| CUDA0.8B cached47→50 exact fullV/all42 | owned kernels, source/numerical identity, original flight/root/quarantine | broader populated contexts, ordinary production |
| CUDA2B ordinary capacity/lifecycle | actual execution/refusal;2B32K pre-allocation refusal | independent whole-model numerical authority |
| HIP0.8B/2B bounded64 single-row full-model | both natural replays/fullV/all state; canonical source pins; healthy/fault/unknown final readers | multi-row/scalable ordinary/serving |
| HIP27B finite full64/fullV/all112 | teacher/natural/late-cancel/final-reader scope | general contexts/residency/serving |
| CPU/HIP eager attention + chunk GDN | exact per-profile arithmetic, initialized-C/signed-zero, actual HIP retirement/status-only Ready | whole-model multi-row support, Vulkan physical GDN |
| Fresh47–50 and512→514 original references | independent source, explicit phases, full last-rowV/all state | not ZG CPU/HIP execution; no chunk invariance claim |
| Qwen mission | l5/x2 model-specific contexts, C1..C64, matched vLLM and all losses/refusals | deferred behind Fabric, not completed or erased |

Latest source checkpoint and commit-scoped evidence:
[reconciliation](CURRENT_SOURCE_RECONCILIATION.md),
[model execution](../model-execution/TODO.md),
[numerics](../numerics-and-context/TODO.md).
All old feature rows below remain a migration checklist, not an updated physical
capability table. S0 is structural only; F4/Q0/Q2 must account for this delta too.

## Current source-owned preparation reconciliation — `860ce38`

| Area | Actual consumer / evidence | Remaining boundary |
|---|---|---|
| Family recording | Qwen ModelDefinition calls the original recorder; ModelProgram retains real source/plan Arcs; approved sparse-source CPU recording passes | recording is not execution, authentication or registration |
| Provider/prepared owner | typed engine preparation → ResidentModelProvider → original ResidentModel loader; ordinary GGUF catalog consumes it | no generic window-bound/finite/Gemma substitution or partial-region/placement compiler |
| Owner identity | four host controls reject copied plan, different owner even with equal ID, unsupported task/context/output; final reader retains source | host probes have no execution/Ready method and do not qualify GPU |
| Standalone GPU SDK | tokenizer admission text-gated, not cache-gated; CPU-free/cache-free Vulkan compiles; CUDA/HIP strict passes | actual GPU source/state/cancel/fault/serving regression unrun |
| Public API/CPU | facade exports source/record/provider/bundle contracts; real CPU audit17.42s and SDK/catalog/HTTP18.48s pass | CPU keeps its original typed TextSession, not a new interpreter |
| Root/transport/CLI review | existing publishers/worker/cohorts/serving unchanged; transport consumes original engine Request/ResponseEvent; inspected OCI/S3 delegate public provider APIs | whole-feature/root/transport/API/CLI audit remains open; intent is not a publisher or admin endpoint |

Final source/host gate: 2,326 Rust executions/0fail/97ignored/303 suites (70 empty); 17 Python suites/182 cases/10 skips.
RUST_TEST_THREADS=4; earlier50ms HTTP header failure retained. Exact+25 HTTP
controls and the full bounded rerun pass without changed deadlines/assertions.
No GPU/trained-Gemma execution, new capability or retagged physical evidence.
Original traversal/lowering/init/math bodies retained after cold Arc normalization;
failed feature/compile logs remain separate. This does not close F4/Q.

## Previous contract/catalog/serving reconciliation — `83060e5` / `2f19224`

Source/host only; the original feature inventory below is not physical acceptance.

| Area | Actual current consumer / tested scope | Remaining refusal/hold |
|---|---|---|
| Numerics and activation | core canonical contracts; CPU projection/quantized preparation and GPU activation layout; exact source controls pass | no new codec/math/kernel capability; Q4_K dot refuses |
| Tensor layout | checked rows/strides/extents in CPU source, GPU host packing and resident regions; original u32 bound retained | layout never grants source, allocation or Ready |
| Cold catalog and aliases | facade ModelLoadRequest → original configure_text/loaders; real SDK/HTTP alias/source retirement passes | no live replacement, generic compiler or per-model unload |
| Serving policy | SDK TextProfile owns original finite/batch guards; explicit captured-serving feature retained | physical GPU serving unrun |
| CLI serving | public catalog/runtime/server; caller-owned config/signals/IO, cumulative shutdown failures | no alternate scheduler or physical ledger |
| Admin/status | real list/resolve/status; metadata-only AdminOperation intent unchanged | no admin control endpoint, live residency or new capability from status |
| Generic architecture | original ForwardPlan/session/Ready/arbiter remain authoritative | ModelDefinition/provider/executable and remaining root/transport/full parity open |

Final source/host gate: 2,322 Rust executions/0fail/97ignored/302 suites (70 empty); 17
Python suites/182 cases/10 skips.
Approved tiny CPU SDK/HTTP and direct/audit cases pass; trained Gemma and GPU
execution remain unrun. Counts are suite executions, not distinct tests or model
qualification. Prior SDK/artifact acceptance remains applicable on unchanged paths.

## SDK/CLI owner reconciliation — `d4e94b1`

Previous source/host checkpoint; superseded above for catalog and serving policy.
Neither checkpoint is whole F4/Q or physical acceptance.

| Existing operation | Current owner / tested scope | Still open |
|---|---|---|
| CPU direct GGUF/source/cache/audit | SDK `run::cpu`, `prepare::selection`, `diagnostics::cpu_text`; genuine sparse-source CLI aliases and SDK roots/audits/final output credit pass | trained model claims remain source-specific |
| Hub/filter/quant/Xet/components | SDK `artifacts` over original acquisition/publisher; provider/conditional/offline/role/durability/cancel tests pass | no new source acquisition or codec |
| Finite CUDA direct route | SDK diagnostic; original source/native/PCI/admission/feedback/close retained;12 host controls pass |2 physical controls ignored; moved execution not requalified |
| Gemma CPU/HIP candidates | SDK diagnostic over original managed inputs/model/sessions; feature checks and error-combination control pass | new real-source CPU case and physical HIP remain unrun |
| Shared CPU/GPU serving | engine/serving owners from `ae7ec11` unchanged; existing source/host evidence retained | physical GPU transport/lifecycle qualification |
| CLI + external application | CLI IO/parser/signal boundary; no alternate GPU/parser implementation; SDK-only example cold help/missing-source refusal passes | final whole-matrix ownership/parity audit |
| Active architecture contracts | original source/ForwardPlan/session/arbiter remain authority | numerical/layout/preparation/provider/root/catalog/residency reconciliation; no metadata-to-Ready substitution |

Final source/host gate:2,315 Rust executions/0fail/97ignored/302 suites (70 empty),
17 Python suites/182 cases/10 skips. No physical, trained-model, performance or
new model/task/codec capability claim follows from these counts.

## 1. Preservation policy and truth states

Record separately for every feature:
- **Runtime-functional:** supported on an exact model/task/source/quant/backend/shape, with actual output and state semantics.
- **Source/binary qualification only:** selected operator/capture/driver/source fixture has evidence, not general model/runtime authorization.
- **Structural/config validation only:** accepted YAML/media/model metadata, but no numerical execution or service.
- **Planned/not implemented:** typed API may exist, must refuse.
- **Held/unsafe:** explicit user/source/hardware authorization or device safety hold; cannot run to fill a gate.

For each row below: current owner, new owner, intended working behavior, required negative tests and final proof. Qualification state does not automatically improve because a file is moved or Cargo package compiles.

## 2. Direct model and source functionality

| Function | Historic current scope | New owner | F4/Q0 expected behavior |
| --- | --- | --- | --- |
| Qwen3.5 GGUF 0.8B and 2B direct CPU | Native source-exact text, tokenization, prefill/decode, state, known context scopes | Family Qwen + CPU provider + model-neutral engine + public Runtime | Same supported numerical/functional output, prompts, stop/cancel; no HTTP or Qwen-CLI dependency |
| Qwen3.5 source tokenizer/config | Qwen tokenizer + exact GGUF role/shape checks | Family/Qwen + public ModelInspector | Same source roles, chat template, Qwen control token sanitation |
| Qwen original HF BF16 | finite source/reference/diagnostic and CUDA candidate paths | Qwen family + source adapter + qualified provider | Preserve *finite* proof and explicit ordinary Unsupported outside qualified source/shape |
| Qwen3.8 27B | indexed/sharded source, CPU windows and physical HIP source/qualification work | Qwen family + artifacts SourceWeights + CPU/GPU | Same source/oracle/lease tests, not general trained/SSD serving |
| Gemma4 E2B | original-source config/tensors/source, CPU text/reference and finite HIP/Vulkan provider experiments | Gemma family + CPU/GPU provider + engine adapter | Preserve exact PLE/attention/state and source validation/qualified scopes; no universal model support |
| Gemma 26B-A4B /31B | separately coded variant geometry, MoE vs dense source work | Gemma family | Preserve variants and Unsupported where no physical/quality proof |
| Model aliases and loaded Qwen GGUF | CPU serving shared immutable source by canonical path, independent sessions per request | Engine ModelCatalog+Router+Residency | Same source share, private RootBinding, exact alias revision, no duplicate weights |
| Metadata-only model targets | core/targets.rs Qwen Flash, GLM, DeepSeek, Llama, image/media etc. | Facade compiled target catalog with exact statuses | Listing/inspection okay; cannot mint LoadedModel/serve permission |

New execution API must be callable from an external Rust crate importing only zerogravity. Any model not genuinely supported should produce structured Unsupported before allocating model GPU work. The facade cannot advertise future model as available merely because its family/format enum exists.

## 3. Formats, quant and source windows

| Capability | Existing scope | Required outcome |
| --- | --- | --- |
| GGUF tensor metadata, parser and bounded open | implemented artifacts source; strict typed tensor layout | same source-length, shape, type, malformed/unknown/overflow guards |
| Safetensors / indexed shards | implemented verified owned source/index and bounded tiny test fixtures | exact shard digest/source owner identity, no incomplete tensor permit |
| F32/F16/BF16 codec | exact source read scalar conversion; BF16 numerical profiles separate | source arithmetic unchanged, correct nonfinite/subnormal/negative zero |
| Q4_0 | decoder, source Q4_0xQ8_0 CPU/repack, selected GPU kernel | same exact blocks, scales, activation and reduction; backend gated |
| Q4_1 | decoder, source Q4_1xQ8_1 CPU and GPU implementation body | preserve signed offset and arithmetic bitwise/reference rules |
| Q6_K | decoder, source Q6_KxQ8_K CPU and selected GPU | preserve signed activation, reduction and original profile |
| Q8_0 | decoder, CPU Q8_0xQ8_0, GPU source body | same bytes/row geometry, source F32 separate |
| Q4_K | scalar decoder but deliberately Unsupported source-exact CPU dot | decoder still works; CPU/model run still refuses unless separately qualified later |
| Q1/Q2/Q3/NVFP4 | format directory readmes only | metadata status remains planned/no real codec; no generic false output |
| EXL3/QTIP/trellis | planning only | derived-source contract reserved, no automatic quantizer or performance claims |
| Activation packing | GGUF dynamic Q8 variants for particular stored encodings | source stored bytes vs compute precision and effective dynamic Q8 stay separate |
| Large immutable SourceWeights | verified_shards, owned source, cache/window, resource admission | source owner and tensor lease survive all consumers; no fake SSD streaming |
| Cache transactional model store | original cache/acquisition/pull features, file-safety and errors | same atomic publication/locking/resume/immutability with exposed public SDK |

Each codec must have exact separate recognition, scalar decode, source operator, native provider and whole-model capability fields. No one bool `supports_quantization` for everything.

## 4. Artifact acquisition and model management

The existing CLI pull/provider implementations and artifacts lower-level acquisition already cover the following categories, with separate permissions, pinned versions and bounded storage contracts:
- Hugging Face basic and indexed safetensors/pinned model selection, sidecars and cache.
- GitHub source trees/release content as applicable.
- Hugging Face Xet transport and explicit cache.
- ModelScope.
- AWS S3 path/bucket/revision/conditional source reads (correct SigV4 and endpoint/proxy policies).
- GCS and Azure object/prefix paths with exact conditional etags/revisions.
- OCI/container image source fetch or manifest selection.
- Generic HTTP(S) object transfer, resume/etag/range and cache views.
- Source GGUF quant basename/source selection (selection != conversion), candidate/filter/component bundle discovery, report/progress and typed unsupported/malformed paths.
- Cache import/manage/run/current inventory, source hashes, transactional snapshot store.

**New owner:** artifacts/src/acquisition/cache/transport for real source work; facade public `ModelStore`/`Acquisition`/`Cache` API for model/source selection, progress/status and callbacks; CLI only input/output. No forced network access in library constructor or inspect. Unsupported transport or unavailable source returns exact refusal, no fallback to untrusted Python or alternate source.

**Tests:** existing CLI pull_*.rs, cache_* integration fixtures should be moved/reused as Rust library tests and small CLI delegate parity tests. Do not drop subtle path traversal/symlink/malformed shard/conditional race test to make CLI tiny. A standalone Rust program must be able to perform the same source selection/pull/cache request without spawning CLI. Use controlled loopback fixture rather than real downloads for cheap phases; full acquisition/isolated loopback suite Q0.

## 5. Device inventory, fit and run configuration

- Linux GPU device selectors (including physical PCI identity, /dev validated mappings and rejected unrelated character/control nodes), local ordered allowlists and CPU fallback rules.
- Separate device identity, vendor backend (CPU, CUDA, HIP, Vulkan), actual physical device and stage role. Do not infer split/run from a matching name or /dev path alone.
- Qualified source task+device+shape and available real physical fit, host/UMA and cgroup headroom; GTT virtual range not free DRAM.
- Model/backing/host+GPU+compiler+scratch+state+output+IO complete peak budgeting, one process-shared ledger; multi-domain future reservations all-or-none before physical work.
- Config precedence, bounded YAML/JSON parser and CLI flags, source/profile/headroom/diagnostics, dry-run/explain and full errors.
- Per-model source/quant/numeric and effective qualified GPU strategy selection, no hidden model change under pressure.
- Public `Runtime::devices`/`RunPlan`/`explain` equivalent plus CLI thin pretty table and argv parser.

Use existing CLI-device-routing plan as exact behavior; no second device selection engine in CLI or server. Preserve disallowed gfx90c device launch and dangerous/unknown reset holds. Code can inspect device capability without launching unsafe GPU/remote operations, but such inspection is not acceptance evidence.

## 6. Serving and batching behavior

| Behavior | Existing accepted subset | New owner / proof |
| --- | --- | --- |
| CPU Qwen GGUF OpenAI HTTP/1.1 | loopback server, GET health/v1/models, POST completions/chat completions | serving lib HTTP and engine Qwen GGUF CPU; preserve request/response errors |
| Greedy sampler API | temp0/top_p1/n1/zero penalties, bounded output, strict unsupported params | source model/engine sampling and serving request validation, no silent ignore |
| JSON/SSE | streaming responses, finish_reason, usage, UTF-8/EOS, selected request metrics | server protocol emitter consumes engine committed output; different delivery clock |
| CPU concurrency | finite workers, active/queued, fair prefill/decode quanta, shared weights and private roots | model-neutral engine scheduler, same limits/behavior |
| GPU Qwen GGUF | one original resident source, independent GPU sessions, explicit device completion | GPU provider + engine coordinator |
| GPU native fused B2/B4/B8 | opt-in qualified same-source decode cohorts with differing frontiers, ragged/unknown fallback B1 | engine work formation using prepared native kernels; no fake multiple B1 work |
| Resource/host credit | bounded connections/request body/state/queued/scratch/credits | serving ingress plus engine physical one arbiter |
| HTTP cancellation | disconnect cancels future worker quanta, preserves submitted GPU backing until Ready/retire | server cancel scope, engine session epoch and physical worker |
| Slow-client backpressure | bounded response output packet, parking, timeouts and survivor handling | server transport credits + engine CommittedEvent stream |
| Server shutdown | recent 50abaf7 worker join and owned provider closure/stop evidence | serving ServerHandle, engine pending admission drain, no device leak |
| Model aliases | canonical shared source and source-bound CPU/GPU request root | engine alias generation and exact model pin |
| Unsupported API scopes | tool calls, grammar, embeddings, Responses, media etc. are not blanket implemented | separate typed server endpoint capability, Unsupported/refuse |
| vLLM goodput parity | not demonstrated by current finite native source cases | not part of transition support claim; later matched benchmark |

No hidden per-model HTTP server, no CPU-path token scheduler copied into serving crate, no independent Socket→StateRoot publisher. HTTP API through public zerogravity::serving and direct Rust run must share engine and model residency.

## 7. Pipeline and multimodal compatibility

- CLI pipeline v1/v2 YAML/flags typed schema, duplicate/unknown/size/depth/symlink/path refusal and type/port graph/DAG checker: already functional *structural* validation only. Move parser and validator to public SDK; CLI display only.
- Source-neutral builtin component kinds (text, asr, diarization, tts, image, decision) and typed ports/clock descriptors: preserve validation/registry, not imply model execution.
- Pipeline run remains deliberate NOT IMPLEMENTED until later real TaskDriver/provider/task-output acceptance. Safe typed library return and CLI error.
- Immutable chat-context grants and source/authorization/freshness typed snapshot views preserve data privacy and permission semantics, even though realtime model execution is future.
- Native actual media (ASR, TTS, image, video), external hosted TaskServiceAdapters, multi-peer state transfer are not current general functional features; skeleton has typed architecture, not a successful fake output.
- A ModelAssembly is not a cross-model TaskPipeline, and compositional pipelines cannot spawn new HTTP inference servers or silently merge different users' StateRoots.
- Prepared/Committed/Delivered/Ack/Played events distinct even in future transport.

## 8. Numerical, state and physical invariants to preserve

- Original task/model/quant/source/binary identity, independent reference profiles, no model or backend width grant inheritance.
- Source GGUF and original HF BF16 arithmetic are different contracts. Pure dequantized F32 oracle is distinct from original GGUF activation coding and per-lane rounding.
- Source-independent region replacement only when all state/output effects and numeric/shape guards pass.
- State root publication checks actual old root, epoch, session/source and all required prepared state domains; no foreign/stale root or two session publishers.
- First-use physical event Ready is *real* driver completion, not submission, Graph capture, worker queue dequeue or diagnostic output.
- ResourceArbiter one process shared physical budget with source ownership, actual backing aliases, preflight peaks, encoder captured readers and unknown/quarantine retention.
- Cancellation/exception handles do not free in-flight DMA reader or reset shared physical GPU for one request.
- Strict unsupported/invalid/limits and source mutation/refusal ordering remains. No false recovery by retrying an unknown submitted GPU operation on another provider.
- Signed historical source/model/backend proof stays attached to exact code identity. Renaming executable/source invalidates blanket transfer of physical acceptance until reconciled.

## 9. Final F4/Q0/Q2 parity table requirements

At F4, mark each row from ``2–8 as:
- retained and rehomed with real implementation reference;
- tested with focused source-free/loopback fixture and exact command;
- held for consolidated Q0 expensive qualification;
- not implemented originally (continue typed Unsupported);
- deliberately removed ONLY if it was a pre-v1 compatibility/API duplicate, not a previously supported behavior.

At Q0 run one full assembled check with source/codec/CPU/GPU/serving+SDK/CLI, preserving ignored/failed/unavailable cases. At Q1 fix at lowest appropriate layer and with small tests. At Q2 final accepted complete gate. No claim "all features working" unless every *originally supported* behavior from this matrix and its original gates is demonstrably present, while future-only cells remain clearly unsupported.

## 10. External library and release-facing coverage

An external Rust application with a **single public zerogravity dependency** can:
- inspect, load and run supported model/quant source;
- use original bounded acquisition/cache/pull/verify and source-manifest methods;
- list model capabilities and device/fit/explain;
- submit concurrent direct text TaskRequests, cancel and observe committed output;
- validate a pipeline YAML/typed graph, get Unsupported for unqualified execution;
- attach and start an independent HTTP OpenAI server via zerogravity::serving and stop it cleanly;
- share immutable model source while maintaining independent live sessions with the same low-level resource authority.

The CLI must be a *consumer of these same functions*, with no separate source importer, model-specific server, quant registry, TaskCoordinator or native kernel selection.

**Transition completion is not a v1 release.** Future model/media/SSD/distributed features need their own gates after architectural acceptance. The user expressly allows Rust name/signature breaking and requires no old compatibility shims at final state.
