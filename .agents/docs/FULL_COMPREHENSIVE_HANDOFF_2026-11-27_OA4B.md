# ZeroGravity — full comprehensive handoff through OA-4B

Date: 27 Nov 2026 (Singapore). Repository: `/home/pvl/spaces/abird/src/zerogravity`.
Implementation checkpoint: `89822f4038b2048affd8b92dac6c7b2bcbeb219e` — bounded joined background Responses and cancellation.
Previous comprehensive takeover: `3df319127b19616dd3e9d4643d63c7ef9cd1a448`, [26 Nov UX-P-M1 pack](FULL_COMPREHENSIVE_HANDOFF_2026-11-26_UX_PM1.md).
Accepted UX-1–10 exit: `1b8de1cb87fa4815655cbb6700e6633946f93305`.
Accepted bounded Phase 1 exit: `d5030ad4ce476b913a46056d492992ccf1b8edf7`.

**Read §§0–13 first; resume at §11, not the historical prompts.** This requested
handoff extends the previous complete pack; it is not a second live task ledger
or new pinning requirement. Live source, executed evidence, AGENTS.md, ordered
full plan and owner TODOs remain authoritative. Original docs and historical
packs remain intact, including original design decisions and rejected approaches.

- Appendix A: complete tracked engineering-document inventory.
- Appendix B: unresolved lexical checklist snapshot (including original spec
  markers), with original files and line numbers for full context.
- Appendix C: complete reachable Git ledger and exact new delta.
- Appendix D: every post-M1 landing: original commit message, exact changed
  paths, contemporaneous checkpoint narrative and plan/doc deltas. Also the
  reconciled live OA-4B record and baseline.
- Appendix E: complete previous UX-P-M1 main body verbatim (historical) and
  linked earlier handoff chain. Earlier packs preserve their full appendices.

## 0. Recovered implementation and current position

The interrupted OA-4B turn already finished. Recovery independently inspected
HEAD, signature presence, clean index/tree, single worktree, durable job list,
actual background/store/supervision code, focused and real-consumer logs,
and final assembled gate marker and counts. No source change or unnecessary
repeat test was required. Do not redo OA-4B.

- Latest signed implementation: 89822f4038b2048affd8b92dac6c7b2bcbeb219e.
- Focused engine/serving: 172 passed / 0 failed / 0 ignored, strict lint.
- Full isolated assembled gate: 2,562 / 0 / 109 across 318 Rust suites
  (377.3 s as recorded at landing). GPU feature-boundary compile gate
  explicitly passed; portable release, TUI/Web, installer and scripts passed.
- Real cached Qwen3.5 CLI/TUI/Firefox/official OpenAI SDK 2.33.0 canary:
  1/0/0 (130.88 s), actual cancelled background request, polling, exact
  Chat parity, continuation, deletion and original model/listener lifecycle.
  Rust SDK 1/0/0 (14.36 s), five filtered in each respective integration binary.
- Read `target/oa4b-focused.log`, `target/oa4b-real-consumers.log`,
  `target/oa4b-assembled.log` if present. Prior check log has an intermediate
  unused import warning, superseded by final strict lint. Local logs are
  run-local, not tracked; commit and owning plans preserve accepted scope.
- Bounded Phase 1, UX-1–10, UX-P4B, model-local execution selector, OA-1,
  OA-2, OA-3, OA-4A and OA-4B accepted. OA-4 remains open.
- No new physical GPU/throughput qualification in serving work; do not
  compare CLI/SDK lifecycle durations to vLLM. Exact Nix source-integrity
  still unverified for unavailable locked inputs (§10.2).

### 0.1 Repository and resource authority

Observed before this documentation-only handoff: L5 main clean, tracking
origin/main `3156ce45750e838c7ace72dc7d8c0c06aa762c3b`, ahead 7 / behind 0
(one local main checkout), no live listed jobs or compile owners. The
implementation carries an SSH signature; signer-principal verification
requires the configured trust material (do not disable signing).
No remote fetch/push performed. Do not infer the remote server's live
state from a local tracking ref.

Last observed L5 RTX 3060 Laptop GPU 6,144 MiB total, 37 MiB used,
5,789 MiB free per nvidia-smi, no listed compute owners; 27 GiB host
RAM, about 21 GiB available, 52 GiB disk free (95% used), swap in use.
These are momentary observations, not fit guarantees. Do not prune
unrelated caches, target variants, processes or devices. Build concurrency
reduced to 4 jobs per lane during OA-4B; use actual available resources.
X2/A1 not revalidated here; execution/testing only. L5 is sole source
authority via dotlink---l5. Exact incremental Git bundles for testing only.
No persistent alternate checkout, destructive Git, GPU reset, unapproved
downloads or privilege changes.

## 1. Bootstrap sequence

1. [AGENTS.md](../../AGENTS.md).
2. [Plan rulebook](../plans/README.md).
3. This handoff §§0–13; Appendix D for exact new landings and Appendix E
   for the original design/history.
4. [Full ordered implementation plan](../plans/FULL_IMPLEMENTATION_PLAN.md).
5. [CURRENT_HANDOFF](CURRENT_HANDOFF.md) and [CURRENT_BASELINE](CURRENT_BASELINE.md).
6. [Shared queue](../plans/TODO.md), [serving design](../plans/serving-and-speculation/NEXT_ITERATION_OPENAI_COMPATIBLE_SERVING.md)
   (especially §§6–10, 17–25, 30–37), [serving TODO](../plans/serving-and-speculation/TODO.md).
7. [SDK/serving](../plans/fabric-vnext/PUBLIC_LIB_AND_SERVING.md),
   [structure/naming](../plans/pipeline-composition/STRUCTURE_AND_NAMING.md)
   and resource, state and scheduler specs referenced in §12.
8. Read [CLI lifecycle](../plans/cli-ux/NEXT_ITERATION_CLI_UX_AND_MODEL_LIFECYCLE.md),
   [polish](../plans/cli-ux/TUI_POLISH_AND_MOUSE_INTERACTION.md),
   [live performance](../plans/cli-ux/LIVE_MODEL_PERFORMANCE.md),
   [CLI/device routing](../plans/cli-device-routing/README.md),
   [acquisition](../plans/model-acquisition/README.md) and
   [platform TODO](../plans/platform-and-release/TODO.md) when touching their
   owners. Earlier P4B next steps are closed; M2 deeper inspection optional.
9. Inspect actual source and tests (§5); continue §11. For performance,
   read owning model/numerics/backend evidence before experiments.

[Standalone next-agent prompt](NEXT_AGENT_2026-11-27_OA4C.md) is reproduced
in §13. OA-4C in its filename is a navigation label for the conversation
slice, not a new independent architecture or acceptance bureaucracy.

## 2. Owner direction and working contract

- Elegant, simple, clear and intuitive. ZeroGravity is a composable Rust
  library/execution fabric, not a single-model benchmark or per-backend
  independent engine. Favor zero-cost abstractions, direct hot paths and
  performance where it remains clean and maintainable. Where high
  performance requires convoluted maintenance, choose clean/simple first
  and target the measured hot area selectively.
- Reuse mature optimized mathematics and actual shared operations. Attack
  large architectural/computational bottlenecks before marginal launch
  counts. Model-specific specializations must consume the authoritative
  model semantics and justify their own benefit.
- Preserve useful correct model-neutral kernels, layouts, transports,
  state/scheduling transforms, providers and completion primitives when
  a consumer/selector loses. Remove temporary probes, losing consumer
  wiring, runners and test-only choices, not reusable implementations.
  Delete a primitive only when incorrect/unsafe, incompatible with its
  contract or strictly superseded; record why.
- Preserve exact source, numerical, layout, state, physical resource,
  ownership, cancellation and completion semantics. Intentional changed
  arithmetic is a separate explicit numerical profile, not a hidden
  approximation. No retry against an uncertain submitted root.
- No overly granular test/evidence/pinning process. Batch related work,
  reuse applicable accepted evidence, validate actual changed consumers
  and meaningful invariants. Full assembled gate at phase exit; synthetic
  kernel speed, compilation and UI lifecycle time are not performance
  parity. Real-source and physical qualification still mandatory where
  the relevant execution path changes.
- One substantial coherent validated signed checkpoint per user turn,
  aim approximately 30 minutes at most; pause with exact state, blockers
  and next unit. Recover interrupted work from live source and executed
  results. Keep active jobs owned; no indefinite background promise.
- Sole L5 source authority, exact incremental bundles, no persistent
  additional worktrees. Coordinator is single writer of shared queue.
  User-authored root README unchanged; architecture in README describes
  final intended engineering system while status belongs in plans.
- Latest product direction: calm, beautiful and minimal by default,
  discoverable detail on demand, not maximal control visibility. Full
  mouse plus keyboard; **no permanent action toolbar** at any width.
  One navigation rail, inline live local search, exact selected-object
  context actions, separate small session menu. Smart human size units.
  Dark/Light/Auto now; additional theme catalog later only if requested.
  Running is the live model dashboard, not a separate monitoring app.
- Full sane Cargo CLI default includes all current production models,
  CPU/CUDA/ROCm/Vulkan, acquisition, serving, TUI/Web while breadth is
  small. Explicit minimal/portable and embedding profiles remain
  available. Do not confuse a compiled backend with qualified execution.
  Error categories and chains must be accurate; safe HTTP responses and
  opt-in trusted diagnostic details, with bounded safe structured logs.

## 3. Original design and architecture to preserve

The [October 10 master plan](../plans/ZEROGRAVITY_MASTER_PLAN_2026-10-10.md),
[vision](VISION_SCOPE_AND_PRINCIPLES.md), [modular reference architecture](MODULAR_AI_SYSTEMS_REFERENCE_ARCHITECTURE.md),
original engineering specifications and critical review define the design
from the beginning. First reachable Git commit is October 13
be95424837deb7919adf34503db71d27d3ad3adb. Appendix E preserves the detailed
previous architecture explanation verbatim; earlier Phase 1 packs preserve
the original design evolution and decisions. This summary does not replace
their source/numerical/lifetime contracts.

### 3.1 One fabric; four distinct concerns

![Execution architecture](../../docs/architecture-overview.svg)

| Concern | Authority |
| --- | --- |
| Fabric/runtime | Model lifecycle, original catalog/registration, admission, placement, scheduling, batching, pipelines, persistent state, publication and serving |
| Model semantics/planning | Verified source-bound computation and task, learned roles/axes, typed shapes/regions/state effects, numerical/quality profile |
| Implementations | Native composed ops, mature vendor math, specialized kernels, captured commands or compiled closed regions/whole programs |
| Backends | CPU, CUDA, HIP/ROCm, Vulkan and future devices; physical memory, command submission, readiness and completion |

Do not conflate operator libraries, graph compilers, artifact formats and
physical backends. Prefer actual mature math (cuBLAS/Lt, CUTLASS, cuDNN,
FlashAttention/FlashInfer where qualified; rocBLAS, hipBLASLt, CK, MIOpen
and suitable HIP primitives; qualified Vulkan cooperative-matrix/shader
implementations and existing CPU optimized/reference operations).
Optional concrete MIGraphX/TensorRT/IREE/ONNX Runtime integrations are
later Phase 6, not dependencies of native CUDA/ROCm/Vulkan. No new
provider is granted by recognizing a format or compiling a library.

### 3.2 Source, semantics and executable lowering

Original HF configuration/tokenizer/processor/indexed Safetensors and
GGUF metadata/codecs/weight tensors are primary. Transformers describes
an architecture/software ecosystem; Hugging Face is source/distribution,
not a storage format. Verify original identities/roles/axes/quantization;
no implicit conversion, remote model code, mandatory Python or mandatory
ONNX conversion. Source eligibility independent of device execution.

Two enduring execution representations: (1) authoritative semantic
ModelProgram/ForwardPlan, mathematical/task regions and state effects;
(2) prepared physical executable and original source/resource owners.
Artifact input and provider-local compiler graphs are not alternative
canonical IRs. Native ModelAssembly/regions and multi-model TaskPipeline
are distinct concerns; compositional descriptions do not create a second
runtime. Select a provider during cold preparation or qualified
committed-root switch. Pre-enqueue miss may take already-qualified
decomposition or refuse; submitted uncertainty cannot be retried
against the same root by silently changing provider.

Direct typed-region lowering is independent of external graph import.
MIGraphX/TensorRT preferably construct from verified regions through
native APIs. Optional ONNX import/ORT needs authenticated graph/weights,
layout, precision, state, effects and resource contracts; IREE suitable
lowered computation. SDK-free scaffold accepted only for bounded seam.

### 3.3 Non-negotiable state and physical ownership

One logical StateRoot publisher per session. Device work creates private
candidates and readiness, not logical commits. All required KV/recurrent/
conv state domains publish a compatible frontier before visible output.
Cancellation/failure cannot leak provisional output/state; uncertain
completion fences/quarantines the original physical owners until actually
safe. Immutable source/backing/range/lease provenance remains authority;
aliases/readers retain real owners; independent mutable session state is
private. Complete-domain compatibility for prefix/cache/COW/page reuse.

One physical ResourceArbiter. Account persistent/context/workspace/
transfer and synchronization costs; explicit headroom, real UMA overlap.
No implicit reclamation or hidden memory grant. Backend adapters own
submission and completion; shared/external views conditional on actual
capability. No new allocator, model registry, scheduler or execution IR.
Separate reference arithmetic/native profile, quality, source compatibility,
physical readiness, feature compilation and serving/performance acceptance.
One fair worker/cohort/scheduler with shared prefill, independent
continuations, paged/hybrid state, feedback, MTP, placement and task
pipeline contracts.

Read [invariants](INVARIANTS_AND_FAILURE_SEMANTICS.md),
[state](STATE_CACHE_AND_TRANSACTIONS.md), [resource/commands](ASYNC_RESOURCE_COMMAND_IR.md),
[execution spine](MINIMAL_EXECUTION_SPINE_AND_PROTOCOL_V4.md),
[regions](../plans/pipeline-composition/REGIONS_AND_PROFILES.md),
[correctness policy](../plans/numerics-and-context/CORRECTNESS_POLICY.md).
The previous comprehensive handoffs preserve all accepted foundation
slices and exact provider/capture/persistence decisions.

### 3.4 Current typed failure contract (supersedes historical coarse translation)

| Core category | Meaning |
| --- | --- |
| Malformed | Actual source/artifact validation failure |
| InvalidInput | Invalid request/configuration/input rather than artifact corruption |
| Unsupported | Backend, device, route or capability unavailable |
| Limit | Configured/static bounds, arithmetic overflow, quotas |
| Resource(ResourceLimit) | Actual measured admission refusal; operation/resource/requested and available bytes/scope |
| State | Invalid lifecycle/internal state |
| Io | Original underlying cause |

In-process ServiceError keeps typed origin through Arc on error path,
including shared conversions, resource amounts and I/O causes. For
synthetic service errors use stable code, then status fallback; never
invent a budget or infer artifact corruption from HTTP 400. The previous
A1 message "malformed model artifact" for missing compiled backend was
misclassification. Original-source Qwen needs eligible CUDA or HIP,
not default CPU/Vulkan; SM89 is not automatically qualified by compiling
CUDA. Preserve actual source and device policy, not merely nicer errors.
No string parsing as admission/fallback policy.

HTTP and SSE preserve status/code/type with fixed safe public message
by default. Optional error.details adds status, code, category, measured
bytes and OS code where available, plus bounded diagnostic text/causes
only after explicit trusted listener setting and valid bearer auth.
Pre-auth/framing/connection refusal never exposes diagnostic text.
The existing bounded control event ring/logs carries safe FailureDetails
tagged by listener generation; arbitrary raw paths/credentials/prompts
not stored in the shared operational ring. Full native diagnostic retained.
See §5 and the previous UX-P-M1 pack Appendix D.14 for the original
diagnostics landing. The native diagnostic setting is not browser
safe-editable. Future public serving must not expose trusted debug text
to ordinary clients.

## 4. Ordered delivery and pending boundaries

| Phase | State / objective |
| --- | --- |
| 1 Foundations | Bounded common exit accepted d5030ad4; broader generalizations open |
| 2 CLI/TUI/Web and public serving | UX-1–10, UX-P and execution chooser accepted; OA-1–3 and OA-4A/B accepted; finish remaining OA-4, then OA-5–10 applicable capability work and ordinary benchmarks |
| 3 ROCm | Match or beat optimized matched vLLM first, AMD |
| 4 CUDA | Match or beat optimized matched vLLM next, NVIDIA |
| 5 Vulkan | Native correctness/performance parity with actual suitable comparators |
| 6 Optional compiled providers | MIGraphX, TensorRT, IREE, ONNX Runtime with demonstrated benefit |

Use the live ordered plan; old priorities in historical handoffs are superseded.
Preserve all pending foundational generalizations and original source/model
qualification. Qwen3.5 0.8B/2B and Qwen3.8 27B first performance family;
Qwen3.5+/3.8, Gemma4, Qwen Flash/Next, DeepSeek and other model families
through actual model contracts. Media tasks/ASR/TTS/image models remain
capability-gated. No model download or execution support inferred from catalog.

### 4.1 Bounded accepted foundation versus remaining generalization

Phase 1 minimum common seams: source-bound semantic and prepared
composition, native operations/typed regions and optional provider scaffold,
authentic HF/GGUF/SDK source lifecycle, original physical/state/control
ownership, paged/streamed/prefix/persistence/cohort, typed tasks and
bounded pipelines, actual SDK and real CLI consumers. Accepted scope,
exact post-matrix 49-commit history and physical tests are preserved in
[Phase 1 exit pack](FULL_COMPREHENSIVE_HANDOFF_2026-11-25_PHASE1_EXIT.md)
§§0–13 and Appendices C–E. General foreign/Existing buffer or context
interop, device-write-to-host-read/coherency, multi-domain and UMA overlap
generalization, non-Linux physical fit/OS-hard reservations, general
multi-source/mixed-backend HF, general DAG/media/context, arbitrary state
migration/persistence, authenticated tenant stores, async SSD and broader
model/codec/profile/device qualifications remain open. Do not infer
general support from a bounded coherent mapped producer or actual
single-device preparation. Preserve all original hold conditions.

### 4.2 Public serving status and remaining order

| Unit | Status |
| --- | --- |
| OA-1 | Async HTTP, strict bounded chunked JSON, multipart and owned WebSocket primitives accepted |
| OA-2 | Shared typed text request/events and model-owned template accepted |
| OA-3 | Actual Models retrieval and stateless Responses JSON/SSE accepted |
| OA-4A | Foreground storage/retrieval/delete, paginated input items, previous-response continuation accepted |
| OA-4B | Joined nonstream background create/poll/cancel accepted |
| OA-4 remaining | Conversations/items CRUD and atomic append, exact input-token count, bounded background streaming/replay/resumption |
| OA-5 | Chat/legacy Completions compatibility close on common semantic path; HTTP ChatML already removed |
| OA-6 | Embeddings only on real embedding runtime/model capability |
| OA-7 | Audio transcription/translation/speech on real ASR/TTS tasks |
| OA-8 | Images generation/edit/variation on real image execution |
| OA-9 | Realtime WebSocket session/pipeline semantics; no fake model support |
| OA-10 | Compatibility close, SDK and endpoint matrix, bounded shutdown/objects/help/docs |

Public API is not a cloud billing/account API clone. Keep independent control
and inference authentication, actual route/tenant capability and original
execution owners. Local loopback canaries are not general exposure,
multi-tenant isolation or complete hosted OpenAI parity. Do not block
ordinary benchmark/performance work on imaginary media implementations;
follow the live ordering and explicitly capability-blocked scopes.
Normal CLI/SDK/server benchmark acceptance is still required before native
backend parity campaigns. The live serving spec §§29–36 owns precise
remaining endpoint and acceptance scope.

## 5. Current source map and implemented contracts

All paths below are repository-relative. Foundation source map in the
previous Phase 1 pack §5 and Appendix E §5. Inspect live file names and
feature guards rather than extrapolate old examples.

| Area | Primary live owners |
| --- | --- |
| Core / source / operators | crates/zerogravity-core, zerogravity-artifacts, zerogravity-cpu, zerogravity-model-qwen35, zerogravity-model-gemma4, zerogravity-gpu, vendored CubeCL |
| Engine | crates/zerogravity-engine/src/{runtime,diagnostics,router}.rs; router/{registry,worker,pending,event_channel,dispatch,error,performance}.rs and actual scheduler/state owners |
| Facade and control | crates/zerogravity/src/{runtime,models,configuration,control,run}.rs; control/{cache,downloads,events,serving}.rs; prepare/, devices/, builtins/text/ |
| Configuration | facade pipelines/{document,yaml}.rs, configuration/file.rs; CLI config_cli.rs, run_cli/config.rs; one typed v2 Document/ConfigFile |
| CLI | crates/zerogravity-cli/src/{main,args,commands,completion,output,model_cli,resident,run_cli,pull}.rs; output/presentation.rs, run_cli/, serve/ |
| Shared host | CLI control_host.rs, control_host/{actions,inventory,logs,settings,work}.rs |
| Native TUI | CLI tui.rs, tui/{view,layout,input,controls,editor,choice,theme,format,performance,library,logs,settings,terminal,work}.rs |
| Embedded control | CLI web.rs, web/{actions,http,settings,wire}.rs; web/src/{main.jsx,settings.jsx,performance.jsx,performance.js,api.js,browser-smoke.mjs,styles.css}, committed dist |
| Inference server | crates/zerogravity-serving/src/{server,http,auth,service}.rs; original text service/events and http/{body,transport,ingress_body,multipart,websocket,responses}.rs; responses/{input,store,background}.rs (not browser control transport) |
| Actual tests | crates/zerogravity-cli/tests/{cli,config_cli,pipeline_config,run_unified}.rs, tests/fixtures/{tui_shell,web_shell,serve_shutdown}.py; engine/router tests, facade tests/fabric_sdk.rs; serving/server tests and artifact/cache fixtures |
| Build and qualification | scripts/check.sh, scripts/check-gpu-feature-boundaries.sh, scripts/native-gpu.py, tools/, nix/workspace.nix, release/installer scripts |

### 5.1 Shared native/Web control and identity

One Runtime, one facade ControlPlane, one process-local Selection and
one shared Actions slot (not a queue). Resident planner executes original
load/invoke/run/unload/listener and transfer owners; browser and native
observe same state. Exact digest/source, registration, listener generation,
transfer operation and action IDs are distinct. Aliases do not create
new owners. Confirmation/request revalidates observed key and resource;
no stale retarget. Selection, pointer motion, filter, settings preview,
view navigation never execute. Cancellation request is not completion.
Bounded output receipt and original terminal retirement/join remain
authoritative. Native-private draft/output separate from Web receipt.
One typed v2 ConfigDocument/ConfigFile, default/named/explicit selection
and CLI precedence, future-only activation, atomic exact compare and
raw repair, no independent store or arbitrary browser file/execution
authority. Existing safe operational log ring, no persistent prompt
history. Bounded original events and SSE resync, no parallel history.

### 5.2 Embedded Web control security

Embedded offline-built SolidJS/Vite assets, independent CLI Web feature
(no terminal dependencies). Numeric loopback, OS random 32-byte control
credential in private URL fragment, consumed/scrubbed and tab-scoped;
lock clears it. Independent inference bearer. Authenticated snapshot,
events and exact typed actions/receipts; bounded connections/headers/
JSON/SSE, strict numeric Host/origin and same-origin rules, no CORS or
credential/path/argv authority. See current wire/actions/http/settings
source and prior UX-9A/B/C/D/E acceptance. Only supported approved
cache/pinned source/recipe and typed actions; explicit consent and offline
policy, no implicit acquisition. Web and TUI can coexist, share actions,
read-only observations, original transfer and config revisions. Local
foreground host retires clients/actions/listener/runtime in owned order.

### 5.3 Latest native UX (supersedes all toolbar designs)

Whisrs v0.2.1 inspired semantic color depth and restrained palette.
Dark/Light/Auto; Auto terminal hint and honest dark fallback, active
probe deferred. Scoped mouse capture and ui.mouse auto/always/never;
F8 session override and terminal restoration, no clipboard bridge.
Actual render-derived hit regions, matching press/release and modal
masking. Per-view focus, rail/split drag, independent scrolling/scrollbars,
maximize/restore, compact collapse/navigation. Keyboard equivalents
remain, Ctrl+arrows split, Alt+arrows rail, F6 panes, F7 size.

User screenshots rejected visual overload, then rejected even the
curated top toolbar. **No toolbar**. Search-only top line in Models,
Devices, Profiles, Config, Running and Logs; ordinary printable input
(including former shortcut letters/digits/punctuation) focuses/filter
locally as typed. Ctrl-F/U, Esc clear, Tab leave, arrows select, filters
per shelf. Alt+1–9 views, explicit Alt operation letters, F1 help,
F4 context, F10 session, Ctrl-Q quit, original modal keys. Non-search
views retain ordinary shortcuts. Row selection inert; selected ellipsis,
matching right-click, Enter model/device/registration context; profile
Enter editor; Config Enter YAML value, typed picker via menu/Alt-e,
F2 YAML fallback. Session menu Web/Appearance/Refresh/Quit. Context
only applicable actions (cached load/run/inspect/verify/alias/removal;
catalog/partial pull/details). Offscreen/collapsed/no-row pane fallback.
No repeated nav/shelf/theme controls, debug slogans or shortcut wall.

Smart IEC B/KiB/MiB/GiB/... with at most two decimals (exact bytes kept
in details and machine data). Concise model name/size/source/verification
status; exact digest/revision/raw size and diagnostics only on demand.
Identity anchors per shelf/list/ready owner/log sequence and response
byte position survive reorder, empty refresh and viewport clamp. F3
distinct Running details/response viewport, F9 follow; response header
registration first, truncate long aliases. Logs safe selected entry
details; retention eviction refuses rather than retarget. Full short
terminal dialogs with scrolling, actual 20×7 tested. Grapheme-aware
bounded drafts, caret, mouse/Shift selection, drag edge by real event,
safe raw YAML and paste, no click-through/implicit submit.

### 5.4 Full production defaults and diagnostics

CLI Cargo default is ["all"], existing production all-models/all-backends,
CPU/CUDA/HIP/Vulkan, Qwen/Gemma model/source features, acquisition, serve,
TUI/Web. Qualification, diagnostic and experimental native attention
features opt-in; do not use --all-features as equivalent to production
default. Facade defaults remain small. Portable Nix/release build remains
explicit CPU/Qwen3.5/text/TUI/Web (nix/workspace.nix). This was a deliberate
implementation distinction at c0e4794d; **full Cargo default is not proof
that portable distributed artifacts now include all backends**. Reconcile
user's intended full normal package vs explicitly portable assets with
platform owner; no silent cross-platform toolchain requirement or
claim of all-backend release.

ServiceError and safe FailureDetails in engine/router/error.rs. Original
typed failures preserved through core/engine/SDK/CLI/HTTP/SSE. Setting
serve.enable_error_details defaults false, CLI --enable-error-details,
requires authenticated loopback listener and ZEROGRAVITY_API_KEY
configured before start. Only authorized requests get optional details;
bounded original diagnostic text (up to 512 characters per text and
three source causes) can contain internal information; trusted local
debug only, not public/reverse-proxied untrusted endpoint. Native retains
actual diagnostic, shared operational log stores only safe metadata.
Actual loopback HTTP and SSE opt-in/redaction tests passed; browser
configuration whitelist deliberately cannot enable diagnostic exposure.
No change to already-owned listener from profile edits.

### 5.5 Live registration performance (UX-P-M1 contract, retained through M2)

[Owning spec](../plans/cli-ux/LIVE_MODEL_PERFORMANCE.md) is precise. One
fixed-size RegistrationPerformance on original registry Route, request
timing on original ModelRead. No second request registry/reader/scheduler
or extra per-token clock, allocation, mutex or GPU sync. Use existing
committed non-EOS frontier/timestamp, count once even if decoder/transport
fails or parked resumes. Relaxed atomics for counts/durations; terminal
receipt stores last sample once under small mutex. Existing sizeof-based
admission prices added fields.

- Age: monotonic registration age, includes idle; new registration resets,
  aliases share counters, native/Web/HTTP share actual owner.
- TTFT: engine acquire to first committed non-EOS token (includes queue,
  tokenizer, prefill; not HTTP ingress-to-byte or GPU time). Valid first
  token of active or ended request contributes even if later failure/
  cancel; zero-output has no sample.
- Average decode TPS: token-weighted valid within-request commit intervals
  / summed durations (includes scheduler/backpressure delays). No time
  across requests, no idle dilution, no rate without interval and time.
  Aggregate includes active and ended; distinct from wire delivery.
- Completed requests: original Done receipt accepted by bounded presenter,
  not scheduler work-step count. Last completed is registration's last
  sample, not necessarily the visible local response. Snapshot values may
  advance between fields; no fictitious exact per-request association.
- Recent throughput: client delta of committed output / elapsed age via
  existing native 1 s / Web 2 s refresh. Selected owner only, <=24 samples,
  suppress sub-750 ms, clear gap >6 s/regression/replacement, observed idle
  zero not missing. No new timer/history. Normalize trend to observed peak,
  label actual latest; not a per-request speed comparison.
- Running TUI/Web selected panel shows age, Avg decode/TTFT, recent
  sparkline, completion/output counts, actual provider and reported layer
  execution; no samples em dash. Alt-D/context performance details and
  Web disclosure, full aliases/technical observations on demand. F3
  response remains separate. No public inference metrics schema change.
- ModelDetails.layer_placement adapter-supplied zero-based half-open
  actual spans. CPU Qwen verified geometry Host CPU; original whole-text
  single-device prepared source exact plan layers + actual resident
  device. Older resident adapter without map stays unknown. Compute
  placement distinct from weights/storage/state/admitted capacity and
  actual physical usage. Do not infer offload/split from RAM/model name.
  Web explicitly projects safe numeric data and original lease bounds.
  CPU real 24-layer placement verified. The M1-era GPU compile-only
  caveat was narrowed by later actual CUDA placement evidence (§5.6).

### 5.6 Latest UX acceptance and selector (supersedes M1-only caveats)

P4B completed full native Cargo-default link/launch, production strict lint,
explicit CPU/cache/acquisition features, Web-only isolation and original
native/browser/SDK/HTTP/transfer/config/typed-refusal regressions. The gate's
two cold_server failures were obsolete fixture expectations, not weaker
admission. M2 aligns Running with the user reference: 38:62 default split,
paired TPS/TTFT, bounded recent chart, concise actual providers and
segmented actual layer map with exact legend/tooltips and mixed/unknown
cells. Details on demand; independent response pane; no toolbar. Admitted
readers indicate queued/active/retiring work, not live per-layer kernel
activity. No new device profiling or synchronization. Full P4B gate
2,539/0/109 at ad7a4178, accepted by dc9b1956. Actual qualified CUDA source
admission/retry reported 24 compute layers; not new HIP/Vulkan execution
qualification. Full native default linked/launch tested; portable packaged
release remains explicitly CPU. M2 deeper per-request/phase/device and
observer-overhead work remains optional/capability-bound.

240bd05e adds one model-local Execution selector (native and Web).
Default inherits active profile, absent explicit policy uses automatic
routing. Explicit backend alone resets inherited device restriction to
automatic; explicit device is one exact physical override. Drafts bound
to exact cached digest, not profile mutation or existing registration
changes. Shared Run/Load/Load-for-Serve intent, original resolver/fit/
admission, cold backend/device discovery only. Actual CPU choice and
exact output in both clients qualified; no silent fallback or source-fit
promise. Gate 2,543/0/109.

### 5.7 Serving transport, semantic request and projection

- OA-1 uses one owned current-thread Tokio runtime and bounded Hyper HTTP/1
  connection tasks, persistent/pipelined, original charged event-channel
  readiness. Header 8 KiB / 32 fields; 10 s ingress deadline, 256 requests
  per connection, cumulative wire budget, 30 s stalled-write bound.
  Connection/server 64 KiB conservative allowances, not exact allocator
  census. Read-only bounded peer-close observer handles buffered pipelines;
  original receipt cancellation and joined shutdown. Actual SSE token
  payload handoff timestamp, not client receipt.
- Vendored Hyper 1.13.0 narrow server framing patch rejects mixed TE/CL
  in either order and repeated CL before normalization; client unchanged.
  Fixed/chunked JSON feeds one charged decoded reader, absolute cap/deadline,
  no unsupported Expect/trailers or transfer coding. Bounded multipart
  Multer 3.1.0 with checksum provenance and strict initial/final delimiter,
  decoded EOF and duplicate-part-header policy; charged per-part/aggregate
  staging. No preamble/epilogue, filename-to-path, implicit transfer
  decoding or persistent upload store. Cached form-data 0.6.0 rejected
  for truncated loop and invalid boundary termination. WebSocket upgrades
  reuse original connection, wire budget, frame/message caps and joined
  lifetime; private test adapter only, no advertised Realtime inference.
- OA-2: core/task/text.rs ChatRole, ChatMessage, TextInput (raw/user/
  conversation); SemanticRequest and original typed ResponseEvent,
  RequestLease/Scheduled/Active. Legacy Request and borrowed TextTask
  adapt, CLI direct CPU uses same Qwen renderer. HTTP only parses typed
  shapes; selected model measures PromptBounds, prices rendered bytes,
  prompt tokens and scratch, validates and tokenizes under original
  owner. Exact prior Qwen template, ordered roles and assistant prefix.
  1 MiB rendered and configured input cap retained; typed chat rejects
  role delimiters including single-user; explicit raw unchanged. No
  arbitrary Jinja or fake media/tool coercion.
- OA-3: accepting registry Models list/exact retrieve, Responses text
  string/ordered items/instructions as developer, bounded metadata and
  explicit greedy controls. Borrowed serde views and charged bounded
  projection; named SSE events, sequence and response/item IDs, length
  incomplete, safe failed state, actual restored cached prompt-token count
  on terminal event. Server correlation IDs with original legacy metrics
  stream ID preserved. Parser-level errors remain Hyper-owned. Static
  parameter names in errors; private details opt-in only. No Chat wire
  translation or added event queue.

### 5.8 OA-4 accepted protocol record and background contracts

The following is the exact current checkpoint narrative, recovered
from the signed implementation. OA-4A details (including earlier
foreground acceptance and limitations) appear in Appendix D and
CURRENT_HANDOFF; only the current live instructions take precedence.

### Current checkpoint — OA-4B accepted; OA-4 conversations/count/replay next (27 Nov)

Started from clean signed 0898734b on L5. Rechecked current background
mode and cancel contracts:
https://developers.openai.com/api/docs/guides/background
https://developers.openai.com/api/reference/resources/responses/methods/cancel
Support text background=true with nonstream create/poll/cancel, default
soonest completion window. Explicit background_options.completion_window
soonest accepted; other windows/continuation-ID controls refuse.
Background streaming/resumption still refuses until a bounded replay
contract is qualified. No webhook or cloud capacity/discount claim.

One listener-owned JoinSet supervised by the existing accept loop
(reap even without incoming connections, abort/join after ingress stop).
Hard maximum 64 task handles including completed/unjoined handles.
No extra inference queue, worker, scheduler, detached task or model state.
Reserve the original shared protocol quota before model admission:
64 pending/retained record slots, 32 MiB conservative backing, same
original ServingCharge. Background reserves 7x at most 1 MiB terminal
serialization limit plus actual input backing and 64 KiB headroom for
immutable initial/cancel/failure/terminal projections and task state;
options metadata retains its own allowance, no unused SSE reservation.
Foreground reservations unchanged. Initial in_progress means accepted
protocol processing, not physical GPU execution or an exact scheduler
phase. After admission, mandatory snapshots are prepared before
publishing the ID and spawning the joined consumer. If preparation
fails, the original receipt is dropped and cancelled; no published ID.
Background owns the original nonstream event receipt independently of
the creation socket, shares the same absolute 180s generation deadline
and terminal parser as foreground JSON. Initial and terminal snapshots
retain one shared input and credit; polling uses Bytes owner references,
never another strong inference receipt. Safe failure records, including
oversized terminal output, always have a pre-priced failed fallback.
No partial output/usage fabricated for cancellation or failed nonstream.

Cancel signals the original receipt, wakes its joined consumer and
atomically replaces the active entry with cancelled; terminal publication
checks active entry identity, so a late Done cannot revive a cancelled
or deleted record. Repeated cancel on terminal background returns the
same final object. Active DELETE refuses until terminal. Foreground
cancel still refuses; foreground disconnect still cancels its request.
Logical cancellation is not a physical completion proof; original
engine owns physical retirement. All resource routes retain original
listener authorization and bounded socket output.

Explicit store=true retains terminal resources 30min and permits
successful/incomplete continuation. Background store=false retains
only temporary polling/input-item resources for 10min from terminal,
no continuation; ordinary foreground store=false retains nothing.
Both remain memory-only and expire on access/idle listener tick or
listener shutdown. Shared quota follows all retained readers even after
expiry/deletion; active tasks cannot age out before their deadline.
These are local bounded policies, not cloud retention guarantees.

Focused engine/serving 172/0/0 and strict lint passed. Actual held-receipt
TCP tests verify create disconnect independence, input items during
processing, active deletion refusal, cancellation/late terminal race,
idempotence, unrelated worker completion and usage, temporary continuation
refusal, joined shutdown and ledger retirement. Worker error and oversized
terminal output publish safe failed records. Existing installed OpenAI
Python SDK 2.33.0 real cached model fixture now exercises background
polling, text/usage parity, terminal cancel idempotence, cancellation
race and stored vs temporary continuation. Actual official SDK
cancellation returned cancelled, and the cached CLI/TUI/Web/SDK
lifecycle passed (130.88s); Rust SDK passed (14.36s).
Full isolated assembled gate exited 0 (377.3s): 2,562/0/109 across
318 Rust suites, backend feature boundaries, strict lint, release TUI/Web,
installer and scripts passed. OA-4B accepted. Build concurrency reduced to 4 jobs per lane
for observed memory pressure; no new dependencies/downloads.
Logs: `target/oa4b-focused.log`, `target/oa4b-real-consumers.log`,
`target/oa4b-assembled.log`. Exact Nix source-integrity remains separately
blocked by unavailable locked inputs. No new physical GPU/performance
claim.

Remaining OA-4: conversations/items CRUD and atomic context append
using same protocol quota; exact input-token counting through original
charged model tokenizer without opening generation; bounded background
streaming/replay and resume contract. No fake support, additional model
state store or scheduler. Do not mark OA-4 complete.

## 6. Original chronology: beginning through Phase 1

Read the [October 10 master plan](../plans/ZEROGRAVITY_MASTER_PLAN_2026-10-10.md),
original vision, modular architecture, October reviews and full prior packs.
First reachable commit: be95424837deb7919adf34503db71d27d3ad3adb.
The previous comprehensive pack §6, Phase 1 exit pack §§6–9 and its
Appendices C–E preserve the original P00–P23 roadmap, Fabric vNext
transition, source/semantics/state/resource/numerics, all 49 post-matrix
foundation landings, physical tests and rejected trials. Appendix C below
extends the exact complete Git ledger; Appendix E reproduces the previous
main body and indexes the original historical files. Do not reconstruct
earlier semantics or experiment conditions from a short overview.

## 7. Complete Phase 2 and public-serving chronology

Historical UX steps through M1, exact table from prior handoff (earlier focused scopes are not new acceptance results):

| Slice | Commit | Accepted/focused scope and result |
| --- | --- | --- |
| UX-1 | e6211608 | Shared CLI presentation, capability/typed failure; full 2431/0/106 |
| UX-2 | bcdb46a8 | Typed tree and cold resolver, exact source/config/device; full 2451/0/106 |
| UX-3 | 3cf1bb2d | SDK ControlPlane, owner snapshots/actions/events; full 2457/0/108 |
| UX-4 | da2385c7 | Shared transfer/checkpoint pause/resume/cancel; full 2459/0/109 |
| UX-5 | 8e382947 | Canonical run, pinned acquisition and reusable resident owner; full 2465/0/109 |
| UX-6 | 84e0681c | Canonical serve, joined listener and real alias equality; full 2466/0/109 |
| UX-7 | 7f5fc4eb / abe98b83 | Unified document/profiles, recovered interrupted acceptance; full 2473/0/109 |
| UX-8A | 08e7e9d3 | Native shell, palette and terminal lifecycle; full 2477/0/109 |
| UX-8B | 0c7ad87a | Native resident actions; full 2479/0/109 |
| UX-8C | 9bf7829a | Native inventory/transfer/device; full 2481/0/109 |
| UX-8D | 71e36933 | Native config/profile/logs; full 2483/0/109 |
| UX-9A | 4e472aed | Authenticated embedded Web shell; full 2485/0/109 |
| UX-9B | 4f23983a | Web resident/listener shared slot; full 2489/0/109 |
| UX-9C | a6744b12 | Web Models/Downloads/Devices, original exact/pinned actions, actual controlled transfers and cached lifecycle (48.13s); full 2489/0/109 |
| UX-9D | 543194ac | Web Profiles/Config/Logs, exact safe settings, bounded logs; gate interrupted, recovered in 9E |
| UX-9E | 1a878990 | Simultaneous clients and shared invalidations/transfer/settings; real cached 54.16s, full 2491/0/109 |
| UX-10 core | 22c6c0a7 | Cold doctor, typed pipeline inspection and command cleanup; 59/0/1, interrupted CUDA rebuild gate |
| UX-10 exit | 1b8de1cb | Cold backend/quota observations, recovered gate; focused 44/0/2; full 2505/0/109 |
| UX-P design | 3b584f72 | User-inserted polish before public serving; design only, original spec in D.6 |
| UX-P1 | b1073e7c | Semantic themes native/Web, Auto hint; 81/0/3 + JS 3/0 |
| UX-P2 | ab3cf974 | Mouse and panes, scoped capture/scroll; 84/0/3 |
| UX-P3A | 5da96251 | Exact bound actions/modal buttons; 82/0/3 + cached 1/0/0 (52.81s) |
| UX-P3B | 8a2c2ae8 | Typed choice/log filters/grapheme editor/local search; 86/0/3 + cached 1/0/0 (55.96s) |
| UX-P4A1 | f723b0b6 | Screenshot-driven concise summaries, Details, smart sizes; 87/0/3, actual PTY/transfer; resident rerun blocked by memory |
| UX-P4A1 correction | d1e27080 | Toolbar-free object context, inline live search and session menu; 88/0/3 + cached 1/0/0 (57.77s); prior memory blocker resolved |
| UX-P4A2 | 0ee66351 | Stable anchors, separate response/log details, compact dialogs/drag; 91/0/3, real native/attached-Web 1/0/0 (34.49s); browser passed separately |
| Defaults/errors | c0e4794d | Full production Cargo default, typed errors, safe HTTP/SSE+opt-in detail and logs; focused 422 pass/3 skips across recorded affected suites; default check passed; link timed out |
| UX-P-M1 | e96cdab9 | Registration live metrics/layer detail; 109 engine, 64/0/3 CLI, Web 4/0, real cached 1/0/0 (55.61s), default check |

Subsequent checkpoints (every commit detailed in Appendix D):

| Commit | Scope / recorded result |
| --- | --- |
| 53cc7ec3 | Full default link/launch, explicit feature gates, GPU lint, actual owner acceptance |
| 5c919c13 | P4B incomplete assembled gate, two cold_server fixture blockers recorded |
| 490fe410 | M2 reference dashboard and exact layer placement, fixture repairs |
| ad7a4178 | Acquisition-only format import fix |
| dc9b1956 | P4B final isolated gate 2,539/0/109; UX-P accepted |
| 240bd05e | Model-local Execution selector; 2,543/0/109 |
| fbb4ab1d | OA-1A async transport; 2,538/0/109 |
| 3156ce45 | OA-1B ingress/WebSocket primitives; 2,544/0/109 |
| 08a9d30e | Strict Hyper framing and public chunked JSON; focused/real passes; assembled timeout |
| 715127cc | Recovered warmed-cache gate 2,546/0/109; chunked acceptance |
| 3f8329f2 | Authorized single Multer archive; bounded multipart; OA-1 accepted 2,551/0/109 |
| a76ddbfe | OA-2 typed text and model template; 2,553/0/109 |
| cea0f18c | OA-3 Models/Responses stateless, official SDK; 2,557/0/109 |
| 0898734b | OA-4A foreground resources and continuation; 2,560/0/109 |
| 89822f40 | OA-4B joined background polling/cancel; 2,562/0/109 |

Do not reinstate historical toolbars, repeat UX-P4B, or treat interim failures
as still-open production defects. Do not treat aggregate counts across
different feature configurations as unique tests. Earlier acceptance/initial
timeouts and corrected fixtures remain preserved in exact original narratives.
Full local tracking synchronization at 3156ce45 was performed only after
explicit user push authorization, via verified GitHub SSH host identity
(HTTPS CA/credential helper unavailable in sandbox). No permanent config
change. The seven newer implementation/acceptance commits were not pushed.

## 8. Current execution and performance authority

[CURRENT_BASELINE](CURRENT_BASELINE.md) and original backend plans govern
all measured claims. UX and serving changes do not start backend performance work.
Retain original real qualified evidence (see historical pack §8 and
Appendix E): X2 HIP original HF V2 0.8B C1/C8 independent continuation/
retirement, explicit native vs source arithmetic; L5 CUDA resident/
streamed/paged/prefix/cohort and actual GGUF B2 capture full-vocab/all42
equality/unknown retention (public graph serving remains Native);
Vulkan named RADV/portable Q8/conv/coherent mapped producer, not whole
learned parity or general external/UMA storage; CPU GGUF 0.8B/2B original
source/cache, committed continuation and language chains.

Retained historical HIP V2 47+3 0.8B C8 43.638 tok/s, 2B C8 30.949 tok/s,
27B C1 2.228 tok/s and older small-model vLLM 219.612/140.847 tok/s
are historical condition-specific numbers, **not a fresh matched parity
comparison**. Exact conditions and rejected alternatives in older pack
Appendix E §§8–9, baseline and owning plans. Approved pinned X2 vLLM
closure previously unavailable; revalidate/restore approved local
closure before new comparison. Do not download replacements or tune
against unverified reference. User's delivery priority ROCm then CUDA
then Vulkan then optional compiled providers. Read
[backend iteration](../plans/cpu-gpu-optimization/NEXT_ITERATION_BACKEND_PARITY_AND_EXECUTION_PERFORMANCE.md),
[backend parity](../plans/cpu-gpu-optimization/BACKEND_PARITY.md),
[backend TODO](../plans/cpu-gpu-optimization/TODO.md),
[numerics TODO](../plans/numerics-and-context/TODO.md) and
[model execution TODO](../plans/model-execution/TODO.md).
M1 instrumentation overhead and GPU placement physical behavior have
not been measured/qualified by a performance comparison. No imagined
TPS/TTFT/layer estimates; illustrative mockup numbers are not evidence.

## 9. Reusable primitives, rejected approaches and holds

Preserve correct model-neutral GEMM/solve, GDN/attention/norm,
tiled/batched/paged/streamed transport, source-window, state/layout,
sampling, graph/cache, admission/lease, completion, pipeline primitives.
Native graph timing at useful widths did not justify production selection;
retain graph substrate/cache. Paged scatter-tail-zero consumer lost;
retain reusable kernel. Earlier exact held/rejected experiments and
reusable results are in Phase 1 exit pack Appendix E §9, earlier packs,
current baseline and original owner plans. Do not repeat without
materially new evidence or remove dormant primitives to silence warnings.
Renoir/gfx90c AMD launch hold remains. Preserve finite captured CUDA
qualification route distinction; no production selector shortcut.
A1 SM89 original-source CUDA qualification remains pending; do not
turn feature compilation into capability. General source/model/codec/
quality/native Vulkan/UMA/foreign/mixed/async SSD holds remain explicit.
Optional compiler integrations after native campaigns, not replacement
for original state/math/provider/physical contracts.

## 10. Build, validation, host and Git procedure

Read AGENTS.md, current scripts, installed toolchain/cache and prior
Phase 1 pack Appendix E §10 for exact historical physical/bundle
recipes. Mandatory assembled gate when outer connector shares network:

~~~sh
cd /home/pvl/spaces/abird/src/zerogravity
export CARGO_BUILD_JOBS=8 RAYON_NUM_THREADS=4 PYTHONDONTWRITEBYTECODE=1
bwrap --unshare-net --bind / / --dev-bind /dev /dev -- \
  python3 -B scripts/native-gpu.py --require-private-network -- bash scripts/check.sh
~~~

If observed host pressure use fewer build jobs; run real cached model
serially with original 8 GiB admission and original cache. Never bypass
mandatory private-network harness, enable host network/privileges or
download unapproved inputs. For focused checks that need local HTTP,
use the same private harness and the loopback setup from scripts/check.sh;
only private owned user/network namespace loopback, not host interfaces.
Use locked/offline builds. Review every result and cleanup verdict.
Inspect actual staged paths before signed commit; no unrelated edits,
remote push or destructive operations. Keep one main worktree, exact
bundle transfer for execution only; do not infer old X2/A1 revision.
No need to create more snapshot artifacts during normal work.

### 10.1 Required feature and consumer distinctions

The old command `--features serve,acquisition` alone now inherits full
GPU default. For explicitly bounded CPU resident fixture use:

~~~sh
ZG_TEST_BROWSER=1 cargo test --locked --offline --release \
  -p zerogravity-cli --no-default-features \
  --features cpu,model-qwen35,text,tui,web,serve,acquisition \
  --test run_unified canonical_serve_profile_and_cached_aliases_use_one_listener_owner \
  -- --ignored --exact --nocapture --test-threads=1
~~~

Run inside the private harness as above. This is the existing real
cached owner fixture, not a new load or comparator. Retain explicit
test/feature separation, Web-only independent of terminal, CPU-only
without optional GPU/serving dependencies, original acquisition and
serving fixtures. Ratatui/Crossterm and locked Unicode libs optional
TUI-only. SolidJS/Vite assets built locally using installed pinned
dependencies; Web README documents npm test/build and Firefox BiDi.
No Node/runtime remote assets. Actual native PTY shell/settings,
appearance, mouse, controls, editor, ergonomics, Web HTTP/Firefox and
controlled/shared downloads are existing fixtures. Follow current
scripts; no new golden matrix. Full production default check is not
the linked execution check; portable release fixture is explicitly CPU.

### 10.2 Current validation caveats (supersedes M1-era pending P4B)

- P4B full native default link/launch, strict lint and complete isolated gate
  passed. No need to redo them merely because the historical pack listed
  them as blockers. Current production build default still not a claim
  that portable distribution bundles all GPU backends.
- Full OA-4B gate ran with observed memory pressure reducing Cargo build
  jobs to 4 per lane. Existing pinned/offline model and SDK closure only.
  Reuse actual evidence for unchanged paths, not an evidence-only gate.
- Exact Nix source-integrity derivation separately blocked by unavailable
  locked hashes of nixpkgs, pyproject-build-systems, pyproject-nix and
  uv2nix. Crane and rust-overlay matched. Nix syntax and vendored-source
  audit passed; do not substitute arbitrary local sources, download without
  approval or claim exact derivation acceptance. Native portable release
  and installer smoke passed; separate all-backend distribution remains
  platform scope.
- No new GPU execution/performance comparison from serving/UX acceptance.
  Original physical backend qualification and hold conditions remain.
- L5 disk was 95% used; avoid duplicate worktrees/build output and
  unrelated cache purges. Inspect actual resource owners each turn.
- After failures or dependency-path cache invalidation, reuse still-valid
  completed checks and warm-cache retry of mandatory assembled gate.
  Never report a compilation timeout as test pass. Previous Hyper path
  change caused timeout, recovered at 715127cc. Multer acquisition was
  specifically authorized; no authority to fetch additional packages/models.
- Legacy SDK delete() -> None in installed 2.33.0; fixture asserts raw HTTP
  deletion object and subsequent 404. Do not misdiagnose as server defect.

### 10.3 Changed serving consumer / official SDK validation

Within the same private harness and loopback setup from scripts/check.sh,
run the existing cached CLI canary in §10.1 plus:

~~~sh
cargo test --locked --offline --release -p zerogravity \
  --no-default-features --features cpu,model-qwen35,text,cache,serve \
  --test fabric_sdk control_clients_share_exact_model_and_listener_lifecycle \
  -- --ignored --exact --nocapture --test-threads=1
~~~

The existing optional fixture `crates/zerogravity-cli/tests/fixtures/responses_sdk.py`
uses the actual cached model and installed OpenAI SDK 2.33.0 with strict
response validation. Enable via ZG_TEST_OPENAI_PYTHON and
ZG_TEST_OPENAI_PYTHONPATH and ZG_TEST_BROWSER=1 when running the CLI test.
Locate the approved installed closure, no downloads (read-only nix-store -qR
against installed /nix/store/*-python3.13-openai-2.33.0, find its Python
3.13 interpreter and site-packages directories). See prior exact test
logs and source fixture; keep it strictly loopback, trust_env=false,
no remote service/key. It covers actual model list/retrieve, Chat parity,
Responses JSON/SSE, stored lifecycle, previous-response replay,
background polling and cancellation. Extend the same fixture for the
next real endpoint, not a parallel qualification server. Deterministic
held-receipt TCP tests cover races independently of real timing.

For focused network tests, use the private wrapper in §10 with the
loopback setup from scripts/check.sh before locked/offline commands;
do not bypass mandatory isolation. The full script performs all required
feature checks, tests, release and smoke. Check terminal result and actual
log; record exact scope of any explicitly skipped or unqualified work.
No reason to rerun OA-4B on this documentation-only checkpoint.

## 11. Next coherent unit — remaining OA-4

The next natural implementation is conversations/items CRUD and atomic
context append; OA-4 remains open until token count and background
event replay/resumption are qualified too. The live design §32 OA-4 and
owner TODO are the source of the remaining requirements. Suggested
coherent sequencing (not an invented accepted specification):

1. **Conversation resources**: recheck current API contract, use existing
   Store and shared quota/ledger, bounded identity and item pagination,
   mutation/update/deletion/expiry and exact owner lifetimes. Model text
   input remains typed; append input and successful terminal assistant
   output atomically with a clear conflict rule. No stale target,
   duplicate append, false cancelled completion or context mutation on
   failed admission/cancel. Decide actual lifecycle semantics against
   current specs and tests, do not silently assume them from this summary.
2. **Exact input-token counting**: selected admitted model tokenizer and
   exact same semantic input/template, validated bounds/temporary scratch,
   no physical generation session, inference worker duplication, invented
   count or cache hit. Qualify actual count vs real request usage.
3. **Background stream/replay/resumption**: current API semantics,
   bounded event storage and stable cursor/sequence, original joined task
   and original receipt, actual polling/cancel, terminal race and replay
   retirement. No detached work, unbounded event history or second queue.

Start by inspecting the actual accepted background and record code:
`crates/zerogravity-serving/src/http/responses/{store,input,background}.rs`,
`responses.rs`, `transport.rs`, original request/worker/model tokenizer
and admission. Reuse existing auth, opaque IDs, store/credit, expiry,
join set, semantic input and typed error conventions. One listener trust
scope; do not silently broaden to multi-tenant permissions. Preserve
background store=false temporary polling-only policy, store=true
continuation, foreground stateless default and previous instruction
replacement. Verify exact changed consumers, cancellation and ledger
retirement; only advertise real capabilities. No arbitrary extra storage
engine, new scheduler or model session retention.

Then OA-5 Chat/Completions, OA-6–10 capability-qualified adapters and
public serving closeout; ordinary benchmarks and ROCm → CUDA → Vulkan →
optional compiled provider sequence unchanged. Do not leapfrog the
required user priorities or treat a missing modality implementation as
authorization for downloads. Optional M2 profiling stays separate.

## 12. Specification navigation and complete historical context

Appendix A links the entire tracked .agents/docs, .agents/plans,
.agents/prior-art, .agents/research and docs corpus. Original source files
remain in place. The older comprehensive packs are retained and their
appendices contain the exact previous chronology and original design.
Use this hierarchy, not a new duplicated source-of-truth:

| Subject | Key authority |
| --- | --- |
| Original design | [master plan](../plans/ZEROGRAVITY_MASTER_PLAN_2026-10-10.md), [vision](VISION_SCOPE_AND_PRINCIPLES.md), [modular architecture](MODULAR_AI_SYSTEMS_REFERENCE_ARCHITECTURE.md), [engineering spec](../../docs/ENGINEERING_SPECIFICATION.md) |
| Source and semantics | [formats](FORMATS_QUANTIZATION_AND_LOADING.md), [model semantics](MODEL_SEMANTICS_AND_COMPATIBILITY.md), [composition](COMPOSITION_AND_EXTENSION_CONTRACTS.md), [derived quant](DERIVED_QUANTIZATION_CONTRACT_2026-10-15.md), [verified assemblies](VERIFIED_MODEL_ASSEMBLIES_AND_SPECIALIZATION_V5_1_2026-10-15.md), [immutable source](../plans/IMMUTABLE_SHARDED_SOURCE_IMPLEMENTATION_SPEC.md), [acquisition spec](../plans/MODEL_ACQUISITION_IMPLEMENTATION_SPEC.md) |
| Regions/provider/physical | [semantic IR](SEMANTIC_IR_AND_COMPILER.md), [regions](../plans/pipeline-composition/REGIONS_AND_PROFILES.md), [policy and reuse](../plans/pipeline-composition/PROFILE_POLICY_AND_COMPONENT_REUSE.md), [shared execution](../plans/pipeline-composition/SHARED_EXECUTION_ARCHITECTURE.md), [naming](../plans/pipeline-composition/STRUCTURE_AND_NAMING.md) |
| Fabric evolution | [Fabric vNext](../plans/fabric-vnext/README.md), [decisions](../plans/fabric-vnext/DECISIONS_AND_OPEN_QUESTIONS.md), [acceptance](../plans/fabric-vnext/IMPLEMENTATION_AND_ACCEPTANCE.md), [transition](../plans/fabric-vnext-transition/README.md), [transition contracts](../plans/fabric-vnext-transition/ARCHITECTURE_AND_CRATE_CONTRACTS.md) |
| Physical/state | [core protocol](ENGINE_CORE_EXECUTION_AND_PROTOCOL.md), [state](STATE_CACHE_AND_TRANSACTIONS.md), [residency](MEMORY_RESIDENCY_AND_IO.md), [virtual tensor/SSD](UNIFIED_VIRTUAL_TENSOR_RESIDENCY_AND_SSD_STREAMING_2026-10-15.md), [memory TODO](../plans/memory-and-streaming/TODO.md), [gfx1151 safety](GFX1151_ROCM_KERNEL_COST_AND_UNIFIED_MEMORY_SAFETY_CONTRACT_2026-10-15.md) |
| Math/numerics/performance | [operator catalog](OPERATOR_AND_FUSION_CATALOG.md), [kernel/codegen](DEVICE_KERNEL_AND_CODEGEN.md), [cost model](COST_MODEL_AND_AUTOTUNING.md), [numerics](NUMERICS_QUALITY_AND_BENCHMARKS.md), [correctness first](../plans/numerics-and-context/CORRECTNESS_POLICY.md), [baseline](CURRENT_BASELINE.md), [experiment program](../plans/EXPERIMENT_PROGRAM.md) |
| Scheduling/serving | [scheduling/speculation](SCHEDULER_BATCHING_AND_SPECULATION.md), [router/residency](MODEL_ROUTING_RESIDENCY_AND_MULTI_MODEL_SERVING_CONTRACT.md), [paged/batched state](../plans/serving-and-speculation/BATCHING_AND_PAGED_STATE.md), [fused batching](../plans/serving-and-speculation/FUSED_GPU_BATCHING.md), [goodput contract](JOINT_CONCURRENT_SERVING_OPTIMIZATION_AND_GOODPUT_CONTRACT_2026-10-15.md), [public serving design](../plans/serving-and-speculation/NEXT_ITERATION_OPENAI_COMPATIBLE_SERVING.md) |
| Product UX | [zero-config API](ZERO_CONFIG_CLI_CONFIG_AND_OPENAI_API_CONTRACT.md), [runtime UX](RUNTIME_SERVING_AND_UX.md), [CLI UX](../plans/cli-ux/NEXT_ITERATION_CLI_UX_AND_MODEL_LIFECYCLE.md), [native polish](../plans/cli-ux/TUI_POLISH_AND_MOUSE_INTERACTION.md), [live performance](../plans/cli-ux/LIVE_MODEL_PERFORMANCE.md), [CLI guide](../../docs/CLI.md) |
| Multimodal/distributed | [multimodal V5.1](UNIVERSAL_MULTIMODAL_EXECUTION_ARCHITECTURE_V5_1_2026-10-15.md), [realtime/QoS](REALTIME_MULTIMODAL_SERVING_AND_QOS_V5_1_2026-10-15.md), [NPU](NPU_GPU_COOPERATIVE_EXECUTION_AND_TASK_PIPELINES.md), [distributed placement](DISTRIBUTED_PLACEMENT_AND_NETWORK_EXECUTION_CONTRACT.md), [remote state handoff](REMOTE_PEER_TRANSPORT_AND_STATE_HANDOFF_PROTOCOL.md), [multimodal TODO](../plans/multimodal/TODO.md), [distributed TODO](../plans/distributed-execution/TODO.md) |
| Reliability/release | [security](SECURITY_AND_RELIABILITY.md), [failure](INVARIANTS_AND_FAILURE_SEMANTICS.md), [engineering](RUST_WORKSPACE_AND_ENGINEERING.md), [delivery gates](../plans/INTEGRATED_PHASE_0_AND_DELIVERY_GATES.md), [release gates](../plans/RELEASE_ACCEPTANCE_GATES.md), [traceability](../plans/REQUIREMENTS_AND_TEST_TRACEABILITY.md), [platform TODO](../plans/platform-and-release/TODO.md) |

All older comprehensive handoffs from Nov 14,15,18,20,21,22,23,24
(including foundations),25 (foundations and Phase 1 exit),26 (UX-9B and UX-P-M1)
are indexed in Appendix E and A. In particular the earlier full
Phase 1 pack preserves exact source, experiment and rejection scope
going back to the beginning; the UX-9B pack preserves UX-1 through
UX-9B. The present pack adds every post-M1 commit and an updated
pending-work snapshot. Do not execute superseded next-agent prompts.
No original files were deleted or replaced. Historical data remains
available at original Git revisions even when live owner prose evolves.

## 13. Copy/paste starting prompt

Also stored in [NEXT_AGENT_2026-11-27_OA4C.md](NEXT_AGENT_2026-11-27_OA4C.md).

~~~~markdown
You are taking over ZeroGravity. Continue in:

`/home/pvl/spaces/abird/src/zerogravity`

### 1. Acquire complete project context

Read these in order:

1. `AGENTS.md`
2. `.agents/plans/README.md`
3. `.agents/docs/FULL_COMPREHENSIVE_HANDOFF_2026-11-27_OA4B.md` — §§0–13 first
4. `.agents/plans/FULL_IMPLEMENTATION_PLAN.md`
5. `.agents/docs/CURRENT_HANDOFF.md`
6. `.agents/docs/CURRENT_BASELINE.md`
7. `.agents/plans/TODO.md`
8. `.agents/plans/serving-and-speculation/NEXT_ITERATION_OPENAI_COMPATIBLE_SERVING.md` — especially §§6–10, 17–25, 30–37 and OA-4
9. `.agents/plans/serving-and-speculation/TODO.md`
10. `.agents/plans/fabric-vnext/PUBLIC_LIB_AND_SERVING.md`, the current router/state/resource/SDK contracts and `.agents/plans/pipeline-composition/STRUCTURE_AND_NAMING.md`.
11. Relevant owning CLI/UX, acquisition, platform, model execution and numerical specifications as the actual change requires.

The comprehensive handoff's Appendix D preserves every post-M1 landing with exact Git references, original narratives and plan changes. Appendix E retains the previous complete main body and links the full historical chain through the initial October design. Appendix A indexes the original engineering corpus; Appendix B preserves unresolved checklist markers and Appendix C the complete reachable Git ledger. Read the historical evidence before revisiting rejected designs or experiments. Earlier continuation prompts are historical. Live source, tests and owning plans remain authoritative.

### 2. Revalidate repository and execution authority

Latest accepted implementation: `89822f4038b2048affd8b92dac6c7b2bcbeb219e` — OA-4B.

Accepted OA-4A: `0898734b`; OA-3: `cea0f18c`; OA-2: `a76ddbfe`; OA-1: `3f8329f2`.
Accepted UX-P4B gate: `dc9b1956`, implementation `ad7a4178`; model-local execution selector: `240bd05e`.
Accepted UX-1–10: `1b8de1cb87fa4815655cbb6700e6633946f93305`.
Accepted bounded Phase 1 exit: `d5030ad4ce476b913a46056d492992ccf1b8edf7`.
The latest comprehensive handoff is a documentation-only commit above OA-4B; inspect actual HEAD rather than assuming it is the implementation SHA.

Use **dotlink---l5 as the sole source-authoring connector**. Main is the only persistent branch/checkout. Revalidate HEAD, staged/unstaged/untracked changes, local tracking ref, worktrees, active durable jobs, physical GPU/host owners and free disk/memory. X2/A1 are execution/testing hosts only; sync exact authorized L5 commits with verified incremental Git bundles and clean fast-forward when necessary. No destructive Git, unapproved download, privilege change, GPU reset or unrelated process cleanup. Sign commits; do not push unless separately instructed. Root README is user-authored; leave it unchanged.

### 3. Current accepted position

Phase 1 bounded common foundations, UX-1–10, UX-P4B (including toolbar-free design, live metrics and exact reported layer spans) and the model-local backend/device chooser are accepted. OA-1 async transport, OA-2 semantic text input and model-owned templates, OA-3 stateless Models/Responses, OA-4A stored foreground resources and OA-4B joined background polling/cancellation are accepted.

OA-4B already committed and passed: 172 focused engine/serving test executions, strict lint, actual cached Qwen3.5 official OpenAI SDK 2.33.0 (including an actual cancelled background response), real CLI/TUI/Firefox lifecycle and Rust SDK, plus the full isolated gate 2,562 passed / 0 failed / 109 skipped across 318 Rust suites. Do not redo the interrupted turn.

Keep one Runtime, ControlPlane, shared Actions slot, original Selection/ConfigDocument, semantic admission, model registry, scheduler, worker, StateRoot, ResourceArbiter, listener, source/cache/transfer owners and original request receipt. Store only protocol records, not model sessions. Current listener trust scope is not a multi-tenant security claim. Foreground store=false retains nothing; background store=false retains temporary polling records only (10 minutes, no continuation). Stored terminal records 30 minutes. Shared quota 64 record slots, 32 MiB conservative backing; pending/residual readers count; background JoinSet also hard-bounded 64 handles. Logical cancellation is not physical completion.

### 4. Next coherent unit — OA-4 conversation resources

Continue the remaining OA-4 work. Start with conversations/items CRUD and atomic append of context on successful generation using the same protocol store/admission authority. Inspect actual accepted `crates/zerogravity-serving/src/http/responses/{store,input,background}.rs`, `responses.rs`, `transport.rs`, engine tokenizer, request and owner contracts. Recheck the current API reference for supported fields, resource shapes and lifecycle. Do not reconstruct the architecture from assumptions.

Qualify exact authorization and identity, bounded pagination, mutation/expiry/deletion and retained-reader lifetimes, conflict/cancellation/failure semantics and original semantic text replay. Avoid partial append, stale retargeting, lost concurrent updates or fake unsupported items. Preserve previous_response_id behavior and replacement of request-local instructions. Keep response and conversation accounting under the same quota/physical ledger, no independent authority or model-state retention. Support only real typed text capabilities. Make the largest coherent tested landing rather than overengineering tiny evidence stages.

Then complete exact input-token counting via the selected original charged tokenizer without a physical generation session, and bounded background streaming/event replay/resumption with the same joined job/receipt. Keep unsupported routes/options explicit until qualified. Do not claim all OA-4 accepted until its remaining contracts are implemented and validated. OA-5 compatibility close and later endpoint families remain ordered; enable new modalities only through real model capabilities.

### 5. Roadmap and engineering contract

After public serving and ordinary CLI/SDK/server benchmark acceptance, match or exceed optimized matched vLLM on ROCm first, CUDA next, native Vulkan third. Concrete optional MIGraphX/TensorRT/IREE/ONNX Runtime later. HF original Safetensors and GGUF primary, no mandatory ONNX conversion. Preserve prior numerical/source/physical/quality and state/completion semantics. No new hardware/performance claims from compile or CPU UI tests.

Keep Rust elegant, simple, clear and intuitive; zero-cost abstractions where practical; use mature optimized mathematics and focus on real bottlenecks. Preserve correct model-neutral primitives when a selector loses. Batch meaningful tests, actual changed-consumer checks and the assembled mandatory private-network workspace gate; reuse applicable accepted evidence. No overengineered pinning/fixtures/receipts. Run locked/offline; use private loopback via the existing `scripts/native-gpu.py --require-private-network` harness and explicit Bubblewrap network namespace on the L5 connector. See handoff §10 for exact commands. Keep build concurrency within available memory and disk (4 jobs per lane was used for OA-4B); do not purge unrelated caches. No new downloads unless authorized.

Aim for one substantial coherent validated signed checkpoint per user turn within approximately 30 minutes. Recover interruptions from actual source and results. Update owning TODOs, current handoff and baseline; report accepted scope, exact commit, blockers and next unit; pause. Do not push unless explicitly requested.

**Revalidate the current state and continue implementation.**
~~~~

## Appendix A — complete engineering document inventory

All tracked files in the original engineering corpus at 89822f4038b2048affd8b92dac6c7b2bcbeb219e. The new handoff and prompt are listed separately. Original source/spec files remain intact; no re-pinning required.

- [.agents/docs/ACCELERATOR_EXECUTION_FAMILIES_AND_BACKEND_CONTRACT.md](ACCELERATOR_EXECUTION_FAMILIES_AND_BACKEND_CONTRACT.md)
- [.agents/docs/ASYNC_RESOURCE_COMMAND_IR.md](ASYNC_RESOURCE_COMMAND_IR.md)
- [.agents/docs/COMPOSITION_AND_EXTENSION_CONTRACTS.md](COMPOSITION_AND_EXTENSION_CONTRACTS.md)
- [.agents/docs/COST_MODEL_AND_AUTOTUNING.md](COST_MODEL_AND_AUTOTUNING.md)
- [.agents/docs/CURRENT_BASELINE.md](CURRENT_BASELINE.md)
- [.agents/docs/CURRENT_HANDOFF.md](CURRENT_HANDOFF.md)
- [.agents/docs/DERIVED_QUANTIZATION_CONTRACT_2026-10-15.md](DERIVED_QUANTIZATION_CONTRACT_2026-10-15.md)
- [.agents/docs/DEVICE_KERNEL_AND_CODEGEN.md](DEVICE_KERNEL_AND_CODEGEN.md)
- [.agents/docs/DISTRIBUTED_PARALLELISM.md](DISTRIBUTED_PARALLELISM.md)
- [.agents/docs/DISTRIBUTED_PLACEMENT_AND_NETWORK_EXECUTION_CONTRACT.md](DISTRIBUTED_PLACEMENT_AND_NETWORK_EXECUTION_CONTRACT.md)
- [.agents/docs/ENGINE_CORE_EXECUTION_AND_PROTOCOL.md](ENGINE_CORE_EXECUTION_AND_PROTOCOL.md)
- [.agents/docs/EXAMPLE_EXECUTION_TRACES.md](EXAMPLE_EXECUTION_TRACES.md)
- [.agents/docs/EXTREME_HYBRID_MODELS_AND_SPECULATIVE_DRAFTER_SEMANTICS_2026-10-15.md](EXTREME_HYBRID_MODELS_AND_SPECULATIVE_DRAFTER_SEMANTICS_2026-10-15.md)
- [.agents/docs/FORMATS_QUANTIZATION_AND_LOADING.md](FORMATS_QUANTIZATION_AND_LOADING.md)
- [.agents/docs/FULL_COMPREHENSIVE_HANDOFF_2026-11-14.md](FULL_COMPREHENSIVE_HANDOFF_2026-11-14.md)
- [.agents/docs/FULL_COMPREHENSIVE_HANDOFF_2026-11-15.md](FULL_COMPREHENSIVE_HANDOFF_2026-11-15.md)
- [.agents/docs/FULL_COMPREHENSIVE_HANDOFF_2026-11-18.md](FULL_COMPREHENSIVE_HANDOFF_2026-11-18.md)
- [.agents/docs/FULL_COMPREHENSIVE_HANDOFF_2026-11-20.md](FULL_COMPREHENSIVE_HANDOFF_2026-11-20.md)
- [.agents/docs/FULL_COMPREHENSIVE_HANDOFF_2026-11-21.md](FULL_COMPREHENSIVE_HANDOFF_2026-11-21.md)
- [.agents/docs/FULL_COMPREHENSIVE_HANDOFF_2026-11-22.md](FULL_COMPREHENSIVE_HANDOFF_2026-11-22.md)
- [.agents/docs/FULL_COMPREHENSIVE_HANDOFF_2026-11-23.md](FULL_COMPREHENSIVE_HANDOFF_2026-11-23.md)
- [.agents/docs/FULL_COMPREHENSIVE_HANDOFF_2026-11-24.md](FULL_COMPREHENSIVE_HANDOFF_2026-11-24.md)
- [.agents/docs/FULL_COMPREHENSIVE_HANDOFF_2026-11-24_FOUNDATIONS.md](FULL_COMPREHENSIVE_HANDOFF_2026-11-24_FOUNDATIONS.md)
- [.agents/docs/FULL_COMPREHENSIVE_HANDOFF_2026-11-25_FOUNDATIONS.md](FULL_COMPREHENSIVE_HANDOFF_2026-11-25_FOUNDATIONS.md)
- [.agents/docs/FULL_COMPREHENSIVE_HANDOFF_2026-11-25_PHASE1_EXIT.md](FULL_COMPREHENSIVE_HANDOFF_2026-11-25_PHASE1_EXIT.md)
- [.agents/docs/FULL_COMPREHENSIVE_HANDOFF_2026-11-26_UX9B_EXIT.md](FULL_COMPREHENSIVE_HANDOFF_2026-11-26_UX9B_EXIT.md)
- [.agents/docs/FULL_COMPREHENSIVE_HANDOFF_2026-11-26_UX_PM1.md](FULL_COMPREHENSIVE_HANDOFF_2026-11-26_UX_PM1.md)
- [.agents/docs/GFX1151_ROCM_KERNEL_COST_AND_UNIFIED_MEMORY_SAFETY_CONTRACT_2026-10-15.md](GFX1151_ROCM_KERNEL_COST_AND_UNIFIED_MEMORY_SAFETY_CONTRACT_2026-10-15.md)
- [.agents/docs/HETEROGENEOUS_PREFILL_AND_RETAINED_REPLAY_CONTRACT_2026-10-15.md](HETEROGENEOUS_PREFILL_AND_RETAINED_REPLAY_CONTRACT_2026-10-15.md)
- [.agents/docs/INVARIANTS_AND_FAILURE_SEMANTICS.md](INVARIANTS_AND_FAILURE_SEMANTICS.md)
- [.agents/docs/JOINT_CONCURRENT_SERVING_OPTIMIZATION_AND_GOODPUT_CONTRACT_2026-10-15.md](JOINT_CONCURRENT_SERVING_OPTIMIZATION_AND_GOODPUT_CONTRACT_2026-10-15.md)
- [.agents/docs/LINUX_DEV_NODE_SELECTORS_2026-10-25.md](LINUX_DEV_NODE_SELECTORS_2026-10-25.md)
- [.agents/docs/MEMORY_RESIDENCY_AND_IO.md](MEMORY_RESIDENCY_AND_IO.md)
- [.agents/docs/MINIMAL_EXECUTION_SPINE_AND_PROTOCOL_V4.md](MINIMAL_EXECUTION_SPINE_AND_PROTOCOL_V4.md)
- [.agents/docs/MODEL_ROUTING_RESIDENCY_AND_MULTI_MODEL_SERVING_CONTRACT.md](MODEL_ROUTING_RESIDENCY_AND_MULTI_MODEL_SERVING_CONTRACT.md)
- [.agents/docs/MODEL_SEMANTICS_AND_COMPATIBILITY.md](MODEL_SEMANTICS_AND_COMPATIBILITY.md)
- [.agents/docs/MODULAR_AI_SYSTEMS_REFERENCE_ARCHITECTURE.md](MODULAR_AI_SYSTEMS_REFERENCE_ARCHITECTURE.md)
- [.agents/docs/NEXT_AGENT_2026-11-26_UX9C.md](NEXT_AGENT_2026-11-26_UX9C.md)
- [.agents/docs/NEXT_AGENT_2026-11-26_UX_P4B.md](NEXT_AGENT_2026-11-26_UX_P4B.md)
- [.agents/docs/NPU_GPU_COOPERATIVE_EXECUTION_AND_TASK_PIPELINES.md](NPU_GPU_COOPERATIVE_EXECUTION_AND_TASK_PIPELINES.md)
- [.agents/docs/NUMERICS_QUALITY_AND_BENCHMARKS.md](NUMERICS_QUALITY_AND_BENCHMARKS.md)
- [.agents/docs/OPERATOR_AND_FUSION_CATALOG.md](OPERATOR_AND_FUSION_CATALOG.md)
- [.agents/docs/PERFORMANCE_HYPOTHESES_AND_RISKS.md](PERFORMANCE_HYPOTHESES_AND_RISKS.md)
- [.agents/docs/PRODUCT_MODEL_PIPELINE_AND_EXTENSIONS.md](PRODUCT_MODEL_PIPELINE_AND_EXTENSIONS.md)
- [.agents/docs/REALTIME_MULTIMODAL_SERVING_AND_QOS_V5_1_2026-10-15.md](REALTIME_MULTIMODAL_SERVING_AND_QOS_V5_1_2026-10-15.md)
- [.agents/docs/REMOTE_PEER_TRANSPORT_AND_STATE_HANDOFF_PROTOCOL.md](REMOTE_PEER_TRANSPORT_AND_STATE_HANDOFF_PROTOCOL.md)
- [.agents/docs/RUNTIME_SERVING_AND_UX.md](RUNTIME_SERVING_AND_UX.md)
- [.agents/docs/RUST_WORKSPACE_AND_ENGINEERING.md](RUST_WORKSPACE_AND_ENGINEERING.md)
- [.agents/docs/SCHEDULER_BATCHING_AND_SPECULATION.md](SCHEDULER_BATCHING_AND_SPECULATION.md)
- [.agents/docs/SECURITY_AND_RELIABILITY.md](SECURITY_AND_RELIABILITY.md)
- [.agents/docs/SEMANTIC_IR_AND_COMPILER.md](SEMANTIC_IR_AND_COMPILER.md)
- [.agents/docs/STATE_CACHE_AND_TRANSACTIONS.md](STATE_CACHE_AND_TRANSACTIONS.md)
- [.agents/docs/TASK_PROGRAM_AND_COMPILER_REFINEMENT_V4.md](TASK_PROGRAM_AND_COMPILER_REFINEMENT_V4.md)
- [.agents/docs/TERMINOLOGY.md](TERMINOLOGY.md)
- [.agents/docs/UNIFIED_MULTIMODAL_EXECUTION_CONTRACT_V5_2026-10-14.md](UNIFIED_MULTIMODAL_EXECUTION_CONTRACT_V5_2026-10-14.md)
- [.agents/docs/UNIFIED_VIRTUAL_TENSOR_RESIDENCY_AND_SSD_STREAMING_2026-10-15.md](UNIFIED_VIRTUAL_TENSOR_RESIDENCY_AND_SSD_STREAMING_2026-10-15.md)
- [.agents/docs/UNIVERSAL_MULTIMODAL_EXECUTION_ARCHITECTURE_V5_1_2026-10-15.md](UNIVERSAL_MULTIMODAL_EXECUTION_ARCHITECTURE_V5_1_2026-10-15.md)
- [.agents/docs/VERIFIED_MODEL_ASSEMBLIES_AND_SPECIALIZATION_V5_1_2026-10-15.md](VERIFIED_MODEL_ASSEMBLIES_AND_SPECIALIZATION_V5_1_2026-10-15.md)
- [.agents/docs/VISION_SCOPE_AND_PRINCIPLES.md](VISION_SCOPE_AND_PRINCIPLES.md)
- [.agents/docs/ZERO_CONFIG_CLI_CONFIG_AND_OPENAI_API_CONTRACT.md](ZERO_CONFIG_CLI_CONFIG_AND_OPENAI_API_CONTRACT.md)
- [.agents/plans/CURRENT_STATUS_AND_TRACKING.md](../plans/CURRENT_STATUS_AND_TRACKING.md)
- [.agents/plans/EXPERIMENT_PROGRAM.md](../plans/EXPERIMENT_PROGRAM.md)
- [.agents/plans/FULL_IMPLEMENTATION_PLAN.md](../plans/FULL_IMPLEMENTATION_PLAN.md)
- [.agents/plans/IMMUTABLE_SHARDED_SOURCE_IMPLEMENTATION_SPEC.md](../plans/IMMUTABLE_SHARDED_SOURCE_IMPLEMENTATION_SPEC.md)
- [.agents/plans/IMPLEMENTATION_ROADMAP.md](../plans/IMPLEMENTATION_ROADMAP.md)
- [.agents/plans/INTEGRATED_PHASE_0_AND_DELIVERY_GATES.md](../plans/INTEGRATED_PHASE_0_AND_DELIVERY_GATES.md)
- [.agents/plans/MODEL_ACQUISITION_IMPLEMENTATION_SPEC.md](../plans/MODEL_ACQUISITION_IMPLEMENTATION_SPEC.md)
- [.agents/plans/README.md](../plans/README.md)
- [.agents/plans/RELEASE_ACCEPTANCE_GATES.md](../plans/RELEASE_ACCEPTANCE_GATES.md)
- [.agents/plans/REQUIREMENTS_AND_TEST_TRACEABILITY.md](../plans/REQUIREMENTS_AND_TEST_TRACEABILITY.md)
- [.agents/plans/TODO.md](../plans/TODO.md)
- [.agents/plans/ZEROGRAVITY_MASTER_PLAN_2026-10-10.md](../plans/ZEROGRAVITY_MASTER_PLAN_2026-10-10.md)
- [.agents/plans/cli-device-routing/README.md](../plans/cli-device-routing/README.md)
- [.agents/plans/cli-device-routing/TODO.md](../plans/cli-device-routing/TODO.md)
- [.agents/plans/cli-ux/LIVE_MODEL_PERFORMANCE.md](../plans/cli-ux/LIVE_MODEL_PERFORMANCE.md)
- [.agents/plans/cli-ux/NEXT_ITERATION_CLI_UX_AND_MODEL_LIFECYCLE.md](../plans/cli-ux/NEXT_ITERATION_CLI_UX_AND_MODEL_LIFECYCLE.md)
- [.agents/plans/cli-ux/TUI_POLISH_AND_MOUSE_INTERACTION.md](../plans/cli-ux/TUI_POLISH_AND_MOUSE_INTERACTION.md)
- [.agents/plans/cpu-gpu-optimization/BACKEND_PARITY.md](../plans/cpu-gpu-optimization/BACKEND_PARITY.md)
- [.agents/plans/cpu-gpu-optimization/BENCHMARK_CONTAINER_TERMINATION.md](../plans/cpu-gpu-optimization/BENCHMARK_CONTAINER_TERMINATION.md)
- [.agents/plans/cpu-gpu-optimization/NEXT_ITERATION_BACKEND_PARITY_AND_EXECUTION_PERFORMANCE.md](../plans/cpu-gpu-optimization/NEXT_ITERATION_BACKEND_PARITY_AND_EXECUTION_PERFORMANCE.md)
- [.agents/plans/cpu-gpu-optimization/SHARED_MEMORY_AND_BOUNDARIES.md](../plans/cpu-gpu-optimization/SHARED_MEMORY_AND_BOUNDARIES.md)
- [.agents/plans/cpu-gpu-optimization/SHORT_HF_RESULTS.md](../plans/cpu-gpu-optimization/SHORT_HF_RESULTS.md)
- [.agents/plans/cpu-gpu-optimization/TODO.md](../plans/cpu-gpu-optimization/TODO.md)
- [.agents/plans/distributed-execution/TODO.md](../plans/distributed-execution/TODO.md)
- [.agents/plans/fabric-vnext-transition/ARCHITECTURE_AND_CRATE_CONTRACTS.md](../plans/fabric-vnext-transition/ARCHITECTURE_AND_CRATE_CONTRACTS.md)
- [.agents/plans/fabric-vnext-transition/CURRENT_SOURCE_RECONCILIATION.md](../plans/fabric-vnext-transition/CURRENT_SOURCE_RECONCILIATION.md)
- [.agents/plans/fabric-vnext-transition/DECISIONS_RISKS_AND_ACCEPTANCE.md](../plans/fabric-vnext-transition/DECISIONS_RISKS_AND_ACCEPTANCE.md)
- [.agents/plans/fabric-vnext-transition/FEATURE_PARITY_MATRIX.md](../plans/fabric-vnext-transition/FEATURE_PARITY_MATRIX.md)
- [.agents/plans/fabric-vnext-transition/HANDOFF.md](../plans/fabric-vnext-transition/HANDOFF.md)
- [.agents/plans/fabric-vnext-transition/PHASED_IMPLEMENTATION.md](../plans/fabric-vnext-transition/PHASED_IMPLEMENTATION.md)
- [.agents/plans/fabric-vnext-transition/QUALIFICATION_AND_COMMIT_POLICY.md](../plans/fabric-vnext-transition/QUALIFICATION_AND_COMMIT_POLICY.md)
- [.agents/plans/fabric-vnext-transition/QUANTIZATION_AND_NUMERICS.md](../plans/fabric-vnext-transition/QUANTIZATION_AND_NUMERICS.md)
- [.agents/plans/fabric-vnext-transition/README.md](../plans/fabric-vnext-transition/README.md)
- [.agents/plans/fabric-vnext-transition/ROADMAP_AFTER_TRANSITION.md](../plans/fabric-vnext-transition/ROADMAP_AFTER_TRANSITION.md)
- [.agents/plans/fabric-vnext-transition/STRUCTURAL_SKELETON.md](../plans/fabric-vnext-transition/STRUCTURAL_SKELETON.md)
- [.agents/plans/fabric-vnext-transition/TODO.md](../plans/fabric-vnext-transition/TODO.md)
- [.agents/plans/fabric-vnext/DECISIONS_AND_OPEN_QUESTIONS.md](../plans/fabric-vnext/DECISIONS_AND_OPEN_QUESTIONS.md)
- [.agents/plans/fabric-vnext/HANDOFF.md](../plans/fabric-vnext/HANDOFF.md)
- [.agents/plans/fabric-vnext/IMPLEMENTATION_AND_ACCEPTANCE.md](../plans/fabric-vnext/IMPLEMENTATION_AND_ACCEPTANCE.md)
- [.agents/plans/fabric-vnext/PRIOR_ART_AND_REVALIDATION.md](../plans/fabric-vnext/PRIOR_ART_AND_REVALIDATION.md)
- [.agents/plans/fabric-vnext/PUBLIC_LIB_AND_SERVING.md](../plans/fabric-vnext/PUBLIC_LIB_AND_SERVING.md)
- [.agents/plans/fabric-vnext/README.md](../plans/fabric-vnext/README.md)
- [.agents/plans/fabric-vnext/TODO.md](../plans/fabric-vnext/TODO.md)
- [.agents/plans/memory-and-streaming/README.md](../plans/memory-and-streaming/README.md)
- [.agents/plans/memory-and-streaming/TODO.md](../plans/memory-and-streaming/TODO.md)
- [.agents/plans/model-acquisition/README.md](../plans/model-acquisition/README.md)
- [.agents/plans/model-acquisition/TODO.md](../plans/model-acquisition/TODO.md)
- [.agents/plans/model-execution/README.md](../plans/model-execution/README.md)
- [.agents/plans/model-execution/TODO.md](../plans/model-execution/TODO.md)
- [.agents/plans/model-execution/archive/populated-hip-serving/TODO.md](../plans/model-execution/archive/populated-hip-serving/TODO.md)
- [.agents/plans/multimodal/TODO.md](../plans/multimodal/TODO.md)
- [.agents/plans/numerics-and-context/CORRECTNESS_FIRST_VLLM_PLAN_2026-10-20.md](../plans/numerics-and-context/CORRECTNESS_FIRST_VLLM_PLAN_2026-10-20.md)
- [.agents/plans/numerics-and-context/CORRECTNESS_POLICY.md](../plans/numerics-and-context/CORRECTNESS_POLICY.md)
- [.agents/plans/numerics-and-context/TODO.md](../plans/numerics-and-context/TODO.md)
- [.agents/plans/pipeline-composition/FORMATS.md](../plans/pipeline-composition/FORMATS.md)
- [.agents/plans/pipeline-composition/PROFILE_POLICY_AND_COMPONENT_REUSE.md](../plans/pipeline-composition/PROFILE_POLICY_AND_COMPONENT_REUSE.md)
- [.agents/plans/pipeline-composition/README.md](../plans/pipeline-composition/README.md)
- [.agents/plans/pipeline-composition/REGIONS_AND_PROFILES.md](../plans/pipeline-composition/REGIONS_AND_PROFILES.md)
- [.agents/plans/pipeline-composition/SHARED_EXECUTION_ARCHITECTURE.md](../plans/pipeline-composition/SHARED_EXECUTION_ARCHITECTURE.md)
- [.agents/plans/pipeline-composition/STRUCTURE_AND_NAMING.md](../plans/pipeline-composition/STRUCTURE_AND_NAMING.md)
- [.agents/plans/pipeline-composition/TODO.md](../plans/pipeline-composition/TODO.md)
- [.agents/plans/platform-and-release/TODO.md](../plans/platform-and-release/TODO.md)
- [.agents/plans/qwen-parity/API_ROUTE_FRONTIER.csv](../plans/qwen-parity/API_ROUTE_FRONTIER.csv)
- [.agents/plans/qwen-parity/CLI_OPTION_FRONTIER.csv](../plans/qwen-parity/CLI_OPTION_FRONTIER.csv)
- [.agents/plans/serving-and-speculation/BATCHING_AND_PAGED_STATE.md](../plans/serving-and-speculation/BATCHING_AND_PAGED_STATE.md)
- [.agents/plans/serving-and-speculation/BENCHMARK_CAMPAIGN_2026-10-27.md](../plans/serving-and-speculation/BENCHMARK_CAMPAIGN_2026-10-27.md)
- [.agents/plans/serving-and-speculation/FUSED_GPU_BATCHING.md](../plans/serving-and-speculation/FUSED_GPU_BATCHING.md)
- [.agents/plans/serving-and-speculation/NEXT_ITERATION_OPENAI_COMPATIBLE_SERVING.md](../plans/serving-and-speculation/NEXT_ITERATION_OPENAI_COMPATIBLE_SERVING.md)
- [.agents/plans/serving-and-speculation/README.md](../plans/serving-and-speculation/README.md)
- [.agents/plans/serving-and-speculation/TODO.md](../plans/serving-and-speculation/TODO.md)
- [.agents/prior-art/ACCELERATOR_PRIOR_ART_AND_PLATFORM_RESEARCH_2026-10-12.md](../prior-art/ACCELERATOR_PRIOR_ART_AND_PLATFORM_RESEARCH_2026-10-12.md)
- [.agents/prior-art/ARCHITECTURE_RECONCILIATION_AND_AUTHORITY.md](../prior-art/ARCHITECTURE_RECONCILIATION_AND_AUTHORITY.md)
- [.agents/prior-art/ARCHITECTURE_V4_CRITICAL_REVIEW_2026-10-12.md](../prior-art/ARCHITECTURE_V4_CRITICAL_REVIEW_2026-10-12.md)
- [.agents/prior-art/COMPARATIVE_COMPILER_AND_RUNTIME_RESEARCH_2026-10-10.md](../prior-art/COMPARATIVE_COMPILER_AND_RUNTIME_RESEARCH_2026-10-10.md)
- [.agents/prior-art/DISTRIBUTED_PRIOR_ART_AND_TUNNEL_RESEARCH_2026-10-12.md](../prior-art/DISTRIBUTED_PRIOR_ART_AND_TUNNEL_RESEARCH_2026-10-12.md)
- [.agents/prior-art/FINAL_ARCHITECTURE_REVIEW_2026-10-12.md](../prior-art/FINAL_ARCHITECTURE_REVIEW_2026-10-12.md)
- [.agents/prior-art/GUFO_FORK_AND_XDNA2_PREFILL_SOURCE_RESEARCH_2026-10-12.md](../prior-art/GUFO_FORK_AND_XDNA2_PREFILL_SOURCE_RESEARCH_2026-10-12.md)
- [.agents/prior-art/MULTI_MODEL_ROUTER_AND_API_PRIOR_ART_RESEARCH_2026-10-12.md](../prior-art/MULTI_MODEL_ROUTER_AND_API_PRIOR_ART_RESEARCH_2026-10-12.md)
- [.agents/prior-art/PRIOR_ART_LICENSE_AND_PROVENANCE_POLICY.md](../prior-art/PRIOR_ART_LICENSE_AND_PROVENANCE_POLICY.md)
- [.agents/prior-art/RESEARCH_SOURCE_ATLAS.md](../prior-art/RESEARCH_SOURCE_ATLAS.md)
- [.agents/prior-art/SSD_STREAMING_AND_RELATED_PRIOR_ART_2026-10-15.md](../prior-art/SSD_STREAMING_AND_RELATED_PRIOR_ART_2026-10-15.md)
- [.agents/prior-art/VLLM_CONCURRENCY_PRIOR_ART_AND_ARCH_RECONCILIATION_2026-10-15.md](../prior-art/VLLM_CONCURRENCY_PRIOR_ART_AND_ARCH_RECONCILIATION_2026-10-15.md)
- [.agents/research/EXTREME_MODEL_SPECIALIZATION_INTEGRATION_AND_ACCEPTANCE_2026-10-15.md](../research/EXTREME_MODEL_SPECIALIZATION_INTEGRATION_AND_ACCEPTANCE_2026-10-15.md)
- [.agents/research/GEMMA4_FAMILY_CONTRACTS_2026-10-16.md](../research/GEMMA4_FAMILY_CONTRACTS_2026-10-16.md)
- [.agents/research/GPU_TEXT_FRONTEND_AND_DEVICE_ORCHESTRATION_ROADMAP_2026-10-16.md](../research/GPU_TEXT_FRONTEND_AND_DEVICE_ORCHESTRATION_ROADMAP_2026-10-16.md)
- [.agents/research/HETEROGENEOUS_PREFILL_AND_REPLAY_ACCEPTANCE_2026-10-15.md](../research/HETEROGENEOUS_PREFILL_AND_REPLAY_ACCEPTANCE_2026-10-15.md)
- [.agents/research/INITIAL_HARDWARE_TARGETS_AND_OPTIMIZATION_2026-10-16.md](../research/INITIAL_HARDWARE_TARGETS_AND_OPTIMIZATION_2026-10-16.md)
- [.agents/research/MODEL_PULL_XDG_CACHE_AND_SOURCE_PARITY_2026-10-16.md](../research/MODEL_PULL_XDG_CACHE_AND_SOURCE_PARITY_2026-10-16.md)
- [.agents/research/MULTIMODAL_MODEL_FAMILY_SOURCE_ATLAS_AND_IMPLEMENTATION_2026-10-14.md](../research/MULTIMODAL_MODEL_FAMILY_SOURCE_ATLAS_AND_IMPLEMENTATION_2026-10-14.md)
- [.agents/research/MULTIMODAL_V5_1_ACCEPTANCE_AND_ROADMAP_2026-10-15.md](../research/MULTIMODAL_V5_1_ACCEPTANCE_AND_ROADMAP_2026-10-15.md)
- [.agents/research/MULTIMODAL_V5_1_DECISIONS_AND_PROVENANCE_2026-10-15.md](../research/MULTIMODAL_V5_1_DECISIONS_AND_PROVENANCE_2026-10-15.md)
- [.agents/research/NATIVE_FP32_FMA_QUALIFICATION_PROPOSAL_2026-10-16.md](../research/NATIVE_FP32_FMA_QUALIFICATION_PROPOSAL_2026-10-16.md)
- [.agents/research/PULL_A1_LOCAL_XDG_CACHE_2026-10-16.md](../research/PULL_A1_LOCAL_XDG_CACHE_2026-10-16.md)
- [.agents/research/QWEN_BASICS_PARITY_AND_ADVANCED_DECODING_ROADMAP_2026-10-16.md](../research/QWEN_BASICS_PARITY_AND_ADVANCED_DECODING_ROADMAP_2026-10-16.md)
- [.agents/research/README_PRE_PRODUCT_REWRITE_SNAPSHOT_2026-10-15.md](../research/README_PRE_PRODUCT_REWRITE_SNAPSHOT_2026-10-15.md)
- [.agents/research/SPECIALIZATION_SOURCE_ATLAS_2026-10-15.md](../research/SPECIALIZATION_SOURCE_ATLAS_2026-10-15.md)
- [.agents/research/SSD_STREAMING_IMPLEMENTATION_ROADMAP_AND_ACCEPTANCE_2026-10-15.md](../research/SSD_STREAMING_IMPLEMENTATION_ROADMAP_AND_ACCEPTANCE_2026-10-15.md)
- [.agents/research/STRIX_HALO_OPTIMIZATION_DOSSIER.md](../research/STRIX_HALO_OPTIMIZATION_DOSSIER.md)
- [.agents/research/VLLM_PARITY_AND_COMBINATORY_SPECIALIZATION_ACCEPTANCE_2026-10-15.md](../research/VLLM_PARITY_AND_COMBINATORY_SPECIALIZATION_ACCEPTANCE_2026-10-15.md)
- [.agents/research/multimodal/ASR_AND_STREAMING_SPEECH_UNDERSTANDING_2026-10-15.md](../research/multimodal/ASR_AND_STREAMING_SPEECH_UNDERSTANDING_2026-10-15.md)
- [.agents/research/multimodal/CROSS_STREAM_ARCHITECTURE_RECONCILIATION_2026-10-15.md](../research/multimodal/CROSS_STREAM_ARCHITECTURE_RECONCILIATION_2026-10-15.md)
- [.agents/research/multimodal/DECISION_MODELS_TERNARY_AND_SPECIALIST_TEXT_2026-10-15.md](../research/multimodal/DECISION_MODELS_TERNARY_AND_SPECIALIST_TEXT_2026-10-15.md)
- [.agents/research/multimodal/IMAGE_GENERATION_EDITING_AND_FLOW_MODELS_2026-10-15.md](../research/multimodal/IMAGE_GENERATION_EDITING_AND_FLOW_MODELS_2026-10-15.md)
- [.agents/research/multimodal/SERVING_API_AND_REALTIME_PRIOR_ART_2026-10-15.md](../research/multimodal/SERVING_API_AND_REALTIME_PRIOR_ART_2026-10-15.md)
- [.agents/research/multimodal/TTS_VOICE_CODECS_AND_FULL_DUPLEX_2026-10-15.md](../research/multimodal/TTS_VOICE_CODECS_AND_FULL_DUPLEX_2026-10-15.md)
- [.agents/research/multimodal/VIDEO_TEMPORAL_MODELS_AND_SYNCHRONIZED_AV_2026-10-15.md](../research/multimodal/VIDEO_TEMPORAL_MODELS_AND_SYNCHRONIZED_AV_2026-10-15.md)
- [docs/CLI.md](../../docs/CLI.md)
- [docs/DEVELOPMENT.md](../../docs/DEVELOPMENT.md)
- [docs/ENGINEERING_SPECIFICATION.md](../../docs/ENGINEERING_SPECIFICATION.md)
- [docs/PUBLICATION_AND_REVIEW_CHECKLIST.md](../../docs/PUBLICATION_AND_REVIEW_CHECKLIST.md)
- [docs/README.md](../../docs/README.md)
- [docs/REFERENCES.md](../../docs/REFERENCES.md)
- [docs/WHITEPAPER.md](../../docs/WHITEPAPER.md)
- [docs/architecture-overview.svg](../../docs/architecture-overview.svg)
- [.agents/docs/FULL_COMPREHENSIVE_HANDOFF_2026-11-27_OA4B.md](FULL_COMPREHENSIVE_HANDOFF_2026-11-27_OA4B.md) (this handoff)
- [.agents/docs/NEXT_AGENT_2026-11-27_OA4C.md](NEXT_AGENT_2026-11-27_OA4C.md) (new prompt)

## Appendix B — complete unresolved checklist snapshot

Literal open/active/blocked markers and immediately indented continuations from non-takeover Markdown in Appendix A at 89822f4038b2048affd8b92dac6c7b2bcbeb219e. Grouped by original file and numbered lines. Fenced examples and historical comprehensive/current-handoff/baseline/next-agent files excluded (indexed in A/E). Historical designs may include unchecked markers; not fresh authorization. The full linked owner prose and tests govern, including prose-only work.

### [.agents/docs/SECURITY_AND_RELIABILITY.md](SECURITY_AND_RELIABILITY.md) — 12 unresolved markers

~~~~text
104: - [ ] Defined model file trust and network resolver policy.
105: - [ ] All external source/data licenses reviewed.
106: - [ ] No unchecked parser arithmetic or format downgrades.
107: - [ ] Public HTTP exposure is explicit and protected.
108: - [ ] Model-state snapshots private, bounded and versioned.
109: - [ ] Physical RAM admission and APU watchdog exercised on hardware.
110: - [ ] Fault injection on expert slots, snapshots, kernels and cancellation.
111: - [ ] No execution of arbitrary Python model/plugin scripts.
112: - [ ] Distributed transport identity/credential and fail-closed policy.
113: - [ ] GPU memory/FFI unsafe blocks audited and sanitizer tests recorded.
114: - [ ] Secrets/model traces excluded from CI and release artifacts.
115: - [ ] Performance and quality claims link reproducible evidence.
~~~~

### [.agents/plans/FULL_IMPLEMENTATION_PLAN.md](../plans/FULL_IMPLEMENTATION_PLAN.md) — 40 unresolved markers

~~~~text
354: - [ ] Reconcile model/source/format/codec/task registration and crate boundaries
355:   against the live Fabric transition. Keep family-specific mathematics in
356:   model crates, shared mathematical operators in core/shared libraries, and
357:   physical execution in backends. Remove only obsolete wiring; retain useful
358:   correct primitives and numerical variants.
359: - [ ] Preserve one authoritative source-bound `ModelProgram` / `ForwardPlan`
360:   and prepared physical executable. Typed values, state effects, layouts,
361:   regions and task dependencies determine valid execution. Distinguish a
362:   mathematical region, a native ModelAssembly and a multi-model TaskPipeline.
363:   Recursive composition is a semantic/planning view, not a recursive second
364:   runtime.
365: - [ ] Complete the existing `RegionExecutionProvider` / `PreparedExecutable`
366:   / `ExecutableBundle` seam for closed regions and whole-model regions.
367:   Preserve the original model/source identity, numerical contract, shape
368:   constraints, input/output bindings and state effects. No arbitrary
369:   subgraph replacement that bypasses the authoritative plan.
370: - [ ] Separate cold policy from physical capability: explicit backend,
371:   implementation/provider identity, numerical/approximation profile,
372:   execution mode and resolved region choices. Select a qualified executable
373:   at preparation or an explicitly supported committed-root boundary. Shape
374:   misses use a qualified pre-enqueue decomposition or refuse. Submitted
375:   failures cannot be retried as another provider on the same root.
413: - [ ] Generalize only proven shared mathematical contracts: keep the dense
414:   seam and add bounded native row-wise, normalization/gating, rotary, attention,
415:   recurrent, convolution and quantized operators as real consumers require.
416:   Keep compact row/batch layouts and explicit precision; avoid forcing every
417:   operator into GEMM or a universal dispatch object.
429: - [ ] Make borrowed device views and external execution/completion integration
430:   possible without promising universal zero-copy interop. A provider may
431:   plan/reuse local intermediate storage inside its admitted region. When it
432:   cannot import caller buffers or use the original stream, admit its internal
433:   memory and actual synchronization/transfers explicitly. No provider may
434:   publish StateRoot, schedule independent sessions or hide unaccounted memory.
435: - [ ] Keep provider features independently optional. Native execution and all
436:   primary model loading remain usable with no graph compiler installed.
437:   Concrete MIGraphX/TensorRT/IREE/ONNX Runtime build, runtime bindings and
438:   model/region support are Phase 6 work, not foundation exit requirements.
442: - [ ] Complete shared source acquisition/import/cache, verified shard/index
443:   and tensor-window identities, license/authorization, exact snapshot
444:   selection, config/tokenizer/processor/adapter validation and finite
445:   encoding/codec support. Preserve HF/Xet, GGUF, local and existing remote
446:   acquisition mechanisms; unsupported providers/format variants refuse
447:   explicitly. Native weights load without ONNX conversion or remote model code.
448: - [ ] Preserve one catalog/SDK source-lifecycle authority, transactional cache
449:   admission, cancellation, download resume, last-reader retirement and safe
450:   cleanup. Retain source provenance, checked metadata and byte bounds;
451:   avoid duplicate allocations for aliases and components. Make actual file
452:   descriptors, mappings and immutable readers the authority.
453: - [ ] Consolidate pre-Client fit/admission, stable device identity and source
454:   eligibility. Derive complete cold/prefill/decode/graph/provider/state/
455:   output cost and headroom from real geometry/physical observations; keep
456:   UMA overlap and host limits explicit. A typed capacity miss is distinct
457:   from a lifecycle error. No silent CPU/backend/profile substitution after
458:   work is submitted.
462: - [ ] Close shared physical backing/range identity, original leases,
463:   immutable source readers, actual readiness and event-backed completion/
464:   quarantine. One ResourceArbiter; optional local allocator strategies and
465:   provider workspaces may vary under its authority. No global reset or
466:   reclamation of unrelated owners.
467: - [ ] Complete the common independent session-state and candidate-publication
468:   contract across KV, recurrent/conv and all required state domains.
469:   Paged/hybrid state, complete-domain prefix reuse and COW, bounded snapshots,
470:   cancellation/recovery and cache compatibility use the same StateRoot.
471:   Only actual completion and the original authority publish; unknown work
472:   retains all affected owners and fences the unsafe service.
473: - [ ] Complete shared resident/paged/streamed tensor materialization and
474:   transfer-retirement boundaries. Preserve resident direct access with no
475:   mandatory host dispatch; actual SSD windows, page/weight cache, prefetch,
476:   bounded async IO and eviction policies mature in later native workloads.
477: - [ ] Finish model-neutral invocation/cohort, fair worker, batching/refill,
478:   backpressure, cancellation and task/pipeline control seams. Continuous
479:   batching, chunked/ragged prefill, speculation/MTP and multi-model placement
480:   share these owners. Do not grant unqualified execution modes from policy.
481: - [ ] Preserve typed media ports/clocks, realtime lanes and remote placement/
482:   transport boundaries without implementing every media codec or network
483:   topology now. Local single-process execution requires no network.
487: - [ ] Complete public Rust SDK and standalone serving library contracts;
488:   CLI/TUI/Web use the same runtime, catalog, configuration and control
489:   events, not a separate frontend-owned execution path. Keep model and
490:   backend features orthogonal, ordinary `all` free of diagnostic features,
491:   backend-only/model-only/serve-only dependency isolation and mixed GPU
492:   development shell. Integrate provider capabilities without requiring
493:   optional SDKs in default CPU builds.
497: - [ ] Close remaining applicable architecture/feature/test-boundary debt.
498:   Keep source, model arithmetic,
499:   quality, cancellation, physical readiness, state/root and real execution
500:   checks distinct. Qualify the existing working native paths; CUDA unresolved
501:   parity is an explicit gate before CUDA performance, not a pretext for
502:   starting its optimization campaign ahead of ROCm.
503: - [ ] Expose a common effective configuration, capabilities, resource
504:   snapshot, plan/region identity, structured error and execution metrics
505:   contract. Keep telemetry on demand, bounded and out of hot-path ownership
506:   decisions where possible. The UI builds on this rather than deriving
507:   semantics from logs.
800: - [ ] Close the [OpenAI-compatible serving iteration](serving-and-speculation/NEXT_ITERATION_OPENAI_COMPATIBLE_SERVING.md):
801:   model catalog/router, OpenAI Chat/Responses and supported completion/
802:   embedding/tool/structured-output contracts, usage, errors, JSON/SSE,
803:   streaming/cancellation/timeout/backpressure and real concurrent requests.
804:   Unsupported tasks/options must refuse rather than simulate compatibility.
805: - [ ] Preserve a separate control surface and inference API with typed
806:   configuration, explicit route/tenant scope, model lifecycle and shared
807:   scheduling. Add future WebSocket/WebRTC/media endpoints through the
808:   existing task contracts; do not wait on unimplemented media models.
809: - [ ] Establish ordinary public CLI/SDK/server benchmark invocation,
810:   effective config and end-to-end metrics, not a qualification-only serving
811:   process. Retain one process/multiple model instances and pipeline routing.
828: - [ ] Restore the approved pinned optimized vLLM provider and establish
829:   fresh matched original-source 0.8B/2B/27B baselines. Use retained V2
830:   results for engineering direction, not fresh comparisons. Get quick real
831:   C1/C8 feedback; extend the declared campaign after working results.
832: - [ ] Native row-wise and attention regions: eliminate unnecessary intermediate
833:   materialization and exact-reference scalar passes within explicitly
834:   qualified native arithmetic. Compose complete normalization/residual,
835:   gating, QK/rotary, attention and recurrent regions; reuse mature native
836:   kernels/providers where capability and model geometry fit. Keep GDN
837:   solve/products and source/reference implementations.
838: - [ ] Use rocBLAS/hipBLASLt for dense products, appropriate CK, MIOpen and
839:   mature attention/flash-decode kernels, model-specific kernels where needed.
840:   Select measured GEMV/GEMM, quantized and batch shapes, not one blanket
841:   provider choice. Native command capture and larger execution regions
842:   amortize host/actor dispatch, without requiring MIGraphX.
843: - [ ] Improve genuine shared prefill, independent stateful cohort work,
844:   continuous ragged refill/regroup, paged/hybrid KV and prefix/COW.
845:   Device-resident sampling/feedback, overlap, fair scheduling, native
846:   graph execution where material. Qualify cancellation and state effects.
847: - [ ] Scale native resident and streamed weights, long context, real
848:   supported low-bit GGUF/source quantization, KV formats and SSD execution.
849:   Keep exact/source/native profiles separate; quantify quality against
850:   reference and compare equivalent precision/workload. Extend family support
851:   through existing registries and source bindings, not separate model loops.
852: - [ ] Close the matched performance acceptance below on the required feasible
853:   AMD cells. Do not wait until the entire grid is green to measure and fix
854:   the dominant gaps. No actual graph compiler integration in this phase.
867: - [ ] Revalidate actual current NVIDIA source paths before performance claims.
868:   Close deterministic C4/B8 identical-lane divergence, independent populated
869:   all-state-domain correctness and missing approved active-context/2B
870:   references. Preserve prior physical acceptance only within its scope.
871: - [ ] Promote the existing owned cuBLAS/cuBLASLt dense adapter into ordinary
872:   execution through shared geometry and original physical owners. Use native
873:   tensor-core/CUTLASS and FlashAttention/FlashInfer where they win; preserve
874:   existing specialized GDN/state and transport alternatives.
875: - [ ] Reuse ROCm phase's mathematical regions and native execution improvements;
876:   tune CUDA-specific attention/paging, continuous batching, compact
877:   device-resident sampling, larger captured graphs and dispatch amortization
878:   only from actual measurements. Keep pre-enqueue qualified misses distinct
879:   from unknown native submissions. Production selection must be physically
880:   accepted, not inferred from diagnostic graph/timing support.
881: - [ ] Run matched optimized vLLM through the same public serving campaign on
882:   approved NVIDIA devices and feasible models/context/concurrency. Extend
883:   source/GGUF model coverage, native quantization and residency according to
884:   measured capacity. TensorRT remains optional Phase 6 work.
893: - [ ] Reuse shared typed operations, region/shape contracts and device
894:   lifetime semantics. Close actual Vulkan model execution, independent
895:   recurrent/KV state, prefill/decode, cohort and resource safety, then
896:   required CPU/CUDA/HIP-compatible capabilities where available.
897: - [ ] Evaluate mature ggml Vulkan cooperative-matrix GEMM implementations,
898:   capability-gated VK_KHR_cooperative_matrix and vendor variants, portable
899:   compute shaders for unsupported geometries, and applicable optimized
900:   quantized/attention kernels. Audit license/provenance and actual backend
901:   layout compatibility before reuse. No fictional universal Vulkan BLAS
902:   and no mandatory compute-engine fork.
903: - [ ] Compare real model latency, throughput, memory and quality against
904:   native Vulkan baselines (e.g. llama.cpp/ggml) on the same device/format;
905:   use matching native/vLLM hardware comparisons only where meaningful.
906:   No unsupported vLLM Vulkan baseline. Extend Linux and other targets by
907:   actual capability and package qualification; retain clean backend feature
908:   isolation. IREE integration is Phase 6.
932: - [ ] MIGraphX on HIP: lower native semantic regions through programmatic
933:   graph construction, qualify runtime/hardware and external device arguments/
934:   stream interop, compile/prepare a closed FFN or other dense region, then
935:   compare native and compiled paths at actual model/serving level. Consider
936:   whole encoders/embeddings/ASR where source format and model semantics fit.
937: - [ ] TensorRT on CUDA: lower native semantic regions through programmatic
938:   network construction, same closed-region/whole-model boundary, compatible
939:   caller-bound tensor addresses/context memory/stream, admitted engine
940:   compilation/cache and explicit supported shape/profile. Benchmark a real
941:   region or conventional model, not another complete LLM serving engine.
942: - [ ] IREE: evaluate compiler/HAL interop on supported device targets, in
943:   particular Vulkan, and compare with the native cooperative-matrix path.
944:   Verify actual import/export/synchronization before promising zero-copy.
945: - [ ] Optional graph import: authenticate ONNX graphs and external weights;
946:   establish their typed region, numerical and effect contracts through the
947:   existing source/preparation boundary. ONNX Runtime executes supported
948:   conventional models, encoder/vision/audio/embedding graphs, with device
949:   I/O binding or explicitly budgeted materialization. MIGraphX/TensorRT
950:   import support is independently qualified; it is not a prerequisite for
951:   direct native-region lowering. Other graph accelerators/TVM only when a
952:   concrete workload justifies them.
953: - [ ] Native/compiled selection happens at admitted preparation, with source,
954:   numerical/quality, layout, shapes, device/runtime, actual costs and state
955:   compatibility explicit. Bound compilation/autotune/artifact caches and
956:   retain useful alternatives. Require actual external-resource lifetime and
957:   completion; a failed submission never falls back by hiding its failure.
958: - [ ] Preserve one engine, state authority, resource fabric and model/pipeline
959:   interfaces. Whole-model opaque managed remote APIs are TaskService adapters,
960:   not native kernels or proof of local physical execution.
~~~~

### [.agents/plans/TODO.md](../plans/TODO.md) — 11 unresolved markers

~~~~text
717: - [ ] P1 serving-and-speculation — @pvl — l5 CUDA `.8B`/`2B`, x2 ROCm `.8B`/`2B`/`27B`: source-real correctness/paged/prefix/fused 2/4/8-row cohorts accepted; multi-token prefill + autoregressive refill/regroup now fast under matched opt1. Public CUDA serving passes4K/8K/16K on both small models,0.8B32K has one full pass, and paged2B32K passes32766 tokens in224.80s. Same32K-capacity1024-token A/B is283.22s contiguous-streamed vs28.56s paged-resident (~9.9x). Generic cold-start smart config now jointly resolves Qwen `state_policy` + `cohort_width` from fresh runtime admission: Exclusive=ignore user knobs, Enabled(default)=user values override Auto per knob, Disabled=no Auto; score=resident weights > wider 8/4-row cohort > Contiguous tie. Default2B32K/1024 Auto-selects paged and passes in26.51s exact. Existing `active=8` SDK with cohort width unspecified Auto-selects8 for both Contiguous/user-Paged halves and publishes `{8}` at135.979/133.004 tok/s exact. Nix-realized vLLM0.33.0/cu130 correctness now passes exact local0.8B/2B on L5 without Podman; the frozen wheel graph is store-realized and an optimized CUDA-graph smoke passes, so the retained concurrency=1/4/8/16/32/64 paired matrix is next on L5 before the exact-commit X2 ROCm campaign. Independent populated all-state-domain closure remains open; X2 ROCm remains environment-gated — [details](serving-and-speculation/TODO.md)
718: - [~] P1 numerics-and-context — @pvl — exact prefill 47 + decode 48–50, lifecycle/capacity and final gate pass; active-context/2B reference inputs missing — [details](numerics-and-context/TODO.md)
719: - [~] P1 memory-and-streaming — @pvl — 32K storage/execution policy is materially closed:0.8B32K has one full public-SDK pass;2B32K passes the full32766-token retained cell with Paged64/64, and canonical default startup Auto-selects paged when contiguous would stream. Final CUDA admission independently clamps provider capacity to original-actor live free VRAM while preserving configured headroom. CUDA-forward source-page transfer now retains the verified `TensorWindow` itself through the existing checkpoint and uses the original actor's bounded owned-window H2D directly, removing the superseded pinned/Bytes copy chain. Matched 2B/32K prompt48 streamed feedback is38.46s vs39.36s at `b839920`; cache-promotion41.00s, moved-pinned44.88s and direct-SSD/cache-bypass41.77s variants were rejected. A same-tree one-page CUDA retirement-ring Qwen A/B is rejected at39.281s vs38.323s serial (~2.5% slower); production remains serial at that boundary and default smart config avoids it when paged state enables resident weights. The model-neutral primitive remains dormant/reusable across backends: CUDA `OwnedCudaEvent` and HIP `OwnedHipTransferRetirement` implement `RetirementMark`, with zero-cost CUDA/HIP ring aliases over the shared `TransferRetirementRing`. Shared completed-root reuse, allocation-free common-frontier reconciliation, exact partial-tail paged COW ownership and canonical physical allocation+range identity are active; CUDA-forward prepared sets, streamed-head transfer binding and production HIP resident-text workspace/root binding use canonical ledger identity. Remaining work is additional real identity consumers plus cache-aware direct/async IO and broader authenticated streaming tiers — [details](memory-and-streaming/TODO.md)
720: - [~] P1 cpu-gpu-optimization — @pvl — forward kernel/transport checks pass; exact whole-model parity before measured tuning and the matched benchmark campaign — [details](cpu-gpu-optimization/TODO.md)
721: - [~] P1 pipeline-composition — @pvl — recursive primitives/pipelines/profiles; shared execution adapters active, reuse current adapters in Fabric; Qwen expansion deferred — [details](pipeline-composition/TODO.md)
722: - [~] P1 cli-device-routing — @pvl — unified run/config/source/device selection; preserve behavior through Fabric SDK/CLI migration — [details](cli-device-routing/TODO.md)
723: - [~] P2 model-execution — @pvl — current Qwen CUDA forward owns source-real fused 2/4/8-row contiguous+paged transport, complete-domain prefix reuse, event-backed/multi-flight primitives, retained stateless + multi-token attention/GDN prefill, reduced 4/8-row GDN layout selection and bounded paired-page KV gather with fused inactive-tail fill. Latest matched opt1 short-wave is135.236 tok/s contiguous/132.949 paged (~1.7% wall-time gap); current68-token refill is169.837/160.919 tok/s. Rejected A/B mechanics are retained dormant rather than deleted: lane regroup, copy+tail-zero,17 BF16 GEMM strategies (including restored terms64 shared-row, paired-boundary and K64 weight-cache variants), recurrent tiled transpose and GDN occupancy variants, with zero production consumers. The restoration sweep also repairs default/qualification/CUDA-native test cfg drift; the largest native-qualification lane is external-SIGKILL incomplete. Public graph serving deliberately remains Native. Large same-contract CUDA numerical acceptance still depends on approved references; populated small HIP and resident27B work remain separate. X2 is shared: preserve other owners and verify only ZeroGravity's incremental resource delta retires — [details](model-execution/TODO.md)
724: - [~] P2 model-acquisition — @pvl — [details](model-acquisition/TODO.md)
728: - [ ] multimodal — [details](multimodal/TODO.md)
729: - [ ] distributed-execution — [details](distributed-execution/TODO.md)
730: - [ ] platform-and-release — [details](platform-and-release/TODO.md)
~~~~

### [.agents/plans/cli-device-routing/TODO.md](../plans/cli-device-routing/TODO.md) — 64 unresolved markers

~~~~text
544: - [ ] Optional deeper exact per-request/phase and individual layer
545:   observations; keep profiling explicit and capability-bound.
546: - [ ] Optional terminal background helper only with dependency/offline
547:   approval; conservative Auto fallback must remain truthful. No additional
548:   theme catalog or persistent layout store now.
550: - [ ] Broader catalog presets, conversation history, dedicated graph
551:   visualization and unsupported routes remain explicit future scope.
552:   Existing direct text/JSON structural inspection is accepted.
553: - [ ] Extend advanced direct provider forced interactive/JSONL presentation
554:   and aggregate component download control without changing independently
555:   published receipts; existing direct provider progress and legacy sequential
556:   component executor remain valid. No general concurrent download scheduler,
557:   persistent event history or automatic partial deletion introduced.
563: - [~] Shared pre-Client transactional admission: accepted `f4746c23` adopts
564:   the original envelope before HIP source Client/runtime bootstrap; actual
565:   Qwen refusal/retry passes without leaked reservation or fencing.
566:   9ae6dd1e/e06213a5 now share exact context-free CUDA PCI and original
567:   configured provider reservation before Client on ordinary/pinned Linux
568:   CUDA routes, actual identity adoption and original live downward clamp
569:   with unused cold credit released. Real Qwen HF peer refusal (no new
570:   observed compute owner)/retry, pinned live fit and retirement, HF C1,
571:   GGUF pipeline/selective unload and HIP V2 C1/C8 pass. Existing Vulkan
572:   diagnostic attachment compiles. Logical pre-admission is conservative;
573:   current free memory still observed on actor after bootstrap when
574:   necessary. cbd2c119/197c2e26 add original shared host-backing constraint,
575:   immutable scoped caps, latest visible cgroup-v2 hard constraint, shared
576:   cold fit and real HIP local VRAM/driver intersection (no host/GTT sum).
577:   Original host lifetime upper remains logical; check actual next cold
578:   allocations. a6548863 extends original preadmission to ordinary Linux HIP
579:   and guarded native HIP; guarded exact-adapter Vulkan intersects local VRAM
580:   total/current use before original registration. Equal declared physical
581:   roles return no inter-domain transfer; distinct domains explicitly refuse.
582:   3b5a520d/fc38db03/e1decff5 close bounded explicit named Linux Vulkan
583:   registry reuse and physical admission: actual peer refusal/raced/warm
584:   reuse, independent envelope retirement and portable healthy/unknown.
585:   14bbbd48/5d952364/5a7535e7 qualify actual coherent mapped Vulkan
586:   producer on the original range and charge. No implicit zero-copy from
587:   colocated placement. Non-Linux, general host-visible GPU backing
588:   aliasing and OS hard-memory guarantees remain open.
589:   Minimum shared foundation exit review in the full plan; shared CLI
590:   presentation (UX-1) now accepted above, preserving the original runtime/selector.
594: - [ ] Recheck live source, ownership and accepted model/backend evidence before
595:   implementation; preserve root README, other plans and active implementation.
599: - [ ] Accept local paths, catalog/model names, `hf:`, `gh:`, `http:` and `https:`
600:   through one shared source selector and the existing acquisition/cache identity
601:   contracts. Recognized source schemes take precedence over local YAML suffix
602:   classification; only a local `.yaml`/`.yml` selector implies a run document.
603: - [ ] Resolve already acquired immutable exact sources, reject ambiguous variants
604:   and preserve explicit acquisition guidance when absent. URI parsing/resolution
605:   is not inference qualification. Reuse provider parsing/source identity rather
606:   than a second downloader or format-specific run loop; dry-run stays read-only.
607: - [ ] Every new CLI semantic option has a typed v2 field. Unit1 implementation
608:   and tests precede ROOT guarded fresh-mtime merge; final assembled gate remains
609:   required. Future split/NPU/remote units retain separate qualification.
617: - [ ] Normalize `run`, backend aliases and common task flags into one library
618:   request; preserve compatible existing invocations and actionable diagnostics.
619: - [ ] Extend the canonical pipeline document/compiler to proposed v2 with one
620:   model table, explicit model/pipeline entrypoint and typed run policy/task/
621:   diagnostics; do not introduce a separate config interpreter or runtime.
622: - [ ] Implement unambiguous positional model/config and explicit selectors;
623:   selector-free run loads only absolute-XDG/HOME `abird/zerogravity/config.yaml`.
624:   Missing/unreadable/invalid config fails without fallback or initialization.
625: - [ ] Implement CLI > selected config > built-in resolution with per-field
626:   provenance, origin-correct paths, whole-list replacement and explicit boolean/
627:   zero/empty semantics. Preserve alias equivalence and final constraint checks.
628: - [ ] Migrate model metadata to `--model-config`; cover every existing task,
629:   tokenizer, teacher-ID, prefill and audit option with typed config equivalents.
630: - [ ] Retain bounded strict YAML/graph validation; check the complete merged
631:   document before effects. Declared routes remain inert under run; graph and
632:   recursive execution still need their independently qualified implementation.
633: - [ ] Resolve exact local source and bounded task/model requirements through
634:   existing adapters; catalog/metadata status grants no executable capability.
635: - [ ] Add read-only inventory and stable physical identity resolution; derive
636:   actual vendor/backend support and deduplicate cross-backend aliases.
637: - [ ] Implement strict ordered allowlists, CPU-terminal semantics, local-only
638:   auto and all-neural-stage forced-backend checks before preparation.
639: - [ ] Implement current-memory observation with explicit provenance/age/unknown
640:   status, OS/cgroup constraints and host/UMA domain relationships. Do not reuse
641:   `max_memory` as free memory or mutate established operator ceilings.
642: - [ ] Resolve a safe default operator-ceiling/bootstrap policy once for a new
643:   process before context effects, so bare `run` needs no manual resource caps;
644:   preserve established domains and benchmark/serving controls.
645: - [ ] Integrate complete cold/prefill/decode/native/compiler/state/output costs
646:   with original ledger snapshots and authentic existing backing/reuse. Close or
647:   expose pre-admission context-bootstrap gaps; unknown costs cannot claim fit.
648: - [ ] Implement configurable headroom, all-GPU full-fit scan and qualified CPU
649:   fallback; explain missing split support when a model would otherwise spill.
650: - [ ] Revalidate and admit through original owners; bound safe startup replanning,
651:   preserve unknown-work fences and retain all submitted owners.
652: - [ ] Provide `devices list`, `--dry-run`, `--explain` and structured refusals
653:   without hidden execution/network/cache mutation during diagnostics.
654: - [ ] Validate the assembled unit on approved source/hardware/backend tuples;
655:   document actual supported tuples and unavailable current-resource probes.
664: - [ ] Define source-specific indivisible groups and embedding/head/shared-weight/
665:   MTP ownership; produce one complete feasible placement before any preparation.
666: - [ ] Extend original arbiter with all-or-none multi-domain admission if needed;
667:   collapse aliases and UMA domains, preserve prepaid-envelope/sublease rules.
668: - [ ] Integrate original state shards, typed activations and actual event/reader
669:   transfers into the existing prepared physical plan and fair workers; join all
670:   Ready events before the original SessionAuthority publishes one logical root.
671: - [ ] Prove same-backend multi-GPU and GPU↔CPU compositions separately before
672:   enabling each; qualify mixed CUDA/ROCm/Vulkan only with its own evidence.
673: - [ ] Verify full-logit/complete-state following-token parity, cancellation,
674:   partial preparation, transfer failure, unknown work and complete retirement.
675: - [ ] Compare full CPU and each feasible single/split route with complete cold/
676:   warm request costs; show capacity enabling separately from speedup.
682: - [ ] Introduce semantic stage-role constraints without model-family CLI routing
683:   branches; preserve full-run forced-backend policy and common allowlist.
684: - [ ] Qualify actual trained draft/MTP source and recursive state semantics;
685:   distinguish native MTP depth from supported proposal length.
686: - [ ] Add an NPU provider only through existing region/state/lifetime boundaries;
687:   prove end-to-end benefit versus available GPU draft and no-speculation routes,
688:   including transfer/verification/rejection/shared-domain costs.
689: - [ ] Resolve explicitly enrolled peer aliases to authenticated identities and
690:   capability/topology epochs; standalone auto performs no remote discovery.
691: - [ ] Coordinate bounded all-or-none holds through original per-host ledgers;
692:   implement authenticated transfers and exact state shard publication/fencing.
693: - [ ] Qualify first two-host contiguous layer/local-state composition on approved
694:   sources/hosts; report directional network/copy/crypto/queue costs and loss.
695: - [ ] Keep remote inference APIs that hide region/state contracts outside native
696:   device routing; no SSH/shell or implicit artifact downloads for placement.
704: - [ ] **Syntax equivalence:** canonical/alias/repeated flags normalize identically;
705:   conflicting backend, mixed `auto`, empty device, unknown host/ID and CPU before
706:   GPU refuse. Unsupported backend is reported without model preparation.
707: - [ ] **Default config:** present/missing/unreadable/invalid default; absolute,
708:   relative and unset XDG base; valid/absent HOME fallback; no cwd/parent search.
709:   Explicit model/config/graph skips default overlay; help/version do not load it.
710: - [ ] **Config discrimination:** YAML suffix, extensionless explicit --config,
711:   --model, odd model filename, malformed/unsupported schema, multiple positional
712:   selectors and old model-JSON --config produce the specified result; no model/
713:   config fallback, automatic pull, initialization or side effects.
714: - [ ] **Config/CLI parity:** direct YAML and flags yield equal typed requests;
715:   graph YAML and declarations share one instance table/compiler. Source metadata,
716:   tokenizer, prompt/context/output/raw/teacher/prefill/audit/numerical/speculation/
717:   role/device/diagnostic fields and semantic validation round-trip equally.
718: - [ ] **Precedence/provenance:** CLI scalar wins, alias equals canonical backend,
719:   default only fills absent values; false/zero/empty/auto are preserved and
720:   validated. Repeated CLI devices replace config list as a whole; graph list
721:   replacement/clear detects dangling references and duplicate IDs.
722: - [ ] **Path bases:** config-origin source/sidecar/audit paths use selected config
723:   directory; CLI replacements use invocation cwd. Moving cwd, config symlinks and
724:   same relative spelling with different origins cannot silently retarget fields.
725: - [ ] **Config safety/entrypoint:** unknown/duplicate fields, expansion/depth/file
726:   limits, custom tags/includes/interpolation, missing/ambiguous entrypoint,
727:   contradictory final backend/device policy and legacy model-config spelling
728:   refuse before execution. Selected graph runs once; declared routes do not bind.
729: - [ ] **Identity:** CUDA/Vulkan aliases of one card count once; AMD discrete and
730:   NVIDIA integrated classifications cannot invert vendor routing. Snapshot
731:   indices never survive reorder/restart without physical revalidation.
732: - [ ] **Forced scope:** CUDA+AMD rejects before allocation/compiler activity;
733:   forced GPU+CPU spill or incompatible NPU draft rejects; forced CPU runs every
734:   requested neural stage on CPU or refuses a missing capability.
735: - [ ] **Full fit:** both vendors fit→NVIDIA; NVIDIA insufficient/occupied and AMD
736:   fits→AMD; later listed GPU full fit precedes first-card split; GPU-only explicit
737:   list cannot reach CPU. NVIDIA unavailable/fenced is a precise exclusion.
738: - [ ] **CPU fallback:** no GPUs, missing GPU build, unqualified GPU model and no
739:   qualified split each reach CPU only if allowed, qualified and currently fitting.
740:   An unqualified model everywhere gives no invented inference route.
741: - [ ] **Current pressure:** large total VRAM with low free bytes does not fit;
742:   cross-process occupancy, OS/cgroup pressure, missing/stale counters and
743:   fragmentation produce conservative proposals and bounded allocation failures.
744: - [ ] **Accounting:** original leases plus actual backing plus unbacked promises;
745:   prepaid envelope and native partition; idle retained pools; authentic reuse;
746:   unknown/captured readers. No duplicate charge/free credit or cap enlargement.
747: - [ ] **UMA:** CPU/GPU/NPU alias one memory domain; cross-device views/transfers
748:   retain proper fences and charge physical backing once. Headroom applied once.
749: - [ ] **Complete peaks:** weights fit but prefill/KV/output/compiler/native/capture/
750:   draft costs do not; reject. Disjoint phase maxima are not blindly added; actual
751:   overlapped lifetimes, COW, transfer and old/new state are not omitted.
752: - [ ] **Headroom:** percentage/bytes/default/floor/overflow validation; resolved
753:   amount shown; reducing margin never raises a physical/operator ceiling or
754:   waives source/shape/native qualification. Bare-run bootstrap records a safe
755:   default once; later live-free changes never resize the registered ceiling.
756: - [ ] **TOCTOU:** external allocation and concurrent internal admission between
757:   proposal/admission/preparation; all-or-none holds and known-safe rollback;
758:   at most one fresh replan. Dry-run and cached plan never grant reservation.
759: - [ ] **Unsafe preparation/flight:** unknown module load/unload, transfer/capture/
760:   launch retains/fences original owners; no alternate-backend replay, partial
761:   root/output publication or reuse based on cancellation/timeout alone.
762: - [ ] **Split feasibility:** largest legal prefix leaves a feasible remainder;
763:   indivisible tail/tied embedding/head cases; GPU→GPU→CPU exact state and
764:   activation transfer. Unqualified mixed numerics refuse rather than retag.
765: - [ ] **Runtime equivalence:** real full affected logits/tensors, every required
766:   state domain, natural following-token continuation and task outputs match the
767:   selected numerical contract for each admitted source/profile/composition.
768: - [ ] **Fairness/lifetime:** overlapping requests/models preserve independent
769:   roots, original fair progress and complete reader retirement under pressure;
770:   successful cleanup cannot hide an earlier semantic failure.
771: - [ ] **Diagnostics:** list/dry-run leave model, GPU context/compiler, resource
772:   ledger, cache and network unchanged; report unknown rather than probe by
773:   execution. Explain identifies each rejection and final route without corrupting
774:   output or exposing prompts/secrets.
775: - [ ] **NPU role, future:** qualified NPU still loses when transfer/verification/
776:   rejection/queue costs dominate; wins only with applicable full-request benefit.
777:   Stale evidence rejects automatic speed claim; default auto selects ordinary
778:   decode without proven benefit, optional draft-device does not force drafting,
779:   and required MTP/draft/source features are not omitted.
780: - [ ] **Remote, future:** unknown enrollment, identity/epoch mismatch, denied
781:   source/data grant, partial holds, slow/relayed link, late Ready, lost ACK and
782:   uncertain DMA preserve original authority/leases; no implied discovery or HA.
783: - [ ] **Performance:** authorized paired whole-request comparisons, cold/hot
784:   preparation, TTFT/ITL and total latency, memory/IO/transfers and fair load;
785:   compare against CPU/single-GPU baseline and include slower-split negative cases.
786: - [ ] **Final assembled gate:** after applicable unit qualification, run
787:   `CARGO_BUILD_JOBS=2 RAYON_NUM_THREADS=4 python3 scripts/native-gpu.py -- bash scripts/check.sh`.
788:   This design-only draft does not run the gate or any GPU/model tests.
~~~~

### [.agents/plans/cli-ux/LIVE_MODEL_PERFORMANCE.md](../plans/cli-ux/LIVE_MODEL_PERFORMANCE.md) — 1 unresolved markers

~~~~text
153: - [ ] UX-P-M2 optional richer request and phase inspection beyond the presentation refinement:
154:   exact current response/request identity, per-request run progress,
155:   prompt/prefill/queue and decode timings where original boundaries
156:   provide them; optional bounded recent request samples/latency
157:   distribution only if justified, no unbounded history. Detailed
158:   layer/device topology across actual adapters including future mixed
159:   placement, layer roles/types from original model geometry, reported
160:   memory by owner class. Per-layer time is explicitly profiled, never
161:   estimated from model TPS or enabled via per-layer GPU synchronization
162:   by default. Reuse authoritative provider completion/timing contracts;
163:   qualify observer overhead and physical execution on relevant hardware.
164:   Keep unreported observations explicit. No requirement to invent
165:   unsupported split/offload or per-layer telemetry to fill a dashboard.
~~~~

### [.agents/plans/cli-ux/TUI_POLISH_AND_MOUSE_INTERACTION.md](../plans/cli-ux/TUI_POLISH_AND_MOUSE_INTERACTION.md) — 1 unresolved markers

~~~~text
601: - [ ] **UX-P4 — finish ergonomics and acceptance:** Unicode wrapping/caret,
602:   scrollbar/follow/selection edge cases, small viewports, accessibility,
603:   focus order, no-color/ANSI, busy/disabled/stale/error/empty states,
604:   cross-client theme revision, browser retirement and terminal cleanup.
605:   Real cached resident inference/serve/download/config simultaneous
606:   regression through mouse and keyboard, full offline isolated gate,
607:   updated CLI guide/help and owner handoff. No physical GPU perf claim.
~~~~

### [.agents/plans/cpu-gpu-optimization/BACKEND_PARITY.md](../plans/cpu-gpu-optimization/BACKEND_PARITY.md) — 5 unresolved markers

~~~~text
108: - [ ] Portable arithmetic assembled integration: existing recorded typed embedding→ordinaryRMS→
109:   projection slice, same ResidentProgram traversal/service/flight, explicit
110:   TorchCpuBf16Avx2MixedSourceV1 contract. Reuse the existing64lane projection
111:   shader; preserve every RMS materialization/reduction and check actual bits.
112:   B1/H1024or2048/W<=32MiB, provider64MiB/physical256MiB/host128MiB diagnostic
113:   bounds; no model/state-root/serving grant. CUDA SM86 rsqrt/native profiles
114:   remain separately unavailable. ROOT reviewed runtime/reference source;
115:   strictHIP/Vulkan/CUDA and focused guards pass. Both authentic pinned public
116:   HF/Torch captures plus independent validators pass all11 boundaries. Actual
117:   ten-process HIP/Vulkan cohort now PASS: both healthy H1024/H2048 exact public
118:   boundaries plus private CPU stages/packed RMS/FP32 projection,3nodes/28
119:   dispatches/17readbacks/callback1/finalresident-event-physical0. RMS/projection
120:   unknown cuts on each provider and both shared-PCI directions retain actual
121:   source/234664B/pending1/unpromoted outputs/64or128MiB envelopes;0D2H/Ready/
122:   callback. ROOT independently reproduced exact collector/outcome/source/
123:   binary/reference/cap/test/group/reap/EOF checks; finaljobsnone/VRAMbaseline
124:   observed, notpeak. Reviewed21paths merged MAIN with freshmtimes/backups;
125:   ISO held for CUDA diagnostic. Next combined fullgate remains pending.
126:   Finite synthetic3node CPU-profile proof only; no broad32MiBgeometry, learned
127:   model, CUDA-specific profile, serving or performance grant. Full model parity
128:   needs its actual backend/reference contract, not a provider retag.
129:   Actual arithmetic results.
130: - [ ] Whole-model HF and serving profiles on each backend, with full vocabulary,
131:   every state domain, private natural feedback and independent concurrent roots.
132:   Gemma E2B whole35 bridge design
133:   maps direct gamma1536/grouped256/512, actual call-M/PV contraction, source
134:   row/full-K tiles and numeric-zero-tie262144 greedy into the original engine.
135:   Reviewed13path source/call/observation/tile contract checkpoint merged MAIN;
136:   core183/Gemma32/Qwen79/strictClippy/fmt pass. Original whole35 traversal,
137:   authentic gamma/unit-V roles, callM/projection/PV geometry and <=1MiB frame
138:   recipes preserve old profiles. Inventory prices one4MiB call; every other
139:   simultaneous call/frame/native owner still needs actual admission. Lowerers
140:   and full model/state/physical qualification continue in isolated checkouts
141:   from current MAIN+portable21. No second traversal/IR or monolithic fit grant.
142: - [ ] Before any large Vulkan matrix consumer, use the device's effective buffer
143:   and storage-binding limits. CubeCL's Vulkan index64 path can expose heap size
144:   as max_page_size; that is not a single-binding limit. Existing actual
145:   transport<=3MiB and arithmetic<=32MiB callers have independent small bounds.
146:   Gemma E2B PLE table262144x8960 BF16=4697620480B exceeds x2's effective
147:   0xfffffffc buffer/0xffffffff storage-binding bounds; use authenticated
148:   selected-row reads or checked paging. No monolithic admission/fit claim.
149: - [ ] Product execution/declarative packaging must use a mutually compatible
150:   host-libc/compiler/backend SDK closure. Preserve the actual per-job loader
151:   requirement in qualification; successful prefixed tests do not qualify a
152:   differently launched run-rocm/serving binary. Avoid global environment repair
153:   or repository-external configuration changes without applicable authorization.
154: - [ ] Matched performance after correctness. Existing owner order remains
155:   .8BC1+C4 superior → C8 superior →2B; Qwen27B/Gemma integrate independently,
156:   with their GPU qualification/benchmarks on pvl-x2 HIP. No timing promotion
157:   from compilation, missing reference runtime or an unmatched source/codec.
~~~~

### [.agents/plans/cpu-gpu-optimization/TODO.md](../plans/cpu-gpu-optimization/TODO.md) — 67 unresolved markers

~~~~text
36: - [ ] After correctness: optimize CUDA `.8B`/`2B` on l5 and ROCm `.8B`/`2B`/`27B`
37:   on x2 across fitted active contexts and C1/C4/C8/C16/C32/C64. High concurrency
38:   belongs on sufficiently capable hardware; preserve actual active/queued widths.
39: - [ ] Beat the matched, pinned vLLM baseline in every required feasible run;
40:   diagnose losses, retain all samples and separate throughput/latency metrics.
41:   Prefer simple shared-layer fixes and measured hot-path changes. Finite source
42:   diagnostics and old timings are not current ordinary-serving performance.
43: - [ ] x2 compute/memory/storage use is authorized without routine requests;
44:   do not relax numerical/source/ownership contracts to obtain a timing win.
61: - [ ] Run current-source `.8B`/`2B` model, cached, fault, shutdown and fitted-context
62:   qualification plus final full gate. `.8B` native/reference inputs were recovered
63:   offline on 29 Oct and fresh phase-1 capture passes with the existing 8 GiB
64:   reference host profile. Ordinary cuBLAS initialization now refuses a private
65:   driver export-table call before model math; preserve allocator ownership.
66: - [ ] Run the one serial campaign specified in
67:   `../serving-and-speculation/BENCHMARK_CAMPAIGN_2026-10-27.md`. Preserve the
68:   historical `2B` C16 loss and all cap/resource asymmetries; no new benchmark
69:   was executed in this checkpoint.
73: - [~] [Shared vLLM/llama.cpp correctness policy](../numerics-and-context/CORRECTNESS_POLICY.md)
74:   across all models; .8B initial/cached47–50, then genuine long contexts and
75:   CPU/CUDA/ROCm/Vulkan. All agents GPT-6.1 Sol xhigh. Earlier B4/C16/27B order
76:   below is historical; no new performance or27B capture before correctness.
77: - [ ] Preserve case5 FAIL101 and quarantine. Original default vLLM N1/B4 also
78:   has the same six QKVZ BF16 differences at frontier47 with identical inputs,
79:   input RMS and48 prior planes. Width-specific original agreement and model
80:   accuracy are distinct; no arithmetic substitution or failed-run relabel.
84: - [ ] Current B4 → fresh optimized matched C16 (.8B +2B) → x2 original27B,
85:   latest owner order; no C2. Base `d0fff261`, all246 signatures verified;
86:   README unchanged. All current lanes: GPT-6.1 Sol xhigh, owner override.
87:   Owned run.
88:   Final evidence index.
271: - [~] User extends both-model sequence to C16, thenC32 andC64; NO C2. Implement
272:   genuine populations and honest active8/B4 observations. Latest user direction:
273:   resume all parallel tracks now. The78 signed logical commits are complete;
274:   C16 correction, Qwen27B, Gemma4, Vulkan, HIP and shared queue have independent
275:   Pi lanes. ROOT coordinates builds/physical ownership and quiet timings.
276:   Default pending count unlimited, optional configured cap, actual HOST
277:   admission unchanged. C32/C64 follow queue acceptance. C16 correction and
278:   grouped controls run separately from product source; old drafts preserved.
279:   Audit/store latest and best verified matching historical C1/C4/C8 plus new
280:   bands, with source/profile and stock arm from SAMEattempt, no cherry-picking.
285: - [~] C16 both executed on frozen21. Clean2B: ZG78.7859tok/s,1.62466s;
286:   vLLM95.9974tok/s,1.33337s. Each16offers/16completed/128tokens/0errors,
287:   original semantic/source/owner closure healthy, actual active8/B4 and compiler0.
288:   Rate refusal is a measured performance loss, not numerical/cleanup failure.
289:   First.8B timing overlaps audit scan; one repeat105.7331vs96.6544tok/s has an
290:   extra sampler and remains diagnostic, not quiet/best evidence. No third rerun.
291:   Scope review.
511: - [ ] Shared compiler lifetime/readiness: fix checked all-exit raw NVRTC cleanup,
512:   retain/fence uncertain destruction and partial module loads; establish actual
513:   context-lifetime host/device ownership before eager exact-recipe preparation.
514:   No global DryRun, dummy roots or unpriced cache. Existing model-local128MiB
515:   loader is host scratch, not a persistent CUDA context charge. Numeric ceilings/
516:   complete fit remain unresolved; Sol HIGH implements the contained foundation.
517:   Astra architecture.
518: - [ ] After current C8/refill and source113 qualification: shared prepared matrix/
519:   compact IO reuse, temporal work, optimized eligible GEMM/attention/GDN, pages/
520:   full-domain prefix/COW, admitted prefetch and graph replay. Preserve numerical
521:   profiles; source113 CPU-order dot is not TensorCore/BLAS parity. Family and new
522:   model gaps are queued in model-execution; no hardware eligibility by name.
655: - [ ] **Parallel implementation, third merge — genuine BF16/HF request batching
656:   (owner directive,23Oct).** Develop the physical multi-request unit now; merge
657:   after current storage/residency/architecture fixes, before further N1 microtuning.
658:   Start with genuine canonical47 B4/B8 initial prefill (188/376 total tokens)
659:   through SAME independent roots into decode and correct partial-batch fallback;
660:   extend mixed frontiers through shared engine
661:   cohort/binding/workspace APIs. Reuse existing root-cohort publication, but
662:   implement compatible BF16 attention/GDN/projection lowering rather than
663:   enabling the older GGUF/F32 path. Preserve independent row states, natural
664:   feedback, fair work, cancellation/quarantine and current resource caps.
665:   Validate row permutations/full logits/all states and physical/API behavior,
666:   then matched C1/C4/C8; preserve the C1 fast path. Request batching is distinct
667:   from operation fusion; no speedup or superiority claim before measurement.
668: - [ ] Lift reusable source identity, bounded immutable reads, prepared bindings,
669:   actual-flight indices and resident matrix transport into shared components.
670:   Select qualified implementations using complete host/device live costs;
671:   a family adapter may restrict mathematics without duplicating common policy.
672: - [ ] Shared resident-backing pool + immutable layout/access metadata + bounded
673:   transfer/frame retirement, from23Oct parallel source audits. Keep existing
674:   model traversal/publication and current root/lease freshness; price live,
675:   idle and unknown owners until actual retirement. Expose minimal common APIs
676:   through existing prepare/bind/submission/completion objects. CUDA/HIP pinned
677:   staging and Vulkan queue/mapping need their own physical proof; no inferred
678:   speedup or zero-copy claim. Follow the coherent units in the linked design.
679: - [ ] Demonstrate superior matched .8B C1 AND C4, then run C8. Only after all
680:   three wins, optimize/benchmark2B. Keep original quality/source/caps and report
681:   single-pass uncertainty. Latest prepared-binding release1 observes C1+78.62%/
682:   C4+26.67%/C8−1.09%; C8 superiority remains open.2B head residency is fixed;
683:   actual physical/API pass and C1+15.32% with the explicit4608MiB profile.
684:   Genuine batching precedes wider fresh timing bands; stable superiority is unproved.
685: - [ ] Merge Qwen27B/Gemma contained units as ready in parallel. Their later GPU
686:   qualification/benchmarks use pvl-x2's AMD gfx1151 through ROCm/HIP.
687: - [ ] Bring Vulkan implementations to CUDA parity independently; AMD uses HIP,
688:   not an implicit Vulkan/CPU substitute. Preserve one engine/physical service,
689:   independent roots, fair work and completion/quarantine through each provider.
690:   Read-only pvl-x2 inventory confirms96GiB GPU memory, existing ROCm7.2.4 and
691:   RADV/Vulkan. Availability/compilation is not numerical or performance parity.
692:   [Shared components and backend implementation units](BACKEND_PARITY.md).
694: - [ ] **OPT-01 matched short baseline:** after parity, established same-source
695:   C1/C2/C4/C8 profiles, original ABBA+BAAB/offered-load controls, real TTFT/true
696:   token ITL/E2E tails, correct committed/delivered SLO-goodput and failures.
697:   Separate unprofiled timing from CPU/driver/kernel/copy/IO diagnostics; pin
698:   actual binary/profile/caps/EOS rather than retagging old GGUF workloads.
699:   Quick .8B canonical wave8 C1/C4 baseline completed: ZG0.9557/0.9511 output
700:   tok/s vsstock38.5021/49.6408. Owner23Oct now requires fixing the gap and
701:   beating vLLM first;2B/full repeated/ragged timing deferred. C8 pair6 is now
702:   completed above; earlier C8 cancellation is superseded. Preliminary
703:   single-pass result, no repeated-run confidence or stable-tail claim.
704: - [ ] **OPT-02 reusable bindings:** profile then qualify stable leased row/
705:   workspace metadata, bounded diff uploads, compact device feedback and removal
706:   of redundant source/state/readout copies in the original actor. Source/root/
707:   epoch/physical generations and unknown-completion ownership stay authoritative.
708:   After the current short baseline, measure the1MiB streamed vocabulary head's
709:   actual source reads/copies/launches. Qualify admitted larger windows or full
710:   residency where the complete model/root/workspace inventory fits; retain
711:   streaming when it does not. Investigate sharing serial cold-geometry storage
712:   by genuine liveness/alias proofs rather than summing six separate cohorts.
713:   Both are performance hypotheses, not measured gains or permission to raise caps.
714:   Current source uses485/970 authenticated1MiB reads+H2D+kernel+same-stream
715:   checkpoints per .8B/2B Head. Existing .8B original full-table selected-head
716:   actor offers a reuse path. Raw508,559,360B .8B table addition stays below the
717:   present4GiB provider/host upper bounds, before deriving new exact overheads;
718:   original estimate is now superseded by actual .8B validation below. 2B cannot
719:   inherit that fit claim; evaluate larger tiles and proved workspace reuse.
720:   Active isolated resident-head unit reuses original full-table actor, keeps
721:   same arithmetic/source/private ownership and actual4GiB/5GiB caps. Separate
722:   warmed-process CPU profile now closed healthy: sampled user CPU cycles57.573%
723:   SHA256,15.202% BF16 finite scanning,10.195% actor spin,7.341% memcmp. Private
724:   stripped addresses classified from actual instructions. These are CPU cycle
725:   fractions, not wall/GPU percentages; perf-stat file was empty. Static30.31GiB/
726:   31,040sync per64forwards is not a measured time percentage.
727:   Assembled resident .8B passes actual40 natural calls/all5 prompt lengths with
728:   exact boundaries/fullV/all-state/feedback and pressure/stale/precancel/retirement.
729:   Actual resident head508,559,360B; source-derived provider upper2,625,406,976B/
730:   host upper3,397,702,556B fit unchanged4GiB limits. Conv/recurrent/lastFA fault
731:   cases pass; head-unknown exposed verifier's old streamed-only error recognizer.
732:   Exactly2 resident strings added behind unchanged error/index254/fault20|21/
733:   actual-enqueue gates; two affected Head faults await rebuilt fixture. Vendor117
734:   CPU guards pass. No speedup or complete new-batch acceptance claim yet.
735:   Follow-up physical Headunknown+cancel PASS; matrix40+all5faults complete.
736:   Quiet resident CLI2 diagnostic8requests/64forwards exact, owned source/retirement
737:   closed: forward6.907854s, bind_program3.629112s, enqueue3.021396s,
738:   complete/readback0.148894s. Nested Head0.005158s,0H2D/0checkpoints/1native per
739:   call; prepare includes forward, so totals must not be added. Instrumented
740:   wave7.115s is not an unprofiled performance win. Next substantial host unit
741:   addresses reusable cold construction and attempt-local state proofs while
742:   preserving dynamic root/session/generation/fence/cancel/ownership checks.
743: - [ ] **OPT-03 prepared optimized regions:** close existing production graph
744:   admission in the serving plan; qualify exact producer/consumer fusion,
745:   prologue/epilogue layouts, finite decode/prefill/verify shape variants and
746:   source-derived packs. Preserve materialized casts/reductions, aliases and
747:   effective fused read lifetimes. Fewer graph submissions are not fewer kernels.
748: - [ ] **OPT-04 temporal/fair work:** extend existing native sequence batches
749:   with actual multi-token dense/GDN prefill and fair mixed-work chunks; retain
750:   independent state and profile-specific arithmetic. Serving owns scheduling
751:   integration; memory owns pages/prefix/COW. Re-run short concurrent baselines
752:   after singles and meaningful combinations before larger contexts.
753: - [ ] **OPT-05 state/residency composition:** full-domain compatible prefix
754:   intersection, pages/COW, exact source chunk/pack sharing, safe scratch reuse
755:   and bounded transfer readiness through existing owners. Benchmark cache hits,
756:   cold arrivals, long prefill versus decode, memory displacement and fair tails.
757: - [ ] **OPT-06 larger dense source:** after long-context prerequisites, original
758:   27B execution using authenticated layer/row windows, coalescing/batch reuse
759:   and admitted prefetch. Price every required dense matrix plus activation IO;
760:   a sparse expert cache cannot turn this checkpoint into a sparse model.
761: - [ ] **OPT-07 optional proposals:** target verifier/all-prefix rollback first,
762:   then bounded n-gram/Recall/suffix and compatible trained MTP/block drafters.
763:   Joint finite cost policy considers batch/graph/depth/IO/retained state and
764:   selects K=0 when it loses. Exact distribution, seeded trace, numeric profile
765:   and quality changes are distinct gates. No acquisition/training implied.
766: - [ ] **OPT-08 nested profiles:** reconcile primitives→recursive pipelines→
767:   profiles with the pipeline-composition plan. Compile/in-line/specialize closed
768:   regions through the original engine; no per-token interpreter/search, new
769:   serialized IR, model loop, allocator or publisher. Structural design can
770:   proceed in parallel; new framework implementation must not delay parity.
771: - [ ] **OPT-09 deferred capabilities:** typed sparse selection, grouped experts/
772:   protected SLRU/hotsets/replicas, deterministic PLE/Engram IO, source-dependent
773:   shared-KV/liveness, hardware-specific MMA/PDL and derived weight/state codecs.
774:   Activate only for qualified Qwen/Gemma variants, then other model/device work.
775: - [ ] **OPT-10 promotion:** source/full-logit/all-state/natural-output, actual
776:   cancel/pressure/unknown-completion/fair serving and complete isolated check.sh;
777:   repeated complete-request benefit including cold/setup/memory/energy and
778:   negative cases. Do not multiply external gains or promise universal superiority.
785: - [ ] Qualify [cross-stage residency/fusion](../pipeline-composition/REGIONS_AND_PROFILES.md#resident-values-and-cross-stage-lowering--owner21-oct)
786:   through the original engine after baseline: compare ordinary, resident-chain,
787:   graph and fused paths; retain intermediates/feedback on device where eligible,
788:   report actual H2D/D2H/device traffic, host waits and reasons for required
789:   bridges. Preserve arithmetic/declared approximation, original commits,
790:   lifetimes/cancel/fairness and measure complete-task benefit/negative cases.
791: - [ ] Complete the dedicated [ExLlamaV3 investigation](../fabric-vnext/PRIOR_ART_AND_REVALIDATION.md):
792:   pinned kernel/packing/transform/codec consumers, shape/model/backend limits,
793:   KV, batching/graphs, CPU offload and partition seams; exact file/dependency
794:   licenses and static C/Rust ABI versus direct-port disposition. Add independent
795:   fixtures and map mechanisms into existing plans; existing EXL3/QTIP calibration
796:   design is not missing and must not be duplicated. Initial scoping is not a
797:   completed audit or a new entry in the completed416-mechanism inventory.
798: - [ ] Implement native kernel-contract comparison/fixture support, reusing or
799:   directly porting suitable TensorFold harness/leaf code after license/ABI
800:   review. Add planted rounding/reduction/NaN/tail/no-write/alias/source defects
801:   and prove every expected rejection. Invalid hardware configurations refuse
802:   before launch. Keep broken controls test-only; independent oracles, actual
803:   model/all-state/physical checks and full isolated check.sh remain required.
804: - [ ] Implement the useful kernels/components identified by these investigations
805:   through existing providers, using static-native reuse first and faithful
806:   direct ports when unsuitable for linking. Qualify the implementation as well
807:   as the harness; an upstream passing test is not local runtime acceptance.
808: - [ ] Specialize first for the NVIDIA cards on both existing machines, then
809:   prioritize Strix Halo/gfx1151 ahead of all other new hardware and cluster
810:   work. Keep broad model/device/platform compatibility as the long-term target,
811:   with portable paths and actual per-target gates; retain the separate gfx90c
812:   hold. Hardware order does not alter parity→short→long→dense27B model order.
814: - [ ] Bench and hyper-optimize in strict user order:
815:   Qwen3.5 2B, dense Qwen3.8 27B including SSD/coalesced layer-window streaming;
816:   Gemma4 E2B/26B/31B; Qwen3.8 Flash-Next Q2/~3.5-bpw/Q4
817:   and MTP/drafter/n-gram pipeline. Optimize complete
818:   request goodput with the same model semantics, source quant
819:   calibration and StateRoot precision. Do not change the
820:   Renoir/gfx90c launch hold or assume measured superiority.
821:   Routed-expert streaming applies only to later source-defined MoE variants.
823: - [ ] Keep the local Renoir/gfx90c hold: 2B initialization aborted and automatic resets wedged it; no further launch/reset/recovery or Drop-retirement proof is implied.
824: - [ ] Establish controlled cold/warm request baselines across all five GPU/three CPU optimization targets; accepted five backend profiles do not mean five cards or fleet-wide maximal capability use.
825: - [ ] Qualify capability-selected AVX2/AVX512/VNNI quantized kernels, cache/thread/topology policies and source-derived repacking while preserving reduction/rounding/zero/subnormal rules.
826: - [ ] Measure native/Vulkan tiling, subgroup/vector loads, packing, source-exact fusion/prepared dispatch; gate tensor/MMA/WMMA/integer-dot use by exact arithmetic, not advertised support.
827: - [ ] Evaluate explicit NativeF32 projection policy only after physical FMA edge tests, all-codec trained vectors and whole-model/cancellation/replay gates; WideF64 stays default meanwhile.
828: - [ ] Profile then qualify captured graphs/replay, bounded device loops/persistent scheduling, CPU/GPU overlap and exact optional GPU tokenization/detokenization; prepared plans alone are not graph capture.
829: - [ ] Keep separate LLVM code-generation paths and broader accelerator backends unqualified until independently built and physically tested; accepted native paths use CPP NVRTC/HIPRTC.
830: - [ ] Qualify optional EXL3/QTIP/trellis/codebook/low-bit derived packs and calibration identity/quality separately from original source semantics. No hidden precision substitution or default steering.
831: - [ ] Promote only repeated complete-request gains with cold/warm cost, tails, memory, power, source correctness, negative results and qualified capability/version guards.
833: - [ ] Gate future compiler/driver/libm and fast-arithmetic profiles with scalar fingerprints, physical edge/stress and trained full-logit/state tests. WideF64 is qualified on measured inputs, not a universal IEEE FMA emulation.
834: - [ ] Qualify real deadline wakeups, concurrent/multi-stream scheduling and actual driver failure/recovery before broader worker claims; queue deadlines currently need scheduling/poll opportunities.
835: - [ ] Keep the local gfx90c hold. If a healthy/recovered profile is separately authorized, localize retained allocation/H2D/final-sync initialization before changing kernels; natural SIGABRT establishes neither Rust Drop quarantine nor reclaimed storage.
836: - [ ] Consider read-mostly publication only after representative contention and ownership evidence. If left-right fits, test deterministic dual replay without IO/RNG/admission/state/device effects, one writer, persistent non-Sync readers, short guards without await/load/device work and explicit publish/drain/shutdown. Account for both indexes/logs while sharing one payload; do not reopen the rejected cold-catalog migration.
838: - [ ] Resolve applicable SYK-011..016 source/derived-quality/LLVM/SLO/pressure/steering choices from the [specialization reference](../../research/EXTREME_MODEL_SPECIALIZATION_INTEGRATION_AND_ACCEPTANCE_2026-10-15.md) before that slice; proposed SYK-002..010 are not frozen user approvals. Preserve negative candidates and the default disabled steering policy.
855: - [~] Shared queue checkpoint preserves application-owned HOST admission,
856:   unlimited default pending count and optional cap. Final batch corrects
857:   tokenizer construction, original Rayon reuse, HTTP framing/serialization
858:   and worker/last-reader ownership. HTTP handles and connection credit survive
859:   through join; original default thread stacks remain runtime-selected.
860:   Platform stacks/TLS/caches and OS/driver headroom remain separate, unquantified
861:   scope: application reservations are not process RSS or a full-process fit.
862:   Contract decision.
863:   CPU functional acceptance is complete; the final full CPU gate passed
864:   rc0 25Oct on the committed tree. Assembled GPU checks PASSED 25Oct
865:   (22/22 rc0: builds/stricts, cuda_compiler_context 4/4, loaded/inventory
866:   4/4, CLI baseline/serve-short_hf, host-profile umbrella, four ignored
867:   route cases alone, five actual source-cost cases) —
868:   execution.
869:   Physical device scopes (CLI direct route, serving) remain open.
900: - [~] Direct short-HF route lifetime correction: preliminary tokenizer drops
901:   before model loading, but its static retained-credit vector never releases.
902:   Return tokenizer plus its actual lease; validate the exact legitimate
903:   process-global bootstrap baseline through route/model closure. No VM9/H14
904:   source overlap. Closed11-path source draft has the typed owner; ROOT/Astra
905:   required collected corrections for request payload/credit lifetimes,
906:   unconditional original close, exact bootstrap/backend baselines and tests.
907:   Those source repairs landed in `52f76a0`; the affected optimized/strict
908:   batch and the final full CPU gate passed rc0 25Oct; real source-cost/
909:   lifetime and GPU serving/ownership checks remain pending.
910: - [~] CUDA B4 HOST requirement needs correction before fresh acceptance.
911:   Source-derived P16 lower bound4,341,525,623B already exceeds4GiB by46,558,327B
912:   before bootstrap/native/other metadata; old source21 HOST fits omit the new
913:   R+G+V credit. Cost-only inventory currently bypasses that owner too. Preserve
914:   all real charges, GPU caps and CPU4096MiB population controls; review honest
915:   configured HOST requirements versus real owner/phase savings. Explicit
916:   short-HF8192MiB policy landed in `52f76a0`; CPU/default4096 and legacy4GiB,
917:   provider/physical and H/G/V caps remain unchanged. Measured 25Oct
918:   (application-owned configured-domain accounting at 8192MiB, gpu_executed
919:   false, no process-RSS or device claim): N1 .8B host_upper 4,075,700,811B,
920:   N1 2B 4,111,001,099B, B4 2B 4,741,293,239B, B4 .8B 4,736,259,779B — all
921:   fit 8GiB; legacy 4GiB refusal tied to authenticated host cost (excess
922:   446,325,943B, provider within cap). Real RAM/cgroup availability and
923:   device/serving execution checks remain open.
934: - [~] Five-lane source is assembled in MAIN; the final affected batch completes
935:   portable core admission, actual spawn-capture ownership and direct-lease
936:   metadata. Original private-network HTTP checks now execute without a fallback.
937:   The final source freeze/full CPU gate passed rc0 25Oct and the five-lane
938:   batch landed as signed commits (`8f01635`..`68e13a8`); assembled GPU/
939:   physical checks, H6/G8/V10 cases and lane-tree retirement remain pending;
940:   no new benchmark yet.
1248: - [ ] Assembled .8B host batch: authenticate immutable114-weight cohort once
1249:   per binding, reuse exact cold allocation analysis/certificates, seal fresh
1250:   native roles once, index actual lease/access coverage. Current source/root/
1251:   permit/generation/fence/cancel and physical range checks remain. Existing
1252:   pre-node0 status clear permits removal of redundant4B upload/sync; no new
1253:   sticky-status semantics or cap changes. Ten paths merged into ROOT ISO;
1254:   frozen physical fixture passes full40+five-fault requalification, actual final
1255:   host0/device0 and original owned closure. All14 pure metadata guards PASS,
1256:   including three new lifetime/rebind/SSA/private-effect/alias guards. Matched
1257:   API wave/ragged checks PASS full IDs/text/usage/finish and actual owned/source
1258:   closure. Clean C1/C4 performance remains pending CPU/IO quiet clearance.
1259: - [ ] Native harness passes positive fixtures and rejects all named planted
1260:   defects; independent oracle and selected-profile equivalence matrix pass.
1261: - [ ] Reused/ported implementation passes exact/approximate declared-contract,
1262:   held-out quality where applicable, full-state/cancel/physical and native
1263:   packaging/link checks independently of the harness's own controls.
1264: - [ ] QTIP-01..14: derived source codec/pack identity, transforms, held-out quality and physical kernel correctness pass independently of original model semantics.
1265: - [ ] HYB-01..18: exact source routing/draft/tap/mask/reduction and target-authoritative verification pass, including cases where a wrong draft is rejected or deeper drafting slows work.
1266: - [ ] GFX-01..18: WMMA/LDS caller/shape coverage, driver/compiler numeric gates and coherent-memory physical headroom pass; require matched complete-request gains. See the [50-test register](../../research/EXTREME_MODEL_SPECIALIZATION_INTEGRATION_AND_ACCEPTANCE_2026-10-15.md#4-acceptance-register-50-planned-tests-not-implemented).
1272: - [ ] Complete current15 fresh matched API/HTTP for .8B and2B; ROOT1254 running. Then quiet C1/C4 and C8 only after both rate+wall wins, no hot compiler activity; previous .8B measurements remain historical.
1273: - [ ] Full assembled check.sh phase gate after CLI/source reconciliation; source12 gate18 remains the last accepted whole gate.
1279: - [ ] Real current-source qualification remains mandatory before production routing: .8B natural/current-root parity including cached 47→50, physical fault/cancellation/retirement and real-model shutdown, then .8B 512/1K/2K/4K/8K and 16K/32K. Admit 2B contexts only where the exact two-root plan fits.
1280: - [ ] Current physical blocker: the approved .8B source is now visible at `/home/pvl/.cache/abird/zerogravity/external-baselines/models/Qwen--Qwen3.5-0.8B/2fc06364715b967f1860aea9cf38778875588b17`; its 1,746,942,600-byte safetensor SHA256 is exactly `04b1c301231dd422b8860db31311ab2721511346a32cb1e079c4c4e5f1fe4696`. The transient `.agents/runs` native bundle was intentionally purged and no `cuda-kernels.json`/legacy `short-hf-native.json` bundle remains under the repo or user cache. Reacquire/regenerate the qualified native/reference bundle through the maintained capture path; do not restore deleted run directories or substitute another model source.
1295: - [ ] These are fit/admission results only. Physical correctness, continuation,
1296:   replay, faults and state-history qualification remain required for each cell
1297:   before production use; .8B real-model qualification is blocked on a qualified
1298:   native/reference bundle, not on model-source visibility.
1314: - [ ] Real physical after-launch cancellation/fault and real-model shutdown
1315:   remain required with the approved .8B source; source-free ownership success
1316:   does not substitute for those device tests.
1338: - [ ] Execute both phases serially as soon as the approved .8B safetensor shard
1339:   becomes visible to the connector. Do not download or substitute source.
1365: - [ ] Physical execution is currently blocked by provider identity, not code:
1366:   the verifier is pinned to libcublas.so.12 bytes=105,140,976 SHA256
1367:   d63456d94f1b8e2408a8a9538c8740eb474c9eba48a84f420b15e2e10fbcd048
1368:   and libcublasLt.so.12 bytes=749,205,904 SHA256
1369:   96ddf4891ca4f4ed1475966783a9185df2abd64a81c3cc713b96d036696fc541.
1370:   Historical validated runs used
1371:   /usr/local/lib/python3.12/dist-packages/nvidia/cublas/lib/, but those held
1372:   DSOs are not visible through the current connector. The available Nix CUDA
1373:   12.9.2.10 DSOs have different sizes/hashes. Do not repin or substitute them
1374:   without an explicit source/provider qualification decision.
1492: - [ ] When the approved source tree, pinned native bundle payloads and pinned
1493:   provider DSOs reappear, use this wrapper with a fresh evidence path; no manual
1494:   phase ordering is required.
~~~~

### [.agents/plans/distributed-execution/TODO.md](../plans/distributed-execution/TODO.md) — 12 unresolved markers

~~~~text
21: - [ ] Evaluate layer, tensor/head, context and expert partitions as independent
22:   placement choices under source-specific partition algebra. Distributed full
23:   softmax summaries and recurrent state are different contracts; changed reduction
24:   or8/16-bit activation wire formats need explicit numerical profiles.
25: - [ ] Admit shared collective/IPC/RDMA backing, staging, graph and encoded-reader
26:   ownership through original per-physical-resource ledgers. UMA host-visible
27:   transfer is not discrete GPUDirect; port ratings/tunnels do not attest throughput.
28: - [ ] Later MoE EP/replicas/EPLB/communication overlap: preserve logical expert
29:   source IDs, topology/version guards, old/new migration readers and rank-agreed
30:   batch/graph/verification shapes. PDL requires actual eligible hardware; DBO is
31:   not generic one-GPU prefill/decode overlap.
32: - [ ] Later produced-state P/D/encoder separation and remote prefix hydrate:
33:   complete hybrid source/profile/tenant/root-frontier handoff, authenticated
34:   producer/consumer epochs, completion notifications and failure fencing. Heartbeat
35:   expiry or peer silence cannot reclaim unknown DMA or publish partial roots.
36: - [ ] Compare placement against local batching and capacity needs using actual
37:   directional links/host copies/crypto/collectives/stragglers and offered-load
38:   tails/goodput. Retain two-device/tiny-model and grouped-pipeline negative cases;
39:   no universal speedup from more ranks or deferred global-service descriptions.
41: - [ ] Preserve zero-network standalone execution; add capability/transport/placement seams through the same model/task/coordinator/resource authorities.
42: - [ ] Qualify optional local CPU/GPU/NPU split projection, expert ownership and prefill/decode placement with actual event/memory/power-fabric safety; Stage N is independent of CPU/GPU release.
43: - [ ] Begin authorized two-node layer-group plus local-state execution and immutable full-state hydration; then evaluate P/D and attention+state versus FFN/expert disaggregation.
44: - [ ] Extend tensor/head/pipeline/data/context/expert/sequence parallelism and collectives only where exact state partition algebra, source numerics and transport cost support them.
45: - [ ] Authenticate/version source/state/tenant/session epochs and complete transfers; loss/timeout/stale peers cannot publish partial roots or release unknown in-flight storage.
46: - [ ] Qualify topology changes only at committed frontiers and compare collocation against fast/slow/tunneled links, including real transfer/copy/crypto cost and fairness.
47: - [ ] Add a peer/transport/placement without rewriting model, task, coordinator or API; fleet SSH qualification alone is not distributed inference.
~~~~

### [.agents/plans/fabric-vnext-transition/TODO.md](../plans/fabric-vnext-transition/TODO.md) — 13 unresolved markers

~~~~text
538: - [ ] E01 original-HF general Qwen3.5/2B and real 27B; independent fresh47–50/long context CPU/CUDA/HIP/Vulkan.
539: - [ ] E02 vLLM continuous batching, persistent row slots, temporal prefill, all-domain paged/COW state, goodput; benchmark.
540: - [ ] E03 shared multi-model residency, replicas and hot reload.
541: - [ ] E04 true large dense SSD authenticated windows and physical streaming.
542: - [ ] E05 Gemma4 E2B/26B-A4B/31B complete native qualification, then new families.
543: - [ ] E06 Q4_K source kernel, Q1/Q2/Q3/NVFP4/EXL3/source derived quant and quality evidence.
544: - [ ] E07 sparse/QSA/GDN/MLA/Engram/MoE/MTP draft state and expert IO.
545: - [ ] E08 native ASR/TTS/codec/image/video/duplex and typed task pipelines.
546: - [ ] E09 NPU/graph/wafer/remote distributed placement and state bridge.
547: - [ ] E10 calibrated cost/autotune, matched vLLM/llama.cpp task benchmarks and release.
578: - [ ] Live insertion/replacement, heterogeneous/native shared residency,
579:   additional independent adapter retirement and context reinitialization.
636: - [ ] Broader qualified physical interop; general DAG/media/context,
637:   mixed backend/multi-source HF and pipeline CLI/API activation only
638:   through original qualified owners.
708: - [ ] Non-Linux bootstrap and general host-visible GPU/UMA backing
709:   aliasing beyond the qualified original coherent mapped producer.
710:   Broader family/codec/task
711:   registration, other resident formats, generalized persistent session storage, cross-provider restore and
712:   additional backend prefix eligibility remain open.
~~~~

### [.agents/plans/fabric-vnext/PUBLIC_LIB_AND_SERVING.md](../plans/fabric-vnext/PUBLIC_LIB_AND_SERVING.md) — 12 unresolved markers

~~~~text
318: - [ ] Compile a standalone Rust program depending ONLY on zerogravity and run a permitted local source/model task end-to-end.
319: - [ ] Compile a standalone Rust program depending ONLY on zerogravity and embed/start/stop a live API server and submit a request with real emitted events.
320: - [ ] Separate zerogravity-serving crate builds and integration tests WITHOUT zerogravity-cli or Qwen/Gemma/GPU import.
321: - [ ] zerogravity-cli depends on zerogravity facade only (and conventional CLI/terminal/logging helpers); no private engine/serving internals.
322: - [ ] All existing CLI actions have programmatic API equivalent with matching errors, rights and result semantics.
323: - [ ] Server and direct SDK share loaded immutable model/physical backing and ResourceArbiter; no duplication by alias/listener.
324: - [ ] Server ingress/reload/shutdown while active sessions run does not change their source/model generation/root unexpectedly.
325: - [ ] Strict loopback/auth/body/stream conformance; slow/disconnected clients do not leak GPU buffers/HTTP queues.
326: - [ ] A source-specific novel model executes via HTTP with no model-specific server or CLI branches.
327: - [ ] Native media/structured task SDK output and server transport share typed event contracts; prepared vs committed vs delivered distinct.
328: - [ ] CPU-only build requires neither network stack nor GPU compiler and can run without an HTTP daemon.
329: - [ ] Rustdoc/examples plus crate/features docs show library-first usage before CLI-specific flags.
~~~~

### [.agents/plans/fabric-vnext/TODO.md](../plans/fabric-vnext/TODO.md) — 42 unresolved markers

~~~~text
28: - [ ] Confirm the user has expressly initiated the **read-only resync/revalidation phase**.
29: - [ ] Read the full frozen Fabric vNext pack and literal user amendments, then AGENTS.md and current owner plans.
30: - [ ] Inspect current live HEAD, branch, checkout, recent commits, in-flight agents and true Cargo feature/dependency graph. Protect third-party work.
31: - [ ] Reconcile current source/architecture docs/active TODOs with historical 26-Oct analysis; classify old path anchors obsolete/new/already done.
32: - [ ] Reconfirm current in-source model families, generic semantics, numerical policies, physical provider, queues, state publication, resource services and source ownership.
33: - [ ] Reconfirm current model-neutral engine/SDK/CLI/serving responsibilities and look specifically for newly added server/engine library work.
34: - [ ] Re-evaluate public zerogravity facade + standalone zerogravity-serving crate + thin CLI against exact new code; revise module names/ownership for simplicity without weakening the newest user decision.
35: - [ ] Validate distinction among mathematical region, native ModelAssembly and multi-model TaskPipeline; complete-domain state and typed media clocks.
36: - [ ] Crosswalk every source/design gap with the current active P00–P23 workstreams, including source/quality/physical and model/backend holds.
37: - [ ] Produce source-grounded diagram, gap table, revised implementation plan, test evidence ledger and explicit remaining decisions. **Stop without coding.**
41: - [ ] Receive explicit user permission to start implementation after the source-grounded resync.
42: - [ ] Freeze a reproducible implementation/test baseline without corrupting active work.
43: - [ ] Introduce minimal generic SemanticProgram/closed-region and prepare/bundle paths from existing ForwardPlan; avoid second IR.
44: - [ ] Remove family math and source-specific source/ABI recipes from generic core, artifact, GPU and serving implementations.
45: - [ ] Refactor model family crates and reverse dependencies, no GPU → Qwen/Gemma import.
46: - [ ] Extract one model-neutral zerogravity-engine or equally enforceable cohesive engine boundary; one TaskCoordinator/StateRoot and physical ResourceArbiter.
47: - [ ] Provide source-qualified model instance/session/TaskDriver interfaces and cold built-in registration, not model-name switches in runtime.
48: - [ ] Implement public zerogravity Rust library for ALL functionality (run, inspect, load/pull/cache, admin, device planning, pipelines, metrics and serving).
49: - [ ] Create independent, embeddable **zerogravity-serving** Rust library for HTTP/OpenAI/Responses/stream/media service protocols and lifecycle; no CLI/model-kernel imports.
50: - [ ] Make zerogravity-cli ONLY call exported zerogravity library and own args/env/terminal/signal/output.
51: - [ ] Prove separate library embed and server start/stop, direct run and HTTP share same model/resource/state; strict auth/stream/errors.
52: - [ ] Prove Qwen3.5 0.8B/2B source-real CPU and selected GPU state/numeric/regression gates survive changes.
53: - [ ] Add Gemma source-real model through same engine and server without source-specific allocator/HTTP/scheduler branching.
54: - [ ] Record accepted and failed/unsupported paths per actual profile/driver/model/corpus and explicitly leave unrun hardware cases open.
58: - [ ] Persistent GPU slots, incremental row metadata, actual async bounded step submission and retirement.
59: - [ ] Native source-qualified continuous batching, real temporal prefill/decode overlap and fairness/ageing.
60: - [ ] Paged and complete-domain hybrid state/COW/prefix checkpoint reconciliation including sparse and recurrent state.
61: - [ ] Region fusion/capture/device-native feedback; optional speculation and draft/target full state semantics.
62: - [ ] Unified physical residency and SSD streaming on one source identity with byte-budgeted cache and real completion.
63: - [ ] Qwen3.8 27B exact source/CPU/selected GPU, Gemma E2B→26B-A4B→31B independent functionality; then richer sparse/quant targets.
64: - [ ] First real non-token task and multi-model typed pipeline via same public library/engine/serving.
65: - [ ] Qualified CPU+GPU+NPU, local split and authorized remote region workers only with actual source/state/physical/latency proof.
66: - [ ] Matched vLLM/llama.cpp complete-request/quality/TTFT/ITL/P95/goodput/memory/energy benchmarks and negative findings.
67: - [ ] Minimality/extension drills for new model, novel stateful op, backend, task, transport with measured shared source change radius.
68: - [ ] Consolidate normative docs/public whitepaper/current status after actual implementation acceptance.
72: - [ ] Keep this plan set internally linked and consistent with the canonical current handoff and four-file planning spine.
73: - [ ] Revalidate source-dependent statements against current main before implementation; historical SHAs remain provenance only.
74: - [ ] Keep the next-agent prompt source-first: read the canonical handoff, current status/traceability, then this plan; do not reconstruct deleted archive snapshots.
75: - [ ] Later: enforce Cargo acyclic layering including serving→engine and facade→serving and CLI→facade.
76: - [ ] Later: public zerogravity-only embedded Rust direct run/server examples compile without CLI, with one shared model/ResourceArbiter and exact source states.
77: - [ ] Later: source/semantic/quality/physical/serving/multimodal/distributed acceptance gates defined in IMPLEMENTATION_AND_ACCEPTANCE.md and original project owners.
78: - [ ] Later: no vLLM/llama.cpp performance parity claim absent true matched workload/driver/profile/quality evidence.
~~~~

### [.agents/plans/memory-and-streaming/TODO.md](../plans/memory-and-streaming/TODO.md) — 29 unresolved markers

~~~~text
22: - [~] Physically qualify each admitted context independently. Preserve current
23:   F32/F32/BF16 two-root semantics; never change precision/source/root semantics merely
24:   to make a cell fit. Public CUDA serving passes4K/8K/16K for both0.8B and2B with exact
25:   replay/frontiers and clean retirement;1K/2K separately retain all42/finite-state
26:   qualification.32K now has split status. The exact direct optimized0.8B32K
27:   retained qualifier has one full current-path pass in185.18s with exact replay,
28:   independent sessions, exact producer frontiers and clean shutdown. The next identical
29:   direct trace run was externally SIGKILLed at~80.7s after >60s of execution. Thus
30:   0.8B32K model capability is positively established once.2B32K public load succeeds and
31:   is now localized to a real long-context response deadline rather than infrastructure:
32:   current matched optimized runs survive normally to the retained SDK's600s
33:   first-response timeout (~616–617s total) without SIGKILL. Do not treat this as
34:   VRAM/admission failure. Simple partial-resident source placement (~2.9GiB resident),
35:   state-path-first pack ordering and a16MiB serial source page did not materially move
36:   the timeout, so those experimental production changes were removed. The obsolete
37:   CUDA-forward attention reserve is now removed from the real admission/ownership graph:
38:   the selected tile-local kernel uses no global scratch, saving138,428,516B at32K. This
39:   makes the low-level two-root2B32K inventory fully resident. Public active1 still
40:   retains four **logical** lifecycle roots, but the paged-state inventory now stops
41:   preallocating unreachable duplicate KV pages: active current roots own complete page
42:   sets while snapshots/displaced/Pending candidates share immutable prefixes and reserve
43:   only bounded divergent COW/appended pages. This preserves lifecycle authority and makes
44:   explicit paged2B32K fully weight-resident. Same-capacity1024-token public A/B improves
45:   283.22→28.56s, and full32766-token paged serving passes in224.80s. The21 Nov generic
46:   smart-config layer supersedes the fixed Contiguous default: canonical `QwenCuda` uses
47:   Enabled mode and Auto selects Paged64/64 when it preserves weight residency, while user
48:   values still win per knob. Final CUDA admission independently re-observes free VRAM from
49:   the original actor and clamps the provider envelope before allocation.
50: - [~] Preserve complete recurrent/conv/KV state history, chunk continuation,
51:   cancellation/quarantine and readback owners at512/1K/2K/4K/8K/16K, then32K.
52:   Public4K/8K/16K execution proves the current chunk-continuation/serving path;0.8B32K
53:   has one full retained public-SDK pass and2B32K now passes the full32766-token retained
54:   cell under explicit Paged64/64 in224.80s. Canonical default2B32K/1024 also physically
55:   Auto-selects that paged policy and passes through the live actor-side VRAM clamp. The
56:   independent populated all42 archive is absent on L5, so32K **numerical** closure remains
57:   open; use the retained32768 qualifier and current smart/storage policy rather than
58:   adding another context fixture or restoring the obsolete refusal assumption.
63: - [~] Bounded completed-root backing reuse is adopted in the shared resident layer.
64:   `ResidentPool`/`BackingPlane` retain one charged physical allocation while every
65:   checkout receives fresh `BufferLeaseId`/allocation generations. `ResidentReadback`
66:   and `ResidentReady` retain the checked-out lease/use; a set returns idle only after
67:   the device ledger proves every backing view idle. Failed/unknown retirement poisons
68:   the use and fences/quarantines instead of reusing it. CUDA-forward/HIP source
69:   sessions install this pool through `install_window_root_pool`. Remaining work in
70:   this roadmap item is the broader authenticated transfer-slot/ring and physical-
71:   backing identity adoption across source/staging paths; do not add another root pool.
78: - [~] The first CUDA-forward source-page transfer cleanup now uses the existing
79:   **owned authenticated window** boundary directly. `TensorWindow` remains the exact
80:   source/cache/authentication owner and already carries the real `TransferStaging` lease;
81:   CUDA wraps only that owner in an `Arc`, installs it in the page's existing pending state
82:   before enqueue, and the original actor reuses its generic bounded owned-window H2D
83:   primitive. The existing page+consumer `sync_buffers` checkpoint is still the only serial
84:   reuse boundary. No payload `Vec`, second `Bytes`, CUDA pinned staging allocation, new pool
85:   or new retirement ring exists on this path; HIP/non-CUDA serial representation remains
86:   unchanged by cfg. The earlier `b839920` source-side pinned experiment is superseded: the
87:   CubeCL pinned `Bytes` controller has no cheap `duplicate`, so `bytes.clone()` deep-copied
88:   to Native and the generic writer then staged to pinned again. A cache-promotion variant
89:   (41.00s) and moving the pinned `Bytes` directly into the writer/drop-queue lifetime
90:   (44.88s) were both rejected against the same 2B/32K Contiguous/Streamed prompt48 cell.
91:   The final retained-owner actor upload passes that exact cell at **38.46s**, versus39.36s
92:   at `b839920` (~2.3% faster), with exact replay/frontiers, independent sessions and clean
93:   shutdown. The earlier direct SSD→pinned/cache-bypass prototype remains rejected at41.77s.
94:   The next simple overlap attempt is now rejected too: generalizing the existing two-slot
95:   retirement ring to CUDA while keeping one device page/same stream passed strict checks,
96:   but matched same-tree Contiguous/Streamed2B32K/prompt48 measured **38.323s serial vs
97:   39.281s ring** (~2.5% slower). The Qwen consumer/selector and temporary Contiguous policy
98:   switch were removed; an Auto control at24.267s chose paged state + resident weights and did
99:   not exercise this fallback. Keep the reusable mechanism: CUDA `OwnedCudaEvent` and HIP
100:   `OwnedHipTransferRetirement` directly implement `RetirementMark`; the zero-cost
101:   `CudaTransferRetirementRing<Owner>` / `HipTransferRetirementRing<Owner>` aliases reuse the
102:   existing generic bounded ring with no wrapper allocation or model policy. These dormant
103:   primitives are available to Gemma, larger streamed models, other CUDA/HIP devices or
104:   workload shapes that can separately prove exact ownership/admission and a measured benefit. Do not re-enable the
105:   rejected Qwen one-page selector merely because the primitive exists. A future second device
106:   page or copy stream still needs separate physical admission and a real workload case that
107:   requires streamed weights. No benchmark fixture/evidence pin was added. Direct/async source
108:   IO remains future work only if it composes with the verified cache/miss policy instead of
109:   bypassing it.
110: - [~] **25 Nov matrix window/transfer follow-up:** `7b36a879` and
111:   `0342e103` share authenticated window row validation, original owned CUDA
112:   H2D for cold packing, and the existing streamed pending/reuse checkpoint;
113:   resident access is direct, HIP retains its actual event retirement ring.
114:   Portable serial staging has an admitted copy/bounce bound. No new page
115:   allocator or cache. `fdfb467f` real Qwen 0.8B resident/streamed exact
116:   B1 full logits/all42 and native B4 tokens/scores/all42, streamed
117:   cancel/discard/retry, SDK C1 and native HIP regressions pass. Fresh
118:   compiler contexts avoid the cumulative PTX limit hit in the first
119:   combined comparison; do not raise the budget.
133: - [~] Canonical physical buffer identity now exists in the model-neutral device
134:   ledger. `DeviceBufferIdentity` is an opaque process-local **ledger namespace +
135:   allocation owner + exact byte range**: direct allocations are owned by their original
136:   lease allocation, while pooled/aliased views name the retained `DeviceBackingId`.
137:   The non-wrapping ledger namespace prevents identical local lease/backing counters from
138:   separate resident services from comparing equal. Equality/`same_allocation`/`overlaps`
139:   therefore describe real storage rather than hashes, chunk coordinates or backend
140:   handles. `DeviceEventLedger::buffer_identity` is the sole public constructor, and
141:   production overlapping-alias admission now uses the same identity semantics. The first
142:   resident consumer is the source-page immutable-cache/output alias guard: both HIP and
143:   CUDA paths validate all outputs through shared `ResidentService::overlaps_any`, which
144:   checks service ownership and performs one ledger lock with no heap allocation or
145:   per-lease identity field. The same primitive now replaces backend-handle range checks in
146:   the native B4 embedding preparation path for output/table/status and feedback aliases;
147:   its four-packet diagnostic pairwise check simply reuses `overlaps_any`. The production
148:   CUDA-forward body now has the first **prepared-set** consumer too: shared
149:   `ResidentService::pairwise_disjoint` validates the complete deduplicated resident lease
150:   set under one ledger lock before binding, while `BufferRange`/`OwnedRanges` remain only
151:   for their legitimate kernel-handle geometry role. The prepared streamed BF16 vocabulary
152:   head now reuses that same helper for its weight-staging/input/output lease triple, so
153:   distinct logical lease IDs cannot hide overlapping physical backing before the existing
154:   two-slot authenticated retirement ring starts. The ring, source-owner lifetime and
155:   same-stream overwrite/retirement protocol are unchanged; no second transfer abstraction
156:   was added. The production HIP resident-text path now uses the same authority too:
157:   cold workspace owners are pairwise ledger-checked once, while per-request root binding
158:   uses a one-lock incremental disjointness check against that already-validated workspace
159:   plus earlier new roots. This preserves the old O(root*workspace + root²) validation shape
160:   instead of rechecking workspace² on every token, and keeps exact handle ranges only for
161:   execution-view length/binding geometry. The helpers allocate nothing internally and cache
162:   no identity into leases. HIP source/public composition checks and the two existing host
163:   source-admission tests pass; X2 physical execution remains environment-gated. Existing B8
164:   public-SDK Contiguous/Paged execution remains exact/clean, and the retained 0.8B
165:   context1024/prompt48 public-SDK replay cell passes again after the shared helper refactor
166:   with exact tokens/frontiers, independent sessions and clean shutdown. No new alias fixture
167:   was added. Six backing-ledger tests cover cross-ledger separation, direct-vs-backed
168:   identity, exact offsets/ranges, overlap/disjoint aliases, continuous charge, cancellation
169:   and unknown fencing. Remaining adoption is to query this identity only where additional
170:   graph views or transfer consumers actually need it, without mandatory hot-path bookkeeping.
171:   Encoded/captured/queued readers still remain part of the broader retirement work.
172: - [ ] Add measured lifetime-aware scratch reuse/phase lending only when no live
173:   root, graph, transfer or reader needs the range. Price borrowed capacity,
174:   old/new COW copies and following decode residency; do not import a hidden arena.
175: - [ ] Protect demand/decode hotsets from transient prefill and unused predicted
176:   reads; compare protected segmented/recency/frequency policies under topic shifts.
177:   Different source codecs use byte-size classes under one cap. Cache policy may
178:   change residency, never source routing, expert count or arithmetic silently.
179: - [ ] Separate known source lookup IDs, dense next-layer windows, produced-state
180:   restore and predicted expert prefetch, while sharing fair bounded IO/host/device
181:   admission. Optional direct/async IO retains the same hash/short-read/cancel
182:   contract. A fully resident path performs no mandatory cache-miss bookkeeping.
183: - [~] Common-frontier prefix reconciliation is now implemented model-neutrally in
184:   `zerogravity-engine::state_cache`: `PrefixDomainCheckpoint` supplies each required
185:   domain's retained-through frontier and zero-based checkpoint alignment, and
186:   `reconcile_prefix_frontier` returns the largest request-bounded frontier legal for
187:   every domain. The hot check is allocation-free O(domains), uses an overflow-safe
188:   bounded LCM and returns0 when no nonempty common aligned checkpoint exists. Exact
189:   partial-tail COW is **already closed by the existing paged-state path**, not by this
190:   frontier algebra: `share_prefix` retains an immutable physical tail page with a narrowed
191:   valid extent; `fork_for_append` transactionally allocates a fresh page and emits a
192:   source→target copy for exactly the valid tail; the append plan pins both handles, the
193:   resident submission clones those Read/Write leases until physical completion, and only
194:   Ready candidate pages can enter the atomic root/domain swap. Displaced domains drop only
195:   after the new root commits. Existing physical materialization/boundary/source-fork
196:   qualifiers prove fresh generations, immutable source bytes, exact partial-tail copy,
197:   zero inactive tail and complete retirement. Do not add a second COW owner abstraction.
198:   Approximate truncated replay and moved-message reuse stay outside exact mode.
199: - [ ] For dense27B, compare layer-major/row-cohort source-window reuse against
200:   extra activation spill, retained states, IO amplification and fair decode
201:   progress. First correct execution may itself require streaming; no full-resident
202:   runtime is assumed feasible under the declared5GiB target.
204: - [ ] **Next SSD source execution:** integrate accepted original-index
205:   owned windows and an async I/O/readiness model with
206:   original CPU/GPU TensorSource and BufferLease, cancellation,
207:   cache/eviction/ABA and independently trained full-logit/StateRoot
208:   parity on real large weights.
210: - [ ] Shared one-process pipeline residency: source-exact immutable
211:   model/component buffers, per-session private StateRoots, one
212:   ResourceArbiter (UMA/GPU/RAM/SSD and device queue), cross-model
213:   lease/zero-copy only when physical placement, codec and lifetime
214:   permit it. Do not replicate weights per route or release a
215:   GPU buffer before actual completion/quarantine retirement.
216: - [ ] Prioritize Qwen3.5 2B resident then Qwen3.8 27B SSD
217:   streaming and Gemma4 E2B/26B/31B, then sparse Qwen Flash-Next
218:   learned 51B n-gram bank and 4B MTP head placements. Compare
219:   source-real logits, resident/streamed StateRoot, cold/warm
220:   TTFT/ITL, I/O/read amplification and throughput with full
221:   physical/cancel/error gates; live audio's soft deadline may
222:   demand resident codecs and refuse cold SSD layouts.
224: - [~] Host scopes share one original physical backing constraint
225:   (cbd2c119/197c2e26); all host lease admissions/retirements/partitions use
226:   the same arbiter. First observed ceiling and later known cgroup constraint
227:   separate; actual source/control/context cold fit, no hot OS polling.
228:   Qualified HIP intersects driver/local VRAM without summing host/GTT.
229:   Real CUDA and HIP source regressions pass; initial lifetime-upper-as-growth
230:   check corrected without weakening logical bound. Remaining direct/legacy
231:   unadmitted owners, general UMA backing alias mapping and non-Linux/OS
232:   hard limits still need qualified consumers. a6548863 shares
233:   original Linux HIP preadmission with the ordinary portable owner and
234:   guarded exact-adapter Vulkan local fit; same declared physical placement
235:   does not establish common backing or zero-copy. Actual portable HIP/Vulkan
236:   two-source healthy/unknown and peer refusal/retry pass.
237:   3b5a520d/fc38db03/e1decff5 close bounded named Linux Vulkan original
238:   service reuse and local fit; actual cold peer refusal/raced/warm reuse
239:   and independent envelope retirement. ce83cc18 exposes actual retained
240:   capture/cache/pin/accepted replay facts without another backing owner;
241:   original retirement/unknown retention remains authoritative.
242:   14bbbd48/5d952364/5a7535e7 qualify exact coherent Vulkan host producer
243:   on original managed backing/range; synchronous borrowed copy under
244:   exclusive Write and actual prior barrier, original Ready, no second
245:   charge. Actual GPU full/subrange/neighbor consumption and retirement,
246:   foreign/extent/duplicate/HIP refusal. General device-write host reads,
247:   foreign imports and aggregate UMA capacity remain unqualified.
248: - [ ] **Paged hybrid state allocator:** extend original leases/StateRoot with
249:   finite page pools/block tables, generations, valid tails, charged metadata
250:   and peak COW/candidate memory. Retain a qualified contiguous policy and typed
251:   recurrent/conv checkpoints. Pin encoded/captured/queued/in-flight readers;
252:   reclaim only after real completion, preserving cancellation/quarantine.
253:   Design and physical acceptance: [batching/paged state](../serving-and-speculation/BATCHING_AND_PAGED_STATE.md#paged-state-allocator--memory-plan).
254:   Signed `0027c28` lands the logical-only shared block-table/COW planner:
255:   exact source/numerical/domain/layout + semantic page/unit compatibility,
256:   immutable full-page sharing, partial-tail COW, transactional allocation
257:   refusal and copy/write owner pinning. It deliberately owns no device allocator
258:   or Ready/root authority. Signed `54340a6` now binds it to a finite
259:   `ResidentPagePool` over existing `ResidentPool`/`ResidentLease` physical
260:   backing/generation/event/quarantine ownership. The CUDA physical qualifier
261:   passes finite pressure, fresh reuse generation, fork/read pinning, completed
262:   GPU readback reuse and submitted-unknown fencing/quarantine that diagnostic
263:   sync cannot reclaim. Signed `1eb6e0f` adds actual COW materialization under
264:   those original leases/events: fresh-page zero, partial-tail device copy and
265:   append H2D ranges physically pass page2048/unit512 with valid[2048,512],
266:   immutable source, zero inactive tail and complete retirement. Model KV
267:   block-table lowering remains open; recurrent/conv domains remain typed
268:   contiguous checkpoints. Signed `3ac2059` adds a typed paired K/V paged-domain
269:   representation and bounded gather-to-contiguous adapter. RTX3060 frontiers3/4/5
270:   around a four-token page match exact contiguous K/V bytes under a four-page
271:   peak cap, including boundary COW, full-page sharing/backing reuse, immutable
272:   parents, zero tails, capacity refusal and final release. At `3ac2059` it was
273:   not yet selected by `ResidentSession::Domain`. Signed `dc6a188` adds transient typed K/V
274:   staging plus device-to-device scatter from the unchanged contiguous candidate
275:   ABI into private pages. Physical prior3->candidate4 stays under one original
276:   flight, does gather2/prefix-copy2/scatter-COW2/scatter-append2 with no host
277:   readback after input, preserves the parent and retires all pages. Session
278:   integration and full-model parity remain the next allocator/COW gate. Signed
279:   `53336b3` adds the canonical paged-KV `Domain` representation and the exact
280:   original root-swap helper used by single-session commit; physical roots0->3->4
281:   publish with initialized page ownership, exact displaced parents and full
282:   retirement. Production ordinary sessions still select contiguous KV. Admission
283:   review rejects a separate full prior+candidate staging pair on top of paged
284:   state because that becomes a third state-sized envelope at full context.
285:   Signed `ef9e4d4` adds the ordinary KvInPlace execution contract: exact fresh
286:   prior/candidate staging alias, no zero/prefix-copy preparation, physical
287:   prefix/new-row/tail exactness and clean retirement. Next split ordinary root
288:   storage into admitted vector/page/staging pools, allow only exact initialized
289:   current-root full pages as inherited Ready state, and add paged snapshot/read/
290:   restore before selecting the production session route. Signed `93468b1`
291:   closes those semantic gates: paged partial/full snapshot +4->3->4 restore pass;
292:   Ready requires only new/COW page Writes and proves2 inherited full pages at
293:   4->5. A pure hybrid inventory plan now proves vector/staging/page maxima stay
294:   inside the old root byte envelope and rejects insufficient fork headroom.
295:   Signed `cb57e1d` constructs the real Vector/staging/page pool owner and
296:   verified-source installer. Physical4-vector-root +1-staging +24-page maximum
297:   equals the old67584-byte envelope exactly, reuses all idle backing and retires
298:   to resident0. Initial hybrid roots are zeroed Vector + empty paged KV; candidate
299:   allocation still refuses without the current root. Signed `b9b04aa` wires
300:   current-root-aware candidate execution: Vector candidate + one staging set +
301:   per-KV paged append under one exact submission, gather before the ordinary body,
302:   scatter after, Ready only for Vector/fresh-COW outputs and paged Domain creation
303:   after completion. Physical one-Vector/two-KV base3->4 is exact with gather4/
304:   scatter8, five Ready outputs, exact continuation/tails and final resident0.
305:   Signed `1413f72` adds the explicit low-level source policy while production
306:   registration stays Contiguous. Paged policy installs the accepted hybrid pools
307:   only for explicit page_tokens/max_append_tokens, enforces B1 and append bounds,
308:   and has an original-HF Contiguous-vs-Paged full-vocab/all42 page-boundary
309:   differential. The real L5 managed cache now contains approved0.8B/2B snapshots;
310:   original0.8B page4/append4 passes full-vocab/all42 at3/4/5/9.
311:   Signed `398e446` adds the matching paged B1 lifecycle target:
312:   prelaunch cancel, abandoned Ready, cancel-after-Ready discard, page-boundary
313:   commits, exact all42 restore and shutdown. Signed `3041457` corrects restore
314:   qualification to fresh root/epoch publication at the saved frontier.
315:   Cold-paged0.8B lifecycle is accepted by retained exit0 dotlink job
316:   `9feb1394...` with `contiguous_warmup=false`, exact lifecycle/all42 and
317:   shutdown. Signed `e6aac53` leaves the qualifier cold-only. Other null-exit
318:   attempts are signalled-Bubblewrap infrastructure events with zero cgroup
319:   OOM/pids and no observed dotlink restart/poll-failure evidence, not model
320:   failures. Deployed dotlink v0.17.3 records a signalled child as
321:   `Completed + exit_code=null + success=false` via `ExitStatus::code()==None`;
322:   main's richer signal enum is not deployed. A direct no-inner-supervisor run
323:   still reproduced null exit. Dotlink v0.17.4 follow-up corrects the earlier
324:   quiet-job interpretation: every connector-initiated stop is now logged, and
325:   the relevant window contains only an explicit `job_cancel`; a silent long
326:   sleep survived until cancellation. Earlier SIGKILL9s therefore belong to the
327:   surrounding harness/orchestration layer, not a dotlink quiet-job watchdog.
328:   With a heartbeat sidecar used only as harness plumbing and a foreground model,
329:   original2B
330:   page4/append4 is accepted: `38c1f321...` differential exits0 in703.17s with
331:   full-vocab/all42 exact at3/4/5/9; `744f36fd...` cold lifecycle exits0 in
332:   438.18s with exact cancel/abandon/restore/all42/shutdown. Signed `6b920a0`
333:   owns the native-GPU harness repair: reuse existing Bubblewrap user/PID
334:   isolation, seccomp-deny socket/io_uring creation when Bubblewrap networking is
335:   shared, retain outer user+net unshare otherwise, expose `--isolation-info`,
336:   and require `--require-private-network` for loopback callers. Signed `4688cba`
337:   adds recognition of an already-entered child netns only when self is loopback-only,
338:   PID1's network view differs and no socket FDs are inherited.31/31 harness tests
339:   plus real child-net loopback+CUDA smoke pass. Direct shared-net L5 still refuses;
340:   an explicit `bwrap --unshare-net --bind / / --dev-bind /dev /dev --` wrapper can
341:   supply the required private loopback namespace. Both small approved models clear source-real paged B1
342:   state/lifecycle. **19 Nov original0.8B also clears true in-flight cancellation:**
343:   a real post-enqueue/pre-Ready launch cut proves Vector/staging/page ownership is
344:   held until known physical completion, cancellation retirement creates no Ready
345:   or initialized candidate Write, committed root/all42 stay exact, pool occupancy
346:   returns exactly, replay continues and shutdown is clean. A deliberate submitted
347:   future drop proves the existing fail-safe unknown path fences the service and
348:   quarantines the pending event,396 handles and completion owners without pool
349:   reuse or root publication. Two externally SIGKILLed attempts remain incomplete;
350:   the terminal known/unknown runs pass. **19 Nov high-level selection is now
351:   explicit but still non-default at the19 Nov checkpoint:** `QwenCudaWithStatePolicy`
352:   carries public `QwenCudaStatePolicy::Paged`; the21 Nov smart-config layer supersedes
353:   the fixed Contiguous default while retaining this API as a user override. The
354:   `page_tokens` remains explicit. Native rows4/8 now share the same Hybrid pool
355:   and per-lane candidate abstraction: paged gather/body/scatter stays under the
356:   original cohort submission and only actual fresh/COW outputs can become Ready.
357:   Contiguous B8 keeps its existing optimized KV transition; paged uses the simple
358:   already-qualified `KvInPlace` append path. Focused selector/policy, strict clippy
359:   and serving-composition checks pass. The compact original0.8B native transport
360:   parity cell is now terminal-green for B2/B4/B8 prefill/decode/all42 (306.84s,
361:   clean shutdown); two earlier external-SIGKILL attempts remain incomplete only.
362:   Public SDK is also terminal-green with8 concurrent page-boundary requests,
363:   native B8 on both Contiguous/Paged, exact returned IDs and clean shutdown
364:   (340.78s). Public HTTP now passes the same workload under the proven child-private
365:   netns: native B8 on both, exact IDs, clean shutdown in358.07s. Prefix reuse and
366:   overlap are still later.
367: - [ ] **Complete-domain prefix reuse/COW:** the first physical fork substrate is
368:   accepted. `ResidentSession::fork_shared_prefix` seeds only a fresh destination,
369:   validates same source/service/capacity/all domains and publishes cloned immutable
370:   owners under the destination's own replacement root; resident-text usage additionally
371:   requires the same resident model. Original0.8B paged page4 prefix3 -> fork ->
372:   append1 matches independent replay for full logits/all42, keeps the cached snapshot
373:   immutable and retires cleanly in179.13s. No replay/readback/copy occurs at fork.
374:   The bounded lookup/eviction substrate is now implemented in the engine:
375:   `2df8cbb3` supersedes numeric compatibility placeholders:
376:   `PrefixCompatibility<K>` binds the actual typed prepared state key and sharing
377:   scope; `PrefixCache<U,T,K>` keeps exact units + immutable `Arc` payloads under
378:   entry/weight caps, does allocation-free greatest-prefix/newest-tie lookup, refreshes
379:   duplicates and evicts FIFO. It creates no restore authority. **Explicit scoped
380:   worker integration is now accepted:** nonzero `PrefixSharingScope` travels beside
381:   the admitted request; ordinary requests stay isolated, explicit SDK calls opt in.
382:   Restore is allowed only into a fresh executor state and must match its actual
383:   frontier; cache insertion happens only after committed state-only prompt progress.
384:   Qwen enables retention only for paged state, with model-local `active` entry and
385:   `active*context` semantic-token bounds. Original0.8B page4/max_append64 prompt65
386:   physically proves65 warm /1 same-scope /65 different-scope model-token work,
387:   identical generated ID and clean retirement in172.98s. The first run exposed and
388:   fixed cache-before-resident retirement ordering. Trusted server-side HTTP scope
389:   injection is now accepted: one `ServerBuilder` scope is fixed by the embedding
390:   application, absent from the wire format, and unaware services fail closed. The
391:   physical original0.8B prompt65 HTTP proof records65 warm /1 same scope /65 different
392:   server scope with exact output ID and clean shutdown in171.97s. **Actual FIFO/
393:   weight eviction with live snapshot owners and explicit stale-scope cache invalidation
394:   is now accepted.** `invalidate_prefix_scope` is the trusted retained-cache purge and
395:   reports exact removed entry/weight totals; it does not cancel already-admitted
396:   work. Stop/drain stale-scope admission first; server stop/scope switch alone does
397:   not purge.
398:   Qwen uses `2*active` entries and `active*context` semantic weight. Physical active1
399:   page4/max_append64 proves prefix64 FIFO oldest miss/newer hit, purge2/128 + invalidated
400:   scope miss, and independent512-weight pressure: prefix320 hit1 while evicted prefix256
401:   makes prompt257 replay257; final purge2/448, repeat purge0, clean shutdown in
402:  179.12s. **Externally imposed resident page-pool pressure is now accepted:** a
403:   qualification-only owner pins real pages from the same finite pool around a cached
404:   prefix64. Original0.8B page4/max_append64 holds5952 pages, fills6144/6144, gets
405:   prelaunch Capacity without root mutation or vector/staging leakage, then after
406:   release reproduces the complete baseline logits exactly and retires all pages
407:   before clean shutdown in168.51s. One startup SIGTERM attempt is incomplete only.
408:   Minimal production diagnostics expose current prefix-cache occupancy/bounds and
409:   paged-pool allocated/idle/occupied/max/page-byte values. No cumulative hot-path
410:   pressure counters were added. External host/device OOM and future multi-checkpoint
411:   domain sets remain outside this checkpoint.
412:   **25 Nov typed compatibility/admission follow-up accepted (`2df8cbb3`):**
413:   original resident service/plan/source/geometry/numerical/port/storage identity
414:   is shared by snapshots and sessions; independent preparations cannot reuse state.
415:   Restore/fork validate source/service/capacity/complete-domain/lease generations
416:   and paged logical frontiers and publish through the same root swap. State-only
417:   lookup excludes the last token (prompt64 against cached64 replays64, correct
418:   output). Admit cache backing before native prep; bound retained token arrays
419:   plus one incoming prefix; charge actual variable cloned snapshot metadata
420:   alongside existing control allowance. Drop cache and backing before resident
421:   retirement. Physical fork/restore, paged/contiguous B1, scoped reuse/eviction/
422:   exact-length/HTTP and native HIP regressions pass. No new portable persistence
423:   or cross-owner/provider restore. Missing recurrent state still requires an
424:   earlier complete checkpoint/recompute, never a KV-only full hit. Exact gates and serving owner:
425:   [prefix design](../serving-and-speculation/BATCHING_AND_PAGED_STATE.md#complete-domain-prefix-reuse-and-cow--memory-plan).
441: - [ ] Extend to other qualified resident layouts/providers and general
442:   compatibility/security/tenant store UX, disk capacity and
443:   restore-versus-recompute. Diagnostic portable checkpoints are not
444:   product cross-provider persistence.
445: - [ ] Add immutable source extents and bounded coalesced IO/materialization with stable logical tensor identity, genuine copy readiness and original IO/device reader leases.
446: - [ ] Protect encoded-but-unsubmitted/captured expert reads and generation ABA; support mixed-codec expert size classes under one byte cap, source-exact routing and hot-cache preservation.
447: - [ ] Add bounded PLE/Engram lookup rows and separately governed disk state/media tiers; fully resident execution must incur no mandatory IO or host cache dispatch.
448: - [ ] Test source-exact resident/streamed parity, short/corrupt IO, cancellation, safe victim refusal, SSD QoS and actual physical lifetime before eviction/free-space claims.
449: - [ ] Optional direct/async/platform IO needs equivalent semantics and explicit support; large memory-pressure/SSD experiments require approved hardware/source scope.
451: - [ ] Distinguish logical envelope retirement from driver allocator pages/pools, fragmentation, cross-process use and actual OS release. Driver VRAM/GTT usage/max counters are not reservations or immediately free bytes.
452: - [ ] Prove trusted teardown/recovery before reclaiming unknown-work handles or clearing a fence. Diagnostic service sync/readback permits inspection/exit only and cannot synthesize Ready, retire unknown events or authorize reuse.
529: - [ ] SSD-01..32: run the [full source/residency suite](../../research/SSD_STREAMING_IMPLEMENTATION_ROADMAP_AND_ACCEPTANCE_2026-10-15.md) for each claimed backing/backend; mock tiers cannot qualify physical IO or trained streaming.
530: - [ ] SSD-14/15: encoded-but-unsubmitted, queued and captured readers prevent generation-ABA slot reuse through actual GPU completion/cancellation.
531: - [ ] SSD-21/22/24: mixed Q2/Q4 expert sizes share one byte cap with safe victims and fair progress; hot-cache retention cannot bypass admission.
532: - [ ] SSD-32: an authorized original trained streamed model matches complete resident logits/StateRoot; measure cold/warm IO, staging, OS headroom and genuine resource lifetime.
~~~~

### [.agents/plans/model-acquisition/TODO.md](../plans/model-acquisition/TODO.md) — 19 unresolved markers

~~~~text
377: - [ ] Extend advanced forced interactive/JSONL and aggregate component
378:   control over independently published original receipts; keep exact
379:   provider refusal and cancellation semantics.
381: - [ ] Canonical source-pull execution order (18 Oct decision): Qwen3.5 .8B/
382:   2B, Qwen3.8 27B; Gemma4 E2B/26B-A4B/31B, each exact chosen GGUF
383:   quants or indexed original safetensors; QwenImage2.1 original
384:   Diffusers components; Qwen3 TTS/ASR; Whisper; NVIDIA Canary and
385:   NeMo Parakeet; Qwen3.8 Flash-Next original plus separately sourced
386:   Q2/~3.5-bpw/Q4 quants and independently verified MTP/n-gram
387:   embedding banks. Do not guess literal Q3.5 quant tags or conflate
388:   runtime speculative n-gram with trained n-gram checkpoint tensors.
389:   Implement exact typed presets/component assembly and run real
390:   metadata-only then authorized actual payload/sidecar/checksum
391:   publication gates. DeepSeek V4.1 Flash and GLM5.3 Flash have an
392:   explicit NO PAYLOAD download hold; prepare metadata pipeline only.
393: - [ ] Extend the new bounded HF/GitHub Auxiliary-only --bundle to
394:   source-verified Diffusers model_index nested shard sets, ASR/TTS
395:   processors/codecs, NeMo archive streaming/extraction and
396:   release-specific recursive dependencies under one pin/manifest.
397:   Current --bundle is a user-selected subset and cannot claim a
398:   complete model/codec assembly, support auto-loading, or run without
399:   newly qualified source parser and license checks.
401: - [ ] Extend cloud profiles beyond the narrow qualified AWS FULL_OBJECT SHA256 object route: broader S3 partitions/compatible vendors/credentials/checksum algorithms, Azure latest/SAS/managed chains and per-object digest-map prefixes. Keep explicit version, caller-SHA and conditional FirstSeen contracts distinct; never infer immutable versions or trusted raw hashes from unqualified validators or multipart COMPOSITE checksums.
402: - [ ] Extend ModelScope only through separately qualified alias authority and redirect/CDN/mirror/credential profiles. Selected missing sizes now use exact pinned HEAD; no alias-to-commit authority was established from revision echoes, timestamps or file last-modified IDs. The current profile does not close broader parity or independently prove server revision/membership behavior.
403: - [ ] Measure and qualify native Xet fetch coalescing/chunk caching and any persistent native-prefix protocol. The current bounded per-intersection complete-signed-set reader has counted overfetch/refetch and no performance claim; native process interruption restarts a full transfer. Keep per-term/partial-xorb proof limits, fixed CAS scope and existing elapsed-time/HTTP-trailer boundaries explicit.
404: - [ ] Close remaining finite convenience/compatibility gaps: required sidecar/adapter/sparse-quant policy, bounded auxiliary/presets, mirror/endpoint/env precedence and gate/license diagnostics. Preserve explicit source authority/no silent numeric fallback and separate loader/runtime plans.
406: - [ ] Qualify any future v1 migration only after separate authorization and proven reader/process quiescence; current removal supports originally protected roots only.
407: - [ ] Extend cleanup only through separately qualified exact ownership/recovery protocols for dead retained staging/general orphans and any later eviction policy. Current exact-snapshot operation deliberately retains them and every stable lock inode.
408: - [ ] Any additional lazy/raw-path/mmap consumer must adopt managed descriptor/open ownership or remain externally quiescent during removal. Observed named cache charge is not a process-global final-close/physical-free-space attestation.
409: - [ ] Qualify additional filesystem pairs before extending the explicit allowlist. Different st_dev values alone do not prove independent backing objects for Btrfs subvolumes, overlay/FUSE or virtualized views; retain branch-specific root device/mount checks and complete inode/ancestry exclusion. No unsupported pair gains acceptance from generic mocks.
410: - [ ] Separately authorize any needed namespace-only hidden-bind/subvolume/overlay filesystem fixtures before mount operations. Real target-root/unselected inode aliases and hidden bind aliases in both directions remain NOT RUN; no host mounts, block devices, new privileges or live cache actions are implied. Current tests use already existing distinct filesystems only.
411: - [ ] Independently qualify any broader local layouts, renamed/unnamespaced roots, missing mount identities or new destination-root creation before exposing support. Keep explicit selection, retained descriptors, bounded topology work and existing cancellation/rollback limits; real import still needs user-selected source/repository/commit/files and copying authorization.
412: - [ ] Finish the finite provider/loader parity ledger beyond qualified HF/GitHub/HTTPS/OCI/GCS/versioned-S3/Azure/ModelScope and explicit local-HF import slices: remaining external-cache layouts/filesystems, additional model hubs, unversioned/broader S3/Azure/compatible profiles, native Xet and broader OCI profiles. Opaque serialization/plugin loaders remain separate runtime work.
413: - [ ] Continue explicit source/license authorization and trusted-digest versus first-seen provenance through additional providers/loaders; acquisition grants no license or loading support.
417: - [ ] ACQ-A12: complete the finite provider/loader parity inventory and independently qualify remaining supported paths.
430: - [ ] Each remaining provider phase requires focused API/CLI/protocol/auth/version/range/offline/fault gates, full frozen isolated check.sh, relevant source validation, actual source-integrity and documentation/dependency/root README checks before signing.
434: - [ ] Unsupported plans fail closed until qualified: unknown-length required/model-closure files, capped resumable transfers, unqualified external-cache layouts/filesystems and remaining cloud/OCI profiles, legacy-root removal and unmanaged-reader concurrent reclamation remain explicit above. Independent full HTTP trailer validation is unqualified under the documented decoder boundary, rather than a reason these payloads universally refuse. Independent component data grants no additional loader eligibility.
435: - [ ] Linux local-filesystem semantics are qualified first. HTTP cancellation obeys configured whole-request bounds; old retained attempts consume real quota and arbitrary same-UID mutation remains outside descriptor isolation.
~~~~

### [.agents/plans/model-execution/TODO.md](../plans/model-execution/TODO.md) — 79 unresolved markers

~~~~text
81: - [ ] Phase 3 (ROCm): replace whole reference row-wise/attention
82:   stages (normalization, gated activations, QK/rotary, attention) with mature
83:   native implementations where available. Re-profile actual V2 C8 and cached
84:   decode, share compatible independent rows, keep actual state/position
85:   semantics. Then amortize large dispatch regions using existing graph/actor
86:   ownership, including device-resident sampling. No scalar microtuning
87:   campaign or general speculative provider registry.
88: - [ ] Phase 4 (CUDA): connect existing owned cuBLAS/cuBLASLt provider to ordinary
89:   resident projection/compatible dense products, not just reference diagnostics.
90:   Preserve existing native tensor-core/CUTLASS primitives; resolve fused lane
91:   divergence, and use FlashAttention/FlashInfer where appropriate. Benchmark
92:   actual correctness and C1/C8 on physical CUDA before wider selection.
93: - [ ] Phase 5 (Vulkan): evaluate reusable mature ggml Vulkan cooperative-matrix GEMM
94:   kernels (`VK_KHR_cooperative_matrix`, vendor cooperative-matrix2 where
95:   supported) under existing owned command/buffer lifecycle. There is no single
96:   portable cuBLAS-equivalent library; inspect actual BF16/FP32 capability,
97:   licensing, layouts and integration cost before selecting. Keep scalar and
98:   existing shader primitives. Do not silently cast or adopt a second engine.
99: - [ ] Phase 3 (ROCm): restore a current matched vLLM comparison when its pinned local closure is
100:   available. No new download or unrelated dependency repin just for evidence.
674: - [ ] Establish independent large-CUDA distribution/state accuracy (session task44)
675:   under its declared numerical contract. Three matching IDs/112 finite domains
676:   and CPU-logit NMSE0.124241/0.090048/0.043837 are diagnostic, not a post-hoc
677:   tolerance grant. Preserve exact small-model references and original failures.
678:   The existing large composition diagnostic now reuses the independently accepted
679:   source-call archive reader to measure full-vocabulary and all112 state-domain
680:   cross-profile error together; CUDA Conv3 is compared explicitly with the retained
681:   history tail of original Conv4, while recurrent/KV retain their logical mappings.
682:   Results remain `same_contract_reference=false`/`full_reference_acceptance=false`:
683:   no27B CUDA-contract oracle exists on l5, and the retained22,126,120,608B original
684:   archive needed for the physical rerun is no longer present on this host.
747: - [ ] Localize ROCm2B C4 warmup and C16/32/64 warmup/timed divergence (task39),
748:   without repinning expected IDs or attributing an unlocalized failure to kernels.
749: - [~] Close ordinary27B HIP HTTP504 and populated1K/2K references/serving (task11).
750:   The historical504 still predates resident75GiB routing, but the20 Nov exact-current
751:   liveness rerun is now blocked before HIP allocation by the X2 runtime closure, not by
752:   ZeroGravity serving: both installed clr7.2.4 roots require an LLVM22 library needing
753:   GLIBC_2.44 while the locked dev shell is glibc2.42. Exact `HIP_PATH/ROCM_PATH` does
754:   not bridge that ABI mismatch. The temporary reference-free HTTP probe compiled under
755:   the accepted opt3/codegen16/non-LTO test profile and private child-net isolation was
756:   accepted, but runtime loading refused before GPU allocation. X2 stayed at the exact
757:  274,448,384B/0% baseline and was restored to `b599db3`; probe/bundle removed. Repair
758:   the X2 ROCm environment separately before another27B/ROCm2B run. Update24Nov:
759:   process-local compatible loader restores actual ROCm SDK execution for all three
760:   models; see CURRENT_HANDOFF and CURRENT_BASELINE. HTTP/populated context
761:   qualification and missing approved 66-call/1K/2K references remain separate.
762:   R10 audit is complete and mixed; R11 remains staged/unlaunched. Review its older
763:   binary/budget before any fresh successor. Both context captures remain interrupted;
764:   the waived12GiB host-available threshold is not a current blocker.
765: - [ ] Continue genuine shared inter-session batching beyond current B8 decode/chunked
766:   prefill cohorts. Aligned prompt chunks up to the model's admitted chunk size share
767:   projection GEMMs while retaining private roots; B8 decode reuses the qualified B4
768:   stateful kernels as two groups. Dynamic grouping remains open, but do **not**
769:   resurrect the rejected global-active/native-B16 experiment as the next step: a
770:   same-path0.8B B16 measurement was slightly worse than two B8 groups
771:   (78.389 vs79.380 tok/s), while2B B16 regressed catastrophically (~44.9s/~42.0s
772:   warm waves). `MAX_ACTIVE_PER_WORKER=8` and native4|8 therefore remain deliberate
773:   until a materially different architecture/measurement justifies revisiting them.
774:   Bounded queueing already handles larger offered populations without widening the
775:   resident/native ceiling: same non-LTO2B47+3 run stayed exact/native-width8 at
776:   offered8/16/32 with66.24/68.10/69.06 output tok/s respectively;16 and32 are
777:   simply two/four queued B8 groups, not hidden B16/B32 execution.
778:   **21 Nov mixed-prefill grouping candidate rejected.** Current source already admits
779:   reduction-compatible mixed prompt chunks (for example48/49) into one native B8 and
780:   shares projection/GDN work, but FlashAttention only batches consecutive equal-row B4
781:   groups. A temporary source-neutral experiment kept scheduler/fairness/publication
782:   order unchanged, stably clustered only the private physical lanes by prompt-row count,
783:   restored logical session/readback order before publication, and let the existing B4
784:   attention kernel consume every equal-row quartet with per-lane fallback for leftovers.
785:   It added no GPU allocation or kernel. The exact public-SDK interleaved
786:   [49,48,49,48,49,48,49,48] +3-output cell remained singleton-exact and native width8,
787:   but matched opt1/codegen64/non-LTO measured299.985ms on unchanged source versus
788:   324.231ms and313.585ms with the candidate (~8.1% and~4.5% slower). The Qwen-specific
789:   wiring/probe was removed, but the allocation-free lane regroup/equal-group primitive is
790:   retained in `performance_candidates::LaneRegroupPlan` for Qwen2B/27B, Gemma4 and other
791:   shapes. Dynamic grouping remains open; do not select this candidate for the measured
792:   .8B cell, but do not treat that loss as a global-model rejection. Move next to the admitted populated-context
793:   ladder on L5:4K first, then8K/longer-fit cells, reusing the existing64-row contract.
794: - [ ] Optimize measured CUDA2B/HIP hot paths and run controlled comparisons after
795:   relevant correctness. Existing short eager/direct-API rates are not general
796:   HTTP/context/graph performance or proof of vLLM superiority.
1247: - [ ] Resolve ROCm2B batch-dependent output mismatch. Retained
1248:   `qwen-rocm-benchmark-3gteresx/vllm-2b-summary-r1.json` confirms C4 warmup
1249:   wave0/request_index3 differs, despite0/8 timed mismatches. C16/C32/C64 have
1250:   2/32,2/64,6/128 timed mismatches, plus1/1/3 warmup mismatches. Expected
1251:   [95726,26516,220], actual[95726,98019,271]. Warmup failures count; earlier
1252:   “C1/C4/C8 match” shorthand was incorrect. Original summaries/logs unchanged.
1253:   Failed run/source bytes/shutdown/original retirement retained. Cause unlocalized;
1254:   not established as a ROCm kernel bug. Do not repin expected IDs or count these
1255:   cells as accepted matched speedups.
1295: - [ ] Complete actual27B ordinary HIP/SDK/serve, not the finite two-position route.
1296:   Authenticated `Qwen/Qwen3.8-27B` geometry now drives the same ordinary template:
1297:   64layers,112domains, untied head,851text roles/18shards. Finite CUDA profiles
1298:   remain closed; source geometry grants no execution or numerical acceptance.
1299: - [ ] Qualify bounded weight streaming and source/physical final readers.
1300:   Whole-source BF16 storage is priced before upload; use existing source-page
1301:   transport when resident weights plus workspace/roots/native owners exceed the
1302:   default24GiB total including4GiB context. Explicit residency uses the separately
1303:   admitted75GiB total; host ledger remains16GiB. Original
1304:   source extent is separate from resident allocation; no float kernels changed.
1305: - [ ] Independently audit the original CPU capture before native comparison.
1306:   Existing approved source/image,66 genuine calls/two repeats, full64layers,
1307:   fresh47/48/49/50 and uniform/ragged512→514. Capture completed:66 calls,
1308:   19848 records,22126120608 raw bytes; contemporaneous foreground/group/container/
1309:   EOF retirement recorded before owner receipt. Independent audit now passes
1310:   all6 cases/6533182464 float words/851 roles, full digests, natural feedback and
1311:   repeats. Accepted manifest`f226ad21`; raw`d9690f46`. No HIP parity, serving or
1312:   CUDA-oracle claim follows. Accepted archive staged on x2 with full stream hashes.
1313:   NativeR4 observed child exited0:66calls/6533182464 exact words, full vocabulary,
1314:   natural feedback, independent roots and source/native final readers independently
1315:   audited in `native-observed-reviewed-r4.json`. The supervisor then failed its
1316:   stale descriptor-proxy PID/executable guard, before x2 rebooted. Observer-free
1317:   never started; original terminal/results are absent. Preserve this failed
1318:   campaign and limited observed acceptance; no full-R4/serving acceptance.
1319:   Observer-free-only continuation keeps frozenR3/references; no observed rerun.
1320:   Fresh guard pins boot/PID/starttime/executable without changing device/KFD/busy
1321:   or12GiB checks. R6 staged but superseded unexecuted after
1322:   review: mainPID0 did not ensure descendant retirement; post-start errors could
1323:   skip watcher attachment. Watch`566a9940` cancelled. R7 reviewed/fixed/staged;
1324:   pure/output controls and four stubbed-OS failure injections pass: launch
1325:   timeout, receipt failure, transient status and interrupted wait. Observation
1326:   survives these errors; signals stop only the exact owned invocation; attempted/
1327:   indeterminate scopes are not labelled unrun. R7 manifest`730bcdeb`; watch
1328:   `e89ad3c8` succeeded. Fresh guards/pins/unit/path checks passed with13650759680B
1329:   available. Observer-free R7 launched2026-11-05T12:41:41Z, invocation
1330:   `a31ca21457a24796ad21fb1ad0e3798e`. Only `state-parity`, unchangedR3/reference.
1331:   Original terminal2026-11-05T14:37:14Z proves nonce/invocation, absent owned
1332:   cgroup/KFD[]/busy0 and no supervision errors. Independent
1333:   `native-state-reviewed-r7.json` passes all66calls/3137737728 state-logit words,
1334:   611450880 KV-tail words, independent roots/natural feedback/final owners.
1335:   Job2666 completed; monitor2667 stopped after retrieval/audit. R4 supervisor
1336:   failure and missing original terminal remain; no full-R4 acceptance. Do not
1337:   rerun observed numerics, duplicate launch/restage R7 or revive R6.
1338: - [ ] Run actual-source admission, full-vocabulary/state parity, observer-free
1339:   execution, cancellation/replay/final-reader and ordinary serving checks; then
1340:   the complete assembled gate. Host tests/strict feature checks pass15 tests,
1341:   including retained finite-source boundaries and exact managed manifest.
1342:   Two earlier host failures (stale helper argument/new frontier assertion) are
1343:   retained and corrected; call-free frontier stays0, runtime calls own extents.
1344:   Evidence: XDG state `qwen-ordinary-large-rHMls4mk`; implementation integrated
1345:   by`4f1e827`, with HTTP/context exceptions preserved above.
1346: - [ ] Finish additive BF16/F32 GDN source storage qualification. Canonical host
1347:   and x2 hostR2 exposed the same exact-type refusal: all48 original large-source
1348:   `A_log` and GDN norm tensors are BF16; small-source F32 remains required.
1349:   Shared CPU-source GDN descriptors now accept both independently, retaining
1350:   original dtype/bytes and exact widening through existing upload; CUDA's F32
1351:   contract stays closed. Model-source validation still requires its pinned types.
1352:   Shared descriptor/ordinary-plan/storage regressions and strict checks pass;
1353:   R7 release diagnostics/SDK built, but actual-source final-reader test failed
1354:   on the tokenizer's old single-source pin. R6's test-helper type error remains.
1355:   Tokenizer JSON differs only by seven added-token entries (33 total; max21B):
1356:   source-specific BPE/automata accounting now shares the same authenticated
1357:   NFC/Split/ByteLevel machinery and global owner; original26-token pricing stays.
1358:   R9 assembled validation passes:22 tokenizer plus8 owner-accounting tests per
1359:   exact source, strict HIP/CUDA consumers, release diagnostics/SDK and actual27B
1360:   canonical source/plan/tokenizer/final-reader test. Small-source pricing stays
1361:   unchanged; R8's incorrect special-token test expectation remains as a failure.
1362:   Canonical514/active2 plan streams53786705920B original weights; host14720655712B,
1363:   physical device15439853568B including4GiB context; estimates, not reservations.
1364:   Repaired binariesR2 frozen unqualified. x2 host-capacityR3 passes all28 cells:
1365:   contexts512/1K/2K active1/2/4/8 admitted as estimates;4K–32K refused under the
1366:   original host/device caps. Source recheck and exact-invocation idle retirement
1367:   pass. Frozen binariesR1 remain unqualified and predate both repairs.
1368:   x2 hostR1 failed before source IO on noclobber lock opening; R2 source refusal
1369:   retired its exact invocation/cgroup with KFD empty; both were host-only.
1370:   Independent1K/2K original capture was interrupted by SIGTERM, child137; owned
1371:   foreground/group/container/EOF retirement is recorded. Partial archive remains
1372:   unqualified:104calls/3888records/all104 digests were the target, not a result.
1373:   Host-private successorR2 preserved byte-identical inputs and R1's missing
1374:   original-name absence evidence; it also received SIGTERM/child137, after11/104
1375:   logged calls. Cleanup recorded; both partial archives remain unqualified.
1376:   No active capture or accepted populated-context reference. A successor requires
1377:   fresh identity and reviewed supervision. Shared scopes preserve small-model
1378:   1K–16K and add large1K/2K schedules. R10 passes19 host reader controls/strict
1379:   checks and release assembly; frozenR3 production bytes equal accepted hostR2.
1380:   Capacity/refusal and reader preparation do not qualify actual contexts.
1381:   Sequential C1 HTTP JSON/SSE/disconnect/replay coverage is authored separately
1382:   from concurrency; existing population scopes stay unchanged. R11 stopped on
1383:   a test-only constant-assert lint; runtime guard corrected. R12 strict/release
1384:   assembly passes (1 host test,8 physical scopes ignored); SDK-onlyR4 frozen.
1385:   Only HTTP qualification support differs from frozenR3; production and native
1386:   example bytes are unchanged. No new HTTP physical acceptance.
1387:   R5 lifecycle/SDK/sequential-HTTP campaign and terminal retrieval/audit are
1388:   prepared only:13 fresh scopes. Its original full-R4 gate cannot pass after
1389:   interruption. Successor R8 consumes limited observedR4 + accepted observer-freeR7
1390:   honestly; same nativeR3/SDK-onlyR4 and deadlines/limits. Exact-owned supervisor
1391:   failure controls and attempted/indeterminate/unrun runner controls pass;
1392:   manifest`e8a35f90`, never staged/launched. Review found inter-case descendant,
1393:   signal-handler lock and unit-name signal races. R8 preserved; R9 preparation
1394:   adds case-cgroup checks, lock-free flags and pidfd-bound cancellation. Eleven
1395:   ownership/four supervision controls pass; independent review found no launch
1396:   blocker. Fresh exact guard/pins/paths passed; R9 exclusively staged and launched
1397:   once2026-11-05T16:20:39Z, invocation`26c8c471c5f44b63896052bae63d596e`,
1398:   nonce`9753bbec952d432abcf9562a33ddf8c8`; manifest`05e6cd11`, supervisor2729,
1399:   monitor2730. SDK-complete/disconnect and healthy/session-known/snapshot-known/
1400:   pending-known/readback-known/snapshot-readback-known exited0 with pass markers.
1401:   HTTP-sequential failed101/HTTP504 at `qwen_hip_http.rs:98`, unchanged180s deadline;
1402:   303.74s is cell duration. Stable live snapshot binds exact invocation and the
1403:   HTTP after-boundary runner-only cgroup/KFD[]/busy0. All13 scopes then completed;
1404:   independent retrieval/audit verifies12 semantic scopes plus HTTP101/504,
1405:   authentic original retirement and no supervision errors.52 ordered case events/
1406:   26 runner-only boundaries checked; indeterminate/unrun empty. Disposition:
1407:   `ordinary-semantic-disposition-r9.json`. Original auditor's clean-supervision
1408:   predicate also required semantic process success; supplemental disposition
1409:   separates successful retirement from HTTP failure without changing old bytes.
1410:   Staging log's inherited uppercase R8 labels are stale; filenames/hashes are R9.
1411:   Never manufacture R4 terminal evidence. Retained-service bytes include state/
1412:   scratch; do not confuse them with weight residency.
1413:   Main now implements shared strided projection output/grouped partial windows
1414:   for the concrete HTTP failure. Float operations/reductions/BF16, page/full-K,
1415:   owners/cancellation and budgets are unchanged; all direct/prepared callers
1416:   updated. Eight host mapping/status/scratch/extent controls and strict HIP/CUDA/
1417:   Vulkan consumers and corrected R14 release assembly pass. Preserve R13's
1418:   test-only integer-inference failure. FrozenR5 executed in R10: all15 scopes
1419:   attempted,14 exited0 and HTTP101/504 retained. Logs report66 observed plus66
1420:   observer-free calls, zero mismatches; independent retrieval/audit remains open.
1421:   Original terminal records retirement for invocation`6448edba84244c1b9eb339908a7a2618`,
1422:   nonce`42cfc1bc2aa8407ba8b1ceb07143254c`, owned cgroup absent/KFD[]/busy0;
1423:   output errors include `BrokenPipeError(32, 'Broken pipe')`. No clean-supervision,
1424:   whole-campaign or numerical acceptance before independent disposition.
1425:   R10 is finished, not live; never restage/relaunch or overwrite its artifacts.
1426:   Grouping did not resolve HTTP504; localize measured latency without changing
1427:   deadlines or budgets. Row grouping is not cross-session native batching.
1428:   No measured speedup, populated-context acceptance, current full gate or signature.
1429:   Owner requires shared, additive capabilities—not per-model forks.
1430: - [ ] Complete independent CUDA references on l5; Main owns implementation and
1431:   execution, no new or resumed subagents. Authenticate runtime/provider/source/kernel identities; fresh
1432:   captures do not replace missing legacy executable authority.
1433:   XDG run `qwen-cuda-regeneration-tDW0EJ1I`: exact RTX3060 UUID/SM86, driver595.91.07;
1434:   original retained image passed a CUDA tensor probe. Runtime-addressable memory
1435:   is6088359936B, distinct from physical6GiB; initial equality assertion failed and
1436:   is retained. Upstream vLLM image then failed LLM import on CUDA13 torchcodec.
1437:   Existing CPU-codec image`f57d77f5` (no installation/download) runs both0.8B/2B:
1438:   two actual47-token greedy requests/model produce `[264,264,264]`; source hashes/
1439:   identities before/after and original foreground/group/container/EOF inspected.
1440:   Earlier controller mount exposed guest-writable CID/receipts; review limitation
1441:   retained with supplemental current-name absence, never replacement receipts.
1442:   New ownerR4 mounts only payload/cache, keeps control files host-private and
1443:   independently checks the original container name after cleanup.
1444:   Torch2.13/cu129, vLLM0.31, Transformers5.17, Triton3.7.1; eager/V2/FA2/Triton-GDN.
1445:   Full-vocabulary R1 refused callable serialization; R2's FP32-only assertion
1446:   rejected native BF16. Deferred async exception reporting masked that first
1447:   error as uncleared pending state, not a proven hook bypass/forward overlap.
1448:   R3 snapshots public `compute_logits` native returns and actual input buffers,
1449:   defers CPU drain, returns the same object and exactly restores hooks/methods.
1450:   No head dtype, arithmetic, async or serialization-security changes. Three
1451:   CPU-only controls pass; both models'48calls/16requests independently audited:
1452:   fresh47–50 uniform/alphabet, two repeats, genuine feedback, all248320 vocabulary
1453:   words/call,11919360BF16 words/23838720rawB per model; exact repeats/source pins/
1454:   foreground-group-container-EOF retirement. Raw08`36350f39`,2b`f0b70f7a`;
1455:   `full-vocabulary-{08,2b}-reviewed-r3.json` are discovery-only acceptance.
1456:   Both genuinely unobserved16-request runs independently match public outputs
1457:   and termination, with exact source/options/retirement verified:
1458:   `unobserved-{08,2b}-reviewed-r1.json`. No plain full-vocabulary/state-bit claim.
1459:   Bounded async transport passes host and independent actual-CUDA native-bit/
1460:   stride/mutation controls:40MiB GPU arena +24MiB minimal snapshots,128MiB pinned
1461:   host; nonblocking producer, independent GPU/host retirement. Both expanded
1462:   captures completed and retired; `expanded-{08,2b}-reviewed-r1.json` audit every
1463:   archived byte (5925722312 /10564249800B), all24 original layer/deferred-add
1464:   boundaries and4032 physical state snapshots/model. Captured valid prefixes
1465:   preserve continuity/repeat bits; live views are disjoint. All320 original text
1466:   parameters associate with actual loaded packing/casts, and parameters/buffers
1467:   match before/after. This producer casts18 original F32 GDN norms/model to BF16;
1468:   native source contracts unchanged. Full vocabulary equals R3, public output/
1469:   termination equals unobserved runs. These expanded captures are not full-reference
1470:   passes: they omit actual context slot mapping, owner-boundary mutable metadata
1471:   and classification inputs; consumer successors below extend that coverage.
1472:   Constructor buffers are not independently regenerated.
1473:   KV layout is head×token×(K256|V256), not a leading K/V axis. Fresh Mamba backing
1474:   is invalid history, not assumed zero. Producer/native CUDA use history3;
1475:   CPU/HIP history4 is separate. Slab padding/plain-internal invariance remain
1476:   unclaimed. At this expanded-capture stage CUPTI was inspected only; later
1477:   standalone lineage diagnostics below do not grant model-wide ABI, native CUDA,
1478:   matched performance or later roadmap acceptance. R1 review found incomplete
1479:   chunk-storage attribution, metadata-reference provenance and lifecycle order,
1480:   not capture corruption. R2 strengthens all three; both actual metadata archives
1481:   and9 adversarial controls pass; both full R2 audits passed. Review then found
1482:   inventory subview order was not linked to archive logical offsets. R3 fixes
1483:   that link and rejects a coordinated descriptor/inventory swap: both metadata
1484:   archives and10 adversaries pass. `expanded-{08,2b}-reviewed-r3.json` rehash both
1485:   unchanged native/vocabulary archives, bind792/1334 chunks and explicitly reuse
1486:   pinned R2 byte/layer/source checks. R1/R2 remain immutable; current scoped
1487:   acceptance is R3, not exhaustive effective-state/reference/native closure.
1488:   Read-only R3 review found no remaining concrete order hole; not reexecution.
1489:   Actual-consumer observer captures KV/context mapping, conv/recurrent arguments,
1490:   gather/mask/scatter and classification CPU/GPU inputs;12 real-Torch CPU controls
1491:   pass. 08 consumerR2 completed48calls/37848records/6873374472B with original
1492:   foreground/group/container/name/EOF retirement; vocabulary/public outputs match.
1493:   Independent08 consumer auditR3 now passes all48 calls/8432 events, actual
1494:   classification/counters/mappings, conv mutation and gather/mask/chunk/scatter
1495:   bytes/order. Earlier audits wrongly treated padded GPU schedules as active CPU
1496:   lists and prompt progress as total decode progress. Source-derived corrections
1497:   check full padded allocations/roles, not weakened expectations;19 adversaries
1498:   pass. `consumer-08-reviewed-r3.json`; failed auditors/logs remain immutable.
1499:   2B consumerR2 failed at step28 after9 public requests; partial5987762129B and
1500:   original cleanup retained. R3 removed duplicate chunk-return copying but failed
1501:   at step1. Both original tracebacks identify host-ring admission, not GPU or
1502:   record-budget exhaustion. Neither failure has a complete archive/contemporaneous
1503:   GPU-after receipt; later idle observations never backfill those omissions.
1504:   GPU retirement was already independent of disk work. Corrected transport splits
1505:   GPU snapshot admission from coordinator-owned host admission/copy; host leases
1506:   retire after raw consumption, record credits after bounded manifest writes.
1507:   Drain acknowledges both GPU and exact host retirement. First failure is latched
1508:   before diagnostics; uncertainty fences/quarantines before logging and survives
1509:   query/close races. No producer waits, async/options/cap changes or extra temporaries.
1510:   Fresh2B R4 completed48calls/38102records/11209912072B; all original foreground/
1511:   group/container/name/EOF retirement independently verified, GPU before/after41MiB.
1512:   Full audit verifies source bytes, all24 layers/4032 states/8432 consumer events,
1513:   exact vocabulary/public outputs, masks/conv/scatter and all transport frontiers.
1514:   `consumer-2b-reviewed-r4.json`;288 real scatter links plus7 adversaries pass.
1515:   GPU transport peak11034368B, host90827008B; final used/padding/pending0, no
1516:   failure/unknown/quarantine. Limits remain40MiB GPU+24MiB minimal/128MiB host.
1517:   Transport controls:24 host scopes/20000 allocator operations/600records;
1518:   actual CUDA9 native-bit/stride/mutation records and3 stalled-writer records.
1519:   Latest `tensor-archive-r6.py` / `expanded-consumer-observation-r7.py` fixes a
1520:   subsequent minimal-error publication race with deterministic interleaving and
1521:   callback-fallback controls. Successful GPU/producer paths are byte-identical;
1522:   reuse physical success evidence explicitly, no model recapture or physical
1523:   uncertainty claim for that delta. Original clean R4 capture remains unchanged.
1524:   `consumer-boundaries-reviewed-r5.json` binds both scoped audits and controls.
1525:   Raw K/V candidateR8 and audit bodyR4 now observe actual entry arguments and
1526:   compare logical page mutation through the existing transport. Both actual
1527:   layouts/24 adversaries and original-once/same-return wrapper controls pass;
1528:   no new model execution, backing-padding or complete effective-readset acceptance.
1529:   Native callback recorder/offline typed lineage/checked ELF decoder assembled
1530:   separately as qualification support. Runtime-compatible build replaces the
1531:   failed host-libc probe; all failed bytes/logs stay. R3 repairs rejected nested
1532:   calls incorrectly popping outer frames;15 native host scopes/400 concurrent
1533:   calls/eight wire adversaries pass. No callback CUDA/GIL/file IO/capacity waits.
1534:   Standalone RTX3060 R3 independently verifies13 primitive cases/launches, three
1535:   public-edge-associated executable images,5544 callback records, original
1536:   foreground/group/container/name/EOF retirement and no inter-case synchronization
1537:   in its driver trace. Cubin-derived extents match eight successful parameter
1538:   queries plus three terminal invalid-index responses;34 metadata/eight query
1539:   adversaries pass. `cuda-parameter-lineage-reviewed-r4.json`. Original CUDA-info
1540:   metadata has two explicitly checked link formats; retain R1's decoder refusal.
1541:   Earlier R2 control staging waits and unload-fixture failure remain recorded.
1542:   Independent constructor regeneration now passes for both source configs:
1543:   `constructors-lineage-reviewed-r3.json`,268435488 fresh original CUDA bytes,
1544:   all30 named buffers/model before/after (25 unique,5 aliases/phase), complete
1545:   fresh archive/source/config and original retirement. Accepted consumer raw-byte
1546:   audits explicitly reused; no observed tensor bytes supplied as constructor input.
1547:   Its24 launches/10 images have typed lineage, but **zero captured arguments**:
1548:   producer-version129 was unsupported. These24 ENTRY gaps are permanent for R3.
1549:   Earlier constructor R1 missing-config-context and R2 enum-check failures stay;
1550:   both precede kernels and lack contemporaneous GPU-after receipts.
1551:   Native argument recorder `cuda-lineage-r6/`, approved-runtime buildR5, now
1552:   supports witnessed123/128/129 metadata, live typed handles, declared host extents,
1553:   packed extra/Ex attributes and explicit unmarked threads.32 native scopes,
1554:   400 concurrent calls,14 actual schemas,166 malformed metadata cases and8 wire
1555:   adversaries pass on CPU; source/binary pins and original retirement inspected.
1556:   Binary`22a8dcaa2bc33f6e96a5bb141c4f60a0631b3ff96999a78d682f9730b8309f5f`.
1557:   Initial native parser refusal remains. No GPU/model run of the new recorder,
1558:   worker bootstrap/event attribution integration or actual GPU argument acceptance.
1559:   Candidate `argument-wire-r1.py` still points to R5 readers; prepare/test an
1560:   explicit successor, not an in-place rewrite of executed capture/audit inputs.
1561:   Paused here by owner; no next campaign launched. Resume combined model-wide
1562:   argument/image observation, effective-state read/write and recurrent-axis proof,
1563:   then native CUDA/serving. Native CUDA history3 already exists; do not impose
1564:   CPU/HIP history4. No aggregate-field/model-ABI/full-reference/native acceptance.
1565:   No retained package found in the inspected readable local/remote scopes;
1566:   five local container housekeeping paths denied.
1567:   Exact2B cubin and original FLA/causal-conv/FlashAttention qualification remain
1568:   unavailable. No CPU substitution or source repin; accepted new discovery is
1569:   not full CUDA reference or legacy-authority acceptance.
1573: - [ ] Extend populated contexts beyond accepted0.8B/2B HIP scopes to remaining
1574:   CUDA/general27B routes; retain full-logit/state truth, cancellation and physical
1575:   ownership at actual extents. The older32K refusal was a pre-streaming capacity result;
1576:   current public2B32K load is admitted; execution is now localized to a reproducible
1577:   >600s first-response performance boundary rather than an indeterminate external kill.
1578: - [ ] Extend/qualify serving concurrency through C1..C64 and general27B/CUDA;
1579:   preserve missing-source/kernel/runtime refusals. Offered C64/active≤8 singleton
1580:   execution is only a baseline, not native-batch acceptance.
1581: - [ ] Measure matched vLLM workloads as each implemented path passes its relevant
1582:   correctness/capacity checks; do not wait for the whole roadmap or batching.
1583:   Match workload/settings/concurrency, report losses and limits. Unqualified
1584:   experiments stay exploratory; no superiority claim without matched measurements.
1588: - [ ] Owner requirement: independently configure queue depth, active admission,
1589:   batch members and per-step token budget; enable genuine continuous batching,
1590:   including all eligible admitted work when it fits. Remove the universal
1591:   eight-session/singleton policy ceiling, not physical/source/kernel safeguards.
1592:   Reuse shared cohorts, original flights/roots and the arbiter. Cross-session
1593:   kernels/state mappings and partial cancellation must work before promotion;
1594:   no metadata-only batching or silent relabeling of serial work. This is queued,
1595:   not part of the current populated-context correctness implementation.
1606: - [ ] Owner30Oct explicitly resumes Qwen ROCm (HIP) on `pvl-x2` independently
1607:   of wider CUDA reference gaps; approved2B/9B/27B model downloads into canonical
1608:   XDG cache are allowed if needed. No subagents, source substitutions, SDK
1609:   downloads or device-privilege changes. Current27B finite source/natural reports
1610:   pass exact full64/fullV/all112-state with supervised exit/retirement in signed
1611:   `825ade8`. Ordinary production/serving and general27B contexts remain open;
1612:   bounded small-model HIP results follow below.
1613: - [ ] Implement ordinary HIP through shared operators/physical owners, not a
1614:   backend flag on the CUDA plan. Declare actual numerical/storage contracts;
1615:   generalize real model geometry beyond24 layers/42 domains for27B. Preserve
1616:   source-priced admission, independent roots, flights and physical retirement.
1620: - [ ] Generalize actual geometry/backend execution for27B. Finite27B still
1621:   pins2 positions/112 domains; it is not the ordinary small-model source path.
1646: - [ ] Qualify `.8B` then `2B` full model, cached continuation, faults, shutdown
1647:   and exact resource fit before enabling an ordinary serving route. Long-context,
1648:   CPU, HIP and Vulkan cells remain independent gates.
1652: - [~] Every existing/future registry target follows the shared
1653:   [correctness policy](../numerics-and-context/CORRECTNESS_POLICY.md): declared
1654:   per-op accuracy, pinned independent task/model quality, separate determinism
1655:   and exact source/state/physical ownership. Registry metadata is no execution
1656:   or model-access authorization; unsupported adapters remain unqualified.
1657: - [ ] Implement/qualify missing full-model/context/backend cells, beginning
1658:   .8B47–50 then genuine long context on CPU/CUDA/ROCm/Vulkan. ROOT coordinates
1659:   exclusively GPT-6.1 Sol xhigh lanes. Older Pi/Astra assignments and performance
1660:   sequence below are superseded. The later30Oct mission resumes independent ROCm
1661:   and correctness-gated timing; older blanket deferrals below are historical.
1686: - [~] Three passes executed on pvl-x2.
1687:   Root causes of passes 1-2 fixed and verified on hardware
1688:   (`06b89da` E0308, `c217c35` V parents, `e6ab074` census,
1689:   `3941a3c` Gemma envelope). Pass 3: builds/admins/binds all PASS;
1690:   h-known-qwen-reload + h-qwen-final-reader-unknown PASS (full
1691:   semantic closures); three real failures preserved
1692:   (h-gemma-feedback-unknown: session-state identity/policy/fence
1693:   assert + `scope closes only between completed proposals`;
1694:   g-resident: parity row-upload fixture contract vs legitimate PLE
1695:   row dedup; v-resident: adapter validator requires a false-permit
1696:   field the producer never emits — physical case itself passed);
1697:   19 cases stopped as dependents. No product defect established
1698:   (qualification-gated paths).
1702: - [~] Qwen natural2-position independent original comparison and Gemma learned
1703:   PLE row-hit/H2D/full-state scope, retaining existing quotas/owners. ROOT reviewed
1704:   Astra's complete next-unit design; two isolated Pi implementation/controller
1705:   preparation lanes active, no GPU/reference execution before review. Qwen old
1706:   teacher capture requires independently verified original fullV/prior/source
1707:   join; otherwise fresh natural reference. Gemma reuses its genuine natural
1708:   matrix and adds an actual-hit full35/fullV comparison plus final-owner scopes.
1709:   No new public
1710:   serving, source21 x2 artifact, general-fit or speed grant from source20 fixture.
1711: - [ ] Gemma PLE twofixture source draft and prepared controller need corrective
1712:   review before merge/physical work. ROOT finds diagnosticD2H exactly1B gate
1713:   instead of positive and regressed dataFD/bounds. Original valid source20
1714:   math/lifetime results unchanged; new all-hit/final-reader scope unexecuted.
1719: - [ ] Close isolated source113 as one checked model unit, then actual x2 source/
1720:   state/fault/lifetime qualification: Qwen full64/all112, Gemma resident20/all35/
1721:   fullV/all15KV and observer-free natural feedback. Grouped isolated CPU/strict
1722:   checks PASS:5strict configurations,907CPU library tests/2comparators,46explicitly
1723:   ignored cases. Exact source113/P1/P2 and original400v4 base remain unchanged;
1724:   contextual410 continuous/compiler reconciliation is reviewed and affected
1725:   strict/166boundary/7compiler-owner checks pass with all jobs closed;21physical
1726:   cases explicitly ignored. Later RMS3/source6 excluded. Gemma's finite actual
1727:   Resident20 and Qwen full64/armed-producer fault results follow below; Qwen's
1728:   final Pending/Snapshot reader fault coverage is now closed on frozen469 below.
1729:   Keep current C8 assembly separate.
1730: - [ ] Reconcile frozen models113/P1 hooks and Gemma final-feedback fix with current
1731:   shared CUDA readiness source, preserve arithmetic/backend boundaries, perform
1732:   affected assembled checks and final complete gate before model merge acceptance.
1733: - [ ] Qwen27: qualify natural and temporal decode/prefill, residency/admitted
1734:   prefetch, packed projections, eligible optimized GEMM/attention/recurrent GDN,
1735:   hybrid prefix checkpoints and serving through shared engine primitives.
1736:   Current native route requires streamed weights and finite teacherfrontiers0/1;
1737:   shared64-lane CPU-order BF16 dot is not TensorCore/BLAS. Preserve source/cast/
1738:   reduction profiles rather than inheriting small-model CUDA permits.
1739: - [ ] Gemma E2B final-reader ownership: source466 review finds Session/Snapshot
1740:   drop `loaded` before final committed feedback's independent12-byte Output
1741:   lease. Native/modules/managed-source guard can retire before a true final
1742:   feedback release refusal. Pending already has safe ordering. Isolated Sol
1743:   fix+fixture unit authorized: place `loaded` after inner+feedback; Session/
1744:   Snapshot × healthy/exact final Output-lease refusal, Weak-only diagnostics and
1745:   original actual service/native/source/quarantine owners. Preserve frozen466,
1746:   completed Qwen P1 and readiness sources. Frozen469 ten-path fix+fixture passes
1747:   grouped6control/4probe/all-target strict checks; independent Sol review and ROOT
1748:   complete source/control/cost/archive review pass. ROOT authorized one bounded
1749:   fresh admin/offlinebuild/four-case unit. Admin33537/offline GPU-free build73520
1750:   CLOSED0; first Session-known28952 FAILED_CLOSED at semantic verifier's exact
1751:   `device_with_six_roots_bytes` formula after native child exited0. Actual groups/
1752:   one-reap/fullEOF/source35FD/cache-parent closure healthy. ROOT and independent
1753:   Sol identify verifier-only omission: greedy12336B+six256B alignment guards
1754:   missing from both exact expected totals; production sums/caps are correct.
1755:   Corrected local verifier6controls PASS; new saved Session-known revalidation
1756:   PASS without GPU replay, original FAILED receipts unchanged. ROOT reviewed
1757:   full225line local collector delta/17pins and authorized remaining3 serial
1758:   native cases, exact-known-error-only/source35/cache/binary/originalowner
1759:   closure required. All4actual scopes now close with corrected local acceptance;
1760:   ROOT independently verifies every report/source/cache/original+reader owner.
1761:   Known:184→0modules/sourcecloseOK/model0/context4GiBdevice+2GiBhost. Fault:
1762:   actual12Bfinalreader refusal precedes native retirement,184modules/617handles
1763:   retained/full20GiBfenced/sourceclose refused. Every full35/30planes/fullV/
1764:   one natural commit; original failed remote collectors and SSH1 remain explicit.
1765:   No Session replay/Rust rebuild/serving or performance grant. Existing source466
1766:   cache reused with only actual publication/activity locks writable; no second
1767:   10GB import. Current11+118path isolated combined8package/alltarget strict and
1768:   189CPU PASS/33explicitphysicalignored; original source/owner/EOF closures pass,
1769:   first controller-only compilerfile cap failure preserved. Current12 differs
1770:   only by test-fixture rustfmt outside118. ROOT now adopted models118 plus the
1771:   source-bound2B cachedB4 unit together:173selected/172effective/46new paths,
1772:   fresh mtimes, original README/lock and all unrelated source preserved. Current13
1773:   has1629full files/492selected; combined strict/20sharedCPU/5ABI-source controls/
1774:   authentic2B CPU inventory/reopen pass. Original excluded-vendor Cargo refusals
1775:   remain failed, not executed vendor-test evidence. Optimized13 CLOSED0 and all
1776:   original19physical scopes/healthy2Btrajectory PASS; original2B same-prior30calls/
1777:   8112sections/fullstate/fullV exact. New2B fault global-vs-flight count mismatch
1778:   is diagnosed; fixture-only15 correction builds, preserving every graph/source/
1779:   fence guard. Current assembled phase and final
1780:   assembled gate still required. Prior.8B measured checkpoint remains source11/12.
1781:   Combined assembly results.
1782:   Four-case report.
1783:   Independent source evidence.
1784: - [ ] Gemma E2B/E4B: dependency-based shared-tail fast prefill, PLE row reuse,
1785:   optimized packed projections and eligible attention, physical SWA reclamation.
1786:   E2B15owners/20shared tail exists; E4B needs24/18 and a new functional route.
1787:   Optional vLLM fast prefill is off by default, not a universal baseline feature.
1788: - [ ] Add functional Gemma E4B and12B routes;12B uses Unified geometry/MM mapping,
1789:   no PLE/shared tail. Extend source/profile contracts before learned qualification.
1790: - [ ] Gemma26B-A4B: learned router/grouped experts/dense+expert composition, then
1791:   admitted expert sharing/hotsets/EP. Gemma31B is dense; qualify its learned dense
1792:   path separately. Reuse the existing single raw K projection for configured
1793:   K=V layers, preserving distinct K/V transforms and state. Select the exact
1794:   optimized-router tie/reduction/scale profile; full-softmax fallback is not an
1795:   identical floating-point schedule. Geometry/analytic contracts alone do not
1796:   implement either learned route.
1797: - [ ] Both families: shared continuous serving/temporal work, paged KV/COW,
1798:   full-domain compatible prefix intersection and graph/capture ownership after
1799:   actual source qualification. Shared target verifier/all-prefix rollback before
1800:   Qwen trained MTP or Gemma assistant/KV sharing/sparse draft vocabulary.
1801: - [ ] Later explicit codec/distributed/MM/LoRA profiles: do not substitute lossy
1802:   FP8/FP4/KV codecs for original BF16 or assume source/hardware eligibility.
1803:   CUDA-only fusion and CDNA/gfx12 AITER are not unconditional gfx1151 gaps;
1804:   current ROCm Triton speculative GDN needs actual eligibility checks. Generic
1805:   MXFP4 registry falls back for dense Linear; DBO rejects relevant hybrid/MM routes.
1893: - [ ] Continue Qwen27B/Gemma source/runtime integration independently; merge
1894:   contained completed units promptly. Use pvl-x2 Radeon8060S/gfx1151 ROCm/HIP
1895:   for their GPU qualification/benchmarks; preserve exact HF source identities,
1896:   source-derived complete allocations and full logits/state/cancellation gates.
1897:   Existing remote Ollama GGUF inputs are not approved-HF equivalents. Current
1898:   Vulkan/CUDA parity work is a separate implementation/qualification obligation.
1899: - [ ] Gemma follow-on: qualify tokenizer/ordinary learned route and broader
1900:   prompt/cached geometry plus learned cancellation/state lifecycle through the
1901:   same engine; selected diagnostic exactness alone grants no public route.
1902: - [ ] Independent follow-on units continue while short-HF validation runs:
1903:   Qwen authentic H48/128 GDN producer and full64/112-domain schedule, plus Gemma
1904:   family source-bound CPU completion. Preserve actual source geometry and exact
1905:   numerical limits; unavailable learned arithmetic must not use legacy fallback.
1906: - [ ] Gemma trained Rust execution requires exact learned BF16 arithmetic and
1907:   independent source/state comparison. Payload/tokenizer/original CPU oracle
1908:   now exist; native/serving/performance acceptance remains separate.
1917: - [ ] Dense27B: bind full trained source/state through bounded row/layer windows;
1918:   amortize immutable reads across compatible prompt/request rows, account
1919:   activation spill and genuine upload readiness. MTP metadata/tensors alone
1920:   do not qualify target or drafter runtime; expert hotsets do not sparsify it.
1921: - [ ] Gemma4 variants: source-real PLE, normalization/logit caps, SWA/global
1922:   attention, configured K=V/cross-layer sharing, head256/512 and26B-A4B routing.
1923:   Check graph padding cannot route experts or advance roots. Matching trained
1924:   assistants may share target KV/embedding only under actual architecture maps;
1925:   first qualify target execution and complete branch/precision ownership.
1926: - [ ] Later sparse Qwen: stable finite top-k/tie/block ordering, source-selected
1927:   expert grouping, hyperconnections and deterministic learned PLE lookup lane;
1928:   runtime n-gram proposals are a separate stage. Source-dependent sparse masks
1929:   and reduction/cast profiles cannot be grafted onto ordinary dense attention.
1930: - [ ] Later DeepSeek: exact CSA/HCA/MLA/indexer/compressor/mHC/YOCO/Engram and
1931:   checkpoint-matched DSpark/Markov/confidence contracts; typed dependency/liveness
1932:   proof may eliminate dead prompt work only after every future state read is
1933:   covered. Truncated bounded replay and perturbed sampling are explicit changed
1934:   profiles, not exact defaults. Existing payload acquisition hold persists.
1935: - [ ] Later GLM: distinguish KDA from GDN, MLA/DSA index/open-tail state, mHC and
1936:   own trained MTP/taps. CPU/GPU placement, cached-expert group order and FP16
1937:   prefill must have declared arithmetic; geometry or source names do not prove
1938:   interchangeability. Existing payload/device holds persist.
1940: - [ ] **Independent trained BF16/FLA checkpoint gate:** original
1941:   Qwen3.8-27B@1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0 acquired
1942:   through the existing bounded Rust cache pipeline; independent31-file/
1943:   18-shard full hashes and1199-tensor header/index checks passed.
1944:   Whole-model27B execution remains unqualified; bounded trained operator
1945:   qualifications below do not establish full logits or112 StateRoot domains.
1946:   Genuine original layer0 GDN reference now validates initial chunk64 and true
1947:   cached continuation for teachers11/353,48 value heads and original BF16
1948:   parameters. Plain/observed complete outputs, convolution/recurrent state,
1949:   raw source readsets, full before/after source hashes and original owned
1950:   retirement match. The serializer's V2 stride failure remains preserved;
1951:   V3 raw-bit transport passes12 real PyTorch layouts. This is an independent
1952:   public target. An isolated native composition now passes the authentic H48
1953:   cached slice:804960 input and798720 output words, including the complete
1954:   recurrent state, match independently before/after callers drop. The distinct
1955:   BF16-gamma gated RMS slice also passes61632 intermediate,12288 final and
1956:   24576 retained input words. Source hashes, original ownership, cancellation,
1957:   real pressure and final reservations0 pass; strict/release builds and237
1958:   focused tests pass. These operator changes remain isolated from MAIN;
1959:   initial chunk64, native exp, complete GDN,112-domain state and full inference
1960:   remain unqualified. A separate genuine public reference passes32 pure
1961:   operator cases over16 roles,881856 raw BF16 input/output words, including
1962:   complete248320-row head and17408-column down projection. The isolated
1963:   native wide-role replay now matches all881856 words exactly; real pressure,
1964:   pre-cancel, retained head/source after caller drop and final reservations0
1965:   pass. Strict/release builds and17 source/relocation fixtures pass. External
1966:   probes for other roles do not establish trained producers or full-model
1967:   parity; these changes also remain isolated from MAIN.
1968:   Six actual bounded native-provider startup/lifetime cases pass with the
1969:   provider unloaded. Physical provider allocation/math admission remains open.
1970:   Original
1971:   Qwen3.5-2B@15852e8c16360a2fea060d615a32b45270f8a8fc is now acquired
1972:   by the same bounded Rust pipeline; independent official Git/LFS full12-file
1973:   hashes,632 headers/320 text shapes,596BF16+36F32 tensors pass. Its4.548GB
1974:   shard remains above the default4GiB complete-source guard. Genuine pinned
1975:   official CPU two-profile reference capture now passes archive and owned
1976:   cleanup validation: tokens11/353, four full248320 logits,168 canonical
1977:   domains and60 actual intermediates. Official BF16 loading has617 BF16
1978:   parameters; restoring only36 original F32 tensors gives581 BF16+36 F32.
1979:   The profiles differ in47/49 raw tensors per step despite identical argmax;
1980:   neither establishes Rust2B HF numerical qualification. First capture's
1981:   cleanup-name bug is preserved; the repaired fresh run proves actual nonce
1982:   container absence and process reap. Original Qwen3.5-0.8B@2fc06364715b967f1860aea9cf38778875588b17 pinned
1983:   Transformers5.17/Torch2.13 CPU captures passed archive validation:
1984:   tokens11,353, full248320 logits/all42 domains, actual layer0 stages.
1985:   Keep official473-BF16 loader and source437-BF16+36-F32 profiles distinct;
1986:   first-token argmax321 vs198. Explicit new CPU64 diagnostic vs source-mixed
1987:   reference remains UNQUALIFIED: embeddings/direct layer0 norms and both
1988:   QKV projections exact; layer0 convolution state exact,41/42 domains still
1989:   differ. Full248320 logits differ214245/214285 words, maxabs .22119140625/
1990:   .1875. First token exposes conv/SiLU signedzero; second exposes GDN state.
1991:   Complete pinned convolution, chunk64/cached recurrent, coefficients and
1992:   gated RMS arithmetic, then original GPU/FLA/state/streaming gates before enabling
1993:   trained HF execution or27B serving. User authorizes required runtime/model
1994:   acquisition/testing; dated source-free slice restrictions below are history.
1995: - [ ] **Next trained reference gate:** qualify exact original HF BF16
1996:   rounding/FLA order and independent source-trained logits/StateRoot;
1997:   then bind the original profile to the existing per-op
1998:   MathematicalBackend/SessionAuthority, with full physical
1999:   device/SSD resource admission. Earlier source-free slices establish no
2000:   full27B logits,112-domain state, native provider admission or GPU support;
2001:   current owner authorization covers the required acquisition and testing.
2003: - [ ] **Following ME-P0 real trained integration:** fuse the
2004:   accepted analytical original source operators into the ONE
2005:   existing forward_token/MathematicalBackend and
2006:   SessionAuthority execution schedule, qualify original HF BF16
2007:   intermediate casts and FLA GDN accuracy, source-trained exact
2008:   full logits and StateRoot, tokenizer, physical mutable-state
2009:   memory accounting, cancellation and device completion. No
2010:   SSD/physical inference claim follows from the diagnostic bridge.
2012: - [ ] **ME-P0 next after diagnostics:** wire the analytically verified
2013:   Qwen original operators into the ONE existing forward-plan and
2014:   TaskDriver/SessionAuthority (no second token loop or state issuer),
2015:   with exact source BF16 intermediate cast and original FLA GDN kernels,
2016:   real trusted tokenizer and full 248320 logits/42-or-more StateRoot
2017:   domains against the pinned checkpoint. Only then begin
2018:   SSD-resident streaming and measured Qwen3.8-27B physical throughput.
2020: - [ ] **ME-P0 next numerical stage:** source-exact original Qwen
2021:   full attention, GDN/MLP, grouped Q/K→V replication, A_log/gate
2022:   transformation and BF16 intermediate numerical schedule using the
2023:   shared original TaskDriver/StateRoot. Qualify tiny independent
2024:   analytic operator/state gates before real approved trained Qwen3.8
2025:   27B original full-vocabulary reference. Measured original Qwen3.5
2026:   2B CPU/GPU end-to-end performance also remains open.
2028: - [ ] **ME-P0 next:** bind source-approved original Qwen3.8 27B
2029:   tensor/index/config and tokenizer to the existing mathematical engine
2030:   with authentic head/axis geometry, strict trained CPU full logits/
2031:   StateRoot, then source-verifiable immutable SSD extents with stable
2032:   leased memory and measured decode/prefill. First finish measured
2033:   Qwen3.5 2B CPU/approved GPU end-to-end performance and
2034:   cancellation/physical readiness. Preserve no model/SDK downloads
2035:   and held device launch policies without explicit authorization.
2037: - [ ] **ACTIVE NEXT: ME-P0 Qwen3.5 2B then Qwen3.8 27B.**
2038:   Begin with original qualified Qwen3.5 .8B/2B CPU and resident GPU
2039:   source/state oracles, inspect release source ownership and exact
2040:   model-2B profiling on the permitted CPU/GPU devices. Keep tests
2041:   bounded to existing 8K context now; 16K/32K extensions remain
2042:   separate numerical gates. Qualify a stable single-model fast path,
2043:   original tokenizer/math/KV/recurrent/StateRoot and actual device
2044:   completion first. Then implement checked immutable source shard leases
2045:   above the present 4GiB executable-source gate and Qwen3.8 27B
2046:   complete trained CPU logits, including SSD streaming and quant-aware
2047:   physical placement. Do not fetch weights or launch held Renoir/gfx90c
2048:   hardware without the corresponding permission and source integrity.
2049: - [ ] **NEXT after Qwen: ME-P1 Gemma4 E2B→26B-A4B→31B**, independently
2050:   source-bind tensors/reference semantics, PLE/MoE/dense quantization
2051:   and trained CPU→GPU correctness before measured throughput
2052:   optimization. **ME-P2 Qwen3.8 Flash-Next** follows with separately
2053:   verified learned n-gram embedding/4B MTP heads, QSA/GDN/MoE and
2054:   explicit Q2/~3.5bpw/Q4 source quant choices (NOT guessed files),
2055:   speculative n-gram as a distinct runtime optimization. Other OSS,
2056:   ASR/TTS/image/video/decision execution and multi-model router remain
2057:   on their own deferred source-real gates.
2058: - [ ] Preserve direct one-model-per-process execution and one
2059:   shared-instance-many-pipelines as **one Rust fabric**; the accepted
2060:   YAML DAG/compiler and extension schemas add no inference backend.
2061:   Latest21Oct owner request authorizes parallel recursive pipeline/profile
2062:   design after independent engine research. It must not delay ME-P0/P1 or
2063:   imply new trained context/media/router execution.
2065: - [ ] Enforce new owner priority: Qwen3.5 .8B regression, Qwen3.5
2066:   2B highest-perf trained CPU/GPU, Qwen3.8 27B with SSD
2067:   streaming/owned shard tensors, Gemma4 E2B, 26B-A4B, dense 31B,
2068:   then Qwen3.8 Flash-Next QSA/GDN/MoE/gated residual/MTP
2069:   and 51B learned n-gram embeddings. Every additional DeepSeek,
2070:   GLM/Muse/Mistral/Llama family has a typed NOT IMPLEMENTED
2071:   source contract until distinct trained CPU and GPU gates pass.
2072:   No subfamily is generalized from metadata-only source support.
2073: - [ ] Treat single-model local GGUF execution and multi-model
2074:   pipeline-qualified execution as two frontends of ONE TaskDriver/
2075:   SourceIdentity/StateRoot fabric. A reusable model instance
2076:   serves multiple routes without another model process or implicit
2077:   source promotion; named replicas/adapters require separate
2078:   physical memory admission.
2080: - [ ] After the accepted source-acquisition v1 cutover, use the existing
2081:   pinned HF index-first path to inspect explicit original Qwen3.8-27B and
2082:   Gemma4 26B-A4B index/config/tokenizer/chat/processor identities
2083:   (see [acquisition v1 handoff](../model-acquisition/README.md)).
2084:   Select an authorized exact revision and admitted local host. Do not treat
2085:   synthetic >51GB transfer-plan tests, public model pages or cached bytes
2086:   as authorization, execution-ready source, or trained CPU parity.
2087:   2026-10-18 tokenless live metadata-only HF source observation resolved
2088:   Qwen3.8-27B to `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`
2089:   (18 shards, 55,586,113,293 selected bytes) and Gemma4-26B-A4B-it
2090:   to `4d7ae4984b7db7de8f8457170b3f1a419ee76d52`
2091:   (2 shards, 51,644,338,194 selected bytes); Gemma4 E2B-it commit
2092:   `3e22461f65e89153144f8adb70e3b8c2cc9845a7`
2093:   (1 unsharded safetensors, 10,278,848,001 selected bytes).
2094:   No trained weights downloaded, locally owned, header-audited or
2095:   model-execution-qualified yet.
2097: - [ ] Acquire/bind authorized original config/index/tokenizer/checkpoint identities and choose a capable admitted host. Large BF16 sources exceed current local RAM; current executable source limit stays 4 GiB.
2098: - [ ] Qualify real large (>4GiB) original trained shard payload on an
2099:   authorized capable host using the already verified owned-shard
2100:   implementation from 93a5622; independently measure resident
2101:   allocation/physical RAM pressure and immutable lease lifetime. Next,
2102:   develop the separately admitted SSD streaming extent architecture.
2103:   The bounded source-only synthetic tests do not establish real large
2104:   learned weight loading.
2105: - [ ] Qwen3.8 27B: original HF grouped-head ordering and Q/K replication 16→48, explicit converted-profile permutations, untied head and source-derived norm/RoPE geometry.
2106: - [ ] Gemma E2B/26B: shared GELU/multiply/scale/soft-cap, proportional RoPE, value norm/sliding/shared KV, E2B PLE/doubled FFN and 26B K=V/MoE branches; choose and qualify the exact reference-framework numerical schedule.
2107: - [ ] Qualify actual trained CPU operators/full vocabulary/state/nonzero replay, official tokenization/templates/reasoning/stop behavior, sessions, rollback/checkpoints and resource refusal.
2108: - [ ] Add the same source-bound forward plans and full physical GPU qualification per eligible device, preserving all state/actual-completion/cancellation guarantees.
2109: - [ ] Keep distinct future heavy/hybrid recipes (Qwen Flash-Next, GLM, MiMo and source-defined QSA/MLA/KDA/indexers/hyperconnections), and broaden dense/encoder/pooling/reranking/LoRA support only through independent model/codec/backend gates.
2110: - [ ] Preserve CPU-first family priorities; already-authorized parallel basics do not authorize model/SDK downloads or unsupported whole-model claims.
2112: - [ ] Bind explicit original HF versus converted Qwen source profiles. Conversion permutes grouped/tiled V projections/constants/convolution/output, uses -exp(A_log) and adds1 to non-GDN norms; GDN swish is SiLU while full-attention gating remains sigmoid.
2113: - [ ] Choose and independently qualify Gemma26B reference-framework routing/top-k/reduction/scale placement before production execution. Five analytic test-local routing cases are not trained source parity.
3684: - [ ] Qualify independently selected NanoJev-like calibrated decision
3685:   model sources under the same model/source/cache/residency authority.
3686:   Register finite choice/boolean/score or include/omit heads without
3687:   assuming a specific research checkpoint. Validate calibrated
3688:   probabilities and per-question source-bound state, batching, thresholds,
3689:   accepted/rejected context records and decision-fallback semantics
3690:   before enabling generic context-filter execution; pipeline check alone
3691:   provides only typed static intent.
3697: - [ ] Real current-source qualification remains mandatory before production routing: .8B natural/current-root parity including cached 47→50, physical fault/cancellation/retirement and real-model shutdown, then .8B 512/1K/2K/4K/8K and 16K/32K. Admit 2B contexts only where the exact two-root plan fits.
3698: - [ ] Current physical blocker: the approved .8B source is now visible at `/home/pvl/.cache/abird/zerogravity/external-baselines/models/Qwen--Qwen3.5-0.8B/2fc06364715b967f1860aea9cf38778875588b17`; its 1,746,942,600-byte safetensor SHA256 is exactly `04b1c301231dd422b8860db31311ab2721511346a32cb1e079c4c4e5f1fe4696`. The transient `.agents/runs` native bundle was intentionally purged and no `cuda-kernels.json`/legacy `short-hf-native.json` bundle remains under the repo or user cache. Reacquire/regenerate the qualified native/reference bundle through the maintained capture path; do not restore deleted run directories or substitute another model source.
3714: - [ ] These are fit/admission results only. Physical correctness, continuation,
3715:   replay, faults and state-history qualification remain required for each cell
3716:   before production use; .8B real-model qualification is blocked on a qualified
3717:   native/reference bundle, not on model-source visibility.
3733: - [ ] Real physical after-launch cancellation/fault and real-model shutdown
3734:   remain required with the approved .8B source; source-free ownership success
3735:   does not substitute for those device tests.
3749: - [ ] This provides the committed-root comparison surface for the real .8B
3750:   47→50 parity harness; it does not substitute for the blocked physical run.
3772: - [ ] Execute both phases serially as soon as the approved .8B safetensor shard
3773:   becomes visible to the connector. Do not download or substitute source.
3801: - [ ] Physical execution is currently blocked by provider identity, not code:
3802:   the verifier is pinned to libcublas.so.12 bytes=105,140,976 SHA256
3803:   d63456d94f1b8e2408a8a9538c8740eb474c9eba48a84f420b15e2e10fbcd048
3804:   and libcublasLt.so.12 bytes=749,205,904 SHA256
3805:   96ddf4891ca4f4ed1475966783a9185df2abd64a81c3cc713b96d036696fc541.
3806:   Historical validated runs used
3807:   /usr/local/lib/python3.12/dist-packages/nvidia/cublas/lib/, but those held
3808:   DSOs are not visible through the current connector. The available Nix CUDA
3809:   12.9.2.10 DSOs have different sizes/hashes. Do not repin or substitute them
3810:   without an explicit source/provider qualification decision.
3928: - [ ] When the approved source tree, pinned native bundle payloads and pinned
3929:   provider DSOs reappear, use this wrapper with a fresh evidence path; no manual
3930:   phase ordering is required.
4903: - [ ] X2 remains physically dirty and the blocker is now the host KFD/process-
4904:   lifetime cleanup boundary represented by owner `2406838`, not source opening,
4905:   HTTP, or generic dotlink execution. Do not launch another27B **or ROCm2B** while
4906:   that owner/VRAM remains. Do not implicitly reset the GPU, restart dotlink, kill
4907:   an unseen host process, or force KFD reclamation. Running kernel is7.3.1 while
4908:   the current-system kernel path points at7.3.3; that is an observation only, not
4909:   evidence that a reboot/update fixes this incident. Any operator-level recovery
4910:   should be explicit before model work resumes. Accepted66-call and populated1K/2K
4911:   references remain absent; never fabricate/repin them.
~~~~

### [.agents/plans/multimodal/TODO.md](../plans/multimodal/TODO.md) — 27 unresolved markers

~~~~text
12: - [ ] Opt-in, source-granted context-filter outputs as bounded,
13:   immutable, typed hints to ASR, acoustic-first diarization,
14:   transcript-text and separately source-qualified speaker-preserving
15:   translation stages. Default only prior committed session user/assistant
16:   records; opt-in tool call/result, app system/developer prompts and
17:   explicitly authorized shared data permitted. Generic include/exclude
18:   and typed decision-model choices operate after source ownership checks.
19:   Unsupported original ASR prompt dialect refuses or uses a separately
20:   verified correction stage. Actual audio A/B WER/CER/DER/translation
21:   adequacy, timestamp/partial-output correctness, consent/authorization
22:   and canceled-turn isolation are required before any runtime claim.
24: - [ ] Model each ASR (Qwen ASR/Whisper/Canary/Parakeet),
25:   diarization (separately sourced speaker recognition), LM, Qwen TTS,
26:   Qwen Image2.1 and future video as registered typed components,
27:   not separate per-node subprocesses. The ASR→speaker-tagged
28:   transcript→LLM→TTS single coherent pipeline must include rational
29:   clocks, codec validity, VAD, per-turn epochs, barge-in,
30:   bounded fanout/backpressure and correct playback ACK frontiers.
31:   Current TOML/audio/image graph skeleton has NO real media runtime.
32: - [ ] Enable native progressive image output only at complete
33:   source-valid VAE/tile/frame boundaries; partial latents or
34:   unverified solver steps are never playable. Actual media clocks,
35:   atomic commit barriers, ROS/transport and tested streaming
36:   quality/latency require source-real gates.
38: - [ ] Add typed sample/frame/codebook/latent/decision values, rational clocks, checked conversions and complete source-bound component/codec/processor assemblies.
39: - [ ] Prove a small real non-token encoder/decision task and bounded iterative task through the existing engine before broad media claims.
40: - [ ] Qualify CPU-first original ASR and non-AR TTS (Whisper/Parakeet and Kokoro candidates); then actual backend execution and measured streaming/audio quality.
41: - [ ] Add depth-AR/RVQ speech, dual-AR codebook TTS and genuine codec-based duplex neural audio (including Qwen-Omni) where source/license/resources permit; mock codecs or text cascades do not establish native support.
42: - [ ] Qualify seeded source-exact image flow/editing conditioner+VAE pipelines, then video temporal/3D-VAE/expert switching and synchronized AV commit groups; candidate families remain separately gated.
43: - [ ] Keep ternary PTQ/Hadamard codecs, schema/calibration decision heads and specialist encoders as independent codec/model/task gates, not aliases of Qwen.
44: - [ ] Implement RealtimeSession lanes/epochs, bounded media buffers, VAD/barge-in, presentation/ack/playback frontiers and transactional rollback without speculative/partial media leakage.
45: - [ ] Add per-model transport capabilities and authorized hosted TaskService adapters with privacy/cost/idempotency/Unknown reconciliation; no duplicate local runtime or guessed realtime guarantees.
46: - [ ] Complete MM-01..24 plus model-specific ASR/TTS/image/video/decision quality/clock/resource suites; no real media-model support follows from source-free mocks.
59: - [ ] MM-D1/MM-01/MM-15: checked rational sample/frame/codec clocks, 44.1/48 kHz to 16/24 kHz and 24000/1001 or 30000/1001 FPS, long-duration remainder, discontinuity, overflow and bounded parser/buffer refusal.
60: - [ ] MM-D2/MM-02: complete source/processor/codec/license assemblies; wrong conditioner/shape/mask, missing source transform and cross-session mutable leakage refuse before use.
61: - [ ] MM-D3/MM-03..09: seeded solver/RNG and direct finite decision state, CTC/RNNT/TDT distinction, complete RVQ depth and joint AV commit; interrupted/stale device work never publishes or exposes media.
62: - [ ] MM-ASR-01..12: actual frontend/window/blank/predictor/duration/language/timestamp/diarization source fixtures; offline-only checkpoint refuses streaming, malicious audio bounded before allocation, sessions isolated.
63: - [ ] MM-TTS-01..12: non-AR phoneme/duration/vocoder and exact PCM count; incomplete 9/10 RVQ frame unplayable, codec-rate swap refused, overlap/voice generation and duplex/barge-in/state/lease boundaries correct.
64: - [ ] MM-IMG-01..12: source conditioner/solver/CFG/seed/mask/VAE/color and generation-entrypoint traces; invalid source stage/crop/processor fails, cancel retains old root and unknown physical handles, source/license/hosted status truthful.
65: - [ ] MM-VID-01..15: temporal VAE/latent crop/frame count, Wan timestep experts, source tile masks/rectangular attention, AV cross-attention/joint clocks, invalid GOP/backpressure, hot reload, cancellation and whole-assembly admission.
66: - [ ] MM-DEC-01..06: finite and joint schema head options/order/null/refusal/calibration against original source; generated text is not source-equivalent, external egress refused without authorization.
67: - [ ] MM-BON-01..08: PTQ1_0/PQ2_0 packed source scales/signs/padding plus mandatory 1024 Hadamard activation transform; wrong codec or plain-Q2 path refused, original scalar/CPU/device source parity separately qualified.
68: - [ ] MM-13..17/MM-19: real per-session/turn/transport epochs; stale ACK ignored, duplicate reconnect cursor idempotent/authenticated/bounded, playback progress remains uncertain when missing, slow clients do not release device leases or grow queues indefinitely.
69: - [ ] MM-18/MM-22/MM-23: ten speech/five video mixed jobs, fair shared physical admission and legal solver/frame yield; no per-packet catalog/router/global snapshot or silent precision/step downgrade, coupled output stays unplayable until both tracks valid.
70: - [ ] MM-SRV-01..12/MM-20/21/24: exact versioned SDK/protocol/field/task conformance, hosted accepted-timeout Unknown/idempotent reconciliation, callback/signature/TTL/MIME bounds, malformed media/unauthorized egress/cross-tenant replay refused before side effects.
71: - [ ] MM-I0 change-radius drill: small encoder/finite task and iterative mock use existing source/state/resource owners; no central router/worker rewrite or duplicate runtime; original Qwen gates stay unchanged.
72: - [ ] Eligible source-real CPU→physical full-model→playable-stream→soft-QoS gates: exact licensed checkpoints/processors/codecs and original quality/state oracles, real negative cancellation/admission/replay tests and measured cold/warm tails/memory. Mock/reference/region/hosted results never promote another tier.
73: - [ ] Resolve realtime section18 policy choices explicitly (media numeric policy, first sources, SLO/clock uncertainty, mid-turn changes, persistence, biometric/asset consent, hosted cost/egress, transport packaging and safe physical recovery). User examples do not establish released model IDs or grant execution/provider/download authority.
~~~~

### [.agents/plans/numerics-and-context/TODO.md](../plans/numerics-and-context/TODO.md) — 49 unresolved markers

~~~~text
534: - [ ] Implement genuine source-profile multi-row math and call-free lowering.
535:   Current GDN phases are `InitialOne`/`CachedOne`; attention/source records cap64.
536:   Pinned original multi-row GDN uses `torch.linalg.solve_triangular` on padded64
537:   systems, not the initial-one simplification. Preserve exact contracts and
538:   separate CPU-source BF16 residual/eager-attention semantics from CUDA math.
539:   Keep all prior accepted arithmetic/lifetime evidence; production stays disabled.
543: - [ ] Owner30Oct expands completion to `pvl-l5` CUDA `.8B`/`2B` and `pvl-x2`
544:   ROCm `.8B`/`2B`/`27B`, genuine active contexts through supported fit, then
545:   C1/C4/C8/C16/C32/C64 serving correctness and matched vLLM optimization.
546:   Use x2 resources as needed; no routine permission prompts. The current27B
547:   finite unit is a prerequisite, not completion of ordinary/context/concurrency
548:   routes. Performance follows qualified cells; record every loss/unrun scope.
553: - [ ] Recover the exact additional2B reference `TextCutlassBf16Sm80` image
554:   (3,965,472 bytes; identity in `cuda/native/cubin_manifests.rs`). Approved2B
555:   model bytes are already in the canonical cache and load successfully. No match
556:   among1,252 SM80 images extracted from retained cuBLAS/cuBLASLt/Torch/VLLM
557:   libraries, nor their accessible embedded ELF inventory. The remote project
558:   cache also has no matching-size candidate. No substitution/repin.
559: - [ ] Obtain independently trusted active-context comparison inputs. The retained
560:   Rust reference is finite cached47→50 at512 capacity; the Python reference
561:   protocol is also finite, and requires retained original/capture-source,
562:   observer/collector and owner inputs not found in the current qualification
563:   root. Capacity-only48-token runs and analytic32K attention do not close this.
564:   Do not restore deleted runs or weaken the reference to manufacture acceptance.
565: - [ ] Finish ROCm on `pvl-x2`, explicitly authorized30Oct without waiting for
566:   missing CUDA references. Reuse cached approved sources; owner permits needed
567:   2B/9B/27B downloads through canonical XDG cache. No SDK/device-privilege changes.
568:   Preserve completed CUDA lifecycle/source gates and separate wider numerical scopes.
591: - [ ] Broader CUDA active-context/2B numerical coverage remains open; the CUDA
592:   checkpoint's final gate passed. ROCm continuation is active as directed above.
608: - [ ] Resolve ordinary cuBLAS ownership before model math. Fresh phase 2 fails
609:   `cublasCreate_v2` status 1 with one denied `cuGetExportTable` call, UUID
610:   `6bd5fb6c5bf4e74a8987d93912fd9df9`; allocation attempts/charges are zero and
611:   observation buffers do not overflow. A legacy `cuGraphInstantiate` lookup
612:   also refuses. The existing model-free dynamic-row identity test reproduces
613:   the same startup failure, before any GEMM. No opaque table passthrough or
614:   preloading around admission.
615: - [ ] Resume fresh ordinary comparison using the independently retained digest;
616:   then physical lifecycle/shutdown and admitted contexts. Production routing
617:   stays disabled. CUDA correctness precedes ROCm on `pvl-x2`; no subagents or
618:   benchmark campaign during this continuation.
652: - [ ] Requalify cached47→48→49→50, affected faults/cancellation/retirement and
653:   real-model shutdown on the assembled source; then `.8B` 512/1K/2K/4K/8K and
654:   16K/32K. Admit `2B` contexts only where the exact two-root resource plan fits.
655: - [ ] Run the final isolated `scripts/check.sh`, then the single serial matched
656:   campaign in
657:   `../serving-and-speculation/BENCHMARK_CAMPAIGN_2026-10-27.md`. No 27 Oct
658:   performance cell has run.
669: - [ ] Remove remaining shared rotary fault assertions/fixed-prompt qualifier
670:   dependency and case-driven model/GDN/native dispatch. Implement genuine
671:   parameterized kernels and explicit numerical configurations; do not move or
672:   rename test whitelists into runtime descriptors. Original CUDA scalar/source
673:   facts stay Rust fixtures. Parametric source proof/design is available only.
677: - [~] Sequential continuation after serving shutdown: fresh .8B prompt47/48/49/50
678:   originals, matching native profiles, real-model shutdown/faults, genuine long
679:   contexts/backend routes, then optimized matched C16 and27B. All23 approved
680:   model members freshly SHA-verified. New parameterized reference collector
681:   preserves original math/state/fullV/source validators and exact owner; source
682:   controls pass. Fresh47/48/49/50 original pairs now pass all24/all42, fullV,
683:   consumed priors and natural feedback;47 uses explicit separate postvalidation
684:   after its original controller report-size failure. Fresh complete packed
685:   v3 captures47–50 now pass; native comparisons remain pending. Native fresh48–50 remains
686:   unsupported until actual source/ABI/physical qualification.
692: - [~] Apply [all-model correctness policy](CORRECTNESS_POLICY.md), based on
693:   pinned vLLM/llama.cpp. Sol6.1 xhigh owns every lane; this supersedes older
694:   Pi/Astra assignments and performance-first continuations below.
695: - [ ] .8B initial47/48/49/50 and cached47→50: actual independent original and
696:   native outputs/state/physical closure. Original case5 strict FAIL101 remains;
697:   original N1/B4 QKVZ itself differs by six BF16 words at identical frontier47
698:   operands. The historical fifteen-transition archive lacks paired N1 frontier50;
699:   fresh47–50 original trajectories are now independently available. This explains
700:   the diagnostic, not native model acceptance.
701: - [ ] Genuine .8B512/1024/2048/4096/8192, then16K/32K context/state/feedback;
702:   corresponding original BF16 CPU/ROCm/Vulkan full-model routes are missing.
703:   GGUF, synthetic operators and finite other-model proofs stay separate. Local
704:   Renoir/gfx90c launch/recovery hold remains. Performance and27B follow this
705:   correctness sequence; earlier finite timing facts remain historical.
707: - [ ] **Next one short-HF unit, then matched benchmarks:** both approved .8B/2B
708:   sources, canonical/ragged47/57/62/67/72 plus genuine cached feedback, original
709:   active8/caps and existing worker. Stage generic bounded storage/liveness,
710:   checked runtime logical views/epochs, complete lowerer access/tap ownership and
711:   pooled primitive scratch; current metadata planner alone grants no alias/fit.
712:   Preserve old source/precision/native gates. Benchmark the whole qualified short
713:   matrix together, then extend long contexts; unrelated27B/Gemma continue.
715: - [ ] Actualcanonical47projectiondiscovery1 completed original PLAIN+OBSERVED/source/ownedcleanup, but exactcommon state.recurrent.after call0 differs. NewOBS cumsum/KKT/H/output selections differacceptedreference1; selectionvariance is verified, specificcause unproved. Independent actual metadata identity/layout gate now passes separately; no fullmodel/packedargs/native/perf acceptance.
716: - [ ] Qualify actual canonical short47input/max8output/EOS248046 and ragged57/62/67/72 prompts; exact source/profile/numerical schedule and original all24/all42/fullV/actual sampler. Extend genuine rotary/ABI/control coverage as required, then original worker/HTTP and C1/2/4/8. Old GGUF diagnostic shapes do not grant HF parity.
717: - [ ] Distinct native2B source opening/2048wide ordinary+dualRMS/projection+6144MLP/head/970window tied table+4096B source-row actor. Actual2B prefill PTX differs in cumsum, scaled-dot KKT, recompute W/U and chunk output; matching GDN geometry and different cache keys are not an execution permit. Preserve actual reference3/source/relative5GiB boundary; derive physical fit, then same fullmodel/natural/fault/serving closure. Matched short benchmarks, long contexts and dense27B follow actual parity. End-to-end benchmarks remain unexecuted.
718: - [ ] Investigate actual2B discovery2 PLAIN outlier: both lanes finish/source/ownedcleanup pass, OBSERVED matches acceptedreference3, but PLAIN first differs at call2/layer8 recurrent state and branch/residual, then consumed downstream/fullV. Conv.after is only its last3 cache rows; no proof all upstream conv outputs match. Ordinary layer0 RMS10boundaries/565248B match independently. Fullmodel discovery remains unaccepted; source audit confirms timing-based autotuning but no specific cause; unchanged PLAINrepeat1 completed with exact reproduction. Independent RMS leaf reference accepted after complete source/IR/byte gates (565248B boundaries/sourceembedding,4096B gamma); native2048 component physical comparisons/faults/fullgate20 now pass separately, explicit parent-failure boundary retained. Unchanged PLAINrepeat1 reproduces discovery2PLAIN exactly across113713152all42+11168652common+6117576final+565248RMS B, still548/21/3different records versusOBSERVED/reference3. Python-only actual selection2 diagnostic now completes18records/lane and strict source/common/final/prior/all42/RMS gates PASS despite four differing selected stageconfigs. Different configuration alone is insufficient; the earlier reproducible outlier remains unresolved.
721: - [ ] Native continuation: genuine47 current-root/position/provider/cache coverage, cached2B histories and whole-model admission remain open. Isolated2B QKVZ/BA phase51 and actual six-node source prefix/GDN phase52 are accepted below; initial1/64 stops before GdnOut. Original full24 permit stays guarded.
722: - [ ] Current original2B KKT causal leaf1:29frozenchildren/23pure guards, actual original source/ownedcleanup and ordinaryRMS565248B pass. Four KKT configs/two repeats produce bit-identical A/F32+Ai/BF16 (all72comparisons0), but every solve uses the SAME2warps/4stages selected kernel: this does not compare solve variants. Strict original model reference still FAILS; current run reproduces prior PLAIN outlier byte-for-byte131564628B/27files. No tuner override or failure waiver; KKT config alone does not explain this case. Physical allocator peak remains unknown.
723: - [ ] Canonical47 failure narrowed in parallel: three separate PLAIN runs share the same all42 record differences; their OBSERVED lanes match acceptedreference1. Original late-H pair1 now reproduces oldbad PLAIN/good OBSERVED actual state bytes under healthy source/owned cleanup; original raw comparison remains FAIL. Inputs to H already differ: W59BF16words, U9BF16words (token34/head4); K/G/H0/history exact, Vnew=U, finalHT1002F32words differ. Later failed-pair solve-counterfactual1 directly captures equal operands/F32A and first solve.Ai differences, below. This earlier late-H capture alone did not localize the cause.
724: - [ ] Post-recompute47 investigation: attempt1 observer refused copied callable-grid assumption; original caller uses literal(1,16), failed attempt/cleanup retained. Narrowv2/28guards/peer review +48files/33parents/21runtime/3historypins PASS; actual attempt2 source/owned cleanup and original pair/common/fullV/all42 gates PASS. Seven actual K/V/beta/G/Ai/W/U records872320B/lane all equal good35; neither lane reproduces badW/U, so no historical cause/fix. Same-input original solve/recompute counterfactual proposed to test the actual observed warp2/warp4 solve selections without production tuner mutation.
725: - [ ] Repair47 exact arithmetic binding at existing provider/preparation seam; authenticate declared candidate and actual original code/ABI/launch histories, qualify finite owned solve variants, then connected full47+cached histories. Broad geometry/source equality cannot establish byte equivalence across two proven unequal solve implementations. No tuner forcing, expected-word selection, tolerance relaxation or failed-parent waiver; all18 GDN layer/eight-call identity coverage and fullmodel gates remain open.
727: - [ ] Retired2B postAttention1 parent gate remains CLOSED: bothlanes healthy/newlayer0pairedboundaries/source/native14op PASS. Fresh L8 pair3 source/lifetimes/full37paired archives verified after22observer+5host-schema guards: dual/RMS/QKVZ/BA/fullConv/preparedQKV/coefficients/cumulativeG/KKT A exact; FIRST solveAi14BF16differences, W96/U17/H2048. Actualsolve configs2warps/3stages versus4/4 and fullPTX differ; currentHT hashes reproduce prior same-lane split. Its original fullV/all42 failures remain fatal, RMSPASS. Separate new original MLP/ACT reference success and fixedOBS-only component successor do not accept this failedparent. Retired same-input originalsolve/cross-recompute leaf3 PASS: same actual794640B inputs; solve2warps/3stages reproducesPLAIN and4/4 reproducesOBS, Ai14/W96/U17 word split exact under BOTH recompute8/3 and8/4, twice; all12 returned completePTX/cachehashes equal actualoriginal variants. This isolates that split to solve configuration; it does not repair the original parent or prove earlier uncaptured inputs. Previous72 comparisons did not vary solve configuration.
728: - [ ] Next: all24/all42/fullV/feedback/cached47 histories and workerHTTP parity. New original2B MLP+activationcapture1 passes strict original common/fullV/prior/all42/RMS/postAttention/MLP reference gates and captures two6144activation packets; ROOTpostvalidation/native2B path/fullmodel remainpending. Earlier failed postAttention parents remainfailed; predeclared fixedOBSpost1 successor target unchanged. MatchedC1/2/4/8 benchmarks follow parity, then longcontexts and27B.
729: - [ ] Post-parity overhead follow-up: consider immutable GdnArithmetic same-object pointer shortcut before structural fallback; source-valid review only, no measured speedup or code change.
885: - [ ] Qualify original-row Q4_0_8x8 QKV/Z using existing source projection math
886:   against native fields6/38 before changing any equation. Native operands are
887:   CPU_REPACK Q4_0, distinct from Q8_0/Zen BLAS; actual inner Q4 dispatch is
888:   source-derived, not freshly instrumented. QKV M6144 needs separate closed
889:   geometry; do not raise tinyBLAS limits. Then compose existing convolution,
890:   SiLU and grouped Q/K normalization with retained raw6144×3 history.
891: - [ ] Nonzero cached-prior, live producer, full temporal/default-bulk route/
892:   profile/state binding and default8K qualification. Retained native bulk64
893:   versus AR1 upstream inputs match2097152 words plus1572864 four-tap operand
894:   words; that comparison is not a Rust producer proof. Current
895:   Q/K/V/prior/Z remain explicit observations; gain lacks separate capture.
896:   Sliced CPU proofs confer no complete-model/vLLM or whole-host RAM qualification.
897: - [ ] MAIN full formatting still flags only the preserved SiLU draft
898:   `resident_program/silu.rs:91`; bytes match its pre-existing checkpoint.
899:   Do not alter another owner's draft.
903: - [ ] **vLLM parity remains open:** implement and physically qualify its own
904:   CUDA arithmetic/cast/layout profile; complete initial/continued prefill and
905:   cached-decode GDN under that profile; close full .8B/2B/27B logits/state and
906:   long-context gates, then matched default/warm/concurrent serving performance.
907:   Current GGUF CPU operator proofs and historical cold-eager TTFT observations
908:   establish no completion percentage, cross-engine equivalence or new speedup.
910: - [ ] **Required three-model correctness gate (owner,20 Oct):** resolve
911:   independent original-source CPU8K GDN divergence for BOTH original
912:   Qwen3.5 models without loosening tolerances or changing declared source
913:   arithmetic. This is required alongside original Qwen3.8-27B correctness,
914:   matched serving/model benchmarks and shared-engine performance work.
915:   Short-context CPU/B1/native exactness and closed original-HF operator
916:   proofs do not close this independent GGUF long-context issue. Recover
917:   exact raw teacher prefixes and pinned native arithmetic/chunk settings,
918:   locate the first state/stage discrepancy, then rerun independent long
919:   context and the accepted short corpus after any implementation fix.
920: - [ ] Extend genuine resident GPU context/corpus qualification beyond the currently accepted 512 consumed tokens, including capacity/window/padding/replay boundaries.
921: - [ ] Run genuine whole-model 16K/32K CPU/GPU gates only after prerequisites; functional 8K success is not independent long-context numerical closure.
922: - [ ] Keep internal full-logit/all-state/trace bit parity distinct from independent native-source stage/top20 comparisons, including the known 2B 56 signed-zero distinctions.
923: - [ ] Broaden prompts/tokenizers/precision profiles and sampling quality only with independent oracles, exact history and explicit qualified ranges.
925: - [ ] Diagnose identical teacher-forced prefixes512/1024/2048/4096/8192 against pinned llama.cpp0.6.0/build11429 `d812350`, CPU/F32 KV and Flash Attention off. Fix the physical microbatch/chunk/arithmetic schedule and compare per-head GDN state, gain/beta, quantization and AR-versus-bulk recurrence; distinguish replay correctness from arithmetic.
926: - [ ] Compare native ubatch1/64/256/512 only within bounded feasible runs; the historical full ubatch1 capture did not finish. Use raw token IDs to avoid incomplete UTF-8 HTTP rows. Recheck short corpus, layers, state and actual8K after any fix; retain original functional0.45 shared-top20-logprob/0.05 top-token-probability limits and strict source-bit operator gates.
1286: - [ ] Naturalfeedback/current-root publication and serving integration through originalSession/ResidentProgram. Forced353 diagnostics remainfrontier0/unpublished; candidate IDs[198,599,8,72,18] were not fed back. Separate source-staged per-call binding/publication unit in progress. No serving/performance/fullmodel-default-route acceptance; newimplementation remainsISOonly.
1288: - [ ] Native2B original-source all-state/natural-feedback qualification and serving integration. Reference success does not establish Zerogravity inference, a driver hard cap or unknown transient highwater; actual baseline is separate from requested relative memory. Benchmarks/longcontexts/27B runtime remain OPEN.
1303: - [ ] Real current-source qualification remains mandatory before production routing: .8B natural/current-root parity including cached 47→50, physical fault/cancellation/retirement and real-model shutdown, then .8B 512/1K/2K/4K/8K and 16K/32K. Admit 2B contexts only where the exact two-root plan fits.
1304: - [ ] Current physical blocker: the approved .8B source is now visible at `/home/pvl/.cache/abird/zerogravity/external-baselines/models/Qwen--Qwen3.5-0.8B/2fc06364715b967f1860aea9cf38778875588b17`; its 1,746,942,600-byte safetensor SHA256 is exactly `04b1c301231dd422b8860db31311ab2721511346a32cb1e079c4c4e5f1fe4696`. The transient `.agents/runs` native bundle was intentionally purged and no `cuda-kernels.json`/legacy `short-hf-native.json` bundle remains under the repo or user cache. Reacquire/regenerate the qualified native/reference bundle through the maintained capture path; do not restore deleted run directories or substitute another model source.
1319: - [ ] These are fit/admission results only. Physical correctness, continuation,
1320:   replay, faults and state-history qualification remain required for each cell
1321:   before production use; .8B real-model qualification is blocked on a qualified
1322:   native/reference bundle, not on model-source visibility.
1338: - [ ] Real physical after-launch cancellation/fault and real-model shutdown
1339:   remain required with the approved .8B source; source-free ownership success
1340:   does not substitute for those device tests.
1354: - [ ] This provides the committed-root comparison surface for the real .8B
1355:   47→50 parity harness; it does not substitute for the blocked physical run.
1377: - [ ] Execute both phases serially as soon as the approved .8B safetensor shard
1378:   becomes visible to the connector. Do not download or substitute source.
1404: - [ ] Physical execution is currently blocked by provider identity, not code:
1405:   the verifier is pinned to libcublas.so.12 bytes=105,140,976 SHA256
1406:   d63456d94f1b8e2408a8a9538c8740eb474c9eba48a84f420b15e2e10fbcd048
1407:   and libcublasLt.so.12 bytes=749,205,904 SHA256
1408:   96ddf4891ca4f4ed1475966783a9185df2abd64a81c3cc713b96d036696fc541.
1409:   Historical validated runs used
1410:   /usr/local/lib/python3.12/dist-packages/nvidia/cublas/lib/, but those held
1411:   DSOs are not visible through the current connector. The available Nix CUDA
1412:   12.9.2.10 DSOs have different sizes/hashes. Do not repin or substitute them
1413:   without an explicit source/provider qualification decision.
1531: - [ ] When the approved source tree, pinned native bundle payloads and pinned
1532:   provider DSOs reappear, use this wrapper with a fresh evidence path; no manual
1533:   phase ordering is required.
~~~~

### [.agents/plans/pipeline-composition/FORMATS.md](../plans/pipeline-composition/FORMATS.md) — 6 unresolved markers

~~~~text
67: - [ ] Pin the exact released variant: original container metadata, byte order,
68:   block sizes, codes, scales/offsets, exceptional values and source hashes.
69:   Add recognition only with independently checked codec fixtures and bounded
70:   malformed/truncated/overflow refusals. Unknown variants remain Unsupported.
71: - [ ] Declare each operator's weight/input/output layout, activation coding,
72:   accumulation/reduction, rounding/cast and geometry contract before measuring.
73:   Preserve exact existing source-bit limits; a different contract is separately
74:   qualified rather than silently substituting dequantized F32 or BF16.
75: - [ ] Validate immutable source backing/identity/ranges, actual producer and
76:   TeacherSpan/SSA/state joins, ordered matrix segments and complete outputs.
77:   Descriptors and folder names mint no source owner, root or native permit.
78: - [ ] Qualify CPU, CUDA, HIP and Vulkan independently using actual original
79:   source and the selected compiler/SDK/driver/device/source/ABI identities.
80:   Exercise finite status, source refusals, cancellation, resource accounting,
81:   completion/fences, quarantine and healthy retirement. Keep gfx90c on hold.
82: - [ ] Qualify each complete model/checkpoint/profile/context against its pinned
83:   independent reference with declared model accuracy and state checks; operator
84:   fixtures do not fill missing model cells. Preserve failed/unsupported records.
85: - [ ] ROOT reviews assembled source, affected tests and final `scripts/check.sh`.
86:   Moving identical arithmetic changes source/type/binary/cache identity where
87:   applicable; do not attach old physical evidence to a renamed binary.
~~~~

### [.agents/plans/pipeline-composition/TODO.md](../plans/pipeline-composition/TODO.md) — 55 unresolved markers

~~~~text
491: - [ ] Extend qualified physical interop, other resident formats and
492:   broader shared host-visible GPU backing (device-write host read,
493:   foreign buffers/streams, aggregate UMA fit). General
494:   DAG/media/context authority, mixed backend/multi-source HF loading and
495:   pipeline transport activation remain unsupported. Continue source lifecycle/placement only through real
496:   consumers. Keep selected shape
497:   bands and typed ports, no naked
498:   scope or implicit retry. Contiguous text invocation/producer ports and the
499:   whole-text prepared boundary are implemented above; arbitrary explicit
500:   position tensors and host/IO effects remain unsupported. Extend
501:   stateful regions only with actual qualified state layouts/bridges:
502:   independent partial execution, committed-frontier conversion/recompute and
503:   arbitrary state schemas remain open. Native and captured consumers retain
504:   original source/numerical/state authority. Graph-only preparation may
505:   lower supported native semantic regions directly, without ONNX.
506:   Extend the bounded same-context CUDA managed-private/secondary-stream
507:   bridge only for real consumers: opaque external allocations, cross-device
508:   or cross-runtime transfers, other-backend completion and capture remain
509:   unsupported. No implicit copy/synchronization or independent publisher;
510:   no additional IR, allocator, scheduler or compiler framework.
511: - [ ] Keep external compiler integration and ONNX import out of the foundation
512:   dependency graph. The concrete MIGraphX/TensorRT direct-lowering adapters
513:   and optional graph import are tracked in
514:   [Phase 6](../FULL_IMPLEMENTATION_PLAN.md#6-optional-compiled-execution-providers--after-native-parity);
515:   do not activate them before the ordered native backend campaigns.
525: - [ ] Finish remaining qualifier/fault and captured testcase dispatch boundaries
526:   across model/GDN/native paths. Parametric GDN source proof/design is ready;
527:   actual runtime implementation and source/state/physical validation remain.
537: - [ ] Extend source-bound .8B qualification to fresh47–50-token prompts, genuine
538:   long contexts and original BF16 CPU/HIP/Vulkan routes. Existing cached
539:   frontiers48–50 and source fixtures do not establish those capabilities.
540: - [ ] After source-bound correctness and legitimate source access: current
541:   independent B4 physical/CPU suite, then optimized release +quiet same-attempt
542:   C16 for both small models; then trusted x2 original27B full-window capture
543:   under its remaining gates. [l5 benchmark container termination](../cpu-gpu-optimization/BENCHMARK_CONTAINER_TERMINATION.md)
544:   now has a supported cleanup-escalation inference; historical signal/OOM proof
545:   remains unavailable. No failed scope or capture gate is waived.
549: - [ ] Natural independent-reference and explicit Gemma row-hit/transfer scopes;
550:   no public serving, current source21 x2, general-fit or performance grant.
558: - [ ] Map each applicable strategy to existing primitive→model operation→family→
559:   model/profile owners: shared matrix storage/layout, compact outputs, prepared
560:   recipes, temporal work, pages/COW/prefix, graph ownership and target verifier
561:   once; Qwen GDN/state and Gemma PLE/KV-sharing/liveness select eligibility.
562:   No second registry/IR/runtime or per-token feature search. Reuse map/callers,
563:   source/math/backend/phase limits and complete costs precede profile selection.
564: - [ ] Preserve distinct numerical strategies: source113 CPU-order BF16 versus
565:   BLAS/TensorCore/split-K, packed casts and explicit lossy codecs. A common
566:   operation does not share its source permit, arithmetic grant or StateRoot.
567:   Genuine N1→B4 cross-width diagnostic first differs at QKVZ2; GdnOut7 has a
568:   separate projection difference even with exact input. Original same-prior B4
569:   comparison now passes all15transitions/30plain+observed/fullstate/fullV/24sums/
570:   natural bits, with genuine HTTP N1→mixedB4/refill separately exact. Explicit
571:   compositional proof, not an independently generated original public trajectory
572:   or universal N1/B4 bit equality. Preserve distinct strategies/source permits;
573:   no production reference imports, weakened limits or cross-width grant.
574:   Astra contract review.
575: - [ ] Use temporal/ragged pipeline liveness for Gemma E2B/E4B shared-tail fast
576:   prefill; retain requested prompt logits/taps and every future KV dependency.
577:   It is inapplicable to12B/26B/31B. Request B4 is not temporal prefill.
578: - [ ] Adopt source-bound phase selection for2B cachedB4 through the existing
579:   family/native adapter: one selector governs geometry/storage/inventory,
580:   native/GDN/RMS definitions, readiness, input admission and scheduler. Genuine
581:   N1prefill/partialfallback; unsupported2BinitialB4 refused before preparation.
582:   Derive width4 root18 from actual8current+4candidate+2Pending+2snapshot+
583:   2displaced, preserving sole retained capacity and physical readers. Exact
584:   selected-phase costs within existing4608MiBprovider+512backend/4GiBHOST;
585:   authentic2B native/RMS/catalog and same-prior numerical proof required.
586:   Guarded source13 adoption now includes the shared model foundation and2B
587:   specialization together. Authentic CPU inventory/reopen reports root18,
588:   fullhead1,017,118,720B, provider upper4,767,472,640B and HOST4,080,276,698B;
589:   remaining64,365,568B/214,690,598B respectively. These are complete source-derived
590:   bounds, not themselves GPU evidence. Optimized13 CLOSED0; original19physical
591:   scopes and genuine2B15transition trajectory/export/healthyretirement PASS.
592:   Original2B SAMEPRIOR30plain+observed/8112sections/fullstate/fullV/natural bits
593:   now exact; first newcached fault fails an unchanged compound retention check,
594:   last2faults unrun. Fixture-only14 logs full actual snapshots without relaxed
595:   checks; API/HTTP/timing/full assembled acceptance remain held.
596:   Astra architecture verdict.
609: - [ ] Implement the Astra next-native redesign
610:   through Sol HIGH: source-neutral physical owner plus source-bound services,
611:   same envelope/queue/ledger/fence, exact model sub-admissions and independent
612:   source/close/reload lifetimes. No per-model bootstrap/cap increase or source
613:   equality weakening. Source work isolated while frozen single-source Gemma
614:   actual qualification and binder316 checks continue.
615:   Physical13 now integrated MAIN/ISO345 with guarded original parents/modes/
616:   README; logical564/provider+default strict PASS, Astra architecture accepts/no
617:   concrete must-fix. Original local required release/full both-model CUDA/state/
618:   fault/API checks PASS; x2 native3/portable6/rawtransport12 and corrected shared4
619:   HIP/Vulkan healthy/unknown PASS. Fixture Oracle shadow aliases corrected only;
620:   production unchanged in345v3. Interrupted original controller and missing
621:   local13 closure preserved separately from successful fresh replay. Final gate
622:   pending after coherent contributions; measured release remains316.
623: - [ ] Extend the original HIP actor's bounded prepared-set lifecycle for multiple
624:   independently source-bound models on one physical owner. Current actor
625:   deliberately refuses a second live set/legacy/graph; shared admission alone
626:   does not grant simultaneous Gemma/Qwen native execution. Preserve exact source/
627:   arithmetic/specialization identities, original envelope/global fence and actual
628:   per-set module/flight retirement. No learned payload sharing by code identity.
629:   Astra selected seam:
630:   nonzero never-reused logical StreamId through all compile/cache/IO/driver joins,
631:   immutable source-service Client including internal enqueue paths, maxphysical
632:   streams1 retained. Cold32/64/32GiB provider/device/host remains a candidate;
633:   derive quotas from complete costs before bootstrap. The earlier two12GiB
634:   draft must not force Gemma's resident projections back to streamed storage.
635:   Current candidate Qwen12GiB/Gemma16GiB plus one4GiBcontext fits the32GiB
636:   provider envelope arithmetically; complete host/init/scoped-key peaks remain
637:   to qualify. Qwen9,917,743,073B is model-only; Gemma13,174,234,632B plus the
638:   shared context gives27,386,945,001B before new metadata. No duplicate context
639:   subtraction, live cap growth or optimistic compiler-host release. Sol HIGH
640:   implements; actual multi-set remains open.
641:   Concrete ABI review:
642:   private scoped key nests the FULL original KernelId; bounded full-Eq routing
643:   reaches actual driver lookup despite Command dropping logical StreamId.
644:   Fixed SingleCompat/Scoped mode preserves original callers without fallback
645:   mixing. Per-set source/module/admission and global unknown fence remain.
646:   Astra HIGH draft-source/API review
647:   approves the bounded direction: shared-only Qwen reload after actual final
648:   reader/native/service retirement; exact pinned-client preflight with a cold
649:   Arc/Weak identity, without cache backedges. Sol implements both corrections.
650:   Cold full-Eq routes and actual family consumers were inspected; completed
651:   fixes, full costs, checks and two-source physical lifecycle remain unqualified.
652:   ROOT reviewed the complete scoped-source delta and compiler successor: Shared
653:   retains up to three admitted8GiB compiler leases on the original persistent
654:   context (two initial sets plus one reload), including cancelled attempts; model
655:   unload gives no compiler-cache credit. Reserve before queue submission so a
656:   fourth ordinary refusal cannot fence a healthy sibling. SingleCompat keeps its
657:   existing policy. Conditional complete HOST upper33,668,886,252B includes two
658:   scoped-metadata maxima and128MiB fixture allowance, leaving690,852,116B under
659:   fixed32GiB; actual costs/fit and two-source lifecycle remain open. Grouped
660:   GPU-free successor checks run separately; no physical or performance grant.
661:   Compiler lifetime correction.
662:   Frozenv4 whole7stage CPU unit now PASS: all3strict,171resident/12core/7compiler
663:   tests and2actual-type source-expression controls. Actual compiled scoped metadata
664:   is observed per set and joined exactly, not inferred as a residual. Corrected
665:   fullHOST upper34,205,757,164B includes separate512MiB supervisor, leaving
666:   153,981,204B under32GiB; startup checks actual coldplan+fixture+controller.
667:   ROOT complete source/control/owner review; isolated x2 optimizedbuild active,
668:   genuine same-actor healthy reload/globalunknown cases still pending.
669: - [ ] Record Qwen27B's one family traversal through the existing backend/plan
670:   with explicit MKL arithmetic revision,112logical domains/128ordinary planes/
671:   fullConv4, before native/multi-model serving. Current scalar-exp/H16/short-HF
672:   binders refuse; accepted CPU INITIAL1→cached1 proof cannot grant their reuse.
674: - [ ] Qualify the integrated `TextCallGeometry` extraction and actual Gemma
675:   consumer. It separates source-bound call metadata from model geometry and
676:   finite numerical/backend eligibility; six copied-core tests pass only.
677: - [ ] Move remaining closed family builders/geometry out of core after generic
678:   semantic consumers migrate. Separate family, numerical and implementation
679:   identity; retain explicit rounding/reduction contracts and actual shape gates.
680: - [ ] Use one prepared-model serving adapter for validation/session/accepted
681:   step/committed output/cancel, retaining the original router/workers/roots.
682: - [ ] Implement sealed NativeCohort in the SAME node loop: authentic perrow
683:   record/runtime/prior/candidate/borrowedphase joins through validation, View/
684:   storage/finite certification and terminal grouped proposals, original physical
685:   cohort flight/Ready and single all-or-none publisher. Physical188/376 is an
686:   explicit request axis, not temporal tokens. Call-local body and enforcedmax8
687:   sessions preserve22root bound. Actual canonical47B4 source discovery passes;
688:   complete native arguments and genuine Rust188prefill→same-rootB4 remain open.
689: - [ ] Share immutable recipe/topology/SSA/storage/native candidates inside the
690:   existing record API; bind small source-issued invocation/root/phase data late.
691:   Current percall record charge4MiB must not multiply into permanent Bplan
692:   rebuild/hash overhead. No new template IR or firstrow provenance shortcut.
693:   Sol18 now integrated MAIN/ISO314selected source paths; Astra HIGH completed
694:   architecture-only review and accepted with no must-fix. Source/profile/shape
695:   cache40 retains original host charges; topology/source semantics Arc-shared,
696:   actual operation/provenance remains invocation-local; warm bind outsideMutex.
697:   Per-call record bound512KiB, conservative fullcache160MiB. Core201/Qwen87/
698:   GPU549/default+provider strict PASS. Original source/site/native validation and
699:   wrapper allocations remain. Both models' complete actual CUDA/state/fault/API
700:   plus streamed2B faults PASS; .8B quiet C1+70.73%/C4+28.81%/C8−2.58%, singlepass.
701:   Full assembled gate15 LAST PASS1752Rust/0failed/89ignored/269suites/allchecks;
702:   314sourcepaths/rootREADME unchanged. No zero allocation/general pipeline claim.
703:   Architecture review.
704:   Residual Sol8 binder now integrated MAIN/ISO316: concrete existing ColdModel
705:   cache40, genuine invocation validation outside lock, original retained
706:   witness/site charges512KiB each +64KiB owner. GPU551/provider+default strict
707:   pass; original joint release and both models' full real source/state/fault/API
708:   plus streamed2B faults PASS. Quiet .8B C1+78.62%/C4+26.67%/C8−1.09%, singlepass;
709:   six original owner closures healthy. Final isolated gate16 PASS1752Rust/0failed/
710:   89ignored/269suites/allchecks;316sourcepaths/rootREADME unchanged. Astra
711:   next-native pass is architecture/planning only; Sol HIGH applies the redesign.
712: - [ ] Before Gemma HIP merge: retain authentic source/snapshot lifecycle through
713:   native unknown/retirement; unsafe lowlevel boundary with one validated safe
714:   facade caller, no safe Any-owner proof. Keep fixed hardware/context/compiler/
715:   module policy in shared backend preparation. Sol44v5 implements these Astra
716:   corrections and complete source/state/lifetime qualifier in donor; reachable
717:   strict/source tests pass. Original frozen MAIN314+44 on x2 now passes full
718:   seven-case/all35/state/fullV/natural/lifetime matrix and fresh actual unknown
719:   retention. Guarded Gemma44 integrated MAIN/ISO343; successor shared physical
720:   owner/serving/performance stay unqualified. Prior316 final gate16 passed;
721:   next assembled gate follows coherent successor/affected physical checks.
722: - [ ] Demonstrate one forced ordinary versus optimized region/profile slice
723:   with joint liveness/observers and device-resident typed stage edges. Build on
724:   existing region metadata and physical lowering, without a universal framework.
725: - [ ] Later multidevice/MoE reuse: source/range/codec identity separate from
726:   physical slot/generation/placement; pin queued/captured/inflight readers,
727:   charge migration old+new/staging, preserve fairness and complete root shards.
728:   Backend tuples qualify CUDA/HIP/Vulkan separately. No current sparse,
729:   distributed or media execution claim follows from this review.
741: - [ ] Carry eligible stage outputs through typed leased device bindings with
742:   source/profile/root, owner/device/context/generation/range, layout/extents,
743:   access and actual producer dependencies. Stage boundaries alone introduce no
744:   host return, serialization, allocation, payload copy or synchronization.
745: - [ ] Prepare cross-stage resident handoff, device chains/graphs, exact fused
746:   regions and explicit bridges through existing ports/providers; force the
747:   accepted decomposition for comparison. Resolve finite compatible candidates
748:   outside the hot loop; no new interpreter, tensor runtime or execution IR.
749: - [ ] Plan joint placement/layout/liveness and admitted scratch/graph/staging
750:   across eligible producers and all consumers. Qualify caller-owned native
751:   component buffers and stream hooks; expose hidden pools/copies/waits and
752:   retain cases where a copy or smaller region improves complete-task cost.
753: - [ ] Keep qualified intermediates, model state and compact generation feedback
754:   on device. Use real event dependencies and bounded control/status transfers;
755:   preserve sampler/grammar, context grants, independent model publication,
756:   cancellation/fairness and output-credit boundaries. Require typed trained/
757:   codec bridges between models; graph composition does not erase host decisions.
758: - [ ] Explain and attribute each required materialization/copy/readback/wait and
759:   graph split. Compare forced ordinary, resident-chain, graph and fused paths
760:   under the same numerical/approximation contract, source and output consumers.
761: - [ ] Prove zero intermediate H2D/D2H payload traffic and no stage-only host wait
762:   on a qualified compatible fixture; test incompatible bridges/refusals,
763:   tails/ragged/multiple-consumer/debug/alias, cancel/backpressure/reload and
764:   unknown completion. Separately qualify trained complete-task results and
765:   actual physical peak/tails; start NVIDIA, then Strix Halo, without assuming UMA
766:   implies zero-copy. Appropriate real-source/physical and full check.sh gates.
773: - [ ] Support explicitly approximate profiles beyond derived weight precision:
774:   declared methods/ranges, changed semantics, held-out quality limits and
775:   permitted adaptive choices; separate variant correctness from original-model
776:   fidelity. Bind effective variant/state formats into cache/snapshot/graph keys.
777: - [ ] Permit interchangeable allocator/pool, provider, codec, scheduling,
778:   residency and transfer implementations under the original ownership spine.
779:   Define replacement boundaries and old-allocation retirement; test old/new
780:   coexistence, failure, graph readers, cancellation and unknown completion.
781:   Replace/rebuild unsuitable modules when real implementations justify it.
782: - [ ] Resolve one shared versioned configuration visible to participating
783:   modules, with explicit local overrides, bounded operational observations and
784:   explainable choices. Existing coordination admits joint batch/graph/draft/
785:   prefetch cost; test conflicts, stale observations and reload boundaries.
786: - [ ] Apply audited MIT/Apache static-native reuse first to every implementation
787:   and harness; otherwise directly port the smallest useful implementation and
788:   fixtures, establish faithful behavior, then integrate through our contracts.
789:   Preserve upstream notices; no Python/PyTorch production runtime is implied.
790: - [ ] Consolidate profile/phase/backend equivalence acceptance: solo/concurrent,
791:   B1/batched reorder/refill, draft off/on/every keep, fresh/restored/COW,
792:   ordinary/graph/fused and contiguous/paged. State required bitwise/error/
793:   distribution/changed-semantic relation; reject incompatible combinations.
794: - [ ] Exercise the first useful region/profile with one reused or ported
795:   component and the native correctness harness; preserve the baseline, complete
796:   state/physical gates and measured complete-request results. New abstractions
797:   must not delay current parity or require a second execution stack.
806: - [ ] Define primitive semantics separately from their kernel implementations;
807:   typed ports/shapes/source/numerical policy and declared state/read/write effects.
808: - [ ] Define a pipeline node that may reference a bounded subpipeline: one
809:   inference's operator/kernel regions and compound multimodal task stages use
810:   consistent composition laws while retaining their distinct value/frontier types.
811: - [ ] Define immutable profiles combining named pipelines and permitted source/
812:   arithmetic/workload/placement strategies. Separate defaults/preferences from
813:   non-overridable source, authorization, physical and completion requirements.
814:   A saved profile is not a live SourceIdentity, Ready or allocation permit.
815: - [ ] Compile hierarchical views into existing semantic/physical plans; legal
816:   flatten/in-line/out-line, exact region replacement, prepared shape selection,
817:   graph replay and optional fused kernels. Preserve two enduring representations,
818:   no new interpreter/runtime/model loop/allocator or per-token plan search.
819: - [ ] Specify child-region effects/exports, alias/liveness after fusion, late
820:   source/root/epoch/generation binding, cancellation, backpressure and completion.
821:   Inner mathematical regions prepare; original session writer publishes once.
822:   Compound task boundaries must not imply global rollback of already committed
823:   model/media stages or allow recursive independent publishers.
824: - [ ] Profile examples: resident interactive, concurrent throughput, long/shared
825:   prefix, memory-bounded dense27B; future sparse/derived-precision/media/distributed.
826:   Mark prerequisites, conflicts, unsupported/fallback and actual qualification.
827: - [ ] Compare minimal efficient kernel paths with ordinary decomposition and
828:   captured schedules; retain cases where fusion loses occupancy or overlap.
829:   Distinguish semantic nodes, submissions and actual device kernel invocations.
830: - [ ] After parity and matched baseline, measure source-only preparation
831:   (e.g. exact gamma widening/shift) and legal normalization-region fusion as
832:   early candidates; preserve casts, original model ownership and admitted bytes.
833:   Compare complete-request cost and actual dispatches before promotion.
834: - [ ] Review extension drills for a new primitive/model/provider/task without
835:   unrelated scheduler/API changes; define exact-source/all-state/physical and
836:   complete-request performance gates before implementing these abstractions.
1150: - [ ] OSS release/distribution naming and "latest" pointers remain
1151:   observational until an exact immutable source and codec version is bound.
1152: - [ ] Remote routed pipelines/peers, encrypted hosted gateways and
1153:   implicit zero-copy cross-device are not enabled by source-free shapes.
1167: - [ ] Connect authenticated application/conversation history providers to
1168:   ContextOwner, with durable and revocable grants, owner-authenticated
1169:   tenant+application+session identity, event/graph epoch binding to the
1170:   existing TaskCoordinator and physical memory/residency admission. Wire
1171:   source-real calibrated NanoJev/other decision heads and streaming media
1172:   A/B tests before enabling an actual context-fed ASR/translation router.
1173:   Do not infer inference support from the in-memory source fixture.
1175: - [ ] Source execution: bind actual learned model, tokenizer, processor
1176:   assembly and port formats in original ModelCatalog/TaskDriver before
1177:   enabling any pipeline run or route listening.
1179: - [ ] Single-process multi-model scheduler: immutable source residency,
1180:   exact shared instance reuse/replicas, typed cross-model buffer leases,
1181:   physical ready fencing, bounded ingress/admission/cancellation and
1182:   model-specific batch/fairness. One TaskCoordinator publishes StateRoots.
1184: - [ ] Pipeline runtime/hot reload: per-session model identity, graph generation,
1185:   buffered media/source epoch, legal stage yields, failure rollback and
1186:   atomic route transition. No stale audio, image or cross-tenant leakage.
1188: - [ ] API serving: explicit auth/bind/route config and tested OpenAI
1189:   chat, ASR, TTS, images; SSE/WS/WebRTC soft-realtime with content/size
1190:   bounds and accurate NOT IMPLEMENTED capability reports until qualified.
1192: - [ ] Context-aware source-real voice pipeline: stage-specific ASR prompt/
1193:   vocab support or separate contextual transcript reranker, acoustic-first
1194:   diarization and source-preserving contextual translation. Implement
1195:   prior-committed same-session snapshot freezing, cancellation epochs,
1196:   explicit consent, redaction, bounded history and no unauthorized
1197:   cross-user disclosures.
1198:   A/B compare WER/CER/DER/translation adequacy/latency for context
1199:   ON/OFF; never override contradicted acoustic evidence. This does not
1200:   follow from shape validation.
1202: - [ ] Actual staged ASR + diarization (speaker-tagged timestamps), ASR
1203:   -> Qwen -> TTS one coherent command; audio sample clocks/codebook
1204:   playback/VAD/barge-in and image latent/complete preview frames with
1205:   backpressure. Validate codec/latency/source on real media hardware.
1207: - [ ] Optimize model execution: Qwen3.5 2B, Qwen3.8 27B
1208:   resident+SSD, Gemma E2B/26B/31B, then Qwen Flash-Next QSA/GDN/MoE/
1209:   learned n-gram embeddings/MTP and separate speculative n-gram/draft,
1210:   with measured model-real speed/quality/physical safety gates.
1212: - [ ] Complete remaining exact Q2/Q3.x/Q4 quant source pipelines,
1213:   full native image/audio/NeMo assemblies, token/codec quality and
1214:   range/resume/cancellation before declaring "all models supported".
1215:   Keep DeepSeek V4.1 Flash and GLM5.3 Flash strictly pipeline-only now.
1217: - [ ] Trained and streaming multi-model parity, real media correctness,
1218:   device completion, SLO/fairness and zero-copy lease gates are owned by
1219:   the model-execution/serving/multimodal/memory TODOs above.
1224: - [!] HIPv4 first healthy57492 refused before child/model/GPU at old-Q import receipt schema; actual source/binary/owner closure preserved, no inference result. Controlv5 real metadata-only preflight accepted both actual receipt schemas but independent review found missing finalizer import alarm_until; distinct control-only successor required before fresh actual serial cases. No source rebuild/import/copy/download needed.
~~~~

### [.agents/plans/platform-and-release/TODO.md](../plans/platform-and-release/TODO.md) — 18 unresolved markers

~~~~text
12: - [ ] Owner23Oct: bring Vulkan execution to CUDA feature/correctness parity;
13:   use ROCm/HIP for AMD including pvl-x2 gfx1151. Vulkan source catalogue
14:   landed in `19ff059`; the real offline build PASSED on x2 25Oct (v-build,
15:   1,006.6s, three artifacts) and v-admin/v-bootstrap closures are green;
16:   V cases are blocked on the v-resident validator/producer field fix
17:   (PM handoff §2.3) — no Vulkan parity or performance claim yet. Reuse existing physical
18:   identity/arbiter/service/queue/quarantine. Separate compiled, available and
19:   qualified source/layout/math capabilities; refuse before allocation/launch
20:   when unsupported. Preserve gfx90c launch hold and source/SDK acquisition rules.
21: - [ ] Apply the owner-approved [static-native reuse/direct-port rule](../pipeline-composition/PROFILE_POLICY_AND_COMPONENT_REUSE.md#reuse-first-across-all-implementations-and-harnesses)
22:   to all implementations and correctness tooling: exact MIT/Apache file and
23:   transitive license/NOTICE review, suitable Rust/static C ABI library first,
24:   otherwise faithful smallest-component port with attribution and fixtures.
25: - [ ] Prove actual final binary linkage for selected reused components across
26:   supported targets: native toolchain and ABI/error/thread/allocator boundaries,
27:   no accidental Python/PyTorch/foreign-server runtime, declared optional device
28:   interfaces, native fixture/diagnostic support and feature-isolated packaging.
29:   Statically linked components do not alone prove the entire binary is static.
30: - [ ] Verify shared effective-configuration explanation/revision handling and
31:   replacement compatibility across allocator/provider implementations; preserve
32:   original resource/session/completion authorities and old-reader retirement.
33: - [ ] Generalize the working typed forward plan into verified effectful semantic regions/task control, closed provider substitution, bounded shape/kernel caches and explicit command/resource legality; no new persistent IR/runtime without a real need.
34: - [ ] Prove artifact→model→compiled task→coordinator→output and explain-plan for supported tasks, including source-qualified preselected fallback and no selected-GPU-error CPU substitution.
35: - [ ] Run extension drills for one stateful semantic op, device/provider, non-token TaskProgram, quant/storage/API adapter and optional ParameterVersion/TrainStep seam; future training is not thereby implemented.
36: - [ ] Finish per-file source/model/license/NOTICE/SBOM policy, malformed-input fuzzing and ownership/security review, authenticated APIs/private caches and qualified driver/device safety.
37: - [ ] Verify Nix/module/native/cross packages and installer/release behavior on actual Linux/Windows/macOS target access; packaging definitions alone do not prove platform support.
38: - [ ] Maintain minimal/default/composed CPU-free feature isolation, one lock/version authority and truthful unsupported capability behavior.
39: - [ ] Gate bounded compiler/autotune/AOT caches by source/device/driver/shape and retain negative candidates; no unbounded first-request search.
40: - [ ] Complete reproducible/signing/release policy and fair model/task/quality/performance matrices with cold/warm, p50/p95/p99, memory/energy and uncertainty. Push/tag/release remain separately authorized actions.
41: - [ ] Demonstrate compatibility/invalidation of operator/task/state/plan caches and bounded change radius before broader platform claims.
43: - [ ] Before a true zero-copy strided View is enabled, validate contiguity against stored value provenance and test public PlanValue retargeting. Current View physically materializes dense output; retain cached-element and dynamic range guards.
44: - [ ] Preserve one GPU-library service/queue/quarantine owner and cold generic-plan admission through packaging changes; verify actual final downstream links, not only compiler/type checks.
105: - [ ] Future all-backend distributed native artifacts: qualify toolchain,
106:   runtime library availability, target linkage and installer authenticity
107:   independently of the existing portable CPU releases; no silent change to
108:   those release definitions or claim of cross-target execution.
110: - [ ] Qualify all six [CPU release definitions](../../../docs/DEVELOPMENT.md#cpu-cross-release-definitions) with actual target execution, installer checksum/version/refusal cases and platform signing/authenticity policy. Build definitions and same-host checks do not establish other-target support.
111: - [ ] Test enabled NixOS service lifecycle and runtime-secret handling after serving is implemented; the currently disabled service definition and module evaluation are not a running-server claim.
~~~~

### [.agents/plans/serving-and-speculation/BATCHING_AND_PAGED_STATE.md](../plans/serving-and-speculation/BATCHING_AND_PAGED_STATE.md) — 8 unresolved markers

~~~~text
1030: - [ ] **Baseline/profile integration:** freeze the actual source/profile and
1031:   existing contiguous/B1/batched results. Preserve the accepted llama.cpp
1032:   regression lane; qualify new vLLM profiles against their own pinned source.
1033:   Reject wrong profile/source/schema/page generations before device access.
1034: - [ ] **Allocator/COW:** test full/partial/tail pages, randomized physical block
1035:   mapping, reference counts, simultaneous forks/readers, repeated reuse and
1036:   exhaustion. Charge peak live+candidate+metadata bytes and verify final
1037:   release or intentional quarantine under actual completion.
1038: - [ ] **Paged kernels:** compare independent contiguous reference, B1 and native
1039:   batches at page_size-1/page_size/page_size+1 and multi-page boundaries;
1040:   include asymmetric heads/positions. Poisoned inactive tails must be masked
1041:   or rejected according to the profile contract. Compare full affected tensors,
1042:   logits and all required state, followed by another token.
1043: - [ ] **Hybrid prefix reuse:** compare cache-disabled replay to miss/partial/full
1044:   hit and forked continuations. Include KV hit with missing recurrent checkpoint,
1045:   mismatched numerical profile, source/template/adapter and unauthorized tenant;
1046:   invalidate stale epochs and corrupt/incompatible restore inputs safely.
1047: - [ ] **Real model composition:** both approved .8B/2B sources, full248320 logits
1048:   and all42 domains, initial/continued-prefill to decode, repeated turns,
1049:   B1/B2/B4/B8, row reorder, refill and canceled neighbor. Original27B/all112
1050:   remains a separately admitted later gate, not implied by small-model success.
1051: - [ ] **Physical failures:** cancel before submit, in flight and after Ready
1052:   before commit; abandoned candidates, stale roots, pressure, delayed/unknown
1053:   completion and graph-held pages. No partial root publication, premature free,
1054:   stale page reuse or clearing quarantine without separately approved recovery.
1055: - [ ] **Worker/HTTP:** real JSON/SSE, raw token IDs, text/usage/EOS/finish, aliases,
1056:   staggered arrivals, slow/disconnected clients and refill. Check independent
1057:   roots, fairness, bounded queue/output credits and actual owned retirement.
1058: - [ ] **Phase acceptance:** relevant source/state/cancel/physical tests, strict
1059:   checks, both approved original release parity routes and the complete isolated
1060:   gate from AGENTS.md. Compilation, mocks and documentation cannot close these.
~~~~

### [.agents/plans/serving-and-speculation/TODO.md](../plans/serving-and-speculation/TODO.md) — 46 unresolved markers

~~~~text
150: - [ ] OA-4 remaining: conversation/item resources and atomic context
151:   append using the same quota; exact input-token counting without
152:   generation through original charged tokenizer; bounded background
153:   streaming/replay and resume. No duplicate model state store,
154:   scheduler or fake capability. Real SDK qualification.
315: - [ ] Full isolated check.sh proceeds through CPU Clippy but stops at
316:   cache_prepare::role_and_staging_admission_fail_without_model_source_allocation
317:   (expected Error::Limit assertion). Investigate this separate CPU cache boundary
318:   before claiming the full gate. CUDA physical provider/correctness unchanged.
453: - [ ] Next HIP priority: localize shared prefill device costs after tiling.
454:   Keep actual shared composition; target remaining private GDN/conv/attention
455:   or projection throughput where measured. No speculative selector machinery
456:   or fixture proliferation. Restore unavailable pinned vLLM closure separately
457:   for fresh paired runs; no acquisition/repin merely to benchmark.
458: - [ ] Parallel CUDA correctness unit: localize and fix the deterministic C4/B8
459:   identical-lane divergence before using wider CUDA throughput as accepted evidence.
460:   The existing population benchmark is the reproducer; do not add a fixture corpus.
461: - [ ] After each coherent fix: rerun only the smallest real-model cells needed to
462:   prove correctness and compare against CURRENT_BASELINE. Prefer measured hot-path
463:   work over additional evidence administration.
528: - [ ] Treat X2 as a **shared physical resource**, not an exclusive test card.
529:   Record VRAM/KFD state before each run, execute only when the requested workload
530:   fits the currently available capacity, and verify only ZeroGravity's incremental
531:   process/VRAM/queue delta retires afterward. Existing KFD owners/VRAM remain
532:   other-workload baseline. Never reset/reclaim/restart/kill those owners merely to
533:   create an empty GPU. Missing27B/populated1K/2K references remain a separate
534:   numerical-acceptance blocker. **20 Nov update:** a near-empty274,448,384B/0% X2
535:   baseline was available and exact-current source plus the accepted private child-net
536:   wrapper were prepared successfully, but both installed clr7.2.4 closures fail
537:   `dlopen` because their LLVM22 requires GLIBC_2.44 while the current locked shell is
538:   glibc2.42. No HIP allocation occurred. Treat X2 ROCm model execution as environment-
539:   gated until that closure is repaired; do not weaken isolation or substitute SDKs.
540: - [ ] L5 remains the source/repository authority. X2 is execution/testing only and
541:   must be synced to the exact current l5 `main` before physical work; do not push
542:   merely to transfer local commits.
546: - [ ] Finish ordinary correctness on `pvl-l5` CUDA `.8B`/`2B` through genuine
547:   active-context fit, and `pvl-x2` ROCm `.8B`/`2B`/`27B`. Current finite
548:   references/capacity checks do not fill missing routes or populated contexts.
549: - [ ] Qualify real C1/C4/C8/C16/C32/C64 scheduling, independent state, faults
550:   and shutdown; place higher concurrency on capable hardware, especially x2.
551: - [ ] Then optimize and execute every feasible matched vLLM cell under the
552:   [expanded campaign](BENCHMARK_CAMPAIGN_2026-10-27.md). Per-run wins are the
553:   goal; preserve losses, active/offered widths and resource asymmetries.
554: - [ ] Use x2 resources as needed; continue without routine permission prompts.
555:   Only consequential external dependencies/decisions justify stopping to ask.
556:   This supersedes the earlier blanket benchmark deferral, not correctness gates.
564: - [ ] Owner resumed. Owned GEMM/RMS/rotary/attention now pass exact ordinary `.8B` cached47→50: full logits/all42/natural feedback and normal shutdown. Normal/fault lifecycle and final gate pass; long-active-context/2B numerical scopes need independent reference inputs; see [numerics](../numerics-and-context/TODO.md). Original cuBLAS private interface remains correctly refused; no alternate allocator or relaxed reference.
565: - [ ] Then exact ordinary cached47→50 comparison, real after-launch cancellation/fault/retirement/shutdown, admitted context correctness and assembled gate. Reference capture alone closes none of those gates.
566: - [ ] Owner30Oct: continue through Qwen ROCm (HIP) on `pvl-x2` now; do not wait on wider CUDA reference gaps. Needed approved2B/9B/27B downloads may use canonical XDG cache. No subagents or benchmark detour; serving remains disabled.
582: - [ ] Extend retained usage/timing for exact context prompt tokens and explicit
583:   provider-side prefill/first-token timing. Prepare strict canonical47, ragged
584:   47/57/62/67/72 and 512..32768 workload/lifecycle inputs under XDG state.
585: - [ ] After ordinary `.8B`/`2B` correctness, cached/fault/shutdown, long-context
586:   fit and final gate: execute one serial C1/C4/C8/C16, then admitted C32/C64 and
587:   context campaign with two balanced ABBA/BAAB repetitions; cold and warm stay
588:   separate. No 27 Oct timing has run.
592: - [ ] Replay applicable original-model/backend shutdown and fault scopes under
593:   their existing source/device gates before a benchmark qualification claim.
788: - [ ] Profile and fix the .8B gap first, then beat the same matched vLLM C1/C4
789:   baseline without changing source/quality/caps. Broader2B/C8/ragged/repeated
790:   timing deferred by explicit owner priority; no unnecessary baseline sweep.
791:   Resident-head candidate passes all40 actual numerical calls/unchanged limits
792:   and all five fault scopes after narrow verifier recognizer repair. Quiet
793:   host phase diagnostic8requests/64forwards exact: binding3.629112s/enqueue
794:   3.021396s, nestedHead0.005158s/zeroH2D/zero checkpoints; original owned closure
795:   clean. Follow-on host unit passes fresh40+all5faults/14metadata guards/API24
796:   per-provider exact; strict GPU Clippy PASS. Fresh unprofiled canonical wave8:
797:   C1 native9.7748 vsstock38.0061 tok/s (6.54748s vs1.68394s); C4 native9.7972
798:   vsstock49.1130 (6.53247s vs1.30312s). All32 offers/256tokens exact, actual
799:   peaks1/4, batchmax1, all original owners/source/EOF close. Preliminary one
800:   pass only; still3.89x/5.01x slower, no certified win. Combined CPU+phase
801:   diagnostic finds memcmp70.66% of532 user-cycle samples/0lost; actual process
802:   6.72user+0.66system over6.77396wall. Nested bind3.074109s/enqueue2.945825s,
803:   complete/readback0.149683s; Head64launches/zeroH2D/checkpoints. CPU fractions
804:   are not GPU or wall fractions. Investigate deep GDN static-source identity
805:   comparisons at their shared metadata owner; do not weaken equality/authority.
806:   Exact static-source equality now preserves all field/content semantics while
807:   bypassing same-pointer+length text scans. Metadata10/strictGPU Clippy/joint
808:   release PASS; frozen pair4 passes fresh actual matrix40/all5faults and matched
809:   API24/24 per provider, exact output/state/owner closure. Fresh unprofiled pair4:
810:   C1 ZG40.4089 vsstock37.0951 tok/s (1.58381s vs1.72530s), native+8.93%;
811:   C4 ZG39.2740 vsstock48.2272 (1.62958s vs1.32705s), native18.57% lower.
812:   All32/32 offers/256tokens exact, original owners/source/EOF closed. One
813:   preliminary pass, no confidence/tail/fullparity claim. Actual HF batchmax1;
814:   source-safe B4 integration is under separate investigation, not yet a causal
815:   percentage attribution. Subsequent pair5 residentembedding reuse passes
816:   fresh40calls/allfivefaults/matchedAPI24offers192tokens perprovider. Quiet
817:   C1 ZG48.9151 vsstock38.8841tok/s (+25.80%); C4 ZG49.9998 vsstock50.4721
818:   (-0.94%), all32offers/256tokens exact/allfour owners/sourceclosed. Native
819:  64roots=forwards=outputs/432steps perarm/peak1|4/batchmax1. Singlepass only,
820:   no C4win/confidence claim. Periodic146path MAIN↔ROOT convergence preserves
821:   parents/freshmtimes; assembled gate5 PASS1645Rust/0failed/86ignored/265suites
822:   plus complete scripted checks. Later common preparation/cache work requires
823:   its own assembled gate and fresh physical/API checks. Models parallel.
824: - [ ] Remaining .8B C4 optimization: source-safe HF physical coalescing/batching.
825:   Current CLI rejects native_rows, HFloaded batchNone/native_batchfalse and
826:   worker excludes originalHF; existing B2/B4/B8 BatchProgram requires GGUF F32.
827:   Reuse root-cohort validation/publication, not incompatible math admission.
828:   Unchanged N1 completioncohort needs genuine private lanes or registered
829:   cohort epochs/status taps; shared RMS/value/readback reuse must await actual
830:   completion. True BF16 B4 needs its own typed arithmetic/source/state lowering.
831:   Existing genuine47/57/62/67 B1 prefills then four distinct cached frontiers
832:   provide strong mixed-row/current-root oracles without another capture; test
833:   row permutations/all24boundaries/all42state/fullV/F32/feedback/fault closure.
834:   This is source planning; no B4 implementation or causal gap percentage claim.
835: - [ ] After source/state/natural-serving parity, execute established short
836:   C1/C2/C4/C8 comparisons with unchanged source/profile/EOS/caps and original
837:   lifecycle controls. Then apply measured substantial generic improvements and
838:   rerun those profiles before long contexts and dense27B. Research alone adds
839:   no throughput/TTFT/goodput result.
840: - [ ] After the already authorized matched benchmarks complete, implement
841:   graceful cancellation plus explicit forced/immediate cancellation with bounded
842:   escalation deadlines. Add hooks for detecting stalled progress, repeated model
843:   loops and runaway CPU work; allow the supervisor to break them cleanly. Define
844:   request/session/model scope, cooperative drain versus forced termination,
845:   timeout policy and observability. Release GPU/native/source resources only
846:   after confirmed completion/termination; keep unknown work fenced/quarantined.
847:   Cover disconnect, stuck execution, timeout/escalation and healthy-survivor
848:   recovery with real physical tests. Owner directive22Oct: defer this extension
849:   until after benchmarks; it must not delay the active parity/benchmark path.
850: - [ ] Compose stable leased rows/diff metadata, compact readout, qualified
851:   production graph regions, exact fusion and true temporal prefill inside the
852:   existing worker. Price extra graph pools/padding, output backpressure and
853:   prefill/decode fairness; unknown capture/launch completion cannot eager-fallback
854:   through a fenced service.
855: - [ ] Extend full-state branch verification before any proposer: prefix0..K,
856:   target correction, all recurrent/conv/KV/indexer/draft/RNG/grammar domains,
857:   EOS/stop and canceled/stale epochs. Compare n-gram/Recall/suffix, compatible
858:   trained MTP/block drafters and K=0 using useful complete-cycle work, not assumed
859:   acceptance or source kernel counts. Keep explicit sampler/distribution profiles.
860: - [ ] Retain profile/shape/layout/source/context-grant compatibility at every
861:   batch/prefix/graph/proposal boundary; dynamic policy chooses only already
862:   qualified compatible variants. The user-facing profile combines pipelines,
863:   while original roots, physical admission and actual completion authorize work.
877: - [ ] Integrate both accepted GGUF/reference and newly qualified vLLM numerical
878:   profiles through the same session/worker/HTTP path. Bind profile/phase/schema
879:   to compiled plans and cache compatibility; reject incompatible state reuse.
880: - [ ] Implement original-plan paged KV append and block-table QK/softmax/AV,
881:   preserving each profile's masks, positions, dtype/reductions and ownership.
882:   Qualify independent B1/B2/B4/B8, page boundaries and full logits/state;
883:   allocator/hybrid prefix/COW dependencies live in the
884:   [memory plan](../memory-and-streaming/TODO.md).
885: - [~] Extend bounded refill/regroup into fair aged chunked prefill/decode
886:   scheduling; qualify actual multi-token temporal prefill separately from
887:   request batching. **15 Nov fair-interleaving sub-slice accepted:** the shared
888:   engine owns a bounded fair-round policy above existing source/phase wave
889:   formation. Every eligible decode row is selected first; young prefill is
890:   deferred without dropping its original state/source owners; after two complete
891:   deferred rounds the oldest prefill wave must make progress. Round admission
892:   requires enough logical-token credit for one maximum admitted prefill wave plus
893:   every active decode row and refuses invalid/unbounded age/budget controls before
894:   worker/session creation. Focused model-neutral coverage passes 18 wave tests and
895:   14 worker tests, including owner retention/age rotation; strict engine clippy
896:   and production `qwen35-cuda`/CLI checks pass. Physical unequal-arrival HTTP
897:   qualification passes both approved 0.8B/2B CUDA models: B is admitted only
898:   after A's first actual ID, the first shared B2 cohort is no earlier than A
899:   output index 4, both sockets remain drained, CPU IDs/text/usage are exact, zero
900:   failed/refused/cancelled/discarded work, and final device use returns 41MiB.
901:   The complete `scripts/check.sh` command set is terminal-green on this exact
902:   source under isolated `native-gpu`: formatting/architecture, full GPU feature
903:   boundary, CPU clippy/workspace tests, minimal/core isolation, cache/source
904:   reference, serving/lifecycle, acquisition, release/installer/shell/Nix and
905:   qualification self-tests all pass. The monolithic connector wrapper itself
906:   disappeared during ordinary cached builds on repeated attempts without a
907:   compiler/test error, surviving child, cgroup OOM event or GPU residue; do not
908:   relabel those wrapper attempts as passes. This fair scheduling sub-slice is
909:   accepted from the complete constituent gate results. The broader item remains
910:   partial. The reusable native B2/B4/B8 equal-chunk candidate chain executes
911:   2..32 host teacher tokens through private candidate domains, keeps original
912:   session roots unpublished until one final all-row Ready, aggregates
913:   state/logit/readback/dispatch receipts and publishes one logical root advance
914:   for the whole chunk. **15 Nov 0.8B B2 functional physical acceptance:** the
915:   retained small `zerogravity-gpu` qualifier is now `temporal_prefill`, shared
916:   across B2/B4/B8. B2 proves abandoned 4-token rollback, one-root 4-token state
917:   publication, one-root 2-token terminal greedy, CPU/B1/full-all42 exactness,
918:   exact producer and final reserved0. The actual production CLI HTTP path passes
919:   prompt10 as chunks4+4+2 with CPU ID264/text ` a`, usage10/1, exactly3 B2
920:   publications, zero failures/refusals and final41MiB.
921:   **15 Nov temporal production checkpoint:** original2B independently
922:   passes full B2/B4/B8 resident qualification with CPU/B1/all42,
923:   rollback/publication/greedy/producer exact and final reserved0; retained bytes
924:   are 1,589,873,428 / 1,833,045,908 / 2,319,390,868. Two unsupervised B4 runs
925:   disappeared before temporal work; a heartbeat-supervised run completed exactly.
926:   Signed `eaec62d` promoted B4 and signed `8e6f14d` promotes **B8**. Production
927:   HTTP passes original0.8B and2B B4 as three exact4+4+2 B4 cohorts/publications with mixed
928:   JSON/SSE CPU text/usage exact, zero failed/refused work and clean retirement;
929:   B8 passes the same exact production contract on both sources. Retained2B B2 HTTP
930:   also passes. B2+B4+B8 now all select temporal4 with equal-frontier prefill.
931:   One earlier supervised0.8B B8 attempt remains incomplete. Canonical
932:   release-profile B2 remains open after another heartbeat-supervised thin-LTO
933:   build disappeared during CLI compile/link before the test ran, without terminal
934:   child/Rust/OOM/Xid/residue evidence; do not loop it.
935:   **Paged-state phase:** `0027c28`/`54340a6`/`1eb6e0f` establish the logical
936:   table, finite resident page ownership and actual CUDA COW materialization.
937:   Signed `3ac2059` adds a typed `StateLayout::Kv` page representation plus
938:   bounded gather-to-contiguous adapter without selecting it in sessions. K/V
939:   tables have distinct compatibility identities and paired frontiers. Final-byte
940:   RTX3060 page-boundary proof uses BF16 row512/page4 and a 4-page cap:
941:   frontiers3/4/5 gather byte-exact to contiguous (dispatches2/2/4), 3->4 performs
942:   two COW copies, 4->5 shares full first pages/reuses backing, prior state is
943:   immutable, inactive tails zero, capacity refusal and final retirement pass.
944:   Lower materialization and submitted-unknown quarantine cells remain green;
945:   production `qwen35-cuda` composition and strict clippy pass. The implementation
946:   also separates pre-submit fresh-target validation from post-Ready candidate
947:   validation, a lifecycle bug found by the first physical boundary attempt.
948:   At `dc6a188`, canonical session selection was still disabled; that commit adds
949:   the missing GPU-only bridge needed for the later wiring:
950:   transient `ResidentPagedKvStaging` exposes the exact old
951:   `StateBuffers::Kv` ABI, and private page appends can scatter only new rows
952:   from contiguous candidate staging under the same original flight while COWing
953:   the partial prior tail from immutable pages. Physical prior3->candidate4 passes
954:   with gather2, prefix-copy2, scatter-COW2, scatter-append2, no host readback
955:   after input, exact parent/child bytes and full retirement. The retained boundary,
956:   materialization and unknown-completion cells plus production composition/clippy
957:   remain green.
958:   Signed `53336b3` now lands the canonical `Domain`/root substrate itself:
959:   `StateBuffers::PagedKv` is a marker, `Domain` retains the actual paged owner
960:   and exact page lease/generation cohort, validation checks the original
961:   `StateRequirement::Kv`/source/service/readiness contract, legacy/native
962:   consumers fail closed, and single-session commit uses the factored original
963:   publication sequence. Physical roots0->3->4 publish successfully with2 page
964:   owners per committed frontier, exact displaced parent bytes and final release;
965:   all prior paged cells plus production composition/clippy remain green.
966:   **Production selection remains open.** Admission analysis rejects the earlier
967:   two-staging sketch: paged committed state + full prior staging + full candidate
968:   staging becomes three state envelopes at full context. Signed `ef9e4d4` now
969:   lands the one-staging execution contract: `StateBindingMode::KvInPlace`
970:   requires exact same fresh K/V lease identities for prior/candidate, and the
971:   actual ordinary `KvAppend` emits zero preparation launches instead of
972:   zeroing/copying the already-gathered prefix. Physical base3/capacity8 pass
973:   preserves the prefix, writes the new row exactly, leaves the tail zero and
974:   retires staging; all retained paged cells and production composition/clippy
975:   remain green. Next replace ordinary root allocation with hybrid admitted
976:   vector/page/staging pools, then select the paged route.
977:   Signed `93468b1` closes the Ready/snapshot semantics first: paged snapshot
978:   reads are typed, ordered and valid-tail bounded; physical partial/full snapshots
979:   plus4->3->4 restore pass. Ready classification permits only exact initialized
980:   full same-plane/index current-root pages as inherited, so3->4 has2 Writes and
981:   4->5 has2 inherited+2 Writes. The new pure hybrid inventory plan proves
982:   vector-root + staging + page maxima fit the old roots*state_bytes envelope and
983:   refuses insufficient active-fork headroom or heterogeneous page geometry.
984:   At `93468b1`, production selection remained open with physical pool
985:   construction and explicit page policy still pending.
986:   Signed `cb57e1d` now constructs the actual vector/staging/page pool owner and
987:   source-owned installer. Physical vector4+staging1+page24 residency equals the
988:   old67584-byte root envelope exactly and retires to0. Initial hybrid roots are
989:   zeroed vectors + empty paged KV; candidate allocation intentionally refuses
990:   until a current-root-aware path exists. Stable compatibility IDs use explicit
991:   numerical tags and checked physical/slab/capacity packing.
992:   Signed `b9b04aa` now lands the one-flight current-root candidate path itself.
993:   One `HybridCandidateBundle` owns Vector candidate + complete staging + all KV
994:   append plans; exact access deduplication merges current Reads and submitted
995:   Writes; gather precedes the unchanged ordinary program, scatter follows it, and
996:   only Vector + fresh/COW pages are promoted. Paged Domains materialize only
997:   after Ready. Physical one-Vector/two-KV base3->4 passes with gather4/scatter8,
998:   five Ready outputs, exact canonical Ready order/frontier4 continuation/zero
999:   tails and final resident0. The exact-current eight-cell suite and production
1000:   strict checks pass.
1001:   Signed `1413f72` adds the explicit low-level CUDA source-state policy while
1002:   leaving public/default serving Contiguous. Paged source loading now requires
1003:   `Paged { page_tokens, max_append_tokens }`, B1, and bounded positive policy;
1004:   inventory/runtime append bounds are enforced. Invalid paged/native/bounds
1005:   policy refuses before source IO. The original-HF differential compares
1006:   Contiguous vs Paged full-vocab/all42 at page-1/page/page+1/2page+1.
1007:   The approved managed0.8B/2B cache inputs are now visible on L5; original0.8B
1008:   page4/append4 passes full-vocab/all42 exact at frontiers3,4,5,9.
1009:   Signed `398e446` adds the matching source-real lifecycle target: prelaunch
1010:   cancel, abandoned Ready, cancel-after-Ready discard, page-boundary commits and
1011:   exact all42 restore to pre-boundary/boundary roots, final shutdown.
1012:   Signed `3041457` corrects restore qualification to fresh root/epoch
1013:   publication at the saved frontier. Cold-paged0.8B lifecycle is accepted by
1014:   retained dotlink job `9feb1394...`: exit0,168.08s,
1015:   `contiguous_warmup=false`, exact cancel/abandon/restore/all42/shutdown.
1016:   Signed `e6aac53` removes the warmup/stage scaffolding and leaves the qualifier
1017:   cold-only. Other intermittent null-exit jobs are signalled-Bubblewrap
1018:   infrastructure events, not ZeroGravity failures. Correction: deployed dotlink
1019:   v0.17.3 does not have main's `SharedProcessExit::Signal`; it records
1020:   `ExitStatus::code()==None` for a signalled child, and its sandbox child is
1021:   Bubblewrap. Timeout/cancel/failure use distinct statuses and Completed disarms
1022:   containment before recording. Controlled inner/sandbox-shell SIGKILL paths
1023:   report numeric137, while failure-time cgroup OOM/pids and restart/poll-failure
1024:   evidence are0. A direct no-inner-supervisor cold run still reproduced null exit.
1025:   **Original2B page4/append4 source-real acceptance is now green.** The repaired
1026:   local native-GPU harness executes inside the existing Bubblewrap namespace.
1027:   Dotlink v0.17.4 forensics correct the earlier interpretation: it logs every
1028:   stop it initiates, and the relevant window shows only the explicit
1029:   `job_cancel`; a silent long sleep survived until cancellation. Thus earlier
1030:   SIGKILL9s are outside dotlink stop paths and do not establish a quiet-job
1031:   watchdog or heartbeat requirement. With the model foreground and a heartbeat
1032:   sidecar used only as harness plumbing,
1033:   differential `38c1f321...` exits0 in703.17s with full-vocab/all42 exact at
1034:   3/4/5/9, and cold lifecycle `744f36fd...` exits0 in438.18s with
1035:   `cold_load=true`, exact cancel/abandon/restore/all42/shutdown.
1036:   **19 Nov true in-flight cancellation gate accepted on original0.8B paged B1.**
1037:   The qualification-only launch cut pauses only after the ordinary CUDA flight has
1038:   actually enqueued work and before Ready. Known cancellation then waits for a
1039:   physical status readback and retires the existing event without initializing
1040:   submitted Writes or creating Ready: root/all42 remain exact, Vector/staging/page
1041:   backing stays held until completion, returns to the exact preflight occupancy,
1042:   replay commits successfully and final shutdown is clean. Dropping the same
1043:   submitted future instead preserves the accepted fail-safe path: root binding is
1044:   unchanged, service fenced, one event pending,396 submitted handles plus source
1045:   completion owners quarantined and no pool reuse. Two externally SIGKILLed runs
1046:   before terminal semantics remain incomplete; retained direct known-completion
1047:   and heartbeat-assisted unknown-completion runs both exit0. Heartbeat is optional
1048:   transport plumbing only. **19 Nov explicit high-level selector accepted without
1049:   changing defaults:** public `QwenCudaStatePolicy::{Contiguous,Paged}` is exposed
1050:   only through `ModelLoadRequest::QwenCudaWithStatePolicy`; that historical fixed
1051:   `QwenCuda` Contiguous default is superseded by the21 Nov generic smart-config layer.
1052:   Paged now admits native rows4/8
1053:   through the same Hybrid owner; zero/out-of-context bounds still fail before
1054:   source IO. Native B2/B4/B8 cohort state transport is implemented by reusing
1055:   `OrdinaryCandidateState` gather/body/scatter per lane under one cohort flight.
1056:   Contiguous B8 keeps its private-KV fast path; paged `KvInPlace` uses existing
1057:   per-lane KV append. Focused policy tests, strict CUDA clippy and CUDA-serving
1058:   composition check pass. The compact original0.8B Contiguous-vs-Paged B2/B4/B8
1059:   prefill/decode/all42 cell is terminal-green: exit0 in306.84s, exact state/output
1060:   and clean shutdown; two earlier external-SIGKILL attempts remain incomplete only.
1061:   Public SDK paged serving is also terminal-green:8 concurrent page-boundary
1062:   requests,2 outputs, native B8 observed for both Contiguous/Paged, exact token IDs,
1063:   clean shutdown in340.78s. **Paged HTTP is now terminal-green** under the explicit
1064:   child-private-net wrapper accepted by `4688cba`: the same8 concurrent page-boundary
1065:   workload observes native B8 for Contiguous/Paged, exact token IDs and clean
1066:   shutdown in358.07s. Strict test-target Clippy passes. Direct shared-net L5 still
1067:   refuses `--require-private-network`; no isolation was weakened. Fresh matched-vLLM
1068:   timing remains open because no local vLLM runtime is present and retained C8 data
1069:   is not this workload. Full-gate composition found and fixed one generic
1070:   `cuda-native` import boundary for shared hybrid inventory; targeted production
1071:   `cuda-native` now checks cleanly. The warmed monolithic rerun was externally
1072:   SIGKILLed later during ordinary-Qwen feature compilation with no diagnostic and
1073:   remains incomplete. Matched-vLLM rerun is blocked by the retained pinned layer
1074:   store no longer exposing the exact image ID to Podman from this connector; exact
1075:   HF source remains intact and no reconstruction was attempted. **Prefix fork/COW
1076:   substrate now passes physically:** fresh session, independent root, no replay,
1077:   paged partial-tail3->4 COW, exact full logits/all42, immutable cached snapshot,
1078:   clean shutdown in179.13s. **Bounded semantic prefix lookup/eviction is now
1079:   implemented** in the shared engine with exact grouped compatibility, exact units,
1080:   greatest-prefix/newest-tie lookup, duplicate refresh, deterministic FIFO eviction
1081:   and explicit entry/weight caps; immutable payloads are `Arc`-shared and lookup
1082:   allocates nothing. Two focused tests, strict engine Clippy and CUDA composition
1083:   pass. **Scoped worker hit/miss is now accepted.** A validated nonzero
1084:   `PrefixSharingScope` travels beside the queued request; ordinary `request_text`
1085:   remains isolated and explicit `request_text_scoped` opts in. Generic worker restore
1086:   verifies its cursor against actual executor frontier and caches only already-
1087:   committed state-only prompt roots. Qwen owns a paged-only model-local bounded cache;
1088:   contiguous state is unaffected. Physical0.8B page4/max_append64 prompt65 produces
1089:   scheduler model-token deltas65 warm /1 same-scope /65 different-scope with exact
1090:   output ID and clean shutdown in172.98s. The first run exposed and fixed retirement
1091:   ordering by clearing cache snapshots before resident shutdown. **Trusted HTTP
1092:   scoped hit/miss is now accepted:** `ServerBuilder::prefix_sharing_scope` supplies
1093:   one fixed optional scope from the embedding application; no request field/header
1094:   can set it, and unaware services reject scoped admission. Physical0.8B
1095:   page4/max_append64 prompt65 over HTTP records scheduler model-token deltas65 warm /
1096:   1 same scope7 /65 second-server scope8, exact output ID and clean shutdown in
1097:  171.97s.7/7 server tests, strict serving Clippy and CUDA-serving composition pass.
1098:   **Bounded live-snapshot eviction/cache invalidation is now accepted:** trusted
1099:   `invalidate_prefix_scope` removes only currently retained entries for one scope and
1100:   returns exact entry/weight totals; it does not cancel already-admitted work. Safe
1101:   stale-scope handling is stop/drain admission, then invalidate; server lifecycle/
1102:   scope switching alone does not purge.
1103:   Qwen cache count=`2*active`, weight=`active*context`. Physical0.8B active1 proves
1104:   FIFO prefix64 oldest miss/newer hit, purge2/128 + invalidated-scope miss, and512-token
1105:   retained-weight pressure where prefix320 hits but prefix256 was evicted; final
1106:   purge2/448, repeated purge0, clean shutdown in179.12s. Actual resident page-pool
1107:   pressure is now qualified too: original0.8B page4/max_append64 cached prefix64 plus
1108:  5952 externally held real pages fills6144/6144; the next tail returns prelaunch
1109:   Capacity with unchanged root and no vector/staging leak, then exact full-logit
1110:   recovery after release with all pages idle at shutdown in168.51s. One startup
1111:   SIGTERM attempt remains incomplete only. Minimal Qwen diagnostics expose current
1112:   prefix-cache occupancy/bounds and paged-pool allocated/idle/occupied/max/page bytes;
1113:   no cumulative hot-path counters. Host/device OOM is not claimed. Physical overlap
1114:   remains behind event/lease and independent-root gates.
1115:   **Fresh overlap review checkpoint:** do not widen worker scheduling yet.
1116:   `PhysicalQueue` is still single-active per physical card, `ResidentSubmission`
1117:   completion is tied to trusted readback/physical retirement, and Qwen CUDA is pinned
1118:   to one logical stream; more runnable work would therefore not establish physical
1119:   overlap. The lower CUDA substrate now has a same-service fresh secondary stream and
1120:   non-clone actor-owned event. Physical L5 proof records primary+secondary events
1121:   before either wait on disjoint8MiB buffers; secondary was still not Ready on first
1122:   query and both readbacks were exact (`P12_CUDA_SECONDARY_STREAM_EVENT_PASS`). This
1123:   is explicitly a prerequisite, not a throughput/TTFT/ITL or prefill/decode overlap
1124:   claim. **Resident event/lease in-flight ownership is now accepted:** one armed
1125:   `ResidentSubmission` and one exact CUDA event stay coupled until explicit resolution.
1126:   Known completion still requires original readback/validation before Ready;
1127:   cancellation retires after event completion with writes uninitialized; unresolved
1128:   drop fences/quarantines and retains the attached completion owner. Physical marker:
1129:   `P12_CUDA_RESIDENT_INFLIGHT_EVENT_PASS ... queue_width_unchanged=true`.
1130:   **Bounded multi-flight substrate is now accepted:** configurable active-ID set in
1131:   `PhysicalQueue`, production registry still width1, and shared first-Submitted/
1132:   last-Ready accounting over one prepaid model envelope. Existing event-ledger access
1133:   admission remains authoritative. Physical bounded2 dual resident flights on distinct
1134:   CUDA streams/outputs retire in reverse order with exact8MiB readbacks and reusable
1135:   service (`P12_CUDA_RESIDENT_DUAL_FLIGHT_PASS`). Real Qwen/source feedback is now in:
1136:   existing public B8 SDK,8 concurrent original0.8B,48+3, one warm then measured wave.
1137:   Exact native B8 passes, but steady contiguous is4888.231ms/4.910 output tok/s and
1138:   paged5080.588ms/4.724 tok/s. Mean first-token commits are4523ms/4567ms respectively,
1139:   so prefill dominates; decode is ~44 tok/s contiguous and ~31-32 tok/s paged. The
1140:   immediate bottleneck is not queue concurrency **or weight streaming**. Revalidation
1141:   proves active8/context512 remains fully resident:18-root total2,076,877,813B plus B8
1142:   runtime reserve still fits the4.5GiB provider envelope. The generic prefill cohort
1143:   shares projections/stateless RMS but serializes the remaining stateful operators by
1144:   lane; decode already has cached GDN batching and one-token B4/B8 attention. The
1145:   stateless QK-RMS/gate/multiply/SiLU prefill lowering is retained: exact warm48+3 B8
1146:   improved4888.231→4696.591ms contiguous (~3.9%) and5080.588→4907.715ms paged (~3.4%)
1147:   versus the retained no-extension baseline. The historical captured engine
1148:   is fixed B4/47 and qualification-only, so it is not a clean serving fallback. Next
1149:   multi-token attention is now a reusable shared primitive and is wired into equal-row
1150:   B4/B8 prefill groups; the warm48+3 wave stayed exact but TTFT was effectively flat
1151:   (4696.591→4701.433ms contiguous;4907.715→5022.809ms paged), so attention is not the
1152:   dominant remaining cost. Shared multi-token GDN prefill is now accepted: generic
1153:   sequence-aware GDN geometry/workspace plus typed private-root import/export drives one
1154:   B4/B8 cohort runtime over packed activations. Exact warm48+3 B8 moved4701.433→1007.301ms
1155:   contiguous and5022.809→1160.106ms paged in the default unoptimized test profile.
1156:   That profile is not a serving-performance baseline. The same commit/existing cell
1157:   rebuilt non-LTO/opt1/codegen64 is183.932ms/130.483 output tok/s contiguous and
1158:   242.488ms/98.974 paged, with exact tokens/native B8. This already clears the retained
1159:   historical eager-vLLM B8 72.20 tok/s short-wave baseline on contiguous. Future perf
1160:   feedback must use a matched optimized host profile. Continuous refill/regroup now has
1161:   real public-SDK feedback too: active8/native8 with4 short +4 long survivors +4 queued
1162:   rows exposed and fixed a reusable band-layout defect. Width8 cohort storage now retains
1163:   exact B4 and B8 GDN ABI descriptors while sharing one max-width physical scratch arena;
1164:   legal B4 fallback no longer binds B8 extents. Warm refill passed with `{4,8}` on every
1165:   survivor/queued row, demonstrating B8→B4→refill→B8 regrouping. Opt1 goodput for68
1166:   generated tokens:178.769 tok/s contiguous,136.488 paged. The first matched paged-gap
1167:   optimization now removes host-zero allocation/H2D from paged KV staging and fresh-page
1168:   initialization by reusing the bounded pinned device-side word-zero primitive. The same
1169:   optimized8x48+3 SDK cell remains exact/native B8 and measures184.409ms/130.145 output
1170:   tok/s contiguous versus205.724ms/116.661 paged, improving paged from the retained
1171:  242.488ms/98.974 without moving contiguous materially; first-token commits are
1172:  133.988/133.531ms. Physical zero work is now counted as dispatches, including Hybrid
1173:   scatter accounting. The next shared transport slice now collapses per-page KV
1174:   gather copies with a fixed15-page-pair/2-destination pinned ABI under the existing
1175:  32-pointer ceiling; larger prefixes chunk through the same runtime-scalar kernel. The
1176:  48-token/page4 workload therefore gathers12 key/value pages together instead of issuing
1177:  24 page-copy launches per KV domain. The same matched cell passes exact/native B8 at
1178:  176.800ms/135.746 tok/s contiguous and182.505ms/131.504 paged, leaving ~3.2%
1179:   matched wall-time overhead. The final paired gather now also zero-fills the inactive
1180:   staging tail when the bounded span permits it, eliminating the remaining two staging
1181:   zero launches for this cell while preserving the explicit fallback for oversized or
1182:   empty-prefix cases. The same matched cell passes at177.467ms/135.236 tok/s contiguous
1183:   and180.520ms/132.949 paged, only ~1.7% overhead. The production refill shape was
1184:   then rerun on unchanged `e72030c`:4 one-token +4 eight-token survivors +4 queued
1185:   eight-token requests,68 generated tokens, matched opt1/codegen64/non-LTO. It passes
1186:   at400.383ms/169.837 output tok/s contiguous and422.572ms/160.919 paged (~5.5%
1187:   wall-time overhead), materially improving the retained old paged136.488 tok/s result.
1188:   A device-scatter append-copy+tail-zero fusion was measured at421.159ms/161.459 paged,
1189:   only ~0.3% faster than accepted source, and was removed rather than retaining another
1190:   kernel path. The temporary refill probe was removed too. Fresh vLLM remains
1191:   unavailable and is now explicitly owner-deferred. The admitted CUDA context ladder has
1192:   now advanced through **4K,8K and16K on both0.8B and2B public SDK serving**:
1193:   context-2 prompt +2 output, exact replay on a new session, exact GPU producer frontiers
1194:   and clean shutdown. The retained qualifier is current-path capability evidence and
1195:   does not substitute for the missing independent all42 populated archive. The first
1196:   0.8B16K attempt was externally SIGKILLed during Rust compilation before test entry;
1197:   clean residue/GPU/kernel checks plus a cached retry pass make that an infrastructure
1198:   event, not a model result.2B16K passes too. The32K slice then invalidated the old
1199:   admission assumption rather than producing a serving pass: the retained qualifier now
1200:   accepts32768;0.8B32K was externally SIGKILLed twice soon after test entry, while2B32K
1201:   public model load succeeded and the real qualifier ran~230s before external SIGKILL.
1202:   Clean residue/GPU/kernel checks followed each termination. The old2B32K refusal came
1203:   from the pre-streaming resident-weight calculation; shared source admission has had
1204:   streamed-weight fallback since `d16f096`. Current32K storage is now closed from
1205:   the loader's exact shared preflight:0.8B Resident total=3,427,684,813B;2B Streamed
1206:   total=1,925,110,221B. The external SIGKILL also reproduces twice during compile-only
1207:   current-source `cargo test --no-run`, before test entry, with no dotlink-cgroup
1208:   OOM/CPU/pids-limit event and idle GPU. Do not force a refusal, add another context
1209:   fixture, or retry-loop32K. Two later direct trace runs on the exact already-built
1210:   optimized0.8B32K public-SDK binary produced one full pass and one external kill:
1211:   run1 completed in185.18s with exact replay, independent sessions, exact producer
1212:   frontiers and clean shutdown; run2 received `SIGKILL from outside` at~80.7s after
1213:   >60s execution. So0.8B32K serving capability is now positively established once, but
1214:   the old external-run reliability issue no longer obscures2B32K. Fresh matched optimized
1215:   contiguous reruns stay alive normally and reproducibly hit the retained SDK's600s
1216:   first-response timeout at~616–617s total, with no SIGKILL. Simple source-placement
1217:   experiments (partial residency, state-path-first pack ordering, larger serial page
1218:   through16MiB) did not materially improve that boundary. The actual fix is explicit
1219:   paged-state admission with physically reachable COW-page pricing: it preserves the same
1220:   four logical lifecycle roots while keeping all2B32K weights resident. Same32K capacity,
1221:   1024-token public A/B is283.22s contiguous-streamed vs28.56s paged-resident (~9.9x),
1222:   and the retained full32766-token paged run passes in224.80s execution/260.34s wall with
1223:   exact replay/frontiers and clean shutdown. Canonical `QwenCuda` now runs Enabled smart
1224:   config and jointly resolves state representation + native cohort width from exact current
1225:   admission: resident weights first, then widest qualified B8/B4 cohort, then Contiguous
1226:   as a tie-break. User values still win per knob under Enabled. The existing active8 SDK
1227:   cell with native width unspecified Auto-selects8 and physically publishes `{8}` for both
1228:   Contiguous and user-Paged halves at135.979/133.004 tok/s. No benchmark result
1229:   from the earlier killed
1230:   short2B B8 attempt is retained; do not return to residual paged micro-tuning.
1248: - [ ] Run profile-matched batching/paging/prefix-hit/overlap ablations and genuine
1249:   pinned-vLLM comparisons: established short C1/2/4/8 first, then qualified long
1250:   contexts. Record quality/full-state acceptance, true ITL/TTFT tails, committed
1251:   SLO-goodput, offered/rejected/canceled work, fairness, actual dispatch/copy
1252:   counts and physical memory/fragmentation. Preserve old-profile regressions;
1253:   code presence or fewer launches cannot close serving/performance parity.
1275: - [ ] **P12 ordered SiLU integration:** private CUDA B1/B2/B4/B8 operator
1276:   passes204 physical old-GPU/CPU differential cases,171562710 exact bit
1277:   comparisons and five fresh failure/abandonment processes. All17 intermediate
1278:   checkpoints, status, canaries and original lease/quarantine ownership pass.
1279:   Independent source review and candidate CUDA/Vulkan/default strict checks
1280:   plus two row-partition tests pass. Candidate library, qualifier, server,
1281:   worker and HTTP release artifacts compile; the exact-source diagnostic
1282:   file-transport unit passes. Paused at the owner's request before starting
1283:   the prepared60-case physical matrix. Full-model integration remains unaccepted:
1284:   complete PTX/loaded-code witnesses, model/StateRoot/serving regressions, both
1285:   original release parity routes, isolated full gate and fresh timing/count
1286:   comparisons remain pending. Existing truncated PTX dump filenames overwrite
1287:   specializations; retain that failed evidence and qualify a collision-safe
1288:   diagnostic before claiming complete instruction coverage.
1290: - [ ] **Stock2B loader and matched benchmark:** genuine pinned stock2B
1291:   V4 plain/observed loader and independent comparison now pass: all335
1292:   stored tensors inventoried, exact15 MTP exclusions and320 backbone roles;
1293:   both complete vocabularies,496640 native BF16 logit words, remain exact.
1294:   Source/runtime and owned process/group/container/EOF cleanup pass; retain
1295:   the original V3 observer refusal. Both model loaders are prerequisites,
1296:   not cross-engine/full42/default-graph or throughput qualification.
1297:   Actual0.8B eager HTTP policy passes for both providers: identical47 prompt
1298:   IDs, original Rust tokenizer, effective greedy/EOS settings, observed
1299:   cgroup12GiB/swap0/CPU4 caps and authenticated GPU process membership.
1300:   Original owned cleanup closes. Podman unresolved HPIDs establish no
1301:   ownership; every sampled GPU PID must match the numeric owned subset.
1302:   Generated IDs differ under the declared activation/cast profiles. Actual
1303:   0.8B full-output quality covers seven prompts; quality equivalence remains
1304:   unqualified. First unprofiled eager/cold-process/solo comparison completes
1305:   all eight ABBA/BAAB samples: four requests/32 raw tokens per provider.
1306:   Median TTFT13.400s native versus1.561s stock; stock is faster here.
1307:   Cold-process means first request after readiness, not storage-cache cold
1308:   or complete startup latency. Both providers are internally stable, but
1309:   cross-engine IDs/text/usage differ. One stock sample coalesces delivery;
1310:   individual-token ITL and source-correct SLO-goodput remain unqualified.
1311:   The2B eager/cold-process/solo run also completes all eight samples:
1312:   four requests/32 tokens per provider, identical raw IDs/output hashes/
1313:   usage/finish on that one prompt. Median TTFT14.540s native versus1.562s
1314:   stock; all samples have individual token delivery, native ITL p95
1315:   0.186–0.199s versus stock0.026–0.030s. Seven-prompt2B quality remains
1316:   unqualified despite five matching full texts; this prompt establishes no
1317:   numerical equivalence. Finish default-route, warm/concurrent workloads
1318:   and physical counts;
1319:   this bounded observation does not establish numerical or serving parity.
1321: - [ ] **Production CUDA graph admission:** signed `5bf440d` first made the
1322:   graph identity/pin/replay/owner substrate available to production `cuda-native`
1323:   without selecting it. Signed `db7be60` now adds explicit opaque graph-resource
1324:   admission through the **same** resident ledger/provider envelope used by managed
1325:   buffers and other foreign driver resources; no second ResourceArbiter domain was
1326:   added. The policy reserve is deliberately labeled policy, not measured CUDA
1327:   internals:64KiB control + one4KiB page per prepaid source-dispatch slot + Rust
1328:   graph/receipt/lease-access-vector metadata. It is reserved before `graph_prepare`,
1329:   can grow only before capture starts, must cover the captured dispatch count,
1330:   travels with CaptureWindow/CapturedResidentGraph/quarantine owners, and releases
1331:   only after the existing trusted retirement path. Capture and retirement queue
1332:   work declare that already-prepaid footprint as `AllocationClass::Other`; the
1333:   physical queue remains scheduling-only and never charges the global envelope
1334:   twice. Actual CUDA driver graph bytes/node enumeration remain **unqualified**.
1365: - [~] One later repeat observed a transient post-capture total resident-byte
1366:   mismatch of exactly15,339,840 bytes with pending_events=0 and an empty physical
1367:   queue. The run failed closed before replay. The condition did **not** reproduce
1368:   in four consecutive subsequent physical runs; all four returned41MiB and the
1369:   final strengthened run explicitly proved charge-retained-while-live=true and
1370:   charge-released-on-retire=true. Do not weaken the idle invariant. If it recurs,
1371:   the `81b7c71` witness reports warm/steady foreign bytes, graph resource bytes,
1372:   charge count and Arc strong counts so the actual ownership boundary can be
1373:   localized. CUDA driver-internal graph bytes/node enumeration remain unqualified.
1374:   This is correctness/lifetime qualification only, not graph performance or
1375:   production serving selection.
1436: - [ ] **P12 fused continuous GPU serving:** extend the accepted CUDA B2/B4/B8
1437:   HTTP path with paged hybrid KV/recurrent cache, physically safe
1438:   prefill/decode overlap and fairness. Unequal decode frontiers and
1439:   compatible refill are accepted within the B7 source/context scope.
1440:   Extend full-logit/StateRoot qualification to other advertised trained
1441:   profiles/backends and full GPU prefill. Measure actual p50/p95 TTFT,
1442:   true token ITL, committed SLO-goodput, fairness, memory and physical
1443:   dispatch counts against genuine pinned vLLM V1 with equivalent
1444:   source/quant/tokenizer/hardware, cold/warm conditions and caps.
1445:   Local release smoke and fewer launches do not qualify performance.
1446:   Separate native0.8B B2 CUPTI diagnostic records64328 actual kernels and
1447:   64328 successful direct launch calls, zero dropped records and exact source
1448:   counter delta; plain/observed complete outputs and owned retirement match.
1449:   A fresh accepted-baseline0.8B canonical C1 diagnostic also validates all
1450:   four native/stock plain/observed processes:236270 actual native kernels
1451:   and direct launches equal the source delta, zero dropped records/errors;
1452:   exact provider-internal outputs and original owned retirement pass.
1453:   Stock public profiler records9318 kernel activities/8810 host launches,
1454:   but its losslessness/all-process coverage remains unverified. Independent
1455:   validators and ROOT/peer metadata reviews pass. These observations qualify
1456:   the native windows only; complete matched hardware counts, profiler device
1457:   allocation admission, source-correct goodput and profiled timing remain
1458:   unqualified. Equivalent default-route/warm/concurrent vLLM timing stays open.
1460: - [ ] Connect accepted core `ContextOwner`/`ContextReader` snapshot
1461:   capability to the real router's authenticated session/turn frontier:
1462:   source-scoped application/shared grants, per-application and per-tenant
1463:   ownership, immutable complete context records, explicit ON/OFF context
1464:   selection, short-lock optimistic source epochs and cancel/revoke
1465:   invalidation. Recheck `require_fresh` at TaskCoordinator delivery and
1466:   bind source/graph generations to model/media StateRoot publication.
1467:   DecisionAdapter is a typed callback boundary, not trained NanoJev
1468:   logits or actual ASR context support. Cross-session/host source sync,
1469:   bounded CPU staging and QoS, media quality A/B and revocation/stream
1470:   cancellation remain open.
1472: - [ ] One router must snapshot each enabled, source-granted context
1473:   stream at the declared turn/graph epoch before live ASR. Default only
1474:   committed user/assistant session content; explicitly selected tool
1475:   call/result, application-owned system/developer prompts and authorized
1476:   shared/cross-user records are eligible. Strict source ownership and
1477:   requested grants, not a blanket category blacklist, govern access.
1478:   No same-turn LLM feedback loop or unauthorized sources/tenants, raw
1479:   unowned KV pointers or tool-secret leaks. The ON/OFF toggle leaves
1480:   typed graph topology stable; OFF performs no retrieval. Bound
1481:   configurable 4KiB..1MiB plus physical admission, ordered filters,
1482:   typed decision models, credit/backpressure, stale ACKs and source-real
1483:   accuracy/consent tests before live mode.
1485: - [ ] Make YAML/repeatable-flag compiled pipeline configs a first-class
1486:   source-bound router input: many model IDs, many typed component DAGs,
1487:   many explicit protocol/route bindings within ONE process; one model
1488:   can serve standalone chat and ASR→diarization→LLM→TTS. Preserve
1489:   direct one-model CLI execution with no config requirement. Current
1490:   pipeline check is pure scaffolding; implement model residency,
1491:   admission, TaskCoordinator execution, generation/hot reload and
1492:   authenticated OpenAI/SSE/WebSocket/WebRTC transport, then enable
1493:   pipeline run/serve only after source/physical tests.
1494: - [ ] Distinguish model-trained MTP weights and Qwen Flash-Next
1495:   learned n-gram embedding bank from runtime n-gram/suffix
1496:   speculative decoding (no extra source required). Verify
1497:   accept/reject and complete rollback/grammar RNG across
1498:   QSA/GDN/MoE before any speed claim.
1500: - [ ] Implement canonical TaskRequest/coordinator workers and CLI/SDK/API integration, model routing/residency lifecycle, explicit supported OpenAI/Anthropic interfaces, streaming/tools/reasoning/schema/sampling and truthful capabilities.
1501: - [ ] Enforce separate API/tenant payload, queued-prefill/residency and physical active-step admission; bounded overload, cancellation, disconnect, stale epochs and output backpressure.
1502: - [ ] Add real compatible continuous/grouped batches, aged chunked prefill, decode/media fairness and finite joint batch/draft/expert/kernel/placement plans; no per-token catalog/JSON/global-lock work.
1503: - [ ] Finish Qwen basics before optimizations; enumerate the complete pinned llama.cpp/vLLM union, public and internal surfaces. Preserve 105 provisional groups/145 document seeds and CLI/API frontiers as unreviewed inputs, not exhaustive parity.
1504: - [ ] Implement every applicable sampler/logits-processor/RNG/logprob/grammar/tool/stop/reasoning composition and its independently verified outputs.
1505: - [ ] Qualify target-authoritative MTP, draft model, n-gram/suffix, EAGLE, DFlash/DFlash2, DSpark and related registered strategies; DSpark2 exact primary source/version remains unresolved.
1506: - [ ] Test all accepted-prefix lengths, target-distribution correction, target/draft RNG/grammar ownership, full KV/recurrent/index/media rollback, physical branch admission and authentic cancellation/quarantine.
1507: - [ ] Jointly benchmark material combinations, adaptive depth/backoff and all offered/rejected/canceled work: committed SLO-goodput, TTFT/ITL tails, fairness, memory and energy versus equivalent pinned vLLM/V1/Omni.
1508: - [ ] Preserve zero untriaged frontier/disposition requirements before any full-parity claim; synthetic acceptance, fewer launches and competitor measurements do not establish local parity or speed.
~~~~

### [docs/PUBLICATION_AND_REVIEW_CHECKLIST.md](../../docs/PUBLICATION_AND_REVIEW_CHECKLIST.md) — 11 unresolved markers

~~~~text
23: - [ ] When experimental code exists, add a **separate reproducibility/results appendix** with real model/quant/driver/OS/link/thermal/workload, independent quality, p95/energy, matched baseline and raw data. *Pending by design: no engine implementation.*
38: - [ ] When real source exists, confirm architecture with working multi-model, non-token and second backend **extension drills**, profiling and resource-fault injection. *Pending by design.*
47: - [ ] **Author credit/byline, institutional affiliation and contact** need the project's owner to select a public identity. No personal or organization attribution was invented.
48: - [ ] Confirm the publication artifact's licensing and third-party permissions before public release, following the owner's [MIT project policy](../CONTRIBUTING.md) and [LICENSE](../LICENSE). Separate third-party material and formal publication clearance still require review.
49: - [ ] **Trademark, third-party material and publication legal review** remain owner responsibility; prior-source code/weights/license rights must be reassessed if implementation incorporates them.
50: - [ ] **Persistent publication date/version/DOI/canonical URL** to be assigned when publicly submitted; internal paths are local developer references, not external URLs.
51: - [ ] **External link rot audit at actual publication time** and citation-style proofreading should be repeated; references were researched as of 12 October 2026.
52: - [ ] **PDF/HTML publication artifact, accessibility/print QA and figure export** optional for release; no PDF was requested, generated or claimed in this design pass.
53: - [ ] **Independent technical peer review** and copyedit/author signoff still required for a formal release.
63: - [ ] Before implementation, inspect the live source revision, working changes and agent ownership; follow [AGENTS.md](../AGENTS.md).
64: - [ ] Record actual test commands/results, relevant model-content identities, fixtures/hardware, failures and unfinished work in the plan/run TODOs. Group completed work under its landing commit after its gates pass, following [README.md](../.agents/plans/README.md); no generated handoff or separate proof inventory is required.
~~~~

## Appendix C — complete implementation and decision ledger

All 927 reachable commits through OA-4B, oldest first; full source ID, author date/time and subject. The new docs-only handoff is deliberately excluded. Git preserves full source diffs.

~~~~text
be95424837deb7919adf34503db71d27d3ad3adb 2026-10-13T12:00:00+08:00 docs(design): product, architecture, prior-art and research plans
1e10b5cc1d5cf34832477bc83ef39117a765c6a0 2026-10-13T12:00:00+08:00 feat(core): immutable source identity, state roots and physical resource ownership
af9a5d3969c9a87951990223b6d289aba69c63fb 2026-10-14T12:00:00+08:00 feat(artifacts): sharded source artifacts and model identity
b276fd01e421f1b7e416ae773610b5823b9da36a 2026-10-14T12:00:00+08:00 test(artifacts): cache transactions
586d440b144727bdd9acea7455afb27f6c38ae25 2026-10-15T12:00:00+08:00 test(artifacts): qwen text contract
782e6fc48f0924cc055232d56338da62b68b1a79 2026-10-15T12:00:00+08:00 feat(cpu): portable CPU math and weights
1fd12da7840151ba47ab43368b4e4f435aac3d3b 2026-10-16T12:00:00+08:00 feat(model-qwen35): source-verified Qwen3.5 forward and tokenizer
901ba9ae2f2ed531d992a6d657a37dc362518bd0 2026-10-16T12:00:00+08:00 test(model-qwen35): source layout contract
e6ecf02bb7d951e01e66b430a24553c690b4ac5b 2026-10-17T12:00:00+08:00 feat(model-gemma4): Gemma contract, MoE and 26B variant
81642a9af4d88d928327a1012af1594168704956 2026-10-17T12:00:00+08:00 test(model-gemma4): e2b contract
7aa19c268026069ad916f6a69440dab602aaa94a 2026-10-18T12:00:00+08:00 test(model-gemma4): family contract
65b026f902501e85afad5fd31f3ca53968a3e836 2026-10-18T12:00:00+08:00 test(model-gemma4): variant 26b contract
b03c16451d65c33a05b3c495c1205240db4ccac0 2026-10-19T12:00:00+08:00 feat(data): vendored CubeCL license and provenance
f491e34fe55e67eed434ac577c32d44eca51430e 2026-10-19T12:00:00+08:00 feat(data): vendor CubeCL cubecl-ir 0.11.0
627c58ec80ff7b33fa1ce5086a729946c5b1cd44 2026-10-20T12:00:00+08:00 feat(data): vendor CubeCL cubecl-cpp 0.11.0
3664f975e79242d8b1a12bc4eefe886f81a33c2f 2026-10-20T12:00:00+08:00 feat(data): vendor CubeCL cubecl-cuda 0.11.0
f77ae37ec0317f39a2e1ce4466d82c645cd344da 2026-10-21T12:00:00+08:00 feat(data): vendor CubeCL cubecl-hip 0.11.0
7d768a080a84b6132595f7cadbc84b75b69106e4 2026-10-21T12:00:00+08:00 feat(gpu): source-exact resident GPU execution and physical ownership
a9723bceb8ad4aa4df639dbe660efcae6c22711b 2026-10-22T12:00:00+08:00 feat(runtime): facade, semantic program, task drivers and state publication
c3d0d0b06fb1548c8bf3db10f2965e5af52be9bc 2026-10-22T12:00:00+08:00 test(runtime): gemma contract
4add50a023d4fb5dd989989cab75ac4298d46d29 2026-10-23T12:00:00+08:00 test(runtime): layout api
b4a36b829031fba498e45bebb71a6cd8f01e7ef9 2026-10-23T12:00:00+08:00 test(runtime): multimodal mm0
a691e21c040d2dcf46a52f2cbcf0e0eb934c8369 2026-10-24T12:00:00+08:00 test(runtime): qwen real model
25ec3e27e3c208158e9763944026003005456461 2026-10-24T12:00:00+08:00 test(runtime): qwen shared model
047e3b70bbcbd7acbb5176712208cc4bcc02dd29 2026-10-25T12:00:00+08:00 feat(cli): single-binary inspect and run
52e37313bff57c2a7e831f50e9236fa983e13d02 2026-10-25T12:00:00+08:00 test(cli): CLI contract
22e0cc5abb4a067a95324206c2881de9e3327427 2026-10-26T12:00:00+08:00 feat(tooling): workspace GPU qualification crate
4d82c6704dad47875fa42bdec65f9a65de79722e 2026-10-26T12:00:00+08:00 test(tooling): independent reference oracle llama_reference_layers
a2d6a02f1577554ea8dba669f77ba7b903c1bb29 2026-10-27T12:00:00+08:00 test(tooling): independent reference oracle llama_reference_logits
a8ea8926b53d7b4a9c6d00f3d9be04906f8d64bb 2026-10-27T12:00:00+08:00 test(tooling): independent reference oracle llama_reference_qk
6a0079eb167d1f1593fe2a9372039138cd949770 2026-10-28T12:00:00+08:00 test(tooling): independent reference oracle llama_reference_recurrent
be88e30733f9459eda5f150bdbd70d9d0f8c6684 2026-10-28T12:00:00+08:00 test(tooling): independent reference oracle llama_reference_rope
77778a2c258751c64c56c4ffd3dba2ff70f7296c 2026-10-29T12:00:00+08:00 test(tooling): independent reference oracle llama_reference_softmax
6015241a5f5c2bb7ee65715a998988a026ff800c 2026-10-29T12:00:00+08:00 test(tooling): independent reference oracle llama_reference_value
e190aff1fdd392696e8086010a7bbc1f12d3df4f 2026-10-30T12:00:00+08:00 test(tooling): independent reference oracle llama_reference_value_padding
0aea8de66e3eb3b704e7d26687731a3e5b07d3bb 2026-10-30T12:00:00+08:00 test(tooling): build-release-artifacts.sh
057987203d5de45ad268ba02cfc5084f76f48229 2026-10-31T12:00:00+08:00 test(tooling): check.sh
c74450a27b1e3080e9576b0f95922f17c7cdca8b 2026-10-31T12:00:00+08:00 test(tooling): compare-attention-value.py
9f54d2435fe6af15b3eaf28980bc1039ad02b559 2026-11-01T12:00:00+08:00 test(tooling): compare-qwen-layers.py
7458ffc286f7e72c70ac18acfd3f776186aa204a 2026-11-01T12:00:00+08:00 test(tooling): cpu-qualification-gate.py
73cb6e55e1795e2c13533f60621dc29619cf5e2c 2026-11-02T12:00:00+08:00 test(tooling): native-gpu.py
80145a7f91724fb34a6f67e9972850a279982bf3 2026-11-02T12:00:00+08:00 test(tooling): test-install.sh
ca1138b8cf866c81bbfb276358d03a62576df16a 2026-11-03T12:00:00+08:00 test(tooling): test-native-gpu.py
5a0636c3f9c744c2f979ded40943a269ba5a1176 2026-11-03T12:00:00+08:00 test(tooling): test-numeric-resume.py
9a6c504ebef037e09bd187d65e2bec7e2e456eef 2026-11-05T12:00:00+08:00 chore(workspace): restore the locked full workspace manifest
57a9ad7fe78bf7c5e64d3870a6ba176ceba4cc07 2026-11-05T12:00:00+08:00 test(tooling): independent reference probe notes
1af2ac434279cb5f4b635a51884c936f19b54aa3 2026-10-16T21:39:45+08:00 fix(nix): give module check a complete base system
0452c594d99df6d3b4c972ceaae8df6ea56925e3 2026-10-16T23:45:32+08:00 docs(plan): record module eval check fix
2afa32aa74aff7f35ec1b52e4caa7079a5f68980 2026-10-16T23:45:32+08:00 docs(readme): reorder engineering and FAQ sections
a054d396e1ec12fc5e9b8e2ca60e8060954870c5 2026-10-17T08:33:07+08:00 docs(agents): document plan layout and TODO spec
cc3cab27a2713fdb64bb3f1a42c8138c060c5dce 2026-10-17T08:33:07+08:00 docs(contributing): align guide and AI disclosure
7e61cb63efd0c1429aa96044f2f7d28fda1e59aa 2026-10-17T08:41:24+08:00 docs(agents): gate pushes and destructive actions
9239719c25128028fd78a6a60f817ce6b626517f 2026-10-17T09:54:54+08:00 docs: retire historical archive bodies
6deebf22228315863f563723b5e6761ac6b51c5c 2026-10-17T10:11:15+08:00 refactor(cache): namespace model cache under abird/zerogravity
fa6affa425089fdf9a15bfa10b330e03f738fec8 2026-10-17T10:30:09+08:00 feat(artifacts): add pinned model pulls
3c97c52075563f65928c564818ce73502168778f 2026-10-17T13:44:29+08:00 feat(artifacts): resume interrupted pulls
baa9d8818061958a63eabeb61a7239a103a738f0 2026-10-17T14:12:22+08:00 feat(artifacts): load pinned cache components
20f449d20320b6f2912698c7b97cfdfd82472ba2 2026-10-17T14:37:44+08:00 feat(artifacts): recover missing pull sizes
f44264c017856a015584a094566a077638939e55 2026-10-17T15:03:41+08:00 feat(cache): add protected snapshot removal
ea700f67fca7ff7871613044a989e03b8ad6f9d7 2026-10-17T15:23:55+08:00 feat(pull)!: add capped auxiliary streaming
03b69482a5fb10b8c1cc7032e7cbdf39e2e5fead 2026-10-17T15:41:45+08:00 feat(pull): pin multi-asset release components
e44f4ee8020d5661dd8167cb07dfcb085906fcb7 2026-10-17T16:16:36+08:00 feat(cache): import local HF snapshots
fce1353e649df2eb73fe6a39437043fc8551aef5 2026-10-17T16:33:28+08:00 feat(cache): qualify cross-filesystem imports
e0073a5593e5641494c0f48e8d2dcc9e9e61a119 2026-10-17T17:05:07+08:00 feat(pull): add pinned HTTPS objects
e82d76b8f8e91f7d4b49c211e26baa081ff523e5 2026-10-17T17:14:19+08:00 feat(pull): add pinned OCI GGUF layers
f74a525366e6280a9806ecaabd7e1f4df46a6a97 2026-10-17T17:26:28+08:00 feat(pull): add version-pinned GCS objects
833d68f716783c255b24691552a081a1d2b88e76 2026-10-17T17:51:00+08:00 feat(pull): add versioned S3 and SigV4
e62222276e0dac62ed554a3fec2f2ae86edad497 2026-10-17T18:13:08+08:00 feat(pull): add pinned Azure blobs
87f6454fb7c4832262d36104a806baad9a9734a7 2026-10-17T18:24:38+08:00 feat(pull): add pinned ModelScope files
223ca794c9d45a0e246ad1cd9ba8613f2f203cd4 2026-10-17T18:33:14+08:00 feat(pull): add Xet hash and chunk primitives
a774ded154ed066c8f4b134ce7f5cba32ccdd09c 2026-10-17T18:45:39+08:00 feat(pull): add cloud content contracts
837e77f5080c0c2c07fee90b2ef9fb791068a7d3 2026-10-17T18:55:23+08:00 fix(pull): enforce offline acquisition rules
52e42fcdf0e7bead1ebefe55d8103501c7bd4f64 2026-10-17T19:15:13+08:00 feat(pull): add native Xet reconstruction
46cf5a5830e7e8d8096234341ef51fafdf63cc16 2026-10-17T19:24:22+08:00 feat(pull): add conditional cloud reads
4db5efbcce973d21bf1c8ca750471e6b2ab956ab 2026-10-17T19:34:10+08:00 feat(pull): recover pinned ModelScope sizes
f22f75972e63ea70771c59fbce065e371d5f29d4 2026-10-17T19:43:00+08:00 feat(pull): list pinned file candidates
f197c9cf4a44e06a3c1cf3c532855545bed945fa 2026-10-17T19:52:21+08:00 feat(pull): select explicit GGUF quant families
1a22597d9fda50ad71c43ae08f8660fb5b014218 2026-10-17T17:44:06Z docs: preserve complete October 17 takeover and paused sources
04363c3e83368517475b6dccd4894ff437639927 2026-10-17T17:44:24Z feat(pull): qualify bounded full-path filters and exclusion closure
8195c18e43bd2aba51d727c03ebe03df6d1fd166 2026-10-17T18:12:21Z feat(pull): reconstruct verified required native Xet indexes
f56b57400eb913240d6e2b2eeea5d60d9ab6f35e 2026-10-17T19:30:00Z feat(pull): authenticate versioned AWS FULL_OBJECT SHA256
d5aa5cb0ba10d2603ae34488e54755c5495f63b3 2026-10-17T19:48:40Z feat(pull): finish v1 original-model companion closure
fee0bb2e2d690956069a0de151770c21168bea63 2026-10-17T19:57:15Z feat(pull): close official Qwen and Gemma source metadata
4bf624e300254f845e9e047fcb344026c9252d5d 2026-10-17T21:31:11Z feat(pipeline): scaffold shared multimodel routing and opt-in chat context
a9984a47052baaf1aeda558154256d361dcc64af 2026-10-18T09:26:30+08:00 chore(gitignore): share one canonical ignore block
ac44e836b46f6c5aff44afa298bdb342ff65433e 2026-10-18T02:23:14Z feat(pipeline): compose source-scoped context filters and typed decisions
f28a62ceb47229279b70283f4e88747d1719e2ed 2026-10-18T03:23:06Z feat(context): snapshot authorized committed history for pipelines
f0746d1d7e22b94b55c93cb734a8c4ad3b2ab67a 2026-10-18T04:38:18Z feat(pipeline): finish YAML-first typed multimodal scaffolding
93a562288f0e00240da92c37c9dd7f4b846d62d3 2026-10-18T05:21:30Z feat(source): bind verified immutable safetensors shards
b796a2e17a4f20086c4db59743aea9558ae1a892 2026-10-18T06:27:12Z feat(qwen): bind original source tensors to shared CPU value operators
b11e58cb9ef7dd0d607f301d90c1870897cebd34 2026-10-18T07:23:47Z feat(qwen): qualify original hybrid layer CPU proposals
1b761119c2de2690723b9531b1d43660a66c1001 2026-10-18T15:29:41+08:00 docs(agents): unify shared conventions and plan model
631e59fd614fa6576985e8ff8bf5503d88f99359 2026-10-18T15:30:21+08:00 docs(plans): migrate historical plans to the unified model
6329cffbb3e1b93811235867fa697f5f76f3d27a 2026-10-18T08:18:40Z feat(qwen): admit source-bound original layer states atomically
e43ba4c63106bab9804ff0f581457ce4f370741e 2026-10-18T09:16:57Z feat(qwen): compose original full-model FP32 diagnostic forward
28e47f5a4728679e33659298db043da728f225c1 2026-10-18T17:17:44+08:00 docs(agents): add terse-writing guidance
7e328411a8de3c0cc2b9cb9690f2e4a0aa9a54c9 2026-10-18T17:24:36+08:00 docs(agents): strengthen ownership and commit-scope rules
aaf9ba6ec94a9d1bbd4e09b747d9338637fc9e87 2026-10-18T10:09:54Z feat(qwen): qualify source-bound BF16 dtype diagnostics
1c0bb41f2e7b2f635b255d5285040ed15b9c25b6 2026-10-18T11:26:23Z feat(qwen): bind pinned source bundles to numerical reference traces
e90ff8868c46f321ff8bd0bf4c2154a71606aab2 2026-10-18T19:38:06+08:00 docs(agents): add terse-output guidance
35b90932f9bdf93127059df426e7d7eeaee54cf5 2026-10-19T00:19:08Z feat(serve): add bounded OpenAI CPU router and verified SSD extents
9790fa2ef6c4993be0b93014ad1e63d5db5cabbb 2026-10-19T02:24:07Z feat(serve): admit resident GPU requests with fair wave scheduling
114bba566bb2d150afe5f8a0fb5701880cae713f 2026-10-19T03:59:32Z feat(gpu): qualify native multi-row F32 projection on CUDA
2869f8784d986dd80973da03afafde89da34c947 2026-10-19T05:11:09Z feat(gpu): qualify batched attention GDN KV and elementwise operators
08be9c64ea60b5dcf4391c54cc183d63666e4a70 2026-10-19T07:43:08Z feat(gpu): qualify resident B2-B8 state cohort publication
3ec431a2df62bbcae5091a048f285801f151da7e 2026-10-19T08:04:35Z feat(gpu): qualify native batched GGUF quantized projections
c85ad250334f590a87c99710e95c71a3deb28323 2026-10-19T08:41:20Z docs(agents): preserve complete Zerogravity history and P12 handoff
35ebc5b343fd82201eac768edf86ba7a81c41846 2026-10-19T18:08:46+08:00 feat(gpu): fuse native resident model batches
904a190cf39aca74e06c979dbaddb0db7f4404f1 2026-10-19T18:09:43+08:00 docs(agents): record native batch acceptance
54fd580815dbcfcf0916e4e737beaf21728818b2 2026-10-19T19:51:39+08:00 feat(gpu): keep batch greedy feedback on device
570d865696ab398f0c5c37f08c5c951595eadd75 2026-10-19T19:52:16+08:00 docs(agents): record batch feedback acceptance
2677d8b664c78b657c26e7bf46c835317b003913 2026-10-19T22:33:08+08:00 feat(serve): run native CUDA B2 waves
7fc9ad475a30a3d2120f307a7da2ff75a80e337a 2026-10-19T22:33:38+08:00 docs(serve): bound native B2 acceptance
f26232e8b03bf5ab3c13e00ed641eb0d602360d2 2026-10-20T00:28:17+08:00 fix(qwen): honor tied original output heads
6bfcdbe72f0133a6c79e2e7de169c3de74b3da8a 2026-10-20T00:30:04+08:00 feat(bench): account for offered serving work
d9fa829ceba092c7f8a3fa9c2746a11a0906eaba 2026-10-20T00:40:08+08:00 feat(serve): qualify native CUDA rows 4 and 8
9815504a6975ce766766fc8089729be8121df3ac 2026-10-20T01:13:36+08:00 feat(qwen): bind original HF diagnostic profiles
dbb5f4f4cc42e28c91b12e906f9091ef76510c03 2026-10-20T01:53:12+08:00 feat(serve): fuse unequal decode frontiers
cadcbd9d1216b00ca4b69cae9bc04a409e347193 2026-10-20T02:07:20+08:00 feat(qwen): qualify pinned CPU HF operators
1b599df06d8d7c8c4912838836b272e28e04aca1 2026-10-20T02:08:49+08:00 feat(bench): isolate owned provider lifecycles
d41163432fd6233c7bd88f07b64fb71b1086a902 2026-10-20T02:11:18+08:00 feat(artifacts): own verified tensor windows
2a32aff615fc4f7f6718f14a15187825730d6897 2026-10-20T02:43:37+08:00 feat(qwen): compose typed source operators
c5eb1f8e717c79146937e585a145a0052eff747b 2026-10-20T03:22:04+08:00 feat(cpu): qualify source sigmoid and RMS
d0a60844217fc106fe8c70bb3a99073eef0387c8 2026-10-20T04:19:25+08:00 perf(gpu): fuse source activation preparation
5350a4d40b535106240d48fec8aedd459d5cd17a 2026-10-20T04:56:45+08:00 feat(cpu): qualify typed source windows
f78892957f09a84838246aede4fef2be040a0d18 2026-10-20T06:13:20+08:00 feat(gpu): fuse native teacher-prefix serving
bfe62df68d763b9d5c6b87c891be6a62b3593de9 2026-10-20T06:35:48+08:00 feat(cpu): execute leased source-window plans
fe2a34213009cc160f4d82068bc7dd1f49d05c6c 2026-10-20T08:12:19+08:00 feat(cpu): qualify original attention fragments
9a199017f18f1bcb1a8053685be3c3c29236ec90 2026-10-20T08:21:04+08:00 feat(gpu): qualify batched graph replay
fff1842594f8011b761d08fcc1690311b72543e3 2026-10-20T09:20:27+08:00 feat: qualify trained BF16 operator slices
fb68d8fa165de16e21d1bd37afef3790b80da971 2026-10-20T13:27:39+08:00 perf: combine GPU state scatter certification
bb089823653f89d64bc97935cd7b11b7c3e66fd3 2026-10-24T16:55:25+08:00 feat(core): add run policy request validation
bd277cf7667381333d54ff3c7f249dae92b8efa6 2026-10-24T16:55:25+08:00 feat(core): add reserved domains and workspace
35c9962d8633599a8b2393b29e1fa26f02677272 2026-10-24T16:55:26+08:00 feat(core): admit device backing views
eb49fa0e945bec802db28472deb50ab94b715284 2026-10-24T16:55:26+08:00 feat(core): record plan topology and source programs
ec30569bc6b921bfc09af888978d22d23e957944 2026-10-24T16:55:26+08:00 refactor(cpu): share reference math lanes
9dd1ebfb18623e27cb5ef07940f5e6839f1ecaed 2026-10-24T16:55:26+08:00 feat(cpu): add original torch CPU operators
a3716fd0de505b5bcf1e7e43ff3da7154c1ac8ff 2026-10-24T16:55:27+08:00 feat(artifacts): select cached sources by locator
77c48b315feec501082c2609077033331cc9fcb7 2026-10-24T16:55:27+08:00 feat(artifacts): authenticate reusable source access
7ba9278b0a1ab97514f9511fc7a72ae90eeccf1a 2026-10-24T16:55:27+08:00 feat(cpu): evaluate reusable text source windows
dbd836351d40289eb8d99f7e1f648c139530de34 2026-10-24T16:55:27+08:00 feat(cpu): add haswell and zen bulk kernels
b026d0f9d5024d06f5b83768191f70258c1aa306 2026-10-24T16:55:28+08:00 feat(tools): add qwen27b output-tail reference
85d86b3c5863d9d78b6bed0b7db6e36ed28686cf 2026-10-24T16:55:28+08:00 feat(tools): add qwen27b attention-tail reference
0c7a4a9770f64d3de1e3c920a438c88e088668c3 2026-10-24T16:55:28+08:00 feat(tools): add qwen27b full-window reference
af5aa8498550fb98e69734c18769cd2c84299278 2026-10-24T16:55:29+08:00 feat(tools): add qwen27b layer9-prefix reference
6dd5970776b7b5839ebc6b60c3694049d31f406b 2026-10-24T16:55:29+08:00 feat(tools): add qwen27b initial-core reference
2c70ccb66c780874998e77bbdb2c9e33387bdf2f 2026-10-24T16:55:29+08:00 feat(tools): add gemma4 tokenizer reference
c870725898c9a0595609f53f7310182c5bec4fac 2026-10-24T16:55:30+08:00 feat(tools): add gemma4-e2b text reference
8b0cc32c951027123ffa2a9dde1e5f7dd6c2a57a 2026-10-24T16:55:30+08:00 feat(tools): add gemma4-e2b CPU matrix
283765cff8bdbb69d4d6674d7fd28360238ea693 2026-10-24T16:55:30+08:00 feat(tools): add portable BF16 slice reference
b5015e7a77e37172236c602b65a25c74a3caead4 2026-10-24T16:55:31+08:00 feat(gemma4): admit dense 31B text metadata
bebfeaf2d81722ebca4c11613e8a40d5834e18df 2026-10-24T16:55:31+08:00 feat(gemma4): add E2B text reference and tokenizer
3002089997fcee9e43abc3dc60404cbeba1f52ab 2026-10-24T16:55:32+08:00 feat(gemma4): add E2B learned-source diagnostics
225fc5e1a520637b45c3c76f36e4b58963c1de08 2026-10-24T16:55:32+08:00 feat(qwen35): record window, bulk and CPU tail sources
9af407bf3ff5763c20f4e9fddbf8061fa16f8179 2026-10-24T16:55:32+08:00 feat(qwen35): add qwen27b window replays
c4cbff17e6c431ee1d60e67cd725b3bf0f2fc4a7 2026-10-24T16:55:32+08:00 feat(cuda): observe device metadata
d9d4f8b9bb500dcf632b37efae50c2926152adf3 2026-10-24T16:55:33+08:00 feat(cuda): own compiler, source and cubin actor paths
4bc7c429d585d720600a9a33f0da25da4baceff4 2026-10-24T16:55:33+08:00 feat(hip): expose availability and prepared native sets
248bcec54c6e3d94992e379b509d84fd5864cbbc 2026-10-24T16:55:33+08:00 build(gpu): add qualification feature surface
68c2f5c050136978f5e7c01958fc08e6f2504d60 2026-10-24T16:55:33+08:00 feat(gpu): add shared kernel source primitives
abca70fdebda8e395b513133e79f938a9b2d84cd 2026-10-24T16:55:33+08:00 feat(gpu): add original source arithmetic kernels
63583681ae8c3f850f1b6e00dad4de8a7f421c56 2026-10-24T16:55:33+08:00 feat(gpu): add native policy and physical identity
fe95ecdde7800a6a9079f70ff7aa90f6780cad15 2026-10-24T16:55:33+08:00 feat(gpu): add vLLM input RMS qualification kernels
405e2462ff310c6ceae1e43ee54c54e2b13907c0 2026-10-24T16:55:34+08:00 feat(gpu): add vLLM GDN instruction kernels
71c59964149f280787359063f1c1e17eb52fef98 2026-10-24T16:55:34+08:00 feat(gpu): add Qwen2B GDN instruction assets
00d4bbabde0491f6f0be513f0223558bc813edd6 2026-10-24T16:55:34+08:00 feat(gpu): add .8B short-HF GDN call catalog
d031166b726a1ab07ef88ea8c1401e01df8285ee 2026-10-24T16:55:35+08:00 feat(gpu): add vLLM CUDA qualification scaffolding
8dbe361c1e53db00738f1e25ab11dca44b5f5840 2026-10-24T16:55:35+08:00 feat(gpu): add layer, attention and GDN cubin checks
13ae2ffb7d0510073fd36dfc137693a9de2f3bb8 2026-10-24T16:55:35+08:00 feat(gpu): add canonical47 model qualifications
bbafb1de8c769c0ad1d158ffc138b7db0144f319 2026-10-24T16:55:35+08:00 feat(gpu): add Qwen2B qualification leaves
f619f7eed8377b8c8bf9838a642b86d049881180 2026-10-24T16:55:35+08:00 feat(gpu): add short-HF and full-model checks
3e89f1919993081535b5fe2820da05ea9d840b82 2026-10-24T16:55:36+08:00 feat(gpu): wire resident leases, program and native sets
136a4a8692d9aefe19e1d7ef25eab4e2fc48d561 2026-10-24T16:55:37+08:00 feat(facade): expose CPU routing preparation
08b30b571e4b3557be78318b66365030c1bf5bb2 2026-10-24T16:55:37+08:00 feat(facade): add canonical47 original-HF loader
bc66207ce9c5a1ca96e609e975151541579c4390 2026-10-24T16:55:37+08:00 feat(facade): add short-HF serving source loader
e44a53f75320e21df9642f5c3fe711efe8fccc04 2026-10-24T16:55:37+08:00 test(facade): add short-HF physical suite
08fc5935e124f6af33c5010fc4309648539e1641 2026-10-24T16:55:37+08:00 test(facade): add short-HF cohort suite
a4a5b6322f343004d1ecdf13b264654da466424f 2026-10-24T16:55:37+08:00 feat(facade): add Gemma4 E2B CPU candidate
2058dcb1ea3f8053b6906561d0698df392c91731 2026-10-24T16:55:37+08:00 feat(facade): add qwen27b HIP teacher source
71f496b190dc2c7ced495f9d4c1f5c5da9495f33 2026-10-24T16:55:37+08:00 feat(facade): add Gemma4 HIP drivers
752642361cb4ad2f769b868f071a3307e3b01651 2026-10-24T16:55:37+08:00 feat(facade): add shared HIP lifecycle driver
2bd8167db2c1d8817d38e4851dd45f9c2c9aa6e4 2026-10-24T16:55:37+08:00 build(cli): gate finite HF serving and run features
b3c5a532d72a266fc3d66e13c5c1a0f7ad1bd084 2026-10-24T16:55:37+08:00 feat(cli): route run requests and add gemma4 candidate
45570bdb3d7f21f531b8cbc840633b8d2eb83e4c 2026-10-24T16:55:38+08:00 perf(serve): plan atomic source prefill waves
e6b07efdbb2d6ec56d979ff09b025c8071038f06 2026-10-24T16:55:38+08:00 feat(serve): route and serve canonical47 sources
9999adb1a66d646f76cd1152c32da51b22c9eded 2026-10-24T16:55:38+08:00 feat(serve): serve short-HF BF16 B4 cohorts
ef9cffd413dfe3a02d8ed54e4a2ff3a5e317e10b 2026-10-24T16:55:38+08:00 chore(scripts): extend gate for HF features
2225af34b82e0e60da0f50dd4c5ec76eb827ca48 2026-10-24T17:01:02+08:00 docs(agents): codify parallel execution cadence
abb6b3b3b233e6887e182acb7c95ed26718601e4 2026-10-24T17:01:02+08:00 docs(research): compare optimization mechanisms
4b44d313bedecc6b82ecd44e317d786db2e08cd5 2026-10-24T17:01:02+08:00 docs(policy): adopt reuse-first provenance rule
46996b66e39e832f05d0174419e9c8f69367128f 2026-10-24T17:01:02+08:00 docs(plans): design composition and batch state
6ef74b048e830f1075d4d5ca4296cc801b8b6871 2026-10-24T17:01:02+08:00 docs(cpu-gpu): record shared memory and parity
fb7ea9a9a77b95543544ae131fb44da26522e7a7 2026-10-24T17:01:02+08:00 docs(numerics): record F32 closure checkpoints
424c672768b388f1738187baffafba7973c4cd08 2026-10-24T17:01:02+08:00 docs(models): record source20 and Qwen audit
da28f851448b8cebdd8b5c132f1a7371a6b39380 2026-10-24T17:01:03+08:00 docs(acquisition): record Gemma E2B acquisition
0c966799d7830c6b0a3d6b354fcf5318d36add9d 2026-10-24T17:01:03+08:00 docs(cpu-gpu): record current checkpoints
c57648ac54ff02a5370c3a26e4dc0c1634cee7da 2026-10-24T17:01:03+08:00 docs(memory): adopt shared pools and paged state
b552e2198d66400c94fbb56d4dc5deeb7604115b 2026-10-24T17:01:03+08:00 docs(composition): record shared HIP checkpoints
6075a4386589ca5d6e0d857d14f3ef4c6a4879b7 2026-10-24T17:01:03+08:00 docs(serving): certify transport and add batching
3b9da530d922814e8e5dea75574c2d4fdbc11462 2026-10-24T17:01:03+08:00 docs(distributed): defer mechanism reconciliation
38e3d77eba85875f0a3b2131a74c0de42933690b 2026-10-24T17:01:03+08:00 docs(platform): add Vulkan parity and reuse gates
a765f221279c9f803f0dacbee314c6ad4126905f 2026-10-24T17:01:03+08:00 docs(cli): add device routing plan
309b17d81bfbe7878fc8734adee424cb7785f533 2026-10-24T17:01:03+08:00 docs(handoff): archive history and run notes
ac17d49c07b2cb0e4653db19dc8621c6f0591279 2026-10-24T17:01:03+08:00 docs(handoff): preserve paused continuation text
7115754b639aa8bfbaab3a9231b97458fe945df5 2026-10-24T17:01:04+08:00 docs(handoff): preserve draft patches
11a47c74ecc55d92749dfd581d90254f02defdd5 2026-10-24T17:01:04+08:00 docs(handoff): preserve takeover and audit reports
b5d814f8fcab784514b4c6d9bfb1a11a7fa635d2 2026-10-24T17:01:04+08:00 docs(handoff): preserve canonical sourcebooks
e11fb986091f8fef71abf188e15177242084e596 2026-10-24T17:01:04+08:00 docs(handoff): index the 20 Oct handoff packet
4bf36731cfc4f66e02aff2f63fc32b653f6e363c 2026-10-24T17:01:04+08:00 docs(plans): activate CLI, composition and compute
8f01635305af920eadba96a7d4f1d6b9e8f588cf 2026-10-25T10:00:19+08:00 feat(core): portable admission and thread-control roots
5d40cf507a2e8f5f159b664ddc83876d73745607 2026-10-25T10:00:23+08:00 feat(cli): shared pending queue and ingress credit
52f76a0e0dc3ce8e09476af1460193a03a551227 2026-10-25T10:00:27+08:00 fix(cli): short-HF tokenizer credit and HOST8
efd7a34220ce168b5a5c851226b400e07e95a095 2026-10-25T10:00:32+08:00 fix(gpu): CUDA fragment inventory and host bound
19ff0598dd79d12ade182d90a855ecf9c60b1f59 2026-10-25T10:00:38+08:00 feat(gpu): Vulkan native catalogue via cubecl-wgpu
29a98c1c7c086d6367308ad9468c427046008cc5 2026-10-25T10:00:44+08:00 feat(gpu): HIP source-page events and witness
e8db111158991339f76e5b06ed3acb8b370f302a 2026-10-25T10:00:48+08:00 feat(examples): Qwen27B HIP matrix, natural tools
332038ea78ec8b1e69a2a29d3ca09370d712ef6c 2026-10-25T10:00:52+08:00 feat(examples): Gemma4 HIP final-owner and matrix
68e13a8114e130ab89ac4c55ee9ad14172faa42f 2026-10-25T10:00:52+08:00 chore(cargo): lock assembled five-lane graph
74d0c71998709de48115db16507ea0109b23f65b 2026-10-25T10:02:44+08:00 docs(plans): record assembly, C16 and handoff
1269b8eae908a40e79d747e923e29b708d84ba12 2026-10-25T10:04:00+08:00 docs(plans): retire five landed lane trees
06b89da455655ef039be2a87fd1e29917ce4d2c3 2026-10-25T10:28:19+08:00 fix(gpu): HIP probe witness page id typing
c217c35d480eb33b04eb981496d552a816b169e9 2026-10-25T10:28:24+08:00 fix(qualification): V state base parents, attempt-2 bases
3941a3c9fe0ee5dec0b9f836a86e992161f5f8ba 2026-10-25T11:02:16+08:00 fix(qualification): Gemma resident report-phase envelope 4GiB
e6ab074e3118a7200ea0a620a92b3a0990bbcac5 2026-10-25T11:33:24+08:00 fix(qualification): H/G source census skips build dirs
1565683442e80f529d810d98c63ca2583c5b01d7 2026-10-25T11:51:52+08:00 docs(plans): record queue GPU acceptance results
2304f8b4d2079c5312a8020f14af49c2c3832ab3 2026-10-25T12:03:16+08:00 fix(cli): precedence, ambiguity, pre-client hold
d3d21a35c5b75272d82947ccd9f351bc7888e3c3 2026-10-25T12:07:48+08:00 feat(gpu): B4 grouped transport and certification
6f1e0947cdf7673cbe4626b983fe894fbad1744d 2026-10-25T12:40:33+08:00 docs(plans): record main checkpoint and B4 merge
9fbe15e9e45c0d9c71ce00e73dfbdde343a77513 2026-10-25T13:08:47+08:00 docs(handoff): record checkpoint, HGV pass 3, successor queue
5749b732dc755c24d2179ba7dd7e8abc9380805e 2026-10-25T13:28:53+08:00 docs(handoff): record Qwen27B capture lane yield detail
488e0ea2c21e2301c2fdeaf82dab6de32789ea08 2026-10-25T08:14:56Z fix: repair native qualification contracts and GPU feature isolation
93b611acaa657b6999a19239e4596d4de47b4de3 2026-10-25T08:26:58Z fix: prevent Gemma HIP source-control payload shadowing
6edc6e2c99d9a2353d16a85ac689e76af3628be5 2026-10-25T08:33:50Z docs: pin refreshed H G packages and completed GPU-free controls
9e67b74000bc085a844d8844d954d82f8e196493 2026-10-25T08:59:57Z docs(audit): independently validate post-4bf GPU boundaries and resume holds
03f2e662f219797da524d0e4d6e2aabef979471b 2026-10-25T11:08:08Z fix(gpu): certify grouped B4 CUDA ABI and exact fixture retirement
3d9866dfd6cf66009c9aa6efe0967e4a134e4e41 2026-10-25T11:09:42Z docs(audit): distinguish 23 model-dependent B4 scopes
a39ca967ea2029400649284285ca18620544152c 2026-10-25T12:34:50Z fix(gpu): qualify grouped transport from independent fresh state and seal KV cache views
7362699b7023b30c815482a71900af40b0a0688f 2026-10-25T12:35:00Z fix(serve): preserve prepublication cancellations and bound decode context
5229e53d7b46c0fd322217b216b2e11a8b94a433 2026-10-25T12:37:00Z docs(review): record architectural and correctness findings and qualification gates
83a31747679385cd5951487e6bcdfb9a2f2d1f24 2026-10-25T13:51:15Z refactor(gpu): one typed closed CUDA ABI for metadata and bound handles
095bf8a4b97dbd7a0561c7ea4316ef7ff8167bdd 2026-10-25T13:53:06Z refactor(core): collocate Qwen3.5 recipes without changing type identity
2a47d25181f1a8d91a9489d9f2d621d3807bb329 2026-10-25T13:53:16Z feat(cli): concise run placement aliases and observed-device inventory
d9c5872c73ee4d7b2bbbe1079446ac7634f8d919 2026-10-25T13:53:41Z docs: scope ABI family and run UX refactor acceptance
5399ab93929eab8e6448078ae8126144b147b238 2026-10-25T14:31:01Z refactor(cli): use device selection without redundant on alias
d84853337bbd2830f50a9c4e704d266990fd2ce4 2026-10-25T14:59:13Z feat(cli): resolve Linux GPU device nodes to verified physical PCI identities
d0fff261904fcb7836c0f71d0682028a12fd53b4 2026-10-25T16:26:13Z docs(handoff): preserve complete ZeroGravity history, open work and takeover prompt
a857d45ddea827457fb59823ea6597f3eca03d22 2026-10-26T02:55:15+08:00 test(gpu): repair B4 qualification fixtures
6610b6195505d7d8fb17640660461310fa1c3d41 2026-10-26T07:57:31+08:00 docs: record B4 and C16 outcomes
6d855599531e8f67ec6cd1e1e00f7e1b186ff7f3 2026-10-26T08:47:53+08:00 test: add shared model correctness checks
195099ca583ace766d075682927f590360c823a5 2026-10-26T08:48:41+08:00 docs: set all-model correctness policy
405a3e6358c85917506f76f780094de89eaec01b 2026-10-26T13:39:06+08:00 refactor(formats): group codecs and format homes
d9a8ecf6187216c8b909c60e44927b06f256afdc 2026-10-26T13:40:43+08:00 refactor(correctness): name input geometry
fa844a663ff9ab4f701146be61bf2353cf580385 2026-10-26T13:41:20+08:00 docs: define module ownership and naming
a5ed7de9e0f0682d86aca09d0ec73336a5f2ecf1 2026-10-26T15:07:57+08:00 refactor(models): contain source plan ownership
c90b8cd7e2fc8b05784d1e27c907347ea4319f97 2026-10-26T15:30:04+08:00 refactor(gpu): separate providers and bindings
41e93370b7b3b9edc45cd9575eaa6b9d6845bed2 2026-10-26T15:41:21+08:00 fix(gpu): preserve plan and capture guards
fc734571e6e342b376f406aac7fc2ef4e42e69f2 2026-10-26T15:53:31+08:00 refactor(app): compose model execution routes
4a0f4552354ad8f5931b7cb45bd85d34a0026eeb 2026-10-26T16:01:43+08:00 docs: record completed source checkpoint
5c2fe532c68e98a0c750b5ff27969ae965ec3f9d 2026-10-26T17:13:15+08:00 test(cuda): restore source admission checks
9c2c51ab291580a5c47d9528f8ff2ed87bd8228a 2026-10-26T17:15:22+08:00 docs: record side-worktree retirement
fe1762568c96eead3893f338d0181b8634a98303 2026-10-26T18:26:57+08:00 refactor(cpu): own shared GDN convolution
263ff791d04eb50fd1aeca85249b0f7aed3cb1ad 2026-10-26T18:27:45+08:00 refactor(gpu): remove model dependency
e7fc3150927104b2f22dd020f63efb8817a75d82 2026-10-26T18:31:38+08:00 docs: explain benchmark container termination
50abaf71121a2ffbb6a649f16aafe185650c1388 2026-10-26T19:20:42+08:00 fix(serve): join workers on process shutdown
c84b98744955c43f7a6a992289c7d8a977a0ee9e 2026-10-26T19:21:14+08:00 fix(bench): retain owned stop evidence
4f427dd814f734afdad798b053141946c998347a 2026-10-26T19:28:49+08:00 docs: record serving shutdown validation
cd4159d26048b0395c40235b320b7d48df6ce53d 2026-10-26T21:58:07+08:00 fix(lifecycle): retain owned command streams
5a18b5bbe6a80976b9255813070c09d2cae61a3c 2026-10-26T21:58:36+08:00 refactor(cuda): derive capture scope ownership
61f095cf760ee6b8d1f015b71b13ac46835a4917 2026-10-26T21:59:12+08:00 feat(reference): capture finite Qwen prompts
3612ecb7077d18685e926c698b812e4e2e60375f 2026-10-26T23:14:16+08:00 docs: define production and test boundaries
1f08e5525b5ede7124054156083e5a4ff67d151c 2026-10-26T23:23:45+08:00 refactor: separate rotary and test preparation
d74f59a18c6345dbea963861acd519c1282a26fa 2026-10-26T23:24:36+08:00 feat(reference): emit captured CUDA recipes
5f4051ecbfeaa1350055db47fcd281fbeb81843b 2026-10-27T00:05:41+08:00 docs: preserve comprehensive agent handoff
a1c51f9d478ec8025c8da09e58a8a2626c4cfe32 2026-10-27T00:34:05+08:00 refactor: separate fixed rotary preparation
91a29586a292a7b8204e85cf4ed21cf03968b718 2026-10-27T00:56:52+08:00 fix(cuda): seal rotary production routes
9ad9aac42e0ea63ab78e8ce9a7836cc1d770ab62 2026-10-27T00:36:42+08:00 parameterize CUDA GDN prefill convolution
05baab5abb2e02702ab565d267b44474e9362e66 2026-10-27T00:37:11+08:00 fix: separate prompt semantics from CUDA qualification
d087012fa5f4d604069a015a7773defed5ae185c 2026-10-27T01:31:53+08:00 fix(cuda): correct parameterized Conv identity
0c63d89a54ee8a45a870cc383cef144acfb09579 2026-10-27T02:46:41+08:00 fix(qwen35-cuda): isolate finite qualification routes
ae936a8ab3102ebac682c947a63dbf105d91418b 2026-10-27T04:32:55+08:00 fix(qwen35-cuda): enforce qualification boundaries
7a910d9aab0f7f09f788ff983c678ec3fad25413 2026-10-27T05:04:07+08:00 fix(qwen35-cuda): make route selection declarative
ca0d2bb0f4323f7584a764e0e162219c47561b52 2026-10-27T05:05:50+08:00 test(qwen35-cuda): cover source-kind neutrality
f61a3ca618a6b1463a185231a86f879f13019f21 2026-10-27T05:32:39+08:00 fix(qwen35-cuda): restrict finite route targets
85b272476e2b61d65d0d5236abad3294b29ca117 2026-10-27T06:10:23+08:00 feat(cuda): parameterize GDN prefill sequence setup
1fe615113e1c859d665957632166bea96095f5fa 2026-10-27T06:40:08+08:00 cuda: parameterize remaining GDN stages
2e3bbeccf97b360ea3a5ab228623996e58228b77 2026-10-27T07:12:15+08:00 feat(cuda-gdn): configure remaining prefill stages
1b898ffdb9ac333b535233ca49c1f9da1c290c4c 2026-10-27T08:14:15+08:00 fix(cuda-gdn): seal explicit implementation selection
c0bdbb183e6d9112ee2c9d8cc1c5281cc35c30f4 2026-10-27T09:13:23+08:00 refactor(cuda-gdn): remove recipe execution bridge
eb38dd5e892f2120ac29cc2fb79b2dbe137a37fe 2026-10-27T12:25:58+08:00 feat(cuda-gdn): add fresh-48 output variant
ba7fb0389978d82e66d1832f50feb32238717eef 2026-10-27T13:50:22+08:00 feat(cuda): add qualification fresh-prefill capture
fadf2b1123309a2118725f2cc005d3edc7ea39f2 2026-10-27T14:35:30+08:00 fix(cuda): use fresh prefill reference layout
ae9a84394d3aa08d77e0edfe4285dc09b7862040 2026-10-27T15:30:30+08:00 fix(cuda): decouple GDN parameters from text kernels
532a35c437d4b941ea9931d9c1e3ca16d9b7d080 2026-10-27T15:46:05+08:00 fix(cuda): use fresh reference prompts
54d7ea7a3012b70f3b92ec8c45800a913e6dc811 2026-10-27T16:00:06+08:00 fix(cuda): admit fresh .8B row extents
e99c76e3e644b612a173a029230d44dd4e781fc2 2026-10-27T16:07:02+08:00 fix(cuda): account fixture host baseline
ad82feb5222d81a5f19fe1311744387b8b9959bf 2026-10-27T17:51:44+08:00 feat(cuda): add call-free Qwen plans
f7ff4f96be5110a9a064e9c78643897212a65537 2026-10-27T18:38:40+08:00 fix(cuda): retain ordinary plan source owner
1921db16695d8b9166399cd10eff793a0ab8e9cb 2026-10-27T17:30:30+08:00 feat(bench): expand bounded comparison matrices
1d278edeaa875aa8d85936566516388943d9227e 2026-10-27T17:47:51+08:00 fix(bench): retain complete matrix cell results
fe228abda89c1388efe9524439aa8d04c4d37c07 2026-10-27T18:36:55+08:00 feat(cuda-gdn): add ordinary chunk runtime API
52bfb519a245b114de90e6e2d5cb1de8a80497ca 2026-10-27T19:34:25+08:00 fix(cuda-gdn): complete ordinary runtime schedule
901d23eb518c4b9167707f8f716aebb50f3cac27 2026-10-28T02:27:44+08:00 docs(handoff): record 27 Oct checkpoint
ea66b3ab3096d08bafe0f28cfa9a5f0c12307e45 2026-10-27T17:55:20+08:00 Add dynamic CUDA ordinary text primitives
68393e5a644d661bf5e101d47cebd2290c1faff2 2026-10-27T18:58:38+08:00 fix(cuda): constrain ordinary text candidate
6b4a2ec7a8f81fcf1995f6aa3063dc2c152a7a5b 2026-10-27T20:38:45+08:00 fix(cuda): close ordinary Qwen review blockers
441fa64f52be6d3d8636e6dc8241d241e6b55889 2026-10-27T19:18:09Z fix(cuda): close ordinary primitive review gaps
ee67aa848f2603bb342029c811692361ec10b1e0 2026-10-28T01:06:15Z docs: close ordinary CUDA primitive review
c2494ac1665508f4eabf9aab08885db5adbfec62 2026-10-28T01:14:59Z core: bind ordinary source invocations
301e912f4046a6ed5d4e0e41c275ae2423a6dfb2 2026-10-28T01:37:46Z gpu: generalize owned BF16 GEMM geometry
dcfbd74f7ec060e640e734cb3765b0b585226bd0 2026-10-28T01:40:19Z gpu: admit verified BF16 matrix upload on CUDA
b116ccdbaba8ff4fa4657bd93e017cbb9d80a9e4 2026-10-28T01:43:13Z gpu: expose owned cuBLAS to ordinary CUDA
b4a926303ccffbea6c21bf1fac5b42a313ff8021 2026-10-28T02:45:47Z gpu: add ordinary CUDA projection owner
40c8721c2c1a257371526e6aad9cabf18a7c652a 2026-10-28T02:53:04Z gpu: bind ordinary primitives to owner status
71cd73779a39cc74ba558efa52c7b4dc5055a13a 2026-10-28T03:00:47Z gpu: add source-exact ordinary CUDA attention
e5f148e04c4ced0b0fda61efadace115b11c6017 2026-10-28T03:14:12Z gpu: reuse ordinary CUDA source projections
18579ffc25818c99f9e8d884daf9fadc5a0c4dac 2026-10-28T03:14:12Z gpu: make resident state precision explicit
9abd1d973d1097681d859d65b1cc0c50d4ed850f 2026-10-28T03:15:33Z gpu: validate typed resident state layouts
4fde7574e75274a8971d82d0a07f1b0d54c8dfa1 2026-10-28T03:22:41Z gpu: share canonical GDN state transport
cba4f5e2ceb689818892bee641ad55d8f4398c94 2026-10-28T03:25:29Z gpu: partition ordinary GDN runtime schedule
8e964cef1d1df61f68ff540131476441e8c72694 2026-10-28T03:28:50Z gpu: expose neutral ordinary GDN primitives
8d5035dc051c86107c35ba4da4dad13284bdbc41 2026-10-28T03:53:51Z gpu: add ordinary CUDA parameter transport
57379b3452c15d4cbd1782fabe1244c54e8333d9 2026-10-28T04:02:12Z gpu: derive ordinary CUDA resource layout
685e9be69162c0480302ca6daf609f2926714276 2026-10-28T04:04:13Z gpu: reuse ordinary CUDA matrix packs
6f5d43d2bb9b92723df5c0deff4260a41e74fcc3 2026-10-28T04:08:42Z gpu: prepare ordinary CUDA cold source owner
11aaf2bb19ceb11bed752945fb5a9973d4372834 2026-10-28T04:12:11Z gpu: bind ordinary CUDA resident body
81318adeb432ec1fda08ef8fa930478253b01ddd 2026-10-28T04:12:33Z gpu: expose ordinary CUDA cold values to body
00ee779cb096ee567be920991d4e7d3f3b4c5ee5 2026-10-28T04:16:42Z gpu: preserve ordinary residual and GDN scratch ownership
3865ffd1ee31b530ab74e3e09184b1949c932ad6 2026-10-28T04:29:42Z gpu: bound ordinary attention to active KV
c7b1962b60ceb2f00ff204954c0a78f0aed17559 2026-10-28T04:34:06Z gpu: retain ordinary GDN local states
a8eef412f2b27020dbdc559ce2b6b074bccc7445 2026-10-28T04:36:01Z gpu: add ordinary CUDA active view transport
d0f557a47e2f1359a08c30822c5b3bb3fce5942e 2026-10-28T04:44:15Z gpu: execute ordinary CUDA through resident owner
9edece37532ed6eaaa09153885d32c3faad4a73a 2026-10-28T04:48:26Z gpu: separate ordinary CUDA owned and execution accesses
be5362e5f26ed4e74db75341892bad9e0f5f2907 2026-10-28T05:00:42Z gpu: load ordinary CUDA resident model
1f69da6fe186a8959a2f8bc20616ae8838f761ae 2026-10-28T05:04:45Z gpu: execute ordinary CUDA as one resident chunk
b379a76963f60893f5197ac3d882f58778de27c5 2026-10-28T05:13:14Z qwen: add ordinary CUDA qualification adapter
8c2459929a81256760e337283eb25b73e152b8ae 2026-10-28T05:47:28Z gpu: close ordinary qualification build edges
c680b6c5b6902a87851a3073e7360586e2b09fa5 2026-10-28T05:54:59Z gpu: add ordinary CUDA explicit shutdown
eeb38bc60f33741d8d3afd17b92648fb2a201a2c 2026-10-28T05:57:51Z docs: mark ordinary CUDA qualification-only
5e29118c2f781ea8dba5b4d85fe04c570284c988 2026-10-28T06:03:39Z gpu: qualify ordinary CUDA context admission
ec44357c57704420731d2a74f3ad888cb45cfb6e 2026-10-28T06:07:58Z gpu: add ordinary CUDA request cancellation
f2603b3d90dcb70dbd668b5b2d23beb07814f140 2026-10-28T06:16:15Z gpu: canonicalize ordinary KV root snapshots
e93e7b42dda9bc9ed2f6ac92c34ae7f369428220 2026-10-28T07:03:53Z qwen: add serial ordinary CUDA parity harness
014b77385211a3431ecc90f4e2eda86659917cfd 2026-10-28T07:07:33Z qwen: add runnable ordinary parity qualification
ceae923844d08dbc4741cdcbc2789034d51f0f64 2026-10-28T07:12:39Z gpu: harden ordinary BLAS retirement boundary
9e566438d6a428ffb7aa363fcc05029fe1a9c59d 2026-10-28T07:15:28Z gpu: harden ordinary BLAS activation failures
487a796a52b65e1170a961ba9a1e23eb5f426844 2026-10-28T07:39:21Z qwen: isolate ordinary parity across processes
ee70ec6aa01e1da846ea385afcd64f7acf280cd4 2026-10-28T08:25:59Z gpu: use source-exact ordinary rotary
78426672edaccbb669254b87a8e47f759586049b 2026-10-28T08:26:14Z docs: record ordinary rotary qualification
ea4b67b3fc8ce8ed56f0c8f367ea7b0575881cc3 2026-10-28T08:32:39Z gpu: refuse legacy ordinary RoPE contract
62ec2cb6fdcb8488b2e0b385ff995d903dcc7c89 2026-10-28T08:45:56Z gpu: isolate ordinary CUDA feature boundaries
13703a4552919f795dc6b747ef2e9547c174a275 2026-10-28T09:23:09Z gpu: add ordinary owned BLAS qualification
9ae5e7ebee018695361ee31246df44d23a676d8e 2026-10-28T09:23:10Z docs: record ordinary provider qualification blocker
ccebfb05c7d5efb4127051059de7fe50eed227ed 2026-10-28T09:40:14Z gpu: qualify ordinary primitive numerics
f6edcbb9a878f9d72ea6e1eeb5fbf2893a07fdc5 2026-10-28T09:40:14Z docs: record ordinary primitive parity
642e24529111ed966a35e4825dee7bac501f3663 2026-10-28T09:45:58Z gpu: qualify ordinary residual RMS
ff512af4a3a28b71c8861501528d04b5f61f1c45 2026-10-28T09:45:58Z docs: record ordinary residual RMS parity
77dcafa4934a02457bf09701e1116ea8a9494181 2026-10-28T10:16:27Z gpu: qualify ordinary multiply and KV append
107e301fe437180b083d8c889345ab956662da65 2026-10-28T10:16:27Z docs: record ordinary multiply and KV qualification
04fad2a20f9c5992c4ee723859c4dddea743b113 2026-10-28T10:19:14Z gpu: qualify ordinary embedding
4274b192e0793b838956f8ef7c4b2b6ccf7009d4 2026-10-28T10:19:14Z docs: record ordinary embedding qualification
f17540b053fe0fa94fa99243a6f3b6bd65b7cb20 2026-10-28T10:24:38Z gpu: qualify ordinary attention wrapper
7c67dd8f2c6caf9422c36d7644554a2c345065fe 2026-10-28T10:24:39Z docs: record ordinary attention wrapper qualification
3eb3f74150066c56d38b9f18ffa4aa492dd9c157 2026-10-28T10:28:16Z gpu: qualify ordinary canonical state transport
a6743c9f1fe7638340639065b5e27fa1bdfe6d9c 2026-10-28T10:28:16Z docs: record ordinary state transport qualification
8f4450c06aee8979f18cc68510152b5bef981366 2026-10-28T10:38:32Z gpu: fix cached GDN gated launch grid
8dc7cdffabbae94b2c8a4d68dea28aa66cd2f975 2026-10-28T10:38:32Z docs: record generic ordinary GDN qualification
ee5dfe10966ae7ee7be47b1904e35657fb07db66 2026-10-28T11:17:10Z scripts: add serial ordinary cached47 parity runner
c4f54ebd0a92d8fa21a420d63c56b1f16d3173ae 2026-10-28T11:17:10Z docs: record serial ordinary parity runbook
b5c353ceef5c44d92511396933a1494b8faf0bb5 2026-10-28T11:22:31Z qwen: harden ordinary parity restart
d03bb3a94c431b9a50a7767a9da3cfe177bebc7c 2026-10-28T11:22:31Z docs: record ordinary parity restart semantics
3ed372de521553bc30d6e12159cf368dfcf198b2 2026-10-28T11:38:17Z docs: add comprehensive 28 Oct handoff
563ad344a0e56ac6500afa19096d5acdcc93cb44 2026-10-28T13:40:57Z feat(cache): add human by-name snapshot aliases
be1e81a69482b9838ba6f2f943fc6fd001dc3d85 2026-10-28T14:03:47Z refactor(cache): group by-name aliases by model
35a025c6562a36342552f62245a3c75cca8b66c4 2026-10-28T14:45:17Z fix(cache): stabilize human snapshot aliases
76d1561824e98139dd8ce07e352bee4ee1fcd11a 2026-10-28T16:17:15Z fix(cache): make human aliases crash-consistent
8bdd54b0f13aed240efac150afc9bb3a8c0747d2 2026-10-28T18:56:39Z fix(qwen35-cuda): authenticate parity evidence
f6209d923a5f1e3d0cf967a93d887528b5cf166f 2026-10-28T19:54:19Z refactor(qwen35-cuda): simplify parity evidence
768b724a45921e3fe2c3fcf305c3fdac2d67817b 2026-10-29T01:58:37Z test(serve): align qwen cuda qualification boundary
932f56499a24ce2a8f38ff5ad203fcb2716c831e 2026-10-29T01:58:42Z refactor(cuda): move gdn recipes to capture tooling
734d29f52ed5e83abc069fdd7ca3902cb6803e47 2026-10-29T01:58:48Z refactor(qwen35): retire obsolete qwen27 replay harnesses
2ff3d49e70a1b80fe708c9e4ae0d8ff741990548 2026-10-29T01:59:07Z docs: add fabric vnext transition design
15f5bd3ae26891f67380edeb1a52f4afb2cae16b 2026-10-29T01:59:14Z docs: record retired worktree and run cleanup
9abd9365d1d9334f6cbaf8a348278e5d977384dc 2026-10-29T02:10:35Z test: retire machine-local capture dependencies
ace934b4a402278392e30ea63e2cfade0f5208df 2026-10-29T02:12:38Z docs: prune superseded handoff archives
7b5e50e926b0724b7ad0e630afd58c6a0075644c 2026-10-29T02:15:10Z docs: normalize current planning provenance
379d91d42aa247b0548b02211505c3aa9743658e 2026-10-29T02:51:16Z test(serve): make pid1 shutdown control capability-aware
ae8cd89d29f1b143f0fa8a7ecffacc9a125e5128 2026-10-29T02:51:16Z test: cover qwen27 layer9 reference contract
ff89b87cd3d288e6dc83a2e4f423e19045142d68 2026-10-29T03:00:59Z docs: add canonical current handoff
11664d02f740b60aad2d9bdee69f9cfd3dc42593 2026-10-29T23:54:30+08:00 test(cache): make private fixtures umask-safe
60907d12861bc3338d5e20d2eef749510aa8cdb6 2026-10-29T23:55:02+08:00 fix(core): report residency refusal extents
97c52ec8ff7530e7f7cc44adc5ea19a5e3df4a05 2026-10-29T23:56:18+08:00 fix(cuda): restore reference capture profile
440768d58869bf672943c64b055cd56cf2091287 2026-10-30T00:36:51+08:00 docs(cuda): record provider boundary findings
2983bb72c5832c6efe5d9117281afa33a31517a2 2026-10-30T21:23:06+08:00 feat(cuda): own BF16 Qwen execution
b262d2e9e5d0b29672c2438e72f72b6754bb9e90 2026-10-30T21:33:20+08:00 docs(cuda): record execution qualification
99f0232d066c8cef7d8302adb1015ad7dce05643 2026-10-31T05:10:36+08:00 fix(gpu): preserve declared parameter rank
825ade8aca3940a838d372823696743683beb88c 2026-10-31T05:20:53+08:00 fix(hip): preserve natural diagnostic ownership
8ca41db024886018afdf0975e34649d4f5a7f082 2026-10-31T05:24:29+08:00 docs: reconcile Qwen qualification campaign
d4d5df4cf771c2981a667de92f526771064359f3 2026-10-31T07:32:12+08:00 fix(cpu): preserve original attention PV order
f6fa4b993687f8d8c5c93566602182fcb8d5bdf9 2026-10-31T07:33:34+08:00 refactor(qwen): share source execution
87ba2e2edd6b6d55c6d24adce89942e5e20f9f87 2026-10-31T07:35:06+08:00 docs: record source attention qualification
8a1c6b1456a6921f1fbec950ad58fc6d54bfd36f 2026-10-31T11:39:56+08:00 fix(gdn): retain initial empty-state addition
1f29a42c656be1540db8996df5f198e174eb2bc5 2026-10-31T11:42:37+08:00 feat(hip): run source-bound Qwen models
85f33c77cbfe5babaa21e008dfec13070d2a80c0 2026-10-31T13:25:24+08:00 fix(hip): retain canonical source ownership
55536ce31e57049c2df378e085969ff141f16578 2026-10-31T15:42:51+08:00 feat(cpu): implement source chunk GDN
e2559b6063b42ac65a331eab2652a079f82258c5 2026-10-31T17:36:26+08:00 feat(gpu): implement source chunk GDN
1c35172b3c22f027c6f2b72fa0d177fe205797a1 2026-10-31T17:39:13+08:00 docs: record verified checkpoint and pause
7396984887e1acd1a1ebfa9b61e09fd100d2ec63 2026-10-31T18:36:28+08:00 feat(fabric): add compilable base skeleton
c853ebb6594868a332462a6343a55a671c3ef797 2026-10-31T18:39:50+08:00 docs(fabric): reconcile plans and recovery
7fc5d17316d16a8edea0df9008e60f91e62aa81e 2026-11-01T12:01:52+08:00 refactor(fabric): unify source and state owners
8591b5221a0e09e22b146b22a9041ab7cc2a716b 2026-11-01T12:02:49+08:00 feat(fabric): wire CPU runtime and serving
27ac27892abe2abb28a08e2303d4db39d3f9f95b 2026-11-01T12:04:59+08:00 docs(fabric): record working key-path checkpoint
d78eb5537e8002823e17191b6ad5bb4c6f91319e 2026-11-01T13:12:18+08:00 fix(engine): reject runtime self-shutdown
ae7ec111114a678d4474e575b3da064cfe72190a 2026-11-01T16:15:56+08:00 refactor(engine)!: share CPU/GPU scheduling
72622ce6ae85db9c181434c55afbdc1e2d04faa5 2026-11-01T16:20:30+08:00 docs(fabric): reconcile scheduler checkpoint
d4e94b1d436b586aaab84cb4d21c795d16bb029c 2026-11-01T20:02:38+08:00 refactor(sdk): own run and artifact orchestration
bab4390c59bf07420270080bff7cb9f898f25d01 2026-11-01T20:20:59+08:00 docs(fabric): reconcile SDK ownership checkpoint
83060e5dd99d6a7aadd92befb9ea271dfd1891a1 2026-11-01T21:33:23+08:00 refactor(core)!: bind numerical/layout contracts
2f192240784d532a673463e6f53a50517e577692 2026-11-01T21:34:07+08:00 refactor(sdk)!: unify catalog and serve loading
51cd294d7c9616cf0e26ef6f9514aa0b44b8ae70 2026-11-01T21:36:41+08:00 docs(fabric): reconcile contract checkpoints
860ce389b5de888604a3edbc56d07dc70b21dfcb 2026-11-01T23:11:03+08:00 feat(engine)!: bind preparation to source owners
d0e03f5f77ea8ce6934fe7cfc7579b8eca7a41ec 2026-11-01T23:15:16+08:00 docs(fabric): reconcile preparation checkpoint
fef7123394d4d200c677edd077d38e7c39462749 2026-11-02T09:42:43+08:00 refactor(sdk)!: own run configuration policy
1f40880546dd1d7ce4b54250124055bee8482742 2026-11-02T09:45:16+08:00 docs(fabric): record source parity checkpoint
8d761c45e07cefbdcafc7d95f551efac83d21ec0 2026-11-03T02:36:41+08:00 fix(gpu): bind full greedy feedback packets
8c43844e0d214c7b9d3f74b772089973ed40c7e3 2026-11-03T02:47:09+08:00 test(gpu): qualify cold Gemma SDK lifecycle
9defb6c6e46b7579f6406305201c3f4c54107989 2026-11-03T03:43:03+08:00 docs(fabric): close scoped transition acceptance
1c999298dde1e4f7fd8d0812cd401681914aabeb 2026-11-03T03:52:24+08:00 docs(fabric): count supervised cases accurately
bb400c42136a798e177be82d3a1e6ca40b3b8431 2026-11-03T18:49:17+08:00 feat(hip): run ordinary Qwen source chunks
8ffb5113900a79c0b99d41bc5999d9892b1c4604 2026-11-05T09:49:52+08:00 fix(hip): qualify populated Qwen serving
b9d8355a67914f65577cd5b9d92928131aafda37 2026-11-07T00:26:35+08:00 perf(artifacts): reuse verified source chunks
a57590e2351beefc637f68129558d277fd8a0aa8 2026-11-07T00:29:41+08:00 fix(cuda): preserve output reduction boundary
4f1e8275f8270387b237223cf269be2cdf47ac79 2026-11-08T08:13:06+08:00 feat(gpu): compose shared source execution
d16f0968655c0775c327025bc9388a8a2b312b23 2026-11-08T10:09:34+08:00 feat(hip): admit explicit total device budgets
b364cbeabdde7f49647eefcd93467d38b1207969 2026-11-08T12:10:15+08:00 docs: reconcile checkpoint handoff
9d73d467e702ff31a66460fe345ff8b7de6575a2 2026-11-09T17:19:28Z perf(cuda): keep greedy feedback on device
9f45471a3cb7d4c8bfce4a87f36a54ea64c90264 2026-11-09T17:45:26Z perf(cuda): specialize decode gemv reductions
5836cb7da52abd55fdd65336b2a9a339d73a545b 2026-11-09T17:46:15Z docs: record repeated CUDA C1 feedback
55e195c640bbea14cc251139fa763fb063319974 2026-11-10T00:41:12Z perf(cuda): batch ordinary source decode
eea45fad75e7938dd6d88151b6635c25a61e3a0b 2026-11-10T02:06:43Z perf(cuda): batch cached GDN convolution
98976bebca69cbbaceef3bc0ac28a798462f795a 2026-11-10T02:17:41Z perf(cuda): batch cached GDN recurrent
03267124e9fa02bfd8fc39364f5fbc1e6452928d 2026-11-10T02:25:53Z perf(cuda): batch cached full attention
f50645e7c39f701bf915a5f41f12d45f288b4ea1 2026-11-10T02:58:31Z perf(cuda): certify BF16 GEMM stores
1e4d76ee6d7f1d43453936feb095548658b4d89e 2026-11-10T03:49:21Z perf(cuda): reuse output-head weights across B4
f28d280a4efeffd701d01f3bb852201a65d3ef75 2026-11-10T05:37:33Z perf(cuda): batch aligned full-prompt prefill
ccf441d46a64fd407217e33cbf5360046d2402d9 2026-11-10T09:25:38Z perf(cuda): widen native source cohorts to B8
67586ebf9244c873d26c4fa49a0e8fab4aba9f92 2026-11-11T00:01:10Z perf(cuda): widen large-K ordered prefill tiles
77ffc23826c2cf393f8341fbeb360f359b03218f 2026-11-11T00:11:13Z perf(cuda): widen ordered B8 decode K tiles
74f642237341f61df9e62f57b5a85da7eb05c02b 2026-11-11T07:31:21Z test(cuda): qualify populated small contexts
5a75a704b9ccb3854956e59a9ab2022810cc4e13 2026-11-11T08:57:06Z perf(cuda): widen rounded B8 K tiles
597da3f25021a711c0a30ab1fc3a54e12fa075e1 2026-11-11T09:14:15Z test(cuda): measure large cross-profile state accuracy
b95acdc8568efc317ab0875d231ab209fde1544f 2026-11-11T13:58:31Z perf(cuda): regroup partial native cohorts
2b8e90fff0df7c3f7ef96971ebaa8754a111ae85 2026-11-11T16:15:46Z perf(cuda): widen interleaved B8 K tiles
fa703fa9af2ea86cd6af5ef0220359bc6b55cf13 2026-11-11T16:49:30Z perf(cuda): batch chunked ragged prefill
0d9a56da48b5330d5d19394f1b44bd5b7c881aaa 2026-11-11T16:54:21Z fix(engine): require pairwise cohort compatibility
6a4e2c9312a7943ddba6b6a270225811b205b337 2026-11-11T16:58:18Z docs(cuda): record chunked prefill boundaries
d927fda0d7c23fc5e2fdcc4149528f774f97ab2d 2026-11-11T17:04:56Z docs(cuda): record queued B8 scaling
96a7ac74152e6e619faabb17aa2d42bf9a812146 2026-11-11T17:11:37Z fix(engine): require mutual cohort compatibility
8f8b4648d64f011b75af4c7e379e9ead1dd4e780 2026-11-11T18:01:12Z docs(cuda): record grouped transport rejection
e22c442e8f8aab5f6303a4af261ef19652db4b6c 2026-11-11T18:11:52Z docs(cuda): record B8 attention rejection
3a14c48203969c0a2a98b2c01596f7ad239381ee 2026-11-11T19:49:49Z docs(cuda): record M128 tile rejection
af7f34c35ebf848e4f0ef14ffdcf3db8345f9d3d 2026-11-11T19:57:27Z docs(cuda): record B8 row-group rejection
ea9b83b55b80b3a4cec602e55436bd143aaa47c2 2026-11-11T20:06:24Z docs(cuda): record ordered N16 rejection
51c776b81d7745e3c7565ab0257b6904abb427d6 2026-11-11T20:11:26Z docs(cuda): record B8 head cache rejection
54ea36859cff72dd4dd84504f9517de80ef2dd94 2026-11-11T21:08:36Z feat(gpu): retain qualified private B4 KV transition
9197cfa4e5116c18e3f77ffb57c2c27ddeb05ffc 2026-11-11T21:37:32Z feat(gpu): retain qualified batched source rotary
083cc6d966253c9604c6d3113a597742535e627c 2026-11-11T21:45:22Z docs(cuda): record paired-column head rejection
5d7dcb4bb1fa1134795464a40a99a6a428955c76 2026-11-11T21:52:55Z feat(gpu): retain ragged private B4 attention
da591acf4131c9c3d99afe584ab7078b8de62166 2026-11-11T22:13:58Z perf(cuda): admit ragged native decode cohorts
1361c1385864b3971793f65a14a7d2031fe14b4f 2026-11-11T22:22:29Z feat(cuda): retain certified BF16 row split
d1ec7236345745424920ded084b9314a41775fc4 2026-11-11T22:51:17Z perf(cuda): batch GDN row splitting
59ae8499621644d870bb72498edd29612ceefb8c 2026-11-11T23:22:11Z feat(cuda): add exact sigmoid multiply primitive
0f0af042e08b8293a3d8601e7666c3b82af982a8 2026-11-12T04:43:35Z perf(cuda): pack B8 head inputs inner-major
265e970051370ac747895e93c9af216a0c663f66 2026-11-12T04:52:15Z feat(gpu): retain scalar-position rotary batching
78ef7f279b52465be3133fb7b3b4083877cec55c 2026-11-12T04:54:17Z test(cuda): keep strict GPU clippy clean
be1cd9e5a29e0a7e4a5d21c1350d70fea993f94e 2026-11-12T07:37:12Z perf(cuda): batch ragged prefill cohorts
9204d75e3984d65d0165ce8bdb654001867f5b19 2026-11-12T07:56:21Z docs(cuda): record unequal-frontier prefill rejection
790d3de0724ca499b9529fc878316177a49fbd2e 2026-11-12T08:25:46Z perf(cuda): pack B4 head inputs inner-major
154483f96c9923357a45688c55ab9689f6fa5a4b 2026-11-12T08:36:10Z perf(cuda): pack B2 head inputs inner-major
0136b9193244e268a409f44d4d3226af04f14d48 2026-11-12T10:19:17Z perf(cuda): gather ragged head rows inner-major
801fb382bbeb91d2dd2701713ed81eb043a15d3d 2026-11-12T11:28:15Z perf(cuda): batch ordinary rotary spans
cffd78e4d4dde13be548bd7ecd510db5360b7130 2026-11-12T11:39:30Z docs(cuda): record explicit-position rotary rejection
f92ed9a1eec64973907ccc279fb7d9ae3ecb5f98 2026-11-12T13:05:51Z feat(cuda): retain qualified stateless row spans
c911c2c5917ef9f9aaa3a04853833ab472b2ca99 2026-11-12T14:20:39Z fix(cuda): share finite compiler diagnostics
c86e6166645c5b27658abdd22f097effa06e3c3f 2026-11-12T14:32:32Z perf(cuda): batch compatible RMS spans
884ab441036d141703176e117087a1ffe6704825 2026-11-12T14:52:53Z feat(cuda): retain stateless gate primitives
43ab4c6f4191d647f398e30e5855d10724c25545 2026-11-12T15:07:35Z feat(cuda): retain stateless QK RMS primitives
b191fc60492e5c6e54c858fe60b50c73ee8e0653 2026-11-12T17:07:57Z fix(cuda): preserve decode reduction contracts
1dce1b0d17fc2b0b15c0252550d76a8f6c8f1ba9 2026-11-13T01:40:15Z feat(cuda): retain cached GDN B8 gated primitive
c6f3c90b71242fbeec43500128c199549a53b94a 2026-11-13T01:47:59Z docs(cuda): record lane-blocked layout rejection
19296300f595588f1a2a7e29ab50fecb867690a6 2026-11-13T02:07:56Z docs(cuda): record decode projection profiling
7331ef1ec3be8ccf8f8cc15220f9dfaa87f4211a 2026-11-13T02:18:11Z docs(cuda): record decode transport and unpack rejections
0951bec114c0b317fb2f9d441f2acea57cd50477 2026-11-13T03:03:23Z docs(cuda): bound remaining decode hot path
fe3b122e1e4a5521b775b6cb5fed9d552bf9c0bb 2026-11-13T03:57:02Z test(cuda): strengthen state transport oracles
0cdfaad87b5f9e1df07946567706e1d768acab92 2026-11-13T06:36:43Z feat(cuda): retain ragged private B8 attention
d196c1b956b6055d10603e7dc5334539736eaedd 2026-11-13T07:22:08Z feat(cuda): qualify private GDN state binding
bad471387db5016eb2b752c311c6eff85b41cd20 2026-11-13T11:53:47Z test(cuda): qualify in-place private GDN state
16fba0ee49a7c14e970b3892c4d4857a11e2c1b3 2026-11-13T13:10:45Z feat(gpu): retain exact-root derived state cache
2f651b7e62819abe6c5ba6606bf67d7b56c670e7 2026-11-13T13:54:34Z fix(gpu): harden resident session root binding
60b048d47b97c60e802e64b86444d0a43cf54913 2026-11-13T14:59:26Z perf(cuda): batch canonical-boundary GDN recurrent core
775dfa6051aaf11571a8b47a253826935248cf44 2026-11-13T15:57:51Z perf(cuda): batch ragged B8 KV transitions
c12acab60c357e0742ad4ecd2f00295cd9b6bb0a 2026-11-13T17:17:08Z feat(cuda): retain grouped B8 recurrent transport
85f458f554edf9be7bac491309e1844f27e76351 2026-11-13T18:14:16Z feat(cuda): retain packed BF16 batch greedy
f303bca53286a7c96ae1ab78b95532244dd756ef 2026-11-13T18:40:01Z feat(cuda): retain exact head greedy partials
68ac8109bd47b6ace8f006ce109bac043f595893 2026-11-13T19:02:53Z feat(cuda): retain grouped ConvB4 transport
da3eae5e8f5ce2d27d167e84690db5e27a703d3d 2026-11-13T19:20:50Z feat(cuda): retain dual QK head RMS
ee347d93bc7b2f7e362b8b07b03cf0e9f9523ea9 2026-11-13T19:43:50Z feat(cuda): retain certified recurrent export
399e43a0a5d0c22927bdc28e1d35775be0393c96 2026-11-13T20:08:14Z perf(cuda): widen B8 private KV CTA
de2eed8cd1eabd059f377d5a18f269f5ef39fc8c 2026-11-13T20:44:50Z docs(cuda): record post-KV optimization probes
cb9e349946c6daceff7d58ff28df1005dfa3ece4 2026-11-13T21:55:57Z docs(cuda): record projection cache probes
6c601f9131ef6ac33c2dcd15deb37daad35432a8 2026-11-13T22:13:13Z docs: refresh comprehensive handoff
c282f20ea10c16680b033586bf11d0e5d62c0189 2026-11-13T23:11:19Z fix(hip): restore source profile compilation
874e67db0b6e737e6d9de31e44ade597b3160143 2026-11-13T23:18:11Z feat(cuda): retain shared-input projection pair
2caf304e89bc12c77308f993df267356113baac7 2026-11-14T03:09:35Z docs: refresh handoff after HIP and CUDA checkpoints
52c8a16187c7d89977c5308ebddfe094aeda5e70 2026-11-14T03:33:12Z docs: record X2 HIP physical health
134a571dc080ac8e2d0f243d3e4aac0ff4cfa5dc 2026-11-14T03:56:08Z fix(hip): isolate CUDA private KV qualification
ca6e99f9b54bf880943f72752350a3cb105c21e9 2026-11-14T04:07:41Z docs: localize 27B X2 host-load blocker
b0cf0b21e5a72a9e89c0431f8437afa2a635982e 2026-11-14T04:43:00Z fix(cache): avoid duplicate Qwen snapshot hashing
f5d2a88ffc8cb82c2a6686ad4e913768186b3b88 2026-11-14T04:45:37Z docs: record Qwen source-open localization
d68b649031ac89ddfb88f6880cb7185208b25c09 2026-11-14T05:04:48Z docs: record X2 27B physical allocation blocker
5c8f14375d4ee46afe3cc02a95d710fe813eae6f 2026-11-14T05:24:29Z docs: localize stranded X2 KFD owner
3cadc3b2ac8eadc95a8e1b19003b922e355c339d 2026-11-14T05:57:30Z docs: reject shared-pair Qwen serving consumer
5bf440d7f891111feac63f50d04a4333b40286bb 2026-11-14T06:17:11Z refactor(cuda): admit graph substrate in native builds
bc9a1b3485776240513ff8be08500881def4054e 2026-11-14T06:18:28Z docs: record production CUDA graph substrate
db7be604addc4f908efb8172eb51306ec436104f 2026-11-14T06:48:17Z feat(cuda): admit captured graph resources
cbf13841b56985c0be0642fbe8de8fbc2022e296 2026-11-14T06:49:29Z docs: record captured graph resource admission
6a5c6101a6979d4c56c20de4c2065f44bffe8463 2026-11-14T09:32:38Z fix(cuda): make nonpersistent NVRTC driver-safe
9dca4b84e8899bfff61716fe3f11ba9c6f23051f 2026-11-14T09:39:32Z docs: record physical graph resource acceptance
81b7c71324e6c3ba99d4d77bff7e4e14b2ddcf28 2026-11-14T09:52:04Z test(cuda): witness graph resource lifetime
b12e8f5395af84bda5a5c0eab960efef2399f6c7 2026-11-14T09:53:10Z docs: record graph lifetime repeatability witness
f1ce532d49c4121324eb5f5a401f85ea900f1251 2026-11-14T10:16:07Z feat(cuda): bind graph replay to resident flight
9c86c0cd8396abd023d4504c6f2a559249d331fd 2026-11-14T10:18:27Z docs: record graph resident-flight authority
bc0e2673bc45e42379a40c23bc26b0d132069b0a 2026-11-14T10:30:06Z docs: add full comprehensive continuation handoff
63ee14e440d73beedab6e2acb96838926968eed9 2026-11-14T12:47:34Z feat(cuda): own resident graph lifecycle cache
4502f5678e8cb1641848be62cd0fb5068bae1d09 2026-11-14T12:49:43Z docs: record resident graph lifecycle checkpoint
9d6613cb0bef84147a7d30453e5afb82ca83411b 2026-11-14T13:52:02Z feat(cuda): select compatible resident graph cache
bc76312a915b1f0f20f20505443d21682dd18724 2026-11-14T14:08:34Z docs: record resident graph selector checkpoint
599cb28945bee79ef70dc48ea35566c526a7f049 2026-11-14T18:19:51Z test(cuda): measure resident graph selector widths
4be26f29a4b43cfa915a89b45839e875bdfd71fd 2026-11-14T18:21:32Z docs: record graph selector width decision
b19bdb481b3bbc194eb4d021ed40112197a92e75 2026-11-14T19:12:58Z docs: record cold prefix reconstruction checkpoint
f4fb04b6c0dec311bb52affd10abfae755a01a54 2026-11-14T22:26:55Z docs: close reconstructed prefix scheduling benchmark
7dea460a80b420d0ce49a60f6c860b3502c67c38 2026-11-14T22:55:14Z feat(serving): age prefill behind decode
ba7e3768197c417e42eecd7315f6043f1d5e9464 2026-11-14T23:18:10Z docs: accept fair serving interleaving
9dab0043719e435ef3914f86f3966079deabc5ae 2026-11-15T00:12:55Z feat(cuda): add temporal fused prefill primitive
1e0f4ebfc5b9650341814ad669d4101aea717623 2026-11-15T00:56:53Z feat(serving): enable qualified temporal B2 prefill
b599db32dec2620014f2445b73d378aab8299d56 2026-11-15T01:08:50Z docs: refresh comprehensive continuation handoff
12f784083f4057dcf36fd343e408d5a7ed482589 2026-11-15T01:49:20Z test(cuda): qualify temporal B4/B8 prefill constituents
80409ac30a12c03f9ce4585068faa44aab791460 2026-11-15T01:49:37Z docs: checkpoint temporal B4/B8 qualification
330150fe23dd97be1fecac42de39b9058e2140e2 2026-11-15T01:50:37Z docs: record post-qualification GPU baseline
36253481d5e26d27ffb577187b21c338a211beaf 2026-11-15T02:06:37Z test(cuda): bound temporal qualifier host state
4d74d7f92bc4dd69b18bf947286dcb6a537755ff 2026-11-15T02:08:24Z docs: close temporal B4/B8 B1 qualification
eaec62d8a40c6179baa49e62be7a358b4b800128 2026-11-15T02:59:51Z feat(serving): enable qualified temporal B4 prefill
9347f5b838e389f68b4eb38f6d51b664b844704d 2026-11-15T03:00:34Z docs: accept temporal B4 production
8e6f14d3defc0aa2e542c26ffb9ccc7c736f1fcb 2026-11-15T03:09:51Z feat(serving): enable qualified temporal B8 prefill
fc9ba86567522b1a9171715ce520b820d7cad60c 2026-11-15T03:11:54Z docs: complete temporal serving foundation
0027c282fb28fcf846c141f032ea0e4e4fdee599 2026-11-15T03:24:04Z feat(state): add paged COW block-table primitive
fc99e9c3354500d1a8f946ad1c19d1f2e0b80d31 2026-11-15T03:24:54Z docs: checkpoint paged state COW foundation
54340a6d848ff5f19bb19333f550b1c057328881 2026-11-15T03:43:39Z feat(state): bind paged tables to resident pages
818cbb67a55518e3b44ad5c6393e85c67311b887 2026-11-15T03:44:22Z docs: accept resident paged ownership
1eb6e0f669f7bf6f31e4a0b7cdbdffa14ef73b7c 2026-11-15T05:33:43Z feat(state): materialize paged COW appends
2c908aef374d817825acd7cf4909358cfe8ec7ad 2026-11-15T05:34:39Z docs: accept paged COW materialization
3ac205991086e4d9e599993bc7960010d2f8706a 2026-11-15T05:56:22Z feat(state): add paged KV gather representation
5e6b391e8b6f417882affb54875439fc0fb19094 2026-11-15T05:58:00Z docs: accept paged KV gather boundary
dc6a188bad8d30a28d6871f0798ae98c4a6803bd 2026-11-15T07:47:05Z feat(state): add paged KV staging scatter
db3c155f52d6c2c16927208ff0010be03dd9d447 2026-11-15T07:47:59Z docs: checkpoint paged KV staging bridge
53336b3d961f2ffc59c0b845f876ff4e8e0033c9 2026-11-15T08:27:03Z feat(state): admit paged KV session domains
4e5aaade5bd83700f907d4f44e380dbbec20bbd7 2026-11-15T08:28:45Z docs: checkpoint paged KV domain publication
ef9e4d4c4364df423d4a6afad108a1c5b562d967 2026-11-15T08:48:09Z feat(cuda): add in-place paged KV staging mode
240e0f23def6d0993c9022a8c43d482cdeda58ff 2026-11-15T08:50:08Z docs: accept in-place paged KV staging
93468b1f5062804f009f817c6f5dd50bc7d0516e 2026-11-15T09:12:34Z feat(state): validate hybrid paged root semantics
4bafad794e36cb2c4b9647d8807bfc0f8818b4c4 2026-11-15T09:13:28Z docs: accept hybrid paged root semantics
cb57e1d8bd4d33b530c4efc771e19450df1ef55e 2026-11-15T09:42:21Z feat(state): construct hybrid paged state pools
9815c2abc63b81cf7ecbbc74041356d9e8a0f10e 2026-11-15T09:43:21Z docs: accept physical hybrid state pools
b9b04aafd9d10b0325ec5daac45cb3f6be8fedf0 2026-11-15T10:12:58Z feat(state): execute hybrid paged candidates
d944eae24cb0686d0d9eb21979f93984bac9c09f 2026-11-15T10:15:33Z docs: accept hybrid paged candidate execution
1413f729ac5db55131b8379752b8cdf0a7b98263 2026-11-15T11:02:06Z feat(state): add explicit paged source policy
919538ef25d338decfba1d5db54c5a2c3a5801e8 2026-11-15T11:05:53Z docs: checkpoint explicit paged source policy
398e4461fc3bd30a84e050951dbdf204b05f0cdd 2026-11-15T11:40:07Z test(state): qualify paged source lifecycle
3a1596db12d0f2e5d104ed1980f57af9ae27a513 2026-11-15T11:40:54Z docs: checkpoint paged source lifecycle qualifier
30414571c5998880d99c7f685227b83434e6c03c 2026-11-17T14:51:42Z test(state): fix paged restore qualification
30b311c1a0ca717a87d53a184b88fad5f29341fc 2026-11-17T14:53:44Z docs: accept 0.8b paged source parity
e6aac539edcf631199bb51628c7f9e361e65057f 2026-11-17T15:55:32Z test(state): keep paged lifecycle cold-only
cb35ce220edc1d69fa17d80050992f630dec997a 2026-11-17T15:58:43Z docs: accept cold 0.8b paged lifecycle
a21ef0fecdc9727bcd1e8f8dbd742975cb3b4a5a 2026-11-17T16:27:22Z docs: correct dotlink signal attribution
28890d1526538352423ff376f53042ad4636462e 2026-11-17T16:39:21Z docs: record 2b paged sandbox blocker
816e123c3fb68b2322aa3f23896a936a5d3c358f 2026-11-18T05:27:19Z docs: record unrestricted 2b qualification route
8b65a8ddfb55e20d948d700fe487ed03e94011fa 2026-11-18T16:30:38Z docs: accept 2b paged source lifecycle
c6ba1c658e77dee6315c77124ff9d15e8df284cd 2026-11-18T17:30:00Z docs: correct heartbeat harness attribution
6b920a0a91e534eae86f81e603cf419dea3b207b 2026-11-18T17:50:59Z fix(harness): reuse existing sandbox isolation
2e0f5f8cd35e1e2896bb5b87fdae77915e566907 2026-11-18T17:54:11Z docs: checkpoint native gpu harness isolation
e65f00a88f8ee92e2b175e3db96f932df59d2a0f 2026-11-18T18:26:26Z docs: create full comprehensive handoff 2026-11-18
cfe734af1dc7e478c826e70469d917812d36dd9a 2026-11-18T19:18:20Z feat(cuda): retire in-flight paged cancellation
3e8c4e71c38f23be304756734726f6ef9e9283a6 2026-11-19T03:18:25Z feat(cuda): expose explicit paged state policy
7c13aa362f10553b8858c15a9dbd8a60f8b5a28e 2026-11-19T04:01:55Z feat(cuda): add paged native cohort transport
3d4de835e1bddaaafe622634789b329b6e5d7ddd 2026-11-19T04:25:26Z test(cuda): qualify paged native sdk
4688cbae9e28838d3b21ff1674f4c48dec519cd9 2026-11-19T04:36:53Z fix(harness): recognize isolated child netns
2452eee189581d7ecb608ac5b937499222b4577d 2026-11-19T04:54:02Z fix(cuda): restore native feature boundary
ca8fa69ecd8e58fe7f3dd0720839bdf58b91ae8a 2026-11-19T04:54:31Z test(cuda): qualify paged native http
30aa1ae7903ea772f12beef8535dadc784f6de17 2026-11-19T11:05:36Z feat(cuda): fork complete prefix snapshots
8c8affa9a3cdf7f2a9b9b0eec13a0392f8809be0 2026-11-19T11:11:47Z feat(engine): add bounded prefix cache
c24246ddbc4aa285a84f93be35ccd8336119b875 2026-11-19T11:54:16Z feat(engine): enable scoped prefix reuse
6303db8ee38f1c7ab5742d17a1c7e05138a70137 2026-11-19T12:10:15Z feat(serving): scope HTTP prefix reuse
f83bcb9a7ca8e7a683b02ad9770114873709c34b 2026-11-19T12:28:19Z feat(engine): bound prefix cache retention
c0d0e69d792169340b64e7e013013911d2332086 2026-11-19T13:17:31Z test(cuda): qualify paged pool pressure
c347fda06f5c267b1ae1c7106447c9e9b66de8e8 2026-11-19T14:01:02Z feat(cuda): add owned secondary stream events
064b99bc0531e0d02b8c63b27e1fe82c670b44ce 2026-11-20T02:46:23Z feat(cuda): retain resident event-backed flights
e1080c0a17a059070089edc2eccf9468053a797a 2026-11-20T05:25:51Z feat(gpu): support bounded resident multi-flight
9147b3fbfaf37266097e14631597f56f7b98d6f0 2026-11-20T06:11:58Z test(cuda): profile native B8 serving
19c012f924856950401c706ad23286fd4a8a02b7 2026-11-20T10:18:42Z docs: correct Qwen B8 prefill diagnosis
05fe44da6af13e44e38d4ed5b3dea6570b5cb5f1 2026-11-20T10:40:17Z feat(cuda): retain stateless prefill cohort lowering
f0f06402c78d4511dea1559c73d4295f88790f93 2026-11-20T13:41:21Z feat(cuda): batch multi-token prefill attention
1222f8a11f0c773fba41a502b84ce0862d30442c 2026-11-20T15:06:56Z feat(cuda): batch multi-token GDN prefill
60ba956c3057019ad615c5b25ede3a9f7167d955 2026-11-20T15:30:39Z docs: align serving perf feedback profile
3afb9902f674d9cfe2fc4488e628b61b61cae395 2026-11-20T16:23:56Z fix(cuda): retain GDN layouts for reduced bands
7beb1be30312680720eaafc9abb10eca23648e51 2026-11-20T16:41:50Z docs: create full comprehensive handoff 2026-11-20
0dac5abfd3a8edcd82d955a6a125c1f2c5426ec7 2026-11-20T17:02:54Z perf(cuda): zero paged state on device
0680b0b8e757ebeb7003f315f08433d7c8bf299f 2026-11-20T17:28:55Z perf(cuda): batch paged KV gathers
e72030cd04f71ad7a3cd0b11d7d771875a3f9d88 2026-11-20T17:43:45Z perf(cuda): fuse paged gather tail zero
dfc3e396c9e969c48a1a55bd6d35cb9a8ced1924 2026-11-20T17:57:54Z docs: record continuous refill goodput
4ec3a0cb36f0a2affd5b092d1262952614428219 2026-11-20T18:25:56Z docs: record x2 ROCm runtime blocker
5a95da56316b9e8757175ef01a97472bc656491e 2026-11-20T23:48:38Z docs: reject mixed prefill regrouping
9122d1e0d4b6b04d0d99b6d5d7ccc083015237d5 2026-11-21T00:04:22Z feat(gpu): retain cross-model perf candidates
29a6238b6c2b59f019928672b3d169cc318111fd 2026-11-21T04:11:14Z feat(gpu): restore dormant kernel candidates
468a36054f6004eaa42d1d7dc14dc52df029af3e 2026-11-21T04:27:11Z test(cuda): qualify populated 4k 8k serving
127dae431a7db55d9f38dd305b87b94e5af88401 2026-11-21T04:59:39Z docs: record populated 16k serving
5466577d3976d2ab56471a27c614debe8ccb2425 2026-11-21T05:14:01Z test(cuda): retain current 32k context qualifier
e79814ce0c0a17a93e412b45d7dd66418e6acfd6 2026-11-21T05:49:19Z refactor(errors): preserve resource exhaustion context
603844ba0556be3491828b9495535f8c9d1dd183 2026-11-21T06:19:23Z feat(cuda): expose exact source admission inventory
973aca4adb7f2ad2dd7246ba1fc51a23c8e44826 2026-11-21T06:44:57Z docs: record persistent external gpu sigkills
f86538345fe4ce412df81864542c5ec183446460 2026-11-21T07:24:08Z docs: record 32k trace pass and external kill
f1cfd26a2f64e04f6d9ba6f7cf313b15480e47f3 2026-11-21T08:53:02Z feat(engine): reconcile exact prefix frontiers
763eb99d58c815bcf1fe22eefb710b9cf8ef6da0 2026-11-21T09:04:46Z feat(core): expose physical buffer identity
82d137ec8002af161cdcfeb3f680ba2608c1df54 2026-11-21T09:14:14Z fix(core): scope physical buffer identities
3802b9de1d7d6157165d4da76d07a0d76e03de33 2026-11-21T10:05:28Z refactor(gpu): centralize resident alias checks
e8c36fb7d5065d63b8e31a221bb6f2a7e4f75d31 2026-11-21T10:32:36Z docs: localize 2b 32k response timeout
9896a85832c3db9338cdb9e114653e1e27ae3bc2 2026-11-21T10:58:57Z perf(cuda): drop unused attention reserve
be02ade8e81ff356c9bca5cbba1796d08ec692ed 2026-11-21T11:21:05Z perf(cuda): price reachable paged state
0dd844044497ced882c59595a49dff8cdf4ffd86 2026-11-21T12:02:59Z feat(runtime): add smart model startup config
28459c231764277c141f7fb078b29dc9ee0d403a 2026-11-21T12:19:20Z fix(cuda): honor live provider headroom
3b5f191eb94b027179604d04ebac7a09b6b1050b 2026-11-21T12:37:27Z feat(runtime): smart-select native cohorts
849cb87d1f295c8acf6dcf4d8bf7638e830ed117 2026-11-21T12:49:59Z docs: close exact partial-tail COW ownership
6c68a361e843c73107a86aaedea5ca697a30258c 2026-11-21T13:08:28Z refactor(cuda): use ledger identity for prepared leases
f9acd9e49a5eaf29e02d89b2c6f31ec468871ef4 2026-11-21T13:43:28Z docs: consolidate 21 nov comprehensive handoff
b616429a41d26a4fefac209bc487dcfa9191c9f1 2026-11-21T13:44:56Z docs: make handoff checkpoint self-stable
f42d7722eb4f4f0f392c9b6e6348555162eb2f7a 2026-11-21T14:04:50Z refactor(cuda): use ledger identity for streamed head
968a3682492e2675e7837d13c767905f895fb2ae 2026-11-21T14:35:20Z refactor(hip): use ledger identity for source calls
b839920ec0553c4a27dc753c076487f0ac2c482d 2026-11-21T15:27:46Z perf(cuda): reuse pinned source staging
8c96a886cf8746ed2ad49208c2c43aaa94e4fd7c 2026-11-21T16:04:37Z perf(cuda): upload source pages from retained owners
b88d3b3728b4eea2e249f4680be9c83edfeeb034 2026-11-21T16:41:16Z refactor: clarify model and cohort terminology
f3f000055cb949f7db367b3d1bc427c472faceb0 2026-11-21T16:58:54Z refactor: name qwen source execution forward
001b4c6fbee3b70770d4065ca4cd8a124982b93b 2026-11-21T17:51:39Z refactor: clarify text invocation terminology
e8cf4d89feaa4fdb6f02ea9191620c25053be9c0 2026-11-21T18:54:28Z refactor: clarify autoregressive terminology
d2adbaf3cbc04dd41c27fd56b90222b786c25dcb 2026-11-21T19:01:21Z refactor: name complete state coverage semantically
ff099e626882201fbded10de3a2f4cbf7242497d 2026-11-21T19:13:42Z refactor: name model-wide coverage semantically
3fb731ffab12d01ed3f58ffe7e6a4c15c5e664ba 2026-11-21T19:20:17Z refactor: name fixed prefill trajectory semantically
c0cc636c10e5378174750896c61602ddc0855348 2026-11-21T19:36:35Z refactor: clarify fixed-prompt workload profiles
4e03edb9a08e9a17772bfd2074620448956c5c57 2026-11-21T19:44:47Z refactor: clarify cohort and concurrency terminology
2a87e3dfc77dd8aab894f719a8a54b89d944a96f 2026-11-21T20:10:22Z refactor: name zero-centered rms explicitly
fb90d9ee199c373d5354101079e3394cb7b52694 2026-11-21T20:40:06Z docs: record rejected cuda source-page runahead
37ec699c4ba67a5cbad4b5088b2a17902669e997 2026-11-21T20:56:11Z refactor(gpu): retain backend transfer retirement primitives
fdcec908987184a2e2248b4da60b30c22472a040 2026-11-21T21:14:02Z feat(cuda): restore reusable gemm candidates
8dfaaf05560e39e2c9916b7ec27ee3dd90db80b8 2026-11-21T21:52:15Z feat(bench): add nix-native vllm cuda provider
4201095e9bb4b9704a486a151145c31adaa7a326 2026-11-22T03:01:51Z docs: consolidate 22 nov comprehensive handoff
15f5c4fde2bc6659f47a822e4b42844a8077f21f 2026-11-22T03:19:53Z feat(bench): realize vllm provider with nix
461ec52698e1907258408dd377a229e07084da03 2026-11-22T03:45:05Z test(bench): add public cuda population benchmark
169a92a519afacb374fe2570cfc72b01544d042f 2026-11-22T03:51:52Z feat(bench): support raw completion workloads
277a27915e5392e6fe950961e2bbf94c4ab4025f 2026-11-22T04:49:14Z test(bench): add hip population benchmark
93d80f3686801bad417f19ff0372f75328046bd6 2026-11-22T07:05:54Z feat(bench): add nix-native vllm rocm provider
4d55d231cf765f4e4448c81cb8709b5f5a81df43 2026-11-22T07:18:42Z docs: record nix-native rocm smoke
c8b1bed7762b1c28262ab2254a7c87686cca4eed 2026-11-22T07:24:52Z fix(bench): provide rocm triton host compiler
6e2382fd2ab53ef23a426ce0816e8da778fec75e 2026-11-22T07:33:55Z docs: record rocm vllm c1 smoke
2a595be84bc7c5dccf7509e35e4f9b3a341662a2 2026-11-22T07:49:18Z docs: record matched rocm vllm small models
b34c025d36d0e7eaca6d559504e498f998552e19 2026-11-22T13:10:28Z test(bench): establish clean current baseline
94a7e2d02db72b3f0d172e46b14eeca7da05e33a 2026-11-22T14:45:15Z feat(hip): batch source decode projections
48d3d944dc003b181c915cc013348ca93f0f137f 2026-11-22T14:55:03Z fix(hip): size decode cohort head scratch
79de45a3ff637ffe096e18a7f202a701edd982b9 2026-11-22T15:50:27Z test(bench): observe hip decode cohorts
cb923cc2b7e79b1d40df705c78989c4bf2abed83 2026-11-22T16:00:35Z fix(hip): price decode cohort projection scratch
189c7187601115b7915ac092ca23d595607ac546 2026-11-22T16:06:15Z test(hip): accept fused cohort observations
b6c60ef3b1aad3ce828519801b748dfe8d4f15f5 2026-11-22T16:10:40Z docs: record hip decode cohort checkpoint
99dcbbee20b50b8e588f167cf2c0ec40b31782c7 2026-11-22T16:25:56Z perf(hip): batch row-wise decode operators
e35e650c6bebeeb814c770f8aee1a031a819415a 2026-11-22T16:28:18Z fix(hip): compare decode cohort semantics structurally
537e5e8542f6d07588dddcbf52a5131f1fc5a6e3 2026-11-22T16:31:31Z docs: advance hip decode cohort baseline
9ee877c7ba87ad233228ef71a2803ed4063996ec 2026-11-22T16:39:02Z perf(hip): batch aligned decode rotary prep
a5ec4c31eb39d480bee499bda6b7757fbf9b4d74 2026-11-22T16:41:48Z fix(hip): seal batched decode rotary variants
a3908378fd217124fdf1581763002df190133b75 2026-11-22T16:44:56Z docs: advance hip aligned decode baseline
1de99ef6210941a662bc0fc5155092e7f366fddc 2026-11-22T16:53:57Z perf(hip): batch independent decode conv state
c33016d649e850d33b13e9f5cdce254e35924353 2026-11-22T16:56:59Z fix(hip): preserve conv coefficient input contract
57cc384c4ff55f5576ccc9a321fe9600cf70ce24 2026-11-22T17:01:24Z docs: record stateful hip conv cohort
491cf9957d83bc46100cc863b0b062909459fbd6 2026-11-22T18:48:50Z feat: unify production GPU build and run paths
6f8f07111fb38eb910c051063cefe76f8c1b75bf 2026-11-22T19:09:38Z refactor(gpu): separate attention semantics from providers
64c06e2fc313d3b24c0a5abe0065a085879f0e90 2026-11-23T00:41:27Z feat(cuda): add opt-in FlashAttention provider
ed0382a192ac882be0fb8e2346e23a4bab252284 2026-11-23T02:00:40Z perf(cuda): keep FlashAttention off decode path
0a35f175629f0841b98d78fc156c0c30eaf88df3 2026-11-23T02:13:58Z perf(cuda): fuse qk normalization and rotary prefill
2805c37577a185d7065fcb9397f57827d1010d90 2026-11-23T02:20:17Z perf(cuda): enable fused qk rotary prefill
179bb1528339c258b7cb125086ac6fc692698d18 2026-11-23T03:01:57Z perf(cuda): fuse GDN recurrent export certification
c829aa9855f3eebfc0a6635122d6448c781d27e5 2026-11-23T08:27:11Z feat(cuda): retain B1 direct GDN conv primitive
f3f872cb567a6885918ae2b4866ee6d535d736d8 2026-11-23T08:44:57Z feat(cuda): retain B1 fused head greedy primitives
fb20345bb7b5ce0b3285470c77564e492331254b 2026-11-23T09:48:25Z docs: consolidate 23 nov comprehensive handoff
0c4d95e6a1fce0e162927ef2b55647fac7d1fba1 2026-11-23T10:35:52Z docs: revalidate production feature convergence
6f4418ba8262baff0dc023923865ea8bcfb7d63c 2026-11-23T15:25:54Z fix(qwen): allow tokenizer-owned chat prompts
2d16cd5aab95949cd295070a8a5df84172409f29 2026-11-23T15:28:47Z docs: accept a1 hip public canary
11f692bf7b2871e1dac974eb96bb27f3968ee316 2026-11-23T15:48:53Z perf(hip): batch recurrent GDN decode state
7fcc9c4739b9bceedb298d4835a55ad5b64c1a84 2026-11-23T15:56:11Z perf(hip): retain recurrent cohort primitives dormant
31eaf496385890005f04a7c920ea5b9ee367404d 2026-11-23T15:59:38Z docs: reject hip recurrent cohort selector
69630ec3426d2ca523cdbe5e2183b01f68b93f49 2026-11-23T16:10:31Z perf(hip): avoid dense recurrent decay scratch
0d201ff0bac05aa92d48dc8faa4e4cbfa2cce7d5 2026-11-23T16:33:50Z perf(hip): fuse cached GDN coefficient exp
879bee798a29f5da5a0fa4d2812840604f0bad76 2026-11-23T20:12:58Z perf(hip): keep singleton recurrent primitives dormant
4070610433a3068943e844b2135d9776beffaa47 2026-11-23T20:27:47Z perf(hip): write singleton projections directly
fdc5a8ec86aa9861576fbc33bfbfb4a4cc4c2ad4 2026-11-23T20:36:10Z perf(hip): reuse packed singleton projection slots
baf4c9f090e4106cc4a35439a30854328adfaea3 2026-11-23T20:39:17Z perf(hip): keep projection alias contract narrow
ec6fb57f08b948c804f869787a5a17f52e038846 2026-11-23T20:43:39Z perf(hip): coalesce singleton projection status
d15b02f6973e36baedb4b6a374058b48e2e76795 2026-11-23T20:51:33Z perf(hip): retain projection status coalescing dormant
e606586dd3cd7ef76c2ec832bb5cb76c3b9c87af 2026-11-23T20:53:29Z docs: record hip singleton projection A/B
e1701e707de5b23589e24509b5b401cc1f138720 2026-11-24T00:43:51Z perf(hip): reuse singleton projection input pack
1d49ec7c4760e0f00c9af2f6d0c7babf250a2530 2026-11-24T00:47:46Z perf(hip): retain prepacked projection path dormant
5d51896532574a08d22912b44da127a6564e3cec 2026-11-24T00:50:48Z perf(hip): certify singleton projection in-kernel
90537b6c2da8f626e08e7665311d49e249c4bad6 2026-11-24T00:58:05Z perf(hip): retain certified projection primitive dormant
9e91a0d26a7b77456411b15074e77098f7b754c3 2026-11-24T01:05:40Z docs: record hip projection and native accounting
c5b3fadd214d9d89a84e6c6d7da8fc756bf5fc76 2026-11-24T01:29:29Z perf(hip): qualify gfx1150 native module allowance
fd4611e12fc1c2f173b2d80d4e8d2105fe601327 2026-11-24T01:37:07Z docs: accept gfx1150 resident hip policy
ffd5ff8f951c5db9b113865c6d5a22a954e282e7 2026-11-24T01:56:10Z perf(hip): fuse residual rms first pass
dfc57183f8fe36f00e065e9b7c1482b6480488ec 2026-11-24T02:03:19Z docs: record resident hip hot path
b44a55c868e72e04f68d7e900ee32b822cc89c32 2026-11-24T02:09:10Z perf(hip): fuse rms scalar cascade
11e61f850557c4303a95c2686a8afb6af5de3591 2026-11-24T02:15:26Z perf(hip): retain rms scalar cascade dormant
9040492544f0ea6e114fd749232cc2b185aa8a65 2026-11-24T02:20:20Z perf(hip): fuse rms vector tail
654adbb17a35d66922ba918477f9b872b704b605 2026-11-24T02:25:24Z perf(hip): retain rms vector tail dormant
288e978666463148475900b9db167f41ee5af99d 2026-11-24T02:26:15Z docs: record resident rms fusion A/B
7ff8784c8d454af634a143d30d228e8164d78527 2026-11-24T02:51:20Z perf(hip): reuse resident singleton projection pack
36c318db569916ac8bf264e33b4d9dc25dfd08dd 2026-11-24T02:55:01Z perf(hip): keep resident projection packing selected
1b25fc30cab76b93e1bb02f3b09e1f6b20bc5ef6 2026-11-24T03:02:07Z perf(hip): share singleton projection input across four rows
5e4cac97fa78eefb1f4efc721bfee1ae0c3317a3 2026-11-24T03:06:20Z perf(hip): retain four-row projection primitive dormant
a5c57c13308cf86b2daa0c2cce4e463f1fb4c447 2026-11-24T03:14:31Z perf(hip): parallelize cached gdn qk normalization
072b79128163e2c18a0bfa946099875b13fd9122 2026-11-24T03:30:33Z perf(hip): retain exact wave32 projection primitive
b4fc23188a10583588a610d3a6657a37fee7cde6 2026-11-24T03:39:52Z perf(hip): keep wave32 projection primitive dormant
d6043af75b9f0e557afee142176df3f3454b16e0 2026-11-24T03:39:58Z docs: record resident projection device profile
253a565b543447833bd113309098f46ee113155e 2026-11-24T03:54:46Z perf(hip): retain exact rows2 projection primitive
6629ee26a9179ac2b3ec62a1e6ef5996cc6e6b04 2026-11-24T04:05:42Z perf(hip): keep rows2 projection primitive dormant
a8c92a2984c6f1de04fe901e367b174c1319eea3 2026-11-24T04:05:55Z docs: record rows2 projection A/B
6d5dc72d3711cd3b0e14db0a3c007851984a0745 2026-11-24T04:37:53Z feat(cuda): integrate flashinfer decode and paired gdn state
55477b4b8b6e52db3b11450cb8fdc68fffbb5cbb 2026-11-24T05:09:21Z perf(hip): retain paired packed-load projection primitive
ba8c5ea7ac6fb8575f102607cdefd0155de510da 2026-11-24T05:17:53Z perf(hip): retain input-certified projection primitives
49e5149887a6687733b7dfb9420eb8f5575043d6 2026-11-24T05:28:26Z perf(hip): keep projection traversal alternates dormant
e1929762a87b1e34de099d9f40b85bcff385c27d 2026-11-24T05:28:32Z docs: record projection traversal A/B
f6bd058c87ceb4eb5fbf7d995fcb07a84972241d 2026-11-24T05:47:23Z docs: add 24 nov comprehensive handoff
d717c7b006c0be9d189773c073d7be27b2a8bcc6 2026-11-24T05:58:27Z perf(gpu): retain exact packed-index projection sibling
4c2165fe43bae068338a04099f737474d17b2218 2026-11-24T06:03:27Z test(hip): stage temporary resident projection A/B
224085b1f3b24d5abcc029d256e32b67de391e68 2026-11-24T06:06:53Z test(hip): seal selected projection A/B kernel
3be8a812bed826b747d50d35ad516facac78e38b 2026-11-24T06:12:14Z perf(hip): select sealed resident packed-index traversal
b0522cece95599a03a9da4e75712b014cbcf951d 2026-11-24T06:14:42Z docs: record HIP projection canary and combined gate blocker
c89d0ee73696e81c4f3391e471ed4e9a9d7cdeae 2026-11-24T06:31:12Z fix(gpu): restore combined backend feature boundaries
fd02260c24af860f0c4ce64ac5953c130fcaf923 2026-11-24T06:31:40Z test(hip): stage resident real-model kernel localization
43c9ab8cfe10e21b71b90899c1af11c915947c16 2026-11-24T06:35:52Z docs: record repaired feature boundaries and resident HIP profile
7d0b395b3af245dd375f6b49a6c8c345016726d6 2026-11-24T06:53:13Z perf(hip): retain exact parallel GDN key subtotals
7d3a130bb7828ecd09908cd70bac7d7eedf50369 2026-11-24T06:55:51Z test(hip): stage sealed parallel GDN reduction A/B
e81a79ec9d95257c757913ef0b706d9afc1c2605 2026-11-24T07:00:42Z perf(hip): retain parallel key reduction without selector
6f2ededf523c74ea190b8a3f25993cea4f6695c6 2026-11-24T07:06:21Z perf(gpu): retain exact two-barrier projection reduction
971b58e756860e522d956a134091ba77c79869c0 2026-11-24T07:08:44Z perf(gpu): retain two-barrier projection sibling without selection
42d4703d38135d53d55f23de7ee7193eb112552f 2026-11-24T07:38:33Z test(hip): verify native set refusal by ownership
11a44d6034a33b1041643b5e142a3a6059b355a5 2026-11-24T07:39:59Z test(hip): sample physical native preparation allocation
38f20ca39039f934614a41f18f6fc486f8818cfe 2026-11-24T07:47:39Z perf(hip): qualify gfx1151 native module allowance
38978663d116733f40f3d33176eaa0fb1e78c34c 2026-11-24T07:54:38Z docs: record X2 resident native qualification and baseline
533c72cc47a5791e54c6748dae6a7836c6dca515 2026-11-24T08:01:31Z test(hip): stage X2 prefill and decode device localization
f7ebf647b23e5a96d2d638a270f05dd922fa74b2 2026-11-24T08:04:06Z perf(gpu): retain exact request-tiled projection
c1cd04f3339e08c2422afad972fae5f0a38a2dd1 2026-11-24T08:07:09Z test(hip): stage gfx1151 hoisted projection model A/B
99eb79905106d39c2e46d819918159a24a556cd4 2026-11-24T08:13:15Z perf(hip): select hoisted projection for qualified gfx1151
7d3e4b4bb57e92f0a85758af221f57e6f324e217 2026-11-24T08:17:56Z docs: record X2 hoisted projection and prefill localization
393d0f56fd23669e84e6adf521921af133096393 2026-11-24T08:26:31Z perf(gpu): retain exact guarded triangular arithmetic
cd86a9655cc3a9f2be58ac16ff81a41e7c3127b3 2026-11-24T08:28:39Z test(hip): stage immutable guarded GDN arithmetic A/B
4a5961dadf11e12188214d880f3f9dbd09414f58 2026-11-24T08:34:39Z perf(hip): select guarded GDN triangular on qualified gfx1151
720f467d82420cd9ceb0db9600fadce4f2a9b14a 2026-11-24T08:39:13Z docs: record guarded GDN triangular qualification and baseline
2ed9c10f2443d8727e293e51c4c8180cf97d8eda 2026-11-24T09:23:59Z feat(hip): share projections across aligned prefill cohorts
282a3437ea822c58b433edf896485bc5becd63cb 2026-11-24T09:27:05Z perf(hip): share stateless prefill work and direct projection outputs
2eee25fab1d55a365c065e7fbf314d51624b9870 2026-11-24T09:28:59Z test(hip): validate distinct prefill roots and readout against singleton
47b9f6f65a6e9846ef0ed73560ee467c1316b8b8 2026-11-24T09:33:20Z test(hip): use admitted context for prefill differential
fd914f63a546f6e69bd8628a39de19d7a271fb8f 2026-11-24T09:35:07Z perf(hip): bound resident cohort projection token tiles
74619a0b31c2aa09e416614bcd42bf5332edefa8 2026-11-24T09:36:59Z test(hip): exercise shared state-only prefill through public scheduler
909cf1e611c2605a10d288b9e334b1a4b95105fc 2026-11-24T09:40:50Z docs: checkpoint aligned HIP prefill and resident tile results
e80216b02a121ac1306eaafab2a09717b2deb52a 2026-11-24T09:48:31Z feat(hip): add native BF16 WMMA projection primitive
2514868a6ee5f08ef1e9ae15699abf2ac974e7d8 2026-11-24T09:49:42Z test(hip): check native WMMA mapping with varied integer rows
7046e618fbb3b0e3d922aa0226cc1767d6dcd3b0 2026-11-24T09:50:59Z test(hip): localize WMMA fragment output mapping
591913f545753e2245a210754ebbbedac647552e 2026-11-24T09:51:58Z fix(hip): materialize WMMA vector element before bit conversion
7b37d338aff3cba28b9cc6cbb6fdda16bd5bac33 2026-11-24T10:02:14Z Add owned HIP rocBLAS GEMM provider and lifecycle qualifier
9b9e49968058297044b2173c46b80137578b006d 2026-11-24T10:03:37Z Use optional device duration in rocBLAS qualifier
e1358b10b0c048b401fed4c46ebc35759cfa488b 2026-11-24T10:07:29Z Prioritize native HIP matrix execution over scalar microtuning
312bbccd467c3181aaf5ad451b9e3b08232813e4 2026-11-24T10:15:46Z feat(hip): integrate native BF16 GEMM into resident source execution
58a10d3c9a50be82ab0df9a7691239e05d070124 2026-11-24T10:20:11Z feat(hip): expose native CLI profile and use full cohort GEMM
2d3e8c5f6ea61bbae571dc46fda91f1410de6c2e 2026-11-24T10:20:49Z test(hip): stage native projection whole-model localization
6c31363d42fc7de386f223f2e39869b36389c911 2026-11-24T10:29:20Z Use native strided batched GEMM for dense GDN chunk products
6a4f2258db9e05442ca161ebbfe98f0b54bf743a 2026-11-24T10:37:04Z Record native model execution baseline and provider priorities
e1901b16ee7227639be69b7bd41498f2f2b53290 2026-11-24T10:45:56Z feat(dense): share native matrix commands and batched triangular solve
79a9073a246a91739fdedbb9c2b91107aa894335 2026-11-24T10:47:32Z refactor(cuda): lower owned BLAS through shared dense geometry
9a571088c8ca696a7514299d09dc70c4a767414e 2026-11-24T10:49:23Z fix(hip): validate dense read-write operands before claiming writes
aabc3d4f132750d4943e90b2379fee9a25c965c7 2026-11-24T10:56:16Z docs: checkpoint shared dense provider and native solve
6f677aa347c3595f5d52cabd66f6c2305c782645 2026-11-24T11:18:00Z docs: consolidate foundations-first native execution roadmap
83f7419dc3d33f93b12e64847e33aa010d555799 2026-11-24T11:18:39Z docs: normalize implementation plan header
668a3050d5205997f17332af4b2c46cd39908aff 2026-11-24T11:37:13Z docs: match architecture diagram to dark interface style
8189cebbbbf49874c8318bff8be94666a22c599a 2026-11-24T11:43:33Z docs: describe final architecture independently of roadmap status
48fbbb0d60a250b40835d630ed1a44326f68f384 2026-11-24T11:46:34Z docs: unify native region lowering and optional graph import contracts
8c34c4725679af32fbda08b7f7dbdf61cfb0a8b3 2026-11-24T11:53:45Z refactor(fabric): prepare borrowed and window-backed source programs uniformly
e0ef64cc3fd42d77fd1dde7db32a82eda71505ae 2026-11-24T12:15:57Z test(fabric): complete source preparation validation checkpoint
17e2b7176eaec5b9a5b78aded583690795e81c30 2026-11-24T12:46:39Z fix(fabric): close source preparation acceptance gates
0094952d10bbe43d9dbb589b65a19c10c9b66c06 2026-11-24T13:33:42Z docs: comprehensive foundations takeover and historical work index
82f895dae795f2bcabf83a95ce52cb6705e64280 2026-11-24T14:01:10Z feat: bind closed stateless regions to original model sources
1ec6b94140a13ea649cc2c8b9bbda112293a2151 2026-11-24T14:14:17Z docs: record closed-region foundation validation
0db60ede170634d6be5f9ffecd82939c2745ab95 2026-11-24T14:31:58Z feat: prepare typed physical region ports and original lease views
053a7c5780d2e1e2f7dd04407ac07520b16454d4 2026-11-24T14:46:14Z docs: accept prepared physical region bindings
77e8b4ae0487a5219140727007cc316f43c01663 2026-11-24T14:54:42Z feat: bind provider resource requirements to original flights
c76708c7b827097f3494fb2fc6fd5b20d2e9a1b6 2026-11-24T15:07:54Z docs: record provider resource and completion validation
262712f1c94daa8423060155633545b632a54e88 2026-11-24T15:23:07Z feat: bind source state regions to original root attempts
6c94694e86b9e01f33e452dea8045279c8492032 2026-11-24T15:38:49Z docs: record source state region validation
6eae81c68c24c58c85e707baebacbccbda523208 2026-11-24T15:55:26Z feat(cuda): bridge private provider transfers and stream completion
32ed4786671ae360729651373baf743808c9c786 2026-11-24T16:06:15Z docs: record private provider bridge validation
7e86aad7d74882baa474a50a584e141f1d58eee3 2026-11-24T16:21:00Z feat: bind prepared text calls and producer observations
d8a046fe7e1218532db33a425fcb6cdd83d645d7 2026-11-24T16:33:33Z docs: accept prepared text execution checkpoint
f4746c2350153b518e4730c3448e8f4ede02eab8 2026-11-24T17:15:17Z Bind cold native source admission to selected device and actual fit
f74e599788c313eb1fccdc38d00719c2f754e91b 2026-11-24T17:27:07Z Document accepted native cold admission checkpoint
057683d1c315fa026e0d4787b073757ca8caf699 2026-11-24T17:44:16Z Unify admitted local model sources and retire sole catalog owners
139bdbdf667de6eead31dfe02acb61e60a725e13 2026-11-24T17:46:36Z Report GGUF canary worker errors without Debug requirement
9e5f678bf7bd808b1ff0c866d4bd379e2cd1476d 2026-11-24T17:52:35Z Document catalog retirement and shared local source acceptance
2b6656b3d427092dea5bcdbb801bc71e59ba000c 2026-11-24T23:27:44Z Retire independent model owners without stopping shared workers
4d6645c8dd311c20b925fb2307e9029127a3724c 2026-11-24T23:35:36Z Document accepted selective model retirement checkpoint
bffad06a9c11efd497b659cff7b735c00e08e718 2026-11-24T23:47:32Z Expose typed loaded model and runtime resource snapshots
94b205b93681cf340ea719af25a519fbbf62d143 2026-11-25T00:01:10Z Document accepted typed runtime observation checkpoint
4477f20ce7de60d722eb6967e1f33b5e565cb154 2026-11-25T00:10:26Z Bind serial language pipelines to original loaded model registrations
6bd6a04d4f367b8c29b8d3ff4953c2bae20eabc7 2026-11-25T00:22:41Z Document accepted source-bound language pipeline checkpoint
2df8cbb3f11ba698278b3537fc4fd16ca52908b5 2026-11-25T00:33:48Z Bind prefix cache to original prepared resident state
4306560392b6c2e82b2ddb83519a7ef8b8429df3 2026-11-25T00:46:42Z Document accepted resident state identity checkpoint
7b36a8795588b41d01634f823965427fb453bc15 2026-11-25T01:00:16Z Share authenticated resident and streamed source window transport
0342e1038749dc802cebe7159962dc7bb1967d07 2026-11-25T01:03:02Z Select source-page transport from original device in CUDA-only builds
fdfb467f7ec86ddb891f78f562698675875a3565 2026-11-25T01:07:05Z Compare source transport with fresh admitted CUDA contexts
3cd320f57f9274827a528c3dc708f7c02444f718 2026-11-25T01:17:08Z Document accepted shared source-window transport checkpoint
48d6a9d334a77733faa6e7da15f12227896aacf8 2026-11-25T01:35:12Z docs: consolidate accepted foundations and complete takeover history
687d9a2467af9479108b6069c12e68545032630b 2026-11-25T01:51:45Z refactor(source): retire parameter backing at physical initialization
06df027fd126a57368984c668dc7e277d3f442bd 2026-11-25T01:58:09Z fix(source): preserve padded parameter views with shared transfer backing
43928332ac906be64372ba9b3c62ade2bb32ac60 2026-11-25T01:59:36Z test(source): observe retirement after consuming model shutdown
f4e09428cd8d1f77302a24e18050687fa8fba6e6 2026-11-25T02:05:42Z test(gpu): assert typed physical admission refusal
1664751483a047c8986dd39bed85dcfb13670722 2026-11-25T02:08:41Z docs: record parameter materialization and retirement foundation checkpoint
3ca16abd538fe8a4ffa0fc07b472865756392bf2 2026-11-25T02:20:53Z feat(provider): bind original source regions and explicit compiled selection
a53a3446dc271c1081ea0e850a851907435e7c74 2026-11-25T02:24:44Z feat(sdk): expose direct region source binding types
2a63b875388661aeb8a289453a287463783e9e3d 2026-11-25T02:34:42Z docs: record compiled-region foundation checkpoint
8eb17bc2125ed73260240db9cda47312c501ac39 2026-11-25T02:57:22Z refactor(source): separate authenticated model eligibility from execution selection
00f57e7a3d75bd83a980ed333211a45bfbc1f7e5 2026-11-25T03:02:22Z test(source): expect typed host capacity refusal in real cache comparison
9374e0ca2daeb8537dc96f6e1ebbe2271fbadf99 2026-11-25T03:03:04Z test(source): identify private cache budget as host memory
eb32ebaefe7ab8c6ceabf463c1d6ec16c8612dbf 2026-11-25T03:08:00Z docs: checkpoint authenticated source eligibility foundations
9ae6dd1eb3c7bdd8c8ffb30479bd7775686a9c2c 2026-11-25T03:23:46Z feat(gpu): admit CUDA physical envelopes before client bootstrap
e06213a517e1a88732fc5ceadac8fd1b9a97b354 2026-11-25T03:27:02Z feat(gpu): expose the admitted physical device identity
f31ba4db038e565f39958155c288d19797d3d043 2026-11-25T03:31:18Z docs: checkpoint CUDA physical pre-admission foundations
cbd2c1198a49ad800945b33ed01ba45ba15dafcc 2026-11-25T04:01:08Z feat(resources): share host backing and intersect cold local memory fit
197c2e26f58e27064d9b574ae8050ac41ab5919f 2026-11-25T04:03:28Z fix(resources): check actual HIP cold reservation rather than lifetime upper bound
814d5c006a56b68890fd2f6a3f6b564056723e7c 2026-11-25T04:14:20Z docs: checkpoint shared host backing and local VRAM foundations
a654886341e6ae3831ab28675fbf8daddbbf605f 2026-11-25T05:27:02Z feat(gpu): preadmit portable HIP and guarded Vulkan physical owners
86a3792a2ed20fe73590f189568e393d55adf18f 2026-11-25T05:41:07Z docs: checkpoint portable physical admission and placement foundations
1d5c4ada3e224e7278ddd487563073e2d8d7be94 2026-11-25T06:13:18Z feat(state): persist admitted committed CPU checkpoints
f9dd5ed654d77548a51e1460ae668aa45d330c22 2026-11-25T06:15:33Z fix(state): gate diagnostic filesystem test import on CPU
35629e6cddc2db727cf181dbbeb899738f77e4b5 2026-11-25T06:19:28Z fix(state): restore typed diagnostic resident checkpoint compatibility
566dcf8dd41716f33da1690fe302a2b857d37f30 2026-11-25T06:24:06Z docs: checkpoint committed CPU state persistence foundation
804d488200ea12b82e487f9e590f90f0d3055672 2026-11-25T06:34:41Z feat(state): persist committed native HIP resident roots
d9b252057ef76cc56bb0ecb37f9fc653a24061f4 2026-11-25T06:38:15Z test(state): qualify unknown resident checkpoint upload
21ce3d7f024ca62d71edb3bb213a46a96fcc60b2 2026-11-25T06:42:10Z test(state): include resident persistence in existing HIP compile gate
ef7c3d6c0997f10185aeff5b0b0ae56c6a74dd56 2026-11-25T06:42:57Z fix(state): retain diagnostic admission through shared publication
de1ef6766d472b13ee4c3236a78898105a6db0bc 2026-11-25T06:50:29Z docs: checkpoint native HIP committed state persistence
ebda04c43d6b3f619cad01500edde1869eae8673 2026-11-25T07:05:35Z feat(pipeline): atomically load source-declared language chains
212f6eabd6e346a9466b2dd2be1b261c5e4cbda0 2026-11-25T07:16:21Z docs: checkpoint source-declared language preparation
b25b5c02347e15980c082719d18a2685e687af3c 2026-11-25T07:33:52Z feat(sdk): unify text task controls and selected owner observations
9a5fc6e658e716bda0e585fe2080c1623f49b404 2026-11-25T07:41:32Z docs: checkpoint shared SDK task controls and review foundation exit
7987f70a566bfb7a5519b8f1a1013f138881de07 2026-11-25T07:57:27Z feat(source): share selected GGUF inputs across SDK and direct preparation
2ec06026853b019876b753f50e38134ea202bee7 2026-11-25T08:03:49Z fix(source): preserve original selector refusal diagnostics
24bba9764b4ff7bf92d6239efc7dba01a8bbcf3f 2026-11-25T08:09:34Z docs: checkpoint shared selected GGUF source preparation
8e3b8993b189beb80216e8b5a10fd8a5e484f7db 2026-11-25T08:26:53Z feat(provider): observe original prepared regions and preflight CUDA transport
33118f162eca2618d97ff2b7fe6885ecae6da69f 2026-11-25T08:36:16Z fix(provider): preserve GEMM orientation and require dense bridge copies
1462687e59b19f29d4e193fddd6c58f420a9e620 2026-11-25T08:41:04Z docs: checkpoint prepared-region observations and provider compatibility
3b5a520d748bd5e353baea00dcaacd30e83d6793 2026-11-25T09:33:25Z feat(vulkan): adopt exact registered clients under original service registry
fc38db0371804a46d189a23fb83f531f3f240651 2026-11-25T09:36:25Z fix(vulkan): require explicit backend for named registry reuse
e1decff57e98b55ee0cb09697015c8e8d998cba4 2026-11-25T09:38:46Z fix(vulkan): expose original client to Vulkan-only qualification
9017af172f700456beff29ca379763fd7546f8e9 2026-11-25T09:51:00Z docs: checkpoint original named Vulkan service reuse
ce83cc18574ab3998d7a1dec84aba6cb300b5fac 2026-11-25T09:59:44Z feat(provider): observe actual captured owner and prepared cohort mode
0886f11f2f66aa5d330184bf443a198b1b930d8a 2026-11-25T10:08:38Z docs: checkpoint actual captured execution observations
14bbbd489bae8cfa79beae0ceffc07d7a0f8817c 2026-11-25T10:34:12Z feat(backing): initialize exact coherent mapped resident views
5d952364f4a2d2ce377644e1255c760f60b05aee 2026-11-25T10:34:40Z fix(backing): keep mapped probe in test qualification
5a7535e75b727f4b6699379b698c507b2dbdb0c5 2026-11-25T10:37:32Z fix(backing): retain original logical upload accounting
d5030ad4ce476b913a46056d492992ccf1b8edf7 2026-11-25T10:42:12Z docs: accept bounded foundations and checkpoint mapped backing
450cea5530dad567ff7706c30e1a1872ff01bda7 2026-11-25T11:13:19Z docs: consolidate complete Phase 1 exit and Phase 2 takeover
e6211608e01980bd7e3dd6de8bbac8fb964805a5 2026-11-25T12:17:01Z feat(cli): unify presentation and diagnostic reporting
bcdb46a8dfb582a95bee7432a2861c5186e50983 2026-11-25T12:49:09Z feat(cli): add typed command tree and shared model resolution
3cf1bb2df27effc1707bd87a836a4648f70e0a6e 2026-11-25T13:22:13Z feat(sdk): share control-plane snapshots and actions
da2385c73ba83f5b130a6fd042d185ddecbb9b63 2026-11-25T13:58:47Z feat(acquisition): share controllable download lifecycle
8e382947e2d4e78269ec4847a492f0dac6d81ebb 2026-11-25T14:35:45Z feat(cli): simplify one-line model execution
84e0681cd07f08173783e61716f2850feed09f31 2026-11-25T15:15:20Z feat(cli): unify canonical model serving and listener lifecycle
7f5fc4ebccc7ab6378d8523cd912387a74ad39f9 2026-11-25T15:45:33Z feat(config): unify typed profiles and atomic configuration edits
abe98b8379546f1e07dbd8dc97d4db28c9345e4a 2026-11-25T16:01:13Z docs: accept unified configuration and profile management
08e7e9d3368a75a76031ddb78e7789fb266ba216 2026-11-25T16:19:49Z feat(tui): add native read-only control shell
0c7ad87a1a794c92e4382fbb1e4b9f103cd30b7b 2026-11-25T16:57:46Z feat(tui): wire resident model and listener actions
9bf7829ae4f01671d6e78e8b50e23da242714ca6 2026-11-25T17:25:08Z feat(tui): wire model acquisition and device management
71e369333f19eee230a1f2a674710c3f8aaf2b9e 2026-11-25T17:56:41Z feat(tui): complete native configuration and operational logs
4e472aed38bac2d0139fa4e5162f483d45574ae2 2026-11-25T18:39:00Z feat(web): embed authenticated shared-runtime control shell
4f23983a5bb33a3ab2e9ef124342a0cc33df2088 2026-11-26T01:00:17Z feat(web): share resident model and listener actions
3333ec5cd462e4520ffc1901a3881785478c86ca 2026-11-26T01:10:11Z docs: consolidate comprehensive takeover through accepted UX-9B
a6744b125ea415ae517a3273f00a5dcc5ffd0e37 2026-11-26T01:40:43Z feat(ux): complete shared Web model download and device management
543194ac851a8db6118eb42e3aea209403edadbc 2026-11-26T02:18:06Z feat(ux): share Web profile configuration and operational logs
1a87899013cac5afd6839f748ee33627a30e3b46 2026-11-26T02:42:46Z test(ux): qualify simultaneous native and Web control lifecycle
22c6c0a73860af4f73188d4b6093c932a33f258d 2026-11-26T03:07:20Z feat(cli): add cold diagnostics and typed pipeline inspection
1b8de1cb87fa4815655cbb6700e6633946f93305 2026-11-26T03:42:40Z feat(cli): complete cold quota and backend diagnostics
3b584f729efc691b8149805785d5c00f81d4c994 2026-11-26T03:54:25Z docs(ux): plan native TUI interaction and visual polish
b1073e7c53e5fbeef1c9f346264e7669e02e93e2 2026-11-26T04:18:10Z feat(ux): establish semantic native and Web appearance
ab3cf974e4df7538384b873af121a8a3747f9334 2026-11-26T04:42:11Z feat(ux): add native mouse navigation and resizable panes
5da962513b051b5296c28e107a321ffb0cbd6afc 2026-11-26T05:01:47Z feat(ux): expose bound native actions and modal controls
8a2c2ae88fbda55790c27249c6782bf852c29544 2026-11-26T05:25:08Z feat(ux): add typed pickers and native grapheme editing
f723b0b6585dd2a0749895702c123a7d0a74db1e 2026-11-26T05:53:06Z refactor(ux): simplify native chrome and disclose details on demand
d1e27080776bfff8bf593bcc9c503df43ff92fa2 2026-11-26T06:37:44Z refactor(ux): replace native toolbar with contextual menus and live search
0ee66351893b707de1687511729cd1ac440655d8 2026-11-26T07:08:36Z feat(ux): preserve scroll anchors and refine response and compact dialogs
c0e4794d9517f03d989852e945d0b8db7e978448 2026-11-26T08:11:31Z feat(cli): enable full production defaults and preserve safe error diagnostics
e96cdab98d47c5c84001c1afaaebd8f126b9e0b3 2026-11-26T08:48:25Z feat(ux): show live registration performance and layer execution details
3df319127b19616dd3e9d4643d63c7ef9cd1a448 2026-11-26T09:09:00Z docs: preserve comprehensive UX-P-M1 takeover and P4B continuation
53cc7ec3e8ef316de1cbc21121f4db069f57b8d4 2026-11-26T09:30:08Z fix(ux): qualify production defaults and live owner acceptance
5c919c135a542d0edacfe447b7cd5d312b147301 2026-11-26T09:40:47Z docs(ux): record P4B isolated serving gate blockers
490fe410299f0c5abf1464e52e0dcf41558ce18d 2026-11-26T10:02:23Z feat(ux): refine live model dashboard and exact layer placement
ad7a4178e3b85ea9e8a0ddcf9f2be798b0416ca8 2026-11-26T10:19:00Z fix(tui): scope acquisition-only formatting import
dc9b1956038676ea8a2ed7da119dae0f65caf1bb 2026-11-26T11:16:19Z docs(ux): accept assembled P4B gate and unblock public serving
240bd05e1e0b51a96616c4c08ef311ef53e53772 2026-11-26T11:39:54Z feat(ux): choose model backend and device from Models
fbb4ab1dd72c4fa4e52b3eaa165d7ca2211ec114 2026-11-26T14:48:29Z feat(serving): own bounded asynchronous HTTP transport
3156ce45750e838c7ace72dc7d8c0c06aa762c3b 2026-11-26T15:07:52Z feat(serving): own bounded ingress and websocket continuations
08a9d30ec5836d6885c937cd60dc0b33471329af 2026-11-26T15:44:57Z feat(serving): admit strictly framed chunked text requests
715127ccfd5568fe1fd35663784c64410cad05c4 2026-11-26T16:31:25Z docs(serving): accept strict chunked text workspace gate
3f8329f2aaad530950a355d63eea5743f4143705 2026-11-26T17:02:06Z feat(serving): own bounded multipart form ingestion
a76ddbfeaf708be9046ca14980dc1410d8f20cde 2026-11-26T19:50:18Z Unify semantic text requests and model-owned prompt templates
cea0f18c0e3e39d374febd1b723c4dd097fa163e 2026-11-26T20:25:54Z feat(serving): expose stateless Responses and model retrieval
0898734b0229dfb231e487b7b4d9b491fa2104ed 2026-11-26T20:47:11Z feat(serving): retain bounded foreground response resources
89822f4038b2048affd8b92dac6c7b2bcbeb219e 2026-11-26T21:13:34Z feat(serving): supervise bounded background responses and cancellation
~~~~

### Exact post-M1 takeover delta

~~~~text
53cc7ec3e8ef316de1cbc21121f4db069f57b8d4 2026-11-26T09:30:08Z fix(ux): qualify production defaults and live owner acceptance
5c919c135a542d0edacfe447b7cd5d312b147301 2026-11-26T09:40:47Z docs(ux): record P4B isolated serving gate blockers
490fe410299f0c5abf1464e52e0dcf41558ce18d 2026-11-26T10:02:23Z feat(ux): refine live model dashboard and exact layer placement
ad7a4178e3b85ea9e8a0ddcf9f2be798b0416ca8 2026-11-26T10:19:00Z fix(tui): scope acquisition-only formatting import
dc9b1956038676ea8a2ed7da119dae0f65caf1bb 2026-11-26T11:16:19Z docs(ux): accept assembled P4B gate and unblock public serving
240bd05e1e0b51a96616c4c08ef311ef53e53772 2026-11-26T11:39:54Z feat(ux): choose model backend and device from Models
fbb4ab1dd72c4fa4e52b3eaa165d7ca2211ec114 2026-11-26T14:48:29Z feat(serving): own bounded asynchronous HTTP transport
3156ce45750e838c7ace72dc7d8c0c06aa762c3b 2026-11-26T15:07:52Z feat(serving): own bounded ingress and websocket continuations
08a9d30ec5836d6885c937cd60dc0b33471329af 2026-11-26T15:44:57Z feat(serving): admit strictly framed chunked text requests
715127ccfd5568fe1fd35663784c64410cad05c4 2026-11-26T16:31:25Z docs(serving): accept strict chunked text workspace gate
3f8329f2aaad530950a355d63eea5743f4143705 2026-11-26T17:02:06Z feat(serving): own bounded multipart form ingestion
a76ddbfeaf708be9046ca14980dc1410d8f20cde 2026-11-26T19:50:18Z Unify semantic text requests and model-owned prompt templates
cea0f18c0e3e39d374febd1b723c4dd097fa163e 2026-11-26T20:25:54Z feat(serving): expose stateless Responses and model retrieval
0898734b0229dfb231e487b7b4d9b491fa2104ed 2026-11-26T20:47:11Z feat(serving): retain bounded foreground response resources
89822f4038b2048affd8b92dac6c7b2bcbeb219e 2026-11-26T21:13:34Z feat(serving): supervise bounded background responses and cancellation
~~~~

## Appendix D — all post-M1 landing narratives and exact plan changes

All fifteen commits after the preceding comprehensive takeover, chronological. Original commit bodies and source paths; the first active checkpoint narrative at each revision is preserved verbatim (may describe an intermediate or unchanged state). Exact tracked documentation/plan diffs record original design changes and decisions, including plan-only acceptance. Source diffs are retained in Git at the listed commits; no copy of vendor code or generated UI assets. Read current §§0–13 for active instructions.

### D.1 — fix(ux): qualify production defaults and live owner acceptance

Commit: `53cc7ec3e8ef316de1cbc21121f4db069f57b8d4`

Original commit message:

~~~~text
fix(ux): qualify production defaults and live owner acceptance

Preserve the explicit CPU/cache/acquisition and Web-only feature boundaries; make full-default strict lint part of the GPU feature gate. Scope GPU DSL lint exceptions and qualification accessors without altering kernel math. Align ordinary CUDA discovery tests with production features and assert the current theme payload. Exercise real cached CPU, listener events, shared transfers and original CUDA placement/performance. Full P4B isolated workspace acceptance remains pending.
~~~~

Original checkpoint narrative at this commit (historical):

~~~~markdown
### Current checkpoint — UX-P4B acceptance progress (26 Nov)

Resume P4B, not M1 or Phase 1. Actual L5 work began from the clean
`3df31912` handoff; main sole worktree, no source edits on X2/A1.
Full native default built and linked offline in the mandatory private
network harness (5m46s); version/help/doctor and real TUI/Web shell passed.
Doctor observed CUDA binding; ROCm runtime/RTC and Vulkan loader absent
from the L5 launch environment. Portable Nix/cross-release remains
explicit CPU; no all-backend distributed release claim.

- Explicit former CPU/cache and acquisition feature selections restored
  in check.sh; Web-only dependency isolation and strict lint added;
  separate full-production strict lint in the GPU feature gate.
- GPU lint fixes gate qualification-only geometry accessors, retain
  reusable direct chunk wrappers and narrowly annotate CubeCL DSL
  idioms/explicit flight arity without changing arithmetic or execution.
  Initial full strict run cleared the GPU crate, then found a nested
  facade source-selector conditional; simplified with identical refusal.
  Web fixture updated for exact existing theme field after actual smoke
  caught obsolete payload expectations. Full-default all-target strict
  Clippy then exposed an alias-gated test discovery helper; both ordinary
  CUDA tests and helper now use the production model/backend intersection.
  Full production check and strict all-target Clippy pass on retry;
  HIP/Vulkan/CUDA-native source-only compile checks also pass.
  No wire or ownership changes.
- Actual isolated release CPU CLI run_unified: 6/0/0 (74.24s), approved
  cached model, native/Firefox/HTTP, exact output, 8 GiB admission, shared
  config/selection/lifecycle and registration metrics. Real cached SDK
  listener event test: 1/0/0 (11.49s), exact failure generation/details
  for both observers, public 404 redaction. Native and shared Firefox
  transfer pause/resume/durable and cancellation pass, same operation
  and joined retirement. Web JS 4/0.
- Actual existing original Qwen3.5-0.8B CUDA SDK preadmission/refusal/retry
  fixture: 1/0/0 (20.85s). New assertions check actual admitted PCI
  0000:01:00.0, 24 compute layers, resident memory admission identity,
  zero initial counters, one completed 3-token request, one TTFT
  sample, two decode intervals after request reader retirement.
  Observed TTFT 2395219198ns, decode intervals 2 / 1558788340ns; these
  are acceptance observations only, not benchmark/parity/overhead
  evidence. GPU returned to 37 MiB and no compute owners; temporary
  existing CUDA include view removed. No download, no hardware reset.
  HIP layer placement not requalified; unsupported/older adapter maps
  remain absent. Optional M2 unchanged.

**Assembled gate:** first complete attempt stopped at the full-default
test helper compilation (E0425); repaired and rerun from the beginning.
Full-default strict lint and preceding source-only GPU checks now pass.
The complete isolated gate remains pending until terminal success.
Do not close P4B until complete actual isolated gate passes. Prior UX-1–10
and bounded Phase 1 remain accepted. Public serving follows P4B; optional
M2 is not a blocker. Actual local evidence in target/ux-p4b-*.log
(ignored build outputs, not required portable evidence).
~~~~

Changed paths:

~~~~text
M	.agents/docs/CURRENT_BASELINE.md
M	.agents/docs/CURRENT_HANDOFF.md
M	.agents/plans/FULL_IMPLEMENTATION_PLAN.md
M	.agents/plans/TODO.md
M	.agents/plans/cli-device-routing/TODO.md
M	.agents/plans/cli-ux/LIVE_MODEL_PERFORMANCE.md
M	.agents/plans/cli-ux/TUI_POLISH_AND_MOUSE_INTERACTION.md
M	.agents/plans/platform-and-release/TODO.md
M	crates/zerogravity-cli/src/run_cli/execute.rs
M	crates/zerogravity-cli/src/run_cli/tests.rs
M	crates/zerogravity-cli/tests/fixtures/web_shell.py
M	crates/zerogravity-gpu/src/bf16_projection_kernel.rs
M	crates/zerogravity-gpu/src/resident_leases/resident_matrix/source.rs
M	crates/zerogravity-gpu/src/resident_program/bf16_projection.rs
M	crates/zerogravity-gpu/src/resident_program/text_execution/execute.rs
M	crates/zerogravity-gpu/src/source_gdn/chunk/execution.rs
M	crates/zerogravity-gpu/src/source_gdn/chunk/prepared.rs
M	crates/zerogravity-gpu/src/source_gdn/recurrent.rs
M	crates/zerogravity/src/builtins/registration.rs
M	crates/zerogravity/tests/fabric_sdk.rs
M	docs/DEVELOPMENT.md
M	scripts/check-gpu-feature-boundaries.sh
M	scripts/check.sh
~~~~

Exact tracked engineering prose delta (zero-context Git diff; full surrounding prose at the cited revision):

~~~~diff
diff --git a/.agents/docs/CURRENT_BASELINE.md b/.agents/docs/CURRENT_BASELINE.md
index df714de9..aa9147eb 100644
--- a/.agents/docs/CURRENT_BASELINE.md
+++ b/.agents/docs/CURRENT_BASELINE.md
@@ -7,0 +8,16 @@ historical comparison numbers into new implementation decisions.
+## 26 Nov — UX-P4B acceptance progress
+
+Full native default linked/launch and TUI/Web smoke pass. Original
+approved cached CPU native/Firefox/HTTP 6/0/0 (74.24s), actual cached
+SDK listener metadata/redaction 1/0/0 (11.49s), native/shared Firefox
+transfer durable/cancel fixtures and JS 4/0 pass. Original CUDA SDK
+admission refusal/retry 1/0/0 (20.85s); exact 24-layer compute placement
+on admitted PCI 0000:01:00.0, one 3-token completion, one first-token
+sample and two decode intervals. Observed TTFT 2.395s, 2/1.5588s decode;
+short qualification only, not a performance baseline or comparison.
+No new numerical/ownership/admission policy. L5 GPU idle restored.
+HIP/Vulkan unavailable in full-default L5 launch environment; HIP map
+and profiling overhead not requalified. Full production strict all-target
+Clippy and preceding GPU source-only compile checks pass. Full assembled
+isolated gate pending. M2 optional; preserve existing public serving order.
+
diff --git a/.agents/docs/CURRENT_HANDOFF.md b/.agents/docs/CURRENT_HANDOFF.md
index b2383c38..6b00f2ef 100644
--- a/.agents/docs/CURRENT_HANDOFF.md
+++ b/.agents/docs/CURRENT_HANDOFF.md
@@ -37 +37,54 @@ reusable work and measured history, not precedence over this order.
-### Current checkpoint — UX-P-M1 live model performance (26 Nov)
+### Current checkpoint — UX-P4B acceptance progress (26 Nov)
+
+Resume P4B, not M1 or Phase 1. Actual L5 work began from the clean
+`3df31912` handoff; main sole worktree, no source edits on X2/A1.
+Full native default built and linked offline in the mandatory private
+network harness (5m46s); version/help/doctor and real TUI/Web shell passed.
+Doctor observed CUDA binding; ROCm runtime/RTC and Vulkan loader absent
+from the L5 launch environment. Portable Nix/cross-release remains
+explicit CPU; no all-backend distributed release claim.
+
+- Explicit former CPU/cache and acquisition feature selections restored
+  in check.sh; Web-only dependency isolation and strict lint added;
+  separate full-production strict lint in the GPU feature gate.
+- GPU lint fixes gate qualification-only geometry accessors, retain
+  reusable direct chunk wrappers and narrowly annotate CubeCL DSL
+  idioms/explicit flight arity without changing arithmetic or execution.
+  Initial full strict run cleared the GPU crate, then found a nested
+  facade source-selector conditional; simplified with identical refusal.
+  Web fixture updated for exact existing theme field after actual smoke
+  caught obsolete payload expectations. Full-default all-target strict
+  Clippy then exposed an alias-gated test discovery helper; both ordinary
+  CUDA tests and helper now use the production model/backend intersection.
+  Full production check and strict all-target Clippy pass on retry;
+  HIP/Vulkan/CUDA-native source-only compile checks also pass.
+  No wire or ownership changes.
+- Actual isolated release CPU CLI run_unified: 6/0/0 (74.24s), approved
+  cached model, native/Firefox/HTTP, exact output, 8 GiB admission, shared
+  config/selection/lifecycle and registration metrics. Real cached SDK
+  listener event test: 1/0/0 (11.49s), exact failure generation/details
+  for both observers, public 404 redaction. Native and shared Firefox
+  transfer pause/resume/durable and cancellation pass, same operation
+  and joined retirement. Web JS 4/0.
+- Actual existing original Qwen3.5-0.8B CUDA SDK preadmission/refusal/retry
+  fixture: 1/0/0 (20.85s). New assertions check actual admitted PCI
+  0000:01:00.0, 24 compute layers, resident memory admission identity,
+  zero initial counters, one completed 3-token request, one TTFT
+  sample, two decode intervals after request reader retirement.
+  Observed TTFT 2395219198ns, decode intervals 2 / 1558788340ns; these
+  are acceptance observations only, not benchmark/parity/overhead
+  evidence. GPU returned to 37 MiB and no compute owners; temporary
+  existing CUDA include view removed. No download, no hardware reset.
+  HIP layer placement not requalified; unsupported/older adapter maps
+  remain absent. Optional M2 unchanged.
+
+**Assembled gate:** first complete attempt stopped at the full-default
+test helper compilation (E0425); repaired and rerun from the beginning.
+Full-default strict lint and preceding source-only GPU checks now pass.
+The complete isolated gate remains pending until terminal success.
+Do not close P4B until complete actual isolated gate passes. Prior UX-1–10
+and bounded Phase 1 remain accepted. Public serving follows P4B; optional
+M2 is not a blocker. Actual local evidence in target/ux-p4b-*.log
+(ignored build outputs, not required portable evidence).
+
+### Prior checkpoint — UX-P-M1 live model performance (26 Nov)
diff --git a/.agents/plans/FULL_IMPLEMENTATION_PLAN.md b/.agents/plans/FULL_IMPLEMENTATION_PLAN.md
index 4ab573d8..74a0a71a 100644
--- a/.agents/plans/FULL_IMPLEMENTATION_PLAN.md
+++ b/.agents/plans/FULL_IMPLEMENTATION_PLAN.md
@@ -747,0 +748,3 @@ UX-1–10 remains accepted; this is additional presentation work.
+  P4B focused real resident/config/listener/transfer regressions, native
+  full-default linked launch/strict lint and actual CUDA layer/counter
+  observations pass; complete isolated gate remains required.
diff --git a/.agents/plans/TODO.md b/.agents/plans/TODO.md
index a3924cf1..dd02aaaa 100644
--- a/.agents/plans/TODO.md
+++ b/.agents/plans/TODO.md
@@ -232,2 +232,6 @@ selected model performance/trend and reported layer placement
-([design](cli-ux/LIVE_MODEL_PERFORMANCE.md)). P4B final full gate next. Complete UX-P before public serving/ordinary benchmarks, then
-ordered performance phases.
+([design](cli-ux/LIVE_MODEL_PERFORMANCE.md)). P4B focused real-owner
+acceptance (CPU/native/Firefox/HTTP, SDK listener, simultaneous transfers,
+actual CUDA layer/counters), full native link/launch and strict lint pass.
+Complete isolated workspace gate remains required; see live handoff.
+Complete UX-P before public serving/ordinary benchmarks, then ordered
+performance phases.
diff --git a/.agents/plans/cli-device-routing/TODO.md b/.agents/plans/cli-device-routing/TODO.md
index 0034095c..6e772f41 100644
--- a/.agents/plans/cli-device-routing/TODO.md
+++ b/.agents/plans/cli-device-routing/TODO.md
@@ -501,14 +501,13 @@ Retain accepted UX-1–10 ownership and keyboard behavior.
-  - [ ] P4B full simultaneous owner regressions and isolated gate.
-    Handoff audit (26 Nov; not an executed new gate): reconcile changed
-    production default with the explicit CPU/cache coverage. At M1 HEAD
-    scripts/check.sh shared CPU cache Clippy/test (lines 131–132) still
-    inherit the old CLI default assumptions; review them and preserve
-    original CPU/cache scope separately from full production checks.
-    Full-default link/launch and strict full GPU lint remain unverified
-    (previous link timed out; previous GPU Clippy 38 errors). Do not
-    remove valid reusable primitives or relax admission/tests. Portable
-    Nix/release CLI still explicitly CPU; reconcile full normal native
-    package intent with platform owner without silently losing portability.
-    Run actual cached SDK ServeRequestFailed generation/metadata case,
-    combined owner regressions and actual supported GPU layer observation
-    where claimed. See [complete takeover §10.2 and §11](../../docs/FULL_COMPREHENSIVE_HANDOFF_2026-11-26_UX_PM1.md).
+  - [ ] P4B comprehensive acceptance: actual combined owner regressions
+    passed (CPU CLI 6/0/0, cached listener SDK 1/0/0, real native/shared
+    Firefox transfer durable/cancel, JS 4/0). Full production linked
+    launch/version/help/doctor and TUI/Web smoke passed. Explicit old
+    CPU/cache and acquisition selection preserved; separate default
+    strict lint passes after feature-intersection test helper fix;
+    Web-only gate added. Original physical CUDA SDK
+    admission/retry 1/0/0 with exact 24-layer admitted-device map,
+    registration counters and joined retirement; HIP map not requalified.
+    Portable Nix/release remains explicit CPU. Remaining complete
+    isolated workspace gate pending; do not close P4B before actual pass.
+    See [live checkpoint](../../docs/CURRENT_HANDOFF.md) and
+    [takeover context](../../docs/FULL_COMPREHENSIVE_HANDOFF_2026-11-26_UX_PM1.md).
diff --git a/.agents/plans/cli-ux/LIVE_MODEL_PERFORMANCE.md b/.agents/plans/cli-ux/LIVE_MODEL_PERFORMANCE.md
index a971941a..fd6e6662 100644
--- a/.agents/plans/cli-ux/LIVE_MODEL_PERFORMANCE.md
+++ b/.agents/plans/cli-ux/LIVE_MODEL_PERFORMANCE.md
@@ -143,2 +143,6 @@ strings, paths, token IDs, credentials or diagnostic history in samples.
-  on original existing routes before claim. Preserve prior GPU lint debt
-  and feature boundaries. No performance comparison claim from these UX
+  on original existing routes before claim. L5 actual original CUDA
+  source SDK 1/0/0 confirms 24 compute layers on admitted PCI, exact
+  registration counters and retirement; HIP map not requalified. Full
+  production linked launch and strict lint pass, prior CPU cache and
+  Web isolation preserved. Complete isolated gate pending.
+  No performance comparison claim from these UX
diff --git a/.agents/plans/cli-ux/TUI_POLISH_AND_MOUSE_INTERACTION.md b/.agents/plans/cli-ux/TUI_POLISH_AND_MOUSE_INTERACTION.md
index 0740dd48..b4cdd9dd 100644
--- a/.agents/plans/cli-ux/TUI_POLISH_AND_MOUSE_INTERACTION.md
+++ b/.agents/plans/cli-ux/TUI_POLISH_AND_MOUSE_INTERACTION.md
@@ -678 +678,5 @@ not mark an untested or interrupted phase accepted.
-    resident/transfer/config acceptance on assembled source. Review
+    resident/transfer/config acceptance on assembled source. Actual
+    combined cached CPU/Firefox/native/HTTP 6/0/0, cached SDK listener
+    1/0/0, native/shared transfer durable/cancel, full default linked
+    TUI/Web shell and strict lint passed (26 Nov). Complete isolated gate
+    still required; see current handoff. Review
diff --git a/.agents/plans/platform-and-release/TODO.md b/.agents/plans/platform-and-release/TODO.md
index 185ea4bd..e88e04b5 100644
--- a/.agents/plans/platform-and-release/TODO.md
+++ b/.agents/plans/platform-and-release/TODO.md
@@ -98,0 +99,11 @@ build or general model/backend grant. Historical failures keep their source scop
+- [x] Separate full native Cargo CLI product from explicit portable CPU
+  Nix/cross-release assets. Full native source build/install uses the
+  production default; L5 isolated linked debug launch/version/help/doctor
+  and native/Web smoke verified. Existing portable feature selections stay
+  unchanged. Loader availability and qualified execution are distinct from
+  compiled features.
+- [ ] Future all-backend distributed native artifacts: qualify toolchain,
+  runtime library availability, target linkage and installer authenticity
+  independently of the existing portable CPU releases; no silent change to
+  those release definitions or claim of cross-target execution.
+
~~~~

### D.2 — docs(ux): record P4B isolated serving gate blockers

Commit: `5c919c135a542d0edacfe447b7cd5d312b147301`

Original commit message:

~~~~text
docs(ux): record P4B isolated serving gate blockers
~~~~

Original checkpoint narrative at this commit (historical):

~~~~markdown
### Current checkpoint — UX-P4B acceptance progress (26 Nov)

Resume P4B, not M1 or Phase 1. Actual L5 work began from the clean
`3df31912` handoff; main sole worktree, no source edits on X2/A1.
Full native default built and linked offline in the mandatory private
network harness (5m46s); version/help/doctor and real TUI/Web shell passed.
Doctor observed CUDA binding; ROCm runtime/RTC and Vulkan loader absent
from the L5 launch environment. Portable Nix/cross-release remains
explicit CPU; no all-backend distributed release claim.

- Explicit former CPU/cache and acquisition feature selections restored
  in check.sh; Web-only dependency isolation and strict lint added;
  separate full-production strict lint in the GPU feature gate.
- GPU lint fixes gate qualification-only geometry accessors, retain
  reusable direct chunk wrappers and narrowly annotate CubeCL DSL
  idioms/explicit flight arity without changing arithmetic or execution.
  Initial full strict run cleared the GPU crate, then found a nested
  facade source-selector conditional; simplified with identical refusal.
  Web fixture updated for exact existing theme field after actual smoke
  caught obsolete payload expectations. Full-default all-target strict
  Clippy then exposed an alias-gated test discovery helper; both ordinary
  CUDA tests and helper now use the production model/backend intersection.
  Full production check and strict all-target Clippy pass on retry;
  HIP/Vulkan/CUDA-native source-only compile checks also pass.
  No wire or ownership changes.
- Actual isolated release CPU CLI run_unified: 6/0/0 (74.24s), approved
  cached model, native/Firefox/HTTP, exact output, 8 GiB admission, shared
  config/selection/lifecycle and registration metrics. Real cached SDK
  listener event test: 1/0/0 (11.49s), exact failure generation/details
  for both observers, public 404 redaction. Native and shared Firefox
  transfer pause/resume/durable and cancellation pass, same operation
  and joined retirement. Web JS 4/0.
- Actual existing original Qwen3.5-0.8B CUDA SDK preadmission/refusal/retry
  fixture: 1/0/0 (20.85s). New assertions check actual admitted PCI
  0000:01:00.0, 24 compute layers, resident memory admission identity,
  zero initial counters, one completed 3-token request, one TTFT
  sample, two decode intervals after request reader retirement.
  Observed TTFT 2395219198ns, decode intervals 2 / 1558788340ns; these
  are acceptance observations only, not benchmark/parity/overhead
  evidence. GPU returned to 37 MiB and no compute owners; temporary
  existing CUDA include view removed. No download, no hardware reset.
  HIP layer placement not requalified; unsupported/older adapter maps
  remain absent. Optional M2 unchanged.

**Assembled gate:** first complete attempt stopped at the full-default
test helper compilation (E0425); repaired and rerun from the beginning.
Full-default strict lint and preceding source-only GPU checks now pass.
Retry reached actual CPU workspace execution after GPU feature-boundary and CPU strict lint passed: 1191 passed, 2 failed, 56 ignored across 116 completed Rust result lines. Exit 101 in zerogravity-serving cold_server: bearer_configuration_refuses_empty_oversized_and_header_control_bytes expects Malformed, while current auth::validate_key returns InvalidInput; cold_builder_reports_configuration_and_unconfigured_runtime_refuses_start expects Limit, but start returns a different refusal. Inspect exact typed admission and shutdown refusals and update only demonstrably obsolete expectations; do not weaken assertions. Remaining workspace stages not reached. Implementation checkpoint 53cc7ec3e8ef316de1cbc21121f4db069f57b8d4. Next: focused cold_server qualification, then complete isolated gate from beginning. Evidence target/ux-p4b-workspace-gate-retry.log.
Do not close P4B until complete actual isolated gate passes. Prior UX-1–10
and bounded Phase 1 remain accepted. Public serving follows P4B; optional
M2 is not a blocker. Actual local evidence in target/ux-p4b-*.log
(ignored build outputs, not required portable evidence).
~~~~

Changed paths:

~~~~text
M	.agents/docs/CURRENT_BASELINE.md
M	.agents/docs/CURRENT_HANDOFF.md
M	.agents/plans/cli-device-routing/TODO.md
~~~~

Exact tracked engineering prose delta (zero-context Git diff; full surrounding prose at the cited revision):

~~~~diff
diff --git a/.agents/docs/CURRENT_BASELINE.md b/.agents/docs/CURRENT_BASELINE.md
index aa9147eb..ea960646 100644
--- a/.agents/docs/CURRENT_BASELINE.md
+++ b/.agents/docs/CURRENT_BASELINE.md
@@ -23,0 +24,2 @@ isolated gate pending. M2 optional; preserve existing public serving order.
+Assembled retry: GPU feature-boundary and CPU strict lint pass; 1191 passed, 2 failed, 56 ignored before cold_server stopped execution (exit 101). See CURRENT_HANDOFF for exact typed refusal assertions. P4B remains open.
+
diff --git a/.agents/docs/CURRENT_HANDOFF.md b/.agents/docs/CURRENT_HANDOFF.md
index 6b00f2ef..ab87333e 100644
--- a/.agents/docs/CURRENT_HANDOFF.md
+++ b/.agents/docs/CURRENT_HANDOFF.md
@@ -84 +84 @@ Full-default strict lint and preceding source-only GPU checks now pass.
-The complete isolated gate remains pending until terminal success.
+Retry reached actual CPU workspace execution after GPU feature-boundary and CPU strict lint passed: 1191 passed, 2 failed, 56 ignored across 116 completed Rust result lines. Exit 101 in zerogravity-serving cold_server: bearer_configuration_refuses_empty_oversized_and_header_control_bytes expects Malformed, while current auth::validate_key returns InvalidInput; cold_builder_reports_configuration_and_unconfigured_runtime_refuses_start expects Limit, but start returns a different refusal. Inspect exact typed admission and shutdown refusals and update only demonstrably obsolete expectations; do not weaken assertions. Remaining workspace stages not reached. Implementation checkpoint 53cc7ec3e8ef316de1cbc21121f4db069f57b8d4. Next: focused cold_server qualification, then complete isolated gate from beginning. Evidence target/ux-p4b-workspace-gate-retry.log.
diff --git a/.agents/plans/cli-device-routing/TODO.md b/.agents/plans/cli-device-routing/TODO.md
index 6e772f41..8841d726 100644
--- a/.agents/plans/cli-device-routing/TODO.md
+++ b/.agents/plans/cli-device-routing/TODO.md
@@ -511 +511 @@ Retain accepted UX-1–10 ownership and keyboard behavior.
-    isolated workspace gate pending; do not close P4B before actual pass.
+    isolated workspace gate retry stopped at two cold_server typed refusal assertions (1191 passed, 2 failed, 56 ignored); qualify exact refusals and rerun full gate. Do not close P4B before actual pass.
~~~~

### D.3 — feat(ux): refine live model dashboard and exact layer placement

Commit: `490fe410299f0c5abf1464e52e0dcf41558ce18d`

Original commit message:

~~~~text
feat(ux): refine live model dashboard and exact layer placement

Keep registration metrics and original execution owner unchanged. Give Running a larger selected pane, responsive paired metrics, recent chart and reported layer cells with mixed-range/unknown semantics. Add an exact selected-owner details link and retain manual split preferences. Qualify native/Firefox/HTTP real cached lifecycle and repair obsolete cold serving typed-error assertions. Full P4B workspace gate remains pending.
~~~~

Original checkpoint narrative at this commit (historical):

~~~~markdown
### Current checkpoint — UX-P-M2 visual refinement and P4B repairs (26 Nov)

User explicitly requested the Running dashboard refinements before
returning to the two failed P4B serving tests. Revalidated clean
`5c919c13` on L5, sole source authoring/main worktree. See
[LIVE_MODEL_PERFORMANCE](../plans/cli-ux/LIVE_MODEL_PERFORMANCE.md).

- Native reference-aligned dashboard: 38:62 default list/detail split
  (explicit manual split retained), side-by-side measured decode/TTFT
  values with narrow fallback, width-bounded one/two-row recent chart
  and emphasized recent observations, concise actual provider labels,
  admitted-reader activity, segmented reported layer map and exact
  ranges/legend. Direct visible details link uses original bound selected
  owner; Alt-D/context and independent response viewport retained. Web
  same hierarchy and responsive proportions; exact tooltip layer cells,
  on-demand details. Actual complete spans only; compressed cross-device
  intervals marked mixed; unreported remains unknown. No new engine or
  backend activity instrumentation; admitted readers include queued,
  active and retiring work, not instantaneous layer execution.
- Focused combined CPU strict Clippy, native 20/0/0 (including actual
  rendered metric pair and link press/release, stale refusal), JS 5/0
  and local asset build. Actual approved cached Qwen CPU/Firefox/native/
  HTTP 1/0/0 (53.65s, five filtered) confirms 24 layer cells, exact
  tooltip endpoints, original 8 GiB admission, output, registration
  counters and reset, shared config/listener and joined retirement.
  First run exposed literal-space PTY expectation where Ratatui uses
  cursor moves; exact render-aware assertion and final rerun passed.
- P4B prior two cold_server failures corrected only at the tests:
  malformed bearer configuration is InvalidInput with exact existing
  diagnostic; unconfigured scheduler is Unsupported with "no text
  scheduler configured". Focused standalone serving 4/0/0. Production
  classification, public redaction and owner/admission unchanged.
- Full mandatory isolated assembled gate still required. No additional
  GPU physical requalification or overhead/throughput benchmark claimed.
  Reuse prior physical CPU/CUDA and full-default launch evidence where
  unchanged. M2 deeper exact request/phase/per-layer timing remains
  optional, capability-bound. Public serving after P4B only.
  Logs: target/ux-pm2-focused.log, target/ux-pm2-final-focused-retry.log,
  target/ux-pm2-p4b-gate.log.
~~~~

Changed paths:

~~~~text
M	.agents/docs/CURRENT_BASELINE.md
M	.agents/docs/CURRENT_HANDOFF.md
M	.agents/plans/FULL_IMPLEMENTATION_PLAN.md
M	.agents/plans/cli-device-routing/TODO.md
M	.agents/plans/cli-ux/LIVE_MODEL_PERFORMANCE.md
M	crates/zerogravity-cli/src/tui/layout.rs
M	crates/zerogravity-cli/src/tui/performance.rs
M	crates/zerogravity-cli/src/tui/view.rs
M	crates/zerogravity-cli/tests/fixtures/tui_shell.py
M	crates/zerogravity-cli/web/dist/app.js
M	crates/zerogravity-cli/web/dist/style.css
M	crates/zerogravity-cli/web/src/browser-smoke.mjs
M	crates/zerogravity-cli/web/src/main.jsx
M	crates/zerogravity-cli/web/src/performance.js
M	crates/zerogravity-cli/web/src/performance.jsx
M	crates/zerogravity-cli/web/src/performance.test.js
M	crates/zerogravity-cli/web/src/styles.css
M	crates/zerogravity-serving/tests/cold_server.rs
~~~~

Exact tracked engineering prose delta (zero-context Git diff; full surrounding prose at the cited revision):

~~~~diff
diff --git a/.agents/docs/CURRENT_BASELINE.md b/.agents/docs/CURRENT_BASELINE.md
index ea960646..d6ef2519 100644
--- a/.agents/docs/CURRENT_BASELINE.md
+++ b/.agents/docs/CURRENT_BASELINE.md
@@ -7,0 +8,10 @@ historical comparison numbers into new implementation decisions.
+## 26 Nov — UX-P-M2 dashboard refinement
+
+Reference-aligned adaptive metrics, recent chart, complete segmented
+layer placement and actual admitted-reader activity. Native 20/0/0,
+JS 5/0, strict combined CPU lint, actual cached native/Firefox/HTTP
+1/0/0 (53.65s) with original admission, exact output and 24 reported CPU
+layers. Existing two cold_server typed-refusal assertions repaired:
+standalone serving 4/0/0. No additional hot-path observation or
+performance comparison. Complete P4B gate pending.
+
diff --git a/.agents/docs/CURRENT_HANDOFF.md b/.agents/docs/CURRENT_HANDOFF.md
index ab87333e..b88dd869 100644
--- a/.agents/docs/CURRENT_HANDOFF.md
+++ b/.agents/docs/CURRENT_HANDOFF.md
@@ -37 +37,41 @@ reusable work and measured history, not precedence over this order.
-### Current checkpoint — UX-P4B acceptance progress (26 Nov)
+### Current checkpoint — UX-P-M2 visual refinement and P4B repairs (26 Nov)
+
+User explicitly requested the Running dashboard refinements before
+returning to the two failed P4B serving tests. Revalidated clean
+`5c919c13` on L5, sole source authoring/main worktree. See
+[LIVE_MODEL_PERFORMANCE](../plans/cli-ux/LIVE_MODEL_PERFORMANCE.md).
+
+- Native reference-aligned dashboard: 38:62 default list/detail split
+  (explicit manual split retained), side-by-side measured decode/TTFT
+  values with narrow fallback, width-bounded one/two-row recent chart
+  and emphasized recent observations, concise actual provider labels,
+  admitted-reader activity, segmented reported layer map and exact
+  ranges/legend. Direct visible details link uses original bound selected
+  owner; Alt-D/context and independent response viewport retained. Web
+  same hierarchy and responsive proportions; exact tooltip layer cells,
+  on-demand details. Actual complete spans only; compressed cross-device
+  intervals marked mixed; unreported remains unknown. No new engine or
+  backend activity instrumentation; admitted readers include queued,
+  active and retiring work, not instantaneous layer execution.
+- Focused combined CPU strict Clippy, native 20/0/0 (including actual
+  rendered metric pair and link press/release, stale refusal), JS 5/0
+  and local asset build. Actual approved cached Qwen CPU/Firefox/native/
+  HTTP 1/0/0 (53.65s, five filtered) confirms 24 layer cells, exact
+  tooltip endpoints, original 8 GiB admission, output, registration
+  counters and reset, shared config/listener and joined retirement.
+  First run exposed literal-space PTY expectation where Ratatui uses
+  cursor moves; exact render-aware assertion and final rerun passed.
+- P4B prior two cold_server failures corrected only at the tests:
+  malformed bearer configuration is InvalidInput with exact existing
+  diagnostic; unconfigured scheduler is Unsupported with "no text
+  scheduler configured". Focused standalone serving 4/0/0. Production
+  classification, public redaction and owner/admission unchanged.
+- Full mandatory isolated assembled gate still required. No additional
+  GPU physical requalification or overhead/throughput benchmark claimed.
+  Reuse prior physical CPU/CUDA and full-default launch evidence where
+  unchanged. M2 deeper exact request/phase/per-layer timing remains
+  optional, capability-bound. Public serving after P4B only.
+  Logs: target/ux-pm2-focused.log, target/ux-pm2-final-focused-retry.log,
+  target/ux-pm2-p4b-gate.log.
+
+### Prior checkpoint — UX-P4B acceptance progress (26 Nov)
diff --git a/.agents/plans/FULL_IMPLEMENTATION_PLAN.md b/.agents/plans/FULL_IMPLEMENTATION_PLAN.md
index 74a0a71a..25441857 100644
--- a/.agents/plans/FULL_IMPLEMENTATION_PLAN.md
+++ b/.agents/plans/FULL_IMPLEMENTATION_PLAN.md
@@ -750,2 +750,5 @@ UX-1–10 remains accepted; this is additional presentation work.
-  observations pass; complete isolated gate remains required.
-  Optional deeper profiling follows actual provider capabilities.
+  observations pass. User-directed M2 reference-aligned dashboard and
+  exact layer strip, visible details and activity refinement focused-
+  qualified; standalone serving refusal assertions corrected and pass.
+  Complete isolated gate remains required. Optional deeper profiling
+  follows actual provider capabilities.
diff --git a/.agents/plans/cli-device-routing/TODO.md b/.agents/plans/cli-device-routing/TODO.md
index 8841d726..fe1896b6 100644
--- a/.agents/plans/cli-device-routing/TODO.md
+++ b/.agents/plans/cli-device-routing/TODO.md
@@ -499 +499,7 @@ Retain accepted UX-1–10 ownership and keyboard behavior.
-  - [ ] UX-P-M2 optional exact per-request/phase and detailed layer
+  - [x] UX-P-M2 user-directed Running visual refinement: larger detail pane
+    by default (preserve explicit split), responsive metric pair, richer
+    recent output chart, exact segmented layer strip/device legend, admitted
+    reader activity and direct exact-owner details link. Native 20/0/0,
+    JS 5/0, strict lint, actual cached native/Firefox/HTTP 1/0/0 (53.65s),
+    unchanged 8 GiB admission and exact output. No inference-path changes.
+  - [ ] Optional deeper exact per-request/phase and individual layer
@@ -511 +517,5 @@ Retain accepted UX-1–10 ownership and keyboard behavior.
-    isolated workspace gate retry stopped at two cold_server typed refusal assertions (1191 passed, 2 failed, 56 ignored); qualify exact refusals and rerun full gate. Do not close P4B before actual pass.
+    isolated workspace gate retry stopped at two cold_server typed refusal
+    assertions (1191 passed, 2 failed, 56 ignored). Both corrected to the
+    exact existing InvalidInput/Unsupported categories and messages;
+    focused cold_server 4/0/0. Complete isolated assembled gate pending.
+    Do not close P4B before actual pass.
diff --git a/.agents/plans/cli-ux/LIVE_MODEL_PERFORMANCE.md b/.agents/plans/cli-ux/LIVE_MODEL_PERFORMANCE.md
index fd6e6662..7ab2b2e8 100644
--- a/.agents/plans/cli-ux/LIVE_MODEL_PERFORMANCE.md
+++ b/.agents/plans/cli-ux/LIVE_MODEL_PERFORMANCE.md
@@ -3 +3 @@
-Status: UX-P-M1 implemented and focused-qualified (26 Nov); optional M2 and assembled P4B pending.
+Status: UX-P-M1 and user-directed M2 presentation refinement focused-qualified (26 Nov); deeper capability-bound profiling and assembled P4B pending.
@@ -38,3 +38,15 @@ Illustrative placement, not a new hardware qualification:
-A future genuinely split owner can show proportional contiguous spans and
-a short legend/ranges per physical device. Never infer a split by dividing
-layers by VRAM, model name or desired configuration. Distinguish compute
+The user-directed M2 visual refinement preserves the reference layout:
+38:62 default list/detail width on wide native/Web views; native explicit
+split preferences remain intact. Side-by-side metric values with compact
+fallback, two-cell-high recent chart where space permits, newest samples
+emphasized, visible exact-owner details link, and actual admitted-reader
+activity. The Web uses one accessible tooltip-labeled cell per reported
+layer up to 48; native caps cells by available width and 48. A complete
+ordered map is required for the strip. A compressed cell spanning devices
+is explicitly mixed, never assigned an invented owner. Legend identity
+and symbols supplement color; exact zero-based ranges remain in details.
+An older/unsupported adapter still has no placement. Activity is
+registration-wide (admitted readers include queue/active/retirement),
+not instantaneous layer utilization. No animation or hot-path instrumentation
+pretends to observe individual layers.
+Never infer a split by dividing layers by VRAM, model name or desired configuration. Distinguish compute
@@ -127 +139,15 @@ strings, paths, token IDs, credentials or diagnostic history in samples.
-- [ ] UX-P-M2 optional richer request and phase inspection after M1:
+- [x] UX-P-M2 presentation refinement, requested before finishing P4B:
+  reference-aligned side-by-side metrics/quiet chart and larger detail
+  pane, actual provider labels, admitted-reader activity, segmented exact
+  layer map with stable device marks and mixed/unknown handling, direct
+  exact-owner details link (mouse and Alt-D/context). Original response
+  viewport, manual split and M1 measurements unchanged. Native focused
+  20/0/0 including rendered side-by-side metrics, actual pointer binding
+  and stale owner rejection; JS 5/0, offline local asset build, strict
+  focused lint. Real cached CPU/Firefox/native/HTTP 1/0/0 (53.65s):
+  exact 24 layer blocks, browser first/last layer labels, shared
+  registration metrics and existing admission/output/retirement.
+  First PTY assertion failed because blank cells were cursor moves;
+  exact render-aware assertion now passes. No new engine/backend
+  observations, timers, synchronization or profiling overhead.
+- [ ] UX-P-M2 optional richer request and phase inspection beyond the presentation refinement:
~~~~

### D.4 — fix(tui): scope acquisition-only formatting import

Commit: `ad7a4178e3b85ea9e8a0ddcf9f2be798b0416ca8`

Original commit message:

~~~~text
fix(tui): scope acquisition-only formatting import
~~~~

Original checkpoint narrative at this commit (historical):

~~~~markdown
### Current checkpoint — UX-P-M2 visual refinement and P4B repairs (26 Nov)

User explicitly requested the Running dashboard refinements before
returning to the two failed P4B serving tests. Revalidated clean
`5c919c13` on L5, sole source authoring/main worktree. See
[LIVE_MODEL_PERFORMANCE](../plans/cli-ux/LIVE_MODEL_PERFORMANCE.md).

- Native reference-aligned dashboard: 38:62 default list/detail split
  (explicit manual split retained), side-by-side measured decode/TTFT
  values with narrow fallback, width-bounded one/two-row recent chart
  and emphasized recent observations, concise actual provider labels,
  admitted-reader activity, segmented reported layer map and exact
  ranges/legend. Direct visible details link uses original bound selected
  owner; Alt-D/context and independent response viewport retained. Web
  same hierarchy and responsive proportions; exact tooltip layer cells,
  on-demand details. Actual complete spans only; compressed cross-device
  intervals marked mixed; unreported remains unknown. No new engine or
  backend activity instrumentation; admitted readers include queued,
  active and retiring work, not instantaneous layer execution.
- Focused combined CPU strict Clippy, native 20/0/0 (including actual
  rendered metric pair and link press/release, stale refusal), JS 5/0
  and local asset build. Actual approved cached Qwen CPU/Firefox/native/
  HTTP 1/0/0 (53.65s, five filtered) confirms 24 layer cells, exact
  tooltip endpoints, original 8 GiB admission, output, registration
  counters and reset, shared config/listener and joined retirement.
  First run exposed literal-space PTY expectation where Ratatui uses
  cursor moves; exact render-aware assertion and final rerun passed.
- P4B prior two cold_server failures corrected only at the tests:
  malformed bearer configuration is InvalidInput with exact existing
  diagnostic; unconfigured scheduler is Unsupported with "no text
  scheduler configured". Focused standalone serving 4/0/0. Production
  classification, public redaction and owner/admission unchanged.
- Assembled isolated retry passed HIP/Vulkan/CUDA-native compilation,
  full-default check and strict Clippy, GPU feature-boundary tests (26/0/2),
  source-only regressions (7/0), architecture (18/0) and CPU dependency
  isolation, then stopped at a TUI format import used only with Linux
  acquisition or tests. Scoped the import accordingly; exact CPU workspace
  strict all-target Clippy passes (6.33s). See target/ux-pm2-cpu-clippy-fix.log.
  Exact isolated CPU workspace test-stage retry (including dependency
  isolation, strict lint and private loopback setup) reached test compilation
  but exceeded the 90s remaining turn budget; no test suites executed.
  See target/ux-pm2-cpu-workspace-retry.log. Next rerun the full gate;
  do not mark P4B accepted yet.
- Full mandatory isolated assembled gate still required. No additional
  GPU physical requalification or overhead/throughput benchmark claimed.
  Reuse prior physical CPU/CUDA and full-default launch evidence where
  unchanged. M2 deeper exact request/phase/per-layer timing remains
  optional, capability-bound. Public serving after P4B only.
  Logs: target/ux-pm2-focused.log, target/ux-pm2-final-focused-retry.log,
  target/ux-pm2-p4b-gate.log.
~~~~

Changed paths:

~~~~text
M	.agents/docs/CURRENT_BASELINE.md
M	.agents/docs/CURRENT_HANDOFF.md
M	.agents/plans/cli-device-routing/TODO.md
M	crates/zerogravity-cli/src/tui/view.rs
~~~~

Exact tracked engineering prose delta (zero-context Git diff; full surrounding prose at the cited revision):

~~~~diff
diff --git a/.agents/docs/CURRENT_BASELINE.md b/.agents/docs/CURRENT_BASELINE.md
index d6ef2519..70477a6b 100644
--- a/.agents/docs/CURRENT_BASELINE.md
+++ b/.agents/docs/CURRENT_BASELINE.md
@@ -16 +16,5 @@ standalone serving 4/0/0. No additional hot-path observation or
-performance comparison. Complete P4B gate pending.
+performance comparison. Assembled retry passed GPU feature-boundary tests
+(26/0/2) and full-default strict lint, then found an acquisition-disabled
+TUI import. Import scope corrected; exact CPU workspace strict all-target
+Clippy passes. CPU workspace test-stage retry timed out during compilation
+(90s budget, no suites executed). Complete P4B gate pending.
diff --git a/.agents/docs/CURRENT_HANDOFF.md b/.agents/docs/CURRENT_HANDOFF.md
index b88dd869..b44f0498 100644
--- a/.agents/docs/CURRENT_HANDOFF.md
+++ b/.agents/docs/CURRENT_HANDOFF.md
@@ -68,0 +69,11 @@ returning to the two failed P4B serving tests. Revalidated clean
+- Assembled isolated retry passed HIP/Vulkan/CUDA-native compilation,
+  full-default check and strict Clippy, GPU feature-boundary tests (26/0/2),
+  source-only regressions (7/0), architecture (18/0) and CPU dependency
+  isolation, then stopped at a TUI format import used only with Linux
+  acquisition or tests. Scoped the import accordingly; exact CPU workspace
+  strict all-target Clippy passes (6.33s). See target/ux-pm2-cpu-clippy-fix.log.
+  Exact isolated CPU workspace test-stage retry (including dependency
+  isolation, strict lint and private loopback setup) reached test compilation
+  but exceeded the 90s remaining turn budget; no test suites executed.
+  See target/ux-pm2-cpu-workspace-retry.log. Next rerun the full gate;
+  do not mark P4B accepted yet.
diff --git a/.agents/plans/cli-device-routing/TODO.md b/.agents/plans/cli-device-routing/TODO.md
index fe1896b6..65b2e273 100644
--- a/.agents/plans/cli-device-routing/TODO.md
+++ b/.agents/plans/cli-device-routing/TODO.md
@@ -520 +520,6 @@ Retain accepted UX-1–10 ownership and keyboard behavior.
-    focused cold_server 4/0/0. Complete isolated assembled gate pending.
+    focused cold_server 4/0/0. M2 assembled retry passed full-default
+    strict lint, GPU feature-boundary tests 26/0/2 and CPU dependency
+    isolation. Acquisition-disabled TUI format import fixed; exact CPU
+    workspace strict all-target lint passes. Isolated CPU test-stage retry
+    timed out during compilation (90s budget, no suites executed). Complete
+    isolated assembled gate pending.
~~~~

### D.5 — docs(ux): accept assembled P4B gate and unblock public serving

Commit: `dc9b1956038676ea8a2ed7da119dae0f65caf1bb`

Original commit message:

~~~~text
docs(ux): accept assembled P4B gate and unblock public serving
~~~~

Original checkpoint narrative at this commit (historical):

~~~~markdown
### Current checkpoint — UX-P4B accepted; public serving next (26 Nov)

Accepted implementation: `ad7a4178e3b85ea9e8a0ddcf9f2be798b0416ca8`. L5 main sole worktree,
clean at gate completion; no source changes needed in this acceptance run.
The complete existing offline workspace gate under mandatory private
network isolation exited 0 in 592.3s: 2,539 passed, zero failures, 109 explicit skips across 318 Rust suites.
Both cold_server assertions passed in the assembled run. Formatting,
architecture, strict full-default and CPU/optional-feature lint, source-only
GPU feature-boundary checks, minimal SDK/inspection/Web/cache/acquisition,
portable CPU release, all release native PTY and Web control/settings,
installer and scripts passed. Local log: target/ux-p4b-assembled-ad7a4178.log.

The prior final real cached CPU/native/Firefox/HTTP lifecycle (6/0/0;
M2 final 1/0/0), exact 8 GiB admission/output and 24 CPU layers, SDK
listener/redaction, native/shared transfer durable/cancel, full native
default link/launch and actual CUDA source admission/retry with 24
reported compute layers provide the applicable owner qualification.
The final gate includes the acquisition-only import correction and both
typed-refusal fixture corrections; no later production behavior changes.
No new GPU execution, HIP/Vulkan placement or profiling-overhead
qualification or performance-parity claim. L5 GPU 37 MiB idle, no compute
owners. No push.

**Next coherent unit: public serving OA-1** under §2.3 of the full plan.
Read [the serving design](../plans/serving-and-speculation/NEXT_ITERATION_OPENAI_COMPATIBLE_SERVING.md)
§32 and §37 and live serving implementation/owning TODO. Verify the
current transport boundary before replacing it with mature bounded async
Rust transport; preserve existing models/Chat/Completions, auth,
cancellation, ingress/backpressure and owned shutdown. No Responses
translation hacks, model semantics or second scheduler in OA-1. Do not
reopen accepted UX-P; optional deeper profiling remains capability-bound.
~~~~

Changed paths:

~~~~text
M	.agents/docs/CURRENT_BASELINE.md
M	.agents/docs/CURRENT_HANDOFF.md
M	.agents/plans/FULL_IMPLEMENTATION_PLAN.md
M	.agents/plans/TODO.md
M	.agents/plans/cli-device-routing/TODO.md
M	.agents/plans/cli-ux/LIVE_MODEL_PERFORMANCE.md
M	.agents/plans/cli-ux/TUI_POLISH_AND_MOUSE_INTERACTION.md
M	.agents/plans/serving-and-speculation/NEXT_ITERATION_OPENAI_COMPATIBLE_SERVING.md
~~~~

Exact tracked engineering prose delta (zero-context Git diff; full surrounding prose at the cited revision):

~~~~diff
diff --git a/.agents/docs/CURRENT_BASELINE.md b/.agents/docs/CURRENT_BASELINE.md
index 70477a6b..5ff3d44a 100644
--- a/.agents/docs/CURRENT_BASELINE.md
+++ b/.agents/docs/CURRENT_BASELINE.md
@@ -8 +8,13 @@ historical comparison numbers into new implementation decisions.
-## 26 Nov — UX-P-M2 dashboard refinement
+## 26 Nov — UX-P4B accepted
+
+Final offline private-network workspace gate on `ad7a4178e3b85ea9e8a0ddcf9f2be798b0416ca8`
+exited 0 in 592.3s: 2,539 passed, zero failures, 109 explicit skips across 318 Rust suites.
+Full-default and explicit-feature strict lint, architecture/source-only GPU
+boundaries, portable CPU release, native PTY (including 20x7), Web control/
+settings, installer and scripts pass. Prior final real CPU/native/Firefox/
+HTTP, SDK listener and transfer qualification, full native link/launch and
+physical CUDA 24-layer/counter evidence remain applicable. No new numerical
+comparison, HIP/Vulkan execution or overhead benchmark. UX-P accepted;
+public serving OA-1 next. Log: target/ux-p4b-assembled-ad7a4178.log.
+
+## 26 Nov — UX-P-M2 dashboard refinement (prior checkpoint)
diff --git a/.agents/docs/CURRENT_HANDOFF.md b/.agents/docs/CURRENT_HANDOFF.md
index b44f0498..d8308fbf 100644
--- a/.agents/docs/CURRENT_HANDOFF.md
+++ b/.agents/docs/CURRENT_HANDOFF.md
@@ -7,3 +7,4 @@ Canonical cross-window entry point for `/home/pvl/spaces/abird/src/zerogravity`.
-Read §§0–13 first, especially §10.2 (default/build/gate caveats) and §11
-(next required P4B acceptance, optional M2). Use the
-[standalone P4B takeover prompt](NEXT_AGENT_2026-11-26_UX_P4B.md).
+Read §§0–13 for historical design and gate context. The
+[standalone P4B takeover prompt](NEXT_AGENT_2026-11-26_UX_P4B.md)
+is superseded by the accepted checkpoint below; public serving OA-1
+is the next unit.
@@ -37 +38,33 @@ reusable work and measured history, not precedence over this order.
-### Current checkpoint — UX-P-M2 visual refinement and P4B repairs (26 Nov)
+### Current checkpoint — UX-P4B accepted; public serving next (26 Nov)
+
+Accepted implementation: `ad7a4178e3b85ea9e8a0ddcf9f2be798b0416ca8`. L5 main sole worktree,
+clean at gate completion; no source changes needed in this acceptance run.
+The complete existing offline workspace gate under mandatory private
+network isolation exited 0 in 592.3s: 2,539 passed, zero failures, 109 explicit skips across 318 Rust suites.
+Both cold_server assertions passed in the assembled run. Formatting,
+architecture, strict full-default and CPU/optional-feature lint, source-only
+GPU feature-boundary checks, minimal SDK/inspection/Web/cache/acquisition,
+portable CPU release, all release native PTY and Web control/settings,
+installer and scripts passed. Local log: target/ux-p4b-assembled-ad7a4178.log.
+
+The prior final real cached CPU/native/Firefox/HTTP lifecycle (6/0/0;
+M2 final 1/0/0), exact 8 GiB admission/output and 24 CPU layers, SDK
+listener/redaction, native/shared transfer durable/cancel, full native
+default link/launch and actual CUDA source admission/retry with 24
+reported compute layers provide the applicable owner qualification.
+The final gate includes the acquisition-only import correction and both
+typed-refusal fixture corrections; no later production behavior changes.
+No new GPU execution, HIP/Vulkan placement or profiling-overhead
+qualification or performance-parity claim. L5 GPU 37 MiB idle, no compute
+owners. No push.
+
+**Next coherent unit: public serving OA-1** under §2.3 of the full plan.
+Read [the serving design](../plans/serving-and-speculation/NEXT_ITERATION_OPENAI_COMPATIBLE_SERVING.md)
+§32 and §37 and live serving implementation/owning TODO. Verify the
+current transport boundary before replacing it with mature bounded async
+Rust transport; preserve existing models/Chat/Completions, auth,
+cancellation, ingress/backpressure and owned shutdown. No Responses
+translation hacks, model semantics or second scheduler in OA-1. Do not
+reopen accepted UX-P; optional deeper profiling remains capability-bound.
+
+### Prior checkpoint — UX-P-M2 visual refinement and P4B repairs (26 Nov)
diff --git a/.agents/plans/FULL_IMPLEMENTATION_PLAN.md b/.agents/plans/FULL_IMPLEMENTATION_PLAN.md
index 25441857..8d740731 100644
--- a/.agents/plans/FULL_IMPLEMENTATION_PLAN.md
+++ b/.agents/plans/FULL_IMPLEMENTATION_PLAN.md
@@ -715,3 +715,3 @@ demand. Preserve the existing detailed
-**Current next priority (26 Nov):** user requests full mouse interaction,
-resizable/focusable panes, independent scrolling and a polished, elegant
-visual system before public serving. See the owning
+**Accepted (26 Nov):** user-requested full mouse interaction,
+resizable/focusable panes, independent scrolling and polished visuals
+completed before public serving. See the owning
@@ -735 +735 @@ UX-1–10 remains accepted; this is additional presentation work.
-- [ ] UX-P4: visual/terminal-depth/accessibility polish, actual PTY and
+- [x] UX-P4: visual/terminal-depth/accessibility polish, actual PTY and
@@ -753,2 +753,3 @@ UX-1–10 remains accepted; this is additional presentation work.
-  Complete isolated gate remains required. Optional deeper profiling
-  follows actual provider capabilities.
+  Final assembled private-network gate on ad7a4178 passed: 2,539 passed, zero failures, 109 explicit skips across 318 Rust suites.
+  Release Web/TUI/installer/scripts passed. UX-P accepted; proceed to §2.3.
+  Optional deeper profiling follows actual provider capabilities.
@@ -757 +758 @@ No new runtime, cache, configuration or execution owner. No inference
-protocol or backend performance work until this bounded polish exit.
+protocol or backend performance work was admitted before this bounded exit.
diff --git a/.agents/plans/TODO.md b/.agents/plans/TODO.md
index dd02aaaa..2fce1882 100644
--- a/.agents/plans/TODO.md
+++ b/.agents/plans/TODO.md
@@ -235,3 +235,3 @@ actual CUDA layer/counters), full native link/launch and strict lint pass.
-Complete isolated workspace gate remains required; see live handoff.
-Complete UX-P before public serving/ordinary benchmarks, then ordered
-performance phases.
+Final assembled private-network gate on ad7a4178 passed 2,539 passed, zero failures, 109 explicit skips across 318 Rust suites; release Web/TUI/installer/scripts passed. UX-P accepted. Proceed to
+public serving OA-1, then ordinary benchmarks and ordered performance phases.
+Optional deeper profiling is not a blocker; see live handoff.
diff --git a/.agents/plans/cli-device-routing/TODO.md b/.agents/plans/cli-device-routing/TODO.md
index 65b2e273..6ac4cc98 100644
--- a/.agents/plans/cli-device-routing/TODO.md
+++ b/.agents/plans/cli-device-routing/TODO.md
@@ -434,3 +434 @@ build or general model/backend grant. Historical failures keep their source scop
-## Uncommitted
-
-### UX-P — native TUI interaction and visual polish (26 Nov decision)
+### UX-P — native TUI interaction and visual polish (26 Nov; accepted at ad7a4178)
@@ -466 +464 @@ Retain accepted UX-1–10 ownership and keyboard behavior.
-- [ ] UX-P4: complete ergonomics, actual PTY and accepted real cached
+- [x] UX-P4: complete ergonomics, actual PTY and accepted real cached
@@ -491 +489 @@ Retain accepted UX-1–10 ownership and keyboard behavior.
-    full default cargo check passed. Full-GPU lint debt remains separate.
+    full default cargo check passed; final full-default strict lint below.
@@ -497 +495 @@ Retain accepted UX-1–10 ownership and keyboard behavior.
-    and reset on re-registration. GPU placement compile-only. See
+    and reset on re-registration. CUDA placement later qualified below. See
@@ -505,3 +503 @@ Retain accepted UX-1–10 ownership and keyboard behavior.
-  - [ ] Optional deeper exact per-request/phase and individual layer
-    observations; keep profiling explicit and capability-bound.
-  - [ ] P4B comprehensive acceptance: actual combined owner regressions
+  - [x] P4B comprehensive acceptance: actual combined owner regressions
@@ -524,3 +520,4 @@ Retain accepted UX-1–10 ownership and keyboard behavior.
-    timed out during compilation (90s budget, no suites executed). Complete
-    isolated assembled gate pending.
-    Do not close P4B before actual pass.
+    timed out during compilation (90s budget, no suites executed). Final
+    assembled offline private-network gate on ad7a4178 exited 0 (592.3s):
+    2,539 passed, zero failures, 109 explicit skips across 318 Rust suites.
+    Strict lint, feature boundaries, release Web/TUI/installer/scripts pass.
@@ -531,4 +528,4 @@ Retain accepted UX-1–10 ownership and keyboard behavior.
-Deferred public serving until UX-P exit. Optional terminal background
-helper requires dependency/offline approval; conservative Auto fallback
-must remain truthful. No additional theme catalog or persistent layout
-store now.
+UX-P accepted; public serving is the next ordered phase. No physical
+performance comparison or new HIP/Vulkan placement qualification.
+
+## Uncommitted
@@ -537,0 +535,6 @@ store now.
+- [ ] Optional deeper exact per-request/phase and individual layer
+  observations; keep profiling explicit and capability-bound.
+- [ ] Optional terminal background helper only with dependency/offline
+  approval; conservative Auto fallback must remain truthful. No additional
+  theme catalog or persistent layout store now.
+
diff --git a/.agents/plans/cli-ux/LIVE_MODEL_PERFORMANCE.md b/.agents/plans/cli-ux/LIVE_MODEL_PERFORMANCE.md
index 7ab2b2e8..2d52e868 100644
--- a/.agents/plans/cli-ux/LIVE_MODEL_PERFORMANCE.md
+++ b/.agents/plans/cli-ux/LIVE_MODEL_PERFORMANCE.md
@@ -3 +3 @@
-Status: UX-P-M1 and user-directed M2 presentation refinement focused-qualified (26 Nov); deeper capability-bound profiling and assembled P4B pending.
+Status: UX-P-M1 and user-directed M2 presentation refinement accepted with assembled P4B (26 Nov; ad7a4178). Optional deeper capability-bound profiling remains open.
@@ -166 +166 @@ strings, paths, token IDs, credentials or diagnostic history in samples.
-- [ ] UX-P4B comprehensive isolated workspace gate on assembled source,
+- [x] UX-P4B comprehensive isolated workspace gate on assembled source,
@@ -173,3 +173,4 @@ strings, paths, token IDs, credentials or diagnostic history in samples.
-  Web isolation preserved. Complete isolated gate pending.
-  No performance comparison claim from these UX
-  observations. Public serving remains after UX-P exit.
+  Web isolation preserved. Final assembled private-network gate exited 0
+  on ad7a4178: 2,539 passed, zero failures, 109 explicit skips across 318 Rust suites;
+  release Web/TUI/installer/scripts passed. No performance comparison
+  claim from these UX observations. Public serving is next.
diff --git a/.agents/plans/cli-ux/TUI_POLISH_AND_MOUSE_INTERACTION.md b/.agents/plans/cli-ux/TUI_POLISH_AND_MOUSE_INTERACTION.md
index b4cdd9dd..6b9d3349 100644
--- a/.agents/plans/cli-ux/TUI_POLISH_AND_MOUSE_INTERACTION.md
+++ b/.agents/plans/cli-ux/TUI_POLISH_AND_MOUSE_INTERACTION.md
@@ -7 +7 @@ Decision: 26 Nov 2026 — polish before public serving
-Status: UX-P1/P2/P3 and UX-P4A1/P4A2 focused-qualified; user follow-up [live model performance](LIVE_MODEL_PERFORMANCE.md) before P4B full acceptance (26 Nov).
+Status: UX-P accepted (26 Nov; assembled gate on ad7a4178). User-directed [live model performance](LIVE_MODEL_PERFORMANCE.md) presentation refinement accepted; optional deeper profiling remains separate.
@@ -677 +677 @@ not mark an untested or interrupted phase accepted.
-  - [ ] P4B full offline isolated workspace gate and final real
+  - [x] P4B full offline isolated workspace gate and final real
@@ -681,6 +681,6 @@ not mark an untested or interrupted phase accepted.
-    TUI/Web shell and strict lint passed (26 Nov). Complete isolated gate
-    still required; see current handoff. Review
-    remaining compact affordances and rendered terminal legibility;
-    no golden matrix. Native Auto remains conservative hint fallback,
-    active probe deferred unless separately approved. No public
-    serving or performance acceptance before P4 exit.
+    TUI/Web shell and strict lint passed (26 Nov). Final assembled gate
+    on ad7a4178 exited 0 (592.3s): 2,539 passed, zero failures, 109 explicit skips across 318 Rust suites.
+    Release PTY shell/settings/appearance/mouse/action/editor/20x7 compact
+    and Web control/settings, installer and scripts passed. Native Auto
+    remains conservative hint fallback; active probe deferred unless approved.
+    UX-P accepted; expanded public serving is next, not yet accepted.
diff --git a/.agents/plans/serving-and-speculation/NEXT_ITERATION_OPENAI_COMPATIBLE_SERVING.md b/.agents/plans/serving-and-speculation/NEXT_ITERATION_OPENAI_COMPATIBLE_SERVING.md
index 7d41962a..1d380936 100644
--- a/.agents/plans/serving-and-speculation/NEXT_ITERATION_OPENAI_COMPATIBLE_SERVING.md
+++ b/.agents/plans/serving-and-speculation/NEXT_ITERATION_OPENAI_COMPATIBLE_SERVING.md
@@ -9 +9 @@ Older ordering below is context; valid owner constraints remain authoritative.
-Status: **design only / dormant**.
+Status: **design only; OA-1 is next** (26 Nov, UX-P accepted).
@@ -12 +12,2 @@ Status: **design only / dormant**.
-must be accepted before this iteration begins.
+was the prerequisite. The assembled P4B gate passed on ad7a4178;
+public serving is unblocked. No new public API acceptance is implied.
~~~~

### D.6 — feat(ux): choose model backend and device from Models

Commit: `240bd05e1e0b51a96616c4c08ef311ef53e53772`

Original commit message:

~~~~text
feat(ux): choose model backend and device from Models

Keep profile-default automatic routing and expose optional model-bound execution choices over the original resolver in native and Web. Qualify explicit cached CPU Load/Run and the complete offline workspace gate.
~~~~

Original checkpoint narrative at this commit (historical):

~~~~markdown
### Current checkpoint — model-local execution selector accepted (26 Nov)

User-directed follow-up to the accepted UX-P4B checkpoint: one quiet
Execution row/selector on the selected cached model in native TUI and Web.
Run, Load and Load for Serve use the same typed execution intent. The default
inherits the active profile, including its device policy; absent explicit
profile settings, routing remains automatic. An explicit backend without a
device resets the inherited device allowlist to automatic; an exact physical
device is a single-device override. Choices are temporary, model-bound UI
drafts, not profile edits or changes to a running registration. Native keeps
one exact-digest draft; Web selection is keyed by digest. The existing Devices
workflow and advanced profile policy are preserved.

Shared cold discovery projects compiled backend intents and observed compatible
physical device identities. These are choices, not source fit guarantees.
No new runtime/admission authority, device initialization on discovery, silent
fallback, or GPU execution claim. The original resolver and fit planner
validate every actual request; explicit unsupported combinations still refuse.

Focused Web-only and combined CPU/TUI/Web/serve/acquisition strict lint passed;
Rust 67/0/3, JS 5/0 and offline local asset build passed. Real cached Qwen3.5-0.8B
native/Firefox/HTTP lifecycle 1/0/0 (57.08s), including explicit CPU selection
from both interfaces, exact output and existing admission, listener and
retirement contracts. Final complete offline private-network gate exited 0
(301.6s): 2,543 passed, zero failures, 109 explicit skips across 318 Rust suites;
release TUI/Web, installer and scripts passed. Logs:
`target/ux-execution-final-retry.log`, `target/ux-model-execution-assembled.log`.
No new physical GPU placement or performance qualification. Public serving
OA-1 under §2.3 remains the next coherent unit; optional deeper profiling
remains separate.
~~~~

Changed paths:

~~~~text
M	.agents/docs/CURRENT_BASELINE.md
M	.agents/docs/CURRENT_HANDOFF.md
M	.agents/plans/FULL_IMPLEMENTATION_PLAN.md
M	.agents/plans/cli-device-routing/TODO.md
M	.agents/plans/cli-ux/TUI_POLISH_AND_MOUSE_INTERACTION.md
M	crates/zerogravity-cli/src/control_host.rs
A	crates/zerogravity-cli/src/control_host/execution.rs
M	crates/zerogravity-cli/src/control_host/work.rs
M	crates/zerogravity-cli/src/tui.rs
M	crates/zerogravity-cli/src/tui/choice.rs
M	crates/zerogravity-cli/src/tui/controls.rs
M	crates/zerogravity-cli/src/tui/library.rs
M	crates/zerogravity-cli/src/tui/view.rs
M	crates/zerogravity-cli/src/web/actions.rs
M	crates/zerogravity-cli/src/web/wire.rs
M	crates/zerogravity-cli/tests/fixtures/tui_shell.py
M	crates/zerogravity-cli/web/dist/app.js
M	crates/zerogravity-cli/web/dist/style.css
M	crates/zerogravity-cli/web/src/browser-smoke.mjs
M	crates/zerogravity-cli/web/src/library.jsx
M	crates/zerogravity-cli/web/src/main.jsx
M	crates/zerogravity-cli/web/src/styles.css
~~~~

Exact tracked engineering prose delta (zero-context Git diff; full surrounding prose at the cited revision):

~~~~diff
diff --git a/.agents/docs/CURRENT_BASELINE.md b/.agents/docs/CURRENT_BASELINE.md
index 5ff3d44a..7a7ff489 100644
--- a/.agents/docs/CURRENT_BASELINE.md
+++ b/.agents/docs/CURRENT_BASELINE.md
@@ -7,0 +8,10 @@ historical comparison numbers into new implementation decisions.
+## 26 Nov — model-local execution selector accepted
+
+Native/Web cached Models share a typed backend/device override, inheriting the
+profile by default and preserving original fit/refusal authority. Explicit
+backend without device selects automatic devices; exact physical choice pins
+one device. Real cached CPU native/Firefox/HTTP lifecycle 1/0/0 (57.08s), focused
+Rust 67/0/3, JS 5/0, strict lint; assembled offline private-network gate exited
+0 (301.6s), 2,543/0/109 across 318 Rust suites plus release TUI/Web/installer and
+scripts. No new GPU execution or performance qualification. OA-1 next.
+
diff --git a/.agents/docs/CURRENT_HANDOFF.md b/.agents/docs/CURRENT_HANDOFF.md
index d8308fbf..f209baa3 100644
--- a/.agents/docs/CURRENT_HANDOFF.md
+++ b/.agents/docs/CURRENT_HANDOFF.md
@@ -38 +38,32 @@ reusable work and measured history, not precedence over this order.
-### Current checkpoint — UX-P4B accepted; public serving next (26 Nov)
+### Current checkpoint — model-local execution selector accepted (26 Nov)
+
+User-directed follow-up to the accepted UX-P4B checkpoint: one quiet
+Execution row/selector on the selected cached model in native TUI and Web.
+Run, Load and Load for Serve use the same typed execution intent. The default
+inherits the active profile, including its device policy; absent explicit
+profile settings, routing remains automatic. An explicit backend without a
+device resets the inherited device allowlist to automatic; an exact physical
+device is a single-device override. Choices are temporary, model-bound UI
+drafts, not profile edits or changes to a running registration. Native keeps
+one exact-digest draft; Web selection is keyed by digest. The existing Devices
+workflow and advanced profile policy are preserved.
+
+Shared cold discovery projects compiled backend intents and observed compatible
+physical device identities. These are choices, not source fit guarantees.
+No new runtime/admission authority, device initialization on discovery, silent
+fallback, or GPU execution claim. The original resolver and fit planner
+validate every actual request; explicit unsupported combinations still refuse.
+
+Focused Web-only and combined CPU/TUI/Web/serve/acquisition strict lint passed;
+Rust 67/0/3, JS 5/0 and offline local asset build passed. Real cached Qwen3.5-0.8B
+native/Firefox/HTTP lifecycle 1/0/0 (57.08s), including explicit CPU selection
+from both interfaces, exact output and existing admission, listener and
+retirement contracts. Final complete offline private-network gate exited 0
+(301.6s): 2,543 passed, zero failures, 109 explicit skips across 318 Rust suites;
+release TUI/Web, installer and scripts passed. Logs:
+`target/ux-execution-final-retry.log`, `target/ux-model-execution-assembled.log`.
+No new physical GPU placement or performance qualification. Public serving
+OA-1 under §2.3 remains the next coherent unit; optional deeper profiling
+remains separate.
+
+### Prior checkpoint — UX-P4B accepted; public serving next (26 Nov)
diff --git a/.agents/plans/FULL_IMPLEMENTATION_PLAN.md b/.agents/plans/FULL_IMPLEMENTATION_PLAN.md
index 8d740731..7ad7f52f 100644
--- a/.agents/plans/FULL_IMPLEMENTATION_PLAN.md
+++ b/.agents/plans/FULL_IMPLEMENTATION_PLAN.md
@@ -756,0 +757,5 @@ UX-1–10 remains accepted; this is additional presentation work.
+  User follow-up: model-local native/Web Execution selector accepted; Run,
+  Load and Load for Serve inherit profile by default, with optional explicit
+  backend/device draft over the original resolver. Actual cached CPU lifecycle
+  and assembled gate 2,543/0/109 (318 suites) pass; OA-1 remains next.
+
diff --git a/.agents/plans/cli-device-routing/TODO.md b/.agents/plans/cli-device-routing/TODO.md
index 6ac4cc98..f318a656 100644
--- a/.agents/plans/cli-device-routing/TODO.md
+++ b/.agents/plans/cli-device-routing/TODO.md
@@ -527,0 +528,9 @@ Retain accepted UX-1–10 ownership and keyboard behavior.
+- [x] User follow-up: model-local Execution selector in native and Web cached
+  Models. Default inherits the active profile; explicit backend can select an
+  automatic or exact observed device. Run/Load/Load for Serve share the draft;
+  no profile writes, cross-model override, or new routing owner. Original
+  fit/refusal remains authoritative; stale explicit device does not fall back.
+  Focused 67/0/3, JS 5/0, real cached CPU lifecycle 1/0/0 (57.08s), full
+  assembled private-network gate 2,543/0/109 (318 suites, 301.6s), release
+  native/Web/installer/scripts passed. No new GPU execution claim.
+
diff --git a/.agents/plans/cli-ux/TUI_POLISH_AND_MOUSE_INTERACTION.md b/.agents/plans/cli-ux/TUI_POLISH_AND_MOUSE_INTERACTION.md
index 6b9d3349..03adaa85 100644
--- a/.agents/plans/cli-ux/TUI_POLISH_AND_MOUSE_INTERACTION.md
+++ b/.agents/plans/cli-ux/TUI_POLISH_AND_MOUSE_INTERACTION.md
@@ -17,0 +18,10 @@ Complete it before §2.3 public serving/ordinary benchmarks.
+**26 Nov Models execution follow-up (accepted).** One object-local Execution
+row in native details (also context action/Alt+e), and one inline Web select.
+Default inherits the profile, otherwise explicit backend and automatic or
+exact observed physical device; Run/Load/Load for Serve consume the same
+model-bound temporary draft. No additional mandatory step, persistent profile
+mutation, active registration change or advanced policy duplication. The
+existing resolver validates all fit and compatibility; cold inventory is not
+an execution guarantee. Real CPU lifecycle and complete assembled gate passed;
+see the owner TODO and current handoff for evidence.
+
~~~~

### D.7 — feat(serving): own bounded asynchronous HTTP transport

Commit: `fbb4ab1dd72c4fa4e52b3eaa165d7ca2211ec114`

Original commit message:

~~~~text
feat(serving): own bounded asynchronous HTTP transport

Preserve original request admission, charged event queues, SSE token delivery, and exact listener retirement under persistent Hyper connections. Qualify real cached CLI/TUI/Web and SDK consumers and assembled private-network workspace gate. Retain explicit remaining streaming upload and upgrade work.
~~~~

Original checkpoint narrative at this commit (historical):

~~~~markdown
### Current checkpoint — OA-1A bounded asynchronous text transport (26 Nov)

Source authoring began from clean signed `240bd05e` on L5. Replaced the
bootstrap per-connection blocking HTTP/1 server with pinned cached Tokio
1.53.2, Hyper 1.13.0 and hyper-util 0.1.21 confined to the serving crate.
One owned current-thread runtime, bounded connection tasks, persistent and
pipelined HTTP/1.1, charged fixed-length JSON ingress, direct SSE, connection
output budget and stalled-client deadlines. The original charged event queue
now offers standard-library waker readiness; no new scheduler/runtime or
model owner. Stop joins protocol-owned tasks and leaves SDK caller-owned
runtime/other work intact. Existing semantic routing, auth, ingress limits,
errors/diagnostic redaction, cancellation and original output owners remain.
A bounded duplicate socket observes EOF even behind a buffered pipeline.
Opt-in token timing measures actual completed SSE payload handoff to the
socket, not client receipt. External runtime headroom is a conservative
allowance, not an exact allocator census.

Hyper owns wire parsing: malformed frames can receive protocol-level 4xx
rather than application JSON; identical repeated Content-Length may be
normalized by Hyper, conflicting lengths remain rejected. Chunked requests
and Expect remain refused; no TLS, HTTP/2, media, WebSocket or general
compatibility claim. No downloads or new physical GPU/performance evidence.

Focused strict lint, serving 25/0/0, engine 110/0/0; actual approved cached CPU
CLI/native/Firefox listener lifecycle 1/0/0 (55.52s) including model list and
both alias completions on one persistent connection, exact output equality,
and original owner retirement; actual Rust SDK shared model/listener
lifecycle 1/0/0 (8.83s). Complete assembled private-network gate exited
0 (819.3s): 2,538 passed, zero failures, 109 explicit skips across 318
Rust suites; release TUI/Web, installer and scripts passed.
Logs: `target/oa1-final-focused.log`, `target/oa1-real-consumers.log`,
`target/oa1-assembled.log`.

Next: OA-1B bounded streaming upload,
multipart primitives and WebSocket foundation using approved dependencies.
Keep unsupported endpoints explicitly refused and no separate admission
authority. OA-2 semantic request/model-template boundary follows complete
OA-1. UX-P stays accepted; performance roadmap remains unchanged.
~~~~

Changed paths:

~~~~text
M	.agents/docs/CURRENT_BASELINE.md
M	.agents/docs/CURRENT_HANDOFF.md
M	.agents/plans/FULL_IMPLEMENTATION_PLAN.md
M	.agents/plans/README.md
M	.agents/plans/TODO.md
M	.agents/plans/serving-and-speculation/NEXT_ITERATION_OPENAI_COMPATIBLE_SERVING.md
M	.agents/plans/serving-and-speculation/TODO.md
M	Cargo.lock
M	crates/zerogravity-cli/tests/run_unified.rs
M	crates/zerogravity-engine/src/router/event_channel.rs
M	crates/zerogravity-engine/src/router/token_metrics.rs
M	crates/zerogravity-serving/Cargo.toml
M	crates/zerogravity-serving/src/http.rs
A	crates/zerogravity-serving/src/http/body.rs
M	crates/zerogravity-serving/src/http/shutdown_tests.rs
A	crates/zerogravity-serving/src/http/transport.rs
M	crates/zerogravity-serving/src/ingress.rs
M	crates/zerogravity-serving/src/lib.rs
M	crates/zerogravity-serving/src/server.rs
M	crates/zerogravity-serving/src/server/runtime_tests.rs
M	crates/zerogravity-serving/src/server/tests.rs
~~~~

Exact tracked engineering prose delta (zero-context Git diff; full surrounding prose at the cited revision):

~~~~diff
diff --git a/.agents/docs/CURRENT_BASELINE.md b/.agents/docs/CURRENT_BASELINE.md
index 7a7ff489..28099cda 100644
--- a/.agents/docs/CURRENT_BASELINE.md
+++ b/.agents/docs/CURRENT_BASELINE.md
@@ -7,0 +8,14 @@ historical comparison numbers into new implementation decisions.
+## 26 Nov — OA-1A bounded async text transport
+
+Tokio/Hyper persistent HTTP/1.1 and bounded connection tasks now use the
+original charged event/admission owner. Focused serving 25/0/0, engine
+110/0/0, strict lint; actual cached CPU CLI/native/Firefox lifecycle 1/0/0
+(55.52s) with persistent model listing and exact alias completions, SDK
+listener/owner lifecycle 1/0/0 (8.83s). Assembled private-network gate
+passed (819.3s), 2,538/0/109 across 318 suites plus release TUI/Web,
+installer and scripts.
+No new numerical, GPU execution or performance comparison.
+Remaining OA-1B: streaming uploads, multipart and WebSocket foundation,
+then OA-2. Logs: target/oa1-final-focused.log,
+target/oa1-real-consumers.log, target/oa1-assembled.log.
+
diff --git a/.agents/docs/CURRENT_HANDOFF.md b/.agents/docs/CURRENT_HANDOFF.md
index f209baa3..60c59495 100644
--- a/.agents/docs/CURRENT_HANDOFF.md
+++ b/.agents/docs/CURRENT_HANDOFF.md
@@ -9,2 +9 @@ Read §§0–13 for historical design and gate context. The
-is superseded by the accepted checkpoint below; public serving OA-1
-is the next unit.
+is historical; the active public serving checkpoint below owns continuation.
@@ -38 +37,40 @@ reusable work and measured history, not precedence over this order.
-### Current checkpoint — model-local execution selector accepted (26 Nov)
+### Current checkpoint — OA-1A bounded asynchronous text transport (26 Nov)
+
+Source authoring began from clean signed `240bd05e` on L5. Replaced the
+bootstrap per-connection blocking HTTP/1 server with pinned cached Tokio
+1.53.2, Hyper 1.13.0 and hyper-util 0.1.21 confined to the serving crate.
+One owned current-thread runtime, bounded connection tasks, persistent and
+pipelined HTTP/1.1, charged fixed-length JSON ingress, direct SSE, connection
+output budget and stalled-client deadlines. The original charged event queue
+now offers standard-library waker readiness; no new scheduler/runtime or
+model owner. Stop joins protocol-owned tasks and leaves SDK caller-owned
+runtime/other work intact. Existing semantic routing, auth, ingress limits,
+errors/diagnostic redaction, cancellation and original output owners remain.
+A bounded duplicate socket observes EOF even behind a buffered pipeline.
+Opt-in token timing measures actual completed SSE payload handoff to the
+socket, not client receipt. External runtime headroom is a conservative
+allowance, not an exact allocator census.
+
+Hyper owns wire parsing: malformed frames can receive protocol-level 4xx
+rather than application JSON; identical repeated Content-Length may be
+normalized by Hyper, conflicting lengths remain rejected. Chunked requests
+and Expect remain refused; no TLS, HTTP/2, media, WebSocket or general
+compatibility claim. No downloads or new physical GPU/performance evidence.
+
+Focused strict lint, serving 25/0/0, engine 110/0/0; actual approved cached CPU
+CLI/native/Firefox listener lifecycle 1/0/0 (55.52s) including model list and
+both alias completions on one persistent connection, exact output equality,
+and original owner retirement; actual Rust SDK shared model/listener
+lifecycle 1/0/0 (8.83s). Complete assembled private-network gate exited
+0 (819.3s): 2,538 passed, zero failures, 109 explicit skips across 318
+Rust suites; release TUI/Web, installer and scripts passed.
+Logs: `target/oa1-final-focused.log`, `target/oa1-real-consumers.log`,
+`target/oa1-assembled.log`.
+
+Next: OA-1B bounded streaming upload,
+multipart primitives and WebSocket foundation using approved dependencies.
+Keep unsupported endpoints explicitly refused and no separate admission
+authority. OA-2 semantic request/model-template boundary follows complete
+OA-1. UX-P stays accepted; performance roadmap remains unchanged.
+
+### Prior checkpoint — model-local execution selector accepted (26 Nov)
diff --git a/.agents/plans/FULL_IMPLEMENTATION_PLAN.md b/.agents/plans/FULL_IMPLEMENTATION_PLAN.md
index 7ad7f52f..f1135c17 100644
--- a/.agents/plans/FULL_IMPLEMENTATION_PLAN.md
+++ b/.agents/plans/FULL_IMPLEMENTATION_PLAN.md
@@ -766,0 +767,5 @@ protocol or backend performance work was admitted before this bounded exit.
+OA-1A bounded async text transport is accepted (complete isolated gate
+2,538/0/109, 318 suites; real cached CLI/TUI/Web and SDK passed). The remaining OA-1B streaming upload/multipart/WebSocket
+foundation precedes the OA-2 semantic request/model-template boundary.
+Preserve the original runtime, admission and listener owner.
+
diff --git a/.agents/plans/README.md b/.agents/plans/README.md
index c10b28bf..3c27f15c 100644
--- a/.agents/plans/README.md
+++ b/.agents/plans/README.md
@@ -19,6 +19,5 @@ UX-P1/P2/P3/P4A, full default/error follow-up and live model metrics.
-Use the [standalone P4B prompt](../docs/NEXT_AGENT_2026-11-26_UX_P4B.md).
-Next required: P4B assembled acceptance in
-[UX-P polish](cli-ux/TUI_POLISH_AND_MOUSE_INTERACTION.md) and
-[live performance](cli-ux/LIVE_MODEL_PERFORMANCE.md); deeper M2 inspection
-is optional and capability-bound. Do not start public serving before
-UX-P exit. The comprehensive handoff §10.2 records known full-default
+UX-P and the model-local execution selector are accepted. The
+[standalone P4B prompt](../docs/NEXT_AGENT_2026-11-26_UX_P4B.md) is historical.
+Current unit: [OA-1 public serving transport](serving-and-speculation/NEXT_ITERATION_OPENAI_COMPATIBLE_SERVING.md#oa-1--async-transport-substrate).
+Read the live handoff for the assembled gate and remaining OA-1 work;
+deeper M2 inspection stays optional and capability-bound. The comprehensive handoff §10.2 records known full-default
diff --git a/.agents/plans/TODO.md b/.agents/plans/TODO.md
index 2fce1882..5c55b030 100644
--- a/.agents/plans/TODO.md
+++ b/.agents/plans/TODO.md
@@ -236 +236 @@ Final assembled private-network gate on ad7a4178 passed 2,539 passed, zero failu
-public serving OA-1, then ordinary benchmarks and ordered performance phases.
+public serving OA-1 (bounded async text substrate accepted, remaining streaming ingress/multipart/WebSocket foundation OA-1B), then ordinary benchmarks and ordered performance phases.
diff --git a/.agents/plans/serving-and-speculation/NEXT_ITERATION_OPENAI_COMPATIBLE_SERVING.md b/.agents/plans/serving-and-speculation/NEXT_ITERATION_OPENAI_COMPATIBLE_SERVING.md
index 1d380936..d35b2777 100644
--- a/.agents/plans/serving-and-speculation/NEXT_ITERATION_OPENAI_COMPATIBLE_SERVING.md
+++ b/.agents/plans/serving-and-speculation/NEXT_ITERATION_OPENAI_COMPATIBLE_SERVING.md
@@ -9 +9 @@ Older ordering below is context; valid owner constraints remain authoritative.
-Status: **design only; OA-1 is next** (26 Nov, UX-P accepted).
+Status: **OA-1A bounded async text transport accepted** (26 Nov). OA-1B streaming upload/multipart/WebSocket foundation remains open; OA-2 and later remain ordered.
@@ -1139,0 +1140,38 @@ Do not add new model semantics yet.
+#### OA-1A — bounded async text transport (26 Nov)
+
+The ordinary server owns one current-thread Tokio runtime and bounded Hyper
+HTTP/1 connection tasks. Pinned, locally cached transport dependencies are
+confined to the serving crate; the engine's charged, fixed-slot event channel
+adds only standard-library async receiver readiness. No second queue, model
+owner or inference thread per client. Successive/pipelined requests use the
+same connection; stop closes ingress, cancels protocol-owned receipts, and
+joins every task without stopping caller-owned SDK work. The existing auth,
+request parsing, body admission, JSON/SSE responses and diagnostic redaction
+remain the semantic boundary.
+
+Explicit text policy: 8 KiB header and 32-header caps, ten-second ingress
+deadlines, bounded connection count, 256 requests per connection, cumulative
+wire output budget and 30-second stalled-write bound. Connection and server
+runtime storage each receive conservative 64 KiB admission allowances,
+not an exact heap census; existing application body/output owners stay charged.
+A bounded read-only duplicate socket observes peer closure even behind a
+buffered pipeline. Hyper owns framing; invalid framing may produce a
+protocol-level 4xx rather than application JSON. Hyper normalizes identical
+repeated Content-Length; conflicting lengths and all chunked/Expect input
+remain refused in this slice. Ordinary output is directly polled from the
+original channel. Opt-in token observations measure completed SSE payload
+handoff via actual socket writes, never client receipt or body polling.
+
+OA-1A is not a general public API, TLS, HTTP/2, or external network exposure
+claim. No new model semantics or GPU performance claim.
+
+#### OA-1B — remaining bounded ingress and upgrade primitives
+
+Finish charged unknown-length streaming request bodies, multipart primitives
+and WebSocket foundations before closing OA-1. Use approved mature cached
+dependencies where available (multipart dependency availability must be
+checked; no automatic downloads). Keep auth, independent route/tenant scope,
+size/deadline limits, cancellation and owned shutdown; no stub modality
+endpoints or Responses translation hacks. Qualify new boundaries independently
+and preserve OA-1A's original text behavior.
+
diff --git a/.agents/plans/serving-and-speculation/TODO.md b/.agents/plans/serving-and-speculation/TODO.md
index 9441b4ee..58e1322d 100644
--- a/.agents/plans/serving-and-speculation/TODO.md
+++ b/.agents/plans/serving-and-speculation/TODO.md
@@ -48,0 +49,16 @@ Contracts: [joint serving/goodput](../../docs/JOINT_CONCURRENT_SERVING_OPTIMIZAT
+### OA-1 — async serving transport (26 Nov; after UX-P)
+
+- [x] OA-1A: mature bounded text HTTP/1 substrate over original service,
+  admission and event owners. Focused serving 25/0/0 and engine 110/0/0,
+  strict lint; actual cached CPU CLI/native/Firefox listener lifecycle with
+  model list and both alias completions on one persistent TCP connection
+  passed (55.52s). Real SDK listener/owner lifecycle passed (8.83s).
+  Assembled offline private-network gate passed (819.3s), 2,538/0/109 across
+  318 Rust suites, release TUI/Web, installer and scripts.
+  No numerical/GPU performance comparison. See the
+  [OA-1 design](NEXT_ITERATION_OPENAI_COMPATIBLE_SERVING.md#oa-1--async-transport-substrate).
+- [ ] OA-1B: bounded charged unknown-length/streaming ingress, multipart
+  primitives and WebSocket foundation, with approved dependencies and
+  explicit lifetime/authorization tests. No modality endpoints until their
+  real contracts. OA-2 remains ordered after OA-1.
+
~~~~

### D.8 — feat(serving): own bounded ingress and websocket continuations

Commit: `3156ce45750e838c7ace72dc7d8c0c06aa762c3b`

Original commit message:

~~~~text
feat(serving): own bounded ingress and websocket continuations

Preserve fixed-length text framing, stage decoded unknown-length bodies under the original ledger, and retain the connection owner and wire budget across admitted WebSocket upgrades. Qualify actual socket lifetimes and cached CLI/TUI/Web/SDK consumers under the complete isolated gate. Public chunked and multipart activation remain blocked.
~~~~

Original checkpoint narrative at this commit (historical):

~~~~markdown
### Current checkpoint — OA-1B bounded ingress and owned WebSocket primitives (26 Nov)

Source authoring began from clean signed `fbb4ab1d` on L5. Migrated existing
fixed-length HTTP text ingress to the shared decoded-body reader. One
absolute deadline and cumulative byte cap; exact charged known-length
backing or charged geometric unknown-length staging (old and new leases
overlap until old allocation drops). Actual Hyper TCP fixtures qualify
fixed/unknown/empty/exact bodies, overflow, trailers, deadline and ledger
retirement. Public text continues to reject all chunked/Expect inputs.

Cached tokio-tungstenite 0.29.0 enables a private adapter-admitted upgrade
continuation after existing authorization and handshake validation.
Upgraded socket retains the original connection task, slot, wire budget,
peer-close detection and joined shutdown. Bounded frame/reassembly and
direct deadline-bounded message IO, with separate charged transport
allowance. Actual internal test adapter verifies authorization/refusal,
oversize frame and fragmented message, original cumulative wire budget,
text/binary/ping/pong, close, abrupt disconnect, listener shutdown and
ledger retirement. No real Realtime adapter is installed; ordinary server
and public capability checks still refuse. No new model semantics, model
owner, scheduler, runtime, TLS or external exposure.

Focused isolated serving 31/0/0 and strict lint passed. Actual cached
CPU CLI/native/Firefox lifecycle 1/0/0 (60.72s) and SDK shared model/
listener lifecycle 1/0/0 (10.33s). Full assembled private-network gate
passed (466.3s): 2,544/0/109 across 318 Rust suites, release TUI/Web,
installer and scripts. Logs: `target/oa1b-focused-final.log`,
`target/oa1b-real-consumers.log`, `target/oa1b-assembled.log`.

Remaining OA-1B activation is blocked by two offline facts: Hyper 1.13.0
normalizes TE+CL by discarding Content-Length before the application
receives headers (connection close alone is not strict refusal), and no
mature multipart decoder including multer is in the local approved Cargo
cache. Preserve strict public framing; seek a mature parser/approved
policy resolution and authorized pinned multipart acquisition, then
qualify bounded streaming per-part and aggregate ownership. No automatic
downloads or duplicate parser. OA-2 follows complete OA-1; no unsupported
media/Realtime endpoints.
~~~~

Changed paths:

~~~~text
M	.agents/docs/CURRENT_BASELINE.md
M	.agents/docs/CURRENT_HANDOFF.md
M	.agents/plans/FULL_IMPLEMENTATION_PLAN.md
M	.agents/plans/TODO.md
M	.agents/plans/serving-and-speculation/NEXT_ITERATION_OPENAI_COMPATIBLE_SERVING.md
M	.agents/plans/serving-and-speculation/TODO.md
M	Cargo.lock
M	crates/zerogravity-serving/Cargo.toml
M	crates/zerogravity-serving/src/http.rs
A	crates/zerogravity-serving/src/http/ingress_body.rs
A	crates/zerogravity-serving/src/http/ingress_body/tests.rs
M	crates/zerogravity-serving/src/http/transport.rs
A	crates/zerogravity-serving/src/http/websocket.rs
M	crates/zerogravity-serving/src/server.rs
A	crates/zerogravity-serving/src/server/websocket_tests.rs
~~~~

Exact tracked engineering prose delta (zero-context Git diff; full surrounding prose at the cited revision):

~~~~diff
diff --git a/.agents/docs/CURRENT_BASELINE.md b/.agents/docs/CURRENT_BASELINE.md
index 28099cda..75812606 100644
--- a/.agents/docs/CURRENT_BASELINE.md
+++ b/.agents/docs/CURRENT_BASELINE.md
@@ -7,0 +8,13 @@ historical comparison numbers into new implementation decisions.
+## 26 Nov — OA-1B bounded ingress and owned WebSocket primitives
+
+Shared charged decoded-body staging and owned WebSocket continuation
+implemented; isolated focused serving 31/0/0 and strict lint passed.
+Unknown-length and internal adapter transport qualified on actual TCP;
+ordinary public text framing and unsupported Realtime refusal preserved.
+No new physical GPU execution or performance evidence. Actual cached CPU
+CLI/native/Firefox 1/0/0 (60.72s), SDK 1/0/0 (10.33s); full isolated
+gate 2,544/0/109 (318 Rust suites, 466.3s), release TUI/Web, installer
+and scripts passed. Remaining OA-1B strict
+mixed-framing resolution and approved multipart decoder acquisition;
+OA-2 follows complete OA-1.
+
diff --git a/.agents/docs/CURRENT_HANDOFF.md b/.agents/docs/CURRENT_HANDOFF.md
index 60c59495..3737e7db 100644
--- a/.agents/docs/CURRENT_HANDOFF.md
+++ b/.agents/docs/CURRENT_HANDOFF.md
@@ -37 +37,40 @@ reusable work and measured history, not precedence over this order.
-### Current checkpoint — OA-1A bounded asynchronous text transport (26 Nov)
+### Current checkpoint — OA-1B bounded ingress and owned WebSocket primitives (26 Nov)
+
+Source authoring began from clean signed `fbb4ab1d` on L5. Migrated existing
+fixed-length HTTP text ingress to the shared decoded-body reader. One
+absolute deadline and cumulative byte cap; exact charged known-length
+backing or charged geometric unknown-length staging (old and new leases
+overlap until old allocation drops). Actual Hyper TCP fixtures qualify
+fixed/unknown/empty/exact bodies, overflow, trailers, deadline and ledger
+retirement. Public text continues to reject all chunked/Expect inputs.
+
+Cached tokio-tungstenite 0.29.0 enables a private adapter-admitted upgrade
+continuation after existing authorization and handshake validation.
+Upgraded socket retains the original connection task, slot, wire budget,
+peer-close detection and joined shutdown. Bounded frame/reassembly and
+direct deadline-bounded message IO, with separate charged transport
+allowance. Actual internal test adapter verifies authorization/refusal,
+oversize frame and fragmented message, original cumulative wire budget,
+text/binary/ping/pong, close, abrupt disconnect, listener shutdown and
+ledger retirement. No real Realtime adapter is installed; ordinary server
+and public capability checks still refuse. No new model semantics, model
+owner, scheduler, runtime, TLS or external exposure.
+
+Focused isolated serving 31/0/0 and strict lint passed. Actual cached
+CPU CLI/native/Firefox lifecycle 1/0/0 (60.72s) and SDK shared model/
+listener lifecycle 1/0/0 (10.33s). Full assembled private-network gate
+passed (466.3s): 2,544/0/109 across 318 Rust suites, release TUI/Web,
+installer and scripts. Logs: `target/oa1b-focused-final.log`,
+`target/oa1b-real-consumers.log`, `target/oa1b-assembled.log`.
+
+Remaining OA-1B activation is blocked by two offline facts: Hyper 1.13.0
+normalizes TE+CL by discarding Content-Length before the application
+receives headers (connection close alone is not strict refusal), and no
+mature multipart decoder including multer is in the local approved Cargo
+cache. Preserve strict public framing; seek a mature parser/approved
+policy resolution and authorized pinned multipart acquisition, then
+qualify bounded streaming per-part and aggregate ownership. No automatic
+downloads or duplicate parser. OA-2 follows complete OA-1; no unsupported
+media/Realtime endpoints.
+
+### Prior checkpoint — OA-1A bounded asynchronous text transport (26 Nov)
diff --git a/.agents/plans/FULL_IMPLEMENTATION_PLAN.md b/.agents/plans/FULL_IMPLEMENTATION_PLAN.md
index f1135c17..d9f0ae5a 100644
--- a/.agents/plans/FULL_IMPLEMENTATION_PLAN.md
+++ b/.agents/plans/FULL_IMPLEMENTATION_PLAN.md
@@ -768,2 +768,7 @@ OA-1A bounded async text transport is accepted (complete isolated gate
-2,538/0/109, 318 suites; real cached CLI/TUI/Web and SDK passed). The remaining OA-1B streaming upload/multipart/WebSocket
-foundation precedes the OA-2 semantic request/model-template boundary.
+2,538/0/109, 318 suites; real cached CLI/TUI/Web and SDK passed). OA-1B
+bounded decoded-body staging and owned WebSocket primitives implemented
+and accepted (2,544/0/109 across 318 isolated Rust suites; real cached
+CLI/TUI/Web and SDK passed). Public streaming
+activation requires strict mixed-framing handling and approved multipart
+decoder acquisition. Remaining OA-1B precedes the OA-2 semantic
+request/model-template boundary.
diff --git a/.agents/plans/TODO.md b/.agents/plans/TODO.md
index 5c55b030..c90e62b1 100644
--- a/.agents/plans/TODO.md
+++ b/.agents/plans/TODO.md
@@ -236 +236 @@ Final assembled private-network gate on ad7a4178 passed 2,539 passed, zero failu
-public serving OA-1 (bounded async text substrate accepted, remaining streaming ingress/multipart/WebSocket foundation OA-1B), then ordinary benchmarks and ordered performance phases.
+public serving OA-1 (bounded async text substrate accepted; OA-1B charged staging and owned WebSocket primitives implemented, public streaming/multipart activation remains open), then ordinary benchmarks and ordered performance phases.
diff --git a/.agents/plans/serving-and-speculation/NEXT_ITERATION_OPENAI_COMPATIBLE_SERVING.md b/.agents/plans/serving-and-speculation/NEXT_ITERATION_OPENAI_COMPATIBLE_SERVING.md
index d35b2777..a310afb9 100644
--- a/.agents/plans/serving-and-speculation/NEXT_ITERATION_OPENAI_COMPATIBLE_SERVING.md
+++ b/.agents/plans/serving-and-speculation/NEXT_ITERATION_OPENAI_COMPATIBLE_SERVING.md
@@ -9 +9 @@ Older ordering below is context; valid owner constraints remain authoritative.
-Status: **OA-1A bounded async text transport accepted** (26 Nov). OA-1B streaming upload/multipart/WebSocket foundation remains open; OA-2 and later remain ordered.
+Status: **OA-1A accepted; OA-1B bounded ingress and WebSocket primitives accepted** (26 Nov). Public streaming/multipart activation remains open; OA-2 and later remain ordered.
@@ -1177,0 +1178,32 @@ and preserve OA-1A's original text behavior.
+**OA-1B current implementation (26 Nov).** The existing fixed-length text
+path consumes a bounded decoded-body reader under one absolute deadline.
+The reader supports known and unknown lengths; its contiguous staging owner
+charges exact backing before allocation, bounded geometric growth for unknown
+length, and both allocations during a growth overlap. It rejects excess bytes,
+incomplete bodies and trailers. No extra scheduler, upload store or unbounded
+receive queue. Unknown-length staging is qualified against actual Hyper TCP
+frames in a private fixture; public text still requires Content-Length.
+
+Cached tokio-tungstenite 0.29.0 (handshake only; no TLS/client connector)
+provides the reusable WebSocket primitive. Existing HTTP authentication and
+header/body policy precede handshake validation and adapter admission. A
+successful upgrade retains the same connection task, wire output budget,
+connection slot, peer-close observation and joined shutdown. A separately
+charged bounded frame/message allowance precedes the handshake; message IO
+is direct and deadline-bounded. The private adapter admission callback owns
+any semantic session/state and retained-message charges. Only a test adapter
+is installed today: ordinary servers continue refusing /v1/realtime, and
+RealtimeTransport::require_implemented still refuses. No modality endpoint,
+model support or external exposure is claimed.
+
+Two activation blockers remain: (1) pinned Hyper 1.13.0 removes
+Content-Length when Transfer-Encoding is also present (and disables keep-alive),
+so application-level headers cannot prove strict rejection of mixed framing.
+Retain refusal of all public chunked/Expect ingress until a mature parser
+exposes/rejects the ambiguity, or an explicit reviewed policy change is
+approved. Do not add a second handwritten parser. (2) no mature multipart
+decoder (including multer) is installed in the approved offline Cargo cache.
+Obtain an approved pinned decoder and qualify bounded per-part/aggregate
+streaming admission; no automatic download, homegrown parser, or fake
+multipart/Realtime endpoint. OA-1 stays open and OA-2 remains ordered.
+
diff --git a/.agents/plans/serving-and-speculation/TODO.md b/.agents/plans/serving-and-speculation/TODO.md
index 58e1322d..fd179326 100644
--- a/.agents/plans/serving-and-speculation/TODO.md
+++ b/.agents/plans/serving-and-speculation/TODO.md
@@ -60,4 +60,21 @@ Contracts: [joint serving/goodput](../../docs/JOINT_CONCURRENT_SERVING_OPTIMIZAT
-- [ ] OA-1B: bounded charged unknown-length/streaming ingress, multipart
-  primitives and WebSocket foundation, with approved dependencies and
-  explicit lifetime/authorization tests. No modality endpoints until their
-  real contracts. OA-2 remains ordered after OA-1.
+- [x] OA-1B primitives: shared decoded-body reader with charged fixed/unknown
+  backing, absolute deadline, cap and trailer refusal; current fixed-length
+  text path migrated. Actual private TCP chunked fixture verifies growth,
+  empty/exact/overflow/deadline and full charge retirement. Cached
+  tokio-tungstenite 0.29.0 owned upgrade path over the original connection
+  and wire budget, bounded frames/reassembly/direct message IO, auth and
+  adapter admission before switching. Actual internal test-adapter sockets
+  verify authorization, invalid handshakes, frame/message and wire budgets,
+  slot retention, ping/pong, close, disconnect, stop and ledger retirement.
+  No public Realtime capability or modality route. Focused strict lint and
+  serving 31/0/0 passed. Actual cached CPU CLI/native/Firefox lifecycle
+  1/0/0 (60.72s) and SDK listener/model lifecycle 1/0/0 (10.33s).
+  Full isolated workspace gate passed (466.3s): 2,544/0/109 across
+  318 Rust suites, release TUI/Web, installer and scripts.
+- [ ] OA-1B remaining activation: public chunked/multipart streaming ingress
+  after resolving pinned Hyper mixed Transfer-Encoding/Content-Length
+  normalization (retain strict refusal) and approved acquisition of a
+  mature multipart decoder absent from the offline cache. Qualify bounded
+  per-part/aggregate admission and cleanup. No second HTTP parser, no
+  automatic downloads or fake modality endpoint. OA-2 remains ordered
+  after OA-1.
~~~~

### D.9 — feat(serving): admit strictly framed chunked text requests

Commit: `08a9d30ec5836d6885c937cd60dc0b33471329af`

Original commit message:

~~~~text
feat(serving): admit strictly framed chunked text requests

Use checksum-verified cached Hyper with a narrow server request parser patch, rejecting mixed and repeated length framing before normalization. Preserve charged body staging, existing auth and model ownership. Focused serving and actual cached CLI/TUI/Web/SDK regressions pass; GPU boundary and CPU lint pass, assembled gate timed out during CPU test compilation. Multipart and complete acceptance remain pending.
~~~~

Original checkpoint narrative at this commit (historical):

~~~~markdown
### Current checkpoint — OA-1B public chunked text / strict framing (26 Nov)

Source authoring began from clean synchronized signed `3156ce45` on L5.
Cached Hyper 1.13.0 archive verified against the locked checksum; vendored
under serving with one upstream source file changed (server request parser
and matching assertions, client unchanged), license and provenance retained.
Mixed TE/CL in either order and all repeated CL now fail before normalization
or application dispatch. One mature HTTP parser remains authoritative.
New chunked ingress only on supported POST text routes, single chunked
coding, no trailers/Expect/upgrade. Original auth, charged reader (absolute
deadline, cumulative cap, growth-overlap charge), complete JSON validation,
model lowering, SSE, persistent socket and scheduler ownership preserved.
No new model semantics or modality endpoints.

Focused isolated serving 33/0/0 and strict lint pass: public actual socket
tests cover fixed/chunked, exact cap, UTF-8 split, complete-body admission,
pipeline, malformed/mixed/duplicate framing, unauthorized and oversize,
trailers and partial disconnect ledger retirement. Real cached-model
CLI/TUI/Web (60.52s) and SDK (9.11s) regressions each passed 1/0/0.
The CLI fixture sends both alias completions (fixed and chunked) over one
connection and compares exact model output. Assembled gate passed all
HIP/Vulkan/CUDA compile boundaries, strict full-default lint, GPU feature
boundary tests (26/0/2), source-only qualification, dependency isolation
and strict CPU workspace Clippy. Timed out at 1100s during CPU workspace
test compilation after dependency-path cache invalidation; no CPU test
suites executed. Full assembled gate remains required; retry with the
warmed cache, then finish OA-1B multipart. Vendor source filter/check
updated; Nix syntax passes, exact source-integrity derivation is blocked
by the locked nixpkgs archive unavailable in the private network.
Logs: `target/oa1b-chunked-focused-final.log`,
`target/oa1b-chunked-real-consumers.log`, `target/oa1b-chunked-assembled.log`,
`target/oa1b-chunked-nix-source.log`.

Next OA-1B multipart: `multer 3.1.0` exists in local index, no archive or
source. Cached form-data 0.6.0 rejected after reviewing decoder: truncated
input may loop, invalid delimiter suffix may be treated as EOF. No
downloads or duplicate parser. Obtain approved pinned decoder and qualify
bounded part/aggregate ownership and cleanup, then OA-2. No real
Realtime/model/media or new physical GPU performance claim.
~~~~

Changed paths:

~~~~text
M	.agents/docs/CURRENT_BASELINE.md
M	.agents/docs/CURRENT_HANDOFF.md
M	.agents/plans/FULL_IMPLEMENTATION_PLAN.md
M	.agents/plans/TODO.md
M	.agents/plans/serving-and-speculation/NEXT_ITERATION_OPENAI_COMPATIBLE_SERVING.md
M	.agents/plans/serving-and-speculation/TODO.md
M	Cargo.lock
M	Cargo.toml
M	crates/zerogravity-cli/tests/run_unified.rs
M	crates/zerogravity-serving/src/http/transport.rs
M	crates/zerogravity-serving/src/ingress.rs
M	crates/zerogravity-serving/src/server/tests.rs
A	crates/zerogravity-serving/vendor/hyper/.cargo_vcs_info.json
A	crates/zerogravity-serving/vendor/hyper/Cargo.toml
A	crates/zerogravity-serving/vendor/hyper/Cargo.toml.orig
A	crates/zerogravity-serving/vendor/hyper/LICENSE
A	crates/zerogravity-serving/vendor/hyper/PATCHES.md
A	crates/zerogravity-serving/vendor/hyper/README.md
A	crates/zerogravity-serving/vendor/hyper/src/body/chan.rs
A	crates/zerogravity-serving/vendor/hyper/src/body/incoming.rs
A	crates/zerogravity-serving/vendor/hyper/src/body/length.rs
A	crates/zerogravity-serving/vendor/hyper/src/body/mod.rs
A	crates/zerogravity-serving/vendor/hyper/src/cfg.rs
A	crates/zerogravity-serving/vendor/hyper/src/client/conn/http1.rs
A	crates/zerogravity-serving/vendor/hyper/src/client/conn/http2.rs
A	crates/zerogravity-serving/vendor/hyper/src/client/conn/mod.rs
A	crates/zerogravity-serving/vendor/hyper/src/client/dispatch.rs
A	crates/zerogravity-serving/vendor/hyper/src/client/mod.rs
A	crates/zerogravity-serving/vendor/hyper/src/client/tests.rs
A	crates/zerogravity-serving/vendor/hyper/src/common/buf.rs
A	crates/zerogravity-serving/vendor/hyper/src/common/date.rs
A	crates/zerogravity-serving/vendor/hyper/src/common/either.rs
A	crates/zerogravity-serving/vendor/hyper/src/common/future.rs
A	crates/zerogravity-serving/vendor/hyper/src/common/io/compat.rs
A	crates/zerogravity-serving/vendor/hyper/src/common/io/mod.rs
A	crates/zerogravity-serving/vendor/hyper/src/common/io/rewind.rs
A	crates/zerogravity-serving/vendor/hyper/src/common/lock.rs
A	crates/zerogravity-serving/vendor/hyper/src/common/mod.rs
A	crates/zerogravity-serving/vendor/hyper/src/common/task.rs
A	crates/zerogravity-serving/vendor/hyper/src/common/time.rs
A	crates/zerogravity-serving/vendor/hyper/src/error.rs
A	crates/zerogravity-serving/vendor/hyper/src/ext/h1_reason_phrase.rs
A	crates/zerogravity-serving/vendor/hyper/src/ext/informational.rs
A	crates/zerogravity-serving/vendor/hyper/src/ext/mod.rs
A	crates/zerogravity-serving/vendor/hyper/src/ffi/body.rs
A	crates/zerogravity-serving/vendor/hyper/src/ffi/client.rs
A	crates/zerogravity-serving/vendor/hyper/src/ffi/error.rs
A	crates/zerogravity-serving/vendor/hyper/src/ffi/http_types.rs
A	crates/zerogravity-serving/vendor/hyper/src/ffi/io.rs
A	crates/zerogravity-serving/vendor/hyper/src/ffi/macros.rs
A	crates/zerogravity-serving/vendor/hyper/src/ffi/mod.rs
A	crates/zerogravity-serving/vendor/hyper/src/ffi/task.rs
A	crates/zerogravity-serving/vendor/hyper/src/headers.rs
A	crates/zerogravity-serving/vendor/hyper/src/lib.rs
A	crates/zerogravity-serving/vendor/hyper/src/mock.rs
A	crates/zerogravity-serving/vendor/hyper/src/proto/h1/conn.rs
A	crates/zerogravity-serving/vendor/hyper/src/proto/h1/decode.rs
A	crates/zerogravity-serving/vendor/hyper/src/proto/h1/dispatch.rs
A	crates/zerogravity-serving/vendor/hyper/src/proto/h1/encode.rs
A	crates/zerogravity-serving/vendor/hyper/src/proto/h1/io.rs
A	crates/zerogravity-serving/vendor/hyper/src/proto/h1/mod.rs
A	crates/zerogravity-serving/vendor/hyper/src/proto/h1/role.rs
A	crates/zerogravity-serving/vendor/hyper/src/proto/h2/client.rs
A	crates/zerogravity-serving/vendor/hyper/src/proto/h2/mod.rs
A	crates/zerogravity-serving/vendor/hyper/src/proto/h2/ping.rs
A	crates/zerogravity-serving/vendor/hyper/src/proto/h2/server.rs
A	crates/zerogravity-serving/vendor/hyper/src/proto/h2/upgrade.rs
A	crates/zerogravity-serving/vendor/hyper/src/proto/mod.rs
A	crates/zerogravity-serving/vendor/hyper/src/rt/bounds.rs
A	crates/zerogravity-serving/vendor/hyper/src/rt/io.rs
A	crates/zerogravity-serving/vendor/hyper/src/rt/mod.rs
A	crates/zerogravity-serving/vendor/hyper/src/rt/timer.rs
A	crates/zerogravity-serving/vendor/hyper/src/server/conn/http1.rs
A	crates/zerogravity-serving/vendor/hyper/src/server/conn/http2.rs
A	crates/zerogravity-serving/vendor/hyper/src/server/conn/mod.rs
A	crates/zerogravity-serving/vendor/hyper/src/server/mod.rs
A	crates/zerogravity-serving/vendor/hyper/src/service/http.rs
A	crates/zerogravity-serving/vendor/hyper/src/service/mod.rs
A	crates/zerogravity-serving/vendor/hyper/src/service/service.rs
A	crates/zerogravity-serving/vendor/hyper/src/service/util.rs
A	crates/zerogravity-serving/vendor/hyper/src/trace.rs
A	crates/zerogravity-serving/vendor/hyper/src/upgrade.rs
M	nix/checks.nix
M	nix/source.nix
~~~~

Exact tracked engineering prose delta (zero-context Git diff; full surrounding prose at the cited revision):

~~~~diff
diff --git a/.agents/docs/CURRENT_BASELINE.md b/.agents/docs/CURRENT_BASELINE.md
index 75812606..c71e7030 100644
--- a/.agents/docs/CURRENT_BASELINE.md
+++ b/.agents/docs/CURRENT_BASELINE.md
@@ -7,0 +8,14 @@ historical comparison numbers into new implementation decisions.
+## 26 Nov — OA-1B public chunked text / strict framing
+
+Cached Hyper 1.13.0 server-parser patch resolves mixed/repeated length
+normalization; bounded public chunked JSON now reuses original auth,
+charged staging, SSE and model scheduler. Actual public socket tests
+and strict lint pass (focused 33/0/0); real cached CLI/TUI/Web and SDK
+regressions pass (60.52s and 9.11s). Assembled GPU boundary tests
+26/0/2, strict default/CPU lint and dependency isolation passed; gate
+timed out at 1100s during CPU workspace test compilation, no CPU suites
+executed. Complete assembled acceptance remains pending. Nix vendor source check blocked by uncached locked
+nixpkgs input; syntax passed. No new physical GPU or performance evidence.
+Multipart remains blocked on approved decoder acquisition (cached
+form-data 0.6.0 not suitable). OA-2 follows complete OA-1.
+
diff --git a/.agents/docs/CURRENT_HANDOFF.md b/.agents/docs/CURRENT_HANDOFF.md
index 3737e7db..2c54097e 100644
--- a/.agents/docs/CURRENT_HANDOFF.md
+++ b/.agents/docs/CURRENT_HANDOFF.md
@@ -37 +37,41 @@ reusable work and measured history, not precedence over this order.
-### Current checkpoint — OA-1B bounded ingress and owned WebSocket primitives (26 Nov)
+### Current checkpoint — OA-1B public chunked text / strict framing (26 Nov)
+
+Source authoring began from clean synchronized signed `3156ce45` on L5.
+Cached Hyper 1.13.0 archive verified against the locked checksum; vendored
+under serving with one upstream source file changed (server request parser
+and matching assertions, client unchanged), license and provenance retained.
+Mixed TE/CL in either order and all repeated CL now fail before normalization
+or application dispatch. One mature HTTP parser remains authoritative.
+New chunked ingress only on supported POST text routes, single chunked
+coding, no trailers/Expect/upgrade. Original auth, charged reader (absolute
+deadline, cumulative cap, growth-overlap charge), complete JSON validation,
+model lowering, SSE, persistent socket and scheduler ownership preserved.
+No new model semantics or modality endpoints.
+
+Focused isolated serving 33/0/0 and strict lint pass: public actual socket
+tests cover fixed/chunked, exact cap, UTF-8 split, complete-body admission,
+pipeline, malformed/mixed/duplicate framing, unauthorized and oversize,
+trailers and partial disconnect ledger retirement. Real cached-model
+CLI/TUI/Web (60.52s) and SDK (9.11s) regressions each passed 1/0/0.
+The CLI fixture sends both alias completions (fixed and chunked) over one
+connection and compares exact model output. Assembled gate passed all
+HIP/Vulkan/CUDA compile boundaries, strict full-default lint, GPU feature
+boundary tests (26/0/2), source-only qualification, dependency isolation
+and strict CPU workspace Clippy. Timed out at 1100s during CPU workspace
+test compilation after dependency-path cache invalidation; no CPU test
+suites executed. Full assembled gate remains required; retry with the
+warmed cache, then finish OA-1B multipart. Vendor source filter/check
+updated; Nix syntax passes, exact source-integrity derivation is blocked
+by the locked nixpkgs archive unavailable in the private network.
+Logs: `target/oa1b-chunked-focused-final.log`,
+`target/oa1b-chunked-real-consumers.log`, `target/oa1b-chunked-assembled.log`,
+`target/oa1b-chunked-nix-source.log`.
+
+Next OA-1B multipart: `multer 3.1.0` exists in local index, no archive or
+source. Cached form-data 0.6.0 rejected after reviewing decoder: truncated
+input may loop, invalid delimiter suffix may be treated as EOF. No
+downloads or duplicate parser. Obtain approved pinned decoder and qualify
+bounded part/aggregate ownership and cleanup, then OA-2. No real
+Realtime/model/media or new physical GPU performance claim.
+
+### Prior checkpoint — OA-1B bounded ingress and owned WebSocket primitives (26 Nov)
diff --git a/.agents/plans/FULL_IMPLEMENTATION_PLAN.md b/.agents/plans/FULL_IMPLEMENTATION_PLAN.md
index d9f0ae5a..e5b63d54 100644
--- a/.agents/plans/FULL_IMPLEMENTATION_PLAN.md
+++ b/.agents/plans/FULL_IMPLEMENTATION_PLAN.md
@@ -771,4 +771,5 @@ and accepted (2,544/0/109 across 318 isolated Rust suites; real cached
-CLI/TUI/Web and SDK passed). Public streaming
-activation requires strict mixed-framing handling and approved multipart
-decoder acquisition. Remaining OA-1B precedes the OA-2 semantic
-request/model-template boundary.
+CLI/TUI/Web and SDK passed). Public chunked text now uses the same
+bounded reader and a narrow cached Hyper server framing patch; focused
+33/0/0 and real cached CLI/TUI/Web/SDK pass; assembled acceptance pending. Remaining OA-1B multipart
+requires authorized pinned decoder acquisition; OA-2 semantic
+request/model-template boundary follows complete OA-1.
diff --git a/.agents/plans/TODO.md b/.agents/plans/TODO.md
index c90e62b1..7c15c3b3 100644
--- a/.agents/plans/TODO.md
+++ b/.agents/plans/TODO.md
@@ -236 +236 @@ Final assembled private-network gate on ad7a4178 passed 2,539 passed, zero failu
-public serving OA-1 (bounded async text substrate accepted; OA-1B charged staging and owned WebSocket primitives implemented, public streaming/multipart activation remains open), then ordinary benchmarks and ordered performance phases.
+public serving OA-1 (bounded async text substrate accepted; OA-1B charged staging and owned WebSocket primitives accepted, strict public chunked text implemented, multipart remains open), then ordinary benchmarks and ordered performance phases.
diff --git a/.agents/plans/serving-and-speculation/NEXT_ITERATION_OPENAI_COMPATIBLE_SERVING.md b/.agents/plans/serving-and-speculation/NEXT_ITERATION_OPENAI_COMPATIBLE_SERVING.md
index a310afb9..0cfb27b0 100644
--- a/.agents/plans/serving-and-speculation/NEXT_ITERATION_OPENAI_COMPATIBLE_SERVING.md
+++ b/.agents/plans/serving-and-speculation/NEXT_ITERATION_OPENAI_COMPATIBLE_SERVING.md
@@ -9 +9 @@ Older ordering below is context; valid owner constraints remain authoritative.
-Status: **OA-1A accepted; OA-1B bounded ingress and WebSocket primitives accepted** (26 Nov). Public streaming/multipart activation remains open; OA-2 and later remain ordered.
+Status: **OA-1A and OA-1B ingress/WebSocket primitives accepted; public chunked text focused-qualified, assembled gate pending** (26 Nov). Multipart remains open; OA-2 and later remain ordered.
@@ -1199,10 +1199,23 @@ model support or external exposure is claimed.
-Two activation blockers remain: (1) pinned Hyper 1.13.0 removes
-Content-Length when Transfer-Encoding is also present (and disables keep-alive),
-so application-level headers cannot prove strict rejection of mixed framing.
-Retain refusal of all public chunked/Expect ingress until a mature parser
-exposes/rejects the ambiguity, or an explicit reviewed policy change is
-approved. Do not add a second handwritten parser. (2) no mature multipart
-decoder (including multer) is installed in the approved offline Cargo cache.
-Obtain an approved pinned decoder and qualify bounded per-part/aggregate
-streaming admission; no automatic download, homegrown parser, or fake
-multipart/Realtime endpoint. OA-1 stays open and OA-2 remains ordered.
+**Public chunked text activation (26 Nov).** The cached checksum-verified
+Hyper 1.13.0 registry source is vendored with one server request parser
+patch: refuse mixed Transfer-Encoding/Content-Length (either order) and
+repeated Content-Length (including identical values), before normalization
+or application dispatch. Client parsing/encoding stays upstream. Preserve
+the exact source revision/license, one root lock, and existing vendor
+packaging; remove this patch when upstream supports the strict policy.
+No duplicate wire parser. The original public text routes now accept
+only single chunked transfer coding on POST; Expect, other codings,
+trailers and upgrades with chunked bodies remain refused. Decoded frames
+use the same charged staging owner and absolute deadline as fixed-length
+JSON, with authorization before reads and complete JSON validation before
+model admission. Persistent connection and SSE ownership are unchanged.
+No new model semantics or modality capability.
+
+Remaining multipart blocker: `multer` is indexed locally (3.1.0 candidate)
+but its archive/source is absent. Cached `form-data 0.6.0` was reviewed and
+rejected: its decoder loops after truncated input and can treat malformed
+boundary suffixes as EOF. Do not substitute that parser or a whole-body
+email MIME decoder for strict streaming form-data. Obtain authorized pinned
+multipart acquisition, then qualify bounded part/aggregate admission and
+cleanup. No automatic download, homegrown parser or fake endpoints.
+OA-1 stays open and OA-2 remains ordered.
diff --git a/.agents/plans/serving-and-speculation/TODO.md b/.agents/plans/serving-and-speculation/TODO.md
index fd179326..0ca5aaa4 100644
--- a/.agents/plans/serving-and-speculation/TODO.md
+++ b/.agents/plans/serving-and-speculation/TODO.md
@@ -74,7 +74,20 @@ Contracts: [joint serving/goodput](../../docs/JOINT_CONCURRENT_SERVING_OPTIMIZAT
-- [ ] OA-1B remaining activation: public chunked/multipart streaming ingress
-  after resolving pinned Hyper mixed Transfer-Encoding/Content-Length
-  normalization (retain strict refusal) and approved acquisition of a
-  mature multipart decoder absent from the offline cache. Qualify bounded
-  per-part/aggregate admission and cleanup. No second HTTP parser, no
-  automatic downloads or fake modality endpoint. OA-2 remains ordered
-  after OA-1.
+- [~] OA-1B public chunked text: pinned cached Hyper 1.13.0 server-parser
+  patch refuses mixed TE/CL and repeated CL before normalization; exact
+  upstream bytes except role.rs and local patch notice. Same charged
+  reader, authorization, JSON/SSE, persistent connection and listener.
+  Actual public socket tests verify exact cap, UTF-8 split, complete-body
+  admission, pipeline, bad framing, unauthorized/oversize/trailer refusal
+  and partial disconnect charge retirement. Focused serving 33/0/0 and
+  strict lint pass; real cached CLI/TUI/Web (60.52s) and SDK (9.11s)
+  regressions pass 1/0/0 each. Assembled HIP/Vulkan/CUDA compile
+  boundaries, full-default/CPU strict lint, GPU boundary tests 26/0/2,
+  source-only regressions and dependency isolation passed. Gate timed out
+  at 1100s compiling CPU workspace tests; no CPU suites executed. Retry
+  full isolated gate with the warmed cache before acceptance.
+  Nix vendor source filter/check updated and syntax passes; actual locked
+  source-integrity check blocked by unavailable nixpkgs archive offline.
+- [ ] OA-1B multipart: acquire approved pinned mature multipart decoder
+  (multer 3.1.0 indexed but bytes absent); cached form-data 0.6.0 rejected
+  on truncated-input loop and invalid boundary termination. Qualify
+  bounded per-part/aggregate admission and cleanup. No second HTTP parser,
+  automatic downloads or fake modality endpoint. OA-2 follows OA-1.
~~~~

### D.10 — docs(serving): accept strict chunked text workspace gate

Commit: `715127ccfd5568fe1fd35663784c64410cad05c4`

Original commit message:

~~~~text
docs(serving): accept strict chunked text workspace gate

Recover the successful isolated assembled retry on 08a9d30e. Preserve the outstanding exact Nix source-integrity blocker and isolate the missing pinned multipart archive.
~~~~

Original checkpoint narrative at this commit (historical):

~~~~markdown
### Current checkpoint — OA-1B public chunked text / strict framing (26 Nov)

Source authoring began from clean synchronized signed `3156ce45` on L5.
Cached Hyper 1.13.0 archive verified against the locked checksum; vendored
under serving with one upstream source file changed (server request parser
and matching assertions, client unchanged), license and provenance retained.
Mixed TE/CL in either order and all repeated CL now fail before normalization
or application dispatch. One mature HTTP parser remains authoritative.
New chunked ingress only on supported POST text routes, single chunked
coding, no trailers/Expect/upgrade. Original auth, charged reader (absolute
deadline, cumulative cap, growth-overlap charge), complete JSON validation,
model lowering, SSE, persistent socket and scheduler ownership preserved.
No new model semantics or modality endpoints.

Focused isolated serving 33/0/0 and strict lint pass: public actual socket
tests cover fixed/chunked, exact cap, UTF-8 split, complete-body admission,
pipeline, malformed/mixed/duplicate framing, unauthorized and oversize,
trailers and partial disconnect ledger retirement. Real cached-model
CLI/TUI/Web (60.52s) and SDK (9.11s) regressions each passed 1/0/0.
The CLI fixture sends both alias completions (fixed and chunked) over one
connection and compares exact model output. Assembled gate passed all
HIP/Vulkan/CUDA compile boundaries, strict full-default lint, GPU feature
boundary tests (26/0/2), source-only qualification, dependency isolation
and strict CPU workspace Clippy. Initial run timed out at 1100s during
CPU workspace test compilation after dependency-path cache invalidation.
The full isolated warmed-cache retry on signed 08a9d30e exited 0:
2,546/0/109 across 318 Rust suites, strict lint, release TUI/Web, installer
and scripts passed. Chunked text accepted; no production source changes
during retry. Vendor source filter/check updated and Nix syntax passed;
the exact Nix source-integrity derivation remains unverified (locked
nixpkgs, pyproject-build-systems, pyproject-nix and uv2nix source hashes
not present among locally available Nix source roots; crane and rust-overlay
matched). No substitutions or downloads.
Logs: `target/oa1b-chunked-focused-final.log`,
`target/oa1b-chunked-real-consumers.log`, `target/oa1b-chunked-assembled.log`,
`target/oa1b-chunked-assembled-retry.log`, `target/oa1b-chunked-nix-source.log`.

Next OA-1B multipart: `multer 3.1.0` exists in local index, no archive or
source. A metadata-only isolated Cargo resolution (exact version, default
features disabled) resolved 20 registry packages; only the multer archive
is absent. Its indexed SHA-256 is
83e87776546dc87511aa5ee218730c92b666d7264ab6ed41f9d215af9cd5224b.
Request approval to acquire that single archive from crates.io and verify
its checksum against the cached index before extraction. Review the actual
decoder source, then integrate under the existing bounded reader,
per-part/aggregate admission, deadlines and joined cancellation.
Cached form-data 0.6.0 rejected (truncated input may loop; invalid
delimiter suffix may be treated as EOF). No automatic downloads, duplicate
parser or fabricated endpoint. OA-2 follows accepted OA-1. No real
Realtime/model/media or new physical GPU performance claim.
~~~~

Changed paths:

~~~~text
M	.agents/docs/CURRENT_BASELINE.md
M	.agents/docs/CURRENT_HANDOFF.md
M	.agents/plans/FULL_IMPLEMENTATION_PLAN.md
M	.agents/plans/TODO.md
M	.agents/plans/serving-and-speculation/NEXT_ITERATION_OPENAI_COMPATIBLE_SERVING.md
M	.agents/plans/serving-and-speculation/TODO.md
~~~~

Exact tracked engineering prose delta (zero-context Git diff; full surrounding prose at the cited revision):

~~~~diff
diff --git a/.agents/docs/CURRENT_BASELINE.md b/.agents/docs/CURRENT_BASELINE.md
index c71e7030..258ed059 100644
--- a/.agents/docs/CURRENT_BASELINE.md
+++ b/.agents/docs/CURRENT_BASELINE.md
@@ -15,4 +15,5 @@ regressions pass (60.52s and 9.11s). Assembled GPU boundary tests
-26/0/2, strict default/CPU lint and dependency isolation passed; gate
-timed out at 1100s during CPU workspace test compilation, no CPU suites
-executed. Complete assembled acceptance remains pending. Nix vendor source check blocked by uncached locked
-nixpkgs input; syntax passed. No new physical GPU or performance evidence.
+26/0/2, strict default/CPU lint and dependency isolation passed; initial
+gate timed out during CPU workspace test compilation. Full isolated retry
+on 08a9d30e exited 0: 2,546/0/109 across 318 Rust suites, release TUI/Web,
+installer and scripts. Chunked text accepted. Nix vendor source derivation
+remains blocked by unavailable exact locked flake inputs; syntax passed. No new physical GPU or performance evidence.
diff --git a/.agents/docs/CURRENT_HANDOFF.md b/.agents/docs/CURRENT_HANDOFF.md
index 2c54097e..740ffcc9 100644
--- a/.agents/docs/CURRENT_HANDOFF.md
+++ b/.agents/docs/CURRENT_HANDOFF.md
@@ -60,6 +60,10 @@ boundary tests (26/0/2), source-only qualification, dependency isolation
-and strict CPU workspace Clippy. Timed out at 1100s during CPU workspace
-test compilation after dependency-path cache invalidation; no CPU test
-suites executed. Full assembled gate remains required; retry with the
-warmed cache, then finish OA-1B multipart. Vendor source filter/check
-updated; Nix syntax passes, exact source-integrity derivation is blocked
-by the locked nixpkgs archive unavailable in the private network.
+and strict CPU workspace Clippy. Initial run timed out at 1100s during
+CPU workspace test compilation after dependency-path cache invalidation.
+The full isolated warmed-cache retry on signed 08a9d30e exited 0:
+2,546/0/109 across 318 Rust suites, strict lint, release TUI/Web, installer
+and scripts passed. Chunked text accepted; no production source changes
+during retry. Vendor source filter/check updated and Nix syntax passed;
+the exact Nix source-integrity derivation remains unverified (locked
+nixpkgs, pyproject-build-systems, pyproject-nix and uv2nix source hashes
+not present among locally available Nix source roots; crane and rust-overlay
+matched). No substitutions or downloads.
@@ -68 +72 @@ Logs: `target/oa1b-chunked-focused-final.log`,
-`target/oa1b-chunked-nix-source.log`.
+`target/oa1b-chunked-assembled-retry.log`, `target/oa1b-chunked-nix-source.log`.
@@ -71,4 +75,11 @@ Next OA-1B multipart: `multer 3.1.0` exists in local index, no archive or
-source. Cached form-data 0.6.0 rejected after reviewing decoder: truncated
-input may loop, invalid delimiter suffix may be treated as EOF. No
-downloads or duplicate parser. Obtain approved pinned decoder and qualify
-bounded part/aggregate ownership and cleanup, then OA-2. No real
+source. A metadata-only isolated Cargo resolution (exact version, default
+features disabled) resolved 20 registry packages; only the multer archive
+is absent. Its indexed SHA-256 is
+83e87776546dc87511aa5ee218730c92b666d7264ab6ed41f9d215af9cd5224b.
+Request approval to acquire that single archive from crates.io and verify
+its checksum against the cached index before extraction. Review the actual
+decoder source, then integrate under the existing bounded reader,
+per-part/aggregate admission, deadlines and joined cancellation.
+Cached form-data 0.6.0 rejected (truncated input may loop; invalid
+delimiter suffix may be treated as EOF). No automatic downloads, duplicate
+parser or fabricated endpoint. OA-2 follows accepted OA-1. No real
diff --git a/.agents/plans/FULL_IMPLEMENTATION_PLAN.md b/.agents/plans/FULL_IMPLEMENTATION_PLAN.md
index e5b63d54..e10f8da4 100644
--- a/.agents/plans/FULL_IMPLEMENTATION_PLAN.md
+++ b/.agents/plans/FULL_IMPLEMENTATION_PLAN.md
@@ -773 +773,2 @@ bounded reader and a narrow cached Hyper server framing patch; focused
-33/0/0 and real cached CLI/TUI/Web/SDK pass; assembled acceptance pending. Remaining OA-1B multipart
+33/0/0, real cached CLI/TUI/Web/SDK and full isolated retry 2,546/0/109
+(318 suites) passed. Remaining OA-1B multipart
diff --git a/.agents/plans/TODO.md b/.agents/plans/TODO.md
index 7c15c3b3..6b934a90 100644
--- a/.agents/plans/TODO.md
+++ b/.agents/plans/TODO.md
@@ -236 +236 @@ Final assembled private-network gate on ad7a4178 passed 2,539 passed, zero failu
-public serving OA-1 (bounded async text substrate accepted; OA-1B charged staging and owned WebSocket primitives accepted, strict public chunked text implemented, multipart remains open), then ordinary benchmarks and ordered performance phases.
+public serving OA-1 (bounded async text substrate accepted; OA-1B charged staging and owned WebSocket primitives accepted, strict public chunked text accepted, multipart remains open), then ordinary benchmarks and ordered performance phases.
diff --git a/.agents/plans/serving-and-speculation/NEXT_ITERATION_OPENAI_COMPATIBLE_SERVING.md b/.agents/plans/serving-and-speculation/NEXT_ITERATION_OPENAI_COMPATIBLE_SERVING.md
index 0cfb27b0..05ec0760 100644
--- a/.agents/plans/serving-and-speculation/NEXT_ITERATION_OPENAI_COMPATIBLE_SERVING.md
+++ b/.agents/plans/serving-and-speculation/NEXT_ITERATION_OPENAI_COMPATIBLE_SERVING.md
@@ -9 +9 @@ Older ordering below is context; valid owner constraints remain authoritative.
-Status: **OA-1A and OA-1B ingress/WebSocket primitives accepted; public chunked text focused-qualified, assembled gate pending** (26 Nov). Multipart remains open; OA-2 and later remain ordered.
+Status: **OA-1A and OA-1B ingress/WebSocket primitives accepted; public chunked text accepted** (27 Nov). Multipart remains open; OA-2 and later remain ordered.
@@ -1214,2 +1214,8 @@ No new model semantics or modality capability.
-Remaining multipart blocker: `multer` is indexed locally (3.1.0 candidate)
-but its archive/source is absent. Cached `form-data 0.6.0` was reviewed and
+The full isolated retry on 08a9d30e exited 0 (2,546/0/109 across 318
+Rust suites; real cached CPU consumers and SDK passed). Exact Nix
+source-integrity remains unverified because locked inputs are absent.
+
+Remaining multipart blocker: `multer` 3.1.0 is indexed locally; a
+metadata-only offline resolution resolves its 20 registry packages and
+only the multer archive/source is absent. Acquire that pinned archive
+with explicit authorization and checksum verification before review. Cached `form-data 0.6.0` was reviewed and
diff --git a/.agents/plans/serving-and-speculation/TODO.md b/.agents/plans/serving-and-speculation/TODO.md
index 0ca5aaa4..05e170ac 100644
--- a/.agents/plans/serving-and-speculation/TODO.md
+++ b/.agents/plans/serving-and-speculation/TODO.md
@@ -74 +74 @@ Contracts: [joint serving/goodput](../../docs/JOINT_CONCURRENT_SERVING_OPTIMIZAT
-- [~] OA-1B public chunked text: pinned cached Hyper 1.13.0 server-parser
+- [x] OA-1B public chunked text: pinned cached Hyper 1.13.0 server-parser
@@ -84,5 +84,6 @@ Contracts: [joint serving/goodput](../../docs/JOINT_CONCURRENT_SERVING_OPTIMIZAT
-  source-only regressions and dependency isolation passed. Gate timed out
-  at 1100s compiling CPU workspace tests; no CPU suites executed. Retry
-  full isolated gate with the warmed cache before acceptance.
-  Nix vendor source filter/check updated and syntax passes; actual locked
-  source-integrity check blocked by unavailable nixpkgs archive offline.
+  source-only regressions and dependency isolation passed. Initial gate
+  timed out at 1100s compiling CPU workspace tests. Full isolated retry
+  on 08a9d30e exited 0: 2,546/0/109 across 318 Rust suites, release TUI/Web,
+  installer and scripts. Nix source filter/check updated and syntax passes;
+  exact locked source-integrity derivation blocked by four unavailable
+  flake source hashes offline. No new physical GPU qualification.
@@ -90 +91,3 @@ Contracts: [joint serving/goodput](../../docs/JOINT_CONCURRENT_SERVING_OPTIMIZAT
-  (multer 3.1.0 indexed but bytes absent); cached form-data 0.6.0 rejected
+  (multer 3.1.0 indexed; offline resolution of 20 registry packages
+  lacks only its archive, checksum recorded in current handoff);
+  cached form-data 0.6.0 rejected
~~~~

### D.11 — feat(serving): own bounded multipart form ingestion

Commit: `3f8329f2aaad530950a355d63eea5743f4143705`

Original commit message:

~~~~text
feat(serving): own bounded multipart form ingestion

Acquire the single approved checksum-verified Multer archive and retain strict decoded EOF and duplicate part header policy. Stage bounded parts through the original serving ledger; keep unsupported modality endpoints refused. Complete OA-1 after real socket, cached CLI/TUI/Web and SDK qualification and the isolated assembled workspace gate.
~~~~

Original checkpoint narrative at this commit (historical):

~~~~markdown
### Current checkpoint — OA-1 accepted; OA-2 semantic request next (27 Nov)

Started from clean signed 715127cc on L5. Acquired only multer 3.1.0
from crates.io after user continuation of the explicit dependency request.
Archive SHA-256 matched the cached index
83e87776546dc87511aa5ee218730c92b666d7264ab6ed41f9d215af9cd5224b.
All other resolved dependencies were already cached; builds remain offline.
Vendor source/license/provenance retained with narrow strict form policy:
initial delimiter at body start, final delimiter with optional transport
padding/CRLF and actual decoded EOF, duplicate part header refusal.
No MIME preambles/epilogues; no second parser. Existing upstream field
streaming and size constraints retained.

Shared Reader feeds Multer cooperatively, under original decoded aggregate
cap, absolute deadline and trailer/length checks. Charge conservative parser
headroom and bounded retained metadata before parsing; reuse exact charged
Collected backing and growth-overlap for individual parts. Completed form
owns parts and metadata together. No filename-to-path use, automatic transfer
decoding, detached task or persistent upload store. Dormant private
primitive until real modality route; ordinary server media refusal and
JSON text policy preserved. Repeated fields preserve order and remain
subject to future endpoint-specific semantic validation.

Focused serving 38/0/0, strict lint and architecture pass. Actual private
TCP fixtures verify authorization before read, fixed/chunked binary/text/
empty parts, fragmented delimiters, actual admission pressure, count/part/
aggregate bounds, malformed and incomplete forms, closing delimiter
without HTTP EOF, trailers, timeout and dropped connection ownership.
Existing public listener still refuses unimplemented media and multipart
on text endpoints. Actual cached CLI/TUI/Web (76.57s) and SDK (14.07s)
regressions each passed 1/0/0. Full assembled isolated gate exited 0
(902.1s): 2,551/0/109 across 318 Rust suites, strict lint, backend
feature boundaries, release TUI/Web, installer and scripts. Logs: `target/oa1b-multipart-focused-final.log`,
`target/oa1b-multipart-real-consumers.log`, `target/oa1b-multipart-assembled.log`.
Exact Nix source-integrity remains separately blocked by four unavailable
locked inputs; prior Nix syntax passed. No new model/physical GPU or
performance claims. OA-1 transport substrate accepted. Next OA-2:
introduce shared typed semantic requests/events and move model-specific
chat/template lowering out of HTTP. Preserve the original engine,
admission, listener, source identity and CLI behavior. Adapt CLI run,
Chat Completions and legacy Completions to the same seam; no new Responses
endpoint until OA-2 is qualified.
~~~~

Changed paths:

~~~~text
M	.agents/docs/CURRENT_BASELINE.md
M	.agents/docs/CURRENT_HANDOFF.md
M	.agents/plans/FULL_IMPLEMENTATION_PLAN.md
M	.agents/plans/README.md
M	.agents/plans/TODO.md
M	.agents/plans/serving-and-speculation/NEXT_ITERATION_OPENAI_COMPATIBLE_SERVING.md
M	.agents/plans/serving-and-speculation/TODO.md
M	Cargo.lock
M	Cargo.toml
M	crates/zerogravity-serving/Cargo.toml
M	crates/zerogravity-serving/src/http.rs
M	crates/zerogravity-serving/src/http/ingress_body.rs
A	crates/zerogravity-serving/src/http/multipart.rs
A	crates/zerogravity-serving/src/http/multipart/tests.rs
M	crates/zerogravity-serving/src/http/transport.rs
M	crates/zerogravity-serving/src/server/tests.rs
A	crates/zerogravity-serving/vendor/multer/.cargo_vcs_info.json
A	crates/zerogravity-serving/vendor/multer/Cargo.toml
A	crates/zerogravity-serving/vendor/multer/Cargo.toml.orig
A	crates/zerogravity-serving/vendor/multer/LICENSE
A	crates/zerogravity-serving/vendor/multer/PATCHES.md
A	crates/zerogravity-serving/vendor/multer/README.md
A	crates/zerogravity-serving/vendor/multer/build.rs
A	crates/zerogravity-serving/vendor/multer/examples/README.md
A	crates/zerogravity-serving/vendor/multer/examples/hyper_server_example.rs
A	crates/zerogravity-serving/vendor/multer/examples/parse_async_read.rs
A	crates/zerogravity-serving/vendor/multer/examples/prevent_dos_attack.rs
A	crates/zerogravity-serving/vendor/multer/examples/simple_example.rs
A	crates/zerogravity-serving/vendor/multer/src/buffer.rs
A	crates/zerogravity-serving/vendor/multer/src/constants.rs
A	crates/zerogravity-serving/vendor/multer/src/constraints.rs
A	crates/zerogravity-serving/vendor/multer/src/content_disposition.rs
A	crates/zerogravity-serving/vendor/multer/src/error.rs
A	crates/zerogravity-serving/vendor/multer/src/field.rs
A	crates/zerogravity-serving/vendor/multer/src/helpers.rs
A	crates/zerogravity-serving/vendor/multer/src/lib.rs
A	crates/zerogravity-serving/vendor/multer/src/multipart.rs
A	crates/zerogravity-serving/vendor/multer/src/size_limit.rs
A	crates/zerogravity-serving/vendor/multer/tests/integration.rs
~~~~

Exact tracked engineering prose delta (zero-context Git diff; full surrounding prose at the cited revision):

~~~~diff
diff --git a/.agents/docs/CURRENT_BASELINE.md b/.agents/docs/CURRENT_BASELINE.md
index 258ed059..60c9af0e 100644
--- a/.agents/docs/CURRENT_BASELINE.md
+++ b/.agents/docs/CURRENT_BASELINE.md
@@ -7,0 +8,12 @@ historical comparison numbers into new implementation decisions.
+## 27 Nov — OA-1B multipart foundation
+
+Pinned checksum-verified Multer 3.1.0, strict complete form termination
+and duplicate part header policy. Shared bounded Reader/charged part
+backing and metadata, actual private TCP validation and rollback; public
+media remains unsupported. Focused 38/0/0, strict lint and architecture
+passed; real cached CLI/TUI/Web (76.57s) and SDK (14.07s) passed.
+Full isolated gate exited 0 (902.1s): 2,551/0/109 across 318 Rust
+suites plus release TUI/Web, installer and scripts. OA-1 accepted;
+OA-2 next. No new physical GPU/performance claim. Exact Nix source-integrity remains
+separately blocked by unavailable locked inputs.
+
diff --git a/.agents/docs/CURRENT_HANDOFF.md b/.agents/docs/CURRENT_HANDOFF.md
index 740ffcc9..532f6029 100644
--- a/.agents/docs/CURRENT_HANDOFF.md
+++ b/.agents/docs/CURRENT_HANDOFF.md
@@ -37 +37,44 @@ reusable work and measured history, not precedence over this order.
-### Current checkpoint — OA-1B public chunked text / strict framing (26 Nov)
+### Current checkpoint — OA-1 accepted; OA-2 semantic request next (27 Nov)
+
+Started from clean signed 715127cc on L5. Acquired only multer 3.1.0
+from crates.io after user continuation of the explicit dependency request.
+Archive SHA-256 matched the cached index
+83e87776546dc87511aa5ee218730c92b666d7264ab6ed41f9d215af9cd5224b.
+All other resolved dependencies were already cached; builds remain offline.
+Vendor source/license/provenance retained with narrow strict form policy:
+initial delimiter at body start, final delimiter with optional transport
+padding/CRLF and actual decoded EOF, duplicate part header refusal.
+No MIME preambles/epilogues; no second parser. Existing upstream field
+streaming and size constraints retained.
+
+Shared Reader feeds Multer cooperatively, under original decoded aggregate
+cap, absolute deadline and trailer/length checks. Charge conservative parser
+headroom and bounded retained metadata before parsing; reuse exact charged
+Collected backing and growth-overlap for individual parts. Completed form
+owns parts and metadata together. No filename-to-path use, automatic transfer
+decoding, detached task or persistent upload store. Dormant private
+primitive until real modality route; ordinary server media refusal and
+JSON text policy preserved. Repeated fields preserve order and remain
+subject to future endpoint-specific semantic validation.
+
+Focused serving 38/0/0, strict lint and architecture pass. Actual private
+TCP fixtures verify authorization before read, fixed/chunked binary/text/
+empty parts, fragmented delimiters, actual admission pressure, count/part/
+aggregate bounds, malformed and incomplete forms, closing delimiter
+without HTTP EOF, trailers, timeout and dropped connection ownership.
+Existing public listener still refuses unimplemented media and multipart
+on text endpoints. Actual cached CLI/TUI/Web (76.57s) and SDK (14.07s)
+regressions each passed 1/0/0. Full assembled isolated gate exited 0
+(902.1s): 2,551/0/109 across 318 Rust suites, strict lint, backend
+feature boundaries, release TUI/Web, installer and scripts. Logs: `target/oa1b-multipart-focused-final.log`,
+`target/oa1b-multipart-real-consumers.log`, `target/oa1b-multipart-assembled.log`.
+Exact Nix source-integrity remains separately blocked by four unavailable
+locked inputs; prior Nix syntax passed. No new model/physical GPU or
+performance claims. OA-1 transport substrate accepted. Next OA-2:
+introduce shared typed semantic requests/events and move model-specific
+chat/template lowering out of HTTP. Preserve the original engine,
+admission, listener, source identity and CLI behavior. Adapt CLI run,
+Chat Completions and legacy Completions to the same seam; no new Responses
+endpoint until OA-2 is qualified.
+
+### Prior checkpoint — OA-1B public chunked text / strict framing (26 Nov)
diff --git a/.agents/plans/FULL_IMPLEMENTATION_PLAN.md b/.agents/plans/FULL_IMPLEMENTATION_PLAN.md
index e10f8da4..138a4e82 100644
--- a/.agents/plans/FULL_IMPLEMENTATION_PLAN.md
+++ b/.agents/plans/FULL_IMPLEMENTATION_PLAN.md
@@ -774,3 +774,5 @@ bounded reader and a narrow cached Hyper server framing patch; focused
-(318 suites) passed. Remaining OA-1B multipart
-requires authorized pinned decoder acquisition; OA-2 semantic
-request/model-template boundary follows complete OA-1.
+(318 suites) passed. OA-1B multipart foundation implemented using approved
+checksum-verified Multer and original bounded ingress/staging owners;
+focused 38/0/0, real cached CLI/TUI/Web and SDK, and full isolated gate
+2,551/0/109 (318 suites) passed. OA-1 accepted. OA-2 shared semantic
+request/model-template boundary is next; OA-3 Responses follows OA-2.
diff --git a/.agents/plans/README.md b/.agents/plans/README.md
index 3c27f15c..96908347 100644
--- a/.agents/plans/README.md
+++ b/.agents/plans/README.md
@@ -21,2 +21,3 @@ UX-P and the model-local execution selector are accepted. The
-Current unit: [OA-1 public serving transport](serving-and-speculation/NEXT_ITERATION_OPENAI_COMPATIBLE_SERVING.md#oa-1--async-transport-substrate).
-Read the live handoff for the assembled gate and remaining OA-1 work;
+OA-1 public serving transport is accepted.
+Current unit: [OA-2 semantic request/model-template boundary](serving-and-speculation/NEXT_ITERATION_OPENAI_COMPATIBLE_SERVING.md#oa-2--semantic-request-model-and-model-template-boundary).
+Read the live handoff for assembled acceptance and ordered OA-2 work;
diff --git a/.agents/plans/TODO.md b/.agents/plans/TODO.md
index 6b934a90..eabbd4c3 100644
--- a/.agents/plans/TODO.md
+++ b/.agents/plans/TODO.md
@@ -236 +236 @@ Final assembled private-network gate on ad7a4178 passed 2,539 passed, zero failu
-public serving OA-1 (bounded async text substrate accepted; OA-1B charged staging and owned WebSocket primitives accepted, strict public chunked text accepted, multipart remains open), then ordinary benchmarks and ordered performance phases.
+public serving OA-1 (bounded async text substrate accepted; OA-1B charged staging and owned WebSocket primitives accepted, OA-1 async transport substrate accepted, OA-2 shared semantic request/model-template boundary next), then ordinary benchmarks and ordered performance phases.
diff --git a/.agents/plans/serving-and-speculation/NEXT_ITERATION_OPENAI_COMPATIBLE_SERVING.md b/.agents/plans/serving-and-speculation/NEXT_ITERATION_OPENAI_COMPATIBLE_SERVING.md
index 05ec0760..d3b2e9f0 100644
--- a/.agents/plans/serving-and-speculation/NEXT_ITERATION_OPENAI_COMPATIBLE_SERVING.md
+++ b/.agents/plans/serving-and-speculation/NEXT_ITERATION_OPENAI_COMPATIBLE_SERVING.md
@@ -9 +9 @@ Older ordering below is context; valid owner constraints remain authoritative.
-Status: **OA-1A and OA-1B ingress/WebSocket primitives accepted; public chunked text accepted** (27 Nov). Multipart remains open; OA-2 and later remain ordered.
+Status: **OA-1 async transport substrate accepted; OA-2 semantic request/model-template boundary next** (27 Nov). OA-3 and later remain ordered.
@@ -1218,10 +1218,31 @@ source-integrity remains unverified because locked inputs are absent.
-Remaining multipart blocker: `multer` 3.1.0 is indexed locally; a
-metadata-only offline resolution resolves its 20 registry packages and
-only the multer archive/source is absent. Acquire that pinned archive
-with explicit authorization and checksum verification before review. Cached `form-data 0.6.0` was reviewed and
-rejected: its decoder loops after truncated input and can treat malformed
-boundary suffixes as EOF. Do not substitute that parser or a whole-body
-email MIME decoder for strict streaming form-data. Obtain authorized pinned
-multipart acquisition, then qualify bounded part/aggregate admission and
-cleanup. No automatic download, homegrown parser or fake endpoints.
-OA-1 stays open and OA-2 remains ordered.
+**Multipart foundation (27 Nov).** Acquired only the authorized
+`multer 3.1.0` registry archive; SHA-256 matches the existing cached
+registry index. Other resolved dependencies were already cached.
+Cached form-data 0.6.0 remains rejected (truncated-input loop and invalid
+termination). Multer owns the parser; a narrow vendored patch rejects
+duplicate part headers and requires the initial delimiter at body start
+and final delimiter, optional transport padding/CRLF, and actual decoded
+HTTP EOF before success. Preamble/epilogue are deliberately outside this
+HTTP form subset. No separate handwritten parser. Source/license and
+patch provenance retained under the serving vendor tree.
+
+The existing decoded-body reader feeds the parser cooperatively, one
+frame at a time. Original aggregate byte cap, absolute deadline and
+trailer refusal remain authoritative. Conservative parser headroom and
+bounded retained metadata are admitted before parsing; each part uses the
+same exact charged backing growth as JSON, including old/new overlap.
+Part count and per-part bytes are bounded; completed forms retain bytes
+and metadata together. No filename-to-path interpretation, implicit
+transfer decoding, persistent upload store or detached parser task.
+Private actual TCP fixtures exercise the shared header/auth validation,
+fixed/chunked forms, fragmented delimiters, repeated fields, binary/empty
+fields, invalid/truncated forms, actual host budget refusal, size/count
+bounds, trailers/deadlines and cancellation retirement. Ordinary server
+still refuses media endpoints and multipart on JSON text routes. No new
+model semantics or public Realtime/media claims.
+
+Focused serving 38/0/0 and strict lint passed. Actual cached CLI/TUI/Web
+and SDK passed (76.57s and 14.07s). Full isolated assembled gate passed
+(902.1s): 2,551/0/109 across 318 Rust suites, release TUI/Web, installer
+and scripts. OA-1 accepted; exact Nix source-integrity remains separately
+unverified because four locked inputs are unavailable. OA-2 is next.
diff --git a/.agents/plans/serving-and-speculation/TODO.md b/.agents/plans/serving-and-speculation/TODO.md
index 05e170ac..4c7e1ea6 100644
--- a/.agents/plans/serving-and-speculation/TODO.md
+++ b/.agents/plans/serving-and-speculation/TODO.md
@@ -90,7 +90,17 @@ Contracts: [joint serving/goodput](../../docs/JOINT_CONCURRENT_SERVING_OPTIMIZAT
-- [ ] OA-1B multipart: acquire approved pinned mature multipart decoder
-  (multer 3.1.0 indexed; offline resolution of 20 registry packages
-  lacks only its archive, checksum recorded in current handoff);
-  cached form-data 0.6.0 rejected
-  on truncated-input loop and invalid boundary termination. Qualify
-  bounded per-part/aggregate admission and cleanup. No second HTTP parser,
-  automatic downloads or fake modality endpoint. OA-2 follows OA-1.
+- [x] OA-1B multipart: acquired only pinned multer 3.1.0 with index
+  checksum verification; other dependencies cached. Vendor strict initial/
+  final delimiter, decoded EOF and duplicate part header policy in its
+  existing state machine. Reuse original bounded reader, shared charged
+  backing, aggregate/per-part/count and metadata bounds, absolute deadline
+  and cancellation ownership. Private real socket tests verify complete
+  forms and rollback including real host-budget refusal; ordinary server
+  refuses media and non-JSON text input. Focused 38/0/0, strict lint,
+  architecture passed. Real cached CLI/TUI/Web and SDK passed (76.57s
+  and 14.07s). Full isolated gate exited 0 (902.1s): 2,551/0/109 across
+  318 Rust suites, release TUI/Web, installer and scripts. Exact Nix
+  source-integrity remains separately unverified. No second parser, media
+  capability or new scheduler. OA-1 accepted.
+- [ ] OA-2: shared typed semantic request/event boundary; model-owned
+  chat/template lowering; CLI run, Chat Completions and legacy Completions
+  consume the same semantic path. Preserve exact source/model/runtime,
+  auth, admission and cancellation; qualify before OA-3 Responses.
~~~~

### D.12 — Unify semantic text requests and model-owned prompt templates

Commit: `a76ddbfeaf708be9046ca14980dc1410d8f20cde`

Original commit message:

~~~~text
Unify semantic text requests and model-owned prompt templates

Accept OA-2 on the original scheduler and event receipt; preserve raw and legacy single-text adapters, admission bounds and real cached consumer equivalence. Qualify with 2553 passing test executions across 318 suites, no failures, 109 explicit skips and release smoke checks.
~~~~

Original checkpoint narrative at this commit (historical):

~~~~markdown
### Current checkpoint — OA-2 accepted; OA-3 Models and Responses text next (27 Nov)

Started from clean signed 3f8329f2 on L5. Core task text intent now has
ChatRole, ChatMessage and TextInput::{Raw, User, Conversation}. Engine
SemanticRequest retains the original model/output/stream/token-observation
controls and RequestLease/Scheduled/Active ownership. Legacy Request and
borrowed TextTask converge at canonical semantic admission; existing
ResponseEvent already provides typed text, token, terminal and error events.
No duplicate registry, runtime, scheduler, queue, model owner or event
publication. Existing direct CPU diagnostic runner uses the same model
intent/renderer without changing its physical/audit executor.

HTTP copies validated typed turns and charges their exact vector/text
backing rather than model-specific ChatML. Original selected source reader
and queue/host admission precede model-owned template measurement and
worker tokenization. PromptBounds prices exact rendered conversation bytes,
prompt tokens and existing admitted tokenizer scratch; enforce 1 MiB
rendered ceiling and configured Runtime input-byte ceiling. Raw and
legacy single-text input contracts stay intact. Qwen tokenizer owns the
existing qualified ChatML spelling and assistant prefix, supports the
four existing text roles, preserves order and exact UTF-8. Typed chat
refuses literal role delimiters, now including CLI/SDK single-user chat;
explicit raw prompt remains raw. No arbitrary Jinja/template execution,
media/tool content or new model support. Legacy TextService/TextTokenizer
adapters keep single-text methods; structured conversation requires an
explicit implementation. PreparedModel validation now takes SemanticRequest.
Public facade request_semantic returns the original typed cancellation
receipt. Original text events are retained rather than inventing another
protocol event stream.

Focused final engine/serving 163/0/0, strict lint; Qwen tokenizer suite
7/0/0 and explicit cached approved-tokenizer 1/0/0 (5.02s) passed,
including exact single/multi-turn token IDs versus previous raw framing.
Real CLI/TUI/Web and SDK regressions compare structured and previous
raw prompts, same alias owner and usage, single-user SDK equivalence,
model-owned role-token refusal and rendered configured byte cap. Actual cached CLI/TUI/Web and SDK regressions passed (67.77s and 15.85s).
Full isolated assembled gate exited 0 (878.6s): 2,553/0/109 across 318
Rust suites, release TUI/Web, installer and scripts passed. Subsequent
non-behavioral import cleanup and public PromptBounds re-export passed
finite CUDA qualification compilation and CPU facade strict lint
(`target/oa2-export-cleanup.log`). Logs:
`target/oa2-focused-final.log`, `target/oa2-focused.log`,
`target/oa2-approved-tokenizer.log`, `target/oa2-real-consumers-final.log`,
`target/oa2-assembled-final.log`. Initial broad validation cancelled to
preserve configured rendered chat cap before final acceptance. Exact Nix
source-integrity remains separately blocked by unavailable locked inputs.
No new physical GPU/performance claim. OA-2 accepted. Next: OA-3 Models
and Responses text with the same semantic boundary and original scheduler.
~~~~

Changed paths:

~~~~text
M	.agents/docs/CURRENT_BASELINE.md
M	.agents/docs/CURRENT_HANDOFF.md
M	.agents/plans/FULL_IMPLEMENTATION_PLAN.md
M	.agents/plans/README.md
M	.agents/plans/TODO.md
M	.agents/plans/serving-and-speculation/NEXT_ITERATION_OPENAI_COMPATIBLE_SERVING.md
M	.agents/plans/serving-and-speculation/TODO.md
M	crates/zerogravity-cli/src/run_cli/qwen_source.rs
M	crates/zerogravity-cli/tests/run_unified.rs
M	crates/zerogravity-core/src/task.rs
A	crates/zerogravity-core/src/task/text.rs
M	crates/zerogravity-engine/src/model_instance.rs
M	crates/zerogravity-engine/src/router/cpu_tests.rs
M	crates/zerogravity-engine/src/router/dispatch.rs
M	crates/zerogravity-engine/src/router/worker.rs
M	crates/zerogravity-engine/src/router/worker/tests.rs
M	crates/zerogravity-engine/src/runtime.rs
M	crates/zerogravity-model-qwen35/src/tokenize.rs
M	crates/zerogravity-serving/src/http.rs
M	crates/zerogravity-serving/src/http/transport.rs
M	crates/zerogravity-serving/src/server/tests.rs
M	crates/zerogravity-serving/src/service.rs
M	crates/zerogravity/src/builtins/text.rs
M	crates/zerogravity/src/builtins/text/gpu/qualification.rs
M	crates/zerogravity/src/diagnostics/cpu_text.rs
M	crates/zerogravity/src/diagnostics/text_gpu/captured.rs
M	crates/zerogravity/src/runtime.rs
M	crates/zerogravity/src/serving/facade.rs
M	crates/zerogravity/tests/fabric_sdk.rs
~~~~

Exact tracked engineering prose delta (zero-context Git diff; full surrounding prose at the cited revision):

~~~~diff
diff --git a/.agents/docs/CURRENT_BASELINE.md b/.agents/docs/CURRENT_BASELINE.md
index 60c9af0e..5991691d 100644
--- a/.agents/docs/CURRENT_BASELINE.md
+++ b/.agents/docs/CURRENT_BASELINE.md
@@ -7,0 +8,11 @@ historical comparison numbers into new implementation decisions.
+## 27 Nov — OA-2 semantic text request/model template
+
+Typed raw/user/conversation intent through original queue and worker,
+model-owned Qwen template measurement/validation/tokenization; exact
+previous tokenizer output preserved in cached approved-tokenizer test.
+Focused engine/serving 163/0/0, strict lint, tokenizer suite 7/0/0
+and approved-source 1/0/0 passed. Actual cached CLI/TUI/Web and SDK
+passed (67.77s and 15.85s); full isolated gate 2,553/0/109 across 318
+Rust suites (878.6s), release TUI/Web, installer and scripts passed.
+No new physical GPU/performance evidence. OA-2 accepted; OA-3 next.
+
diff --git a/.agents/docs/CURRENT_HANDOFF.md b/.agents/docs/CURRENT_HANDOFF.md
index 532f6029..210cf84f 100644
--- a/.agents/docs/CURRENT_HANDOFF.md
+++ b/.agents/docs/CURRENT_HANDOFF.md
@@ -37 +37,50 @@ reusable work and measured history, not precedence over this order.
-### Current checkpoint — OA-1 accepted; OA-2 semantic request next (27 Nov)
+### Current checkpoint — OA-2 accepted; OA-3 Models and Responses text next (27 Nov)
+
+Started from clean signed 3f8329f2 on L5. Core task text intent now has
+ChatRole, ChatMessage and TextInput::{Raw, User, Conversation}. Engine
+SemanticRequest retains the original model/output/stream/token-observation
+controls and RequestLease/Scheduled/Active ownership. Legacy Request and
+borrowed TextTask converge at canonical semantic admission; existing
+ResponseEvent already provides typed text, token, terminal and error events.
+No duplicate registry, runtime, scheduler, queue, model owner or event
+publication. Existing direct CPU diagnostic runner uses the same model
+intent/renderer without changing its physical/audit executor.
+
+HTTP copies validated typed turns and charges their exact vector/text
+backing rather than model-specific ChatML. Original selected source reader
+and queue/host admission precede model-owned template measurement and
+worker tokenization. PromptBounds prices exact rendered conversation bytes,
+prompt tokens and existing admitted tokenizer scratch; enforce 1 MiB
+rendered ceiling and configured Runtime input-byte ceiling. Raw and
+legacy single-text input contracts stay intact. Qwen tokenizer owns the
+existing qualified ChatML spelling and assistant prefix, supports the
+four existing text roles, preserves order and exact UTF-8. Typed chat
+refuses literal role delimiters, now including CLI/SDK single-user chat;
+explicit raw prompt remains raw. No arbitrary Jinja/template execution,
+media/tool content or new model support. Legacy TextService/TextTokenizer
+adapters keep single-text methods; structured conversation requires an
+explicit implementation. PreparedModel validation now takes SemanticRequest.
+Public facade request_semantic returns the original typed cancellation
+receipt. Original text events are retained rather than inventing another
+protocol event stream.
+
+Focused final engine/serving 163/0/0, strict lint; Qwen tokenizer suite
+7/0/0 and explicit cached approved-tokenizer 1/0/0 (5.02s) passed,
+including exact single/multi-turn token IDs versus previous raw framing.
+Real CLI/TUI/Web and SDK regressions compare structured and previous
+raw prompts, same alias owner and usage, single-user SDK equivalence,
+model-owned role-token refusal and rendered configured byte cap. Actual cached CLI/TUI/Web and SDK regressions passed (67.77s and 15.85s).
+Full isolated assembled gate exited 0 (878.6s): 2,553/0/109 across 318
+Rust suites, release TUI/Web, installer and scripts passed. Subsequent
+non-behavioral import cleanup and public PromptBounds re-export passed
+finite CUDA qualification compilation and CPU facade strict lint
+(`target/oa2-export-cleanup.log`). Logs:
+`target/oa2-focused-final.log`, `target/oa2-focused.log`,
+`target/oa2-approved-tokenizer.log`, `target/oa2-real-consumers-final.log`,
+`target/oa2-assembled-final.log`. Initial broad validation cancelled to
+preserve configured rendered chat cap before final acceptance. Exact Nix
+source-integrity remains separately blocked by unavailable locked inputs.
+No new physical GPU/performance claim. OA-2 accepted. Next: OA-3 Models
+and Responses text with the same semantic boundary and original scheduler.
+
+### Prior checkpoint — OA-1 accepted; OA-2 semantic request next (27 Nov)
diff --git a/.agents/plans/FULL_IMPLEMENTATION_PLAN.md b/.agents/plans/FULL_IMPLEMENTATION_PLAN.md
index 138a4e82..165cd871 100644
--- a/.agents/plans/FULL_IMPLEMENTATION_PLAN.md
+++ b/.agents/plans/FULL_IMPLEMENTATION_PLAN.md
@@ -778 +778,5 @@ focused 38/0/0, real cached CLI/TUI/Web and SDK, and full isolated gate
-request/model-template boundary is next; OA-3 Responses follows OA-2.
+request/model-template boundary is implemented (core typed input,
+original semantic queue/events and model-owned Qwen template/pricing;
+focused and actual tokenizer checks passed). Actual cached consumers
+and assembled gate passed (2,553/0/109, 318 suites, 878.6s). OA-2
+accepted; OA-3 Models and Responses text next.
diff --git a/.agents/plans/README.md b/.agents/plans/README.md
index 96908347..38f6536f 100644
--- a/.agents/plans/README.md
+++ b/.agents/plans/README.md
@@ -21,3 +21,3 @@ UX-P and the model-local execution selector are accepted. The
-OA-1 public serving transport is accepted.
-Current unit: [OA-2 semantic request/model-template boundary](serving-and-speculation/NEXT_ITERATION_OPENAI_COMPATIBLE_SERVING.md#oa-2--semantic-request-model-and-model-template-boundary).
-Read the live handoff for assembled acceptance and ordered OA-2 work;
+OA-1 public serving transport and OA-2 semantic request/model-template boundary are accepted.
+Current unit: [OA-3 Models and Responses text](serving-and-speculation/NEXT_ITERATION_OPENAI_COMPATIBLE_SERVING.md#oa-3--models--responses-text).
+Read the live handoff for assembled acceptance and ordered OA-3 work;
diff --git a/.agents/plans/TODO.md b/.agents/plans/TODO.md
index eabbd4c3..06c359b1 100644
--- a/.agents/plans/TODO.md
+++ b/.agents/plans/TODO.md
@@ -236 +236 @@ Final assembled private-network gate on ad7a4178 passed 2,539 passed, zero failu
-public serving OA-1 (bounded async text substrate accepted; OA-1B charged staging and owned WebSocket primitives accepted, OA-1 async transport substrate accepted, OA-2 shared semantic request/model-template boundary next), then ordinary benchmarks and ordered performance phases.
+public serving (OA-1 async transport and OA-2 shared semantic request/model-template boundary accepted; OA-3 Models and Responses text next), then ordinary benchmarks and ordered performance phases.
diff --git a/.agents/plans/serving-and-speculation/NEXT_ITERATION_OPENAI_COMPATIBLE_SERVING.md b/.agents/plans/serving-and-speculation/NEXT_ITERATION_OPENAI_COMPATIBLE_SERVING.md
index d3b2e9f0..632b84b2 100644
--- a/.agents/plans/serving-and-speculation/NEXT_ITERATION_OPENAI_COMPATIBLE_SERVING.md
+++ b/.agents/plans/serving-and-speculation/NEXT_ITERATION_OPENAI_COMPATIBLE_SERVING.md
@@ -9 +9 @@ Older ordering below is context; valid owner constraints remain authoritative.
-Status: **OA-1 async transport substrate accepted; OA-2 semantic request/model-template boundary next** (27 Nov). OA-3 and later remain ordered.
+Status: **OA-1 and OA-2 accepted** (27 Nov). OA-3 Models and Responses text next; later units remain ordered.
@@ -1264 +1264,35 @@ onto the same semantic request where practical.
-No Responses endpoint until this shared seam is sound.
+**OA-2 implementation (27 Nov).** Protocol-neutral ChatRole, ChatMessage
+and TextInput (raw, one user, ordered conversation) live in core/task;
+engine SemanticRequest carries the existing generation and observation
+controls. Legacy Request and borrowed TextTask are adapters; the original
+RequestLease/Scheduled/Active worker path retains typed input and existing
+ResponseEvent publication/cancellation. No extra execution owner, registry,
+scheduler or event queue. Existing text events are the semantic event
+contract for this slice; additional content/control/event kinds need
+real model capability, not placeholder wire coercion.
+
+HTTP validates only supported JSON shape and admits retained turn-vector
+and string copies. The exact selected model capability and tokenizer
+measure/validate the prompt before enqueue; PromptBounds prices rendered
+input, prompt token storage and original tokenizer scratch. Structured
+chat retains the historical 1 MiB rendered cap and configured runtime
+input limit; legacy single text retains its original content-length
+contract. The worker lowers the admitted typed input on the original
+tokenizer; Qwen owns the same qualified ChatML template, preserving role
+order, UTF-8 and assistant generation prefix. Single-user convenience
+uses that renderer; literal role delimiters in typed chat now refuse
+(including single-user CLI/SDK chat). Explicit raw text remains raw.
+No arbitrary Jinja evaluation or new model capability. Legacy-only
+TextService/Tokenizer implementations refuse structured chat unless they
+explicitly implement it; PreparedModel validation consumes SemanticRequest.
+The facade exposes request_semantic with the original typed event receipt.
+The existing direct CPU diagnostic runner also consumes the same model
+text intent/renderer; its physical runner and audit path are unchanged.
+
+Focused final engine/serving 163/0/0 and strict lint passed. Qwen tokenizer
+suite 7/0/0; explicit cached approved-tokenizer test 1/0/0 verifies exact
+single/multi-turn token IDs and measured prompt bytes against the previous
+raw spelling. Actual cached CLI/TUI/Web and SDK regressions passed
+(67.77s and 15.85s); full isolated assembled gate exited 0 (878.6s),
+2,553/0/109 across 318 Rust suites plus release TUI/Web, installer and
+scripts. OA-2 accepted; no Responses endpoint introduced in this slice.
diff --git a/.agents/plans/serving-and-speculation/TODO.md b/.agents/plans/serving-and-speculation/TODO.md
index 4c7e1ea6..2d653fc4 100644
--- a/.agents/plans/serving-and-speculation/TODO.md
+++ b/.agents/plans/serving-and-speculation/TODO.md
@@ -103,4 +103,19 @@ Contracts: [joint serving/goodput](../../docs/JOINT_CONCURRENT_SERVING_OPTIMIZAT
-- [ ] OA-2: shared typed semantic request/event boundary; model-owned
-  chat/template lowering; CLI run, Chat Completions and legacy Completions
-  consume the same semantic path. Preserve exact source/model/runtime,
-  auth, admission and cancellation; qualify before OA-3 Responses.
+- [x] OA-2: core typed text intent and engine SemanticRequest retained
+  by original queue/worker; legacy single text and SDK/CLI converge.
+  HTTP no longer frames Qwen prompts. Model-owned validation, exact
+  rendered byte and prompt/scratch bounds, shared Qwen renderer, original
+  typed ResponseEvent/receipt. Preserve configured rendered chat cap;
+  typed single-user chat also rejects role delimiters; raw unchanged.
+  Legacy-only service/tokenizer adapters refuse structured chat rather
+  than flatten. Focused engine/serving 163/0/0 and strict lint, tokenizer
+  suite 7/0/0 and actual cached approved-tokenizer test 1/0/0 passed.
+  Actual cached CLI/TUI/Web and SDK regressions passed (67.77s and
+  15.85s); full isolated assembled gate 2,553/0/109 across 318 Rust
+  suites (878.6s), release TUI/Web, installer and scripts passed.
+  No Responses, arbitrary Jinja, new modality, model owner or scheduler.
+- [ ] OA-3: Models list/retrieve and Responses text create/stream through
+  the accepted semantic request/events. Preserve original listener,
+  runtime, admission, cancellation and selected model template. Qualify
+  actual request fields, output/SSE grammar, request IDs, canonical
+  errors and smallest real cached text-model canary. Stateful response/
+  conversation resources belong to OA-4.
~~~~

### D.13 — feat(serving): expose stateless Responses and model retrieval

Commit: `cea0f18c0e3e39d374febd1b723c4dd097fa163e`

Original commit message:

~~~~text
feat(serving): expose stateless Responses and model retrieval

Project the original typed text events through canonical Responses JSON and named SSE, with bounded charged output, actual restored prefix usage and stable correlation IDs. Qualify official cached OpenAI SDK strict validation and real CLI/TUI/Web and Rust SDK lifecycles, plus the full isolated gate (2557 passed, zero failures, 109 skipped). Accept OA-3; stateful resources remain OA-4.
~~~~

Original checkpoint narrative at this commit (historical):

~~~~markdown
### Current checkpoint — OA-3 accepted; OA-4 response/conversation resources next (27 Nov)

Started from clean signed a76ddbfe on L5. Current OpenAI Responses create
and streaming reference checked before selecting the supported text slice:
https://developers.openai.com/api/reference/resources/responses/methods/create
https://developers.openai.com/api/reference/resources/responses/streaming-events
No new dependency acquisition. GET /v1/models and exact GET /v1/models/{id}
use only the accepting original registry aliases; preserve the legacy
created=0 unknown timestamp. POST /v1/responses accepts fixed/chunked
strict JSON, text or ordered message input, text parts concatenated in
their role, instructions as a developer turn, bounded metadata, existing
greedy controls and max output. Explicit full assistant text can be
replayed; its optional item ID is not a stored-resource reference.
No translation through Chat JSON. Same charged SemanticRequest, selected
model template, queue, scheduler, original ResponseEvent and receipt.

Local stateless default is store=false; explicit store=true, background
work, continuation/conversation references, non-text input/output,
structured output, sampling/reasoning/tool capabilities and unsupported
fields refuse before model admission. Neutral no-tools/default text
controls accepted explicitly. OA-4 owns stateful response/conversation
resources and retrieval/continuation; input-token counting still open.
No separate reasoning-output capability. No fake cloud tool behavior.

Responses projection uses borrowed serde views, bounded cumulative payload
writer, original socket wire budget and one conservative projection lease
(6x configured output payload limit plus retained metadata and 64 KiB
headroom). The lease travels with retained response bytes; no new producer
task or event queue. SSE uses named canonical created/in_progress,
item/part added, text deltas, text/part/item done and completed/incomplete/
failed terminal events; monotonic sequence and stable item/response IDs.
Length finish is incomplete; failures retain already delivered partial
text, safe canonical error and no invented usage. Disconnect/stop/output
exhaustion retires original receipt. Additive cached_prompt_tokens on
ResponseEvent::Done reports exact original executor-restored prefix;
legacy presentation unchanged, exhaustive native matches adapted.
Generated request correlation IDs for application HTTP responses/errors
(process namespace/counter); legacy opt-in token metrics stream ID kept.
Parser-level malformed framing remains Hyper-owned. Static parameter names
on typed ServiceError enter the public error envelope, arbitrary diagnostic
messages remain hidden unless explicitly authorized.

Focused engine/serving 167/0/0 and strict lint passed, actual TCP tests
cover model retrieval, typed inputs, fixed/chunked, IDs, all terminal
states, usage, refusal and cancellation/ledger retirement. Existing
cached official OpenAI Python SDK 2.33.0 with strict response validation
passed Models and Responses JSON/SSE, exact actual cached Qwen3.5 Chat
output/usage parity, store refusal and correlation on the real listener.
Optional SDK fixture takes ZG_TEST_OPENAI_PYTHON and
ZG_TEST_OPENAI_PYTHONPATH; isolated existing Nix store closure, no download.
Real native/browser/official SDK lifecycle (80.88s) and Rust SDK
lifecycle (15.02s) each passed 1/0/0. Full isolated assembled gate exited
0 (617.5s): 2,557/0/109 across 318 Rust suites, strict lint, backend
feature boundaries, release TUI/Web, installer and scripts passed.
OA-3 accepted. Next OA-4 bounded response/conversation protocol resources,
storage, cancellation, retrieval/deletion, input items and input-token
counting over the original model/runtime. No model-state store duplication.
Logs: `target/oa3-focused.log`, `target/oa3-real-consumers.log`,
`target/oa3-assembled.log`. Existing exact Nix source-integrity blocker
remains separate. No new physical GPU or performance evidence.
~~~~

Changed paths:

~~~~text
M	.agents/docs/CURRENT_BASELINE.md
M	.agents/docs/CURRENT_HANDOFF.md
M	.agents/plans/FULL_IMPLEMENTATION_PLAN.md
M	.agents/plans/README.md
M	.agents/plans/TODO.md
M	.agents/plans/serving-and-speculation/NEXT_ITERATION_OPENAI_COMPATIBLE_SERVING.md
M	.agents/plans/serving-and-speculation/TODO.md
M	crates/zerogravity-cli/src/control_host/work.rs
A	crates/zerogravity-cli/tests/fixtures/responses_sdk.py
M	crates/zerogravity-cli/tests/run_unified.rs
M	crates/zerogravity-engine/src/router/dispatch.rs
M	crates/zerogravity-engine/src/router/error.rs
M	crates/zerogravity-engine/src/router/worker.rs
M	crates/zerogravity-serving/src/http.rs
M	crates/zerogravity-serving/src/http/body.rs
A	crates/zerogravity-serving/src/http/responses.rs
A	crates/zerogravity-serving/src/http/responses/input.rs
M	crates/zerogravity-serving/src/http/shutdown_tests.rs
M	crates/zerogravity-serving/src/http/transport.rs
M	crates/zerogravity-serving/src/server/tests.rs
M	crates/zerogravity/src/builtins/text/gpu/qualification.rs
M	crates/zerogravity/src/builtins/text/tests.rs
M	crates/zerogravity/src/builtins/text/tests/serving_population/checkpoint.rs
M	crates/zerogravity/src/builtins/text/tokenizer_tests.rs
M	crates/zerogravity/tests/qwen_hip_sdk.rs
M	crates/zerogravity/tests/support/language_pipeline.rs
~~~~

Exact tracked engineering prose delta (zero-context Git diff; full surrounding prose at the cited revision):

~~~~diff
diff --git a/.agents/docs/CURRENT_BASELINE.md b/.agents/docs/CURRENT_BASELINE.md
index 5991691d..2fbc3813 100644
--- a/.agents/docs/CURRENT_BASELINE.md
+++ b/.agents/docs/CURRENT_BASELINE.md
@@ -7,0 +8,12 @@ historical comparison numbers into new implementation decisions.
+## 27 Nov — OA-3 stateless Models and Responses text
+
+Actual registry model retrieval and stateless Responses JSON/SSE over
+original typed requests/events, bounded projection ownership, correlation
+and exact restored-prefix usage. Focused 167/0/0 and strict lint passed;
+installed OpenAI Python SDK 2.33.0 strict real cached-model canary passed
+Models, Responses JSON/SSE, exact Chat parity, refusal and IDs. Real
+native/browser/official SDK (80.88s) and Rust SDK (15.02s) passed.
+Full isolated gate 2,557/0/109 (318 suites, 617.5s), release TUI/Web,
+installer and scripts passed. OA-3 accepted; OA-4 next.
+No storage/continuation, new model capability or physical GPU evidence.
+
diff --git a/.agents/docs/CURRENT_HANDOFF.md b/.agents/docs/CURRENT_HANDOFF.md
index 210cf84f..cd4f7f30 100644
--- a/.agents/docs/CURRENT_HANDOFF.md
+++ b/.agents/docs/CURRENT_HANDOFF.md
@@ -37 +37,62 @@ reusable work and measured history, not precedence over this order.
-### Current checkpoint — OA-2 accepted; OA-3 Models and Responses text next (27 Nov)
+### Current checkpoint — OA-3 accepted; OA-4 response/conversation resources next (27 Nov)
+
+Started from clean signed a76ddbfe on L5. Current OpenAI Responses create
+and streaming reference checked before selecting the supported text slice:
+https://developers.openai.com/api/reference/resources/responses/methods/create
+https://developers.openai.com/api/reference/resources/responses/streaming-events
+No new dependency acquisition. GET /v1/models and exact GET /v1/models/{id}
+use only the accepting original registry aliases; preserve the legacy
+created=0 unknown timestamp. POST /v1/responses accepts fixed/chunked
+strict JSON, text or ordered message input, text parts concatenated in
+their role, instructions as a developer turn, bounded metadata, existing
+greedy controls and max output. Explicit full assistant text can be
+replayed; its optional item ID is not a stored-resource reference.
+No translation through Chat JSON. Same charged SemanticRequest, selected
+model template, queue, scheduler, original ResponseEvent and receipt.
+
+Local stateless default is store=false; explicit store=true, background
+work, continuation/conversation references, non-text input/output,
+structured output, sampling/reasoning/tool capabilities and unsupported
+fields refuse before model admission. Neutral no-tools/default text
+controls accepted explicitly. OA-4 owns stateful response/conversation
+resources and retrieval/continuation; input-token counting still open.
+No separate reasoning-output capability. No fake cloud tool behavior.
+
+Responses projection uses borrowed serde views, bounded cumulative payload
+writer, original socket wire budget and one conservative projection lease
+(6x configured output payload limit plus retained metadata and 64 KiB
+headroom). The lease travels with retained response bytes; no new producer
+task or event queue. SSE uses named canonical created/in_progress,
+item/part added, text deltas, text/part/item done and completed/incomplete/
+failed terminal events; monotonic sequence and stable item/response IDs.
+Length finish is incomplete; failures retain already delivered partial
+text, safe canonical error and no invented usage. Disconnect/stop/output
+exhaustion retires original receipt. Additive cached_prompt_tokens on
+ResponseEvent::Done reports exact original executor-restored prefix;
+legacy presentation unchanged, exhaustive native matches adapted.
+Generated request correlation IDs for application HTTP responses/errors
+(process namespace/counter); legacy opt-in token metrics stream ID kept.
+Parser-level malformed framing remains Hyper-owned. Static parameter names
+on typed ServiceError enter the public error envelope, arbitrary diagnostic
+messages remain hidden unless explicitly authorized.
+
+Focused engine/serving 167/0/0 and strict lint passed, actual TCP tests
+cover model retrieval, typed inputs, fixed/chunked, IDs, all terminal
+states, usage, refusal and cancellation/ledger retirement. Existing
+cached official OpenAI Python SDK 2.33.0 with strict response validation
+passed Models and Responses JSON/SSE, exact actual cached Qwen3.5 Chat
+output/usage parity, store refusal and correlation on the real listener.
+Optional SDK fixture takes ZG_TEST_OPENAI_PYTHON and
+ZG_TEST_OPENAI_PYTHONPATH; isolated existing Nix store closure, no download.
+Real native/browser/official SDK lifecycle (80.88s) and Rust SDK
+lifecycle (15.02s) each passed 1/0/0. Full isolated assembled gate exited
+0 (617.5s): 2,557/0/109 across 318 Rust suites, strict lint, backend
+feature boundaries, release TUI/Web, installer and scripts passed.
+OA-3 accepted. Next OA-4 bounded response/conversation protocol resources,
+storage, cancellation, retrieval/deletion, input items and input-token
+counting over the original model/runtime. No model-state store duplication.
+Logs: `target/oa3-focused.log`, `target/oa3-real-consumers.log`,
+`target/oa3-assembled.log`. Existing exact Nix source-integrity blocker
+remains separate. No new physical GPU or performance evidence.
+
+### Prior checkpoint — OA-2 accepted; OA-3 Models and Responses text next (27 Nov)
diff --git a/.agents/plans/FULL_IMPLEMENTATION_PLAN.md b/.agents/plans/FULL_IMPLEMENTATION_PLAN.md
index 165cd871..cb865fb5 100644
--- a/.agents/plans/FULL_IMPLEMENTATION_PLAN.md
+++ b/.agents/plans/FULL_IMPLEMENTATION_PLAN.md
@@ -782 +782,5 @@ and assembled gate passed (2,553/0/109, 318 suites, 878.6s). OA-2
-accepted; OA-3 Models and Responses text next.
+accepted; OA-3 Models/retrieve and stateless Responses text JSON/SSE
+implemented over the original typed receipt, focused 167/0/0 and strict
+actual cached official SDK canary passed. Native/browser/Rust SDK
+regressions and assembled gate (2,557/0/109, 318 suites, 617.5s)
+passed. OA-3 accepted; OA-4 bounded response/conversation resources next.
diff --git a/.agents/plans/README.md b/.agents/plans/README.md
index 38f6536f..5a41ced2 100644
--- a/.agents/plans/README.md
+++ b/.agents/plans/README.md
@@ -21,3 +21,3 @@ UX-P and the model-local execution selector are accepted. The
-OA-1 public serving transport and OA-2 semantic request/model-template boundary are accepted.
-Current unit: [OA-3 Models and Responses text](serving-and-speculation/NEXT_ITERATION_OPENAI_COMPATIBLE_SERVING.md#oa-3--models--responses-text).
-Read the live handoff for assembled acceptance and ordered OA-3 work;
+OA-1 public serving transport, OA-2 semantic request/model-template boundary and OA-3 Models/stateless Responses text are accepted.
+Current unit: [OA-4 response/conversation resources](serving-and-speculation/NEXT_ITERATION_OPENAI_COMPATIBLE_SERVING.md#oa-4--responseconversation-resources).
+Read the live handoff for assembled acceptance and ordered OA-4 work;
diff --git a/.agents/plans/TODO.md b/.agents/plans/TODO.md
index 06c359b1..37780c2a 100644
--- a/.agents/plans/TODO.md
+++ b/.agents/plans/TODO.md
@@ -236 +236 @@ Final assembled private-network gate on ad7a4178 passed 2,539 passed, zero failu
-public serving (OA-1 async transport and OA-2 shared semantic request/model-template boundary accepted; OA-3 Models and Responses text next), then ordinary benchmarks and ordered performance phases.
+public serving (OA-1 async transport, OA-2 semantic request/model-template boundary and OA-3 stateless Models/Responses text accepted; OA-4 response/conversation resources next), then ordinary benchmarks and ordered performance phases.
diff --git a/.agents/plans/serving-and-speculation/NEXT_ITERATION_OPENAI_COMPATIBLE_SERVING.md b/.agents/plans/serving-and-speculation/NEXT_ITERATION_OPENAI_COMPATIBLE_SERVING.md
index 632b84b2..d7051f47 100644
--- a/.agents/plans/serving-and-speculation/NEXT_ITERATION_OPENAI_COMPATIBLE_SERVING.md
+++ b/.agents/plans/serving-and-speculation/NEXT_ITERATION_OPENAI_COMPATIBLE_SERVING.md
@@ -9 +9 @@ Older ordering below is context; valid owner constraints remain authoritative.
-Status: **OA-1 and OA-2 accepted** (27 Nov). OA-3 Models and Responses text next; later units remain ordered.
+Status: **OA-1 through OA-3 accepted** (27 Nov). OA-4 bounded response/conversation resources next; later units remain ordered.
@@ -1313,0 +1314,28 @@ Use the smallest real text model canary.
+**OA-3 implementation (27 Nov).** Rechecked the current
+[create](https://developers.openai.com/api/reference/resources/responses/methods/create)
+and [streaming](https://developers.openai.com/api/reference/resources/responses/streaming-events)
+reference. Models list/retrieve use accepting registered aliases only.
+Responses text input string or ordered explicit message items, plain
+text parts, instructions (developer), metadata and greedy generation
+controls lower directly to SemanticRequest. Local stateless default
+store=false; explicit store=true, background, prior-response/conversation
+references, tools, structured output, other modalities/sampling/reasoning
+and unsupported fields refuse with named static parameters where known.
+No implicit resource lookup from a replayed complete assistant item ID.
+
+Borrowed typed wire views and charged bounded projection over original
+ResponseEvent/RequestEvents; exact restored prefix from the executor
+cursor is exposed as additive cached_prompt_tokens on Done and projected
+into input_tokens_details. Separate reasoning capability not claimed.
+Named ordered SSE lifecycle, length -> incomplete, backend failure ->
+failed with already-delivered partial text and safe canonical code.
+No Chat chunks or [DONE] sentinel. Same listener, queue/cancellation,
+deadline, backpressure, socket cumulative budget. Server correlation ID
+and response/item IDs; preserve explicit legacy metrics-stream wire ID.
+Focused 167/0/0 and strict lint; cached official OpenAI SDK 2.33.0 strict
+validation passed actual cached Qwen3.5 Models/retrieve, Responses JSON
+and SSE, exact Chat output/usage parity, refusal and correlation. Native/
+browser/official SDK lifecycle (80.88s) and Rust SDK lifecycle (15.02s)
+passed. Full isolated gate 2,557/0/109 across 318 Rust suites (617.5s),
+release TUI/Web, installer and scripts passed. OA-3 accepted.
+
diff --git a/.agents/plans/serving-and-speculation/TODO.md b/.agents/plans/serving-and-speculation/TODO.md
index 2d653fc4..feebafcd 100644
--- a/.agents/plans/serving-and-speculation/TODO.md
+++ b/.agents/plans/serving-and-speculation/TODO.md
@@ -116,6 +116,18 @@ Contracts: [joint serving/goodput](../../docs/JOINT_CONCURRENT_SERVING_OPTIMIZAT
-- [ ] OA-3: Models list/retrieve and Responses text create/stream through
-  the accepted semantic request/events. Preserve original listener,
-  runtime, admission, cancellation and selected model template. Qualify
-  actual request fields, output/SSE grammar, request IDs, canonical
-  errors and smallest real cached text-model canary. Stateful response/
-  conversation resources belong to OA-4.
+- [x] OA-3: Models list/retrieve accepting aliases, stateless Responses
+  text create/stream directly on SemanticRequest and original receipt.
+  Typed text items/instructions, bounded metadata/greedy controls,
+  charged bounded projection and exact restored prefix usage. Canonical
+  ordered SSE completed/incomplete/failed, server request/response/item
+  IDs, named parameter refusals; preserve legacy metrics stream ID.
+  Focused 167/0/0 and strict lint passed. Cached official OpenAI SDK
+  2.33.0 strict actual Qwen3.5 Models/Responses JSON/SSE and Chat parity
+  passed; real native/browser/official SDK (80.88s), Rust SDK (15.02s)
+  and full isolated gate 2,557/0/109 (318 suites, 617.5s), release
+  TUI/Web, installer and scripts passed. Store defaults false locally;
+  explicit persistence/continuation, tools, non-text and unsupported
+  controls refuse. OA-3 accepted.
+- [ ] OA-4: bounded response/conversation resource owner, explicit store/
+  continuation, cancellation, retrieval/deletion, input-item retrieval
+  and input-token counting. Recheck current API contract, preserve
+  original model/source/session/scheduler/receipt; no duplicate model
+  state store. Qualify actual cached text-model/SDK lifecycle.
~~~~

### D.14 — feat(serving): retain bounded foreground response resources

Commit: `0898734b0229dfb231e487b7b4d9b491fa2104ed`

Original commit message:

~~~~text
feat(serving): retain bounded foreground response resources

Accept OA-4A explicit memory-only stored responses, retrieval, deletion, paginated normalized input items and previous-response continuation. Reserve the listener quota and physical admission before generation; retire on expiry, disconnect and last reader. Preserve original semantic request, model template, listener and scheduler. Qualify strict cached OpenAI SDK lifecycle and complete isolated workspace gate (2560 passed, zero failed, 109 skipped). Background cancellation, conversations and input token counting remain OA-4.
~~~~

Original checkpoint narrative at this commit (historical):

~~~~markdown
### Current checkpoint — OA-4A accepted; OA-4 background/conversations/count next (27 Nov)

Started from clean signed cea0f18c on L5. Current Responses retrieve,
delete, input_items, create/previous_response_id, and cancel reference
reviewed. Explicit cancellation is a background-response capability;
do not report a cancelled foreground response. OA-4A enables explicit
store=true for foreground text; local default remains store=false.
Listener-local memory-only Store, one quota and host admission.
64 pending/retained record slots, 32 MiB aggregate conservative backing
reservation, 30-minute TTL from terminal publication; expire on access
and a 30s accept-loop tick, drop all on joined listener stop. Quota and
ServingCharge follow the last reader even after logical deletion/expiry
(Bytes::from_owner retains the record). Reserve before original model
admission, drop pending records on failed ingress/admission/disconnect.
No disk persistence, session/model-state retention or background task.
All resource routes pass existing listener bearer/loopback trust policy;
single listener trust scope, not multitenant access control.

Retain normalized input text (excluding request-local instructions),
terminal assistant text and immutable bounded serialized response.
Store cap clamps the configured wire payload bound to at most 1 MiB;
each record retains conservative worst-case reservation including
serialization growth and actual history, so byte quota can be reached
before count quota. Successful/incomplete foreground JSON and SSE
terminal records support retrieval and deletion; streamed failed records
retain safe terminal/partial text, never opt-in diagnostic details and
cannot be continued. Foreground nonstream generation error keeps the
original HTTP error and publishes no stored record. Pending foreground
records are not addressable before terminal publication. Existing socket
drop cancels original request. Explicit POST cancel of a stored
foreground record refuses per current contract; background is still
unsupported. GET /v1/responses/{id}/input_items supports bounded
limit/order/after pagination of normalized history, server-owned stable
IDs within each response snapshot (not cross-response item identities).
previous_response_id reads the completed/incomplete stored parent,
copies charged full history plus actual assistant text, prepends only
current instructions, applies original input/context caps and submits
the requested selected model. Child snapshots do not retain parent
pointers; deleting a parent does not invalidate existing children.
No token count/session claim. Strict unknown/unsupported fields refuse.

Focused engine/serving 170/0/0 and strict lint passed: actual TCP tests
verify store/retrieve/pagination/deletion/continuation, stateless refusal,
SSE terminal retention, failed-record continuation refusal, authorization
and joined disconnect/shutdown. Quota test covers pending count, byte
bound, expiry and retained wire reader retirement. Cached official SDK
2.33.0 canary extended with real stored responses/continuation, strict
schema and Chat parity. First canary failed only because SDK delete()
returns None; fixed the fixture to assert its actual raw deletion
object and later 404, no production change. Final real CLI/TUI/Web/
official SDK 1/0/0 (134.83s) and Rust SDK 1/0/0 (15.18s) passed.
Full isolated assembled gate exited 0 (690.5s): 2,560/0/109 across
318 Rust suites, backend boundaries, strict lint, release TUI/Web,
installer and scripts passed. OA-4A accepted. Logs:
`target/oa4a-focused-final.log`,
`target/oa4a-real-consumers.log`, `target/oa4a-real-consumers-final.log`,
`target/oa4a-assembled.log`. No new dependencies/downloads. Exact Nix
source-integrity remains separately blocked by unavailable locked inputs.

Remaining OA-4: explicit background response owner and cancel semantics
(joined bounded work in existing listener runtime; no detached task or
scheduler), conversation CRUD/items and atomic context append with the
same protocol quota, exact input-token counting through original charged
model tokenizer without opening a physical generation session. Explicit
unsupported until each is qualified. Do not mark OA-4 complete yet.
~~~~

Changed paths:

~~~~text
M	.agents/docs/CURRENT_BASELINE.md
M	.agents/docs/CURRENT_HANDOFF.md
M	.agents/plans/FULL_IMPLEMENTATION_PLAN.md
M	.agents/plans/serving-and-speculation/NEXT_ITERATION_OPENAI_COMPATIBLE_SERVING.md
M	.agents/plans/serving-and-speculation/TODO.md
M	crates/zerogravity-cli/tests/fixtures/responses_sdk.py
M	crates/zerogravity-engine/src/router/error.rs
M	crates/zerogravity-serving/src/http/body.rs
M	crates/zerogravity-serving/src/http/responses.rs
M	crates/zerogravity-serving/src/http/responses/input.rs
A	crates/zerogravity-serving/src/http/responses/store.rs
M	crates/zerogravity-serving/src/http/shutdown_tests.rs
M	crates/zerogravity-serving/src/http/transport.rs
M	crates/zerogravity-serving/src/server/tests.rs
~~~~

Exact tracked engineering prose delta (zero-context Git diff; full surrounding prose at the cited revision):

~~~~diff
diff --git a/.agents/docs/CURRENT_BASELINE.md b/.agents/docs/CURRENT_BASELINE.md
index 2fbc3813..faf11b6d 100644
--- a/.agents/docs/CURRENT_BASELINE.md
+++ b/.agents/docs/CURRENT_BASELINE.md
@@ -7,0 +8,11 @@ historical comparison numbers into new implementation decisions.
+## 27 Nov — OA-4A foreground response resources
+
+Memory-only charged listener records, retrieval/deletion/input pages,
+previous-response text replay excluding old instructions, stateless
+default retained. Focused 170/0/0 and strict lint passed; real installed
+SDK/CLI/TUI/Web (134.83s) and Rust SDK (15.18s) passed. Assembled
+gate 2,560/0/109 across 318 Rust suites (690.5s), release TUI/Web,
+installer and scripts passed. OA-4A accepted. Background/cancel,
+conversations and input token counting remain OA-4. No new model or
+physical GPU/performance claim.
+
diff --git a/.agents/docs/CURRENT_HANDOFF.md b/.agents/docs/CURRENT_HANDOFF.md
index cd4f7f30..3390f40d 100644
--- a/.agents/docs/CURRENT_HANDOFF.md
+++ b/.agents/docs/CURRENT_HANDOFF.md
@@ -37 +37,67 @@ reusable work and measured history, not precedence over this order.
-### Current checkpoint — OA-3 accepted; OA-4 response/conversation resources next (27 Nov)
+### Current checkpoint — OA-4A accepted; OA-4 background/conversations/count next (27 Nov)
+
+Started from clean signed cea0f18c on L5. Current Responses retrieve,
+delete, input_items, create/previous_response_id, and cancel reference
+reviewed. Explicit cancellation is a background-response capability;
+do not report a cancelled foreground response. OA-4A enables explicit
+store=true for foreground text; local default remains store=false.
+Listener-local memory-only Store, one quota and host admission.
+64 pending/retained record slots, 32 MiB aggregate conservative backing
+reservation, 30-minute TTL from terminal publication; expire on access
+and a 30s accept-loop tick, drop all on joined listener stop. Quota and
+ServingCharge follow the last reader even after logical deletion/expiry
+(Bytes::from_owner retains the record). Reserve before original model
+admission, drop pending records on failed ingress/admission/disconnect.
+No disk persistence, session/model-state retention or background task.
+All resource routes pass existing listener bearer/loopback trust policy;
+single listener trust scope, not multitenant access control.
+
+Retain normalized input text (excluding request-local instructions),
+terminal assistant text and immutable bounded serialized response.
+Store cap clamps the configured wire payload bound to at most 1 MiB;
+each record retains conservative worst-case reservation including
+serialization growth and actual history, so byte quota can be reached
+before count quota. Successful/incomplete foreground JSON and SSE
+terminal records support retrieval and deletion; streamed failed records
+retain safe terminal/partial text, never opt-in diagnostic details and
+cannot be continued. Foreground nonstream generation error keeps the
+original HTTP error and publishes no stored record. Pending foreground
+records are not addressable before terminal publication. Existing socket
+drop cancels original request. Explicit POST cancel of a stored
+foreground record refuses per current contract; background is still
+unsupported. GET /v1/responses/{id}/input_items supports bounded
+limit/order/after pagination of normalized history, server-owned stable
+IDs within each response snapshot (not cross-response item identities).
+previous_response_id reads the completed/incomplete stored parent,
+copies charged full history plus actual assistant text, prepends only
+current instructions, applies original input/context caps and submits
+the requested selected model. Child snapshots do not retain parent
+pointers; deleting a parent does not invalidate existing children.
+No token count/session claim. Strict unknown/unsupported fields refuse.
+
+Focused engine/serving 170/0/0 and strict lint passed: actual TCP tests
+verify store/retrieve/pagination/deletion/continuation, stateless refusal,
+SSE terminal retention, failed-record continuation refusal, authorization
+and joined disconnect/shutdown. Quota test covers pending count, byte
+bound, expiry and retained wire reader retirement. Cached official SDK
+2.33.0 canary extended with real stored responses/continuation, strict
+schema and Chat parity. First canary failed only because SDK delete()
+returns None; fixed the fixture to assert its actual raw deletion
+object and later 404, no production change. Final real CLI/TUI/Web/
+official SDK 1/0/0 (134.83s) and Rust SDK 1/0/0 (15.18s) passed.
+Full isolated assembled gate exited 0 (690.5s): 2,560/0/109 across
+318 Rust suites, backend boundaries, strict lint, release TUI/Web,
+installer and scripts passed. OA-4A accepted. Logs:
+`target/oa4a-focused-final.log`,
+`target/oa4a-real-consumers.log`, `target/oa4a-real-consumers-final.log`,
+`target/oa4a-assembled.log`. No new dependencies/downloads. Exact Nix
+source-integrity remains separately blocked by unavailable locked inputs.
+
+Remaining OA-4: explicit background response owner and cancel semantics
+(joined bounded work in existing listener runtime; no detached task or
+scheduler), conversation CRUD/items and atomic context append with the
+same protocol quota, exact input-token counting through original charged
+model tokenizer without opening a physical generation session. Explicit
+unsupported until each is qualified. Do not mark OA-4 complete yet.
+
+### Prior checkpoint — OA-3 accepted; OA-4 response/conversation resources next (27 Nov)
diff --git a/.agents/plans/FULL_IMPLEMENTATION_PLAN.md b/.agents/plans/FULL_IMPLEMENTATION_PLAN.md
index cb865fb5..543dcbf5 100644
--- a/.agents/plans/FULL_IMPLEMENTATION_PLAN.md
+++ b/.agents/plans/FULL_IMPLEMENTATION_PLAN.md
@@ -786 +786,7 @@ regressions and assembled gate (2,557/0/109, 318 suites, 617.5s)
-passed. OA-3 accepted; OA-4 bounded response/conversation resources next.
+passed. OA-3 accepted. OA-4A foreground response storage/retrieval,
+deletion/input pages and continuation implemented using the original
+semantic request and listener-owned bounded charged protocol store;
+focused 170/0/0, strict lint and real installed official SDK/CLI/TUI/Web/
+Rust SDK and full isolated gate 2,560/0/109 (318 suites, 690.5s)
+passed. OA-4A accepted. OA-4 remains open for background cancellation,
+conversations/items and exact input-token count.
diff --git a/.agents/plans/serving-and-speculation/NEXT_ITERATION_OPENAI_COMPATIBLE_SERVING.md b/.agents/plans/serving-and-speculation/NEXT_ITERATION_OPENAI_COMPATIBLE_SERVING.md
index d7051f47..520aa620 100644
--- a/.agents/plans/serving-and-speculation/NEXT_ITERATION_OPENAI_COMPATIBLE_SERVING.md
+++ b/.agents/plans/serving-and-speculation/NEXT_ITERATION_OPENAI_COMPATIBLE_SERVING.md
@@ -9 +9 @@ Older ordering below is context; valid owner constraints remain authoritative.
-Status: **OA-1 through OA-3 accepted** (27 Nov). OA-4 bounded response/conversation resources next; later units remain ordered.
+Status: **OA-1 through OA-3 and OA-4A foreground response resources accepted** (27 Nov). OA-4 remains open for background, conversations and token count; later units remain ordered.
@@ -1356,0 +1357,24 @@ Keep protocol state separate from model StateRoot.
+**OA-4A foreground resource slice (27 Nov).** Explicit store=true, terminal
+response retrieval/deletion, paginated normalized input items and
+previous_response_id text continuation. Memory-only single listener
+trust scope, bounded count (64), conservative reserved backing (32 MiB)
+and terminal age (30 min); listener-owned expiry and final reader
+ownership. Reserve before model admission; no retained model session,
+disk storage, detached tasks or additional scheduler. Record only
+normalized history and output, serialized terminal view. Omitted store
+stays false. Prior request-local instructions are not inherited; child
+snapshots survive parent deletion. Pending foreground responses are
+not addressable before terminal; failed stream records cannot continue.
+Explicit cancel of foreground record refuses (current API requires
+background). See live handoff for exact bounds and retention policy.
+Focused 170/0/0 and strict lint pass; real installed SDK 2.33.0,
+CLI/TUI/Web (134.83s) and Rust SDK (15.18s) lifecycle passed.
+Full isolated assembled gate 2,560/0/109 (318 suites, 690.5s),
+release TUI/Web, installer and scripts passed. OA-4A accepted.
+
+Remaining: background work/cancel with joined owner; conversation/items
+CRUD and atomic append on successful generation; exact input_tokens
+count without generation through original charged tokenizer. No false
+capability until qualified. Retain one protocol quota; no model-state
+store or second scheduler.
+
diff --git a/.agents/plans/serving-and-speculation/TODO.md b/.agents/plans/serving-and-speculation/TODO.md
index feebafcd..e4b2c960 100644
--- a/.agents/plans/serving-and-speculation/TODO.md
+++ b/.agents/plans/serving-and-speculation/TODO.md
@@ -129,5 +129,16 @@ Contracts: [joint serving/goodput](../../docs/JOINT_CONCURRENT_SERVING_OPTIMIZAT
-- [ ] OA-4: bounded response/conversation resource owner, explicit store/
-  continuation, cancellation, retrieval/deletion, input-item retrieval
-  and input-token counting. Recheck current API contract, preserve
-  original model/source/session/scheduler/receipt; no duplicate model
-  state store. Qualify actual cached text-model/SDK lifecycle.
+- [x] OA-4A: foreground store/retrieve/delete/input-items pagination and
+  previous_response_id. Listener-local memory-only charged quota 64
+  pending/retained slots, 32 MiB conservative reservation, 30min expiry
+  and final-reader credit retirement. Normalized text history excludes
+  prior instructions; actual assistant output replay, no parent chain
+  or model-state retention. Foreground cancellation explicitly refused;
+  disconnect cancels original receipt. Focused 170/0/0 and strict lint
+  passed. Real installed SDK 2.33.0/CLI/TUI/Web (134.83s) and Rust SDK
+  (15.18s) passed, including exact Chat parity and deletion. Full
+  isolated gate 2,560/0/109 (318 suites, 690.5s), release TUI/Web,
+  installer and scripts passed. OA-4A accepted.
+- [ ] OA-4 remaining: bounded joined background response execution and
+  explicit cancellation; conversation/item resources and atomic context
+  append using the same quota; exact input-token counting without
+  generation through original charged tokenizer. No duplicate model
+  state store, scheduler or fake capability. Real SDK qualification.
~~~~

### D.15 — feat(serving): supervise bounded background responses and cancellation

Commit: `89822f4038b2048affd8b92dac6c7b2bcbeb219e`

Original commit message:

~~~~text
feat(serving): supervise bounded background responses and cancellation

Accept OA-4B background polling and cancellation on the original inference receipt and listener-owned join set. Share charged protocol storage; preserve foreground behavior, terminal race ownership, safe failure and shutdown retirement. Qualify actual cached official SDK cancellation and complete isolated workspace gate (2562 passed, zero failures, 109 skips). Conversations, input token count and background replay remain OA-4.
~~~~

Original checkpoint narrative at this commit (historical):

~~~~markdown
### Current checkpoint — OA-4B accepted; OA-4 conversations/count/replay next (27 Nov)

Started from clean signed 0898734b on L5. Rechecked current background
mode and cancel contracts:
https://developers.openai.com/api/docs/guides/background
https://developers.openai.com/api/reference/resources/responses/methods/cancel
Support text background=true with nonstream create/poll/cancel, default
soonest completion window. Explicit background_options.completion_window
soonest accepted; other windows/continuation-ID controls refuse.
Background streaming/resumption still refuses until a bounded replay
contract is qualified. No webhook or cloud capacity/discount claim.

One listener-owned JoinSet supervised by the existing accept loop
(reap even without incoming connections, abort/join after ingress stop).
Hard maximum 64 task handles including completed/unjoined handles.
No extra inference queue, worker, scheduler, detached task or model state.
Reserve the original shared protocol quota before model admission:
64 pending/retained record slots, 32 MiB conservative backing, same
original ServingCharge. Background reserves 7x at most 1 MiB terminal
serialization limit plus actual input backing and 64 KiB headroom for
immutable initial/cancel/failure/terminal projections and task state;
options metadata retains its own allowance, no unused SSE reservation.
Foreground reservations unchanged. Initial in_progress means accepted
protocol processing, not physical GPU execution or an exact scheduler
phase. After admission, mandatory snapshots are prepared before
publishing the ID and spawning the joined consumer. If preparation
fails, the original receipt is dropped and cancelled; no published ID.
Background owns the original nonstream event receipt independently of
the creation socket, shares the same absolute 180s generation deadline
and terminal parser as foreground JSON. Initial and terminal snapshots
retain one shared input and credit; polling uses Bytes owner references,
never another strong inference receipt. Safe failure records, including
oversized terminal output, always have a pre-priced failed fallback.
No partial output/usage fabricated for cancellation or failed nonstream.

Cancel signals the original receipt, wakes its joined consumer and
atomically replaces the active entry with cancelled; terminal publication
checks active entry identity, so a late Done cannot revive a cancelled
or deleted record. Repeated cancel on terminal background returns the
same final object. Active DELETE refuses until terminal. Foreground
cancel still refuses; foreground disconnect still cancels its request.
Logical cancellation is not a physical completion proof; original
engine owns physical retirement. All resource routes retain original
listener authorization and bounded socket output.

Explicit store=true retains terminal resources 30min and permits
successful/incomplete continuation. Background store=false retains
only temporary polling/input-item resources for 10min from terminal,
no continuation; ordinary foreground store=false retains nothing.
Both remain memory-only and expire on access/idle listener tick or
listener shutdown. Shared quota follows all retained readers even after
expiry/deletion; active tasks cannot age out before their deadline.
These are local bounded policies, not cloud retention guarantees.

Focused engine/serving 172/0/0 and strict lint passed. Actual held-receipt
TCP tests verify create disconnect independence, input items during
processing, active deletion refusal, cancellation/late terminal race,
idempotence, unrelated worker completion and usage, temporary continuation
refusal, joined shutdown and ledger retirement. Worker error and oversized
terminal output publish safe failed records. Existing installed OpenAI
Python SDK 2.33.0 real cached model fixture now exercises background
polling, text/usage parity, terminal cancel idempotence, cancellation
race and stored vs temporary continuation. Actual official SDK
cancellation returned cancelled, and the cached CLI/TUI/Web/SDK
lifecycle passed (130.88s); Rust SDK passed (14.36s).
Full isolated assembled gate exited 0 (377.3s): 2,562/0/109 across
318 Rust suites, backend feature boundaries, strict lint, release TUI/Web,
installer and scripts passed. OA-4B accepted. Build concurrency reduced to 4 jobs per lane
for observed memory pressure; no new dependencies/downloads.
Logs: `target/oa4b-focused.log`, `target/oa4b-real-consumers.log`,
`target/oa4b-assembled.log`. Exact Nix source-integrity remains separately
blocked by unavailable locked inputs. No new physical GPU/performance
claim.

Remaining OA-4: conversations/items CRUD and atomic context append
using same protocol quota; exact input-token counting through original
charged model tokenizer without opening generation; bounded background
streaming/replay and resume contract. No fake support, additional model
state store or scheduler. Do not mark OA-4 complete.
~~~~

Changed paths:

~~~~text
M	.agents/docs/CURRENT_BASELINE.md
M	.agents/docs/CURRENT_HANDOFF.md
M	.agents/plans/FULL_IMPLEMENTATION_PLAN.md
M	.agents/plans/serving-and-speculation/NEXT_ITERATION_OPENAI_COMPATIBLE_SERVING.md
M	.agents/plans/serving-and-speculation/TODO.md
M	crates/zerogravity-cli/tests/fixtures/responses_sdk.py
M	crates/zerogravity-serving/src/http/responses.rs
A	crates/zerogravity-serving/src/http/responses/background.rs
M	crates/zerogravity-serving/src/http/responses/input.rs
M	crates/zerogravity-serving/src/http/responses/store.rs
M	crates/zerogravity-serving/src/http/shutdown_tests.rs
M	crates/zerogravity-serving/src/http/transport.rs
M	crates/zerogravity-serving/src/server/tests.rs
~~~~

Exact tracked engineering prose delta (zero-context Git diff; full surrounding prose at the cited revision):

~~~~diff
diff --git a/.agents/docs/CURRENT_BASELINE.md b/.agents/docs/CURRENT_BASELINE.md
index faf11b6d..42ce0e22 100644
--- a/.agents/docs/CURRENT_BASELINE.md
+++ b/.agents/docs/CURRENT_BASELINE.md
@@ -7,0 +8,12 @@ historical comparison numbers into new implementation decisions.
+## 27 Nov — OA-4B background polling and cancellation
+
+Listener-owned joined nonstream background jobs over original receipt,
+shared charged protocol quota and immutable polling snapshots; actual
+cancellation, terminal race/idempotence and safe failure fallback.
+Focused 172/0/0 and strict lint passed; real installed SDK/CLI/TUI/Web
+(130.88s), including actual cancelled, and Rust SDK (14.36s) passed.
+Full isolated assembled gate 2,562/0/109 (318 suites, 377.3s), release
+TUI/Web, installer and scripts passed. OA-4B accepted. Temporary store=false polling, no
+continuation; background SSE/replay, conversations and input token
+counting remain OA-4. No new physical GPU/performance claim.
+
diff --git a/.agents/docs/CURRENT_HANDOFF.md b/.agents/docs/CURRENT_HANDOFF.md
index 3390f40d..3f9ab11e 100644
--- a/.agents/docs/CURRENT_HANDOFF.md
+++ b/.agents/docs/CURRENT_HANDOFF.md
@@ -37 +37,81 @@ reusable work and measured history, not precedence over this order.
-### Current checkpoint — OA-4A accepted; OA-4 background/conversations/count next (27 Nov)
+### Current checkpoint — OA-4B accepted; OA-4 conversations/count/replay next (27 Nov)
+
+Started from clean signed 0898734b on L5. Rechecked current background
+mode and cancel contracts:
+https://developers.openai.com/api/docs/guides/background
+https://developers.openai.com/api/reference/resources/responses/methods/cancel
+Support text background=true with nonstream create/poll/cancel, default
+soonest completion window. Explicit background_options.completion_window
+soonest accepted; other windows/continuation-ID controls refuse.
+Background streaming/resumption still refuses until a bounded replay
+contract is qualified. No webhook or cloud capacity/discount claim.
+
+One listener-owned JoinSet supervised by the existing accept loop
+(reap even without incoming connections, abort/join after ingress stop).
+Hard maximum 64 task handles including completed/unjoined handles.
+No extra inference queue, worker, scheduler, detached task or model state.
+Reserve the original shared protocol quota before model admission:
+64 pending/retained record slots, 32 MiB conservative backing, same
+original ServingCharge. Background reserves 7x at most 1 MiB terminal
+serialization limit plus actual input backing and 64 KiB headroom for
+immutable initial/cancel/failure/terminal projections and task state;
+options metadata retains its own allowance, no unused SSE reservation.
+Foreground reservations unchanged. Initial in_progress means accepted
+protocol processing, not physical GPU execution or an exact scheduler
+phase. After admission, mandatory snapshots are prepared before
+publishing the ID and spawning the joined consumer. If preparation
+fails, the original receipt is dropped and cancelled; no published ID.
+Background owns the original nonstream event receipt independently of
+the creation socket, shares the same absolute 180s generation deadline
+and terminal parser as foreground JSON. Initial and terminal snapshots
+retain one shared input and credit; polling uses Bytes owner references,
+never another strong inference receipt. Safe failure records, including
+oversized terminal output, always have a pre-priced failed fallback.
+No partial output/usage fabricated for cancellation or failed nonstream.
+
+Cancel signals the original receipt, wakes its joined consumer and
+atomically replaces the active entry with cancelled; terminal publication
+checks active entry identity, so a late Done cannot revive a cancelled
+or deleted record. Repeated cancel on terminal background returns the
+same final object. Active DELETE refuses until terminal. Foreground
+cancel still refuses; foreground disconnect still cancels its request.
+Logical cancellation is not a physical completion proof; original
+engine owns physical retirement. All resource routes retain original
+listener authorization and bounded socket output.
+
+Explicit store=true retains terminal resources 30min and permits
+successful/incomplete continuation. Background store=false retains
+only temporary polling/input-item resources for 10min from terminal,
+no continuation; ordinary foreground store=false retains nothing.
+Both remain memory-only and expire on access/idle listener tick or
+listener shutdown. Shared quota follows all retained readers even after
+expiry/deletion; active tasks cannot age out before their deadline.
+These are local bounded policies, not cloud retention guarantees.
+
+Focused engine/serving 172/0/0 and strict lint passed. Actual held-receipt
+TCP tests verify create disconnect independence, input items during
+processing, active deletion refusal, cancellation/late terminal race,
+idempotence, unrelated worker completion and usage, temporary continuation
+refusal, joined shutdown and ledger retirement. Worker error and oversized
+terminal output publish safe failed records. Existing installed OpenAI
+Python SDK 2.33.0 real cached model fixture now exercises background
+polling, text/usage parity, terminal cancel idempotence, cancellation
+race and stored vs temporary continuation. Actual official SDK
+cancellation returned cancelled, and the cached CLI/TUI/Web/SDK
+lifecycle passed (130.88s); Rust SDK passed (14.36s).
+Full isolated assembled gate exited 0 (377.3s): 2,562/0/109 across
+318 Rust suites, backend feature boundaries, strict lint, release TUI/Web,
+installer and scripts passed. OA-4B accepted. Build concurrency reduced to 4 jobs per lane
+for observed memory pressure; no new dependencies/downloads.
+Logs: `target/oa4b-focused.log`, `target/oa4b-real-consumers.log`,
+`target/oa4b-assembled.log`. Exact Nix source-integrity remains separately
+blocked by unavailable locked inputs. No new physical GPU/performance
+claim.
+
+Remaining OA-4: conversations/items CRUD and atomic context append
+using same protocol quota; exact input-token counting through original
+charged model tokenizer without opening generation; bounded background
+streaming/replay and resume contract. No fake support, additional model
+state store or scheduler. Do not mark OA-4 complete.
+
+### Prior checkpoint — OA-4A accepted; OA-4 background/conversations/count next (27 Nov)
diff --git a/.agents/plans/FULL_IMPLEMENTATION_PLAN.md b/.agents/plans/FULL_IMPLEMENTATION_PLAN.md
index 543dcbf5..fe003094 100644
--- a/.agents/plans/FULL_IMPLEMENTATION_PLAN.md
+++ b/.agents/plans/FULL_IMPLEMENTATION_PLAN.md
@@ -791,2 +791,7 @@ Rust SDK and full isolated gate 2,560/0/109 (318 suites, 690.5s)
-passed. OA-4A accepted. OA-4 remains open for background cancellation,
-conversations/items and exact input-token count.
+passed. OA-4A accepted. OA-4B joined nonstream background polling/
+cancel implemented on original receipt and shared quota; focused
+172/0/0 and strict lint passed; real official SDK/CLI/TUI/Web and Rust
+SDK passed, including actual cancelled. Full isolated assembled gate
+2,562/0/109 (318 suites, 377.3s) passed. OA-4B accepted.
+OA-4 remains open for conversations/items, exact input-token count
+and bounded background stream/replay.
diff --git a/.agents/plans/serving-and-speculation/NEXT_ITERATION_OPENAI_COMPATIBLE_SERVING.md b/.agents/plans/serving-and-speculation/NEXT_ITERATION_OPENAI_COMPATIBLE_SERVING.md
index 520aa620..4f597db4 100644
--- a/.agents/plans/serving-and-speculation/NEXT_ITERATION_OPENAI_COMPATIBLE_SERVING.md
+++ b/.agents/plans/serving-and-speculation/NEXT_ITERATION_OPENAI_COMPATIBLE_SERVING.md
@@ -9 +9 @@ Older ordering below is context; valid owner constraints remain authoritative.
-Status: **OA-1 through OA-3 and OA-4A foreground response resources accepted** (27 Nov). OA-4 remains open for background, conversations and token count; later units remain ordered.
+Status: **OA-1 through OA-3 and OA-4A/B accepted** (27 Nov). OA-4 remains open for conversations, token count and background replay; later units remain ordered.
@@ -1375,5 +1375,24 @@ release TUI/Web, installer and scripts passed. OA-4A accepted.
-Remaining: background work/cancel with joined owner; conversation/items
-CRUD and atomic append on successful generation; exact input_tokens
-count without generation through original charged tokenizer. No false
-capability until qualified. Retain one protocol quota; no model-state
-store or second scheduler.
+**OA-4B background polling/cancellation (27 Nov).** Rechecked current
+[background mode](https://developers.openai.com/api/docs/guides/background)
+and [cancel](https://developers.openai.com/api/reference/resources/responses/methods/cancel).
+Nonstream background=true create/poll/cancel through listener-supervised
+bounded JoinSet and original event receipt; no second inference queue.
+Same protocol quota and charged immutable snapshots. Actual receipt
+cancel and atomic active-entry transition, late terminal cannot overwrite;
+repeated cancel returns final object. Active delete refuses. Safe failure
+fallback even on terminal serialization overflow; original deadline
+and shutdown retirement. Local store=true 30min with continuation,
+background store=false temporary 10min with polling only, foreground
+store=false remains unretained. Default/soonest completion window only;
+background streaming/resume refuses until bounded replay exists.
+Focused 172/0/0 and strict lint passed. Real installed SDK/CLI/TUI/Web
+(130.88s) and Rust SDK (14.36s) passed, including actual cancelled
+on the cached model. Full isolated gate 2,562/0/109 across 318 Rust
+suites (377.3s), release TUI/Web, installer and scripts passed.
+OA-4B accepted.
+
+Remaining: conversation/items CRUD and atomic append on successful
+generation; exact input_tokens count without generation through original
+charged tokenizer; bounded background streaming/replay and resume.
+Retain one protocol quota, no model-state store or second scheduler.
+No false capability until qualified.
diff --git a/.agents/plans/serving-and-speculation/TODO.md b/.agents/plans/serving-and-speculation/TODO.md
index e4b2c960..cefb4e25 100644
--- a/.agents/plans/serving-and-speculation/TODO.md
+++ b/.agents/plans/serving-and-speculation/TODO.md
@@ -140,2 +140,11 @@ Contracts: [joint serving/goodput](../../docs/JOINT_CONCURRENT_SERVING_OPTIMIZAT
-- [ ] OA-4 remaining: bounded joined background response execution and
-  explicit cancellation; conversation/item resources and atomic context
+- [x] OA-4B: bounded joined nonstream background response execution,
+  independent of create connection, polling and explicit original receipt
+  cancellation, terminal race/idempotence, safe failure fallback and
+  joined shutdown. Same quota and final reader ownership, hard 64 task
+  handles. store=true 30min continuation, background store=false 10min
+  temporary polling only; foreground unchanged. Default/soonest window.
+  Background stream/replay refused. Focused 172/0/0, strict lint passed;
+  real installed SDK/CLI/TUI/Web (130.88s, actual cancelled) and Rust
+  SDK (14.36s) passed. Full isolated gate 2,562/0/109 (318 suites,
+  377.3s), release TUI/Web, installer and scripts passed. OA-4B accepted.
+- [ ] OA-4 remaining: conversation/item resources and atomic context
@@ -143,2 +152,3 @@ Contracts: [joint serving/goodput](../../docs/JOINT_CONCURRENT_SERVING_OPTIMIZAT
-  generation through original charged tokenizer. No duplicate model
-  state store, scheduler or fake capability. Real SDK qualification.
+  generation through original charged tokenizer; bounded background
+  streaming/replay and resume. No duplicate model state store,
+  scheduler or fake capability. Real SDK qualification.
~~~~

### D.16 — reconciled current baseline and implementation evidence

Exact latest baseline entries from the accepted source; complete original baseline remains linked.

~~~~markdown
## 27 Nov — OA-4B background polling and cancellation

Listener-owned joined nonstream background jobs over original receipt,
shared charged protocol quota and immutable polling snapshots; actual
cancellation, terminal race/idempotence and safe failure fallback.
Focused 172/0/0 and strict lint passed; real installed SDK/CLI/TUI/Web
(130.88s), including actual cancelled, and Rust SDK (14.36s) passed.
Full isolated assembled gate 2,562/0/109 (318 suites, 377.3s), release
TUI/Web, installer and scripts passed. OA-4B accepted. Temporary store=false polling, no
continuation; background SSE/replay, conversations and input token
counting remain OA-4. No new physical GPU/performance claim.

## 27 Nov — OA-4A foreground response resources

Memory-only charged listener records, retrieval/deletion/input pages,
previous-response text replay excluding old instructions, stateless
default retained. Focused 170/0/0 and strict lint passed; real installed
SDK/CLI/TUI/Web (134.83s) and Rust SDK (15.18s) passed. Assembled
gate 2,560/0/109 across 318 Rust suites (690.5s), release TUI/Web,
installer and scripts passed. OA-4A accepted. Background/cancel,
conversations and input token counting remain OA-4. No new model or
physical GPU/performance claim.

## 27 Nov — OA-3 stateless Models and Responses text

Actual registry model retrieval and stateless Responses JSON/SSE over
original typed requests/events, bounded projection ownership, correlation
and exact restored-prefix usage. Focused 167/0/0 and strict lint passed;
installed OpenAI Python SDK 2.33.0 strict real cached-model canary passed
Models, Responses JSON/SSE, exact Chat parity, refusal and IDs. Real
native/browser/official SDK (80.88s) and Rust SDK (15.02s) passed.
Full isolated gate 2,557/0/109 (318 suites, 617.5s), release TUI/Web,
installer and scripts passed. OA-3 accepted; OA-4 next.
No storage/continuation, new model capability or physical GPU evidence.

## 27 Nov — OA-2 semantic text request/model template

Typed raw/user/conversation intent through original queue and worker,
model-owned Qwen template measurement/validation/tokenization; exact
previous tokenizer output preserved in cached approved-tokenizer test.
Focused engine/serving 163/0/0, strict lint, tokenizer suite 7/0/0
and approved-source 1/0/0 passed. Actual cached CLI/TUI/Web and SDK
passed (67.77s and 15.85s); full isolated gate 2,553/0/109 across 318
Rust suites (878.6s), release TUI/Web, installer and scripts passed.
No new physical GPU/performance evidence. OA-2 accepted; OA-3 next.

## 27 Nov — OA-1B multipart foundation

Pinned checksum-verified Multer 3.1.0, strict complete form termination
and duplicate part header policy. Shared bounded Reader/charged part
backing and metadata, actual private TCP validation and rollback; public
media remains unsupported. Focused 38/0/0, strict lint and architecture
passed; real cached CLI/TUI/Web (76.57s) and SDK (14.07s) passed.
Full isolated gate exited 0 (902.1s): 2,551/0/109 across 318 Rust
suites plus release TUI/Web, installer and scripts. OA-1 accepted;
OA-2 next. No new physical GPU/performance claim. Exact Nix source-integrity remains
separately blocked by unavailable locked inputs.

## 26 Nov — OA-1B public chunked text / strict framing

Cached Hyper 1.13.0 server-parser patch resolves mixed/repeated length
normalization; bounded public chunked JSON now reuses original auth,
charged staging, SSE and model scheduler. Actual public socket tests
and strict lint pass (focused 33/0/0); real cached CLI/TUI/Web and SDK
regressions pass (60.52s and 9.11s). Assembled GPU boundary tests
26/0/2, strict default/CPU lint and dependency isolation passed; initial
gate timed out during CPU workspace test compilation. Full isolated retry
on 08a9d30e exited 0: 2,546/0/109 across 318 Rust suites, release TUI/Web,
installer and scripts. Chunked text accepted. Nix vendor source derivation
remains blocked by unavailable exact locked flake inputs; syntax passed. No new physical GPU or performance evidence.
Multipart remains blocked on approved decoder acquisition (cached
form-data 0.6.0 not suitable). OA-2 follows complete OA-1.

## 26 Nov — OA-1B bounded ingress and owned WebSocket primitives

Shared charged decoded-body staging and owned WebSocket continuation
implemented; isolated focused serving 31/0/0 and strict lint passed.
Unknown-length and internal adapter transport qualified on actual TCP;
ordinary public text framing and unsupported Realtime refusal preserved.
No new physical GPU execution or performance evidence. Actual cached CPU
CLI/native/Firefox 1/0/0 (60.72s), SDK 1/0/0 (10.33s); full isolated
gate 2,544/0/109 (318 Rust suites, 466.3s), release TUI/Web, installer
and scripts passed. Remaining OA-1B strict
mixed-framing resolution and approved multipart decoder acquisition;
OA-2 follows complete OA-1.

## 26 Nov — OA-1A bounded async text transport

Tokio/Hyper persistent HTTP/1.1 and bounded connection tasks now use the
original charged event/admission owner. Focused serving 25/0/0, engine
110/0/0, strict lint; actual cached CPU CLI/native/Firefox lifecycle 1/0/0
(55.52s) with persistent model listing and exact alias completions, SDK
listener/owner lifecycle 1/0/0 (8.83s). Assembled private-network gate
passed (819.3s), 2,538/0/109 across 318 suites plus release TUI/Web,
installer and scripts.
No new numerical, GPU execution or performance comparison.
Remaining OA-1B: streaming uploads, multipart and WebSocket foundation,
then OA-2. Logs: target/oa1-final-focused.log,
target/oa1-real-consumers.log, target/oa1-assembled.log.

## 26 Nov — model-local execution selector accepted

Native/Web cached Models share a typed backend/device override, inheriting the
profile by default and preserving original fit/refusal authority. Explicit
backend without device selects automatic devices; exact physical choice pins
one device. Real cached CPU native/Firefox/HTTP lifecycle 1/0/0 (57.08s), focused
Rust 67/0/3, JS 5/0, strict lint; assembled offline private-network gate exited
0 (301.6s), 2,543/0/109 across 318 Rust suites plus release TUI/Web/installer and
scripts. No new GPU execution or performance qualification. OA-1 next.

## 26 Nov — UX-P4B accepted

Final offline private-network workspace gate on `ad7a4178e3b85ea9e8a0ddcf9f2be798b0416ca8`
exited 0 in 592.3s: 2,539 passed, zero failures, 109 explicit skips across 318 Rust suites.
Full-default and explicit-feature strict lint, architecture/source-only GPU
boundaries, portable CPU release, native PTY (including 20x7), Web control/
settings, installer and scripts pass. Prior final real CPU/native/Firefox/
HTTP, SDK listener and transfer qualification, full native link/launch and
physical CUDA 24-layer/counter evidence remain applicable. No new numerical
comparison, HIP/Vulkan execution or overhead benchmark. UX-P accepted;
public serving OA-1 next. Log: target/ux-p4b-assembled-ad7a4178.log.

## 26 Nov — UX-P-M2 dashboard refinement (prior checkpoint)

Reference-aligned adaptive metrics, recent chart, complete segmented
layer placement and actual admitted-reader activity. Native 20/0/0,
JS 5/0, strict combined CPU lint, actual cached native/Firefox/HTTP
1/0/0 (53.65s) with original admission, exact output and 24 reported CPU
layers. Existing two cold_server typed-refusal assertions repaired:
standalone serving 4/0/0. No additional hot-path observation or
performance comparison. Assembled retry passed GPU feature-boundary tests
(26/0/2) and full-default strict lint, then found an acquisition-disabled
TUI import. Import scope corrected; exact CPU workspace strict all-target
Clippy passes. CPU workspace test-stage retry timed out during compilation
(90s budget, no suites executed). Complete P4B gate pending.

## 26 Nov — UX-P4B acceptance progress

Full native default linked/launch and TUI/Web smoke pass. Original
approved cached CPU native/Firefox/HTTP 6/0/0 (74.24s), actual cached
SDK listener metadata/redaction 1/0/0 (11.49s), native/shared Firefox
transfer durable/cancel fixtures and JS 4/0 pass. Original CUDA SDK
admission refusal/retry 1/0/0 (20.85s); exact 24-layer compute placement
on admitted PCI 0000:01:00.0, one 3-token completion, one first-token
sample and two decode intervals. Observed TTFT 2.395s, 2/1.5588s decode;
short qualification only, not a performance baseline or comparison.
No new numerical/ownership/admission policy. L5 GPU idle restored.
HIP/Vulkan unavailable in full-default L5 launch environment; HIP map
and profiling overhead not requalified. Full production strict all-target
Clippy and preceding GPU source-only compile checks pass. Full assembled
isolated gate pending. M2 optional; preserve existing public serving order.

Assembled retry: GPU feature-boundary and CPU strict lint pass; 1191 passed, 2 failed, 56 ignored before cold_server stopped execution (exit 101). See CURRENT_HANDOFF for exact typed refusal assertions. P4B remains open.

## 26 Nov — UX-P-M1 live model performance

Registration-scoped TTFT, weighted decode intervals, exact output and
completed receipt counts; bounded client trend and selected read-only
layer/lease details. Engine 109/0/0, combined CPU CLI 64/0/3, Web 4/0,
strict focused lint and full default cargo check pass. Actual approved
cached CPU/browser/native/HTTP 1/0/0 (55.61s), exact shared counters and
reset, 24 CPU layers, original admission/output and owner checks retained.
GPU placement compile-only, no new performance or overhead comparison
claim. Optional M2 and full P4B gate pending.

## 26 Nov — full native CLI defaults and typed error diagnostics

Default Cargo CLI production all; explicit embedded/portable CPU profiles
unchanged. Typed failure chain and resource/I/O detail preserved; HTTP/SSE
safe public projection, explicit authenticated loopback diagnostic opt-in,
safe structured operational log metadata. Focused core/engine/serving,
SDK and CLI/config/pipeline suites passed; full default cargo check passed.
Linked default cargo run timed out during final CLI compilation; launch
not yet verified.
No new performance or physical GPU/model qualification claim. Full GPU
Clippy pre-existing debt and assembled P4B gate remain pending.

## 26 Nov — UX-P4A2 final ergonomics

Stable identity/byte scroll anchors, distinct response/detail viewport,
safe operational log details, compact scrollable dialogs/palette and
editor drag edges. Strict lint, 18 focused TUI tests, optimized 91/0/3,
actual rendered cells and real compact 20x7/native regressions, native
and simultaneous Firefox shared transfer durable/cancelled pass. Final
response heading fix (registration first, alias truncated), F9 reveal;
strict combined lint/18 focused tests rerun. Actual native/attached-Web
cached CPU/HTTP 1/0/0 (34.49s), original 8 GiB admission/exact output;
browser portion passed separately. No new dependency or runtime
authority/performance/GPU claim. Final
isolated workspace gate/assembled owner acceptance P4B next.

## 26 Nov — UX-P4A1 follow-up toolbar-free contextual UI

User-directed removal of permanent toolbar; selected-row contextual
menus, exact right-click release, inline live local search and separate
session menu. Strict feature lint, 15 focused native Rust tests,
optimized 88/0/3, actual rendered cells and native PTY context/search/
settings/mouse/theme/transfer regressions. Original execution/identity/
configuration ownership retained; no new dependency or GPU claim.
Actual approved cached CPU/browser/native/HTTP 1/0/0 (57.77s, five
filtered) with original admission/exact output; previous memory blocker
resolved. Shared native/Firefox transfer durable and cancelled pass.

## 26 Nov — UX-P4A1 screenshot-driven minimal presentation

Curated quick actions + More, one navigation rail, concise summaries
and explicit exact Details, smart IEC sizes without changing bytes or
machine-readable authority. Actual rendered cell inspection, focused
Rust 87/0/3, strict lint, native mouse/action/editor/settings/theme and
controlled transfer pass. No new dependency, model download, GPU or
performance claim. Real cached owner final rerun pending: host refused
original 8 GiB envelope (7188 MiB after headroom); browser load timeout
last combined rerun after earlier browser resident passes. Remaining
ergonomics P4A2, full gate P4B.

## 26 Nov — native UX-P polish (P1/P2/P3A/P3B)

P3B typed finite settings/log choices, local shelf search, grapheme
caret/selection and independent editor viewport. Focused Rust 86/0/3,
strict lint, actual native editor/choices and stale draft refusal,
original control/appearance/settings/mouse PTY and approved cached
CPU/browser/native/HTTP 1/0/0 (55.96s, five filtered) pass. Original
admitted scheduler/registration/generation and exact output unchanged.
Only existing locked Unicode dependency edges exposed to optional TUI;
no version changes/download, physical GPU or performance claim. P4A
remaining ergonomics and visual review, P4B full isolated acceptance.

### Earlier UX-P1/P2/P3A results

Presentation-only theme, scoped mouse, resizable panes/scroll and bound
action buttons/More/explicit modal controls. UX-P1 and P2 focused
acceptance, P3A optimized Rust 82/0/3, strict lint and actual native
PTY shell/config/theme/mouse regressions; controlled transfer mouse
pause/resume/durable/cancel and approved cached CPU/browser/native/HTTP
1/0/0 (52.81s, five filtered) pass on original owners. No measured
performance comparison or new GPU qualification. UX-P3B accepted above; overall
UX-P exit requires full isolated gate at P4 before public serving.

## 26 Nov — CLI/TUI/Web acceptance (UX-10)

Original CUDA/HIP/RTC/Vulkan loader observations and existing-only
nonblocking cooperative quota charge/reservations; no client, model,
cache bootstrap or second accounting owner. Focused 44/0/2 and strict
minimal/cache/combined lint pass. Complete isolated gate sequence passed 2,505/0/109 (318 Rust suites), release Web/TUI/installer/scripts. Original run stopped at two transient busy-observation fixture assertions in acquisition/cache transactions; test-only bounded busy retry, reran affected cache-only/diagnostic/acquisition suites and exact remaining gate commands. No production change after the initial run.
UX-1–10 accepted; additional UX-P polish precedes public serving. No physical GPU execution,
live model acquisition or performance qualification inferred.

## 26 Nov — direct CLI polish (UX-10 core)

Cold doctor, original planner device/model dry-run, typed pipeline
list/inspect/check, fresh explicit positional source precedence, help/
compatibility and CLI guide. Strict minimal/combined Clippy; final isolated
minimal CLI/pipeline 27/0/0, optimized combined CLI/pipeline/run 32/0/1.
Existing populated cache unchanged by doctor. General pipeline execution
refuses; unobserved runtime bindings and physical quota usage are not
claimed. Isolated formatting/architecture and HIP/Vulkan compile checks
pass; initial full gate stopped during CUDA native transport rebuild.
Full UX-10 acceptance completed above. No model download, new
dependency, physical GPU or performance claim.

## 26 Nov — simultaneous native/Web acceptance (UX-9E)

Combined release CLI 50/0/3; real native PTY/attached Firefox same-operation
safe pause -> resume/durable and cancellation/partial retention; original
child exact range reopen, terminal observations and joined host retirement.
Approved cached Qwen 0.8B GGUF optimized real browser/TUI/Web/HTTP
integration 1/0/0 (54.16s, five filtered): native active context edit,
stale Web draft refusal, Web restore, unchanged admitted scheduler/
registration/listener and exact HTTP output. Idle native log revision
redraw passed. Complete isolated workspace gate 2,491/0/109
(318 Rust suites), release Web/TUI/installer/scripts passed. UX-9D/E
accepted; UX-10 next. No model download, new dependency, performance or
physical GPU claim.

## 26 Nov — Web profile, configuration and logs (UX-9D)

Shared original ConfigFile/Selection/action slot, safe typed and normalized
YAML editing, protected profiles and one bounded operational ring; exact
stale refusal and cross-client invalidation. Strict feature/lint/dependency
isolation, JS 3/0, actual HTTP/native PTY attached-Web edit and Firefox
settings/log smoke pass. Initial gate interrupted intentionally during
Vulkan feature-boundary rebuild; complete UX-9E rerun above passed and
UX-9D accepted.
No performance, admission, model download or physical GPU claim.

## 26 Nov — Web model, download and device management (UX-9C)

Original cache and device observations shared by both frontends;
authenticated cold inventory, exact digest and physical device actions,
explicit pinned files/quant/approved recipe consent, exact original
transfer controls, shared status-at-emission invalidation. Strict
feature/lint and terminal dependency isolation, JS 3/0, cold HTTP/
Firefox, controlled HTTP transfer and native transfer PTY, real Firefox
pause/range-resume/publication pass. Original cache and admission
authority unchanged; no new dependency, live remote acquisition, GPU
qualification or performance claim. UX-9C optimized cached integration passed 1/0/0 (48.13s, five filtered); full isolated gate 2,489/0/109 (318 Rust suites), release Web/TUI/installer/scripts passed. UX-9C accepted.

## 26 Nov — Web resident model and listener actions (UX-9B)

One shared native/Web host action slot, original resident planner and
OutputStream, exact action/registration/generation IDs, authenticated
bounded typed POST and Web-only latest receipt, explicit cancellation
and confirmations, same-process selection and retirement. Browser Home/
Running/Serve actions, actual cold HTTP and Firefox typed refusal,
strict TUI/Web/Web-only CPU/combined lint, JS 3/0 and focused tests pass.
Other five combined release tests passed; final optimized isolated
cached browser/TUI/Web/HTTP regression 1/0/0 (46.14s, five filtered)
passed after PTY assertion corrections and host-memory recovery,
without reducing the 8 GiB envelope. Full isolated gate 2,489/0/109,
release Web/TUI/installer/scripts passed. UX-9B accepted.
No new dependencies, model downloads, physical GPU or measured
performance change.

## 26 Nov — embedded SolidJS Web shell (UX-9A)

Committed offline-built Vite assets, one shared foreground Runtime and
active Selection, loopback numeric Host/origin/bearer boundary separate
from inference, bounded read-only snapshot/SSE transport. Actual HTTP/
Firefox responsive smoke and strict minimal/combined lint pass, JS 2/0.
Serial optimized combined run_unified 6/0/0 (32.71s), actual cached Qwen
TUI-attached Web same registration/generation/selection/stop joins.
Final asset/license refinement passes HTTP/Firefox; complete isolated
rerun 2,485/0/109 and release Web/TUI/installer/scripts pass after TCP
status-line fixture correction. UX-9A accepted.
No new inference or acquisition grant, physical GPU run or performance
change. Cargo reuses two existing packages, no newly resolved versions.

## 26 Nov — native configuration/profile/log management (UX-8D)

Shared ConfigFile optimistic atomic owner, observed typed setting editor
and bounded raw YAML, explicit process-local profile activation, protected
and confirmed removal, coalesced cold reads, bounded operational action/
owner event log and filters. Focused reducer/ring 5/0, strict minimal and
combined lint, actual settings PTY and serial cached release lifecycle/
profile-switch pass (22.95s); other five release tests passed. Full isolated
gate 2,483/0/109, release PTY/installer/scripts pass. UX-8D accepted.
No new dependencies, model admission, remote acquisition, physical GPU
qualification or measured performance change.

## 26 Nov — native model / download / device management (UX-8C)

Exact model/digest/physical-device selections and shared resident/cache
actions, static catalog and partial checkpoint observations, coalesced
read worker, same-runtime original controlled transfer with explicit
pinned intent, original pause/resume/cancel and durability. TUI reducers
4/0, native intent 1/0, actual controlled-transfer PTY, strict minimal and
combined lint pass. Actual cached Qwen lifecycle and remaining release
tests pass; offline shell PTY passes. Final full isolated gate passes
(2,481 passed, zero failed, 109 ignored; release PTY/installer/scripts).
UX-8C accepted. No remote download, new physical GPU qualification or performance change.

## 26 Nov — native model and listener actions (UX-8B)

Shared canonical resident planner for CLI serve and native UI; original
source/device/budget/control ownership. Real load/run, exact registered
invoke/unload, listener generation start/stop, scoped action worker,
bounded output copy and cooperative cancellation/foreground retirement.
First serial optimized combined run_unified 6/0/0 (24.59s), actual cached
Qwen 0.8B GGUF TUI/HTTP lifecycle at unchanged 8 GiB envelope, strict lint
and CUDA/HIP combined compile pass. Final combined release other 5 tests
pass; serial real cached PTY/HTTP rerun passes (35.16s), with repeated
invoke/paste and original 60s HTTP read bound after shorter timeout under
host contention. Final full isolated gate passes (2,479 passed, zero
failed, 109 ignored; release PTY, installer and scripts pass). UX-8B
accepted. No physical GPU run, remote
acquisition or measured performance change.

## 26 Nov — native terminal shell (UX-8A)

Read-only Ratatui/Crossterm shell uses original Runtime/ControlPlane and
cold config/cache/device owners. Default capable TTY launch, nine views,
responsive rail/overlay, palette and scoped terminal/signal retirement.
Actual Linux PTY and fixed-cell navigation/render tests pass; terminal-only
and combined lint pass. Full isolated gate 2477/0/109, release PTY,
installer/scripts pass. No physical execution, source grant or measured
performance change. TUI dependencies stay in CLI; cross builds not run.
~~~~

## Appendix E — complete prior main body and historical chain

The prior comprehensive pack at 3df319127b19616dd3e9d4643d63c7ef9cd1a448 is retained unchanged. Its Appendices A–D preserve all earlier exact decisions and landings through M1; its Appendix E links the previous full UX-9B main body and earlier foundation history. Earlier prompts and next steps below are historical, not current work authorization. Read the full linked originals when resolving earlier design details. No material is discarded or replaced by this summary.

- [.agents/docs/FULL_COMPREHENSIVE_HANDOFF_2026-11-14.md](FULL_COMPREHENSIVE_HANDOFF_2026-11-14.md)
- [.agents/docs/FULL_COMPREHENSIVE_HANDOFF_2026-11-15.md](FULL_COMPREHENSIVE_HANDOFF_2026-11-15.md)
- [.agents/docs/FULL_COMPREHENSIVE_HANDOFF_2026-11-18.md](FULL_COMPREHENSIVE_HANDOFF_2026-11-18.md)
- [.agents/docs/FULL_COMPREHENSIVE_HANDOFF_2026-11-20.md](FULL_COMPREHENSIVE_HANDOFF_2026-11-20.md)
- [.agents/docs/FULL_COMPREHENSIVE_HANDOFF_2026-11-21.md](FULL_COMPREHENSIVE_HANDOFF_2026-11-21.md)
- [.agents/docs/FULL_COMPREHENSIVE_HANDOFF_2026-11-22.md](FULL_COMPREHENSIVE_HANDOFF_2026-11-22.md)
- [.agents/docs/FULL_COMPREHENSIVE_HANDOFF_2026-11-23.md](FULL_COMPREHENSIVE_HANDOFF_2026-11-23.md)
- [.agents/docs/FULL_COMPREHENSIVE_HANDOFF_2026-11-24.md](FULL_COMPREHENSIVE_HANDOFF_2026-11-24.md)
- [.agents/docs/FULL_COMPREHENSIVE_HANDOFF_2026-11-24_FOUNDATIONS.md](FULL_COMPREHENSIVE_HANDOFF_2026-11-24_FOUNDATIONS.md)
- [.agents/docs/FULL_COMPREHENSIVE_HANDOFF_2026-11-25_FOUNDATIONS.md](FULL_COMPREHENSIVE_HANDOFF_2026-11-25_FOUNDATIONS.md)
- [.agents/docs/FULL_COMPREHENSIVE_HANDOFF_2026-11-25_PHASE1_EXIT.md](FULL_COMPREHENSIVE_HANDOFF_2026-11-25_PHASE1_EXIT.md)
- [.agents/docs/FULL_COMPREHENSIVE_HANDOFF_2026-11-26_UX9B_EXIT.md](FULL_COMPREHENSIVE_HANDOFF_2026-11-26_UX9B_EXIT.md)
- [.agents/docs/FULL_COMPREHENSIVE_HANDOFF_2026-11-26_UX_PM1.md](FULL_COMPREHENSIVE_HANDOFF_2026-11-26_UX_PM1.md)
- [.agents/docs/NEXT_AGENT_2026-11-26_UX9C.md](NEXT_AGENT_2026-11-26_UX9C.md)
- [.agents/docs/NEXT_AGENT_2026-11-26_UX_P4B.md](NEXT_AGENT_2026-11-26_UX_P4B.md)

### E.1 — verbatim UX-P-M1 main body (historical)

# ZeroGravity — full comprehensive handoff through UX-P-M1

Date: 26 Nov 2026 (Singapore). Repository: `/home/pvl/spaces/abird/src/zerogravity`.
Implementation checkpoint: `e96cdab98d47c5c84001c1afaaebd8f126b9e0b3` — live registration performance and layer execution details.
Accepted CLI/TUI/Web UX-1–10 exit: `1b8de1cb87fa4815655cbb6700e6633946f93305`.
Accepted bounded Phase 1 exit: `d5030ad4ce476b913a46056d492992ccf1b8edf7`.
Previous comprehensive takeover: `3333ec5cd462e4520ffc1901a3881785478c86ca`,
[26 Nov UX-9B pack](FULL_COMPREHENSIVE_HANDOFF_2026-11-26_UX9B_EXIT.md).

**Read §§0–13 first. Resume at §11, not the historical continuation prompts.**
This is the project owner's requested one-time comprehensive takeover. It is
not a second live task ledger or a new evidence-pinning requirement. Live
source, executed results, AGENTS.md, the ordered full plan and owning TODOs
take precedence. All original docs, plans, previous handoffs, research and
history remain intact. No implementation is changed by this handoff.

- Appendix A: exhaustive tracked engineering document index.
- Appendix B: literal unresolved checklist snapshot from the implementation
  checkpoint; original full owning files retain all prose and task context.
- Appendix C: complete reachable Git ledger from the initial design through
  M1, plus the exact post-UX-9B delta.
- Appendix D: verbatim landing narratives for every post-UX-9B checkpoint,
  original design-only UX-P specification, changed-path navigation and the
  reconciled current handoff/baseline excerpts.
- Appendix E: previous complete UX-9B main body verbatim (historical);
  direct navigation to its original appendices and the earlier foundation
  packs. This preserves original design, chronology, failed experiments,
  reusable primitives, actual numerical/physical evidence and exact
  procedures without recopying the multi-megabyte historical appendices.

## 0. Recovered implementation and current position

The interrupted live performance turn had already finished and signed M1.
It was recovered by inspecting the actual clean tree, commit, job statuses,
owning specification and final test logs. Do not repeat M1 or revert it.
UX-1–10 are accepted. UX-P1, P2, P3A/B, P4A1 including the toolbar-free
correction, P4A2, full CLI defaults/error diagnostics and UX-P-M1 are
implemented and focused-qualified. **Overall UX-P is not yet accepted.**
UX-P4B is the required next exit. M2 deeper measurements are optional,
bounded by actual owner capabilities; they are not a prerequisite to
invent unsupported profiler/topology features.

Latest complete assembled isolated gate: UX-10, 2,505 Rust test executions /
0 failures / 109 explicit skips, 318 suites, release Web/TUI, installer and
scripts passed. The sequence was recovered after two transient cache busy
fixture assertions by test-only bounded retry and exact remaining checks;
no production change after the initial run. That is accepted evidence for
that source, **not a final gate for the subsequent UX-P/default/error/M1
changes**. Prior UX-9C 2,489/0/109; UX-9D/E 2,491/0/109.

Latest real changed-consumer M1 run: approved cached Qwen 0.8B GGUF CPU /
Firefox / native / HTTP lifecycle, 1 pass, 0 failures, 55.61 s, five
filtered. Shared registration counters across native/HTTP/Web, reset on
reload, 24 CPU layer spans, exact output and owner/selection/stale checks;
original 8 GiB host admission. Engine 109/0/0, combined CPU CLI 64/0/3,
Web 4/0, strict focused lint, production default cargo check. Three
existing CLI skips. This is not a throughput benchmark or new GPU
qualification. Final full-default executable link/launch still unverified.

### 0.1 Repository and resource authority

- L5/pvl-l5 sole source authoring connector, `dotlink---l5`. Before this
  documentation-only commit: main at e96cdab98d47c5c84001c1afaaebd8f126b9e0b3,
  local origin/main at a6744b125ea415ae517a3273f00a5dcc5ffd0e37, ahead 14 /
  behind 0, clean working tree/index, one main worktree. Do not infer the
  remote server's current state from its local tracking ref. No fetch/push.
- No running or queued jobs on the inspected L5 bridge. Revalidate at
  takeover, preserve jobs of other clients and actual shared GPU/host owners.
- Last observed L5 NVIDIA GeForce RTX 3060 Laptop GPU, 6,144 MiB total,
  37 MiB used, 5,789 MiB reported free, no listed compute apps; available
  host memory about 20.9 GiB, free disk about 45 GiB, 96% used. Earlier
  observations in this session varied. These are momentary readings,
  not admission guarantees. Do not purge unrelated build/cache state.
- X2 and A1 are execution/testing hosts only. No source authoring there.
  Not revalidated for this handoff; last A1 observation from the earlier
  investigation was an older accessible checkout (971b58e7) and RTX 4060
  Laptop SM89/AMD Strix iGPU. Do not assume HEAD, GPU availability, runtime
  or physical capability on either. Sync exact authorized L5 commits
  through verified incremental Git bundles and clean fast-forward only.
- Main is the only persistent local branch/worktree. No destructive Git,
  privilege changes, GPU reset, unapproved download, unrelated process
  cleanup or push. The implementation commit signature is cryptographically
  valid; signer-principal verification unavailable because local
  allowed_signers is absent. Never disable signing.

## 1. Bootstrap sequence

1. [AGENTS.md](../../AGENTS.md).
2. [Planning rulebook](../plans/README.md).
3. This handoff §§0–13 first. Appendix D contains the exact newest landing
   narratives; Appendix E links the full pre-UX-9C and Phase 1 history.
4. [Ordered full implementation plan](../plans/FULL_IMPLEMENTATION_PLAN.md).
5. [CURRENT_HANDOFF](CURRENT_HANDOFF.md), [CURRENT_BASELINE](CURRENT_BASELINE.md).
6. [Plan queue](../plans/TODO.md) and relevant owning TODOs. Earlier next
   steps within historical narratives are not active instructions.
7. [Canonical CLI/UX specification](../plans/cli-ux/NEXT_ITERATION_CLI_UX_AND_MODEL_LIFECYCLE.md)
   §§33–46, §46A, §51; [UX-P polish](../plans/cli-ux/TUI_POLISH_AND_MOUSE_INTERACTION.md),
   especially the 26 Nov user-directed toolbar-free correction;
   [Live model performance](../plans/cli-ux/LIVE_MODEL_PERFORMANCE.md).
8. [CLI/device routing](../plans/cli-device-routing/README.md) and
   [its TODO](../plans/cli-device-routing/TODO.md),
   [acquisition](../plans/model-acquisition/README.md) and
   [its TODO](../plans/model-acquisition/TODO.md),
   [serving TODO](../plans/serving-and-speculation/TODO.md),
   [public serving design](../plans/serving-and-speculation/NEXT_ITERATION_OPENAI_COMPATIBLE_SERVING.md),
   [SDK/serving](../plans/fabric-vnext/PUBLIC_LIB_AND_SERVING.md),
   [platform](../plans/platform-and-release/TODO.md),
   [structure and naming](../plans/pipeline-composition/STRUCTURE_AND_NAMING.md).
9. Inspect actual source and tests (§5), then §11. For subsequent backend
   work read the owning backend/numerics/model execution specifications and
   actual rejected trials (§§8–9,12), not just summary numbers.

The standalone [starting prompt](NEXT_AGENT_2026-11-26_UX_P4B.md) is also
reproduced in §13. Do not pass older prompts as current instructions.

## 2. Owner direction and working contract

- Elegant, simple, clear and intuitive. ZeroGravity is a composable Rust
  library/execution fabric, not a single-model benchmark or per-backend
  independent engine. Favor zero-cost abstractions, direct hot paths and
  performance where it remains clean and maintainable. Where high
  performance requires convoluted maintenance, choose clean/simple first
  and target the measured hot area selectively.
- Reuse mature optimized mathematics and actual shared operations. Attack
  large architectural/computational bottlenecks before marginal launch
  counts. Model-specific specializations must consume the authoritative
  model semantics and justify their own benefit.
- Preserve useful correct model-neutral kernels, layouts, transports,
  state/scheduling transforms, providers and completion primitives when
  a consumer/selector loses. Remove temporary probes, losing consumer
  wiring, runners and test-only choices, not reusable implementations.
  Delete a primitive only when incorrect/unsafe, incompatible with its
  contract or strictly superseded; record why.
- Preserve exact source, numerical, layout, state, physical resource,
  ownership, cancellation and completion semantics. Intentional changed
  arithmetic is a separate explicit numerical profile, not a hidden
  approximation. No retry against an uncertain submitted root.
- No overly granular test/evidence/pinning process. Batch related work,
  reuse applicable accepted evidence, validate actual changed consumers
  and meaningful invariants. Full assembled gate at phase exit; synthetic
  kernel speed, compilation and UI lifecycle time are not performance
  parity. Real-source and physical qualification still mandatory where
  the relevant execution path changes.
- One substantial coherent validated signed checkpoint per user turn,
  aim approximately 30 minutes at most; pause with exact state, blockers
  and next unit. Recover interrupted work from live source and executed
  results. Keep active jobs owned; no indefinite background promise.
- Sole L5 source authority, exact incremental bundles, no persistent
  additional worktrees. Coordinator is single writer of shared queue.
  User-authored root README unchanged; architecture in README describes
  final intended engineering system while status belongs in plans.
- Latest product direction: calm, beautiful and minimal by default,
  discoverable detail on demand, not maximal control visibility. Full
  mouse plus keyboard; **no permanent action toolbar** at any width.
  One navigation rail, inline live local search, exact selected-object
  context actions, separate small session menu. Smart human size units.
  Dark/Light/Auto now; additional theme catalog later only if requested.
  Running is the live model dashboard, not a separate monitoring app.
- Full sane Cargo CLI default includes all current production models,
  CPU/CUDA/ROCm/Vulkan, acquisition, serving, TUI/Web while breadth is
  small. Explicit minimal/portable and embedding profiles remain
  available. Do not confuse a compiled backend with qualified execution.
  Error categories and chains must be accurate; safe HTTP responses and
  opt-in trusted diagnostic details, with bounded safe structured logs.

## 3. Original design and architecture to preserve

The [October 10 master plan](../plans/ZEROGRAVITY_MASTER_PLAN_2026-10-10.md),
[vision](VISION_SCOPE_AND_PRINCIPLES.md), [modular reference architecture](MODULAR_AI_SYSTEMS_REFERENCE_ARCHITECTURE.md),
original engineering specifications and critical review define the design
from the beginning. First reachable Git commit is October 13
be95424837deb7919adf34503db71d27d3ad3adb. Appendix E preserves the detailed
previous architecture explanation verbatim; earlier Phase 1 packs preserve
the original design evolution and decisions. This summary does not replace
their source/numerical/lifetime contracts.

### 3.1 One fabric; four distinct concerns

![Execution architecture](../../docs/architecture-overview.svg)

| Concern | Authority |
| --- | --- |
| Fabric/runtime | Model lifecycle, original catalog/registration, admission, placement, scheduling, batching, pipelines, persistent state, publication and serving |
| Model semantics/planning | Verified source-bound computation and task, learned roles/axes, typed shapes/regions/state effects, numerical/quality profile |
| Implementations | Native composed ops, mature vendor math, specialized kernels, captured commands or compiled closed regions/whole programs |
| Backends | CPU, CUDA, HIP/ROCm, Vulkan and future devices; physical memory, command submission, readiness and completion |

Do not conflate operator libraries, graph compilers, artifact formats and
physical backends. Prefer actual mature math (cuBLAS/Lt, CUTLASS, cuDNN,
FlashAttention/FlashInfer where qualified; rocBLAS, hipBLASLt, CK, MIOpen
and suitable HIP primitives; qualified Vulkan cooperative-matrix/shader
implementations and existing CPU optimized/reference operations).
Optional concrete MIGraphX/TensorRT/IREE/ONNX Runtime integrations are
later Phase 6, not dependencies of native CUDA/ROCm/Vulkan. No new
provider is granted by recognizing a format or compiling a library.

### 3.2 Source, semantics and executable lowering

Original HF configuration/tokenizer/processor/indexed Safetensors and
GGUF metadata/codecs/weight tensors are primary. Transformers describes
an architecture/software ecosystem; Hugging Face is source/distribution,
not a storage format. Verify original identities/roles/axes/quantization;
no implicit conversion, remote model code, mandatory Python or mandatory
ONNX conversion. Source eligibility independent of device execution.

Two enduring execution representations: (1) authoritative semantic
ModelProgram/ForwardPlan, mathematical/task regions and state effects;
(2) prepared physical executable and original source/resource owners.
Artifact input and provider-local compiler graphs are not alternative
canonical IRs. Native ModelAssembly/regions and multi-model TaskPipeline
are distinct concerns; compositional descriptions do not create a second
runtime. Select a provider during cold preparation or qualified
committed-root switch. Pre-enqueue miss may take already-qualified
decomposition or refuse; submitted uncertainty cannot be retried
against the same root by silently changing provider.

Direct typed-region lowering is independent of external graph import.
MIGraphX/TensorRT preferably construct from verified regions through
native APIs. Optional ONNX import/ORT needs authenticated graph/weights,
layout, precision, state, effects and resource contracts; IREE suitable
lowered computation. SDK-free scaffold accepted only for bounded seam.

### 3.3 Non-negotiable state and physical ownership

One logical StateRoot publisher per session. Device work creates private
candidates and readiness, not logical commits. All required KV/recurrent/
conv state domains publish a compatible frontier before visible output.
Cancellation/failure cannot leak provisional output/state; uncertain
completion fences/quarantines the original physical owners until actually
safe. Immutable source/backing/range/lease provenance remains authority;
aliases/readers retain real owners; independent mutable session state is
private. Complete-domain compatibility for prefix/cache/COW/page reuse.

One physical ResourceArbiter. Account persistent/context/workspace/
transfer and synchronization costs; explicit headroom, real UMA overlap.
No implicit reclamation or hidden memory grant. Backend adapters own
submission and completion; shared/external views conditional on actual
capability. No new allocator, model registry, scheduler or execution IR.
Separate reference arithmetic/native profile, quality, source compatibility,
physical readiness, feature compilation and serving/performance acceptance.
One fair worker/cohort/scheduler with shared prefill, independent
continuations, paged/hybrid state, feedback, MTP, placement and task
pipeline contracts.

Read [invariants](INVARIANTS_AND_FAILURE_SEMANTICS.md),
[state](STATE_CACHE_AND_TRANSACTIONS.md), [resource/commands](ASYNC_RESOURCE_COMMAND_IR.md),
[execution spine](MINIMAL_EXECUTION_SPINE_AND_PROTOCOL_V4.md),
[regions](../plans/pipeline-composition/REGIONS_AND_PROFILES.md),
[correctness policy](../plans/numerics-and-context/CORRECTNESS_POLICY.md).
The previous comprehensive handoffs preserve all accepted foundation
slices and exact provider/capture/persistence decisions.

### 3.4 Current typed failure contract (supersedes historical coarse translation)

| Core category | Meaning |
| --- | --- |
| Malformed | Actual source/artifact validation failure |
| InvalidInput | Invalid request/configuration/input rather than artifact corruption |
| Unsupported | Backend, device, route or capability unavailable |
| Limit | Configured/static bounds, arithmetic overflow, quotas |
| Resource(ResourceLimit) | Actual measured admission refusal; operation/resource/requested and available bytes/scope |
| State | Invalid lifecycle/internal state |
| Io | Original underlying cause |

In-process ServiceError keeps typed origin through Arc on error path,
including shared conversions, resource amounts and I/O causes. For
synthetic service errors use stable code, then status fallback; never
invent a budget or infer artifact corruption from HTTP 400. The previous
A1 message "malformed model artifact" for missing compiled backend was
misclassification. Original-source Qwen needs eligible CUDA or HIP,
not default CPU/Vulkan; SM89 is not automatically qualified by compiling
CUDA. Preserve actual source and device policy, not merely nicer errors.
No string parsing as admission/fallback policy.

HTTP and SSE preserve status/code/type with fixed safe public message
by default. Optional error.details adds status, code, category, measured
bytes and OS code where available, plus bounded diagnostic text/causes
only after explicit trusted listener setting and valid bearer auth.
Pre-auth/framing/connection refusal never exposes diagnostic text.
The existing bounded control event ring/logs carries safe FailureDetails
tagged by listener generation; arbitrary raw paths/credentials/prompts
not stored in the shared operational ring. Full native diagnostic retained.
See §5 and Appendix D.14. The native diagnostic setting is not browser
safe-editable. Future public serving must not expose trusted debug text
to ordinary clients.

## 4. Ordered delivery and pending boundaries

| Phase | Status / next objective |
| --- | --- |
| 1 Foundations | Bounded common exit accepted at d5030ad4; broader extensions open |
| 2 CLI/TUI/Web, public serving | UX-1–10 accepted; UX-P/M1 focused-qualified; P4B assembled acceptance next, M2 optional; then public serving and ordinary benchmarks |
| 3 ROCm | Match or beat optimized matched vLLM first, AMD |
| 4 CUDA | Matched vLLM next on NVIDIA; real lane/context/quality and hardware qualification |
| 5 Vulkan | Native feature/correctness/performance parity, suitable actual comparators |
| 6 Optional compiled providers | Concrete MIGraphX, TensorRT, IREE, ONNX Runtime with measured benefit, original authority |

This order supersedes early roadmap order. The full plan §§1–6 and its
crosswalk retain open details. Do not reopen bounded foundations merely
because broader seams remain open. Do not mark owner plan done for one
checked slice. Original source/model priorities: Qwen3.5 0.8B/2B and
Qwen3.8 27B for first performance family; Qwen3.5+/3.8 variants, Gemma4,
Qwen Flash/Next, DeepSeek and other qualified model families later through
shared contracts. Media tasks/pipelines including Qwen image/audio/TTS/
ASR, Whisper, Parakeet and additional models remain their owning scope.
Do not treat a desired catalog entry as an executable model.

### 4.1 Bounded accepted foundation versus remaining generalization

Phase 1 minimum common seams: source-bound semantic and prepared
composition, native operations/typed regions and optional provider scaffold,
authentic HF/GGUF/SDK source lifecycle, original physical/state/control
ownership, paged/streamed/prefix/persistence/cohort, typed tasks and
bounded pipelines, actual SDK and real CLI consumers. Accepted scope,
exact post-matrix 49-commit history and physical tests are preserved in
[Phase 1 exit pack](FULL_COMPREHENSIVE_HANDOFF_2026-11-25_PHASE1_EXIT.md)
§§0–13 and Appendices C–E. General foreign/Existing buffer or context
interop, device-write-to-host-read/coherency, multi-domain and UMA overlap
generalization, non-Linux physical fit/OS-hard reservations, general
multi-source/mixed-backend HF, general DAG/media/context, arbitrary state
migration/persistence, authenticated tenant stores, async SSD and broader
model/codec/profile/device qualifications remain open. Do not infer
general support from a bounded coherent mapped producer or actual
single-device preparation. Preserve all original hold conditions.

### 4.2 Phase 2 product state and later public serving

See §7 and Appendix D for each checkpoint. User explicitly inserted UX-P
before public serving. Current UI is toolbar-free with metrics; M2 is
optional on-demand depth. After required P4B exit, read the dormant
[OpenAI-compatible serving iteration](../plans/serving-and-speculation/NEXT_ITERATION_OPENAI_COMPATIBLE_SERVING.md).
Complete ordinary local API and normal CLI/SDK/server benchmarks:
qualified catalog/router/model/pipeline, Chat/Responses and supported
completion/embedding/tool/structured outputs, usage/errors, JSON/SSE,
streaming/cancellation/deadline/backpressure/concurrency. Refuse unsupported
requests rather than simulate success. Preserve separate control/inference
auth, explicit listener/route/tenant scope. Future WebSocket/WebRTC/media
through actual task contracts; not a clone of hosted billing/account APIs.
Use original serving/runtime/scheduler and state contracts. No premature
backend tuning, no public exposure implied by local loopback acceptance.

## 5. Current source map and implemented contracts

All paths below are repository-relative. Foundation source map in the
previous Phase 1 pack §5 and Appendix E §5. Inspect live file names and
feature guards rather than extrapolate old examples.

| Area | Primary live owners |
| --- | --- |
| Core / source / operators | crates/zerogravity-core, zerogravity-artifacts, zerogravity-cpu, zerogravity-model-qwen35, zerogravity-model-gemma4, zerogravity-gpu, vendored CubeCL |
| Engine | crates/zerogravity-engine/src/{runtime,diagnostics,router}.rs; router/{registry,worker,pending,event_channel,dispatch,error,performance}.rs and actual scheduler/state owners |
| Facade and control | crates/zerogravity/src/{runtime,models,configuration,control,run}.rs; control/{cache,downloads,events,serving}.rs; prepare/, devices/, builtins/text/ |
| Configuration | facade pipelines/{document,yaml}.rs, configuration/file.rs; CLI config_cli.rs, run_cli/config.rs; one typed v2 Document/ConfigFile |
| CLI | crates/zerogravity-cli/src/{main,args,commands,completion,output,model_cli,resident,run_cli,pull}.rs; output/presentation.rs, run_cli/, serve/ |
| Shared host | CLI control_host.rs, control_host/{actions,inventory,logs,settings,work}.rs |
| Native TUI | CLI tui.rs, tui/{view,layout,input,controls,editor,choice,theme,format,performance,library,logs,settings,terminal,work}.rs |
| Embedded control | CLI web.rs, web/{actions,http,settings,wire}.rs; web/src/{main.jsx,settings.jsx,performance.jsx,performance.js,api.js,browser-smoke.mjs,styles.css}, committed dist |
| Inference server | crates/zerogravity-serving/src/{server,http,auth,service}.rs; original text service and events (not browser control transport) |
| Actual tests | crates/zerogravity-cli/tests/{cli,config_cli,pipeline_config,run_unified}.rs, tests/fixtures/{tui_shell,web_shell,serve_shutdown}.py; engine/router tests, facade tests/fabric_sdk.rs; serving/server tests and artifact/cache fixtures |
| Build and qualification | scripts/check.sh, scripts/check-gpu-feature-boundaries.sh, scripts/native-gpu.py, tools/, nix/workspace.nix, release/installer scripts |

### 5.1 Shared native/Web control and identity

One Runtime, one facade ControlPlane, one process-local Selection and
one shared Actions slot (not a queue). Resident planner executes original
load/invoke/run/unload/listener and transfer owners; browser and native
observe same state. Exact digest/source, registration, listener generation,
transfer operation and action IDs are distinct. Aliases do not create
new owners. Confirmation/request revalidates observed key and resource;
no stale retarget. Selection, pointer motion, filter, settings preview,
view navigation never execute. Cancellation request is not completion.
Bounded output receipt and original terminal retirement/join remain
authoritative. Native-private draft/output separate from Web receipt.
One typed v2 ConfigDocument/ConfigFile, default/named/explicit selection
and CLI precedence, future-only activation, atomic exact compare and
raw repair, no independent store or arbitrary browser file/execution
authority. Existing safe operational log ring, no persistent prompt
history. Bounded original events and SSE resync, no parallel history.

### 5.2 Embedded Web control security

Embedded offline-built SolidJS/Vite assets, independent CLI Web feature
(no terminal dependencies). Numeric loopback, OS random 32-byte control
credential in private URL fragment, consumed/scrubbed and tab-scoped;
lock clears it. Independent inference bearer. Authenticated snapshot,
events and exact typed actions/receipts; bounded connections/headers/
JSON/SSE, strict numeric Host/origin and same-origin rules, no CORS or
credential/path/argv authority. See current wire/actions/http/settings
source and prior UX-9A/B/C/D/E acceptance. Only supported approved
cache/pinned source/recipe and typed actions; explicit consent and offline
policy, no implicit acquisition. Web and TUI can coexist, share actions,
read-only observations, original transfer and config revisions. Local
foreground host retires clients/actions/listener/runtime in owned order.

### 5.3 Latest native UX (supersedes all toolbar designs)

Whisrs v0.2.1 inspired semantic color depth and restrained palette.
Dark/Light/Auto; Auto terminal hint and honest dark fallback, active
probe deferred. Scoped mouse capture and ui.mouse auto/always/never;
F8 session override and terminal restoration, no clipboard bridge.
Actual render-derived hit regions, matching press/release and modal
masking. Per-view focus, rail/split drag, independent scrolling/scrollbars,
maximize/restore, compact collapse/navigation. Keyboard equivalents
remain, Ctrl+arrows split, Alt+arrows rail, F6 panes, F7 size.

User screenshots rejected visual overload, then rejected even the
curated top toolbar. **No toolbar**. Search-only top line in Models,
Devices, Profiles, Config, Running and Logs; ordinary printable input
(including former shortcut letters/digits/punctuation) focuses/filter
locally as typed. Ctrl-F/U, Esc clear, Tab leave, arrows select, filters
per shelf. Alt+1–9 views, explicit Alt operation letters, F1 help,
F4 context, F10 session, Ctrl-Q quit, original modal keys. Non-search
views retain ordinary shortcuts. Row selection inert; selected ellipsis,
matching right-click, Enter model/device/registration context; profile
Enter editor; Config Enter YAML value, typed picker via menu/Alt-e,
F2 YAML fallback. Session menu Web/Appearance/Refresh/Quit. Context
only applicable actions (cached load/run/inspect/verify/alias/removal;
catalog/partial pull/details). Offscreen/collapsed/no-row pane fallback.
No repeated nav/shelf/theme controls, debug slogans or shortcut wall.

Smart IEC B/KiB/MiB/GiB/... with at most two decimals (exact bytes kept
in details and machine data). Concise model name/size/source/verification
status; exact digest/revision/raw size and diagnostics only on demand.
Identity anchors per shelf/list/ready owner/log sequence and response
byte position survive reorder, empty refresh and viewport clamp. F3
distinct Running details/response viewport, F9 follow; response header
registration first, truncate long aliases. Logs safe selected entry
details; retention eviction refuses rather than retarget. Full short
terminal dialogs with scrolling, actual 20×7 tested. Grapheme-aware
bounded drafts, caret, mouse/Shift selection, drag edge by real event,
safe raw YAML and paste, no click-through/implicit submit.

### 5.4 Full production defaults and diagnostics

CLI Cargo default is ["all"], existing production all-models/all-backends,
CPU/CUDA/HIP/Vulkan, Qwen/Gemma model/source features, acquisition, serve,
TUI/Web. Qualification, diagnostic and experimental native attention
features opt-in; do not use --all-features as equivalent to production
default. Facade defaults remain small. Portable Nix/release build remains
explicit CPU/Qwen3.5/text/TUI/Web (nix/workspace.nix). This was a deliberate
implementation distinction at c0e4794d; **full Cargo default is not proof
that portable distributed artifacts now include all backends**. Reconcile
user's intended full normal package vs explicitly portable assets with
platform owner; no silent cross-platform toolchain requirement or
claim of all-backend release.

ServiceError and safe FailureDetails in engine/router/error.rs. Original
typed failures preserved through core/engine/SDK/CLI/HTTP/SSE. Setting
serve.enable_error_details defaults false, CLI --enable-error-details,
requires authenticated loopback listener and ZEROGRAVITY_API_KEY
configured before start. Only authorized requests get optional details;
bounded original diagnostic text (up to 512 characters per text and
three source causes) can contain internal information; trusted local
debug only, not public/reverse-proxied untrusted endpoint. Native retains
actual diagnostic, shared operational log stores only safe metadata.
Actual loopback HTTP and SSE opt-in/redaction tests passed; browser
configuration whitelist deliberately cannot enable diagnostic exposure.
No change to already-owned listener from profile edits.

### 5.5 Live registration performance (UX-P-M1, latest)

[Owning spec](../plans/cli-ux/LIVE_MODEL_PERFORMANCE.md) is precise. One
fixed-size RegistrationPerformance on original registry Route, request
timing on original ModelRead. No second request registry/reader/scheduler
or extra per-token clock, allocation, mutex or GPU sync. Use existing
committed non-EOS frontier/timestamp, count once even if decoder/transport
fails or parked resumes. Relaxed atomics for counts/durations; terminal
receipt stores last sample once under small mutex. Existing sizeof-based
admission prices added fields.

- Age: monotonic registration age, includes idle; new registration resets,
  aliases share counters, native/Web/HTTP share actual owner.
- TTFT: engine acquire to first committed non-EOS token (includes queue,
  tokenizer, prefill; not HTTP ingress-to-byte or GPU time). Valid first
  token of active or ended request contributes even if later failure/
  cancel; zero-output has no sample.
- Average decode TPS: token-weighted valid within-request commit intervals
  / summed durations (includes scheduler/backpressure delays). No time
  across requests, no idle dilution, no rate without interval and time.
  Aggregate includes active and ended; distinct from wire delivery.
- Completed requests: original Done receipt accepted by bounded presenter,
  not scheduler work-step count. Last completed is registration's last
  sample, not necessarily the visible local response. Snapshot values may
  advance between fields; no fictitious exact per-request association.
- Recent throughput: client delta of committed output / elapsed age via
  existing native 1 s / Web 2 s refresh. Selected owner only, <=24 samples,
  suppress sub-750 ms, clear gap >6 s/regression/replacement, observed idle
  zero not missing. No new timer/history. Normalize trend to observed peak,
  label actual latest; not a per-request speed comparison.
- Running TUI/Web selected panel shows age, Avg decode/TTFT, recent
  sparkline, completion/output counts, actual provider and reported layer
  execution; no samples em dash. Alt-D/context performance details and
  Web disclosure, full aliases/technical observations on demand. F3
  response remains separate. No public inference metrics schema change.
- ModelDetails.layer_placement adapter-supplied zero-based half-open
  actual spans. CPU Qwen verified geometry Host CPU; original whole-text
  single-device prepared source exact plan layers + actual resident
  device. Older resident adapter without map stays unknown. Compute
  placement distinct from weights/storage/state/admitted capacity and
  actual physical usage. Do not infer offload/split from RAM/model name.
  Web explicitly projects safe numeric data and original lease bounds.
  CPU real 24-layer placement verified; GPU compile-only.

## 6. Original chronology: beginning through Phase 1

Original October 10 design, October 12 review, first October 13 commit,
P00–P23 roadmap, Fabric vNext transition, source/state/resource/numerics,
native CUDA/HIP and portable CPU, scheduler/serving and model execution
history are preserved in original docs and prior comprehensive packs.
Appendix C extends their exact Git ledger through latest M1. Appendix E
retains UX-9B main body including exact accepted post-matrix foundation
table (§6.1), plus direct previous pack references. Read the
[25 Nov Phase 1 exit](FULL_COMPREHENSIVE_HANDOFF_2026-11-25_PHASE1_EXIT.md)
§§6–7 and Appendix D (all 49 post-matrix commits and landing narratives),
its Appendix E and older packs for full earlier chronology. Do not
reconstruct designs from a brief summary or recreate rejected experiments.

## 7. Every Phase 2 and UX-P step

The UX-9B pack Appendix D preserves every earlier Phase 2 landing (UX-1
through UX-9B). Current Appendix D preserves every one of the subsequent
15 commits, including the plan-only UX-P decision. Numbers below are
historically recorded scopes, not summed unique tests or a new full pass.

| Slice | Commit | Accepted/focused scope and result |
| --- | --- | --- |
| UX-1 | e6211608 | Shared CLI presentation, capability/typed failure; full 2431/0/106 |
| UX-2 | bcdb46a8 | Typed tree and cold resolver, exact source/config/device; full 2451/0/106 |
| UX-3 | 3cf1bb2d | SDK ControlPlane, owner snapshots/actions/events; full 2457/0/108 |
| UX-4 | da2385c7 | Shared transfer/checkpoint pause/resume/cancel; full 2459/0/109 |
| UX-5 | 8e382947 | Canonical run, pinned acquisition and reusable resident owner; full 2465/0/109 |
| UX-6 | 84e0681c | Canonical serve, joined listener and real alias equality; full 2466/0/109 |
| UX-7 | 7f5fc4eb / abe98b83 | Unified document/profiles, recovered interrupted acceptance; full 2473/0/109 |
| UX-8A | 08e7e9d3 | Native shell, palette and terminal lifecycle; full 2477/0/109 |
| UX-8B | 0c7ad87a | Native resident actions; full 2479/0/109 |
| UX-8C | 9bf7829a | Native inventory/transfer/device; full 2481/0/109 |
| UX-8D | 71e36933 | Native config/profile/logs; full 2483/0/109 |
| UX-9A | 4e472aed | Authenticated embedded Web shell; full 2485/0/109 |
| UX-9B | 4f23983a | Web resident/listener shared slot; full 2489/0/109 |
| UX-9C | a6744b12 | Web Models/Downloads/Devices, original exact/pinned actions, actual controlled transfers and cached lifecycle (48.13s); full 2489/0/109 |
| UX-9D | 543194ac | Web Profiles/Config/Logs, exact safe settings, bounded logs; gate interrupted, recovered in 9E |
| UX-9E | 1a878990 | Simultaneous clients and shared invalidations/transfer/settings; real cached 54.16s, full 2491/0/109 |
| UX-10 core | 22c6c0a7 | Cold doctor, typed pipeline inspection and command cleanup; 59/0/1, interrupted CUDA rebuild gate |
| UX-10 exit | 1b8de1cb | Cold backend/quota observations, recovered gate; focused 44/0/2; full 2505/0/109 |
| UX-P design | 3b584f72 | User-inserted polish before public serving; design only, original spec in D.6 |
| UX-P1 | b1073e7c | Semantic themes native/Web, Auto hint; 81/0/3 + JS 3/0 |
| UX-P2 | ab3cf974 | Mouse and panes, scoped capture/scroll; 84/0/3 |
| UX-P3A | 5da96251 | Exact bound actions/modal buttons; 82/0/3 + cached 1/0/0 (52.81s) |
| UX-P3B | 8a2c2ae8 | Typed choice/log filters/grapheme editor/local search; 86/0/3 + cached 1/0/0 (55.96s) |
| UX-P4A1 | f723b0b6 | Screenshot-driven concise summaries, Details, smart sizes; 87/0/3, actual PTY/transfer; resident rerun blocked by memory |
| UX-P4A1 correction | d1e27080 | Toolbar-free object context, inline live search and session menu; 88/0/3 + cached 1/0/0 (57.77s); prior memory blocker resolved |
| UX-P4A2 | 0ee66351 | Stable anchors, separate response/log details, compact dialogs/drag; 91/0/3, real native/attached-Web 1/0/0 (34.49s); browser passed separately |
| Defaults/errors | c0e4794d | Full production Cargo default, typed errors, safe HTTP/SSE+opt-in detail and logs; focused 422 pass/3 skips across recorded affected suites; default check passed; link timed out |
| UX-P-M1 | e96cdab9 | Registration live metrics/layer detail; 109 engine, 64/0/3 CLI, Web 4/0, real cached 1/0/0 (55.61s), default check |

The earlier UX-P3 toolbar/overflow and P4A1 curated strip are **historical
implementation steps**, not current design. Do not reinstate them.
UX-P4A1 memory blocker subsequently resolved without lowering admission;
P4A2 browser/native tests were separated after heading and ANSI assertions.
Latest M1 combined cached lifecycle passed; required assembled gate still
open. Diagnostic error-chain fixes corrected an old SDK assertion from
generic Limit to Unsupported for unconfigured scheduler; config fixture
separated run-only context from valid serving profile; no weakened
resource or safety contract. Appendix D retains exact intermediate
failures, corrections and final observed scopes.

## 8. Current execution and performance authority

[CURRENT_BASELINE](CURRENT_BASELINE.md) and original backend plans govern
all measured claims. UX changes do not start backend performance work.
Retain original real qualified evidence (see historical pack §8 and
Appendix E): X2 HIP original HF V2 0.8B C1/C8 independent continuation/
retirement, explicit native vs source arithmetic; L5 CUDA resident/
streamed/paged/prefix/cohort and actual GGUF B2 capture full-vocab/all42
equality/unknown retention (public graph serving remains Native);
Vulkan named RADV/portable Q8/conv/coherent mapped producer, not whole
learned parity or general external/UMA storage; CPU GGUF 0.8B/2B original
source/cache, committed continuation and language chains.

Retained historical HIP V2 47+3 0.8B C8 43.638 tok/s, 2B C8 30.949 tok/s,
27B C1 2.228 tok/s and older small-model vLLM 219.612/140.847 tok/s
are historical condition-specific numbers, **not a fresh matched parity
comparison**. Exact conditions and rejected alternatives in older pack
Appendix E §§8–9, baseline and owning plans. Approved pinned X2 vLLM
closure previously unavailable; revalidate/restore approved local
closure before new comparison. Do not download replacements or tune
against unverified reference. User's delivery priority ROCm then CUDA
then Vulkan then optional compiled providers. Read
[backend iteration](../plans/cpu-gpu-optimization/NEXT_ITERATION_BACKEND_PARITY_AND_EXECUTION_PERFORMANCE.md),
[backend parity](../plans/cpu-gpu-optimization/BACKEND_PARITY.md),
[backend TODO](../plans/cpu-gpu-optimization/TODO.md),
[numerics TODO](../plans/numerics-and-context/TODO.md) and
[model execution TODO](../plans/model-execution/TODO.md).
M1 instrumentation overhead and GPU placement physical behavior have
not been measured/qualified by a performance comparison. No imagined
TPS/TTFT/layer estimates; illustrative mockup numbers are not evidence.

## 9. Reusable primitives, rejected approaches and holds

Preserve correct model-neutral GEMM/solve, GDN/attention/norm,
tiled/batched/paged/streamed transport, source-window, state/layout,
sampling, graph/cache, admission/lease, completion, pipeline primitives.
Native graph timing at useful widths did not justify production selection;
retain graph substrate/cache. Paged scatter-tail-zero consumer lost;
retain reusable kernel. Earlier exact held/rejected experiments and
reusable results are in Phase 1 exit pack Appendix E §9, earlier packs,
current baseline and original owner plans. Do not repeat without
materially new evidence or remove dormant primitives to silence warnings.
Renoir/gfx90c AMD launch hold remains. Preserve finite captured CUDA
qualification route distinction; no production selector shortcut.
A1 SM89 original-source CUDA qualification remains pending; do not
turn feature compilation into capability. General source/model/codec/
quality/native Vulkan/UMA/foreign/mixed/async SSD holds remain explicit.
Optional compiler integrations after native campaigns, not replacement
for original state/math/provider/physical contracts.

## 10. Build, validation, host and Git procedure

Read AGENTS.md, current scripts, installed toolchain/cache and prior
Phase 1 pack Appendix E §10 for exact historical physical/bundle
recipes. Mandatory assembled gate when outer connector shares network:

~~~sh
cd /home/pvl/spaces/abird/src/zerogravity
export CARGO_BUILD_JOBS=8 RAYON_NUM_THREADS=4 PYTHONDONTWRITEBYTECODE=1
bwrap --unshare-net --bind / / --dev-bind /dev /dev -- \
  python3 -B scripts/native-gpu.py --require-private-network -- bash scripts/check.sh
~~~

If observed host pressure use fewer build jobs; run real cached model
serially with original 8 GiB admission and original cache. Never bypass
mandatory private-network harness, enable host network/privileges or
download unapproved inputs. For focused checks that need local HTTP,
use the same private harness and the loopback setup from scripts/check.sh;
only private owned user/network namespace loopback, not host interfaces.
Use locked/offline builds. Review every result and cleanup verdict.
Inspect actual staged paths before signed commit; no unrelated edits,
remote push or destructive operations. Keep one main worktree, exact
bundle transfer for execution only; do not infer old X2/A1 revision.
No need to create more snapshot artifacts during normal work.

### 10.1 Required feature and consumer distinctions

The old command `--features serve,acquisition` alone now inherits full
GPU default. For explicitly bounded CPU resident fixture use:

~~~sh
ZG_TEST_BROWSER=1 cargo test --locked --offline --release \
  -p zerogravity-cli --no-default-features \
  --features cpu,model-qwen35,text,tui,web,serve,acquisition \
  --test run_unified canonical_serve_profile_and_cached_aliases_use_one_listener_owner \
  -- --ignored --exact --nocapture --test-threads=1
~~~

Run inside the private harness as above. This is the existing real
cached owner fixture, not a new load or comparator. Retain explicit
test/feature separation, Web-only independent of terminal, CPU-only
without optional GPU/serving dependencies, original acquisition and
serving fixtures. Ratatui/Crossterm and locked Unicode libs optional
TUI-only. SolidJS/Vite assets built locally using installed pinned
dependencies; Web README documents npm test/build and Firefox BiDi.
No Node/runtime remote assets. Actual native PTY shell/settings,
appearance, mouse, controls, editor, ergonomics, Web HTTP/Firefox and
controlled/shared downloads are existing fixtures. Follow current
scripts; no new golden matrix. Full production default check is not
the linked execution check; portable release fixture is explicitly CPU.

### 10.2 Current acceptance caveats to recover

- Last full default link (`cargo run --locked --offline -p zerogravity-cli
  -- --version`) timed out after 300 s in final CLI build. Reattempt
  using actual existing products/resources, inspect job status and exit.
  Then actual --version/help/doctor and appropriate launch; not merely
  feature graph. Do not claim CUDA/HIP/Vulkan inference from linking.
- The last full-production strict Clippy run at c0e4794d failed in
  zerogravity-gpu with 38 errors under -D warnings, including unused
  reusable primitives and existing idiom/arity lints. M1 default cargo
  check passes with three unused-primitive warnings; full strict lint
  remains unqualified. Inspect latest actual results; address cleanly
  where required by gate, do not remove valid primitives, blanket-disable
  lint or weaken numerical behavior.
- Audited gate concern: scripts/check.sh lines 131–132 at implementation
  checkpoint still run shared CPU cache clippy/test with
  -p zerogravity -p zerogravity-cli --features zerogravity-cli/cache but
  no --no-default-features. These inherited old CPU CLI defaults; now
  include all production GPU features. Review and explicitly preserve the
  intended original CPU/cache coverage separately from the full-product
  check. Also inspect other implicit default invocations (line 143
  acquisition check); keep default coverage intentional. This is a
  handoff audit finding, not a newly executed failure or script edit.
- Existing portable Nix/release assets remain explicitly CPU. Full native
  default packaging scope vs portable assets needs explicit platform
  reconciliation. See §5.4; no silent change to portability.
- ControlEventKind::ServeRequestFailed listener-generation/metadata
  assertion added to real cached facade SDK fixture during error work;
  not established by the focused non-cached SDK tests. Qualify actual
  affected event case at P4B. Inspect the exact live test and its
  authorized cache prerequisites/ignore flags.
- M1 GPU layer placement compiled only; actual hardware observation on
  already-qualified routes and observer-overhead qualification pending.
  No new GPU route/SDK download/physical performance claim needed from
  this docs-only task. See exact owning scope. If a resource/backend hold
  blocks a required check, record and keep phase unaccepted.

## 11. Next coherent implementation unit — UX-P4B

**Required next according to live ordered plan, queue and M1 design:**
finish assembled UX acceptance before public serving. Previous final
chat also identified optional M2 as a natural deeper-inspection follow-up;
preserve it, but the owning M1 plan explicitly says not to delay P4B for
imaginary mixed placement or a new profiler.

1. Revalidate actual repo/HEAD/index/worktrees, active jobs, physical
   owners and available disk/memory before editing. Verify old failed/
   interrupted checks from actual logs/results; do not replay accepted
   M1 or earlier foundation work.
2. Reconcile the new Cargo default with explicit feature gates,
   full-product linked launch, portable asset scope and the precise
   cache-only gate issue (§10.2). Preserve typed error semantics,
   public-default redaction, trusted diagnostic opt-in and log metadata.
   Repair concrete failures cleanly. Do not grant hardware/model
   qualification to force a build/test result.
3. Run affected meaningful focused checks, actual cached CPU
   native/Firefox/HTTP lifecycle, exact registration metrics/layers,
   stale identities, real event metadata, config changes and same-owner
   simultaneous transfer publication/cancel. Preserve unchanged 8 GiB
   bound and exact output. Review real visible TUI/Web layout, compact/
   mouse/keyboard/no-color and on-demand detail where relevant. Full
   product compile/link/launch and physical placement evidence as
   actually supported; unsupported GPU map/qualification must stay explicit.
4. Run the complete assembled mandatory-private-network workspace gate
   last; no claim of P4B/UX-P acceptance until its required scope passes.
   If blocked, record exact failed scope and continue independent ready
   work within a coherent turn, don't invent acceptance. Retain explicit
   skips with reasons. Signed owned checkpoint and clean tree.
5. Only after P4B exit, public serving/ordinary benchmark iteration,
   then ROCm parity, CUDA parity, native Vulkan parity, concrete optional
   compiled providers in that order.

### 11.1 Optional UX-P-M2 (not lost, not a second runtime)

If the user chooses deeper performance inspection next, or a bounded
actual refinement is justified within P4B, follow LIVE_MODEL_PERFORMANCE
M2. Preserve original owner and exact request/response identity, current
progress and explicit phase metrics only where existing trustworthy
boundaries provide them: queue, tokenization, prefill, decode. Do not
label last registration sample as the current local response under
concurrency. Optional bounded recent distributions/history only if
justified, not another request registry. Actual layer roles and ranges
from verified geometry, per-device topology from admitted placement,
memory by original owner class (weights/KV/recurrent/state/scratch and
capacity distinguished). Exact detailed layer timing only via explicit
profiling using real completion/timing contracts; no guessed percentages,
no default per-layer sync or extra hot-path overhead. Qualify physical
execution and instrumentation overhead before claims. Unknown remains
unknown. No mandate to invent unsupported mixed placement/offload,
future GPU profiler or new backend. Maintain minimal Running panel with
technical inspection on demand, not a second monitoring UI.

### 11.2 Other open scopes to retain

All owner TODOs and unchecked full-plan scopes remain. Appendix B
captures literal pending markers; original documents include additional
prose, tests, deferred constraints and historical design checklists.
Broader catalog/presets/conversation history, dedicated graph
visualization, advanced forced provider interactive/JSONL progress and
aggregate component download controls, future model/media/distributed
and generalized state/physical interop are not silently closed by UX-P.
Optional Auto terminal background probe or additional themes only after
applicable approval. No new execution abstractions for presentation.
Public serving design remains dormant until UX-P exit.

## 12. Specification navigation and complete historical context

Appendix A links the entire tracked .agents/docs, .agents/plans,
.agents/prior-art, .agents/research and docs corpus. Original source files
remain in place. The older comprehensive packs are retained and their
appendices contain the exact previous chronology and original design.
Use this hierarchy, not a new duplicated source-of-truth:

| Subject | Key authority |
| --- | --- |
| Original design | [master plan](../plans/ZEROGRAVITY_MASTER_PLAN_2026-10-10.md), [vision](VISION_SCOPE_AND_PRINCIPLES.md), [modular architecture](MODULAR_AI_SYSTEMS_REFERENCE_ARCHITECTURE.md), [engineering spec](../../docs/ENGINEERING_SPECIFICATION.md) |
| Source and semantics | [formats](FORMATS_QUANTIZATION_AND_LOADING.md), [model semantics](MODEL_SEMANTICS_AND_COMPATIBILITY.md), [composition](COMPOSITION_AND_EXTENSION_CONTRACTS.md), [derived quant](DERIVED_QUANTIZATION_CONTRACT_2026-10-15.md), [verified assemblies](VERIFIED_MODEL_ASSEMBLIES_AND_SPECIALIZATION_V5_1_2026-10-15.md), [immutable source](../plans/IMMUTABLE_SHARDED_SOURCE_IMPLEMENTATION_SPEC.md), [acquisition spec](../plans/MODEL_ACQUISITION_IMPLEMENTATION_SPEC.md) |
| Regions/provider/physical | [semantic IR](SEMANTIC_IR_AND_COMPILER.md), [regions](../plans/pipeline-composition/REGIONS_AND_PROFILES.md), [policy and reuse](../plans/pipeline-composition/PROFILE_POLICY_AND_COMPONENT_REUSE.md), [shared execution](../plans/pipeline-composition/SHARED_EXECUTION_ARCHITECTURE.md), [naming](../plans/pipeline-composition/STRUCTURE_AND_NAMING.md) |
| Fabric evolution | [Fabric vNext](../plans/fabric-vnext/README.md), [decisions](../plans/fabric-vnext/DECISIONS_AND_OPEN_QUESTIONS.md), [acceptance](../plans/fabric-vnext/IMPLEMENTATION_AND_ACCEPTANCE.md), [transition](../plans/fabric-vnext-transition/README.md), [transition contracts](../plans/fabric-vnext-transition/ARCHITECTURE_AND_CRATE_CONTRACTS.md) |
| Physical/state | [core protocol](ENGINE_CORE_EXECUTION_AND_PROTOCOL.md), [state](STATE_CACHE_AND_TRANSACTIONS.md), [residency](MEMORY_RESIDENCY_AND_IO.md), [virtual tensor/SSD](UNIFIED_VIRTUAL_TENSOR_RESIDENCY_AND_SSD_STREAMING_2026-10-15.md), [memory TODO](../plans/memory-and-streaming/TODO.md), [gfx1151 safety](GFX1151_ROCM_KERNEL_COST_AND_UNIFIED_MEMORY_SAFETY_CONTRACT_2026-10-15.md) |
| Math/numerics/performance | [operator catalog](OPERATOR_AND_FUSION_CATALOG.md), [kernel/codegen](DEVICE_KERNEL_AND_CODEGEN.md), [cost model](COST_MODEL_AND_AUTOTUNING.md), [numerics](NUMERICS_QUALITY_AND_BENCHMARKS.md), [correctness first](../plans/numerics-and-context/CORRECTNESS_POLICY.md), [baseline](CURRENT_BASELINE.md), [experiment program](../plans/EXPERIMENT_PROGRAM.md) |
| Scheduling/serving | [scheduling/speculation](SCHEDULER_BATCHING_AND_SPECULATION.md), [router/residency](MODEL_ROUTING_RESIDENCY_AND_MULTI_MODEL_SERVING_CONTRACT.md), [paged/batched state](../plans/serving-and-speculation/BATCHING_AND_PAGED_STATE.md), [fused batching](../plans/serving-and-speculation/FUSED_GPU_BATCHING.md), [goodput contract](JOINT_CONCURRENT_SERVING_OPTIMIZATION_AND_GOODPUT_CONTRACT_2026-10-15.md), [public serving design](../plans/serving-and-speculation/NEXT_ITERATION_OPENAI_COMPATIBLE_SERVING.md) |
| Product UX | [zero-config API](ZERO_CONFIG_CLI_CONFIG_AND_OPENAI_API_CONTRACT.md), [runtime UX](RUNTIME_SERVING_AND_UX.md), [CLI UX](../plans/cli-ux/NEXT_ITERATION_CLI_UX_AND_MODEL_LIFECYCLE.md), [native polish](../plans/cli-ux/TUI_POLISH_AND_MOUSE_INTERACTION.md), [live performance](../plans/cli-ux/LIVE_MODEL_PERFORMANCE.md), [CLI guide](../../docs/CLI.md) |
| Multimodal/distributed | [multimodal V5.1](UNIVERSAL_MULTIMODAL_EXECUTION_ARCHITECTURE_V5_1_2026-10-15.md), [realtime/QoS](REALTIME_MULTIMODAL_SERVING_AND_QOS_V5_1_2026-10-15.md), [NPU](NPU_GPU_COOPERATIVE_EXECUTION_AND_TASK_PIPELINES.md), [distributed placement](DISTRIBUTED_PLACEMENT_AND_NETWORK_EXECUTION_CONTRACT.md), [remote state handoff](REMOTE_PEER_TRANSPORT_AND_STATE_HANDOFF_PROTOCOL.md), [multimodal TODO](../plans/multimodal/TODO.md), [distributed TODO](../plans/distributed-execution/TODO.md) |
| Reliability/release | [security](SECURITY_AND_RELIABILITY.md), [failure](INVARIANTS_AND_FAILURE_SEMANTICS.md), [engineering](RUST_WORKSPACE_AND_ENGINEERING.md), [delivery gates](../plans/INTEGRATED_PHASE_0_AND_DELIVERY_GATES.md), [release gates](../plans/RELEASE_ACCEPTANCE_GATES.md), [traceability](../plans/REQUIREMENTS_AND_TEST_TRACEABILITY.md), [platform TODO](../plans/platform-and-release/TODO.md) |

All older comprehensive handoffs from Nov 14,15,18,20,21,22,23,24
(including foundations),25 (foundations and Phase 1 exit),26 (UX-9B)
are indexed in Appendix E and A. In particular the earlier full
Phase 1 pack preserves exact source, experiment and rejection scope
going back to the beginning; the UX-9B pack preserves UX-1 through
UX-9B. The present pack adds every post-UX-9B commit and an updated
pending-work snapshot. Do not execute superseded next-agent prompts.
No original files were deleted or replaced. Historical data remains
available at original Git revisions even when live owner prose evolves.

## 13. Copy/paste starting prompt

The following is also stored in [NEXT_AGENT_2026-11-26_UX_P4B.md](NEXT_AGENT_2026-11-26_UX_P4B.md).

~~~~markdown
You are taking over ZeroGravity. Continue in:

`/home/pvl/spaces/abird/src/zerogravity`

### 1. Acquire complete project context

Read in order:

1. `AGENTS.md`
2. `.agents/plans/README.md`
3. `.agents/docs/FULL_COMPREHENSIVE_HANDOFF_2026-11-26_UX_PM1.md` §§0–13 first
4. `.agents/plans/FULL_IMPLEMENTATION_PLAN.md`
5. `.agents/docs/CURRENT_HANDOFF.md`
6. `.agents/docs/CURRENT_BASELINE.md`
7. `.agents/plans/TODO.md` and relevant owning TODOs
8. `.agents/plans/cli-ux/NEXT_ITERATION_CLI_UX_AND_MODEL_LIFECYCLE.md` (especially §46A)
9. `.agents/plans/cli-ux/TUI_POLISH_AND_MOUSE_INTERACTION.md` and `.agents/plans/cli-ux/LIVE_MODEL_PERFORMANCE.md`
10. Relevant CLI/device-routing, acquisition, serving, SDK, platform and structure/naming specifications and TODOs.

The comprehensive handoff Appendix D preserves every checkpoint since UX-9B, including original landing narratives, decisions and exact changed paths. Appendix E and the linked earlier complete handoffs preserve the original architecture, every historical foundation/UX implementation step, accepted evidence, rejected experiments, reusable primitives and pending work going back to the October design. Use the detailed sources and previous exact experiment conditions; do not reconstruct from assumptions or repeat rejected trials without materially new evidence. Appendix B is a historical pending-marker snapshot, not a second task queue.

### 2. Repository and execution authority

Implementation checkpoint: `e96cdab98d47c5c84001c1afaaebd8f126b9e0b3` (UX-P-M1).
Accepted UX-1–10 exit: `1b8de1cb87fa4815655cbb6700e6633946f93305`.
Accepted bounded Phase 1 exit: `d5030ad4ce476b913a46056d492992ccf1b8edf7`.
The new comprehensive handoff is a later documentation-only commit; find its exact current SHA with Git. Revalidate actual HEAD, working tree, staging, tracking ref, worktrees, jobs, available resources and original shared physical owners. Live source, executed tests and owning plans take precedence if advanced.

Use **dotlink---l5 as the sole source-authoring connector**. Main is the only persistent branch/worktree. X2/A1 execution/testing only; synchronize exact L5 commits via verified incremental Git bundles and clean fast-forward when necessary. Preserve unrelated work and shared physical owners. No destructive Git, unapproved downloads, privilege changes, GPU resets, unrelated process cleanup or push. Sign commits; do not disable signing.

### 3. Current implementation and product direction

Phase 1 bounded foundations and UX-1–10 are accepted. UX-P1/P2/P3A/B/P4A1 (including toolbar-free correction)/P4A2, full native Cargo CLI defaults/error diagnostics, and UX-P-M1 live metrics are implemented and focused-qualified. The interrupted M1 turn had actually committed; do not redo it. Latest complete full isolated gate was UX-10: 2,505 Rust test executions, zero failures, 109 explicit skips, 318 suites, release Web/TUI/installer/scripts. This is not final acceptance of later changes.

Latest M1 approved cached Qwen 0.8B GGUF CPU/browser/native/HTTP lifecycle passed (55.61s, five filtered), shared registration counters/reset, exact 24 CPU layers, original 8 GiB admission, exact output and stale/shared settings retained. Engine 109/0/0, combined CPU CLI 64/0/3, Web 4/0, focused strict lint and full default cargo check. GPU layer placement compile-only. No performance parity claim.

Preserve one Runtime, ControlPlane, shared Actions slot, process-local Selection, typed ConfigDocument, original source/cache/transfer/registration/scheduler/listener owners, physical ResourceArbiter and sole StateRoot publication. Browser control auth is separate from inference API auth.

User-directed UX: no permanent action toolbar; one rail, inline live local search (ordinary printable keys filter), exact selected-object ellipsis/right-click/context menu, separate session menu, concise normal summaries and details on demand, smart IEC sizes, Dark/Light/Auto, complete native mouse/keyboard and resizable independent panes/editors. Running is the quiet live metrics dashboard. Preserve exact registration TPS/TTFT semantics and actual layer maps, independent response viewport, no fictitious telemetry or hidden extra ownership.

Default Cargo CLI is production all (all current model features, CPU/CUDA/ROCm/Vulkan, acquisition, serve, TUI/Web); no diagnostic/experimental features. Preserve explicit minimal/portable and embedding builds. Full default check passed, linked default cargo run timed out in final compilation; verify actual launch. Full-GPU Clippy debt remains. Portable Nix/release explicitly CPU; do not imply all-backend distribution without verifying platform intent. Error chain retains typed resource/I/O causes, accurate InvalidInput/Unsupported vs Malformed. HTTP/SSE safe public message/stable status+code, optional authenticated loopback debug details, safe structured operational logs regardless of HTTP setting. No pre-auth leakage or browser-safe toggle of trusted diagnostic exposure.

### 4. Next coherent unit — UX-P4B required assembled acceptance

Read comprehensive handoff §11, especially §10.2 known gate/default/build caveats. Reconcile actual full production link/launch, explicit feature-gate coverage (old cache-only gate commands inherited changed defaults), existing GPU lint debt and real changed-consumer regressions. Do not reduce admission, remove valid dormant primitives to silence lint, weaken tests or bypass private-network checks. Qualify the new exact listener error-event assertion and actual cached browser/native/HTTP/shared transfer/config behavior under unchanged owner policies and 8 GiB bound. Real GPU layer reporting requires actual already-qualified adapter observation before claiming it; unsupported remains explicit.

Optional UX-P-M2 is preserved in LIVE_MODEL_PERFORMANCE.md. If justified by actual existing boundaries or explicitly requested, implement bounded exact response/request identity, phase timings and deeper actual layer/device/memory inspection. No fake offload/split, no estimated layer time, no default profiling sync/overhead, no second runtime/request registry. Do not make optional profiler architecture a blocker to P4B. The current ordered plan/queue requires P4B next; no public serving before UX-P acceptance.

Use the existing isolated harness. On outer shared-network connector:
~~~sh
export CARGO_BUILD_JOBS=8 RAYON_NUM_THREADS=4 PYTHONDONTWRITEBYTECODE=1
bwrap --unshare-net --bind / / --dev-bind /dev /dev -- \
  python3 -B scripts/native-gpu.py --require-private-network -- bash scripts/check.sh
~~~
Reduce build jobs for observed resource pressure. Use locked/offline approved inputs, real cached owner fixture serially with explicit CPU/model/text/TUI/Web/serve/acquisition features and --no-default-features, original admission; existing private loopback setup. Full assembled gate last. Reuse applicable evidence; no premature phase exit or claim from compilation alone.

### 5. Remaining delivery order

1. UX-P4B comprehensive acceptance (optional capability-bound M2 as appropriate).
2. Public OpenAI-compatible serving and normal CLI/SDK/server benchmark acceptance (owning dormant serving specification).
3. Match or beat optimized matched vLLM on ROCm first.
4. Match or beat optimized matched vLLM on CUDA.
5. Native Vulkan parity.
6. Concrete optional MIGraphX, TensorRT, IREE and ONNX Runtime integrations.

Original Hugging Face/Safetensors and GGUF remain primary source paths. Native execution must not require ONNX conversion or external graph compiler SDKs. Do not reopen bounded Phase 1 or start premature backend performance tuning. Preserve historical rejected approach/held device/model/quality conditions; restore approved comparator only with actual authorization.

### 6. Engineering principles and cadence

Elegant, simple, clear, intuitive. Rust library/execution fabric: zero-cost abstractions/direct hot paths and high performance when clean; simple maintainable design first where complexity is not justified, optimize measured hot areas. Reuse mature optimized mathematics and attack large bottlenecks first. Preserve useful correct model-neutral primitives even when an individual consumer loses; remove losing selector and temporary probes, not reusable implementations. Preserve exact source/numerical/layout/state/resource/ownership/completion/cancellation. No overengineered test/evidence/receipt/pinning administration. Reuse still-applicable evidence and meaningful focused/actual changed-consumer checks; assembled mandatory-private-network gate last.

Aim for one substantial coherent, validated, signed checkpoint per turn within approximately 30 minutes. Update owning TODOs/current handoff/baseline as appropriate. Recover interrupted work from actual source and results, not assumptions. Report completed scope, exact commits, tests, limitations, blockers, next unit and clean worktree, then pause. Do not push unless explicitly instructed.

**Revalidate current state and continue the next required coherent acceptance unit, UX-P4B.**
~~~~

End of historical body. Continue at current §11, not the prior prompt.
