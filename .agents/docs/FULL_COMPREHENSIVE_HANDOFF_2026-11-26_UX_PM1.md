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

## Appendix A — complete engineering document inventory

All 172 tracked files under the engineering documentation, planning, prior-art, research and public docs directories at e96cdab98d47c5c84001c1afaaebd8f126b9e0b3. Original locations and exact bytes are retained; this is a navigation index, not another maintained inventory. New handoff/prompt and pointer-only documentation changes follow this baseline. Consult live owner files for current work. Prior-art/research are not capability grants.


### .agents/docs/ (58 files)

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


### .agents/plans/ (70 files)

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


### .agents/prior-art/ (12 files)

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


### .agents/research/ (24 files)

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


### docs/ (8 files)

- [docs/CLI.md](../../docs/CLI.md)
- [docs/DEVELOPMENT.md](../../docs/DEVELOPMENT.md)
- [docs/ENGINEERING_SPECIFICATION.md](../../docs/ENGINEERING_SPECIFICATION.md)
- [docs/PUBLICATION_AND_REVIEW_CHECKLIST.md](../../docs/PUBLICATION_AND_REVIEW_CHECKLIST.md)
- [docs/README.md](../../docs/README.md)
- [docs/REFERENCES.md](../../docs/REFERENCES.md)
- [docs/WHITEPAPER.md](../../docs/WHITEPAPER.md)
- [docs/architecture-overview.svg](../../docs/architecture-overview.svg)

Root authority: [AGENTS.md](../../AGENTS.md) and the protected, user-authored [README.md](../../README.md).

## Appendix B — complete unresolved checklist snapshot

Literal open/active/blocked checklist markers and immediately indented continuations from non-takeover Markdown in Appendix A at e96cdab98d47c5c84001c1afaaebd8f126b9e0b3. Fenced code examples and the historical comprehensive/current-handoff/baseline/prompt files are excluded (they are indexed verbatim at original locations). Historical designs may include unchecked markers; they are not fresh work authorization. Source line numbers refer to the exact implementation checkpoint. The live owner TODOs and complete original prose, constraints, tests and decisions remain authoritative; this lexical snapshot does not pretend that prose-only work can be identified by a checkbox parser.


### [.agents/docs/SECURITY_AND_RELIABILITY.md](SECURITY_AND_RELIABILITY.md) — 12 unresolved markers

**Lines 104–104**

~~~~markdown
- [ ] Defined model file trust and network resolver policy.
~~~~

**Lines 105–105**

~~~~markdown
- [ ] All external source/data licenses reviewed.
~~~~

**Lines 106–106**

~~~~markdown
- [ ] No unchecked parser arithmetic or format downgrades.
~~~~

**Lines 107–107**

~~~~markdown
- [ ] Public HTTP exposure is explicit and protected.
~~~~

**Lines 108–108**

~~~~markdown
- [ ] Model-state snapshots private, bounded and versioned.
~~~~

**Lines 109–109**

~~~~markdown
- [ ] Physical RAM admission and APU watchdog exercised on hardware.
~~~~

**Lines 110–110**

~~~~markdown
- [ ] Fault injection on expert slots, snapshots, kernels and cancellation.
~~~~

**Lines 111–111**

~~~~markdown
- [ ] No execution of arbitrary Python model/plugin scripts.
~~~~

**Lines 112–112**

~~~~markdown
- [ ] Distributed transport identity/credential and fail-closed policy.
~~~~

**Lines 113–113**

~~~~markdown
- [ ] GPU memory/FFI unsafe blocks audited and sanitizer tests recorded.
~~~~

**Lines 114–114**

~~~~markdown
- [ ] Secrets/model traces excluded from CI and release artifacts.
~~~~

**Lines 115–115**

~~~~markdown
- [ ] Performance and quality claims link reproducible evidence.
~~~~


### [.agents/plans/FULL_IMPLEMENTATION_PLAN.md](../plans/FULL_IMPLEMENTATION_PLAN.md) — 41 unresolved markers

**Lines 354–358**

~~~~markdown
- [ ] Reconcile model/source/format/codec/task registration and crate boundaries
  against the live Fabric transition. Keep family-specific mathematics in
  model crates, shared mathematical operators in core/shared libraries, and
  physical execution in backends. Remove only obsolete wiring; retain useful
  correct primitives and numerical variants.
~~~~

**Lines 359–364**

~~~~markdown
- [ ] Preserve one authoritative source-bound `ModelProgram` / `ForwardPlan`
  and prepared physical executable. Typed values, state effects, layouts,
  regions and task dependencies determine valid execution. Distinguish a
  mathematical region, a native ModelAssembly and a multi-model TaskPipeline.
  Recursive composition is a semantic/planning view, not a recursive second
  runtime.
~~~~

**Lines 365–369**

~~~~markdown
- [ ] Complete the existing `RegionExecutionProvider` / `PreparedExecutable`
  / `ExecutableBundle` seam for closed regions and whole-model regions.
  Preserve the original model/source identity, numerical contract, shape
  constraints, input/output bindings and state effects. No arbitrary
  subgraph replacement that bypasses the authoritative plan.
~~~~

**Lines 370–375**

~~~~markdown
- [ ] Separate cold policy from physical capability: explicit backend,
  implementation/provider identity, numerical/approximation profile,
  execution mode and resolved region choices. Select a qualified executable
  at preparation or an explicitly supported committed-root boundary. Shape
  misses use a qualified pre-enqueue decomposition or refuse. Submitted
  failures cannot be retried as another provider on the same root.
~~~~

**Lines 413–417**

~~~~markdown
- [ ] Generalize only proven shared mathematical contracts: keep the dense
  seam and add bounded native row-wise, normalization/gating, rotary, attention,
  recurrent, convolution and quantized operators as real consumers require.
  Keep compact row/batch layouts and explicit precision; avoid forcing every
  operator into GEMM or a universal dispatch object.
~~~~

**Lines 429–434**

~~~~markdown
- [ ] Make borrowed device views and external execution/completion integration
  possible without promising universal zero-copy interop. A provider may
  plan/reuse local intermediate storage inside its admitted region. When it
  cannot import caller buffers or use the original stream, admit its internal
  memory and actual synchronization/transfers explicitly. No provider may
  publish StateRoot, schedule independent sessions or hide unaccounted memory.
~~~~

**Lines 435–438**

~~~~markdown
- [ ] Keep provider features independently optional. Native execution and all
  primary model loading remain usable with no graph compiler installed.
  Concrete MIGraphX/TensorRT/IREE/ONNX Runtime build, runtime bindings and
  model/region support are Phase 6 work, not foundation exit requirements.
~~~~

**Lines 442–447**

~~~~markdown
- [ ] Complete shared source acquisition/import/cache, verified shard/index
  and tensor-window identities, license/authorization, exact snapshot
  selection, config/tokenizer/processor/adapter validation and finite
  encoding/codec support. Preserve HF/Xet, GGUF, local and existing remote
  acquisition mechanisms; unsupported providers/format variants refuse
  explicitly. Native weights load without ONNX conversion or remote model code.
~~~~

**Lines 448–452**

~~~~markdown
- [ ] Preserve one catalog/SDK source-lifecycle authority, transactional cache
  admission, cancellation, download resume, last-reader retirement and safe
  cleanup. Retain source provenance, checked metadata and byte bounds;
  avoid duplicate allocations for aliases and components. Make actual file
  descriptors, mappings and immutable readers the authority.
~~~~

**Lines 453–458**

~~~~markdown
- [ ] Consolidate pre-Client fit/admission, stable device identity and source
  eligibility. Derive complete cold/prefill/decode/graph/provider/state/
  output cost and headroom from real geometry/physical observations; keep
  UMA overlap and host limits explicit. A typed capacity miss is distinct
  from a lifecycle error. No silent CPU/backend/profile substitution after
  work is submitted.
~~~~

**Lines 462–466**

~~~~markdown
- [ ] Close shared physical backing/range identity, original leases,
  immutable source readers, actual readiness and event-backed completion/
  quarantine. One ResourceArbiter; optional local allocator strategies and
  provider workspaces may vary under its authority. No global reset or
  reclamation of unrelated owners.
~~~~

**Lines 467–472**

~~~~markdown
- [ ] Complete the common independent session-state and candidate-publication
  contract across KV, recurrent/conv and all required state domains.
  Paged/hybrid state, complete-domain prefix reuse and COW, bounded snapshots,
  cancellation/recovery and cache compatibility use the same StateRoot.
  Only actual completion and the original authority publish; unknown work
  retains all affected owners and fences the unsafe service.
~~~~

**Lines 473–476**

~~~~markdown
- [ ] Complete shared resident/paged/streamed tensor materialization and
  transfer-retirement boundaries. Preserve resident direct access with no
  mandatory host dispatch; actual SSD windows, page/weight cache, prefetch,
  bounded async IO and eviction policies mature in later native workloads.
~~~~

**Lines 477–480**

~~~~markdown
- [ ] Finish model-neutral invocation/cohort, fair worker, batching/refill,
  backpressure, cancellation and task/pipeline control seams. Continuous
  batching, chunked/ragged prefill, speculation/MTP and multi-model placement
  share these owners. Do not grant unqualified execution modes from policy.
~~~~

**Lines 481–483**

~~~~markdown
- [ ] Preserve typed media ports/clocks, realtime lanes and remote placement/
  transport boundaries without implementing every media codec or network
  topology now. Local single-process execution requires no network.
~~~~

**Lines 487–493**

~~~~markdown
- [ ] Complete public Rust SDK and standalone serving library contracts;
  CLI/TUI/Web use the same runtime, catalog, configuration and control
  events, not a separate frontend-owned execution path. Keep model and
  backend features orthogonal, ordinary `all` free of diagnostic features,
  backend-only/model-only/serve-only dependency isolation and mixed GPU
  development shell. Integrate provider capabilities without requiring
  optional SDKs in default CPU builds.
~~~~

**Lines 497–502**

~~~~markdown
- [ ] Close remaining applicable architecture/feature/test-boundary debt.
  Keep source, model arithmetic,
  quality, cancellation, physical readiness, state/root and real execution
  checks distinct. Qualify the existing working native paths; CUDA unresolved
  parity is an explicit gate before CUDA performance, not a pretext for
  starting its optimization campaign ahead of ROCm.
~~~~

**Lines 503–507**

~~~~markdown
- [ ] Expose a common effective configuration, capabilities, resource
  snapshot, plan/region identity, structured error and execution metrics
  contract. Keep telemetry on demand, bounded and out of hot-path ownership
  decisions where possible. The UI builds on this rather than deriving
  semantics from logs.
~~~~

**Lines 735–748**

~~~~markdown
- [ ] UX-P4: visual/terminal-depth/accessibility polish, actual PTY and
  simultaneous-client lifecycle acceptance; full isolated gate.
  P4A1 user screenshot-driven minimalism: at most three quick actions,
  one navigation rail, concise summaries, on-demand exact Details and
  smart units. Follow-up removes toolbar entirely: selected exact-row
  context menus and inline live local search, explicit Alt shortcuts.
  P4A2 identity scroll anchors, separate response, safe log details,
  compact scrollable dialogs/editor drag edges focused-qualified
  (91/0/3 and native/shared transfer PTY). User follow-up: CLI Cargo
  default full production, preserve typed errors and safe public/opt-in
  authenticated diagnostic projection and structured logs. User follow-up:
  [live model performance and layer inspection](cli-ux/LIVE_MODEL_PERFORMANCE.md),
  registration-owned TPS/TTFT and selected object details before P4B.
  Optional deeper profiling follows actual provider capabilities.
~~~~

**Lines 755–759**

~~~~markdown
- [ ] Close the [OpenAI-compatible serving iteration](serving-and-speculation/NEXT_ITERATION_OPENAI_COMPATIBLE_SERVING.md):
  model catalog/router, OpenAI Chat/Responses and supported completion/
  embedding/tool/structured-output contracts, usage, errors, JSON/SSE,
  streaming/cancellation/timeout/backpressure and real concurrent requests.
  Unsupported tasks/options must refuse rather than simulate compatibility.
~~~~

**Lines 760–763**

~~~~markdown
- [ ] Preserve a separate control surface and inference API with typed
  configuration, explicit route/tenant scope, model lifecycle and shared
  scheduling. Add future WebSocket/WebRTC/media endpoints through the
  existing task contracts; do not wait on unimplemented media models.
~~~~

**Lines 764–766**

~~~~markdown
- [ ] Establish ordinary public CLI/SDK/server benchmark invocation,
  effective config and end-to-end metrics, not a qualification-only serving
  process. Retain one process/multiple model instances and pipeline routing.
~~~~

**Lines 783–786**

~~~~markdown
- [ ] Restore the approved pinned optimized vLLM provider and establish
  fresh matched original-source 0.8B/2B/27B baselines. Use retained V2
  results for engineering direction, not fresh comparisons. Get quick real
  C1/C8 feedback; extend the declared campaign after working results.
~~~~

**Lines 787–792**

~~~~markdown
- [ ] Native row-wise and attention regions: eliminate unnecessary intermediate
  materialization and exact-reference scalar passes within explicitly
  qualified native arithmetic. Compose complete normalization/residual,
  gating, QK/rotary, attention and recurrent regions; reuse mature native
  kernels/providers where capability and model geometry fit. Keep GDN
  solve/products and source/reference implementations.
~~~~

**Lines 793–797**

~~~~markdown
- [ ] Use rocBLAS/hipBLASLt for dense products, appropriate CK, MIOpen and
  mature attention/flash-decode kernels, model-specific kernels where needed.
  Select measured GEMV/GEMM, quantized and batch shapes, not one blanket
  provider choice. Native command capture and larger execution regions
  amortize host/actor dispatch, without requiring MIGraphX.
~~~~

**Lines 798–801**

~~~~markdown
- [ ] Improve genuine shared prefill, independent stateful cohort work,
  continuous ragged refill/regroup, paged/hybrid KV and prefix/COW.
  Device-resident sampling/feedback, overlap, fair scheduling, native
  graph execution where material. Qualify cancellation and state effects.
~~~~

**Lines 802–806**

~~~~markdown
- [ ] Scale native resident and streamed weights, long context, real
  supported low-bit GGUF/source quantization, KV formats and SSD execution.
  Keep exact/source/native profiles separate; quantify quality against
  reference and compare equivalent precision/workload. Extend family support
  through existing registries and source bindings, not separate model loops.
~~~~

**Lines 807–809**

~~~~markdown
- [ ] Close the matched performance acceptance below on the required feasible
  AMD cells. Do not wait until the entire grid is green to measure and fix
  the dominant gaps. No actual graph compiler integration in this phase.
~~~~

**Lines 822–825**

~~~~markdown
- [ ] Revalidate actual current NVIDIA source paths before performance claims.
  Close deterministic C4/B8 identical-lane divergence, independent populated
  all-state-domain correctness and missing approved active-context/2B
  references. Preserve prior physical acceptance only within its scope.
~~~~

**Lines 826–829**

~~~~markdown
- [ ] Promote the existing owned cuBLAS/cuBLASLt dense adapter into ordinary
  execution through shared geometry and original physical owners. Use native
  tensor-core/CUTLASS and FlashAttention/FlashInfer where they win; preserve
  existing specialized GDN/state and transport alternatives.
~~~~

**Lines 830–835**

~~~~markdown
- [ ] Reuse ROCm phase's mathematical regions and native execution improvements;
  tune CUDA-specific attention/paging, continuous batching, compact
  device-resident sampling, larger captured graphs and dispatch amortization
  only from actual measurements. Keep pre-enqueue qualified misses distinct
  from unknown native submissions. Production selection must be physically
  accepted, not inferred from diagnostic graph/timing support.
~~~~

**Lines 836–839**

~~~~markdown
- [ ] Run matched optimized vLLM through the same public serving campaign on
  approved NVIDIA devices and feasible models/context/concurrency. Extend
  source/GGUF model coverage, native quantization and residency according to
  measured capacity. TensorRT remains optional Phase 6 work.
~~~~

**Lines 848–851**

~~~~markdown
- [ ] Reuse shared typed operations, region/shape contracts and device
  lifetime semantics. Close actual Vulkan model execution, independent
  recurrent/KV state, prefill/decode, cohort and resource safety, then
  required CPU/CUDA/HIP-compatible capabilities where available.
~~~~

**Lines 852–857**

~~~~markdown
- [ ] Evaluate mature ggml Vulkan cooperative-matrix GEMM implementations,
  capability-gated VK_KHR_cooperative_matrix and vendor variants, portable
  compute shaders for unsupported geometries, and applicable optimized
  quantized/attention kernels. Audit license/provenance and actual backend
  layout compatibility before reuse. No fictional universal Vulkan BLAS
  and no mandatory compute-engine fork.
~~~~

**Lines 858–863**

~~~~markdown
- [ ] Compare real model latency, throughput, memory and quality against
  native Vulkan baselines (e.g. llama.cpp/ggml) on the same device/format;
  use matching native/vLLM hardware comparisons only where meaningful.
  No unsupported vLLM Vulkan baseline. Extend Linux and other targets by
  actual capability and package qualification; retain clean backend feature
  isolation. IREE integration is Phase 6.
~~~~

**Lines 887–891**

~~~~markdown
- [ ] MIGraphX on HIP: lower native semantic regions through programmatic
  graph construction, qualify runtime/hardware and external device arguments/
  stream interop, compile/prepare a closed FFN or other dense region, then
  compare native and compiled paths at actual model/serving level. Consider
  whole encoders/embeddings/ASR where source format and model semantics fit.
~~~~

**Lines 892–896**

~~~~markdown
- [ ] TensorRT on CUDA: lower native semantic regions through programmatic
  network construction, same closed-region/whole-model boundary, compatible
  caller-bound tensor addresses/context memory/stream, admitted engine
  compilation/cache and explicit supported shape/profile. Benchmark a real
  region or conventional model, not another complete LLM serving engine.
~~~~

**Lines 897–899**

~~~~markdown
- [ ] IREE: evaluate compiler/HAL interop on supported device targets, in
  particular Vulkan, and compare with the native cooperative-matrix path.
  Verify actual import/export/synchronization before promising zero-copy.
~~~~

**Lines 900–907**

~~~~markdown
- [ ] Optional graph import: authenticate ONNX graphs and external weights;
  establish their typed region, numerical and effect contracts through the
  existing source/preparation boundary. ONNX Runtime executes supported
  conventional models, encoder/vision/audio/embedding graphs, with device
  I/O binding or explicitly budgeted materialization. MIGraphX/TensorRT
  import support is independently qualified; it is not a prerequisite for
  direct native-region lowering. Other graph accelerators/TVM only when a
  concrete workload justifies them.
~~~~

**Lines 908–912**

~~~~markdown
- [ ] Native/compiled selection happens at admitted preparation, with source,
  numerical/quality, layout, shapes, device/runtime, actual costs and state
  compatibility explicit. Bound compilation/autotune/artifact caches and
  retain useful alternatives. Require actual external-resource lifetime and
  completion; a failed submission never falls back by hiding its failure.
~~~~

**Lines 913–915**

~~~~markdown
- [ ] Preserve one engine, state authority, resource fabric and model/pipeline
  interfaces. Whole-model opaque managed remote APIs are TaskService adapters,
  not native kernels or proof of local physical execution.
~~~~


### [.agents/plans/TODO.md](../plans/TODO.md) — 11 unresolved markers

**Lines 713–713**

~~~~markdown
- [ ] P1 serving-and-speculation — @pvl — l5 CUDA `.8B`/`2B`, x2 ROCm `.8B`/`2B`/`27B`: source-real correctness/paged/prefix/fused 2/4/8-row cohorts accepted; multi-token prefill + autoregressive refill/regroup now fast under matched opt1. Public CUDA serving passes4K/8K/16K on both small models,0.8B32K has one full pass, and paged2B32K passes32766 tokens in224.80s. Same32K-capacity1024-token A/B is283.22s contiguous-streamed vs28.56s paged-resident (~9.9x). Generic cold-start smart config now jointly resolves Qwen `state_policy` + `cohort_width` from fresh runtime admission: Exclusive=ignore user knobs, Enabled(default)=user values override Auto per knob, Disabled=no Auto; score=resident weights > wider 8/4-row cohort > Contiguous tie. Default2B32K/1024 Auto-selects paged and passes in26.51s exact. Existing `active=8` SDK with cohort width unspecified Auto-selects8 for both Contiguous/user-Paged halves and publishes `{8}` at135.979/133.004 tok/s exact. Nix-realized vLLM0.33.0/cu130 correctness now passes exact local0.8B/2B on L5 without Podman; the frozen wheel graph is store-realized and an optimized CUDA-graph smoke passes, so the retained concurrency=1/4/8/16/32/64 paired matrix is next on L5 before the exact-commit X2 ROCm campaign. Independent populated all-state-domain closure remains open; X2 ROCm remains environment-gated — [details](serving-and-speculation/TODO.md)
~~~~

**Lines 714–714**

~~~~markdown
- [~] P1 numerics-and-context — @pvl — exact prefill 47 + decode 48–50, lifecycle/capacity and final gate pass; active-context/2B reference inputs missing — [details](numerics-and-context/TODO.md)
~~~~

**Lines 715–715**

~~~~markdown
- [~] P1 memory-and-streaming — @pvl — 32K storage/execution policy is materially closed:0.8B32K has one full public-SDK pass;2B32K passes the full32766-token retained cell with Paged64/64, and canonical default startup Auto-selects paged when contiguous would stream. Final CUDA admission independently clamps provider capacity to original-actor live free VRAM while preserving configured headroom. CUDA-forward source-page transfer now retains the verified `TensorWindow` itself through the existing checkpoint and uses the original actor's bounded owned-window H2D directly, removing the superseded pinned/Bytes copy chain. Matched 2B/32K prompt48 streamed feedback is38.46s vs39.36s at `b839920`; cache-promotion41.00s, moved-pinned44.88s and direct-SSD/cache-bypass41.77s variants were rejected. A same-tree one-page CUDA retirement-ring Qwen A/B is rejected at39.281s vs38.323s serial (~2.5% slower); production remains serial at that boundary and default smart config avoids it when paged state enables resident weights. The model-neutral primitive remains dormant/reusable across backends: CUDA `OwnedCudaEvent` and HIP `OwnedHipTransferRetirement` implement `RetirementMark`, with zero-cost CUDA/HIP ring aliases over the shared `TransferRetirementRing`. Shared completed-root reuse, allocation-free common-frontier reconciliation, exact partial-tail paged COW ownership and canonical physical allocation+range identity are active; CUDA-forward prepared sets, streamed-head transfer binding and production HIP resident-text workspace/root binding use canonical ledger identity. Remaining work is additional real identity consumers plus cache-aware direct/async IO and broader authenticated streaming tiers — [details](memory-and-streaming/TODO.md)
~~~~

**Lines 716–716**

~~~~markdown
- [~] P1 cpu-gpu-optimization — @pvl — forward kernel/transport checks pass; exact whole-model parity before measured tuning and the matched benchmark campaign — [details](cpu-gpu-optimization/TODO.md)
~~~~

**Lines 717–717**

~~~~markdown
- [~] P1 pipeline-composition — @pvl — recursive primitives/pipelines/profiles; shared execution adapters active, reuse current adapters in Fabric; Qwen expansion deferred — [details](pipeline-composition/TODO.md)
~~~~

**Lines 718–718**

~~~~markdown
- [~] P1 cli-device-routing — @pvl — unified run/config/source/device selection; preserve behavior through Fabric SDK/CLI migration — [details](cli-device-routing/TODO.md)
~~~~

**Lines 719–719**

~~~~markdown
- [~] P2 model-execution — @pvl — current Qwen CUDA forward owns source-real fused 2/4/8-row contiguous+paged transport, complete-domain prefix reuse, event-backed/multi-flight primitives, retained stateless + multi-token attention/GDN prefill, reduced 4/8-row GDN layout selection and bounded paired-page KV gather with fused inactive-tail fill. Latest matched opt1 short-wave is135.236 tok/s contiguous/132.949 paged (~1.7% wall-time gap); current68-token refill is169.837/160.919 tok/s. Rejected A/B mechanics are retained dormant rather than deleted: lane regroup, copy+tail-zero,17 BF16 GEMM strategies (including restored terms64 shared-row, paired-boundary and K64 weight-cache variants), recurrent tiled transpose and GDN occupancy variants, with zero production consumers. The restoration sweep also repairs default/qualification/CUDA-native test cfg drift; the largest native-qualification lane is external-SIGKILL incomplete. Public graph serving deliberately remains Native. Large same-contract CUDA numerical acceptance still depends on approved references; populated small HIP and resident27B work remain separate. X2 is shared: preserve other owners and verify only ZeroGravity's incremental resource delta retires — [details](model-execution/TODO.md)
~~~~

**Lines 720–720**

~~~~markdown
- [~] P2 model-acquisition — @pvl — [details](model-acquisition/TODO.md)
~~~~

**Lines 724–724**

~~~~markdown
- [ ] multimodal — [details](multimodal/TODO.md)
~~~~

**Lines 725–725**

~~~~markdown
- [ ] distributed-execution — [details](distributed-execution/TODO.md)
~~~~

**Lines 726–726**

~~~~markdown
- [ ] platform-and-release — [details](platform-and-release/TODO.md)
~~~~


### [.agents/plans/cli-device-routing/TODO.md](../plans/cli-device-routing/TODO.md) — 65 unresolved markers

**Lines 466–467**

~~~~markdown
- [ ] UX-P4: complete ergonomics, actual PTY and accepted real cached
  resident/transfer/config regression, isolated workspace gate.
~~~~

**Lines 499–500**

~~~~markdown
  - [ ] UX-P-M2 optional exact per-request/phase and detailed layer
    observations; keep profiling explicit and capability-bound.
~~~~

**Lines 501–501**

~~~~markdown
  - [ ] P4B full simultaneous owner regressions and isolated gate.
~~~~

**Lines 510–512**

~~~~markdown
- [ ] Broader catalog presets, conversation history, dedicated graph
  visualization and unsupported routes remain explicit future scope.
  Existing direct text/JSON structural inspection is accepted.
~~~~

**Lines 513–517**

~~~~markdown
- [ ] Extend advanced direct provider forced interactive/JSONL presentation
  and aggregate component download control without changing independently
  published receipts; existing direct provider progress and legacy sequential
  component executor remain valid. No general concurrent download scheduler,
  persistent event history or automatic partial deletion introduced.
~~~~

**Lines 523–550**

~~~~markdown
- [~] Shared pre-Client transactional admission: accepted `f4746c23` adopts
  the original envelope before HIP source Client/runtime bootstrap; actual
  Qwen refusal/retry passes without leaked reservation or fencing.
  9ae6dd1e/e06213a5 now share exact context-free CUDA PCI and original
  configured provider reservation before Client on ordinary/pinned Linux
  CUDA routes, actual identity adoption and original live downward clamp
  with unused cold credit released. Real Qwen HF peer refusal (no new
  observed compute owner)/retry, pinned live fit and retirement, HF C1,
  GGUF pipeline/selective unload and HIP V2 C1/C8 pass. Existing Vulkan
  diagnostic attachment compiles. Logical pre-admission is conservative;
  current free memory still observed on actor after bootstrap when
  necessary. cbd2c119/197c2e26 add original shared host-backing constraint,
  immutable scoped caps, latest visible cgroup-v2 hard constraint, shared
  cold fit and real HIP local VRAM/driver intersection (no host/GTT sum).
  Original host lifetime upper remains logical; check actual next cold
  allocations. a6548863 extends original preadmission to ordinary Linux HIP
  and guarded native HIP; guarded exact-adapter Vulkan intersects local VRAM
  total/current use before original registration. Equal declared physical
  roles return no inter-domain transfer; distinct domains explicitly refuse.
  3b5a520d/fc38db03/e1decff5 close bounded explicit named Linux Vulkan
  registry reuse and physical admission: actual peer refusal/raced/warm
  reuse, independent envelope retirement and portable healthy/unknown.
  14bbbd48/5d952364/5a7535e7 qualify actual coherent mapped Vulkan
  producer on the original range and charge. No implicit zero-copy from
  colocated placement. Non-Linux, general host-visible GPU backing
  aliasing and OS hard-memory guarantees remain open.
  Minimum shared foundation exit review in the full plan; shared CLI
  presentation (UX-1) now accepted above, preserving the original runtime/selector.
~~~~

**Lines 554–555**

~~~~markdown
- [ ] Recheck live source, ownership and accepted model/backend evidence before
  implementation; preserve root README, other plans and active implementation.
~~~~

**Lines 559–562**

~~~~markdown
- [ ] Accept local paths, catalog/model names, `hf:`, `gh:`, `http:` and `https:`
  through one shared source selector and the existing acquisition/cache identity
  contracts. Recognized source schemes take precedence over local YAML suffix
  classification; only a local `.yaml`/`.yml` selector implies a run document.
~~~~

**Lines 563–566**

~~~~markdown
- [ ] Resolve already acquired immutable exact sources, reject ambiguous variants
  and preserve explicit acquisition guidance when absent. URI parsing/resolution
  is not inference qualification. Reuse provider parsing/source identity rather
  than a second downloader or format-specific run loop; dry-run stays read-only.
~~~~

**Lines 567–569**

~~~~markdown
- [ ] Every new CLI semantic option has a typed v2 field. Unit1 implementation
  and tests precede ROOT guarded fresh-mtime merge; final assembled gate remains
  required. Future split/NPU/remote units retain separate qualification.
~~~~

**Lines 577–578**

~~~~markdown
- [ ] Normalize `run`, backend aliases and common task flags into one library
  request; preserve compatible existing invocations and actionable diagnostics.
~~~~

**Lines 579–581**

~~~~markdown
- [ ] Extend the canonical pipeline document/compiler to proposed v2 with one
  model table, explicit model/pipeline entrypoint and typed run policy/task/
  diagnostics; do not introduce a separate config interpreter or runtime.
~~~~

**Lines 582–584**

~~~~markdown
- [ ] Implement unambiguous positional model/config and explicit selectors;
  selector-free run loads only absolute-XDG/HOME `abird/zerogravity/config.yaml`.
  Missing/unreadable/invalid config fails without fallback or initialization.
~~~~

**Lines 585–587**

~~~~markdown
- [ ] Implement CLI > selected config > built-in resolution with per-field
  provenance, origin-correct paths, whole-list replacement and explicit boolean/
  zero/empty semantics. Preserve alias equivalence and final constraint checks.
~~~~

**Lines 588–589**

~~~~markdown
- [ ] Migrate model metadata to `--model-config`; cover every existing task,
  tokenizer, teacher-ID, prefill and audit option with typed config equivalents.
~~~~

**Lines 590–592**

~~~~markdown
- [ ] Retain bounded strict YAML/graph validation; check the complete merged
  document before effects. Declared routes remain inert under run; graph and
  recursive execution still need their independently qualified implementation.
~~~~

**Lines 593–594**

~~~~markdown
- [ ] Resolve exact local source and bounded task/model requirements through
  existing adapters; catalog/metadata status grants no executable capability.
~~~~

**Lines 595–596**

~~~~markdown
- [ ] Add read-only inventory and stable physical identity resolution; derive
  actual vendor/backend support and deduplicate cross-backend aliases.
~~~~

**Lines 597–598**

~~~~markdown
- [ ] Implement strict ordered allowlists, CPU-terminal semantics, local-only
  auto and all-neural-stage forced-backend checks before preparation.
~~~~

**Lines 599–601**

~~~~markdown
- [ ] Implement current-memory observation with explicit provenance/age/unknown
  status, OS/cgroup constraints and host/UMA domain relationships. Do not reuse
  `max_memory` as free memory or mutate established operator ceilings.
~~~~

**Lines 602–604**

~~~~markdown
- [ ] Resolve a safe default operator-ceiling/bootstrap policy once for a new
  process before context effects, so bare `run` needs no manual resource caps;
  preserve established domains and benchmark/serving controls.
~~~~

**Lines 605–607**

~~~~markdown
- [ ] Integrate complete cold/prefill/decode/native/compiler/state/output costs
  with original ledger snapshots and authentic existing backing/reuse. Close or
  expose pre-admission context-bootstrap gaps; unknown costs cannot claim fit.
~~~~

**Lines 608–609**

~~~~markdown
- [ ] Implement configurable headroom, all-GPU full-fit scan and qualified CPU
  fallback; explain missing split support when a model would otherwise spill.
~~~~

**Lines 610–611**

~~~~markdown
- [ ] Revalidate and admit through original owners; bound safe startup replanning,
  preserve unknown-work fences and retain all submitted owners.
~~~~

**Lines 612–613**

~~~~markdown
- [ ] Provide `devices list`, `--dry-run`, `--explain` and structured refusals
  without hidden execution/network/cache mutation during diagnostics.
~~~~

**Lines 614–615**

~~~~markdown
- [ ] Validate the assembled unit on approved source/hardware/backend tuples;
  document actual supported tuples and unavailable current-resource probes.
~~~~

**Lines 624–625**

~~~~markdown
- [ ] Define source-specific indivisible groups and embedding/head/shared-weight/
  MTP ownership; produce one complete feasible placement before any preparation.
~~~~

**Lines 626–627**

~~~~markdown
- [ ] Extend original arbiter with all-or-none multi-domain admission if needed;
  collapse aliases and UMA domains, preserve prepaid-envelope/sublease rules.
~~~~

**Lines 628–630**

~~~~markdown
- [ ] Integrate original state shards, typed activations and actual event/reader
  transfers into the existing prepared physical plan and fair workers; join all
  Ready events before the original SessionAuthority publishes one logical root.
~~~~

**Lines 631–632**

~~~~markdown
- [ ] Prove same-backend multi-GPU and GPU↔CPU compositions separately before
  enabling each; qualify mixed CUDA/ROCm/Vulkan only with its own evidence.
~~~~

**Lines 633–634**

~~~~markdown
- [ ] Verify full-logit/complete-state following-token parity, cancellation,
  partial preparation, transfer failure, unknown work and complete retirement.
~~~~

**Lines 635–636**

~~~~markdown
- [ ] Compare full CPU and each feasible single/split route with complete cold/
  warm request costs; show capacity enabling separately from speedup.
~~~~

**Lines 642–643**

~~~~markdown
- [ ] Introduce semantic stage-role constraints without model-family CLI routing
  branches; preserve full-run forced-backend policy and common allowlist.
~~~~

**Lines 644–645**

~~~~markdown
- [ ] Qualify actual trained draft/MTP source and recursive state semantics;
  distinguish native MTP depth from supported proposal length.
~~~~

**Lines 646–648**

~~~~markdown
- [ ] Add an NPU provider only through existing region/state/lifetime boundaries;
  prove end-to-end benefit versus available GPU draft and no-speculation routes,
  including transfer/verification/rejection/shared-domain costs.
~~~~

**Lines 649–650**

~~~~markdown
- [ ] Resolve explicitly enrolled peer aliases to authenticated identities and
  capability/topology epochs; standalone auto performs no remote discovery.
~~~~

**Lines 651–652**

~~~~markdown
- [ ] Coordinate bounded all-or-none holds through original per-host ledgers;
  implement authenticated transfers and exact state shard publication/fencing.
~~~~

**Lines 653–654**

~~~~markdown
- [ ] Qualify first two-host contiguous layer/local-state composition on approved
  sources/hosts; report directional network/copy/crypto/queue costs and loss.
~~~~

**Lines 655–656**

~~~~markdown
- [ ] Keep remote inference APIs that hide region/state contracts outside native
  device routing; no SSH/shell or implicit artifact downloads for placement.
~~~~

**Lines 664–666**

~~~~markdown
- [ ] **Syntax equivalence:** canonical/alias/repeated flags normalize identically;
  conflicting backend, mixed `auto`, empty device, unknown host/ID and CPU before
  GPU refuse. Unsupported backend is reported without model preparation.
~~~~

**Lines 667–669**

~~~~markdown
- [ ] **Default config:** present/missing/unreadable/invalid default; absolute,
  relative and unset XDG base; valid/absent HOME fallback; no cwd/parent search.
  Explicit model/config/graph skips default overlay; help/version do not load it.
~~~~

**Lines 670–673**

~~~~markdown
- [ ] **Config discrimination:** YAML suffix, extensionless explicit --config,
  --model, odd model filename, malformed/unsupported schema, multiple positional
  selectors and old model-JSON --config produce the specified result; no model/
  config fallback, automatic pull, initialization or side effects.
~~~~

**Lines 674–677**

~~~~markdown
- [ ] **Config/CLI parity:** direct YAML and flags yield equal typed requests;
  graph YAML and declarations share one instance table/compiler. Source metadata,
  tokenizer, prompt/context/output/raw/teacher/prefill/audit/numerical/speculation/
  role/device/diagnostic fields and semantic validation round-trip equally.
~~~~

**Lines 678–681**

~~~~markdown
- [ ] **Precedence/provenance:** CLI scalar wins, alias equals canonical backend,
  default only fills absent values; false/zero/empty/auto are preserved and
  validated. Repeated CLI devices replace config list as a whole; graph list
  replacement/clear detects dangling references and duplicate IDs.
~~~~

**Lines 682–684**

~~~~markdown
- [ ] **Path bases:** config-origin source/sidecar/audit paths use selected config
  directory; CLI replacements use invocation cwd. Moving cwd, config symlinks and
  same relative spelling with different origins cannot silently retarget fields.
~~~~

**Lines 685–688**

~~~~markdown
- [ ] **Config safety/entrypoint:** unknown/duplicate fields, expansion/depth/file
  limits, custom tags/includes/interpolation, missing/ambiguous entrypoint,
  contradictory final backend/device policy and legacy model-config spelling
  refuse before execution. Selected graph runs once; declared routes do not bind.
~~~~

**Lines 689–691**

~~~~markdown
- [ ] **Identity:** CUDA/Vulkan aliases of one card count once; AMD discrete and
  NVIDIA integrated classifications cannot invert vendor routing. Snapshot
  indices never survive reorder/restart without physical revalidation.
~~~~

**Lines 692–694**

~~~~markdown
- [ ] **Forced scope:** CUDA+AMD rejects before allocation/compiler activity;
  forced GPU+CPU spill or incompatible NPU draft rejects; forced CPU runs every
  requested neural stage on CPU or refuses a missing capability.
~~~~

**Lines 695–697**

~~~~markdown
- [ ] **Full fit:** both vendors fit→NVIDIA; NVIDIA insufficient/occupied and AMD
  fits→AMD; later listed GPU full fit precedes first-card split; GPU-only explicit
  list cannot reach CPU. NVIDIA unavailable/fenced is a precise exclusion.
~~~~

**Lines 698–700**

~~~~markdown
- [ ] **CPU fallback:** no GPUs, missing GPU build, unqualified GPU model and no
  qualified split each reach CPU only if allowed, qualified and currently fitting.
  An unqualified model everywhere gives no invented inference route.
~~~~

**Lines 701–703**

~~~~markdown
- [ ] **Current pressure:** large total VRAM with low free bytes does not fit;
  cross-process occupancy, OS/cgroup pressure, missing/stale counters and
  fragmentation produce conservative proposals and bounded allocation failures.
~~~~

**Lines 704–706**

~~~~markdown
- [ ] **Accounting:** original leases plus actual backing plus unbacked promises;
  prepaid envelope and native partition; idle retained pools; authentic reuse;
  unknown/captured readers. No duplicate charge/free credit or cap enlargement.
~~~~

**Lines 707–708**

~~~~markdown
- [ ] **UMA:** CPU/GPU/NPU alias one memory domain; cross-device views/transfers
  retain proper fences and charge physical backing once. Headroom applied once.
~~~~

**Lines 709–711**

~~~~markdown
- [ ] **Complete peaks:** weights fit but prefill/KV/output/compiler/native/capture/
  draft costs do not; reject. Disjoint phase maxima are not blindly added; actual
  overlapped lifetimes, COW, transfer and old/new state are not omitted.
~~~~

**Lines 712–715**

~~~~markdown
- [ ] **Headroom:** percentage/bytes/default/floor/overflow validation; resolved
  amount shown; reducing margin never raises a physical/operator ceiling or
  waives source/shape/native qualification. Bare-run bootstrap records a safe
  default once; later live-free changes never resize the registered ceiling.
~~~~

**Lines 716–718**

~~~~markdown
- [ ] **TOCTOU:** external allocation and concurrent internal admission between
  proposal/admission/preparation; all-or-none holds and known-safe rollback;
  at most one fresh replan. Dry-run and cached plan never grant reservation.
~~~~

**Lines 719–721**

~~~~markdown
- [ ] **Unsafe preparation/flight:** unknown module load/unload, transfer/capture/
  launch retains/fences original owners; no alternate-backend replay, partial
  root/output publication or reuse based on cancellation/timeout alone.
~~~~

**Lines 722–724**

~~~~markdown
- [ ] **Split feasibility:** largest legal prefix leaves a feasible remainder;
  indivisible tail/tied embedding/head cases; GPU→GPU→CPU exact state and
  activation transfer. Unqualified mixed numerics refuse rather than retag.
~~~~

**Lines 725–727**

~~~~markdown
- [ ] **Runtime equivalence:** real full affected logits/tensors, every required
  state domain, natural following-token continuation and task outputs match the
  selected numerical contract for each admitted source/profile/composition.
~~~~

**Lines 728–730**

~~~~markdown
- [ ] **Fairness/lifetime:** overlapping requests/models preserve independent
  roots, original fair progress and complete reader retirement under pressure;
  successful cleanup cannot hide an earlier semantic failure.
~~~~

**Lines 731–734**

~~~~markdown
- [ ] **Diagnostics:** list/dry-run leave model, GPU context/compiler, resource
  ledger, cache and network unchanged; report unknown rather than probe by
  execution. Explain identifies each rejection and final route without corrupting
  output or exposing prompts/secrets.
~~~~

**Lines 735–739**

~~~~markdown
- [ ] **NPU role, future:** qualified NPU still loses when transfer/verification/
  rejection/queue costs dominate; wins only with applicable full-request benefit.
  Stale evidence rejects automatic speed claim; default auto selects ordinary
  decode without proven benefit, optional draft-device does not force drafting,
  and required MTP/draft/source features are not omitted.
~~~~

**Lines 740–742**

~~~~markdown
- [ ] **Remote, future:** unknown enrollment, identity/epoch mismatch, denied
  source/data grant, partial holds, slow/relayed link, late Ready, lost ACK and
  uncertain DMA preserve original authority/leases; no implied discovery or HA.
~~~~

**Lines 743–745**

~~~~markdown
- [ ] **Performance:** authorized paired whole-request comparisons, cold/hot
  preparation, TTFT/ITL and total latency, memory/IO/transfers and fair load;
  compare against CPU/single-GPU baseline and include slower-split negative cases.
~~~~

**Lines 746–748**

~~~~markdown
- [ ] **Final assembled gate:** after applicable unit qualification, run
  `CARGO_BUILD_JOBS=2 RAYON_NUM_THREADS=4 python3 scripts/native-gpu.py -- bash scripts/check.sh`.
  This design-only draft does not run the gate or any GPU/model tests.
~~~~


### [.agents/plans/cli-ux/LIVE_MODEL_PERFORMANCE.md](../plans/cli-ux/LIVE_MODEL_PERFORMANCE.md) — 2 unresolved markers

**Lines 127–139**

~~~~markdown
- [ ] UX-P-M2 optional richer request and phase inspection after M1:
  exact current response/request identity, per-request run progress,
  prompt/prefill/queue and decode timings where original boundaries
  provide them; optional bounded recent request samples/latency
  distribution only if justified, no unbounded history. Detailed
  layer/device topology across actual adapters including future mixed
  placement, layer roles/types from original model geometry, reported
  memory by owner class. Per-layer time is explicitly profiled, never
  estimated from model TPS or enabled via per-layer GPU synchronization
  by default. Reuse authoritative provider completion/timing contracts;
  qualify observer overhead and physical execution on relevant hardware.
  Keep unreported observations explicit. No requirement to invent
  unsupported split/offload or per-layer telemetry to fill a dashboard.
~~~~

**Lines 140–145**

~~~~markdown
- [ ] UX-P4B comprehensive isolated workspace gate on assembled source,
  actual cached resident and simultaneous transfer/config qualification,
  default full production link/launch; hardware placement qualification
  on original existing routes before claim. Preserve prior GPU lint debt
  and feature boundaries. No performance comparison claim from these UX
  observations. Public serving remains after UX-P exit.
~~~~


### [.agents/plans/cli-ux/TUI_POLISH_AND_MOUSE_INTERACTION.md](../plans/cli-ux/TUI_POLISH_AND_MOUSE_INTERACTION.md) — 2 unresolved markers

**Lines 591–597**

~~~~markdown
- [ ] **UX-P4 — finish ergonomics and acceptance:** Unicode wrapping/caret,
  scrollbar/follow/selection edge cases, small viewports, accessibility,
  focus order, no-color/ANSI, busy/disabled/stale/error/empty states,
  cross-client theme revision, browser retirement and terminal cleanup.
  Real cached resident inference/serve/download/config simultaneous
  regression through mouse and keyboard, full offline isolated gate,
  updated CLI guide/help and owner handoff. No physical GPU perf claim.
~~~~

**Lines 677–682**

~~~~markdown
  - [ ] P4B full offline isolated workspace gate and final real
    resident/transfer/config acceptance on assembled source. Review
    remaining compact affordances and rendered terminal legibility;
    no golden matrix. Native Auto remains conservative hint fallback,
    active probe deferred unless separately approved. No public
    serving or performance acceptance before P4 exit.
~~~~


### [.agents/plans/cpu-gpu-optimization/BACKEND_PARITY.md](../plans/cpu-gpu-optimization/BACKEND_PARITY.md) — 5 unresolved markers

**Lines 108–129**

~~~~markdown
- [ ] Portable arithmetic assembled integration: existing recorded typed embedding→ordinaryRMS→
  projection slice, same ResidentProgram traversal/service/flight, explicit
  TorchCpuBf16Avx2MixedSourceV1 contract. Reuse the existing64lane projection
  shader; preserve every RMS materialization/reduction and check actual bits.
  B1/H1024or2048/W<=32MiB, provider64MiB/physical256MiB/host128MiB diagnostic
  bounds; no model/state-root/serving grant. CUDA SM86 rsqrt/native profiles
  remain separately unavailable. ROOT reviewed runtime/reference source;
  strictHIP/Vulkan/CUDA and focused guards pass. Both authentic pinned public
  HF/Torch captures plus independent validators pass all11 boundaries. Actual
  ten-process HIP/Vulkan cohort now PASS: both healthy H1024/H2048 exact public
  boundaries plus private CPU stages/packed RMS/FP32 projection,3nodes/28
  dispatches/17readbacks/callback1/finalresident-event-physical0. RMS/projection
  unknown cuts on each provider and both shared-PCI directions retain actual
  source/234664B/pending1/unpromoted outputs/64or128MiB envelopes;0D2H/Ready/
  callback. ROOT independently reproduced exact collector/outcome/source/
  binary/reference/cap/test/group/reap/EOF checks; finaljobsnone/VRAMbaseline
  observed, notpeak. Reviewed21paths merged MAIN with freshmtimes/backups;
  ISO held for CUDA diagnostic. Next combined fullgate remains pending.
  Finite synthetic3node CPU-profile proof only; no broad32MiBgeometry, learned
  model, CUDA-specific profile, serving or performance grant. Full model parity
  needs its actual backend/reference contract, not a provider retag.
  Actual arithmetic results.
~~~~

**Lines 130–141**

~~~~markdown
- [ ] Whole-model HF and serving profiles on each backend, with full vocabulary,
  every state domain, private natural feedback and independent concurrent roots.
  Gemma E2B whole35 bridge design
  maps direct gamma1536/grouped256/512, actual call-M/PV contraction, source
  row/full-K tiles and numeric-zero-tie262144 greedy into the original engine.
  Reviewed13path source/call/observation/tile contract checkpoint merged MAIN;
  core183/Gemma32/Qwen79/strictClippy/fmt pass. Original whole35 traversal,
  authentic gamma/unit-V roles, callM/projection/PV geometry and <=1MiB frame
  recipes preserve old profiles. Inventory prices one4MiB call; every other
  simultaneous call/frame/native owner still needs actual admission. Lowerers
  and full model/state/physical qualification continue in isolated checkouts
  from current MAIN+portable21. No second traversal/IR or monolithic fit grant.
~~~~

**Lines 142–148**

~~~~markdown
- [ ] Before any large Vulkan matrix consumer, use the device's effective buffer
  and storage-binding limits. CubeCL's Vulkan index64 path can expose heap size
  as max_page_size; that is not a single-binding limit. Existing actual
  transport<=3MiB and arithmetic<=32MiB callers have independent small bounds.
  Gemma E2B PLE table262144x8960 BF16=4697620480B exceeds x2's effective
  0xfffffffc buffer/0xffffffff storage-binding bounds; use authenticated
  selected-row reads or checked paging. No monolithic admission/fit claim.
~~~~

**Lines 149–153**

~~~~markdown
- [ ] Product execution/declarative packaging must use a mutually compatible
  host-libc/compiler/backend SDK closure. Preserve the actual per-job loader
  requirement in qualification; successful prefixed tests do not qualify a
  differently launched run-rocm/serving binary. Avoid global environment repair
  or repository-external configuration changes without applicable authorization.
~~~~

**Lines 154–157**

~~~~markdown
- [ ] Matched performance after correctness. Existing owner order remains
  .8BC1+C4 superior → C8 superior →2B; Qwen27B/Gemma integrate independently,
  with their GPU qualification/benchmarks on pvl-x2 HIP. No timing promotion
  from compilation, missing reference runtime or an unmatched source/codec.
~~~~


### [.agents/plans/cpu-gpu-optimization/TODO.md](../plans/cpu-gpu-optimization/TODO.md) — 67 unresolved markers

**Lines 36–38**

~~~~markdown
- [ ] After correctness: optimize CUDA `.8B`/`2B` on l5 and ROCm `.8B`/`2B`/`27B`
  on x2 across fitted active contexts and C1/C4/C8/C16/C32/C64. High concurrency
  belongs on sufficiently capable hardware; preserve actual active/queued widths.
~~~~

**Lines 39–42**

~~~~markdown
- [ ] Beat the matched, pinned vLLM baseline in every required feasible run;
  diagnose losses, retain all samples and separate throughput/latency metrics.
  Prefer simple shared-layer fixes and measured hot-path changes. Finite source
  diagnostics and old timings are not current ordinary-serving performance.
~~~~

**Lines 43–44**

~~~~markdown
- [ ] x2 compute/memory/storage use is authorized without routine requests;
  do not relax numerical/source/ownership contracts to obtain a timing win.
~~~~

**Lines 61–65**

~~~~markdown
- [ ] Run current-source `.8B`/`2B` model, cached, fault, shutdown and fitted-context
  qualification plus final full gate. `.8B` native/reference inputs were recovered
  offline on 29 Oct and fresh phase-1 capture passes with the existing 8 GiB
  reference host profile. Ordinary cuBLAS initialization now refuses a private
  driver export-table call before model math; preserve allocator ownership.
~~~~

**Lines 66–69**

~~~~markdown
- [ ] Run the one serial campaign specified in
  `../serving-and-speculation/BENCHMARK_CAMPAIGN_2026-10-27.md`. Preserve the
  historical `2B` C16 loss and all cap/resource asymmetries; no new benchmark
  was executed in this checkpoint.
~~~~

**Lines 73–76**

~~~~markdown
- [~] [Shared vLLM/llama.cpp correctness policy](../numerics-and-context/CORRECTNESS_POLICY.md)
  across all models; .8B initial/cached47–50, then genuine long contexts and
  CPU/CUDA/ROCm/Vulkan. All agents GPT-6.1 Sol xhigh. Earlier B4/C16/27B order
  below is historical; no new performance or27B capture before correctness.
~~~~

**Lines 77–80**

~~~~markdown
- [ ] Preserve case5 FAIL101 and quarantine. Original default vLLM N1/B4 also
  has the same six QKVZ BF16 differences at frontier47 with identical inputs,
  input RMS and48 prior planes. Width-specific original agreement and model
  accuracy are distinct; no arithmetic substitution or failed-run relabel.
~~~~

**Lines 84–88**

~~~~markdown
- [ ] Current B4 → fresh optimized matched C16 (.8B +2B) → x2 original27B,
  latest owner order; no C2. Base `d0fff261`, all246 signatures verified;
  README unchanged. All current lanes: GPT-6.1 Sol xhigh, owner override.
  Owned run.
  Final evidence index.
~~~~

**Lines 271–280**

~~~~markdown
- [~] User extends both-model sequence to C16, thenC32 andC64; NO C2. Implement
  genuine populations and honest active8/B4 observations. Latest user direction:
  resume all parallel tracks now. The78 signed logical commits are complete;
  C16 correction, Qwen27B, Gemma4, Vulkan, HIP and shared queue have independent
  Pi lanes. ROOT coordinates builds/physical ownership and quiet timings.
  Default pending count unlimited, optional configured cap, actual HOST
  admission unchanged. C32/C64 follow queue acceptance. C16 correction and
  grouped controls run separately from product source; old drafts preserved.
  Audit/store latest and best verified matching historical C1/C4/C8 plus new
  bands, with source/profile and stock arm from SAMEattempt, no cherry-picking.
~~~~

**Lines 285–291**

~~~~markdown
- [~] C16 both executed on frozen21. Clean2B: ZG78.7859tok/s,1.62466s;
  vLLM95.9974tok/s,1.33337s. Each16offers/16completed/128tokens/0errors,
  original semantic/source/owner closure healthy, actual active8/B4 and compiler0.
  Rate refusal is a measured performance loss, not numerical/cleanup failure.
  First.8B timing overlaps audit scan; one repeat105.7331vs96.6544tok/s has an
  extra sampler and remains diagnostic, not quiet/best evidence. No third rerun.
  Scope review.
~~~~

**Lines 511–517**

~~~~markdown
- [ ] Shared compiler lifetime/readiness: fix checked all-exit raw NVRTC cleanup,
  retain/fence uncertain destruction and partial module loads; establish actual
  context-lifetime host/device ownership before eager exact-recipe preparation.
  No global DryRun, dummy roots or unpriced cache. Existing model-local128MiB
  loader is host scratch, not a persistent CUDA context charge. Numeric ceilings/
  complete fit remain unresolved; Sol HIGH implements the contained foundation.
  Astra architecture.
~~~~

**Lines 518–522**

~~~~markdown
- [ ] After current C8/refill and source113 qualification: shared prepared matrix/
  compact IO reuse, temporal work, optimized eligible GEMM/attention/GDN, pages/
  full-domain prefix/COW, admitted prefetch and graph replay. Preserve numerical
  profiles; source113 CPU-order dot is not TensorCore/BLAS parity. Family and new
  model gaps are queued in model-execution; no hardware eligibility by name.
~~~~

**Lines 655–667**

~~~~markdown
- [ ] **Parallel implementation, third merge — genuine BF16/HF request batching
  (owner directive,23Oct).** Develop the physical multi-request unit now; merge
  after current storage/residency/architecture fixes, before further N1 microtuning.
  Start with genuine canonical47 B4/B8 initial prefill (188/376 total tokens)
  through SAME independent roots into decode and correct partial-batch fallback;
  extend mixed frontiers through shared engine
  cohort/binding/workspace APIs. Reuse existing root-cohort publication, but
  implement compatible BF16 attention/GDN/projection lowering rather than
  enabling the older GGUF/F32 path. Preserve independent row states, natural
  feedback, fair work, cancellation/quarantine and current resource caps.
  Validate row permutations/full logits/all states and physical/API behavior,
  then matched C1/C4/C8; preserve the C1 fast path. Request batching is distinct
  from operation fusion; no speedup or superiority claim before measurement.
~~~~

**Lines 668–671**

~~~~markdown
- [ ] Lift reusable source identity, bounded immutable reads, prepared bindings,
  actual-flight indices and resident matrix transport into shared components.
  Select qualified implementations using complete host/device live costs;
  a family adapter may restrict mathematics without duplicating common policy.
~~~~

**Lines 672–678**

~~~~markdown
- [ ] Shared resident-backing pool + immutable layout/access metadata + bounded
  transfer/frame retirement, from23Oct parallel source audits. Keep existing
  model traversal/publication and current root/lease freshness; price live,
  idle and unknown owners until actual retirement. Expose minimal common APIs
  through existing prepare/bind/submission/completion objects. CUDA/HIP pinned
  staging and Vulkan queue/mapping need their own physical proof; no inferred
  speedup or zero-copy claim. Follow the coherent units in the linked design.
~~~~

**Lines 679–684**

~~~~markdown
- [ ] Demonstrate superior matched .8B C1 AND C4, then run C8. Only after all
  three wins, optimize/benchmark2B. Keep original quality/source/caps and report
  single-pass uncertainty. Latest prepared-binding release1 observes C1+78.62%/
  C4+26.67%/C8−1.09%; C8 superiority remains open.2B head residency is fixed;
  actual physical/API pass and C1+15.32% with the explicit4608MiB profile.
  Genuine batching precedes wider fresh timing bands; stable superiority is unproved.
~~~~

**Lines 685–686**

~~~~markdown
- [ ] Merge Qwen27B/Gemma contained units as ready in parallel. Their later GPU
  qualification/benchmarks use pvl-x2's AMD gfx1151 through ROCm/HIP.
~~~~

**Lines 687–692**

~~~~markdown
- [ ] Bring Vulkan implementations to CUDA parity independently; AMD uses HIP,
  not an implicit Vulkan/CPU substitute. Preserve one engine/physical service,
  independent roots, fair work and completion/quarantine through each provider.
  Read-only pvl-x2 inventory confirms96GiB GPU memory, existing ROCm7.2.4 and
  RADV/Vulkan. Availability/compilation is not numerical or performance parity.
  [Shared components and backend implementation units](BACKEND_PARITY.md).
~~~~

**Lines 694–703**

~~~~markdown
- [ ] **OPT-01 matched short baseline:** after parity, established same-source
  C1/C2/C4/C8 profiles, original ABBA+BAAB/offered-load controls, real TTFT/true
  token ITL/E2E tails, correct committed/delivered SLO-goodput and failures.
  Separate unprofiled timing from CPU/driver/kernel/copy/IO diagnostics; pin
  actual binary/profile/caps/EOS rather than retagging old GGUF workloads.
  Quick .8B canonical wave8 C1/C4 baseline completed: ZG0.9557/0.9511 output
  tok/s vsstock38.5021/49.6408. Owner23Oct now requires fixing the gap and
  beating vLLM first;2B/full repeated/ragged timing deferred. C8 pair6 is now
  completed above; earlier C8 cancellation is superseded. Preliminary
  single-pass result, no repeated-run confidence or stable-tail claim.
~~~~

**Lines 704–742**

~~~~markdown
- [ ] **OPT-02 reusable bindings:** profile then qualify stable leased row/
  workspace metadata, bounded diff uploads, compact device feedback and removal
  of redundant source/state/readout copies in the original actor. Source/root/
  epoch/physical generations and unknown-completion ownership stay authoritative.
  After the current short baseline, measure the1MiB streamed vocabulary head's
  actual source reads/copies/launches. Qualify admitted larger windows or full
  residency where the complete model/root/workspace inventory fits; retain
  streaming when it does not. Investigate sharing serial cold-geometry storage
  by genuine liveness/alias proofs rather than summing six separate cohorts.
  Both are performance hypotheses, not measured gains or permission to raise caps.
  Current source uses485/970 authenticated1MiB reads+H2D+kernel+same-stream
  checkpoints per .8B/2B Head. Existing .8B original full-table selected-head
  actor offers a reuse path. Raw508,559,360B .8B table addition stays below the
  present4GiB provider/host upper bounds, before deriving new exact overheads;
  original estimate is now superseded by actual .8B validation below. 2B cannot
  inherit that fit claim; evaluate larger tiles and proved workspace reuse.
  Active isolated resident-head unit reuses original full-table actor, keeps
  same arithmetic/source/private ownership and actual4GiB/5GiB caps. Separate
  warmed-process CPU profile now closed healthy: sampled user CPU cycles57.573%
  SHA256,15.202% BF16 finite scanning,10.195% actor spin,7.341% memcmp. Private
  stripped addresses classified from actual instructions. These are CPU cycle
  fractions, not wall/GPU percentages; perf-stat file was empty. Static30.31GiB/
  31,040sync per64forwards is not a measured time percentage.
  Assembled resident .8B passes actual40 natural calls/all5 prompt lengths with
  exact boundaries/fullV/all-state/feedback and pressure/stale/precancel/retirement.
  Actual resident head508,559,360B; source-derived provider upper2,625,406,976B/
  host upper3,397,702,556B fit unchanged4GiB limits. Conv/recurrent/lastFA fault
  cases pass; head-unknown exposed verifier's old streamed-only error recognizer.
  Exactly2 resident strings added behind unchanged error/index254/fault20|21/
  actual-enqueue gates; two affected Head faults await rebuilt fixture. Vendor117
  CPU guards pass. No speedup or complete new-batch acceptance claim yet.
  Follow-up physical Headunknown+cancel PASS; matrix40+all5faults complete.
  Quiet resident CLI2 diagnostic8requests/64forwards exact, owned source/retirement
  closed: forward6.907854s, bind_program3.629112s, enqueue3.021396s,
  complete/readback0.148894s. Nested Head0.005158s,0H2D/0checkpoints/1native per
  call; prepare includes forward, so totals must not be added. Instrumented
  wave7.115s is not an unprofiled performance win. Next substantial host unit
  addresses reusable cold construction and attempt-local state proofs while
  preserving dynamic root/session/generation/fence/cancel/ownership checks.
~~~~

**Lines 743–747**

~~~~markdown
- [ ] **OPT-03 prepared optimized regions:** close existing production graph
  admission in the serving plan; qualify exact producer/consumer fusion,
  prologue/epilogue layouts, finite decode/prefill/verify shape variants and
  source-derived packs. Preserve materialized casts/reductions, aliases and
  effective fused read lifetimes. Fewer graph submissions are not fewer kernels.
~~~~

**Lines 748–752**

~~~~markdown
- [ ] **OPT-04 temporal/fair work:** extend existing native sequence batches
  with actual multi-token dense/GDN prefill and fair mixed-work chunks; retain
  independent state and profile-specific arithmetic. Serving owns scheduling
  integration; memory owns pages/prefix/COW. Re-run short concurrent baselines
  after singles and meaningful combinations before larger contexts.
~~~~

**Lines 753–756**

~~~~markdown
- [ ] **OPT-05 state/residency composition:** full-domain compatible prefix
  intersection, pages/COW, exact source chunk/pack sharing, safe scratch reuse
  and bounded transfer readiness through existing owners. Benchmark cache hits,
  cold arrivals, long prefill versus decode, memory displacement and fair tails.
~~~~

**Lines 757–760**

~~~~markdown
- [ ] **OPT-06 larger dense source:** after long-context prerequisites, original
  27B execution using authenticated layer/row windows, coalescing/batch reuse
  and admitted prefetch. Price every required dense matrix plus activation IO;
  a sparse expert cache cannot turn this checkpoint into a sparse model.
~~~~

**Lines 761–765**

~~~~markdown
- [ ] **OPT-07 optional proposals:** target verifier/all-prefix rollback first,
  then bounded n-gram/Recall/suffix and compatible trained MTP/block drafters.
  Joint finite cost policy considers batch/graph/depth/IO/retained state and
  selects K=0 when it loses. Exact distribution, seeded trace, numeric profile
  and quality changes are distinct gates. No acquisition/training implied.
~~~~

**Lines 766–770**

~~~~markdown
- [ ] **OPT-08 nested profiles:** reconcile primitives→recursive pipelines→
  profiles with the pipeline-composition plan. Compile/in-line/specialize closed
  regions through the original engine; no per-token interpreter/search, new
  serialized IR, model loop, allocator or publisher. Structural design can
  proceed in parallel; new framework implementation must not delay parity.
~~~~

**Lines 771–774**

~~~~markdown
- [ ] **OPT-09 deferred capabilities:** typed sparse selection, grouped experts/
  protected SLRU/hotsets/replicas, deterministic PLE/Engram IO, source-dependent
  shared-KV/liveness, hardware-specific MMA/PDL and derived weight/state codecs.
  Activate only for qualified Qwen/Gemma variants, then other model/device work.
~~~~

**Lines 775–778**

~~~~markdown
- [ ] **OPT-10 promotion:** source/full-logit/all-state/natural-output, actual
  cancel/pressure/unknown-completion/fair serving and complete isolated check.sh;
  repeated complete-request benefit including cold/setup/memory/energy and
  negative cases. Do not multiply external gains or promise universal superiority.
~~~~

**Lines 785–790**

~~~~markdown
- [ ] Qualify [cross-stage residency/fusion](../pipeline-composition/REGIONS_AND_PROFILES.md#resident-values-and-cross-stage-lowering--owner21-oct)
  through the original engine after baseline: compare ordinary, resident-chain,
  graph and fused paths; retain intermediates/feedback on device where eligible,
  report actual H2D/D2H/device traffic, host waits and reasons for required
  bridges. Preserve arithmetic/declared approximation, original commits,
  lifetimes/cancel/fairness and measure complete-task benefit/negative cases.
~~~~

**Lines 791–797**

~~~~markdown
- [ ] Complete the dedicated [ExLlamaV3 investigation](../fabric-vnext/PRIOR_ART_AND_REVALIDATION.md):
  pinned kernel/packing/transform/codec consumers, shape/model/backend limits,
  KV, batching/graphs, CPU offload and partition seams; exact file/dependency
  licenses and static C/Rust ABI versus direct-port disposition. Add independent
  fixtures and map mechanisms into existing plans; existing EXL3/QTIP calibration
  design is not missing and must not be duplicated. Initial scoping is not a
  completed audit or a new entry in the completed416-mechanism inventory.
~~~~

**Lines 798–803**

~~~~markdown
- [ ] Implement native kernel-contract comparison/fixture support, reusing or
  directly porting suitable TensorFold harness/leaf code after license/ABI
  review. Add planted rounding/reduction/NaN/tail/no-write/alias/source defects
  and prove every expected rejection. Invalid hardware configurations refuse
  before launch. Keep broken controls test-only; independent oracles, actual
  model/all-state/physical checks and full isolated check.sh remain required.
~~~~

**Lines 804–807**

~~~~markdown
- [ ] Implement the useful kernels/components identified by these investigations
  through existing providers, using static-native reuse first and faithful
  direct ports when unsuitable for linking. Qualify the implementation as well
  as the harness; an upstream passing test is not local runtime acceptance.
~~~~

**Lines 808–812**

~~~~markdown
- [ ] Specialize first for the NVIDIA cards on both existing machines, then
  prioritize Strix Halo/gfx1151 ahead of all other new hardware and cluster
  work. Keep broad model/device/platform compatibility as the long-term target,
  with portable paths and actual per-target gates; retain the separate gfx90c
  hold. Hardware order does not alter parity→short→long→dense27B model order.
~~~~

**Lines 814–821**

~~~~markdown
- [ ] Bench and hyper-optimize in strict user order:
  Qwen3.5 2B, dense Qwen3.8 27B including SSD/coalesced layer-window streaming;
  Gemma4 E2B/26B/31B; Qwen3.8 Flash-Next Q2/~3.5-bpw/Q4
  and MTP/drafter/n-gram pipeline. Optimize complete
  request goodput with the same model semantics, source quant
  calibration and StateRoot precision. Do not change the
  Renoir/gfx90c launch hold or assume measured superiority.
  Routed-expert streaming applies only to later source-defined MoE variants.
~~~~

**Lines 823–823**

~~~~markdown
- [ ] Keep the local Renoir/gfx90c hold: 2B initialization aborted and automatic resets wedged it; no further launch/reset/recovery or Drop-retirement proof is implied.
~~~~

**Lines 824–824**

~~~~markdown
- [ ] Establish controlled cold/warm request baselines across all five GPU/three CPU optimization targets; accepted five backend profiles do not mean five cards or fleet-wide maximal capability use.
~~~~

**Lines 825–825**

~~~~markdown
- [ ] Qualify capability-selected AVX2/AVX512/VNNI quantized kernels, cache/thread/topology policies and source-derived repacking while preserving reduction/rounding/zero/subnormal rules.
~~~~

**Lines 826–826**

~~~~markdown
- [ ] Measure native/Vulkan tiling, subgroup/vector loads, packing, source-exact fusion/prepared dispatch; gate tensor/MMA/WMMA/integer-dot use by exact arithmetic, not advertised support.
~~~~

**Lines 827–827**

~~~~markdown
- [ ] Evaluate explicit NativeF32 projection policy only after physical FMA edge tests, all-codec trained vectors and whole-model/cancellation/replay gates; WideF64 stays default meanwhile.
~~~~

**Lines 828–828**

~~~~markdown
- [ ] Profile then qualify captured graphs/replay, bounded device loops/persistent scheduling, CPU/GPU overlap and exact optional GPU tokenization/detokenization; prepared plans alone are not graph capture.
~~~~

**Lines 829–829**

~~~~markdown
- [ ] Keep separate LLVM code-generation paths and broader accelerator backends unqualified until independently built and physically tested; accepted native paths use CPP NVRTC/HIPRTC.
~~~~

**Lines 830–830**

~~~~markdown
- [ ] Qualify optional EXL3/QTIP/trellis/codebook/low-bit derived packs and calibration identity/quality separately from original source semantics. No hidden precision substitution or default steering.
~~~~

**Lines 831–831**

~~~~markdown
- [ ] Promote only repeated complete-request gains with cold/warm cost, tails, memory, power, source correctness, negative results and qualified capability/version guards.
~~~~

**Lines 833–833**

~~~~markdown
- [ ] Gate future compiler/driver/libm and fast-arithmetic profiles with scalar fingerprints, physical edge/stress and trained full-logit/state tests. WideF64 is qualified on measured inputs, not a universal IEEE FMA emulation.
~~~~

**Lines 834–834**

~~~~markdown
- [ ] Qualify real deadline wakeups, concurrent/multi-stream scheduling and actual driver failure/recovery before broader worker claims; queue deadlines currently need scheduling/poll opportunities.
~~~~

**Lines 835–835**

~~~~markdown
- [ ] Keep the local gfx90c hold. If a healthy/recovered profile is separately authorized, localize retained allocation/H2D/final-sync initialization before changing kernels; natural SIGABRT establishes neither Rust Drop quarantine nor reclaimed storage.
~~~~

**Lines 836–836**

~~~~markdown
- [ ] Consider read-mostly publication only after representative contention and ownership evidence. If left-right fits, test deterministic dual replay without IO/RNG/admission/state/device effects, one writer, persistent non-Sync readers, short guards without await/load/device work and explicit publish/drain/shutdown. Account for both indexes/logs while sharing one payload; do not reopen the rejected cold-catalog migration.
~~~~

**Lines 838–838**

~~~~markdown
- [ ] Resolve applicable SYK-011..016 source/derived-quality/LLVM/SLO/pressure/steering choices from the [specialization reference](../../research/EXTREME_MODEL_SPECIALIZATION_INTEGRATION_AND_ACCEPTANCE_2026-10-15.md) before that slice; proposed SYK-002..010 are not frozen user approvals. Preserve negative candidates and the default disabled steering policy.
~~~~

**Lines 855–869**

~~~~markdown
- [~] Shared queue checkpoint preserves application-owned HOST admission,
  unlimited default pending count and optional cap. Final batch corrects
  tokenizer construction, original Rayon reuse, HTTP framing/serialization
  and worker/last-reader ownership. HTTP handles and connection credit survive
  through join; original default thread stacks remain runtime-selected.
  Platform stacks/TLS/caches and OS/driver headroom remain separate, unquantified
  scope: application reservations are not process RSS or a full-process fit.
  Contract decision.
  CPU functional acceptance is complete; the final full CPU gate passed
  rc0 25Oct on the committed tree. Assembled GPU checks PASSED 25Oct
  (22/22 rc0: builds/stricts, cuda_compiler_context 4/4, loaded/inventory
  4/4, CLI baseline/serve-short_hf, host-profile umbrella, four ignored
  route cases alone, five actual source-cost cases) —
  execution.
  Physical device scopes (CLI direct route, serving) remain open.
~~~~

**Lines 900–909**

~~~~markdown
- [~] Direct short-HF route lifetime correction: preliminary tokenizer drops
  before model loading, but its static retained-credit vector never releases.
  Return tokenizer plus its actual lease; validate the exact legitimate
  process-global bootstrap baseline through route/model closure. No VM9/H14
  source overlap. Closed11-path source draft has the typed owner; ROOT/Astra
  required collected corrections for request payload/credit lifetimes,
  unconditional original close, exact bootstrap/backend baselines and tests.
  Those source repairs landed in `52f76a0`; the affected optimized/strict
  batch and the final full CPU gate passed rc0 25Oct; real source-cost/
  lifetime and GPU serving/ownership checks remain pending.
~~~~

**Lines 910–923**

~~~~markdown
- [~] CUDA B4 HOST requirement needs correction before fresh acceptance.
  Source-derived P16 lower bound4,341,525,623B already exceeds4GiB by46,558,327B
  before bootstrap/native/other metadata; old source21 HOST fits omit the new
  R+G+V credit. Cost-only inventory currently bypasses that owner too. Preserve
  all real charges, GPU caps and CPU4096MiB population controls; review honest
  configured HOST requirements versus real owner/phase savings. Explicit
  short-HF8192MiB policy landed in `52f76a0`; CPU/default4096 and legacy4GiB,
  provider/physical and H/G/V caps remain unchanged. Measured 25Oct
  (application-owned configured-domain accounting at 8192MiB, gpu_executed
  false, no process-RSS or device claim): N1 .8B host_upper 4,075,700,811B,
  N1 2B 4,111,001,099B, B4 2B 4,741,293,239B, B4 .8B 4,736,259,779B — all
  fit 8GiB; legacy 4GiB refusal tied to authenticated host cost (excess
  446,325,943B, provider within cap). Real RAM/cgroup availability and
  device/serving execution checks remain open.
~~~~

**Lines 934–940**

~~~~markdown
- [~] Five-lane source is assembled in MAIN; the final affected batch completes
  portable core admission, actual spawn-capture ownership and direct-lease
  metadata. Original private-network HTTP checks now execute without a fallback.
  The final source freeze/full CPU gate passed rc0 25Oct and the five-lane
  batch landed as signed commits (`8f01635`..`68e13a8`); assembled GPU/
  physical checks, H6/G8/V10 cases and lane-tree retirement remain pending;
  no new benchmark yet.
~~~~

**Lines 1248–1258**

~~~~markdown
- [ ] Assembled .8B host batch: authenticate immutable114-weight cohort once
  per binding, reuse exact cold allocation analysis/certificates, seal fresh
  native roles once, index actual lease/access coverage. Current source/root/
  permit/generation/fence/cancel and physical range checks remain. Existing
  pre-node0 status clear permits removal of redundant4B upload/sync; no new
  sticky-status semantics or cap changes. Ten paths merged into ROOT ISO;
  frozen physical fixture passes full40+five-fault requalification, actual final
  host0/device0 and original owned closure. All14 pure metadata guards PASS,
  including three new lifetime/rebind/SSA/private-effect/alias guards. Matched
  API wave/ragged checks PASS full IDs/text/usage/finish and actual owned/source
  closure. Clean C1/C4 performance remains pending CPU/IO quiet clearance.
~~~~

**Lines 1259–1260**

~~~~markdown
- [ ] Native harness passes positive fixtures and rejects all named planted
  defects; independent oracle and selected-profile equivalence matrix pass.
~~~~

**Lines 1261–1263**

~~~~markdown
- [ ] Reused/ported implementation passes exact/approximate declared-contract,
  held-out quality where applicable, full-state/cancel/physical and native
  packaging/link checks independently of the harness's own controls.
~~~~

**Lines 1264–1264**

~~~~markdown
- [ ] QTIP-01..14: derived source codec/pack identity, transforms, held-out quality and physical kernel correctness pass independently of original model semantics.
~~~~

**Lines 1265–1265**

~~~~markdown
- [ ] HYB-01..18: exact source routing/draft/tap/mask/reduction and target-authoritative verification pass, including cases where a wrong draft is rejected or deeper drafting slows work.
~~~~

**Lines 1266–1266**

~~~~markdown
- [ ] GFX-01..18: WMMA/LDS caller/shape coverage, driver/compiler numeric gates and coherent-memory physical headroom pass; require matched complete-request gains. See the [50-test register](../../research/EXTREME_MODEL_SPECIALIZATION_INTEGRATION_AND_ACCEPTANCE_2026-10-15.md#4-acceptance-register-50-planned-tests-not-implemented).
~~~~

**Lines 1272–1272**

~~~~markdown
- [ ] Complete current15 fresh matched API/HTTP for .8B and2B; ROOT1254 running. Then quiet C1/C4 and C8 only after both rate+wall wins, no hot compiler activity; previous .8B measurements remain historical.
~~~~

**Lines 1273–1273**

~~~~markdown
- [ ] Full assembled check.sh phase gate after CLI/source reconciliation; source12 gate18 remains the last accepted whole gate.
~~~~

**Lines 1279–1279**

~~~~markdown
- [ ] Real current-source qualification remains mandatory before production routing: .8B natural/current-root parity including cached 47→50, physical fault/cancellation/retirement and real-model shutdown, then .8B 512/1K/2K/4K/8K and 16K/32K. Admit 2B contexts only where the exact two-root plan fits.
~~~~

**Lines 1280–1280**

~~~~markdown
- [ ] Current physical blocker: the approved .8B source is now visible at `/home/pvl/.cache/abird/zerogravity/external-baselines/models/Qwen--Qwen3.5-0.8B/2fc06364715b967f1860aea9cf38778875588b17`; its 1,746,942,600-byte safetensor SHA256 is exactly `04b1c301231dd422b8860db31311ab2721511346a32cb1e079c4c4e5f1fe4696`. The transient `.agents/runs` native bundle was intentionally purged and no `cuda-kernels.json`/legacy `short-hf-native.json` bundle remains under the repo or user cache. Reacquire/regenerate the qualified native/reference bundle through the maintained capture path; do not restore deleted run directories or substitute another model source.
~~~~

**Lines 1295–1298**

~~~~markdown
- [ ] These are fit/admission results only. Physical correctness, continuation,
  replay, faults and state-history qualification remain required for each cell
  before production use; .8B real-model qualification is blocked on a qualified
  native/reference bundle, not on model-source visibility.
~~~~

**Lines 1314–1316**

~~~~markdown
- [ ] Real physical after-launch cancellation/fault and real-model shutdown
  remain required with the approved .8B source; source-free ownership success
  does not substitute for those device tests.
~~~~

**Lines 1338–1339**

~~~~markdown
- [ ] Execute both phases serially as soon as the approved .8B safetensor shard
  becomes visible to the connector. Do not download or substitute source.
~~~~

**Lines 1365–1374**

~~~~markdown
- [ ] Physical execution is currently blocked by provider identity, not code:
  the verifier is pinned to libcublas.so.12 bytes=105,140,976 SHA256
  d63456d94f1b8e2408a8a9538c8740eb474c9eba48a84f420b15e2e10fbcd048
  and libcublasLt.so.12 bytes=749,205,904 SHA256
  96ddf4891ca4f4ed1475966783a9185df2abd64a81c3cc713b96d036696fc541.
  Historical validated runs used
  /usr/local/lib/python3.12/dist-packages/nvidia/cublas/lib/, but those held
  DSOs are not visible through the current connector. The available Nix CUDA
  12.9.2.10 DSOs have different sizes/hashes. Do not repin or substitute them
  without an explicit source/provider qualification decision.
~~~~

**Lines 1492–1494**

~~~~markdown
- [ ] When the approved source tree, pinned native bundle payloads and pinned
  provider DSOs reappear, use this wrapper with a fresh evidence path; no manual
  phase ordering is required.
~~~~


### [.agents/plans/distributed-execution/TODO.md](../plans/distributed-execution/TODO.md) — 12 unresolved markers

**Lines 21–24**

~~~~markdown
- [ ] Evaluate layer, tensor/head, context and expert partitions as independent
  placement choices under source-specific partition algebra. Distributed full
  softmax summaries and recurrent state are different contracts; changed reduction
  or8/16-bit activation wire formats need explicit numerical profiles.
~~~~

**Lines 25–27**

~~~~markdown
- [ ] Admit shared collective/IPC/RDMA backing, staging, graph and encoded-reader
  ownership through original per-physical-resource ledgers. UMA host-visible
  transfer is not discrete GPUDirect; port ratings/tunnels do not attest throughput.
~~~~

**Lines 28–31**

~~~~markdown
- [ ] Later MoE EP/replicas/EPLB/communication overlap: preserve logical expert
  source IDs, topology/version guards, old/new migration readers and rank-agreed
  batch/graph/verification shapes. PDL requires actual eligible hardware; DBO is
  not generic one-GPU prefill/decode overlap.
~~~~

**Lines 32–35**

~~~~markdown
- [ ] Later produced-state P/D/encoder separation and remote prefix hydrate:
  complete hybrid source/profile/tenant/root-frontier handoff, authenticated
  producer/consumer epochs, completion notifications and failure fencing. Heartbeat
  expiry or peer silence cannot reclaim unknown DMA or publish partial roots.
~~~~

**Lines 36–39**

~~~~markdown
- [ ] Compare placement against local batching and capacity needs using actual
  directional links/host copies/crypto/collectives/stragglers and offered-load
  tails/goodput. Retain two-device/tiny-model and grouped-pipeline negative cases;
  no universal speedup from more ranks or deferred global-service descriptions.
~~~~

**Lines 41–41**

~~~~markdown
- [ ] Preserve zero-network standalone execution; add capability/transport/placement seams through the same model/task/coordinator/resource authorities.
~~~~

**Lines 42–42**

~~~~markdown
- [ ] Qualify optional local CPU/GPU/NPU split projection, expert ownership and prefill/decode placement with actual event/memory/power-fabric safety; Stage N is independent of CPU/GPU release.
~~~~

**Lines 43–43**

~~~~markdown
- [ ] Begin authorized two-node layer-group plus local-state execution and immutable full-state hydration; then evaluate P/D and attention+state versus FFN/expert disaggregation.
~~~~

**Lines 44–44**

~~~~markdown
- [ ] Extend tensor/head/pipeline/data/context/expert/sequence parallelism and collectives only where exact state partition algebra, source numerics and transport cost support them.
~~~~

**Lines 45–45**

~~~~markdown
- [ ] Authenticate/version source/state/tenant/session epochs and complete transfers; loss/timeout/stale peers cannot publish partial roots or release unknown in-flight storage.
~~~~

**Lines 46–46**

~~~~markdown
- [ ] Qualify topology changes only at committed frontiers and compare collocation against fast/slow/tunneled links, including real transfer/copy/crypto cost and fairness.
~~~~

**Lines 47–47**

~~~~markdown
- [ ] Add a peer/transport/placement without rewriting model, task, coordinator or API; fleet SSH qualification alone is not distributed inference.
~~~~


### [.agents/plans/fabric-vnext-transition/TODO.md](../plans/fabric-vnext-transition/TODO.md) — 13 unresolved markers

**Lines 538–538**

~~~~markdown
- [ ] E01 original-HF general Qwen3.5/2B and real 27B; independent fresh47–50/long context CPU/CUDA/HIP/Vulkan.
~~~~

**Lines 539–539**

~~~~markdown
- [ ] E02 vLLM continuous batching, persistent row slots, temporal prefill, all-domain paged/COW state, goodput; benchmark.
~~~~

**Lines 540–540**

~~~~markdown
- [ ] E03 shared multi-model residency, replicas and hot reload.
~~~~

**Lines 541–541**

~~~~markdown
- [ ] E04 true large dense SSD authenticated windows and physical streaming.
~~~~

**Lines 542–542**

~~~~markdown
- [ ] E05 Gemma4 E2B/26B-A4B/31B complete native qualification, then new families.
~~~~

**Lines 543–543**

~~~~markdown
- [ ] E06 Q4_K source kernel, Q1/Q2/Q3/NVFP4/EXL3/source derived quant and quality evidence.
~~~~

**Lines 544–544**

~~~~markdown
- [ ] E07 sparse/QSA/GDN/MLA/Engram/MoE/MTP draft state and expert IO.
~~~~

**Lines 545–545**

~~~~markdown
- [ ] E08 native ASR/TTS/codec/image/video/duplex and typed task pipelines.
~~~~

**Lines 546–546**

~~~~markdown
- [ ] E09 NPU/graph/wafer/remote distributed placement and state bridge.
~~~~

**Lines 547–547**

~~~~markdown
- [ ] E10 calibrated cost/autotune, matched vLLM/llama.cpp task benchmarks and release.
~~~~

**Lines 578–579**

~~~~markdown
- [ ] Live insertion/replacement, heterogeneous/native shared residency,
  additional independent adapter retirement and context reinitialization.
~~~~

**Lines 636–638**

~~~~markdown
- [ ] Broader qualified physical interop; general DAG/media/context,
  mixed backend/multi-source HF and pipeline CLI/API activation only
  through original qualified owners.
~~~~

**Lines 708–712**

~~~~markdown
- [ ] Non-Linux bootstrap and general host-visible GPU/UMA backing
  aliasing beyond the qualified original coherent mapped producer.
  Broader family/codec/task
  registration, other resident formats, generalized persistent session storage, cross-provider restore and
  additional backend prefix eligibility remain open.
~~~~


### [.agents/plans/fabric-vnext/PUBLIC_LIB_AND_SERVING.md](../plans/fabric-vnext/PUBLIC_LIB_AND_SERVING.md) — 12 unresolved markers

**Lines 318–318**

~~~~markdown
- [ ] Compile a standalone Rust program depending ONLY on zerogravity and run a permitted local source/model task end-to-end.
~~~~

**Lines 319–319**

~~~~markdown
- [ ] Compile a standalone Rust program depending ONLY on zerogravity and embed/start/stop a live API server and submit a request with real emitted events.
~~~~

**Lines 320–320**

~~~~markdown
- [ ] Separate zerogravity-serving crate builds and integration tests WITHOUT zerogravity-cli or Qwen/Gemma/GPU import.
~~~~

**Lines 321–321**

~~~~markdown
- [ ] zerogravity-cli depends on zerogravity facade only (and conventional CLI/terminal/logging helpers); no private engine/serving internals.
~~~~

**Lines 322–322**

~~~~markdown
- [ ] All existing CLI actions have programmatic API equivalent with matching errors, rights and result semantics.
~~~~

**Lines 323–323**

~~~~markdown
- [ ] Server and direct SDK share loaded immutable model/physical backing and ResourceArbiter; no duplication by alias/listener.
~~~~

**Lines 324–324**

~~~~markdown
- [ ] Server ingress/reload/shutdown while active sessions run does not change their source/model generation/root unexpectedly.
~~~~

**Lines 325–325**

~~~~markdown
- [ ] Strict loopback/auth/body/stream conformance; slow/disconnected clients do not leak GPU buffers/HTTP queues.
~~~~

**Lines 326–326**

~~~~markdown
- [ ] A source-specific novel model executes via HTTP with no model-specific server or CLI branches.
~~~~

**Lines 327–327**

~~~~markdown
- [ ] Native media/structured task SDK output and server transport share typed event contracts; prepared vs committed vs delivered distinct.
~~~~

**Lines 328–328**

~~~~markdown
- [ ] CPU-only build requires neither network stack nor GPU compiler and can run without an HTTP daemon.
~~~~

**Lines 329–329**

~~~~markdown
- [ ] Rustdoc/examples plus crate/features docs show library-first usage before CLI-specific flags.
~~~~


### [.agents/plans/fabric-vnext/TODO.md](../plans/fabric-vnext/TODO.md) — 42 unresolved markers

**Lines 28–28**

~~~~markdown
- [ ] Confirm the user has expressly initiated the **read-only resync/revalidation phase**.
~~~~

**Lines 29–29**

~~~~markdown
- [ ] Read the full frozen Fabric vNext pack and literal user amendments, then AGENTS.md and current owner plans.
~~~~

**Lines 30–30**

~~~~markdown
- [ ] Inspect current live HEAD, branch, checkout, recent commits, in-flight agents and true Cargo feature/dependency graph. Protect third-party work.
~~~~

**Lines 31–31**

~~~~markdown
- [ ] Reconcile current source/architecture docs/active TODOs with historical 26-Oct analysis; classify old path anchors obsolete/new/already done.
~~~~

**Lines 32–32**

~~~~markdown
- [ ] Reconfirm current in-source model families, generic semantics, numerical policies, physical provider, queues, state publication, resource services and source ownership.
~~~~

**Lines 33–33**

~~~~markdown
- [ ] Reconfirm current model-neutral engine/SDK/CLI/serving responsibilities and look specifically for newly added server/engine library work.
~~~~

**Lines 34–34**

~~~~markdown
- [ ] Re-evaluate public zerogravity facade + standalone zerogravity-serving crate + thin CLI against exact new code; revise module names/ownership for simplicity without weakening the newest user decision.
~~~~

**Lines 35–35**

~~~~markdown
- [ ] Validate distinction among mathematical region, native ModelAssembly and multi-model TaskPipeline; complete-domain state and typed media clocks.
~~~~

**Lines 36–36**

~~~~markdown
- [ ] Crosswalk every source/design gap with the current active P00–P23 workstreams, including source/quality/physical and model/backend holds.
~~~~

**Lines 37–37**

~~~~markdown
- [ ] Produce source-grounded diagram, gap table, revised implementation plan, test evidence ledger and explicit remaining decisions. **Stop without coding.**
~~~~

**Lines 41–41**

~~~~markdown
- [ ] Receive explicit user permission to start implementation after the source-grounded resync.
~~~~

**Lines 42–42**

~~~~markdown
- [ ] Freeze a reproducible implementation/test baseline without corrupting active work.
~~~~

**Lines 43–43**

~~~~markdown
- [ ] Introduce minimal generic SemanticProgram/closed-region and prepare/bundle paths from existing ForwardPlan; avoid second IR.
~~~~

**Lines 44–44**

~~~~markdown
- [ ] Remove family math and source-specific source/ABI recipes from generic core, artifact, GPU and serving implementations.
~~~~

**Lines 45–45**

~~~~markdown
- [ ] Refactor model family crates and reverse dependencies, no GPU → Qwen/Gemma import.
~~~~

**Lines 46–46**

~~~~markdown
- [ ] Extract one model-neutral zerogravity-engine or equally enforceable cohesive engine boundary; one TaskCoordinator/StateRoot and physical ResourceArbiter.
~~~~

**Lines 47–47**

~~~~markdown
- [ ] Provide source-qualified model instance/session/TaskDriver interfaces and cold built-in registration, not model-name switches in runtime.
~~~~

**Lines 48–48**

~~~~markdown
- [ ] Implement public zerogravity Rust library for ALL functionality (run, inspect, load/pull/cache, admin, device planning, pipelines, metrics and serving).
~~~~

**Lines 49–49**

~~~~markdown
- [ ] Create independent, embeddable **zerogravity-serving** Rust library for HTTP/OpenAI/Responses/stream/media service protocols and lifecycle; no CLI/model-kernel imports.
~~~~

**Lines 50–50**

~~~~markdown
- [ ] Make zerogravity-cli ONLY call exported zerogravity library and own args/env/terminal/signal/output.
~~~~

**Lines 51–51**

~~~~markdown
- [ ] Prove separate library embed and server start/stop, direct run and HTTP share same model/resource/state; strict auth/stream/errors.
~~~~

**Lines 52–52**

~~~~markdown
- [ ] Prove Qwen3.5 0.8B/2B source-real CPU and selected GPU state/numeric/regression gates survive changes.
~~~~

**Lines 53–53**

~~~~markdown
- [ ] Add Gemma source-real model through same engine and server without source-specific allocator/HTTP/scheduler branching.
~~~~

**Lines 54–54**

~~~~markdown
- [ ] Record accepted and failed/unsupported paths per actual profile/driver/model/corpus and explicitly leave unrun hardware cases open.
~~~~

**Lines 58–58**

~~~~markdown
- [ ] Persistent GPU slots, incremental row metadata, actual async bounded step submission and retirement.
~~~~

**Lines 59–59**

~~~~markdown
- [ ] Native source-qualified continuous batching, real temporal prefill/decode overlap and fairness/ageing.
~~~~

**Lines 60–60**

~~~~markdown
- [ ] Paged and complete-domain hybrid state/COW/prefix checkpoint reconciliation including sparse and recurrent state.
~~~~

**Lines 61–61**

~~~~markdown
- [ ] Region fusion/capture/device-native feedback; optional speculation and draft/target full state semantics.
~~~~

**Lines 62–62**

~~~~markdown
- [ ] Unified physical residency and SSD streaming on one source identity with byte-budgeted cache and real completion.
~~~~

**Lines 63–63**

~~~~markdown
- [ ] Qwen3.8 27B exact source/CPU/selected GPU, Gemma E2B→26B-A4B→31B independent functionality; then richer sparse/quant targets.
~~~~

**Lines 64–64**

~~~~markdown
- [ ] First real non-token task and multi-model typed pipeline via same public library/engine/serving.
~~~~

**Lines 65–65**

~~~~markdown
- [ ] Qualified CPU+GPU+NPU, local split and authorized remote region workers only with actual source/state/physical/latency proof.
~~~~

**Lines 66–66**

~~~~markdown
- [ ] Matched vLLM/llama.cpp complete-request/quality/TTFT/ITL/P95/goodput/memory/energy benchmarks and negative findings.
~~~~

**Lines 67–67**

~~~~markdown
- [ ] Minimality/extension drills for new model, novel stateful op, backend, task, transport with measured shared source change radius.
~~~~

**Lines 68–68**

~~~~markdown
- [ ] Consolidate normative docs/public whitepaper/current status after actual implementation acceptance.
~~~~

**Lines 72–72**

~~~~markdown
- [ ] Keep this plan set internally linked and consistent with the canonical current handoff and four-file planning spine.
~~~~

**Lines 73–73**

~~~~markdown
- [ ] Revalidate source-dependent statements against current main before implementation; historical SHAs remain provenance only.
~~~~

**Lines 74–74**

~~~~markdown
- [ ] Keep the next-agent prompt source-first: read the canonical handoff, current status/traceability, then this plan; do not reconstruct deleted archive snapshots.
~~~~

**Lines 75–75**

~~~~markdown
- [ ] Later: enforce Cargo acyclic layering including serving→engine and facade→serving and CLI→facade.
~~~~

**Lines 76–76**

~~~~markdown
- [ ] Later: public zerogravity-only embedded Rust direct run/server examples compile without CLI, with one shared model/ResourceArbiter and exact source states.
~~~~

**Lines 77–77**

~~~~markdown
- [ ] Later: source/semantic/quality/physical/serving/multimodal/distributed acceptance gates defined in IMPLEMENTATION_AND_ACCEPTANCE.md and original project owners.
~~~~

**Lines 78–78**

~~~~markdown
- [ ] Later: no vLLM/llama.cpp performance parity claim absent true matched workload/driver/profile/quality evidence.
~~~~


### [.agents/plans/memory-and-streaming/TODO.md](../plans/memory-and-streaming/TODO.md) — 29 unresolved markers

**Lines 22–49**

~~~~markdown
- [~] Physically qualify each admitted context independently. Preserve current
  F32/F32/BF16 two-root semantics; never change precision/source/root semantics merely
  to make a cell fit. Public CUDA serving passes4K/8K/16K for both0.8B and2B with exact
  replay/frontiers and clean retirement;1K/2K separately retain all42/finite-state
  qualification.32K now has split status. The exact direct optimized0.8B32K
  retained qualifier has one full current-path pass in185.18s with exact replay,
  independent sessions, exact producer frontiers and clean shutdown. The next identical
  direct trace run was externally SIGKILLed at~80.7s after >60s of execution. Thus
  0.8B32K model capability is positively established once.2B32K public load succeeds and
  is now localized to a real long-context response deadline rather than infrastructure:
  current matched optimized runs survive normally to the retained SDK's600s
  first-response timeout (~616–617s total) without SIGKILL. Do not treat this as
  VRAM/admission failure. Simple partial-resident source placement (~2.9GiB resident),
  state-path-first pack ordering and a16MiB serial source page did not materially move
  the timeout, so those experimental production changes were removed. The obsolete
  CUDA-forward attention reserve is now removed from the real admission/ownership graph:
  the selected tile-local kernel uses no global scratch, saving138,428,516B at32K. This
  makes the low-level two-root2B32K inventory fully resident. Public active1 still
  retains four **logical** lifecycle roots, but the paged-state inventory now stops
  preallocating unreachable duplicate KV pages: active current roots own complete page
  sets while snapshots/displaced/Pending candidates share immutable prefixes and reserve
  only bounded divergent COW/appended pages. This preserves lifecycle authority and makes
  explicit paged2B32K fully weight-resident. Same-capacity1024-token public A/B improves
  283.22→28.56s, and full32766-token paged serving passes in224.80s. The21 Nov generic
  smart-config layer supersedes the fixed Contiguous default: canonical `QwenCuda` uses
  Enabled mode and Auto selects Paged64/64 when it preserves weight residency, while user
  values still win per knob. Final CUDA admission independently re-observes free VRAM from
  the original actor and clamps the provider envelope before allocation.
~~~~

**Lines 50–58**

~~~~markdown
- [~] Preserve complete recurrent/conv/KV state history, chunk continuation,
  cancellation/quarantine and readback owners at512/1K/2K/4K/8K/16K, then32K.
  Public4K/8K/16K execution proves the current chunk-continuation/serving path;0.8B32K
  has one full retained public-SDK pass and2B32K now passes the full32766-token retained
  cell under explicit Paged64/64 in224.80s. Canonical default2B32K/1024 also physically
  Auto-selects that paged policy and passes through the live actor-side VRAM clamp. The
  independent populated all42 archive is absent on L5, so32K **numerical** closure remains
  open; use the retained32768 qualifier and current smart/storage policy rather than
  adding another context fixture or restoring the obsolete refusal assumption.
~~~~

**Lines 63–71**

~~~~markdown
- [~] Bounded completed-root backing reuse is adopted in the shared resident layer.
  `ResidentPool`/`BackingPlane` retain one charged physical allocation while every
  checkout receives fresh `BufferLeaseId`/allocation generations. `ResidentReadback`
  and `ResidentReady` retain the checked-out lease/use; a set returns idle only after
  the device ledger proves every backing view idle. Failed/unknown retirement poisons
  the use and fences/quarantines instead of reusing it. CUDA-forward/HIP source
  sessions install this pool through `install_window_root_pool`. Remaining work in
  this roadmap item is the broader authenticated transfer-slot/ring and physical-
  backing identity adoption across source/staging paths; do not add another root pool.
~~~~

**Lines 78–109**

~~~~markdown
- [~] The first CUDA-forward source-page transfer cleanup now uses the existing
  **owned authenticated window** boundary directly. `TensorWindow` remains the exact
  source/cache/authentication owner and already carries the real `TransferStaging` lease;
  CUDA wraps only that owner in an `Arc`, installs it in the page's existing pending state
  before enqueue, and the original actor reuses its generic bounded owned-window H2D
  primitive. The existing page+consumer `sync_buffers` checkpoint is still the only serial
  reuse boundary. No payload `Vec`, second `Bytes`, CUDA pinned staging allocation, new pool
  or new retirement ring exists on this path; HIP/non-CUDA serial representation remains
  unchanged by cfg. The earlier `b839920` source-side pinned experiment is superseded: the
  CubeCL pinned `Bytes` controller has no cheap `duplicate`, so `bytes.clone()` deep-copied
  to Native and the generic writer then staged to pinned again. A cache-promotion variant
  (41.00s) and moving the pinned `Bytes` directly into the writer/drop-queue lifetime
  (44.88s) were both rejected against the same 2B/32K Contiguous/Streamed prompt48 cell.
  The final retained-owner actor upload passes that exact cell at **38.46s**, versus39.36s
  at `b839920` (~2.3% faster), with exact replay/frontiers, independent sessions and clean
  shutdown. The earlier direct SSD→pinned/cache-bypass prototype remains rejected at41.77s.
  The next simple overlap attempt is now rejected too: generalizing the existing two-slot
  retirement ring to CUDA while keeping one device page/same stream passed strict checks,
  but matched same-tree Contiguous/Streamed2B32K/prompt48 measured **38.323s serial vs
  39.281s ring** (~2.5% slower). The Qwen consumer/selector and temporary Contiguous policy
  switch were removed; an Auto control at24.267s chose paged state + resident weights and did
  not exercise this fallback. Keep the reusable mechanism: CUDA `OwnedCudaEvent` and HIP
  `OwnedHipTransferRetirement` directly implement `RetirementMark`; the zero-cost
  `CudaTransferRetirementRing<Owner>` / `HipTransferRetirementRing<Owner>` aliases reuse the
  existing generic bounded ring with no wrapper allocation or model policy. These dormant
  primitives are available to Gemma, larger streamed models, other CUDA/HIP devices or
  workload shapes that can separately prove exact ownership/admission and a measured benefit. Do not re-enable the
  rejected Qwen one-page selector merely because the primitive exists. A future second device
  page or copy stream still needs separate physical admission and a real workload case that
  requires streamed weights. No benchmark fixture/evidence pin was added. Direct/async source
  IO remains future work only if it composes with the verified cache/miss policy instead of
  bypassing it.
~~~~

**Lines 110–119**

~~~~markdown
- [~] **25 Nov matrix window/transfer follow-up:** `7b36a879` and
  `0342e103` share authenticated window row validation, original owned CUDA
  H2D for cold packing, and the existing streamed pending/reuse checkpoint;
  resident access is direct, HIP retains its actual event retirement ring.
  Portable serial staging has an admitted copy/bounce bound. No new page
  allocator or cache. `fdfb467f` real Qwen 0.8B resident/streamed exact
  B1 full logits/all42 and native B4 tokens/scores/all42, streamed
  cancel/discard/retry, SDK C1 and native HIP regressions pass. Fresh
  compiler contexts avoid the cumulative PTX limit hit in the first
  combined comparison; do not raise the budget.
~~~~

**Lines 133–171**

~~~~markdown
- [~] Canonical physical buffer identity now exists in the model-neutral device
  ledger. `DeviceBufferIdentity` is an opaque process-local **ledger namespace +
  allocation owner + exact byte range**: direct allocations are owned by their original
  lease allocation, while pooled/aliased views name the retained `DeviceBackingId`.
  The non-wrapping ledger namespace prevents identical local lease/backing counters from
  separate resident services from comparing equal. Equality/`same_allocation`/`overlaps`
  therefore describe real storage rather than hashes, chunk coordinates or backend
  handles. `DeviceEventLedger::buffer_identity` is the sole public constructor, and
  production overlapping-alias admission now uses the same identity semantics. The first
  resident consumer is the source-page immutable-cache/output alias guard: both HIP and
  CUDA paths validate all outputs through shared `ResidentService::overlaps_any`, which
  checks service ownership and performs one ledger lock with no heap allocation or
  per-lease identity field. The same primitive now replaces backend-handle range checks in
  the native B4 embedding preparation path for output/table/status and feedback aliases;
  its four-packet diagnostic pairwise check simply reuses `overlaps_any`. The production
  CUDA-forward body now has the first **prepared-set** consumer too: shared
  `ResidentService::pairwise_disjoint` validates the complete deduplicated resident lease
  set under one ledger lock before binding, while `BufferRange`/`OwnedRanges` remain only
  for their legitimate kernel-handle geometry role. The prepared streamed BF16 vocabulary
  head now reuses that same helper for its weight-staging/input/output lease triple, so
  distinct logical lease IDs cannot hide overlapping physical backing before the existing
  two-slot authenticated retirement ring starts. The ring, source-owner lifetime and
  same-stream overwrite/retirement protocol are unchanged; no second transfer abstraction
  was added. The production HIP resident-text path now uses the same authority too:
  cold workspace owners are pairwise ledger-checked once, while per-request root binding
  uses a one-lock incremental disjointness check against that already-validated workspace
  plus earlier new roots. This preserves the old O(root*workspace + root²) validation shape
  instead of rechecking workspace² on every token, and keeps exact handle ranges only for
  execution-view length/binding geometry. The helpers allocate nothing internally and cache
  no identity into leases. HIP source/public composition checks and the two existing host
  source-admission tests pass; X2 physical execution remains environment-gated. Existing B8
  public-SDK Contiguous/Paged execution remains exact/clean, and the retained 0.8B
  context1024/prompt48 public-SDK replay cell passes again after the shared helper refactor
  with exact tokens/frontiers, independent sessions and clean shutdown. No new alias fixture
  was added. Six backing-ledger tests cover cross-ledger separation, direct-vs-backed
  identity, exact offsets/ranges, overlap/disjoint aliases, continuous charge, cancellation
  and unknown fencing. Remaining adoption is to query this identity only where additional
  graph views or transfer consumers actually need it, without mandatory hot-path bookkeeping.
  Encoded/captured/queued readers still remain part of the broader retirement work.
~~~~

**Lines 172–174**

~~~~markdown
- [ ] Add measured lifetime-aware scratch reuse/phase lending only when no live
  root, graph, transfer or reader needs the range. Price borrowed capacity,
  old/new COW copies and following decode residency; do not import a hidden arena.
~~~~

**Lines 175–178**

~~~~markdown
- [ ] Protect demand/decode hotsets from transient prefill and unused predicted
  reads; compare protected segmented/recency/frequency policies under topic shifts.
  Different source codecs use byte-size classes under one cap. Cache policy may
  change residency, never source routing, expert count or arithmetic silently.
~~~~

**Lines 179–182**

~~~~markdown
- [ ] Separate known source lookup IDs, dense next-layer windows, produced-state
  restore and predicted expert prefetch, while sharing fair bounded IO/host/device
  admission. Optional direct/async IO retains the same hash/short-read/cancel
  contract. A fully resident path performs no mandatory cache-miss bookkeeping.
~~~~

**Lines 183–198**

~~~~markdown
- [~] Common-frontier prefix reconciliation is now implemented model-neutrally in
  `zerogravity-engine::state_cache`: `PrefixDomainCheckpoint` supplies each required
  domain's retained-through frontier and zero-based checkpoint alignment, and
  `reconcile_prefix_frontier` returns the largest request-bounded frontier legal for
  every domain. The hot check is allocation-free O(domains), uses an overflow-safe
  bounded LCM and returns0 when no nonempty common aligned checkpoint exists. Exact
  partial-tail COW is **already closed by the existing paged-state path**, not by this
  frontier algebra: `share_prefix` retains an immutable physical tail page with a narrowed
  valid extent; `fork_for_append` transactionally allocates a fresh page and emits a
  source→target copy for exactly the valid tail; the append plan pins both handles, the
  resident submission clones those Read/Write leases until physical completion, and only
  Ready candidate pages can enter the atomic root/domain swap. Displaced domains drop only
  after the new root commits. Existing physical materialization/boundary/source-fork
  qualifiers prove fresh generations, immutable source bytes, exact partial-tail copy,
  zero inactive tail and complete retirement. Do not add a second COW owner abstraction.
  Approximate truncated replay and moved-message reuse stay outside exact mode.
~~~~

**Lines 199–202**

~~~~markdown
- [ ] For dense27B, compare layer-major/row-cohort source-window reuse against
  extra activation spill, retained states, IO amplification and fair decode
  progress. First correct execution may itself require streaming; no full-resident
  runtime is assumed feasible under the declared5GiB target.
~~~~

**Lines 204–208**

~~~~markdown
- [ ] **Next SSD source execution:** integrate accepted original-index
  owned windows and an async I/O/readiness model with
  original CPU/GPU TensorSource and BufferLease, cancellation,
  cache/eviction/ABA and independently trained full-logit/StateRoot
  parity on real large weights.
~~~~

**Lines 210–215**

~~~~markdown
- [ ] Shared one-process pipeline residency: source-exact immutable
  model/component buffers, per-session private StateRoots, one
  ResourceArbiter (UMA/GPU/RAM/SSD and device queue), cross-model
  lease/zero-copy only when physical placement, codec and lifetime
  permit it. Do not replicate weights per route or release a
  GPU buffer before actual completion/quarantine retirement.
~~~~

**Lines 216–222**

~~~~markdown
- [ ] Prioritize Qwen3.5 2B resident then Qwen3.8 27B SSD
  streaming and Gemma4 E2B/26B/31B, then sparse Qwen Flash-Next
  learned 51B n-gram bank and 4B MTP head placements. Compare
  source-real logits, resident/streamed StateRoot, cold/warm
  TTFT/ITL, I/O/read amplification and throughput with full
  physical/cancel/error gates; live audio's soft deadline may
  demand resident codecs and refuse cold SSD layouts.
~~~~

**Lines 224–247**

~~~~markdown
- [~] Host scopes share one original physical backing constraint
  (cbd2c119/197c2e26); all host lease admissions/retirements/partitions use
  the same arbiter. First observed ceiling and later known cgroup constraint
  separate; actual source/control/context cold fit, no hot OS polling.
  Qualified HIP intersects driver/local VRAM without summing host/GTT.
  Real CUDA and HIP source regressions pass; initial lifetime-upper-as-growth
  check corrected without weakening logical bound. Remaining direct/legacy
  unadmitted owners, general UMA backing alias mapping and non-Linux/OS
  hard limits still need qualified consumers. a6548863 shares
  original Linux HIP preadmission with the ordinary portable owner and
  guarded exact-adapter Vulkan local fit; same declared physical placement
  does not establish common backing or zero-copy. Actual portable HIP/Vulkan
  two-source healthy/unknown and peer refusal/retry pass.
  3b5a520d/fc38db03/e1decff5 close bounded named Linux Vulkan original
  service reuse and local fit; actual cold peer refusal/raced/warm reuse
  and independent envelope retirement. ce83cc18 exposes actual retained
  capture/cache/pin/accepted replay facts without another backing owner;
  original retirement/unknown retention remains authoritative.
  14bbbd48/5d952364/5a7535e7 qualify exact coherent Vulkan host producer
  on original managed backing/range; synchronous borrowed copy under
  exclusive Write and actual prior barrier, original Ready, no second
  charge. Actual GPU full/subrange/neighbor consumption and retirement,
  foreign/extent/duplicate/HIP refusal. General device-write host reads,
  foreign imports and aggregate UMA capacity remain unqualified.
~~~~

**Lines 248–366**

~~~~markdown
- [ ] **Paged hybrid state allocator:** extend original leases/StateRoot with
  finite page pools/block tables, generations, valid tails, charged metadata
  and peak COW/candidate memory. Retain a qualified contiguous policy and typed
  recurrent/conv checkpoints. Pin encoded/captured/queued/in-flight readers;
  reclaim only after real completion, preserving cancellation/quarantine.
  Design and physical acceptance: [batching/paged state](../serving-and-speculation/BATCHING_AND_PAGED_STATE.md#paged-state-allocator--memory-plan).
  Signed `0027c28` lands the logical-only shared block-table/COW planner:
  exact source/numerical/domain/layout + semantic page/unit compatibility,
  immutable full-page sharing, partial-tail COW, transactional allocation
  refusal and copy/write owner pinning. It deliberately owns no device allocator
  or Ready/root authority. Signed `54340a6` now binds it to a finite
  `ResidentPagePool` over existing `ResidentPool`/`ResidentLease` physical
  backing/generation/event/quarantine ownership. The CUDA physical qualifier
  passes finite pressure, fresh reuse generation, fork/read pinning, completed
  GPU readback reuse and submitted-unknown fencing/quarantine that diagnostic
  sync cannot reclaim. Signed `1eb6e0f` adds actual COW materialization under
  those original leases/events: fresh-page zero, partial-tail device copy and
  append H2D ranges physically pass page2048/unit512 with valid[2048,512],
  immutable source, zero inactive tail and complete retirement. Model KV
  block-table lowering remains open; recurrent/conv domains remain typed
  contiguous checkpoints. Signed `3ac2059` adds a typed paired K/V paged-domain
  representation and bounded gather-to-contiguous adapter. RTX3060 frontiers3/4/5
  around a four-token page match exact contiguous K/V bytes under a four-page
  peak cap, including boundary COW, full-page sharing/backing reuse, immutable
  parents, zero tails, capacity refusal and final release. At `3ac2059` it was
  not yet selected by `ResidentSession::Domain`. Signed `dc6a188` adds transient typed K/V
  staging plus device-to-device scatter from the unchanged contiguous candidate
  ABI into private pages. Physical prior3->candidate4 stays under one original
  flight, does gather2/prefix-copy2/scatter-COW2/scatter-append2 with no host
  readback after input, preserves the parent and retires all pages. Session
  integration and full-model parity remain the next allocator/COW gate. Signed
  `53336b3` adds the canonical paged-KV `Domain` representation and the exact
  original root-swap helper used by single-session commit; physical roots0->3->4
  publish with initialized page ownership, exact displaced parents and full
  retirement. Production ordinary sessions still select contiguous KV. Admission
  review rejects a separate full prior+candidate staging pair on top of paged
  state because that becomes a third state-sized envelope at full context.
  Signed `ef9e4d4` adds the ordinary KvInPlace execution contract: exact fresh
  prior/candidate staging alias, no zero/prefix-copy preparation, physical
  prefix/new-row/tail exactness and clean retirement. Next split ordinary root
  storage into admitted vector/page/staging pools, allow only exact initialized
  current-root full pages as inherited Ready state, and add paged snapshot/read/
  restore before selecting the production session route. Signed `93468b1`
  closes those semantic gates: paged partial/full snapshot +4->3->4 restore pass;
  Ready requires only new/COW page Writes and proves2 inherited full pages at
  4->5. A pure hybrid inventory plan now proves vector/staging/page maxima stay
  inside the old root byte envelope and rejects insufficient fork headroom.
  Signed `cb57e1d` constructs the real Vector/staging/page pool owner and
  verified-source installer. Physical4-vector-root +1-staging +24-page maximum
  equals the old67584-byte envelope exactly, reuses all idle backing and retires
  to resident0. Initial hybrid roots are zeroed Vector + empty paged KV; candidate
  allocation still refuses without the current root. Signed `b9b04aa` wires
  current-root-aware candidate execution: Vector candidate + one staging set +
  per-KV paged append under one exact submission, gather before the ordinary body,
  scatter after, Ready only for Vector/fresh-COW outputs and paged Domain creation
  after completion. Physical one-Vector/two-KV base3->4 is exact with gather4/
  scatter8, five Ready outputs, exact continuation/tails and final resident0.
  Signed `1413f72` adds the explicit low-level source policy while production
  registration stays Contiguous. Paged policy installs the accepted hybrid pools
  only for explicit page_tokens/max_append_tokens, enforces B1 and append bounds,
  and has an original-HF Contiguous-vs-Paged full-vocab/all42 page-boundary
  differential. The real L5 managed cache now contains approved0.8B/2B snapshots;
  original0.8B page4/append4 passes full-vocab/all42 at3/4/5/9.
  Signed `398e446` adds the matching paged B1 lifecycle target:
  prelaunch cancel, abandoned Ready, cancel-after-Ready discard, page-boundary
  commits, exact all42 restore and shutdown. Signed `3041457` corrects restore
  qualification to fresh root/epoch publication at the saved frontier.
  Cold-paged0.8B lifecycle is accepted by retained exit0 dotlink job
  `9feb1394...` with `contiguous_warmup=false`, exact lifecycle/all42 and
  shutdown. Signed `e6aac53` leaves the qualifier cold-only. Other null-exit
  attempts are signalled-Bubblewrap infrastructure events with zero cgroup
  OOM/pids and no observed dotlink restart/poll-failure evidence, not model
  failures. Deployed dotlink v0.17.3 records a signalled child as
  `Completed + exit_code=null + success=false` via `ExitStatus::code()==None`;
  main's richer signal enum is not deployed. A direct no-inner-supervisor run
  still reproduced null exit. Dotlink v0.17.4 follow-up corrects the earlier
  quiet-job interpretation: every connector-initiated stop is now logged, and
  the relevant window contains only an explicit `job_cancel`; a silent long
  sleep survived until cancellation. Earlier SIGKILL9s therefore belong to the
  surrounding harness/orchestration layer, not a dotlink quiet-job watchdog.
  With a heartbeat sidecar used only as harness plumbing and a foreground model,
  original2B
  page4/append4 is accepted: `38c1f321...` differential exits0 in703.17s with
  full-vocab/all42 exact at3/4/5/9; `744f36fd...` cold lifecycle exits0 in
  438.18s with exact cancel/abandon/restore/all42/shutdown. Signed `6b920a0`
  owns the native-GPU harness repair: reuse existing Bubblewrap user/PID
  isolation, seccomp-deny socket/io_uring creation when Bubblewrap networking is
  shared, retain outer user+net unshare otherwise, expose `--isolation-info`,
  and require `--require-private-network` for loopback callers. Signed `4688cba`
  adds recognition of an already-entered child netns only when self is loopback-only,
  PID1's network view differs and no socket FDs are inherited.31/31 harness tests
  plus real child-net loopback+CUDA smoke pass. Direct shared-net L5 still refuses;
  an explicit `bwrap --unshare-net --bind / / --dev-bind /dev /dev --` wrapper can
  supply the required private loopback namespace. Both small approved models clear source-real paged B1
  state/lifecycle. **19 Nov original0.8B also clears true in-flight cancellation:**
  a real post-enqueue/pre-Ready launch cut proves Vector/staging/page ownership is
  held until known physical completion, cancellation retirement creates no Ready
  or initialized candidate Write, committed root/all42 stay exact, pool occupancy
  returns exactly, replay continues and shutdown is clean. A deliberate submitted
  future drop proves the existing fail-safe unknown path fences the service and
  quarantines the pending event,396 handles and completion owners without pool
  reuse or root publication. Two externally SIGKILLed attempts remain incomplete;
  the terminal known/unknown runs pass. **19 Nov high-level selection is now
  explicit but still non-default at the19 Nov checkpoint:** `QwenCudaWithStatePolicy`
  carries public `QwenCudaStatePolicy::Paged`; the21 Nov smart-config layer supersedes
  the fixed Contiguous default while retaining this API as a user override. The
  `page_tokens` remains explicit. Native rows4/8 now share the same Hybrid pool
  and per-lane candidate abstraction: paged gather/body/scatter stays under the
  original cohort submission and only actual fresh/COW outputs can become Ready.
  Contiguous B8 keeps its existing optimized KV transition; paged uses the simple
  already-qualified `KvInPlace` append path. Focused selector/policy, strict clippy
  and serving-composition checks pass. The compact original0.8B native transport
  parity cell is now terminal-green for B2/B4/B8 prefill/decode/all42 (306.84s,
  clean shutdown); two earlier external-SIGKILL attempts remain incomplete only.
  Public SDK is also terminal-green with8 concurrent page-boundary requests,
  native B8 on both Contiguous/Paged, exact returned IDs and clean shutdown
  (340.78s). Public HTTP now passes the same workload under the proven child-private
  netns: native B8 on both, exact IDs, clean shutdown in358.07s. Prefix reuse and
  overlap are still later.
~~~~

**Lines 367–425**

~~~~markdown
- [ ] **Complete-domain prefix reuse/COW:** the first physical fork substrate is
  accepted. `ResidentSession::fork_shared_prefix` seeds only a fresh destination,
  validates same source/service/capacity/all domains and publishes cloned immutable
  owners under the destination's own replacement root; resident-text usage additionally
  requires the same resident model. Original0.8B paged page4 prefix3 -> fork ->
  append1 matches independent replay for full logits/all42, keeps the cached snapshot
  immutable and retires cleanly in179.13s. No replay/readback/copy occurs at fork.
  The bounded lookup/eviction substrate is now implemented in the engine:
  `2df8cbb3` supersedes numeric compatibility placeholders:
  `PrefixCompatibility<K>` binds the actual typed prepared state key and sharing
  scope; `PrefixCache<U,T,K>` keeps exact units + immutable `Arc` payloads under
  entry/weight caps, does allocation-free greatest-prefix/newest-tie lookup, refreshes
  duplicates and evicts FIFO. It creates no restore authority. **Explicit scoped
  worker integration is now accepted:** nonzero `PrefixSharingScope` travels beside
  the admitted request; ordinary requests stay isolated, explicit SDK calls opt in.
  Restore is allowed only into a fresh executor state and must match its actual
  frontier; cache insertion happens only after committed state-only prompt progress.
  Qwen enables retention only for paged state, with model-local `active` entry and
  `active*context` semantic-token bounds. Original0.8B page4/max_append64 prompt65
  physically proves65 warm /1 same-scope /65 different-scope model-token work,
  identical generated ID and clean retirement in172.98s. The first run exposed and
  fixed cache-before-resident retirement ordering. Trusted server-side HTTP scope
  injection is now accepted: one `ServerBuilder` scope is fixed by the embedding
  application, absent from the wire format, and unaware services fail closed. The
  physical original0.8B prompt65 HTTP proof records65 warm /1 same scope /65 different
  server scope with exact output ID and clean shutdown in171.97s. **Actual FIFO/
  weight eviction with live snapshot owners and explicit stale-scope cache invalidation
  is now accepted.** `invalidate_prefix_scope` is the trusted retained-cache purge and
  reports exact removed entry/weight totals; it does not cancel already-admitted
  work. Stop/drain stale-scope admission first; server stop/scope switch alone does
  not purge.
  Qwen uses `2*active` entries and `active*context` semantic weight. Physical active1
  page4/max_append64 proves prefix64 FIFO oldest miss/newer hit, purge2/128 + invalidated
  scope miss, and independent512-weight pressure: prefix320 hit1 while evicted prefix256
  makes prompt257 replay257; final purge2/448, repeat purge0, clean shutdown in
 179.12s. **Externally imposed resident page-pool pressure is now accepted:** a
  qualification-only owner pins real pages from the same finite pool around a cached
  prefix64. Original0.8B page4/max_append64 holds5952 pages, fills6144/6144, gets
  prelaunch Capacity without root mutation or vector/staging leakage, then after
  release reproduces the complete baseline logits exactly and retires all pages
  before clean shutdown in168.51s. One startup SIGTERM attempt is incomplete only.
  Minimal production diagnostics expose current prefix-cache occupancy/bounds and
  paged-pool allocated/idle/occupied/max/page-byte values. No cumulative hot-path
  pressure counters were added. External host/device OOM and future multi-checkpoint
  domain sets remain outside this checkpoint.
  **25 Nov typed compatibility/admission follow-up accepted (`2df8cbb3`):**
  original resident service/plan/source/geometry/numerical/port/storage identity
  is shared by snapshots and sessions; independent preparations cannot reuse state.
  Restore/fork validate source/service/capacity/complete-domain/lease generations
  and paged logical frontiers and publish through the same root swap. State-only
  lookup excludes the last token (prompt64 against cached64 replays64, correct
  output). Admit cache backing before native prep; bound retained token arrays
  plus one incoming prefix; charge actual variable cloned snapshot metadata
  alongside existing control allowance. Drop cache and backing before resident
  retirement. Physical fork/restore, paged/contiguous B1, scoped reuse/eviction/
  exact-length/HTTP and native HIP regressions pass. No new portable persistence
  or cross-owner/provider restore. Missing recurrent state still requires an
  earlier complete checkpoint/recompute, never a KV-only full hit. Exact gates and serving owner:
  [prefix design](../serving-and-speculation/BATCHING_AND_PAGED_STATE.md#complete-domain-prefix-reuse-and-cow--memory-plan).
~~~~

**Lines 441–444**

~~~~markdown
- [ ] Extend to other qualified resident layouts/providers and general
  compatibility/security/tenant store UX, disk capacity and
  restore-versus-recompute. Diagnostic portable checkpoints are not
  product cross-provider persistence.
~~~~

**Lines 445–445**

~~~~markdown
- [ ] Add immutable source extents and bounded coalesced IO/materialization with stable logical tensor identity, genuine copy readiness and original IO/device reader leases.
~~~~

**Lines 446–446**

~~~~markdown
- [ ] Protect encoded-but-unsubmitted/captured expert reads and generation ABA; support mixed-codec expert size classes under one byte cap, source-exact routing and hot-cache preservation.
~~~~

**Lines 447–447**

~~~~markdown
- [ ] Add bounded PLE/Engram lookup rows and separately governed disk state/media tiers; fully resident execution must incur no mandatory IO or host cache dispatch.
~~~~

**Lines 448–448**

~~~~markdown
- [ ] Test source-exact resident/streamed parity, short/corrupt IO, cancellation, safe victim refusal, SSD QoS and actual physical lifetime before eviction/free-space claims.
~~~~

**Lines 449–449**

~~~~markdown
- [ ] Optional direct/async/platform IO needs equivalent semantics and explicit support; large memory-pressure/SSD experiments require approved hardware/source scope.
~~~~

**Lines 451–451**

~~~~markdown
- [ ] Distinguish logical envelope retirement from driver allocator pages/pools, fragmentation, cross-process use and actual OS release. Driver VRAM/GTT usage/max counters are not reservations or immediately free bytes.
~~~~

**Lines 452–452**

~~~~markdown
- [ ] Prove trusted teardown/recovery before reclaiming unknown-work handles or clearing a fence. Diagnostic service sync/readback permits inspection/exit only and cannot synthesize Ready, retire unknown events or authorize reuse.
~~~~

**Lines 529–529**

~~~~markdown
- [ ] SSD-01..32: run the [full source/residency suite](../../research/SSD_STREAMING_IMPLEMENTATION_ROADMAP_AND_ACCEPTANCE_2026-10-15.md) for each claimed backing/backend; mock tiers cannot qualify physical IO or trained streaming.
~~~~

**Lines 530–530**

~~~~markdown
- [ ] SSD-14/15: encoded-but-unsubmitted, queued and captured readers prevent generation-ABA slot reuse through actual GPU completion/cancellation.
~~~~

**Lines 531–531**

~~~~markdown
- [ ] SSD-21/22/24: mixed Q2/Q4 expert sizes share one byte cap with safe victims and fair progress; hot-cache retention cannot bypass admission.
~~~~

**Lines 532–532**

~~~~markdown
- [ ] SSD-32: an authorized original trained streamed model matches complete resident logits/StateRoot; measure cold/warm IO, staging, OS headroom and genuine resource lifetime.
~~~~


### [.agents/plans/model-acquisition/TODO.md](../plans/model-acquisition/TODO.md) — 19 unresolved markers

**Lines 377–379**

~~~~markdown
- [ ] Extend advanced forced interactive/JSONL and aggregate component
  control over independently published original receipts; keep exact
  provider refusal and cancellation semantics.
~~~~

**Lines 381–392**

~~~~markdown
- [ ] Canonical source-pull execution order (18 Oct decision): Qwen3.5 .8B/
  2B, Qwen3.8 27B; Gemma4 E2B/26B-A4B/31B, each exact chosen GGUF
  quants or indexed original safetensors; QwenImage2.1 original
  Diffusers components; Qwen3 TTS/ASR; Whisper; NVIDIA Canary and
  NeMo Parakeet; Qwen3.8 Flash-Next original plus separately sourced
  Q2/~3.5-bpw/Q4 quants and independently verified MTP/n-gram
  embedding banks. Do not guess literal Q3.5 quant tags or conflate
  runtime speculative n-gram with trained n-gram checkpoint tensors.
  Implement exact typed presets/component assembly and run real
  metadata-only then authorized actual payload/sidecar/checksum
  publication gates. DeepSeek V4.1 Flash and GLM5.3 Flash have an
  explicit NO PAYLOAD download hold; prepare metadata pipeline only.
~~~~

**Lines 393–399**

~~~~markdown
- [ ] Extend the new bounded HF/GitHub Auxiliary-only --bundle to
  source-verified Diffusers model_index nested shard sets, ASR/TTS
  processors/codecs, NeMo archive streaming/extraction and
  release-specific recursive dependencies under one pin/manifest.
  Current --bundle is a user-selected subset and cannot claim a
  complete model/codec assembly, support auto-loading, or run without
  newly qualified source parser and license checks.
~~~~

**Lines 401–401**

~~~~markdown
- [ ] Extend cloud profiles beyond the narrow qualified AWS FULL_OBJECT SHA256 object route: broader S3 partitions/compatible vendors/credentials/checksum algorithms, Azure latest/SAS/managed chains and per-object digest-map prefixes. Keep explicit version, caller-SHA and conditional FirstSeen contracts distinct; never infer immutable versions or trusted raw hashes from unqualified validators or multipart COMPOSITE checksums.
~~~~

**Lines 402–402**

~~~~markdown
- [ ] Extend ModelScope only through separately qualified alias authority and redirect/CDN/mirror/credential profiles. Selected missing sizes now use exact pinned HEAD; no alias-to-commit authority was established from revision echoes, timestamps or file last-modified IDs. The current profile does not close broader parity or independently prove server revision/membership behavior.
~~~~

**Lines 403–403**

~~~~markdown
- [ ] Measure and qualify native Xet fetch coalescing/chunk caching and any persistent native-prefix protocol. The current bounded per-intersection complete-signed-set reader has counted overfetch/refetch and no performance claim; native process interruption restarts a full transfer. Keep per-term/partial-xorb proof limits, fixed CAS scope and existing elapsed-time/HTTP-trailer boundaries explicit.
~~~~

**Lines 404–404**

~~~~markdown
- [ ] Close remaining finite convenience/compatibility gaps: required sidecar/adapter/sparse-quant policy, bounded auxiliary/presets, mirror/endpoint/env precedence and gate/license diagnostics. Preserve explicit source authority/no silent numeric fallback and separate loader/runtime plans.
~~~~

**Lines 406–406**

~~~~markdown
- [ ] Qualify any future v1 migration only after separate authorization and proven reader/process quiescence; current removal supports originally protected roots only.
~~~~

**Lines 407–407**

~~~~markdown
- [ ] Extend cleanup only through separately qualified exact ownership/recovery protocols for dead retained staging/general orphans and any later eviction policy. Current exact-snapshot operation deliberately retains them and every stable lock inode.
~~~~

**Lines 408–408**

~~~~markdown
- [ ] Any additional lazy/raw-path/mmap consumer must adopt managed descriptor/open ownership or remain externally quiescent during removal. Observed named cache charge is not a process-global final-close/physical-free-space attestation.
~~~~

**Lines 409–409**

~~~~markdown
- [ ] Qualify additional filesystem pairs before extending the explicit allowlist. Different st_dev values alone do not prove independent backing objects for Btrfs subvolumes, overlay/FUSE or virtualized views; retain branch-specific root device/mount checks and complete inode/ancestry exclusion. No unsupported pair gains acceptance from generic mocks.
~~~~

**Lines 410–410**

~~~~markdown
- [ ] Separately authorize any needed namespace-only hidden-bind/subvolume/overlay filesystem fixtures before mount operations. Real target-root/unselected inode aliases and hidden bind aliases in both directions remain NOT RUN; no host mounts, block devices, new privileges or live cache actions are implied. Current tests use already existing distinct filesystems only.
~~~~

**Lines 411–411**

~~~~markdown
- [ ] Independently qualify any broader local layouts, renamed/unnamespaced roots, missing mount identities or new destination-root creation before exposing support. Keep explicit selection, retained descriptors, bounded topology work and existing cancellation/rollback limits; real import still needs user-selected source/repository/commit/files and copying authorization.
~~~~

**Lines 412–412**

~~~~markdown
- [ ] Finish the finite provider/loader parity ledger beyond qualified HF/GitHub/HTTPS/OCI/GCS/versioned-S3/Azure/ModelScope and explicit local-HF import slices: remaining external-cache layouts/filesystems, additional model hubs, unversioned/broader S3/Azure/compatible profiles, native Xet and broader OCI profiles. Opaque serialization/plugin loaders remain separate runtime work.
~~~~

**Lines 413–413**

~~~~markdown
- [ ] Continue explicit source/license authorization and trusted-digest versus first-seen provenance through additional providers/loaders; acquisition grants no license or loading support.
~~~~

**Lines 417–417**

~~~~markdown
- [ ] ACQ-A12: complete the finite provider/loader parity inventory and independently qualify remaining supported paths.
~~~~

**Lines 430–430**

~~~~markdown
- [ ] Each remaining provider phase requires focused API/CLI/protocol/auth/version/range/offline/fault gates, full frozen isolated check.sh, relevant source validation, actual source-integrity and documentation/dependency/root README checks before signing.
~~~~

**Lines 434–434**

~~~~markdown
- [ ] Unsupported plans fail closed until qualified: unknown-length required/model-closure files, capped resumable transfers, unqualified external-cache layouts/filesystems and remaining cloud/OCI profiles, legacy-root removal and unmanaged-reader concurrent reclamation remain explicit above. Independent full HTTP trailer validation is unqualified under the documented decoder boundary, rather than a reason these payloads universally refuse. Independent component data grants no additional loader eligibility.
~~~~

**Lines 435–435**

~~~~markdown
- [ ] Linux local-filesystem semantics are qualified first. HTTP cancellation obeys configured whole-request bounds; old retained attempts consume real quota and arbitrary same-UID mutation remains outside descriptor isolation.
~~~~


### [.agents/plans/model-execution/TODO.md](../plans/model-execution/TODO.md) — 79 unresolved markers

**Lines 81–87**

~~~~markdown
- [ ] Phase 3 (ROCm): replace whole reference row-wise/attention
  stages (normalization, gated activations, QK/rotary, attention) with mature
  native implementations where available. Re-profile actual V2 C8 and cached
  decode, share compatible independent rows, keep actual state/position
  semantics. Then amortize large dispatch regions using existing graph/actor
  ownership, including device-resident sampling. No scalar microtuning
  campaign or general speculative provider registry.
~~~~

**Lines 88–92**

~~~~markdown
- [ ] Phase 4 (CUDA): connect existing owned cuBLAS/cuBLASLt provider to ordinary
  resident projection/compatible dense products, not just reference diagnostics.
  Preserve existing native tensor-core/CUTLASS primitives; resolve fused lane
  divergence, and use FlashAttention/FlashInfer where appropriate. Benchmark
  actual correctness and C1/C8 on physical CUDA before wider selection.
~~~~

**Lines 93–98**

~~~~markdown
- [ ] Phase 5 (Vulkan): evaluate reusable mature ggml Vulkan cooperative-matrix GEMM
  kernels (`VK_KHR_cooperative_matrix`, vendor cooperative-matrix2 where
  supported) under existing owned command/buffer lifecycle. There is no single
  portable cuBLAS-equivalent library; inspect actual BF16/FP32 capability,
  licensing, layouts and integration cost before selecting. Keep scalar and
  existing shader primitives. Do not silently cast or adopt a second engine.
~~~~

**Lines 99–100**

~~~~markdown
- [ ] Phase 3 (ROCm): restore a current matched vLLM comparison when its pinned local closure is
  available. No new download or unrelated dependency repin just for evidence.
~~~~

**Lines 674–684**

~~~~markdown
- [ ] Establish independent large-CUDA distribution/state accuracy (session task44)
  under its declared numerical contract. Three matching IDs/112 finite domains
  and CPU-logit NMSE0.124241/0.090048/0.043837 are diagnostic, not a post-hoc
  tolerance grant. Preserve exact small-model references and original failures.
  The existing large composition diagnostic now reuses the independently accepted
  source-call archive reader to measure full-vocabulary and all112 state-domain
  cross-profile error together; CUDA Conv3 is compared explicitly with the retained
  history tail of original Conv4, while recurrent/KV retain their logical mappings.
  Results remain `same_contract_reference=false`/`full_reference_acceptance=false`:
  no27B CUDA-contract oracle exists on l5, and the retained22,126,120,608B original
  archive needed for the physical rerun is no longer present on this host.
~~~~

**Lines 747–748**

~~~~markdown
- [ ] Localize ROCm2B C4 warmup and C16/32/64 warmup/timed divergence (task39),
  without repinning expected IDs or attributing an unlocalized failure to kernels.
~~~~

**Lines 749–764**

~~~~markdown
- [~] Close ordinary27B HIP HTTP504 and populated1K/2K references/serving (task11).
  The historical504 still predates resident75GiB routing, but the20 Nov exact-current
  liveness rerun is now blocked before HIP allocation by the X2 runtime closure, not by
  ZeroGravity serving: both installed clr7.2.4 roots require an LLVM22 library needing
  GLIBC_2.44 while the locked dev shell is glibc2.42. Exact `HIP_PATH/ROCM_PATH` does
  not bridge that ABI mismatch. The temporary reference-free HTTP probe compiled under
  the accepted opt3/codegen16/non-LTO test profile and private child-net isolation was
  accepted, but runtime loading refused before GPU allocation. X2 stayed at the exact
 274,448,384B/0% baseline and was restored to `b599db3`; probe/bundle removed. Repair
  the X2 ROCm environment separately before another27B/ROCm2B run. Update24Nov:
  process-local compatible loader restores actual ROCm SDK execution for all three
  models; see CURRENT_HANDOFF and CURRENT_BASELINE. HTTP/populated context
  qualification and missing approved 66-call/1K/2K references remain separate.
  R10 audit is complete and mixed; R11 remains staged/unlaunched. Review its older
  binary/budget before any fresh successor. Both context captures remain interrupted;
  the waived12GiB host-available threshold is not a current blocker.
~~~~

**Lines 765–793**

~~~~markdown
- [ ] Continue genuine shared inter-session batching beyond current B8 decode/chunked
  prefill cohorts. Aligned prompt chunks up to the model's admitted chunk size share
  projection GEMMs while retaining private roots; B8 decode reuses the qualified B4
  stateful kernels as two groups. Dynamic grouping remains open, but do **not**
  resurrect the rejected global-active/native-B16 experiment as the next step: a
  same-path0.8B B16 measurement was slightly worse than two B8 groups
  (78.389 vs79.380 tok/s), while2B B16 regressed catastrophically (~44.9s/~42.0s
  warm waves). `MAX_ACTIVE_PER_WORKER=8` and native4|8 therefore remain deliberate
  until a materially different architecture/measurement justifies revisiting them.
  Bounded queueing already handles larger offered populations without widening the
  resident/native ceiling: same non-LTO2B47+3 run stayed exact/native-width8 at
  offered8/16/32 with66.24/68.10/69.06 output tok/s respectively;16 and32 are
  simply two/four queued B8 groups, not hidden B16/B32 execution.
  **21 Nov mixed-prefill grouping candidate rejected.** Current source already admits
  reduction-compatible mixed prompt chunks (for example48/49) into one native B8 and
  shares projection/GDN work, but FlashAttention only batches consecutive equal-row B4
  groups. A temporary source-neutral experiment kept scheduler/fairness/publication
  order unchanged, stably clustered only the private physical lanes by prompt-row count,
  restored logical session/readback order before publication, and let the existing B4
  attention kernel consume every equal-row quartet with per-lane fallback for leftovers.
  It added no GPU allocation or kernel. The exact public-SDK interleaved
  [49,48,49,48,49,48,49,48] +3-output cell remained singleton-exact and native width8,
  but matched opt1/codegen64/non-LTO measured299.985ms on unchanged source versus
  324.231ms and313.585ms with the candidate (~8.1% and~4.5% slower). The Qwen-specific
  wiring/probe was removed, but the allocation-free lane regroup/equal-group primitive is
  retained in `performance_candidates::LaneRegroupPlan` for Qwen2B/27B, Gemma4 and other
  shapes. Dynamic grouping remains open; do not select this candidate for the measured
  .8B cell, but do not treat that loss as a global-model rejection. Move next to the admitted populated-context
  ladder on L5:4K first, then8K/longer-fit cells, reusing the existing64-row contract.
~~~~

**Lines 794–796**

~~~~markdown
- [ ] Optimize measured CUDA2B/HIP hot paths and run controlled comparisons after
  relevant correctness. Existing short eager/direct-API rates are not general
  HTTP/context/graph performance or proof of vLLM superiority.
~~~~

**Lines 1247–1255**

~~~~markdown
- [ ] Resolve ROCm2B batch-dependent output mismatch. Retained
  `qwen-rocm-benchmark-3gteresx/vllm-2b-summary-r1.json` confirms C4 warmup
  wave0/request_index3 differs, despite0/8 timed mismatches. C16/C32/C64 have
  2/32,2/64,6/128 timed mismatches, plus1/1/3 warmup mismatches. Expected
  [95726,26516,220], actual[95726,98019,271]. Warmup failures count; earlier
  “C1/C4/C8 match” shorthand was incorrect. Original summaries/logs unchanged.
  Failed run/source bytes/shutdown/original retirement retained. Cause unlocalized;
  not established as a ROCm kernel bug. Do not repin expected IDs or count these
  cells as accepted matched speedups.
~~~~

**Lines 1295–1298**

~~~~markdown
- [ ] Complete actual27B ordinary HIP/SDK/serve, not the finite two-position route.
  Authenticated `Qwen/Qwen3.8-27B` geometry now drives the same ordinary template:
  64layers,112domains, untied head,851text roles/18shards. Finite CUDA profiles
  remain closed; source geometry grants no execution or numerical acceptance.
~~~~

**Lines 1299–1304**

~~~~markdown
- [ ] Qualify bounded weight streaming and source/physical final readers.
  Whole-source BF16 storage is priced before upload; use existing source-page
  transport when resident weights plus workspace/roots/native owners exceed the
  default24GiB total including4GiB context. Explicit residency uses the separately
  admitted75GiB total; host ledger remains16GiB. Original
  source extent is separate from resident allocation; no float kernels changed.
~~~~

**Lines 1305–1337**

~~~~markdown
- [ ] Independently audit the original CPU capture before native comparison.
  Existing approved source/image,66 genuine calls/two repeats, full64layers,
  fresh47/48/49/50 and uniform/ragged512→514. Capture completed:66 calls,
  19848 records,22126120608 raw bytes; contemporaneous foreground/group/container/
  EOF retirement recorded before owner receipt. Independent audit now passes
  all6 cases/6533182464 float words/851 roles, full digests, natural feedback and
  repeats. Accepted manifest`f226ad21`; raw`d9690f46`. No HIP parity, serving or
  CUDA-oracle claim follows. Accepted archive staged on x2 with full stream hashes.
  NativeR4 observed child exited0:66calls/6533182464 exact words, full vocabulary,
  natural feedback, independent roots and source/native final readers independently
  audited in `native-observed-reviewed-r4.json`. The supervisor then failed its
  stale descriptor-proxy PID/executable guard, before x2 rebooted. Observer-free
  never started; original terminal/results are absent. Preserve this failed
  campaign and limited observed acceptance; no full-R4/serving acceptance.
  Observer-free-only continuation keeps frozenR3/references; no observed rerun.
  Fresh guard pins boot/PID/starttime/executable without changing device/KFD/busy
  or12GiB checks. R6 staged but superseded unexecuted after
  review: mainPID0 did not ensure descendant retirement; post-start errors could
  skip watcher attachment. Watch`566a9940` cancelled. R7 reviewed/fixed/staged;
  pure/output controls and four stubbed-OS failure injections pass: launch
  timeout, receipt failure, transient status and interrupted wait. Observation
  survives these errors; signals stop only the exact owned invocation; attempted/
  indeterminate scopes are not labelled unrun. R7 manifest`730bcdeb`; watch
  `e89ad3c8` succeeded. Fresh guards/pins/unit/path checks passed with13650759680B
  available. Observer-free R7 launched2026-11-05T12:41:41Z, invocation
  `a31ca21457a24796ad21fb1ad0e3798e`. Only `state-parity`, unchangedR3/reference.
  Original terminal2026-11-05T14:37:14Z proves nonce/invocation, absent owned
  cgroup/KFD[]/busy0 and no supervision errors. Independent
  `native-state-reviewed-r7.json` passes all66calls/3137737728 state-logit words,
  611450880 KV-tail words, independent roots/natural feedback/final owners.
  Job2666 completed; monitor2667 stopped after retrieval/audit. R4 supervisor
  failure and missing original terminal remain; no full-R4 acceptance. Do not
  rerun observed numerics, duplicate launch/restage R7 or revive R6.
~~~~

**Lines 1338–1345**

~~~~markdown
- [ ] Run actual-source admission, full-vocabulary/state parity, observer-free
  execution, cancellation/replay/final-reader and ordinary serving checks; then
  the complete assembled gate. Host tests/strict feature checks pass15 tests,
  including retained finite-source boundaries and exact managed manifest.
  Two earlier host failures (stale helper argument/new frontier assertion) are
  retained and corrected; call-free frontier stays0, runtime calls own extents.
  Evidence: XDG state `qwen-ordinary-large-rHMls4mk`; implementation integrated
  by`4f1e827`, with HTTP/context exceptions preserved above.
~~~~

**Lines 1346–1429**

~~~~markdown
- [ ] Finish additive BF16/F32 GDN source storage qualification. Canonical host
  and x2 hostR2 exposed the same exact-type refusal: all48 original large-source
  `A_log` and GDN norm tensors are BF16; small-source F32 remains required.
  Shared CPU-source GDN descriptors now accept both independently, retaining
  original dtype/bytes and exact widening through existing upload; CUDA's F32
  contract stays closed. Model-source validation still requires its pinned types.
  Shared descriptor/ordinary-plan/storage regressions and strict checks pass;
  R7 release diagnostics/SDK built, but actual-source final-reader test failed
  on the tokenizer's old single-source pin. R6's test-helper type error remains.
  Tokenizer JSON differs only by seven added-token entries (33 total; max21B):
  source-specific BPE/automata accounting now shares the same authenticated
  NFC/Split/ByteLevel machinery and global owner; original26-token pricing stays.
  R9 assembled validation passes:22 tokenizer plus8 owner-accounting tests per
  exact source, strict HIP/CUDA consumers, release diagnostics/SDK and actual27B
  canonical source/plan/tokenizer/final-reader test. Small-source pricing stays
  unchanged; R8's incorrect special-token test expectation remains as a failure.
  Canonical514/active2 plan streams53786705920B original weights; host14720655712B,
  physical device15439853568B including4GiB context; estimates, not reservations.
  Repaired binariesR2 frozen unqualified. x2 host-capacityR3 passes all28 cells:
  contexts512/1K/2K active1/2/4/8 admitted as estimates;4K–32K refused under the
  original host/device caps. Source recheck and exact-invocation idle retirement
  pass. Frozen binariesR1 remain unqualified and predate both repairs.
  x2 hostR1 failed before source IO on noclobber lock opening; R2 source refusal
  retired its exact invocation/cgroup with KFD empty; both were host-only.
  Independent1K/2K original capture was interrupted by SIGTERM, child137; owned
  foreground/group/container/EOF retirement is recorded. Partial archive remains
  unqualified:104calls/3888records/all104 digests were the target, not a result.
  Host-private successorR2 preserved byte-identical inputs and R1's missing
  original-name absence evidence; it also received SIGTERM/child137, after11/104
  logged calls. Cleanup recorded; both partial archives remain unqualified.
  No active capture or accepted populated-context reference. A successor requires
  fresh identity and reviewed supervision. Shared scopes preserve small-model
  1K–16K and add large1K/2K schedules. R10 passes19 host reader controls/strict
  checks and release assembly; frozenR3 production bytes equal accepted hostR2.
  Capacity/refusal and reader preparation do not qualify actual contexts.
  Sequential C1 HTTP JSON/SSE/disconnect/replay coverage is authored separately
  from concurrency; existing population scopes stay unchanged. R11 stopped on
  a test-only constant-assert lint; runtime guard corrected. R12 strict/release
  assembly passes (1 host test,8 physical scopes ignored); SDK-onlyR4 frozen.
  Only HTTP qualification support differs from frozenR3; production and native
  example bytes are unchanged. No new HTTP physical acceptance.
  R5 lifecycle/SDK/sequential-HTTP campaign and terminal retrieval/audit are
  prepared only:13 fresh scopes. Its original full-R4 gate cannot pass after
  interruption. Successor R8 consumes limited observedR4 + accepted observer-freeR7
  honestly; same nativeR3/SDK-onlyR4 and deadlines/limits. Exact-owned supervisor
  failure controls and attempted/indeterminate/unrun runner controls pass;
  manifest`e8a35f90`, never staged/launched. Review found inter-case descendant,
  signal-handler lock and unit-name signal races. R8 preserved; R9 preparation
  adds case-cgroup checks, lock-free flags and pidfd-bound cancellation. Eleven
  ownership/four supervision controls pass; independent review found no launch
  blocker. Fresh exact guard/pins/paths passed; R9 exclusively staged and launched
  once2026-11-05T16:20:39Z, invocation`26c8c471c5f44b63896052bae63d596e`,
  nonce`9753bbec952d432abcf9562a33ddf8c8`; manifest`05e6cd11`, supervisor2729,
  monitor2730. SDK-complete/disconnect and healthy/session-known/snapshot-known/
  pending-known/readback-known/snapshot-readback-known exited0 with pass markers.
  HTTP-sequential failed101/HTTP504 at `qwen_hip_http.rs:98`, unchanged180s deadline;
  303.74s is cell duration. Stable live snapshot binds exact invocation and the
  HTTP after-boundary runner-only cgroup/KFD[]/busy0. All13 scopes then completed;
  independent retrieval/audit verifies12 semantic scopes plus HTTP101/504,
  authentic original retirement and no supervision errors.52 ordered case events/
  26 runner-only boundaries checked; indeterminate/unrun empty. Disposition:
  `ordinary-semantic-disposition-r9.json`. Original auditor's clean-supervision
  predicate also required semantic process success; supplemental disposition
  separates successful retirement from HTTP failure without changing old bytes.
  Staging log's inherited uppercase R8 labels are stale; filenames/hashes are R9.
  Never manufacture R4 terminal evidence. Retained-service bytes include state/
  scratch; do not confuse them with weight residency.
  Main now implements shared strided projection output/grouped partial windows
  for the concrete HTTP failure. Float operations/reductions/BF16, page/full-K,
  owners/cancellation and budgets are unchanged; all direct/prepared callers
  updated. Eight host mapping/status/scratch/extent controls and strict HIP/CUDA/
  Vulkan consumers and corrected R14 release assembly pass. Preserve R13's
  test-only integer-inference failure. FrozenR5 executed in R10: all15 scopes
  attempted,14 exited0 and HTTP101/504 retained. Logs report66 observed plus66
  observer-free calls, zero mismatches; independent retrieval/audit remains open.
  Original terminal records retirement for invocation`6448edba84244c1b9eb339908a7a2618`,
  nonce`42cfc1bc2aa8407ba8b1ceb07143254c`, owned cgroup absent/KFD[]/busy0;
  output errors include `BrokenPipeError(32, 'Broken pipe')`. No clean-supervision,
  whole-campaign or numerical acceptance before independent disposition.
  R10 is finished, not live; never restage/relaunch or overwrite its artifacts.
  Grouping did not resolve HTTP504; localize measured latency without changing
  deadlines or budgets. Row grouping is not cross-session native batching.
  No measured speedup, populated-context acceptance, current full gate or signature.
  Owner requires shared, additive capabilities—not per-model forks.
~~~~

**Lines 1430–1569**

~~~~markdown
- [ ] Complete independent CUDA references on l5; Main owns implementation and
  execution, no new or resumed subagents. Authenticate runtime/provider/source/kernel identities; fresh
  captures do not replace missing legacy executable authority.
  XDG run `qwen-cuda-regeneration-tDW0EJ1I`: exact RTX3060 UUID/SM86, driver595.91.07;
  original retained image passed a CUDA tensor probe. Runtime-addressable memory
  is6088359936B, distinct from physical6GiB; initial equality assertion failed and
  is retained. Upstream vLLM image then failed LLM import on CUDA13 torchcodec.
  Existing CPU-codec image`f57d77f5` (no installation/download) runs both0.8B/2B:
  two actual47-token greedy requests/model produce `[264,264,264]`; source hashes/
  identities before/after and original foreground/group/container/EOF inspected.
  Earlier controller mount exposed guest-writable CID/receipts; review limitation
  retained with supplemental current-name absence, never replacement receipts.
  New ownerR4 mounts only payload/cache, keeps control files host-private and
  independently checks the original container name after cleanup.
  Torch2.13/cu129, vLLM0.31, Transformers5.17, Triton3.7.1; eager/V2/FA2/Triton-GDN.
  Full-vocabulary R1 refused callable serialization; R2's FP32-only assertion
  rejected native BF16. Deferred async exception reporting masked that first
  error as uncleared pending state, not a proven hook bypass/forward overlap.
  R3 snapshots public `compute_logits` native returns and actual input buffers,
  defers CPU drain, returns the same object and exactly restores hooks/methods.
  No head dtype, arithmetic, async or serialization-security changes. Three
  CPU-only controls pass; both models'48calls/16requests independently audited:
  fresh47–50 uniform/alphabet, two repeats, genuine feedback, all248320 vocabulary
  words/call,11919360BF16 words/23838720rawB per model; exact repeats/source pins/
  foreground-group-container-EOF retirement. Raw08`36350f39`,2b`f0b70f7a`;
  `full-vocabulary-{08,2b}-reviewed-r3.json` are discovery-only acceptance.
  Both genuinely unobserved16-request runs independently match public outputs
  and termination, with exact source/options/retirement verified:
  `unobserved-{08,2b}-reviewed-r1.json`. No plain full-vocabulary/state-bit claim.
  Bounded async transport passes host and independent actual-CUDA native-bit/
  stride/mutation controls:40MiB GPU arena +24MiB minimal snapshots,128MiB pinned
  host; nonblocking producer, independent GPU/host retirement. Both expanded
  captures completed and retired; `expanded-{08,2b}-reviewed-r1.json` audit every
  archived byte (5925722312 /10564249800B), all24 original layer/deferred-add
  boundaries and4032 physical state snapshots/model. Captured valid prefixes
  preserve continuity/repeat bits; live views are disjoint. All320 original text
  parameters associate with actual loaded packing/casts, and parameters/buffers
  match before/after. This producer casts18 original F32 GDN norms/model to BF16;
  native source contracts unchanged. Full vocabulary equals R3, public output/
  termination equals unobserved runs. These expanded captures are not full-reference
  passes: they omit actual context slot mapping, owner-boundary mutable metadata
  and classification inputs; consumer successors below extend that coverage.
  Constructor buffers are not independently regenerated.
  KV layout is head×token×(K256|V256), not a leading K/V axis. Fresh Mamba backing
  is invalid history, not assumed zero. Producer/native CUDA use history3;
  CPU/HIP history4 is separate. Slab padding/plain-internal invariance remain
  unclaimed. At this expanded-capture stage CUPTI was inspected only; later
  standalone lineage diagnostics below do not grant model-wide ABI, native CUDA,
  matched performance or later roadmap acceptance. R1 review found incomplete
  chunk-storage attribution, metadata-reference provenance and lifecycle order,
  not capture corruption. R2 strengthens all three; both actual metadata archives
  and9 adversarial controls pass; both full R2 audits passed. Review then found
  inventory subview order was not linked to archive logical offsets. R3 fixes
  that link and rejects a coordinated descriptor/inventory swap: both metadata
  archives and10 adversaries pass. `expanded-{08,2b}-reviewed-r3.json` rehash both
  unchanged native/vocabulary archives, bind792/1334 chunks and explicitly reuse
  pinned R2 byte/layer/source checks. R1/R2 remain immutable; current scoped
  acceptance is R3, not exhaustive effective-state/reference/native closure.
  Read-only R3 review found no remaining concrete order hole; not reexecution.
  Actual-consumer observer captures KV/context mapping, conv/recurrent arguments,
  gather/mask/scatter and classification CPU/GPU inputs;12 real-Torch CPU controls
  pass. 08 consumerR2 completed48calls/37848records/6873374472B with original
  foreground/group/container/name/EOF retirement; vocabulary/public outputs match.
  Independent08 consumer auditR3 now passes all48 calls/8432 events, actual
  classification/counters/mappings, conv mutation and gather/mask/chunk/scatter
  bytes/order. Earlier audits wrongly treated padded GPU schedules as active CPU
  lists and prompt progress as total decode progress. Source-derived corrections
  check full padded allocations/roles, not weakened expectations;19 adversaries
  pass. `consumer-08-reviewed-r3.json`; failed auditors/logs remain immutable.
  2B consumerR2 failed at step28 after9 public requests; partial5987762129B and
  original cleanup retained. R3 removed duplicate chunk-return copying but failed
  at step1. Both original tracebacks identify host-ring admission, not GPU or
  record-budget exhaustion. Neither failure has a complete archive/contemporaneous
  GPU-after receipt; later idle observations never backfill those omissions.
  GPU retirement was already independent of disk work. Corrected transport splits
  GPU snapshot admission from coordinator-owned host admission/copy; host leases
  retire after raw consumption, record credits after bounded manifest writes.
  Drain acknowledges both GPU and exact host retirement. First failure is latched
  before diagnostics; uncertainty fences/quarantines before logging and survives
  query/close races. No producer waits, async/options/cap changes or extra temporaries.
  Fresh2B R4 completed48calls/38102records/11209912072B; all original foreground/
  group/container/name/EOF retirement independently verified, GPU before/after41MiB.
  Full audit verifies source bytes, all24 layers/4032 states/8432 consumer events,
  exact vocabulary/public outputs, masks/conv/scatter and all transport frontiers.
  `consumer-2b-reviewed-r4.json`;288 real scatter links plus7 adversaries pass.
  GPU transport peak11034368B, host90827008B; final used/padding/pending0, no
  failure/unknown/quarantine. Limits remain40MiB GPU+24MiB minimal/128MiB host.
  Transport controls:24 host scopes/20000 allocator operations/600records;
  actual CUDA9 native-bit/stride/mutation records and3 stalled-writer records.
  Latest `tensor-archive-r6.py` / `expanded-consumer-observation-r7.py` fixes a
  subsequent minimal-error publication race with deterministic interleaving and
  callback-fallback controls. Successful GPU/producer paths are byte-identical;
  reuse physical success evidence explicitly, no model recapture or physical
  uncertainty claim for that delta. Original clean R4 capture remains unchanged.
  `consumer-boundaries-reviewed-r5.json` binds both scoped audits and controls.
  Raw K/V candidateR8 and audit bodyR4 now observe actual entry arguments and
  compare logical page mutation through the existing transport. Both actual
  layouts/24 adversaries and original-once/same-return wrapper controls pass;
  no new model execution, backing-padding or complete effective-readset acceptance.
  Native callback recorder/offline typed lineage/checked ELF decoder assembled
  separately as qualification support. Runtime-compatible build replaces the
  failed host-libc probe; all failed bytes/logs stay. R3 repairs rejected nested
  calls incorrectly popping outer frames;15 native host scopes/400 concurrent
  calls/eight wire adversaries pass. No callback CUDA/GIL/file IO/capacity waits.
  Standalone RTX3060 R3 independently verifies13 primitive cases/launches, three
  public-edge-associated executable images,5544 callback records, original
  foreground/group/container/name/EOF retirement and no inter-case synchronization
  in its driver trace. Cubin-derived extents match eight successful parameter
  queries plus three terminal invalid-index responses;34 metadata/eight query
  adversaries pass. `cuda-parameter-lineage-reviewed-r4.json`. Original CUDA-info
  metadata has two explicitly checked link formats; retain R1's decoder refusal.
  Earlier R2 control staging waits and unload-fixture failure remain recorded.
  Independent constructor regeneration now passes for both source configs:
  `constructors-lineage-reviewed-r3.json`,268435488 fresh original CUDA bytes,
  all30 named buffers/model before/after (25 unique,5 aliases/phase), complete
  fresh archive/source/config and original retirement. Accepted consumer raw-byte
  audits explicitly reused; no observed tensor bytes supplied as constructor input.
  Its24 launches/10 images have typed lineage, but **zero captured arguments**:
  producer-version129 was unsupported. These24 ENTRY gaps are permanent for R3.
  Earlier constructor R1 missing-config-context and R2 enum-check failures stay;
  both precede kernels and lack contemporaneous GPU-after receipts.
  Native argument recorder `cuda-lineage-r6/`, approved-runtime buildR5, now
  supports witnessed123/128/129 metadata, live typed handles, declared host extents,
  packed extra/Ex attributes and explicit unmarked threads.32 native scopes,
  400 concurrent calls,14 actual schemas,166 malformed metadata cases and8 wire
  adversaries pass on CPU; source/binary pins and original retirement inspected.
  Binary`22a8dcaa2bc33f6e96a5bb141c4f60a0631b3ff96999a78d682f9730b8309f5f`.
  Initial native parser refusal remains. No GPU/model run of the new recorder,
  worker bootstrap/event attribution integration or actual GPU argument acceptance.
  Candidate `argument-wire-r1.py` still points to R5 readers; prepare/test an
  explicit successor, not an in-place rewrite of executed capture/audit inputs.
  Paused here by owner; no next campaign launched. Resume combined model-wide
  argument/image observation, effective-state read/write and recurrent-axis proof,
  then native CUDA/serving. Native CUDA history3 already exists; do not impose
  CPU/HIP history4. No aggregate-field/model-ABI/full-reference/native acceptance.
  No retained package found in the inspected readable local/remote scopes;
  five local container housekeeping paths denied.
  Exact2B cubin and original FLA/causal-conv/FlashAttention qualification remain
  unavailable. No CPU substitution or source repin; accepted new discovery is
  not full CUDA reference or legacy-authority acceptance.
~~~~

**Lines 1573–1577**

~~~~markdown
- [ ] Extend populated contexts beyond accepted0.8B/2B HIP scopes to remaining
  CUDA/general27B routes; retain full-logit/state truth, cancellation and physical
  ownership at actual extents. The older32K refusal was a pre-streaming capacity result;
  current public2B32K load is admitted; execution is now localized to a reproducible
  >600s first-response performance boundary rather than an indeterminate external kill.
~~~~

**Lines 1578–1580**

~~~~markdown
- [ ] Extend/qualify serving concurrency through C1..C64 and general27B/CUDA;
  preserve missing-source/kernel/runtime refusals. Offered C64/active≤8 singleton
  execution is only a baseline, not native-batch acceptance.
~~~~

**Lines 1581–1584**

~~~~markdown
- [ ] Measure matched vLLM workloads as each implemented path passes its relevant
  correctness/capacity checks; do not wait for the whole roadmap or batching.
  Match workload/settings/concurrency, report losses and limits. Unqualified
  experiments stay exploratory; no superiority claim without matched measurements.
~~~~

**Lines 1588–1595**

~~~~markdown
- [ ] Owner requirement: independently configure queue depth, active admission,
  batch members and per-step token budget; enable genuine continuous batching,
  including all eligible admitted work when it fits. Remove the universal
  eight-session/singleton policy ceiling, not physical/source/kernel safeguards.
  Reuse shared cohorts, original flights/roots and the arbiter. Cross-session
  kernels/state mappings and partial cancellation must work before promotion;
  no metadata-only batching or silent relabeling of serial work. This is queued,
  not part of the current populated-context correctness implementation.
~~~~

**Lines 1606–1612**

~~~~markdown
- [ ] Owner30Oct explicitly resumes Qwen ROCm (HIP) on `pvl-x2` independently
  of wider CUDA reference gaps; approved2B/9B/27B model downloads into canonical
  XDG cache are allowed if needed. No subagents, source substitutions, SDK
  downloads or device-privilege changes. Current27B finite source/natural reports
  pass exact full64/fullV/all112-state with supervised exit/retirement in signed
  `825ade8`. Ordinary production/serving and general27B contexts remain open;
  bounded small-model HIP results follow below.
~~~~

**Lines 1613–1616**

~~~~markdown
- [ ] Implement ordinary HIP through shared operators/physical owners, not a
  backend flag on the CUDA plan. Declare actual numerical/storage contracts;
  generalize real model geometry beyond24 layers/42 domains for27B. Preserve
  source-priced admission, independent roots, flights and physical retirement.
~~~~

**Lines 1620–1621**

~~~~markdown
- [ ] Generalize actual geometry/backend execution for27B. Finite27B still
  pins2 positions/112 domains; it is not the ordinary small-model source path.
~~~~

**Lines 1646–1648**

~~~~markdown
- [ ] Qualify `.8B` then `2B` full model, cached continuation, faults, shutdown
  and exact resource fit before enabling an ordinary serving route. Long-context,
  CPU, HIP and Vulkan cells remain independent gates.
~~~~

**Lines 1652–1656**

~~~~markdown
- [~] Every existing/future registry target follows the shared
  [correctness policy](../numerics-and-context/CORRECTNESS_POLICY.md): declared
  per-op accuracy, pinned independent task/model quality, separate determinism
  and exact source/state/physical ownership. Registry metadata is no execution
  or model-access authorization; unsupported adapters remain unqualified.
~~~~

**Lines 1657–1661**

~~~~markdown
- [ ] Implement/qualify missing full-model/context/backend cells, beginning
  .8B47–50 then genuine long context on CPU/CUDA/ROCm/Vulkan. ROOT coordinates
  exclusively GPT-6.1 Sol xhigh lanes. Older Pi/Astra assignments and performance
  sequence below are superseded. The later30Oct mission resumes independent ROCm
  and correctness-gated timing; older blanket deferrals below are historical.
~~~~

**Lines 1686–1698**

~~~~markdown
- [~] Three passes executed on pvl-x2.
  Root causes of passes 1-2 fixed and verified on hardware
  (`06b89da` E0308, `c217c35` V parents, `e6ab074` census,
  `3941a3c` Gemma envelope). Pass 3: builds/admins/binds all PASS;
  h-known-qwen-reload + h-qwen-final-reader-unknown PASS (full
  semantic closures); three real failures preserved
  (h-gemma-feedback-unknown: session-state identity/policy/fence
  assert + `scope closes only between completed proposals`;
  g-resident: parity row-upload fixture contract vs legitimate PLE
  row dedup; v-resident: adapter validator requires a false-permit
  field the producer never emits — physical case itself passed);
  19 cases stopped as dependents. No product defect established
  (qualification-gated paths).
~~~~

**Lines 1702–1710**

~~~~markdown
- [~] Qwen natural2-position independent original comparison and Gemma learned
  PLE row-hit/H2D/full-state scope, retaining existing quotas/owners. ROOT reviewed
  Astra's complete next-unit design; two isolated Pi implementation/controller
  preparation lanes active, no GPU/reference execution before review. Qwen old
  teacher capture requires independently verified original fullV/prior/source
  join; otherwise fresh natural reference. Gemma reuses its genuine natural
  matrix and adds an actual-hit full35/fullV comparison plus final-owner scopes.
  No new public
  serving, source21 x2 artifact, general-fit or speed grant from source20 fixture.
~~~~

**Lines 1711–1714**

~~~~markdown
- [ ] Gemma PLE twofixture source draft and prepared controller need corrective
  review before merge/physical work. ROOT finds diagnosticD2H exactly1B gate
  instead of positive and regressed dataFD/bounds. Original valid source20
  math/lifetime results unchanged; new all-hit/final-reader scope unexecuted.
~~~~

**Lines 1719–1729**

~~~~markdown
- [ ] Close isolated source113 as one checked model unit, then actual x2 source/
  state/fault/lifetime qualification: Qwen full64/all112, Gemma resident20/all35/
  fullV/all15KV and observer-free natural feedback. Grouped isolated CPU/strict
  checks PASS:5strict configurations,907CPU library tests/2comparators,46explicitly
  ignored cases. Exact source113/P1/P2 and original400v4 base remain unchanged;
  contextual410 continuous/compiler reconciliation is reviewed and affected
  strict/166boundary/7compiler-owner checks pass with all jobs closed;21physical
  cases explicitly ignored. Later RMS3/source6 excluded. Gemma's finite actual
  Resident20 and Qwen full64/armed-producer fault results follow below; Qwen's
  final Pending/Snapshot reader fault coverage is now closed on frozen469 below.
  Keep current C8 assembly separate.
~~~~

**Lines 1730–1732**

~~~~markdown
- [ ] Reconcile frozen models113/P1 hooks and Gemma final-feedback fix with current
  shared CUDA readiness source, preserve arithmetic/backend boundaries, perform
  affected assembled checks and final complete gate before model merge acceptance.
~~~~

**Lines 1733–1738**

~~~~markdown
- [ ] Qwen27: qualify natural and temporal decode/prefill, residency/admitted
  prefetch, packed projections, eligible optimized GEMM/attention/recurrent GDN,
  hybrid prefix checkpoints and serving through shared engine primitives.
  Current native route requires streamed weights and finite teacherfrontiers0/1;
  shared64-lane CPU-order BF16 dot is not TensorCore/BLAS. Preserve source/cast/
  reduction profiles rather than inheriting small-model CUDA permits.
~~~~

**Lines 1739–1783**

~~~~markdown
- [ ] Gemma E2B final-reader ownership: source466 review finds Session/Snapshot
  drop `loaded` before final committed feedback's independent12-byte Output
  lease. Native/modules/managed-source guard can retire before a true final
  feedback release refusal. Pending already has safe ordering. Isolated Sol
  fix+fixture unit authorized: place `loaded` after inner+feedback; Session/
  Snapshot × healthy/exact final Output-lease refusal, Weak-only diagnostics and
  original actual service/native/source/quarantine owners. Preserve frozen466,
  completed Qwen P1 and readiness sources. Frozen469 ten-path fix+fixture passes
  grouped6control/4probe/all-target strict checks; independent Sol review and ROOT
  complete source/control/cost/archive review pass. ROOT authorized one bounded
  fresh admin/offlinebuild/four-case unit. Admin33537/offline GPU-free build73520
  CLOSED0; first Session-known28952 FAILED_CLOSED at semantic verifier's exact
  `device_with_six_roots_bytes` formula after native child exited0. Actual groups/
  one-reap/fullEOF/source35FD/cache-parent closure healthy. ROOT and independent
  Sol identify verifier-only omission: greedy12336B+six256B alignment guards
  missing from both exact expected totals; production sums/caps are correct.
  Corrected local verifier6controls PASS; new saved Session-known revalidation
  PASS without GPU replay, original FAILED receipts unchanged. ROOT reviewed
  full225line local collector delta/17pins and authorized remaining3 serial
  native cases, exact-known-error-only/source35/cache/binary/originalowner
  closure required. All4actual scopes now close with corrected local acceptance;
  ROOT independently verifies every report/source/cache/original+reader owner.
  Known:184→0modules/sourcecloseOK/model0/context4GiBdevice+2GiBhost. Fault:
  actual12Bfinalreader refusal precedes native retirement,184modules/617handles
  retained/full20GiBfenced/sourceclose refused. Every full35/30planes/fullV/
  one natural commit; original failed remote collectors and SSH1 remain explicit.
  No Session replay/Rust rebuild/serving or performance grant. Existing source466
  cache reused with only actual publication/activity locks writable; no second
  10GB import. Current11+118path isolated combined8package/alltarget strict and
  189CPU PASS/33explicitphysicalignored; original source/owner/EOF closures pass,
  first controller-only compilerfile cap failure preserved. Current12 differs
  only by test-fixture rustfmt outside118. ROOT now adopted models118 plus the
  source-bound2B cachedB4 unit together:173selected/172effective/46new paths,
  fresh mtimes, original README/lock and all unrelated source preserved. Current13
  has1629full files/492selected; combined strict/20sharedCPU/5ABI-source controls/
  authentic2B CPU inventory/reopen pass. Original excluded-vendor Cargo refusals
  remain failed, not executed vendor-test evidence. Optimized13 CLOSED0 and all
  original19physical scopes/healthy2Btrajectory PASS; original2B same-prior30calls/
  8112sections/fullstate/fullV exact. New2B fault global-vs-flight count mismatch
  is diagnosed; fixture-only15 correction builds, preserving every graph/source/
  fence guard. Current assembled phase and final
  assembled gate still required. Prior.8B measured checkpoint remains source11/12.
  Combined assembly results.
  Four-case report.
  Independent source evidence.
~~~~

**Lines 1784–1787**

~~~~markdown
- [ ] Gemma E2B/E4B: dependency-based shared-tail fast prefill, PLE row reuse,
  optimized packed projections and eligible attention, physical SWA reclamation.
  E2B15owners/20shared tail exists; E4B needs24/18 and a new functional route.
  Optional vLLM fast prefill is off by default, not a universal baseline feature.
~~~~

**Lines 1788–1789**

~~~~markdown
- [ ] Add functional Gemma E4B and12B routes;12B uses Unified geometry/MM mapping,
  no PLE/shared tail. Extend source/profile contracts before learned qualification.
~~~~

**Lines 1790–1796**

~~~~markdown
- [ ] Gemma26B-A4B: learned router/grouped experts/dense+expert composition, then
  admitted expert sharing/hotsets/EP. Gemma31B is dense; qualify its learned dense
  path separately. Reuse the existing single raw K projection for configured
  K=V layers, preserving distinct K/V transforms and state. Select the exact
  optimized-router tie/reduction/scale profile; full-softmax fallback is not an
  identical floating-point schedule. Geometry/analytic contracts alone do not
  implement either learned route.
~~~~

**Lines 1797–1800**

~~~~markdown
- [ ] Both families: shared continuous serving/temporal work, paged KV/COW,
  full-domain compatible prefix intersection and graph/capture ownership after
  actual source qualification. Shared target verifier/all-prefix rollback before
  Qwen trained MTP or Gemma assistant/KV sharing/sparse draft vocabulary.
~~~~

**Lines 1801–1805**

~~~~markdown
- [ ] Later explicit codec/distributed/MM/LoRA profiles: do not substitute lossy
  FP8/FP4/KV codecs for original BF16 or assume source/hardware eligibility.
  CUDA-only fusion and CDNA/gfx12 AITER are not unconditional gfx1151 gaps;
  current ROCm Triton speculative GDN needs actual eligibility checks. Generic
  MXFP4 registry falls back for dense Linear; DBO rejects relevant hybrid/MM routes.
~~~~

**Lines 1893–1898**

~~~~markdown
- [ ] Continue Qwen27B/Gemma source/runtime integration independently; merge
  contained completed units promptly. Use pvl-x2 Radeon8060S/gfx1151 ROCm/HIP
  for their GPU qualification/benchmarks; preserve exact HF source identities,
  source-derived complete allocations and full logits/state/cancellation gates.
  Existing remote Ollama GGUF inputs are not approved-HF equivalents. Current
  Vulkan/CUDA parity work is a separate implementation/qualification obligation.
~~~~

**Lines 1899–1901**

~~~~markdown
- [ ] Gemma follow-on: qualify tokenizer/ordinary learned route and broader
  prompt/cached geometry plus learned cancellation/state lifecycle through the
  same engine; selected diagnostic exactness alone grants no public route.
~~~~

**Lines 1902–1905**

~~~~markdown
- [ ] Independent follow-on units continue while short-HF validation runs:
  Qwen authentic H48/128 GDN producer and full64/112-domain schedule, plus Gemma
  family source-bound CPU completion. Preserve actual source geometry and exact
  numerical limits; unavailable learned arithmetic must not use legacy fallback.
~~~~

**Lines 1906–1908**

~~~~markdown
- [ ] Gemma trained Rust execution requires exact learned BF16 arithmetic and
  independent source/state comparison. Payload/tokenizer/original CPU oracle
  now exist; native/serving/performance acceptance remains separate.
~~~~

**Lines 1917–1920**

~~~~markdown
- [ ] Dense27B: bind full trained source/state through bounded row/layer windows;
  amortize immutable reads across compatible prompt/request rows, account
  activation spill and genuine upload readiness. MTP metadata/tensors alone
  do not qualify target or drafter runtime; expert hotsets do not sparsify it.
~~~~

**Lines 1921–1925**

~~~~markdown
- [ ] Gemma4 variants: source-real PLE, normalization/logit caps, SWA/global
  attention, configured K=V/cross-layer sharing, head256/512 and26B-A4B routing.
  Check graph padding cannot route experts or advance roots. Matching trained
  assistants may share target KV/embedding only under actual architecture maps;
  first qualify target execution and complete branch/precision ownership.
~~~~

**Lines 1926–1929**

~~~~markdown
- [ ] Later sparse Qwen: stable finite top-k/tie/block ordering, source-selected
  expert grouping, hyperconnections and deterministic learned PLE lookup lane;
  runtime n-gram proposals are a separate stage. Source-dependent sparse masks
  and reduction/cast profiles cannot be grafted onto ordinary dense attention.
~~~~

**Lines 1930–1934**

~~~~markdown
- [ ] Later DeepSeek: exact CSA/HCA/MLA/indexer/compressor/mHC/YOCO/Engram and
  checkpoint-matched DSpark/Markov/confidence contracts; typed dependency/liveness
  proof may eliminate dead prompt work only after every future state read is
  covered. Truncated bounded replay and perturbed sampling are explicit changed
  profiles, not exact defaults. Existing payload acquisition hold persists.
~~~~

**Lines 1935–1938**

~~~~markdown
- [ ] Later GLM: distinguish KDA from GDN, MLA/DSA index/open-tail state, mHC and
  own trained MTP/taps. CPU/GPU placement, cached-expert group order and FP16
  prefill must have declared arithmetic; geometry or source names do not prove
  interchangeability. Existing payload/device holds persist.
~~~~

**Lines 1940–1994**

~~~~markdown
- [ ] **Independent trained BF16/FLA checkpoint gate:** original
  Qwen3.8-27B@1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0 acquired
  through the existing bounded Rust cache pipeline; independent31-file/
  18-shard full hashes and1199-tensor header/index checks passed.
  Whole-model27B execution remains unqualified; bounded trained operator
  qualifications below do not establish full logits or112 StateRoot domains.
  Genuine original layer0 GDN reference now validates initial chunk64 and true
  cached continuation for teachers11/353,48 value heads and original BF16
  parameters. Plain/observed complete outputs, convolution/recurrent state,
  raw source readsets, full before/after source hashes and original owned
  retirement match. The serializer's V2 stride failure remains preserved;
  V3 raw-bit transport passes12 real PyTorch layouts. This is an independent
  public target. An isolated native composition now passes the authentic H48
  cached slice:804960 input and798720 output words, including the complete
  recurrent state, match independently before/after callers drop. The distinct
  BF16-gamma gated RMS slice also passes61632 intermediate,12288 final and
  24576 retained input words. Source hashes, original ownership, cancellation,
  real pressure and final reservations0 pass; strict/release builds and237
  focused tests pass. These operator changes remain isolated from MAIN;
  initial chunk64, native exp, complete GDN,112-domain state and full inference
  remain unqualified. A separate genuine public reference passes32 pure
  operator cases over16 roles,881856 raw BF16 input/output words, including
  complete248320-row head and17408-column down projection. The isolated
  native wide-role replay now matches all881856 words exactly; real pressure,
  pre-cancel, retained head/source after caller drop and final reservations0
  pass. Strict/release builds and17 source/relocation fixtures pass. External
  probes for other roles do not establish trained producers or full-model
  parity; these changes also remain isolated from MAIN.
  Six actual bounded native-provider startup/lifetime cases pass with the
  provider unloaded. Physical provider allocation/math admission remains open.
  Original
  Qwen3.5-2B@15852e8c16360a2fea060d615a32b45270f8a8fc is now acquired
  by the same bounded Rust pipeline; independent official Git/LFS full12-file
  hashes,632 headers/320 text shapes,596BF16+36F32 tensors pass. Its4.548GB
  shard remains above the default4GiB complete-source guard. Genuine pinned
  official CPU two-profile reference capture now passes archive and owned
  cleanup validation: tokens11/353, four full248320 logits,168 canonical
  domains and60 actual intermediates. Official BF16 loading has617 BF16
  parameters; restoring only36 original F32 tensors gives581 BF16+36 F32.
  The profiles differ in47/49 raw tensors per step despite identical argmax;
  neither establishes Rust2B HF numerical qualification. First capture's
  cleanup-name bug is preserved; the repaired fresh run proves actual nonce
  container absence and process reap. Original Qwen3.5-0.8B@2fc06364715b967f1860aea9cf38778875588b17 pinned
  Transformers5.17/Torch2.13 CPU captures passed archive validation:
  tokens11,353, full248320 logits/all42 domains, actual layer0 stages.
  Keep official473-BF16 loader and source437-BF16+36-F32 profiles distinct;
  first-token argmax321 vs198. Explicit new CPU64 diagnostic vs source-mixed
  reference remains UNQUALIFIED: embeddings/direct layer0 norms and both
  QKV projections exact; layer0 convolution state exact,41/42 domains still
  differ. Full248320 logits differ214245/214285 words, maxabs .22119140625/
  .1875. First token exposes conv/SiLU signedzero; second exposes GDN state.
  Complete pinned convolution, chunk64/cached recurrent, coefficients and
  gated RMS arithmetic, then original GPU/FLA/state/streaming gates before enabling
  trained HF execution or27B serving. User authorizes required runtime/model
  acquisition/testing; dated source-free slice restrictions below are history.
~~~~

**Lines 1995–2001**

~~~~markdown
- [ ] **Next trained reference gate:** qualify exact original HF BF16
  rounding/FLA order and independent source-trained logits/StateRoot;
  then bind the original profile to the existing per-op
  MathematicalBackend/SessionAuthority, with full physical
  device/SSD resource admission. Earlier source-free slices establish no
  full27B logits,112-domain state, native provider admission or GPU support;
  current owner authorization covers the required acquisition and testing.
~~~~

**Lines 2003–2010**

~~~~markdown
- [ ] **Following ME-P0 real trained integration:** fuse the
  accepted analytical original source operators into the ONE
  existing forward_token/MathematicalBackend and
  SessionAuthority execution schedule, qualify original HF BF16
  intermediate casts and FLA GDN accuracy, source-trained exact
  full logits and StateRoot, tokenizer, physical mutable-state
  memory accounting, cancellation and device completion. No
  SSD/physical inference claim follows from the diagnostic bridge.
~~~~

**Lines 2012–2018**

~~~~markdown
- [ ] **ME-P0 next after diagnostics:** wire the analytically verified
  Qwen original operators into the ONE existing forward-plan and
  TaskDriver/SessionAuthority (no second token loop or state issuer),
  with exact source BF16 intermediate cast and original FLA GDN kernels,
  real trusted tokenizer and full 248320 logits/42-or-more StateRoot
  domains against the pinned checkpoint. Only then begin
  SSD-resident streaming and measured Qwen3.8-27B physical throughput.
~~~~

**Lines 2020–2026**

~~~~markdown
- [ ] **ME-P0 next numerical stage:** source-exact original Qwen
  full attention, GDN/MLP, grouped Q/K→V replication, A_log/gate
  transformation and BF16 intermediate numerical schedule using the
  shared original TaskDriver/StateRoot. Qualify tiny independent
  analytic operator/state gates before real approved trained Qwen3.8
  27B original full-vocabulary reference. Measured original Qwen3.5
  2B CPU/GPU end-to-end performance also remains open.
~~~~

**Lines 2028–2035**

~~~~markdown
- [ ] **ME-P0 next:** bind source-approved original Qwen3.8 27B
  tensor/index/config and tokenizer to the existing mathematical engine
  with authentic head/axis geometry, strict trained CPU full logits/
  StateRoot, then source-verifiable immutable SSD extents with stable
  leased memory and measured decode/prefill. First finish measured
  Qwen3.5 2B CPU/approved GPU end-to-end performance and
  cancellation/physical readiness. Preserve no model/SDK downloads
  and held device launch policies without explicit authorization.
~~~~

**Lines 2037–2048**

~~~~markdown
- [ ] **ACTIVE NEXT: ME-P0 Qwen3.5 2B then Qwen3.8 27B.**
  Begin with original qualified Qwen3.5 .8B/2B CPU and resident GPU
  source/state oracles, inspect release source ownership and exact
  model-2B profiling on the permitted CPU/GPU devices. Keep tests
  bounded to existing 8K context now; 16K/32K extensions remain
  separate numerical gates. Qualify a stable single-model fast path,
  original tokenizer/math/KV/recurrent/StateRoot and actual device
  completion first. Then implement checked immutable source shard leases
  above the present 4GiB executable-source gate and Qwen3.8 27B
  complete trained CPU logits, including SSD streaming and quant-aware
  physical placement. Do not fetch weights or launch held Renoir/gfx90c
  hardware without the corresponding permission and source integrity.
~~~~

**Lines 2049–2057**

~~~~markdown
- [ ] **NEXT after Qwen: ME-P1 Gemma4 E2B→26B-A4B→31B**, independently
  source-bind tensors/reference semantics, PLE/MoE/dense quantization
  and trained CPU→GPU correctness before measured throughput
  optimization. **ME-P2 Qwen3.8 Flash-Next** follows with separately
  verified learned n-gram embedding/4B MTP heads, QSA/GDN/MoE and
  explicit Q2/~3.5bpw/Q4 source quant choices (NOT guessed files),
  speculative n-gram as a distinct runtime optimization. Other OSS,
  ASR/TTS/image/video/decision execution and multi-model router remain
  on their own deferred source-real gates.
~~~~

**Lines 2058–2063**

~~~~markdown
- [ ] Preserve direct one-model-per-process execution and one
  shared-instance-many-pipelines as **one Rust fabric**; the accepted
  YAML DAG/compiler and extension schemas add no inference backend.
  Latest21Oct owner request authorizes parallel recursive pipeline/profile
  design after independent engine research. It must not delay ME-P0/P1 or
  imply new trained context/media/router execution.
~~~~

**Lines 2065–2072**

~~~~markdown
- [ ] Enforce new owner priority: Qwen3.5 .8B regression, Qwen3.5
  2B highest-perf trained CPU/GPU, Qwen3.8 27B with SSD
  streaming/owned shard tensors, Gemma4 E2B, 26B-A4B, dense 31B,
  then Qwen3.8 Flash-Next QSA/GDN/MoE/gated residual/MTP
  and 51B learned n-gram embeddings. Every additional DeepSeek,
  GLM/Muse/Mistral/Llama family has a typed NOT IMPLEMENTED
  source contract until distinct trained CPU and GPU gates pass.
  No subfamily is generalized from metadata-only source support.
~~~~

**Lines 2073–2078**

~~~~markdown
- [ ] Treat single-model local GGUF execution and multi-model
  pipeline-qualified execution as two frontends of ONE TaskDriver/
  SourceIdentity/StateRoot fabric. A reusable model instance
  serves multiple routes without another model process or implicit
  source promotion; named replicas/adapters require separate
  physical memory admission.
~~~~

**Lines 2080–2095**

~~~~markdown
- [ ] After the accepted source-acquisition v1 cutover, use the existing
  pinned HF index-first path to inspect explicit original Qwen3.8-27B and
  Gemma4 26B-A4B index/config/tokenizer/chat/processor identities
  (see [acquisition v1 handoff](../model-acquisition/README.md)).
  Select an authorized exact revision and admitted local host. Do not treat
  synthetic >51GB transfer-plan tests, public model pages or cached bytes
  as authorization, execution-ready source, or trained CPU parity.
  2026-10-18 tokenless live metadata-only HF source observation resolved
  Qwen3.8-27B to `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`
  (18 shards, 55,586,113,293 selected bytes) and Gemma4-26B-A4B-it
  to `4d7ae4984b7db7de8f8457170b3f1a419ee76d52`
  (2 shards, 51,644,338,194 selected bytes); Gemma4 E2B-it commit
  `3e22461f65e89153144f8adb70e3b8c2cc9845a7`
  (1 unsharded safetensors, 10,278,848,001 selected bytes).
  No trained weights downloaded, locally owned, header-audited or
  model-execution-qualified yet.
~~~~

**Lines 2097–2097**

~~~~markdown
- [ ] Acquire/bind authorized original config/index/tokenizer/checkpoint identities and choose a capable admitted host. Large BF16 sources exceed current local RAM; current executable source limit stays 4 GiB.
~~~~

**Lines 2098–2104**

~~~~markdown
- [ ] Qualify real large (>4GiB) original trained shard payload on an
  authorized capable host using the already verified owned-shard
  implementation from 93a5622; independently measure resident
  allocation/physical RAM pressure and immutable lease lifetime. Next,
  develop the separately admitted SSD streaming extent architecture.
  The bounded source-only synthetic tests do not establish real large
  learned weight loading.
~~~~

**Lines 2105–2105**

~~~~markdown
- [ ] Qwen3.8 27B: original HF grouped-head ordering and Q/K replication 16→48, explicit converted-profile permutations, untied head and source-derived norm/RoPE geometry.
~~~~

**Lines 2106–2106**

~~~~markdown
- [ ] Gemma E2B/26B: shared GELU/multiply/scale/soft-cap, proportional RoPE, value norm/sliding/shared KV, E2B PLE/doubled FFN and 26B K=V/MoE branches; choose and qualify the exact reference-framework numerical schedule.
~~~~

**Lines 2107–2107**

~~~~markdown
- [ ] Qualify actual trained CPU operators/full vocabulary/state/nonzero replay, official tokenization/templates/reasoning/stop behavior, sessions, rollback/checkpoints and resource refusal.
~~~~

**Lines 2108–2108**

~~~~markdown
- [ ] Add the same source-bound forward plans and full physical GPU qualification per eligible device, preserving all state/actual-completion/cancellation guarantees.
~~~~

**Lines 2109–2109**

~~~~markdown
- [ ] Keep distinct future heavy/hybrid recipes (Qwen Flash-Next, GLM, MiMo and source-defined QSA/MLA/KDA/indexers/hyperconnections), and broaden dense/encoder/pooling/reranking/LoRA support only through independent model/codec/backend gates.
~~~~

**Lines 2110–2110**

~~~~markdown
- [ ] Preserve CPU-first family priorities; already-authorized parallel basics do not authorize model/SDK downloads or unsupported whole-model claims.
~~~~

**Lines 2112–2112**

~~~~markdown
- [ ] Bind explicit original HF versus converted Qwen source profiles. Conversion permutes grouped/tiled V projections/constants/convolution/output, uses -exp(A_log) and adds1 to non-GDN norms; GDN swish is SiLU while full-attention gating remains sigmoid.
~~~~

**Lines 2113–2113**

~~~~markdown
- [ ] Choose and independently qualify Gemma26B reference-framework routing/top-k/reduction/scale placement before production execution. Five analytic test-local routing cases are not trained source parity.
~~~~

**Lines 3684–3691**

~~~~markdown
- [ ] Qualify independently selected NanoJev-like calibrated decision
  model sources under the same model/source/cache/residency authority.
  Register finite choice/boolean/score or include/omit heads without
  assuming a specific research checkpoint. Validate calibrated
  probabilities and per-question source-bound state, batching, thresholds,
  accepted/rejected context records and decision-fallback semantics
  before enabling generic context-filter execution; pipeline check alone
  provides only typed static intent.
~~~~

**Lines 3697–3697**

~~~~markdown
- [ ] Real current-source qualification remains mandatory before production routing: .8B natural/current-root parity including cached 47→50, physical fault/cancellation/retirement and real-model shutdown, then .8B 512/1K/2K/4K/8K and 16K/32K. Admit 2B contexts only where the exact two-root plan fits.
~~~~

**Lines 3698–3698**

~~~~markdown
- [ ] Current physical blocker: the approved .8B source is now visible at `/home/pvl/.cache/abird/zerogravity/external-baselines/models/Qwen--Qwen3.5-0.8B/2fc06364715b967f1860aea9cf38778875588b17`; its 1,746,942,600-byte safetensor SHA256 is exactly `04b1c301231dd422b8860db31311ab2721511346a32cb1e079c4c4e5f1fe4696`. The transient `.agents/runs` native bundle was intentionally purged and no `cuda-kernels.json`/legacy `short-hf-native.json` bundle remains under the repo or user cache. Reacquire/regenerate the qualified native/reference bundle through the maintained capture path; do not restore deleted run directories or substitute another model source.
~~~~

**Lines 3714–3717**

~~~~markdown
- [ ] These are fit/admission results only. Physical correctness, continuation,
  replay, faults and state-history qualification remain required for each cell
  before production use; .8B real-model qualification is blocked on a qualified
  native/reference bundle, not on model-source visibility.
~~~~

**Lines 3733–3735**

~~~~markdown
- [ ] Real physical after-launch cancellation/fault and real-model shutdown
  remain required with the approved .8B source; source-free ownership success
  does not substitute for those device tests.
~~~~

**Lines 3749–3750**

~~~~markdown
- [ ] This provides the committed-root comparison surface for the real .8B
  47→50 parity harness; it does not substitute for the blocked physical run.
~~~~

**Lines 3772–3773**

~~~~markdown
- [ ] Execute both phases serially as soon as the approved .8B safetensor shard
  becomes visible to the connector. Do not download or substitute source.
~~~~

**Lines 3801–3810**

~~~~markdown
- [ ] Physical execution is currently blocked by provider identity, not code:
  the verifier is pinned to libcublas.so.12 bytes=105,140,976 SHA256
  d63456d94f1b8e2408a8a9538c8740eb474c9eba48a84f420b15e2e10fbcd048
  and libcublasLt.so.12 bytes=749,205,904 SHA256
  96ddf4891ca4f4ed1475966783a9185df2abd64a81c3cc713b96d036696fc541.
  Historical validated runs used
  /usr/local/lib/python3.12/dist-packages/nvidia/cublas/lib/, but those held
  DSOs are not visible through the current connector. The available Nix CUDA
  12.9.2.10 DSOs have different sizes/hashes. Do not repin or substitute them
  without an explicit source/provider qualification decision.
~~~~

**Lines 3928–3930**

~~~~markdown
- [ ] When the approved source tree, pinned native bundle payloads and pinned
  provider DSOs reappear, use this wrapper with a fresh evidence path; no manual
  phase ordering is required.
~~~~

**Lines 4903–4911**

~~~~markdown
- [ ] X2 remains physically dirty and the blocker is now the host KFD/process-
  lifetime cleanup boundary represented by owner `2406838`, not source opening,
  HTTP, or generic dotlink execution. Do not launch another27B **or ROCm2B** while
  that owner/VRAM remains. Do not implicitly reset the GPU, restart dotlink, kill
  an unseen host process, or force KFD reclamation. Running kernel is7.3.1 while
  the current-system kernel path points at7.3.3; that is an observation only, not
  evidence that a reboot/update fixes this incident. Any operator-level recovery
  should be explicit before model work resumes. Accepted66-call and populated1K/2K
  references remain absent; never fabricate/repin them.
~~~~


### [.agents/plans/multimodal/TODO.md](../plans/multimodal/TODO.md) — 27 unresolved markers

**Lines 12–22**

~~~~markdown
- [ ] Opt-in, source-granted context-filter outputs as bounded,
  immutable, typed hints to ASR, acoustic-first diarization,
  transcript-text and separately source-qualified speaker-preserving
  translation stages. Default only prior committed session user/assistant
  records; opt-in tool call/result, app system/developer prompts and
  explicitly authorized shared data permitted. Generic include/exclude
  and typed decision-model choices operate after source ownership checks.
  Unsupported original ASR prompt dialect refuses or uses a separately
  verified correction stage. Actual audio A/B WER/CER/DER/translation
  adequacy, timestamp/partial-output correctness, consent/authorization
  and canceled-turn isolation are required before any runtime claim.
~~~~

**Lines 24–31**

~~~~markdown
- [ ] Model each ASR (Qwen ASR/Whisper/Canary/Parakeet),
  diarization (separately sourced speaker recognition), LM, Qwen TTS,
  Qwen Image2.1 and future video as registered typed components,
  not separate per-node subprocesses. The ASR→speaker-tagged
  transcript→LLM→TTS single coherent pipeline must include rational
  clocks, codec validity, VAD, per-turn epochs, barge-in,
  bounded fanout/backpressure and correct playback ACK frontiers.
  Current TOML/audio/image graph skeleton has NO real media runtime.
~~~~

**Lines 32–36**

~~~~markdown
- [ ] Enable native progressive image output only at complete
  source-valid VAE/tile/frame boundaries; partial latents or
  unverified solver steps are never playable. Actual media clocks,
  atomic commit barriers, ROS/transport and tested streaming
  quality/latency require source-real gates.
~~~~

**Lines 38–38**

~~~~markdown
- [ ] Add typed sample/frame/codebook/latent/decision values, rational clocks, checked conversions and complete source-bound component/codec/processor assemblies.
~~~~

**Lines 39–39**

~~~~markdown
- [ ] Prove a small real non-token encoder/decision task and bounded iterative task through the existing engine before broad media claims.
~~~~

**Lines 40–40**

~~~~markdown
- [ ] Qualify CPU-first original ASR and non-AR TTS (Whisper/Parakeet and Kokoro candidates); then actual backend execution and measured streaming/audio quality.
~~~~

**Lines 41–41**

~~~~markdown
- [ ] Add depth-AR/RVQ speech, dual-AR codebook TTS and genuine codec-based duplex neural audio (including Qwen-Omni) where source/license/resources permit; mock codecs or text cascades do not establish native support.
~~~~

**Lines 42–42**

~~~~markdown
- [ ] Qualify seeded source-exact image flow/editing conditioner+VAE pipelines, then video temporal/3D-VAE/expert switching and synchronized AV commit groups; candidate families remain separately gated.
~~~~

**Lines 43–43**

~~~~markdown
- [ ] Keep ternary PTQ/Hadamard codecs, schema/calibration decision heads and specialist encoders as independent codec/model/task gates, not aliases of Qwen.
~~~~

**Lines 44–44**

~~~~markdown
- [ ] Implement RealtimeSession lanes/epochs, bounded media buffers, VAD/barge-in, presentation/ack/playback frontiers and transactional rollback without speculative/partial media leakage.
~~~~

**Lines 45–45**

~~~~markdown
- [ ] Add per-model transport capabilities and authorized hosted TaskService adapters with privacy/cost/idempotency/Unknown reconciliation; no duplicate local runtime or guessed realtime guarantees.
~~~~

**Lines 46–46**

~~~~markdown
- [ ] Complete MM-01..24 plus model-specific ASR/TTS/image/video/decision quality/clock/resource suites; no real media-model support follows from source-free mocks.
~~~~

**Lines 59–59**

~~~~markdown
- [ ] MM-D1/MM-01/MM-15: checked rational sample/frame/codec clocks, 44.1/48 kHz to 16/24 kHz and 24000/1001 or 30000/1001 FPS, long-duration remainder, discontinuity, overflow and bounded parser/buffer refusal.
~~~~

**Lines 60–60**

~~~~markdown
- [ ] MM-D2/MM-02: complete source/processor/codec/license assemblies; wrong conditioner/shape/mask, missing source transform and cross-session mutable leakage refuse before use.
~~~~

**Lines 61–61**

~~~~markdown
- [ ] MM-D3/MM-03..09: seeded solver/RNG and direct finite decision state, CTC/RNNT/TDT distinction, complete RVQ depth and joint AV commit; interrupted/stale device work never publishes or exposes media.
~~~~

**Lines 62–62**

~~~~markdown
- [ ] MM-ASR-01..12: actual frontend/window/blank/predictor/duration/language/timestamp/diarization source fixtures; offline-only checkpoint refuses streaming, malicious audio bounded before allocation, sessions isolated.
~~~~

**Lines 63–63**

~~~~markdown
- [ ] MM-TTS-01..12: non-AR phoneme/duration/vocoder and exact PCM count; incomplete 9/10 RVQ frame unplayable, codec-rate swap refused, overlap/voice generation and duplex/barge-in/state/lease boundaries correct.
~~~~

**Lines 64–64**

~~~~markdown
- [ ] MM-IMG-01..12: source conditioner/solver/CFG/seed/mask/VAE/color and generation-entrypoint traces; invalid source stage/crop/processor fails, cancel retains old root and unknown physical handles, source/license/hosted status truthful.
~~~~

**Lines 65–65**

~~~~markdown
- [ ] MM-VID-01..15: temporal VAE/latent crop/frame count, Wan timestep experts, source tile masks/rectangular attention, AV cross-attention/joint clocks, invalid GOP/backpressure, hot reload, cancellation and whole-assembly admission.
~~~~

**Lines 66–66**

~~~~markdown
- [ ] MM-DEC-01..06: finite and joint schema head options/order/null/refusal/calibration against original source; generated text is not source-equivalent, external egress refused without authorization.
~~~~

**Lines 67–67**

~~~~markdown
- [ ] MM-BON-01..08: PTQ1_0/PQ2_0 packed source scales/signs/padding plus mandatory 1024 Hadamard activation transform; wrong codec or plain-Q2 path refused, original scalar/CPU/device source parity separately qualified.
~~~~

**Lines 68–68**

~~~~markdown
- [ ] MM-13..17/MM-19: real per-session/turn/transport epochs; stale ACK ignored, duplicate reconnect cursor idempotent/authenticated/bounded, playback progress remains uncertain when missing, slow clients do not release device leases or grow queues indefinitely.
~~~~

**Lines 69–69**

~~~~markdown
- [ ] MM-18/MM-22/MM-23: ten speech/five video mixed jobs, fair shared physical admission and legal solver/frame yield; no per-packet catalog/router/global snapshot or silent precision/step downgrade, coupled output stays unplayable until both tracks valid.
~~~~

**Lines 70–70**

~~~~markdown
- [ ] MM-SRV-01..12/MM-20/21/24: exact versioned SDK/protocol/field/task conformance, hosted accepted-timeout Unknown/idempotent reconciliation, callback/signature/TTL/MIME bounds, malformed media/unauthorized egress/cross-tenant replay refused before side effects.
~~~~

**Lines 71–71**

~~~~markdown
- [ ] MM-I0 change-radius drill: small encoder/finite task and iterative mock use existing source/state/resource owners; no central router/worker rewrite or duplicate runtime; original Qwen gates stay unchanged.
~~~~

**Lines 72–72**

~~~~markdown
- [ ] Eligible source-real CPU→physical full-model→playable-stream→soft-QoS gates: exact licensed checkpoints/processors/codecs and original quality/state oracles, real negative cancellation/admission/replay tests and measured cold/warm tails/memory. Mock/reference/region/hosted results never promote another tier.
~~~~

**Lines 73–73**

~~~~markdown
- [ ] Resolve realtime section18 policy choices explicitly (media numeric policy, first sources, SLO/clock uncertainty, mid-turn changes, persistence, biometric/asset consent, hosted cost/egress, transport packaging and safe physical recovery). User examples do not establish released model IDs or grant execution/provider/download authority.
~~~~


### [.agents/plans/numerics-and-context/TODO.md](../plans/numerics-and-context/TODO.md) — 49 unresolved markers

**Lines 534–539**

~~~~markdown
- [ ] Implement genuine source-profile multi-row math and call-free lowering.
  Current GDN phases are `InitialOne`/`CachedOne`; attention/source records cap64.
  Pinned original multi-row GDN uses `torch.linalg.solve_triangular` on padded64
  systems, not the initial-one simplification. Preserve exact contracts and
  separate CPU-source BF16 residual/eager-attention semantics from CUDA math.
  Keep all prior accepted arithmetic/lifetime evidence; production stays disabled.
~~~~

**Lines 543–548**

~~~~markdown
- [ ] Owner30Oct expands completion to `pvl-l5` CUDA `.8B`/`2B` and `pvl-x2`
  ROCm `.8B`/`2B`/`27B`, genuine active contexts through supported fit, then
  C1/C4/C8/C16/C32/C64 serving correctness and matched vLLM optimization.
  Use x2 resources as needed; no routine permission prompts. The current27B
  finite unit is a prerequisite, not completion of ordinary/context/concurrency
  routes. Performance follows qualified cells; record every loss/unrun scope.
~~~~

**Lines 553–558**

~~~~markdown
- [ ] Recover the exact additional2B reference `TextCutlassBf16Sm80` image
  (3,965,472 bytes; identity in `cuda/native/cubin_manifests.rs`). Approved2B
  model bytes are already in the canonical cache and load successfully. No match
  among1,252 SM80 images extracted from retained cuBLAS/cuBLASLt/Torch/VLLM
  libraries, nor their accessible embedded ELF inventory. The remote project
  cache also has no matching-size candidate. No substitution/repin.
~~~~

**Lines 559–564**

~~~~markdown
- [ ] Obtain independently trusted active-context comparison inputs. The retained
  Rust reference is finite cached47→50 at512 capacity; the Python reference
  protocol is also finite, and requires retained original/capture-source,
  observer/collector and owner inputs not found in the current qualification
  root. Capacity-only48-token runs and analytic32K attention do not close this.
  Do not restore deleted runs or weaken the reference to manufacture acceptance.
~~~~

**Lines 565–568**

~~~~markdown
- [ ] Finish ROCm on `pvl-x2`, explicitly authorized30Oct without waiting for
  missing CUDA references. Reuse cached approved sources; owner permits needed
  2B/9B/27B downloads through canonical XDG cache. No SDK/device-privilege changes.
  Preserve completed CUDA lifecycle/source gates and separate wider numerical scopes.
~~~~

**Lines 591–592**

~~~~markdown
- [ ] Broader CUDA active-context/2B numerical coverage remains open; the CUDA
  checkpoint's final gate passed. ROCm continuation is active as directed above.
~~~~

**Lines 608–614**

~~~~markdown
- [ ] Resolve ordinary cuBLAS ownership before model math. Fresh phase 2 fails
  `cublasCreate_v2` status 1 with one denied `cuGetExportTable` call, UUID
  `6bd5fb6c5bf4e74a8987d93912fd9df9`; allocation attempts/charges are zero and
  observation buffers do not overflow. A legacy `cuGraphInstantiate` lookup
  also refuses. The existing model-free dynamic-row identity test reproduces
  the same startup failure, before any GEMM. No opaque table passthrough or
  preloading around admission.
~~~~

**Lines 615–618**

~~~~markdown
- [ ] Resume fresh ordinary comparison using the independently retained digest;
  then physical lifecycle/shutdown and admitted contexts. Production routing
  stays disabled. CUDA correctness precedes ROCm on `pvl-x2`; no subagents or
  benchmark campaign during this continuation.
~~~~

**Lines 652–654**

~~~~markdown
- [ ] Requalify cached47→48→49→50, affected faults/cancellation/retirement and
  real-model shutdown on the assembled source; then `.8B` 512/1K/2K/4K/8K and
  16K/32K. Admit `2B` contexts only where the exact two-root resource plan fits.
~~~~

**Lines 655–658**

~~~~markdown
- [ ] Run the final isolated `scripts/check.sh`, then the single serial matched
  campaign in
  `../serving-and-speculation/BENCHMARK_CAMPAIGN_2026-10-27.md`. No 27 Oct
  performance cell has run.
~~~~

**Lines 669–673**

~~~~markdown
- [ ] Remove remaining shared rotary fault assertions/fixed-prompt qualifier
  dependency and case-driven model/GDN/native dispatch. Implement genuine
  parameterized kernels and explicit numerical configurations; do not move or
  rename test whitelists into runtime descriptors. Original CUDA scalar/source
  facts stay Rust fixtures. Parametric source proof/design is available only.
~~~~

**Lines 677–686**

~~~~markdown
- [~] Sequential continuation after serving shutdown: fresh .8B prompt47/48/49/50
  originals, matching native profiles, real-model shutdown/faults, genuine long
  contexts/backend routes, then optimized matched C16 and27B. All23 approved
  model members freshly SHA-verified. New parameterized reference collector
  preserves original math/state/fullV/source validators and exact owner; source
  controls pass. Fresh47/48/49/50 original pairs now pass all24/all42, fullV,
  consumed priors and natural feedback;47 uses explicit separate postvalidation
  after its original controller report-size failure. Fresh complete packed
  v3 captures47–50 now pass; native comparisons remain pending. Native fresh48–50 remains
  unsupported until actual source/ABI/physical qualification.
~~~~

**Lines 692–694**

~~~~markdown
- [~] Apply [all-model correctness policy](CORRECTNESS_POLICY.md), based on
  pinned vLLM/llama.cpp. Sol6.1 xhigh owns every lane; this supersedes older
  Pi/Astra assignments and performance-first continuations below.
~~~~

**Lines 695–700**

~~~~markdown
- [ ] .8B initial47/48/49/50 and cached47→50: actual independent original and
  native outputs/state/physical closure. Original case5 strict FAIL101 remains;
  original N1/B4 QKVZ itself differs by six BF16 words at identical frontier47
  operands. The historical fifteen-transition archive lacks paired N1 frontier50;
  fresh47–50 original trajectories are now independently available. This explains
  the diagnostic, not native model acceptance.
~~~~

**Lines 701–705**

~~~~markdown
- [ ] Genuine .8B512/1024/2048/4096/8192, then16K/32K context/state/feedback;
  corresponding original BF16 CPU/ROCm/Vulkan full-model routes are missing.
  GGUF, synthetic operators and finite other-model proofs stay separate. Local
  Renoir/gfx90c launch/recovery hold remains. Performance and27B follow this
  correctness sequence; earlier finite timing facts remain historical.
~~~~

**Lines 707–713**

~~~~markdown
- [ ] **Next one short-HF unit, then matched benchmarks:** both approved .8B/2B
  sources, canonical/ragged47/57/62/67/72 plus genuine cached feedback, original
  active8/caps and existing worker. Stage generic bounded storage/liveness,
  checked runtime logical views/epochs, complete lowerer access/tap ownership and
  pooled primitive scratch; current metadata planner alone grants no alias/fit.
  Preserve old source/precision/native gates. Benchmark the whole qualified short
  matrix together, then extend long contexts; unrelated27B/Gemma continue.
~~~~

**Lines 715–715**

~~~~markdown
- [ ] Actualcanonical47projectiondiscovery1 completed original PLAIN+OBSERVED/source/ownedcleanup, but exactcommon state.recurrent.after call0 differs. NewOBS cumsum/KKT/H/output selections differacceptedreference1; selectionvariance is verified, specificcause unproved. Independent actual metadata identity/layout gate now passes separately; no fullmodel/packedargs/native/perf acceptance.
~~~~

**Lines 716–716**

~~~~markdown
- [ ] Qualify actual canonical short47input/max8output/EOS248046 and ragged57/62/67/72 prompts; exact source/profile/numerical schedule and original all24/all42/fullV/actual sampler. Extend genuine rotary/ABI/control coverage as required, then original worker/HTTP and C1/2/4/8. Old GGUF diagnostic shapes do not grant HF parity.
~~~~

**Lines 717–717**

~~~~markdown
- [ ] Distinct native2B source opening/2048wide ordinary+dualRMS/projection+6144MLP/head/970window tied table+4096B source-row actor. Actual2B prefill PTX differs in cumsum, scaled-dot KKT, recompute W/U and chunk output; matching GDN geometry and different cache keys are not an execution permit. Preserve actual reference3/source/relative5GiB boundary; derive physical fit, then same fullmodel/natural/fault/serving closure. Matched short benchmarks, long contexts and dense27B follow actual parity. End-to-end benchmarks remain unexecuted.
~~~~

**Lines 718–718**

~~~~markdown
- [ ] Investigate actual2B discovery2 PLAIN outlier: both lanes finish/source/ownedcleanup pass, OBSERVED matches acceptedreference3, but PLAIN first differs at call2/layer8 recurrent state and branch/residual, then consumed downstream/fullV. Conv.after is only its last3 cache rows; no proof all upstream conv outputs match. Ordinary layer0 RMS10boundaries/565248B match independently. Fullmodel discovery remains unaccepted; source audit confirms timing-based autotuning but no specific cause; unchanged PLAINrepeat1 completed with exact reproduction. Independent RMS leaf reference accepted after complete source/IR/byte gates (565248B boundaries/sourceembedding,4096B gamma); native2048 component physical comparisons/faults/fullgate20 now pass separately, explicit parent-failure boundary retained. Unchanged PLAINrepeat1 reproduces discovery2PLAIN exactly across113713152all42+11168652common+6117576final+565248RMS B, still548/21/3different records versusOBSERVED/reference3. Python-only actual selection2 diagnostic now completes18records/lane and strict source/common/final/prior/all42/RMS gates PASS despite four differing selected stageconfigs. Different configuration alone is insufficient; the earlier reproducible outlier remains unresolved.
~~~~

**Lines 721–721**

~~~~markdown
- [ ] Native continuation: genuine47 current-root/position/provider/cache coverage, cached2B histories and whole-model admission remain open. Isolated2B QKVZ/BA phase51 and actual six-node source prefix/GDN phase52 are accepted below; initial1/64 stops before GdnOut. Original full24 permit stays guarded.
~~~~

**Lines 722–722**

~~~~markdown
- [ ] Current original2B KKT causal leaf1:29frozenchildren/23pure guards, actual original source/ownedcleanup and ordinaryRMS565248B pass. Four KKT configs/two repeats produce bit-identical A/F32+Ai/BF16 (all72comparisons0), but every solve uses the SAME2warps/4stages selected kernel: this does not compare solve variants. Strict original model reference still FAILS; current run reproduces prior PLAIN outlier byte-for-byte131564628B/27files. No tuner override or failure waiver; KKT config alone does not explain this case. Physical allocator peak remains unknown.
~~~~

**Lines 723–723**

~~~~markdown
- [ ] Canonical47 failure narrowed in parallel: three separate PLAIN runs share the same all42 record differences; their OBSERVED lanes match acceptedreference1. Original late-H pair1 now reproduces oldbad PLAIN/good OBSERVED actual state bytes under healthy source/owned cleanup; original raw comparison remains FAIL. Inputs to H already differ: W59BF16words, U9BF16words (token34/head4); K/G/H0/history exact, Vnew=U, finalHT1002F32words differ. Later failed-pair solve-counterfactual1 directly captures equal operands/F32A and first solve.Ai differences, below. This earlier late-H capture alone did not localize the cause.
~~~~

**Lines 724–724**

~~~~markdown
- [ ] Post-recompute47 investigation: attempt1 observer refused copied callable-grid assumption; original caller uses literal(1,16), failed attempt/cleanup retained. Narrowv2/28guards/peer review +48files/33parents/21runtime/3historypins PASS; actual attempt2 source/owned cleanup and original pair/common/fullV/all42 gates PASS. Seven actual K/V/beta/G/Ai/W/U records872320B/lane all equal good35; neither lane reproduces badW/U, so no historical cause/fix. Same-input original solve/recompute counterfactual proposed to test the actual observed warp2/warp4 solve selections without production tuner mutation.
~~~~

**Lines 725–725**

~~~~markdown
- [ ] Repair47 exact arithmetic binding at existing provider/preparation seam; authenticate declared candidate and actual original code/ABI/launch histories, qualify finite owned solve variants, then connected full47+cached histories. Broad geometry/source equality cannot establish byte equivalence across two proven unequal solve implementations. No tuner forcing, expected-word selection, tolerance relaxation or failed-parent waiver; all18 GDN layer/eight-call identity coverage and fullmodel gates remain open.
~~~~

**Lines 727–727**

~~~~markdown
- [ ] Retired2B postAttention1 parent gate remains CLOSED: bothlanes healthy/newlayer0pairedboundaries/source/native14op PASS. Fresh L8 pair3 source/lifetimes/full37paired archives verified after22observer+5host-schema guards: dual/RMS/QKVZ/BA/fullConv/preparedQKV/coefficients/cumulativeG/KKT A exact; FIRST solveAi14BF16differences, W96/U17/H2048. Actualsolve configs2warps/3stages versus4/4 and fullPTX differ; currentHT hashes reproduce prior same-lane split. Its original fullV/all42 failures remain fatal, RMSPASS. Separate new original MLP/ACT reference success and fixedOBS-only component successor do not accept this failedparent. Retired same-input originalsolve/cross-recompute leaf3 PASS: same actual794640B inputs; solve2warps/3stages reproducesPLAIN and4/4 reproducesOBS, Ai14/W96/U17 word split exact under BOTH recompute8/3 and8/4, twice; all12 returned completePTX/cachehashes equal actualoriginal variants. This isolates that split to solve configuration; it does not repair the original parent or prove earlier uncaptured inputs. Previous72 comparisons did not vary solve configuration.
~~~~

**Lines 728–728**

~~~~markdown
- [ ] Next: all24/all42/fullV/feedback/cached47 histories and workerHTTP parity. New original2B MLP+activationcapture1 passes strict original common/fullV/prior/all42/RMS/postAttention/MLP reference gates and captures two6144activation packets; ROOTpostvalidation/native2B path/fullmodel remainpending. Earlier failed postAttention parents remainfailed; predeclared fixedOBSpost1 successor target unchanged. MatchedC1/2/4/8 benchmarks follow parity, then longcontexts and27B.
~~~~

**Lines 729–729**

~~~~markdown
- [ ] Post-parity overhead follow-up: consider immutable GdnArithmetic same-object pointer shortcut before structural fallback; source-valid review only, no measured speedup or code change.
~~~~

**Lines 885–890**

~~~~markdown
- [ ] Qualify original-row Q4_0_8x8 QKV/Z using existing source projection math
  against native fields6/38 before changing any equation. Native operands are
  CPU_REPACK Q4_0, distinct from Q8_0/Zen BLAS; actual inner Q4 dispatch is
  source-derived, not freshly instrumented. QKV M6144 needs separate closed
  geometry; do not raise tinyBLAS limits. Then compose existing convolution,
  SiLU and grouped Q/K normalization with retained raw6144×3 history.
~~~~

**Lines 891–896**

~~~~markdown
- [ ] Nonzero cached-prior, live producer, full temporal/default-bulk route/
  profile/state binding and default8K qualification. Retained native bulk64
  versus AR1 upstream inputs match2097152 words plus1572864 four-tap operand
  words; that comparison is not a Rust producer proof. Current
  Q/K/V/prior/Z remain explicit observations; gain lacks separate capture.
  Sliced CPU proofs confer no complete-model/vLLM or whole-host RAM qualification.
~~~~

**Lines 897–899**

~~~~markdown
- [ ] MAIN full formatting still flags only the preserved SiLU draft
  `resident_program/silu.rs:91`; bytes match its pre-existing checkpoint.
  Do not alter another owner's draft.
~~~~

**Lines 903–908**

~~~~markdown
- [ ] **vLLM parity remains open:** implement and physically qualify its own
  CUDA arithmetic/cast/layout profile; complete initial/continued prefill and
  cached-decode GDN under that profile; close full .8B/2B/27B logits/state and
  long-context gates, then matched default/warm/concurrent serving performance.
  Current GGUF CPU operator proofs and historical cold-eager TTFT observations
  establish no completion percentage, cross-engine equivalence or new speedup.
~~~~

**Lines 910–919**

~~~~markdown
- [ ] **Required three-model correctness gate (owner,20 Oct):** resolve
  independent original-source CPU8K GDN divergence for BOTH original
  Qwen3.5 models without loosening tolerances or changing declared source
  arithmetic. This is required alongside original Qwen3.8-27B correctness,
  matched serving/model benchmarks and shared-engine performance work.
  Short-context CPU/B1/native exactness and closed original-HF operator
  proofs do not close this independent GGUF long-context issue. Recover
  exact raw teacher prefixes and pinned native arithmetic/chunk settings,
  locate the first state/stage discrepancy, then rerun independent long
  context and the accepted short corpus after any implementation fix.
~~~~

**Lines 920–920**

~~~~markdown
- [ ] Extend genuine resident GPU context/corpus qualification beyond the currently accepted 512 consumed tokens, including capacity/window/padding/replay boundaries.
~~~~

**Lines 921–921**

~~~~markdown
- [ ] Run genuine whole-model 16K/32K CPU/GPU gates only after prerequisites; functional 8K success is not independent long-context numerical closure.
~~~~

**Lines 922–922**

~~~~markdown
- [ ] Keep internal full-logit/all-state/trace bit parity distinct from independent native-source stage/top20 comparisons, including the known 2B 56 signed-zero distinctions.
~~~~

**Lines 923–923**

~~~~markdown
- [ ] Broaden prompts/tokenizers/precision profiles and sampling quality only with independent oracles, exact history and explicit qualified ranges.
~~~~

**Lines 925–925**

~~~~markdown
- [ ] Diagnose identical teacher-forced prefixes512/1024/2048/4096/8192 against pinned llama.cpp0.6.0/build11429 `d812350`, CPU/F32 KV and Flash Attention off. Fix the physical microbatch/chunk/arithmetic schedule and compare per-head GDN state, gain/beta, quantization and AR-versus-bulk recurrence; distinguish replay correctness from arithmetic.
~~~~

**Lines 926–926**

~~~~markdown
- [ ] Compare native ubatch1/64/256/512 only within bounded feasible runs; the historical full ubatch1 capture did not finish. Use raw token IDs to avoid incomplete UTF-8 HTTP rows. Recheck short corpus, layers, state and actual8K after any fix; retain original functional0.45 shared-top20-logprob/0.05 top-token-probability limits and strict source-bit operator gates.
~~~~

**Lines 1286–1286**

~~~~markdown
- [ ] Naturalfeedback/current-root publication and serving integration through originalSession/ResidentProgram. Forced353 diagnostics remainfrontier0/unpublished; candidate IDs[198,599,8,72,18] were not fed back. Separate source-staged per-call binding/publication unit in progress. No serving/performance/fullmodel-default-route acceptance; newimplementation remainsISOonly.
~~~~

**Lines 1288–1288**

~~~~markdown
- [ ] Native2B original-source all-state/natural-feedback qualification and serving integration. Reference success does not establish Zerogravity inference, a driver hard cap or unknown transient highwater; actual baseline is separate from requested relative memory. Benchmarks/longcontexts/27B runtime remain OPEN.
~~~~

**Lines 1303–1303**

~~~~markdown
- [ ] Real current-source qualification remains mandatory before production routing: .8B natural/current-root parity including cached 47→50, physical fault/cancellation/retirement and real-model shutdown, then .8B 512/1K/2K/4K/8K and 16K/32K. Admit 2B contexts only where the exact two-root plan fits.
~~~~

**Lines 1304–1304**

~~~~markdown
- [ ] Current physical blocker: the approved .8B source is now visible at `/home/pvl/.cache/abird/zerogravity/external-baselines/models/Qwen--Qwen3.5-0.8B/2fc06364715b967f1860aea9cf38778875588b17`; its 1,746,942,600-byte safetensor SHA256 is exactly `04b1c301231dd422b8860db31311ab2721511346a32cb1e079c4c4e5f1fe4696`. The transient `.agents/runs` native bundle was intentionally purged and no `cuda-kernels.json`/legacy `short-hf-native.json` bundle remains under the repo or user cache. Reacquire/regenerate the qualified native/reference bundle through the maintained capture path; do not restore deleted run directories or substitute another model source.
~~~~

**Lines 1319–1322**

~~~~markdown
- [ ] These are fit/admission results only. Physical correctness, continuation,
  replay, faults and state-history qualification remain required for each cell
  before production use; .8B real-model qualification is blocked on a qualified
  native/reference bundle, not on model-source visibility.
~~~~

**Lines 1338–1340**

~~~~markdown
- [ ] Real physical after-launch cancellation/fault and real-model shutdown
  remain required with the approved .8B source; source-free ownership success
  does not substitute for those device tests.
~~~~

**Lines 1354–1355**

~~~~markdown
- [ ] This provides the committed-root comparison surface for the real .8B
  47→50 parity harness; it does not substitute for the blocked physical run.
~~~~

**Lines 1377–1378**

~~~~markdown
- [ ] Execute both phases serially as soon as the approved .8B safetensor shard
  becomes visible to the connector. Do not download or substitute source.
~~~~

**Lines 1404–1413**

~~~~markdown
- [ ] Physical execution is currently blocked by provider identity, not code:
  the verifier is pinned to libcublas.so.12 bytes=105,140,976 SHA256
  d63456d94f1b8e2408a8a9538c8740eb474c9eba48a84f420b15e2e10fbcd048
  and libcublasLt.so.12 bytes=749,205,904 SHA256
  96ddf4891ca4f4ed1475966783a9185df2abd64a81c3cc713b96d036696fc541.
  Historical validated runs used
  /usr/local/lib/python3.12/dist-packages/nvidia/cublas/lib/, but those held
  DSOs are not visible through the current connector. The available Nix CUDA
  12.9.2.10 DSOs have different sizes/hashes. Do not repin or substitute them
  without an explicit source/provider qualification decision.
~~~~

**Lines 1531–1533**

~~~~markdown
- [ ] When the approved source tree, pinned native bundle payloads and pinned
  provider DSOs reappear, use this wrapper with a fresh evidence path; no manual
  phase ordering is required.
~~~~


### [.agents/plans/pipeline-composition/FORMATS.md](../plans/pipeline-composition/FORMATS.md) — 6 unresolved markers

**Lines 67–70**

~~~~markdown
- [ ] Pin the exact released variant: original container metadata, byte order,
  block sizes, codes, scales/offsets, exceptional values and source hashes.
  Add recognition only with independently checked codec fixtures and bounded
  malformed/truncated/overflow refusals. Unknown variants remain Unsupported.
~~~~

**Lines 71–74**

~~~~markdown
- [ ] Declare each operator's weight/input/output layout, activation coding,
  accumulation/reduction, rounding/cast and geometry contract before measuring.
  Preserve exact existing source-bit limits; a different contract is separately
  qualified rather than silently substituting dequantized F32 or BF16.
~~~~

**Lines 75–77**

~~~~markdown
- [ ] Validate immutable source backing/identity/ranges, actual producer and
  TeacherSpan/SSA/state joins, ordered matrix segments and complete outputs.
  Descriptors and folder names mint no source owner, root or native permit.
~~~~

**Lines 78–81**

~~~~markdown
- [ ] Qualify CPU, CUDA, HIP and Vulkan independently using actual original
  source and the selected compiler/SDK/driver/device/source/ABI identities.
  Exercise finite status, source refusals, cancellation, resource accounting,
  completion/fences, quarantine and healthy retirement. Keep gfx90c on hold.
~~~~

**Lines 82–84**

~~~~markdown
- [ ] Qualify each complete model/checkpoint/profile/context against its pinned
  independent reference with declared model accuracy and state checks; operator
  fixtures do not fill missing model cells. Preserve failed/unsupported records.
~~~~

**Lines 85–87**

~~~~markdown
- [ ] ROOT reviews assembled source, affected tests and final `scripts/check.sh`.
  Moving identical arithmetic changes source/type/binary/cache identity where
  applicable; do not attach old physical evidence to a renamed binary.
~~~~


### [.agents/plans/pipeline-composition/TODO.md](../plans/pipeline-composition/TODO.md) — 55 unresolved markers

**Lines 491–510**

~~~~markdown
- [ ] Extend qualified physical interop, other resident formats and
  broader shared host-visible GPU backing (device-write host read,
  foreign buffers/streams, aggregate UMA fit). General
  DAG/media/context authority, mixed backend/multi-source HF loading and
  pipeline transport activation remain unsupported. Continue source lifecycle/placement only through real
  consumers. Keep selected shape
  bands and typed ports, no naked
  scope or implicit retry. Contiguous text invocation/producer ports and the
  whole-text prepared boundary are implemented above; arbitrary explicit
  position tensors and host/IO effects remain unsupported. Extend
  stateful regions only with actual qualified state layouts/bridges:
  independent partial execution, committed-frontier conversion/recompute and
  arbitrary state schemas remain open. Native and captured consumers retain
  original source/numerical/state authority. Graph-only preparation may
  lower supported native semantic regions directly, without ONNX.
  Extend the bounded same-context CUDA managed-private/secondary-stream
  bridge only for real consumers: opaque external allocations, cross-device
  or cross-runtime transfers, other-backend completion and capture remain
  unsupported. No implicit copy/synchronization or independent publisher;
  no additional IR, allocator, scheduler or compiler framework.
~~~~

**Lines 511–515**

~~~~markdown
- [ ] Keep external compiler integration and ONNX import out of the foundation
  dependency graph. The concrete MIGraphX/TensorRT direct-lowering adapters
  and optional graph import are tracked in
  [Phase 6](../FULL_IMPLEMENTATION_PLAN.md#6-optional-compiled-execution-providers--after-native-parity);
  do not activate them before the ordered native backend campaigns.
~~~~

**Lines 525–527**

~~~~markdown
- [ ] Finish remaining qualifier/fault and captured testcase dispatch boundaries
  across model/GDN/native paths. Parametric GDN source proof/design is ready;
  actual runtime implementation and source/state/physical validation remain.
~~~~

**Lines 537–539**

~~~~markdown
- [ ] Extend source-bound .8B qualification to fresh47–50-token prompts, genuine
  long contexts and original BF16 CPU/HIP/Vulkan routes. Existing cached
  frontiers48–50 and source fixtures do not establish those capabilities.
~~~~

**Lines 540–545**

~~~~markdown
- [ ] After source-bound correctness and legitimate source access: current
  independent B4 physical/CPU suite, then optimized release +quiet same-attempt
  C16 for both small models; then trusted x2 original27B full-window capture
  under its remaining gates. [l5 benchmark container termination](../cpu-gpu-optimization/BENCHMARK_CONTAINER_TERMINATION.md)
  now has a supported cleanup-escalation inference; historical signal/OOM proof
  remains unavailable. No failed scope or capture gate is waived.
~~~~

**Lines 549–550**

~~~~markdown
- [ ] Natural independent-reference and explicit Gemma row-hit/transfer scopes;
  no public serving, current source21 x2, general-fit or performance grant.
~~~~

**Lines 558–563**

~~~~markdown
- [ ] Map each applicable strategy to existing primitive→model operation→family→
  model/profile owners: shared matrix storage/layout, compact outputs, prepared
  recipes, temporal work, pages/COW/prefix, graph ownership and target verifier
  once; Qwen GDN/state and Gemma PLE/KV-sharing/liveness select eligibility.
  No second registry/IR/runtime or per-token feature search. Reuse map/callers,
  source/math/backend/phase limits and complete costs precede profile selection.
~~~~

**Lines 564–574**

~~~~markdown
- [ ] Preserve distinct numerical strategies: source113 CPU-order BF16 versus
  BLAS/TensorCore/split-K, packed casts and explicit lossy codecs. A common
  operation does not share its source permit, arithmetic grant or StateRoot.
  Genuine N1→B4 cross-width diagnostic first differs at QKVZ2; GdnOut7 has a
  separate projection difference even with exact input. Original same-prior B4
  comparison now passes all15transitions/30plain+observed/fullstate/fullV/24sums/
  natural bits, with genuine HTTP N1→mixedB4/refill separately exact. Explicit
  compositional proof, not an independently generated original public trajectory
  or universal N1/B4 bit equality. Preserve distinct strategies/source permits;
  no production reference imports, weakened limits or cross-width grant.
  Astra contract review.
~~~~

**Lines 575–577**

~~~~markdown
- [ ] Use temporal/ragged pipeline liveness for Gemma E2B/E4B shared-tail fast
  prefill; retain requested prompt logits/taps and every future KV dependency.
  It is inapplicable to12B/26B/31B. Request B4 is not temporal prefill.
~~~~

**Lines 578–596**

~~~~markdown
- [ ] Adopt source-bound phase selection for2B cachedB4 through the existing
  family/native adapter: one selector governs geometry/storage/inventory,
  native/GDN/RMS definitions, readiness, input admission and scheduler. Genuine
  N1prefill/partialfallback; unsupported2BinitialB4 refused before preparation.
  Derive width4 root18 from actual8current+4candidate+2Pending+2snapshot+
  2displaced, preserving sole retained capacity and physical readers. Exact
  selected-phase costs within existing4608MiBprovider+512backend/4GiBHOST;
  authentic2B native/RMS/catalog and same-prior numerical proof required.
  Guarded source13 adoption now includes the shared model foundation and2B
  specialization together. Authentic CPU inventory/reopen reports root18,
  fullhead1,017,118,720B, provider upper4,767,472,640B and HOST4,080,276,698B;
  remaining64,365,568B/214,690,598B respectively. These are complete source-derived
  bounds, not themselves GPU evidence. Optimized13 CLOSED0; original19physical
  scopes and genuine2B15transition trajectory/export/healthyretirement PASS.
  Original2B SAMEPRIOR30plain+observed/8112sections/fullstate/fullV/natural bits
  now exact; first newcached fault fails an unchanged compound retention check,
  last2faults unrun. Fixture-only14 logs full actual snapshots without relaxed
  checks; API/HTTP/timing/full assembled acceptance remain held.
  Astra architecture verdict.
~~~~

**Lines 609–622**

~~~~markdown
- [ ] Implement the Astra next-native redesign
  through Sol HIGH: source-neutral physical owner plus source-bound services,
  same envelope/queue/ledger/fence, exact model sub-admissions and independent
  source/close/reload lifetimes. No per-model bootstrap/cap increase or source
  equality weakening. Source work isolated while frozen single-source Gemma
  actual qualification and binder316 checks continue.
  Physical13 now integrated MAIN/ISO345 with guarded original parents/modes/
  README; logical564/provider+default strict PASS, Astra architecture accepts/no
  concrete must-fix. Original local required release/full both-model CUDA/state/
  fault/API checks PASS; x2 native3/portable6/rawtransport12 and corrected shared4
  HIP/Vulkan healthy/unknown PASS. Fixture Oracle shadow aliases corrected only;
  production unchanged in345v3. Interrupted original controller and missing
  local13 closure preserved separately from successful fresh replay. Final gate
  pending after coherent contributions; measured release remains316.
~~~~

**Lines 623–668**

~~~~markdown
- [ ] Extend the original HIP actor's bounded prepared-set lifecycle for multiple
  independently source-bound models on one physical owner. Current actor
  deliberately refuses a second live set/legacy/graph; shared admission alone
  does not grant simultaneous Gemma/Qwen native execution. Preserve exact source/
  arithmetic/specialization identities, original envelope/global fence and actual
  per-set module/flight retirement. No learned payload sharing by code identity.
  Astra selected seam:
  nonzero never-reused logical StreamId through all compile/cache/IO/driver joins,
  immutable source-service Client including internal enqueue paths, maxphysical
  streams1 retained. Cold32/64/32GiB provider/device/host remains a candidate;
  derive quotas from complete costs before bootstrap. The earlier two12GiB
  draft must not force Gemma's resident projections back to streamed storage.
  Current candidate Qwen12GiB/Gemma16GiB plus one4GiBcontext fits the32GiB
  provider envelope arithmetically; complete host/init/scoped-key peaks remain
  to qualify. Qwen9,917,743,073B is model-only; Gemma13,174,234,632B plus the
  shared context gives27,386,945,001B before new metadata. No duplicate context
  subtraction, live cap growth or optimistic compiler-host release. Sol HIGH
  implements; actual multi-set remains open.
  Concrete ABI review:
  private scoped key nests the FULL original KernelId; bounded full-Eq routing
  reaches actual driver lookup despite Command dropping logical StreamId.
  Fixed SingleCompat/Scoped mode preserves original callers without fallback
  mixing. Per-set source/module/admission and global unknown fence remain.
  Astra HIGH draft-source/API review
  approves the bounded direction: shared-only Qwen reload after actual final
  reader/native/service retirement; exact pinned-client preflight with a cold
  Arc/Weak identity, without cache backedges. Sol implements both corrections.
  Cold full-Eq routes and actual family consumers were inspected; completed
  fixes, full costs, checks and two-source physical lifecycle remain unqualified.
  ROOT reviewed the complete scoped-source delta and compiler successor: Shared
  retains up to three admitted8GiB compiler leases on the original persistent
  context (two initial sets plus one reload), including cancelled attempts; model
  unload gives no compiler-cache credit. Reserve before queue submission so a
  fourth ordinary refusal cannot fence a healthy sibling. SingleCompat keeps its
  existing policy. Conditional complete HOST upper33,668,886,252B includes two
  scoped-metadata maxima and128MiB fixture allowance, leaving690,852,116B under
  fixed32GiB; actual costs/fit and two-source lifecycle remain open. Grouped
  GPU-free successor checks run separately; no physical or performance grant.
  Compiler lifetime correction.
  Frozenv4 whole7stage CPU unit now PASS: all3strict,171resident/12core/7compiler
  tests and2actual-type source-expression controls. Actual compiled scoped metadata
  is observed per set and joined exactly, not inferred as a residual. Corrected
  fullHOST upper34,205,757,164B includes separate512MiB supervisor, leaving
  153,981,204B under32GiB; startup checks actual coldplan+fixture+controller.
  ROOT complete source/control/owner review; isolated x2 optimizedbuild active,
  genuine same-actor healthy reload/globalunknown cases still pending.
~~~~

**Lines 669–672**

~~~~markdown
- [ ] Record Qwen27B's one family traversal through the existing backend/plan
  with explicit MKL arithmetic revision,112logical domains/128ordinary planes/
  fullConv4, before native/multi-model serving. Current scalar-exp/H16/short-HF
  binders refuse; accepted CPU INITIAL1→cached1 proof cannot grant their reuse.
~~~~

**Lines 674–676**

~~~~markdown
- [ ] Qualify the integrated `TextCallGeometry` extraction and actual Gemma
  consumer. It separates source-bound call metadata from model geometry and
  finite numerical/backend eligibility; six copied-core tests pass only.
~~~~

**Lines 677–679**

~~~~markdown
- [ ] Move remaining closed family builders/geometry out of core after generic
  semantic consumers migrate. Separate family, numerical and implementation
  identity; retain explicit rounding/reduction contracts and actual shape gates.
~~~~

**Lines 680–681**

~~~~markdown
- [ ] Use one prepared-model serving adapter for validation/session/accepted
  step/committed output/cancel, retaining the original router/workers/roots.
~~~~

**Lines 682–688**

~~~~markdown
- [ ] Implement sealed NativeCohort in the SAME node loop: authentic perrow
  record/runtime/prior/candidate/borrowedphase joins through validation, View/
  storage/finite certification and terminal grouped proposals, original physical
  cohort flight/Ready and single all-or-none publisher. Physical188/376 is an
  explicit request axis, not temporal tokens. Call-local body and enforcedmax8
  sessions preserve22root bound. Actual canonical47B4 source discovery passes;
  complete native arguments and genuine Rust188prefill→same-rootB4 remain open.
~~~~

**Lines 689–711**

~~~~markdown
- [ ] Share immutable recipe/topology/SSA/storage/native candidates inside the
  existing record API; bind small source-issued invocation/root/phase data late.
  Current percall record charge4MiB must not multiply into permanent Bplan
  rebuild/hash overhead. No new template IR or firstrow provenance shortcut.
  Sol18 now integrated MAIN/ISO314selected source paths; Astra HIGH completed
  architecture-only review and accepted with no must-fix. Source/profile/shape
  cache40 retains original host charges; topology/source semantics Arc-shared,
  actual operation/provenance remains invocation-local; warm bind outsideMutex.
  Per-call record bound512KiB, conservative fullcache160MiB. Core201/Qwen87/
  GPU549/default+provider strict PASS. Original source/site/native validation and
  wrapper allocations remain. Both models' complete actual CUDA/state/fault/API
  plus streamed2B faults PASS; .8B quiet C1+70.73%/C4+28.81%/C8−2.58%, singlepass.
  Full assembled gate15 LAST PASS1752Rust/0failed/89ignored/269suites/allchecks;
  314sourcepaths/rootREADME unchanged. No zero allocation/general pipeline claim.
  Architecture review.
  Residual Sol8 binder now integrated MAIN/ISO316: concrete existing ColdModel
  cache40, genuine invocation validation outside lock, original retained
  witness/site charges512KiB each +64KiB owner. GPU551/provider+default strict
  pass; original joint release and both models' full real source/state/fault/API
  plus streamed2B faults PASS. Quiet .8B C1+78.62%/C4+26.67%/C8−1.09%, singlepass;
  six original owner closures healthy. Final isolated gate16 PASS1752Rust/0failed/
  89ignored/269suites/allchecks;316sourcepaths/rootREADME unchanged. Astra
  next-native pass is architecture/planning only; Sol HIGH applies the redesign.
~~~~

**Lines 712–721**

~~~~markdown
- [ ] Before Gemma HIP merge: retain authentic source/snapshot lifecycle through
  native unknown/retirement; unsafe lowlevel boundary with one validated safe
  facade caller, no safe Any-owner proof. Keep fixed hardware/context/compiler/
  module policy in shared backend preparation. Sol44v5 implements these Astra
  corrections and complete source/state/lifetime qualifier in donor; reachable
  strict/source tests pass. Original frozen MAIN314+44 on x2 now passes full
  seven-case/all35/state/fullV/natural/lifetime matrix and fresh actual unknown
  retention. Guarded Gemma44 integrated MAIN/ISO343; successor shared physical
  owner/serving/performance stay unqualified. Prior316 final gate16 passed;
  next assembled gate follows coherent successor/affected physical checks.
~~~~

**Lines 722–724**

~~~~markdown
- [ ] Demonstrate one forced ordinary versus optimized region/profile slice
  with joint liveness/observers and device-resident typed stage edges. Build on
  existing region metadata and physical lowering, without a universal framework.
~~~~

**Lines 725–729**

~~~~markdown
- [ ] Later multidevice/MoE reuse: source/range/codec identity separate from
  physical slot/generation/placement; pin queued/captured/inflight readers,
  charge migration old+new/staging, preserve fairness and complete root shards.
  Backend tuples qualify CUDA/HIP/Vulkan separately. No current sparse,
  distributed or media execution claim follows from this review.
~~~~

**Lines 741–744**

~~~~markdown
- [ ] Carry eligible stage outputs through typed leased device bindings with
  source/profile/root, owner/device/context/generation/range, layout/extents,
  access and actual producer dependencies. Stage boundaries alone introduce no
  host return, serialization, allocation, payload copy or synchronization.
~~~~

**Lines 745–748**

~~~~markdown
- [ ] Prepare cross-stage resident handoff, device chains/graphs, exact fused
  regions and explicit bridges through existing ports/providers; force the
  accepted decomposition for comparison. Resolve finite compatible candidates
  outside the hot loop; no new interpreter, tensor runtime or execution IR.
~~~~

**Lines 749–752**

~~~~markdown
- [ ] Plan joint placement/layout/liveness and admitted scratch/graph/staging
  across eligible producers and all consumers. Qualify caller-owned native
  component buffers and stream hooks; expose hidden pools/copies/waits and
  retain cases where a copy or smaller region improves complete-task cost.
~~~~

**Lines 753–757**

~~~~markdown
- [ ] Keep qualified intermediates, model state and compact generation feedback
  on device. Use real event dependencies and bounded control/status transfers;
  preserve sampler/grammar, context grants, independent model publication,
  cancellation/fairness and output-credit boundaries. Require typed trained/
  codec bridges between models; graph composition does not erase host decisions.
~~~~

**Lines 758–760**

~~~~markdown
- [ ] Explain and attribute each required materialization/copy/readback/wait and
  graph split. Compare forced ordinary, resident-chain, graph and fused paths
  under the same numerical/approximation contract, source and output consumers.
~~~~

**Lines 761–766**

~~~~markdown
- [ ] Prove zero intermediate H2D/D2H payload traffic and no stage-only host wait
  on a qualified compatible fixture; test incompatible bridges/refusals,
  tails/ragged/multiple-consumer/debug/alias, cancel/backpressure/reload and
  unknown completion. Separately qualify trained complete-task results and
  actual physical peak/tails; start NVIDIA, then Strix Halo, without assuming UMA
  implies zero-copy. Appropriate real-source/physical and full check.sh gates.
~~~~

**Lines 773–776**

~~~~markdown
- [ ] Support explicitly approximate profiles beyond derived weight precision:
  declared methods/ranges, changed semantics, held-out quality limits and
  permitted adaptive choices; separate variant correctness from original-model
  fidelity. Bind effective variant/state formats into cache/snapshot/graph keys.
~~~~

**Lines 777–781**

~~~~markdown
- [ ] Permit interchangeable allocator/pool, provider, codec, scheduling,
  residency and transfer implementations under the original ownership spine.
  Define replacement boundaries and old-allocation retirement; test old/new
  coexistence, failure, graph readers, cancellation and unknown completion.
  Replace/rebuild unsuitable modules when real implementations justify it.
~~~~

**Lines 782–785**

~~~~markdown
- [ ] Resolve one shared versioned configuration visible to participating
  modules, with explicit local overrides, bounded operational observations and
  explainable choices. Existing coordination admits joint batch/graph/draft/
  prefetch cost; test conflicts, stale observations and reload boundaries.
~~~~

**Lines 786–789**

~~~~markdown
- [ ] Apply audited MIT/Apache static-native reuse first to every implementation
  and harness; otherwise directly port the smallest useful implementation and
  fixtures, establish faithful behavior, then integrate through our contracts.
  Preserve upstream notices; no Python/PyTorch production runtime is implied.
~~~~

**Lines 790–793**

~~~~markdown
- [ ] Consolidate profile/phase/backend equivalence acceptance: solo/concurrent,
  B1/batched reorder/refill, draft off/on/every keep, fresh/restored/COW,
  ordinary/graph/fused and contiguous/paged. State required bitwise/error/
  distribution/changed-semantic relation; reject incompatible combinations.
~~~~

**Lines 794–797**

~~~~markdown
- [ ] Exercise the first useful region/profile with one reused or ported
  component and the native correctness harness; preserve the baseline, complete
  state/physical gates and measured complete-request results. New abstractions
  must not delay current parity or require a second execution stack.
~~~~

**Lines 806–807**

~~~~markdown
- [ ] Define primitive semantics separately from their kernel implementations;
  typed ports/shapes/source/numerical policy and declared state/read/write effects.
~~~~

**Lines 808–810**

~~~~markdown
- [ ] Define a pipeline node that may reference a bounded subpipeline: one
  inference's operator/kernel regions and compound multimodal task stages use
  consistent composition laws while retaining their distinct value/frontier types.
~~~~

**Lines 811–814**

~~~~markdown
- [ ] Define immutable profiles combining named pipelines and permitted source/
  arithmetic/workload/placement strategies. Separate defaults/preferences from
  non-overridable source, authorization, physical and completion requirements.
  A saved profile is not a live SourceIdentity, Ready or allocation permit.
~~~~

**Lines 815–818**

~~~~markdown
- [ ] Compile hierarchical views into existing semantic/physical plans; legal
  flatten/in-line/out-line, exact region replacement, prepared shape selection,
  graph replay and optional fused kernels. Preserve two enduring representations,
  no new interpreter/runtime/model loop/allocator or per-token plan search.
~~~~

**Lines 819–823**

~~~~markdown
- [ ] Specify child-region effects/exports, alias/liveness after fusion, late
  source/root/epoch/generation binding, cancellation, backpressure and completion.
  Inner mathematical regions prepare; original session writer publishes once.
  Compound task boundaries must not imply global rollback of already committed
  model/media stages or allow recursive independent publishers.
~~~~

**Lines 824–826**

~~~~markdown
- [ ] Profile examples: resident interactive, concurrent throughput, long/shared
  prefix, memory-bounded dense27B; future sparse/derived-precision/media/distributed.
  Mark prerequisites, conflicts, unsupported/fallback and actual qualification.
~~~~

**Lines 827–829**

~~~~markdown
- [ ] Compare minimal efficient kernel paths with ordinary decomposition and
  captured schedules; retain cases where fusion loses occupancy or overlap.
  Distinguish semantic nodes, submissions and actual device kernel invocations.
~~~~

**Lines 830–833**

~~~~markdown
- [ ] After parity and matched baseline, measure source-only preparation
  (e.g. exact gamma widening/shift) and legal normalization-region fusion as
  early candidates; preserve casts, original model ownership and admitted bytes.
  Compare complete-request cost and actual dispatches before promotion.
~~~~

**Lines 834–836**

~~~~markdown
- [ ] Review extension drills for a new primitive/model/provider/task without
  unrelated scheduler/API changes; define exact-source/all-state/physical and
  complete-request performance gates before implementing these abstractions.
~~~~

**Lines 1150–1151**

~~~~markdown
- [ ] OSS release/distribution naming and "latest" pointers remain
  observational until an exact immutable source and codec version is bound.
~~~~

**Lines 1152–1153**

~~~~markdown
- [ ] Remote routed pipelines/peers, encrypted hosted gateways and
  implicit zero-copy cross-device are not enabled by source-free shapes.
~~~~

**Lines 1167–1173**

~~~~markdown
- [ ] Connect authenticated application/conversation history providers to
  ContextOwner, with durable and revocable grants, owner-authenticated
  tenant+application+session identity, event/graph epoch binding to the
  existing TaskCoordinator and physical memory/residency admission. Wire
  source-real calibrated NanoJev/other decision heads and streaming media
  A/B tests before enabling an actual context-fed ASR/translation router.
  Do not infer inference support from the in-memory source fixture.
~~~~

**Lines 1175–1177**

~~~~markdown
- [ ] Source execution: bind actual learned model, tokenizer, processor
  assembly and port formats in original ModelCatalog/TaskDriver before
  enabling any pipeline run or route listening.
~~~~

**Lines 1179–1182**

~~~~markdown
- [ ] Single-process multi-model scheduler: immutable source residency,
  exact shared instance reuse/replicas, typed cross-model buffer leases,
  physical ready fencing, bounded ingress/admission/cancellation and
  model-specific batch/fairness. One TaskCoordinator publishes StateRoots.
~~~~

**Lines 1184–1186**

~~~~markdown
- [ ] Pipeline runtime/hot reload: per-session model identity, graph generation,
  buffered media/source epoch, legal stage yields, failure rollback and
  atomic route transition. No stale audio, image or cross-tenant leakage.
~~~~

**Lines 1188–1190**

~~~~markdown
- [ ] API serving: explicit auth/bind/route config and tested OpenAI
  chat, ASR, TTS, images; SSE/WS/WebRTC soft-realtime with content/size
  bounds and accurate NOT IMPLEMENTED capability reports until qualified.
~~~~

**Lines 1192–1200**

~~~~markdown
- [ ] Context-aware source-real voice pipeline: stage-specific ASR prompt/
  vocab support or separate contextual transcript reranker, acoustic-first
  diarization and source-preserving contextual translation. Implement
  prior-committed same-session snapshot freezing, cancellation epochs,
  explicit consent, redaction, bounded history and no unauthorized
  cross-user disclosures.
  A/B compare WER/CER/DER/translation adequacy/latency for context
  ON/OFF; never override contradicted acoustic evidence. This does not
  follow from shape validation.
~~~~

**Lines 1202–1205**

~~~~markdown
- [ ] Actual staged ASR + diarization (speaker-tagged timestamps), ASR
  -> Qwen -> TTS one coherent command; audio sample clocks/codebook
  playback/VAD/barge-in and image latent/complete preview frames with
  backpressure. Validate codec/latency/source on real media hardware.
~~~~

**Lines 1207–1210**

~~~~markdown
- [ ] Optimize model execution: Qwen3.5 2B, Qwen3.8 27B
  resident+SSD, Gemma E2B/26B/31B, then Qwen Flash-Next QSA/GDN/MoE/
  learned n-gram embeddings/MTP and separate speculative n-gram/draft,
  with measured model-real speed/quality/physical safety gates.
~~~~

**Lines 1212–1215**

~~~~markdown
- [ ] Complete remaining exact Q2/Q3.x/Q4 quant source pipelines,
  full native image/audio/NeMo assemblies, token/codec quality and
  range/resume/cancellation before declaring "all models supported".
  Keep DeepSeek V4.1 Flash and GLM5.3 Flash strictly pipeline-only now.
~~~~

**Lines 1217–1219**

~~~~markdown
- [ ] Trained and streaming multi-model parity, real media correctness,
  device completion, SLO/fairness and zero-copy lease gates are owned by
  the model-execution/serving/multimodal/memory TODOs above.
~~~~

**Lines 1224–1224**

~~~~markdown
- [!] HIPv4 first healthy57492 refused before child/model/GPU at old-Q import receipt schema; actual source/binary/owner closure preserved, no inference result. Controlv5 real metadata-only preflight accepted both actual receipt schemas but independent review found missing finalizer import alarm_until; distinct control-only successor required before fresh actual serial cases. No source rebuild/import/copy/download needed.
~~~~


### [.agents/plans/platform-and-release/TODO.md](../plans/platform-and-release/TODO.md) — 17 unresolved markers

**Lines 12–20**

~~~~markdown
- [ ] Owner23Oct: bring Vulkan execution to CUDA feature/correctness parity;
  use ROCm/HIP for AMD including pvl-x2 gfx1151. Vulkan source catalogue
  landed in `19ff059`; the real offline build PASSED on x2 25Oct (v-build,
  1,006.6s, three artifacts) and v-admin/v-bootstrap closures are green;
  V cases are blocked on the v-resident validator/producer field fix
  (PM handoff §2.3) — no Vulkan parity or performance claim yet. Reuse existing physical
  identity/arbiter/service/queue/quarantine. Separate compiled, available and
  qualified source/layout/math capabilities; refuse before allocation/launch
  when unsupported. Preserve gfx90c launch hold and source/SDK acquisition rules.
~~~~

**Lines 21–24**

~~~~markdown
- [ ] Apply the owner-approved [static-native reuse/direct-port rule](../pipeline-composition/PROFILE_POLICY_AND_COMPONENT_REUSE.md#reuse-first-across-all-implementations-and-harnesses)
  to all implementations and correctness tooling: exact MIT/Apache file and
  transitive license/NOTICE review, suitable Rust/static C ABI library first,
  otherwise faithful smallest-component port with attribution and fixtures.
~~~~

**Lines 25–29**

~~~~markdown
- [ ] Prove actual final binary linkage for selected reused components across
  supported targets: native toolchain and ABI/error/thread/allocator boundaries,
  no accidental Python/PyTorch/foreign-server runtime, declared optional device
  interfaces, native fixture/diagnostic support and feature-isolated packaging.
  Statically linked components do not alone prove the entire binary is static.
~~~~

**Lines 30–32**

~~~~markdown
- [ ] Verify shared effective-configuration explanation/revision handling and
  replacement compatibility across allocator/provider implementations; preserve
  original resource/session/completion authorities and old-reader retirement.
~~~~

**Lines 33–33**

~~~~markdown
- [ ] Generalize the working typed forward plan into verified effectful semantic regions/task control, closed provider substitution, bounded shape/kernel caches and explicit command/resource legality; no new persistent IR/runtime without a real need.
~~~~

**Lines 34–34**

~~~~markdown
- [ ] Prove artifact→model→compiled task→coordinator→output and explain-plan for supported tasks, including source-qualified preselected fallback and no selected-GPU-error CPU substitution.
~~~~

**Lines 35–35**

~~~~markdown
- [ ] Run extension drills for one stateful semantic op, device/provider, non-token TaskProgram, quant/storage/API adapter and optional ParameterVersion/TrainStep seam; future training is not thereby implemented.
~~~~

**Lines 36–36**

~~~~markdown
- [ ] Finish per-file source/model/license/NOTICE/SBOM policy, malformed-input fuzzing and ownership/security review, authenticated APIs/private caches and qualified driver/device safety.
~~~~

**Lines 37–37**

~~~~markdown
- [ ] Verify Nix/module/native/cross packages and installer/release behavior on actual Linux/Windows/macOS target access; packaging definitions alone do not prove platform support.
~~~~

**Lines 38–38**

~~~~markdown
- [ ] Maintain minimal/default/composed CPU-free feature isolation, one lock/version authority and truthful unsupported capability behavior.
~~~~

**Lines 39–39**

~~~~markdown
- [ ] Gate bounded compiler/autotune/AOT caches by source/device/driver/shape and retain negative candidates; no unbounded first-request search.
~~~~

**Lines 40–40**

~~~~markdown
- [ ] Complete reproducible/signing/release policy and fair model/task/quality/performance matrices with cold/warm, p50/p95/p99, memory/energy and uncertainty. Push/tag/release remain separately authorized actions.
~~~~

**Lines 41–41**

~~~~markdown
- [ ] Demonstrate compatibility/invalidation of operator/task/state/plan caches and bounded change radius before broader platform claims.
~~~~

**Lines 43–43**

~~~~markdown
- [ ] Before a true zero-copy strided View is enabled, validate contiguity against stored value provenance and test public PlanValue retargeting. Current View physically materializes dense output; retain cached-element and dynamic range guards.
~~~~

**Lines 44–44**

~~~~markdown
- [ ] Preserve one GPU-library service/queue/quarantine owner and cold generic-plan admission through packaging changes; verify actual final downstream links, not only compiler/type checks.
~~~~

**Lines 99–99**

~~~~markdown
- [ ] Qualify all six [CPU release definitions](../../../docs/DEVELOPMENT.md#cpu-cross-release-definitions) with actual target execution, installer checksum/version/refusal cases and platform signing/authenticity policy. Build definitions and same-host checks do not establish other-target support.
~~~~

**Lines 100–100**

~~~~markdown
- [ ] Test enabled NixOS service lifecycle and runtime-secret handling after serving is implemented; the currently disabled service definition and module evaluation are not a running-server claim.
~~~~


### [.agents/plans/serving-and-speculation/BATCHING_AND_PAGED_STATE.md](../plans/serving-and-speculation/BATCHING_AND_PAGED_STATE.md) — 8 unresolved markers

**Lines 1030–1033**

~~~~markdown
- [ ] **Baseline/profile integration:** freeze the actual source/profile and
  existing contiguous/B1/batched results. Preserve the accepted llama.cpp
  regression lane; qualify new vLLM profiles against their own pinned source.
  Reject wrong profile/source/schema/page generations before device access.
~~~~

**Lines 1034–1037**

~~~~markdown
- [ ] **Allocator/COW:** test full/partial/tail pages, randomized physical block
  mapping, reference counts, simultaneous forks/readers, repeated reuse and
  exhaustion. Charge peak live+candidate+metadata bytes and verify final
  release or intentional quarantine under actual completion.
~~~~

**Lines 1038–1042**

~~~~markdown
- [ ] **Paged kernels:** compare independent contiguous reference, B1 and native
  batches at page_size-1/page_size/page_size+1 and multi-page boundaries;
  include asymmetric heads/positions. Poisoned inactive tails must be masked
  or rejected according to the profile contract. Compare full affected tensors,
  logits and all required state, followed by another token.
~~~~

**Lines 1043–1046**

~~~~markdown
- [ ] **Hybrid prefix reuse:** compare cache-disabled replay to miss/partial/full
  hit and forked continuations. Include KV hit with missing recurrent checkpoint,
  mismatched numerical profile, source/template/adapter and unauthorized tenant;
  invalidate stale epochs and corrupt/incompatible restore inputs safely.
~~~~

**Lines 1047–1050**

~~~~markdown
- [ ] **Real model composition:** both approved .8B/2B sources, full248320 logits
  and all42 domains, initial/continued-prefill to decode, repeated turns,
  B1/B2/B4/B8, row reorder, refill and canceled neighbor. Original27B/all112
  remains a separately admitted later gate, not implied by small-model success.
~~~~

**Lines 1051–1054**

~~~~markdown
- [ ] **Physical failures:** cancel before submit, in flight and after Ready
  before commit; abandoned candidates, stale roots, pressure, delayed/unknown
  completion and graph-held pages. No partial root publication, premature free,
  stale page reuse or clearing quarantine without separately approved recovery.
~~~~

**Lines 1055–1057**

~~~~markdown
- [ ] **Worker/HTTP:** real JSON/SSE, raw token IDs, text/usage/EOS/finish, aliases,
  staggered arrivals, slow/disconnected clients and refill. Check independent
  roots, fairness, bounded queue/output credits and actual owned retirement.
~~~~

**Lines 1058–1060**

~~~~markdown
- [ ] **Phase acceptance:** relevant source/state/cancel/physical tests, strict
  checks, both approved original release parity routes and the complete isolated
  gate from AGENTS.md. Compilation, mocks and documentation cannot close these.
~~~~


### [.agents/plans/serving-and-speculation/TODO.md](../plans/serving-and-speculation/TODO.md) — 45 unresolved markers

**Lines 208–211**

~~~~markdown
- [ ] Full isolated check.sh proceeds through CPU Clippy but stops at
  cache_prepare::role_and_staging_admission_fail_without_model_source_allocation
  (expected Error::Limit assertion). Investigate this separate CPU cache boundary
  before claiming the full gate. CUDA physical provider/correctness unchanged.
~~~~

**Lines 346–350**

~~~~markdown
- [ ] Next HIP priority: localize shared prefill device costs after tiling.
  Keep actual shared composition; target remaining private GDN/conv/attention
  or projection throughput where measured. No speculative selector machinery
  or fixture proliferation. Restore unavailable pinned vLLM closure separately
  for fresh paired runs; no acquisition/repin merely to benchmark.
~~~~

**Lines 351–353**

~~~~markdown
- [ ] Parallel CUDA correctness unit: localize and fix the deterministic C4/B8
  identical-lane divergence before using wider CUDA throughput as accepted evidence.
  The existing population benchmark is the reproducer; do not add a fixture corpus.
~~~~

**Lines 354–356**

~~~~markdown
- [ ] After each coherent fix: rerun only the smallest real-model cells needed to
  prove correctness and compare against CURRENT_BASELINE. Prefer measured hot-path
  work over additional evidence administration.
~~~~

**Lines 421–432**

~~~~markdown
- [ ] Treat X2 as a **shared physical resource**, not an exclusive test card.
  Record VRAM/KFD state before each run, execute only when the requested workload
  fits the currently available capacity, and verify only ZeroGravity's incremental
  process/VRAM/queue delta retires afterward. Existing KFD owners/VRAM remain
  other-workload baseline. Never reset/reclaim/restart/kill those owners merely to
  create an empty GPU. Missing27B/populated1K/2K references remain a separate
  numerical-acceptance blocker. **20 Nov update:** a near-empty274,448,384B/0% X2
  baseline was available and exact-current source plus the accepted private child-net
  wrapper were prepared successfully, but both installed clr7.2.4 closures fail
  `dlopen` because their LLVM22 requires GLIBC_2.44 while the current locked shell is
  glibc2.42. No HIP allocation occurred. Treat X2 ROCm model execution as environment-
  gated until that closure is repaired; do not weaken isolation or substitute SDKs.
~~~~

**Lines 433–435**

~~~~markdown
- [ ] L5 remains the source/repository authority. X2 is execution/testing only and
  must be synced to the exact current l5 `main` before physical work; do not push
  merely to transfer local commits.
~~~~

**Lines 439–441**

~~~~markdown
- [ ] Finish ordinary correctness on `pvl-l5` CUDA `.8B`/`2B` through genuine
  active-context fit, and `pvl-x2` ROCm `.8B`/`2B`/`27B`. Current finite
  references/capacity checks do not fill missing routes or populated contexts.
~~~~

**Lines 442–443**

~~~~markdown
- [ ] Qualify real C1/C4/C8/C16/C32/C64 scheduling, independent state, faults
  and shutdown; place higher concurrency on capable hardware, especially x2.
~~~~

**Lines 444–446**

~~~~markdown
- [ ] Then optimize and execute every feasible matched vLLM cell under the
  [expanded campaign](BENCHMARK_CAMPAIGN_2026-10-27.md). Per-run wins are the
  goal; preserve losses, active/offered widths and resource asymmetries.
~~~~

**Lines 447–449**

~~~~markdown
- [ ] Use x2 resources as needed; continue without routine permission prompts.
  Only consequential external dependencies/decisions justify stopping to ask.
  This supersedes the earlier blanket benchmark deferral, not correctness gates.
~~~~

**Lines 457–457**

~~~~markdown
- [ ] Owner resumed. Owned GEMM/RMS/rotary/attention now pass exact ordinary `.8B` cached47→50: full logits/all42/natural feedback and normal shutdown. Normal/fault lifecycle and final gate pass; long-active-context/2B numerical scopes need independent reference inputs; see [numerics](../numerics-and-context/TODO.md). Original cuBLAS private interface remains correctly refused; no alternate allocator or relaxed reference.
~~~~

**Lines 458–458**

~~~~markdown
- [ ] Then exact ordinary cached47→50 comparison, real after-launch cancellation/fault/retirement/shutdown, admitted context correctness and assembled gate. Reference capture alone closes none of those gates.
~~~~

**Lines 459–459**

~~~~markdown
- [ ] Owner30Oct: continue through Qwen ROCm (HIP) on `pvl-x2` now; do not wait on wider CUDA reference gaps. Needed approved2B/9B/27B downloads may use canonical XDG cache. No subagents or benchmark detour; serving remains disabled.
~~~~

**Lines 475–477**

~~~~markdown
- [ ] Extend retained usage/timing for exact context prompt tokens and explicit
  provider-side prefill/first-token timing. Prepare strict canonical47, ragged
  47/57/62/67/72 and 512..32768 workload/lifecycle inputs under XDG state.
~~~~

**Lines 478–481**

~~~~markdown
- [ ] After ordinary `.8B`/`2B` correctness, cached/fault/shutdown, long-context
  fit and final gate: execute one serial C1/C4/C8/C16, then admitted C32/C64 and
  context campaign with two balanced ABBA/BAAB repetitions; cold and warm stay
  separate. No 27 Oct timing has run.
~~~~

**Lines 485–486**

~~~~markdown
- [ ] Replay applicable original-model/backend shutdown and fault scopes under
  their existing source/device gates before a benchmark qualification claim.
~~~~

**Lines 681–716**

~~~~markdown
- [ ] Profile and fix the .8B gap first, then beat the same matched vLLM C1/C4
  baseline without changing source/quality/caps. Broader2B/C8/ragged/repeated
  timing deferred by explicit owner priority; no unnecessary baseline sweep.
  Resident-head candidate passes all40 actual numerical calls/unchanged limits
  and all five fault scopes after narrow verifier recognizer repair. Quiet
  host phase diagnostic8requests/64forwards exact: binding3.629112s/enqueue
  3.021396s, nestedHead0.005158s/zeroH2D/zero checkpoints; original owned closure
  clean. Follow-on host unit passes fresh40+all5faults/14metadata guards/API24
  per-provider exact; strict GPU Clippy PASS. Fresh unprofiled canonical wave8:
  C1 native9.7748 vsstock38.0061 tok/s (6.54748s vs1.68394s); C4 native9.7972
  vsstock49.1130 (6.53247s vs1.30312s). All32 offers/256tokens exact, actual
  peaks1/4, batchmax1, all original owners/source/EOF close. Preliminary one
  pass only; still3.89x/5.01x slower, no certified win. Combined CPU+phase
  diagnostic finds memcmp70.66% of532 user-cycle samples/0lost; actual process
  6.72user+0.66system over6.77396wall. Nested bind3.074109s/enqueue2.945825s,
  complete/readback0.149683s; Head64launches/zeroH2D/checkpoints. CPU fractions
  are not GPU or wall fractions. Investigate deep GDN static-source identity
  comparisons at their shared metadata owner; do not weaken equality/authority.
  Exact static-source equality now preserves all field/content semantics while
  bypassing same-pointer+length text scans. Metadata10/strictGPU Clippy/joint
  release PASS; frozen pair4 passes fresh actual matrix40/all5faults and matched
  API24/24 per provider, exact output/state/owner closure. Fresh unprofiled pair4:
  C1 ZG40.4089 vsstock37.0951 tok/s (1.58381s vs1.72530s), native+8.93%;
  C4 ZG39.2740 vsstock48.2272 (1.62958s vs1.32705s), native18.57% lower.
  All32/32 offers/256tokens exact, original owners/source/EOF closed. One
  preliminary pass, no confidence/tail/fullparity claim. Actual HF batchmax1;
  source-safe B4 integration is under separate investigation, not yet a causal
  percentage attribution. Subsequent pair5 residentembedding reuse passes
  fresh40calls/allfivefaults/matchedAPI24offers192tokens perprovider. Quiet
  C1 ZG48.9151 vsstock38.8841tok/s (+25.80%); C4 ZG49.9998 vsstock50.4721
  (-0.94%), all32offers/256tokens exact/allfour owners/sourceclosed. Native
 64roots=forwards=outputs/432steps perarm/peak1|4/batchmax1. Singlepass only,
  no C4win/confidence claim. Periodic146path MAIN↔ROOT convergence preserves
  parents/freshmtimes; assembled gate5 PASS1645Rust/0failed/86ignored/265suites
  plus complete scripted checks. Later common preparation/cache work requires
  its own assembled gate and fresh physical/API checks. Models parallel.
~~~~

**Lines 717–727**

~~~~markdown
- [ ] Remaining .8B C4 optimization: source-safe HF physical coalescing/batching.
  Current CLI rejects native_rows, HFloaded batchNone/native_batchfalse and
  worker excludes originalHF; existing B2/B4/B8 BatchProgram requires GGUF F32.
  Reuse root-cohort validation/publication, not incompatible math admission.
  Unchanged N1 completioncohort needs genuine private lanes or registered
  cohort epochs/status taps; shared RMS/value/readback reuse must await actual
  completion. True BF16 B4 needs its own typed arithmetic/source/state lowering.
  Existing genuine47/57/62/67 B1 prefills then four distinct cached frontiers
  provide strong mixed-row/current-root oracles without another capture; test
  row permutations/all24boundaries/all42state/fullV/F32/feedback/fault closure.
  This is source planning; no B4 implementation or causal gap percentage claim.
~~~~

**Lines 728–732**

~~~~markdown
- [ ] After source/state/natural-serving parity, execute established short
  C1/C2/C4/C8 comparisons with unchanged source/profile/EOS/caps and original
  lifecycle controls. Then apply measured substantial generic improvements and
  rerun those profiles before long contexts and dense27B. Research alone adds
  no throughput/TTFT/goodput result.
~~~~

**Lines 733–742**

~~~~markdown
- [ ] After the already authorized matched benchmarks complete, implement
  graceful cancellation plus explicit forced/immediate cancellation with bounded
  escalation deadlines. Add hooks for detecting stalled progress, repeated model
  loops and runaway CPU work; allow the supervisor to break them cleanly. Define
  request/session/model scope, cooperative drain versus forced termination,
  timeout policy and observability. Release GPU/native/source resources only
  after confirmed completion/termination; keep unknown work fenced/quarantined.
  Cover disconnect, stuck execution, timeout/escalation and healthy-survivor
  recovery with real physical tests. Owner directive22Oct: defer this extension
  until after benchmarks; it must not delay the active parity/benchmark path.
~~~~

**Lines 743–747**

~~~~markdown
- [ ] Compose stable leased rows/diff metadata, compact readout, qualified
  production graph regions, exact fusion and true temporal prefill inside the
  existing worker. Price extra graph pools/padding, output backpressure and
  prefill/decode fairness; unknown capture/launch completion cannot eager-fallback
  through a fenced service.
~~~~

**Lines 748–752**

~~~~markdown
- [ ] Extend full-state branch verification before any proposer: prefix0..K,
  target correction, all recurrent/conv/KV/indexer/draft/RNG/grammar domains,
  EOS/stop and canceled/stale epochs. Compare n-gram/Recall/suffix, compatible
  trained MTP/block drafters and K=0 using useful complete-cycle work, not assumed
  acceptance or source kernel counts. Keep explicit sampler/distribution profiles.
~~~~

**Lines 753–756**

~~~~markdown
- [ ] Retain profile/shape/layout/source/context-grant compatibility at every
  batch/prefix/graph/proposal boundary; dynamic policy chooses only already
  qualified compatible variants. The user-facing profile combines pipelines,
  while original roots, physical admission and actual completion authorize work.
~~~~

**Lines 770–772**

~~~~markdown
- [ ] Integrate both accepted GGUF/reference and newly qualified vLLM numerical
  profiles through the same session/worker/HTTP path. Bind profile/phase/schema
  to compiled plans and cache compatibility; reject incompatible state reuse.
~~~~

**Lines 773–777**

~~~~markdown
- [ ] Implement original-plan paged KV append and block-table QK/softmax/AV,
  preserving each profile's masks, positions, dtype/reductions and ownership.
  Qualify independent B1/B2/B4/B8, page boundaries and full logits/state;
  allocator/hybrid prefix/COW dependencies live in the
  [memory plan](../memory-and-streaming/TODO.md).
~~~~

**Lines 778–1123**

~~~~markdown
- [~] Extend bounded refill/regroup into fair aged chunked prefill/decode
  scheduling; qualify actual multi-token temporal prefill separately from
  request batching. **15 Nov fair-interleaving sub-slice accepted:** the shared
  engine owns a bounded fair-round policy above existing source/phase wave
  formation. Every eligible decode row is selected first; young prefill is
  deferred without dropping its original state/source owners; after two complete
  deferred rounds the oldest prefill wave must make progress. Round admission
  requires enough logical-token credit for one maximum admitted prefill wave plus
  every active decode row and refuses invalid/unbounded age/budget controls before
  worker/session creation. Focused model-neutral coverage passes 18 wave tests and
  14 worker tests, including owner retention/age rotation; strict engine clippy
  and production `qwen35-cuda`/CLI checks pass. Physical unequal-arrival HTTP
  qualification passes both approved 0.8B/2B CUDA models: B is admitted only
  after A's first actual ID, the first shared B2 cohort is no earlier than A
  output index 4, both sockets remain drained, CPU IDs/text/usage are exact, zero
  failed/refused/cancelled/discarded work, and final device use returns 41MiB.
  The complete `scripts/check.sh` command set is terminal-green on this exact
  source under isolated `native-gpu`: formatting/architecture, full GPU feature
  boundary, CPU clippy/workspace tests, minimal/core isolation, cache/source
  reference, serving/lifecycle, acquisition, release/installer/shell/Nix and
  qualification self-tests all pass. The monolithic connector wrapper itself
  disappeared during ordinary cached builds on repeated attempts without a
  compiler/test error, surviving child, cgroup OOM event or GPU residue; do not
  relabel those wrapper attempts as passes. This fair scheduling sub-slice is
  accepted from the complete constituent gate results. The broader item remains
  partial. The reusable native B2/B4/B8 equal-chunk candidate chain executes
  2..32 host teacher tokens through private candidate domains, keeps original
  session roots unpublished until one final all-row Ready, aggregates
  state/logit/readback/dispatch receipts and publishes one logical root advance
  for the whole chunk. **15 Nov 0.8B B2 functional physical acceptance:** the
  retained small `zerogravity-gpu` qualifier is now `temporal_prefill`, shared
  across B2/B4/B8. B2 proves abandoned 4-token rollback, one-root 4-token state
  publication, one-root 2-token terminal greedy, CPU/B1/full-all42 exactness,
  exact producer and final reserved0. The actual production CLI HTTP path passes
  prompt10 as chunks4+4+2 with CPU ID264/text ` a`, usage10/1, exactly3 B2
  publications, zero failures/refusals and final41MiB.
  **15 Nov temporal production checkpoint:** original2B independently
  passes full B2/B4/B8 resident qualification with CPU/B1/all42,
  rollback/publication/greedy/producer exact and final reserved0; retained bytes
  are 1,589,873,428 / 1,833,045,908 / 2,319,390,868. Two unsupervised B4 runs
  disappeared before temporal work; a heartbeat-supervised run completed exactly.
  Signed `eaec62d` promoted B4 and signed `8e6f14d` promotes **B8**. Production
  HTTP passes original0.8B and2B B4 as three exact4+4+2 B4 cohorts/publications with mixed
  JSON/SSE CPU text/usage exact, zero failed/refused work and clean retirement;
  B8 passes the same exact production contract on both sources. Retained2B B2 HTTP
  also passes. B2+B4+B8 now all select temporal4 with equal-frontier prefill.
  One earlier supervised0.8B B8 attempt remains incomplete. Canonical
  release-profile B2 remains open after another heartbeat-supervised thin-LTO
  build disappeared during CLI compile/link before the test ran, without terminal
  child/Rust/OOM/Xid/residue evidence; do not loop it.
  **Paged-state phase:** `0027c28`/`54340a6`/`1eb6e0f` establish the logical
  table, finite resident page ownership and actual CUDA COW materialization.
  Signed `3ac2059` adds a typed `StateLayout::Kv` page representation plus
  bounded gather-to-contiguous adapter without selecting it in sessions. K/V
  tables have distinct compatibility identities and paired frontiers. Final-byte
  RTX3060 page-boundary proof uses BF16 row512/page4 and a 4-page cap:
  frontiers3/4/5 gather byte-exact to contiguous (dispatches2/2/4), 3->4 performs
  two COW copies, 4->5 shares full first pages/reuses backing, prior state is
  immutable, inactive tails zero, capacity refusal and final retirement pass.
  Lower materialization and submitted-unknown quarantine cells remain green;
  production `qwen35-cuda` composition and strict clippy pass. The implementation
  also separates pre-submit fresh-target validation from post-Ready candidate
  validation, a lifecycle bug found by the first physical boundary attempt.
  At `dc6a188`, canonical session selection was still disabled; that commit adds
  the missing GPU-only bridge needed for the later wiring:
  transient `ResidentPagedKvStaging` exposes the exact old
  `StateBuffers::Kv` ABI, and private page appends can scatter only new rows
  from contiguous candidate staging under the same original flight while COWing
  the partial prior tail from immutable pages. Physical prior3->candidate4 passes
  with gather2, prefix-copy2, scatter-COW2, scatter-append2, no host readback
  after input, exact parent/child bytes and full retirement. The retained boundary,
  materialization and unknown-completion cells plus production composition/clippy
  remain green.
  Signed `53336b3` now lands the canonical `Domain`/root substrate itself:
  `StateBuffers::PagedKv` is a marker, `Domain` retains the actual paged owner
  and exact page lease/generation cohort, validation checks the original
  `StateRequirement::Kv`/source/service/readiness contract, legacy/native
  consumers fail closed, and single-session commit uses the factored original
  publication sequence. Physical roots0->3->4 publish successfully with2 page
  owners per committed frontier, exact displaced parent bytes and final release;
  all prior paged cells plus production composition/clippy remain green.
  **Production selection remains open.** Admission analysis rejects the earlier
  two-staging sketch: paged committed state + full prior staging + full candidate
  staging becomes three state envelopes at full context. Signed `ef9e4d4` now
  lands the one-staging execution contract: `StateBindingMode::KvInPlace`
  requires exact same fresh K/V lease identities for prior/candidate, and the
  actual ordinary `KvAppend` emits zero preparation launches instead of
  zeroing/copying the already-gathered prefix. Physical base3/capacity8 pass
  preserves the prefix, writes the new row exactly, leaves the tail zero and
  retires staging; all retained paged cells and production composition/clippy
  remain green. Next replace ordinary root allocation with hybrid admitted
  vector/page/staging pools, then select the paged route.
  Signed `93468b1` closes the Ready/snapshot semantics first: paged snapshot
  reads are typed, ordered and valid-tail bounded; physical partial/full snapshots
  plus4->3->4 restore pass. Ready classification permits only exact initialized
  full same-plane/index current-root pages as inherited, so3->4 has2 Writes and
  4->5 has2 inherited+2 Writes. The new pure hybrid inventory plan proves
  vector-root + staging + page maxima fit the old roots*state_bytes envelope and
  refuses insufficient active-fork headroom or heterogeneous page geometry.
  At `93468b1`, production selection remained open with physical pool
  construction and explicit page policy still pending.
  Signed `cb57e1d` now constructs the actual vector/staging/page pool owner and
  source-owned installer. Physical vector4+staging1+page24 residency equals the
  old67584-byte root envelope exactly and retires to0. Initial hybrid roots are
  zeroed vectors + empty paged KV; candidate allocation intentionally refuses
  until a current-root-aware path exists. Stable compatibility IDs use explicit
  numerical tags and checked physical/slab/capacity packing.
  Signed `b9b04aa` now lands the one-flight current-root candidate path itself.
  One `HybridCandidateBundle` owns Vector candidate + complete staging + all KV
  append plans; exact access deduplication merges current Reads and submitted
  Writes; gather precedes the unchanged ordinary program, scatter follows it, and
  only Vector + fresh/COW pages are promoted. Paged Domains materialize only
  after Ready. Physical one-Vector/two-KV base3->4 passes with gather4/scatter8,
  five Ready outputs, exact canonical Ready order/frontier4 continuation/zero
  tails and final resident0. The exact-current eight-cell suite and production
  strict checks pass.
  Signed `1413f72` adds the explicit low-level CUDA source-state policy while
  leaving public/default serving Contiguous. Paged source loading now requires
  `Paged { page_tokens, max_append_tokens }`, B1, and bounded positive policy;
  inventory/runtime append bounds are enforced. Invalid paged/native/bounds
  policy refuses before source IO. The original-HF differential compares
  Contiguous vs Paged full-vocab/all42 at page-1/page/page+1/2page+1.
  The approved managed0.8B/2B cache inputs are now visible on L5; original0.8B
  page4/append4 passes full-vocab/all42 exact at frontiers3,4,5,9.
  Signed `398e446` adds the matching source-real lifecycle target: prelaunch
  cancel, abandoned Ready, cancel-after-Ready discard, page-boundary commits and
  exact all42 restore to pre-boundary/boundary roots, final shutdown.
  Signed `3041457` corrects restore qualification to fresh root/epoch
  publication at the saved frontier. Cold-paged0.8B lifecycle is accepted by
  retained dotlink job `9feb1394...`: exit0,168.08s,
  `contiguous_warmup=false`, exact cancel/abandon/restore/all42/shutdown.
  Signed `e6aac53` removes the warmup/stage scaffolding and leaves the qualifier
  cold-only. Other intermittent null-exit jobs are signalled-Bubblewrap
  infrastructure events, not ZeroGravity failures. Correction: deployed dotlink
  v0.17.3 does not have main's `SharedProcessExit::Signal`; it records
  `ExitStatus::code()==None` for a signalled child, and its sandbox child is
  Bubblewrap. Timeout/cancel/failure use distinct statuses and Completed disarms
  containment before recording. Controlled inner/sandbox-shell SIGKILL paths
  report numeric137, while failure-time cgroup OOM/pids and restart/poll-failure
  evidence are0. A direct no-inner-supervisor cold run still reproduced null exit.
  **Original2B page4/append4 source-real acceptance is now green.** The repaired
  local native-GPU harness executes inside the existing Bubblewrap namespace.
  Dotlink v0.17.4 forensics correct the earlier interpretation: it logs every
  stop it initiates, and the relevant window shows only the explicit
  `job_cancel`; a silent long sleep survived until cancellation. Thus earlier
  SIGKILL9s are outside dotlink stop paths and do not establish a quiet-job
  watchdog or heartbeat requirement. With the model foreground and a heartbeat
  sidecar used only as harness plumbing,
  differential `38c1f321...` exits0 in703.17s with full-vocab/all42 exact at
  3/4/5/9, and cold lifecycle `744f36fd...` exits0 in438.18s with
  `cold_load=true`, exact cancel/abandon/restore/all42/shutdown.
  **19 Nov true in-flight cancellation gate accepted on original0.8B paged B1.**
  The qualification-only launch cut pauses only after the ordinary CUDA flight has
  actually enqueued work and before Ready. Known cancellation then waits for a
  physical status readback and retires the existing event without initializing
  submitted Writes or creating Ready: root/all42 remain exact, Vector/staging/page
  backing stays held until completion, returns to the exact preflight occupancy,
  replay commits successfully and final shutdown is clean. Dropping the same
  submitted future instead preserves the accepted fail-safe path: root binding is
  unchanged, service fenced, one event pending,396 submitted handles plus source
  completion owners quarantined and no pool reuse. Two externally SIGKILLed runs
  before terminal semantics remain incomplete; retained direct known-completion
  and heartbeat-assisted unknown-completion runs both exit0. Heartbeat is optional
  transport plumbing only. **19 Nov explicit high-level selector accepted without
  changing defaults:** public `QwenCudaStatePolicy::{Contiguous,Paged}` is exposed
  only through `ModelLoadRequest::QwenCudaWithStatePolicy`; that historical fixed
  `QwenCuda` Contiguous default is superseded by the21 Nov generic smart-config layer.
  Paged now admits native rows4/8
  through the same Hybrid owner; zero/out-of-context bounds still fail before
  source IO. Native B2/B4/B8 cohort state transport is implemented by reusing
  `OrdinaryCandidateState` gather/body/scatter per lane under one cohort flight.
  Contiguous B8 keeps its private-KV fast path; paged `KvInPlace` uses existing
  per-lane KV append. Focused policy tests, strict CUDA clippy and CUDA-serving
  composition check pass. The compact original0.8B Contiguous-vs-Paged B2/B4/B8
  prefill/decode/all42 cell is terminal-green: exit0 in306.84s, exact state/output
  and clean shutdown; two earlier external-SIGKILL attempts remain incomplete only.
  Public SDK paged serving is also terminal-green:8 concurrent page-boundary
  requests,2 outputs, native B8 observed for both Contiguous/Paged, exact token IDs,
  clean shutdown in340.78s. **Paged HTTP is now terminal-green** under the explicit
  child-private-net wrapper accepted by `4688cba`: the same8 concurrent page-boundary
  workload observes native B8 for Contiguous/Paged, exact token IDs and clean
  shutdown in358.07s. Strict test-target Clippy passes. Direct shared-net L5 still
  refuses `--require-private-network`; no isolation was weakened. Fresh matched-vLLM
  timing remains open because no local vLLM runtime is present and retained C8 data
  is not this workload. Full-gate composition found and fixed one generic
  `cuda-native` import boundary for shared hybrid inventory; targeted production
  `cuda-native` now checks cleanly. The warmed monolithic rerun was externally
  SIGKILLed later during ordinary-Qwen feature compilation with no diagnostic and
  remains incomplete. Matched-vLLM rerun is blocked by the retained pinned layer
  store no longer exposing the exact image ID to Podman from this connector; exact
  HF source remains intact and no reconstruction was attempted. **Prefix fork/COW
  substrate now passes physically:** fresh session, independent root, no replay,
  paged partial-tail3->4 COW, exact full logits/all42, immutable cached snapshot,
  clean shutdown in179.13s. **Bounded semantic prefix lookup/eviction is now
  implemented** in the shared engine with exact grouped compatibility, exact units,
  greatest-prefix/newest-tie lookup, duplicate refresh, deterministic FIFO eviction
  and explicit entry/weight caps; immutable payloads are `Arc`-shared and lookup
  allocates nothing. Two focused tests, strict engine Clippy and CUDA composition
  pass. **Scoped worker hit/miss is now accepted.** A validated nonzero
  `PrefixSharingScope` travels beside the queued request; ordinary `request_text`
  remains isolated and explicit `request_text_scoped` opts in. Generic worker restore
  verifies its cursor against actual executor frontier and caches only already-
  committed state-only prompt roots. Qwen owns a paged-only model-local bounded cache;
  contiguous state is unaffected. Physical0.8B page4/max_append64 prompt65 produces
  scheduler model-token deltas65 warm /1 same-scope /65 different-scope with exact
  output ID and clean shutdown in172.98s. The first run exposed and fixed retirement
  ordering by clearing cache snapshots before resident shutdown. **Trusted HTTP
  scoped hit/miss is now accepted:** `ServerBuilder::prefix_sharing_scope` supplies
  one fixed optional scope from the embedding application; no request field/header
  can set it, and unaware services reject scoped admission. Physical0.8B
  page4/max_append64 prompt65 over HTTP records scheduler model-token deltas65 warm /
  1 same scope7 /65 second-server scope8, exact output ID and clean shutdown in
 171.97s.7/7 server tests, strict serving Clippy and CUDA-serving composition pass.
  **Bounded live-snapshot eviction/cache invalidation is now accepted:** trusted
  `invalidate_prefix_scope` removes only currently retained entries for one scope and
  returns exact entry/weight totals; it does not cancel already-admitted work. Safe
  stale-scope handling is stop/drain admission, then invalidate; server lifecycle/
  scope switching alone does not purge.
  Qwen cache count=`2*active`, weight=`active*context`. Physical0.8B active1 proves
  FIFO prefix64 oldest miss/newer hit, purge2/128 + invalidated-scope miss, and512-token
  retained-weight pressure where prefix320 hits but prefix256 was evicted; final
  purge2/448, repeated purge0, clean shutdown in179.12s. Actual resident page-pool
  pressure is now qualified too: original0.8B page4/max_append64 cached prefix64 plus
 5952 externally held real pages fills6144/6144; the next tail returns prelaunch
  Capacity with unchanged root and no vector/staging leak, then exact full-logit
  recovery after release with all pages idle at shutdown in168.51s. One startup
  SIGTERM attempt remains incomplete only. Minimal Qwen diagnostics expose current
  prefix-cache occupancy/bounds and paged-pool allocated/idle/occupied/max/page bytes;
  no cumulative hot-path counters. Host/device OOM is not claimed. Physical overlap
  remains behind event/lease and independent-root gates.
  **Fresh overlap review checkpoint:** do not widen worker scheduling yet.
  `PhysicalQueue` is still single-active per physical card, `ResidentSubmission`
  completion is tied to trusted readback/physical retirement, and Qwen CUDA is pinned
  to one logical stream; more runnable work would therefore not establish physical
  overlap. The lower CUDA substrate now has a same-service fresh secondary stream and
  non-clone actor-owned event. Physical L5 proof records primary+secondary events
  before either wait on disjoint8MiB buffers; secondary was still not Ready on first
  query and both readbacks were exact (`P12_CUDA_SECONDARY_STREAM_EVENT_PASS`). This
  is explicitly a prerequisite, not a throughput/TTFT/ITL or prefill/decode overlap
  claim. **Resident event/lease in-flight ownership is now accepted:** one armed
  `ResidentSubmission` and one exact CUDA event stay coupled until explicit resolution.
  Known completion still requires original readback/validation before Ready;
  cancellation retires after event completion with writes uninitialized; unresolved
  drop fences/quarantines and retains the attached completion owner. Physical marker:
  `P12_CUDA_RESIDENT_INFLIGHT_EVENT_PASS ... queue_width_unchanged=true`.
  **Bounded multi-flight substrate is now accepted:** configurable active-ID set in
  `PhysicalQueue`, production registry still width1, and shared first-Submitted/
  last-Ready accounting over one prepaid model envelope. Existing event-ledger access
  admission remains authoritative. Physical bounded2 dual resident flights on distinct
  CUDA streams/outputs retire in reverse order with exact8MiB readbacks and reusable
  service (`P12_CUDA_RESIDENT_DUAL_FLIGHT_PASS`). Real Qwen/source feedback is now in:
  existing public B8 SDK,8 concurrent original0.8B,48+3, one warm then measured wave.
  Exact native B8 passes, but steady contiguous is4888.231ms/4.910 output tok/s and
  paged5080.588ms/4.724 tok/s. Mean first-token commits are4523ms/4567ms respectively,
  so prefill dominates; decode is ~44 tok/s contiguous and ~31-32 tok/s paged. The
  immediate bottleneck is not queue concurrency **or weight streaming**. Revalidation
  proves active8/context512 remains fully resident:18-root total2,076,877,813B plus B8
  runtime reserve still fits the4.5GiB provider envelope. The generic prefill cohort
  shares projections/stateless RMS but serializes the remaining stateful operators by
  lane; decode already has cached GDN batching and one-token B4/B8 attention. The
  stateless QK-RMS/gate/multiply/SiLU prefill lowering is retained: exact warm48+3 B8
  improved4888.231→4696.591ms contiguous (~3.9%) and5080.588→4907.715ms paged (~3.4%)
  versus the retained no-extension baseline. The historical captured engine
  is fixed B4/47 and qualification-only, so it is not a clean serving fallback. Next
  multi-token attention is now a reusable shared primitive and is wired into equal-row
  B4/B8 prefill groups; the warm48+3 wave stayed exact but TTFT was effectively flat
  (4696.591→4701.433ms contiguous;4907.715→5022.809ms paged), so attention is not the
  dominant remaining cost. Shared multi-token GDN prefill is now accepted: generic
  sequence-aware GDN geometry/workspace plus typed private-root import/export drives one
  B4/B8 cohort runtime over packed activations. Exact warm48+3 B8 moved4701.433→1007.301ms
  contiguous and5022.809→1160.106ms paged in the default unoptimized test profile.
  That profile is not a serving-performance baseline. The same commit/existing cell
  rebuilt non-LTO/opt1/codegen64 is183.932ms/130.483 output tok/s contiguous and
  242.488ms/98.974 paged, with exact tokens/native B8. This already clears the retained
  historical eager-vLLM B8 72.20 tok/s short-wave baseline on contiguous. Future perf
  feedback must use a matched optimized host profile. Continuous refill/regroup now has
  real public-SDK feedback too: active8/native8 with4 short +4 long survivors +4 queued
  rows exposed and fixed a reusable band-layout defect. Width8 cohort storage now retains
  exact B4 and B8 GDN ABI descriptors while sharing one max-width physical scratch arena;
  legal B4 fallback no longer binds B8 extents. Warm refill passed with `{4,8}` on every
  survivor/queued row, demonstrating B8→B4→refill→B8 regrouping. Opt1 goodput for68
  generated tokens:178.769 tok/s contiguous,136.488 paged. The first matched paged-gap
  optimization now removes host-zero allocation/H2D from paged KV staging and fresh-page
  initialization by reusing the bounded pinned device-side word-zero primitive. The same
  optimized8x48+3 SDK cell remains exact/native B8 and measures184.409ms/130.145 output
  tok/s contiguous versus205.724ms/116.661 paged, improving paged from the retained
 242.488ms/98.974 without moving contiguous materially; first-token commits are
 133.988/133.531ms. Physical zero work is now counted as dispatches, including Hybrid
  scatter accounting. The next shared transport slice now collapses per-page KV
  gather copies with a fixed15-page-pair/2-destination pinned ABI under the existing
 32-pointer ceiling; larger prefixes chunk through the same runtime-scalar kernel. The
 48-token/page4 workload therefore gathers12 key/value pages together instead of issuing
 24 page-copy launches per KV domain. The same matched cell passes exact/native B8 at
 176.800ms/135.746 tok/s contiguous and182.505ms/131.504 paged, leaving ~3.2%
  matched wall-time overhead. The final paired gather now also zero-fills the inactive
  staging tail when the bounded span permits it, eliminating the remaining two staging
  zero launches for this cell while preserving the explicit fallback for oversized or
  empty-prefix cases. The same matched cell passes at177.467ms/135.236 tok/s contiguous
  and180.520ms/132.949 paged, only ~1.7% overhead. The production refill shape was
  then rerun on unchanged `e72030c`:4 one-token +4 eight-token survivors +4 queued
  eight-token requests,68 generated tokens, matched opt1/codegen64/non-LTO. It passes
  at400.383ms/169.837 output tok/s contiguous and422.572ms/160.919 paged (~5.5%
  wall-time overhead), materially improving the retained old paged136.488 tok/s result.
  A device-scatter append-copy+tail-zero fusion was measured at421.159ms/161.459 paged,
  only ~0.3% faster than accepted source, and was removed rather than retaining another
  kernel path. The temporary refill probe was removed too. Fresh vLLM remains
  unavailable and is now explicitly owner-deferred. The admitted CUDA context ladder has
  now advanced through **4K,8K and16K on both0.8B and2B public SDK serving**:
  context-2 prompt +2 output, exact replay on a new session, exact GPU producer frontiers
  and clean shutdown. The retained qualifier is current-path capability evidence and
  does not substitute for the missing independent all42 populated archive. The first
  0.8B16K attempt was externally SIGKILLed during Rust compilation before test entry;
  clean residue/GPU/kernel checks plus a cached retry pass make that an infrastructure
  event, not a model result.2B16K passes too. The32K slice then invalidated the old
  admission assumption rather than producing a serving pass: the retained qualifier now
  accepts32768;0.8B32K was externally SIGKILLed twice soon after test entry, while2B32K
  public model load succeeded and the real qualifier ran~230s before external SIGKILL.
  Clean residue/GPU/kernel checks followed each termination. The old2B32K refusal came
  from the pre-streaming resident-weight calculation; shared source admission has had
  streamed-weight fallback since `d16f096`. Current32K storage is now closed from
  the loader's exact shared preflight:0.8B Resident total=3,427,684,813B;2B Streamed
  total=1,925,110,221B. The external SIGKILL also reproduces twice during compile-only
  current-source `cargo test --no-run`, before test entry, with no dotlink-cgroup
  OOM/CPU/pids-limit event and idle GPU. Do not force a refusal, add another context
  fixture, or retry-loop32K. Two later direct trace runs on the exact already-built
  optimized0.8B32K public-SDK binary produced one full pass and one external kill:
  run1 completed in185.18s with exact replay, independent sessions, exact producer
  frontiers and clean shutdown; run2 received `SIGKILL from outside` at~80.7s after
  >60s execution. So0.8B32K serving capability is now positively established once, but
  the old external-run reliability issue no longer obscures2B32K. Fresh matched optimized
  contiguous reruns stay alive normally and reproducibly hit the retained SDK's600s
  first-response timeout at~616–617s total, with no SIGKILL. Simple source-placement
  experiments (partial residency, state-path-first pack ordering, larger serial page
  through16MiB) did not materially improve that boundary. The actual fix is explicit
  paged-state admission with physically reachable COW-page pricing: it preserves the same
  four logical lifecycle roots while keeping all2B32K weights resident. Same32K capacity,
  1024-token public A/B is283.22s contiguous-streamed vs28.56s paged-resident (~9.9x),
  and the retained full32766-token paged run passes in224.80s execution/260.34s wall with
  exact replay/frontiers and clean shutdown. Canonical `QwenCuda` now runs Enabled smart
  config and jointly resolves state representation + native cohort width from exact current
  admission: resident weights first, then widest qualified B8/B4 cohort, then Contiguous
  as a tie-break. User values still win per knob under Enabled. The existing active8 SDK
  cell with native width unspecified Auto-selects8 and physically publishes `{8}` for both
  Contiguous and user-Paged halves at135.979/133.004 tok/s. No benchmark result
  from the earlier killed
  short2B B8 attempt is retained; do not return to residual paged micro-tuning.
~~~~

**Lines 1141–1146**

~~~~markdown
- [ ] Run profile-matched batching/paging/prefix-hit/overlap ablations and genuine
  pinned-vLLM comparisons: established short C1/2/4/8 first, then qualified long
  contexts. Record quality/full-state acceptance, true ITL/TTFT tails, committed
  SLO-goodput, offered/rejected/canceled work, fairness, actual dispatch/copy
  counts and physical memory/fragmentation. Preserve old-profile regressions;
  code presence or fewer launches cannot close serving/performance parity.
~~~~

**Lines 1168–1181**

~~~~markdown
- [ ] **P12 ordered SiLU integration:** private CUDA B1/B2/B4/B8 operator
  passes204 physical old-GPU/CPU differential cases,171562710 exact bit
  comparisons and five fresh failure/abandonment processes. All17 intermediate
  checkpoints, status, canaries and original lease/quarantine ownership pass.
  Independent source review and candidate CUDA/Vulkan/default strict checks
  plus two row-partition tests pass. Candidate library, qualifier, server,
  worker and HTTP release artifacts compile; the exact-source diagnostic
  file-transport unit passes. Paused at the owner's request before starting
  the prepared60-case physical matrix. Full-model integration remains unaccepted:
  complete PTX/loaded-code witnesses, model/StateRoot/serving regressions, both
  original release parity routes, isolated full gate and fresh timing/count
  comparisons remain pending. Existing truncated PTX dump filenames overwrite
  specializations; retain that failed evidence and qualify a collision-safe
  diagnostic before claiming complete instruction coverage.
~~~~

**Lines 1183–1212**

~~~~markdown
- [ ] **Stock2B loader and matched benchmark:** genuine pinned stock2B
  V4 plain/observed loader and independent comparison now pass: all335
  stored tensors inventoried, exact15 MTP exclusions and320 backbone roles;
  both complete vocabularies,496640 native BF16 logit words, remain exact.
  Source/runtime and owned process/group/container/EOF cleanup pass; retain
  the original V3 observer refusal. Both model loaders are prerequisites,
  not cross-engine/full42/default-graph or throughput qualification.
  Actual0.8B eager HTTP policy passes for both providers: identical47 prompt
  IDs, original Rust tokenizer, effective greedy/EOS settings, observed
  cgroup12GiB/swap0/CPU4 caps and authenticated GPU process membership.
  Original owned cleanup closes. Podman unresolved HPIDs establish no
  ownership; every sampled GPU PID must match the numeric owned subset.
  Generated IDs differ under the declared activation/cast profiles. Actual
  0.8B full-output quality covers seven prompts; quality equivalence remains
  unqualified. First unprofiled eager/cold-process/solo comparison completes
  all eight ABBA/BAAB samples: four requests/32 raw tokens per provider.
  Median TTFT13.400s native versus1.561s stock; stock is faster here.
  Cold-process means first request after readiness, not storage-cache cold
  or complete startup latency. Both providers are internally stable, but
  cross-engine IDs/text/usage differ. One stock sample coalesces delivery;
  individual-token ITL and source-correct SLO-goodput remain unqualified.
  The2B eager/cold-process/solo run also completes all eight samples:
  four requests/32 tokens per provider, identical raw IDs/output hashes/
  usage/finish on that one prompt. Median TTFT14.540s native versus1.562s
  stock; all samples have individual token delivery, native ITL p95
  0.186–0.199s versus stock0.026–0.030s. Seven-prompt2B quality remains
  unqualified despite five matching full texts; this prompt establishes no
  numerical equivalence. Finish default-route, warm/concurrent workloads
  and physical counts;
  this bounded observation does not establish numerical or serving parity.
~~~~

**Lines 1214–1227**

~~~~markdown
- [ ] **Production CUDA graph admission:** signed `5bf440d` first made the
  graph identity/pin/replay/owner substrate available to production `cuda-native`
  without selecting it. Signed `db7be60` now adds explicit opaque graph-resource
  admission through the **same** resident ledger/provider envelope used by managed
  buffers and other foreign driver resources; no second ResourceArbiter domain was
  added. The policy reserve is deliberately labeled policy, not measured CUDA
  internals:64KiB control + one4KiB page per prepaid source-dispatch slot + Rust
  graph/receipt/lease-access-vector metadata. It is reserved before `graph_prepare`,
  can grow only before capture starts, must cover the captured dispatch count,
  travels with CaptureWindow/CapturedResidentGraph/quarantine owners, and releases
  only after the existing trusted retirement path. Capture and retirement queue
  work declare that already-prepaid footprint as `AllocationClass::Other`; the
  physical queue remains scheduling-only and never charges the global envelope
  twice. Actual CUDA driver graph bytes/node enumeration remain **unqualified**.
~~~~

**Lines 1258–1268**

~~~~markdown
- [~] One later repeat observed a transient post-capture total resident-byte
  mismatch of exactly15,339,840 bytes with pending_events=0 and an empty physical
  queue. The run failed closed before replay. The condition did **not** reproduce
  in four consecutive subsequent physical runs; all four returned41MiB and the
  final strengthened run explicitly proved charge-retained-while-live=true and
  charge-released-on-retire=true. Do not weaken the idle invariant. If it recurs,
  the `81b7c71` witness reports warm/steady foreign bytes, graph resource bytes,
  charge count and Arc strong counts so the actual ownership boundary can be
  localized. CUDA driver-internal graph bytes/node enumeration remain unqualified.
  This is correctness/lifetime qualification only, not graph performance or
  production serving selection.
~~~~

**Lines 1329–1351**

~~~~markdown
- [ ] **P12 fused continuous GPU serving:** extend the accepted CUDA B2/B4/B8
  HTTP path with paged hybrid KV/recurrent cache, physically safe
  prefill/decode overlap and fairness. Unequal decode frontiers and
  compatible refill are accepted within the B7 source/context scope.
  Extend full-logit/StateRoot qualification to other advertised trained
  profiles/backends and full GPU prefill. Measure actual p50/p95 TTFT,
  true token ITL, committed SLO-goodput, fairness, memory and physical
  dispatch counts against genuine pinned vLLM V1 with equivalent
  source/quant/tokenizer/hardware, cold/warm conditions and caps.
  Local release smoke and fewer launches do not qualify performance.
  Separate native0.8B B2 CUPTI diagnostic records64328 actual kernels and
  64328 successful direct launch calls, zero dropped records and exact source
  counter delta; plain/observed complete outputs and owned retirement match.
  A fresh accepted-baseline0.8B canonical C1 diagnostic also validates all
  four native/stock plain/observed processes:236270 actual native kernels
  and direct launches equal the source delta, zero dropped records/errors;
  exact provider-internal outputs and original owned retirement pass.
  Stock public profiler records9318 kernel activities/8810 host launches,
  but its losslessness/all-process coverage remains unverified. Independent
  validators and ROOT/peer metadata reviews pass. These observations qualify
  the native windows only; complete matched hardware counts, profiler device
  allocation admission, source-correct goodput and profiled timing remain
  unqualified. Equivalent default-route/warm/concurrent vLLM timing stays open.
~~~~

**Lines 1353–1363**

~~~~markdown
- [ ] Connect accepted core `ContextOwner`/`ContextReader` snapshot
  capability to the real router's authenticated session/turn frontier:
  source-scoped application/shared grants, per-application and per-tenant
  ownership, immutable complete context records, explicit ON/OFF context
  selection, short-lock optimistic source epochs and cancel/revoke
  invalidation. Recheck `require_fresh` at TaskCoordinator delivery and
  bind source/graph generations to model/media StateRoot publication.
  DecisionAdapter is a typed callback boundary, not trained NanoJev
  logits or actual ASR context support. Cross-session/host source sync,
  bounded CPU staging and QoS, media quality A/B and revocation/stream
  cancellation remain open.
~~~~

**Lines 1365–1376**

~~~~markdown
- [ ] One router must snapshot each enabled, source-granted context
  stream at the declared turn/graph epoch before live ASR. Default only
  committed user/assistant session content; explicitly selected tool
  call/result, application-owned system/developer prompts and authorized
  shared/cross-user records are eligible. Strict source ownership and
  requested grants, not a blanket category blacklist, govern access.
  No same-turn LLM feedback loop or unauthorized sources/tenants, raw
  unowned KV pointers or tool-secret leaks. The ON/OFF toggle leaves
  typed graph topology stable; OFF performs no retrieval. Bound
  configurable 4KiB..1MiB plus physical admission, ordered filters,
  typed decision models, credit/backpressure, stale ACKs and source-real
  accuracy/consent tests before live mode.
~~~~

**Lines 1378–1386**

~~~~markdown
- [ ] Make YAML/repeatable-flag compiled pipeline configs a first-class
  source-bound router input: many model IDs, many typed component DAGs,
  many explicit protocol/route bindings within ONE process; one model
  can serve standalone chat and ASR→diarization→LLM→TTS. Preserve
  direct one-model CLI execution with no config requirement. Current
  pipeline check is pure scaffolding; implement model residency,
  admission, TaskCoordinator execution, generation/hot reload and
  authenticated OpenAI/SSE/WebSocket/WebRTC transport, then enable
  pipeline run/serve only after source/physical tests.
~~~~

**Lines 1387–1391**

~~~~markdown
- [ ] Distinguish model-trained MTP weights and Qwen Flash-Next
  learned n-gram embedding bank from runtime n-gram/suffix
  speculative decoding (no extra source required). Verify
  accept/reject and complete rollback/grammar RNG across
  QSA/GDN/MoE before any speed claim.
~~~~

**Lines 1393–1393**

~~~~markdown
- [ ] Implement canonical TaskRequest/coordinator workers and CLI/SDK/API integration, model routing/residency lifecycle, explicit supported OpenAI/Anthropic interfaces, streaming/tools/reasoning/schema/sampling and truthful capabilities.
~~~~

**Lines 1394–1394**

~~~~markdown
- [ ] Enforce separate API/tenant payload, queued-prefill/residency and physical active-step admission; bounded overload, cancellation, disconnect, stale epochs and output backpressure.
~~~~

**Lines 1395–1395**

~~~~markdown
- [ ] Add real compatible continuous/grouped batches, aged chunked prefill, decode/media fairness and finite joint batch/draft/expert/kernel/placement plans; no per-token catalog/JSON/global-lock work.
~~~~

**Lines 1396–1396**

~~~~markdown
- [ ] Finish Qwen basics before optimizations; enumerate the complete pinned llama.cpp/vLLM union, public and internal surfaces. Preserve 105 provisional groups/145 document seeds and CLI/API frontiers as unreviewed inputs, not exhaustive parity.
~~~~

**Lines 1397–1397**

~~~~markdown
- [ ] Implement every applicable sampler/logits-processor/RNG/logprob/grammar/tool/stop/reasoning composition and its independently verified outputs.
~~~~

**Lines 1398–1398**

~~~~markdown
- [ ] Qualify target-authoritative MTP, draft model, n-gram/suffix, EAGLE, DFlash/DFlash2, DSpark and related registered strategies; DSpark2 exact primary source/version remains unresolved.
~~~~

**Lines 1399–1399**

~~~~markdown
- [ ] Test all accepted-prefix lengths, target-distribution correction, target/draft RNG/grammar ownership, full KV/recurrent/index/media rollback, physical branch admission and authentic cancellation/quarantine.
~~~~

**Lines 1400–1400**

~~~~markdown
- [ ] Jointly benchmark material combinations, adaptive depth/backoff and all offered/rejected/canceled work: committed SLO-goodput, TTFT/ITL tails, fairness, memory and energy versus equivalent pinned vLLM/V1/Omni.
~~~~

**Lines 1401–1401**

~~~~markdown
- [ ] Preserve zero untriaged frontier/disposition requirements before any full-parity claim; synthetic acceptance, fewer launches and competitor measurements do not establish local parity or speed.
~~~~


### [docs/PUBLICATION_AND_REVIEW_CHECKLIST.md](../../docs/PUBLICATION_AND_REVIEW_CHECKLIST.md) — 11 unresolved markers

**Lines 23–23**

~~~~markdown
- [ ] When experimental code exists, add a **separate reproducibility/results appendix** with real model/quant/driver/OS/link/thermal/workload, independent quality, p95/energy, matched baseline and raw data. *Pending by design: no engine implementation.*
~~~~

**Lines 38–38**

~~~~markdown
- [ ] When real source exists, confirm architecture with working multi-model, non-token and second backend **extension drills**, profiling and resource-fault injection. *Pending by design.*
~~~~

**Lines 47–47**

~~~~markdown
- [ ] **Author credit/byline, institutional affiliation and contact** need the project's owner to select a public identity. No personal or organization attribution was invented.
~~~~

**Lines 48–48**

~~~~markdown
- [ ] Confirm the publication artifact's licensing and third-party permissions before public release, following the owner's [MIT project policy](../CONTRIBUTING.md) and [LICENSE](../LICENSE). Separate third-party material and formal publication clearance still require review.
~~~~

**Lines 49–49**

~~~~markdown
- [ ] **Trademark, third-party material and publication legal review** remain owner responsibility; prior-source code/weights/license rights must be reassessed if implementation incorporates them.
~~~~

**Lines 50–50**

~~~~markdown
- [ ] **Persistent publication date/version/DOI/canonical URL** to be assigned when publicly submitted; internal paths are local developer references, not external URLs.
~~~~

**Lines 51–51**

~~~~markdown
- [ ] **External link rot audit at actual publication time** and citation-style proofreading should be repeated; references were researched as of 12 October 2026.
~~~~

**Lines 52–52**

~~~~markdown
- [ ] **PDF/HTML publication artifact, accessibility/print QA and figure export** optional for release; no PDF was requested, generated or claimed in this design pass.
~~~~

**Lines 53–53**

~~~~markdown
- [ ] **Independent technical peer review** and copyedit/author signoff still required for a formal release.
~~~~

**Lines 63–63**

~~~~markdown
- [ ] Before implementation, inspect the live source revision, working changes and agent ownership; follow [AGENTS.md](../AGENTS.md).
~~~~

**Lines 64–64**

~~~~markdown
- [ ] Record actual test commands/results, relevant model-content identities, fixtures/hardware, failures and unfinished work in the plan/run TODOs. Group completed work under its landing commit after its gates pass, following [README.md](../.agents/plans/README.md); no generated handoff or separate proof inventory is required.
~~~~


**Snapshot audit:** 629 unresolved markers in 23 documents. All 15 tracked live/archive plan TODO files below are retained complete at original locations (including unmarked prose and completed histories).

- [.agents/plans/TODO.md](../plans/TODO.md) — 11 literal open/active/blocked markers.
- [.agents/plans/cli-device-routing/TODO.md](../plans/cli-device-routing/TODO.md) — 65 literal open/active/blocked markers.
- [.agents/plans/cpu-gpu-optimization/TODO.md](../plans/cpu-gpu-optimization/TODO.md) — 67 literal open/active/blocked markers.
- [.agents/plans/distributed-execution/TODO.md](../plans/distributed-execution/TODO.md) — 12 literal open/active/blocked markers.
- [.agents/plans/fabric-vnext-transition/TODO.md](../plans/fabric-vnext-transition/TODO.md) — 13 literal open/active/blocked markers.
- [.agents/plans/fabric-vnext/TODO.md](../plans/fabric-vnext/TODO.md) — 42 literal open/active/blocked markers.
- [.agents/plans/memory-and-streaming/TODO.md](../plans/memory-and-streaming/TODO.md) — 29 literal open/active/blocked markers.
- [.agents/plans/model-acquisition/TODO.md](../plans/model-acquisition/TODO.md) — 19 literal open/active/blocked markers.
- [.agents/plans/model-execution/TODO.md](../plans/model-execution/TODO.md) — 79 literal open/active/blocked markers.
- [.agents/plans/model-execution/archive/populated-hip-serving/TODO.md](../plans/model-execution/archive/populated-hip-serving/TODO.md) — 0 literal open/active/blocked markers.
- [.agents/plans/multimodal/TODO.md](../plans/multimodal/TODO.md) — 27 literal open/active/blocked markers.
- [.agents/plans/numerics-and-context/TODO.md](../plans/numerics-and-context/TODO.md) — 49 literal open/active/blocked markers.
- [.agents/plans/pipeline-composition/TODO.md](../plans/pipeline-composition/TODO.md) — 55 literal open/active/blocked markers.
- [.agents/plans/platform-and-release/TODO.md](../plans/platform-and-release/TODO.md) — 17 literal open/active/blocked markers.
- [.agents/plans/serving-and-speculation/TODO.md](../plans/serving-and-speculation/TODO.md) — 45 literal open/active/blocked markers.

## Appendix C — complete implementation and decision ledger

All 911 reachable commits through UX-P-M1, oldest first. Exact source ID, author date and subject. Initial design and all implementation/docs/fixes retained. This handoff's own documentation commit is deliberately excluded. Git remains the full diff/provenance authority.

~~~~text
be95424837deb7919adf34503db71d27d3ad3adb 2026-10-13 docs(design): product, architecture, prior-art and research plans
1e10b5cc1d5cf34832477bc83ef39117a765c6a0 2026-10-13 feat(core): immutable source identity, state roots and physical resource ownership
af9a5d3969c9a87951990223b6d289aba69c63fb 2026-10-14 feat(artifacts): sharded source artifacts and model identity
b276fd01e421f1b7e416ae773610b5823b9da36a 2026-10-14 test(artifacts): cache transactions
586d440b144727bdd9acea7455afb27f6c38ae25 2026-10-15 test(artifacts): qwen text contract
782e6fc48f0924cc055232d56338da62b68b1a79 2026-10-15 feat(cpu): portable CPU math and weights
1fd12da7840151ba47ab43368b4e4f435aac3d3b 2026-10-16 feat(model-qwen35): source-verified Qwen3.5 forward and tokenizer
901ba9ae2f2ed531d992a6d657a37dc362518bd0 2026-10-16 test(model-qwen35): source layout contract
e6ecf02bb7d951e01e66b430a24553c690b4ac5b 2026-10-17 feat(model-gemma4): Gemma contract, MoE and 26B variant
81642a9af4d88d928327a1012af1594168704956 2026-10-17 test(model-gemma4): e2b contract
7aa19c268026069ad916f6a69440dab602aaa94a 2026-10-18 test(model-gemma4): family contract
65b026f902501e85afad5fd31f3ca53968a3e836 2026-10-18 test(model-gemma4): variant 26b contract
b03c16451d65c33a05b3c495c1205240db4ccac0 2026-10-19 feat(data): vendored CubeCL license and provenance
f491e34fe55e67eed434ac577c32d44eca51430e 2026-10-19 feat(data): vendor CubeCL cubecl-ir 0.11.0
627c58ec80ff7b33fa1ce5086a729946c5b1cd44 2026-10-20 feat(data): vendor CubeCL cubecl-cpp 0.11.0
3664f975e79242d8b1a12bc4eefe886f81a33c2f 2026-10-20 feat(data): vendor CubeCL cubecl-cuda 0.11.0
f77ae37ec0317f39a2e1ce4466d82c645cd344da 2026-10-21 feat(data): vendor CubeCL cubecl-hip 0.11.0
7d768a080a84b6132595f7cadbc84b75b69106e4 2026-10-21 feat(gpu): source-exact resident GPU execution and physical ownership
a9723bceb8ad4aa4df639dbe660efcae6c22711b 2026-10-22 feat(runtime): facade, semantic program, task drivers and state publication
c3d0d0b06fb1548c8bf3db10f2965e5af52be9bc 2026-10-22 test(runtime): gemma contract
4add50a023d4fb5dd989989cab75ac4298d46d29 2026-10-23 test(runtime): layout api
b4a36b829031fba498e45bebb71a6cd8f01e7ef9 2026-10-23 test(runtime): multimodal mm0
a691e21c040d2dcf46a52f2cbcf0e0eb934c8369 2026-10-24 test(runtime): qwen real model
25ec3e27e3c208158e9763944026003005456461 2026-10-24 test(runtime): qwen shared model
047e3b70bbcbd7acbb5176712208cc4bcc02dd29 2026-10-25 feat(cli): single-binary inspect and run
52e37313bff57c2a7e831f50e9236fa983e13d02 2026-10-25 test(cli): CLI contract
22e0cc5abb4a067a95324206c2881de9e3327427 2026-10-26 feat(tooling): workspace GPU qualification crate
4d82c6704dad47875fa42bdec65f9a65de79722e 2026-10-26 test(tooling): independent reference oracle llama_reference_layers
a2d6a02f1577554ea8dba669f77ba7b903c1bb29 2026-10-27 test(tooling): independent reference oracle llama_reference_logits
a8ea8926b53d7b4a9c6d00f3d9be04906f8d64bb 2026-10-27 test(tooling): independent reference oracle llama_reference_qk
6a0079eb167d1f1593fe2a9372039138cd949770 2026-10-28 test(tooling): independent reference oracle llama_reference_recurrent
be88e30733f9459eda5f150bdbd70d9d0f8c6684 2026-10-28 test(tooling): independent reference oracle llama_reference_rope
77778a2c258751c64c56c4ffd3dba2ff70f7296c 2026-10-29 test(tooling): independent reference oracle llama_reference_softmax
6015241a5f5c2bb7ee65715a998988a026ff800c 2026-10-29 test(tooling): independent reference oracle llama_reference_value
e190aff1fdd392696e8086010a7bbc1f12d3df4f 2026-10-30 test(tooling): independent reference oracle llama_reference_value_padding
0aea8de66e3eb3b704e7d26687731a3e5b07d3bb 2026-10-30 test(tooling): build-release-artifacts.sh
057987203d5de45ad268ba02cfc5084f76f48229 2026-10-31 test(tooling): check.sh
c74450a27b1e3080e9576b0f95922f17c7cdca8b 2026-10-31 test(tooling): compare-attention-value.py
9f54d2435fe6af15b3eaf28980bc1039ad02b559 2026-11-01 test(tooling): compare-qwen-layers.py
7458ffc286f7e72c70ac18acfd3f776186aa204a 2026-11-01 test(tooling): cpu-qualification-gate.py
73cb6e55e1795e2c13533f60621dc29619cf5e2c 2026-11-02 test(tooling): native-gpu.py
80145a7f91724fb34a6f67e9972850a279982bf3 2026-11-02 test(tooling): test-install.sh
ca1138b8cf866c81bbfb276358d03a62576df16a 2026-11-03 test(tooling): test-native-gpu.py
5a0636c3f9c744c2f979ded40943a269ba5a1176 2026-11-03 test(tooling): test-numeric-resume.py
9a6c504ebef037e09bd187d65e2bec7e2e456eef 2026-11-05 chore(workspace): restore the locked full workspace manifest
57a9ad7fe78bf7c5e64d3870a6ba176ceba4cc07 2026-11-05 test(tooling): independent reference probe notes
1af2ac434279cb5f4b635a51884c936f19b54aa3 2026-10-16 fix(nix): give module check a complete base system
0452c594d99df6d3b4c972ceaae8df6ea56925e3 2026-10-16 docs(plan): record module eval check fix
2afa32aa74aff7f35ec1b52e4caa7079a5f68980 2026-10-16 docs(readme): reorder engineering and FAQ sections
a054d396e1ec12fc5e9b8e2ca60e8060954870c5 2026-10-17 docs(agents): document plan layout and TODO spec
cc3cab27a2713fdb64bb3f1a42c8138c060c5dce 2026-10-17 docs(contributing): align guide and AI disclosure
7e61cb63efd0c1429aa96044f2f7d28fda1e59aa 2026-10-17 docs(agents): gate pushes and destructive actions
9239719c25128028fd78a6a60f817ce6b626517f 2026-10-17 docs: retire historical archive bodies
6deebf22228315863f563723b5e6761ac6b51c5c 2026-10-17 refactor(cache): namespace model cache under abird/zerogravity
fa6affa425089fdf9a15bfa10b330e03f738fec8 2026-10-17 feat(artifacts): add pinned model pulls
3c97c52075563f65928c564818ce73502168778f 2026-10-17 feat(artifacts): resume interrupted pulls
baa9d8818061958a63eabeb61a7239a103a738f0 2026-10-17 feat(artifacts): load pinned cache components
20f449d20320b6f2912698c7b97cfdfd82472ba2 2026-10-17 feat(artifacts): recover missing pull sizes
f44264c017856a015584a094566a077638939e55 2026-10-17 feat(cache): add protected snapshot removal
ea700f67fca7ff7871613044a989e03b8ad6f9d7 2026-10-17 feat(pull)!: add capped auxiliary streaming
03b69482a5fb10b8c1cc7032e7cbdf39e2e5fead 2026-10-17 feat(pull): pin multi-asset release components
e44f4ee8020d5661dd8167cb07dfcb085906fcb7 2026-10-17 feat(cache): import local HF snapshots
fce1353e649df2eb73fe6a39437043fc8551aef5 2026-10-17 feat(cache): qualify cross-filesystem imports
e0073a5593e5641494c0f48e8d2dcc9e9e61a119 2026-10-17 feat(pull): add pinned HTTPS objects
e82d76b8f8e91f7d4b49c211e26baa081ff523e5 2026-10-17 feat(pull): add pinned OCI GGUF layers
f74a525366e6280a9806ecaabd7e1f4df46a6a97 2026-10-17 feat(pull): add version-pinned GCS objects
833d68f716783c255b24691552a081a1d2b88e76 2026-10-17 feat(pull): add versioned S3 and SigV4
e62222276e0dac62ed554a3fec2f2ae86edad497 2026-10-17 feat(pull): add pinned Azure blobs
87f6454fb7c4832262d36104a806baad9a9734a7 2026-10-17 feat(pull): add pinned ModelScope files
223ca794c9d45a0e246ad1cd9ba8613f2f203cd4 2026-10-17 feat(pull): add Xet hash and chunk primitives
a774ded154ed066c8f4b134ce7f5cba32ccdd09c 2026-10-17 feat(pull): add cloud content contracts
837e77f5080c0c2c07fee90b2ef9fb791068a7d3 2026-10-17 fix(pull): enforce offline acquisition rules
52e42fcdf0e7bead1ebefe55d8103501c7bd4f64 2026-10-17 feat(pull): add native Xet reconstruction
46cf5a5830e7e8d8096234341ef51fafdf63cc16 2026-10-17 feat(pull): add conditional cloud reads
4db5efbcce973d21bf1c8ca750471e6b2ab956ab 2026-10-17 feat(pull): recover pinned ModelScope sizes
f22f75972e63ea70771c59fbce065e371d5f29d4 2026-10-17 feat(pull): list pinned file candidates
f197c9cf4a44e06a3c1cf3c532855545bed945fa 2026-10-17 feat(pull): select explicit GGUF quant families
1a22597d9fda50ad71c43ae08f8660fb5b014218 2026-10-17 docs: preserve complete October 17 takeover and paused sources
04363c3e83368517475b6dccd4894ff437639927 2026-10-17 feat(pull): qualify bounded full-path filters and exclusion closure
8195c18e43bd2aba51d727c03ebe03df6d1fd166 2026-10-17 feat(pull): reconstruct verified required native Xet indexes
f56b57400eb913240d6e2b2eeea5d60d9ab6f35e 2026-10-17 feat(pull): authenticate versioned AWS FULL_OBJECT SHA256
d5aa5cb0ba10d2603ae34488e54755c5495f63b3 2026-10-17 feat(pull): finish v1 original-model companion closure
fee0bb2e2d690956069a0de151770c21168bea63 2026-10-17 feat(pull): close official Qwen and Gemma source metadata
4bf624e300254f845e9e047fcb344026c9252d5d 2026-10-17 feat(pipeline): scaffold shared multimodel routing and opt-in chat context
a9984a47052baaf1aeda558154256d361dcc64af 2026-10-18 chore(gitignore): share one canonical ignore block
ac44e836b46f6c5aff44afa298bdb342ff65433e 2026-10-18 feat(pipeline): compose source-scoped context filters and typed decisions
f28a62ceb47229279b70283f4e88747d1719e2ed 2026-10-18 feat(context): snapshot authorized committed history for pipelines
f0746d1d7e22b94b55c93cb734a8c4ad3b2ab67a 2026-10-18 feat(pipeline): finish YAML-first typed multimodal scaffolding
93a562288f0e00240da92c37c9dd7f4b846d62d3 2026-10-18 feat(source): bind verified immutable safetensors shards
b796a2e17a4f20086c4db59743aea9558ae1a892 2026-10-18 feat(qwen): bind original source tensors to shared CPU value operators
b11e58cb9ef7dd0d607f301d90c1870897cebd34 2026-10-18 feat(qwen): qualify original hybrid layer CPU proposals
1b761119c2de2690723b9531b1d43660a66c1001 2026-10-18 docs(agents): unify shared conventions and plan model
631e59fd614fa6576985e8ff8bf5503d88f99359 2026-10-18 docs(plans): migrate historical plans to the unified model
6329cffbb3e1b93811235867fa697f5f76f3d27a 2026-10-18 feat(qwen): admit source-bound original layer states atomically
e43ba4c63106bab9804ff0f581457ce4f370741e 2026-10-18 feat(qwen): compose original full-model FP32 diagnostic forward
28e47f5a4728679e33659298db043da728f225c1 2026-10-18 docs(agents): add terse-writing guidance
7e328411a8de3c0cc2b9cb9690f2e4a0aa9a54c9 2026-10-18 docs(agents): strengthen ownership and commit-scope rules
aaf9ba6ec94a9d1bbd4e09b747d9338637fc9e87 2026-10-18 feat(qwen): qualify source-bound BF16 dtype diagnostics
1c0bb41f2e7b2f635b255d5285040ed15b9c25b6 2026-10-18 feat(qwen): bind pinned source bundles to numerical reference traces
e90ff8868c46f321ff8bd0bf4c2154a71606aab2 2026-10-18 docs(agents): add terse-output guidance
35b90932f9bdf93127059df426e7d7eeaee54cf5 2026-10-19 feat(serve): add bounded OpenAI CPU router and verified SSD extents
9790fa2ef6c4993be0b93014ad1e63d5db5cabbb 2026-10-19 feat(serve): admit resident GPU requests with fair wave scheduling
114bba566bb2d150afe5f8a0fb5701880cae713f 2026-10-19 feat(gpu): qualify native multi-row F32 projection on CUDA
2869f8784d986dd80973da03afafde89da34c947 2026-10-19 feat(gpu): qualify batched attention GDN KV and elementwise operators
08be9c64ea60b5dcf4391c54cc183d63666e4a70 2026-10-19 feat(gpu): qualify resident B2-B8 state cohort publication
3ec431a2df62bbcae5091a048f285801f151da7e 2026-10-19 feat(gpu): qualify native batched GGUF quantized projections
c85ad250334f590a87c99710e95c71a3deb28323 2026-10-19 docs(agents): preserve complete Zerogravity history and P12 handoff
35ebc5b343fd82201eac768edf86ba7a81c41846 2026-10-19 feat(gpu): fuse native resident model batches
904a190cf39aca74e06c979dbaddb0db7f4404f1 2026-10-19 docs(agents): record native batch acceptance
54fd580815dbcfcf0916e4e737beaf21728818b2 2026-10-19 feat(gpu): keep batch greedy feedback on device
570d865696ab398f0c5c37f08c5c951595eadd75 2026-10-19 docs(agents): record batch feedback acceptance
2677d8b664c78b657c26e7bf46c835317b003913 2026-10-19 feat(serve): run native CUDA B2 waves
7fc9ad475a30a3d2120f307a7da2ff75a80e337a 2026-10-19 docs(serve): bound native B2 acceptance
f26232e8b03bf5ab3c13e00ed641eb0d602360d2 2026-10-20 fix(qwen): honor tied original output heads
6bfcdbe72f0133a6c79e2e7de169c3de74b3da8a 2026-10-20 feat(bench): account for offered serving work
d9fa829ceba092c7f8a3fa9c2746a11a0906eaba 2026-10-20 feat(serve): qualify native CUDA rows 4 and 8
9815504a6975ce766766fc8089729be8121df3ac 2026-10-20 feat(qwen): bind original HF diagnostic profiles
dbb5f4f4cc42e28c91b12e906f9091ef76510c03 2026-10-20 feat(serve): fuse unequal decode frontiers
cadcbd9d1216b00ca4b69cae9bc04a409e347193 2026-10-20 feat(qwen): qualify pinned CPU HF operators
1b599df06d8d7c8c4912838836b272e28e04aca1 2026-10-20 feat(bench): isolate owned provider lifecycles
d41163432fd6233c7bd88f07b64fb71b1086a902 2026-10-20 feat(artifacts): own verified tensor windows
2a32aff615fc4f7f6718f14a15187825730d6897 2026-10-20 feat(qwen): compose typed source operators
c5eb1f8e717c79146937e585a145a0052eff747b 2026-10-20 feat(cpu): qualify source sigmoid and RMS
d0a60844217fc106fe8c70bb3a99073eef0387c8 2026-10-20 perf(gpu): fuse source activation preparation
5350a4d40b535106240d48fec8aedd459d5cd17a 2026-10-20 feat(cpu): qualify typed source windows
f78892957f09a84838246aede4fef2be040a0d18 2026-10-20 feat(gpu): fuse native teacher-prefix serving
bfe62df68d763b9d5c6b87c891be6a62b3593de9 2026-10-20 feat(cpu): execute leased source-window plans
fe2a34213009cc160f4d82068bc7dd1f49d05c6c 2026-10-20 feat(cpu): qualify original attention fragments
9a199017f18f1bcb1a8053685be3c3c29236ec90 2026-10-20 feat(gpu): qualify batched graph replay
fff1842594f8011b761d08fcc1690311b72543e3 2026-10-20 feat: qualify trained BF16 operator slices
fb68d8fa165de16e21d1bd37afef3790b80da971 2026-10-20 perf: combine GPU state scatter certification
bb089823653f89d64bc97935cd7b11b7c3e66fd3 2026-10-24 feat(core): add run policy request validation
bd277cf7667381333d54ff3c7f249dae92b8efa6 2026-10-24 feat(core): add reserved domains and workspace
35c9962d8633599a8b2393b29e1fa26f02677272 2026-10-24 feat(core): admit device backing views
eb49fa0e945bec802db28472deb50ab94b715284 2026-10-24 feat(core): record plan topology and source programs
ec30569bc6b921bfc09af888978d22d23e957944 2026-10-24 refactor(cpu): share reference math lanes
9dd1ebfb18623e27cb5ef07940f5e6839f1ecaed 2026-10-24 feat(cpu): add original torch CPU operators
a3716fd0de505b5bcf1e7e43ff3da7154c1ac8ff 2026-10-24 feat(artifacts): select cached sources by locator
77c48b315feec501082c2609077033331cc9fcb7 2026-10-24 feat(artifacts): authenticate reusable source access
7ba9278b0a1ab97514f9511fc7a72ae90eeccf1a 2026-10-24 feat(cpu): evaluate reusable text source windows
dbd836351d40289eb8d99f7e1f648c139530de34 2026-10-24 feat(cpu): add haswell and zen bulk kernels
b026d0f9d5024d06f5b83768191f70258c1aa306 2026-10-24 feat(tools): add qwen27b output-tail reference
85d86b3c5863d9d78b6bed0b7db6e36ed28686cf 2026-10-24 feat(tools): add qwen27b attention-tail reference
0c7a4a9770f64d3de1e3c920a438c88e088668c3 2026-10-24 feat(tools): add qwen27b full-window reference
af5aa8498550fb98e69734c18769cd2c84299278 2026-10-24 feat(tools): add qwen27b layer9-prefix reference
6dd5970776b7b5839ebc6b60c3694049d31f406b 2026-10-24 feat(tools): add qwen27b initial-core reference
2c70ccb66c780874998e77bbdb2c9e33387bdf2f 2026-10-24 feat(tools): add gemma4 tokenizer reference
c870725898c9a0595609f53f7310182c5bec4fac 2026-10-24 feat(tools): add gemma4-e2b text reference
8b0cc32c951027123ffa2a9dde1e5f7dd6c2a57a 2026-10-24 feat(tools): add gemma4-e2b CPU matrix
283765cff8bdbb69d4d6674d7fd28360238ea693 2026-10-24 feat(tools): add portable BF16 slice reference
b5015e7a77e37172236c602b65a25c74a3caead4 2026-10-24 feat(gemma4): admit dense 31B text metadata
bebfeaf2d81722ebca4c11613e8a40d5834e18df 2026-10-24 feat(gemma4): add E2B text reference and tokenizer
3002089997fcee9e43abc3dc60404cbeba1f52ab 2026-10-24 feat(gemma4): add E2B learned-source diagnostics
225fc5e1a520637b45c3c76f36e4b58963c1de08 2026-10-24 feat(qwen35): record window, bulk and CPU tail sources
9af407bf3ff5763c20f4e9fddbf8061fa16f8179 2026-10-24 feat(qwen35): add qwen27b window replays
c4cbff17e6c431ee1d60e67cd725b3bf0f2fc4a7 2026-10-24 feat(cuda): observe device metadata
d9d4f8b9bb500dcf632b37efae50c2926152adf3 2026-10-24 feat(cuda): own compiler, source and cubin actor paths
4bc7c429d585d720600a9a33f0da25da4baceff4 2026-10-24 feat(hip): expose availability and prepared native sets
248bcec54c6e3d94992e379b509d84fd5864cbbc 2026-10-24 build(gpu): add qualification feature surface
68c2f5c050136978f5e7c01958fc08e6f2504d60 2026-10-24 feat(gpu): add shared kernel source primitives
abca70fdebda8e395b513133e79f938a9b2d84cd 2026-10-24 feat(gpu): add original source arithmetic kernels
63583681ae8c3f850f1b6e00dad4de8a7f421c56 2026-10-24 feat(gpu): add native policy and physical identity
fe95ecdde7800a6a9079f70ff7aa90f6780cad15 2026-10-24 feat(gpu): add vLLM input RMS qualification kernels
405e2462ff310c6ceae1e43ee54c54e2b13907c0 2026-10-24 feat(gpu): add vLLM GDN instruction kernels
71c59964149f280787359063f1c1e17eb52fef98 2026-10-24 feat(gpu): add Qwen2B GDN instruction assets
00d4bbabde0491f6f0be513f0223558bc813edd6 2026-10-24 feat(gpu): add .8B short-HF GDN call catalog
d031166b726a1ab07ef88ea8c1401e01df8285ee 2026-10-24 feat(gpu): add vLLM CUDA qualification scaffolding
8dbe361c1e53db00738f1e25ab11dca44b5f5840 2026-10-24 feat(gpu): add layer, attention and GDN cubin checks
13ae2ffb7d0510073fd36dfc137693a9de2f3bb8 2026-10-24 feat(gpu): add canonical47 model qualifications
bbafb1de8c769c0ad1d158ffc138b7db0144f319 2026-10-24 feat(gpu): add Qwen2B qualification leaves
f619f7eed8377b8c8bf9838a642b86d049881180 2026-10-24 feat(gpu): add short-HF and full-model checks
3e89f1919993081535b5fe2820da05ea9d840b82 2026-10-24 feat(gpu): wire resident leases, program and native sets
136a4a8692d9aefe19e1d7ef25eab4e2fc48d561 2026-10-24 feat(facade): expose CPU routing preparation
08b30b571e4b3557be78318b66365030c1bf5bb2 2026-10-24 feat(facade): add canonical47 original-HF loader
bc66207ce9c5a1ca96e609e975151541579c4390 2026-10-24 feat(facade): add short-HF serving source loader
e44a53f75320e21df9642f5c3fe711efe8fccc04 2026-10-24 test(facade): add short-HF physical suite
08fc5935e124f6af33c5010fc4309648539e1641 2026-10-24 test(facade): add short-HF cohort suite
a4a5b6322f343004d1ecdf13b264654da466424f 2026-10-24 feat(facade): add Gemma4 E2B CPU candidate
2058dcb1ea3f8053b6906561d0698df392c91731 2026-10-24 feat(facade): add qwen27b HIP teacher source
71f496b190dc2c7ced495f9d4c1f5c5da9495f33 2026-10-24 feat(facade): add Gemma4 HIP drivers
752642361cb4ad2f769b868f071a3307e3b01651 2026-10-24 feat(facade): add shared HIP lifecycle driver
2bd8167db2c1d8817d38e4851dd45f9c2c9aa6e4 2026-10-24 build(cli): gate finite HF serving and run features
b3c5a532d72a266fc3d66e13c5c1a0f7ad1bd084 2026-10-24 feat(cli): route run requests and add gemma4 candidate
45570bdb3d7f21f531b8cbc840633b8d2eb83e4c 2026-10-24 perf(serve): plan atomic source prefill waves
e6b07efdbb2d6ec56d979ff09b025c8071038f06 2026-10-24 feat(serve): route and serve canonical47 sources
9999adb1a66d646f76cd1152c32da51b22c9eded 2026-10-24 feat(serve): serve short-HF BF16 B4 cohorts
ef9cffd413dfe3a02d8ed54e4a2ff3a5e317e10b 2026-10-24 chore(scripts): extend gate for HF features
2225af34b82e0e60da0f50dd4c5ec76eb827ca48 2026-10-24 docs(agents): codify parallel execution cadence
abb6b3b3b233e6887e182acb7c95ed26718601e4 2026-10-24 docs(research): compare optimization mechanisms
4b44d313bedecc6b82ecd44e317d786db2e08cd5 2026-10-24 docs(policy): adopt reuse-first provenance rule
46996b66e39e832f05d0174419e9c8f69367128f 2026-10-24 docs(plans): design composition and batch state
6ef74b048e830f1075d4d5ca4296cc801b8b6871 2026-10-24 docs(cpu-gpu): record shared memory and parity
fb7ea9a9a77b95543544ae131fb44da26522e7a7 2026-10-24 docs(numerics): record F32 closure checkpoints
424c672768b388f1738187baffafba7973c4cd08 2026-10-24 docs(models): record source20 and Qwen audit
da28f851448b8cebdd8b5c132f1a7371a6b39380 2026-10-24 docs(acquisition): record Gemma E2B acquisition
0c966799d7830c6b0a3d6b354fcf5318d36add9d 2026-10-24 docs(cpu-gpu): record current checkpoints
c57648ac54ff02a5370c3a26e4dc0c1634cee7da 2026-10-24 docs(memory): adopt shared pools and paged state
b552e2198d66400c94fbb56d4dc5deeb7604115b 2026-10-24 docs(composition): record shared HIP checkpoints
6075a4386589ca5d6e0d857d14f3ef4c6a4879b7 2026-10-24 docs(serving): certify transport and add batching
3b9da530d922814e8e5dea75574c2d4fdbc11462 2026-10-24 docs(distributed): defer mechanism reconciliation
38e3d77eba85875f0a3b2131a74c0de42933690b 2026-10-24 docs(platform): add Vulkan parity and reuse gates
a765f221279c9f803f0dacbee314c6ad4126905f 2026-10-24 docs(cli): add device routing plan
309b17d81bfbe7878fc8734adee424cb7785f533 2026-10-24 docs(handoff): archive history and run notes
ac17d49c07b2cb0e4653db19dc8621c6f0591279 2026-10-24 docs(handoff): preserve paused continuation text
7115754b639aa8bfbaab3a9231b97458fe945df5 2026-10-24 docs(handoff): preserve draft patches
11a47c74ecc55d92749dfd581d90254f02defdd5 2026-10-24 docs(handoff): preserve takeover and audit reports
b5d814f8fcab784514b4c6d9bfb1a11a7fa635d2 2026-10-24 docs(handoff): preserve canonical sourcebooks
e11fb986091f8fef71abf188e15177242084e596 2026-10-24 docs(handoff): index the 20 Oct handoff packet
4bf36731cfc4f66e02aff2f63fc32b653f6e363c 2026-10-24 docs(plans): activate CLI, composition and compute
8f01635305af920eadba96a7d4f1d6b9e8f588cf 2026-10-25 feat(core): portable admission and thread-control roots
5d40cf507a2e8f5f159b664ddc83876d73745607 2026-10-25 feat(cli): shared pending queue and ingress credit
52f76a0e0dc3ce8e09476af1460193a03a551227 2026-10-25 fix(cli): short-HF tokenizer credit and HOST8
efd7a34220ce168b5a5c851226b400e07e95a095 2026-10-25 fix(gpu): CUDA fragment inventory and host bound
19ff0598dd79d12ade182d90a855ecf9c60b1f59 2026-10-25 feat(gpu): Vulkan native catalogue via cubecl-wgpu
29a98c1c7c086d6367308ad9468c427046008cc5 2026-10-25 feat(gpu): HIP source-page events and witness
e8db111158991339f76e5b06ed3acb8b370f302a 2026-10-25 feat(examples): Qwen27B HIP matrix, natural tools
332038ea78ec8b1e69a2a29d3ca09370d712ef6c 2026-10-25 feat(examples): Gemma4 HIP final-owner and matrix
68e13a8114e130ab89ac4c55ee9ad14172faa42f 2026-10-25 chore(cargo): lock assembled five-lane graph
74d0c71998709de48115db16507ea0109b23f65b 2026-10-25 docs(plans): record assembly, C16 and handoff
1269b8eae908a40e79d747e923e29b708d84ba12 2026-10-25 docs(plans): retire five landed lane trees
06b89da455655ef039be2a87fd1e29917ce4d2c3 2026-10-25 fix(gpu): HIP probe witness page id typing
c217c35d480eb33b04eb981496d552a816b169e9 2026-10-25 fix(qualification): V state base parents, attempt-2 bases
3941a3c9fe0ee5dec0b9f836a86e992161f5f8ba 2026-10-25 fix(qualification): Gemma resident report-phase envelope 4GiB
e6ab074e3118a7200ea0a620a92b3a0990bbcac5 2026-10-25 fix(qualification): H/G source census skips build dirs
1565683442e80f529d810d98c63ca2583c5b01d7 2026-10-25 docs(plans): record queue GPU acceptance results
2304f8b4d2079c5312a8020f14af49c2c3832ab3 2026-10-25 fix(cli): precedence, ambiguity, pre-client hold
d3d21a35c5b75272d82947ccd9f351bc7888e3c3 2026-10-25 feat(gpu): B4 grouped transport and certification
6f1e0947cdf7673cbe4626b983fe894fbad1744d 2026-10-25 docs(plans): record main checkpoint and B4 merge
9fbe15e9e45c0d9c71ce00e73dfbdde343a77513 2026-10-25 docs(handoff): record checkpoint, HGV pass 3, successor queue
5749b732dc755c24d2179ba7dd7e8abc9380805e 2026-10-25 docs(handoff): record Qwen27B capture lane yield detail
488e0ea2c21e2301c2fdeaf82dab6de32789ea08 2026-10-25 fix: repair native qualification contracts and GPU feature isolation
93b611acaa657b6999a19239e4596d4de47b4de3 2026-10-25 fix: prevent Gemma HIP source-control payload shadowing
6edc6e2c99d9a2353d16a85ac689e76af3628be5 2026-10-25 docs: pin refreshed H G packages and completed GPU-free controls
9e67b74000bc085a844d8844d954d82f8e196493 2026-10-25 docs(audit): independently validate post-4bf GPU boundaries and resume holds
03f2e662f219797da524d0e4d6e2aabef979471b 2026-10-25 fix(gpu): certify grouped B4 CUDA ABI and exact fixture retirement
3d9866dfd6cf66009c9aa6efe0967e4a134e4e41 2026-10-25 docs(audit): distinguish 23 model-dependent B4 scopes
a39ca967ea2029400649284285ca18620544152c 2026-10-25 fix(gpu): qualify grouped transport from independent fresh state and seal KV cache views
7362699b7023b30c815482a71900af40b0a0688f 2026-10-25 fix(serve): preserve prepublication cancellations and bound decode context
5229e53d7b46c0fd322217b216b2e11a8b94a433 2026-10-25 docs(review): record architectural and correctness findings and qualification gates
83a31747679385cd5951487e6bcdfb9a2f2d1f24 2026-10-25 refactor(gpu): one typed closed CUDA ABI for metadata and bound handles
095bf8a4b97dbd7a0561c7ea4316ef7ff8167bdd 2026-10-25 refactor(core): collocate Qwen3.5 recipes without changing type identity
2a47d25181f1a8d91a9489d9f2d621d3807bb329 2026-10-25 feat(cli): concise run placement aliases and observed-device inventory
d9c5872c73ee4d7b2bbbe1079446ac7634f8d919 2026-10-25 docs: scope ABI family and run UX refactor acceptance
5399ab93929eab8e6448078ae8126144b147b238 2026-10-25 refactor(cli): use device selection without redundant on alias
d84853337bbd2830f50a9c4e704d266990fd2ce4 2026-10-25 feat(cli): resolve Linux GPU device nodes to verified physical PCI identities
d0fff261904fcb7836c0f71d0682028a12fd53b4 2026-10-25 docs(handoff): preserve complete ZeroGravity history, open work and takeover prompt
a857d45ddea827457fb59823ea6597f3eca03d22 2026-10-26 test(gpu): repair B4 qualification fixtures
6610b6195505d7d8fb17640660461310fa1c3d41 2026-10-26 docs: record B4 and C16 outcomes
6d855599531e8f67ec6cd1e1e00f7e1b186ff7f3 2026-10-26 test: add shared model correctness checks
195099ca583ace766d075682927f590360c823a5 2026-10-26 docs: set all-model correctness policy
405a3e6358c85917506f76f780094de89eaec01b 2026-10-26 refactor(formats): group codecs and format homes
d9a8ecf6187216c8b909c60e44927b06f256afdc 2026-10-26 refactor(correctness): name input geometry
fa844a663ff9ab4f701146be61bf2353cf580385 2026-10-26 docs: define module ownership and naming
a5ed7de9e0f0682d86aca09d0ec73336a5f2ecf1 2026-10-26 refactor(models): contain source plan ownership
c90b8cd7e2fc8b05784d1e27c907347ea4319f97 2026-10-26 refactor(gpu): separate providers and bindings
41e93370b7b3b9edc45cd9575eaa6b9d6845bed2 2026-10-26 fix(gpu): preserve plan and capture guards
fc734571e6e342b376f406aac7fc2ef4e42e69f2 2026-10-26 refactor(app): compose model execution routes
4a0f4552354ad8f5931b7cb45bd85d34a0026eeb 2026-10-26 docs: record completed source checkpoint
5c2fe532c68e98a0c750b5ff27969ae965ec3f9d 2026-10-26 test(cuda): restore source admission checks
9c2c51ab291580a5c47d9528f8ff2ed87bd8228a 2026-10-26 docs: record side-worktree retirement
fe1762568c96eead3893f338d0181b8634a98303 2026-10-26 refactor(cpu): own shared GDN convolution
263ff791d04eb50fd1aeca85249b0f7aed3cb1ad 2026-10-26 refactor(gpu): remove model dependency
e7fc3150927104b2f22dd020f63efb8817a75d82 2026-10-26 docs: explain benchmark container termination
50abaf71121a2ffbb6a649f16aafe185650c1388 2026-10-26 fix(serve): join workers on process shutdown
c84b98744955c43f7a6a992289c7d8a977a0ee9e 2026-10-26 fix(bench): retain owned stop evidence
4f427dd814f734afdad798b053141946c998347a 2026-10-26 docs: record serving shutdown validation
cd4159d26048b0395c40235b320b7d48df6ce53d 2026-10-26 fix(lifecycle): retain owned command streams
5a18b5bbe6a80976b9255813070c09d2cae61a3c 2026-10-26 refactor(cuda): derive capture scope ownership
61f095cf760ee6b8d1f015b71b13ac46835a4917 2026-10-26 feat(reference): capture finite Qwen prompts
3612ecb7077d18685e926c698b812e4e2e60375f 2026-10-26 docs: define production and test boundaries
1f08e5525b5ede7124054156083e5a4ff67d151c 2026-10-26 refactor: separate rotary and test preparation
d74f59a18c6345dbea963861acd519c1282a26fa 2026-10-26 feat(reference): emit captured CUDA recipes
5f4051ecbfeaa1350055db47fcd281fbeb81843b 2026-10-27 docs: preserve comprehensive agent handoff
a1c51f9d478ec8025c8da09e58a8a2626c4cfe32 2026-10-27 refactor: separate fixed rotary preparation
91a29586a292a7b8204e85cf4ed21cf03968b718 2026-10-27 fix(cuda): seal rotary production routes
9ad9aac42e0ea63ab78e8ce9a7836cc1d770ab62 2026-10-27 parameterize CUDA GDN prefill convolution
05baab5abb2e02702ab565d267b44474e9362e66 2026-10-27 fix: separate prompt semantics from CUDA qualification
d087012fa5f4d604069a015a7773defed5ae185c 2026-10-27 fix(cuda): correct parameterized Conv identity
0c63d89a54ee8a45a870cc383cef144acfb09579 2026-10-27 fix(qwen35-cuda): isolate finite qualification routes
ae936a8ab3102ebac682c947a63dbf105d91418b 2026-10-27 fix(qwen35-cuda): enforce qualification boundaries
7a910d9aab0f7f09f788ff983c678ec3fad25413 2026-10-27 fix(qwen35-cuda): make route selection declarative
ca0d2bb0f4323f7584a764e0e162219c47561b52 2026-10-27 test(qwen35-cuda): cover source-kind neutrality
f61a3ca618a6b1463a185231a86f879f13019f21 2026-10-27 fix(qwen35-cuda): restrict finite route targets
85b272476e2b61d65d0d5236abad3294b29ca117 2026-10-27 feat(cuda): parameterize GDN prefill sequence setup
1fe615113e1c859d665957632166bea96095f5fa 2026-10-27 cuda: parameterize remaining GDN stages
2e3bbeccf97b360ea3a5ab228623996e58228b77 2026-10-27 feat(cuda-gdn): configure remaining prefill stages
1b898ffdb9ac333b535233ca49c1f9da1c290c4c 2026-10-27 fix(cuda-gdn): seal explicit implementation selection
c0bdbb183e6d9112ee2c9d8cc1c5281cc35c30f4 2026-10-27 refactor(cuda-gdn): remove recipe execution bridge
eb38dd5e892f2120ac29cc2fb79b2dbe137a37fe 2026-10-27 feat(cuda-gdn): add fresh-48 output variant
ba7fb0389978d82e66d1832f50feb32238717eef 2026-10-27 feat(cuda): add qualification fresh-prefill capture
fadf2b1123309a2118725f2cc005d3edc7ea39f2 2026-10-27 fix(cuda): use fresh prefill reference layout
ae9a84394d3aa08d77e0edfe4285dc09b7862040 2026-10-27 fix(cuda): decouple GDN parameters from text kernels
532a35c437d4b941ea9931d9c1e3ca16d9b7d080 2026-10-27 fix(cuda): use fresh reference prompts
54d7ea7a3012b70f3b92ec8c45800a913e6dc811 2026-10-27 fix(cuda): admit fresh .8B row extents
e99c76e3e644b612a173a029230d44dd4e781fc2 2026-10-27 fix(cuda): account fixture host baseline
ad82feb5222d81a5f19fe1311744387b8b9959bf 2026-10-27 feat(cuda): add call-free Qwen plans
f7ff4f96be5110a9a064e9c78643897212a65537 2026-10-27 fix(cuda): retain ordinary plan source owner
1921db16695d8b9166399cd10eff793a0ab8e9cb 2026-10-27 feat(bench): expand bounded comparison matrices
1d278edeaa875aa8d85936566516388943d9227e 2026-10-27 fix(bench): retain complete matrix cell results
fe228abda89c1388efe9524439aa8d04c4d37c07 2026-10-27 feat(cuda-gdn): add ordinary chunk runtime API
52bfb519a245b114de90e6e2d5cb1de8a80497ca 2026-10-27 fix(cuda-gdn): complete ordinary runtime schedule
901d23eb518c4b9167707f8f716aebb50f3cac27 2026-10-28 docs(handoff): record 27 Oct checkpoint
ea66b3ab3096d08bafe0f28cfa9a5f0c12307e45 2026-10-27 Add dynamic CUDA ordinary text primitives
68393e5a644d661bf5e101d47cebd2290c1faff2 2026-10-27 fix(cuda): constrain ordinary text candidate
6b4a2ec7a8f81fcf1995f6aa3063dc2c152a7a5b 2026-10-27 fix(cuda): close ordinary Qwen review blockers
441fa64f52be6d3d8636e6dc8241d241e6b55889 2026-10-27 fix(cuda): close ordinary primitive review gaps
ee67aa848f2603bb342029c811692361ec10b1e0 2026-10-28 docs: close ordinary CUDA primitive review
c2494ac1665508f4eabf9aab08885db5adbfec62 2026-10-28 core: bind ordinary source invocations
301e912f4046a6ed5d4e0e41c275ae2423a6dfb2 2026-10-28 gpu: generalize owned BF16 GEMM geometry
dcfbd74f7ec060e640e734cb3765b0b585226bd0 2026-10-28 gpu: admit verified BF16 matrix upload on CUDA
b116ccdbaba8ff4fa4657bd93e017cbb9d80a9e4 2026-10-28 gpu: expose owned cuBLAS to ordinary CUDA
b4a926303ccffbea6c21bf1fac5b42a313ff8021 2026-10-28 gpu: add ordinary CUDA projection owner
40c8721c2c1a257371526e6aad9cabf18a7c652a 2026-10-28 gpu: bind ordinary primitives to owner status
71cd73779a39cc74ba558efa52c7b4dc5055a13a 2026-10-28 gpu: add source-exact ordinary CUDA attention
e5f148e04c4ced0b0fda61efadace115b11c6017 2026-10-28 gpu: reuse ordinary CUDA source projections
18579ffc25818c99f9e8d884daf9fadc5a0c4dac 2026-10-28 gpu: make resident state precision explicit
9abd1d973d1097681d859d65b1cc0c50d4ed850f 2026-10-28 gpu: validate typed resident state layouts
4fde7574e75274a8971d82d0a07f1b0d54c8dfa1 2026-10-28 gpu: share canonical GDN state transport
cba4f5e2ceb689818892bee641ad55d8f4398c94 2026-10-28 gpu: partition ordinary GDN runtime schedule
8e964cef1d1df61f68ff540131476441e8c72694 2026-10-28 gpu: expose neutral ordinary GDN primitives
8d5035dc051c86107c35ba4da4dad13284bdbc41 2026-10-28 gpu: add ordinary CUDA parameter transport
57379b3452c15d4cbd1782fabe1244c54e8333d9 2026-10-28 gpu: derive ordinary CUDA resource layout
685e9be69162c0480302ca6daf609f2926714276 2026-10-28 gpu: reuse ordinary CUDA matrix packs
6f5d43d2bb9b92723df5c0deff4260a41e74fcc3 2026-10-28 gpu: prepare ordinary CUDA cold source owner
11aaf2bb19ceb11bed752945fb5a9973d4372834 2026-10-28 gpu: bind ordinary CUDA resident body
81318adeb432ec1fda08ef8fa930478253b01ddd 2026-10-28 gpu: expose ordinary CUDA cold values to body
00ee779cb096ee567be920991d4e7d3f3b4c5ee5 2026-10-28 gpu: preserve ordinary residual and GDN scratch ownership
3865ffd1ee31b530ab74e3e09184b1949c932ad6 2026-10-28 gpu: bound ordinary attention to active KV
c7b1962b60ceb2f00ff204954c0a78f0aed17559 2026-10-28 gpu: retain ordinary GDN local states
a8eef412f2b27020dbdc559ce2b6b074bccc7445 2026-10-28 gpu: add ordinary CUDA active view transport
d0f557a47e2f1359a08c30822c5b3bb3fce5942e 2026-10-28 gpu: execute ordinary CUDA through resident owner
9edece37532ed6eaaa09153885d32c3faad4a73a 2026-10-28 gpu: separate ordinary CUDA owned and execution accesses
be5362e5f26ed4e74db75341892bad9e0f5f2907 2026-10-28 gpu: load ordinary CUDA resident model
1f69da6fe186a8959a2f8bc20616ae8838f761ae 2026-10-28 gpu: execute ordinary CUDA as one resident chunk
b379a76963f60893f5197ac3d882f58778de27c5 2026-10-28 qwen: add ordinary CUDA qualification adapter
8c2459929a81256760e337283eb25b73e152b8ae 2026-10-28 gpu: close ordinary qualification build edges
c680b6c5b6902a87851a3073e7360586e2b09fa5 2026-10-28 gpu: add ordinary CUDA explicit shutdown
eeb38bc60f33741d8d3afd17b92648fb2a201a2c 2026-10-28 docs: mark ordinary CUDA qualification-only
5e29118c2f781ea8dba5b4d85fe04c570284c988 2026-10-28 gpu: qualify ordinary CUDA context admission
ec44357c57704420731d2a74f3ad888cb45cfb6e 2026-10-28 gpu: add ordinary CUDA request cancellation
f2603b3d90dcb70dbd668b5b2d23beb07814f140 2026-10-28 gpu: canonicalize ordinary KV root snapshots
e93e7b42dda9bc9ed2f6ac92c34ae7f369428220 2026-10-28 qwen: add serial ordinary CUDA parity harness
014b77385211a3431ecc90f4e2eda86659917cfd 2026-10-28 qwen: add runnable ordinary parity qualification
ceae923844d08dbc4741cdcbc2789034d51f0f64 2026-10-28 gpu: harden ordinary BLAS retirement boundary
9e566438d6a428ffb7aa363fcc05029fe1a9c59d 2026-10-28 gpu: harden ordinary BLAS activation failures
487a796a52b65e1170a961ba9a1e23eb5f426844 2026-10-28 qwen: isolate ordinary parity across processes
ee70ec6aa01e1da846ea385afcd64f7acf280cd4 2026-10-28 gpu: use source-exact ordinary rotary
78426672edaccbb669254b87a8e47f759586049b 2026-10-28 docs: record ordinary rotary qualification
ea4b67b3fc8ce8ed56f0c8f367ea7b0575881cc3 2026-10-28 gpu: refuse legacy ordinary RoPE contract
62ec2cb6fdcb8488b2e0b385ff995d903dcc7c89 2026-10-28 gpu: isolate ordinary CUDA feature boundaries
13703a4552919f795dc6b747ef2e9547c174a275 2026-10-28 gpu: add ordinary owned BLAS qualification
9ae5e7ebee018695361ee31246df44d23a676d8e 2026-10-28 docs: record ordinary provider qualification blocker
ccebfb05c7d5efb4127051059de7fe50eed227ed 2026-10-28 gpu: qualify ordinary primitive numerics
f6edcbb9a878f9d72ea6e1eeb5fbf2893a07fdc5 2026-10-28 docs: record ordinary primitive parity
642e24529111ed966a35e4825dee7bac501f3663 2026-10-28 gpu: qualify ordinary residual RMS
ff512af4a3a28b71c8861501528d04b5f61f1c45 2026-10-28 docs: record ordinary residual RMS parity
77dcafa4934a02457bf09701e1116ea8a9494181 2026-10-28 gpu: qualify ordinary multiply and KV append
107e301fe437180b083d8c889345ab956662da65 2026-10-28 docs: record ordinary multiply and KV qualification
04fad2a20f9c5992c4ee723859c4dddea743b113 2026-10-28 gpu: qualify ordinary embedding
4274b192e0793b838956f8ef7c4b2b6ccf7009d4 2026-10-28 docs: record ordinary embedding qualification
f17540b053fe0fa94fa99243a6f3b6bd65b7cb20 2026-10-28 gpu: qualify ordinary attention wrapper
7c67dd8f2c6caf9422c36d7644554a2c345065fe 2026-10-28 docs: record ordinary attention wrapper qualification
3eb3f74150066c56d38b9f18ffa4aa492dd9c157 2026-10-28 gpu: qualify ordinary canonical state transport
a6743c9f1fe7638340639065b5e27fa1bdfe6d9c 2026-10-28 docs: record ordinary state transport qualification
8f4450c06aee8979f18cc68510152b5bef981366 2026-10-28 gpu: fix cached GDN gated launch grid
8dc7cdffabbae94b2c8a4d68dea28aa66cd2f975 2026-10-28 docs: record generic ordinary GDN qualification
ee5dfe10966ae7ee7be47b1904e35657fb07db66 2026-10-28 scripts: add serial ordinary cached47 parity runner
c4f54ebd0a92d8fa21a420d63c56b1f16d3173ae 2026-10-28 docs: record serial ordinary parity runbook
b5c353ceef5c44d92511396933a1494b8faf0bb5 2026-10-28 qwen: harden ordinary parity restart
d03bb3a94c431b9a50a7767a9da3cfe177bebc7c 2026-10-28 docs: record ordinary parity restart semantics
3ed372de521553bc30d6e12159cf368dfcf198b2 2026-10-28 docs: add comprehensive 28 Oct handoff
563ad344a0e56ac6500afa19096d5acdcc93cb44 2026-10-28 feat(cache): add human by-name snapshot aliases
be1e81a69482b9838ba6f2f943fc6fd001dc3d85 2026-10-28 refactor(cache): group by-name aliases by model
35a025c6562a36342552f62245a3c75cca8b66c4 2026-10-28 fix(cache): stabilize human snapshot aliases
76d1561824e98139dd8ce07e352bee4ee1fcd11a 2026-10-28 fix(cache): make human aliases crash-consistent
8bdd54b0f13aed240efac150afc9bb3a8c0747d2 2026-10-28 fix(qwen35-cuda): authenticate parity evidence
f6209d923a5f1e3d0cf967a93d887528b5cf166f 2026-10-28 refactor(qwen35-cuda): simplify parity evidence
768b724a45921e3fe2c3fcf305c3fdac2d67817b 2026-10-29 test(serve): align qwen cuda qualification boundary
932f56499a24ce2a8f38ff5ad203fcb2716c831e 2026-10-29 refactor(cuda): move gdn recipes to capture tooling
734d29f52ed5e83abc069fdd7ca3902cb6803e47 2026-10-29 refactor(qwen35): retire obsolete qwen27 replay harnesses
2ff3d49e70a1b80fe708c9e4ae0d8ff741990548 2026-10-29 docs: add fabric vnext transition design
15f5bd3ae26891f67380edeb1a52f4afb2cae16b 2026-10-29 docs: record retired worktree and run cleanup
9abd9365d1d9334f6cbaf8a348278e5d977384dc 2026-10-29 test: retire machine-local capture dependencies
ace934b4a402278392e30ea63e2cfade0f5208df 2026-10-29 docs: prune superseded handoff archives
7b5e50e926b0724b7ad0e630afd58c6a0075644c 2026-10-29 docs: normalize current planning provenance
379d91d42aa247b0548b02211505c3aa9743658e 2026-10-29 test(serve): make pid1 shutdown control capability-aware
ae8cd89d29f1b143f0fa8a7ecffacc9a125e5128 2026-10-29 test: cover qwen27 layer9 reference contract
ff89b87cd3d288e6dc83a2e4f423e19045142d68 2026-10-29 docs: add canonical current handoff
11664d02f740b60aad2d9bdee69f9cfd3dc42593 2026-10-29 test(cache): make private fixtures umask-safe
60907d12861bc3338d5e20d2eef749510aa8cdb6 2026-10-29 fix(core): report residency refusal extents
97c52ec8ff7530e7f7cc44adc5ea19a5e3df4a05 2026-10-29 fix(cuda): restore reference capture profile
440768d58869bf672943c64b055cd56cf2091287 2026-10-30 docs(cuda): record provider boundary findings
2983bb72c5832c6efe5d9117281afa33a31517a2 2026-10-30 feat(cuda): own BF16 Qwen execution
b262d2e9e5d0b29672c2438e72f72b6754bb9e90 2026-10-30 docs(cuda): record execution qualification
99f0232d066c8cef7d8302adb1015ad7dce05643 2026-10-31 fix(gpu): preserve declared parameter rank
825ade8aca3940a838d372823696743683beb88c 2026-10-31 fix(hip): preserve natural diagnostic ownership
8ca41db024886018afdf0975e34649d4f5a7f082 2026-10-31 docs: reconcile Qwen qualification campaign
d4d5df4cf771c2981a667de92f526771064359f3 2026-10-31 fix(cpu): preserve original attention PV order
f6fa4b993687f8d8c5c93566602182fcb8d5bdf9 2026-10-31 refactor(qwen): share source execution
87ba2e2edd6b6d55c6d24adce89942e5e20f9f87 2026-10-31 docs: record source attention qualification
8a1c6b1456a6921f1fbec950ad58fc6d54bfd36f 2026-10-31 fix(gdn): retain initial empty-state addition
1f29a42c656be1540db8996df5f198e174eb2bc5 2026-10-31 feat(hip): run source-bound Qwen models
85f33c77cbfe5babaa21e008dfec13070d2a80c0 2026-10-31 fix(hip): retain canonical source ownership
55536ce31e57049c2df378e085969ff141f16578 2026-10-31 feat(cpu): implement source chunk GDN
e2559b6063b42ac65a331eab2652a079f82258c5 2026-10-31 feat(gpu): implement source chunk GDN
1c35172b3c22f027c6f2b72fa0d177fe205797a1 2026-10-31 docs: record verified checkpoint and pause
7396984887e1acd1a1ebfa9b61e09fd100d2ec63 2026-10-31 feat(fabric): add compilable base skeleton
c853ebb6594868a332462a6343a55a671c3ef797 2026-10-31 docs(fabric): reconcile plans and recovery
7fc5d17316d16a8edea0df9008e60f91e62aa81e 2026-11-01 refactor(fabric): unify source and state owners
8591b5221a0e09e22b146b22a9041ab7cc2a716b 2026-11-01 feat(fabric): wire CPU runtime and serving
27ac27892abe2abb28a08e2303d4db39d3f9f95b 2026-11-01 docs(fabric): record working key-path checkpoint
d78eb5537e8002823e17191b6ad5bb4c6f91319e 2026-11-01 fix(engine): reject runtime self-shutdown
ae7ec111114a678d4474e575b3da064cfe72190a 2026-11-01 refactor(engine)!: share CPU/GPU scheduling
72622ce6ae85db9c181434c55afbdc1e2d04faa5 2026-11-01 docs(fabric): reconcile scheduler checkpoint
d4e94b1d436b586aaab84cb4d21c795d16bb029c 2026-11-01 refactor(sdk): own run and artifact orchestration
bab4390c59bf07420270080bff7cb9f898f25d01 2026-11-01 docs(fabric): reconcile SDK ownership checkpoint
83060e5dd99d6a7aadd92befb9ea271dfd1891a1 2026-11-01 refactor(core)!: bind numerical/layout contracts
2f192240784d532a673463e6f53a50517e577692 2026-11-01 refactor(sdk)!: unify catalog and serve loading
51cd294d7c9616cf0e26ef6f9514aa0b44b8ae70 2026-11-01 docs(fabric): reconcile contract checkpoints
860ce389b5de888604a3edbc56d07dc70b21dfcb 2026-11-01 feat(engine)!: bind preparation to source owners
d0e03f5f77ea8ce6934fe7cfc7579b8eca7a41ec 2026-11-01 docs(fabric): reconcile preparation checkpoint
fef7123394d4d200c677edd077d38e7c39462749 2026-11-02 refactor(sdk)!: own run configuration policy
1f40880546dd1d7ce4b54250124055bee8482742 2026-11-02 docs(fabric): record source parity checkpoint
8d761c45e07cefbdcafc7d95f551efac83d21ec0 2026-11-03 fix(gpu): bind full greedy feedback packets
8c43844e0d214c7b9d3f74b772089973ed40c7e3 2026-11-03 test(gpu): qualify cold Gemma SDK lifecycle
9defb6c6e46b7579f6406305201c3f4c54107989 2026-11-03 docs(fabric): close scoped transition acceptance
1c999298dde1e4f7fd8d0812cd401681914aabeb 2026-11-03 docs(fabric): count supervised cases accurately
bb400c42136a798e177be82d3a1e6ca40b3b8431 2026-11-03 feat(hip): run ordinary Qwen source chunks
8ffb5113900a79c0b99d41bc5999d9892b1c4604 2026-11-05 fix(hip): qualify populated Qwen serving
b9d8355a67914f65577cd5b9d92928131aafda37 2026-11-07 perf(artifacts): reuse verified source chunks
a57590e2351beefc637f68129558d277fd8a0aa8 2026-11-07 fix(cuda): preserve output reduction boundary
4f1e8275f8270387b237223cf269be2cdf47ac79 2026-11-08 feat(gpu): compose shared source execution
d16f0968655c0775c327025bc9388a8a2b312b23 2026-11-08 feat(hip): admit explicit total device budgets
b364cbeabdde7f49647eefcd93467d38b1207969 2026-11-08 docs: reconcile checkpoint handoff
9d73d467e702ff31a66460fe345ff8b7de6575a2 2026-11-09 perf(cuda): keep greedy feedback on device
9f45471a3cb7d4c8bfce4a87f36a54ea64c90264 2026-11-09 perf(cuda): specialize decode gemv reductions
5836cb7da52abd55fdd65336b2a9a339d73a545b 2026-11-09 docs: record repeated CUDA C1 feedback
55e195c640bbea14cc251139fa763fb063319974 2026-11-10 perf(cuda): batch ordinary source decode
eea45fad75e7938dd6d88151b6635c25a61e3a0b 2026-11-10 perf(cuda): batch cached GDN convolution
98976bebca69cbbaceef3bc0ac28a798462f795a 2026-11-10 perf(cuda): batch cached GDN recurrent
03267124e9fa02bfd8fc39364f5fbc1e6452928d 2026-11-10 perf(cuda): batch cached full attention
f50645e7c39f701bf915a5f41f12d45f288b4ea1 2026-11-10 perf(cuda): certify BF16 GEMM stores
1e4d76ee6d7f1d43453936feb095548658b4d89e 2026-11-10 perf(cuda): reuse output-head weights across B4
f28d280a4efeffd701d01f3bb852201a65d3ef75 2026-11-10 perf(cuda): batch aligned full-prompt prefill
ccf441d46a64fd407217e33cbf5360046d2402d9 2026-11-10 perf(cuda): widen native source cohorts to B8
67586ebf9244c873d26c4fa49a0e8fab4aba9f92 2026-11-11 perf(cuda): widen large-K ordered prefill tiles
77ffc23826c2cf393f8341fbeb360f359b03218f 2026-11-11 perf(cuda): widen ordered B8 decode K tiles
74f642237341f61df9e62f57b5a85da7eb05c02b 2026-11-11 test(cuda): qualify populated small contexts
5a75a704b9ccb3854956e59a9ab2022810cc4e13 2026-11-11 perf(cuda): widen rounded B8 K tiles
597da3f25021a711c0a30ab1fc3a54e12fa075e1 2026-11-11 test(cuda): measure large cross-profile state accuracy
b95acdc8568efc317ab0875d231ab209fde1544f 2026-11-11 perf(cuda): regroup partial native cohorts
2b8e90fff0df7c3f7ef96971ebaa8754a111ae85 2026-11-11 perf(cuda): widen interleaved B8 K tiles
fa703fa9af2ea86cd6af5ef0220359bc6b55cf13 2026-11-11 perf(cuda): batch chunked ragged prefill
0d9a56da48b5330d5d19394f1b44bd5b7c881aaa 2026-11-11 fix(engine): require pairwise cohort compatibility
6a4e2c9312a7943ddba6b6a270225811b205b337 2026-11-11 docs(cuda): record chunked prefill boundaries
d927fda0d7c23fc5e2fdcc4149528f774f97ab2d 2026-11-11 docs(cuda): record queued B8 scaling
96a7ac74152e6e619faabb17aa2d42bf9a812146 2026-11-11 fix(engine): require mutual cohort compatibility
8f8b4648d64f011b75af4c7e379e9ead1dd4e780 2026-11-11 docs(cuda): record grouped transport rejection
e22c442e8f8aab5f6303a4af261ef19652db4b6c 2026-11-11 docs(cuda): record B8 attention rejection
3a14c48203969c0a2a98b2c01596f7ad239381ee 2026-11-11 docs(cuda): record M128 tile rejection
af7f34c35ebf848e4f0ef14ffdcf3db8345f9d3d 2026-11-11 docs(cuda): record B8 row-group rejection
ea9b83b55b80b3a4cec602e55436bd143aaa47c2 2026-11-11 docs(cuda): record ordered N16 rejection
51c776b81d7745e3c7565ab0257b6904abb427d6 2026-11-11 docs(cuda): record B8 head cache rejection
54ea36859cff72dd4dd84504f9517de80ef2dd94 2026-11-11 feat(gpu): retain qualified private B4 KV transition
9197cfa4e5116c18e3f77ffb57c2c27ddeb05ffc 2026-11-11 feat(gpu): retain qualified batched source rotary
083cc6d966253c9604c6d3113a597742535e627c 2026-11-11 docs(cuda): record paired-column head rejection
5d7dcb4bb1fa1134795464a40a99a6a428955c76 2026-11-11 feat(gpu): retain ragged private B4 attention
da591acf4131c9c3d99afe584ab7078b8de62166 2026-11-11 perf(cuda): admit ragged native decode cohorts
1361c1385864b3971793f65a14a7d2031fe14b4f 2026-11-11 feat(cuda): retain certified BF16 row split
d1ec7236345745424920ded084b9314a41775fc4 2026-11-11 perf(cuda): batch GDN row splitting
59ae8499621644d870bb72498edd29612ceefb8c 2026-11-11 feat(cuda): add exact sigmoid multiply primitive
0f0af042e08b8293a3d8601e7666c3b82af982a8 2026-11-12 perf(cuda): pack B8 head inputs inner-major
265e970051370ac747895e93c9af216a0c663f66 2026-11-12 feat(gpu): retain scalar-position rotary batching
78ef7f279b52465be3133fb7b3b4083877cec55c 2026-11-12 test(cuda): keep strict GPU clippy clean
be1cd9e5a29e0a7e4a5d21c1350d70fea993f94e 2026-11-12 perf(cuda): batch ragged prefill cohorts
9204d75e3984d65d0165ce8bdb654001867f5b19 2026-11-12 docs(cuda): record unequal-frontier prefill rejection
790d3de0724ca499b9529fc878316177a49fbd2e 2026-11-12 perf(cuda): pack B4 head inputs inner-major
154483f96c9923357a45688c55ab9689f6fa5a4b 2026-11-12 perf(cuda): pack B2 head inputs inner-major
0136b9193244e268a409f44d4d3226af04f14d48 2026-11-12 perf(cuda): gather ragged head rows inner-major
801fb382bbeb91d2dd2701713ed81eb043a15d3d 2026-11-12 perf(cuda): batch ordinary rotary spans
cffd78e4d4dde13be548bd7ecd510db5360b7130 2026-11-12 docs(cuda): record explicit-position rotary rejection
f92ed9a1eec64973907ccc279fb7d9ae3ecb5f98 2026-11-12 feat(cuda): retain qualified stateless row spans
c911c2c5917ef9f9aaa3a04853833ab472b2ca99 2026-11-12 fix(cuda): share finite compiler diagnostics
c86e6166645c5b27658abdd22f097effa06e3c3f 2026-11-12 perf(cuda): batch compatible RMS spans
884ab441036d141703176e117087a1ffe6704825 2026-11-12 feat(cuda): retain stateless gate primitives
43ab4c6f4191d647f398e30e5855d10724c25545 2026-11-12 feat(cuda): retain stateless QK RMS primitives
b191fc60492e5c6e54c858fe60b50c73ee8e0653 2026-11-12 fix(cuda): preserve decode reduction contracts
1dce1b0d17fc2b0b15c0252550d76a8f6c8f1ba9 2026-11-13 feat(cuda): retain cached GDN B8 gated primitive
c6f3c90b71242fbeec43500128c199549a53b94a 2026-11-13 docs(cuda): record lane-blocked layout rejection
19296300f595588f1a2a7e29ab50fecb867690a6 2026-11-13 docs(cuda): record decode projection profiling
7331ef1ec3be8ccf8f8cc15220f9dfaa87f4211a 2026-11-13 docs(cuda): record decode transport and unpack rejections
0951bec114c0b317fb2f9d441f2acea57cd50477 2026-11-13 docs(cuda): bound remaining decode hot path
fe3b122e1e4a5521b775b6cb5fed9d552bf9c0bb 2026-11-13 test(cuda): strengthen state transport oracles
0cdfaad87b5f9e1df07946567706e1d768acab92 2026-11-13 feat(cuda): retain ragged private B8 attention
d196c1b956b6055d10603e7dc5334539736eaedd 2026-11-13 feat(cuda): qualify private GDN state binding
bad471387db5016eb2b752c311c6eff85b41cd20 2026-11-13 test(cuda): qualify in-place private GDN state
16fba0ee49a7c14e970b3892c4d4857a11e2c1b3 2026-11-13 feat(gpu): retain exact-root derived state cache
2f651b7e62819abe6c5ba6606bf67d7b56c670e7 2026-11-13 fix(gpu): harden resident session root binding
60b048d47b97c60e802e64b86444d0a43cf54913 2026-11-13 perf(cuda): batch canonical-boundary GDN recurrent core
775dfa6051aaf11571a8b47a253826935248cf44 2026-11-13 perf(cuda): batch ragged B8 KV transitions
c12acab60c357e0742ad4ecd2f00295cd9b6bb0a 2026-11-13 feat(cuda): retain grouped B8 recurrent transport
85f458f554edf9be7bac491309e1844f27e76351 2026-11-13 feat(cuda): retain packed BF16 batch greedy
f303bca53286a7c96ae1ab78b95532244dd756ef 2026-11-13 feat(cuda): retain exact head greedy partials
68ac8109bd47b6ace8f006ce109bac043f595893 2026-11-13 feat(cuda): retain grouped ConvB4 transport
da3eae5e8f5ce2d27d167e84690db5e27a703d3d 2026-11-13 feat(cuda): retain dual QK head RMS
ee347d93bc7b2f7e362b8b07b03cf0e9f9523ea9 2026-11-13 feat(cuda): retain certified recurrent export
399e43a0a5d0c22927bdc28e1d35775be0393c96 2026-11-13 perf(cuda): widen B8 private KV CTA
de2eed8cd1eabd059f377d5a18f269f5ef39fc8c 2026-11-13 docs(cuda): record post-KV optimization probes
cb9e349946c6daceff7d58ff28df1005dfa3ece4 2026-11-13 docs(cuda): record projection cache probes
6c601f9131ef6ac33c2dcd15deb37daad35432a8 2026-11-13 docs: refresh comprehensive handoff
c282f20ea10c16680b033586bf11d0e5d62c0189 2026-11-13 fix(hip): restore source profile compilation
874e67db0b6e737e6d9de31e44ade597b3160143 2026-11-13 feat(cuda): retain shared-input projection pair
2caf304e89bc12c77308f993df267356113baac7 2026-11-14 docs: refresh handoff after HIP and CUDA checkpoints
52c8a16187c7d89977c5308ebddfe094aeda5e70 2026-11-14 docs: record X2 HIP physical health
134a571dc080ac8e2d0f243d3e4aac0ff4cfa5dc 2026-11-14 fix(hip): isolate CUDA private KV qualification
ca6e99f9b54bf880943f72752350a3cb105c21e9 2026-11-14 docs: localize 27B X2 host-load blocker
b0cf0b21e5a72a9e89c0431f8437afa2a635982e 2026-11-14 fix(cache): avoid duplicate Qwen snapshot hashing
f5d2a88ffc8cb82c2a6686ad4e913768186b3b88 2026-11-14 docs: record Qwen source-open localization
d68b649031ac89ddfb88f6880cb7185208b25c09 2026-11-14 docs: record X2 27B physical allocation blocker
5c8f14375d4ee46afe3cc02a95d710fe813eae6f 2026-11-14 docs: localize stranded X2 KFD owner
3cadc3b2ac8eadc95a8e1b19003b922e355c339d 2026-11-14 docs: reject shared-pair Qwen serving consumer
5bf440d7f891111feac63f50d04a4333b40286bb 2026-11-14 refactor(cuda): admit graph substrate in native builds
bc9a1b3485776240513ff8be08500881def4054e 2026-11-14 docs: record production CUDA graph substrate
db7be604addc4f908efb8172eb51306ec436104f 2026-11-14 feat(cuda): admit captured graph resources
cbf13841b56985c0be0642fbe8de8fbc2022e296 2026-11-14 docs: record captured graph resource admission
6a5c6101a6979d4c56c20de4c2065f44bffe8463 2026-11-14 fix(cuda): make nonpersistent NVRTC driver-safe
9dca4b84e8899bfff61716fe3f11ba9c6f23051f 2026-11-14 docs: record physical graph resource acceptance
81b7c71324e6c3ba99d4d77bff7e4e14b2ddcf28 2026-11-14 test(cuda): witness graph resource lifetime
b12e8f5395af84bda5a5c0eab960efef2399f6c7 2026-11-14 docs: record graph lifetime repeatability witness
f1ce532d49c4121324eb5f5a401f85ea900f1251 2026-11-14 feat(cuda): bind graph replay to resident flight
9c86c0cd8396abd023d4504c6f2a559249d331fd 2026-11-14 docs: record graph resident-flight authority
bc0e2673bc45e42379a40c23bc26b0d132069b0a 2026-11-14 docs: add full comprehensive continuation handoff
63ee14e440d73beedab6e2acb96838926968eed9 2026-11-14 feat(cuda): own resident graph lifecycle cache
4502f5678e8cb1641848be62cd0fb5068bae1d09 2026-11-14 docs: record resident graph lifecycle checkpoint
9d6613cb0bef84147a7d30453e5afb82ca83411b 2026-11-14 feat(cuda): select compatible resident graph cache
bc76312a915b1f0f20f20505443d21682dd18724 2026-11-14 docs: record resident graph selector checkpoint
599cb28945bee79ef70dc48ea35566c526a7f049 2026-11-14 test(cuda): measure resident graph selector widths
4be26f29a4b43cfa915a89b45839e875bdfd71fd 2026-11-14 docs: record graph selector width decision
b19bdb481b3bbc194eb4d021ed40112197a92e75 2026-11-14 docs: record cold prefix reconstruction checkpoint
f4fb04b6c0dec311bb52affd10abfae755a01a54 2026-11-14 docs: close reconstructed prefix scheduling benchmark
7dea460a80b420d0ce49a60f6c860b3502c67c38 2026-11-14 feat(serving): age prefill behind decode
ba7e3768197c417e42eecd7315f6043f1d5e9464 2026-11-14 docs: accept fair serving interleaving
9dab0043719e435ef3914f86f3966079deabc5ae 2026-11-15 feat(cuda): add temporal fused prefill primitive
1e0f4ebfc5b9650341814ad669d4101aea717623 2026-11-15 feat(serving): enable qualified temporal B2 prefill
b599db32dec2620014f2445b73d378aab8299d56 2026-11-15 docs: refresh comprehensive continuation handoff
12f784083f4057dcf36fd343e408d5a7ed482589 2026-11-15 test(cuda): qualify temporal B4/B8 prefill constituents
80409ac30a12c03f9ce4585068faa44aab791460 2026-11-15 docs: checkpoint temporal B4/B8 qualification
330150fe23dd97be1fecac42de39b9058e2140e2 2026-11-15 docs: record post-qualification GPU baseline
36253481d5e26d27ffb577187b21c338a211beaf 2026-11-15 test(cuda): bound temporal qualifier host state
4d74d7f92bc4dd69b18bf947286dcb6a537755ff 2026-11-15 docs: close temporal B4/B8 B1 qualification
eaec62d8a40c6179baa49e62be7a358b4b800128 2026-11-15 feat(serving): enable qualified temporal B4 prefill
9347f5b838e389f68b4eb38f6d51b664b844704d 2026-11-15 docs: accept temporal B4 production
8e6f14d3defc0aa2e542c26ffb9ccc7c736f1fcb 2026-11-15 feat(serving): enable qualified temporal B8 prefill
fc9ba86567522b1a9171715ce520b820d7cad60c 2026-11-15 docs: complete temporal serving foundation
0027c282fb28fcf846c141f032ea0e4e4fdee599 2026-11-15 feat(state): add paged COW block-table primitive
fc99e9c3354500d1a8f946ad1c19d1f2e0b80d31 2026-11-15 docs: checkpoint paged state COW foundation
54340a6d848ff5f19bb19333f550b1c057328881 2026-11-15 feat(state): bind paged tables to resident pages
818cbb67a55518e3b44ad5c6393e85c67311b887 2026-11-15 docs: accept resident paged ownership
1eb6e0f669f7bf6f31e4a0b7cdbdffa14ef73b7c 2026-11-15 feat(state): materialize paged COW appends
2c908aef374d817825acd7cf4909358cfe8ec7ad 2026-11-15 docs: accept paged COW materialization
3ac205991086e4d9e599993bc7960010d2f8706a 2026-11-15 feat(state): add paged KV gather representation
5e6b391e8b6f417882affb54875439fc0fb19094 2026-11-15 docs: accept paged KV gather boundary
dc6a188bad8d30a28d6871f0798ae98c4a6803bd 2026-11-15 feat(state): add paged KV staging scatter
db3c155f52d6c2c16927208ff0010be03dd9d447 2026-11-15 docs: checkpoint paged KV staging bridge
53336b3d961f2ffc59c0b845f876ff4e8e0033c9 2026-11-15 feat(state): admit paged KV session domains
4e5aaade5bd83700f907d4f44e380dbbec20bbd7 2026-11-15 docs: checkpoint paged KV domain publication
ef9e4d4c4364df423d4a6afad108a1c5b562d967 2026-11-15 feat(cuda): add in-place paged KV staging mode
240e0f23def6d0993c9022a8c43d482cdeda58ff 2026-11-15 docs: accept in-place paged KV staging
93468b1f5062804f009f817c6f5dd50bc7d0516e 2026-11-15 feat(state): validate hybrid paged root semantics
4bafad794e36cb2c4b9647d8807bfc0f8818b4c4 2026-11-15 docs: accept hybrid paged root semantics
cb57e1d8bd4d33b530c4efc771e19450df1ef55e 2026-11-15 feat(state): construct hybrid paged state pools
9815c2abc63b81cf7ecbbc74041356d9e8a0f10e 2026-11-15 docs: accept physical hybrid state pools
b9b04aafd9d10b0325ec5daac45cb3f6be8fedf0 2026-11-15 feat(state): execute hybrid paged candidates
d944eae24cb0686d0d9eb21979f93984bac9c09f 2026-11-15 docs: accept hybrid paged candidate execution
1413f729ac5db55131b8379752b8cdf0a7b98263 2026-11-15 feat(state): add explicit paged source policy
919538ef25d338decfba1d5db54c5a2c3a5801e8 2026-11-15 docs: checkpoint explicit paged source policy
398e4461fc3bd30a84e050951dbdf204b05f0cdd 2026-11-15 test(state): qualify paged source lifecycle
3a1596db12d0f2e5d104ed1980f57af9ae27a513 2026-11-15 docs: checkpoint paged source lifecycle qualifier
30414571c5998880d99c7f685227b83434e6c03c 2026-11-17 test(state): fix paged restore qualification
30b311c1a0ca717a87d53a184b88fad5f29341fc 2026-11-17 docs: accept 0.8b paged source parity
e6aac539edcf631199bb51628c7f9e361e65057f 2026-11-17 test(state): keep paged lifecycle cold-only
cb35ce220edc1d69fa17d80050992f630dec997a 2026-11-17 docs: accept cold 0.8b paged lifecycle
a21ef0fecdc9727bcd1e8f8dbd742975cb3b4a5a 2026-11-17 docs: correct dotlink signal attribution
28890d1526538352423ff376f53042ad4636462e 2026-11-17 docs: record 2b paged sandbox blocker
816e123c3fb68b2322aa3f23896a936a5d3c358f 2026-11-18 docs: record unrestricted 2b qualification route
8b65a8ddfb55e20d948d700fe487ed03e94011fa 2026-11-18 docs: accept 2b paged source lifecycle
c6ba1c658e77dee6315c77124ff9d15e8df284cd 2026-11-18 docs: correct heartbeat harness attribution
6b920a0a91e534eae86f81e603cf419dea3b207b 2026-11-18 fix(harness): reuse existing sandbox isolation
2e0f5f8cd35e1e2896bb5b87fdae77915e566907 2026-11-18 docs: checkpoint native gpu harness isolation
e65f00a88f8ee92e2b175e3db96f932df59d2a0f 2026-11-18 docs: create full comprehensive handoff 2026-11-18
cfe734af1dc7e478c826e70469d917812d36dd9a 2026-11-18 feat(cuda): retire in-flight paged cancellation
3e8c4e71c38f23be304756734726f6ef9e9283a6 2026-11-19 feat(cuda): expose explicit paged state policy
7c13aa362f10553b8858c15a9dbd8a60f8b5a28e 2026-11-19 feat(cuda): add paged native cohort transport
3d4de835e1bddaaafe622634789b329b6e5d7ddd 2026-11-19 test(cuda): qualify paged native sdk
4688cbae9e28838d3b21ff1674f4c48dec519cd9 2026-11-19 fix(harness): recognize isolated child netns
2452eee189581d7ecb608ac5b937499222b4577d 2026-11-19 fix(cuda): restore native feature boundary
ca8fa69ecd8e58fe7f3dd0720839bdf58b91ae8a 2026-11-19 test(cuda): qualify paged native http
30aa1ae7903ea772f12beef8535dadc784f6de17 2026-11-19 feat(cuda): fork complete prefix snapshots
8c8affa9a3cdf7f2a9b9b0eec13a0392f8809be0 2026-11-19 feat(engine): add bounded prefix cache
c24246ddbc4aa285a84f93be35ccd8336119b875 2026-11-19 feat(engine): enable scoped prefix reuse
6303db8ee38f1c7ab5742d17a1c7e05138a70137 2026-11-19 feat(serving): scope HTTP prefix reuse
f83bcb9a7ca8e7a683b02ad9770114873709c34b 2026-11-19 feat(engine): bound prefix cache retention
c0d0e69d792169340b64e7e013013911d2332086 2026-11-19 test(cuda): qualify paged pool pressure
c347fda06f5c267b1ae1c7106447c9e9b66de8e8 2026-11-19 feat(cuda): add owned secondary stream events
064b99bc0531e0d02b8c63b27e1fe82c670b44ce 2026-11-20 feat(cuda): retain resident event-backed flights
e1080c0a17a059070089edc2eccf9468053a797a 2026-11-20 feat(gpu): support bounded resident multi-flight
9147b3fbfaf37266097e14631597f56f7b98d6f0 2026-11-20 test(cuda): profile native B8 serving
19c012f924856950401c706ad23286fd4a8a02b7 2026-11-20 docs: correct Qwen B8 prefill diagnosis
05fe44da6af13e44e38d4ed5b3dea6570b5cb5f1 2026-11-20 feat(cuda): retain stateless prefill cohort lowering
f0f06402c78d4511dea1559c73d4295f88790f93 2026-11-20 feat(cuda): batch multi-token prefill attention
1222f8a11f0c773fba41a502b84ce0862d30442c 2026-11-20 feat(cuda): batch multi-token GDN prefill
60ba956c3057019ad615c5b25ede3a9f7167d955 2026-11-20 docs: align serving perf feedback profile
3afb9902f674d9cfe2fc4488e628b61b61cae395 2026-11-20 fix(cuda): retain GDN layouts for reduced bands
7beb1be30312680720eaafc9abb10eca23648e51 2026-11-20 docs: create full comprehensive handoff 2026-11-20
0dac5abfd3a8edcd82d955a6a125c1f2c5426ec7 2026-11-20 perf(cuda): zero paged state on device
0680b0b8e757ebeb7003f315f08433d7c8bf299f 2026-11-20 perf(cuda): batch paged KV gathers
e72030cd04f71ad7a3cd0b11d7d771875a3f9d88 2026-11-20 perf(cuda): fuse paged gather tail zero
dfc3e396c9e969c48a1a55bd6d35cb9a8ced1924 2026-11-20 docs: record continuous refill goodput
4ec3a0cb36f0a2affd5b092d1262952614428219 2026-11-20 docs: record x2 ROCm runtime blocker
5a95da56316b9e8757175ef01a97472bc656491e 2026-11-20 docs: reject mixed prefill regrouping
9122d1e0d4b6b04d0d99b6d5d7ccc083015237d5 2026-11-21 feat(gpu): retain cross-model perf candidates
29a6238b6c2b59f019928672b3d169cc318111fd 2026-11-21 feat(gpu): restore dormant kernel candidates
468a36054f6004eaa42d1d7dc14dc52df029af3e 2026-11-21 test(cuda): qualify populated 4k 8k serving
127dae431a7db55d9f38dd305b87b94e5af88401 2026-11-21 docs: record populated 16k serving
5466577d3976d2ab56471a27c614debe8ccb2425 2026-11-21 test(cuda): retain current 32k context qualifier
e79814ce0c0a17a93e412b45d7dd66418e6acfd6 2026-11-21 refactor(errors): preserve resource exhaustion context
603844ba0556be3491828b9495535f8c9d1dd183 2026-11-21 feat(cuda): expose exact source admission inventory
973aca4adb7f2ad2dd7246ba1fc51a23c8e44826 2026-11-21 docs: record persistent external gpu sigkills
f86538345fe4ce412df81864542c5ec183446460 2026-11-21 docs: record 32k trace pass and external kill
f1cfd26a2f64e04f6d9ba6f7cf313b15480e47f3 2026-11-21 feat(engine): reconcile exact prefix frontiers
763eb99d58c815bcf1fe22eefb710b9cf8ef6da0 2026-11-21 feat(core): expose physical buffer identity
82d137ec8002af161cdcfeb3f680ba2608c1df54 2026-11-21 fix(core): scope physical buffer identities
3802b9de1d7d6157165d4da76d07a0d76e03de33 2026-11-21 refactor(gpu): centralize resident alias checks
e8c36fb7d5065d63b8e31a221bb6f2a7e4f75d31 2026-11-21 docs: localize 2b 32k response timeout
9896a85832c3db9338cdb9e114653e1e27ae3bc2 2026-11-21 perf(cuda): drop unused attention reserve
be02ade8e81ff356c9bca5cbba1796d08ec692ed 2026-11-21 perf(cuda): price reachable paged state
0dd844044497ced882c59595a49dff8cdf4ffd86 2026-11-21 feat(runtime): add smart model startup config
28459c231764277c141f7fb078b29dc9ee0d403a 2026-11-21 fix(cuda): honor live provider headroom
3b5f191eb94b027179604d04ebac7a09b6b1050b 2026-11-21 feat(runtime): smart-select native cohorts
849cb87d1f295c8acf6dcf4d8bf7638e830ed117 2026-11-21 docs: close exact partial-tail COW ownership
6c68a361e843c73107a86aaedea5ca697a30258c 2026-11-21 refactor(cuda): use ledger identity for prepared leases
f9acd9e49a5eaf29e02d89b2c6f31ec468871ef4 2026-11-21 docs: consolidate 21 nov comprehensive handoff
b616429a41d26a4fefac209bc487dcfa9191c9f1 2026-11-21 docs: make handoff checkpoint self-stable
f42d7722eb4f4f0f392c9b6e6348555162eb2f7a 2026-11-21 refactor(cuda): use ledger identity for streamed head
968a3682492e2675e7837d13c767905f895fb2ae 2026-11-21 refactor(hip): use ledger identity for source calls
b839920ec0553c4a27dc753c076487f0ac2c482d 2026-11-21 perf(cuda): reuse pinned source staging
8c96a886cf8746ed2ad49208c2c43aaa94e4fd7c 2026-11-21 perf(cuda): upload source pages from retained owners
b88d3b3728b4eea2e249f4680be9c83edfeeb034 2026-11-21 refactor: clarify model and cohort terminology
f3f000055cb949f7db367b3d1bc427c472faceb0 2026-11-21 refactor: name qwen source execution forward
001b4c6fbee3b70770d4065ca4cd8a124982b93b 2026-11-21 refactor: clarify text invocation terminology
e8cf4d89feaa4fdb6f02ea9191620c25053be9c0 2026-11-21 refactor: clarify autoregressive terminology
d2adbaf3cbc04dd41c27fd56b90222b786c25dcb 2026-11-21 refactor: name complete state coverage semantically
ff099e626882201fbded10de3a2f4cbf7242497d 2026-11-21 refactor: name model-wide coverage semantically
3fb731ffab12d01ed3f58ffe7e6a4c15c5e664ba 2026-11-21 refactor: name fixed prefill trajectory semantically
c0cc636c10e5378174750896c61602ddc0855348 2026-11-21 refactor: clarify fixed-prompt workload profiles
4e03edb9a08e9a17772bfd2074620448956c5c57 2026-11-21 refactor: clarify cohort and concurrency terminology
2a87e3dfc77dd8aab894f719a8a54b89d944a96f 2026-11-21 refactor: name zero-centered rms explicitly
fb90d9ee199c373d5354101079e3394cb7b52694 2026-11-21 docs: record rejected cuda source-page runahead
37ec699c4ba67a5cbad4b5088b2a17902669e997 2026-11-21 refactor(gpu): retain backend transfer retirement primitives
fdcec908987184a2e2248b4da60b30c22472a040 2026-11-21 feat(cuda): restore reusable gemm candidates
8dfaaf05560e39e2c9916b7ec27ee3dd90db80b8 2026-11-21 feat(bench): add nix-native vllm cuda provider
4201095e9bb4b9704a486a151145c31adaa7a326 2026-11-22 docs: consolidate 22 nov comprehensive handoff
15f5c4fde2bc6659f47a822e4b42844a8077f21f 2026-11-22 feat(bench): realize vllm provider with nix
461ec52698e1907258408dd377a229e07084da03 2026-11-22 test(bench): add public cuda population benchmark
169a92a519afacb374fe2570cfc72b01544d042f 2026-11-22 feat(bench): support raw completion workloads
277a27915e5392e6fe950961e2bbf94c4ab4025f 2026-11-22 test(bench): add hip population benchmark
93d80f3686801bad417f19ff0372f75328046bd6 2026-11-22 feat(bench): add nix-native vllm rocm provider
4d55d231cf765f4e4448c81cb8709b5f5a81df43 2026-11-22 docs: record nix-native rocm smoke
c8b1bed7762b1c28262ab2254a7c87686cca4eed 2026-11-22 fix(bench): provide rocm triton host compiler
6e2382fd2ab53ef23a426ce0816e8da778fec75e 2026-11-22 docs: record rocm vllm c1 smoke
2a595be84bc7c5dccf7509e35e4f9b3a341662a2 2026-11-22 docs: record matched rocm vllm small models
b34c025d36d0e7eaca6d559504e498f998552e19 2026-11-22 test(bench): establish clean current baseline
94a7e2d02db72b3f0d172e46b14eeca7da05e33a 2026-11-22 feat(hip): batch source decode projections
48d3d944dc003b181c915cc013348ca93f0f137f 2026-11-22 fix(hip): size decode cohort head scratch
79de45a3ff637ffe096e18a7f202a701edd982b9 2026-11-22 test(bench): observe hip decode cohorts
cb923cc2b7e79b1d40df705c78989c4bf2abed83 2026-11-22 fix(hip): price decode cohort projection scratch
189c7187601115b7915ac092ca23d595607ac546 2026-11-22 test(hip): accept fused cohort observations
b6c60ef3b1aad3ce828519801b748dfe8d4f15f5 2026-11-22 docs: record hip decode cohort checkpoint
99dcbbee20b50b8e588f167cf2c0ec40b31782c7 2026-11-22 perf(hip): batch row-wise decode operators
e35e650c6bebeeb814c770f8aee1a031a819415a 2026-11-22 fix(hip): compare decode cohort semantics structurally
537e5e8542f6d07588dddcbf52a5131f1fc5a6e3 2026-11-22 docs: advance hip decode cohort baseline
9ee877c7ba87ad233228ef71a2803ed4063996ec 2026-11-22 perf(hip): batch aligned decode rotary prep
a5ec4c31eb39d480bee499bda6b7757fbf9b4d74 2026-11-22 fix(hip): seal batched decode rotary variants
a3908378fd217124fdf1581763002df190133b75 2026-11-22 docs: advance hip aligned decode baseline
1de99ef6210941a662bc0fc5155092e7f366fddc 2026-11-22 perf(hip): batch independent decode conv state
c33016d649e850d33b13e9f5cdce254e35924353 2026-11-22 fix(hip): preserve conv coefficient input contract
57cc384c4ff55f5576ccc9a321fe9600cf70ce24 2026-11-22 docs: record stateful hip conv cohort
491cf9957d83bc46100cc863b0b062909459fbd6 2026-11-22 feat: unify production GPU build and run paths
6f8f07111fb38eb910c051063cefe76f8c1b75bf 2026-11-22 refactor(gpu): separate attention semantics from providers
64c06e2fc313d3b24c0a5abe0065a085879f0e90 2026-11-23 feat(cuda): add opt-in FlashAttention provider
ed0382a192ac882be0fb8e2346e23a4bab252284 2026-11-23 perf(cuda): keep FlashAttention off decode path
0a35f175629f0841b98d78fc156c0c30eaf88df3 2026-11-23 perf(cuda): fuse qk normalization and rotary prefill
2805c37577a185d7065fcb9397f57827d1010d90 2026-11-23 perf(cuda): enable fused qk rotary prefill
179bb1528339c258b7cb125086ac6fc692698d18 2026-11-23 perf(cuda): fuse GDN recurrent export certification
c829aa9855f3eebfc0a6635122d6448c781d27e5 2026-11-23 feat(cuda): retain B1 direct GDN conv primitive
f3f872cb567a6885918ae2b4866ee6d535d736d8 2026-11-23 feat(cuda): retain B1 fused head greedy primitives
fb20345bb7b5ce0b3285470c77564e492331254b 2026-11-23 docs: consolidate 23 nov comprehensive handoff
0c4d95e6a1fce0e162927ef2b55647fac7d1fba1 2026-11-23 docs: revalidate production feature convergence
6f4418ba8262baff0dc023923865ea8bcfb7d63c 2026-11-23 fix(qwen): allow tokenizer-owned chat prompts
2d16cd5aab95949cd295070a8a5df84172409f29 2026-11-23 docs: accept a1 hip public canary
11f692bf7b2871e1dac974eb96bb27f3968ee316 2026-11-23 perf(hip): batch recurrent GDN decode state
7fcc9c4739b9bceedb298d4835a55ad5b64c1a84 2026-11-23 perf(hip): retain recurrent cohort primitives dormant
31eaf496385890005f04a7c920ea5b9ee367404d 2026-11-23 docs: reject hip recurrent cohort selector
69630ec3426d2ca523cdbe5e2183b01f68b93f49 2026-11-23 perf(hip): avoid dense recurrent decay scratch
0d201ff0bac05aa92d48dc8faa4e4cbfa2cce7d5 2026-11-23 perf(hip): fuse cached GDN coefficient exp
879bee798a29f5da5a0fa4d2812840604f0bad76 2026-11-23 perf(hip): keep singleton recurrent primitives dormant
4070610433a3068943e844b2135d9776beffaa47 2026-11-23 perf(hip): write singleton projections directly
fdc5a8ec86aa9861576fbc33bfbfb4a4cc4c2ad4 2026-11-23 perf(hip): reuse packed singleton projection slots
baf4c9f090e4106cc4a35439a30854328adfaea3 2026-11-23 perf(hip): keep projection alias contract narrow
ec6fb57f08b948c804f869787a5a17f52e038846 2026-11-23 perf(hip): coalesce singleton projection status
d15b02f6973e36baedb4b6a374058b48e2e76795 2026-11-23 perf(hip): retain projection status coalescing dormant
e606586dd3cd7ef76c2ec832bb5cb76c3b9c87af 2026-11-23 docs: record hip singleton projection A/B
e1701e707de5b23589e24509b5b401cc1f138720 2026-11-24 perf(hip): reuse singleton projection input pack
1d49ec7c4760e0f00c9af2f6d0c7babf250a2530 2026-11-24 perf(hip): retain prepacked projection path dormant
5d51896532574a08d22912b44da127a6564e3cec 2026-11-24 perf(hip): certify singleton projection in-kernel
90537b6c2da8f626e08e7665311d49e249c4bad6 2026-11-24 perf(hip): retain certified projection primitive dormant
9e91a0d26a7b77456411b15074e77098f7b754c3 2026-11-24 docs: record hip projection and native accounting
c5b3fadd214d9d89a84e6c6d7da8fc756bf5fc76 2026-11-24 perf(hip): qualify gfx1150 native module allowance
fd4611e12fc1c2f173b2d80d4e8d2105fe601327 2026-11-24 docs: accept gfx1150 resident hip policy
ffd5ff8f951c5db9b113865c6d5a22a954e282e7 2026-11-24 perf(hip): fuse residual rms first pass
dfc57183f8fe36f00e065e9b7c1482b6480488ec 2026-11-24 docs: record resident hip hot path
b44a55c868e72e04f68d7e900ee32b822cc89c32 2026-11-24 perf(hip): fuse rms scalar cascade
11e61f850557c4303a95c2686a8afb6af5de3591 2026-11-24 perf(hip): retain rms scalar cascade dormant
9040492544f0ea6e114fd749232cc2b185aa8a65 2026-11-24 perf(hip): fuse rms vector tail
654adbb17a35d66922ba918477f9b872b704b605 2026-11-24 perf(hip): retain rms vector tail dormant
288e978666463148475900b9db167f41ee5af99d 2026-11-24 docs: record resident rms fusion A/B
7ff8784c8d454af634a143d30d228e8164d78527 2026-11-24 perf(hip): reuse resident singleton projection pack
36c318db569916ac8bf264e33b4d9dc25dfd08dd 2026-11-24 perf(hip): keep resident projection packing selected
1b25fc30cab76b93e1bb02f3b09e1f6b20bc5ef6 2026-11-24 perf(hip): share singleton projection input across four rows
5e4cac97fa78eefb1f4efc721bfee1ae0c3317a3 2026-11-24 perf(hip): retain four-row projection primitive dormant
a5c57c13308cf86b2daa0c2cce4e463f1fb4c447 2026-11-24 perf(hip): parallelize cached gdn qk normalization
072b79128163e2c18a0bfa946099875b13fd9122 2026-11-24 perf(hip): retain exact wave32 projection primitive
b4fc23188a10583588a610d3a6657a37fee7cde6 2026-11-24 perf(hip): keep wave32 projection primitive dormant
d6043af75b9f0e557afee142176df3f3454b16e0 2026-11-24 docs: record resident projection device profile
253a565b543447833bd113309098f46ee113155e 2026-11-24 perf(hip): retain exact rows2 projection primitive
6629ee26a9179ac2b3ec62a1e6ef5996cc6e6b04 2026-11-24 perf(hip): keep rows2 projection primitive dormant
a8c92a2984c6f1de04fe901e367b174c1319eea3 2026-11-24 docs: record rows2 projection A/B
6d5dc72d3711cd3b0e14db0a3c007851984a0745 2026-11-24 feat(cuda): integrate flashinfer decode and paired gdn state
55477b4b8b6e52db3b11450cb8fdc68fffbb5cbb 2026-11-24 perf(hip): retain paired packed-load projection primitive
ba8c5ea7ac6fb8575f102607cdefd0155de510da 2026-11-24 perf(hip): retain input-certified projection primitives
49e5149887a6687733b7dfb9420eb8f5575043d6 2026-11-24 perf(hip): keep projection traversal alternates dormant
e1929762a87b1e34de099d9f40b85bcff385c27d 2026-11-24 docs: record projection traversal A/B
f6bd058c87ceb4eb5fbf7d995fcb07a84972241d 2026-11-24 docs: add 24 nov comprehensive handoff
d717c7b006c0be9d189773c073d7be27b2a8bcc6 2026-11-24 perf(gpu): retain exact packed-index projection sibling
4c2165fe43bae068338a04099f737474d17b2218 2026-11-24 test(hip): stage temporary resident projection A/B
224085b1f3b24d5abcc029d256e32b67de391e68 2026-11-24 test(hip): seal selected projection A/B kernel
3be8a812bed826b747d50d35ad516facac78e38b 2026-11-24 perf(hip): select sealed resident packed-index traversal
b0522cece95599a03a9da4e75712b014cbcf951d 2026-11-24 docs: record HIP projection canary and combined gate blocker
c89d0ee73696e81c4f3391e471ed4e9a9d7cdeae 2026-11-24 fix(gpu): restore combined backend feature boundaries
fd02260c24af860f0c4ce64ac5953c130fcaf923 2026-11-24 test(hip): stage resident real-model kernel localization
43c9ab8cfe10e21b71b90899c1af11c915947c16 2026-11-24 docs: record repaired feature boundaries and resident HIP profile
7d0b395b3af245dd375f6b49a6c8c345016726d6 2026-11-24 perf(hip): retain exact parallel GDN key subtotals
7d3a130bb7828ecd09908cd70bac7d7eedf50369 2026-11-24 test(hip): stage sealed parallel GDN reduction A/B
e81a79ec9d95257c757913ef0b706d9afc1c2605 2026-11-24 perf(hip): retain parallel key reduction without selector
6f2ededf523c74ea190b8a3f25993cea4f6695c6 2026-11-24 perf(gpu): retain exact two-barrier projection reduction
971b58e756860e522d956a134091ba77c79869c0 2026-11-24 perf(gpu): retain two-barrier projection sibling without selection
42d4703d38135d53d55f23de7ee7193eb112552f 2026-11-24 test(hip): verify native set refusal by ownership
11a44d6034a33b1041643b5e142a3a6059b355a5 2026-11-24 test(hip): sample physical native preparation allocation
38f20ca39039f934614a41f18f6fc486f8818cfe 2026-11-24 perf(hip): qualify gfx1151 native module allowance
38978663d116733f40f3d33176eaa0fb1e78c34c 2026-11-24 docs: record X2 resident native qualification and baseline
533c72cc47a5791e54c6748dae6a7836c6dca515 2026-11-24 test(hip): stage X2 prefill and decode device localization
f7ebf647b23e5a96d2d638a270f05dd922fa74b2 2026-11-24 perf(gpu): retain exact request-tiled projection
c1cd04f3339e08c2422afad972fae5f0a38a2dd1 2026-11-24 test(hip): stage gfx1151 hoisted projection model A/B
99eb79905106d39c2e46d819918159a24a556cd4 2026-11-24 perf(hip): select hoisted projection for qualified gfx1151
7d3e4b4bb57e92f0a85758af221f57e6f324e217 2026-11-24 docs: record X2 hoisted projection and prefill localization
393d0f56fd23669e84e6adf521921af133096393 2026-11-24 perf(gpu): retain exact guarded triangular arithmetic
cd86a9655cc3a9f2be58ac16ff81a41e7c3127b3 2026-11-24 test(hip): stage immutable guarded GDN arithmetic A/B
4a5961dadf11e12188214d880f3f9dbd09414f58 2026-11-24 perf(hip): select guarded GDN triangular on qualified gfx1151
720f467d82420cd9ceb0db9600fadce4f2a9b14a 2026-11-24 docs: record guarded GDN triangular qualification and baseline
2ed9c10f2443d8727e293e51c4c8180cf97d8eda 2026-11-24 feat(hip): share projections across aligned prefill cohorts
282a3437ea822c58b433edf896485bc5becd63cb 2026-11-24 perf(hip): share stateless prefill work and direct projection outputs
2eee25fab1d55a365c065e7fbf314d51624b9870 2026-11-24 test(hip): validate distinct prefill roots and readout against singleton
47b9f6f65a6e9846ef0ed73560ee467c1316b8b8 2026-11-24 test(hip): use admitted context for prefill differential
fd914f63a546f6e69bd8628a39de19d7a271fb8f 2026-11-24 perf(hip): bound resident cohort projection token tiles
74619a0b31c2aa09e416614bcd42bf5332edefa8 2026-11-24 test(hip): exercise shared state-only prefill through public scheduler
909cf1e611c2605a10d288b9e334b1a4b95105fc 2026-11-24 docs: checkpoint aligned HIP prefill and resident tile results
e80216b02a121ac1306eaafab2a09717b2deb52a 2026-11-24 feat(hip): add native BF16 WMMA projection primitive
2514868a6ee5f08ef1e9ae15699abf2ac974e7d8 2026-11-24 test(hip): check native WMMA mapping with varied integer rows
7046e618fbb3b0e3d922aa0226cc1767d6dcd3b0 2026-11-24 test(hip): localize WMMA fragment output mapping
591913f545753e2245a210754ebbbedac647552e 2026-11-24 fix(hip): materialize WMMA vector element before bit conversion
7b37d338aff3cba28b9cc6cbb6fdda16bd5bac33 2026-11-24 Add owned HIP rocBLAS GEMM provider and lifecycle qualifier
9b9e49968058297044b2173c46b80137578b006d 2026-11-24 Use optional device duration in rocBLAS qualifier
e1358b10b0c048b401fed4c46ebc35759cfa488b 2026-11-24 Prioritize native HIP matrix execution over scalar microtuning
312bbccd467c3181aaf5ad451b9e3b08232813e4 2026-11-24 feat(hip): integrate native BF16 GEMM into resident source execution
58a10d3c9a50be82ab0df9a7691239e05d070124 2026-11-24 feat(hip): expose native CLI profile and use full cohort GEMM
2d3e8c5f6ea61bbae571dc46fda91f1410de6c2e 2026-11-24 test(hip): stage native projection whole-model localization
6c31363d42fc7de386f223f2e39869b36389c911 2026-11-24 Use native strided batched GEMM for dense GDN chunk products
6a4f2258db9e05442ca161ebbfe98f0b54bf743a 2026-11-24 Record native model execution baseline and provider priorities
e1901b16ee7227639be69b7bd41498f2f2b53290 2026-11-24 feat(dense): share native matrix commands and batched triangular solve
79a9073a246a91739fdedbb9c2b91107aa894335 2026-11-24 refactor(cuda): lower owned BLAS through shared dense geometry
9a571088c8ca696a7514299d09dc70c4a767414e 2026-11-24 fix(hip): validate dense read-write operands before claiming writes
aabc3d4f132750d4943e90b2379fee9a25c965c7 2026-11-24 docs: checkpoint shared dense provider and native solve
6f677aa347c3595f5d52cabd66f6c2305c782645 2026-11-24 docs: consolidate foundations-first native execution roadmap
83f7419dc3d33f93b12e64847e33aa010d555799 2026-11-24 docs: normalize implementation plan header
668a3050d5205997f17332af4b2c46cd39908aff 2026-11-24 docs: match architecture diagram to dark interface style
8189cebbbbf49874c8318bff8be94666a22c599a 2026-11-24 docs: describe final architecture independently of roadmap status
48fbbb0d60a250b40835d630ed1a44326f68f384 2026-11-24 docs: unify native region lowering and optional graph import contracts
8c34c4725679af32fbda08b7f7dbdf61cfb0a8b3 2026-11-24 refactor(fabric): prepare borrowed and window-backed source programs uniformly
e0ef64cc3fd42d77fd1dde7db32a82eda71505ae 2026-11-24 test(fabric): complete source preparation validation checkpoint
17e2b7176eaec5b9a5b78aded583690795e81c30 2026-11-24 fix(fabric): close source preparation acceptance gates
0094952d10bbe43d9dbb589b65a19c10c9b66c06 2026-11-24 docs: comprehensive foundations takeover and historical work index
82f895dae795f2bcabf83a95ce52cb6705e64280 2026-11-24 feat: bind closed stateless regions to original model sources
1ec6b94140a13ea649cc2c8b9bbda112293a2151 2026-11-24 docs: record closed-region foundation validation
0db60ede170634d6be5f9ffecd82939c2745ab95 2026-11-24 feat: prepare typed physical region ports and original lease views
053a7c5780d2e1e2f7dd04407ac07520b16454d4 2026-11-24 docs: accept prepared physical region bindings
77e8b4ae0487a5219140727007cc316f43c01663 2026-11-24 feat: bind provider resource requirements to original flights
c76708c7b827097f3494fb2fc6fd5b20d2e9a1b6 2026-11-24 docs: record provider resource and completion validation
262712f1c94daa8423060155633545b632a54e88 2026-11-24 feat: bind source state regions to original root attempts
6c94694e86b9e01f33e452dea8045279c8492032 2026-11-24 docs: record source state region validation
6eae81c68c24c58c85e707baebacbccbda523208 2026-11-24 feat(cuda): bridge private provider transfers and stream completion
32ed4786671ae360729651373baf743808c9c786 2026-11-24 docs: record private provider bridge validation
7e86aad7d74882baa474a50a584e141f1d58eee3 2026-11-24 feat: bind prepared text calls and producer observations
d8a046fe7e1218532db33a425fcb6cdd83d645d7 2026-11-24 docs: accept prepared text execution checkpoint
f4746c2350153b518e4730c3448e8f4ede02eab8 2026-11-24 Bind cold native source admission to selected device and actual fit
f74e599788c313eb1fccdc38d00719c2f754e91b 2026-11-24 Document accepted native cold admission checkpoint
057683d1c315fa026e0d4787b073757ca8caf699 2026-11-24 Unify admitted local model sources and retire sole catalog owners
139bdbdf667de6eead31dfe02acb61e60a725e13 2026-11-24 Report GGUF canary worker errors without Debug requirement
9e5f678bf7bd808b1ff0c866d4bd379e2cd1476d 2026-11-24 Document catalog retirement and shared local source acceptance
2b6656b3d427092dea5bcdbb801bc71e59ba000c 2026-11-24 Retire independent model owners without stopping shared workers
4d6645c8dd311c20b925fb2307e9029127a3724c 2026-11-24 Document accepted selective model retirement checkpoint
bffad06a9c11efd497b659cff7b735c00e08e718 2026-11-24 Expose typed loaded model and runtime resource snapshots
94b205b93681cf340ea719af25a519fbbf62d143 2026-11-25 Document accepted typed runtime observation checkpoint
4477f20ce7de60d722eb6967e1f33b5e565cb154 2026-11-25 Bind serial language pipelines to original loaded model registrations
6bd6a04d4f367b8c29b8d3ff4953c2bae20eabc7 2026-11-25 Document accepted source-bound language pipeline checkpoint
2df8cbb3f11ba698278b3537fc4fd16ca52908b5 2026-11-25 Bind prefix cache to original prepared resident state
4306560392b6c2e82b2ddb83519a7ef8b8429df3 2026-11-25 Document accepted resident state identity checkpoint
7b36a8795588b41d01634f823965427fb453bc15 2026-11-25 Share authenticated resident and streamed source window transport
0342e1038749dc802cebe7159962dc7bb1967d07 2026-11-25 Select source-page transport from original device in CUDA-only builds
fdfb467f7ec86ddb891f78f562698675875a3565 2026-11-25 Compare source transport with fresh admitted CUDA contexts
3cd320f57f9274827a528c3dc708f7c02444f718 2026-11-25 Document accepted shared source-window transport checkpoint
48d6a9d334a77733faa6e7da15f12227896aacf8 2026-11-25 docs: consolidate accepted foundations and complete takeover history
687d9a2467af9479108b6069c12e68545032630b 2026-11-25 refactor(source): retire parameter backing at physical initialization
06df027fd126a57368984c668dc7e277d3f442bd 2026-11-25 fix(source): preserve padded parameter views with shared transfer backing
43928332ac906be64372ba9b3c62ade2bb32ac60 2026-11-25 test(source): observe retirement after consuming model shutdown
f4e09428cd8d1f77302a24e18050687fa8fba6e6 2026-11-25 test(gpu): assert typed physical admission refusal
1664751483a047c8986dd39bed85dcfb13670722 2026-11-25 docs: record parameter materialization and retirement foundation checkpoint
3ca16abd538fe8a4ffa0fc07b472865756392bf2 2026-11-25 feat(provider): bind original source regions and explicit compiled selection
a53a3446dc271c1081ea0e850a851907435e7c74 2026-11-25 feat(sdk): expose direct region source binding types
2a63b875388661aeb8a289453a287463783e9e3d 2026-11-25 docs: record compiled-region foundation checkpoint
8eb17bc2125ed73260240db9cda47312c501ac39 2026-11-25 refactor(source): separate authenticated model eligibility from execution selection
00f57e7a3d75bd83a980ed333211a45bfbc1f7e5 2026-11-25 test(source): expect typed host capacity refusal in real cache comparison
9374e0ca2daeb8537dc96f6e1ebbe2271fbadf99 2026-11-25 test(source): identify private cache budget as host memory
eb32ebaefe7ab8c6ceabf463c1d6ec16c8612dbf 2026-11-25 docs: checkpoint authenticated source eligibility foundations
9ae6dd1eb3c7bdd8c8ffb30479bd7775686a9c2c 2026-11-25 feat(gpu): admit CUDA physical envelopes before client bootstrap
e06213a517e1a88732fc5ceadac8fd1b9a97b354 2026-11-25 feat(gpu): expose the admitted physical device identity
f31ba4db038e565f39958155c288d19797d3d043 2026-11-25 docs: checkpoint CUDA physical pre-admission foundations
cbd2c1198a49ad800945b33ed01ba45ba15dafcc 2026-11-25 feat(resources): share host backing and intersect cold local memory fit
197c2e26f58e27064d9b574ae8050ac41ab5919f 2026-11-25 fix(resources): check actual HIP cold reservation rather than lifetime upper bound
814d5c006a56b68890fd2f6a3f6b564056723e7c 2026-11-25 docs: checkpoint shared host backing and local VRAM foundations
a654886341e6ae3831ab28675fbf8daddbbf605f 2026-11-25 feat(gpu): preadmit portable HIP and guarded Vulkan physical owners
86a3792a2ed20fe73590f189568e393d55adf18f 2026-11-25 docs: checkpoint portable physical admission and placement foundations
1d5c4ada3e224e7278ddd487563073e2d8d7be94 2026-11-25 feat(state): persist admitted committed CPU checkpoints
f9dd5ed654d77548a51e1460ae668aa45d330c22 2026-11-25 fix(state): gate diagnostic filesystem test import on CPU
35629e6cddc2db727cf181dbbeb899738f77e4b5 2026-11-25 fix(state): restore typed diagnostic resident checkpoint compatibility
566dcf8dd41716f33da1690fe302a2b857d37f30 2026-11-25 docs: checkpoint committed CPU state persistence foundation
804d488200ea12b82e487f9e590f90f0d3055672 2026-11-25 feat(state): persist committed native HIP resident roots
d9b252057ef76cc56bb0ecb37f9fc653a24061f4 2026-11-25 test(state): qualify unknown resident checkpoint upload
21ce3d7f024ca62d71edb3bb213a46a96fcc60b2 2026-11-25 test(state): include resident persistence in existing HIP compile gate
ef7c3d6c0997f10185aeff5b0b0ae56c6a74dd56 2026-11-25 fix(state): retain diagnostic admission through shared publication
de1ef6766d472b13ee4c3236a78898105a6db0bc 2026-11-25 docs: checkpoint native HIP committed state persistence
ebda04c43d6b3f619cad01500edde1869eae8673 2026-11-25 feat(pipeline): atomically load source-declared language chains
212f6eabd6e346a9466b2dd2be1b261c5e4cbda0 2026-11-25 docs: checkpoint source-declared language preparation
b25b5c02347e15980c082719d18a2685e687af3c 2026-11-25 feat(sdk): unify text task controls and selected owner observations
9a5fc6e658e716bda0e585fe2080c1623f49b404 2026-11-25 docs: checkpoint shared SDK task controls and review foundation exit
7987f70a566bfb7a5519b8f1a1013f138881de07 2026-11-25 feat(source): share selected GGUF inputs across SDK and direct preparation
2ec06026853b019876b753f50e38134ea202bee7 2026-11-25 fix(source): preserve original selector refusal diagnostics
24bba9764b4ff7bf92d6239efc7dba01a8bbcf3f 2026-11-25 docs: checkpoint shared selected GGUF source preparation
8e3b8993b189beb80216e8b5a10fd8a5e484f7db 2026-11-25 feat(provider): observe original prepared regions and preflight CUDA transport
33118f162eca2618d97ff2b7fe6885ecae6da69f 2026-11-25 fix(provider): preserve GEMM orientation and require dense bridge copies
1462687e59b19f29d4e193fddd6c58f420a9e620 2026-11-25 docs: checkpoint prepared-region observations and provider compatibility
3b5a520d748bd5e353baea00dcaacd30e83d6793 2026-11-25 feat(vulkan): adopt exact registered clients under original service registry
fc38db0371804a46d189a23fb83f531f3f240651 2026-11-25 fix(vulkan): require explicit backend for named registry reuse
e1decff57e98b55ee0cb09697015c8e8d998cba4 2026-11-25 fix(vulkan): expose original client to Vulkan-only qualification
9017af172f700456beff29ca379763fd7546f8e9 2026-11-25 docs: checkpoint original named Vulkan service reuse
ce83cc18574ab3998d7a1dec84aba6cb300b5fac 2026-11-25 feat(provider): observe actual captured owner and prepared cohort mode
0886f11f2f66aa5d330184bf443a198b1b930d8a 2026-11-25 docs: checkpoint actual captured execution observations
14bbbd489bae8cfa79beae0ceffc07d7a0f8817c 2026-11-25 feat(backing): initialize exact coherent mapped resident views
5d952364f4a2d2ce377644e1255c760f60b05aee 2026-11-25 fix(backing): keep mapped probe in test qualification
5a7535e75b727f4b6699379b698c507b2dbdb0c5 2026-11-25 fix(backing): retain original logical upload accounting
d5030ad4ce476b913a46056d492992ccf1b8edf7 2026-11-25 docs: accept bounded foundations and checkpoint mapped backing
450cea5530dad567ff7706c30e1a1872ff01bda7 2026-11-25 docs: consolidate complete Phase 1 exit and Phase 2 takeover
e6211608e01980bd7e3dd6de8bbac8fb964805a5 2026-11-25 feat(cli): unify presentation and diagnostic reporting
bcdb46a8dfb582a95bee7432a2861c5186e50983 2026-11-25 feat(cli): add typed command tree and shared model resolution
3cf1bb2df27effc1707bd87a836a4648f70e0a6e 2026-11-25 feat(sdk): share control-plane snapshots and actions
da2385c73ba83f5b130a6fd042d185ddecbb9b63 2026-11-25 feat(acquisition): share controllable download lifecycle
8e382947e2d4e78269ec4847a492f0dac6d81ebb 2026-11-25 feat(cli): simplify one-line model execution
84e0681cd07f08173783e61716f2850feed09f31 2026-11-25 feat(cli): unify canonical model serving and listener lifecycle
7f5fc4ebccc7ab6378d8523cd912387a74ad39f9 2026-11-25 feat(config): unify typed profiles and atomic configuration edits
abe98b8379546f1e07dbd8dc97d4db28c9345e4a 2026-11-25 docs: accept unified configuration and profile management
08e7e9d3368a75a76031ddb78e7789fb266ba216 2026-11-25 feat(tui): add native read-only control shell
0c7ad87a1a794c92e4382fbb1e4b9f103cd30b7b 2026-11-25 feat(tui): wire resident model and listener actions
9bf7829ae4f01671d6e78e8b50e23da242714ca6 2026-11-25 feat(tui): wire model acquisition and device management
71e369333f19eee230a1f2a674710c3f8aaf2b9e 2026-11-25 feat(tui): complete native configuration and operational logs
4e472aed38bac2d0139fa4e5162f483d45574ae2 2026-11-25 feat(web): embed authenticated shared-runtime control shell
4f23983a5bb33a3ab2e9ef124342a0cc33df2088 2026-11-26 feat(web): share resident model and listener actions
3333ec5cd462e4520ffc1901a3881785478c86ca 2026-11-26 docs: consolidate comprehensive takeover through accepted UX-9B
a6744b125ea415ae517a3273f00a5dcc5ffd0e37 2026-11-26 feat(ux): complete shared Web model download and device management
543194ac851a8db6118eb42e3aea209403edadbc 2026-11-26 feat(ux): share Web profile configuration and operational logs
1a87899013cac5afd6839f748ee33627a30e3b46 2026-11-26 test(ux): qualify simultaneous native and Web control lifecycle
22c6c0a73860af4f73188d4b6093c932a33f258d 2026-11-26 feat(cli): add cold diagnostics and typed pipeline inspection
1b8de1cb87fa4815655cbb6700e6633946f93305 2026-11-26 feat(cli): complete cold quota and backend diagnostics
3b584f729efc691b8149805785d5c00f81d4c994 2026-11-26 docs(ux): plan native TUI interaction and visual polish
b1073e7c53e5fbeef1c9f346264e7669e02e93e2 2026-11-26 feat(ux): establish semantic native and Web appearance
ab3cf974e4df7538384b873af121a8a3747f9334 2026-11-26 feat(ux): add native mouse navigation and resizable panes
5da962513b051b5296c28e107a321ffb0cbd6afc 2026-11-26 feat(ux): expose bound native actions and modal controls
8a2c2ae88fbda55790c27249c6782bf852c29544 2026-11-26 feat(ux): add typed pickers and native grapheme editing
f723b0b6585dd2a0749895702c123a7d0a74db1e 2026-11-26 refactor(ux): simplify native chrome and disclose details on demand
d1e27080776bfff8bf593bcc9c503df43ff92fa2 2026-11-26 refactor(ux): replace native toolbar with contextual menus and live search
0ee66351893b707de1687511729cd1ac440655d8 2026-11-26 feat(ux): preserve scroll anchors and refine response and compact dialogs
c0e4794d9517f03d989852e945d0b8db7e978448 2026-11-26 feat(cli): enable full production defaults and preserve safe error diagnostics
e96cdab98d47c5c84001c1afaaebd8f126b9e0b3 2026-11-26 feat(ux): show live registration performance and layer execution details
~~~~


### Post-UX-9B takeover delta (15 commits)

~~~~text
a6744b125ea415ae517a3273f00a5dcc5ffd0e37 2026-11-26 feat(ux): complete shared Web model download and device management
543194ac851a8db6118eb42e3aea209403edadbc 2026-11-26 feat(ux): share Web profile configuration and operational logs
1a87899013cac5afd6839f748ee33627a30e3b46 2026-11-26 test(ux): qualify simultaneous native and Web control lifecycle
22c6c0a73860af4f73188d4b6093c932a33f258d 2026-11-26 feat(cli): add cold diagnostics and typed pipeline inspection
1b8de1cb87fa4815655cbb6700e6633946f93305 2026-11-26 feat(cli): complete cold quota and backend diagnostics
3b584f729efc691b8149805785d5c00f81d4c994 2026-11-26 docs(ux): plan native TUI interaction and visual polish
b1073e7c53e5fbeef1c9f346264e7669e02e93e2 2026-11-26 feat(ux): establish semantic native and Web appearance
ab3cf974e4df7538384b873af121a8a3747f9334 2026-11-26 feat(ux): add native mouse navigation and resizable panes
5da962513b051b5296c28e107a321ffb0cbd6afc 2026-11-26 feat(ux): expose bound native actions and modal controls
8a2c2ae88fbda55790c27249c6782bf852c29544 2026-11-26 feat(ux): add typed pickers and native grapheme editing
f723b0b6585dd2a0749895702c123a7d0a74db1e 2026-11-26 refactor(ux): simplify native chrome and disclose details on demand
d1e27080776bfff8bf593bcc9c503df43ff92fa2 2026-11-26 refactor(ux): replace native toolbar with contextual menus and live search
0ee66351893b707de1687511729cd1ac440655d8 2026-11-26 feat(ux): preserve scroll anchors and refine response and compact dialogs
c0e4794d9517f03d989852e945d0b8db7e978448 2026-11-26 feat(cli): enable full production defaults and preserve safe error diagnostics
e96cdab98d47c5c84001c1afaaebd8f126b9e0b3 2026-11-26 feat(ux): show live registration performance and layer execution details
~~~~


## Appendix D — verbatim post-UX-9B landing narratives and source paths

All fifteen checkpoints since the prior comprehensive takeover, chronological. Narratives extracted from each exact commit's CURRENT_HANDOFF.md, including old next steps, incomplete tests and limitations. The UX-P plan-only checkpoint additionally retains its original complete specification as written; later user-driven corrections supersede its toolbar details. Changed paths are navigation, not a second acceptance system. The latest main body and live plans govern execution.


### D.1 — UX-9C Web model/download/device

Commit: a6744b125ea415ae517a3273f00a5dcc5ffd0e37

Subject: feat(ux): complete shared Web model download and device management

Original landing narrative (historical instructions):

~~~~markdown
### Current checkpoint — Phase 2.1 UX-9C (26 Nov)

Models / Downloads / Devices now use the original cache, transfer and
resident device authorities from either client. No new scheduler,
runtime, cache registry, transfer queue, persistent config or filesystem
authority. One shared Actions slot; exact download commands bypass only
the action slot and address the original active operation.

- control_host/inventory.rs shares original XDG cache observations and
  native/Web invalidation predicates. Cold authenticated inventory GET
  /zg/control/inventory/{models,devices}, bounded one Web read at a time;
  models expose exact cached manifest metadata, static target/catalog
  recipe metadata and original partial checkpoints. Device inventory uses
  ControlPlane::devices, not execution eligibility. Safe remote labels
  only; no arbitrary local source paths/raw errors. Partial keys and
  listing are observations, not verification or admission.
- Shared worker promotes inspect/verify, exact-digest dry-run/apply
  reclamation, additive by-name aliases, original CheckFit and load.
  Busy/Referenced/Incomplete remain failures with accurate safe messages;
  incomplete effects are not reported rolled back. Actual removal
  revalidates original cache activity, references and readers. Resident
  materialization may release cache readers: a loaded registration alone
  is not a cache lock. Typed Web grammar rejects arbitrary paths,
  noncanonical digest/device IDs, missing consent, credentials and
  unknown fields. Explicit source/device actions require exact canonical
  cpu:0 or pci identity, not an ordinal or backend override.
- Original interactive pull Intent now shared by Web/native; pinned
  HF/GitHub full commits with exact files or explicit GGUF quant, or
  approved Qwen recipe. Preview does not acquire, confirm must apply
  to the captured intent, offline policy checked before admission and
  in executor. Original quota/recipe expected snapshot/source/transfer
  admission and publication unchanged. No implicit pull, auto-load,
  eviction or quota mutation. Download pause/resume/cancel uses exact
  decimal operation ID, not action ID; accepted command is not terminal.
- SDK DownloadChanged carries observed status at emission so a replaced
  operation cannot hide the earlier terminal invalidation. Native and
  Web observe relevant cache/terminal transfer and load/unload events;
  Web SSE models/devices invalidation or resync, visible shelf refresh,
  one coalesced cold request, no progress/token-triggered cache scans.
  Native private receipts, independent control/inference authentication,
  exact action identities and foreground joined retirement preserved.
- Solid cached/catalog/partial searchable shelves, inspect/verify,
  alias, load/run/serve, removal preview/confirm, pinned pull draft and
  consent, live transfer phase/bytes/checkpoint/rate/ETA and controls,
  physical-device source check/explicit load; settings/logs remain UX-9D.
  Browser selection is client-local and never changes persistent policy.
  Cancelled previews cannot authorize later effects.
- Strict TUI-only/Web-only/Web-only CPU+acquisition/combined lint,
  terminal-free Web dependency graph, JS 3/0, actual cold HTTP/Firefox,
  controlled HTTP transfer (range resume/durable and cancel/partial,
  stale IDs, referenced retirement refusal), native transfer PTY and
  real Firefox pause/range-resume/publication passed. Fixture corrections:
  resident registration is not an artificial cache lock; PTY pause
  waits for a prefix; browser waits for actual empty route phase after
  retirement; PTY count accepts ANSI cell positioning and confirms the
  still-active Web host quit. Final real cached browser/TUI/Web/HTTP
  lifecycle passes with the unchanged 8 GiB envelope. No changed admission envelope or live remote acquisition.

UX-9C optimized cached integration passed 1/0/0 (48.13s, five filtered); full isolated gate 2,489/0/109 (318 Rust suites), release Web/TUI/installer/scripts passed. UX-9C accepted.
No new GPU or performance claim.

**Next: UX-9D — Web Profiles / Config / Logs.** Promote original
ConfigFile typed/raw atomic snapshot/validation/edit, profile copy/
protected remove/activation and bounded operational observations to the
same authenticated control seam. No second configuration store,
arbitrary browser path/credential authority or invented durable logs.
Then UX-9E simultaneous native/Web mutation acceptance, UX-10 and
public serving; the ordered performance roadmap is unchanged.
~~~~

Changed paths:

~~~~text
M	.agents/docs/CURRENT_BASELINE.md
M	.agents/docs/CURRENT_HANDOFF.md
M	.agents/plans/FULL_IMPLEMENTATION_PLAN.md
M	.agents/plans/TODO.md
M	.agents/plans/cli-device-routing/TODO.md
M	.agents/plans/cli-ux/NEXT_ITERATION_CLI_UX_AND_MODEL_LIFECYCLE.md
M	.agents/plans/model-acquisition/TODO.md
M	crates/zerogravity-cli/src/control_host.rs
M	crates/zerogravity-cli/src/control_host/actions.rs
A	crates/zerogravity-cli/src/control_host/inventory.rs
M	crates/zerogravity-cli/src/control_host/work.rs
M	crates/zerogravity-cli/src/pull.rs
M	crates/zerogravity-cli/src/pull_monitor.rs
M	crates/zerogravity-cli/src/pull_native.rs
M	crates/zerogravity-cli/src/tui.rs
M	crates/zerogravity-cli/src/tui/library.rs
M	crates/zerogravity-cli/src/web.rs
M	crates/zerogravity-cli/src/web/actions.rs
M	crates/zerogravity-cli/src/web/http.rs
M	crates/zerogravity-cli/src/web/wire.rs
M	crates/zerogravity-cli/tests/fixtures/tui_shell.py
M	crates/zerogravity-cli/tests/fixtures/web_shell.py
M	crates/zerogravity-cli/web/README.md
M	crates/zerogravity-cli/web/dist/app.js
M	crates/zerogravity-cli/web/dist/style.css
M	crates/zerogravity-cli/web/src/api.js
M	crates/zerogravity-cli/web/src/api.test.js
M	crates/zerogravity-cli/web/src/browser-smoke.mjs
M	crates/zerogravity-cli/web/src/control-state.js
A	crates/zerogravity-cli/web/src/inventory-state.js
A	crates/zerogravity-cli/web/src/library.jsx
M	crates/zerogravity-cli/web/src/main.jsx
M	crates/zerogravity-cli/web/src/resident.jsx
M	crates/zerogravity-cli/web/src/styles.css
M	crates/zerogravity-cli/web/src/work-state.js
M	crates/zerogravity/src/control/downloads.rs
M	crates/zerogravity/src/control/events.rs
M	crates/zerogravity/tests/fabric_artifacts.rs
~~~~


### D.2 — UX-9D Web settings/logs

Commit: 543194ac851a8db6118eb42e3aea209403edadbc

Subject: feat(ux): share Web profile configuration and operational logs

Original landing narrative (historical instructions):

~~~~markdown
### Current checkpoint — Phase 2.1 UX-9D implementation (26 Nov)

Profiles / Config / Logs are implemented for both foreground clients; the
focused qualification is complete. Full phase acceptance remains pending.

- The original native configuration actions and typed observations now live
  in control_host/settings.rs. One ConfigFile snapshot/validation/atomic
  writer, one shared action slot and process-local active Selection. Exact
  stale writes refuse; activation only affects future actions. Native raw
  repair remains available; inactive edits invalidate both clients without
  reconfiguring admitted owners.
- Web profiles/config authenticated cold reads retain one latest exact
  snapshot with an opaque identity; reading a different document can require
  refresh. Typed safe policy fields and normalized YAML for wholly safe
  documents only. No paths, prompts, graphs, credentials or comments on the
  wire; protected fields are not writable by Web. Trusted existing profiles
  can be activated/copied without disclosure. Default, portable and active
  removal refused by original owner. Native-only unrestricted raw repair.
- Shared host-owned 128-entry operational ring in control_host/logs.rs,
  safe action/owner correlations; one event collector per host. Web/native
  view-local filtering and pause/follow; no prompts/model output or durable
  history. SSE configuration/log invalidation; bounded retries only for
  refused cold reads, never mutations. Completed Web settings receipts
  reread configuration before enabling the next edit.
- Strict minimal TUI/Web, Web-only CPU/serve/acquisition and combined
  Clippy pass; Web-only tree terminal-free; JS 3/0; actual isolated HTTP,
  native PTY plus attached Web cross-client edit/activation/logs, and real
  Firefox profile/config/log and responsive smoke pass. Fixed fixture SSE
  response closure and a real cold-read navigation/receipt refresh race.
  No live acquisition, new dependency, model or GPU/performance change.
- Full isolated gate began after strict lint; stopped intentionally at the
  turn boundary during unchanged Vulkan feature-boundary compilation,
  not a reported test failure. Do not count it as passed. The original
  UX-9C cached-model and transfer evidence remains unchanged.

**Next:** complete the private-network scripts/check.sh gate for this
checkpoint, then accept UX-9D and proceed to UX-9E simultaneous native/Web
mutation acceptance, including real admitted owners and controlled transfer.
No further configuration or runtime store; UX-10 and the ordered public
serving/performance roadmap remain unchanged.
~~~~

Changed paths:

~~~~text
M	.agents/docs/CURRENT_BASELINE.md
M	.agents/docs/CURRENT_HANDOFF.md
M	.agents/plans/FULL_IMPLEMENTATION_PLAN.md
M	.agents/plans/README.md
M	.agents/plans/TODO.md
M	.agents/plans/cli-device-routing/TODO.md
M	.agents/plans/cli-ux/NEXT_ITERATION_CLI_UX_AND_MODEL_LIFECYCLE.md
M	crates/zerogravity-cli/src/control_host.rs
M	crates/zerogravity-cli/src/control_host/actions.rs
A	crates/zerogravity-cli/src/control_host/logs.rs
A	crates/zerogravity-cli/src/control_host/settings.rs
M	crates/zerogravity-cli/src/control_host/work.rs
M	crates/zerogravity-cli/src/tui.rs
M	crates/zerogravity-cli/src/tui/logs.rs
M	crates/zerogravity-cli/src/tui/settings.rs
M	crates/zerogravity-cli/src/tui/work.rs
M	crates/zerogravity-cli/src/web.rs
M	crates/zerogravity-cli/src/web/actions.rs
M	crates/zerogravity-cli/src/web/http.rs
A	crates/zerogravity-cli/src/web/settings.rs
M	crates/zerogravity-cli/tests/fixtures/tui_shell.py
M	crates/zerogravity-cli/tests/fixtures/web_shell.py
M	crates/zerogravity-cli/web/README.md
M	crates/zerogravity-cli/web/dist/app.js
M	crates/zerogravity-cli/web/dist/style.css
M	crates/zerogravity-cli/web/src/api.js
M	crates/zerogravity-cli/web/src/api.test.js
M	crates/zerogravity-cli/web/src/browser-smoke.mjs
M	crates/zerogravity-cli/web/src/control-state.js
M	crates/zerogravity-cli/web/src/inventory-state.js
M	crates/zerogravity-cli/web/src/main.jsx
A	crates/zerogravity-cli/web/src/settings.jsx
M	crates/zerogravity-cli/web/src/styles.css
M	crates/zerogravity-cli/web/src/work-state.js
~~~~


### D.3 — UX-9E simultaneous acceptance

Commit: 1a87899013cac5afd6839f748ee33627a30e3b46

Subject: test(ux): qualify simultaneous native and Web control lifecycle

Original landing narrative (historical instructions):

~~~~markdown
### Current checkpoint — Phase 2.1 UX-9E acceptance (26 Nov)

UX-9D and UX-9E accepted. Same foreground runtime, one action slot,
original ConfigFile/cache/transfer/listener owners. No new dependencies,
model execution paths, admission changes or performance claims.

- Native Logs redraw follows the shared bounded ring revision while visible;
  a fast Web-only validation cannot disappear between active-action samples.
  Existing event cursor and per-client filters remain unchanged.
- Actual native PTY and attached Firefox share one original protected
  transfer (existing deterministic fixture, no provider access): native
  resume then safe pause after an accepted prefix, browser observes that
  checkpoint and resumes to durable publication or cancels to a preserved
  partial. Exact operation identity, stale control refusal and original
  range reopen; closing browser leaves native host alive, native quit
  joins host/worker. Same test child and original transfer owner.
- Actual approved cached Qwen 0.8B GGUF optimized lifecycle: original
  browser cache/device and resident/HTTP checks; native edit of active
  run.task.context 64 -> 128 with live admitted model/listener, stale Web
  draft refused, fresh Web restore to 64 seen in native editor; admitted
  scheduler capacity, registration, generation and exact HTTP text
  unchanged. Original 8 GiB host admission retained. Native/Web logs
  share safe success/failure correlation, no private prompts or drafts.
  Existing stale registration/generation, cache retirement, cancellation
  and joined shutdown checks pass.
- Combined release CLI unit tests 50/0/3; controlled shared Firefox/PTY
  durable and cancelled scenarios pass; real cached combined integration
  1/0/0 (54.16s, five filtered); idle native log repaint after Web
  validation passed without resize/navigation after command. Initial
  settings fixture assertion corrected for ANSI cursor-separated spaces;
  no production behavior relaxed.
- The interrupted UX-9D gate was rerun across the assembled D/E source
  (UI change before CPU compilation). HIP/Vulkan/CUDA feature boundary,
  strict CPU/optional feature checks, all 318 Rust suites 2,491/0/109,
  native/Web release, offline installer and scripts passed. No physical
  GPU or public serving qualification inferred.

**Next:** UX-10 doctor/pipeline/device presentation polish and compatibility
cleanup. Audit normal help/docs against original direct typed commands;
retain hidden compatibility aliases and remove duplicate normal paths only
after their replacements are proven. Then public serving and ordinary
benchmarks, ROCm vs optimized vLLM, CUDA, Vulkan, optional compiled providers
in the ordered full plan. No return to accepted foundations or UX owners.
~~~~

Changed paths:

~~~~text
M	.agents/docs/CURRENT_BASELINE.md
M	.agents/docs/CURRENT_HANDOFF.md
M	.agents/plans/FULL_IMPLEMENTATION_PLAN.md
M	.agents/plans/README.md
M	.agents/plans/TODO.md
M	.agents/plans/cli-device-routing/TODO.md
M	.agents/plans/cli-ux/NEXT_ITERATION_CLI_UX_AND_MODEL_LIFECYCLE.md
M	crates/zerogravity-cli/src/tui.rs
M	crates/zerogravity-cli/tests/fixtures/tui_shell.py
M	crates/zerogravity-cli/web/README.md
M	crates/zerogravity-cli/web/src/browser-smoke.mjs
~~~~


### D.4 — UX-10 core

Commit: 22c6c0a73860af4f73188d4b6093c932a33f258d

Subject: feat(cli): add cold diagnostics and typed pipeline inspection

Original landing narrative (historical instructions):

~~~~markdown
### Current checkpoint — Phase 2.1 UX-10 core (26 Nov)

Core direct CLI implementation and focused verification complete.
- Doctor uses existing cold selected-document resolution/validation
  (shared SDK inspect_policy), device inventory and cache metadata.
  Safe text/JSON status; explicit errors return 2 with report. Missing local
  sources and incompatible selected device policy warn. Cache logical bytes
  vs configured limit are not physical quota usage; no unobserved runtime
  binding is claimed available.
- Device inspect with optional source uses exact observed physical ID and
  existing run dry-run planner. No acquisition, loading, audit write or new
  reservation. Fresh positional model replaces configured entrypoint and
  model table, retaining policy; legacy explicit override remains.
- Pipeline list/inspect/check, selected profiles/config and positional
  files; original typed topology, links, declared routes/context, text/JSON.
  General pipeline run refuses. Help/completions reflect normal commands;
  compatibility aliases preserved. docs/CLI.md and examples updated.
- Final strict minimal/combined lint and isolated focused 59/0/1
  (minimal 27/0/0; optimized combined 32/0/1). Actual synthetic GGUF
  CPU execution and source forms; existing populated cache tree unchanged.
  One explicit real cached serving fixture skipped in focused run.
  Initial obsolete help expectation/document-prefixed help and conditional
  unused-import failures corrected before final rerun.

The full isolated gate passed formatting, architecture/dependency checks,
HIP-only (1m53s) and Vulkan-only (2m08s) compilation. Intentionally
stopped during production CUDA native transport rebuild to close this
checkpoint; no full gate acceptance or physical qualification claimed.

**Next:** finish UX-10 comprehensive gate and remaining bounded cold
diagnostic observations (runtime-library presence and physical cache
quota usage without introducing a second owner). Direct graph inspection
is text/JSON; no dedicated frontend graph destination yet. Then public
serving and ordinary benchmarks, then ordered ROCm/CUDA/Vulkan/compiled
provider phases. Do not reopen accepted foundation or frontend ownership.
~~~~

Changed paths:

~~~~text
M	.agents/docs/CURRENT_BASELINE.md
M	.agents/docs/CURRENT_HANDOFF.md
M	.agents/plans/FULL_IMPLEMENTATION_PLAN.md
M	.agents/plans/README.md
M	.agents/plans/TODO.md
M	.agents/plans/cli-device-routing/TODO.md
M	.agents/plans/cli-ux/NEXT_ITERATION_CLI_UX_AND_MODEL_LIFECYCLE.md
M	crates/zerogravity-cli/src/args.rs
M	crates/zerogravity-cli/src/commands.rs
A	crates/zerogravity-cli/src/doctor.rs
M	crates/zerogravity-cli/src/main.rs
M	crates/zerogravity-cli/src/output.rs
M	crates/zerogravity-cli/src/pipeline.rs
M	crates/zerogravity-cli/src/run_cli.rs
M	crates/zerogravity-cli/src/run_cli/args.rs
M	crates/zerogravity-cli/src/run_cli/devices.rs
M	crates/zerogravity-cli/src/run_cli/execute.rs
M	crates/zerogravity-cli/tests/cli.rs
M	crates/zerogravity-cli/tests/pipeline_config.rs
M	crates/zerogravity-cli/tests/run_unified.rs
M	crates/zerogravity/src/configuration.rs
A	docs/CLI.md
M	docs/README.md
M	examples/pipelines/README.md
~~~~


### D.5 — UX-10 accepted

Commit: 1b8de1cb87fa4815655cbb6700e6633946f93305

Subject: feat(cli): complete cold quota and backend diagnostics

Original landing narrative (historical instructions):

~~~~markdown
### Current checkpoint — Phase 2.1 UX-10 acceptance (26 Nov)

UX-10 and the bounded CLI/TUI/Web iteration are accepted. Cold diagnostics
now include original backend loader probes and original cache accounting.

- CUDA driver presence uses the existing cudarc loader check; HIP
  runtime/RTC uses existing CubeCL HIP loader; Vulkan uses the same ash
  entry loading as wgpu-hal. Optional uncompiled probes are unknown.
  Library loadability, observed physical binding, source/ABI/compiler
  qualification and execution admission are distinct. No explicit new
  client/context/model creation; no new dependency.
- CacheMetadata::observe_usage obtains only existing publication/attempt
  locks with zero wait and read-only descriptors, reuses original
  cooperative inode-charge/live reservation arithmetic and persistent
  policy, and never creates lock files, directories or reservations.
  Busy publication is unknown, not zero usage. Charged + reserved vs
  configured quota are not filesystem free bytes; logical manifest totals
  can share blobs. Existing cache tree remains unchanged.
- Strict minimal, cache, combined lint and focused 44/0/2 (25 cache
  transaction, 14 CLI and five optimized real synthetic GGUF/source-form
  tests). Equality with original live/dead reservation accounting,
  contended publication and missing-lock refusal pass.
- Complete isolated gate sequence passed 2,505/0/109 (318 Rust suites), release Web/TUI/installer/scripts. Original run stopped at two transient busy-observation fixture assertions in acquisition/cache transactions; test-only bounded busy retry, reran affected cache-only/diagnostic/acquisition suites and exact remaining gate commands. No production change after the initial run.
  The parallel fixture can observe a transient nonblocking refusal
  (e.g. inherited lock description or interrupted probe); precise cause
  not established. Test waits only for observation; explicit held-owner
  refusal stays unchanged. No live acquisition, new dependency, physical
  GPU execution qualification or performance claim.

**Next:** public OpenAI-compatible serving, starting with
[the serving iteration](../plans/serving-and-speculation/NEXT_ITERATION_OPENAI_COMPATIBLE_SERVING.md)
and its live owner TODO, then ordinary public benchmarks, ROCm vs
optimized vLLM, CUDA, Vulkan, optional compiled providers per full plan.
Keep the accepted CLI/config/runtime/control owners. Graph inspection
remains direct text/JSON; general graph/media execution and advanced
provider transfer modes remain explicitly unsupported or deferred.
~~~~

Changed paths:

~~~~text
M	.agents/docs/CURRENT_BASELINE.md
M	.agents/docs/CURRENT_HANDOFF.md
M	.agents/plans/FULL_IMPLEMENTATION_PLAN.md
M	.agents/plans/README.md
M	.agents/plans/TODO.md
M	.agents/plans/cli-device-routing/TODO.md
M	.agents/plans/cli-ux/NEXT_ITERATION_CLI_UX_AND_MODEL_LIFECYCLE.md
M	crates/zerogravity-artifacts/src/cache/paths.rs
M	crates/zerogravity-artifacts/src/cache/store.rs
M	crates/zerogravity-artifacts/src/cache/store/metadata.rs
M	crates/zerogravity-artifacts/tests/cache_transactions.rs
M	crates/zerogravity-cli/src/doctor.rs
M	crates/zerogravity-cli/tests/cli.rs
M	crates/zerogravity-cli/tests/run_unified.rs
M	crates/zerogravity-gpu/src/lib.rs
M	crates/zerogravity-gpu/vendor/cubecl-cuda/src/lib.rs
M	crates/zerogravity-gpu/vendor/cubecl-cuda/src/observation.rs
M	crates/zerogravity-gpu/vendor/cubecl-wgpu/src/lib.rs
M	crates/zerogravity/src/devices/backend.rs
M	docs/CLI.md
~~~~


### D.6 — UX-P design decision

Commit: 3b584f729efc691b8149805785d5c00f81d4c994

Subject: docs(ux): plan native TUI interaction and visual polish

Original landing narrative (historical instructions):

~~~~markdown
### Current next priority — UX-P native TUI polish (26 Nov)

User decision: before public serving, complete mouse support throughout
the TUI (click pane/row/tab/action/setting, drag splitters, independent
scroll), elegant clean colors and Dark/Light/Auto theme selection.
[Full UX-P specification](../plans/cli-ux/TUI_POLISH_AND_MOUSE_INTERACTION.md)
defines current-source audit, inspected Whisrs v0.2.1 design, semantic
palette, appearance/Auto and terminal depth policy, responsive geometry,
render-derived hit targets, stable typed identities, all nine views and
ordered UX-P1–4 checkpoints. Design-only checkpoint; no implementation or
UX-P acceptance claimed. Existing UX-1–10 remains accepted.

**Next:** read the UX-P specification and live CLI/device TODO, implement
UX-P1 then UX-P2–4 with one coherent validated checkpoint per turn.
No new runtime, cache, configuration or execution authority; mouse
selection is inert, explicit actions use original owners. Keep both
keyboard and mouse pathways. Optional local Auto background probe
dependency requires separate approval/offline review; no new dependency
or download at design checkpoint. Resume public serving only after UX-P
acceptance.
~~~~

Original UX-P design-only specification at this commit (historical, complete; later toolbar-free/minimalism decisions in live owning spec supersede applicable sections):

~~~~markdown
# ZeroGravity — native TUI interaction and visual polish

Author: Project owner
Co-authors: GPT-6.2
Owner: @pvl
Decision: 26 Nov 2026 — polish before public serving
Status: planned; no implementation or acceptance claimed

This is the follow-on to the accepted UX-1–10 iteration at `1b8de1cb`.
It supersedes only the relevant interaction/presentation details of
[NEXT_ITERATION_CLI_UX_AND_MODEL_LIFECYCLE.md](NEXT_ITERATION_CLI_UX_AND_MODEL_LIFECYCLE.md)
§46A.6–12 and §46A.29. Keep the [full plan](../FULL_IMPLEMENTATION_PLAN.md),
[CLI/device owner TODO](../cli-device-routing/TODO.md), [current handoff](../../docs/CURRENT_HANDOFF.md)
and accepted SDK/control/runtime/config contracts. This is a bounded
product polish phase (UX-P), not a redesign of the execution fabric.
Complete it before §2.3 public serving/ordinary benchmarks.

## 1. Intent

A quiet, beautiful instrument panel that feels like a native application,
not an assortment of terminal printouts. Elegant because the hierarchy is
obvious, not because everything is decorated. Comfortable at 80x24 and
useful across a wide terminal. Mouse-first discoverability and equally
complete keyboard operation.

- Click any visible navigable destination, row, tab, field, button, picker,
  confirmation, scrollable pane and resize handle. No action requires
  remembering a shortcut or navigating by an unrelated key.
- Resize and maximize/restore meaningful panes; scroll each inner pane
  independently. Preserve selection, search, scroll anchor and preferred
  split through resize, navigation, refresh and temporary collapse.
- Use one clean semantic color vocabulary, with restrained accents,
  readable typography/cell spacing and clear focus. Dark, Light, Auto.
- Keep exact source/registration/generation/operation/document identity
  and truthful current-state labels; never hide a refusal behind styling.
- Reuse original effects, read workers, confirmation flows, snapshots
  and typed configuration writes. No new runtime, cache, scheduler,
  provider, execution or persistent presentation owner.

## 2. Inspected baseline and reference

### Current ZeroGravity

Source inspected at the accepted checkpoint:

- `crates/zerogravity-cli/src/tui.rs`: dirty/event-driven foreground loop
  handles resize, paste and keys; no mouse event dispatch.
- `tui/terminal.rs`: scoped raw mode, alternate screen and bracketed paste;
  no mouse capture. Restore on ordinary errors, quit and unwind.
- `tui/view.rs`: two fixed rail widths, fixed 50/50 library splits,
  single per-view scroll counter, header/footer full of key hints,
  independent keyboard modal handling. Cyan bold and DarkGray are the
  primary semantic styles. Library list offsets derive from selection;
  detail scrolling counts unwrapped source lines rather than visual rows.
- `tui/library.rs`: stable typed row keys and original bounded observation
  worker already exist. `tui/logs.rs` observes the bounded host ring.
- `control_host/{settings,work,actions}.rs`: exact ConfigFile snapshot,
  typed mutation, original shared action slot and runtime lifetimes.
- `pipelines/document.rs` UiDoc and configuration introspection:
  color/progress/interactive settings exist; no theme.
- Web `src/styles.css`: already uses slate/teal dark and light palettes;
  `main.jsx` has a local binary light/dark toggle, initially dark.

Do not rebuild the UI from scratch or create another widget/runtime framework.

### Whisrs (actual source, not the desktop overlay mistaken for its TUI)

The local Nix configuration pins Whisrs v0.2.1,
`5f5be26e8886e575caa1464133e961b449561fa3`. Reviewed the corresponding
[TUI theme](https://github.com/y0sif/whisrs/blob/5f5be26e8886e575caa1464133e961b449561fa3/src/cli/tui/theme.rs),
[palette](https://github.com/y0sif/whisrs/blob/5f5be26e8886e575caa1464133e961b449561fa3/src/overlay/palette.rs),
[Home layout](https://github.com/y0sif/whisrs/blob/5f5be26e8886e575caa1464133e961b449561fa3/src/cli/tui/screens/home.rs)
and [manifest](https://github.com/y0sif/whisrs/blob/5f5be26e8886e575caa1464133e961b449561fa3/Cargo.toml).

Borrow the visual discipline, not its application architecture:
Carbon's soft white, charcoal/gray and teal success; Ember's warm amber
and Cyan's blue accent are examples of the same semantic roles. Whisrs
uses terminal-native backgrounds, two levels of muted text/rules,
light-aware roles, truecolor/ANSI16/NO_COLOR fallback and a bounded
terminal background probe outside its input loop. Its narrow home stacks
information instead of overcrowding columns. ZeroGravity already has
a compatible teal/slate Web identity, so adopt one original coordinated
dark/light palette instead of copying those named themes. No Whisrs
code, theme dependency, agent bridge or extra frontend framework needed.

## 3. Visual system

### 3.1 Palette

Proposed baseline tokens (hex RGB; tune by inspecting real terminal cells
during UX-P1, not by adding a theme engine). Colors express meaning rather
than object type; a GPU vendor is not a color.

| Semantic role | Dark | Light | Use |
| --- | --- | --- | --- |
| Canvas | #111318 | #F5F6F8 | Whole shell |
| Surface | #171A20 | #FFFFFF | Main pane, navigation |
| Raised | #1E232B | #EEF1F5 | Dialogs, input/action chrome |
| Hover | #202731 | #E9EEF2 | Pointer affordance, optional |
| Selection | #233735 | #DCEFEA | Selected resource/navigation |
| Text | #E8EBF0 | #26313C | Body/value/output |
| Secondary | #ADB6C3 | #4F5D6D | Labels, supporting metadata |
| Muted | #929EAD | #596879 | Help/idle text, still readable |
| Rule | #343D48 | #D7DEE6 | Noninteractive hairline |
| Handle | #7C8998 | #788797 | Divider, scrollbar, affordance |
| Accent | #80D1C5 | #116B63 | Focus, active destination, primary action |
| Success | #83CFA8 | #206C4B | Ready, durable, saved |
| Warning | #E5BC79 | #8A560E | Paused, caution, incomplete |
| Danger | #F1979C | #A33743 | Failed, destructive confirmation |
| Info | #9DB8E6 | #315E99 | In-progress/status information |
| On accent | #111D1D | #FFFFFF | Primary button text |

Use solid background only on small semantic regions: one active selection,
focused input, primary action, modal. No gradients, rainbow badges,
bright borders on every panel, full-page cyan, excessive iconography or
fake dashboard cards. Whitespace and alignment are the main hierarchy.
Use compact 1-cell separators, one-cell padding where geometry allows,
aligned label/value columns and consistent units. Soft neutral titles;
accent only active/focused elements. One clear primary action per view,
ordinary secondary buttons are low-key, destructive action styled only
when relevant. No flashing selection, animated decorative chrome or
attention-grabbing progress when idle.

Provide a small copyable palette value and role-to-Style functions in
`tui/theme.rs`, not per-widget RGB or a configurable theme framework.
Only Dark and Light palettes; Auto is a selector, not a third palette.
Use the same tokens/vocabulary in Web CSS (small parity pass; no wholesale
Web redesign). Primary readable foregrounds against painted surfaces
target WCAG AA 4.5:1; actionable dividers/focus indicators 3:1. Passive
decorative hairlines need not be prominent. Initial RGB token pairs were
checked for contrast; verify actual state combinations, ANSI fallback
and readability in real terminal renders.

### 3.2 Appearance settings and terminal depth

Add optional `ui.theme: auto|dark|light` to the existing strict UiDoc,
default auto if absent. A separate typed ThemeMode enum, not UiSwitch
(always/never). Preserve old documents. Suggested `ui.mouse` reuses
UiSwitch (auto default; never disables capture; always requests capture
only on a capable terminal). No extra YAML schema, persistent UI database,
theme directory, downloads or user palettes now. Example:

~~~yaml
ui:
  theme: auto
  color: auto
  mouse: auto
~~~

Show Theme as a three-choice picker in Config → Appearance and a compact
header control. Click or keys preview; Apply/Cancel explicit, active target
shown. Preview is client-local and reversible. Apply uses the original
exact observed ConfigFile/typed action slot, validates and saves once.
No implicit document creation on launch/preview. Missing default document
can be created only by an explicit authorized save. Inactive profile
changes affect only that profile; active document changes repaint the
native UI and attached Web through configuration invalidation, without
reloading models or rebinding the listener. Failed or stale saves retain
the draft and allow reload/retry; never silently retry writes.
`zg config set ui.theme dark` is the CLI path. No extra global flag needed
for v1. Extend existing safe Web settings schema/typed picker for theme;
replace binary browser toggle with Auto/Dark/Light; browser Auto follows
prefers-color-scheme, including changes. Do not allow appearance settings
to bypass or alter existing CLI color/progress/interactive policy.

Native Auto resolves terminal appearance once at session start:
1. Known local terminal background (safe bounded OSC 10/11 probe before
   raw mode/input reader, never while keys are being consumed).
2. Recognized COLORFGBG background hint.
3. Dark fallback when unknown, stated in appearance help.
Explicit Dark/Light skips detection. Cache the observation for the session;
switching to Auto uses it or the hint/fallback. No ongoing terminal query,
no background process. Skip active probe on SSH/multiplexer or incapable
terminal, where delayed replies can become input. The Whisrs approach
uses terminal-colorsaurus; consider that small audited TUI-only dependency
for reliable local detection, subject to existing dependency/offline
authorization. It was not found in the accessible local Cargo source cache.
Do not download/install it during this planning work or hand-roll a risky
blocking OSC parser. If not approved/available, implement the conservative
environment fallback and document/track the local query limitation
honestly; do not claim true automatic detection on all terminals.

Explicit Dark/Light and resolved Auto use the selected full canvas in
truecolor for predictable contrast. This is an intentional difference
from Whisrs' transparent terminal-native Carbon. Truecolor when supported,
256-color mapping if available, role-specific ANSI16 fallback otherwise;
no assumption that user-defined ANSI colors have exact RGB values.
Unknown terminal capabilities stay conservative. `--color never` and
NO_COLOR (respect existing explicit --color always precedence) suppress
RGB/background color; semantic markers, bold/reverse and text carry the
same information. TERM=dumb and non-TTY direct output are unchanged.
Do not change emulator palette via OSC or set global terminal colors.
Theme choice persists even when color is disabled.

### 3.3 Semantic status

Always pair hue with text/shape: Ready ✓, Listening ●, Loading/Running ◐,
Paused ‖, Pending …, Stopped ○, Failed !, Cancelled ×, Unavailable —.
ASCII-compatible markers where glyph width/terminal support is uncertain.
Use actual typed status, not inferred success from a progress percentage
or absence of events. Distinguish selected item, keyboard-focused pane,
active destination, active profile and admitted owner. Disabled actions
are dimmed but labeled, with a reason (not compiled, offline, busy,
stale, not applicable). Do not make disabled areas clickable. Unknown
memory/total/progress remains Unknown/—, never 0. User/model output remains
body text; do not interpret it as terminal escape sequences.

## 4. Layout

Keep nine destinations and their existing key numbers. Group rail labels
only when height permits: Workspace (Home, Running, Serve), Resources
(Models, Downloads, Devices), Settings (Profiles, Config, Logs).
Header: ZeroGravity / current view; active profile, compact appearance
control and connection/owner status if space. Rail has one active marker.
Workspace has one title and context actions, not a title inside a
title. Footer: current operation/notice (higher priority than help),
then compact context hints and clickable Help/Quit. No redundant
navigation instructions in every content pane.

Representative wide view (not literal exact cell artwork):

~~~text
 ZeroGravity / Models                    laptop •     Theme: Auto ▾
───────────────────────────────────────────────────────────────────────
 WORKSPACE    │ Models                [ Cached ] Catalog  Partial
   Home       │ Search models…                     [Pull model]
   Running    │
   Serve      │ NAME                  SIZE   STATUS │ Qwen3.5-0.8B
 RESOURCES    │ ▎ Qwen3.5-0.8B        0.5 G  Cached │ GGUF · snapshot
 ▎ Models     │   Qwen3.5-2B          1.4 G  Cached │ Source / digest
   Downloads  │   …                                 │ Files / bytes
   Devices    │                                     │ Verification
 SETTINGS     │                                     │
   Profiles   │                                     │ [Load] [Run]
   Config     │                                     │ [Verify] […]
   Logs       │                                     │
───────────────────────────────────────────────────────────────────────
 Ready · 2 cached models             Tab Focus  / Search  : Go to  ? Help
~~~

The rail and list/detail boundary are draggable. Header maximize/restore
and collapse/expand controls appear on meaningful panes, not on every
decorative surface. Keep minimum sizes and exactly one title owner.
Proposed defaults, measured against the actual terminal Rect and content
minima rather than hard width branches scattered in views:

| Terminal / workspace | Default |
| --- | --- |
| <20x7 usable minimum | Resize hint + quit, never panic or invoke hidden controls |
| Narrow (<70 cols) | Workspace only, clickable nav overlay, detail drill-in with Back |
| 70–99 cols | Compact rail + primary; stack list/detail if enough height, else detail drill-in |
| 100–139 cols | Rail + primary and contextual split when both fit; otherwise vertical/collapsed |
| >=140 cols | Rail + list/workspace + detail where useful |
| Short heights | Collapse chrome/action overflow, then secondary pane; no lost state |

Initial rail 22 cols (compact ~18, clamp min 15/max 32); list/details
default ~55/45 with list min 32, detail min 28 cells; vertical split
min ~6 rows per pane plus headers. These are starting design constants;
fit actual content and terminal geometry. Never squeeze below minima.
User drag adjusts preferred rail width or split ratio in current
orientation. Terminal resize clamps actual geometry, not preferred ratio;
when space returns restore it. Separate remembered preferences per
layout family/orientation, not per-resource. Maximize one pane and
restore exact prior preferred geometry. Collapse and reopen preserves
selection, filters and anchor. Pane layout/focus/scroll stays session
local, no persistence file in this phase.

## 5. Unified mouse and keyboard interaction

### 5.1 Event routing and geometry

Use existing Crossterm 0.29 MouseEvent (Down, Up, Drag, Moved, ScrollUp,
ScrollDown, ScrollLeft/Right where emitted) and scoped EnableMouseCapture /
DisableMouseCapture in terminal::Session. Restore terminal raw/alternate
screen/paste/mouse/cursor on normal exit, error, unwind, signals and any
supported suspend path. Unsupported events harmless; no global mouse
capture outside foreground TUI. Disable capture when requested by setting.
Terminal selection/copy remains terminal-emulator owned (typically
Shift+drag bypass); document variance, keyboard way to disable capture,
no automatic OSC clipboard writes or unrelated command execution.

One layout pass yields pane Rects, visible viewports, splitter tracks and
visible hit targets derived from actual rendered cells. A compact
`HitTarget { rect, target }` collection in paint order, with typed semantic
targets for pane, nav, row key, field, action, modal button, tab, scrollbar
or splitter. Reuse exact existing typed identity and existing action
handlers. Do not build a second geometry model, per-cell map, widget tree,
runtime store, or string-command dispatch. Modal layer exclusively owns
its hit regions; nothing clicks through.

Press records one target and drag state. Ordinary effect buttons activate
on release only if the same enabled semantic target is still visible and
valid; cancel on different target, resize, modal/route transition or
invalidated owner. Revalidate original owner/request identity on dispatch
even if a new snapshot appeared. A drag never activates a button; release
clears drag. Mouse move only affects hover/drag presentation, never starts
read workers, mutations or model tasks. Ignore right/middle by default;
optional context menu only if it reuses the visible action menu and offers
no unique or hidden effect. Avoid essential double-click semantics: optional
double-click may drill into read-only detail/open an editor, never confirm,
load, pull, start, remove or submit. Single click and visible actions are
sufficient. Keyboard equivalent for all controls.

### 5.2 Focus, selection and scrolling

- Click rail item navigates immediately (inert view change), click pane
  focuses it, click visible row selects exact Key/Owner and updates detail;
  click a non-action row never executes. Tab/Shift-Tab cycles logical
  panes/controls, arrows/j/k select/scroll where appropriate; maintain
  old shortcuts except in actual input fields. Focus indicator and
  selection marker differ; Enter invokes a safe drill-in or the focused
  explicit button.
- Separate list, detail, output, modal, editor and log viewport offsets.
  Wheel routes to the innermost scrollable pane under pointer, not the
  current keyboard focus, and does not change selected identity. Keyboard
  moves only the focused viewport; selection movement reveals selected
  row if requested. Page/Home/End and scrollbars reach the full content.
  Draggable thumb/track click, clamped position and visible-range math.
  Scroll only in actual visible area; no scrolling the parent when a
  child is active unless explicitly targeting it.
- Compute scroll bounds from real visible/wrapped visual rows with cell
  width (Unicode wide/combining characters, graphemes where appropriate),
  not bytes, code points or unwrapped line count. Reuse Ratatui primitives
  or existing locked Unicode utilities; no editor framework. Maintain
  stable item/log sequence or output-line anchor on refresh/resize; clamp
  when data disappears. Distinguish selection-reveal from manual browsing.
  Logs/output follow only at tail; manual scroll away pauses auto-follow,
  visible Follow button returns to tail. Bounded ring drops show gap;
  never invent history.
- Drag splitter on its visible track with pointer captured through the drag,
  clamp against minima; horizontal and vertical as applicable. Release or
  modal/resize cancels drag safely. Keyboard focus splitter + arrows
  resizes, Enter/double-click splitter may reset to default. Maximize and
  restore controls have keyboard equivalents. Pointer does not reorder
  panes or alter runtime resources.
- For text inputs and raw YAML: click to place caret in visible cell,
  drag selection within editor, keyboard edits apply to selection,
  scrolling independent of underlying view. Keep valid UTF-8/grapheme
  cursor boundaries and existing bounded paste/input semantics. No
  click outside to submit; focus changes do not implicitly save. Prompt
  has explicit Send/Run and chat/raw picker, raw editor Save/Cancel,
  existing Ctrl-S/Ctrl-J/Enter semantics where applicable. Preserve
  drafts across resize and modal-local scrolling; discard only explicit
  Cancel/Esc (confirm unsaved changes where needed). No broad text-editor
  rewrite, syntax-highlighter dependency or clipboard bridge.
- Modal forms/pickers have visible buttons, clear default action, click
  rows and scroll. Confirmations display exact target and effect;
  destructive/default action is Cancel; Apply/Confirm is explicit.
  Backdrop clicks never confirm or silently discard. Escape and keyboard
  confirmation retain original behavior. Palette one instance; clickable
  matching destinations and Appearance/Help shortcuts, not a second
  command interpreter.

### 5.3 Responsive action bar

Every action currently exposed solely by key hints gets a visible labeled
button, tab, menu item or picker. Wide bars show primary + most useful
secondary actions; narrow views put the rest behind one `More…` button
with full mouse/keyboard access. Click hitbox includes padding and label,
not just a glyph. Context hints (shortcut next to action) replace repeated
long key lists. Keep action visibility and enabled predicate from the
original typed owner/status/selection, not a parallel policy table.
Unavailable optional feature has clear state and no fake enabled button.

## 6. View-by-view design

| View | Main layout and mouse actions | Ownership boundary |
| --- | --- | --- |
| Home | Compact live overview: resident models, API listener, active transfer, profile. Quick Run, Load, Open Models, Start/Manage Serve, Open Web, refresh/help. Click status summary navigates to related view. Empty state explains next action. | No auto-load/serve/pull; status is current snapshot. |
| Running | Registration list + selected owner details and task response. Click selects exact registration; Load, Invoke, Unload/Retire, Cancel active action, chat/raw selector, prompt Send, copyable identity. Independent response scrolling/follow. | Selected owner is not a new registration. Unload confirmation and cancellation use exact original IDs. No persistence of prompts/output or new history. |
| Serve | Listener status/address, registered routes, counters, selected policy; explicit Start (form), Stop (exact generation), Open Web, Manage Models. Show disabled reasons and terminal join status. | No implicit listener on selection; no server config change applies to an admitted listener. Don't expose inference key. |
| Models | Click Cached/Catalog/Partial tabs, local search/filter, exact row selection, details with source/digest/verification/bytes. Load, Run, Load for Serve, Pull, Verify, Alias, Reclaim preview/Remove via action bar. | Static catalog not remote search/readiness; no implicit acquisition. Partial controls only if original owner supports them. Remove exact digest, preview and confirm. |
| Downloads | Operation and phase, observed vs durable/checkpoint bytes, known-total progress (indeterminate otherwise), file counts/status. Pause/Resume/Cancel buttons gated by exact live operation; transfer remains visible while browsing other pages. | Preserve original safe-pause vs request distinction. No synthetic progress, no duplicate transfer scheduler. |
| Devices | Observed physical rows, vendor/class and memory, stable selector and backend metadata/exclusion; click details, Check model policy and Load on selected device via form. | Only observed hardware names/data. Cold fit is not admission. Original device/source planner unchanged. |
| Profiles | Active marker separate from selected row; click selects, Open editor, Create, Copy, Activate, Remove (confirm). Show defaults and relative path context. | Open inert; activation affects future operations only. Original exact document/path guards. |
| Config | Grouped searchable rows, current value + provenance; detail with description, constraints and active/editing target. Click value or Edit opens typed picker/editor. Enum choices, bool toggle draft, bounded numeric/text input; explicit Apply/Unset/Validate, Copy/Activate/Remove, Raw View/Raw Edit. Appearance group Theme/Color/Mouse. | Exact original snapshot and typed validation/atomic save. Unsupported/complex fields keep raw repair path. No silent policy mutation or second schema. |
| Logs | Search, level/component pickers, Follow/Pause, scrollable operational entries, dropped count. Click row reveals safe existing correlation/details. | Host-owned bounded ring only; no model output, raw prompt, secrets or durable log store. |

All views support focus, scroll, refresh, narrow navigation and contextual
empty/loading/error states. Keep progress visible across view changes in
the footer, but not an extra persistent status dashboard. Use clear
action verbs and consistent units (B/KiB/MiB/GiB; never pretend byte or
time estimates). Long exact identities/URLs remain available through
detail, wrap/scroll; not truncated into ambiguous actions. Browser URL
modal remains private; close it without stopping host.

## 7. Implementation boundaries (simple Rust)

Preserve `App::handle`, existing Navigation/Form/Confirmation, Library
stable keys, shared control_host Actions/Selection/Document, work/read
workers and lifecycle. Add only minimal presentation modules:

- `tui/theme.rs`: semantic palette, depth/effective theme and style helpers.
- `tui/layout.rs`: pure responsive geometry and preferred split/focus/viewport state.
- `tui/input.rs` (or small existing navigation extension): mouse capture state,
  typed target dispatch into the same existing Action path; keyboard
  and mouse converge on view intent. No dynamic dispatcher/event bus.
- Shared row/button/input/scroll helper functions in `tui/widgets.rs` if
  useful. Split oversized view.rs by views only when it simplifies
  ownership; no generic declarative UI framework or TUI trait hierarchy.

Render stores only visible targets and small state per pane. No unbounded
layout/hit histories, polling thread, per-mouse-move inventory refresh or
full snapshot cloning. Dirty redraw only on actual presentation/event
change; coalesce drag/motion when useful without losing release/press
ordering. Prefer zero-cost typed enums/Rects, direct rendering and
existing bounded observations. No unsafe code or new dependencies except
separately approved terminal background helper if needed; no renderer
dependencies in SDK or CLI-only build. Do not optimize non-hot code at the
expense of clarity. Device/cache/serving owners remain original.

Shared config changes go through UiDoc, introspection, safe Web typed
whitelist and original Settings request; no new persistence path. Browser
scope is visual token/theme parity only; keep existing responsive
functional Web acceptance. No new network API/serving protocol work until
UX-P exit. Reuse actual accepted runtime tests, no massive evidence
framework, new pinned artifacts or test snapshots.

## 8. Ordered implementation checkpoints

One coherent signed, validated checkpoint per turn (~30 min upper
target); prefer practical slices, not a multitude of tiny commits. Do
not mark an untested or interrupted phase accepted.

- [ ] **UX-P1 — visual foundation and appearance:** semantic roles,
  restrained chrome, Dark/Light/Auto, config/typed appearance selector,
  depth/NO_COLOR and safe Auto fallback, minimal Web token/theme parity.
  Actual rendered preview in each supported mode; agree final tokens
  against real terminal view. Preserve all original keyboard functions.
- [ ] **UX-P2 — input/layout foundation:** scoped mouse capture/restoration,
  one render-derived hit layout, pane focus, rail navigation, drags,
  maximize/restore, responsive collapse and independent scroll primitives.
  Actual PTY click/nav/resize/scroll with exact geometry; no effectful
  click path until identity/confirmation mapping is proven.
- [ ] **UX-P3 — full view controls:** action bars/overflow and mouse row/tab/
  filter selection for all nine views; pickers/modal buttons and typed
  settings/editor mouse interaction. Shared original actions and guards,
  no keyboard-only remaining visible operation. Verify controlled actual
  cache/download and configuration mutation plus explicit confirmations.
- [ ] **UX-P4 — finish ergonomics and acceptance:** Unicode wrapping/caret,
  scrollbar/follow/selection edge cases, small viewports, accessibility,
  focus order, no-color/ANSI, busy/disabled/stale/error/empty states,
  cross-client theme revision, browser retirement and terminal cleanup.
  Real cached resident inference/serve/download/config simultaneous
  regression through mouse and keyboard, full offline isolated gate,
  updated CLI guide/help and owner handoff. No physical GPU perf claim.

Use these as order, not four compulsory isolated branches. Preserve
working keyboard controls during every slice. Continue from the live
source and TODO rather than reopen accepted foundations.

## 9. Acceptance (minimal, high-value)

- Actual wide/normal/narrow terminal renders (e.g. 150x40, 110x30, 80x24,
  50x16, 30x8) reviewed for legibility, no duplicated headings or clipped
  essential controls. Dark/Light/Auto, truecolor and limited/no-color.
  Theme switching and persisted reopen of active/inactive profile; original
  config bytes not changed by preview/cancel or stale refused mutation.
- Focused pure geometry/navigation tests for bounds, hit translation from
  rendered viewport, modal precedence, press/release/drag, stable identity,
  wide chars and wrapped scroll. Actual Crossterm SGR mouse sequences via
  existing PTY fixture, not just direct reducer calls: nav click, row
  select after list scroll, wheel over two panes without changing focus,
  splitter drag both orientations, maximize/restore and resize. Click
  settings picker then save, cancel modal and stale release. Keyboard
  parity and correct teardown of capture on quit/error/signal.
- Actual source-bound cached Qwen test and original controlled transfer
  fixture: explicit mouse Run/Load/Unload/Serve Start/Stop and pause/resume/
  cancel, exact target identity and stale refusal. No selection/scroll/
  resize/theme effect on admitted registration, generation, operation or
  resource ownership; browser close independent, host quit joins. No
  provider download or fresh physical GPU qualification necessary.
- Existing shared config and attached Web tests, terminal-free Web
  feature isolation, CLI-only/no-default features, strict lint, normal
  full workspace/release gate. Check generated assets only if CSS/JS
  changed. Keep evidence concise (one screenshot/transcript per useful
  layout/theme and actual action results), no golden per-cell test matrix.
  Note limitations (e.g. unknown terminal auto probe or emulator capture
  bypass), do not claim full mouse/auto support where unverified.

Exit: all normal foreground TUI controls are mouse-accessible and
keyboard-accessible; resizable panes and scroll behave coherently;
the UI is polished in Dark/Light/Auto and graceful in restricted
terminals; existing runtime and configuration semantics unchanged.
Public serving resumes only after this bounded polish acceptance.
Additional named/custom themes, persistent layout history, dedicated
pipeline graph visualization, generalized terminal clipboard or remote
agent widget API are deliberately deferred.
~~~~

Changed paths:

~~~~text
M	.agents/docs/CURRENT_HANDOFF.md
M	.agents/plans/FULL_IMPLEMENTATION_PLAN.md
M	.agents/plans/README.md
M	.agents/plans/TODO.md
M	.agents/plans/cli-device-routing/TODO.md
M	.agents/plans/cli-ux/NEXT_ITERATION_CLI_UX_AND_MODEL_LIFECYCLE.md
A	.agents/plans/cli-ux/TUI_POLISH_AND_MOUSE_INTERACTION.md
M	.agents/plans/serving-and-speculation/NEXT_ITERATION_OPENAI_COMPATIBLE_SERVING.md
~~~~


### D.7 — UX-P1 appearance

Commit: b1073e7c53e5fbeef1c9f346264e7669e02e93e2

Subject: feat(ux): establish semantic native and Web appearance

Original landing narrative (historical instructions):

~~~~markdown
### Current checkpoint — UX-P1 visual foundation (26 Nov)

Typed ui.theme Auto/Dark/Light, native T shortcut → Config appearance
picker (preview/explicit apply/cancel), semantic color roles, restrained
pane rules and header, 24-bit/256/16/no-color, coordinated Web tokens and
active theme observation. No execution/control owner or dependency change.
Web Auto follows matchMedia, native Auto captures COLORFGBG else dark;
active OSC probe deferred. Focused Rust 81/0/3; minimal/Web-only/combined
strict lint, JS 3/0, actual native PTY appearance/settings/shell and Firefox
settings, mobile/tablet/desktop passed. Stale native draft vs newer Web
write refused, inactive profile isolation and persisted reopen proven.
Initial browser header overflow fixed with compact mobile glyph; final
PTY fixture waits for actual selected row instead of stale repaint.
Full gate and real resident/transfer regression at UX-P4.

**Next:** UX-P2 scoped mouse capture/restore, render-derived hit regions,
pane focus, resize/maximize, responsive layout and independent scrolling.
Keep all existing typed actions and owner identities. UI-specific mouse
preference and full actionable controls follow the UX-P plan.

### Original UX-P priority decision — native TUI polish (26 Nov)

User decision: before public serving, complete mouse support throughout
the TUI (click pane/row/tab/action/setting, drag splitters, independent
scroll), elegant clean colors and Dark/Light/Auto theme selection.
[Full UX-P specification](../plans/cli-ux/TUI_POLISH_AND_MOUSE_INTERACTION.md)
defines current-source audit, inspected Whisrs v0.2.1 design, semantic
palette, appearance/Auto and terminal depth policy, responsive geometry,
render-derived hit targets, stable typed identities, all nine views and
ordered UX-P1–4 checkpoints. UX-P1 is focused-qualified above; overall
UX-P acceptance pending. Existing UX-1–10 remains accepted.

Read the UX-P specification and live CLI/device TODO, continue
UX-P2–4 with one coherent validated checkpoint per turn.
No new runtime, cache, configuration or execution authority; mouse
selection is inert, explicit actions use original owners. Keep both
keyboard and mouse pathways. Optional local Auto background probe
dependency requires separate approval/offline review; no new dependency
or download at design checkpoint. Resume public serving only after UX-P
acceptance.
~~~~

Changed paths:

~~~~text
M	.agents/docs/CURRENT_HANDOFF.md
M	.agents/plans/FULL_IMPLEMENTATION_PLAN.md
M	.agents/plans/README.md
M	.agents/plans/TODO.md
M	.agents/plans/cli-device-routing/TODO.md
M	.agents/plans/cli-ux/TUI_POLISH_AND_MOUSE_INTERACTION.md
M	crates/zerogravity-cli/src/control_host.rs
M	crates/zerogravity-cli/src/control_host/settings.rs
M	crates/zerogravity-cli/src/tui.rs
A	crates/zerogravity-cli/src/tui/theme.rs
M	crates/zerogravity-cli/src/tui/view.rs
M	crates/zerogravity-cli/src/web/settings.rs
M	crates/zerogravity-cli/src/web/wire.rs
M	crates/zerogravity-cli/tests/config_cli.rs
M	crates/zerogravity-cli/tests/fixtures/tui_shell.py
M	crates/zerogravity-cli/web/dist/app.js
M	crates/zerogravity-cli/web/dist/style.css
M	crates/zerogravity-cli/web/src/browser-smoke.mjs
M	crates/zerogravity-cli/web/src/main.jsx
M	crates/zerogravity-cli/web/src/settings.jsx
M	crates/zerogravity-cli/web/src/styles.css
M	crates/zerogravity/src/configuration.rs
M	crates/zerogravity/src/pipelines.rs
M	crates/zerogravity/src/pipelines/document.rs
M	docs/CLI.md
~~~~


### D.8 — UX-P2 mouse/panes

Commit: ab3cf974e4df7538384b873af121a8a3747f9334

Subject: feat(ux): add native mouse navigation and resizable panes

Original landing narrative (historical instructions):

~~~~markdown
### Current checkpoint — UX-P2 mouse and pane foundation (26 Nov)

- Scoped Crossterm capture (including quit/error/signal restore), typed
  ui.mouse auto/always/never; F8 session-only override.
- Presentation-only layout/input modules; one set of visible render
  hit regions, release match and original exact Row Key/ready Owner
  revalidation. Click rail/rows/shelf, theme preview, palette/help and
  pane focus. Modal masking; no effectful mouse confirmation/submit yet.
- Preferred rail width and both orientation ratios, per-view focus,
  maximize/restore, narrow nav and detail drill-in, separate visual-row
  scroll offsets and draggable scrollbar. Running/response secondary
  viewport; log wheel off tail pauses original ring view. Existing pinned
  Ratatui line-count feature; no new dependency.
- Final isolated focused Rust 84/0/3, strict minimal/Web-only/TUI-Web/
  combined lint; real SGR PTY wheel/scrolled exact-row, both split drags,
  maximize/restore, raw-YAML independent detail scrollbar/selection,
  modal masking, opt-out/toggle, terminal restoration and original
  shell/appearance/settings/Firefox regressions all pass. Corrected
  initial ANSI differential fixture assertions. Logs: target/ux-p2-final.log.
- Full gate/real cached resident and transfer at assembled UX-P4.
  Stable anchors across data reorder, full modal/form/editor scrolling,
  caret/selection and full mouse action buttons/confirmations remain
  P3/P4. Native Auto theme probe remains deferred as before.

**Next:** UX-P3 complete visible mouse-accessible action controls, typed
pickers/modal buttons and editor interaction across nine views. Reuse
the original App::handle/request/confirmation and exact-owner guards;
no alternate runtime or configuration authority. UX-P4 final ergonomics
and acceptance before public serving.
~~~~

Changed paths:

~~~~text
M	.agents/docs/CURRENT_HANDOFF.md
M	.agents/plans/FULL_IMPLEMENTATION_PLAN.md
M	.agents/plans/README.md
M	.agents/plans/TODO.md
M	.agents/plans/cli-device-routing/TODO.md
M	.agents/plans/cli-ux/TUI_POLISH_AND_MOUSE_INTERACTION.md
M	crates/zerogravity-cli/Cargo.toml
M	crates/zerogravity-cli/src/control_host/settings.rs
M	crates/zerogravity-cli/src/tui.rs
A	crates/zerogravity-cli/src/tui/input.rs
A	crates/zerogravity-cli/src/tui/layout.rs
M	crates/zerogravity-cli/src/tui/terminal.rs
M	crates/zerogravity-cli/src/tui/view.rs
M	crates/zerogravity-cli/src/web/settings.rs
M	crates/zerogravity-cli/tests/config_cli.rs
M	crates/zerogravity-cli/tests/fixtures/tui_shell.py
M	crates/zerogravity/src/configuration.rs
M	crates/zerogravity/src/pipelines/document.rs
M	docs/CLI.md
~~~~


### D.9 — UX-P3A action controls

Commit: 5da962513b051b5296c28e107a321ffb0cbd6afc

Subject: feat(ux): expose bound native actions and modal controls

Original landing narrative (historical instructions):

~~~~markdown
### Current checkpoint — UX-P3A native action controls (26 Nov)

- New presentation-only controls.rs: all nine views have width-stable
  action strips, F4/More overflow with disabled reasons. One Rc scope
  per rendered surface captures exact row/ready owner/observed file,
  config revision/active document, listener generation, transfer and
  active action identities plus availability. Press/release matching,
  fresh live snapshot and shared slot revalidation before original
  Action dispatch; conservative refusal on observation change. No
  new runtime/controller/authority or dependency.
- Modal Apply/Send/Save/Confirm/Cancel/Close and prompt chat/raw buttons
  converge on existing keyboard submit. Confirm preserves exact Request,
  disabled buttons have no hit, stale press refuses, busy submit retains
  form. Menu scroll is local. Keyboard shortcuts unchanged, F4 opens
  menu, F9 output follow. No auto execution on selecting any row.
- Strict minimal/Web-only/TUI-Web/combined lint and optimized Rust
  82/0/3. Real PTY controls (create, theme preview/cancel/save, overflow
  activate, remove cancel/confirm, release mismatch, concurrent Web edit,
  narrow keyboard/mouse menu), original P2 mouse/shell/settings/appearance
  pass. New tests integrated in final default-release gate; exact
  default CPU/model/text/TUI/Web build and all four native PTY
  invocations passed without serve/acquisition.
- Actual controlled download safe pause/resume/durable/cancel via native
  mouse; actual approved cached Qwen 0.8B CPU/browser/native/HTTP fixture
  1/0/0 (52.81s, five filtered), original 8 GiB admission, mouse device
  policy/cache inspect/verify/alias/removal decline/run/prompt submit/
  listener start/stop and retire. Exact stale Web identities and shared
  active policy checks remain intact. Logs target/ux-p3a-final.log,
  target/ux-p3a-pty-rerun2.log, target/ux-p3a-owners.log.
- Initial PTY caught variable disabled width; fixed slots. Remaining
  initial assertion failures were ANSI differential output; waited on
  actual selected/control labels with exact file/outcome assertions.
  Final focused tests pass; full workspace acceptance at P4.

**Next:** UX-P3B typed settings and log filter pickers, model shelf search,
mouse caret/text selection and editor scrolling, remaining compact/modal
and keyboard focus affordances. P4 stable scroll anchors, final visual/
ergonomic review, original real owner regressions and full gate before
public serving. Native Auto active probe still deferred.
~~~~

Changed paths:

~~~~text
M	.agents/docs/CURRENT_BASELINE.md
M	.agents/docs/CURRENT_HANDOFF.md
M	.agents/plans/FULL_IMPLEMENTATION_PLAN.md
M	.agents/plans/README.md
M	.agents/plans/TODO.md
M	.agents/plans/cli-device-routing/TODO.md
M	.agents/plans/cli-ux/TUI_POLISH_AND_MOUSE_INTERACTION.md
M	crates/zerogravity-cli/src/tui.rs
A	crates/zerogravity-cli/src/tui/controls.rs
M	crates/zerogravity-cli/src/tui/input.rs
M	crates/zerogravity-cli/src/tui/theme.rs
M	crates/zerogravity-cli/src/tui/view.rs
M	crates/zerogravity-cli/tests/fixtures/tui_shell.py
M	docs/CLI.md
M	scripts/check.sh
~~~~


### D.10 — UX-P3B pickers/editor

Commit: 8a2c2ae88fbda55790c27249c6782bf852c29544

Subject: feat(ux): add typed pickers and native grapheme editing

Original landing narrative (historical instructions):

~~~~markdown
### Current checkpoint — UX-P3B settings/filter and editor interaction (26 Nov)

- New choice.rs finite setting and log picker; initialized from the
  actual observed effective choice. Edit/e uses typed picker, Enter
  retains direct YAML value editing, F2 edits chosen value as YAML,
  nonlisted/custom values retain lossless original form. Exact
  ConfigFile snapshot and existing Submit/Set owner; Apply explicit.
  Stale choice refuses, draft retained via existing retry path.
  Local Logs l/f pickers, uppercase L/F original cycles.
- Rows now retains original inventory and visible indices; local
  per-shelf search, no network/acquisition or second catalog. Selection
  only visible exact keys, refresh reconciles, clearing restores
  inventory. Original config observation/filter ownership preserved.
- editor.rs one draft caret/anchor/visual viewport; full grapheme
  boundary handling (wide/combining/CRLF), rendered row-range hit
  translation, mouse click/drag, Shift+arrows, selection replacement
  and bounded paste, wheel/drag scrollbar/PageUp/Down. Display sanitizes
  controls, tab has visible four-cell representation; raw bytes remain
  unmodified except explicit edit. Resize preserves visual byte anchor
  for manual scroll; actual cursor hidden off viewport. Shared
  Viewport usize offsets support >65k editor rows; ordinary Paragraph
  u16 bound explicit. Existing locked unicode-segmentation 1.13.3 and
  unicode-width 0.2.2 become direct optional TUI deps only; lock edges
  changed, no package/version/download. No clipboard or new authority.
- Strict feature lint, optimized Rust 86/0/3 (58/0/3 CLI units), actual
  typed picker Apply/Cancel, stale finite draft refusal, log/mouse local
  model search, long raw scroll, wide+combining drag/caret exact saved
  bytes; original controls/mouse/shell/settings/appearance PTY pass.
  Actual cached Qwen CPU/browser/native/HTTP 1/0/0 (55.96s, five filtered)
  with original admission and exact output/stale identities/shared
  settings retained. Actual controlled transfer mouse pause/resume/
  durable/cancel and default CPU/model/text/TUI/Web release plus all
  native PTY fixtures passed. New editor fixture integrated in full
  default release gate; target/ux-p3b-default-release.log. Initial fixture assertions corrected for differential
  rendering and the expected ERROR log after deliberate stale refusal.
  Logs target/ux-p3b-final.log and target/ux-p3b-final-rerun.log.
- **Next UX-P4A:** final ergonomic/visual review, stable item/log/output
  scroll anchors, distinct Running selected detail/response affordance,
  safe log-row detail, small/modal/keyboard focus and editor drag edge
  cases; finish remaining view actions. P4B final real simultaneous
  resident/transfer/config and full isolated gate (separate coherent
  turn if needed). No accepted UX-P exit or public serving yet.
  Native Auto probe remains deferred; no performance/GPU claims.
~~~~

Changed paths:

~~~~text
M	.agents/docs/CURRENT_BASELINE.md
M	.agents/docs/CURRENT_HANDOFF.md
M	.agents/plans/FULL_IMPLEMENTATION_PLAN.md
M	.agents/plans/README.md
M	.agents/plans/TODO.md
M	.agents/plans/cli-device-routing/TODO.md
M	.agents/plans/cli-ux/TUI_POLISH_AND_MOUSE_INTERACTION.md
M	Cargo.lock
M	crates/zerogravity-cli/Cargo.toml
M	crates/zerogravity-cli/src/tui.rs
A	crates/zerogravity-cli/src/tui/choice.rs
M	crates/zerogravity-cli/src/tui/controls.rs
A	crates/zerogravity-cli/src/tui/editor.rs
M	crates/zerogravity-cli/src/tui/input.rs
M	crates/zerogravity-cli/src/tui/layout.rs
M	crates/zerogravity-cli/src/tui/library.rs
M	crates/zerogravity-cli/src/tui/logs.rs
M	crates/zerogravity-cli/src/tui/settings.rs
M	crates/zerogravity-cli/src/tui/view.rs
M	crates/zerogravity-cli/tests/fixtures/tui_shell.py
M	docs/CLI.md
M	scripts/check.sh
~~~~


### D.11 — UX-P4A1 minimalism

Commit: f723b0b6585dd2a0749895702c123a7d0a74db1e

Subject: refactor(ux): simplify native chrome and disclose details on demand

Original landing narrative (historical instructions):

~~~~markdown
### Current checkpoint — UX-P4A1 minimal presentation (26 Nov)

User supplied two actual Alacritty screenshots and requested less
visual overload, no redundant top-bar destinations or random text,
details on request and smart size units. This overrides the earlier
wide action strip. Owning UX-P specification §1 and §5.3 updated.

- At most three contextual quick actions + More/F4 regardless of width.
  Cached Models Load/Run/Search, Catalog/Partial Pull/Search, Home
  Run/Load; secondary operations, appearance, refresh/quit in menu.
  No second visible navigation list or duplicate shelf tabs in toolbar.
  Header brand/profile, quiet footer and active cancellation. Clear
  old view notice on navigation; retain original work/result identity.
- Compact Home/Serve/Downloads/Running summaries, one library context
  row, aligned name/size annotations, selected summary, active profile
  distinct from selection, compact log/filter/empty states. When both
  panes visible, no redundant cross-pane switch; maximize retained.
  Details/D read-only scrollable modal includes exact selected key
  metadata and last action result, no implicit Inspect/Verify/effect.
  Row selection/identity and existing More/confirmation guards intact.
- format.rs automatically selects IEC B/KiB/MiB/GiB/TiB/PiB/EiB with
  at most two decimals; cached/partial, memory and transfer summaries.
  Exact bytes available in technical observation, unchanged wire and
  original effect results. Unknown never becomes zero.
- Actual TestBackend cell review (example cached model sizes), no
  raw digest or exact byte count in normal list/detail, explicit
  Details reveals full digest. Strict feature lint and optimized Rust
  87/0/3. Native mouse split/scroll/Details, action controls, editor,
  shell/settings/appearance and actual controlled download passed. Exact default CPU/model/text/
  TUI/Web release and all native PTY invocations also passed (target/
  ux-p4a-minimal-default.log).
  PTY fixture geometry and differential assertions adjusted (including
  private Web URL fresh paint without logging credential).
  Actual cached-owner combined reruns reached browser resident success
  twice; native Details, inspect, verify and alias worked but fixture
  proceeded before settled receipt/inventory (assertions corrected,
  guards retained). Last combined rerun timed out at browser model
  load. Native-only retry explicitly refused original 8 GiB envelope:
  7,538,163,712 bytes after headroom available (7188 MiB). No lowering
  admission or touching unrelated host jobs. Final cached-owner rerun
  pending healthy host, P4A2/P4B; no new full acceptance claim.
  Logs target/ux-p4a-minimal-final.log, -pty2.log, -pty3.log,
  -owners2.log, -resident-final.log, -resident-rerun.log, -native-owner.log
  (all under target/ux-p4a-minimal prefix).
  No new dependency, runtime authority or model download.
- Remaining UX-P4A2 stable item/log/output anchors, distinct Running
  selected-detail/response, safe log details, compact modal/keyboard
  focus and editor drag edges; P4B simultaneous-owner/full isolated
  acceptance. No public serving or overall UX-P exit yet.
~~~~

Changed paths:

~~~~text
M	.agents/docs/CURRENT_BASELINE.md
M	.agents/docs/CURRENT_HANDOFF.md
M	.agents/plans/FULL_IMPLEMENTATION_PLAN.md
M	.agents/plans/README.md
M	.agents/plans/TODO.md
M	.agents/plans/cli-device-routing/TODO.md
M	.agents/plans/cli-ux/TUI_POLISH_AND_MOUSE_INTERACTION.md
M	crates/zerogravity-cli/src/tui.rs
M	crates/zerogravity-cli/src/tui/controls.rs
A	crates/zerogravity-cli/src/tui/format.rs
M	crates/zerogravity-cli/src/tui/input.rs
M	crates/zerogravity-cli/src/tui/library.rs
M	crates/zerogravity-cli/src/tui/settings.rs
M	crates/zerogravity-cli/src/tui/view.rs
M	crates/zerogravity-cli/tests/fixtures/tui_shell.py
M	docs/CLI.md
~~~~


### D.12 — UX-P4A1 toolbar-free correction

Commit: d1e27080776bfff8bf593bcc9c503df43ff92fa2

Subject: refactor(ux): replace native toolbar with contextual menus and live search

Original landing narrative (historical instructions):

~~~~markdown
### Current checkpoint — UX-P4A1 follow-up: toolbar-free contextual UI (26 Nov)

User rejected even a minimal top toolbar: select an object, show its
actions in context, only inline search at the top. Supersedes previous
wide/curated action strips and modal search.

- Removed action_bar and FormKind::Search, replaced by the existing
  grapheme editor in a single-line filter. All printable input in
  Models/Devices/Profiles/Config/Running/Logs filters immediately,
  including former operation/digit keys; Ctrl-F focuses, Ctrl-U clears,
  Esc clears active search, arrows select, Tab leaves search. Local
  per-shelf/row indices, config rows moved from observed projection into
  same Rows owner, no duplicate filtered inventory/provider search.
- Selected row ellipsis/right-click and F4 open one compact menu
  titled with exact selected context, no effect on row selection.
  Cached Load/Run/Details/Inspect/Verify/Alias/Load-for-Serve/removal;
  catalog/partial only applicable pull/details. Ready registration,
  device, profile and setting context. One pane fallback when selected
  row not visible; Home/Serve/Downloads/Logs use pane action menu.
  Footer session menu/F10 holds Web/Appearance/Refresh/Quit. Existing
  context Scope and Bound dispatch revalidate live owner and slot;
  matching right-click release, stale row refusal, no motion effects.
- Search owns printable keys. Alt+1–9 navigation, Alt+operation letters,
  F1/Alt+? help, Alt+: palette, Ctrl-Q quit outside dialog. Non-search
  views retain ordinary shortcuts. Enter model/device/registration
  context, profiles open editor, Config direct YAML edit; typed Edit
  via menu/Alt+e. No new runtime or configuration ownership or dependency.
- Strict minimal/Web-only/TUI-Web/combined lint; 15/0 focused native
  Rust tests and actual rendered-cell inspection, optimized Rust 88/0/3.
  Real SGR ellipsis/right-click, mouse/split/scroll, local type/drag
  filtering, typed picker, profile/context/confirmation and stale Web
  release, narrow/session menu, original shell/settings/appearance,
  controlled download pause/resume/durable/cancel pass. Actual approved
  cached Qwen CPU/browser/native/HTTP 1/0/0 (57.77s, five filtered) with
  original 8 GiB admission and exact output, mouse context inspect/verify/
  alias/run/removal, listener and stale identities, shared settings pass.
  Previous host memory blocker resolved without policy change. Actual
  attached Firefox/native shared transfer durable and cancelled pass.
  Exact default CPU/model/text/TUI/Web release and all native PTY
  invocations passed. Additional logs: target/ux-context-resident.log,
  target/ux-context-default.log.
  Test-only geometry and differential-render assertions adjusted; no effect guard
  weakened. Logs target/ux-context-acceptance.log, -pty-rerun.log,
  -pty-final.log. Remaining P4A2 stable scroll anchors, separate Running
  selected detail/response, safe log row details, compact/focus/editor
  edges and P4B full acceptance. No public serving or performance claim.
~~~~

Changed paths:

~~~~text
M	.agents/docs/CURRENT_BASELINE.md
M	.agents/docs/CURRENT_HANDOFF.md
M	.agents/plans/FULL_IMPLEMENTATION_PLAN.md
M	.agents/plans/README.md
M	.agents/plans/TODO.md
M	.agents/plans/cli-device-routing/TODO.md
M	.agents/plans/cli-ux/TUI_POLISH_AND_MOUSE_INTERACTION.md
M	crates/zerogravity-cli/src/tui.rs
M	crates/zerogravity-cli/src/tui/controls.rs
M	crates/zerogravity-cli/src/tui/input.rs
M	crates/zerogravity-cli/src/tui/library.rs
M	crates/zerogravity-cli/src/tui/logs.rs
M	crates/zerogravity-cli/src/tui/view.rs
M	crates/zerogravity-cli/tests/fixtures/tui_shell.py
M	docs/CLI.md
~~~~


### D.13 — UX-P4A2 ergonomics

Commit: 0ee66351893b707de1687511729cd1ac440655d8

Subject: feat(ux): preserve scroll anchors and refine response and compact dialogs

Original landing narrative (historical instructions):

~~~~markdown
### Current checkpoint — UX-P4A2 final ergonomics (26 Nov)

Toolbar-free/object-first layout retained; no new permanent chrome.

- layout::Anchor<K> keeps preferred visible identity/relative row gap,
  last paint and numeric fallback through reorder, empty refresh and
  viewport clamp. Per-library shelf/list, Running ready owner, log
  sequence and response byte-position anchors. Explicit scroll/reveal
  supersedes preferred anchor. Shelf switch restores last offset,
  first visit reveals selection. No duplicate inventory/history.
- Running separates model metadata from response (detail title/F3);
  response has an independent Viewport and follow state in each
  Home/Running view. New response follows and focuses detail; selecting
  another registration shows model details. Visible grapheme ranges
  only are painted from original bounded Response text; byte anchor
  preserves position through wrapping/append. Scrolling metadata never
  changes response follow. Tail/F9 resumes original output.
- Log list uses single-row entries, exact safe sequence selection
  and Details via ellipsis/right-click/Enter/F4. Original bounded ring
  and paused revision cap, no sensitive raw payload or extra history.
  Selecting/manual scroll pauses; End/downward tail resumes; evicted
  sequence refuses and anchors fall back to first retained entry.
- Dialogs use full screen when height <12; Help/Web URL/Details/
  confirm share bounded scroll rendering. Compact destination palette
  scrolls to cursor and masks hidden destinations. Explicit editor
  capture supports edge drag (one step per real event, no timer) and
  horizontal clamp without click-through. Tiny view does not turn
  search q into quit. All original typed action/confirmation guards.
- 18/0 focused native tests incl actual TestBackend rendered anchors
  (reorder/clamp/shelf), distinct model/response viewports and 20x7
  selectable menu. Strict minimal/Web-only/TUI-Web/combined lint,
  optimized Rust 91/0/3, actual compact 20x7 PTY context/palette/help,
  original native mouse/action/editor/settings/theme/shell with safe
  log-detail/End-follow tests, controlled download and attached
  native/Firefox durable/cancelled shared transfer passed. New
  compact fixture in standard release gate. Exact default CPU/TUI/Web
  release and all native PTY passed before final response-only fix.
  Real browser/native lifecycle first exposed digest alias concealing
  registration in the new single-line heading: put registration first,
  truncate alias, F9 focuses detail in collapsed/maximized layout.
  Strict combined lint/18 focused tests rerun on final source. Second
  combined run passed browser resident and reached native F3 metadata;
  PTY split Context/capacity across ANSI updates. Corrected assertion
  to capacity:. Final native/attached-Web cached CPU/HTTP 1/0/0
  (34.49s, five filtered), original 8 GiB admission and exact output,
  response/model switch, stale identities/shared settings all pass.
  Browser portion passed both combined runs; full combined final and
  full workspace gate remain P4B. No new dependency or GPU/performance
  claim. Logs target/ux-p4a2-resident.log, -resident-final.log,
  -native-final.log.
- Next UX-P4B: full isolated workspace gate, final actual cached
  resident/transfer/config regressions on assembled source; retain
  original 8 GiB admission, no relaxed tests. Native Auto remains
  hint/dark fallback; no active probe. No overall UX-P acceptance
  or public serving yet. Logs target/ux-p4a2-render.log,
  target/ux-p4a2-acceptance.log, target/ux-p4a2-default.log.
~~~~

Changed paths:

~~~~text
M	.agents/docs/CURRENT_BASELINE.md
M	.agents/docs/CURRENT_HANDOFF.md
M	.agents/plans/FULL_IMPLEMENTATION_PLAN.md
M	.agents/plans/README.md
M	.agents/plans/TODO.md
M	.agents/plans/cli-device-routing/TODO.md
M	.agents/plans/cli-ux/TUI_POLISH_AND_MOUSE_INTERACTION.md
M	crates/zerogravity-cli/src/control_host/logs.rs
M	crates/zerogravity-cli/src/tui.rs
M	crates/zerogravity-cli/src/tui/controls.rs
M	crates/zerogravity-cli/src/tui/editor.rs
M	crates/zerogravity-cli/src/tui/input.rs
M	crates/zerogravity-cli/src/tui/layout.rs
M	crates/zerogravity-cli/src/tui/logs.rs
M	crates/zerogravity-cli/src/tui/view.rs
M	crates/zerogravity-cli/tests/fixtures/tui_shell.py
M	docs/CLI.md
M	scripts/check.sh
~~~~


### D.14 — Full production defaults and error diagnostics

Commit: c0e4794d9517f03d989852e945d0b8db7e978448

Subject: feat(cli): enable full production defaults and preserve safe error diagnostics

Original landing narrative (historical instructions):

~~~~markdown
### Current checkpoint — full native CLI defaults and typed error diagnostics (26 Nov)

User requested sane full-production Cargo defaults and accurate failure
classification after pvl-a1 original-source load reported missing CUDA/ROCm
as a malformed model. User also requested optional HTTP diagnostic metadata,
hidden from public responses by default and carried in operational logs.

- CLI default = ["all"], existing production all-models/all-backends plus
  acquisition, serve, TUI, Web. CPU/CUDA/HIP/Vulkan and current Qwen/Gemma
  source/model features; no experimental attention or diagnostic/qualification
  features. Facade embedding defaults and explicit portable CPU Nix/release
  definitions unchanged. No physical capability grant: CUDA source still
  SM86-qualified, HIP needs eligible AMD and runtime, Gemma source contract
  is not native inference qualification. Explicit --no-default-features
  remains supported; default feature graph/compile added to GPU boundary gate.
- engine router/error.rs owns ServiceError conversion and FailureDetails.
  Retain original core failure through shared Arc on error path, including
  concrete resource budget and I/O causes. InvalidInput distinct from
  malformed artifact; synthetic service errors classified by code, then
  status fallback, no invented resource amount. Native original-source
  preflight reports missing backends, unsupported source/device and exact
  CUDA capacity; no qualification/selection or admission guard relaxed.
- One ordinary HTTP/SSE error projection: original status/code/type and
  fixed safe public message. serve.enable_error_details false by default,
  --enable-error-details or typed native config; explicit loopback listener
  with configured bearer key required before loading/listening. Only
  successfully authenticated requests get error.details (safe metadata,
  bounded diagnostic and up to three causes). Pre-auth/framing/admission
  errors always redacted. Browser-safe editor cannot enable diagnostics.
  Trusted debugging only, not an untrusted reverse-proxied endpoint.
- FailureDetails contains safe category/code/status and measured requested/
  available bytes/OS code when present, no arbitrary text. Listener's
  optional error observer captures only the existing bounded control event
  ring, tagged by listener generation. Shared native/Web operational logs
  retain structured metadata regardless of HTTP detail setting, expose
  details on demand, never raw error paths/prompts/credentials. Original
  full diagnostic remains in native error/CLI. No new runtime/history owner.
- Focused core/engine/serving 190/108/27, SDK 5, native CLI 63/0/3,
  CLI grammar 15, config 1, pipeline 13 passed across final affected suites.
  Actual loopback HTTP and SSE failure test checks authenticated opt-in,
  public and pre-auth redaction and safe observer data. Existing SDK
  assertion updated from generic limit to exact unconfigured-runtime
  refusal. New config test uses valid separate serving profile; original
  run-only context fixture retained. Native log filter test checks failure
  metadata before unrelated registration filtering. Minimal and combined
  CPU-feature strict lint, JS build/tests and default cargo check passed.
  Linked default cargo run --version reached final CLI compilation but
  timed out at 300s; linked launch and doctor not yet verified. New
  control-listener event assertion in cached SDK fixture awaits P4B.
  Full-production strict lint exposed existing GPU primitive dead-code/
  Clippy debt; not removed/hidden. Full workspace gate, assembled actual
  model/client acceptance and hardware qualification remain pending.
- Logs: target/ux-default-error-chain-check.log,
  target/ux-default-production.log, target/ux-error-details-final.log,
  target/ux-error-default-acceptance.log, target/ux-error-default-finish.log.
  Next P4B full isolated gate and resident/transfer/config acceptance;
  public serving remains dormant. Keep error projection and defaults in
  subsequent serving design.
~~~~

Changed paths:

~~~~text
M	.agents/docs/CURRENT_BASELINE.md
M	.agents/docs/CURRENT_HANDOFF.md
M	.agents/plans/FULL_IMPLEMENTATION_PLAN.md
M	.agents/plans/TODO.md
M	.agents/plans/cli-device-routing/TODO.md
M	.agents/plans/serving-and-speculation/NEXT_ITERATION_OPENAI_COMPATIBLE_SERVING.md
M	crates/zerogravity-cli/Cargo.toml
M	crates/zerogravity-cli/README.md
M	crates/zerogravity-cli/src/control_host/actions.rs
M	crates/zerogravity-cli/src/control_host/logs.rs
M	crates/zerogravity-cli/src/control_host/settings.rs
M	crates/zerogravity-cli/src/control_host/work.rs
M	crates/zerogravity-cli/src/output/presentation.rs
M	crates/zerogravity-cli/src/resident.rs
M	crates/zerogravity-cli/src/run_cli/qwen_source.rs
M	crates/zerogravity-cli/src/serve/canonical.rs
M	crates/zerogravity-cli/src/serve/mod.rs
M	crates/zerogravity-cli/src/tui/logs.rs
M	crates/zerogravity-cli/src/tui/settings.rs
M	crates/zerogravity-cli/src/web/http.rs
M	crates/zerogravity-cli/tests/config_cli.rs
M	crates/zerogravity-cli/web/dist/app.js
M	crates/zerogravity-cli/web/src/settings.jsx
M	crates/zerogravity-core/src/lib.rs
M	crates/zerogravity-engine/src/router.rs
M	crates/zerogravity-engine/src/router/dispatch.rs
A	crates/zerogravity-engine/src/router/error.rs
M	crates/zerogravity-engine/src/runtime.rs
M	crates/zerogravity-serving/src/auth.rs
M	crates/zerogravity-serving/src/http.rs
M	crates/zerogravity-serving/src/http/shutdown_tests.rs
M	crates/zerogravity-serving/src/server.rs
M	crates/zerogravity-serving/src/server/tests.rs
M	crates/zerogravity/src/configuration.rs
M	crates/zerogravity/src/control.rs
M	crates/zerogravity/src/control/events.rs
M	crates/zerogravity/src/control/serving.rs
M	crates/zerogravity/src/pipelines/document.rs
M	crates/zerogravity/tests/fabric_sdk.rs
M	docs/CLI.md
M	docs/DEVELOPMENT.md
M	scripts/check-gpu-feature-boundaries.sh
M	scripts/check.sh
~~~~


### D.15 — UX-P-M1 live metrics

Commit: e96cdab98d47c5c84001c1afaaebd8f126b9e0b3

Subject: feat(ux): show live registration performance and layer execution details

Original landing narrative (historical instructions):

~~~~markdown
### Current checkpoint — UX-P-M1 live model performance (26 Nov)

User asked for elegant live TPS/TTFT averages and optional deeper layer/
device performance inspection. Owning design: [LIVE_MODEL_PERFORMANCE.md](../plans/cli-ux/LIVE_MODEL_PERFORMANCE.md).

- Existing exact registry Route owns fixed registration counters;
  original ModelRead carries request-local timing, no new reader, registry,
  scheduler or execution grant. Existing observed committed non-EOS
  timestamp and generated ID frontier drive measurement, once per token;
  no extra per-token clock, allocation, mutex or GPU synchronization.
  Relaxed atomic output/interval sums and first-token sums; terminal
  receipt records completed count and last sample once. Existing
  sizeof-based admission prices route/queued/active additions.
  TTFT = engine acquire to first commit (queue/prefill included), not
  wire delivery; decode = token-weighted within-request intervals,
  active/ended samples incl partial, no idle across requests. Zero-token
  completion has no TTFT/rate sample. Alias shares registration; reload
  resets. Snapshot fields may advance between observations.
- Running selected pane and Web: two prominent metrics, monotonic
  registration age, total output/completions, recent aggregate-output
  sparkline at existing client sampling cadence, <=24 selected-owner
  samples, reset on >6s gap, regression or replacement. No persistence/
  extra timer. No toolbar or duplicate nav; digest-only labels shortened
  with observed family, full exact aliases and secondary technical facts
  in Details. F3 response switch unchanged. Alt-D/context opens read-only
  performance details; Web disclosure. Last completed is registration
  sample, not attributed to an unrelated visible local response.
- Adapter ModelDetails.layer_placement: actual zero-based half-open
  layer ranges. CPU Qwen uses verified geometry on Host CPU; original
  whole-text single-device prepared source uses exact plan boundaries
  and resident device. Older resident adapter does not report a map.
  Layer compute distinct from weights/state/admission; safe memory
  bounds and metrics projected explicitly into authenticated Web
  snapshot, no change to public inference protocol.
- Focused engine 109/0/0, combined CPU CLI 64/0/3, Web 4/0 and asset
  build, strict engine/minimal/combined CPU lint, full default cargo
  check passed (known GPU warnings). Actual approved cached Qwen CPU/
  browser/native/HTTP 1/0/0 (55.61s, five filtered), original 8 GiB
  admission, exact output and stale owner/shared settings retained.
  Actual two native + HTTP requests share counts, Web increments same
  registration, reload resets, exact 24 CPU layers, native F3 and
  scrollable details/Firefox visible metrics. Initial fixture failures
  corrected for differential terminal output and capitalization;
  substantive counters/placement assertions passed. Logs
  target/ux-live-performance-final.log, target/ux-live-performance-web-final.log,
  target/ux-live-performance-resident-final.log. GPU placement
  compiled only, no new GPU performance/physical qualification claim.
- Remaining M2 optional exact per-request/response and phase timings,
  actual additional adapter layer maps and explicit detailed profiling
  if justified; no imaginary offload or default per-layer sync. P4B
  full offline isolated gate and assembled owner acceptance; linked
  full production launch from prior checkpoint remains pending. No
  public serving acceptance yet.
~~~~

Changed paths:

~~~~text
M	.agents/docs/CURRENT_BASELINE.md
M	.agents/docs/CURRENT_HANDOFF.md
M	.agents/plans/FULL_IMPLEMENTATION_PLAN.md
M	.agents/plans/README.md
M	.agents/plans/TODO.md
M	.agents/plans/cli-device-routing/TODO.md
A	.agents/plans/cli-ux/LIVE_MODEL_PERFORMANCE.md
M	.agents/plans/cli-ux/TUI_POLISH_AND_MOUSE_INTERACTION.md
M	crates/zerogravity-cli/src/tui.rs
M	crates/zerogravity-cli/src/tui/controls.rs
A	crates/zerogravity-cli/src/tui/performance.rs
M	crates/zerogravity-cli/src/tui/view.rs
M	crates/zerogravity-cli/src/web/wire.rs
M	crates/zerogravity-cli/tests/fixtures/tui_shell.py
M	crates/zerogravity-cli/web/dist/app.js
M	crates/zerogravity-cli/web/dist/style.css
M	crates/zerogravity-cli/web/src/browser-smoke.mjs
M	crates/zerogravity-cli/web/src/main.jsx
A	crates/zerogravity-cli/web/src/performance.js
A	crates/zerogravity-cli/web/src/performance.jsx
A	crates/zerogravity-cli/web/src/performance.test.js
M	crates/zerogravity-cli/web/src/styles.css
M	crates/zerogravity-engine/src/diagnostics.rs
M	crates/zerogravity-engine/src/router.rs
M	crates/zerogravity-engine/src/router/cpu_tests.rs
A	crates/zerogravity-engine/src/router/performance.rs
M	crates/zerogravity-engine/src/router/registry.rs
M	crates/zerogravity-engine/src/router/worker.rs
M	crates/zerogravity/src/builtins/text/cpu.rs
M	crates/zerogravity/src/builtins/text/gpu.rs
M	crates/zerogravity/src/builtins/text/qwen_source.rs
M	crates/zerogravity/src/models.rs
M	docs/CLI.md
~~~~


### D.16 — reconciled latest landing narratives at implementation HEAD

Verbatim current-hand-off checkpoint sections from M1 back through UX-9C. This captures later corrections and acceptance state, without replacing the original landing snapshots above.

~~~~markdown
### Current checkpoint — UX-P-M1 live model performance (26 Nov)

User asked for elegant live TPS/TTFT averages and optional deeper layer/
device performance inspection. Owning design: [LIVE_MODEL_PERFORMANCE.md](../plans/cli-ux/LIVE_MODEL_PERFORMANCE.md).

- Existing exact registry Route owns fixed registration counters;
  original ModelRead carries request-local timing, no new reader, registry,
  scheduler or execution grant. Existing observed committed non-EOS
  timestamp and generated ID frontier drive measurement, once per token;
  no extra per-token clock, allocation, mutex or GPU synchronization.
  Relaxed atomic output/interval sums and first-token sums; terminal
  receipt records completed count and last sample once. Existing
  sizeof-based admission prices route/queued/active additions.
  TTFT = engine acquire to first commit (queue/prefill included), not
  wire delivery; decode = token-weighted within-request intervals,
  active/ended samples incl partial, no idle across requests. Zero-token
  completion has no TTFT/rate sample. Alias shares registration; reload
  resets. Snapshot fields may advance between observations.
- Running selected pane and Web: two prominent metrics, monotonic
  registration age, total output/completions, recent aggregate-output
  sparkline at existing client sampling cadence, <=24 selected-owner
  samples, reset on >6s gap, regression or replacement. No persistence/
  extra timer. No toolbar or duplicate nav; digest-only labels shortened
  with observed family, full exact aliases and secondary technical facts
  in Details. F3 response switch unchanged. Alt-D/context opens read-only
  performance details; Web disclosure. Last completed is registration
  sample, not attributed to an unrelated visible local response.
- Adapter ModelDetails.layer_placement: actual zero-based half-open
  layer ranges. CPU Qwen uses verified geometry on Host CPU; original
  whole-text single-device prepared source uses exact plan boundaries
  and resident device. Older resident adapter does not report a map.
  Layer compute distinct from weights/state/admission; safe memory
  bounds and metrics projected explicitly into authenticated Web
  snapshot, no change to public inference protocol.
- Focused engine 109/0/0, combined CPU CLI 64/0/3, Web 4/0 and asset
  build, strict engine/minimal/combined CPU lint, full default cargo
  check passed (known GPU warnings). Actual approved cached Qwen CPU/
  browser/native/HTTP 1/0/0 (55.61s, five filtered), original 8 GiB
  admission, exact output and stale owner/shared settings retained.
  Actual two native + HTTP requests share counts, Web increments same
  registration, reload resets, exact 24 CPU layers, native F3 and
  scrollable details/Firefox visible metrics. Initial fixture failures
  corrected for differential terminal output and capitalization;
  substantive counters/placement assertions passed. Logs
  target/ux-live-performance-final.log, target/ux-live-performance-web-final.log,
  target/ux-live-performance-resident-final.log. GPU placement
  compiled only, no new GPU performance/physical qualification claim.
- Remaining M2 optional exact per-request/response and phase timings,
  actual additional adapter layer maps and explicit detailed profiling
  if justified; no imaginary offload or default per-layer sync. P4B
  full offline isolated gate and assembled owner acceptance; linked
  full production launch from prior checkpoint remains pending. No
  public serving acceptance yet.

### Previous checkpoint — full native CLI defaults and typed error diagnostics (26 Nov; c0e4794d)

User requested sane full-production Cargo defaults and accurate failure
classification after pvl-a1 original-source load reported missing CUDA/ROCm
as a malformed model. User also requested optional HTTP diagnostic metadata,
hidden from public responses by default and carried in operational logs.

- CLI default = ["all"], existing production all-models/all-backends plus
  acquisition, serve, TUI, Web. CPU/CUDA/HIP/Vulkan and current Qwen/Gemma
  source/model features; no experimental attention or diagnostic/qualification
  features. Facade embedding defaults and explicit portable CPU Nix/release
  definitions unchanged. No physical capability grant: CUDA source still
  SM86-qualified, HIP needs eligible AMD and runtime, Gemma source contract
  is not native inference qualification. Explicit --no-default-features
  remains supported; default feature graph/compile added to GPU boundary gate.
- engine router/error.rs owns ServiceError conversion and FailureDetails.
  Retain original core failure through shared Arc on error path, including
  concrete resource budget and I/O causes. InvalidInput distinct from
  malformed artifact; synthetic service errors classified by code, then
  status fallback, no invented resource amount. Native original-source
  preflight reports missing backends, unsupported source/device and exact
  CUDA capacity; no qualification/selection or admission guard relaxed.
- One ordinary HTTP/SSE error projection: original status/code/type and
  fixed safe public message. serve.enable_error_details false by default,
  --enable-error-details or typed native config; explicit loopback listener
  with configured bearer key required before loading/listening. Only
  successfully authenticated requests get error.details (safe metadata,
  bounded diagnostic and up to three causes). Pre-auth/framing/admission
  errors always redacted. Browser-safe editor cannot enable diagnostics.
  Trusted debugging only, not an untrusted reverse-proxied endpoint.
- FailureDetails contains safe category/code/status and measured requested/
  available bytes/OS code when present, no arbitrary text. Listener's
  optional error observer captures only the existing bounded control event
  ring, tagged by listener generation. Shared native/Web operational logs
  retain structured metadata regardless of HTTP detail setting, expose
  details on demand, never raw error paths/prompts/credentials. Original
  full diagnostic remains in native error/CLI. No new runtime/history owner.
- Focused core/engine/serving 190/108/27, SDK 5, native CLI 63/0/3,
  CLI grammar 15, config 1, pipeline 13 passed across final affected suites.
  Actual loopback HTTP and SSE failure test checks authenticated opt-in,
  public and pre-auth redaction and safe observer data. Existing SDK
  assertion updated from generic limit to exact unconfigured-runtime
  refusal. New config test uses valid separate serving profile; original
  run-only context fixture retained. Native log filter test checks failure
  metadata before unrelated registration filtering. Minimal and combined
  CPU-feature strict lint, JS build/tests and default cargo check passed.
  Linked default cargo run --version reached final CLI compilation but
  timed out at 300s; linked launch and doctor not yet verified. New
  control-listener event assertion in cached SDK fixture awaits P4B.
  Full-production strict lint exposed existing GPU primitive dead-code/
  Clippy debt; not removed/hidden. Full workspace gate, assembled actual
  model/client acceptance and hardware qualification remain pending.
- Logs: target/ux-default-error-chain-check.log,
  target/ux-default-production.log, target/ux-error-details-final.log,
  target/ux-error-default-acceptance.log, target/ux-error-default-finish.log.
  Next P4B full isolated gate and resident/transfer/config acceptance;
  public serving remains dormant. Keep error projection and defaults in
  subsequent serving design.

### Previous checkpoint — UX-P4A2 final ergonomics (26 Nov; 0ee66351)

Toolbar-free/object-first layout retained; no new permanent chrome.

- layout::Anchor<K> keeps preferred visible identity/relative row gap,
  last paint and numeric fallback through reorder, empty refresh and
  viewport clamp. Per-library shelf/list, Running ready owner, log
  sequence and response byte-position anchors. Explicit scroll/reveal
  supersedes preferred anchor. Shelf switch restores last offset,
  first visit reveals selection. No duplicate inventory/history.
- Running separates model metadata from response (detail title/F3);
  response has an independent Viewport and follow state in each
  Home/Running view. New response follows and focuses detail; selecting
  another registration shows model details. Visible grapheme ranges
  only are painted from original bounded Response text; byte anchor
  preserves position through wrapping/append. Scrolling metadata never
  changes response follow. Tail/F9 resumes original output.
- Log list uses single-row entries, exact safe sequence selection
  and Details via ellipsis/right-click/Enter/F4. Original bounded ring
  and paused revision cap, no sensitive raw payload or extra history.
  Selecting/manual scroll pauses; End/downward tail resumes; evicted
  sequence refuses and anchors fall back to first retained entry.
- Dialogs use full screen when height <12; Help/Web URL/Details/
  confirm share bounded scroll rendering. Compact destination palette
  scrolls to cursor and masks hidden destinations. Explicit editor
  capture supports edge drag (one step per real event, no timer) and
  horizontal clamp without click-through. Tiny view does not turn
  search q into quit. All original typed action/confirmation guards.
- 18/0 focused native tests incl actual TestBackend rendered anchors
  (reorder/clamp/shelf), distinct model/response viewports and 20x7
  selectable menu. Strict minimal/Web-only/TUI-Web/combined lint,
  optimized Rust 91/0/3, actual compact 20x7 PTY context/palette/help,
  original native mouse/action/editor/settings/theme/shell with safe
  log-detail/End-follow tests, controlled download and attached
  native/Firefox durable/cancelled shared transfer passed. New
  compact fixture in standard release gate. Exact default CPU/TUI/Web
  release and all native PTY passed before final response-only fix.
  Real browser/native lifecycle first exposed digest alias concealing
  registration in the new single-line heading: put registration first,
  truncate alias, F9 focuses detail in collapsed/maximized layout.
  Strict combined lint/18 focused tests rerun on final source. Second
  combined run passed browser resident and reached native F3 metadata;
  PTY split Context/capacity across ANSI updates. Corrected assertion
  to capacity:. Final native/attached-Web cached CPU/HTTP 1/0/0
  (34.49s, five filtered), original 8 GiB admission and exact output,
  response/model switch, stale identities/shared settings all pass.
  Browser portion passed both combined runs; full combined final and
  full workspace gate remain P4B. No new dependency or GPU/performance
  claim. Logs target/ux-p4a2-resident.log, -resident-final.log,
  -native-final.log.
- Next UX-P4B: full isolated workspace gate, final actual cached
  resident/transfer/config regressions on assembled source; retain
  original 8 GiB admission, no relaxed tests. Native Auto remains
  hint/dark fallback; no active probe. No overall UX-P acceptance
  or public serving yet. Logs target/ux-p4a2-render.log,
  target/ux-p4a2-acceptance.log, target/ux-p4a2-default.log.

### Previous checkpoint — UX-P4A1 follow-up: toolbar-free contextual UI (26 Nov; d1e27080)

User rejected even a minimal top toolbar: select an object, show its
actions in context, only inline search at the top. Supersedes previous
wide/curated action strips and modal search.

- Removed action_bar and FormKind::Search, replaced by the existing
  grapheme editor in a single-line filter. All printable input in
  Models/Devices/Profiles/Config/Running/Logs filters immediately,
  including former operation/digit keys; Ctrl-F focuses, Ctrl-U clears,
  Esc clears active search, arrows select, Tab leaves search. Local
  per-shelf/row indices, config rows moved from observed projection into
  same Rows owner, no duplicate filtered inventory/provider search.
- Selected row ellipsis/right-click and F4 open one compact menu
  titled with exact selected context, no effect on row selection.
  Cached Load/Run/Details/Inspect/Verify/Alias/Load-for-Serve/removal;
  catalog/partial only applicable pull/details. Ready registration,
  device, profile and setting context. One pane fallback when selected
  row not visible; Home/Serve/Downloads/Logs use pane action menu.
  Footer session menu/F10 holds Web/Appearance/Refresh/Quit. Existing
  context Scope and Bound dispatch revalidate live owner and slot;
  matching right-click release, stale row refusal, no motion effects.
- Search owns printable keys. Alt+1–9 navigation, Alt+operation letters,
  F1/Alt+? help, Alt+: palette, Ctrl-Q quit outside dialog. Non-search
  views retain ordinary shortcuts. Enter model/device/registration
  context, profiles open editor, Config direct YAML edit; typed Edit
  via menu/Alt+e. No new runtime or configuration ownership or dependency.
- Strict minimal/Web-only/TUI-Web/combined lint; 15/0 focused native
  Rust tests and actual rendered-cell inspection, optimized Rust 88/0/3.
  Real SGR ellipsis/right-click, mouse/split/scroll, local type/drag
  filtering, typed picker, profile/context/confirmation and stale Web
  release, narrow/session menu, original shell/settings/appearance,
  controlled download pause/resume/durable/cancel pass. Actual approved
  cached Qwen CPU/browser/native/HTTP 1/0/0 (57.77s, five filtered) with
  original 8 GiB admission and exact output, mouse context inspect/verify/
  alias/run/removal, listener and stale identities, shared settings pass.
  Previous host memory blocker resolved without policy change. Actual
  attached Firefox/native shared transfer durable and cancelled pass.
  Exact default CPU/model/text/TUI/Web release and all native PTY
  invocations passed. Additional logs: target/ux-context-resident.log,
  target/ux-context-default.log.
  Test-only geometry and differential-render assertions adjusted; no effect guard
  weakened. Logs target/ux-context-acceptance.log, -pty-rerun.log,
  -pty-final.log. Remaining P4A2 landed above; P4B full acceptance
  pending. No public serving or performance claim.

### Previous checkpoint — UX-P4A1 minimal presentation (26 Nov; f723b0b6)

User supplied two actual Alacritty screenshots and requested less
visual overload, no redundant top-bar destinations or random text,
details on request and smart size units. This overrides the earlier
wide action strip. Owning UX-P specification §1 and §5.3 updated.

- At most three contextual quick actions + More/F4 regardless of width.
  Cached Models Load/Run/Search, Catalog/Partial Pull/Search, Home
  Run/Load; secondary operations, appearance, refresh/quit in menu.
  No second visible navigation list or duplicate shelf tabs in toolbar.
  Header brand/profile, quiet footer and active cancellation. Clear
  old view notice on navigation; retain original work/result identity.
- Compact Home/Serve/Downloads/Running summaries, one library context
  row, aligned name/size annotations, selected summary, active profile
  distinct from selection, compact log/filter/empty states. When both
  panes visible, no redundant cross-pane switch; maximize retained.
  Details/D read-only scrollable modal includes exact selected key
  metadata and last action result, no implicit Inspect/Verify/effect.
  Row selection/identity and existing More/confirmation guards intact.
- format.rs automatically selects IEC B/KiB/MiB/GiB/TiB/PiB/EiB with
  at most two decimals; cached/partial, memory and transfer summaries.
  Exact bytes available in technical observation, unchanged wire and
  original effect results. Unknown never becomes zero.
- Actual TestBackend cell review (example cached model sizes), no
  raw digest or exact byte count in normal list/detail, explicit
  Details reveals full digest. Strict feature lint and optimized Rust
  87/0/3. Native mouse split/scroll/Details, action controls, editor,
  shell/settings/appearance and actual controlled download passed. Exact default CPU/model/text/
  TUI/Web release and all native PTY invocations also passed (target/
  ux-p4a-minimal-default.log).
  PTY fixture geometry and differential assertions adjusted (including
  private Web URL fresh paint without logging credential).
  Actual cached-owner combined reruns reached browser resident success
  twice; native Details, inspect, verify and alias worked but fixture
  proceeded before settled receipt/inventory (assertions corrected,
  guards retained). Last combined rerun timed out at browser model
  load. Native-only retry explicitly refused original 8 GiB envelope:
  7,538,163,712 bytes after headroom available (7188 MiB). No lowering
  admission or touching unrelated host jobs. Final cached-owner rerun
  pending healthy host, P4A2/P4B; no new full acceptance claim.
  Logs target/ux-p4a-minimal-final.log, -pty2.log, -pty3.log,
  -owners2.log, -resident-final.log, -resident-rerun.log, -native-owner.log
  (all under target/ux-p4a-minimal prefix).
  No new dependency, runtime authority or model download.
- Remaining UX-P4A2 stable item/log/output anchors, distinct Running
  selected-detail/response, safe log details, compact modal/keyboard
  focus and editor drag edges; P4B simultaneous-owner/full isolated
  acceptance. No public serving or overall UX-P exit yet.

### Previous checkpoint — UX-P3B settings/filter and editor interaction (26 Nov; 8a2c2ae8)

- New choice.rs finite setting and log picker; initialized from the
  actual observed effective choice. Edit/e uses typed picker, Enter
  retains direct YAML value editing, F2 edits chosen value as YAML,
  nonlisted/custom values retain lossless original form. Exact
  ConfigFile snapshot and existing Submit/Set owner; Apply explicit.
  Stale choice refuses, draft retained via existing retry path.
  Local Logs l/f pickers, uppercase L/F original cycles.
- Rows now retains original inventory and visible indices; local
  per-shelf search, no network/acquisition or second catalog. Selection
  only visible exact keys, refresh reconciles, clearing restores
  inventory. Original config observation/filter ownership preserved.
- editor.rs one draft caret/anchor/visual viewport; full grapheme
  boundary handling (wide/combining/CRLF), rendered row-range hit
  translation, mouse click/drag, Shift+arrows, selection replacement
  and bounded paste, wheel/drag scrollbar/PageUp/Down. Display sanitizes
  controls, tab has visible four-cell representation; raw bytes remain
  unmodified except explicit edit. Resize preserves visual byte anchor
  for manual scroll; actual cursor hidden off viewport. Shared
  Viewport usize offsets support >65k editor rows; ordinary Paragraph
  u16 bound explicit. Existing locked unicode-segmentation 1.13.3 and
  unicode-width 0.2.2 become direct optional TUI deps only; lock edges
  changed, no package/version/download. No clipboard or new authority.
- Strict feature lint, optimized Rust 86/0/3 (58/0/3 CLI units), actual
  typed picker Apply/Cancel, stale finite draft refusal, log/mouse local
  model search, long raw scroll, wide+combining drag/caret exact saved
  bytes; original controls/mouse/shell/settings/appearance PTY pass.
  Actual cached Qwen CPU/browser/native/HTTP 1/0/0 (55.96s, five filtered)
  with original admission and exact output/stale identities/shared
  settings retained. Actual controlled transfer mouse pause/resume/
  durable/cancel and default CPU/model/text/TUI/Web release plus all
  native PTY fixtures passed. New editor fixture integrated in full
  default release gate; target/ux-p3b-default-release.log. Initial fixture assertions corrected for differential
  rendering and the expected ERROR log after deliberate stale refusal.
  Logs target/ux-p3b-final.log and target/ux-p3b-final-rerun.log.
- **Historical P4A scope (P4A1 minimalism landed above):** stable item/log/output
  scroll anchors, distinct Running selected detail/response affordance,
  safe log-row detail, small/modal/keyboard focus and editor drag edge
  cases; finish remaining view actions. P4B final real simultaneous
  resident/transfer/config and full isolated gate (separate coherent
  turn if needed). No accepted UX-P exit or public serving yet.
  Native Auto probe remains deferred; no performance/GPU claims.

### Previous checkpoint — UX-P3A native action controls (26 Nov; 5da96251)

- New presentation-only controls.rs: all nine views have width-stable
  action strips, F4/More overflow with disabled reasons. One Rc scope
  per rendered surface captures exact row/ready owner/observed file,
  config revision/active document, listener generation, transfer and
  active action identities plus availability. Press/release matching,
  fresh live snapshot and shared slot revalidation before original
  Action dispatch; conservative refusal on observation change. No
  new runtime/controller/authority or dependency.
- Modal Apply/Send/Save/Confirm/Cancel/Close and prompt chat/raw buttons
  converge on existing keyboard submit. Confirm preserves exact Request,
  disabled buttons have no hit, stale press refuses, busy submit retains
  form. Menu scroll is local. Keyboard shortcuts unchanged, F4 opens
  menu, F9 output follow. No auto execution on selecting any row.
- Strict minimal/Web-only/TUI-Web/combined lint and optimized Rust
  82/0/3. Real PTY controls (create, theme preview/cancel/save, overflow
  activate, remove cancel/confirm, release mismatch, concurrent Web edit,
  narrow keyboard/mouse menu), original P2 mouse/shell/settings/appearance
  pass. New tests integrated in final default-release gate; exact
  default CPU/model/text/TUI/Web build and all four native PTY
  invocations passed without serve/acquisition.
- Actual controlled download safe pause/resume/durable/cancel via native
  mouse; actual approved cached Qwen 0.8B CPU/browser/native/HTTP fixture
  1/0/0 (52.81s, five filtered), original 8 GiB admission, mouse device
  policy/cache inspect/verify/alias/removal decline/run/prompt submit/
  listener start/stop and retire. Exact stale Web identities and shared
  active policy checks remain intact. Logs target/ux-p3a-final.log,
  target/ux-p3a-pty-rerun2.log, target/ux-p3a-owners.log.
- Initial PTY caught variable disabled width; fixed slots. Remaining
  initial assertion failures were ANSI differential output; waited on
  actual selected/control labels with exact file/outcome assertions.
  Final focused tests pass; full workspace acceptance at P4.

UX-P3B landed above. Continue final UX-P4A/B before public serving.

### Previous checkpoint — UX-P2 mouse and pane foundation (26 Nov; ab3cf974)

- Scoped Crossterm capture (including quit/error/signal restore), typed
  ui.mouse auto/always/never; F8 session-only override.
- Presentation-only layout/input modules; one set of visible render
  hit regions, release match and original exact Row Key/ready Owner
  revalidation. Click rail/rows/shelf, theme preview, palette/help and
  pane focus. Modal masking; no effectful mouse confirmation/submit yet.
- Preferred rail width and both orientation ratios, per-view focus,
  maximize/restore, narrow nav and detail drill-in, separate visual-row
  scroll offsets and draggable scrollbar. Running/response secondary
  viewport; log wheel off tail pauses original ring view. Existing pinned
  Ratatui line-count feature; no new dependency.
- Final isolated focused Rust 84/0/3, strict minimal/Web-only/TUI-Web/
  combined lint; real SGR PTY wheel/scrolled exact-row, both split drags,
  maximize/restore, raw-YAML independent detail scrollbar/selection,
  modal masking, opt-out/toggle, terminal restoration and original
  shell/appearance/settings/Firefox regressions all pass. Corrected
  initial ANSI differential fixture assertions. Logs: target/ux-p2-final.log.
- Full gate/real cached resident and transfer at assembled UX-P4.
  Stable anchors across data reorder, full modal/form/editor scrolling,
  caret/selection and full mouse action buttons/confirmations remain
  P3/P4. Native Auto theme probe remains deferred as before.

UX-P3A action controls landed above; remaining UX-P3B editor/pickers,
then UX-P4 acceptance.

### Previous checkpoint — UX-P1 visual foundation (26 Nov; b1073e7c)

Typed ui.theme Auto/Dark/Light, native T shortcut → Config appearance
picker (preview/explicit apply/cancel), semantic color roles, restrained
pane rules and header, 24-bit/256/16/no-color, coordinated Web tokens and
active theme observation. No execution/control owner or dependency change.
Web Auto follows matchMedia, native Auto captures COLORFGBG else dark;
active OSC probe deferred. Focused Rust 81/0/3; minimal/Web-only/combined
strict lint, JS 3/0, actual native PTY appearance/settings/shell and Firefox
settings, mobile/tablet/desktop passed. Stale native draft vs newer Web
write refused, inactive profile isolation and persisted reopen proven.
Initial browser header overflow fixed with compact mobile glyph; final
PTY fixture waits for actual selected row instead of stale repaint.
Full gate and real resident/transfer regression at UX-P4.

The mouse/layout foundation landed in UX-P2 above. Continue UX-P3/P4.

### Original UX-P priority decision — native TUI polish (26 Nov)

User decision: before public serving, complete mouse support throughout
the TUI (click pane/row/tab/action/setting, drag splitters, independent
scroll), elegant clean colors and Dark/Light/Auto theme selection.
[Full UX-P specification](../plans/cli-ux/TUI_POLISH_AND_MOUSE_INTERACTION.md)
defines current-source audit, inspected Whisrs v0.2.1 design, semantic
palette, appearance/Auto and terminal depth policy, responsive geometry,
render-derived hit targets, stable typed identities, all nine views and
ordered UX-P1–4 checkpoints. UX-P1 is focused-qualified above; overall
UX-P acceptance pending. Existing UX-1–10 remains accepted.

Read the UX-P specification and live CLI/device TODO, continue
UX-P2–4 with one coherent validated checkpoint per turn.
No new runtime, cache, configuration or execution authority; mouse
selection is inert, explicit actions use original owners. Keep both
keyboard and mouse pathways. Optional local Auto background probe
dependency requires separate approval/offline review; no new dependency
or download at design checkpoint. Resume public serving only after UX-P
acceptance.

### Previous checkpoint — Phase 2.1 UX-10 acceptance (26 Nov; 1b8de1cb)

UX-10 and the bounded CLI/TUI/Web iteration are accepted. Cold diagnostics
now include original backend loader probes and original cache accounting.

- CUDA driver presence uses the existing cudarc loader check; HIP
  runtime/RTC uses existing CubeCL HIP loader; Vulkan uses the same ash
  entry loading as wgpu-hal. Optional uncompiled probes are unknown.
  Library loadability, observed physical binding, source/ABI/compiler
  qualification and execution admission are distinct. No explicit new
  client/context/model creation; no new dependency.
- CacheMetadata::observe_usage obtains only existing publication/attempt
  locks with zero wait and read-only descriptors, reuses original
  cooperative inode-charge/live reservation arithmetic and persistent
  policy, and never creates lock files, directories or reservations.
  Busy publication is unknown, not zero usage. Charged + reserved vs
  configured quota are not filesystem free bytes; logical manifest totals
  can share blobs. Existing cache tree remains unchanged.
- Strict minimal, cache, combined lint and focused 44/0/2 (25 cache
  transaction, 14 CLI and five optimized real synthetic GGUF/source-form
  tests). Equality with original live/dead reservation accounting,
  contended publication and missing-lock refusal pass.
- Complete isolated gate sequence passed 2,505/0/109 (318 Rust suites), release Web/TUI/installer/scripts. Original run stopped at two transient busy-observation fixture assertions in acquisition/cache transactions; test-only bounded busy retry, reran affected cache-only/diagnostic/acquisition suites and exact remaining gate commands. No production change after the initial run.
  The parallel fixture can observe a transient nonblocking refusal
  (e.g. inherited lock description or interrupted probe); precise cause
  not established. Test waits only for observation; explicit held-owner
  refusal stays unchanged. No live acquisition, new dependency, physical
  GPU execution qualification or performance claim.

**Historical next after UX-10:** public OpenAI-compatible serving (now
preceded by UX-P above), starting with
[the serving iteration](../plans/serving-and-speculation/NEXT_ITERATION_OPENAI_COMPATIBLE_SERVING.md)
and its live owner TODO, then ordinary public benchmarks, ROCm vs
optimized vLLM, CUDA, Vulkan, optional compiled providers per full plan.
Keep the accepted CLI/config/runtime/control owners. Graph inspection
remains direct text/JSON; general graph/media execution and advanced
provider transfer modes remain explicitly unsupported or deferred.

### Previous checkpoint — Phase 2.1 UX-10 core (26 Nov; 22c6c0a7)

Core direct CLI implementation and focused verification complete.
- Doctor uses existing cold selected-document resolution/validation
  (shared SDK inspect_policy), device inventory and cache metadata.
  Safe text/JSON status; explicit errors return 2 with report. Missing local
  sources and incompatible selected device policy warn. Cache logical bytes
  vs configured limit are not physical quota usage; no unobserved runtime
  binding is claimed available.
- Device inspect with optional source uses exact observed physical ID and
  existing run dry-run planner. No acquisition, loading, audit write or new
  reservation. Fresh positional model replaces configured entrypoint and
  model table, retaining policy; legacy explicit override remains.
- Pipeline list/inspect/check, selected profiles/config and positional
  files; original typed topology, links, declared routes/context, text/JSON.
  General pipeline run refuses. Help/completions reflect normal commands;
  compatibility aliases preserved. docs/CLI.md and examples updated.
- Final strict minimal/combined lint and isolated focused 59/0/1
  (minimal 27/0/0; optimized combined 32/0/1). Actual synthetic GGUF
  CPU execution and source forms; existing populated cache tree unchanged.
  One explicit real cached serving fixture skipped in focused run.
  Initial obsolete help expectation/document-prefixed help and conditional
  unused-import failures corrected before final rerun.

The full isolated gate passed formatting, architecture/dependency checks,
HIP-only (1m53s) and Vulkan-only (2m08s) compilation. Intentionally
stopped during production CUDA native transport rebuild to close this
checkpoint; no full gate acceptance or physical qualification claimed.

The remaining cold diagnostics and comprehensive gate are accepted in the
current checkpoint above. Public serving follows.

### Previous checkpoint — Phase 2.1 UX-9E acceptance (26 Nov; 1a878990)

UX-9D and UX-9E accepted. Same foreground runtime, one action slot,
original ConfigFile/cache/transfer/listener owners. No new dependencies,
model execution paths, admission changes or performance claims.

- Native Logs redraw follows the shared bounded ring revision while visible;
  a fast Web-only validation cannot disappear between active-action samples.
  Existing event cursor and per-client filters remain unchanged.
- Actual native PTY and attached Firefox share one original protected
  transfer (existing deterministic fixture, no provider access): native
  resume then safe pause after an accepted prefix, browser observes that
  checkpoint and resumes to durable publication or cancels to a preserved
  partial. Exact operation identity, stale control refusal and original
  range reopen; closing browser leaves native host alive, native quit
  joins host/worker. Same test child and original transfer owner.
- Actual approved cached Qwen 0.8B GGUF optimized lifecycle: original
  browser cache/device and resident/HTTP checks; native edit of active
  run.task.context 64 -> 128 with live admitted model/listener, stale Web
  draft refused, fresh Web restore to 64 seen in native editor; admitted
  scheduler capacity, registration, generation and exact HTTP text
  unchanged. Original 8 GiB host admission retained. Native/Web logs
  share safe success/failure correlation, no private prompts or drafts.
  Existing stale registration/generation, cache retirement, cancellation
  and joined shutdown checks pass.
- Combined release CLI unit tests 50/0/3; controlled shared Firefox/PTY
  durable and cancelled scenarios pass; real cached combined integration
  1/0/0 (54.16s, five filtered); idle native log repaint after Web
  validation passed without resize/navigation after command. Initial
  settings fixture assertion corrected for ANSI cursor-separated spaces;
  no production behavior relaxed.
- The interrupted UX-9D gate was rerun across the assembled D/E source
  (UI change before CPU compilation). HIP/Vulkan/CUDA feature boundary,
  strict CPU/optional feature checks, all 318 Rust suites 2,491/0/109,
  native/Web release, offline installer and scripts passed. No physical
  GPU or public serving qualification inferred.

The following UX-10 core implementation is recorded in the current checkpoint above.

### Previous checkpoint — Phase 2.1 UX-9D (26 Nov; 543194ac)

Profiles / Config / Logs are implemented for both foreground clients;
focused and comprehensive acceptance passed with the UX-9E gate above.

- The original native configuration actions and typed observations now live
  in control_host/settings.rs. One ConfigFile snapshot/validation/atomic
  writer, one shared action slot and process-local active Selection. Exact
  stale writes refuse; activation only affects future actions. Native raw
  repair remains available; inactive edits invalidate both clients without
  reconfiguring admitted owners.
- Web profiles/config authenticated cold reads retain one latest exact
  snapshot with an opaque identity; reading a different document can require
  refresh. Typed safe policy fields and normalized YAML for wholly safe
  documents only. No paths, prompts, graphs, credentials or comments on the
  wire; protected fields are not writable by Web. Trusted existing profiles
  can be activated/copied without disclosure. Default, portable and active
  removal refused by original owner. Native-only unrestricted raw repair.
- Shared host-owned 128-entry operational ring in control_host/logs.rs,
  safe action/owner correlations; one event collector per host. Web/native
  view-local filtering and pause/follow; no prompts/model output or durable
  history. SSE configuration/log invalidation; bounded retries only for
  refused cold reads, never mutations. Completed Web settings receipts
  reread configuration before enabling the next edit.
- Strict minimal TUI/Web, Web-only CPU/serve/acquisition and combined
  Clippy pass; Web-only tree terminal-free; JS 3/0; actual isolated HTTP,
  native PTY plus attached Web cross-client edit/activation/logs, and real
  Firefox profile/config/log and responsive smoke pass. Fixed fixture SSE
  response closure and a real cold-read navigation/receipt refresh race.
  No live acquisition, new dependency, model or GPU/performance change.
- The initial full gate stopped intentionally at the turn boundary during
  Vulkan feature-boundary compilation, not a reported failure. The
  complete private-network scripts/check.sh rerun in UX-9E above passed
  2,491/0/109 (318 suites), release Web/TUI/installer/scripts; UX-9D accepted.
  The original UX-9C evidence remains valid.

Historical next UX-9E completed above; UX-10 follows.

### Previous checkpoint — Phase 2.1 UX-9C (26 Nov; a6744b12)

Models / Downloads / Devices now use the original cache, transfer and
resident device authorities from either client. No new scheduler,
runtime, cache registry, transfer queue, persistent config or filesystem
authority. One shared Actions slot; exact download commands bypass only
the action slot and address the original active operation.

- control_host/inventory.rs shares original XDG cache observations and
  native/Web invalidation predicates. Cold authenticated inventory GET
  /zg/control/inventory/{models,devices}, bounded one Web read at a time;
  models expose exact cached manifest metadata, static target/catalog
  recipe metadata and original partial checkpoints. Device inventory uses
  ControlPlane::devices, not execution eligibility. Safe remote labels
  only; no arbitrary local source paths/raw errors. Partial keys and
  listing are observations, not verification or admission.
- Shared worker promotes inspect/verify, exact-digest dry-run/apply
  reclamation, additive by-name aliases, original CheckFit and load.
  Busy/Referenced/Incomplete remain failures with accurate safe messages;
  incomplete effects are not reported rolled back. Actual removal
  revalidates original cache activity, references and readers. Resident
  materialization may release cache readers: a loaded registration alone
  is not a cache lock. Typed Web grammar rejects arbitrary paths,
  noncanonical digest/device IDs, missing consent, credentials and
  unknown fields. Explicit source/device actions require exact canonical
  cpu:0 or pci identity, not an ordinal or backend override.
- Original interactive pull Intent now shared by Web/native; pinned
  HF/GitHub full commits with exact files or explicit GGUF quant, or
  approved Qwen recipe. Preview does not acquire, confirm must apply
  to the captured intent, offline policy checked before admission and
  in executor. Original quota/recipe expected snapshot/source/transfer
  admission and publication unchanged. No implicit pull, auto-load,
  eviction or quota mutation. Download pause/resume/cancel uses exact
  decimal operation ID, not action ID; accepted command is not terminal.
- SDK DownloadChanged carries observed status at emission so a replaced
  operation cannot hide the earlier terminal invalidation. Native and
  Web observe relevant cache/terminal transfer and load/unload events;
  Web SSE models/devices invalidation or resync, visible shelf refresh,
  one coalesced cold request, no progress/token-triggered cache scans.
  Native private receipts, independent control/inference authentication,
  exact action identities and foreground joined retirement preserved.
- Solid cached/catalog/partial searchable shelves, inspect/verify,
  alias, load/run/serve, removal preview/confirm, pinned pull draft and
  consent, live transfer phase/bytes/checkpoint/rate/ETA and controls,
  physical-device source check/explicit load; settings/logs remain UX-9D.
  Browser selection is client-local and never changes persistent policy.
  Cancelled previews cannot authorize later effects.
- Strict TUI-only/Web-only/Web-only CPU+acquisition/combined lint,
  terminal-free Web dependency graph, JS 3/0, actual cold HTTP/Firefox,
  controlled HTTP transfer (range resume/durable and cancel/partial,
  stale IDs, referenced retirement refusal), native transfer PTY and
  real Firefox pause/range-resume/publication passed. Fixture corrections:
  resident registration is not an artificial cache lock; PTY pause
  waits for a prefix; browser waits for actual empty route phase after
  retirement; PTY count accepts ANSI cell positioning and confirms the
  still-active Web host quit. Final real cached browser/TUI/Web/HTTP
  lifecycle passes with the unchanged 8 GiB envelope. No changed admission envelope or live remote acquisition.

UX-9C optimized cached integration passed 1/0/0 (48.13s, five filtered); full isolated gate 2,489/0/109 (318 Rust suites), release Web/TUI/installer/scripts passed. UX-9C accepted.
No new GPU or performance claim.

**Historical next (implemented in UX-9D above): Web Profiles / Config / Logs.** Promote original
ConfigFile typed/raw atomic snapshot/validation/edit, profile copy/
protected remove/activation and bounded operational observations to the
same authenticated control seam. No second configuration store,
arbitrary browser path/credential authority or invented durable logs.
Then UX-9E simultaneous native/Web mutation acceptance, UX-10 and
public serving; the ordered performance roadmap is unchanged.
~~~~


### D.17 — current baseline delta through UX-9C

Verbatim excerpt. Complete CURRENT_BASELINE.md retains all earlier exact conditions and measurement history.

~~~~markdown
# Current performance baseline — 22 Nov 2026

This is the active performance baseline for ZeroGravity work after 2a595be.
It replaces older benchmark/evidence chronology as engineering authority. Git
history preserves prior experiments; do not carry old benchmark artifacts or
historical comparison numbers into new implementation decisions.

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
~~~~


## Appendix E — complete prior handoff main body and historical chain

The previous comprehensive UX-9B handoff is retained intact at its original path. Its Appendices A–E preserve the engineering inventory, pending scopes, full prior ledger, every UX-1–9B implementation and entire preceding Phase 1 main body. Earlier comprehensive handoffs retain the precise original designs, foundation chronology, experimental data, rejected choices, reusable primitives and all previously accepted boundaries. These are historical evidence, not new work instructions.

- [.agents/docs/FULL_COMPREHENSIVE_HANDOFF_2026-11-26_UX9B_EXIT.md](FULL_COMPREHENSIVE_HANDOFF_2026-11-26_UX9B_EXIT.md)
- [.agents/docs/FULL_COMPREHENSIVE_HANDOFF_2026-11-25_PHASE1_EXIT.md](FULL_COMPREHENSIVE_HANDOFF_2026-11-25_PHASE1_EXIT.md)
- [.agents/docs/FULL_COMPREHENSIVE_HANDOFF_2026-11-25_FOUNDATIONS.md](FULL_COMPREHENSIVE_HANDOFF_2026-11-25_FOUNDATIONS.md)
- [.agents/docs/FULL_COMPREHENSIVE_HANDOFF_2026-11-24_FOUNDATIONS.md](FULL_COMPREHENSIVE_HANDOFF_2026-11-24_FOUNDATIONS.md)
- [.agents/docs/FULL_COMPREHENSIVE_HANDOFF_2026-11-24.md](FULL_COMPREHENSIVE_HANDOFF_2026-11-24.md)
- [.agents/docs/FULL_COMPREHENSIVE_HANDOFF_2026-11-23.md](FULL_COMPREHENSIVE_HANDOFF_2026-11-23.md)
- [.agents/docs/FULL_COMPREHENSIVE_HANDOFF_2026-11-22.md](FULL_COMPREHENSIVE_HANDOFF_2026-11-22.md)
- [.agents/docs/FULL_COMPREHENSIVE_HANDOFF_2026-11-21.md](FULL_COMPREHENSIVE_HANDOFF_2026-11-21.md)
- [.agents/docs/FULL_COMPREHENSIVE_HANDOFF_2026-11-20.md](FULL_COMPREHENSIVE_HANDOFF_2026-11-20.md)
- [.agents/docs/FULL_COMPREHENSIVE_HANDOFF_2026-11-18.md](FULL_COMPREHENSIVE_HANDOFF_2026-11-18.md)
- [.agents/docs/FULL_COMPREHENSIVE_HANDOFF_2026-11-15.md](FULL_COMPREHENSIVE_HANDOFF_2026-11-15.md)
- [.agents/docs/FULL_COMPREHENSIVE_HANDOFF_2026-11-14.md](FULL_COMPREHENSIVE_HANDOFF_2026-11-14.md)

### E.1 — exact previous UX-9B main body (historical, verbatim)

The original Appendices A–E are **not duplicated** here; read them in the linked complete prior file. The full original main body below is reproduced verbatim, including its obsolete UX-9C starting prompt. The current starting prompt is §13, not this excerpt. Relative links in historical text resolve from .agents/docs.

~~~~markdown
# ZeroGravity — full comprehensive handoff after UX-9B

Date: 26 Nov 2026 (Singapore).
Repository: /home/pvl/spaces/abird/src/zerogravity.
Latest accepted implementation: 4f23983a5bb33a3ab2e9ef124342a0cc33df2088 — feat(web): share resident model and listener actions.
Accepted bounded Phase 1 exit: d5030ad4ce476b913a46056d492992ccf1b8edf7.
Previous complete takeover: [25 Nov Phase 1 exit](FULL_COMPREHENSIVE_HANDOFF_2026-11-25_PHASE1_EXIT.md), 450cea5530dad567ff7706c30e1a1872ff01bda7.

**Read §§0–13 first. Resume at §11 (UX-9C).** This is a requested
one-time comprehensive takeover, not another live tracking/acceptance
system. Live source, executed results, ordered full plan and owning TODOs
remain authoritative. If implementation has advanced, revalidate it.

- Appendix A: complete engineering document inventory and navigation.
- Appendix B: literal unresolved checklist snapshot, with exact source
  lines and indented continuation; not a second live queue.
- Appendix C: all 895 reachable commits from the beginning through UX-9B,
  plus exact fourteen-commit Phase 2 delta.
- Appendix D: exact verbatim landing narratives and changed paths for
  all fourteen Phase 2 commits, and current Phase 2 baseline excerpt.
- Appendix E: entire prior Phase 1 comprehensive main body, verbatim,
  with all older appendices and handoffs retained at original locations.
  These preserve original design, detailed foundation chronology, exact
  experiments, rejected approaches, performance and operational recipes.

No historical specification, work item, source evidence or handoff is
deleted. Do not regenerate inventories, per-helper receipts or ledgers
during normal implementation turns. The new handoff commit is deliberately
not self-referenced; find its SHA via Git.

## 0. Recovered implementation and current position

Live main has advanced beyond UX-9A: UX-9B is already accepted. Do not
repeat it. Phase 1 minimum common foundation accepted; UX-1–7, native
UX-8A–D and embedded Web UX-9A/B accepted. Next UX-9C, then UX-9D/E,
UX-10 and public serving.

Latest recorded full gate: **2,489 Rust executions passed / 0 failed /
109 explicit skips**, release Web/TUI, installer and scripts passed.
Latest actual changed-consumer regression: optimized cached Qwen 0.8B GGUF
browser/TUI/Web/HTTP lifecycle passed 1/0/0 (46.14s, five filtered);
the other five combined release tests had already passed. Original 8 GiB
host admission unchanged. This extends frontend/product qualification,
not physical GPU, remote acquisition, public exposure, numerical or
throughput qualification. No full implementation gate rerun for this
documentation-only task; rely on existing accepted recorded evidence.

### 0.1 Repository and resource authority

- L5 (pvl-l5) is sole authoring host, main exact above, clean staged/working
  tree and one worktree. Local origin/main 668a3050d5205997f17332af4b2c46cd39908aff (ahead 100,
  behind 0 before this documentation commit). No push or remote fetch.
- An unrelated AW source-checkpoint test was running on the shared bridge.
  It is not ours; no cancellation, acknowledgement or resource cleanup.
- Observed L5 RTX 3060 Laptop GPU 6,144 MiB, 37 MiB used, no listed compute
  apps; ~11 GiB available host memory and ~43 GiB free disk (96% used).
  These are transient observations, not admission guarantees.
- X2 and A1 execution/testing only. Not revalidated in this handoff; the
  prior exact Phase 1 L5/X2 synchronization is historical, not current
  HEAD parity. Revalidate before use. Sync exact L5 commits via verified
  incremental bundles and clean fast-forward only.
- Main is the only persistent branch and checkout. Preserve unrelated
  work, signed Git, and all shared physical owners. No unapproved
  downloads, resets, destructive Git, privilege changes or push.

## 1. Bootstrap sequence

1. [AGENTS.md](../../AGENTS.md).
2. [Planning rulebook](../plans/README.md).
3. This file §§0–13 first; Appendix D exact Phase 2 records and Appendix E /
   [Phase 1 exit pack](FULL_COMPREHENSIVE_HANDOFF_2026-11-25_PHASE1_EXIT.md)
   for original design, measurements and all foundation decisions.
4. [Full implementation plan](../plans/FULL_IMPLEMENTATION_PLAN.md).
5. [CURRENT_HANDOFF](CURRENT_HANDOFF.md), [CURRENT_BASELINE](CURRENT_BASELINE.md).
6. [Plan queue](../plans/TODO.md) and owning TODOs.
7. [Canonical CLI/UX design](../plans/cli-ux/NEXT_ITERATION_CLI_UX_AND_MODEL_LIFECYCLE.md)
   §§33–46, §46A (especially 46A.9 and 46A.13–29), §51.
8. [CLI/device-routing](../plans/cli-device-routing/TODO.md),
   [acquisition](../plans/model-acquisition/TODO.md),
   [serving](../plans/serving-and-speculation/TODO.md),
   [OpenAI serving iteration](../plans/serving-and-speculation/NEXT_ITERATION_OPENAI_COMPATIBLE_SERVING.md),
   [public SDK/serving](../plans/fabric-vnext/PUBLIC_LIB_AND_SERVING.md),
   [platform](../plans/platform-and-release/TODO.md),
   [structure/naming](../plans/pipeline-composition/STRUCTURE_AND_NAMING.md).
9. Actual source/exports/tests in §§5/11; original detailed contracts in
   §12 and historical packs before reopening any rejected experiment.

The standalone starting prompt is [here](NEXT_AGENT_2026-11-26_UX9C.md) and in §13.

## 2. Owner direction and working contract

The user's repeated instructions are durable engineering policy, not optional
style advice:

- Elegant, simple, clear, intuitive Rust. Favor zero-cost abstractions,
  compile-time specialization and direct hot-path implementation when clean.
  Where fast code would add convoluted maintenance, choose the simple design
  first and optimize the measured hot area selectively.
- ZeroGravity is a composable library and execution fabric, not a single Qwen
  benchmark or collection of per-backend/model engines. Attack the large
  architectural/computational chunks before small launch-count improvements.
- Reuse mature optimized mathematics where possible. Specialize model-specific
  regions when it actually wins. Do not recreate years of tuned BLAS work.
- Preserve a correct reusable model-neutral kernel, transport, layout,
  scheduling transform, provider or completion mechanism when one production
  selector loses. Reject the consumer/selection, not the primitive. Delete the
  primitive only if incorrect/unsafe, incompatible with its contract, or
  strictly superseded by a retained equivalent. Remove temporary probes,
  selectors, runners and experiment-only wiring.
- Preserve exact source, numerical, layout, state, ownership, resource,
  completion and cancellation semantics. Intentional changed arithmetic is
  a separate numerical profile, not a hidden approximation.
- No fixture/evidence/pinning bureaucracy. Test meaningful invariants, reuse
  still-applicable evidence, batch related work, run real source/model and
  matched-provider feedback as soon as implementation is ready. Do not create
  a new acceptance micro-phase per helper or rerun large models merely because
  a test expectation changed. Do not equate synthetic kernel speedup or fewer
  launches with a model-level win.
- Aim for one substantial coherent implementation checkpoint per user turn,
  approximately 30 minutes at most; validate, record pending work, sign and
  pause. Recover interrupted jobs from actual results before restarting.
- Keep signed commits, exact bundle transfer, sole L5 authoring authority,
  one main worktree and preservation of other agents' work. No push unless
  explicitly requested. No unapproved model/SDK acquisition, privilege
  changes, host recovery, GPU reset, or interference with unrelated owners.
- Use existing owner TODOs for live tracking. No duplicate task ledger; the
  snapshot appendices below exist solely for this requested handoff. The
  coordinator is the single writer of the shared plan queue.
- The root README describes the **final engineering architecture** decisively:
  implementation state, pending integrations and qualifications belong in
  its roadmap/status section and plans. AGENTS.md protects the user-authored
  README; do not edit it without explicit authorization. The approved
  architecture SVG is a genuine vector with dark UI-style cards and simple
  arrows at `docs/architecture-overview.svg`.

## 3. Original design, and the architecture we are preserving

The original [v4.3 master plan](../plans/ZEROGRAVITY_MASTER_PLAN_2026-10-10.md)
is the source for the product argument: build a practical modular Rust-owned
AI execution runtime, inference-first but model/task-general, with maximum
reuse across math, runtime, devices and memory tiers. The initial October
design, critical review, publication-facing specifications and later Fabric
transition are linked in Appendix A. The first reachable Git commit is the
13 Oct design commit, whose design dates reach back to 10 Oct. Earlier drafts
retained in those files are not new unqualified implementation promises.

### 3.1 One execution fabric, four distinct concerns

![ZeroGravity execution architecture](../../docs/architecture-overview.svg)

| Concern | Owns |
| --- | --- |
| Execution fabric | Model lifecycle, catalog, admission, placement, scheduling, batching, pipelines, persistent state, publication and serving |
| Model semantics and planning | Authenticated source-bound mathematical/task definitions, typed regions, shapes, effects and numerical policy |
| Execution implementations | Native composed operations, mature vendor libraries, specialized kernels, captured command graphs or compiled closed regions/whole programs |
| Device backends | CPU, CUDA, HIP/ROCm, Vulkan and future devices: physical storage, command submission, readiness and completion |

Mature operator libraries are not graph compilers, and neither is a hardware
backend. Native CUDA can use cuBLAS/Lt, CUTLASS, cuDNN, FlashAttention and
FlashInfer; native HIP can use rocBLAS, hipBLASLt, CK, MIOpen and suitable
specialized kernels. Vulkan has no required universal BLAS library: evaluate
mature cooperative-matrix shaders (including ggml kernels with compatible
license/provenance) and actual device capabilities. IREE is a later compiled
execution candidate, not a condition for native Vulkan execution. CPU uses
existing Rust/reference and appropriate optimized library paths.

### 3.2 Source ingestion, semantics and executable lowering are orthogonal

The primary source formats are original Hugging Face configuration,
tokenizer/processor and indexed Safetensors checkpoint weights, plus GGUF
metadata, codecs and weight tensors. Transformers describes an architecture/
software distribution ecosystem, not a tensor storage format. Preserve
original learned roles, exact tensor axes, source identity and quantization
semantics. No implicit conversion, remote model code execution or mandatory
Python dependency. Recognizing a format is not mathematical model support.

Keep **two enduring execution representations**:

1. The authoritative semantic computation (ModelProgram/ForwardPlan, typed
   mathematical and task regions and state effects).
2. The prepared physical executable and its original source/resource owners.

Source artifacts are inputs, not a third IR. Provider-local graphs and
compiler views are internal lowering artifacts, not alternative canonical
model representations. Source-specific loaders authenticate and bind source
owners; the same semantic plan is consumed by native or compiled preparation.

Closed-region direct lowering is independent of optional external graph import.
MIGraphX and TensorRT should preferentially construct their own programs/networks
from the verified native regions through their native APIs, not require ONNX
export/import or another generic compiler framework. ONNX import is an independent
optional path, including ONNX Runtime, with verified graph/external-weight
identity, layout, precision, state and effect contracts. IREE can consume
appropriate lowered computations. All concrete integrations remain Phase 6.
Foundations establish only the useful preparation/binding/ownership seam.
A provider's supported shapes, workspace, interop, precision, resource needs
and completion must be real; compilation/export alone grants neither quality,
performance nor execution support.

### 3.3 Region composition is not a second runtime

Native ModelAssembly, closed mathematical regions, and multi-model TaskPipeline
are different concerns. A recursive semantic/pipeline description permits
reuse and replacement; it does not create a recursive runtime/allocator/
scheduler. Use the existing source plan and region/provider boundary; make
backend-specific lowering and capability/refusal explicit. Do not invent a
parallel registry, second canonical model IR, second allocator or universal
GEMM abstraction for non-dense operators.

Choose an implementation during cold preparation or a qualified committed-root
switching boundary. Shape/layout/capability misses before enqueue can use an
already-qualified decomposition or refuse. Submitted execution failures cannot
be retried by secretly changing provider against the same root.

### 3.4 Non-negotiable state, physical ownership and correctness

- One logical StateRoot publisher per session. Device work produces private
  candidates and readiness, not logical commits. Publish all required KV,
  recurrent/conv and other state domains at one compatible frontier before
  user-visible output. Cancellation or failure cannot leak provisional output
  or partially committed state. Completion uncertainty fences/quarantines
  affected physical owners; only actual safe completion retires them.
- Original source owner, backing/range identity, leases and provenance remain
  the authority. Identity is not merely a hash or a reconstructed handle.
  Shared immutable readers and aliases retain their real owners; mutable
  independent session state stays private. Cache reuse requires complete-domain
  identity/position/profile compatibility. Preserve actual COW/page generation,
  event dependency, original read/write validation and retirement semantics.
- One ResourceArbiter owns physical budget/admission. Device-local allocation
  and native provider workspaces may be optimized inside that budget.
  Explicitly account for persistent context/module/provider allocations,
  internal workspace and external synchronization/transfers. No implicit
  reclamation; UMA host/VRAM overlap and physical headroom are real.
- Backend adapters own physical submission/completion. Borrowed tensor views
  and actual external stream/buffer interop are conditional capabilities.
  A library's internal region temporaries may be admitted; it cannot own the
  global scheduler, session roots or model placement.
- Separate reference arithmetic, native numerical profiles, quality grants,
  source/model compatibility, physical readiness, and serving/benchmark
  acceptance. A native profile may use the correct trained mathematics with
  a different reduction tree, rounding or approximation only when the changed
  contract is explicit and qualified. Exact CPU-reference paths remain intact.
- One fair scheduler/worker/cohort lifecycle, shared prefill, independent
  continuation, paged/hybrid state, device-resident feedback, speculation/MTP,
  placement and pipelines. No independent per-provider inference loop.
- Source/math effects, memory and commands stay Rust-owned where practical;
  use FFI and native vendor code without hiding source or resource ownership.
  Keep model/backend/serve features orthogonal; production cannot depend on
  test fixtures or finite captured qualification recipes.

Read the exact [invariants](INVARIANTS_AND_FAILURE_SEMANTICS.md),
[state contract](STATE_CACHE_AND_TRANSACTIONS.md),
[resource/command contract](ASYNC_RESOURCE_COMMAND_IR.md),
[execution spine](MINIMAL_EXECUTION_SPINE_AND_PROTOCOL_V4.md), and
[numerical policy](../plans/numerics-and-context/CORRECTNESS_POLICY.md) for
implementation rather than replacing them with this summary.

### 3.5 Typed error contract (now fully accepted by the workspace gate)

| Error | Meaning |
| --- | --- |
| Malformed | Invalid/corrupt external input or artifact |
| Unsupported | Requested capability or semantics absent |
| Limit | Configured/static bounds, arithmetic overflow, configured quotas |
| Resource(ResourceLimit) | Measured physical admission fails; operation, resource, exact requested/available bytes and scope |
| State | Invalid lifecycle/internal state |
| Io | Underlying I/O cause |

Do not parse error messages to choose scheduling or fallback. Physical pressure
is backpressure only at an explicitly owned prelaunch policy seam. Preserve
post-submission failure and quarantine. Public `ServiceError::inference`
maps Resource to 503/resource_exhausted, Limit to 400/limit_exceeded,
Malformed/Unsupported to 400 and Io/State to 500; HTTP server errors are
classified separately from invalid requests. Empty serving model configuration
now refuses before backend selection, while a valid model declaration for an
uncompiled backend still refuses as backend_not_compiled.

## 4. Ordered delivery and pending scopes

| Phase | State / objective |
| --- | --- |
| 1 Foundations | Minimum shared exit accepted for existing qualified local/native/portable consumers; broader specializations explicitly remain open. |
| 2 CLI/TUI/Web/public serving | Current UX-9C; finish Web management, simultaneous acceptance, UX-10 and public serving/benchmark. |
| 3 ROCm | Match or beat optimized matched vLLM first on AMD. Restore approved pinned comparator before fresh parity claim. |
| 4 CUDA | Then NVIDIA matched vLLM and remaining actual lane/context/quality qualification. |
| 5 Vulkan | Native whole-model feature/correctness/performance parity; suitable native comparators, no invented vLLM Vulkan. |
| 6 Optional compiled providers | Concrete MIGraphX, TensorRT, IREE, ONNX Runtime after native campaigns. Real source/region lowering, ownership and measured benefit. |

Canonical UX sequence: UX-1–7, UX-8A/B/C/D, UX-9A/B/**C**/D/E, UX-10.
Public serving follows. Broader unchecked outcomes in the live full plan
remain open despite accepted bounded checkpoints. Do not mark entire
owner plans complete merely because individual milestones are checked.

### 4.1 Accepted Phase 1 exit audit (historical, exact)

### Phase 1 bounded exit audit — 25 Nov

The minimum shared foundation is implemented for the existing qualified
local text/native and portable execution scope. See CURRENT_HANDOFF for
the final verification and qualification boundaries.

| Seam | Exit coverage |
| --- | --- |
| 1.1 Semantic composition | Original source-bound ModelProgram/ForwardPlan, typed closed-region/whole-text binding, original model/task/pipeline identity and explicit prepared selection. |
| 1.2 Operators/providers | Real native typed math/ports/state, managed private CUDA buffers/secondary queue, actual graph observation, bounded coherent Vulkan mapped backing; SDK-free compiled provider scaffold, explicit refusal. |
| 1.3 Source/lifecycle | Authenticated original HF and held local/verified cached GGUF, shared declared SDK/pipeline/CLI source eligibility, cache/source/alias/last-reader authority and original physical admission. |
| 1.4 State/memory/control | Original backing/range/lease, completion/quarantine, independent state and sole publication, paged/streamed/prefix/persistence and cohort/queue/task/pipeline paths, typed media/clock/placement contracts. |
| 1.5 SDK/verification | Actual public source-authenticated native SDK and real CLI local/cache runs, original runtime/catalog/control/poll and selected plan/region/capture facts, feature/architecture and full gate. |

The broader engineering work listed below remains the durable expansion
queue, not additional mandatory Phase 1 exit gates. No second catalog,
allocator, publisher or execution IR. General UMA/foreign/context
interop, non-Linux fit, general DAG/media/context, multi-source HF/mixed
backends, generalized persistence/async SSD and unqualified model/
device/codec/quality paths remain explicitly open. Native parity
requirements belong to their backend phases; concrete graph SDKs Phase 6.
The full gate and affected final physical/feature checks pass.
**Proceed to Phase 2.1 UX-1.**

### 4.2 Explicit unresolved boundaries at the foundation exit

Preserve general foreign/Existing buffers, external contexts/streams, multi-domain transfer, aggregate UMA heap/RAM accounting, CPU read of GPU-written mapped bytes, noncoherent visibility, non-Linux physical fit and OS-hard memory reservation as unsupported until qualified. Same PCI/addressability does not prove shared storage.

Other resident persistence layouts/providers, arbitrary cross-provider restore, scheduler-managed persistent sessions, authenticated/encrypted tenant store, disk quota/restore-versus-recompute and async SSD remain open. Local trusted CPU GGUF and bounded native HIP V2 contiguous F32 persistence are qualified, not general state migration or prompt-prefix authorization. Mixed backend/multi-source HF, generalized DAG/media/context and new model/codec/task eligibility are open.

Native learned Vulkan parity, deterministic CUDA lane/quality/active-context holds, ROCm parity, production capture selection and external graph compiler SDKs are later work. Preserve all reusable primitives; no general capability claim. The Renoir/gfx90c hold and approved source/comparator constraints remain explicit. These do not block the accepted minimum common foundation for the qualified consumers.

### 4.3 Exact current Phase 2 plan

Exact excerpt at accepted UX-9B (headings and relative links adapted to this file), preserving accepted and broader open
outcomes. This excerpt is a snapshot; live plan remains authoritative.

#### Phase 2 — CLI, TUI and Web

Make ZeroGravity one small, coherent tool: minimal by default and transparent on
demand. Preserve the existing detailed
[CLI/model lifecycle/terminal UX specification](../plans/cli-ux/NEXT_ITERATION_CLI_UX_AND_MODEL_LIFECYCLE.md)
(including UX-1 through UX-10 and §46A).

##### 2.1 Shared SDK, control plane and CLI

- [x] UX-1 shared presentation foundation accepted: terminal detection,
  rich/plain, color/NO_COLOR, quiet/verbose/progress, typed reporting and
  failures/hints, original machine stdout/stderr contracts. Full isolated
  gate 2431 pass/0 fail/106 skips, release PTY/device JSON and existing
  CPU run equivalence. No source/backend/runtime policy change.
  [CLI owning checkpoint](../plans/cli-device-routing/TODO.md).
- [x] UX-2 typed command tree + shared cold model/config resolver accepted:
  canonical hierarchy and compatibility dispatch, shared globals/help/static
  completion, exact digest and explicit by-name selector through original
  cache metadata/leases, one selected run document/profile and original
  overrides, model/device reads and explicit lifecycle forwarding.
  Full isolated gate 2451 pass/0 fail/106 skips; synthetic CPU equivalence
  and release completion/JSON smoke pass. No new runtime, acquisition,
  physical or numerical qualification.
- [x] UX-3 shared native ControlPlane: one facade control state, original
  runtime/cache/device/config/serving observations and typed actions,
  bounded transient event cursors with resync, exact registration/generation
  guards and original task output/cancellation. Canonical model/device CLI
  consumer; real approved Qwen 0.8B GGUF two-client lifecycle and loopback
  integration pass. Full gate 2457 pass/0 fail/108 skips (+1 focused
  model integration); no duplicate execution or transport owner.
- [x] UX-4 bounded canonical single-plan lifecycle: shared download
  observation/actions with exact operation IDs, original strong-validator
  checkpoint/range pause/resume and safe-boundary fallback, cancellation,
  cold partial inventory, interactive/plain/quiet/JSONL and actual receipt.
  Direct provider plans share progress; advanced forced modes and aggregate
  component controls remain open. Full isolated gate 2459 pass/0 fail/109
  skips; separate SDK acquisition integration and five PTY/JSONL fixture
  runs pass. No duplicate transfer owner or new provider grant.
- [x] UX-5 bounded canonical one-line run: shared durable name/approved
  catalog selection, exact provider/snapshot pin, positional/file/stdin
  prompts, original CPU/native Qwen owner retained across independent
  TTY requests, actual selected model/backend/device status. Explicit
  consented 2B/27B exact-source recipes, original shared acquisition
  operation and precommit digest guard. Legacy 0.8B manifest recreation
  and unsupported routes refuse. Synthetic CPU equality and acquisition
  guard, final combined lint/feature build pass; full gate 2465 Rust
  executions pass/0 fail/109 skips, release/installer/scripts pass; real
  Qwen 0.8B GGUF CPU PTY two-prompt one-shot equality and clean exit pass.
  No remote transfer or physical GPU requalification.
- [x] UX-6 stable canonical serve: positional/explicit aliases and shared
  run/source/config/device policy; typed serve settings and named profiles.
  CPU GGUF and single approved original HF snapshot alias sets, same
  eligible acquisition consent and original loader. Legacy serving profiles
  retained; incompatible route sets refuse cold. Original ControlPlane
  listener with actual startup, sampled status and joined shutdown. Real
  cached Qwen 0.8B GGUF release alias equality/one-owner/SIGTERM test and
  cold refusal pass; combined lint and GPU feature compile pass. No protocol
  rewrite or physical/remote acquisition evidence. Full isolated gate
  2466 pass/0 fail/109 skips; release/installer/scripts pass.
- [x] UX-7 unified configuration/profile management: existing v2 Document
  is ConfigDocument, one selected document with implicit default fallback
  and CLI precedence. Typed UI/offer preferences, cold validation/
  provenance, atomic ConfigFile replacement, profile copy/create/remove
  and temporary validated editor, basic inline managers. Focused SDK/CLI,
  PTY and combined lint/feature builds pass. Corrected full isolated gate
  2473 pass/0 fail/109 skips; release/installer/scripts pass. Serial
  combined release run_unified 6/0/0 including real cached GGUF serving
  with unchanged admission bound. No new source, network or execution authority.
- [x] UX-8A native terminal shell: bare capable TTY opens read-only native
  view over shared Runtime/ControlPlane and cold owner observations.
  Responsive navigation/palette, scoped terminal/signal lifecycle, optional
  CLI-only Ratatui/Crossterm (ordinary default/all and product releases).
  Actual PTY and cell rendering/reducer tests, strict lint pass; full
  isolated gate 2477 pass/0 fail/109 skips, release PTY and installer/
  scripts pass. No implicit execution or new backend qualification.
- [x] UX-8B native Home / Running / Serve: shared canonical resident route
  planning, original source/device/admission and listener policy, one
  runtime and owned action worker. Load/quick-run, exact registration
  invoke/unload, generation start/stop, bounded output and cooperative
  cancellation; confirmed foreground retirement. First actual cached
  GGUF TUI/HTTP lifecycle and combined release 6/0/0, strict lint and
  combined feature compile pass. Final repeat-invoke/paste real PTY/HTTP
  rerun and remaining release tests pass; full isolated gate passes
  (2,479 passed, zero failed, 109 ignored), release PTY/installer/scripts
  pass. UX-8B accepted.
- [x] UX-8C native Models / Downloads / Devices: exact cached digest
  actions, static model catalog/partial observations, non-replacing
  alias and original reclaim outcomes, physical device source-policy
  check/load, coalesced inventory reader off presentation loop. Same
  runtime original hub/recipe acquisition, explicit pinned file/quant
  consent, bare offline, exact operation pause/resume/cancel and observed
  progress. Reducers/intent, real controlled-transfer PTY, strict lint
  pass; actual cached lifecycle and remaining release tests pass; full
  isolated gate 2,481/0/109, release PTY/installer/scripts pass. UX-8C
  accepted.
- [x] UX-8D native Profiles / Config / Logs: shared ConfigFile atomic
  snapshot-based editor, typed search/provenance and raw repair,
  create/copy/activate/confirmed remove, all cold reads on original
  observation worker, bounded operational action/control-event log with
  search/level/component/follow. Profile activation is process-local
  for future actions, no new runtime or settings owner. Focused reducers,
  strict lint and actual settings PTY pass; serial cached release lifecycle/
  profile-switch passes (22.95s), remaining release tests pass. Full isolated
  gate 2,483/0/109, release PTY/installer/scripts pass. UX-8D accepted.
- [x] UX-9A embedded SolidJS/Vite Web shell: committed locally reproducible
  assets/license, optional frontend-only feature, shared foreground
  Runtime/Selection/retirement, loopback independent random bearer,
  numeric Host/origin boundary, bounded read-only snapshot/SSE, responsive
  shell and singular palette. CLI web and TUI same-process private URL,
  exact owner identities, no new model/cache/inference owner or actions.
  Actual HTTP/Firefox responsive smoke, JS tests/strict lint and serial
  cached TUI/HTTP/Web shared-owner release 6/0/0 pass. Complete isolated
  rerun 2,485/0/109, release Web/TUI/installer/scripts pass. UX-9A accepted.
- [x] UX-9B Web Home / Running / Serve: original resident worker promoted
  into one shared host action slot, exact identities and native-private
  results, bounded authenticated JSON POST/action receipt, explicit
  cached/configured source load/run/invoke/unload and listener start/stop,
  cancellation and original streaming output. Solid actual forms and
  terminal usage, original runtime/inference owners. Strict lint,
  JS/focused/cold HTTP/Firefox pass; real optimized cached browser/
  TUI/Web/HTTP lifecycle passed (46.14s), full isolated gate
  2,489/0/109 plus release Web/TUI/installer/scripts. Accepted.
  **Next: UX-9C Web Models / Downloads / Devices (§46A.28).**

- [ ] Close typed command tree, ConfigDocument/profile merge and shared
  presentation/action/snapshot/event model. One resolver for local paths,
  model names, explicit source locators, backend aliases and canonical
  source identity. CLI overrides selected config overrides defaults;
  no implicit remote/download/approximation authority.
- [ ] Complete model/download/cache lifecycle, progress, pause/resume/cancel,
  device discovery, headroom, qualified auto-selection, one-line run/serve,
  config/profile management, pipeline inspection and doctor/explain.
  Explicit backend aliases mean selected backend or factual refusal.
- [ ] Stable scripting output, JSON and plain non-TTY behavior; beautiful,
  responsive terminal mode. Show backend, provider, numerical/quality profile,
  shape and execution plan, supported features and actual selected/refused
  modes. Expose benchmark and diagnostics through the same SDK.

##### 2.2 Native TUI, then embedded Web

- [ ] Build native TUI (UX-8A–D): bare interactive command opens a responsive
  shell; Home/Running/Serve, Models/Downloads/Devices, Profiles/Config/Logs.
  Shared events, command palette, keyboard/navigation, cancellation,
  resize/accessibility; no action on mere visual selection.
- [ ] Build embedded SolidJS/Vite Web (UX-9A–E), following the existing AW
  reuse/design direction and same visual vocabulary. Same high-value
  views/actions as TUI; `zg web` and opening Web from TUI reuse original
  runtime/control plane. Keep presentation state local; allow simultaneous
  clients without duplicating owner state.
- [ ] Keep one ConfigDocument and control protocol; secure local control
  auth and browser transport independently of inference auth. No separate
  model/scheduler/cache implementation in a frontend. Complete UX-10
  doctor/pipeline polish, compatibility cleanup and real simultaneous-client
  lifecycle acceptance.

##### 2.3 Public serving and performance interface

- [ ] Close the [OpenAI-compatible serving iteration](../plans/serving-and-speculation/NEXT_ITERATION_OPENAI_COMPATIBLE_SERVING.md):
  model catalog/router, OpenAI Chat/Responses and supported completion/
  embedding/tool/structured-output contracts, usage, errors, JSON/SSE,
  streaming/cancellation/timeout/backpressure and real concurrent requests.
  Unsupported tasks/options must refuse rather than simulate compatibility.
- [ ] Preserve a separate control surface and inference API with typed
  configuration, explicit route/tenant scope, model lifecycle and shared
  scheduling. Add future WebSocket/WebRTC/media endpoints through the
  existing task contracts; do not wait on unimplemented media models.
- [ ] Establish ordinary public CLI/SDK/server benchmark invocation,
  effective config and end-to-end metrics, not a qualification-only serving
  process. Retain one process/multiple model instances and pipeline routing.

**Exit:** ordinary one-line run/serve; coherent model/device/config lifecycle;
both TUI and Web use one control model and real owner events; public serving
works for qualified model/task/backend paths; API behavior and actual
concurrency can be benchmarked through normal product surfaces.
Owners: [CLI/device routing](../plans/cli-device-routing/TODO.md),
[UX design](../plans/cli-ux/NEXT_ITERATION_CLI_UX_AND_MODEL_LIFECYCLE.md),
[serving](../plans/serving-and-speculation/TODO.md),
[platform](../plans/platform-and-release/TODO.md).

### 4.4 Later phases

Use full plan §§3–6 and matched-performance acceptance criteria. Primary
original authenticated HF/Safetensors and GGUF; first Qwen3.5 0.8B/2B and
Qwen3.8 27B; extend Qwen variants, Gemma4, DeepSeek/GLM/Mistral/Llama and
image/audio/ASR/TTS through shared source/model/task contracts. No
catalog entry is a capability grant. Preserve broader media, distributed,
SSD, state persistence, quantization and compiled-provider work in owners.

## 5. Current source map and implemented contracts

Original foundation source map and detailed contracts remain in the prior
Phase 1 pack §5 and its Appendix E §5. They are not rewritten by the UX
work: one semantic source-bound ModelProgram/ForwardPlan, one StateRoot
authority, one ResourceArbiter and original physical service/leases/
completion/quarantine; original source/artifact/cache ownership, native
model semantics, scheduler/worker/cohort and runtime catalog/registration.
See crates/zerogravity-core, zerogravity-artifacts, zerogravity-model-*,
zerogravity-gpu, vendor CubeCL, zerogravity-engine and zerogravity facade.

| Area | Live source owners (repository-relative) |
| --- | --- |
| Product command tree, global policies, completion and reporting | crates/zerogravity-cli/src/{main,args,commands,completion,output}.rs and output/presentation.rs |
| Canonical source/run/serve | run_cli.rs, run_cli/{args,config,devices,execute,input,source,qwen_source,qwen35_cuda}.rs; resident.rs; serve/{canonical,mod,shutdown}.rs |
| Cache/acquisition | model_cli.rs, cache_{import,manage,run}.rs, pull.rs, pull_monitor.rs, pull_native.rs and provider-specific pull_*.rs |
| One SDK ControlPlane | crates/zerogravity/src/control.rs and control/{cache,downloads,events,serving}.rs; original runtime.rs, models.rs, run.rs, prepare/, devices/ |
| One typed configuration | crates/zerogravity/src/pipelines/{document,yaml}.rs; configuration.rs and configuration/file.rs; run/config.rs; CLI config_cli.rs |
| Shared foreground interactive host | crates/zerogravity-cli/src/control_host.rs, control_host/{actions,work}.rs |
| Native client | tui.rs, tui/{library,logs,settings,terminal,view,work}.rs |
| Separate embedded control transport | web.rs, web/{actions,http,wire}.rs; not the inference transport in zerogravity-serving |
| Solid frontend | crates/zerogravity-cli/web/{package.json,package-lock.json,vite.config.js,index.html,README.md,dist/}; src/{main.jsx,resident.jsx,api.js,api.test.js,control-state.js,work-state.js,styles.css,browser-smoke.mjs} |
| Real integration | crates/zerogravity-cli/tests/{cli,config_cli,run_unified}.rs; tests/fixtures/{tui_shell,web_shell,serve_shutdown}.py; pull_monitor.rs controlled fixture; facade/engine/artifact tests |
| Packaging/quality | scripts/check.sh, scripts/native-gpu.py, nix/{source,workspace}.nix, existing release/installer scripts |

### 5.1 One cross-client ownership model

Direct CLI retains original SDK/owner paths. Canonical resident serving
and interactive clients share resident.rs. One Runtime/ControlPlane and
process-local Selection, one shared Actions slot. control_host/work.rs
executes original resident prepare/load/invoke/unload/start/stop. Native
tui/work.rs holds presentation ticket/log/detail/draft metadata only.
Slot has exact monotonic action ID, origin, JoinHandle/cancellation and
bounded response; no queue. Native consumes native ticket, Web observes
only its own active/latest receipt (not native prompt/output); Web receipt
retention independent of intervening native actions. Original runtime
registration/generation/download IDs are distinct from action IDs.

Original owner events and selection/action revision invalidate snapshots.
SSE is bounded refresh/resync, not durable event/token history. Browser
samples its active receipt with bounded 64 KiB UTF-8 output prefix and
actual final usage/reason; original OutputStream unchanged. Cancellation
request is not terminal completion. Host closes/joins Web connections
and action before inference listener/runtime retirement. Tab/modal close
and navigation do not retire owners.

### 5.2 Web control security and wire

- Optional CLI-only web feature in ordinary product default/all and Nix.
  Web-only needs no terminal deps, inference serve independent, no Node
  runtime or external assets. Pinned locally built dist/license.
- Numeric loopback, OS-generated 32-byte random control credential in
  private URL fragment, consumed/scrubbed and tab-scoped storage; lock
  clears it. Inference bearer is independent. No public exposure.
- /ui/ assets, authenticated /zg/control/snapshot and /zg/events,
  POST /zg/control/actions and GET /zg/control/actions/ID. Exact grammar
  in web/actions.rs/http.rs. Numeric Host/origin/Fetch-Site, explicit
  bearer, CSP/no-store/no-referrer/frame denial, no CORS. Eight connection
  slots, four SSE slots, bounded headers/read/write/SSE lifetime; POST
  bounded 256 KiB application/json after auth, full same-origin Origin,
  deny unknown fields and duplicates, no chunked. 202 is admission, not
  completion; busy/stale cancel 409, malformed 400.
- Existing browser resident source input: blank trusted selected document,
  exact cache/digest/catalog/by-name or pinned provider commit, optional
  validated ID=MODEL. No arbitrary explicit local/HTTP path, argv,
  credentials/device/execution overrides, automatic pull/fallback.
- Real browser load/run/invoke/unload/start/stop. Exact registration,
  generation, origin and confirmation. Cache/device/pull and settings
  variants still native feature-gated in shared worker; next slices
  promote necessary neutral operations rather than terminal presentation.
- TUI Home/Serve b opens same-process private URL. Standalone web starts
  empty original runtime. No second execution host. UX-9A read-only
  statements are historical, superseded by UX-9B resident mutations.

## 6. Original chronology — beginning through Phase 1

Original v4.3 design dates to 10 Oct; first reachable Git design commit
13 Oct. Earlier critical reviews, P00–P23 roadmap, Fabric vNext, source/
mathematical/physical and model/backend decisions remain in original files.
The [25 Nov Phase 1 exit pack](FULL_COMPREHENSIVE_HANDOFF_2026-11-25_PHASE1_EXIT.md)
is complete: Appendix E §§6–7 has beginning-to-24-Nov chronology; §7
and Appendix D preserve every post-matrix foundation landing and exact
evidence. Its Appendix A/B/C/D/E and earlier handoffs are retained intact.
Appendix E of this new pack reproduces its entire main body. Appendix C
here extends the full ledger from the first design commit to UX-9B.
Use historical source/experiment details, not old continuation prompts.

### 6.1 Post-matrix foundation checkpoint map (exact historical excerpt)

#### Accepted post-matrix steps

The following groups cover every post-matrix implementation slice. Intermediate fixes, tests and docs commits are in the exact 49-commit delta (the prior Phase 1 pack Appendix C); complete detailed accepted narratives, faults, gate counts, ownership/quality boundaries and measured real workload results in the prior Phase 1 pack Appendix D.

| Implementation / corrections | Accepted scope | Documentation |
| --- | --- | --- |
| `687d9a24`, `06df027f`, `43928332`, `f4e09428` | Authenticated parameter materialization, Original/BF16 RNE/F32Exact and portable padded views, original Write and physical staging retirement; real CUDA resident/streamed B1/B4 and HIP V2 C1/C8 | `16647514` |
| `3ca16abd`, `a53a3446` | Direct original typed region source operands and explicit compiled selection; SDK-free scaffold, no external compiler execution | `2a63b875` |
| `8eb17bc2`, `00f57e7a`, `9374e0ca` | Source/model eligibility independent of task/backend; actual cached/direct CPU and HF comparisons, CUDA/HIP canaries; historical cache-admission fixture correction | `eb32ebae` |
| `9ae6dd1e`, `e06213a5` | CUDA pre-Client physical envelope, exact PCI and live clamp; real peer-pressure refusal/retry and native HF/GGUF | `f31ba4db` |
| `cbd2c119`, `197c2e26` | Aggregate host backing/cgroup-v2/current fit, actual HIP driver/local VRAM and cold reservation instead of lifetime upper-as-growth | `814d5c00` |
| `a6548863` | Ordinary HIP and guarded Vulkan original preadmission, same declared physical-domain placement; real shared healthy/unknown/retry | `86a3792a` |
| `1d5c4ada`, `f9dd5ed6`, `35629e6c` | Optional native CPU GGUF committed checkpoint v2; immutable source/profile, bounded SHA256/atomic IO, import lease, original authority; actual 0.8B/2B full-vocab exact continuation and refusal/retirement; legacy diagnostic v1 preserved | `566dcf8d` |
| `804d4882`, `d9b25205`, `21ce3d7f`, `ef7c3d6c` | Native HIP V2 F32 contiguous resident persistence, source semantic revision, exact prepared geometry, original root pool/completion; actual same/fresh-process HF all-domain/full-vocab equality and unknown H2D retention/fence | `de1ef676` |
| `ebda04c4` | Source-declared serial language load with exact original owner/alias, atomic router+plan and rollback; actual CPU 0.8B→2B, CUDA three-stage two-owner and HIP same-source chain | `212f6eab` |
| `b25b5c02` | Common borrowed TextTask/model/plan, controls, bounded polling/cancel, structured errors, exact selected-owner/config observations; actual CPU framed nonstream and CPU/CUDA/HIP chains | `9a5fc6e6` |
| `7987f70a`, `2ec06026` | Shared held local/cache GGUF selection and actual target/context, TextLoadPolicy, raw lease/exact cache digest, retarget refusal and exact direct selection; preserve CLI diagnostic contracts | `24bba976` |
| `8e3b8993`, `33118f16` | Original prepared program/region observations and actual retained CUDA projection, dense identical-layout same-context transport and secondary stream; preserve GEMM orientation, actual completion/fence | `1462687e` |
| `3b5a520d`, `fc38db03`, `e1decff5` | Original CubeCL registry noncreating lookup/atomic factory and exact Client; named Linux Vulkan preadmission/current fit, cold peer/raced/warm reuse and real healthy/unknown/retirement | `9017af17` |
| `ce83cc18` | Actual captured owner profile/source/stream/receipt/cache/pin/service and accepted replay; successful cohort Graph vs Native; real CUDA GGUF B2 fullV/all42, fallback/retirement/unknown, HIP V2 C1/C8 | `0886f11f` |
| `14bbbd48`, `5d952364`, `5a7535e7` | Exact coherent host-mapped resident producer, one actual backing, original range/Write/completion, real RADV full/subrange and refusal/retirement; logical upload accounting; bounded foundation exit | `d5030ad4` |

The latest mapped-backing gate is 2396 workspace test executions/0 failures/106 skips plus separately repeated affected final checks. The prior capture gate is 2398 assembled (2396 workspace + 2 focused graph tests). Do not equate counts or correctness with a new speed/quality claim. All earlier per-slice exact scopes remain in the prior Phase 1 pack Appendix D and original owner plans.

## 7. Every Phase 2 step through UX-9B

Exact SHA/date/subject in Appendix C; exact implementation, evidence,
corrections, limitations and changed paths in Appendix D. Counts below
are workspace test executions / failures / explicit skips, not unique
tests or a new performance claim.

| Slice | Commit | Scope | Gate |
| --- | --- | --- | --- |
| UX-1 | e6211608 | Shared presentation, terminal capability/color/NO_COLOR, quiet/verbose/progress, typed failures and machine streams. | 2431/0/106 |
| UX-2 | bcdb46a8 | Typed command tree, exact cache/digest/by-name, common cold source/config/device resolver, compatibility forwarding. | 2451/0/106 |
| UX-3 | 3cf1bb2d | Shared SDK ControlPlane, typed snapshots/actions/events, exact registration/generation, original task stream and two-client lifecycle. | 2457/0/108 |
| UX-4 | da2385c7 | Original download executor, operation identity, progress/receipt and safe strong-validator checkpoint/range pause/resume/cancel, partial observations. | 2459/0/109 |
| UX-5 | 8e382947 | Canonical one-line run, exact cached/approved source selection, explicit acquisition consent/recipe, reusable resident owner across independent prompts. | 2465/0/109 |
| UX-6 | 84e0681c | Canonical serve, shared run/source/config/device resolution, typed serve policy, one-owner alias equality and actual joined SIGTERM. | 2466/0/109 |
| UX-7 implementation | 7f5fc4eb | One typed v2 ConfigDocument, atomic edits/profiles and provenance. Full acceptance interrupted at this commit. | Partial; see D.7 |
| UX-7 acceptance | abe98b83 | Recovered parser compatibility and complete serial real-model/full-gate acceptance; unchanged 8 GiB serving envelope. | 2473/0/109 |
| UX-8A | 08e7e9d3 | Native responsive Ratatui/Crossterm shell, nine views, palette and scoped terminal/signal retirement. | 2477/0/109 |
| UX-8B | 0c7ad87a | Real native resident load/run/invoke/unload/start/stop through original planner/owners, one scoped worker, bounded output. | 2479/0/109 |
| UX-8C | 9bf7829a | Native cache/catalog/partials/device views and exact cache/pull/device actions; pinned consent, pause/resume/cancel, one cold reader. | 2481/0/109 |
| UX-8D | 71e36933 | Native profiles/config/raw YAML/logs; exact atomic snapshots, future-only activation, all cold reads off rendering loop. | 2483/0/109 |
| UX-9A | 4e472aed | Offline-built embedded SolidJS/Vite shell, shared host/selection, separate authenticated loopback snapshots/SSE. | 2485/0/109 |
| UX-9B | 4f23983a | Shared native/Web action slot; authenticated real resident actions, exact IDs/confirmation/cancel and bounded Web-owned output receipt. | 2489/0/109 |

### 7.1 Durable product decisions

One selected v2 ConfigDocument, default/named/explicit replacement and
CLI precedence; no inheritance or settings DB. Atomic ConfigFile
snapshot compare, no-clobber create, copy preserves relative source
meaning, raw repair, no symlink mutation. Activation future-only and
process-local, no persisted selection pointer. One original scheduler,
source resolver, cache/transfer, registration and inference listener.
Canonical resident route set bounded CPU GGUF or one approved original
GPU snapshot aliases; mixed/distinct original GPU routes and broader
backends remain explicit refusals. Native and Web selection inert, actual
owner identities and explicit action consent; bracketed paste data only.
Operational log ring 128 local entries, not persistent model output.
One shared action worker/slot; long cold observations off rendering loop.
Separate control/inference auth, original feature boundaries.

### 7.2 Recovered failures and qualification lessons

UX-6 stale CLI expectations corrected. UX-7 parser error-category
regression corrected; initial host-pressure refusal at 8 GiB and
interrupted full gate closed by serial validation at abe98b83.
Later PTY differential ANSI/escape/label fixture corrections, UX-9A
fragment relaunch/TCP status-line read correction, UX-9B PTY endpoint/
status and actual host-pressure refusal (7,615 MiB available) retained.
Do not relax admission, expected output or auth to make tests pass.
Original detailed failures and reruns in Appendix D. Reuse applicable
evidence and existing fixtures. UI lifecycle wall-clock not throughput.

## 8. Current execution and performance authority

Read CURRENT_BASELINE and CURRENT_HANDOFF. Phase 2 work does not start
a backend performance campaign. Retain the original qualified evidence:
X2 HIP original HF V2 Qwen 0.8B C1/C8 independent continuation and
retirement, distinct native vs exact source arithmetic; L5 CUDA original
HF/GGUF resident/streamed/paged/prefix/cohort and real GGUF B2 capture
full-vocab/all42 equality and unknown retention, public graph serving
Native; Vulkan named RADV/portable Q8/conv and coherent mapped producer,
not learned parity or general foreign/UMA backing; CPU GGUF 0.8B/2B
source/cache, committed continuation and language chains.

Retained pre-foundation HIP V2 47+3 0.8B C8 43.638 tok/s, 2B C8
30.949 tok/s, 27B C1 2.228 tok/s. Small-model retained vLLM
219.612/140.847 tok/s are not fresh matched comparisons. Approved
pinned X2 vLLM closure unavailable as recorded; restore approved local
closure before Phase 3. Full measurement conditions and rejected
approaches remain in old Phase 1 pack Appendix E §§8–9, earlier packs
and live baseline. No new measurements from documentation or UX work.

## 9. Reusable primitives and held/rejected experiments

Preserve correct reusable shared GEMM/solve, GDN/attention/norm, tiled/batched/paged/streamed transport, source-window, state/layout, sampling, graph/cache, admission/lease, completion and pipeline primitives when a particular consumer/selector loses. Remove losing selection and temporary probes, not model-neutral implementations. Delete only incorrect/unsafe/incompatible or strictly superseded primitives with reasons in the owning plan.

The exact old rejection and hold map is preserved in the prior Phase 1 pack Appendix E §9 and earlier handoffs. Native graph timing at useful widths did not justify production selection; retain graph substrate/cache without promoting selector. Paged scatter-tail-zero did not justify production consumer; retain reusable kernel. Do not repeat unless materially new evidence. Respect source/model/hardware/quality holds and do not weaken them for nicer UI. Concrete compiler SDKs are later, not substitutes for native operators.

## 10. Build, validation, host and Git procedure

AGENTS.md and installed toolchain/cache are authoritative. Mandatory
assembled gate on outer shared-network connector:

~~~sh
cd /home/pvl/spaces/abird/src/zerogravity
export CARGO_BUILD_JOBS=8 RAYON_NUM_THREADS=4 PYTHONDONTWRITEBYTECODE=1
bwrap --unshare-net --bind / / --dev-bind /dev /dev -- \
  python3 -B scripts/native-gpu.py --require-private-network -- bash scripts/check.sh
~~~

Never bypass mandatory private-network harness. Locked/offline approved
inputs only. Focused compile/tests for actual changes, meaningful real
consumer checks, full assembled gate last. Use 4 Cargo jobs if shared
memory pressure; run real cached serving serially at unchanged 8 GiB
envelope, no competing release build. Do not clean unrelated caches/jobs
or weaken resource bounds. Check staged paths, HEAD/tracking/worktree and
shared physical owners. Sign owned commits; verify status/signature if
connector cleanup warning. L5 authors, X2/A1 execution/testing only,
verified incremental bundles and clean fast-forward; no remote push/fetch
or destructive Git. Previous exact host/runtime/library/cache/bundle and
physical qualification recipes are preserved in prior Phase 1 pack
Appendix E §10. Historical paths are not assumed available now.

### 10.1 Product/frontend checks

Ratatui 0.30.0/Crossterm 0.29.0; SolidJS 1.9.15/Vite 8.3.2/
vite-plugin-solid 2.11.14/esbuild 0.28.2 pinned already cached.
[Web README](../../crates/zerogravity-cli/web/README.md) documents
offline npm ci, npm run test/build, committed dist/license and installed
Firefox BiDi smoke. Rebuild only for frontend changes; no Node runtime
or remote dependencies. Preserve strict minimal TUI-only, Web-only,
minimal CPU and combined serve/acquisition/backend feature boundaries.
Existing Python terminal/HTTP fixtures, actual cached-model release
run_unified and controlled transfer fixture are authoritative. The
focused real cached case
canonical_serve_profile_and_cached_aliases_use_one_listener_owner
exercises original inference/TUI/Web lifecycle. Use current flags and
loopback setup in scripts/check.sh, e.g. cargo test --locked --offline
--release -p zerogravity-cli --features serve,acquisition --test
run_unified -- --include-ignored --nocapture --test-threads=1.
Run real controlled transfer for acquisition changes; no production
test transport override or unapproved remote transfer. Full gate
includes feature/architecture, release Web/TUI/installer/scripts;
browser smoke is separate focused verification. No cross-build or
physical execution claim from compilation.

This is docs-only takeover; prior accepted implementation evidence
retained, no fresh implementation gate claimed.

## 11. Next coherent implementation unit — UX-9C

Read live canonical §46A.9 and §46A.13–29 plus native UX-8C and Web
UX-9A/B. Do not reopen Phase 1 or UX-9B.

### 11.1 Starting source and invariants

- control_host/actions.rs: original single shared effect slot, exact
  ActionSummary, origin-aware start/active/cancel/poll_native/observe/
  web_result/shutdown. Do not queue another effect or conflate ID types.
- control_host/work.rs: shared resident Request/Owner/Prompt, original
  preparation/invocation; Inspect/Verify/Reclaim/Alias/CheckFit/Pull and
  Settings still native-feature gated. Promote needed neutral operations,
  not terminal presentation. Native ticket/log/detail/draft metadata
  remains client-local.
- tui/library.rs: one coalesced cold reader, exact digest/slug/partial/
  physical-device row identity, contextual list/detail; reuse neutral
  observations. Profile/config and logs are next UX-9D.
- pull_native.rs + pull.rs: original explicit pinned intent/recipe and
  common hub/recipe transfer. SDK control/downloads.rs: exact operation,
  safe pause/resume/cancel, transfer owner and actual outcome.
  control/cache.rs: exact manifest/payload verification, additive alias,
  original preview/apply reclaim outcomes.
- web/actions.rs/http.rs/wire.rs: bounded authenticated typed input,
  allowlisted output and exact ids, current snapshot/SSE/action receipts.
  Work-state/api/control-state already implement admission vs completion,
  no automatic POST retry, exact origin/cancel and stale refresh.
- SDK devices/ and resident.rs/run/config.rs/run_policy: actual physical
  observation, source/device selection/admission. No inferred eligibility.

### 11.2 Coherent deliverable

Real browser counterpart of accepted native UX-8C, not a separate
engine or business-rule implementation:
1. Cached/catalog/partial/device observation rows with stable exact
   identity, cold/on relevant invalidation rather than token/progress
   scan; responsive contextual list/detail, no fictitious rows.
2. Exact-digest inspect/verify/load/run/load-for-Serve, non-replacing
   alias, dry-run/confirmed reclaim. Preserve busy/referenced/durable/
   incomplete semantics. No arbitrary root/path/ref mutation.
3. Explicit pinned full-commit HF/GitHub files/quant or approved Qwen
   recipe, consent, quota/admission and expected snapshot digest guard.
   Offline before transport; no URL/tag/quant guess, auto load, quota
   increase or automatic eviction.
4. Original transfer phase/status, bytes and known total/maximum,
   logical rate and checkpoint/resumed/terminal durability. Exact
   operation pause/resume/cancel available while pull owns slot. Resume
   partial by repeating original validated intent, not opaque checkpoint.
5. Observed physical device identity, memory/backend/provenance/
   exclusions, selected exact source/device cold policy check or explicit
   load via original resident planner; no implicit fallback or persisted
   preference on selection.
6. Preserve separate loopback control/inference auth, bounded typed
   same-origin JSON, canonical exact IDs, consent/stale refusal,
   Web-owned receipt privacy and shared invalidations. No arbitrary
   argv/credentials/raw config/path or new execution authority.

### 11.3 Qualification and closure

Use existing real cached model/cache and original controlled-transfer
fixtures, practical HTTP/auth/body/identity/offline/stale tests, JS and
real Firefox controls, native PTY and shared-owner lifecycle. Preserve
actual cancellation/retirement and minimal feature isolation. Full
mandatory private-network gate last; do not require remote acquisition
or GPU rerun when no changed physical consumer, and do not claim
unqualified execution. One signed coherent checkpoint, ~30 minutes;
document incomplete checks if blocked rather than infer acceptance.

### 11.4 Ordered follow-ups

UX-9D Web Profiles/Config/Logs: original ConfigFile atomic snapshot/
validation/edit, protected remove/activation, same Selection, operational
observations and redaction; no second DB or invented durable logs.
UX-9E simultaneous TUI/Web mutation acceptance using §46A.29 and exact
owner identities, stale refusal and lifecycle. UX-10 doctor/pipeline/
dynamic completion/compatibility cleanup. Then public serving and normal
benchmark acceptance via separate serving iteration. Then ROCm, CUDA,
Vulkan, optional concrete compiled providers. All broader model,
distributed, media, persistence and resource work remains in owning plans.

## 12. Specification navigation and complete historical context

The original source navigation below is preserved from the Phase 1 pack.
Earlier continuation advice is historical. All original source docs,
prior art, research, owning plans and complete old packs remain in place.

### 12.1 Original specifications and historical chain (historical)

| Subject | Authoritative detailed documents |
| --- | --- |
| Original product/design/critique | [master plan](../plans/ZEROGRAVITY_MASTER_PLAN_2026-10-10.md), [vision](VISION_SCOPE_AND_PRINCIPLES.md), [modular architecture](MODULAR_AI_SYSTEMS_REFERENCE_ARCHITECTURE.md), [critical review](../prior-art/ARCHITECTURE_V4_CRITICAL_REVIEW_2026-10-12.md), [engineering specification](../../docs/ENGINEERING_SPECIFICATION.md), [whitepaper](../../docs/WHITEPAPER.md) |
| Source/model/semantics | [composition](COMPOSITION_AND_EXTENSION_CONTRACTS.md), [model semantics](MODEL_SEMANTICS_AND_COMPATIBILITY.md), [formats](FORMATS_QUANTIZATION_AND_LOADING.md), [derived quant](DERIVED_QUANTIZATION_CONTRACT_2026-10-15.md), [verified model assemblies](VERIFIED_MODEL_ASSEMBLIES_AND_SPECIALIZATION_V5_1_2026-10-15.md), [immutable sharded source](../plans/IMMUTABLE_SHARDED_SOURCE_IMPLEMENTATION_SPEC.md), [acquisition](../plans/MODEL_ACQUISITION_IMPLEMENTATION_SPEC.md) |
| Shared region/provider/physical executable | [semantic IR](SEMANTIC_IR_AND_COMPILER.md), [task/compiler refinement](TASK_PROGRAM_AND_COMPILER_REFINEMENT_V4.md), [accelerator families](ACCELERATOR_EXECUTION_FAMILIES_AND_BACKEND_CONTRACT.md), [regions](../plans/pipeline-composition/REGIONS_AND_PROFILES.md), [policy/reuse](../plans/pipeline-composition/PROFILE_POLICY_AND_COMPONENT_REUSE.md), [composition TODO](../plans/pipeline-composition/TODO.md) |
| Fabric transition | [Fabric vNext](../plans/fabric-vnext/README.md), [decisions](../plans/fabric-vnext/DECISIONS_AND_OPEN_QUESTIONS.md), [acceptance](../plans/fabric-vnext/IMPLEMENTATION_AND_ACCEPTANCE.md), [transition contracts](../plans/fabric-vnext-transition/ARCHITECTURE_AND_CRATE_CONTRACTS.md), [phased transition](../plans/fabric-vnext-transition/PHASED_IMPLEMENTATION.md), [transition TODO](../plans/fabric-vnext-transition/TODO.md) |
| Physical/state/memory | [execution core](ENGINE_CORE_EXECUTION_AND_PROTOCOL.md), [minimal spine](MINIMAL_EXECUTION_SPINE_AND_PROTOCOL_V4.md), [state](STATE_CACHE_AND_TRANSACTIONS.md), [residency](MEMORY_RESIDENCY_AND_IO.md), [unified virtual tensors/SSD](UNIFIED_VIRTUAL_TENSOR_RESIDENCY_AND_SSD_STREAMING_2026-10-15.md), [memory TODO](../plans/memory-and-streaming/TODO.md), [gfx1151 safety](GFX1151_ROCM_KERNEL_COST_AND_UNIFIED_MEMORY_SAFETY_CONTRACT_2026-10-15.md) |
| Math/performance/numerics | [operator catalog](OPERATOR_AND_FUSION_CATALOG.md), [device kernels](DEVICE_KERNEL_AND_CODEGEN.md), [cost model](COST_MODEL_AND_AUTOTUNING.md), [numerics/quality](NUMERICS_QUALITY_AND_BENCHMARKS.md), [correctness policy](../plans/numerics-and-context/CORRECTNESS_POLICY.md), [execution TODO](../plans/model-execution/TODO.md), [backend parity TODO](../plans/cpu-gpu-optimization/TODO.md), [current baseline](CURRENT_BASELINE.md) |
| Scheduling/serving/UX | [scheduling/speculation](SCHEDULER_BATCHING_AND_SPECULATION.md), [router/residency](MODEL_ROUTING_RESIDENCY_AND_MULTI_MODEL_SERVING_CONTRACT.md), [zero-config/API](ZERO_CONFIG_CLI_CONFIG_AND_OPENAI_API_CONTRACT.md), [runtime/UX](RUNTIME_SERVING_AND_UX.md), [CLI UX](../plans/cli-ux/NEXT_ITERATION_CLI_UX_AND_MODEL_LIFECYCLE.md), [serving plan](../plans/serving-and-speculation/NEXT_ITERATION_OPENAI_COMPATIBLE_SERVING.md), [batching/paged state](../plans/serving-and-speculation/BATCHING_AND_PAGED_STATE.md) |
| Multimodal/distributed | [multimodal V5.1](UNIVERSAL_MULTIMODAL_EXECUTION_ARCHITECTURE_V5_1_2026-10-15.md), [realtime/QoS](REALTIME_MULTIMODAL_SERVING_AND_QOS_V5_1_2026-10-15.md), [NPU cooperation](NPU_GPU_COOPERATIVE_EXECUTION_AND_TASK_PIPELINES.md), [distributed placement](DISTRIBUTED_PLACEMENT_AND_NETWORK_EXECUTION_CONTRACT.md), [peer state handoff](REMOTE_PEER_TRANSPORT_AND_STATE_HANDOFF_PROTOCOL.md), [multimodal TODO](../plans/multimodal/TODO.md), [distributed TODO](../plans/distributed-execution/TODO.md) |
| Reliability/release | [security](SECURITY_AND_RELIABILITY.md), [Rust engineering](RUST_WORKSPACE_AND_ENGINEERING.md), [failure semantics](INVARIANTS_AND_FAILURE_SEMANTICS.md), [delivery gates](../plans/INTEGRATED_PHASE_0_AND_DELIVERY_GATES.md), [release acceptance](../plans/RELEASE_ACCEPTANCE_GATES.md), [traceability](../plans/REQUIREMENTS_AND_TEST_TRACEABILITY.md), [platform TODO](../plans/platform-and-release/TODO.md) |

### Historical handoff chain (preserve, do not rewrite)

- [24 Nov foundations](FULL_COMPREHENSIVE_HANDOFF_2026-11-24_FOUNDATIONS.md): original accepted whole-program preparation through `17e2b717`, complete earlier designs/open work and 800-commit chronology. Its old continuation prompt is superseded.
- [24 Nov morning](FULL_COMPREHENSIVE_HANDOFF_2026-11-24.md): original mission,
  full pre-native chronology, A1/HIP projection experiments, CUDA provider
  landing caution, selected vs dormant primitives, full earlier ledger.
- [23 Nov](FULL_COMPREHENSIVE_HANDOFF_2026-11-23.md): build/feature/product
  convergence, current CUDA/HIP paths, prior design and execution delta.
- [22 Nov](FULL_COMPREHENSIVE_HANDOFF_2026-11-22.md) and
  [21 Nov](FULL_COMPREHENSIVE_HANDOFF_2026-11-21.md): detailed source, native
  provider, paged/KV/state, graph/lifecycle, serving and physical evidence archives.
- [20 Nov](FULL_COMPREHENSIVE_HANDOFF_2026-11-20.md),
  [18 Nov](FULL_COMPREHENSIVE_HANDOFF_2026-11-18.md),
  [15 Nov](FULL_COMPREHENSIVE_HANDOFF_2026-11-15.md),
  [14 Nov](FULL_COMPREHENSIVE_HANDOFF_2026-11-14.md): earlier CUDA/HIP,
  original source and resource, scheduling and model execution chronology.
- Original October specs, P00–P23 roadmap, review/prior art, Fabric vNext
  decision/transition files and complete Git history in the prior Phase 1 pack Appendix C cover
  the design and implementation from its beginning.

No historical pack was deleted or replaced. Older claims of a failing full
gate, a concurrent uncommitted FlashInfer slice, HIP host-runtime failure,
or immediate scalar/CUDA-first next steps are superseded by this checkpoint
where explicitly reconciled. Their measured historical outcomes still matter.
Where an old file gives contradictory instructions, use the full current
plan, actual source and newest accepted result.

All tracked research documents are linked in the prior Phase 1 pack Appendix A as well as all
maintained specs and plans. Their original locations and exact contents
are preserved; research is not a production capability claim.

### Phase 2 authority refresh

Read [CLI UX](../plans/cli-ux/NEXT_ITERATION_CLI_UX_AND_MODEL_LIFECYCLE.md),
[CLI/device routing](../plans/cli-device-routing/TODO.md),
[public SDK/serving](../plans/fabric-vnext/PUBLIC_LIB_AND_SERVING.md),
[serving iteration](../plans/serving-and-speculation/NEXT_ITERATION_OPENAI_COMPATIBLE_SERVING.md),
[serving TODO](../plans/serving-and-speculation/TODO.md) and
[platform TODO](../plans/platform-and-release/TODO.md).
The preceding [25 Nov foundations takeover](FULL_COMPREHENSIVE_HANDOFF_2026-11-25_FOUNDATIONS.md)
is now historical for its continuation prompt, but its entire original
history and Appendices A–D remain available. All prior handoffs are linked
in the prior Phase 1 pack Appendix A.

### 12.2 Current product authority

Read the canonical UX specification and owner TODOs linked in §1,
especially native UX-8C / Web UX-9A/B accepted contracts and §46A.29.
The original [Phase 1 exit takeover](FULL_COMPREHENSIVE_HANDOFF_2026-11-25_PHASE1_EXIT.md)
retains all appendices A–E, including exact foundation 49-commit delta,
final mapped backing and original physical evidence, previous chronology
back to October, exact measured/rejected experiments, and operating
recipes. This pack extends rather than replaces it. Appendix A lists
every engineering source document. No historical file deleted.

## 13. Copy/paste starting prompt

You are taking over ZeroGravity. Continue in:

/home/pvl/spaces/abird/src/zerogravity

### 1. Acquire complete project context

Read in order:
1. AGENTS.md
2. .agents/plans/README.md
3. .agents/docs/FULL_COMPREHENSIVE_HANDOFF_2026-11-26_UX9B_EXIT.md
4. .agents/plans/FULL_IMPLEMENTATION_PLAN.md
5. .agents/docs/CURRENT_HANDOFF.md
6. .agents/docs/CURRENT_BASELINE.md
7. .agents/plans/TODO.md
8. .agents/plans/cli-ux/NEXT_ITERATION_CLI_UX_AND_MODEL_LIFECYCLE.md
9. Relevant CLI/device-routing, acquisition, serving, public SDK and platform specifications/TODOs linked by the handoff.

Read §§0–13 first. Appendix D contains every exact Phase 2 landing
narrative, including UX-7 implementation and later acceptance. Appendix E
and the 25 Nov Phase 1 exit pack preserve the original design, full
foundation implementation, prior chronology, physical/source evidence,
reusable primitives and rejected experiments. Appendix B is a dated
literal open-item snapshot; live owning TODOs take precedence. Do not
reconstruct architecture or repeat rejected experiments without materially
new evidence.

### 2. Revalidate actual repository and execution authority

Latest accepted implementation: 4f23983a5bb33a3ab2e9ef124342a0cc33df2088 (UX-9B).
Accepted Phase 1 exit: d5030ad4ce476b913a46056d492992ccf1b8edf7.
Preceding comprehensive Phase 1 takeover: 450cea5530dad567ff7706c30e1a1872ff01bda7.

The new handoff is a documentation-only descendant; determine its SHA
from Git. Revalidate actual HEAD, staged/working tree, tracking ref,
worktrees, jobs and shared physical resource owners. Live source,
executed tests and owning plans take precedence if advanced.

Use dotlink---l5 as the sole source-authoring connector. Main is the only
persistent branch and checkout. X2 and A1 are execution/testing only;
synchronize exact L5 commits by verified incremental Git bundles when
needed. Preserve unrelated agents' work/jobs/resources. No destructive
Git, privilege/recovery changes, GPU resets, unapproved downloads, or
push without explicit user approval.

### 3. Current implementation position

Minimum Phase 1 shared foundations accepted within bounded qualified
scope. UX-1–7, UX-8A–D, UX-9A and UX-9B accepted. Latest full isolated
workspace gate 2,489 passed, zero failed, 109 explicit skips; release
Web/TUI, installer and scripts pass. Actual cached Qwen 0.8B GGUF
browser/TUI/HTTP lifecycle passed with original 8 GiB host envelope.
No new physical GPU, public-network, numerical or performance grant.

One Runtime/ControlPlane, one process-local Selection, one shared Actions
slot, original model/registration/OutputStream/cache/download/listener
owners and one typed ConfigDocument. Resident native and Web actions
share the worker. Browser control auth is independent of inference auth.
Exact wire IDs are canonical strings. Native cache/download/device and
settings operations still require their respective ordered Web slices.

### 4. Next coherent unit — UX-9C: Web Models / Downloads / Devices

Read canonical UX §46A.9, §46A.13–29 and the UX-8C/9A/9B narratives.
Inspect control_host.rs, control_host/{actions,work}.rs,
tui/{library,work,view}.rs, tui.rs, web/{actions,http,wire}.rs, web.rs,
resident.rs, pull_native.rs and pull.rs, original SDK control/cache.rs,
control/downloads.rs, device policy and artifact/cache/acquisition owners.
Inspect SolidJS web/src/{main.jsx,resident.jsx,api.js,control-state.js,
work-state.js,styles.css} and actual tests.

Implement bounded native-equivalent Web model/catalog/partial, download
and device actions and observations through the original owners:
- Promote the relevant native request variants and cold observations into
  the existing host-neutral seam; no duplicate cache/registry, terminal
  dependency in Web-only builds, or new action queue.
- Exact digest inspect/verify/load/run/load-for-Serve, non-replacing alias,
  dry-run/confirmed reclaim, original referenced/busy/durable/incomplete
  outcomes. No arbitrary cache root/path, raw diagnostics or provider
  credentials. Catalog/partial metadata is not execution/resume authority.
- Use original pinned explicit HF/GitHub files/quant or approved Qwen
  recipe, consent, cache quota/admission/expected digest and publication.
  Offline before transport; no automatic download, quota increase,
  tag/quant guess, opaque checkpoint resume or automatic load.
- Display original download phase/status, actual bytes/total/maximum/rate/
  checkpoint/resumed and terminal durability. Pause/resume/cancel targets
  exact operation ID even while pull owns the shared slot, not a second
  action queue.
- Show observed physical device identity, memory/backend/provenance/
  exclusion, exact source/device policy check or explicit load through
  original resident planner. No inferred support, implicit fallback or
  persisted preference on visual selection.
- Preserve bounded deny-unknown-fields JSON, independent loopback bearer
  and same-origin auth, exact action/registration/generation/operation
  IDs, confirmation/stale refusal, native-private result handling and
  original event invalidations. No arbitrary filesystem/argv/credential
  or unrestricted execution grant. Client disconnect is not retirement.
- Reuse actual cached lifecycle, controlled transfer, Web HTTP/Firefox and
  native PTY fixtures. Cover real cache lifecycle, offline/feature/auth/
  stale refusal, original pause/range resume/cancel, shared owner
  observations and joined retirement. Meaningful focused checks then the
  complete mandatory isolated gate. No remote transfer or physical GPU
  requalification unless actually needed and authorized.

Do not start UX-9D/E, UX-10, public serving expansion or backend
performance prematurely.

### 5. Ordered plan and engineering principles

1. Shared foundations — bounded exit accepted.
2. CLI, native TUI, embedded SolidJS Web, compatibility and public serving.
3. Match or beat optimized vLLM on ROCm.
4. Match or beat optimized vLLM on CUDA.
5. Native Vulkan feature/correctness/performance parity.
6. Optional concrete MIGraphX, TensorRT, IREE and ONNX Runtime.

After UX-9C: UX-9D Profiles/Config/Logs, UX-9E simultaneous client
acceptance, UX-10 doctor/pipeline/compatibility, then public serving and
normal benchmark acceptance. One SDK/runtime/catalog/ConfigDocument/
actions/snapshots/events. Separate inference/control auth. Original
HF/Safetensors and GGUF primary; no mandatory ONNX conversion, second
canonical IR or external graph SDK for native execution.

Elegant, simple, clear and intuitive Rust. Favor zero-cost abstractions
and clean direct hot paths when practical; targeted optimization of
measured bottlenecks when complexity otherwise grows. Reuse mature math.
Preserve correct reusable model-neutral primitives even if a production
selector loses. Remove losing consumer/probe wiring, not useful
implementations. Preserve exact source/numerical/layout/state/resource/
ownership/completion/cancellation.

No overengineered granularity, evidence pinning or redundant qualification.
Reuse accepted evidence and actual changed-consumer checks, then the
assembled full private-network gate. Use locked/offline approved inputs,
existing scripts/check.sh, and coordinate build/memory/device owners.
Keep the original 8 GiB real cached serving admission bound.

### 6. Cadence and closeout

One substantial coherent validated signed checkpoint per user turn,
approximately 30 minutes. Recover interrupted work from actual source
and job results. Update owning TODOs, CURRENT_HANDOFF and CURRENT_BASELINE
as appropriate. Report completed work, exact commits, accepted scope,
limitations, failed/interrupted checks and concrete next unit. Preserve
unrelated work and resources. Do not push.

Revalidate current state and implement UX-9C.
~~~~
