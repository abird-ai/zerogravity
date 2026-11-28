# CLI and device routing

**24 Nov delivery order:** the [full implementation plan](../FULL_IMPLEMENTATION_PLAN.md)
sets phase priority. Shared foundations (including optional compiled-provider
scaffolding), CLI/TUI/Web and serving, ROCm, CUDA, Vulkan, then concrete
compiled providers. Native Hugging Face/Safetensors and GGUF are primary.
Older ordering below is context; valid owner constraints remain authoritative.

Base: `fb68d8fa165de16e21d1bd37afef3790b80da971`

**Implementation authorized24Oct; Pi implementation/execution, ROOT merge gate.** Owner: @pvl.
Read [the contract](README.md). Base records the draft's inspected HEAD; live
source and current tests outrank it. ROOT reviews/reconciles/merges; the source
commits below are signed and local. Existing model/backend/device holds remain.

## Committed

### `b3c5a532d72a` — unified model, config and device run frontend

Related source commits: `bb089823653f` (core-run-policy), `136a4a8692d9` (facade-cpu-routing-prepare), `c4cbff17e6c4` (vendor-cuda-device-observation), `2bd8167db2c1` (cli-features).
Aggregate source21 qualification applies to the assembled tree; no per-commit
build or general model/backend grant. Historical failures keep their source scope.

#### Reviewed source checkpoint — 24 Oct

- [x] Current source21 assembled CPU/strict/release/physical23/full2Bmath and
  both-model API/HTTP/C1/C4/C8 pass ROOT review. Compiler counts stay448/.8B
  and396/2B from cold through first requests and whole measured waves. Original
  source20 first-use refusal stays failed; no broad CLI/model parity claim.

- [x] ROOT reviewed the complete CLI source unit plus final cache/HTTP/family
  seams and Pi corrections; fresh-byte merged34 owned paths into main. Preserve
  newer source15 cached2B fault fixture, unrelated changes and root README.
  Cargo.lock receives only the two intentional CLI dependencies, contextually.

- [x] Typed run/config/XDG/aliases/source selectors, per-field origins, strict
  graph/list overrides, context-free inventory/current-memory policy and original
  CPU/finite-CUDA adapters implemented. Original source/state/physical owners
  reused; unsupported graph/split/NPU/remote/backend execution still refuses.

- [x] Pi review corrections: scalar `[]` cannot bypass validation; declared
  empty targets fail before source access; explicit `catalog:` and existing
  default pull-reference namespaces resolve as catalogue identities. Config
  sidecars retain the approved per-field override semantics.

- [x] ROOT independently checked actual affected logs: cache CLI56PASS,
  acquisition CLI115PASS/1ignored, core204PASS, strict/fmt/whitespace PASS.
  Genuine generated GGUF source/alias/YAML/XDG/provider/catalogue equivalence
  executed. Earlier733 core/artifact checks passed; no learned-model grant.

- [x] Isolated complete source17 gate:1793 Rust PASS/0failed/89ignored plus
  installer/script checks. ROOT verified full logs and unchanged source bytes.
  This establishes CPU/build contracts, not current GPU inference or parity.

- [x] Source18 merges five unchanged item-order fixes; assembled CUDA all-target
  strict PASS. Main and CLI build root match across all1642 source files.

- [x] Source18 strict and original optimized build pass; actual emitted CLI
  frozen and used below. Preserve the initial artifact-selection error and the
  later unnecessary rebuild interruption separately from the successful build.

- [x] Source18 real CUDA CLI four-case unit: .8B direct/alias and2B direct/XDG
  default run. All canonical47/context512/output8 decoded texts match stock;
  complete source before/after, known close, groups/reap and both EOF pass.
  ROOT independently inspected all four records/raw outputs. No exposed-token,
  full-state/logit, zero-hot-compiler or performance grant from this CLI unit.

- [x] Current source20 full CPU gate1793PASS and combined all-target strict PASS;
  shared Qwen/Gemma source and standalone-feature correction integrated.

- [x] Current source20 optimized CLI/N1/B4 build and independent ROOT artifact /
  complete source review pass.

- [x] Fresh current physical23 source/state/cancellation/fault/owner unit PASS,
  including both model matrices and2B genuine trajectory export.

- [x] Current source20 full2B raw numerical composition PASS; ROOT independently
  verifies53raw members and complete8112section tuple with authenticated original30.

#### Review and authorization

- [x] User explicitly resumes implementation of the designed run CLI and asks
  for checkout/subagent execution in parallel, followed by merge into main24Oct.

- [x] Coordinator schedules a Sol HIGH isolated lane and adds this active plan;
  ongoing2B/HIP/model work proceeds independently.

### Phase 2.1 UX-1 — shared presentation (e6211608, 25 Nov)

- [x] Single dependency-free terminal policy and rich/plain stderr renderer,
  color auto/always/never and NO_COLOR, quiet/verbose/diagnostic events,
  line-oriented progress auto/always/never and typed failure codes/hints.
  Borrowed formatting; skip the stderr lock on suppressed events.
- [x] Route existing run, pull/cache, serve and diagnostic presentation
  through the same reporter. Preserve primary stdout, explicit machine
  diagnostics, original error detail/code/status, exit 2 on command
  failures and independent CLI feature sets. Actionable partial/durability
  warnings remain visible even in quiet mode.
- [x] Leading presentation flags only before the existing subcommand;
  never scan or rewrite subcommand values. Global format auto/text only,
  existing explicit JSON/JSONL/explain/audit unaffected; invalid values
  and unknown top-level commands do not echo raw argv.
- [x] Focused parser/renderer/subprocess tests and existing synthetic GGUF
  CPU run output equality. Actual release PTY (rich, NO_COLOR, dumb,
  plain, forced color) and quiet device JSON pass. Full mandatory-private-
  network workspace gate exits 0: 2431 Rust executions pass, 0 failures,
  106 explicit skips; architecture/feature/Clippy/release/installer/scripts
  pass. No execution/physical/performance scope change.

### Phase 2.1 UX-2 — typed command tree and shared resolution (25 Nov; bcdb46a8)

- [x] Typed canonical hierarchy, global grammar, concise help/command help,
  compatibility routes, generated static Bash/Zsh/Fish/PowerShell completion.
  Globals at known grammar boundaries, original provider argv opaque, offline
  restriction forwarded before owner option values. No new dependency.
- [x] Shared core SourceSelector exact digest and explicit by-name reference;
  original CacheMetadata resolves exact canonical identity, no projection
  following, no implicit network. Existing source/device/lease checks intact.
  SDK DocumentSelection and original typed YAML overrides select one
  default/explicit/profile document (no layers). Observed-device inspect
  and run use the same resolver.
- [x] Canonical model list/inspect metadata-only, verify full original lease,
  exact removal requires dry-run or yes; static family intent search; explicit
  acquisition and advanced alias/import forward original owner grammar.
  JSON for model reads, device reads and run dry-run; unsupported structured/
  interactive operations refuse. Existing machine outputs unchanged.
- [x] Isolated full gate 2451 Rust executions pass, 0 failures, 106 explicit
  skips; original source/feature boundaries, strict Clippy, release/installer
  and script checks pass. Synthetic CPU profile/digest/alias output equivalence,
  cache verification/metadata immutability, corrupt-projection selection and
  refusal tests pass. Actual release Bash/Fish completion and cold JSON
  smoke pass; Zsh/PowerShell generation only. No new physical/numerical claim.

### Phase 2.1 UX-3 — shared ControlPlane (3cf1bb2d, 25 Nov)

- [x] Shared public SDK control state per Runtime clone family. Original
  RuntimeSnapshot, cold device/cache/config views, native typed actions
  and original core/service errors; no new owner or runtime.
- [x] Model load/load-declared, exact-registration invoke/unload under
  original lifecycle lock, task inspection and original cancellable output.
  Bounded 64-entry transient event ring, independent subscriber cursors
  and explicit resync. Notifications invalidate observations, not authoritative
  history; no prompt/output/credential retention.
- [x] One optional control-owned ServerHandle slot with original observed
  bind/metrics and exact-generation stop/restart; independent SDK serving
  remains independent. Canonical model/cache and device CLI operations
  use shared methods; retain JSON and original verified/reclaim contracts.
- [x] Actual approved local Qwen 0.8B GGUF release integration (8.80s):
  two control views, one alias-shared owner, equal direct/control output,
  loopback /v1/models, duplicate start/stale stop refusal, stop preserves
  loaded model, unload/reload and stale registration refusal. Cold SDK
  refusal and subscriber overflow/resync tests pass. Full isolated gate
  2457 pass/0 fail/108 explicit skips (+1 focused model integration);
  architecture/feature/strict lint/release/installer/scripts pass.
  No new dependency, physical GPU qualification or numerical/performance claim.

### Phase 2.1 UX-4 — shared download lifecycle (25 Nov; da2385c7)

- [x] Shared per-runtime single-operation view, exact ID actions and
  invalidation events. Original synchronous prepared-plan executor,
  cancellation, resource/cache owners and typed publication result.
  Pause-request/paused/cancel-request/terminal status distinguished.
- [x] Exact proper-prefix strong-validator HTTP checkpoint/drop-response/
  park/rehash/range reopen; native Xet and unsuitable prefixes defer to
  safe file/pre-publication boundaries. Cold partial records text/JSON.
- [x] Canonical explicit online HF/GH/Xet pulls share plain/rich/quiet/JSONL
  progress and receipts; Linux foreground TTY Space/C/Ctrl-C with
  restoration. One scoped worker when needed, no extra quiet worker.
  Confirmed cancellation exit 130. Offline/dry-run/unsupported forced
  modes refuse before credential/cache IO. Existing rustix features only.
- [x] Full isolated gate 2459 pass/0 fail/109 skips; exact range recovery,
  separate two-client SDK control integration, five actual isolated
  monitor fixture runs (normal/quiet JSONL, PTY resume/C/Ctrl-C and
  restoration) pass. No network download, new qualification or
  numerical/performance claim.

### Phase 2.1 UX-5 — one-line run (25 Nov; 8e382947)

- [x] Shared exact cached model selection: flat lowercase name resolves
  canonical/human durable claims (different digests refuse), then approved
  catalog default only if absent. Explicit pin/selector and provider
  ambiguity retained; no broken-binding repair or symlink following.
  Model management and CPU preparation reuse this SDK resolution.
  Direct run binds digest before execution; original native Qwen route
  identifies approved source by digest, not alias.
- [x] One quoted positional prompt, --prompt-file, stdin and explicit/config
  prompts. Preserve option-shaped values via --, precedence and refusal;
  dry-run avoids stdin. Actual model/backend/device status on stderr.
  TTY no-prompt reuses one prepared CPU model/native Qwen runtime for
  independent requests; /help, /model, /new, /exit, EOF. No conversation
  history/persistence; finite diagnostic and repeated audit modes refuse.
- [x] Explicit --pull or affirmative eligible TTY offer for exact original
  2B/27B catalog recipes (or matching immutable HF locator). Original
  inventory members, existing disk/admission policy, shared download
  operation and required final manifest digest before original publisher.
  Legacy 0.8B manifest recreation, unsupported routes, floating/unregistered
  sources and incompatible overrides refuse; no implicit quota increase.
  Offline and dry-run never acquire; cached/local source no-op.
- [x] Full isolated gate 2465 pass/0 fail/109 skips; strict Clippy, feature
  boundaries, release/installer/scripts pass. Final combined acquisition/
  serve lint and CUDA/HIP/acquisition build pass; synthetic CPU prompt/
  cached alias equality and conflict refusal, exact recipe and expected
  digest guard pass. Real existing Qwen 0.8B GGUF release PTY two prompts
  equal quiet one-shot output, one startup, /model, /exit, no cache
  creation. First full gate test-only clone lint fixed and rerun passes.
  No new package/lockfile, remote acquisition, physical GPU, numerical or
  performance qualification.

### Phase 2.1 UX-6 — stable serve (25 Nov; 84e0681c)

- [x] Canonical positional/explicit aliases and selected document/profile
  use shared run/source/config resolver. Typed serve section overrides
  shared defaults, CLI overrides first; explicit routes replace model
  table. Legacy explicit local flags/YAML retain original profile semantics.
- [x] Resolve all routes before acquisition; CPU GGUF and single approved
  original HF source alias sets use original model owners, exact cache pins,
  admission and eligible consent recipe. No mixed/distinct original HF or
  canonical GPU GGUF grant; legacy GPU forms unchanged. Common CUDA
  preflight requires ordinal 0 and observed availability after headroom.
- [x] Original control-owned listener, actual startup/status and exact
  listener-then-worker shutdown; active acquisition shares shutdown flag.
  No HTTP rewrite or new authentication policy.
- [x] Focused release tests pass (2/0/0, 8.39s): actual approved Qwen
  0.8B GGUF cached aliases share one owner and identical completions,
  SIGTERM joins and closes listener; incompatible routes refuse cold.
  Combined strict acquisition/serve lint and CUDA/HIP/acquisition/serve
  compile pass. No live remote transfer or physical GPU run.
- [x] Full isolated workspace gate 2466 pass/0 fail/109 skips;
  release/installer/scripts pass. Two stale CLI test expectations adjusted
  for supported serve interaction and missing default XDG document;
  full rerun passes.

### Phase 2.1 UX-7 — configuration and profiles (25–26 Nov; 7f5fc4eb / abe98b83)

- [x] Single typed v2 document (ConfigDocument alias), selected default/
  named/explicit file with built-in fallback only for absent implicit default.
  Actual UI and download offer preferences; CLI overrides including explicit
  auto. Direct positional source selects fresh route; targeted overrides
  retain original pin contract. No implicit transfer, inheritance or pointer.
- [x] SDK cold settings validation, mapping-key edits, provenance, profile
  inventory and copy preserving relative source meaning. Atomic replacement
  with validated candidate, original-byte stale compare, cooperative lock,
  no-clobber create, same-directory publish and Unix sync; temporary editor
  draft, no symlink mutation. Typed edits normalize YAML; raw edit preserves
  draft bytes. No dedicated credential fields; stored HTTP query/userinfo/
  fragment refused.
- [x] Complete config/profile CLI and static completions, text/JSON reads,
  basic line-based interactive managers, explicit profile removal consent.
  New SDK/CLI integration and actual PTY smoke pass; combined lint and
  CUDA/HIP/acquisition/serve compile pass. Workspace gate GPU boundaries
  and CPU Clippy passed before a pipeline error-category regression;
  corrected minimal SDK/CLI tests 36/0.
- [x] UX-7 acceptance: corrected full isolated gate 2473 pass/0 fail/
  109 skips, architecture/feature/release/installer/scripts pass; combined
  strict acquisition/serve Clippy. Serial combined release run_unified
  6/0/0 (14.43s), real cached Qwen 0.8B GGUF serving passes unchanged
  8 GiB budget after earlier build-pressure refusal.

### Phase 2.1 UX-8A — native terminal shell (26 Nov; 08e7e9d3)

- [x] Bare capable TTY launches read-only shell; plain/quiet/non-TTY
  retain help, explicit commands stay direct. Selected ConfigDocument
  preference/CLI precedence. Frontend-only optional tui feature default/
  all and Nix release; single Ratatui/Crossterm stack resolved offline.
- [x] Nine destinations, responsive rail/narrow nav overlay, per-view
  scroll/focus, destination palette/help and scoped terminal restoration.
  Original ControlPlane snapshot/event cursor and cold views, no duplicated
  owner or implicit effect; bounded local notifications, coarse active
  observations. PTY quit/SIGTERM/resize/fallback and render/reducer pass.
  Terminal-only and combined strict lint pass. Full isolated gate 2477
  pass/0 fail/109 skips; actual optimized release PTY, installer/scripts
  pass. Nix cross-target builds not run. No new inference authority.

### Phase 2.1 UX-8B — native model/listener actions (26 Nov; 0c7ad87a)

- [x] Shared resident planner used by standalone canonical serve and TUI:
  original exact source selection/device/host/native route policy and
  original ControlPlane load. Reuse Listener transport/auth policy. No new
  inference route, download or implicit fallback.
- [x] Home load/quick-run, Running exact registration selection/invoke/
  confirmed unload, Serve actual start/stop exact generation. One runtime,
  one owned action worker, bounded attributed output copy, original
  stream cancellation, explicit process quit retires listener then runtime
  after worker join. Bracketed paste/form input ownership, no second queue.
- [x] First serial optimized combined run_unified 6/0/0 (24.59s), actual
  cached Qwen 0.8B TUI/HTTP lifecycle unchanged 8 GiB envelope. Minimal
  and combined strict lint, CUDA/HIP/acquisition/serve compile pass.
  Final reducer 3/0, strict lint, five other release tests pass; real
  repeat-invoke/paste PTY/HTTP rerun passes (35.16s) at existing 60s HTTP
  read bound after shorter fixture timeout under host contention. Plain
  shell PTY pass. Full isolated gate passes (2,479 passed, zero failed,
  109 ignored), release PTY/installer/scripts pass. UX-8B accepted.

### Phase 2.1 UX-8C — native model/download/device management (26 Nov)

- [x] Exact cached snapshot actions (inspect/verify/load/run/serve/
  non-replacing alias/preview/confirmed reclaim), static intent catalog,
  partial observations and same pinned intent for resume. Original cache
  owner preserves busy/referenced/durable/incomplete semantics.
- [x] Physical device observations, exact selected device cold planner
  check or explicit load; one coalesced read worker off presentation
  loop, existing effect worker. No duplicate cache/registry.
- [x] Explicit full commit HF/GitHub file or quant intent, approved Qwen
  recipe, scrollable consent, no auto load/quota increase; bare offline
  preserved. Original hub/recipe transfer shared with CLI; same runtime
  download operation, live observed progress/phase, exact id pause/
  resume/cancel and terminal durability. Advanced providers remain CLI.
- [x] Focused reducers 4/0, native intent 1/0, actual controlled transfer
  PTY range pause/resume/durable and cancel pass, strict minimal/combined
  lint pass. Actual cached Qwen lifecycle and remaining release tests pass;
  full isolated gate 2,481/0/109, release PTY/installer/scripts pass.
  UX-8C accepted.

### Phase 2.1 UX-8D — native configuration/profile/log management (26 Nov)

- [x] Native profile create/open/copy/activate/remove and searchable grouped
  settings with provenance/help, typed edits/unset/validate and raw YAML
  view/editor. Exact observed ConfigFile snapshot across editing/
  confirmation, original validation and stale/atomic owner. No implicit
  activate, delete default/portable/active profile or mutate admitted owners.
- [x] One coalesced read worker for all cold views; original action worker
  handles settings mutations. Shared resolved-path read, process-local
  active selection, preserve offline and CLI color precedence. Bounded
  local operational action/event ring with level/component/search and
  pause/follow, original correlation ids, no model output or raw drafts.
- [x] Focused TUI 5/0, strict minimal/combined lint, actual settings PTY
  invalid/stale refusal/raw repair/activate/copy/delete/log filters pass.
  Serial release cached listener/profile-switch passes (22.95s), remaining
  release tests pass. Full isolated gate 2,483/0/109, release PTY/installer/
  scripts pass. UX-8D accepted.

### Phase 2.1 UX-9A — embedded Web shell (26 Nov)

- [x] Pinned locally cached SolidJS/Vite build, committed assets/license,
  frontend-only optional web feature in default/release/Nix. No Node
  runtime or newly resolved Cargo packages.
- [x] Shared native foreground host/runtime/active Selection. Explicit
  web command or native Home/Serve b private URL; no duplicate runtime
  or owner, client closure inert, process retirement joins.
- [x] Bounded independent loopback control bearer/Host/origin boundary,
  read-only allowlisted snapshot and SSE invalidations, exact wire IDs,
  no raw diagnostics/credentials/config paths; responsive shell/palette/
  auth/reconnect. Real HTTP/Firefox/JS and strict lint pass; serial
  cached TUI-attached Web release 6/0/0 (32.71s). Full isolated rerun
  2,485/0/109 and release Web/TUI/installer/scripts pass. UX-9A accepted.

### Phase 2.1 UX-9B — Web resident actions (26 Nov)

- [x] Original resident action worker/OutputStream promoted to one
  native/Web host slot; exact action identities, no queue, selection
  update before release, native ticket/private result, latest Web
  receipt, exact cancel and owned join.
- [x] Typed same-origin authenticated bounded POST, cached selectors/
  trusted document only; actual load/run/invoke/unload/start/stop,
  destructive confirmation and exact registration/generation.
  Solid quick-run/owner/serve forms and bounded progressive output/
  original usage; no implicit acquisition or inference-key exposure.
- [x] Strict lint incl Web-only CPU, focused owner/intent/TUI/JS,
  actual cold HTTP/Firefox pass. Real optimized cached Firefox and
  TUI-attached Web/HTTP regression passed (46.14s, five filtered;
  other five previously passed). Exact stale-owner refusal, unchanged
  8 GiB envelope, independent inference auth and joined retirement.
  Full isolated gate 2,489/0/109 plus release Web/TUI/installer/scripts.
  UX-9B accepted.

### Phase 2.1 UX-9C — Web model/download/device management (26 Nov)

- [x] Shared original cache/pull/device worker and cold observations;
  authenticated separate model/device inventory, exact digest inspect/
  verify/alias/reclaim and cached load/run/serve, pinned pull preview/
  consent, exact original transfer pause/resume/cancel, physical device
  and source policy check/load. No new runtime, queue, cache or policy.
- [x] Native/Web relevant owner invalidations incl terminal download status
  at emission, visible cold reads rather than progress scans. Web-only
  build remains terminal-free; strict lint/JS/cold HTTP/Firefox and
  actual controlled HTTP/native PTY/Firefox transfer pass.
- [x] UX-9C optimized cached integration passed 1/0/0 (48.13s, five filtered); full isolated gate 2,489/0/109 (318 Rust suites), release Web/TUI/installer/scripts passed. UX-9C accepted.

### Phase 2.1 UX-9D — Web profile/config/log implementation (26 Nov)

- [x] Shared original configuration observation and actions, exact snapshot
  validation/atomic write, active/default/portable protection, Web safe
  typed and normalized YAML policy, native unrestricted repair; shared
  bounded host log, safe action/owner correlations, cross-client refresh.
- [x] Strict feature/lint/dependency isolation, JS, actual HTTP, attached
  Web/native PTY cross-client settings and real Firefox smoke pass.

### Phase 2.1 UX-9E — simultaneous native/Web acceptance (26 Nov)

- [x] Native log view observes host ring revision, including fast Web-only
  validation while native is idle. Real attached settings PTY passes.
- [x] Native safe pause after accepted prefix, attached Firefox resume to
  durable or cancel to partial; exact original operation identity,
  range reopen, stale refusal and joined retirement. Combined release CLI
  50/0/3; actual two-client transfer scenarios pass.
- [x] Actual approved cached Qwen 0.8B GGUF optimized native/Web/HTTP
  integration 1/0/0 (54.16s): native policy edit, stale Web refusal, fresh
  Web restore; unchanged admitted scheduler, registration, listener and
  exact output. Original 8 GiB admission unchanged.
- [x] Full isolated workspace gate 2,491/0/109 (318 Rust suites), release Web/TUI/installer/scripts passed. Initial UX-9D gate interruption recovered; UX-9D/E accepted.

### Phase 2.1 UX-10 — cold diagnostics and compatibility (26 Nov)

- [x] Replace doctor stub with cold selected document/policy, cache metadata,
  physical observation, compiled-backend and obvious local source checks.
  Safe text/JSON; errors retain report and fail, warnings remain observations.
  Cache logical snapshot bytes and configured quota are distinct from physical
  quota charge. Unobserved runtime bindings remain explicitly unprobed.
- [x] Device inspect with optional source uses exact physical identity and
  original run dry-run; positional direct source supersedes configured model
  table/entrypoint while retaining selected policy. Audit writes and
  acquisition excluded from device check. Explicit legacy override preserved.
- [x] Pipeline list/inspect/check and positional file/name grammar reuse
  typed structural validation, topology and declared route/context metadata;
  general execution refuses. Stable JSON and compact normal help/completions;
  compatibility aliases remain. CLI guide and examples updated.
- [x] Strict minimal/combined lint; final isolated minimal CLI/pipeline
  27/0/0, optimized combined CLI/pipeline/run 32/0/1. Existing populated
  cache tree unchanged after doctor; dry-run retains original planner.
  Full isolated gate formatting/architecture and HIP/Vulkan compilation
  pass; intentionally stopped during production CUDA transport rebuild.

### Phase 2.1 UX-10 — final cold diagnostics and acceptance (26 Nov)

- [x] Original CUDA/HIP/RTC/Vulkan loader presence separate from binding
  and execution eligibility; uncompiled probes unknown.
- [x] CacheMetadata existing-only nonblocking quota observation; same
  physical inode charge and live reservations, original policy and locks,
  no writes/creation. Busy unknown; exact charge match and no cache-tree
  effects. Focused 44/0/2, strict minimal/cache/combined lint.
- [x] Complete isolated gate sequence passed 2,505/0/109 (318 Rust suites), release Web/TUI/installer/scripts. Original run stopped at two transient busy-observation fixture assertions in acquisition/cache transactions; test-only bounded busy retry, reran affected cache-only/diagnostic/acquisition suites and exact remaining gate commands. No production change after the initial run.
  UX-1–10 accepted. Additional UX-P user priority below precedes serving.

### UX-P — native TUI interaction and visual polish (26 Nov; accepted at ad7a4178)

Design: [TUI polish and mouse interaction](../cli-ux/TUI_POLISH_AND_MOUSE_INTERACTION.md).
Retain accepted UX-1–10 ownership and keyboard behavior.

- [x] UX-P1: semantic palette, restrained chrome, Dark/Light/Auto,
  typed appearance preview/save, safe terminal depth/Auto, Web parity.
  Focused Rust 81/0/3, JS 3/0, strict minimal/Web-only/combined lint,
  real PTY and Firefox settings/appearance passed. Native Auto hint or
  dark fallback; no OSC probe. Full gate at assembled UX-P4.
- [x] UX-P2: scoped mouse capture/restore and ui.mouse, exact visible
  inert row/shelf/nav hit targets, focus, rail/inner split drag, maximize/
  restore, independent wrapped scroll/scrollbars and narrow navigation.
  Strict minimal/Web-only/TUI-Web/combined lint; focused 84/0/3 and actual
  SGR/native/Firefox regressions pass. Stable data-reorder anchors and
  full editor/action interaction remain P3/P4.
- [x] UX-P3: mouse-accessible controls in all nine views, actions,
  modal/editor/picker interaction, exact owner/document and confirmations.
  - [x] UX-P3A: width-stable action bars/More/F4, exact observed identity
    binding and fresh dispatch revalidation, modal buttons and prompt
    mode, busy draft retained. Focused Rust 82/0/3, strict lint, real
    native control/settings/appearance/mouse PTY, actual controlled
    download mouse and approved cached Qwen browser/native/HTTP 1/0/0
    (52.81s) pass.
  - [x] UX-P3B: typed setting/log pickers, local model search, grapheme
    caret/selection and editor viewport. Strict lint, focused Rust
    86/0/3, actual editor/choices and stale refusal, original PTY,
    real cached lifecycle 1/0/0 (55.96s) passed. Existing locked Unicode
    packages now direct optional TUI dependencies; no version change.
    Remaining compact/modal/focus and view affordance review at P4A.
- [x] UX-P4: complete ergonomics, actual PTY and accepted real cached
  resident/transfer/config regression, isolated workspace gate.
  - [x] P4A1 screenshot-driven minimalism: curated toolbar (max three
    actions + More), no repeated rail/shelf/theme actions, compact
    summaries and exact Details/D, human IEC sizes. Strict lint,
    focused 87/0/3, actual rendered cells/native PTY and controlled
    transfer pass.
  - [x] P4A1 follow-up: no toolbar, selected-row context menus/right-click,
    inline live local search across list views, session footer menu,
    explicit Alt shortcuts. Strict lint, optimized Rust 88/0/3, actual
    rendered cell/native/context/search/settings and transfer PTY pass.
    Actual cached Qwen browser/native/HTTP 1/0/0 (57.77s), unchanged
    admission and exact output; shared Firefox/native transfer outcomes pass.
  - [x] P4A2 stable list/shelf/log/output anchors, separate Running
    response viewport/F3, safe exact log detail, compact scrollable
    dialogs and palette, editor edge selection. 18 focused TUI tests,
    optimized 91/0/3, strict lint, 20x7 native PTY and all native/
    controlled and simultaneous Firefox transfer regressions passed.
  - [x] User follow-up: full production Cargo CLI default (all current
    models/CPU/CUDA/HIP/Vulkan/acquisition/serve/TUI/Web), explicit
    minimal/portable CPU profiles retained. Accurate original-source
    eligibility errors and typed service/core translation; measured
    resource/I/O causes preserved. Safe HTTP/SSE default, authenticated
    opt-in details and safe native/Web operational log metadata.
    Core/engine/serving, SDK, CLI/config/pipeline focused regressions and
    full default cargo check passed; final full-default strict lint below.
  - [x] UX-P-M1: registration-owned measured live TPS/TTFT and bounded
    client-local trend, exact selected performance details and reported
    layer execution/lease bounds, native and Web. Focused engine 109,
    CLI 64/0/3, JS 4, strict lint/default check; actual cached CPU/
    Firefox/native/HTTP 1/0/0 (55.61s), exact counters, CPU layer map
    and reset on re-registration. CUDA placement later qualified below. See
    [live performance design](../cli-ux/LIVE_MODEL_PERFORMANCE.md).
  - [x] UX-P-M2 user-directed Running visual refinement: larger detail pane
    by default (preserve explicit split), responsive metric pair, richer
    recent output chart, exact segmented layer strip/device legend, admitted
    reader activity and direct exact-owner details link. Native 20/0/0,
    JS 5/0, strict lint, actual cached native/Firefox/HTTP 1/0/0 (53.65s),
    unchanged 8 GiB admission and exact output. No inference-path changes.
  - [x] P4B comprehensive acceptance: actual combined owner regressions
    passed (CPU CLI 6/0/0, cached listener SDK 1/0/0, real native/shared
    Firefox transfer durable/cancel, JS 4/0). Full production linked
    launch/version/help/doctor and TUI/Web smoke passed. Explicit old
    CPU/cache and acquisition selection preserved; separate default
    strict lint passes after feature-intersection test helper fix;
    Web-only gate added. Original physical CUDA SDK
    admission/retry 1/0/0 with exact 24-layer admitted-device map,
    registration counters and joined retirement; HIP map not requalified.
    Portable Nix/release remains explicit CPU. Remaining complete
    isolated workspace gate retry stopped at two cold_server typed refusal
    assertions (1191 passed, 2 failed, 56 ignored). Both corrected to the
    exact existing InvalidInput/Unsupported categories and messages;
    focused cold_server 4/0/0. M2 assembled retry passed full-default
    strict lint, GPU feature-boundary tests 26/0/2 and CPU dependency
    isolation. Acquisition-disabled TUI format import fixed; exact CPU
    workspace strict all-target lint passes. Isolated CPU test-stage retry
    timed out during compilation (90s budget, no suites executed). Final
    assembled offline private-network gate on ad7a4178 exited 0 (592.3s):
    2,539 passed, zero failures, 109 explicit skips across 318 Rust suites.
    Strict lint, feature boundaries, release Web/TUI/installer/scripts pass.
    See [live checkpoint](../../docs/CURRENT_HANDOFF.md) and
    [takeover context](../../docs/FULL_COMPREHENSIVE_HANDOFF_2026-11-26_UX_PM1.md).


- [x] User follow-up: model-local Execution selector in native and Web cached
  Models. Default inherits the active profile; explicit backend can select an
  automatic or exact observed device. Run/Load/Load for Serve share the draft;
  no profile writes, cross-model override, or new routing owner. Original
  fit/refusal remains authoritative; stale explicit device does not fall back.
  Focused 67/0/3, JS 5/0, real cached CPU lifecycle 1/0/0 (57.08s), full
  assembled private-network gate 2,543/0/109 (318 suites, 301.6s), release
  native/Web/installer/scripts passed. No new GPU execution claim.

UX-P accepted; public serving is the next ordered phase. No physical
performance comparison or new HIP/Vulkan placement qualification.

## Uncommitted

### UX-11 — user-priority native presentation and diagnostics (28 Nov)

Design: [TUI polish follow-up](../cli-ux/TUI_POLISH_AND_MOUSE_INTERACTION.md).
- [x] A: explicit search focus, bare numeric views, retained filters,
  compact brand/profile rail, narrow Nav; mouse/actions and tiny
  terminal preserved.
- [x] B: exact-source task observations in selected Models pane,
  original verification bytes/files, shared download counters,
  requested/resolved backend, runtime registration/device, last
  observed native result; distinguish missing weight role.
  Safe Web action wire carries stage/counters, not native error text.
- [x] C: typed action failure propagation and local/public boundaries;
  timestamped mode-aware operational sink, real verification progress,
  bounded private retention, foreground stderr/journal, explicit
  file/format/level. Actual L5 source/runtime/capacity reproduction,
  configured CUDA original 0.8B real generation and retirement passed.
  Focused process, Rust, strict feature-specific Clippy and Web gates
  passed; assembled process regression wired and passed.
  Full assembled gate remains open. See current handoff for exact scope.
- [x] D source-aware guidance: shared manifest layout and canonical
  route classification; TUI/Web compatible choices and structural
  action refusals; actual planner differentiates missing payload,
  unsupported format/consumer and unavailable runtime. Auto checks
  CUDA compiler/binding and exact HIP PCI before selection; native
  numerical policy requires HIP. No post-selection fallback or
  metadata-derived admission.
  Actual L5 Web inventory, missing/unsupported source refusals,
  unavailable plain GPU bindings, configured HIP exact resource
  refusal and configured Auto->CUDA original 0.8B generation passed;
  native PTY/Web, focused integration and strict lint passed.
- [x] D native Nix launch contract: x86-64 Linux default uses Cargo
  defaults, portable CPU packages separate; shared pinned GPU SDK
  paths/header view/library closure, matching cudarc SDK version,
  normal wrapped launch. Actual derivation evaluation passed.
- [ ] D remaining package qualification: realize pinned native
  package and exercise ordinary packaged startup/physical load.
  Missing CUDA12.9/ROCm7.2.3 and build closure (3457 derivations in
  offline plan); no lock changes or implicit acquisitions.
  Complete assembled gate and return to measured P3D.

### Other deferred work

- [ ] Optional deeper exact per-request/phase and individual layer
  observations; keep profiling explicit and capability-bound.
- [ ] Optional terminal background helper only with dependency/offline
  approval; conservative Auto fallback must remain truthful. No additional
  theme catalog or persistent layout store now.

- [ ] Broader catalog presets, conversation history, dedicated graph
  visualization and unsupported routes remain explicit future scope.
  Existing direct text/JSON structural inspection is accepted.
- [ ] Extend advanced direct provider forced interactive/JSONL presentation
  and aggregate component download control without changing independently
  published receipts; existing direct provider progress and legacy sequential
  component executor remain valid. No general concurrent download scheduler,
  persistent event history or automatic partial deletion introduced.

### Reviewed source checkpoint — 24 Oct

Historical source20 .8B first-use refusal (two attempts after readiness) remains
recorded. Current21 repairs it within the finite accepted scope above.
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

### Review and authorization

- [ ] Recheck live source, ownership and accepted model/backend evidence before
  implementation; preserve root README, other plans and active implementation.

### Explicit source-form addition — 24 Oct

- [ ] Accept local paths, catalog/model names, `hf:`, `gh:`, `http:` and `https:`
  through one shared source selector and the existing acquisition/cache identity
  contracts. Recognized source schemes take precedence over local YAML suffix
  classification; only a local `.yaml`/`.yml` selector implies a run document.
- [ ] Resolve already acquired immutable exact sources, reject ambiguous variants
  and preserve explicit acquisition guidance when absent. URI parsing/resolution
  is not inference qualification. Reuse provider parsing/source identity rather
  than a second downloader or format-specific run loop; dry-run stays read-only.
- [ ] Every new CLI semantic option has a typed v2 field. Unit1 implementation
  and tests precede ROOT guarded fresh-mtime merge; final assembled gate remains
  required. Future split/NPU/remote units retain separate qualification.

### Proposed unit 1 — useful local run policy

Deliver one local command that selects a qualified whole-model CPU/GPU route,
explains exclusions and safely refuses without silent backend/source changes.
This unit does not claim automatic layer splitting.

- [ ] Normalize `run`, backend aliases and common task flags into one library
  request; preserve compatible existing invocations and actionable diagnostics.
- [ ] Extend the canonical pipeline document/compiler to proposed v2 with one
  model table, explicit model/pipeline entrypoint and typed run policy/task/
  diagnostics; do not introduce a separate config interpreter or runtime.
- [ ] Implement unambiguous positional model/config and explicit selectors;
  selector-free run loads only absolute-XDG/HOME `abird/zerogravity/config.yaml`.
  Missing/unreadable/invalid config fails without fallback or initialization.
- [ ] Implement CLI > selected config > built-in resolution with per-field
  provenance, origin-correct paths, whole-list replacement and explicit boolean/
  zero/empty semantics. Preserve alias equivalence and final constraint checks.
- [ ] Migrate model metadata to `--model-config`; cover every existing task,
  tokenizer, teacher-ID, prefill and audit option with typed config equivalents.
- [ ] Retain bounded strict YAML/graph validation; check the complete merged
  document before effects. Declared routes remain inert under run; graph and
  recursive execution still need their independently qualified implementation.
- [ ] Resolve exact local source and bounded task/model requirements through
  existing adapters; catalog/metadata status grants no executable capability.
- [ ] Add read-only inventory and stable physical identity resolution; derive
  actual vendor/backend support and deduplicate cross-backend aliases.
- [ ] Implement strict ordered allowlists, CPU-terminal semantics, local-only
  auto and all-neural-stage forced-backend checks before preparation.
- [ ] Implement current-memory observation with explicit provenance/age/unknown
  status, OS/cgroup constraints and host/UMA domain relationships. Do not reuse
  `max_memory` as free memory or mutate established operator ceilings.
- [ ] Resolve a safe default operator-ceiling/bootstrap policy once for a new
  process before context effects, so bare `run` needs no manual resource caps;
  preserve established domains and benchmark/serving controls.
- [ ] Integrate complete cold/prefill/decode/native/compiler/state/output costs
  with original ledger snapshots and authentic existing backing/reuse. Close or
  expose pre-admission context-bootstrap gaps; unknown costs cannot claim fit.
- [ ] Implement configurable headroom, all-GPU full-fit scan and qualified CPU
  fallback; explain missing split support when a model would otherwise spill.
- [ ] Revalidate and admit through original owners; bound safe startup replanning,
  preserve unknown-work fences and retain all submitted owners.
- [ ] Provide `devices list`, `--dry-run`, `--explain` and structured refusals
  without hidden execution/network/cache mutation during diagnostics.
- [ ] Validate the assembled unit on approved source/hardware/backend tuples;
  document actual supported tuples and unavailable current-resource probes.

### Proposed unit 2 — local layer spill with real state

Deliver one source-qualified model that cannot fit on a single allowed GPU but
runs through admitted legal local groups and, when allowed, CPU remainder.
Coordinate with existing memory/distributed/composition owners, not a second
implementation of their allocator or task engine.

- [ ] Define source-specific indivisible groups and embedding/head/shared-weight/
  MTP ownership; produce one complete feasible placement before any preparation.
- [ ] Extend original arbiter with all-or-none multi-domain admission if needed;
  collapse aliases and UMA domains, preserve prepaid-envelope/sublease rules.
- [ ] Integrate original state shards, typed activations and actual event/reader
  transfers into the existing prepared physical plan and fair workers; join all
  Ready events before the original SessionAuthority publishes one logical root.
- [ ] Prove same-backend multi-GPU and GPU↔CPU compositions separately before
  enabling each; qualify mixed CUDA/ROCm/Vulkan only with its own evidence.
- [ ] Verify full-logit/complete-state following-token parity, cancellation,
  partial preparation, transfer failure, unknown work and complete retirement.
- [ ] Compare full CPU and each feasible single/split route with complete cold/
  warm request costs; show capacity enabling separately from speedup.

### Proposed unit 3 — qualified roles and explicit peers

Future separately authorized work, not prerequisites for local command delivery.

- [ ] Introduce semantic stage-role constraints without model-family CLI routing
  branches; preserve full-run forced-backend policy and common allowlist.
- [ ] Qualify actual trained draft/MTP source and recursive state semantics;
  distinguish native MTP depth from supported proposal length.
- [ ] Add an NPU provider only through existing region/state/lifetime boundaries;
  prove end-to-end benefit versus available GPU draft and no-speculation routes,
  including transfer/verification/rejection/shared-domain costs.
- [ ] Resolve explicitly enrolled peer aliases to authenticated identities and
  capability/topology epochs; standalone auto performs no remote discovery.
- [ ] Coordinate bounded all-or-none holds through original per-host ledgers;
  implement authenticated transfers and exact state shard publication/fencing.
- [ ] Qualify first two-host contiguous layer/local-state composition on approved
  sources/hosts; report directional network/copy/crypto/queue costs and loss.
- [ ] Keep remote inference APIs that hide region/state contracts outside native
  device routing; no SSH/shell or implicit artifact downloads for placement.

## Tests and verification

Each scenario checks its own semantic and cleanup results. Mocks prove policy,
not physical availability, inference, state transfer or performance. Use actual
approved sources/devices for every runtime claim; respect the gfx90c hold.

- [ ] **Syntax equivalence:** canonical/alias/repeated flags normalize identically;
  conflicting backend, mixed `auto`, empty device, unknown host/ID and CPU before
  GPU refuse. Unsupported backend is reported without model preparation.
- [ ] **Default config:** present/missing/unreadable/invalid default; absolute,
  relative and unset XDG base; valid/absent HOME fallback; no cwd/parent search.
  Explicit model/config/graph skips default overlay; help/version do not load it.
- [ ] **Config discrimination:** YAML suffix, extensionless explicit --config,
  --model, odd model filename, malformed/unsupported schema, multiple positional
  selectors and old model-JSON --config produce the specified result; no model/
  config fallback, automatic pull, initialization or side effects.
- [ ] **Config/CLI parity:** direct YAML and flags yield equal typed requests;
  graph YAML and declarations share one instance table/compiler. Source metadata,
  tokenizer, prompt/context/output/raw/teacher/prefill/audit/numerical/speculation/
  role/device/diagnostic fields and semantic validation round-trip equally.
- [ ] **Precedence/provenance:** CLI scalar wins, alias equals canonical backend,
  default only fills absent values; false/zero/empty/auto are preserved and
  validated. Repeated CLI devices replace config list as a whole; graph list
  replacement/clear detects dangling references and duplicate IDs.
- [ ] **Path bases:** config-origin source/sidecar/audit paths use selected config
  directory; CLI replacements use invocation cwd. Moving cwd, config symlinks and
  same relative spelling with different origins cannot silently retarget fields.
- [ ] **Config safety/entrypoint:** unknown/duplicate fields, expansion/depth/file
  limits, custom tags/includes/interpolation, missing/ambiguous entrypoint,
  contradictory final backend/device policy and legacy model-config spelling
  refuse before execution. Selected graph runs once; declared routes do not bind.
- [ ] **Identity:** CUDA/Vulkan aliases of one card count once; AMD discrete and
  NVIDIA integrated classifications cannot invert vendor routing. Snapshot
  indices never survive reorder/restart without physical revalidation.
- [ ] **Forced scope:** CUDA+AMD rejects before allocation/compiler activity;
  forced GPU+CPU spill or incompatible NPU draft rejects; forced CPU runs every
  requested neural stage on CPU or refuses a missing capability.
- [ ] **Full fit:** both vendors fit→NVIDIA; NVIDIA insufficient/occupied and AMD
  fits→AMD; later listed GPU full fit precedes first-card split; GPU-only explicit
  list cannot reach CPU. NVIDIA unavailable/fenced is a precise exclusion.
- [ ] **CPU fallback:** no GPUs, missing GPU build, unqualified GPU model and no
  qualified split each reach CPU only if allowed, qualified and currently fitting.
  An unqualified model everywhere gives no invented inference route.
- [ ] **Current pressure:** large total VRAM with low free bytes does not fit;
  cross-process occupancy, OS/cgroup pressure, missing/stale counters and
  fragmentation produce conservative proposals and bounded allocation failures.
- [ ] **Accounting:** original leases plus actual backing plus unbacked promises;
  prepaid envelope and native partition; idle retained pools; authentic reuse;
  unknown/captured readers. No duplicate charge/free credit or cap enlargement.
- [ ] **UMA:** CPU/GPU/NPU alias one memory domain; cross-device views/transfers
  retain proper fences and charge physical backing once. Headroom applied once.
- [ ] **Complete peaks:** weights fit but prefill/KV/output/compiler/native/capture/
  draft costs do not; reject. Disjoint phase maxima are not blindly added; actual
  overlapped lifetimes, COW, transfer and old/new state are not omitted.
- [ ] **Headroom:** percentage/bytes/default/floor/overflow validation; resolved
  amount shown; reducing margin never raises a physical/operator ceiling or
  waives source/shape/native qualification. Bare-run bootstrap records a safe
  default once; later live-free changes never resize the registered ceiling.
- [ ] **TOCTOU:** external allocation and concurrent internal admission between
  proposal/admission/preparation; all-or-none holds and known-safe rollback;
  at most one fresh replan. Dry-run and cached plan never grant reservation.
- [ ] **Unsafe preparation/flight:** unknown module load/unload, transfer/capture/
  launch retains/fences original owners; no alternate-backend replay, partial
  root/output publication or reuse based on cancellation/timeout alone.
- [ ] **Split feasibility:** largest legal prefix leaves a feasible remainder;
  indivisible tail/tied embedding/head cases; GPU→GPU→CPU exact state and
  activation transfer. Unqualified mixed numerics refuse rather than retag.
- [ ] **Runtime equivalence:** real full affected logits/tensors, every required
  state domain, natural following-token continuation and task outputs match the
  selected numerical contract for each admitted source/profile/composition.
- [ ] **Fairness/lifetime:** overlapping requests/models preserve independent
  roots, original fair progress and complete reader retirement under pressure;
  successful cleanup cannot hide an earlier semantic failure.
- [ ] **Diagnostics:** list/dry-run leave model, GPU context/compiler, resource
  ledger, cache and network unchanged; report unknown rather than probe by
  execution. Explain identifies each rejection and final route without corrupting
  output or exposing prompts/secrets.
- [ ] **NPU role, future:** qualified NPU still loses when transfer/verification/
  rejection/queue costs dominate; wins only with applicable full-request benefit.
  Stale evidence rejects automatic speed claim; default auto selects ordinary
  decode without proven benefit, optional draft-device does not force drafting,
  and required MTP/draft/source features are not omitted.
- [ ] **Remote, future:** unknown enrollment, identity/epoch mismatch, denied
  source/data grant, partial holds, slow/relayed link, late Ready, lost ACK and
  uncertain DMA preserve original authority/leases; no implied discovery or HA.
- [ ] **Performance:** authorized paired whole-request comparisons, cold/hot
  preparation, TTFT/ITL and total latency, memory/IO/transfers and fair load;
  compare against CPU/single-GPU baseline and include slower-split negative cases.
- [ ] **Final assembled gate:** after applicable unit qualification, run
  `CARGO_BUILD_JOBS=2 RAYON_NUM_THREADS=4 python3 scripts/native-gpu.py -- bash scripts/check.sh`.
  This design-only draft does not run the gate or any GPU/model tests.
