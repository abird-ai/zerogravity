# ZeroGravity current comprehensive handoff — 26 Nov 2026

Canonical cross-window entry point for `/home/pvl/spaces/abird/src/zerogravity`.

**Latest complete takeover (26 Nov, through UX-P-M1):**
[FULL_COMPREHENSIVE_HANDOFF_2026-11-26_UX_PM1.md](FULL_COMPREHENSIVE_HANDOFF_2026-11-26_UX_PM1.md).
Read §§0–13 first, especially §10.2 (default/build/gate caveats) and §11
(next required P4B acceptance, optional M2). Use the
[standalone P4B takeover prompt](NEXT_AGENT_2026-11-26_UX_P4B.md).
Appendix D preserves all fifteen post-UX-9B landing narratives and
changed paths; Appendix B records the literal pending-marker snapshot.
The complete [UX-9B pack](FULL_COMPREHENSIVE_HANDOFF_2026-11-26_UX9B_EXIT.md),
[Phase 1 exit pack](FULL_COMPREHENSIVE_HANDOFF_2026-11-25_PHASE1_EXIT.md)
and preceding historical handoffs retain original design, exact
implementation chronology, experiments, rejected choices, reusable
primitives and all historical work items. Their old continuation
prompts are historical. This is a one-time requested navigation
snapshot; live source, tests, owning TODOs and ordered plan remain
authoritative. Resume from the live checkpoint below.

**24 Nov priority decision:** start with the
[full implementation plan](../plans/FULL_IMPLEMENTATION_PLAN.md). Complete
all shared foundations, including the existing closed-region/compiled-provider
scaffolding (no concrete graph SDK required). Then finish CLI/TUI/Web and
public serving; match vLLM on ROCm first, CUDA next, then native Vulkan.
Actual MIGraphX/TensorRT/IREE/ONNX Runtime integrations are a later optional
phase. Original Hugging Face checkpoints/Safetensors and GGUF remain the
primary source paths. The
[closed-region specification](../plans/pipeline-composition/REGIONS_AND_PROFILES.md#source-ingestion-semantics-and-executable-lowering)
separates verified ingestion, authoritative semantics and executable lowering.
Phase 1 supports direct lowering without ONNX; Phase 6 prefers programmatic
MIGraphX/TensorRT construction and qualifies optional graph import separately.
No additional canonical IR, conversion pipeline or compiler dependency in
Phase 1. Older next-step performance notes below describe
reusable work and measured history, not precedence over this order.

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

### Prior checkpoint — UX-P-M1 live model performance (26 Nov)

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

### Previous checkpoint — Phase 2.1 UX-9B (26 Nov; 4f23983a)

Browser Home / Running / Serve actions now use the native resident
planner, OutputStream and inference transport, one shared host slot.
No alternate model executor, action queue, runtime or server registry.

- control_host/work.rs is the original TUI Request/Owner/Prompt,
  resident resolve/prepare/load/invoke/unload/start/stop and cold
  action execution. Native tui/work.rs retains only ticket/log/detail/
  draft-retry metadata. Native-only settings/cache/download operations
  remain in the same worker until 9C/D promote those request variants;
  feature-minimal Web has no terminal dependencies. Existing
  MissingSource expected-snapshot cfg includes Web-only CPU routes.
- control_host/actions.rs: one shared Actions slot, monotonic exact
  action ID, origin, cancellation, original scoped JoinHandle and
  bounded response. Either native or Web action may own the slot; reject
  another rather than queue. Finished native worker is joined/consumed
  by its exact native ticket; Web reaps only finished worker and retains
  at most latest Web terminal result, independent of intervening native
  actions. Selection changes apply inside the shared worker before
  releasing admission. TUI active footer/guards observe both origins;
  native cancel targets exact observed action, Web cancel only its own
  origin. Host shutdown closes Web connections, cancels/joins action
  then retires original listener/runtime. Slot drop also joins.
- POST /zg/control/actions uses serde deny_unknown_fields tagged
  grammar, full same-origin Origin and application/json, independent
  bearer, bounded 256 KiB body read only after auth, original finite
  header+body deadline. No duplicate fields, chunked bodies, arbitrary
  argv/credentials/device or execution overrides. load/run_source
  accepts blank selected trusted document, exact cached name/digest/
  catalog/by-name or pinned HF/GitHub commit, optional valid ID=MODEL;
  explicit arbitrary filesystem paths and HTTP sources refused. No
  automatic acquisition. invoke/unload exact alias + registration,
  start numeric loopback or selected listener policy, stop exact
  generation; unload/stop require confirm=true. All IDs canonical
  decimal strings, never JS number. 202 accepted is not completion;
  busy/stale cancellation 409, malformed 400. Original runtime may
  refuse accepted action later.
- GET /zg/control/actions/ID observes only that active/latest Web
  receipt, running or terminal, safe error category, original bounded
  UTF-8 output prefix 64 KiB and actual final usage/finish reason;
  no native prompt/output, raw path/errors or credentials in general
  snapshots/events. OutputStream unchanged, UI samples only active
  Web receipt (200ms visible/1s hidden), SSE coalesced action lifecycle/
  selection and owner invalidations, no token event history. Explicit
  cancellation request remains distinct from terminal result. Browser
  retains last Web receipt across reload, no automatic POST retry.
- Solid Home quick load/run from cached/configured source or exact
  ready owner, Running selected owner prompt/retire confirmation,
  Serve start/stop with generation-confirmation; bounded response panel
  shows live text, actual usage, action status and cancel. Same
  loopback control credential, inference API key host-configured only.
  Cold libraries/config/log actions still native until UX-9C/D.
  New frontend sources resident.jsx and work-state.js, committed Vite
  output, same dependency graph.
- Focused strict TUI-only/Web-only/Web-only CPU/combined lint pass,
  Web parser/intent and shared-slot exact cancel/private native receipt
  regression pass, TUI reducers 5/0, JS 3/0. Actual cold HTTP POST/auth/
  body/identity/security/SSE and Firefox responsive typed refusal pass.
  Initial combined release other five tests passed. Cached lifecycle
  needed differential PTY status/endpoint assertion corrections; one
  rerun also correctly refused the unchanged 8 GiB envelope at only
  7,615 MiB available. Final isolated optimized cached regression
  passed 1/0/0 (46.14s, five filtered): actual Firefox resident
  load/invoke/output/usage, separately authenticated inference
  start/stop, unload, combined load/run and receipt reload;
  TUI-attached Web invoke/unload/load-and-run/start, stale registration/
  generation refusal, profile/owner invalidation and joined retirement.
  Full isolated workspace gate passed 2,489/0/109, release Web/TUI/
  installer/script checks. UX-9B accepted. No remote download/physical
  GPU/public exposure or measured performance claim.

**Historical next (implemented in UX-9C above): Web Models / Downloads / Devices.**
Lift existing cache/pull/device actions and observations out of native
presentation into shared host. Exact digest and operation identity,
explicit pinned acquisition consent, original pause/resume/cancel
available even while pull owns slot, no hidden file/credential grant.
Reuse current authenticated transport, bounded observation/worker,
native and Web cross-client event path. UX-9D settings/profile/logs then
UX-9E simultaneous mutation acceptance, UX-10 and public serving.

### Previous checkpoint — Phase 2.1 UX-9A (26 Nov; 4e472aed)

Embedded SolidJS/Vite Web shell over the original Runtime/ControlPlane.
UX-9A is intentionally read-only; real Web effect routes and the remaining
management views are UX-9B–E, not fake actions or another runtime.

- Product CLI optional web feature (default/all, CPU release/Nix) embeds
  committed Vite dist/index.html, app.js, style.css and bundled license.
  SolidJS 1.9.15, Vite 8.3.2, plugin-solid 2.11.14, esbuild 0.28.2;
  exact locally cached AW versions, pruned lock, npm ci --offline.
  Two consecutive production builds byte-identical; no Node runtime or
  network assets. Existing Rust getrandom 0.4.3 and httparse 1.10.1
  become optional direct frontend dependencies, no newly resolved Cargo
  packages. Minimal tui/web/CPU isolation retained; Nix source includes
  embedded dist and fixture, excludes node_modules. Cross builds not run.
- CLI web [--listen numeric-IP:PORT] default 127.0.0.1:0, --profile/
  --config and --offline as existing global selection. Validates selected
  document, starts original foreground runtime with same host limits,
  prints private URL. Native Home/Serve b starts/reuses that listener on
  the same Runtime and shows private URL to open manually; no subprocess,
  browser auto-launch, automatic load or inference listener. Closing a
  modal/tab never retires owners; confirmed process quit/Unix signals
  stop/join Web connections then original worker/listener/runtime.
- control_host.rs extracts original TUI Target/Document, runtime builder,
  Stop and retirement/combine. One Arc/RwLock process-local Selection
  shared by TUI and Web; cold worker snapshots settings at action start,
  selection updates invalidate browser observation (atomic revision);
  original runtime/load owners and ConfigFile stay authoritative.
- Separate bounded loopback HTTP control host (CLI frontend only),
  independent from inference server. OS-generated 32-byte random key,
  explicit private URL fragment, scrub on initial/same-document launch,
  tab sessionStorage (lock clears it); never query/header echo, raw
  diagnostics, provider token or inference bearer. Numeric bound Host
  including port, reject duplicate/invalid security headers, nonzero body,
  transfer encoding, non-GET and non-allowlisted paths; API requires bearer
  and same-origin/Fetch-Site policy, no CORS. CSP/no-store/no-referrer/
  frame denial. Static assets public but no owner data. No remote bind.
  8 bounded connections, 4 SSE slots, 8 KiB headers/32 fields, finite
  read/write timeout and 60s SSE connection lifetime; stop joins workers.
- GET /zg/control/snapshot: allowlisted projection of existing owner
  snapshot, original aliases/lifecycle/caps/counters, exact registration/
  generation/download IDs as opaque decimal strings, actual download
  bytes/phase, active profile label/offline, feature flags. No full
  runtime diagnostic graph, arbitrary paths or raw configuration.
  GET /zg/events authenticated SSE refresh/resync only, subscribe before
  initial invalidation, coalesced original ControlPlane events + shared
  selection revision; no invented durable event history. Browser one
  in-flight snapshot, coalesced refresh, reconnect, 2s visible/live
  counters (idle follows events), stale indication, abort on lock.
- Solid responsive rail/compact/mobile overlay, Home/Running/Serve/
  Downloads actual observations, Config/Profiles selection label and
  explicit unavailable/pending cold surfaces, no fake mutation buttons.
  One Ctrl-Shift-P/F1 palette, dark/light, keyboard/focus, exact selected
  registration, responsive detail, tab-local credential. MIT notice
  included/served at /ui/licenses.txt. web/README.md documents build
  and browser smoke; usual Rust gate needs no Node/browser.
- Focused strict terminal-only/web-only/combined lint pass; Web parser
  test 1/0 and JS 2/0. Real embedded HTTP fixture passes independent
  inference-key refusal, host/origin/CSRF checks, static/license routes,
  exact owner projection, SSE capacity/active stream/slow-header join.
  Actual installed Firefox BiDi smoke passes 1440/900/390 widths,
  authenticated Home, navigation/palette, reload/lock/reopen, independent
  browser lifetime; initially caught same-document fragment relaunch,
  fixed hashchange consumption and reran. Optimized combined run_unified
  6/0/0 (32.71s) incl actual cached Qwen 0.8B original 8 GiB envelope:
  TUI attaches Web to exact existing registration and inference generation,
  real profile/stop invalidations, no new runtime, final joins. Final
  packaging-only asset/license refinement requalified HTTP/Firefox/lint.
  Initial full gate passed all Rust tests but the duplicate-Host PTY/HTTP
  fixture used one TCP recv instead of the whole status line; corrected
  the fixture without changing auth. Complete isolated rerun passes
  2,485/0/109, release Web/TUI, installer and scripts. UX-9A accepted.
  No physical GPU/remote transfer, performance or public exposure claim.

Historical next UX-9B and UX-9C are implemented above.
UX-9D/E follow the live order.

### Previous checkpoint — Phase 2.1 UX-8D (26 Nov; 71e36933)

Native profile/configuration/log surfaces complete the bounded TUI design.
The existing ConfigFile/ConfigDocument and action/observation workers remain
the only owners; no new dependencies, settings store, runtime or policy.

- Profiles lists default/named documents without creating anything. Enter
  opens an editor target, not activation. Create/copy no overwrite; copy
  uses the shared relative-source anchoring. Activate explicitly selects
  future action settings within this process, never persists a selection
  pointer or changes already admitted model/worker/listener state.
  Default/portable files and the active named profile are protected from
  removal; other named removal captures exact observed bytes and confirms.
- Settings observer reads exact ConfigFile snapshots on the existing single
  coalesced read worker (now all cold inventory views). Shared explain/
  document_value supplies typed current values and provenance; optional
  unset schema fields are visible without invented defaults. Grouped
  contextual rows, / search, Enter value editor, z confirmed unset,
  t validate, v raw view, w raw edit, a activate, y copy, x remove.
  Captured snapshot is retained across form/confirmation, original
  ConfigFile.compare/replace/remove enforce stale refusal. Invalid save
  keeps original bytes and reopens the draft if still in its editor.
  Raw YAML uses the existing bounded UTF-8 input owner, arrows/Home/End,
  Ctrl-U clear, Enter newline, Tab two spaces, Ctrl-S save, Esc discard.
  Invalid other-profile documents support raw repair; malformed startup
  selection still refuses and direct config edit can repair it.
- SDK resolved-path read shares the original default/explicit selection
  policy, including missing-default-only fallback. Active config changes
  update future prompt default and TUI color using original CLI UI
  precedence; interactive/progress startup policy stays startup-scoped.
  Bare offline retained across activation. No model, device, network or
  graph permission derived from validation, editing or selection.
- Operational log ring (128 local entries): actual local action starts/
  terminal results/cancel requests and original transient control events,
  download phase/status transitions (not per-byte spam), missed-event
  notices. Search/level/component filtering and paused-view cutoff /
  follow; ring keeps collecting and bounds retention even when paused.
  Correlate original registration/generation/download ids and local
  action sequence; not a fabricated scheduler request id. No prompt/
  output/credential/raw config text, persistent trace or complete backend
  diagnostic capture. Future Web can consume same owner event/action
  model; this client ring is presentation only.
- Focused strict minimal and combined lint, TUI reducer/render/ring tests
  5/0, real settings PTY create/search/edit, invalid/stale refusal, raw
  repair, activation/protected removal, copy, declined/confirmed removal,
  log filter/pause/follow pass. Initial PTY assertions missed ANSI
  differential cells and adjacent Escape decoding; corrected fixture
  with actual resize repaint and standalone Escape boundary, no weakened
  write/refusal requirements. Initial combined release lifecycle exposed
  clipped absolute-path profile labels; relative labels with absolute detail
  fixed it. Serial release cached Qwen lifecycle/profile-switch passed
  (22.95s), preserving the live listener and scheduler; other five release
  tests passed. Full isolated gate 2,483/0/109 and release PTY/installer/scripts
  pass. UX-8D accepted. No new physical GPU/remote acquisition or measured
  performance claim.

Historical next UX-9A is implemented at the current checkpoint.
UX-9B–E follow the live order.

### Previous checkpoint — Phase 2.1 UX-8C (26 Nov; 9bf7829a)

Models / Downloads / Devices are active native clients of the same
Runtime, cache/artifact/device owners, and original DownloadOperation.
No separate acquisition runtime, scheduler, cache, source grant or
production test transport override.

- Cached, Catalog (static MODEL_TARGETS intent/metadata/holds), Partial
  shelves. Selection anchored to exact digest/slug/checkpoint key, not
  a mutable row position or provider alias. Responsive contextual
  list/detail split; move, inspect, verify, exact-digest load/run or
  load for Serve, alias, removal preview/confirmed apply. Alias is an
  additive non-replacing ControlAction/ControlPlane operation on original
  CacheStore owner; no hidden reference mutation. Removal preserves
  original Busy/Referenced/Durable/Incomplete outcomes. No inferred
  remote freshness or execution qualification. Partial records only
  observations: resume by re-entering original pinned pull intent.
- Models/device inventories use one coalesced read-only worker separate
  from existing single effect worker; no extra effect queue. Latest
  observation and exact selection survive navigation; terminal cache
  owner events/resync/explicit refresh invalidate, no scan on every
  download progress event. Reader joined before foreground retirement.
  Profiles/Config cold reads remain synchronous for UX-8D.
- Device rows show observed physical id, class/vendor, memory, backend
  availability, provenance and exclusions. Check selected source +
  exact device by same canonical resident planner (metadata/host policy,
  no execution grant), or explicit load on selected device using
  original admission and one cold route-set policy. No persisted
  preference change from selecting a row.
- Explicit native pull accepts approved Qwen recipe (original exact
  digest/member closure) or HF/GitHub full lowercase 40-character commit
  and explicit file paths or quant tag; no implicit source/tag/quant,
  arbitrary URL or payload on catalog hold. Scrollable consent shows
  exact intent. Re-enter same pinned intent for checkpoint resume;
  original resume validation remains authoritative. --offline on bare
  TTY is retained and refuses acquisition before transport/credentials.
  Existing cache quota or 16 GiB default for new root, no automatic
  quota increase; original transfer memory/admission bounds.
- Extract original hub client/resolver/publisher sequence shared with
  CLI, and original approved-recipe transfer helper (same expected
  manifest digest, no extra ref). TUI executes directly via its runtime
  ControlPlane on owned action worker; no nested CLI terminal monitor.
  Download view shows exact operation id, phase/status, current file,
  totals/maxima, cached/resumed/checkpoint bytes, logical accepted-byte
  rate and approximate remaining transfer ETA only when observed;
  publication/terminal durability distinct. p/s/c target observed exact
  id and run even with transfer worker busy. Cancel flag shared; quit
  restores terminal, joins worker and then reader/listener/runtime.
- Focused TUI selection/render/form reducers 4/0 and native pinned-intent
  test 1/0 pass. Real native PTY over existing controlled strong-validator
  transfer fixture: original safe mid-file pause (acknowledged), exact
  range reopen, durable original publication, cancel; no secret URL in
  presentation. Strict minimal TUI and combined acquisition/serve lint
  pass. Actual cached Qwen 0.8B GGUF lifecycle passes: cold selected
  physical CPU policy, inspect/verify, additive alias, declined removal,
  exact-digest run, repeated invoke, original HTTP equality, listener
  stop/unload/reload/quit, reclaim preview and durable removal (21.80s).
  Other five release tests pass; corrected offline shell PTY passes.
  Final full isolated gate passes (2,481 passed, zero failed, 109
  ignored), release PTY, installer and script contracts pass. UX-8C
  accepted. Initial PTY assertions missed
  differential ANSI cells; corrected to stable new rows/tokens without
  relaxing owner outcomes.
- No actual remote model download, physical GPU execution or new model
  performance claim. Native hub scope explicit files/quant and approved
  recipe only; advanced provider/aggregate/filter/Xet flows remain direct
  CLI, no automatic partial eviction or alias/reference retargeting.

Historical next UX-8D is implemented at the current checkpoint.
Embedded Web follows the live order.

### Previous checkpoint — Phase 2.1 UX-8B (26 Nov; 0c7ad87a)

Home, Running and Serve are now active over the original shared runtime.
No second scheduler/registry/model loader or network grant.

- CLI canonical serve and TUI share resident.rs: original alias/route
  construction, typed serving request checks, cold source selection,
  cache pins, CPU GGUF geometry and host envelope, original CUDA/ROCm
  Qwen source routing/budget, and original ControlPlane load owner.
  Standalone serve retains exact per-plan outer scheduler limits and
  acquisition consent/cancel path; TUI never offers or performs a pull.
  Same loopback Listener/auth/ingress options used by both. Direct
  one-shot run and legacy explicit serving retain their original paths.
- TUI one Runtime with finite outer worker envelope (32*8) and original
  host-backed pending policy; per-load config remains authoritative.
  One cold route set at a time; original selective/last retirement
  precedes replacement. Home Enter opens source then prompt if empty,
  or prompts for selected ready owner. l loads MODEL / ID=MODEL or
  selected document's declared set (blank). Quick run requires a single
  route; explicit load can include configured multiple aliases. Shared
  serve/run capacity settings and CLI source policy, no sidecar override
  or graph/unsupported route grants. Chat/raw defaults from selected
  document at startup, Ctrl-R in prompt toggles per request; no persistent
  chat history. Invocation uses admitted scheduler output bound.
- Running selected row targets exact ready alias/registration, first alias
  of one shared owner. j/k, Enter invoke, u confirms retirement of all
  owner's aliases. Actual capability/provider/device/memory/request
  observations; not guessed eligibility or per-owner throughput.
  Serve s starts numeric loopback address (blank selected document/default)
  with loaded routes, t confirms exact generation stop; stop retains
  models. Actual bound endpoint/counters/aliases. Listener feature stays
  optional; no implicit listener start, no browser/clipboard action yet.
- One owned action worker, no second execution queue. Navigation/input
  remains responsive during effectful load/invoke/unload/listener actions.
  Original stream polled with cooperative cancellation, bounded 64 KiB
  UTF-8 display copy identified by alias/registration, terminal output
  count/reason from actual completion. c or Ctrl-C requests cancellation;
  synchronous load/retirement/start/stop cannot be interrupted inside
  original owner and are joined truthfully. Bracketed paste is data only;
  forms own typing, bounded 32 KiB, multiline prompt, explicit Enter.
- q confirms process retirement if live owners/work; declining/navigation
  leaves owners alone. Unix stop signal requests normal exit; restore
  terminal, request cancellation and join owned worker, stop exact owned
  listener, then runtime shutdown. Combine cleanup failures; no detached
  worker. Borrowed UI session does not own runtime retirement.
- Cold views still synchronous; logs remain transient control notifications.
  Read-only source/cache/device/config views and missing-source refusal,
  acquisition/model-management actions and rich config/log screens are
  next UX-8C/D. No remote download, new GPU execution/numerical/perf claim.
- First serial optimized combined run_unified 6/0/0 (24.59s), reuses actual
  approved cached Qwen 0.8B GGUF fixture and unchanged 8 GiB host envelope:
  real PTY quick run, HTTP equality, navigation/declined quit, exact stop,
  unload/reload, process listener/terminal retirement pass. Minimal TUI
  reducer/render and strict combined acquisition/serve lint pass;
  combined CUDA/HIP/acquisition/serve compile pass. Final reducer/render
  tests 3/0, terminal-only and combined strict lint pass. Final combined
  release 5/0 other tests; real cached PTY/HTTP initially timed out at the
  fixture's shorter 15s HTTP read under observed host contention. Align
  with existing 60s read bound, no changed admission or output expectation:
  serial focused real test passes (35.16s) with bracketed paste, repeated
  invoke, original HTTP equality, stop/unload/reload/quit. Plain shell PTY
  passes. Full isolated workspace gate passes (2,479 passed, zero failed,
  109 ignored; release PTY, offline installer and script contracts pass).
  UX-8B accepted.

Historical next UX-8C is implemented in the current checkpoint; UX-8D
profiles/config/logs and embedded Web follow the live order.

### Previous checkpoint — Phase 2.1 UX-8A (26 Nov; 08e7e9d3)

Native Rust TUI shell, read-only observations over the original Runtime,
ControlPlane and ConfigFile. No frontend-specific scheduler, registry, cache,
settings store or execution authority.

- Bare command on capable foreground TTY opens the shell; non-TTY,
  TERM=dumb, quiet/plain/interactive-never retain concise help. Explicit
  commands remain direct. Existing --config/--profile selects one document
  and existing UI preference/CLI precedence applies; no implicit creation
  or acquisition. Optional tui feature in ordinary CLI default/all and Nix
  product releases; no terminal dependencies in explicitly minimal CPU
  feature set. Ratatui 0.30.0 + Crossterm 0.29.0 pinned, resolved offline;
  use one Crossterm version rather than AW's older 0.28 plus 0.29.
- Nine destinations Home/Running/Serve/Models/Downloads/Devices/Profiles/
  Config/Logs. Real-cell rail at >=70 cols, narrow nav overlay, persistent
  view scroll, focused navigation, 1–9, Tab, j/k/arrows, : destination
  palette, ? help, r refresh, Esc and q/Ctrl-C. Modal owns text entry;
  no duplicate global command/search system. No empty permanent detail pane.
- Borrowed session sees original runtime/control snapshots and bounded
  transient events, resync after gaps; active owners sampled at 1s, idle
  no snapshot polling, 200ms input/event multiplex. Cold cache/device/
  profile/config observation only on entry/refresh/invalidation, no model
  payload verification or device execution. Runtime/serve/download facts
  are this process only. Logs are bounded client-local notifications, not
  operational log history. All cold reads currently synchronous.
- Host owns runtime and joins it on explicit foreground exit; borrowed
  session does not retire owners when changing views. Scoped Crossterm
  raw/alternate screen and cursor restoration on exit/error/unwind;
  Unix INT/TERM/HUP request normal exit, Ctrl-C handled as input.
  Linux PTY and reducer/real-cell rendering at 60/90/150 cols pass;
  no new GPU or inference qualification. Focused terminal-only and
  combined acquisition/serve strict lint pass. Full isolated workspace
  gate 2477 Rust executions pass/0 fail/109 skips; release PTY including
  selected profile interactive preference/CLI override, installer and
  scripts pass. Nix feature wiring/format checked, cross-target release
  builds not performed. Initial PTY assertion expected contiguous
  whitespace in differential ANSI output; corrected to stable tokens.

Historical next UX-8B is implemented in the current checkpoint.
UX-8C/D and Web follow the live order.

### Previous checkpoint — Phase 2.1 UX-7 (25–26 Nov; 7f5fc4eb / abe98b83)

One typed ConfigDocument (the existing pipeline v2 Document, compatibility
alias) for model/graph/run/serve declarations and actual UI/download
preferences. No settings database, profile inheritance or second schema.

- Shared selected-document read: explicit --config or --profile replaces
  implicit XDG config.yaml. Missing implicit default => built-ins without
  creation; malformed/dangling/explicit missing documents refuse. Named
  "default" maps only to config.yaml. Canonical run/serve use the same
  selected document even with a positional model. Direct positional run
  replaces document model/graph entrypoint (does not inherit its source
  pin); explicit targeted --model retains old override semantics. Explicit
  serve model list replaces the model table. CLI settings override selected
  document. Canonical model/device commands can select a document for UI
  policy; opaque legacy commands retain their original grammar.
- Typed ui.color/progress/interactive and downloads.ask_before_pull fields
  control real behavior. Explicit CLI auto/always/never overrides stored
  preferences; stored interactive always is best effort, never forces
  nonterminal input. False ask_before_pull suppresses offer; it does not
  authorize a transfer. No unimplemented concurrency/symbol knobs or secret
  fields. Persisted HTTP sources refuse query/fragment/userinfo; transient
  CLI locators retain original contracts.
- SDK configuration::ConfigFile and cold validate/get/explain/change/copy
  use original bounded YAML parser, core model-table/graph and run-policy
  validators. Settings-only documents are allowed without a fake execution
  route. Existing serve limits/loopback validation retained; cold validation
  is not backend/source admission. Provenance uses existing run resolver
  plus actual settings fields, conditional worker defaults. Actual selected
  device/CLI provenance remains run --dry-run/--explain.
- config show/path/get/set/unset/validate/edit/explain/init; profile
  list/show/create/copy/edit/remove. Read-only commands do not initialize
  files; create/copy never overwrite, removal explicit --yes or affirmative
  TTY and default entry protected. All editors share ConfigFile. Dotted
  mapping paths and whole YAML typed values; typed edits normalize YAML
  (comments/anchors not retained). Raw edit uses VISUAL/EDITOR executable,
  no shell, temporary draft, validate before publish. Same-base copy
  preserves bytes; cross-base copy anchors relative source/sidecar/audit
  paths to original base.
- Atomic file replacement: bounded regular file, symlink mutation refusal,
  private temporary file synced, short cooperative parent-directory lock,
  exact original-byte compare for stale clients, no-clobber hard-link create,
  same-directory rename replace, Unix parent sync (post-publication sync
  failure reports unknown durability). No lock held over editor. Unix
  directory lock does not add persistent files; non-Unix uses lock sidecar,
  no parent-sync claim. Noncooperating external writers are not serialized.
- Basic inline terminal config/profile managers reuse the same actions;
  richer searchable/native TUI screens are UX-8D, not a second implementation.
  SDK cold edit/copy/stale/invalid/symlink and feature-minimal CLI lifecycle
  pass; actual terminal PTY set/get/invalid refusal/create/copy/decline/
  confirm remove/exit pass with no cache effects. Combined strict
  acquisition/serve Clippy and CUDA/HIP/acquisition/serve compile pass.
  Earlier full gate exposed an invalid-source error-category regression;
  preserved the existing parser contract at 7f5fc4eb. Real cached serving
  initially refused its original 8 GiB envelope during competing builds.
  Serial corrected release run_unified with acquisition/serve, including
  the real Qwen 0.8B GGUF test, passes all 6 tests (14.43s); no lowered
  budget, remote transfer or physical GPU run. Corrected combined strict
  Clippy and full mandatory-private-network workspace gate pass: 2473
  Rust test executions, 0 failures, 109 explicit skips. Architecture,
  GPU feature boundaries, release, installer and script checks pass.
  UX-7 accepted at this scope.

The historical next unit UX-8A is implemented in the current checkpoint.
Reuse original ControlPlane and ConfigFile; Web follows native TUI.

### Previous checkpoint — Phase 2.1 UX-6 (25 Nov; 84e0681c)

Canonical serve uses the original runtime, source preparation, model registry
and HTTP owner; run and serve share cold source selection, configuration and
device policy. No duplicate transport/scheduler, dependency or model grant.

- `serve MODEL...` or `--model ID=MODEL`; exact cached names/locators/digests,
  local GGUF, selected v2 document or named profile. Explicit routes replace
  the document model table. Shared run settings and optional typed `serve:`
  listen/worker/context/output/host/pending/metrics settings, CLI overrides
  first; no run prompt/teacher IDs carried into HTTP. Old explicit local
  flags and standalone legacy YAML retain their existing profile loader.
- Cold selection completes across the route set before acquisition. CPU
  GGUF direct models/aliases use the original role-bound preparation and
  host envelope; original HF uses one approved snapshot shared by aliases
  with the original CUDA/ROCm source loader. Mixed GGUF/HF and distinct
  original HF sources, graph serving and canonical GPU GGUF refuse; legacy
  qualified GPU GGUF/diagnostic serving remains explicit. Common CUDA route
  preflight requires actual ordinal 0, observed SM86 and available VRAM
  after headroom. No new physical qualification or silent fallback.
- Same explicit eligible 2B/27B acquisition recipe/consent/digest guard
  as run; offline never transfers. Shared shutdown flag cancels active
  transfer; reobserve device before load, require same selected backend.
  No transfer or GPU execution performed in this checkpoint. Foreground
  consent still uses existing blocking line input before any acquisition.
- Original ControlPlane owns exact listener generation. Startup reports
  admitted aliases, actual backend/device, bound endpoint and effective
  worker bounds; rich live counter samples at two-second intervals only
  when changed, without model/resource enumeration in the poll loop.
  Stop listener and join HTTP handlers before retiring inference workers;
  preserve original startup and retirement errors. Loopback and original
  optional inference bearer policy retained. No protocol rewrite.
- Focused release integration (8.39s, two passed): existing approved local
  Qwen 0.8B GGUF/cache import, named profile and two aliases share one owner
  and identical actual HTTP output, SIGTERM joins and closes socket.
  Cold incompatible routes refuse without cache creation. Synthetic
  tokenizer correctly refused by original inventory guard; real fixture
  retains that guard. Direct model document validation uses original
  direct-entrypoint check. Strict combined acquisition/serve lint and
  CUDA/HIP/acquisition/serve compile pass. Full isolated workspace gate
  2466 Rust executions pass/0 fail/109 skips, release/installer/scripts
  pass. Two outdated CLI expectations (forced serve interaction and
  missing default document) corrected; complete rerun passes.

Historical next unit UX-7 is implemented in the current checkpoint above.
Native TUI and Web follow in order.

### Previous checkpoint — Phase 2.1 UX-5 (25 Nov; 8e382947)

One-line run joins existing cold cache identity, selected config, source
admission and original text execution owners. No parallel loader, cache,
runtime, device authority or new model/backend/numerical qualification.

- Bare lowercase names select a durable canonical ref or clean durable
  human claim. Different digests refuse ambiguity; explicit catalog: and
  by-name: remain independent. A missing name can bind an exact approved
  Qwen catalog snapshot, never an arbitrary repository/quantization.
  Distinguish absent directory/claim from malformed, dirty or dangling state.
  Provider selectors retain exact pin and ambiguity checks; HTTP locators
  retain the compiled cache contract and explicit snapshot requirement.
  Model inspect/verify, cache selection and run use the same SDK selector;
  run binds the chosen digest before original preparation and derives native
  Qwen routing from the approved digest, not the requested alias.
- One quoted positional prompt, --prompt, --prompt-file, configured prompt
  or bounded piped stdin. Conflicting explicit inputs refuse; -- preserves
  literal flag-shaped prompt values. Config/profile and device/headroom
  policies remain original; dry-run does not read stdin. Normal output is
  stdout, actual model/backend/device/context status is stderr, quiet output
  unchanged. Explicit backend/source mismatches refuse.
- Capable foreground terminal with no prompt enters a repeated independent
  request session (/help, /model, /new, /exit, EOF). CPU retains one admitted
  PreparedCpu and borrows original generator; native Qwen retains one runtime
  registration, invokes original request_text. No accumulated conversation
  history, persisted session or automatic repeated teacher/audit diagnostics.
  Finite CUDA diagnostics remain explicit one-shot.
- --pull is explicit consent; eligible interactive missing-model offer
  shows exact repository/revision/bytes/backend and asks y/N. Offline and
  dry-run never acquire, existing local/cached source does not transfer.
  Only original immutable 2B/27B Qwen inventories provide automatic recipes;
  0.8B legacy digest-provenance manifest recreation, floating/unregistered
  locators, unsupported backend and insufficient cache policy refuse.
  The original resolver checks exact member set, lengths and supplied hashes;
  original executor checks required final snapshot digest before commit.
  ControlPlane download operation, original cache publisher and progress/
  cancellation shared; no reference retargeting, implicit quota increase,
  or fallback. Explicit acquisition remains available separately for other
  source selections. Recipe tests and mismatched-publication test pass;
  real remote acquisition and physical GPU execution not performed.
- Full mandatory-private-network workspace gate passes: 2465 Rust test
  executions, zero failures, 109 explicit skips. Strict lint, architecture,
  feature boundaries, release/installer/scripts pass. Final combined
  acquisition/serve Clippy and CUDA/HIP/acquisition compile pass; exact
  inventory recipe and required-snapshot publication guard fixtures pass.
  Initial full gate stopped on test-only unnecessary clone lint; corrected,
  complete rerun passes. Real existing Qwen 0.8B GGUF CPU release PTY:
  two independent prompts match their quiet one-shot stdout, one startup,
  /model and /exit, no cache creation. No new package/lockfile or performance
  evidence; no actual remote transfer or physical GPU execution.

Historical next unit UX-6 is implemented in the current checkpoint above.
UX-7 ConfigDocument/profile management, then native TUI and Web remain ordered.

### Previous checkpoint — Phase 2.1 UX-4 (25 Nov; da2385c7)

The canonical single-plan download lifecycle shares the existing native
ControlPlane, original resolver, acquisition executor, admission and cache
transaction. No parallel downloader, daemon, source grant or backend change.

- One control-owned operation view per runtime clone family, exact operation
  ID, typed phase/status/counters, cancel-request distinction and original
  durable/uncertain publication outcome. Caller owns the synchronous
  transfer; bounded events invalidate the latest snapshot. Rate is newly
  accepted logical bytes, not network throughput. Direct SDK/external cache
  activity is not a complete operation history.
- Ordinary HTTP with exact length, strong validator and a proper prefix can
  pause within a file. Original checkpoint, drop response, park the same
  admitted transaction, rehash held descriptor, reopen the exact validated
  range. No retry-budget reset or new cache owner. Native Xet, capped,
  weak-validator and unsuitable prefixes park at a safe completed-file or
  pre-publication boundary. Pause is a request until acknowledged; cancellation
  remains cooperative, never implies rollback of a durable/uncertain commit.
- Cold `model list --partial` (text/JSON) reports bounded checkpoint records
  by key and lengths, not authenticated or guaranteed resumable payloads;
  records may include active attempts. Existing recovery still validates
  identity, prefix, extent and validator before reuse. No implicit deletion.
- Canonical explicit HF/GH and native Xet online single-plan pull supports
  plain/rich progress, quiet, JSONL observations and final typed receipt;
  foreground Linux TTY Space toggles pause/resume, C/Ctrl-C cancels.
  RAII terminal restoration on normal return/unwind; one scoped CLI worker
  only when presentation/input needs it; quiet path synchronous.
  Confirmed cancellation exits 130, other failures preserve original errors.
  JSONL is latest observations, not durable event replay. Dry-run/offline and
  unsupported forced interaction/JSONL refuse before credentials/cache IO.
- Direct object/cloud/provider plans use the same operation/progress view per
  independently published plan; advanced forced interaction/JSONL and
  multi-component aggregate control remain explicit follow-up, components
  retain their original sequential executor/progress. No inferred catalog
  acquisition or provider fallback. Existing locked rustix termios/process
  features enabled; no added package or lockfile change.
- Original exact checkpoint/range continuation and cold partial inventory
  pass; two SDK clients pause/resume/cancel the same operation, reject stale
  and duplicate controls and preserve previous ref/resource accounting.
  Real isolated PTY fixture passes resume, C cancel and Ctrl-C cancel with
  restored modes. Normal JSONL and quiet JSONL pass (terminal durable state
  and one receipt, no URL/validator leakage). Full mandatory-private-network
  workspace gate exits 0: 2459 Rust test executions pass, 0 fail, 109 explicit
  skips; separate SDK acquisition test and five explicitly run monitor-child
  scenarios pass. Architecture/feature boundaries, strict artifact and
  combined acquisition/serve lint, release/installer/scripts pass.
  Development compile/lint issues fixed (closure annotation, CacheLimits
  scope, unused imports and boolean predicates). No new physical, provider,
  model or numerical qualification.

**Historical next unit (implemented above): UX-5 — one-line run (§46A.28).**
Complete canonical model/config/prompt resolution and direct execution through
the original owners, with explicit acquisition consent and factual backend
selection/refusal. UX-6 stable serve, UX-7 ConfigDocument/profile management,
native TUI then Web remain ordered. Keep advanced provider presentation
extension and aggregate component controls as bounded later lifecycle work.

### Previous checkpoint — Phase 2.1 UX-3 (25 Nov; 3cf1bb2d)

One shared native SDK ControlPlane over the existing Runtime and original
cache/device/config/transport owners. No frontend runtime, scheduler, catalog,
cache hierarchy, dependency or backend policy change.

- `Runtime::control()` borrows a lazily initialized control state shared by
  facade clones. `ControlSnapshot` reads the original runtime/catalog and
  optional control-owned serving observations; cold device, cache and
  document queries are separate. These are bounded observations, not a
  globally atomic transaction or an execution permit.
- `ControlAction`/`ControlReply` and typed methods delegate to original
  model load/load-declared, text invocation, unload, task inspection,
  devices, config checking, cache inspect/list/verify/reclaim and optional
  serving. Preserve core errors and original service status/code. Native
  inputs are trusted caller data; future remote transport must validate
  and authenticate them independently.
- Model invoke/unload binds exact observed registration under the existing
  engine lifecycle/admission lock. Alias replacement cannot retarget a
  stale action; no extra model handle remains through retirement. Inference
  output and cancellation stay with the original OutputStream.
- Independent subscriber cursors share a bounded 64-entry transient event
  ring, with explicit Resync on lag. Subscribe before snapshot; refresh
  owner state after invalidation. Notifications describe actual outcomes
  or accepted requests (not fabricated start/completion); no prompt,
  output, credential, source handle or durable event replay. Direct SDK,
  independent HTTP and external cache activity are visible via fresh
  snapshots, not a complete control event history.
- Optional serving slot owns one original ServerHandle, observed address,
  generation and live metrics. Exact-generation stop joins that handle,
  leaves runtime/models open, rejects replacement-targeted stale stop and
  duplicate start. Transition observations do not wait for joins. Direct
  SDK servers retain their independent lifecycle; explicitly stop the
  control-owned listener before process/runtime shutdown. Inference key
  is not control auth; no Web/control listener added.
- Canonical CLI cached model list/inspect/verify/remove and device
  list/inspect use the shared methods. Existing JSON schema, exact verified
  lease/reclaim outcomes and compatibility execution routes remain intact.
  Empty existing-cache text now matches absent-cache text.
- Approved local Qwen 0.8B GGUF release integration with two clients:
  two aliases/one model owner; direct and control output equal; actual
  loopback /v1/models, duplicate start refusal, stop/restart with distinct
  generations, stale stop refusal, model retained across listener stop,
  unload/reload and stale invoke/unload refusal. Passed in 8.80s with
  no download. Cold SDK shared refusal and bounded-cursor/resync tests
  pass. Full mandatory-private-network workspace gate exits 0: 2457
  passing Rust test executions, zero failures, 108 explicit skips.
  Including the separately executed real-model integration: 2458 pass,
  0 fail (one source-backed test executed explicitly). Architecture,
  feature boundaries, strict CPU and combined acquisition/serve lint,
  release/installer/scripts pass.
  Initial nonminimal-bool and test drop-non-drop lint refusals corrected;
  no physical GPU qualification claimed.

**Historical next unit (now implemented above): UX-4 — model lifecycle + downloads (§46A.28).**
Attach original acquisition/cache transactions to the shared control
state; typed progress, cooperative pause/resume/cancel, retained partial
reporting and quiet/plain/JSONL/interactive presentation. No second
download owner, implicit acquisition or expanded provider qualification.
UX-5 one-line run, UX-6 stable serve, UX-7 unified ConfigDocument/profile
management, native TUI then Web remain ordered. Preserve explicit source
and backend authority.

### Previous checkpoint — Phase 2.1 UX-2 (25 Nov; bcdb46a8)

Typed CLI command tree and unified cold model/config selection through the
existing SDK and artifact owners. No new dependency, runtime, cache, numerical
profile, backend admission, or implicit download. UX-1 presentation remains
shared; the Phase 1 minimum foundation exit remains accepted.

- `zerogravity-cli/src/args.rs` owns the typed command hierarchy, global options
  and compatibility dispatch. Concise help and static shell completion derive
  from the same command tree; legacy run aliases, devices, models, inspect,
  pull, run-cache and cache retain their original execution owners.
  Consume globals only at grammar/argument boundaries, preserving flag-like
  prompts and opaque legacy provider values. Global offline is inserted before
  owner options, never appended as a possibly missing value.
- The core `SourceSelector` and artifact `CacheMetadata::select` remain the
  common cold selector. Exact SHA-256 (bare or sha256:), explicit by-name:,
  canonical reference and acquired provider identities bind exact manifests.
  Explicit by-name reads clean durable claims, not disposable symlinks; dirty,
  ambiguous or missing names refuse. Existing canonical refs remain independent
  of names; catalog: escapes names that look like a digest. All actual loads
  still authenticate source through their original leases and model owners.
  No inferred GPU eligibility from an alias or digest.
- SDK `run::config::select_document` chooses one explicit file, named profile
  under XDG profiles/, default config.yaml, or no file; existing typed YAML
  resolution and CLI override provenance remain authoritative. No profile
  layering, editing, or new serving-document grammar.
- Canonical model list/inspect are bounded metadata-only (missing cache list
  empty without creation); verify uses original full verified lease. Remove
  requires exactly one of --dry-run/--yes and original exact reclamation.
  Search exposes static family intent only. Model pull forwards explicit
  provider selections/locators; advanced alias/import use original cache
  owners. Device inspect uses the same original observed-device resolution
  as run. JSON supported for model list/inspect/verify, device list/inspect,
  run dry-run; unsupported structured/interactive requests refuse before
  effects. General JSONL events, rich transfers, dynamic completions,
  bare human-name lookup, one-line run/serve and profile management stay open.
- Full mandatory-private-network workspace gate: **2451 passing Rust test
  executions, zero failures, 106 explicit skips**; feature isolation,
  architecture, strict Clippy, release/installer/scripts pass. Real synthetic
  GGUF CLI equality covers profile/override, digest and explicit alias with
  metadata/verify/list and unchanged cold cache. Existing cache transaction
  tests exercise durable name selection despite a corrupt projection.
  Release Bash/Fish syntax and actual completion pass, absent-cache JSON
  list creates nothing, observed CPU JSON passes. All four shell outputs
  generated/tested; Zsh/PowerShell not locally installed. No learned-model
  or physical GPU requalification; prior evidence stands.

**Historical next unit (completed above): UX-3 — shared ControlPlane snapshots, actions and
events** (CLI UX §46A.28), over the original Runtime/model catalog,
acquisition/cache, serving and config/device owners. No duplicated frontend
authority; lifecycle/download work (UX-4) follows. Keep TUI before Web and
the subsequent performance campaign in canonical order.

### Previous checkpoint — Phase 2.1 UX-1 accepted (25 Nov; e6211608)

The shared CLI presentation foundation is implemented and accepted in this
signed checkpoint on L5. The Phase 1 exit remains `d5030ad4`, with last
accepted foundation implementation `5a7535e7`; no execution or numerical
qualification changed.

- `zerogravity-cli/src/output/presentation.rs` owns the dependency-free
  terminal capability, options, borrowed typed events, reporter and failure
  renderer. Detect stdin/stderr TTYs, TERM=dumb and nonempty NO_COLOR once;
  explicit color always/never overrides auto. Default pipes remain plain.
  Typed core error categories and original serving status/code survive;
  hints follow the category, causes are diagnostic-only and existing
  process failure status 2 is retained.
- The existing command tree accepts a leading presentation prefix:
  `-q/--quiet`, `-v/--verbose`, `-vv`, `--format auto|text`,
  `--color auto|always|never`, `--progress auto|always|never`.
  Subcommand argv is left intact; unknown global values are not echoed.
  Rich symbols/colors are only human stderr presentation; quiet suppresses
  ordinary status/progress, not errors, actionable partial/durability
  warnings or explicitly requested machine diagnostics. The progress
  policy uses existing line-oriented updates/throttling for auto/always,
  and never disables them. No interactive prompts or cursor rewriting yet.
- Run/pull/cache/serve status and failures now share the reporter. Primary
  stdout, existing JSON/JSONL/explain/audit/logit outputs, source selection,
  acquisition semantics, existing cancellation, HTTP contracts and feature
  dependencies remain unchanged. No frontend runtime, new dependency or
  second domain event owner. Reporter events are presentation-level, not
  the future ControlPlane event model.
- Assembled mandatory-private-network workspace gate exited 0:
  **2431 passing Rust test executions, zero failures, 106 explicit skips**;
  format/architecture, backend feature isolation/qualification compilation,
  strict Clippy, release, installer and script checks passed. Existing
  synthetic GGUF CPU inference/local/alias/config/cache cases prove equal
  normal/quiet/diagnostic stdout. Release binary PTY checks passed for
  capable terminal, NO_COLOR, TERM=dumb, text mode and explicit color
  override; forced-color quiet device JSON remained parseable and
  unstyled with empty stderr. No new learned-model or physical GPU
  qualification required or claimed for this CLI-only change.

**Historical next unit (completed above): UX-2 (typed command tree + unified model/config
resolver)** under CLI UX §46A.28. Preserve the compatibility grammar and
machine output contracts while expanding global options/command grammar,
concise help/completion, and resolution through the original authorities.
Full global JSON/JSONL selection, richer interactive transfer lifecycle,
ConfigDocument management, shared ControlPlane snapshots/actions/events
(UX-3), TUI then Web and public serving remain ordered work. Do not begin
backend performance tuning.

### Previous foundation exit checkpoint

**Signed exact coherent mapped resident producer: 14bbbd489bae8cfa79beae0ceffc07d7a0f8817c.**
Qualification boundary correction: 5d952364.
Original logical upload accounting: 5a7535e7.

- Reuse the existing WgpuResource coherent device-local mapping, allocated
  only by the original Vulkan allocator for the qualified integrated
  memory type. WgpuResource::host_visible checks the actual mapping and
  byte range; the existing nonblocking mapped-write path uses the same
  checked copy primitive. No new allocator, foreign pointer import or
  staging fallback in the explicit producer.
- ResidentService::initialize_host_visible borrows exact caller bytes for
  a fresh private scratch/activation lease. Cold source/service/extent/
  layout/initialization checks; original fair queue selection; actual
  ManagedResource pins the exact backing against relocation. Mapping
  absence refuses before a submitted Write; use original
  DeviceBufferIdentity/BackingRange and existing exclusive Write event.
  Actual sync_buffers precedes the synchronous host copy; original
  finish_ready retires the event and marks the lease initialized.
  No mapped pointer or host slice escapes, no device-write CPU read,
  no alternate event, source/root publisher or host allocation charge.
  Existing logical upload accounting counts the payload, not DMA bytes.
  Cancellation/unknown prior barrier uses the original armed flight
  quarantine; no new asynchronous host work remains after the copy.
- Actual X2 RADV integrated Vulkan: one charged 4096-byte backing and
  1024-byte view at offset512; initialize full and subrange through
  actual coherent mapping, same original physical allocation identity.
  Original GPU copy operator sees every changed and untouched byte;
  checked status zero. Foreign-source, wrong-length, duplicate-write
  and HIP unsupported mapping refusals occur without fence or event.
  Host metadata/source, output/status and shared backing retire to
  original baseline. Existing shared Vulkan healthy/unknown and HIP
  healthy physical fixtures pass; final accounting correction rebuild
  and mapped healthy test pass. X2 idle 259768320 bytes.
  This is a real physical backing consumer, not learned-model parity,
  new Vulkan performance selection or device-write host-read proof.
- Initial combined compile found the mapped test helper excluded from
  the non-test qualifier; 5d952364 gates its test-only invocation.
  Final Vulkan-only and combined HIP/Vulkan qualification compile pass.
  Full isolated gate: 2396 Rust test executions, zero failures, 106 explicit skips; release/installer/scripts pass.
  The gate began before the accounting-only correction; final Vulkan-only
  and combined HIP/Vulkan compile and actual mapped healthy rerun pass.
  Prior learned CUDA/HIP source evidence applies unchanged (no math or
  serving selector changes).
- Scope: exact original coherent device-local host mapping, synchronous
  CPU producer for a fresh private scratch/activation view under original
  queue/event/lifetime; ordinary uploads keep their nonblocking policy.
  No arbitrary host imports, foreign backend/context/stream or
  untracked aliases, coherent CPU reads of GPU writes, general mutable
  state/weights mapping, noncoherent flush/invalidate, aggregate UMA
  heap/RAM alias capacity or OS-hard memory guarantee. Such capabilities
  require explicit new physical consumers; placement equality grants no
  storage identity.

**Phase 1 minimum exit audit:** the existing qualified local text/native
and portable consumers satisfy the ordered plan's common foundation
contracts, including the completed isolated gate above.
**Minimum Phase 1 exit accepted; Phase 2 is next.** This bounded exit
is not completion of every broader checkbox in owning plans.

| Foundation | Bounded accepted implementation |
| --- | --- |
| 1.1 Semantic composition | Original source-bound ModelProgram/ForwardPlan, ModelRegion, typed state/invocation ports, exact RegionSource and prepared executable/selection. Actual native source text and closed projection/region consumers; SDK-free direct compiled-provider scaffold. |
| 1.2 Provider/operator | Shared typed math and physical layout/profile, actual CUDA managed-private copy and secondary-stream completion, graph/native observation, native HIP/CUDA and bounded portable Vulkan. Explicit provider/shape/interop refusal; no graph SDK dependency or implied compiled execution. |
| 1.3 Source/lifecycle | Authenticated original HF, local/cache GGUF held and exact snapshots, verified tensor windows, common declared loader and eligibility, transactional acquisition/cache and original last-reader/alias/catalog lifecycle; explicit backend and actual pre-Client physical fit. |
| 1.4 Memory/state/control | Original range backing/leases, ResourceArbiter/queue, actual coherent mapped producer and completion/quarantine; complete-domain StateRoot, resident/paged/streamed and scoped prefix, bounded CPU/HIP persistence, original native cohorts/chunked calls and serial language pipeline. Typed media clock/port/placement contracts remain descriptive; unsupported work refuses. |
| 1.5 SDK/build/observability | Actual public SDK original HF native HIP/CUDA and CPU/CUDA GGUF, original CLI unified local/cache real runs, standalone router, task/poll/cancel and selected owner/effective configuration/plan/region/capture observations. Existing independent feature/architecture gates; no frontend-owned runtime. |

Remaining general foreign/UMA backing and multi-domain transfers,
non-Linux/OS-hard fit, generalized checkpoint/async SSD, family/codec/
media/DAG, wider learned Vulkan and CUDA quality/parity, production
capture selection and concrete external graph SDKs remain explicit in
their owning plans. They do not block the minimum shared foundation
exit for already-qualified local paths. No new model/performance claim.

**Historical entry instruction (Phase 2.1, UX-1; completed above):** inspect actual
zerogravity-cli and existing facade command/output/control owners, read
the canonical cli-ux/NEXT_ITERATION_CLI_UX_AND_MODEL_LIFECYCLE.md and
the owning cli-device-routing and serving TODOs. Implement the shared
presentation foundation (terminal capability, rich/plain/color/NO_COLOR,
quiet/verbose, typed reporter/events, unified error/hint and stdout/stderr
contract) through existing commands, without duplicating the runtime,
catalog, resolver or scheduling. Retain direct APIs and structured
diagnostics. Then follow the superseding §46A.28 ordering: UX-2 typed command tree and
unified model/config resolver, UX-3 shared ControlPlane and subsequent UX sequence. Native TUI and embedded Web
follow the same presentation/control contracts; no ROCm/CUDA performance
campaign before Phase 2 closes.

### Previous accepted captured execution observation checkpoint

**Signed actual captured-owner and prepared cohort observations: ce83cc18574ab3998d7a1dec84aba6cb300b5fac.**

- Recovered interrupted previous full gate (exit 0, 2396 Rust executions,
  0 failures, 106 skips) and already-written Vulkan handoff. Signed
  9017af17 and synchronized; original hardware results preserved.
- Existing CapturedResidentGraph::observation copies actual source,
  service generation, logical stream, BatchGraphProfile, original capture
  receipt and accepted replay enqueues. Uses core SelectedMode::Graph
  (CUDA), not Compiled. Original profile is now public descriptive
  metadata; no duplicate plan/shape identity, owner or graph pin. Replay
  counts are enqueue facts, not completion or root publication.
- ResidentModel::describe_graph_cache returns None for no batch, otherwise
  actual rows, workspace pin, original service admission/fence snapshot,
  and typed Cold/Capturing/Live/Spent projection. Live contains the real
  cached capture observation; an externally owned graph may pin a Cold
  cache. Spent includes failed/removed attempts, not proof of retirement;
  original terminal barriers, pin and opaque charge remain authoritative.
  The legacy qualification state/count methods use the same projection.
  No new registry, telemetry state, device work or capture authority.
- PreparedResidentCohort::selected_mode reports original provider and
  actual successful body dispatch mode. The existing launch branch
  records Native or Graph; all native/serial/source cohorts stay Native.
  One small mode value, no graph handle or observation copy in the hot
  path, no new allocation, state publication or selection policy.
  Public capture observation types compile under the original native
  graph substrate; actual capture/replay remains qualification-only.
- Existing CUDA graph owner qualification uses direct and cache
  observations, actual selected Graph vs Native fallback, exact source/
  shape/charge, external pin and explicit retirement/replacement; no new
  fixture or acceptance machinery. Native CUDA and HIP-only compile,
  qualifier compile, two graph cache/resource tests pass. Initial release
  build succeeded but executable was named zerogravity-gpu-probe rather
  than zerogravity-gpu-qualify; corrected invocation, no source failure.
- Actual L5 original Qwen 0.8B GGUF B2: independent frontiers, swapped
  rows, full 248320 vocabulary and all42 state domains exact against
  original CPU/B1/native B2, four committed steps and six real replays.
  Direct/cached observation, graph-compatible selection, native mismatch
  fallback without replay, explicit retirement and B2->B4 replacement
  pass. Opaque conservative policy charge 15651152 bytes retires to zero;
  no driver internal memory/graph-node-count claim. Existing timing probe
  remains diagnostic, no new performance selection. Actual graph-abort
  case: one accepted enqueue, unknown completion, original roots
  unchanged, graph/handles/envelope retained, fence visible and pin
  preserved; old native Ready still cannot publish. GPU idle 37MiB.
  Actual X2 original HF Qwen 0.8B HIP V2 C1/C8 all six waves
  [248068,198,90700], lane invariant, width8, zero failed GPU quanta,
  selected-owner chain and native/tokenizer retirement unfenced; idle
  259768320 bytes. No new learned model/profile or public graph claim.
- Full assembled isolated gate: 2398 Rust test executions (2396 workspace gate + 2 focused graph tests), zero failures, 106 explicit skips; release/installer/scripts pass. Combined CUDA/HIP SDK all-targets compile passed.

The bounded exact coherent host producer and foundation exit audit
are implemented at 14bbbd48/5d952364/5a7535e7 above. General foreign/
UMA backing and interop remain explicit.

### Previous accepted named Vulkan registry checkpoint

**Signed named Vulkan physical admission and original service reuse: 3b5a520d748bd5e353baea00dcaacd30e83d6793.**
Explicit backend guard: fc38db0371804a46d189a23fb83f531f3f240651.
Vulkan-only qualification accessor: e1decff57e98b55ee0cb09697015c8e8d998cba4.

- Resume from live 1462687e; previous semantic/region observations and
  CUDA typed bridge already accepted. No duplicate implementation.
- CubeCL's original DeviceHandle registry now supports non-creating
  get_existing and atomic fallible get_or_try_insert_with, under the same
  service initialization lock as ordinary lazy creation/explicit insert.
  Client returns the actual registered handle and utilities (no subsequent
  selector lookup). Channel, mutex and reentrant transports retain their
  original registries. Existing lazy and explicit registration semantics
  remain; factory must not reenter the registry. No new cache/registry.
  The two newly vendored 0.11.0 common/runtime crates are the exact local
  locked archive sources plus these narrow patches; license files retained,
  Cargo.lock changes only their source paths, no dependency version changes
  or external downloads. The original WgpuServer creation remains the one
  factory. Explicit named backend required for preflight reuse; preserve
  cold default override and ordinal validation. Existing explicit
  setup-registration APIs keep their original strict insert semantics.
- Ordinary and guarded Linux ResidentPhysical Vulkan share one cold
  selection/admission/adoption path: hold the original registered Client
  if present, otherwise exact uninitialized adapter. Require driver PCI
  and actual local total/used VRAM, intersect capacity and current fit,
  reserve original ResourceArbiter provider envelope before device/queue.
  Atomic registration can reuse a racing initializer; actual key/capacity
  must agree before adopting. Only actual factory invocation arms the
  existing bootstrap uncertainty guard; known reuse failure releases its
  unused charge; uncertain new bootstrap retains/fences as before.
  No separate service generation allocator, physical quota, or new
  provider/model eligibility. Different physical wrappers may share the
  actual service but still each admit their own envelope; model readers
  attached to one wrapper share its original quota/queue.
- Grouped original registry regression (workspace GPU test): absent lookup,
  fallible factory refusal without registration, four competing callers
  invoke factory once and reach same service state, ordinary lazy lookup
  reuses it and strict insert refuses. Pass. Vendor packages excluded from
  workspace tests; use this existing workspace owner rather than separate
  vendor test harness. Bare Vulkan and combined HIP/Vulkan qualification
  compile passed. Initial full gate found a client accessor still gated
  on HIP; e1decff5 widens it for Vulkan-only qualification.
- Actual X2 RADV named integrated Vulkan on attested PCI: cold peer-only
  refusal no Client, retry, earlier selected adapter whose registration
  is won by another initializer, warm ordinary load. Original client
  properties Arc/service identity shared across three admitted envelopes;
  dropping extras leaves exactly original 64MiB unfenced. Existing
  portable Q8/conv two-source healthy and unknown cases pass; original
  independent close/reload and last-reader retirement, unknown retained
  handles/both source owners and sibling fencing with unchanged roots.
  Final explicit-backend guard rebuilt and actual Vulkan cases pass.
  Actual portable HIP two-source healthy/refusal/retry and retirement
  pass too. This fixture is actual hardware, not learned-model Vulkan
  numerical/performance qualification. Final learned HF Qwen 0.8B HIP V2
  C1/C8 all six waves pass [248068,198,90700], replay/lane invariant,
  width8 and zero failed GPU quanta, native/tokenizer retirement unfenced.
  X2 returned to idle 259768320 bytes. Prior real CUDA evidence retained;
  no fresh physical CUDA execution claim in this slice.
- Full isolated gate: 2396 Rust test executions, zero failures, 106 explicit skips; release/installer/scripts pass.
- Scope: exact same named explicit Vulkan selector on Linux with attested
  local counters. No merging of different selector aliases into one
  context, imported/Existing/foreign device, non-Linux physical observation,
  Nvidia Vulkan without actual local counters, generic cross-context/
  cross-backend import or total UMA backing/OS-hard reservation guarantee.
  Legacy external setup registration may construct its own device; no
  guarantee for callers bypassing original named service preparation.
  No new capture, compiled-provider, kernel or learned Vulkan claim.

The bounded live capture observation continuation is implemented at
ce83cc18 above. Actual graph selection remains qualification-only;
general host-visible backing/interop remains open.

### Previous accepted prepared-plan/provider checkpoint

**Signed original prepared-program/region observations and CUDA provider compatibility: 8e3b8993b189beb80216e8b5a10fd8a5e484f7db.**
Layout tightening: 33118f162eca2618d97ff2b7fe6885ecae6da69f.

- Existing ExecutionDescription now contains optional ProgramDescription
  (original live source issuer, semantic node/value/state counts, context,
  source-template geometry, logits and frontier delta) and RegionDescription
  (original node range, prepared row band, actual logical/physical ports,
  state versions/storage, invocation/observation support and incremental
  provider requirements). Serialize validated original descriptor types,
  not a second IR or plan ID. Structural equality is not ownership.
  PreparedExecutable::description and ExecutableBundle::description use
  the original retained contract; whole text and ordinary tensor-source
  adapters report actual plan facts in existing SDK observations. Direct
  CPU without a recorded plan stays absent. No additional per-token
  graph/registry/reader or execution permission.
- ResidentModel/ResidentTextModel::describe_region(node) reads the original
  retained CUDA projection binding; None for an unretained independent
  region, invalid node refuses. No reconstructed region or generic
  partial-execution permit. Whole native selection stays Native; do not
  infer live graph replay or external compiler support from this metadata.
- CUDA same-context private/secondary-stream bridge: cold borrowed binding
  validates exact original service/queue and required stream relationship,
  scratch identity/extent, distinct opaque charges and capture refusal
  before submission. Original armed flight still validates exact accesses
  and resources, retains provider/queue/charge owners, orders before/after
  events and uses original completion/quarantine. Explicit borrowed views
  validate declared format/geometry, alignment, bounds, same service and
  nonoverlap. Copy requires identical dense layout/encoding; no padding
  overwrite, conversion, general zero-copy or foreign allocation import.
  Existing native/streamed GEMM share actual row-major/inner-major layout;
  streamed output uses original typed strided column view, not a copy.
  Other cohort byte views and alternate native primitives remain intact.
- Existing nine engine preparation tests with original selected region/
  source/state observation; strict core/engine/facade Clippy, combined
  CUDA/HIP compile and CUDA-specific feature compile passed. Actual X2
  HF HIP V2 C1/C8 all six waves [248068,198,90700], lane invariance,
  width8, zero failed GPU quanta, selected program observation and original
  alias chain/retirement passed. X2 idle 259768320 bytes.
- L5 CUDA original HF 0.8B resident vs actual streamed B1/C4 full logits
  and all state domains exact, selected retained projection facts,
  known cancellation/retry and shutdown passed. Actual ordinary CUDA GGUF
  three-stage two-owner chain selected program observation and selective
  unload passed. Physical existing certified BF16 GEMM private/secondary
  stream bridge passed preflight refusal while unfenced, exact transfer/
  original event return and no early Ready, interruption retention/fence.
  First physical attempt used obsolete Nix CUDA13.4 paths; use current
  installed CUDA13.5 runtime/CCCL/NVRTC and cuBLAS13.10, no downloads.
  First cuda-native-only binary matched zero bridge tests; actual
  cuda-native,cuda-cublas exact test passed. 33118f16 tightens dense-copy
  and shared GEMM orientation. Actual final dense/padded/layout/stream
  preflight, certified GEMM/copy and interruption retention/fence pass.
  Final learned resident/streamed B1/C4 full logits/all domains and clean
  shutdown also pass; both GPUs idle, L5 37MiB.
- Full isolated gate: 2396 Rust test executions, zero failures, 106 explicit skips; release/installer/scripts pass.
  Gate started at 8e3b8993; the later CUDA-only layout tightening was
  separately rebuilt and physically revalidated as above.
- Scope: observations of retained native/compiled-contract metadata, not
  a new qualified compiled executor. No live captured graph observation
  projection beyond existing capture-owner receipts yet; capture remains
  distinct from compiled/native. HIP cold projection validation does not
  retain an independent region object, so None is truthful. Private
  bridge remains a reusable bounded CUDA primitive, not automatic
  selection for learned models. Foreign allocations, cross-device/
  context/other-backend stream import and arbitrary partial state
  conversion remain unqualified.

The bounded named Linux Vulkan registry/admission continuation is
implemented at 3b5a520d/fc38db03/e1decff5 above. Actual live capture
observation and general shared-backing/interop boundaries remain open.

### Previous accepted shared GGUF source checkpoint

**Signed shared selected GGUF source preparation: 7987f70a566bfb7a5519b8f1a1013f138881de07.**
Refusal diagnostic compatibility correction: 2ec06026853b019876b753f50e38134ea202bee7.

- Shared prepare/source.rs resolves explicit existing GGUF locations:
  original local paths/sidecars, held regular weight descriptor and extent;
  cached provider/catalog selection resolves once to the exact immutable
  manifest, unique weights/config/tokenizer roles and bounded sidecars.
  Direct CPU Selection uses the same location and actual family/target/
  context validation. Read-only cache metadata grants no source authority;
  actual preparation opens the exact selected digest, never its mutable ref.
  Retain original source verification and actual model/codec validation.
- Original local CPU/GPU loader keeps its tokenizer bootstrap, retained
  inventory, approved tokenizer bytes/constructor and original weight
  admission order. Local materialization uses the same held descriptor
  that was sized; source extent changes refuse. Cached serving uses the
  original verified SnapshotLease for actual sidecars and weights, same
  approved tokenizer and family checks, original SourceWeights verified
  owned copy. Its raw byte lease remains with the actual immutable source;
  the serving wrapper reserves only the remaining conservative envelope.
  No duplicate raw-weight charge or new allocator. Direct CPU planning
  and serving retain their separate existing memory policies.
- TextLoadPolicy in models.rs is the existing pipeline policy generalized
  to cold declarations. ModelCatalog::load_declared and
  Runtime::load_pipeline_with_cache use the same builtins registration
  and original router lifecycle. Existing LanguageLoadPolicy name is a
  reexport; existing load_pipeline and legacy path/approved HF requests
  remain. Canonical local paths or exact (cache root, snapshot digest)
  aliases share the same prepared owner; conflicting declared targets
  refuse. Exact verified cache snapshot is exposed on GGUF artifact
  observation, not claimed as an authenticated remote revision.
  Original HF policy uses its existing approved inventory and explicit
  CacheOptions, one distinct snapshot; no change to source binding/math.
- Explicit cache root optional; default follows original cache policy.
  No automatic acquisition or remote execution. Local SDK sidecar/native
  artifact overrides remain unqualified (direct CPU local config/tokenizer
  overrides remain explicit); cached roles must come from the selected
  snapshot. Unsupported formats/families/target/context/sidecar and mixed
  backend/HF source policies remain explicit.
- Initial compile found unused Read import and stale test expecting the
  SDK-specific error fields from the shared HeldWeights core error; fixed.
  Minimal, model-only, strict CPU facade Clippy, combined CUDA/HIP SDK
  all-targets and three held-source/grouping tests pass.
- Actual L5 imported approved local Qwen 0.8B GGUF: declared SDK load,
  two aliases/one exact snapshot owner and task equality, unload/reload
  as graph-declared chain, selected artifact digest, original sequential
  chain equality and final observations. Retarget the mutable ref to
  config-only manifest while loaded, existing owner unchanged; later
  source declaration/pin refuses, no partial catalog. Earlier direct
  CPU Selection materializes the original exact digest after ref
  retarget, original envelope retires. Separate real CPU and CUDA runs
  pass. Existing local CPU 0.8B -> 2B target mismatch/retry/chain and
  direct framed task; CUDA GGUF three-stage two-owner chain/selective
  unload also pass. L5 idle 37MiB. X2 actual original HF native HIP V2
  C1/C8 all six waves [248068,198,90700], lane invariance, width8, zero
  failed GPU quanta, declared alias chain/final observations and original
  retirement pass; idle 259768320 bytes. No new kernel/profile or
  performance claim. Cached physical qualification is CPU/CUDA; other
  compiled local GPU policies are not new physical cache qualification.
- First full isolated gate passed feature/Clippy and reached the CPU CLI
  unified real-source fixture; its missing-acquisition diagnostic assertion
  caught wording changed during extraction. 2ec06026 restores the original
  HTTP missing-feature and sidecar diagnostic wording, and original SDK
  model_source_changed status/code for held extent changes. Healthy
  source/materialization behavior unchanged. Both unified CLI source-selection
  variants pass on the full corrected rerun. Full isolated gate: 2396 Rust test executions, zero failures, 106 explicit skips; release/installer/scripts pass.

The bounded original plan/region observation and CUDA compatibility
continuation is implemented at 8e3b8993/33118f16 above.

### Previous accepted SDK text-task controls checkpoint

**Signed shared SDK text-task controls and selected-owner observations: b25b5c02347e15980c082719d18a2685e687af3c.**

- Borrowed TextTask/TaskTarget::Model or ::Pipeline, TextControls and
  Runtime::submit_task use the original loaded alias/exact prepared plan,
  scheduler, source admission and event transport. Existing run,
  run_pipeline and direct request APIs retained. Explicit bounded output,
  raw/chat for models, streaming and optional original token observations;
  serial pipelines remain raw committed-text handoff, per-stage token
  ceiling, presentation and observations only on the final stage. No
  extra text copy, worker, model reader, queue, sampler or root authority.
- OutputStream::poll_next uses the original bounded receiver: Event,
  Pending or Finished; zero timeout nonblocking with respect to receiver
  wait, capped individual wait. At an intermediate terminal, admit the
  moved committed String before retiring its original transport and return
  Pending. Cancellation between polls cancels only the accepted request,
  never submits a future stage. Terminal/disconnection/stale-owner
  behavior retained; no hard wall-clock admission guarantee or concurrent
  cross-thread cancellation handle. Original next/cancel APIs retained,
  external stream implementations may explicitly refuse polling.
- submit_task and poll_next preserve structured ServiceError codes; errors
  from actual execution remain original ResponseEvent::Error. Legacy
  wrappers preserve the applicable core error categories. Read-only
  Runtime::inspect_task checks the same selected registrations, bounds
  and framing policy; reports selected stages, requested/effective text
  controls, actual scheduler limits/config/counters, input limit, and
  one original ModelSnapshot per selected owner. Process-local registration
  ID is shared by aliases and renewed on re-registration; no second plan
  identity generator or registry. Do not infer unreported numerical profile,
  region/external-buffer eligibility or actual context/resource fit from
  inspection. No source acquisition, tokenization, reader or admission.
- Existing grouped engine chain tests cover read-only host/worker
  observations, exact registration/stale same-owner reload, missing model,
  output bound, unsupported chat pipeline, final-stage observed tokens,
  cancellation between polls and structured next-stage retirement refusal.
  All engine tests, minimal facade and strict core/engine/facade Clippy
  pass; final focused three pipeline tests pass.
- Actual L5 CPU GGUF 0.8B -> 2B source-declared chain and direct framed,
  nonstreaming model task: selected owner identity, exact sequential
  terminal output/usage, three final token observations; direct task uses
  actual framed prompt length, three token observations with no text
  before terminal. Actual CUDA GGUF three-stage/two-owner chain, exact
  sequential output/usage, selected owner observations and three final
  tokens, selective unload/stale plan and final reader retirement pass.
  X2 actual native HIP V2 original HF 0.8B graph-declared two-stage shared
  owner chain, selected owner and three final observations, C1/C8 all
  six replay waves [248068,198,90700], lane invariance, width8, zero failed
  GPU quanta, native/tokenizer retirement and unfenced. L5 idle 37MiB,
  X2 idle 259768320 bytes. No new math/profile/performance claim.
- Full isolated gate: 2396 Rust test executions, zero failures, 104 explicit skips; release/installer/scripts pass.

**Phase 1 exit review (bounded foundations, not all backend specializations):**
- 1.1–1.2: original ModelProgram/ForwardPlan, typed source binding and
  closed-region prepared executable, native/capture distinction and
  SDK-free compiled-region preparation are established on actual consumers.
  Generic family/format/task source selection and remaining original
  plan/region observation and external stream/buffer compatibility seams
  are not wholly closed. Concrete compiler SDKs remain Phase 6.
- 1.3: authenticated HF and GGUF source eligibility, held descriptor/cache
  admission, original lifecycle, pre-Client qualified CUDA/HIP and guarded
  Vulkan local fit/host-backing constraints are established. Direct CPU
  selection can resolve an existing cached GGUF source; the SDK graph
  local route currently takes local paths and original-HF policy has
  its own exact snapshot resolution. Shared cold selector/adapter
  resolution still needs convergence without weakening source provenance.
  General named Vulkan registry reuse and exact shared GPU/UMA backing
  remain explicit unqualified gaps.
- 1.4: original independent state authority, paged/contiguous roots,
  completion/quarantine, admitted matrix/parameter streaming and bounded
  committed CPU/HIP persistence, same-domain placement, shared worker
  and bounded serial language composition/control have actual consumers.
  Other persistence/layout/provider combinations, heterogeneous transport,
  async SSD IO and media/realtime/general DAG are not qualified; keep
  the common contracts and defer unneeded specialization to native phases.
- 1.5: public SDK/standalone serving/catalog observations and common text
  task submission/poll/cancel/selected facts now share the original router.
  The historical CPU cache-admission failure was corrected by 00f57e7a
  and 9374e0ca (typed resource refusal and correct host test key);
  four actual cache comparison tests and subsequent complete gates passed.
  Continue remaining applicable architecture/feature and selected
  semantic-region observation debt; no new frontend registry or scheduler.

The bounded shared GGUF local/cache source resolution and SDK/pipeline
declaration seam is implemented at 7987f70a above. Broader family/codec/
task eligibility remains explicit.

### Previous accepted source-declared language preparation checkpoint

**Signed source-declared language preparation: ebda04c43d6b3f619cad01500edde1869eae8673.**

- Runtime::load_pipeline(Config, name, LanguageLoadPolicy) resolves only
  the selected existing serial language graph's unique model declarations.
  Original model IDs are the exact registered aliases. Other declarations
  are not opened. Structural validation still grants no execution/source
  permission; existing manually loaded binding API still refuses unresolved
  pins. The core chain structure and engine exact-registration plan binder
  are shared, not a second graph/runtime.
- The existing lifecycle load slot owns both private router and admitted
  immutable plan. Finalize outside the lifecycle mutex; no catalog or task
  admission while Loading. Exact alias coverage and plan host charge must
  succeed before the single Ready publication. Failed preparation/binding,
  pressure or racing shutdown retire original workers/source owners before
  releasing waiters. Preserve original sticky retirement failure, physical
  unknown-completion retention and process-lifetime context rules. No
  independent pipeline catalog, worker, model reader or state publisher.
- Explicit CPU/GPU local policy uses the original GGUF loader, original
  tokenizer/config siblings and canonical path alias groups. Actual family
  geometry must match declared Qwen 0.8B/2B target before weight
  materialization/device preparation; all aliases of one canonical path
  agree. Source eligibility still comes from the held artifact and model
  preparation. Explicit native HIP / CUDA original-HF policy requires one
  exact approved snapshot across selected declarations, target matching
  original inventory and optional exact hf:repository selector. Original
  HF authentication, admission and smart/backend policy unchanged.
- Unsupported explicit sidecar/native artifact overrides, local snapshot
  pins, remote acquisition/catalog refs for local GGUF, multiple distinct
  HF snapshots and conflicting source/target constraints refuse. No implicit
  download, remote code, backend selection/fallback or graph-as-permit.
  Local paths are caller-resolved or relative to process working directory.
  Mixed backend profiles, generalized DAG/media/context, HTTP/CLI pipeline
  activation and incremental load into an already configured runtime remain
  open. Original preloaded binding/run/retirement behavior preserved.
- Existing engine fixture actual Router/host ledger: mismatched private
  aliases and plan admission under full host pressure retire the router;
  no source exposure while Loading, empty/retry succeeds, unused model
  unregistered; existing stage order and terminal stream. Initial fixture
  default max_active was too small for its intentional wrong-alias router;
  fixed test limit, production unchanged. Focused regression and strict
  core/engine/facade Clippy pass; minimal and combined CUDA/HIP SDK compile.
- L5 actual local GGUF 0.8B -> 2B CPU chain, actual wrong second target
  refuses and retires first prepared owner before corrected retry on same
  runtime, unused path not opened, distinct owners, exact sequential
  outputs/final usage and selective retirement/stale-plan refusal pass.
  Actual CUDA local GGUF three stages, two independent source allocations
  with canonical alias reuse, sequential chain equality, selective unload
  and retained tokenizer/owner lifetime pass. CUDA idle baseline 37MiB.
- X2 actual native HIP V2 original HF 0.8B same-snapshot aliases loaded from
  graph declarations with persistence enabled: C1/C8 all six replay waves
  [248068,198,90700], width8, zero failed GPU quanta, lane invariant.
  Existing two-stage functional chain matches separate sequential calls,
  original native/tokenizer retirement, unfenced; idle 259768320 bytes.
  CUDA original-HF declared policy compile-checked; actual new CUDA graph
  qualification is local GGUF, not HF. No new numerical/performance claim.
- Full isolated gate: 2396 Rust test executions, zero failures, 104 explicit skips; release/installer/scripts pass.

The bounded shared SDK text-task controls and selected observations
continuation is implemented at b25b5c02 above. Other task controls and
unqualified transport remain explicit.

### Previous accepted native HIP resident persistence checkpoint

**Signed committed native HIP resident persistence: 804d488200ea12b82e487f9e590f90f0d3055672.**
Actual upload unknown-completion qualification: d9b252057ef76cc56bb0ecb37f9fc653a24061f4.

- Existing optional state-checkpoint feature now weakly enables resident
  persistence and core source-semantic revision metadata. Source factory
  declares its actual immutable config/source-bundle and explicit semantic
  revision on the original static ForwardPlan before sharing; no new plan
  graph, source issuer or runtime registry. No additional plan field in
  feature-disabled builds. This is trusted semantic compatibility metadata,
  not authentication or new model/provider eligibility.
- ResidentTextSnapshot/ResidentTextSession save_committed_checkpoint and
  ResidentTextSession restore_committed_checkpoint use the original
  completion-backed snapshot read and artifact bounded SHA256/atomic file IO.
  Separate resident format (ZGRESTEXT_STATE1) binds the original verified
  bundle, source-owned semantic revision, exact native HIP V2 profile,
  actual compiler/device fingerprint and ordered F32 domain/plane geometry
  including prepared context capacity. No local root/session/service
  identities, unused KV tail, prompt history or sampler data on disk.
  File payload is bounded by the original physical shape (max 1GiB).
- Restore validates full extent, finite words, compatibility and integrity
  while private. Decode active words directly into native full planes with
  canonical zero tails, no intermediate flattened F32 payload or second
  device copy; move owned bytes into existing initialization. Cold HOST
  workspace protects packed/transfer bytes and IO scratch; retains source/
  model owner on uncertainty. Use the original admission lock, root pool
  checkout and queue; known pre-submission refusals leave current root and
  original backing intact. Actual Ready service/lease/generation coverage
  precedes existing SessionAuthority domain swap; invalidate derived state
  and stale drafts. Legacy diagnostic import reuses this final publication
  check; existing fresh upload path and zeroing protocol remain intact.
- Actual X2 original HF Qwen 0.8B HIP V2 context512: complete committed
  numerical state file 20815960 bytes. Restore into independent session
  with outstanding draft, exact all-domain bits, exact full 248320-vocab
  decode continuation; corrupt-trailer refusal leaves root/host reservation
  unchanged. Real fresh-process replay reopens the original source and
  file, exact domains and continuation again. Local replacement advances
  receiver root/epoch and invalidates the previous draft; temporary host
  charge retires; ordinary shutdown retains only original 4GiB native
  context until process retirement.
- d9b25205 reuses existing hardware-abort probe after real state H2D.
  Actual X2 fault: cancel pending restore at submitted/unobserved boundary,
  both committed roots unchanged, host/source/physical ownership retained
  and service fenced. Unknown backing released only by process retirement.
  No new fault harness/receipt machinery. GPU returned to idle baseline.
- Actual X2 native HIP V2 0.8B C1/C8 population with state-checkpoint
  enabled: all six waves [248068,198,90700], replay/lane invariant,
  width8, zero failed GPU quanta, native/tokenizer retirement, unfenced.
  Opt-in HIP and combined CUDA/HIP SDK compilation and strict core/model
  Clippy passed. Include state-checkpoint in the existing HIP compile gate
  (21ce3d7f); updated command passed. Combined legacy diagnostic build
  initially found a borrow-check error at the shared publication seam;
  ef7c3d6c retains the original model across the admission lock.
  Final combined diagnostic compilation and both focused import tests pass.
  Full isolated gate: 2395 Rust test executions, zero failures, 102 explicit skips; release/installer/scripts pass.
  The gate began at 804d4882; subsequent fault-only, diagnostic-borrow
  and compile-gate changes were separately qualified as described above.
- Scope: only qualified native HIP V2 contiguous F32 original-HF text
  state, exact prepared capacity, explicit caller-owned trusted local
  checkpoint. No implicit CPU/HIP/CUDA conversion, paged/BF16 resident
  restore, arbitrary graph/version migration, token/prompt prefix grant,
  scheduler store, tenant authentication, encryption, disk quota or
  asynchronous file IO. Integrity is not authenticity. The model/backend
  revisions must be advanced on incompatible math/layout changes.

Bounded source-declared language preparation and atomic pipeline load/bind
are implemented at ebda04c4 above. General task/pipeline controls and
unqualified transport remain explicit.

### Previous accepted CPU checkpoint

**Signed committed CPU checkpoint foundation: 1d5c4ada3e224e7278ddd487563073e2d8d7be94.**

- Optional production state-checkpoint feature (weakly forwarded by facade;
  included in production all) exposes QwenDenseCpu::save_committed_checkpoint
  and restore_committed_checkpoint on the original native GGUF CPU session.
  No CLI/router/prefix-cache activation; no diagnostic feature required.
  Default CPU builds retain the original state layout and no extra clone.
- The v2 format binds exact immutable owned GGUF content (lazy SHA256 on the
  held allocation), actual semantic geometry/state specs and the native CPU
  reference numerical/layout version. No caller-asserted digest, prompt
  history, sampler state, process-local session/root ID or device handle.
  Same content can rebind a new loaded allocation; another source/profile
  or external kernel explicitly refuses. The CPU reference version must
  change when its persisted numerical/state compatibility changes.
- Share the original v1 layer body encoder/decoder (bounded chunked f32
  bit transport); preserve original diagnostic v1 bytes and APIs.
  New model-neutral artifact checkpoint IO supplies a held regular file,
  exact bounded streaming payload and SHA256 trailer, private same-directory
  create_new staging, fsync/rename and Unix directory sync. Linux refuses
  symlink/FIFO opens. Trusted local directory/files only: SHA256 detects
  corruption, not malicious recomputed content, authorization or encryption.
  Pre-publication failure preserves the old file; a directory-sync failure
  after rename can report an uncertain durable publication.
- CPU import checks the complete file extent from actual geometry, exact
  layer/domain counts, all finite numerical words and integrity before
  publication. Caller borrows the existing host TransferAdmission; cold
  IO/one-row conversion scratch retires; new retained import envelope
  stays with the immutable arrays via the original QwenState readers,
  including diagnostic views when both features enabled. New snapshot/
  candidate clones retain the same charge; reset drops it only after
  last reader. This prices imported storage, not future session growth.
  No new allocator or second state authority. Reserve a local replacement
  through SessionAuthority, publish once, invalidate stale attempts; no
  import of a local RootBinding from disk.
- Actual L5 public SDK feature-only Qwen GGUF 0.8B and 2B: save committed
  root despite outstanding draft, reload same source into a different
  allocation, full 248320-vocabulary continuation bits and state
  fingerprint exact; stale attempt, wrong actual model, typed budget
  and corrupt-trailer refusal leave the original session untouched.
  Retained import charge survives advance/reset and final immutable reader,
  then retires. Original diagnostic golden v1 bytes, complete-domain
  mapping and real original 0.8B/2B v1 bridge tests all pass. Artifact
  atomic-write/corruption regression and strict production-only Clippy pass.
  No new kernel/profile, GPU execution or performance change.
- Test-only CPU-free diagnostic filesystem import corrected at f9dd5ed6.
  Explicit portable-checkpoint build exposed pre-existing stale GPU diagnostic
  import (35629e6c): use the existing ResidentBufferLayout and plane bytes,
  refuse non-F32/paged state before H2D, retain original completion/publication;
  correct the existing cublas diagnostic test feature guard. No persisted GPU
  format or production provider path added. CPU-free and combined feature
  builds, strict CPU/CPU-free Clippy and both focused diagnostic import tests
  pass. Full isolated gate: 2395 Rust test executions, zero failures, 102
  explicit skips; default CPU release/installer/scripts pass. The gate
  started on the unchanged production implementation; the later diagnostic
  fixes were separately compile/test checked.
- Scope: explicit native GGUF CPU numerical checkpoint persistence only.
  No learned HF resident/CUDA/HIP persisted restore, cross-provider
  conversion, prompt-prefix authorization, general scheduler-managed
  persistent sessions, tenant/authenticated store, encryption, disk quota
  or untrusted directory sandbox. Existing caller-managed host scopes and
  live state allocation contracts remain authoritative. Other OS/filesystem
  behavior is not newly qualified.

The bounded native HIP V2 resident continuation is implemented at
804d4882/d9b25205 above. Other resident formats and cross-provider
restoration remain explicitly unqualified.

### Previous accepted portable admission checkpoint

**Signed portable HIP / guarded Vulkan admission and same-domain placement: a654886341e6ae3831ab28675fbf8daddbbf605f.**

- Ordinary Linux HIP ResidentPhysical and guarded HIP native bootstrap
  (including callers without an already supplied reservation) adopt the
  original PreAdmittedDevice lease before Client creation. Use the previous
  selected PCI, driver total/current free and local VRAM intersection.
  Typed current-fit and peer-pressure refusals do not construct a Client;
  HIP driver metadata/memory queries themselves may initialize the runtime.
  Preserve original source/physical owner, shared quotas, completion/fences
  and native context lifetime. No new device/provider eligibility.
- Extract the existing exact guarded Vulkan adapter selection/registration
  to the physical owner. Require its reported PCI identity, actual local
  VRAM total/used and intersection with its driver-reported local heap
  before registration. Reuse the same original preadmitted lease and exact
  adapter (no re-selection or second charge). Cold-only explicit guarded
  path, not automatic general Vulkan initialization.
- PlacementRequest::colocated_device and the original SDK Devices::plan
  accept the four equal declared physical domains with an empty inter-domain
  transfer plan. Unknown/nonphysical compute and distinct role domains
  refuse; no bytes, allocation reuse, external buffer/stream compatibility
  or execution fit inferred. Read-only and requires open runtime.
- Actual X2 portable Q8/conv two-source shared-owner HIP and Vulkan healthy
  and unknown-completion cases pass with cold competing-envelope refusal/
  retry. Vulkan uses the exact preflighted RADV adapter and original shared
  physical owner. Both use one Client/envelope and independent close/reload;
  unknown retains actual handles/source owners, unchanged roots and sibling
  fence. No learned-model or performance qualification from this fixture.
  Public device/placement tests 6/6, strict core/engine Clippy, bare HIP,
  combined HIP/Vulkan qualification and CUDA/HIP SDK compilation pass.
  X2 initial offline Vulkan build lacked seven locked crate archives;
  transferred the existing L5 cached archives verified against Cargo.lock.
  No external downloads, SDK or privilege changes.
- X2 actual original HF Qwen 0.8B HIP V2 C1/C8 all six replay waves
  match [248068,198,90700], lane invariance, zero failed GPU quanta, clean
  native and final tokenizer reader retirement. X2 idle 259768320 bytes;
  no active compute owners. CUDA execution unchanged, accepted real-source
  evidence retained; no new CUDA performance or numerical claim.
- Full isolated gate: 2394 Rust test executions, zero failures, 101 explicit skips; release/installer/scripts pass.
- Scope: same declared physical key is not common backing or zero-copy.
  Generic repeated named Vulkan initialization still lacks a non-creating
  existing-service lookup / atomic selected-adapter initialization in the
  CubeCL registry. Do not call its cold-only try_init_preflight on every load
  or invent a second registry. Keep this gap for the qualified portable
  Vulkan consumer; no new runtime/context registry, generic Vulkan route,
  non-Linux fit, or complete host-visible GPU/UMA alias accounting.

The bounded CPU committed persistence continuation is implemented at
1d5c4ada above. General resident persisted restore and transfer remain open.

### Previous accepted host backing checkpoint

**Signed host backing / local VRAM observation: cbd2c1198a49ad800945b33ed01ba45ba15dafcc.**
Actual HIP cold-allocation correction: 197c2e26f58e27064d9b574ae8050ac41ab5919f.

- Original ResourceArbiter counts all host-scoped reservations against one
  shared backing constraint. Keep each configured operator scope and its
  exact leases; first attested host ceiling is immutable, newest known hard
  OS/cgroup ceiling may tighten current admission without resizing the
  original cap. Partition preserves aggregate credit; known retirement
  releases, unknown work retains. No new resource row/owner or allocator.
  RuntimeSnapshot reports coherent host_backing and scoped rows.
- Move bounded Linux host observation into the shared owner. MemTotal and
  MemAvailable intersect the visible cgroup-v2 ancestor hard limits/current
  use. Unknown/incomplete hierarchy does not become a capacity guess.
  Known over-limit use is zero available. No OS polling in hot admissions.
- Existing serving/raw-cache/direct CPU routes establish shared host backing
  before cold allocations; direct proposal and revalidation use the same
  conservative fit. Actual GGUF weight, HF source control and HIP context
  reservations use the same cold host check. Existing outstanding charges
  without exact backing mapping are conservatively protected as possible
  growth, not claimed physically unbacked. Explicit SDK routes retain their
  scoped logical bounds on unknown live observation; no automatic fit claim.
- LocalVramObservation is shared by DRM inventory and actual qualified HIP
  device preadmission. Intersect HIP driver total/free with local VRAM
  total/used; integrated devices require both local counters. Never add
  OS-visible host RAM, VRAM and GTT or treat GPU addressability as capacity.
  X2's 96GiB local VRAM aperture and OS-visible RAM remain separate constraints,
  not evidence of independently additive pools. Host-visible GPU allocations
  and arbitrary UMA alias backing still need real backing/ownership mapping.
- Initial real HIP run refused because its conservative lifetime HOST upper
  (including existing backing and possible scratch) was treated as wholly
  incremental. 197c2e26 keeps the original source-shaped logical bound and
  checks actual upcoming 2GiB context reservation instead. Real X2 native
  HF 0.8B V2 C1/C8 six waves pass with [248068,198,90700], replay, no failed
  quanta, original reader retirement and no fence. Existing 32k capacity
  refusal before native context creation passes. L5 HF CUDA peer refusal/retry,
  C1 replay and local GGUF two-owner pipeline/selective unload with coherent
  aggregate host snapshot pass. Both devices return to idle baseline.
  Initial core 190/190, strict core/engine Clippy and combined SDK checks pass.
- Full isolated gate: 2394 Rust test executions, zero failures, 101 explicit skips; release/installer/scripts pass.
  Gate began at cbd2c119; 197c2e26 landed during it before CPU checks.
  Final combined CUDA/HIP SDK and strict core/engine Clippy pass; actual
  HIP runs cover the correction and CUDA cold-check behavior is unchanged.
  Fresh release CLI real local Qwen3.5 0.8B GGUF direct CPU run (context32,
  eight prompt tokens, one generated token) passes under the same isolation.
- Scope: logical aggregate host admission and observed cold availability,
  not an OS-wide memory reservation or precise RSS/backing attribution.
  Visible cgroup-v2 only; unobserved ancestors, external allocation races and
  later changes require fresh observation. No generic portable HIP/Vulkan
  pre-Client adoption yet; HIP driver observation itself may initialize the
  runtime. No new source support, physical Vulkan or performance claim.

The bounded ordinary HIP/guarded Vulkan admission and same-domain placement
continuation is addressed by a6548863 above. The general named Vulkan,
non-Linux and shared-backing/transport gaps remain explicit.

### Previous accepted CUDA pre-Client checkpoint

**Signed CUDA pre-Client physical admission: 9ae6dd1eb3c7bdd8c8ffb30479bd7775686a9c2c.**
Public admitted physical identity accessor: e06213a517e1a88732fc5ceadac8fd1b9a97b354.

- Shared existing PreAdmittedDevice bootstrap/adoption owner now covers Linux
  CUDA as well as the original HIP/Vulkan guarded paths. Context-free driver
  metadata selects exact PCI identity and total capacity, rejects ambiguous
  PCI views and preserves source-specific SM86/compiler headroom. No new
  inventory, allocator or estimator.
- Ordinary HF and GGUF CUDA resident constructors reserve the configured
  provider envelope through the original ResourceArbiter before Client.
  Adopt exactly that charge/owner and verify actual Client identity/capacity.
  GGUF capability validation uses the admitted service Client.
- Pinned CUDA keeps existing actor cleanup and live free-memory clamp against
  original configured headroom; release only unused cold credit through
  partition_reserved before model admission. Known post-bootstrap capacity
  misses drop unused admission; unknown bootstrap retains charge and fences
  through the original guard. Source-shaped admission is recomputed only when
  the live cap narrows. Separate compiler-context lifetime unchanged.
- L5 actual Qwen 0.8B HF SDK peer pressure refuses before any new observed
  NVML compute owner, exact peer-only reservation, no fence, same-runtime
  retry/inference/unload succeeds. Compiler context remains independently
  charged until process retirement. Actual pinned live fit narrows 2147483648
  to 1987510272 bytes with one model envelope, zero after owner drop. Existing
  HF C1 three-wave replay, actual same-process contiguous/paged native C8
  exact token replay with retained compiler-context reuse, and GGUF two-stage
  distinct-owner pipeline/selective unload pass. X2 HIP V2 C1/C8 all six
  waves match [248068,198,90700], original
  retirement and no fence. Both devices return to original idle usage.
- No OS/driver-wide memory reservation guarantee: the pre-Client credit is the
  conservative configured logical provider envelope, not free VRAM inferred
  from total capacity. Logical conflicts can refuse before the later live
  clamp. SDK smart config retains context-free NVML planning; actual actor
  observation is post-bootstrap. Compiler-context charges still precede
  compiler installation, not primary-context creation. Generic non-Linux
  CUDA, portable HIP/Vulkan and finite diagnostic-only direct Client routes
  retain their documented gaps. No fresh Vulkan physical or performance claim.
- Full mandatory-private-network gate: 2396 Rust test executions, zero
  failures, 101 explicit skips; architecture, feature isolation, release,
  installer and scripts pass. Gate started on 9ae6dd1e; the visibility-only
  accessor e06213a5 landed during it. Final combined CUDA/HIP SDK all-targets
  compilation passes; actual native runs cover unchanged execution code.

The bounded host/VRAM observation continuation is closed by cbd2c119/197c2e26
above. Portable bootstrap and general shared-backing placement remain open.

### Previous accepted source/model eligibility checkpoint

**Signed source/model eligibility: 8eb17bc2125ed73260240db9cda47312c501ac39.**
Test-only corrections: 00f57e7a, 9374e0ca.

- Qwen ForwardSource authenticates the original HF source, roles and geometry
  independently of task/numerical lowering. ForwardSourceSeal binds an explicit
  supported language task and arithmetic contract over the same owner, without
  a new source, allocation or metadata reservation. Legacy binding wrappers
  and SDK load request variants remain available.
- The single approved HF inventory supports backend-free snapshot discovery.
  HF registration shares cold preflight, refuses unapproved digests before
  host/cache/device preparation and carries the selected spec into actual
  descriptor/config/member verification. Existing integrity/provenance pins and
  explicit backend policy are unchanged.
- GGUF routing, direct preparation and local CPU/GPU SDK loading share bounded
  format/family/embedding checks and consume the original held descriptor.
  Verified cached preparation retains its existing lease and layout checks.
  No second catalog or new codec, task, model-family or numerical support.
- L5 real CPU original/cache and private HF-imported GGUF comparisons for Qwen
  0.8B/2B: full vocabulary and all42 state domains bit-identical at two positions,
  isolated sessions/mutation/reclamation and final raw-lease retirement. Initial
  stale Limit assertion and synthetic host identity corrected in tests only;
  all four cache-preparation tests pass. Combined GPU SDK compilation, focused
  pre-cache source refusal, model-only inventory, original seal ownership and
  unsupported task/arithmetic tests, strict CPU/model Clippy pass.
- L5 actual HF CUDA C1 and local GGUF two-stage pipeline/selective unload;
  X2 actual HF HIP V2 C1/C8 all pass. Original HF expected IDs/replay and native
  retirement unchanged. Both devices return to prior idle usage. No new
  performance or cross-backend numerical claim. Capture code unchanged.
- Full mandatory-private-network gate: 2396 Rust test executions, zero
  failures, 101 explicit skips; architecture, feature isolation, release,
  installer and scripts pass. Gate started on unchanged production 8eb17bc2;
  the two test-only corrections landed during it, with all four corrected
  cache-preparation tests run separately against real sources.

The ordinary Linux CUDA bootstrap continuation is closed by 9ae6dd1e above.
Remaining physical placement, broader family/codec/task support and SDK/control
foundations remain open.

### Previous accepted compiled-region checkpoint

**Signed SDK-free compiled-region foundation: 3ca16abd538fe8a4ffa0fc07b472865756392bf2.**
Public SDK reexports: a53a3446dc271c1081ea0e850a851907435e7c74.

- One cold typed RegionSource binding seam over the original ProgramSource.
  Borrowed sources return validated encoded references; verified-window
  sources return authenticated matrix/parameter descriptors without loading
  payloads. ModelRegion::operators borrows original nodes, ordered operands,
  and source identity; no export, duplicate IR or second cache. CUDA packed
  projection and HIP text preparation share the exact matrix check.
- SelectedMode distinguishes native, captured Graph, and Compiled(provider)
  independently of the device. Engine preparation verifies the selection
  captured before await against the prepared executable and original
  source/plan/scope. Custom provider implementations supply selected();
  prepared executables report their actual selection. Same-backend wrong
  modes/compiler identities and missing compiled contracts refuse.
- PreparedRegion::bind_call couples the original call/positions and typed
  physical shape; existing state effects, admission, leases and completion
  stay authoritative. No hot-path allocation, new kernel selector, implicit
  fallback or independent publisher.
- Engine preparation 9/9, tensor windows 30/30, strict core/engine Clippy
  and combined CUDA/HIP SDK all-target checks pass. SDK-free graph-only
  preparation tests cover direct original operand binding, selected
  identity, shape/observation refusal and ownership. L5 actual Qwen 0.8B
  CUDA C1 and X2 native HIP V2 C1/C8 pass three waves each, exact token IDs
  [248068,198,90700], replay and retirement; devices return to 37 MiB and
  259768320 bytes respectively, no compute/KFD owners. Prior captured graph
  proof applies to unchanged capture code; no fresh captured or legacy
  GGUF physical qualification in this slice. No compiler SDK execution,
  general stateful dispatch, numerical parity or performance claim.
  Full mandatory-private-network gate: 2394 Rust test executions, zero
  failures, 101 explicit skips; architecture, feature isolation, release,
  installer and scripts pass. SDK reexports landed before the CPU/minimal
  SDK checks; the native runs cover unchanged 3ca16abd production code.

The bounded source-registration continuation is closed by 8eb17bc2 above.
Broader family/codec/task registration and remaining placement, state/task and
SDK/control foundations remain open before Phase 2.

### Previous accepted parameter-retirement checkpoint

**Signed parameter-retirement checkpoint: f4e09428cd8d1f77302a24e18050687fa8fba6e6.**
Production commits: 687d9a2467af9479108b6069c12e68545032630b and
06df027fd126a57368984c668dc7e277d3f442bd; 43928332 is the CUDA
qualification-only consuming-shutdown correction. f4e09428 corrects the
old portable diagnostic Limit assertion to the actual Resource refusal
introduced in e79814ce, with requested > available and original zero
allocation/event checks unchanged.

- Complete ParameterWindow validation shares the exact authenticated
  issuer/member/geometry/extent. ForwardParameter retains the sealed
  VerifiedParameter and descriptor, codec, storage policy and original
  device lease, not transient raw or conversion payloads. The typed-slice
  diagnostic similarly separates retained source bindings/control budget
  from its initialization inputs.
- One materializer preserves Original, BF16 RNE, F32Exact, signed zero,
  finite/overflow checks and portable 256-byte padded F32 layout. Charge
  conversion and possible alignment bounce before allocating. CUDA and HIP
  use the original logical Write with one shared payload; actual window,
  payload and charge are retained by the original initialization until
  known physical completion, or quarantine on uncertainty. Resident hot
  access is unchanged.
- The attempted native complete-allocation CUDA parameter upload refused
  a padded physical view in the real consumer. Remove that parameter
  selector, not the reusable strict native matrix/page upload primitive.
  No change to the native ABI or original allocator.
- Artifact tensor-window tests 30/30 and focused parameter conversion
  tests 3/3 pass. L5 actual Qwen 0.8B resident/streamed B1 full vocabulary/
  all42, native B4 token/score/all42, known-completed cancellation/discard/
  retry, cold host transfer-staging baseline and shutdown assertions pass.
  GPU returns to 37 MiB/no compute owners. X2 native HIP V2 C1/C8 expected
  tokens, replay and original retirement pass on 06df027f (final CUDA-only
  test correction does not alter HIP). X2 returns to 259768320 bytes/no KFD
  process. X2 actual portable typed-slice HIP healthy at widths 1024/2048
  matches CPU and the existing independently accepted PyTorch archives,
  with original host/device owners retired, no pending events or fencing.
  No new performance, parity or unknown cold-upload fault claim.

Full mandatory-private-network workspace gate: 2393 Rust test executions,
zero failures, 101 explicit skips; release, installer, architecture, feature
isolation and script checks pass. Assembled on unchanged production
06df027f; the two qualification-only corrections landed during the gate,
with affected physical checks executed separately. No new full rerun of
unaffected suites or exact-source evidence pinning.

The subsequent SDK-free compiled-region checkpoint above supersedes the
continuation. General asynchronous SSD IO, opaque cross-runtime/device
interop, unknown cold-upload injection, portable persistence and
cross-provider restore remain open.

### Previous accepted matrix transport checkpoint

**Signed source checkpoint `fdfb467f7ec86ddb891f78f562698675875a3565` —
shared authenticated resident/streamed matrix transport.**
`7b36a879` shares the original `TensorWindow::require_rows` exact
issuer/member/expert-plane/row-range/byte-extent check among resident matrices,
CUDA packing and source pages. The immutable CUDA pack authenticates its
members once during preparation; invocation still compares the exact original
descriptors. Resident calls borrow device leases directly, without a source
lookup or cache dispatch. Singleton, native decode and independent-prefill
cohort projection consumers share one resident/streamed transport; selection
of arithmetic/reduction and row layout stays with the original operation.
No new allocator, worker, numerical profile or publisher.

Cold CUDA packing uses the existing owned-window actor upload instead of a
host payload copy. The initialization's retained pending owner holds the
actual admitted window until the original physical barrier, or quarantine on
unknown completion. The streamed page uses the same window type and its
existing flight/reuse checkpoint; HIP event retirement remains unchanged.
Portable serial staging retains the charged copy and shared Bytes; native
CUDA is selected by the actual physical device (`0342e103` fixes the
CUDA-only feature boundary). Existing transfer rings and dormant alternatives
remain reusable. Packed row geometry stays within the existing bounded page.

Focused source binding and strict artifact Clippy, combined SDK compilation,
L5 actual Qwen 0.8B resident/streamed fit, singleton full vocabulary/all42
and native B4 prefill/decode token/score/all42 match; known-completed streamed
cancel discards and the next execution succeeds. `fdfb467f` isolates the
two physical preparations in fresh processes and compares actual bit-content
digests, not a pinned reference. The earlier same-process combined comparison
exhausted the existing cumulative PTX budget; no cap was raised. Ordinary
CUDA C1 three waves and X2 native HIP V2 C1/C8 six waves match expected
IDs/replay; X2 two-stage alias chain and original retirement pass. The HIP
run predates the CUDA-only correction and test-only isolation, which do not
change its build. Full isolated gate: 2393 Rust test executions, zero failures, 101 explicit skips; release/installer/scripts pass.
No performance claim, new async SSD IO, unknown cold-upload fault injection
or broader Vulkan physical/combined-context eligibility.

The matrix checkpoint's parameter follow-up is implemented in the newer
checkpoint above. Phase 1 remains open; prior deferred scopes remain deferred.

Prior accepted resident state identity checkpoint:

**Signed source checkpoint `2df8cbb3f11ba698278b3537fc4fd16ca52908b5` —
prepared resident state identity and scoped prefix compatibility.**
`PrefixCache<U,T,K>` compares the original owner's exact typed key with an
explicit sharing scope and exact semantic units; it never grants restore.
`ResidentTextStateIdentity` derives from the original immutable prepared
source/plan and service generation, selected provider/mode, numerical
profile, token/contiguous-position geometry/ports and resident state policy.
No additional identity issuer or portable fingerprint. Snapshot/session keys
come from their retained owner; even equal source IDs with independently
prepared services cannot share state. Qwen checks both payload identity and
frontier before the existing physical fork. Lookup remains bounded,
allocation-free and longest-exact-prefix/newest-tie; no second cache.
State-only lookup excludes the final prompt token, including an exactly
matching cached prefix, so original feedback must still execute.

Same-session restore and pristine-destination fork share source/service/
capacity/frontier, complete-domain/initialized-lease validation (including
actual paged logical token frontiers) and the original root/domain swap.
Neither invents Ready or transfers another service's state. The original
cache's host entry/token backing is admitted before resident preparation
with one incoming prefix included; its original immutable snapshot readers
retain their separate control and actual variable domain/lease/generation/
page-table metadata charge. `snapshot_backing_bytes` and cloned page-table
byte counts are reusable, without allocating another physical page.
Cache retirement drops both payloads and host backing before native teardown.
No per-token counters, new device allocation policy or duplicate publisher.

105 engine tests, strict core/engine Clippy and combined HIP/CUDA SDK compile
pass. L5 physical Qwen 0.8B complete-domain fork/replay and same-session
restore, B1 contiguous/paged exact at frontiers 3/4/5/9, scoped reuse
(65 warm / 1 hit / 65 foreign), exact-length prefix replay64, bounded
eviction/purge and scoped HTTP all pass. X2 native HIP V2 C1/C8 six replay
waves with `[248068,198,90700]`, lane equality, two-stage alias chain and
original retirement pass. Full isolated gate: 2393 Rust test executions, zero failures, 101 explicit skips; release/installer/scripts pass.
No new performance or broader numerical parity claim. Existing CUDA paged
prefix sharing only; no HIP/Vulkan prefix qualification, portable snapshot
format, arbitrary positions, cross-provider restore or context authorization
extension. Host backing charges cover the retained production cache/snapshot
owners, not all OS/driver RAM or all legacy transient clones.

The matrix window/transfer-retirement slice is implemented in
`7b36a879` and its follow-ups above. General media/DAG/context authority,
pipeline source loading and transport activation remain outside the serial
language subset.

Prior accepted source-bound language pipeline checkpoint:

**Signed source checkpoint `4477f20ce7de60d722eb6967e1f33b5e565cb154` —
source-bound serial language pipelines.**
The existing structural `Config::language_chain` selects a verified
non-realtime text-to-text llm chain in topological order. No second graph
representation or generic interpreter. The SDK's `prepare_pipeline` requires
one explicit `LanguageBinding` per referenced model declaration: config ID,
registered alias and exact loaded ModelInstance. The target registry remains
structural task intent, not proof of model identity. Source-loading pins on
selected declarations refuse rather than being silently ignored; resolve
source selection separately before supplying a loaded binding. No automatic
acquisition, model/device selection or transport startup.

A small issued registration ID on the original typed Route distinguishes
a new registration from the same source ID/Arc, shares across aliases and
never wraps. The immutable PreparedLanguagePipeline owns only ordered stage
metadata and its original host admission, not models or runtime handles.
`run_pipeline` checks every exact registration and output ceiling before
starting the first original text request. Each next stage checks its own
registration under the existing lifecycle mutex; future stages are not
pre-admitted and do not block unload. Earlier completed state is not rolled
back on later refusal. Intermediate stages are ordinary non-streaming
requests; their actual terminal text is moved into the next request under
the existing ingress charge while the predecessor's output transport is
still retained. Only final-stage deltas, terminal output and usage are
exposed; per-stage output limit is common, not an aggregate usage claim.
Cancellation/drop cancels only the current original request and never
submits another stage. The existing blocking OutputStream is pull-driven,
not a new asynchronous cancellation API. Direct SDK text shares the same
admitted-copy and event-receive helpers; no new hot-path work or device math.

Focused engine tests pass topology, exact registered ownership, final
usage/text, cancellation, selective retirement and same-Arc re-registration
refusal; source-free pipeline tests pass unsupported pins, media and cold
runtime refusal. Strict CPU Clippy and combined HIP/CUDA SDK compile pass.
Real CPU GGUF fixture SDK/HTTP executes serial aliases and distinct owners,
refuses stale plans after cold reload and unloads without retained models.
L5 actual two-owner CUDA GGUF chain matches ordinary sequential requests,
then selective unload preserves the surviving owner; native CUDA C1
passes all three replay waves. X2 native HIP V2 Qwen 0.8B C1/C8 passes all
six waves with `[248068,198,90700]`, replay/lane equality and zero failed
quanta; outside timing, two-stage alias chain matches ordinary sequential
requests and original retirement passes. Both hosts idle at their prior
memory levels; A1 untouched. Full isolated gate: 2393 Rust test executions, zero failures, 101 explicit skips; release/installer/scripts pass.
No new numerical or performance baseline claim.

The next snapshot/prefix slice is now implemented in `2df8cbb3` above.
General DAG/media execution, context authority, cross-backend placement,
automatic loading from pipeline declarations and CLI/API route activation
remain outside the serial language subset.

Prior accepted runtime-observation checkpoint:

**Signed source checkpoint `bffad06a9c11efd497b659cff7b735c00e08e718` —
typed loaded-owner and runtime observations.**
`Runtime::snapshot()` and `models().snapshot()` read the original catalog,
worker configuration/counters and resource ledger on demand. Runtime phase
and sticky failure are explicit; legacy loaded-model count means aliases,
while loaded-owner count groups exact prepared owners, not source IDs.
Selective drain/retirement/failure states and admitted readers use the
original route lifecycle. Retired alias groups remain bounded until the
cold route set closes; descriptions retain no execution or source reader.
Whole-route joining has no available router snapshot. The process ledger
remains observable after close, including retained or fenced resources.

CPU GGUF describes its actual local source and host admission; ordinary GPU
GGUF describes the actual prepared program, device and quota/lease bytes.
Native HF derives its execution description from PreparedTextProgram and
reuses ResidentTextAdmission and ResidentStatePolicy, including actual
row/observation and prefix-sharing policy. No duplicate physical accounting;
model-level limits, reserves, resident bytes and host charges are not
additive to the process ledger. Unknown adapter details remain absent.
Direct CPU has no separately recorded numerical profile. The existing
scheduler configuration is retained once; Router context/output are
read-only accessors. Existing HTTP diagnostics remain, with additive typed
catalog/scheduler fields; process-wide resource owner labels stay in the
local SDK. Snapshots are not a cross-owner atomic transaction or OS memory
telemetry. No new per-token locks, counters, allocation or device calls.

Focused 103 engine tests, core fenced-resource check, strict Clippy and
combined SDK compilation pass. Real CPU fixture SDK/HTTP lifecycle and
health snapshot pass; L5 two-owner CUDA GGUF selective retirement and
native contiguous/paged C8 exact tokens pass; X2 native HIP V2 C1/C8 exact
replay/lane consistency, zero failed quanta, typed/legacy admission
agreement and original retirement pass. Both GPUs idle at original levels.
Full isolated gate: 2389 Rust test executions, zero failures, 101 explicit skips; release/installer/scripts pass.
No new performance, numerical-parity or broader backend support claim.

Next: bind the existing typed task/pipeline structure to exact loaded model
owners and the real text request/stream path. Start with bounded language
stages; retain explicit unsupported media, context grants, heterogeneous
placement and compiled execution until real implementations qualify.
Do not introduce another catalog, worker, source authority or state publisher.

Prior accepted selective-retirement checkpoint:

**Signed source checkpoint `2b6656b3d427092dea5bcdbb801bc71e59ba000c` —
selective catalog retirement with the original workers.**
Recovered the interrupted 24 Nov source edits and completed the assembled
slice on 25 Nov. One immutable bounded alias table, with one typed Route
per exact prepared owner; equal source IDs do not merge distinct owners.
A ModelRead moves from request admission to queued job to active row
and drops after physical state, model/presentation and source-reader
owners. Queue sweep, refusal and last-consumer destruction release it.
The owner closes all aliases atomically under the existing lifecycle lock;
accepted readers drain and the selected adapter cleanup runs unlocked.
Retiring keeps the original Router available to unrelated owners. A second
unload/load refuses during that operation; racing shutdown waits for it,
then joins the remaining workers. Last owner still uses the original full
queue drain/join and cold reload boundary. Failure remains sticky for
shutdown/new-load policy. No per-token registry lock, new queue, or new
physical ledger. Stable retired alias slots remain bounded until the route
set closes; no live insertion or replacement.

Executor::selective_retirement is an optional independently safe cleanup
hook. HostLogits and ordinary GPU GGUF opt in. CPU actual Arc readers retain
their source charges; ordinary GPU reuses the prior resident shutdown and
takes its executable once despite metadata/tokenizer readers. Final actor
cleanup uses the original physical queue after graph retirement, so it
does not overlap another model's commands. Device fences still govern all
affected owners. Native HF and finite/captured adapters retain their
original whole-set retirement and do not acquire selective eligibility
from this contract. Adapter snapshots are bounded cold views, not another
catalog. No physical mixed-owner HIP/Vulkan or cross-backend residency claim.

Revalidation: all 102 engine unit tests, strict engine Clippy, combined
HIP/CUDA SDK compilation. Real CPU GGUF fixture SDK/HTTP unload/reload and
mixed-owner survivor inference pass. L5 real Qwen 0.8B GGUF two independent
owners (same bytes at distinct canonical paths, two aliases on target)
pass target/survivor accepted work, exact device reservation 4 GiB -> 2 GiB
-> 0, released target host weight credit, retained tokenizer and continued
survivor inference. Native CUDA C1 and X2 native HIP V2 C1/C8 pass
`[248068,198,90700]`, replay/lane consistency and original retirement
(HIP zero failed quanta). Both GPUs idle at original levels, no compute
owners. X2 clean at exact signed source via incremental bundle; A1 untouched.
Full isolated gate passes (2388 Rust test executions, zero failures, 101 explicit skips; release/installer/scripts pass). Combined CPU/CUDA/HIP all-target compile passes. No new performance or independent
numerical parity claim. Next: reconcile typed catalog/SDK effective
capabilities, lifecycle and resource reporting through the actual loaded
owners; continue remaining source/state/pipeline foundations in plan order.
Do not build a second registry or promote hot replacement/heterogeneous
residency or backend-context reinitialization without actual consumers.

Prior accepted sole-owner checkpoint:

**Source checkpoint `057683d1c315fa026e0d4787b073757ca8caf699` —
shared local source admission and bounded catalog retirement.**
L5 test-only diagnostic correction `139bdbdf`; X2 executed the original
implementation checkpoint. No source/kernel changes in the correction.
`LocalRoutes` validates the full alias declaration before path access, resolves
canonical local paths and shares the same prepared owner for aliases.
CPU and ordinary GPU GGUF use a single held-descriptor loader: preserve the
approved tokenizer's separate retained/constructor credits, admit the original
conservative host-weight envelope before materialization, bound config bytes,
and retain the weight lease on the actual `AdmittedGguf` TensorSource and
ModelDefinition owner. Native HF keeps its existing authenticated snapshot
and source plan; alias validation is shared. Canonical-path aliasing is not
content-addressed deduplication across distinct paths, nor a guarantee
against concurrent mutation of a local file.

`ModelCatalog::aliases/unload` checks exact registered prepared-owner identity
(not a name or equal source ID). Sole-owner route sets retire every alias
together: existing admission closes under the lifecycle mutex, existing
workers drain/join outside it, and the same Joining phase releases waiters
before returning to Empty (or Closed if shutdown raced). Failed retirement
stays sticky and prevents new loads. Mixed-owner selective retirement refuses
before changing routes. Ordinary GPU GGUF now takes its executable once on
worker retirement, uses the existing graph/lease/namespace cleanup, and
leaves retained model metadata/tokenizer readers valid without retaining
the executable. Cohort matching uses the original session model reference,
not a new lock at each step. CPU retained source readers keep their own
host credits until final drop. Native context lifetime remains backend-owned;
cold reload is not a promise of same-process HIP context reinitialization.

Focused catalog/loader tests pass (2 + 3). Genuine CPU GGUF fixture SDK/HTTP
lifecycle unloads and reloads successfully. X2 native V2 Qwen 0.8B C1/C8
passes expected tokens/replay/lane invariance and unload while retaining
original handles, leaves only the context reservation, and returns to idle.
L5 original native CUDA C1 and actual local Qwen 0.8B GGUF inference via two
aliases pass unload; the GGUF test verifies zero device reservation,
unfenced card, released host weight credit and valid final tokenizer.
Combined CPU/CUDA/HIP all-target compile passes after migrating stale
qualification-only invocation accessors. Full isolated gate: 2388 passed, 0 failed, 101 explicit skips; release/installer/scripts pass.
No new numerical parity/performance baseline or wider provider eligibility.
Next: selective retirement in mixed-owner route sets and remaining shared
catalog/source lifecycle where required, preserving one registry/queue and
actual in-flight ownership; advance other Phase 1 state, scheduling, task and
observability seams without a second runtime.

Prior accepted cold admission checkpoint:

**Accepted source checkpoint `f4746c2350153b518e4730c3448e8f4ede02eab8` —
cold capability, selected device and effective native admission.**
Shared `ProviderMemoryBudget` applies the same headroom-preserving clamp to
discovery and the original CUDA actor's live capacity observation. The
native source policy checks explicit provider/device agreement; CUDA checks
SM86, selected ordinal/PCI identity and the original 5 GiB physical /
512 MiB context-headroom contract before Client creation. The actor still
performs its live clamp and exact resource admission after Client creation;
there is no general transactional pre-Client CUDA allocation guarantee.
HIP single/shared native source loaders acquire the existing physical
envelope before one-shot runtime configuration or Client construction,
then attach that lease and partition the original context. Ordinary pressure
refusal drops only unused admission; uncertain bootstrap fences and retains
the original envelope. HIP driver observation itself is not a promise of
zero driver-side initialization or an OS VRAM reservation. The existing
Vulkan diagnostic preadmission uses the same guarded bootstrap, compile
checked only this slice. No new device support.
`ResidentTextAdmission` retains the actual native fit, not a smart-config
estimate: original PCI identity, configured/admitted model capacity,
required bytes, separate context budget, physical limit, source/resident
weights, logical root and actual pool reserve, session/cohort limits, and
HIP host upper bound (CUDA complete host upper remains unknown). The SDK
reports it alongside the original prepared mode, profile, rows and observation
capability. Aliases reference the same admission; these are bounds, not live
memory use. Known capacity misses stay typed; unsupported candidate policy
may be excluded; other failures do not trigger another selection.
Focused headroom test and combined HIP/CUDA SDK compilation pass. Physical
Qwen 0.8B HIP pre-Client pressure refusal leaves host/card reservations
unchanged and unfenced; same-process retry observes 653 producers and
matches final logits. X2 native V2 C1/C8 and exact-reference C1 pass expected
tokens, replay, lane invariance, zero failed GPU quanta, actual SDK report
and original last-reader retirement. L5 CUDA C1 and C8 contiguous/paged
pass expected singleton tokens / matching cohort outputs, report bounds and
shutdown. Both hosts clean at exact signed source; idle L5 37 MiB, X2
259768320 B, no compute owners. A1 untouched. No new numerical parity or
performance baseline. Full isolated workspace gate passes (2384 Rust test
executions, zero failures, 101 explicit skips; release/installer/scripts).
Next: reconcile catalog/model lifecycle and immutable source selection
through the existing SDK/runtime and actual HF/GGUF consumers; examine
model-level retirement/alias authority without another catalog or scheduler.
Continue remaining physical admission and provider interop only for actual
consumers; this is not Phase 1 closure.

Prior accepted prepared text checkpoint:

**Accepted source checkpoint `7e86aad7d74882baa474a50a584e141f1d58eee3` —
prepared text calls and producer observations.**
Source-template regions derive token and absolute-position ports and original
producer observation sites, including internal values. A borrowed
phase-explicit BoundTextCall keeps the original invocation/phase for
region and state binding. PreparedTextProgram owns the original ModelProgram,
selected provider, row band and observation support; engine verifies the
returned whole-text executable against the original source/plan/provider.
Actual HIP/CUDA resident singleton/cohort preparation uses this contract
before host credit/materialization. HIP reads the original active producer
sites before reuse; unsupported CUDA observations refuse before submission
without fencing the resident owner. Original physical state, scheduling,
resource and publication ownership stays unchanged. Contiguous text
positions only; no standalone partial dispatch, arbitrary state conversion,
general host/IO observation, or graph SDK activation. Focused preparation
tests (8), strict core/engine Clippy and both backend builds pass.
Actual Qwen 0.8B HIP observed 653 producer values over state-only and
logits calls; observed logits equal original readback bit-for-bit. CUDA
unsupported observations refuse before submission, and subsequent ordinary
calls commit on the same session. X2 native V2 C1/C8, exact-reference C1,
multichunk continuation and L5 native C1/C8 paged/contiguous canaries pass
expected tokens/replay and original retirement; no new performance or
independent numerical parity claim. Full isolated gate: 2383 Rust test
executions, zero failures, 101 explicit skips, release/installer/scripts pass.
Both GPUs returned to baseline (L5 37 MiB, X2 259768320 B) with no compute
owners. Exact signed source on X2 by verified incremental bundle; A1 untouched.
Next: reconcile cold capability, provider/device selection and resource
admission/reporting through actual native loader/SDK owners; retain legacy
non-template and backend-specific physical contracts.

Prior accepted private transport checkpoint:

**Accepted source checkpoint `6eae81c68c24c58c85e707baebacbccbda523208` —
explicit private provider transport/completion.**
A bounded CUDA provider scope consumes the original armed ResidentSubmission,
checks actual scratch Write leases and distinct opaque charges, binds the
exact original or same-context secondary queue/client and retains its owner.
A secondary scope inserts actual CUDA event waits both ways; original
readback/status validation alone can construct Ready. Managed private
buffer copies use checked same-layout lease views and the original actor's
device-copy driver/write-taint scope. The existing captured attention copy
shares that transport without changing its native set policy. Failure after
enqueue consumes the original flight and quarantines retained owners; no
independent allocation, scheduler, ledger or publisher. Same-context CUDA
managed staging only; opaque external-pointer imports, cross-device/runtime
transfers, other-backend completion and capture integration remain open.
Physical L5 certified BF16 GEMM with private input/output copies and actual
secondary-stream event ordering passes, including exact readback/status,
no early Ready and interruption after accepted enqueue retaining original
owners in quarantine. Existing event-flight completion/cancellation/unknown
retirement passes. Ordinary Qwen 0.8B L5 CUDA C1 and X2 HIP native V2 C1/C8
pass expected tokens/replay; HIP lane invariance and zero failed GPU quanta.
Original resource retirement passes; L5 37 MiB, X2 259768320 B, no compute
owners. X2 executes the exact signed source via verified bundle; A1 unchanged.
Full isolated gate passes (2382 Rust test executions, zero failures, 101
explicit skips), including release/installer and script checks.
The bridge is internal and not selected by ordinary serving. No new finite
captured-attention qualification, external-runtime support or performance
baseline. Next: remaining implicit region token/position/observation ports
and whole-model prepared execution boundaries; extend concrete interop only
for real consumers, not an additional generic transport framework.

Prior accepted stateful region checkpoint:

**Accepted source checkpoint `262712f1c94daa8423060155633545b632a54e88` —
stateful region effects and original-layout bridges.**
Core derives ordered source-template state writer/read edges and exact
logical plane geometry for native CUDA/HIP preflight. CUDA convolution
history and mixed full-window layouts retain their distinct numerical
contracts; recurrent precision and paired KV storage stay explicit.
ClosedRegion exports state versions 0/1 with their original layouts;
prepared stateful regions require explicit storage and bind the exact
original invocation/advance attempt. Same-layout provisional handoff
checks source/plan/order/version/storage, never grants Ready or publication.
Native state binding shares exact original lease plane validation without
per-domain heap allocation. Existing paged KV staging and distinct root
publication remain unchanged. StateCarry without a declared layout,
arbitrary observations/embedding/rotary and output-policy crossings still
refuse as closed regions. No general standalone stateful region dispatch,
committed-root conversion/recompute or external completion bridge.
Focused preparation (7 tests), strict core/engine Clippy and combined GPU
compilation pass. Physical Qwen 0.8B on X2: native V2 C1/C8, exact C1 and
distinct-prompt multichunk continuation all pass; expected token IDs,
replay, lane invariance and zero failed GPU quanta. L5: native CUDA C1
expected tokens/replay and C8 contiguous-versus-paged matched token
comparison pass (48 prompt + 3 output tokens per request). Original
native/final-reader retirement and CUDA shutdown pass. X2 returns to
259768320 B VRAM, no KFD owners; L5 to 37 MiB, no CUDA compute owners.
The C8 comparison is state-layout consistency, not independent fused-lane
qualification or a new throughput baseline. X2 runs the exact signed
source via verified incremental bundle; A1 unchanged. Full isolated
workspace gate passes (2382 Rust test executions, zero failures,
101 explicit skips), including release CLI, installer and script checks.
Next: extend actual admitted private-buffer transfer and foreign-stream
completion bridges through original physical owners; then close remaining
region input/observation and whole-model preparation boundaries as needed
by real consumers. No standalone stateful dispatch or conversion claim.

Prior accepted provider resource checkpoint:

**Accepted source checkpoint `77e8b4ae0487a5219140727007cc316f43c01663` —
provider requirements and original-flight binding.**
PreparedRegion reports additional provider-local workspace/opaque device
reservations and buffer/completion compatibility. No double-charge of
original tensors, scratch or shared context/module envelopes. Native CUDA
binds the original GEMM/context owner once per flight; HIP dense uses one
workspace requirement across preflight, allocation and execution. Captured
CUDA graph resources use the same requirement contract and replay binds
its actual pinned stream, leases and charge owner at launch. Capture-stop
failure/untransferred ownership retains the graph reservation in quarantine.
No arithmetic, public graph selection or allocator/queue change.
Focused preparation, strict core/engine Clippy and GPU feature builds pass.
X2 Qwen 0.8B native V2 C1/C8, exact C1 and distinct multichunk
continuation, and L5 CUDA SDK C1 pass expected tokens and replay/retirement.
The original GGUF CUDA graph B2 lifecycle passes full-vocabulary logits,
all 42 state domains, compatible replay, incompatible-profile fallback,
recapture and retirement before workspace replacement; final reserved
bytes are zero. Actual driver-internal bytes remain unqualified.
Both GPUs returned to their initial idle memory with no compute owners.
Full isolated workspace gate passes (2381 Rust test executions, zero
failures, 101 explicit skips); X2 runs the exact signed source received
by verified bundle. A1 unchanged.
Next: close stateful region effects/bridges on original state owners;
reuse existing physical event/queue ownership for explicit external
completion and transfer bridges when needed by a real consumer. The
current adapters refuse private buffers and foreign completion. No new
numerical profile, throughput baseline or wider backend parity claim.

Prior accepted physical-region checkpoint:

**Accepted source checkpoint `0db60ede170634d6be5f9ffecd82939c2745ab95` —
prepared physical region ports.** PreparedRegion
replaces bare prepared scope with original source/plan and selected
provider/mode, finite row band, per-port storage encoding and checked
physical row layout. Native CUDA projection owns the contract; HIP
admission uses its F32 layout. The original text model validates invocation
plan identity before singleton/cohort admission. Existing native flights
borrow exact lease views (CUDA packed BF16 row/inner-major, HIP widened
F32); no independent allocator, queue, stream or completion owner.
Standalone partial dispatch, captured/compiled workspace and external
completion bridges, and stateful effects remain open. Focused preparation,
allocation identity, strict core/engine Clippy and combined GPU builds passed.
The complete isolated workspace gate passed (2381 Rust test executions,
zero failures, 101 explicit skips). Real Qwen 0.8B on X2 native V2 C1/C8,
exact C1 and distinct multichunk continuation, plus L5 CUDA public SDK C1,
all passed expected tokens/replay and original retirement. X2 received
the exact source commit by verified bundle; both GPUs returned to idle
(259768320 B on X2, 37 MiB on L5), no compute owners. A1 unchanged.
No fresh performance or wider parity claim. One feature-dependent unused
range-check helper warning remains in a non-consuming GPU build.
Next: workspace/persistent-resource requirements and original
stream/completion ownership through actual provider consumers; then
stateful effects/bridges. Preserve native primitives and original owners.

Prior accepted region checkpoint:

**Accepted source checkpoint `82f895dae795f2bcabf83a95ce52cb6705e64280` —
bounded closed stateless regions.** ModelRegion retains
the original ProgramSource issuer and ForwardPlan; ClosedRegion derives dense
SSA boundary ports, precision and active/last-row layout. Binding requires
the exact invocation plan and output schedule. The existing provider seam
verifies the exact prepared scope; whole-only providers refuse partial
preparation. CUDA projection cold owners retain the region and verify source
issuer/packed geometry. HIP admission checks region capacities against
widened F32 storage. No arithmetic, dispatch selection, allocator or
publication changes. Six engine preparation tests, strict core/engine Clippy
and ordinary CUDA serving compilation pass. The complete
`scripts/check.sh` gate passed with mandatory private-network isolation
(exit 0; 2381 successful Rust test executions across configurations, zero
failures, 101 explicit skips). Initial CUDA cohort accessor compile error
was corrected before the signed source checkpoint and full rerun.

Actual approved Qwen 0.8B source on the exact checkpoint: X2 native V2
C1/C8 (three waves each), exact arithmetic C1 (three waves), distinct-prompt
native multichunk singleton/continuation test; L5 CUDA public SDK C1
(three waves, existing optimized test profile). Fixed-prompt IDs remain
248068/198/90700, replay exact, HIP lane invariant and zero failed quanta;
native final reader/retirement and CUDA shutdown pass. X2 returns to
259768320 B and no KFD owners; L5 returns to 37 MiB and no CUDA compute
processes. These are bounded correctness canaries, not fresh independent
reference or throughput/parity qualification. X2 received the exact signed
source via verified incremental Git bundle; A1 was unchanged. No push.

Next: extend the existing bounded region contract through real consumers
for explicit physical capability/shape misses, workspace and external
buffer ownership, state effects/bridges, completion and retirement,
including captured/compiled preparation scaffolding. Standalone partial
execution and general stateful region composition are not implemented.
Keep one original plan/source and the existing physical fabric; no new IR
or graph SDK. Broader Phase 1 foundations remain open.

Prior accepted source-preparation checkpoint:

`8c34c472` unifies borrowed/window-backed source preparation through typed
ProgramSource, original ForwardPlan and PreparedExecutable; real window-backed
Qwen HIP/CUDA uses the same engine prepare boundary. No inference arithmetic
changed. X2 host source/final reader, native V2 C1/C8, distinct multichunk and
exact-reference C1 pass; C8 warm552.423ms/43.44tok/s, unchanged dispatches,
expected tokens/replay/lane invariance, original idle retirement. CUDA
test-only import restored; GPU feature boundaries and CPU Clippy pass.
The assembled source-preparation acceptance gate now passes in full:
`scripts/check.sh` completed under the required private child network namespace.
Physical budget refusals assert typed Resource with requested > available;
configured/overflow/disk-quota limits retain Limit and rollback checks.
Empty serving model declarations refuse before backend selection, valid
unsupported backends retain their own error, and isolation-test fixtures no
longer depend on the outer environment. CPU and optional acquisition sweeps
passed (1102 and 592 tests respectively), with explicit qualification skips.
No production inference or resource-admission change in the acceptance fixes.
Retain the original X2 physical evidence; no fresh CUDA physical validation.
Whole-program source preparation is accepted; the bounded next region
slice is described above. The broader Phase 1 foundations remain open. See the full plan and composition TODO.

Full dated takeover snapshot:
[`FULL_COMPREHENSIVE_HANDOFF_2026-11-25_FOUNDATIONS.md`](FULL_COMPREHENSIVE_HANDOFF_2026-11-25_FOUNDATIONS.md).
The 25 Nov pack covers the original architecture, complete historical chain,
all accepted checkpoints through `fdfb467f`, all 830 reachable commits
through the last acceptance documentation checkpoint, open work and the
next parameter-staging unit. The [24 Nov foundations handoff](FULL_COMPREHENSIVE_HANDOFF_2026-11-24_FOUNDATIONS.md)
and earlier packs remain intact; their dated continuation prompts are
superseded. The newest main body is the first reading; inventories and
historical packs are navigation aids, not a competing task ledger.

The earlier [24 Nov morning takeover](FULL_COMPREHENSIVE_HANDOFF_2026-11-24.md)
preserves the pre-native chronology at `e1929762`. The previously concurrent
FlashInfer/paired-GDN slice was committed/pushed as
`6d5dc72d3711cd3b0e14db0a3c007851984a0745`; source is landed but fresh
physical CUDA correctness/provider qualification remains open. Revalidate
actual Git state and use the current ordered plan, not older next-step prose.

This file remains the maintained current map and pointer into live source, tests
and plans. Git history retains superseded checkpoints; current source, tests and
live TODOs are authoritative.

### 24 Nov — native throughput priority (supersedes scalar tuning queue)

Current user's direction: attack order-of-magnitude gaps before marginal kernel
wins; no test/evidence proliferation; keep reusable primitives and clean
zero-cost ownership. The new owned ROCm provider is the next real-model
integration seam, not another scalar microkernel campaign.

- e80216b0/591913f5: retained BF16 wave32 WMMA with actual gfx1151 mapping,
  partial/strided output, finite checks and direct timings. Mixed B1/expansion
  results; no selection.
- 7b37d338/9b9e4996: new optional Linux rocBLAS provider on the existing
  PreparedHipNative set and HIP actor. External workspace is an exclusive
  resident lease. The library/handle is retained until actual stream
  synchronization and destruction; uncertain state fences rather than
  releasing code/ownership. Persistent context caches remain charged.
  Physical unscoped original-actor qualifier passed four cycles, mean
  .094436 ms on 6144x1024, 128 input rows versus 1.934770 ms scalar
  (~20.5x kernel-level opportunity). No model-level speedup yet.
- `312bbccd` / `58a10d3c`: real native source/cohort integration, finite
  upload certification, once-admitted resident scratch, explicit
  `HipRocBlasBf16MixedV1` identity. SDK `ModelLoadRequest::qwen_hip_native`,
  CLI `run-rocm MODEL --numerical native`; default exact source preserved.
  Actual projection-only X2 C8 0.8B/2B 30.978/23.737 tok/s; 27B C1 1.979
  tok/s. Resident qualified gfx1151 only; no streamed fallback or new engine.
- `6c31363d`: six strided-batched FP32 GEMMs for four dense GDN chunk
  products, original pointwise epilogues and solve retained. Native GDN
  numerical contract explicit, same original actor, scratch and state roots.
  Physical NN/NT/TN batched layout qualifier, 0.8B/2B C1/C8, distinct
  multichunk continuation and 27B C1 pass. Final C8 39.778/29.088 tok/s,
  27B C1 2.201 tok/s; physical idle baseline restored. See current measurements in CURRENT_BASELINE.
  Next larger gaps: native triangular solve, mature fused attention and
  graph/dispatch amortization; benchmark real end-to-end execution early.
  Standard optimized libraries first: production cuBLAS/cuBLASLt integration
  on NVIDIA (existing owned diagnostic provider), cooperative-matrix
  GEMM via mature ggml Vulkan shaders where capability and licensing allow.
  Preserve all custom and exact-reference primitives; avoid duplicating
  the execution fabric or silently changing precision.
- `e1901b16` / `79a9073a` / `9a571088`: shared compact row-major
  `Gemm` and `UnitLowerSolve` core geometry, borrowed GPU command and
  HIP lowering; existing CUDA owned BLAS diagnostic accepts the same
  GEMM descriptor. No new provider owner/registry or CUDA production claim.
  In-place solve validates original read dependency before new write scope.
  Owned physical qualifier passes all original GEMMs and unit-lower solve.
  New native V2 arithmetic identity (batched TRSM with original RHS copy/
  finite checks); V1 source binding/loader and exact source solver retained.
  Actual 0.8B/2B C8 warm549.986/775.460ms (43.638/30.949tok/s), C1
  123.070/281.740ms; exact expected output, replay, distinct state-only
  continuation, exact-source C1 canary, 27B C1 at68GiB1346.401ms/2.228tok/s,
  original idle retirement. Native SDK/
  CLI selects V2, exact source stays default. Next native row-wise and
  attention regions and dispatch amortization, not another general framework.
  See model execution TODO for the shared operator composition boundary.
- X2's existing rocBLAS 7.2.4 closure
  /nix/store/ngf5gxxijv293l72lb1jdc77b7y2zjrd-rocblas-7.2.4/lib
  matches the already recovered HIP/glibc runtime. Use the existing local loader
  recipe (see X2 runtime recovery below), adding this lib directory to its
  library path; no SDK acquisition. Physical qualifier:
  `hip::qualification::actual_native_rocblas_lifecycle` with hip,qualification.
  Native selection is explicitly opt-in on the qualified resident route;
  real-model checks above supersede synthetic-only evidence. The workspace
  gate passes formatting, architecture, combined GPU feature checks and CPU
  Clippy; it still stops at the pre-existing cache_prepare.rs:142 CPU
  admission assertion. No new CUDA or Vulkan physical performance claim.
- The missing pinned vLLM closure remains a blocker for a fresh paired
  comparison. CURRENT_BASELINE records the retained comparison and native
  provider opportunity separately.

### Host authority — 24 Nov 2026

- L5 / `dotlink---l5` is the only source-authoring host; use the single persistent
  `main` worktree at `/home/pvl/spaces/abird/src/zerogravity`.
- X2 / `dotlink---x2` is execution/testing only for AMD/ROCm large-VRAM work; move
  exact L5 commits by verified Git bundle rather than pushing merely for transport.
- A1 is execution/testing only for mixed NVIDIA SM89 + AMD gfx1150 build/runtime
  validation. Source changes land on L5 first and exact commits move by bundle.
- All hosts use repo `/home/pvl/spaces/abird/src/zerogravity`, cache
  `/home/pvl/.cache/abird/zerogravity`, `CARGO_BUILD_JOBS=8`, and
  `RUSTC_WRAPPER=/run/current-system/sw/bin/kache` unless concrete resource pressure
  requires less.
- Physical GPU work goes through `python3 -B scripts/native-gpu.py -- ...` when the
  owning qualification requires the native-GPU envelope; loopback traffic uses the
  accepted private-network envelope and `--require-private-network`.
- Source checkpoint immediately before the 24 Nov handoff docs is
  `e1929762a87b1e34de099d9f40b85bcff385c27d`, and tracked `origin/main` matched it.
  Revalidate actual HEAD because the handoff-doc commit may sit above that source point.
- There is no longer a concurrent dirty FlashInfer/paired-GDN slice. The complete
  previously protected work was explicitly committed/pushed as
  `6d5dc72d3711cd3b0e14db0a3c007851984a0745`; revalidate its current CUDA physical
  correctness/provider cells before treating it as new performance authority.

### Production build/run convergence revalidation — 23 Nov 2026

Exact committed `fb20345bb7b5ce0b3285470c77564e492331254b` was archived outside
the active worktree so the concurrent FlashInfer/paired-GDN dirt could not affect the
result. In `nix develop .#gpu`, with `CARGO_BUILD_JOBS=8` and the required kache
wrapper, `zerogravity-cli --no-default-features --features all` passes `cargo check
--locked` and an optimized `cargo build --locked --release`. The emitted 69 MiB
binary has SHA-256
`cd3df0f8173dc55ecda00af4d078214f51c9963a058bb15c4e4a06dfaf8b9e9f` and
truthfully reports CPU, Qwen3.5, text, OpenAI HTTP, Vulkan, CUDA and HIP/ROCm
capabilities; `run-cuda` and `run-rocm` expose the same canonical run surface.
This closes fresh source/build revalidation of the production feature algebra without
creating another feature-matrix fixture.

A1 and X2 were both fast-forwarded by verified Git bundle for execution-only
follow-up; neither origin was pushed. A1's exact mixed-GPU shell is now realized and
reports ROCm7.2.3 on Radeon890M `gfx1150` / wave32. Its cache had the exact complete
0.8B payload but only partial cache-control metadata, so only L5's canonical 201-byte
`policy.json` plus the empty protected cache-directory skeleton were synchronized; no
model payload was transferred. The 2B payload on A1 is incomplete, so do not copy
gigabytes merely to fill that canary.

The first real public 0.8B run exposed one stale product refusal rather than a HIP bug:
the source GPU adapter rejected `chat=true` even though the approved Qwen tokenizer
already owns the exact single-user ChatML framing and its prompt/scratch bounds. Signed
`6f4418ba8262baff0dc023923865ea8bcfb7d63c` deletes only that 17-line refusal and
falls back to the existing `PreparedModel` defaults. A clean HIP+Qwen production
composition check passes, then exact-commit A1
`run-rocm qwen3.5-0.8b --prompt hello --max-tokens 2 --context 512` succeeds through
the public Runtime seam **without** `--raw`; two tokens are generated and the KFD owner
set returns exactly to its pre-run baseline with no lingering ZeroGravity process.
SM89 CUDA remains deliberately unqualified until the remaining SM86-closed leaves are
generalized and physically qualified.

### Terminology architecture — 21 Nov 2026

Active naming follows [TERMINOLOGY.md](TERMINOLOGY.md). The first cleanup
separates model identity from runtime policy
(`QwenModelVariant::{Qwen800M,Qwen2B}` vs `QwenServingProfile`), shortens
arithmetic names to `NumericalProfile` / `NumericalContract`, replaces
`native_rows` and bare B-width scheduler language with explicit
`cohort_width` / fused cohorts, and stores historical GDN captures under
`reference/<variant>/<phase>/<tokens>`. The second cleanup is now complete:
the normal Qwen source-backed path is `QwenForwardPlan` /
`QwenForwardInvocation` with one backend-neutral `ForwardSourceSeal`; CUDA
mechanics live under `cuda_forward` / `CudaForward*`, and active qualification
uses semantic `prefill 47 + decode 48–50` names plus `FORWARD_*` env/receipt
labels. No compatibility aliases retain the superseded execution vocabulary.
Historical chronology/schemas/provenance remain intact. The third cleanup is now
complete too: the old `SourceCall*` family is replaced by the core semantic pipeline
`InvocationInput -> BoundInvocation -> TextInvocation { TokenSpan, InvocationPhase }`;
Qwen layers model semantics as `PromptInvocation`, `QwenGdnInvocation` and
`QwenModelInvocation`; live execution ownership is `ResidentText*`; the backend body is
`text_execution`; and observation is `TextFrame` / `TextObserver`. Active developer
interfaces use `TEXT_INVOCATION_*`, `RESIDENT_TEXT_*` and
`ZG_QWEN_TEXT_INVOCATIONS_MANIFEST_SHA256`. The legacy reference schema
`zg.original-hf-source-call-reference.v1` and captured status
`CAPTURED_ORIGINAL_SOURCE_CALL_REFERENCE_ONLY` remain unchanged compatibility values.

The fourth terminology cleanup is now complete: active committed-feedback execution uses
`autoregressive` throughout. The broad runtime family is rooted at
`native_fragment::autoregressive`; inputs/state/publication use `AutoregressiveInput`,
`AutoregressiveState`, `AutoregressiveCommit`, `AutoregressiveFeedback` and
`AutoregressiveSlot`; cold/reuse/fault/observer/receipt ownership follows the same prefix;
fixed-prompt paths use `FixedPromptAutoregressive*`; and public methods/modules use
`autoregressive_*`. The broader old `natural_publication` module was intentionally shortened
to `autoregressive` because it owns the whole feedback lifecycle, not only publication.
Frozen external Qwen27B/Gemma reference schemas, artifact filenames, receipt fields and the
historical `qwen27b_hip_natural` diagnostic entrypoint remain compatibility/provenance
vocabulary; its code now calls the new runtime API without rewriting the frozen evidence
format. Active Qwen35 CUDA/batching and Qwen text diagnostics now use autoregressive
feedback/history/IDs as well.

The fifth cleanup retires `all42` as an engineering term. Active APIs, tests, diagnostics,
comments and receipt labels now say **all state domains** / `all_state_domains`; the number
42 remains only an implementation/model property. The accepted Qwen35 reference archive
coordinates `plain/all42/...`, `all42-prior/...` and key
`all24_all42_post_validation` remain unchanged compatibility/provenance values and are not
runtime vocabulary.

The sixth cleanup retires model-size shorthand from active execution terminology: complete
model execution is **full model** / `full_model`, complete layer coverage is **all layers** /
`all_layers`, and complete vocabulary coverage is **full vocabulary** / `full_vocabulary`.
Active production diagnostics, errors, tests and receipt labels no longer use `full24`,
`all24` or `fullV`. The accepted Qwen35 key
`common_final_fullv_greedy_validation` and the older
`all24_all42_post_validation` key remain unchanged provenance/compatibility values. Real
numeric dimensions such as `full248320` are not terminology and remain untouched.

The seventh cleanup retires fixed-trajectory archaeology from active code: the 47-token
prefill is **prefill 47** / `prefill47`; decode continuation is named by explicit bases or
frontiers; and `canonical47`, `cached47`, `true47` and `case5` no longer appear in active
runtime/test terminology. The frozen batching oracle profile `canonical47-repeated-v1`
remains unchanged as a compatibility/provenance value.

The eighth cleanup retires provider/workload shorthand. The qualified max-eight fixed-prompt
lifecycle is the **eight-session benchmark profile** and keeps the explicit Rust variant
`FixedPromptLifecycleProfile::EightSessionBenchmark`; the smaller profile is `TwoSession`;
its tests/helpers use `eight_session_*` and
its resource contract is stated directly rather than as `active8`. Production temporal
prefill is described as **4-token equal-frontier prefill chunks** rather than `temporal4`.
The generic scheduler's real `prefill_quantum: 32` value remains unchanged; only the opaque
`quantum32` nickname was removed.

The ninth cleanup retires bare row-width/concurrency shorthand from the active semantic
surface. Scheduler/config code uses `cohort_width`; prose uses **singleton** and
**2/4/8-row cohort**; current workload prose uses `active=N` / `concurrency=N`.
`B4`/`B8` remains intentionally in fixed physical ABI/kernel/artifact names and historical
provenance, while `C*` remains only in historical benchmark coordinates. Public CLI help,
high-level scheduler/qualification assertions and the normative batching design now use
explicit row-count language; qualification receipt markers are preserved as evidence
coordinates rather than churned for style.

The tenth cleanup resolves the deferred `ordinary RMS` arithmetic name. The typed operator
is **zero-centered RMS**: the checkpoint stores an offset and runtime gamma is
`1 + checkpoint_weight`. Core contracts/plan nodes are `ZeroCenteredRmsContract` and
`TypedZeroCenteredRms`; CPU numerics are `ZeroCenteredRms` / `zero_centered_rms`; Qwen
operator slices/reference helpers use `zero_centered_*`. Qwen residual RMS and BF16-text /
gated direct-gamma RMS remain separate contract families. The pinned CPU golden bytes are
unchanged; only their owning module/test path moved from `ordinary` to `zero_centered`.
Historical milestone prose retains the old name where it records the terminology current at
the time.

Cumulative terminology validation remains green across strict core/Qwen/GPU/public-CUDA
Clippy plus public CUDA/HIP, CUDA qualification, 27B HIP qualification and CUDA CLI
compilation. Core invocation tests pass 10/10; the Qwen forward suite passes 5/5; focused
zero-centered RMS validation passes core typed-plan 3/3, pinned CPU goldens 2/2 and Qwen
slice/backend 4/4 with the golden SHA unchanged; the standard CUDA GPU lib-test and Qwen35
CUDA integration targets compile; and the frozen
`qwen27b_hip_natural` example compiles against the new autoregressive API. A combined
`cuda-native-qualification,cuda-cublas` GPU unit-test attempt gets past the stale invocation
test accessors fixed here and stops only on the pre-existing `CudaDiagnostic.compiler`
test-matrix drift in unrelated attention/GEMM/RMS tests. The renamed `cuda::text_forward`
static suite also retains its pre-existing 16/17 fixture drift: one assertion expects five
`!isfinite(mean)` / inverse-zero guards while the byte-identical kernel already contained
six at `b88d3b3`. A focused `qwen35-cuda-qualification` public lib-test attempt for the new
eight-session profile tests is additionally blocked before test execution by unrelated
pre-existing test-only drift: `gpu/cohort.rs` references a stale `native` variable and
`qwen_source/execute.rs` still calls the retired `fused_phase_compatible` helper. Production,
qualification and integration compilation remain green; do not repair those test-matrix
issues absent direct relevance. No physical GPU run was added for this behavior-free naming
slice.

### Current performance baseline — 22 Nov 2026

The active performance authority is now CURRENT_BASELINE.md.
It contains only the fresh current-path CUDA/ROCm measurements for 0.8B, 2B and 27B,
including the matched Nix-native vLLM baselines and L5 27B SSD-streamed result.
Historical eager-vLLM, Podman, superseded provider runs and old microbenchmark evidence
are no longer active engineering inputs; use Git history only when archaeology is
explicitly required.

The current implementation priorities implied by that baseline are:

1. fix the fused CUDA C4+ lane-invariance correctness bug before accepting wider CUDA
   performance comparisons;
2. HIP cohorts share aligned prefill and decode projections/stateless work,
   with decode additionally sharing QK/rotary and the18 GDN Conv4 updates.
   A direct private-state recurrent cohort candidate (`11f692b`) stayed exact and cut
   A1 C4/C8 dispatches by ~4.6%/~5.5%, but warm wall time was only ~0.24% better at
   C4 and ~1.83% worse at C8. The selector/load-time preparation was therefore removed
   in signed `7fcc9c4`; the clean row-aware recurrent kernels remain dormant reusable
   primitives for other devices/models/shapes. Do not promote that consumer from launch
   count alone;
3. HIP singleton localization is now measured on A1: one cached0.8B decode token is
   4,474 forward dispatches, of which BF16 projection is3,360 (~75%); the47-token
   prefill is9,080 dispatches with the same3,360 projection share. B1 recurrent
   decay/coefficient consumers (`69630ec`/`0d201ff`) reduced launches but not wall time,
   so signed `879bee7` restores the accepted consumer while retaining those reusable
   kernels dormant. Signed `4070610` removes the singleton projection scatter only when
   input/output leases are distinct:47+3 dispatches18,028→17,653 with warm mean
   ~6.401→6.359s (~0.65%), while a decode-heavy16-output cell improves
   ~32.070→31.350s (~2.25%); it stays selected. Alias widening (`fdc5a8e`) produced no
   extra dispatch reduction and was reverted by `baf4c9f`. Singleton status coalescing
   (`ec6fb57`) cut far more launches but lost to `4070610` on the decode-heavy cell
   (~31.518s vs~31.350s), so `d15b02f` keeps that model-neutral transform dormant.
   Reusing one packed singleton activation across matrices (`e1701e7`) saved144
   dispatches but regressed47+3 warm mean ~6.359→6.613s (~4.0%); signed `1d49ec7`
   restores per-matrix packing and keeps the prepacked entry point dormant. A shared
   exact projection core plus certified in-kernel status (`5d51896`) cut47+3
   17,653→14,182 dispatches and output16 73,384→50,504, but lost wall time
   (~6.419s and~31.886s versus selected~6.359s/~31.350s); `90537b6` removes only
   its selector/preparation and retains the certified primitive dormant. The larger
   finding is cold accounting: an A1 14GiB test envelope has1,503,789,056B weights,
   ~38.4MB workspace and18×45,809,664B roots, but9GiB native-device reservation
   (8GiB `MODULE_OPAQUE` +1GiB max code) forces streaming. A1 HIP reports15.232GiB
   total, so forcing residency under that policy would leave only~17MiB headroom and
   is unsafe. Production `run-rocm` instead derives its budget from current HIP free
   memory minus configured headroom; do not generalize the14GiB A1 benchmark to X2.
   Signed `c5b3fad` now closes that unit with an exact backend-qualified gfx1150 policy.
   The existing four-cycle native lifecycle was rerun on A1's exact gfx1150 / ROCm+driver
   7.2.53211 / RTC9.0 tuple using its pre-existing1GiB opaque policy: all four cold
   compile/load/launch/unload cycles, second-live-set refusal and final retirement passed.
   The four probe modules total81,396B load segments; first cold prepare consumed about
   149MB and first launch added about8.4MB, with the persistent first-use cost retained
   after module unload as expected. The complete real0.8B catalogue then prepared92
   modules with7,843,872B retained-host bytes /2,398,636B load segments and an observed
   cold native delta of150,908,928B. `c5b3fad` therefore keeps the8GiB fallback for every
   other HIP source environment but gives integrated gfx1150 a1GiB first-module allowance
   only after cold nomination and exact actor revalidation of fingerprint, compiler
   options, runtime/driver70253211 and RTC9.0 before module load. Device-free preflight
   remains conservative. Under the same explicit14GiB A1 envelope this switches0.8B
   weights from streamed to resident:47+3 dispatches17,653→9,817 and stable runs land
   around0.53–0.56s (one~0.895s outlier), while output16 drops73,384→31,592 dispatches
   and warm mean~31.350s→~0.962s (~32.6× faster), with exact tokens/replay/retirement.
   gfx1151/X2 was still on8GiB at that point; see24Nov X2 recovery below.
   Resident-path re-localization on exact `fd4611e` changes the singleton picture again.
   One47-row prefill is6,467 dispatches; one cached decode token is1,675:
   projection561 (~33.5%), RMS/residual-RMS489 (~29.2%), GDN360 (~21.5%),
   QK/rotary150 (~9.0%), with all remaining families smaller. C1 warm47+3 is
   ~0.541–0.551s; C4 warm~1.978–2.010s; C8 warm~4.044–4.584s, so C4 gives
   modest goodput gain while C8 is not clearly better per request. Signed `ffd5ff8`
   fuses the BF16 residual add into RMS's first squares pass while preserving the exact
   downstream reduction tree and retaining the standalone residual primitive. It removes
   48 launches per prefill/decode invocation:47+3 9,817→9,673; C4 32,082→31,794;
   output16 31,592→30,824. C4 warm mean improves~1.994→1.973s (~1.1%) and
   output16~0.962→0.951s (~1.1%), exact/replay-stable. C1 remained noisy, so acceptance
   rests on the stronger C4/decode-heavy cells. Two broader RMS launch fusions were then
   rejected while retaining their model-neutral primitives. `b44a55c` fused the exact
   mean→epsilon→sqrt→reciprocal scalar chain using one shared `scalar_value` definition;
   47+3 dispatches9,673→8,260 and output16 30,824→26,330, but decode-heavy warm mean
   regressed~0.951→1.001s (~5.2%). `11e61f8` restores separate scalar stages and keeps
   `scalar_cascade` dormant. `9040492` fused RMS normalize→gamma→store into one exact
   vector-tail kernel; output16 dispatches30,824→28,314 but warm mean regressed
   ~0.951→0.963s (~1.3%). `654adbb` restores the selected tail while retaining the
   shared normalize/weight helpers, fused-tail kernel and alternate launchers dormant.
   Both final dispositions pass HIP+Qwen and all-features compilation. Stop treating
   launch count as the next resident optimization signal. Current source then selected
   `a5c57c`, an exact cached-GDN Q/K normalization that maps one32-lane workgroup per
   value head while retaining the prior scalar normalization primitive.
   Temporary serialized HIP device-event profiling on exact current resident output16
   ranks steady decode device time very differently from launch counts: projection
   ~19.8–21.5ms/token (~60–62%), GDN~6.6–7.5ms (~20–22%), RMS~3.0–3.4ms
   (~9–10%), all other families individually ~2% or less. Prefill has the same order:
   projection~296ms vs GDN~156ms. Projection sub-profiling shows the GEMM body owns
   561.1ms of604.3ms total projection time (~92.9%); pack~22.2ms and status~21.0ms.
   Largest output16 GEMM shapes are3584x1024~171.8ms,6144x1024~103.1ms,
   248320x1024 head~81.7ms,1024x3584~75.1ms,1024x2048~54.9ms.
   Signed `072b791` retains an exact one-wave `project_cpu_bf16_wave32` primitive that
   carries original CPU partials lane/lane+32 separately and performs the unchanged
   +32/+16/+8/final tree. Do not select it on gfx1150: serialized GEMM time regressed
   561.1→577.7ms (~3.0%) and clean output16 timed mean regressed from~0.955s
   (1.002/0.908s) to~1.101s (1.283/0.920s), exact/replay-stable.
   Signed `253a565` then retains exact `project_cpu_bf16_rows2`, sharing one packed
   activation across two adjacent rows while preserving independent64-partial trees.
   Do not select rows2 on gfx1150 either. Decode-only direct timing for3584x1024
   flipped sign across repeats (rows2~62.8ms vs rows1~81.4ms once, then rows2~77.6ms
   vs rows1~66.0ms paired).1024x3584 loses~24%; head248320x1024 loses~4.4%.
   6144x1024,2048x1024 and4096x1024 improve only~2.6–2.9%. There is no clean
   geometry rule with stable aggregate upside, so rows1 remains selected and rows2/rows4/
   wave32 all remain reusable dormant primitives. Stop row-grouping experiments here.
   The exact inner K64 traversal was then probed without changing arithmetic. Signed
   `55477b4` retains a paired packed-load primitive: adjacent lanes share one packed u32
   through `plane_shuffle_xor` while keeping the same64 partials. Reject the gfx1150
   consumer: direct decode timing regressed3584x1024 65.20→153.73ms (~2.36x) and
   6144x1024 42.82→87.91ms (~2.05x). Signed `ba8c5ea` retains a projection-specific
   input packer plus an input-certified GEMM sibling: the pack publishes the same eventual
   shared input status bits once, allowing the GEMM to skip the repeated activation
   exponent check while preserving FMAs/reduction/output status. Reject that consumer too:
   same-commit paired timing is3584x1024 61.786→61.422ms (~0.6% faster) but
   6144x1024 38.045→45.533ms (~19.7% slower). No whole-model A/B was warranted.
   Rows1/canonical checks stay selected; both new primitives remain dormant/reusable.
   Weight-finiteness investigation complete: authenticated source/resident upload
   proves exact bytes/coverage, not finite BF16 values. Keep both exponent checks;
   CUDA qualification finite-window/tied-table checks do not cover production HIP.
   Signed `d717c7b` retains exact packed-word address/index hoisting. Clean direct
   paired device timing (96 launches/window) improves3584x1024 8.313→7.481ms
   and6144x1024 13.377→12.193ms; all accumulator/rounded/status bits match,
   including nonfinite and strided multi-request probes. Actual resident0.8B
   47+16 A/B `224085b1` (C1,14GiB, ABBA, eight timed waves/variant) improves
   warm mean924.5→830.2ms (~10.2%), all16 IDs/replay exact, 30,824 dispatches,
   zero failed quanta and clean retirement. Select the hoisted traversal at cold
   preparation initially for the qualified gfx1150 resident HIP policy;
   native catalogue and launch share the immutable operand choice. Other paths
   keep canonical; prior alternatives remain dormant. Final `3be8a812` CLI 47+16
   canary passed, original KFD owners restored. Temp runner/selector removed.
   `c89d0ee7` restores combined HIP/Vulkan/CUDA-native feature boundaries:
   distinct BF16 producer/KV observer and resident invocation-frame observer,
   shared HIP original8GiB opaque allowance and exact X2 environment gate,
   CUDA-native exact range check, CPU-free CLI refusal expectation and renamed
   eight-session test filter. GPU feature boundary gate plus the corrected2/2
   lifecycle tests pass. Full isolated check.sh reaches CPU workspace tests
   after CPU Clippy and stops at cache_prepare admission Error::Limit assertion;
   full phase gate remains open.
   Current A1 real0.8B resident 47+1/47+16 serialized device-event localization
   (temporary observer at `fd02260`, removed): repeat mean-subtraction gives
   37.27ms/cached step, projection19.48ms (~52%), RMS7.35ms (~20%),
   GDN6.73ms (~18%). Exact greedy prefix/replay and original KFD retirement
   in both physical runs. Largest remaining individual non-projection kernel
   is GDN key_reduce3.10ms/step (36 launches). `7d0b395b` retains the exact
   parallel8x16-subtotal kernel (32 adjacent values,256 threads,one LDS barrier,
   no global scratch; original final carry/rounding/status). Direct gfx1150
   values2048 FP32/BF16 timing improves25.5%/27.8%, finite bits and fault
   status/classification pass. Real0.8B 47+16 `7d3a130b` same-binary ABBA
   (12 timed requests/variant,14GiB) means836.7→830.4ms (~0.75%), medians
   841.4→823.9ms, exact all16 IDs/replay and clean retirement. Gain too small
   relative to variation to justify production selector; retain scalar consumer
   and reusable parallel sibling. Temp runner/selector/catalogue removed.
   RMS scalar review gives no compelling arithmetic cut. `6f2ededf` retains
   exact hoisted two-barrier projection subtrees; direct A1 accumulator,
   rounded output and status are bit-exact (including strided/fault inputs),
   but timing gains are <=1.5% with mixed paired windows. Keep current
   four-barrier hoisted production selection; no whole-model candidate A/B.
   Probe removed. X2 recovery below closes native accounting and qualifies
   the same hoisted projection primitive on gfx1151;
4. treat L5 27B SSD streaming as its own memory-constrained mode: the current exact
   47+3-token cell takes about 305.95 s timed median, so source/transfer overlap and
   streaming mechanics are the relevant optimization target there.

The matched vLLM provider is Nix-native for both CUDA and ROCm. Podman/FUSE is not part
of the current baseline architecture. A1 is explicitly available for execution/testing;
its gfx1150 HIP path is useful for quick physical A/Bs, while X2 remains the canonical
large-VRAM ROCm baseline host.

### X2 runtime recovery and native accounting — 24 Nov 2026

X2's pinned GPU dev shell fails to realize offline (missing rust-overlay input,
CA error on retrieval); its previous vLLM provider is also absent. Do not repin
the flake or weaken TLS. The installed CLR7.2.4 closure is healthy when invoked
with its matching glibc2.44 loader instead of the current system's glibc2.42.
No host/profile changes or model downloads. Native lifecycle and all real
0.8B/2B/27B SDK cells pass. Use the existing population benchmark rather than
new fixtures. Native module allowance now1GiB on both qualified integrated
gfx1150/gfx1151, exact runtime/driver70253211,RTC9.0 and compiler/fingerprint
revalidation before load (`38f20ca3`). Unknown versions/architectures and
device-free/finite/shared qualification retain 8GiB; source-derived code bound
and ownership unchanged. `99eb7990` now selects the hoisted packed traversal
for both qualified integrated gfx1150/gfx1151 resident policies. X2 profiling
(`533c72c`) separates prefill from cached decode:0.8B C1 199.62/23.65ms,
C8 1,589.56/87.68ms;2B C1 332.48/41.14ms. GEMM dominates; GDN triangular
prefill is next (32.83ms/C1,262.65ms/C8). `f7ebf647` retains exact
request-tiled weight reuse (2/4 input rows), but direct timing loses.
Existing hoisted sibling wins all12 direct cells. `c1cd04f3` same-binary
ABBA improves all six real0.8B/2B C1/C4/C8 means by7.75–13.83%, exact IDs
and lane/replay/ownership, dispatch count unchanged. Temp probe/selector
removed. Final0.8B24GiB/27B68GiB C1 canaries pass at211.004/5,376.029ms
warm means, exact tokens/replay and original retirement. GPU feature and
CPU Clippy gates pass; full gate remains blocked by existing CPU cache
assertion. `393d0f56` adds the reusable separately rounded guarded FP32
`mul_then_add_normal` with exact zero-product handling and original widened
fallback; guarded triangular shares the canonical sequential64-row solve
and four-accumulator reduction. Direct gfx1151 finite bits/status exact,
including padded rows/faults;47/64 active rows device time falls63.1%/46.8%.
Existing scalar-body arithmetic check passes both denormal modes.
`cd86a965` actual0.8B/2B C1/C4/C8 ABBA at24GiB,47+3 improves all six
means4.74–10.95%, exact outputs/replay/lane invariance, unchanged dispatches
and clean retirement. `4a5961da` selects guarded triangular at cold
preparation only for qualified gfx1151; same immutable choice seals catalogue
and launch. Canonical solver/reference route retained; temporary probes
and switch removed. Final0.8B24GiB/27B68GiB C1 canaries pass at
190.322/5,301.120ms warm means, exact IDs/replay and original retirement.
GPU feature boundaries/CPU Clippy pass; full gate still stops at existing
CPU cache admission assertion. `2ed9c10f` now implements actual shared HIP
prefill using the existing cohort traversal, admission and completion owners:
aligned equal-length2..64-token chunks, same output policy, private state roots,
one-token tails singleton. `282a3437` / `2eee25fa` share stateless operators
and direct single-matrix output; `fd914f63` bounds resident projection tiles
at128 input rows while streamed weights retain whole-cohort traversal.
Current real0.8B/2B C4/C8 warm means657.981/1283.333ms and1249.184/2531.391ms
(3.5–6.9% lower versus previous checkpoint), exact IDs/replay, three actual
fused cohorts/wave,29,564/55,695 dispatches, zero failed quanta.
Distinct-prompt complete state/logit differential passes22,922,240 exact
FP32 words across state-only/terminal and discard/replay; public81-token
four-session distinct-prompt test observes state-only fusion and exact
singleton-equivalent outputs. Clean retirement. Final feature boundaries and
CPU Clippy pass; full isolated gate stops at the existing CPU cache-admission
assertion. Next localize selected shared prefill costs; remaining stateful work is private per session.
See CURRENT_BASELINE; no fresh paired vLLM claim.

Local X2 recovery recipe (revalidate store paths after GC; use the normal locked
shell once available). Build with existing Nix cargo/rustc1.95, gcc15.3 and
pkg-config on PATH, kache, four jobs given ~10GiB available host RAM,
`CARGO_PROFILE_RELEASE_LTO=false CARGO_PROFILE_RELEASE_CODEGEN_UNITS=16`;
run `cargo test --no-run --locked --offline --release -p zerogravity --no-default-features --features qwen35-hip,zerogravity-gpu/qualification --test qwen_hip_sdk`
through the private-network native-gpu wrapper. Use the emitted binary:

```sh
py=/nix/store/xahxrjs30p3mw0a42bqg9ikgkikskw8m-python3-3.14.8/bin/python3.14
glibc=/nix/store/pqyiidz5mi73wv5f91d48s9plcqyj5bz-glibc-2.44-57
hip=/nix/store/nyccgqvlq59jxsk7klrxipq260l1ibin-clr-7.2.4
libs="$glibc/lib:/nix/store/l3msfz6rylwsyhnjzsbr8ylfshh1xw5f-gcc-prefix/x86_64-unknown-linux-gnu/lib:/nix/store/7a2cjhcgzyym5zaz6m4fksw47qvm150w-gcc-16.2.0-lib/lib:$hip/lib"
export HIP_VISIBLE_DEVICES=0 ROCR_VISIBLE_DEVICES=0 HIP_PATH="$hip" ROCM_PATH="$hip" ROCM_HOME="$hip"
export PATH="$hip/bin:$PATH" PYTHONDONTWRITEBYTECODE=1
export HIP_DEVICE_LIB_PATH=/nix/store/yk7752h3h1k7r66h509ps2k0q3bnyg6v-rocm-device-libs-22.0.0-rocm/amdgcn/bitcode
"$py" -B scripts/native-gpu.py -- "$glibc/lib/ld-linux-x86-64.so.2" --library-path "$libs" "$binary" --ignored --exact population::public_source_population_benchmark --nocapture --test-threads=1
```

Set the existing `ZG_QWEN_HIP_SDK_MODEL`, `ZG_QWEN_HIP_TOTAL_DEVICE_BYTES`,
`ZG_QWEN_HIP_BENCH_CLIENTS`, and `ZG_QWEN_HIP_COHORT_WIDTH` for the chosen
cell (model labels `08`, `2b`, `27b`); check shared free capacity and KFD owners before/after. Use an external
periodic VRAM/status heartbeat for long silent X2 loads; don't add a production
telemetry loop. Scope loader/library-path overrides to the test process, never
the host shell. Retained vLLM measurements are the reference until the pinned
provider closure is restored; these are not fresh paired performance claims.

### Error architecture — 21 Nov 2026

The cross-layer error contract is now intentionally small and typed:

- `Malformed` = invalid/corrupt external input or artifact.
- `Unsupported` = requested capability/semantic is not implemented.
- `Limit` = deterministic configured/static bounds and arithmetic overflow. It must
  not masquerade as a measured memory/VRAM shortage when requested/available bytes are
  known.
- `Resource(ResourceLimit)` = a real resource budget cannot fit the attempted work.
  It carries resource, operation, exact requested bytes, exact available bytes and an
  optional scope. Shared physical/resource admission includes device identity, owner,
  already-reserved bytes and capacity in that scope.
- `State` = invalid lifecycle/internal state; `Io` retains the underlying I/O cause.

Do **not** parse error prose to choose scheduling behavior. Prelaunch typed
`Resource` failures become capacity/backpressure where the scheduler explicitly owns
that policy; post-submission failures remain fatal. Resource pressure in optional
optimizations may fall back only at an already-defined typed capacity seam.

`ServiceError::inference` is the single public classification boundary:
`Resource -> 503/resource_exhausted`, deterministic `Limit -> 400/limit_exceeded`,
`Malformed/Unsupported -> 400`, and `Io/State -> 500`. HTTP 5xx bodies are
`server_error`, not `invalid_request_error`. Runtime/server reverse adapters use the
shared conversion and no longer wrap `ServiceError::to_string()`, avoiding duplicated
status/code prefixes. Real host/VRAM admission paths in the resource arbiter, resident
allocator/state/checkpoint, CUDA-forward/resident-text admission, HIP source budgets,
shared BF16 Qwen/Gemma paths and backing/model sub-admission now report concrete
requested/available bytes. Static shape/file/queue bounds deliberately remain ordinary
`Limit` errors. Preserve this distinction for new models/backends.

Validation for this slice is green on the default workspace, CUDA-native GPU tests,
Qwen CUDA serving, Qwen HIP serving, Gemma4 HIP and Gemma4 Vulkan shared-BF16 surfaces,
the existing resident-ledger concurrency/pressure tests, resident-text scheduler capacity
tests, focused core/engine/HTTP classification tests and strict Clippy for the changed
production crates. One combined Gemma HIP→Vulkan compile was externally SIGKILLed while
entering Vulkan dependencies; the HIP check had already passed and one cached
Vulkan-only retry passed, so no source failure follows.

## Resume here

### Repository/topology checkpoint

- **Repository/source authority:** use `dotlink---l5`; work only in
  `/home/pvl/spaces/abird/src/zerogravity` on `main`. Do not create another Git
  worktree.
- **AMD large-VRAM execution/test host:** use `dotlink---x2` only for running and
  testing, in the same path `/home/pvl/spaces/abird/src/zerogravity`. Do not use
  X2 as an independent source-authoring branch; source changes land on l5 first.
- Accepted native-GPU harness checkpoint is
  `6b920a0a91e534eae86f81e603cf419dea3b207b`
  (`fix(harness): reuse existing sandbox isolation`). The paged-state lineage
  remains `e6aac53` -> `3041457` -> `398e446`: `e6aac53` keeps the
  lifecycle qualifier cold-only, `3041457` corrected restore qualification,
  and `398e446` added the source-real lifecycle target; `1413f72` is the
  runtime/source-policy checkpoint; `b9b04aa` remains the accepted current-root
  gather/KvInPlace/scatter/Ready candidate execution; `cb57e1d` remains the
  physical pool substrate and `93468b1` remains snapshot/restore/inherited-Ready
  authority. Current main additionally carries the 19 Nov true in-flight
  cancellation implementation/qualification described below.
  `1413f72` added `CudaSourceStatePolicy::{Contiguous,Paged}` at the original
  resident-text loader seam; at that historical checkpoint the old loader spelling
  retained an exact Contiguous wrapper. Current source names the CUDA entrypoint
  `load_cuda_forward_source`; the explicit policy-bearing loader installs hybrid
  Vector/staging/page pools only for `Paged { page_tokens,
  max_append_tokens }`. At `1413f72` Paged was B1-only; current main now permits
  native rows 4/8 through the same hybrid state owner. Zero policy extents still
  fail closed. `HybridStateInventoryPlan` rejects max_append_tokens above state
  capacity, and runtime candidate construction returns `Limit` before checkout if
  a call exceeds the admitted append bound.
  At `1413f72` the high-level/public `ModelLoadRequest` was intentionally
  unchanged and production registration passed `CudaSourceStatePolicy::Contiguous`
  explicitly. The 19 Nov follow-up added an explicit policy-bearing high-level request
  while preserving `QwenCuda` as Contiguous **at that checkpoint**. That fixed default is
  superseded by the21 Nov generic smart-config layer recorded below; the explicit request
  now supplies a user value for that knob under Enabled mode. Pre-source-IO unit tests
  prove invalid paged/native/bound policies return
  `invalid_source_state_policy` before cache/device access.
  The real managed L5 cache is now visible at
  `/home/pvl/.cache/abird/zerogravity/models` with the exact approved0.8B
  snapshot `d7c7bf56...` and2B snapshot `88606494...`.
  The original-HF differential qualifier
  opens one authenticated source, runs Contiguous then Paged sequentially, and
  compares every full-vocabulary logit word plus all42 state words bit-for-bit at
  frontiers page_size-1/page_size/page_size+1/(2*page_size+1). Success marker:
  `P12_SOURCE_PAGED_B1_PASS ... full_vocab=exact all42=exact
  contiguous_reference=true final_shutdown=true`.
  On original0.8B with page_tokens4/max_append_tokens4 it now passes physically:
  `P12_SOURCE_PAGED_B1_PASS page_tokens=4 max_append_tokens=4
  frontiers=3,4,5,9 full_vocab=exact all42=exact
  contiguous_reference=true final_shutdown=true`.
  `398e446` adds a second ignored source-real paged B1 lifecycle qualifier on
  the same authenticated-source fixture. It proves, once inputs are present:
  prelaunch cancellation leaves root/all42 unchanged; dropping a completed Ready
  candidate publishes nothing; cancellation toggled after Ready yields
  `Discarded` with exact rollback; committed execution crosses page_size-1 ->
  page_size -> page_size+1; restore back to the saved pre-boundary and boundary
  roots reproduces exact all42 state; final model shutdown is clean. Success
  marker: `P12_SOURCE_PAGED_LIFECYCLE_PASS ... all42=exact
  final_shutdown=true`.
  Cold original0.8B paged lifecycle is accepted from retained dotlink job
  `9feb1394-1a3b-498b-8f92-fdd3e607c6b3`: normal `exit_code=0`, 168.08s,
  no contiguous preconditioning, full lifecycle marker
  `P12_SOURCE_PAGED_LIFECYCLE_PASS page_tokens=4 contiguous_warmup=false
  prelaunch_cancel=true abandoned_ready=true cancel_after_ready=true
  restore_before_boundary=true restore_at_boundary=true all42=exact
  final_shutdown=true`.
  `e6aac53` removes the temporary warmup branch and lifecycle stage spam; the
  retained qualifier is cold-only by construction and emits `cold_load=true`.
  The first completed warm run exposed a qualifier bug: restore intentionally
  publishes one fresh root and increments session epoch while restoring the saved
  frontier/state; the test incorrectly required equality with the old snapshot
  RootBinding. `3041457` fixes the assertion to require same source/session,
  root+1, epoch+1, saved frontier and exact all42 bytes.
  Several other0.8B cold attempts terminated intermittently with deployed
  dotlink **v0.17.3** reporting `status=completed`, `exit_code=null`,
  `success=false`, with no Rust failure, OOM/Xid/coredump, live child or VRAM
  residue. This is **not** a ZeroGravity cold-load blocker. Revalidation against
  the deployed `release/0.17` tree matters here: v0.17.3 does **not** contain
  main's newer `SharedProcessExit::Signal` projection. Its runner records
  `JobCompletion::completed(status.code(), status.success(), ...)`; Rust
  `ExitStatus::code()==None` means the process dotlink waited on was terminated
  by a signal, but v0.17.3 discards the signal number. The child dotlink waits on
  for a sandboxed shell command is Bubblewrap. Timeout/cancel/spawn/stdin/spool
  failure paths use distinct non-`Completed` statuses, and the normal
  `Completed` path disarms the process-tree guard before recording completion,
  so the observed null-exit jobs were not dotlink timeout/cancel results.
  Controlled probes further distinguish the case: SIGKILL of an inner child is
  observed by the surviving shell as status137, and SIGKILL of the sandbox shell
  is translated by surviving Bubblewrap to numeric exit137; the intermittent
  jobs instead have null exit code, meaning the Bubblewrap child itself was
  signalled. At the exact failures `dotlink.service` reported memory
  `oom=0`, `oom_kill=0`, pids `max=0`; no dotlink restart/poll failure was
  observed. The source of that external signal is not proven. A clean reproduction
  with no inner tmux/setsid/heartbeat supervisor
  (`native-gpu -> env -> test binary`) still produced the same null-exit shape,
  so the earlier tmux/setsid harness is not required to trigger it. Classify such
  runs as connector-infrastructure incomplete and require a full terminal pass for
  model evidence. Do not add loader workarounds or warmup.
  All Cargo compilation/check/clippy in this slice used
  `RUSTC_WRAPPER=/run/current-system/sw/bin/kache`.
  Final-current available validation is green: four hybrid admission units, all
  eight RTX3060 pool/paging/in-place cells, top-level `qwen35-hip`,
  CUDA tests and strict production clippy. A combined cross-feature wrapper
  disappeared after a terminal-green HIP command while entering the next build;
  separate CUDA commands completed green. One first GPU relink also disappeared
  late; one bounded warm lib-only retry completed.
  Production ordinary sessions still construct contiguous KV today; no serving
  selector, direct paged attention, prefix reuse or public capability changed.
  Parent `8e6f14d3defc0aa2e542c26ffb9ccc7c736f1fcb`
  (`feat(serving): enable qualified temporal B8 prefill`) remains the current
  production temporal checkpoint. The retained width-parameterized qualifier
  fully passes original0.8B and2B B2/B4/B8 with CPU/B1/all42,
  rollback/publication, greedy bits/producer and final reserved0. Production
  HTTP passes B4/B8 on both original sources and retained2B B2; B2/B4/B8 all
  use 4-token equal-frontier prefill chunks while decode remains unchanged.
  B2 production HTTP still proves prompt10 executes as native chunks4+4+2 with
  three publications, two state-only + one greedy, exact CPU text/usage, zero
  failures/refusals and final41MiB. Parent `9dab004` added the reusable temporal
  candidate-chain primitive; `7dea460`/`ba7e376` establish fair aged
  decode/prefill scheduling.
  Earlier `9d6613c`/`599cb28` graph selector/timing remain accepted, but the
  shrinking B2/B4/B8 graph gain does not justify public graph serving; ordinary
  serving remains Native.
- Prefix-scheduling cold reconstruction is now complete for **0.8B and 2B**.
  Fresh-process/no-request-warmup ABBA compares accepted max32-prefix `d0a6084`
  against native teacher-prefix `f788929` with symmetric benchmark-only exact-SM
  cubin compatibility matching later `6a5c610`. 0.8B completes 32/32 exact requests:
  mean wave 17.4307s->15.4529s (-11.35%), TTFT p95 15.0368s->14.9527s and
  delivery-ITL p95 0.8188s->0.7804s. 2B also completes 32/32 exact requests with
  zero timeout/refusal/failure/cancel/discard: mean wave 29.5280s->25.0937s
  (-15.02%), rate 2.1693->2.5512tok/s (+17.60%) and delivery-ITL p95
  1.5598s->1.3234s (-15.15%), while TTFT p95 regresses 23.1625s->23.8931s
  (+3.15%). Before measurement, staged 2B C1/B2/B4/B8 diagnostics proved the
  rebuilt old provider healthy through exact B8; two earlier first-work process
  disappearances remain incomplete compiler-preconditioning attempts, with no
  cgroup OOM evidence. Every accepted physical scope retires to 41MiB. The original
  frozen benchmark binaries/workload are gone, so both cells use retained source-real
  `f788929` HTTP prompts and are reconstructed diagnostics, not byte-identical
  historical reruns, source-correct-goodput evidence or vLLM claims. The P12 prefix
  scheduling performance item is closed; continue with the remaining fused-serving
  paged-state/overlap/fairness and genuine equivalent-source vLLM work.
- **P12-C fair-interleaving sub-slice is accepted at `7dea460`.** Shared
  `zerogravity-engine` fair-round scheduling adds finite `round_tokens` and
  `prefill_age_rounds` above the accepted source/phase/frontier wave builder:
  every eligible decode row executes first, young prefill retains exact
  state/source owners while deferred, and the oldest prefill must make progress
  after two deferred rounds. Admission requires room for one maximum prefill wave
  plus every active decode row. Executor/Ready/publication/cancellation authority
  is unchanged. 18 wave tests, 14 worker tests, strict engine clippy and
  production CUDA/CLI composition pass. Physical unequal-arrival HTTP passes both
  approved 0.8B/2B CUDA sources: B arrives only after A's first actual ID; first
  shared B2 is no earlier than A index 4; both sockets drain; CPU IDs/text/usage
  remain exact; no failed/refused/cancelled/discarded work; final GPU 41MiB.
  Every `scripts/check.sh` command has a terminal-green result on these exact
  bytes under isolated `native-gpu`, including feature-boundary, CPU workspace,
  serving/lifecycle, acquisition and release/packaging blocks. Repeated attempts
  to keep the whole shell wrapper in one connector job disappeared during normal
  cached builds without a compiler/test error, surviving child, cgroup OOM or GPU
  residue; those wrapper attempts remain incomplete. The following P12-C slice
  is the actual multi-token temporal prefill work described below.
  **0.8B B2 temporal prefill is functionally/physically accepted in the
  debug qualification scope.** The retained `zerogravity-gpu` example is now the
  width-parameterized `temporal_prefill`: B2 still proves abandoned 4-token
  rollback, one-root 4-token state publication, one-root 2-token terminal greedy,
  CPU/B1/full-all42 exactness, exact packet/producer and final reserved0. The
  production HTTP path still passes prompt10 -> native chunks4+4+2 with CPU
  ID264/text ` a`, usage10/1, exactly3 B2 publications (2 state-only +1 greedy),
  zero failures/refusals and final41MiB.
  **15 Nov follow-up:** the same retained qualifier isolates width-specific CUDA
  work while full mode keeps B1 mandatory. Original0.8B B4 and B8 now each
  physically pass full mode on the exact `3625348` source: completed private
  4-token chains can be abandoned with roots/all42 unchanged; 4-token state and
  2-token greedy chunks each publish one root; every native row matches CPU and
  B1 all42 state plus greedy bits; packet producer lineage is exact; and final
  reservation returns to zero. B4 retained-band bytes were 1,230,916,660; B8
  1,802,356,148. Earlier full B4/oracle attempts and one exact-current B4
  confirmation disappeared before temporal work, with no Rust diagnostic, OOM,
  live child or GPU residue; the bounded retry passed and the disappeared runs
  remain incomplete, not failures. Two exact-current B2 qualifier confirmations
  likewise disappeared just after session setup before temporal work; the
  already-accepted B2 production/physical evidence is unchanged and those retries
  are not new failures.
  **2B temporal B2/B4/B8 each pass full debug qualification** with the same
  private-candidate/root contract as0.8B. Two unsupervised B4 launches vanished
  before temporal work, but the materially different heartbeat-supervised run
  completed B4 exactly; this establishes an execution-envelope issue rather than
  a B4 semantic failure. On signed `8e6f14d`, original0.8B and2B B8 HTTP now
  each pass exact4+4+2 native B8 execution and accounting; the earlier vanished
  0.8B B8 attempt remains incomplete rather than being relabeled. The retained
  2B B2 HTTP cell also passes exact4+4+2. Production `NativeText` now selects
  4-token equal-frontier prefill chunks for **B2, B4 and B8**, with explicit
  native prefill cohort eligibility for all three widths. These physical proofs
  use the debug Rust profile; canonical thin-LTO/codegen-units=1 B2 is
  still open after another heartbeat-supervised release attempt disappeared
  while compiling/linking `zerogravity-cli`, before the test ran and without a
  terminal child status, Rust diagnostic, OOM/Xid, live child or GPU residue.
  Do not loop that unstable build. Paged-state control/ownership/materialization/
  typed-KV gather/scatter/domain publication/in-place execution/hybrid candidate/
  source-policy plumbing is now
  `0027c28` + `54340a6` + `1eb6e0f` + `3ac2059` + `dc6a188` +
  `53336b3` + `ef9e4d4` + `93468b1` + `cb57e1d` + `b9b04aa` +
  `1413f72` + `398e446` + `3041457` + `e6aac53`.
  **18 Nov original2B source-real paged B1 acceptance is complete.** Signed
  `6b920a0` (`fix(harness): reuse existing sandbox isolation`) now owns the
  native-GPU harness repair. `scripts/native-gpu.py` detects an existing
  Bubblewrap PID/user sandbox instead of nesting another user namespace. If
  Bubblewrap already owns a private network namespace it executes directly;
  otherwise it installs a fail-closed x86_64 seccomp socket-creation filter
  (including io_uring socket-creation bypass prevention) and executes directly.
  Signed `4688cba` extends this generically: if the caller has already entered a
  child netns, the harness accepts it only when `/proc/self/net/dev` is exactly
  loopback, PID1's network view differs, and no inherited socket FDs exist. This
  lets a shared-net connector supply `bwrap --unshare-net --bind / / --dev-bind
  /dev /dev --` without weakening `--require-private-network` or making the
  harness create another user namespace. Outside Bubblewrap it retains the original
  `unshare --user --map-root-user --net` route.
  `--isolation-info` reports the exact planned mode and
  `ZG_NATIVE_GPU_ISOLATION` records it in the child. Commands requiring TCP/UDP
  loopback must pass `--require-private-network`; the shared-network seccomp
  fallback refuses them rather than weakening isolation. `scripts/check.sh`
  now fails immediately in that fallback, and `AGENTS.md` uses
  `native-gpu.py --require-private-network -- bash scripts/check.sh`.
  Focused validation is now31/31 offline harness tests plus Python compile,
  real current-sandbox `--isolation-info`, AF_UNIX allowed/AF_INET denied,
  io_uring_setup denied with EPERM, exact direct exit73 propagation, direct TERM
  propagation as status143, shared-net private-network refusal, and real child-net
  `--require-private-network` loopback+CUDA smoke. Direct invocation still refuses
  this L5 outer shared-net profile; the explicit child-Bubblewrap wrapper is the
  accepted way to supply the required private loopback namespace.
  The upgraded connector now reports external signal numbers explicitly and, in
  v0.17.4, emits one lifecycle log line for every stop it initiates. Follow-up
  connector review found only the explicitly requested `job_cancel` stop in the
  relevant window; a silent `sleep 300` remained alive until that cancellation.
  Therefore the earlier SIGKILL9 observations must **not** be interpreted as a
  dotlink quiet-job watchdog or as evidence that periodic output is required.
  They are best classified as surrounding harness/orchestration artifacts outside
  dotlink's stop paths. Host/service memory limits were unbounded,
  OOM/high/max/pids events were0, PSI was0, and GPU retired cleanly, so they also
  remain non-model evidence. The heartbeat sidecar used below was an operational
  stabilization technique for our harness, not an accepted runtime requirement.
  Under that envelope and approved snapshot `88606494...`,
  page_tokens4/max_append_tokens4:
  - differential job `38c1f321-0347-4a80-b2dc-c6d9d46d0249` completed
    exit0 in703.17s with
    `P12_SOURCE_PAGED_B1_PASS page_tokens=4 max_append_tokens=4
    frontiers=3,4,5,9 full_vocab=exact all42=exact
    contiguous_reference=true final_shutdown=true`;
  - cold lifecycle job `744f36fd-b145-47b9-aa61-084e7d4b414b` completed
    exit0 in438.18s with
    `P12_SOURCE_PAGED_LIFECYCLE_PASS page_tokens=4 cold_load=true
    prelaunch_cancel=true abandoned_ready=true cancel_after_ready=true
    restore_before_boundary=true restore_at_boundary=true all42=exact
    final_shutdown=true`.
  Therefore both approved original0.8B and2B now pass source-real paged B1
  full-vocab/all42 boundary parity and cold lifecycle semantics.
  **19 Nov P12-C true in-flight CUDA cancellation is accepted on original0.8B
  paged B1.** A qualification-only launch witness pauses the original ordinary
  CUDA flight only after real gather/body/scatter enqueue and before Ready. Known
  cancellation waits for physical status readback, marks no candidate Write
  initialized and creates no Ready: root/all42 remain exact, Vector/staging/page
  resources stay held through completion, pool occupancy returns exactly, replay
  commits and final shutdown returns RTX3060 to37MiB. A paired deliberate drop at
  the same submitted cut proves the accepted fail-safe unknown path: root binding
  unchanged, service fenced, event pending,396 handles and source completion
  owners quarantined, no pool reuse/recovery. Two external-SIGKILL attempts before
  semantic completion remain incomplete; direct known-completion and final
  heartbeat-assisted unknown-completion runs exit0. Heartbeat is optional transport
  plumbing only. Run physical cells through `native-gpu`; reserve
  `--require-private-network` for genuine TCP/UDP loopback gates.
  **Historical19 Nov selector checkpoint (superseded for default selection on21 Nov).**
  Public `QwenCudaStatePolicy::{Contiguous,Paged}` was initially available only via
  `ModelLoadRequest::QwenCudaWithStatePolicy`, with `QwenCuda` then exactly Contiguous. **Native paged B2/B4/B8 state transport is now implemented**
  by reusing `OrdinaryCandidateState`: each lane gathers paged KV into its admitted
  staging, executes the existing native cohort body, scatters under the same flight,
  promotes only actual Vector/fresh-COW Writes, then materializes canonical paged
  Domains after completion. Contiguous B8 keeps its private-KV fast path; paged
  `KvInPlace` cleanly uses the existing per-lane append rather than adding another
  paging kernel. Focused policy tests pass, strict CUDA clippy and CUDA-serving
  composition checks pass. The compact original0.8B Contiguous-vs-Paged B2/B4/B8
  prefill+decode/all42 cell is now physically accepted: the same test completed
  exit0 in306.84s with `P12_SOURCE_PAGED_NATIVE_TRANSPORT_PASS widths=2,4,8
  prefill=exact decode=exact all42=exact contiguous_reference=true
  final_shutdown=true`. The earlier two external-SIGKILL attempts remain incomplete,
  not failures. Public SDK integration is also physically accepted:8 concurrent
  requests,4-token page-boundary prefill and2 output tokens produced actual native
  B8 publication in both Contiguous and Paged, identical token IDs and clean
  shutdown (`P12_SOURCE_PAGED_SDK_PASS ... contiguous_cohorts={8}
  paged_cohorts={8} tokens=exact final_shutdown=true`, exit0 in340.78s). Strict
  serving-test Clippy passes. **Public HTTP paged serving is now physically
  accepted too** under the explicit child-private-net wrapper:8 concurrent
  4-token/page-boundary requests with2 outputs observed actual native B8 for both
  Contiguous and Paged, returned identical token IDs and shut down cleanly.
  `P12_SOURCE_PAGED_HTTP_PASS ... contiguous_cohorts={8} paged_cohorts={8}
  tokens=exact final_shutdown=true`, exit0 in358.07s. Strict HTTP/SDK test-target
  Clippy passes. A fresh matched-vLLM attempt did not run: the exact pinned
  `f57d77f5...b4b0dbe` layer store is still present (~25GiB), but its image index is
  no longer registered for this connector's Podman view (`image not known` after a
  temporary user+mount namespace gets past the outer permission/mount issue). The
  canonical managed HF0.8B file tree remains exact at revision `2fc06364...` with
  safetensor `04b1c301...`; do not rebuild/re-register orphaned image layers just to
  manufacture a benchmark. The first full isolated `scripts/check.sh` attempt through the
  child-net envelope found one real source-boundary bug: `HybridStateInventoryPlan`
  is shared by production `cuda-native`, but `ResidentPlaneLayout` was not imported
  for that feature. The import cfg is corrected while CUBLAS-only map/lease imports
  stay gated; targeted `cargo check -p zerogravity-gpu --no-default-features
  --features cuda-native` passes. A warmed full-gate rerun then progressed beyond
  that boundary and was externally SIGKILLed during the later ordinary-Qwen feature
  compile with no compiler/test diagnostic, so the monolithic gate remains
  incomplete rather than failed. **Complete-domain prefix fork/COW substrate is now
  accepted:** `ResidentSession::fork_shared_prefix` only seeds a fresh destination,
  validates the same service/source/capacity/full domain set, publishes through the
  destination's own `begin_validated_replacement`, and clones immutable domain
  ownership without replay/readback/copy. `ResidentTextSession` additionally requires
  the exact same resident model. Original0.8B paged page4 physically passes a
  3-token partial-page prefix fork then1-token COW continuation versus independent
  replay: logits/all42 exact, cached snapshot immutable, independent root/session,
  clean shutdown (`P12_SOURCE_PAGED_PREFIX_FORK_PASS`,179.13s). **The bounded
  semantic prefix lookup/eviction layer is now implemented in the model-neutral
  engine.** The original grouped numeric compatibility placeholders are
  superseded by `2df8cbb3`: `PrefixCompatibility<K>` compares an exact typed
  owner key and explicit sharing scope.
  `PrefixCache<U,T,K>` keeps immutable `Arc` payloads under explicit entry/weight
  bounds, does allocation-free newest-first lookup, returns the greatest exact
  semantic-unit prefix, refreshes exact duplicates and evicts FIFO deterministically.
  It never grants physical restore: the backend must still run the accepted full-
  domain validation/fork path. Focused state-cache tests, strict engine Clippy and
  production `qwen35-cuda` composition check pass. **The source-only common-frontier
  reconciliation primitive is now also in the same model-neutral engine layer.**
  `PrefixDomainCheckpoint` describes only a required domain's retained-through
  frontier and zero-based checkpoint alignment; `reconcile_prefix_frontier` computes
  the greatest request-bounded frontier legal for every domain using an allocation-free
  O(domains), overflow-safe bounded LCM. It returns0 when no nonempty aligned common
  checkpoint exists and grants no physical restore/copy authority. **The separate exact
  partial-tail COW ownership path is already complete.** Core `PagedBlockTable::share_prefix`
  shares the immutable physical page while narrowing only its logical valid extent;
  `fork_for_append` replaces a partial tail with a fresh target and a copy descriptor that
  retains both handles. Resident paged append exposes the old page as submitted Read and
  the fresh page as Write, and `ResidentSubmission` clones those exact leases for the
  armed flight until physical retirement. Candidate pages cannot publish before Ready;
  `publish_domain_swap` atomically installs the validated domains/root and displaced
  owners drop only after commit. Retained physical COW/materialization/page-boundary and
  source-real prefix-fork qualification already prove source immutability, fresh
  generations, exact valid-tail copy/append, zero tail and full retirement. No duplicate
  COW abstraction or new fixture is needed. The existing shared `ResidentPool` root path
  was re-audited at the same checkpoint:
  fresh logical lease IDs/generations are issued over retained backing, readbacks/Ready
  owners retain the checkout, and reuse occurs only after ledger-proven idle backing;
  unknown retirement poisons/fences/quarantines rather than returning a set idle.
  **Physical alias identity now has one shared core representation too.**
  `device_events::DeviceBufferIdentity` identifies an opaque process-unique ledger
  namespace + real allocation + exact byte range; direct buffers use their lease
  allocation while pooled/views use the retained backing allocation. The extra ledger
  namespace is required so independent services that both mint local ID1 cannot alias by
  value. The ledger is the sole constructor/query authority and its production
  alias-conflict path calls the same `overlaps` primitive, so there is no parallel
  hash/handle/range definition. Shared resident `overlaps_any` now checks lease/service
  ownership and resolves one reference against a lease slice under a single ledger lock,
  with no heap allocation or per-lease identity field. Both HIP/CUDA source-page alias
  guards use it, and the native B4 embedding preparation now uses the same primitive for
  output/table/status and feedback alias checks instead of reconstructing backend-handle
  ranges. The four-packet diagnostic pairwise check reuses `overlaps_any`. The production
  ordinary-CUDA body now also uses canonical identity for the complete prepared resident
  lease set via `ResidentService::pairwise_disjoint`: one ledger lock, no heap allocation
  inside the helper, and no backend handle inspection. This cleanly separates physical
  ownership truth from the still-required `BufferRange`/`OwnedRanges` kernel binding
  geometry. Strict CUDA Clippy/serving compile pass and the existing native-B8 public SDK
  Contiguous/Paged cell remains exact with clean shutdown after the migration; no new
  fixture was added. Six backing-ledger tests include cross-ledger separation. The next
  real transfer consumer is now migrated too: prepared streamed BF16 vocabulary-head
  binding validates the weight-staging/input/output lease triple with the same
  `pairwise_disjoint` ledger identity before its existing two-slot authenticated
  retirement ring can run. This closes the logical-ID-only alias hole without changing
  the ring, host-window retention, same-stream overwrite ordering or kernel-handle
  geometry. Production `cuda-cublas` and public `qwen35-cuda` checks pass. The retained
  public-SDK 0.8B context1024/prompt48 replay cell also passes physically after recreating
  the documented CUDA13.4 include view: exact two-token replay, distinct sessions, exact
  producer frontiers and clean shutdown in14.42s. The next production source consumer is
  now canonical too: HIP resident-text cold workspace leases use
  `pairwise_disjoint`, and request root validation uses
  `pairwise_disjoint_extension` to compare only new roots against the already-disjoint
  workspace and earlier roots under one ledger lock. This avoids a workspace² hot-path
  recheck while preserving exact handle ranges solely for execution-view geometry. HIP
  GPU/public composition checks compile, existing resident-text admission tests pass2/2,
  and the retained CUDA context1024/prompt48 public-SDK cell passes again in15.23s after
  the shared helper refactor. X2 HIP physical execution remains behind the recorded ROCm
  ABI environment gate. No new alias fixture was added. Remaining identity adoption is
  incremental for additional real graph/transfer consumers; do not cache identity into
  every lease merely for future diagnostics.
  **Ordinary CUDA source-page transfer now uses one retained authenticated owner.** The
  exact verified `TensorWindow`/window-cache result is wrapped only in an `Arc`, installed
  in the page's existing pending state before enqueue, and handed to the original CUDA
  actor's generic bounded owned-window H2D primitive. The unchanged page+consumer
  `sync_buffers` checkpoint therefore proves both source DMA and consumer completion before
  that owner can retire. This supersedes `b839920`'s source-side pinned staging: inspection
  found CubeCL pinned `Bytes` has no cheap `duplicate`, so the required clone became a Native
  deep copy and generic `Client::write` then staged to pinned again. The new path removes
  that payload `Vec`/`Bytes`/pinned chain, plus the now-unused transfer-staging reservation
  API, without adding a new pool or ring; HIP/non-CUDA serial storage is cfg-unchanged.
  Matched physical feedback on the same temporary 2B/32K Contiguous/Streamed prompt48 SDK
  cell: `b839920`39.36s; cache-promotion candidate41.00s rejected; moving the pinned owner
  into the generic writer/drop queue44.88s rejected; final retained-owner direct actor H2D
  **38.46s** (~2.3% faster than `b839920`). Exact replay/frontiers, independent sessions and
  clean shutdown remain intact. The older direct SSD→pinned/cache-bypass candidate remains
  rejected at41.77s. A 22 Nov follow-up tested the next simple overlap candidate without
  adding fixtures: reuse the existing two-slot retirement-ring state machine on CUDA with
  same-stream `OwnedCudaEvent` marks, the same single 1MiB device page, and exact retained
  `TensorWindow`/output owners. Strict CUDA/HIP checks and the existing retirement/accounting
  tests passed, but a same-tree matched opt1/codegen64/non-LTO explicit
  Contiguous/Streamed2B32K/prompt48 A/B measured **38.323s serial vs39.281s ring**; the
  Qwen one-page consumer/selector was ~2.5% slower and remains rejected. A separate24.267s
  Auto run selected paged state plus resident weights and therefore is not evidence for this
  streamed-path candidate; it does confirm the default smart policy already avoids this
  fallback when residency fits. The reusable mechanism is deliberately retained dormant:
  backend completion witnesses now implement the shared `RetirementMark` contract directly:
  CUDA `OwnedCudaEvent` and HIP `OwnedHipTransferRetirement`. The zero-cost aliases
  `CudaTransferRetirementRing<Owner>` / `HipTransferRetirementRing<Owner>` specialize the
  existing generic `TransferRetirementRing` without wrapper allocation. The ring owns only
  completion marks plus caller-supplied exact owners; model/session/publication policy stays
  with the future consumer. This preserves the primitive for Gemma, larger streamed models,
  other CUDA/HIP devices or different workload shapes without selecting it for Qwen. Temporary policy overrides/include view were removed
  and no new evidence fixture was retained. Do not retry the rejected Qwen one-page selector
  without new evidence; a future consumer may reuse the primitive after separately proving
  its owner graph/admission and a measured benefit. Meaningful Qwen streamed overlap beyond
  this still requires separately admitted extra device-page/copy-stream authority. Future
  direct/async IO must compose at the authenticated cache-miss boundary.
  **Scoped worker prefix reuse is
  now implemented and physically accepted.** `PrefixSharingScope` is a validated
  nonzero Copy ID. Existing `Request` stays unchanged; ordinary `request_text`
  remains isolated, while the explicit `request_text_scoped(request, scope)` path
  carries authorization alongside the queued request. Generic executor hooks restore
  only immediately after opening a fresh state, and the worker verifies restored
  cursor == actual executor frontier; cache insertion runs only after an already
  committed state-only prompt step. Qwen owns one model-local paged-only cache
  (`active` entry cap, `active*context` semantic-token weight); contiguous state pays
  no snapshot-retention cost. Because the cache instance is owned by one exact loaded
  Model and `ResidentTextSnapshot` additionally requires that exact resident model,
  loader/numerical/tokenizer/position/state-schema cannot cross instances; sharing
  scope is the only request-varying authorization class. The first physical 0.8B
  page4/max_append64 worker cell uses a65-token prompt: warm scopeA consumes65 model
  tokens, same-scope repeat restores the64-token complete-domain prefix and consumes
  only1, different scopeB consumes65 again; generated ID is identical and shutdown
  is clean (`P12_SOURCE_PAGED_PREFIX_CACHE_PASS`,172.98s). An initial run exposed a
  real lifecycle issue: cache-held snapshots outlived resident retirement; executor
  retirement now clears the cache before source shutdown. **Trusted HTTP scope
  injection is now accepted.** `ServerBuilder::prefix_sharing_scope` owns one optional
  validated scope for the whole server instance; connection captures carry only that
  Copy value. HTTP parsing exposes no scope field/header. `TextService` defaults to a
  fail-closed `prefix_sharing_unsupported` error for scoped admission, while the
  engine Runtime/Router explicitly implement it; the focused server suite is7/7 and
  strict serving Clippy passes. Original0.8B page4/max_append64 physical HTTP proof
  uses ordinary OpenAI JSON only: server scope7 warm consumes65 model tokens, repeat
  consumes1, a new server over the same runtime/cache with scope8 consumes65, all
  output IDs match and shutdown is clean (`P12_SOURCE_PAGED_PREFIX_HTTP_PASS`,
  171.97s). No client can select/override the scope through this HTTP API. **Bounded
  live-snapshot eviction and explicit stale-scope cache invalidation are now accepted.** The
  model-neutral cache exposes `PrefixCacheEviction { entries, weight }` and
  `invalidate_scope`; trusted control routes that through Executor -> typed Registry
  -> Router -> engine/public Runtime as `invalidate_prefix_scope`. Cache invalidation
  is explicit: stopping a server or selecting another scope does not purge prior
  state. The embedding owner must first stop issuing/drain a stale scope, then call
  `invalidate_prefix_scope`; this purges currently retained entries but does not
  retroactively cancel already-admitted work or invalidate a snapshot clone already
  owned by such work.
  Qwen now admits `2*active` cache entries while retaining only `active*context`
  semantic-token weight, so count and weight bounds are independently meaningful.
  Original0.8B page4/max_append64 physical `active=1` proof shows: three distinct
  prefix64 snapshots trigger FIFO count eviction (newer hit, oldest miss); explicit
  purge releases exactly2 entries/128 weight and the invalidated scope misses afterward;
  a prompt321 trajectory hits the512-weight bound, retains prefix320 for a1-token hit,
  but evicts prefix256 so prompt257 replays all257 tokens; final purge releases2
  entries/448 weight, repeated purge returns zero, and shutdown is clean
  (`P12_SOURCE_PAGED_PREFIX_EVICTION_PASS`,179.12s). This is cache-retention pressure.
  **Externally imposed resident page-pool pressure is now physically accepted too.**
  The qualification-only holder checks out real pages from the same finite
  `ResidentPagePool`; production only exposes read-only diagnostics. With original
  0.8B page4/max_append64, a cached prefix64 leaves192 pages occupied; the qualifier
  holds the remaining5952 pages and fills the6144-page pool. The next cached tail is
  refused prelaunch as `Capacity`, the root stays unchanged, vector/staging candidate
  leases unwind to their prior occupancy, and the cached snapshot remains valid. After
  releasing pressure, the same tail commits and its complete vocabulary logits match
  the pre-pressure baseline bit-for-bit; all allocated pages are idle before clean
  shutdown (`P12_SOURCE_PAGED_PREFIX_PAGE_PRESSURE_PASS`,168.51s). One earlier
  external SIGTERM at ~8s during startup remains incomplete only. Minimal production
  diagnostics now report current `prefix_cache` occupancy/bounds and `paged_pool`
  allocated/idle/occupied/max/page-bytes under each Qwen source model; there are no
  cumulative hot-path counters. This qualifies actual resident page-pool exhaustion
  and recovery, not host/device OOM.
  **Fresh 19 Nov review + overlap prerequisite:** the entire recent prefix/paging/
  serving lineage was re-read from the live tree and reviewed across engine/router/
  serving/GPU ownership, scope isolation, alias deduplication, cache invalidation,
  pressure unwind, cancellation and feature boundaries. No prefix correctness or
  authorization regression was found. The review did find the hard overlap boundary:
  `PhysicalQueue` still owns exactly one `active: Option<u64>` region per card, and
  `ResidentSubmission` only retires a physical region after its trusted readback/
  validation path completes. The current Qwen CUDA source also uses one pinned
  `OwnedCudaQueue` logical stream. Therefore worker-level or scheduler-level
  prefill/decode concurrency would still serialize physically (or would weaken
  completion safety if detached naively); do not claim overlap from more runnable
  rows/threads.
  As the first independent prerequisite, the patched CubeCL CUDA layer now supports
  `OwnedCudaQueue::fork_stream` for a same-service client clone and a non-clone
  `OwnedCudaEvent` that records/queries/waits/destroys one event on that exact actor/
  logical stream. No second CUDA service/context is created and cross-stream buffer
  dependencies remain the caller's responsibility. Physical qualification on L5
  enqueued disjoint8MiB writes on primary and fresh secondary streams, recorded both
  events before waiting either, observed the secondary event still not Ready at the
  first query, then waited both and read back both buffers exactly:
  `P12_CUDA_SECONDARY_STREAM_EVENT_PASS same_service=true streams_distinct=true
  disjoint_buffers=true events_recorded_before_wait=true primary_ready_before_wait=true
  secondary_ready_before_wait=false readback=exact`. This proves a live independent
  secondary stream/event substrate, **not** prefill/decode performance overlap and
  not multiple resident regions in flight. **The event/lease-backed resident in-flight
  owner is now accepted too.** `ResidentCudaInFlight<'a>` owns the original armed
  `ResidentSubmission<'a>` plus one same-service `OwnedCudaEvent`; it does not own the
  physical queue slot and creates no Ready/publication authority. Conversion is an
  explicit unsafe boundary: `into_cuda_in_flight(&OwnedCudaQueue)` records the event
  itself on the supplied queue, and the caller must prove that exact logical stream
  already received every command covered by the submission and that no later command
  will be appended. Known completion
  waits/queries the CUDA event, then still runs the existing trusted readback +
  validator path before write initialization/Ready. Cancellation marks the existing
  logical device event cancelled, waits the CUDA event, retires event pins + physical
  guard without initializing writes and without constructing Ready. Dropping unresolved
  ownership first drops the armed submission, preserving completion-unknown fencing,
  exact lease quarantine and completion-owner retention; the CUDA event token is
  destroyed only afterward. Physical L5 lifecycle proof is terminal-green:
  `P12_CUDA_RESIDENT_INFLIGHT_EVENT_PASS known_readback=exact known_initialized=true
  cancel_retired=true cancel_initialized=false unresolved_fenced=true
  unresolved_initialized=false completion_owner_retained=true quarantine_owner_delta=1
  queue_width_unchanged=true`. **The reusable bounded multi-flight substrate is now
  accepted.** `PhysicalQueue` uses a bounded active-ID set with configurable active
  capacity, while the production registry intentionally remains at active capacity1
  until model adoption. `PhysicalServiceBudget` now multiplexes its one prepaid
  envelope across overlapping known flights with a tiny shared counter: first flight
  marks the retained lease Submitted, intermediate flights add no charge/phase
  transition, and the last known completion marks it Ready; any unknown flight still
  fences the same envelope/card. Existing `DeviceEventLedger::submit` remains the
  access-conflict authority—no duplicate queue-level alias checker was added.
  Host contracts pass (queue6/6, physical-service5/5). Physical L5 proof acquires two
  active slots from a bounded2 queue, submits two disjoint resident outputs on primary
  and secondary streams under the same prepaid envelope, records both completion
  events, retires the second flight before the first and reads both8MiB outputs exactly:
  `P12_CUDA_RESIDENT_DUAL_FLIGHT_PASS active_capacity=2 same_prepaid_envelope=true
  streams_distinct=true disjoint_outputs=true events_before_completion=true
  reverse_retirement=true readback=exact outputs_initialized=true service_reusable=true`.
  Next move directly into the real Qwen/source execution path and measure actual
  prefill/decode multi-flight behavior; do not add another synthetic evidence matrix
  unless that integration finds a concrete missing invariant.
  **20 Nov realistic serving feedback:** rather than adding another harness, the existing
  public B8 SDK qualification now uses eight concurrent original0.8B requests at48
  prompt +3 output tokens, warms once on the loaded model, and times the second wave
  using existing token publication clocks. Both contiguous and paged form native B8
  and return exact-identical tokens. Steady measured wave: contiguous4888.231ms
  (4.910 output tok/s), mean token commits [4523.086,4705.276,4886.792]ms; paged
  5080.588ms (4.724 output tok/s), commits [4567.294,4818.167,5075.042]ms. Thus
  fresh prefill/TTFT (~4.5s) dominates; steady B8 decode intervals are ~182ms
  contiguous (~44 output tok/s) and ~251/257ms paged (~31-32 output tok/s). Warmup
  does not remove the regression. Retained historical eager-vLLM B8 is72.20 output
  tok/s on47-50 input/3 output, but the exact vLLM image is not currently runnable,
  so this is directional history rather than a fresh matched comparison.
  **Correction after interrupted-phase revalidation:** the previous residency diagnosis
  was wrong. Exact admission arithmetic for original0.8B/context512 shows the fully
  resident two-root plan at1,652,990,965B and the fully resident18-root `active=8` plan at
  2,076,877,813B. B8 cohort reserve is51,210,272B, greedy reserve11,904B, leaving a
  fit cap4,780,616,032B under the4,831,838,208B provider envelope; the fitted layout
  remains `resident_weights=true`. Streaming/root pressure therefore does **not** explain
  the ~4.5s TTFT. The real hot structure is `execute_prefill_cohort`: projections and
  stateless RMS are physically shared, but remaining stateful operators execute in a
  lane loop. Decode already has dedicated cached GDN cohort lowering and single-token
  B4/B8 attention; prefill has no general multi-token attention/GDN cohort primitive.
  The stateless QK-RMS/gate/multiply/SiLU cohort lowering was restored and retained:
  it is localized/composable, preserves lane-specific rotary/root authority, and the
  latest exact warm48+3 B8 run improved the retained no-extension baseline from
  4888.231→4696.591ms contiguous (~3.9%) and5080.588→4907.715ms paged (~3.4%).
  This does not solve the dominant stateful prefill bottleneck, but the gain is useful
  and compounds with later work. The old fast `native_cohort` path is also not a reusable general answer:
  its cold envelope is explicitly B4 and initial phase is fixed around47x4=188 rows;
  the serving loader remains qualification-gated. Next implementation target is shared,
  model-neutral **multi-token stateful prefill batching**. Multi-token attention is now
  retained: `Bf16CausalAttentionBatch4::new_ragged_rows` generalizes the existing B4
  kernel to equal multi-row private sequences, and ordinary B8 prefill composes two B4
  groups while incompatible/ragged cases keep the old per-lane fallback. The real
  warm48+3 B8 run stayed exact but TTFT was effectively flat versus the retained
  stateless baseline (contiguous4696.591→4701.433ms; paged4907.715→5022.809ms), so
  attention is not the dominant remaining cost.
  **Multi-token GDN prefill is now accepted.** The shared GDN runtime gained explicit
  `TextPrefillCohort { rows, lanes }` geometry, sequence-aware controls/cache/state
  workspace, strided packed recurrent gather/scatter, and a width-scoped cohort GDN
  workspace owned by `OrdinaryCudaCohortValues`. Ordinary B4/B8 prefill imports each
  private Conv/Recurrent root through existing typed `StateTransport`, executes one
  shared cohort GDN runtime over packed activations, then exports each candidate root;
  no scheduler/root/publication authority moved into the kernel layer. Runtime word
  transport keeps the old1,000,000-word global contract; only its dedicated runtime
  transport path has a2^21-word bound, exactly covering B8 recurrent state.
  Real warm48+3 B8 exact serving moved dramatically versus the attention checkpoint:
  contiguous4701.433→1007.301ms (~78.6% faster), paged5022.809→1160.106ms (~76.9%);
  mean TTFT4341.373→616.681ms contiguous (~85.8% lower) and4503.152→649.944ms paged
  (~85.6% lower). Those numbers were produced by Cargo's unoptimized test profile and
  are correctness/relative-debug feedback only; they must not be used as serving-perf
  baselines. Rebuilding the **same commit and same existing 8x48+3 SDK cell** with the
  historical non-LTO/opt1/codegen64 host profile (`CARGO_PROFILE_TEST_OPT_LEVEL=1`,
  `CARGO_PROFILE_TEST_CODEGEN_UNITS=64`, `CARGO_PROFILE_TEST_LTO=false`, debug info0)
  changes the result without any source change: contiguous183.932ms/130.483 output
  tok/s with commits [133.297,158.866,183.617]ms; paged242.488ms/98.974 tok/s with
  commits [153.943,198.058,241.371]ms. Exact tokens/native B8 remain unchanged.
  Contiguous is therefore already well above the retained historical eager-vLLM B8
  72.20 tok/s and old ZG76.77 tok/s on the same short47-50+3 workload class, although
  that vLLM number is still historical rather than a fresh matched rerun. Approximate
  decode-only cadence is ~25ms/B8 cohort contiguous (~318 output tok/s) and ~44ms paged
  (~183 output tok/s). Do **not** optimize against debug-profile timing; future serving
  performance feedback must use a matched optimized host profile.
  **Continuous refill/regroup is now physically exercised and one banding bug is fixed.**
  The production worker already refills freed active slots from the admitted FIFO before
  each wave and recomputes compatible native bands. A temporary public-SDK probe used
  four1-token rows + four8-token survivors + four queued8-token rows on an `active=8` / `cohort_width=8`
  model. It first exposed a real width-reduction defect: the width8 cohort arena retained
  only the B8 GDN compiled-ABI layout, so a legal B4 band bound `PackedBa` against a
  32768-byte B8 ABI while the B4 activation was correctly16384B. The fix retains both
  B4 and B8 `GdnRuntimeWorkspaceLayout` descriptors for a width8 arena while allocating
  physical GDN scratch once at the maximum width; active plans select the exact band
  descriptor with zero per-token allocation. B4 on a width8 model then passed with every
  token reporting native width4. The warmed refill probe passed contiguous and paged with
  survivor/queued width sets `{4,8}`, proving actual B8→B4→refill/prefill→B8 regrouping.
  Matched opt1/codegen64/non-LTO goodput for68 generated tokens was380.379ms /178.769
  output tok/s contiguous and498.212ms /136.488 paged. The temporary probe was removed.
  **The first optimized paged-gap slice is now implemented and physically measured.**
  Paged KV gather and fresh COW/page initialization previously allocated host-zero vectors
  and uploaded complete staging/page extents on every use. They now reuse the existing
  pinned runtime word-zero CUDA transport, chunked at its admitted2^21-word bound; zero
  work is counted as real dispatches, including Hybrid scatter accounting. No root/COW/
  Ready/publication rule or default state policy changed. On the same matched
  opt1/codegen64/non-LTO8x48+3 public SDK cell, exact native B8 passes at184.409ms /
 130.145 output tok/s contiguous and205.724ms /116.661 paged; retained pre-change results
  were183.932/130.483 and242.488/98.974. Mean first-token commits in the new run are
 133.988ms contiguous and133.531ms paged. Strict CUDA `zerogravity-gpu` Clippy passes.
  A focused low-level `physical_paged_kv_` attempt was externally SIGKILLed while
  compiling before any test body; a later exact-source SDK confirmation was externally
  SIGKILLed after compilation and test entry with no Rust/CUDA diagnostic. Both left no
  process/GPU residue and remain incomplete infrastructure evidence. The terminal-green
  measurement above predates only the dispatch-count accounting correction, not the
  compute/zero path. **The next paged transport slice is now physically accepted.**
  A source-neutral `RuntimePagedPairGather` uses the existing32-pointer pinned-kernel
  ceiling as15 key/value page pairs plus two destination planes. Page count, active words
  and destination offset are runtime scalars; longer prefixes chunk through the same ABI,
  so no page-count-specific kernel family or pointer-table allocation is introduced.
  Existing page/root/COW/access/completion authority is unchanged. With page_tokens4 the
 48-token workload has12 pages per plane and replaces24 page-copy launches per KV domain
  with one paired gather launch. The same matched8x48+3 SDK cell is terminal-green,
  exact/native B8: contiguous176.800ms/135.746 output tok/s and paged182.505ms/131.504,
  with mean commit clocks[129.424,153.479,176.519]ms versus
  [129.212,155.426,181.469]ms. That slice left ~3.2% matched wall-time overhead.
  **The final staging-zero launch gap is now removed on the bounded common path.** The
  paired gather copies its active prefix and zero-fills the remaining destination span in
  the same final launch whenever the staging plane fits the existing runtime-word bound.
  Oversized planes keep the previous bounded device-zero fallback, and empty prefixes
  explicitly keep it as well, preserving inactive-tail initialization without a larger
  ABI or new allocation. The same matched8x48+3 SDK cell is terminal-green at
  contiguous177.467ms/135.236 output tok/s and paged180.520ms/132.949, with commit clocks
  [129.797,153.805,177.178]ms versus[128.897,154.402,179.491]ms. Remaining matched
  wall-time overhead is ~1.7%, versus ~11.6% after device-zeroing and ~31.8% at the
  retained pre-paged-optimization baseline. The first physical attempt was externally
  SIGKILLed during compilation before test entry; process/GPU/kernel checks were clean and
  one cached retry passed. The connector still has no vLLM executable/module and Podman
  still reports the pinned image `image not known`; do not claim a fresh vLLM comparison.
  **Current continuous-refill goodput is now remeasured on unchanged `e72030c`
  source.** The same public-SDK4 one-token +4 eight-token survivor +4 queued eight-token
  shape (`active=8` / `cohort_width=8`,68 generated tokens) passes under the matched
  opt1/codegen64/non-LTO profile at400.383ms/169.837 output tok/s contiguous and
  422.572ms/160.919 paged, about5.5% paged wall-time overhead. This materially improves
  the older paged498.212ms/136.488 result while avoiding a new retained fixture; the
  temporary probe was removed. I also tested the obvious next shared scatter candidate:
  fuse device-to-device append copy with inactive-tail zeroing so fresh pages do not need
  a separate zero launch. The first post-change run was externally SIGKILLed after test
  entry with no residue/OOM/Xid; one cached retry completed at395.849ms/171.783
  contiguous and421.159ms/161.459 paged. That is only ~0.3% paged throughput gain versus
  accepted source, so the extra kernel path was rejected from production selection. The
  implementation is now retained as the model-neutral dormant CUDA candidate
  `cuda::performance_candidates::launch_copy_zero_tail`; source behavior stays `e72030c`. The connector still
  has no vLLM executable/module and Podman still reports the pinned image `image not
  known`; do not claim a fresh vLLM comparison. Owner now explicitly defers vLLM.
  Do not add refill fixtures, direct paged-attention complexity or debug-profile
  micro-tuning as the next step; advance the populated CUDA context ladder instead.
  **Post-check L5 baseline watch:**
  after the accepted B8 process had exited
  and no ZeroGravity/native-gpu process remained, the isolated connector view of
  `nvidia-smi` later reported ~1,785MiB/P0 instead of the initial 41MiB, while
  NVML/pmon exposed no process owner and the kernel log showed no Xid/OOM. A
  subsequent passive resample returned to 41MiB/0% without reset or intervention.
  Treat the transient as unexplained rather than a ZeroGravity leak; preserve any
  unrelated host owner and continue resampling the physical baseline before CUDA
  work.
- Earlier key implementation checkpoints include `f1ce532`
  (exact graph replay storage/access authority), `b0cf0b2`
  (`fix(cache): avoid duplicate Qwen snapshot hashing`), `134a571`
  (`fix(hip): isolate CUDA private KV qualification`), `874e67d`
  (`feat(cuda): retain shared-input projection pair`), `c282f20`
  (`fix(hip): restore source profile compilation`), `5bf440d` / `db7be60`
  (production CUDA graph substrate/resource admission), `6a5c610` (driver-safe
  nonpersistent NVRTC) and `81b7c71` (explicit graph-resource lifetime witness).
  Their docs checkpoints through `b12e8f5` are pushed.
- Pushed/tracked `origin/main` is `f0f06402c78d4511dea1559c73d4295f88790f93`
  (`feat(cuda): batch multi-token prefill attention`). The post-origin local lineage
  includes `1222f8a` (multi-token GDN prefill), `60ba956` (matched performance
  profile), `3afb990` (reduced-band GDN layouts/refill), docs-only `7beb1be`,
  `0dac5ab` (device-side paged zeroing), `0680b0b` (batched paired-page gather), and the
  current fused-tail-zero slice described above. Revalidate actual HEAD for the newest
  signed source checkpoint.
  Nothing after `f0f0640` is pushed unless explicitly requested. Never use push merely
  for X2 transport.
- Before any new physical X2 run, sync it to the exact current l5 `main`
  bytes/commit without using a push merely as transport, then revalidate
  status/process/GPU state.
- Local docs `5c8f143` recorded the host-global X2 KFD observation and `3cadc3b`
  records rejection of the ordinary-Qwen `874e67d` consumer after completed
  loaded-model A/B; the primitive itself remains retained. Later host context
  corrected `5c8f143`'s ownership inference: X2 is shared with Ollama/model
  workloads, so that KFD owner is not attributable to ZeroGravity without host
  process evidence. `5bf440d` is the first broader post-micro-optimization CUDA
  implementation checkpoint.
- Important earlier accepted commits:
  - `60b048d` — canonical-boundary private B8 GDN recurrent core accepted.
  - `775dfa6` — ragged private B8 KV transition accepted.
  - `c12acab` — grouped B8 recurrent transport primitive retained, serving
    consumer rejected.
- At this refresh no Cargo/model process is active in the connector-visible
  command namespaces and l5 RTX3060 is idle at ~37 MiB. X2 is a **shared GPU host**
  whose baseline must be resampled every run: the latest observation is
  total103,079,215,104 bytes VRAM, used272,351,232/busy0 with no KFD owners, while
  earlier observations included unrelated Ollama/model residents around31GB. Do not
  infer ownership from a prior baseline or require an empty GPU. Record the current
  baseline, stay within available capacity, verify only our incremental VRAM/queue/
  process delta retires, and never reset/reclaim/restart/kill other owners without
  explicit authorization.
- Root `README.md` is user-authored. Leave it unchanged unless explicitly asked.
- Keep one persistent `main` checkout per host. Reuse canonical target/cache state;
  do not recreate old worktree pools or `.agents/runs`.
- Commits are signed. Inspect `git diff --cached --stat` before every commit and
  stage only owned paths.
- On l5, `git log --show-signature` reports a good ED25519 signature while also
  printing a missing local allowed-signers principal warning. That warning is not
  a bad signature.

### Read order / authority

A new agent should read in this order:

1. Current source/tests for the surface being changed.
2. `AGENTS.md`.
3. `.agents/plans/README.md`.
4. `.agents/docs/CURRENT_HANDOFF.md`.
5. `.agents/docs/FULL_COMPREHENSIVE_HANDOFF_2026-11-21.md`.
6. `.agents/plans/TODO.md`.
7. `.agents/plans/model-execution/TODO.md`.
8. `.agents/plans/numerics-and-context/CORRECTNESS_POLICY.md`.
9. The owning plan(s), especially:
   - `serving-and-speculation/TODO.md`
   - `cpu-gpu-optimization/TODO.md`
   - `memory-and-streaming/TODO.md`
   - `pipeline-composition/TODO.md`
   - `numerics-and-context/TODO.md`
   - `cli-device-routing/TODO.md`
   - `model-acquisition/TODO.md`
10. Relevant architecture docs under `.agents/docs/`.
11. Git history for removed experiments and superseded handoffs.

Current source/tests and live TODOs outrank this handoff if main advances.

### Tracking / handoff method

The old generated handoff/SHA/archive workflow is retired.

- `CURRENT_HANDOFF.md` is the canonical cross-window pointer.
- `.agents/plans/TODO.md` is the global queue.
- Per-plan TODOs are the detailed accepted/rejected/open-work ledgers.
- Git history is the archive and provenance record.
- Do not recreate `.agents/runs`, source snapshots, monolithic receipts, per-file
  hash catalogues or durable side-worktree coordination state.
- Actual model/content identities, numerical contracts and physical acceptance
  limits still belong in source/tests/plans where semantically required.
- Historical failures stay failures. A later pass/cleanup cannot relabel them.

## Owner direction / working style

The user has repeatedly steered the project toward implementation-first,
shared-primitives-first work:

- Prefer **implementation → quick real-model feedback → controlled comparison**.
- Solve cleanly at shared layers when the semantics genuinely benefit multiple
  models/backends/tools.
- Model/source policy belongs in model/source layers. Reusable numerical, layout,
  transport, state and physical execution primitives belong in shared runtime/GPU
  layers.
- Keep independently useful model-neutral primitives even when their first Qwen
  serving consumer is rejected; remove the slow consumer, not the clean primitive.
- Exact source/reduction/state/root/resource/cancellation semantics are never
  weakened to chase benchmark numbers.
- Unsupported capability fails closed. Never silently select another arithmetic
  profile, captured case or model-specific fallback.
- Rust preference: explicit ownership, simple robust APIs, zero-cost abstractions
  where natural, targeted hot-path optimization only when repeatably material.
- Do not over-engineer tests, evidence, receipts, pinning or phase granularity.
- Reuse accepted validation for unchanged primitives.
- Validate in coherent checkpoints rather than after every helper.
- Treat fewer launches/loads/instructions as a hypothesis, not acceptance evidence.
- For small expected gains use one-binary/one-model forward/reverse or ABBA
  measurement; cross-process clock/thermal drift repeatedly produced false wins.
- Remove all temporary selectors, profiling hooks, `_tmp` runners and include
  views after the decision is recorded.
- Recent CUDA work has used ~30-minute coherent checkpoints; continue that unless
  the user changes the cadence.
- Do not ask the user to reconfirm already-authorized continuation steps.
- Never push unless explicitly asked.

## Canonical architecture / non-negotiable invariants

ZeroGravity is one Rust-owned execution fabric, not a per-model collection of
engines/servers.

- One semantic/model definition per family/source contract.
- One logical `StateRoot` publication authority.
- Canonical roots remain checkpoint/rollback/publication authority even when
  private/native/derived representations exist.
- One process-shared physical `ResourceArbiter`.
- Device workers own real asynchronous compute/events, but cannot invent logical
  commits.
- Release follows actual completion/fence/event state; completion-unknown work is
  quarantined.
- Source/weight owners survive until the real final consumer retires.
- Cancellation cannot publish partial candidate state.
- Independent sessions/roots remain independent through physical batching.
- Physical batch width is separate from each text invocation's authenticated numerical
  reduction contract.
- Finite captured cases are qualification evidence, never production selectors.
- Metadata/registry slugs never grant execution support.
- CPU/CUDA/HIP/Vulkan and quant/source profiles remain separate qualification
  claims.
- Performance follows semantic/source/state/ownership correctness.

The Fabric transition is complete and now forms the shared architecture:
`zerogravity-core` owns semantic/state/provider/physical contracts;
`zerogravity-artifacts` owns immutable formats/source/acquisition/store; model
crates own family semantics; CPU/GPU crates own physical providers;
`zerogravity-engine` owns model-neutral runtime/session/scheduling/residency;
`zerogravity-serving` owns reusable protocol/server behavior; `zerogravity` is the
public composition root; CLI stays presentation/arguments/signals. Do not restart
the old Fabric S/F/Q transition sequence.

## Design/spec navigation

Do not read all design docs for every kernel change. Use the relevant map:

- Product/vision: `VISION_SCOPE_AND_PRINCIPLES.md`,
  `MODULAR_AI_SYSTEMS_REFERENCE_ARCHITECTURE.md`,
  `PRODUCT_MODEL_PIPELINE_AND_EXTENSIONS.md`.
- Engine/IR: `ENGINE_CORE_EXECUTION_AND_PROTOCOL.md`,
  `MINIMAL_EXECUTION_SPINE_AND_PROTOCOL_V4.md`,
  `TASK_PROGRAM_AND_COMPILER_REFINEMENT_V4.md`,
  `SEMANTIC_IR_AND_COMPILER.md`.
- Model/formats/numerics: `MODEL_SEMANTICS_AND_COMPATIBILITY.md`,
  `FORMATS_QUANTIZATION_AND_LOADING.md`,
  `NUMERICS_QUALITY_AND_BENCHMARKS.md` and the correctness policy.
- Scheduling/state/memory: `SCHEDULER_BATCHING_AND_SPECULATION.md`,
  `STATE_CACHE_AND_TRANSACTIONS.md`, `MEMORY_RESIDENCY_AND_IO.md` and
  `UNIFIED_VIRTUAL_TENSOR_RESIDENCY_AND_SSD_STREAMING_2026-10-15.md`.
- GPU/device: `DEVICE_KERNEL_AND_CODEGEN.md`,
  `ACCELERATOR_EXECUTION_FAMILIES_AND_BACKEND_CONTRACT.md`,
  `OPERATOR_AND_FUSION_CATALOG.md`, `COST_MODEL_AND_AUTOTUNING.md`.
- Serving/routing: `RUNTIME_SERVING_AND_UX.md`,
  `ZERO_CONFIG_CLI_CONFIG_AND_OPENAI_API_CONTRACT.md`,
  `MODEL_ROUTING_RESIDENCY_AND_MULTI_MODEL_SERVING_CONTRACT.md`.
- Composition: `COMPOSITION_AND_EXTENSION_CONTRACTS.md` and
  `pipeline-composition/STRUCTURE_AND_NAMING.md`.
- Distributed/multimodal: the distributed-placement/transport docs and the
  universal/realtime multimodal contracts.
- Historical Fabric rationale lives under `.agents/plans/fabric-vnext/` and
  `fabric-vnext-transition/`; those plans are closed.

## Global plan ownership / pending-work map

The global queue in `.agents/plans/TODO.md` is authoritative. At handoff creation
the work ledgers are:

- **serving-and-speculation** — active P1: serving correctness/context fit,
  population/batching/paged-state/speculation, matched vLLM campaigns and shutdown/
  cancellation behavior. Its TODO retains the full serving chronology from the
  initial CPU/OpenAI router through resident GPU cohorts.
- **numerics-and-context** — active P1: independent numerical/source acceptance,
  genuine context progression, reference capture, CUDA-forward/HIP parity and
  remaining long-context/backend cells. The common acceptance rules are in
  `CORRECTNESS_POLICY.md`.
- **memory-and-streaming** — active P1: exact admission/fit, source windows,
  complete-domain state/prefix/COW/persistence, SSD streaming and residency
  boundaries.
- **cpu-gpu-optimization** — active P1: backend parity and measured performance,
  with correctness-first gates before matched optimization claims.
- **pipeline-composition** — active P1: recursive programs/profiles, component
  reuse, format/model/provider boundaries and production/test separation.
- **cli-device-routing** — active P1: unified run/config/source/device selection,
  exact device identity and future local spill/peer routing.
- **model-execution** — active P2: the detailed CUDA/HIP/Qwen implementation,
  qualification, performance and rejection ledger summarized above.
- **model-acquisition** — active P2: authenticated source location/cache/provider
  flows, HF/Xet/cloud/object acquisition, aliases and source rights.
- **multimodal** — icebox: multimodal execution/product work; do not infer current
  implementation from architecture docs.
- **distributed-execution** — icebox: remote/distributed placement/transport work.
- **platform-and-release** — icebox: assembled platform/release acceptance work.
- **fabric-vnext-transition** — closed: historical architecture/migration rationale
  and acceptance; do not reopen as the current queue.

When work crosses plans, update the lowest owning TODO(s) rather than creating a
parallel status ledger. The handoff is a navigation layer, not a replacement for
those work-item files.

## Current signed implementation state

### Fabric / shared runtime

Accepted milestones include `860ce38`, `fef7123`, `8d761c4` / `8c43844` and
`1c99929`. The transition is complete; expansion belongs to the active model,
serving, memory and composition plans.

### HIP / ordinary Qwen

- `8ffb511`: populated 0.8B/2B HIP serving accepted through qualified scopes.
- `d16f096`: explicit total HIP device-budget admission and resident 27B
  comparison. Resident 27B C1..C64 output/state/retirement is verified under the
  declared large budget.
- `c282f20`: fixes a real HIP-only cross-feature compile regression. The shared
  packed-BF16 numeric-zero batch-greedy kernel is model-neutral and no longer
  incorrectly hidden behind `cuda-cublas`; HIP-facing test/support profiles also
  state the existing singleton policy explicitly with `native_rows: 0`.
- X2 compile/link validation after `c282f20` passed `zerogravity-gpu` with
  `hip-source`, `qwen_hip_sdk` and `fabric_sdk` with `qwen35-hip,serve`, plus a
  temporary release 27B SDK/HTTP diagnostic link.
- The fresh 27B diagnostic still does **not** establish a 27B physical result:
  its isolated process was externally terminated during model load with no Rust
  panic/error. VRAM returned to baseline, no KFD client remained, and host RAM/swap
  did not establish an obvious OOM.
- Tiny physical HIP health is now re-established independently. X2 was synced to
  exact l5 `2caf304`; a temporary self-validating HIP C++ kernel built with the
  installed CLR 7.2.4 toolchain and its referenced ROCm device-library store ran
  only through `python3 -B scripts/native-gpu.py -- ...` and exited0 on AMD Radeon
  8060S / `gfx1151`: one device, 256 exact `i*17+5` words, checksum `556160`.
  Post-run VRAM returned to `272351232/103079215104` bytes, busy0, no KFD client
  remained and the dotlink cgroup still reported `oom=0`/`oom_kill=0`.
- The earlier tiny-smoke termination is therefore localized away from HIP runtime
  health. Focused Rust test builds could still disappear without an exit code both
  inside and outside `native-gpu.py`; no cgroup/kernel/systemd-oomd OOM evidence
  was found. The temporary diagnostic source-gate experiment was discarded.
- `134a571` fixes another cross-feature HIP qualification regression exposed by
  the tracked 27B context diagnostic: generic batch-KV qualification no longer
  unconditionally invokes the CUDA-only private-BF16 KV qualifier. Exact l5 HIP
  qualification check and X2 release context-diagnostic build pass.
- Exact X2 27B host-only `cache-preflight` passes on canonical snapshot
  `98cea13c169310196c4d8a113765f8fe58ab7a7961f0d1be69212c70dba8015e`;
  source remains unchanged and 512/1K/2K capacity cells complete with no GPU.
- Runtime `open_source` is now split more precisely. Exact canonical27B
  tokenizer construction is independently green:12,809,320-byte `tokenizer.json`,
  SHA256 `0997f410c57a1f4e53b09e4be8f4a172d90edd9564368fb0847030937229b9f3`,
  existing release tokenizer test pass0.71s.
- A smaller host-only exact-source process also authenticates all18 27B shards and
  completes the full Qwen binding/seal/1K plan:55,563,006,776 source bytes,1,199
  tensors,851 text/348 other roles,64 layers/states,64 plan boundaries/112 domains.
  The deterministic Qwen binding/plan path itself is therefore green.
- `b0cf0b2` fixes a concrete redundant verification boundary discovered here.
  Runtime previously fully hashed all31 snapshot payloads (55,586,113,293 bytes)
  in `open_snapshot`, then `VerifiedWindowPlan::open` authenticated the exact
  shards again. Qwen now binds a manifest-authenticated `SnapshotDescriptorLease`
  that claims no payload verification; every non-shard payload keeps its existing
  compiled SHA check and every shard keeps the authoritative whole-file
  `ExtentSource::scan_verified`. Normal `SnapshotLease` remains fully verified.
- The full canonical runtime host path is now green. A warmed optimized test-profile
  build produced the full runtime unit-test binary; running that exact binary under
  an external2s heartbeat wrapper completed the canonical source/tokenizer/final-
  reader test in75.20s with about16..37MiB RSS and `gpu=false`. The heartbeat only
  kept the connector execution envelope alive; it is not part of ZeroGravity.
  Therefore the earlier silent first-shard termination was not evidence of a Qwen
  source/runtime correctness bug.
- After that closure, one reference-free exact75GiB public-HTTP liveness probe was
  launched under `native-gpu.py`, private loopback and the unchanged180s server
  deadline. It reached real HIP allocation (~31.376GB VRAM, busy6%) but the entire
  isolated process disappeared after ~10..12s despite an external5s heartbeat,
  before any HTTP response. No numerical/reference acceptance follows.
- The earlier “no visible owner” note was incomplete because each dotlink shell
  has a Bubblewrap-private PID namespace. Host-global KFD sysfs exposes process
  entry `/sys/class/kfd/kfd/proc/2406838`, owning31,098,232,832 bytes VRAM and
  six queues with `pasid=0`, zero CU occupancy/faults. **The later host-context
  correction matters:** X2 also runs Ollama/model workloads outside the sandbox,
  so this KFD owner cannot be attributed to the failed ZeroGravity run. The old
  “stranded allocation / clean272MiB baseline” interpretation is retired; the
  observed KFD/VRAM state is a shared-host baseline unless ownership is positively
  established.
- Dotlink's exposed GPU mount sources are pinned with `O_PATH|O_NOFOLLOW|O_CLOEXEC`
  and used only as Bubblewrap `--dev-bind` sources; their fdinfo has no DRM memory
  accounting. A matched non-GPU durable-Job → Bubblewrap → `nix develop` →
  `native-gpu.py` → nested-unshare control ran35s and exited0. Therefore the
  physical run's10..12s loss is not a generic connector/native-gpu lifetime limit.
- The75GiB route selects resident storage for the canonical ~53.8GB source-weight
  extent and preparation uploads projection operands sequentially, but because the
  host already has shared GPU residents the observed ~31.376GB total usage does
  **not** localize how many bytes belonged to the failed ZeroGravity process. No
  ZeroGravity lease/drop defect is established.
- X2 is usable as a shared execution host. Before each physical run, snapshot
  total/available VRAM and KFD owners, launch only if the requested workload fits
  alongside existing residents, and compare after-run state to that exact
  pre-run baseline. Clean up only ZeroGravity's own delta. The KFD process subtree
  is read-only from the sandbox; the AMDGPU reset path is operator-level and was
  not invoked. Do not reset GPU, restart dotlink, kill unseen host processes or
  force KFD reclamation merely to create exclusive access. Running kernel7.3.1 vs
  current-system7.3.3 remains only an environment observation.
- Historical R9/R10 HTTP504 ran 5 Nov; `d16f096` explicit75GiB resident-budget
  plumbing landed 8 Nov. The old ~240s streamed path exceeded the unchanged180s
  deadline; the later resident C1 path was ~5.5s. Preserve that chronology rather
  than treating the old 504 as a resident-route result.
- The accepted 66-call 27B reference and populated1K/2K reference material are not
  present in the current X2 cache. Full numerical/context acceptance remains
  blocked until exact approved material is recovered or a properly supervised
  successor capture is established; never substitute/repin.
- **20 Nov current-source 27B HTTP recheck is environment-gated before HIP allocation.**
  X2 was synced to exact L5 `dfc3e39` bytes by Git bundle and had a clean passive
  baseline of274,448,384B VRAM /0% busy, so the explicit75GiB envelope fit. A temporary
  reference-free active1/context1K/output3/47-token HTTP probe compiled successfully
  with the already-accepted optimized test profile (opt3/codegen16/non-LTO/debug0);
  release thin-LTO compilation was again externally SIGKILLed and was not looped.
  The accepted child-net wrapper
  `bwrap --unshare-net --bind / / --dev-bind /dev /dev --` plus guarded loopback
  setup was recognized by `native-gpu.py --require-private-network` as
  `bubblewrap-child-network-namespace`. The runtime then failed cleanly before HIP
  allocation because the installed ROCm runtime closure cannot be loaded in the current
  locked shell: both installed `clr-7.2.4` roots pull `libLLVM.so.22.0` requiring
  `GLIBC_2.44`, while the shell provides glibc2.42. Exporting the exact HIP/ROCM root
  therefore does not repair the ABI mismatch. No HTTP/model result follows, no GPU delta
  occurred, and X2 returned exactly to274,448,384B/0%. The temporary probe and bundle
  were removed and X2 restored to its original `b599db3` main checkout. Do not spend
  model-execution turns reconstructing this host ROCm closure; repair the X2 environment
  separately before further27B/ROCm2B physical work. vLLM work is explicitly deferred by
  owner for now; continue L5-only non-vLLM implementation/correctness work next.
- **21 Nov dynamic mixed-prefill grouping was tested and deliberately rejected.**
  Current CUDA source already admits reduction-compatible ragged prompt chunks into one
  native cohort:48/49 share the authenticated source-derived projection reduction class,
  projections/GDN execute jointly, and logical scheduler order remains first-seen/fair.
  The remaining attention path batches only consecutive equal-row quartets. A temporary
  candidate kept scheduler order untouched, stably clustered at most8 private physical
  lanes by row count, temporarily permuted session refs only for physical publication,
  restored logical session/readback order, and allowed the existing B4 attention kernel
  to consume each equal-row quartet with per-lane fallback for leftovers. There was no
  new GPU allocation/kernel or changed root/state authority. A temporary public-SDK cell
  used singleton48/49 references and an interleaved native-B8
  [49,48,49,48,49,48,49,48] wave with3 outputs; every row remained singleton-exact and
  reported native width8. Under the matched opt1/codegen64/non-LTO profile, unchanged
  source measured299.985ms after warmup; the candidate measured324.231ms and313.585ms
  on two runs, about8.1% and4.5% slower. It remains rejected for this Qwen .8B selector,
  but its reusable allocation-free lane clustering/logical↔physical mapping/equal-group
  implementation is retained in `performance_candidates::LaneRegroupPlan`; only the
  temporary Qwen-specific wiring/probe was removed. Re-evaluate the retained candidate
  on other model/shape cells rather than reconstructing it. Next L5 implementation/
  qualification target was the genuine populated-context ladder rather than another
  batching fixture. **21 Nov current public CUDA serving now passes4K,8K and16K for both
  0.8B and2B.** The retained env-parameterized public-SDK qualifier loads only the
  approved small snapshots, uses active1/native_rows0 and the declared context capacity,
  submits an exact context-2 token prompt, observes two GPU-rooted output tokens, then
  repeats the request on a fresh session.4K/8K were already green; the16K slice now also
  passes0.8B and2B with exact replay, distinct session IDs, exact producer frontiers and
  clean shutdown. The first0.8B16K attempt was externally SIGKILLed during Rust compile,
  before the test binary ran; residue/GPU/kernel checks were clean and one cached retry
  passed in~62s.2B16K then passed in~76s. The qualifier remains retained in
  `fabric_sdk.rs`; no new fixture or token pin was added. This proves current public
  serving/chunk continuation through16K; it does **not** claim independent
  all42/full-logit numerical closure because the populated reference archive is absent
  on L5. **The old2B32K mandatory-refusal statement is now superseded.** It came from
  the28Oct fully-resident-weight calculation (~8.98MiB over4608MiB), but current shared
  source admission has supported streamed-weight fallback since `d16f096` on8Nov.
  A21Nov public2B32K `QwenCuda` load returned `Ok(())`, so current code must not be
  forced back to that historical refusal. The exact current load admission is now
  inspectable through the retained host-only `ResidentModel::ordinary_cuda_source_preflight`
  primitive, which shares the loader's actual fit helper rather than duplicating an
  estimator. With active1/native_rows0/provider4608MiB at32K,0.8B remains fully resident:
  provider=4,831,838,208B, fit=4,831,826,472B, source weights=1,503,789,056B resident,
  roots=4x422,854,656B, runtime reserve=11,736B, model admission=3,427,673,077B,
  total=3,427,684,813B.2B deliberately streams its3,762,552,832B source weights:
  resident weights=0, roots=4x422,854,656B, runtime reserve=11,736B,
  model admission=1,925,098,485B and total=1,925,110,221B. Both are far below the
  configured provider cap; the old2B refusal was purely a pre-streaming calculation.
  32K admission is closed; execution is now **partially closed**. Historical0.8B32K
  attempts were externally SIGKILLed, and2B32K loaded successfully but remained
  execution-open after a~230s external SIGKILL. The external termination also reproduced
  on compile-only current-source builds, proving it was not unique to inference. After
  the owner requested two additional trace runs, the exact already-built optimized
  0.8B32K public-SDK binary was executed directly under `native-gpu.py`, with Cargo
  absent from the process tree. **Trace run1 passed end-to-end in185.18s**:
  `P12_CUDA_POPULATED_CONTEXT_PASS model=08 context=32768 prompt_tokens=32766
  output_tokens=2 replay=exact independent_sessions=true producer_frontiers=exact
  final_shutdown=true independent_all42_reference=false`. This is current positive
  capability evidence for0.8B32K serving/chunk continuation. The **identical trace run2**
  then received explicit `SIGKILL from outside` at~80.7s after crossing the60s execution
  mark. Post-run state remained clean: GPU37MiB/0%, no process/kernel OOM/Xid,
  service-cgroup peak6,862,434,304B, memory.max/high unlimited, oom/oom_kill=0,
  pids-limit events=0, CPU quota unlimited, and no visible oomd/journal kill record.
  Therefore separate model capability from infrastructure reliability:0.8B32K has one
  full current-path pass; repeated execution is still host/external-SIGKILL flaky.
  The retained **2B32K** qualifier is now localized beyond the old external-SIGKILL
  ambiguity. With the matched opt1/codegen64/non-LTO profile, current execution builds
  cleanly and remains alive normally until the SDK's existing600s first-response
  deadline, then fails as `populated CUDA context response deadline: Timeout`
  (representative total615.82s). Multiple follow-up runs likewise remained healthy from
  the harness/process perspective and ended at~616–617s, not SIGKILL. This is a genuine
  long-context progress/performance boundary, not evidence of VRAM refusal, OOM or
  numerical mismatch. Simple placement hypotheses were measured and rejected from
  production: keeping~2.9GiB of3.76GiB source weights resident, prioritizing packs used
  by `state_nodes()`, and increasing the serial source page through16MiB did not
  materially move the deadline. A32MiB page experiment hit the then-current verified
  source-workspace ticket bound before inference. The underlying host-accounting bug is
  now fixed generically: the serial source path retains two independent page-sized host
  owners (`Bytes` payload + backend/bounce headroom), so it holds two separately bounded
  `WindowWorkspace` tickets instead of one artificial `2*n` ticket. The existing
  per-ticket32MiB source bound is unchanged; no bound was weakened. Partial-residency and
  sized-page production selection from the old experiment remain removed.
  One source-only admission cleanup is now accepted before further profiling: ordinary
  CUDA attention carried a capacity-sized `workspace_reserve` even though the retained
  tile-local attention kernel consumes no global scratch. Removing that dead lease from
  layout admission, cold allocation and access ownership cuts32K ordinary-CUDA workspace
  from151,012,757B to12,584,241B (−138,428,516B). The low-level two-root2B32K inventory
  therefore fits fully resident at4,702,464,913B under the4608MiB provider envelope.
  Public active1 resident-text execution still deliberately uses four logical root slots; do **not**
  reduce the `2*max_sessions+2` lifecycle pool merely to force residency—the extra
  ownership covers public Pending/snapshot/displaced-root lifetimes. The resolution is
  instead the existing paged-state policy: immutable committed prefix pages are shared,
  so admission now prices one full committed page set per active session plus only the
  physically reachable COW/appended pages for the remaining logical root slots. This
  preserves all four logical roots while avoiding four full contiguous32K KV slabs.
  Host qualification proves the paged2B32K four-root plan keeps all3,762,552,832B source
  weights resident. A matched public-SDK A/B at the **same2B/32K capacity and1024-token
  prompt** moved from283.22s contiguous/streamed to28.56s paged/resident (~9.9x faster),
  both exact with clean shutdown. The retained full32766-token paged qualifier then passed
  end-to-end in224.80s test execution /260.34s wall, inside the existing600s first-response
  contract, with exact replay, independent sessions, exact producer frontiers and clean
  shutdown. **Model-start smart configuration now owns the default choice.** The shared
  `smart_config` layer is model-neutral and resolves independent typed knobs once at cold
  model start; adapters can add/remove knobs without changing the core precedence/report
  contract. Modes are exactly: `Exclusive` = Auto owns supported knobs and user values for
  those knobs are ignored; `Enabled` (default) = user-provided values win per knob and Auto
  resolves the rest; `Disabled` = no Auto, use user values or static model defaults. Each
  Auto evaluation may consume a fresh read-only runtime/device observation, but ordinary
  resource admission remains authoritative and no decision enters the token hot path.
  Resolved choices emit one compact `smart-config` startup line with mode, observation time,
  source (`auto`/`user`/`default`) and rationale. Qwen CUDA is the first adapter and now
  resolves **two independent knobs jointly**: `state_policy` and `native_rows`. Candidate
  startup configs are tiny (at most `{Contiguous,Paged64/64} × {8,4,0}` filtered by active
  capacity/user constraints) and use the exact resident-text preflight against current CUDA
  ordinal0 free VRAM. Selection priority is deliberately simple: preserve resident weights
  first, then widest qualified native cohort, then Contiguous as the lower-overhead tie.
  This prevents smart batching from accidentally reintroducing weight streaming. Canonical
  `QwenCuda` uses `Enabled`; `QwenCudaWithStatePolicy` supplies only that user knob while
  native width remains Auto; `QwenCudaWithSmartConfig` exposes the global mode plus
  extensible `QwenCudaSmartOverrides`. Legacy nonzero `profile.native_rows` is folded into
  the override map; zero means unspecified under smart CUDA. Explicit `Some(0)` forces
  sequential execution without disabling Auto for other knobs.
  Physical default-Qwen acceptance on2B/32K/1024 logged Auto selecting paged because
  contiguous streamed weights while paged stayed resident, then passed in26.51s with exact
  replay/frontiers and clean shutdown. The existing `active=8` native SDK cell now leaves
  `native_rows=0` unspecified: Auto selected **8** for both halves while the state knob was
  Auto→Contiguous on one half and user→Paged4/64 on the other. Actual publication remained
  `{8}` with exact tokens/clean shutdown; matched opt1 measured176.498ms/135.979 tok/s
  contiguous and180.446ms/133.004 tok/s paged. No new benchmark fixture was added.
  **Authoritative live CUDA admission now follows the
  advisory smart decision.** `new_pinned_cuda` cleans reclaimable cold allocator/cache
  blocks, queries device-wide free VRAM from the original CUDA actor/context, preserves the
  configured `device_cap-provider_cap` headroom, and clamps the provider envelope to
  `min(configured_provider, free-headroom)`. If the cap falls, the loader reruns the same
  exact admission/placement helper under that lower cap before source binding/allocation;
  the resident ledger then sub-admits inside that clamped envelope. This is intentionally
  distinct from the read-only smart observation: a race or changed environment cannot
  overrule final resource admission. Pure arithmetic tests cover full-cap, reduced-cap,
  headroom exhaustion and invalid envelopes. A real default2B/32K/1024 run through this
  actor-side clamp passed in27.58s exact with clean shutdown on an idle card. The qualifier
  reuses the existing test; no duplicate fixture or token pin was added.
- **Retained-candidate policy:** reusable performance experiments are no longer deleted
  merely because one model/shape loses an A/B. Model-neutral candidates live dormant,
  with no production selector/callsite, so Qwen2B/27B, Gemma4 and future models can
  measure them independently. A negative result remains authoritative only for its
  measured cell.
- **21 Nov deleted-candidate restoration sweep is complete for the recoverable kernel
  mechanics.** Existing retained primitives such as ragged B4/B8 attention, packed
  private-KV transition, batched rotary, grouped state transport and stateless fused
  math were not duplicated. The dormant inventory now includes
  `performance_candidates::LaneRegroupPlan`,
  `cuda::performance_candidates::launch_copy_zero_tail`, seventeen
  `Bf16GemmCandidateKind` strategies, two recurrent-transpose directions and cached-GDN
  occupancy candidates for min-block hints8/16/24. The GEMM set covers the prior
  inner-major LaneBlocked B2/B4/B8, LaneStrided CTA64/CTA256, RoundedSegments M32/K64,
  paired-BF16 unpack, Ordered M128/N16, row-major shared-input/paired-column/split4 and
  inner-major warp-shared/paired-column/unroll8 strategies, plus the22Nov recovery of
  LaneBlocked shared-row terms64, Rounded-K64 paired BF16 segment-boundary conversion,
  Ordered-K64 weight-only `.cg` staging and Rounded-K64 weight-only `.cg` staging. Git history plus1,433
  unreachable blobs contained the historical measurements/docs but no surviving source
  blobs for those uncommitted A/B kernels, so they are clean reconstructions over the
  current accepted arithmetic source rather than falsely claimed byte recovery. The
  shared-input candidate is explicitly named `...SharedInputF32` because the old notes
  establish shared staging+widening but not the vanished literal implementation.
  Production selection is unchanged and there are zero callsites outside the dormant
  candidate modules.
  Physical compile-only SM86 checks passed first for the initial five GEMM candidates,
  then for7 GEMM +2 recurrent +3 GDN occupancy candidates, and finally for the expanded
  candidate translation unit after the remaining GEMM strategies were added. The22Nov
  recovery extension also physically compiled all four newly restored GEMM variants on
  SM86; the reconstructed paired BF16 boundary variant additionally matched accepted
  Rounded-K64 bytes and status exactly on a physical two-segment case. These are
  code-readiness/correctness results only, not performance acceptance for a new model.
  A fresh sweep of the remaining historical `removed` candidate notes found no other lost
  reusable implementation: grouped recurrent/Conv, packed greedy, head partials, dual RMS,
  certified recurrent, shared-input projection, attention/KV/rotary and stateless mechanics
  are already retained; partial residency, pack ordering, page-size and CTA-width trials are
  parameter/policy experiments over mechanisms that still exist; temporary profilers/runners
  remain disposable; and ownership/ABI-invalid forms such as the >32-binding single-launch
  B8 KV design remain intentionally dead rather than preserved as unsafe primitives.
- **GPU test/cfg drift exposed by the restoration sweep was repaired without changing
  runtime behavior.** `ResidentPlaneLayout` ownership is no longer hidden behind an
  unrelated feature gate; graph witness/test-support/quarantine observers are available
  under the test cfgs that consume them; old cached-GDN manifests explicitly reject the
  newer `TextPrefillCohort` phase; and stale GDN runtime tests pass the required lane
  count. Default, plain-`qualification` and CUDA-native `cargo check --tests` now
  type-check. The larger `cuda-native-qualification` test/library checks were repeatedly
  externally SIGKILLed immediately after entering the GPU crate with no Rust diagnostic;
  do not claim that matrix green or infer a source failure from those kills.
- ROCm2B C4 warmup plus C16/C32/C64 warmup/timed mismatches remain unlocalized.
  Preserve failed cells and expected IDs; do not repin expectations.
- These remaining campaigns are explicitly authorized X2 execution/test work;
  repository/source edits still originate on l5.

### CUDA composition / serving foundation

The CUDA route is a real source-derived resident path, not a finite-case
dispatcher.

Important foundation commits:

- `4f1e827` shared source-derived CUDA composition.
- `a57590e` output-reduction repair.
- `9d73d46` compact on-device greedy/token feedback.
- `9f45471` exact decode GEMV specialization.
- `55e195c` onward: genuine ordinary inter-session cohorts.
- `b191fc6`: exact independent source reductions survive physical row widening.
- `fa703fa` / `96a7ac7`: same-frontier chunked/ragged prefill and mutual cohort
  compatibility.

Current serving uses genuine B4/B8 physical cohorts with B2 partial regrouping.
Same-frontier ragged/chunked prefill is accepted. Unequal-frontier prefill remains
fail-closed. The retained `874e67d` shared-input Ordered+LaneBlocked B8 projection
pair remains model-neutral and exact, but its ordinary-Qwen serving consumer is now
**rejected**: public Qwen3.5-2B B8 prompts47..54/16 outputs preserved all128 ID264
tokens and native width8, while selector-off/on/reverse process-order measurements
had much larger side drift than the ~1.04% favorable two-sided candidate mean.
Keep the primitive; do not select the consumer from that noisy aggregate.

## CUDA execution/performance chronology

The exact measurements/tests live in `model-execution/TODO.md`. This is the
takeover map.

### 9–10 Nov — C1 to genuine B4/B8

- Compact greedy feedback moved to a tiny device packet instead of full-vocab host
  readback.
- Exact decode GEMV specializations landed.
- 4-row forward-source cohorts shared stateless projection/head while preserving
  private state.
- B4 GDN convolution, recurrent core and full attention became genuine physical
  batching primitives.
- BF16 finite certification was folded into GEMM stores where legal.
- B4 lane-strided output head shared weights across rows without changing per-row
  arithmetic.
- Full-prompt prefill cohorts landed.
- Native width widened to B8 while stateful GDN/attention initially remained
  composed from qualified B4 groups.

### 11 Nov — dense B8, chunking and partial populations

Accepted generic K64 families:
`67586eb` ordered large-K prefill row64, `77ffc23` ordered B8 decode,
`5a75a70` RoundedSegments B8 and `2b8e90f` interleaved-pair B8.

Other work:
`74f6422` populated small contexts; `597da3f` cross-profile measurement support;
`b95acdc` B2 partial regrouping; `fa703fa` chunked/ragged state-only intermediate
prefill; `0d9a56d` / `96a7ac7` mutual cohort compatibility.

Decisions:
- B4/B8 remain native widths; B2 handles partial populations.
- Larger offered populations become multiple B8 groups, not hidden B16.
- **B16 is rejected**: 0.8B was slightly worse than two B8 groups; 2B regressed
  catastrophically. Do not resurrect without materially different architecture.
- Unequal-frontier prefill previously diverged and remains held.

### 12 Nov — exact arithmetic + reusable stateless primitives

`b191fc6` fixed the key correctness boundary: physical widening must not select a
different source reduction. Exact B2/B4/B8 row kernels, ragged cohorts, inner-major
head packing, two-segment row split, sigmoid×multiply, stateless row spans,
RMS/residual-RMS, gate, Q/K RMS and source-exact rotary primitives followed.
Several consumers were rejected because lower dispatch count did not yield
repeatable serving gain; the clean primitives remain.

### 13 Nov — stateful primitives, roots and hot-path profiling

Important decisions before the current selected B8 state path:

- B8 gated GDN primitive retained; ordinary consumer rejected.
- LaneBlocked inner-major rejected once pack cost was included.
- Multiple dominant LaneStrided<16,512> CTA/load/unpack/warp/unroll variants were
  exact but flat/unstable/slower and removed.
- A shared-memory tiled recurrent transpose was ~12–13% faster in isolated copy
  timing but slower under sustained serving; removed.
- `fe3b122` strengthened transport oracles with an independent CPU index oracle.
- `0cdfaad` retained private ragged B8 attention; current ordinary selection held.
- `d196c1b` / `bad4713` qualified private cached-GDN state binding and in-place
  alias behavior.
- `16fba0e` retained an exact-root model-neutral derived-state cache; its first
  private-GDN serving representation was ~25.6% slower despite exact-root hits,
  so that consumer is rejected.
- `2f651b7` centralized session construction and removed first-lane prefill-binding
  coupling while keeping same-frontier admission.

### Selected: `60b048d` canonical-boundary private B8 recurrent core

Current serving selects this.

- Canonical F32 recurrent roots stay the only publication/checkpoint authority.
- Eight lanes still import exact canonical state, execute one private B8 recurrent
  core in-place, then export exact canonical candidates.
- Gated output remains two accepted B4 groups.
- No native recurrent state persists across tokens.
- Qwen3.5-2B B8 47+16 kept sixteen ID264 tokens/lane.
- Controlled candidate mean ~716.377 ms vs `2f651b7` baseline ~745.102 ms:
  ~3.86% lower latency / ~4.01% higher throughput in that short workload.

### Selected: `775dfa6` ragged private B8 KV

Current serving selects this.

- Extends packed-BF16 private KV from B4 to independent ragged B8 bases.
- Canonical BF16 K/V roots remain publication authority.
- Candidate fully writes prefix copy + append + future zero tail + nonfinite status.
- A single 32-root launch was rejected because it exceeded
  `MAX_BINDINGS=32`; the cap was not widened and no opaque pointer table was added.
- B8 therefore uses two reusable plane launches, key then value.
- Physical ragged bases/guards/nonfinite/multi-token/capacity1024 qualifier passes.
- Controlled candidate mean ~709.600 ms vs bracketed `60b048d` baseline
  ~722.432 ms: ~1.78% lower latency / ~1.80% higher throughput.

### Retained: `c12acab` grouped B8 recurrent transport; consumer rejected

This is the current remote origin.

- Exact model-neutral 8-root recurrent transport, 17 explicit bindings.
- Qualified against scalar transport and CPU index oracle.
- Serving grouped import/export was only ~0.24% better against a two-sided
  baseline and slower than the first baseline side; rejected.
- Primitive remains; serving does not select it.

### 13–14 Nov retained primitives; serving consumers rejected

These signed local commits are **retained shared primitives**, not selected
ordinary serving paths:

- `85f458f` packed BF16 batch greedy:
  exact B2/B4/B8 numeric-zero ordering; serving evidence only ~0.44% one-sided and
  lacked a clean reverse bracket, so not selected.
- `f303bca` exact B8 head→greedy partials:
  exact accepted logit arithmetic and packets; serial variants were ~8.8% and
  ~2.9% slower, CTA-preserving path ~0.12% slower/noise. Full head + greedy stays.
- `68ac810` grouped ConvB4 transport:
  exact grouped/scalar/CPU-oracle parity; ordinary 20→8 launch composition was
  flat/slightly slower (~688.920 vs 689.195 ms).
- `da3eae5` dual Q/K head RMS:
  exact physical composition; serving ~0.28% slower (~692.004 vs 693.917 ms).
- `ee347d9` certified recurrent export:
  exact RecurrentOut + gated BF16 certification, including exact status 0x0a
  combined fault case; serving composition ~0.33% slower (~679.122 vs 681.355 ms).

### Selected: `399e43a` B8 private-KV CTA256

Latest selected performance change.

- Only B8 private-KV CTA width changes 32 → **256 threads**; B4 remains 32.
- One U32/two BF16 values per thread remains unchanged.
- No arithmetic, root/state, ABI, prefix/append/tail/status change.
- Full guarded physical matrix passes at widths 32/64/128/256.
- Same-process Qwen3.5-2B B8 means:
  - 32: 687.760 ms
  - 64: 678.130 ms
  - 128: 678.374 ms
  - 256: **675.905 ms**, stable 675.148–676.997 ms
- Accepted benefit vs CTA32: ~1.72% lower latency / ~1.75% higher throughput.

### 14 Nov post-KV probes: `de2eed8` / `cb9e349`

These commits record rejections only; production source was restored.

Fresh relative post-CTA256 projection ranking from temporary serialized device
events:

- 24× 12288×2048 LaneStrided<16,512>: 5252.096 µs total / 218.837 mean.
- 24× 2048×6144 RoundedSegments<2048>: 3336.128 / 139.005 µs.
- 18× 8192×2048 Ordered: 2639.872 / 146.660 µs.
- 24× 2048×2048 LaneStrided<16,256>: 1148.160 / 47.840 µs.
- 18× 32×2048 LaneBlocked<32,64>: 1124.352 / 62.464 µs.
- 6× 5120×2048 LaneStrided<16,512>: 682.080 / 113.680 µs.
- Two observed 248320-head launches totalled ~7068.768 µs.

Use these as relative family ranking only; the probe serialized kernels.

Rejected after CTA256:
- recurrent transport CTA128/256/512: exact, no stable gain; CTA256 had a severe
  stall and CTA512 order effects crossed.
- Rounded paired BF16x2 segment conversion: exact, ~0.12% device-time difference.
- LaneBlocked<32,64> B2/B4/B8 shared-weight specialization: exact, but direct
  kernel timing showed accepted ~91.9 µs after first outlier vs candidate
  ~107.5 µs; apparent whole-model ~0.74% win was noise.
- weight-only `cp.async.cg` for K64 Ordered/Rounded: exact; Ordered repeatedly
  ~1.2% slower and Rounded flipped direction between repeats.
- All experimental helpers/selectors/tests/runners were removed.

### Retained: `874e67d` shared-input B8 projection pair; consumer rejected

- Adds one model-neutral B8 physical primitive that composes an exact Ordered K64
  projection with an independent `LaneBlocked<32,T>` projection under one launch.
  Both original reduction orders, row-major activation semantics, independent
  outputs/status and source/session/root/publication boundaries remain unchanged.
- The secondary projection remains one row per CTA; this is **not** the previously
  rejected LaneBlocked64 row-sharing specialization and is also distinct from the
  old rejected LaneStrided shared-input activation-staging experiment.
- RTX3060 physical qualification matches both accepted GEMMs bit-for-bit with zero
  status. Direct alternating timing at the real 2B hot pair
  `8192x8x2048 + 32x8x2048` measured separate launches at 168.230 µs mean versus
  155.752 µs composed (~7.42% lower primitive time).
- The missing loaded-model A/B is now complete. A temporary structural consumer
  composed only the real adjacent `Qkvz(normal)` then `Ba(normal)` B8 projections
  when both shared the exact SSA input, retained resident weights, and matched the
  primitive's Ordered + LaneBlocked<32,T> contracts. Public Qwen3.5-2B ragged
  prompts47..54 with16 outputs preserved all128 ID264 tokens and native width8.
- Reject ordinary-Qwen selection: selector-off first side mean665.690ms, candidate
  mean701.917ms, reverse baseline722.543ms, reverse candidate671.881ms. The
  two-sided aggregate favors candidate only ~1.04%, while the sign flips by order
  and side drift is much larger than the net effect. This does not meet the
  repeatability bar. Temporary consumer/selector/runner/include-view artifacts were
  removed; keep the model-neutral primitive for a materially different consumer.

## Current CUDA forward selected path

### Selected

- Exact independent source reduction contracts (`b191fc6`).
- B4/B8 cohorts with B2 partial regrouping.
- Same-frontier chunked/ragged prefill.
- Accepted K64 Ordered/Rounded/Interleaved GEMM families.
- Exact row-split GDN consumer (`d1ec723`).
- Canonical-boundary private B8 recurrent core (`60b048d`).
- Ragged private B8 KV (`775dfa6`).
- B8 private-KV CTA256 (`399e43a`).
- Existing B4 GDN gated path.
- Existing B4 attention composition for ordinary B8 serving.
- Full BF16 output head + existing greedy path.
- Canonical state/root publication.

### Retained but not selected in ordinary serving

- Private ragged B8 attention (`0cdfaad`).
- Cached-GDN B8 gated primitive.
- Exact-root derived-state cache (`16fba0e`).
- Grouped B8 recurrent transport (`c12acab`).
- Packed BF16 batch greedy (`85f458f`).
- Exact B8 head→greedy partials (`f303bca`).
- Grouped ConvB4 transport (`68ac810`).
- Dual Q/K head RMS (`da3eae5`).
- Certified recurrent export (`ee347d9`).
- Shared-input Ordered + LaneBlocked B8 projection pair (`874e67d`); ordinary-Qwen
  consumer rejected after order-sensitive loaded-model A/B, primitive retained.
- Other stateless row/RMS/gate/QK/rotary primitives whose current consumer
  economics were rejected.

Never infer “primitive exists” == “serving selects it”.

## Reusable shared primitives that must not be lost

Retain model-neutral:
exact BF16 GEMM reductions; B2/B4/B8 physical rows; K64
Ordered/Rounded/Interleaved staging; row-major↔inner-major BF16 transport;
StatelessRowExtent/RMS/residual-RMS/SiLU/gate/QK-RMS; sigmoid×multiply;
two-segment row split; B4/B8 packed-BF16 KV; source-exact rotary; private
B4/B8 GDN recurrent + alias properties; B4/B8 gated; ragged B4/B8 attention;
exact-root derived state; scalar/grouped recurrent and Conv transports; packed
BF16 greedy/continuation; head→greedy partials; dual Q/K RMS; certified recurrent
export; exact shared-input Ordered+LaneBlocked B8 projection composition;
compiler/transport/oracle utilities.

A rejected Qwen consumer is not a reason to delete these primitives.

## Important rejected experiments / do not casually repeat

Exact data is in the owning TODO. Major rejections:

- B16/native16.
- Unequal-frontier serving admission without new per-lane source qualification.
- Persistent/derived native GDN serving representation.
- Grouped B8 recurrent transport consumer.
- Tiled recurrent transpose consumer.
- recurrent CTA width sweep.
- grouped ConvB4 consumer.
- B8 gated consumer.
- private B8 attention consumer without materially new composition.
- certified export consumer.
- dual Q/K RMS consumer.
- packed BF16 greedy selection on weak/noisy evidence.
- existing head→greedy tilings.
- LaneBlocked inner-major with pack cost.
- LaneBlocked64 shared-weight specialization.
- Ordered M128/N16.
- Rounded M32/two-warp and paired-BF16 segment boundary.
- K64 weight-only `.cg` cache hint.
- dominant LaneStrided CTA-only/**old LaneStrided shared-input staging**/
  paired-column/warp-sharing/paired-unpack/unroll variants. Do not confuse this
  rejected family with retained `874e67d`, which composes two different exact
  projections sharing one activation boundary.
- full broad requalification of unchanged primitives.
- large CUDA acceptance from cross-profile NMSE/greedy-token equality alone.

## Current performance frontier

Do not generalize quick non-LTO/direct API numbers to HTTP, long context, graph
capture, another GPU or another model.

- Original C1 anchor: 0.8B native ~65.22 ms vs eager vLLM ~90.64 ms; 2B native
  ~114.29 ms vs eager vLLM ~84.34 ms.
- Warm 2B B8 cached-decode traversal has historically been ~25–28 ms/token
  depending clocks/build state.
- Dense projection remains the largest measured region; the fresh family ranking
  above is the current local guide.
- B8 recurrent core, B8 KV transition and CTA256 are accepted material wins.
- `874e67d` reduces one directly timed real 2B Ordered+LaneBlocked projection pair
  from 168.230 to 155.752 µs mean (~7.42%), but has no completed loaded-model A/B;
  it is a retained primitive opportunity, not a serving performance claim.
- Recent 47..54 +16 whole-model waves commonly land ~675–705 ms depending
  clock/thermal state.
- Cross-process timing has repeatedly reversed sub-1% conclusions.
- For sub-1% candidates, direct device timing + same-process ABBA/reverse order is
  the minimum useful evidence.

## Numerical / state / ownership rules

- Exact source identity/geometry explicit.
- No backend grant from metadata, registry names or another target.
- Every numerical profile owns its arithmetic contract; no universal epsilon.
- Physical widening preserves original source reduction order.
- Canonical roots remain publication/checkpoint/rollback authority.
- Derived representations are subordinate exact-root caches only.
- Restore/reset/replacement cannot reuse stale derivatives.
- Private in-place mutation is only a physical primitive property until a
  lifecycle explicitly qualifies/selects it.
- One ResourceArbiter; no hidden second allocator/budget system.
- Completion/event/fence governs release/quarantine.
- Cancellation cannot publish candidate state.
- Sessions remain independent through cohorts.
- Canonical future KV tail stays exact zero where required.
- Passing cleanup/final cases do not relabel earlier failures.
- Compilation, synthetic fixtures and token equality alone do not establish full
  model/backend acceptance.
- Operator correctness, model distribution quality and determinism are separate
  claims. Follow `CORRECTNESS_POLICY.md`.

## Current blockers / open work

### l5 source/repository work

1. **Do not force another CUDA micro-optimization.** Easy launch-count, CTA-width,
   cache-hint and instruction-count variants are heavily exhausted. `874e67d`
   remains a useful exact primitive, but its real adjacent ordinary-Qwen consumer
   has now been measured and rejected because order/thermal drift dwarfed the
   ~1.04% favorable two-sided aggregate. Do not retry the same selection without a
   materially different composition.
2. **Production CUDA graph selector is privately integrated at signed `9d6613c`.**
   `5bf440d`/`db7be60`/`6a5c610`/`81b7c71`/`f1ce532` establish the graph substrate,
   resource admission, driver-safe compilation, lifetime witness and exact captured
   storage/access authority; `63ee14e` adds the workspace-owned lifecycle/cache.
   `9d6613c` adds the fail-closed selector under that same workspace authority. Exact
   compatible live cache selects Graph; absent/cold/spent, incompatible profile or
   device-token route select Native before graph enqueue; identity/lifecycle drift,
   unresolved selection or replay failure is an error with no Native retry. External
   graph APIs remain qualification-only and ordinary serving still selects
   `FusedExecution::Native`.
3. The graph policy remains64KiB control +4KiB per prepaid source-dispatch slot +
   Rust graph/receipt/lease-access-vector metadata; actual CUDA driver bytes remain
   unqualified. The charge is acquired before `graph_prepare`, may grow only before
   capture starts, survives CaptureWindow/CapturedResidentGraph/quarantine ownership
   and releases only after trusted retirement. The current charge remains exactly
   **15,356,224 bytes** (32,768 bytes owned lease/access-vector metadata). The final
   supervised original0.8B B2 lifecycle run preserves warm `[3725,3725]`, body3632,
   six replay enqueues,14,900 committed kernels, full248320/all42 exactness and
   final41MiB/reserved0; it additionally proves compatible cache lookup,
   incompatible-profile miss, second-capture refusal, explicit retire->Spent/
   foreign0, fresh-workspace recapture, and B2->B4 trusted retire-before-replace->
   Cold/foreign0. One preceding fresh lifecycle invocation disappeared before a
   result with no retained process/VRAM delta; keep it incomplete.
4. One older post-`9dca4b` repeat failed closed at the first post-capture idle
   check: total resident bytes fell by exactly15,339,840 with pending_events=0 and
   an empty queue. It did not reproduce in four subsequent pre-`63ee14e` runs or
   the final lifecycle/selector runs. Do not weaken the idle invariant or call the
   transient explained. Driver-internal graph bytes/node enumeration remain
   unqualified. The `9d6613c` same-process B2 AB/BA diagnostic measured Native
   mean273,933us/median267,847us versus selected Graph mean264,881us/median263,238us
   (~3.3% mean/~1.7% median). That is positive but modest: do **not** enable or claim
   production graph serving from B2 alone. Next obtain matched B4/B8 evidence and
   decide whether capture/selection policy earns the retained graph complexity.
5. **Large CUDA same-contract distribution/state acceptance is open but blocked.**
   Cross-profile diagnostics are not acceptance. Approved original reference/archive
   material needed for the intended rerun is missing on l5; do not invent
   tolerances, repin or substitute another model/reference.
6. **Unequal-frontier prefill remains held.** First-lane coupling was hardened, but
   admission stays same-frontier. Per-lane source position/base/state semantics
   still need approved HF source plus private-singleton physical comparison before
   scheduler admission changes.
7. Broader HTTP/context/vLLM campaigns remain separate from quick CUDA direct
   feedback. Continue the larger runtime/serving capability rather than returning
   to endless sub-percent kernel churn.

### x2 AMD large-VRAM execution/testing — 27B host-load boundary next

1. **Tiny physical HIP health stays green.** The isolated `gfx1151` smoke passed
   and returned to baseline; do not repeat it unless host/runtime state changes.
2. **Canonical27B host loading is green.** Exact tokenizer construction,
   isolated source binding/seal/1K plan and the full runtime host test all pass.
   The full-runtime test needed only an external progress heartbeat to keep the X2
   connector execution alive; do not mistake that transport behavior for a model
   requirement or source bug.
3. **X2 is a shared GPU host.** Existing KFD owner `2406838` / ~31.1GB is
   not a blocker and cannot be attributed to ZeroGravity from the private PID
   namespace because Ollama/model workloads also run on the host. The old
   “recover owner before any model retry” rule is retired. A matched35s non-GPU
   native-gpu execution control remains green.
4. Before a new27B/ROCm run, record exact total/used/available VRAM and KFD baseline,
   then run only if the requested ZeroGravity configuration fits alongside
   existing workloads. Track our process and incremental VRAM/queues, and require
   those deltas to retire after normal exit. Never evict/reset/reclaim another
   owner. The last observed ~31.38GB shared baseline left ~71.7GB before overhead,
   so re-evaluate the exact75GiB route against **current available physical
   capacity** instead of assuming exclusive-card capacity. Preserve the unchanged
   180s HTTP deadline. Missing approved27B populated/reference material remains a
   separate numerical-acceptance blocker.
5. Before every X2 physical run, sync X2 to the exact current l5 `main`, confirm
   one checkout, record active workload/KFD/VRAM state, and avoid colliding with
   insufficient remaining capacity. X2 is not an independent source-authoring
   branch.

## Validation / execution cadence

1. Revalidate status, signed HEAD, one-worktree topology, active processes and GPU.
2. Read the latest owning TODO section.
3. Implement one substantial shared unit.
4. Use rustfmt/diff/compile/narrow checks during implementation.
5. Add physical qualification only for a genuinely new numerical/layout/ownership
   contract.
6. Get quick real-model feedback.
7. Use same-process ABBA/reverse order/device events when the expected effect is
   small.
8. Reject marginal/unstable complexity and remove its selector/consumer.
9. Keep independently useful shared primitives where justified.
10. Remove all temporary profiling/runners/include views.
11. Run strict Clippy/focused checks for changed surfaces.
12. Run `scripts/check.sh` only at an assembled acceptance phase.
13. Stage only owned paths, inspect cached diff, signed commit.
14. Never push without explicit user instruction.

## l5/x2 build and GPU operational notes

Defaults:
`CARGO_BUILD_JOBS=8`, `RAYON_NUM_THREADS=4`, `RUST_TEST_THREADS=1` where useful,
`CARGO_NET_OFFLINE=true`, `PYTHONDONTWRITEBYTECODE=1` and `python3 -B` where the
host environment provides Python.

Physical GPU checks use `python3 -B scripts/native-gpu.py -- <command>` from the
established host environment. l5 is the repository/source authority; X2 is the AMD
large-VRAM execution/test host only. X2's connector shell can have a reduced PATH,
so do not treat a missing `python3`/Cargo command in the raw connector environment
as proof that the established qualification environment is absent.

Connector behavior:
- dotlink bash timeout is adjustable; there is no fixed 114-second wall.
- Long Cargo/test calls can return ambiguous null exit/success false responses.
  Never treat that as success; check processes and binary timestamp/size/executable
  bit/marker strings.
- Prefer durable `job_start` / `job_get` / `job_output` for long builds.
- Heavy debug test links have occasionally been killed without a compiler error;
  `CARGO_PROFILE_TEST_DEBUG=0` is a build-only linker-pressure workaround.
- Warmed second link attempts often succeed; verify the actual executable.

Temporary CUDA include view, recreate only when required:
- cudart:
  `/nix/store/nhr5mz136csd28mn8h5gdrbq6cbmbg0m-cuda13.4-cuda_cudart-13.4.92/include`
- CCCL:
  `/nix/store/ramrzw37i39f9fbnj6agp8kym2g056y7-cuda13.4-cccl-13.3.4.3.1/include`
- NVRTC:
  `/nix/store/njh4s557p8z871vpqwcy7i23k6swv23m-cuda13.4-cuda_nvrtc-13.4.92-lib/lib`
- view: `tmp/_tmp_cuda_include_view`
- helper: `scripts/cuda-nix-include-view.sh`
- set CUDA_PATH to the view and LD_LIBRARY_PATH to NVRTC + driver lib.
- remove the view after use.

Do not download models/SDKs, alter privileges/recovery or activate held gfx90c
paths without applicable user authorization.

## Worktree / disk discipline

- `main` is the only persistent checkout **per host**.
- No second l5 or X2 Git worktree for implementation/profiling/review.
- Author/edit source on l5. Sync exact l5 commits to X2 only as needed for AMD
  execution/testing; do not let X2 drift into a separate implementation branch.
- Reuse canonical target/cache state on each host.
- Temporary `_tmp` runners and profiling files are disposable.
- Do not recreate old worktree pools or `.agents/runs`.
- Preserve unrelated dirty work.
- No destructive Git operations without explicit authorization.

## Next agent must not redo

Do not replay or reopen without materially new architecture/evidence:

- the completed Fabric vNext transition or cancelled recurrence/watch campaigns;
- B16 active/native serving;
- unequal-frontier **prefill** admission without new approved source qualification;
- persistent derived-GDN consumer, grouped recurrent transport consumer, tiled recurrent
  transpose, recurrent CTA sweeps, grouped ConvB4 consumer, B8 gated consumer,
  dual Q/K RMS consumer, certified recurrent-export consumer;
- existing head→greedy tilings, packed greedy selection on the old weak evidence,
  LaneBlocked inner-major/terms64, Ordered M128/N16, Rounded M32/paired BF16,
  K64 `.cg`, dominant LaneStrided CTA/old shared-input staging/paired-column/
  warp-sharing/paired-unpack/unroll variants;
- the `874e67d` ordinary-Qwen consumer: the exact Ordered+LaneBlocked pair primitive
  remains useful, but its loaded-model consumer was measured and rejected because
  order/thermal drift dwarfed the small two-sided aggregate;
- the CUDA graph PTX compatibility blocker: `6a5c610` already fixes non-persistent
  NVRTC by emitting exact-SM cubin/SASS;
- graph opaque-resource admission/lifetime: `db7be60` + `81b7c71` already price,
  retain, witness and retire the charge;
- same-card graph storage/event authority: `f1ce532` already requires every replay
  to prove the exact captured `(ResidentLease, DeviceAccess)` set against the armed
  `ResidentSubmission`;
- the old X2 "dirty GPU / wait for KFD owner to disappear" interpretation. X2 is
  shared with Ollama/model workloads; existing KFD/VRAM is baseline unless
  positively attributed to our process.

Do not broadly requalify unchanged primitives merely because a new phase starts.
On X2, do not reset/reclaim/restart/kill other GPU owners. Tiny physical HIP health
is already green; repeat it only if host/runtime state changes.

## Historical next-agent prompt (superseded)

Use §13 of the [25 Nov takeover](FULL_COMPREHENSIVE_HANDOFF_2026-11-25_FOUNDATIONS.md) for the actual starting prompt. The following is retained only as history.

> Use `dotlink---l5` as the only source-authoring connector. Continue in
> `/home/pvl/spaces/abird/src/zerogravity` on the single persistent `main`
> worktree. Use `dotlink---x2` only for AMD large-VRAM execution/testing;
> synchronize exact L5 commits by Git bundle rather than pushing merely for
> transport. Do not create another persistent worktree or branch. Preserve all
> unrelated dirty/staged work.
>
> Keep each implementation turn to roughly 30 minutes. Aim for the next coherent
> implementation checkpoint, update durable owning docs/plans, sign the commit,
> then pause. Do not push unless explicitly requested.
>
> Read first, in order:
> 1. `AGENTS.md`
> 2. `.agents/plans/README.md`
> 3. `.agents/docs/FULL_COMPREHENSIVE_HANDOFF_2026-11-21.md`
> 4. `.agents/docs/CURRENT_HANDOFF.md`
> 5. `.agents/plans/TODO.md`
> 6. `.agents/plans/model-execution/TODO.md`
> 7. `.agents/plans/numerics-and-context/CORRECTNESS_POLICY.md`
> 8. `.agents/plans/serving-and-speculation/TODO.md`
> 9. `.agents/plans/serving-and-speculation/BATCHING_AND_PAGED_STATE.md`
> 10. `.agents/plans/memory-and-streaming/TODO.md`
> 11. Any owning docs/plans linked from those files.
>
> Revalidate before editing: `git status --short --branch`, `git rev-parse HEAD`,
> `git rev-parse origin/main`, remote GitHub `main` when network transport is
> available, `git log -1 --show-signature`, worktrees, active
> Cargo/rustc/model/native-gpu/Bubblewrap processes, physical GPU state,
> source/cache identities, disk space and `_tmp` artifacts.
>
> Latest signed implementation/source checkpoint at handoff creation is
> `6c68a361e843c73107a86aaedea5ca697a30258c`
> — `refactor(cuda): use ledger identity for prepared leases`.
> The signed handoff-consolidation docs commit(s) may sit above it; revalidate
> actual `HEAD` first. Pushed/tracked `origin/main` is
> `f86538345fe4ce412df81864542c5ec183446460`
> — `docs: record 32k trace pass and external kill`.
> The implementation/source lineage was 12 commits above origin before handoff
> consolidation. Do not push unless explicitly requested.
>
> For every Cargo invocation use `CARGO_BUILD_JOBS=8` and
> `RUSTC_WRAPPER=/run/current-system/sw/bin/kache`. Run physical GPU work only
> through `python3 -B scripts/native-gpu.py -- ...`. Commands genuinely requiring
> TCP/UDP loopback must use the accepted private-network envelope and
> `--require-private-network`; never weaken harness isolation.
>
> Preserve the current production decisions:
> - ordinary public CUDA graph serving remains Native; graph serving is disabled;
> - exact StateRoot publication, cancellation/unknown-completion authority,
>   ResourceArbiter ownership and physical lifetime semantics are non-negotiable;
> - canonical `QwenCuda` now uses the generic smart startup layer in Enabled mode;
>   user values override Auto per knob, Exclusive ignores supported user knob
>   values, Disabled uses user/static defaults only;
> - current Qwen smart knobs are `state_policy` and `native_rows`; exact startup
>   candidate scoring is resident weights first, then wider qualified native
>   cohort, then Contiguous as the lower-overhead tie;
> - smart config is advisory and load-time only; authoritative CUDA admission
>   re-observes live free VRAM from the original actor, preserves configured
>   device headroom and reruns exact admission under any lower live cap;
> - explicit paged 2B/32K with page64/append64 passes the full 32766-token public
>   SDK cell in 224.80s execution / 260.34s wall; same-capacity 1024-token A/B is
>   283.22s contiguous-streamed vs 28.56s paged-resident (~9.9x);
> - canonical default 2B/32K/1024 Auto-selects paged and passes in ~26–28s;
> - `active=8` SDK with native width unspecified Auto-selects B8 and physically
>   publishes `{8}`; matched opt1 is 135.979 tok/s Contiguous and 133.004 tok/s
>   user-Paged with exact tokens/clean shutdown;
> - 0.8B/32K has a complete retained current-path pass; independent populated
>   all42 numerical closure at larger contexts remains open because the approved
>   reference archive is absent;
> - exact partial-tail paged COW ownership is already closed; do not build another
>   COW abstraction;
> - canonical `DeviceBufferIdentity` is the physical allocation/range authority;
>   CUDA-forward prepared resident sets and source-page/native-embedding alias
>   safety already use ledger identity. Handle ranges remain execution binding
>   geometry, not ownership truth.
>
> Current reusable performance work must not be lost: dormant lane-regroup,
> copy+tail-zero, 17 BF16 GEMM strategies, recurrent transpose, GDN occupancy and
> backend transfer-retirement variants are retained without production consumers.
> **Reject the selector/consumer, not a correct reusable primitive.** A performance
> loss means the current selector/default/adapter is rejected; retain the clean
> model-neutral implementation, physical provider, kernel strategy or composition
> primitive so future models, shapes, batches and backends can requalify it cheaply.
> Delete the primitive only for correctness/safety/resource-contract failure or
> when an equivalent retained primitive strictly supersedes its useful domain.
> Temporary probes, runners, evidence-only fixtures and model-specific wiring should
> still be removed normally. A dormant primitive must not be described as selected
> production behavior or as a performance win.
>
> Fresh matched vLLM is still deferred by environment, not policy: the pinned
> `vllm-0.31.0-cu129` store exists, but this connector cannot enter the rootless
> Podman user namespace because `/etc/subuid`/`subgid` are unavailable. Do not
> rebuild/re-register/mutate the baseline merely for feedback. The retained
> eager-vLLM 72.20 tok/s figure is historical only.
>
> X2 ROCm physical work remains environment-gated; preserve other GPU owners and
> never reset/reclaim/restart/kill their workloads. L5 owns source changes.
>
> Highest-value next work should come from the live TODOs, not stale prose.
> Current likely lanes are: graph/transfer physical-identity adoption only where
> a real consumer exists, authenticated transfer rings/broader streaming tiers,
> independent larger-context numerical references when approved inputs become
> available, and measured smart-config knobs only when existing evidence supports
> a clean startup rule. Avoid speculative fixture matrices, debug-profile
> micro-tuning and duplicate paging/COW mechanisms.
>
> Preserve the project working style: implementation and quick real-model
> feedback first; reuse accepted evidence for unchanged primitives; keep
> model-neutral primitives clean/composable; favor zero-cost Rust abstractions
> where natural; optimize measured hot paths aggressively when the fast design
> stays simple; otherwise prefer the simpler maintainable design.
---

## Historical design foundation — retained context, not current execution instructions

The sections below preserve the original product/architecture evolution and enduring design
context from earlier handoffs. **Do not treat old “next step”, worktree, campaign or status
phrasing inside historical material as current instructions.** The current continuation is
entirely above and in the live TODOs.

## 3. Original mission and enduring product design

ZeroGravity began as a Rust-native inference compiler/runtime and evolved into a broader execution fabric.

The enduring product goal is one Rust-owned execution fabric that can support:

- text generation;
- embeddings/reranking and encoder tasks;
- image generation/editing;
- video/temporal generation;
- ASR;
- TTS and audio codecs;
- realtime/full-duplex media;
- combined multimodal programs;
- future optional training/update tasks where explicitly designed.

The user-facing aspiration is simple:

- one binary;
- one Rust SDK;
- local-first operation;
- automatic hardware selection when safe;
- explicit fit/capacity explanations;
- OpenAI-compatible serving where implemented;
- no Python inference server dependency for qualified local paths;
- the same runtime for portable and highly specialized execution rather than a separate server per model or modality.

The performance goal is measured Pareto leadership where specialization is justified, never a blanket “always fastest” claim.

The architectural idea from the beginning has been:

- model mathematics and source semantics are defined once;
- state/effects are explicit;
- portable/reference execution exists first;
- optimized hardware paths replace closed semantic regions only when legal and qualified;
- one logical session/root authority publishes committed state;
- one shared physical resource authority owns real memory/device budgets;
- device workers own actual asynchronous compute/events but cannot invent logical commits;
- serving, model loading, scheduling and resource ownership are not cloned for each specialization.

The initial vision and full early architecture remain in:

- VISION_SCOPE_AND_PRINCIPLES.md
- ../plans/ZEROGRAVITY_MASTER_PLAN_2026-10-10.md
- the current public Engineering Specification and Whitepaper under ../../docs/
- current Fabric vNext plans for the next structural transition.

## 4. Engineering philosophy and non-negotiable invariants

Priority order:

1. semantic/state correctness;
2. source and quality fidelity;
3. safety, ownership and crash behavior;
4. compatibility and user clarity;
5. stable latency/throughput;
6. peak specialized speed;
7. experimental complexity only when evidence earns it.

Rust design preference:

- clean, robust, intuitive ownership;
- shared lowest layer when it genuinely benefits multiple callers;
- zero-cost abstractions where natural;
- hot-path optimization where simple and maintainable;
- prefer a clean design over convoluted optimization when the two conflict;
- no second runtime/allocator/state publisher hidden behind “optimization.”

Hard invariants:

- exact source/model identity remains explicit;
- model metadata or a registry slug never grants execution support;
- a finite captured testcase is qualification evidence, not general runtime authority;
- prompt IDs, prompt lengths, ordinals, packet contents, capture hashes or nearest-case tables must not select ordinary production math;
- one logical StateRoot publication authority;
- one process-shared physical ResourceArbiter;
- actual completion/event/fence governs release;
- completion-unknown work is quarantined, not silently recycled;
- source readers/owners survive until actual final consumers retire;
- cancellation cannot publish partial state;
- unsupported shape/source/backend is explicit Unsupported/refusal, never silent fallback to another implementation;
- a passing cleanup or final case cannot relabel an earlier failure;
- compile/source-only success does not imply physical inference support;
- performance follows correctness;
- no universal numeric epsilon; each operator/model/profile owns its declared contract.

The shared correctness policy is:

    ../plans/numerics-and-context/CORRECTNESS_POLICY.md

It deliberately combines the useful ideas from vLLM and llama.cpp without blindly copying either:

- exact semantics/source/state checks;
- independent operator numeric checks;
- model-level output/distribution/quality checks;
- determinism/batch-invariance as separate claims.

## 5. Important rejected patterns

Do not reintroduce:

- one model-specific server per family;
- duplicate resource wallets;
- duplicate StateRoot publishers;
- generic production dispatch based on testcase/capture identity;
- arbitrary Hugging Face remote Python execution;
- metadata-only “support” claims;
- universal tolerance constants;
- hidden precision changes to hit benchmark numbers;
- KV-only prefix reuse for models with additional recurrent/SSM/GDN state;
- benchmark claims before exact source/output/ownership gates;
- a giant permanent graph interpreter in the token hot path;
- per-kernel trait/object hierarchies with no demonstrated need;
- durable agent side branches/worktrees as coordination state;
- monolithic generated handoff archives or documentation SHA catalogues.

## 6. Architecture evolution from the beginning

### 10–12 Oct: initial architecture and v4 consolidation

The early design converged on:

- semantic/effect-aware model programs;
- portable CPU/reference execution;
- optimized device regions;
- extreme model/hardware specialization as optional qualified providers;
- TaskProgram/TaskDriver for non-autoregressive tasks;
- explicit model/session/resource/device boundaries;
- multi-model routing and OpenAI-compatible serving;
- local-first model store/cache/acquisition;
- future SSD and distributed placement.

The v4 minimality principle reduced durable representations to the semantic program and prepared executable/bundle; lower-level compiler/device views are transient.

### 13–16 Oct: source formats, quantization, model families and GPU ownership

Work established:

- GGUF/safetensors/sharded source handling;
- Qwen3.5 and Gemma source/model contracts;
- CPU source-exact math and quant paths;
- GPU backend separation for CUDA/HIP/Vulkan;
- physical ResourceArbiter and source/device ownership;
- exact source/state/fixture policies;
- XDG cache/source behavior;
- early model acquisition/cache foundations;
- multimodal/task architecture research.

Qwen .8B and 2B became the immediate correctness vehicles.

### 17–22 Oct: full-model numerics, GPU native paths and serving foundations

Work expanded through:

- full 24-layer Qwen execution/reference comparisons;
- all 42 Qwen state domains;
- full 248,320 logits;
- RMSNorm/GDN/attention/FFN numerical investigation;
- retained source/window ownership;
- GPU native image/launch qualification;
- cancellation/fault/quarantine;
- serving queues, streaming and shutdown;
- OpenAI-style API subset;
- preliminary matched vLLM comparisons.

Historical numeric differences remain evidence, not current acceptance.

### 23–26 Oct: integration, routing, acquisition and broader backend work

Work covered:

- device routing, including exact /dev selectors where appropriate;
- CLI/config/source precedence;
- acquisition providers and offline policies;
- cache transactions/reclamation;
- pipeline/config validation;
- Qwen27 and Gemma finite qualification lanes;
- HIP/Vulkan qualification work;
- serving concurrency/queue scaling;
- shared state/resource ownership audits;
- repeated fresh bug/regression reviews.

### 27 Oct: Fabric vNext design

The user requested a clean future architecture where the whole target structure becomes visible early in code, then current implementation migrates into it.

The first structural milestone is intended to introduce model-neutral engine and reusable serving crates, visible typed contracts and explicit unsupported stubs before deep migration.

That was the design-only checkpoint. Signed`7396984` now implements S0; functional migration remains open.

### 28 Oct: ordinary source-derived Qwen CUDA integration

Current source now has ordinary Qwen3.5 .8B/2B CUDA execution assembled under the existing resident owner:

- one call-free ForwardPlan;
- authenticated source owner;
- request-local materialization;
- cuBLAS projection path;
- source-exact attention;
- arbitrary-row parameterized GDN;
- F32/F32/BF16 two-root state;
- one queue/flight/completion/quarantine path;
- existing StateRoot publication authority.

It remains qualification-only. No production ordinary CUDA route is enabled.

### 28–29 Oct: evidence hardening, cache human aliases and repository normalization

Recent work also:

- made cache by-name aliases crash-consistent and explicitly non-authoritative;
- simplified ordinary parity evidence so the checked-in capture procedure defines the reference and only the evidence content digest crosses processes;
- removed code/doc SHA catalogues from parity evidence;
- made compare-existing independent of the historical native bundle;
- retired machine-local capture-only tests and dead run-artifact references;
- reduced topology to main-only;
- pruned superseded handoff archives;
- normalized current planning provenance;
- reviewed and fixed environment-sensitive PID1 serving signal tests;
- added Qwen27 layer9 source-contract coverage to the shared check gate;
- refreshed vNext handoff method to use this canonical pointer + maintained files + Git history.

## 7. Current repository/runtime structure

Current workspace owners remain:

- zerogravity-core
- zerogravity-artifacts
- zerogravity-cpu
- zerogravity-gpu
- zerogravity-model-qwen35
- zerogravity-model-gemma4
- zerogravity
- zerogravity-cli
- zerogravity-gpu-qualify
- zerogravity-engine (cold model-neutral shell; F0 migration pending)
- zerogravity-serving (cold embedded API; F2 protocol migration pending)

Signed`7396984` adds the two Fabric crates without enabling new model routes.

Current logical ownership:

- core: semantic/state/resource/device/pipeline contracts;
- artifacts: source formats, immutable sources, cache, acquisition;
- model crates: family math/source/state/tokenizer semantics;
- CPU/GPU: physical execution providers;
- zerogravity facade: current public composition surface;
- CLI: still owns more reusable application/serving policy than final vNext design wants;
- serving currently lives under CLI and is intended to move to a reusable library during vNext.

## 8. Current Qwen ordinary CUDA status

### What is implemented

Ordinary .8B/2B source-derived CUDA is integrated under the resident execution owner.

Source-free tests establish:

- ordinary cancellation checks before allocation, before flight, during source execution, after launch/completion and before commit;
- armed-flight cancellation enters existing unknown-completion/quarantine behavior;
- explicit provider/pool shutdown;
- unique model ownership before shutdown;
- zero resident bytes/events before namespace close;
- source/provider ownership survives until retirement;
- no alternate root publication path.

The ordinary production route remains intentionally refused until real-current-source qualification closes.

### Exact approved .8B source

Repository/model:

    Qwen/Qwen3.5-0.8B

Revision:

    2fc06364715b967f1860aea9cf38778875588b17

Visible approved source path:

    /home/pvl/.cache/abird/zerogravity/external-baselines/models/Qwen--Qwen3.5-0.8B/2fc06364715b967f1860aea9cf38778875588b17

Weight shard:

    model.safetensors-00001-of-00001.safetensors
    bytes: 1,746,942,600
    SHA256: 04b1c301231dd422b8860db31311ab2721511346a32cb1e079c4c4e5f1fe4696

The tracked source manifest is:

    crates/zerogravity/src/qwen35_cuda/source08.json

### Recovered reference and original blocker — 29 Oct

The old assembled bundle was purged, but its exact `.8B` N1 inputs are recovered
offline under XDG state. Retained canonical images plus two whole SM80
FlashAttention modules extracted from the pinned local reference image and the
original cuBLAS providers match current Rust identities. All 11 model members
verify; maintained bundle validation and isolated structural preflight pass.

Fresh phase-1 cached47→50 reference capture passes after selecting the existing
8 GiB reference-fixture host profile, with all 1,008 source windows retired.
The original 4 GiB refusal stays failed. The independently emitted evidence
content digest is retained in the owning numerics plan and external evidence.

Fresh ordinary comparison fails before model math: pinned `cublasCreate_v2`
returns status 1 after the owned allocator denies a private `cuGetExportTable`
call. Zero allocation attempts/charges and bounded resolver observations identify
this as a provider ownership blocker, not missing model/bundle bytes. Do not
forward opaque tables or preload cuBLAS around admission to make it run.

The current evidence and remaining work live in
`../plans/numerics-and-context/TODO.md`. `.8B` cached47→50 is exact; broader
lifecycle/context and production routing remain unqualified. Owner order is CUDA correctness first,
then ROCm on `pvl-x2`, with no subagents.

Maintained tooling:

- tools/qwen35-cuda-reference/
- tools/qwen35-cuda-bundle/
- scripts/qualify-qwen35-ordinary-cached47-50.sh

The Python discovery/packed-capture protocol still requires explicit retained
external capture sources. Exact offline bundle recovery is documented in the
bundle helper README; the Rust two-process parity wrapper uses that recovered
bundle. These are qualification inputs, not production authority.

### Owned execution checkpoint — `2983bb7`

Owner approved replacing ordinary opaque BLAS with shared driver-only GEMM.
The allocation-free tensor-core/SIMT primitive passes 180 synthetic cases and
eight authenticated prefill/cached projections exactly, including guards and
retirement. Typed SSA roles choose explicit reductions without captured node or
prompt lookup. Original compiler-context reservations and physical flights remain
the owners; pinned reference/source/resource limits are unchanged. Ordinary
comparison no longer requires cuBLAS paths.

Corrected the rotary admission omission (+280 bytes), GDN partition binding and
premature BF16 rounding of residual RMS sums. Row RMS retains Torch vector-four
FP32 boundaries; Q/K RMS instead follows the fused Triton adjacent-pair/warp
reduction. CUDA rotary now uses FP32 FMA and final BF16 rounding; the CPU-target
BF16 product contract is unchanged. All211 GPU host tests (43 ignored) and an
independent host-FMA rotary comparison at14 positions through32767 pass.

Owned tile-64 CUDA attention now preserves FlashAttention's BF16 unnormalized
probabilities, FTZ FP32 online softmax and tensor-core QK/PV. Runtime geometry,
not captured-case routing; no new device scratch or allocation authority. Both
independently authenticated prefill/cached attention captures match exactly.
Nine analytic causal/multitile cases through a32K active prefix and three
nonfinite refusals pass with guards and retirement. They are not long-context
model parity.

Fresh `.8B` cached47→48→49→50 passes exactly: all248,320 logits, all42 state
domains and natural feedback at every step, with normal model shutdown. The
reference bytes/digest are unchanged. The earlier229,459-logit /4,359,077-state
failure (max0.28125) remains failed historical evidence.

Replaced obsolete row/residual CPU-profile CUDA oracles with independent staged
CUDA/Torch comparisons:16 cases across widths1024/2048, rows1/2/17/64 and residual
modes pass exactly, including non-BF16 residual sums, guards and retirement.
Real post-launch cancellation and abandoned-future probes pass:396 original
handles quarantined, prior bits/tails intact, unchanged root, peer refusal and
retained charges after external drops. At that historical pre-streaming checkpoint,
normal lifecycle passed all13 admitted capacities (`.8B` through32K,2B through16K),
each at48 active tokens;2B32K refused at pure preflight. This refusal is superseded by
the21Nov current-source streamed-admission note above. Long-active-context/2B numerical
parity remains open. Final
isolated repository gate passes2,129 Rust executions/0fail/83ignored;14 Python
suites and ancillary gates pass. CUDA/native host675/0fail/54ignored and strict
CUDA/HIP/Vulkan feature checks pass.
The2B approved snapshot exists in the canonical cache; its extra reference image
must be recovered independently, not replaced by the `.8B` inventory.

Owned execution landed in signed main commit `2983bb7`; nothing is pushed and
production remains disabled. The extra exact2B reference image was not found
among1,252 extracted SM80 modules; retained original capture inputs/long-context
comparisons are also missing from the current qualification root. Do not replace
or repin them. ROCm is now active on `pvl-x2` gfx1151/96GiB, independently of
those CUDA input gaps. Current source is in bounded run-local execution scratch,
not another Git worktree. The exact31-member Qwen3.8-27B snapshot was imported and
verified in canonical XDG cache from existing inputs; no download was needed.
Fresh independent natural full64/fullV/all112-state reference capture validates
both seed11→220→16 cohorts and source/process/FD closure. Capture/validator
protocol repairs have14 Python controls;26 Rust comparator/admission controls and
narrow/shared HIP strict checks pass. Signed `99f0232` fixes rank3 Conv4 preflight
through the shared exact binding/30 source-window tests. SDK-matched child-only
loader resolves build-libc versus GLIBC2.44 runtime refusal without replacing
SDKs or changing shell libc. Actual teacher full4 and healthy natural9-call reports
match full64/fullV/all112-state exactly; all six repaired natural modes pass
semantic validation. Known reader paths retire121 modules; release faults retain
121 modules/original service/source/1,477 handles. Teacher unknown and independent
Pending/Snapshot final-owner scopes also pass. Earlier failed scopes stay failed.
The old healthy controller exit record was lost; that controller scope stays
incomplete. A fresh durable healthy-only replay passes with exit0 and no remaining
processes. Signed `825ade8` lands the bounded diagnostic repairs. Source gate
passes2,132 Rust/17 Python suites with explicit skips. Genuine ordinary contexts, small-model HIP, C1..C64
and performance remain open; production stays disabled. See the owning numerics
TODO for exact scopes and continuation.

Do not recreate deleted `.agents/runs` paths or treat old run names as provenance.

## 9. Ordinary cached47→50 parity protocol

Current parity design was deliberately simplified.

The checked-in reference capture procedure defines how phase 1 is produced.

Evidence schema v3 contains:

- source coordinates;
- prompt;
- 47→48→49→50 observations;
- full logits;
- all state domains;
- selected IDs/max-bit receipts;
- teardown/close receipt;
- an embedded corruption digest.

Cross-process trust:

- phase 1 emits the evidence content SHA256 in stdout;
- normal wrapper execution carries that digest in memory to phase 2;
- compare-existing requires the independently supplied trusted digest;
- the embedded digest is corruption detection, not a self-authenticating implementation identity.

Removed intentionally:

- reference implementation SHA graph;
- source-code SHA catalogues;
- migration pin catalogues;
- stage-receipt authority;
- code-hash-based evidence identity.

Phase 2 does not need the historical native bundle; it needs:

- captured evidence;
- trusted phase-1 evidence digest;
- current approved model source;
- ordinary CUDA providers.

Normal phase 1 still needs the qualified native/reference bundle.

The wrapper is:

    scripts/qualify-qwen35-ordinary-cached47-50.sh

Modes:

- normal: structural checks, ephemeral legacy staging if necessary, reference capture, digest handoff, ordinary compare;
- preflight-only: structural/environment checks only, no large copy;
- compare-existing: no historical bundle required.

## 10. Exact ordinary CUDA context admission

Under the unchanged 4608 MiB provider / 5120 MiB physical limits, totals include
cold model, two roots and retained conservative admission headroom. The owned
GEMM allocates no scratch; these are not claims of physical scratch usage.
The 30-Oct correction includes the previously omitted 280-byte rotary table.

### .8B

- 512: 1,652,916,585 B
- 1K: 1,667,662,185 B
- 2K: 1,697,153,385 B
- 4K: 1,756,135,785 B
- 8K: 1,874,100,585 B
- 16K: 2,110,030,185 B
- 32K: 2,581,889,385 B

All fit admission.

### 2B

- 512: 3,911,846,249 B
- 1K: 3,926,591,849 B
- 2K: 3,956,083,049 B
- 4K: 4,015,065,449 B
- 8K: 4,133,030,249 B
- 16K: 4,368,959,849 B
- 32K: 4,840,819,049 B

512 through 16K fit.

Historically, before streamed-weight admission became the current shared-source policy,
32K exceeded 4,831,838,208 B by 8,980,841 B and was expected to refuse before physical
allocation. The21Nov public2B32K load supersedes that refusal as current authority.

These are admission/fit results only. Every admitted context still needs real-model correctness, continuation, fault and ownership qualification.

## 11. Deferred Qwen mission

Fabric transition is now first. Do not rerun already accepted cached47→50,
bounded64 HIP, finite27B, lifetime or GDN scopes merely to restart the campaign.
After migration, resume genuine multi-row0.8B/2B HIP and call-free lowering,
then general27B/populated fit contexts/C1..C64/matched vLLM. Preserve the separate
CUDA2B/wider reference and ROCm baseline authority gaps. Exact current scope:
`../plans/fabric-vnext-transition/CURRENT_SOURCE_RECONCILIATION.md` and owning
model-execution/numerics/serving TODOs. Production remains disabled.

## 12. Serving and router state

The project already has:

- bounded HTTP handling;
- health/models;
- OpenAI-style text/chat completions subset;
- JSON/SSE streaming;
- request/output usage accounting;
- bounded pending queue;
- cancellation/disconnect propagation;
- CPU route;
- finite/native CUDA routes where qualified;
- source/model route validation before side effects;
- graceful worker join/shutdown;
- source-free C16/C32/C64 queue/population coverage.

The source-free production router queue is not the blocker for ordinary Qwen CUDA; the model route is.

Recent fresh review found the process-signal fixture incorrectly assumed every environment could nest a bubblewrap PID1 namespace. The shared fixture now:

- always exercises ordinary SIGTERM/SIGINT process shutdown;
- preflights PID1 namespace capability;
- skips PID1 rows only for explicit namespace/permission unavailability;
- hard-fails unexpected bubblewrap/preflight errors;
- reports skip reason explicitly;
- Rust asserts ordinary rows cannot skip and PID1 rows pass when capability exists.

Current serve all-targets result after the fix:

- 89 passed;
- 0 failed;
- 4 intentionally ignored source/hardware fixtures.

## 13. Cache/acquisition current state

Canonical cache data is authoritative; human by-name naming is not.

Human projection design:

    models/
      snapshots/<digest>/...
      refs/...
      .by-name-initialized
      .by-name-state/
      by-name/

Key properties:

- runtime/model selection never reads by-name state;
- by-name projection is disposable;
- explicit aliases and automatic names live in durable human state;
- dirty modes recover projection/reverse/reconciliation;
- publication can leave a pending human update and later self-heal;
- missing state after prior initialization refuses rather than silently losing aliases;
- reset requires explicit alias discard;
- mount-boundary checks use mount identity, not just st_dev;
- process-exit/fault matrices exercise human metadata and reclamation recovery;
- canonical snapshot/ref availability is not downgraded by human metadata failure.

Acquisition/cache supports bounded, source-aware providers and offline exact-cache behavior for the implemented providers, including Hugging Face/GitHub/Xet/OCI/S3/GCS/Azure/ModelScope/object flows where the relevant feature is enabled.

No provider metadata or alias may upgrade source rights.

## 14. Device/CLI/config state

Current CLI/device work includes:

- explicit CPU/device routing;
- exact PCI/device identity;
- validated direct /dev selectors for supported DRM/NVIDIA nodes;
- NVIDIA minor is not treated as CUDA ordinal;
- aliases converge on one observed device identity;
- source/alias/device policy precedes execution side effects;
- bounded YAML pipeline/config checking;
- strict config/env/argv precedence;
- dry-run and refusal behavior.

Pipeline configuration is more complete than pipeline execution. A valid YAML graph does not imply the task pipeline is implemented.

## 15. Memory/SSD/state direction

Implemented foundations include:

- shared physical resource accounting;
- immutable source extents;
- bounded source windows;
- source descriptor retention;
- transfer staging admission;
- retained GPU readers through actual completion;
- state/root/cancellation/quarantine semantics.

Open work remains for:

- production paged complete-domain state;
- complete-domain prefix reuse/COW;
- product snapshot persistence/restore;
- real SSD model streaming;
- expert/source IO QoS;
- eviction under actual GPU completion;
- broader multi-model residency.

Do not call KV-only reuse a hybrid-state prefix hit when recurrent/GDN/SSM/other domains are missing.

## 16. Model-family status discipline

Qwen .8B/2B are the primary current ordinary CUDA correctness path.

Qwen27, Gemma4 and other families have substantial source/reference/finite backend work, but do not inherit .8B support automatically.

Keep these distinctions:

- source parser support;
- family mathematical support;
- CPU reference support;
- a finite backend qualification;
- a production route;
- long-context support;
- serving support;
- performance qualification.

They are separate claims.

Future/target families and tasks discussed in the design include broader Qwen, Gemma4, Llama/Mistral, GLM, DeepSeek, image, ASR, TTS, Whisper/Canary/Parakeet-style audio systems and multimodal pipelines. Treat these as roadmap targets unless current source/tests explicitly implement and qualify them.

## 17. Fabric vNext — historical design that led to the completed transition

Historical state when this section was written: S0 was signed in `7396984` and
S1–S3/F0–F4/Q0–Q2 were still open. The transition is now complete; use the
current resume section and closed `fabric-vnext-transition` plan for present status.

User intent:

- make the final architecture visible in code immediately;
- break pre-v1 internal/public APIs when that produces a cleaner design;
- move current working code into the new ownership model instead of carrying compatibility layers forever;
- allow private compile-safe placeholders/Unsupported where implementation is not yet migrated;
- no reachable todo panic on an advertised supported path;
- preserve all current exact source/numeric/state/resource safety;
- qualify the assembled transition once, then fix regressions in small focused commits.

Target dependency/ownership idea:

- zerogravity-core: semantic/numeric/state/provider/physical contracts;
- zerogravity-artifacts: containers/codecs/source/acquisition/store;
- model crates: family source/math/state/tokenization/specialization;
- CPU/GPU: physical providers;
- new zerogravity-engine: model/backend-neutral runtime, scheduling, sessions, residency;
- new zerogravity-serving: reusable server/protocol layer depending on engine/core, not facade/CLI;
- zerogravity facade: composition root/public SDK;
- CLI: arguments/presentation/signals only.

The facade is the composition root. Engine must not depend on Qwen/Gemma or physical CUDA/CPU implementations. Serving must not depend on facade/CLI. CLI should call the public SDK.

### Planned transition sequence

S0 — full structural skeleton first

- create engine and serving crates;
- expose top-level target ownership/contracts;
- compile;
- explicit Unsupported for pending operations;
- architecture checker updated;
- no heavyweight model qualification yet.

S1 — exact source/quant/encoding migration.

S2 — family math + GPU provider/qualification separation.

S3 — canonical state/root/resource publication and ownership.

F0 — model-neutral engine.

F1 — complete public Rust SDK.

F2 — reusable serving library.

F3 — thin CLI and deletion of shadow runtime copies.

F4 — feature parity matrix across all currently real behavior/refusals.

Q0 — one assembled broad qualification.

Q1 — focused bugfix queue at lowest shared layers.

Q2 — final full gate, physical/source qualification and capability matrix.

Only after transition:

- broader models;
- vLLM-style continuous batching/paged state;
- larger model residency;
- SSD execution;
- Gemma broad native routes;
- additional quant formats;
- sparse/MoE/MLA/Engram/MTP;
- multimodal ASR/TTS/image/video;
- NPU/remote distributed placement;
- calibrated autotuning/benchmarking.

The vNext documents no longer require or promise a monolithic archive or documentation SHA index. This handoff is the cross-window pointer. Git history is the archive.