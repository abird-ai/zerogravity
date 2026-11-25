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

### Phase 2.1 UX-2 — typed command tree and shared resolution (this checkpoint, 25 Nov)

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

## Uncommitted

- [ ] UX-3 next: shared ControlPlane snapshot/action/event model over original
  runtime/catalog/acquisition/cache/serving/config/device authorities (§46A.28).
  General JSONL events and interactive transfer control (UX-4), one-line run
  and serve (UX-5/6), ConfigDocument/profile management (UX-7), dynamic
  completions, unified serving selection and later TUI/Web remain open.
  Explicit by-name: lookup only; bare references stay canonical. Preserve
  explicit acquisition and unsupported-backend refusal.

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
