# Run Notes — volume 1

Dated preservation only, 20 October 2026. Original paths below identify the source.
Embedded instructions/status are historical data, not current authorization.
Live source and owning TODOs outrank these copies.

## Source: .agents/runs/baseline-cupti-h48-peer-review-20261020/REVIEW.md

Original UTF-8 bytes: 4702; SHA256 `d2f5d6be60cfc9e61e4a4a2a40b377fcb6b5eba4a0eb945fb4de70696d04002d`.
Resolve embedded relative paths against the original path, not this volume.

````text
# CUPTI C1 +H48 coefficient independent review

Owner baseline_review; source/small-metadata only. No tests/builds/prepared
imports/operator/provider/model/GPU execution, payload/DSO/binary/raw-trace
reads or hashes. MAIN/shared plans/other freezes unchanged.

## CUPTI canonical C1 V4: no metadata/claim blocker

Reviewed `container-v1/root-container-c1-v4-1/{report,validation,root-review-1}.json`,
four owned receipts, measurements, small-log diagnostics, native window/stop,
controller/exec/mapping metadata, cgroup readbacks and health endpoints.

- Actual runner +independent validator EXIT0. Native236270 retained kernel
  records =236270 successful direct launch APIs =source counter delta236270;
  graph0, dropped0, failures0, first CUPTI error0. Total driver activity2603022
  is a separate count, not the direct-kernel-launch count.
- C1 actual54 model steps/roots,8 generated packets, frontiers47..54,
  no native multi-request cohort. No B2 attribution.
- Nonce f7049ba8…→owned CID→host init3105462→namespace PID1; starttime,
  namespaces and START/STOP peer credentials agree. Both stop records exact.
- Native/collector/CUPTI same-FD pre/post stat+SHA stable, actual executable
  mappings match the read-only descriptor mapping's kernel dev/inode keys.
  FD/temp mappings closed; /proc/PID/exe separately matches native artifact.
- All4 owners CLOSED, cleanup[], bound group gone/reaped, nonzero signals
  before reap, container absent, EOF/untruncated logs,3 quiet samples afterward.
  Every sampled NVIDIA compute PID is in owned numeric container PID subset.
- Actual mounted cgroup readbacks:12GiB RAM, swap0, quota4CPUs, affinity0..3.
  Device samples are point observations; exact GPU/pool allocation admission
  remains false. Source bounds4GiB/provider+5GiB device are not stock hard caps.
- One offered/completed request and8 tokens per owner. Plain/observed IDs,
  finish/usage/output hashes agree within each provider; providers differ.
- Stock9318 kernel activity records/8810 host-launch events are recorded only:
  losslessness, all-process/init coverage and workload binding remain false.
  No matched complete counts, timing, numerical equality or goodput claim.

ROOT's raw-stream validator was not rerun;439MB raw activity stayed unopened.

## H48 coefficient freeze05095281: one replay blocker

Read complete3-path production/test delta +replay helper, transport,
source/binary attestation, validator, record-only signal guard and owned cleanup.
All28 live/frozen source/metadata files match manifest05095281. Factory,
owned_lifetime and TERM/INT guard are byte-exact accepted parents.

**Blocker:** `replay/main.rs:521` uses layer1 as a refused "wrong layer".
Authentic original27B config has interval4: layers0/1/2 are LinearAttention,
layer3 FullAttention. The private input binds source/profile, not a layer;
`coefficient_roles` legitimately accepts layer1. This call can execute, making
the asserted negative fail before the actual target replay. Synthetic fixture
has layer1 FullAttention and hides the mismatch. Additive harness-only repair:
use authentic FullAttention layer3; preserve production math/guards and freeze.

No further source blocker found:

- Only unevaluated A_log codec guard expands to genuine BF16 H48/96B;
  F32 remains heads×4, BF16 H16 refused. Exact role/name/source/profile,
  raw dt codec and issuer gates remain. A_log never enters preparation math.
- CPU sigmoid/add/softplus arithmetic unchanged: one H48 call, vector32+
  scalar16 beta, raw BF16 dt explicitly widened. Exp/g/decay remain absent.
- Same two typed state-free nodes, same HostMath/ForwardPlan and sealed851-role
  owner. No state/root/frontier/logit/full-model/GPU permit.
- Both V3 teachers11/353 metadata exactly matches7 selected arrays/call and
  source617/619 dispatches. True initial→cached prior link retained. Native
  targets are288 output +192 retained input words; codec widening preserves
  signed bits. Raw A_log/dt provenance is separate from consumed math.
- Steady80MiB/4 reservations, two2MiB admitted input/output groups→live84,
  caller-drop retained76, final0. Both teachers have real one-byte-short
  pressure/pre-cancel/foreign issuer negatives. No total-host-RAM claim.
- Existing unreaped-leader TERM/grace/KILL/reap/EOF path reused. Signals record
  only through Popen handoff and cleanup; receipts are written afterward.
- 521-parent +3-path overlay is explicitly522 source inputs in the isolated
  tree. Source attestation is not compiler/binary proof; ROOT build provenance,
  native execution and independent full raw validation remain required.

ROOT and author received the blocker before composition. Originals untouched.

````

## Source: .agents/runs/baseline-cupti-h48-peer-review-20261020/TODO.md

Original UTF-8 bytes: 561; SHA256 `57cf0155e64bd1a1c992c17e4f8b8558446b93fec9a9fbf04edf5feda1a4d6b4`.
Resolve embedded relative paths against the original path, not this volume.

````text
# Independent CUPTI C1 +H48 source review

Parent serving-and-speculation/model-execution; owner baseline_review.

- [x] Review actual CUPTI canonical C1 V4 summary/ownership/mapping/limits.
- [x] Preserve stock completeness/timing/numerical admission limitations.
- [x] Review complete H48 coefficient05095281 proposal/replay; report layer1
  wrong-layer negative blocker before ROOT composition.
- [ ] ROOT additive harness repair +actual isolated source/build/replay/validator.

No MAIN/shared plan or other frozen bytes changed; no execution/payload reads.

````

## Source: .agents/runs/gpu-serving-batching-2026-10-19/TODO.md

Original UTF-8 bytes: 1528; SHA256 `6a05b27042045a3a92670b85cb62c7f64179d8de6a6f96d7b3afa8b6f8c9cbe7`.
Resolve embedded relative paths against the original path, not this volume.

````text
# GPU OpenAI serving and batch-wave foundation

Base: 35b9093
Plan: [serving](../../plans/serving-and-speculation/TODO.md)

- [x] Resident GPU routes, source aliases, independent sessions,
  device greedy, shared CPU/GPU OpenAI SSE output and cancellation.
- [x] Finite source/phase/frontier wave planner, truthful fusion=0.
- [x] CUDA preflight with runtime and CCCL headers, Nix symlink
  helper, no download or driver change.
- [x] Physical RTX3060 CUDA one and two concurrent requests,
  separate sessions, real GPU completions, cancellation+recovery.
- [x] Actual source-real SSE benchmark: 2/2, 4 tokens, 20.02s
  debug wall; p95 TTFT 19.78s (no vLLM equivalence).
- [x] Benchmark harness selftest and repeatable YAML GPU config.
- [x] Strict CPU, CUDA and Vulkan serving Clippy -D warnings.
- [x] Complete isolated scripts/check.sh passed 1,147 Rust tests,
  0 failed, 69 ignored across 241 suites, installer and CPU
  comparator tests, 18 Python fixtures, benchmark selftest,
  Nix and shell formatting.
- [x] Original approved Qwen3.5 .8B/2B direct-source and
  HF-import release parity both passed: all 248320 logits
  and 42 StateRoot domains, independent leases/sessions.
- [!] Nix filtered source-integrity offline derivation blocked:
  pinned nixpkgs input uncached; TLS trust bundle inaccessible
  to this connector. Nix formatting passed independently.
- [ ] Signed scoped local commit; no push.
- [ ] True batch-native fused GPU plan/kernels, paged KV and
  pinned vLLM goodput acceptance remain unimplemented.

````

## Source: .agents/runs/model-acquisition-c3-resume/TODO.md

Original UTF-8 bytes: 2712; SHA256 `37318cdd964eb48ec19353ebb0025a9bcf1d8c9c8ce12edaa0817f134d9624f1`.
Resolve embedded relative paths against the original path, not this volume.

````text
# TODO — resumed C3 filters integration

Status: active; awaiting final repository gates and signed acceptance
Provenance: f197c9cf4a44e06a3c1cf3c532855545bed945fa
Main plan: ../../plans/model-acquisition/TODO.md
Owner: current C3 integration on MAIN; preserve old unfinished SIDEs.

## Verified in this run

- [x] Revalidated paused C3 matcher/oracle against MAIN and accepted C2.
- [x] Implemented additive source-bound Unicode FnmatchCaseV1 selection
  with bounded 16Mi work, 512KiB scratch, 16MiB extra original admission,
  required-index veto, optional omission and same source/publisher.
- [x] Added HF/GitHub/native-HF single pin and shared CLI --include/--exclude
  and --quant/--exclude, strict digest-only offline exact subset equality.
- [x] Copied only 11 reviewed Rust files onto MAIN with fresh mtimes;
  375 other SIDE crate files matched accepted MAIN at review time.
- [x] Frozen original CPython-derived 73 literals and 33345 pairs pass;
  matcher/adversarial units 4/4; acquisition_filters 11/11;
  enabled CLI filter suite 4/4 and disabled CLI filter 1/1.
- [x] New synthetic original executor/cache publisher test with source,
  cache ref, managed content verification and release of host reservations
  passes on HF and GitHub.
- [x] Canonical private user/network isolation and guarded namespace-only
  loopback: artifact library 160, candidate 11, quant 11 and filter 11 pass.
  CLI candidate 5, quant 5 and filter 4 pass.
- [x] Production/diagnostic strict artifact and CLI Clippy pass.
- [x] Record the initial unconfigured private-loopback HTTP/TLS fixture
  failures (18/157) and the subsequent 160/160 green after invoking the
  original scripts/check.sh NS_GET_USERNS/uid_map guarded setup. No host
  network change. Two ordinary Clippy corrections and a missing test-fixture
  LFS announcement were repaired and rerun; production protocol unchanged.

## Still open

- [ ] Revalidate frozen merged source/tests and documentation.
- [ ] Complete frozen isolated scripts/check.sh gate; the connector has a
  30-second per-command limit, so accurately report any interrupted run.
- [ ] Actual offline Nix source-integrity with all three new Rust inputs.
- [ ] Both approved original release cache_prepare routes separately.
- [ ] Check local links, protected README, Cargo.lock/manifests, staged paths.
- [ ] Sign locally verified implementation/tests/plan with exact paths; no
  push/tag/publication; roll completed items under commit in next update.
- [ ] Reconcile C4 index draft on this accepted C3 base only after C3 gates.

Do not touch held gfx90c, devices, accounts, live caches, model/SDK sources,
mounts, privileges, remote publication or unrelated worktree draft files.

````

## Source: .agents/runs/model-acquisition-filters/TODO.md

Original UTF-8 bytes: 17141; SHA256 `7ffe188e452887b2d67baa18a84dc934eb221679bbee2c984ff1450a0a81969d`.
Resolve embedded relative paths against the original path, not this volume.

````text
# TODO — additive filters and exclusions cold design

Status: active approved SIDE source implementation; all Rust checks UNRUN.
Provenance: C1 frozen9path candidate inventory + C2 source-only quant11paths at46cf5a5 overlay; MAIN conditional/candidate gates coordinated separately.
Main plan: [model acquisition](../../plans/model-acquisition/TODO.md).
Owned scope: approved11Rustpaths only in /home/pvl/spaces/abird/src/zerogravity-filters-work (4db5efb +exact C1/C2 overlays) plus this own MAIN run. No MAIN source/Cargo/Git/canonical writes, payload/provider/account/model/SDK/cache/device actions.

## Uncommitted

- [x] Read original byte-component */? matcher, closure and current candidate/quant initial-choice seams without treating drafting code as frozen proof.
- [x] Verify primary Python/HF/vLLM filename matching requirements and choose exact additive full-path UTF8 grammar, brackets/ranges/negation/malformed literals/slash/case.
- [x] Define deterministic nonexponential bounded work/memory/cancel, includeunion->exclude->familyambiguity and required-versus-optional closure exclusion policy.
- [x] Propose additive API/CLI, explicit quant/native composition and meaningful tests, preserving legacy Files grammar and sole provider inventory/budget/publisher.

## Primary requirements and chosen grammar

[Python3.14 fnmatch documentation](https://docs.python.org/3.14/library/fnmatch.html) establishes full-string, case-sensitive fnmatchcase semantics: separators and leadingperiods have no special status; ? consumes one character, * any sequence. [HF download documentation](https://huggingface.co/docs/huggingface_hub/en/guides/download#filter-files-to-download) supports combined allow/ignore lists. These justify an additive Unicode path matcher, not changing existing component-local byte Files behavior.

Use named fixed grammar `FnmatchCaseV1` on valid UTF8 scalar strings; no locale/case/Unicode normalization. `*` includes slash; consecutive stars collapse with no specialglobstar/recursive-directory operator. Literal slash remains literal, and ? may itself consume slash. Exact repository paths remain validated separately; patterns only choose among owned pinned paths, never authorize URL/path traversal or discovery of a new source. With a subfolder, enforce its literal component-boundary scope first and match the remaining full relative suffix; never interpolate the folder text into pattern syntax (folders can themselves contain brackets). Without it, match full repository-relative paths.

[CPython v3.14.0 matcher source](https://github.com/python/cpython/blob/v3.14.0/Lib/fnmatch.py) and [its fixed tests](https://github.com/python/cpython/blob/v3.14.0/Lib/test/test_fnmatch.py) define bracket edge behavior: [seq], leading! negation, Unicodeordinal ranges; ^ is literal. Initial ] inside a class is literal. Unclosed [ is literal, then remaining pattern parses normally. Hyphens at class ends are literal; descending closed ranges contribute nothing, and negating an empty normalized set matches any scalar. Nested/set-operation punctuation has no POSIX/intersection meaning. Backslash is literal, never a general escape. Port the bounded class normalization into token/range data, not a regex runtime, and validate unusual multiple-hyphen cases against the pinned tests before claiming parity.

Patterns: nonempty <=4096UTF8bytes each; combined include+exclude count<=4096, aggregate pattern bytes plus inventory path bytes<=8MiB. Reject NUL/control inputs before HTTP and do not support bytes/lone-surrogate Python inputs; actual sourcepaths already exclude them. No requirement that every include pattern matches individually: include lists are union, and unmatched excludes are harmless. Empty final selection is an explicit refusal, never select-all or first-file fallback. Excludes-only without an explicit quantbase also refuses; callers can deliberately request --include '*'.

Pinned [vLLM source a8be1500bad93d3c459e435bf71194aa8062e49d](https://github.com/vllm-project/vllm/blob/a8be1500bad93d3c459e435bf71194aa8062e49d/vllm/model_executor/model_loader/weight_utils.py#L399) forwards ignorepatterns and may choose a first matching format. Zerogravity retains explicit selection and refuses ambiguity; it does not inherit that numeric/source fallback.

## Ordered selection and closure

1. Validate bounded request/patterns before provider activity; acquire original operation controls and matcher scratch before allocation. Pin/list once through the same complete private inventory used by C1/C2; validate completeinventory before subfolder projection, preserving accepted collision fix.
2. Match all include patterns as a union over scoped regularpaths, then subtract union of excludes. Determine family ambiguity only AFTER this subtraction; calling current select_paths first would reject before exclusion can intentionally narrow choices. Reuse original GGUF family parser/complete1..N and safetensors/index logic, not new filename heuristics.
3. Apply original model closure with private exclusion policy/reason awareness. Required config/tokenizer, complete split/shard cohort and source index cannot be excluded: fail before any forbidden payload/plan/publication. Excluded optional known sidecars (README/LICENSE/tokenizer_config/etc) are skipped; they cannot be silently re-added by generic closure. If any path is both optional and structurally required, required wins. Include patterns constrain initialchoice; they do not forbid required dependencies. Excludes are the explicit veto.
4. Excluding every file of a whole competingfamily may disambiguate deliberately; excluding only one member of a stillselected splitfamily must fail requiredclosure, not produce a partial model. Excluding an index while its shards remain cannot erase the dependency. Preserve original ambiguousindex/family/regularmode/path checks and explicit external sidecar supplier authority.
5. Final exact per-file bounds/provenance/source identity, rawSHA/EOF/LFS/progress/cancel and sole original publisher remain unchanged. Selectionpolicy is not new content authority or tensor/loader support. No filteredplan with placeholders or postpublication file deletion.

Index policy decision for root before source: safest smallslice forbids reading an explicitly excluded index body. Existing closure discovers relevant indexes by reading adjacent candidates, so an excluded adjacent candidate needed to establish membership must cause typed refusal before GET, even when later parsing might have proved it unrelated. Do not skip it and silently weaken closure. Record this conservative excluded-candidate-index boundary explicitly; broader membership discovery can be separately qualified if useful. No user permission gate or new source authority is implied.

Private code seam: existing require(path) gets Required reason+policy; optional additions use add_if_present_allowed(path). Old Files/quant/component/default paths use AllowAll policy byte-for-byte behavior. HF's original closure and shared GH/ModelScope closure have differences (including legacy .bin roles), so do not unify all implementations under this feature merely to share excludes. Share only the new pure matcher and small exclusion decision; preserve each old owner and provider metadata callbacks.

## Bounded matcher and admission

- Compile one pattern at a time to Literal/Any/Star/Class tokens; no global cache of compiled patterns, no regex crate/runtime and no recursive backtracking. Two DP/NFA state rows scan Unicode scalar filename input with star epsilon transitions. Work polynomial in patternunits*pathscalars, with class-range comparisons charged too; parse/normalize including malformed bracket lookahead must consume work before proceeding.
- Fixed cumulative16Mi state/character/range/compile transitions per complete operation, never reset perpattern/path/include/exclude/closure/refresh. Exhaustion returns typed Limit, not false/no-match/fallback. Cancellation checked before allocations/HTTP and at most4096transitions apart. Collapse repeated stars and bound unmatched-bracket/range normalization work; avoid naive exponential star matching.
- Scratch ceiling512KiB for one <=4096byte token/class representation, two state rows and Unicode path buffer. Precompute exclusion membership once over inventory, then reuse checked lookup during required/optional closure rather than reevaluating patterns each callback. Bounds must account actual retained path keys/collectionnodes; use compact inventory indices or separately charged bounded copies, no uncharged second full inventory.
- Recommended conservative resolve-only extra16MiB control allowance in the SAME existing TransferAdmission/arbiter owner, covering scratch +worstbounded excludedpath keys/nodes. Combine with original operation reservation before new allocations; no new resource domain/allocator/ledger and no silent reduction of existing metadata buffers. Default original80MiB metadata+16MiB remains below ordinary128MiB, but explicit lower caller limits still legitimately refuse. This number must be reconciled with actual chosen representation and tested, not treated as physical all-RAM telemetry. Drop matcher/control state before payload/cache transaction; native256MiB policy unchanged.
- All existing logicalmetadata/body/probe/file/path/manifest bounds remain cumulative across the one pin/tree/closure operation; original HTTP still owns actual retries/deadlines. Never discover_candidates()->resolve(), reset budget inside a selectionpass or materialize a second catalog/cache.

## Additive public and CLI seam

Proposed `acquisition::filters` public private-field type:

```rust
FilterRequest::new(source: CandidateRequest, includes: Vec<String>, excludes: Vec<String>) -> Result<Self>
FilterRequest::for_quant(quant: GgufQuantRequest, excludes: Vec<String>) -> Result<Self>
filters::resolve(transport, &FilterRequest, &PullLimits, &TransferAdmission, &AtomicBool) -> Result<ResolvedPullPlan>
filters::validate_cached_selection(&SnapshotLease, &FilterRequest, &PullLimits, &AtomicBool) -> Result<()>
xet::resolve_filtered(transport, &FilterRequest, &PullLimits, &XetOptions, &TransferAdmission, &AtomicBool) -> Result<ResolvedPullPlan>
```

Names are a proposal until root freezes C2 interfaces. No Selection enum/public PullRequest/PullLimits change. Private initialchoice hook grows from C2 Files/Quant to Filtered(Patterns or Quantbase,excludes), sharing inventory and finish closure; normal old Files and quant route untouched. One new patternengine implementation serves HF/GH/newnative/offline paths.

CLI proposed additions on explicit HF/GH/native Xet file routes: repeated --include GLOB with optional repeated --exclude GLOB; require at least one include. New flags reject any --file/asset/release ambiguity before I/O, so old --file byte*/? grammar never silently changes. Explicit --quant TAG --exclude GLOB is a second base mode: match all requested-tag candidates, subtract excludes BEFORE familychoice, then enforcecompletefamily. Ban --quant+--include in the firstslice rather than invent their intersection/auxiliary semantics. Pure --quant without excludes retains qualified C2 behavior. --exclude alone is invalid. Candidates/release/components/cloud/ModelScope modes do not acquire new semantics implicitly.

Native HF applies filtering INSIDE original Hub/pin/tree metadata pass before discovery; reuse its exact request/body/control budgets, read grants/credential suppression/Xetroot/rawSHA/sourceEOF/ephemeral guard. Must retain >=1announcednative file and current native-index refusal; excluded nativeobjects cannot trigger compatibility fallback. No ordinaryfiltersresolve then second nativealiasresolve. If this thin path cannot be qualified in C3, reject its flags explicitly and keep nativefilter item unfinished.

Offline terminal branch precedes credentials/client/admission/signals, using retained fully verified original source lease/currentlimits/pointer policy and the same bounded matcher/strict closure under honest cachedsubset scope. It neither removes excluded cached data nor publishes a filtered snapshot. Validate whole requested result consistently with C2's final offline contract; if returning a whole SnapshotLease as exact result, reject manifestextras outside finalclosure instead of implying excludedbytes vanished. No remote-global-family completeness/freshness/privateaccess/nativeexecution proof from retainedsubset.

## Meaningful future qualification (all UNRUN)

- [ ] Golden primary fnmatchcases: UTF8scalar? vs original byte?, slashcrossing*/?/classes, dots, caseexactness, composed/decomposedUnicode distinction, collapsed**; classpositive/negation/ranges/nonASCII/leading]/literal^/hyphens/backslash/metacharquoting/unclosedbrackets/descendingranges and setpunctuation. Fixed expected cases from primarytests/independent Python oracle BEFORE Rust implementation, never regenerate expectations to fit new engine.
- [ ] Bounded exhaustive/differential short-string/alphabet tests versus pinned fnmatchcase oracle, with fixed seed/finite cases; no Python runtime dependency or compiler-generated expected values at productruntime. Reject malformedinputpolicy deviations explicitly in docs.
- [ ] Adversarial repeatedstar/no-finalmatch, longmissingbrackets, largeclasses and manyindividuallyfit patterns exhaust one sharedwork cap with typedLimit, bounded measured buffers and cancellation, not exponential hang or false no-match. Admission/explicit tiny-RAM refusal before HTTP/new allocations and default128/native256 envelope positives.
- [ ] HF/GH includeunion thenexclude narrows twoGGUFfamilies correctly, multipleunmatchedinclude patterns tolerated, emptyresult nofallback; old --file still componentlocal bytegrammar exactly. Split partialexclusion/missingmembers/conflictingtotals and sourceindex/config/tokenizer exclusions refuse; optionalREADME/tokenizer_config/LICENSE exclusion stays omitted, never re-added.
- [ ] Adjacentexcludedindex policy refuses before itsGET, no unqualified membershipskip; malformed/ambiguous/incomplete source closures retain oldcriteria. Exactcase/subfolders containingbrackets treated literally; samefolder componentboundary and fullinventory regularancestor collision refusal preserved.
- [ ] Movingbranch/fulltreepagination prove one pin/list and sourcecommit through HEAD/index/body; cumulativelogicalrequests/metadatabytes/header/probes/path/file/control checked once, no reset/extra catalog. Old provenance/digest/source identities equal corresponding explicitfile acquisition.
- [ ] Actual originalowner/native flow tokenisolation/metadataonlydryrun/defaults/cancel/root/fullSHA/EOF/oldref/rollback; cacheddata choice and explicitoffline before env/FIFO/stdin/client/admission/signals, source/currentlimit/pointer/closure/extras negatives, no data deletion/new snapshot.
- [ ] New CLI modes/conflicts/JSONsafe diagnostics/optionalfeatures and old API/exhaustivematch compatibility; strict production/diagnostic, original provider/component/native/legacy regressions, full isolated gate/sourceNix and original model support evidence stay coordinator-owned. No loader/tensor quant support claim.

## Remaining choices before implementation

- [ ] Root freeze final source-qualified C2 initialchoice/API names, excluded-adjacent-index policy and exact matcher control allowance. This is normal design ownership, not a requirement for new user authorization.
- [ ] Source grant, implementation, independent review and actual tests remain pending. C3 is not closed by this proposal or by typed Unsupported boundaries; broader filter compatibility must follow verified implementation.

## Approved C3 source grant

- [x] Root approved exact11path/API outline: artifact acquisition/{filters.rs NEW,quant.rs,selection.rs,huggingface.rs,github.rs,mod.rs,xet/mod.rs}, tests/acquisition_filters.rs NEW; CLI src/{pull.rs,pull_xet.rs}, tests/pull_filters.rs NEW. candidates.rs only if concrete need arises (none planned). API uses private FilterRequest::new(source,Vec includes,Vec excludes), ::for_quant(quant,Vec excludes), source/includes/excludes/quant_request getters; filters resolve/cachedvalidator and native resolve_filtered.
- [x] Root approved excluded-index refusal only where original closure would attempt that index; keep unrelated/GGUF paths unaffected. Fixed matcher bounds and conservative combined extra16MiB SAME admission; offline new mode exact closed snapshot subset and digest-only.
- [ ] Implement matcher and provider/closure/CLI/native reuse, independent fixture coverage, static formatting and independent source review. All Rust compile/tests/lints/gates UNRUN until MAIN grant.
- [ ] Oracle setup initial attempt failed before freezing any output: pinned Python3.14 fnmatch source uses new strict-end regex spelling \z; installed Python3.13 rejects it because initial replacement accidentally looked for two backslashes. Correct only the spelling adaptation to synonymous strict-end \Z, then freeze literals before Rust matcher. This is oracle tooling, not a Rust test failure/RED or fitted expected value.

````

## Source: .agents/runs/model-acquisition-v1-completion/TODO.md

Original UTF-8 bytes: 1031; SHA256 `ab3951061087170c4e5626c66c5d5e81ebad7397307a6963282c7ff771093a53`.
Resolve embedded relative paths against the original path, not this volume.

````text
# Model pull v1 source closure cutover

Status: active until exact v1 gate and signed acceptance
Base: f56b57400eb913240d6e2b2eeea5d60d9ab6f35e
Main plan: ../../plans/model-acquisition/TODO.md

- [x] Revalidated accepted C3/C4/C5 and distinct provider-parity TODOs.
- [x] Shared bounded same-folder optional companions in HF/GitHub/cache
      closure. Metadata stays opaque; no new loaders or default model.
- [x] Synthetic Qwen/Gemma-like exact indexed >51GB source-plan refuses
      unadmitted disk before any payload GET.
- [x] CLI explicit index-first and large disk admission input test passes.
- [ ] Offline exact closed-snapshot companion and exclude tests.
- [ ] Strict production and diagnostic lint; complete isolated check.sh.
- [ ] Actual Nix filtered-source-integrity, README/Cargo/links checks.
- [ ] Two original release cache_prepare source-model tests separately.
- [ ] Integrate only verified Rust and plans, signed commit, no push; pivot
      to model execution without claiming future provider/alias parity.

````

## Source: .agents/runs/model-acquisition-xet-index/TODO.md

Original UTF-8 bytes: 5809; SHA256 `8823513c28666b74e86e503b267f3dd50222d460b7ee772089f9277a486b405c`.
Resolve embedded relative paths against the original path, not this volume.

````text
# TODO — native Xet required index reconstruction

Status: paused at coordinator/user request; incomplete source-only draft, all Rust checks UNRUN.
Provenance: f22f759 + exact11frozenC2overlay in disposable zerogravity-xet-index-work, fresh copied mtimes. Root integrates C2/C3 later; preserve quant draft frozen.
Main plan: [model acquisition](../../plans/model-acquisition/TODO.md).
Owned scope: acquisition/huggingface.rs, xet/{mod,budget}.rs, CLI src/{pull,pull_xet}.rs, new artifacts/tests/acquisition_xet_indexes.rs and this run. No MAIN/Cargo/Git/canonical/dependency/provider/model/SDK/device actions.

## Uncommitted

- [x] Read current rules and exact C2/native/HF/source/budget/helper seams. Root approved private defaultfalse XetOptions index policy with additive with_index_reconstruction(self) builder, preserving all resolve signatures.
- [~] One pinned HF callback: exactlength/LFS rawSHA/native complete root/actual EOF before original parser, same closure/tree and original reader/HTTP budget/scratch. No index body cache; reconstruct again during original publication and charge both.
- [ ] Full consumed-counter Result seed across planning/execution, poison/currentlimit refusal and focused private tests.
- [ ] CLI opts native clone into index reconstruction only for apply; dryrun/default keeps no CAS/grants/signed bodies, no added CLI flag or Files/Quant public API multiplication.
- [ ] Authored actual original-HTTP synthetic index+weight publication and scripted limits/root/hash/EOF/cancel/admission/oldref fixtures; all Rust tests UNRUN until exclusive grant.

## Tests and verification

- [ ] Source static comparison/review + scoped formatting/whitespace.
- [ ] New production/diagnostic source/API/CLI fixtures; all UNRUN.
- [ ] Old HF/GH/selection/C1/C2/native/cache/CLI regressions and strict configurations; root target grant required.
- [ ] Full isolated gate/Nix/approved originals after final merge, coordinator owned.

## Boundaries

No compatibility-body fallback or repin/relist; same mathematical engine/source issuer/cache/publisher. Required index is bounded metadata, not new model execution. Default library resolve/dryrun retains noCAS policy; explicit builder authorizes required native index data only. RawSHA independent from native complete root, all counters cumulative, process buffer cap not allRAM. No performance/inference claim.


### Pause handoff — user-requested wrap/interruption boundary

- [x] STOP source drafting after coordinator relayed explicit user pause. All tool commands have completed; no Cargo/rustc/build/test command or owned active child/session was started. No MAIN/Git/index/refs/merge/dependency/network/model action.
- [x] Actual edited production paths ONLY: `crates/zerogravity-artifacts/src/acquisition/huggingface.rs`, `crates/zerogravity-artifacts/src/acquisition/xet/mod.rs`, `crates/zerogravity-artifacts/src/acquisition/xet/budget.rs`. Quant SIDE remains frozen/unmodified; C2 overlay in index SIDE remains context only. CLI pull.rs/pull_xet.rs and acquisition_xet_indexes.rs have NOT been changed/created for this slice.
- [x] Preserve complete prior cold design at sibling `model-acquisition-xet-index-design/TODO.md` in THIS index SIDE. Its old additive resolver recommendation is superseded by the coordinator's builder decision below; useful protocol/accounting/tests remain applicable. Do not depend on retired quant SIDE for context.
- [x] Drafted PRIVATE defaultfalse XetOptions policy with public additive `with_index_reconstruction(self)->Self`; same resolve/resolve_quant signatures. Draft native/HF callback borrows original pinned index context, exact LFS rawSHA/length/validator, reuses discovery/Access/XetReader/Invocation scratch admission, exact Vec plus EOF sentinel/SHA before parser, no index body cache. Existing normal HF index GET remains when no callback/LFS. Indexed spec is retained for final plan without another announcement; later executor reconstructs index again.
- [x] Drafted ALL-counter private Seed with Result snapshot/poison refusal: attempts/wire/metadata/decoded/leaves/refreshes/accepted terms/groups/ranges, validation on restore. Two private budget unit cases authored (carry/currentlimit refusal and poison), both UNCOMPILED/UNRUN.
- [x] Source-only scoped rustfmt --edition2024 --config skip_children=true and corresponding --check on the three edited production files; git diff --check PASS. These establish formatting/syntax parsing only, NOT Rust type/lifetime correctness, protocol acceptance or runtime support.
- [ ] INCOMPLETE: no new acquisition_xet_indexes.rs integration tests yet, no actual original-HTTP synthetic index+weights publication, no counter/admission/root/SHA/EOF/cancel/rollback qualification, no CLI opted-in clone/native-apply wiring/help changes. Default stillfalse, so CLI has no newly enabled feature. Do not merge/advertise this draft.
- [ ] Resume with independent read/type/lifetime/source-preservation review of private callback/reader borrowing and error/cancellation dominance before first compile. In particular check original native announcement/validator behavior and no hidden source-policy change, same metadata body/counter budget and declared default256MiB envelope, all-counter seed poison/current-limit behavior, exact EOF/root before parser and priorref unchanged on any failure. No test/RED/GREEN result is claimed.
- [ ] Root must first reconcile accepted C2/C3 production overlap onto the eventual integration base; this draft starts f22f759+frozenC2, contains NO C3 changes. Preserve C3 selection/callback policy when integrating later. CLI apply opts in via native-options clone ONLY; dryrun/default retains noCAS/no-signed-body, no API function multiplication. No fresh source/Cargo work until coordinator/user resumes.

````

## Source: .agents/runs/model-acquisition-xet-index-design/TODO.md

Original UTF-8 bytes: 13746; SHA256 `b1e0e5115c089a0363ba5f6edf254e3cb3434496e82e880228910c5cee1138c7`.
Resolve embedded relative paths against the original path, not this volume.

````text
# TODO — native Xet required-index reconstruction design

Status: cold design; no source or runtime qualification.
Provenance: accepted native52e42fc; read current MAIN accepted implementation while C1 gates run. This run lives only in zerogravity-quant-work and is separate from frozen C2.
Main plan: [model acquisition](../../plans/model-acquisition/TODO.md).
Owned scope: this TODO only. No source/Cargo/MAIN/Git mutation, provider requests, payload/model/SDK/device/live-cache actions.

## Uncommitted

- [x] Read accepted native README and HF closure, Xet Hub/discovery/Access/Reader/Invocation/Budget and executor index validation.
- [x] Identify exact one-pin index-reader seam and verification/admission/control/cumulative work obligations below.
- [ ] Coordinator chooses dry-run/API policy and authorizes a source slice; no implementation authorized by this design.
- [ ] Implement/qualify only after that grant, reconciling accepted C1/C2 without changing legacy selection or native identities.

## Smallest useful slice

Support an HF LFS-backed selected/adjacent `.safetensors.index.json` required by ORIGINAL closure when its fixed commit URL announces native Xet. Reconstruct that small index through ORIGINAL Access + XetReader + hash/codec/multipart/HTTP owners before SafetensorsIndex parses it. Preserve the same one exact commit, pinned tree and remaining metadata bytes/logical requests. Keep non-LFS ordinary index GET behavior; no repin/relist, compatibility redirect/body fallback, new source issuer, cache/publisher, CAS URL/token authority or chunk runtime. Required model weights still transfer only after final closure and normal cache transaction admission.

### Private provider seam

1. Add an optional PRIVATE admitted index-reader callback at HF's existing exact-size/index GET point, after ensure_length and before original `SafetensorsIndex::from_json`. Supply a borrowed pinned context: original repository/commit/path/URL, exact length, LFS raw SHA and existing validator. Do not fake an HttpResponse/header or infer index status from an arbitrary URL in Hub::get.
2. Ordinary HF/compatibility Files and quant routes keep their original callback/GET path. Native entrypoint supplies the callback; it uses the SAME retained Hub budget and caller TransferAdmission. The callback cannot authorize another source/path or change the original selected/adjacent-index policy. GGUF quant normally needs no index; its shared initial-choice path must remain unchanged.
3. For native LFS index, factor/reuse the EXISTING announcement checker: exact fixed commit URL, header-only302, x-xet-hash parse, all present commit/size/rawSHA bindings checked. Require the pinned LFS rawSHA plus exact length; Git OID is not rawSHA. No follow/compatibility response body. A missing/changed announcement refuses.
4. Keep bounded path→XetSpec for an index already observed so final plan can attach that exact spec without another discovery call. Other final files use original discovery. No index-body cache or new persisted records. Original per-source plan/provenance/path identity stays unchanged.

### Exact bytes before parser

- Refuse before CAS/scratch allocation if index exact length exceeds caller max_file, remaining cumulative HF metadata bytes,8MiB parser bound, or addressable allocation. A selected unknown length must first use the already-approved exact metadata/HEAD policy; no cap/partial JSON mode.
- Lazily acquire ORIGINAL Scratch::reservation from caller's same registered transfer domain BEFORE constructing Scratch/Access/reconstruction vectors. Reuse one scratch sequentially for multiple indexes; retain the outer original resolution control lease. Do not create/register a new arbiter/key or weaken limits.
- Allocate the index Vec with exact checked capacity under the existing resolution control envelope, perform bounded reads/cancellation, incrementally compute actual SHA256. After exactly expected bytes, request the separate EOF sentinel: `take(expected)` alone is WRONG because it can skip XetReader's final full-root/drain/EOF verification. Require no extra logical byte and actual terminal native completion.
- The original XetReader must verify complete ordered file root and exact logical length, including extra edge decoding, signed complete interval set/drain/multipart terminal semantics and replay rules. Compare actual rawSHA with pinned LFS SHA independently. Neither Xet hash nor transport metadata replaces rawSHA.
- Reuse reject_pointer; only then return bytes to original static-error SafetensorsIndex parser and required-shard closure. Never parse tentative prefixes/unverified raw bytes; no shard URL/body before closure. Raw index bytes are metadata/data, not executable readiness or tensor total-size authority.
- Decrement ONE cumulative HF index/body byte budget and logical request budget across all selected/adjacent indexes. Charge decoded index bytes to the native semantic metadata budget (without pretending decompressed bytes are additional physical wire), while actual compressed/index/CAS/headers/retries remain charged by the original native HTTP work owner. Existing observed metadata budget includes headers, so limits may conservatively refuse before HF body-only budget is exhausted.

### Admission and lifetime accounting

Current defaults: outer metadata control envelope80MiB (`8*8MiB+16MiB`), native scratch90.5625MiB (includes40*262144 leaf storage, fixed scratch/control and8*metadata). During index resolution their simultaneous declared peak is170.5625MiB; existing native CLI256MiB can fit,128MiB must deny before signed index body. The bounded index Vec/parser is covered by the original outer metadata envelope; do not add an unaccounted second copy. Custom larger limits must use the actual checked formulas, not this default arithmetic. This scopes process transfer/control buffers, not all host RAM/model memory.

Drop resolution scratch, grants and index bytes after their last parser/closure use and before returning the final plan. The final payload invocation then uses original scratch+cache transaction envelopes (~162.6875MiB defaults). First slice may reconstruct the tiny required index again during publication using existing executor so it is actually included in the immutable snapshot; keep that second transfer bounded/charged and honest rather than inventing a temporary index-body cache. The same pinned rawSHA/spec must verify it again, and original executor reparses staged verified index closure before commit. No cache root/ref mutation during resolution; failures/cancellation leave old refs untouched.

### Mandatory cumulative seed change

CURRENT `budget::Seed` carries ONLY attempts/wire/metadata because accepted resolution has not decoded native bodies. Adding pre-plan index reconstruction without changing that seed would RESET terms/groups/ranges/decoded/leaves/refreshes at Invocation::new, exceeding declared whole-operation caps.

Extend the PRIVATE seed/snapshot to carry ALL consumed counters: attempts,wire,metadata,decoded chunks,accepted logical leaves,refreshes,accepted terms/groups/ranges. Budget::new validates every seeded count against the same options and starts from them. Budget::seed must return Result if reading the accepted Mutex can fail; never silently default/zero on poison. Final plan seed is captured only after all index reconstruction and final discovery. Reader scratch leaves can clear per file, but cumulative Budget leaf/decode counts cannot. Count retried/discarded edge chunks and accepted reconstruction entries on every reissue exactly as now. Every physical HTTP retry/redirect attempt retains original begin_attempt behavior; metadata logical calls are a distinct finite counter, not wire telemetry.

Public plan reuse still begins an independent admitted execution from its resolution seed, matching current semantics; do not claim one global cross-process grant/work counter. No grant/token/signed URL persisted in the plan.

## Root choices before source grant

1. **Recommended preserve existing dry-run promise:** existing `xet::resolve`/CLI dry-run remains Hub metadata+announcement only, no CAS grants/signed bodies. Add a named additive native resolver that explicitly permits required-index metadata reconstruction for APPLY (e.g. `resolve_with_indexes`, private shared implementation). CLI apply chooses it; dry-run that NEEDS native index returns explicit Unsupported before grant and cannot claim complete closure/plan. No new flag or silent dry-run network expansion. Alternatively root can deliberately revise dry-run to permit CAS for index metadata, but current help/README's explicit no-CAS/no-signed-body promise must change and be tested; that is a policy decision, not an implementation shortcut.
2. **Recommended first slice require trusted LFS rawSHA for native indexes.** Already qualified ordinary non-LFS index metadata remains allowed under existing pinned/TLS/FirstSeen rules. Native announced index lacking a pinned rawSHA refuses before CAS; do not promote a newly observed hash to expected authority.
3. **Recommended no pre-plan cache reuse/staging.** Reconstruct required index twice if needed (planning then normal immutable publication), charge both against cumulative limits, document the bounded cost. Avoid introducing a new cache ownership/transaction-before-closure protocol merely to save <=8MiB. Future retained verified index optimization is separate.

## Proposed source/test ownership after approval

Native/HF owner: HF private index callback and native entrypoints/discovery helper/Invocation scratch constructor; xet/budget.rs full seed; same reader/access APIs wherever a narrow helper permits reuse. Shared scratch/codec/hash/multipart stay exact, no duplicate parser or new transport. CLI owner: only apply versus metadata-only resolver choice and precise failure/help wording in existing shared run_fetch, preserving ordinary None and C2 Files|Quant source choice. Tests: new acquisition_xet_indexes.rs plus existing native/ordinary HF/GH/selection/legacy/candidate/quant suites. Root owns docs/gates/dependencies (none expected), integration/Git. Concrete paths/API finalized only after root's dry-run decision.

## Tests and verification — all proposed, UNRUN

- [ ] Golden tiny LFS-index native reconstruction produces verified JSON, original closure requires every same-commit shard+sidecar, native plans retain original per-file/source/provenance. Fixed one alias lookup and one tree listing; adjacent index uses same authority. No weight body before closure; required missing shard refuses without publication.
- [ ] Actual original HTTP owner synthetic Hub→read token→CAS reconstruction→signed ranges for index; Hub token never reaches CAS/ranges, CASgrant never Hub/ranges, signed URL auth suppression/no-follow unchanged; final model source roots/rawSHA valid independently. No live provider/account.
- [ ] Valid JSON with wrong rawSHA or wrong Xet root refuses BEFORE index parser/model body; correct root/SHA but malformed/unsafe/duplicate/overflow index refuses with static error. Truncated/missing signed interval/trailing data/replayed changed chunk/cancellation at exact final length refuse; original root/drain/EOF checks actually reached.
- [ ] Multiple required indexes share total logical request/body/cardinality limits; max_terms/groups/ranges/chunks/refreshes configured to allow each file individually but reject index+later payload cumulatively. Decode/refetch edge work and second planning/publication index copy consume remaining limits. Unit seed snapshot restores every counter and poisoned accepted owner returns error.
- [ ] Full-default128MiB admission fails BEFORE signed index body;256MiB succeeds same fixture and releases all declared memory after drops. Custom metadata/chunk bounds prove arithmetic, exact allocation/no geometric growth and no hidden globalRAM claim.
- [ ] Metadata-only dry-run performs zero CAS grants/signed bodies even when native index is required; apply explicitly reconstructs bounded metadata. Offline requires no credentials/client/provider/admission; existing cached data does not become native execution proof.
- [ ] Grant401/range403 retries charge original total attempts/reissues and preserve reconstruction contract; end-to-end old-ref preservation on each new preplan refusal/cancel. Existing publisher fault matrix remains unchanged; no redundant new generic archival/proof framework.
- [ ] Original HF/GH compatibility index/shard/sidecar, legacy offline, C1 discovery, C2 quant, native non-index and ordinary sidecar resume tests; strict feature/dependency matrices, full isolated gate/Nix, approved originals as root decides. No new inference/throughput claim.

## Current code anchors

- `acquisition/huggingface.rs::resolve_from` exact index length→metadata_checked→verify_metadata_digest→SafetensorsIndex parser, under one existing pinned tree/metadata_left.
- `xet/mod.rs::resolve`, `Invocation::new/reader` and `PlanState::seed`: existing original Hub/discovery and reader/admission owner.
- `xet/access.rs::Access::{reconstruction,fetch,refresh}`: scoped grants/typed target and cumulative accepted reconstruction owner.
- `xet/reader.rs::{Scratch::reservation,XetReader::finish_source}`: actual allocation envelope, terminal complete root/EOF.
- `xet/budget.rs::{Seed,Budget::new,seed,reconstruction,decode_chunk,file_leaf,refresh}`: seed currently partial, MUST extend before decoding during resolve.
- `acquisition/executor.rs::execute_with_progress`: sole transaction/importer and staged index reparse before commit. README explicitly leaves native LFS/index reconstruction open and says dry-run no CAS grants/signed bodies.

````

## Source: .agents/runs/model-execution-bf16-operator-gap-2026-10-20/HF_SHARED_ENGINE_SEAM.md

Original UTF-8 bytes: 6565; SHA256 `cad11bfc92fa95c8548f249f5f10c4038c9d80c3a78a9444bcd5a4206f6c649f`.
Resolve embedded relative paths against the original path, not this volume.

````text
# Original HF through the existing Qwen engine

Owner: baseline_review. Read-only source review; no production edits/tests/GPU.
Sources inspected: model-qwen35 lib/forward/host_math/checkpoint; core accelerator
program; artifacts qwen_owned/qwen/source_weights; facade original_qwen math,
precision/state; actual slow-GDN observation. This is a proposal, not inference
qualification.

## Current hard boundaries

- LoadedQwenModel already owns the one verified SourceWeights allocation, but
  from_checked_source requires architecture qwen35, GGUF names/raw matrix axes,
  equal GDN key/value heads, and converted scalar-vector semantics.
- forward_token is the one generic schedule; run_layer_sequence is already
  shared with original analytical diagnostics. Sharing traversal alone does
  not make the analytical proposal path a MathematicalBackend/ForwardPlan.
- MathematicalBackend/PlanOperation cannot currently distinguish ordinary
  source zero-centered RMS, BF16 cast boundaries, source L2, query scale order,
  original-F32 gated gamma, or initial-chunk versus cached GDN semantics.
- SourceTensor MatrixAxes pending patch solves genuine raw source shape order;
  it does not solve these arithmetic/profile distinctions.
- OriginalQwenLayerMath is in the facade, which depends on model-qwen35.
  Calling the facade from model-qwen35 would create a dependency cycle.
- QwenState and current checkpoint v1 have no numerical-profile tag. v1 is
  intentionally frozen; source bytes/config alone do not identify runtime
  reduction/cast policy.

## Minimal safe production-facing slice

1. Add a closed source/numerical binding on LoadedQwenModel. Keep existing
   GGUF construction/default profile byte-for-byte semantically unchanged.
   New explicit original-source construction receives the already verified
   SourceWeights plus exact config/index pins. Borrow QwenOriginalText only
   during validation; then retain geometry, actual source role descriptors,
   raw small learned parameters with original dtype, tied/untied head choice,
   and source/index/bundle/config binding. Move the SAME SourceWeights into
   LoadedQwenModel. No self-borrow, full transpose, alias source identity,
   tensor metadata rewrite, duplicate source owner or silent dtype conversion.
2. Replace hard-coded names in the SAME forward_token with typed role
   descriptors. GGUF descriptors remain InputOutput; actual HF matrices use
   OutputInput. Embedding/head rows stay genuine original names. Preserve
   all existing GGUF node/operator order as a regression condition.
3. Add explicit closed profile operations/attributes for observable HF
   semantics. At minimum: dtype cast, zero-centered/raw-gamma norm policy,
   grouped head repeat/packing, source L2, pre-dot query scaling, source
   coefficient interpretation, phase-aware GDN, gated norm with parameter
   dtype, config rope and tied-head selection. Record these distinctions in
   ForwardPlan; an unaware CPU/provider/GPU lowerer must return Unsupported
   before launch/publication. Do not select new math by tensor-name strings
   while emitting indistinguishable old nodes.
4. Reuse checked scalar/shared kernels from lower crates. Move or re-export
   pure OriginalQwenLayerMath operator helpers down into model/cpu/core;
   facade diagnostics and HostMath call those same helpers. The new explicit
   BF16 CPU projection profile already lives in the shared CPU row kernel.
   Do not copy OriginalQwenStateBridge's proposal/token loop into production.
5. Retain QwenDenseCpu/SessionAuthority publication, resident physical leases,
   Ready/status/quarantine and canonical StateSlot domains. Generalize delta
   geometry from StateSpec: conv=(2*Nk*Dk+Nv*Dv)*(kernel-1), recurrent=Nv*Dk*Dv;
   remove equal-head assumptions only behind original validated geometry.
   Source key-major recurrence becomes canonical value-major by bit transport.
   Bind snapshots/checkpoints to numerical profile. Keep legacy v1 unchanged;
   reject nonlegacy v1 I/O until a separately tagged format exists. New in-memory
   imports must reject cross-profile payloads even for identical source bytes.

The first implementation can end at exact source-descriptor/plan construction
and deliberate Unsupported execution. Then qualify one closed HF operator at
atime before allowing complete CPU execution. A native BF16 GPU lowerer is
separate work; MatrixAxes plus source loading cannot enable GPU serving.

## Numerical work immediately after projection64

- Clean-input replay first: use actual GDN observer q/k and compare all words of
  normalized and scaled outputs. Pinned slow source uses FP32 square/sum/rsqrt
  and pre-dot scaling; current GGUF grouped RMS uses F64 sum/mean/sqrt-recip.
  Preserve GGUF implementation. Exact reduction and rsqrt implementation need
  source-real replay, not algebraic substitution or tolerance widening.
- Qualify actual first-token chunk64 separately from cached recurrent step.
  The latter materializes products, sums key-major dim-2, and separately rounds
  multiply/add; GGUF FMA dot/update is a different profile. First-token current
  acceptance scope is source T=1 padded to64, not arbitrary full-prompt prefill.
- Gated RMS original restored F32 gamma must preserve promotion to FP32 between
  BF16-normalized core and FP32 SiLU gate. Current unconditional intermediate
  BF16 round is an extra cast. Parameter dtype must be source-bound, and an
  all-BF16 loader is a distinct reference profile.
- Recheck convolution/SiLU and coefficient transcendental outputs on exact
  captured inputs before assigning downstream numerical errors. Current full
  Rust trace is upstream-contaminated; captures are observations, not passes.

## Suggested disjoint ownership

- Coordinator: loaded-model source/role binding, one schedule, profile admission,
  source/runtime scope, final complete source tests and GPU coordination.
- Core/CPU agent: closed per-op profile descriptors/shared scalar helpers and
  exact clean-input operator tests, no session/serving changes.
- State reviewer: existing checkpoint/domain codec profile binding and exact
  layout transport, preserve all legacy v1 bytes/rollback ownership.
- Independent reviewer: GGUF plan regression, provider Unsupported behavior,
  actual source/config/index/tensor validation and full StateRoot/logit proof.

No code, backend, family, model or throughput capability is established by this
proposal. Original27B still requires physical source streaming/admission; its
55GB source cannot be materialized in current host RAM.

````

## Source: .agents/runs/model-execution-bf16-operator-gap-2026-10-20/TODO.md

Original UTF-8 bytes: 3101; SHA256 `05728d00622c5fcf121a824ef4fc98bd8441eb3e7e78a88edb8a29f0befdedd7`.
Resolve embedded relative paths against the original path, not this volume.

````text
# Original BF16 first-operator gap

Parent: model-execution. Owner: baseline_review.
Scope: ignored CPU-only operator scripts, pinned public Torch source reads.
No MAIN source/Cargo/GPU/production math changes. Runtime pinned0.31 image,
Torch2.13+cu129@cf30153c4c131c8164ee7798e5022d810682e2cb, CPUAVX2,4threads
and1interop, networknone/source+reference+Rustreadonly,4CPUs/12GiB.

- [x] Actual standalone original Torch RMSNorm and BF16linear exactly reproduce
  captured source-36-F32-restored layer0 norm1024/QKV6144, tokens11/353.
- [x] Explicit scalar GGUFnorm SIMULATION equals captured BF16norm1024/1024
  bothsteps; this is not the missing actual Rust live trace.
- [x] Explicit serial F32dot SIMULATION BF16cast matches authenticated actual
  Rust QKV trace6144/6144 bothsteps; reproduces1/3 differences vs Torch.
- [x] Exact selected dyadic dots show F64 is not the requested Torch profile:
  step0row3553 exact=-504639/8589934592 roundsBF16b876 likeRust,
  actualTorchBF16linear b877. No tolerance widening or precision shortcut.
- [x] Pinned source identifies non-AVX512BF16 transposedB1GEMV route to
  bf16_dot_with_fp32_arith.8F32registersx8AVX2lanes=64partials; per64
  original elements FMA; register reduction offsets4/2/1 thenhorizontal
  lanes4/2/1; vector16-element and scalar tails; BF16castonce.
- [x] Source-derived topology SIMULATION exactlymatches all6144 actualTorch
  QKV BF16outputs BOTHsteps. Actual Rust invocation awaits MAIN kernel work.
- [x] Authentic third-run standalone Rust ordinary_norm dumps agreeALL1024
  Torch captured norm bits BOTHtokens. Authentic rawF32projection agreesALL
  6144serialsimulation bits BOTHtokens. Firstcausalgap projectionreduction.
  Separate operator invocation only; missing live trace remains missing.
  actual-rust-direct-operator-comparison.json retains exacthashes/scope.
- [x] Actual pinnedTorch synthetic tailprobe:28widths1..1024,7rows each,
  source-derivedtail formula matchesALLBF16bits. Deterministic28casegoldens
  plus128wide cancellationcounterexample retained for isolated Rust tests.
- [ ] Source-profile-scoped shared BF16 kernel change and complete source-real
  fulltrace/all42/logit acceptance remain MAIN work. Preserve accepted GGUF,
  F16/F32 and analytical profiles; no family-specific copied math.

Evidence: operator-probe.json/log, lane-probe.json/log, unrounded F32 arrays,
pinned-torch-source/manifest.json+exact files, all invocation JSONs.

Primary source:
- https://github.com/pytorch/pytorch/blob/cf30153c4c131c8164ee7798e5022d810682e2cb/aten/src/ATen/native/CPUBlas.cpp
- https://github.com/pytorch/pytorch/blob/cf30153c4c131c8164ee7798e5022d810682e2cb/aten/src/ATen/native/cpu/BlasKernel.cpp
- https://github.com/pytorch/pytorch/blob/cf30153c4c131c8164ee7798e5022d810682e2cb/aten/src/ATen/native/cpu/ReducedPrecisionFloatGemvFastPathKernel.cpp
- https://github.com/pytorch/pytorch/blob/cf30153c4c131c8164ee7798e5022d810682e2cb/aten/src/ATen/cpu/vec/functional_base.h

Torch reduction file itself notes adaptation from GGML under MIT; preserve
applicable attribution if translating source into Rust.

````

## Source: .agents/runs/model-execution-conv-zero-observation-2026-10-20/BACKEND_TOPOLOGY.md

Original UTF-8 bytes: 6505; SHA256 `c43f59a3c6bfaa5e435dff749d2b87f7f75be0dcf78ed97408a42cba9de09f2d`.
Resolve embedded relative paths against the original path, not this volume.

````text
# Pinned original CPU convolution: exact signed-zero cause

Scope: CPU only, no Cargo/GPU/model loop/default-math edits. Original function
outputs remain oracle; labelled order candidates never replace source output.

## Pins

- Torch2.13.0+cu129 git `cf30153c4c131c8164ee7798e5022d810682e2cb`.
- Stock image digest `b18abb2df97b8f798e81862bd93f872ea18613372e2c3adc0cc2ac21e66ac12f`.
- Original .8B revision `2fc06364715b967f1860aea9cf38778875588b17`.
- Modeling SHA256 `762feb6c7426a7f15b5bf830df54c07438bf9e7c27b8cdb23179045920412c3b`.
- CPU AVX2, MKL2024.2, oneDNN3.12.0; exact build in backend-report.json.
- libtorch_cpu.so SHA256 `1b4ed646ce51a4c341808552426a0f9e2e2c04961955759701066d335c1631ab`.
- Source archive344928B, SHA256
  `2c13b27d9477a9e05e9e5860d75a9e1b1697a6015dbba542b92d0ef614121b83`; unchanged.
- All containers pullnever/networknone/12GiB/4CPU/capdropall/offline/source-ro.
  CPU-only profiler performs an unavailable GPU device-metadata query (code35),
  retained in logs; no GPU work/device mapping occurs. No benchmark claim.

## Actual first differing operation

Untouched causal_conv1d_fn at source line287 yields the preactivation before
SiLU. All24576 function-return BF16 words match prior accepted observation.
Current CPU64 Rust raw QKV matches all6144 words. Only newest-token channel2527
conv/SiLU differs: actual +0, old Rust -0. Actual SiLU(-0) remains -0, so the
convolution is causal, not activation. conv-report.json retains exact operands.

Channel2527 widened BF16 input `[0,0,0,83150000]`, weights
`[82200000,82200000,82200000,02200000]` produce four separate FP32 products -0.
The original scalar expression gives -0; actual convolution gives +0.

## Actual dispatch + pinned compiled arithmetic

- backend-report.json:6,144 aten::thnn_conv2d / aten::_slow_conv2d_forward calls,
  each input/weight [1,1,1,4]; no aten::mkldnn_convolution.
- Source Convolution.cpp:1722–1734 splits grouped Slow2d by original channels.
  ConvUtils.h:424 selects suggested format, not ambiguous is_contiguous alone.
  Public native replay on authentic groups0/2527 agrees bitwise and produces
  contiguous output stride[7,7,7,1]; group-backend-report.json.
- ConvolutionMM2d.cpp:268–286 therefore calls BF16 CPUBlas NN,
  m7/n1/k4, alpha1/beta0.
- mkldnn/Matmul.cpp:166–170 refuses m*n*k<=4096, here28.
- Actual BF16 CPUBlas overload at ELF0x1c975e0 (size0x225) calls
  mkldnn_bf16_gemm then gemm_stub on refusal; no sbgemm call/branch.
  compiled-bf16-cpublas-hot.txt proves this runtime build.
- Configured AVX2 gemm_stub points to ELF0x778c030, calling main dtype lambda
  at0x77896e0 then BF16-specific lambda0x777bcd0 (BF16 widens via <<16).
  Read-only static pointer receipt: native-blas-dispatch-v2-report.json.
  This receipt is static dispatch provenance, not an instruction-level trace.
- Pinned BlasKernel.cpp sum(k) has four initial +0 partials. Its actual AVX2
  BF16 compilation uses four independent FMA instructions at0x777c25d/27c/29f/2ba,
  with +0 initialization at0x777c210–21c; direct additions at0x777c329/32d/331.
  Source: ../model-execution-bf16-operator-gap-2026-10-20/pinned-torch-source;
  compiled proof: compiled-bf16-generic-23.txt. Thus k4 is:
  p[i]=fma(weight[i],window[i],+0); raw=((p0+p1)+p2)+p3; BF16-RNE(raw).
- No new +0 accumulator after partials. Zero signs follow arithmetic.
  This is a closed finite-BF16 four-tap profile, not general GEMM equivalence.

## Failed proposal, corrected proof

Initial source-only reading inferred separate products +0 accumulation and
proposed reusing CPU64 projection scalar tail. The trained6144 vectors matched,
but a meaningful adversarial native case disproved general reuse:
all four inputs0x8080, weights0x0080 → source BF16 -0, separate-product tail +0.
The source C++ permits compiler contraction; compiled FMA proof above resolves it.
No scalar-tail or default GGUF implementation was changed.

Four-FMA proposal then compared against untouched original convolution:
- all6144 observed trained preactivations exact;
- all2055 source-free preactivations exact (7 explicit edge cases +2048 random
  broad-exponent finite BF16 rows, seed57021);
- actual profiler confirms all2055 use the same slow-convolution path.
See contracted-four-partials-report.json and compact-conv-goldens.json.
These comparisons are source-derived candidate qualification, not Rust execution.
Raw-F32 synthetic words are candidate/IEEE checks; actual source exposes BF16.

Cached original causal_conv1d_update also uses F.conv1d, concatenating source
cache4+token1 and selecting last output. That newest k4 window is previous3+input;
source kernel has m2/n1/k4 (8<=4096). model-conv-functions.json pins exact code.
This does not by itself qualify cache transport or complete recurrent inference.

## Synthetic reduction fixture correction

Original key-axis synthetic16384 expectation was wrong at the power-of-two
boundary (finer spacing toward zero). Actual pinned public torch.sum verifies
large=16385 and -16385, key0=large/key16=-large/key17=2^-10:
source subtotal result+0, serial key-order result2^-10. sum-fixture-report.json.
No reduction helper arithmetic changed here.

## Retained unsuccessful observation attempts

- TorchDispatchMode recorded no internal convolution ops; dispatch-metadata-report
  establishes output preservation only, not backend metadata.
- Dynamic-only ELF symbol lookup found no private CPUBlas symbol; local symbols
  were then used. First local dump accidentally retained the cold clone; hot
  dump is separately named and exact-sized.
- First microkernel search required AVX2 namespace absent from hidden lambdas;
  generalized local symbol search produced exact functions.
- native-blas-dispatch-report.json used an invalid minimum mapping bias when
  translating ELF offsets; v2 uses zero-offset ELF mapping and supersedes it.
  The global stub first pointer was null in this observation; static AVX2
  configuration and source+compiled control flow are the dispatch evidence.

## Implementation boundary

Dedicated helper lives only in isolated p12-hf-conv-20261020, original_hf/conv.rs
and one module declaration. Genuine immutable rank3 BF16 source, bound identity
and original tensor name; exact widened BF16 activation/history; all bindings
checked before arithmetic; private canonical next3 + rawF32/BF16 outputs.
No SiLU/facade/default GGUF/provider/ForwardPlan/session/StateRoot change.
Root must compile/test and replay complete authentic6144 vectors in Rust.

````

## Source: .agents/runs/model-execution-conv-zero-observation-2026-10-20/TODO.md

Original UTF-8 bytes: 1193; SHA256 `1e36c88d7e9f563337f82796cda9228b366b8d7704359808053ee23a304ccfc1`.
Resolve embedded relative paths against the original path, not this volume.

````text
# Authentic conv/SiLU signed-zero diagnosis

Parent: model-execution. Owner: batch_design_review.
Scope: this ignored run only; pinned offlineCPU12GiB/4CPU, no model loop,
MAIN/worktree edits, Cargo/GPU, zero canonicalization or altered source output.

- [x] Observe original first-token causal_conv1d_fn preactivation and actual
  SiLU return from accepted original inputs/weights; compare full prior output.
- [x] Verify CPU64 Rust QKV bit-equal and isolate exact channel2527 operands.
- [x] Compare separately labelled public-op source-order candidates; retain
  authentic source output as sole oracle and report evidence before any fix.

- [x] Actual CPU profiler + pinned source/compiled backend topology.
- [x] Correct rejected scalar-tail proposal with all-underflow signed-zero
  counterexample; four-FMA source-derived candidate matches6144+2055 BF16 outputs.
- [x] Verify corrected key-reduction adversarial constant16385 under actual Torch.
- [x] Preserve authentic archive unchanged; BACKEND_TOPOLOGY.md distinguishes
  source observations, candidates, failed probes and remaining Rust qualification.
- [ ] Root Rust helper tests/full trained replay; isolated implementation only.

````

## Source: .agents/runs/model-execution-cuda-cpu-bf16-projection-20261020/README.md

Original UTF-8 bytes: 10075; SHA256 `d9f32b7e1c867d933e115b7b4f6aa0773fe2b85219ec99664e043e90bd0e501c`.
Resolve embedded relative paths against the original path, not this volume.

````text
# CUDA BF16 projection: accepted CPU target

Author: @batch_design_review

Status: **source-only, uncompiled, unexecuted**. Parent: model-execution /
serving-and-speculation. ROOT owns all merge, compilation, operator execution,
physical acceptance and shared-plan reconciliation.

## Scope

- New private qualifier-only CUDA implementation of the existing
  `TorchCpuBf16Avx2MixedSourceV1` BF16 projection. No production selection change.
- Three new GPU files in `proposed-src/`; seven-path implementation delta includes
  the separate four-path `WIRING.patch`.
- No model/token loop, learned cache, SessionAuthority, StateRoot, checkpoint,
  HTTP mode or full-HF eligibility changes. No cuBLAS/vendor provider, foreign
  stream/context, direct CUDA driver launch, CPU execution fallback or tolerance.
- This targets the accepted **CPU** profile. It does not implement the different
  genuine vLLM CUDA BF16 profile or claim vLLM arithmetic/throughput parity.

## Closed mathematical contract

- Authentic sealed `VerifiedTensorSource` + `TensorWindow`; full unchanged
  `SourceTensor` descriptor, original BF16 OutputInput axes/name/shape/kind,
  original byte offset/length, exact transported row range/source offset and
  verified bundle. No reduced/fake tensor descriptor or provider region.
- Raw finite BF16 weights; explicit raw finite BF16 `[B,columns]` inputs. No casts
  of F32 learned sources. Input transport is not a trained-producer claim.
- Columns positive/divisible64, at most8192; output range1..16384 rows, at most
 32MiB raw weights; B exactly1/2/4/8. Refuse unsupported tails, dtypes, layouts,
  profiles, geometry and issuer before device work. Full original rows <=1M.
- One64-thread workgroup per `[request, original output row]`, ONE actual native
  launch for all B rows. Lane `register*8+lane` owns the corresponding existing
  CPU partial and sequentially applies `fma(weight,input,partial)` to columns
  `lane + 64*k`.
- Collapse partials64→32→16→8 exactly matches CPU register offsets4/2/1. Vec8
  horizontal order is `((v0+v4)+(v2+v6))+((v1+v5)+(v3+v7))`. Add the CPU empty
  tail's positive zero last. Final BF16 RNE uses unchanged HostMath bit formula;
  expose raw F32 accumulation AND BF16-widened F32 output, with per-output status.
- CubeCL's FAdd literal-zero simplifier drops `x+0` irrespective of zero sign.
  The candidate therefore reads an immutable, physically initialized **runtime
  +0** buffer for both the initial partials and mandatory final empty-tail add.
  This follows the original addition, not an output zero canonicalizer.
- Attribution/license copied from the existing CPU reducer. No new reduction
  schedule was inferred from cuBLAS or TensorCore behavior.

## Original ownership

- `PreparedProjection::prepare(existing Arc<ResidentService>, sealed weights,
  sealed inputs, pure correlation, cancellation)` accepts only the existing
  cold-pinned CUDA client/service. It never creates another engine/queue/context.
- Same-source RAM admission precedes private input copy, queued H2D copies and
  D2H arrays. SourceOwner retains original staged weights/input plus all those
  charges; output readback retains that owner until its last consumer drops.
- Original physical prepaid envelope, fair ticket/slot, initialization guard,
  six original BufferLeases and original submission/Ready/retirement only.
  Recheck capacity after fair waiting before allocation/H2D. Every actual payload
  has exact bytes/service/source/generation/alignment/unique lease preflight.
- GPU bytes: `2*rows*columns + 2*B*columns + 4 + 12*B*rows`. The final term covers
  raw accumulation, rounded output and status. Kernel shared storage512B.
- Status flags:1 nonfinite weight,2 nonfinite input,4 incorrect zero,8 nonfinite
  accumulation,16 nonfinite BF16 rounding. Cold finite validation precedes math;
  complete actual readback/status validation precedes the private Ready.
- `execute` consumes the prepared storage. Prior Ready outputs cannot be
  overwritten by a later operator invocation. No logical state publication.
- `CompletionOwner` attaches BEFORE first allocation/H2D or kernel. Every unknown
  path follows original armed init/submission Drop into existing quarantine.
  The only missing shared seam is the narrow init setter in `WIRING.patch`;
  it preserves the reviewed attach-before-ensure semantics.

## Prepared physical proof

`root-helper/` uses the existing admitted original source opener and the original
CPU attention observation validator. No Python math/Torch/model import.

For BOTH real original .8B/2B sources, actual layer3 q_proj/k_proj/v_proj, complete
4096/512/512 rows, teachers11→353:

1. Authenticate unchanged accepted source36-F32-restored CPU observations and
   raw BF16 hidden/projection archive spans through the existing parent validator.
2. Full source-file FD/digest before/after, existing original320-role config
   binding, source metadata and two4MiB binding charges, private input admission.
3. B1 teacher11, independent B1 teacher353, then real native B2/4/8 (alternating
   actual teacher inputs, no padded requests). Compare EVERY raw CUDA F32
   accumulation bit to the existing CPU window operator; EVERY final BF16 word
   to the untouched original public CPU projection archive.
4. Real original GPU pressure allocation leaves four bytes less than required;
   require exact prelaunch capacity refusal without H2D/extra lease/event. Check
   pre-prepare and initialized-but-unlaunched cancellation, wrong descriptor
   source/name/axes/full-row count; no release/fence bypass.
5. Exactly5 native math launches per matrix,15 per model,30 across both. Each
   model compares81920 raw accumulations AND81920 original public BF16 words.
   Root helper checks final host0; original qualifier checks each service/event
   and physical process-global budget final0. External validator closes the
   complete native receipt multiplicity and source/mode/width/lifecycle scope.
6. Separate fresh-process `abandon` submits a REAL kernel and drops its original
   flight without trusted completion. Require original fence, pending event,
   retained authentic source/window/leases and coarse envelope. No Ready or
   final-zero claim; process exit is not a logical reclamation receipt.

B2/4/8 here qualifies this stateless operator only. It does not prove independent
model-state batching, a complete HF layer/token, GPU attention/FLA or full logits.
ROOT separately reports actual CPU attention **Rust** fragment acceptance for both
models/all47104 words plus pressure/cancel/lifetime/final0 (signed fe2a342). This
candidate still compares genuine public archives and raw existing CPU accumulators;
that CPU result is not CUDA qualification.

## ROOT-only invocation

First review all files, compose the three new source files + `WIRING.patch` onto
MAIN with fresh mtimes, serialize the init setter with the graph work. `base-src/`
is exact before, `main-after/` is the private formatted four-file composition.
No vendor/actor/retirement changes are required.

```sh
RUN=/home/pvl/spaces/abird/src/zerogravity/.agents/runs/model-execution-cuda-cpu-bf16-projection-20261020
CARGO_BUILD_JOBS=2 RAYON_NUM_THREADS=4 python3 scripts/native-gpu.py -- \
  cargo clippy --offline -p zerogravity-gpu --no-default-features \
  --features cuda,qualification --lib --tests -- -D warnings
CARGO_BUILD_JOBS=2 RAYON_NUM_THREADS=4 python3 scripts/native-gpu.py -- \
  cargo test --offline -p zerogravity-gpu --no-default-features \
  --features cuda,qualification cpu_bf16_projection::tests -- --test-threads=1
python3 -m unittest discover -s "$RUN/root-helper" -p test_proof.py
CARGO_BUILD_JOBS=2 RAYON_NUM_THREADS=4 python3 scripts/native-gpu.py -- \
  cargo build --offline --release --manifest-path "$RUN/root-helper/Cargo.toml"
```

ROOT binds the fresh build/source/binary log and supplies its SHA; no source file
claims its own binary was built. Reuse the already authenticated raw extraction
from the attention replay or root-execute its unchanged exporter. ROOT selects
fresh names for output directories and actual input/binary SHA strings:

```sh
python3 scripts/native-gpu.py -- python3 "$RUN/root-helper/run_replay.py" \
  --binary "$BINARY" --binary-sha256 "$BINARY_SHA" \
  --inputs "$AUTHENTIC_INPUTS" --inputs-sha256 "$INPUTS_SHA" \
  --new-output "$RUN/root-normal-1" --mode normal
python3 "$RUN/root-helper/validate_replay.py" --output "$RUN/root-normal-1"
```

For uncertainty, same command with `--mode abandon --model 0.8b` and fresh output
under the run, then independent validator. Each child owns a fresh process; it
uses the existing GPU0 only. No launch on AMD. All work is serialized by ROOT.
The helper has900s child/3300s overall and4MiB combined-log bounds, exact four CPU
pinning, original reviewed unreaped-leader TERM/grace/KILL/reap/pipe cleanup. No
CUDA virtual address-space bound is misrepresented as resident-memory admission;
source/window/transport/output and original GPU buffers retain their real ledgers.

## Pending acceptance

- Source-only author rustfmt + AST parse only; NO build/test/operator execution.
- ROOT compilation/strict checks and genuine .8B/2B native CUDA all-row proof.
- Actual generated/compiled F32 FMA and addition schedule must retain gradual
  underflow and round-to-nearest; NVRTC source has no fast-math/ftz flag, but
  source inspection is not a compiled-code or physical arithmetic receipt.
- Before widening general numerical eligibility: meaningful physical signed-zero,
  subnormal, FMA-vs-product/add, register-order cancellation, RNE tie/overflow
  adversaries against the existing independently qualified CPU target. The two
  prepared Rust unit tests check bounds/transport only, not GPU arithmetic.
- Nonzero output-row window physical proof and future genuine27B width5120
  inputs remain open. The binding supports them; source-only support is no proof.
- Full HF GPU schedule/profile, state ownership integration, all42/fullV, vLLM
  CUDA math, matched benchmark and performance remain open. No speed claim.

````

## Source: .agents/runs/model-execution-cuda-cpu-bf16-projection-20261020/TODO.md

Original UTF-8 bytes: 1408; SHA256 `6c3fc183b22acec13acafcf974f6d2824296ce019d0cfe1638e2bb04ced48844`.
Resolve embedded relative paths against the original path, not this volume.

````text
# Working checklist

- Parent: model-execution; serving-and-speculation.
- Owner: @batch_design_review. Scope: isolated proposed source only; ROOT merges,
  tests/executes/reconciles. No MAIN tracked files or shared plans edited.
- [x] Audit exact existing CPU64 reducer and BF16 boundary; source/issuer/window
  binding and original CUDA service/queue/leases/Ready/quarantine.
- [x] Prepare native CUDA B1/2/4/8 kernel + closed source bindings and original
  resident initialization/submission wrappers; preserve original source dtypes.
- [x] Prepare real two-teacher dual-oracle/all-row qualification, pressure,
  prelaunch cancel and separate actual uncertain-completion quarantine proof.
- [x] Prepare original .8B/2B authenticated archive/source helper + finite owned
  process launcher and receipt coverage validator. No expected math generated.
- [x] Author rustfmt and Python AST parse only. No tests/build/runtime executed.
- [ ] ROOT independent review, compile/strict Clippy/source-free fixtures.
- [ ] ROOT actual physical native matrix qualification for both approved sources.
- [ ] ROOT actual abort/fence/quarantine process and normal global reservations0.
- [ ] Physical arithmetic adversaries, nonzero row-range and27B widths.
- [ ] Full model/HF source profile/session/math integration remains outside this
  candidate; no inference/performance/vLLM claim or production eligibility change.

````

## Source: .agents/runs/model-execution-cuda-cpu-bf16-projection-20261020/frozen/README.md

Original UTF-8 bytes: 10075; SHA256 `d9f32b7e1c867d933e115b7b4f6aa0773fe2b85219ec99664e043e90bd0e501c`.
Resolve embedded relative paths against the original path, not this volume.

````text
# CUDA BF16 projection: accepted CPU target

Author: @batch_design_review

Status: **source-only, uncompiled, unexecuted**. Parent: model-execution /
serving-and-speculation. ROOT owns all merge, compilation, operator execution,
physical acceptance and shared-plan reconciliation.

## Scope

- New private qualifier-only CUDA implementation of the existing
  `TorchCpuBf16Avx2MixedSourceV1` BF16 projection. No production selection change.
- Three new GPU files in `proposed-src/`; seven-path implementation delta includes
  the separate four-path `WIRING.patch`.
- No model/token loop, learned cache, SessionAuthority, StateRoot, checkpoint,
  HTTP mode or full-HF eligibility changes. No cuBLAS/vendor provider, foreign
  stream/context, direct CUDA driver launch, CPU execution fallback or tolerance.
- This targets the accepted **CPU** profile. It does not implement the different
  genuine vLLM CUDA BF16 profile or claim vLLM arithmetic/throughput parity.

## Closed mathematical contract

- Authentic sealed `VerifiedTensorSource` + `TensorWindow`; full unchanged
  `SourceTensor` descriptor, original BF16 OutputInput axes/name/shape/kind,
  original byte offset/length, exact transported row range/source offset and
  verified bundle. No reduced/fake tensor descriptor or provider region.
- Raw finite BF16 weights; explicit raw finite BF16 `[B,columns]` inputs. No casts
  of F32 learned sources. Input transport is not a trained-producer claim.
- Columns positive/divisible64, at most8192; output range1..16384 rows, at most
 32MiB raw weights; B exactly1/2/4/8. Refuse unsupported tails, dtypes, layouts,
  profiles, geometry and issuer before device work. Full original rows <=1M.
- One64-thread workgroup per `[request, original output row]`, ONE actual native
  launch for all B rows. Lane `register*8+lane` owns the corresponding existing
  CPU partial and sequentially applies `fma(weight,input,partial)` to columns
  `lane + 64*k`.
- Collapse partials64→32→16→8 exactly matches CPU register offsets4/2/1. Vec8
  horizontal order is `((v0+v4)+(v2+v6))+((v1+v5)+(v3+v7))`. Add the CPU empty
  tail's positive zero last. Final BF16 RNE uses unchanged HostMath bit formula;
  expose raw F32 accumulation AND BF16-widened F32 output, with per-output status.
- CubeCL's FAdd literal-zero simplifier drops `x+0` irrespective of zero sign.
  The candidate therefore reads an immutable, physically initialized **runtime
  +0** buffer for both the initial partials and mandatory final empty-tail add.
  This follows the original addition, not an output zero canonicalizer.
- Attribution/license copied from the existing CPU reducer. No new reduction
  schedule was inferred from cuBLAS or TensorCore behavior.

## Original ownership

- `PreparedProjection::prepare(existing Arc<ResidentService>, sealed weights,
  sealed inputs, pure correlation, cancellation)` accepts only the existing
  cold-pinned CUDA client/service. It never creates another engine/queue/context.
- Same-source RAM admission precedes private input copy, queued H2D copies and
  D2H arrays. SourceOwner retains original staged weights/input plus all those
  charges; output readback retains that owner until its last consumer drops.
- Original physical prepaid envelope, fair ticket/slot, initialization guard,
  six original BufferLeases and original submission/Ready/retirement only.
  Recheck capacity after fair waiting before allocation/H2D. Every actual payload
  has exact bytes/service/source/generation/alignment/unique lease preflight.
- GPU bytes: `2*rows*columns + 2*B*columns + 4 + 12*B*rows`. The final term covers
  raw accumulation, rounded output and status. Kernel shared storage512B.
- Status flags:1 nonfinite weight,2 nonfinite input,4 incorrect zero,8 nonfinite
  accumulation,16 nonfinite BF16 rounding. Cold finite validation precedes math;
  complete actual readback/status validation precedes the private Ready.
- `execute` consumes the prepared storage. Prior Ready outputs cannot be
  overwritten by a later operator invocation. No logical state publication.
- `CompletionOwner` attaches BEFORE first allocation/H2D or kernel. Every unknown
  path follows original armed init/submission Drop into existing quarantine.
  The only missing shared seam is the narrow init setter in `WIRING.patch`;
  it preserves the reviewed attach-before-ensure semantics.

## Prepared physical proof

`root-helper/` uses the existing admitted original source opener and the original
CPU attention observation validator. No Python math/Torch/model import.

For BOTH real original .8B/2B sources, actual layer3 q_proj/k_proj/v_proj, complete
4096/512/512 rows, teachers11→353:

1. Authenticate unchanged accepted source36-F32-restored CPU observations and
   raw BF16 hidden/projection archive spans through the existing parent validator.
2. Full source-file FD/digest before/after, existing original320-role config
   binding, source metadata and two4MiB binding charges, private input admission.
3. B1 teacher11, independent B1 teacher353, then real native B2/4/8 (alternating
   actual teacher inputs, no padded requests). Compare EVERY raw CUDA F32
   accumulation bit to the existing CPU window operator; EVERY final BF16 word
   to the untouched original public CPU projection archive.
4. Real original GPU pressure allocation leaves four bytes less than required;
   require exact prelaunch capacity refusal without H2D/extra lease/event. Check
   pre-prepare and initialized-but-unlaunched cancellation, wrong descriptor
   source/name/axes/full-row count; no release/fence bypass.
5. Exactly5 native math launches per matrix,15 per model,30 across both. Each
   model compares81920 raw accumulations AND81920 original public BF16 words.
   Root helper checks final host0; original qualifier checks each service/event
   and physical process-global budget final0. External validator closes the
   complete native receipt multiplicity and source/mode/width/lifecycle scope.
6. Separate fresh-process `abandon` submits a REAL kernel and drops its original
   flight without trusted completion. Require original fence, pending event,
   retained authentic source/window/leases and coarse envelope. No Ready or
   final-zero claim; process exit is not a logical reclamation receipt.

B2/4/8 here qualifies this stateless operator only. It does not prove independent
model-state batching, a complete HF layer/token, GPU attention/FLA or full logits.
ROOT separately reports actual CPU attention **Rust** fragment acceptance for both
models/all47104 words plus pressure/cancel/lifetime/final0 (signed fe2a342). This
candidate still compares genuine public archives and raw existing CPU accumulators;
that CPU result is not CUDA qualification.

## ROOT-only invocation

First review all files, compose the three new source files + `WIRING.patch` onto
MAIN with fresh mtimes, serialize the init setter with the graph work. `base-src/`
is exact before, `main-after/` is the private formatted four-file composition.
No vendor/actor/retirement changes are required.

```sh
RUN=/home/pvl/spaces/abird/src/zerogravity/.agents/runs/model-execution-cuda-cpu-bf16-projection-20261020
CARGO_BUILD_JOBS=2 RAYON_NUM_THREADS=4 python3 scripts/native-gpu.py -- \
  cargo clippy --offline -p zerogravity-gpu --no-default-features \
  --features cuda,qualification --lib --tests -- -D warnings
CARGO_BUILD_JOBS=2 RAYON_NUM_THREADS=4 python3 scripts/native-gpu.py -- \
  cargo test --offline -p zerogravity-gpu --no-default-features \
  --features cuda,qualification cpu_bf16_projection::tests -- --test-threads=1
python3 -m unittest discover -s "$RUN/root-helper" -p test_proof.py
CARGO_BUILD_JOBS=2 RAYON_NUM_THREADS=4 python3 scripts/native-gpu.py -- \
  cargo build --offline --release --manifest-path "$RUN/root-helper/Cargo.toml"
```

ROOT binds the fresh build/source/binary log and supplies its SHA; no source file
claims its own binary was built. Reuse the already authenticated raw extraction
from the attention replay or root-execute its unchanged exporter. ROOT selects
fresh names for output directories and actual input/binary SHA strings:

```sh
python3 scripts/native-gpu.py -- python3 "$RUN/root-helper/run_replay.py" \
  --binary "$BINARY" --binary-sha256 "$BINARY_SHA" \
  --inputs "$AUTHENTIC_INPUTS" --inputs-sha256 "$INPUTS_SHA" \
  --new-output "$RUN/root-normal-1" --mode normal
python3 "$RUN/root-helper/validate_replay.py" --output "$RUN/root-normal-1"
```

For uncertainty, same command with `--mode abandon --model 0.8b` and fresh output
under the run, then independent validator. Each child owns a fresh process; it
uses the existing GPU0 only. No launch on AMD. All work is serialized by ROOT.
The helper has900s child/3300s overall and4MiB combined-log bounds, exact four CPU
pinning, original reviewed unreaped-leader TERM/grace/KILL/reap/pipe cleanup. No
CUDA virtual address-space bound is misrepresented as resident-memory admission;
source/window/transport/output and original GPU buffers retain their real ledgers.

## Pending acceptance

- Source-only author rustfmt + AST parse only; NO build/test/operator execution.
- ROOT compilation/strict checks and genuine .8B/2B native CUDA all-row proof.
- Actual generated/compiled F32 FMA and addition schedule must retain gradual
  underflow and round-to-nearest; NVRTC source has no fast-math/ftz flag, but
  source inspection is not a compiled-code or physical arithmetic receipt.
- Before widening general numerical eligibility: meaningful physical signed-zero,
  subnormal, FMA-vs-product/add, register-order cancellation, RNE tie/overflow
  adversaries against the existing independently qualified CPU target. The two
  prepared Rust unit tests check bounds/transport only, not GPU arithmetic.
- Nonzero output-row window physical proof and future genuine27B width5120
  inputs remain open. The binding supports them; source-only support is no proof.
- Full HF GPU schedule/profile, state ownership integration, all42/fullV, vLLM
  CUDA math, matched benchmark and performance remain open. No speed claim.

````

## Source: .agents/runs/model-execution-cuda-cpu-bf16-projection-20261020/frozen/TODO.md

Original UTF-8 bytes: 1408; SHA256 `6c3fc183b22acec13acafcf974f6d2824296ce019d0cfe1638e2bb04ced48844`.
Resolve embedded relative paths against the original path, not this volume.

````text
# Working checklist

- Parent: model-execution; serving-and-speculation.
- Owner: @batch_design_review. Scope: isolated proposed source only; ROOT merges,
  tests/executes/reconciles. No MAIN tracked files or shared plans edited.
- [x] Audit exact existing CPU64 reducer and BF16 boundary; source/issuer/window
  binding and original CUDA service/queue/leases/Ready/quarantine.
- [x] Prepare native CUDA B1/2/4/8 kernel + closed source bindings and original
  resident initialization/submission wrappers; preserve original source dtypes.
- [x] Prepare real two-teacher dual-oracle/all-row qualification, pressure,
  prelaunch cancel and separate actual uncertain-completion quarantine proof.
- [x] Prepare original .8B/2B authenticated archive/source helper + finite owned
  process launcher and receipt coverage validator. No expected math generated.
- [x] Author rustfmt and Python AST parse only. No tests/build/runtime executed.
- [ ] ROOT independent review, compile/strict Clippy/source-free fixtures.
- [ ] ROOT actual physical native matrix qualification for both approved sources.
- [ ] ROOT actual abort/fence/quarantine process and normal global reservations0.
- [ ] Physical arithmetic adversaries, nonzero row-range and27B widths.
- [ ] Full model/HF source profile/session/math integration remains outside this
  candidate; no inference/performance/vLLM claim or production eligibility change.

````

## Source: .agents/runs/model-execution-cuda-cpu-bf16-projection-adversary-20261020/README.md

Original UTF-8 bytes: 6844; SHA256 `79739cffe6db8ef648df8779ec3566134b24da39f4dccce9e310bb13f185d604`.
Resolve embedded relative paths against the original path, not this volume.

````text
# CUDA CPU-profile BF16 adversaries — prepared only

Author: batch_design_review

- Parent: `.agents/plans/serving-and-speculation/TODO.md` + original-HF shared-engine work.
- Source-only additive qualification. No author compilation, tests, arithmetic,
  model, CUDA, container or runtime execution.
- Baseline b5decd62 / 679d8690 remains immutable. ROOT V3 a30e1ba6 is a
  dependency: preserve its complete CPU-reference-length and actual quarantine
  completion-owner assertions. This delta does not replace that child qualifier.
- All expected finite accumulations come from the unchanged original
  `original_hf::window_projection::project_bf16_window` CPU64 operator.
- Target `TorchCpuBf16Avx2MixedSourceV1`; distinct from actual vLLM CUDA GEMM.
  No full-HF, trained-model, vLLM CUDA, throughput or profile promotion.

## New scope

- One new private qualifier child `cpu_bf16_projection/adversary.rs`.
- Two minimal module/public qualification wiring additions. No original kernel,
  reducer, parameter codec, production bindings or resident-service changes.
- Explicitly synthetic Safetensors, generated by ROOT in a fresh private
  attempt: 108 original five-row BF16 matrices, full descriptor/offset/index
  identity verified through existing `VerifiedTensorSource` and `TensorWindow`.
  Selected nonzero original rows `1..4`; never a fake reduced source descriptor.
- Normal: 328 source-certified original kernel launch sites, 3,690 raw F32 and
  3,690 BF16-rounded words; widths64/128/1024/2048/5120/8192 and B1/2/4/8.
  These are launch-site receipts, not driver-enumerated kernel counts.
- Arithmetic cases: signed-zero transport, final empty-tail +0, gradual positive
  and negative BF16/F32 subnormals, CPU64 horizontal cancellation, normal and
  subnormal RNE ties (both normal signs), fused underflow and overflow
  cancellation. Member0 retains each analytic witness; other members have
  distinct finite marker outputs, detecting row broadcast/aliasing.
- Cold: 12 authenticated nonfinite-source refusals, 12 explicit nonfinite-input
  refusals, real authenticated source widths65/8193, and B3 refusal at all six
  accepted widths. They must allocate/upload no GPU buffers.
- Separate fresh fault processes: finite-input F32 overflow or finite
  accumulation whose BF16 rounding overflows. Same six guarded physical leases,
  exact status and readonly bits checked before refusing `Ready`; original
  source/host/device/completion owners remain fenced/quarantined. No final-zero
  claim or further device operation in these processes.

## Physical ownership and independent gates

- Six original leased buffers; 16-byte prefix and suffix canaries on every
  binding, with exact interior subviews. All 48 canary words and every readonly
  source/input/runtime-zero byte are read back. Write payloads start poisoned.
- Original fair physical queue, pinned original CUDA actor/client, source issuer,
  `ResidentInitialization`, `ResidentSubmission`, completion owner, event,
  ledger and quarantine. RAM and device staging/workspaces admitted before
  private payload creation; original window retained through terminal work.
- Before private `Ready`, compare every F32 accumulation, rounded F32 word and
  status against original CPU output. Synthetic analytic witnesses additionally
  check that chosen recipes discriminate the intended arithmetic.
- Successful cases require real output/event retirement and final original GPU
  reservation0. The normal helper also requires host admission0 after source
  drop. Fault cases require the opposite: genuine retained completion owner,
  pending event, device fence and retained source RAM; no `Ready`.
- Owned launcher: reviewed byte-identical cleanup functions, unreaped leader
  identity, independently bounded TERM/KILL/reap stages, PGID absence and pipe
  EOF/close. All post-spawn paths enter cleanup; combined log4MiB, wall900s,
  four explicit CPUs. Runtime duration is not a benchmark.
- Validator independently enforces exact complete case/band receipt multiset,
  source full SHA/FD identity/index/header/EOF, actual affinity, closure and
  cleanup. Ten prepared source-free Python negatives and three prepared Rust
  transport tests are not numerical qualification.

## ROOT application and checks

1. Independently review `NEW_DELTA.patch`, all new source/helper files and
   `freeze-manifest.json`. Apply only this delta to the ROOT V3 baseline with
   fresh timestamps. `main_after_merge` pins all 372 intended shared sources;
   `baseline_files` pins original V1 frozen preparation, not a replacement tree.
2. ROOT source-free fixtures: `python3 root-helper/test_receipts.py`.
3. ROOT serial strict Clippy and focused GPU-crate unit tests with
   `--no-default-features --features cuda,qualification`; retain any failures.
4. ROOT builds the helper in release, using original native isolation wrapper,
   offline dependency lock generation and same native CUDA environment as the
   base qualifier. Source pin closure and actual build/binary attestation are
   prerequisites; a supplied binary digest alone is not compilation evidence.

```bash
CARGO_BUILD_JOBS=2 RAYON_NUM_THREADS=4 python3 scripts/native-gpu.py -- \
  cargo build --release --offline --manifest-path \
  .agents/runs/model-execution-cuda-cpu-bf16-projection-adversary-20261020/root-helper/Cargo.toml
```

5. Let `RUN` be this absolute directory and `BINARY` the actual attested helper.
   ROOT executes each fresh output under the same native isolation wrapper;
   `--binary-sha256` must be the actual post-build executable SHA.

```bash
python3 "$RUN/root-helper/run_replay.py" --binary "$BINARY" \
  --binary-sha256 "$BINARY_SHA" --new-output "$RUN/root-normal-1"
python3 "$RUN/root-helper/validate_replay.py" "$RUN/root-normal-1"
```

6. For each B1/2/4/8, repeat launcher+validator in a separate fresh process/output
   with `--mode fp32-overflow --batch B`, then `bf16-overflow`. A fenced fault
   process is never reused. Do not accept normal-only receipts as status proof.
7. Preserve first failures and executed source bytes. Any correction is a fresh
   additive freeze. No full-model acceptance or performance assertion follows
   from this generic operator gate; trained source/base qualification and all
   integration gates remain required.

## Remaining scope

- ROOT source review, compilation/Clippy/unit/receipt tests and all physical
  normal/fault proofs remain unexecuted by the author.
- Compiled floating-point contraction/FTZ behavior must be decided by the actual
  accepted CPU64-to-CUDA adversaries; no tolerance or signed-zero exceptions.
- No random/large-shape exhaustive arithmetic proof, 27B native-model support,
  all42 full-HF graph/state evidence, cancellation integration or performance
  claim. Existing production refusal/profile guards remain closed.

````

## Source: .agents/runs/model-execution-cuda-cpu-bf16-projection-adversary-20261020/TODO.md

Original UTF-8 bytes: 1181; SHA256 `3eadae3992fe4bf78fdb4f67e45f0089c1fdb9067b604252eef7b5a1af50190e`.
Resolve embedded relative paths against the original path, not this volume.

````text
# Working checklist

Parent: `.agents/plans/serving-and-speculation/TODO.md` / shared original-HF engine

Owner: batch_design_review

- [x] Keep original b5decd62/679d8690 baseline immutable; retain ROOT V3 child
  qualifier defensive coverage/quarantine changes as dependency.
- [x] Prepare one same-kernel guarded physical qualifier, source-authenticated
  synthetic fixtures, CPU64 numerical oracle, cold refusals and fresh overflow
  processes. No alternate reducer or trained expected tensors.
- [x] Prepare original-source generator/helper, owned finite launcher,
  independent receipt/source/cleanup validator and source-free negative tests.
- [x] Source-review owned code; AST parse and rustfmt only, no test execution.
- [ ] ROOT independent source review and exact additive merge.
- [ ] ROOT focused strict/unit/receipt checks, release build with actual attestation.
- [ ] ROOT actual CUDA 328-case normal and eight separate status-fault runs.
- [ ] Resolve any actual arithmetic/status/owner failure causally, without
  tolerance, signed-zero exceptions, CPU fallback or profile changes.
- [ ] Trained-model/integration/full-HF/all42/performance remain separately open.

````

## Source: .agents/runs/model-execution-cuda-cpu-bf16-projection-adversary-20261020/frozen/README.md

Original UTF-8 bytes: 6844; SHA256 `79739cffe6db8ef648df8779ec3566134b24da39f4dccce9e310bb13f185d604`.
Resolve embedded relative paths against the original path, not this volume.

````text
# CUDA CPU-profile BF16 adversaries — prepared only

Author: batch_design_review

- Parent: `.agents/plans/serving-and-speculation/TODO.md` + original-HF shared-engine work.
- Source-only additive qualification. No author compilation, tests, arithmetic,
  model, CUDA, container or runtime execution.
- Baseline b5decd62 / 679d8690 remains immutable. ROOT V3 a30e1ba6 is a
  dependency: preserve its complete CPU-reference-length and actual quarantine
  completion-owner assertions. This delta does not replace that child qualifier.
- All expected finite accumulations come from the unchanged original
  `original_hf::window_projection::project_bf16_window` CPU64 operator.
- Target `TorchCpuBf16Avx2MixedSourceV1`; distinct from actual vLLM CUDA GEMM.
  No full-HF, trained-model, vLLM CUDA, throughput or profile promotion.

## New scope

- One new private qualifier child `cpu_bf16_projection/adversary.rs`.
- Two minimal module/public qualification wiring additions. No original kernel,
  reducer, parameter codec, production bindings or resident-service changes.
- Explicitly synthetic Safetensors, generated by ROOT in a fresh private
  attempt: 108 original five-row BF16 matrices, full descriptor/offset/index
  identity verified through existing `VerifiedTensorSource` and `TensorWindow`.
  Selected nonzero original rows `1..4`; never a fake reduced source descriptor.
- Normal: 328 source-certified original kernel launch sites, 3,690 raw F32 and
  3,690 BF16-rounded words; widths64/128/1024/2048/5120/8192 and B1/2/4/8.
  These are launch-site receipts, not driver-enumerated kernel counts.
- Arithmetic cases: signed-zero transport, final empty-tail +0, gradual positive
  and negative BF16/F32 subnormals, CPU64 horizontal cancellation, normal and
  subnormal RNE ties (both normal signs), fused underflow and overflow
  cancellation. Member0 retains each analytic witness; other members have
  distinct finite marker outputs, detecting row broadcast/aliasing.
- Cold: 12 authenticated nonfinite-source refusals, 12 explicit nonfinite-input
  refusals, real authenticated source widths65/8193, and B3 refusal at all six
  accepted widths. They must allocate/upload no GPU buffers.
- Separate fresh fault processes: finite-input F32 overflow or finite
  accumulation whose BF16 rounding overflows. Same six guarded physical leases,
  exact status and readonly bits checked before refusing `Ready`; original
  source/host/device/completion owners remain fenced/quarantined. No final-zero
  claim or further device operation in these processes.

## Physical ownership and independent gates

- Six original leased buffers; 16-byte prefix and suffix canaries on every
  binding, with exact interior subviews. All 48 canary words and every readonly
  source/input/runtime-zero byte are read back. Write payloads start poisoned.
- Original fair physical queue, pinned original CUDA actor/client, source issuer,
  `ResidentInitialization`, `ResidentSubmission`, completion owner, event,
  ledger and quarantine. RAM and device staging/workspaces admitted before
  private payload creation; original window retained through terminal work.
- Before private `Ready`, compare every F32 accumulation, rounded F32 word and
  status against original CPU output. Synthetic analytic witnesses additionally
  check that chosen recipes discriminate the intended arithmetic.
- Successful cases require real output/event retirement and final original GPU
  reservation0. The normal helper also requires host admission0 after source
  drop. Fault cases require the opposite: genuine retained completion owner,
  pending event, device fence and retained source RAM; no `Ready`.
- Owned launcher: reviewed byte-identical cleanup functions, unreaped leader
  identity, independently bounded TERM/KILL/reap stages, PGID absence and pipe
  EOF/close. All post-spawn paths enter cleanup; combined log4MiB, wall900s,
  four explicit CPUs. Runtime duration is not a benchmark.
- Validator independently enforces exact complete case/band receipt multiset,
  source full SHA/FD identity/index/header/EOF, actual affinity, closure and
  cleanup. Ten prepared source-free Python negatives and three prepared Rust
  transport tests are not numerical qualification.

## ROOT application and checks

1. Independently review `NEW_DELTA.patch`, all new source/helper files and
   `freeze-manifest.json`. Apply only this delta to the ROOT V3 baseline with
   fresh timestamps. `main_after_merge` pins all 372 intended shared sources;
   `baseline_files` pins original V1 frozen preparation, not a replacement tree.
2. ROOT source-free fixtures: `python3 root-helper/test_receipts.py`.
3. ROOT serial strict Clippy and focused GPU-crate unit tests with
   `--no-default-features --features cuda,qualification`; retain any failures.
4. ROOT builds the helper in release, using original native isolation wrapper,
   offline dependency lock generation and same native CUDA environment as the
   base qualifier. Source pin closure and actual build/binary attestation are
   prerequisites; a supplied binary digest alone is not compilation evidence.

```bash
CARGO_BUILD_JOBS=2 RAYON_NUM_THREADS=4 python3 scripts/native-gpu.py -- \
  cargo build --release --offline --manifest-path \
  .agents/runs/model-execution-cuda-cpu-bf16-projection-adversary-20261020/root-helper/Cargo.toml
```

5. Let `RUN` be this absolute directory and `BINARY` the actual attested helper.
   ROOT executes each fresh output under the same native isolation wrapper;
   `--binary-sha256` must be the actual post-build executable SHA.

```bash
python3 "$RUN/root-helper/run_replay.py" --binary "$BINARY" \
  --binary-sha256 "$BINARY_SHA" --new-output "$RUN/root-normal-1"
python3 "$RUN/root-helper/validate_replay.py" "$RUN/root-normal-1"
```

6. For each B1/2/4/8, repeat launcher+validator in a separate fresh process/output
   with `--mode fp32-overflow --batch B`, then `bf16-overflow`. A fenced fault
   process is never reused. Do not accept normal-only receipts as status proof.
7. Preserve first failures and executed source bytes. Any correction is a fresh
   additive freeze. No full-model acceptance or performance assertion follows
   from this generic operator gate; trained source/base qualification and all
   integration gates remain required.

## Remaining scope

- ROOT source review, compilation/Clippy/unit/receipt tests and all physical
  normal/fault proofs remain unexecuted by the author.
- Compiled floating-point contraction/FTZ behavior must be decided by the actual
  accepted CPU64-to-CUDA adversaries; no tolerance or signed-zero exceptions.
- No random/large-shape exhaustive arithmetic proof, 27B native-model support,
  all42 full-HF graph/state evidence, cancellation integration or performance
  claim. Existing production refusal/profile guards remain closed.

````

## Source: .agents/runs/model-execution-cuda-cpu-bf16-projection-adversary-20261020/frozen/TODO.md

Original UTF-8 bytes: 1181; SHA256 `3eadae3992fe4bf78fdb4f67e45f0089c1fdb9067b604252eef7b5a1af50190e`.
Resolve embedded relative paths against the original path, not this volume.

````text
# Working checklist

Parent: `.agents/plans/serving-and-speculation/TODO.md` / shared original-HF engine

Owner: batch_design_review

- [x] Keep original b5decd62/679d8690 baseline immutable; retain ROOT V3 child
  qualifier defensive coverage/quarantine changes as dependency.
- [x] Prepare one same-kernel guarded physical qualifier, source-authenticated
  synthetic fixtures, CPU64 numerical oracle, cold refusals and fresh overflow
  processes. No alternate reducer or trained expected tensors.
- [x] Prepare original-source generator/helper, owned finite launcher,
  independent receipt/source/cleanup validator and source-free negative tests.
- [x] Source-review owned code; AST parse and rustfmt only, no test execution.
- [ ] ROOT independent source review and exact additive merge.
- [ ] ROOT focused strict/unit/receipt checks, release build with actual attestation.
- [ ] ROOT actual CUDA 328-case normal and eight separate status-fault runs.
- [ ] Resolve any actual arithmetic/status/owner failure causally, without
  tolerance, signed-zero exceptions, CPU fallback or profile changes.
- [ ] Trained-model/integration/full-HF/all42/performance remain separately open.

````

## Source: .agents/runs/model-execution-cuda-cpu-bf16-projection-adversary-root-v2-20261020/README.md

Original UTF-8 bytes: 7005; SHA256 `c8d6f71088dff242f50376130443028728be17c769269f5785f8df3213ffb21a`.
Resolve embedded relative paths against the original path, not this volume.

````text
# CUDA CPU-profile BF16 adversaries — prepared only

Author: batch_design_review

- Parent: `.agents/plans/serving-and-speculation/TODO.md` + original-HF shared-engine work.
- Source-only additive qualification. No author compilation, tests, arithmetic,
  model, CUDA, container or runtime execution.
- Baseline b5decd62 / 679d8690 remains immutable. ROOT V3 a30e1ba6 is a
  dependency: preserve its complete CPU-reference-length and actual quarantine
  completion-owner assertions. This delta does not replace that child qualifier.
- All expected finite accumulations come from the unchanged original
  `original_hf::window_projection::project_bf16_window` CPU64 operator.
- Target `TorchCpuBf16Avx2MixedSourceV1`; distinct from actual vLLM CUDA GEMM.
  No full-HF, trained-model, vLLM CUDA, throughput or profile promotion.

## New scope

- One new private qualifier child `cpu_bf16_projection/adversary.rs`.
- Two minimal module/public qualification wiring additions. No original kernel,
  reducer, parameter codec, production bindings or resident-service changes.
- Explicitly synthetic Safetensors, generated by ROOT in a fresh private
  attempt: 108 original five-row BF16 matrices, full descriptor/offset/index
  identity verified through existing `VerifiedTensorSource` and `TensorWindow`.
  Selected nonzero original rows `1..4`; never a fake reduced source descriptor.
- Normal: 328 source-certified original kernel launch sites, 3,690 raw F32 and
  3,690 BF16-rounded words; widths64/128/1024/2048/5120/8192 and B1/2/4/8.
  These are launch-site receipts, not driver-enumerated kernel counts.
- Arithmetic cases: signed-zero transport, final empty-tail +0, gradual positive
  and negative BF16/F32 subnormals, CPU64 horizontal cancellation, normal and
  subnormal RNE ties (both normal signs), fused underflow and overflow
  cancellation. Member0 retains each analytic witness; other members have
  distinct finite marker outputs, detecting row broadcast/aliasing.
- Cold: 12 authenticated nonfinite-source refusals, 12 explicit nonfinite-input
  refusals, real authenticated source widths65/8193, and B3 refusal at all six
  accepted widths. They must allocate/upload no GPU buffers.
- Separate fresh fault processes: finite-input F32 overflow or finite
  accumulation whose BF16 rounding overflows. Same six guarded physical leases,
  exact status and readonly bits checked before refusing `Ready`; original
  source/host/device/completion owners remain fenced/quarantined. No final-zero
  claim or further device operation in these processes.

## Physical ownership and independent gates

- Six original leased buffers; 16-byte prefix and suffix canaries on every
  binding, with exact interior subviews. All 48 canary words and every readonly
  source/input/runtime-zero byte are read back. Write payloads start poisoned.
- Original fair physical queue, pinned original CUDA actor/client, source issuer,
  `ResidentInitialization`, `ResidentSubmission`, completion owner, event,
  ledger and quarantine. RAM and device staging/workspaces admitted before
  private payload creation; original window retained through terminal work.
- Before private `Ready`, compare every F32 accumulation, rounded F32 word and
  status against original CPU output. Synthetic analytic witnesses additionally
  check that chosen recipes discriminate the intended arithmetic.
- Successful cases require real output/event retirement and final original GPU
  reservation0. The normal helper also requires host admission0 after source
  drop. Fault cases require the opposite: genuine retained completion owner,
  pending event, device fence and retained source RAM; no `Ready`.
- Owned launcher: reviewed byte-identical cleanup functions, unreaped leader
  identity, independently bounded TERM/KILL/reap stages, PGID absence and pipe
  EOF/close. All post-spawn paths enter cleanup; combined log4MiB, wall900s,
  four explicit CPUs. Runtime duration is not a benchmark.
- Validator independently enforces exact complete case/band receipt multiset,
  source full SHA/FD identity/index/header/EOF, actual affinity, closure and
  cleanup. Ten prepared source-free Python negatives and three prepared Rust
  transport tests are not numerical qualification.

## ROOT application and checks

1. Independently review `NEW_DELTA.patch`, all new source/helper files and
   `freeze-manifest.json`. Apply only this delta to the ROOT V3 baseline with
   fresh timestamps. `main_after_merge` pins all 372 intended shared sources;
   `baseline_files` pins original V1 frozen preparation, not a replacement tree.
2. ROOT source-free fixtures: `python3 root-helper/test_receipts.py`.
3. ROOT serial strict Clippy and focused GPU-crate unit tests with
   `--no-default-features --features cuda,qualification`; retain any failures.
4. ROOT builds the helper in release, using original native isolation wrapper,
   offline dependency lock generation and same native CUDA environment as the
   base qualifier. Source pin closure and actual build/binary attestation are
   prerequisites; a supplied binary digest alone is not compilation evidence.

```bash
CARGO_BUILD_JOBS=2 RAYON_NUM_THREADS=4 python3 scripts/native-gpu.py -- \
  cargo build --release --offline --manifest-path \
  .agents/runs/model-execution-cuda-cpu-bf16-projection-adversary-20261020/root-helper/Cargo.toml
```

5. Let `RUN` be this absolute directory and `BINARY` the actual attested helper.
   ROOT executes each fresh output under the same native isolation wrapper;
   `--binary-sha256` must be the actual post-build executable SHA.

```bash
python3 "$RUN/root-helper/run_replay.py" --binary "$BINARY" \
  --binary-sha256 "$BINARY_SHA" --new-output "$RUN/root-normal-1"
python3 "$RUN/root-helper/validate_replay.py" "$RUN/root-normal-1"
```

6. For each B1/2/4/8, repeat launcher+validator in a separate fresh process/output
   with `--mode fp32-overflow --batch B`, then `bf16-overflow`. A fenced fault
   process is never reused. Do not accept normal-only receipts as status proof.
7. Preserve first failures and executed source bytes. Any correction is a fresh
   additive freeze. No full-model acceptance or performance assertion follows
   from this generic operator gate; trained source/base qualification and all
   integration gates remain required.

## Remaining scope

- ROOT source review, compilation/Clippy/unit/receipt tests and all physical
  normal/fault proofs remain unexecuted by the author.
- Compiled floating-point contraction/FTZ behavior must be decided by the actual
  accepted CPU64-to-CUDA adversaries; no tolerance or signed-zero exceptions.
- No random/large-shape exhaustive arithmetic proof, 27B native-model support,
  all42 full-HF graph/state evidence, cancellation integration or performance
  claim. Existing production refusal/profile guards remain closed.

ROOT V2: retain original E0283 strict compile failure; make readonly readback comparison explicitly &[u8] via &actual[..]. No arithmetic/kernel/control change.

````

## Source: .agents/runs/model-execution-cuda-cpu-bf16-projection-adversary-root-v2-20261020/TODO.md

Original UTF-8 bytes: 1181; SHA256 `3eadae3992fe4bf78fdb4f67e45f0089c1fdb9067b604252eef7b5a1af50190e`.
Resolve embedded relative paths against the original path, not this volume.

````text
# Working checklist

Parent: `.agents/plans/serving-and-speculation/TODO.md` / shared original-HF engine

Owner: batch_design_review

- [x] Keep original b5decd62/679d8690 baseline immutable; retain ROOT V3 child
  qualifier defensive coverage/quarantine changes as dependency.
- [x] Prepare one same-kernel guarded physical qualifier, source-authenticated
  synthetic fixtures, CPU64 numerical oracle, cold refusals and fresh overflow
  processes. No alternate reducer or trained expected tensors.
- [x] Prepare original-source generator/helper, owned finite launcher,
  independent receipt/source/cleanup validator and source-free negative tests.
- [x] Source-review owned code; AST parse and rustfmt only, no test execution.
- [ ] ROOT independent source review and exact additive merge.
- [ ] ROOT focused strict/unit/receipt checks, release build with actual attestation.
- [ ] ROOT actual CUDA 328-case normal and eight separate status-fault runs.
- [ ] Resolve any actual arithmetic/status/owner failure causally, without
  tolerance, signed-zero exceptions, CPU fallback or profile changes.
- [ ] Trained-model/integration/full-HF/all42/performance remain separately open.

````

## Source: .agents/runs/model-execution-cuda-cpu-bf16-projection-adversary-root-v3-20261020/README.md

Original UTF-8 bytes: 7124; SHA256 `6c569ee6bca15f3b4738118edb6b1c9bfe9d84f888c9efdbf8abc9e8b41cd642`.
Resolve embedded relative paths against the original path, not this volume.

````text
# CUDA CPU-profile BF16 adversaries — prepared only

Author: batch_design_review

- Parent: `.agents/plans/serving-and-speculation/TODO.md` + original-HF shared-engine work.
- Source-only additive qualification. No author compilation, tests, arithmetic,
  model, CUDA, container or runtime execution.
- Baseline b5decd62 / 679d8690 remains immutable. ROOT V3 a30e1ba6 is a
  dependency: preserve its complete CPU-reference-length and actual quarantine
  completion-owner assertions. This delta does not replace that child qualifier.
- All expected finite accumulations come from the unchanged original
  `original_hf::window_projection::project_bf16_window` CPU64 operator.
- Target `TorchCpuBf16Avx2MixedSourceV1`; distinct from actual vLLM CUDA GEMM.
  No full-HF, trained-model, vLLM CUDA, throughput or profile promotion.

## New scope

- One new private qualifier child `cpu_bf16_projection/adversary.rs`.
- Two minimal module/public qualification wiring additions. No original kernel,
  reducer, parameter codec, production bindings or resident-service changes.
- Explicitly synthetic Safetensors, generated by ROOT in a fresh private
  attempt: 108 original five-row BF16 matrices, full descriptor/offset/index
  identity verified through existing `VerifiedTensorSource` and `TensorWindow`.
  Selected nonzero original rows `1..4`; never a fake reduced source descriptor.
- Normal: 328 source-certified original kernel launch sites, 3,690 raw F32 and
  3,690 BF16-rounded words; widths64/128/1024/2048/5120/8192 and B1/2/4/8.
  These are launch-site receipts, not driver-enumerated kernel counts.
- Arithmetic cases: signed-zero transport, final empty-tail +0, gradual positive
  and negative BF16/F32 subnormals, CPU64 horizontal cancellation, normal and
  subnormal RNE ties (both normal signs), fused underflow and overflow
  cancellation. Member0 retains each analytic witness; other members have
  distinct finite marker outputs, detecting row broadcast/aliasing.
- Cold: 12 authenticated nonfinite-source refusals, 12 explicit nonfinite-input
  refusals, real authenticated source widths65/8193, and B3 refusal at all six
  accepted widths. They must allocate/upload no GPU buffers.
- Separate fresh fault processes: finite-input F32 overflow or finite
  accumulation whose BF16 rounding overflows. Same six guarded physical leases,
  exact status and readonly bits checked before refusing `Ready`; original
  source/host/device/completion owners remain fenced/quarantined. No final-zero
  claim or further device operation in these processes.

## Physical ownership and independent gates

- Six original leased buffers; 16-byte prefix and suffix canaries on every
  binding, with exact interior subviews. All 48 canary words and every readonly
  source/input/runtime-zero byte are read back. Write payloads start poisoned.
- Original fair physical queue, pinned original CUDA actor/client, source issuer,
  `ResidentInitialization`, `ResidentSubmission`, completion owner, event,
  ledger and quarantine. RAM and device staging/workspaces admitted before
  private payload creation; original window retained through terminal work.
- Before private `Ready`, compare every F32 accumulation, rounded F32 word and
  status against original CPU output. Synthetic analytic witnesses additionally
  check that chosen recipes discriminate the intended arithmetic.
- Successful cases require real output/event retirement and final original GPU
  reservation0. The normal helper also requires host admission0 after source
  drop. Fault cases require the opposite: genuine retained completion owner,
  pending event, device fence and retained source RAM; no `Ready`.
- Owned launcher: reviewed byte-identical cleanup functions, unreaped leader
  identity, independently bounded TERM/KILL/reap stages, PGID absence and pipe
  EOF/close. All post-spawn paths enter cleanup; combined log4MiB, wall900s,
  four explicit CPUs. Runtime duration is not a benchmark.
- Validator independently enforces exact complete case/band receipt multiset,
  source full SHA/FD identity/index/header/EOF, actual affinity, closure and
  cleanup. Ten prepared source-free Python negatives and three prepared Rust
  transport tests are not numerical qualification.

## ROOT application and checks

1. Independently review `NEW_DELTA.patch`, all new source/helper files and
   `freeze-manifest.json`. Apply only this delta to the ROOT V3 baseline with
   fresh timestamps. `main_after_merge` pins all 372 intended shared sources;
   `baseline_files` pins original V1 frozen preparation, not a replacement tree.
2. ROOT source-free fixtures: `python3 root-helper/test_receipts.py`.
3. ROOT serial strict Clippy and focused GPU-crate unit tests with
   `--no-default-features --features cuda,qualification`; retain any failures.
4. ROOT builds the helper in release, using original native isolation wrapper,
   offline dependency lock generation and same native CUDA environment as the
   base qualifier. Source pin closure and actual build/binary attestation are
   prerequisites; a supplied binary digest alone is not compilation evidence.

```bash
CARGO_BUILD_JOBS=2 RAYON_NUM_THREADS=4 python3 scripts/native-gpu.py -- \
  cargo build --release --offline --manifest-path \
  .agents/runs/model-execution-cuda-cpu-bf16-projection-adversary-20261020/root-helper/Cargo.toml
```

5. Let `RUN` be this absolute directory and `BINARY` the actual attested helper.
   ROOT executes each fresh output under the same native isolation wrapper;
   `--binary-sha256` must be the actual post-build executable SHA.

```bash
python3 "$RUN/root-helper/run_replay.py" --binary "$BINARY" \
  --binary-sha256 "$BINARY_SHA" --new-output "$RUN/root-normal-1"
python3 "$RUN/root-helper/validate_replay.py" "$RUN/root-normal-1"
```

6. For each B1/2/4/8, repeat launcher+validator in a separate fresh process/output
   with `--mode fp32-overflow --batch B`, then `bf16-overflow`. A fenced fault
   process is never reused. Do not accept normal-only receipts as status proof.
7. Preserve first failures and executed source bytes. Any correction is a fresh
   additive freeze. No full-model acceptance or performance assertion follows
   from this generic operator gate; trained source/base qualification and all
   integration gates remain required.

## Remaining scope

- ROOT source review, compilation/Clippy/unit/receipt tests and all physical
  normal/fault proofs remain unexecuted by the author.
- Compiled floating-point contraction/FTZ behavior must be decided by the actual
  accepted CPU64-to-CUDA adversaries; no tolerance or signed-zero exceptions.
- No random/large-shape exhaustive arithmetic proof, 27B native-model support,
  all42 full-HF graph/state evidence, cancellation integration or performance
  claim. Existing production refusal/profile guards remain closed.

ROOT V2: retain original E0283 strict compile failure; make readonly readback comparison explicitly &[u8] via &actual[..]. No arithmetic/kernel/control change.

ROOT V3: retain V2 strict collapsible_if failure; use equivalent witness let-chain. No numerical/kernel/proof change.

````

## Source: .agents/runs/model-execution-cuda-cpu-bf16-projection-adversary-root-v3-20261020/TODO.md

Original UTF-8 bytes: 1181; SHA256 `3eadae3992fe4bf78fdb4f67e45f0089c1fdb9067b604252eef7b5a1af50190e`.
Resolve embedded relative paths against the original path, not this volume.

````text
# Working checklist

Parent: `.agents/plans/serving-and-speculation/TODO.md` / shared original-HF engine

Owner: batch_design_review

- [x] Keep original b5decd62/679d8690 baseline immutable; retain ROOT V3 child
  qualifier defensive coverage/quarantine changes as dependency.
- [x] Prepare one same-kernel guarded physical qualifier, source-authenticated
  synthetic fixtures, CPU64 numerical oracle, cold refusals and fresh overflow
  processes. No alternate reducer or trained expected tensors.
- [x] Prepare original-source generator/helper, owned finite launcher,
  independent receipt/source/cleanup validator and source-free negative tests.
- [x] Source-review owned code; AST parse and rustfmt only, no test execution.
- [ ] ROOT independent source review and exact additive merge.
- [ ] ROOT focused strict/unit/receipt checks, release build with actual attestation.
- [ ] ROOT actual CUDA 328-case normal and eight separate status-fault runs.
- [ ] Resolve any actual arithmetic/status/owner failure causally, without
  tolerance, signed-zero exceptions, CPU fallback or profile changes.
- [ ] Trained-model/integration/full-HF/all42/performance remain separately open.

````

## Source: .agents/runs/model-execution-cuda-cpu-bf16-projection-root-v2-20261020/README.md

Original UTF-8 bytes: 10289; SHA256 `900d0ac540573a23df94a75392c7e5525d93c53ab57fdaf8bc0cf907ad6c1700`.
Resolve embedded relative paths against the original path, not this volume.

````text
# CUDA BF16 projection: accepted CPU target

Author: @batch_design_review

Status: **source-only, uncompiled, unexecuted**. Parent: model-execution /
serving-and-speculation. ROOT owns all merge, compilation, operator execution,
physical acceptance and shared-plan reconciliation.

## Scope

- New private qualifier-only CUDA implementation of the existing
  `TorchCpuBf16Avx2MixedSourceV1` BF16 projection. No production selection change.
- Three new GPU files in `proposed-src/`; seven-path implementation delta includes
  the separate four-path `WIRING.patch`.
- No model/token loop, learned cache, SessionAuthority, StateRoot, checkpoint,
  HTTP mode or full-HF eligibility changes. No cuBLAS/vendor provider, foreign
  stream/context, direct CUDA driver launch, CPU execution fallback or tolerance.
- This targets the accepted **CPU** profile. It does not implement the different
  genuine vLLM CUDA BF16 profile or claim vLLM arithmetic/throughput parity.

## Closed mathematical contract

- Authentic sealed `VerifiedTensorSource` + `TensorWindow`; full unchanged
  `SourceTensor` descriptor, original BF16 OutputInput axes/name/shape/kind,
  original byte offset/length, exact transported row range/source offset and
  verified bundle. No reduced/fake tensor descriptor or provider region.
- Raw finite BF16 weights; explicit raw finite BF16 `[B,columns]` inputs. No casts
  of F32 learned sources. Input transport is not a trained-producer claim.
- Columns positive/divisible64, at most8192; output range1..16384 rows, at most
 32MiB raw weights; B exactly1/2/4/8. Refuse unsupported tails, dtypes, layouts,
  profiles, geometry and issuer before device work. Full original rows <=1M.
- One64-thread workgroup per `[request, original output row]`, ONE actual native
  launch for all B rows. Lane `register*8+lane` owns the corresponding existing
  CPU partial and sequentially applies `fma(weight,input,partial)` to columns
  `lane + 64*k`.
- Collapse partials64→32→16→8 exactly matches CPU register offsets4/2/1. Vec8
  horizontal order is `((v0+v4)+(v2+v6))+((v1+v5)+(v3+v7))`. Add the CPU empty
  tail's positive zero last. Final BF16 RNE uses unchanged HostMath bit formula;
  expose raw F32 accumulation AND BF16-widened F32 output, with per-output status.
- CubeCL's FAdd literal-zero simplifier drops `x+0` irrespective of zero sign.
  The candidate therefore reads an immutable, physically initialized **runtime
  +0** buffer for both the initial partials and mandatory final empty-tail add.
  This follows the original addition, not an output zero canonicalizer.
- Attribution/license copied from the existing CPU reducer. No new reduction
  schedule was inferred from cuBLAS or TensorCore behavior.

## Original ownership

- `PreparedProjection::prepare(existing Arc<ResidentService>, sealed weights,
  sealed inputs, pure correlation, cancellation)` accepts only the existing
  cold-pinned CUDA client/service. It never creates another engine/queue/context.
- Same-source RAM admission precedes private input copy, queued H2D copies and
  D2H arrays. SourceOwner retains original staged weights/input plus all those
  charges; output readback retains that owner until its last consumer drops.
- Original physical prepaid envelope, fair ticket/slot, initialization guard,
  six original BufferLeases and original submission/Ready/retirement only.
  Recheck capacity after fair waiting before allocation/H2D. Every actual payload
  has exact bytes/service/source/generation/alignment/unique lease preflight.
- GPU bytes: `2*rows*columns + 2*B*columns + 4 + 12*B*rows`. The final term covers
  raw accumulation, rounded output and status. Kernel shared storage512B.
- Status flags:1 nonfinite weight,2 nonfinite input,4 incorrect zero,8 nonfinite
  accumulation,16 nonfinite BF16 rounding. Cold finite validation precedes math;
  complete actual readback/status validation precedes the private Ready.
- `execute` consumes the prepared storage. Prior Ready outputs cannot be
  overwritten by a later operator invocation. No logical state publication.
- `CompletionOwner` attaches BEFORE first allocation/H2D or kernel. Every unknown
  path follows original armed init/submission Drop into existing quarantine.
  The only missing shared seam is the narrow init setter in `WIRING.patch`;
  it preserves the reviewed attach-before-ensure semantics.

## Prepared physical proof

`root-helper/` uses the existing admitted original source opener and the original
CPU attention observation validator. No Python math/Torch/model import.

For BOTH real original .8B/2B sources, actual layer3 q_proj/k_proj/v_proj, complete
4096/512/512 rows, teachers11→353:

1. Authenticate unchanged accepted source36-F32-restored CPU observations and
   raw BF16 hidden/projection archive spans through the existing parent validator.
2. Full source-file FD/digest before/after, existing original320-role config
   binding, source metadata and two4MiB binding charges, private input admission.
3. B1 teacher11, independent B1 teacher353, then real native B2/4/8 (alternating
   actual teacher inputs, no padded requests). Compare EVERY raw CUDA F32
   accumulation bit to the existing CPU window operator; EVERY final BF16 word
   to the untouched original public CPU projection archive.
4. Real original GPU pressure allocation leaves four bytes less than required;
   require exact prelaunch capacity refusal without H2D/extra lease/event. Check
   pre-prepare and initialized-but-unlaunched cancellation, wrong descriptor
   source/name/axes/full-row count; no release/fence bypass.
5. Exactly5 native math launches per matrix,15 per model,30 across both. Each
   model compares81920 raw accumulations AND81920 original public BF16 words.
   Root helper checks final host0; original qualifier checks each service/event
   and physical process-global budget final0. External validator closes the
   complete native receipt multiplicity and source/mode/width/lifecycle scope.
6. Separate fresh-process `abandon` submits a REAL kernel and drops its original
   flight without trusted completion. Require original fence, pending event,
   retained authentic source/window/leases and coarse envelope. No Ready or
   final-zero claim; process exit is not a logical reclamation receipt.

B2/4/8 here qualifies this stateless operator only. It does not prove independent
model-state batching, a complete HF layer/token, GPU attention/FLA or full logits.
ROOT separately reports actual CPU attention **Rust** fragment acceptance for both
models/all47104 words plus pressure/cancel/lifetime/final0 (signed fe2a342). This
candidate still compares genuine public archives and raw existing CPU accumulators;
that CPU result is not CUDA qualification.

## ROOT-only invocation

First review all files, compose the three new source files + `WIRING.patch` onto
MAIN with fresh mtimes, serialize the init setter with the graph work. `base-src/`
is exact before, `main-after/` is the private formatted four-file composition.
No vendor/actor/retirement changes are required.

```sh
RUN=/home/pvl/spaces/abird/src/zerogravity/.agents/runs/model-execution-cuda-cpu-bf16-projection-20261020
CARGO_BUILD_JOBS=2 RAYON_NUM_THREADS=4 python3 scripts/native-gpu.py -- \
  cargo clippy --offline -p zerogravity-gpu --no-default-features \
  --features cuda,qualification --lib --tests -- -D warnings
CARGO_BUILD_JOBS=2 RAYON_NUM_THREADS=4 python3 scripts/native-gpu.py -- \
  cargo test --offline -p zerogravity-gpu --no-default-features \
  --features cuda,qualification cpu_bf16_projection::tests -- --test-threads=1
python3 -m unittest discover -s "$RUN/root-helper" -p test_proof.py
CARGO_BUILD_JOBS=2 RAYON_NUM_THREADS=4 python3 scripts/native-gpu.py -- \
  cargo build --offline --release --manifest-path "$RUN/root-helper/Cargo.toml"
```

ROOT binds the fresh build/source/binary log and supplies its SHA; no source file
claims its own binary was built. Reuse the already authenticated raw extraction
from the attention replay or root-execute its unchanged exporter. ROOT selects
fresh names for output directories and actual input/binary SHA strings:

```sh
python3 scripts/native-gpu.py -- python3 "$RUN/root-helper/run_replay.py" \
  --binary "$BINARY" --binary-sha256 "$BINARY_SHA" \
  --inputs "$AUTHENTIC_INPUTS" --inputs-sha256 "$INPUTS_SHA" \
  --new-output "$RUN/root-normal-1" --mode normal
python3 "$RUN/root-helper/validate_replay.py" --output "$RUN/root-normal-1"
```

For uncertainty, same command with `--mode abandon --model 0.8b` and fresh output
under the run, then independent validator. Each child owns a fresh process; it
uses the existing GPU0 only. No launch on AMD. All work is serialized by ROOT.
The helper has900s child/3300s overall and4MiB combined-log bounds, exact four CPU
pinning, original reviewed unreaped-leader TERM/grace/KILL/reap/pipe cleanup. No
CUDA virtual address-space bound is misrepresented as resident-memory admission;
source/window/transport/output and original GPU buffers retain their real ledgers.

## Pending acceptance

- Source-only author rustfmt + AST parse only; NO build/test/operator execution.
- ROOT compilation/strict checks and genuine .8B/2B native CUDA all-row proof.
- Actual generated/compiled F32 FMA and addition schedule must retain gradual
  underflow and round-to-nearest; NVRTC source has no fast-math/ftz flag, but
  source inspection is not a compiled-code or physical arithmetic receipt.
- Before widening general numerical eligibility: meaningful physical signed-zero,
  subnormal, FMA-vs-product/add, register-order cancellation, RNE tie/overflow
  adversaries against the existing independently qualified CPU target. The two
  prepared Rust unit tests check bounds/transport only, not GPU arithmetic.
- Nonzero output-row window physical proof and future genuine27B width5120
  inputs remain open. The binding supports them; source-only support is no proof.
- Full HF GPU schedule/profile, state ownership integration, all42/fullV, vLLM
  CUDA math, matched benchmark and performance remain open. No speed claim.

ROOT V2: preserve V1 failed strict log. Iterate existing CPU expected rows; require actual original quarantine completion-owner count increase after real abandoned launch. No arithmetic/production-policy changes.

````

## Source: .agents/runs/model-execution-cuda-cpu-bf16-projection-root-v2-20261020/TODO.md

Original UTF-8 bytes: 1408; SHA256 `6c3fc183b22acec13acafcf974f6d2824296ce019d0cfe1638e2bb04ced48844`.
Resolve embedded relative paths against the original path, not this volume.

````text
# Working checklist

- Parent: model-execution; serving-and-speculation.
- Owner: @batch_design_review. Scope: isolated proposed source only; ROOT merges,
  tests/executes/reconciles. No MAIN tracked files or shared plans edited.
- [x] Audit exact existing CPU64 reducer and BF16 boundary; source/issuer/window
  binding and original CUDA service/queue/leases/Ready/quarantine.
- [x] Prepare native CUDA B1/2/4/8 kernel + closed source bindings and original
  resident initialization/submission wrappers; preserve original source dtypes.
- [x] Prepare real two-teacher dual-oracle/all-row qualification, pressure,
  prelaunch cancel and separate actual uncertain-completion quarantine proof.
- [x] Prepare original .8B/2B authenticated archive/source helper + finite owned
  process launcher and receipt coverage validator. No expected math generated.
- [x] Author rustfmt and Python AST parse only. No tests/build/runtime executed.
- [ ] ROOT independent review, compile/strict Clippy/source-free fixtures.
- [ ] ROOT actual physical native matrix qualification for both approved sources.
- [ ] ROOT actual abort/fence/quarantine process and normal global reservations0.
- [ ] Physical arithmetic adversaries, nonzero row-range and27B widths.
- [ ] Full model/HF source profile/session/math integration remains outside this
  candidate; no inference/performance/vLLM claim or production eligibility change.

````

## Source: .agents/runs/model-execution-cuda-cpu-bf16-projection-root-v3-20261020/README.md

Original UTF-8 bytes: 10500; SHA256 `81b28ff6e8b51e0bf828ddeb6099b4dd0f7e78d0706d168142574917e976d9ba`.
Resolve embedded relative paths against the original path, not this volume.

````text
# CUDA BF16 projection: accepted CPU target

Author: @batch_design_review

Status: **source-only, uncompiled, unexecuted**. Parent: model-execution /
serving-and-speculation. ROOT owns all merge, compilation, operator execution,
physical acceptance and shared-plan reconciliation.

## Scope

- New private qualifier-only CUDA implementation of the existing
  `TorchCpuBf16Avx2MixedSourceV1` BF16 projection. No production selection change.
- Three new GPU files in `proposed-src/`; seven-path implementation delta includes
  the separate four-path `WIRING.patch`.
- No model/token loop, learned cache, SessionAuthority, StateRoot, checkpoint,
  HTTP mode or full-HF eligibility changes. No cuBLAS/vendor provider, foreign
  stream/context, direct CUDA driver launch, CPU execution fallback or tolerance.
- This targets the accepted **CPU** profile. It does not implement the different
  genuine vLLM CUDA BF16 profile or claim vLLM arithmetic/throughput parity.

## Closed mathematical contract

- Authentic sealed `VerifiedTensorSource` + `TensorWindow`; full unchanged
  `SourceTensor` descriptor, original BF16 OutputInput axes/name/shape/kind,
  original byte offset/length, exact transported row range/source offset and
  verified bundle. No reduced/fake tensor descriptor or provider region.
- Raw finite BF16 weights; explicit raw finite BF16 `[B,columns]` inputs. No casts
  of F32 learned sources. Input transport is not a trained-producer claim.
- Columns positive/divisible64, at most8192; output range1..16384 rows, at most
 32MiB raw weights; B exactly1/2/4/8. Refuse unsupported tails, dtypes, layouts,
  profiles, geometry and issuer before device work. Full original rows <=1M.
- One64-thread workgroup per `[request, original output row]`, ONE actual native
  launch for all B rows. Lane `register*8+lane` owns the corresponding existing
  CPU partial and sequentially applies `fma(weight,input,partial)` to columns
  `lane + 64*k`.
- Collapse partials64→32→16→8 exactly matches CPU register offsets4/2/1. Vec8
  horizontal order is `((v0+v4)+(v2+v6))+((v1+v5)+(v3+v7))`. Add the CPU empty
  tail's positive zero last. Final BF16 RNE uses unchanged HostMath bit formula;
  expose raw F32 accumulation AND BF16-widened F32 output, with per-output status.
- CubeCL's FAdd literal-zero simplifier drops `x+0` irrespective of zero sign.
  The candidate therefore reads an immutable, physically initialized **runtime
  +0** buffer for both the initial partials and mandatory final empty-tail add.
  This follows the original addition, not an output zero canonicalizer.
- Attribution/license copied from the existing CPU reducer. No new reduction
  schedule was inferred from cuBLAS or TensorCore behavior.

## Original ownership

- `PreparedProjection::prepare(existing Arc<ResidentService>, sealed weights,
  sealed inputs, pure correlation, cancellation)` accepts only the existing
  cold-pinned CUDA client/service. It never creates another engine/queue/context.
- Same-source RAM admission precedes private input copy, queued H2D copies and
  D2H arrays. SourceOwner retains original staged weights/input plus all those
  charges; output readback retains that owner until its last consumer drops.
- Original physical prepaid envelope, fair ticket/slot, initialization guard,
  six original BufferLeases and original submission/Ready/retirement only.
  Recheck capacity after fair waiting before allocation/H2D. Every actual payload
  has exact bytes/service/source/generation/alignment/unique lease preflight.
- GPU bytes: `2*rows*columns + 2*B*columns + 4 + 12*B*rows`. The final term covers
  raw accumulation, rounded output and status. Kernel shared storage512B.
- Status flags:1 nonfinite weight,2 nonfinite input,4 incorrect zero,8 nonfinite
  accumulation,16 nonfinite BF16 rounding. Cold finite validation precedes math;
  complete actual readback/status validation precedes the private Ready.
- `execute` consumes the prepared storage. Prior Ready outputs cannot be
  overwritten by a later operator invocation. No logical state publication.
- `CompletionOwner` attaches BEFORE first allocation/H2D or kernel. Every unknown
  path follows original armed init/submission Drop into existing quarantine.
  The only missing shared seam is the narrow init setter in `WIRING.patch`;
  it preserves the reviewed attach-before-ensure semantics.

## Prepared physical proof

`root-helper/` uses the existing admitted original source opener and the original
CPU attention observation validator. No Python math/Torch/model import.

For BOTH real original .8B/2B sources, actual layer3 q_proj/k_proj/v_proj, complete
4096/512/512 rows, teachers11→353:

1. Authenticate unchanged accepted source36-F32-restored CPU observations and
   raw BF16 hidden/projection archive spans through the existing parent validator.
2. Full source-file FD/digest before/after, existing original320-role config
   binding, source metadata and two4MiB binding charges, private input admission.
3. B1 teacher11, independent B1 teacher353, then real native B2/4/8 (alternating
   actual teacher inputs, no padded requests). Compare EVERY raw CUDA F32
   accumulation bit to the existing CPU window operator; EVERY final BF16 word
   to the untouched original public CPU projection archive.
4. Real original GPU pressure allocation leaves four bytes less than required;
   require exact prelaunch capacity refusal without H2D/extra lease/event. Check
   pre-prepare and initialized-but-unlaunched cancellation, wrong descriptor
   source/name/axes/full-row count; no release/fence bypass.
5. Exactly5 native math launches per matrix,15 per model,30 across both. Each
   model compares81920 raw accumulations AND81920 original public BF16 words.
   Root helper checks final host0; original qualifier checks each service/event
   and physical process-global budget final0. External validator closes the
   complete native receipt multiplicity and source/mode/width/lifecycle scope.
6. Separate fresh-process `abandon` submits a REAL kernel and drops its original
   flight without trusted completion. Require original fence, pending event,
   retained authentic source/window/leases and coarse envelope. No Ready or
   final-zero claim; process exit is not a logical reclamation receipt.

B2/4/8 here qualifies this stateless operator only. It does not prove independent
model-state batching, a complete HF layer/token, GPU attention/FLA or full logits.
ROOT separately reports actual CPU attention **Rust** fragment acceptance for both
models/all47104 words plus pressure/cancel/lifetime/final0 (signed fe2a342). This
candidate still compares genuine public archives and raw existing CPU accumulators;
that CPU result is not CUDA qualification.

## ROOT-only invocation

First review all files, compose the three new source files + `WIRING.patch` onto
MAIN with fresh mtimes, serialize the init setter with the graph work. `base-src/`
is exact before, `main-after/` is the private formatted four-file composition.
No vendor/actor/retirement changes are required.

```sh
RUN=/home/pvl/spaces/abird/src/zerogravity/.agents/runs/model-execution-cuda-cpu-bf16-projection-20261020
CARGO_BUILD_JOBS=2 RAYON_NUM_THREADS=4 python3 scripts/native-gpu.py -- \
  cargo clippy --offline -p zerogravity-gpu --no-default-features \
  --features cuda,qualification --lib --tests -- -D warnings
CARGO_BUILD_JOBS=2 RAYON_NUM_THREADS=4 python3 scripts/native-gpu.py -- \
  cargo test --offline -p zerogravity-gpu --no-default-features \
  --features cuda,qualification cpu_bf16_projection::tests -- --test-threads=1
python3 -m unittest discover -s "$RUN/root-helper" -p test_proof.py
CARGO_BUILD_JOBS=2 RAYON_NUM_THREADS=4 python3 scripts/native-gpu.py -- \
  cargo build --offline --release --manifest-path "$RUN/root-helper/Cargo.toml"
```

ROOT binds the fresh build/source/binary log and supplies its SHA; no source file
claims its own binary was built. Reuse the already authenticated raw extraction
from the attention replay or root-execute its unchanged exporter. ROOT selects
fresh names for output directories and actual input/binary SHA strings:

```sh
python3 scripts/native-gpu.py -- python3 "$RUN/root-helper/run_replay.py" \
  --binary "$BINARY" --binary-sha256 "$BINARY_SHA" \
  --inputs "$AUTHENTIC_INPUTS" --inputs-sha256 "$INPUTS_SHA" \
  --new-output "$RUN/root-normal-1" --mode normal
python3 "$RUN/root-helper/validate_replay.py" --output "$RUN/root-normal-1"
```

For uncertainty, same command with `--mode abandon --model 0.8b` and fresh output
under the run, then independent validator. Each child owns a fresh process; it
uses the existing GPU0 only. No launch on AMD. All work is serialized by ROOT.
The helper has900s child/3300s overall and4MiB combined-log bounds, exact four CPU
pinning, original reviewed unreaped-leader TERM/grace/KILL/reap/pipe cleanup. No
CUDA virtual address-space bound is misrepresented as resident-memory admission;
source/window/transport/output and original GPU buffers retain their real ledgers.

## Pending acceptance

- Source-only author rustfmt + AST parse only; NO build/test/operator execution.
- ROOT compilation/strict checks and genuine .8B/2B native CUDA all-row proof.
- Actual generated/compiled F32 FMA and addition schedule must retain gradual
  underflow and round-to-nearest; NVRTC source has no fast-math/ftz flag, but
  source inspection is not a compiled-code or physical arithmetic receipt.
- Before widening general numerical eligibility: meaningful physical signed-zero,
  subnormal, FMA-vs-product/add, register-order cancellation, RNE tie/overflow
  adversaries against the existing independently qualified CPU target. The two
  prepared Rust unit tests check bounds/transport only, not GPU arithmetic.
- Nonzero output-row window physical proof and future genuine27B width5120
  inputs remain open. The binding supports them; source-only support is no proof.
- Full HF GPU schedule/profile, state ownership integration, all42/fullV, vLLM
  CUDA math, matched benchmark and performance remain open. No speed claim.

ROOT V2: preserve V1 failed strict log. Iterate existing CPU expected rows; require actual original quarantine completion-owner count increase after real abandoned launch. No arithmetic/production-policy changes.

ROOT V3: independent review requires exactly two complete CPU oracle rows before iteration; avoid any future short-oracle truncation. V2 strict/two unit passes remain preserved. No numerical operation changes.

````

## Source: .agents/runs/model-execution-cuda-cpu-bf16-projection-root-v3-20261020/TODO.md

Original UTF-8 bytes: 1408; SHA256 `6c3fc183b22acec13acafcf974f6d2824296ce019d0cfe1638e2bb04ced48844`.
Resolve embedded relative paths against the original path, not this volume.

````text
# Working checklist

- Parent: model-execution; serving-and-speculation.
- Owner: @batch_design_review. Scope: isolated proposed source only; ROOT merges,
  tests/executes/reconciles. No MAIN tracked files or shared plans edited.
- [x] Audit exact existing CPU64 reducer and BF16 boundary; source/issuer/window
  binding and original CUDA service/queue/leases/Ready/quarantine.
- [x] Prepare native CUDA B1/2/4/8 kernel + closed source bindings and original
  resident initialization/submission wrappers; preserve original source dtypes.
- [x] Prepare real two-teacher dual-oracle/all-row qualification, pressure,
  prelaunch cancel and separate actual uncertain-completion quarantine proof.
- [x] Prepare original .8B/2B authenticated archive/source helper + finite owned
  process launcher and receipt coverage validator. No expected math generated.
- [x] Author rustfmt and Python AST parse only. No tests/build/runtime executed.
- [ ] ROOT independent review, compile/strict Clippy/source-free fixtures.
- [ ] ROOT actual physical native matrix qualification for both approved sources.
- [ ] ROOT actual abort/fence/quarantine process and normal global reservations0.
- [ ] Physical arithmetic adversaries, nonzero row-range and27B widths.
- [ ] Full model/HF source profile/session/math integration remains outside this
  candidate; no inference/performance/vLLM claim or production eligibility change.

````

## Source: .agents/runs/model-execution-cuda-cpu-bf16-projection-root-v4-20261020/README.md

Original UTF-8 bytes: 10848; SHA256 `5b83f364064ef890b49e7045c38ff037e05ac67fbc71d65e66d6df59b0f1f45c`.
Resolve embedded relative paths against the original path, not this volume.

````text
# CUDA BF16 projection: accepted CPU target

Author: @batch_design_review

Status: **source-only, uncompiled, unexecuted**. Parent: model-execution /
serving-and-speculation. ROOT owns all merge, compilation, operator execution,
physical acceptance and shared-plan reconciliation.

## Scope

- New private qualifier-only CUDA implementation of the existing
  `TorchCpuBf16Avx2MixedSourceV1` BF16 projection. No production selection change.
- Three new GPU files in `proposed-src/`; seven-path implementation delta includes
  the separate four-path `WIRING.patch`.
- No model/token loop, learned cache, SessionAuthority, StateRoot, checkpoint,
  HTTP mode or full-HF eligibility changes. No cuBLAS/vendor provider, foreign
  stream/context, direct CUDA driver launch, CPU execution fallback or tolerance.
- This targets the accepted **CPU** profile. It does not implement the different
  genuine vLLM CUDA BF16 profile or claim vLLM arithmetic/throughput parity.

## Closed mathematical contract

- Authentic sealed `VerifiedTensorSource` + `TensorWindow`; full unchanged
  `SourceTensor` descriptor, original BF16 OutputInput axes/name/shape/kind,
  original byte offset/length, exact transported row range/source offset and
  verified bundle. No reduced/fake tensor descriptor or provider region.
- Raw finite BF16 weights; explicit raw finite BF16 `[B,columns]` inputs. No casts
  of F32 learned sources. Input transport is not a trained-producer claim.
- Columns positive/divisible64, at most8192; output range1..16384 rows, at most
 32MiB raw weights; B exactly1/2/4/8. Refuse unsupported tails, dtypes, layouts,
  profiles, geometry and issuer before device work. Full original rows <=1M.
- One64-thread workgroup per `[request, original output row]`, ONE actual native
  launch for all B rows. Lane `register*8+lane` owns the corresponding existing
  CPU partial and sequentially applies `fma(weight,input,partial)` to columns
  `lane + 64*k`.
- Collapse partials64→32→16→8 exactly matches CPU register offsets4/2/1. Vec8
  horizontal order is `((v0+v4)+(v2+v6))+((v1+v5)+(v3+v7))`. Add the CPU empty
  tail's positive zero last. Final BF16 RNE uses unchanged HostMath bit formula;
  expose raw F32 accumulation AND BF16-widened F32 output, with per-output status.
- CubeCL's FAdd literal-zero simplifier drops `x+0` irrespective of zero sign.
  The candidate therefore reads an immutable, physically initialized **runtime
  +0** buffer for both the initial partials and mandatory final empty-tail add.
  This follows the original addition, not an output zero canonicalizer.
- Attribution/license copied from the existing CPU reducer. No new reduction
  schedule was inferred from cuBLAS or TensorCore behavior.

## Original ownership

- `PreparedProjection::prepare(existing Arc<ResidentService>, sealed weights,
  sealed inputs, pure correlation, cancellation)` accepts only the existing
  cold-pinned CUDA client/service. It never creates another engine/queue/context.
- Same-source RAM admission precedes private input copy, queued H2D copies and
  D2H arrays. SourceOwner retains original staged weights/input plus all those
  charges; output readback retains that owner until its last consumer drops.
- Original physical prepaid envelope, fair ticket/slot, initialization guard,
  six original BufferLeases and original submission/Ready/retirement only.
  Recheck capacity after fair waiting before allocation/H2D. Every actual payload
  has exact bytes/service/source/generation/alignment/unique lease preflight.
- GPU bytes: `2*rows*columns + 2*B*columns + 4 + 12*B*rows`. The final term covers
  raw accumulation, rounded output and status. Kernel shared storage512B.
- Status flags:1 nonfinite weight,2 nonfinite input,4 incorrect zero,8 nonfinite
  accumulation,16 nonfinite BF16 rounding. Cold finite validation precedes math;
  complete actual readback/status validation precedes the private Ready.
- `execute` consumes the prepared storage. Prior Ready outputs cannot be
  overwritten by a later operator invocation. No logical state publication.
- `CompletionOwner` attaches BEFORE first allocation/H2D or kernel. Every unknown
  path follows original armed init/submission Drop into existing quarantine.
  The only missing shared seam is the narrow init setter in `WIRING.patch`;
  it preserves the reviewed attach-before-ensure semantics.

## Prepared physical proof

`root-helper/` uses the existing admitted original source opener and the original
CPU attention observation validator. No Python math/Torch/model import.

For BOTH real original .8B/2B sources, actual layer3 q_proj/k_proj/v_proj, complete
4096/512/512 rows, teachers11→353:

1. Authenticate unchanged accepted source36-F32-restored CPU observations and
   raw BF16 hidden/projection archive spans through the existing parent validator.
2. Full source-file FD/digest before/after, existing original320-role config
   binding, source metadata and two4MiB binding charges, private input admission.
3. B1 teacher11, independent B1 teacher353, then real native B2/4/8 (alternating
   actual teacher inputs, no padded requests). Compare EVERY raw CUDA F32
   accumulation bit to the existing CPU window operator; EVERY final BF16 word
   to the untouched original public CPU projection archive.
4. Real original GPU pressure allocation leaves four bytes less than required;
   require exact prelaunch capacity refusal without H2D/extra lease/event. Check
   pre-prepare and initialized-but-unlaunched cancellation, wrong descriptor
   source/name/axes/full-row count; no release/fence bypass.
5. Exactly5 native math launches per matrix,15 per model,30 across both. Each
   model compares81920 raw accumulations AND81920 original public BF16 words.
   Root helper checks final host0; original qualifier checks each service/event
   and physical process-global budget final0. External validator closes the
   complete native receipt multiplicity and source/mode/width/lifecycle scope.
6. Separate fresh-process `abandon` submits a REAL kernel and drops its original
   flight without trusted completion. Require original fence, pending event,
   retained authentic source/window/leases and coarse envelope. No Ready or
   final-zero claim; process exit is not a logical reclamation receipt.

B2/4/8 here qualifies this stateless operator only. It does not prove independent
model-state batching, a complete HF layer/token, GPU attention/FLA or full logits.
ROOT separately reports actual CPU attention **Rust** fragment acceptance for both
models/all47104 words plus pressure/cancel/lifetime/final0 (signed fe2a342). This
candidate still compares genuine public archives and raw existing CPU accumulators;
that CPU result is not CUDA qualification.

## ROOT-only invocation

First review all files, compose the three new source files + `WIRING.patch` onto
MAIN with fresh mtimes, serialize the init setter with the graph work. `base-src/`
is exact before, `main-after/` is the private formatted four-file composition.
No vendor/actor/retirement changes are required.

```sh
RUN=/home/pvl/spaces/abird/src/zerogravity/.agents/runs/model-execution-cuda-cpu-bf16-projection-20261020
CARGO_BUILD_JOBS=2 RAYON_NUM_THREADS=4 python3 scripts/native-gpu.py -- \
  cargo clippy --offline -p zerogravity-gpu --no-default-features \
  --features cuda,qualification --lib --tests -- -D warnings
CARGO_BUILD_JOBS=2 RAYON_NUM_THREADS=4 python3 scripts/native-gpu.py -- \
  cargo test --offline -p zerogravity-gpu --no-default-features \
  --features cuda,qualification cpu_bf16_projection::tests -- --test-threads=1
python3 -m unittest discover -s "$RUN/root-helper" -p test_proof.py
CARGO_BUILD_JOBS=2 RAYON_NUM_THREADS=4 python3 scripts/native-gpu.py -- \
  cargo build --offline --release --manifest-path "$RUN/root-helper/Cargo.toml"
```

ROOT binds the fresh build/source/binary log and supplies its SHA; no source file
claims its own binary was built. Reuse the already authenticated raw extraction
from the attention replay or root-execute its unchanged exporter. ROOT selects
fresh names for output directories and actual input/binary SHA strings:

```sh
python3 scripts/native-gpu.py -- python3 "$RUN/root-helper/run_replay.py" \
  --binary "$BINARY" --binary-sha256 "$BINARY_SHA" \
  --inputs "$AUTHENTIC_INPUTS" --inputs-sha256 "$INPUTS_SHA" \
  --new-output "$RUN/root-normal-1" --mode normal
python3 "$RUN/root-helper/validate_replay.py" --output "$RUN/root-normal-1"
```

For uncertainty, same command with `--mode abandon --model 0.8b` and fresh output
under the run, then independent validator. Each child owns a fresh process; it
uses the existing GPU0 only. No launch on AMD. All work is serialized by ROOT.
The helper has900s child/3300s overall and4MiB combined-log bounds, exact four CPU
pinning, original reviewed unreaped-leader TERM/grace/KILL/reap/pipe cleanup. No
CUDA virtual address-space bound is misrepresented as resident-memory admission;
source/window/transport/output and original GPU buffers retain their real ledgers.

## Pending acceptance

- Source-only author rustfmt + AST parse only; NO build/test/operator execution.
- ROOT compilation/strict checks and genuine .8B/2B native CUDA all-row proof.
- Actual generated/compiled F32 FMA and addition schedule must retain gradual
  underflow and round-to-nearest; NVRTC source has no fast-math/ftz flag, but
  source inspection is not a compiled-code or physical arithmetic receipt.
- Before widening general numerical eligibility: meaningful physical signed-zero,
  subnormal, FMA-vs-product/add, register-order cancellation, RNE tie/overflow
  adversaries against the existing independently qualified CPU target. The two
  prepared Rust unit tests check bounds/transport only, not GPU arithmetic.
- Nonzero output-row window physical proof and future genuine27B width5120
  inputs remain open. The binding supports them; source-only support is no proof.
- Full HF GPU schedule/profile, state ownership integration, all42/fullV, vLLM
  CUDA math, matched benchmark and performance remain open. No speed claim.

ROOT V2: preserve V1 failed strict log. Iterate existing CPU expected rows; require actual original quarantine completion-owner count increase after real abandoned launch. No arithmetic/production-policy changes.

ROOT V3: independent review requires exactly two complete CPU oracle rows before iteration; avoid any future short-oracle truncation. V2 strict/two unit passes remain preserved. No numerical operation changes.

ROOT V4: isolate all five accepted CPU observer module basenames during import, verify every original module path, restore this harness module/path namespace in finally. V3 pre-child import failure retained. No Rust/helper/kernel/source/arithmetical change or rebuild; reuse exact ROOT V3 binary with its original build and 371source attestation.

````

## Source: .agents/runs/model-execution-cuda-cpu-bf16-projection-root-v4-20261020/TODO.md

Original UTF-8 bytes: 1408; SHA256 `6c3fc183b22acec13acafcf974f6d2824296ce019d0cfe1638e2bb04ced48844`.
Resolve embedded relative paths against the original path, not this volume.

````text
# Working checklist

- Parent: model-execution; serving-and-speculation.
- Owner: @batch_design_review. Scope: isolated proposed source only; ROOT merges,
  tests/executes/reconciles. No MAIN tracked files or shared plans edited.
- [x] Audit exact existing CPU64 reducer and BF16 boundary; source/issuer/window
  binding and original CUDA service/queue/leases/Ready/quarantine.
- [x] Prepare native CUDA B1/2/4/8 kernel + closed source bindings and original
  resident initialization/submission wrappers; preserve original source dtypes.
- [x] Prepare real two-teacher dual-oracle/all-row qualification, pressure,
  prelaunch cancel and separate actual uncertain-completion quarantine proof.
- [x] Prepare original .8B/2B authenticated archive/source helper + finite owned
  process launcher and receipt coverage validator. No expected math generated.
- [x] Author rustfmt and Python AST parse only. No tests/build/runtime executed.
- [ ] ROOT independent review, compile/strict Clippy/source-free fixtures.
- [ ] ROOT actual physical native matrix qualification for both approved sources.
- [ ] ROOT actual abort/fence/quarantine process and normal global reservations0.
- [ ] Physical arithmetic adversaries, nonzero row-range and27B widths.
- [ ] Full model/HF source profile/session/math integration remains outside this
  candidate; no inference/performance/vLLM claim or production eligibility change.

````

## Source: .agents/runs/model-execution-full-forward-2026-10-18/TODO.md

Original UTF-8 bytes: 1184; SHA256 `bd5cb9772188fda56fe65771fd49beda3944153cc981366905b7e70e854a1fdd`.
Resolve embedded relative paths against the original path, not this volume.

````text
# ME-P0 — original Qwen full-forward FP32 diagnostic

Base: 6329cff
Plan: [model execution](../../plans/model-execution/TODO.md)
Scope: one shared Qwen layer traversal and full synthetic model
under one root; no production BF16, device or trained 27B claim.

- [x] Shared Qwen layer traversal preserves GGUF operator math.
- [x] Complete synthetic four-layer source model: seven domains,
  full diagnostic vocabulary, two tokens and atomic root.
- [x] Independent analytic final logit ~8 and first-layer state
  equality; invalid/foreign/stale/competing proposals refuse.
- [x] Strict Clippy default/composed (all targets),
  complete isolated scripts/check.sh 1,119 passed/0 failed/
  69 ignored across 238 Rust suites + 18 Python fixtures.
  Precommit Nix source-integrity passed; postcommit still open.
- [x] Original trained Qwen3.5 .8B/2B direct-cache and HF-import
  independent release parity tests passed: exact logits/all42 StateRoot
  domains and two-token source/session mutation/lease behavior.
- [ ] Scoped local signed commit without push.
- [ ] Deferred: reference-exact BF16/FLA, original profile per-op
  MathematicalBackend and SessionAuthority, real 27B/SSD/GPU.

````

## Source: .agents/runs/model-execution-gated-rms-observation-2026-10-20/TODO.md

Original UTF-8 bytes: 1857; SHA256 `7fe87623ddbf76760d25481aeb4dce546a143e67a155b53a9c1fd1d6d344ac2b`.
Resolve embedded relative paths against the original path, not this volume.

````text
# Actual pinned layer0 gated RMS observations

Parent: model-execution. Owner: /root/spine_review.
Own ignored run only; no tracked production/model/GPU/default math edits.
Root executes once when CPU/GPU scheduling permits. No agent runtime launch.

- [x] Copy existing pinned slow-GDN capture/runner/validator; retain their source hashes in BASE.json.
- [x] Preserve tokens11,353, original source36F32-restored profile, source/model/runtime pins, 98 prior full-logit/all-cache exact checks and layer0 GDN frame observations.
- [x] Add layer0 gated-norm actual forward inputs/weight/selected line locals/return; narrowly enter TorchDispatchMode in that module's prehook and exit in always-called posthook.
- [x] Record every original norm op argument/result; call each original op exactly once and return its unmodified original result. Observer clones are suppressed from the operation log.
- [x] Identify actual squares/mean/denominator/rsqrt/core cast/normalizedBF16/F32 gamma product/gate cast/SiLUFP32/product/final BF16 through pinned source line and operator identity. No reconstructed arithmetic.
- [x] Exclusive new output and invocation guards, 64MiB total tensor payload, CPU-only/networknone/source+reference readonly/12GiB/4CPU container runner.
- [x] Python syntax compilation only. No model/container/Cargo/GPU execution.
- [ ] Root run_probe.py execution and validate_capture.py archive validation; TorchDispatchMode viability/operator overload names remain unexecuted assumptions until that run.

Caveats: extra observer work invalidates performance measurements; unchanged 98 raw-reference checks must pass. Actual SiLU FP32 is observed, not its internal exp instructions. Missing/changed expected source line/operator/dtype fails closed; do not relabel another op to make capture pass. No Rust/full-model/CUDA/FLA qualification follows.

````

## Source: .agents/runs/model-execution-gated-rms-observation-attempt-2-2026-10-20/TODO.md

Original UTF-8 bytes: 1253; SHA256 `45bc421da915b1317ae07f6fe32f1dc3042b21ef82a496a1567c045d3830ba0d`.
Resolve embedded relative paths against the original path, not this volume.

````text
# Gated RMS actual-call observation attempt 2

Parent: model-execution. Owner: /root/spine_review. Root is sole runtime/Cargo/GPU coordinator.

- [x] Preserve failed first attempt and actual pinned public cast-dispatch probe.
- [x] Copy scripts into new exclusive run; keep original inference mode, model operators, source restoration and all 98 prior raw full-logit/cache checks.
- [x] Match the actual inference-mode aten.to.dtype label for four casts; keep exact pinned source-line/dtype/shape matching.
- [x] Persist bounded actual dispatch/events/metadata + raw archive before semantic-stage validation. Diagnostic status remains UNQUALIFIED_PENDING_VALIDATION even if later validation passes; only final manifest can record completed checks.
- [x] AST syntax parse only; no model/container/GPU execution by this agent.
- [ ] Root execute run_probe.py when scheduled; then validate_capture.py.

Limits: CPU only, no network/GPU, 12GiB cap/4 CPUs/4 threads; source and prior reference read-only; global observation tensor payload <=64MiB; per-step diagnostic JSON <=4MiB; dispatch <=64 operations. Observer returns each actual op result unchanged, suppressing its own clone tracing. No Rust parity, trained GPU FLA, throughput or 27B qualification.

````

## Source: .agents/runs/model-execution-gdn-coefficients-observation-2026-10-20/TODO.md

Original UTF-8 bytes: 1764; SHA256 `da267d3b3c7f3351e03d2da0b112d84c7746576beb106f85d030bb9ec1d625ce`.
Resolve embedded relative paths against the original path, not this volume.

````text
# Actual original layer0 GDN coefficient observation

Parent: model-execution. Owner: /root/spine_review. Root alone executes runtime/Cargo/GPU/MAIN phases.

- [x] Copy successful attempt2 observer pins/loader/source36F32 restoration and all98 original raw248320-logit/cache checks into a fresh exclusive run.
- [x] Observe original forward a/b/A_log/dt_bias/beta/g locals and exact selected forward617/619 operation args/results. A_logF32, dt_biasBF16, a/b/betaBF16, gF32 explicit.
- [x] Outer layer0 GDN mode records only selected forward lines and actual exp calls at chunk386/413/414/415/cached483. Each retained selected-op tensor <=1024 elements, <=32 ops; pairwise381 omitted. Invoke every original operation exactly once and return its actual result.
- [x] Suppress observer clones in both outer coefficient and nested norm modes; preserve original inference_mode/operations. Retain all12 norm stages.
- [x] Save raw archives and bounded actual dispatch/events prevalidation diagnostics before strict semantic bindings; permit absence only for an undispatched already-F32 A_log no-op cast, with actual exp-input bits still required to match original A_log.
- [x] Byte/dtype/layout/argument/provenance validator written; AST syntax validation only. No Torch/tensor/model/container/Cargo/GPU execution by this agent.
- [ ] Root execute run_probe.py then validate_capture.py; failed-attempt files must remain exclusive.

Run bounds: networknone, no GPU/CDI, read-only pinned checkpoint/prior reference, CPU4/threads4,12GiB memory, total observation tensor payload64MiB, each diagnosticJSON4MiB. Coefficient logic is observed, never substituted or reconstructed. No scalar Rust sigmoid/exp/softplus, trained FLA, full Rust source parity or27B inference claim.

````

## Source: .agents/runs/model-execution-gdn-vllm-resume-20261020/REPORT.md

Original UTF-8 bytes: 18403; SHA256 `778252f090269007b00aaedebb29a3e7b6c4f0e317b6f0da819b1fa17e400063`.
Resolve embedded relative paths against the original path, not this volume.

````text
# Pinned vLLM GDN arithmetic audit

Author: `/root/spine_review`; 2026-10-20. Parent: `numerics-and-context` and
`model-execution`. Investigation/planning only. No implementation, compilation,
tests, imports, model/operator/provider execution or weight/binary/DSO scans.
Only this owned run is written; paused SASS/SiLU work is untouched.

## Decision

Use the pinned vLLM CUDA GDN path as the target. Port its operation graph,
storage boundaries and head mapping into the existing mathematical engine.
Keep the accepted GGUF/GGML CPU schedule and the independently qualified Torch
CPU helpers explicit; their different arithmetic is intentional source behavior.

The first **source-proven postprojection mismatch** against vLLM is convolution
finalization in `crates/zerogravity/src/original_qwen_math.rs:806–813`: the old
diagnostic stores BF16 raw convolution, then applies SiLU and stores BF16 again.
vLLM's CUDA convolution applies SiLU to its FP32 accumulator and stores BF16
once. This statement assumes identical projection results and prior convolution
history. It is not evidence that convolution is the first actual trained-model
divergence: input norm/projection arithmetic can already differ.

Next concrete differences: vLLM beta stays FP32; prefill stores normalized Q/K
as BF16; normal decode keeps normalized Q/K in FP32 and scales Q before readout;
its fused gated RMS keeps normalization, gamma and SiLU multiplication FP32
until one BF16 output. The old diagnostic does not implement those boundaries.

No algebraic state-orientation/head-repeat bug was found in the inspected Rust
AR recurrence. There is no source basis for globally replacing its accepted
GGUF schedule or for treating all existing CPU numerical differences as bugs.

## Exact sources

- vLLM `0.31.0+cu129`, commit
  `db9527a46873454610df6dbedf79a36d6bf1a7f6`.
- Installed source root `V`:
  `/home/pvl/.cache/abird/zerogravity/external-baselines/vllm-0.31.0-cu129/containers/overlay/3d48be3a9656809e23715095eb422187fd959f80ff8ecdeab724e98dfcba213b/diff/usr/local/lib/python3.12/dist-packages`.
- Official plugin commit `e2b8ad532b8b5ea175100202c30430c1d2b5e6a8`;
  source root `P`:
  `.agents/runs/serving-and-speculation-vllm-gguf-baseline-20261020/source`.
- Original HF Python source `H`:
  `.agents/runs/serving-and-speculation-pinned-reference-2026-10-19/pinned-runtime-source/modeling_qwen3_5.py`.
- Source hashes observed during this audit are in `SOURCE_HASHES.md`.

Primary revision links:

- [Qwen3.5 model/packed-role loader](https://github.com/vllm-project/vllm/blob/db9527a46873454610df6dbedf79a36d6bf1a7f6/vllm/model_executor/models/qwen3_5.py).
- [GDN dispatch/cache/projections](https://github.com/vllm-project/vllm/blob/db9527a46873454610df6dbedf79a36d6bf1a7f6/vllm/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py).
- [Packed recurrent decode](https://github.com/vllm-project/vllm/blob/db9527a46873454610df6dbedf79a36d6bf1a7f6/vllm/third_party/flash_linear_attention/ops/fused_recurrent.py).
- [Prefill post-convolution preparation](https://github.com/vllm-project/vllm/blob/db9527a46873454610df6dbedf79a36d6bf1a7f6/vllm/third_party/flash_linear_attention/ops/fused_gdn_prefill_post_conv.py).
- [Chunk graph](https://github.com/vllm-project/vllm/blob/db9527a46873454610df6dbedf79a36d6bf1a7f6/vllm/third_party/flash_linear_attention/ops/chunk.py).
- [Official plugin Qwen mapping](https://github.com/vllm-project/vllm-gguf-plugin/blob/e2b8ad532b8b5ea175100202c30430c1d2b5e6a8/vllm_gguf_plugin/weights_adapter/qwen3_5.py).

## Select the actual path before comparing bits

`V/vllm/envs.py:1224–1235` defaults packed recurrent decode ON and the GDN decode
selector to `cuda`. The selector is not proof that every decode runs the C++
MTP kernel: `qwen_gdn_linear_attn.py:1849–1891` only takes that route when the
MTP metadata guard permits it, otherwise it calls `_forward_core` and the fused
norm separately. Normal decode-only, no speculation, with packed decode enabled
takes `_forward_core_decode_non_spec` (`1267–1288`, `1632–1693`): CUDA causal
convolution followed by the Triton packed recurrent kernel.

On SM86, automatic prefill selection resolves to Triton, not FlashInfer;
`qwen_gdn_linear_attn.py:93–151` restricts its automatic FlashInfer routes to
other device capability/runtime combinations. Explicit overrides, mixed
prefill/decode and speculation must be recorded separately.

## Arithmetic/storage table

Notation: `HK` key/query heads; `HV` value heads; `K` key width; `Vd` value width;
`r = HV/HK`; `R_B` BF16 round-to-nearest-even storage; `S[h,v,k]` recurrent state.
BF16 table entries assume the accepted BF16 model/cache configuration; verify
actual cache dtype separately. Norm epsilon is `config.rms_norm_eps` via
`mamba/gdn/base.py:38`, while Q/K L2 epsilon is independently fixed `1e-6`.

| Stage | Pinned vLLM CUDA | Existing Rust / original HF CPU distinction |
| --- | --- | --- |
| Source roles | Flat Qwen3.5 `Q,K,V,Z`; BA is **B,A**. `qwen3_5.py:224–227`; `qwen_gdn_linear_attn.py:632–640,703–713`. | Old diagnostic's separate QKV/Z/A/B names are correct. Do not copy Qwen3-Next interleaved packing. |
| GQA mapping | Value head `h` uses Q/K head `floor(h/r)`; `fused_recurrent.py:291`. | `repeat_qk_to_value_heads` repeats each Q/K head contiguously; this agrees. No head-order bug proved there. |
| Conv history | K4 convolution uses previous three raw projected samples plus current; cache layout can be dim-first or transposed. Source weight is `[C,1,4]`. | Canonical three-history representation is suitable; CPU original cache's K4 physical buffer needs its explicit codec. History stores raw QKV, never activated QKV. |
| Conv + SiLU | FP32 accumulator; SiLU is `acc/(1+tl.exp(-acc))`; BF16 store only after activation. `causal_conv1d.py:425–478,1000–1093`. | Old diagnostic rounds raw conv before SiLU. Authentic `H:250–289` CPU fallback also produces BF16 `F.conv1d` before separate SiLU; that CPU boundary must remain valid under its own profile. |
| Prefill Q/K | Conv BF16 → FP32 square/sum → reciprocal sqrt of `sum+1e-6` → multiply → **BF16 Q/K store**. `fused_gdn_prefill_post_conv.py:74–115,204–208,238`. | Old diagnostic's GGML grouped RMS produces F32. CPU slow HF promotes Q/K to F32 and uses its own `rsqrt` sequence; no BF16 Q/K store there. |
| Decode Q/K | Conv BF16 → FP32; `q / sqrt(sum(q*q)+1e-6)` and same for K; **remain FP32**. Packed kernel `317–320`. | GGUF uses RMS(`eps/K`) then scale `1/sqrt(K)`, with separate rounded F32 operations and a different reduction. Algebraic equivalence does not imply equal bits. |
| Alpha/decay | `a,b,dt_bias,A_log` loaded/widened to FP32. `x=a+dt_bias`; `g=-exp(A_log)*softplus(x)`; state gain `exp(g)`. | Original HF raw `A_log` and GGUF negative-A storage are different representations. Do not add a second negation or log transform. |
| Prefill softplus | If `x<=20`: positive branch `x+log(1+exp(-x))`, nonpositive branch `log(1+exp(x))`; otherwise `x`. Post-conv kernel `136–140`. | CPU GGUF helper uses `x>20→x`, `x<-20→exp(x)`, otherwise `ln(1+exp(x))`. Pinned Torch CPU softplus is another provider schedule. |
| Decode softplus | `where(x<=20, log(1+tl.exp(x)), x)`; packed kernel `327`. | Do not replace with the stable prefill expression if exact decode bits are the target. Host libm does not establish Triton exp/log parity. |
| Beta | FP32 `tl.sigmoid(b.float())`; prefill buffers explicitly F32. Packed kernel `329`, prep `145,208`. | Old diagnostic rounds beta BF16 at `original_qwen_math.rs:831–835`. That matches original `H:617` BF16 sigmoid output but not vLLM fused CUDA beta. |
| State | Canonical value-major `[sequence,HV,Vd,K]`, key contiguous; FP32 for accepted recurrent cache. Packed kernel `310–313,340–344`. | Rust accepted canonical value-major state agrees. Original HF slow state is `[sequence,HV,K,Vd]`; transpose codec is required, not relabeling. |
| Read scale | Packed decode scales Q FP32 by `K**-0.5` **before** state read reduction (`320,335`). | Accepted GGML AR helper scales completed output dot **after** reduction (`lib.rs:1503–1520`). Preserve it under its qualified source profile. |
| Core return | FP32 recurrent computation; output stored in BF16; state stored in configured recurrent dtype. | Existing diagnostic BF16 core return is appropriate as a boundary, not proof of its internal arithmetic. |
| Gated RMS | FP32 core → `rsqrt(sum(core²)/Vd+eps)` → FP32 norm × widened raw gamma × FP32 `z*sigmoid(z)` → one BF16 store. `layernorm_guard.py:119–176`. | Original CPU `H:229–234` casts normalized core BF16 **before** gamma; BF16 gamma yields BF16 weighted result, F32 gamma promotes it. Closed CPU helpers retain those distinctions. The old diagnostic rounds weighted gamma even when F32 gamma would promote, so it is not a complete CPU-profile implementation either. |
| Initial/prefill core | Chunk64 cumulative log-gates, lower triangular system, solve/WY, BF16 intermediate matrices and FP32 final state; details below. | Existing GGUF AR scalar schedule and original CPU slow F32 triangular-solve schedule are neither a bitwise substitute nor evidence of incorrect vLLM. |

### Exact normal packed-decode recurrence

The following is an operation-level transcription of
`fused_recurrent.py:291–344`, not a portable host implementation or a claim
about the final compiled instructions. Every arithmetic temporary is FP32.

```text
hq = floor(hv / (HV/HK))
q,k,v = widen_BF16(conv_silu_qkv slices)
q = q / sqrt(reduce_sum(q*q) + 1e-6)
k = k / sqrt(reduce_sum(k*k) + 1e-6)
q = q * (K**-0.5)
x = widen(a[hv]) + widen(dt_bias[hv])
sp = select(x <= 20, log(1 + exp(x)), x)
g = -exp(widen(A_log[hv])) * sp
beta = sigmoid(widen(b[hv]))
S[v,k] = S[v,k] * exp(g)
residual[v] = v[v] - reduce_sum_k(S[v,k] * k[k])
delta[v] = residual[v] * beta
S[v,k] = S[v,k] + delta[v] * k[k]
o[v] = reduce_sum_k(S[v,k] * q[k])
store_BF16(o); store_configured_cache_dtype(S)
```

`exp(g)` in this kernel uses `ops/op.py`'s alias: `tl.exp` by default,
`tldevice.fast_expf` if `FLA_USE_FAST_OPS=1`. Its coefficient `exp(A_log)`,
softplus `exp/log` and sigmoid are direct Triton operations in the kernel.
Do not silently equate these with SLEEF, glibc, MKL VML or host Rust `exp`.

**Contraction/reduction uncertainty:** Triton expressions allow compiler fusion
and determine a device reduction tree; they do not independently prove the
exact FMA count, multiply rounding, instruction selection or exp approximation
of a CubeCL rewrite. The accepted Rust GGML helper explicitly uses `mul_add`
and its SIMD/FMA dot tree. Reusing that helper would import a different schedule.
An actual pinned vLLM operator replay plus compiled-kernel inspection is needed
before asserting instruction-exact parity. No private ABI or host-expression
fit is proposed.

### Exact prefill/initial chunk boundaries

Source: `ops/chunk.py:35–99`, `chunk_scaled_dot_kkt.py:90–122`,
`wy_fast.py:75–137`, `chunk_delta_h.py:93–313,367–371`,
`chunk_o.py:101–162`. Chunk size is 64 (`ops/utils.py:31`).

1. Prepare normalized Q/K/V in BF16, beta/log-gate in FP32 as above.
2. FP32 chunk-local cumulative log-gate `G = cumsum(g)`.
3. Strict-lower system `A_ij = beta_i * dot(k_i,k_j) * exp(G_i-G_j)`;
   CUDA source retains promoted FP32 beta-scaled dot operands, outputs FP32.
4. `solve_tril` returns inverse/solved representation **in K dtype BF16**.
   Its FP32 block-merge dot precision is separately selected by
   `FLA_TRIL_PRECISION` (`solve_tril.py:22–26`): default `ieee`, CUDA also
   permits `tf32`/`tf32x3`. Record this setting; it is not implicit in BF16
   output storage. The solved representation is the inverse of `I + A_lower`.
5. Recompute `u` and `w` with BF16 operands/results: beta-scaled V and
   beta/exp-scaled K explicitly cast BF16 before dot; outputs BF16.
6. Chunk state accumulator remains FP32 `[HV,Vd,K]`. Initial state is either
   exact prior or zero according to metadata. Save each chunk's entering state
   as **BF16 h**. Compute `v_new = u - w @ h`; store BF16 v_new. Apply chunk
   decay to FP32 state; cast decay-adjusted delta BF16 before its K/delta dot
   update. Final recurrent state buffer is FP32.
7. Output uses BF16 Q/K/h/v_new dots, FP32 accumulators and causal mask;
   gated attention coefficients cast BF16 before the V dot. Scale both output
   terms, add, then BF16 output store.

This is not merely an AR loop batched in time. A clean port must preserve the
actual mixed-precision WY graph, including the triangular-solve output dtype,
or explicitly stop short of claiming prefill parity. The original CPU slow
fallback instead promotes the whole graph to F32 and uses public Torch
triangular solves (`H:337–430`); its accepted captures remain a separate target.

## Immutable GGUF source transformation is part of the profile

Plugin `weights_adapter/qwen3_5.py:43–65,188–203,282–349`:

- GGUF `ssm_a` contains negative exponentiated decay: stock loader reconstructs
  vLLM `A_log` using `torch.log(-weight)`. Re-exp later need not reproduce the
  original stored GGUF bits. Record the loaded tensor, not only raw source kind.
- Ordinary norm weights subtract one for vLLM's zero-centered loader convention;
  GDN norm gamma is excluded from that transform.
- If `HV/HK>1`, restore tiled V/Z/A/B/conv-V/dt/A_log head rows. Q/K rows stay
  in their existing order. Dense out-projection columns can be restored;
  packed quantized columns cannot be permuted as ordinary scalar columns.
  The latter use `GGUFHeadTilingLayout.input_to_gguf` at runtime.
- `quantization/layout.py:49–91` maps stored repeat-major/group-major head tiles
  to contiguous per-key-head groups and inversely maps the output activation.
  For the approved small models `HV==HK`, this particular head tiling is absent;
  for 27B grouped heads it must be explicit.
- Identical GGUF weights do not imply identical projection arithmetic:
  Zerogravity activation preparation uses Q8_0/Q8_1/Q8_K by codec, while the
  plugin's quantized path uses its own Q8_1/native kernels and cast policy.
  A first GDN comparison should feed both cores the **same captured actual
  projection operands/results**, then qualify the projection route separately.

Actual accepted vLLM original-HF loaded-cast profile previously audited by ROOT
is also distinct: A_log F32; GDN gamma BF16. Neither the CPU official-all-BF16
profile nor the restored-36-F32 profile is that loaded GPU profile. Full-model
claims must bind the chosen loaded source/cast behavior, not infer it from
checkpoint metadata.

## Minimal directly portable work and fastest gates

No implementation is authorized now. Recommended sequence inside the existing
engine once resumed:

1. **One layer0 GDN-core golden from the pinned actual vLLM route.** Capture
   postprojection QKV/Z/A/B, original raw conv history/weights, loaded A_log/dt/
   gamma, conv+SiLU, normalized Q/K, beta/g/gain, entering/proposed state, core
   and gated output. First initial prefill and then genuine cached decode,
   teachers 11→353. Compare plain versus observed full logits/state first.
   This localizes real divergence and avoids changing exponentials first when
   an earlier cast boundary is already wrong.
2. **Port the narrow conv+SiLU and post-conv preparation operation graph.**
   Same PlanBuilder/backend/source leases; FP32 activation until BF16 output;
   prefill/decode Q/K boundaries explicit. Preserve CPU helper semantics under
   `TorchCpuBf16Avx2MixedSourceV1` instead of widening that profile to CUDA.
   A distinct execution profile in the existing enum is enough; no new engine.
3. **Port normal cached recurrence and fused norm.** Canonical `[HV,Vd,K]`
   state, contiguous Q/K head replication, FP32 beta/decay, pre-scaled Q and
   final casts from the table. Original SessionAuthority/Ready/leases publish
   exactly as before. Qualify zero/nonzero prior independently; verify every
   state word and output, not just generated text.
4. **Port the actual chunk64 WY graph** with its BF16 intermediates, then qualify
   lengths 1/2/63/64/65 and continued nonzero prior. Equal mathematics alone
   cannot justify replacing it with the accepted CPU AR/slow chunk schedule.
5. Full layer/full model only after the narrow core plus projection/norm route
   agree with actual vLLM; then full vocabulary and all independently owned
   state domains, cancellation/Ready discard/unknown completion tests.

Meaningful narrow tests:

- Convolution inputs crossing a BF16 rounding boundary before SiLU: genuine
  fused vLLM output must distinguish one final store from the old two-store
  diagnostic. Keep raw state history bit-exact.
- Nonzero beta showing F32 coefficient versus prematurely BF16-rounded beta;
  negative/positive softplus and threshold20 cases in each actual route.
- Q/K zero, very small values, nontrivial sums; enforce fixed `1e-6` and the
  prefill BF16/decode FP32 distinction. Do not substitute model RMS epsilon.
- `HK=2,HV=6` head sentinels prove `[0,0,0,1,1,1]` Q/K grouping; roundtrip only
  the proper GGUF V/head rows and quantized-output activation layout.
- Signed/nonuniform state and Q expose scale-before-reduction and state-axis
  errors; checking only all-zero prior would hide both.
- Gated RMS example where BF16 gamma multiplication differs from an FP32
  product before gating, paired with the existing CPU counterexample. Expected
  vLLM bits come from the actual public kernel, not the proposed port.

License for any later source port: retain vLLM Apache-2.0 notices. Vendored FLA
files additionally identify the original MIT code and copyright
`Songlin Yang, Yu Zhang (2023–2025)`; retain that original notice/license too.
The plugin's source has the vLLM Apache-2.0 contributor notice. This audit copies
no implementation into production.

## Proof boundary

- Proven here: source paths, explicit storage dtypes, head/state axes, branch
  conditions and operation order described above.
- Not proven here: actual selected runtime branch/environment, compiled
  contraction/reduction topology, per-operator vLLM→Rust numerical parity, or
  first causal trained-model difference. No model/operator was executed.
- Prior CPU/GGUF exact receipts stay valid within their own declared schedules;
  they are not vLLM CUDA evidence. Existing owning TODOs remain authoritative.


````

## Source: .agents/runs/model-execution-gdn-vllm-resume-20261020/SOURCE_HASHES.md

Original UTF-8 bytes: 2902; SHA256 `6f34dc03eccaabe45fdc63722c23e31a96748a65d2d3f89befa9f863f37b9a99`.
Resolve embedded relative paths against the original path, not this volume.

````text
# Small source files read for GDN investigation

SHA256; prefixes `V`, `P`, `H` are defined in REPORT.md. No weight/DSO/binary
payload hashed. Observed source bytes only; no new runtime-source attestation.

```text
d261726aae3ce5d52c37b868c317d9377d5e1d7c344961707acc07785a4ec4ab V/vllm/model_executor/models/qwen3_5.py
e326cba09aceca3e247d0b51e6073240d04591ca433dbb20359c720167869fa0 V/vllm/model_executor/layers/mamba/gdn/base.py
3bd2f805f0f4b821e6902ef18da8ba66ad42a7e5a633b76abe96892532293d9c V/vllm/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py
cb16cc9250c4195c09d6d83b43bba607c2d749b621d8d135a920443d1577265f V/vllm/model_executor/layers/mamba/ops/causal_conv1d.py
590717b2107abc90d72548de06b5d315229eb2775c94614f3f545df11cc2b5df V/vllm/third_party/flash_linear_attention/ops/fused_gdn_prefill_post_conv.py
fe6f1311014809040497aa0a623e7fa97f4b3457cdc7ee1f1f8aed21f326f755 V/vllm/third_party/flash_linear_attention/ops/fused_recurrent.py
000ab8996af9788fdb8843a6a3b91833e7a14c8acc0e1ea073a536330f64cb6f V/vllm/third_party/flash_linear_attention/ops/fused_sigmoid_gating.py
c105ead259e86aedc248c53e7c4cef07034d0ff12da608d80265d1dd0990803f V/vllm/third_party/flash_linear_attention/ops/chunk.py
7ae41df3755acc7e925a134d848f78bb5fc00b90069039d86403c5853323ea42 V/vllm/third_party/flash_linear_attention/ops/chunk_scaled_dot_kkt.py
d1a4ff27623a938825052500afa7e2aa24fadd2fa4c9be6376c9128824f86f82 V/vllm/third_party/flash_linear_attention/ops/solve_tril.py
c9fc8259db0a8746dc04f1d799cb73333457118cf89a0bdce8dde8a80a28cd96 V/vllm/third_party/flash_linear_attention/ops/wy_fast.py
1b3ad391f939d9443c6b7adb19e57fe381bd5dccea064e8417a4f85b0e713b26 V/vllm/third_party/flash_linear_attention/ops/chunk_delta_h.py
3b63b836a8cc561723c23db5a528a0b474d9cd78508d610f0b348c585a889a19 V/vllm/third_party/flash_linear_attention/ops/chunk_o.py
b4f696187a1326f9aac44f23d0bb5db814dc28ebe4fc31da000edc24365e09c6 V/vllm/third_party/flash_linear_attention/ops/layernorm_guard.py
456da84fd12411cf53c96a90dd4f78f49afb454c3d51d2c524ea8e7106fe8ae5 V/vllm/third_party/flash_linear_attention/ops/op.py
1d7101800f67b803a700fbf7489292cf1f0022b570085166ad407e6b1513dc15 V/vllm/envs.py
ed19007102c518a50e89a92fa2084c4034980f59aabd0881ce13df78df1fa2d5 P/vllm_gguf_plugin/weights_adapter/qwen3_5.py
6ffb4c25b6e862a3953ff8d00fba3c9a59febbafd791cb15ae0c44f0404ef799 P/vllm_gguf_plugin/quantization/layout.py
762feb6c7426a7f15b5bf830df54c07438bf9e7c27b8cdb23179045920412c3b H
cb7f6a80d1d602e26045aec3584ad53f49b9386808eae1db58bb4493137b1266 crates/zerogravity/src/original_qwen_math.rs
848a0bd39e3b98b7209499e86abae8fa96af61b7eea48208bae72290a7e65d5b crates/zerogravity/src/original_qwen_precision.rs
9a26849b304a8e9dfdd57478414c7845c6069f41a8d49c48742ec6debae22c4b crates/zerogravity-model-qwen35/src/lib.rs
a96a993f73df31ff56ec20b3704480f48c4dc03c8e62d016fa83956d88aa5820 crates/zerogravity-model-qwen35/src/forward.rs
```


````

## Source: .agents/runs/model-execution-gdn-vllm-resume-20261020/TODO.md

Original UTF-8 bytes: 998; SHA256 `a5362847370d855a8486d0c98fc3030b8642d3aeeab8adedac87723fe779f999`.
Resolve embedded relative paths against the original path, not this volume.

````text
# GDN resume source investigation

Parent plans: `numerics-and-context`, `model-execution`.
Owner: `/root/spine_review`. Scope: pinned-source review and own report only.

- [x] Read pinned vLLM flat packing, convolution, prefill/decode coefficients,
  normalization, recurrence, WY/chunk graph and fused norm.
- [x] Read official plugin source transforms/head tiling and live Rust CPU path.
- [x] Report conditional earliest postprojection cast mismatch, full arithmetic
  table, exact recurrence, profile distinctions and minimal next parity gates.
- [x] Preserve all paused SASS/SiLU source, MAIN and shared plans.
- [ ] ROOT consolidate supporting design/queue; no production edits authorized.
- [ ] Actual source-bound vLLM operator/full-model parity remains unexecuted by
  this run. No compiled-contraction/reduction or eligibility claim.

Findings and source-only limits: `REPORT.md`; observed small source pins:
`SOURCE_HASHES.md`. No build/test/runtime failures occurred because none ran.

````

## Source: .agents/runs/model-execution-hf-bf16-gamma-gated-rms-20261020/API.md

Original UTF-8 bytes: 3372; SHA256 `a7e100f1d5a0b5803943af18e778f5136716fd4ff20afcb98c80b080c0b87df9`.
Resolve embedded relative paths against the original path, not this volume.

````text
# Original27B BF16 gamma gated RMS — proposed closed operator

- Owner: spine_review; parent: model-execution. Source-only, no MAIN changes.
- Exact existing CPU profile + new `GatedRmsGammaBf16Contract::TorchCpuBf16Avx2H48Width128`.
- New typed node: two BF16 `[48,128]` values → one BF16 `[48,128]` value; no state effects/frontier/publication.
- Actual role: original layer0 `linear_attn.norm.weight`, BF16 `[128]`, 256 bytes; immutable verified window issuer, full original descriptor and source extent. H48/key16, widths/axes/kinds/source/epsilon checked before input admission or recording.
- Existing F32 gamma `head_rms_128` remains exact. No global codec guard widening.

Arithmetic:

1. BF16 core → F32 squares → accepted `sum_inner_128` → F32 `/128`, `+1e-6`, sqrt/divide1.
2. F32 normalized product → BF16 RNE.
3. **BF16 raw gamma × normalized BF16 → BF16 RNE here**.
4. Accepted SLEEF SiLU on widened BF16 gate → F32; separate F32 product → BF16 RNE.

The former F32 gamma path retains its F32 weighted product; it is not this contract. The new CPU helper reuses the existing reducer, finite RNE and SiLU implementations. No host-exp, provider callback, new model loop or learned parameter cache.

APIs:

- CPU `norm::gated_rms_128_bf16_gamma_h48(SourceIdentity, SourceTensorRef, core, gate)`: seven private stage arrays, borrowed getters only.
- Backend default `gated_rms_gamma_bf16_profiled`: immediate refusal; PlanBuilder records the explicit contract.
- `OriginalWindowOperators::execute_gated_rms_bf16_gamma_slice(0, GatedRmsSliceInput, cancel)`: SAME generic forward/HostMath, original sealed raw parameter window, retained input and result backing; borrowed outputs only.
- `LoadedQwenModel::append_gated_rms_bf16_gamma_slice`: record-only pure fragment. Complete original-HF execution/session guard is unchanged.

Admission/lifetime:

- Before input Vecs: Output49,152 + Scratch65,536 bytes under the source's actual host arbiter/owner.
- Before helper/output allocation: Output24,576 + Scratch983,040 bytes; this existing conservative floor covers all helper arrays, squares/sums, stack gamma and final copy (well below floor).
- Gamma256-byte staged window plus actual extent-offset metadata charged separately, transiently, through existing read_parameter.
- Retained inputs/results keep their original immutable source issuer and aggregate reservations; data drops before its lease. Final external output drop retires all owned reservations. No `into_vec` escape.
- Cancel before import, after import and after compute fails closed without publishing any root. Provider adapters and strided inputs refuse before callbacks/math; same-source exact role guards precede result admission.

Prepared tests (not executed): 3 CPU +3 core +3 model full-header fixtures. Literal dyadic witness `253/256 *129/128` has F32 `3f7efa00`, but required BF16 product `3f7f0000`; source codec/extent/source/profile, valid-geometry strided pack, optional provider, pre/post-import cancellation, one-byte-short aggregate admission and actual owner retention/final0 are separate fixtures. Synthetic fixtures are wiring evidence only.

CUDA/Vulkan typed lowering refuses this node cold; no numerical fallback. No A_log/exp, initial/cached recurrence producer, full GDN, original27B 112 StateRoots, full model/session or GPU/performance eligibility.

````

## Source: .agents/runs/model-execution-hf-bf16-gamma-gated-rms-20261020/ROOT.md

Original UTF-8 bytes: 3112; SHA256 `59b86e0dc6f6c9aa3ae514a0002d0fed966e667102d5a3a355deb830c58f7925`.
Resolve embedded relative paths against the original path, not this volume.

````text
# ROOT gates — no execution by author

1. Review exact16-path delta and baseline hashes. Compose narrow parent hunks with concurrent cached/chunk work; preserve every other parent path and all default GGUF bytes.
2. Focused core, CPU original_hf norm and model `cpu,original-hf-binding` tests; strict default/CPU/CUDA/Vulkan checks. New node needs all3 explicit GPU refusal matches. Existing full isolated gate and original approved parity routes remain required before implementation acceptance.
3. `replay-bindings.json` is **metadata only** from independently accepted original27B V3; author read no archive/model/DSO payloads. ROOT must authenticate all prior manifest/invocation/validation/preparation/source metadata and complete archive payload hashes again before extracting any expected data. Do not synthesize expected values from candidate.
4. For each actual teacher11 initial and teacher353 truecached call, take exactly `norm.input.hidden_states` + `norm.input.gate` BF16 raw6144 each; authenticate actual original gamma BF16 `[128]`/256B from the SAME complete verified source window. Require source gamma bytes equal captured actual raw gamma hash70d11f29...; no converted F32 gamma reference or tensor aliases.
5. Replay SAME typed `execute_gated_rms_bf16_gamma_slice` with genuine original factory/source owner. Compare both retained input arrays and all6144 final BF16 output bits against the original call. Keep core/gate external; this does not qualify a live GDN/recurrence producer.
6. Separately, under an admitted aggregate diagnostic workspace, call the exact same new CPU helper with that genuine source parameter view to compare all7 stage getters (mean48/inverse48 and5×6144 arrays) to bound actual dispatch/frame outputs. `replay-bindings.json` resolves by scope/source-line/operator, independently cross-checks semantic frame weighted/return aliases. 14 stage checks/61,632 words +2 typed-final checks/12,288 words, all bit exact; no tolerance. Report first/count/maxabs mismatch and UNQUALIFIED on any difference.
7. Use unchanged accepted bounded native lifetime helper and original18-shard sealed window factory/source-pin metadata, no complete55GB Vec/mmap or fake admission. Reserve copied raw/expected/emitted buffers before allocation; account exact temporary gamma staging separately. ROOT must pin local source-pin.json in CARGO_MANIFEST_DIR if reusing the factory. Current configured128MiB host pool and bounded512MiB native process suffice by explicit arithmetic only, not a claim of physical available RAM or observed peak.
8. Require actual pre-cancel, after-import cancel, one-byte-short real ResourceArbiter refusal, unchanged owner/control reservations, retained private output after caller/source drop, no fence and final reserved0. These are pure operator/source-window gates; no whole StateRoot/model proof.

Unchanged boundaries: F32 gamma helper; native original GPU queue/Ready/quarantine; GGUF paths; exp/A_log; initial chunk; canonical state encoding/publication; trained full-HF/session eligibility and original27B112-root/full model/performance gates.

````

## Source: .agents/runs/model-execution-hf-bf16-gamma-gated-rms-20261020/TODO.md

Original UTF-8 bytes: 2172; SHA256 `111483adf3fdc4f67e17a0a296b7fc3f6b424bf1d22a62bc37d8c828ab378429`.
Resolve embedded relative paths against the original path, not this volume.

````text
# Original27B BF16 gamma gated RMS

Parent: model-execution; owner spine_review. Ignored source-only candidate.

- [x] Inspect genuine original V3 profile/shape metadata and existing CPU/typed seams.
- [x] Preserve existing F32 gamma prefix, SLEEF SiLU and accepted reducers/RNE.
- [x] Coordinate disjoint ownership with Batch GPU orderedSiLU; Baseline native chunk ownership is disjoint.
- [x] Distinct H48 BF16gamma-before-gate rounding helper/contract/typedwindow slice.
- [x] Original raw parameter issuer/extent, aggregate leases, cancellation/retention.
- [x] Default/backend/CUDA/Vulkan cold refusal; all full-model/exp/session guards closed.
- [x] Source-free rounding/type/source/pressure/lastdrop tests prepared, not executed.
- [x] Authentic originalV3 initial/cached6144-word metadata-bound replay preparation.
- [x] Freeze16-path candidate c89eb6c0 / manifest43c400a2; sent ROOT +Batch for independent source review.
- [ ] Independent review, source integration, actual focused/strict/operator/full gates by coordinator.

No MAIN/sharedplan, build/test/Torch/model/provider/GPU/runtime or payload hashes.

## Additive followups

- [x] Batch complete source review43c400a2: only2 redundant test drop(math) calls; no math/owner/cold-refusal blocker.
- [x] Test-only removal: test-style-v2 manifest7f266003 /patch86384ec0; original16-path freeze unchanged.
- [x] ROOT-only authentic two-phase replay: replay/prepared-manifest f1c1f28c /NEW_PREPARATION59a8d2fb,13files; source-only AST7/rustfmt main;8fixtures prepared, not executed.
- [x] Full actual V3 reference metadata/source/runtime/lifecycle pins;14 primitive stage checks61,632words versus2 typed-final checks12,288words +24,576 retained inputwords.
- [x] Original host control before native metadata/raw copies, full-archive retainedFD streaming and bounded stage reads; original sourcegamma extent/raw bytes, aggregate pressure/cancel/retention/final0 protocol prepared.
- [ ] Independent replay source review; ROOT build/8fixtures/trained H48 native execution and independent raw validation.

No MAIN/source integration, runtime qualification or full-GDN/model/StateRoot/exp/GPU/performance promotion.

````

## Source: .agents/runs/model-execution-hf-bf16-gamma-gated-rms-20261020/replay/ROOT.md

Original UTF-8 bytes: 5568; SHA256 `b2ff32e5a5eaaf8c82a3d2e04b368289afaa50fd9b740615a3377c83bff97eae`.
Resolve embedded relative paths against the original path, not this volume.

````text
# ROOT-only authentic H48 gated RMS qualification

- Preparation only. Author read small source/JSON metadata, not tensor archives, learned payloads or DSO bytes; no model/provider/operator/tests/build/runtime.
- Dependency: reviewed16-path43c400a2 /c89eb6c0 candidate +test-only7f266003 /86384ec0 followup. Full HF/session/GDN/initial/exp/GPU guards stay closed.
- Original teacher11 INITIAL paddedchunk64 and teacher353 TRUECACHED original module calls; both norm inputs are actual BF16 `[48,128]`, original gamma BF16 `[128]`/256B. Exact accepted V3 source/runtime/invocation/manifest/validation/prevalidation pins in source-bindings.json. Metadata codec names are normalized only (`torch.bfloat16`→Safetensors`BF16`); original_torch_dtype retained, no numerical cast.

Scope/counts:

- Same `execute_gated_rms_bf16_gamma_slice`: 2×6144 =12,288 independent final words; 2×(core6144+gate6144)=24,576 retained input words.
- SAME new CPU helper with SAME original sealed source gamma window: 2×(mean48+inverse48+5×6144)=61,632 words /14 checks. Separate primitive receipts; no alternate fallback/model loop.
- Both primitive outputs and typed finals are independently compared to untouched actual public archive words. No tolerance or candidate-derived expectation. Original signed zeros/subnormals retained; selected nonfinite stages refuse rather than filter.
- Source authenticity does not create a live GDN/core/exp producer or certify full27B112StateRoots/session math.

Resources/lifetime:

- Native `ResourceArbiter`128MiB host pool; original factory reserves64MiB source metadata, original binding8MiB. Native8MiB control charge precedes metadata/archive/source reads and selected payload copies. This bounds private admitted payload/offset/control buffers, not OS allocator bookkeeping/page cache/free physical RAM.
- Before primitive decode/helper arrays: additional1MiB original-source Scratch. Raw gamma staged through existing charged data+offset window and compared exactly to original public raw256B. No fabricated F32gamma ref.
- Before each typed import/output: original aggregate tickets (input49,152+65,536scratch; result24,576+983,040scratch). Both live outputs retain all6 actual data arrays and original source, long after source/operator callers drop. Temporary diagnostics drop before live ledger checks.
- Native parses/authenticates each full archive via retained FD and64KiB hash buffer, bounds header before read, checks complete dtypes/shapes/spans/no trailing bytes, and reads at most24,576B per selected stage. Never copies whole47/27MiB archives. Same FD stat/hash rechecked after every selection/fullproof. Python metadata preparation opens no raw archive. Independent raw validator uses bounded same-FD streaming too.
- Exact steady80MiB; two live typed results80MiB+2,244,608B; retained72MiB+2,244,608B; actual final reserved0/reservations0/no fence. Configured pool is not physical RAM admission.
- Real pre-cancel/wronglayer and ONE-BYTE-short aggregate pressure checked for each teacher; failures restore exact live ledger. Deterministic after-import cancellation remains the prepared internal model fixture, not claimed as an external native timing test.
- Exact accepted owned_lifetime.py; owned_native.py differs only in the scope label. Same WNOWAIT/TERM/KILL-before-reap/PGID/pipe ownership,512MiB RLIMIT_AS, four actual CPU affinity IDs,900s child/2400s outer/4MiB log. No Torch/container/GPU/new engine.

ROOT sequence (never author execution):

1. Review implementation +harness +source bindings. Run8source-free fixtures (test_contracts.py) and focused/strict source tests after integration. All original freezes remain immutable.
2. Copy exact local factory.rs/source-pin.json into this Cargo manifest directory. source-pin.json is accepted7058B4c74fa28; factory is exact existing2302f6a0 copy. It scans all18 original shards through existing admitted window factory; no55GB Vec/mmap.
3. Compile this standalone Cargo target using coordinator's existing controlled build; optional rustc against freshly built MAIN libraries must preserve the same flags/full linked-source receipt. No downloads. ROOT build-source attestation must list ALL required build paths, exact new children/final repaired test and unchanged numerical sources, plus local Cargo/main/factory/source-pin/source-bindings. Status ROOT_REVIEWED_MERGED_BF16_GAMMA_GATED_RMS_SOURCE; prepared_manifest_sha256; false full_model_StateRoots_exp_initial_GDN_GPU_or_performance_qualified. Caller attestation is review binding, not independent build proof.
4. `python3 prepare_reference.py --new-output "$RUN/root-reference-metadata-1"`: metadata only. Actual raw archives are authenticated/extracted in admitted Rust work, not this phase. No independent raw PASS yet.
5. `python3 run_replay.py --binary "$BINARY" --binary-sha256 "$EXACT_SHA" --reference-directory "$RUN/root-reference-metadata-1" --implementation "$ROOT_BUILD_RECEIPT" --new-output "$RUN/root-native-1"`. Only ROOT runs, after active timing/phase gates.
6. `python3 validate_replay.py --output "$RUN/root-native-1" --reference-directory "$RUN/root-reference-metadata-1" --implementation "$ROOT_BUILD_RECEIPT"`. Independent stdlib full signed-bit checks/whole archive SHA/source/receipts/lifecycle/ledger/tree. Final native artifact tree<=8MiB, validation receipt reserved64KiB.

Acceptance is only `PASS_PRIVATE_ORIGINAL27B_H48_BF16_GAMMA_GATED_RMS_ONLY`. No full-GDN, A_log/exp, initial chunk/recurrence, canonical/root publication, trained whole model, GPU or performance eligibility.

````

## Source: .agents/runs/model-execution-hf-bf16-gamma-gated-rms-20261020/replay/frozen-prepared/ROOT.md

Original UTF-8 bytes: 5568; SHA256 `b2ff32e5a5eaaf8c82a3d2e04b368289afaa50fd9b740615a3377c83bff97eae`.
Resolve embedded relative paths against the original path, not this volume.

````text
# ROOT-only authentic H48 gated RMS qualification

- Preparation only. Author read small source/JSON metadata, not tensor archives, learned payloads or DSO bytes; no model/provider/operator/tests/build/runtime.
- Dependency: reviewed16-path43c400a2 /c89eb6c0 candidate +test-only7f266003 /86384ec0 followup. Full HF/session/GDN/initial/exp/GPU guards stay closed.
- Original teacher11 INITIAL paddedchunk64 and teacher353 TRUECACHED original module calls; both norm inputs are actual BF16 `[48,128]`, original gamma BF16 `[128]`/256B. Exact accepted V3 source/runtime/invocation/manifest/validation/prevalidation pins in source-bindings.json. Metadata codec names are normalized only (`torch.bfloat16`→Safetensors`BF16`); original_torch_dtype retained, no numerical cast.

Scope/counts:

- Same `execute_gated_rms_bf16_gamma_slice`: 2×6144 =12,288 independent final words; 2×(core6144+gate6144)=24,576 retained input words.
- SAME new CPU helper with SAME original sealed source gamma window: 2×(mean48+inverse48+5×6144)=61,632 words /14 checks. Separate primitive receipts; no alternate fallback/model loop.
- Both primitive outputs and typed finals are independently compared to untouched actual public archive words. No tolerance or candidate-derived expectation. Original signed zeros/subnormals retained; selected nonfinite stages refuse rather than filter.
- Source authenticity does not create a live GDN/core/exp producer or certify full27B112StateRoots/session math.

Resources/lifetime:

- Native `ResourceArbiter`128MiB host pool; original factory reserves64MiB source metadata, original binding8MiB. Native8MiB control charge precedes metadata/archive/source reads and selected payload copies. This bounds private admitted payload/offset/control buffers, not OS allocator bookkeeping/page cache/free physical RAM.
- Before primitive decode/helper arrays: additional1MiB original-source Scratch. Raw gamma staged through existing charged data+offset window and compared exactly to original public raw256B. No fabricated F32gamma ref.
- Before each typed import/output: original aggregate tickets (input49,152+65,536scratch; result24,576+983,040scratch). Both live outputs retain all6 actual data arrays and original source, long after source/operator callers drop. Temporary diagnostics drop before live ledger checks.
- Native parses/authenticates each full archive via retained FD and64KiB hash buffer, bounds header before read, checks complete dtypes/shapes/spans/no trailing bytes, and reads at most24,576B per selected stage. Never copies whole47/27MiB archives. Same FD stat/hash rechecked after every selection/fullproof. Python metadata preparation opens no raw archive. Independent raw validator uses bounded same-FD streaming too.
- Exact steady80MiB; two live typed results80MiB+2,244,608B; retained72MiB+2,244,608B; actual final reserved0/reservations0/no fence. Configured pool is not physical RAM admission.
- Real pre-cancel/wronglayer and ONE-BYTE-short aggregate pressure checked for each teacher; failures restore exact live ledger. Deterministic after-import cancellation remains the prepared internal model fixture, not claimed as an external native timing test.
- Exact accepted owned_lifetime.py; owned_native.py differs only in the scope label. Same WNOWAIT/TERM/KILL-before-reap/PGID/pipe ownership,512MiB RLIMIT_AS, four actual CPU affinity IDs,900s child/2400s outer/4MiB log. No Torch/container/GPU/new engine.

ROOT sequence (never author execution):

1. Review implementation +harness +source bindings. Run8source-free fixtures (test_contracts.py) and focused/strict source tests after integration. All original freezes remain immutable.
2. Copy exact local factory.rs/source-pin.json into this Cargo manifest directory. source-pin.json is accepted7058B4c74fa28; factory is exact existing2302f6a0 copy. It scans all18 original shards through existing admitted window factory; no55GB Vec/mmap.
3. Compile this standalone Cargo target using coordinator's existing controlled build; optional rustc against freshly built MAIN libraries must preserve the same flags/full linked-source receipt. No downloads. ROOT build-source attestation must list ALL required build paths, exact new children/final repaired test and unchanged numerical sources, plus local Cargo/main/factory/source-pin/source-bindings. Status ROOT_REVIEWED_MERGED_BF16_GAMMA_GATED_RMS_SOURCE; prepared_manifest_sha256; false full_model_StateRoots_exp_initial_GDN_GPU_or_performance_qualified. Caller attestation is review binding, not independent build proof.
4. `python3 prepare_reference.py --new-output "$RUN/root-reference-metadata-1"`: metadata only. Actual raw archives are authenticated/extracted in admitted Rust work, not this phase. No independent raw PASS yet.
5. `python3 run_replay.py --binary "$BINARY" --binary-sha256 "$EXACT_SHA" --reference-directory "$RUN/root-reference-metadata-1" --implementation "$ROOT_BUILD_RECEIPT" --new-output "$RUN/root-native-1"`. Only ROOT runs, after active timing/phase gates.
6. `python3 validate_replay.py --output "$RUN/root-native-1" --reference-directory "$RUN/root-reference-metadata-1" --implementation "$ROOT_BUILD_RECEIPT"`. Independent stdlib full signed-bit checks/whole archive SHA/source/receipts/lifecycle/ledger/tree. Final native artifact tree<=8MiB, validation receipt reserved64KiB.

Acceptance is only `PASS_PRIVATE_ORIGINAL27B_H48_BF16_GAMMA_GATED_RMS_ONLY`. No full-GDN, A_log/exp, initial chunk/recurrence, canonical/root publication, trained whole model, GPU or performance eligibility.

````

## Source: .agents/runs/model-execution-hf-bf16-gamma-gated-rms-20261020/replay/isolated-root-v2/ISOLATED_ROOT.md

Original UTF-8 bytes: 5796; SHA256 `6ca8b67cd33a1d95d5985321beff1e32f9ebef9fbb249cb0b102a529a583816a`.
Resolve embedded relative paths against the original path, not this volume.

````text
# Isolated ROOT replay source relocation — coordinator only

- Additive preparation; original13 files/freeze/frozen copies remain immutable. `ROOT.md` is retained byte-identical provenance. This file supersedes only its MAIN merge/build-path instructions.
- Target `/home/pvl/spaces/abird/src/zerogravity-worktrees/root-bf16-composition-20261020`, final ROOT source freeze7 `036ce0dc73bf0d1671d521059fac0b75c6aa62bc3e8b30475a576ae8d7d7e615` (521 Rust/manifests files). MAIN remains unchanged by this preparation.
- All main.rs/factory/source-pin/source-bindings/reference exporter/raw validator/runner/owned lifetime/old fixtures are exact original bytes. Cargo changes only3 or4 absolute crate dependency roots. contracts.py changes only frozen()/implementation() source attestation dispatch; every finite/geometry/raw-word/count/source/lifetime/numerical check remains exact original.
- Exact original parent/stride/H48/source binding checks still execute through the authenticated original pure metadata contract. Original13 live/frozen files are revalidated. No old MERGED receipt is accepted.
- relocation.json conserves every original required path and expected child. Whole521 current source inventory must match the exact final ROOT freeze; each recorded adapter/test exception is an exact unique text delta against immutable parent source. Eight exceptions cover only ROOT corrections2–7. The sole production adapter adds checked coefficient output-size arguments with the same old floors; all other exceptions are tests. A retained attention test baseline is byte-bound to the original ROOT freeze1, without a mutable MAIN dependency.
- ROOT already executed strict selected CPU/core/model and237 source-free focused tests (0fail/2ignored); exact root-cpu-focused-7.json is pinned. These do not qualify trained native replay, full HF/session/StateRoots/exp/initial GDN/GPU/performance.

## ROOT sequence

1. Review these18 files, original preparation and recorded ROOT exceptions. Run `python3 -m unittest -v test_relocation.py test_contracts.py` **from this directory**, coordinator only. Fourteen new relocation fixture methods are prepared, not executed by the author. They use tiny synthetic source files/metadata, no model/provider/operator/runtime. Whole521 fixture cardinality is a metadata inventory test.
2. Build this local Cargo manifest with the coordinator's existing controlled/offline source closure. `Cargo.toml` points only at the isolated ROOT crates, with original dependencies/features unchanged. Do not compile against MAIN or label the source MAIN-merged. A fresh local Cargo.lock, if created, is an additional mandatory actual build input. Actual compiler/flags/external-package/binary binding remains the coordinator's build evidence.
3. After actual ROOT source/build review, create an **isolated source attestation** with the fields below. Hash every final file after build, before execution. This review receipt is not an independent proof of the binary's compiler inputs.
4. Run the unchanged reference metadata preparation, native replay and independent validator shown in retained ROOT.md, **using the scripts in this directory** and the new isolated implementation receipt. Use fresh exclusive output directories. The original reference source/runtime/archive pins and full raw word/lifetime/negative limits are unchanged; raw payload authentication occurs only in the admitted native worker and independent validator, never source preparation.
5. Revalidate source freeze/receipt after execution via the unchanged runner/validator. Any source mismatch, missing/excess build path, old MAIN status, unexpected arithmetic delta or lifetime failure refuses acceptance.

Required receipt fields:

```json
{
  "status": "ROOT_REVIEWED_ISOLATED_BF16_GAMMA_GATED_RMS_SOURCE",
  "prepared_manifest_sha256": "SHA256 of this directory's prepared-manifest.json",
  "original_prepared_manifest_sha256": "f1c1f28c21616fb1057dad082a5b27a42a587bdac95ec9603d7f0c316c4eb8d4",
  "root_source_freeze_sha256": "036ce0dc73bf0d1671d521059fac0b75c6aa62bc3e8b30475a576ae8d7d7e615",
  "workspace_root": "/home/pvl/spaces/abird/src/zerogravity-worktrees/root-bf16-composition-20261020",
  "MAIN_merged": false,
  "full_model_StateRoots_exp_initial_GDN_GPU_or_performance_qualified": false,
  "files": ["exact path/bytes/sha256 rows described below"]
}
```

- `files` = all521 final ROOT freeze entries with workspace-root absolute paths, plus this directory's exact Cargo.toml/main.rs/factory.rs/source-pin.json/source-bindings.json; plus this directory's Cargo.lock if present. Exactly526 or527 unique rows, no MAIN aliases/missing/unlisted source. `relocated_source.workspace` returns the authenticated521 rows; `relocation.json.copy_map[*].relocated` supplies the local5. All expected-child assertions (including any repaired test child) must match relocation.json. The actual optional local lock must be pinned with the same2MiB BEFORE-read/fstat/regular-file policy.
- Source-only relocation preserves independent input/result owner charges, cancellation/pressure/refusal probes, retained result/source lifetime, final reserved0 and all raw exact-bit limits. No allowance to treat caller attestation as independent runtime or numerical proof.

Acceptance scope: SAME execute_gated_rms_bf16_gamma_slice; original initial11/cached353 typed finals12,288 words +retained inputs24,576 words; SAME helper fourteen stage checks61,632 words =2×(48+48+5×6144). BF16 gamma product rounding before gate is unchanged. Original25 required paths and9 expected children conserved. Diagnostic primitive parameter/window/stage owners retire on primitive() return before caller ledger/final0 checks. No initial GDN/exp/recurrence/full27B112StateRoots/session/GPU/performance eligibility.

````

## Source: .agents/runs/model-execution-hf-bf16-gamma-gated-rms-20261020/replay/isolated-root-v2/ROOT.md

Original UTF-8 bytes: 5568; SHA256 `b2ff32e5a5eaaf8c82a3d2e04b368289afaa50fd9b740615a3377c83bff97eae`.
Resolve embedded relative paths against the original path, not this volume.

````text
# ROOT-only authentic H48 gated RMS qualification

- Preparation only. Author read small source/JSON metadata, not tensor archives, learned payloads or DSO bytes; no model/provider/operator/tests/build/runtime.
- Dependency: reviewed16-path43c400a2 /c89eb6c0 candidate +test-only7f266003 /86384ec0 followup. Full HF/session/GDN/initial/exp/GPU guards stay closed.
- Original teacher11 INITIAL paddedchunk64 and teacher353 TRUECACHED original module calls; both norm inputs are actual BF16 `[48,128]`, original gamma BF16 `[128]`/256B. Exact accepted V3 source/runtime/invocation/manifest/validation/prevalidation pins in source-bindings.json. Metadata codec names are normalized only (`torch.bfloat16`→Safetensors`BF16`); original_torch_dtype retained, no numerical cast.

Scope/counts:

- Same `execute_gated_rms_bf16_gamma_slice`: 2×6144 =12,288 independent final words; 2×(core6144+gate6144)=24,576 retained input words.
- SAME new CPU helper with SAME original sealed source gamma window: 2×(mean48+inverse48+5×6144)=61,632 words /14 checks. Separate primitive receipts; no alternate fallback/model loop.
- Both primitive outputs and typed finals are independently compared to untouched actual public archive words. No tolerance or candidate-derived expectation. Original signed zeros/subnormals retained; selected nonfinite stages refuse rather than filter.
- Source authenticity does not create a live GDN/core/exp producer or certify full27B112StateRoots/session math.

Resources/lifetime:

- Native `ResourceArbiter`128MiB host pool; original factory reserves64MiB source metadata, original binding8MiB. Native8MiB control charge precedes metadata/archive/source reads and selected payload copies. This bounds private admitted payload/offset/control buffers, not OS allocator bookkeeping/page cache/free physical RAM.
- Before primitive decode/helper arrays: additional1MiB original-source Scratch. Raw gamma staged through existing charged data+offset window and compared exactly to original public raw256B. No fabricated F32gamma ref.
- Before each typed import/output: original aggregate tickets (input49,152+65,536scratch; result24,576+983,040scratch). Both live outputs retain all6 actual data arrays and original source, long after source/operator callers drop. Temporary diagnostics drop before live ledger checks.
- Native parses/authenticates each full archive via retained FD and64KiB hash buffer, bounds header before read, checks complete dtypes/shapes/spans/no trailing bytes, and reads at most24,576B per selected stage. Never copies whole47/27MiB archives. Same FD stat/hash rechecked after every selection/fullproof. Python metadata preparation opens no raw archive. Independent raw validator uses bounded same-FD streaming too.
- Exact steady80MiB; two live typed results80MiB+2,244,608B; retained72MiB+2,244,608B; actual final reserved0/reservations0/no fence. Configured pool is not physical RAM admission.
- Real pre-cancel/wronglayer and ONE-BYTE-short aggregate pressure checked for each teacher; failures restore exact live ledger. Deterministic after-import cancellation remains the prepared internal model fixture, not claimed as an external native timing test.
- Exact accepted owned_lifetime.py; owned_native.py differs only in the scope label. Same WNOWAIT/TERM/KILL-before-reap/PGID/pipe ownership,512MiB RLIMIT_AS, four actual CPU affinity IDs,900s child/2400s outer/4MiB log. No Torch/container/GPU/new engine.

ROOT sequence (never author execution):

1. Review implementation +harness +source bindings. Run8source-free fixtures (test_contracts.py) and focused/strict source tests after integration. All original freezes remain immutable.
2. Copy exact local factory.rs/source-pin.json into this Cargo manifest directory. source-pin.json is accepted7058B4c74fa28; factory is exact existing2302f6a0 copy. It scans all18 original shards through existing admitted window factory; no55GB Vec/mmap.
3. Compile this standalone Cargo target using coordinator's existing controlled build; optional rustc against freshly built MAIN libraries must preserve the same flags/full linked-source receipt. No downloads. ROOT build-source attestation must list ALL required build paths, exact new children/final repaired test and unchanged numerical sources, plus local Cargo/main/factory/source-pin/source-bindings. Status ROOT_REVIEWED_MERGED_BF16_GAMMA_GATED_RMS_SOURCE; prepared_manifest_sha256; false full_model_StateRoots_exp_initial_GDN_GPU_or_performance_qualified. Caller attestation is review binding, not independent build proof.
4. `python3 prepare_reference.py --new-output "$RUN/root-reference-metadata-1"`: metadata only. Actual raw archives are authenticated/extracted in admitted Rust work, not this phase. No independent raw PASS yet.
5. `python3 run_replay.py --binary "$BINARY" --binary-sha256 "$EXACT_SHA" --reference-directory "$RUN/root-reference-metadata-1" --implementation "$ROOT_BUILD_RECEIPT" --new-output "$RUN/root-native-1"`. Only ROOT runs, after active timing/phase gates.
6. `python3 validate_replay.py --output "$RUN/root-native-1" --reference-directory "$RUN/root-reference-metadata-1" --implementation "$ROOT_BUILD_RECEIPT"`. Independent stdlib full signed-bit checks/whole archive SHA/source/receipts/lifecycle/ledger/tree. Final native artifact tree<=8MiB, validation receipt reserved64KiB.

Acceptance is only `PASS_PRIVATE_ORIGINAL27B_H48_BF16_GAMMA_GATED_RMS_ONLY`. No full-GDN, A_log/exp, initial chunk/recurrence, canonical/root publication, trained whole model, GPU or performance eligibility.

````

## Source: .agents/runs/model-execution-hf-bf16-gamma-gated-rms-20261020/replay/isolated-root-v2/frozen-prepared/ISOLATED_ROOT.md

Original UTF-8 bytes: 5796; SHA256 `6ca8b67cd33a1d95d5985321beff1e32f9ebef9fbb249cb0b102a529a583816a`.
Resolve embedded relative paths against the original path, not this volume.

````text
# Isolated ROOT replay source relocation — coordinator only

- Additive preparation; original13 files/freeze/frozen copies remain immutable. `ROOT.md` is retained byte-identical provenance. This file supersedes only its MAIN merge/build-path instructions.
- Target `/home/pvl/spaces/abird/src/zerogravity-worktrees/root-bf16-composition-20261020`, final ROOT source freeze7 `036ce0dc73bf0d1671d521059fac0b75c6aa62bc3e8b30475a576ae8d7d7e615` (521 Rust/manifests files). MAIN remains unchanged by this preparation.
- All main.rs/factory/source-pin/source-bindings/reference exporter/raw validator/runner/owned lifetime/old fixtures are exact original bytes. Cargo changes only3 or4 absolute crate dependency roots. contracts.py changes only frozen()/implementation() source attestation dispatch; every finite/geometry/raw-word/count/source/lifetime/numerical check remains exact original.
- Exact original parent/stride/H48/source binding checks still execute through the authenticated original pure metadata contract. Original13 live/frozen files are revalidated. No old MERGED receipt is accepted.
- relocation.json conserves every original required path and expected child. Whole521 current source inventory must match the exact final ROOT freeze; each recorded adapter/test exception is an exact unique text delta against immutable parent source. Eight exceptions cover only ROOT corrections2–7. The sole production adapter adds checked coefficient output-size arguments with the same old floors; all other exceptions are tests. A retained attention test baseline is byte-bound to the original ROOT freeze1, without a mutable MAIN dependency.
- ROOT already executed strict selected CPU/core/model and237 source-free focused tests (0fail/2ignored); exact root-cpu-focused-7.json is pinned. These do not qualify trained native replay, full HF/session/StateRoots/exp/initial GDN/GPU/performance.

## ROOT sequence

1. Review these18 files, original preparation and recorded ROOT exceptions. Run `python3 -m unittest -v test_relocation.py test_contracts.py` **from this directory**, coordinator only. Fourteen new relocation fixture methods are prepared, not executed by the author. They use tiny synthetic source files/metadata, no model/provider/operator/runtime. Whole521 fixture cardinality is a metadata inventory test.
2. Build this local Cargo manifest with the coordinator's existing controlled/offline source closure. `Cargo.toml` points only at the isolated ROOT crates, with original dependencies/features unchanged. Do not compile against MAIN or label the source MAIN-merged. A fresh local Cargo.lock, if created, is an additional mandatory actual build input. Actual compiler/flags/external-package/binary binding remains the coordinator's build evidence.
3. After actual ROOT source/build review, create an **isolated source attestation** with the fields below. Hash every final file after build, before execution. This review receipt is not an independent proof of the binary's compiler inputs.
4. Run the unchanged reference metadata preparation, native replay and independent validator shown in retained ROOT.md, **using the scripts in this directory** and the new isolated implementation receipt. Use fresh exclusive output directories. The original reference source/runtime/archive pins and full raw word/lifetime/negative limits are unchanged; raw payload authentication occurs only in the admitted native worker and independent validator, never source preparation.
5. Revalidate source freeze/receipt after execution via the unchanged runner/validator. Any source mismatch, missing/excess build path, old MAIN status, unexpected arithmetic delta or lifetime failure refuses acceptance.

Required receipt fields:

```json
{
  "status": "ROOT_REVIEWED_ISOLATED_BF16_GAMMA_GATED_RMS_SOURCE",
  "prepared_manifest_sha256": "SHA256 of this directory's prepared-manifest.json",
  "original_prepared_manifest_sha256": "f1c1f28c21616fb1057dad082a5b27a42a587bdac95ec9603d7f0c316c4eb8d4",
  "root_source_freeze_sha256": "036ce0dc73bf0d1671d521059fac0b75c6aa62bc3e8b30475a576ae8d7d7e615",
  "workspace_root": "/home/pvl/spaces/abird/src/zerogravity-worktrees/root-bf16-composition-20261020",
  "MAIN_merged": false,
  "full_model_StateRoots_exp_initial_GDN_GPU_or_performance_qualified": false,
  "files": ["exact path/bytes/sha256 rows described below"]
}
```

- `files` = all521 final ROOT freeze entries with workspace-root absolute paths, plus this directory's exact Cargo.toml/main.rs/factory.rs/source-pin.json/source-bindings.json; plus this directory's Cargo.lock if present. Exactly526 or527 unique rows, no MAIN aliases/missing/unlisted source. `relocated_source.workspace` returns the authenticated521 rows; `relocation.json.copy_map[*].relocated` supplies the local5. All expected-child assertions (including any repaired test child) must match relocation.json. The actual optional local lock must be pinned with the same2MiB BEFORE-read/fstat/regular-file policy.
- Source-only relocation preserves independent input/result owner charges, cancellation/pressure/refusal probes, retained result/source lifetime, final reserved0 and all raw exact-bit limits. No allowance to treat caller attestation as independent runtime or numerical proof.

Acceptance scope: SAME execute_gated_rms_bf16_gamma_slice; original initial11/cached353 typed finals12,288 words +retained inputs24,576 words; SAME helper fourteen stage checks61,632 words =2×(48+48+5×6144). BF16 gamma product rounding before gate is unchanged. Original25 required paths and9 expected children conserved. Diagnostic primitive parameter/window/stage owners retire on primitive() return before caller ledger/final0 checks. No initial GDN/exp/recurrence/full27B112StateRoots/session/GPU/performance eligibility.

````

## Source: .agents/runs/model-execution-hf-bf16-gamma-gated-rms-20261020/replay/isolated-root-v2/frozen-prepared/ROOT.md

Original UTF-8 bytes: 5568; SHA256 `b2ff32e5a5eaaf8c82a3d2e04b368289afaa50fd9b740615a3377c83bff97eae`.
Resolve embedded relative paths against the original path, not this volume.

````text
# ROOT-only authentic H48 gated RMS qualification

- Preparation only. Author read small source/JSON metadata, not tensor archives, learned payloads or DSO bytes; no model/provider/operator/tests/build/runtime.
- Dependency: reviewed16-path43c400a2 /c89eb6c0 candidate +test-only7f266003 /86384ec0 followup. Full HF/session/GDN/initial/exp/GPU guards stay closed.
- Original teacher11 INITIAL paddedchunk64 and teacher353 TRUECACHED original module calls; both norm inputs are actual BF16 `[48,128]`, original gamma BF16 `[128]`/256B. Exact accepted V3 source/runtime/invocation/manifest/validation/prevalidation pins in source-bindings.json. Metadata codec names are normalized only (`torch.bfloat16`→Safetensors`BF16`); original_torch_dtype retained, no numerical cast.

Scope/counts:

- Same `execute_gated_rms_bf16_gamma_slice`: 2×6144 =12,288 independent final words; 2×(core6144+gate6144)=24,576 retained input words.
- SAME new CPU helper with SAME original sealed source gamma window: 2×(mean48+inverse48+5×6144)=61,632 words /14 checks. Separate primitive receipts; no alternate fallback/model loop.
- Both primitive outputs and typed finals are independently compared to untouched actual public archive words. No tolerance or candidate-derived expectation. Original signed zeros/subnormals retained; selected nonfinite stages refuse rather than filter.
- Source authenticity does not create a live GDN/core/exp producer or certify full27B112StateRoots/session math.

Resources/lifetime:

- Native `ResourceArbiter`128MiB host pool; original factory reserves64MiB source metadata, original binding8MiB. Native8MiB control charge precedes metadata/archive/source reads and selected payload copies. This bounds private admitted payload/offset/control buffers, not OS allocator bookkeeping/page cache/free physical RAM.
- Before primitive decode/helper arrays: additional1MiB original-source Scratch. Raw gamma staged through existing charged data+offset window and compared exactly to original public raw256B. No fabricated F32gamma ref.
- Before each typed import/output: original aggregate tickets (input49,152+65,536scratch; result24,576+983,040scratch). Both live outputs retain all6 actual data arrays and original source, long after source/operator callers drop. Temporary diagnostics drop before live ledger checks.
- Native parses/authenticates each full archive via retained FD and64KiB hash buffer, bounds header before read, checks complete dtypes/shapes/spans/no trailing bytes, and reads at most24,576B per selected stage. Never copies whole47/27MiB archives. Same FD stat/hash rechecked after every selection/fullproof. Python metadata preparation opens no raw archive. Independent raw validator uses bounded same-FD streaming too.
- Exact steady80MiB; two live typed results80MiB+2,244,608B; retained72MiB+2,244,608B; actual final reserved0/reservations0/no fence. Configured pool is not physical RAM admission.
- Real pre-cancel/wronglayer and ONE-BYTE-short aggregate pressure checked for each teacher; failures restore exact live ledger. Deterministic after-import cancellation remains the prepared internal model fixture, not claimed as an external native timing test.
- Exact accepted owned_lifetime.py; owned_native.py differs only in the scope label. Same WNOWAIT/TERM/KILL-before-reap/PGID/pipe ownership,512MiB RLIMIT_AS, four actual CPU affinity IDs,900s child/2400s outer/4MiB log. No Torch/container/GPU/new engine.

ROOT sequence (never author execution):

1. Review implementation +harness +source bindings. Run8source-free fixtures (test_contracts.py) and focused/strict source tests after integration. All original freezes remain immutable.
2. Copy exact local factory.rs/source-pin.json into this Cargo manifest directory. source-pin.json is accepted7058B4c74fa28; factory is exact existing2302f6a0 copy. It scans all18 original shards through existing admitted window factory; no55GB Vec/mmap.
3. Compile this standalone Cargo target using coordinator's existing controlled build; optional rustc against freshly built MAIN libraries must preserve the same flags/full linked-source receipt. No downloads. ROOT build-source attestation must list ALL required build paths, exact new children/final repaired test and unchanged numerical sources, plus local Cargo/main/factory/source-pin/source-bindings. Status ROOT_REVIEWED_MERGED_BF16_GAMMA_GATED_RMS_SOURCE; prepared_manifest_sha256; false full_model_StateRoots_exp_initial_GDN_GPU_or_performance_qualified. Caller attestation is review binding, not independent build proof.
4. `python3 prepare_reference.py --new-output "$RUN/root-reference-metadata-1"`: metadata only. Actual raw archives are authenticated/extracted in admitted Rust work, not this phase. No independent raw PASS yet.
5. `python3 run_replay.py --binary "$BINARY" --binary-sha256 "$EXACT_SHA" --reference-directory "$RUN/root-reference-metadata-1" --implementation "$ROOT_BUILD_RECEIPT" --new-output "$RUN/root-native-1"`. Only ROOT runs, after active timing/phase gates.
6. `python3 validate_replay.py --output "$RUN/root-native-1" --reference-directory "$RUN/root-reference-metadata-1" --implementation "$ROOT_BUILD_RECEIPT"`. Independent stdlib full signed-bit checks/whole archive SHA/source/receipts/lifecycle/ledger/tree. Final native artifact tree<=8MiB, validation receipt reserved64KiB.

Acceptance is only `PASS_PRIVATE_ORIGINAL27B_H48_BF16_GAMMA_GATED_RMS_ONLY`. No full-GDN, A_log/exp, initial chunk/recurrence, canonical/root publication, trained whole model, GPU or performance eligibility.

````

## Source: .agents/runs/model-execution-hf-cached-gdn-contract-20261020/API.md

Original UTF-8 bytes: 2691; SHA256 `8de1781d94a9ceafdd786a251c6f3ebd577886690fd215ac8c66331986088943`.
Resolve embedded relative paths against the original path, not this volume.

````text
# Cached GDN private fragment — SOURCE ONLY

- SAME MathematicalBackend/PlanBuilder/HostMath/ForwardPlan and immutable sealed
  OriginalWindowOperators owner. No CPU arithmetic changes, new token/layer loop,
  model/SessionAuthority/StateRoot/checkpoint admission or production provider.
- Existing cached_step_128 does all arithmetic, including original final BF16
  RNE. Two recorded nodes express FP32 core+key-major state then BF16 return;
  HostMath returns the helper's actual three outputs. record() stays unchanged.
- Public `execute_cached_gdn_slice(layer, CachedGdnSliceInput, cancelled)` and
  `append_cached_gdn_slice(builder, layer, CachedGdnOperands<PlanValue>)`.
  External input is observed cached one-token q/k/v/beta BF16 LE plus exp(g)/
  prior FP32 LE. Full H*128 vectors, H beta/decay and H*128*128 state; H1..64,
  actual model configured heads and width128 checked. Already repeated Q/K.
- No InitialToken option, implicit zero initialization, recurrence-for-chunk,
  decay computation/producer assertion or canonical state transpose. Zero prior
  remains cached arithmetic. Output state is explicitly [head,key,value].
- Source/profile/issuer/precision/full geometry/contiguity and finite/beta/decay
  ranges checked before numerical work. Unknown region providers and GPU cold
  plans refuse before callbacks/source admission. Original HF full guards CLOSED.
- Six retained input Vecs share one admitted source-backed HostValue reservation;
  three result Vecs share another. Inputs:4*H*(16384+384+2) Output+64KiB Scratch.
  Results:4*H*(16384+256) Output+(H*(6*16384*4+16*128*4)+256KiB) Scratch.
  Admission precedes allocation. Conservative scratch covers existing helper
  temporary products/states; no opaque external allocator is called.
- Borrowed output getters only; clone shares actual backing/issuer/lease,
  into_vec remains refused, last drop releases source reservations. Pre-cancel,
  after-import cancel, result-pressure, foreign issuer, bad stride and unaware
  callback fixtures are prepared, NOT run by author.
- All live coefficient/provider work remains gated until exact public ABI math
  AND physical allocation/lifetime admission pass. An authenticated observed
  exp(g) vector in this private qualifier is not that gate or a model producer.
- Proposal baseline: current9a199 MAIN +ROOT GDN18-path overlay. Pending coefficient
  preparation is excluded; compose narrow parent hunks, preserve both contracts
  and its separate validate_record_inputs factoring. GPU_REFUSAL.patch is ONLY
  an additive parent-owned match extension; preserve row-transport/GPU work.
- Actual3 primitive receipt is recorded in AUDIT.md; no new runtime evidence.

````

## Source: .agents/runs/model-execution-hf-cached-gdn-contract-20261020/AUDIT.md

Original UTF-8 bytes: 6922; SHA256 `a279e67a9791d33ebd0d8d593d1faf96eaf5b233a5084f628ba5ba6060e27955`.
Resolve embedded relative paths against the original path, not this volume.

````text
# Shortest original-HF continuation

Parent: model-execution. Owner: baseline_review. Source-only; no new tracker,
model engine, learned cache, SessionAuthority or execution eligibility.

1. Finish the already prepared convolution+SiLU and coefficient-preparation
   independent trained gates. Keep raw in_proj_a/in_proj_b BF16, original F32
   A_log, original dt_bias dtype, BF16 beta, and F32 biased-alpha/softplus.
   SourceParameter binds original name/kind/raw shape/bytes/issuer. Never reuse
   converted GGUF -exp(A_log) or adjusted (1+norm weight) buffers.
2. Land the narrow CACHED core proposal here after ROOT review/qualification.
   It calls accepted CPU cached_step_128 through the same MathematicalBackend,
   ForwardPlan/HostMath and leased operator slice. Explicit observed BF16 q/k/v/
   beta + FP32 exp(g) + FP32 key-major prior; no coefficient/exp call. Two plan
   nodes preserve FP32 core/state and BF16 return without changing record().
   This removes a real shared-engine gap independently of coefficient work;
   it does not enable a first token or supply a missing trained producer.
3. Close exact public exp provider AND its real allocation/lifetime admission.
   Only then compose g=-exp(raw F32 A_log)*F32 softplus and exp(g) via a closed
   per-op contract. Preserve source order, original provider/libm/runtime,
   LP64 mode0x140102, read-set/alias/range/MXCSR and actual completion boundaries.
   MatVecKernel/MathematicalBackend remains the provider seam; no Python/RPC/
   model/session/token loop and no private SLEEF exp substitution.
4. Implement initial/chunk64 as a DISTINCT typed region using admitted public
   primitive ABI calls or independently source-derived exact arithmetic. Do
   not run cached recurrence with zeros. Source pads even one first token to64,
   normalizes Q/K FP32, MULTIPLIES Q by 128**-0.5, FP32 beta*K/V, cumulative decay,
   masks upper triangle to -Inf, then pairwise exp, six BMM families and two
   unit-triangular solves. Cumsum internal F64 accumulation and actual FP32
   materialization are part of the contract. Solves/new_values keep their
   source column-major strides; outputs cast to BF16 only at source return.
   Cached instead DIVIDES Q by sqrt128 and separately rounds state*decay,
   products/reductions, (V-memory)*beta, K*delta and state+update. No FMA rewrite.
5. Complete gated RMS using its accepted prefix: BF16 normalized core multiplied
   by ORIGINAL F32 gamma stays F32; multiply source SiLU(z.float()), then final
   BF16 cast. Add typed BF16 residual, FFN SiLU/multiply and raw source output
   projections. Existing analytical facade casts are not trained permissions.
6. Finish original full-attention after accepted Q/K/V+noms: exact original
   RoPE/casts, KV layout, causal QK BMM/scale, accepted F32 softmax, AV BMM,
   BF16 casts, sigmoid gate and output projection. GGUF operators and vLLM CUDA
   requests do not independently qualify pinned Torch CPU BMM/trig arithmetic.
7. Compose these nodes into the ONE existing forward_token/run_layer_sequence
   traversal. Generalize source resolution to the already sealed window owner,
   retaining full descriptors and full-column matrix rows; never fabricate a
   partial TensorSource/LoadedQwenModel or bypass its4GiB guard. Bind the existing
   SessionAuthority only after complete source-bit logits/state proof and real
   source/state/provider admission. Preserve diagnostic wire schemas0x10/0x20;
   no reinterpretation as a new trained state codec.

## Actual native primitive evidence and missing admission

Read-only metadata: `.agents/runs/model-execution-pinned-native-provider-probe-20261020/root-actual-3/`.
Status PASS_BOUNDED_NATIVE_PUBLIC_ABI_ONLY, all1,048,640 output words exact:
4 vmsExp(exp16),6 cblas_sgemm_batch,32 strsm_ calls; four actual CPUs/threads,
LP64 GNU-thread oneMKL2024.0 Update2 build20240605, CNRoff/Dyn0/FastMM1.
Actual primitive closures pin container GNU loader/libc/libm/libtorch_cpu and
OpenMP before/load/after; same Rust executable runs inside that closure.
This is NOT a host dlopen equivalence or portable vendor kernel implementation.
The receipt explicitly says native_internal_memory_admission UNQUALIFIED and
model_profile_execution REFUSED_UNCHANGED. Mapped DSO pages, constructors,
OpenMP stacks/workers, FastMM pools and hidden scratch need finite owned
admission before load/call, retention through last use and checked release.
The observed VmPeak709700KiB is observation, not a proven allocation bound;
12GiB container containment is not ResourceArbiter physical admission.

The four exp cases cover coefficient A_log16 and cached/chunk terminal g16,
NOT every initial-chunk pairwise/cumulative exp shape or -Inf masked lane.
Likewise matrix replay is clean-input primitive evidence, not the composed
chunk result/complete model. H48 requires separately observed/admitted public
batch48 plus96 triangular solves; do not extrapolate the H16 receipt.

## Layout and27B order

- Original cache: FP32 [head,key128,value128]; canonical domain:
  FP32 [head,value128,key128]. Transport U32 bits, explicit axes, no arithmetic.
- Original convolution [channel,4] -> existing canonical newest3 oldest→newest;
  initial no-cache is a separate source phase, not a guessed zero cache.
- .8B/2B: genuine independent two-teacher captures and42 canonical domains each.
-27B: original key heads16/value heads48; repeat_interleave Q/K by3 in the SAME
  engine before the core, never permutation-by-assumption. Existing full851-role
  window resolver and5120-column/10240-row three-node40,960-word ROOT proof
  are usable source/lease prerequisites, not112-domain/full-model parity.
- Generalize role resolution beyond layer0 only as each pure fragment closes.
  First complete .8B then2B full248320/all42; then27B source-real all112/fullV,
  source tokenizer, cancellation/snapshot branches and bounded streaming
  mutable state. GPU BF16/FLA remains a distinct same-queue/lease/Ready gate.

## Source anchors (current MAIN unless noted)

- `crates/zerogravity-cpu/src/original_hf/recurrent.rs:1` accepted pure cached core.
- `crates/zerogravity-model-qwen35/src/forward.rs:19` sole layer traversal;
  existing GGUF GDN body at285 must stay profile-isolated.
- `crates/zerogravity-model-qwen35/src/original_binding/raw_state.rs:1` phase/layout
  metadata only; no state capability.
- `crates/zerogravity-model-qwen35/src/original_binding/window.rs:1` sealed resolver.
- `crates/zerogravity-core/src/accelerator/program.rs:239` same backend contracts.
- Pinned `modeling_qwen3_5.py` SHA762feb6c…412c3b:225–234 gated RMS,
  294–425 chunk,429–497 cached,617–647 beta/g/repeat/phase dispatch.
- `model-execution-original-gdn-coefficient-preparation-20261020/README.md` frozen
  pending beta/alpha-only prefix; exp/g/decay explicitly excluded.

No source/expected tensor payload was hashed or numerically executed by author.

````

## Source: .agents/runs/model-execution-hf-cached-gdn-contract-20261020/ROOT-QUALIFICATION.md

Original UTF-8 bytes: 4289; SHA256 `dda2352316253324685ba8dda4c7f33d8b566bb1fe69acd58a15f50f0ba31119`.
Resolve embedded relative paths against the original path, not this volume.

````text
ROOT ONLY; author executed no tests/build/provider/operator/model/GPU runtime.

1. Review/apply NEW_DELTA.patch plus GPU_REFUSAL.patch by hunk with fresh mtimes.
   Preserve current GDN and separately pending coefficient-preparation nodes.
   No full parent file replacement. Focused core cached_gdn tests and model
   operator_slice::tests::windows::cached_gdn tests; strict default/CPU/CUDA/
   Vulkan/HF feature combinations, original release routes and full isolated
   scripts/check.sh gate before accepting implementation phase.
2. Synthetic tests are wiring/ownership only. They use the accepted CPU helper
   as math reference, preserve signed bits and all private inputs, and check
   aggregate actual Output+Scratch reservation, rejected final Vec escape,
   wrong full-attention layer, finite/dtype/span, configured heads, foreign
   issuer, valid-geometry stride, unaware callback, pre-cancel, real pool
   shortage1B before result reservation, deterministic after-import cancel,
   live cloned state and final source reservations0. No physical RAM claim.
3. Source-real .8B observed cached step1 (teacher353 after actual teacher11):
   source36-F32-restored only. Existing accepted archive in MAIN ignored
   model-execution-slow-gdn-observation-2026-10-20/captured-gdn/step-1/
   actual-gdn-frames.safetensors, SHA c8b7c72d0644c418dfa5c74e974678f7ee76e3d7cc9e83a7c157c3a9f836c0fa.
   ROOT hashes immutable source+archive and exports exactly these raw inputs:
   recurrent.input.{query,key,value,beta},
   recurrent.decay_and_prior.decay_t,
   recurrent.input.initial_state (must equal actual prior returned state).
   Keep original dtypes/strides and logical byte-copy metadata; no arithmetic.
   Match source36 restored config/index/bundle and exact original H16/width128.
   Targets are actual recurrent.fp32_core.core_attn_out FP32 H*128,
   recurrent.return.actual_return_0 BF16 H*128,
   recurrent.return.actual_return_1 FP32 H*128*128 (262144 state words).
   Existing98 fullV/cache transparency and archive validators are prerequisites.
4. Run SAME public execute_cached_gdn_slice, with no independent recurrence or
   exp call. Compare ALL signed words, exact source/profiles/layer/heads, all six
   retained inputs and all three results before/after source caller drops.
   H16 output count266240 words (2048 raw core+2048 BF16+262144 FP32 state).
   Reject output length differences before zipping. Preserve original key-major
   order; any canonical transpose is separately exact U32 transport, not math.
   This is source-real observed-input cached arithmetic, not fullproducer/state.
5. 2B original mixed-source cached GDN raw tensors are NOT available from the
   accepted generic metadata alone. ROOT must independently capture actual
   first-chunk output state and second cached input/output at teachers11/353,
   preserving all196 raw/fourfullV/all168 canonical cache checks across both
   reference loader profiles. No invented expected values, copied .8B states,
   inferred shape math or source-only receipt may fill this gap. Then repeat
   exact same H16 cached private fragment under its own SourceIdentity.
6.27B H48 geometry is covered only by bounded contract/source-free tests. No
   trained27B cached output exists in this proposal. ROOT needs independently
   source-trained actual initial+cached inputs/outputs for48 repeated value
   heads and full786432-word key-major state, source-real rawA_log/dt/matrices
   and exact Q/K repeat_interleave(3). Do not infer from the .8B/H16 receipt.
7. Both real-source attempts use original full-header verified window factory,
   actual source-before/after FD hashes, bounded owned helper/SID/PGID cleanup,
   finite host budget and fresh outputs. Retain actual per-class/peak/final0.
   Reuse accepted owned_lifetime.py/runner rather than invent a supervisor.
   Pre/mid-cancel+real capacity failure must leave old state/roots untouched
   (this private slice never owns/publishes any logical root).
8. Live GDN coefficients, external exact exp, initial chunk64, gated norm,
   attention/fullforward/session/statecodec/provider physical memory/GPU/serving
   remain CLOSED. Neither this source-only patch nor primitive ABI arithmetic
   permits external native allocation or trained full-model claims.

````

## Source: .agents/runs/model-execution-hf-cached-gdn-contract-20261020/TODO.md

Original UTF-8 bytes: 718; SHA256 `f25e1f0bb6779a63771deffffcb277f480d463a8ea6b495e7a73c158fe637103`.
Resolve embedded relative paths against the original path, not this volume.

````text
Parent: model-execution. Owner: baseline_review. SOURCE ONLY.

- [x] Read current typed engine/state/window and exact public source.
- [x] Inspect actual3 ABI receipt: primitive arithmetic PASS; native admission UNQUALIFIED.
- [x] Prepare ordered continuation + cached-only same-backend/leased slice.
- [x] Prepare7 source-free fixtures, ROOT exact archived replay scope and cold guards.
- [x] Rustfmt source parse/whitespace only; no compiler/tests/operator runtime.
- [ ] ROOT+independent peer source review/compose/focused strict and full gates.
- [ ] ROOT actual .8B/2B raw core/BF16/all-state signed-bit qualification/final0.
- [ ] H48/27B real reference, exp/provider admission, chunk64, full HF/session CLOSED.

````

## Source: .agents/runs/model-execution-hf-cached-gdn-contract-20261020/followup-h48-v3/TODO.md

Original UTF-8 bytes: 1175; SHA256 `fbc09de0f2752300b1e586270947585a86d5ddea77789b762e1c8b67117fcc46`.
Resolve embedded relative paths against the original path, not this volume.

````text
# Cached H48 source continuation

Parent: model-execution; owner spine_review; ignored source-only proposal.

- [x] Inspect actual original27B V3 public metadata; all6inputs/3targets available.
- [x] Confirm gamma is UNUSED by cached recurrence; no norm/exp guard conflation.
- [x] Freeze narrow typed codec/extent followup26d9; original57a/stride unchanged.
- [x] Prepare full-header BF16/H48 synthetic owner/cancel/realpressure negatives.
- [x] Freeze genuine archived SAMEtyped replay4ecc; acceptedfactory/report and
  original two supervisor bodies exact, complete1,603,680 words before/afterdrop.
- [x] Rustfmt proposedRust, eight Python ASTs only; no fixtures or actual runtime.
- [ ] Peer/ROOT source review, merge and actual compile/focused/strict/full gate.
- [ ] ROOT authenticate/export original archives; source payloads uninspected by author.
- [ ] ROOT actual H16 +H48 trained raw-word/ledger/lifecycle qualification.
- [ ] FullGDN/gatednorm/Al/exp/initial/session/112StateRoots/GPU remain closed.

Prep failure: first replay rustfmt rejected inner doc after added mod; corrected
order and main-only rustfmt passed, prior draft retained. No compile/model/test.

````

## Source: .agents/runs/model-execution-hf-cached-gdn-contract-20261020/followup-h48-v3/replay/ROOT.md

Original UTF-8 bytes: 3981; SHA256 `4be3506715b46adf49998cd6b3e7dfe2fd18707dd5e811077a57984701210470`.
Resolve embedded relative paths against the original path, not this volume.

````text
# ROOT-only authentic H48 cached typed replay

Author: spine_review. Original .8B931d/57a/stride proposal and captures unchanged.
No author build/tests/Torch/provider/operator/model/GPU execution or payload hash.

- Source: genuine original27B public V3 root-reference-v3-1, actual11 initial
  paddedchunk64→353 cached, original BF16 source gamma/Alog/dt. Existing independent
  validation pins all genuine plain/observed outputs and actual source/cache state.
- Export ALL804960 original inputs and798720 outputs, exact signed bits/layout:
  q/k/v/beta BF16; observed exp(g)F32; previous786432-word key-major state;
  core6144F32, return6144BF16, next786432F32. Entire actual archives are authenticated;
  unselected original chunk exp masks may contain-Inf, selectednine stages must be
  finite. No native exp, head repetition, reconstructed expected values or cache.
- Source factory.rs +source-pin.json are byte-identical to accepted27B factory
  and7058B4c74fa28 acquisition metadata. SAME VerifiedTensorSource full18-shard
  55.563GB bounded scan/header851text closure, no whole-source Vec/mmap or4GiB
  resident-guard change. All source scans happen ONLY when ROOT schedules them.
- Native uses SAME execute_cached_gdn_slice/PlanBuilder/HostMath/CPU128 reducer.
  Only separately reviewed typed followup26d9 permits actual unused metadata
  gammaBF16[128]/256B alongside existingF32[128]/512B. Gamma is never consumed by
  cached recurrence. Gatednorm/Alog/exp/initial/fullGDN/112StateRoots/session/GPU
  and complete27B inference remain closed. No separate primitive fallback.
- SAME configured128MiB host ledger, source64MiB+bindings8MiB. Qualifier control
 16MiB covers actual rawinputs3,182,880B +largest expected3,145,728B +largest
  emitted-F32Vec3,145,728B +two<=1MiB metadata buffers and bounded JSON overhead.
  This is a larger actual reservation, not a widened pool or physical-RAM claim.
  Import3,219,840B +results3,194,880B +scratch19,595,264B; live118,284,672B<128MiB.
  Real pool exhaustion/pre-cancel/wrongFA-layer refusal, retained source+allnine
  stages after caller drops, and final reservations0/nonfenced are mandatory.
- Exact old owned_lifetime.py and owned_native.py retained. Exec-FD-bound child,
  four CPUs/noGPU,512MiBAS/900s child/2400s outer/4MiBperfile logs. Original
  WNOWAIT owned TERM/grace/KILL-before-reap/independent reap/groupabsence/EOF.
  Closed process observation is not a GPUReady or recovery mechanism.

ROOT sequence after review/merge and timing holds:

1. Run five prepared stdlib test_contracts.py methods, then export_reference.py
   --new-output ABS_THIS_REPLAY/root-reference-1. Exporter calls no model/operator.
2. Review/apply original57a cacheddelta+stride03dc+H48metadata26d9 by hunk. Keep
   concurrent coefficient/pure-role nodes. Run focused/strict/full isolated gates.
3. Build this Cargo.toml with ROOT's sole offline Cargo coordinator. Preserve actual
   lock/command/log/source/binary proof. Write original required source attestation:
   ROOT_REVIEWED_MERGED_CACHED_GDN_SOURCE, prepared_manifest_sha256=this freeze,
   full_model_or_StateRoot_or_exp_or_GPU_qualified=false and complete files pins.
   It must include all required_build_paths, expected updated children and exact
   local Cargo/main/factory/source-pin/source-bindings; markers alone are not proof.
4. Run run_replay.py with original arguments --binary/--binary-sha256,
   --reference-directory ABS_THIS_REPLAY/root-reference-1, --implementation,
   --new-output ABS_THIS_REPLAY/root-native-1; then validate_replay.py with output,
   reference-directory and implementation. Compare ALL1,603,680 words before and
   after caller drops, including full786432 states, never tolerance or prefix zip.

Both original .8B/H16 and new genuine27B/H48 gates remain required independently.
No new receipt/public-hash declaration establishes native support before ROOT's
actual compile, independent trained comparison and lifecycle/ledger qualification.

````

## Source: .agents/runs/model-execution-hf-cached-gdn-contract-20261020/followup-h48-v3/replay/frozen-prepared/ROOT.md

Original UTF-8 bytes: 3981; SHA256 `4be3506715b46adf49998cd6b3e7dfe2fd18707dd5e811077a57984701210470`.
Resolve embedded relative paths against the original path, not this volume.

````text
# ROOT-only authentic H48 cached typed replay

Author: spine_review. Original .8B931d/57a/stride proposal and captures unchanged.
No author build/tests/Torch/provider/operator/model/GPU execution or payload hash.

- Source: genuine original27B public V3 root-reference-v3-1, actual11 initial
  paddedchunk64→353 cached, original BF16 source gamma/Alog/dt. Existing independent
  validation pins all genuine plain/observed outputs and actual source/cache state.
- Export ALL804960 original inputs and798720 outputs, exact signed bits/layout:
  q/k/v/beta BF16; observed exp(g)F32; previous786432-word key-major state;
  core6144F32, return6144BF16, next786432F32. Entire actual archives are authenticated;
  unselected original chunk exp masks may contain-Inf, selectednine stages must be
  finite. No native exp, head repetition, reconstructed expected values or cache.
- Source factory.rs +source-pin.json are byte-identical to accepted27B factory
  and7058B4c74fa28 acquisition metadata. SAME VerifiedTensorSource full18-shard
  55.563GB bounded scan/header851text closure, no whole-source Vec/mmap or4GiB
  resident-guard change. All source scans happen ONLY when ROOT schedules them.
- Native uses SAME execute_cached_gdn_slice/PlanBuilder/HostMath/CPU128 reducer.
  Only separately reviewed typed followup26d9 permits actual unused metadata
  gammaBF16[128]/256B alongside existingF32[128]/512B. Gamma is never consumed by
  cached recurrence. Gatednorm/Alog/exp/initial/fullGDN/112StateRoots/session/GPU
  and complete27B inference remain closed. No separate primitive fallback.
- SAME configured128MiB host ledger, source64MiB+bindings8MiB. Qualifier control
 16MiB covers actual rawinputs3,182,880B +largest expected3,145,728B +largest
  emitted-F32Vec3,145,728B +two<=1MiB metadata buffers and bounded JSON overhead.
  This is a larger actual reservation, not a widened pool or physical-RAM claim.
  Import3,219,840B +results3,194,880B +scratch19,595,264B; live118,284,672B<128MiB.
  Real pool exhaustion/pre-cancel/wrongFA-layer refusal, retained source+allnine
  stages after caller drops, and final reservations0/nonfenced are mandatory.
- Exact old owned_lifetime.py and owned_native.py retained. Exec-FD-bound child,
  four CPUs/noGPU,512MiBAS/900s child/2400s outer/4MiBperfile logs. Original
  WNOWAIT owned TERM/grace/KILL-before-reap/independent reap/groupabsence/EOF.
  Closed process observation is not a GPUReady or recovery mechanism.

ROOT sequence after review/merge and timing holds:

1. Run five prepared stdlib test_contracts.py methods, then export_reference.py
   --new-output ABS_THIS_REPLAY/root-reference-1. Exporter calls no model/operator.
2. Review/apply original57a cacheddelta+stride03dc+H48metadata26d9 by hunk. Keep
   concurrent coefficient/pure-role nodes. Run focused/strict/full isolated gates.
3. Build this Cargo.toml with ROOT's sole offline Cargo coordinator. Preserve actual
   lock/command/log/source/binary proof. Write original required source attestation:
   ROOT_REVIEWED_MERGED_CACHED_GDN_SOURCE, prepared_manifest_sha256=this freeze,
   full_model_or_StateRoot_or_exp_or_GPU_qualified=false and complete files pins.
   It must include all required_build_paths, expected updated children and exact
   local Cargo/main/factory/source-pin/source-bindings; markers alone are not proof.
4. Run run_replay.py with original arguments --binary/--binary-sha256,
   --reference-directory ABS_THIS_REPLAY/root-reference-1, --implementation,
   --new-output ABS_THIS_REPLAY/root-native-1; then validate_replay.py with output,
   reference-directory and implementation. Compare ALL1,603,680 words before and
   after caller drops, including full786432 states, never tolerance or prefix zip.

Both original .8B/H16 and new genuine27B/H48 gates remain required independently.
No new receipt/public-hash declaration establishes native support before ROOT's
actual compile, independent trained comparison and lifecycle/ledger qualification.

````

## Source: .agents/runs/model-execution-hf-cached-gdn-contract-20261020/followup-h48-v3/replay/isolated-root-v2/ISOLATED_ROOT.md

Original UTF-8 bytes: 5603; SHA256 `fd85b27d997f32712fca96566537e644e2d13f975518064f38c6479da0f0c70b`.
Resolve embedded relative paths against the original path, not this volume.

````text
# Isolated ROOT replay source relocation — coordinator only

- Additive preparation; original13 files/freeze/frozen copies remain immutable. `ROOT.md` is retained byte-identical provenance. This file supersedes only its MAIN merge/build-path instructions.
- Target `/home/pvl/spaces/abird/src/zerogravity-worktrees/root-bf16-composition-20261020`, final ROOT source freeze7 `036ce0dc73bf0d1671d521059fac0b75c6aa62bc3e8b30475a576ae8d7d7e615` (521 Rust/manifests files). MAIN remains unchanged by this preparation.
- All main.rs/factory/source-pin/source-bindings/reference exporter/raw validator/runner/owned lifetime/old fixtures are exact original bytes. Cargo changes only3 or4 absolute crate dependency roots. contracts.py changes only frozen()/implementation() source attestation dispatch; every finite/geometry/raw-word/count/source/lifetime/numerical check remains exact original.
- Exact original parent/stride/H48/source binding checks still execute through the authenticated original pure metadata contract. Original13 live/frozen files are revalidated. No old MERGED receipt is accepted.
- relocation.json conserves every original required path and expected child. Whole521 current source inventory must match the exact final ROOT freeze; each recorded adapter/test exception is an exact unique text delta against immutable parent source. Eight exceptions cover only ROOT corrections2–7. The sole production adapter adds checked coefficient output-size arguments with the same old floors; all other exceptions are tests. A retained attention test baseline is byte-bound to the original ROOT freeze1, without a mutable MAIN dependency.
- ROOT already executed strict selected CPU/core/model and237 source-free focused tests (0fail/2ignored); exact root-cpu-focused-7.json is pinned. These do not qualify trained native replay, full HF/session/StateRoots/exp/initial GDN/GPU/performance.

## ROOT sequence

1. Review these18 files, original preparation and recorded ROOT exceptions. Run `python3 -m unittest -v test_relocation.py test_contracts.py` **from this directory**, coordinator only. Fourteen new relocation fixture methods are prepared, not executed by the author. They use tiny synthetic source files/metadata, no model/provider/operator/runtime. Whole521 fixture cardinality is a metadata inventory test.
2. Build this local Cargo manifest with the coordinator's existing controlled/offline source closure. `Cargo.toml` points only at the isolated ROOT crates, with original dependencies/features unchanged. Do not compile against MAIN or label the source MAIN-merged. A fresh local Cargo.lock, if created, is an additional mandatory actual build input. Actual compiler/flags/external-package/binary binding remains the coordinator's build evidence.
3. After actual ROOT source/build review, create an **isolated source attestation** with the fields below. Hash every final file after build, before execution. This review receipt is not an independent proof of the binary's compiler inputs.
4. Run the unchanged reference metadata preparation, native replay and independent validator shown in retained ROOT.md, **using the scripts in this directory** and the new isolated implementation receipt. Use fresh exclusive output directories. The original reference source/runtime/archive pins and full raw word/lifetime/negative limits are unchanged; raw payload authentication occurs only in the admitted native worker and independent validator, never source preparation.
5. Revalidate source freeze/receipt after execution via the unchanged runner/validator. Any source mismatch, missing/excess build path, old MAIN status, unexpected arithmetic delta or lifetime failure refuses acceptance.

Required receipt fields:

```json
{
  "status": "ROOT_REVIEWED_ISOLATED_CACHED_GDN_SOURCE",
  "prepared_manifest_sha256": "SHA256 of this directory's prepared-manifest.json",
  "original_prepared_manifest_sha256": "4ecc28732a41c800dccb0e3d6a3bafe88049f0c1d09f6c39f3110940453b8118",
  "root_source_freeze_sha256": "036ce0dc73bf0d1671d521059fac0b75c6aa62bc3e8b30475a576ae8d7d7e615",
  "workspace_root": "/home/pvl/spaces/abird/src/zerogravity-worktrees/root-bf16-composition-20261020",
  "MAIN_merged": false,
  "full_model_or_StateRoot_or_exp_or_GPU_qualified": false,
  "files": ["exact path/bytes/sha256 rows described below"]
}
```

- `files` = all521 final ROOT freeze entries with workspace-root absolute paths, plus this directory's exact Cargo.toml/main.rs/factory.rs/source-pin.json/source-bindings.json; plus this directory's Cargo.lock if present. Exactly526 or527 unique rows, no MAIN aliases/missing/unlisted source. `relocated_source.workspace` returns the authenticated521 rows; `relocation.json.copy_map[*].relocated` supplies the local5. All expected-child assertions (including any repaired test child) must match relocation.json. The actual optional local lock must be pinned with the same2MiB BEFORE-read/fstat/regular-file policy.
- Source-only relocation preserves independent input/result owner charges, cancellation/pressure/refusal probes, retained result/source lifetime, final reserved0 and all raw exact-bit limits. No allowance to treat caller attestation as independent runtime or numerical proof.

Acceptance scope: SAME execute_cached_gdn_slice, authentic H48 cached teacher353 following original initial11 padded64 state;804,960 input +798,720 output =1,603,680 exact raw words. Observed external decay remains external. No initial/exp/fullGDN/gatednorm/canonical StateRoot/session/GPU eligibility. Original33 required paths and12 expected children conserved.

````

## Source: .agents/runs/model-execution-hf-cached-gdn-contract-20261020/followup-h48-v3/replay/isolated-root-v2/ROOT.md

Original UTF-8 bytes: 3981; SHA256 `4be3506715b46adf49998cd6b3e7dfe2fd18707dd5e811077a57984701210470`.
Resolve embedded relative paths against the original path, not this volume.

````text
# ROOT-only authentic H48 cached typed replay

Author: spine_review. Original .8B931d/57a/stride proposal and captures unchanged.
No author build/tests/Torch/provider/operator/model/GPU execution or payload hash.

- Source: genuine original27B public V3 root-reference-v3-1, actual11 initial
  paddedchunk64→353 cached, original BF16 source gamma/Alog/dt. Existing independent
  validation pins all genuine plain/observed outputs and actual source/cache state.
- Export ALL804960 original inputs and798720 outputs, exact signed bits/layout:
  q/k/v/beta BF16; observed exp(g)F32; previous786432-word key-major state;
  core6144F32, return6144BF16, next786432F32. Entire actual archives are authenticated;
  unselected original chunk exp masks may contain-Inf, selectednine stages must be
  finite. No native exp, head repetition, reconstructed expected values or cache.
- Source factory.rs +source-pin.json are byte-identical to accepted27B factory
  and7058B4c74fa28 acquisition metadata. SAME VerifiedTensorSource full18-shard
  55.563GB bounded scan/header851text closure, no whole-source Vec/mmap or4GiB
  resident-guard change. All source scans happen ONLY when ROOT schedules them.
- Native uses SAME execute_cached_gdn_slice/PlanBuilder/HostMath/CPU128 reducer.
  Only separately reviewed typed followup26d9 permits actual unused metadata
  gammaBF16[128]/256B alongside existingF32[128]/512B. Gamma is never consumed by
  cached recurrence. Gatednorm/Alog/exp/initial/fullGDN/112StateRoots/session/GPU
  and complete27B inference remain closed. No separate primitive fallback.
- SAME configured128MiB host ledger, source64MiB+bindings8MiB. Qualifier control
 16MiB covers actual rawinputs3,182,880B +largest expected3,145,728B +largest
  emitted-F32Vec3,145,728B +two<=1MiB metadata buffers and bounded JSON overhead.
  This is a larger actual reservation, not a widened pool or physical-RAM claim.
  Import3,219,840B +results3,194,880B +scratch19,595,264B; live118,284,672B<128MiB.
  Real pool exhaustion/pre-cancel/wrongFA-layer refusal, retained source+allnine
  stages after caller drops, and final reservations0/nonfenced are mandatory.
- Exact old owned_lifetime.py and owned_native.py retained. Exec-FD-bound child,
  four CPUs/noGPU,512MiBAS/900s child/2400s outer/4MiBperfile logs. Original
  WNOWAIT owned TERM/grace/KILL-before-reap/independent reap/groupabsence/EOF.
  Closed process observation is not a GPUReady or recovery mechanism.

ROOT sequence after review/merge and timing holds:

1. Run five prepared stdlib test_contracts.py methods, then export_reference.py
   --new-output ABS_THIS_REPLAY/root-reference-1. Exporter calls no model/operator.
2. Review/apply original57a cacheddelta+stride03dc+H48metadata26d9 by hunk. Keep
   concurrent coefficient/pure-role nodes. Run focused/strict/full isolated gates.
3. Build this Cargo.toml with ROOT's sole offline Cargo coordinator. Preserve actual
   lock/command/log/source/binary proof. Write original required source attestation:
   ROOT_REVIEWED_MERGED_CACHED_GDN_SOURCE, prepared_manifest_sha256=this freeze,
   full_model_or_StateRoot_or_exp_or_GPU_qualified=false and complete files pins.
   It must include all required_build_paths, expected updated children and exact
   local Cargo/main/factory/source-pin/source-bindings; markers alone are not proof.
4. Run run_replay.py with original arguments --binary/--binary-sha256,
   --reference-directory ABS_THIS_REPLAY/root-reference-1, --implementation,
   --new-output ABS_THIS_REPLAY/root-native-1; then validate_replay.py with output,
   reference-directory and implementation. Compare ALL1,603,680 words before and
   after caller drops, including full786432 states, never tolerance or prefix zip.

Both original .8B/H16 and new genuine27B/H48 gates remain required independently.
No new receipt/public-hash declaration establishes native support before ROOT's
actual compile, independent trained comparison and lifecycle/ledger qualification.

````

## Source: .agents/runs/model-execution-hf-cached-gdn-contract-20261020/followup-h48-v3/replay/isolated-root-v2/frozen-prepared/ISOLATED_ROOT.md

Original UTF-8 bytes: 5603; SHA256 `fd85b27d997f32712fca96566537e644e2d13f975518064f38c6479da0f0c70b`.
Resolve embedded relative paths against the original path, not this volume.

````text
# Isolated ROOT replay source relocation — coordinator only

- Additive preparation; original13 files/freeze/frozen copies remain immutable. `ROOT.md` is retained byte-identical provenance. This file supersedes only its MAIN merge/build-path instructions.
- Target `/home/pvl/spaces/abird/src/zerogravity-worktrees/root-bf16-composition-20261020`, final ROOT source freeze7 `036ce0dc73bf0d1671d521059fac0b75c6aa62bc3e8b30475a576ae8d7d7e615` (521 Rust/manifests files). MAIN remains unchanged by this preparation.
- All main.rs/factory/source-pin/source-bindings/reference exporter/raw validator/runner/owned lifetime/old fixtures are exact original bytes. Cargo changes only3 or4 absolute crate dependency roots. contracts.py changes only frozen()/implementation() source attestation dispatch; every finite/geometry/raw-word/count/source/lifetime/numerical check remains exact original.
- Exact original parent/stride/H48/source binding checks still execute through the authenticated original pure metadata contract. Original13 live/frozen files are revalidated. No old MERGED receipt is accepted.
- relocation.json conserves every original required path and expected child. Whole521 current source inventory must match the exact final ROOT freeze; each recorded adapter/test exception is an exact unique text delta against immutable parent source. Eight exceptions cover only ROOT corrections2–7. The sole production adapter adds checked coefficient output-size arguments with the same old floors; all other exceptions are tests. A retained attention test baseline is byte-bound to the original ROOT freeze1, without a mutable MAIN dependency.
- ROOT already executed strict selected CPU/core/model and237 source-free focused tests (0fail/2ignored); exact root-cpu-focused-7.json is pinned. These do not qualify trained native replay, full HF/session/StateRoots/exp/initial GDN/GPU/performance.

## ROOT sequence

1. Review these18 files, original preparation and recorded ROOT exceptions. Run `python3 -m unittest -v test_relocation.py test_contracts.py` **from this directory**, coordinator only. Fourteen new relocation fixture methods are prepared, not executed by the author. They use tiny synthetic source files/metadata, no model/provider/operator/runtime. Whole521 fixture cardinality is a metadata inventory test.
2. Build this local Cargo manifest with the coordinator's existing controlled/offline source closure. `Cargo.toml` points only at the isolated ROOT crates, with original dependencies/features unchanged. Do not compile against MAIN or label the source MAIN-merged. A fresh local Cargo.lock, if created, is an additional mandatory actual build input. Actual compiler/flags/external-package/binary binding remains the coordinator's build evidence.
3. After actual ROOT source/build review, create an **isolated source attestation** with the fields below. Hash every final file after build, before execution. This review receipt is not an independent proof of the binary's compiler inputs.
4. Run the unchanged reference metadata preparation, native replay and independent validator shown in retained ROOT.md, **using the scripts in this directory** and the new isolated implementation receipt. Use fresh exclusive output directories. The original reference source/runtime/archive pins and full raw word/lifetime/negative limits are unchanged; raw payload authentication occurs only in the admitted native worker and independent validator, never source preparation.
5. Revalidate source freeze/receipt after execution via the unchanged runner/validator. Any source mismatch, missing/excess build path, old MAIN status, unexpected arithmetic delta or lifetime failure refuses acceptance.

Required receipt fields:

```json
{
  "status": "ROOT_REVIEWED_ISOLATED_CACHED_GDN_SOURCE",
  "prepared_manifest_sha256": "SHA256 of this directory's prepared-manifest.json",
  "original_prepared_manifest_sha256": "4ecc28732a41c800dccb0e3d6a3bafe88049f0c1d09f6c39f3110940453b8118",
  "root_source_freeze_sha256": "036ce0dc73bf0d1671d521059fac0b75c6aa62bc3e8b30475a576ae8d7d7e615",
  "workspace_root": "/home/pvl/spaces/abird/src/zerogravity-worktrees/root-bf16-composition-20261020",
  "MAIN_merged": false,
  "full_model_or_StateRoot_or_exp_or_GPU_qualified": false,
  "files": ["exact path/bytes/sha256 rows described below"]
}
```

- `files` = all521 final ROOT freeze entries with workspace-root absolute paths, plus this directory's exact Cargo.toml/main.rs/factory.rs/source-pin.json/source-bindings.json; plus this directory's Cargo.lock if present. Exactly526 or527 unique rows, no MAIN aliases/missing/unlisted source. `relocated_source.workspace` returns the authenticated521 rows; `relocation.json.copy_map[*].relocated` supplies the local5. All expected-child assertions (including any repaired test child) must match relocation.json. The actual optional local lock must be pinned with the same2MiB BEFORE-read/fstat/regular-file policy.
- Source-only relocation preserves independent input/result owner charges, cancellation/pressure/refusal probes, retained result/source lifetime, final reserved0 and all raw exact-bit limits. No allowance to treat caller attestation as independent runtime or numerical proof.

Acceptance scope: SAME execute_cached_gdn_slice, authentic H48 cached teacher353 following original initial11 padded64 state;804,960 input +798,720 output =1,603,680 exact raw words. Observed external decay remains external. No initial/exp/fullGDN/gatednorm/canonical StateRoot/session/GPU eligibility. Original33 required paths and12 expected children conserved.

````

## Source: .agents/runs/model-execution-hf-cached-gdn-contract-20261020/followup-h48-v3/replay/isolated-root-v2/frozen-prepared/ROOT.md

Original UTF-8 bytes: 3981; SHA256 `4be3506715b46adf49998cd6b3e7dfe2fd18707dd5e811077a57984701210470`.
Resolve embedded relative paths against the original path, not this volume.

````text
# ROOT-only authentic H48 cached typed replay

Author: spine_review. Original .8B931d/57a/stride proposal and captures unchanged.
No author build/tests/Torch/provider/operator/model/GPU execution or payload hash.

- Source: genuine original27B public V3 root-reference-v3-1, actual11 initial
  paddedchunk64→353 cached, original BF16 source gamma/Alog/dt. Existing independent
  validation pins all genuine plain/observed outputs and actual source/cache state.
- Export ALL804960 original inputs and798720 outputs, exact signed bits/layout:
  q/k/v/beta BF16; observed exp(g)F32; previous786432-word key-major state;
  core6144F32, return6144BF16, next786432F32. Entire actual archives are authenticated;
  unselected original chunk exp masks may contain-Inf, selectednine stages must be
  finite. No native exp, head repetition, reconstructed expected values or cache.
- Source factory.rs +source-pin.json are byte-identical to accepted27B factory
  and7058B4c74fa28 acquisition metadata. SAME VerifiedTensorSource full18-shard
  55.563GB bounded scan/header851text closure, no whole-source Vec/mmap or4GiB
  resident-guard change. All source scans happen ONLY when ROOT schedules them.
- Native uses SAME execute_cached_gdn_slice/PlanBuilder/HostMath/CPU128 reducer.
  Only separately reviewed typed followup26d9 permits actual unused metadata
  gammaBF16[128]/256B alongside existingF32[128]/512B. Gamma is never consumed by
  cached recurrence. Gatednorm/Alog/exp/initial/fullGDN/112StateRoots/session/GPU
  and complete27B inference remain closed. No separate primitive fallback.
- SAME configured128MiB host ledger, source64MiB+bindings8MiB. Qualifier control
 16MiB covers actual rawinputs3,182,880B +largest expected3,145,728B +largest
  emitted-F32Vec3,145,728B +two<=1MiB metadata buffers and bounded JSON overhead.
  This is a larger actual reservation, not a widened pool or physical-RAM claim.
  Import3,219,840B +results3,194,880B +scratch19,595,264B; live118,284,672B<128MiB.
  Real pool exhaustion/pre-cancel/wrongFA-layer refusal, retained source+allnine
  stages after caller drops, and final reservations0/nonfenced are mandatory.
- Exact old owned_lifetime.py and owned_native.py retained. Exec-FD-bound child,
  four CPUs/noGPU,512MiBAS/900s child/2400s outer/4MiBperfile logs. Original
  WNOWAIT owned TERM/grace/KILL-before-reap/independent reap/groupabsence/EOF.
  Closed process observation is not a GPUReady or recovery mechanism.

ROOT sequence after review/merge and timing holds:

1. Run five prepared stdlib test_contracts.py methods, then export_reference.py
   --new-output ABS_THIS_REPLAY/root-reference-1. Exporter calls no model/operator.
2. Review/apply original57a cacheddelta+stride03dc+H48metadata26d9 by hunk. Keep
   concurrent coefficient/pure-role nodes. Run focused/strict/full isolated gates.
3. Build this Cargo.toml with ROOT's sole offline Cargo coordinator. Preserve actual
   lock/command/log/source/binary proof. Write original required source attestation:
   ROOT_REVIEWED_MERGED_CACHED_GDN_SOURCE, prepared_manifest_sha256=this freeze,
   full_model_or_StateRoot_or_exp_or_GPU_qualified=false and complete files pins.
   It must include all required_build_paths, expected updated children and exact
   local Cargo/main/factory/source-pin/source-bindings; markers alone are not proof.
4. Run run_replay.py with original arguments --binary/--binary-sha256,
   --reference-directory ABS_THIS_REPLAY/root-reference-1, --implementation,
   --new-output ABS_THIS_REPLAY/root-native-1; then validate_replay.py with output,
   reference-directory and implementation. Compare ALL1,603,680 words before and
   after caller drops, including full786432 states, never tolerance or prefix zip.

Both original .8B/H16 and new genuine27B/H48 gates remain required independently.
No new receipt/public-hash declaration establishes native support before ROOT's
actual compile, independent trained comparison and lifecycle/ledger qualification.

````

## Source: .agents/runs/model-execution-hf-cached-gdn-contract-20261020/followup-h48-v3/replay/pre-review-draft-1/frozen-prepared/ROOT.md

Original UTF-8 bytes: 3981; SHA256 `4be3506715b46adf49998cd6b3e7dfe2fd18707dd5e811077a57984701210470`.
Resolve embedded relative paths against the original path, not this volume.

````text
# ROOT-only authentic H48 cached typed replay

Author: spine_review. Original .8B931d/57a/stride proposal and captures unchanged.
No author build/tests/Torch/provider/operator/model/GPU execution or payload hash.

- Source: genuine original27B public V3 root-reference-v3-1, actual11 initial
  paddedchunk64→353 cached, original BF16 source gamma/Alog/dt. Existing independent
  validation pins all genuine plain/observed outputs and actual source/cache state.
- Export ALL804960 original inputs and798720 outputs, exact signed bits/layout:
  q/k/v/beta BF16; observed exp(g)F32; previous786432-word key-major state;
  core6144F32, return6144BF16, next786432F32. Entire actual archives are authenticated;
  unselected original chunk exp masks may contain-Inf, selectednine stages must be
  finite. No native exp, head repetition, reconstructed expected values or cache.
- Source factory.rs +source-pin.json are byte-identical to accepted27B factory
  and7058B4c74fa28 acquisition metadata. SAME VerifiedTensorSource full18-shard
  55.563GB bounded scan/header851text closure, no whole-source Vec/mmap or4GiB
  resident-guard change. All source scans happen ONLY when ROOT schedules them.
- Native uses SAME execute_cached_gdn_slice/PlanBuilder/HostMath/CPU128 reducer.
  Only separately reviewed typed followup26d9 permits actual unused metadata
  gammaBF16[128]/256B alongside existingF32[128]/512B. Gamma is never consumed by
  cached recurrence. Gatednorm/Alog/exp/initial/fullGDN/112StateRoots/session/GPU
  and complete27B inference remain closed. No separate primitive fallback.
- SAME configured128MiB host ledger, source64MiB+bindings8MiB. Qualifier control
 16MiB covers actual rawinputs3,182,880B +largest expected3,145,728B +largest
  emitted-F32Vec3,145,728B +two<=1MiB metadata buffers and bounded JSON overhead.
  This is a larger actual reservation, not a widened pool or physical-RAM claim.
  Import3,219,840B +results3,194,880B +scratch19,595,264B; live118,284,672B<128MiB.
  Real pool exhaustion/pre-cancel/wrongFA-layer refusal, retained source+allnine
  stages after caller drops, and final reservations0/nonfenced are mandatory.
- Exact old owned_lifetime.py and owned_native.py retained. Exec-FD-bound child,
  four CPUs/noGPU,512MiBAS/900s child/2400s outer/4MiBperfile logs. Original
  WNOWAIT owned TERM/grace/KILL-before-reap/independent reap/groupabsence/EOF.
  Closed process observation is not a GPUReady or recovery mechanism.

ROOT sequence after review/merge and timing holds:

1. Run five prepared stdlib test_contracts.py methods, then export_reference.py
   --new-output ABS_THIS_REPLAY/root-reference-1. Exporter calls no model/operator.
2. Review/apply original57a cacheddelta+stride03dc+H48metadata26d9 by hunk. Keep
   concurrent coefficient/pure-role nodes. Run focused/strict/full isolated gates.
3. Build this Cargo.toml with ROOT's sole offline Cargo coordinator. Preserve actual
   lock/command/log/source/binary proof. Write original required source attestation:
   ROOT_REVIEWED_MERGED_CACHED_GDN_SOURCE, prepared_manifest_sha256=this freeze,
   full_model_or_StateRoot_or_exp_or_GPU_qualified=false and complete files pins.
   It must include all required_build_paths, expected updated children and exact
   local Cargo/main/factory/source-pin/source-bindings; markers alone are not proof.
4. Run run_replay.py with original arguments --binary/--binary-sha256,
   --reference-directory ABS_THIS_REPLAY/root-reference-1, --implementation,
   --new-output ABS_THIS_REPLAY/root-native-1; then validate_replay.py with output,
   reference-directory and implementation. Compare ALL1,603,680 words before and
   after caller drops, including full786432 states, never tolerance or prefix zip.

Both original .8B/H16 and new genuine27B/H48 gates remain required independently.
No new receipt/public-hash declaration establishes native support before ROOT's
actual compile, independent trained comparison and lifecycle/ledger qualification.

````

## Source: .agents/runs/model-execution-hf-cached-gdn-contract-20261020/followup-v2/TODO.md

Original UTF-8 bytes: 1685; SHA256 `70dd059eae0927aacf9858c6c7ebd7e100e52cccda4ec61c3fe6d4379e6dcad5`.
Resolve embedded relative paths against the original path, not this volume.

````text
Parent plan: `.agents/plans/model-execution/`. Owner: spine_review.

- [x] Source-review original57a4 freeze;20 live/frozen,7 baseline,12 source pins exact.
- [x] Add ONLY stride fixture rows256→2 and query→decay; original bytes preserved.
- [x] Inspect actual `.8B` cached/initial archive headers and payload pins; complete
  H16 inputs/targets available and real first-return→cached-prior link retained.
- [x] Inspect actual2B generic archives and27B public reference; missing cached
  operands explicitly unavailable. Check27B source header BF16 norm guard.
- [x] Prepare archived-reference exporter, original native shared-engine helper,
  source/build attestation, owned launcher, independent byte/ledger validator.
- [x] Freeze13 replay files931d3442 plus stride-only71e5e656; AST7/rustfmt1 and
  static byte checks only. Three source-free fixtures prepared, NOT executed.
- [ ] ROOT independent review/merge, focused/strict tests and full repository gate.
- [ ] ROOT actual archive export, attested native build/run and independent full
 266240-output-word/268320-input-word proof, source/lease/cancel/pressure/final0.
- [ ] Independent actual2B/H48 cached captures and source-kind compatibility
  before any corresponding trained qualifier. No replacement/synthetic proof.

No author Cargo/tests/provider/model/operator/GPU/container execution, MAIN,
shared-plan edits or commits. First local diff-generation assertion counted
the +++ header; inspected/corrected generation before freeze, original untouched.
Internal preparation drafts preserved; no draft or final qualifier executed.
Live exp/initial chunk/StateRoot/full-HF/27B/GPU/serving qualification stays closed.

````

## Source: .agents/runs/model-execution-hf-cached-gdn-contract-20261020/followup-v2/replay/ROOT.md

Original UTF-8 bytes: 5055; SHA256 `77956d3ac5faf3932813aa5614e4ab0a0debd490b0f4796ee27b5ad0980d01bb`.
Resolve embedded relative paths against the original path, not this volume.

````text
# ROOT-only archived cached GDN gate

Author: spine_review. Parent: model-execution.

- Source-only preparation. No author build, tests, operator/provider/model,
  container, GPU, MAIN or shared-plan changes. Preserve original57a4e64 freeze.
- Parent `STRIDE_FOLLOWUP.patch` changes ONLY rows256→2 and bad.query→bad.decay.
  Original F32 backing, same issuer and valid two-element decay shape now reach
  the intended contiguous guard. No production change in that followup.
- `.8B` actual public step1 archive provides all six cached inputs and all three
  results. Whole archive SHA, selected byte SHA, dtype/shape/stride, previous
  real chunk return and98 historical fullV/cache checks are mandatory.
- This is a replay against an already captured authentic public call. Exporter
  reads actual original archive bytes; it calls no Torch/operator/model and
  derives no expected value. It does not claim a new public runtime execution.
- Imported exp(g) is the actual archived decay. No native exp, live coefficient
  producer, initial-token substitute, state publication, canonical transpose,
  source mutation, model/session admission or new engine is introduced.
- Native helper opens and verifies the genuine complete `.8B` shard through
  existing SourceWeights/OriginalWindowOperators and calls ONLY the public
  execute_cached_gdn_slice. All266240 output words are compared to independent
  archive bits, plus268320 retained input words. Signed zero/subnormals survive.
- Existing owned_lifetime.py copied byte-exact; owned_native.py retains original
  source_files/execute bodies, changing only scope wording and removing unused
  coefficient main. Exec-FD SHA, exact four CPUs, noGPU environment,512MiB AS,
  finite900s child/2400s outer deadline,4MiB combined logs, WNOWAIT/group-bound
  TERM/grace/two-KILL-before-any-reap/independent reap/EOF remain.
- Host ledger128MiB:64MiB sealed source +two4MiB bindings +8MiB qualifier control.
  Aggregate six-input and three-result payloads and conservative helper scratch
  use original same-source host reservations. Real pool pressure/pre-cancel/
  wrong-layer refusal, exact live/retained accounting and last-drop0 required.
  Configured ledger and AS limits are not a whole-host physical-RAM claim.
- ROOT must independently review/merge both prepared source patches, record the
  complete actual build/source list and attest the resulting native binary.
  `--implementation` is an explicit ROOT review/build attestation, not proof
  obtained merely by providing hashes. All changed paths are required; exact
  new child and unchanged original arithmetic pins are checked independently.
  Parent Program/HostMath merges must preserve concurrent coefficient nodes.

## Availability limits

- 2B actual paired generic archive has final recurrent state and QKV/conv, but
  lacks cached q/k/v/beta/decay/core-call inputs/targets. Do not reconstruct them.
- 27B actual public reference contains embedding/InputNorm/QKV only. No authentic
  H48 cached operands/results exist in the inspected archives; historical27B
  runs do not supply them. Current source-level cached_heads also requires F32
  GdnNorm while original27B has BF16 gamma. Thus H48 dimension compatibility
  does not make the current original27B slice executable. No dtype gate changed.
- `availability.json` records actual inspected archive headers/reference scope.
  No synthetic, repeated `.8B`, pre-convolution QKV or guessed state substitutes.
  Future2B/H48 need actual original public initial+cached capture with immutable
  source/runtime pins and independently transparent full-model evidence first.

## ROOT sequence

Run only after current benchmark/GPU holds and source integration permit it.
Set `RUN` to this `replay/` directory. All output directories are fresh.

```bash
python3 "$RUN/test_contracts.py"
python3 "$RUN/export_reference.py" --new-output "$RUN/root-reference-1"
CARGO_BUILD_JOBS=2 RAYON_NUM_THREADS=4 python3 scripts/native-gpu.py -- \
  cargo build --release --offline --manifest-path "$RUN/Cargo.toml"
python3 "$RUN/run_replay.py" --binary "$ROOT_ATTESTED_BINARY" \
  --binary-sha256 "$ROOT_BINARY_SHA" \
  --reference-directory "$RUN/root-reference-1" \
  --implementation "$ROOT_SOURCE_ATTESTATION" --new-output "$RUN/root-native-1"
python3 "$RUN/validate_replay.py" --output "$RUN/root-native-1" \
  --reference-directory "$RUN/root-reference-1" --implementation "$ROOT_SOURCE_ATTESTATION"
```

ROOT source attestation JSON: `status=ROOT_REVIEWED_MERGED_CACHED_GDN_SOURCE`,
`prepared_manifest_sha256=<this prepared-manifest.json SHA>`,
`full_model_or_StateRoot_or_exp_or_GPU_qualified=false`, and `files` containing
the complete reviewed actual build-source rows `{path,bytes,sha256}`. Required
paths and exact child/dependency pins live in prepared-manifest.json. Preserve
actual Cargo lock/artifact/build/source receipts independently.

Actual acceptance still requires focused/strict feature tests, original source
routes and full isolated repository gate. No new acceptance is claimed here.

````

## Source: .agents/runs/model-execution-hf-cached-gdn-contract-20261020/followup-v2/replay/frozen-prepared/ROOT.md

Original UTF-8 bytes: 5055; SHA256 `77956d3ac5faf3932813aa5614e4ab0a0debd490b0f4796ee27b5ad0980d01bb`.
Resolve embedded relative paths against the original path, not this volume.

````text
# ROOT-only archived cached GDN gate

Author: spine_review. Parent: model-execution.

- Source-only preparation. No author build, tests, operator/provider/model,
  container, GPU, MAIN or shared-plan changes. Preserve original57a4e64 freeze.
- Parent `STRIDE_FOLLOWUP.patch` changes ONLY rows256→2 and bad.query→bad.decay.
  Original F32 backing, same issuer and valid two-element decay shape now reach
  the intended contiguous guard. No production change in that followup.
- `.8B` actual public step1 archive provides all six cached inputs and all three
  results. Whole archive SHA, selected byte SHA, dtype/shape/stride, previous
  real chunk return and98 historical fullV/cache checks are mandatory.
- This is a replay against an already captured authentic public call. Exporter
  reads actual original archive bytes; it calls no Torch/operator/model and
  derives no expected value. It does not claim a new public runtime execution.
- Imported exp(g) is the actual archived decay. No native exp, live coefficient
  producer, initial-token substitute, state publication, canonical transpose,
  source mutation, model/session admission or new engine is introduced.
- Native helper opens and verifies the genuine complete `.8B` shard through
  existing SourceWeights/OriginalWindowOperators and calls ONLY the public
  execute_cached_gdn_slice. All266240 output words are compared to independent
  archive bits, plus268320 retained input words. Signed zero/subnormals survive.
- Existing owned_lifetime.py copied byte-exact; owned_native.py retains original
  source_files/execute bodies, changing only scope wording and removing unused
  coefficient main. Exec-FD SHA, exact four CPUs, noGPU environment,512MiB AS,
  finite900s child/2400s outer deadline,4MiB combined logs, WNOWAIT/group-bound
  TERM/grace/two-KILL-before-any-reap/independent reap/EOF remain.
- Host ledger128MiB:64MiB sealed source +two4MiB bindings +8MiB qualifier control.
  Aggregate six-input and three-result payloads and conservative helper scratch
  use original same-source host reservations. Real pool pressure/pre-cancel/
  wrong-layer refusal, exact live/retained accounting and last-drop0 required.
  Configured ledger and AS limits are not a whole-host physical-RAM claim.
- ROOT must independently review/merge both prepared source patches, record the
  complete actual build/source list and attest the resulting native binary.
  `--implementation` is an explicit ROOT review/build attestation, not proof
  obtained merely by providing hashes. All changed paths are required; exact
  new child and unchanged original arithmetic pins are checked independently.
  Parent Program/HostMath merges must preserve concurrent coefficient nodes.

## Availability limits

- 2B actual paired generic archive has final recurrent state and QKV/conv, but
  lacks cached q/k/v/beta/decay/core-call inputs/targets. Do not reconstruct them.
- 27B actual public reference contains embedding/InputNorm/QKV only. No authentic
  H48 cached operands/results exist in the inspected archives; historical27B
  runs do not supply them. Current source-level cached_heads also requires F32
  GdnNorm while original27B has BF16 gamma. Thus H48 dimension compatibility
  does not make the current original27B slice executable. No dtype gate changed.
- `availability.json` records actual inspected archive headers/reference scope.
  No synthetic, repeated `.8B`, pre-convolution QKV or guessed state substitutes.
  Future2B/H48 need actual original public initial+cached capture with immutable
  source/runtime pins and independently transparent full-model evidence first.

## ROOT sequence

Run only after current benchmark/GPU holds and source integration permit it.
Set `RUN` to this `replay/` directory. All output directories are fresh.

```bash
python3 "$RUN/test_contracts.py"
python3 "$RUN/export_reference.py" --new-output "$RUN/root-reference-1"
CARGO_BUILD_JOBS=2 RAYON_NUM_THREADS=4 python3 scripts/native-gpu.py -- \
  cargo build --release --offline --manifest-path "$RUN/Cargo.toml"
python3 "$RUN/run_replay.py" --binary "$ROOT_ATTESTED_BINARY" \
  --binary-sha256 "$ROOT_BINARY_SHA" \
  --reference-directory "$RUN/root-reference-1" \
  --implementation "$ROOT_SOURCE_ATTESTATION" --new-output "$RUN/root-native-1"
python3 "$RUN/validate_replay.py" --output "$RUN/root-native-1" \
  --reference-directory "$RUN/root-reference-1" --implementation "$ROOT_SOURCE_ATTESTATION"
```

ROOT source attestation JSON: `status=ROOT_REVIEWED_MERGED_CACHED_GDN_SOURCE`,
`prepared_manifest_sha256=<this prepared-manifest.json SHA>`,
`full_model_or_StateRoot_or_exp_or_GPU_qualified=false`, and `files` containing
the complete reviewed actual build-source rows `{path,bytes,sha256}`. Required
paths and exact child/dependency pins live in prepared-manifest.json. Preserve
actual Cargo lock/artifact/build/source receipts independently.

Actual acceptance still requires focused/strict feature tests, original source
routes and full isolated repository gate. No new acceptance is claimed here.

````

## Source: .agents/runs/model-execution-hf-cached-gdn-contract-20261020/followup-v2/replay-internal-draft-1/ROOT.md

Original UTF-8 bytes: 5055; SHA256 `77956d3ac5faf3932813aa5614e4ab0a0debd490b0f4796ee27b5ad0980d01bb`.
Resolve embedded relative paths against the original path, not this volume.

````text
# ROOT-only archived cached GDN gate

Author: spine_review. Parent: model-execution.

- Source-only preparation. No author build, tests, operator/provider/model,
  container, GPU, MAIN or shared-plan changes. Preserve original57a4e64 freeze.
- Parent `STRIDE_FOLLOWUP.patch` changes ONLY rows256→2 and bad.query→bad.decay.
  Original F32 backing, same issuer and valid two-element decay shape now reach
  the intended contiguous guard. No production change in that followup.
- `.8B` actual public step1 archive provides all six cached inputs and all three
  results. Whole archive SHA, selected byte SHA, dtype/shape/stride, previous
  real chunk return and98 historical fullV/cache checks are mandatory.
- This is a replay against an already captured authentic public call. Exporter
  reads actual original archive bytes; it calls no Torch/operator/model and
  derives no expected value. It does not claim a new public runtime execution.
- Imported exp(g) is the actual archived decay. No native exp, live coefficient
  producer, initial-token substitute, state publication, canonical transpose,
  source mutation, model/session admission or new engine is introduced.
- Native helper opens and verifies the genuine complete `.8B` shard through
  existing SourceWeights/OriginalWindowOperators and calls ONLY the public
  execute_cached_gdn_slice. All266240 output words are compared to independent
  archive bits, plus268320 retained input words. Signed zero/subnormals survive.
- Existing owned_lifetime.py copied byte-exact; owned_native.py retains original
  source_files/execute bodies, changing only scope wording and removing unused
  coefficient main. Exec-FD SHA, exact four CPUs, noGPU environment,512MiB AS,
  finite900s child/2400s outer deadline,4MiB combined logs, WNOWAIT/group-bound
  TERM/grace/two-KILL-before-any-reap/independent reap/EOF remain.
- Host ledger128MiB:64MiB sealed source +two4MiB bindings +8MiB qualifier control.
  Aggregate six-input and three-result payloads and conservative helper scratch
  use original same-source host reservations. Real pool pressure/pre-cancel/
  wrong-layer refusal, exact live/retained accounting and last-drop0 required.
  Configured ledger and AS limits are not a whole-host physical-RAM claim.
- ROOT must independently review/merge both prepared source patches, record the
  complete actual build/source list and attest the resulting native binary.
  `--implementation` is an explicit ROOT review/build attestation, not proof
  obtained merely by providing hashes. All changed paths are required; exact
  new child and unchanged original arithmetic pins are checked independently.
  Parent Program/HostMath merges must preserve concurrent coefficient nodes.

## Availability limits

- 2B actual paired generic archive has final recurrent state and QKV/conv, but
  lacks cached q/k/v/beta/decay/core-call inputs/targets. Do not reconstruct them.
- 27B actual public reference contains embedding/InputNorm/QKV only. No authentic
  H48 cached operands/results exist in the inspected archives; historical27B
  runs do not supply them. Current source-level cached_heads also requires F32
  GdnNorm while original27B has BF16 gamma. Thus H48 dimension compatibility
  does not make the current original27B slice executable. No dtype gate changed.
- `availability.json` records actual inspected archive headers/reference scope.
  No synthetic, repeated `.8B`, pre-convolution QKV or guessed state substitutes.
  Future2B/H48 need actual original public initial+cached capture with immutable
  source/runtime pins and independently transparent full-model evidence first.

## ROOT sequence

Run only after current benchmark/GPU holds and source integration permit it.
Set `RUN` to this `replay/` directory. All output directories are fresh.

```bash
python3 "$RUN/test_contracts.py"
python3 "$RUN/export_reference.py" --new-output "$RUN/root-reference-1"
CARGO_BUILD_JOBS=2 RAYON_NUM_THREADS=4 python3 scripts/native-gpu.py -- \
  cargo build --release --offline --manifest-path "$RUN/Cargo.toml"
python3 "$RUN/run_replay.py" --binary "$ROOT_ATTESTED_BINARY" \
  --binary-sha256 "$ROOT_BINARY_SHA" \
  --reference-directory "$RUN/root-reference-1" \
  --implementation "$ROOT_SOURCE_ATTESTATION" --new-output "$RUN/root-native-1"
python3 "$RUN/validate_replay.py" --output "$RUN/root-native-1" \
  --reference-directory "$RUN/root-reference-1" --implementation "$ROOT_SOURCE_ATTESTATION"
```

ROOT source attestation JSON: `status=ROOT_REVIEWED_MERGED_CACHED_GDN_SOURCE`,
`prepared_manifest_sha256=<this prepared-manifest.json SHA>`,
`full_model_or_StateRoot_or_exp_or_GPU_qualified=false`, and `files` containing
the complete reviewed actual build-source rows `{path,bytes,sha256}`. Required
paths and exact child/dependency pins live in prepared-manifest.json. Preserve
actual Cargo lock/artifact/build/source receipts independently.

Actual acceptance still requires focused/strict feature tests, original source
routes and full isolated repository gate. No new acceptance is claimed here.

````

## Source: .agents/runs/model-execution-hf-cached-gdn-contract-20261020/followup-v2/replay-internal-draft-2/ROOT.md

Original UTF-8 bytes: 5055; SHA256 `77956d3ac5faf3932813aa5614e4ab0a0debd490b0f4796ee27b5ad0980d01bb`.
Resolve embedded relative paths against the original path, not this volume.

````text
# ROOT-only archived cached GDN gate

Author: spine_review. Parent: model-execution.

- Source-only preparation. No author build, tests, operator/provider/model,
  container, GPU, MAIN or shared-plan changes. Preserve original57a4e64 freeze.
- Parent `STRIDE_FOLLOWUP.patch` changes ONLY rows256→2 and bad.query→bad.decay.
  Original F32 backing, same issuer and valid two-element decay shape now reach
  the intended contiguous guard. No production change in that followup.
- `.8B` actual public step1 archive provides all six cached inputs and all three
  results. Whole archive SHA, selected byte SHA, dtype/shape/stride, previous
  real chunk return and98 historical fullV/cache checks are mandatory.
- This is a replay against an already captured authentic public call. Exporter
  reads actual original archive bytes; it calls no Torch/operator/model and
  derives no expected value. It does not claim a new public runtime execution.
- Imported exp(g) is the actual archived decay. No native exp, live coefficient
  producer, initial-token substitute, state publication, canonical transpose,
  source mutation, model/session admission or new engine is introduced.
- Native helper opens and verifies the genuine complete `.8B` shard through
  existing SourceWeights/OriginalWindowOperators and calls ONLY the public
  execute_cached_gdn_slice. All266240 output words are compared to independent
  archive bits, plus268320 retained input words. Signed zero/subnormals survive.
- Existing owned_lifetime.py copied byte-exact; owned_native.py retains original
  source_files/execute bodies, changing only scope wording and removing unused
  coefficient main. Exec-FD SHA, exact four CPUs, noGPU environment,512MiB AS,
  finite900s child/2400s outer deadline,4MiB combined logs, WNOWAIT/group-bound
  TERM/grace/two-KILL-before-any-reap/independent reap/EOF remain.
- Host ledger128MiB:64MiB sealed source +two4MiB bindings +8MiB qualifier control.
  Aggregate six-input and three-result payloads and conservative helper scratch
  use original same-source host reservations. Real pool pressure/pre-cancel/
  wrong-layer refusal, exact live/retained accounting and last-drop0 required.
  Configured ledger and AS limits are not a whole-host physical-RAM claim.
- ROOT must independently review/merge both prepared source patches, record the
  complete actual build/source list and attest the resulting native binary.
  `--implementation` is an explicit ROOT review/build attestation, not proof
  obtained merely by providing hashes. All changed paths are required; exact
  new child and unchanged original arithmetic pins are checked independently.
  Parent Program/HostMath merges must preserve concurrent coefficient nodes.

## Availability limits

- 2B actual paired generic archive has final recurrent state and QKV/conv, but
  lacks cached q/k/v/beta/decay/core-call inputs/targets. Do not reconstruct them.
- 27B actual public reference contains embedding/InputNorm/QKV only. No authentic
  H48 cached operands/results exist in the inspected archives; historical27B
  runs do not supply them. Current source-level cached_heads also requires F32
  GdnNorm while original27B has BF16 gamma. Thus H48 dimension compatibility
  does not make the current original27B slice executable. No dtype gate changed.
- `availability.json` records actual inspected archive headers/reference scope.
  No synthetic, repeated `.8B`, pre-convolution QKV or guessed state substitutes.
  Future2B/H48 need actual original public initial+cached capture with immutable
  source/runtime pins and independently transparent full-model evidence first.

## ROOT sequence

Run only after current benchmark/GPU holds and source integration permit it.
Set `RUN` to this `replay/` directory. All output directories are fresh.

```bash
python3 "$RUN/test_contracts.py"
python3 "$RUN/export_reference.py" --new-output "$RUN/root-reference-1"
CARGO_BUILD_JOBS=2 RAYON_NUM_THREADS=4 python3 scripts/native-gpu.py -- \
  cargo build --release --offline --manifest-path "$RUN/Cargo.toml"
python3 "$RUN/run_replay.py" --binary "$ROOT_ATTESTED_BINARY" \
  --binary-sha256 "$ROOT_BINARY_SHA" \
  --reference-directory "$RUN/root-reference-1" \
  --implementation "$ROOT_SOURCE_ATTESTATION" --new-output "$RUN/root-native-1"
python3 "$RUN/validate_replay.py" --output "$RUN/root-native-1" \
  --reference-directory "$RUN/root-reference-1" --implementation "$ROOT_SOURCE_ATTESTATION"
```

ROOT source attestation JSON: `status=ROOT_REVIEWED_MERGED_CACHED_GDN_SOURCE`,
`prepared_manifest_sha256=<this prepared-manifest.json SHA>`,
`full_model_or_StateRoot_or_exp_or_GPU_qualified=false`, and `files` containing
the complete reviewed actual build-source rows `{path,bytes,sha256}`. Required
paths and exact child/dependency pins live in prepared-manifest.json. Preserve
actual Cargo lock/artifact/build/source receipts independently.

Actual acceptance still requires focused/strict feature tests, original source
routes and full isolated repository gate. No new acceptance is claimed here.

````

## Source: .agents/runs/model-execution-hf-cached-gdn-contract-20261020/frozen-prepared/API.md

Original UTF-8 bytes: 2691; SHA256 `8de1781d94a9ceafdd786a251c6f3ebd577886690fd215ac8c66331986088943`.
Resolve embedded relative paths against the original path, not this volume.

````text
# Cached GDN private fragment — SOURCE ONLY

- SAME MathematicalBackend/PlanBuilder/HostMath/ForwardPlan and immutable sealed
  OriginalWindowOperators owner. No CPU arithmetic changes, new token/layer loop,
  model/SessionAuthority/StateRoot/checkpoint admission or production provider.
- Existing cached_step_128 does all arithmetic, including original final BF16
  RNE. Two recorded nodes express FP32 core+key-major state then BF16 return;
  HostMath returns the helper's actual three outputs. record() stays unchanged.
- Public `execute_cached_gdn_slice(layer, CachedGdnSliceInput, cancelled)` and
  `append_cached_gdn_slice(builder, layer, CachedGdnOperands<PlanValue>)`.
  External input is observed cached one-token q/k/v/beta BF16 LE plus exp(g)/
  prior FP32 LE. Full H*128 vectors, H beta/decay and H*128*128 state; H1..64,
  actual model configured heads and width128 checked. Already repeated Q/K.
- No InitialToken option, implicit zero initialization, recurrence-for-chunk,
  decay computation/producer assertion or canonical state transpose. Zero prior
  remains cached arithmetic. Output state is explicitly [head,key,value].
- Source/profile/issuer/precision/full geometry/contiguity and finite/beta/decay
  ranges checked before numerical work. Unknown region providers and GPU cold
  plans refuse before callbacks/source admission. Original HF full guards CLOSED.
- Six retained input Vecs share one admitted source-backed HostValue reservation;
  three result Vecs share another. Inputs:4*H*(16384+384+2) Output+64KiB Scratch.
  Results:4*H*(16384+256) Output+(H*(6*16384*4+16*128*4)+256KiB) Scratch.
  Admission precedes allocation. Conservative scratch covers existing helper
  temporary products/states; no opaque external allocator is called.
- Borrowed output getters only; clone shares actual backing/issuer/lease,
  into_vec remains refused, last drop releases source reservations. Pre-cancel,
  after-import cancel, result-pressure, foreign issuer, bad stride and unaware
  callback fixtures are prepared, NOT run by author.
- All live coefficient/provider work remains gated until exact public ABI math
  AND physical allocation/lifetime admission pass. An authenticated observed
  exp(g) vector in this private qualifier is not that gate or a model producer.
- Proposal baseline: current9a199 MAIN +ROOT GDN18-path overlay. Pending coefficient
  preparation is excluded; compose narrow parent hunks, preserve both contracts
  and its separate validate_record_inputs factoring. GPU_REFUSAL.patch is ONLY
  an additive parent-owned match extension; preserve row-transport/GPU work.
- Actual3 primitive receipt is recorded in AUDIT.md; no new runtime evidence.

````

## Source: .agents/runs/model-execution-hf-cached-gdn-contract-20261020/frozen-prepared/AUDIT.md

Original UTF-8 bytes: 6922; SHA256 `a279e67a9791d33ebd0d8d593d1faf96eaf5b233a5084f628ba5ba6060e27955`.
Resolve embedded relative paths against the original path, not this volume.

````text
# Shortest original-HF continuation

Parent: model-execution. Owner: baseline_review. Source-only; no new tracker,
model engine, learned cache, SessionAuthority or execution eligibility.

1. Finish the already prepared convolution+SiLU and coefficient-preparation
   independent trained gates. Keep raw in_proj_a/in_proj_b BF16, original F32
   A_log, original dt_bias dtype, BF16 beta, and F32 biased-alpha/softplus.
   SourceParameter binds original name/kind/raw shape/bytes/issuer. Never reuse
   converted GGUF -exp(A_log) or adjusted (1+norm weight) buffers.
2. Land the narrow CACHED core proposal here after ROOT review/qualification.
   It calls accepted CPU cached_step_128 through the same MathematicalBackend,
   ForwardPlan/HostMath and leased operator slice. Explicit observed BF16 q/k/v/
   beta + FP32 exp(g) + FP32 key-major prior; no coefficient/exp call. Two plan
   nodes preserve FP32 core/state and BF16 return without changing record().
   This removes a real shared-engine gap independently of coefficient work;
   it does not enable a first token or supply a missing trained producer.
3. Close exact public exp provider AND its real allocation/lifetime admission.
   Only then compose g=-exp(raw F32 A_log)*F32 softplus and exp(g) via a closed
   per-op contract. Preserve source order, original provider/libm/runtime,
   LP64 mode0x140102, read-set/alias/range/MXCSR and actual completion boundaries.
   MatVecKernel/MathematicalBackend remains the provider seam; no Python/RPC/
   model/session/token loop and no private SLEEF exp substitution.
4. Implement initial/chunk64 as a DISTINCT typed region using admitted public
   primitive ABI calls or independently source-derived exact arithmetic. Do
   not run cached recurrence with zeros. Source pads even one first token to64,
   normalizes Q/K FP32, MULTIPLIES Q by 128**-0.5, FP32 beta*K/V, cumulative decay,
   masks upper triangle to -Inf, then pairwise exp, six BMM families and two
   unit-triangular solves. Cumsum internal F64 accumulation and actual FP32
   materialization are part of the contract. Solves/new_values keep their
   source column-major strides; outputs cast to BF16 only at source return.
   Cached instead DIVIDES Q by sqrt128 and separately rounds state*decay,
   products/reductions, (V-memory)*beta, K*delta and state+update. No FMA rewrite.
5. Complete gated RMS using its accepted prefix: BF16 normalized core multiplied
   by ORIGINAL F32 gamma stays F32; multiply source SiLU(z.float()), then final
   BF16 cast. Add typed BF16 residual, FFN SiLU/multiply and raw source output
   projections. Existing analytical facade casts are not trained permissions.
6. Finish original full-attention after accepted Q/K/V+noms: exact original
   RoPE/casts, KV layout, causal QK BMM/scale, accepted F32 softmax, AV BMM,
   BF16 casts, sigmoid gate and output projection. GGUF operators and vLLM CUDA
   requests do not independently qualify pinned Torch CPU BMM/trig arithmetic.
7. Compose these nodes into the ONE existing forward_token/run_layer_sequence
   traversal. Generalize source resolution to the already sealed window owner,
   retaining full descriptors and full-column matrix rows; never fabricate a
   partial TensorSource/LoadedQwenModel or bypass its4GiB guard. Bind the existing
   SessionAuthority only after complete source-bit logits/state proof and real
   source/state/provider admission. Preserve diagnostic wire schemas0x10/0x20;
   no reinterpretation as a new trained state codec.

## Actual native primitive evidence and missing admission

Read-only metadata: `.agents/runs/model-execution-pinned-native-provider-probe-20261020/root-actual-3/`.
Status PASS_BOUNDED_NATIVE_PUBLIC_ABI_ONLY, all1,048,640 output words exact:
4 vmsExp(exp16),6 cblas_sgemm_batch,32 strsm_ calls; four actual CPUs/threads,
LP64 GNU-thread oneMKL2024.0 Update2 build20240605, CNRoff/Dyn0/FastMM1.
Actual primitive closures pin container GNU loader/libc/libm/libtorch_cpu and
OpenMP before/load/after; same Rust executable runs inside that closure.
This is NOT a host dlopen equivalence or portable vendor kernel implementation.
The receipt explicitly says native_internal_memory_admission UNQUALIFIED and
model_profile_execution REFUSED_UNCHANGED. Mapped DSO pages, constructors,
OpenMP stacks/workers, FastMM pools and hidden scratch need finite owned
admission before load/call, retention through last use and checked release.
The observed VmPeak709700KiB is observation, not a proven allocation bound;
12GiB container containment is not ResourceArbiter physical admission.

The four exp cases cover coefficient A_log16 and cached/chunk terminal g16,
NOT every initial-chunk pairwise/cumulative exp shape or -Inf masked lane.
Likewise matrix replay is clean-input primitive evidence, not the composed
chunk result/complete model. H48 requires separately observed/admitted public
batch48 plus96 triangular solves; do not extrapolate the H16 receipt.

## Layout and27B order

- Original cache: FP32 [head,key128,value128]; canonical domain:
  FP32 [head,value128,key128]. Transport U32 bits, explicit axes, no arithmetic.
- Original convolution [channel,4] -> existing canonical newest3 oldest→newest;
  initial no-cache is a separate source phase, not a guessed zero cache.
- .8B/2B: genuine independent two-teacher captures and42 canonical domains each.
-27B: original key heads16/value heads48; repeat_interleave Q/K by3 in the SAME
  engine before the core, never permutation-by-assumption. Existing full851-role
  window resolver and5120-column/10240-row three-node40,960-word ROOT proof
  are usable source/lease prerequisites, not112-domain/full-model parity.
- Generalize role resolution beyond layer0 only as each pure fragment closes.
  First complete .8B then2B full248320/all42; then27B source-real all112/fullV,
  source tokenizer, cancellation/snapshot branches and bounded streaming
  mutable state. GPU BF16/FLA remains a distinct same-queue/lease/Ready gate.

## Source anchors (current MAIN unless noted)

- `crates/zerogravity-cpu/src/original_hf/recurrent.rs:1` accepted pure cached core.
- `crates/zerogravity-model-qwen35/src/forward.rs:19` sole layer traversal;
  existing GGUF GDN body at285 must stay profile-isolated.
- `crates/zerogravity-model-qwen35/src/original_binding/raw_state.rs:1` phase/layout
  metadata only; no state capability.
- `crates/zerogravity-model-qwen35/src/original_binding/window.rs:1` sealed resolver.
- `crates/zerogravity-core/src/accelerator/program.rs:239` same backend contracts.
- Pinned `modeling_qwen3_5.py` SHA762feb6c…412c3b:225–234 gated RMS,
  294–425 chunk,429–497 cached,617–647 beta/g/repeat/phase dispatch.
- `model-execution-original-gdn-coefficient-preparation-20261020/README.md` frozen
  pending beta/alpha-only prefix; exp/g/decay explicitly excluded.

No source/expected tensor payload was hashed or numerically executed by author.

````

## Source: .agents/runs/model-execution-hf-cached-gdn-contract-20261020/frozen-prepared/ROOT-QUALIFICATION.md

Original UTF-8 bytes: 4289; SHA256 `dda2352316253324685ba8dda4c7f33d8b566bb1fe69acd58a15f50f0ba31119`.
Resolve embedded relative paths against the original path, not this volume.

````text
ROOT ONLY; author executed no tests/build/provider/operator/model/GPU runtime.

1. Review/apply NEW_DELTA.patch plus GPU_REFUSAL.patch by hunk with fresh mtimes.
   Preserve current GDN and separately pending coefficient-preparation nodes.
   No full parent file replacement. Focused core cached_gdn tests and model
   operator_slice::tests::windows::cached_gdn tests; strict default/CPU/CUDA/
   Vulkan/HF feature combinations, original release routes and full isolated
   scripts/check.sh gate before accepting implementation phase.
2. Synthetic tests are wiring/ownership only. They use the accepted CPU helper
   as math reference, preserve signed bits and all private inputs, and check
   aggregate actual Output+Scratch reservation, rejected final Vec escape,
   wrong full-attention layer, finite/dtype/span, configured heads, foreign
   issuer, valid-geometry stride, unaware callback, pre-cancel, real pool
   shortage1B before result reservation, deterministic after-import cancel,
   live cloned state and final source reservations0. No physical RAM claim.
3. Source-real .8B observed cached step1 (teacher353 after actual teacher11):
   source36-F32-restored only. Existing accepted archive in MAIN ignored
   model-execution-slow-gdn-observation-2026-10-20/captured-gdn/step-1/
   actual-gdn-frames.safetensors, SHA c8b7c72d0644c418dfa5c74e974678f7ee76e3d7cc9e83a7c157c3a9f836c0fa.
   ROOT hashes immutable source+archive and exports exactly these raw inputs:
   recurrent.input.{query,key,value,beta},
   recurrent.decay_and_prior.decay_t,
   recurrent.input.initial_state (must equal actual prior returned state).
   Keep original dtypes/strides and logical byte-copy metadata; no arithmetic.
   Match source36 restored config/index/bundle and exact original H16/width128.
   Targets are actual recurrent.fp32_core.core_attn_out FP32 H*128,
   recurrent.return.actual_return_0 BF16 H*128,
   recurrent.return.actual_return_1 FP32 H*128*128 (262144 state words).
   Existing98 fullV/cache transparency and archive validators are prerequisites.
4. Run SAME public execute_cached_gdn_slice, with no independent recurrence or
   exp call. Compare ALL signed words, exact source/profiles/layer/heads, all six
   retained inputs and all three results before/after source caller drops.
   H16 output count266240 words (2048 raw core+2048 BF16+262144 FP32 state).
   Reject output length differences before zipping. Preserve original key-major
   order; any canonical transpose is separately exact U32 transport, not math.
   This is source-real observed-input cached arithmetic, not fullproducer/state.
5. 2B original mixed-source cached GDN raw tensors are NOT available from the
   accepted generic metadata alone. ROOT must independently capture actual
   first-chunk output state and second cached input/output at teachers11/353,
   preserving all196 raw/fourfullV/all168 canonical cache checks across both
   reference loader profiles. No invented expected values, copied .8B states,
   inferred shape math or source-only receipt may fill this gap. Then repeat
   exact same H16 cached private fragment under its own SourceIdentity.
6.27B H48 geometry is covered only by bounded contract/source-free tests. No
   trained27B cached output exists in this proposal. ROOT needs independently
   source-trained actual initial+cached inputs/outputs for48 repeated value
   heads and full786432-word key-major state, source-real rawA_log/dt/matrices
   and exact Q/K repeat_interleave(3). Do not infer from the .8B/H16 receipt.
7. Both real-source attempts use original full-header verified window factory,
   actual source-before/after FD hashes, bounded owned helper/SID/PGID cleanup,
   finite host budget and fresh outputs. Retain actual per-class/peak/final0.
   Reuse accepted owned_lifetime.py/runner rather than invent a supervisor.
   Pre/mid-cancel+real capacity failure must leave old state/roots untouched
   (this private slice never owns/publishes any logical root).
8. Live GDN coefficients, external exact exp, initial chunk64, gated norm,
   attention/fullforward/session/statecodec/provider physical memory/GPU/serving
   remain CLOSED. Neither this source-only patch nor primitive ABI arithmetic
   permits external native allocation or trained full-model claims.

````

## Source: .agents/runs/model-execution-hf-cached-gdn-contract-20261020/frozen-prepared/TODO.md

Original UTF-8 bytes: 718; SHA256 `f25e1f0bb6779a63771deffffcb277f480d463a8ea6b495e7a73c158fe637103`.
Resolve embedded relative paths against the original path, not this volume.

````text
Parent: model-execution. Owner: baseline_review. SOURCE ONLY.

- [x] Read current typed engine/state/window and exact public source.
- [x] Inspect actual3 ABI receipt: primitive arithmetic PASS; native admission UNQUALIFIED.
- [x] Prepare ordered continuation + cached-only same-backend/leased slice.
- [x] Prepare7 source-free fixtures, ROOT exact archived replay scope and cold guards.
- [x] Rustfmt source parse/whitespace only; no compiler/tests/operator runtime.
- [ ] ROOT+independent peer source review/compose/focused strict and full gates.
- [ ] ROOT actual .8B/2B raw core/BF16/all-state signed-bit qualification/final0.
- [ ] H48/27B real reference, exp/provider admission, chunk64, full HF/session CLOSED.

````

## Source: .agents/runs/model-execution-hf-combined525-attestation-20261020/READINESS.md

Original UTF-8 bytes: 3067; SHA256 `51f1a4d7f57d30f78380fca4aae300ca67e924c0604ab71cdedc6d17bacebe48`.
Resolve embedded relative paths against the original path, not this volume.

````text
# Source-only readiness review

- Initial3cda106a is the author's fresh SELF-AUDIT, not an independent review.
  Exact pinned primary cumsum uses CPU float→double accumulation and stores each
  sequential prefix as F32. Proposed helper keeps the carry F64, query MUL uses
  fixed128^-0.5 F32, and Cached/DIV is refused. Explicit x86_64/AVX2/MXCSR boundary,
  finite complete spans, source-bound admission before buffers, retained original
  issuer and actual last-drop/cancel/pressure fixtures remain leaf-only.
- H48 original05095281 and FA3 e4be9998 reviewed independently. The only learned
  codec expansion is UNEVALUATED A_log rank1 BF16[48]/96B under the same role,
  source and profile. dt's existing BF16→F32 path and source617/619 beta/alpha/
  softplus schedule remain unchanged. Both real teacher imports retain the
  source-owned typed ForwardPlan/HostMath route and bounded backing.
- FA3 derivative fixes the real27B negative: config layers0/1/2 are GDN,3 is FA.
  It checks that actual schedule before calling the negative. Original13 replay
  support files are exact; original main/attestor/files remain frozen. No second
  source/state/model owner or expected arithmetic is introduced.
- Selected actual ROOT521 source seams: CPU original_hf/module/Cargo, original
  TensorWindow/WindowWorkspace and model/helper coefficient paths match036ce0.
  Full521/525 scans and actual compile/source/runtime checks remain ROOT work.

## Concrete packaging blocker and repair

Original initial524 and H48522 exact-source attestors both name the same isolated
tree; they reject the correct joint525 composition. This derivative is the repair,
not a relaxation: original521 +seven disjoint path deltas +retained BSD license,
fresh helper directories/source attestations, mandatory actual compiler receipts
and always-emitted per-lane compiled identity. Old binaries cannot satisfy that
identity merely by rewriting their sidecar receipts.

No remaining concrete arithmetic/API/source-owner blocker found by this source
review. The two original CPU proposals and the combined derivative are uncompiled
and unexecuted by this author. Root's existing accepted CPU evidence does not
automatically cover these new deltas.

## Next qualification cut

1. Independent second-author initial-leaf/combined-binder review.
2. ROOT composes exact525+license with fresh mtimes, then focused tests/strict/full
   checks. Generate each standalone lock offline against unchanged parent tuples.
3. New source receipts, actual distinct builds and ROOT compiler/binary receipts.
4. Serialized genuine archived reference preparation/native child/raw validators.
   Initial9216 output+9216 input words and H48288 output+192 input words, real
   pressure/cancel/retained source/backing/final0, complete owned group+EOF.

No initial64 BMM/TRSM solver, exp/g/decay provider admission, actual model producer,
complete GDN, full112-domain StateRoot/logits, original27B model/GPU/performance
eligibility follows from these leaves. Shared full-HF/session/checkpoint guards
remain closed.

````

## Source: .agents/runs/model-execution-hf-combined525-attestation-20261020/README.md

Original UTF-8 bytes: 6238; SHA256 `d45beef1b6ce67724bef7656f4f4153d0353f171f5aaf1c69ac00f0a3480bf68`.
Resolve embedded relative paths against the original path, not this volume.

````text
# Combined ROOT525 source-only derivative

- Parent: exact immutable ROOT036ce0 /521 Rust+manifest inputs.
- Union: initial3cda four Rust paths + H48 coefficient0509 three Rust paths;
  three existing paths change and four new paths give525. Initial BSD notice is
  copied separately. All seven deltas and prior bytes are in COMPOSITION.json.
- H48 replay starts from wrong-layer e4be9998: actual FA3 negative, not layer1.
- No mathematical source changes, source retagging, MAIN or isolated-tree writes
  by the preparer. Original initial/H48/FA3 manifests, files and binaries remain
  untouched. Seven selected parent seams matched036ce0; no full521 scan by author.
- Review attribution: initial leaf is the author's fresh SELF-AUDIT. H48 metadata,
  typed/helper/owner/replay paths received independent source review. A second
  author still reviews the initial leaf; this derivative does not replace that.

## Why fresh entrypoints

The original initial attestor admits524 inputs and the original H48 attestor522.
Neither accepts the combined525 tree. These fresh entrypoints validate exactly
the union, retain every parent freeze, and require a new per-lane source receipt
and actual ROOT build receipt. Each main.rs adds only an always-emitted compiled
build-identity.json stamp. Run and independent validation require that exact stamp;
an old successful binary/report cannot acquire the new label by retagging receipts.
The stamp proves harness identity, not mathematical correctness or engine support.

Compiler receipts are explicit ROOT attestations of actual recorded execution.
This preparer does not execute a compiler or infer successful compilation from
metadata. A build receipt and its source/binary inputs are bound before the child;
actual signed raw output validation remains separate.

## ROOT composition and build

1. Review COMPOSITION.json, seven proposed Rust files, BSD notice and REVIEW.diff.
   Apply only proposed/ bytes with fresh mtimes to the named isolated tree. Never
   replace MAIN or overwrite another overlay. The exact parent source closure,
   prior metadata for all modified paths and original supplier bytes are checked.
   There must be exactly525 Rust/manifest inputs after composition. Extra changes
   require another explicit derivative; do not delete work to satisfy this one.
2. Run ROOT's relevant focused CPU/cache/noncache and strict/full gates. Both
   full original-HF/session/checkpoint and external-exp/provider guards stay closed.
3. For EACH of initial/replay and h48/replay, generate its standalone Cargo.lock
   offline. Every dependency tuple must belong to the unchanged parent lock.
   New source receipts include all525, BSD notice, six actual compiler inputs
   (Cargo.toml, main.rs, factory.rs, source-pin, source-bindings, build-identity)
   and the generated standalone lock:533 total file rows.
4. From that exact replay directory, write a fresh source attestation:

```sh
python3 write_implementation.py --new-output root-source-1.json
```

5. ROOT runs and records the real direct build command, with its actual stdout,
   stderr/exit, compiler environment and effective lockfile in separate evidence:

```sh
CARGO_BUILD_JOBS=2 RAYON_NUM_THREADS=4 cargo build --release --offline --locked \
  --manifest-path /ABS/THIS/FRESH/REPLAY/Cargo.toml
```

6. Retain the actual command result; write a fresh ROOT receipt matching the
   following schema. `source_before_sha256` and `source_after_sha256` are the
   SAME fresh source receipt, independently revalidated before and after build.
   `binary` is the actual canonical binary path, length and SHA. No old binary
   or old compiler receipt is acceptable. The receipt is ROOT's execution
   attestation; the next command does not itself run or verify a compiler.

```json
{
  "schema": "zerogravity.combined525.root-build-receipt.v1",
  "status": "ROOT_BUILD_COMPLETED",
  "lane": "initial",
  "cwd": "/ABS/THIS/FRESH/REPLAY",
  "argv": ["cargo", "build", "--release", "--offline", "--locked", "--manifest-path", "/ABS/THIS/FRESH/REPLAY/Cargo.toml"],
  "exit_code": 0,
  "source_before_sha256": "ACTUAL_FRESH_SOURCE_RECEIPT_SHA",
  "source_after_sha256": "ACTUAL_SAME_SOURCE_RECEIPT_SHA",
  "preparation_sha256": "ACTUAL_NEW_PREPARED_MANIFEST_SHA",
  "binary": {"path": "/ABS/ACTUAL/BINARY", "bytes": 1, "sha256": "ACTUAL_BINARY_SHA"}
}
```

Use lane `h48` in that replay. The two actual compiled binaries/stamps differ.
Only after successful ROOT execution:

```sh
python3 attest_build.py --root-build-receipt root-build-1.json \
  --implementation root-source-1.json --binary /ABS/ACTUAL/BINARY \
  --new-output root-build-attestation-1.json
```

## ROOT bounded replay (each lane, serialized)

```sh
python3 prepare_reference.py --new-output root-reference-1
python3 run_replay.py --binary /ABS/ACTUAL/BINARY \
  --binary-sha256 ACTUAL_BINARY_SHA --implementation root-source-1.json \
  --build-attestation root-build-attestation-1.json \
  --reference-directory root-reference-1 --new-output root-native-1
python3 validate_replay.py --output root-native-1 \
  --reference-directory root-reference-1 --implementation root-source-1.json \
  --build-attestation root-build-attestation-1.json
```

All outputs/source receipts stay in their own fresh replay directory. Source and
archive whole-file scans happen only under ROOT's serialized qualification slot,
never during timing. Original WNOWAIT/SID/PGID/TERM/INT-safe owned helper, caps,
finite raw transport and independent validators are retained; no second owner.

- Initial leaf scope: source-bound authentic teacher11 query6144 +prefix3072
  outputs and9216 retained input words, actual74240B payload/scratch, cancel,
  foreign/Cached refusal, real one-byte-short pressure and final0.
- H48 scope: authentic teachers11/353 preparation outputs288 +inputs192, raw
  BF16 dt consumed, raw A_log metadata retained UNEVALUATED, actual FA3 refusal,
  aggregate workspace/retention/final0. No exp, g or decay production.
- These do not establish initial chunk64 solver/BMM, complete layer/token engine,
  full112 domains/logits, trained 27B execution, external provider memory admission,
  CUDA or performance eligibility. All source-only fixtures remain unexecuted.

````

## Source: .agents/runs/model-execution-hf-combined525-attestation-20261020/TODO.md

Original UTF-8 bytes: 792; SHA256 `f02f892007b59ca93d1a49532c2ddf076d992b7863044b6e0cce69029db7189f`.
Resolve embedded relative paths against the original path, not this volume.

````text
Parent: model-execution. Owned: ignored combined525 attestation derivative only.

- [x] Initial leaf fresh self-audit; H48 independent source readiness review.
- [x] Identify exact524/522 attestor conflict; prepare exact525 disjoint union.
- [x] Fresh per-lane source/compiler/binary/stamp/run/validator bindings.
- [x] Preserve all original freezes, failures and source/ownership/math guards.
- [ ] Independent source review of this derivative and initial leaf.
- [ ] ROOT actual composition/locks/focused strict/full gate and two fresh builds.
- [ ] ROOT serialized genuine raw replay +independent validation, actual final0.

No builds/tests/imported scripts/operators/providers/models/GPU/payload scans by
author. MAIN/isolated worktree/shared plans unchanged; no numerical qualification.

````

## Source: .agents/runs/model-execution-hf-combined525-attestation-20261020/frozen-prepared/READINESS.md

Original UTF-8 bytes: 3067; SHA256 `51f1a4d7f57d30f78380fca4aae300ca67e924c0604ab71cdedc6d17bacebe48`.
Resolve embedded relative paths against the original path, not this volume.

````text
# Source-only readiness review

- Initial3cda106a is the author's fresh SELF-AUDIT, not an independent review.
  Exact pinned primary cumsum uses CPU float→double accumulation and stores each
  sequential prefix as F32. Proposed helper keeps the carry F64, query MUL uses
  fixed128^-0.5 F32, and Cached/DIV is refused. Explicit x86_64/AVX2/MXCSR boundary,
  finite complete spans, source-bound admission before buffers, retained original
  issuer and actual last-drop/cancel/pressure fixtures remain leaf-only.
- H48 original05095281 and FA3 e4be9998 reviewed independently. The only learned
  codec expansion is UNEVALUATED A_log rank1 BF16[48]/96B under the same role,
  source and profile. dt's existing BF16→F32 path and source617/619 beta/alpha/
  softplus schedule remain unchanged. Both real teacher imports retain the
  source-owned typed ForwardPlan/HostMath route and bounded backing.
- FA3 derivative fixes the real27B negative: config layers0/1/2 are GDN,3 is FA.
  It checks that actual schedule before calling the negative. Original13 replay
  support files are exact; original main/attestor/files remain frozen. No second
  source/state/model owner or expected arithmetic is introduced.
- Selected actual ROOT521 source seams: CPU original_hf/module/Cargo, original
  TensorWindow/WindowWorkspace and model/helper coefficient paths match036ce0.
  Full521/525 scans and actual compile/source/runtime checks remain ROOT work.

## Concrete packaging blocker and repair

Original initial524 and H48522 exact-source attestors both name the same isolated
tree; they reject the correct joint525 composition. This derivative is the repair,
not a relaxation: original521 +seven disjoint path deltas +retained BSD license,
fresh helper directories/source attestations, mandatory actual compiler receipts
and always-emitted per-lane compiled identity. Old binaries cannot satisfy that
identity merely by rewriting their sidecar receipts.

No remaining concrete arithmetic/API/source-owner blocker found by this source
review. The two original CPU proposals and the combined derivative are uncompiled
and unexecuted by this author. Root's existing accepted CPU evidence does not
automatically cover these new deltas.

## Next qualification cut

1. Independent second-author initial-leaf/combined-binder review.
2. ROOT composes exact525+license with fresh mtimes, then focused tests/strict/full
   checks. Generate each standalone lock offline against unchanged parent tuples.
3. New source receipts, actual distinct builds and ROOT compiler/binary receipts.
4. Serialized genuine archived reference preparation/native child/raw validators.
   Initial9216 output+9216 input words and H48288 output+192 input words, real
   pressure/cancel/retained source/backing/final0, complete owned group+EOF.

No initial64 BMM/TRSM solver, exp/g/decay provider admission, actual model producer,
complete GDN, full112-domain StateRoot/logits, original27B model/GPU/performance
eligibility follows from these leaves. Shared full-HF/session/checkpoint guards
remain closed.

````

## Source: .agents/runs/model-execution-hf-combined525-attestation-20261020/frozen-prepared/README.md

Original UTF-8 bytes: 6238; SHA256 `d45beef1b6ce67724bef7656f4f4153d0353f171f5aaf1c69ac00f0a3480bf68`.
Resolve embedded relative paths against the original path, not this volume.

````text
# Combined ROOT525 source-only derivative

- Parent: exact immutable ROOT036ce0 /521 Rust+manifest inputs.
- Union: initial3cda four Rust paths + H48 coefficient0509 three Rust paths;
  three existing paths change and four new paths give525. Initial BSD notice is
  copied separately. All seven deltas and prior bytes are in COMPOSITION.json.
- H48 replay starts from wrong-layer e4be9998: actual FA3 negative, not layer1.
- No mathematical source changes, source retagging, MAIN or isolated-tree writes
  by the preparer. Original initial/H48/FA3 manifests, files and binaries remain
  untouched. Seven selected parent seams matched036ce0; no full521 scan by author.
- Review attribution: initial leaf is the author's fresh SELF-AUDIT. H48 metadata,
  typed/helper/owner/replay paths received independent source review. A second
  author still reviews the initial leaf; this derivative does not replace that.

## Why fresh entrypoints

The original initial attestor admits524 inputs and the original H48 attestor522.
Neither accepts the combined525 tree. These fresh entrypoints validate exactly
the union, retain every parent freeze, and require a new per-lane source receipt
and actual ROOT build receipt. Each main.rs adds only an always-emitted compiled
build-identity.json stamp. Run and independent validation require that exact stamp;
an old successful binary/report cannot acquire the new label by retagging receipts.
The stamp proves harness identity, not mathematical correctness or engine support.

Compiler receipts are explicit ROOT attestations of actual recorded execution.
This preparer does not execute a compiler or infer successful compilation from
metadata. A build receipt and its source/binary inputs are bound before the child;
actual signed raw output validation remains separate.

## ROOT composition and build

1. Review COMPOSITION.json, seven proposed Rust files, BSD notice and REVIEW.diff.
   Apply only proposed/ bytes with fresh mtimes to the named isolated tree. Never
   replace MAIN or overwrite another overlay. The exact parent source closure,
   prior metadata for all modified paths and original supplier bytes are checked.
   There must be exactly525 Rust/manifest inputs after composition. Extra changes
   require another explicit derivative; do not delete work to satisfy this one.
2. Run ROOT's relevant focused CPU/cache/noncache and strict/full gates. Both
   full original-HF/session/checkpoint and external-exp/provider guards stay closed.
3. For EACH of initial/replay and h48/replay, generate its standalone Cargo.lock
   offline. Every dependency tuple must belong to the unchanged parent lock.
   New source receipts include all525, BSD notice, six actual compiler inputs
   (Cargo.toml, main.rs, factory.rs, source-pin, source-bindings, build-identity)
   and the generated standalone lock:533 total file rows.
4. From that exact replay directory, write a fresh source attestation:

```sh
python3 write_implementation.py --new-output root-source-1.json
```

5. ROOT runs and records the real direct build command, with its actual stdout,
   stderr/exit, compiler environment and effective lockfile in separate evidence:

```sh
CARGO_BUILD_JOBS=2 RAYON_NUM_THREADS=4 cargo build --release --offline --locked \
  --manifest-path /ABS/THIS/FRESH/REPLAY/Cargo.toml
```

6. Retain the actual command result; write a fresh ROOT receipt matching the
   following schema. `source_before_sha256` and `source_after_sha256` are the
   SAME fresh source receipt, independently revalidated before and after build.
   `binary` is the actual canonical binary path, length and SHA. No old binary
   or old compiler receipt is acceptable. The receipt is ROOT's execution
   attestation; the next command does not itself run or verify a compiler.

```json
{
  "schema": "zerogravity.combined525.root-build-receipt.v1",
  "status": "ROOT_BUILD_COMPLETED",
  "lane": "initial",
  "cwd": "/ABS/THIS/FRESH/REPLAY",
  "argv": ["cargo", "build", "--release", "--offline", "--locked", "--manifest-path", "/ABS/THIS/FRESH/REPLAY/Cargo.toml"],
  "exit_code": 0,
  "source_before_sha256": "ACTUAL_FRESH_SOURCE_RECEIPT_SHA",
  "source_after_sha256": "ACTUAL_SAME_SOURCE_RECEIPT_SHA",
  "preparation_sha256": "ACTUAL_NEW_PREPARED_MANIFEST_SHA",
  "binary": {"path": "/ABS/ACTUAL/BINARY", "bytes": 1, "sha256": "ACTUAL_BINARY_SHA"}
}
```

Use lane `h48` in that replay. The two actual compiled binaries/stamps differ.
Only after successful ROOT execution:

```sh
python3 attest_build.py --root-build-receipt root-build-1.json \
  --implementation root-source-1.json --binary /ABS/ACTUAL/BINARY \
  --new-output root-build-attestation-1.json
```

## ROOT bounded replay (each lane, serialized)

```sh
python3 prepare_reference.py --new-output root-reference-1
python3 run_replay.py --binary /ABS/ACTUAL/BINARY \
  --binary-sha256 ACTUAL_BINARY_SHA --implementation root-source-1.json \
  --build-attestation root-build-attestation-1.json \
  --reference-directory root-reference-1 --new-output root-native-1
python3 validate_replay.py --output root-native-1 \
  --reference-directory root-reference-1 --implementation root-source-1.json \
  --build-attestation root-build-attestation-1.json
```

All outputs/source receipts stay in their own fresh replay directory. Source and
archive whole-file scans happen only under ROOT's serialized qualification slot,
never during timing. Original WNOWAIT/SID/PGID/TERM/INT-safe owned helper, caps,
finite raw transport and independent validators are retained; no second owner.

- Initial leaf scope: source-bound authentic teacher11 query6144 +prefix3072
  outputs and9216 retained input words, actual74240B payload/scratch, cancel,
  foreign/Cached refusal, real one-byte-short pressure and final0.
- H48 scope: authentic teachers11/353 preparation outputs288 +inputs192, raw
  BF16 dt consumed, raw A_log metadata retained UNEVALUATED, actual FA3 refusal,
  aggregate workspace/retention/final0. No exp, g or decay production.
- These do not establish initial chunk64 solver/BMM, complete layer/token engine,
  full112 domains/logits, trained 27B execution, external provider memory admission,
  CUDA or performance eligibility. All source-only fixtures remain unexecuted.

````

## Source: .agents/runs/model-execution-hf-combined525-attestation-20261020/frozen-prepared/TODO.md

Original UTF-8 bytes: 792; SHA256 `f02f892007b59ca93d1a49532c2ddf076d992b7863044b6e0cce69029db7189f`.
Resolve embedded relative paths against the original path, not this volume.

````text
Parent: model-execution. Owned: ignored combined525 attestation derivative only.

- [x] Initial leaf fresh self-audit; H48 independent source readiness review.
- [x] Identify exact524/522 attestor conflict; prepare exact525 disjoint union.
- [x] Fresh per-lane source/compiler/binary/stamp/run/validator bindings.
- [x] Preserve all original freezes, failures and source/ownership/math guards.
- [ ] Independent source review of this derivative and initial leaf.
- [ ] ROOT actual composition/locks/focused strict/full gate and two fresh builds.
- [ ] ROOT serialized genuine raw replay +independent validation, actual final0.

No builds/tests/imported scripts/operators/providers/models/GPU/payload scans by
author. MAIN/isolated worktree/shared plans unchanged; no numerical qualification.

````

## Source: .agents/runs/model-execution-hf-combined525-attestation-20261020/frozen-prepared/h48/replay/ROOT.md

Original UTF-8 bytes: 426; SHA256 `d0878b3d6756e37b66b173bf1b9fd2e70f6c1ba3359ff0a71563724ae42bf1ce`.
Resolve embedded relative paths against the original path, not this volume.

````text
# ROOT-only combined525 derivative

See ../../README.md for the only current composition/build/invocation sequence.
This fresh harness requires its own exact combined525 source attestation, actual
ROOT compiler receipt, mandatory build attestation and compiled lane stamp.
Original parent source/runner/validator freezes remain immutable.

No model, initial GDN core, exp/provider, session, full StateRoot or GPU eligibility.

````

## Source: .agents/runs/model-execution-hf-combined525-attestation-20261020/frozen-prepared/initial/replay/ROOT.md

Original UTF-8 bytes: 426; SHA256 `d0878b3d6756e37b66b173bf1b9fd2e70f6c1ba3359ff0a71563724ae42bf1ce`.
Resolve embedded relative paths against the original path, not this volume.

````text
# ROOT-only combined525 derivative

See ../../README.md for the only current composition/build/invocation sequence.
This fresh harness requires its own exact combined525 source attestation, actual
ROOT compiler receipt, mandatory build attestation and compiled lane stamp.
Original parent source/runner/validator freezes remain immutable.

No model, initial GDN core, exp/provider, session, full StateRoot or GPU eligibility.

````

## Source: .agents/runs/model-execution-hf-combined525-attestation-20261020/h48/replay/ROOT.md

Original UTF-8 bytes: 426; SHA256 `d0878b3d6756e37b66b173bf1b9fd2e70f6c1ba3359ff0a71563724ae42bf1ce`.
Resolve embedded relative paths against the original path, not this volume.

````text
# ROOT-only combined525 derivative

See ../../README.md for the only current composition/build/invocation sequence.
This fresh harness requires its own exact combined525 source attestation, actual
ROOT compiler receipt, mandatory build attestation and compiled lane stamp.
Original parent source/runner/validator freezes remain immutable.

No model, initial GDN core, exp/provider, session, full StateRoot or GPU eligibility.

````

## Source: .agents/runs/model-execution-hf-combined525-attestation-20261020/initial/replay/ROOT.md

Original UTF-8 bytes: 426; SHA256 `d0878b3d6756e37b66b173bf1b9fd2e70f6c1ba3359ff0a71563724ae42bf1ce`.
Resolve embedded relative paths against the original path, not this volume.

````text
# ROOT-only combined525 derivative

See ../../README.md for the only current composition/build/invocation sequence.
This fresh harness requires its own exact combined525 source attestation, actual
ROOT compiler receipt, mandatory build attestation and compiled lane stamp.
Original parent source/runner/validator freezes remain immutable.

No model, initial GDN core, exp/provider, session, full StateRoot or GPU eligibility.

````

## Source: .agents/runs/model-execution-hf-gdn-conv-contract-20261020/API.md

Original UTF-8 bytes: 4630; SHA256 `cb5b8c534eb1185ae93a9c9b387c1a6944641b2cf2315621fcd0f192732cceaf`.
Resolve embedded relative paths against the original path, not this volume.

````text
# Original GDN conv + SiLU private region

- SOURCE ONLY. No author build/test/operator/model/container/GPU execution.
- Parent bytes: `BASELINE.json`; 16 implementation/test paths + separate two-path
  GPU refusal proposal. ROOT reviews, merges with fresh mtimes, compiles/tests.
- Same `MathematicalBackend` → `PlanBuilder`/`HostMath` → `ForwardPlan` seam.
  No model/layer/token schedule, StateRoot owner, state effect or publication.
- Full original-HF forward/session/checkpoint guards remain CLOSED. No GGUF,
  old schema0x10/0x20, resident kernel or numerical provider behavior changes.

## Region and source

`CausalConvSiluContract::TorchCpuBf16Avx2` requires a bound raw
`SourceParameter`: original BF16 `[C,1,4]`, full `8C` bytes, `0<C<=16384`,
`C%32==0`; input C and canonical history3C are exact finite widened BF16.
Three private BF16 outputs: raw convolution C, final SiLU C, next history3C.
Plan recording requires the explicit CPU profile and a state-free slice;
foreign values, strided values, shape/precision/source mismatches refuse.
Unaware backends refuse before callbacks; GPU preflight/lowering cold-refuse.

`VerifiedTensorSource::conv4_parameter(name)` is an additional sealed original
rank3 descriptor. Existing `parameter()` stays rank1-only. Existing
`read_parameter*` preserves whole original metadata on partial ranges, same
issuer/location Arc, raw offsets/hash/FD, admission/cancel/drop. No fake rank1
matrix or partial `TensorSource` is created. Numerical code requires fullstage.

`OriginalWindowOperators::{append,execute}_gdn_convolution_silu_slice` validates
actual layer kind, original GdnConvolution role/name and full GdnQkv row count.
Loaded models only append existing same-builder operands; execution requires
window backing. Initial/cached external operands do not assert a producer,
committed root, epoch, token frontier or current cache ownership.

## Initial and cached operands

`GdnConvSliceInput` carries explicit raw BF16 activation/history bytes, bound
source/profile and `ConvInputPhase`:

- `InitialPadding`: actual original left3 padding must be supplied as +0 bytes;
  any negative-zero bit refuses, without canonicalization.
- `CachedTail`: exact original incoming `[C,4]` cache's last3 entries, oldest to
  newest. Signed zero is transported unchanged.

The accepted conv helper preserves the source four independent FMA partials
and three adds, then BF16 RNE; next3 is `[prior1,prior2,current]`. Accepted
contiguous16 BF16-input SiLU computes FP32, then the existing HostMath BF16 RNE
casts once. No stdlib exp substitution, zero canonicalization or tolerance.

IMPORTANT: accepted conv4 and contiguous SiLU are separate operator receipts.
Logical packing does NOT prove the original noncontiguous view selected the
same unary schedule. Fresh actual four-call comparisons below are mandatory;
this proposed region is UNQUALIFIED until ROOT executes them. No current
metadata/profile may be promoted on the basis of this source patch.

## Actual admitted backing

Both input Vecs share one reservation made BEFORE either allocation:
Output `4C*4` + Scratch960KiB. All three outputs share a separate aggregate
reservation BEFORE convolution/SILU allocations: Output`5C*4` + Scratch960KiB.
At C6144, live retained output payload is221184B and total input+result
reservations2187264B; full raw parameter49152B plus actual offset staging is
separately admitted. Scratch conservatively covers raw FP32, SiLU FP32,
rounding/output temporaries and small control allocations. Existing fixed
three-node/attention workspace policy is unchanged.

HostValue clones/views retain original backing and source owner. Borrowed
outputs only; `into_vec()` refuses leased storage. Buffers drop before their
shared reservation; all source/stage/input/result reservations retire after
last owners. These are actual ResourceArbiter charges, not a full-host RSS or
free-RAM admission claim.

## Prepared checks (not executed)

- Core: three BF16 outputs/no effects; raw kind/rank/byte/source/history guards;
  valid-geometry strided refusal then explicit pack; unaware callback refusal.
- Artifacts: full/partial metadata; foreign issuer; raw geometry/dtype;
  corruption, pre-cancel, actual pressure, retained window/final0.
- Model: original roles/same recorded backend; accepted helper wiring;
  signed-zero history; exact aggregate output counts; live clones/no Vec escape;
  pre-cancel and deterministic after-import cancellation; full-pool pressure
  and input-fit/result-one-byte-short pressure; foreign source/provider/strides;
  existing full-HF session/forward refusal and final0.

````

## Source: .agents/runs/model-execution-hf-gdn-conv-contract-20261020/ROOT-QUALIFICATION.md

Original UTF-8 bytes: 5076; SHA256 `40605f52876518d1cb6a28f40d6631a9a1618873945cdc5fecdadfc879521751`.
Resolve embedded relative paths against the original path, not this volume.

````text
# ROOT-only genuine qualification

Status: PREPARED / UNQUALIFIED. Latest additive freeze: `freeze-manifest-v2.json`;
initial `freeze-manifest.json`/`frozen-prepared/` retained unchanged. Author parsed/formatted source only. No raw
model/tensor payload was hashed or operator/reference/harness executed.
`bindings.json` copies expected source/capture hashes from genuine existing
receipts; author independently read/pinned only small source/metadata.

1. Independently review/apply NEW_DELTA + GPU_REFUSAL to the exact parent
   bytes; do not overwrite intervening work. Refresh modification times.
   Run focused CPU/core/artifact fixtures and relevant strict default/CPU/CUDA
   Clippy plus full isolated check. New tests are currently uncompiled.
2. ROOT runs `prepare_inputs.py --output <ownrun>/root-inputs-1` bounded under
   CPU4/finite wall and memory. It scans full original files with1MiB transfer
   buffers before/after, verifies original source and existing actual raw
   capture hashes, selects full original BF16 conv0 weights, observed QKV,
   actual raw K4 caches and captured activated output. Four cases:
   .8B/2B × teacher11(initial)/353(cached), source-36-F32-restored profile.
   Raw conv expectation is NOT fabricated or reconstructed here.
3. ROOT uses its reviewed owned-container lifecycle with the acquired pinned
   CPU image closure, networknone/no devices/12GiB/4CPUs, private nonce/name/CID,
   bounded logs/deadline and complete owned cleanup. Mount prepared own run
   read-only `/probe`, prepared input directory read-only `/inputs`, and a
   fresh private output parent writable `/results`. Inner argv:

   `python3 /probe/reference.py --inputs /inputs/inputs.json --inputs-sha256 ACTUAL_INPUTS_SHA --output /results/root-reference-1`

   Required environment: `ZG_CONV_CPUS=actual,four,allowed,CPUs`,
   `ATEN_CPU_CAPABILITY=avx2`, `OMP_NUM_THREADS=4`, `MKL_NUM_THREADS=4`,
   `OPENBLAS_NUM_THREADS=4`, CUDA/NVIDIA visibility disabled. Script sets/checks
   affinity before Torch load; pins Torch2.13+cu129/git, official modeling
   source, CPU AVX2/mkldnn/thread policy. ROOT binds actual image/DSO closure,
   invocation, script/input/output hashes and cleanup separately; a script
   report alone is not a runtime-closure receipt. No new standalone container
   lifecycle is introduced by this patch.

   Genuine public `F.conv1d` and `F.silu`, no model load or foreign token loop:
   initial actual padded4 → padding3 convolution → first4 view → SiLU → last
   teacher; cached actual incomingK4+new → conv → last1 view → SiLU. Preserve
   actual original slices/strides. Public updated K4 and activated bytes MUST
   equal the original unchanged-inference archives before any raw conv target
   is written. Public next3 must equal actual current cache's last3 bytes.
   Fresh raw conv targets are separate public-operator evidence, never a
   missing live trace. Any mismatch fails; never fit expectations or widen a
   tolerance. Original F32 learned36 profile remains unchanged; this region's
   actual conv weights are raw original BF16.
4. ROOT copies the genuine reference directory intact under this own run and
   pins actual input/reference reports, then builds `Cargo.toml` in its normal
   offline isolated coordinator path AFTER source merge. Cargo paths target
   MAIN; Rust helper calls only existing OriginalWindowOperators/HostMath.
   Run via `run_replay.py --binary ACTUAL_HELPER --binary-sha256 SHA --inputs
   ownrun/root-inputs-1/inputs.json --inputs-sha256 SHA --reference
   ownrun/root-reference-1/reference.json --reference-sha256 SHA --new-output
   ownrun/root-rust-1`.

   Unchanged accepted native leader cleanup is copied in `owned_lifetime.py`;
   ROOT must inspect/source-bind it and run its existing cleanup fixtures.
   Runner caps actual helper address space512MiB, four CPUs, finite900s per
   model/3300s total,4MiB log, exact executable FD and full source before/after.
   Both independent sources scan completely through VerifiedTensorSource and
   full320-text-role binding. Compare all6144 raw +6144 activated +18432 history
   words per call, signed bits including zeros:122880 words total across four
   calls. Actual caller input/prior bytes are also compared and retained.
5. ROOT independently compares saved actual raw BF16 outputs to public oracle
   targets and validates report/source/runtime/lifecycle bindings. Retained
   result views must match after original callers/source descriptors drop.
   Real pre-cancel, wrong-layer and complete host-pool pressure must leave the
   original exact ledger; synthetic deterministic after-import cancellation
   test is separate and is not relabeled as a trained mid-operator interrupt.
   Record actual per-class live input/result/scratch/staging reservations,
   final0, source equality and unchanged default full-HF guards.

Only exact selected private CPU operator evidence may pass. No full forward,
StateRoot/logits, cached GDN, initial chunk, source-streamed whole-model, GPU,
serving or performance eligibility follows.

````

## Source: .agents/runs/model-execution-hf-gdn-conv-contract-20261020/TODO.md

Original UTF-8 bytes: 1211; SHA256 `9bf6a9f764ca651fcc616833730bcf99c0940f306d4bfd6e0ea0170aaec3d99d`.
Resolve embedded relative paths against the original path, not this volume.

````text
# GDN conv+SiLU shared-engine working checklist

- Parent: model-execution; owner baseline_review; ROOT sole build/test/runtime/merge.
- Scope: proposed source only under this run; no MAIN/shared-plan edits.
- [x] Read accepted closed conv/SiLU helpers and pinned source first/cached routes.
- [x] Inspect genuine .8B/2B two-profile metadata/header only; no tensor payload/hash.
- [x] Prepare typed default-refusing pure region + same HostMath/ForwardPlan/role binding.
- [x] Prepare sealed raw rank3 conv parameter stage, exact aggregate output/scratch leases.
- [x] Prepare sourcefree ownership/geometry/refusal/cancel/pressure/lifetime tests.
- [x] Prepare ROOT authentic selected-operand public oracle and replay, no expected reconstruction.
- [x] Freeze source-only proposal for ROOT review.
- [ ] Independent peer source review (Spine reprioritized; none claimed).
- [ ] ROOT focused/strict/full gates, genuine operator oracle and122880-word replay; no execution claimed.

Full HF forward/session/checkpoint/publication and GPU eligibility stay CLOSED.

Source-only checks: rustfmt18 + helper Rust syntax; Python AST6; whitespace18.
Prepared fixtures:11 Rust +2 Python. No fixture was executed by author.

````

## Source: .agents/runs/model-execution-hf-gdn-conv-contract-20261020/frozen-prepared/API.md

Original UTF-8 bytes: 4630; SHA256 `cb5b8c534eb1185ae93a9c9b387c1a6944641b2cf2315621fcd0f192732cceaf`.
Resolve embedded relative paths against the original path, not this volume.

````text
# Original GDN conv + SiLU private region

- SOURCE ONLY. No author build/test/operator/model/container/GPU execution.
- Parent bytes: `BASELINE.json`; 16 implementation/test paths + separate two-path
  GPU refusal proposal. ROOT reviews, merges with fresh mtimes, compiles/tests.
- Same `MathematicalBackend` → `PlanBuilder`/`HostMath` → `ForwardPlan` seam.
  No model/layer/token schedule, StateRoot owner, state effect or publication.
- Full original-HF forward/session/checkpoint guards remain CLOSED. No GGUF,
  old schema0x10/0x20, resident kernel or numerical provider behavior changes.

## Region and source

`CausalConvSiluContract::TorchCpuBf16Avx2` requires a bound raw
`SourceParameter`: original BF16 `[C,1,4]`, full `8C` bytes, `0<C<=16384`,
`C%32==0`; input C and canonical history3C are exact finite widened BF16.
Three private BF16 outputs: raw convolution C, final SiLU C, next history3C.
Plan recording requires the explicit CPU profile and a state-free slice;
foreign values, strided values, shape/precision/source mismatches refuse.
Unaware backends refuse before callbacks; GPU preflight/lowering cold-refuse.

`VerifiedTensorSource::conv4_parameter(name)` is an additional sealed original
rank3 descriptor. Existing `parameter()` stays rank1-only. Existing
`read_parameter*` preserves whole original metadata on partial ranges, same
issuer/location Arc, raw offsets/hash/FD, admission/cancel/drop. No fake rank1
matrix or partial `TensorSource` is created. Numerical code requires fullstage.

`OriginalWindowOperators::{append,execute}_gdn_convolution_silu_slice` validates
actual layer kind, original GdnConvolution role/name and full GdnQkv row count.
Loaded models only append existing same-builder operands; execution requires
window backing. Initial/cached external operands do not assert a producer,
committed root, epoch, token frontier or current cache ownership.

## Initial and cached operands

`GdnConvSliceInput` carries explicit raw BF16 activation/history bytes, bound
source/profile and `ConvInputPhase`:

- `InitialPadding`: actual original left3 padding must be supplied as +0 bytes;
  any negative-zero bit refuses, without canonicalization.
- `CachedTail`: exact original incoming `[C,4]` cache's last3 entries, oldest to
  newest. Signed zero is transported unchanged.

The accepted conv helper preserves the source four independent FMA partials
and three adds, then BF16 RNE; next3 is `[prior1,prior2,current]`. Accepted
contiguous16 BF16-input SiLU computes FP32, then the existing HostMath BF16 RNE
casts once. No stdlib exp substitution, zero canonicalization or tolerance.

IMPORTANT: accepted conv4 and contiguous SiLU are separate operator receipts.
Logical packing does NOT prove the original noncontiguous view selected the
same unary schedule. Fresh actual four-call comparisons below are mandatory;
this proposed region is UNQUALIFIED until ROOT executes them. No current
metadata/profile may be promoted on the basis of this source patch.

## Actual admitted backing

Both input Vecs share one reservation made BEFORE either allocation:
Output `4C*4` + Scratch960KiB. All three outputs share a separate aggregate
reservation BEFORE convolution/SILU allocations: Output`5C*4` + Scratch960KiB.
At C6144, live retained output payload is221184B and total input+result
reservations2187264B; full raw parameter49152B plus actual offset staging is
separately admitted. Scratch conservatively covers raw FP32, SiLU FP32,
rounding/output temporaries and small control allocations. Existing fixed
three-node/attention workspace policy is unchanged.

HostValue clones/views retain original backing and source owner. Borrowed
outputs only; `into_vec()` refuses leased storage. Buffers drop before their
shared reservation; all source/stage/input/result reservations retire after
last owners. These are actual ResourceArbiter charges, not a full-host RSS or
free-RAM admission claim.

## Prepared checks (not executed)

- Core: three BF16 outputs/no effects; raw kind/rank/byte/source/history guards;
  valid-geometry strided refusal then explicit pack; unaware callback refusal.
- Artifacts: full/partial metadata; foreign issuer; raw geometry/dtype;
  corruption, pre-cancel, actual pressure, retained window/final0.
- Model: original roles/same recorded backend; accepted helper wiring;
  signed-zero history; exact aggregate output counts; live clones/no Vec escape;
  pre-cancel and deterministic after-import cancellation; full-pool pressure
  and input-fit/result-one-byte-short pressure; foreign source/provider/strides;
  existing full-HF session/forward refusal and final0.

````

## Source: .agents/runs/model-execution-hf-gdn-conv-contract-20261020/frozen-prepared/ROOT-QUALIFICATION.md

Original UTF-8 bytes: 4955; SHA256 `86871430185768dababc000ec587379f4e8d6bc2bb785a56c506ba366f3ad94d`.
Resolve embedded relative paths against the original path, not this volume.

````text
# ROOT-only genuine qualification

Status: PREPARED / UNQUALIFIED. Author parsed/formatted source only. No raw
model/tensor payload was hashed or operator/reference/harness executed.
`bindings.json` copies expected source/capture hashes from genuine existing
receipts; author independently read/pinned only small source/metadata.

1. Independently review/apply NEW_DELTA + GPU_REFUSAL to the exact parent
   bytes; do not overwrite intervening work. Refresh modification times.
   Run focused CPU/core/artifact fixtures and relevant strict default/CPU/CUDA
   Clippy plus full isolated check. New tests are currently uncompiled.
2. ROOT runs `prepare_inputs.py --output <ownrun>/root-inputs-1` bounded under
   CPU4/finite wall and memory. It scans full original files with1MiB transfer
   buffers before/after, verifies original source and existing actual raw
   capture hashes, selects full original BF16 conv0 weights, observed QKV,
   actual raw K4 caches and captured activated output. Four cases:
   .8B/2B × teacher11(initial)/353(cached), source-36-F32-restored profile.
   Raw conv expectation is NOT fabricated or reconstructed here.
3. ROOT uses its reviewed owned-container lifecycle with the acquired pinned
   CPU image closure, networknone/no devices/12GiB/4CPUs, private nonce/name/CID,
   bounded logs/deadline and complete owned cleanup. Mount prepared own run
   read-only `/probe`, prepared input directory read-only `/inputs`, and a
   fresh private output parent writable `/results`. Inner argv:

   `python3 /probe/reference.py --inputs /inputs/inputs.json --inputs-sha256 ACTUAL_INPUTS_SHA --output /results/root-reference-1`

   Required environment: `ZG_CONV_CPUS=actual,four,allowed,CPUs`,
   `ATEN_CPU_CAPABILITY=avx2`, `OMP_NUM_THREADS=4`, `MKL_NUM_THREADS=4`,
   `OPENBLAS_NUM_THREADS=4`, CUDA/NVIDIA visibility disabled. Script sets/checks
   affinity before Torch load; pins Torch2.13+cu129/git, official modeling
   source, CPU AVX2/mkldnn/thread policy. ROOT binds actual image/DSO closure,
   invocation, script/input/output hashes and cleanup separately; a script
   report alone is not a runtime-closure receipt. No new standalone container
   lifecycle is introduced by this patch.

   Genuine public `F.conv1d` and `F.silu`, no model load or foreign token loop:
   initial actual padded4 → padding3 convolution → first4 view → SiLU → last
   teacher; cached actual incomingK4+new → conv → last1 view → SiLU. Preserve
   actual original slices/strides. Public updated K4 and activated bytes MUST
   equal the original unchanged-inference archives before any raw conv target
   is written. Public next3 must equal actual current cache's last3 bytes.
   Fresh raw conv targets are separate public-operator evidence, never a
   missing live trace. Any mismatch fails; never fit expectations or widen a
   tolerance. Original F32 learned36 profile remains unchanged; this region's
   actual conv weights are raw original BF16.
4. ROOT copies the genuine reference directory intact under this own run and
   pins actual input/reference reports, then builds `Cargo.toml` in its normal
   offline isolated coordinator path AFTER source merge. Cargo paths target
   MAIN; Rust helper calls only existing OriginalWindowOperators/HostMath.
   Run via `run_replay.py --binary ACTUAL_HELPER --binary-sha256 SHA --inputs
   ownrun/root-inputs-1/inputs.json --inputs-sha256 SHA --reference
   ownrun/root-reference-1/reference.json --reference-sha256 SHA --new-output
   ownrun/root-rust-1`.

   Unchanged accepted native leader cleanup is copied in `owned_lifetime.py`;
   ROOT must inspect/source-bind it and run its existing cleanup fixtures.
   Runner caps actual helper address space512MiB, four CPUs, finite900s per
   model/3300s total,4MiB log, exact executable FD and full source before/after.
   Both independent sources scan completely through VerifiedTensorSource and
   full320-text-role binding. Compare all6144 raw +6144 activated +18432 history
   words per call, signed bits including zeros:122880 words total across four
   calls. Actual caller input/prior bytes are also compared and retained.
5. ROOT independently compares saved actual raw BF16 outputs to public oracle
   targets and validates report/source/runtime/lifecycle bindings. Retained
   result views must match after original callers/source descriptors drop.
   Real pre-cancel, wrong-layer and complete host-pool pressure must leave the
   original exact ledger; synthetic deterministic after-import cancellation
   test is separate and is not relabeled as a trained mid-operator interrupt.
   Record actual per-class live input/result/scratch/staging reservations,
   final0, source equality and unchanged default full-HF guards.

Only exact selected private CPU operator evidence may pass. No full forward,
StateRoot/logits, cached GDN, initial chunk, source-streamed whole-model, GPU,
serving or performance eligibility follows.

````

## Source: .agents/runs/model-execution-hf-gdn-conv-contract-20261020/frozen-prepared/TODO.md

Original UTF-8 bytes: 1211; SHA256 `9bf6a9f764ca651fcc616833730bcf99c0940f306d4bfd6e0ea0170aaec3d99d`.
Resolve embedded relative paths against the original path, not this volume.

````text
# GDN conv+SiLU shared-engine working checklist

- Parent: model-execution; owner baseline_review; ROOT sole build/test/runtime/merge.
- Scope: proposed source only under this run; no MAIN/shared-plan edits.
- [x] Read accepted closed conv/SiLU helpers and pinned source first/cached routes.
- [x] Inspect genuine .8B/2B two-profile metadata/header only; no tensor payload/hash.
- [x] Prepare typed default-refusing pure region + same HostMath/ForwardPlan/role binding.
- [x] Prepare sealed raw rank3 conv parameter stage, exact aggregate output/scratch leases.
- [x] Prepare sourcefree ownership/geometry/refusal/cancel/pressure/lifetime tests.
- [x] Prepare ROOT authentic selected-operand public oracle and replay, no expected reconstruction.
- [x] Freeze source-only proposal for ROOT review.
- [ ] Independent peer source review (Spine reprioritized; none claimed).
- [ ] ROOT focused/strict/full gates, genuine operator oracle and122880-word replay; no execution claimed.

Full HF forward/session/checkpoint/publication and GPU eligibility stay CLOSED.

Source-only checks: rustfmt18 + helper Rust syntax; Python AST6; whitespace18.
Prepared fixtures:11 Rust +2 Python. No fixture was executed by author.

````

## Source: .agents/runs/model-execution-hf-gdn-conv-contract-20261020/frozen-v2/API.md

Original UTF-8 bytes: 4630; SHA256 `cb5b8c534eb1185ae93a9c9b387c1a6944641b2cf2315621fcd0f192732cceaf`.
Resolve embedded relative paths against the original path, not this volume.

````text
# Original GDN conv + SiLU private region

- SOURCE ONLY. No author build/test/operator/model/container/GPU execution.
- Parent bytes: `BASELINE.json`; 16 implementation/test paths + separate two-path
  GPU refusal proposal. ROOT reviews, merges with fresh mtimes, compiles/tests.
- Same `MathematicalBackend` → `PlanBuilder`/`HostMath` → `ForwardPlan` seam.
  No model/layer/token schedule, StateRoot owner, state effect or publication.
- Full original-HF forward/session/checkpoint guards remain CLOSED. No GGUF,
  old schema0x10/0x20, resident kernel or numerical provider behavior changes.

## Region and source

`CausalConvSiluContract::TorchCpuBf16Avx2` requires a bound raw
`SourceParameter`: original BF16 `[C,1,4]`, full `8C` bytes, `0<C<=16384`,
`C%32==0`; input C and canonical history3C are exact finite widened BF16.
Three private BF16 outputs: raw convolution C, final SiLU C, next history3C.
Plan recording requires the explicit CPU profile and a state-free slice;
foreign values, strided values, shape/precision/source mismatches refuse.
Unaware backends refuse before callbacks; GPU preflight/lowering cold-refuse.

`VerifiedTensorSource::conv4_parameter(name)` is an additional sealed original
rank3 descriptor. Existing `parameter()` stays rank1-only. Existing
`read_parameter*` preserves whole original metadata on partial ranges, same
issuer/location Arc, raw offsets/hash/FD, admission/cancel/drop. No fake rank1
matrix or partial `TensorSource` is created. Numerical code requires fullstage.

`OriginalWindowOperators::{append,execute}_gdn_convolution_silu_slice` validates
actual layer kind, original GdnConvolution role/name and full GdnQkv row count.
Loaded models only append existing same-builder operands; execution requires
window backing. Initial/cached external operands do not assert a producer,
committed root, epoch, token frontier or current cache ownership.

## Initial and cached operands

`GdnConvSliceInput` carries explicit raw BF16 activation/history bytes, bound
source/profile and `ConvInputPhase`:

- `InitialPadding`: actual original left3 padding must be supplied as +0 bytes;
  any negative-zero bit refuses, without canonicalization.
- `CachedTail`: exact original incoming `[C,4]` cache's last3 entries, oldest to
  newest. Signed zero is transported unchanged.

The accepted conv helper preserves the source four independent FMA partials
and three adds, then BF16 RNE; next3 is `[prior1,prior2,current]`. Accepted
contiguous16 BF16-input SiLU computes FP32, then the existing HostMath BF16 RNE
casts once. No stdlib exp substitution, zero canonicalization or tolerance.

IMPORTANT: accepted conv4 and contiguous SiLU are separate operator receipts.
Logical packing does NOT prove the original noncontiguous view selected the
same unary schedule. Fresh actual four-call comparisons below are mandatory;
this proposed region is UNQUALIFIED until ROOT executes them. No current
metadata/profile may be promoted on the basis of this source patch.

## Actual admitted backing

Both input Vecs share one reservation made BEFORE either allocation:
Output `4C*4` + Scratch960KiB. All three outputs share a separate aggregate
reservation BEFORE convolution/SILU allocations: Output`5C*4` + Scratch960KiB.
At C6144, live retained output payload is221184B and total input+result
reservations2187264B; full raw parameter49152B plus actual offset staging is
separately admitted. Scratch conservatively covers raw FP32, SiLU FP32,
rounding/output temporaries and small control allocations. Existing fixed
three-node/attention workspace policy is unchanged.

HostValue clones/views retain original backing and source owner. Borrowed
outputs only; `into_vec()` refuses leased storage. Buffers drop before their
shared reservation; all source/stage/input/result reservations retire after
last owners. These are actual ResourceArbiter charges, not a full-host RSS or
free-RAM admission claim.

## Prepared checks (not executed)

- Core: three BF16 outputs/no effects; raw kind/rank/byte/source/history guards;
  valid-geometry strided refusal then explicit pack; unaware callback refusal.
- Artifacts: full/partial metadata; foreign issuer; raw geometry/dtype;
  corruption, pre-cancel, actual pressure, retained window/final0.
- Model: original roles/same recorded backend; accepted helper wiring;
  signed-zero history; exact aggregate output counts; live clones/no Vec escape;
  pre-cancel and deterministic after-import cancellation; full-pool pressure
  and input-fit/result-one-byte-short pressure; foreign source/provider/strides;
  existing full-HF session/forward refusal and final0.

````

## Source: .agents/runs/model-execution-hf-gdn-conv-contract-20261020/frozen-v2/ROOT-QUALIFICATION.md

Original UTF-8 bytes: 5076; SHA256 `40605f52876518d1cb6a28f40d6631a9a1618873945cdc5fecdadfc879521751`.
Resolve embedded relative paths against the original path, not this volume.

````text
# ROOT-only genuine qualification

Status: PREPARED / UNQUALIFIED. Latest additive freeze: `freeze-manifest-v2.json`;
initial `freeze-manifest.json`/`frozen-prepared/` retained unchanged. Author parsed/formatted source only. No raw
model/tensor payload was hashed or operator/reference/harness executed.
`bindings.json` copies expected source/capture hashes from genuine existing
receipts; author independently read/pinned only small source/metadata.

1. Independently review/apply NEW_DELTA + GPU_REFUSAL to the exact parent
   bytes; do not overwrite intervening work. Refresh modification times.
   Run focused CPU/core/artifact fixtures and relevant strict default/CPU/CUDA
   Clippy plus full isolated check. New tests are currently uncompiled.
2. ROOT runs `prepare_inputs.py --output <ownrun>/root-inputs-1` bounded under
   CPU4/finite wall and memory. It scans full original files with1MiB transfer
   buffers before/after, verifies original source and existing actual raw
   capture hashes, selects full original BF16 conv0 weights, observed QKV,
   actual raw K4 caches and captured activated output. Four cases:
   .8B/2B × teacher11(initial)/353(cached), source-36-F32-restored profile.
   Raw conv expectation is NOT fabricated or reconstructed here.
3. ROOT uses its reviewed owned-container lifecycle with the acquired pinned
   CPU image closure, networknone/no devices/12GiB/4CPUs, private nonce/name/CID,
   bounded logs/deadline and complete owned cleanup. Mount prepared own run
   read-only `/probe`, prepared input directory read-only `/inputs`, and a
   fresh private output parent writable `/results`. Inner argv:

   `python3 /probe/reference.py --inputs /inputs/inputs.json --inputs-sha256 ACTUAL_INPUTS_SHA --output /results/root-reference-1`

   Required environment: `ZG_CONV_CPUS=actual,four,allowed,CPUs`,
   `ATEN_CPU_CAPABILITY=avx2`, `OMP_NUM_THREADS=4`, `MKL_NUM_THREADS=4`,
   `OPENBLAS_NUM_THREADS=4`, CUDA/NVIDIA visibility disabled. Script sets/checks
   affinity before Torch load; pins Torch2.13+cu129/git, official modeling
   source, CPU AVX2/mkldnn/thread policy. ROOT binds actual image/DSO closure,
   invocation, script/input/output hashes and cleanup separately; a script
   report alone is not a runtime-closure receipt. No new standalone container
   lifecycle is introduced by this patch.

   Genuine public `F.conv1d` and `F.silu`, no model load or foreign token loop:
   initial actual padded4 → padding3 convolution → first4 view → SiLU → last
   teacher; cached actual incomingK4+new → conv → last1 view → SiLU. Preserve
   actual original slices/strides. Public updated K4 and activated bytes MUST
   equal the original unchanged-inference archives before any raw conv target
   is written. Public next3 must equal actual current cache's last3 bytes.
   Fresh raw conv targets are separate public-operator evidence, never a
   missing live trace. Any mismatch fails; never fit expectations or widen a
   tolerance. Original F32 learned36 profile remains unchanged; this region's
   actual conv weights are raw original BF16.
4. ROOT copies the genuine reference directory intact under this own run and
   pins actual input/reference reports, then builds `Cargo.toml` in its normal
   offline isolated coordinator path AFTER source merge. Cargo paths target
   MAIN; Rust helper calls only existing OriginalWindowOperators/HostMath.
   Run via `run_replay.py --binary ACTUAL_HELPER --binary-sha256 SHA --inputs
   ownrun/root-inputs-1/inputs.json --inputs-sha256 SHA --reference
   ownrun/root-reference-1/reference.json --reference-sha256 SHA --new-output
   ownrun/root-rust-1`.

   Unchanged accepted native leader cleanup is copied in `owned_lifetime.py`;
   ROOT must inspect/source-bind it and run its existing cleanup fixtures.
   Runner caps actual helper address space512MiB, four CPUs, finite900s per
   model/3300s total,4MiB log, exact executable FD and full source before/after.
   Both independent sources scan completely through VerifiedTensorSource and
   full320-text-role binding. Compare all6144 raw +6144 activated +18432 history
   words per call, signed bits including zeros:122880 words total across four
   calls. Actual caller input/prior bytes are also compared and retained.
5. ROOT independently compares saved actual raw BF16 outputs to public oracle
   targets and validates report/source/runtime/lifecycle bindings. Retained
   result views must match after original callers/source descriptors drop.
   Real pre-cancel, wrong-layer and complete host-pool pressure must leave the
   original exact ledger; synthetic deterministic after-import cancellation
   test is separate and is not relabeled as a trained mid-operator interrupt.
   Record actual per-class live input/result/scratch/staging reservations,
   final0, source equality and unchanged default full-HF guards.

Only exact selected private CPU operator evidence may pass. No full forward,
StateRoot/logits, cached GDN, initial chunk, source-streamed whole-model, GPU,
serving or performance eligibility follows.

````

## Source: .agents/runs/model-execution-hf-gdn-conv-contract-20261020/frozen-v2/TODO.md

Original UTF-8 bytes: 1211; SHA256 `9bf6a9f764ca651fcc616833730bcf99c0940f306d4bfd6e0ea0170aaec3d99d`.
Resolve embedded relative paths against the original path, not this volume.

````text
# GDN conv+SiLU shared-engine working checklist

- Parent: model-execution; owner baseline_review; ROOT sole build/test/runtime/merge.
- Scope: proposed source only under this run; no MAIN/shared-plan edits.
- [x] Read accepted closed conv/SiLU helpers and pinned source first/cached routes.
- [x] Inspect genuine .8B/2B two-profile metadata/header only; no tensor payload/hash.
- [x] Prepare typed default-refusing pure region + same HostMath/ForwardPlan/role binding.
- [x] Prepare sealed raw rank3 conv parameter stage, exact aggregate output/scratch leases.
- [x] Prepare sourcefree ownership/geometry/refusal/cancel/pressure/lifetime tests.
- [x] Prepare ROOT authentic selected-operand public oracle and replay, no expected reconstruction.
- [x] Freeze source-only proposal for ROOT review.
- [ ] Independent peer source review (Spine reprioritized; none claimed).
- [ ] ROOT focused/strict/full gates, genuine operator oracle and122880-word replay; no execution claimed.

Full HF forward/session/checkpoint/publication and GPU eligibility stay CLOSED.

Source-only checks: rustfmt18 + helper Rust syntax; Python AST6; whitespace18.
Prepared fixtures:11 Rust +2 Python. No fixture was executed by author.

````

## Source: .agents/runs/model-execution-hf-gdn-conv-root-v5-20261020/API.md

Original UTF-8 bytes: 4630; SHA256 `cb5b8c534eb1185ae93a9c9b387c1a6944641b2cf2315621fcd0f192732cceaf`.
Resolve embedded relative paths against the original path, not this volume.

````text
# Original GDN conv + SiLU private region

- SOURCE ONLY. No author build/test/operator/model/container/GPU execution.
- Parent bytes: `BASELINE.json`; 16 implementation/test paths + separate two-path
  GPU refusal proposal. ROOT reviews, merges with fresh mtimes, compiles/tests.
- Same `MathematicalBackend` → `PlanBuilder`/`HostMath` → `ForwardPlan` seam.
  No model/layer/token schedule, StateRoot owner, state effect or publication.
- Full original-HF forward/session/checkpoint guards remain CLOSED. No GGUF,
  old schema0x10/0x20, resident kernel or numerical provider behavior changes.

## Region and source

`CausalConvSiluContract::TorchCpuBf16Avx2` requires a bound raw
`SourceParameter`: original BF16 `[C,1,4]`, full `8C` bytes, `0<C<=16384`,
`C%32==0`; input C and canonical history3C are exact finite widened BF16.
Three private BF16 outputs: raw convolution C, final SiLU C, next history3C.
Plan recording requires the explicit CPU profile and a state-free slice;
foreign values, strided values, shape/precision/source mismatches refuse.
Unaware backends refuse before callbacks; GPU preflight/lowering cold-refuse.

`VerifiedTensorSource::conv4_parameter(name)` is an additional sealed original
rank3 descriptor. Existing `parameter()` stays rank1-only. Existing
`read_parameter*` preserves whole original metadata on partial ranges, same
issuer/location Arc, raw offsets/hash/FD, admission/cancel/drop. No fake rank1
matrix or partial `TensorSource` is created. Numerical code requires fullstage.

`OriginalWindowOperators::{append,execute}_gdn_convolution_silu_slice` validates
actual layer kind, original GdnConvolution role/name and full GdnQkv row count.
Loaded models only append existing same-builder operands; execution requires
window backing. Initial/cached external operands do not assert a producer,
committed root, epoch, token frontier or current cache ownership.

## Initial and cached operands

`GdnConvSliceInput` carries explicit raw BF16 activation/history bytes, bound
source/profile and `ConvInputPhase`:

- `InitialPadding`: actual original left3 padding must be supplied as +0 bytes;
  any negative-zero bit refuses, without canonicalization.
- `CachedTail`: exact original incoming `[C,4]` cache's last3 entries, oldest to
  newest. Signed zero is transported unchanged.

The accepted conv helper preserves the source four independent FMA partials
and three adds, then BF16 RNE; next3 is `[prior1,prior2,current]`. Accepted
contiguous16 BF16-input SiLU computes FP32, then the existing HostMath BF16 RNE
casts once. No stdlib exp substitution, zero canonicalization or tolerance.

IMPORTANT: accepted conv4 and contiguous SiLU are separate operator receipts.
Logical packing does NOT prove the original noncontiguous view selected the
same unary schedule. Fresh actual four-call comparisons below are mandatory;
this proposed region is UNQUALIFIED until ROOT executes them. No current
metadata/profile may be promoted on the basis of this source patch.

## Actual admitted backing

Both input Vecs share one reservation made BEFORE either allocation:
Output `4C*4` + Scratch960KiB. All three outputs share a separate aggregate
reservation BEFORE convolution/SILU allocations: Output`5C*4` + Scratch960KiB.
At C6144, live retained output payload is221184B and total input+result
reservations2187264B; full raw parameter49152B plus actual offset staging is
separately admitted. Scratch conservatively covers raw FP32, SiLU FP32,
rounding/output temporaries and small control allocations. Existing fixed
three-node/attention workspace policy is unchanged.

HostValue clones/views retain original backing and source owner. Borrowed
outputs only; `into_vec()` refuses leased storage. Buffers drop before their
shared reservation; all source/stage/input/result reservations retire after
last owners. These are actual ResourceArbiter charges, not a full-host RSS or
free-RAM admission claim.

## Prepared checks (not executed)

- Core: three BF16 outputs/no effects; raw kind/rank/byte/source/history guards;
  valid-geometry strided refusal then explicit pack; unaware callback refusal.
- Artifacts: full/partial metadata; foreign issuer; raw geometry/dtype;
  corruption, pre-cancel, actual pressure, retained window/final0.
- Model: original roles/same recorded backend; accepted helper wiring;
  signed-zero history; exact aggregate output counts; live clones/no Vec escape;
  pre-cancel and deterministic after-import cancellation; full-pool pressure
  and input-fit/result-one-byte-short pressure; foreign source/provider/strides;
  existing full-HF session/forward refusal and final0.

````

## Source: .agents/runs/model-execution-hf-gdn-conv-root-v5-20261020/ROOT-QUALIFICATION.md

Original UTF-8 bytes: 5076; SHA256 `40605f52876518d1cb6a28f40d6631a9a1618873945cdc5fecdadfc879521751`.
Resolve embedded relative paths against the original path, not this volume.

````text
# ROOT-only genuine qualification

Status: PREPARED / UNQUALIFIED. Latest additive freeze: `freeze-manifest-v2.json`;
initial `freeze-manifest.json`/`frozen-prepared/` retained unchanged. Author parsed/formatted source only. No raw
model/tensor payload was hashed or operator/reference/harness executed.
`bindings.json` copies expected source/capture hashes from genuine existing
receipts; author independently read/pinned only small source/metadata.

1. Independently review/apply NEW_DELTA + GPU_REFUSAL to the exact parent
   bytes; do not overwrite intervening work. Refresh modification times.
   Run focused CPU/core/artifact fixtures and relevant strict default/CPU/CUDA
   Clippy plus full isolated check. New tests are currently uncompiled.
2. ROOT runs `prepare_inputs.py --output <ownrun>/root-inputs-1` bounded under
   CPU4/finite wall and memory. It scans full original files with1MiB transfer
   buffers before/after, verifies original source and existing actual raw
   capture hashes, selects full original BF16 conv0 weights, observed QKV,
   actual raw K4 caches and captured activated output. Four cases:
   .8B/2B × teacher11(initial)/353(cached), source-36-F32-restored profile.
   Raw conv expectation is NOT fabricated or reconstructed here.
3. ROOT uses its reviewed owned-container lifecycle with the acquired pinned
   CPU image closure, networknone/no devices/12GiB/4CPUs, private nonce/name/CID,
   bounded logs/deadline and complete owned cleanup. Mount prepared own run
   read-only `/probe`, prepared input directory read-only `/inputs`, and a
   fresh private output parent writable `/results`. Inner argv:

   `python3 /probe/reference.py --inputs /inputs/inputs.json --inputs-sha256 ACTUAL_INPUTS_SHA --output /results/root-reference-1`

   Required environment: `ZG_CONV_CPUS=actual,four,allowed,CPUs`,
   `ATEN_CPU_CAPABILITY=avx2`, `OMP_NUM_THREADS=4`, `MKL_NUM_THREADS=4`,
   `OPENBLAS_NUM_THREADS=4`, CUDA/NVIDIA visibility disabled. Script sets/checks
   affinity before Torch load; pins Torch2.13+cu129/git, official modeling
   source, CPU AVX2/mkldnn/thread policy. ROOT binds actual image/DSO closure,
   invocation, script/input/output hashes and cleanup separately; a script
   report alone is not a runtime-closure receipt. No new standalone container
   lifecycle is introduced by this patch.

   Genuine public `F.conv1d` and `F.silu`, no model load or foreign token loop:
   initial actual padded4 → padding3 convolution → first4 view → SiLU → last
   teacher; cached actual incomingK4+new → conv → last1 view → SiLU. Preserve
   actual original slices/strides. Public updated K4 and activated bytes MUST
   equal the original unchanged-inference archives before any raw conv target
   is written. Public next3 must equal actual current cache's last3 bytes.
   Fresh raw conv targets are separate public-operator evidence, never a
   missing live trace. Any mismatch fails; never fit expectations or widen a
   tolerance. Original F32 learned36 profile remains unchanged; this region's
   actual conv weights are raw original BF16.
4. ROOT copies the genuine reference directory intact under this own run and
   pins actual input/reference reports, then builds `Cargo.toml` in its normal
   offline isolated coordinator path AFTER source merge. Cargo paths target
   MAIN; Rust helper calls only existing OriginalWindowOperators/HostMath.
   Run via `run_replay.py --binary ACTUAL_HELPER --binary-sha256 SHA --inputs
   ownrun/root-inputs-1/inputs.json --inputs-sha256 SHA --reference
   ownrun/root-reference-1/reference.json --reference-sha256 SHA --new-output
   ownrun/root-rust-1`.

   Unchanged accepted native leader cleanup is copied in `owned_lifetime.py`;
   ROOT must inspect/source-bind it and run its existing cleanup fixtures.
   Runner caps actual helper address space512MiB, four CPUs, finite900s per
   model/3300s total,4MiB log, exact executable FD and full source before/after.
   Both independent sources scan completely through VerifiedTensorSource and
   full320-text-role binding. Compare all6144 raw +6144 activated +18432 history
   words per call, signed bits including zeros:122880 words total across four
   calls. Actual caller input/prior bytes are also compared and retained.
5. ROOT independently compares saved actual raw BF16 outputs to public oracle
   targets and validates report/source/runtime/lifecycle bindings. Retained
   result views must match after original callers/source descriptors drop.
   Real pre-cancel, wrong-layer and complete host-pool pressure must leave the
   original exact ledger; synthetic deterministic after-import cancellation
   test is separate and is not relabeled as a trained mid-operator interrupt.
   Record actual per-class live input/result/scratch/staging reservations,
   final0, source equality and unchanged default full-HF guards.

Only exact selected private CPU operator evidence may pass. No full forward,
StateRoot/logits, cached GDN, initial chunk, source-streamed whole-model, GPU,
serving or performance eligibility follows.

````

## Source: .agents/runs/model-execution-hf-gdn-conv-root-v5-20261020/TODO.md

Original UTF-8 bytes: 1211; SHA256 `9bf6a9f764ca651fcc616833730bcf99c0940f306d4bfd6e0ea0170aaec3d99d`.
Resolve embedded relative paths against the original path, not this volume.

````text
# GDN conv+SiLU shared-engine working checklist

- Parent: model-execution; owner baseline_review; ROOT sole build/test/runtime/merge.
- Scope: proposed source only under this run; no MAIN/shared-plan edits.
- [x] Read accepted closed conv/SiLU helpers and pinned source first/cached routes.
- [x] Inspect genuine .8B/2B two-profile metadata/header only; no tensor payload/hash.
- [x] Prepare typed default-refusing pure region + same HostMath/ForwardPlan/role binding.
- [x] Prepare sealed raw rank3 conv parameter stage, exact aggregate output/scratch leases.
- [x] Prepare sourcefree ownership/geometry/refusal/cancel/pressure/lifetime tests.
- [x] Prepare ROOT authentic selected-operand public oracle and replay, no expected reconstruction.
- [x] Freeze source-only proposal for ROOT review.
- [ ] Independent peer source review (Spine reprioritized; none claimed).
- [ ] ROOT focused/strict/full gates, genuine operator oracle and122880-word replay; no execution claimed.

Full HF forward/session/checkpoint/publication and GPU eligibility stay CLOSED.

Source-only checks: rustfmt18 + helper Rust syntax; Python AST6; whitespace18.
Prepared fixtures:11 Rust +2 Python. No fixture was executed by author.

````

## Source: .agents/runs/model-execution-hf-gdn-conv-root-v6-20261020/API.md

Original UTF-8 bytes: 4630; SHA256 `cb5b8c534eb1185ae93a9c9b387c1a6944641b2cf2315621fcd0f192732cceaf`.
Resolve embedded relative paths against the original path, not this volume.

````text
# Original GDN conv + SiLU private region

- SOURCE ONLY. No author build/test/operator/model/container/GPU execution.
- Parent bytes: `BASELINE.json`; 16 implementation/test paths + separate two-path
  GPU refusal proposal. ROOT reviews, merges with fresh mtimes, compiles/tests.
- Same `MathematicalBackend` → `PlanBuilder`/`HostMath` → `ForwardPlan` seam.
  No model/layer/token schedule, StateRoot owner, state effect or publication.
- Full original-HF forward/session/checkpoint guards remain CLOSED. No GGUF,
  old schema0x10/0x20, resident kernel or numerical provider behavior changes.

## Region and source

`CausalConvSiluContract::TorchCpuBf16Avx2` requires a bound raw
`SourceParameter`: original BF16 `[C,1,4]`, full `8C` bytes, `0<C<=16384`,
`C%32==0`; input C and canonical history3C are exact finite widened BF16.
Three private BF16 outputs: raw convolution C, final SiLU C, next history3C.
Plan recording requires the explicit CPU profile and a state-free slice;
foreign values, strided values, shape/precision/source mismatches refuse.
Unaware backends refuse before callbacks; GPU preflight/lowering cold-refuse.

`VerifiedTensorSource::conv4_parameter(name)` is an additional sealed original
rank3 descriptor. Existing `parameter()` stays rank1-only. Existing
`read_parameter*` preserves whole original metadata on partial ranges, same
issuer/location Arc, raw offsets/hash/FD, admission/cancel/drop. No fake rank1
matrix or partial `TensorSource` is created. Numerical code requires fullstage.

`OriginalWindowOperators::{append,execute}_gdn_convolution_silu_slice` validates
actual layer kind, original GdnConvolution role/name and full GdnQkv row count.
Loaded models only append existing same-builder operands; execution requires
window backing. Initial/cached external operands do not assert a producer,
committed root, epoch, token frontier or current cache ownership.

## Initial and cached operands

`GdnConvSliceInput` carries explicit raw BF16 activation/history bytes, bound
source/profile and `ConvInputPhase`:

- `InitialPadding`: actual original left3 padding must be supplied as +0 bytes;
  any negative-zero bit refuses, without canonicalization.
- `CachedTail`: exact original incoming `[C,4]` cache's last3 entries, oldest to
  newest. Signed zero is transported unchanged.

The accepted conv helper preserves the source four independent FMA partials
and three adds, then BF16 RNE; next3 is `[prior1,prior2,current]`. Accepted
contiguous16 BF16-input SiLU computes FP32, then the existing HostMath BF16 RNE
casts once. No stdlib exp substitution, zero canonicalization or tolerance.

IMPORTANT: accepted conv4 and contiguous SiLU are separate operator receipts.
Logical packing does NOT prove the original noncontiguous view selected the
same unary schedule. Fresh actual four-call comparisons below are mandatory;
this proposed region is UNQUALIFIED until ROOT executes them. No current
metadata/profile may be promoted on the basis of this source patch.

## Actual admitted backing

Both input Vecs share one reservation made BEFORE either allocation:
Output `4C*4` + Scratch960KiB. All three outputs share a separate aggregate
reservation BEFORE convolution/SILU allocations: Output`5C*4` + Scratch960KiB.
At C6144, live retained output payload is221184B and total input+result
reservations2187264B; full raw parameter49152B plus actual offset staging is
separately admitted. Scratch conservatively covers raw FP32, SiLU FP32,
rounding/output temporaries and small control allocations. Existing fixed
three-node/attention workspace policy is unchanged.

HostValue clones/views retain original backing and source owner. Borrowed
outputs only; `into_vec()` refuses leased storage. Buffers drop before their
shared reservation; all source/stage/input/result reservations retire after
last owners. These are actual ResourceArbiter charges, not a full-host RSS or
free-RAM admission claim.

## Prepared checks (not executed)

- Core: three BF16 outputs/no effects; raw kind/rank/byte/source/history guards;
  valid-geometry strided refusal then explicit pack; unaware callback refusal.
- Artifacts: full/partial metadata; foreign issuer; raw geometry/dtype;
  corruption, pre-cancel, actual pressure, retained window/final0.
- Model: original roles/same recorded backend; accepted helper wiring;
  signed-zero history; exact aggregate output counts; live clones/no Vec escape;
  pre-cancel and deterministic after-import cancellation; full-pool pressure
  and input-fit/result-one-byte-short pressure; foreign source/provider/strides;
  existing full-HF session/forward refusal and final0.

````

## Source: .agents/runs/model-execution-hf-gdn-conv-root-v6-20261020/ROOT-QUALIFICATION.md

Original UTF-8 bytes: 5416; SHA256 `8a186c3ccb0ed0c7dee2631dea89156b6f5dafc39b2701230e17173f01d892c4`.
Resolve embedded relative paths against the original path, not this volume.

````text
# ROOT-only genuine qualification

Status: PREPARED / UNQUALIFIED. Latest additive freeze: `freeze-manifest-v2.json`;
initial `freeze-manifest.json`/`frozen-prepared/` retained unchanged. Author parsed/formatted source only. No raw
model/tensor payload was hashed or operator/reference/harness executed.
`bindings.json` copies expected source/capture hashes from genuine existing
receipts; author independently read/pinned only small source/metadata.

1. Independently review/apply NEW_DELTA + GPU_REFUSAL to the exact parent
   bytes; do not overwrite intervening work. Refresh modification times.
   Run focused CPU/core/artifact fixtures and relevant strict default/CPU/CUDA
   Clippy plus full isolated check. New tests are currently uncompiled.
2. ROOT runs `prepare_inputs.py --output <ownrun>/root-inputs-1` bounded under
   CPU4/finite wall and memory. It scans full original files with1MiB transfer
   buffers before/after, verifies original source and existing actual raw
   capture hashes, selects full original BF16 conv0 weights, observed QKV,
   actual raw K4 caches and captured activated output. Four cases:
   .8B/2B × teacher11(initial)/353(cached), source-36-F32-restored profile.
   Raw conv expectation is NOT fabricated or reconstructed here.
3. ROOT uses its reviewed owned-container lifecycle with the acquired pinned
   CPU image closure, networknone/no devices/12GiB/4CPUs, private nonce/name/CID,
   bounded logs/deadline and complete owned cleanup. Mount prepared own run
   read-only `/probe`, prepared input directory read-only `/inputs`, and a
   fresh private output parent writable `/results`. Inner argv:

   `python3 /probe/reference.py --inputs /inputs/inputs.json --inputs-sha256 ACTUAL_INPUTS_SHA --output /results/root-reference-1`

   Required environment: `ZG_CONV_CPUS=actual,four,allowed,CPUs`,
   `ATEN_CPU_CAPABILITY=avx2`, `OMP_NUM_THREADS=4`, `MKL_NUM_THREADS=4`,
   `OPENBLAS_NUM_THREADS=4`, CUDA/NVIDIA visibility disabled. Script sets/checks
   affinity before Torch load; pins Torch2.13+cu129/git, official modeling
   source, CPU AVX2/mkldnn/thread policy. ROOT binds actual image/DSO closure,
   invocation, script/input/output hashes and cleanup separately; a script
   report alone is not a runtime-closure receipt. No new standalone container
   lifecycle is introduced by this patch.

   Genuine public `F.conv1d` and `F.silu`, no model load or foreign token loop:
   initial actual padded4 → padding3 convolution → first4 view → SiLU → last
   teacher; cached actual incomingK4+new → conv → last1 view → SiLU. Preserve
   actual original slices/strides. Public updated K4 and activated bytes MUST
   equal the original unchanged-inference archives before any raw conv target
   is written. Public next3 must equal actual current cache's last3 bytes.
   Fresh raw conv targets are separate public-operator evidence, never a
   missing live trace. Any mismatch fails; never fit expectations or widen a
   tolerance. Original F32 learned36 profile remains unchanged; this region's
   actual conv weights are raw original BF16.
4. ROOT copies the genuine reference directory intact under this own run and
   pins actual input/reference reports, then builds `Cargo.toml` in its normal
   offline isolated coordinator path AFTER source merge. Cargo paths target
   MAIN; Rust helper calls only existing OriginalWindowOperators/HostMath.
   Run via `run_replay.py --binary ACTUAL_HELPER --binary-sha256 SHA --inputs
   ownrun/root-inputs-1/inputs.json --inputs-sha256 SHA --reference
   ownrun/root-reference-1/reference.json --reference-sha256 SHA --new-output
   ownrun/root-rust-1`.

   Unchanged accepted native leader cleanup is copied in `owned_lifetime.py`;
   ROOT must inspect/source-bind it and run its existing cleanup fixtures.
   Runner caps actual helper address space512MiB, four CPUs, finite900s per
   model/3300s total,4MiB log, exact executable FD and full source before/after.
   Both independent sources scan completely through VerifiedTensorSource and
   full320-text-role binding. Compare all6144 raw +6144 activated +18432 history
   words per call, signed bits including zeros:122880 words total across four
   calls. Actual caller input/prior bytes are also compared and retained.
5. ROOT independently compares saved actual raw BF16 outputs to public oracle
   targets and validates report/source/runtime/lifecycle bindings. Retained
   result views must match after original callers/source descriptors drop.
   Real pre-cancel, wrong-layer and complete host-pool pressure must leave the
   original exact ledger; synthetic deterministic after-import cancellation
   test is separate and is not relabeled as a trained mid-operator interrupt.
   Record actual per-class live input/result/scratch/staging reservations,
   final0, source equality and unchanged default full-HF guards.

Only exact selected private CPU operator evidence may pass. No full forward,
StateRoot/logits, cached GDN, initial chunk, source-streamed whole-model, GPU,
serving or performance eligibility follows.

ROOT V6 composes immutable V2 implementation with reviewed test-only V3 non-Drop cleanup and V4 actual74MiB/4reservation harness ledger. Independent actual public reference/inputs copied byte-for-byte from original owned runs; originals retained. V5 source-preparation path failure preserved, no MAIN change/build/runtime in that failure.

````

## Source: .agents/runs/model-execution-hf-gdn-conv-root-v6-20261020/TODO.md

Original UTF-8 bytes: 1211; SHA256 `9bf6a9f764ca651fcc616833730bcf99c0940f306d4bfd6e0ea0170aaec3d99d`.
Resolve embedded relative paths against the original path, not this volume.

````text
# GDN conv+SiLU shared-engine working checklist

- Parent: model-execution; owner baseline_review; ROOT sole build/test/runtime/merge.
- Scope: proposed source only under this run; no MAIN/shared-plan edits.
- [x] Read accepted closed conv/SiLU helpers and pinned source first/cached routes.
- [x] Inspect genuine .8B/2B two-profile metadata/header only; no tensor payload/hash.
- [x] Prepare typed default-refusing pure region + same HostMath/ForwardPlan/role binding.
- [x] Prepare sealed raw rank3 conv parameter stage, exact aggregate output/scratch leases.
- [x] Prepare sourcefree ownership/geometry/refusal/cancel/pressure/lifetime tests.
- [x] Prepare ROOT authentic selected-operand public oracle and replay, no expected reconstruction.
- [x] Freeze source-only proposal for ROOT review.
- [ ] Independent peer source review (Spine reprioritized; none claimed).
- [ ] ROOT focused/strict/full gates, genuine operator oracle and122880-word replay; no execution claimed.

Full HF forward/session/checkpoint/publication and GPU eligibility stay CLOSED.

Source-only checks: rustfmt18 + helper Rust syntax; Python AST6; whitespace18.
Prepared fixtures:11 Rust +2 Python. No fixture was executed by author.

````

