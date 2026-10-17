# TODO — model acquisition

Status: active
Provenance: 9239719
Scope: P03; prerequisite to new acquisition/loader claims.
Design: [README.md](README.md).
Contracts: [A2/A3 specification](../MODEL_ACQUISITION_IMPLEMENTATION_SPEC.md),
[pull/provider parity](../../research/MODEL_PULL_XDG_CACHE_AND_SOURCE_PARITY_2026-10-16.md).

## Committed

Completed phases through signed `fce1353` and their passed checks are in
[the completed checklist](archive/through-fce1353/TODO.md).

Completed phases from `e0073a5` through signed `a774ded` are in
[the next completed checklist](archive/through-a774ded/TODO.md).

### 837e77f — legacy offline acquisition eligibility

- [x] Reproduce five legacy offline policy failures before implementation using eight public-API tests: correct-hash LFS pointers passed repository/component eligibility; incomplete GGUF/index shards and reset index budgets passed component opening. All five failed at intended assertions, with three existing-semantics positives passing; generic CacheStore full verification legitimately accepted literal bytes.
- [x] Add one private managed-descriptor128B prefix helper using original pointer policy, cancellation checks and premature-EOF refusal. Legacy HF/GitHub offline API remains initial-selection/role-agnostic and keeps its public signature; generic cache inspect/verify remains literal-byte access. No second hash/cache/source issuer, new transport or realtime cancellation promise.
- [x] Reuse original pure split/index/sidecar closure only for explicit component Weights snapshots. Existing independent config/tokenizer suppliers, source/release coherence, roles and totals stay authoritative. Actual retained index reads share cumulative max_metadata_bytes/max_pages bounds across the whole component set; complete nested/split/release positives pass.
- [x] Focused production/diagnostic legacy8+8 and original API/lib/component/release/length/progress/transaction183 plus original CLI8+8 pass:215 actual executions/0 failures/2 intentional helpers ignored. All four strict artifact/CLI production/diagnostic configurations and scoped formatting pass. No unexpected compile/test/lint failure or post-GREEN source correction.
- [x] Full frozen isolated check.sh passes810 Rust executions/0 failures/68 ignored/174 suites plus18 namespace tests and complete lint/build/release/installer/comparator/resume gates. Original release cache_prepare passes both original routes in42.34s with both approved models/all248320 logits/all42 domains/source/session/raw-lease/mutation/reclamation criteria exact. Actual Nix source-integrity includes the new test; protected README, lock and dependency manifests remain unchanged. No new model/context/GPU qualification.
- [x] Integrate all findings/results and retain native Xet, conditional/cloud/provider/convenience and separately authorized filesystem work open. Move eight completed groups and ten checked gate records through a774ded verbatim into a linked Markdown TODO archive, keeping every unfinished item live. Retire the completed content-contract worktree after14 exact source comparisons and its two integrated runs plus two primitive runs; preserve active worktrees/runs. No evidence archive, download, live account/cache/device/privilege/mount action or publication.

### 52e42fc — native HF Xet reconstruction

- [x] Wire the same exact Xet primitives into explicit native HF discovery, scoped CAS and signed-range reconstruction. Preserve original immutable HF source/pin/inventory/selection/closure and the sole HTTP/executor/cache/publisher. At least one announced native LFS object is required; discovery is body-free/no-follow, ordinary Git sidecars retain their original route and announced native failures never substitute compatibility HTTP.
- [x] Add sealed native Hub/CAS/signed-range capabilities in the original HTTP owner. Keep transient64K scoped grants separate from original bearer limits; fixed CAS origin/repository/commit and expiry apply. Native metadata and signed payloads never follow redirects; exact raw signed URI and complete interval sets are retained. CAS/signed calls suppress original credentials, including same-origin cases. Ordinary sidecar redirects use the current wire URL and strip cross-origin auth.
- [x] Validate ordered/repeated/overlapping terms and exact full-file lengths/coverage, unique bounded maps, arbitrary-order complete multipart sets, codecs, terminal source EOF and complete keyed Xet file root before publication. Fetched edge/refetched chunks count decode/work but not logical progress/root; accepted-leaf replay compares hash/length. Original raw SHA/size/LFS/cache-hit/publisher remain independent, with no invented per-term or partial-xorb authentication.
- [x] Enforce invocation-wide actual attempts/observed body+control bytes/decoded chunks/logical leaves/refreshes and cumulative accepted term/group/range counts across files/reissues. Overflow charges at most one sentinel then forbids later reads/attempts; exact-limit EOF probes work. Scratch is admitted before allocation and held alongside the original transaction. Native CLI default256MiB covers both envelopes; explicit limits and ordinary128MiB remain binding/unchanged. No global RAM/kernel/TLS traffic or whole-operation deadline claim.
- [x] Add exact ephemeral object streams through the existing cache locks/admission/private stage/installer: no native HTTP-prefix recovery, validators or checkpoints. Actual child exit73 preserves the old ref and a fresh process starts a full native transfer; ordinary sidecars keep qualified durable resume. Root review caught a draft blanket-range restriction affecting ordinary sidecars; narrow allow-only-OrdinaryPayload fix and actual retry/range/budget/ref tests pass.
- [x] Add native CLI through the original shared fetch path, metadata/discovery-only dry-run and existing-cache offline branch before credentials/client/admission/signals. Offline source/selection/current bounds/pointer/full bytes pass without claiming native execution/freshness/access. Private-source/limit/miss/pointer/corruption and feature-disabled subprocess checks pass. Existing provider routes/defaults and dependency/lock bytes are unchanged.
- [x] Final artifact551pass/0finalfail/7intentional helper ignores across diagnostic188, original compatibility179 and production184; both artifact strict all-target configurations pass. Final CLI175pass/0fail/2helpersignored/56suites across production/diagnostic/default/three disabled profiles; both CLI strict configurations pass. Actual native HTTP, default128refusal/256success, complete weights+ordinary sidecars, codecs, reordered multipart, replay, expiry/reissue, cancellation, process exit and old-ref gates pass. Independent long128KiB keyed hash/tree vectors follow six official scalar-oracle checks; no expected result was fitted to Rust.
- [x] Record all repairs: initial E0624 test access failure (outer tee exit0 hid Cargo status) fixed with test-only factory; exactly7 unused pure helpers become test-only, without arithmetic changes. Initial executable154lib/1externalpass/19fail arose from payload.bin correctly triggering model closure; change synthetic auxiliary path and add a complete weights closure case. Next154lib/20externalpass/1fail was original sanitized State versus raw Limit in an ordinary-route assertion. CLI4pass/1fail was grammatical do/does help mismatch. All fixes are narrow, final identical cohorts pass, and no runtime tolerance/redaction/closure was relaxed. Independent generator copied65/66hex XOF prefixes before Rust; correct official32byte prefixes then verify all six official vectors before freezing custom goldens.
- [x] Complete final frozen isolated check.sh867 Rust executions/0fail/69ignored/181suites plus18 namespace tests and all lint/build/release/installer/comparator/resume gates. Original release cache_prepare2routes43.26s (44.87s release rebuild) pass both approved models/all248320 logits/all42 domains/source/session/raw-lease/mutation/reclamation exact. Actual offline Nix includes all9new Rust files;28 current local links, protected README and unchanged dependency/lock manifests pass. This adds no model/context/GPU qualification or live-provider throughput claim.
- [x] Integrate implementation, reviews, numerical/protocol findings, failures/results and all pending work atomically for signed local acceptance. Keep conditional cloud, selected ModelScope sizing, metadata candidate discovery and the wider provider/loader/runtime frontier open. Independent cold reviews find no conditional/sizing blocker but their execution remains unrun; do not promote alias echoes to immutable commits. No model/SDK/provider payload/account/cache/device/privilege/mount action, push, duplicate runtime or evidence archive.

### 46cf5a5 — strong conditional cloud observations

- [x] Add explicit object-only S3/Azure conditional requests and pure pins with separate s3-conditional/azure-conditional source contracts. Bind normalized endpoint/profile/locator/path/role/exact bytes/strong ETag; final original raw SHA stays FirstSeen. HEAD is admitted before transport, body-free/exact200 with complete identity metadata and original provider guards. No caller SHA, fabricated remote version, implicit fallback, prefix or cap profile.
- [x] Add one private typed If-Match condition to the existing opaque HttpTarget, None in all six original/native constructors. Every initial/range/retry emits one exact condition; original validator checks ETag before body. A mismatched resume validator refuses before signing/wire. SigV4 and Azure standard SharedKey slot match independent HMACs; HTTP412 refuses without refreshed HEAD or downgrade. Preserve all native Xet isolation/raw-target/ordinary-resume/counter tests and old cloud defaults.
- [x] Reuse the original one-entry managed verifier with private ExpectedObject/ExpectedDigest::TrustedSha256|FirstSeen, preserving HTTPS ref behavior and native digest-only pins. Exact source/path/role/bytes/provenance/current limits/full hash/pointer/cancel checks remain. No dummy expected SHA, second cache/hash/runtime/publisher or public source issuer.
- [x] Add --conditional-read to existing S3/Azure object handlers and exact offline --bytes/--etag/snapshot inputs. Online dry-run performs HEAD metadata only; offline finishes before env/credential file/FIFO/stdin/client/admission/signals. Escaped JSON ETag receipts preserve declared observation without asserting remote freshness/access. Explicit version/caller-SHA conflicts refuse; original auth/SAS scope and shared fetch/publisher remain unchanged.
- [x] Root and independent review find no protocol blocker; three literal HMAC vectors independently match Python stdlib before Rust gates. Reconcile conditional delta onto signed native52e42fc, preserving all original/native target code/tests; eighteen unowned baseline paths stay exact and only eleven approved source paths merge. A scoped SIDE formatting import-order failure was corrected before freeze; no MAIN compile/test/lint/source repair occurs.
- [x] Final focused872passes/0fail/16intentional ignores:714artifact (443diagnostic/271production),151CLI (75production/76diagnostic),4default-diagnostic and3disabled-profile tests. New conditional API10diagnostic/8production, all wire/signature/HEAD/resume/ETag-before-body/412/old-ref/offline tests pass. All four strict artifact/CLI configurations pass; original provider/native/legacy/cache/process/fault regressions stay green.
- [x] Complete final frozen isolated check.sh886 Rust executions/0fail/69ignored/188suites plus18 namespace and full lint/build/release/installer/comparator/resume gates. Actual Nix includes both new tests and exact Azure fixture;28 local links/protected README/unchanged lock/dependencies pass. Separate exact ignored release cache_prepare routes pass1test17.40s and1test23.69s (sum of test durations41.09s, not one invocation/wall time), both approved models/all248320 logits/all42 domains/source/session/raw-lease/mutation/reclamation exact. No new model/context/GPU or live-cloud qualification.
- [x] Integrate all implementation/review/results and remaining provider work atomically for signed local acceptance. Retire two completed native worktrees after26 exact native source matches and CLI9current+2exact superseded-baseline matches; remove five integrated native runs. Preserve active conditional/ModelScope/candidate/quant work. No new dependency, download, account/cache/device/privilege/mount action, publication or evidence archive.

### 4db5efb — ModelScope selected exact sizing

- [x] Recover absent selected ModelScope file/index Size with one admitted body-free exact200 HEAD at the same fullcommit Direct target. Cache the exact length once before index GET/final planning, retain native raw SHA/provenance and source identity. Unselected missing sizes do not probe; malformed present/null blob sizes still refuse, with existing directory-null behavior preserved. Update the old malformed fixture to explicit null and clarify CLI sizing help.
- [x] Extend only the original provider metadata Budget for HEAD requests and returned header controls, shared with inventory/index bytes and logical call limits. Exact identity Content-Length, file/snapshot limits and no-follow status/framing rules apply. Strong ETag is checked before index/payload bytes; weak/absent validators grant no resume authority. Shared HTTP, native Xet, cloud profiles, cache, runtime, public APIs and dependencies remain unchanged.
- [x] Root proposal/source review and independent five-path review find no policy blocker. Native official-token GETtree/HEAD/GET and no-follow wire tests pass; selected/index size caching, unknown/unselected/zero/oversize/malformed fields, cancellation/budgets/ETags/rawSHA/old-ref/closure/source identity cases pass. Alias/CDN authority stays explicitly open; synthetic metadata grants no real provider or loader qualification.
- [x] Initial diagnostic all-target command exits101:156library tests and earlier providers pass, new sizing1pass/9fail. The old missing-size error is absent from live source but present in retained non-test rlib/executable; preserved SIDE mtimes predate previous builds. Refresh ONLY5merged-file timestamps after exact byte equality, then identical rerun453pass/0fail/13helpersignored confirms causal stale build. No Rust, fixture expectation, protocol or tolerance repair. Future copies use fresh modification times; record this Cargo constraint in AGENTS.md.
- [x] Final focused891passes/0fail/16intentional ignores:733artifact (453diagnostic/280production),151CLI,4default-diagnostic and3disabled. New sizing10diagnostic/9production pass and original provider/native/conditional/legacy/cache/process/resume/CLI tests stay green. All four strict artifact/CLI configurations pass with no lint/source correction.
- [x] Complete final frozen isolated check.sh896 Rust executions/0fail/69ignored/191suites plus18 namespace and complete lint/build/release/installer/comparator/resume gates. Actual Nix includes the new sizing test;28 current local links/protected README/unchanged lock/dependencies pass. Separate exact ignored original release routes pass17.64s and23.60s, both approved models/all248320 logits/all42 domains/source/session/raw-lease/mutation/reclamation exact. No additional model/context/GPU qualification.
- [x] Integrate findings/failure/results and every unfinished task atomically for signed acceptance. Retire completed conditionalSIDE after29exact source/baseline matches to46cf5a5 and its three integrated runs; preserve active sizing/candidate/quant/filter work. No model/SDK/provider payload/account/cache/device/privilege/mount action, push, new dependency/runtime or evidence archive.

### f22f759 — metadata-only pinned file candidates

- [x] Add bounded metadata-only HF/GitHub candidates API/CLI through the original pinned-tree owner. List all regular choices without selecting a family, probing lengths, reading indexes/payloads or publishing. Full inventory collision/control validation precedes subfolder projection; metadata admission and logical-call/body budgets remain cumulative.
- [x] Expose honest pinned-tree versus verified cached-subset coverage, optional sizes and filename-role hints. JSON labels stored-Git versus HF tree/LFS versus verified-content lengths; Git OIDs and cached FirstSeen hashes never become trusted expected SHA. Offline source/current bounds/full hashes/pointers precede credentials/client/admission/signals; original Files/Release semantics remain.
- [x] Independent candidate review catches a regular-ancestor collision hidden by subfolder projection; fix complete-inventory ordering and add HF/GH negatives plus real-directory positive before execution. No executed RED claim. Final943focused passes/0fail/16helpers and four strict configurations pass without a compile/assertion/lint/source correction.
- [x] Frozen candidate full isolated `CARGO_BUILD_JOBS=2 RAYON_NUM_THREADS=4 python3 scripts/native-gpu.py -- bash scripts/check.sh` passes915Rust/0fail/69ignored/198suites plus18namespace/all normal quality gates. Two separate exact release cache_prepare calls pass: approved_cached_models_match_direct_source_exactly17.47s and approved_hf_import_models_match_direct_source_exactly23.65s (`--release --locked --offline -p zerogravity --features cache --test cache_prepare NAME -- --ignored --exact --nocapture`, same wrapper). Both approved.8B/2B/all248320logits/all42domains/two tokens/session/rawlease/mutation/reclamation remain exact. These are individual test durations, not one combined invocation.
- [x] Integrate implementation/reviews/results and every unfinished task for signed candidate acceptance. Keep explicit quant, full filters/excludes, sidecar/preset/auth/provider/runtime frontier open. Preserve root README and dependencies; no model/SDK/provider payload/account/livecache/device/privilege/mount action, push or new runtime. Native source and cache bytes grant no additional model execution qualification.
- [x] Actual candidate source-integrity Nix passes with four new files included; all9source paths match frozen SIDE,26local documentation links/root README/dependencies pass. Retire completed ModelScope sizing SIDE after5exact signed4db5efb/MAIN matches and no process use; integrate future finite CDN authority/credential/refresh boundaries before deleting four completed sizing/review/frontier runs.

### f197c9c — explicit GGUF filename quant families

- [x] Add private-field GgufQuantRequest and explicit HF/GitHub/native filename-quant selection within the original retained provider tree and closure. Keep public Selection/PullRequest shape, original source identity, mandatory sidecars, same admission/logical counter and sole shared fetch/publisher. No default/rank/fallback or tensor eligibility follows.
- [x] Reuse strict canonical split decomposition, ASCII1..32letter-led tag grammar and terminal basename boundaries. Every matching family must be complete before uniqueness; malformed/incomplete/conflicting-total/case/single+split ambiguities refuse. Independent review's directory -of-/tag counterexample is repaired with basename-only inspection, preserving full path family identity; regression cases drafted before execution, no RED claim.
- [x] New quant offline requires explicit snapshot digest and verifies original source/current bounds/managed hashes/pointers/complete selected closure plus Weights/Config/Tokenizer roles before credentials/client/admission/signals. Report retained-subset scope, keep other retained data and original Files offline semantics. New native quant fixture is announcement/plan qualification only, not a CAS execution or inference test.
- [x] Quant focused975pass/0fail/16intentionalhelpers and all4strictartifact/CLI production/diagnostic configurations pass without compile/assertion/lint/source repair. New11API/5enabledCLI/disabled profiles and all original provider/component/native/legacy/cache/resume cohorts pass. After interruption, all11MAINsource paths still equal frozen draft; the sole existing Cargo owner continued.
- [x] Frozen full isolated `CARGO_BUILD_JOBS=2 RAYON_NUM_THREADS=4 python3 scripts/native-gpu.py -- bash scripts/check.sh` passes934Rust/0fail/69ignored/205suites plus18namespace, all formatting/workspace/dependency-isolation/lint/build/release/installer/comparator/resume gates. Source remains unchanged. No new provider/model/SDK download; installer fetches are original owned local fixtures.
- [x] Both original exact release cache_prepare routes pass separately under jobs2/Rayon4/native isolation: approved_cached_models_match_direct_source_exactly17.60s; approved_hf_import_models_match_direct_source_exactly23.70s (`cargo test --release --locked --offline -p zerogravity --features cache --test cache_prepare NAME -- --ignored --exact --nocapture`). Each verifies both approved.8B/2B/full248320logits/all42domains/two tokens/sessions/rawsource lease/mutation/reclamation exact; individual reported test durations, not one aggregate call.
- [x] Wrap currentphase after explicit user pause: revalidate HEAD/worktrees/all11MAINfrozen sourcepaths and all15C3baselinepaths/C4exact3draftdeltas, originalrootREADME/deps and26local links; actualNix passes after3newpaths included and again after interruption. Both draft owners stopped/nochildren, full design/pending tests preserved in namedSIDEs. Quant validator returned exclusive Cargo after allcommands/children exit0. Integrate acceptance and all unfinishedwork for signed local commit, then pause; no push/download/model/GPU support change. Earlier C1 rolling-doc script expected7checkeditems but correctly stopped before mutation on actual6; inspected/corrected count and rolled exactly6, no implementation or test change.
- [x] Actual quant source-integrity passes after3newfiles add-N; all11source paths match frozen draft,26local links/root README/allCargo manifests+lock unchanged. Root approves C3 isolated source draft with fixed bounded fnmatchcase/exclusion/closure semantics and C4 native-index isolated source draft with explicit apply opt-in/default no-CAS and all-counter continuation; neither draft is execution evidence.

### 1a22597 — completed comprehensive handoff and source preservation

- [x] At the owner's documentation-only request, create the [comprehensive17Octoberhandoff](../../docs/FULL_COMPREHENSIVE_AGENT_HANDOFF_2026-10-17.md) and [copyabletakeoverprompt](../NEXT_AGENT_PROMPT_2026-10-17.md), preserving runtimepause. Consolidate all U056–070 decisions, V1–V5.1/currentcontracts,77reachable commits, all121openplanitems, actualprofile/test/failure limits and complete document/research catalogs. Independentreview recovers an older94-source U040/G3b archive nested at7e61cb6; compare all94ranges and13+3 outersections exactly, retain explicit original33graph/later129/191 gaps. Preserve two byte-exact portable sourcepatches/fourunfinishedfiles plus pausedcontexts/inertoracle underdocs; actualapply and independentin-memory reconstruction pass. Links/coverage/README/appendonlydecisions/runtime/deps/index/pausedSIDEs remain intact. No runtime/model/build/GPU execution or publication; retire only completed documentationruns after integration.

- [x] After signedf197c9c acceptance, retire completed quantSIDE after15exact source/baseline matches toHEAD/no process use and exact cold-design copy in preserved indexSIDE; remove four integrated completed quant runs. Leave interrupted filters MAIN/SIDE runs and both unfinished source worktrees. This rolling TODO update belongs to the next commit; no new implementation or testing remains active.

### 04363c3 — bounded C3 Unicode filters and exclusion closure

- [x] C3 full-path filters resumed and source-qualified after the pause:
  exactly 11 intended Rust paths (3 new, 8 changed), fresh copy mtimes,
  one-pin HF/GitHub/native-Xet filtering, required-index-before-GET/shard/
  config/tokenizer veto, optional omission, exact retained offline closure,
  old Files/quant policy and shared original cache/admission/publisher.
  Independent CPython3.14 literal73 +33345-pair frozen digest, 4 filter
  matcher/adversarial units, 11 API integration tests (including actual
  private cache/ref publication on both providers), 4 enabled CLI tests
  and disabled-feature refusal pass. The original C2 candidate/quant suites
  and real private-loopback HTTP/TLS fixtures continue to pass.

- [x] Frozen C3 complete isolated check.sh: 956 Rust executions/0fail/
  69ignored/212 suites, plus 18 namespace fixtures, strict artifact/CLI
  production+diagnostic lint, workspace/dependency isolation, CPU build,
  installer, shell/Nix formatting and reference/comparator/resume self-tests.
  The first long run was interrupted during release compilation by a tool
  runtime failure (not an implementation test failure); the next frozen
  full run exited0 after the source-equal release build completed.

- [x] Actual Nix filtered-source integrity succeeds with all 3 new Rust
  paths included; checked current Markdown links, unchanged root README,
  Cargo.lock/manifests and no unrelated source differences. Two fresh
  separately invoked isolated original release cache_prepare tests pass:
  direct/cache 16.17s and HF-import 21.64s; each validates both approved
  Qwen3.5 checkpoints/all248320 logits/all42 state domains/2 tokens/sessions/
  original rawlease, mutation and removal behavior bit-for-bit. No new
  model, GPU, context or performance qualification is inferred.

- [x] Record the October17 user pause as a historical handoff only:
  C2 was signed at f197c9c, two preserved source drafts survived, and the
  explicit October18 resume moved C3 onto MAIN without losing C4.

- [x] Recover and complete the C3 paused source from
  /home/pvl/spaces/abird/src/zerogravity-filters-work. At takeover only
  filters.rs existed and was unwired/uncompiled; its frozen primary vectors
  and source/ownership policy are now implemented with API/CLI/native/closure
  regressions. Preserve the original paused source/run for traceability until
  reviewed retirement; no new SDK/model/provider/account payload fetched.

### 8195c18 — verified required native Xet indexes

- [x] C4 required native Xet indexes recovered from the preserved
  /home/pvl/spaces/abird/src/zerogravity-xet-index-work source-only draft.
  One isolated C3-base worktree performed a strict three-way merge of the
  three C4 production source deltas; exactly two HF/Xet dispatch conflicts
  were resolved without losing C3 selection/required-index veto. A cold
  typed IndexReader replaces the draft's uncompileable higher-ranked FnMut
  closure; original HF metadata parser, native Access/reader and publisher
  stay authoritative. Only seven reviewed Rust paths differ from C3 MAIN
  (five modified and two new integration tests). The original paused SIDE
  and its portable patch/design context remain untouched.
- [x] Preserve default library resolution and Xet CLI dry-run as body-free:
  required native LFS indexes refuse before CAS grants or signed body.
  Only CLI APPLY clones native options with explicit
  with_index_reconstruction; no new public flag, network client, cache,
  source issuer, device/runtime or parser. Original same-commit pinned
  index gets exact length+trusted LFS rawSHA, full keyed Xet root/EOF,
  raw SHA and LFS-pointer verification before JSON parse/shard closure.
  Planning uses one original admitted scratch; publication reconstructs
  and verifies the same index again in the existing cache transaction.
- [x] Carry ALL Xet consumed attempts, observed wire/control bytes, decoded
  chunks, logical leaves, refreshes and accepted terms/groups/ranges as a
  validated Result seed across resolution, publication and later weights;
  poisoned/tightened owners never silently zero. Native default256MiB
  admits index planning; explicit128MiB denies before signed index data.
  Ten new real original-owner API tests include independent abc Xet golden,
  exact index→shard→cache/ref publication, C3 native filtered positive
  and excluded-index-before-GET negative, wrong SHA/root/JSON/truncated
  signed body/cancel/old-ref, missing shard and aggregate budget refusal.
  Two private budget tests, two enabled CLI and one default-disabled CLI
  test also pass. Existing native Xet, C1 candidates, C2 quant, C3 filters,
  HTTP/TLS, offline and cache regressions remain green.
- [x] Final frozen isolated C4 check.sh exits0: 973 Rust executions,
  0 failures, 69 ignored, 219 suite runs, plus18 native namespace fixtures
  and full CPU/dependency/strict-lint/build/release/installer/Nix-format/
  comparator/resume gates. All four strict artifact/CLI production/diagnostic
  configurations pass after one test-only initializer style correction.
  An early 4/5 fixture failure selected an absent shard before reaching
  the required index; corrected fixture to select existing index. A
  separate truncated-body fixture initially ran out of scripted retries;
  bound its original request budget so refusal occurs before unscripted
  transport. Both corrected focused suites and final gate pass; no
  source arithmetic/protocol/tolerance relaxation or claimed RED test.
- [x] Actual final offline Nix filtered-source check includes both new Rust
  tests. Current local document links, original README, Cargo.lock and
  all dependency manifests are unchanged. Two separate current accepted
  Qwen3.5 release cache_prepare routes pass from the real cargo command:
  cache/direct16.08s and HF-import21.67s; each covers both original checkpoints,
  full248320 logits, all42 state domains, two tokens, independent sessions,
  source leases, mutation and reclamation exact. No new model/GPU/context,
  live provider/account/SDK/data download, privilege/mount/device operation,
  push or publication is implied.

### f56b574 — trusted AWS-standard FULL_OBJECT SHA256

- [x] C5 AWS-standard FULL_OBJECT SHA256 checksum admission:
  one six-file Rust slice with explicit --aws-full-object-sha256 and
  library builder. Only canonical AWS commercial-region immutable-version
  object HEAD requests x-amz-checksum-mode: ENABLED; original SigV4 signer
  includes the exact header, anonymous public profile sends it literally.
  HEAD redirects and absent/COMPOSITE/malformed/untyped checksum data refuse;
  only canonical base64 raw32 SHA256 from FULL_OBJECT plus matching pinned
  version/length becomes TrustedSha256. Independently supplied caller SHA
  must agree. No compatible endpoint, conditional/content mode, prefix or
  other algorithm gains remote SHA authority or fallback.
- [x] Preserve source/byte/provenance and original single publisher;
  dry-run remains metadata-only, offline requires the observed --sha256,
  exact version/length/snapshot before credentials. Seven dedicated
  original executor/cache API fixtures (exact AWS checksum, trusted vs
  FirstSeen, incompatible scope, malformed/composite/missing SHA, offline
  source, wrong GET/ref rollback, admission/cancellation) and one S3 CLI
  extended enabled test pass. One independent frozen Python-stdlib SigV4
  literal golden and two actual signed/unsigned private-loopback HTTP HEAD
  /redirect tests pass; real S3/C3/C4 regression suites remain green.
- [x] Final frozen C5 source/acceptance checks: 984 Rust executions /
  0 failures / 69 intentionally ignored / 222 suite executions in
  isolated check.sh, plus18 Python namespace fixtures, complete CPU,
  strict artifact/CLI production+diagnostic Clippy, release/installer,
  shell/Nix formatting and reference self-tests. Source-integrity passed
  using git add -N for new S3 API suite; local Markdown links, root README,
  Cargo.lock/manifests unchanged. Two independently invoked original
  release cache_prepare model routes pass: cache/direct16.03s and HF
  import21.45s, each both approved Qwen3.5 checkpoints/full248320 logits/
  all42domains/two tokens/independent sessions/source leases/mutation/
  reclaim exact. No inference/multimodal/GPU performance qualification
  is implied. Acceptance requires a signed local exact-path commit;
  no push or external provider payload testing is implied.

### d5aa5cb — original-model v1 source closure and execution cutover

- [x] Model-pull v1 execution cutover: close the finite companion gap
  for original index-first Qwen3.8 and Gemma4-like source layouts,
  using one shared bounded optional same-folder companion list in original
  HF and shared GH/cache closure. Known chat template, processor/image/audio/
  video metadata and alternate tokenizer files remain opaque source bytes;
  no arbitrary directory recursion, inferred model variant or auto-loader.
  Source-only two-provider synthetic >51GB indexed shard tests confirm
  no weight GET or cache publication when disk admission is insufficient,
  plus explicit GGUF/companion filtering/source closure and CLI index-first
  large-disk-budget argument validation. Keep 4GiB executable-source gate,
  original model numerical/CPU/GPU and authorization tasks in model-execution.
- [x] Final frozen model-pull v1 source/transfer gate: 988 Rust executions,
  0 failed, 69 intentionally ignored, 222 suite executions under exact
  original scripts/check.sh with private user/network namespace and guarded
  loopback. Strict artifact and CLI production+diagnostic lint, CPU workspace,
  release build/installer, Nix formatting and 18 Python namespace fixtures
  pass. Actual offline Nix filtered-source-integrity passes; root README,
  Cargo.lock/manifests unchanged, current local plan links checked.
  Independently reran original approved release cache_prepare sources:
  cache/direct16.38s and HF-import21.88s (both 0.8B+2B,
  all248320 logits/all42domains/2 tokens/source lease/mutation/reclaim exact).
  New source-closure tests cover original HF/GH GGUF+12 known metadata
  companions, filtered optional omissions, strict exact offline subsets
  and synthetic two-shard >51.6GB index plan refusing insufficient disk
  before any weight GET; no actual weight/model/SDK download and no license,
  trained-execution, GPU or general-provider compatibility qualification.
  Four bounded Rust files and four plan files only; remaining separately
  qualified provider/mirror/gating/filesystem/SSD throughput work stays
  explicitly unchecked without obstructing the model-execution CPU-first plan.

## Uncommitted

- [x] Complete the observed public Qwen3.8+Gemma4 source metadata
  filenames in original pinned same-folder closure. 2026-10-18 actual
  tokenless HF --dry-run returned Qwen3.8-27B commit
  1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0,
  18 shards/55,586,113,293 selected bytes; Gemma4 26B-A4B-it commit
  4d7ae4984b7db7de8f8457170b3f1a419ee76d52,
  2 shards/51,644,338,194 selected bytes. A third tokenless
  HF --dry-run resolves Gemma4 E2B-it commit
  3e22461f65e89153144f8adb70e3b8c2cc9845a7,
  1 unsharded safetensors/10,278,848,001 selected bytes. The actual
  observed Qwen video_preprocessor_config.json and crc32.txt (opaque
  non-trust data) required two additive optional companion filenames.
  Both present in real result; no credentials, tensor weights, cache
  refs, SDK, model execution or GPU touched. Synthetic HF/GH model/quant/filter and
  offline exact closure tests cover both additional filenames.
  Final exact source cutover gates pass: isolated check.sh 988 Rust
  executions/0fail/69ignored/222 suites plus18 namespace fixtures and full
  strict production/diagnostic lint, CPU/release/installer/Nix-format/
  reference self-tests. Actual offline Nix source-integrity and local
  docs links pass; unchanged root README, Cargo.lock/dependency manifests.
  Separate original release source/cached model regressions both pass:
  cache/direct16.88s and HF-import21.72s, each both approved Qwen3.5
  checkpoints/all248320 logits/all42 state domains/two nonzero tokens/
  independent sessions/lease/mutation/reclaim bit-exact. This remains
  tokenless live metadata-only Hub resolution; there is no trained-source
  byte acquisition, inference, 8K drift closure, model-format or physical
  GPU qualification. Sign only the two additive Rust paths and relevant
  plan documentation; no push, tag or user-cache/device mutation.

- [ ] Extend cloud profiles beyond the narrow qualified AWS FULL_OBJECT SHA256 object route: broader S3 partitions/compatible vendors/credentials/checksum algorithms, Azure latest/SAS/managed chains and per-object digest-map prefixes. Keep explicit version, caller-SHA and conditional FirstSeen contracts distinct; never infer immutable versions or trusted raw hashes from unqualified validators or multipart COMPOSITE checksums.
- [ ] Extend ModelScope only through separately qualified alias authority and redirect/CDN/mirror/credential profiles. Selected missing sizes now use exact pinned HEAD; no alias-to-commit authority was established from revision echoes, timestamps or file last-modified IDs. The current profile does not close broader parity or independently prove server revision/membership behavior.
- [ ] Measure and qualify native Xet fetch coalescing/chunk caching and any persistent native-prefix protocol. The current bounded per-intersection complete-signed-set reader has counted overfetch/refetch and no performance claim; native process interruption restarts a full transfer. Keep per-term/partial-xorb proof limits, fixed CAS scope and existing elapsed-time/HTTP-trailer boundaries explicit.
- [ ] Close remaining finite convenience/compatibility gaps: required sidecar/adapter/sparse-quant policy, bounded auxiliary/presets, mirror/endpoint/env precedence and gate/license diagnostics. Preserve explicit source authority/no silent numeric fallback and separate loader/runtime plans.

- [ ] Qualify any future v1 migration only after separate authorization and proven reader/process quiescence; current removal supports originally protected roots only.
- [ ] Extend cleanup only through separately qualified exact ownership/recovery protocols for dead retained staging/general orphans and any later eviction policy. Current exact-snapshot operation deliberately retains them and every stable lock inode.
- [ ] Any additional lazy/raw-path/mmap consumer must adopt managed descriptor/open ownership or remain externally quiescent during removal. Observed named cache charge is not a process-global final-close/physical-free-space attestation.
- [ ] Qualify additional filesystem pairs before extending the explicit allowlist. Different st_dev values alone do not prove independent backing objects for Btrfs subvolumes, overlay/FUSE or virtualized views; retain branch-specific root device/mount checks and complete inode/ancestry exclusion. No unsupported pair gains acceptance from generic mocks.
- [ ] Separately authorize any needed namespace-only hidden-bind/subvolume/overlay filesystem fixtures before mount operations. Real target-root/unselected inode aliases and hidden bind aliases in both directions remain NOT RUN; no host mounts, block devices, new privileges or live cache actions are implied. Current tests use already existing distinct filesystems only.
- [ ] Independently qualify any broader local layouts, renamed/unnamespaced roots, missing mount identities or new destination-root creation before exposing support. Keep explicit selection, retained descriptors, bounded topology work and existing cancellation/rollback limits; real import still needs user-selected source/repository/commit/files and copying authorization.
- [ ] Finish the finite provider/loader parity ledger beyond qualified HF/GitHub/HTTPS/OCI/GCS/versioned-S3/Azure/ModelScope and explicit local-HF import slices: remaining external-cache layouts/filesystems, additional model hubs, unversioned/broader S3/Azure/compatible profiles, native Xet and broader OCI profiles. Opaque serialization/plugin loaders remain separate runtime work.
- [ ] Continue explicit source/license authorization and trusted-digest versus first-seen provenance through additional providers/loaders; acquisition grants no license or loading support.

## Tests and verification

- [ ] ACQ-A12: complete the finite provider/loader parity inventory and independently qualify remaining supported paths.

- [x] Legacy offline tests-first5RED/3positive, then215 focused passes/0 failures/2 helpers ignored and all4 strict configurations pass. Full frozen isolated check.sh810pass/0fail/68ignored/174suites and original cache_prepare2routes42.34s pass exact original criteria; actual source-integrity/current links/protected README/unchanged dependencies pass. No unexpected repair.
- [x] Native Xet final artifact551+CLI175 passes and all4 strict configs, documented test-access/fixture/grammar fixes only; full frozen isolated check.sh867pass/0fail/69ignored/181suites, actual source-integrity/28links/README/unchanged dependencies and original cache_prepare2routes43.26s pass exact original criteria. Native index closure/coalescing/persistence remain open.
- [x] Conditional cloud final872focused passes/all4strict/no MAIN repairs; full isolated886pass/0fail/69ignored/188suites, actual source-integrity/28links/README/unchanged dependencies and two original exact filtered routes17.40s+23.69s pass. Broader cloud authority/auth/prefix profiles remain open.
- [x] Selected ModelScope sizing891focused/all4strict and full896pass/0fail/69ignored/191suites pass; actual Nix/28links/README/unchanged dependencies and original exact routes17.64s+23.60s pass. Initial stale-library101 is resolved by byte-equal timestamp refresh, with no source/policy repair.
- [x] Candidate final943focused/all4strict and full915pass/0fail/69ignored/198suites+18namespace pass without repairs; actualNix/26local links/README/dependencies and original exact cache/direct17.47s+HF-import23.65s pass. Coverage remains metadata discovery/verified retained subset.
- [x] Quant final975focused/all4strict and full934pass/0fail/69ignored/205suites+18namespace pass without repairs; actualNix/26local links/rootREADME/dependencies and original exact cache/direct17.60s+HF-import23.70s pass. Interruption revalidation found no source drift; user requested pause and unfinished C3/C4 drafts remain preserved.
- [x] C4 exact native LFS-index all-counter/source/authorization validation,
  original HF/GH/Xet/C1/C2/C3 plus CLI/default-offline regressions,
  all four strict feature/lint configurations and final frozen source-integrity
  are qualified above. Full final check.sh 973/0fail/69ignored/219suites and
  two original release cache_prepare routes16.08s/21.67s pass separately.
- [ ] Each remaining provider phase requires focused API/CLI/protocol/auth/version/range/offline/fault gates, full frozen isolated check.sh, relevant source validation, actual source-integrity and documentation/dependency/root README checks before signing.

## Deferred

- [ ] Unsupported plans fail closed until qualified: unknown-length required/model-closure files, capped resumable transfers, unqualified external-cache layouts/filesystems and remaining cloud/OCI profiles, legacy-root removal and unmanaged-reader concurrent reclamation remain explicit above. Independent full HTTP trailer validation is unqualified under the documented decoder boundary, rather than a reason these payloads universally refuse. Independent component data grants no additional loader eligibility.
- [ ] Linux local-filesystem semantics are qualified first. HTTP cancellation obeys configured whole-request bounds; old retained attempts consume real quota and arbitrary same-UID mutation remains outside descriptor isolation.
