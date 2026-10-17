# TODO — model acquisition

Status: active
Provenance: 9239719
Scope: P03; prerequisite to new acquisition/loader claims.
Design: [README.md](README.md).
Contracts: [A2/A3 specification](../MODEL_ACQUISITION_IMPLEMENTATION_SPEC.md),
[pull/provider parity](../../research/MODEL_PULL_XDG_CACHE_AND_SOURCE_PARITY_2026-10-16.md).

## Committed

### 3787519

- [x] Local Linux cache A1 accepted. Cooperative quotas are not global host/UMA reservation; descriptors do not prevent arbitrary same-UID mutation.

- [x] Actual A1 full gate:194 Rust executions passed,27 optional executions ignored; strict lint/release/installer/isolation checks passed. No inference/HTTP claim follows from cache publication.
- [x] Publication fault/recovery:18 operation types,38 callbacks,76 injected I/O and76 real child-exit fixtures; retained refs/leases and concurrent quota rebuild regressions passed.
- [x] Review fixes: retained named ref backing, private snapshot reservation accounting and hardlink-only ctime verification; deterministic regressions passed. Early readiness-marker and cfg-import harness issues were fixed before acceptance.

Technical API/bounds: [A1 cache reference](../../research/PULL_A1_LOCAL_XDG_CACHE_2026-10-16.md).

### fa6affa — bounded A2 acquisition

- [x] Add one optional synchronous verified HTTPS owner; bounded metadata/headers, redirects, origin-scoped credentials, identity encoding, retry/backoff and whole-request deadlines. Default CPU/core/inspection retain no HTTP/TLS/GPU dependency.
- [x] Extend the existing A1 attempt for exact-size unknown raw digests: one-pass admitted staging/hash, private non-publishable planning keys, actual inode accounting and persisted TrustedSha256/FirstSeen; legacy entries remain Unspecified.
- [x] Pin HF revision/tree and GitHub commit/tree or release/asset IDs; bound aggregate metadata, pages and filter work. Git object IDs never count as raw SHA256.
- [x] Close selected/adjacent safetensors indexes and shards, ordered GGUF splits and same-repository config/tokenizer/available sidecars. Missing closure, mixed weight families, changed exact pins and LFS pointer bodies refuse.
- [x] Add explicit optional facade/CLI pull, metadata-only dry-run, offline, list, inspect and verify. Offline branches precede credentials/client/provider; cache data grants no SourceIdentity or inference readiness.
- [x] Keep metadata and streaming work on the caller's existing ResourceArbiter TransferAdmission. Narrow production provider resolvers to the admitted entry point; prove denied/canceled resolution never requests metadata.
- [x] Validate actual loopback HTTPS chain/hostname/downgrade refusal, origin auth stripping, framing/limits, deadlines/cancellation, bounded retries and exact strong-ETag range append. Ignored200 or changed validators preserve prior refs.
- [x] Persist checksum trust strength, preserve exact original bytes and all A1 durability/lease/accounting limits; retain the concurrent user-authored6deebf2 XDG root abird/zerogravity/models.
- [x] Integrate independent cache/provider/transport/CLI reviews. Correct uppercase-pin checks, adjacent-index and weight-family omissions, public admission bypass and staged-index error redaction; regression tests cover refusal and exact old-ref preservation.
- [x] Correct ureq whole-body timeout interpretation and TRACE URL/Location disclosure. Header and total deadlines are distinct; acquisition-only log/max_level_debug caps process-wide log TRACE, with the downstream feature constraint documented.
- [x] Complete frozen-source gate:312 Rust executions passed,0 failed,30 optional/fixture entries ignored (counts include repeated feature builds); strict lint, production-only/diagnostic/combined features, CPU release/installer,18 namespace Python tests and comparator/resume self-tests passed.
- [x] Actual offline Nix source-integrity build passed, including exact synthetic TLS fixture retention;31 changed-document links, single tracked Cargo.lock and byte-identical user root README passed. Only four already-cached URL dependency packages were newly locked; no model/SDK/provider acquisition or device action.
- [x] Record incremental fixes: Read/Write by_ref ambiguity, BTreeSet iteration and fixture cfg scope; ordinary strict-lint borrow/tuple/as_ref/const/conditional reports were fixed without production allowances. The early four-boundary discovery trial was replaced by the complete40-boundary matrix.
- [x] Record gate/setup corrections: restricted /proc/1 namespace access replaced with mapped-UID plus NS_GET_USERNS ownership checks; safe private-loopback positive/host-map negative checks passed. First gate stopped130 for the reviewed redaction fix; the next run failed127 after editing its active Bash file. The final frozen script explicitly enables acquisition plus both diagnostic features and passed in full.

### 3c97c52 — A2 durable resume and pull UX

- [x] Add cache-owned durable proper-prefix records bound to immutable source/object/role/length/expected hash, strong ETag and actual inode/prefix SHA256. Recovery rehashes/copies into a new exclusive stage using the existing admitted attempt and sole A1 installer/publisher.
- [x] Batch checkpoints8MiB/1s plus interrupted-read/cancel paths; sync payload before atomic record replacement and parent sync. Ignore charged uncommitted tails and forbid saved prefixes at exact total until fresh HTTP terminal framing is proved.
- [x] Keep unchanged resumed prefixes ephemeral in the new attempt: repeated refused/no-progress retries preserve the original checkpoint without retaining duplicate copies. Genuine progressed old attempts remain named and charged; new full copy plus24KiB/file control peak must fit before transfer.
- [x] Close the public trusted-digest private-copy quota escape: hold one original digest lock through the guard, refuse already verified named content before stage creation, and reuse that lock during install. Direct same-hash alias/concurrent publisher tests validate physical charge and zero body reads on refusal.
- [x] Preserve execute and add borrowed source-bound execute_with_progress/PullProgress. Accepted file/aggregate bytes stay bounded and monotonic; cached, resumed and downloaded counts are distinct, and100% bytes is not publication.
- [x] Resume only exact If-Range/206 with matching validator/start/total/length/identity;200, changed source or malformed range refuse without changing the previous ref. Full raw digest and actual EOF remain mandatory, including empty objects.
- [x] Add optional Linux CLI SIGINT/SIGTERM flags and RAII callback cleanup, throttled numeric stderr progress and separate final stdout receipt; offline/help/cache/default CPU install no handler. Cached signal-hook0.3.18/registry1.4.8 are the only newly locked packages.
- [x] Share one bounded HF initial selector online and offline for exact/glob/subfolder/Unicode/missing/ambiguous/budget behavior. Offline verifies cached paths before credentials/client/provider; legacy/manual manifests gain no model closure or freshness claim.
- [x] Integrate three independent ownership/review streams. Fix exact-total EOF bypass, duplicate no-progress checkpoint copies, post-read/pre-write cancellation checkpoint loss and public known-digest quota escape; deterministic and real-process regressions pass.
- [x] Retain prior accepted bytes when a newly returned read is canceled before the cache write guard; checkpoint on both prefix and normal write errors. The deterministic case accepts128 of256 returned bytes, resumes at128 and preserves the old reference.
- [x] Verify final cache13, transfer11 and CLI/filter focused suites; preserve original A1 and discovery failure/process-exit regressions. Initial cache clippy needless-borrow report in the shared-installer refactor was fixed; strict rechecks pass without allowances.
- [x] Run the full frozen-source gate:353 Rust executions passed,0 failed,34 optional/helper entries ignored (feature repetitions included), strict lint/release/installer and namespace/comparator/resume self-tests passed. No numerical tolerance, inference operator, checkpoint model or device qualification changed.
- [x] Verify actual offline Nix filtered-source integrity,23 local documentation links, single tracked Cargo.lock and byte-identical root README. No dependency/model/SDK/provider acquisition, device action or publication occurred.
- [x] Integrate completed run outcomes into this checklist and design reference; retire temporary run checklists/logs. Keep independent A3, removal, component and parity gates open.

## Uncommitted

- [x] Add explicit independently pinned component requests/plans, per-label credential transport adapters, bounded role suppliers and separate original verified snapshots. No composite source manifest, mutable ref, second publisher/runtime or model-readiness claim.
- [x] Reuse the first exact commit for repeated file-source aliases and preserve strict ordinary model closure. Relax only explicitly supplied config/tokenizer sidecars; retain missing-shard/index/split/role refusal. Repeated release aliases remain a separate unsupported frontier.
- [x] Bound actual aggregate component metadata bytes/requests, file counts, payload totals and plan metadata under the caller's existing transfer admission. Every component uses the original A1 executor with no ref; a complete receipt requires every publication durable and each descriptor lease verified. Earlier data can remain after later failure.
- [x] Add pure offline component opening before credentials/client/provider/admission, with original source/selection/exact-pin/role/shared-pin checks. Move the existing ordinary cached-selection validator into the common owner; no fabricated offline transfer receipt or freshness assertion.
- [x] Add original logical-name Artifact descriptor construction and Unix positioned reads. Cloned OS cursors cannot interfere; object/root/ref replacement cannot retarget the owned descriptor.
- [x] A3: verify the exact bounded owned source allocation against the manifest length/SHA256, reparse those same bytes and invoke the existing sole SourceIdentity issuer. Exact preallocation plus a separate EOF sentinel preserves the GGUF-only4GiB cap; mutation/digest/format/role/length/cap refusals release resources.
- [x] Admit raw ImmutableWeights bytes before allocation in the caller's existing ResourceArbiter domain and retain that lease inside SourceWeights until the actual allocation/all model/session Arc owners drop. Legacy ordinary open keeps its prior accounting boundary; metadata/tokenizer/session/global-RAM accounting is not asserted.
- [x] Add PreparedQwen shared cold preparation and bounded actual-copy config/tokenizer verification (1MiB/64MiB). Hold sidecar staging admission through parsing/construction; normal and cache paths reuse family validation and one unchanged CPU inference loop.
- [x] Keep the new family constructor safe: accept bounded config bytes through the existing parser rather than trusting publicly mutable ModelGeometry. Reuse a private checked constructor and cheap early family eligibility check before raw-source copying; no duplicate numerical validator or mathematical-body change.
- [x] Add explicit HTTP-free CLI cache feature and run-cache selectors, separate config/tokenizer snapshots, logical paths and raw-source/staging cap (default5GiB). Cache misses refuse, ordinary run never acquires, and default CPU/cache-only closures exclude HTTP/TLS/signal/GPU owners.
- [x] Add pull components PLAN.json with fixed schema,1MiB/16-component bounds, unknown/duplicate-field and conflicting-selector refusal, metadata-only dry-run, explicit offline bindings, separate label-scoped origin credentials and honest per-source output. Tokens/URLs are not plan fields and errors redact unrecognized input.
- [x] Integrate three independent implementation/review streams. Fix the broader-store/narrower-offline-file-limit gap with an actual public regression, unsafe mutable-geometry construction and late unsupported-family allocation. An early overly broad HF edit was corrected before compilation; no acceptance command failed or numerical tolerance changed.
- [x] Revalidate the six approved local checkpoint/config/tokenizer identities, CPU/RAM/disk and initial worktree. Qualify both original .8B/2B models through independently opened config/tokenizer snapshots: all248320 logits and all42 full state-domain vectors match direct CPU loading bit-for-bit over two tokens/nonzero state.
- [x] Verify actual independent sessions/unchanged sibling roots, distinct allocation source IDs, foreign-source restore refusal, ref retarget/private cache mutation isolation, denied staging/raw admission and exact retained raw-source charge until the last owner drops, then zero.
- [x] Run the frozen complete isolated gate:419 Rust executions passed,0 failed,54 optional/helper entries ignored (feature-build repetitions included). Strict minimal/default/cache/acquisition/diagnostic checks, CPU/cache HTTP exclusion, release/installer,18 Python namespace tests and comparator/resume self-tests pass; the ignored real-model gate was separately selected and passed for both models.
- [x] Verify actual offline Nix filtered-source integrity, changed-document local links, one tracked Cargo.lock and byte-identical user root README. No new dependency packages, model/SDK/provider fetch, device action or publication; optional CLI serde reuses the already locked package.
- [x] Integrate completed implementation run outcomes into this checklist/design reference and retire their temporary tracking. Keep source shards/new families, missing lengths, removal/reclamation and broader provider parity open.

- [ ] Qualify plans with missing provider metadata lengths; this slice requires exact listed sizes, including for chunked bodies.
- [ ] Independently qualify repeated release-alias multi-asset component plans; these currently refuse rather than silently mixing source identities.
- [ ] Separately qualify snapshot/retained-partial lease-aware removal, orphan reclamation and actual release accounting; automatic removal remains Unsupported until then.
- [ ] Finish the provisional acquisition capability ledger: external-cache read-only import, additional hosted model hubs, S3/GCS/Azure/S3-compatible and OCI/Xet profiles; opaque serialization/plugin loaders remain separate format/runtime work.
- [ ] Continue explicit source/license authorization and trusted-digest versus first-seen provenance through A3 and every additional provider; acquisition does not grant a license or loading support.
- [ ] Add a verified cached-result path for concurrent cache fill between initial lookup and partial guard acquisition; currently a known-object race refuses safely and callers retry explicitly.
- [ ] Poll cancellation through large local cached-object hash scans; current cooperative checks can observe the flag only after that existing scan completes. Blocking filesystem calls do not gain real-time interruption guarantees.

## Tests and verification

- [x] ACQ-A01/A02/A04/A05/A06/A07: original A1 path/schema/admission/publication/crash/concurrency/lease regressions pass. Existing38-callback matrix retains76 injected I/O failures and76 real child exits.
- [x] ACQ-A03: discovered first-seen/trusted streaming, exact length/digest, LFS, closure, cancellation and ref rollback pass. Actual40 callback ordinals across absent/existing refs exercise80 I/O failures and80 real child exits, post-rename DurabilityUnknown, quota rebuild and subsequent publication. Final acquisition transactions:11 passed plus1 child helper ignored.
- [x] ACQ-A08: isolated offline hit/miss, source/selection/exact-pin/locator-kind refusal precede credentials/provider/client. Public metadata admission and pre-cancel tests use panic-on-request transport.
- [x] ACQ-A09: optional artifacts/facade/CLI strict lint with production-only and diagnostic acquisition; default CPU+acquisition compile and minimal/default CPU/core/inspection checks pass. Normal+build CPU closure excludes HTTP/TLS/GPU.
- [x] ACQ-A10 bounded slice: actual HTTP12/HF7/resume2 focused fixtures passed, including real verified TLS, redaction at requested runtime TRACE, framing, redirects, identity/range validators, retry/backoff/deadline/cancellation. The continuation extends durable resume/progress below; automatic reclamation remains open.
- [x] GitHub15 focused provider tests passed: exact commit/tree or release/asset pin/digest, truncation/missing/unsafe/LFS, complete model closure, uppercase identity and mixed-family refusal.
- [x] Focused provider/parser/CLI checks:23 executions passed across optional/default-disabled commands; full gate repeats relevant suites. Offline operations verify actual cached bytes; mutable branch/tag freshness is not asserted.
- [x] Full CARGO_BUILD_JOBS=2 RAYON_NUM_THREADS=4 python3 scripts/native-gpu.py -- bash scripts/check.sh passed on frozen sources with discovery crash diagnostics enabled; documentation integrity, original root README and actual Nix filtered-source gate passed.
- [x] ACQ-A11: actual A3 owned-copy original .8B/2B bridge passed. Isolated jobs2/Rayon4 `cargo test --release --locked --offline -p zerogravity --features cache --test cache_prepare approved_cached_models_match_direct_source_exactly -- --ignored --exact --nocapture`:1 selected test covers both approved models,2 tokens each, all248320 logits and all42 full domains exact; execution18.49s. No8K/new-family/GPU requalification claim.
- [ ] ACQ-A12: complete the finite provider/loader parity inventory and independently qualify remaining supported paths.

- [x] Continuation cache:13 partial tests passed;16 I/O faults and16 real exits across8 initial/replaced checkpoint callbacks, uncommitted tails, binding/tampering, live process locks, no-progress cleanup, proper-prefix bounds and retained-old plus full-new peak admission. Original A1 transactions15 passed/one child helper ignored.
- [x] Continuation transport:11 combined tests passed/one child helper ignored;9 production tests passed. Actual loopback range/refusal matrix, live retry, weak/missing validators, full resumed digest, source/object mismatch, periodic-checkpoint child exit and exact-total/trailing-byte plus post-read cancellation regressions passed. Existing discovery11 tests and80 I/O/80 real-exit matrix remain green.
- [x] Continuation CLI:4 unit tests include actual SIGINT/SIGTERM children, callback unregister and real cached+streamed progress;6 cache/filter integration tests and8 HF tests pass, plus default-disabled refusal.19 distinct focused names/23 executions including the repeated4 after quota repair; full gate repeats final suites.
- [x] Continuation full gate:353 Rust executions passed,0 failed,34 optional/child-helper entries ignored (including feature-build repeats). Strict production/diagnostic/combined checks, default CPU signal/HTTP/TLS/GPU exclusion, release/installer,18 Python namespace tests and comparator/resume self-tests pass. Actual offline Nix source-integrity,23 local links, one tracked Cargo.lock and unchanged root README pass.

- [x] Components/A3 source gates:7 synthetic bridge tests,30 artifact library tests, original13 partial/15 transaction regressions,37 provider/transport tests,11 progress and11 acquisition transactions pass. Final component suite10 passed including the tighter public offline per-file bound; strict production/diagnostic artifact lint passed without allowances.
- [x] Components/A3 facade/CLI gates:21 selected executions passed, including8 frontend cases, tiny preparation refusal,10 corrected component tests, minimal cache-only refusal and1 explicitly selected real-model test exercising both checkpoints. Strict all-target default CPU+cache/acquisition and minimal cache lint/check passed; same-descriptor/cursor/path/mutation/lease guarantees received independent review.
- [x] Components/A3 full gate: `CARGO_BUILD_JOBS=2 RAYON_NUM_THREADS=4 python3 scripts/native-gpu.py -- bash scripts/check.sh` passed once on frozen source:419 Rust executions/0 failures/54 ignored across75 suite executions. Release/installer/lint/18 namespace and comparator/resume checks passed. Actual offline Nix source-integrity, all changed-document local links, single tracked lock and original root README passed.

## Deferred

- [ ] Unsupported plans fail closed until qualified: missing metadata lengths, repeated release-alias multi-asset plans and physical removal/reclamation remain explicit above. Independent component data grants no additional loader eligibility.
- [ ] Linux local-filesystem semantics are qualified first. HTTP cancellation obeys configured whole-request bounds; old retained attempts consume real quota and arbitrary same-UID mutation remains outside descriptor isolation.
