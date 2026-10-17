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

## Uncommitted

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
- [ ] Add separately authorized and independently pinned cross-repository component plans; required sidecars absent from the selected repository currently refuse.
- [ ] Qualify plans with missing provider metadata lengths; this slice requires exact listed sizes, including for chunked bodies.
- [ ] A3: bind original opened cache descriptors, hash the actual owned-copy bytes, reparse them and issue only the existing SourceIdentity; verify original .8B/2B inference through that bridge.
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
- [ ] ACQ-A11: A3 owned-copy original-model bridge and actual .8B/2B loading/inference qualification.
- [ ] ACQ-A12: complete the finite provider/loader parity inventory and independently qualify remaining supported paths.

- [x] Continuation cache:13 partial tests passed;16 I/O faults and16 real exits across8 initial/replaced checkpoint callbacks, uncommitted tails, binding/tampering, live process locks, no-progress cleanup, proper-prefix bounds and retained-old plus full-new peak admission. Original A1 transactions15 passed/one child helper ignored.
- [x] Continuation transport:11 combined tests passed/one child helper ignored;9 production tests passed. Actual loopback range/refusal matrix, live retry, weak/missing validators, full resumed digest, source/object mismatch, periodic-checkpoint child exit and exact-total/trailing-byte plus post-read cancellation regressions passed. Existing discovery11 tests and80 I/O/80 real-exit matrix remain green.
- [x] Continuation CLI:4 unit tests include actual SIGINT/SIGTERM children, callback unregister and real cached+streamed progress;6 cache/filter integration tests and8 HF tests pass, plus default-disabled refusal.19 distinct focused names/23 executions including the repeated4 after quota repair; full gate repeats final suites.
- [x] Continuation full gate:353 Rust executions passed,0 failed,34 optional/child-helper entries ignored (including feature-build repeats). Strict production/diagnostic/combined checks, default CPU signal/HTTP/TLS/GPU exclusion, release/installer,18 Python namespace tests and comparator/resume self-tests pass. Actual offline Nix source-integrity,23 local links, one tracked Cargo.lock and unchanged root README pass.

## Deferred

- [ ] Unsupported plans fail closed until qualified: missing metadata lengths, independent cross-repository components and physical removal/reclamation remain explicit above.
- [ ] Linux local-filesystem semantics are qualified first. HTTP cancellation obeys configured whole-request bounds; old retained attempts consume real quota and arbitrary same-UID mutation remains outside descriptor isolation.
