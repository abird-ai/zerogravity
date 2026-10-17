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

## Uncommitted

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
- [ ] Finish remaining A2 UX: live byte-progress callbacks/reporting and cooperative CLI signal cancellation; current CLI reports final summaries, with library AtomicBool cancellation and bounded blocking deadlines.
- [ ] Expand offline HF filters against verified cached paths; current offline exact paths work and unsupported globs refuse safely.
- [ ] Qualify persistent cross-invocation partial resume, owned-prefix rehash/recovery and reservation accounting; current retry support is live-attempt only.
- [ ] Add separately authorized and independently pinned cross-repository component plans; required sidecars absent from the selected repository currently refuse.
- [ ] Qualify plans with missing provider metadata lengths; this slice requires exact listed sizes, including for chunked bodies.
- [ ] A3: bind original opened cache descriptors, hash the actual owned-copy bytes, reparse them and issue only the existing SourceIdentity; verify original .8B/2B inference through that bridge.
- [ ] Separately qualify lease-aware removal/orphan recovery and actual release accounting; removal remains Unsupported until then.
- [ ] Finish the provisional acquisition capability ledger: external-cache read-only import, additional hosted model hubs, S3/GCS/Azure/S3-compatible and OCI/Xet profiles; opaque serialization/plugin loaders remain separate format/runtime work.
- [ ] Continue explicit source/license authorization and trusted-digest versus first-seen provenance through A3 and every additional provider; acquisition does not grant a license or loading support.

## Tests and verification

- [x] ACQ-A01/A02/A04/A05/A06/A07: original A1 path/schema/admission/publication/crash/concurrency/lease regressions pass. Existing38-callback matrix retains76 injected I/O failures and76 real child exits.
- [x] ACQ-A03: discovered first-seen/trusted streaming, exact length/digest, LFS, closure, cancellation and ref rollback pass. Actual40 callback ordinals across absent/existing refs exercise80 I/O failures and80 real child exits, post-rename DurabilityUnknown, quota rebuild and subsequent publication. Final acquisition transactions:11 passed plus1 child helper ignored.
- [x] ACQ-A08: isolated offline hit/miss, source/selection/exact-pin/locator-kind refusal precede credentials/provider/client. Public metadata admission and pre-cancel tests use panic-on-request transport.
- [x] ACQ-A09: optional artifacts/facade/CLI strict lint with production-only and diagnostic acquisition; default CPU+acquisition compile and minimal/default CPU/core/inspection checks pass. Normal+build CPU closure excludes HTTP/TLS/GPU.
- [x] ACQ-A10 bounded slice: actual HTTP12/HF7/resume2 focused fixtures passed, including real verified TLS, redaction at requested runtime TRACE, framing, redirects, identity/range validators, retry/backoff/deadline/cancellation. Persistent resume and live progress remain open above.
- [x] GitHub15 focused provider tests passed: exact commit/tree or release/asset pin/digest, truncation/missing/unsafe/LFS, complete model closure, uppercase identity and mixed-family refusal.
- [x] Focused provider/parser/CLI checks:23 executions passed across optional/default-disabled commands; full gate repeats relevant suites. Offline operations verify actual cached bytes; mutable branch/tag freshness is not asserted.
- [x] Full CARGO_BUILD_JOBS=2 RAYON_NUM_THREADS=4 python3 scripts/native-gpu.py -- bash scripts/check.sh passed on frozen sources with discovery crash diagnostics enabled; documentation integrity, original root README and actual Nix filtered-source gate passed.
- [ ] ACQ-A11: A3 owned-copy original-model bridge and actual .8B/2B loading/inference qualification.
- [ ] ACQ-A12: complete the finite provider/loader parity inventory and independently qualify remaining supported paths.

## Deferred

- [ ] Unsupported plans fail closed until their gates pass: durable resume, missing metadata lengths and independent cross-repository components are explicitly listed in Uncommitted.
- [ ] Linux filesystem semantics are qualified first. Blocking reads obey the configured whole-request deadline, not an immediate cancellation or idle-timeout guarantee; arbitrary same-UID mutation remains outside descriptor isolation.
