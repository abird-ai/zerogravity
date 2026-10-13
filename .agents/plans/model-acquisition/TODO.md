# Model acquisition

Provenance commit: `3787519`.
Scope: P03; prerequisite to new acquisition/loader claims.
Contracts: [A2/A3 specification](../MODEL_ACQUISITION_IMPLEMENTATION_SPEC.md),
[pull/provider parity](../../research/MODEL_PULL_XDG_CACHE_AND_SOURCE_PARITY_2026-10-16.md).

## Uncommitted

- [ ] Implement one bounded synchronous HTTPS owner: TLS, origin-scoped auth/redaction, redirects, limits, timeout/retry/backoff, progress, validator-bound ranges/resume and cancellation.
- [ ] Add exact-revision HF file/subfolder/filter and GitHub repository/release resolution; refuse ambiguous quant, incomplete sidecars/shards, changed revisions and unqualified LFS pointers.
- [ ] Add explicit facade/CLI pull, dry-run, offline/list/inspect/verify with honest unsupported formats; run/inspect/catalog misses never implicitly acquire.
- [ ] Prove offline branches precede provider/credential construction; use controlled HTTP fixtures for auth/range/cancel/error and no hidden network.
- [ ] A3: bind original opened cache descriptors, hash the actual owned-copy bytes, reparse them and issue only the existing SourceIdentity; verify original .8B/2B inference through that bridge.
- [ ] Separately qualify lease-aware removal/orphan recovery and actual release accounting; A1 removal remains Unsupported until then.
- [ ] Finish the provisional acquisition capability ledger: external-cache read-only import, additional hosted model hubs, S3/GCS/Azure/S3-compatible and OCI profiles; opaque serialization/plugin loaders remain separate format/runtime work.
- [ ] Retain explicit source/license authorization, trusted-digest versus first-seen provenance, default dependency isolation and all A1 accounting/durability limits.

## Committed

### 3787519

- [x] Local Linux cache A1 accepted. Cooperative quotas are not global host/UMA reservation; descriptors do not prevent arbitrary same-UID mutation.

- [x] Actual A1 full gate:194 Rust executions passed,27 optional executions ignored; strict lint/release/installer/isolation checks passed. No inference/HTTP claim follows from cache publication.
- [x] Publication fault/recovery:18 operation types,38 callbacks,76 injected I/O and76 real child-exit fixtures; retained refs/leases and concurrent quota rebuild regressions passed.
- [x] Review fixes: retained named ref backing, private snapshot reservation accounting and hardlink-only ctime verification; deterministic regressions passed. Early readiness-marker and cfg-import harness issues were fixed before acceptance.

Technical API/bounds: [A1 cache reference](../../research/PULL_A1_LOCAL_XDG_CACHE_2026-10-16.md).
