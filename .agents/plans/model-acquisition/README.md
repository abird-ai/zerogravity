Author: GPT-6

# Model acquisition

Explicitly acquire bounded, pinned source files into the existing XDG cache.
This is data acquisition, separate from loading a model or granting execution
support. Implementation and tests are tracked in [TODO.md](TODO.md); the
[full acquisition specification](../MODEL_ACQUISITION_IMPLEMENTATION_SPEC.md)
and [provider parity research](../../research/MODEL_PULL_XDG_CACHE_AND_SOURCE_PARITY_2026-10-16.md)
retain the broader scope.

## Design

One optional synchronous HTTPS owner serves both providers. Native Rustls TLS
verification, origin-scoped credentials, bounded metadata/headers/redirects and
identity encoding protect exact transferred bytes. Default CPU, core and source
inspection builds keep the transport dependency closure disabled.

ureq's TRACE formatting exposes signed URL queries and redirect Location values.
Selecting acquisition therefore caps the process-wide Rust `log` facade at DEBUG
at compile time, including when a runtime logger requests TRACE. Default builds
are unaffected; `tracing` and the engine's explicit JSON diagnostics are separate.
Downstream builds cannot select a conflicting `log/max_level_*` feature. This
explicit dependency constraint avoids silently leaking credentials through a
host application's logger; it can be revisited when upstream safely redacts TRACE.

Resolution uses a caller-owned transfer admission, pins a repository commit or
release asset identity, validates file paths and selects the required closure.
The returned plan has private validated fields; expiring download URLs stay in
memory. It grants neither SourceIdentity nor model/device readiness.

The existing CacheStore admits exact planned lengths before object requests.
Its acquisition transaction streams each file once, checks the actual raw digest
and preserves TrustedSha256 versus FirstSeen provenance in the manifest. Actual
entries replace private accounting placeholders before the original snapshot/ref
publisher can run. Existing leases, stable locks, failure outcomes and quota
ownership remain authoritative; there is no second cache or allocator.

A live interrupted transfer can append only after an exact 206 response with the
same strong ETag, offset, total and identity encoding. A 200 or changed validator
fails that attempt without concatenation or changing the previous reference.
Persistent partial recovery is a separate unfinished protocol.

## Decisions

- Pulls are explicit. Ordinary run, inspect and catalog misses do not acquire.
- Offline commands take the local branch before creating clients or credentials.
  They do not establish remote freshness or token-revocation state.
- Tokens come from the selected environment profile, never command-line values,
  cache manifests, progress text or upstream error strings. No remote code runs.
- Source-provided SHA256 and first-seen local content hashes have different trust
  strength. Neither automatically supplies a model license or execution contract.
- Exact metadata lengths are required in this slice; chunked bodies can still
  transfer against that bound. Host admission covers this process's transfer
  work, and disk quotas cover cooperating cache writers, not global OS resources.
- Linux local-filesystem semantics are qualified first. Open/read-only cache
  descriptors do not prevent arbitrary same-UID mutation; A3 must hash the bytes
  of its actual owned source copy before the existing source issuer runs.

## Interfaces and remaining scope

The optional `acquisition` feature exposes the library facade and explicit CLI
pull/cache commands. `diagnostic-acquisition` adds controlled fixture endpoints
and trust roots; it is not a production insecure-TLS option. Current supported
commands and limits are shown by `zerogravity pull --help` in that build.

The A2 slice uses one repo/revision per plan. Required selected-model sidecars
and shards must be available there; independently pinned cross-repository
components remain future work. Git LFS pointer payloads, missing metadata sizes,
persisted cross-invocation resume and lease-aware removal refuse in this slice.
Acquiring an auxiliary file or unqualified model format grants no loader support. A3 cache-to-model loading and the wider
Hub/cloud/OCI/Xet/parity inventory remain separate unchecked work.

Blocking HTTP reads are bounded by the configured whole-request deadline;
cancellation is checked between reads and at those deadlines. The synchronous
CLI keeps normal process-signal behavior and does not claim a graceful signal
handler or real-time cancellation guarantee.

## References

- [Hugging Face Hub API](https://huggingface.co/docs/hub/api) and
  [official OpenAPI](https://huggingface.co/.well-known/openapi.md).
- [GitHub Git trees](https://docs.github.com/en/rest/git/trees) and
  [release assets](https://docs.github.com/en/rest/releases/assets).
- [ureq 3.4.2](https://docs.rs/ureq/3.4.2/ureq/), with its exact cached source
  used to check timeout, TLS and redirect behavior.
- [A1 local cache contract](../../research/PULL_A1_LOCAL_XDG_CACHE_2026-10-16.md),
  [current workflow](../../../AGENTS.md) and [TODO rules](../TODO-SPEC.md).
