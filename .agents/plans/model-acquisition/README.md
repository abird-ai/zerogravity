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

The current continuation qualifies durable partial recovery through that same
attempt owner. Only a synced proper prefix (shorter than the complete object)
can be saved, bound to immutable source/object identity, exact total, strong ETag,
prefix digest and inode identity. Recovery rehashes and copies those bytes into
an exclusively created file under a new full-peak admission. It never reopens a
shared cache inode for writing. An exact206 response must match validator,
offset, total and identity encoding before append;200 or changed metadata refuse.
Even a cancellation at the final byte cannot bypass the final HTTP EOF check.

Checkpoints batch after8MiB or1second of accepted progress, plus interrupted-read
or cancellation cleanup. File sync precedes atomic record replacement and its
parent sync. Uncheckpointed crash tails remain charged but are not resumed.
Unchanged resumed prefixes create no second retained checkpoint. Once additional
progress is checkpointed, old retained attempts remain named and physically
charged; automatic reclamation awaits the separate removal protocol. Recovery
therefore needs space for the retained old files plus the new full transfer peak,
including24KiB/file for partial control records/lock/replacement. Repeated genuine
interruptions can consume quota; no global free-space or reclamation claim follows.

The stable partial guard serializes one source object's transfer at a time.
For a trusted checksum it also holds the original digest lock through staging,
so another publisher cannot create a named blob that falsely credits a private
copy's reservation. An already verified named object refuses the partial-write
API before file creation; callers use try_cached instead. Publication and digest
dedup remain A1's authority. A concurrent cache fill between the executor's initial
lookup and guard acquisition can require a retry; an automatic post-wait cache-hit
result is a remaining optimization, not a network single-flight guarantee.

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

Ordinary pulls retain strict same-repository model closure. Explicit component
plans can supply config/tokenizer from separate authorized sources. Each member
retains its own original provider/repository/immutable revision and manifest;
there is no merged-source manifest, composite ref or cross-source transaction.
Git LFS pointer payloads, missing metadata sizes and lease-aware removal refuse.
Acquiring an auxiliary file or unqualified model format grants no loader support.
The wider Hub/cloud/OCI/Xet/parity inventory remains separate unchecked work.

The continuation adds a borrowed source-bound progress callback to the existing
executor, preserving its original entry point. Logical file/aggregate bytes count
accepted streamed bytes, verified cache hits and rehashed resumed prefixes;
cached/resumed bytes are distinct. Complete byte progress is not publication.
The CLI throttles numeric progress on stderr; its final receipt stays on stdout.
Offline HF filters reuse the same bounded initial selector against verified
cached paths, before credentials/client/provider creation. This establishes no
remote freshness or additional loader eligibility.

Online CLI pulls register safe optional Linux signal-hook flags for SIGINT and
SIGTERM and unregister their callbacks on return. No handler is registered for
help/offline/cache commands or default CPU builds. signal-hook leaves its OS
dispatcher installed after callbacks are removed; this standalone single-command
CLI exits afterward rather than claiming restoration of prior dispositions.
Blocking HTTP reads remain bounded by the configured whole-request deadline;
cancellation is checked between reads and at deadlines. Existing whole-object cached verification can finish its hash scan before the
flag is observed. Filesystem syncs and blocked reads do not gain a real-time
cancellation guarantee.

## Explicit components

`pull components PLAN.json` accepts the fixed `zerogravity.pull.components.v1`
schema, bounded to 1 MiB and 16 components. Each component declares a unique label,
role (`weights`, `config`, `tokenizer`, `auxiliary`), provider (`huggingface` or
`github`), repository and either revision/files/optional subfolder or release/asset.
Unknown/duplicate fields, URL/token fields and conflicting selectors refuse.
For example, independently authorized original files can be selected with:

```json
{
  "schema": "zerogravity.pull.components.v1",
  "components": [
    {"label":"weights","role":"weights","provider":"huggingface","repository":"owner/converted-model","revision":"main","files":["model.gguf"]},
    {"label":"config","role":"config","provider":"huggingface","repository":"owner/original-model","revision":"main","files":["config.json"]},
    {"label":"tokenizer","role":"tokenizer","provider":"huggingface","repository":"owner/original-model","revision":"main","files":["tokenizer.json"]}
  ]
}
```

These repository names illustrate the schema; they do not select an approved
source or imply format/model eligibility. Resolution pins aliases before payload
transfer. Repeated file-source provider/repository/requested-revision tuples reuse
the first exact commit, preventing alias movement between components. Repeated
release aliases remain Unsupported pending exact multi-asset qualification.
Only an explicitly declared config/tokenizer supplier relaxes the corresponding
missing weight-repository sidecar; shard/index/split closure remains strict.

Actual metadata bytes and requests share aggregate budgets across the set, as do
resolved file counts, payload bytes and plan metadata. The caller's existing
transfer admission covers resolution and the same A1 executor handles each source.
Credential clients are bound to explicit labels and provider origins. A final
component receipt requires every individual publication to be durable and its
original snapshot descriptors verified. Earlier complete immutable snapshots can
remain after a later failure; no mutable reference is changed by component pulls.

`--dry-run` resolves metadata only. `--offline` requires an explicit `snapshot`
selector on every component and verifies the complete local set before any
credentials/client/provider work. Online plans reject those offline selectors.
The pure library `open_components` takes no transport. Offline verification cannot
prove mutable selector freshness, token revocation, model compatibility or licensing.

## Cache-to-model preparation

The independent `cache` feature enables Linux cache loading without enabling
HTTP, TLS or signal handlers. With CPU/Qwen/text also selected, use:

```text
zerogravity run-cache MODEL_SNAPSHOT --model-file model.gguf \
  --config-snapshot CONFIG_SNAPSHOT --config-file config.json \
  --tokenizer-snapshot TOKENIZER_SNAPSHOT --tokenizer-file tokenizer.json \
  --prompt "Hello" --max-tokens 2
```

Selectors are explicit snapshot digests or existing refs. Missing sources refuse;
run-cache never acquires. Normal paths and cache preparation share `PreparedQwen`,
the family's geometry/tensor/tokenizer validation and the same CPU inference loop.
The family constructor accepts bounded config bytes and reuses the existing
parser; publicly mutable ModelGeometry is never a trusted construction argument.
A cheap family eligibility check rejects unsupported geometry before copying
weights, and the constructor still parses those verified config bytes itself.
The supported executable sources remain original approved Qwen3.5 .8B/2B GGUF;
new families, shards, formats or GPU eligibility do not follow from this bridge.

`SnapshotLease` supplies the original opened descriptors and original logical
format names. Positioned reads avoid OS cursor sharing across cloned descriptors.
The source bridge copies within the existing 4 GiB cap, verifies the SHA256 of that
actual owned allocation, reparses those same bytes and invokes the sole existing
SourceIdentity issuer. Config/tokenizer copies are likewise bounded and hashed
before parsing (1 MiB/64 MiB respectively). Path/ref replacement cannot retarget
these descriptors. Arbitrary same-UID inode mutation can still occur, so the
owned-copy digest is mandatory; descriptors alone are not immutable storage.

The caller's existing ResourceArbiter domain admits raw source bytes before
allocation. Its ImmutableWeights lease stays inside SourceWeights through all
shared model owners and releases after the actual raw allocation is dropped.
Sidecar staging admission lasts through copying/hashing/parsing; parsed metadata,
tokenizer structures, session memory and global host/UMA capacity are outside
this scoped accounting. The CLI registers one process-shared source domain with
a 5 GiB default cap, adjustable via `--source-memory-limit-bytes`. Legacy path-open
accounting keeps its existing boundary. Cache preparation grants a fresh source
allocation identity, while sessions on one loaded Arc share that immutable source
and retain independent StateRoots. Numerical acceptance is tracked in TODO.md.

## References

- [Hugging Face Hub API](https://huggingface.co/docs/hub/api) and
  [official OpenAPI](https://huggingface.co/.well-known/openapi.md).
- [GitHub Git trees](https://docs.github.com/en/rest/git/trees) and
  [release assets](https://docs.github.com/en/rest/releases/assets).
- [signal-hook flags](https://docs.rs/signal-hook/0.3.18/signal_hook/flag/) and
  [registry cleanup](https://docs.rs/signal-hook-registry/1.4.8/signal_hook_registry/fn.unregister.html),
  checked against the exact already-cached dependency sources.
- [ureq 3.4.2](https://docs.rs/ureq/3.4.2/ureq/), with its exact cached source
  used to check timeout, TLS and redirect behavior.
- [A1 local cache contract](../../research/PULL_A1_LOCAL_XDG_CACHE_2026-10-16.md),
  [current workflow](../../../AGENTS.md) and [TODO rules](../TODO-SPEC.md).
