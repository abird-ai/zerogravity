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

The existing CacheStore admits exact planned lengths or explicit full reservation
caps before object requests. Its acquisition transaction streams each file once,
checks the actual raw digest and preserves TrustedSha256 versus FirstSeen
provenance in the manifest. Actual
entries replace private accounting placeholders before the original snapshot/ref
publisher can run. Existing leases, stable locks, failure outcomes and quota
ownership remain authoritative; there is no second cache or allocator.

Durable partial recovery uses that same attempt owner. Only a synced proper
prefix (shorter than the complete object) can be saved, bound to immutable
source/object identity, exact total, strong ETag,
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
dedup remain A1's authority. The executor rechecks a known digest after acquiring
those locks. A concurrent fill returns a typed verified cached result before a
new private payload or HTTP
body is requested. It is counted as cached progress, with the same prefix/LFS
checks and original transaction completion. The legacy partial-write API still
refuses a known object; callers can explicitly select the cached-or-partial API.
Unknown-digest first-seen transfers do not gain a network single-flight guarantee.

## Decisions

- Pulls are explicit. Ordinary run, inspect and catalog misses do not acquire.
- Offline commands take the local branch before creating clients or credentials.
  They do not establish remote freshness or token-revocation state.
- Tokens come from the selected environment profile, never command-line values,
  cache manifests, progress text or upstream error strings. No remote code runs.
- Source-provided SHA256 and first-seen local content hashes have different trust
  strength. Neither automatically supplies a model license or execution contract.
- Exact lengths remain the default. Omitted listing lengths first use pinned LFS
  metadata or bounded body-free HEAD; explicit capped auxiliary streaming is
  described below. Chunked bodies can transfer against an established exact
  length or authorized cap. Host admission covers this process's transfer work,
  and disk quotas cover cooperating cache writers, not global OS resources.
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
Git LFS pointer payloads refuse. Lengths still unknown after bounded recovery
require the explicit auxiliary-only cap below; other unknown lengths refuse.
Explicit cache removal has the separate protected-reader contract below.
Acquiring an auxiliary file or unqualified model format grants no loader support.
The wider Hub/cloud/OCI/Xet/parity inventory remains separate unchecked work.

A borrowed source-bound progress callback extends the existing
executor, preserving its original entry point. Logical file/aggregate bytes count
accepted streamed bytes, verified cache hits and rehashed resumed prefixes;
cached/resumed bytes are distinct. Exact totals are optional and reservation
maxima are separate; a cap is never displayed as an observed or exact total.
A final file_complete event follows verified body/cache completion, including
when its accepted-byte count repeats the prior event. It is not publication.
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
cancellation is checked between reads and at deadlines. Online acquisition
threads that same flag through cached-object verification before/after each
64 KiB read, EOF and digest finalization. Additive cancellable snapshot/ref APIs
allow an existing caller flag to cover final verification. Legacy convenience
APIs without a cancellation input retain their existing behavior. Parsing,
inventory, filesystem syncs and blocked reads do not gain a real-time
cancellation guarantee.

## Exact length recovery

Resolution first uses valid size metadata for the immutable selected object.
HF LFS metadata can supply a missing tree length while retaining its original
SHA256 trust; contradictory lengths refuse. A selected/required file whose size
is still absent can use HEAD on its exact pinned HF commit, GitHub raw commit or
release-asset-ID URL. Unselected files require no speculative probe. One shared
HTTP owner retains TLS, origin credentials, redirect policy, identity encoding,
request deadlines/retries and error redaction. Custom transports that do not
implement HEAD return Unsupported; there is no fallback GET for size discovery.

An exact probe must return HTTP200 with a valid Content-Length within the file
limit. Missing lengths refuse by default. Overflowing, contradictory, encoded
or ranged/ambiguous framing always refuses. No response payload is read; dry-run
remains metadata-only. Probe counts are bounded per source by max_pages independently of listing-page counts.
Component GET and HEAD calls also share the set's single max_pages request
budget. Metadata body limits keep their existing separate authority.
The resulting exact lengths enter the original full-peak cache admission; no
zero placeholder, guessed total or unreserved growing payload is introduced.

A strong probe ETag, when supplied, must match the initial transfer and applicable
range responses. Weak/absent validators grant no extra resume authority. ETags
and Git object IDs do not become raw SHA256, and recovery does not strengthen
first-seen content provenance. Final original raw digest, exact bytes and terminal
EOF checks remain mandatory. The optional bounded path below retains these
source and integrity boundaries.

## Explicit capped auxiliary streaming

`--unknown-size-cap-bytes N` authorizes a positive per-file reservation cap,
no greater than `--max-file-bytes`. It applies only when pinned auxiliary files
still lack lengths after normal metadata and HEAD recovery. Weights, indexes,
config, tokenizer and license objects keep their exact-length requirement and
ordinary required-file closure. Model/index selections keep all their files
exact, including auxiliary closure members. The flag is rejected with offline
mode before cache, plan-file or credential work. Components use the same global cap without
changing their JSON schema or independent source identities.

The fallback is deliberately narrow: a validated HTTP200 HEAD with no length,
or HTTP405/501 indicating unsupported HEAD. Other status, transport, TLS,
malformed metadata, cancellation and budget failures do not authorize streaming.
A strong successful HEAD ETag still binds GET. Validators and lengths on405/501
describe the error representation and grant no object authority. HEAD never
reads a body; dry-run stays metadata-only.

The shared ByteLength distinguishes Exact from AtMost. Exact constructors retain
their behavior; explicit new_with_length constructors authorize a bounded entry.
Provider resolvers additionally require PullLimits.unknown_length_cap; a trusted
caller constructing a resolved plan directly supplies that authority through
AtMost itself. Resolved-file bytes(), plan/component total_bytes() and progress
exact-total getters now return Option rather than treating caps as exact lengths.
Callers needing admission bounds use length()/maximum_bytes() and the explicit
progress maximum getters. ManifestEntry and published manifest byte counts remain
exact. This is a source API change in the optional acquisition interface.

The existing attempt reserves full maxima before any payload I/O. Same trusted
digests with unequal caps reserve their maximum once; unrelated first-seen files
reserve independently. A private one-shot admission input feeds the original
reservation ledger and sole installer/publisher. Final manifest entries contain
only observed bytes and the actual raw SHA256. A known-digest hit verifies the
original opened file's actual length against the bound and hashes those actual
bytes; an insufficient cap or corrupt hit refuses before HTTP/private payload
staging. The admitted attempt and reservation already exist at that lookup.

The original partial guard performs ephemeral capped transfers without recovery,
saved partial validators or durable checkpoints. Its explicit finish_after_eof
requires the caller to have proved source completion. Cancellation, errors and saved bytes
never imply EOF. The reader consumes at most the remaining bound plus one byte
for overflow detection, including a nonempty sentinel read at the cap. Empty
and short payloads are valid after EOF and hash verification; LFS pointer bodies
still refuse before accepted progress. Existing exact resumable transfers retain
their stricter total/range/checkpoint contract.

For capped HTTP GET, require200, identity encoding, no Content-Range, and either
a valid Content-Length within the cap or sole chunked transfer framing. Conflicting
or stacked framing and ambiguous close-delimited bodies refuse. A declared GET
length must still be reached exactly, including for custom transports. The single
existing ureq decoder determines payload completion: the pinned version accepts
a zero-sized chunk line followed by connection close without independently
validating a complete trailer section. Tests retain this explicit boundary;
trailer metadata is never source or checksum authority. Missing zero chunks,
truncated data chunks and short declared lengths fail. No duplicate HTTP parser
or silent completion-on-read-error path is introduced.

Progress and final receipts count actual accepted/cached bytes. Reaching a cap
alone is not completion; file_complete follows the verified terminal read and
cache guard finish. Complete component receipts still require every separate
publication and descriptor verification. A later failed member can leave earlier
immutable data; no cross-source atomic publication is asserted. Unknown-length
required model objects and resumable capped transfers remain separate work.

## Explicit protected-cache removal

Fresh empty roots use persistent policy v2. New code still opens existing v1
roots without changing their policy bytes, but refuses their reclamation. Old
CacheStore openers reject v2 before they can open unprotected managed readers. A
policy-less root with existing data cannot adopt v2. There is no automatic or live migration.

Every managed regular-file open acquires a shared Linux flock on that exact open
file description and verifies its bound inode. File cloning, dup and inherited
child handles preserve that pin until their final close; wrapper lifetime alone
is insufficient. No explicit unlock runs on wrapper drop. Lazy Artifact readers
constructed from managed descriptor clones retain the same pin. Advisory locks
cover cooperating cache users. Generic local-path run/inspect, arbitrary raw
File::open, mappings or deliberate descriptor unlocks do not join this protocol;
raw access to cache-owned paths requires external quiescence during removal.
Use run-cache or the managed SnapshotLease API for concurrent loading.

In protected roots, a stable shared activity lock spans each complete transaction
and each snapshot or reference binding window. Legacy roots do not gain this
extra charged control inode. Maintenance takes activity exclusively, without
waiting, before publication. Active transfers therefore return Busy, including
first-seen/deduplicated blobs missing from planned checksum reservations. It
never waits on object or attempt locks under publication. Recovered partial
payloads close before their old attempt owner, and transaction cleanup finishes
before activity is released. Other metadata inventory operations hold publication
until their temporary descriptors close.

Under those existing authorities, one bounded graph captures current refs and
conservative shared-blob reachability through other manifests. Referenced targets
refuse. Stable locks, reference-record backing, dead/retained staging and general
orphans are retained. Maintenance probes separate actual candidate descriptors
with nonblocking exclusive locks and validates names/inodes; escaped readers
produce Busy before any target mutation. All local collector payload/control/tree
handles close before the final unlink. The scope is one exact target, with no
new cache owner, collector daemon or reservation ledger.

A fresh retirement needs the existing conservative 256 KiB control allowance to
fit under the cooperative disk policy. Dry-run and apply both check that
headroom; a full root can refuse without mutation. Activity and publication
exclude competing admissions throughout this bounded operation.

Apply atomically moves the snapshot into retirement/DIGEST and syncs the affected
parents before destructive cleanup. The moved manifest and then an empty retired
directory preserve exact retry through partial cleanup. A pending retirement
blocks publication of that same snapshot. Every mutation and directory sync has
diagnostic fault points. Errors after mutation return Incomplete, never implied
rollback. An already absent exact target is an idempotent zero-removal result;
apply syncs its parents, without collecting unrelated orphan manifests or blobs.

ReclaimReport counts named entries, including hard links, and reports rebuilt
unique-inode conservative cache charge before and after. An unavailable post-error
inventory is unknown. This is neither a physical-block-release attestation nor a
filesystem free-space reservation. Concurrent process creation can briefly inherit
collector descriptors until exec closes them, and kernel close effects can lag;
local close is not a process-global last-close attestation. An incomplete removal
can retain charged names; no optimistic release credit follows from a wrapper or
namespace change.

The independent cache-only CLI requires an existing explicit absolute root and
one mode:

```text
zerogravity cache remove DIGEST --dry-run --cache /absolute/cache
zerogravity cache remove DIGEST --apply --cache /absolute/cache
```

Use the root's existing disk-policy limit with --disk-limit-bytes when it differs
from the CLI default. No aliases, force option, hidden reference deletion,
automatic eviction, dead-stage sweep, credentials, transport or signal handlers
are involved. Dry-run observes eligibility and is not a reservation; apply always
revalidates. Tests and acceptance remain in TODO.md. Implementation fixtures use
private tiny caches; no live user cache is deleted or migrated by this work.

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
GitHub release selections reuse the first validated release ID for the same
literal provider/repository/requested-tag tuple. Later components select assets
from that release ID through their own label-scoped transport, without resolving
the tag again. Separate file and release pin maps prevent equal selector strings
from crossing locator kinds. Ordinary single-asset resolution keeps its existing
behavior; no synthetic tag or public pin argument is introduced.

Every label still lists and validates assets through its own credentials and the
shared metadata/request budget. Listings retain path/name/ID, pagination and
count checks; the selected asset must be uploaded and satisfy digest validation
before supplying an asset-ID download URL. Each selected asset retains its original release-ID/asset-ID source
locator, exact or explicitly capped length and TrustedSha256/FirstSeen provenance.
A stable release ID does not freeze its mutable asset set: re-listing can observe
a renamed, missing or replaced asset, and identity/digest checks apply to the
actually selected object. This does not establish a cross-asset atomic snapshot.
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
The pure library `open_components` takes no transport. A shared canonical locator
parser requires positive decimal release and asset IDs without alternate spellings.
Repeated release bindings must agree on the release ID for the requested source
while permitting different asset IDs; repeated file bindings must still agree on
their exact commit. Offline verification cannot prove mutable selector freshness,
remote asset membership, token revocation, model compatibility or licensing.

## Explicit read-only HF cache import

The independent Linux `cache` feature exposes `cache::import_hf` and an explicit
local CLI. No HTTP, TLS, provider client, credential lookup or online signal
handler is involved:

```text
zerogravity cache import-hf \
  --source /absolute/hub/models--owner--repo --repository owner/repo \
  --commit 0123456789abcdef0123456789abcdef01234567 \
  --file weights:model.gguf --file config:config.json \
  --file tokenizer:tokenizer.json --dry-run --cache /absolute/destination
```

Use `--apply` for copying and optional `--sha256 PATH=DIGEST` for independently
authorized raw checksums. Each path and role is explicit; this is a data import,
not a model-closure resolver. The example names illustrate layout and select no
approved source. Ordinary run/inspect and cache misses do not import anything.
No environment-based cache discovery, branch/ref lookup or remote code runs.

Import requires a two-component repository name and a matching
`models--owner--repo` source basename, a40-digit lowercase commit and an existing
absolute destination directory with0700 permissions. An empty destination can be
initialized after validation; missing directories are never created. Renamed or
unnamespaced repository caches remain outside this slice. Source/destination
may use the original same-mount path or one narrowly qualified cross-filesystem
pair: Btrfs with tmpfs in either direction, or two distinct tmpfs filesystems.
Cross-filesystem roots must differ in both Linux mount ID and device ID, and
retained-FD fstatfs must report that permitted type pair. Each traversed node and
selected source descriptor must remain on its own root device and mount.
Across different mounts, same-device bind aliases, Btrfs-to-Btrfs pairs,
overlay/FUSE and other unqualified pairs refuse. Missing statx mount identity
also refuses. This allowlist is a bounded qualification policy, not a claim
that different device IDs alone
prove independent storage. The original same-mount behavior is unchanged.

Read-only source capabilities accept regular snapshot files or canonical HF leaf
links into flat40/64-digit lowercase blob names in the same repository root.
Intermediate/chained/absolute/escaping links, special selected files and incomplete
selected paths refuse. Positioned reads retain the original opened files rather
than reopening paths. Metadata checks bind device/inode, length, mtime and ctime
before/after copying and each read; a separate nonempty EOF sentinel proves the
exact bound. Git LFS pointer bodies refuse during streaming.

Preparation admits a bounded control envelope in the caller's existing physical
ResourceArbiter domain before opening source metadata. The plan retains that
lease and its source descriptors. Destination checking walks bounded source and
target directory metadata without reading refs contents, cached trees/*.json or
unselected file bodies. No symlink is followed during that topology walk.
Bidirectional ancestry, mount and inode checks reject overlapping trees and hardlink aliases, including
unselected source inodes and the destination root itself. This explicit
selected-repository safety inspection is not a cache-discovery scan. A successful dry-run is not a reservation or permission
for a later changed tree; apply repeats the checks.

Apply initializes the exact preflighted destination directory capability through
the original CacheStore initializer, then uses its existing full-peak admission,
private staging, hash/EOF validation, sole installer and snapshot publisher.
No source inode is locked, chmodded, linked, renamed or deleted by import. Source
bytes are copied into owned cache storage. A known-digest destination hit still
hashes and verifies the retained external source before commit; receipt
copied_bytes counts only bytes consumed by the importer, so such a hit reports
zero copied bytes. Normal copies reuse the original importer's SHA256 result.
No mutable reference is published, and typed DurabilityUnknown remains distinct
from a successful durable snapshot.

Manifests persist provider `local-hf-cache`, the caller-declared repository/commit
and actual exact lengths/hashes. They cannot satisfy online-HF cached-selection
checks as authenticated remote metadata. Content is FirstSeen unless an explicit
caller checksum verifies; cache filenames and commit-directory names supply no
raw checksum or membership authority. The existing run-cache bridge can consume
explicit resulting snapshots through its unchanged loader eligibility checks.

These are operation-local checks, not an atomic snapshot or a freeze against
arbitrary same-UID in-place writes, renames or mount changes. Blocked filesystem
operations do not gain real-time cancellation. The original root initializer
retains its existing lock-deadline behavior: cancellation after preflight may
leave initialized destination metadata before transaction admission refuses;
initialization metadata is not rolled back. Reading can update source atime;
tests preserve source names/content/modes/mtime. Ownership and mutation checks do
not establish global RAM/UMA accounting or filesystem free-space reservation.
A tmpfs destination remains volatile: the original publisher's successful fsync
acknowledgments do not promise survival across unmount, reboot or power loss.
No filesystem is mounted or privileges changed by import or by the selected
existing-filesystem fixtures. Real hidden bind/subvolume/overlay alias fixtures
remain separately unqualified; those ambiguous cross-filesystem pairs refuse.
Acceptance and remaining filesystem/layout work are tracked in TODO.md.

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
and retain independent StateRoots. The original-checkpoint gate also removes private cached snapshot/weight names
after source copying, then verifies both original models through the same exact
logit/state/session checks while their raw source reservation remains owned.
Numerical acceptance is tracked in TODO.md.

## References

- [Hugging Face Hub API](https://huggingface.co/docs/hub/api),
  [official OpenAPI](https://huggingface.co/.well-known/openapi.md) and
  [official file-cache layout](https://huggingface.co/docs/huggingface_hub/guides/manage-cache),
  used for the explicit local snapshot/blob import contract.
- [GitHub Git trees](https://docs.github.com/en/rest/git/trees),
  [release tag lookup](https://docs.github.com/en/rest/releases/releases#get-a-release-by-tag-name)
  and [release-ID asset listing/download](https://docs.github.com/en/rest/releases/assets),
  checked against the official endpoint contracts for this component extension.
- [signal-hook flags](https://docs.rs/signal-hook/0.3.18/signal_hook/flag/) and
  [registry cleanup](https://docs.rs/signal-hook-registry/1.4.8/signal_hook_registry/fn.unregister.html),
  checked against the exact already-cached dependency sources.
- [ureq 3.4.2](https://docs.rs/ureq/3.4.2/ureq/), with its exact cached source
  used to check timeout, TLS and redirect behavior.
- [HTTP/1.1 framing and chunked completion, RFC9112](https://www.rfc-editor.org/rfc/rfc9112.html#section-6.3),
  with the pinned ureq/ureq-proto decoder sources checked locally.
- [Linux tmpfs](https://docs.kernel.org/filesystems/tmpfs.html),
  [Btrfs subvolumes](https://btrfs.readthedocs.io/en/latest/Subvolumes.html),
  [overlay identities](https://docs.kernel.org/filesystems/overlayfs.html) and
  [FUSE passthrough](https://docs.kernel.org/filesystems/fuse/fuse-passthrough.html),
  used to bound the qualified filesystem-pair policy. The pair policy follows
  from these filesystem semantics plus actual local fixtures; it is not a general
  cross-filesystem identity guarantee stated by those documents.
  Retained-identity review also checked Linux v6.17
  [anonymous superblocks](https://raw.githubusercontent.com/torvalds/linux/v6.17/fs/super.c),
  [tmpfs implementation](https://raw.githubusercontent.com/torvalds/linux/v6.17/mm/shmem.c)
  and [Btrfs inode identities](https://raw.githubusercontent.com/torvalds/linux/v6.17/fs/btrfs/inode.c).
- [Linux flock lifetime and advisory semantics](https://man7.org/linux/man-pages/man2/flock.2.html).
- [A1 local cache contract](../../research/PULL_A1_LOCAL_XDG_CACHE_2026-10-16.md),
  [current workflow](../../../AGENTS.md) and [TODO rules](../TODO-SPEC.md).
