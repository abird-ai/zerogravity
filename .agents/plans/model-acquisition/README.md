Author: GPT-6

# Model acquisition

Explicitly acquire bounded, pinned source files into the existing XDG cache.
This is data acquisition, separate from loading a model or granting execution
support. Implementation and tests are tracked in [TODO.md](TODO.md); the
[full acquisition specification](../MODEL_ACQUISITION_IMPLEMENTATION_SPEC.md)
and [provider parity research](../../research/MODEL_PULL_XDG_CACHE_AND_SOURCE_PARITY_2026-10-16.md)
retain the broader scope.

## Design

One optional synchronous HTTPS owner serves the provider resolvers. Native Rustls TLS
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

Resolution uses a caller-owned transfer admission, pins a repository commit,
release asset, versioned object or explicit content contract, validates file
paths and selects only the applicable declared closure.
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

Legacy HF/GitHub offline selection keeps its original data-only initial selection:
file roles and complete model closure are not inferred for ordinary cached data.
It now checks every retained managed descriptor for LFS pointer payloads after
original full cache verification. Generic CacheStore integrity alone grants no
acquisition eligibility. Explicit component weights additionally reuse the same
strict split/index closure as online resolution; independent config/tokenizer
suppliers relax only those sidecars. Reading retained indexes shares one cumulative
metadata-byte and index-request budget across the complete component set. These
checks use existing descriptors and parsers, with no transport or second cache.

## Explicit HTTPS objects

`acquisition::object::HttpsObjectSpec` describes one caller-authorized object:
a strict bare origin, logical path, role, exact byte count (including zero) and
raw SHA256. It stores no download URL. Its admitted `prepare(url, ...)` performs
local validation and returns the original `ResolvedPullPlan`; the existing
executor owns transfer, resume, hashing, staging and publication.

```text
zerogravity pull object --origin https://objects.example \
  --file payload.gguf --role weights --bytes 123 --sha256 EXPECTED_SHA256 \
  --url-file /absolute/private-url.txt --dry-run --cache /absolute/cache
```

The example selects no approved source. Online defaults to apply; omit
`--dry-run` only for an authorized fetch. `--url-file -` explicitly reads stdin.
URL files use no-follow, nonblocking opens, must be regular files and fit16KiB;
one final LF/CRLF is accepted. Other controls, whitespace, userinfo and fragments
refuse. Input errors omit the contents. Explicit stdin can block, and intermediate
path components are not a directory-capability ownership proof. No automatic
credential-file or source discovery occurs.

The expected origin is supplied separately and normalized by the existing URL
parser. The actual URL must match it before client/credential construction.
Production requires HTTPS; diagnostic HTTP remains literal-loopback only. Signed
URL paths and queries stay in memory within the existing redacted plan; changing
a signed locator on the same origin does not change the content contract.
Optional `--token-env NAME` reads only that named environment variable during
online execution and scopes the bearer to the expected origin. Existing redirect
stripping, TLS, framing, deadline, cancellation and progress rules apply.

The manifest provider is `https-object`. Its repository identity hashes the
versioned canonical origin; its revision hashes the versioned complete tuple of
origin, logical path, role, exact length and caller raw digest. Receipts/manifests
and Debug do not contain object URL paths/queries or credential values. The raw
checksum is independently caller-authorized, not inferred from an ETag, Git ID
or URL; normal raw-byte verification records TrustedSha256. These source labels
identify the declared data contract, not remote repository membership or a model
SourceIdentity. This first object frontend publishes no mutable reference.

`--dry-run` validates the caller contract and URL locally, without credentials,
client, remote metadata, payload transfer or cache initialization. Printed sizes
are caller-supplied exact bounds, not observations of a remote object. Preparation
admits1MiB of bounded operation-local control work; it does not retain a transfer
or TLS/global-memory reservation after returning the plan.

Offline needs no URL input at all:

```text
zerogravity pull object --origin https://objects.example \
  --file payload.gguf --role weights --bytes 123 --sha256 EXPECTED_SHA256 \
  --offline --snapshot SNAPSHOT_DIGEST --cache /absolute/cache
```

The API's `open_cached` and CLI offline path precede URL-file/stdin inspection,
credential/client construction and transfer admission. They open original managed
descriptors, verify all bytes and the exact single-entry source/path/role/length/
SHA/TrustedSha256 contract under caller limits, and reuse LFS-pointer refusal.
An existing ordinary cache ref may be selected instead of a snapshot digest;
that lookup grants no online freshness, license acceptance or inference support.
No repository sidecars or model closure are inferred for this explicit object.

## Native OCI raw GGUF layers

`acquisition::oci::OciRequest` resolves one OCI/Docker schema2 image manifest
by explicit tag or raw manifest SHA256. It hashes received manifest bytes before
parsing, checks a supplied manifest digest and any Docker-Content-Digest header,
and requires matching supported response/JSON media types. A recognized raw
`application/vnd.docker.ai.gguf.v3` layer supplies its exact size and SHA256 to
the original transfer/cache executor. Unique selection refuses multiple matching
layers; an explicit layer digest selects exactly one recognized layer.

```text
zerogravity pull oci --origin https://registry.example --profile anonymous \
  --repository owner/model --tag selected --file model.gguf --dry-run \
  --cache /absolute/cache
```

The example authorizes no source. Use `--manifest-sha256 RAW_SHA256` instead of
`--tag` for a caller-pinned manifest, and `--layer-sha256 RAW_SHA256` when selecting
a particular layer. Dry-run performs admitted broker/manifest metadata requests
but no layer transfer or cache initialization. Online defaults to apply.

Profiles are explicit anonymous access, preissued bearer via `--token-env NAME`,
or `dockerhub` public anonymous pull. Docker Hub fixes the registry to
`https://registry-1.docker.io` and token broker to `https://auth.docker.io/token`
with service `registry.docker.io` and the selected repository's pull-only scope.
No server-provided realm, arbitrary login, Basic credential, refresh, SDK or
ambient account discovery runs. The shared HTTP owner retains origin-scoped
credentials, TLS verification, secret redaction, redirects and bounded requests.
The broker consumes at most64KiB and half the metadata cap; manifest metadata
uses the remaining cap. Docker Hub requires a two-call logical metadata budget.

Provider `oci` records the raw manifest digest as revision. Its repository label
hashes the full versioned request contract: canonical origin, profile, repository,
requested tag-or-digest, logical file and layer selector. Secret values and URLs
are absent from the persisted identity. The original executor verifies actual
layer bytes against the descriptor's raw digest, including cache hits, framing,
EOF, LFS-pointer refusal and existing cancellation/rollback rules. No mutable
reference is published by this frontend.

Offline requires `--offline --snapshot DIGEST_OR_EXISTING_REF`, together with
the original request flags. It opens and hashes original managed cached bytes,
checks exact one-entry source/path/Weights/trusted-digest constraints and caller
limits, and branches before credentials, broker, client, transfer admission or
signal construction. The cache does not retain original raw manifest JSON:
offline contract matching cannot independently reauthenticate descriptor
membership, tag freshness, registry signatures or current remote access.

Indexes/platform selection, compressed/container layers, arbitrary descriptor
URLs, config extraction, generic authentication challenges and private Docker
login remain unsupported. Acquiring a GGUF layer supplies no model sidecars,
model closure, license acceptance or new inference eligibility. This is a bounded
native registry-data path, not full OCI conformance or container execution.

## Native GCS versioned objects

The native JSON API resolver accepts explicit objects or bounded auxiliary
prefixes at `https://storage.googleapis.com`. `GcsObjectRequest` resolves metadata
once; `GcsObjectPin` records generation, metageneration and exact size. Public
`resolve_with_pin`/`resolve_with_pins` expose those observed pins alongside the
original plans. Ordinary `resolve` still returns original plans alone.

```text
zerogravity pull gcs object --bucket selected-bucket --object selected/model.gguf \
  --file model.gguf --role weights --generation GENERATION --sha256 RAW_SHA256 \
  --dry-run --cache /absolute/cache
zerogravity pull gcs prefix --bucket selected-bucket --prefix selected/ \
  --dry-run --cache /absolute/cache
```

These examples authorize no bucket or data. Object roles are explicit; caller
SHA256 is optional and independently authorized. Without it the final original
raw hash remains FirstSeen: GCS MD5, CRC32C and ETag are not raw SHA256 authority.
Omitting generation resolves latest metadata once to a concrete generation.
Dry-run reads only admitted bounded metadata; online defaults to apply.

Media requests derive locally from the validated bucket/name and carry both
`generation` and `ifMetagenerationMatch`. Returned mediaLink values are ignored.
Pinning metadata version also protects stored representation: nonidentity
contentEncoding refuses before transfer, including a later representation change
via the metageneration precondition. Customer-key and soft-deleted profiles
remain unsupported. The original HTTP owner refuses all media redirects so a
Location cannot discard these query preconditions. Existing custom transports
must explicitly implement that direct-response capability or fail closed.

One private typed binding in the original executor checks contradicting PRESENT
x-goog-generation/metageneration headers before initial, resumed and live-retry
bodies. JSON API does not require those response headers; absence is allowed,
with query preconditions supplying the protocol binding. Original duplicate-
header, exact length/hash/EOF, strong-validator resume and publication rules
remain independent checks. There is no additional transfer engine.

Prefixes must be nonempty and slash terminated. The resolver bounds all page
bytes, calls, tokens, file count, control metadata and aggregate selected bytes;
cycles, duplicates, unsafe paths and file/directory collisions refuse. Every
selected object yields an independent Auxiliary/FirstSeen plan. Listing and
publication are not atomic: a later failure may leave earlier verified snapshots.
No config/tokenizer/index/model closure is inferred.

Anonymous and explicitly named bearer-environment profiles reuse the shared
HTTP owner. There is no ADC, metadata-server discovery, service-account minting,
SDK, refresh or requester-pays fallback. Custom production GCS endpoints remain
unsupported; diagnostic literal-loopback fixtures are separate.

Offline object selection requires the original bucket/name/logical path/role,
exact `--generation`, `--metageneration`, `--bytes` and snapshot digest, plus the
same caller SHA if used online. It verifies original managed bytes and exact
source/selection/trust under caller limits before credentials/client/admission/
signals. Receipts print the observed pin for later reuse. Prefix offline selection
refuses; choose each explicit pinned object instead. Offline byte/source matching
attests neither present remote access nor bucket membership/freshness.

## Native versioned S3 objects

The S3 resolver uses explicit regional AWS commercial endpoints or a caller's
bare compatible endpoint, region and path/virtual-host addressing. It resolves
HEAD once to an exact non-null object version and size; selected versions then
use the original transfer/cache executor. Typed pins expose actual version and
bytes for receipts/offline reuse. Null/unversioned and special bucket/partition
profiles remain outside this slice.

```text
zerogravity pull s3 object --endpoint-profile aws --region us-east-1 \
  --addressing path --bucket selected-bucket --key selected/model.gguf \
  --file model.gguf --role weights --version-id SELECTED_VERSION \
  --sha256 RAW_SHA256 --dry-run --cache /absolute/cache
```

The example authorizes no source. `--endpoint-profile compatible` requires an
explicit `--endpoint`; AWS profile rejects overrides. Virtual-host credentials
bind to the bucket-specific origin, and dotted virtual-host buckets refuse under
this TLS profile. Online defaults to apply; dry-run reads admitted metadata only.

Anonymous and explicit SigV4 profiles share the original HTTPS owner. SigV4 takes
only named `--access-key-env`, `--secret-key-env` and optional
`--session-token-env` inputs. It signs the actual method/Host/Range/If-Range and
amz headers on each attempt with checked UTC. No SDK, ambient AWS config/default
credential chain, identity service, region discovery/redirect or refresh runs.
Credentials stay in that origin's client; they never enter source labels or plans.

The opaque `HttpTarget` preserves the exact encoded request path/query through
that same HTTP owner. Its constructor is private to validated provider code, and
custom transports must explicitly support its default-refusing capability. S3
keys retain dot segments and repeated slashes; UTF8, percent and query values are
encoded once, with sorted canonical query pairs. Redirects refuse. This avoids
ordinary URL normalization silently selecting a different key or invalidating a
signature; no second socket or executor is introduced.

The original executor requires matching x-amz-version-id and refuses delete
markers before accepting body bytes at each initial/resume/retry response.
Strong ETag, exact range/framing/length/full raw SHA/terminal EOF and cache
publication retain their existing separate checks. Caller raw SHA gives
TrustedSha256; otherwise final content remains FirstSeen. ETag, MD5, CRC and
multipart/composite checksum labels never become raw SHA256 authority.

`pull s3 prefix` selects Auxiliary/FirstSeen objects independently. `--listing
current` uses ListObjectsV2 continuation tokens; `latest-versions` uses paired
key/version markers and selects latest live entries. Complete bounded XML and
HEAD validation precede returned plans, with aggregate request/metadata/control/
file/byte budgets, cycle/duplicate/path/collision refusal, and no DTD/entity
expansion. A prefix is not atomic, and later failure may retain earlier published
objects. Neither listing infers model closure.

Offline object mode needs the same endpoint/region/addressing/bucket/key/path/
role contract, `--version-id`, `--bytes`, snapshot digest and any original caller
SHA. It verifies original managed bytes/source/trust before credentials, client,
admission or signals. Prefix offline refuses; choose explicit object pins. Cached
bytes do not establish present listing membership, access or remote freshness.
Broader vendor features, presigning, archive restoration, customer keys, delegated
credential acquisition and FULL_OBJECT checksum-mode integration remain open.

## Native Azure BlockBlob revisions

`pull azure object` selects an explicit opaque version ID or snapshot on the
native public-cloud account endpoint. HEAD supplies exact size and a transient
strong ETag; GET and every retry/resume preserve the same revision query through
the existing opaque target and HTTP owner. Responses require a complete,
identity-encoded BlockBlob; structured bodies, customer-key representations and
incomplete copies refuse. A present version echo must agree, but the official
protocol does not require that echo. Snapshots and version IDs remain separate
namespaces.

```text
zerogravity pull azure object --origin https://ACCOUNT.blob.core.windows.net \
  --container selected-container --blob selected/model.gguf \
  --file model.gguf --role weights --version-id SELECTED_VERSION \
  --sha256 RAW_SHA256 --dry-run --cache /absolute/cache
```

The example authorizes no source. Caller SHA256 gives TrustedSha256; otherwise
original final hashing remains FirstSeen. ETag, MD5 and CRC64 are separate from
raw SHA authority. Source identity binds the exact origin/container/blob,
revision kind/value, logical path/role, observed bytes and optional caller SHA;
credentials and the transient ETag do not become offline source inputs.

Anonymous, preissued bearer, account SharedKey and a bounded ad-hoc serviceSAS
are explicit mutually exclusive profiles. SharedKey signs each actual request
with checked UTC, fixed REST version2023-11-03 and independently tested
HMAC-SHA256 canonicalization. Account keys come only from a named environment
variable. SAS comes from a bounded8KiB nofollow regular file or explicit stdin;
its escaped query stays private inside the same HTTP owner and is preserved
verbatim. The supported serviceSAS profile requires HTTPS/read access, an
appropriate version/snapshot/container resource, and sv2019-12-12 through
2023-11-03. The owner appends unsigned api-version=2023-11-03 to pin REST behavior.
Stored policies, account/delegation SAS, response overrides, ambient credential
chains, minting, SDK login and refresh remain outside this profile. Pinned
requests never redirect.

`pull azure prefix` uses bounded current-version XML listing and returns
independent Auxiliary/FirstSeen plans. It checks the entire selected listing,
continuations, version/type/copy properties, paths, aggregate controls and byte
budgets before returning. It grants no atomic container snapshot or model
closure; earlier complete publications may survive later failure.

Offline object mode requires the same source contract, revision kind/value,
exact bytes (including zero), cached snapshot digest and original caller SHA if
used. It verifies the original managed bytes and provenance before any secret
file/stdin/environment read, client, admission or signal setup. Prefix offline
selection and mutable refs refuse. Cached bytes attest neither current access nor
remote freshness. Implicit latest/no-SHA current objects, other endpoints/blob types and
broader credential profiles remain open work.

## Explicit unversioned cloud content contracts

S3 and Azure object commands accept an explicit `--content-contract --bytes N
--sha256 RAW_SHA256` profile. It supports current/unversioned locators without
inventing a remote version. Exact bytes (including zero) and the caller's raw
SHA256 are mandatory. Version/snapshot selectors, prefix mode, mutable refs and
unknown-size caps cannot be combined with this profile. A missing remote version
never silently selects it.

```text
zerogravity pull azure object --origin https://ACCOUNT.blob.core.windows.net \
  --container selected-container --blob selected/model.gguf \
  --file model.gguf --role weights --content-contract \
  --bytes EXACT_BYTES --sha256 RAW_SHA256 --dry-run
```

The example authorizes no source. S3 uses the same flag set with its existing
endpoint/region/addressing/bucket/key selectors. Preparation and dry-run are local
contract validation under bounded control admission: no HEAD, metadata, secret
input, client or cache publication. Apply uses the original exact transfer,
strong-validator resume, full raw SHA/EOF and sole publisher. A current locator
may change, but only bytes matching the complete authorized content contract can
install. The source providers s3-content and azure-content bind normalized source
selectors, logical path/role, exact size and SHA; they attest no remote version,
present membership or freshness.

Typed targets preserve S3 raw paths and cloud credentials through the same HTTP
owner; redirects refuse. S3 delete markers refuse. Azure retains identity-encoded
BlockBlob/copy/key/structured-body checks. Azure content SAS accepts read b or c
resources; version/snapshot requests retain their distinct bv/bs/c rules. Returned
version headers do not replace the caller's SHA. Offline requires the original
snapshot digest and identical content contract before credentials/client/signals
or transfer admission, using the shared managed-file verifier. HTTPS retains its
existing ref selection. Strong-conditional observations without caller SHA and
per-object digest-map prefixes remain separate work.

## Strong conditional cloud observations

The explicit object-only `--conditional-read` mode supports S3 and Azure current
objects when the caller supplies no raw SHA. The existing arbiter admits metadata
work before a body-free HEAD to the exact native target. An exact200 response must
provide complete identity-representation length and one strong ETag within the
original limits and provider representation checks. Missing/weak validators,
redirects, unsupported status/framing and exhausted bounds refuse.

A separate `s3-conditional` or `azure-conditional` source contract binds the full
normalized locator/profile, logical path/role, exact bytes and ETag. This is a
conditional observation. Actual raw SHA remains FirstSeen; neither ETag nor the
current locator supplies a trusted raw digest or immutable remote version.
Changing the declared observation changes its contract and retained-prefix key.

The original opaque HTTP target carries a private typed If-Match condition. Every
initial GET, range and retry sends it, with exact response ETag validation before
body reads. A differing resume validator refuses before wire/signing. SigV4 signs
the same If-Match bytes; Azure SharedKey uses the standard If-Match slot. Existing
anonymous, bearer and serviceSAS profiles retain their original ownership. HTTP412
refuses without refreshing metadata, downgrading the condition or choosing another
profile. The original exact length/hash/EOF/LFS/cancellation and sole cache
publisher remain authoritative.

Online dry-run resolves HEAD metadata only. Offline takes the same explicit
conditional request plus exact bytes, quoted ETag and snapshot digest; it checks
source/path/role/size/FirstSeen provenance, original full managed bytes and current
limits before credentials, file/FIFO/stdin, client, admission or signals. The pure
library pin constructor represents a caller-declared observation; offline access
asserts no present remote freshness or authorization. Receipts escape the ETag as
a JSON string. There is no conditional prefix/digest-map or implicit fallback;
existing remote-version and caller-SHA contracts keep their original semantics.

## Native ModelScope explicit commits

`pull modelscope OWNER/REPO --commit FULL40HEX --file PATH` uses the native
modelscope.cn API directly, without a Python SDK or persisted login. Repeat
`--file` for bounded selectors and use `--subfolder` for a repository directory.
Only a canonical lowercase40hex commit is accepted; aliases/default revisions,
mirror endpoints and missing exact file sizes remain separate work. The pinned
official client passes an explicit revision through its listing and raw-file
requests. This profile trusts that endpoint contract; it does not independently
verify a Git/Merkle tree or prove live server behavior from synthetic tests.

Resolution walks bounded shallow directories with Revision, Recursive=False and
relative Root. The official legacy API silently caps a directory at3000 rows
without pagination or completeness authority. A directory reaching that boundary
refuses, even when the requested file is already visible. One cumulative request
and byte budget covers the complete inventory and required index reads; caller
file/control limits also bound the inventory. No partial walk becomes model
closure.

The original HF initial selector and extracted private GitHub closure logic
supply required config/tokenizer, indexed shards and split-GGUF checks, including
ambiguous-model refusal. GitHub retains its original metadata/size/raw-URL/LFS
owner around that pure callback. A canonical source-provided raw SHA gives
TrustedSha256; absent SHA stays FirstSeen. Git object IDs are not raw hashes.

Anonymous and explicit `--profile official-token --token-env NAME` are the only
profiles. The official-token client sends Bearer and m_session_id together within
one exact origin, with cookie-safe token validation and no jar/login/refresh.
Private Direct targets preserve the selected commit and refuse all redirects
for metadata and payloads; redirecting/CDN profiles remain unqualified. S3/Azure
credentials cannot authorize Direct, and official-token credentials cannot
authorize their targets. The sole HTTP/executor/cache/publication owner remains.

Offline requires the same repository/commit/selectors and a snapshot digest. It
verifies all original managed bytes, source, role/provenance, exact selection,
current bounds and pointer policy before credentials/client/admission/signals.
The source label includes normalized origin identity so diagnostic fixtures cannot
alias native data. Offline data does not prove present access, remote membership,
license or inference support. No mutable ref or capped required-file mode is
introduced.

## Xet hash and chunk primitives

The native reader shares exact hash/codec sources with the standalone primitive
test wrapper. The implementation follows protocol1.1.0 and its pinned xet-core
reference, with separate raw32-byte and limb-reversed textual
identifiers, keyed BLAKE3 chunks, in-place ordered Merkle aggregation, zero-key
file finalization and the special empty root. Raw SHA256 remains a separate
content/provenance authority. Download metadata supplies no expected per-term
verification hash; computing a shard/upload verification primitive does not
authenticate an absent field or a partial xorb root.

The safe decoder borrows caller buffers and allocates nothing. It bounds encoded
chunks to256KiB and decoded chunks to128KiB, validates the8-byte header and exact
consumption, and supports NONE, framed LZ4 and BG4/LZ4 including every tail
remainder. Standard frame/block checksums, linked versus independent history,
end markers and terminal EOF are checked. Unsupported dictionary, legacy,
skippable and concatenated frames refuse. Bounded cancellation checks cover
copying, checksumming, decoding and hashing. Literal upstream Merkle vectors and
independent hash/frame goldens exercise exact namespaces and malformed inputs.

Only two direct acquisition dependency edges, cached BLAKE3(std/pure) and
xxhash32, are added; CPU/cache-only transport isolation remains. The explicit
native route below uses these same primitives. Primitive tests alone grant no
native-transfer throughput or model capability.

## Native HF Xet

`pull xet OWNER/REPO --revision REV --file PATH` explicitly selects native
reconstruction through the existing optional acquisition owner. Original HF
pinning, complete bounded inventory, selectors, model/shard/sidecar closure and
source identity stay authoritative. At least one selected LFS object must announce
an Xet file identity. Discovery observes headers without following its redirect or
reading a compatibility payload. Declared ordinary Git sidecars retain the original
HTTP route, redirect credential stripping and strong-validator suffix resume.
Announced native files have no silent compatibility fallback on protocol failure.

A repository/commit-scoped read grant is obtained from a locally constructed Hub
route. The grant is ephemeral and restricted to the fixed CAS origin; arbitrary
refresh links do not supply authority. CAS uses only that grant, and signed range
requests suppress all Hub/cloud credentials even at the same origin. The original
HTTP owner preserves TLS, raw signed URI bytes, exact complete signed interval
sets, no-follow policy for native metadata/payloads, redaction and request deadlines.
Custom transports must explicitly implement the opaque target/accounting contract
or refuse; no secret getter or arbitrary-header interface is added.

Native reconstruction tries v2, with v1 only after404/501. Scoped401 and signed403
can refresh within one finite combined reissue budget. The file identity and
ordered semantic term contract must stay fixed. Parsing validates full-file offset,
exact aggregate lengths, unique maps, bounded terms/groups/ranges/chunks and full
coverage. Terms may repeat or overlap; only their selected logical chunks enter
the final file root, in file order. Fetched edge chunks are decoded and charged,
but do not enter that root or logical progress. Missing sibling data cannot prove
a full/partial-xorb root, and the protocol supplies no per-term verification hash.

The reader holds one decoded chunk and admitted codec/hash scratch. It requests
the complete signed range set, validates every multipart boundary, interval,
length, representation and terminal EOF, and handles parts in arbitrary order.
It can reissue a full set for later selected spans; fetched/decoded repeats remain
charged. Interrupted refetch compares already accepted leaves without duplicating
logical progress. This bounded reader has no throughput or fetch-coalescing claim;
measured coalescing remains future work.

One invocation bounds actual HTTP attempts, observed body/control bytes, decoded
chunks, logical leaves and refreshes. Accepted term/group/range counts accumulate
across files and metadata reissues, while each document is also bounded. At most
one observed overflow sentinel is charged; after overflow, the invocation refuses
further reads or attempts. Exact-budget zero-byte EOF probes remain possible.
Per-request deadlines plus finite work do not create a whole-operation elapsed
bound or kernel/TLS traffic accounting.

The original arbiter admits scratch before allocation and retains it alongside
the original cache transaction. Native CLI defaults to256MiB to cover both full
admission envelopes; ordinary pulls remain128MiB and explicit limits are unchanged.
The same cache locks, private staging, raw SHA/length/LFS/EOF checks and sole
publisher apply. Native objects use an exact ephemeral stream: HTTP prefixes are
neither recovered nor checkpointed, and process exit requires a new full native
transfer. The final Xet file root precedes successful source EOF and publication.
Cache hits remain verified cached data, with no assertion of native execution.

Dry-run performs Hub metadata/discovery only, with no CAS grant or signed body.
Offline opens an existing cache, checks original source/selection/current bounds
and pointer policy before credentials, client, admission or signals. It proves
cached data, not native protocol execution, freshness or present access. Native
LFS/index reconstruction during closure and measured fetch coalescing remain open;
acquired bytes grant no new loader, inference, license or model qualification.

## Provider and compatibility frontier

The retired research captures once counted57 rows; that historical count is not
an extant exhaustive ledger. The current finite frontier below reflects inspected
native entrypoints and the pinned reference classes. A disposition is not a
passing provider gate. Open implementation and acceptance work stays in TODO.md.

| Surface | Current qualified scope and remaining work |
| --- | --- |
| HF repository/files/revisions | Native exact commit pinning, subfolder, bounded `*`/`?`, required index/shard and config/tokenizer closure. Excludes/full fnmatch, quant shorthand/candidates, broader required sidecar policy and endpoint/env compatibility remain open. |
| GitHub files/releases/components | Native commit/tree and release/asset pins, exact sizing, independent pinned components and alias pin reuse. Qualified LFS-pointer resolution is separate; pointers refuse. |
| Direct HTTPS objects | Explicit origin/role/path/exact size/caller SHA; original transfer, offline verification and cache publisher. Broader missing-size/digest or repository discovery is outside this slice. |
| Local HF caches | Explicit pinned selections and independent verified copies, original same-mount policy plus qualified Btrfs/tmpfs and distinct tmpfs pairs. Renamed/unnamespaced roots and other filesystems remain open. |
| XDG/offline/removal | Native XDG/explicit root, zero-network offline, inspect/list/verify and protected exact-snapshot removal. Compatibility aliases, legacy migration, automatic eviction/general orphan cleanup and unmanaged readers remain separate open work. |
| Auth/endpoints | Origin-scoped explicit bearer environment inputs, safe redirects and TLS. Mirror/profile policy, broader compatibility env aliases and provider-specific credentials remain open; offline bytes do not attest current remote access. |
| OCI/Docker | Native tag-or-manifest pin/raw-GGUF layer, public/preissued bearer and fixed Docker Hub anonymous broker. Index/compressed/container/signature/generic private-login profiles remain open; no model closure follows. |
| GCS | Native JSON object generation/metageneration media pins, auxiliary prefix/pageToken listing, anonymous/preissued bearer and explicit offline object pins. ADC/minting/requester-pays/custom endpoints and atomic/closure semantics remain outside this profile. |
| S3/compatible endpoints | Native non-null versions, exact-path regional/path/virtual-host GET/HEAD/ranges, bounded V2/latest-version prefixes and static/session SigV4. Explicit current/unversioned objects additionally support caller SHA/size contracts; Strong conditional FirstSeen observations are explicit; broader partitions/vendor/credential profiles and FULL_OBJECT checksum mode remain open; presigned HTTPS alone grants no native S3 parity. |
| Azure Blob | Native explicit BlockBlob versions/snapshots, bounded current-version auxiliary listing, pinned ranges and anonymous/bearer/SharedKey/ad-hoc serviceSAS. Explicit current/unversioned objects additionally support caller SHA/size contracts; Strong conditional FirstSeen observations are explicit; latest convenience, other endpoints/types/SAS and managed/default credentials remain open. |
| ModelScope | Native explicit40hex commit, bounded complete shallow walk, exact files/model closure, anonymous/official-token and digest-only offline. Aliases, redirect/CDN/mirror profiles and missing sizes remain open; no SDK/login runtime. |
| HF Xet | Explicit bounded native discovery, scoped CAS, signed ranges/multipart, ordered NONE/LZ4/BG4 reconstruction and original ephemeral importer/CLI. Native index closure and measured coalescing/performance remain open; cache hits/compatibility HTTPS grant no native execution proof. |
| Auxiliary/preset convenience | Explicit auxiliary/component bytes are supported data. Specialized mmproj/draft/MTP/DFlash/Eagle3/DSpark discovery and bounded presets remain open convenience work; a preset cannot authorize new sources or limits. |
| Transport/streaming | Exact strong-validator resume, bounded retry/deadline/cancel/progress and explicit capped Auxiliary streams are qualified. Required unknown lengths, capped resume, independent full HTTP trailer validation and broader concurrency/performance remain open. |
| Loader/runtime delegates | Safetensors/PT/serialized tensors, RunAI/ModelExpress internals, IPC/distributed/lazy loading, quant reconstruction and speculative/multimodal execution belong to the execution/streaming plans. Acquiring opaque bytes grants none of those capabilities. |

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

- [Xet protocol1.1.0](https://github.com/huggingface/hub-docs/blob/39aa2452668744552cc6690c75cd239771148e58/docs/xet/index.md),
  [pinned hash/codec reference](https://github.com/huggingface/xet-core/tree/c3c726bed5cf54ded92a63fab892cfb7857c751a/xet_core_structures/src), and
  [LZ4 v1.10.0 format](https://github.com/lz4/lz4/blob/ebb370ca83af193212df4dcbadcc5d87bc0de2f0/doc/lz4_Frame_format.md),
  used for exact hash namespaces, aggregation, framing and BG4 behavior.

- [Official ModelScope client0.4.6 source](https://github.com/modelscope/modelscope_hub/blob/65255a78b451c3785f2dfbfaff97172552fea3af/src/modelscope_hub/_legacy_api.py),
  [explicit revision flow](https://github.com/modelscope/modelscope_hub/blob/65255a78b451c3785f2dfbfaff97172552fea3af/src/modelscope_hub/_download.py), and
  [silent directory truncation tests](https://github.com/modelscope/modelscope_hub/blob/65255a78b451c3785f2dfbfaff97172552fea3af/tests/test_repo_files_truncation.py),
  pinned to65255a78b451c3785f2dfbfaff97172552fea3af for the native first profile.

- [Azure GET Blob](https://learn.microsoft.com/en-us/rest/api/storageservices/get-blob),
  [properties](https://learn.microsoft.com/en-us/rest/api/storageservices/get-blob-properties),
  [listing](https://learn.microsoft.com/en-us/rest/api/storageservices/list-blobs),
  [SharedKey](https://learn.microsoft.com/en-us/rest/api/storageservices/authorize-with-shared-key),
  [serviceSAS](https://learn.microsoft.com/en-us/rest/api/storageservices/create-service-sas) and
  [REST versioning](https://learn.microsoft.com/en-us/rest/api/storageservices/versioning-for-the-azure-storage-services),
  used to bound revision echoes, canonical authentication and SAS protocol version.

- [S3 SigV4 canonicalization and official signature vectors](https://docs.aws.amazon.com/AmazonS3/latest/developerguide/sig-v4-header-based-auth.html),
  [HeadObject](https://docs.aws.amazon.com/AmazonS3/latest/API/API_HeadObject.html) and
  [version listing](https://docs.aws.amazon.com/AmazonS3/latest/API/API_ListObjectVersions.html),
  used to bound the explicit native regional/versioned profile.

- [GCS JSON objects.get](https://docs.cloud.google.com/storage/docs/json_api/v1/objects/get),
  [objects.list](https://docs.cloud.google.com/storage/docs/json_api/v1/objects/list) and
  [decompressive transcoding](https://docs.cloud.google.com/storage/docs/transcoding),
  used for native generation/metageneration and representation preconditions.

- [OCI Distribution v1.1.1 manifest/blob protocol](https://github.com/opencontainers/distribution-spec/blob/v1.1.1/spec.md) and
  [Docker registry authentication](https://docs.docker.com/reference/api/registry/auth/),
  used to bound the explicit registry and fixed anonymous-broker profiles.

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
